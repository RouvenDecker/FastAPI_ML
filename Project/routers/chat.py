from starlette import status
from fastapi import APIRouter, Depends, HTTPException

from sqlalchemy.orm import Session
from sqlalchemy import and_
import datetime

from Project.database import SessionLocal
from Project.models import User, ChatMessage, ChatSession, RAGSession
from Project.schemas import SentenceRequest, SentencesRequest, WikiRequest
from typing import Annotated
from langchain_openai import ChatOpenAI
from langchain_core.runnables import RunnableParallel, RunnablePassthrough
from langchain_chroma.vectorstores import Chroma
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import HumanMessagePromptTemplate, ChatPromptTemplate, PromptTemplate
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_community.utilities import WikipediaAPIWrapper
from langchain_community.tools import WikipediaQueryRun
from ..preprocessing import clean_sentences, simple_sentiment_pipeline  # type: ignore
from .auth import get_current_user


router = APIRouter(prefix="/chat", tags=["chat"])


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


db_dependency = Annotated[Session, Depends(get_db)]
user_dependency = Annotated[dict, Depends(get_current_user)]


@router.post("/rag", status_code=status.HTTP_200_OK)
async def rag_dialog(db: db_dependency, user: user_dependency, request: SentenceRequest):
    if user is None:
        raise HTTPException(status_code=401, detail="Authentication failed")
    # check is a ragsession exists
    rag_session = db.query(RAGSession).filter(RAGSession.user_id == user.get("id")).first()
    if not rag_session:
        raise HTTPException(status_code=400, detail="no files for rag session uploaded, upload files")

    # if exists load the corresponding chroma database
    embeddings = OpenAIEmbeddings()
    vectorstore = Chroma(persist_directory=rag_session.chroma_path, embedding_function=embeddings)  # type: ignore
    # create chain
    retriever = vectorstore.as_retriever(search_type="mmr", search_kwargs={"k": 3, "lambda_mult": 0.7})

    template = """
    Answer the following question:
    {question}

    To answer the question, only use the following context:
    {context}
    """
    prompt_template = PromptTemplate.from_template(template)
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.7)
    chain = (
        RunnableParallel({"context": retriever, "question": RunnablePassthrough()})
        | prompt_template
        | llm
        | StrOutputParser()
    )
    # return ai output
    response = chain.invoke(request.sentence)
    return {"response": response}


@router.post("/dialog", status_code=status.HTTP_200_OK)
async def chat_dialog(db: db_dependency, user: user_dependency, request: SentenceRequest):
    if user is None:
        raise HTTPException(status_code=401, detail="Authentication failed")
    current_user = db.query(User).filter(User.id == user.get("id")).first()
    if current_user is None:
        raise HTTPException(status_code=404, detail="User not found")

    # read user sessions
    user_sessions = db.query(ChatSession).filter(ChatSession.user_id == user.get("id")).all()
    active_session: ChatSession = None

    # create first session
    if len(user_sessions) == 0:
        first_session = ChatSession(
            session_id=0, user_id=user.get("id"), last_change=datetime.datetime.now(datetime.timezone.utc)
        )
        db.add(first_session)
        db.commit()
        active_session = first_session
    else:
        # load last session
        last_session = (
            db.query(ChatSession)
            .filter(ChatSession.user_id == user.get("id"))
            .order_by(ChatSession.last_change.desc())
            .first()
        )

        # check if session is still active
        if still_active_session(last_session, datetime.timedelta(minutes=5)):
            active_session = last_session
        else:
            active_session = ChatSession(
                session_id=last_session.session_id + 1,
                user_id=user.get("id"),
                last_change=datetime.datetime.now(tz=datetime.timezone.utc),
            )
        db.add(active_session)
        db.commit()

    # load all messages of active session
    db_messages = db.query(ChatMessage).filter(ChatMessage.session_id == active_session.session_id).all()

    # if no messsage exists, create first system message
    if len(db_messages) == 0:
        db_messages.append(
            ChatMessage(
                message_id=0,
                session_id=active_session.session_id,
                role="system",
                content="du bist ein hilfreicher assistent der Fragen beantwortet",
                date=datetime.datetime.now(tz=datetime.timezone.utc),
            )
        )

    # building the prompts
    chat = ChatOpenAI(model="gpt-4", temperature=0.7, max_completion_tokens=100)
    system_m = SystemMessage(db_messages[0].content)  # type: ignore

    history = ChatMessageHistory()
    build_message_history(history, db_messages)

    message_h = HumanMessage(request.sentence)
    chat_template = ChatPromptTemplate.from_messages([system_m, *history.messages, message_h])

    chain = chat_template | chat
    response = chain.invoke({})

    # append new user and ai message to db_messages
    db_messages.append(
        ChatMessage(
            message_id=db_messages[-1].message_id + 1,
            session_id=active_session.session_id,
            role="user",
            content=request.sentence,
            date=datetime.datetime.now(tz=datetime.timezone.utc),
        )
    )
    db_messages.append(
        ChatMessage(
            message_id=db_messages[-1].message_id + 2,
            session_id=active_session.session_id,
            role="ai",
            content=response.content,
            date=datetime.datetime.now(tz=datetime.timezone.utc),
        )
    )
    db.add_all(db_messages)
    db.commit()
    return response.content


def build_message_history(history: ChatMessageHistory, messages: list[ChatMessage]) -> ChatMessageHistory:
    for m in messages:
        if m.role == "ai":
            history.add_ai_message(m.content)  # type: ignore
        if m.role == "user":
            history.add_user_message(m.content)  # type: ignore
    return history


def still_active_session(chat_session: ChatSession, timedelta: datetime.timedelta) -> bool:
    last_activity: datetime.datetime = chat_session.last_change  # type: ignore
    # sqlite doesnt save datetimes with timezone info
    # the code below attaches the tz info to the datettime after loading it from the db
    last_activity = last_activity.replace(tzinfo=datetime.timezone.utc)
    now = datetime.datetime.now(tz=datetime.timezone.utc)
    return now - last_activity < timedelta


@router.get("/history", status_code=status.HTTP_200_OK)
async def get_chat_history(user: user_dependency, db: db_dependency):
    if user is None:
        raise HTTPException(status_code=401, detail="Authentication failed")
    chat_session = db.query(ChatSession).filter(ChatSession.user_id == user.get("id")).first()
    chat_messages = (
        db.query(ChatMessage)
        .filter(and_(ChatMessage.chat_session == chat_session, chat_session.user_id == user.get("id")))  # type: ignore
        .all()
    )
    return [(f"{m.date} - {m.role} :", m.content) for m in chat_messages]


@router.post("/sentiment/sentence", status_code=status.HTTP_201_CREATED)
async def predict_sentence(request: SentenceRequest):
    s = clean_sentences([request.sentence])
    return simple_sentiment_pipeline(s)


@router.post("/sentiment/sentences", status_code=status.HTTP_201_CREATED)
async def predict_sentences(request: SentencesRequest):
    s = clean_sentences(request.sentences)
    return simple_sentiment_pipeline(s)


@router.post("/response/cynical", status_code=status.HTTP_200_OK, description="chat with a cynical Chatbot")
async def respond_cynical(request: SentenceRequest):
    cynical_sys = "Du bist ein Assistent der Fragen auf zynische weise beantwortet"
    sys_msg = SystemMessage(cynical_sys)
    human_msg = HumanMessagePromptTemplate.from_template("{question}")
    prompt_template = ChatPromptTemplate.from_messages([sys_msg, human_msg])
    chat_value = prompt_template.invoke({"question": request.sentence})

    chat = ChatOpenAI(model_name="gpt-4o-mini", temperature=request.temperature, max_tokens=request.max_tokens)  # type: ignore

    return chat.invoke(chat_value).content


@router.post(
    "/wikipedia/search",
    status_code=status.HTTP_200_OK,
    description="ask a question about a topic and get a wikipedia response",
)
async def wiki_question(request: WikiRequest):
    wikipedia_api = WikipediaAPIWrapper(  # type: ignore
        doc_content_chars_max=request.max_character_output, top_k_results=request.top_k_results
    )
    wikipedia_tool = WikipediaQueryRun(api_wrapper=wikipedia_api)
    template = """
    wandle die folgende Nutzerfrage in eine Wikipedia suche.
    Wenn es keine frage ist dann suche direkt nach dem Satz aber beantworte nicht die Frage.
    {input}"""
    prompt_template = PromptTemplate.from_template(template)
    chat = ChatOpenAI(model_name="gpt-4o-mini", temperature=request.temperature, max_tokens=request.max_tokens)  # type: ignore
    chain = prompt_template | chat | StrOutputParser() | wikipedia_tool
    result = chain.invoke({"input": request.sentence})
    return result

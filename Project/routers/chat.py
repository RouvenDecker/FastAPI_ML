from starlette import status
from pydantic import BaseModel
from fastapi import APIRouter, Depends, HTTPException
from ..database import SessionLocal
from ..models import User, ChatMessage, ChatSession
from typing import Annotated
from sqlalchemy.orm import Session
from sqlalchemy import desc
import datetime
from .auth import get_current_user


from langchain_openai import ChatOpenAI
from langchain.chains.llm import LLMChain
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_core.prompts import HumanMessagePromptTemplate, ChatPromptTemplate
from langchain_community.chat_message_histories import ChatMessageHistory
from ..preprocessing import clean_sentences, simple_sentiment_pipeline  # type: ignore


router = APIRouter(prefix="/chat", tags=["chat"])


class SentenceRequest(BaseModel):
    sentence: str


class SentencesRequest(BaseModel):
    sentences: list[str]


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


db_dependency = Annotated[Session, Depends(get_db)]
user_dependency = Annotated[dict, Depends(get_current_user)]


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
        if still_active_session(last_session, datetime.timedelta(minutes=10)):  # dtime ändern auf 10 minuten
            active_session = last_session
        else:
            active_session = ChatSession(
                session_id=last_session.session_id + 1, user_id=user.get("id"), last_change=datetime.datetime.now()
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
    system_m = SystemMessage(db_messages[0].content)

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


def build_message_history(history: ChatMessageHistory, messages: ChatMessage) -> ChatMessageHistory:
    for m in messages:
        if m.role == "ai":
            history.add_ai_message(m.content)
        if m.role == "user":
            history.add_user_message(m.content)
    return history


# def convert_to_langchain_messages(db_session_messages: list[ChatMessage]) -> list:
#     lc_messages = []
#     for m in db_session_messages:
#         if m.role == "user":
#             lc_messages.append(HumanMessage(content=m.content))
#         elif m.role == "assistant":
#             lc_messages.append(AIMessage(content=m.content))
#         elif m.role == "system":
#             lc_messages.append(SystemMessage(content=m.content))
#         # Optional: weitere Rollen, z. B. function, tool, etc.

#     return lc_messages


def still_active_session(chat_session: ChatSession, timedelta: datetime.timedelta) -> bool:
    last_activity = chat_session.last_change
    return datetime.datetime.now() - last_activity < timedelta


@router.post("/sentiment/sentence", status_code=status.HTTP_201_CREATED)
async def predict_sentence(request: SentenceRequest):
    s = clean_sentences([request.sentence])
    return simple_sentiment_pipeline(s)


@router.post("/sentiment/sentences", status_code=status.HTTP_201_CREATED)
async def predict_sentences(request: SentencesRequest):
    s = clean_sentences(request.sentences)
    return simple_sentiment_pipeline(s)


@router.post("/response/cynical", status_code=status.HTTP_200_OK)
async def respond_cynical(request: SentenceRequest):
    cynical_sys = "Du bist ein Assistent der Fragen auf zynische weise beantwortet"
    sys_msg = SystemMessage(cynical_sys)
    human_msg = HumanMessagePromptTemplate.from_template("{question}")
    prompt_template = ChatPromptTemplate.from_messages([sys_msg, human_msg])
    chat_value = prompt_template.invoke({"question": request.sentence})

    chat = ChatOpenAI(model_name="gpt-4o-mini", temperature=0.6, max_tokens=100)  # type: ignore

    return chat.invoke(chat_value).content

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


def build_message_history(history: ChatMessageHistory, messages: list[ChatMessage]) -> ChatMessageHistory:
    for m in messages:
        if m.role == "ai":
            history.add_ai_message(m.content)
        if m.role == "user":
            history.add_user_message(m.content)
    return history


def still_active_session(chat_session: ChatSession, timedelta: datetime.timedelta) -> bool:
    last_activity: datetime.datetime = chat_session.last_change
    # sqlite doesnt save datetimes with timezone info
    # the code below attaches the tz info to the datettime after loading it from the db
    last_activity = last_activity.replace(tzinfo=datetime.timezone.utc)
    now = datetime.datetime.now(tz=datetime.timezone.utc)
    return now - last_activity < timedelta


@router.get("/chat/historys")
async def get_chat_historys(user: user_dependency, db: db_dependency):
    if user is None:
        raise HTTPException(status_code=401, detail="Authentication failed")
    chat_sessions = db.query(ChatSession).filter(ChatSession.user_id == user.get("id")).all()
    session_ids = [x.session_id for x in chat_sessions]
    print(session_ids)


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

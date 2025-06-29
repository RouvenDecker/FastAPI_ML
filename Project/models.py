from .database import Base
from sqlalchemy import Column, Integer, String, Boolean, ForeignKey, DateTime, JSON
from sqlalchemy.orm import relationship
import datetime

# database tables


class WebSession(Base):
    __tablename__ = "web_sessions"
    session_id = Column(Integer, primary_key=True, index=True)
    created = Column(DateTime(timezone=True), default=lambda x: datetime.datetime.now(tz=datetime.timezone.utc))
    chroma_path = Column(String)
    websites = Column(JSON)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    user = relationship("User", back_populates="web_sessions")


class RAGSession(Base):
    __tablename__ = "rag_sessions"
    session_id = Column(Integer, primary_key=True, index=True)
    created = Column(DateTime(timezone=True), default=lambda x: datetime.datetime.now(tz=datetime.timezone.utc))
    chroma_path = Column(String)
    files = Column(JSON)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    user = relationship("User", back_populates="rag_sessions")


class ChatSession(Base):
    __tablename__ = "chat_sessions"
    session_id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    last_change = Column(DateTime(timezone=True), default=lambda x: datetime.datetime.now(tz=datetime.timezone.utc))
    chat_messages = relationship("ChatMessage", back_populates="chat_session")
    user = relationship("User", back_populates="chat_sessions")


class ChatMessage(Base):
    __tablename__ = "chat_messages"
    message_id = Column(Integer, primary_key=True)
    session_id = Column(Integer, ForeignKey("chat_sessions.session_id"), primary_key=True, nullable=False)
    content = Column(String)
    role = Column(String)
    date = Column(DateTime(timezone=True), default=lambda x: datetime.datetime.now(datetime.timezone.utc))
    chat_session = relationship("ChatSession", back_populates="chat_messages")


class User(Base):
    __tablename__ = "users"
    id = Column(Integer, primary_key=True, index=True)
    email = Column(String, unique=True)
    username = Column(String)
    first_name = Column(String)
    last_name = Column(String)
    hashed_password = Column(String)
    is_active = Column(Boolean, default=True)
    role = Column(String)
    phone_number = Column(String)
    chat_sessions = relationship("ChatSession", back_populates="user")
    rag_sessions = relationship("RAGSession", back_populates="user")
    web_sessions = relationship("WebSession", back_populates="user")

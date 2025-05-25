from .database import Base
from sqlalchemy import Column, Integer, String, Boolean, ForeignKey, JSON, DateTime
import datetime

# database tables

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

class ChatSession(Base):
    __tablename__ = "chat_sessions"
    session_id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"),nullable=False)
    chat_history = Column(JSON)
    created_at = Column(DateTime, default=datetime.datetime.now(datetime.timezone.utc))

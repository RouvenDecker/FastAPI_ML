from starlette import status
from pydantic import BaseModel
from fastapi import APIRouter, Depends, HTTPException, UploadFile
from ..database import SessionLocal
from ..models import User, ChatMessage, ChatSession
from typing import Annotated
from sqlalchemy.orm import Session
from sqlalchemy import desc
import datetime
from .auth import get_current_user


router = APIRouter(prefix="/upload", tags=["upload"])


@router.post("/upload")
async def file_upload(file: UploadFile):
    return {"filename": file.filename,"headers":file.headers}

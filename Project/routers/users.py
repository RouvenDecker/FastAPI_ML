from fastapi import APIRouter
from fastapi import Depends, HTTPException
from ..models import User
from ..database import SessionLocal
from typing import Annotated
from starlette import status
from pydantic import BaseModel, Field
from .auth import get_current_user
from passlib.context import CryptContext

router = APIRouter(prefix="/user", tags=["user"])


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


db_dependency = Annotated[SessionLocal, Depends(get_db)]
user_dependency = Annotated[dict, Depends(get_current_user)]
bcrypt_context = CryptContext(schemes=["bcrypt"], deprecated="auto")


class UserVerification(BaseModel):
    password: str
    new_password: str = Field(min_length=6)


@router.get("/", status_code=status.HTTP_200_OK)
async def get_user(user: user_dependency, db: db_dependency):
    if user is None:
        raise HTTPException(status_code=401, detail="Authentication failed")
    return db.query(User).filter(user.get("id") == User.id).first()


@router.put("/password", status_code=status.HTTP_204_NO_CONTENT)
async def change_password(user: user_dependency, db: db_dependency, user_verification: UserVerification):
    if user is None:
        raise HTTPException(status_code=401, detail="Authentication failed")
    users_model = db.query(User).filter(user.get("id") == User.id).first()
    if not bcrypt_context.verify(user_verification.password, users_model.hashed_password):
        raise HTTPException(status_code=401, detail="Error on password change")

    users_model.hashed_password = bcrypt_context.hash(user_verification.new_password)
    db.add(users_model)
    db.commit()


@router.put("/phonenumber/{phone_number}", status_code=status.HTTP_204_NO_CONTENT)
async def change_phone_number(user: user_dependency, db: db_dependency, phone_number: str):
    if user is None:
        raise HTTPException(status_code=401, detail="Authentication failed")
    user_model = db.query(User).filter(user.get("id") == User.id).first()
    user_model.phone_number = phone_number  # type: ignore
    db.add(user_model)
    db.commit()

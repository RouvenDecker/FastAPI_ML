from fastapi import APIRouter, Depends, HTTPException
from Project.database import SessionLocal
from typing import Annotated
from sqlalchemy.orm import Session
import datetime
from pathlib import Path
from .auth import get_current_user
from ..models import WebSession
from ..schemas import WebRequest

from langchain_openai.embeddings import OpenAIEmbeddings


router = APIRouter(prefix="/agents", tags=["agents"])
VECTORSTORES = Path.cwd() / "Vectorstores"
EMBEDDINGS = OpenAIEmbeddings()


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


db_dependency = Annotated[Session, Depends(get_db)]
user_dependency = Annotated[dict, Depends(get_current_user)]


# 2. FAQ-Bot für eine Webseite oder ein Unternehmen
# Was?: Ein Bot, der auf Basis einer FAQ-Seite oder Dokumentation Fragen beantwortet.
# Tools:
# Webscraping (z. B. mit BeautifulSoup oder direkt per HTML-Text)
# Vectorstore (z. B. FAISS oder Chroma)
# Warum einfach?: Du kannst die Inhalte vorverarbeiten und dann als statischen QA-Agent einsetzen.
# Use Case: Du trainierst den Bot mit der Firmen-FAQ → Agent antwortet Nutzern.


@router.post("/agent/web")
async def web_agent(db: db_dependency, user: user_dependency, web_request: WebRequest):
    if user is None:
        raise HTTPException(status_code=401, detail="Authentication failed")
    web_session = db.query(WebSession).filter(WebSession.user_id == user.get("id")).first()
    if not web_session:
        web_session = WebSession(
            session_id=0,
            created=datetime.datetime.now(tz=datetime.timezone.utc),
            websites=[web_request.url],
            chroma_path=str(VECTORSTORES / "web_0"),
            user_id=user.get("id")
        )
    else:
        if web_request.url in web_session.websites:
            raise HTTPException(status_code=400, detail="Website already exists in vectorstore")
        web_session.websites = [*web_session.websites, web_request.url]
    db.add(web_session)
    db.commit()

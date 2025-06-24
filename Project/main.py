from fastapi import FastAPI
import os
import gdown
from pathlib import Path
from sqlalchemy.orm import Session
from contextlib import asynccontextmanager
from apscheduler.schedulers.background import BackgroundScheduler
from dotenv import load_dotenv
import datetime
import shutil
from .models import Base, RAGSession
from .database import engine, SessionLocal
from .routers import chat, auth, users, upload, agents

VECTORSTORES = Path.cwd() / "Vectorstores"


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


scheduler = BackgroundScheduler()


# cyclic methods
def delete_rag_session(time_offset: datetime.timedelta) -> None:
    db_gen = get_db()
    db: Session = next(db_gen)
    try:
        datetime_cutoff = datetime.datetime.now(tz=datetime.timezone.utc) - time_offset
        rag_sessions = db.query(RAGSession).filter(RAGSession.created <= datetime_cutoff).all()
        for rag_session in rag_sessions:
            shutil.rmtree(VECTORSTORES / str(rag_session.session_id))
            db.delete(rag_session)
            db.commit()

    finally:
        try:
            next(db_gen)
        except StopIteration:
            pass


# lifespan manager
@asynccontextmanager
async def lifespan(app: FastAPI):
    load_dotenv()

    cwd = Path.cwd()
    model_path = cwd / "Project" / "ML_models"
    if not os.path.exists(model_path):
        os.mkdir(model_path)

    if not os.path.exists(model_path / "tfidf_SVC" / "CountVec_1.pkl"):
        gdown.download(id="18-3HYLD6CvoLDOFEdAXNOP2EFqGI_gax", output=str(model_path / "tfidf_SVC" / "CountVec_1.pkl"))

    if not os.path.exists(model_path / "tfidf_SVC" / "SVC_1.pkl"):
        gdown.download(id="1u0dhZ3mEEWYRXv9H8Y2_6DhaIWZonCmD", output=str(model_path / "tfidf_SVC" / "SVC_1.pkl"))

    if not os.path.exists(model_path / "tfidf_SVC" / "tfidf_transformer_1.pkl"):
        gdown.download(
            id="19UQzUJrvl7togtQnsxk8wlpSeNHTpEPt", output=str(model_path / "tfidf_SVC" / "tfidf_transformer_1.pkl")
        )

    scheduler.add_job(delete_rag_session, "interval", minutes=10, args=[datetime.timedelta(minutes=30)])
    scheduler.start()
    yield
    scheduler.shutdown()


app = FastAPI(lifespan=lifespan)
Base.metadata.create_all(bind=engine)
app.include_router(chat.router)
app.include_router(auth.router)
app.include_router(users.router)
app.include_router(upload.router)
app.include_router(agents.router)

from fastapi import FastAPI
import os
import gdown
from pathlib import Path
from contextlib import asynccontextmanager
from dotenv import load_dotenv
from .models import Base
from .database import engine
from .routers import chat,auth,users


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
        gdown.download(id="19UQzUJrvl7togtQnsxk8wlpSeNHTpEPt", output=str(model_path / "tfidf_SVC" / "tfidf_transformer_1.pkl"))

    yield


app = FastAPI(lifespan=lifespan)
Base.metadata.create_all(bind=engine)
app.include_router(chat.router)
app.include_router(auth.router)
app.include_router(users.router)

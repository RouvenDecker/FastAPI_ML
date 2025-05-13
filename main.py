from fastapi import FastAPI
from pydantic import BaseModel
from starlette import status
import joblib
import os
import gdown
from pathlib import Path
from contextlib import asynccontextmanager

from schemas import SentenceRequest, SentencesRequest

from preprocessing import clean_sentences, simple_sentiment_pipeline
from sklearn.feature_extraction.text import CountVectorizer




@asynccontextmanager
async def lifespan(app: FastAPI):
    cwd = Path.cwd()
    model_path = cwd / "ML_models" / "tfidf_SVC"
    if not os.path.exists(model_path):
        os.mkdir(model_path)

    if not os.path.exists(model_path / "CountVec_1.pkl"):
        gdown.download(id="18-3HYLD6CvoLDOFEdAXNOP2EFqGI_gax", output=str(model_path / "CountVec_1.pkl"))

    if not os.path.exists(model_path / "SVC_1.pkl"):
        gdown.download(id="1u0dhZ3mEEWYRXv9H8Y2_6DhaIWZonCmD", output=str(model_path / "SVC_1.pkl"))

    if not os.path.exists(model_path / "tfidf_transformer_1.pkl"):
        gdown.download(id="19UQzUJrvl7togtQnsxk8wlpSeNHTpEPt", output=str(model_path / "tfidf_transformer_1.pkl"))

    yield



app = FastAPI(lifespan=lifespan)





@app.post("/predict/sentence", status_code=status.HTTP_201_CREATED)
async def predict_sentence(request: SentenceRequest):
    s = clean_sentences([request.sentence])
    return simple_sentiment_pipeline(s)


@app.post("/predict/sentences", status_code=status.HTTP_201_CREATED)
async def predict_sentences(request: SentencesRequest):
    s = clean_sentences(request.sentences)
    return simple_sentiment_pipeline(s)

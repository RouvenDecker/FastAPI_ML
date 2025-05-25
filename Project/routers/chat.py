from starlette import status
from pydantic import BaseModel
from fastapi import APIRouter

from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage
from langchain_core.prompts import HumanMessagePromptTemplate, ChatPromptTemplate

from ..preprocessing import clean_sentences, simple_sentiment_pipeline


router = APIRouter(
    prefix= "/chat",
    tags=["chat"]
)


class SentenceRequest(BaseModel):
    sentence: str


class SentencesRequest(BaseModel):
    sentences: list[str]


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

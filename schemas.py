from pydantic import BaseModel


class SentenceRequest(BaseModel):
    sentence: str


class SentencesRequest(BaseModel):
    sentences: list[str]

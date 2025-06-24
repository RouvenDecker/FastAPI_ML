from pydantic import BaseModel, Field


class SentenceRequest(BaseModel):
    sentence: str
    temperature: float = Field(gt=0, lt=1, default=0.5)
    max_tokens: int = Field(gt=0, lt=500, default=150)


class SentencesRequest(BaseModel):
    sentences: list[str]
    temperature: float = Field(gt=0, lt=1, default=0.5)
    max_tokens: int = Field(gt=0, lt=500, default=150)


class WikiRequest(BaseModel):
    sentence: str = Field(
        description="""your question about some website provided""")
    temperature: float = Field(gt=0, lt=1, default=0.5)
    max_tokens: int = Field(gt=0, lt=500, default=150)
    max_character_output: int = Field(gt=0, default=1000)
    top_k_results: int = 1


class WebRequest(BaseModel):
    sentence: str
    temperature: float = Field(gt=0, lt=1, default=0.5)
    max_tokens: int = Field(gt=0, lt=500, default=150)
    url: str | None

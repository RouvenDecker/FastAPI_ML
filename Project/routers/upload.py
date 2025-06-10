from starlette import status

from fastapi import APIRouter, Depends, HTTPException, UploadFile
from ..database import SessionLocal
from ..models import RAGSession
from typing import Annotated
from sqlalchemy.orm import Session
import datetime
from .auth import get_current_user
from pathlib import Path
import shutil
import os
import re

from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_core.documents.base import Document

VECTORSTORES = Path.cwd() / "Vectorstores"
TEMP = Path.cwd() / "Temp"
EMBEDDINGS = OpenAIEmbeddings()


router = APIRouter(prefix="/upload", tags=["upload"])


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


db_dependency = Annotated[Session, Depends(get_db)]
user_dependency = Annotated[dict, Depends(get_current_user)]


def save_file(file: UploadFile, filename: str | Path) -> None:
    """save a FastAPI UploadFile object

    Args:
        file (UploadFile): FastAPI UploadFile object
        path (str | Path): full qualified filename with path
    """
    with open(filename, "wb") as f:
        f.write(file.file.read())


def delete_file(filename: str | Path) -> None:
    """delete a file

    Args:
        path (str | Path): full qualified filename with path
    """
    os.remove(filename)


def get_documents(file: UploadFile, chunksize: int = 500, chunk_overlap: int = 50) -> list[Document]:
    """generate a Documents from  Uploadfile object

    Args:
        file (UploadFile): FastAPI UploadFile object
        chunksize (int, optional): Document chunksize. Defaults to 500.
        chunk_overlap (int, optional): Document chunk overlap. Defaults to 50.

    Raises:
        HTTPException: throws if content type is not supported

    Returns:
        list[Document]: list of chunked Documents
    """
    file_location = TEMP / file.filename

    save_file(file, file_location)
    match file.content_type.split("/")[-1].lower():
        case "pdf":
            pdf_loader = PyPDFLoader(file_location)
            pages = pdf_loader.load()
        case "docx":
            docx_loader = Docx2txtLoader(file_location)
            pages = docx_loader.load()
        case _:
            raise HTTPException(status_code=400, detail="unprocessable entity")

    for i in range(len(pages)):
        pages[i].page_content = " ".join(pages[i].page_content.split())

    text_splitter = CharacterTextSplitter(separator=".", chunk_size=chunksize, chunk_overlap=chunk_overlap)
    docs = text_splitter.split_documents(pages)
    delete_file(file_location)
    return docs


@router.post("/upload_file")
async def file_upload(file: UploadFile, user: user_dependency, db: db_dependency):
    if user is None:
        raise HTTPException(status_code=401, detail="Authentication failed")

    filetype = file.content_type.split("/")[-1].lower()
    if filetype not in ["pdf", "docx"]:
        raise HTTPException(status_code=422, detail="unprocessable datatype, allowed are [.pdf,.docx]")

    if re.search(r" ", file.filename):
        raise HTTPException(status_code=400, detail="no whitespace in filename allowed")

    rag_session = db.query(RAGSession).filter(user.get("id") == RAGSession.user_id).first()  # type: ignore
    vectorstore: Chroma = None

    documents = get_documents(file, chunksize=500, chunk_overlap=50)

    if rag_session:  # session does exist
        if file.filename in rag_session.files:
            raise HTTPException(status_code=400, detail="file already exists in database")
        # load the corresponding chroma database
        print("session found")
        vectorstore = Chroma(
            persist_directory=str(VECTORSTORES / str(rag_session.session_id)), embedding_function=EMBEDDINGS
        )
        rag_session.files = [*rag_session.files, file.filename]

        db.add(rag_session)
        db.commit()

    else:
        # if no rag session exists create a session
        print("new RAG Session created")

        new_session = RAGSession(
            session_id=0,
            created=datetime.datetime.now(tz=datetime.timezone.utc),
            chroma_path=str(VECTORSTORES / "0"),
            files=[file.filename],
            user_id=user.get("id"),
        )
        db.add(new_session)
        db.commit()
        # create a new vectorstore
        vectorstore = Chroma(persist_directory=str(VECTORSTORES / "0"), embedding_function=EMBEDDINGS)

    vectorstore.add_documents(documents)

    return {"filename": file.filename, "document_count": len(documents)}

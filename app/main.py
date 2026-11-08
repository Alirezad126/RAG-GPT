import logging
import shutil
import uuid
from pathlib import Path
from typing import List, Optional, Union

from fastapi import (
    Cookie,
    Depends,
    FastAPI,
    File,
    HTTPException,
    Response,
    UploadFile,
)
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from app.config import (
    ALLOWED_ORIGINS,
    MAX_UPLOAD_BYTES,
    UPLOAD_DIR,
    VECTOR_DIR,
)
from app.rag.chain import answer
from app.rag.embeddings import create_vectorstore
from app.rag.pdf import process_pdf

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="RAG-GPT", description="Chat with your own PDF.")

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class Turn(BaseModel):
    message: str
    type: str
    id: int
    loading: Union[bool, None] = None


class ChatRequest(BaseModel):
    message: str
    conversationState: List[Turn]
    session_id: str


def get_session_id(session_id: Optional[str] = Cookie(None)) -> str:
    return session_id or str(uuid.uuid4())


def session_paths(session_id: str):
    """Resolve a session's directories, rejecting ids that escape the data dir."""
    try:
        uuid.UUID(session_id)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid session id.")
    return UPLOAD_DIR / session_id, VECTOR_DIR / session_id


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.post("/upload")
async def upload_file(
    response: Response,
    session_id: str = Depends(get_session_id),
    file: UploadFile = File(...),
):
    if not (file.filename or "").lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are accepted.")

    contents = await file.read()
    if len(contents) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail="File is too large.")

    upload_dir, _ = session_paths(session_id)
    upload_dir.mkdir(parents=True, exist_ok=True)
    destination = upload_dir / Path(file.filename).name
    destination.write_bytes(contents)

    try:
        docs = process_pdf(destination)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    create_vectorstore(docs, session_id)

    response.set_cookie(key="session_id", value=session_id, samesite="none", secure=True)
    logger.info("Indexed %d chunks for session %s", len(docs), session_id)
    return {"message": "File uploaded and processed successfully", "session_id": session_id}


@app.post("/chat")
async def chat(body: ChatRequest):
    session_paths(body.session_id)
    return {
        "AIResponse": answer(body.session_id, body.message, body.conversationState)
    }


@app.post("/end_session")
async def end_session(response: Response, session_id: str = Depends(get_session_id)):
    for directory in session_paths(session_id):
        if directory.exists():
            shutil.rmtree(directory, ignore_errors=True)
    response.delete_cookie(key="session_id")
    logger.info("Ended session %s", session_id)
    return {"message": "Session ended and files deleted"}

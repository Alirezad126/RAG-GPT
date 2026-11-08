import io
import logging
import uuid
from typing import List, Optional, Union

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from groq import APIStatusError
from pydantic import BaseModel

from app import store
from app.config import ALLOWED_ORIGINS, MAX_UPLOAD_BYTES, MODELS
from app.rag.answer import answer, available_models, default_model, resolve
from app.rag.pdf import chunk_pdf
from app.rag.retrieval import embed

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="RAG-GPT", description="Chat with your own PDF.")

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


class Turn(BaseModel):
    message: str
    type: str
    id: Optional[int] = None
    loading: Union[bool, None] = None


class ChatRequest(BaseModel):
    message: str
    conversationState: List[Turn] = []
    session_id: str
    model: Optional[str] = None


class SessionRequest(BaseModel):
    session_id: str


def valid_session_id(session_id):
    """Sessions are client-held UUIDs; reject anything else before it reaches S3."""
    try:
        return str(uuid.UUID(str(session_id)))
    except (ValueError, AttributeError, TypeError):
        raise HTTPException(status_code=400, detail="Invalid session id.")


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.get("/models")
async def models():
    """The models the client may choose from, for the picker in the UI."""
    return {
        "default": default_model(),
        "models": [
            {"id": model, **MODELS[model]} for model in available_models()
        ],
    }


@app.post("/upload")
async def upload(
    file: UploadFile = File(...),
    session_id: Optional[str] = Form(None),
):
    if not (file.filename or "").lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are accepted.")

    contents = await file.read()
    if len(contents) > MAX_UPLOAD_BYTES:
        raise HTTPException(
            status_code=413,
            detail=f"PDF is larger than {MAX_UPLOAD_BYTES // (1024 * 1024)} MB.",
        )

    session_id = valid_session_id(session_id) if session_id else str(uuid.uuid4())

    try:
        chunks = chunk_pdf(io.BytesIO(contents))
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))

    store.save(session_id, chunks, embed(chunks))
    logger.info("Indexed %d chunks for session %s", len(chunks), session_id)
    return {
        "message": "File uploaded and processed successfully",
        "session_id": session_id,
        "chunks": len(chunks),
    }


@app.post("/chat")
async def chat(body: ChatRequest):
    session_id = valid_session_id(body.session_id)
    index = store.load(session_id)
    if index is None:
        raise HTTPException(status_code=404, detail="Upload a document first.")

    chunks, vectors = index
    model = resolve(body.model)
    try:
        response = answer(
            body.message, body.conversationState, chunks, vectors, model
        )
    except APIStatusError as exc:
        # Pass the provider's own status through rather than turning a rate
        # limit or a retired model into an opaque 500.
        detail = f"The model provider returned {exc.status_code}."
        if exc.status_code == 429:
            detail = "Rate limited by the free tier — try again in a minute."
        logger.warning("Provider error for %s: %s", model, exc)
        raise HTTPException(status_code=502 if exc.status_code != 429 else 429, detail=detail)
    return {"AIResponse": response, "model": model}


@app.post("/end_session")
async def end_session(body: SessionRequest):
    session_id = valid_session_id(body.session_id)
    store.delete(session_id)
    logger.info("Ended session %s", session_id)
    return {"message": "Session ended and files deleted"}

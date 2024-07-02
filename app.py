from fastapi import FastAPI, UploadFile, File, Depends, Response, Cookie, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, ValidationError
from typing import List, Union, Optional
from RAGModel.LLMModel import get_completion
from RAGModel.pdfReader import process_pdf
from RAGModel.embeddingModel import create_embedding_from_texts, load_embedding_vectordb
import uvicorn
import os
import uuid
import shutil
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://alirezadaneshvar.com", "http://localhost:5173"],  # Allows only specific origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

UPLOAD_FOLDER = "./pdf_files"
SESSION_DATA = {}

class ConversationState(BaseModel):
    message: str
    type: str
    id: int
    loading: Union[bool, None] = None  # `None` allows for optional field

class RequestBody(BaseModel):
    message: str
    conversationState: List[ConversationState]
    session_id: str  # Ensure session_id is part of the request body

def get_session_id(session_id: Optional[str] = Cookie(None)):
    if session_id is None:
        session_id = str(uuid.uuid4())
    return session_id

@app.post("/upload")
async def upload_file(response: Response, session_id: str = Depends(get_session_id), file: UploadFile = File(...)):
    file_location = os.path.join(UPLOAD_FOLDER, session_id, file.filename)
    
    # Ensure the directory exists
    os.makedirs(os.path.dirname(file_location), exist_ok=True)
    
    with open(file_location, "wb+") as file_object:
        file_object.write(file.file.read())

    docs = process_pdf(file_location)
    create_embedding_from_texts(docs, session_id)

    # Track the file with the session ID
    if session_id not in SESSION_DATA:
        SESSION_DATA[session_id] = []
    SESSION_DATA[session_id].append(file_location)
    
    # Set the session ID in the response cookie
    response.set_cookie(key="session_id", value=session_id)

    logger.info(f"File uploaded and processed successfully for session_id: {session_id}")
    return {"message": "File uploaded and processed successfully", "session_id": session_id}

@app.post("/chat")
async def result(body: dict, session_id: str = Depends(get_session_id)):
    try:
        body = RequestBody(**body)
    except ValidationError as e:
        raise HTTPException(status_code=422, detail=e.errors())

    completion = get_completion(body)
    return {"AIResponse": completion}

@app.post("/end_session")
async def end_session(response: Response, session_id: str = Depends(get_session_id)):
    logger.info(f"Ending session for session_id: {session_id}")

    # Delete the entire directory associated with the session
    if session_id in SESSION_DATA:
        session_dir = os.path.dirname(SESSION_DATA[session_id][0])
        if os.path.exists(session_dir):
            logger.info(f"Deleting session directory: {session_dir}")
            shutil.rmtree(session_dir)
        else:
            logger.warning(f"Session directory not found: {session_dir}")
        del SESSION_DATA[session_id]
    
    # Remove session-specific database directory
    db_dir = os.path.join("./RAGModel/vectorDB", session_id)
    if os.path.exists(db_dir):
        logger.info(f"Deleting database directory: {db_dir}")
        shutil.rmtree(db_dir)
    else:
        logger.warning(f"Database directory not found: {db_dir}")
    
    # Remove the session cookie
    response.delete_cookie(key="session_id")

    return {"message": "Session ended and files deleted"}

if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000, log_level="info", reload=True)

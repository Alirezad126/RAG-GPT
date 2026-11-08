import os
from pathlib import Path

# OpenAI
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
CHAT_MODEL = os.environ.get("CHAT_MODEL", "gpt-4o-mini")
EMBEDDING_MODEL = os.environ.get("EMBEDDING_MODEL", "text-embedding-3-small")

# Retrieval
CHUNK_SIZE = int(os.environ.get("CHUNK_SIZE", 512))
CHUNK_OVERLAP = int(os.environ.get("CHUNK_OVERLAP", 20))
TOP_K = int(os.environ.get("TOP_K", 8))

# Storage: both uploads and vector stores live under one data directory so a
# deployment only has to mount a single volume (or none, for ephemeral use).
DATA_DIR = Path(os.environ.get("DATA_DIR", "./data"))
UPLOAD_DIR = DATA_DIR / "uploads"
VECTOR_DIR = DATA_DIR / "vectorstores"

MAX_UPLOAD_BYTES = int(os.environ.get("MAX_UPLOAD_BYTES", 20 * 1024 * 1024))

ALLOWED_ORIGINS = [
    origin.strip()
    for origin in os.environ.get(
        "ALLOWED_ORIGINS", "http://localhost:5173"
    ).split(",")
    if origin.strip()
]

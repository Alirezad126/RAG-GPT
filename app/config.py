import os

# OpenAI
CHAT_MODEL = os.environ.get("CHAT_MODEL", "gpt-4o-mini")
EMBEDDING_MODEL = os.environ.get("EMBEDDING_MODEL", "text-embedding-3-small")

# Retrieval
CHUNK_SIZE = int(os.environ.get("CHUNK_SIZE", 512))
CHUNK_OVERLAP = int(os.environ.get("CHUNK_OVERLAP", 20))
TOP_K = int(os.environ.get("TOP_K", 8))

# Where session embeddings live. S3 rather than local disk, because a Lambda
# invocation keeps nothing: the upload and the questions that follow it are
# separate invocations on possibly different execution environments.
S3_BUCKET = os.environ.get("S3_BUCKET")
S3_PREFIX = os.environ.get("S3_PREFIX", "sessions")

# A Lambda function URL accepts at most 6 MB per request, and binary bodies are
# base64-encoded on the way in, which inflates them by about a third.
MAX_UPLOAD_BYTES = int(os.environ.get("MAX_UPLOAD_BYTES", 4 * 1024 * 1024))

ALLOWED_ORIGINS = [
    origin.strip()
    for origin in os.environ.get(
        "ALLOWED_ORIGINS", "http://localhost:5173"
    ).split(",")
    if origin.strip()
]

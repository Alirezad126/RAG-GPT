import os

# Generation. Groq serves open-weight models on an OpenAI-compatible API, and
# its free tier covers portfolio traffic. MODELS is an allowlist: the client
# picks from it, so a request cannot point the function at an arbitrary model.
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")

MODELS = {
    "openai/gpt-oss-20b": {
        "label": "GPT-OSS 20B",
        "detail": "Smallest and quickest of the three.",
    },
    "openai/gpt-oss-120b": {
        "label": "GPT-OSS 120B",
        "detail": "Strongest reasoning, a little slower.",
    },
    "qwen/qwen3-32b": {
        "label": "Qwen3 32B",
        "detail": "Good on dense, technical documents.",
    },
}
DEFAULT_MODEL = os.environ.get("DEFAULT_MODEL", "openai/gpt-oss-20b")
MAX_TOKENS = int(os.environ.get("MAX_TOKENS", 512))

# Embeddings run inside the function on CPU — an ONNX model baked into the
# image, so indexing a document costs nothing and calls no API.
EMBEDDING_MODEL = os.environ.get("EMBEDDING_MODEL", "BAAI/bge-small-en-v1.5")

# Retrieval
CHUNK_SIZE = int(os.environ.get("CHUNK_SIZE", 512))
CHUNK_OVERLAP = int(os.environ.get("CHUNK_OVERLAP", 64))
TOP_K = int(os.environ.get("TOP_K", 6))

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

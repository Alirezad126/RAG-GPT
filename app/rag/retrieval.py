import numpy as np
from langchain_openai import OpenAIEmbeddings

from app.config import EMBEDDING_MODEL, TOP_K

_embeddings = None


def embeddings():
    global _embeddings
    if _embeddings is None:
        _embeddings = OpenAIEmbeddings(model=EMBEDDING_MODEL)
    return _embeddings


def embed_chunks(chunks):
    return np.asarray(embeddings().embed_documents(chunks), dtype=np.float32)


def top_chunks(question, chunks, vectors, k=TOP_K):
    """The k chunks whose embeddings are closest to the question's."""
    query = np.asarray(embeddings().embed_query(question), dtype=np.float32)

    # OpenAI returns unit-length vectors, so a dot product is the cosine
    # similarity; normalize anyway so the ranking survives a model that doesn't.
    norms = np.linalg.norm(vectors, axis=1) * np.linalg.norm(query)
    scores = vectors @ query / np.where(norms == 0, 1, norms)

    best = np.argsort(scores)[::-1][:k]
    return [chunks[i] for i in best]

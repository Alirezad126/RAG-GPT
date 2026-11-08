import numpy as np

from app.config import EMBEDDING_MODEL, TOP_K

_model = None


def model():
    """The ONNX embedding model, loaded once per execution environment."""
    global _model
    if _model is None:
        from fastembed import TextEmbedding

        _model = TextEmbedding(EMBEDDING_MODEL)
    return _model


def embed(texts):
    return np.asarray(list(model().embed(texts)), dtype=np.float32)


def top_chunks(question, chunks, vectors, k=TOP_K):
    """The k chunks whose embeddings are closest to the question's."""
    query = embed([question])[0]

    # The model returns unit-length vectors, so a dot product is the cosine
    # similarity; normalize anyway so the ranking survives a model that doesn't.
    norms = np.linalg.norm(vectors, axis=1) * np.linalg.norm(query)
    scores = vectors @ query / np.where(norms == 0, 1, norms)

    best = np.argsort(scores)[::-1][:k]
    return [chunks[i] for i in best]

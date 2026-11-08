import os

os.environ.setdefault("S3_BUCKET", "test-bucket")
os.environ.setdefault("GROQ_API_KEY", "test-key")
os.environ.setdefault("AWS_ACCESS_KEY_ID", "testing")
os.environ.setdefault("AWS_SECRET_ACCESS_KEY", "testing")
os.environ.setdefault("AWS_DEFAULT_REGION", "us-east-1")

import numpy as np
import pytest
from fastapi.testclient import TestClient
from moto import mock_aws


@pytest.fixture
def calls():
    """Records the model each generation call was made with."""
    return []


@pytest.fixture
def client(monkeypatch, calls):
    """A test client with S3 mocked, and embeddings and Groq stubbed out."""
    import app.main
    import app.rag.answer
    import app.rag.retrieval
    import app.store

    def fake_embed(texts):
        return np.eye(len(texts), 4, dtype=np.float32)

    monkeypatch.setattr(app.main, "embed", fake_embed)
    monkeypatch.setattr(app.rag.retrieval, "embed", fake_embed)

    def fake_complete(messages, model, max_tokens=None):
        calls.append(model)
        return "stub answer"

    monkeypatch.setattr(app.rag.answer, "complete", fake_complete)

    with mock_aws():
        app.store._client = None
        app.store.client().create_bucket(Bucket="test-bucket")
        yield TestClient(app.main.app)
        app.store._client = None

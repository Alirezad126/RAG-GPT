import os

os.environ.setdefault("S3_BUCKET", "test-bucket")
os.environ.setdefault("OPENAI_API_KEY", "test-key")
os.environ.setdefault("AWS_ACCESS_KEY_ID", "testing")
os.environ.setdefault("AWS_SECRET_ACCESS_KEY", "testing")
os.environ.setdefault("AWS_DEFAULT_REGION", "us-east-1")

import numpy as np
import pytest
from fastapi.testclient import TestClient
from moto import mock_aws


@pytest.fixture
def client(monkeypatch):
    """A test client with S3 mocked and the OpenAI calls stubbed out."""
    import app.main
    import app.rag.answer
    import app.rag.retrieval
    import app.store

    def fake_embed_chunks(chunks):
        return np.eye(len(chunks), 4, dtype=np.float32)

    monkeypatch.setattr(app.main, "embed_chunks", fake_embed_chunks)
    monkeypatch.setattr(
        app.rag.retrieval,
        "embeddings",
        lambda: type("E", (), {"embed_query": lambda self, q: [1.0, 0.0, 0.0, 0.0]})(),
    )
    monkeypatch.setattr(
        app.rag.answer,
        "llm",
        lambda: type(
            "L", (), {"invoke": lambda self, m: type("R", (), {"content": "stub"})()}
        )(),
    )

    with mock_aws():
        app.store._client = None
        app.store.client().create_bucket(Bucket="test-bucket")
        yield TestClient(app.main.app)
        app.store._client = None

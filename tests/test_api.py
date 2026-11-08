import io

from pypdf import PdfWriter


def make_pdf():
    """A one-page PDF with no text, used where parsing should fail."""
    writer = PdfWriter()
    writer.add_blank_page(width=200, height=200)
    buffer = io.BytesIO()
    writer.write(buffer)
    return buffer.getvalue()


def test_health(client):
    assert client.get("/health").json() == {"status": "ok"}


def test_models_lists_the_allowlist(client):
    body = client.get("/models").json()
    assert body["default"] in [model["id"] for model in body["models"]]
    assert all({"id", "label", "detail"} <= set(model) for model in body["models"])


def test_upload_rejects_non_pdf(client):
    response = client.post("/upload", files={"file": ("a.txt", b"hello", "text/plain")})
    assert response.status_code == 400


def test_upload_rejects_pdf_without_text(client):
    response = client.post(
        "/upload", files={"file": ("blank.pdf", make_pdf(), "application/pdf")}
    )
    assert response.status_code == 422


def test_chat_rejects_bad_session_id(client):
    response = client.post(
        "/chat", json={"message": "hi", "conversationState": [], "session_id": "nope"}
    )
    assert response.status_code == 400


def test_chat_without_upload_is_not_found(client):
    response = client.post(
        "/chat",
        json={
            "message": "hi",
            "conversationState": [],
            "session_id": "11111111-1111-4111-8111-111111111111",
        },
    )
    assert response.status_code == 404


def test_round_trip(client, monkeypatch):
    import app.main

    monkeypatch.setattr(app.main, "chunk_pdf", lambda data: ["first", "second"])

    upload = client.post(
        "/upload", files={"file": ("doc.pdf", make_pdf(), "application/pdf")}
    )
    assert upload.status_code == 200
    session_id = upload.json()["session_id"]
    assert upload.json()["chunks"] == 2

    chat = client.post(
        "/chat",
        json={"message": "what is it", "conversationState": [], "session_id": session_id},
    )
    assert chat.status_code == 200
    assert chat.json()["AIResponse"] == "stub answer"

    assert client.post("/end_session", json={"session_id": session_id}).status_code == 200

    gone = client.post(
        "/chat",
        json={"message": "again", "conversationState": [], "session_id": session_id},
    )
    assert gone.status_code == 404


def test_model_choice_is_honoured(client, monkeypatch, calls):
    import app.main

    monkeypatch.setattr(app.main, "chunk_pdf", lambda data: ["first", "second"])
    session_id = client.post(
        "/upload", files={"file": ("doc.pdf", make_pdf(), "application/pdf")}
    ).json()["session_id"]

    response = client.post(
        "/chat",
        json={
            "message": "what is it",
            "conversationState": [],
            "session_id": session_id,
            "model": "qwen/qwen3-32b",
        },
    )
    assert response.json()["model"] == "qwen/qwen3-32b"
    assert calls == ["qwen/qwen3-32b"]


def test_unknown_model_falls_back_to_the_default(client, monkeypatch, calls):
    import app.main
    from app.config import DEFAULT_MODEL

    monkeypatch.setattr(app.main, "chunk_pdf", lambda data: ["first", "second"])
    session_id = client.post(
        "/upload", files={"file": ("doc.pdf", make_pdf(), "application/pdf")}
    ).json()["session_id"]

    response = client.post(
        "/chat",
        json={
            "message": "what is it",
            "conversationState": [],
            "session_id": session_id,
            "model": "something/not-allowed",
        },
    )
    assert response.json()["model"] == DEFAULT_MODEL
    assert calls == [DEFAULT_MODEL]

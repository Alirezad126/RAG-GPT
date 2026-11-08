# RAG-GPT

A small retrieval-augmented generation service: upload a PDF, then ask questions
about it. Each visitor gets an isolated session with its own vector store, which
is deleted when the session ends.

The repository contains the FastAPI backend. It is used by the chat widget on
[alirezadaneshvar.com](https://alirezadaneshvar.com).

## How it works

1. **Ingest** — the uploaded PDF is read with `pypdf` and split into overlapping
   512-character chunks.
2. **Embed** — chunks are embedded with OpenAI embeddings and persisted to a
   per-session Chroma collection.
3. **Retrieve** — a history-aware retriever first rewrites the question into a
   standalone one, then pulls the top `k` chunks for it.
4. **Answer** — the chunks and the chat history are passed to the chat model,
   which is instructed to answer only from the document and to say when it does
   not know.

```
app/
├── main.py        FastAPI app and endpoints
├── config.py      configuration, read from the environment
└── rag/
    ├── pdf.py         PDF parsing and chunking
    ├── embeddings.py  per-session Chroma vector store
    ├── prompts.py     prompt templates and chain assembly
    └── chain.py       question answering
```

## API

| Method | Path           | Description                                             |
| ------ | -------------- | ------------------------------------------------------- |
| `GET`  | `/health`      | Liveness check.                                         |
| `POST` | `/upload`      | Multipart `file` (PDF). Indexes it and sets a session cookie. |
| `POST` | `/chat`        | `{ message, conversationState, session_id }` → `{ AIResponse }`. |
| `POST` | `/end_session` | Deletes the session's upload and vector store.          |

Interactive docs are served at `/docs`.

## Running locally

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env        # then put your OpenAI key in it
set -a && source .env && set +a
uvicorn app.main:app --reload
```

## Configuration

All settings come from the environment; see [`.env.example`](.env.example).
`OPENAI_API_KEY` is the only required one. `ALLOWED_ORIGINS` is a comma-separated
list of browser origins permitted to call the API, and `DATA_DIR` is where
uploads and vector stores are written.

## Deployment

The image is self-contained and listens on `$PORT`:

```bash
docker build -t rag-gpt .
docker run -p 8000:8000 -e OPENAI_API_KEY=sk-... rag-gpt
```

[`render.yaml`](render.yaml) is a ready blueprint for Render: it builds the
Dockerfile, mounts a 1 GB disk at `/data`, health-checks `/health` and expects
`OPENAI_API_KEY` to be set in the dashboard.

Any other container host works too. Sessions keep state on disk, so either attach a volume
at `DATA_DIR` or accept that an instance restart clears in-flight sessions — and
run a single instance, since a session's vector store lives on the instance that
created it.

## License

MIT — see [LICENSE](LICENSE).

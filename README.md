# RAG-GPT

A retrieval-augmented generation service: upload a PDF, then ask questions
about it. Each visitor gets an isolated session, and the session's data is
deleted on request.

It runs on open-weight models and costs nothing to operate. Embeddings are
computed inside the function by a small ONNX model, so indexing calls no API at
all; generation goes to [Groq](https://console.groq.com), which serves
open-weight models on a free tier. The visitor picks which model answers.

Deployed as a single AWS Lambda function behind a function URL, with session
embeddings in S3 — no server and no vector database to keep alive.

## How it works

1. **Ingest** — the PDF is read with `pypdf` and split into overlapping
   512-character chunks, breaking on paragraphs before lines before sentences.
2. **Embed** — the chunks are embedded locally with `bge-small-en-v1.5`
   (384 dimensions, ~64 MB of ONNX weights). The texts and vectors are written
   to S3 under the session id.
3. **Retrieve** — a follow-up question is first rewritten into a standalone one
   using the chat history, then embedded and compared against the stored vectors
   by cosine similarity. The top `k` chunks become the context.
4. **Answer** — context and history go to the chosen model, which is instructed
   to answer only from the document and to say when it does not know.

Two deliberate omissions. There is **no vector database**: one PDF is a few
hundred chunks, so the index loads in a single request and a NumPy dot product
searches it in microseconds — Chroma or pgvector would mean a process to keep
running without being any faster at this size. And there is **no LLM in the
function**: a model worth showing needs a GPU, so generation is an API call,
while embedding is small enough to run on Lambda's CPU in milliseconds.

```
app/
├── main.py            FastAPI app and endpoints
├── lambda_handler.py  Mangum adapter — the Lambda entry point
├── config.py          configuration and the model allowlist
├── store.py           per-session index, persisted to S3
└── rag/
    ├── pdf.py         PDF text extraction
    ├── chunking.py    recursive character splitting
    ├── retrieval.py   ONNX embeddings and similarity search
    └── answer.py      question rewriting and answering
```

## Models

| Model                 | Notes                                 |
| --------------------- | ------------------------------------- |
| `openai/gpt-oss-20b`  | Default. Smallest and quickest.       |
| `openai/gpt-oss-120b` | Strongest reasoning, a little slower. |
| `qwen/qwen3.8-27b`    | Good on dense, technical documents.   |

`GET /models` returns this list for the picker in the UI, and `POST /chat`
accepts a `model` field. The list in `app/config.py` is an allowlist — an
unrecognized model falls back to the default rather than being passed through,
so a request cannot point the function at an arbitrary or paid model.

It is also intersected at runtime with the models the provider reports it is
serving. Groq retires models from the free catalogue without notice, so an
entry that disappears stops being offered instead of failing when a visitor
picks it, and `config.py` can carry candidates that are not currently served.

Groq's free tier is roughly 30 requests per minute and 1,000 per day, shared
across these models, which is ample for a portfolio demo but worth knowing
before putting it anywhere busier. A provider error is passed through with its
own status — 429 for a rate limit, 502 otherwise — rather than becoming an
opaque 500.

## API

| Method | Path           | Body                                            | Returns                |
| ------ | -------------- | ----------------------------------------------- | ---------------------- |
| `GET`  | `/health`      | —                                               | `{status}`             |
| `GET`  | `/models`      | —                                               | `{default, models}`    |
| `POST` | `/upload`      | multipart `file` (PDF), optional `session_id`   | `{session_id, chunks}` |
| `POST` | `/chat`        | `{message, conversationState, session_id, model}` | `{AIResponse, model}`  |
| `POST` | `/end_session` | `{session_id}`                                  | `{message}`            |

The session id is held by the client and passed explicitly — not in a cookie,
which a browser would treat as third-party and drop, since the API and the front
end are on different domains. Interactive docs are at `/docs` when running
locally.

Uploads are capped at 4 MB: a Lambda function URL accepts 6 MB per request and
binary bodies arrive base64-encoded, which costs about a third of that. For
larger files the upload would have to go straight to S3 with a presigned URL.

## Running locally

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements-dev.txt
cp .env.example .env        # then fill in GROQ_API_KEY and S3_BUCKET
set -a && source .env && set +a
uvicorn app.main:app --reload
```

The embedding model downloads on first use and is cached. S3 is used even
locally, so the AWS credentials in your environment need read/write access to
the bucket. Run the tests with `pytest` — they mock S3 and stub both the
embedder and Groq, so they need no credentials and no network.

## Deploying

One-time setup, with `ACCOUNT_ID`, `REGION` and a globally unique bucket name
filled in:

```bash
# 1. Bucket for session data, with sessions expiring after a day
aws s3 mb s3://rag-gpt-sessions-SUFFIX
aws s3api put-bucket-lifecycle-configuration \
  --bucket rag-gpt-sessions-SUFFIX \
  --lifecycle-configuration '{"Rules":[{"ID":"expire-sessions","Status":"Enabled","Filter":{"Prefix":"sessions/"},"Expiration":{"Days":1}}]}'

# 2. ECR repository for the image
aws ecr create-repository --repository-name rag-gpt

# 3. Execution role: basic Lambda logging plus access to that one bucket
aws iam create-role --role-name rag-gpt-lambda \
  --assume-role-policy-document '{"Version":"2012-10-17","Statement":[{"Effect":"Allow","Principal":{"Service":"lambda.amazonaws.com"},"Action":"sts:AssumeRole"}]}'
aws iam attach-role-policy --role-name rag-gpt-lambda \
  --policy-arn arn:aws:iam::aws:policy/service-role/AWSLambdaBasicExecutionRole
aws iam put-role-policy --role-name rag-gpt-lambda --policy-name s3-sessions \
  --policy-document '{"Version":"2012-10-17","Statement":[{"Effect":"Allow","Action":["s3:GetObject","s3:PutObject","s3:DeleteObject"],"Resource":"arn:aws:s3:::rag-gpt-sessions-SUFFIX/*"}]}'
```

Build and push the first image with `./scripts/deploy.sh` — it will fail at the
last step until the function exists, which is the next command:

```bash
aws lambda create-function --function-name rag-gpt \
  --package-type Image \
  --code ImageUri=ACCOUNT_ID.dkr.ecr.REGION.amazonaws.com/rag-gpt:latest \
  --role arn:aws:iam::ACCOUNT_ID:role/rag-gpt-lambda \
  --architectures arm64 \
  --timeout 60 --memory-size 1536 \
  --environment '{"Variables":{"GROQ_API_KEY":"gsk_...","S3_BUCKET":"rag-gpt-sessions-SUFFIX","ALLOWED_ORIGINS":"https://example.github.io,http://localhost:5173"}}'

aws lambda create-function-url-config --function-name rag-gpt --auth-type NONE

# A public function URL needs both of these; one alone returns 403.
aws lambda add-permission --function-name rag-gpt \
  --statement-id public-url --action lambda:InvokeFunctionUrl \
  --principal '*' --function-url-auth-type NONE
aws lambda add-permission --function-name rag-gpt \
  --statement-id public-invoke --action lambda:InvokeFunction --principal '*'
```

`--environment` is given as JSON rather than the `Variables={...}` shorthand,
which cannot express a value containing a comma — as `ALLOWED_ORIGINS` does.

The URL that comes back is what the front end calls. Every later deploy is just
`./scripts/deploy.sh`.

The image is built for arm64, which is native on an Apple Silicon Mac and
cheaper to run; `ARCH=x86_64 ./scripts/deploy.sh` and a matching
`--architectures` build the other way. The build runs through buildx with
attestations disabled, because Lambda rejects the OCI image index that buildx
otherwise pushes, and on a Python 3.12 base, because the 3.11 image is Amazon
Linux 2 and its glibc is too old for current onnxruntime wheels.

Notes on those settings: the 60-second timeout covers embedding a whole document
on upload; 1536 MB is comfortable once onnxruntime is loaded, and raising memory
also raises CPU, which both shortens cold starts and speeds up embedding. The
model weights are baked into the image so no cold start waits on a download.
CORS is handled in the application rather than in the function URL config, so
`ALLOWED_ORIGINS` is the only place origins are listed. The bucket lifecycle rule
is the backstop for sessions whose visitor never calls `/end_session`.

Cost: within the Lambda free tier at portfolio traffic, a few cents of S3, and
nothing for inference.

## License

MIT — see [LICENSE](LICENSE).

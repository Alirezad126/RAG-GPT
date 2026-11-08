# RAG-GPT

A retrieval-augmented generation service: upload a PDF, then ask questions about
it. Each visitor gets an isolated session, and the session's data is deleted on
request.

Runs as a single AWS Lambda function behind a function URL, with session
embeddings in S3 — no server and no vector database to keep alive. It backs the
live demo on the RAG-GPT page of
[my portfolio](https://github.com/Alirezad126/Alirezad126.github.io).

## How it works

1. **Ingest** — the uploaded PDF is read with `pypdf` and split into
   overlapping 512-character chunks.
2. **Embed** — the chunks are embedded in one OpenAI call. The texts and their
   vectors are written to S3 under the session id.
3. **Retrieve** — a question is first rewritten into a standalone one using the
   chat history, then embedded and compared against the stored vectors by cosine
   similarity; the top `k` chunks become the context.
4. **Answer** — context and history go to the chat model, which is instructed to
   answer only from the document and to say when it does not know.

One PDF is a few hundred chunks, so the index loads in a single request and is
searched with a NumPy dot product. That is why there is no vector database here:
Chroma or pgvector would mean a process to keep running, and at this size they
would not be any faster.

```
app/
├── main.py            FastAPI app and endpoints
├── lambda_handler.py  Mangum adapter — the Lambda entry point
├── config.py          configuration, read from the environment
├── store.py           per-session index, persisted to S3
└── rag/
    ├── pdf.py         PDF parsing and chunking
    ├── retrieval.py   embeddings and similarity search
    └── answer.py      question rewriting and answering
```

## API

| Method | Path           | Body                                        | Returns              |
| ------ | -------------- | ------------------------------------------- | -------------------- |
| `GET`  | `/health`      | —                                           | `{status}`           |
| `POST` | `/upload`      | multipart `file` (PDF), optional `session_id` | `{session_id, chunks}` |
| `POST` | `/chat`        | `{message, conversationState, session_id}`  | `{AIResponse}`       |
| `POST` | `/end_session` | `{session_id}`                              | `{message}`          |

The session id is held by the client and passed explicitly — not in a cookie,
which a browser would treat as third-party and drop, since the API and the
front end are on different domains. Interactive docs are at `/docs` when running
locally.

Uploads are capped at 4 MB: a Lambda function URL accepts 6 MB per request and
binary bodies arrive base64-encoded, which costs about a third of that. For
larger files the upload would have to go straight to S3 with a presigned URL.

## Running locally

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements-dev.txt
cp .env.example .env        # then fill in OPENAI_API_KEY and S3_BUCKET
set -a && source .env && set +a
uvicorn app.main:app --reload
```

S3 is used even locally, so the AWS credentials in your environment need
read/write access to the bucket. Run the tests with `pytest` — they mock S3 and
stub the OpenAI calls, so they need neither.

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
  --timeout 60 --memory-size 1024 \
  --environment 'Variables={OPENAI_API_KEY=sk-...,S3_BUCKET=rag-gpt-sessions-SUFFIX,ALLOWED_ORIGINS=https://alirezad126.github.io}'

aws lambda create-function-url-config --function-name rag-gpt --auth-type NONE
aws lambda add-permission --function-name rag-gpt \
  --statement-id public-url --action lambda:InvokeFunctionUrl \
  --principal '*' --function-url-auth-type NONE
```

The URL that comes back is what the front end calls. Every later deploy is just
`./scripts/deploy.sh`.

Notes on the settings above: the 60-second timeout covers embedding a whole
document on upload; 1024 MB is about the floor once NumPy and the OpenAI client
are loaded, and raising memory also raises CPU, which shortens cold starts. CORS
is handled in the application rather than in the function URL config, so
`ALLOWED_ORIGINS` is the only place the allowed origins are listed. The bucket
lifecycle rule is the backstop for sessions whose visitor never calls
`/end_session`.

Cost: within the Lambda free tier at portfolio traffic, plus a few cents of S3
and whatever OpenAI charges per document and question.

## License

MIT — see [LICENSE](LICENSE).

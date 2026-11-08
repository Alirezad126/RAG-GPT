"""Per-session document index, persisted to S3.

A session is two objects: the chunk texts as JSON and their embedding vectors
as a .npy array, in the same row order. One PDF is a few hundred chunks, so the
whole index loads in one request and is searched with a dot product — no vector
database to run or pay for.
"""

import io
import json

import boto3
import numpy as np
from botocore.exceptions import ClientError

from app.config import S3_BUCKET, S3_PREFIX

_client = None


def client():
    global _client
    if _client is None:
        _client = boto3.client("s3")
    return _client


def _keys(session_id):
    base = f"{S3_PREFIX}/{session_id}"
    return f"{base}/chunks.json", f"{base}/vectors.npy"


def save(session_id, chunks, vectors):
    chunks_key, vectors_key = _keys(session_id)
    buffer = io.BytesIO()
    np.save(buffer, np.asarray(vectors, dtype=np.float32))
    client().put_object(
        Bucket=S3_BUCKET,
        Key=vectors_key,
        Body=buffer.getvalue(),
    )
    client().put_object(
        Bucket=S3_BUCKET,
        Key=chunks_key,
        Body=json.dumps(chunks).encode(),
        ContentType="application/json",
    )


def load(session_id):
    """Return (chunks, vectors), or None if the session has no index."""
    chunks_key, vectors_key = _keys(session_id)
    try:
        chunks = json.loads(
            client().get_object(Bucket=S3_BUCKET, Key=chunks_key)["Body"].read()
        )
        body = client().get_object(Bucket=S3_BUCKET, Key=vectors_key)["Body"].read()
    except ClientError as exc:
        if exc.response["Error"]["Code"] in ("NoSuchKey", "404"):
            return None
        raise
    return chunks, np.load(io.BytesIO(body), allow_pickle=False)


def delete(session_id):
    keys = [{"Key": key} for key in _keys(session_id)]
    client().delete_objects(Bucket=S3_BUCKET, Delete={"Objects": keys})

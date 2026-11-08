# Lambda container image: the deployment package is past the 250 MB zip limit
# once onnxruntime and numpy are in it.
FROM public.ecr.aws/lambda/python:3.11

# The embedding model is baked in rather than downloaded on first use: the task
# root is read-only at runtime, and a cold start should not wait on a download.
ENV FASTEMBED_CACHE_PATH=/opt/fastembed \
    HF_HUB_DISABLE_PROGRESS_BARS=1

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

RUN python -c "from fastembed import TextEmbedding; TextEmbedding('BAAI/bge-small-en-v1.5')"

COPY app ${LAMBDA_TASK_ROOT}/app

CMD ["app.lambda_handler.handler"]

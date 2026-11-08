# Lambda container image: the deployment package is past the 250 MB zip limit
# once onnxruntime and numpy are in it.
# 3.12, not 3.11: the 3.11 image is Amazon Linux 2 with glibc 2.26, too old for
# the manylinux_2_28 wheels that every current onnxruntime build ships as.
FROM public.ecr.aws/lambda/python:3.12

# The embedding model is baked in rather than downloaded on first use: the task
# root is read-only at runtime, and a cold start should not wait on a download.
ENV FASTEMBED_CACHE_PATH=/opt/fastembed \
    HF_HUB_DISABLE_PROGRESS_BARS=1

RUN pip install --no-cache-dir --upgrade pip

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

RUN python -c "from fastembed import TextEmbedding; TextEmbedding('BAAI/bge-small-en-v1.5')"

COPY app ${LAMBDA_TASK_ROOT}/app

CMD ["app.lambda_handler.handler"]

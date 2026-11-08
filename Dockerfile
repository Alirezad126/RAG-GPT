# Lambda container image: the deployment package is far past the 250 MB zip
# limit once openai and numpy are in it.
FROM public.ecr.aws/lambda/python:3.11

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app ${LAMBDA_TASK_ROOT}/app

CMD ["app.lambda_handler.handler"]

#!/usr/bin/env bash
# Build the image, push it to ECR and point the Lambda function at it.
# First-time setup (bucket, repository, role, function, function URL) is in the
# README; this script is for every deploy after that.
set -euo pipefail

: "${AWS_REGION:=us-east-1}"
: "${FUNCTION_NAME:=rag-gpt}"
: "${ECR_REPO:=rag-gpt}"

ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)
REGISTRY="${ACCOUNT_ID}.dkr.ecr.${AWS_REGION}.amazonaws.com"
IMAGE="${REGISTRY}/${ECR_REPO}:latest"

aws ecr get-login-password --region "$AWS_REGION" \
  | docker login --username AWS --password-stdin "$REGISTRY"

# Lambda runs on x86_64 unless the function is configured for arm64.
docker build --platform linux/amd64 -t "$IMAGE" .
docker push "$IMAGE"

aws lambda update-function-code \
  --function-name "$FUNCTION_NAME" \
  --image-uri "$IMAGE" \
  --region "$AWS_REGION" \
  --no-cli-pager >/dev/null

aws lambda wait function-updated --function-name "$FUNCTION_NAME" --region "$AWS_REGION"
echo "Deployed ${IMAGE} to ${FUNCTION_NAME}."

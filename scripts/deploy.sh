#!/usr/bin/env bash
# Build the image, push it to ECR and point the Lambda function at it.
# First-time setup (bucket, repository, role, function, function URL) is in the
# README; this script is for every deploy after that.
set -euo pipefail

: "${AWS_REGION:=us-east-1}"
: "${FUNCTION_NAME:=rag-gpt}"
: "${ECR_REPO:=rag-gpt}"
# arm64 by default: it builds natively on an Apple Silicon Mac, and Lambda
# bills Graviton at a lower rate. The function must be created to match.
: "${ARCH:=arm64}"

ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)
REGISTRY="${ACCOUNT_ID}.dkr.ecr.${AWS_REGION}.amazonaws.com"
IMAGE="${REGISTRY}/${ECR_REPO}:latest"

aws ecr get-login-password --region "$AWS_REGION" \
  | docker login --username AWS --password-stdin "$REGISTRY"

# --provenance/--sbom off: with them, buildx pushes an OCI image index, and
# Lambda only accepts a Docker v2 schema 2 manifest.
docker buildx build --platform "linux/${ARCH}" \
  --provenance=false --sbom=false \
  -t "$IMAGE" --push .

aws lambda update-function-code \
  --function-name "$FUNCTION_NAME" \
  --image-uri "$IMAGE" \
  --region "$AWS_REGION" \
  --no-cli-pager >/dev/null

aws lambda wait function-updated --function-name "$FUNCTION_NAME" --region "$AWS_REGION"
echo "Deployed ${IMAGE} to ${FUNCTION_NAME}."

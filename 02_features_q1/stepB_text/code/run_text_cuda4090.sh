#!/usr/bin/env bash
set -euo pipefail

WORKSPACE="${1:-/path/to/E_workspace}"
MODEL_PATH="${2:-bert-base-uncased}"
BATCH_SIZE="${BATCH_SIZE:-32}"
MAX_LENGTH="${MAX_LENGTH:-256}"
export CUBLAS_WORKSPACE_CONFIG="${CUBLAS_WORKSPACE_CONFIG:-:4096:8}"

python "$(dirname "$0")/extract_text_features.py" \
  --workspace "$WORKSPACE" \
  --device cuda:0 \
  --model-path "$MODEL_PATH" \
  --batch-size "$BATCH_SIZE" \
  --max-length "$MAX_LENGTH"

python "$(dirname "$0")/validate_text_features.py" \
  --workspace "$WORKSPACE" \
  --model-name bert-base-uncased

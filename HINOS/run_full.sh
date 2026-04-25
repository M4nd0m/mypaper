#!/bin/bash
set -euo pipefail

DATASETS=(${DATASETS:-school dblp brain patent arXivAI arXivCS})
EPOCHS="${EPOCHS:-100}"
LAMBDA_COMMUNITY="${LAMBDA_COMMUNITY:-0.1}"
PYTHON="${PYTHON:-python}"
RUN_EVAL="${RUN_EVAL:-1}"

for ds in "${DATASETS[@]}"; do
  echo "Running dataset=${ds}, offline full-graph NCut, lambda=${LAMBDA_COMMUNITY}"
  "$PYTHON" main.py \
    --dataset "$ds" \
    --epoch "$EPOCHS" \
    --lambda_community "$LAMBDA_COMMUNITY" \
    --num_clusters 0

  if [[ "$RUN_EVAL" == "1" ]]; then
    echo "Evaluating dataset=${ds}, epoch=${EPOCHS}"
    "$PYTHON" evaluate.py \
      --dataset "$ds" \
      --epoch "$EPOCHS"
  fi
done

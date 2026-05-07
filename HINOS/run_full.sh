#!/bin/bash
set -euo pipefail

PYTHON="${PYTHON:-python}"
LOG_DIR="${LOG_DIR:-logs/full}"
RUN_PREFIX="${RUN_PREFIX:-search_proto}"
RUN_EVAL="${RUN_EVAL:-0}"
EPOCH_OVERRIDE="${EPOCH:-}"
export KMP_DUPLICATE_LIB_OK="${KMP_DUPLICATE_LIB_OK:-TRUE}"

# Data4TGC-style datasets. The three very large arXiv variants are excluded
# here: arxivLarge, arxivMath, and arxivPhy.
DATASETS=(${DATASETS:-school dblp brain patent arXivAI arXivCS})

mkdir -p "$LOG_DIR"
LOG_FILE="${LOG_FILE:-${LOG_DIR}/${RUN_PREFIX}_$(date +%Y%m%d_%H%M%S).log}"

echo "[Full] datasets: ${DATASETS[*]}" | tee -a "$LOG_FILE"
echo "[Full] excluded: arxivLarge arxivMath arxivPhy" | tee -a "$LOG_FILE"
echo "[Full] log: $LOG_FILE" | tee -a "$LOG_FILE"

dataset_config() {
  local ds="$1"
  case "$ds" in
    school)
      EPOCH_DEFAULT=40
      TPPR_K=5
      TAPS_BETA=0.7
      EVAL_INTERVAL=5
      ;;
    dblp)
      EPOCH_DEFAULT=40
      TPPR_K=5
      TAPS_BETA=0.1
      EVAL_INTERVAL=5
      ;;
    brain)
      EPOCH_DEFAULT=40
      TPPR_K=5
      TAPS_BETA=0.2
      EVAL_INTERVAL=5
      ;;
    patent)
      EPOCH_DEFAULT=40
      TPPR_K=5
      TAPS_BETA=0.2
      EVAL_INTERVAL=5
      ;;
    arXivAI)
      EPOCH_DEFAULT=25
      TPPR_K=5
      TAPS_BETA=0.02
      EVAL_INTERVAL=5
      ;;
    arXivCS)
      EPOCH_DEFAULT=20
      TPPR_K=5
      TAPS_BETA=0.005
      EVAL_INTERVAL=5
      ;;
    *)
      echo "[Full] unknown dataset '$ds'. Add it to dataset_config first." >&2
      exit 1
      ;;
  esac
  EPOCH="${EPOCH_OVERRIDE:-$EPOCH_DEFAULT}"
}

run_one() {
  local ds="$1"
  dataset_config "$ds"
  local tag="${RUN_PREFIX}_${ds}_proto_k${TPPR_K}_b${TAPS_BETA}_e${EPOCH}"

  {
    echo "============================================================"
    echo "[Full] start=$(date '+%Y-%m-%d %H:%M:%S') dataset=${ds}"
    echo "[Full] epoch=${EPOCH} tppr_K=${TPPR_K} taps_beta=${TAPS_BETA}"
    echo "[Full] objective=search_proto assignment=prototype lambda_community=${LAMBDA_COMMUNITY:-0.1} lambda_ncut_orth=${LAMBDA_NCUT_ORTH:-100} kl=off"
    echo "[Full] run_tag=${tag}"
    echo "============================================================"

    "$PYTHON" main.py \
      --dataset "$ds" \
      --objective_mode search_proto \
      --epoch "$EPOCH" \
      --prototype_alpha 1.0 \
      --prototype_lr_scale 0.01 \
      --lambda_community "${LAMBDA_COMMUNITY:-0.1}" \
      --lambda_ncut_orth "${LAMBDA_NCUT_ORTH:-100}" \
      --tppr_K "$TPPR_K" \
      --taps_budget_mode nlogn \
      --taps_budget_beta "$TAPS_BETA" \
      --eval_interval "$EVAL_INTERVAL" \
      --main_pred_mode argmax_s \
      --run_tag "$tag"

    "$PYTHON" summarize_phase0.py \
      --dataset "$ds" \
      --metrics_csv "emb/${ds}/${ds}_TGC_${EPOCH}_${tag}_metrics.csv"

    if [[ "$RUN_EVAL" == "1" ]]; then
      "$PYTHON" evaluate.py \
        --dataset "$ds" \
        --pred_path "emb/${ds}/${ds}_TGC_${EPOCH}_${tag}_pred.txt"
    fi

    echo "[Full] finished=$(date '+%Y-%m-%d %H:%M:%S') dataset=${ds}"
    echo
  } 2>&1 | tee -a "$LOG_FILE"
}

for ds in "${DATASETS[@]}"; do
  case "$ds" in
    arxivLarge|arxivMath|arxivPhy)
      echo "[Full] refusing excluded large dataset: $ds" | tee -a "$LOG_FILE"
      exit 1
      ;;
  esac
  run_one "$ds"
done

echo "[Full] all runs finished. log: $LOG_FILE" | tee -a "$LOG_FILE"

#!/bin/bash
set -euo pipefail

PYTHON="${PYTHON:-python}"
LOG_DIR="${LOG_DIR:-logs/full}"
PHASE="${PHASE:-0}"
RUN_PREFIX="${RUN_PREFIX:-phase${PHASE}_school_phase01}"
RUN_EVAL="${RUN_EVAL:-0}"
EPOCH_OVERRIDE="${EPOCH:-}"
export KMP_DUPLICATE_LIB_OK="${KMP_DUPLICATE_LIB_OK:-TRUE}"

# Data4TGC-style datasets used for Phase 0/1. The three very large arXiv
# variants are excluded here: arxivLarge, arxivMath, and arxivPhy.
DATASETS=(${DATASETS:-school dblp brain patent arXivAI arXivCS})

mkdir -p "$LOG_DIR"
LOG_FILE="${LOG_FILE:-${LOG_DIR}/${RUN_PREFIX}_$(date +%Y%m%d_%H%M%S).log}"

echo "[Full] datasets: ${DATASETS[*]}" | tee -a "$LOG_FILE"
echo "[Full] phase: $PHASE" | tee -a "$LOG_FILE"
echo "[Full] excluded: arxivLarge arxivMath arxivPhy" | tee -a "$LOG_FILE"
echo "[Full] unified log: $LOG_FILE" | tee -a "$LOG_FILE"

dataset_config() {
  local ds="$1"
  case "$ds" in
    school)
      EPOCH_DEFAULT=40
      WARMUP=5
      RAMP=15
      TPPR_K=5
      TAPS_BETA=0.5
      EVAL_INTERVAL=5
      ;;
    dblp)
      EPOCH_DEFAULT=40
      WARMUP=10
      RAMP=30
      TPPR_K=3
      TAPS_BETA=0.3
      EVAL_INTERVAL=5
      ;;
    brain)
      EPOCH_DEFAULT=40
      WARMUP=8
      RAMP=25
      TPPR_K=5
      TAPS_BETA=0.1
      EVAL_INTERVAL=5
      ;;
    patent)
      EPOCH_DEFAULT=40
      WARMUP=8
      RAMP=25
      TPPR_K=5
      TAPS_BETA=0.2
      EVAL_INTERVAL=5
      ;;
    arXivAI)
      EPOCH_DEFAULT=25
      WARMUP=5
      RAMP=15
      TPPR_K=5
      TAPS_BETA=0.05
      EVAL_INTERVAL=5
      ;;
    arXivCS)
      EPOCH_DEFAULT=20
      WARMUP=5
      RAMP=10
      TPPR_K=5
      TAPS_BETA=0.02
      EVAL_INTERVAL=5
      ;;
    *)
      echo "[Full] unknown dataset '$ds'. Add it to dataset_config first." >&2
      exit 1
      ;;
  esac
  EPOCH="${EPOCH_OVERRIDE:-$EPOCH_DEFAULT}"
}

phase_args() {
  case "$PHASE" in
    0)
      echo "--unified_mode off"
      ;;
    1)
      echo "--unified_mode on --rho_anchor ${RHO_ANCHOR:-1.0} --U_init_mode ${U_INIT_MODE:-log_pi} --log_unified_align 1"
      ;;
    *)
      echo "[Full] unsupported PHASE='$PHASE'; use 0 or 1 for this plan." >&2
      exit 1
      ;;
  esac
}

run_one() {
  local ds="$1"
  dataset_config "$ds"
  local tag="${RUN_PREFIX}_${ds}_k${TPPR_K}_b${TAPS_BETA}_e${EPOCH}"
  local extra_args
  extra_args="$(phase_args)"

  {
    echo "============================================================"
    echo "[Full] start=$(date '+%Y-%m-%d %H:%M:%S') dataset=${ds}"
    echo "[Full] epoch=${EPOCH} warmup=${WARMUP} ramp=${RAMP} tppr_K=${TPPR_K} taps_beta=${TAPS_BETA}"
    echo "[Full] run_tag=${tag}"
    echo "============================================================"

    "$PYTHON" main.py \
      --dataset "$ds" \
      --objective_mode cut_main \
      --epoch "$EPOCH" \
      --assign_mode prototype \
      --prototype_alpha 1.0 \
      --prototype_lr_scale 0.01 \
      --lambda_temp 0.01 \
      --lambda_batch 0.01 \
      --lambda_com 1.0 \
      --rho_cut 1.0 \
      --rho_kl 1.0 \
      --rho_bal 0.1 \
      --tppr_K "$TPPR_K" \
      --taps_budget_mode nlogn \
      --taps_budget_beta "$TAPS_BETA" \
      --target_update_interval 5 \
      --kl_target_mode dynamic_tgc \
      --balance_mode hinos \
      --batch_recon_mode ones \
      --warmup_epochs "$WARMUP" \
      --com_ramp_epochs "$RAMP" \
      --eval_interval "$EVAL_INTERVAL" \
      --main_pred_mode argmax_s \
      --run_tag "$tag" \
      $extra_args

    if [[ "$PHASE" == "0" ]]; then
      "$PYTHON" summarize_phase0.py \
        --dataset "$ds" \
        --metrics_csv "emb/${ds}/${ds}_TGC_${EPOCH}_${tag}_metrics.csv"
    fi

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

echo "[Full] all runs finished. unified log: $LOG_FILE" | tee -a "$LOG_FILE"

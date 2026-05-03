#!/bin/bash
set -euo pipefail

PYTHON="${PYTHON:-python}"

EPOCH="${EPOCH:-40}"
WARMUP_EPOCHS="${WARMUP_EPOCHS:-10}"
COM_RAMP_EPOCHS="${COM_RAMP_EPOCHS:-30}"
LOG_DIR="${LOG_DIR:-logs/dblp}"
RUN_PREFIX="${RUN_PREFIX:-dblp_raw_tppr_cut_k3_beta03_e${EPOCH}}"
LOG_FILE="${LOG_FILE:-${LOG_DIR}/${RUN_PREFIX}_$(date +%Y%m%d_%H%M%S).log}"

mkdir -p "$LOG_DIR"

echo "[DBLP] unified log: $LOG_FILE" | tee -a "$LOG_FILE"

run_dblp() {
  local tag="$1"

  {
    echo "============================================================"
    echo "[DBLP] start=$(date '+%Y-%m-%d %H:%M:%S')"
    echo "[DBLP] raw TPPR-Cut tag=${tag}"
    echo "============================================================"

    "$PYTHON" main.py \
      --dataset dblp \
      --objective_mode cut_main \
      --epoch "$EPOCH" \
      --assign_mode prototype \
      --prototype_alpha 1.0 \
      --prototype_lr_scale 0.01 \
      --lambda_temp 0.01 \
      --lambda_batch 0.01 \
      --lambda_com 1.0 \
      --rho_cut 0.1 \
      --rho_kl 10.0 \
      --rho_bal 5.0 \
      --tppr_K 3 \
      --taps_budget_mode nlogn \
      --taps_budget_beta 0.3 \
      --target_update_interval 5 \
      --kl_target_mode dynamic_tgc \
      --balance_mode hinos \
      --batch_recon_mode ones \
      --warmup_epochs "$WARMUP_EPOCHS" \
      --com_ramp_epochs "$COM_RAMP_EPOCHS" \
      --eval_interval 5 \
      --main_pred_mode argmax_s \
      --run_tag "$tag"

    echo "[DBLP] finished=$(date '+%Y-%m-%d %H:%M:%S') tag=${tag}"
    echo
  } 2>&1 | tee -a "$LOG_FILE"
}

run_dblp "${RUN_PREFIX}_raw_ncut"

echo "[DBLP] all runs finished. unified log: $LOG_FILE" | tee -a "$LOG_FILE"
echo "[DBLP] metrics files are under emb/dblp with prefix: dblp_TGC_${EPOCH}_${RUN_PREFIX}_"

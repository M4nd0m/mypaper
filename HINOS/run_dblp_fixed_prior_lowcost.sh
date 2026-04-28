#!/bin/bash
set -euo pipefail

PYTHON="${PYTHON:-python}"

"$PYTHON" main.py \
  --dataset dblp \
  --objective_mode cut_main \
  --epoch 100 \
  --assign_mode prototype \
  --freeze_prototypes \
  --prototype_alpha 1.0 \
  --lambda_temp 0.01 \
  --lambda_batch 0.01 \
  --lambda_com 0.2 \
  --rho_kl 25 \
  --rho_bal 0.0 \
  --kl_target_mode fixed_initial \
  --balance_mode none \
  --batch_recon_mode ones \
  --warmup_epochs 20 \
  --com_ramp_epochs 50 \
  --eval_interval 5 \
  --main_pred_mode argmax_s \
  --taps_budget_mode nlogn \
  --taps_budget_beta 0.1 \
  --run_tag dblp_fixed_prior_freeze_proto_b01_lc02_rho25_w20r50_e100

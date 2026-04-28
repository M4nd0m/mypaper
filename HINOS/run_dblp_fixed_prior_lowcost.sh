#!/bin/bash
set -euo pipefail

PYTHON="${PYTHON:-python}"

"$PYTHON" main.py \
  --dataset dblp \
  --objective_mode cut_main \
  --epoch 100 \
  --assign_mode prototype \
  --prototype_alpha 1.0 \
  --prototype_lr_scale 0.01 \
  --lambda_temp 0.01 \
  --lambda_batch 0.01 \
  --lambda_com 1.0 \
  --rho_cut 1.0 \
  --rho_kl 5.0 \
  --rho_bal 5.0 \
  --target_update_interval 5 \
  --kl_target_mode dynamic_tgc \
  --balance_mode hinos \
  --batch_recon_mode ones \
  --warmup_epochs 20 \
  --com_ramp_epochs 50 \
  --eval_interval 5 \
  --main_pred_mode argmax_s \
  --taps_budget_mode nlogn \
  --taps_budget_beta 0.1 \
  --run_tag dblp_dtgc_batchkl_hinos_bal_lc1_rkl5_rbal5_plr001_e100

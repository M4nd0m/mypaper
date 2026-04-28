# HINOS Method Notes

## Objective

The project now targets temporal community discovery by adapting the HINOS offline stage. The training loss still has three top-level terms:

```text
L = lambda_temp * L_temp
  + lambda_batch * L_batch
  + lambda_com(e) / B * L_com
```

`L_com` contains the community-discovery structure:

```text
L_com =
  rho_cut * L_TPPR-Cut
  + rho_kl * L_TGC-KL
  + rho_bal * L_HINOS-Bal
```

The split is intentional:

- `L_TPPR-Cut`: HINOS higher-order temporal community regularization on the TPPR-induced graph.
- `L_TGC-KL`: TGC-style Student-t assignment alignment with a detached sharpened target.
- `L_HINOS-Bal`: HINOS TPPR-volume balance/sharpness penalty that prevents collapsed or overly diffuse assignments.

`loss_assign_penalty` in metrics is the TGC KL term. `loss_hinos_bal` is the HINOS penalty term. Legacy `rho_assign` and `lambda_bal` remain only for compatibility with old commands.

## Assignment

For the main method, use:

```text
assign_mode=prototype
main_pred_mode=argmax_s
kl_target_mode=dynamic_tgc
balance_mode=hinos
```

The soft assignment is the Student-t distribution from TGC:

```text
Q_ik = (1 + ||z_i - c_k||^2 / nu)^(-(nu + 1) / 2)
       / sum_l (1 + ||z_i - c_l||^2 / nu)^(-(nu + 1) / 2)
```

Prototype centers are initialized by KMeans over pretrained node2vec features. They remain learnable, but use a smaller learning rate through `--prototype_lr_scale` to reduce prototype drift.

The final community label is:

```text
y_i = argmax_k Q_ik
```

## Dynamic TGC KL

The target distribution is refreshed every `--target_update_interval` epochs:

```text
f_k = sum_i Q_ik
P_ik = (Q_ik^2 / (f_k + eps)) / sum_l (Q_il^2 / (f_l + eps))
```

`P` is detached before KL is evaluated:

```text
L_TGC-KL = sum_i sum_k P_ik * log((P_ik + eps) / (Q_ik + eps))
```

`kl_target_mode=fixed_initial` is kept only for ablation of the previous fixed-prior experiment.

## HINOS Balance

The HINOS penalty is implemented directly from the original formula:

```text
L_p = 1 / (sqrt(K) - 1)
      * (sqrt(K)
         - 1 / sqrt(2m_pi) * sum_j ||s_j o d_pi^(1/2)||_2)
```

This term is query-agnostic. It depends only on the full-graph assignment, TPPR degree, and TPPR volume, so it is suitable for community discovery. It encourages assignments that are both TPPR-volume balanced and sharp enough for `argmax_s`.

## Batch Reconstruction

`batch_recon_mode=ones` remains the recommended default. Pseudo-label reconstruction modes are kept for ablations:

- `ones`: all observed source-target and source-history pairs have target `1`.
- `soft_pseudo`: target is `stopgrad(S_u S_v^T)`.
- `hard_pseudo`: target is `1[argmax S_u == argmax S_v]`.
- `hard_pseudo_gate`: reconstructs only high-confidence same-pseudo observed pairs.

## Server Commands

Run experiments on the GPU server, not on the local workstation.

Recommended DBLP:

```bash
cd HINOS
python main.py \
  --dataset dblp \
  --objective_mode cut_main \
  --assign_mode prototype \
  --epoch 100 \
  --lambda_temp 0.01 \
  --lambda_batch 0.01 \
  --lambda_com 0.2 \
  --rho_cut 1.0 \
  --rho_kl 1.0 \
  --rho_bal 0.1 \
  --prototype_alpha 1.0 \
  --prototype_lr_scale 0.1 \
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
  --run_tag tgc_student_hinos_bal_tppr_cut
```

Recommended School:

```bash
cd HINOS
python main.py \
  --dataset school \
  --objective_mode cut_main \
  --assign_mode prototype \
  --epoch 50 \
  --lambda_temp 0.005 \
  --lambda_batch 0.005 \
  --lambda_com 0.5 \
  --rho_cut 1.0 \
  --rho_kl 1.0 \
  --rho_bal 0.1 \
  --prototype_alpha 1.0 \
  --prototype_lr_scale 0.1 \
  --target_update_interval 5 \
  --kl_target_mode dynamic_tgc \
  --balance_mode hinos \
  --batch_recon_mode ones \
  --warmup_epochs 5 \
  --com_ramp_epochs 10 \
  --eval_interval 5 \
  --main_pred_mode argmax_s \
  --taps_budget_mode nlogn \
  --taps_budget_beta 0.5 \
  --run_tag tgc_student_hinos_bal_tppr_cut
```

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

`L_TPPR-Cut` uses the raw symmetrized TPPR graph. The previous full-null degree-corrected trace variant was removed because, under row-stochastic \(Q\), its correction reduces to the constant shift \(\gamma(1-K)\) and does not change the optimization direction. The later edge-level residual variant was also removed after ablation because it changed cut values but did not materially change NMI or assignment entropy. See `docs/degree_aware_tppr_cut.md`.

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

## DTGC Batch-Level KL

For `kl_target_mode=dynamic_tgc`, the KL term follows the DTGC/TGC batch-level node distribution. For the current source-node batch \(B\), the detached target is built from fixed pretrained features \(Z^0\):

```text
q0 = StudentT(Z0_B, C)
f_k = sum_{i in B} q0_ik
P_ik = (q0_ik^2 / (f_k + eps)) / sum_l (q0_il^2 / (f_l + eps))
```

The live distribution is computed from the current trainable embedding \(Z^t\):

```text
q_prime = StudentT(Zt_B, C)
L_TGC-KL = KL(P || q_prime)
```

The implementation uses `F.kl_div(log(q_prime), P.detach(), reduction="batchmean")`. `target_update_interval` is kept only for command compatibility and does not control the main `dynamic_tgc` path. `kl_target_mode=fixed_initial` is kept only as an ablation of the previous fixed-prior experiment.

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
  --epoch 40 \
  --lambda_temp 0.01 \
  --lambda_batch 0.01 \
  --lambda_com 1.0 \
  --rho_cut 0.1 \
  --rho_kl 10.0 \
  --rho_bal 5.0 \
  --prototype_alpha 1.0 \
  --prototype_lr_scale 0.01 \
  --target_update_interval 5 \
  --kl_target_mode dynamic_tgc \
  --balance_mode hinos \
  --batch_recon_mode ones \
  --warmup_epochs 10 \
  --com_ramp_epochs 20 \
  --eval_interval 5 \
  --main_pred_mode argmax_s \
  --taps_budget_mode nlogn \
  --taps_budget_beta 0.1 \
  --run_tag dblp_tppr_diag_weakcut_rkl10_rbal5_e40
```

The metrics CSV also records static TPPR-label diagnostics when labels are available:

```text
purity_at_5_pi, purity_at_10_pi, purity_at_20_pi
leakage_at_5_pi, leakage_at_10_pi, leakage_at_20_pi
ncut_gt_pi
```

For faster TPPR/TAPS parameter inspection without training, use:

```bash
python diagnose_tppr.py --dataset dblp --tppr_K 3 --taps_budget_beta 0.05 --spectral_topk 20
```

Treat `tppr_K=1` only as a first-order lower-bound diagnostic. Main candidates should remain moderate-order TPPR settings such as `tppr_K` in `{2, 3, 4, 5}`.

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
  --run_tag school_dtgc_batchkl_hinos_bal_e50
```

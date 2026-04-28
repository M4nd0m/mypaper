# DBLP Experiment Log

This note records the recent DBLP experiments around DTGC Student-t/KL, HINOS TPPR-Cut, HINOS balance, and TPPR/TAPS diagnostics.

## Current Loss Design

The current community objective keeps the three existing terms:

\[
\mathcal{L}_{com}
=
\rho_{cut}\mathcal{L}_{TPPR-Cut}
+
\rho_{kl}\mathcal{L}_{DTGC-KL}
+
\rho_{bal}\mathcal{L}_{HINOS-Bal}.
\]

No extra sharpness loss is used. Final clustering is still:

\[
\hat y_i=\arg\max_k Q_{ik}.
\]

## Training Runs

| Run | Epoch | Main setting | Best `nmi_argmax_s` | Final `nmi_argmax_s` | Final cut | Final KL | Final balance | Final entropy | Note |
|---|---:|---|---:|---:|---:|---:|---:|---:|---|
| `dblp_TGC_40_dblp_tgc_hinos_bal_b01_e40_metrics.csv` | 40 | early dynamic target cache | 0.3552 @ 10 | 0.3499 | 3.0538 | 6737.1634 | 0.4189 | 1.2218 | KL became invalid/unstable; early negative KL then huge KL. |
| `dblp_TGC_40_dblp_dtgc_batchkl_hinos_bal_e40_metrics.csv` | 40 | DTGC batch KL, `rho_cut=1`, `rho_kl=1`, `rho_bal=0.1` | 0.3552 @ 10 | 0.3248 | 1.2435 | 0.0061 | 0.9902 | 2.2834 | KL fixed, but TPPR-Cut optimization reduced NMI. |
| `dblp_TGC_100_dblp_dtgc_batchkl_hinos_bal_lc1_rkl5_rbal5_plr001_e100_metrics.csv` | 100 | stronger community, `rho_cut=1`, `rho_kl=5`, `rho_bal=5` | 0.3569 @ 20 | 0.3175 | 0.3657 | 0.0186 | 0.9815 | 2.2633 | TPPR-Cut decreased strongly, but `argmax(Q)` NMI kept dropping. |
| `dblp_TGC_100_dblp_fixed_prior_freeze_proto_b01_lc02_rho25_w20r50_e100_metrics.csv` | 100 | fixed prior / frozen prototype ablation | 0.3569 @ 20 | 0.3315 | 1.0404 | 0.0164 | n/a | 2.2582 | Better than strong TPPR-Cut run at final epoch, but still below warmup. |

Important observation:

\[
\mathcal{L}_{TPPR-Cut}\downarrow
\quad\text{while}\quad
\mathrm{NMI}_{argmax(Q)}\downarrow.
\]

This suggests the raw TPPR affinity is not fully aligned with DBLP label communities.

## TPPR Label Diagnostics

The following diagnostics are static graph diagnostics. They do not participate in training.

All three diagnostics are computed on the same graph used by TPPR-Cut:

\[
W_\Pi=\frac{1}{2}(\Pi+\Pi^\top),\quad (W_\Pi)_{ii}=0.
\]

Therefore these numbers directly describe the structural signal seen by the cut loss.

### Purity@K

For each node \(i\), let \(\mathcal{N}_K^\Pi(i)\) be the top-\(K\) neighbors of \(i\) ranked by \(W_{\Pi,ij}\). Self-loops are excluded.

\[
\mathrm{Purity@K}(\Pi,y)
=
\frac{1}{N}
\sum_i
\frac{1}{K_i}
\sum_{j\in\mathcal{N}_K^\Pi(i)}
\mathbf{1}[y_j=y_i].
\]

Meaning:

- High Purity@K means TPPR nearest neighbors are mostly within the same ground-truth label.
- Low Purity@K means local TPPR neighborhoods contain many cross-label nodes.
- For TPPR-Cut, low Purity@K is risky because the cut loss encourages high-weight TPPR neighbors to receive similar assignments.

### Leakage@K

\[
\mathrm{Leakage@K}(\Pi,y)=1-\mathrm{Purity@K}(\Pi,y).
\]

Meaning:

- Leakage@K directly measures cross-label contamination in TPPR top-\(K\) neighborhoods.
- High Leakage@K means TPPR contains many edges that may pull different label communities together.
- In the current DBLP experiments, Leakage@10 around 0.50 means about half of each node's top-10 TPPR neighbors are cross-label.

### Ground-Truth Ncut on TPPR

Let \(Y\in\{0,1\}^{N\times C}\) be the one-hot ground-truth label matrix. The diagnostic puts the true labels into the current TPPR cut graph:

\[
\mathrm{Ncut}_\Pi(y)
=
\operatorname{Tr}
\left[
(Y^\top D_\Pi Y+\epsilon I)^{-1}
Y^\top L_\Pi Y
\right].
\]

Meaning:

- Low \(\mathrm{Ncut}_\Pi(y)\) means the ground-truth labels form a natural low-cut partition on \(W_\Pi\).
- High \(\mathrm{Ncut}_\Pi(y)\) means the ground-truth labels cut through many high-weight TPPR edges.
- If \(\mathrm{Ncut}_\Pi(y)\) is high, aggressively minimizing TPPR-Cut may move \(Q\) away from label communities.

### Spectral NMI Diagnostics

`nmi_spectral_pi` runs spectral clustering on the raw symmetrized TPPR affinity. `nmi_spectral_topk_pi` first applies row top-k sparsification inside the spectral diagnostic.

Meaning:

- If spectral NMI is low but Purity@K is high, TPPR may have useful local neighborhoods but poor global partition structure.
- If both spectral NMI and Purity@K are low, raw TPPR is not aligned with label communities locally or globally.
- These metrics are diagnostic only; they are not used as training targets.

## TPPR/TAPS Grid Results

| `tppr_K` | `taps_beta` | Density | Purity@5 | Purity@10 | Purity@20 | Leakage@10 | `ncut_gt_pi` | `nmi_spectral_pi` | `nmi_spectral_topk_pi` | Comment |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 5 | 0.1 | n/a | n/a | 0.3936 | n/a | 0.6064 | 4.5974 | ~0.0054 | n/a | Original diagnostic from training startup. |
| 1 | 0.2 | 0.000970 | 0.5353 | 0.5095 | 0.4906 | 0.4905 | 4.2682 | 0.0066 | 0.0060 | First-order lower bound; best purity/Ncut so far but weakens high-order TPPR narrative. |
| 2 | 0.05 | 0.000237 | 0.3022 | 0.2926 | 0.2868 | 0.7074 | 4.2394 | 0.0039 | 0.0044 | Too sparse/noisy; local purity poor. |
| 2 | 0.2 | 0.006652 | 0.5314 | 0.4987 | 0.4668 | 0.5013 | 4.3834 | 0.0063 | 0.0065 | Best efficient candidate: good purity with low density and lower ground-truth Ncut. |
| 3 | 0.05 | 0.000529 | 0.3012 | 0.2886 | 0.2788 | 0.7114 | n/a | n/a | n/a | Increasing K at low beta did not improve purity. |
| 3 | 0.1 | 0.005069 | 0.4212 | 0.3949 | 0.3706 | 0.6051 | 4.4611 | 0.0050 | 0.0062 | Budget increase improves local TPPR quality. |
| 3 | 0.2 | 0.037210 | 0.5311 | 0.4965 | 0.4611 | 0.5035 | 4.5459 | 0.0069 | 0.0069 | Moderate-order candidate; similar purity to K=2 but denser. |
| 3 | 0.3 | 0.095887 | 0.5808 | 0.5435 | 0.5021 | 0.4565 | 4.5894 | 0.0075 | 0.0071 | Best local purity, but graph is much denser. |
| 4 | 0.2 | 0.137544 | 0.5314 | 0.4967 | 0.4598 | 0.5033 | 4.6394 | 0.0059 | 0.0073 | No purity gain over K=3, much denser. |
| 5 | 0.2 | 0.312156 | 0.5316 | 0.4968 | 0.4598 | 0.5032 | 4.7406 | 0.0063 | 0.0060 | No useful purity gain; density becomes very high. |

## Interpretation

1. Low TAPS budget is harmful on DBLP.

   At `tppr_K=3`, increasing `taps_beta` from `0.05` to `0.2` raises:

   \[
   \mathrm{Purity@10}: 0.2886 \rightarrow 0.4965.
   \]

   This indicates that the earlier low-budget TPPR graph was under-sampled and noisy.

2. TAPS budget is the dominant factor in the current grid.

   At low budget (`taps_beta=0.05`), both `tppr_K=2` and `tppr_K=3` have poor local purity:

   \[
   \mathrm{Purity@10}\approx 0.29.
   \]

   At `taps_beta=0.2`, the same family of TPPR graphs improves to around:

   \[
   \mathrm{Purity@10}\approx 0.50.
   \]

   This suggests the TAPS sampling budget controls whether the TPPR graph is reliable enough for neighborhood diagnostics.

3. TPPR truncation order mainly affects density/cost after the budget is sufficient.

   At `taps_beta=0.2`, increasing `tppr_K` from `3` to `5` barely changes local purity:

   \[
   \mathrm{Purity@10}: 0.4965 \rightarrow 0.4968,
   \]

   but density increases sharply:

   \[
   0.0372 \rightarrow 0.3122.
   \]

   This suggests over-propagation/densification without useful label-neighborhood gain.

4. `tppr_K=1, taps_beta=0.2` is the best diagnostic lower bound, but not ideal as the main TPPR setting.

   It gives the best local purity and lowest ground-truth Ncut so far:

   \[
   \mathrm{Purity@10}=0.5095,\quad
   \mathrm{Ncut}_{\Pi}(y)=4.2682.
   \]

   However, `tppr_K=1` mostly reflects first-order temporal proximity. It is useful as a lower-bound diagnostic, but using it as the main method weakens the high-order TPPR contribution.

5. `tppr_K=2, taps_beta=0.2` is the best efficient high-order configuration so far.

   Compared with `tppr_K=3, taps_beta=0.2`, it gives nearly the same local purity:

   \[
   \mathrm{Purity@10}: 0.4987 \text{ vs. } 0.4965,
   \]

   but with much lower density:

   \[
   0.0067 \text{ vs. } 0.0372.
   \]

   It also has lower ground-truth Ncut:

   \[
   4.3834 \text{ vs. } 4.5459.
   \]

   This makes it a strong efficient candidate, although `tppr_K=3` remains more attractive if the main paper narrative emphasizes moderate-order TPPR.

6. Raw TPPR is still not strongly aligned with DBLP labels.

   Even the better configurations have:

   \[
   \mathrm{Leakage@10}\approx 0.50.
   \]

   This explains why strong TPPR-Cut can reduce cut loss while hurting `argmax(Q)` NMI.

## Current Recommendation

Use two DBLP candidates depending on the priority:

Efficient candidate:

```text
tppr_K = 2
taps_budget_beta = 0.2
rho_cut = 0.1
rho_kl = 10.0
rho_bal = 5.0
epoch = 40
warmup_epochs = 10
com_ramp_epochs = 20
```

Moderate-order candidate:

```text
tppr_K = 3
taps_budget_beta = 0.2
rho_cut = 0.1
rho_kl = 10.0
rho_bal = 5.0
epoch = 40
warmup_epochs = 10
com_ramp_epochs = 20
```

Rationale:

- `tppr_K=2, taps_beta=0.2` has the best cost-quality tradeoff so far.
- `tppr_K=3, taps_beta=0.2` keeps a stronger moderate-order TPPR story with similar local purity but higher graph density.
- `taps_beta=0.2` substantially improves TPPR local purity while keeping graph density much lower than `beta=0.3` or `K>=4`.
- `rho_cut=0.1` keeps TPPR-Cut in the objective but avoids letting a still-leaky TPPR graph dominate clustering.

If only one DBLP run can be done, use `tppr_K=2, taps_beta=0.2` first. If the paper narrative needs stronger high-order emphasis, run `tppr_K=3, taps_beta=0.2` as the paired comparison. If server memory and time are sufficient, `tppr_K=3, taps_beta=0.3` is a secondary candidate because it has the best local purity, but its graph density is much higher.

## Suggested Next Commands

Diagnostic only:

```bash
python diagnose_tppr.py --dataset dblp --tppr_K 3 --taps_budget_beta 0.2
python diagnose_tppr.py --dataset dblp --tppr_K 3 --taps_budget_beta 0.3
```

Training candidate:

```bash
python main.py \
  --dataset dblp \
  --objective_mode cut_main \
  --epoch 40 \
  --assign_mode prototype \
  --prototype_alpha 1.0 \
  --prototype_lr_scale 0.01 \
  --lambda_temp 0.01 \
  --lambda_batch 0.01 \
  --lambda_com 1.0 \
  --rho_cut 0.1 \
  --rho_kl 10.0 \
  --rho_bal 5.0 \
  --tppr_K 2 \
  --taps_budget_mode nlogn \
  --taps_budget_beta 0.2 \
  --kl_target_mode dynamic_tgc \
  --balance_mode hinos \
  --batch_recon_mode ones \
  --warmup_epochs 10 \
  --com_ramp_epochs 20 \
  --eval_interval 5 \
  --main_pred_mode argmax_s \
  --run_tag dblp_K2_beta02_weakcut_rkl10_rbal5_e40
```

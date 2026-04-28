# Degree-Corrected TPPR-Cut

This note documents the implemented degree-corrected variant of the existing TPPR-Cut term.

It does not add a new top-level loss. It does not modify Student-t assignment, DTGC-KL, HINOS-Bal, temporal loss, batch reconstruction, TPPR/TAPS cache construction, or final prediction.

The full community objective remains:

\[
\mathcal{L}_{com}
=
\rho_{cut}\mathcal{L}_{TPPR\text{-}Cut}
+
\rho_{kl}\mathcal{L}_{DTGC\text{-}KL}
+
\rho_{bal}\mathcal{L}_{HINOS\text{-}Bal}.
\]

Only the internal numerator of \(\mathcal{L}_{TPPR\text{-}Cut}\) is changed when:

```text
--tppr_cut_objective degree_corrected
```

The default remains:

```text
--tppr_cut_objective ncut
```

so the original TPPR-Cut behavior is preserved unless the new option is explicitly selected.

## Motivation

DBLP diagnostics show that raw high-order TPPR contains useful closure information but also hub-mediated cross-label leakage. The second-order path diagnostic shows that endpoint label purity decreases sharply when the middle node has high static degree:

| Middle node degree bucket | Endpoint label purity |
|---|---:|
| \(\le 5\) | 0.5926 |
| 6-10 | 0.6195 |
| 11-20 | 0.5505 |
| 21-50 | 0.4935 |
| \(>50\) | 0.4115 |
| overall | 0.4873 |

This explains why the original TPPR-Cut can decrease while `argmax(Q)` NMI decreases:

\[
\mathcal{L}_{TPPR\text{-}Cut}\downarrow,
\quad
\mathrm{NMI}_{argmax(Q)}\downarrow.
\]

The goal is to keep the current TPPR-Cut framework while adding a degree-preserving null-model correction to its Laplacian numerator.

## Literature Basis

Newman's modularity compares observed affinity with a degree-preserving null model:

\[
B_{ij}
=
A_{ij}
-
\frac{k_i k_j}{2m}.
\]

The term \(k_i k_j/(2m)\) represents the expected affinity under a configuration-model null graph. It prevents high-degree nodes from being treated as community evidence merely because they connect broadly.

Reference:

M. E. J. Newman, "Modularity and community structure in networks," PNAS, 2006.

Karrer and Newman further show that degree heterogeneity can distort community detection and motivate degree-corrected community models.

Reference:

B. Karrer and M. E. J. Newman, "Stochastic blockmodels and community structure in networks," Physical Review E, 2011.

## Original TPPR-Cut

The current TPPR cut graph is:

\[
W_\Pi
=
\frac{1}{2}(\Pi+\Pi^\top),
\quad
(W_\Pi)_{ii}=0.
\]

Degree:

\[
d_i^\Pi
=
\sum_j W_{\Pi,ij},
\quad
2m_\Pi
=
\sum_i d_i^\Pi.
\]

Laplacian:

\[
L_\Pi
=
D_\Pi-W_\Pi.
\]

The original TPPR-Cut is:

\[
\mathcal{L}_{TPPR\text{-}Cut}^{raw}
=
\operatorname{Tr}
\left[
G^{-1}L_Q
\right],
\]

where:

\[
G
=
Q^\top D_\Pi Q+\epsilon I,
\]

\[
L_Q
=
Q^\top L_\Pi Q.
\]

In code, \(L_Q\) is computed from sparse upper-triangular edges:

```python
delta = Q_i - Q_j
l_raw = delta.T @ (w_ij * delta)
g_mat = Q.T @ (degree * Q)
loss_tppr_cut = trace(solve(g_mat + eps * I, l_raw))
```

## Degree-Corrected Laplacian Numerator

Define the degree-corrected residual affinity:

\[
B_\Pi
=
W_\Pi
-
\gamma
\frac{
d^\Pi(d^\Pi)^\top
}{
2m_\Pi
}.
\]

The residual degree is:

\[
D_B
=
(1-\gamma)D_\Pi.
\]

Therefore the residual Laplacian is:

\[
L_\Pi^{dc}
=
D_B-B_\Pi
=
D_\Pi-W_\Pi
+
\gamma
\left(
\frac{
d^\Pi(d^\Pi)^\top
}{
2m_\Pi
}
-
D_\Pi
\right).
\]

The \(-\gamma D_\Pi\) term is necessary. Without it, the expression is not the Laplacian of the residual graph.

The implemented degree-corrected TPPR-Cut is:

\[
\mathcal{L}_{TPPR\text{-}Cut}^{dc}
=
\operatorname{Tr}
\left[
(Q^\top D_\Pi Q+\epsilon I)^{-1}
Q^\top L_\Pi^{dc}Q
\right].
\]

Equivalently:

\[
\mathcal{L}_{TPPR\text{-}Cut}^{dc}
=
\operatorname{Tr}
\left[
G^{-1}
\left(
L_Q
+
\gamma(E_Q-G_0)
\right)
\right],
\]

where:

\[
G_0=Q^\top D_\Pi Q,
\]

\[
E_Q
=
\frac{
(Q^\top d^\Pi)(Q^\top d^\Pi)^\top
}{
2m_\Pi+\epsilon
}.
\]

The denominator remains the original:

\[
Q^\top D_\Pi Q+\epsilon I.
\]

Thus the implementation is strictly a modification of the existing TPPR-Cut numerator, not an additional loss term.

## Implementation Location

The implementation is in `trainer.py`:

```python
_compute_tppr_cut(assign)
```

The relevant parameters are:

```text
--tppr_cut_objective ncut
--tppr_cut_objective degree_corrected
--tppr_cut_gamma
```

When `tppr_cut_objective=ncut`, the function uses the original numerator:

\[
L_Q.
\]

When `tppr_cut_objective=degree_corrected`, the function uses:

\[
L_Q+\gamma(E_Q-G_0).
\]

The outer loss is unchanged:

```python
l_com = rho_cut * l_tppr_cut + rho_kl * l_assign + rho_bal * l_hinos_bal
```

## Important Risks

This variant has several properties that must be considered during experiments.

1. The numerator can be indefinite.

The residual matrix:

\[
B_\Pi
=
W_\Pi-\gamma\frac{d^\Pi(d^\Pi)^\top}{2m_\Pi}
\]

can contain negative effective affinities. Therefore:

\[
L_\Pi^{dc}
\]

is not guaranteed to be positive semidefinite.

2. `loss_tppr_cut` can be negative.

Because the corrected numerator can be indefinite, the trace ratio:

\[
\operatorname{Tr}
\left[
G^{-1}
Q^\top L_\Pi^{dc}Q
\right]
\]

may become negative. This is not automatically a bug. It means the degree-corrected residual objective is rewarding assignments whose observed TPPR affinity exceeds the degree-preserving null expectation.

3. \(\gamma=1.0\) may be too strong.

The value:

\[
\gamma=1.0
\]

corresponds to the full Newman-style degree null correction. On DBLP, where TPPR is already noisy and hub leakage is strong, this correction may dominate the raw cut numerator.

4. Loss decrease alone is not sufficient.

For this variant, success should not be judged by:

\[
\mathcal{L}_{TPPR\text{-}Cut}\downarrow
\]

alone. The main checks must include:

\[
\mathrm{NMI}_{argmax(Q)}
\]

and:

\[
\mathrm{assignment\ entropy}.
\]

A lower or negative `loss_tppr_cut` is useful only if `argmax(Q)` quality does not collapse and assignment entropy remains in a reasonable range.

## Metrics To Track

The CSV includes:

```text
tppr_cut_objective
tppr_cut_gamma
loss_tppr_cut
loss_tppr_cut_raw
tppr_cut_expected_trace
tppr_cut_degree_trace
tppr_cut_dc_correction_trace
```

Interpretation:

\[
loss\_tppr\_cut
=
loss\_tppr\_cut\_raw
+
\gamma\cdot tppr\_cut\_dc\_correction\_trace.
\]

Training should also be judged using:

```text
nmi_argmax_s
acc_argmax_s
ari_argmax_s
f1_argmax_s
assignment_entropy
cluster_volume_entropy
loss_assign_penalty
loss_hinos_bal
loss_temp
loss_batch
```

## First DBLP Experiment Grid

Do not only run \(\gamma=1.0\). The first batch should include smaller corrections:

| Run | `tppr_cut_objective` | `tppr_cut_gamma` | Purpose |
|---|---|---:|---|
| Raw baseline | `ncut` | n/a | Original TPPR-Cut behavior |
| Weak correction | `degree_corrected` | 0.25 | Conservative degree correction |
| Medium correction | `degree_corrected` | 0.5 | Stronger but not full null correction |
| Full correction | `degree_corrected` | 1.0 | Newman-style full degree correction |

Use the same non-cut loss settings across all runs:

```text
lambda_temp = 0.01
lambda_batch = 0.01
lambda_com = 1.0
rho_cut = 0.1
rho_kl = 10.0
rho_bal = 5.0
warmup_epochs = 10
com_ramp_epochs = 20
epoch = 40
```

Suggested graph setting:

```text
tppr_K = 3
taps_budget_beta = 0.3
```

## Commands

Raw baseline:

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
  --tppr_K 3 \
  --taps_budget_mode nlogn \
  --taps_budget_beta 0.3 \
  --tppr_cut_objective ncut \
  --target_update_interval 5 \
  --kl_target_mode dynamic_tgc \
  --balance_mode hinos \
  --batch_recon_mode ones \
  --warmup_epochs 10 \
  --com_ramp_epochs 20 \
  --eval_interval 5 \
  --main_pred_mode argmax_s \
  --run_tag tppr_ncut_k3_beta03
```

Degree-corrected \(\gamma=0.25\):

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
  --tppr_K 3 \
  --taps_budget_mode nlogn \
  --taps_budget_beta 0.3 \
  --tppr_cut_objective degree_corrected \
  --tppr_cut_gamma 0.25 \
  --target_update_interval 5 \
  --kl_target_mode dynamic_tgc \
  --balance_mode hinos \
  --batch_recon_mode ones \
  --warmup_epochs 10 \
  --com_ramp_epochs 20 \
  --eval_interval 5 \
  --main_pred_mode argmax_s \
  --run_tag tppr_dc_laplacian_k3_beta03_gamma025
```

Degree-corrected \(\gamma=0.5\):

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
  --tppr_K 3 \
  --taps_budget_mode nlogn \
  --taps_budget_beta 0.3 \
  --tppr_cut_objective degree_corrected \
  --tppr_cut_gamma 0.5 \
  --target_update_interval 5 \
  --kl_target_mode dynamic_tgc \
  --balance_mode hinos \
  --batch_recon_mode ones \
  --warmup_epochs 10 \
  --com_ramp_epochs 20 \
  --eval_interval 5 \
  --main_pred_mode argmax_s \
  --run_tag tppr_dc_laplacian_k3_beta03_gamma05
```

Degree-corrected \(\gamma=1.0\):

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
  --tppr_K 3 \
  --taps_budget_mode nlogn \
  --taps_budget_beta 0.3 \
  --tppr_cut_objective degree_corrected \
  --tppr_cut_gamma 1.0 \
  --target_update_interval 5 \
  --kl_target_mode dynamic_tgc \
  --balance_mode hinos \
  --batch_recon_mode ones \
  --warmup_epochs 10 \
  --com_ramp_epochs 20 \
  --eval_interval 5 \
  --main_pred_mode argmax_s \
  --run_tag tppr_dc_laplacian_k3_beta03_gamma1
```

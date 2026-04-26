# Loss Function Design

## 1. Overview

The current HINOS offline objective is

\[
\mathcal{L}
=
\mathcal{L}_{\mathrm{temp}}
+
\lambda_{\mathrm{com}}\mathcal{L}_{\mathrm{com}}
+
\lambda_{\mathrm{batch}}\mathcal{L}_{\mathrm{batch}}.
\]

The three terms have separate roles:

- \(\mathcal{L}_{\mathrm{temp}}\): temporal dynamics preservation.
- \(\mathcal{L}_{\mathrm{com}}\): TPPR-induced community-aware cut regularization.
- \(\mathcal{L}_{\mathrm{batch}}\): batch-level local structure reconstruction.

TPPR + Cut is the core advantage of this pipeline and should not be removed or replaced by ordinary adjacency-based clustering losses.

## 2. TPPR-induced temporal affinity

The TPPR-induced temporal-structural similarity matrix is

\[
\Pi \in \mathbb{R}^{N \times N}.
\]

Its degree matrix is

\[
D_{\Pi}=\operatorname{diag}(\Pi\mathbf{1}).
\]

\(\Pi\) is produced by temporal path sparsification and truncated TPPR. It is not a plain adjacency matrix; it encodes higher-order temporal-structural proximity.

## 3. Community-aware loss

The community-aware loss is

\[
\mathcal{L}_{\mathrm{com}}
=
\mathcal{L}_{\mathrm{TPPR\text{-}Cut}}
+
\rho\mathcal{R}_{\Pi}(S).
\]

The main term is TPPR-induced normalized cut:

\[
\mathcal{L}_{\mathrm{TPPR\text{-}Cut}}
=
\operatorname{Tr}
\left[
\left(S^{\top}D_{\Pi}S\right)^{-1}
S^{\top}
\left(D_{\Pi}-\Pi\right)
S
\right].
\]

This is the core community objective. The assignment penalty only stabilizes the soft assignment matrix.

## 4. TPPR-aware assignment penalty

The unified assignment penalty is

\[
\mathcal{R}_{\Pi}(S)
=
\operatorname{KL}(P_{\Pi}\|S).
\]

Expanded:

\[
\mathcal{R}_{\Pi}(S)
=
\sum_{i=1}^{N}
\bar{d}_{\Pi,i}
\sum_{k=1}^{K}
P^{\Pi}_{ik}
\log
\frac{
P^{\Pi}_{ik}+\epsilon
}{
S_{ik}+\epsilon
}.
\]

The normalized TPPR degree weight is

\[
\bar{d}_{\Pi,i}
=
\frac{d_{\Pi,i}}{\sum_{j=1}^{N}d_{\Pi,j}}.
\]

The soft volume of cluster \(k\) on the TPPR graph is

\[
f_k
=
\sum_{i=1}^{N}
\bar{d}_{\Pi,i}S_{ik}.
\]

The target distribution is

\[
P^{\Pi}_{ik}
=
\frac{
S_{ik}^{2}/(f_k+\epsilon)
}{
\sum_{\ell=1}^{K}
S_{i\ell}^{2}/(f_{\ell}+\epsilon)
}.
\]

This design has one unified penalty:

- \(S_{ik}^2\) increases assignment confidence.
- \(f_k\) prevents large clusters from absorbing all nodes.
- \(\bar d_{\Pi,i}\) makes the penalty TPPR-aware.
- It is one function \(\mathcal{R}_{\Pi}(S)\), not a stack of multiple losses.

## 5. Temporal dynamics loss

The temporal score follows the existing Hawkes-style implementation. For an event \((u,v,t)\),

\[
\mu(u,v,t)=-\|z_u-z_v\|_2^2.
\]

\[
\alpha(x,u,t)
=
\frac{
\exp(\mu(x,u,t))
}{
\sum_{y\in\mathcal{N}_{u,t}}\exp(\mu(y,u,t))
}.
\]

\[
s(u,v,t)
=
\mu(u,v,t)
+
\sum_{x\in\mathcal{N}_{u,t}}
\alpha(x,u,t)
\mu(x,v,t)
\exp(-\delta(t-t_x)).
\]

\[
\mathcal{L}_{\mathrm{temp}}
=
-\frac{1}{|\mathcal{B}|}
\sum_{(u,v,t)\in\mathcal{B}}
\left[
\log\sigma(s(u,v,t))
+
\sum_{n\in\mathcal{N}^{-}_{u}}
\log\sigma(-s(u,n,t))
\right].
\]

The implementation uses \(\delta_{+}=\operatorname{softplus}(\delta)\), so the decay is

\[
\exp(-\delta_{+}(t-t_x))
\]

instead of

\[
\exp(\delta(t-t_x)).
\]

Larger time gaps therefore have smaller historical influence.

## 6. Batch-level reconstruction loss

The batch reconstruction term preserves local interaction structure inside each mini-batch. It keeps observed source-target and source-history pairs close and pushes sampled negatives away. It supports representation learning but does not carry the core clustering objective.

Using cosine similarity,

\[
\operatorname{sim}(u,w)
=
\frac{z_u^\top z_w}
{\|z_u\|_2\|z_w\|_2}.
\]

\[
\mathcal{L}_{\mathrm{batch}}
=
\frac{1}{|\mathcal{B}|}
\sum_{(u,v,t)\in\mathcal{B}}
\left[
(1-\operatorname{sim}(u,v))^2
+
\frac{1}{|\mathcal{N}_{u,t}|}
\sum_{h\in\mathcal{N}_{u,t}}
(1-\operatorname{sim}(u,h))^2
+
\frac{1}{|\mathcal{N}^{-}_{u}|}
\sum_{n\in\mathcal{N}^{-}_{u}}
\operatorname{sim}(u,n)^2
\right].
\]

The code may use equivalent norm-based reductions, but the role is batch-level local structure reconstruction.

## 7. Implementation mapping

- `HINOS/sparsification.py`: TPPR construction and cut graph inputs via `compute_tppr_cached(...)` and `build_ncut_graph(...)`.
- `HINOS/trainer.py`: temporal loss, batch reconstruction loss, TPPR-Cut, TPPR-aware assignment penalty, metrics logging, and prediction export.
- `HINOS/main.py`: hyperparameters and legacy argument aliases.
- \(\mathcal{L}_{\mathrm{temp}}\): temporal score loss.
- \(\mathcal{L}_{\mathrm{com}}\): TPPR-Cut + TPPR-aware assignment penalty.
- \(\mathcal{L}_{\mathrm{batch}}\): batch-level reconstruction.

## 8. Recommended commands

Run the school dataset:

```bash
python main.py \
  --dataset school \
  --objective_mode cut_main \
  --epoch 100 \
  --lambda_com 1.0 \
  --rho_assign 0.1 \
  --lambda_batch 0.01 \
  --warmup_epochs 20 \
  --eval_interval 5 \
  --grad_eval_interval 5 \
  --run_tag tppr_cut_dta_v1
```

Sweep \(\rho\):

```bash
for rho in 0.01 0.05 0.1 0.2 0.5
do
  python main.py \
    --dataset school \
    --objective_mode cut_main \
    --epoch 100 \
    --lambda_com 1.0 \
    --rho_assign ${rho} \
    --lambda_batch 0.01 \
    --warmup_epochs 20 \
    --eval_interval 5 \
    --grad_eval_interval 5 \
    --run_tag tppr_cut_dta_rho${rho}
done
```

## 9. Expected diagnostics

Track:

- `acc_kmeans_z`
- `nmi_kmeans_z`
- `ari_kmeans_z`
- `acc_argmax_s`
- `nmi_argmax_s`
- `ari_argmax_s`
- `cluster_volume_min`
- `cluster_volume_max`
- `assignment_entropy`
- `loss_tppr_cut`
- `loss_assign_penalty`
- `loss_com`

Expected behavior:

- `kmeans_z` should not degrade.
- `argmax_s` should be more stable than with the old standalone `loss_bal`.
- `cluster_volume_min` should not stay close to 0 for long.
- `assignment_entropy` should decrease without collapsing too early.

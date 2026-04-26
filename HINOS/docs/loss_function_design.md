# Loss Function Design

## 1. Overview

The current HINOS offline objective is

\[
\mathcal{L}
=
\lambda_{\mathrm{temp}}\mathcal{L}_{\mathrm{temp}}
+
\lambda_{\mathrm{com}}\mathcal{L}_{\mathrm{com}}
+
\lambda_{\mathrm{batch}}\mathcal{L}_{\mathrm{batch}}.
\]

The three terms have separate roles:

- \(\mathcal{L}_{\mathrm{temp}}\): temporal dynamics preservation.
- \(\mathcal{L}_{\mathrm{com}}\): TPPR-induced community-aware objective.
- \(\mathcal{L}_{\mathrm{batch}}\): batch-level local structure reconstruction.

Temporal loss is not removed. TPPR + Cut is the core advantage of this pipeline and should not be replaced by ordinary adjacency-based clustering losses.

## 2. TPPR-induced community objective

The community-aware loss is

\[
\mathcal{L}_{\mathrm{com}}
=
\mathcal{L}_{\mathrm{TPPR\text{-}Cut}}
+
\rho_{\mathrm{assign}}\mathcal{R}_{\Pi}(S).
\]

The main term is TPPR-induced normalized cut:

\[
\mathcal{L}_{\mathrm{TPPR\text{-}Cut}}
=
\operatorname{Tr}
\left[
(S^{\top}D_{\Pi}S)^{-1}
S^{\top}(D_{\Pi}-\Pi)S
\right].
\]

\(\Pi\) is the TPPR-induced temporal-structural affinity matrix, not a plain adjacency matrix. It is built from temporal path sparsification and truncated TPPR, then converted into the full-graph cut representation used by the optimizer.

## 3. Why previous KL failed

The previous KL penalty was not necessarily wrong. The issue was that the relaxed assignment source \(S\) was too weak.

When the soft assignment is close to uniform,

\[
S_{ik}\approx 1/K,
\]

the self-target distribution constructed from \(S\) is also close to uniform:

\[
P^{\Pi}_{ik}\approx S_{ik}.
\]

Therefore

\[
\operatorname{KL}(P_{\Pi}\|S)\approx 0.
\]

In that regime, the self-target KL does not actively break uniform soft assignments. The fix is to improve the source of \(S\), while keeping the TPPR-aware KL design.

## 4. Prototype-based assignment

By default, \(S\) is computed with prototype-based Student-t assignment:

\[
S_{ik}
=
\frac{
(1+\|z_i-c_k\|_2^2/\alpha)^{-(\alpha+1)/2}
}{
\sum_{\ell=1}^{K}
(1+\|z_i-c_{\ell}\|_2^2/\alpha)^{-(\alpha+1)/2}
}.
\]

Here \(c_k\) is a learnable cluster prototype and \(\alpha\) defaults to 1.0. Prototypes are initialized by KMeans on pretrained node2vec embeddings. The node2vec embeddings only provide the warm start; the final representation is still refined by temporal loss, TPPR-Cut, TPPR-aware KL, and batch reconstruction.

The older MLP head is still available through `--assign_mode mlp` for ablations.

## 5. TPPR-aware KL assignment penalty

The normalized TPPR degree weight is

\[
\bar d_{\Pi,i}
=
\frac{d_{\Pi,i}}{\sum_j d_{\Pi,j}}.
\]

The TPPR-weighted soft cluster volume is

\[
f_k
=
\sum_i \bar d_{\Pi,i}S_{ik}.
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

The assignment penalty is

\[
\mathcal{R}_{\Pi}(S)
=
\sum_i
\bar d_{\Pi,i}
\sum_k
P^{\Pi}_{ik}
\log
\frac{P^{\Pi}_{ik}+\epsilon}{S_{ik}+\epsilon}.
\]

This remains a single penalty function:

- \(S_{ik}^2\) sharpens high-confidence assignments.
- \(f_k\) reduces large-cluster bias.
- \(\bar d_{\Pi,i}\) makes the penalty TPPR-aware.
- It is not a stack of BSA, entropy, or balance losses.

In implementation, \(P^{\Pi}\) is detached before the KL term is evaluated.

## 6. Temporal dynamics loss

The temporal score keeps the existing Hawkes-style structure:

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

The implementation uses positive decay \(\delta_+=\operatorname{softplus}(\delta)\), so historical influence decays as

\[
\exp(-\delta_+(t-t_x))
\]

instead of

\[
\exp(\delta(t-t_x)).
\]

## 7. Batch reconstruction loss

The batch reconstruction term preserves local interaction structure inside each mini-batch. It keeps observed source-target and source-history pairs close and pushes sampled negatives away. It supports representation learning but does not replace the TPPR-induced community objective.

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

## 8. Implementation mapping

- `HINOS/main.py`: CLI arguments for objective weights, assignment mode, prototype alpha, and main prediction export.
- `HINOS/trainer.py`: prototype assignment, MLP assignment ablation, community loss, temporal loss, batch reconstruction, diagnostics, and prediction export.
- `HINOS/sparsification.py`: TPPR construction and cut graph construction via `compute_tppr_cached(...)` and `build_ncut_graph(...)`.
- Pretrained embeddings: initialize both \(Z\) and prototype centers when `--assign_mode prototype`.

## 9. Recommended commands

40-epoch debug run:

```bash
python main.py \
  --dataset school \
  --objective_mode cut_main \
  --epoch 40 \
  --assign_mode prototype \
  --prototype_alpha 1.0 \
  --lambda_temp 0.01 \
  --lambda_com 1.0 \
  --rho_assign 0.1 \
  --lambda_batch 0.01 \
  --warmup_epochs 10 \
  --eval_interval 5 \
  --grad_eval_interval 5 \
  --run_tag proto_kl_debug
```

100-epoch run:

```bash
python main.py \
  --dataset school \
  --objective_mode cut_main \
  --epoch 100 \
  --assign_mode prototype \
  --prototype_alpha 1.0 \
  --lambda_temp 0.01 \
  --lambda_com 1.0 \
  --rho_assign 0.1 \
  --lambda_batch 0.01 \
  --warmup_epochs 20 \
  --eval_interval 5 \
  --grad_eval_interval 5 \
  --run_tag proto_kl_100
```

## 10. Diagnostics

Track:

- `loss_temp`
- `loss_tppr_cut`
- `loss_assign_penalty`
- `loss_com`
- `weighted_temp`
- `weighted_com`
- `acc_kmeans_z`
- `acc_argmax_s`
- `assignment_entropy`
- `cluster_volume_min`
- `cluster_volume_max`

Expected behavior:

- `kmeans_z` should not degrade.
- `loss_assign_penalty` should no longer stay near 0.
- `argmax_s` should be more stable than the MLP-head version.
- TPPR-Cut remains present and participates in optimization.

# Loss Function Design

## 1. Overview

The current method adapts the offline stage of HINOS to temporal community discovery. It keeps the three top-level offline objectives:

\[
\mathcal{L}
=
\lambda_{\mathrm{temp}}\mathcal{L}_{\mathrm{temp}}
+
\lambda_{\mathrm{batch}}\mathcal{L}_{\mathrm{batch}}
+
\frac{\lambda_{\mathrm{com}}^{(e)}}{B}\mathcal{L}_{\mathrm{com}},
\]

where \(B\) is the number of mini-batches in the epoch. The community loss is full-graph and is divided by \(B\) so it is not implicitly applied \(B\) times per epoch.

\[
\mathcal{L}_{\mathrm{com}}
=
\rho_{\mathrm{cut}}\mathcal{L}_{\mathrm{TPPR\text{-}Cut}}
+
\rho_{\mathrm{KL}}\mathcal{L}_{\mathrm{TGC\text{-}KL}}
+
\rho_{\mathrm{bal}}\mathcal{L}_{\mathrm{HINOS\text{-}Bal}}.
\]

Default weights:

\[
\rho_{\mathrm{cut}}=1,\quad
\rho_{\mathrm{KL}}=1,\quad
\rho_{\mathrm{bal}}=0.1.
\]

## 2. Student-t Assignment

The main assignment head is TGC-style Student-t assignment. Let

\[
Z=[z_1,\dots,z_N]^\top \in \mathbb{R}^{N\times d},
\quad
C=[c_1,\dots,c_K]^\top \in \mathbb{R}^{K\times d}.
\]

The centers \(C\) are initialized with KMeans on pretrained node2vec features. For node \(i\) and cluster \(k\):

\[
Q_{ik}
=
\frac{
\left(1+\frac{\lVert z_i-c_k\rVert_2^2}{\nu}\right)^{-\frac{\nu+1}{2}}
}{
\sum_{\ell=1}^{K}
\left(1+\frac{\lVert z_i-c_\ell\rVert_2^2}{\nu}\right)^{-\frac{\nu+1}{2}}
}.
\]

The default is \(\nu=1\), exposed as `--prototype_alpha 1.0`. In the implementation,

\[
S := Q.
\]

The final end-to-end cluster output is:

\[
\hat y_i = \arg\max_k Q_{ik}.
\]

Prototype centers are learnable, but they use a smaller learning rate:

\[
\eta_C = \gamma_C \eta_Z,
\]

with default `--prototype_lr_scale 0.1`.

## 3. DTGC Batch-Level KL

Following DTGC/TGC, the KL term uses a detached sharpened target from the fixed pretrained feature of the current source-node batch \(B\), not a cached full-graph target. First compute the Student-t distribution from \(Z^0\):

\[
q^0_{ik}
=
\frac{
\left(1+\lVert z_i^0-c_k\rVert_2^2/\nu\right)^{-\frac{\nu+1}{2}}
}{
\sum_{\ell=1}^{K}
\left(1+\lVert z_i^0-c_\ell\rVert_2^2/\nu\right)^{-\frac{\nu+1}{2}}
},
\quad i\in B.
\]

Then compute the batch target:

\[
f_k = \sum_{i\in B}q^0_{ik}.
\]

\[
P_{ik}
=
\frac{
(q^0_{ik})^{2}/(f_k+\epsilon)
}{
\sum_{\ell=1}^{K}(q^0_{i\ell})^{2}/(f_\ell+\epsilon)+\epsilon
}.
\]

The live assignment is computed from the current trainable embedding:

\[
q'_{ik}
=
\operatorname{StudentT}(z_i^t,c_k).
\]

The target is detached:

\[
P \leftarrow \operatorname{stopgrad}(P).
\]

The KL term is normalized by batch size:

\[
\mathcal{L}_{\mathrm{TGC\text{-}KL}}^{(B)}
=
\frac{1}{|B|}
\sum_{i\in B}
\sum_{k=1}^{K}
P_{ik}
\log
\frac{P_{ik}+\epsilon}{q'_{ik}+\epsilon}.
\]

The full-graph current assignment

\[
Q=\operatorname{StudentT}(Z^t,C)
\]

is still used by TPPR-Cut, HINOS-Bal, and `argmax_s`. `target_update_interval` is retained only for command compatibility. `kl_target_mode fixed_initial` keeps the previous fixed-prior target only as an ablation.

## 4. TPPR-Cut

HINOS constructs a TPPR-induced temporal-structural affinity matrix:

\[
\Pi \in \mathbb{R}^{N\times N}.
\]

For cut optimization, the implementation uses the symmetrized graph:

\[
W_{\Pi}=\frac{1}{2}(\Pi+\Pi^\top),\quad (W_\Pi)_{ii}=0.
\]

The degree and Laplacian are:

\[
d_i^\Pi = \sum_j (W_\Pi)_{ij},
\quad
D_\Pi=\operatorname{diag}(d_1^\Pi,\dots,d_N^\Pi),
\quad
L_\Pi=D_\Pi-W_\Pi.
\]

The relaxed TPPR-Cut term is:

\[
\mathcal{L}_{\mathrm{TPPR\text{-}Cut}}
=
\operatorname{Tr}
\left[
(Q^\top D_\Pi Q+\epsilon I)^{-1}
Q^\top L_\Pi Q
\right].
\]

The code computes \(Q^\top L_\Pi Q\) through sparse edges:

\[
Q^\top L_\Pi Q
=
\sum_{(i,j)\in E_\Pi}
w_{ij}(Q_i-Q_j)^\top(Q_i-Q_j).
\]

This term encourages nodes with strong high-order temporal-structural proximity to receive similar Student-t community assignments.

## 5. HINOS Balance

HINOS adds a query-agnostic penalty to avoid collapsed assignments and overly diffuse memberships. Let \(q_j=Q_{:,j}\), and let

\[
2m_\Pi = \sum_i d_i^\Pi.
\]

The implemented penalty follows the original HINOS formula:

\[
\mathcal{L}_{\mathrm{HINOS\text{-}Bal}}
=
\frac{1}{\sqrt{K}-1}
\left(
\sqrt{K}
-
\frac{1}{\sqrt{2m_\Pi}}
\sum_{j=1}^{K}
\left\|
q_j \odot (d_\Pi)^{1/2}
\right\|_2
\right).
\]

This term is suitable for community discovery because it depends only on full-graph assignment \(Q\), TPPR degree \(d_\Pi\), and total TPPR volume \(m_\Pi\). It is minimized when assignments are sharp and TPPR-volume balanced.

## 6. Temporal Dynamics Loss

The temporal loss keeps the Hawkes-style positive/negative interaction modeling:

\[
\mu(u,v,t)=-\lVert z_u-z_v\rVert_2^2.
\]

\[
s(u,v,t)
=
\mu(u,v,t)
+
\sum_{x\in\mathcal{N}_{u,t}}
\alpha(x,u,t)\mu(x,v,t)\exp(-\delta_+(t-t_x)).
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

The implementation uses \(\delta_+=\operatorname{softplus}(\delta)\) to ensure positive temporal decay.

## 7. Batch Reconstruction

`batch_recon_mode=ones` is the recommended default:

\[
\mathcal{L}_{\mathrm{batch}}
=
\operatorname{MSE}(\cos(z_u,z_v),1)
+
\operatorname{MSE}(\cos(z_u,z_h),1)
+
\operatorname{MSE}(\cos(z_u,z_n),0).
\]

Pseudo-label reconstruction modes remain available as ablations.

## 8. Implementation Mapping

- `main.py`: CLI flags and objective defaults.
- `trainer.py`: Student-t assignment, dynamic TGC target, TPPR-Cut, HINOS balance, ramped full-graph community loss, metrics, prediction export.
- `sparsification.py`: adaptive TAPS budget, TPPR construction, and symmetrized cut graph.

Important metric columns:

- `loss_tppr_cut`
- `loss_assign_penalty` for \(\mathcal{L}_{\mathrm{TGC\text{-}KL}}\)
- `loss_hinos_bal`
- `weighted_cut`
- `weighted_kl`
- `weighted_hinos_bal`
- `assignment_entropy`
- `cluster_volume_min`
- `cluster_volume_max`
- `nmi_argmax_s`
- `nmi_kmeans_z`
- `nmi_spectral_pi`

## 9. Recommended Checks

On a local workstation, use only static checks:

```bash
python -m py_compile main.py trainer.py data_load.py sparsification.py clustering_utils.py evaluate.py
python main.py --help
```

Run smoke tests and full experiments on the GPU server.

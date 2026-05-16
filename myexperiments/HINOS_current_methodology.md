# HINOS 当前方法说明

本文描述当前 `HINOS` 实现。当前代码主线是面向全图时间社区发现的 offline pipeline：

```text
temporal edges
  -> TAPS time-aware sparsified graph
  -> Laplacian-form TPPR
  -> symmetrized TPPR affinity W_pi
  -> Student-t prototype assignment Q
  -> fixed sparse TPPR-NCut + orth loss + temporal loss + batch reconstruction
  -> full-graph cluster labels
```

当前入口固定为：

```text
objective_mode = search_proto
```

默认主预测为：

```text
main_pred_mode = argmax_s
```

即：

```math
\hat y_i = \arg\max_k q_{ik}.
```

## 1. 输入与目标

输入是时间边流：

```math
\mathcal{E}=\{(u_m,v_m,t_m)\}_{m=1}^{M}.
```

训练目标是学习节点表示：

```math
Z\in\mathbb{R}^{N\times d}
```

以及由 Student-t prototype assignment 得到的软社区分配：

```math
Q\in\mathbb{R}^{N\times K},
\qquad q_{ik}\ge 0,
\qquad \sum_k q_{ik}=1.
```

最终导出的社区标签为：

```math
\hat y_i=\arg\max_k q_{ik}.
```

需要区分两个时间口径：

- TAPS/TPPR 构图阶段会把时间戳压缩为连续索引。
- mini-batch 训练阶段直接使用 `TGCDataSet` 中的原始 `target_time/history_times`。
- EAH 的 `birth_time` 与 batch 中的 `target_time/history_times` 使用同一时间口径，即原始时间值。

## 2. TAPS 时间稀疏图

TAPS 在原始时间图的静态支撑上采样 time-aware paths，构造时间稀疏图 `A_time`。

当前支持两种采样预算：

```math
N_{\mathrm{TAPS}}=\lceil\sqrt{|E_s|}\rceil
```

以及默认使用的：

```math
N_{\mathrm{TAPS}}
=
\left\lceil
\beta |V|\log(|V|+1)
\right\rceil.
```

对应参数：

```text
--taps_budget_mode sqrt_edges | nlogn
--taps_budget_beta
```

路径长度按截断几何分布采样，时间步上限由：

```text
--taps_T_cap
```

控制。路径扩展时，下一个节点按与当前参考时间的时间接近度采样。最终得到的 `A_time` 是对称稀疏矩阵，并被缓存到 `cache/`；训练期间不更新 TAPS 图。

## 3. Laplacian-form TPPR

当前 TPPR 构造与 `HiNoS_main` 的 Laplacian-form TPPR 对齐。

关键区别是：

```text
TAPS 图 A_time 只作为 Laplacian source；
B 和 XQ 由原始 temporal edge stream 构造，而不是由 TAPS 非零边构造。
```

令原始 temporal edge stream 为：

```math
\mathcal{E}_{stream}=[(u_\ell,v_\ell)]_{\ell=1}^{M}.
```

定义：

```math
B_{\ell j}=\mathbf{1}[j=v_\ell],
```

```math
(XQ)_{i\ell}
=
\begin{cases}
1/d_i^{out}, & i=u_\ell,\\
0, & \text{otherwise}.
\end{cases}
```

于是：

```math
S = XQ B.
```

TAPS Laplacian 为：

```math
L_{\mathrm{TAPS}} = D_{\mathrm{TAPS}} - A_{\mathrm{time}}.
```

当前 TPPR 为：

```math
\Pi
=
\alpha S
+
\left(
\sum_{r=1}^{K_{\mathrm{TPPR}}}
\alpha(1-\alpha)^r
\right)
\left(
S - S D_{\mathrm{orig}}^{-1} L_{\mathrm{TAPS}}
\right).
```

默认参数：

```text
--tppr_alpha 0.2
--tppr_K 5
```

TPPR 和 TAPS 都是固定图先验，只在训练前构造或从 cache 读取。

## 4. Fixed TPPR-NCut Graph

NCut 使用固定的对称 TPPR affinity：

```math
W_\Pi
=
\frac{1}{2}(\Pi+\Pi^\top),
\qquad
\operatorname{diag}(W_\Pi)=0.
```

`W_pi` 被存为 PyTorch sparse CSR tensor。训练期间：

```text
TAPS 不更新
TPPR 不更新
W_pi 不更新
```

梯度只更新：

```text
node embeddings Z
Hawkes delta
Student-t prototypes C
```

## 5. Student-t Prototype Assignment

当前 assignment 使用 learnable prototypes，而不是 MLP head。

Prototype 由预训练 node2vec 特征上的 KMeans 初始化：

```math
C=[c_1,\dots,c_K]^\top.
```

每个节点的软分配为：

```math
\tilde q_{ik}
=
\left(
1+\frac{\|z_i-c_k\|_2^2}{\alpha_p}
\right)^{-(\alpha_p+1)/2},
```

```math
q_{ik}
=
\frac{\tilde q_{ik}}
{\sum_{\ell=1}^{K}\tilde q_{i\ell}}.
```

对应参数：

```text
--prototype_alpha
--prototype_lr_scale
--freeze_prototypes
```

默认 prototype 可学习，其学习率为：

```math
\eta_C = \gamma_C \eta_Z.
```

其中 `gamma_C = prototype_lr_scale`。

## 6. Sparse TPPR-NCut 与 Orth Loss

令：

```math
D_\Pi=\operatorname{diag}(W_\Pi\mathbf{1}).
```

当前 soft assignment 为 `Q`，定义：

```math
L_Q = Q^\top(D_\Pi Q - W_\Pi Q),
```

```math
G_Q = Q^\top D_\Pi Q.
```

NCut loss 为：

```math
\mathcal{L}_{cut}
=
\frac{1}{K}
\operatorname{Tr}
\left[
(G_Q+\epsilon I)^{-1}L_Q
\right].
```

当前还加入 orthogonality regularization：

```math
\mathcal{L}_{orth}
=
\left\|
\frac{G_Q}{\|G_Q\|_F+\epsilon}
-
\frac{I_K}{\sqrt K}
\right\|_F.
```

社区项为：

```math
\mathcal{L}_{com}
=
\mathcal{L}_{cut}
+
\lambda_{\mathrm{ncut\_orth}}
\mathcal{L}_{orth}.
```

默认：

```text
--lambda_ncut 0.5
--lambda_ncut_orth 5.0
```

## 7. Batch Reconstruction Loss

当前支持两种 batch reconstruction：

```text
--batch_recon_mode ones
--batch_recon_mode cebr
```

### 7.1 ones

`ones` 模式保持原 DTGC 风格：观察到的 source-target 和 source-history pair 目标相似度为 1，负样本 pair 目标相似度为 0。

### 7.2 cebr

`cebr` 模式使用 normalized cosine：

```math
r_{ij}=\frac{1+\cos(z_i,z_j)}{2}
```

以及 community evidence：

```math
g_{sn}=Q_s^\top Q_n.
```

负样本项约束：

```math
(r_{sn}-g_{sn})^2.
```

当前 full run 脚本中默认倾向使用 `cebr`。

## 8. Temporal Loss

当前 temporal loss 支持三种模式：

```text
--temp_loss_type original
--temp_loss_type eah
--temp_loss_type eah_no_old
```

默认是：

```text
--temp_loss_type eah
--lambda_entry 0.1
--lambda_temp 1.0
```

### 8.1 原始 Hawkes Loss

`original` 保留原 DTGC / HTNE / CT-VAE / MVTGC 风格 Hawkes temporal score。

基础强度为：

```math
\mu(a,b)=-\|z_a-z_b\|_2^2.
```

对 source node `u` 的历史节点 `h`，attention 为：

```math
a_{u,h}
=
\operatorname{softmax}
\left(
-\|z_u-z_h\|_2^2
\right).
```

正样本 score：

```math
s(u,v,t)
=
\mu(u,v)
+
\sum_{h\in H_u(t)}
a_{u,h}
\mu(h,v)
\exp(\delta_u |t-t_h|).
```

负样本 score 同理：

```math
s(u,n,t)
=
\mu(u,n)
+
\sum_{h\in H_u(t)}
a_{u,h}
\mu(h,n)
\exp(\delta_u |t-t_h|).
```

loss 为：

```math
\mathcal{L}_{temp}
=
\frac{1}{|\mathcal{B}|}
\sum_{(u,v,t)}
\left[
-\log\sigma(s(u,v,t))
-
\sum_n \log\sigma(-s(u,n,t))
\right].
```

其中 `delta` 是 unconstrained learnable parameter。

### 8.2 Entry-Adjusted Counterfactual Hawkes Loss

`eah` 使用节点入场时间 birth time：

```math
b_i=\min\{t:(i,j,t)\in E \text{ or } (j,i,t)\in E\}.
```

对候选节点 `c`，source node `u` 的历史被分为：

```math
H_{co}(u|c,t)=\{(h,t_h): b_c \le t_h < t\},
```

```math
H_{old}(u|c,t)=\{(h,t_h): t_h < b_c\}.
```

co-entry score 为：

```math
s_{co}(u,c,t)
=
\mu(u,c)
+
\sum_{h\in H_{co}}
\alpha_{co}(u,h|c,t)
\mu(h,c)
\exp(-\delta_u(t-t_h)).
```

old-history counterfactual score 为：

```math
s_{old}(u,c,t)
=
\mu(u,c)
+
\sum_{h\in H_{old}}
\alpha_{old}(u,h|c,t)
\mu(h,c)
\exp(-\delta_u(t-t_h)).
```

其中 attention 只在对应 mask 内归一化。若对应集合为空，则 history term 为 0。

最终 logit 为：

```math
\psi(u,c,t)
=
s_{co}(u,c,t)
-
\lambda_{entry}
\left[
\operatorname{softplus}(s_{old})
+
\operatorname{softplus}(s_{old}-s_{co})
\right].
```

temporal loss 为：

```math
\mathcal{L}_{temp}^{EAH}
=
\frac{1}{|\mathcal{B}|}
\sum_{(u,v,t)}
\left[
-\log\sigma(\psi(u,v,t))
-
\sum_n \log\sigma(-\psi(u,n,t))
\right].
```

### 8.3 eah_no_old

`eah_no_old` 是 ablation：

```math
\psi(u,c,t)=s_{co}(u,c,t).
```

它只保留入场后的 co-entry history，不加入 old-history counterfactual penalty。

## 9. Risk-set Negative Sampling

当前负采样从原来的全局 degree table 改为 risk-set negative sampling。

对事件时间 `t`：

```math
R(t)=\{n:b_n\le t\}.
```

采样仍尽量保持 degree^0.75 负采样思想，但最终必须满足：

```math
b_n \le t.
```

同时优先排除：

```text
source node u
positive target v
```

若早期 risk set 太小，则允许有放回采样。训练日志中记录：

```text
risk_negative_resample_count
```

用于观察过滤和重采样强度。

## 10. Total Objective

当前总目标为：

```math
\mathcal{L}_{total}
=
\lambda_{batch}\mathcal{L}_{batch}
+
\lambda_{temp}\mathcal{L}_{temp}
+
\lambda_{ncut}
\left(
\mathcal{L}_{cut}
+
\lambda_{ncut\_orth}\mathcal{L}_{orth}
\right).
```

默认：

```text
lambda_batch = 1.0
lambda_temp = 1.0
lambda_ncut = 0.5
lambda_ncut_orth = 5.0
lambda_entry = 0.1
```

注意：当前没有启用 `rho_cut/rho_kl/rho_bal` 那套旧 `cut_main` 目标；当前主线是 `search_proto`。

## 11. Prediction and Evaluation

默认主预测：

```text
argmax_s
```

即：

```math
\hat y_i=\arg\max_k Q_{ik}.
```

同时保留诊断预测模式：

```text
kmeans_z
kmeans_s
spectral_pi
spectral_topk_pi
```

当 label 可用时记录：

```text
ACC
NMI
ARI
macro F1
```

还记录 TPPR graph diagnostics：

```text
purity_at_5_pi
purity_at_10_pi
purity_at_20_pi
ncut_gt_pi
ncut_pred_pi
ncut_pred_over_gt_pi
```

EAH 相关日志包括：

```text
loss_temp_eah
loss_temp_pos
loss_temp_neg
mean_s_co_pos
mean_s_old_pos
mean_psi_pos
mean_psi_neg
mean_h_co_size_pos
mean_h_old_size_pos
ratio_empty_h_co_pos
ratio_empty_h_old_pos
ratio_new_target_events
mean_birth_time_pos_target
risk_negative_resample_count
```

## 12. 当前实现边界

当前 `HINOS` 的方法边界是：

```text
fixed TAPS/TPPR graph prior
+ Student-t learnable prototype assignment
+ sparse fixed TPPR-NCut
+ NCut orthogonality regularization
+ batch reconstruction
+ switchable temporal loss: original / EAH / EAH-no-old
+ risk-set negative sampling
```

不包含：

```text
query-aware community search
BFS candidate generation
top-k retrieval evaluation
rho_cut/rho_kl/rho_bal cut_main objective
dynamic TGC KL target
HINOS balance penalty
SeqFilter / CNN / RNN / memory module
```

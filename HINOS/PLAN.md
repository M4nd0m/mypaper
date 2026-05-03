# TPPR-Anchored Unified Similarity Learning for Temporal Graph Clustering — 实施计划

## 0. 文档说明

本文件**替换**之前的 edge-level residual TPPR-Cut 计划（已在 `docs/degree_aware_tppr_cut.md` 归档）。

本计划面向 WWW 投稿，**核心贡献压成一句**：

> Learn a TPPR-anchored unified similarity matrix U whose spectral cut structure is aligned with semantic clustering.

围绕这一句展开，**不做双贡献并列**。η 只是 Phase 2 对 U 的事件层进一步增强，**论文叙事不主打 rhythm**。但 rhythm/FSTS 等架构机制**可以用作 `η_θ` 的具体参数化**（method 章节展开，不入 abstract）。

---

## 1. 现状与两个待解决问题

诊断已确证两个问题：

- **P1（事件层）**：brain 在 TAPS-TPPR 全网格内 `purity@5_pi` 上界 0.214；其它密集时序图也存在类似上界。
- **P2（割与标签结构性失配）**：dblp 上 `ncut_pred_pi=2.65 < ncut_gt_pi=4.59`，`nmi_spectral_pi≈0.007`。**所有数据集**都不同程度存在此问题，brain 最严重。

P2 是论文主线（Phase 1 解决），P1 是补充（Phase 2 解决）。

---

## 2. 数据集范围

| 数据集 | n | m | 平均度 | c | 备注 |
|---|---:|---:|---:|---:|---|
| school | ~327 | ~6e5 | ~3.7e3 | 9 | warmup 已饱和，作 saturation 案例 |
| dblp | ~28k | 236k | 8.3 | 10 | **Phase 1 主验证集**（P2 严重） |
| brain | 5000 | 1.96M | 781 | 10 | **Phase 2 主验证集**（P1 + P2） |
| patent | 12214 | 41916 | 3.4 | 6 | 验证迁移 |
| arXivAI | ~28k | ~3M | ~107 | 5 | 中等规模 |
| arXivCS | ~170k | ~9M | ~106 | 40 | 大规模 + 多类 |
| arxivMath | ~140k | ~7M | ~100 | 31 | 大规模 + 多类 |
| arxivPhy | ~85k | ~1.3M | ~30 | 19 | 中等规模 |
| arxivLarge | ~1.5M | 30M+ | ~40 | 8 | **计算约束最强** |

Phase 1 验证集：dblp + brain + patent + arXivAI（先确保正确性）。
Phase 2 验证集：brain + patent + dblp。
Phase 3 全跑：9 个数据集（arxivLarge 最后）。

---

## 3. 数学公式

### 3.1 现有 HINOS 损失（不动）

```
L = λ_temp · L_temp + λ_batch · L_batch + λ_com(e)/B · L_com

L_com = ρ_cut · L_TPPR-Cut(Q, W_Π)
      + ρ_kl  · L_DTGC-KL(Q, target(Z⁰))
      + ρ_bal · L_HINOS-Bal(Q, W_Π)
```

`Q ∈ R^{n×c}` simplex per row，由 prototype Student-t 给出。**保留**。

### 3.2 Phase 1 新损失：只增加 1 个 anchor 项

引入新对象：

- **U ∈ R^{n×n}**：可学统一相似图，sparsity pattern 固定为 `support(W_Π)`，不引入新边
- **W_U = (U + U^T)/2**，对角清零，作为 cut/bal 的实际作用图
- **W̄_Π**：row-normalized 后的 baseline `W_Π`：`W̄_Π[i,j] = W_Π[i,j] / Σ_k W_Π[i,k]`

参数化：U 用 `row-softmax(θ)`，θ 为 sparse logits，shape = `nnz(W_Π)`。初始化 `θ = log(W_Π[i,j] + ε)`，使 U 初值 ≈ W̄_Π。

新损失项（只 1 个）：

```
L_anchor = ‖U − W̄_Π‖_F^2
```

完整训练损失：

```
L_com_new = ρ_cut    · L_TPPR-Cut(Q, W_U)
          + ρ_kl     · L_DTGC-KL(Q, target(Z⁰))
          + ρ_bal    · L_HINOS-Bal(Q, W_U)
          + ρ_anchor · L_anchor
```

**关键点**：

1. cut 和 bal 的图源从 `W_Π` 切换到 `W_U`，但 W_U 由 anchor 锁在 W̄_Π 附近 → cut 现在作用在"被三方力（cut 自己 + KL via Q + anchor）共同雕刻"的可学图上。
2. **不加 `Tr(H^T L_U H)` 和 ρ_rank 自适应**。SCCD 那条 rank 路径在我们的端到端设置里太激进，容易把 U 推向退化结构。Phase 1 先验证更保守版本。
3. **谱隙作为诊断指标，不入 loss**：每 eval 算 `λ_c / λ_{c+1}` 监控 U 的块结构是否自然形成。如 Phase 1b 所有数据集都没有自然形成谱隙，再考虑加 spec-gap loss（见 §3.3）。

损失项总数：cut + KL + bal + anchor = 4 个，比原 HINOS 多 1 个。

### 3.3 兜底（仅当 Phase 1b 谱隙未自然形成）

如果在 Phase 1b 验证后所有数据集 `λ_c / λ_{c+1} > 0.1`（块结构没出来），再加一个温和的 spec-gap loss：

```
L_spec-gap = Σ_{r=1..c} λ_r(L_U)  −  γ · λ_{c+1}(L_U)
```

但这是兜底，不作为默认设计。**Phase 1b 默认不加**。

### 3.4 Phase 2 新损失：η_align 重加权（叙事 = factorization，架构可用 rhythm/FSTS）

把 W_Π 替换为 `W_Π_η = (η_align · W_Π + (η_align · W_Π)^T)/2`：

- **primitive η_align**（Phase 2a，无参）：`η_ij = count(i,j) / sqrt(deg_i · deg_j)`（已在 `verify_eta_factorization.py` 验证，brain 0.21→0.31）
- **learnable η_align**（Phase 2b）：参数化函数 `η_θ(features_i, features_j, edge_stats_ij)`，端到端可微

`η_θ` 的内部参数化（**只是实现选择，论文叙事不入 abstract**）分三档，按数据集和算力预算选：

| 档位 | 实现 | 输入 | 适用数据集 | 计算开销 | 借鉴 |
|---|---|---|---|---|---|
| **2b-MLP** | `η_θ = MLP([统计特征 of edge ij])` | 共现次数、TPPR 度、recency、事件熵等 | school、低预算场景 | 极低 | 无 |
| **2b-Rhythm** | `η_θ = MLP([Rhythm(T_i) ⊕ Rhythm(T_j) ⊕ φ(Δt_avg)])` | 节点事件时间序列 | dblp、patent、arXivAI/CS/Math/Phy | 中 | SeqFilter Sec 5.2-5.3 的 DepthwiseConv1D + recency-aware |
| **2b-FSTS** | `η_θ = MLP(FSTS_denoise(events_{i,j}))` | 边 (i,j) 的事件序列经 DFT/低通/重加权/IDFT/Conv1D | brain（密集噪声）、有富算力时 | 高 | SeqFilter Sec 6 的 local→global→local 频域 filter |

三档在 abstract / contribution 不区分，**统一表述为"learnable evidence-reliability factor η_align"**。method 章节展开三档实现并对比。这是**架构借鉴 SeqFilter，不是任务对齐 SeqFilter**——SeqFilter 解决 link prediction（"未来谁连谁"），我们解决 community evidence（"哪些事件像同一社区证据"），两者目标完全不同，但 rhythm encoder / FSTS denoiser 这两个组件是任务无关的。

Phase 2 anchor 项相应改为：

```
L_anchor = ‖U − W̄_Π_η‖_F^2
```

η 的 SGD 更新和 U 的 SGD 更新共享 optimizer，端到端反向。**只增加可学参数，不增加损失项**——损失项总数仍是 4 个（cut + KL + bal + anchor）。

### 3.5 与 SCCD 原文的差异

| SCCD 原文 | 本工作 |
|---|---|
| 多层 RWR + β 学习 | 单层 TPPR Π，不学 β |
| H 为 spectral 嵌入并由它给标签 | 标签由 prototype Q 的 argmax 给出，**不引入 H** |
| 显式 rank loss `Tr(H^T L_U H)` | **删除**，谱隙仅作诊断 |
| 静态多视图 | 时序事件流，Phase 2 加 η_align |
| 无 embedding-side label proxy | 保留 DTGC-KL 提供 Z⁰ 标签代理 |

---

## 4. 代码改动清单

### 4.1 新增 `HINOS/sccd_offline.py`（Phase 1a 离线诊断）

不动 trainer。给定 baseline 训好的 Q 和 W_Π，离线优化 U：

```
def fit_offline_U(W_pi_csr, Q_init, c, args):
    """
    输入：baseline 训好的 W_Π（csr）和 Q（n×c numpy）
    输出：训好的 U（csr）和诊断 dict

    损失：
      L = ρ_cut · cut(Q, W_U) + ρ_anchor · ‖U − W̄_Π‖_F^2
      （Q 固定不变，只学 U；KL 和 bal 不参与，因为 Q 不动）

    返回：
      U (csr), spec_gap_c, nmi_spectral_W_U, ncut_pred_W_U,
      ncut_gt_W_U, ncut_pred_over_gt, purity_at_5_W_U
    """
```

输出 CSV `diagnostics/offline_U_{dataset}_{tag}.csv`，每个数据集一行，列对应 baseline vs offline U 的差异。

**Phase 1a 门控（dblp 必过）**：

- `nmi_spectral_W_U ≥ 5 · nmi_spectral_W_Π`（baseline ~0.007，目标 ≥ 0.04）
- `ncut_pred_W_U / ncut_gt_W_U ≥ 0.70`（baseline ~0.58）
- 任意数据集 `purity_at_5_W_U ≥ purity_at_5_W_Π`（不退化）

只有 Phase 1a 通过，才进 Phase 1b 改 trainer。**这是关键的早停 gate**——offline 的 U 都救不了 dblp，端到端只会更难。

### 4.2 新增 `HINOS/sccd_unified.py`（Phase 1b 端到端模块）

```
class UnifiedSimilarity(nn.Module):
    - __init__(pi_cut_csr, init_mode='log_pi')
    - row-softmax(θ, indptr) -> data of W_U
    - forward() -> W_U sparse tensor + per-row degree

def compute_anchor_loss(U_data, pi_bar_data) -> torch.Tensor

def compute_spec_gap_diag(L_U_csr, c) -> {'lambda_c', 'lambda_c1', 'gap_ratio'}
    # 仅诊断，不参与梯度
```

### 4.3 新增 `HINOS/diagnose_unified_alignment.py`（验证脚本）

每 eval 调用：算 `nmi_spectral_W_U`、`ncut_pred_W_U`、`ncut_gt_W_U`、`purity_at_5_W_U`、`spec_gap_c`，写到 `diagnostics/unified_alignment_{dataset}_{tag}.csv`。

### 4.4 修改 `HINOS/trainer.py`

**Phase 1b 改动**（约 80 行）：

- `_prepare_full_ncut_tensors`：保留 W_Π 张量；新增 W_U 计算（从 UnifiedSimilarity 模块）
- `_compute_tppr_cut(assign)` → `_compute_cut(assign, W_data, edge_i, edge_j, degree)`，把 W 从参数传入
- `_compute_hinos_balance(assign)` → 接受 `degree` 和 `two_m` 参数，从 W_U 算
- 新增 `_compute_anchor_loss()`
- `compute_loss_terms`：拼接 `ρ_anchor · L_anchor`
- METRICS_COLUMNS 加列：`loss_anchor`、`weighted_anchor`、`rho_anchor`、`spec_gap_c`、`spec_gap_ratio`、`delta_U_pi_fro`
- `_eval_step`：调 `diagnose_unified_alignment`

**Phase 2 改动**（约 50 行）：

- 在 `_prepare_full_ncut_tensors` 之前，按 `--eta_mode` 重加权 W_Π
- η 参数加入 optimizer

### 4.5 修改 `HINOS/main.py`

```
# Phase 1
--unified_mode {off, on}              default: off
--rho_anchor          float           default: 1.0
--U_init_mode         {log_pi, uniform}     default: log_pi

# Phase 2
--eta_mode {off, primitive, learnable}    default: off
--eta_arch {mlp, rhythm, fsts}            default: mlp   (仅 learnable 时生效)
--eta_dim             int               default: 32
--eta_hist_len        int               default: 64    (rhythm/fsts 用)
--eta_kernel_M        int               default: 4     (rhythm 多尺度基数)
--eta_fsts_tau        float             default: 0.3   (fsts 频域能量阈值)
--eta_clip            float             default: 0.05

# Logging
--log_unified_align   int               default: 1
```

`--unified_mode off` 时所有新代码不触发，行为与改动前一致（回归测试基础）。

### 4.6 修改 `HINOS/run_full.sh`

加 `--phase` 维度，case 切换 `--unified_mode` / `--eta_mode` 组合：

```bash
phase=0      # baseline (回归基线)
phase=1      # unified_mode=on, eta_mode=off
phase=2a     # unified_mode=on, eta_mode=primitive
phase=2b-mlp     # unified_mode=on, eta_mode=learnable, eta_arch=mlp
phase=2b-rhythm  # unified_mode=on, eta_mode=learnable, eta_arch=rhythm
phase=2b-fsts    # unified_mode=on, eta_mode=learnable, eta_arch=fsts
```

---

## 5. 实施阶段与门控

### Phase 0：Baseline（1 天）

- 跑 9 数据集 baseline（`--unified_mode off`，现有 HINOS）
- 跑 `diagnose_prediction_cut.py` 在 dblp/brain/patent/arXivAI 上
- 写 `diagnostics/baseline_metrics_summary.md`：每数据集一行，含 `nmi_argmax_Q`、`acc_argmax_Q`、`nmi_spectral_pi`、`purity@5_pi`、`ncut_gt_pi`、`ncut_pred_pi`、`ncut_pred / ncut_gt`、`spectral_gap_c`

**门控**：表必须填齐。Phase 1 改进无锚点不可信。

### Phase 1a：Offline U 验证（2 天，**关键早停 gate**）

不动 trainer。写 `sccd_offline.py`，对每个数据集：

1. 加载 baseline Q（`*_soft_assign.npy`）和 W_Π（从 cache 重建）
2. 离线优化 `min_U  ρ_cut · cut(Q_fixed, W_U) + ρ_anchor · ‖U − W̄_Π‖²`
3. 报告 §4.1 的诊断指标

**Phase 1a 重新定位（v2）：implementation / gradient feasibility gate，不再做 gt 对齐验证**

原 v1 标准 `nmi_spectral_W_U ≥ 0.04` 设错了。Phase 1a offline 的损失是：

```
min_U  ρ_cut · cut(Q_fixed, W_U) + ρ_anchor · ‖U − W̄_Π‖²
```

它只能证明"U 能让当前 baseline Q 在新图上 cut 更低"，**不能保证 spectral(W_U) 直接对齐 gt**。dblp 的 baseline Q NMI 只有 0.345，offline 阶段要求 `nmi_spectral_W_U ≥ 0.04` 既不符合这个目标也太苛刻。

另外：实测中出现 `ncut_pred_pi = 0` 的列，**不要解释成"塌成一个簇"**。更准确是 spectral NCut 诊断在断开图/多连通分量上不可靠（spectral partition 可能正好落在零 cut 连通分量上）。这一列**暂时不要作为 gate**。

**v2 gate 标准**：

**(A) school 作为 positive sanity case**

- `nmi_spectral_W_U > nmi_spectral_pi`（U 能改善谱结构相对 baseline；不再设绝对阈值）
- `delta_U_pi_fro ≤ 0.30`（U 没漂离 W̄_Π 太远；catch ρ_anchor 过弱导致 U 完全自由的退化情形）

**(B) dblp / patent / arXivAI**

- `ncut_q_W_U / ncut_q_pi ≤ 0.85`（Q 在 W_U 上的 NCut 比在 W_Π 上低 ≥ 15%，证明 cut 梯度有效流过 U）
- `delta_U_pi_fro > 0`（U 真的在动）
- `purity_at_5_W_U` 不明显下降

**(C) brain**

允许 Phase 1a fail，标记为 Phase 2b 的 rhythm/FSTS η_align 目标。Π 局部纯度 0.11 太低，无任何 U 重加权能让 cut 显著下降是预期内的。

按 v2 标准实测：

- school PASS（at ρ_anchor=1.0：nmi_spectral_W_U 0.760 > 0.596，delta_fro 0.200 ≤ 0.30）
- dblp PASS（at ρ_anchor=0.1：ncut_q 比 0.80）
- patent PASS（多档都过）
- arXivAI PASS（at ρ_anchor=0.1：ncut_q 比 0.81）
- brain FAIL，符合预期

**Phase 1a 通过的正确表述**：

> Phase 1a 证明 U 的梯度通路和 anchor-cut 机制可用；真正 paper gate 改到 Phase 1b 端到端训练。

不要说"Phase 1 已经成功"。Phase 1b 才是真正的 gt 对齐检验——它有 DTGC-KL 提供 Z⁰ 标签代理 + Q 端到端进化，弥补 offline 的盲区。

不达标（school 不过）→ **停下来 debug**：
- ρ_anchor 太强 (U ≡ W̄_Π) → 降到 0.1 试
- ρ_cut 太弱（梯度信号小）→ 升到 1.0 试
- row-softmax 数值（梯度死）→ 加温度 τ 调节

### Phase 1b：端到端 Phase 1（3-4 天）

实施次序：

1. 写 `sccd_unified.py` 的 nn.Module 版本
2. 改 trainer.py 的 cut/bal 图源 + anchor loss
3. 改 main.py + run_full.sh
4. **dblp e=10 烟雾测试**：损失曲线、ρ_anchor 配比、U 的 fro 距离收敛性
5. **dblp e=40 全量验证**：与 baseline 对比

**Phase 1b dblp 必过项（v2，软化版）**：

硬 gate（必须满足）：
- `nmi_argmax_s ≥ baseline 0.345`（**不退化是底线**）
- `ncut_gt_W_U < ncut_gt_pi`（gt 标签在 W_U 上的 NCut 比在 W_Π 上低，即 W_U 与 gt 兼容性提高）
- `delta_U_pi_fro ≤ 0.30`（U 在合理范围内移动）
- `purity_at_5_W_U ≥ purity_at_5_pi - 0.02`（不明显退化）

软 gate（涨最好，不强制；offline 的 0.04 不再作硬 gate）：
- `nmi_spectral_W_U` 涨（baseline ~0.0075，Phase 1a 离线只能到 0.008，端到端 Phase 1b 最佳 0.04+ 才算论文级 finding）
- `ncut_pred_W_U / ncut_gt_W_U` 涨（spectral 诊断在断开图上不稳，仅作参考）

**论文叙事**：硬 gate 全过 + 软 gate 涨 → Phase 1b 成功，**这才是论文的真 gate**。不要再说"Phase 1a 已经成功"。

**Phase 1b 跨数据集门控**（dblp 通过后）：

- brain：`nmi_argmax_Q` 不退化（baseline 末期退化到 0.41，Phase 1b 至少持平 warmup ~0.50）
- patent：`nmi_argmax_Q` 提升 ≥ 5%
- arXivAI：不退化

### Phase 2a：Primitive η_align（1-2 天）

- 把 `verify_eta_factorization.py` 的 primitive η 接进 trainer
- 与 Phase 1b 比较：dblp / brain / patent / arXivAI

**Phase 2a 门控**：

- 至少在 brain 上 `nmi_argmax_Q` 比 Phase 1b 提升 ≥ 3%
- 其它数据集不退化

### Phase 2b：Learnable η_align（分三档，4-6 天）

按"代价从低到高、增益从浅到深"顺序推进：

#### Phase 2b-MLP（1-2 天，**所有数据集必跑**）

- 写 `eta_align_module.py` 的 MLP 版本
- 输入：边级统计特征（共现次数、TPPR 度、recency、事件熵）
- 输出：`η_ij ∈ (eta_clip, 1)`
- 是 learnable η 的最低基线，论文里作 `MLP-η` baseline 列

**Phase 2b-MLP 门控**：
- 至少 1 个数据集 `nmi_argmax_Q` 比 Phase 2a primitive 提升 ≥ 2%（证明可学 > 无参）
- 否则 learnable 路径整个停下来 debug：可能 features 选错了，或者 ρ_anchor 锁死了 W_Π_η 的偏移

#### Phase 2b-Rhythm（2-3 天，**关键数据集**）

- 写 `RhythmEncoder`：DepthwiseConv1D + recency-aware（参考 SeqFilter Sec 5.2-5.3，但只用其架构）
- 输入：节点事件时间序列 → `r_i = Rhythm(T_i)`
- η_θ：`MLP([r_i ⊕ r_j ⊕ φ(Δt_avg)])`
- 适用数据集：dblp、patent、brain、arXivAI/CS/Math/Phy（all 6 个 new-pair 主导或节奏强的）
- 不适用：school（小且 warmup 已饱和）、arxivLarge（计算约束）

**Phase 2b-Rhythm 门控（brain 关键）**：
- brain：`nmi_argmax_Q ≥ 0.55`（baseline 退化到 0.41，目标突破 warmup 的 0.50）
- arxiv 系列：相对 Phase 2b-MLP 提升 ≥ 2% NMI（证明 rhythm 是真增量，不是 Adamic-Adar 重新发现）
- dblp：不退化

#### Phase 2b-FSTS（2-3 天，**仅 brain 必跑，其它可选**）

- 写 `FSTSDenoiser`：Sec 6.3-6.5 的 DFT → low-pass mask → 复值频带重加权 → IDFT → Conv1D
- 输入：边 (i,j) 的事件序列（按时间序列编码，长度 padding 到 `eta_hist_len`）
- η_θ：`MLP(FSTS_denoise(events_{i,j}))`
- 适用数据集：brain（密集噪声 + 周期活动），其它有富算力时可加
- **不适用**：arxivLarge（per-edge DFT 不可行）

**Phase 2b-FSTS 门控（brain）**：
- brain `nmi_argmax_Q ≥ Phase 2b-Rhythm 提升 ≥ 2%`
- 不达标 → FSTS 在 brain 上没增量，论文里把它写成 negative result（"频域 denoise 在 community 任务上未带来超出 rhythm 的额外增益"），保留作 ablation

#### Phase 2b 总判定

三档跑完后选**该数据集上最优档**作为论文主表的 "Ours (full)"。论文消融表里把三档都列出来，让读者看到不同复杂度档位的相对收益。

### Phase 3：消融 + 理论 + 全实验（5-7 天）

1. 跑 9 数据集全量（baseline / Phase 1b / Phase 2a / Phase 2b 共 4 档）
2. 消融见 §6
3. 写最小 perturbation 命题：
   > 命题：若 W_U 与真实 c 块相似图 W* 的 Frobenius 误差 ≤ ε，且 L_U 满足 `λ_{c+1} − λ_c ≥ Δ`，则 W_U 谱聚类的子空间与 W* 真实子空间在主角距离意义下 ≤ O(ε/Δ)。
4. 写论文

---

## 6. 消融矩阵

| 配置 | unified | eta_mode | eta_arch | ρ_anchor | ρ_kl | 论文标签 |
|---|---|---|---|---:|---:|---|
| baseline | off | off | n/a | n/a | 5.0 | HINOS (existing) |
| no-anchor | on | off | n/a | 0 | 5.0 | U 完全自由 |
| no-kl | on | off | n/a | 1.0 | 0 | U 无标签代理 |
| **phase 1b full** | on | off | n/a | 1.0 | 5.0 | **Ours (no eta)** |
| primitive eta | on | primitive | n/a | 1.0 | 5.0 | + co-occurrence reweight |
| eta-mlp | on | learnable | mlp | 1.0 | 5.0 | + MLP-η |
| eta-rhythm | on | learnable | rhythm | 1.0 | 5.0 | + Rhythm-η |
| eta-fsts | on | learnable | fsts | 1.0 | 5.0 | + FSTS-η |
| **best of three** | on | learnable | (per dataset) | 1.0 | 5.0 | **Ours (full)** |

**保留**：所有 eta_arch 三档作为消融对比，让读者看到不同复杂度档位的相对收益。
**已删除**：no-rank ablation（rank loss 不再是默认设计）。

预期消融叙事：

- baseline → no-kl：Z⁰ proxy 是关键
- no-anchor → phase 1b full：anchor 必要
- phase 1b full → primitive eta：事件层因子化是补充
- primitive eta → eta-mlp：可学 η 比无参 η 增量
- eta-mlp → eta-rhythm：节奏特征比统计特征强（**对 new-pair 主导数据集**）
- eta-rhythm → eta-fsts：频域 denoise 是否对 brain 这种密集噪声场景再加增量（不一定有效，论文里诚实报告）

---

## 7. 每数据集配置表

| 数据集 | epoch | warmup | ramp | tppr_K | tppr_α | taps_β | ρ_cut | ρ_kl | ρ_bal | ρ_anchor | 推荐 eta_arch | 备注 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|
| school | 40 | 5 | 15 | 5 | 0.2 | 0.5 | 0.5 | 1.0 | 0.5 | 1.0 | mlp | warmup 已饱和，最低档够 |
| dblp | 40 | 10 | 25 | 2 | 0.2 | 0.2 | 0.1 | 5.0 | 1.0 | 1.0 | rhythm | 主验证集；career rhythm 强 |
| brain | 40 | 8 | 25 | 2 | 0.8 | 0.8 | 0.1 | 5.0 | 1.0 | 1.0 | rhythm + fsts | 密集噪声，FSTS 重点候选 |
| patent | 40 | 8 | 25 | 2 | 0.1 | 1.5 | 0.5 | 1.0 | 0.5 | 1.0 | rhythm | 申请节奏强，new-pair 主导 |
| arXivAI | 25 | 5 | 15 | 3 | 0.2 | 0.2 | 0.5 | 1.0 | 0.5 | 1.0 | rhythm | 研究方向节奏 |
| arXivCS | 20 | 5 | 10 | 3 | 0.2 | 0.1 | 0.5 | 1.0 | 0.5 | 1.0 | rhythm | c=40 多类 |
| arxivMath | 20 | 5 | 10 | 3 | 0.2 | 0.1 | 0.5 | 1.0 | 0.5 | 1.0 | rhythm |  |
| arxivPhy | 25 | 5 | 15 | 3 | 0.2 | 0.2 | 0.5 | 1.0 | 0.5 | 1.0 | rhythm |  |
| arxivLarge | 15 | 3 | 8 | 2 | 0.2 | 0.05 | 0.5 | 1.0 | 0.5 | 1.0 | mlp | 见 §8；FSTS 不可行 |

通用：`λ_temp = λ_batch = 0.01`、`λ_com = 1.0`、`prototype_lr_scale = 0.01`（dblp/brain/patent）/ `0.1`（school/arXiv*）；`assign_mode=prototype`、`balance_mode=hinos`、`kl_target_mode=dynamic_tgc`、`batch_recon_mode=ones`、`main_pred_mode=argmax_s`。

---

## 8. arxivLarge 计算策略

arxivLarge n ≈ 1.5M, m ≈ 30M+：

### 8.1 谱隙诊断改 sparse Lanczos

`scipy.sparse.linalg.eigsh(L_U, k=c+1, which='SM', sigma=0)` 求最小 c+1 特征值。每 eval_interval 算一次（≈10 epoch）。预估 30-60s/次。

### 8.2 U 的 sparsity 收紧

`taps_β=0.05` 限 nnz < 100M。如仍 OOM：

- U logits 用 sparse coo + custom backward
- anchor loss 用 sparse Frobenius：`((U.values() − pi_bar.values())**2).sum()`

### 8.3 训练加速

- `batch_size=4096`，混合精度
- `eval_interval=5`、`H_update_interval` 不存在（已删除 H）

### 8.4 兜底

24h 内跑不完一个完整 epoch → 降级到 baseline + Phase 1b only，不上 Phase 2。论文坦白说明。

---

## 9. Logging / Metrics 扩展

### Phase 1b 新增列

```
loss_anchor, weighted_anchor, rho_anchor
spec_gap_c              # λ_c
spec_gap_c1             # λ_{c+1}
spec_gap_ratio          # λ_c / λ_{c+1}（越小说明块结构越清晰）
delta_U_pi_fro          # ‖U − W̄_Π‖_F / ‖W̄_Π‖_F
nnz_U                   # = nnz(W_Π)
purity_at_5_W_U         # 关键改进指标
ncut_gt_W_U
ncut_pred_W_U
ncut_pred_over_gt_W_U   # 关键改进指标（→ 1）
nmi_spectral_W_U        # 关键改进指标（baseline 0.007）
nmi_spectral_topk_W_U
```

### Phase 2 新增列

```
eta_min, eta_median, eta_max
eta_grad_norm
purity_at_5_pi_eta      # η 重加权后的 Π 纯度
ncut_gt_pi_eta
```

### 论文图表数据

每个 dataset/phase 训完自动写：

- `diagnostics/{dataset}_{tag}_summary.json`
- `diagnostics/{dataset}_{tag}_curves.json`（NMI、cut、anchor loss、spec_gap_ratio 随 epoch）

---

## 10. 风险与缓解

| 风险 | 概率 | 影响 | 缓解 |
|---|---|---|---|
| Phase 1a dblp `nmi_spectral_W_U` 不达标 | 中 | 高 | 调 ρ_cut/ρ_anchor；改 anchor 为 KL（U vs W̄_Π 的 row-wise KL）；测 row-softmax 温度 |
| Phase 1b 谱隙不自然形成 | 中 | 中 | §3.3 兜底加温和 spec-gap loss |
| brain Phase 2 仍达不到 0.40 | 中 | 高 | 升档 eta_arch（mlp → rhythm → fsts）；最后考虑 raw_static 作 W_Π |
| arxivLarge OOM | 中 | 中 | §8.4 兜底；arxivLarge 强制 eta_arch=mlp，禁用 rhythm/fsts |
| learnable η 不超 primitive η | 低-中 | 高 | 升档；rhythm 用更长 hist_len；最后 fsts |
| FSTS per-edge DFT 计算开销爆炸 | 中 | 中 | 限定 eta_hist_len ≤ 64；只在 brain/小规模数据上用；其它降级 rhythm |
| 三档 eta_arch 没有显著差异 | 低-中 | 中（消融叙事弱） | 论文里诚实写"在某数据集上 mlp 已足够，rhythm/fsts 增益边际" |
| 实验时间不够 | 中 | 中 | 9 数据集分批：4 个先验证，5 个论文写作期间补 |

---

## 11. 时间线（理想）

| 周 | 工作 |
|---|---|
| W1 (3 天) | Phase 0 baseline 全跑 + 9 数据集 baseline 表 |
| W1 (2 天) | Phase 1a offline U 验证（dblp 必过 gate） |
| W2 (5 天) | Phase 1b 端到端 + dblp / brain / patent / arXivAI |
| W3 (5 天) | Phase 2a primitive η + Phase 2b-MLP |
| W4 (5 天) | Phase 2b-Rhythm（dblp/brain/patent/arXiv*）+ Phase 2b-FSTS（仅 brain） |
| W5 (5 天) | Phase 3 消融 + 理论命题 + arXiv 五个数据集 |
| W6-W7 (10 天) | 论文写作 |
| W8 (3 天) | revision + buffer |

---

## 12. 关键决策（不要再改）

1. 核心方法是 TPPR + cut + 端到端，不丢
2. 不做多视图（单 U，不学层权 β）
3. **论文叙事不主推 SeqFilter / rhythm / FSTS**：abstract 和 contribution list 只讲 `η_align` factorization；rhythm/FSTS 是 method 章节里 `η_θ` 的具体参数化档位
4. **架构层可以借 SeqFilter 的 Rhythm Encoder 和 FSTS Denoiser**：作为 Phase 2b 的 eta_arch 实现选项；Phase 2b 三档 mlp/rhythm/fsts 全做消融对比，但写论文时强调"factorization 为骨架，rhythm/fsts 为可替换组件"
5. 保留 prototype Q + DTGC-KL，作标签代理
6. **不引入 spectral H 矩阵作显式 rank loss**：Phase 1b 默认只加 anchor loss
7. **谱隙作诊断指标，不入 loss**（除非 §3.3 兜底触发）
8. W_U 改作 cut 和 bal 的实际作用图，W_Π 保留作初始化和诊断
9. U 的 sparsity pattern 固定为 `support(W_Π)`：不引入新边
10. 默认参数化用 row-softmax，不显式 simplex projection
11. **Phase 1a offline 验证是早停 gate**：不通过不进端到端
12. ~~不做 fsts negative ablation~~ → **改为 fsts 作为正向 ablation 档位**：跑 fsts 但不主打"FSTS 重要"，只在 brain 等密集噪声场景作为最高复杂度档展示

---

## 13. 一句话总结（论文 abstract 种子）

> Existing temporal graph clustering optimizes a normalized cut on a fixed TPPR similarity Π that is constructed independently of the clustering objective. We empirically show this yields cut-optimal partitions structurally misaligned with semantic labels: even on label-coherent datasets like DBLP, the cut-optimum has 42% lower NCut than the ground-truth partition yet only NMI=0.35 with it; and on event-noisy datasets like brain, Π's local label coherence caps at 0.21 across the full TPPR hyperparameter grid. We propose to learn a TPPR-anchored unified similarity matrix U whose spectral cut structure is aligned with semantic clustering through a Frobenius anchor to a normalized Π and a joint cut + embedding-side label proxy. Optionally, we further factorize the temporal transition into connectivity × evidence-reliability η at the event level, with η parameterized by a small MLP over edge statistics or by deeper temporal encoders adapted from prior work as needed. We prove a perturbation bound under a spectral gap condition, and show consistent NMI gains across 9 temporal graph datasets including the previously-failing brain dataset.

---

## 14. 文件改动检查清单

### Phase 1a（offline 验证）

- [ ] 新增 `HINOS/sccd_offline.py`（约 200 行）
- [ ] 用 baseline 输出（`*_soft_assign.npy`）跑 9 数据集
- [ ] 写 `diagnostics/offline_U_summary.md`

### Phase 1b（端到端）

- [ ] 新增 `HINOS/sccd_unified.py`（约 150 行）
- [ ] 新增 `HINOS/diagnose_unified_alignment.py`（约 120 行）
- [ ] 修改 `HINOS/trainer.py`：约 80 行
- [ ] 修改 `HINOS/main.py`：约 10 行
- [ ] 修改 `HINOS/run_full.sh`：加 phase 维度

### Phase 2a

- [ ] 把 `verify_eta_factorization.py` 的 primitive η 接进 trainer（约 30 行 hook）

### Phase 2b-MLP

- [ ] 新增 `HINOS/eta_align_module.py`：MLP 版本（约 80 行）
- [ ] 修改 `HINOS/trainer.py`：约 50 行（η hook + optimizer）
- [ ] 修改 `HINOS/main.py`：约 10 行（`--eta_arch mlp` 默认）

### Phase 2b-Rhythm

- [ ] 在 `eta_align_module.py` 加 `RhythmEncoder` 类（约 100 行）
  - DepthwiseConv1D + recency-aware（参考 SeqFilter Sec 5.2-5.3 架构）
  - 节点事件时间戳 → r_i embedding
- [ ] 在 trainer 里新增节点事件序列预处理（约 30 行）
  - 每节点缓存历史事件时间戳（限 `eta_hist_len` 长度）
- [ ] 切换 `--eta_arch rhythm`

### Phase 2b-FSTS

- [ ] 在 `eta_align_module.py` 加 `FSTSDenoiser` 类（约 150 行）
  - DFT → low-pass mask → 复值频带重加权 → IDFT → Conv1D
  - 输入 per-edge 事件序列（pad 到 `eta_hist_len`）
- [ ] 切换 `--eta_arch fsts`
- [ ] **仅在 brain 上必跑**，其它数据集可选

### Phase 3

- [ ] 新增 `HINOS/run_ablations_v2.sh`
- [ ] 写 `docs/sccd_adaptation_recovery_proposition.md`（理论命题）

---

## 15. 启动顺序

1. 读完本文件
2. **跑 §5 Phase 0 的 baseline 全套**（9 数据集 × 现有 HINOS）
3. 把结果写到 `diagnostics/baseline_metrics_summary.md`
4. **停下来给我看这张表**，再开始 Phase 1a
5. **Phase 1a offline 验证**：dblp 必过 gate；不通过不进 Phase 1b
6. Phase 1a 通过后才开始改 trainer.py（Phase 1b）

不要在 baseline 表完成之前就开始改 trainer。不要在 Phase 1a 通过之前就开始 Phase 1b。这两个 gate 是论文叙事的锚点，缺一个都会让后续工作变成无效实验。

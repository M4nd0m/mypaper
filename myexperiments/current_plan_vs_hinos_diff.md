# `current_plan_tppr_ncut.md` 与当前 `HINOS` 项目的差异整理

## 1. 文档目的

本文档用于对照：

- 我的方案说明文档：`myexperiments/current_plan_tppr_ncut.md`
- 当前代码实现：`HINOS/README.md`、`HINOS/main.py`、`HINOS/trainer.py`、`HINOS/sparsification.py`、`HINOS/data_load.py`

目标不是判断谁对谁错，而是明确：

1. 两者在哪些层面是一致的；
2. 哪些地方只是叙述抽象程度不同；
3. 哪些地方已经出现了真正的实现级偏差。

---

## 2. 总体结论

结论可以概括为一句话：

> `current_plan_tppr_ncut.md` 与当前 `HINOS` 项目在“研究主线”上基本一致，但在若干关键数学细节和实现细节上并未完全对齐。

更具体地说：

- **一致的是主骨架**：offline、直接优化 embedding、temporal loss、TPPR-based community objective、batch reconstruction。
- **不完全一致的是细节定义**：TPPR 的输入构图过程、Ncut 正则形式、temporal 项中的时间衰减写法。

因此，当前 `current_plan_tppr_ncut.md` 更适合作为“研究方向说明”；
如果要把它当作“严格对应当前代码的技术说明”，则仍需修订。

---

## 3. 明确一致的部分

以下部分在你的方案文档与当前 `HINOS` 代码中是明显一致的。

### 3.1 都只讨论 `offline` 主线

当前 `HINOS/README.md` 明确写的是：

- 仅保留原始 HINOS 的 offline training pipeline
- 移除了 query-aware community search、BFS candidate generation、top-k evaluation 等在线 search 部分

这和 `current_plan_tppr_ncut.md` 中“紧贴师兄 offline 主线”的定位是一致的。

### 3.2 都采用“直接优化 embedding”的训练范式

你的文档中明确写到：

- 先得到初始 embedding
- 再把 embedding 作为优化变量直接更新

代码中也确实如此：

- `trainer.py` 里 `self.node_emb = nn.Parameter(...)`
- 训练时直接把 `node_emb` 放进优化器

因此，“embedding as optimization variable” 这一点是完全对齐的。

### 3.3 都保留三类核心损失/目标

从整体结构看，两边都保留了：

1. temporal loss
2. community objective on TPPR graph
3. batch/local reconstruction loss

`HINOS/README.md` 里保留项写得很直接：

- reconstruction loss
- temporal loss
- full-graph NCut loss
- offline embedding optimization

这和你文档中的“temporal loss + TPPR-Ncut + batch reconstruction”的主干是一致的。

### 3.4 都没有走 stage-wise 多阶段显式聚类变量路线

你的文档明确说当前不优先引入：

- 多阶段表示矩阵
- 多阶段软分配矩阵
- 跨阶段 assignment consistency

当前代码也没有这些结构。现有实现是单一的：

- 一个全局 `node_emb`
- 一个 `cluster_mlp`
- 一个全局 TPPR 图
- 一个全图 NCut 损失

所以“先不另起一套多阶段框架”的策略与当前项目是一致的。

---

## 4. 关键差异

下面这些不是“表述角度不同”这么简单，而是已经到了实现定义层面的差异。

### 4.1 TPPR 的输入不是“直接原始事件流”，而是先经过 TAPS 稀疏化

#### 文档中的写法

`current_plan_tppr_ncut.md` 中的叙述更像是：

- 对原始事件流计算 TPPR
- 得到高阶时序相似图 `Pi`

这种写法在概念上没有问题，但比较抽象，读起来容易理解成“直接在原始 temporal graph 上做 TPPR”。

#### 当前代码中的做法

当前 `HINOS` 不是直接从原始事件流一步得到 TPPR，而是：

1. 读取 temporal edges
2. 构造 time-aware path sparsifier
3. 用 `sqrt(static_edges)` 预算做 TAPS sampling
4. 先得到稀疏化后的时序图 `A_time`
5. 再在该图上构造 TPPR
6. 将 TAPS 和 TPPR 都缓存到 `cache/`

也就是说，现实现实中是：

> 原始事件流 -> TAPS 稀疏化图 -> TPPR -> 对称化后 NCut 图

#### 影响

因此，你当前文档里“TPPR 高阶相似图”的那一节，作为抽象层描述是成立的；
但如果要严格对应当前代码，应该明确写出：

- TPPR 不是直接从原始图算出来的；
- 它是建立在 TAPS 稀疏化结果上的；
- 代码中还有缓存逻辑。

---

### 4.2 社区损失中的正则项不是文档里的 balance penalty，而是 orthogonality penalty

#### 文档中的写法

在 `current_plan_tppr_ncut.md` 中，社区损失被写为：

`L_cut = L_cut_base + lambda_p * L_bal`

其中额外项是一个 balance penalty，强调避免簇塌缩、保持簇间平衡。

#### 当前代码中的做法

当前 `trainer.py` 的 `full_ncut_loss()` 里，实际计算的是：

1. 先用 `cluster_mlp(node_emb)` 得到 soft assignment `H`
2. 基于全图边和度矩阵构造 `L_mat` 与 `G`
3. 计算类似 trace 形式的 NCut 主项 `Lc`
4. 再加入一个 orthogonality regularizer `Lo`
5. 最后返回 `Lc + lambda_ncut_orth * Lo`

因此，代码实现的附加项不是你文档中的 balance penalty，而是：

> 用正交约束逼近簇分配矩阵的规范化结构

#### 影响

这意味着：

- 你的文档目前表达的是“一种你想采用的 Ncut 正则形式”；
- 当前代码实现的是“另一种更接近正交约束的正则形式”。

两者的目标都和防塌缩、避免退化解有关，但并不是同一个数学对象。

如果你后续论文要严格对应当前代码，这一节必须改写。

---

### 4.3 temporal loss 中的时间项符号与文档不一致

#### 文档中的写法

你的文档把 temporal 项写成：

`exp(-delta_t * (t - t_x))`

这表达的是一个很标准的“随时间差增大而衰减”的历史影响机制。

#### 当前代码中的做法

当前 `trainer.py` 中实际写的是：

- `d_time = abs(t_times.unsqueeze(1) - h_times)`
- `torch.exp(delta * d_time)`

注意这里指数前是 **正号**，不是负号。

并且：

- `delta` 初始值是全 1
- 代码里没有把它强制成负数

这意味着按当前实现，历史时间差越大，对应项可能越大，而不是越小。

#### 影响

这是当前最需要特别警惕的一处差异，因为它不是“叙述抽象 vs 代码细节”的问题，而是：

> 文档表达的是“时间衰减”，代码实现看起来更像“时间放大”。

这里至少有三种可能：

1. 文档是你想要的方法，代码这里还没改对；
2. 代码是继承来的写法，但符号与原始理论表述不一致；
3. 代码里 `delta` 被设计为学习到负值，只是初始化时为正，但这一点当前实现并没有保证。

无论是哪一种，这一项都值得单独核查。

---

### 4.4 文档里的相似度写法比代码更统一，代码实际上混用了不同相似度

#### 文档中的写法

文档里：

- temporal 部分主要用负欧氏距离平方来写 `mu(u,v,t)`
- reconstruction 部分用通用的 `sim(u,v,t)` 记号

整体阅读上比较统一。

#### 当前代码中的做法

代码里其实混用了两种相似度：

- reconstruction loss 用的是 `cosine_similarity`
- temporal intensity 里的 `mu`、`alpha` 用的是负平方距离

也就是说，当前代码并不是“所有地方都围绕同一个相似度定义展开”。

#### 影响

这不一定是错误，但它说明：

- 你的文档更偏“方法层叙述”
- 当前实现更偏“工程上分项选不同度量”

如果后续要严格写论文方法，需要决定：

- 是继续保留这种混合设计；
- 还是在文字上把 reconstruction 与 temporal 模块的相似度定义分开写清楚。

---

### 4.5 当前代码的 NCut 是显式的 full-graph NCut

#### 文档中的写法

你的文档写的是：

- 在 TPPR 相似图上施加 Ncut 型社区目标

这本身没错，但没有特别强调损失是“全图范围”计算。

#### 当前代码中的做法

代码和 README 都明确使用的是：

- `full-graph NCut loss`

也就是说，虽然训练样本通过 batch 喂给 temporal/reconstruction 项，但社区项本身是全图的。

#### 影响

这不是原则性冲突，只是文档精度还可以更高。

如果要更贴近代码，建议在文档中明确写成：

> temporal/reconstruction 是 batch 驱动的，而 NCut 项是基于全图 TPPR affinity 计算的。

---

## 5. 哪些差异只是“抽象程度不同”

并不是所有不一样的地方都需要修改。下面这些更像“文档概念层更高，代码实现层更具体”。

### 5.1 文档写的是研究主线，代码写的是工程管线

例如：

- 文档讲“TPPR 图”
- 代码讲 “TAPS -> TPPR -> cache -> symmetrize”

这更多是抽象层级差异，不一定构成冲突。

### 5.2 文档强调方法定位，代码强调可运行实现

例如：

- 文档会讨论“是否引入 stage-wise 变量”
- 代码则直接给出实际使用的参数、缓存路径、输出文件

这也不是矛盾，而是关注重点不同。

---

## 6. 哪些差异需要优先处理

从后续研究和写论文的角度，优先级最高的差异有三项。

### 第一优先级：temporal 时间项符号

这是当前最需要核查的问题。

原因：

- 文档是时间衰减；
- 代码看起来像时间放大；
- 这会直接影响 temporal 建模的物理含义。

### 第二优先级：Ncut 正则项定义

如果你后续要在“改 Ncut”这个方向上推进，那么必须先明确：

- 当前 baseline 到底是 balance penalty 版本；
- 还是 orthogonality regularization 版本。

否则后续改进目标会漂移。

### 第三优先级：TPPR 前是否显式写出 TAPS

如果你的论文/方案文档想准确承接当前代码，那么建议把：

- TAPS 稀疏化
- TPPR 缓存
- 全图 NCut 图构造

这三个步骤补进方法说明中。

---

## 7. 对你当前研究定位的意义

这份差异对你当前研究的意义是：

### 7.1 你的方案主线没有偏

从总体方向上看，你现在的 `current_plan_tppr_ncut.md` 与当前 `HINOS` 项目是对得上的：

- 以师兄 offline 为主干
- 不另起一套复杂多阶段框架
- 重点围绕 TPPR / Ncut 做最小增强

这个判断是成立的。

### 7.2 但如果要把当前代码当作“精确 baseline”，文档还不够精确

尤其是下面三件事必须统一：

1. temporal 项到底是不是衰减
2. Ncut 的附加正则到底是哪一种
3. TPPR 前面是否显式承认 TAPS 稀疏化

### 7.3 因此，当前文档更适合做“研究计划”，不适合直接当“实现说明”

现阶段最准确的定位应该是：

> `current_plan_tppr_ncut.md` 是一个与当前代码主线一致、但尚未逐式对齐实现细节的研究计划文档。

---

## 8. 建议的后续动作

建议按下面顺序处理。

### 8.1 先核查 temporal 项符号

优先确认：

- `torch.exp(delta * d_time)` 是否真的是你想保留的实现
- 还是应该改成 `torch.exp(-delta * d_time)`

### 8.2 再决定 baseline 的 Ncut 版本

明确：

- 你要以当前代码里的 orthogonality regularization 作为 baseline
- 还是要回到你文档里的 balance penalty 版本

### 8.3 最后再修正文档

如果你要让计划文档与代码严格一致，建议后续再出一版：

- 保留当前研究定位
- 但把 TPPR/TAPS、full-graph NCut、temporal 项公式逐项改成实现对应版本

---

## 9. 最后总结

一句话总结：

> 你的 `current_plan_tppr_ncut.md` 和当前 `HINOS` 代码在研究主线层面是一致的，但在 TPPR 构图流程、Ncut 正则形式、temporal 时间项符号这三处存在必须明确的实现级差异。

因此，当前最合理的判断不是“完全一致”或“完全不一致”，而是：

> **方向一致，细节未完全对齐。**

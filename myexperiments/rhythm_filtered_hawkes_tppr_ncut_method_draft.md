# 方法草稿（详细版）
## Rhythm-Filtered Hawkes with Evidence-Aware TPPR-Ncut

---

## 0. 文档定位

这份草稿的目标，不是重新搭一个完全不同的时序图聚类框架，而是：

- **保留当前 offline 主线**：初始化 embedding → 直接优化 embedding → temporal loss + TPPR-Ncut + batch reconstruction；
- **结合 WWW 2026 的 SeqFilter 思想**，但不照搬其 link prediction 模型；
- 将其真正有价值的处理手段——**时间节律建模**与**历史结构证据过滤**——迁移到我们自己的时序图聚类问题上；
- 使方法叙事从“怎么处理某种特殊变化”转向更根本的问题：

> **在 event-sequence 场景、且 repeated pairs 较弱时，哪些历史时序信息应当被视为 community evidence，并进入聚类优化？**

这也是本方案相较于单纯沿用 Hawkes、单纯沿用 TPPR、或者直接套用 SeqFilter 的根本区别。

---

## 1. 问题重述

### 1.1 当前真正要解决的不是“时序图能不能聚类”

已有工作已经说明，时序图上做聚类并不是空白任务。真正尚未说清的问题是：

> **当 pair-level recurrence 较弱时，社区证据究竟来自哪里？**

在静态图里，社区证据通常直接来自固定图结构的“内密外疏”。
但在 event-sequence 时序图里：

- 同一对节点不一定反复交互；
- 当前事件可能更多受顺序依赖驱动，而不是固定 pair 复用驱动；
- 社区证据未必体现为 repeated edges，而可能体现为更高层次的时间规律与局部结构持续性。

因此，我们当前的问题，不再是简单地让历史影响未来，而是要回答：

> **哪些历史信息对社区有解释力，哪些历史只是表面事件波动。**

### 1.2 为什么只靠原始 Hawkes 不够

在现有方案中，temporal loss 主要通过 Hawkes 风格的历史累加项来定义。这种做法默认：

- 过去事件可以通过时间衰减核持续影响当前事件；
- 只要时间上还没衰减到足够小，历史就应继续参与当前强度计算。

但在 low-repeat 的 event-sequence 数据中，这一默认假设过强。因为：

1. 并不是所有历史都具有 community evidence；
2. 单一指数衰减只是在表达“多久以前发生”，但没有表达“这段历史在社区意义上是否可信”；
3. 历史邻居的累加项中，混入了大量可能只是短期事件波动的交互。

因此，标准 Hawkes 的问题不是“有历史项”，而是：

> **历史项缺少 evidence filtering。**

### 1.3 为什么只改 TPPR 也不够

TPPR 的作用是把时序交互关系进一步传播成高阶 affinity。如果进入 TPPR 的历史本身就已经混杂了大量非社区证据，那么高阶扩散只会把这些不够可靠的影响进一步传播出去。

因此，从因果顺序上看：

- **先**要在 temporal loss 中区分哪些历史值得继承；
- **后**才谈得上这些历史如何通过 TPPR 形成更可靠的高阶社区结构。

这也是为什么本方案的第一落点应该是 **Hawkes 历史项重构**，而不是先改 cut objective。

---

## 2. 从 SeqFilter 借什么，不借什么

WWW 2026 的 SeqFilter 面向的是 temporal link prediction，而不是 clustering。但它对我们非常有启发，因为它把传统时序图模型中“历史信息如何进入模型”这件事重新做了拆解。

### 2.1 借什么

我们主要借用 SeqFilter 的两条核心思想：

#### （1）时间节律优先于邻居身份记忆
SeqFilter 认为，在 repeated pairs 很弱时，单纯记住“和谁交互过”并不能很好地概括有意义的历史模式；相比之下，节点**在什么时候发生交互**，更可能反映其行为节律。

因此，它用 **node rhythm memory** 来建模绝对时间轴上的节律，并加入 recency-aware 状态。

这对我们的启发是：

> **节点的时间节律可以作为历史是否值得继承的重要判据。**

#### （2）历史邻域不应原样进入结构模块，而应先经过 evidence filter
SeqFilter 的 FSTS encoder 不是无差别聚合所有历史邻域，而是先做频率选择，再做局部结构学习。

这对我们的启发是：

> **历史邻域中的事件，并不都等价地构成社区证据。应先过滤，再传播。**

### 2.2 不借什么

我们不直接照搬以下内容：

1. **不照搬 SeqFilter 的最终 link prediction 头。**
   因为我们的目标是 clustering，不是未来边预测。

2. **不把整个模型改造成 online memory-based T-GNN。**
   我们仍保持“embedding 作为优化变量”的 offline 主线。

3. **不直接继承它关于 Wiener 最优去噪器的理论结论。**
   因为那个理论依赖其原始结构 encoder 的频域设定；迁移到我们的 Hawkes + TPPR-Ncut 框架后，最多借其过滤思想，而不能直接声称保留同样理论。

因此，我们的迁移方式应当是：

> **把 SeqFilter 改造成一个 community-evidence filter，而不是替换当前框架。**

---

## 3. 总体框架

### 3.1 数据定义

给定 event-sequence 形式的连续时间时序图：

$$
\mathcal{G} = (V,E), \qquad
E = \{(u_r,v_r,t_r, e_r)\}_{r=1}^{M}, \quad t_1 < t_2 < \cdots < t_M.
$$

其中：

- $V$ 为节点集合；
- 每条事件 $(u_r,v_r,t_r,e_r)$ 表示节点 $u_r$ 与 $v_r$ 在时间 $t_r$ 发生一次交互；
- $e_r$ 表示可选的事件特征。

节点初始表示记为：

$$
Z^{(0)} \in \mathbb{R}^{|V|\times d}.
$$

例如，可由 node2vec 或其他轻量预训练方式得到。

训练中被优化的是：

$$
Z \in \mathbb{R}^{|V|\times d}.
$$

### 3.2 方法总览

本方法由四部分组成：

1. **Rhythm Encoder**：从节点的历史事件时间序列中提取节律表示；
2. **Rhythm-Filtered Hawkes**：用节律表示控制 Hawkes 历史项，筛选更像 community evidence 的历史；
3. **Evidence-Aware TPPR**：将筛选后的历史证据前移到 TPPR 的时序传播中，构造高阶 affinity；
4. **TPPR-Ncut + Batch Reconstruction**：在更可靠的 affinity 上做社区划分，同时保留局部结构。

其总目标函数写为：

$$
L = L_{\mathrm{temp}} + \lambda_{\mathrm{cut}}L_{\mathrm{cut}} + \lambda_{\mathrm{batch}}L_{\mathrm{batch}}.
$$

与原方案相比，最核心变化不在于总损失形式，而在于：

> **我们重新定义了历史如何进入 $L_{\mathrm{temp}}$，以及这些更可信的历史如何进入 TPPR。**

---

## 4. 模块一：节点节律编码（Rhythm Encoder）

### 4.1 设计动机

在 low-repeat 场景下，单纯记“哪个邻居出现过”并不能稳定刻画节点历史模式。相比之下，节点何时发生交互，往往更稳定地反映其行为组织规律。

例如，一个节点可能不断更换交互对象，但其交互行为仍呈现：

- 某种周期性；
- 某种多尺度时间分布；
- 某种长期-短期混合的活跃模式。

这些模式未必直接说明社区，但它们可以帮助判断：

> **一段历史与当前状态是否协调，从而是否值得继续作为 community evidence。**

### 4.2 节律输入

对每个节点 $i$，收集其截至时间 $t$ 的历史交互时间序列：

$$
\mathcal{T}_i(t)=\{t_{i,1},t_{i,2},\dots,t_{i,m_i(t)}\}, \quad t_{i,1}<\cdots<t_{i,m_i(t)}<t.
$$

我们只用这些时间戳构造节律表示，而不直接输入邻居身份。

### 4.3 历史节律表示

我们定义：

$$
r_i^{\mathrm{hist}}(t)=\mathrm{RhythmEnc}(\mathcal{T}_i(t)).
$$

`RhythmEnc` 可实现为以下任一轻量结构：

- depthwise 1D convolution over time encodings；
- 多尺度时间基函数聚合；
- 频域特征提取后接 MLP；
- 绝对时间编码序列上的轻量 Conv/MLP Mixer。

为了与 SeqFilter 保持思想一致，推荐使用“**时间编码 + depthwise Conv**”的实现。

设时间编码函数为 $\phi(\cdot)$，则可写为：

$$
H_i(t)=\big[\phi(t_{i,1}),\phi(t_{i,2}),\dots,\phi(t_{i,m_i(t)})\big],
$$

$$
r_i^{\mathrm{hist}}(t)=\mathrm{Pool}\big(\mathrm{DepthwiseConv1D}(H_i(t))\big).
$$

### 4.4 最近活跃状态融合

为了避免只建长期节律而忽略“当前节点离最近一次交互有多久”，引入 recency-aware 状态。

记节点 $i$ 最近一次交互时间为：

$$
t_i^{\mathrm{last}}(t)=\max\{\tau\in \mathcal{T}_i(t)\}.
$$

定义 recency 编码：

$$
c_i^{\mathrm{rec}}(t)=\phi\big(t-t_i^{\mathrm{last}}(t)\big).
$$

最终节律表示为：

$$
r_i(t)=\mathrm{MLP}\big(r_i^{\mathrm{hist}}(t) \oplus c_i^{\mathrm{rec}}(t)\big).
$$

这个表示的意义是：

- $r_i^{\mathrm{hist}}(t)$ 描述节点长期时间行为模式；
- $c_i^{\mathrm{rec}}(t)$ 描述其当前是否仍处于该模式的活跃阶段。

---

## 5. 模块二：Rhythm-Filtered Hawkes

### 5.1 原始 Hawkes 形式

当前方案中的 temporal loss，本质上可以理解为以事件强度为基础的正负样本对比优化。设当前目标事件为 $(u,v,t)$，原始得分函数写为：

$$
s(u,v,t)
=
\mu(u,v,t)
+
\sum_{x\in N_{u,t}} w(x,u,t)\,\mu(x,v,t)\,\exp(-\delta(t-t_x)),
$$

其中：

- $N_{u,t}$ 为节点 $u$ 在时间 $t$ 之前的历史相关节点集合；
- $t_x$ 为与历史项 $x$ 对应的交互时间；
- $\mu(\cdot)$ 是基于当前 embedding 的相似度函数；
- $\exp(-\delta(t-t_x))$ 为固定指数时间衰减核。

问题在于：

1. 所有历史都被按同一核形式继承；
2. 时间衰减只考虑“离现在多久”，不考虑“这段历史是否像社区证据”；
3. 历史贡献中缺少对节点当前节律状态的条件化。

### 5.2 节律条件化的多尺度时间核

为此，我们将固定指数核替换为节点条件化的多尺度时间核。

设一组时间基函数为：

$$
\{\psi_1(\Delta t),\psi_2(\Delta t),\dots,\psi_M(\Delta t)\}.
$$

例如可选：

- 指数基：$\exp(-\beta_m\Delta t)$；
- 高斯基；
- piecewise basis；
- spline basis。

定义节点 $u$ 在时间 $t$ 的核混合系数：

$$
\pi_u(t)=\mathrm{softmax}(W_r r_u(t)+b_r) \in \mathbb{R}^{M}.
$$

于是得到节律条件化时间核：

$$
\kappa_u(\Delta t\mid r_u(t))
=
\sum_{m=1}^{M}\pi_{u,m}(t)\,\psi_m(\Delta t).
$$

它的含义不是简单地“某个节点记忆更长/更短”，而是：

> **当前节点在当前节律状态下，更愿意从哪些时间尺度的历史中提取证据。**

### 5.3 节律一致性门控

即便时间尺度合适，某条具体历史也未必和当前状态协调。因此，我们进一步定义节律一致性门控：

$$
g_{u,x}^{(r)}(t)
=
\sigma\Big(
\mathrm{MLP}_r\big(r_u(t)\oplus r_x(t)\oplus \phi(t-t_x)\big)
\Big),
$$

其中：

- $r_u(t)$ 表示当前节点 $u$ 的节律状态；
- $r_x(t)$ 表示历史节点 $x$ 的节律状态；
- $\phi(t-t_x)$ 表示这条历史距离当前的时间差。

$g_{u,x}^{(r)}(t)\in(0,1)$ 用来表示：

> **从时间节律角度看，这条历史是否值得继承。**

### 5.4 结构证据门控（为后续 FSTS 做准备）

仅靠时间节律仍不够，因为某条历史即便在时间上协调，也不一定在邻域结构模式上构成社区证据。因此进一步定义结构证据门控 $g_{u,x}^{(s)}(t)$，具体将在第 6 节由 FSTS 产生。

### 5.5 改进后的 Hawkes 得分函数

最终，当前事件 $(u,v,t)$ 的得分函数写为：

$$
s(u,v,t)
=
\mu(u,v,t)
+
\sum_{x\in N_{u,t}}
 g_{u,x}^{(r)}(t)
 g_{u,x}^{(s)}(t)
 w(x,u,t)
 \mu(x,v,t)
 \kappa_u(t-t_x\mid r_u(t)).
$$

其中，若在初期实验中不加入 FSTS，可暂时令：

$$
g_{u,x}^{(s)}(t)=1.
$$

### 5.6 Temporal loss

继续使用事件级正负样本优化：

$$
L_{\mathrm{temp}}
=
-\log\sigma\big(s(u,v,t)\big)
-\sum_{n\sim P_n}\log\sigma\big(-s(u,n,t)\big).
$$

这里的核心变化不是 loss 形式，而是：

> **历史项已从“统一衰减累加”改为“节律与结构共同筛选后的证据累加”。**

---

## 6. 模块三：Frequency-Selective Structure Evidence Filter

### 6.1 设计动机

在 repeated pairs 较弱时，仅用时间节律还不能充分区分：

- 哪些历史只是偶然交互；
- 哪些历史反映节点所处局部结构的持续性；
- 哪些历史更可能与社区层 evidence 有关。

SeqFilter 的 FSTS 给出的启发是：

> **历史邻域应先经过结构模式过滤，再进入后续模块。**

在我们的框架中，这个模块不再直接输出最终节点表示，而是输出：

> **某段历史在结构上是否像 community evidence 的评分。**

### 6.2 历史邻域窗口构造

对于当前节点 $u$ 和时刻 $t$，收集其最近 $K$ 个历史相关事件，按时间顺序排列：

$$
\mathcal{X}_{u,t}=\{x_u^{(1)},x_u^{(2)},\dots,x_u^{(K)}\}.
$$

其中每个 token 定义为：

$$
x_u^{(\ell)}
=
 e_{u j_\ell}
 \oplus \phi(t-t_\ell)
 \oplus z_{j_\ell}^{(0)}.
$$

这里：

- $e_{u j_\ell}$ 为事件特征；
- $\phi(t-t_\ell)$ 为相对时间编码；
- $z_{j_\ell}^{(0)}$ 为邻居初始表示。

将其堆叠为矩阵：

$$
X_{u,t}\in\mathbb{R}^{K\times d_x}.
$$

### 6.3 频率选择式过滤

为了识别历史邻域中的主要结构模式，我们仿照 SeqFilter 的 local-global-local 思想，对 $X_{u,t}$ 做三阶段过滤。

#### （1）局部频率筛选
先将序列变到频域：

$$
\widehat{X}_{u,t}=\mathrm{DFT}(X_{u,t}).
$$

再通过可学习阈值或 soft mask 保留主要频率成分：

$$
\widehat{X}^{\mathrm{low}}_{u,t}
=
M_{u,t}\odot \widehat{X}_{u,t},
$$

其中 $M_{u,t}\in[0,1]^{K\times d_x}$ 为频率掩码，可由：

$$
M_{u,t}=\sigma\big(\mathrm{MLP}_m(|\widehat{X}_{u,t}|)\big)
$$

产生。

这一步的目的不是强调“异常”或“突发”，而是：

> **先过滤掉不稳定、解释性弱的局部频率分量，保留更可能代表结构持续性的频率模式。**

#### （2）全局频带重加权

对筛过的频谱施加复值或实值可学习映射：

$$
\widehat{X}^{\mathrm{proj}}_{u,t}
=
\widehat{X}^{\mathrm{low}}_{u,t} W_c,
$$

或更简单地：

$$
\widehat{X}^{\mathrm{proj}}_{u,t}
=
\Gamma\odot \widehat{X}^{\mathrm{low}}_{u,t},
$$

其中 $\Gamma$ 是可学习频带增益。

这一步的意义是：

> **不同数据、不同节点、不同时间窗口，真正有用的结构时间尺度并不一样，因此需要自适应调节。**

#### （3）时域局部相关学习

将其变回时域：

$$
X^{\mathrm{proj}}_{u,t}=\mathrm{IDFT}(\widehat{X}^{\mathrm{proj}}_{u,t}),
$$

并用 Conv1D 学习局部模式：

$$
H^{\mathrm{str}}_{u,t}=\mathrm{Conv1D}(X^{\mathrm{proj}}_{u,t}).
$$

随后进行 pooling 得到结构证据表示：

$$
h_u(t)=\mathrm{Pool}(H^{\mathrm{str}}_{u,t}).
$$

### 6.4 结构证据门控

利用结构表示 $h_u(t)$，为每一条历史项定义结构证据门控：

$$
g_{u,x}^{(s)}(t)
=
\sigma\Big(
\mathrm{MLP}_s\big(h_u(t)\oplus z_x^{(0)}\oplus \phi(t-t_x)\big)
\Big).
$$

其含义是：

> **从邻域结构模式角度看，这条历史是否像“同一类高层组织持续性”的一部分。**

这一步是整个方法中最能体现“community evidence”问题意识的部分。因为它并不直接假设 repeated pair 才重要，而是在问：

- 这条历史是否落在一个更稳定的局部结构模式中？
- 这个模式是否会反复以相似方式出现？
- 它是否更可能指向社区持续性，而不是短期事件波动？

---

## 7. 模块四：Evidence-Aware TPPR

### 7.1 为什么不能事后再修正 TPPR

一种简单但不够理想的办法是：

- 先按原始方式计算 TPPR；
- 再用某个权重去修正最终 affinity。

这种做法的问题是：

- 不可靠历史已经在传播过程中被高阶扩散放大；
- 后验修正只能补救，而不能改变传播路径本身。

因此，更自然的方式是：

> **在 TPPR 的转移层面就注入 evidence filtering。**

### 7.2 原始 TPPR 视角

设 TPPR 的时序传播建立在事件级或边级转移概率上。对某一步转移 $a\to b$，原始转移概率记为：

$$
T_{a\to b}.
$$

它通常由时间顺序、邻接关系和时间衰减共同决定。

### 7.3 证据加权转移

对每个候选转移事件 $b$，定义其 evidence weight：

$$
\eta_b = g_b^{(r)}\cdot g_b^{(s)}.
$$

这里：

- $g_b^{(r)}$ 表示该事件从节律角度是否可信；
- $g_b^{(s)}$ 表示该事件从结构模式角度是否可信。

于是得到过滤后的转移：

$$
\widetilde{T}_{a\to b}
=
\frac{T_{a\to b}\,\eta_b}
{\sum_c T_{a\to c}\,\eta_c}.
$$

这样，TPPR 传播就不再把所有历史事件等价视为时序证据，而是优先沿着：

> **节律上更协调、结构上更像 community evidence 的历史路径**

进行高阶扩散。

### 7.4 过滤后的 TPPR 相似图

在 $\widetilde{T}$ 上运行 TPPR，得到：

$$
\widetilde{\Pi}\in \mathbb{R}^{|V|\times |V|}.
$$

定义其度矩阵：

$$
D_{\widetilde{\Pi}}=\mathrm{diag}(\widetilde{\Pi}\mathbf{1}).
$$

这样构造出的 affinity graph 与原始 TPPR 相比，关键差异在于：

- 原始 TPPR：传播所有符合时间顺序的历史关联；
- 本方法：只重点传播更可能体现 community evidence 的历史关联。

---

## 8. 社区损失：Evidence-Aware TPPR-Ncut

### 8.1 软分配矩阵

基于当前 embedding $Z$，通过 assignment head 得到软分配矩阵：

$$
S=\mathrm{softmax}(\mathrm{MLP}_{\mathrm{assign}}(Z)),
$$

其中：

$$
S\in\mathbb{R}^{|V|\times K},
$$

$K$ 为聚类数。

### 8.2 Ncut 目标

在过滤后的高阶 affinity graph $\widetilde{\Pi}$ 上定义 Ncut 型损失：

$$
L_{\mathrm{cut}}^{\mathrm{base}}
=
\mathrm{Tr}\Big(
(S^\top D_{\widetilde{\Pi}}S+\varepsilon I)^{-1}
S^\top(D_{\widetilde{\Pi}}-\widetilde{\Pi})S
\Big).
$$

该目标鼓励：

- 社区内部保留更强的 evidence-aware 高阶关联；
- 社区之间切断较弱的 evidence-aware 高阶关联。

### 8.3 Balance penalty

为避免塌缩解，加入平衡约束：

$$
L_{\mathrm{bal}}
=
\frac{1}{\sqrt{K}-1}
\left(
\sqrt{K}-
\frac{1}{\sqrt{2m_{\widetilde{\Pi}}}}
\sum_{j=1}^{K}
\|s_j\odot d_{\widetilde{\Pi}}^{1/2}\|_2
\right),
$$

其中：

- $s_j$ 为 $S$ 的第 $j$ 列；
- $d_{\widetilde{\Pi}}$ 为过滤后 affinity 的度向量；
- $m_{\widetilde{\Pi}}$ 为总边权的一半。

最终社区损失为：

$$
L_{\mathrm{cut}}=L_{\mathrm{cut}}^{\mathrm{base}}+\lambda_pL_{\mathrm{bal}}.
$$

### 8.4 这一项的解释

原方案中的 TPPR-Ncut 主要回答：

> **高阶时序相似图上的社区应如何切分。**

而现在的 Evidence-Aware TPPR-Ncut 更进一步回答：

> **在只保留更可信 community evidence 的高阶相似图上，社区应如何切分。**

这使得 cut objective 不再被动接受全部历史传播结果，而是建立在筛选后的 community evidence graph 上。

---

## 9. Batch reconstruction loss

为了保留 batch 内局部结构，仍保留辅助重构项。对一个 mini-batch $\mathcal{B}$，定义：

$$
\begin{aligned}
L_{\mathrm{batch}}
= \frac{1}{|\mathcal{B}|}
\sum_{(u,v,t)\in\mathcal{B}}\Bigg[
&\big(1-\mathrm{sim}(u,v,t)\big)
\\
&+\frac{1}{|N_{u,t}|}\sum_{h\in N_{u,t}}\big(1-\mathrm{sim}(u,h,t)\big)
\\
&+\frac{1}{|\bar N_{u,t}|}\sum_{\bar h\in \bar N_{u,t}}\big|\mathrm{sim}(u,\bar h,t)\big|
\Bigg].
\end{aligned}
$$

该项的作用保持不变：

1. 拉近当前真实交互节点对；
2. 拉近历史邻居；
3. 压低非邻居相似度。

在本方法中，这一项不是主要创新点，因此可基本保留原形式。

---

## 10. 总损失与完整训练目标

最终总目标写为：

$$
L
=
L_{\mathrm{temp}}
+\lambda_{\mathrm{cut}}L_{\mathrm{cut}}
+\lambda_{\mathrm{batch}}L_{\mathrm{batch}}.
$$

更展开地：

$$
\begin{aligned}
L
=
&\Bigg[
-\log\sigma\big(s(u,v,t)\big)
-\sum_{n\sim P_n}\log\sigma\big(-s(u,n,t)\big)
\Bigg]
\\
&+\lambda_{\mathrm{cut}}\Bigg[
\mathrm{Tr}\Big(
(S^\top D_{\widetilde{\Pi}}S+\varepsilon I)^{-1}
S^\top(D_{\widetilde{\Pi}}-\widetilde{\Pi})S
\Big)
+\lambda_pL_{\mathrm{bal}}
\Bigg]
\\
&+\lambda_{\mathrm{batch}}L_{\mathrm{batch}}.
\end{aligned}
$$

这里最核心的部分是：

- $L_{\mathrm{temp}}$ 中的历史项不再无差别累加；
- $L_{\mathrm{cut}}$ 中的 affinity 不再是原始 TPPR，而是 evidence-aware TPPR；
- 因此 temporal evidence selection 与 community partitioning 实现了更紧密耦合。

---

## 11. 训练流程（算法草稿）

### 11.1 预处理阶段

1. 从原始 event-sequence 构造训练事件流；
2. 通过 node2vec 或其他轻量方法初始化节点表示 $Z^{(0)}$；
3. 为每个节点整理历史交互时间序列；
4. 根据时间顺序维护历史窗口，供 Rhythm Encoder 和 FSTS 使用。

### 11.2 训练阶段

对每个 mini-batch，执行：

#### Step 1：节律表示更新
- 对 batch 中涉及的节点提取历史时间序列；
- 计算 $r_i^{\mathrm{hist}}(t)$；
- 融合 recency，得到 $r_i(t)$。

#### Step 2：历史结构证据表示更新
- 为 batch 中当前节点构造历史邻域窗口 $X_{u,t}$；
- 通过 FSTS 得到 $h_u(t)$；
- 计算结构证据门控 $g_{u,x}^{(s)}(t)$。

#### Step 3：节律门控与时间核计算
- 由 $r_u(t)$ 产生混合核系数 $\pi_u(t)$；
- 计算节律条件化核 $\kappa_u(\Delta t\mid r_u(t))$；
- 计算节律一致性门控 $g_{u,x}^{(r)}(t)$。

#### Step 4：事件得分与 temporal loss
- 用改进后的历史项计算 $s(u,v,t)$；
- 对正样本和负样本计算 $L_{\mathrm{temp}}$。

#### Step 5：Evidence-aware TPPR 构图
- 根据 $g^{(r)}$ 与 $g^{(s)}$ 计算事件 evidence weight；
- 用加权转移概率构造 $\widetilde{T}$；
- 基于 $\widetilde{T}$ 更新或近似计算 $\widetilde{\Pi}$。

#### Step 6：社区损失与重构损失
- 基于当前 $Z$ 计算 assignment matrix $S$；
- 用 $\widetilde{\Pi}$ 计算 $L_{\mathrm{cut}}$；
- 在当前 batch 上计算 $L_{\mathrm{batch}}$。

#### Step 7：参数更新
- 对 $Z$、Rhythm Encoder、FSTS、assignment head、以及相关门控网络参数做反向传播。

### 11.3 伪代码（简版）

```text
Input: Event sequence E, initial embedding Z(0), cluster number K
Output: Optimized embedding Z, soft assignment S

Initialize Z <- Z(0)
for epoch = 1 to T do
    for mini-batch B in chronological order do
        Extract involved nodes and their historical timestamps
        Compute rhythm representations r_i(t)
        Build historical neighborhood windows X_{u,t}
        Compute structure evidence vectors h_u(t)
        Compute g^(r), g^(s), and adaptive kernel kappa_u
        Compute event scores s(u,v,t)
        Compute L_temp

        Construct evidence-aware transition \tilde{T}
        Compute/update evidence-aware TPPR affinity \tilde{Pi}
        Compute assignment matrix S
        Compute L_cut and L_batch

        L = L_temp + lambda_cut * L_cut + lambda_batch * L_batch
        Update parameters by gradient descent
    end for
end for
return Z, S
```

---

## 12. 方法相较于原方案的核心提升

### 12.1 相较于原始 Hawkes

原始 Hawkes 只回答：

- 过去事件多久之前发生；
- 其影响按什么固定核衰减。

本方法额外回答：

- 这条历史在当前节律状态下是否可信；
- 它是否在邻域模式上更像 community evidence；
- 哪些时间尺度上的历史更值得被继承。

因此，提升不在于“更复杂”，而在于：

> **历史影响机制从统一衰减，变成了 community-evidence-aware filtering。**

### 12.2 相较于直接套用 SeqFilter

SeqFilter 直接服务于 link prediction，而本方法将其思想改造成：

- **Rhythm module** 不再输出最终 link representation，而用于调节 Hawkes；
- **FSTS module** 不再直接做节点结构编码，而用于判断历史是否像 community evidence；
- **filtered evidence** 不直接用于打分未来边，而进一步进入 TPPR-Ncut。

因此，我们不是做“SeqFilter for clustering”，而是：

> **借其时序证据过滤思想，重构适合 clustering 的历史信息流。**

### 12.3 相较于只改 TPPR/Ncut

如果只改 TPPR 或 Ncut，而不改 temporal loss 的历史项，那么：

- 不可靠历史仍然会先进入事件级建模；
- 后续结构模块只是被动接受这些混杂影响。

本方法则把 evidence filtering 前移到 Hawkes 历史项中，再向 TPPR 传播，因此信息流更自然、也更一致。

---

## 13. 方法的创新点表述（可直接写进论文）

### 13.1 创新点 1：问题视角创新

现有时序图聚类方法更常把重点放在模型结构、损失设计或训练框架适配上，而本工作进一步提出：

> **在 event-sequence 定义下，当 repeated pairs 较弱时，关键并不只是如何建模历史，而是如何识别哪些历史真正构成社区证据。**

### 13.2 创新点 2：Rhythm-Filtered Hawkes

我们提出在 Hawkes 风格 temporal loss 中引入节点节律表示，将固定时间衰减扩展为节律条件化的多尺度时间核，并引入节律一致性门控，从而使历史事件的影响不再由统一衰减规则决定，而由其与当前节点状态的时间一致性共同决定。

### 13.3 创新点 3：Frequency-Selective Structure Evidence Filter

我们借鉴 frequency-selective filtering 的思想，将历史邻域序列映射为结构证据表示，并进一步定义结构证据门控，以识别更可能反映社区持续性的历史结构模式。

### 13.4 创新点 4：Evidence-Aware TPPR-Ncut

我们不再在原始时序传播基础上构造 TPPR，而是将节律门控与结构证据门控前移到传播转移层，构造 evidence-aware TPPR affinity，使 cut objective 直接建立在更可信的 community evidence graph 上。

---

## 14. 可做的消融实验设计

为了验证每一部分的必要性，建议做如下消融：

### 14.1 Rhythm 相关
1. **w/o Rhythm**：去掉节律表示，只保留原始 Hawkes；
2. **w/o Recency**：只保留长期节律，不加入最近活跃时间；
3. **Single-scale kernel**：将多尺度核退化为单一指数核；
4. **w/o Gate-r**：去掉节律一致性门控，只保留节律条件化时间核。

### 14.2 Structure evidence 相关
5. **w/o FSTS**：去掉结构证据模块，令 $g^{(s)}=1$；
6. **w/o frequency selection**：取消频率筛选，只做时域 Conv；
7. **w/o global reweighting**：保留频率筛选，但去掉频带重加权；
8. **w/o local Conv**：保留频率变换，但不学局部相关。

### 14.3 TPPR/Ncut 相关
9. **post-hoc TPPR weighting**：先算普通 TPPR，再乘 evidence score；
10. **pre-filtered TPPR**：使用本文 प्रस्तावित 的 evidence-aware transition；
11. **original Ncut**：在原始 TPPR 上做 cut；
12. **evidence-aware Ncut**：在 $\widetilde{\Pi}$ 上做 cut。

### 14.4 问题视角验证相关
13. 比较不同数据集上：
   - repeated pair ratio；
   - rhythm consistency；
   - filtered evidence concentration；
   - final clustering performance。

这有助于证明：

> **方法效果并不是因为简单加模块，而是因为更准确地筛选了 community evidence。**

---

## 15. 复杂度讨论（草稿）

设：

- 节点数为 $N$；
- 事件数为 $M$；
- embedding 维度为 $d$；
- 历史窗口长度为 $K$；
- 时间基函数数为 $M_k$；
- 聚类数为 $C$。

### 15.1 Rhythm Encoder
若对每个节点只维护长度不超过 $K_r$ 的历史时间窗口，则节律编码复杂度约为：

$$
O(BK_r d_r)
$$

其中 $B$ 为 batch 中涉及节点数。

### 15.2 FSTS
若对每个当前节点历史窗口做一次 DFT，复杂度约为：

$$
O(BK\log K + BKd_x).
$$

### 15.3 Hawkes 历史项
若每个事件只考虑 $K_h$ 个候选历史项，则 temporal loss 相关复杂度约为：

$$
O(|\mathcal{B}|K_h d).
$$

### 15.4 Evidence-aware TPPR
这是最重的一部分。实践中可采用：

- 截断步数；
- 稀疏传播；
- mini-batch 近似；
- 仅更新 batch 涉及局部子图。

否则全局精确更新 $\widetilde{\Pi}$ 的代价会较高。

因此在实现上更建议：

> **局部近似 evidence-aware TPPR，而不是每轮全图精确重算。**

---

## 16. 当前方案的优势与可能风险

### 16.1 优势

1. **与当前 offline 主线衔接自然**：不需要完全推翻现有框架；
2. **问题意识更清楚**：围绕 community evidence，而不是泛泛建模历史；
3. **方法逻辑前后一致**：先筛历史，再传播，再切分；
4. **与 SeqFilter 的结合有选择性**：借其精华，不照搬任务设定。

### 16.2 风险

1. **模块稍多**：Rhythm + FSTS + TPPR 耦合后，训练会更复杂；
2. **证据门控可解释性需额外验证**：需要统计和可视化支持；
3. **TPPR 更新开销可能较大**：需要近似策略；
4. **若数据本身时间节律不明显，Rhythm branch 的收益可能有限。**

因此，实际研究推进中建议先做分阶段实现，而不是一步到位。

---

## 17. 推荐的实现顺序

为了降低复杂度并增强论文叙事的可控性，建议按以下三步推进：

### 第一阶段：只改 Hawkes 的时间核
实现：

$$
\kappa_u(\Delta t\mid r_u(t))
$$

目标：验证“统一指数核不如节律条件化时间核”。

### 第二阶段：加入节律门控
实现：

$$
g_{u,x}^{(r)}(t)
$$

目标：验证“不是所有时间上接近的历史都一样重要”。

### 第三阶段：加入结构证据过滤与 evidence-aware TPPR
实现：

$$
g_{u,x}^{(s)}(t), \qquad \widetilde{T}, \qquad \widetilde{\Pi}.
$$

目标：验证“在经过 community-evidence filtering 后，高阶传播更适合聚类”。

这个顺序的优点是：

- 每一步都对应一个清晰假设；
- 每一步都容易做消融；
- 更适合论文中逐步展开问题与方法。

---

## 18. 一段可直接放进论文的方法总述

我们提出一种 **Rhythm-Filtered Hawkes with Evidence-Aware TPPR-Ncut** 框架，用于 event-sequence 场景下的时序图聚类。在 repeated pairs 较弱时，我们认为问题的关键不在于无差别地继承历史，而在于识别哪些历史时序信息真正构成 community evidence。为此，我们首先引入节点节律编码模块，从绝对时间序列中提取节点的长期行为节律，并结合最近活跃状态构造节律表示。随后，在 Hawkes 风格的 temporal loss 中，我们用节律条件化的多尺度时间核替换固定指数核，并进一步通过节律一致性门控与结构证据门控筛选历史项，从而使进入事件强度建模的历史更加聚焦于具有社区意义的时序证据。在此基础上，我们将筛选后的 evidence score 前移到 TPPR 的转移层，构造 evidence-aware 的高阶时序相似图，并在该图上施加 TPPR-Ncut 社区目标。最终，该方法实现了从历史证据筛选、到高阶传播、再到社区切分的统一建模，使时序图聚类更加贴合“什么样的时序信息可以作为社区证据”这一核心问题。

---

## 19. 参考文献（写作时建议保留）

1. Yuanyuan Xu, Danni Wu, Xuemin Lin, Dong Wen, Wenjie Zhang, Lei Chen, and Ying Zhang. **Exploring Sequential Dynamics on Temporal Graphs via Composite Filtering**. Proceedings of The ACM Web Conference 2026, pp. 487–498.
2. Jain Malaviya et al. **Survey on Modeling Intensity Function of Hawkes Process**. arXiv:2104.11092, 2021.
3. Hongyuan Mei and Jason Eisner. **The Neural Hawkes Process: A Neurally Self-Modulating Multivariate Point Process**. NeurIPS 2017.
4. Yagmur Isik, Marta Kwiatkowska, and Changhee Son. **Hawkes Process with Flexible Triggering Kernels**. PMLR 219, 2023.
5. Ke Zhou, Hongyuan Zha, and Le Song. **Learning Triggering Kernels for Multi-dimensional Hawkes Processes**. ICML 2013.  
   （正式写作时可补充为你最终采用的非参数 Hawkes 参考文献。）

---

## 20. 最后一句话总结

如果把整个方案压缩成一句话，那么它的核心就是：

> **不是让所有历史继续影响当前，而是先判断哪些历史在时间节律和局部结构上更像 community evidence，再让这些历史进入 Hawkes 与 TPPR，并最终服务于 TPPR-Ncut 聚类。**


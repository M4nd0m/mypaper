# 当前方案整理（基于师兄 offline 主线的简化版）

## 1. 方案定位

当前方案不再优先走“另起一套多阶段时序聚类框架”的路线，而是**紧贴师兄论文的 offline 主线**，在其基础上做针对性的改进。

核心判断如下：

1. 师兄的 offline 阶段，本质上可以理解为一个 **clustering-oriented 的 embedding optimization** 过程；
2. 它和 DTGC 在训练逻辑上是相通的：先初始化 embedding，再把 embedding 当作可优化变量，通过损失函数不断更新；
3. 师兄相对 DTGC 的关键变化，不是整体训练范式变了，而是**把原本的聚类损失替换成了基于 TPPR 的 Ncut**；
4. 因此，我当前的工作重点不应是大幅改写整体框架，而应是：
   - 保留师兄 offline 的主骨架；
   - 围绕 **TPPR**、**Ncut**、以及“突变/过平滑”问题做最小而关键的增强。

---

## 2. 当前不优先采用的方案

经过讨论，当前版本**暂不优先采用**以下设计：

- 多阶段可训练表示矩阵 $Z_b$；
- 多阶段软分配矩阵 $Q_b$；
- 显式跨阶段 assignment consistency 项；
- 重新搭建一整套 stage-wise temporal clustering 框架。

原因不是这些设计没有道理，而是：

1. 它们会显著增加模型复杂度；
2. 会引入更多超参数与训练不稳定因素；
3. 会削弱“这是在师兄基础上继续做”的自然性；
4. 目前其收益是否明显高于复杂度成本，**尚无充分证据**。

因此，当前阶段更稳妥的做法是：**先把主线收缩到师兄 offline 框架本身，再聚焦一个最关键的创新点。**

---

## 3. 当前方案的核心思想

### 3.1 基本思路

保留如下主线：

1. 用 node2vec 或其他轻量方式初始化节点表示；
2. 基于时间事件流和历史邻居关系，构造 temporal loss；
3. 基于 TPPR 构造高阶时序相似图；
4. 在该相似图上施加 Ncut 型社区目标；
5. 用 batch-level reconstruction 保持局部结构；
6. 直接优化 embedding 本身。

因此，当前方案可以概括为：

> 在“embedding 作为优化变量”的训练范式下，将 TPPR 提供的高阶时序结构信息与 Ncut 社区目标结合，并通过 temporal loss 与 batch reconstruction 共同约束 embedding，使 learned embedding 更适合 temporal graph clustering。

---

## 4. 模型变量与输入

给定连续时间时序图：

$$
\mathcal{G} = (V, E)
$$

其中事件集合为：

$$
E = \{(u_r, v_r, t_r)\}_{r=1}^{M}
$$

节点初始表示记为：

$$
Z^{(0)} \in \mathbb{R}^{|V| \times d}.
$$

这里 $Z^{(0)}$ 可由 node2vec 或其他轻量预训练方式得到。

训练中，真正被优化的是节点表示矩阵：

$$
Z \in \mathbb{R}^{|V| \times d}.
$$

也就是说，当前方案不强依赖一个复杂 encoder，而是直接把 embedding 视为优化对象。

---

## 5. TPPR 高阶时序相似图

对原始事件流计算 TPPR，得到节点间的高阶时序相似性矩阵：

$$
\Pi \in \mathbb{R}^{|V| \times |V|}.
$$

其中，$\Pi_{ij}$ 表示节点 $i$ 到节点 $j$ 的高阶时序关联强度。

在此基础上定义度矩阵：

$$
D_{\Pi} = \operatorname{diag}(\Pi \mathbf{1})
$$

该矩阵不是普通的一阶邻接，而是包含了时间顺序约束和高阶传播信息的 affinity graph，因此更适合作为 cut objective 的结构基础。

---

## 6. 当前方案的损失函数

当前总目标函数写为：

$$
L = L_{\mathrm{temp}} + \lambda_{\mathrm{cut}} L_{\mathrm{cut}} + \lambda_{\mathrm{batch}} L_{\mathrm{batch}}.
$$

这里：

- $L_{\mathrm{temp}}$：保持事件级时间动态；
- $L_{\mathrm{cut}}$：在 TPPR 相似图上施加社区划分约束；
- $L_{\mathrm{batch}}$：保留 batch 内局部连接结构。

这个写法和师兄 offline 的主干高度一致，但当前研究重点会放在 **如何改进 $L_{\mathrm{cut}}$**，使其更适合你的问题设定。

---

## 7. Temporal loss

对于一个事件 $(u, v, t)$，定义条件交互强度：

$$
\begin{aligned}
s(u,v,t)
= {} & \mu(u,v,t) \\
& + \sum_{x \in N_{u,t}} w(x,u,t)\,\mu(x,v,t)\,
\exp\bigl(-\delta_t (t - t_x)\bigr)
\end{aligned}
$$

其中基础相似度可写为：

$$
\mu(u,v,t) = -\lVert z_u^t - z_v^t \rVert_2^2
$$

于是 temporal loss 写为：

$$
L_{\mathrm{temp}}
= -\log \sigma\bigl(s(u,v,t)\bigr)
  - \sum_{n \sim P(n)} \log \sigma\bigl(-s(u,n,t)\bigr)
$$

该项的作用是：

- 提高真实交互节点对的打分；
- 压低随机负样本的打分；
- 让 embedding 保留事件序列中的时间依赖信息。

这部分当前可以基本保留，不作为主要创新点。

---

## 8. TPPR-Ncut 社区损失

### 8.1 当前写法

在 embedding $Z$ 的基础上，通过一个轻量 assignment head 得到软分配矩阵：

$$
S = \operatorname{softmax}(\operatorname{MLP}(Z))
$$

然后在 TPPR 相似图 $\Pi$ 上定义 Ncut 型目标：

$$
L_{\mathrm{cut}}^{\mathrm{base}}
= \operatorname{Tr}\!\Bigl(
\bigl(S^\top D_{\Pi} S + \varepsilon I\bigr)^{-1}
S^\top \bigl(D_{\Pi} - \Pi\bigr) S
\Bigr)
$$

为防止塌缩解，再加入 balance penalty：

$$
L_{\mathrm{bal}}
= \frac{1}{\sqrt{K} - 1}
\left(
\sqrt{K}
- \frac{1}{\sqrt{2m_{\Pi}}}
\sum_{j=1}^{K} \lVert s_j \odot d_{\Pi}^{1/2} \rVert_2
\right)
$$

其中 $s_j$ 为 $S$ 的第 $j$ 列，$d_{\Pi}$ 为 TPPR 图上的度向量。

因此，当前社区损失写作：

$$
L_{\mathrm{cut}} = L_{\mathrm{cut}}^{\mathrm{base}} + \lambda_p L_{\mathrm{bal}}
$$

### 8.2 这一项的定位

这一项是当前方案里**最值得进一步改进的核心模块**。

因为从现在的研究目标看，你真正想解决的问题并不是“有没有必要再造一个完整大框架”，而是：

> 师兄当前的 TPPR-Ncut，在面对时序突变、社区边界变化、或者过平滑风险时，是否仍然足够好？

因此，下一步更合理的工作重点应当是：

- 改进 TPPR；或
- 改进 Ncut；或
- 改进二者之间的耦合方式。

而不是优先引入大量新的 stage-wise 变量。

---

## 9. Batch reconstruction loss

为了保持 batch 内部的局部邻接结构，定义：

$$
\begin{aligned}
L_{\mathrm{batch}}
= \frac{1}{|\mathcal{B}|}
\sum_{(u,v,t)\in \mathcal{B}}
\Biggl[
& \bigl(1 - \operatorname{sim}(u,v,t)\bigr) \\
& + \frac{1}{|N_{u,t}|} \sum_{h \in N_{u,t}} \bigl(1 - \operatorname{sim}(u,h,t)\bigr) \\
& + \frac{1}{|\bar N_{u,t}|} \sum_{\bar h \in \bar N_{u,t}} \bigl| \operatorname{sim}(u,\bar h,t) \bigr|
\Biggr]
\end{aligned}
$$

该项作用是：

1. 拉近当前交互节点对；
2. 拉近历史邻居；
3. 压低非邻居相似度。

这一项仍可保留为辅助项，不需要在当前阶段大改。

---

## 10. 当前方案与 DTGC、师兄 offline 的关系

### 10.1 与 DTGC 的关系

当前方案与 DTGC 的相似之处在于：

1. 都采用“初始化 embedding，然后不断优化 embedding”的训练逻辑；
2. 都服务于 temporal graph clustering；
3. 都不是单纯地先学表示、后面再完全独立聚类，而是让 clustering objective 参与表示优化。

不同之处在于：

1. DTGC 原始 clustering loss 不是 TPPR-Ncut；
2. 当前方案的社区约束不是一般的节点分布校准，而是建立在 **TPPR 高阶时序相似图** 上的 cut objective；
3. 当前方案更强调“高阶时序社区结构”，而非仅仅适配 batch-processing 的 clustering module。

### 10.2 与师兄 offline 的关系

当前方案与师兄 offline 更接近。

可以理解为：

> 保留师兄 offline 的基本训练范式，只把后续研究重点集中到“如何让 TPPR-Ncut 更适合你的问题设定”。

因此，当前方案不是完全另起炉灶，而是**以师兄 offline 为主干、以 TPPR-Ncut 为核心、以最小代价往前推进**。

---

## 11. 当前阶段的研究重点

基于目前讨论，当前阶段最合理的研究重点不应是“再造一个完整新框架”，而应是以下问题之一：

### 方向一：改 TPPR

核心问题：

- TPPR 在突变阶段是否会过度继承旧历史；
- TPPR 的时间衰减是否过于固定；
- 高阶扩散是否会把突变边界抹平。

如果答案是“会”，那么就可以从 TPPR 的转移概率、时间核、扩散范围控制等角度做改进。

### 方向二：改 Ncut

核心问题：

- 当前 cut objective 是否默认偏好时间平滑；
- 是否会对真实突变阶段产生过大惩罚；
- 是否缺少对边界变化、簇分裂/合并的适应能力。

如果答案是“会”，那么就可以围绕 TPPR-Ncut 的目标函数做修正。

### 方向三：改二者耦合方式

核心问题：

- TPPR 提供的 affinity 与 Ncut 之间，现在是否只是“先算 affinity，再做 cut”；
- 是否可以让 cut 对 TPPR 的构图偏好形成反向引导；
- 是否可以设计更贴近社区边界的高阶扩散。

这一方向更强，但也更难，是中后期可以考虑的升级方向。

---

## 12. 当前结论

当前最稳妥、最符合研究目标的方案可以总结为：

> 以师兄 offline 为基础，保留“初始化 embedding → 直接优化 embedding → temporal loss + TPPR-Ncut + batch reconstruction”的主骨架，不优先引入多阶段显式时序聚类变量，而是将创新重点集中到 TPPR 或 Ncut 的针对性改进上，特别关注突变场景下的过平滑与社区边界保持问题。

这一路线的优点是：

1. 与师兄工作衔接自然；
2. 方法叙事更简洁；
3. 更容易证明“这是合理推进，而不是另起炉灶”；
4. 更适合后续逐步加入针对突变的最小增强模块。

---

## 13. 后续建议

在当前方案基础上，下一步最值得优先明确的问题是：

### 到底改 TPPR，还是改 Ncut？

更具体地说，可以先回答：

1. 你认为“突变”首先破坏的是高阶扩散，还是 cut 的平滑偏好？
2. 你希望你的贡献更像“更好的时序相似图”，还是“更好的社区目标函数”？
3. 你是想做“最小改动但扎实”，还是“结构更强但更复杂”的版本？

只有这一点明确了，当前方案才会真正收束。

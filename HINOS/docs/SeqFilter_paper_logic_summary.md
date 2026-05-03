# Exploring Sequential Dynamics on Temporal Graphs via Composite Filtering 论文逻辑整理

> 论文：Yuanyuan Xu, Danni Wu, Xuemin Lin, Dong Wen, Wenjie Zhang, Lei Chen, Ying Zhang. **Exploring Sequential Dynamics on Temporal Graphs via Composite Filtering**. WWW 2026.
>
> 方法名：**SeqFilter**
>
> 任务：Temporal graph link prediction / temporal link prediction
>
> 核心关键词：Temporal graph, sequential dynamics, composite filter, node rhythm memory, frequency-selective structure encoder

---

## 1. 论文整体主线

这篇论文讨论的是**具有复杂序列动态的时序图**上的未来链路预测问题。

论文的基本判断是：很多真实时序图不是由大量重复交互驱动的，而是由**事件顺序和行为意图演化**驱动的。也就是说，未来边不一定是历史边的重复，而可能是沿着某种行为序列出现的新交互。

例如，用户刚买了跑步机，下一步更可能买跑鞋、护膝等相关商品，而不是再买一台跑步机。因此，预测未来交互的关键不只是记住“这个用户过去和谁交互过”，而是理解“这个用户的行为在时间上如何推进”。

论文认为现有 Temporal Graph Neural Networks，简称 T-GNNs，在这类图上表现不稳定，主要原因有两个：

1. 现有 memory module 和 neighbor co-occurrence encoding 往往依赖历史邻居身份或共同邻居假设；
2. 现有模型容易把突发交互、离群交互当成普通事件，从而影响泛化。

因此，论文提出 **SeqFilter**，把时序图表示学习重新理解为一个**信号过滤问题**。模型不再重点记忆“节点和谁交互”，而是重点建模“节点什么时候交互”，并通过频域过滤提取结构演化中的有效模式。

---

## 2. 问题背景：什么是 sequential dynamics？

论文中的时序图定义为：

$$
G = (V, E)
$$

其中时序边集合按照时间排序：

$$
G = \{\delta(t_1), \delta(t_2), \cdots\}
$$

每一个事件为：

$$
\delta(t) = (\mathbf{v}_i, \mathbf{v}_j, \mathbf{e}_{ij}(t), t)
$$

其中：

- $\mathbf{v}_i, \mathbf{v}_j$：节点特征；
- $\mathbf{e}_{ij}(t)$：边特征；
- $t$：事件发生时间。

论文关注的是 TGB-Seq benchmark 中的 sequential dynamic graphs。这类图的典型特点是：

- 边重复率很低；
- 未来交互往往是未见过的新交互；
- 预测目标更偏向理解历史事件序列如何推动下一步行为；
- 共同邻居、重复边等传统信号不再稳定可靠。

论文把问题表述为：给定节点历史交互，模型需要预测未来是否会发生未见过的交互。

---

## 3. 现有 T-GNN 的三个问题

论文在方法之前先做了一个 “Rethinking Mainstream T-GNNs” 的分析。它不是直接提出模型，而是先解释为什么主流 T-GNN 在 sequential dynamics 上会失效。

### 3.1 问题一：memory module 记住了 who，但忽略了 when

很多 T-GNN 使用 memory module 保存节点历史状态。典型形式是：

$$
\mathbf{m}_i(t) \leftarrow MEM(\mathbf{m}_i(t'), \mathbf{v}_i, \mathbf{m}_j(t'), \mathbf{e}_{ij}(t), \Delta t)
$$

这个公式表示：节点 $v_i$ 的 memory 由自己的旧 memory、对方节点的旧 memory、边特征、相对时间间隔等信息更新。

论文认为这种做法本质上是在记忆：

> 节点过去和谁交互过，以及这些交互的具体邻居信息。

这种机制适合重复交互多的图。例如用户反复联系同一个朋友，或者同一个用户反复购买同类商品。

但是在 sequential dynamic graphs 中，边重复很少，过去的具体邻居身份不一定能直接帮助预测未来边。因此，论文认为 memory 应该从：

$$
\text{memorize whom nodes interact with}
$$

转向：

$$
\text{memorize when nodes interact}
$$

也就是说，应该建模节点的行为节奏，而不是只记历史邻居身份。

---

### 3.2 问题二：neighbor co-occurrence assumption 不再可靠

很多时序图模型会使用共同邻居或邻居共现编码。它们的基本假设是：

> 如果两个节点历史上共享更多邻居，那么它们未来更可能发生交互。

形式上，结构编码可以写成：

$$
\mathbf{h}_i(t) \leftarrow AGG(\mathbf{v}_i, \{\mathbf{v}_j, \mathbf{e}_{ij}(t), \Delta t, \mathbf{c}_{ij}(t)\} \mid v_j \in \mathcal{N}^k_i(t))
$$

其中 $\mathbf{c}_{ij}(t)$ 是邻居共现编码：

$$
\mathbf{c}_{ij}(t) \leftarrow CO\text{-}NEG(\mathcal{N}^k_i(t), \mathcal{N}^k_j(t))
$$

论文认为，这个假设在重复边多的图上有效，但在 sequential dynamic graphs 中会变弱。原因是：

- 未来交互主要受事件顺序影响；
- 节点对之间的共同邻居可能很稀疏；
- 扩大邻居采样范围虽然能增加共现，但会引入更多噪声和计算开销。

论文还提到，在 Patent 数据集上，用 DyGFormer 统计时，很多 batch 中节点对的平均邻居共现比例低于 1%。这说明共同邻居信号在这类数据上非常稀疏。

---

### 3.3 问题三：模型对 abrupt / outlier interactions 敏感

真实时序图中会有很多突发事件或离群事件，尤其是在更新频率很高的数据中。例如 Yelp 有超过一千万个时间戳。

论文认为，现有 T-GNN 往往把这些突发或离群交互当作普通事件处理，因此容易：

- 过拟合不重要的异常交互；
- 学不到稳定的结构演化模式；
- 降低泛化能力。

所以，模型需要一种机制来抑制无意义的高频噪声，同时保留真正有用的结构变化。

---

## 4. SeqFilter 的总体思想

SeqFilter 的核心思想是：

> 把时序图表示学习看作一个过滤过程，用 filter 从复杂序列动态中提取有效的时间模式和结构模式。

它包含两个主要模块：

1. **Node Rhythm Memory**
2. **Frequency-Selective Temporal Structure Encoder，简称 FSTS Encoder**

最后，模型把这两个模块的输出融合，得到节点表示。

整体逻辑是：

```text
Temporal events
   ↓
Node Rhythm Memory
   → 建模节点行为节奏，强调 when
   ↓
Frequency-Selective Temporal Structure Encoder
   → 在频域中过滤结构信号，抑制噪声和突发事件
   ↓
Fusion
   → 生成节点表示
   ↓
Link prediction
```

---

## 5. 模块一：Node Rhythm Memory

### 5.1 为什么需要 Node Rhythm Memory？

传统 memory module 重点保存节点历史邻居信息。SeqFilter 认为这在低重复交互场景中不合适。

Node Rhythm Memory 的目标是：

> 不依赖具体历史邻居身份，而是只用绝对时间戳建模节点行为节奏。

它关心的是节点在什么时间活跃、行为是否有周期、最近一次交互离当前有多久。

---

### 5.2 历史 rhythm memory 更新

给定事件：

$$
\delta(t) = (\mathbf{v}_i, \mathbf{v}_j, \mathbf{e}_{ij}(t), t)
$$

对于节点 $v_i$，模型把当前绝对时间编码和旧 rhythm memory 拼接：

$$
[\phi(t) \oplus \mathbf{r}^{\iota}_i(t')]
$$

然后用 depthwise convolution 更新 rhythm memory：

$$
\mathbf{r}^{\iota}_i(t)
= Linear(DepthwiseConv1D([\phi(t) \oplus \mathbf{r}^{\iota}_i(t')]))
$$

其中：

- $\phi(t)$：时间编码函数；
- $\mathbf{r}^{\iota}_i(t')$：节点旧的历史 rhythm memory；
- $\oplus$：拼接；
- DepthwiseConv1D：逐通道卷积。

这个设计的作用是：

- 使用绝对时间戳，把所有事件放到统一时间轴上；
- 不需要知道节点具体和谁交互；
- 可以捕捉慢节奏和快节奏的行为模式；
- 不同通道可以独立处理不同时间维度。

---

### 5.3 Recency-aware node state

只有长期节奏还不够，因为历史模式会随着时间变旧。如果一个节点很久没有交互，旧的 rhythm 可能不再可靠。

因此论文加入 recency awareness，即显式建模当前时间和上一次交互时间之间的间隔：

$$
\mathbf{r}_i(t)
= LayerNorm(DepthwiseConv1D([\mathbf{r}^{\iota}_i(t') \oplus \phi(t - t')]))
$$

其中：

- $t$：当前时间；
- $t'$：节点上一次交互时间；
- $\phi(t - t')$：相对时间间隔编码。

这一项的作用是避免过旧的历史 rhythm 过度影响当前预测。

---

## 6. 模块二：Frequency-Selective Temporal Structure Encoder

### 6.1 为什么需要 FSTS Encoder？

在 sequential dynamic graphs 中，结构演化复杂，重复边少，突发事件多。普通结构编码器或者邻居共现编码很难区分：

- 哪些变化是真正有意义的结构模式；
- 哪些变化只是突发噪声或离群交互。

SeqFilter 因此把邻居结构信息变成频域信号，再用一组可学习 filter 处理。

FSTS Encoder 的核心范式是：

$$
\text{local} \rightarrow \text{global} \rightarrow \text{local}
$$

具体包括三步：

1. 局部低通去噪：过滤高频噪声；
2. 全局频带重加权：自适应增强有用频率；
3. 局部结构相关学习：用 Conv1D 学习邻居局部相关性。

---

### 6.2 构造结构输入

对于节点 $v_i$，先采样其在时间 $t$ 之前的 $k$ 个历史邻居：

$$
\mathcal{N}^k_i(t)
$$

对每个邻居 $v_j$，构造输入：

$$
\mathbf{x}_{ij}(t)
= [\mathbf{e}_{ij}(t) \oplus \phi(\Delta t) \oplus \mathbf{v}_j]
$$

即把边特征、相对时间编码、邻居节点特征拼接起来。

把一个 mini-batch 中的这些输入堆叠成矩阵：

$$
\mathbf{X}(t)
$$

然后做 DFT 转到频域：

$$
X(f) = DFT(\mathbf{X}(t))
$$

---

### 6.3 第一步：低通去噪

论文认为，突发交互或离群交互往往对应高频成分。因此先用 energy-based masking 过滤高频信息：

$$
X_{low}(f)
= I(|X(f)|^2 > \tau) \odot X(f)
$$

其中：

- $|X(f)|^2$：频率响应的能量；
- $\tau$：能量阈值；
- $I(\cdot)$：指示函数；
- $\odot$：逐元素乘法。

论文的表述是：$\tau$ 用来区分底层有效信号频率和 abrupt / outlier 频率。较大的 $\tau$ 会更强地过滤异常信号。

这一层对应低通 filter，记作：

$$
H_L(f)
$$

---

### 6.4 第二步：全局频带自适应调整

不同数据集和不同节点可能有不同周期，例如分钟级、天级、周级、月级。因此不能固定只保留某些频率。

论文用复值线性投影调整频带响应：

$$
X_{proj}(t)
= IDFT(X_{low}(f) \cdot W_C)
$$

其中：

$$
W_C = W_r + i \cdot W_i
$$

这里 $W_C$ 是复值权重，能够对不同频带进行可学习的增强或抑制。

这一层对应全局频率调整 filter，记作：

$$
H_P(f)
$$

---

### 6.5 第三步：局部结构相关学习

完成频域去噪和频带调整后，模型回到时域，用标准 Conv1D 学习邻居之间的局部结构相关性：

$$
\mathbf{X}_{out}(t)
= Conv1D(X_{proj}(t))
$$

最后和 Node Rhythm Memory 的输出融合：

$$
\mathbf{Z}(t)
= tanh(\mathbf{R}(t) + MeanPooling(\mathbf{X}_{out}(t)))
$$

其中：

- $\mathbf{R}(t)$：当前 batch 节点的 rhythm embedding；
- $\mathbf{X}_{out}(t)$：结构编码输出；
- $\mathbf{Z}(t)$：最终节点表示。

---

## 7. 理论部分：为什么叫 composite filter？

论文的理论部分主要有两个定理。

---

### 7.1 Theorem 1：FSTS Encoder 是级联过滤器

论文证明，FSTS Encoder 在频域中可以写成三个 filter 的乘积：

$$
X_{out}(f)
= H_{FSTS}(f) \cdot X(f)
$$

其中：

$$
H_{FSTS}(f)
= H_C(f) \cdot H_P(f) \cdot H_L(f)
$$

三个部分分别是：

- $H_L(f)$：低通去噪 filter，对应 Eq. (6) 的高频抑制；
- $H_P(f)$：复值线性投影，对应 Eq. (7) 的全局频带重加权；
- $H_C(f)$：Conv1D 的 FIR filter，对应 Eq. (8) 的局部结构相关学习。

所以 FSTS Encoder 不是单个简单卷积，而是一个级联的 composite filter。

直观理解：

```text
输入结构信号 X(f)
   ↓ H_L(f)：去掉高频噪声
   ↓ H_P(f)：调整不同频率的重要性
   ↓ H_C(f)：学习局部结构相关
输出 X_out(f)
```

---

### 7.2 Theorem 2：FSTS 可以近似 Wiener optimal denoiser

论文进一步证明，FSTS Encoder 可以近似最优线性去噪器。

假设观测信号为：

$$
X(t) = (h * S)(t) + N(t)
$$

其中：

- $S(t)$：真实底层结构信号；
- $N(t)$：噪声；
- $h$：系统响应；
- $*$：卷积。

经典 Wiener filter 的最优频率响应是：

$$
W_{opt}(f)
= \frac{H^*(f) S_s(f)}{P(f)}
$$

其中：

$$
P(f) = |H(f)|^2 S_s(f) + S_n(f)
$$

论文证明，对于任意 $\epsilon > 0$，FSTS 可以选择参数使得：

$$
\sup_{f \in [0,1)} |H_{FSTS}(f) - W_{opt}(f)| < \epsilon
$$

并且：

$$
MSE_{FSTS} - MSE_{min}
< \epsilon^2 \int_0^1 P(f) df
$$

这表示 FSTS 的去噪效果可以任意接近 Wiener optimal solution。

论文用这个定理说明：FSTS 不是经验性地加一个频域模块，而是可以从信号处理角度解释为一个可学习的近似最优线性去噪器。

---

## 8. 实验设置

### 8.1 数据集

论文使用 TGB-Seq benchmark 中的 8 个 sequential dynamic datasets。

| Dataset | # Nodes | # Edges | # Timestamps | Repeat ratio |
|---|---:|---:|---:|---:|
| ML-20M | 100,785 / 9,646 | 14,494,325 | 9,993,250 | 0 |
| Taobao | 760,617 / 863,016 | 18,853,792 | 139,171 | 0.17 |
| Yelp | 1,338,688 / 405,081 | 19,760,293 | 14,646,734 | 0.25 |
| GoogleLocal | 206,244 / 267,336 | 1,913,967 | 1,771,060 | 0 |
| Flickr | 233,836 | 7,223,559 | 134 | 0 |
| YouTube | 402,422 | 3,288,028 | 203 | 0 |
| Patent | 2,241,784 | 12,749,824 | 1,632 | 0 |
| WikiLink | 1,361,972 | 34,163,774 | 2,198 | 0 |

这些数据覆盖电商、点评、视频、图片、专利引用、百科链接等场景。

---

### 8.2 Baselines

论文比较了 11 个 baseline，分成几类：

1. Memory-based T-GNNs
   - JODIE
   - DyRep
   - TGN

2. Neighbor co-occurrence / structure encoding methods
   - CAWN
   - DyGFormer
   - FreeDyG
   - CNEN
   - TPNet

3. Simple T-GNNs
   - TGAT
   - TCL
   - GraphMixer

4. Frequency-enhanced baseline
   - FreeDyG

---

### 8.3 评价指标

主实验使用未来链路预测任务，采用 ranking setting。

每个 positive edge 配 100 个 negative edges，评价指标是：

$$
MRR
$$

也就是 Mean Reciprocal Rank。

同时，论文还在 appendix 中报告了 binary setting 下的 AUC 结果。

---

## 9. 主实验结果

### 9.1 整体效果

论文报告，SeqFilter 在 8 个 TGB-Seq 数据集上的平均 MRR 为：

$$
72.85
$$

相比 11 个 baseline，平均提升：

$$
15.82\%
$$

主表中 SeqFilter 的结果为：

| Dataset | SeqFilter MRR |
|---|---:|
| ML-20M | 64.46 |
| Taobao | 83.44 |
| Yelp | 89.61 |
| GoogleLocal | 47.52 |
| Flickr | 89.02 |
| YouTube | 87.15 |
| Patent | 45.68 |
| WikiLink | 75.99 |
| Average | 72.85 |

论文的解释是：SeqFilter 通过 filter-driven network 捕捉时间和结构模式，因此在预测 unseen interactions 时泛化更强。

---

### 9.2 与 memory-based 方法的比较

Memory-based 方法如 JODIE、DyRep、TGN 在部分数据上有效，但整体不稳定。

论文强调：SeqFilter 相比最好的 memory model 最高提升 40.47%。这支持了论文的核心判断：

$$
\text{encoding when a node interacts}
$$

比：

$$
\text{memorizing whom a node interacts with}
$$

更适合 sequential dynamics。

---

### 9.3 与 neighbor co-occurrence 方法的比较

DyGFormer、TPNet 等依赖邻居共现或邻居结构假设的方法，在长程或稀疏时序图上不稳定。

论文认为原因是：shared neighbors 在这类图中很稀疏，未来交互更依赖序列依赖，而不是长期共同邻居关系。

---

### 9.4 与 frequency-enhanced 方法的比较

FreeDyG 也是使用频域思想的 baseline，但论文认为它没有显式处理 abrupt / outlier events。

SeqFilter 相比 FreeDyG 最高提升 51.16% MRR。论文认为这是因为 FSTS Encoder 有显式的低通去噪和级联过滤结构。

---

## 10. Robustness 实验

论文在 YouTube 和 Flickr 上注入不同程度的噪声，噪声比例从 0 到 0.2。

结果显示：

- 当噪声达到 15% 时，SeqFilter 的 MRR 下降少于 5%；
- 当噪声达到 20% 时，SeqFilter 仍保持竞争性，MRR 下降少于 10%。

论文用这个实验支持 Theorem 2 的去噪解释：FSTS Encoder 能抑制噪声并保留底层结构模式。

---

## 11. Efficiency 实验

论文在 Patent 和 Yelp 两个数据集上比较训练速度和 GPU 显存。

### 11.1 时间效率

SeqFilter 在 12 个 T-GNN 中每 epoch 训练时间最快。

论文给出的解释是：

- 模型结构轻量；
- 没有复杂的 neighbor co-occurrence cache；
- memory 和 structure 没有高度耦合；
- 使用 filter-driven design 直接学习表示。

SeqFilter 相比 TGN 最多快 26 倍，并且比 FreeDyG 快一个数量级。

### 11.2 显存效率

SeqFilter 的 GPU 显存消耗也较低。

论文提到，在 Patent 数据集上，相比最耗显存的 TPNet，SeqFilter 显存使用最多减少 221%。

---

## 12. Ablation Study

论文设计了 6 个消融版本：

1. w/o Low-pass：去掉高频去噪；
2. w/o Global：去掉复值线性投影；
3. w/o Conv1D：去掉结构编码器中的 Conv1D；
4. w/o Rhythm：去掉长期 rhythm；
5. w/o Recency：去掉 recency awareness；
6. w RNN：用 RNN 替换 rhythm memory 里的 depthwise convolution。

### 12.1 FSTS Encoder 的作用

消融结果显示，FSTS 的三个部分都重要：

- local denoising 带来 41.64% 提升；
- adaptive global adjustment 带来 7.07% 提升；
- local correlation learning 带来 37.23% 提升。

这说明结构编码器不是只靠某一个组件，而是依赖完整的 composite filter。

### 12.2 Node Rhythm Memory 的作用

去掉长期 rhythm 后，平均性能下降 5.59%。

去掉 recency awareness 后，平均性能下降 3.22%。

用 RNN 替代 depthwise convolution 后，也比完整 SeqFilter 差，论文报告完整 rhythm memory 相比 RNN 版本平均提升 2.37%。

这说明 Node Rhythm Memory 的关键不只是“有 memory”，而是用 filter 方式建模时间节奏。

---

## 13. Visualization 实验

论文可视化了 Node Rhythm Memory 和 FSTS Encoder 中卷积权重的频域响应。

具体做法是：

1. 取模型中学到的 convolution weights；
2. 做 128-point FFT；
3. 得到单边频谱幅值；
4. 画 curve 和 heatmap。

### 13.1 Node Rhythm Memory 的可视化

论文发现，rhythm memory 的权重在 Taobao 和 YouTube 上都比较稳定地关注中频段。

这被解释为：节点行为中存在相对稳定的周期节奏。例如 YouTube 中可能有 4 到 7 天左右的周期。

### 13.2 Structure Encoder 的可视化

结构编码器的频率响应在不同数据集上不同：

- 在 Taobao 上，更保留低频信号，用于捕捉长期结构演化；
- 在 YouTube 上，会增强中高频信号，用于捕捉周期性结构模式和有意义的突发结构变化。

这说明结构 filter 不是固定低通，而是可以根据数据自适应调整频带。

---

## 14. 参数敏感性分析

论文主要分析两个超参数：

1. 能量阈值 $\tau$；
2. 采样邻居数量。

实验发现：

- 如果 $\tau$ 太小，会保留太多高频噪声；
- 如果 $\tau$ 太大，会压制有用的中高频信息；
- 通常 $\tau \in [0.2, 0.4]$ 表现较好；
- 邻居数量太少会缺乏结构上下文；
- 邻居数量超过 20 后性能趋于平稳；
- 邻居越多，可能需要更大的 $\tau$，因为更多邻居会带来更多噪声。

---

## 15. Related Work 中的定位

论文把相关工作分成两类。

### 15.1 Temporal graph neural networks

早期 T-GNN 依赖 node memory 保存历史交互，例如 JODIE、DyRep、TGN。

后来一些方法使用 neighbor co-occurrence encoding，例如 CAWN、DyGFormer、TPNet 等。

这些方法在重复边丰富的场景中有效，但在 sequential dynamics 中会遇到泛化问题。

### 15.2 Frequency-enhanced learning

频域方法已经用于时间序列、动态图、计算机视觉等任务。

FreeDyG 是与本文较接近的动态图库频域方法，但本文认为 FreeDyG 缺少显式的 abrupt event 处理机制。

SeqFilter 的区别是：它直接设计一个 filter-driven network，用可学习 filter 权重捕捉时序和结构模式。

---

## 16. 论文贡献总结

论文的贡献可以总结为三点。

### 16.1 问题层面

论文指出了 sequential dynamics 对现有 T-GNN 的挑战：

- 边重复少；
- 历史邻居身份不稳定；
- 邻居共现稀疏；
- 突发或离群事件影响泛化。

### 16.2 方法层面

论文提出 SeqFilter，包括：

- Node Rhythm Memory：从 who-based memory 转向 when-based rhythm modeling；
- FSTS Encoder：用 local-global-local 的级联 filter 学习结构演化；
- Fusion：融合 rhythm embedding 和 structure embedding 得到节点表示。

### 16.3 理论层面

论文证明：

- FSTS Encoder 是一个由三个 filter 级联组成的 composite filter；
- FSTS Encoder 可以近似 Wiener optimal linear denoiser。

### 16.4 实验层面

论文在 8 个 TGB-Seq sequential dynamic datasets 上验证 SeqFilter，结果显示：

- 平均 MRR 达到 72.85；
- 相比 11 个 baseline 平均提升 15.82%；
- 对噪声鲁棒；
- 训练速度和显存效率较好；
- 消融和可视化支持模型设计。

---

## 17. 这篇论文的内部逻辑链

可以把全文逻辑压缩成下面这条链：

```text
真实时序图存在 sequential dynamics
        ↓
边重复少，未来交互更多由事件顺序驱动
        ↓
传统 T-GNN 的 who-based memory、neighbor co-occurrence、普通结构聚合不稳定
        ↓
需要建模 when + 提取底层结构模式 + 抑制突发噪声
        ↓
把时序图学习看成 signal filtering
        ↓
Node Rhythm Memory 建模节点交互节奏
        ↓
FSTS Encoder 用低通去噪 + 复值频带重加权 + Conv1D 学结构相关
        ↓
理论上是 composite filter，并可近似 Wiener optimal denoiser
        ↓
在 TGB-Seq 上取得更好的 MRR、鲁棒性、效率和可解释频域可视化
```

---

## 18. 论文的关键公式总览

### Temporal event

$$
\delta(t) = (\mathbf{v}_i, \mathbf{v}_j, \mathbf{e}_{ij}(t), t)
$$

### Existing memory module

$$
\mathbf{m}_i(t) \leftarrow MEM(\mathbf{m}_i(t'), \mathbf{v}_i, \mathbf{m}_j(t'), \mathbf{e}_{ij}(t), \Delta t)
$$

### Existing structure encoder

$$
\mathbf{h}_i(t) \leftarrow AGG(\mathbf{v}_i, \{\mathbf{v}_j, \mathbf{e}_{ij}(t), \Delta t, \mathbf{c}_{ij}(t)\} \mid v_j \in \mathcal{N}^k_i(t))
$$

### Co-neighbor encoding

$$
\mathbf{c}_{ij}(t) \leftarrow CO\text{-}NEG(\mathcal{N}^k_i(t), \mathcal{N}^k_j(t))
$$

### Node rhythm memory update

$$
\mathbf{r}^{\iota}_i(t)
= Linear(DepthwiseConv1D([\phi(t) \oplus \mathbf{r}^{\iota}_i(t')]))
$$

### Recency-aware rhythm embedding

$$
\mathbf{r}_i(t)
= LayerNorm(DepthwiseConv1D([\mathbf{r}^{\iota}_i(t') \oplus \phi(t - t')]))
$$

### DFT of structure input

$$
X(f) = DFT(\mathbf{X}(t))
$$

### Low-pass denoising

$$
X_{low}(f)
= I(|X(f)|^2 > \tau) \odot X(f)
$$

### Complex-valued projection

$$
X_{proj}(t)
= IDFT(X_{low}(f) \cdot W_C)
$$

$$
W_C = W_r + i \cdot W_i
$$

### Conv1D structure output

$$
\mathbf{X}_{out}(t)
= Conv1D(X_{proj}(t))
$$

### Final node representation

$$
\mathbf{Z}(t)
= tanh(\mathbf{R}(t) + MeanPooling(\mathbf{X}_{out}(t)))
$$

### Composite filter

$$
X_{out}(f)
= H_{FSTS}(f) \cdot X(f)
$$

$$
H_{FSTS}(f)
= H_C(f) \cdot H_P(f) \cdot H_L(f)
$$

### Wiener optimal approximation

$$
\sup_{f \in [0,1)} |H_{FSTS}(f) - W_{opt}(f)| < \epsilon
$$

$$
MSE_{FSTS} - MSE_{min}
< \epsilon^2 \int_0^1 P(f) df
$$

---

## 19. 一句话概括

这篇论文认为 sequential dynamic temporal graphs 中边重复少、共同邻居稀疏、突发事件多，所以传统 T-GNN 的 memory 和邻居共现机制不稳定；它提出 SeqFilter，把节点表示学习看成信号过滤过程，用 Node Rhythm Memory 建模节点行为节奏，用 FSTS Encoder 在频域中过滤结构信号，并通过 composite filtering 和 Wiener denoising 近似理论解释其有效性。

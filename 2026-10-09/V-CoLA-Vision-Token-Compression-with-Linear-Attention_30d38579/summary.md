---
title: "V-CoLA-Vision-Token-Compression-with-Linear-Attention"
source: https://arxiv.org/pdf/2610.11251v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-11-01 07:00:30"
field: "多模态大模型推理优化"
keywords: ["Vision Token Compression", "Linear Attention", "Hybrid VLM", "Inference Acceleration", "Qwen3.5", "Gated DeltaNet"]
innovations: ["基于线性注意力状态重构误差的独特性感知Token重要性准则", "自适应分块Token合并与Early Exit协同的免训练压缩框架"]
benchmarks: ["MME", "MMB", "GQA", "ScienceQA", "TextVQA", "POPE", "VizWiz", "MMStar"]
---

# 论文速读：V-CoLA-Vision-Token-Compression-with-Linear-Attention

## 一句话总结
本文针对包含线性注意力（Linear Attention）的混合架构VLM（如Qwen3.5），提出了V-CoLA——一种免训练的视觉Token压缩框架，通过利用线性注意力的内在信息保留信号识别关键Token并结合自适应分块合并策略，在仅保留50%视觉Token时仍能保持99.5%的原始性能，推理加速达1.86×至6.15×。

## 研究问题与动机
1. **混合架构下现有压缩方法失效**：当前视觉Token压缩方法主要面向Softmax Attention设计，但Qwen3.5等采用Linear Attention（Gated DeltaNet）的混合架构中，这些方法（如FastV、DART）甚至不如随机裁剪基线。
2. **线性注意力的信息瓶颈**：线性注意力的有界循环状态（bounded recurrent state）造成信息容量上限，导致传统基于注意力分数的浅层集中性和基于特征相似度的Token可区分性均难以有效工作。
3. **缺乏适配Linear Attention的压缩方案**：虽然DTP等方法尝试针对Mamba架构压缩，但仅利用中间输出进行重要性排序，性能次优；针对线性注意力VLM的专用压缩方法仍是空白。

## 核心贡献（创新点）
1. **系统性分析线性注意力下现有方法的失效机制**：揭示了Softmax注意力浅层集中性不足和特征相似度判别力下降的结构原因，并证明线性注意力的选择性保留行为本身编码了Token重要性信号。
2. **独特性感知的Token重要性准则**：结合长期保留率（基于状态重构误差）和短期独特性（基于伪查询探测的状态变化）两个维度评估Token重要性，避免仅依赖最终状态导致的冗余Token选择偏差。
3. **自适应分块Token合并策略**：根据重要性分布密度动态划分Token序列为多个chunk，在重要性密集区域分配更细粒度chunk，稀疏区域分配粗粒度chunk，实现压缩率的自适应分配。
4. **视觉Token Early Exit机制**：发现深层网络中视觉Token重要性急剧下降，提出在深层（如第24层后）直接丢弃所有视觉Token以进一步减少计算负担。
5. **实现级优化适配块级并行**：将所有组件优化为与线性注意力的chunk-wise并行兼容，Extended Gated DeltaRule在4-query推理下获得3.85×加速，Token合并仅增加0.86ms开销。

## 方法详解
**1. 线性注意力的内在指示器（Section 3.1）**
- 基于Gated DeltaNet的状态更新可以视为在线梯度下降过程：$\mathcal{L}_t(\mathbf{S}) = \frac{1}{2}\|\mathbf{S}\boldsymbol{k}_t - \boldsymbol{v}_t\|^2$
- 状态矩阵$\mathbf{S}$作为关联记忆，将每个key映射到对应value
- 实验发现：浅层保留率高且随深度递减，相邻层保留率排名相关系数高，表明线性注意力的选择性保留行为编码了Token重要性

**2. 独特性感知的重要性准则（Section 3.2）**
- **长期保留率**：$\mathcal{R}_t^{\mathrm{imp}} = 1 - \psi(\mathbf{S}_L\boldsymbol{k}_t, \boldsymbol{v}_t)$，其中$\psi$为归一化余弦距离，衡量Token信息被最终状态保留的程度
- **短期独特性**：$\mathcal{R}_t^{\mathrm{uni}} = \psi(\mathbf{S}_t\mathbf{Q}_t^{*\top}, \mathbf{S}_{t-1}\mathbf{Q}_t^{*\top})$，使用伪查询$Q_t^*$探测Token t前后状态变化，衡量该Token引入的新颖性
- **综合重要性**：$\mathcal{R}_t = (1-\lambda)\cdot\mathcal{R}_t^{\mathrm{imp}} + \lambda\cdot\mathcal{R}_t^{\mathrm{uni}}$，默认$\lambda=0.1$

**3. 自适应分块Token合并（Section 3.3）**
- 根据重要性分布将序列划分为$L'=L\times\gamma$个chunk，目标使每chunk重要性总和相等：$\arg\min\sum_{i=1}^{L'}\left|\bar{\mathcal{R}}-\sum_{t=t_i}^{t_{i+1}}\mathcal{R}_t\right|$
- Chunk内加权合并：$o'_i = \sum_{t=t_i}^{t_{i+1}} w_t o_t$，权重由softmax计算：$w_t = \frac{\exp(\mathcal{R}_t/\tau)}{\sum\exp(\mathcal{R}_k/\tau)}$，温度参数$\tau=1.0$

**4. 视觉Token Early Exit（Section 3.4）**
- 基于PPL分析和MME基准验证：在Layer-24后丢弃视觉Token对多模态推理性能影响微小（MME从2401.6降至2398.2）
- 策略：在深层（默认Layer-24）之后丢弃所有视觉Token

## 实验与结果
**模型与数据集**：主要测试Qwen3.5-9B，扩展测试Qwen3.5-27B和InfiniteVL；基准包括MME、MMB、GQA、SQA、TextVQA、POPE、VizWiz、MMStar。

**主要结果（Qwen3.5-9B）**：
- **50% Token保留率**：V-CoLA平均性能达到原始模型的99.5%，MME得分为2398.5（原始2398.2），显著优于FastV（97.5%）、DART（95.5%）、VisionZip（99.0%）、DTP（94.2%）
- **12.5% Token保留率**：V-CoLA仍保持88.0%原始性能，远超其他方法（VisionZip 86.4%、DTP 75.7%）
- **推理加速**：2K-8K Token场景下prefill加速1.86×至6.15×；End-to-End加速最高1.733×（TTFT降低38.6%-64.6%）
- **开销**：压缩开销仅占forward-pass成本的1.6%

## 相关工作脉络
1. **注意力-based方法（FastV、SparseVLM等）**：依赖Softmax注意力分数裁剪Token，但线性注意力架构中注意力分布模式不同，浅层注意力不足以覆盖所有重要Token。
2. **相似度-based方法（DART、VisionZip等）**：假设隐藏表征保留可靠的相似度结构，但线性注意力的状态更新改变了这一假设。
3. **DTP（Delta-guided Two-stage Pruning）**：首个针对Mamba架构的压缩方法，但仅利用中间输出进行排序，性能次优；本文从信息论角度揭示线性注意力的内在重要性信号。
4. **Visual-Word Tokenizer**：基于图像统计在编码器阶段合并patch，属于静态分组成策略，与本文基于状态递推的重要性信号互补。
5. **纯线性注意力架构（RWKV、Mamba-2等）**：本文主要关注混合架构，但方法可推广至其他支持矩阵状态读取的循环骨干网络。

## 局限性与未来方向
1. **架构覆盖范围有限**：实验主要集中在Qwen3.5和InfiniteVL等混合架构，纯Mamba或全线性注意力VLM尚未覆盖，需重新审视唯一性指标与无周期性Softmax层的交互。
2. **线性注意力形式限制**：基于Gated DeltaNet formulation设计，其他线性注意力变体（如其他gate变体）的泛化性需进一步验证。
3. **超参数通用性待验证**：虽然$(w_1,w_2)=(0,4)$和$\lambda=0.1$在多模型上表现良好，但跨更多模型规模的系统性调优仍待探索。

## 研究启发与可借鉴点
1. **从架构内在动力学提取重要性信号**：对于新架构（如各类SSM/线性注意力），分析其状态更新过程本身可能蕴含比外部监督更可靠的重要性指标。
2. **块级并行兼容性设计**：将压缩模块设计为与chunk-wise并行兼容的矩阵操作，可在不破坏现有加速框架的前提下实现高效压缩。
3. **自适应压缩率分配**：按重要性密度动态划分chunk而非固定阈值裁剪，可更好地保留关键语义区域的完整性，尤其适合多尺度视觉内容。
4. **Early Exit与Token压缩的结合**：验证深层视觉信息可丢弃性，实现"浅层精细压缩+深层零Token"的分级策略，最大化效率增益。

## 关键术语表
**Linear Attention / Gated DeltaNet**：用有界状态矩阵替代Softmax注意力，将历史序列压缩为固定大小记忆，计算复杂度从二次降至线性。
**Information Ceiling**：线性注意力的有界状态容量导致信息损失下界，当序列长度超过状态维度时损失线性增长。
**Reconstruction Error as Importance**：利用状态更新的目标函数（最小化$\|\mathbf{S}\boldsymbol{k}_t - \boldsymbol{v}_t\|^2$）反向度量Token重要性，低重构误差表示高重要性。
**Uniqueness-aware Criterion**：结合长期信息保留率和短期状态变化独特性的双维度重要性评分，避免冗余Token选择。
**Adaptive Chunk-wise Merging**：按重要性分布密度动态划分序列区块，在关键区域保留更高分辨率，实现自适应压缩率分配。
**Early Exit for Vision Tokens**：在深层网络直接丢弃视觉Token，将计算预算重新分配到浅层视觉理解阶段。

## 可复现要素
- **数据集**：MME、MMB、GQA、ScienceQA、TextVQA、POPE、VizWiz、MMStar（公开基准）
- **代码/权重**：论文未提及开源声明
- **关键超参**：$\lambda=0.1$（独特性权重）、$(w_1,w_2)=(0,4)$（伪查询窗口）、$\tau=1.0$（合并温度）、Early Exit Layer=24、压缩发生在第1层末尾

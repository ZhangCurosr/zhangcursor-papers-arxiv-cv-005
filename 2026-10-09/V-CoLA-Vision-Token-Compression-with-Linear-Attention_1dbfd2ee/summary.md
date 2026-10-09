---
title: "V-CoLA-Vision-Token-Compression-with-Linear-Attention"
source: https://arxiv.org/pdf/2610.11251v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-10-09 17:29:01"
field: "多模态大模型推理优化"
keywords: ["vision token compression", "linear attention", "VLM efficiency", "Gated DeltaNet", "training-free compression", "hybrid architecture"]
innovations: ["提出基于线性注意力状态重建误差与独特性的无训练 token 重要性度量", "设计自适应分块 token 合并策略，按重要性密度动态分配压缩粒度"]
benchmarks: ["MME", "MMB", "GQA", "SQA", "TextVQA", "POPE", "VizWiz", "MMStar"]
---

# 论文速读：V-CoLA-Vision-Token-Compression-with-Linear-Attention

## 一句话总结
V-CoLA 是一个面向含线性注意力（Linear Attention）混合架构 VLM 的无训练（training-free）视觉 token 压缩框架，通过在状态递推中挖掘 token 重要性与独特性信号，结合自适应分块合并与早期退出机制，在仅保留 50% vision token 时达到原始性能的 99.5%，保留 12.5% 时仍保持 88% 以上性能，并实现 1.86×–6.15× prefill 加速。

## 研究问题与动机
- 视觉语言模型（VLM）中 vision token 数量远超 text token，带来巨大的计算与内存开销，token 压缩是提升推理效率的关键方向。
- 现有主流方法分为注意力-based（依赖 softmax 注意力分数）与相似度-based（依赖隐层特征相似性），但在含线性注意力的混合架构（如 Qwen3.5、InfiniteVL）上均严重失效，甚至不如随机剪枝基线。
- 理论分析表明，线性注意力的有界递归状态构成信息瓶颈（information ceiling），破坏了浅层注意力集中度与 token 可区分性，但也隐含了选择性的 token 重要性信号可供利用。
- 亟需一种与线性注意力架构原生兼容的压缩方法，以充分利用其内在动力学机制。

## 核心贡献（创新点）
- **提出 V-CoLA 框架**：首个专为线性注意力混合 VLM 设计的 training-free vision token 压缩方案，通过状态重建误差与短期独特性联合度量 token 重要性，与依赖 softmax 分数或特征相似性的方法形成本质区分。
- **设计自适应分块 token 合并策略**：根据重要性密度自动将 token 序列划分为重要性均衡的 chunk，并在每个 chunk 内执行加权合并，而非简单 top-k 硬剪枝，从而在关键语义区域保留更高分辨率。
- **实现级优化以兼容 chunk-wise 并行**：将重要性计算、伪查询注入与分块合并全部适配到线性注意力的分块并行范式下，额外运行时开销仅占 forward pass 成本的约 1.6%（合并部分仅 0.86ms）。
- **视觉 token 早期退出机制**：发现深层网络中视觉 token 重要性急剧下降，提出在第 24 层后丢弃所有 vision token，进一步释放计算冗余而不损失多模态推理性能。
- **系统性地揭示了既有方法在混合架构下的结构性失效原因**：通过信息论分析（数据加工不等式）量化了线性注意力状态的信息上限，并从实验上验证了 FastV、DART 等方法在 Qwen3.5 上的失效模式。

## 方法详解
- **线性注意力的内在重要性指示**：基于 Gated DeltaNet 的状态递推公式，将状态视为关联记忆，逐 token 重建误差 $\|S_L k_t - v_t\|^2$ 可作为 token 信息被保留程度的度量；浅层保留率高、深层逐渐衰减，且相邻层间保留率排名具有较高 Pearson 相关性。
- **唯一性感知重要性准则**：最终分数 $\mathcal{R}_t = (1-\lambda)\mathcal{R}_t^{\text{imp}} + \lambda \mathcal{R}_t^{\text{uni}}$，其中 $\mathcal{R}_t^{\text{imp}} = 1 - \psi(S_L k_t, v_t)$ 衡量长程保留程度，$\mathcal{R}_t^{\text{uni}} = \psi(S_t Q_t^*, S_{t-1} Q_t^*)$ 衡量引入该 token 后对邻近查询响应状态的改变量（即独特性），$\psi$ 为归一化余弦距离，$\lambda=0.1$。
- **自适应分块合并**：将序列划分为 $L' = L \times \gamma$ 个 chunk，使每 chunk 内重要性之和尽可能均衡（目标均值 $\bar{\mathcal{R}}$），在密集重要区域分配更细粒度 chunk；chunk 内合并权重 $w_t = \frac{\exp(\mathcal{R}_t/\tau)}{\sum \exp(\mathcal{R}_k/\tau)}$，温度 $\tau=1.0$。
- **早期退出**：通过 PPL 分析与 MME 实验验证，第 24 层之后视觉 token 对输出影响可忽略，直接丢弃以节省后续层的计算。
- **实现优化**：扩展 Gated DeltaRule 支持多查询注入（bypass 输入不参与状态递推），通过移位键构造伪查询 $Q_t^*$，单次前向即可获取所有 $\mathcal{R}_t^{\text{uni}}$；分块划分基于累积重要性曲线的并行矩阵运算。

## 实验与结果
- **模型与数据集**：主实验在 Qwen3.5-9B 上进行，泛化验证覆盖 Qwen3.5-27B 与 InfiniteVL；评测基准包括 MME、MMB、GQA、SQA、TextVQA、POPE、VizWiz、MMStar，均使用 LMMs-Eval 评估。
- **核心结果**（Qwen3.5-9B，对比 FastV/SparseVLM/DART/VisionZip/DTP）：
  - 保留 50% token：平均性能达原始的 **99.5%**，显著领先各基线（第二优 VisionZip 为 99.0%）。
  - 保留 12.5% token：平均性能仍达 **88.0%**，远超 DTP（86.3%）、VisionZip（86.4%）。
  - Prefill 加速：1.86×（50%）至 **6.15×**（12.5%，8K tokens），压缩额外开销仅占 forward 成本约 1.6%。
  - E2E 延迟（INT4 TensorRT on Jetson Thor）：TTFT 降低 38.6%–64.6%，E2E 最高加速 **1.733×**。
- **消融结论**：自适应合并与早期退出均贡献显著；$\lambda \in [0.05, 0.10]$ 表现稳定；窗口 $(w_1,w_2)=(0,4)$ 最优。

## 相关工作脉络
- **FastV（Chen et al., 2024）**：基于 softmax 注意力分数的单步剪枝方法，在 Qwen3.5 上仅保留 97.5% 性能（50% token），因线性注意力中浅层注意力分布稀疏而失效。
- **DART（Wen et al., 2025）**：基于特征相似性的冗余 token 检测，在混合架构中判别性被状态递推破坏，50% 压缩下仅达 95.5% 性能。
- **DTP（Park et al., 2025）**：专为 Mamba 架构设计，利用中间输出做两阶段剪枝，但性能显著低于 V-CoLA（50% token 下 94.2% vs 99.5%）。
- **VisionZip（Yang et al., 2025b）**：基于注意力分数的 progressive 压缩，在 Qwen3.5 上接近随机基线水平，说明注意力分数在混合架构中已不可靠。
- **Gated DeltaNet（Yang et al., 2024）**：本文所针对的线性注意力基础架构，支持选择性写入/擦除/遗忘的递归状态更新，是 Qwen3.5 与 InfiniteVL 的核心组件。
- **SparseVLM（Zhang et al., 2024b）**：文本引导的稀疏化 + 特征回收策略，适用于纯 softmax 架构，未考虑线性注意力状态的动力学特性。

## 局限性与未来方向
- 实验仅覆盖含 Gated DeltaNet 的混合架构（Qwen3.5、InfiniteVL），未扩展到纯 Mamba 架构或全线性注意力 VLM。
- 重要性准则的理论分析主要基于 Gated DeltaNet 公式，对其他变体（如 RWKV、Mamba-2）的泛化性有待系统验证。
- 与 Visual-Word Tokenizer 等编码器侧分组方法的互补性未充分探索。
- 作者指出将方法扩展至形如 $S_t = A_t S_{t-1} + B_t v_t k_t^\top$ 的一般递归状态 backbone 是自然方向。

## 研究启发与可借鉴点
- **从状态动力学提取重要性信号**：对于任何具有有界递归状态的架构，重建误差与状态变化量可作为无需训练的 token 重要性度量，避免对 softmax 分数的依赖。
- **chunk-wise 并行兼容的实现思路**：通过扩展 operator 支持多查询注入与移位键构造，可将原本串行的状态操作适配到硬件友好的并行范式，值得在类似场景中复用。
- **重要性均衡分块替代 top-k 硬剪枝**：将序列按重要性密度自适应划分 chunk 并在块内加权合并，比固定比例裁剪更能保护关键语义区域的分辨率。
- **早期退出与压缩的协同设计**：发现深层视觉依赖衰减后主动退出，可将"省下的 token 预算"重新分配到浅层关键区域，实现更优的效率-性能权衡。
- **信息论视角分析架构限制**：用数据加工不等式量化状态的信息上限，为理解为何既有方法失效提供了普适性的解释框架，可迁移至其他状态空间模型的分析。

## 关键术语表
- **Linear Attention**：用固定大小的递归状态替代 full sequence attention，将复杂度从 $O(N^2)$ 降为 $O(N)$ 的注意力变体。
- **Gated DeltaNet**：一种带选择性写入/擦除/遗忘的门控线性注意力机制，状态更新可解释为在线梯度下降过程，被 Qwen3.5 等模型采用。
- **Token Compression**：通过剪枝或合并减少视觉 token 数量以降低 VLM 推理计算开销的技术。
- **Chunk-wise Parallelism**：将序列切分为多个 chunk，chunk 间保持递归依赖、chunk 内并行计算，以兼顾线性注意力效率与硬件利用率。
- **Reconstruction Error（重建误差）**：状态 $S$ 对 key $k_t$ 的映射与对应 value $v_t$ 之间的距离，反映该 token 信息被状态保留的程度。
- **Information Ceiling（信息天花板）**：有界状态容量 $O(d^2)$ 限制了其对 $O(Ld)$ 序列信息的保留能力，序列越长信息损失越大。
- **Adaptive Token Merging**：根据 token 重要性密度自适应划分 chunk 并在块内加权合并，而非全局均匀裁剪。
- **Early Exit**：在模型较深层直接丢弃 vision token，利用深层视觉依赖衰减的特性节省计算。

## 可复现要素
- 数据集：MME、MMB、GQA、SQA、TextVQA、POPE、VizWiz、MMStar、MMDU、V*-Bench，均为公开基准。
- 代码/权重：论文未明确声明代码开源状态（截至论文版本 arXiv:2610.11251v1）；实验基于公开模型 Qwen3.5-9B/27B 与 InfiniteVL。
- 关键超参：压缩位置在第 1 层末尾；早期退出在第 24 层；$\lambda=0.1$；窗口 $(w_1,w_2)=(0,4)$；温度 $\tau=1.0$；余弦距离作为 $\psi$。

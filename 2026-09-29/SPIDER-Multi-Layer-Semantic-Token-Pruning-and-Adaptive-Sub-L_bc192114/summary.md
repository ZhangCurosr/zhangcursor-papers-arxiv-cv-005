---
title: "SPIDER-Multi-Layer-Semantic-Token-Pruning-and-Adaptive-Sub-L"
source: https://arxiv.org/pdf/2609.34977v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-10-01 03:47:09"
field: "多模态大模型高效推理"
keywords: ["Multimodal Large Language Models", "Token Pruning", "Layer Skipping", "Efficient Inference", "Vision-Language Models", "Training-free Acceleration", "Sub-layer Skipping"]
innovations: ["提出利用视觉编码器中间层与深层特征的多层语义Token剪枝，纠正最后层语义焦点偏移", "首次系统量化MLLM Decoder中Attention与FFN子层对视觉Token的贡献差异，提出自适应子层跳过机制", "构建统一的训练无关框架SPIDER，联合消除编码器端数据冗余与解码器端计算冗余"]
benchmarks: ["GQA", "VQAv2", "MME", "TextVQA", "POPE", "MMB", "MMVet", "MMStar", "DocVQA"]
---

# 论文速读：SPIDER: Multi-Layer Semantic Token Pruning and Adaptive Sub-Layer Skipping in Multimodal Large Language Models

## 一句话总结
SPIDER 是一个训练无关的统一加速框架，通过**利用视觉编码器中层与深层特征的语义差异进行多粒度 Token 剪枝（MSV-Prune）**，以及**在 LLM Decoder 中对 Attention/FFN 子层进行自适应跳过（ASL-Skip）**，联合消除 MLLM 推理中的数据冗余与计算冗余，在 LLaVA-NeXT-7B 上将 FLOPs 降低 79% 的同时保持基线 96% 的性能。

## 研究问题与动机
1. **现有 Token 剪枝仅依赖视觉编码器最后一层特征**，忽视了中间层（如 CLIP ViT Layer 12）包含更丰富的物体中心（object-centric）细粒度信息，深层反而趋向全局语义抽象，导致关键视觉片段被错误丢弃。
2. **现有 LLM Decoder 加速方法以粗粒度整层跳过为主**，忽略了各 decoder 层中 Attention 与 FFN 子层对视觉 Token 的贡献存在显著差异，一刀切式跳过会引入不必要的性能损失。
3. **数据冗余与计算冗余未被统一建模**：Token 剪枝与层跳过通常作为独立模块处理，未考虑视觉 Token 在整条推理流水线（编码器→解码器）中随深度演变的"效用衰减"规律。
4. **细粒度子层冗余缺乏量化分析**：Attention 与 FFN 子层在视觉 Token 处理上的 KL 散度贡献差异尚未被系统研究，无法指导精确的子层跳过决策。

## 核心贡献（创新点）
1. **提出多层语义视觉 Token 剪枝（MSV-Prune）**：同时利用视觉编码器的中间层（Layer 12）和深层（Layer 24）特征进行语义聚类和相似度计算；与已有工作（如 FastV、VisPruner 仅用最后层或纯文本-视觉注意力）的本质区别在于显式利用中间层物体中心特征纠正"最后层偏见"。
2. **量化并揭示 Attention vs. FFN 子层的贡献异质性**：通过引入子层贡献分数（SLC，基于 KL 散度），证明视觉 Token 的 Attention 子层普遍比 FFN 子层更可跳过，这是首次对 MLLM Decoder 内部子层冗余的细粒度系统化度量。
3. **提出自适应子层跳过（ASL-Skip）机制**：融合在线 Token 级可跳过分数（熵 + 图文相关性）与离线子层贡献分数，动态决定每个保留 Token 是否跳过、何时跳过、跳过哪个子层；与 ShortV 等整体层跳过的本质区别在于粒度更细（子层级别）且决策是 Token 自适应的。
4. **构建统一训练无关框架 SPIDER**：将 MSV-Prune 与 ASL-Skip 视为同一个"深度依赖效用分配"问题的两个阶段，而非简单拼接；与既往独立优化方案的本质区别在于端到端的一致性建模。

## 方法详解
**总体流程**：输入图像 → MSV-Prune（视觉编码器侧，减少 Token 数量）→ 保留 Token 送入 LLM Decoder → ASL-Skip（Decoder 侧，跳过冗余子层计算）。

### 1. 多层语义视觉 Token 剪枝（MSV-Prune）
保留 Token 由两部分组成：**锚点 Token** $\mathbf{T}_v^{\text{anc}}$（占比 $r$）和**互补 Token** $\mathbf{T}_v^{\text{cmp}}$（占比 $1-r$）。

- **注意力锚点选择**：对视觉编码器自注意力矩阵沿 Head 维求平均得 $\mathbf{a}_v \in \mathbb{R}^n$，动态阈值 $\tau$ 选出 top-$n \times R \times r$ 个 Token 作为 $\mathbf{T}_v^{\text{anc}}$（公式 1）。
- **粗粒度语义聚类**：对非锚点 Token 的高层特征 $\mathbf{F}_v^L$ 做 K-means 聚类，按簇大小比例分配补全名额 $N_k$（公式 2）。
- **细粒度多层相似度**：对簇内 Token 对 $(i,j)$ 计算多层相似度分 $\mathbf{S}_{ij}$（公式 3）：
  $$\mathbf{S}_{ij} = \underbrace{(\sin(\mathbf{F}_i^L, \mathbf{F}_j^L) + \sin(\mathbf{F}_i^M, \mathbf{F}_j^M))}_{\text{多层簇内相似度}} + \underbrace{\Big(\max_{p \in \mathbf{T}_v^{\text{anc}}}\sin(\mathbf{F}_i^L, \mathbf{F}_p^L) + \max_{p \in \mathbf{T}_v^{\text{anc}}}\sin(\mathbf{F}_j^L, \mathbf{F}_p^L)\Big)}_{\text{与锚点 Token 相似度}}$$
  每簇选取 $N_k$ 个最低分 Token 构成 $\mathbf{T}_v^{\text{cmp}}$，与锚点按空间顺序拼接后送入 LLM。

### 2. 自适应子层跳过（ASL-Skip）
每个保留 Token 在每个 Decoder 层（直至 $L/2$）计算累积跳过分数，超过阈值 $T_{\text{skip}}$ 后进入 Skip 模式。

- **在线可跳过分数** $\mathbf{S}_{sa}(i,\ell)$（公式 4）：
  - 内在信息熵 $\mathbf{E}_{ii}$：Token 隐状态经 unembed 矩阵投影到词表的 Shannon 熵，低熵=语义已收敛=可跳过。
  - 图文相关性因子 $\mathbf{F}_{itc}$：Token 隐状态与文本平均向量的余弦相似度。
  - 保留分 $\mathbf{R} = \mathbf{E}_{ii} + \mathbf{F}_{itc}$，$\mathbf{S}_{sa} = \text{ReLU}(1 - \mathbf{R})$。
- **离线子层贡献分数（SLC）**（公式 5）：在多个基准上计算跳过 Attention 或 FFN 子层后的输出 KL 散度，归一化后取高者作为该层跳过目标 $m(\ell)$。
- **分数融合与累积**（公式 6）：$\mathrm{S}_{\text{fuse}}(i,\ell) = w_1 \cdot \mathbf{S}_{sa}(i,\ell) + w_2 \cdot \mathrm{SLC}^{\text{Norm}}_{m(\ell)}(\ell)$，累加至 $S_{\text{skip}}$，达到 $T_{\text{skip}}$ 则冻结跳过目标和 Skip 模式，后续所有层均绕过该子层。

## 实验与结果
**数据集**：GQA、VQAv2（test-dev）、MME（perception）、TextVQA、POPE、MMB（EN/CN）、MMVet、MMStar、DocVQA。

**模型**：LLaVA-1.5-7B（576 tokens）、LLaVA-NeXT-7B（2880 tokens）、Qwen2.5-VL-3B-Instruct、Qwen3-VL-8B-Instruct。

**主要结果**：
- **LLaVA-NeXT-7B ~50% FLOPs**：SPIDER（N=1920, Rs=50%）综合准确率 **99.11%**，优于 FastV（97.10%）、VTW（89.71%）、ShortV（96.67%）、VisPruner（98.81%）；VQAv2 达 80.2，略超 Vanilla（80.0）。
- **LLaVA-NeXT-7B ~21% FLOPs**：SPIDER（N=710, Rs=30%）综合准确率 **96.09%**，在 MMStar（37.3）、DocVQA（51.5）上显著领先 ShortV（63.56%/14.1）。
- **LLaVA-1.5-7B ~56% FLOPs**：SPIDER 综合准确率 **99.87%**，VQAv2=76.6，GQA=60.9，MMStar=34.9。
- **最强结果**：在 LLaVA-NeXT-7B 上 FLOPs 降低 **79%**，仍保持基线 **96%** 的综合性能。
- **MSV-Prune ablation**：保留 320 tokens（↓88.9%）时，MSV-Prune 达 95.93% Acc，显著优于 VisionZip（94.22%）、VisPruner（89.45%）等。
- **ASL-Skip ablation**：80% FLOPs 下，ASL-Skip 达 99.81% Acc，而 Skip All FFN 仅 71.34%，验证子层选择的必要性。
- **跨架构泛化**：在 Qwen3-VL-8B-Instruct 上 55% FLOPs 达 99.06%，45% FLOPs 达 98.33%，匹敌 ERASE/IVC-Prune 等最新方法。

## 相关工作脉络
1. **FastV（Chen et al., ECCV 2024）**：在 Layer K 后基于文本-视觉注意力固定比例剪枝 Token；SPIDER 在此基础上用多层语义特征替代单最后层注意力，并进一步在 Decoder 侧做子层跳过。
2. **VisPruner（Zhang et al., 2024）**：主张使用视觉线索而非文本-视觉注意力，但仅依赖最后层；SPIDER 扩展为同时利用中层和深层特征，纠正语义焦点偏移。
3. **ShortV（Yuan et al., 2025）**：整层替换为 SparseV 层，对所有视觉 Token 一刀切；SPIDER 在子层（Attention/FFN）级别做自适应跳过，且决策是 Token 个性化的。
4. **VisionZip（Yang et al., CVPR 2025）**：训练无关的多层 Token 压缩，但未探索中间层对细粒度任务的独特价值；SPIDER 明确利用中层物体中心特征提升 TextVQA 等 OCR 任务表现。
5. **LLaVA-PruMerge（Shang et al., 2024）**：需微调的 Token 压缩方法；SPIDER 训练无关模式下即匹敌其性能，微调后可超越。
6. **SparseVLM（Zhang et al., ICML 2024）**：基于文本-视觉注意力稀疏化；SPIDER 证明纯视觉多层语义信息可以超越此类方法的细粒度任务表现。

## 局限性与未来方向
1. **对低质量/退化图像鲁棒性不足**：故障案例分析表明，运动模糊（车牌识别）和医学影像（X 光诊断）场景下，基于注意力的 Token 选择可能保留噪声区域而遗漏关键语义区域。
2. **离线 SLC 预计算需要额外基准采样**：虽显示跨基准具有 rank 一致性（Top-16 重叠约 13/16），但仍需额外 profiling 步骤，未完全实现零开销。
3. **非规则稀疏性难以被 GPU 高效利用**：Token 级动态稀疏在 Decode 阶段无法被标准密集 GPU Kernel 充分利用，需稀疏算子支持。
4. **未来方向**：探索任务感知的 Token 保留策略（如轻量分割模块保护关键区域）、引入解剖学先验适应医疗场景、设计稀疏感知算子加速 Decode 阶段。

## 研究启发与可借鉴点
1. **"中间层特征价值"的发现可迁移**：视觉编码器中间层在物体中心细粒度信息上的优势不仅限于 ViT/CLIP，可推广到 Swin Transformer、ConvNeXt 等不同架构的 Token 选择策略设计中。
2. **子层贡献量化（KL 散度 SLC）的方法论**：该评估范式可直接复用于其他 Decoder 架构（如 Qwen-VL、InternVL）的子层冗余分析，指导定制化的跳过策略。
3. **双阶段统一效用分配思想**：将 Token 剪枝与层跳过视为同一深度依赖问题的两个阶段，而非独立模块拼接，这一视角可迁移至视频理解（时空双冗余）、长上下文推理等场景。
4. **在线熵+离线结构化分数的融合模式**：$S_{\text{sa}}$（输入自适应）与 $SLC^{\text{Norm}}$（架构先验）的加权融合设计简洁有效，可应用于其他需兼顾动态输入和静态结构信息的加速场景。
5. **锚点比例的非单调特性**：Table IX 显示锚点比例 $r$ 存在最优值（0.7），过度保留或过度剪枝均有害，提示后续工作需精细化控制 Token 选择的多样性-重要性权衡。

## 关键术语表
- **SPIDER**：一种训练无关的 MLLM 推理加速框架，整合多层语义 Token 剪枝与自适应子层跳过。
- **MSV-Prune（Multi-Layer Semantic Visual Token Pruning）**：利用视觉编码器中间层和深层特征进行语义聚类和 Token 选择的剪枝策略。
- **ASL-Skip（Adaptive Sub-Layer Skipping）**：根据在线可跳过分数和离线子层贡献分数，自适应决定每个视觉 Token 跳过哪个子层（Attention/FFN）的机制。
- **SLC（Sub-Layer Contribution Score）**：衡量跳过某一子层后对模型输出分布影响的 KL 散度，越低表示该子层越可跳过。
- **Key Object Coverage Ratio（R*）**：中间层 Top-25% 注意力 Token 对关键物体的覆盖率，用于量化语义焦点偏移。
- **VSkip-Attn / VSkip-FFN**：分别绕过 Attention 或 FFN 计算的稀疏 Decoder 层变体。
- **FLOPs Ratio**：加速后模型的实际浮点运算量占原始模型的比例，用于公平比较不同方法的效率。
- **Aggregate Accuracy（Acc.%）**：各基准得分相对于 Vanilla 模型的归一化平均值，用于跨不同量纲基准的统一性能衡量。

## 可复现要素
- **数据集**：GQA、VQAv2、MME、TextVQA、POPE、MMB、MMVet、MMStar、DocVQA（均为公开基准）。
- **代码/权重**：论文未明确声明开源，未提及代码仓库链接。
- **关键超参**：保留比例 $R$、锚点占比 $r=0.7$、聚类数 $K$、跳过阈值 $T_{\text{skip}}=20$、权重比 $w_1/w_2=3$、$w_2=1$、累积阶段至 $L/2$。
- **模型**：LLaVA-1.5-7B、LLaVA-NeXT-7B、Qwen2.5-VL-3B-Instruct、Qwen3-VL-8B-Instruct（均为公开模型）。
- **中间层选择**：默认取 $\lfloor L/2 \rfloor$（LLaVA-1.5-7B 取 Layer 12，Qwen3-VL-8B-Instruct 取 Layer 14）。

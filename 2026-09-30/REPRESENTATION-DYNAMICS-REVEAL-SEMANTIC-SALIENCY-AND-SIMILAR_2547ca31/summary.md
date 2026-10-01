---
title: "REPRESENTATION-DYNAMICS-REVEAL-SEMANTIC-SALIENCY-AND-SIMILAR"
source: https://arxiv.org/pdf/2609.36916v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-10-01 09:38:25"
field: "多模态大模型高效推理"
keywords: ["visual token pruning", "MLLM efficiency", "representation dynamics", "training-free", "semantic saliency"]
innovations: ["系统分析encoder深度依赖的token更新幅度动态，揭示前景显著性与深度的阶段依赖关系", "证明更新方向余弦相似度比输出特征相似度具有更强的语义区分能力", "提出MSDG-Prune，结合更新幅度与方向实现无需训练的group-wise语义剪枝"]
benchmarks: ["LLaVA-1.5-7B", "LLaVA-NeXT-7B", "Mini-Gemini-7B", "Qwen2.5-VL-7B", "GQA", "MMBench", "MME", "POPE", "VQAv2"]
---

# 论文速读：REPRESENTATION-DYNAMICS-REVEAL-SEMANTIC-SALIENCY-AND-SIMILAR

## 一句话总结
本文分析了多模态大语言模型（MLLM）视觉编码器中token表示的动态变化，发现更新幅度与前景显著性的关系具有深度依赖性，且更新方向的语义区分能力优于输出特征；在此基础上提出MSDG-Prune，一种无需训练的视觉token剪枝方法，利用更新幅度估计显著性、更新方向进行语义分组，在极稀疏token预算下实现高效推理。

## 研究问题与动机
1. **推理效率瓶颈**：MLLM将每幅图像编码为数百至数千个visual token，导致prefilling延迟高、KV-cache内存占用大，但其中存在大量冗余信息。
2. **Attention方法的可靠性不足**：Final-layer encoder attention易过度强调信息有限的高范数异常token；text-to-vision attention存在位置偏差，使得attention scores作为剪枝依据并不可靠。
3. **特征相似度与语义不一致**：Vision encoder输出特征的余弦相似度与语义一致性对齐较差——大量高相似度token属于不同语义类别，直接用于剪枝可能误删查询所需的视觉证据。
4. **表示变化cue的理论基础薄弱**：已有利用表示变化的方法（如Representation Shift、EvoCut）主要构造token级重要性分数，但"何时/如何这些变化反映前景显著性与语义一致性"仍缺乏系统分析。

## 核心贡献（创新点）
1. **深度依赖的前景显著性分析**：首次在多个MLLM上系统分析visual token更新幅度的逐层空间分布，揭示早期前景增强、中间sink主导、晚期前景增强、近输出背景增强四个阶段，证明更新幅度作为显著性cue的深度依赖性。
2. **更新方向语义一致性优势**：证明经归一化后的更新方向余弦相似度在区分同类token对与不同类token对上显著优于输出特征相似度（COCO-Stuff上AP提升约10个百分点），为语义分组提供了更可靠的表征依据。
3. **MSDG-Prune训练无关剪枝框架**：提出结合更新幅度（Salience）与更新方向（Grouping）的剪枝方法，通过前景增强窗口内的endpoint displacement估计显著性，再结合查询相关性实现分组预算分配与组内排序选择，无需任何训练且兼容FlashAttention。

## 方法详解
1. **Sink Token过滤**：通过图像均值max-to-median更新幅度比值 $\frac{\max_i v_i^\ell}{\text{median}_i v_i^\ell}$ 定位sink-dominated峰值层 $\ell_\star$，在诊断坐标 $d_s$ 上以阈值 $\tau=50$ 剔除高激活孤立token，剩余候选集记为 $\mathcal{V}$。
2. **更新方向定义**：取初始化阶段后早期状态 $h_i^{\ell_0}$（$\ell_0=2$）与编码器末期状态 $h_i^{\ell_d}$（$\ell_d=L-1$），均归一化后计算位移方向 $\Delta h_i = \text{unit}(h_i^{\ell_d}) - \text{unit}(h_i^{\ell_0})$，再归一化得单位方向 $d_i$。
3. **视觉显著性估计**：在选定前景增强窗口 $[\ell_s, \ell_e]$（CLIP-ViT-L为14→19，Qwen2.5-VL为19→24，宽度5层）内计算endpoint displacement $u_i = \|h_i^{\ell_e} - h_i^{\ell_s}\|$ 作为显著性得分。
4. **查询相关性加权**：对每个投影后visual token $\boldsymbol{x}_i$，计算其与各query token嵌入 $t_j$ 的最大余弦相似度 $\alpha_i = \max_{j \in \mathcal{Q}} \frac{\boldsymbol{x}_i^\top t_j}{\|\boldsymbol{x}_i\|\|t_j\|}$，最终query-weighted saliency为 $s_i = \alpha_i \cdot u_i$。
5. **基于更新方向的语义分组**：对 $d_i$ 使用spherical k-means（$K=20$）进行聚类，目标函数为 $\max \sum_c \sum_{i \in \mathcal{C}_c} d_i^\top \mu_c$（等价于最小化单位方向与中心点的平方弦距离），centroid初始化采用确定性最远点采样。
6. **分组预算分配与组内选择**：各组均值显著性 $g_c = \frac{1}{|\mathcal{C}_c|}\sum_{i \in \mathcal{C}_c} s_i$，经softmax得预算比例 $p_c = \frac{\exp(g_c)}{\sum_r \exp(g_r)}$，分配整数预算 $b_c$ 满足 $\sum b_c = B$；组内保留 $s_i$ 最高的 $b_c$ 个token，按原始顺序拼接送入LLM。

## 实验与结果
- **评估模型**：LLaVA-1.5-7B/13B、LLaVA-NeXT-7B/13B、Mini-Gemini-7B、Qwen2.5-VL-7B/32B。
- **对比基线**：FastV、SparseVLM、DART、DivPrune、VisionZip、HiPrune、EvoCut、PruneSID。
- **评测基准**：GQA、MMBench、MME、POPE、ScienceQA、VQAv2、VQAText、SEED-Bench、VizWiz、HRB8K等。
- **LLaVA-1.5-7B**：11.1% token保留下RelAcc. 95.4%，优于PruneSID/DivPrune的95.1%。
- **LLaVA-NeXT-7B**：5.6% token保留下RelAcc. 91.9%，POPE F1达87.1（PruneSID为76.9，差距10.2pp）；prefilling从218ms降至27.8ms，加速7.8×。
- **Mini-Gemini-7B**：11.1%保留下RelAcc. 95.2%，POPE F1 82.3（PruneSID 76.0，+6.3pp）。
- **Qwen2.5-VL-7B**：11.1%保留下RelAcc. 92.4%，优于PruneSID的90.9%（+1.5pp）。
- **大模型泛化**：在13B/32B模型上持续领先，随token预算降低优势进一步放大。

## 相关工作脉络
1. **Attention-based方法（FastV、SparseVLM、VisionZip、HiPrune）**：依赖encoder或LLM内attention scores选择token；本文与它们的本质区别是不提取attention maps，避免final-layer attention对高范数异常token的过拟合和text-to-vision attention的位置偏差。
2. **Similarity-based方法（DivPrune、DART、CDPruner、PruneSID）**：用输出特征余弦相似度衡量token冗余；本文指出输出特征相似度与语义一致性不对齐（COCO上同类/异类分布重叠严重），改用更新方向相似度构建语义一致的分组。
3. **表示变化方法（Representation Shift、TransPrune、EvoCut）**：将表示变化用于构造token级重要性排序分数；本文的核心区别在于利用变化方向做语义分组、变化幅度做显著性估计，实现group-wise预算分配，而非flat token ranking。
4. **Token Merging方法（ToMe）**：在encoder内merge相似token；本文在LLM prefilling前做token selection而非merging，保留了原始token信息完整性，且无需修改encoder内部结构。

## 局限性与未来方向
1. **需要访问encoder中间状态**：方法需获取vision encoder各层输出以计算更新幅度和方向，无法通过仅暴露最终预测的API接口使用。
2. **新架构适配成本**：泛化到新的backbone需重新分析并定位sink-dominated阶段、前景增强窗口等模型特定组件，依赖对架构的深入理解。
3. **极端压缩下小目标识别受限**：定性分析显示在11.1%预算下对小型物体（如微波下的香蕉）和细粒度属性比较（如同色短裤与鞋子）存在失败案例。
4. **未来方向**：探索自适应窗口定位（而非固定层范围）、将group-wise分组思想迁移至其他压缩任务、与训练后量化等方案结合。

## 研究启发与可借鉴点
1. **"表示动态"作为剪枝cue的系统分析方法**：通过逐层可视化更新幅度空间分布、计算max-to-median浓度比和foreground recall曲线来定位各阶段，这一诊断流程可迁移至ViT/VLM任意模型的表示分析任务。
2. **方向vs幅度分离设计的启示**：将表示变化拆分为direction（语义分组）和magnitude（显著性估计）两个正交信号，分别承担不同功能，避免了单指标同时求解多样性和重要性时的权衡困境。
3. **Group-wise budget allocation机制**：先分组再按比例分配预算的思路可用于任何需要兼顾覆盖率与多样性的特征筛选场景（如RAG文档chunk选择、音频token压缩）。
4. **Query-weighted saliency的简洁性**：仅需visual token与query token的最大余弦相似度，无需额外训练或attention计算，可快速集成到现有MLLM推理pipeline中。
5. **与团队方向结合机会**：可将update direction grouping思想引入多模态检索中的图像patch筛选，或用于视频token压缩任务中的语义一致性分组。

## 关键术语表
**Visual Token Pruning**：在MLLM推理前从图像编码产生的大量visual tokens中选择子集（保留预算B << N），以降低prefilling延迟和KV-cache占用的技术。
**Update Magnitude**：token在相邻encoder层间的表示变化幅度 $v_i^\ell = \|h_i^{\ell+1} - h_i^\ell\|$，映射到图像空间可揭示各层的空间激活模式。
**Update Direction**：将早期与晚期归一化表示的单位向量之差再归一化得到，反映token表示在encoder深度上的语义演化方向。
**Sink Token**：在encoder中间层出现异常大更新（高浓度比值）的孤立高范数token，主要用于内部计算和全局信息存储，携带较少图像特定语义。
**Query-Weighted Saliency**：视觉显著性 $u_i$ 与查询相关性 $\alpha_i$ 的乘积，综合衡量token对当前多模态查询的保留价值。
**Spherical K-Means**：在单位球面上对token更新方向进行聚类的算法，目标函数为最大化组内方向向量与聚类中心的余弦相似度之和。
**RelAcc.（Relative Accuracy）**：各基准得分相对于未压缩基线得分的比值均值（百分比），用于跨模型/跨预算公平比较剪枝性能保持率。
**Endpoint Displacement**：在窗口 $[\ell_s, \ell_e]$ 内计算首尾两层token状态的L2距离 $\|h_i^{\ell_e} - h_i^{\ell_s}\|$，作为累积层间更新的紧凑替代表征。

## 可复现要素
- **数据集**：COCO 2017（公开）、COCO-Stuff（公开）；评测基准均为公开数据集。
- **代码**：论文声明接受后将在 https://github.com/liweixuan-hitsz/MSDG-Prune 开源（双盲评审期间未公开）。
- **权重**：所有基座模型（LLaVA-1.5、LLaVA-NeXT、Mini-Gemini、Qwen2.5-VL的7B/13B/32B）均为公开预训练权重。
- **关键超参**：分组数 $K=20$；saliency window宽度5层（CLIP-ViT-L: 14→19，Qwen2.5-VL: 19→24）；$\ell_0=2$，$\ell_d=L-1$；sink阈值 $\tau=50$；$d_s=650$（CLIP-ViT-L）/ $849$（Qwen2.5-VL）；centroid初始化seed $s=0$；收敛阈值 $10^{-5}$，最大迭代10次。

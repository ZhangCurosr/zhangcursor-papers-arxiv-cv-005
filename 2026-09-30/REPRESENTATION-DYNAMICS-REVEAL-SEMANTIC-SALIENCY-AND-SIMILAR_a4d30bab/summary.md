---
title: "REPRESENTATION-DYNAMICS-REVEAL-SEMANTIC-SALIENCY-AND-SIMILAR"
source: https://arxiv.org/pdf/2609.36916v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-10-01 09:38:24"
field: "多模态大模型高效推理"
keywords: ["visual token pruning", "multimodal large language models", "representation dynamics", "training-free compression", "semantic saliency"]
innovations: ["揭示更新幅度层依赖的前景显著性与更新方向的语义可分性", "提出查询加权显著性+更新方向分组的无训练三级剪枝框架", "在四种不同视觉管线架构上验证强泛化与高压缩下优势放大"]
benchmarks: ["LLaVA-1.5-7B", "LLaVA-NeXT-7B", "Mini-Gemini-7B", "Qwen2.5-VL-7B"]
---

# 论文速读：REPRESENTATION-DYNAMICS-REVEAL-SEMANTIC-SALIENCY-AND-SIMILAR

## 一句话总结
论文提出 MSDG-Prune，一种无需训练的视觉 token 剪枝方法，通过系统分析 vision encoder 中层间表示动力学（更新幅度与更新方向），分别用于估计前景显著性和构建语义一致分組，在 LLaVA-NeXT 上以仅 5.6% token 保留率实现 91.9% 相对性能与 7.8× 预填充加速。

## 研究问题与动机
- MLLM 将每张图片编码为数百至数千个视觉 token，导致预填充延迟与 KV-cache 内存开销巨大，但其中存在大量冗余。
- 现有注意力剪枝方法（如 FastV、SparseVLM）依赖最终层 attention 或文本-视觉 attention，但 final-layer encoder attention 易过度关注高范数离群 token，而 text-to-vision attention 存在位置偏差。
- 现有相似度剪枝方法（如 DivPrune）使用 encoder 输出特征的余弦相似度估计冗余，但该相似度与语义一致性并不对齐——同语义 class 的 token 对与跨 class token 对在输出特征空间中的相似度分布重叠严重。
- 尽管已有工作尝试用表示变化作为剪枝线索，但“何时、如何”通过更新幅度与方向分别捕获前景显著性与语义一致性仍缺乏系统性解释。

## 核心贡献（创新点）
1. **系统揭示表示动力学的两层规律**：更新幅度与前景显著性的关系呈层依赖（早期与 sink 之后均出现 foreground-enhanced 阶段），更新方向的同类/异类可分性优于输出特征，为剪枝信号设计提供理论依据。
2. **提出 MSDG-Prune 无训练剪枝框架**：以 sink 过滤为前提，用前景增强窗口的端点位移估计视觉显著性，用归一化更新方向做球形 k-means 语义分组，再以查询相关性加权实现跨组预算分配与组内选择。
3. **在四种不同视觉管线架构上验证强泛化**：覆盖固定分辨率（LLaVA-1.5）、图像分块（LLaVA-NeXT）、双编码器（Mini-Gemini）与原生动态分辨率+空间合并（Qwen2.5-VL），在不同 retention 下均取得最高或并列最高 RelAcc.。

## 方法详解
- **Sink 过滤**：定位 encoder 中 max-to-median 更新比值峰值层 ℓ⋆，选取低熵高激活坐标 d_s，过滤 |activation|>τ 的高范数 outlier token，剩余候选集记为 V。
- **显著性窗口**：避开 sink-dominated 阶段，在其后选连续 5 层窗口计算端点位移 $u_i = \|h_i^{\ell_e} - h_i^{\ell_s}\|$；默认 CLIP-ViT-L 取 14→19，Qwen2.5-VL 取 19→24。
- **查询相关性**：$\alpha_i = \max_{j \in \mathcal{Q}} \mathrm{cosine}(\boldsymbol{x}_i, \boldsymbol{t}_j)$，衡量投影后视觉 token 与问题 token 的最近邻余弦相似度。
- **查询加权显著性**：$s_i = \alpha_i \cdot u_i$，作为 token 重要性的统一标量。
- **更新方向**：取初始化后 early 层 $\ell_0=2$ 与 near-final 层 $\ell_d=L-1$ 的归一化状态差 $\Delta h_i$，再归一化为单位方向 $d_i$。
- **球形 k-means 分组**：以 $d_i$ 为输入做 K=20 组划分，目标最大化组内方向与单位质心的余弦对齐；初始质心由确定性最远点采样产生。
- **跨组预算分配**：组 $c$ 内平均显著性 $g_c$ 经 softmax 得比例 $p_c$，按 $b_c = \lfloor B p_c \rfloor$ 分配整数预算并补齐余量。
- **组内选择**：各组保留 $b_c$ 个 $s_i$ 最高的 token，按原顺序拼接输出给 LLM。

## 实验与结果
- **模型/基准**：LLaVA-1.5-7B、LLaVA-NeXT-7B、Mini-Gemini-7B、Qwen2.5-VL-7B；评测套件 LMMs-Eval（GQA、MMB、MME、POPE、SQA、VQAv2、TextVQA、SEED、VizWiz、HRB^8K 等）。
- **LLaVA-1.5（11.1% 保留）**：RelAcc. 95.4%，超 PruneSID/DivPrune 0.3pp；192/128/64 token 下三项均为最高或并列最高。
- **LLaVA-NeXT（5.6% 保留，160 token）**：RelAcc. 91.9%，POPE F1=87.1（vs PruneSID 76.9、VisionZip 74.8）；预填充时间 27.8ms，达 7.8× 加速。
- **Mini-Gemini（11.1% 保留）**：RelAcc. 95.2%，POPE F1=82.3（超 PruneSID 6.3pp）；随预算缩小优势扩大。
- **Qwen2.5-VL（11.1% 保留）**：RelAcc. 92.4%，超 PruneSID 1.5pp；对非 CLIP 架构同样有效。
- **大模型扩展**：LLaVA-1.5-13B、LLaVA-NeXT-13B、Qwen2.5-VL-32B 上均保持领先或持平最强基线。

## 相关工作脉络
- **注意力派（FastV、SparseVLM、VisionZip）**：依赖 encoder 或 LLM 内 attention 做 token 打分；MSDG-Prune 不提取 attention，避免高范数 outlier 干扰与位置偏差。
- **相似性派（DivPrune、DART、CDPruner、PruneSID）**：以输出特征相似度度量冗余；本文指出该相似度与语义一致性错位，改用更新方向聚类实现语义一致分组。
- **表示变化派（Representation Shift、TransPrune、EvoCut）**：聚焦 token 级重要性构造；本文进一步将方向用于分组、幅度用于窗口内显著性估计，并结合查询相关性实现分组预算分配。
- **层级注意力派（HiPrune）**：利用不同深度 attention 保留互补 token；本文从动力学阶段视角给出更细粒度的窗口选取依据（避开 sink-dominated 层）。
- **Token 合并派（ToMe、LLaVA-PruMerge）**：在 encoder 内做合并；本文在 LLM prefill 前静态剪枝，兼容 FlashAttention 且无需改变 encoder 结构。

## 局限性与未来方向
- 依赖访问 vision encoder 中间层状态，无法直接用于只提供最终预测输出的黑盒接口。
- 迁移到新 backbone 需人工重新识别 sink-dominated 阶段、foreground-enhanced 窗口与诊断坐标，对架构理解要求较高。
- 在极端压缩（≤11.1%）下，小目标区域与细粒度属性比较仍存在失败案例（如 Qwen2.5-VL 上香蕉、短裤颜色判断）。
- 未来可向自适应窗口选址、跨模型自动 sink 检测、以及与生成式补全结合的方向拓展。

## 研究启发与可借鉴点
- **层间动力学信号优于静态输出**：用端点位移 $h^{\ell_e}-h^{\ell_s}$ 而非单点 attention 或最终特征，能更稳健地捕捉与前景相关的信息变化。
- **“分组-预算-组内”三级框架**：先按语义方向聚类，再按组显著性 softmax 分配预算，最后在组内按查询加权显著性排序，兼顾多样性与任务相关性，可直接迁移到其他 token 压缩任务。
- **Sink 过滤的实用启发**：通过 max-to-median 比值峰值定位 sink-dominated 层、再用单一低熵坐标阈值剔除高范数 token，是一种轻量且跨模型通用的异常过滤策略。
- **小预算下优势放大**：本文在更低 retention 时相对于 PruneSID 的提升增大，提示表示动力学信号在高压缩场景更具判别力，值得在超低带宽推理场景中重点验证。

## 关键术语表
- **Update Magnitude**：相邻/窗口端点层间 token 表征变化的范数，反映该 token 携带的信息量变化强度。
- **Update Direction**：归一化 early 与 late 表征差的单位向量，刻画 token 语义演变的朝向。
- **Sink Token**：在中间层出现极大更新幅度、高范数且激活低熵的异常 patch token，主要承担全局内部计算而非局部图像语义。
- **Query-Weighted Saliency**：视觉显著性 $u_i$ 与查询相关性 $\alpha_i$ 的乘积，作为统一排序标量。
- **Group-wise Pruning**：先以更新方向做球形 k-means 语义分组，再以组平均显著性 softmax 分配保留预算、组内按 $s_i$ 选 token 的三级剪枝策略。
- **RelAcc.**：各基准得分除以 uncompressed baseline 对应得分后取均值，以百分比表示的综合相对性能指标。

## 可复现要素
- **数据集**：COCO 2017（用于动力学分析与 sink 定位）、LMMs-Eval benchmark suite（评测）；基准数据集均为公开。
- **代码/权重**：代码已公布于 https://github.com/liweixuan-hitsz/MSDG-Prune；基于模型（LLaVA、Mini-Gemini、Qwen2.5-VL）均为公开权重。
- **关键超参**：组数 $K=20$；saliency window 宽度 $w=5$；早期/晚期端点 $\ell_0=2,\ \ell_d=L-1$；sink 阈值 $\tau=50$；初始化 seed $s=0$；收敛容差 $10^{-5}$、最大迭代 10。

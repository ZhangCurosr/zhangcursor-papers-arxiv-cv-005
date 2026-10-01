---
title: "RESCOMEMB-EFFECTIVE-AND-EFFICIENT-MULTI-MODAL-EMBEDDING-VIA"
source: https://arxiv.org/pdf/2609.37225v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-10-01 09:39:00"
field: "多模态表示学习与检索"
keywords: ["multi-modal embedding", "multimodal large language model", "visual document retrieval", "token compression", "late interaction", "Matryoshka representation"]
innovations: ["提出可训练的残差齐次压缩 RHC，显式解耦同粒度冗余与跨粒度重复并在 MLLM 输出端端到端优化", "基于 Matryoshka 嵌套监督生成弹性多粒度前缀表示，支持推理时按预算截断", "长度自适应双向晚期交互匹配，通过 TopK 均值与 token 数加权缓解弱匹配累积"]
benchmarks: ["MMEB", "ViDoRe V1", "ViDoRe V2"]
---

# 论文速读：RESCOMEMB: EFFECTIVE AND EFFICIENT MULTI-MODAL EMBEDDING VIA RESIDUAL HOMOGENEITY COMPRESSION

## 一句话总结
论文提出 **RESCOMEMB**，一个可训练的通用多向量多模态嵌入框架，通过多粒度残差齐次压缩（RHC）在显式 token 预算下合并冗余、保留互补证据，并配合嵌套监督与双向晚期交互匹配，在 MMEB 和 ViDoRe 基准上达到最优效果，仅用 ColQwen2.5 37.5% 的视觉 token 预算即实现更高检索精度。

## 研究问题与动机
- **多向量 vs 单向量的权衡困境**：现有 MLLM 嵌入方法要么将图像编码为单向量（丢失细粒度局部证据），要么保留全量视觉 token 序列（存储与两两交互成本过高）。
- **现有压缩方法的不足**：Token pruning/merge 类方法在激进压缩下牺牲整体上下文；Learnable-token 类方法在紧预算下限制局部细节；MURE 等使用非训练性聚类，压缩标准无法获得损失反馈，易丢失相关表示或保留无关 token。
- **粒度间冗余未被显式建模**：现有工作未有效区分"同一粒度内的 token 重复"与"跨粒度间的证据重复"，难以在紧凑表示下同时保留全局语义与细粒度证据。
- **缺乏可弹性伸缩的多粒度表示**：不同下游场景对存储/计算预算的需求差异大，希望模型能生成一套前缀嵌套的多级表示，支持推理时按需截断而无需重训练。

## 核心贡献（创新点）
1. **提出多粒度残差齐次压缩（RHC）框架**：在已上下文化的 MLLM 输出上，显式区分并分别压缩同一粒度内的冗余与跨粒度的重复证据，在给定 token 预算下生成紧凑的多向量表示。（与 MURE 的非训练聚类本质不同，RHC 可端到端反传梯度。）
2. **设计 Intra-Resolution Assessor + Inter-Resolution Compressor 的协同机制**：Assessor 通过重要性分数与相似度矩阵评估 token；Compressor 结合粗粒度锚点计算新颖度，得到优先级向量后执行 merge，保证高优先级 token 不易被合并。（相比单信号选 token，双重信号互补。）
3. **引入 Matryoshka 嵌套监督（MRL）**：对逐级拼接的前缀表示施加嵌套对比损失，使得任意长度前缀均具备语义有效性，支持弹性推理。（区别于仅对完整表示优化的方法。）
4. **提出长度自适应的双向晚期交互匹配（Bidirectional Late-Interaction Matching）**：分别取 query→document 与 document→query 方向 Top-K 均值并依有效 token 数加权，缓解弱匹配累积问题。（相比朴素 MaxSim，显著提升泛化与稳健性。）

## 方法详解
- **多粒度视觉编码**：使用共享 MLLM（Qwen2.5-VL）以原生动态分辨率编码三种视图——全局 1×1、宽高比感知中间视图（1×2 或 2×1，共 2 个 crop）、细粒度视图（2×2，共 4 个 crop）。经最后层隐藏态与投影后分离出文本 token T 与三路视觉序列 g₁、g₂、g₃。
- **Intra-Resolution Assessor**：对 g_k 经 MLP 上下文编码器得 h，importance score $S_i = \text{MinMax}(f_{imp}(h_i))$ 做 stage 内归一化；将交替位置划分为源 token h_A 与目标 token h_B，计算余弦相似度 $\mathcal{C}_{ij} = \frac{h_{A,i}^\top h_{B,j}}{\|h_{A,i}\|\|h_{B,j}\|}$。
- **Inter-Resolution Compressor**：对 k=1  novelty $\mathcal{N}_i=1$；k>1 时 $\mathcal{N}_i = \text{MinMax}\left(1 - \max_{a \in A_k} \frac{h_i^\top a}{\|h_i\|\|a\|}\right)$，衡量当前 token 未被累积粗锚点覆盖的程度。优先级 $\mathcal{P}_i = S_i + \eta \mathcal{N}_i$，merge 评分 $\mathcal{U}_{ij} = \mathcal{C}_{ij} - \alpha \mathcal{P}_i$；源 token 按最高评分分配到目标，直至满足阶段预算 $B_k$，聚合后 l2 归一化得到 r_k。
- **嵌套表示（Matryoshka）**：$\mathbf{E}_{x,k} = \text{Concat}(\mathbf{T}_x, \mathbf{r}_{x,1}, \ldots, \mathbf{r}_{x,k})$，每个 k 级是 k+1 级的前缀，文本前缀固定，视觉部分逐步追加更细粒度。
- **长度自适应双向晚期交互**：取 $K_q = \min(K, N_q)$、$K_d = \min(K, N_d)$，分别收集 Top-K 最高相似 token 索引集 Ω_q、Ω_d，计算 $S_{QT}^{(K)} = \frac{1}{K_q}\sum_{i \in \Omega_q}\max_j e_{q,i}^\top e_{d,j}$ 与 $S_{TQ}^{(K)}$ 对称形式；最终得分 $S_{final} = w_1 S_{QT}^{(K)} + w_2 S_{TQ}^{(K)}$，其中 $w_1 = \min(\rho_{max}, \max(0.5, \frac{N_d}{N_q+N_d}))$，$w_2 = 1-w_1$，实现随文档变长自动偏向 query→document 方向。
- **训练损失**：对每层 k 计算对比损失 $\mathcal{L}_k = -\frac{1}{|\mathcal{I}_k|}\sum_{i \in \mathcal{I}_k} \log \frac{\exp(s_{ii}^{(k)}/\tau)}{\sum_{d_j \in \mathcal{C}_{i,k}} \exp(s_{ij}^{(k)}/\tau)}$，总损失 $\mathcal{L}_{train} = \sum_{k \in \mathcal{K}_B} \lambda_k \mathcal{L}_k$；RHC 的 merge assignment 在反向时视为固定，仅 value aggregation 可导（Appendix B 给出解析梯度）。

## 实验与结果
- **数据集**：MMEB（36 任务，Precision@1）、ViDoRe V1（10 子集，NDCG@5）、ViDoRe V2（7 子集，NDCG@5）。
- **基线**：单向量如 VLM2Vec-V2、GME、LamRA、MMRet；多向量如 ColPali、ColQwen2/2.5、ColMate-Pali、ColMate、MetaEmbed、MURE。
- **主干与预算**：RESCOMEMB 与 ColQwen2.5 同用 Qwen2.5-VL-3B，视觉 token 预算 128/128/128=384，仅为 ColQwen2.5 全量（1024）的 37.5%。
- **MMEB**：RESCOMEMB 平均 Precision@1 达 **67.4**，超 VLM2Vec-V2（64.9）2.5 点，超同规模 VLM2Vec-7B（65.5）1.9 点；分类/检索/ grounding 全面提升。
- **ViDoRe V1**：平均 NDCG@5 达 **90.4**，超最强多向量基线 ColQwen2.5（89.4）1.0 点；其中 InfoQ(94.2)、TabF(95.2)、Gov(97.9)、Health(99.3) 领先明显。
- **ViDoRe V2**：平均 NDCG@5 达 **61.6**，超 ColQwen2.5（60.6）1.0 点；在 ESG Human(69.8)、Bio(64.8)、Eco(62.7) 均创最优。
- **消融要点**：去掉 MRL 在 128/256/384 三种前缀下均使 ViDoRe V2 下降 5.7~7.6 点、MMEB 下降 2.8~3.0 点；去掉 novelty 比去掉 importance 造成更大跌幅（V2 降 3.5  vs 2.1），说明跨粒度新颖度信号更关键；Late-Interaction 中去掉 TopK 或自适应加权各降 7~8 点。

## 相关工作脉络
- **CLIP / SigLIP**：双塔对比预训练，生成全局 pooled 向量，无法保留局部细粒度对应；RESCOMEMB 转向 MLLM 语境化多向量路线。
- **VLM2Vec / VLM2Vec-V2 / E5-V**：将 MLLM 用作单向量嵌入器；RESCOMEMB 保留多向量并引入可训练压缩。
- **MURE**：同样采用多粒度 + 聚类压缩，但聚类不可训练且无法区分 intra/granularity 与跨粒度冗余；RESCOMEMB 以 RHC 实现对两种冗余的联合端到端优化。
- **ColPali / ColQwen2 / ColMate**：基于 late interaction 的多向量检索模型；RESCOMEMB 在其基础上增加可训练压缩与弹性前缀，以更低 token 预算逼近甚至超越其性能。
- **MetaEmbed**：用可学习 abstract token 做全局聚合，紧预算下易丢失局部证据；RESCOMEMB 的有序多粒度 + 残差合并更适配输入-specific 细节。
- **Token merging / pruning（TokenMerger、VisionZip、SCOPE、FOLDER）**：多作用于 MLLM 内部或上游以加速推理；RESCOMEMB 定位为下游可训练压缩器，不改 backbone。

## 局限性与未来方向
- **静态粒度划分**：当前三级 1×1/1×2-2×1/2×2 为固定策略，未根据输入内容自适应调整 crop 数量与位置。
- **RHC 的离散分配在反向时近似固定**：虽 value aggregation 可导，但 merge assignment 依赖 argmax/排序，换步后重新计算，理论上存在最优分配难以精确梯度传递的局限（附录 B 已分析）。
- **仅评估通用与文档检索**：未在其他多模态任务（视频、3D、图表理解）上验证泛化；跨语言支持仅通过数据体现，无显式对齐机制。
- **未来方向**：可探索 task-adaptive 冗余建模（按下游任务动态调节 budget 分布）、将 RHC 迁移至其他表示生成设置（如 generation、continual learning）、以及引入可微排序松弛（SoftSort 等）进一步打通梯度。

## 研究启发与可借鉴点
- **两种冗余显式解耦的思路可迁移**：将 intra-granularity redundancy 与 cross-granularity repetition 分别量化（相似度矩阵 vs 锚点新颖度），适用于任何多尺度/多粒度特征压缩场景，如视频 token 管理、长文档 chunk 合并。
- **Matryoshka 嵌套监督 + 多级前缀**：为弹性推理提供即插即用范式，任何需要"随预算伸缩"的 embedding 任务均可借鉴其层级对比训练。
- **长度自适应双向 late interaction 的权重设计**：$w_1 = \min(\rho_{max}, \max(0.5, N_d/(N_q+N_d)))$ 简洁且有效，可直接复用于其他多向量检索管线替代固定 0.5/0.5 融合。
- **RHC 中优先级保护机制**：将重要性 + 新颖度作为 source ranking 而非 destination assignment 依据，确保高价值 token 不被轻易合并；这种"保源不保目标"的策略在稀疏表征压缩中有参考价值。
- **可结合本团队方向的创新机会**：将 RHC 迁移至图文混合检索、或将双向 late interaction 与 MoE/路由结构结合，实现任务自适应 token budget 分配。

## 关键术语表
- **Residual Homogeneity Compression (RHC)**：一种可训练的粗到细多粒度压缩模块，在同一粒度内合并相似 token，跨粒度优先保留未被粗表示覆盖的残差证据。
- **Matryoshka Representation Learning (MRL)**：嵌套表示学习，使模型输出从粗到细的前缀子序列均具备语义有效性，支持多级截断。
- **Bidirectional Late-Interaction Matching**：双向晚期交互匹配，分别从 query→doc 与 doc→query 方向取 Top-K 最高相似度均值并按有效 token 数自适应加权融合。
- **Intra-Resolution Assessor**：RHC 内的阶段内评估器，输出 token 重要性分数与阶段内相似度矩阵，用于指导后续 merge。
- **Inter-Resolution Compressor**：RHC 内的跨阶段压缩器，结合粗粒度锚点计算新颖度，得到优先级向量后执行 merge 以满足 budget。
- **Novelty Score $\mathcal{N}_i$**：衡量当前 token 与其前序粗锚点集合的最大余弦相似度补集，值越高表示未被粗粒度覆盖的残差信息越多。
- **Preservation Priority $\mathcal{P}_i$**：由重要性 S_i 与新颖度 N_i 线性组合得到的 token 保留优先级，用于抑制高优先级 token 被合并。
- **ViDoRe / MMEB**：ViDoRe 是视觉文档检索基准（V1 含 10 子集，V2 新增 ESG/生物等多领域）；MMEB 是涵盖分类、VQA、检索、定位的 36 任务通用多模态嵌入评测。

## 可复现要素
- **数据集**：MMEB（训练/测试均在 HuggingFace 公开）、ViDoRe V1/V2（公开）、MoCa hard negatives（公开）。
- **代码/权重**：论文 Reproducibility Statement 列出所有公开数据集与 checkpoint 链接；基线 ColQwen2.5-base 亦为公开权重（HuggingFace）。
- **关键超参**：AdamW，global batch=64，8×A100，30k steps；LoRA rank=scale=32，dropout=0.1；learning rate 从 1e-4 线性衰减至 0；温度 τ=0.03；每阶段 RHC budget 128/128/128（总 384 visual tokens）；Late-Interaction 中 K 与 ρ_max 见附录 D（论文未给出具体值，需查附录）。
- **实现细节**：bfloat16、FlashAttention-2、gradient checkpointing；RHC 中 α、η 等比例系数由论文给出但未在正文明确数值（需在附录/代码中确认）。

---
title: "When-Where-and-Why-Hierarchical-Causal-Explainability-for-Vi"
source: https://arxiv.org/pdf/2610.17791v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-11-01 00:49:56"
---

# 论文速读：When-Where-and-Why-Hierarchical-Causal-Explainability-for-Vi

## 一句话总结
本文提出了 VideoFace-X-LLM 框架，通过构建弱监督视频-文本语料 ExplainFace 并微调 MLLM，结合 H-TCC 分层因果解释器，实现了对视频中面部行为的“何时发生（When）、何处显现（Where）、为何必要（Why）”的可验证解释。

## 研究问题与动机
- **核心问题**：现有 MLLM 在细粒度面部行为理解（微表情、时序动态、多模态耦合线索）上表现薄弱，且难以提供忠实、可量化的预测解释。
- **数据缺口**：当前缺乏大规模、时间对齐的面部行为视频-文本数据集；现有 FaceLLM 等工作依赖静态元数据（如 FairFace），无法覆盖视频的时序变化。
- **解释单一化**：既有方法通常只回答 When、Where 或 Why 中的一个维度，时空定位脱节，且难以区分“相关”与“因果必要”。
- **验证缺失**：现有可解释视频模型多依赖定性案例展示，缺乏对解释忠实度（faithfulness）的系统量化评估。

## 核心贡献（创新点）
- **ExplainFace 弱监督语料库**：利用 Gemini 自动生成 CelebV-HQ 的八维结构化描述，并通过双模型审计将矛盾率控制在 ≤2.3%。与已有工作的本质区别：摆脱静态元数据约束，首次面向视频场景实现包含时序动态的弱监督语料构建，并引入可量化的偏差审计流水线。
- **H-TCC 分层因果解释框架**：通过相位 1（多尺度时序显著性）、相位 2（分层空间注意力）、相位 3（定向遮挡因果验证）完整回答 When/Where/Why。与已有工作的本质区别：将解释过程形式化为可测量的因果干预指标，而非仅输出热力图或定性案例。
- **双骨干可迁移微调范式**：在 InternVideo2.5 与 Qwen3-VL 上分别微调 ExplainFace，验证语料与解释框架不绑定单一架构。与已有工作的本质区别：强调“语料+解释”的通用性，同一监督信号可在不同 backbone 上持续 gains。

## 方法详解
- **ExplainFace 构建**：采用固定 System Prompt（Prompt 3.1）引导 Gemini 从 Theme & Narrative、Visual Elements、Actions & Movements、Audio Components、Emotional Tone、Technical Aspects、Temporal Flow、Context & Relationships 八个维度生成 Markdown 描述；各维度映射下游任务（如 Visual Elements → Appearance，Temporal Flow → Deepfake/Temporal Emotion）。
- **偏差审计（Subjectivity Audit）**：第二阶段使用独立 Prompt（Prompt 3.2）识别主观推断语句并替换为可观测事实；通过 BERTScore（保持语义）、BLEU（降低词重叠）、NLI（Entailment/Neutral/Contradiction）验证改写质量，矛盾率仅 1.6%~2.3%。
- **H-TCC Phase 1（When）**：计算目标标签与竞争标签的 margin $m = g(a) - \max_{c\neq a}g(c)$；对每帧 token 计算梯度内积 $s_f = \sum_{t \in \mathcal{T}_f} |\langle \nabla_{\mathbf{e}_t} y_a, \mathbf{e}_t \rangle|$；在 stride=1,4,8 三尺度平滑归一化后相乘得共识得分 $C_f$，峰值即为 $t^\star$；同时报告多尺度 Spearman 稳定性与 Adebayo 随机化 sanity check。
- **H-TCC Phase 2（Where）**：在峰值帧提取 ViT 全 24 层注意力，阈值裁剪（剔除 ≤θ_0.7 值）后残差累积生成空间热图 $R$；通过 Face Lift、Edge Lift、Point Lift 量化注意力是否聚焦面部且回避边缘；同时解析 LLM 输出文本，提取属性相关证据短语排序。
- **H-TCC Phase 3（Why）**：对 Phase 1/2 定位的关键帧与 Top-10% 高注意力 patch 施加高强度高斯模糊，对比随机遮挡后的预测翻转率；因果分数 = Peak Impact / Random Impact，>1.0 表明证据具因果必要性。
- **微调配置**：基于 InternVideo2.5（ViT-24层/1024隐层 + 2层MLP Connector + InternLM2.5-7B），FSDP 全分片，2×A100，学习率 4e-5，batch size 16，8 帧/视频，1 epoch，3% warmup + cosine decay。

## 实验与结果
- **数据集**：CelebV-HQ（Appearance/Action/Emotion，按 ArcFace ID 与源视频划分 ID-independent 8:2 test split）、FaceForensics++（Deepfake 二分类）、DISFA（Action Units 序列强度）。
- **基线**：InternVideo2.5 (Base)、Qwen3-VL、Qwen2.5-VL、LLaVA-NeXT、Phi-4、MiniCPM-V。
- **主要数值**：
  - Appearance Macro F1：19.48% → 32.87%（+13.39）；Qwen3-VL 36.85% → 39.99%（+3.14）。
  - Action Macro F1：16.44% → 20.82%（+4.38）。
  - Temporal Emotion Macro F1：20.04% → 26.81%（+6.77），为所有模型最优。
  - Deepfake：Accuracy 56.32%，F1 45.80%（较 Base 33.96% 提升 +11.84），为最强结果。
  - Action Units（DISFA）：Clip F1 21.6% → 32.2%，Frame F1 21.1% → 31.6%。
- **解释忠实度**：Point Lift 1.40~1.42×，Face Lift 1.23~1.24×，Edge Lift 0.80×（<1 表示回避边缘伪影），Adebayo endpoint 降至 -0.12~-0.22，多尺度稳定性 0.39~0.46，Phase 3 因果分数 Temporal 1.28~1.76×、Spatial 1.44~1.93×，证明定位非偶然且证据具必要性。

## 相关工作脉络
- **通用 MLLM 视频理解**（GPT-4o、InternVideo2.5、Qwen3-VL、LLaVA-NeXT）：擅长开放域 caption/VQA，但缺乏面部细粒度时序先验；本文通过专用语料与微调填补垂直领域空白。
- **人脸基础表征**（CLIP、MARLIN、FaRL、ComFace）：侧重静态特征学习或自监督预训练，未提供时间对齐的视频级监督与因果解释；本文聚焦动态行为理解。
- **时序可解释性**（TimeSuite、SeViLA、VTimeLLM、Grounded-VideoLLM）：多关注单尺度关键帧或时间戳文本编码；本文引入三尺度共识与稳定性度量，避免重叠事件漏检。
- **空间可解释性**（Grad-CAM、Attention Rollout、GMAR、LVLM-Interpret）：热力图缺乏时序对齐且难适配 Transformer 视频编码器；本文在峰值帧提取多层注意力并过滤底部 70% 噪声。
- **因果遮蔽验证**（Zeiler & Fergus、RISE、Fong & Vedaldi）：盲目随机遮蔽效率低且不适用于视频 MLLM；本文依托 Phase 1/2 预定位实施定向遮挡，显著降低计算成本并提升可读性。
- **偏差审计**：现有工作多关注静态人脸数据集的人口统计不平衡；本文首次在视频面部描述生成流程中引入双阶段客观性审计与 NLI 一致性验证。

## 局限性与未来方向
- 模型可能继承 CelebV-HQ 中的群体人口统计偏差，且 MLL

---
title: "When-Where-and-Why-Hierarchical-Causal-Explainability-for-Vi"
source: https://arxiv.org/pdf/2610.17791v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-11-01 00:49:52"
---

# 论文速读：When-Where-and-Why-Hierarchical-Causal-Explainability-for-Vi

## 一句话总结
本文提出 VideoFace-X-LLM 框架，通过弱监督自动构建的 ExplainFace 语料库微调多模态大语言模型，并结合 H-TCC 三层机制（时序定位 When、空间定位 Where、因果必要性验证 Why），首次实现对视频人脸行为的细粒度、可量化验证的因果解释。

## 研究问题与动机
- 现有通用 MLLM（如 GPT-4o、Qwen3-VL、InternVideo2.5）擅长开放域视频描述，但缺乏人脸微表情、时序动态与跨模态因果推理的领域先验，难以捕捉细微且随时间演变的人脸行为。
- 现有可解释方法通常只回答 When/Where/Why 中的单一问题，且多为定性示例展示，缺乏针对视频 MLLM 的时序-空间联合定位与因果必要性定量度量。
- 细粒度人脸行为理解依赖大规模、时间对齐的“视频-文本”数据，但现有语料标注成本极高；图像领域的弱监督构建（如 FaceLLM）难以直接迁移至视频，因视频需同时处理时空对齐与小幅度连续变化推理。
- 人脸数据集隐含人口统计学偏差，直接训练的 MLLM 易继承甚至放大刻板印象，亟需在语料生成阶段引入客观性审计与主体性约束。

## 核心贡献（创新点）
1. **ExplainFace 弱监督语料库**：利用 Gemini 从 CelebV-HQ 自动生成 8 维度的细粒度视频描述，并通过双阶段人工评估与双模型偏差审计验证质量；与已有工作相比，首次将 LLM 生成范式扩展至视频人脸领域，并以 ≤2.3% 的矛盾率与多级检索验证证明其可作为可靠监督信号，解决了视频细粒度标注成本与偏见并存的难题。
2. **H-TCC 分层因果解释框架**：提出 When/Where/Why 三阶段联合定位方法，每步均有严格数学定义；与现有单尺度或单模态解释工作的本质区别在于，它不仅在时空上定位证据，还通过 Targeted Occlusion 验证该证据是否为预测的必要条件，而非仅呈现相关性。
3. **VideoFace-X-LLM 多骨干适配模型**：基于 InternVideo2.5 与 Qwen3-VL 进行端到端 FSDP 微调；与仅做提示工程或冻结 LLM 的工作不同，本文证明 ExplainFace 作为数据中心型监督信号可在不同架构上带来一致且显著的性能跃升。
4. **多维度解释忠实度量化评估体系**

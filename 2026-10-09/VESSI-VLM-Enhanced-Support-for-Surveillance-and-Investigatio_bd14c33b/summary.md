---
title: "VESSI-VLM-Enhanced-Support-for-Surveillance-and-Investigatio"
source: https://arxiv.org/pdf/2610.11674v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-10-09 09:47:27"
field: "多模态监控与视频理解"
keywords: ["视觉语言模型", "视频监控分析", "异常检测", "幻觉评估", "无参考评估", "法证分析"]
innovations: ["提出VESSI模块化帧级提示查询框架，支持可追溯证据检索", "引入CMUS无参考复合评估指标，融合覆盖率/效率/幻觉代理", "建立参考-free与人工标注一致性验证的实验范式"]
benchmarks: ["UCF-Crime", "Normal Video Control Group (50 videos)", "Human-Reference Subset (5909 frames)"]
---

# 论文速读：VESSI-VLM-Enhanced-Support-for-Surveillance-and-Investigatio

## 一句话总结
本文提出 VESSI（VLM-Enhanced Support for Surveillance and Investigations），一种基于视觉语言模型（VLM）的模块化监控分析框架，通过提示驱动的帧级语义查询实现可追溯的证据检索，并引入无参考评估指标 CMUS 在缺乏逐帧标注的监控场景中对四种 7B 参数 VLM 进行综合评估。

## 研究问题与动机
1. **核心问题**：当前视频监控分析仍严重依赖人工审查，耗时且易出错；现有 VLM 方法多聚焦物体检测与跟踪，缺乏结构化查询、可追溯帧检索、操作画像生成等全链条能力。
2. **缺乏评估基准**：监控视频数据通常仅有视频级标签而无逐帧标注，现有评估方法（如 VALOR-EVAL）假设细粒度人工标注存在，无法直接适用于此类场景。
3. **部署约束**：VLM 在嵌入设备或分布式摄像头网络中面临计算资源与内存压力，且对模糊/OOD 输入可能产生幻觉，影响法证场景可靠性。
4. **操作需求**：执法、无人机监控、国防等高风险场景需要平衡语义选择性、推理效率、硬件需求与鲁棒性的框架。

## 核心贡献（创新点）
1. **轻量模块化架构**：提出帧级监控询问与语义提示的可追溯证据检索框架，区别于端到端视频理解模型的刚性依赖，支持灵活集成预处理阶段与硬件适配。
2. **CMUS 评估指标**：引入复合模型效用评分（Composite Model Utility Score），融合检测覆盖率、时间节省、检测率与精炼幻觉指数，适用于无逐帧标注的监控场景，区别于需要人工标注的 VALOR-EVAL。
3. **部署技术指南**：定义面向云基 SOC 与边缘系统（无人机、嵌入式摄像头）的适配指南，现有工作缺乏此类操作可行性分析。
4. **7B 级 VLM 基准测试**：对 LLaVA-1.5、InstructBLIP、Qwen2.5-VL、DeepSeek-VL 四模型进行跨语义选择性、延迟、硬件需求与鲁棒性的综合对比，填补多模型横向比较空白。
5. **参考-free 一致性验证**：通过人工参考 5,909 帧与 50 视频正常控制组，验证 CMUS 排序与 FPR、Hallucination Index 的一致性，证明无参考评估的可靠性。

## 方法详解
- **三阶段流水线**：
  - **采集阶段**：输入 UCF-Crime 数据集的 13 类犯罪视频，按类别组织。
  - **分析阶段**：均匀采样帧（采样率 r=1Hz），每帧应用三个语义复杂度递增的提示（Easy/Medium/Hard），将可视化特征转为文本查询，VLM 返回 Yes/No/Unclear 三元归一化输出。
  - **威胁评估阶段**：对阳性响应按置信度 τ（取模型第10高置信度）过滤，保存标记帧、时间戳、提示与计算日志至 CSV，供人工审查。
- **置信度公式**：$\zeta = \frac{1}{T} \sum_{i=1}^{T} \max_j \sigma(\mathbf{s}_i)_j$，即生成各 token 步最大 softmax 概率均值。
- **CMUS 评分**：$\text{CMUS}_m = \alpha \cdot C_m + \beta \cdot T_m + \gamma \cdot D_m + \delta \cdot H_m$，其中 $\alpha=1.0, \beta=1.0, \gamma=-0.5, \delta=-1.0$；$H_m$ 为五区间桶加权幻觉指数（权重 {-0.5, 0, 0.5, 1.0, 1.5}）。
- **提示可读性**：使用 Flesch-Kincaid Grade Level 量化提示复杂度（Easy: 1.47, Medium: 6.28, Hard: 11.09）。

## 实验与结果
- **数据集**：UCF-Crime（13 类犯罪视频，950 个视频，2025 分钟）+ 50 个正常视频控制组 + 人工标注 65 视频/5,909 帧。
- **基线模型**：LLaVA-1.5、InstructBLIP、Qwen2.5-VL-7B-Instruct、DeepSeek-VL-7B-Chat（均为 ~7B 参数）。
- **关键结果**：
  - **Qwen2.5-VL** CMUS 最高（1.6031），覆盖率 66.9%，检测率 0.1579，幻觉指数 -0.1598，最保守。
  - **InstructBLIP** 覆盖率最高（99.08%），但检测率 0.8304、幻觉指数 1.2030，过度激活。
  - **时间节省**：InstructBLIP 90.96%、Qwen2.5-VL 85.32%、DeepSeek-VL 74.13%、LLaVA 64.59%。
  - **推理速度**：InstructBLIP ~90ms/frame、Qwen2.5-VL ~145ms、DeepSeek-VL ~260ms、LLaVA ~350ms。
  - **人工验证一致性**：CMUS 排序与 FPR_human 排序一致（Qwen2.5-VL < LLaVA < DeepSeek-VL < InstructBLIP）。
- **最强结果**：Qwen2.5-VL 在 CMUS 排名第一，以 66% 视频命中率与 85% 时间节省实现选择性-效率平衡。

## 相关工作脉络
1. **UCF-Crime [16]**：首个大规模监控异常检测数据集，弱监督学习，但无语义标注与交互式查询能力。
2. **UCA [17]**：扩展监控视频理解，引入时空语言标注，需专用模型（Surveillance SwinBERT），缺乏模块化。
3. **LAVAD [25] / VERA [26]**：免训练异常检测流水线，通过帧字幕聚合生成异常分，但聚焦 clip 级而非帧级语义查询。
4. **CPVAD [28]**：边界感知采样减少查询量（146K→5K），保留时序上下文，但无法提供可追溯证据链。
5. **VANE-Bench [29]**：视频语言模型异常检测基准，评测多模型，但仅限 benchmark 导向，无操作法证流水线。
6. **VALOR-EVAL [30]**：多维权衡评估 VLM，但假设细粒度标注，不针对监控流的低监督场景。

## 局限性与未来方向
- **数据集限制**：UCF-Crime 仅为视频级标签，活动可能模糊、重叠、超出镜头或受遮挡/模糊影响。
- **帧级独立处理**：当前流水线未建模时序连续性，无法保证事件的连续定位。
- **硬件特定**：实验基于单工作站与现成模型，未做领域微调，功耗测量为瞬时采样（W）而非积分能耗（Wh）。
- **提示敏感性**：LLaVA 在 medium 提示下 79.88% 输出为 Unclear，表明性能受提示表述影响。
- **未来方向**：轻量时序融合、自适应帧采样、领域适配、模型压缩、无人机/边缘部署控制试验。

## 研究启发与可借鉴点
1. **无参考评估设计**：CMUS 通过"持续阳性激活作为幻觉代理"的思路，可迁移至其他缺乏逐帧标注的安防/医疗视频分析任务。
2. **置信度过滤策略**：使用模型内部 softmax 均值作为过滤阈值，无需额外标注即可实现操作级权衡，适用于资源受限场景。
3. **提示复杂度系统化测试**：三档语义复杂度提示设计可复用于 VLM 稳健性评估，揭示模型对模糊输入的敏感度差异。
4. **模块化帧级流水线**：将视频预处理（分镜、运动检测）与 VLM 推理解耦，提升系统灵活性与跨平台适配性。
5. **多模型横向基准**：固定参数量（7B）比较不同架构，为团队选型提供可操作的决策框架。

## 关键术语表
- **VESSI**：VLM-Enhanced Support for Surveillance and Investigations，一种基于视觉语言模型的模块化监控分析框架。
- **CMUS**：Composite Model Utility Score，复合模型效用评分，融合覆盖率、时间节省、检测率与幻觉指数的无参考评估指标。
- **Hallucination Index ($H_m$)**：精炼幻觉指数，基于五区间桶加权统计模型持续阳性激活程度的操作代理指标。
- **UCF-Crime**：大学犯罪视频数据集，包含 13 类犯罪活动的监控视频，仅提供视频级标签。
- **Frame-level Semantic Interrogation**：帧级语义询问，通过自然语言提示对单帧进行二元决策查询的分析范式。
- **Confidence-based Filtering**：置信度过滤，利用模型生成 token 的最大 softmax 概率均值设定阈值筛选阳性响应。
- **Flesch-Kincaid Grade Level**：弗莱什-金凯德年级水平，量化提示文本可读性的英文教育等级指标。
- **Reference-free Evaluation**：无参考评估，在缺乏逐帧人工标注条件下，通过检测行为模式与对照组进行模型比较的方法学。

## 可复现要素
- **数据集**：UCF-Crime（公开）、50 视频正常控制组（论文未说明来源）、人工标注 65 视频/5,909 帧（论文内部分享，非公开）。
- **代码**：开源，GitHub 链接 https://github.com/SvrCvs/vlm_surv（论文声明）。
- **模型权重**：LLaVA-1.5、InstructBLIP、Qwen2.5-VL-7B-Instruct、DeepSeek-VL-7B-Chat 均为公开模型。
- **关键超参**：采样率 r=1Hz，置信度阈值 τ 取每模型第 10 高置信度，CMUS 权重 α=1.0, β=1.0, γ=-0.5, δ=-1.0。
- **硬件**：Intel i9-14900KF, 62GB DRAM, NVIDIA RTX 4090 (24GB VRAM)，CUDA 12.1。
- **软件环境**：Python 3.10，Transformers 4.49.0，PyTorch 2.1.2/2.6.0，llava 1.2.2.post1，deepseek-vl 1.0.0。

---
title: "VESSI-VLM-Enhanced-Support-for-Surveillance-and-Investigatio"
source: https://arxiv.org/pdf/2610.11674v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-11-01 00:48:06"
field: "监控视频理解与VLM评估"
keywords: ["vision-language models", "video surveillance", "anomaly detection", "hallucination evaluation", "reference-free benchmarking", "digital forensics"]
innovations: ["提出VESSI逐帧提示式监控检索与可追溯告警管线", "引入参考自由CMUS综合指标联合评估检测覆盖率、效率与幻觉风险", "在多类别、多提示复杂度下对比四个7B VLM并给出操作权衡画像"]
benchmarks: ["UCF-Crime", "VANE-Bench", "VALOR-EVAL", "normal-video control group"]
---

# 论文速读：VESSI-VLM-Enhanced-Support-for-Surveillance-and-Investigatio

## 一句话总结
本文提出VESSI框架，将视觉语言模型（VLM）应用于视频监控的逐帧语义查询与证据检索，并在无逐帧标注的现实场景下设计了参考自由的复合模型效用评分（CMUS），为执法与法医调查场景提供了可落地、可追溯的VLM监控分析方案。

## 研究问题与动机
- 传统视频监控分析依赖人工回放，处理大规模录像耗时长、易出错，且缺乏语义级态势感知与可交互取证能力。
- 现有VLM方案多聚焦目标检测/跟踪或通用视频理解，对结构化查询、可追溯帧级证据检索、以及在缺乏细粒度标注的监控数据上的评估仍显不足。
- VLM在模糊或分布外输入下可能出现幻觉与不一致输出，在法医与高可靠场景下可能引发严重误报并削弱操作人员信任。
- 现有数据集与评测缺少面向操作场景的无参考度量：如UCF-Crime仅有视频级标签，VALOR-EVAL等依赖细粒度人工标注且面向静态图像，无法直接衡量监控流式场景中的检测覆盖率、效率与幻觉风险。

## 核心贡献（创新点）
- 提出轻量模块化VESSI管线，支持按类别的逐帧语义提示查询与可追溯告警记录，区别于以往端到端视频理解工作，更强调结构化提示、帧级检索与可审计日志。
- 引入参考自由的CMUS指标，将检测覆盖率、时间节省、检测率与细化幻觉指数联合评估，区别于依赖逐帧ground truth的传统评测，适合长期监控这种弱监督场景。
- 定义适配云端安全运营中心（SOC）与受限边缘/无人机部署的技术指南，强调可移植性与硬件感知部署策略。
- 基准测试四个约7B参数的SOTA VLM，系统揭示语义选择性、推理延迟、硬件开销与鲁棒性之间的操作性权衡，区别于仅报告单一精度指标的比较工作。

## 方法详解
- 三阶段管线：获取（Acquisition）—分析（Analysis）—威胁评估（Threat Evaluation）。
- 获取阶段使用UCF-Crime数据集，包含13个犯罪类别共950个视频；不依赖逐帧标注，视频级类别标签用于构造提示。
- 分析阶段以固定1 Hz均匀采样提取帧，每个帧输入由视频级类别生成的三类提示：Easy/Medium/Hard，语义复杂度递增；提示设计为简短二元决策，输出解析为Yes/No/Unclear三种归一化类别。
- 置信度计算：$\zeta = \frac{1}{T} \sum_{i=1}^{T} \max_j \sigma(\mathbf{s}_i)_j$，其中$\sigma(\mathbf{s}_i)$为生成步$i$的softmax分布，$T$为生成token数，用于过滤与排序告警帧。
- 威胁评估阶段将Positive响应与视频级类别关联，仅保留$\zeta \geq \tau$（取各模型正响应中第十高置信度作为阈值）的帧作为人工复核告警；同时记录帧ID、时间戳、提示、模型、响应、置信度与GPU指标。
- CMUS公式：$\text{CMUS}_m = \alpha C_m + \beta T_m + \gamma D_m + \delta H_m$，权重设置为$\alpha=1.0$、$\beta=1.0$、$\gamma=-0.5$、$\delta=-1.0$。
- 幻觉指数采用五档检测率分桶加权：权重为$\{-0.5, 0.0, 0.5, 1.0, 1.5\}$，分别对应$\lt 10\%$、$10-40\%$、$40-70\%$、$70-95\%$、$\geq 95\%$，以持续高激活作为幻觉风险操作代理。
- 代码开源：https://github.com/SvrCvs/vlm_surv。

## 实验与结果
- 数据集：UCF-Crime（950视频、13类别），另含50个正常视频作为控制组；人工参考子集为65视频、5,909帧的二元标注（2,342正、3,567负）。
- 模型：LLaVA-1.5、InstructBLIP、Qwen2.5-VL-7B-Instruct、DeepSeek-VL-7B-Chat，均在7B参数级别进行公平比较。
- 覆盖率与检测率：InstructBLIP平均覆盖率最高（99.08%），检测率0.83；DeepSeek-VL覆盖率95.62%，检测率0.62；LLaVA覆盖率74.06%，检测率0.27；Qwen2.5-VL覆盖率66.90%，检测率0.16。
- 幻觉指数：Qwen2.5-VL最优（$H_m=-0.1598$），LLaVA次之（$H_m=0.0609$），DeepSeek-VL较高（$H_m=0.7063$），InstructBLIP最高（$H_m=1.2030$）。
- 时间节省：InstructBLIP 90.96%、Qwen2.5-VL 85.32%、DeepSeek-VL 74.13%、LLaVA 64.59%；平均原始视频时长155.77分钟。
- CMUS排名：Qwen2.5-VL 1.6031（最强）、LLaVA 1.1909、DeepSeek-VL 0.6826、InstructBLIP 0.2822。
- 人工参考验证：Qwen2.5-VL的FPR为0.0644、TPR为0.2535；InstructBLIP的FPR为0.8195、TPR为0.9471；正常视频控制、参考自由$H_m$与人工FPR三者得出一致的模型排序。
- 硬件与延迟：InstructBLIP约90 ms/帧、功耗约170W；Qwen2.5-VL约145 ms/帧、约215W；DeepSeek-VL约260 ms/帧、约295W；LLaVA约350 ms/帧、约310W；VRAM使用约16.5–17.5GB。
- 主要结论：VESSI显著提升人工操作员分析效率；Qwen2.5-VL以66%以上视频覆盖率与85%以上审查时间减少，在选择性与效率之间取得最实用平衡；无参考CMUS与有参考人工标注及控制组结果一致，支持其操作可用性。

## 相关工作脉络
- UCF-Crime与UCA基准：前者为弱监督异常检测奠定基础但缺乏语言交互，后者引入时序语言标注但依赖专用密集时空建模，本文通过类别提示与逐帧检索补足可追溯与灵活性。
- VANE-Bench与VALOR-EVAL：前者面向视频语言异常问答基准评估，后者关注开放词汇描述的忠实度与覆盖率并依赖细粒度标注；本文强调执法可操作性与弱监督下的无参考联合评分。
- LAVAD、VERA、ASK-HINT、CPVAD等训练-free或冻结VLM异常检测方案：主要解决片段级异常评分与时序边界；本文更强调类别特定帧级检索、提示复杂度分析与告警可追溯存储。
- VIVID等结构化提示方法：聚焦暴力识别等单一事件分类；本文扩展到多类别、多提示复杂度的综合操作基准与模型对比。
- Clip2Safety、VLPD、LlavaGuard：面向安全合规、行人检测或内容审核；本文专注于法医调查与监控运维的可审计流程与硬件适配指南。

## 局限性与未来方向
- UCF-Crime仅具视频级标签，难以提供 exhaustive 逐帧定位；人工参考仅覆盖65视频、5,909帧，不足以转化为全面ground truth。
- 当前管线对帧独立分析，不建模时序连续性，无法保证事件的连续定位。
- 评测在单机工作站与商用模型上进行，未做领域微调；功耗记录为采样瞬时瓦数而非积分能量，硬件画像具有配置依赖性。
- “Unclear”响应比例与提示配方敏感性影响表现（如LLaVA在中等提示下大量Unclear），需进一步优化归一化与提示鲁棒性。
- 未来方向包括轻量时序融合、自适应帧采样、领域适配、模型压缩及在边缘/无人机/固定摄像头与SOC等不同部署环境中的受控实验。

## 研究启发与可借鉴点
- 参考自由CMUS思路可迁移至其他缺少细粒度标注的流式监控与检测任务，将覆盖率、效率与幻觉风险统一为可比较的操作得分。
- 三类递增语义复杂度的提示设计有助于剖析模型对措辞鲁棒性，可作为VLM能力剖面评估的通用范式。
- 置信度阈值策略（如按正响应排序选取分位阈值）为低资源场景下的告警过滤提供可复用的工程实践。
- 将结构化日志（帧ID、时间戳、提示、响应、置信度与GPU指标）作为可审计证据的设计，值得在法医与合规类系统中推广。
- 与团队结合时，可探索：自适应采样、跨帧时序一致性、对抗/分布外检测以及将CMUS扩展为多任务综合效用指标。

## 关键术语表
- **VESSI**：基于VLM的监控与调查增强框架，采用结构化提示逐帧检索与可追溯告警输出。
- **CMUS（Composite Model Utility Score）**：将检测覆盖率、时间节省、检测率与幻觉指数加权融合的参考自由综合评分。
- **Refined Hallucination Index（$H_m$）**：基于检测率五档分桶的加权指数，用于在无逐帧标注时衡量持续高激活的幻觉风险。
- **置信度$\zeta$**：生成token层面最大softmax概率的平均值，用于排序与过滤正响应告警帧。
- **UCF-Crime**：大规模真实监控异常视频数据集，仅提供视频级类别标签，常用于弱监督异常检测研究。
- **VANE-Bench**：针对视频语言模型在异常检测与定位任务上的基准评测，侧重通用问答能力而非操作化法医流程。
- **LAVAD/VERA/CPVAD**：基于冻结VLM的训练-free异常检测方案，侧重片段级评分与时序边界提取。
- **Prompt Complexity（Easy/Medium/Hard）**：按语义直接性、间接性与推理强度递进的三类查询提示，用于评估模型输出稳定性。

## 可复现要素
- 数据集：UCF-Crime（公开），另含50个正常视频作为控制组；人工参考子集为65视频、5,909帧（论文提供标注规则与统计，但未声明为开放数据集）。
- 代码：已开源，地址为https://github.com/SvrCvs/vlm_surv。
- 模型权重：LLaVA-1.5、InstructBLIP、Qwen2.5-VL-7B-Instruct、DeepSeek-VL-7B-Chat均通过Hugging Face transformers加载。
- 关键超参：帧采样率1 Hz；置信度阈值取各模型正响应中第十高置信度；CMUS权重$\alpha=1.0$、$\beta=1.0$、$\gamma=-0.5$、$\delta=-1.0$；幻觉分桶权重$\{-0.5, 0.0, 0.5, 1.0, 1.5\}$。
- 硬件环境：Intel Core i9-14900KF、62GB DRAM、NVIDIA RTX 4090 24GB VRAM；Ubuntu 20.04 LTS，CUDA 12.1。
- 软件环境：Transformers 4.49.0、PyTorch 2.1.2/2.6.0；LLaVA相关环境使用llava 1.2.2.post1；DeepSeek-VL使用deepseek-vl 1.0.0。
- GPU日志：通过pynvml记录GPU利用率、VRAM、瞬时功率与推理延迟。

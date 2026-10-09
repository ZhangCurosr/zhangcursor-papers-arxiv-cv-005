---
title: "Towards-Unified-Evaluation-of-Prompt-Enhancers-for-Video-Gen"
source: https://arxiv.org/pdf/2610.11736v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-10-09 17:29:25"
---

# 论文速读：Towards-Unified-Evaluation-of-Prompt-Enhancers-for-Video-Gen

## 一句话总结
本文提出 PEBench，首个面向视频生成提示词增强器（Prompt Enhancer, PE）的统一直接评测基准，通过 1,100 条专家验证案例与 1,005 个视觉资产覆盖 T2V/I2V/R2V 三类任务，并配套基于证据提取与 24 项细粒度量表的评估框架，实现脱离下游视频渲染的 PE 质量独立、高效、可解释评测。

## 研究问题与动机
- **核心问题**：现有 PE 开发依赖“生成视频→自动打分/人工盲评”的间接评估闭环，计算与人工作业成本高昂，严重拖慢 PE 的训练与迭代节奏。
- **归因困难**：渲染视频的质量同时受 PE 与下游生成器能力影响，错误难以可靠归因于提示词层面，导致 PE 优化信号噪声大。
- **评测基准缺失**：现有视频生成基准（如 VBench、EvalCrafter、Video-Bench 等）均面向渲染结果设计，无法直接衡量提示词的语义保真度、内部一致性与导演级创作规划能力。
- **动机**：亟需一种解耦 PE 质量与生成器行为的直接评测机制，为 PE 的快速迭代提供细粒度、标准化、低延迟的质量反馈。

## 核心贡献（创新点）
- **提出 PEBench 统一基准**：首个覆盖 T2V-PE、I2V-PE、R2V-PE 三类任务的直接提示词评测基准，包含 1,100 条专家验证案例与 1,005 个视觉资产，跨度 35 种细粒度任务类型。
- **设计证据 grounded 的评估框架**：结合模态感知的视觉事实提取（ImageFacts/AssetFacts）与基于固定量表的 24 项细粒度准则评估，实现跨任务、跨方法的标准化、可解释提示词质量分析。
- **揭示 PE 能力演进趋势与训练策略差异**：发现工业级 PE 正从细粒度描述扩展转向意图保留的电影化规划；caption-reconstruction 类方法（如 WanPE）在导演创作上占优，forward-refinement 类方法（如 SCMaPR、VPO）在语义保真与内部一致性上更强，二者具备互补性。
- **验证直接评测与下游效用的强一致性**：PEBench 评分与专家对增强提示词及下游视频（Wan3.0、MiniMax-H3）的排序达到 Spearman 相关系数 >0.8，证明提示词级评估可可靠反映下游生成效用。

## 方法详解
- **任务定义**：统一三个增强设定：
  - **T2V-PE**：仅输入用户提示词与目标时长（$R = \emptyset$），输出忠实扩展的视频内容描述。
  - **I2V-PE**：输入提示词 + 首帧图像（$\mathcal{R} = \{I_0\}$）+ 目标时长，输出保留初始状态并描述连贯时序延续的提示词。
  - **R2V-PE**：输入提示词 + 多参考图像/视频（$\mathcal{R} = \{a_i\}_{i=1}^N$）+ 目标时长，输出将各资产按指定角色（主体/场景/风格/运动/摄影等）整合为单一连贯提示词。
- **数据构建四阶段流水线**：
  1. **分类法设计**：专家定义 9 大类视频题材与 35 种细粒度任务，并结合 5 类主体、8 类场景、7 类视觉风格、8 类音频域合成 $(genre, subject, scene, style, audio)$ quintuple。
  2. **提示词合成与多样化**：Gemini 3.1 Pro 根据任务、模态、时长合成用户提示词，覆盖从短概念到长分镜脚本、从单镜头到多镜头结构。
  3. **参考资产获取**：通过公开数据集筛选与模型生成（Wan-Image、Seedream、GPT Image 2）构建 1,005 个视觉资产池，支持单图/多图/纯视频/混合配置。
  4. **自动化验证与专家策展**：Gemini 3.1 Pro 执行可行性/一致性检查，十位领域专家剔除歧义、模板化或不可行案例，最终保留 1,100 条高质量案例。
- **三维度 24 项评估准则**：
  - **Semantic Fidelity (SF)**：检验增强提示词对原始指令与视觉参考的忠实保留，含 9 项子准则（GSF、SAP、ANF、DLI、ARP、CCP、LCF、SSP、OTI）。
  - **Internal Consistency (IC)**：检验增强提示词内部的逻辑、时空、物理与音画一致性，含 6 项子准则（SAC、ANL、TSC、SPC、PL、AVS）。
  - **Directorial Creation (DC)**：检验增强提示词是否提供可执行的导演级创作规划，含 9 项子准则（STC、CDS、ETD、LCS、NDE、CIF、SDI、SPS、VSS）。
- **Evidence-Grounded 评估流程**：
  1. **证据构建**：图像输入由 GPT-5.4 提取静态视觉事实（ImageFacts）；含视频参考由 Qwen3.5-Omni-Plus 联合分析提取静态/时序/音视频/跨资产证据（AssetFacts）。证据与源资产绑定，一次性构建后跨方法共享。
  2. **准则评分**：GPT-5.4 作为统一评测器，结合原始指令、增强提示词与共享证据，依据固定准则量表逐项判定“无问题/轻微问题/严重问题”，最终输出各准则通过率与维度综合得分。

## 实验与结果
- **数据集与基线**：在 PEBench 上评测 10 个代表性 PE 系统（H3-Context-IR、WanPE-397B/35B、LTX-2.5-PE、LingBot-Video-PE、SCMaPR、VPO、Prompt-A-Video-Cog/OS、RAPO），所有系统在 T2V-PE 上评测，支持多模态的系统额外在 I2V-PE 与 R2V-PE 评测。
- **主要结果**：
  - **T2V-PE**：WanPE-397B 在 DC 维度排名第一（Overall 78.97），SF 位列第二（74.89）；SCMaPR 以 87.92 领跑 SF；VPO 以 95.07 领跑 IC。
  - **I2V-PE & R2V-PE**：H3-Context-IR 在两个任务的全部三个维度均排名第一，显著优势明显。
  - 无单一方法在所有任务/维度上全面主导，表明当前 PE 存在差异化的能力谱系。
- **关键发现与数字**：

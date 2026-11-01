---
title: "Towards-Unified-Evaluation-of-Prompt-Enhancers-for-Video-Gen"
source: https://arxiv.org/pdf/2610.11736v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-11-01 07:00:29"
field: "视频生成提示工程与评估"
keywords: ["Prompt Enhancer", "Video Generation", "Benchmark", "Prompt Evaluation", "T2V", "I2V", "R2V"]
innovations: ["首个统一视频提示增强器直接评估基准PEBench，解耦PE质量与下游生成器行为", "证据驱动的双阶段评估框架：模态感知事实提取+24项固定rubric准则级评分", "揭示PE从描述扩展向cinematic planning演进的工业趋势及caption-reconstruction与forward-refinement的互补能力"]
benchmarks: ["PEBench", "VBench", "EvalCrafter", "MSAVBench", "OpenS2V-Nexus"]
---

# 论文速读：Towards-Unified-Evaluation-of-Prompt-Enhancers-for-Video-Gen

## 一句话总结
本文提出了PEBench——首个面向视频生成提示增强器(Prompt Enhancer, PE)的统一直接评估基准，通过24项细粒度标准在语义保真度、内部一致性、导演创作三个维度评估增强后的提示质量，无需渲染下游视频即可衡量PE能力。

## 研究问题与动机
- 现有PE评估依赖"渲染视频→自动打分/人工盲评"的间接闭环，计算与人力成本高，反馈延迟大
- 渲染视频结果混杂了PE质量与下游生成器能力，难以将错误可靠归因于提示增强器本身
- 缺乏跨T2V-PE、I2V-PE、R2V-PE三种任务的统一评估框架，且现有视频生成基准（如VBench、EvalCrafter等）均面向已渲染视频而非提示本身

## 核心贡献（创新点）
- **首个统一PE直接评估基准**：PEBench覆盖T2V/I2V/R2V三种任务，含1,100个专家验证案例与1,005个视觉资产，35类细粒度任务，填补了PE独立评估的空白
- **证据驱动的双阶段评估框架**：先通过模态感知的事实提取构建ImageFacts/AssetFacts（图片用GPT-5.4、视频用Qwen3.5-Omni-Plus），再由同一评估器按24项固定量表打分，确保跨方法可比性
- **揭示PE演进趋势与能力分化**：系统评测10个开/闭源PE，发现工业级PE正从"细粒度描述扩展"转向"意图保留的cinematic planning"；caption-reconstruction类（如WanPE）擅长导演创作，forward-refinement类（如SCMaPR、VPO）擅长语义保真或内部一致性
- **验证prompt-level评估的下游可用性**：PEBench三维评分与Wan3.0/MiniMax-H3生成的下游视频专家排名Spearman相关系数达0.809–1.000，证明提示级评估可可靠反映下游效用

## 方法详解
- **任务定义**：T2V-PE（纯文本输入→详细提示）、I2V-PE（文本+首帧→时间延续描述）、R2V-PE（文本+多参考图像/视频→整合多源资产的 coherent prompt）
- **数据构建四阶段流水线**：①专家定义9大类35类任务 taxonomy + 5类主体/8类场景/7类风格/8类音频组合；②Gemini 3.1 Pro 合成多样化用户提示（指令粗细、镜头结构单/多 shot 变化）；③参考资产来自公开数据集（OpenS2V-Nexus、OpenVid-1M、Goku等）与模型生成（Wan-Image、Seedream、GPT Image 2），支持单图/多图/视频/混合条件；④两阶段质量控制：Gemini自检可行性与角色对应→10名领域专家审阅
- **24项评估标准**（三维度）：
  - Semantic Fidelity (SF) ×9：GSF(体裁风格)、SAP(主体属性)、ANF(动作叙事)、DLI(对白歌词)、ARP(音频要求)、CCP(相机约束)、LCF(光照色彩)、SSP(场景空间)、OTI(屏上文字)
  - Internal Consistency (IC) ×6：SAC(主体连续性)、ANL(动作叙事逻辑)、TSC(时序状态连续)、SPC(空间一致性)、PL(物理逻辑)、AVS(音视频同步)
  - Directorial Creation (DC) ×9：STC(分镜时序完整性)、CDS(相机设计具体性)、ETD(剪辑转场设计)、LCS(光照色彩规范)、NDE(叙事发展)、CIF(核心意图聚焦)、SDI(音效设计整合)、SPS(主体表演具体性)、VSS(视觉风格规范)
- **评估流程**：①模态感知证据提取（ImageFacts/AssetFacts，一次性构建供所有方法共享）；②GPT-5.4作为统一评估器，按固定rubric逐项判定"无问题/轻微/严重"，输出pass rate与维度综合分（考虑问题数量与严重程度）

## 实验与结果
- **评测基线**：10个PE系统（H3-Context-IR、WanPE-397B/35B、LTX-2.5-PE、LingBot-Video-PE、SCMaPR、VPO、Prompt-A-Video-Cog/OS、RAPO）
- **T2V-PE**：WanPE-397B在DC维度第一(78.97)、SF第二(74.89)；SCMaPR SF最高(87.92)；VPO IC最高(95.07)
- **I2V-PE / R2V-PE**：H3-Context-IR在SF/IC/DC三维均领先（IC分别为94.50/94.90，DC为100.00/100.00，注：表中显示为满分或接近满分，需核实原文表格）
- **关键发现**：
  - 原提示在T2V-PE上SF=100但DC仅33.56，说明仅保真意图远不足以下达完整导演规划
  - 输出长度与DC无正相关：最长四分位比最短低5.19–10.09分
  - 时序复杂度：3+镜头事件SF/IC/DC分别下降3.67/3.92/1.98分；3+显式镜头下降9.97/7.61/5.52分
  - 参考复杂度：3+图像SF下降7.42分；混合模态(Hybrid)SF仅45.62、DC仅55.69，显著低于单模态
- **人类对齐**：PEBench Evaluation与专家排名的Spearman相关0.818–1.000，显著优于Direct VLM Scoring（0.464–0.618）；下游视频层面Wan3.0/MiniMax-H3相关亦达0.809–1.000

## 相关工作脉络
- **视频生成基准**（VBench、EvalCrafter、Video-Bench、MSAVBench、StreamAV-Bench等）：面向已渲染视频质量评测，无法解耦PE与生成器贡献；PEBench首次实现prompt-level直接评估
- **Prompt Enhancement方法**：WanPE采用real-video caption反向重建训练（caption-reconstruction范式），获得强导演创作能力；SCMaPR/VPO采用policy-conditioned rewriting与preference optimization（forward-refinement范式），擅长语义/一致性约束
- **多模态评估模型**：GPT-5.4与Qwen3.5-Omni-Plus用于事实提取与评估，体现VLM作为通用评估器的趋势
- **Reference-conditioned生成**（OpenS2V-Nexus等）：提供多参考资产构建资源，但仅支持T2V侧评测

## 局限性与未来方向
- 基准规模有限：1,100案例虽覆盖广，但无法代表快速演进的生成需求全貌；未来可扩展更多领域、语言、参考配置与新控制格式
- I2V/R2V基线模型覆盖不足：目前仅少数闭源系统支持多模态PE（如H3-Context-IR），随新方法出现将持续扩充
- Hybrid模态（图+视频混合参考）当前表现显著偏低，暴露参考融合瓶颈
- 未深入分析开源小模型与闭源大模型的能力差距机制

## 研究启发与可借鉴点
- **解耦评估思路可迁移**：将"评估对象与生成过程解耦"的思路可推广至图像/3D/音频生成PE评估，避免端到端评估的混淆偏差
- **证据先行的评估架构**：先构建一次性的模态感知evidence（ImageFacts/AssetFacts），再统一评估，既节约算力又保证跨方法公平比较；可复用于其他"增强器→生成器"链路评测
- **三维评估框架（保真/一致/创作）具有良好的可解释性**：DC维度鼓励结构化导演规划而非冗余扩写，对PE训练目标设计有启示——可将DC相关指标纳入reward/loss
- **复杂条件下能力衰减分析可作为系统性评测模板**：按事件阶段数/镜头数/参考数量/模态混合度的分层分析，为后续工作提供标准化的压力测试范式
- **Caption-reconstruction vs Forward-refinement的互补性**：未来PE系统可结合两者——用真实视频caption学习cinematic knowledge，同时引入语义验证与逻辑约束防止幻觉

## 关键术语表
- **Prompt Enhancer (PE)**：将用户简短指令与多模态参考转化为结构化、细节丰富的导演级提示词的模型/系统
- **T2V-PE / I2V-PE / R2V-PE**：Text/Image/Reference-to-Video Prompt Enhancement的缩写，分别对应纯文本、单首帧图、多参考资产的三种增强任务
- **Semantic Fidelity (SF)**：增强后提示对原始意图与参考资产信息的保真程度（9项细粒度标准）
- **Internal Consistency (IC)**：增强后提示在叙事、时空、物理、音视频同步方面的逻辑自洽程度（6项标准）
- **Directorial Creation (DC)**：增强后提示在分镜规划、相机设计、剪辑、光影、音效等方面的导演级创作指导质量（9项标准）
- **Caption-reconstruction范式**：以真实视频caption为监督信号反向重建用户提示的训练策略（如WanPE）
- **Forward-refinement范式**：通过策略条件重写、原子语义验证、多反馈偏好优化直接迭代改进提示的策略（如SCMaPR、VPO）
- **Evidence-Grounded Evaluation**：先提取可追溯的视觉事实(evidence)，再基于固定rubric进行准则级评估的框架

## 可复现要素
- **数据集**：PEBench 1,100案例 + 1,005视觉资产；论文声明发布verified prompts与合法可分享的参考资产（见Project page: https://github.com/yawen-shao/PEBench）
- **代码/评估框架**：评估框架开源，使用GPT-5.4与Qwen3.5-Omni-Plus作为事实提取与评估器（需API访问）
- **关键超参**：论文未明确列出；评估器使用固定rubric与structured checklist，按quartile分组的长度分析为后验统计

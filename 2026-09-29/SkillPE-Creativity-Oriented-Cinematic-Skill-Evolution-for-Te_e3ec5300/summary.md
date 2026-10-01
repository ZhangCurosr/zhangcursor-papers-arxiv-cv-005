---
title: "SkillPE-Creativity-Oriented-Cinematic-Skill-Evolution-for-Te"
source: https://arxiv.org/pdf/2609.34335v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-10-01 03:48:13"
field: "文本到视频生成与提示工程"
keywords: ["text-to-video", "prompt engineering", "skill evolution", "cinematic generation", "creative AI", "video generation"]
innovations: ["提出三路参考分类（共振器/不和谐/偏离者）驱动的电影技能演化框架", "细粒度结构化电影技能表示，将导演知识分解为可操作的多维度镜头格式", "视频生成驱动的 PF/CQ/NA/CR 四维度评估选择，实证刻画保真度-创意权衡"]
benchmarks: ["StoryEval", "VBench", "VidProM"]
---

# 论文速读：SkillPE: Creativity-Oriented Cinematic Skill Evolution for Text-to-Video Prompt Engineering

## 一句话总结
论文提出 SkillPE，一种以创意为导向的文本到视频提示工程框架，通过从专家种子技能出发，利用电影参考进行保守细化（共振器/不和谐）与可控发散探索（偏离者），生成可复用电影技能库，从而提升生成视频的 cinematic quality、narrative appeal 和 creativity。

## 研究问题与动机
- 现有 PE 方法（如 Prompt-A-Video、VPO、RAPO）主要在提示文本层面操作，缺乏对镜头逻辑、构图、光影等可复用电影结构的系统性建模。
- 基于固定 Agent 管道的 PE 方法（如 Mora）工具池静态，难以适应不同创意语境的需求，限制了创造性反思与自我改进。
- 已有技能自进化工作（如 Voyager、SkillWeaver）依赖随机探索或事后总结，偏重实用性而忽视创造性，而叙事视频生成对创意广度至关重要。
- 如何把外部电影专家的创作机制转化为可复用的创意变体，同时显式保留用户意图的语义，尚未被充分探索。

## 核心贡献（创新点）
1. **结构化电影技能表示**：将抽象的电影知识转化为细粒度、可复用的镜头逻辑、构图、光影、声音设计等指导，与仅做词法改写的基线方法本质不同。
2. **三路参考分类框架**：将电影参考划分为共振器（good matches）、不和谐（weak matches）和偏离者（creatively useful near-misses），前者用于保守细化，后者用于受控发散探索，区别于现有方法的单一路径优化。
3. **视频基评估驱动的技能选择**：通过 PF/CQ/NA/CR 四维度评估生成视频来筛选最优技能库，实证刻画了保真度—创意的权衡关系，填补了创意导向 PE 评估范式的空白。

## 方法详解
**技能公式化（Skill Formulation）**：与导演/电影美学专家合作构建 20 个初始技能，每个技能包含描述、镜头模板、应用场景、镜头逻辑、音乐逻辑和示例；再通过 Gemini 3.1 Pro 将其归一化为细粒度格式，每个镜头补充 duration、location、atmosphere、angle、composition、lighting、cinematography、visual content、dialogue、sound effects 字段。

**自适应技能演化（Adaptive Skill Evolution）**：
- **参考发现**：从 LSMDC（60,521 个片段）和 Movie101 构建参考库；每技能先检索 200 个语义相似视频，再加 19×10=190 个多样性视频组成 390 候选池，用多模态 LLM 按 6 维度评分后选取 10 个共振器、15 个偏离者、5 个不和谐。
- **R&D 细化**：共振器提炼核心不变量与 cinematic patterns，不和谐推导负向适用边界，共同生成 $s_i^{\text{R\&D}}$。
- **偏离者探索**：在 9 个维度（镜头结构、运镜、视角、空间调度、光影、节奏、过渡、表现性视觉装置、视听协调）上进行突变，生成 Bold（修改 1–2 维）、Wilder（修改 3–4 维，含至少一个结构维）、Extreme（修改≥5 维，含镜头结构或表现性视觉变换）三级变体。
- **候选集**：$\bar{\mathcal{S}}_i = \{s_i^{\text{Seed}}, s_i^{\text{R\&D}}, s_i^{\text{Bold}}, s_i^{\text{Wilder}}, s_i^{\text{Extreme}}\}$。
- **评估与选择**：在开发集（400 个 VidProM 提示）上生成视频，由 Gemini 3.1 Pro 从 PF、CQ、NA、CR 四维度评分（7分Likert），构造三类技能库：Top Creativity（PF≥中位数时取 CQ/NA/CR 均分最高）、Top-2 Creativity（同理取前2）、Top Overall（四维度均值最高）。

## 实验与结果
- **数据集**：StoryEval（故事叙述）、VBench（视频质量）、VidProM（开发集）；两个开源视频生成骨干 MiniMax-H3 和 LTX-2.5。
- **基线**：Raw Prompt、Gemini 3.1 Pro 直接改写、VPO、Prompt-A-Video、RAPO、Mora、Seed Skill（仅种子技能）。
- **StoryEval 最佳结果**（MiniMax-H3）：SkillPE Top Overall 在 CQ=6.61、NA=5.91、CR=4.89、Overall=5.94；较最强外部基线（Gemini 3.1 Pro）提升 1.40 分（Overall），较 Seed Skill 提升 0.51 分。
- **VBench 最佳结果**（LTX-2.5）：SkillPE Top Overall Overall=5.88，较 Seed Skill（5.37）提升 +0.51；CQ=6.43、NA=5.44、CR=5.49 均为最高。
- **人工评测**（10位 annotator，LTX-2.5）：SkillPE Top Creativity 总体得分 5.97，Seed-skill PE 为 5.61（+0.36），最强非技能基线 Gemini 3.1 Pro 为 4.86（+1.11）。
- **消融关键发现**：从 Bold→Wilder→Extreme，CR 单调递增（StoryEval：4.91→5.50→5.88），PF 单调下降，验证了保真度—创意权衡；R&D 分支贡献最终库中 15–25% 的技能。

## 相关工作脉络
1. **Prompt-A-Video / VPO / RAPO**：词法级 PE 方法，通过偏好对齐/RL/检索增强重写提示；SkillPE 在可复用电影结构层面推理而非句子级改写。
2. **Mora**：多 Agent PE 框架，依赖固定管道和静态技能池；SkillPE 通过参考引导的探索动态演化技能库。
3. **Voyager / SkillWeaver**：自主技能发现工作，依赖随机 rollout 或成功轨迹总结，偏重 utility；SkillPE 引入偏离者概念鼓励创造性探索。
4. **Mind-Brush / NEWTON / GenAgent**：多模态 Agent 用于图像/视频生成；多采用角色专用 Agent 协作，而非结构化技能演化。
5. **Movie101 / LSMDC**：电影理解基准数据集，本文作为技能演化参考来源，区别于此前仅用作训练数据的用法。
6. **StoryEval**：故事叙述评估基准，本文在其上验证 cinematic/skill-based PE 的叙事增益。

## 局限性与未来方向
- 当前实验仅覆盖两种视频生成骨干和 10 秒短片，未扩展到长视频或其他视频模型。
- 技能库初始化仅来自 20 个专家技能和有限电影参考，可能未覆盖所有流派和视觉风格。
- 保真度—创意权衡无法完全消除，需未来探索查询自适应或用户可控的选择策略。
- 离线构建技能库有较高一次性成本（~68.2K API 调用、~625 GPU小时）；推理时额外引入 2–3 次 LLM 调用。
- 部分阶段依赖闭源模型（Gemini 3.1 Pro），未来计划用更强开源多模态模型替代以提高可复现性。

## 研究启发与可借鉴点
1. **三路参考分类（共振器/不和谐/偏离者）**：概念学习中利用正例、反例和近失例的方法可迁移至其他技能的创意演化，如代码生成技能库、写作风格库等。
2. **视频生成驱动的评估选择**：用生成结果本身（而非纯文本指标）评估技能质量，这一范式可推广至图像生成、3D 内容生成等领域。
3. **受控突变等级（Bold/Wilder/Extreme）**：在多维度空间设置分级修改强度，可用于其他创意生成任务（如文案、音乐编排）的探索。
4. **保真度—创意权衡的实证刻画**：通过系统消融量化 trade-off 曲线，为后续研究提供可复用的评估协议（四维度 7 分制）。
5. **细粒度电影技能格式**：将导演知识分解为 duration/composition/lighting 等结构化字段，可与现有文本到视频模型的 prompt schema 对齐，便于即插即用。

## 关键术语表
**SkillPE**：以创意为导向的电影技能演化提示工程框架，用于文本到视频生成。
**Resonators（共振器）**：与种子技能高度对齐的电影参考片段，用于提炼可复用 cinematic patterns 和应用指导。
**Dissonants（不和谐）**：表面相似但低对齐度的参考片段，用于推导技能的负向适用边界。
**Divergents（偏离者）**：创意上富有启发的近失案例，用于探索不同突变程度的替代实现方案。
**Prompt Fidelity (PF)**：四维度评估之一，衡量生成视频对原用户意图（主体、动作、关系、事件语义）的忠实程度。
**Cinematic Quality (CQ)**：衡量视频运用镜头语言（构图、运镜、光影、调度、视听）的有效性和协调性。
**Narrative Appeal (NA)**：衡量视频组织事件为连贯叙事的能力，包括时间逻辑和情感吸引力。
**Creativity (CR)**：衡量视频在保留用户意图前提下，通过新颖、非显而易见的视听与叙事选择实现的表达野心。

## 可复现要素
- **数据集**：LSMDC（需邮件获取授权，已获作者批准）、Movie101（HuggingFace 官方申请）、VidProM（已公开）、StoryEval 和 VBench（已公开）。
- **代码/权重**：论文声明将发布生成的 artifacts 和详细实验配置，但未明确提及代码开源；使用 MiniMax-H3 和 LTX-2.5 开源模型生成视频。
- **关键超参**：视频生成 10秒、24FPS、1344×768、50 inference steps；参考检索阈值为 cosine similarity ≥ 0.35；开发集 400 提示（每技能 20 个）；评估温度 0；路由温度 0.1。

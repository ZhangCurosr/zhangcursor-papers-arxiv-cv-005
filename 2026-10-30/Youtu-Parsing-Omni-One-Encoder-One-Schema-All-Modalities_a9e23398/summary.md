---
title: "Youtu-Parsing-Omni-One-Encoder-One-Schema-All-Modalities"
source: https://arxiv.org/pdf/2610.39301v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-11-01 07:02:13"
field: "多模态理解与文档解析"
keywords: ["omni-modal parsing", "unified encoder", "on-policy distillation", "temporal alignment", "structured output", "multimodal perception"]
innovations: ["基于文本LM初始化的统一跨模态编码器+分块轴因式化RoPE实现感知级音视频融合", "OmniSchema统一十任务输出格式并支持OSAD自演化token-role区分蒸馏", "Schema-routed RLVR结合通用格式惩罚与任务特定content reward"]
benchmarks: ["OmniDocBench v1.6", "OmniParsingBench", "PureDocBench", "ChemOCR", "PDMX-Synth", "Video-MME-v2"]
---

# 论文速读：Youtu-Parsing-Omni: One Encoder, One Schema, All Modalities

## 一句话总结
提出了 Youtu-Parsing-Omni，一个全模态解析器，通过"一个编码器处理所有模态、一个 Schema 统一所有任务"的设计，将文档、图像、音频、视频转换为时空锚定的结构化记录，以 5B 参数在 OmniDocBench v1.6 和 OmniParsingBench 上均达到 SOTA。

## 研究问题与动机
- **多模态解析覆盖不全**：现有文档解析器对化学结构、五线谱等符号内容仍需专用模型；全模态模型多用于 captioning/问答，缺乏接地结构化输出能力。
- **跨模态感知割裂**：现有全模态 parser 采用双塔（独立视觉+音频编码器），感知阶段无跨模态融合，且音视频时间索引独立，时序对齐被推迟到语言模型层处理。
- **跨任务联合优化困难**：十类任务输出字段各异，难以用单一标量 reward 进行 RLVR；SFT 仅在参考前缀上训练，与推理时的自生成前缀存在分布偏移。
- **专用化与通用化的权衡**：扩展模态往往牺牲单模态精度，需证明单一模型可在保持文档解析深度的同时覆盖音频/视频。

## 核心贡献（创新点）
1. **统一感知编码器 Youtu-Omni-Encoder**：用单个从文本 LM 初始化的 Transformer 替代双塔，视觉/音频 token 共享时空 RoPE，少数融合层在感知内完成短时音视频交互。
2. **统一输出模式 OmniSchema**：将十个解析任务（文档/化学/乐谱/图表/流程图/几何图/自然图像/音频/自然视频/文本丰富视频）统一为单一 JSON 结构，每个单元由显式空间或时间锚点定位。
3. **自演化后训练 OSAD**：提出 token-role-based 目标，内容 token 用 JS 散度、结构 token 用 KL 散度；教师模型以自身参考条件版本提供逐 token 目标，每轮将学生适配器合并回基础权重实现自我进化，无需外部教师或手工设计 reward。
4. **分阶段课程训练**：四阶段预训练（感知对齐→全模态预训练→退火→长上下文扩展）配合三阶段后训练（SFT→RLVR→OSAD），逐步扩展可训练参数、序列长度、融合层数和扰动策略。

## 方法详解
- **编码器架构**：模态专属 stem（视觉：线性映射 16×16 patch；音频：3 层 stride-2 卷积+线性投影，13 tokens/s）→ 共享 Transformer（28 层，GQA，双向注意力，RM-SNorm，SwiGLU）→ token merger（相邻 token 沿特征轴拼接降采样）→ 两层 MLP projector 映射到解码器宽度。
- **分块轴因式化 RoPE**：将 $d_{rot}=64$ 个旋转槽分为 T（21）、H（21）、W（22）三个轴段，每轴独立逆频率表，保证每轴覆盖完整频率范围；视觉 token 取帧索引 + 块坐标，音频 token 取时间偏移并设 $h=w=0$，实现跨模态共用时间轴。
- **融合路由**：视频经融合路径（概率 $p_{fus}$，训练中从 Stage 2 起设为 0.8）或独立路径；融合路径按 $(I^+ A^*)$ 分组形成注意力窗口，帧与音频在同时间尺度 $\Delta t_{audio}=1/\text{token}, \Delta t_{vision}=r_a/\text{fps}$ 上对齐，稀疏融合通过 per-layer 选择不同的 segmentation vector 实现。
- **OmniSchema 结构**：每个输出含 modality、elements（图像）/segments（音视频）、global_description；element 以 bbox 锚定并分类（text/table/formula/chemical/score 等），segment 以 HH:MM:SS 时间戳锚定，含 OCR/ASR/事件标签/ speaker 等字段。
- **OSAD 目标函数**：
  - 内容 token 位置：$d_t^C = \text{JS}(q_t^{(\tau_c)} \| p_t^{(\tau_c)})$（有界，允许多种合法续写）
  - 结构 token 位置：$d_t^S = \text{KL}(q_t^{(\tau_s)} \| p_t^{(\tau_s)})$（$\tau_s > \tau_c$，鼓励覆盖教师结构预测）
  - 总损失：$\mathcal{L}_{row} = (\bar{d}^{\text{all}} + \alpha_c \bar{d}^C + \alpha_s \bar{d}^S)/(1+\alpha_c+\alpha_s)$
  - 每轮学生 LoRA 训练，轮边界合并适配器至基础权重作为下一轮教师（stopgrad），共 6 轮；$\alpha_c$ 从 0.50 逐步增至 0.80，强调内容监督。
- **RLVR Reward**：$R = \mathbb{1}[\text{decodes}] \cdot \mathbb{1}[\text{has payload}] \cdot \text{clip}_{[0,1]}(\Omega \cdot \text{core}_\nu)$，其中 $\Omega$ 结合格式分数与重复/覆盖不足惩罚，$\text{core}_\nu$ 为任务特定的 grounding + 描述相似度。

## 实验与结果
- **OmniDocBench v1.6**：96.96 Overall，高于 TeleOCR（96.91）0.05，是唯二超过 96 且同时支持音/视频的系统；比 Youtu-Parsing 提升集中在 tables（+4.8 TEDS）和 formulas（+3.2 CDM）。
- **OmniParsingBench**：75.08 Avg.，领先开源模型 Logics-Parsing-Omni（72.26）2.82；在音频（+2.5）、自然视频（+6.8）、文本丰富视频（+10.3）上优势显著；图表解析 95.02 为所有系统最高。
- **PureDocBench**：Avg3 76.29，三类 track（clean/degraded/recaptured）均领先 end-to-end 专家模型约 8 分。
- **ChemOCR**：Tani@1.0 = 54.2，超越 DeepSeek-OCR 2（49.6）。
- **PDMX-Synth**：CER = 22.73，优于专用 LEGATO（23.30），远超其他通用系统（45–94）。
- **Video-MME-v2**：Group score 11.5，L1=20.0 远超非 thinking 基线（≥5 分优势），L2/L3 与 Qwen3-Omni-30B-A3B-Think 差距约 2 分。
- **消融**：文本初始化编码器在 300B tokens 处 OCRBench 领先随机初始化 77.2 分；OSAD 贡献 SFT 后 1.99 分提升中的 1.75 分（88%）。

## 相关工作脉络
- **Document parsers**（TeleOCR、MinerU2.5-Pro、PaddleOCR-VL）：专注于文档模态，未覆盖音视频；本文将其扩展至全模态且保持文档解析精度。
- **Omni-modal models**（Qwen2.5-Omni、Qwen3-Omni、VITA-1.5）：多为 captioning/QA 导向的双塔架构；本文聚焦 grounded structured output，并在感知阶段即实现跨模态融合。
- **Logics-Parsing-Omni**：首个全模态解析框架并引入 OmniParsingBench，但采用双塔（0.5B ViT + 0.6B AuT + 30B MoE decoder）；本文以单编码器+5B dense decoder 实现更高时空解析性能。
- **Shared encoder work**（VATT、Omnivore、TVLT、OmniEncoder）：探索跨模态共享 Transformer，但未针对 grounding parsing 设计统一位置编码和跨模态融合窗口。
- **Text-LM initialization**（TrOCR、DeepSeek-OCR 2、Mono-InternVL）：将预训练 LM 权重用于视觉任务；本文首次将其用于跨视觉/音频的统一感知 backbone，并证明对 OCR/parsing 的关键作用。
- **On-policy distillation**（GKD、OPSD、MiniLLM）：在 learner 生成序列上蒸馏；本文扩展至 schema-aware 逐 token 角色区分监督并引入自演化教师刷新机制。

## 局限性与未来方向
- **推理加速缺失**：相比 predecessor Youtu-Parsing 的并行解码和 token/query 级加速，Youtu-Parsing-Omni 采用端到端自回归解码，吞吐较低，尤其是密集结构化内容（如表格）；作者计划引入并行解码技术。
- **更高阶时序推理受限**：Video-MME-v2 的 L2/L3（时序理解与复杂推理）成绩与 thinking 模式大模型仍有约 2 分差距，说明跨帧聚合后的多步推理仍需增强。
- **静态模态（自然图像/几何）略逊**：相比专注静态场景的双塔系统，在 natural image 和 geometry 上存在小幅劣势，可能源于参数预算分配向时序模态倾斜。

## 研究启发与可借鉴点
1. **文本 LM 初始化跨模态编码器**：从 Qwen3-0.6B-Base 初始化共享 Transformer 可大幅提升 OCR 和解析样本效率，为多模态感知层初始化提供了低成本高回报的范式。
2. **分块轴因式化 RoPE**：将旋转槽按语义轴（时间/高度/宽度）分段并独立归一化，使单编码器同时表达 2D 图像几何和 1D 音频时间且保持跨模态时间对齐，可直接迁移至其他需时空联合建模的任务。
3. **OSAD 自演化 distillation 框架**：以自身参考条件版本为教师、按 token 角色区分 JS/KL 损失、定期刷新教师权重的设计，无需额外 reward designer 即可实现结构化输出的逐 token 精细优化。
4. **Schema-routed RLVR**：将通用 well-formedness 惩罚与任务特定 content score 分离组合，为异构多任务 RL 提供了一个可扩展的 reward 设计模板。
5. **融合/独立双路由训练**：训练时以概率 $p_{fus}<1$ 混合融合路径与独立路径，使推理时两者均可用，有效提升了音视频时序对齐的鲁棒性。

## 关键术语表
- **Youtu-Omni-Encoder**：从预训练文本 LM 初始化的单一共享 Transformer 感知编码器，以极小比例模态专属 stem/projector 处理视觉和音频输入。
- **OmniSchema**：统一输出 JSON 模式，将十个跨模态解析任务的输出标准化为含时空锚点单元的结构化记录。
- **OSAD（OmniSchema-Aware On-Policy Distillation）**：自演化 on-policy 蒸馏方法，按 token 角色（内容/结构）分配 JS/KL 损失，并以参考条件自蒸馏实现逐轮教师刷新。
- **Chunked axis-factorized RoPE**：将 rotary slots 按语义轴分段并为每轴独立逆频率表的位置编码方案，使视觉和音频共享时间轴同时保留空间轴。
- **Schema-routed RLVR**：基于 GRPO 的训练阶段，reward 由通用格式因子 Ω 与任务特定 content factor $\text{core}_\nu$ 相乘构成。
- **Tani@1.0**：ChemOCR 上的精确指纹匹配率，预测 SMILES 与参考 SMILES 的 Tanimoto 相似度等于 1.0 的比例。
- **Perception vs. Cognition**：OmniParsingBench 双组件评估——Perception 测量接地字段（定位/OCR/ASR）准确度，Cognition 通过 LLM judge 基于纯 parse 文本回答多选题。
- **Fused/Independent route**：视频编码的双路由机制，融合路径在少数层内让帧与同时音频共享注意力窗口，独立路径保持各模态各自窗口。

## 可复现要素
- **数据集**：OmniDocBench v1.6、PureDocBench、OmniParsingBench、Video-MME-v2、ChemOCR、PDMX-Synth；合成数据包括从 PubChem/ChEMBL 渲染的 1M 分子图、PDMX 乐谱、程序化几何图；音频使用内部 audiobooks/podcasts/conversations。论文未明确说明自有训练数据的公开状态。
- **代码**：已开源，https://github.com/TencentCloudADP/youtu-parsing
- **权重**：已开源，https://huggingface.co/tencent/Youtu-Parsing-Omni
- **关键超参**：编码器 28 层、hidden 1024、$d_{rot}=64$、轴分配 (21,21,22)；预训练 2.24T tokens；SFT 16B tokens；RLVR 0.6B tokens；OSAD 0.6B tokens，6 轮；LoRA 训练；序列长度最大 128K；视频融合层为 13、14、27、28（共 28 层）。

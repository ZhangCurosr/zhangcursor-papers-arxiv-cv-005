---
title: "Youtu-Parsing-Omni-One-Encoder-One-Schema-All-Modalities"
source: https://arxiv.org/pdf/2610.39301v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-11-01 07:02:16"
field: "多模态文档与结构化解析"
keywords: ["omni-modal parsing", "unified encoder", "on-policy distillation", "structured output", "audio-visual fusion", "document parsing", "schema-aware training"]
innovations: ["文本LM初始化的统一跨模态编码器替代双塔架构", "OmniSchema统一十任务输出模式配合token-role分层蒸馏", "自演进OSAD在线策略蒸馏实现无外部教师信号迭代优化"]
benchmarks: ["OmniDocBench v1.6", "OmniParsingBench", "PureDocBench", "Video-MME-v2", "ChemOCR", "PDMX-Synth"]
---

# 论文速读：Youtu-Parsing-Omni: One Encoder, One Schema, All Modalities

## 一句话总结
本文提出 Youtu-Parsing-Omni，一个基于"一编码器、一模式"的跨模态解析模型，用共享的 Youtu-Omni-Encoder 和统一的 OmniSchema 将文档、图像、音频、视频统一解析为带时空锚点的结构化记录，5B 参数单 checkpoint 在 OmniDocBench v1.6（96.96）和 OmniParsingBench 开源模型（75.08）上均取得最优成绩。

## 研究问题与动机
- **跨模态联合感知不足**：现有 omni-modal 解析器（如 Logics-Parsing-Omni）采用分离预训练的视觉/音频双塔，感知阶段无跨模态融合，时间轴独立，帧与对应音频的时间对齐只能依赖解码器事后推断。
- **多任务联合优化困难**：十项解析任务输出字段异质，序列级奖励（如 GRPO）难以定义跨任务标量 reward；SFT 仅在参考前缀上训练，与推理时模型自生成前缀存在分布偏移。
- **模态覆盖碎片化**：文档解析器强于页面排版与公式/表格识别，但化学结构、五线谱等符号内容仍依赖领域专用模型；图像/音频/视频的解析成熟度远低于文档。
- **单模型兼顾广度与深度存疑**：缺乏证据证明一个统一模型能在覆盖全模态的同时，保持对专用文档解析器同等精度的结构解析能力。

## 核心贡献（创新点）
1. **Youtu-Omni-Encoder（文本初始化统一编码器）**：用预训练文本 LM 初始化共享 Transformer，替代双塔架构；视听 token 共享同一时空旋转位置编码，在融合层内完成短时窗跨模态交互，而非依赖解码器后验对齐。与双塔相比，感知参数大幅减少且具备跨模态感知能力。
2. **OmniSchema（统一十任务输出模式）**：将文档、自然图像、音频、视频的十项解析任务统一为单一 JSON 模式，每个元素/片段均有显式空间或时间锚点，使不同任务的监督目标可被同一 token-role 体系捕获。
3. **OSAD（OmniSchema-Aware On-Policy Distillation，自演进在线策略蒸馏）**：对 SFT+RLVR 后的模型，用其自身 rollout 作为蒸馏源，teacher 额外接收参考解析作为 privileged information；按 token 角色区分内容（JS 散度，温度 τ_c）与结构（前向 KL，温度 τ_s）的优化目标；每轮结束后 teacher 刷新为学生新 checkpoint，实现"解析器与监督共同进化"。
4. **四阶段预训练 + 三阶段后训练的渐进课程**：Stage 0–3 依次扩展可训练参数、输入尺度、跨模态融合层数、序列长度（最高 128k tokens、512 帧）；post-training 依次为 SFT（建立格式）、RLVR（序列级优化）、OSAD（token 级自演进），形成完整训练闭环。
5. **单一 5B 模型在文档/图像/音频/视频全谱系基准上达到 SOTA 或最强开源成绩**：证明"One encoder, one schema"既能覆盖广，又能保精度（如在化学结构 Tani@1.0=54.2 领先 DeepSeek-OCR 2，在乐谱 CER=22.73 匹敌专用 LEGATO）。

## 方法详解

### 2.1 Youtu-Omni-Encoder
- **Stems**：视觉 stem 将原分辨率图片切为 16×16 patch，经单一线性映射为 d 维 token，无卷积/绝对位置表/class token；音频 stem 对 log-mel 谱图（100 mel/帧，13 tokens/s）做 3 层 stride-2 卷积下采样后再线性投影。两 stem 合计仅占编码器参数的 <3%。
- **Shared Transformer**：28 层 pre-norm Transformer（RMSNorm + SwiGLU FFN + GQA + QK-Norm），**双向注意力**。Rotary 位置编码从 per-token 旋转角矩阵读取，支持轴分解（见 2.2）。
- **Token Merger & Projector**：视觉 token 按 2×2 区块沿特征轴拼接（4→1），音频相邻 token 按时间合并；随后经 RMSNorm + 2 层 MLP 投影至解码器宽度 d_lm=2560。

### 2.2 统一位置编码与音视频融合
- **轴分解 Chunked RoPE**：将 d_rot=64 个 rotary slot 划分为 (d_T=21, d_H=21, d_W=22) 三段，每段独立归一化 inverse-frequency table，保证每轴覆盖完整频率范围。视觉 token 取 (T, H, W)=(frame_idx, h, w)，音频 token 取 T=time_offset, H=W=0，两者共享同一时间轴。
- **Fused vs. Independent 双路由**：视频经独立路由（每帧/音频块各自 attention window）或融合路由（按 (I⁺A*) 分组，在 4 个 fusion 层内让帧与同期音频 cross-attend）。训练时融合概率 p_fus=0.8，保证推理时两路由均可用。
- **时空对齐公式**：Δt_audio = 1 per token，Δt_vision = r_a / fps per frame，使视听在同一时间尺度推进。

### 3 OmniSchema
- 输出为 JSON，必需字段：modality、elements（图像）或 segments（音频/视频）、global_description。
- Element 含 category（text/table/formula/chemical/score/chart/flowchart/geometric/figure/seal 等）、bbox（0–1000 归一化）、content（按类别填 text/latex/otsl/markdown/mermaid/geometric/ABC/SMILES 等）。
- Segment 含 start/end_timestamp（HH:MM:SS）、kind（vocal/nonvocal）、speaker_id、audio（asr_text、timbre_caption 等）和 visual（elements、atomic_actions、camera_motion 等）。
- 十项任务：文档解析、化学结构识别、乐谱识别、图表解析、流程图解析、几何图解析、自然图像解析、音频解析、自然视频解析、图文视频解析。

### 4 训练流程
- **Stage 0（Perception Alignment，320B tokens）**：仅训练 encoder，decoder 冻结；限制小输入（≤256 tokens/image，≤16 帧），0 个 fusion layer。
- **Stage 1（Omni-modal Pre-training，1.28T tokens）**：全部参数训练；序列 8k–32k，4 fusion layers，混入文本数据防灾难性遗忘。
- **Stage 2（Annealing，512B tokens）**：cosine LR 从 1e-4 衰减至 1e-5；引入 layer-wise LR decay（γ=0.9）保护 stems/lower layers；p_fus=0.8 随机路由。
- **Stage 3（Long-Context Extension，128B tokens）**：序列最高 128k，图像/视频单输入最高 102400 tokens，最多 512 帧。
- **SFT（16B tokens）**：统一 OmniSchema 序列化与指令遵循；混合 parsing data + 通用 VLM 数据。
- **RLVR（0.6B tokens）**：GRPO 优化，reward = 1[可解码]·1[有 payload]·clip(Ω·core_ν)，其中 Ω 含格式分 Fmt、重复惩罚 0.6·P_repeat、覆盖不足惩罚 0.7·P_under；core_ν 为任务特定 grounded-field 精度与描述相似度均值。
- **OSAD（0.6B tokens，6 rounds）**：学生 LoRA 更新、teacher 冻结并接收 [y* ; x]；角色损失 L_row = (d_all + α_c·d^C + α_s·d^S)/(1+α_c+α_s)，其中 d^C 用 JS 散度（τ_c），d^S 用前向 KL（τ_s>τ_c）；每轮结束 θ_{r+1} ← stopgrad(θ_r)，teacher 刷新。权重随轮次递增：α_c 从 0.50→0.80，α_s 从 0.10→0.08。

## 实验与结果
- **OmniDocBench v1.6**：Youtu-Parsing-Omni 获 **96.96 Overall**，排名第一（TeleOCR 96.91 为次优）；文本 Edit=0.0271，Formula CDM=96.80，Table TEDS=96.79，Reading Order Edit=0.1104（最优）。与专家长度文档解析器 TeleOCR（1.2B）差距 <0.21 分。
- **PureDocBench**：End-to-end 模型中 Avg₃=76.29 最高，clean/degraded/recaptured 三轨均优于所有 end-to-end 专用模型；相对前作 Youtu-Parsing 每轨提升约 8 分。
- **OmniParsingBench**：开源最强 **Avg=75.08**，超 Logics-Parsing-Omni（72.26）+2.82；图表解析 95.02 最高（超 Gemini-3-Pro 的 92.79）；在音频（+2.5）、自然视频（+6.8）、图文视频（+10.3）上优势集中；静态图像/图形略落后 Gemini-3-Pro。
- **ChemOCR**：Tani@1.0 = **54.2**（领先 DeepSeek-OCR 2 的 49.6）；Avg. Sim. = 74.6（略低于 DeepSeek-OCR 2 的 74.9，但精确匹配更关键）。
- **PDMX-Synth**：CER = **22.73**，匹敌专用 LEGATO（23.30），远超其他通用系统（45–94 CER）。
- **Video-MME-v2**：Group score = 11.5，L1=20.0 大幅领先所有非 thinking 基线；L2/L3 与非 thinking 模型相当，与 Qwen3-Omni-30B-A3B-Think 差距约 1.8–1.9 分。
- **消融**：
  - 文本初始化统一编码器 vs. 随机初始化：OCRBench 领先 130.6 分（767.0 vs. 636.4），Avg-6 差距从 1.3 分缩至 0.4 分。
  - 随机初始化统一编码器 vs. 双塔（SigLIP-2+SenseVoice-small）：perception 参数少 2/3，Avg-6 差距 1.3 分，但 OCRBench 差距达 53.4 分；文本初始化后差距几乎消除。
  - OSAD 贡献：SFT→最终模型在 OmniDocBench v1.6 提升 1.99 分，其中 OSAD 贡献 1.75 分（88%），RLVR 仅 0.24 分。

## 相关工作脉络
1. **Logics-Parsing-Omni**：首个 omni-modal parsing benchmark 提出者；采用双塔（ViT+Audio Tower）+ MoE 解码器（30B-A3B）。本文用单编码器替代双塔，用 5B 密集模型在时序任务上反超。
2. **MinerU2.5-Pro / PaddleOCR-VL-1.6**：文档解析专用端到端模型（分别 1.2B/0.9B），专注页面解析；本文扩展至音视频并统一十任务，性能仍匹敌文档专用器。
3. **Qwen2.5-Omni / Qwen3-Omni**：多模态理解与生成模型，采用 TMRoPE 对齐音视频；本文更关注结构化解析输出与 grounding，且编码器从文本 LM 初始化。
4. **VATT / TVLT / ImageBind**：共享 Transformer 或 embedding space 的多模态表示工作；本文面向 grounded parsing，强调空间-时间联合编码与融合层内的跨模态交互。
5. **DeepSeek-OCR 2**：视觉 encoder 从 Qwen2 初始化，使用双向视觉注意力；本文将其理念推广至视觉+音频共享 encoder，并加入统一位置编码与音频视觉融合。
6. **OPSD / GKD / MiniLLM**：on-policy distillation 相关工作；本文 OSAD 创新性地在 token-role 级别分别处理内容与结构（JS vs. KL），并引入参考 conditioned 的 self-distillation 与 teacher refresh。

## 局限性与未来方向
- **推理效率不足**：未实现 token/query 级并行解码和 shared encoder feature 复用，对密集结构化内容（如表格）吞吐较低；计划未来引入并行解码与加速技术。
- **静态图像/几何解析仍有提升空间**：相对 Gemini-3-Pro，在自然图像（−4.7）和几何（−2.0）上落后；几何关系与颜色/位置语义描述偏弱。
- **ASR 精度低于专业基线**：音频和视频 ASR 分数为各分组最低，反映模型更擅长时空定位与事件描述，而非纯语音转写。
- **高层视频推理受限于非 thinking 模式**：Video-MME-v2 的 L2/L3 得分与思考型大模型差距主要来自多步推理，感知层已充分训练。
- **长视频融合路由计算开销**：Stage 3 支持 512 帧，但融合层跨模态 attention 的计算复杂度随帧数增长，可能限制实际部署。

## 研究启发与可借鉴点
1. **用文本 LM 初始化多模态感知编码器**是极低成本高效的手段：相比从零训练，文本初始化在 OCR 和通用视觉任务上带来数量级的样本效率提升，同时保留 ASR 潜力。对于团队在多模态编码器初始化策略上有可直接复用的思路。
2. **Token-role 分层蒸馏（内容 JS + 结构 KL）**将结构化输出的监督细粒度到 token 级别，解决多任务异质输出的联合优化难题；这对任何需要输出结构化 JSON/XML 的多任务模型均有借鉴价值。
3. **参考条件自蒸馏 + Teacher 周期性刷新**构成无外部教师信号的自演进训练闭环：仅依赖自身 rollout 和参考标注即可持续迭代，避免 RLVR reward 设计的人工成本。
4. **轴分解 Chunked RoPE 实现跨模态统一时空编码**：将 2D 空间（H,W）和 1D 时间（T）分配到独立 rotary 段，且音频与视频帧共享时间轴，为后续音频-视频对齐任务提供了一套可复用的位置编码范式。
5. **SFT → RLVR → OSAD 三阶段后训练课程**：先建格式（SFT），再序列优化（RLVR），最后 token 级精细化（OSAD），层层递进，避免一开始就引入 on-policy 信号导致的格式崩溃。

## 关键术语表
- **Youtu-Omni-Encoder**：从文本 LM 初始化的共享 Transformer 感知编码器，同时处理图像、音频、视频，含模态专属 stem 和 projector。
- **OmniSchema**：统一十项解析任务的 JSON 输出模式，每个元素/片段均带空间 bbox 或时间戳锚点。
- **OSAD**：OmniSchema-Aware On-Policy Distillation，按 token 角色区分内容与结构损失的自演进在线策略蒸馏方法。
- **Chunked axis-factorized RoPE**：将 rotary slot 按语义轴（T/H/W）分段分配独立逆频率表的位置编码方案。
- **Fused vs. Independent route**：视频在编码器内两条处理路径，融合路由在 fusion layer 跨模态 attention，独立路由保持单模态窗口。
- **Schema-routed RLVR**：按任务路由 reward 函数的 GRPO 优化，Ω 衡量格式与覆盖率，core_ν 衡量内容精度。
- **Tani@1.0**：ChemOCR 主指标，预测 SMILES 与参考分子指纹 Tanimoto 相似度等于 1.0 的样本比例。
- **Parse rate**：OmniParsingBench 评估中，修复 schema 违规后仍可解析的输出占比，用于惩罚格式错误。

## 可复现要素
- **数据集**：OmniDocBench v1.6、PureDocBench、OmniParsingBench、Video-MME-v2、ChemOCR、PDMX-Synth；训练数据含自建文档数据（含合成页面、PubChem/ChEMBL 分子、PDMX 乐谱）、自然图像、音频（自制有声书/播客/对话）、视频；通用数据来自 Youtu-LLM/Youtu-VL/Long-VITA/VITA-1.5/VITA-Audio。
- **代码**：公开 https://github.com/TencentCloudADP/youtu-parsing
- **模型权重**：公开 https://huggingface.co/tencent/Youtu-Parsing-Omni（5B 参数）
- **关键超参**：encoder 28 层/d=1024；decoder 40 层/d_lm=2560；d_rot=64，轴分割 (21,21,22)；融合层为 13/14/27/28 层；p_fus=0.8；OSAD 学习率 2e-7（1–2 轮）/1.5e-7（3–6 轮）；τ_s>τ_c；LoRA 冻结基础权重训练。

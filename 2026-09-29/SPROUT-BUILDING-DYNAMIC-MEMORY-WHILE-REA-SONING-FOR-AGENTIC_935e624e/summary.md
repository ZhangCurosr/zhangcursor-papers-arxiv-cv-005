---
title: "SPROUT-BUILDING-DYNAMIC-MEMORY-WHILE-REA-SONING-FOR-AGENTIC"
source: https://arxiv.org/pdf/2609.35497v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-10-01 03:46:59"
---

# 论文速读：SPROUT-BUILDING-DYNAMIC-MEMORY-WHILE-REA-SONING-FOR-AGENTIC

## 一句话总结
论文提出 Sprout 框架，让智能体在回答问题的同时逐步构建视频记忆（而非预先离线构建），通过低帧率观看→文本记录→高帧率重访的动态过程，实现按需生长且可跨问题复用的时序记忆树，显著降低上下文占用与每问题 token 成本，同时在长视频理解基准上达到或超越现有离线记忆方法。

## 研究问题与动机
- **离线构建成本高**：现有 build-then-reasoning 方法需在第一个问题到来前对整段视频做一次性记忆构建，若视频仅被少量问题查询，这种"提前全量记录"的成本远大于实际问答所需。
- **静态记忆无法累积**：已有记忆的查询阶段是只读的，模型在推理中获得的观察与结论无法沉淀到后续问题中，导致多问题场景下每次都要重新检索相同证据。
- **上下文窗口压力**：直接将视频逐帧送入模型会随时长线性膨胀视觉 token；稀疏采样又容易丢失问题依赖的关键瞬间。
- **实践痛点**：长视频常被多个问题共用，但现有方法要么构建成本偏高，要么不支持在线增长，导致在少问题/多问题两端都受限。

## 核心贡献（创新点）
1. **提出"边推理边构建"的在线记忆范式**：与以往"先构建后问答"的静态 pipeline 本质不同，Sprout 从空记忆出发，随问题到来逐步长出一个临时树，不付任何预构建代价。
2. **设计 watch-remember-revisit 三动作机制**：以低帧率流式观看、以文本节点写入记忆、以高帧率重访细节并回写子节点，形成"粗到细"的按需细化路径。
3. **帧释放策略（context pruning）**：一旦某视频片段被记录为文本即从上下文历史中移除，仅保留最新一轮的视觉输入，避免帧堆积，上下文峰值降至同类方法的约一半。
4. **跨问题记忆与问答历史共享**：时间树与 question–answer 记录在所有问题间持久化，后续问题可直接复用、细化或扩展已有记忆。
5. **实证"无需嵌入模型也可检索"**：采用字面匹配替代 embedding 检索，省去了额外编码开销，精度与嵌入检索相当（差异≤1 point）。

## 方法详解
- **动态时序记忆结构**：以视频时长 [0, T] 为根的时序树，每个节点覆盖区间 [s_i, e_i]，存储事件摘要、观察到的事实与关键实体；子节点按时间顺序划分父节点区间。只有被问题细看的分支才会加深，未观看区域保持树末端。
- **Watch（粗粒度探索）**：通过 `read_next()` 从 watch record 取下一个未观看块，以低帧率 f_c 附加帧（与可用 ASR 转录）；Agent 可在任一问题判断"已有足够证据"时停止观看。
- **Remember（记录并释放帧）**：每观看一块后立即调用 `write_chunk_memory` 写回约 4–8 个片段、每片段 5–10 条客观事实；随后该块的帧从上下文历史中移除，仅保留文本结果与模型响应，从而将上下文长度由 L_t^{retain}=B_t+∑v_j 降为 L_t^{release}=B_t+v_t。
- **Revisit（检索-稠密视察-回写）**：`search_memory(pattern, window)` 基于字面匹配与时段窗口返回事实与相关历史问答；当需细节时，Agent 选 f_r∈{1,2,4,8} fps 调用 `view_node` 或 `view_time` 重看对应区间，再分别通过 `refine_node`（拆分叶节点）或 `add_observation`（在叶节点下新增子节点）回写；被重看的帧同样在写回后释放。
- **跨问题累积**：每回答完一问，树与 watch record 保留；`answer` 调用同时把问题、选项与预测答案追加到 question history，下一问可复用全部已建记忆。
- **工具接口约束**：每次写回工具仅接受刚刚 view 的区间，保证树中时间顺序与父子关系一致。

## 实验与结果
- **数据集与基线**：LVBench（1,549 问/103 段小时级视频）、Video-MME Long（900 问/300 段 30–60 min 视频）、Video-Holmes（1,837 问/270 段短悬疑视频）、EgoLifeQA（500 问/44.3h 单主体 7 天记录）；基线含 Direct、WorldMM、MERIT、VideoSeek、Qwen-MM-Plugins 等。
- **主要精度结果（表1）**：三模型 GPT-5 / Qwen3.8-Max / Gemini 3.8 Flash 上，Sprout 均达或超离线记忆方法；Gemini 3.8 Flash 在 LVBench 88.2%、Video-MME Long 91.2%、Video-Holmes 76.8%，均为表内最高；在 EgoLifeQA 上 Qwen3.8-Max 达 80.6%、Gemini 3.8 Flash 达 80.2%，同样最优。
- **Cost 对比（表3，Gemini 3.8 Flash）**：Sprout 每问耗 token LVBench 71.7K / Video-MME Long 99.3K，分别比 Direct 低 74% 与 55%；Qwen-MM-Plugins 因离线构建摊销成本，在 Video-MME Long 总成本达 293.7K，高于 Direct。
- **帧释放有效性（Fig.3）**：不释放帧使上下文窗口最大膨胀至 4.3×，但精度不变；Sprout 峰值约 24.3K，不到"不释放"变体的 1/2，且全程 <55K。
- **问答历史价值（表4）**：以 Causal 方式让后续问题看到前序问答文本，Video-Holmes 提升 3.9 point，其余两基准变化 ≤1 point。

## 相关工作脉络
- **Agentic 多模态推理**（MM-REACT、ViperGPT、VideoAgent、DrVideo、VideoSeek、VideoSearcher）：均强调在推理中调用外部工具收集证据；Sprout 将其与持久记忆管理衔接，使每次 inspected 结果落盘为节点、可跨问题复用。
- **高效长视频理解**（LongVILA、Keye-VL-2.0、LongVU、VideoTree）：通过扩展上下文、压缩表示或自适应分配实现 coverage-detail 权衡；Sprout 的关注点是"观察密度与保留上下文随问答序列的动态演化"，用 coarse→fine 的按需细化替代全量喂入。
- **视频记忆与检索**（MA-LMM、M3-Agent、WorldMM、MemDreamer、Qwen-MM-Plugins、MERIT、HippoMM、LLoVi、Video-RAG、iRAG）：都属于 build-then-reasoning 范式，分 hierarchical/graph/flat 三类；Sprout 与之的本质区别在于记忆随问题在线生长并可被后续问题持续修正，而非一次性只读。
- **Literal vs embedding 检索**：与 Qwen-MM-Plugins 的 embedding 检索对比显示字面匹配精度相当（差异 ≤1 point），揭示强 agent 模型自身的 query 改写能力可弥补检索简单性，省去额外编码开销。

## 局限性与未来方向
- **自述局限**：当前记忆缺少多键索引、节点合并/遗忘机制，难以处理重复或矛盾观察；现有搜索仅依赖字面匹配与时段窗口，复杂语义检索能力有限。
- **检索强度受限于字面匹配**：虽省去了 embedding 成本，但对同义/ paraphrase 敏感场景可能召回不全，需依赖 agent 自行构造更精确 pattern。
- **未显式讨论极端噪声/弱帧场景**：低帧率 f_c=0.1 fps 下的初始粗记录可能遗漏关键瞬间，高度依赖 revisit 弥补；若 agent 判定失误未触发重访，则细节永久缺失。
- **Future direction（论文提出）**： richer tree retrieval（多键索引）、合并/遗忘节点机制、以及在更大规模/多模态音频-视觉任务上验证。

## 研究启发与可借鉴点
- **"观察即释放"的上下文压缩范式**：将视觉片段立即转写为文本节点并从 context history 中剔除，既能维持精度又可把峰值上下文压至 ~25K，适合任意多轮 agent 任务中缓解 token 压力。
- **粗-细两级记忆增长（coarse-to-fine temporal tree）**：用叶节点可拆分的时序树承载"摘要+细节"，便于按问题需求按需加深，避免一次性昂贵全局表征。
- **利用 agent 自身检索能力替代嵌入模型**：在强 LLM 场景下，字面匹配 + 多关键词 pattern 即可达到与 embedding 检索相近效果，可显著降低系统开销；值得在其他知识密集型 agent 任务中复验。
- **跨任务记忆复用与问答历史沉淀**：将 question history 作为独立可检索实体，即使前序答案有误也"不损害且常在共享内容场景有帮助"，为多轮对话/连续推理系统设计提供了证据复用的思路。
- **可与本团队方向结合的机会**：将 Sprout 的 watch-remember-revisit 框架迁移至多模态 Agent 的文档/代码/图谱理解；在需要超长上下文的视频-动作规划、机器人具身感知等任务中应用"观测→文本化→释放"的范式。

## 关键术语表
- **Sprout**：一种 agentic 视频理解框架，通过在推理过程中动态生长时序记忆树来按需构建视频记忆。
- **Build-while-reasoning**：与 build-then-reasoning 相对，指记忆不是预先离线构建，而是随问题解答同步增删与细化。
- **Temporal tree（时序树）**：以视频时间为轴的树形记忆结构，根覆盖全视频，节点存区间与事件摘要，子节点按时间顺序展开细节。
- **Watch / Remember / Revisit**：Sprout 的三核心动作——低帧率观看新段、将观察写为文本节点、对关键区间高帧率重访并回写细化。
- **Frame release（帧释放）**：在片段被记录为文本后立即从其上下文历史中移除，仅保留最新一轮的视觉输入以降低 token 占用。
- **Literal matching（字面匹配检索）**：基于关键词大小写不敏感子串匹配的事实检索方式，区别于 embedding 相似度检索。
- **Question history（问答历史）**：同一视频所有前序问题、选项与模型预测答案的紧凑记录，供后续问题检索复用。
- **Context window（上下文窗口）**：单问题在多轮推理中的最大上下文 token 峰值，本文用它度量方法的显存/计费压力。

## 可复现要素
- **数据集**：LVBench、Video-MME Long、Video-Holmes、EgoLifeQA（Jake 500 问/44.3h）；论文未声明公开与否（基准本身均已公开发布）。
- **代码/权重**：代码已开源 https://github.com/HKUST-LongGroup/Sprout；模型使用 GPT-5、Qwen3.8-Max、Gemini 3.8 Flash 官方接口/模型卡，权重非本地训练因此无训练权重发布。
- **关键超参**：f_c 在 LVBench/Video-MME 上为 0.1 fps（块 10 min，GPT-5 为 5 min）；Video-Holmes 上 f_c=1 fps（块 30 s）；f_r∈{1,2,4,8} fps；单轮最多附加帧数 Qwen3.8-Max 768 / GPT-5 48 / Gemini 2,048；每问最多 32 轮；单 chunk 写入 4–8 个 segment、每 segment 5–10 条事实；search_memory 最多返回 40 条事实 + 5 条历史问答。
- **ASR**：Video-MME Long 附加转录；LVBench 与 Video-Holmes 未使用。
- **Prompt**：系统提示见 Appendix C（Prompt 1/2），工具 schema 见 Table 5。

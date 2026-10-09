---
title: "VAMR-MULTI-QUESTION-AGENTIC-REASONING-FOR-EFFICIENT-LONG-FOR"
source: https://arxiv.org/pdf/2610.11171v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-10-09 09:48:09"
field: "长视频多模态理解与 Agent 推理"
keywords: ["long-form video understanding", "multi-question reasoning", "agentic video agent", "policy optimization", "shared trajectory", "question-conditioned perception"]
innovations: ["提出共享回路多问题视频推理框架 VAMR，首次在同一轨迹中协调多个问题的工具调用与答案提交", "设计 QHPO 算法，通过问题级 Critic 和回合对齐实现共享轨迹中异步终止问题的信用分配"]
benchmarks: ["LVBench", "Video-Holmes", "LongVideoBench"]
---

# 论文速读：VAMR-MULTI-QUESTION-AGENTIC-REASONING-FOR-EFFICIENT-LONG-FOR

## 一句话总结
VAMR 提出了一种共享回路的视频 Agent 框架，将同一视频的多个问题置于同一个 ReAct 推理循环中协调执行，配合问题条件感知（QA-inspector）和分层记忆结构，并通过 QHPO（Question-Horizon Policy Optimization）算法以问题级优势估计完成策略优化；在 LVBench、Video-Holmes 和 LongVideoBench 三个长视频基准上实现了最高准确率与最少推理轮次。

## 研究问题与动机
1. **现有视频 Agent 以单个问题为单位运行独立推理轨迹**，对同一视频的多问题场景会导致重复的视频探索与记忆重建，造成大量冗余计算。
2. **跨问题证据无法协同获取**：不同问题可能指向视频的同一时段或不同侧面，独立循环无法让各问题联合指导取证，也无法渐进构建共享的视频理解。
3. **多问题效率与准确性难以兼顾**：现有 end-to-end 方法受限于固定视觉上下文，迭代方法虽可自适应检索但仍围绕单查询组织，无法利用同视频内的问题密度来压缩交互轮次。
4. **缺少面向共享多问题轨迹的强化学习优化方法**：已有策略优化（PPO/GRPO 等）将一条轨迹关联到单一任务结果，无法处理"问题进度不同、终止时间各异"的共享轨迹中的信用分配问题。

## 核心贡献（创新点）
1. **共享回路多问题推理框架（VAMR）**：首次将多个问题的视频推理统一到一个持久化的 ReAct 循环中，每个回合输出联合动作（可同时提交答案+发起多个工具调用），而不再是独立的 per-question 轨迹。与已有工作的本质区别：现有 agentic 方法（如 VideoARM、DVD）均以单问题为中心组织状态与工具调用，VAMR 将跨问题协调内化为轨迹本身的决策结构。
2. **问题条件感知工具（QA-inspector）**：在一次工具调用中同时服务多个问题，返回共享段落摘要和每个问题的专属线索证据，使感知阶段即可实现跨问题复用。与已有工作的本质区别：通用 visual inspector 只做片段描述，不针对问题-选项对进行判别性检索。
3. **分层多问题记忆结构（Layered Memory）**：维护共享故事层（`G_t`，可持续复用的时间锚定视频叙事）和独立的问题线索层（`E_t^i`，每个活跃问题的专属证据），并在答案提交后完成清理与回溯。与已有工作的本质区别：单层记忆会混入无关的问题特定证据，分层设计保证新激活问题继承视频级知识但不继承无关线索。
4. **QHPO 策略优化算法**：引入问题级 Critic 估计每个活跃问题的成功概率，通过 GAE 得到问题级优势后仅在与该问题直接相关的回合（validated round alignment）保留优势并聚合为回合级更新信号。与已有工作的本质区别：标准 PPO 对共享轨迹只赋予一个回合级价值，无法区分同一状态下不同问题的学习信号；QHPO 在共享轨迹中实现问题粒度的信用分配。

## 方法详解
**共享回路建模（Shared Loop）**：设视频 $V$ 含 $n$ 个问题 $Q = \{q_i\}$。每一轮 $t$，策略模型在活跃窗口 $B_t$（最多 5 个未解问题）和分层记忆 $M_t$ 上输出联合动作 $\mathbf{a}_t$，包含若干答案提交和工具调用；已回答问题移出窗口并从后备队列（backlog）补充新问题。轨迹持续直到所有问题解答完毕。

**QA-inspector**：给定时间段 $V_{[l,r]}$ 和目标问题集合 $Q_t^{\text{tar}}$，工具输出共享摘要 $g_t$ 和各问题线索 $\{e_t^i\}$：
$$(g_t, \{e_t^i\}) = \text{QA-inspector}(V_{[l,r]}, \{(q_i, A_i)\}_{q_i \in Q_t^{\text{tar}}})$$
视觉上采样 30/60/90/150 帧，细粒度 inspect 最多 30 秒片段；ASR 工具最多处理 90 秒片段且不含问题标识。

**分层记忆**：$M_t = (G_t, \{E_t^i\}_{q_i \in B_t}, T_t)$，其中 $G_t$ 为共享故事层，$E_t^i$ 为问题 $q_i$ 的线索层，$T_t$ 为暂存缓冲。答案被接受后，缓冲内容整合入 $G_t$，问题线索路由到对应 $E_t^i$，已完成问题的线索层丢弃，缓冲清空。

**两阶段 SFT 初始化**：
- Stage 1：使用标注答案 $P_t$ 和证据区间构造联合动作 $\mathbf{a}_t^\star$（特权信息仅用于指导动作选择，不进入模型输入）。
- Stage 2：移除特权信息，由教师模型（Qwen3-VL-32B）生成推理链 $z_t^\star$，训练目标为 $(B_t, M_t) \to (z_t^\star, \mathbf{a}_t^\star)$ 的自回归概率。

**QHPO 训练**：
- 问题级 Critic 损失（二分类交叉熵）：$\mathcal{L}_{\text{critic}} = -\mathbb{E}[R_i \log V_\phi + (1-R_i)\log(1-V_\phi)]$，$V_\phi(B_t, M_t, q_i)$ 估计问题 $q_i$ 的成功概率。
- 问题级 GAE：对每个问题 $q_i$ 计算 $\delta_{t,i} = r_{t,i} + \gamma V_{\phi,t+1} - V_{\phi,t}$，$A_{t,i} = \sum_{k=0}^{e_i-t}(\gamma\lambda)^k \delta_{t+k,i}$，其中 $r_{e_i,i} = R_i$，其余回合 $r_{t,i}=0$。
- **Round Alignment（回合对齐）**：仅当回合 $t$ 的联合动作直接服务于 $q_i$（如针对 $q_i$ 的视觉调用或其答案提交）时保留 $A_{t,i}$，否则置零。
- 回合级优势聚合：$A_t = \frac{1}{|S_t|}\sum_{i \in S_t} A_{t,i}$，其中 $S_t = \{i: q_i \in B_t, A_{t,i} \neq 0\}$。
- Actor 采用标准 Clipped PPO：$\mathcal{L}_{\text{actor}} = -\mathbb{E}[\min(\rho_t(\theta)A_t, \text{clip}(\rho_t(\theta),1-\epsilon,1+\epsilon)A_t)]$，$\epsilon=0.2$，$\gamma=1$，$\lambda=0.95$，KL 系数 0.02，Actor 为 rank-64 LoRA（scale 128，lr $10^{-6}$），Critic lr $10^{-4}$。

## 实验与结果
- **数据集**：训练使用 CG-Bench（1,219 个视频，12,129 个问题实例，SFT/QHPO 分区严格分离视频 ID）；评测在 LVBench（103 视频，均长 68.4 分钟，均 15.04 问题/视频）、Video-Holmes（270 视频，均长 2.8 分钟）和 LongVideoBench（753 视频，均长 7.9 分钟）上进行。
- **基线**：End-to-End（Video-R1、DeepVideo-R1）、Iterative（FrameThinker、TimeSearch-R）、Agentic（DVD、VideoHV-Agent、VideoARM）及内部对照（Direct、Ind. Loops SFT+PPO）。
- **主要结果（LVBench）**：
  - **VAMR 达 62.1% 准确率**，超越最强 Agentic 基线 VideoARM（57.8%）**+4.3 点**；推理轮次 0.9 vs 6.4（**-85.9%**），处理帧数 84.9 vs 220.2（**-61.4%**）。
  - 与同架构独立循环对照相比：准确率 +5.7 点，Rounds/Q 从 4.1→0.9，Frames/Q 从 224.0→84.9。
- **Video-Holmes**：VAMR 58.2% vs VideoARM 54.6%（+3.6 点），Rounds/Q 1.4 vs 4.4（-68.2%）。
- **LongVideoBench**：VAMR 66.5% vs VideoARM 65.9%（+0.6 点），Rounds/Q 1.8 vs 6.6（-72.7%）。
- **架构消融（LVBench）**：独立循环 45.2% → 共享回路 49.6% → +QA-inspector 53.9% → +Layered Memory 52.3% → 完整 VAMR 56.8%（+11.6 点 vs 独立循环，-72.9% 轮次）。
- **训练消融**：Backbone 56.8% → SFT 57.9% → PPO 58.7% → QHPO w/o alignment 60.0% → 完整 QHPO **62.1%**（+3.4 点 vs PPO，对齐贡献 +2.1 点）。
- **问题密度扩展**：同视频 1→16 问题共享一回路，准确率 58.1%→62.8%，Frames/Q 219.2→87.7。
- **证据重叠分析**：即使证据重叠率最低的四分位（Q1，仅 1.4% 问题对重叠），VAMR 仍提升 +7.0 点并减少 63.4% 帧处理量，证明收益不依赖高证据重叠。

## 相关工作脉络
1. **VideoARM / DVD / VideoHV-Agent**：现有代表性视频 Agent，均以单问题为单位组织在线状态、自适应检索和停止策略；VAMR 的核心差异是将多问题纳入同一个持久化循环，允许问题异步终止和共享取证。
2. **UniVA（Liang et al., 2025b）**：将多个相关问题打包为复合目标一次性输出所有答案；VAMR 保持每个问题为独立决策目标，允许它们在推理过程中分别指导工具调用并异步终止。
3. **FrameThinker / TimeSearch-R**：迭代式长视频推理方法，通过分层表示或时间搜索扩展视觉上下文；其循环结构仍围绕单查询，无法利用同视频内多问题的互补证据需求。
4. **ARCHER（Zhou et al., 2024）/ StepSearch / ToRL**：面向多轮工具使用 Agent 的策略优化方法，将轨迹关联到单一任务结果；QHPO 的创新在于在共享轨迹中区分多个异步终止问题的不同学习信号。
5. **PEI et al. / Lei et al.（2020, 2025）**：利用问题间语义关系作为辅助信息做一次性预测或特征提取；VAMR 的不同在于在在线推理阶段持续利用跨问题协调，而非仅在离线训练/特征阶段。
6. **Video-R1 / DeepVideo-R1**：端到端视频推理方法，在固定上下文中增强时序推理；VAMR 在其基础上引入可自适应检索的工具使用和持久化记忆，且支持多问题共享。

## 局限性与未来方向
1. **适用场景受限**：当前聚焦于"同一视频上多个问题"的场景，对跨视频、跨模态的多任务协同推理尚未涉及。
2. **共享回路扩展性待验证**：活跃窗口固定为 5 个问题，当问题数量极大或问题间关联度极低时，协调开销和记忆管理可能成为瓶颈。
3. **强化学习训练成本**：QHPO 需要完整的 on-policy  rollout 和专门的 Critic 训练，相比 SFT-only 方法训练流程更复杂。
4. **未来方向**：可扩展至其他共享轨迹的 Agent 训练场景；探索更高效的问题调度与窗口管理策略；结合更强原生模型配置（论文 Appendix G 中 training-free 版本已达 80.4% on LVBench）进一步验证泛化性。

## 研究启发与可借鉴点
1. **共享回路 + 问题级信用分配**：QHPO 的"问题级 Critic → 回合对齐 → 聚合更新"范式可迁移到其他多目标 Agent 场景（如多文档问答、多轮对话系统），解决"子任务异步终止"时的信用分配难题。
2. **两阶段 SFT（特权信息指导动作、去除特权生成推理）**：Stage 1 用标注答案/证据区间构造动作、Stage 2 仅用公开状态生成推理链，此设计可广泛应用于需要隐藏标注信息的 Agent 训练，防止推理链依赖不可得信息。
3. **问题条件感知检索**：将问题文本和选项作为条件注入视觉检索工具，使一次调用同时服务多个问题并返回差异化线索，该设计可推广至文档检索、代码搜索等需要多视角取证的任务。
4. **分层记忆（共享层 + 专属层）**：共享叙事层与问题专属层分离的设计在保证跨问题知识复用的同时避免无关线索干扰，可借鉴至多查询 RAG 系统和多智能体协作场景。
5. **高问题密度下收益放大**：实验表明问题密度越高 VAMR 相对独立循环的收益越大（+7.0 点在低重叠 quartile 依然显著），提示本方法特别适合密集多问题场景（如内容审核、安全监控、长视频索引检索）。

## 关键术语表
**VAMR**：Video Agent for Multi-Question Reasoning，一种将同一视频的多个问题协调到共享工具使用回路中的长视频理解 Agent 框架。

**QHPO（Question-Horizon Policy Optimization）**：面向共享多问题轨迹的策略优化算法，通过问题级 Critic 估计每个问题的优势，经回合对齐后聚合为回合级信号更新 Actor。

**QA-inspector**：问题条件感知工具，接受时间段和一组（问题, 选项）对，返回共享段落摘要和各问题的专属证据线索，支持单次调用服务多个问题。

**Shared Loop（共享回路）**：替代 per-question 独立轨迹的单个持久化 ReAct 循环，每回合输出包含多个问题答案提交和工具调用的联合动作，直到所有问题解答完毕。

**Layered Memory（分层记忆）**：由共享故事层 $G_t$、各问题专属线索层 $\{E_t^i\}$ 和观察缓冲 $T_t$ 组成的三层记忆结构，支持跨问题知识复用和独立问题证据隔离。

**Active Question Window（活跃问题窗口）**：当前暴露给策略模型的未解问题集合（上限 5 个），其余问题保留在有序后备队列中，随窗口空位逐步补充。

**Round Alignment（回合对齐）**：QHPO 中的信用分配机制，仅将问题 $q_i$ 的优势值保留在直接服务于该问题的回合（如针对 $q_i$ 的视觉调用或答案提交），其余回合置零。

**Frames/Q 与 Rounds/Q**：两个互补的效率评估指标，Frames/Q 统计每问题处理的视觉帧数（含重复），Rounds/Q 统计每问题的有效推理回合数（可低于 1）。

## 可复现要素
- **训练数据集**：CG-Bench（Chen et al., 2025），按视频 ID 划分 SFT（610 视频/5,775 问题）和 QHPO（584 视频/6,001 问题）分区，数据overlap 已审计；**公开**。
- **评测数据集**：LVBench、Video-Holmes、LongVideoBench，均为公开基准。
- **代码/权重开源情况**：论文未提及代码和权重的开源声明。
- **关键超参**：Actor LoRA rank=64, scale=128, lr=$10^{-6}$；Critic lr=$10^{-4}$；PPO clip $\epsilon=0.2$；GAE $\gamma=1, \lambda=0.95$；KL 系数 0.02；活跃窗口大小 5；每回合最多 6 次工具调用；SFT 教师模型 Qwen3-VL-32B；QHPO 3 轮 on-policy pass（1 轮 Critic warm-up + 2 轮联合更新），每更新采集 8 条视频轨迹，minibatch 4 视频，3 PPO epoch + 2 Critic epoch。

---
title: "WorldGuide-Goal-Directed-Video-World-Model-for-Procedural-Ta"
source: https://arxiv.org/pdf/2610.12459v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-11-01 00:49:01"
field: "视频生成与规划"
keywords: ["视频生成", "闭环规划执行", "世界模型", "过程性任务", "长程视频合成", "层级记忆"]
innovations: ["提出闭环规划-执行框架，联合训练视觉状态感知的动作预测与视频生成", "构建步骤级过程视频数据集 WorldGuide Bench 并提供显式完成监督", "引入层级视觉记忆机制有界控制长程生成的 token 成本"]
benchmarks: ["WorldGuide Bench", "Video-CraftBench"]
---

# 论文速读：WorldGuide-Goal-Directed-Video-World-Model-for-Procedural-Task-Execution

## 一句话总结
本文提出 WorldGuide，一种面向过程性任务执行的闭环视频世界模型，通过协同训练规划器（ContextPlanner）与执行器（Executor），使模型能根据生成的视觉状态自主决策下一步原子动作或任务终止，显著提升了长程任务完成的可靠性与成功率。

## 研究问题与动机
- 现有开环视频生成模型在生成长程过程性视频（如折纸、组装、烹饪）时无法根据已生成状态调整后续动作，容易导致步骤错误或任务完成后继续生成。
- 已有闭环规划-执行系统多依赖预训练或冻结的执行器，规划器正确指令仍可能因执行器无法实现原子动作而失败，缺乏对生成视觉状态的直接反馈。
- 任务完成判断通常依赖固定生成长度或外部重试预算，而非从视觉进展中学习终止决策。
- 长程执行需要持久的视觉上下文以维持状态连续性，但现有方法缺乏对历史 token 成本的有界控制。
- 缺少提供原子动作-视频配对及显式完成信号的步骤级监督数据集，制约了闭环过程性训练的可行性。

## 核心贡献（创新点）
- 提出 WorldGuide，将过程性视频生成建模为闭环任务执行，规划器根据视觉状态预测原子动作或完成信号，执行器直接实现该动作。
- 构建 WorldGuide Bench，包含约59K步骤级标注视频、245个任务和27个过程类别，提供原子动作、任务目标和显式完成监督。
- 设计层级视觉记忆机制，按时间远近以不同空间分辨率压缩历史潜在帧，有界控制 Executor 的 token 成本同时保持长程状态连续性。
- 证明闭环执行相比开环可将 Task Success 提升21.62个百分点，视觉反馈提升18.61个百分点，记忆模块在 WorldGuide Bench 和 Video-CraftBench 上分别带来5.77%和17.91%的提升。
- 在 WorldGuide Bench 和 Video-CraftBench 上取得最高 Task Success（33.33% 和 47.69%），超越 MiniMax-H3 等强基线，并保持最优美学质量和整体一致性。

## 方法详解
- **闭环框架**：给定初始图像 $x_0$ 和任务目标 $g$，在每一步 $t$，ContextPlanner $\pi_\theta$ 根据 $g$、当前视觉状态 $x_t$ 和最近 $K=3$ 个 clip-action 历史 $h_t$ 预测原子动作 $a_t \in \mathcal{A} \cup \{\text{DONE}\}$；若 $a_t = \text{DONE}$ 则终止，否则 Executor $f_\phi$ 生成下一 clip $x_{t+1}$。
- **ContextPlanner**：基于 Qwen2.5-VL-7B 监督微调，输入为任务目标、视觉状态和历史动作-clip对，输出自然语言动作或完成 token，使用 token-level cross-entropy 仅对响应部分计算损失。
- **Executor**：基于 HunyuanVideo-1.5 扩散 Transformer，在 flow-matching 参数化下预测残差速度，接受来自冻结 Planner 的动作嵌入、参考图像及层级视觉记忆作为条件，训练时仅更新 $\phi$。
- **层级记忆压缩**：采用 YUME/FramePack 的空间压缩调度，最新 latent frames 以高空间分辨率（$r=1$）处理，较旧 frames 依次以 $r=2,4,8,16,64$ 压缩，锚点（最老帧）以 $r=2$ 处理，确保历史 token 总数不超过8个全分辨率 latent frame 当量。
- **训练协议**：先微调 ContextPlanner，冻结后以其嵌入的动作 caption 微调 Executor；数据预处理在最后一个步骤后追加一个仅含 `<|Task Completed|>` 的目标记录，专门监督 Planner 的终止预测。

## 实验与结果
- **数据集**：WorldGuide Bench（58,679 训练视频，980 测试视频，245任务/27类别）和 Video-CraftBench（294样本）。
- **评估指标**：Task Success Rate、Plan Accuracy、Order Score、Repeat/Skip、VBench五维度（Aesthetic Quality、Imaging Quality、Dynamic Degree、Motion Smoothness、Overall Consistency）、Paired Alignment。
- **主要结果**：
  - WorldGuide（8.3B）在 WorldGuide Bench 上 Task Success 达 33.33%，Plan Accuracy 62.85%，Aesthetic Quality 49.36 最高；MiniMax-H3（33B）受参考动作计划约束仅得 29.90%。
  - 在 Video-CraftBench 目标条件生成下，WorldGuide Task Success 47.69%，显著高于 MiniMax-H3 的 32.73%。
  - 消融显示：闭环 vs 开环提升 +21.62% Task Success；移除视觉反馈降至 14.72%；移除记忆降至 27.56%（WorldGuide Bench）和 29.78%（Video-CraftBench）。
- **基线对比**：超越 PhysAgent、TempAct、Bernini 等闭环规划-执行方法，以及 Cosmos、Wan、HunyuanVideo 等开环生成模型。
- **人类评估**：WorldGuide 以平均 2.750/5 排名第一，验证自动评测有效性。

## 相关工作脉络
- **视频生成与 world models**：Veo、Sora、CogVideoX 等开环模型缺乏从视觉状态选择动作和判断终止的能力；VideoWorld 2、SPIRAL 等虽引入计划但不直接联合训练动作执行。
- **Planner-Executor 系统**：ORCA、CollabVR 使用冻结预训练执行器，正确计划可能执行失败；SPIRAL 依赖独立 critic 和强化学习优化执行；WorldGuide 联合训练规划与执行，动作预测直接驱动视频生成。
- **长上下文视频生成**：FramePack、YUME 通过压缩视觉历史维持连续性，但未结合过程性任务决策；WorldGuide 将其与闭环执行耦合。
- **过程性视频数据集**：COIN、CrossTask、YouCook2 等仅提供步骤定位或弱监督，缺少原子动作-视频配对和完成信号；WorldGuide Bench 填补这一空白。

## 局限性与未来方向
- **误差累积**：规划器可能重复动作、遗漏前置步骤或过早终止；执行器可能对正确指令产生几何或外观失真；错误在闭环中传播且无法撤销。
- **顺序训练限制**：Planner 和 Executor 分阶段训练，未共享任务完成目标，可能导致接口语义漂移；未来需联合优化。
- **记忆压缩损耗**：有界记忆丢弃旧细节可能影响后期步骤一致性；更选择性保留任务相关信息可改善长程表现。
- **视觉质量权衡**：为优先任务正确性，部分指标（如 Imaging Quality、Overall Consistency）略低于专业生成模型，需在过程可靠性和视觉保真度间更好平衡。
- **恢复机制缺失**：缺乏对失败步骤的显式检测与恢复，未来可引入状态有效性验证与修复策略。

## 研究启发与可借鉴点
- **闭环过程性建模**：将视频生成视为视觉状态驱动的循环决策过程，而非单次生成，对长程任务合成具有普适参考价值。
- **规划-执行联合训练**：在同一示范数据上先后微调 Planner 和 Executor，并通过冻结 Planner 保证动作嵌入与训练接口一致，可复用于其他多模态 agent 系统。
- **层级记忆压缩**：按时间远近分配不同空间分辨率的 token 成本，有效控制长序列计算开销，可迁移至任意需要长程视觉上下文的生成模型。
- **完成信号监督**：在序列末尾追加仅含终止 token 的训练样本，显式教授模型判断任务结束，适用于任何顺序决策任务。
- **数据构建流程**：使用大语言模型（Gemini）自动筛选、时间标注和步骤拆分，结合人工审计保证质量，为大规模过程视频数据集构建提供可操作范式。

## 关键术语表
- **ContextPlanner**：基于视觉进展预测下一步原子动作或任务完成信号的语言-视觉模型。
- **Executor**：条件扩散 Transformer，将 Planner 的动作嵌入渲染为下一段视频片段。
- **Hierarchical Memory**：按时间远近以递增压缩率存储历史 latent frames 的视觉记忆模块，控制 token 开销。
- **Task Success Rate**：最终生成视频是否可视完成给定任务目标的百分比。
- **WorldGuide Bench**：包含约59K步骤级标注过程视频的基准数据集，覆盖245任务27类别。
- **Flow-Matching**：扩散模型训练目标，预测从噪声到数据的残差速度。
- **Open-loop vs Closed-loop**：前者预先确定动作序列，后者根据生成状态实时调整决策。
- **Atomic Action**：描述单一物理操作的动作单元（如“折叠右下角”、“添加香料”）。

## 可复现要素
- **数据集**：WorldGuide Bench（训练集57,699视频，测试集980视频），论文计划公开来源标识、时间戳和步骤标注，视频本身未开源。
- **代码/权重**：GitHub mbzuai-oryx/WorldGuide 和项目页面提供交互式比较视频；模型权重未明确提及开源状态。
- **关键超参**：Planner 历史长度 K=3；Executor 目标分辨率 480×832，24 FPS；训练精度 BF16；Planner 学习率 $2\times10^{-5}$，Executor 学习率 $1\times10^{-5}$；采样 10步 denoising，guidance scale 7.5，flow shift 5.0。
- **硬件**：Planner 训练 8节点×8 AMD Instinct MI210（64GB）；Executor 训练同配置，42天。

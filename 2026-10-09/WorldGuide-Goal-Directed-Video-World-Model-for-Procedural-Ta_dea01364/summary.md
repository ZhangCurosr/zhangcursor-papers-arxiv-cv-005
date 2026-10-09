---
title: "WorldGuide-Goal-Directed-Video-World-Model-for-Procedural-Ta"
source: https://arxiv.org/pdf/2610.12459v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-10-09 09:49:07"
field: "视频生成与程序化任务执行"
keywords: ["视频生成", "世界模型", "程序化任务", "闭环生成", "规划器-执行器", "长程视频"]
innovations: ["规划器-执行器联合训练的闭环程序视频生成框架", "层级压缩视觉记忆机制控制长程token开销", "WorldGuide Bench：59K步骤标注程序视频数据集"]
benchmarks: ["WorldGuide Bench", "Video-CraftBench"]
---

# 论文速读：WorldGuide-Goal-Directed-Video-World-Model-for-Procedural-Task-Execution

## 一句话总结
WorldGuide 提出了一种用于程序化任务执行的闭环视频世界模型，通过将程序视频生成建模为"规划器预测原子动作→执行器渲染对应片段→利用生成结果选择下一步或终止"的闭环循环，显著提升了目标导向的长程程序视频生成成功率。

## 研究问题与动机
- **开环生成无法适应执行结果**：现有视频生成模型在 rollout 前固定 prompt/动作序列，若某步骤出错则无法纠正，且可能继续生成直到任务已完成（图1a）。
- **已有闭环系统规划-执行脱节**：ORCA、CollabVR 等方法将预训练 VLM 规划器与冻结的视频执行器耦合，正确规划仍可能因执行器无法实现目标动作而失败（图1b）；SPIRAL 依赖独立评审器和强化学习而非直接监督原子动作执行。
- **长程任务需要持久视觉上下文**：每一步生成的片段影响后续决策，但无界历史会导致 token 开销线性增长。
- **缺乏步骤级动作-视频监督数据**：现有指令视频数据集（EgoPlan-IT、COIN、YouCook2 等）缺少原子动作标注、任务完成信号和任务目标的联合监督，无法直接用于闭环程序训练。

## 核心贡献（创新点）
1. **闭环程序视频生成框架**：提出 ContextPlanner + Executor 的递归规划-执行循环，规划器从生成的视觉状态预测下一个原子动作或 DONE 标记，执行器直接渲染该动作。*本质区别：与 ORCA/CollabVR 等冻结执行器的方案不同，执行器与规划器在同一套演示数据上联合训练。*
2. **层级压缩视觉记忆机制**：采用 YUME/FramePack 的多尺度空间压缩策略，将较旧的历史 latent 帧以更低空间分辨率嵌入，使 token 开销有界（≤8 个完整 latent 帧等价），支持长程状态连续性。
3. **WorldGuide Bench 数据集**：构建约 59K 步骤标注程序视频，覆盖 245 个任务、27 个程序类别，提供原子动作-视频配对和显式完成信号，填补了程序视频闭环训练的监督空白。
4. **规划-执行联合训练的实证分析**：系统验证了闭环执行（+21.62% Task Success）、视觉反馈（+18.61% Task Success）、记忆机制（+5.77%/+17.91%）各自贡献，揭示了规划与执行错误的强耦合关系。

## 方法详解
**整体框架**（Eq. 1）：
给定初始图像 $x_0$ 和任务目标 $g$，在每步 $t$：
- ContextPlanner $\pi_\theta$ 预测 $a_t \in \mathcal{A} \cup \{\text{DONE}\}$，输入为 $g$、当前视觉状态 $x_t$ 和历史 $h_t$（最近 K=3 个 clip-action 对）；
- 若 $a_t = \text{DONE}$ 则终止；否则 Executor $f_\phi$ 生成下一 clip $x_{t+1}$，输入为 $a_t$、$x_t$ 和层级记忆 $M_t$；
- 更新 $h_{t+1} = h_t + a_t$，$M_{t+1} = \text{update}(M_t, x_{t+1})$。

**ContextPlanner**：基于 Qwen2.5-VL-7B-Instruct，全参数 SFT，用 token-level cross-entropy 仅监督 response tokens，学习预测下一步原子动作或完成标记，无架构修改。

**Executor**：基于 HunyuanVideo-1.5 扩散 Transformer，采用 flow-matching 参数化（Eq. 2）：
$$\mathcal{L}_{\text{exec}} = \mathbb{E}_{z,\epsilon,\sigma}\left[\frac{\|m \odot (\hat{y}_\phi - (\epsilon - z))\|_2^2}{\max(\sum m, 1)}\right]$$
其中 text condition 包含由冻结 ContextPlanner 嵌入的动作 caption，vision condition 包含参考图像和层级记忆。

**层级记忆压缩**（§3.4）：
- 历史容量 $T \leq 1365$，最旧帧作为 anchor，总历史长度 $F \leq T+1$；
- 多尺度 patch embedding：$E_r$ 为 3D 卷积，stride $(1, pr, pr)$，$r \in \{1,2,4,8,16\}$，最粗 $E_{64}$ 通过 $E_{16} \circ \Phi$ 实现，有效空间 stride=128；
- Token 预算约束（Eq. 6）：$\frac{p^2}{HW}|\mathcal{M}| \leq 8$，确保 context token 数有界。

**训练流程**：先 SFT 训练 ContextPlanner，冻结后将其 action embeddings 用于 Executor fine-tuning（顺序训练，非端到端），以保持自然语言接口的可解释性和诊断性。

## 实验与结果
**数据集**：WorldGuide Bench（58,679 训练视频 + 980 测试视频，245 任务/27 类别）；Video-CraftBench（294 样本，积木拼装和折纸任务）。

**基线**：MiniMax-H3（33B）、HunyuanVideo-1.5-8.3B、Wan2.2-14B、LTX-2.5-22B、FlashMotion-34B、Yume-1.5-5B 等主流视频生成模型；Bernini、PhysAgent、TempAct 等闭环 planner-executor 方法。

**核心结果**：
- **WorldGuide Bench**：Task Success 33.33%（最强），Plan Accuracy 62.85%，Aesthetic Quality 49.36（最高）；对比 MiniMax-H3 的 29.90%（接受 reference action plan），WorldGuide 仅凭 goal-only 实现超越。
- **Video-CraftBench**：Task Success 47.69%（最强），对比 MiniMax-H3 的 32.73%，提升显著。
- **消融**（Table 4）：open-loop → 33.33%（+21.62%）；w/o visual → 33.33%（+18.61%）；w/o Mem → 27.56%（WorldGuide Bench）/29.78%（Video-CraftBench，+17.91%）；Oracle ContextPlanner → 38.82%（+5.49%），说明规划误差仍有优化空间。
- **人工评估**：WorldGuide 均分 2.750/5，高于 MiniMax-H3 的 2.542（Table 15）。

## 相关工作脉络
- **Bernini [34]**：在 latent 语义空间规划后再渲染，不从生成进度中递归预测连续程序动作；WorldGuide 则在视觉空间中闭环执行。
- **ORCA [14]/CollabVR [15]**：预训练 VLM 规划器 + 冻结视频执行器，正确规划仍可能执行失败；WorldGuide 将执行器与规划器在同一数据上联合训练。
- **SPIRAL [44]**：联合训练规划与生成，但依赖独立 critic 和 RL 精炼执行；WorldGuide 直接监督原子动作执行，无需额外评审器。
- **PhysAgent [19]**：通过场景重建和物理模拟精炼物理程序后再合成；WorldGuide 直接在视觉空间学习程序执行，不依赖物理引擎。
- **TempAct [39]**：联合训练规划与执行但遵循预设时间跨度，无视觉状态重规划；WorldGuide 的规划器从生成进展中动态选择下一步。
- **Video Language Planning [6]**：结合策略、值函数和树搜索；WorldGuide 采用更轻量的单循环闭环，无需树搜索开销。

## 局限性与未来方向
- **错误累积传播**：规划器重复/跳过动作或执行器几何畸变会导致后续决策基于错误视觉状态，无法回溯修正。
- **顺序训练限制**：规划器与执行器未联合优化，Executor 的 loss 不反向传播至 Planner，存在接口优化空间。
- **有界记忆的遗忘问题**：压缩较旧历史帧可能丢弃后续步骤所需细节，启用记忆同时增加了 Repeat/Skip 比率。
- **后期视觉质量退化**：长程 rollout 后期出现纹理/场景退化（图13、14）。
- **未来方向**：引入状态有效性信号和显式恢复机制；联合优化规划器-执行器；更智能的记忆选择性保留。

## 研究启发与可借鉴点
1. **规划-执行联合训练范式**：将 planner 和 executor 在同一套演示数据上训练（planner 先训、冻结后训 executor），可确保执行器真正学会实现规划器预测的动作，而非仅依赖预训练能力。
2. **闭环反馈的价值量化**：通过 w/o visual / w/o Mem / oracle 等消融，清晰分离了规划误差与执行误差的贡献，为后续改进提供明确方向。
3. **层级记忆压缩的工程实践**：多尺度空间压缩（$r \in \{1,2,4,8,16,64\}$）将 token 开销严格限制在 ≤8 个完整帧等价，是长程视频生成的实用技术，可直接迁移至其他长上下文生成任务。
4. **完成信号的设计**：在最终 step 后追加一条只有 planner 学习的"无 clip 目标"样本（target = `<|Task Completed|>`），以自然语言 token 形式监督终止决策，设计简洁有效。
5. **数据集构建方法**：使用 Gemini 2.5 Flash 自动标注原子动作时间戳和任务完成信号，并通过人类审核（81.6%-84.5% 正确率）验证质量，为程序视频数据构建提供了可复用的 pipeline。

## 关键术语表
**ContextPlanner**：基于 Qwen2.5-VL-7B 的规划模块，从任务目标和生成的视觉历史中预测下一步原子动作或完成标记。
**Executor**：基于 HunyuanVideo-1.5 的扩散 Transformer，接收规划器嵌入的动作 caption 和视觉上下文，生成对应的视频片段。
**闭环任务执行（Closed-loop Task Execution）**：每步根据已生成的视觉状态重新决策下一步动作或终止，而非开环固定计划。
**层级视觉记忆（Hierarchical Visual Memory）**：对历史 latent 帧按空间分辨率分级压缩的多尺度记忆机制，控制 token 开销有界。
**WorldGuide Bench**：约 59K 步骤标注程序视频数据集，覆盖 245 任务/27 类别，提供原子动作-视频配对和完成信号。
**Flow Matching**：扩散模型的参数化方式，模型预测残差速度 $\hat{y}_\phi$ 以逼近 $\epsilon - z$。
**Task Success Rate**：视频级评估指标，衡量最终可见状态是否达成完整任务目标。
**Plan Success**：文本级评估指标，衡量预测的动作序列若被正确执行是否能达成目标。

## 可复现要素
- **数据集**：WorldGuide Bench 不公开视频本身，计划开源源标识符、时间戳和步骤标注；Video-CraftBench 公开。
- **代码**：GitHub https://github.com/mbzuai-oryx/WorldGuide（已开源）。
- **权重**：项目页提供交互式结果和生成视频；模型权重未在论文中明确声明开源状态（需进一步确认）。
- **关键超参**：ContextPlanner 基于 Qwen2.5-VL-7B-Instruct，history 长度 K=3；Executor 基于 HunyuanVideo-1.5-8.3B；训练精度 BF16，DeepSpeed ZeRO-3，有效全局 batch=1024，学习率 2×10⁻⁵（planner）/1×10⁻⁵（executor）；推理采样 10 denoising steps，guidance scale 7.5，flow shift 5.0；rollout 上限 80 steps。

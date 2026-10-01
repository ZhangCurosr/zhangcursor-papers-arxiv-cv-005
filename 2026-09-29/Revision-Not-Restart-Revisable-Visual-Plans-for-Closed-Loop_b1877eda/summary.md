---
title: "Revision-Not-Restart-Revisable-Visual-Plans-for-Closed-Loop"
source: https://arxiv.org/pdf/2609.35439v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-10-01 09:34:27"
field: "具身智能与机器人控制"
keywords: ["world-action model", "robotic manipulation", "diffusion policy", "feedback-driven planning", "visual future revision", "closed-loop control"]
innovations: ["将中间去噪状态作为可修订的持久视觉计划而非丢弃重采样", "三阶段解耦训练（历史适配→修订桥梁→偏差估计器）", "自适应更新策略在保留/两档桥接/新鲜间按偏差阈值优先级选择"]
benchmarks: ["RoboMME", "RMBench"]
---

# 论文速读：Revision-Not-Restart-Revisable-Visual-Plans-for-Closed-Loop

## 一句话总结
论文提出 **Revisable Temporal Planning (RTP)** 方法，将世界-动作模型（WAM）预测的视觉未来作为持久化的动作条件，在执行反馈到来时通过**学习的修订桥梁（revision bridge）**恢复中间去噪状态并适配当前观测，而非放弃已有预测从头重生成，从而在 RoboMME（48.6%）和 RMBench（84.8%）上显著优于从头重新规划基线。

## 研究问题与动机
1. **预测-反馈不一致问题**：WAM 预测的视觉未来在执行过程中可能因接触时机延迟、物体状态变化等与实际观测产生偏差；直接复用原预测会保留时序误差，而从头噪声重生成又重复了昂贵的视觉生成过程。
2. **已有方法不足**：现有持久记忆方法仅保留历史观测；效率型 WAM 减少生成开销但不解决反馈修订；diffusion planner 支持轨迹修复但依赖重去噪起点而非保存的中间状态。
3. **核心科学问题**：如何在保留已有预测的计划结构信息的前提下，根据新观测对视觉计划的未执行部分进行**连续修正**，并将其用于下一次动作解码？
4. **闭环性能关联**：视觉修订质量如何真正转化为任务成功率的提升？需要隔离"学习修正""检查点来源""动作前缀条件"等组件的独立贡献。

## 核心贡献（创新点）
1. **持久化视觉动作条件的修订框架**：将未执行视觉计划作为可跨越反馈边界持续存在的条件，区别于仅维护观测历史的工作（如 MemoryWAM），首次系统性地将视觉计划的修订纳入闭环控制。
2. **学习的修订桥梁（revision bridge）**：通过恢复冻结视觉生成器在生成过程中保存的中间去噪状态，并在当前观测条件下追加学习的速度修正项继续演化，与 Diffusion ReRoll（从重去噪终点重新生成）的本质区别在于**起点是真实中间状态而非随机噪声**。
3. **三阶段训练解耦**：Stage A 适配时间感知历史；Stage B 冻结基座训练修订桥梁（视觉+动作联合监督+新鲜一致性正则）；Stage C 拟合偏差估计器，分离了各子模块的学习难度，避免端到端训练的不稳定性。
4. **自适应更新策略与偏差估计器**：基于估计偏差与校准阈值的比较，在"保留 / Bridge-5 / Bridge-10 / 重新规划"间做优先级选择，将计算工作量从固定深度策略的 12.33 步降至 7.78 步（降幅 36.9%）。

## 方法详解
**时间感知历史（Time-Aware History）**
- 控制边界 $b_n$ 为执行动作块后的物理采样时刻。
- **时间金字塔采样**：三档跨度/步长 $(q_\ell, s_\ell) \in \{(12,1), (76,4), (204,8)\}$，保留近期细节与远期证据，并按物理时间戳组织。
-  packed 上下文经 Prefill 得到层级的 KV 缓存 $C_n = \{K_n^{(\ell)}, V_n^{(\ell)}\}$ 作为事实接口 $G_n$。

**预测状态与检查点**
- 每次重规划生成 $H=4$ 组视觉窗（20 步 Euler 积分），存储完整清洁端点 $x_N$ 及 Bridge-5/Bridge-10 所需的中间检查点 $S_{n,j}$。
- 每执行 $H_c=1$ 组，消耗索引 $c_n$ 递增，剩余 $L_n = H - c_n$ 组保持活跃。

**修订桥梁（Feedback-Conditioned Revision Bridge）**
- 反馈描述符：
$$\Delta_n = \text{DifEncode}\big(\text{Endpoint}(P_n),\, E_v(o_{n+1}),\, A_n^{\text{applied}},\, s_n, s_{n+1},\, \text{GroundSummary}(G_{n+1}),\, M\big)$$
- 恢复 $S_{n,N-k}$（$k \in \{5, 10\}$）后继续演化：
$$x_{j+1} = x_j + h_j \left[ v_\theta(\cdot) + r_\eta\!\big(h_\theta(\cdot),\, \Delta_n,\, u_j,\, a_n\big)\right]$$
其中 $v_\theta$ 为冻结的视觉速度场，$r_\eta$ 为零初始化输出层的 MLP 修正项。

**损失函数**
$$\mathcal{L}_B^{(k)} = d_v(\hat{z}^{(k)}, \text{sg}(Z^*)) + d_a(A^{(k)}, \text{sg}(A^*)) + 0.1 \cdot d_v(\hat{z}^{(k)}, \text{sg}(\hat{z}^{\text{fresh}}))$$
三项分别为：观测续行视觉 MSE、行为动作 MSE、新鲜参考一致性正则（系数 0.1）。

**自适应选择**
- 偏差估计器拟合 heteroscedastic 绝对误差目标：
$$\mathcal{L}_S = \sum_{m,r}\left[\frac{|y_r^m - \widehat{d}_r^m|}{\widehat{u}_r^m} + \log\widehat{u}_r^m\right]$$
- 按优先级 retention → Bridge-5 → Bridge-10 选取首个通过 $\widehat{d}_r^m + \beta \widehat{u}_r^m \leq \tau_r$ 的策略；全部失败时回退 fresh。

## 实验与结果
- **数据集**：RoboMME（16 任务，50 重置/任务，共 800 重置）、RMBench（9 双臂任务，100 重置/任务，共 900 重置）。
- **基线**：$\pi_{0.5}$、MME-VLA（TTT/FrameSamp）、MemER、Fast-WAM、LingBot-VA（dense/time-aware）、MemoryWAM。
- **主要结果**（任务平均成功率）：
  - **RoboMME**：RTP **48.6%** vs. LingBot-VA (time-aware) 40.4%，提升 **+8.2 pp**；vs. Fixed bridge-10 47.4%，提升 +1.2 pp。
  - **RMBench**：RTP **84.8%** vs. LingBot-VA (time-aware) 79.9%，提升 **+4.9 pp**；vs. MemoryWAM 83.0%，提升 +1.8 pp。
- **计算效率**：RTP 平均每次非初调用视觉步数 7.78 步，较 Fixed bridge-10（12.33 步）降低 **36.9%**；均次调用时间 1.051s vs. 1.115s。
- **关键消融**：
  - 学习修正 vs. 零残差 Fixed bridge-10：**+5.8 pp**（Table 17，最清晰的组件证据）。
  - 时间感知历史 vs. 密集采样：RTP 提升 +8.2 pp。
  - 保存检查点 vs. 独立重去噪重建：名义上 +2.8 pp，但 95% CI 含 0，效应估计不精确。
  - 动作前缀效应：当前事实下修订 vs. 保留提升 +2.5 pp，CI [0.0, 5.0] 触及 0。
  - 移除动作监督：视觉误差降低但任务成功率下降 2.5 pp，证明预测精度≠控制效用。

## 相关工作脉络
1. **Diffusion Policy / Streaming Diffusion Policy**（Chi et al., 2023; Høeg et al., 2024）：支持滚动视界动作序列生成与增量更新，但依赖重去噪端点；RTP 利用**真实保存的中间状态**而非重扰动起点。
2. **Diffusion ReRoll**（Kim et al., 2026）：token-wise 重去噪的增量生成；本质是"重新采样"，RTP 是"在原有轨迹上纠偏"。
3. **MemoryWAM**（Yang et al., 2026）：持久世界-动作记忆，仅维护历史观测；RTP 额外维护**可修订的预测计划**。
4. **LingBot-VA**（Li et al., 2026a）：本文基座模型，原生因果预训练；RTP 在其上叠加时间感知历史与修订机制，二者是**基础-扩展**关系。
5. **Adaptive Online Replanning**（Zhou et al., 2023）：基于规划似然选择保留/修复/重采样；RTP 的候选族（保留/两档桥接/新鲜）与偏差估计器提供更细粒度的计算-性能权衡。
6. **Fast-WAM / Light-WAM**（Yuan et al., 2026; Li et al., 2026c）：聚焦生成效率；RTP 聚焦**反馈条件下的计划修订**，两者正交可组合。

## 局限性与未来方向
1. **有限预测窗口**：仅处理 $H=4$ 组，更长交互下累积误差未知；作者明确提到"extend across longer interactions"是未来方向。
2. **同步执行假设**：实验使用同步环境步进，未覆盖异步/高延迟部署。
3. **组件效应估计不精确**：检查点来源（checkpoint source）与动作前缀效应（prefix effect）的 95% CI 跨越 0，隔离贡献的证据强度有限。
4. **反复拟合收益有限**：Recurrent fitting 降低偏差超标率（1.43%→0.49%）但未在任务成功率上显著超越匹配重拟合（+0.125pp，CI [-2.5, 2.75]）。
5. **未在大尺度真实机器人上验证**：仅在仿真基准（RoboMME、RMBench）测试。

## 研究启发与可借鉴点
1. **中间状态复用范式**：将生成器的中间 denoising state 作为"计划草稿"保存并在反馈后微调，思路可迁移至视频生成、轨迹规划、任何分步生成的世界模型场景。
2. **三阶段解耦训练**：先适配基础模型，再冻结基座训练修订模块，最后拟合选择器——这种分离大幅降低联合训练的优化难度，可作为复杂多组件系统的通用训练协议。
3. **视觉-动作联合监督 + 新鲜一致性正则**：修订损失中用 0.1 系数对齐"未修正部分与从头生成的分布距离"，兼顾修订的针对性与全局一致性，技巧简洁有效。
4. **配对 bootstrap 置信区间评估**：以 reset key 为抽样单元做 10,000 次重采样，报告任务平均与配对差异的 95% CI，为后续工作的公平比较树立统计规范。
5. **与团队方向的结合点**：团队关注的**流式视觉理解 / 长时间记忆 / 世界模型**可直接借鉴 RTP 的时间金字塔历史与修订桥梁设计；若做具身 Agent，可将 revision bridge 抽象为"基于反馈的 future state correction module"。

## 关键术语表
**World–Action Model (WAM)**：联合建模视觉动态与机器人动作的生成模型，用预测的未来视觉序列条件化下一步动作。
**Revisable Temporal Planning (RTP)**：本文提出的框架，通过学习的修订桥梁在反馈到来时修正持久化的视觉计划。
**Revision Bridge**：恢复冻结视觉生成器的中间去噪检查点，并在当前观测条件下继续演化剩余步数的修正模块。
**Time-Aware History**：以原始环境时间戳组织的多尺度金字塔历史记录，保留近期细节与远期证据。
**Feedback Descriptor ($\Delta_n$)**：由 DifEncode 输出的向量，拼接预测/观测 latent、控制、本体感与其差值，作为修订的条件输入。
**Discrepancy Estimator**：Stage C 训练的异方差绝对误差回归器，估计各候选更新模式与新鲜生成之间的视觉/动作偏差及其不确定度。
**Bridge-5 / Bridge-10**：分别从第 15 步或第 10 步（共 20 步）的检查点继续修订的两种深度。
**Recurrent Fitting**：在 RTP 实际运行积累的过渡状态上重新拟合修订桥梁与估计器的二次训练策略。

## 可复现要素
- **数据集**：RoboMME（公开基准）、RMBench（公开基准）；训练数据为各自 benchmark 附带的演示集（RMBench 50 演示/任务，RoboMME 按 manifest）。
- **代码/权重**：论文声明"**will release the code and model checkpoints**"（复现性声明），但截至文章版本未提供下载链接。
- **关键超参**：预测窗口 $H=4$、消耗 $H_c=1$、Euler 步数 $N=20$、动作步数 $N_a=50$、时间金字塔 $(12,1),(76,4),(204,8)$、历史预算 $B_H=60/72$（RoboMME/RMBench）、AdamW 学习率 $10^{-5}$（Stage A）/ $10^{-4}$（Stage B/C）、$\beta=2.1291$（RoboMME）/ $2.1114$（RMBench）。
- **硬件**：单卡 NVIDIA H200 141GB GPU，bfloat16 推理。

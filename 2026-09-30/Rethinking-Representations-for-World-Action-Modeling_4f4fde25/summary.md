---
title: "Rethinking-Representations-for-World-Action-Modeling"
source: https://arxiv.org/pdf/2609.38163v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-10-01 09:41:09"
---

# 论文速读：Rethinking-Representations-for-World-Action-Modeling

## 一句话总结
提出ReWAM，一种以表示为核心的世界-动作模型（WAM），通过将冻结的预训练DINO特征经校准与时间瓶颈压缩为紧凑状态，并结合行动接地表示塑造（AGRS）的非对称梯度路由机制，在无需生成式视频预训练的条件下实现了高效的联合世界预测与机器人策略学习。

## 研究问题与动机
- 现有WAM多依赖Video-VAE潜变量进行未来观测预测，该类潜空间以视觉重建保真度为优化目标，未必显式优先保留对控制至关重要的语义与动态信息。
- 直接采用预训练感知特征（如DINO）虽在策略学习上优于重建导向潜变量，但帧级特征缺乏显式时序动态编码，且统计分布与扩散模型训练不匹配。
- 重建保真度本身并不能保证表示对控制任务的有效利用，需系统研究表示空间的构建、校准与适配机制。
- 如何合理划分“策略”与“世界模型”在共享表示空间中的分工，避免世界预测损失反向重塑表示导致的信息塌陷或预测捷径？

## 核心贡献（创新点）
1. 系统验证了表示空间设计对WAM性能的决定性作用，证明单纯追求像素重建保真度无法保证控制效用，且预训练感知特征需经结构化校准方可适配动态建模。
2. 提出ReWAM框架，通过多层聚合、运行归一化与表示感知噪声调度，将冻结DINO特征转化为适合扩散预测的紧凑时序状态。
3. 设计行动接地表示塑造（AGRS）机制，采用非对称梯度路由：仅让动作损失梯度回传更新TRB，世界预测损失梯度被阻断，实现“策略定义表示语义、世界模型学习表示演化”的角色分离。
4. 在无需生成式视频预训练的条件下，于RoboTwin 2.0与RoboDojo基准上取得最优或领先性能，并在真实双臂机器人上验证了部署优势。

## 方法详解
- **整体架构**：ReWAM采用Mixture-of-Transformers (MoT)设计，包含世界DiT分支 $W_\theta$ 和动作DiT分支 $A_\theta$。给定历史/当前多视角观测 $(o_{-1}, o_0)$、本体感觉 $s_0$ 及语言指令 $\ell$，编码器 $E_\phi$ 输出当前状态 $z_0$，并预测未来状态序列 $z_{1:H}$ 与动作块 $a_{1:H}$。条件分布因式分解为 $p_\theta(z_{1:H} \mid z_0, c) \, p_\theta(a_{1:H} \mid z_0, c)$，两分支共享参数 $\theta$ 但注意力屏蔽跨组交互。
- **特征校准 (Feature Calibration)**：
  - **多层聚合**：从DINOv3-L的指定层 $\mathcal{S}=\{12,14,16,18,20,22,224\}$ 提取patch特征，平均后加上最后一层的空间均值，融合深层语义与图像级上下文，公式：$F_p^{\mathrm{MLA}} = \frac{1}{|\mathcal{S}|}\sum_{l\in\mathcal{S}} F_p^{(l)} + \frac{1}{N}\sum_{q=1}^N F_q^{(24)}$。
  - **归一化**：TRB输出采用基于历史批次的逐通道running mean/variance进行Causal running normalization，推理时统计量固定，避免分布漂移。
  - **表示感知噪声调度**：根据目标张量标量元素总数 $D$ 与基准 $D_{\mathrm{base}}=4096$ 计算 $\alpha=\sqrt{D/D_{\mathrm{base}}}$，动态调整flow matching的噪声采样时间 $\tau = \frac{\alpha u}{1+(\alpha-1)u}$，使更大特征维度自适应更高噪声输入。
- **时间表示瓶颈 (TRB)**：将连续两帧的DINO特征沿时空维度切分为 $2\times2\times2$ block，展平为 $8d$ 维token，跨视角拼接后经过单层8头自注意力与线性投影压缩至128维，形成紧凑的世界状态 $z_j$。
- **行动接地表示塑造 (AGRS)**：联合训练损失 $\mathcal{L} = \mathcal{L}_{\mathrm{world}} + \mathcal{L}_{\mathrm{action}}$ 基于flow matching。关键设计在于梯度路由：$\nabla_\theta \mathcal{L}$ 包含两部分，但 $\nabla_\phi \mathcal{L} = \left(\frac{\partial z_0}{\partial \phi}\right)^\top \frac{\partial \mathcal{L}_{\mathrm{action}}}{\partial z_0}$。世界分支的当前态与未来态表示均使用stop-gradient，确保TRB仅受动作目标驱动更新，防止预测损失“反向污染”表示语义。

## 实验与结果
- **数据集与基线**：RoboTwin 2.0（50任务，clean/random双设置）、RoboDojo（5类能力评估）、真实机器人Piper双臂平台（Collect Objects, Fold Clothes, Unpack Lunchbox）。基线涵盖 $\pi_0$、$\pi_{0.5}$、Fast-WAM、LingBot-VA、AHA-WAM、LDA-1B、X-VLA、Spatial Forcing等。
- **主要结果**：
  - **RoboTwin 2.0**：ReWAM在clean与random设置下均达到 **93.6%** 成功率，超越最强无预训练基线AHA-WAM（avg 92.8%），且**无需生成式视频预训练**。
  - **RoboDojo（无预训练）**：平均得分 **9.42**，成功率 **5.82%**，显著领先同类方法；配合约 **600小时** 实机预训练数据后，平均分提升至 **12.29**，成功率 **8.28%**，在Generalization与Long-Horizon类别上取得最高分。
  - **真实机器人**：三任务平均成功率 **91.7%**（55/60），对比Fast-WAM的80.0%（48/60），在Fold Clothes与Unpack Lunchbox上分别提升15与20个百分点，且观察到无 demonstrations 覆盖的自主重试行为。
- **消融结论**：特征校准累计提升平均成功率 **12.20pp**；TRB相比固定随机投影更具效率优势（128维即达最佳，64维仍有效）；AGRS阻断世界损失回传是关键，全通路径平均下降6.30pp，证实防止表示坍塌的重要性；RGB重建

---
title: "Rethinking-Representations-for-World-Action-Modeling"
source: https://arxiv.org/pdf/2609.38163v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-10-01 09:40:31"
field: "具身智能与世界模型"
keywords: ["World-Action Model", "Representation Learning", "Robot Manipulation", "DINO Features", "Diffusion Model", "Action-Grounded Training"]
innovations: ["不对称梯度路由实现策略对表征空间的主动塑形", "无生成式视频预训练下通过DINO特征校准达到93.6%成功率"]
benchmarks: ["RoboTwin 2.0", "RoboDojo"]
---

# 论文速读：Rethinking-Representations-for-World-Action-Modeling

## 一句话总结
论文提出ReWAM，一种不依赖生成式视频预训练的表征为中心的世界动作模型，通过冻结DINO特征校准与动作驱动的表征塑形（AGRS），让策略决定"表征编码什么"而世界模型学习"表征如何演化"，在RoboTwin 2.0上达到93.6%成功率。

## 研究问题与动机
1. **表征空间设计缺失**：现有WAM（如Motus、Fast-WAM）依赖Video-VAE潜空间，该空间针对视觉重建优化而非控制任务，缺乏对控制相关信息的显式优先。
2. **重建保真度≠控制效用**：论文通过受控比较发现，即使Video-VAE的像素重建SSIM（0.91）高于DINO（0.70），在无生成式预训练时Raw DINO仍比Video-VAE潜量提升4.90/9.22个百分点。
3. **感知特征需适配动力学建模**：帧级DINO特征缺乏显式时间变化编码，统计分布与扩散训练不匹配，直接预测效果受限。
4. **角色分工不清**：当前方法中世界模型与策略共享同一表征空间但梯度同时更新两者，导致表征可能被世界预测目标反向塑造。

## 核心贡献（创新点）
1. **揭示表征设计的三层问题**：不仅编码器选择重要，还需解决"特征如何为预测准备→如何按时间组织→如何适配控制需求"，三者缺一不可。
2. **Feature Calibration**：通过多层聚合、归一化、表征感知噪声调度，将冻结DINO特征校准至扩散建模适配的分布，在RoboTwin 2.0上平均成功率提升12.20个百分点。
3. **Action-Grounded Representation Shaping（AGRS）**：实现不对称梯度路由——仅动作损失梯度回传至TRB，世界模型只学习表征的演化，策略决定表征内容；该设计使TRB维数从512降至128时仍提升成功率。
4. **无生成式预训练的SOTA结果**：无需大规模视频生成预训练，在RoboTwin 2.0达到93.6%成功率，在RoboDojo上以约600小时具身数据超越多类基线。

## 方法详解
### 整体架构
- **输入**：历史观测$o_{-1}$、当前观测$o_0$、本体感觉$s_0$、语言指令$\ell$
- **编码器**：冻结DINOv3-L backbone + 特征校准模块 + 可训练TRB，参数$\phi$
- **生成器**：Mixture-of-Transformers (MoT)，世界分支$W_\theta$与动作分支$A_\theta$，各自独立DiT（30层，hidden width 2048/1024，总参3.58B）
- **条件分解**：$p_\theta(z_{1:H}, a_{1:H}|z_0, c) = p_\theta(z_{1:H}|z_0, c) \cdot p_\theta(a_{1:H}|z_0, c)$

### Feature Calibration
1. **Multi-layer aggregation**：从DINOv3-L块$\mathcal{S}=\{12,14,16,18,20,22,24\}$提取patch特征，平均后叠加末层空间均值：
$$F_p^{\text{MLA}} = \frac{1}{|\mathcal{S}|}\sum_{l \in \mathcal{S}} F_p^{(l)} + \frac{1}{N}\sum_{q=1}^N F_q^{(24)}$$
2. **Normalization**：TRB输出使用running statistics（指数移动平均）标准化
3. **Representation-aware noise schedule**：按目标维度缩放flow time：$\tau = \frac{\alpha u}{1+(\alpha-1)u}$，其中$\alpha=\sqrt{D/D_{\text{base}}}$，$D_{\text{base}}=4096$

### Temporal Representation Bottleneck (TRB)
- 将连续两帧的非重叠$2\times2\times2$时空块flatten为8d token
- 跨视角concatenation后，经单层multi-head self-attention（8头，2D RoPE）和线性投影压缩至128维

### Action-Grounded Representation Shaping (AGRS)
- **损失函数**：
$$\mathcal{L}_{\text{world}} = \mathbb{E}\|W_\theta(\bar{z}^{\tau_z}, \tau_z; \bar{z}_0, c) - (\epsilon_z - \bar{z})\|_2^2$$
$$\mathcal{L}_{\text{action}} = \mathbb{E}\|A_\theta(a^{\tau_a}, \tau_a; z_0, c) - (\epsilon_a - a)\|_2^2$$
- **不对称梯度路由**：
$$\nabla_\phi \mathcal{L} = \left(\frac{\partial z_0}{\partial \phi}\right)^\top \frac{\partial \mathcal{L}_{\text{action}}}{\partial z_0}$$
世界损失梯度在$z_0$和$z_{1:H}$处均stop-gradient，仅动作损失梯度流经$z_0$到达TRB。

## 实验与结果
### 数据集与设置
- **RoboTwin 2.0**：50任务双臂操作，clean与random两种设置，各100 episode评估
- **RoboDojo**：5类能力（Generalization/Precision/Long-Horizon/Memory/Open），35任务3,500轨迹
- **实机评估**：Piper双臂平台，3任务（Collect/Fold/Unpack），各20次尝试

### 主要结果
| 方法 | 具身预训练 | RoboTwin 2.0 Clean | RoboTwin 2.0 Random | RoboDojo Avg Score | RoboDojo SR |
|------|-----------|-------------------|-------------------|-------------------|-------------|
| Fast-WAM (no PT) | ✗ | 91.9 | 91.8 | 3.48 | 2.03 |
| AHA-WAM | ✗ | 93.4 | 92.2 | 4.82 | 2.39 |
| **ReWAM (Ours)** | ✗ | **93.6** | **93.6** | **9.42** | **5.82** |
| Fast-WAM* (600h PT) | ✓ | — | — | 10.95 | 6.80 |
| **ReWAM** | ✓ | — | — | **12.29** | **8.28** |

- ReWAM在RoboTwin 2.0领先第二优方法0.2/1.4个百分点
- 实机平均成功率：ReWAM 91.7% vs Fast-WAM 80.0%

### 关键消融
- **Feature Calibration**：Normalization (+3.92pp) → Noise schedule (+1.41pp) → MLA (+6.87pp)，累计+12.20pp
- **TRB**：Fixed random projection (+2.89pp)，但AGRS再+3.94pp
- **AGRS梯度路由**：阻断世界→TRB（当前）得90.70%/89.12%；开放两者分别降至84.32%/79.40%，证实不对称路由必要性
- **TRB维度**：AGRS在128维最优（90.70%/89.12%），64维仍可维持89.90%/89.08%，远优于fixed projection随维度下降的敏感性

## 相关工作脉络
1. **Fast-WAM [4]**：移除测试时未来视频去噪，仅保留联合训练效益；本文在其架构上替换表征空间，聚焦表征设计而非推理加速。
2. **LDA-1B [5] / DexWorldModel [6]**：直接预测DINO特征；本文引入时间压缩（TRB）与动作塑形（AGRS），解决原始特征的时序结构与控制适配问题。
3. **V-JEPA [34,35] / DINO-world [36]**：预测学习框架验证特征空间有效性；本文进一步将其引入联合预测-控制训练并验证不对称梯度路由的价值。
4. **RAE [41,42] / REPA [38]**：表示自编码器用于生成建模；本文借鉴其校准思想但目标从生成转为控制，并通过AGRS实现控制对表征的主动塑形。
5. **X-VLA [56] / Spatial Forcing [57]**：显式几何/空间对齐方法；本文证明无几何增强时，表征塑形本身即可在Precision任务上达到有竞争力表现。
6. **π0/π0.5 [48,49]**：VLA基线；本文表明即使在无大规模具身预训练下，表征设计优化的WAM可匹敌甚至超越部分VLA。

## 局限性与未来方向
1. **Open指令遵循能力不足**：在RoboDojo Open类别上成功率仅0.25%，作者归因于缺少生成式视频预训练导致的指令泛化限制。
2. **Precision任务仍有差距**：未引入显式几何增强，低于X-VLA等几何敏感方法。
3. **表征分析局限于两任务**：t-SNE与方向一致性分析仅覆盖Move Stapler Pad和Place Mouse Pad，泛化性需进一步验证。
4. **实机评估规模有限**：仅3个任务，且依赖10k步post-training，未见zero-shot实机迁移结果。

## 研究启发与可借鉴点
1. **不对称梯度路由是可复用的表征塑形机制**：AGRS将"策略定义什么"与"世界模型学习如何演化"解耦，可迁移至任何共享表征的联合预测-控制架构。
2. **表征设计应分三阶段思考**：校准（适配预测分布）→ 组织（时空压缩）→ 塑形（动作目标导向），这一框架可系统化指导其他世界模型工作。
3. **TRB维度的鲁棒性**：AGRS在128维即达最优，说明动作驱动的表征学习能高效压缩信息，为低维可控表征设计提供实证支持。
4. **无生成式预训练的可行性**：证明了高质量DINO特征+合理表征设计可媲美依赖Video-VAE的方法，降低了对大规模视频生成预训练的依赖。
5. **与团队方向结合机会**：若团队关注低资源/小数据场景，AGRS提供的轻量表征塑造路径可减少数据依赖；若关注多模态交互，可将此框架扩展至语言/触觉等多模态表征联合塑形。

## 关键术语表
**World-Action Model (WAM)**：联合学习机器人策略与未来观测预测的统一模型框架。
**Feature Calibration**：对冻结DINO特征进行多层聚合、归一化与噪声调度，使其适配扩散建模的分布。
**Temporal Representation Bottleneck (TRB)**：将帧级DINO特征压缩为紧凑时空token的表示瓶颈层。
**Action-Grounded Representation Shaping (AGRS)**：仅将动作损失梯度路由至TRB的不对称训练机制，使策略塑形表征而世界模型学习演化。
**Flow Matching**：一种连续时间扩散建模方法，通过线性插值将数据分布映射到高斯分布。
**Representation Autoencoder (RAE)**：使用预训练特征定义生成潜空间的自编码器架构。
**RoboTwin 2.0**：包含50任务的仿真实验基准，用于评估双臂机器人操作策略。
**RoboDojo**：覆盖Generalization/Precision/Long-Horizon/Memory/Open五类能力的综合性机器人操作基准。

## 可复现要素
- **数据集**：RoboTwin 2.0（仿真，需许可证）、RoboDojo（仿真）、自收集600小时实机演示数据（论文未公开）
- **代码**：开源于 https://github.com/hustvl/ReWAM
- **预训练权重**：DINOv3-L（公开）、UMT5（公开）、ReWAM模型权重（论文未明确声明是否开源）
- **关键超参**：batch size=512，峰值学习率=$10^{-4}$，cosine decay至$10^{-6}$，AdamW优化器，32×H20 GPU训练

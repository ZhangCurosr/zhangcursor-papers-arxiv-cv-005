---
title: "VersaCamVLA-Camera-Configurable-VLA-Policies-for-Robotic-Man"
source: https://arxiv.org/pdf/2610.12451v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-10-09 09:48:16"
field: "机器人视觉语言动作模型"
keywords: ["VLA", "Camera-Configurable", "Scene Token", "Robot Manipulation", "Viewpoint Generalization", "Multi-Camera"]
innovations: ["统一场景token接口解耦相机表示与动作学习", "WAPS利用腕部相机自然运动提供pose多样性", "多信号目标视图预测（RGB+语义+边缘）学习view-consistent表征"]
benchmarks: ["RoboTwin 2.0", "LIBERO", "Cobot Magic Real Robot"]
---

# 论文速读：VersaCamVLA: Camera-Configurable VLA Policies for Robotic Manipulation

## 一句话总结
论文提出VersaCamVLA框架，通过学习统一场景token接口将任意数量的 posed RGB 视图映射为固定大小的隐式场景token，使预训练VLA策略能够灵活适应部署时变化的相机数量和位姿，无需显式3D重建。

## 研究问题与动机
- VLA模型在训练中依赖固定相机配置，部署时相机数量或位姿变化会导致视觉token分布改变，策略性能显著下降。
- 现有方法（多姿态演示采集、新视角合成、几何先验注入）存在采集成本高、合成 fidelity 受限或需要额外3D感知的局限。
- 直接多视角方法将额外相机token拼接到策略输入，仍绑定训练时的相机数量和布局，且token长度随输入视图数增长。
- 实际部署中相机布置受机器人本体、工作空间或任务约束，理想方案应支持变量相机配置且对位姿无假设。

## 核心贡献（创新点）
- 首次 formulate 相机可配置VLA策略学习问题，要求策略在变量 posed RGB 视图输入下保持固定大小视觉接口。
- 提出两阶段VersaCamVLA框架：第一阶段学习多信号目标视图预测的 unified scene-token interface，第二阶段将冻结的场景encoder注入预训练VLA作为补充视觉条件。
- 引入Wrist-Augmented Pose Sampling (WAPS)，利用腕部相机自然运动提供免费的位姿多样性，无需额外数据采集即可覆盖多样相机配置。
- 在RoboTwin 2.0、LIBERO和真实机器人平台上验证，在相机位姿扰动和数量变化下均优于 prior VLA方法和直接多视角baseline。

## 方法详解
**整体框架**：两阶段学习，解耦相机集合表示学习与动作学习。

**Stage I - 统一场景token接口学习**：
- Scene Encoder：给定源相机集S，对每个posed视图计算Plücker ray embedding map，经ViT-style patchification得到patch token，RGB patch与ray embedding拼接后线性投影为source token。L个可学习scene token与source tokens拼接，经Transformer encoder聚合为固定大小scene tokens Z ∈ R^(L×d)。
- Scene Decoder：查询目标相机ray，与Z经Transformer decoder得到target tokens，经轻量预测头输出RGB、语义图、边缘图。
- Multi-signal target-view supervision：L_scene = L_rgb + λ_sem L_sem + λ_edge L_edge，其中L_rgb包含MSE+perceptual loss，语义和边缘用MSE监督。

**WAPS (Wrist-Augmented Pose Sampling)**：
- 每帧从静态相机随机采样n个(1~|C_stat|)作为源视图子集，腕部相机以0.5概率独立入选。
- 目标视图集合为所有可用相机，decoder始终预测全部视图。
- 提供self-reconstruction（源视图重建）和cross-view prediction（ Held-out视图预测）两种监督信号。

**Stage II - 场景token条件化VLA策略学习**：
- 丢弃decoder和预测头，冻结scene encoder。
- 引入convolutional spatial encoder g_η将L个scene tokens压缩为紧凑表示 Z̄_t（3072→192 tokens）。
- 策略条件于语言指令L、原生RGB观测V_t、补充场景表示Z̄_t和本体感知P_t。
- 优化flow-matching action regression loss：L_action = E[||v_η,ω(A_t^τ; L, V_t, Z̄_t, P_t, τ) - (A_t - ε)||²]。

**关键设计**：无需显式3D重建/深度估计/点云处理/新视角渲染，仅需RGB图像与相机内外参。

## 实验与结果
**数据集与基准**：
- RoboTwin 2.0：16个双臂精细操作任务，Clean和DR设置，各100 trials。
- LIBERO：4个suite（Spatial/Goal/Object/Long），每任务50 trials。
- 真实机器人：Cobot Magic双臂平台，3个任务（PickCube/StackCube/Insert Test Tube），各20 trajectories。

**主要结果**：
- RoboTwin 2.0 Clean：VersaCamVLA平均成功率52.56%，最优；DR设置23.00%，显著优于baseline。
- LIBERO：平均95.9%，与π0.5(95.8%)持平，优于OpenVLA-OFT(94.5%)和Diffusion Policy(72.4%)。
- 相机位姿鲁棒性（RoboTwin 2.0）：位姿扰动下，π0.5成功率下降16.0 pp，VersaCamVLA仅降0.8 pp；真实机器人下降28.3 pp vs +5.0 pp。
- 相机数量变化：VersaCamVLA在3-6视图下平均成功率稳定在52.56%-55.63%，而π0.5需为不同视图数单独训练。
- 真实机器人4视图seen pose：VersaCamVLA 53.33% vs π0.5 41.67%；unseen pose：58.33% vs 13.33%（+45.0 pp）。

## 相关工作脉络
- VLA模型敏感性研究：Recent evaluations (Libero-Plus, VLAst) 显示VLA对相机配置变化敏感，本文针对此underexplored问题。
- 视角泛化方法：多姿态演示采集[9-12]成本高，新视角合成[13]依赖fidelity，几何先验[15-20]需3D感知，本文无需这些额外组件。
- 多相机感知：直接拼接多视角token[21-23,38-43]绑定相机数量和布局，本文通过scene-token接口解耦。
- LVSM[46]：本文scene encoder-decoder初始化借鉴其目标视图预测思想，但应用于VLA策略增强而非纯view synthesis。
- 位姿鲁棒VLA：AnycamVLA[14]为零样本相机适应，本文通过显式scene representation学习实现。

## 局限性与未来方向
- 依赖标定相机内外参，未处理uncalibrated camera场景。
- 聚焦桌面双臂操作，缺乏自主视角选择(active perception)能力。
- 未探索移动端工作空间(camera配置在线优化)的扩展。
- 安全关键部署需在大型相机配置偏移下进一步验证任务特定失败模式。

## 研究启发与可借鉴点
- **表示-动作解耦设计**：将视觉表示学习与策略学习分离，通过固定大小token接口兼容变量输入，可迁移至其他多模态输入适应问题。
- **WAPSpose多样性获取策略**：利用机器人本体自然运动（腕部相机）作为免费pose augmentation source，无需额外数据采集，可推广至其他sensor配置。
- **多信号目标视图监督**：RGB+语义+边缘的组合监督平衡photometric reconstruction与结构感知，启发其他representation learning任务设计。
- **轻量spatial bottleneck**：用convolutional encoder压缩scene tokens为task-oriented bottleneck，避免直接注入大量tokens影响推理效率。
- **与团队方向结合**：若团队研究多相机机器人感知或VLA部署适配，此框架可作为即插即用的camera-configurable module。

## 关键术语表
**Vision-Language-Action (VLA) Model**：将预训练视觉语言模型适配到机器人控制，映射视觉观测、语言指令和本体感知到机器人动作的foundation policy。
**Scene Token Interface**：VersaCamVLA提出的统一表示层，将任意数量posed RGB视图压缩为固定大小latent tokens，解耦相机配置与动作学习。
**Plücker Ray Embedding**：用6维Plücker坐标（方向和力矩）编码相机ray的几何表示，用于注入空间位置先验。
**Wrist-Augmented Pose Sampling (WAPS)**：训练时随机采样静态相机和腕部相机作为源视图子集，利用腕部自然运动提供多样位姿监督。
**Multi-signal Target-View Prediction**：通过RGB重建、语义分割和边缘检测三路监督信号，强制scene tokens编码view-consistent空间信息。
**Flow-Matching Action Regression**：VLA策略的动作预测目标，通过噪声到 demonstration action 的线性插值学习velocity field。
**Domain Randomization (DR)**：在仿真中随机化外观和场景属性，评估策略对视觉变化的鲁棒性。

## 可复现要素
- **数据集**：RoboTwin 2.0（仿真）、LIBERO（仿真）、Cobot Magic真实机器人平台（论文提供任务描述）
- **代码/权重**：项目页面https://boyaohan.github.io/VersaCamVLA.github.io/，论文未明确声明开源状态
- **关键超参**：
  - Patch size p=8, 输入分辨率224×224
  - Scene encoder/decoder: 12-layer Transformer, d=768, 12 attention heads
  - 场景token数L=3072 (3×32×32 spatial layout)
  - 压缩后token数192 (3×8×8, channel=2048)
  - Stage I: lr=1e-4, batch=208, 100k steps, AdamW (β1=0.9, β2=0.95)
  - Stage II: lr=5e-5, batch=512, 30k steps, AdamW
  - Loss weights: λ_sem=λ_edge=1.0, λ_perc=0.5
  - 8×NVIDIA H100 GPUs, bf16 mixed precision

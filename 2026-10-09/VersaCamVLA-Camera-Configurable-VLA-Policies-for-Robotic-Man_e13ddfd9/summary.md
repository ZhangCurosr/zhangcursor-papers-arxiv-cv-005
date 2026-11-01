---
title: "VersaCamVLA-Camera-Configurable-VLA-Policies-for-Robotic-Man"
source: https://arxiv.org/pdf/2610.12451v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-11-01 00:48:50"
field: "具身智能与机器人操作"
keywords: ["VLA", "相机配置鲁棒性", "场景token", "多视角操作", "位姿泛化", "flow-matching"]
innovations: ["两阶段解耦的场景token接口学习，实现任意相机配置下的统一视觉表示", "WAPS利用腕部相机自然运动作为免费位姿多样性来源", "多信号目标视角预测（RGB+语义+边缘）提升场景token的空间结构编码能力"]
benchmarks: ["RoboTwin 2.0", "LIBERO", "真实机器人双臂操作"]
---

# 论文速读：VersaCamVLA: Camera-Configurable VLA Policies for Robotic Manipulation

## 一句话总结
论文提出 VersaCamVLA，一个相机可配置的 VLA 框架，通过学习统一的场景 token 接口，将任意数量、任意位姿的 RGB 相机视图压缩为固定大小的隐式场景表示，再注入预训练 VLA 策略，实现不同相机数量和未知相机位姿下的鲁棒操作。

## 研究问题与动机
- **现有 VLA 对相机配置高度敏感**：预训练 VLA（如 $\pi_{0.5}$、OpenVLA）的视觉输入与训练时的相机数量和位姿紧密绑定，部署时增加/减少相机或改变相机位姿会导致视觉 token 分布偏移，策略性能显著下降。
- **多视图拼接方案存在缺陷**：直接拼接多相机视觉 token 的方法（如 append 多视图特征）使策略 token 长度随输入相机数线性增长，且仍受限于训练时的固定相机布局，无法支持部署时动态变化的相机配置。
- **多姿态数据采集成本高**：通过采集多姿态演示或合成新视角来增强视角泛化的方法，在真实机器人上采集代价昂贵，且基于合成视角的方法依赖渲染/生成保真度。
- **几何先验方法引入额外开销**：使用深度、点云或 3D 重建的方法需要额外的 3D 传感模块或显式重建步骤，部署复杂度高。

## 核心贡献（创新点）
- **提出相机可配置的 VLA 策略学习范式**：要求策略在输入相机数可变、相机位姿未固定的情况下，仍通过统一固定大小的视觉接口进行动作预测，解耦相机集表示学习与动作学习。
- **设计两阶段统一场景 token 接口**：Stage I 通过多信号目标视角预测（RGB + 语义图 + 边缘图）训练场景编码器；Stage II 冻结编码器，用轻量空间编码器压缩场景 token 后注入预训练 VLA，无需重设计骨干网络。
- **引入 WAPS（腕部增强位姿采样）技术**：利用机械臂自然运动带来的腕部相机位姿多样性，随机采样静态相机子集与腕部相机作为编码器输入，迫使场景 token 支持跨视角泛化，无需额外数据采集。
- **系统实验验证**：在 RoboTwin 2.0、LIBERO 和真实双臂机器人平台上，VersaCamVLA 在相机位姿扰动下成功率仅下降 0.8 pp（对比 $\pi_{0.5}$ 下降 16 pp），支持 3–6 路相机灵活输入，真实机器人未见过位姿下提升 45 pp。

## 方法详解
**整体框架**：两阶段学习，解耦场景表示学习与动作策略学习。

**Stage I — 统一场景 Token 接口学习**：
- **Plücker 射线嵌入**：对每个已知内外参的输入相机视图，计算 Plücker 射线嵌入图 $\mathbf{R}^i \in \mathbb{R}^{H \times W \times 6}$（3 维方向 + 3 维矩），与 RGB 通道拼接为 9 维输入。
- **ViT 分块化**：图像 resize 到 224×224，patch size $p=8$，每视图 $P=784$ 个 patch，经线性投影 $\phi_{\text{src}}$ 映射到 $d$ 维 source token。
- **Transformer 场景编码器**：将 $N \times P$ 个变长 source token 与 $L$ 个可学习 scene token $\mathbf{E}$ 拼接，经 Transformer encoder 自注意力聚合，输出固定大小 $\mathbf{Z} \in \mathbb{R}^{L \times d}$（$L=3072$，组织为 $3 \times 32 \times 32$ 空间布局）。
- **场景解码器与多信号目标视角预测**：给定目标相机 $c$，将其 Plücker 射线经 $\phi_{\text{tgt}}$ 投影为 query token $\mathbf{Q}^c$，与 $\mathbf{Z}$ 拼接经 Transformer decoder 输出 $\mathbf{Y}^c$，再经三个轻量卷积预测头分别恢复 RGB 图像（MSE + 感知损失）、语义分割图（MSE）、边缘图（Canny，MSE）。总损失：
$$\mathcal{L}_{\text{scene}} = \mathcal{L}_{\text{rgb}} + \lambda_{\text{sem}} \mathcal{L}_{\text{sem}} + \lambda_{\text{edge}} \mathcal{L}_{\text{edge}}, \quad \lambda_{\text{sem}}=\lambda_{\text{edge}}=1.0, \lambda_{\text{perc}}=0.5$$
- **WAPS（腕部增强位姿采样）**：每帧从静态相机池随机采样 $n \in [1, |C_{\text{stat}}|]$ 个视角，腕部相机以 0.5 概率独立入选，目标集为全部可用相机。迫使场景 token 同时支持源视角自重建和目标视角跨视角预测。

**Stage II — 场景 Token 条件化 VLA 策略学习**：
- 丢弃场景解码器和三个预测头，冻结场景编码器。
- **空间编码器压缩**：轻量卷积模块 $g_\eta$ 将 $3 \times 32 \times 32$（ch=768）的场景 token 网格压缩为 $3 \times 8 \times 8$（ch=2048），即 192 个 token，作为 VLA 的附加视觉条件。
- 策略输入为语言指令 $\mathcal{L}$、原生 RGB 观测 $\mathcal{V}_t$、压缩场景 token $\bar{\mathbf{Z}}_t$、本体感觉 $\mathcal{P}_t$，通过 $\pi_{0.5}$ 的动作专家以 flow-matching 损失预测动作 chunk（$H_a=50$）：
$$\mathcal{L}_{\text{action}} = \mathbb{E}_{\tau,\epsilon}\left[\|v_{\eta,\omega}(A_t^\tau; \mathcal{L}, \mathcal{V}_t, \bar{\mathbf{Z}}_t, \mathcal{P}_t, \tau) - (A_t - \epsilon)\|_2^2\right]$$
- 推理时仅需 RGB 图像 + 相机内外参，无需 3D 重建或新视角渲染。

## 实验与结果
**数据集与设置**：
- **RoboTwin 2.0**：双臂操作，16 个精细任务，6 环境相机 + 2 腕部相机，Clean 和域随机化（DR）两种设置，每任务 100 trials。
- **LIBERO**：单臂操作，4 个套件（Spatial/Goal/Object/Long），6 环境相机 + 1 腕部相机，每任务 50 trials。
- **真实机器人**：Cobot Magic 双臂平台，3 个任务（PickCube/StackCube/Insert Test Tube），20 trajectories/task，4 环境相机 + 2 腕部相机。

**主要结果**：
| 基准 | RoboTwin Clean | RoboTwin DR | LIBERO Avg |
|------|---------------|-------------|------------|
| $\pi_{0.5}$ | 35.88% | 21.63% | 95.8% |
| **VersaCamVLA** | **52.56%** (+16.68) | **23.00%** (+1.37) | **95.9%** (+0.1) |

- **相机位姿扰动鲁棒性**（4 视图匹配）：$\pi_{0.5}$ 成功率从 48.88% 降至 32.88%（−16.0 pp），VersaCamVLA 仅从 53.44% 降至 52.63%（−0.8 pp）；真实机器人上 $\pi_{0.5}$ 下降 28.3 pp，VersaCamVLA 反而提升 5.0 pp。
- **相机数量灵活性**：VersaCamVLA 单模型在 3–6 视图下平均成功率稳定在 52.56%–55.63%；$\pi_{0.5}$ 在 3/4 视图分别训练，无法支持 5/6 视图。
- **消融实验**：WAPS 带来 +3.84 pp；多信号监督（RGB+Edge+Sem）较仅 RGB 提升 8.56 pp；192 个压缩 token 最优（55.63% vs 48→50.06%, 432→47.50%）；LVSM 预训练初始化较从头训练提升 4.38 pp。
- **推理延迟**：4 视图下 231.20 ms（vs $\pi_{0.5}$ 230.93 ms），从 4 增至 6 视图仅增加 7.06 ms（vs $\pi_{0.5}$ 29.47 ms）。

## 相关工作脉络
- **VLA 基础模型（RT-1/RT-2/Octo/OpenVLA/π₀/π₀.₅）**：将预训练 VLM 适配到机器人控制，但均固定输入相机数和位序，本文填补了相机配置灵活性的空白。
- **视角泛化方法**：包括多姿态数据采集 [9–12]、新视角合成 [13] 和测试时适应 [14]——本文不依赖数据合成，而是通过跨视角预测目标直接学习位姿不变表示。
- **几何感知方法（DP3/GeoPredict/SPATIALVLA 等）**：引入深度、点云或 3D 特征，需要额外 3D 传感；本文仅使用 RGB + 相机内外参，零额外传感开销。
- **多相机策略（ACT/OpenVLA-OFT/PEAFOWL）**：直接拼接多视图 token，token 长度随相机数增长且绑定训练布局；本文通过固定大小场景 token 实现任意相机数兼容。
- **LVSM 场景表示**：Inspired by LVSM 的目标视角预测框架，本文将其引入 VLA 策略学习并扩展到多信号监督与腕部采样。

## 局限性与未来方向
- 假设相机内外参已标定，未处理未标定相机场景。
- 聚焦桌面双臂操作，未扩展到移动平台或更大尺度 workspace。
- 缺少自主视角选择能力，依赖人工布置的相机配置池。
- 未来方向：扩展到未标定相机、在线主动感知优化相机配置、移动工作空间部署。

## 研究启发与可借鉴点
- **场景 token 接口的两阶段解耦设计**可迁移到其他需要灵活多模态输入的机器人策略学习中，如多传感器融合（LiDAR + RGB）、多机器人协作。
- **多信号目标视角监督**（RGB + 语义 + 边缘）值得借鉴，尤其在数据稀缺时，结构化监督信号比纯像素重建更能学到 manipulation-relevant 的几何表示。
- **WAPS 的思想**（利用设备自带传感器的自然运动作为免费位姿多样性来源）可推广到其他具身平台，如移动底盘的轮式里程计轨迹、无人机的气动扰动等。
- **空间编码器的信息瓶颈设计**（3072 → 192 token）为后续研究提供了可复用的压缩范式，平衡了信息保留与策略计算效率。

## 关键术语表
**VLA (Vision-Language-Action)**：将预训练视觉-语言模型适配到机器人控制的多模态策略模型，输入视觉观测 + 语言指令，输出连续动作序列。
**Plücker 射线嵌入**：用 6 维向量（方向 + 矩）参数化空间中每条射线，用于编码相机位姿的几何信息。
**Scene Token**：场景编码器输出的固定大小隐式表示，聚合多视角观测的空间信息，独立于输入相机数量和位姿。
**WAPS (Wrist-Augmented Pose Sampling)**：利用腕部相机的自然运动提供位姿多样性，通过随机采样静态/腕部相机子集训练场景编码器。
**Flow-Matching**：基于流匹配理论的持续动作预测方法，通过噪声到动作的线性插值路径建模条件概率流。
**CAMERA-CONFIGURABLE**：指策略可接受变长、变位姿的相机输入集合，而非训练时的固定相机布局。

## 可复现要素
- **数据集**：RoboTwin 2.0（开源模拟器）、LIBERO（开源基准）、自建真实机器人数据集（论文未公开原始视频）。
- **代码/权重**：项目页面 https://boyaohan.github.io/VersaCamVLA.github.io/，论文声明代码将在审核期间公开。
- **关键超参**：patch size $p=8$，分辨率 224×224，场景 token 数 $L=3072$，压缩后 192 token；Stage I 学习率 $1\times10^{-4}$，100K steps，batch=208；Stage II 学习率 $5\times10^{-5}$，30K steps，batch=512；8×H100 GPU，bf16 混合精度。
- **初始化**：场景编解码器从预训练 LVSM checkpoint 微调，空间编码器随机初始化，$\pi_{0.5}$ 使用官方预训练权重。

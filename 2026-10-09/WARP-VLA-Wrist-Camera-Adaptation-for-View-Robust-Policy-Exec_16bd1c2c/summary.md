---
title: "WARP-VLA-Wrist-Camera-Adaptation-for-View-Robust-Policy-Exec"
source: https://arxiv.org/pdf/2610.11508v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-10-09 17:29:28"
field: "具身智能/机器人视觉-语言-动作模型"
keywords: ["Vision-Language-Action", "Wrist Camera", "Mixture of Experts", "View Robustness", "Robot Policy", "Feature Adaptation", "Plug-and-Play"]
innovations: ["基于 MoE 的特征级适配器无需相机标定即可适配腕部视角变化", "伪规范锚点+空间token统计的路由机制隐式感知视角偏差", "跨VLA backbone即插即用且模拟到真实机器人zero-shot transfer"]
benchmarks: ["LIBERO", "LIBERO-Plus"]
---

# 论文速读：WARP-VLA-Wrist-Camera-Adaptation-for-View-Robust-Policy-Exec

## 一句话总结
针对机械臂腕部相机安装位姿变化导致 VLA 策略性能骤降的问题，作者提出 WARP-VLA，通过基于 MoE 架构的特征级适配器将扰动视角的腕部特征映射到"伪规范特征空间"，在无需相机标定信息、无需策略微调的情况下实现即插即用的部署鲁棒性。

## 研究问题与动机
1. **核心问题**：VLA 策略在真实部署中难以复现训练时的腕部相机位姿（存在安装公差、振动等扰动），而腕部相机提供精细操作所需的近距离视图，微小位姿偏移即可改变夹爪-物体的空间关系，导致策略严重失效。
2. **现有方法不足**：
   - 基于相机几何的方法（如 Plücker ray embeddings）需要准确的相机外参，而腕部相机标定往往不可靠或缺失；
   - 基于新颖视角合成（NVS）的方法（如 AnyCamVLA）需额外训练视图合成模型且在仿真域需要多视角数据；
   - 现有方法主要针对固定外部相机设计，难以直接迁移到随机械臂运动的腕部相机场景。
3. **为什么是腕部相机**：腕部相机视场随末端执行器移动，安装位置差异直接影响交互区域的关键几何线索，且跨机器人硬件安装方式各异，复现原始配置更加困难。

## 核心贡献（创新点）
1. **提出首个系统性的腕部相机视角鲁棒性评测基准**：在 LIBERO 和 LIBERO-Plus 上构建覆盖多种腕部相机配置的 view-augmented 数据集，填补该领域评测空白。
2. **设计基于 MoE 的特征级适配器 WARP-VLA**：用多个专家学习不同相机位姿区域的特征变换，通过特征条件路由隐式融合，无需相机外参输入即可实现部署时无微调的即插即用适配。
3. **引入 pseudo-canonical feature anchor 与空间 token 统计的路由机制**：通过通道/空间统计残差表征感知视角偏差并路由至相应专家，避免引入额外视觉编码器。
4. **实验验证跨 VLA backbone 的通用性**：在 π_0.5 和 GR00T N1.7 上均有效，LIBERO 平均成功率提升 39.1pp（39.2%→78.3%），LIBERO-Plus 外域迁移提升 27.7pp，并在真实机器人上验证 transfer 有效性。

## 方法详解
1. **问题定义**：给定规范视角 token X^c 和扰动视角 token X^p（S×D 维），适配器 f_φ 学习映射 X^p→X^c，使规范策略可无缝接收适配后特征而不需重新微调。
2. **伪规范特征 anchor**：对策略微调数据集所有帧的腕部特征求平均得到 anchor A，跨场景/任务平均降低观察特定方差，提供稳定参考。
3. **MoE 适配器架构**：插入在 image encoder 与 action policy 之间，将扰动腕部特征映射回规范特征空间。E=15 个专家，共享 8 层 ViT 骨干（hidden dim=2048, attention dim=256, 4 heads），专家输出加权融合。
4. **Anchor-Guided Routing**：路由输入 z = Φ_stat(X^p - A) ⊕ Φ_stat(A)，其中 Φ_stat 提取通道均值 μ、标准差 σ 及空间一阶矩 m^x、m^y（8D 拼接）。路由网络为两層 MLP（hidden dim=2048）+ GELU。
5. **分组路由**：15 专家分为两组——translation 组含 13 个专家（覆盖半圆形 (r,θ) 偏移空间）和 tilt 组含 2 个专家；各组独立归一化路由 logit，避免无关因素竞争。
6. **专家参数化（低秩因子化）**：借鉴 LoRA，每个专家投影矩阵 W_e,m^l = W_m^l + A_e,m^l B_e,m^l（rank r=16），初始化时 B 置零使所有专家从相同变换出发逐渐专业化。
7. **损失函数**：总损失 L = L_e + L_g；L_e 为各专家独立 MSE 重建误差（d_e = ||F_e(X^p, A) - X^c||_F² / SD），L_g 为路由监督交叉熵；责任估计 h_e 结合路由概率 g_e、teacher mask T_e（基于已知相机扰动 soft assignment）和重建误差高斯核，温度参数控制敏感度。
8. **推理部署**：推理时仅依赖输入特征的残差统计量路由，无需相机参数，冻结 backbone，plug-and-play 可直接部署。

## 实验与结果
- **数据集**：LIBERO（40 任务，2000 演示，10 种扰动配置重渲染）；LIBERO-Plus（10K episodes）。
- **评估条件**：Canonical / Small (r=2cm, τ=3°) / Medium (r=4cm, τ=6°) / Large (r=6cm, τ=9°)，共 14 种配置/级别（7个方位角 × 2个倾斜角）。
- **基线**：Backbone（π_0.5, GR00T N1.7）、Geometry-aware（Spatial Forcing, GAM）、Camera-aware（KYC, AnyCamVLA）。
- **LIBERO In-Domain**：
  - π_0.5 基础：Canonical 95.2%，Small 90.5%，Medium 20.9%，Large 6.1%，平均 39.2%
  - **+WARP-VLA（π_0.5）**：Canonical 98.2%，Small 95.5%，Medium 80.5%，Large 58.8%，**平均 78.3%**（提升 +39.1pp）
  - 超越 AnyCamVLA（平均 72.3%）和 KYC（平均 62.3%）
- **LIBERO-Plus 外域迁移（无额外微调）**：
  - π_0.5 + WARP-VLA：平均 56.5%，超越 AnyCamVLA 54.7%
  - GR00T N1.7 + WARP-VLA：平均 30.8%，提升 +19.1pp
- **真实机器人实验**（xArm6 + Robotiq 2F-85，10任务，36种相机配置）：
  - 规范设置下：WARP-VLA 80%，π_0.5 和 AnyCamVLA 均为 78%
  - 扰动设置下平均：**WARP-VLA 48.7%**，π_0.5 36.7%（+12.0pp），AnyCamVLA 40.7%（+8.0pp）
- **消融实验**：
  - 去掉 Experts（单 adapter）：平均 74.8% vs 78.3%
  - 去掉 Router（均匀权重）：平均 74.5% vs 78.3%，证明路由的重要性

## 相关工作脉络
1. **VISTA / RoVi-Aug**：通过合成演示扩展视角覆盖，减少多视角数据收集，但依赖策略从增强观测中推断几何关系；WARP-VLA 在特征层直接适配，不依赖数据扩充。
2. **KYC / StereoVLA**：显式编码相机参数（Plücker 射线、立体线索）作为输入条件，需要准确标定；WARP-VLA 无需标定信息，隐式学习视角变换。
3. **AnyCamVLA**：使用 NVS 将测试观测变换回训练视角，但 NVS 模型需额外训练且依赖多视角数据；WARP-VLA 直接在特征空间校准，泛化至未见配置更优。
4. **Spatial Forcing / GeoAware-VLA / G³VLA**：将几何先验注入视觉编码或 cross-view fusion；WARP-VLA 面向冻结策略的特征对齐目标，不修改 VLA 内部结构。
5. **OC-VLA / CamVLA**：预测相机帧动作再转换到机器人帧，简化视觉-动作对应但仍有覆盖局限；WARP-VLA 专注于腕部相机特征校准这一独特挑战。
6. **SpatialVLA / PointVLA / Any3D-VLA**：引入深度/点云表征；WARP-VLA 在无深度感知输入的情况下，仅凭 2D 特征统计实现视角自适应。

## 局限性与未来方向
1. **泛化受限于训练相机分布**：无法补偿任务相关区域完全被遮挡或超出视场的情况；大偏移下出现夹爪过早打开的失败案例。
2. **有限夹爪可见性影响抓取状态推断**：作者指出需更可靠的 grasp-state 估计以提升任务成功率。
3. **未来方向**：① 训练时扩展相机覆盖范围；② 探索时序记忆机制；③ 利用互补相机观测（如外部相机与腕部相机联合）。
4. **模拟→真实 transfer 仍有差距**：LIBERO 平均提升 39.1pp，但真实机器人扰动下仅 48.7%，sim-to-real gap 仍有改进空间。

## 研究启发与可借鉴点
1. **伪规范 anchor 的设计思想**：通过数据集特征平均构造参考 anchor，为特征级适配提供了简洁有效的监督信号，可迁移到其他需要 canonical reference 的场景（如传感器漂移校正）。
2. **特征统计路由替代显式几何建模**：仅用通道均值/标准差和空间一阶矩即可编码视角偏差信息，避免了引入额外网络模块，轻量化且可插拔，适用于资源受限的部署环境。
3. **分组 MoE 路由结构**：将专家按物理扰动因素（平移 vs 倾斜）分组独立归一化，避免无关维度专家竞争，对多因素系统适配设计有参考价值。
4. **LoRA-style 专家参数化**：共享骨干 + 低秩残差的专家特化方案，兼顾参数效率与视角专门化，可推广到其他需要多专家适配的 VLA 子模块。
5. **WARP-VLA 的 plug-and-play 范式**：冻结 backbone，仅训练适配器，保持规范策略部署，该"适配器层"设计模式可复用于其他 VLA 的后处理适配任务（如光照变化、背景杂波）。

## 关键术语表
- **Vision-Language-Action Model (VLA)**：将视觉、语言理解与动作生成统一建模的端到端机器人策略网络，可直接根据图像和语言指令输出机械臂控制命令。
- **Wrist Camera**：安装于机械臂末端执行器上的相机，提供近距离、随动视角的交互区域观测，对精细操作至关重要。
- **Mixture of Experts (MoE)**：由多个专业化"专家"网络和一个"路由器"组成的架构，路由器根据输入动态选择并加权组合各专家输出。
- **Pseudo-canonical Feature Anchor**：通过对训练集所有帧的腕部视觉特征取平均构造的稳定参考特征，作为适配器学习的目标规范表示。
- **Spatial Token Statistics**：从特征张量中提取的通道均值、标准差及空间一阶矩统计量，用于隐式编码相机视角偏离程度。
- **Low-Rank Factorization (LoRA-style)**：将专家参数分解为共享矩阵与低秩残差之和，以少量额外参数实现专家特化。
- **Teacher Mask**：基于已知相机扰动分布生成的软性 supervision mask，用于训练中估计专家责任分配。
- **LIBERO / LIBERO-Plus**：面向 lifelong robot learning 的仿真基准，LIBERO 含 40 任务/4 suite，LIBERO-Plus 提供更深入的鲁棒性评测。

## 可复现要素
- **数据集**：基于 LIBERO（40 任务，2000 演示）构建 view-augmented 数据集（10 种扰动配置）；LIBERO-Plus（10K episodes）。**论文声明开源 wrist viewpoint robustness benchmark**。
- **代码/权重**：论文声明 release "plug-and-play implementation"，具体仓库信息未详述。
- **关键超参**：E=15 专家，rank r=16；共享骨干 8 层 ViT（hidden dim=2048，attention dim=256，4 heads）；路由器为两層 MLP（hidden dim=2048）；训练 10,000 steps，batch size=4096，lr=3×10⁻⁴，Adam optimizer，8× NVIDIA RTX PRO 6000 GPU，约 30 小时。
- **硬件**：仿真环境为 LIBERO/LIBERO-Plus；真机实验使用 xArm6 + Robotiq 2F-85 + RealSense D435（agent）+ D405（wrist）。

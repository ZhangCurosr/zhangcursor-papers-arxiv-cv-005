---
title: "WARP-VLA-Wrist-Camera-Adaptation-for-View-Robust-Policy-Exec"
source: https://arxiv.org/pdf/2610.11508v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-11-01 07:01:31"
---

# 论文速读：WARP-VLA: Wrist-Camera Adaptation for View-Robust Policy Execution in Vision-Language-Action Models

## 一句话总结
本文提出 WARP-VLA，一种面向腕部相机安装扰动的特征级即插即用适配器，通过伪规范锚点与分组 MoE 路由隐式学习视角变换，无需任何相机标定参数即可将冻结的 VLA 策略部署至多样化的真实腕部视角，使 $\pi_{0.5}$ 在 LIBERO 扰动基准上的平均成功率从 39.2% 跃升至 78.3%。

## 研究问题与动机
- **腕部相机部署脆弱性**：VLA 策略严重依赖视觉输入，而真实机器人腕部相机受安装公差、机械振动等影响，微小位姿偏移会显著改变夹爪-物体的精细几何关系，导致策略性能断崖式下降。
- **现有视角鲁棒方法不适配**：主流方法多针对固定外置相机设计，依赖显式相机外参（如 Plücker 射线嵌入）或新视角合成（NVS），对近距离腕部场景的标定误差极为敏感，且往往需要重新微调或额外 3D 模型。
- **连续异构视角难以用单一变换覆盖**：腕部相机扰动同时包含平移与倾斜分量，特征空间呈连续分布，单一共享适配器无法拟合高度异质的视角偏移模式。
- **现场部署缺乏相机先验**：实际跨设备迁移时通常无法获取准确的相机内参/外参，亟需一种纯数据驱动、推理时仅需图像输入的校准自由方案。

## 核心贡献（创新点）
- 构建首个系统性腕部相机视角扰动评测协议，基于 LIBERO 渲染生成多档平移/倾斜配置的训练与测试集，填补了 VLA 腕部鲁棒性基准的空白。与关注静态外置相机的工作不同，本文首次量化了安装偏移对近场交互策略的实际破坏程度。
- 提出 WARP-VLA 特征适配器，利用全局平均得到的伪规范锚点 $\mathbf{A}$ 与 MoE 架构将扰动 token 映射回规范特征空间。与 AnyCamVLA 等依赖显式 3D 合成或相机参数的方法本质不同，本方法完全免标定且仅在潜特征层操作，不修改预训练主干。
- 设计基于空间 token 统计的锚点引导路由机制，通过提取通道均值、标准差与一阶空间矩隐式推断视角偏移量。相比均匀融合或独立训练单一适配器的方案，路由实现了平移与倾斜维度的解耦与平滑插值。
- 验证了跨仿真域与真机的即插即用泛化能力：在 LIBERO-Plus 上零样本迁移达到 56.5% 平均成功率，真实 xArm6 实验中较基线提升 8~12 个百分点，证明特征级适配可有效桥接仿真与物理部署。

## 方法详解
- **问题建模**：输入规范视角 token $\mathbf{X}^c$ 与扰动视角 token $\mathbf{X}^p$，目标函数为 $\hat{\mathbf{X}}^c = f_\phi(\mathbf{X}^p, \mathbf{A})$，其中伪锚点 $\mathbf{A}$ 通过对规范训练数据全帧特征取平均获得，用于稳定参考并抑制场景/任务特异性波动。
- **MoE 特征适配器**：插入图像编码器与动作策略主干之间，保持 token 数量与维度不变。总输出为 $F_{\mathrm{MoE}}(\mathbf{X}^p, \mathbf{A}) = \sum_{e=0}^{E-1} w_e F_e(\mathbf{X}^p, \mathbf{A})$，共享主干为 8 层 ViT（hidden dim 2048，attention dim 256，4 头）。
- **解耦分组路由**：路由器输入 $\mathbf{z} = \Phi_{\mathrm{stat}}(\mathbf{X}^p - \mathbf{A}) \oplus \Phi_{\mathrm{stat}}(\mathbf{A}) \in \mathbb{R}^{8D}$，其中 $\Phi_{\mathrm{stat}}$ 拼接通道均值、标准差与 $x/y$ 一阶空间矩。15 个专家分为平移组（13 个，对应 $(r,\theta)$ 半圆区域）与倾斜组（2 个），每组内独立 softmax 归一化得到权重 $w_e$，实现多物理因子的解耦加权。
- **低秩专家参数化**：专家特异性通过 LoRA-style 残差注入共享投影矩阵：$W_{e,m}^l = W_m^l + A_{e,m}^l B_{e,m}^l$（rank $r=16$）。初始化令 $B_{e,m}^l = 0$，使所有专家起点同质，训练中逐步分化至各自视角区域。
- **训练损失**：$\mathcal{L} = \mathcal{L}_e + \mathcal{L}_g$。$\mathcal{L}_e = \frac{1}{G}\mathbb{E}[\sum_e h_e d_e]$ 为责任加权重建损失，$d_e$ 为单专家 MSE；$\mathcal{L}_g = -\frac{1}{G}\mathbb{E}[\sum_e h_e \log g_e]$ 为路由交叉熵。责任 $h_e$ 由路由概率、教师软掩码（基于已知扰动先验的重叠 pose bins）与重构误差共同决定。推理时仅依赖 $\mathbf{z}$，无需任何相机参数或策略微调。

## 实验与结果
- **数据集与基线**：LIBERO（40 任务，2000 演示，10 种扰动配置）与 LIBERO-Plus（跨域迁移）。基线涵盖骨干模型（$\pi_{0.5}$、GR00T N1.7）、几何感知（Spatial Forcing、GAM）与相机感知（KYC、AnyCamVLA）。
- **仿真主要结果**：$\pi_{0.5}$ 原始平均成功率 39.2%，加入 WARP-VLA 后达 **78.3%**（+39.1pp）；大扰动（6 cm / 9°）下从 6.1% 提升至 **58.8%**（+52.

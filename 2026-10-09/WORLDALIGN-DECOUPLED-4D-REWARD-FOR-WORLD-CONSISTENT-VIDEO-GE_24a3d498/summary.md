---
title: "WORLDALIGN-DECOUPLED-4D-REWARD-FOR-WORLD-CONSISTENT-VIDEO-GE"
source: https://arxiv.org/pdf/2610.12382v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-10-09 17:30:04"
---

# 论文速读：WORLDALIGN-DECOUPLED-4D-REWARD-FOR-WORLD-CONSISTENT-VIDEO-GE

## 一句话总结
本文提出 WorldAlign，一种解耦的 4D 奖励框架，通过语义分割将生成视频中的静态背景与动态主体分离，并分别利用几何世界先验（掩码重投影）和视觉语言模型评审器（VLM-as-a-Judge）提供奖励信号，在不修改生成器架构且无需人类偏好标注的情况下，在线 GRPO 后训练同步提升视频的静态几何一致性与动态主体一致性。

## 研究问题与动机
1. 现有大规模视频生成模型的预训练目标主要优化 RGB/latent 空间预测，缺乏显式 4D 世界一致性约束，导致生成视频中静态结构易滑移/形变、动态主体运动不合理或外观漂移。
2. 主流的几何感知后训练方法（如 Epipolar-DPO、VideoGPA）强依赖“静态场景”假设，在含移动主体的视频中会将合法主体运动误判为几何错误并施加惩罚，甚至诱导生成器产出冻结画面。
3. 支持动态场景的方法（如 VGGRPO、GeoFlow）依赖 scene flow 或 optical flow 线索剔除动态区域，但运动线索也会将静态区域的意外滑动/形变误识别为动态，削弱应被惩罚的几何错误反馈；且对动态主体的一致性缺乏专项评估。
4. 亟需一种能可靠区分静/动区域、分别提供高质量一致性反馈、且不依赖人工偏好标注的后训练奖励机制。

## 核心贡献（创新点）
1. **解耦的 4D 奖励框架**：首次基于语义跟踪将视频拆分为静态背景与动态主体，分别匹配适配其物理假设的世界先验，彻底摒弃全场景静态假设。
2. **掩码重投影静态奖励**：利用几何基础模型（GFM）在剔除动态主体轨迹的静态点云上计算跨视图重投影误差，并引入辅助相机运动奖励防止“冻结画面”式的奖励作弊。
3. **样本专属 VLM 检查清单动态奖励**：针对每个 prompt/image 对自动生成覆盖动力学、物理合理性、形状与纹理一致性的二进制检查清单，由强 VLM 逐题评审，实现细粒度动态一致性反馈。
4. **免人工偏好的在线后训练**：将三项奖励组内归一化后聚合为优势函数，直接代入 clipped GRPO 更新生成器，无需构建 preference pair，同步改善静态与动态一致性且保持整体画质与运动幅度。

## 方法详解
- **静动语义解耦**：以条件图像中的 subject mask 初始化 segmentation tracker（SAM3），沿视频帧传播得到动态掩码 $M_n$，补集 $\overline{M}_n$ 即为静态区域。
- **静态世界先验对齐（Masked Reprojection Reward）**：
  - 调用 GFM（VGGT-1B）预测每帧像素对齐的 3D 点图 $P_n$ 与相机内外参 $(K_n, C_n)$。
  - 仅保留静态掩码对应的 3D 点构建共享点云：$\mathcal{P}_{\mathrm{static}} = \{ P_n(x) \mid x \in \overline{M}_n \}$，避免动态主体轨迹污染点云。
  - 将 $\mathcal{P}_{\mathrm{static}}$ 按预测相机重投影至各帧，仅在有效静态像素集 $\Omega$ 上计算重投影误差 $E_{\mathrm{rep}}$，奖励 $R_{\mathrm{static}} = -E_{\mathrm{rep}}$。
  - **相机运动奖励**：定义相机运动幅度 $m(C_{1:N})$ 为相邻帧平移范数与旋转角均值，施加 hinge 惩罚 $R_{\mathrm{cam}} = -\max(0, \tau - m)$（阈值 $\tau=0.01$），防止生成器通过冻结相机/画面投机降低几何误差。
- **动态世界先验对齐（VLM-as-a-Judge Reward）**：
  - 训练前由 VLM（Gemini-3.1-Pro-Preview）根据 $(I^c, y)$ 生成样本专属二进制清单 $\mathcal{Q} = \{q_j\}_{j=1}^J$，覆盖 Dynamicity、Physical plausibility、Shape、Texture 四维度。
  - 训练时 VLM（

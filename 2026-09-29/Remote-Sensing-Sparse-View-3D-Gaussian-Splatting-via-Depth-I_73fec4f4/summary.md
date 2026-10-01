---
title: "Remote-Sensing-Sparse-View-3D-Gaussian-Splatting-via-Depth-I"
source: https://arxiv.org/pdf/2609.35612v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-10-01 03:44:52"
field: "稀疏视图新视角合成与遥感三维重建"
keywords: ["Sparse-view novel view synthesis", "3D Gaussian Splatting", "Depth image-based rendering", "Remote sensing", "Neural rendering"]
innovations: ["渐进式DIBR伪视图监督策略，为未观测区域提供几何与外观一致性约束", "高度约束锚点生长策略，抑制近天底场景高斯元向相机方向的异常膨胀"]
benchmarks: ["LEVIR-NVS"]
---

# 论文速读：Remote-Sensing-Sparse-View-3D-Gaussian-Splatting-via-Depth-I

## 一句话总结
本文提出 DIBR-GS，一种结合深度图像渲染（DIBR）与神经 3D 高斯溅射的远程感知稀疏视图新视角合成框架，通过几何初始化、跨视图外观先验和渐进式伪视图监督，在仅 3 个输入视图条件下实现高质量、高一致性的场景重建。

## 研究问题与动机
- **核心问题**：遥感场景受卫星重访周期、无人机飞行路径和遮挡限制，通常仅有 3~5 个稀疏观测视角，缺乏足够的多视角几何约束，导致新视角合成出现几何歧义、结构不完整和外观不一致。
- **现有 NeRF/3DGS 方法的不足**：
  - NeRF 类方法依赖正则化或深度监督，但这些策略主要针对物体中心场景设计，在遥感复杂地物结构和受限俯视角下效果有限。
  - 3DGS 类方法引入深度先验或正则化，但单目深度存在尺度歧义，启发式优化难以捕捉遥感场景的复杂结构。
  - 已有方法缺乏有效机制联合利用跨视图几何与外观线索，未观测区域缺乏充分监督。

## 核心贡献（创新点）
1. **提出 DIBR-GS 框架**：将深度图像渲染与神经 3D 高斯溅射统一，有效利用几何与外观先验，区别于传统仅依赖输入视图的监督方式。
2. **密集深度先验的几何初始化**：通过仿射变换对齐单目深度与稀疏 SfM 重建，结合多视图融合生成可靠初始锚点，解决了单目深度尺度歧义与结构失真问题。
3. **IBR 跨视图特征自适应融合**：基于图像渲染范式提取跨视图外观特征，并通过调制网络自适应控制融合权重，缓解稀疏视图下特征匹配不可靠的噪声影响。
4. **渐进式 DIBR 伪视图监督策略**：从观测视角向未观测视角传播深度先验，生成伪视图提供额外的多视角一致性约束，有效缓解弱观测区域的空洞与结构退化。
5. **高度约束锚点生长策略**：利用遥感场景的近天底视角先验，构建高度图软约束，抑制高斯元向相机方向的异常膨胀，减少漂浮伪影。

## 方法详解
**整体架构**：双分支框架——DIBR 分支负责生成伪视图与几何初始化，神经高斯分支负责场景表示与渲染优化。

**几何初始化**：
- 使用冻结的 Depth Anything v2 提取单目深度图 $\mathbf{D} = F_\theta(\mathbf{I})$，通过仿射变换 $\hat{\mathbf{D}} = a \cdot \mathbf{D} + b$ 与 COLMAP 估计的稀疏 SfM 点云对齐（最小二乘求解 $a, b$）。
- 多视图深度反投影到世界坐标系：$\mathbf{P} = \mathbf{R}_i(\hat{\mathbf{D}}_i(u,v)\mathbf{K}_i^{-1}\mathbf{p}) + \mathbf{t}_i$。
- 重叠点云去除：在 XY 平面网格内对高度分布进行 Dip 检验（显著性水平 < 0.05 视为多峰），利用 KD-tree 识别水平距离 < $\varepsilon_{xy}$ 且垂直距离 > $\varepsilon_z$ 的重叠点对，保留较低高度点作为初始化锚点。
- 体素下采样（voxel size = 1.0）后与 SfM 点云合并。

**IBR 特征提取与自适应融合**：
- 使用冻结的 DINO 预训练 ResNet-50（前 4 层）和 ViT-B/8 提取局部特征金字塔 $\{\mathbf{F}_r^\ell\}$ 和全局 token $\mathbf{F}_t$。
- 通过可学习投影模块 $\mathcal{P}$ 映射到统一维度，上采样后相加得到参考特征图 $\mathbf{F}_v$（维度 $C_r = 32$）。
- 锚点在世界坐标 $\boldsymbol{\mu}_w$ 投影到各视图 $\mathbf{p}_i$，双线性采样获取特征向量 $\mathbf{f}_{v,i} = \Psi(\mathbf{F}_v, \mathbf{p}_i)$，聚合得到 $\mathbf{f}_v$。
- 调制网络 $\mathcal{E}_{\mathrm{mod}}$ 预测通道级调制权重 $\mathbf{g}$，融合特征 $\hat{\mathbf{f}}_s = \mathbf{f}_s + \mathbf{g} \odot \mathcal{E}_{\mathrm{ref}}(\mathbf{f}_v)$。

**高度约束锚点生长**：
- 从初始密集点云构建高度图 $\mathcal{H}(x,y)$：将 XY 平面划分为网格（size = 10），对每格高度取 95 分位数并乘以缩放因子得到软上限。
- 锚点生长条件：$\mathbb{I}_{\mathrm{grow}}(P) = 1$ 当且仅当 $\nabla_g > \tau_g$ 且 $z \leq \mathcal{H}(x,y)$，否则不生长。

**渐进式 DIBR 伪视图合成**：
- 在相机轨迹上采样中间视角，从密集点云进行深度补全得到伪深度 $\mathbf{D}_j^{\mathrm{pse}}$。
- 深度战争：$\mathbf{p}_{j\to i} = \mathbf{K}_i \mathbf{T}_{ji} \mathbf{D}_j^{\mathrm{pse}}(u,v) \mathbf{K}_j^{-1} \mathbf{p}_j$。
- 按目标视角与各源视图的距离排序，由近及远逐步填充伪视图，未覆盖区域用 padding 策略补全。
- 训练时采用调度策略：前期仅用 GT 图像，中期每 50 次迭代引入伪视图监督，后期关闭伪视图监督聚焦细节优化。

**优化目标**：
$$\mathcal{L} = (1 - \lambda)\mathcal{L}_1(\hat{\mathbf{I}}, \mathbf{I}) + \lambda\mathcal{L}_{\mathrm{D-SSIM}}(\hat{\mathbf{I}}, \mathbf{I}), \quad \lambda = 0.2$$

## 实验与结果
- **数据集**：LEVIR-NVS（16 个遥感场景，每场景 21 张 $512 \times 512$ 图像），按 TriDF 协议均匀选取 3 张训练、其余测试。
- **评估指标**：PSNR、SSIM、LPIPS、AVGE。
- **基线方法**：RegNeRF、FreeNeRF、MPNeRF、TriDF、3DGS、FSGS、CoR-GS、DropGaussian。
- **主要结果**（Table I）：
  - Ours：PSNR **30.90**，SSIM **0.938**，LPIPS **0.076**，AVGE **0.025**，FPS 177。
  - 相比最佳基线 TriDF（PSNR 24.07）提升 **6.83 dB**，SSIM 相对提升约 14%，LPIPS 相对提升约 60%。
- **逐场景结果**（Table II）：在全部 16 个场景中均取得最优或接近最优的 PSNR 与 SSIM，验证方法稳定性。
- **消融实验**（Table III）：移除伪视图监督导致 PSNR 骤降至 21.06，证明其对弱观测区域重建至关重要；移除密集初始化使 PSNR 降至 29.18；移除 IBR 特征使 PSNR 降至 30.56。

## 相关工作脉络
1. **3DGS 稀疏视图方法**（FSGS、DNGaussian、CoR-GS、DropGaussian）：这些方法主要通过深度正则化或结构剪枝缓解稀疏视角过拟合，但未解决遥感场景单目深度尺度歧义和未观测区域监督缺失问题。
2. **NeRF 稀疏视图方法**（RegNeRF、FreeNeRF、D-NeRF）：引入频率正则化或深度监督，但依赖密集采样的 per-ray 推理效率低，且正则化策略在遥感复杂地物结构下效果受限。
3. **遥感专用方法**（MPNeRF、TriDF）：针对遥感场景设计，但 MPNeRF 依赖多平面先验，TriDF 使用三平面加速密度场，两者均未充分利用 DIBR 生成伪视图进行跨视角一致性约束。
4. **神经场景表示**（Scaffold-GS、2DGS）：Scaffold-GS 引入锚点分层结构组织高斯元，本文在其基础上进一步结合深度先验与 IBR 特征；2DGS 将高斯约束到物体表面，更适合几何精度要求高的场景。
5. **基于图像的渲染**（IBRNet）：IBRNet 学习多视图图像的隐式表示，本文借鉴其跨视图特征聚合思想，但将其与显式 3D 高斯表示结合并引入 DIBR 伪视图监督。

## 局限性与未来方向
- **逐场景优化**：当前方法依赖 per-scene optimization，计算时间随场景复杂度增长，泛化能力受限。
- **单目深度误差累积**：伪视图合成基于单目深度先验，存在尺度歧义和结构扭曲，影响伪视图质量（文中报告伪视图 PSNR 仅 15.82）。
- **近天底场景假设**：高度约束策略针对近天底遥感视角设计，对大斜视角或复杂地形可能失效。
- **未来方向**：探索前馈式（feed-forward）高效通用方法，提升推理速度与跨场景泛化能力。

## 研究启发与可借鉴点
1. **DIBR 伪视图监督思路**：将深度图像渲染从"生成完整图像"转化为"提供几何与外观一致性约束"，为稀疏视角重建中的未观测区域监督问题提供了新思路。
2. **高度约束正则化策略**：针对特定场景先验（近天底视角的高度-距离负相关）设计软约束，比纯数据驱动的 depth regularization 更具物理可解释性，可迁移至其他有几何先验的场景。
3. **IBR 特征自适应调制机制**：通过调制网络控制跨视图特征的贡献度，有效处理稀疏视图下匹配不可靠的问题，该设计可与多种神经渲染方法结合。
4. **密集初始化与冗余去除**：通过 Dip 检验和 KD-tree 识别重叠点云并保留较低高度点，兼顾了几何完整性与优化稳定性，为其他 3D 重建任务提供了初始化参考方案。

## 关键术语表
- **3D Gaussian Splatting (3DGS)**：一种基于显式各向异性高斯原语的神经渲染方法，通过可微分栅格化实现实时高质量新视角合成。
- **Depth Image-Based Rendering (DIBR)**：基于深度图像的渲染技术，利用深度信息将源视图像素重投影到目标视图以合成新视角。
- **Neural Gaussian**：基于锚点的神经高斯表示，每个锚点控制多个高斯原语，通过轻量 MLP 解码旋转、缩放、不透明度和颜色等属性。
- **IBR (Image-Based Rendering)**：基于图像的渲染范式，从多视图参考图像中提取特征并聚合到场景表示中，补充几何信息。
- **Adaptive Feature Fusion**：通过调制网络自适应预测通道级权重，控制跨视图参考特征与锚点特征的融合比例，抑制不可靠特征干扰。
- **Height-constrained Anchor Growth**：利用遥感场景高度先验构建软约束，仅在合理高度范围内允许高斯锚点生长，抑制漂浮伪影。
- **Progressive Pseudo-view Synthesis**：按源视图与目标视角的距离由近及远逐步填充伪视图，优先使用可靠性更高的近邻视图对应关系。
- **Dip Test**：检验数据分布单峰性的统计方法，用于识别点云中因深度重叠产生的多峰高度分布。

## 可复现要素
- **数据集**：LEVIR-NVS，公开可用（来源于 Google Earth）。
- **代码**：已开源，GitHub: https://github.com/kanehub/DIBR-GS
- **关键超参数**：
  - 训练迭代：30,000 次
  - $\lambda = 0.2$（L1 与 D-SSIM 损失权重）
  - 体素下采样 size = 1.0
  - 锚点 voxel size = 0.03，每锚点 $k = 10$ 个高斯
  - 特征维度 $C_f = 32$，$C_r = 32$
  - XY 网格高度图划分 size = 10
  - 重叠点去除阈值：$\varepsilon_{xy} = 0.1$，$\varepsilon_z = 1.0$
  - Dip 检验显著性水平：0.05
  - 伪视图监督引入：迭代 3,000–20,000，每 50 次迭代
  - 锚点加密：迭代 1,500–15,000，每 100 次
- **硬件**：单卡 Nvidia GeForce RTX 4090
- **预训练模型**：Depth Anything v2（冻结）、DINO 预训练 ResNet-50 与 ViT-B/8（冻结特征提取）

---
title: "When-Integral-Meets-Decomposition-A-Signal-Level-Self-Superv"
source: https://arxiv.org/pdf/2609.39004v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-10-01 21:33:23"
field: "多模态图像融合"
keywords: ["多模态图像融合", "特征分解", "自监督学习", "信号级建模", "积分优化", "红外可见光融合", "医学图像融合"]
innovations: ["提出1D信号级自监督特征分解范式，将模糊的2D图像级监督重构为积分驱动的显式优化目标", "设计互补的双预置任务（信号级分解+图像级重建）实现跨尺度监督", "通过DWT频率分解显式解耦共同特征与模态独特特征"]
benchmarks: ["M³FD", "TNO", "MSRS", "MRI-CT", "MRI-PET", "MRI-SPECT", "LYTRO", "MFFW"]
---

# 论文速读：When-Integral-Meets-Decomposition-A-Signal-Level-Self-Supervised-Feature-Decompose-Paradigm-for-Multi-Modal-Image-Fusion

## 一句话总结
本文提出了一种 **1D 信号级自监督特征分解范式**，将多模态图像融合中的特征分解问题从"缺乏明确监督的2D图像级优化"重构为"**积分驱动的1D信号级优化**"，通过双预置任务（信号级分解 + 图像级重建）在无GT条件下实现了更准确的特征解耦，并在 VIF 与 MIF 任务上达到 SOTA。

---

## 研究问题与动机
1. **缺乏 GT 监督**：现有基于特征分解的 MMIF 方法将源图像分解为共同特征 $C$ 和独特特征 $(U_A, U_B)$，但**不存在共同/独特特征的 ground-truth 特征图**，导致分解过程无法获得明确监督。
2. **像素级多属性耦合**：每个像素同时耦合了强度、对比度、边缘、纹理、轮廓、颜色等多种视觉属性，**仅靠单一或少数损失项无法覆盖全部属性**，经验性混合损失天然存在信息不完整的问题。
3. **损失指标间可能冲突**：不同 loss 项（如 $L_1$、SSIM、梯度 loss）描述不同视觉属性，是否存在冲突、何种组合最优均**无先验知识**，训练目标模糊且不稳定。
4. **动机**：能否将特征分解问题从"模糊的2D图像级监督"重新 formulate 为一个**具有明确数学目标的1D信号级优化问题**？

---

## 核心贡献（创新点）
1. **提出 1D 信号级自监督特征分解范式**：将特征分解从 2D 图像空间提升到 1D 信号空间，利用连续积分约束替代多指标混合的模糊监督，使分解目标从"未知"变为"可计算"。与已有方法（DIDFuse、FD-Fuse 等）相比，本质区别在于**从架构设计转向目标重构**。

2. **将特征分解 reformulate 为积分驱动优化问题**：共同信号通过最小化其与两个源信号的积分平方误差来学习，提供了**显式的数学目标**，避免了经验性 metric-mixing 带来的冲突和歧义，这是本文与 SigFusion 的本质差异——SigFusion 的信号级建模旨在数据合成，而本文旨在解决分解监督缺失。

3. **设计互补的双预置任务（Dual Pretext Tasks）**：信号级任务提供严格的数学约束用于精确分解，图像级重建任务确保空间结构保真，两者共同构成**跨尺度的完整监督体系**。

4. **提出两阶段 SSL 框架**：Stage I 完成信号级分解与图像级重建的预训练，Stage II 冻结分解器仅训练融合头，最终实现端到端的无监督特征分解融合。

---

## 方法详解
### 总体框架
模型由四部分组成：编码器 $\zeta$、信号级特征分解器 $SD(\cdot)$、基于 MLP 的融合模块 $\delta$、解码器 $\xi$，前两者与解码器均采用 Restormer Block。整体遵循两阶段自监督学习（SSL）范式。

### Stage I：自监督特征分解（双预置任务）
1. **特征提取与信号化**：
   - 通过编码器 $\zeta$ 提取源图像 $I_A, I_B$ 的高层特征 $F_A, F_B$。
   - 经 $1\times1$ 卷积 + Flatten 投影为 1D 信号 $s_A(l), s_B(l)$，$l$ 对应空间位置的展平索引。

2. **1D 小波分解**：
   - 对每个信号进行 1D DWT：$L_m(l), \{H_m^i(l)\}_{i=1}^N = \mathrm{DWT}(s_m(l))$，其中 $m\in\{A,B\}$。
   - **低频分量**编码跨模态共享的全局结构，**高频分量**捕获模态特有的细节（纹理、边缘）。

3. **信号级分解**：
   - **共同路径**：输入低频对 $(L_A(l), L_B(l))$，估计共同信号 $c_s(l) = SD(L_A(l), L_B(l))$。
   - **独特路径**：输入高频子带，估计独特信号 $u_{s^A}(l), u_{s^B}(l)$。

4. **信号级损失 $\mathcal{L}_{\mathrm{dec}}$（核心创新）**：
   $$\mathcal{L}_{\mathrm{dec}} = \int (c_s(l) - s_A(l))^2 dl + \int (c_s(l) - s_B(l))^2 dl$$
   即共同信号与两个源信号之间的**积分面积**，以源信号本身作为 pseudo label，提供全局一致性约束。

5. **图像级重建**：
   - 通过 IDWT 将 $c_s(l)$ 与 $u_{s^m}(l)$ 重构回 1D 信号 $\hat{s}_m(l)$，再经逆投影 $\mathcal{H}^T$ 与解码器 $\xi$ 重建图像 $\hat{I}_m$。
   - 重建损失：$\mathcal{L}_{\mathrm{rec}} = \mathrm{MSE}(\hat{I}_A, I_A) + \mathrm{MSE}(\hat{I}_B, I_B)$，以源图像为 pseudo label 约束空间结构。

6. **Stage I 总损失**：$\mathcal{L}_{\mathrm{stage1}} = \alpha \mathcal{L}_{\mathrm{dec}} + \beta \mathcal{L}_{\mathrm{rec}}$，其中 $\alpha=1, \beta=2$。

### Stage II：特征融合
- **冻结** $\zeta$ 和 $SD$，仅训练融合模块 $\delta$ 和解码器 $\xi$。
- 将独特信号 $u_{s^A}(l), u_{s^B}(l)$ 经 MLP $\delta$ 融合为 $u_f(l)$，再与共同信号 $c_s(l)$ 经 IDWT 合并得到融合信号 $f_s(l)$，最终解码为 $I_f$。
- Stage II 损失：$\mathcal{L}_{\mathrm{stage2}} = \gamma \mathcal{L}_{\mathrm{SSIM}} + \lambda \mathcal{L}_{L1} + \omega \mathcal{L}_{\mathrm{Grad}}$，权重 $\gamma=1, \lambda=1, \omega=15.6$。

---

## 实验与结果
### 数据集
- **VIF（可见光-红外融合）**：训练集 MSRS（400对）；测试集 M³FD（100对）、TNO（25对）、MSRS（361对）。
- **MIF（医学图像融合）**：训练集来自 Harvard Medical Website 的 453 对注册医学图像；测试集 MRI–CT（21对）、MRI–PET（42对）、MRI–SPECT（73对）。
- **多焦点融合（MFIF，扩展实验）**：LYTRO 与 MFFW 数据集。

### 评估基线
VIF 基线：CDDFuse、LRRNet、EMMA、TC-MoA、Text-Difuse、DCEvo、SAGE、TD-Fusion、Omni-Fuse、C2RF、SigFusion。
MIF 基线：除上述外增加 CCF、BSA-Fusion、Mask-Difuser、MTG-Fusion。

### 主要结果
- **VIF 任务**（Table 1）：在 M³FD、MSRS、TNO 三个数据集上，本文方法在 QMI、QNICE、QP、QCB、MI、VIFp、QY 等多项指标上均取得最高分。例如在 M³FD 数据集上 QMI=0.8405（第二名为 SigFusion 的 0.8034），提升约 **3.7%**；在 MSRS 上 MI=5.4547（第二名为 SigFusion 的 4.9123），提升约 **5.43** 绝对值。
- **MIF 任务**：在 MRI-CT、MRI-PET、MRI-SPECT 三个子集上，本文方法在多项指标上均位列第一，QMI 最高达 0.9369（MRI-CT），对比 SigFusion 的 0.9239 提升约 **1.3%**。
- **消融实验**（Table 2）：移除信号级分解器导致 VIF 任务 QMI 从 0.6608 降至 0.6282（**-4.9%**）；移除 DWT 导致 QMI 从 0.6608 降至 0.6425（**-2.8%**）；双预置任务各自移除均有明显下降，验证了各组件有效性。
- **下游任务**：在 YOLOv12 目标检测中，本文方法的 mAP=0.656，优于 CDDFuse（0.636）、DCEvo（0.655）等；在 UniverSeg 医学分割中，Dice 达 0.856，优于 MTG-Fusion（0.846）。

### 最强结果与提升幅度
- VIF M³FD：QMI=**0.8405**（SOTA，较次优提升 ~3.7%）；VIF MSRS：MI=**5.4547**（较次优 SigFusion 的 4.9123 提升显著）。
- MIF MRI-CT：QY=**0.9439**（SOTA）；MIF MRI-PET：QMI=**0.8843**（SOTA）。

---

## 相关工作脉络
1. **DIDFuse**（arXiv 2020）：首次提出端到端的特征分解模块，显式分离共同与独特特征，但同样面临无 GT 监督的问题，依赖经验性 loss。本文在其基础上从**监督目标本身**入手重构问题。
2. **FD-Fuse**（TIIM 2025）：通过预置任务间接约束分解过程，未解决信号级监督缺失的本质问题。本文提出**显式积分约束**替代隐式间接约束。
3. **CU-Net**（TPAMI 2020）：将特征分解形式化为稀疏优化问题，依赖手工设计的 objective。本文从**学习式积分优化**出发，无需手工设计。
4. **SigFusion**（AAAI 26）：同样采用信号级建模，但目标是通过信号分布建模合成大规模训练数据以缓解数据稀缺。本文的信号级建模旨在**为分解本身提供明确监督**，二者出发点不同。
5. **DeFusion**（ECCV 2022）：利用自监督掩码图像修复学习跨模态语义关系，但未针对特征分解设计专门监督。本文专注于**分解问题的监督重构**。
6. **C2RF**（IJCV 2025）：通过对比学习挖掘共性，但未解决低频/高频信息解耦的监督模糊性。本文通过 DWT 频率分解+积分约束**显式建模**共性与独特性。

---

## 局限性与未来方向
1. **依赖预配准图像对**：当前方法假设输入已正确配准，这在真实场景中可能受限，与主流方法相同。
2. **计算效率随图像尺寸增长**：1D 信号的扁平化长度与像素数成正比，大分辨率图像处理速度较慢，作者在 Appendix I 中明确承认此局限。
3. **未来方向**：探索更高效的 1D 信号处理方式以支持高分辨率输入；尝试将信号级范式扩展到非配准场景下的无监督融合。

---

## 研究启发与可借鉴点
1. **"问题重构"思路**：当监督信号缺失时，可将问题从原空间（2D 图像）映射到新空间（1D 信号），利用目标空间中的自然数学约束构建**显式学习目标**。这一思路可迁移到其他缺乏 GT 的视觉任务（如去噪、超分中的分解子任务）。
2. **双预置任务设计范式**：信号级提供精确的数学约束，图像级保证结构保真，两者互补。这种"跨尺度双任务"设计值得在其他自监督框架中复用。
3. **频率域先验的利用**：通过 DWT 将信号分解为低频（共性）和高频（独特性）子带，为特征解耦提供了**可学习的物理先验**，该策略可与其它频域分析方法（如 FFT、Wavelet packet）结合。
4. **可扩展至多焦点融合等更广泛任务**：论文已在 MFIF 任务上验证了范式泛化能力，提示团队可探索其在影像配准、图像修复等任务中的应用潜力。

---

## 关键术语表
**Multimodal Image Fusion (MMIF)**：将来自不同模态（如红外、可见光、CT、PET）的图像信息整合到单一高质量输出图像中的任务。

**Feature Decomposition**：将源图像分解为共同特征（common，跨模态共享的场景结构）和独特特征（unique，模态特有的细节信息）的过程。

**Self-Supervised Learning (SSL)**：从数据本身构造监督信号（pretext task）进行学习，无需人工标注的方法范式。

**1D Signal-level Representation**：将 2D 特征图通过卷积+展平映射为一维序列，用于构建积分优化的信号表示。

**Discrete Wavelet Transform (DWT)**：将信号分解为低频（近似）和高频（细节）子带的多分辨率分析工具，此处用于 1D 信号的频域分解。

**Integral-driven Loss**：以共同信号与源信号之间的积分平方误差作为优化目标，提供全局一致的分解监督。

**Dual Pretext Tasks**：同时设计信号级分解任务和图像级重建任务，分别在信号域和图像域提供互补监督。

**Two-stage SSL Framework**：第一阶段预训练分解器（信号级+图像级），第二阶段冻结分解器仅训练融合头的分阶段学习范式。

---

## 可复现要素
- **代码开源**：已开源，GitHub 地址为 `github.com/Wangjiayu0512/SIDFusion`。
- **数据集**：VIF 使用 MSRS、M³FD、TNO（公开）；MIF 使用 Harvard Medical Website 的 MRI-CT/PET/SPECT 配对数据（公开）；MFIF 使用 LYTRO 和 MFFW（公开）。
- **关键超参**：Stage I 中 $\alpha=1, \beta=2$，训练 200 个 epoch；Stage II 中 $\gamma=1, \lambda=1, \omega=15.6$，训练 200 个 epoch；AdamW 优化器，初始学习率 $3\times10^{-4}$，每 20 epoch 减半。
- **硬件环境**：Ubuntu 22.04，Intel i9-14900K CPU，RTX 4090 GPU，64GB RAM，PyTorch 2.3.0。

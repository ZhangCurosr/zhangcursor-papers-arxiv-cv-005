---
title: "Spatial-Frequency-Aware-Implicit-Neural-Representation-of"
source: https://arxiv.org/pdf/2610.11296v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-11-01 00:46:40"
field: "隐式神经表示与频域建模"
keywords: ["隐式神经表示", "Kolmogorov-Arnold Networks", "频谱偏差", "小波变换", "多维信号重建"]
innovations: ["MLP-KAN双分支异构架构实现互补频率建模", "小波域加法融合与带分离正则化协同机制"]
benchmarks: ["DIV2K", "HCI 4D LF Benchmark", "Stanford SDF", "MSU Video Compression Benchmark"]
---

# 论文速读：Spatial-Frequency-Aware-Implicit-Neural-Representation-of

## 一句话总结
论文提出一种空间-频率感知的隐式神经表示（INR）框架，通过MLP-KAN双分支架构结合小波域加法融合与带分离正则化，缓解MLP的频谱偏差问题，显著提升多维权信号重建保真度。

## 研究问题与动机
- MLP-based INR存在固有的频谱偏差（spectral bias），偏好低频成分，难以捕捉高频细节。
- 现有缓解方法（如Fourier features、SIREN、Hybrid explicit-implicit）或依赖人工调参，或引入额外存储开销，未能充分利用MLP的低频归纳偏置。
- 单纯小波分解无法自动引导不同分支的频率专业化，需要额外的谱约束机制。
- 如何协调异构网络结构以实现互补的频率建模行为，同时避免硬频率分配导致的精度下降。

## 核心贡献（创新点）
- **MLP-KAN双分支异构架构**：MLP分支负责低频平滑结构，KAN分支补充局部变化和精细细节，二者互补建模而非硬分配频段。
- **小波域加法融合机制**：对两分支输出进行DWT分解，对应小波系数加权相加后通过IDWT重建，保留双分支对最终重建的贡献。
- **带分离正则化（Band-Separation Regularization）**：抑制MLP分支的高频响应和KAN分支的低频响应，鼓励互补的频率行为，不强制独占子带分配。
- **跨维度通用性验证**：在1D信号、2D图像、3D体积/SDF、视频、4D光场等多样化任务上系统评估，证明框架的广泛适用性。

## 方法详解
- **双分支架构**：输入坐标x分别送入MLP和KAN，MLP输出$F_M$为低频导向表示，KAN输出$F_K$提供局部细节响应。
- **小波分解与融合**：对两分支输出进行J层1D Haar（db1）小波变换，沿采样坐标轴线操作；对应子带系数（LL近似带、LH/HL/HH细节带）进行元素级加法融合$\tilde{Y}_b^F = Y_b^M + Y_b^K$。
- **带分离损失**：$\mathcal{L}_{\text{band}} = \alpha \sum_{b \in B_H} \frac{1}{N_b}\|Y_b^M\|_F^2 + (1-\alpha)\frac{1}{N_{LL}}\|Y_{LL}^K\|_F^2$，惩罚MLP的高频响应和KAN的低频响应，$\alpha=0.5$平衡两项。
- **整体目标函数**：$\mathcal{L} = \mathcal{L}_{\text{recon}} + \lambda_{\text{band}}\mathcal{L}_{\text{band}}$，其中$\lambda_{\text{band}}=10^{-3}$；逆问题可加入数据保真项和TV正则项。
- **多维权推广**：沿各坐标轴独立进行1D小波变换，近似系数递归分解，所有细节系数保留。

## 实验与结果
- **数据集**：DIV2K 700张图像（512×512）、Head/Standard 3D体积、Stanford SDF数据集、MSU视频序列、HCI 4D光场基准。
- **2D图像**：PSNR 45.72 dB，超越第二名FINER（39.36 dB）6.36 dB；SSIM 0.9912，LPIPS 0.0030。
- **SDF重建**：Sphere/Armadillo/Dragon三个模型均取得最低Chamfer距离和最高IoU。
- **视频**：BALL/COLORS/PARALLAX三序列中九项指标获八项最优（仅PARALLAX SSIM次优）。
- **4D光场**：PSNR 29.06 dB、SSIM 0.8102，超越次优MLP-PE分别+0.17 dB和+0.0534。
- **消融验证**：去除带分离损失后性能显著下降；固定路由（Hard Frequency Assignment）虽强化频率分离但精度大幅降低（PSNR仅19.21 dB vs 44.60 dB）。
- **参数量控制实验**：将对比方法调整至相近参数规模（~4.13M）后，本方法仍保持领先，证明改进非单纯容量增长所致。

## 相关工作脉络
- **MLP-based INR（如 InstantNGP、DINER）**：采用可训练嵌入增强高频建模，但引入存储开销；本文不依赖显式嵌入，通过异构分支协同解决频谱偏差。
- **Fourier Feature/SIREN类方法**：修改坐标编码或激活函数（正弦/高斯/小波）以突破频谱偏差；本文保留MLP低频频偏并补充KAN分支，从网络结构层面实现互补。
- **Wavelet-based INR（如 WIRE）**：将小波基纳入坐标编码；本文在小波域对分支输出直接融合并施加谱正则，关注分支间频率协调而非单支编码增强。
- **KAN基础工作**：KAN通过可学习单变量函数提供局部灵活性；本文首次将KAN与MLP组成双分支用于INR，并设计小波域协调机制。
- **WaveKAN**：将自适应小波分解与KAN结合用于时间序列预测；本文面向多维信号隐式表示，核心差异在于双分支异构协同与带分离正则。
- **FINER/MFN/FR**：通过可变周期激活、乘性滤波、傅里叶重参数化改进高频拟合；本文从网络异构性角度切入，不修改单分支内部结构。

## 局限性与未来方向
- **计算效率权衡**：双分支架构参数量较大（4.13M），在高维任务（3D/4D）收敛时间明显高于单分支基线。
- **超参数敏感性**：KAN架构对网格大小、B-spline次数敏感；MLP分支相对稳定，但最优配置需针对信号维度调整。
- **硬性频率分配不可行**：实验表明固定路由策略导致精度大幅下降，当前加法融合+正则方案为更优平衡。
- **未来方向**：轻量级异构分支设计、自适应频域协调机制、更高效CUDA实现以改善精度-效率权衡。

## 研究启发与可借鉴点
- **异构分支协同范式**：MLP-KAN双分支设计为解决INR频谱偏差提供了新思路，可迁移至其他坐标网络任务（如NeRF、隐式表面重建）。
- **小波域正则化机制**：带分离损失作为软频率约束，避免硬分配损失灵活性，该思想可推广至其他双路径融合架构。
- **频域分析驱动消融**：通过分支小波能量比例定量分析频率倾向，为模型解释和调参提供可量化依据。
- **跨维度统一框架**：同一套机制适配1D至4D信号，证明了框架的泛化能力，可作为多维信号表示的通用baseline。
- **参数量控制实验设计**：通过调整对比方法参数规模排除容量优势混淆，该实验策略值得借鉴于架构对比研究。

## 关键术语表
- **隐式神经表示（INR）**：将连续坐标映射到信号值的神经函数，替代离散网格表示，支持任意分辨率查询。
- **频谱偏差（Spectral Bias）**：MLP倾向于先学习低频成分、后学习高频成分的归纳偏置现象。
- **Kolmogorov-Arnold Network（KAN）**：用可学习单变量函数替代固定激活函数的网络，边权重为B-spline基函数组合。
- **离散小波变换（DWT）**：将信号分解为多尺度近似系数和细节系数的多分辨率分析工具。
- **带分离正则化（Band-Separation Regularization）**：抑制特定分支高频/低频响应的谱约束损失，促进互补频率建模。
- **加法小波融合**：两分支对应小波系数直接相加后重建，保留双路径贡献而非硬分配频段。
- **固定路由（Fixed Routing）**：强制MLP仅负责近似带、KAN仅负责细节带的硬性频率分配策略。

## 可复现要素
- **数据集**：DIV2K、HCI 4D LF Benchmark公开；1D音频/方波合成数据；3D体积（Head/Standard）、SDF（Stanford 3D Scanning Repository）、视频（MSU VCR Benchmark）部分公开。
- **代码**：论文未明确声明开源状态；1D实验使用原始KAN代码库，2D-4D实验使用自定义CUDA KAN层（PyTorch 2.0.0 + CUDA 11.3）。
- **关键超参**：$\lambda_{\text{band}}=10^{-3}$、$\alpha=0.5$、J=3层小波分解、db1 Haar小波基；KAN参数k=3（B-spline次数）、grid size=64；MLP 3×64，KAN 128-256-128；优化器AdamW，迭代5000/20000次。

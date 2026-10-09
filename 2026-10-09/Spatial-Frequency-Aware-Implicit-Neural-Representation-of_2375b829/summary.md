---
title: "Spatial-Frequency-Aware-Implicit-Neural-Representation-of"
source: https://arxiv.org/pdf/2610.11296v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-10-09 09:46:05"
field: "隐式神经表示与多维信号建模"
keywords: ["隐式神经表示", "Kolmogorov-Arnold Networks", "频谱偏差", "小波变换", "多维信号重建", "MLP-KAN 双分支"]
innovations: ["MLP-KAN 异构双分支实现低频-高频互补表示", "小波域加法系数融合配合带分离正则化引导频域协调", "在 1D-4D 多维度信号上统一验证框架有效性"]
benchmarks: ["DIV2K", "HCI 4D Light Field Benchmark", "MSU Video Compression SR Benchmark", "Stanford 3D Scanning Repository (SDF)", "Head & Standard 3D Volumes"]
---

# 论文速读：Spatial-Frequency-Aware-Implicit-Neural-Representation-of

## 一句话总结
提出一种基于 MLP-KAN 双分支的频域感知隐式神经表示（INR）框架，通过小波域加法融合与带分离正则化，协调 MLP（低频偏置）与 KAN（局部细节响应更强）两支网络实现多维度信号的高质量重构。

## 研究问题与动机
- **MLP-INR 的谱偏差（spectral bias）问题**：MLP 类 INR 天然偏好低频分量，难以拟合高频细节（边缘、纹理、局部突变）。
- **现有缓解方法的局限**：Fourier 特征映射依赖人工调参频率；SIREN 等周期激活需精心设计初始化与网络结构；InstantNGP/DINER 等混合方法引入额外信号相关存储开销。
- **已有工作未从"异构表示互补"角度切入**：多数方法仍围绕 MLP 本身修改输入编码或激活函数，未利用不同网络结构的频率特性差异。
- **KOLMGOROV-ARNOLD NETWORK (KAN) 的潜力**：KAN 用可学习一元函数替代固定激活，在合适配置下表现出较弱的低频偏置，有望作为 MLP 的高频互补分支。

## 核心贡献（创新点）
- **MLP-KAN 异构双分支架构**：MLP 负责平滑全局结构（低频），KAN 负责局部变化与精细细节（相对高频），两者通过异构结构天然形成频率互补。*与先前方法本质区别在于不再改造单一 MLP 的谱特性，而是让两个结构不同的网络各展所长。*
- **小波域加法系数融合机制**：对两支输出分别做 DWT，将对应子带系数逐元素相加后再 IDWT 重建，保留两分支对每个频带的贡献灵活性。*与固定路由（hard subband assignment）不同，不强制硬性频带分配，保持双分支协同能力。*
- **带分离正则化（band-separation regularization）**：引入 L_band 惩罚 MLP 的高频响应和 KAN 的低频响应，在加法融合基础上显式引导两支形成互补频域行为。*这是通过正则化而非结构约束实现软频域协调的关键设计。*
- **系统性多维度验证**：在 1D 信号、2D 图像、3D 体数据/SDF、视频及 4D 光场等多个任务上统一验证，证明框架通用性。

## 方法详解
- **双分支架构**：
  - MLP 分支：L_M 层标准全连接网络，输出 f_{θ_M}(x)，承载低频全局结构表示。
  - KAN 分支：L_K 层，每条边使用可学习一元函数 φ_{ℓ,q,p}(t) = w_{ℓ,q,p}b(t) + Σ c_{ℓ,q,p,r}B_r(t)，其中 B_r 为 B 样条基函数，输出 g_{θ_K}(x)，承载局部细节响应。
- **小波分解与融合**：
  - 在离散坐标网格上同时前向计算 F_M = f_{θ_M}(X) 和 F_K = g_{θ_K}(X)。
  - 对两支输出分别做 J 级 1D Haar（db1）DWT（沿坐标轴逐线处理，非完整 D 维张量）。
  - 各子带（LL、LH、HL、HH）对应系数逐元素相加：Ỹ_b^F = Y_b^M + Y_b^K，b∈{LL, LH, HL, HH}。
  - 融合后通过 IDWT 重建：Î = W_ψ^{-1}(Ỹ^F)。
- **带分离正则化损失**：
  - L_band = α·Σ_{b∈B_H} (1/N_b)||Y_b^M||_F² + (1-α)·(1/N_LL)||Y_LL^K||_F²，其中 B_H={LH,HL,HH}。
  - 总损失：L = L_recon + λ_band·L_band，λ_band=10^{-3}，α=0.5。
  - 目标：压制 MLP 高频响应、压制 KAN 低频响应，促使互补，但不强制独占子带。
- **逆问题扩展**：引入数据保真项 L_data = ||A(Î)-y||² 和各向异性 TV 正则项 R_TV，可用于 CT 重建等逆问题场景。

## 实验与结果
- **数据集**：DIV2K（700 张 HR 图像， resized 512×512）、HCI 4D Light Field Benchmark（4 场景 9×9 子孔径图）、MSU Video Compression SR Benchmark（BALL/COLORS/PARALLAX）、Head & Standard 3D 体数据、Stanford 3D Scanning Repository（Sphere/Armadillo/Dragon SDF）、1D 方波及 audio（bach/counting）。
- **2D 图像（DIV2K）**：PSNR **45.72 dB**（第二优 FINER 39.36 dB，提升 +6.36 dB）；SSIM 0.9912；LPIPS 0.0030，三项均最优。
- **3D 体数据**：Head 与 Standard 体积重建中，本文方法保留更清晰解剖边界、更少伪影。
- **SDF 重建**（Sphere/Armadillo/Dragon）：Chamfer 距离最低、IoU 最高，三项均最优（如 Dragon Chamfer 6.309×10^{-6} vs 次优 BACON 6.564×10^{-6}）。
- **视频重建**：BALL（PSNR 43.77 dB）、COLORS（32.77 dB）、PARALLAX（27.42 dB）三项 PSNR 均最优；LPIPS 三项均最优；PARALLAX 上 SSIM（0.7744）略低于 SIREN（0.8545）。
- **4D 光场**：平均 PSNR **29.06 dB**（次优 MLP-PE 28.89 dB，+0.17 dB）；SSIM **0.8102** 最优；LPIPS 0.3607（次优 MLP-PE 0.2803）。
- **参数控制实验**（Table IX）：将对比方法调至与本文相当参数量（~4.13M）后，本文仍保持 45.72 dB PSNR，说明性能优势不完全来自参数规模。

## 相关工作脉络
- **Fourier Feature / MLP-PE**：将坐标映射到高频空间，但需手动选择频率与维度；本文不依赖坐标编码，而是通过结构异质性自然互补。
- **SIREN**：用周期激活函数增强高频建模；需精细初始化；本文保留 MLP 低频频偏的同时利用 KAN 补充细节。
- **FINER / FR / MFN / WIRE / BACON**：均通过修改激活或网络乘法结构改善 INR 高频拟合；本文从双分支协同与小波正则化角度切入，机制不同。
- **KAN（原始工作）**：提出用可学习一元函数替代固定激活；本文首次将 KAN 与 MLP 构成互补双分支 INR 框架。
- **WaveKAN**：将自适应小波分解与 KAN 结合用于时间序列预测；本文面向多维信号隐式表示，融合策略与正则化设计完全不同。
- **Wavelet-based INR（如 WIRE）**：将小波纳入坐标编码；本文对小波的应用是在分支输出端的频域融合与正则化，定位差异明显。

## 局限性与未来方向
- **计算开销增加**：双分支架构参数多于单分支基线（如 2D 图像任务 4.13M vs FINER 198K），收敛时间和显存占用更高。
- **超参敏感**：KAN 分支对网络深度/宽度、B 样条阶数 k、网格尺寸敏感，需针对性调优；MLP 分支相对稳定。
- **硬频带分配不可行**：消融实验表明固定路由（hard subband assignment）虽能增强频带集中度但显著降低重建精度，软正则化是当前最优折衷。
- **未报告推理速度优化**：强调优化收敛时间，未深入讨论实际部署时的推理效率。
- **未来方向（作者自述）**：更轻量的异构分支设计、自适应频域协调机制、更高效 CUDA 实现以改善精度-效率平衡。

## 研究启发与可借鉴点
- **"异构结构互补"思路**：不单纯改造同一网络，而是引入不同结构（MLP+KAN）让各自偏置形成互补，可迁移到其他需要同时建模全局与局部信息的表示任务。
- **小波域软正则化而非硬路由**：用 L_band 正则化引导分支频域行为，而非强制频带分配，既保留灵活性又实现软约束，设计思路值得借鉴。
- **参数控制对比实验**：Table IX 中将基线调至相近参数量再比较，有效排除了"仅靠更大模型"的质疑，是严谨的消融范式。
- **在多维信号上的统一框架**：1D→4D 统一处理（沿坐标轴逐线做 1D DWT 而非构建完整 D 维小波张量），是高维扩展的实用技巧。
- **与逆问题结合的自然性**：公式（17）-（21）展示了将 INR 嵌入 forward operator A 的通用逆问题框架，可对接 CT、光场超分等下游应用。

## 关键术语表
- **Implicit Neural Representation (INR)**：将信号参数化为从连续坐标到信号值的神经网络映射，支持任意坐标查询，替代离散网格表示。
- **Spectral Bias（谱偏差）**：MLP 等神经网络天然优先学习低频分量、难以拟合高频细节的现象。
- **Kolmogorov-Arnold Network (KAN)**：受 Kolmogorov-Arnold 表示定理启发，用可学习一元函数（B 样条参数化）替代 MLP 固定激活的网络结构。
- **Discrete Wavelet Transform (DWT)**：将信号分解为近似子带（低频）与细节子带（高频）的多分辨率分析工具，本文用 Haar（db1）沿坐标轴逐线实现。
- **Band-Separation Regularization（带分离正则化）**：通过在频域惩罚 MLP 高频响应和 KAN 低频响应，引导双分支形成互补频域行为的损失项。
- **Signed Distance Function (SDF)**：表示隐式几何表面的标量场，场内点到表面有符号距离，本文用于 3D 形状重建评估。
- **Light Field (LF)**：记录场景中光线在空间与角度维度的 4D 数据，本文用作 4D 信号表示的评测任务。

## 可复现要素
- **数据集**：DIV2K（公开）、HCI 4D LF Benchmark（公开）、MSU Video SR Benchmark（公开）、Stanford 3D Scanning Repository（公开）、Head/Standard 体数据（公开）、1D 方波及 audio（来自 SIREN 论文，公开）。
- **代码**：1D 实验使用原始 KAN 代码库；2D-4D 实验使用自定义 CUDA KAN 层（集成 PyTorch 2.0.0 + CUDA 11.3，单卡 RTX 4090）；论文未提供 GitHub 链接。
- **权重**：论文未提及开源权重，每目标信号独立优化。
- **关键超参**：λ_band = 10^{-3}，α = 0.5，J = 3（小波分解级数），db1（Haar 小波基），KAN k=3、grid size=64，MLP 3 层宽 64，KAN 宽度 128-256-128，迭代次数 5000（1D/2D/4D）或 20000（3D/视频）。

---
title: "RELAYVSR-LARGE-SMALL-MODEL-COLLABORA-TION-FOR-EFFICIENT-REAL"
source: https://arxiv.org/pdf/2609.37850v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-10-01 09:38:19"
field: "视频超分辨率与生成模型效率优化"
keywords: ["视频超分辨率", "流式VSR", "大模型小模型协作", "稀疏关键帧", "强化学习优化", "Diffusion-NFT", "感知质量"]
innovations: ["稀疏生成中继机制：大模型仅稀疏处理关键帧，轻量网络复用参考逐帧超分", "双记忆视频Transformer：持久关键帧记忆+滚动视频记忆的联合注意力设计", "VARO双层奖励强化学习：系统级与参考级奖励协同优化大-小模型协作间隙"]
benchmarks: ["REDS30", "UDM10", "YouHQ40", "VideoLQ", "LongVSR60"]
---

# 论文速读：RELAYVSR-LARGE-SMALL-MODEL-COLLABORA-TION-FOR-EFFICIENT-REAL

## 一句话总结
RelayVSR 提出了一种稀疏生成中继（Sparse Generative Relay）流式视频超分辨率框架：大型生成模型仅负责稀疏关键帧的参考 latent 生成，轻量级双记忆视频 Transformer 则利用这些参考和低分辨率视频逐帧超分；通过视频感知参考优化（VARO）的强化学习机制弥合大-小模型协作间隙，实现了高质量与高效率的平衡。

## 研究问题与动机
1. **计算开销问题**：基于扩散的大生成模型能恢复逼真细节，但全视频逐帧推理在高分辨率下计算成本极高，尤其是每次前向传播都需要处理整段视频的时空 latent。
2. **已有优化瓶颈**：现有工作（如 FlashVSR、SwiftVR）通过单次前向传播、稀疏注意力、轻量解码器等方式降低开销，但即使单次推理仍需用大模型处理整个视频序列的 latent，进一步优化生成骨干的边际收益递减。
3. **协作优化间隙（Collaboration Gap）**：关键帧 latent 被多帧复用会传播并累积纹理/颜色误差，仅优化关键帧质量无法保证最终视频质量——轻量网络利用关键帧 latent 的下游效应未被考虑。
4. **流式实时性需求**：真实应用场景需要高效流式 VSR，同时保持 perceptual quality 和 temporal consistency。

## 核心贡献（创新点）
1. **稀疏生成中继机制**：大模型仅对稀疏关键帧（间隔Δ帧）生成参考 latent 并直接传递，轻量网络用 LR 视频+参考 latent 逐帧超分，将大模型计算均摊到每帧；与 FlashVSR 等需大模型处理全视频 latent 的方法本质不同。
2. **双记忆视频 Transformer**：持久关键帧记忆缓存参考 layer-wise K/V，滚动视频记忆缓存近期帧 K/V，当前帧联合查询两者以实现细节复用与本地时序上下文融合；区别于 SparkVSR 直接将 sparse latent 送入大扩散模型的做法。
3. **视频感知参考优化（VARO）**：用强化学习更新大模型，设计双层奖励——系统级奖励评估固定轻量网络生成的完整视频质量（非关键帧质量、时序一致性），参考级奖励评估解码关键帧质量；与 RFSR、GDPO-SR 等仅优化模型直接输出的方法相比，面向下游协作链路优化。
4. **首个-keyframe 与双端点两种推理模式**：支持无未来帧依赖的因果推理（first-keyframe）和有界前瞻的双端点推理（dual-endpoint），适配不同延迟预算。
5. **实证验证大-小模型协作收益**：在多个 benchmark 上取得最优 perceptual 指标，同时实现 29.29 FPS @ 1080p（单卡 A100），较 FlashVSR-Tiny 吞吐提升 3.76×，显存降低 43.5%。

## 方法详解

### 整体架构
RelayVSR 由三部分组成：（1）Sparse Generative Relay：预训练视频生成模型经两阶段适配为一步 GAN 蒸馏参考生成器；（2）Dual-Memory Video Transformer：轻量 VSR 网络；（3）VARO：用 Diffusion-NFT 进行强化学习后训练。

### Sparse Generative Relay
- **Stage 1（帧级 latent 适配）**：冻结 VAE 编码器 E，独立编码每个 HR 关键帧为 $z_i^★ = \mathcal{E}(y_{k_i})$，用 LoRA（rank=512）微调 Wan2.2 模型，flow-matching 目标 $\mathcal{L}_{FM} = \mathbb{E}[\text{MSE}(v_\theta(Z_\tau, \tau, c), v^★)]$，block-causal 注意力限制 temporal context 为当前关键帧及固定窗口内近邻关键帧。
- **Stage 2（一步 GAN 蒸馏）**：结合 latent 重建损失与 relativistic adversarial loss：
  $$\mathcal{L}_G^{1st} = \text{MSE}(\hat{Z}, Z^★) + \lambda_{adv}\mathcal{L}_{RSGAN}^G, \quad \mathcal{L}_D = \mathcal{L}_{RSGAN}^D + \gamma_{R1}\mathcal{L}_{feature-R1}$$
  推理时用 KV cache 实现流式生成：$(\hat{z}_j, s_j^G) = G_\theta(x_{k_j}, \epsilon_j, s_{j-1}^G)$。

### Dual-Memory Video Transformer
- 特征提取：Conv encoder 处理 LR 帧，独立 projection 处理参考 latent，两者统一 spatial token 维度。
- 参考独立自注意力预计算 layer-wise K/V，存入持久关键帧记忆（persistent keyframe memory）。
- 滚动视频记忆（rolling video memory）按帧更新。
- 联合注意力：
  $$a_t = \text{Attn}(Q_t, [K_t^r; K_t^v], [V_t^r; V_t^v])$$
  参考使用更大空间邻域（48×48 tokens），视频邻域较小（24×24），均限制局部计算。
- 输出头：投影 + PixelShuffle×16，与 bicubic upsampling 的 LR 帧相加得 4× 超分结果。训练用 $L_1 + 0.1 \cdot \text{LPIPS}$，右参考以 0.25 概率随机丢弃以支持两种模式。

### VARO（Video-Aware Reference Optimization）
- 固定轻量网络参数 $\phi^★$，生成因果参考序列 $\hat{Z}$，得到 $\hat{Y} = F_{\phi^★}(X, \hat{Z})$。
- 双层归一化奖励：
  $$R_{total} = \lambda_{sys}\tilde{R}_{sys}(\hat{Y}, Y) + \lambda_{ref}\tilde{R}_{ref}(\hat{Z}, Y_\mathcal{K})$$
  - 系统级奖励：RGB $L_1$ + DOVER++（技术质量）+ MUSIQ（空采帧）+ 光流 warping error（连续帧）；权重 $(0.40, 0.25, 0.25, 0.10)$。
  - 参考级奖励：RGB $L_1$ + MUSIQ（每解码参考）+ 光流 warping error；权重 $(0.50, 0.40, 0.10)$。
  - 层级权重 $(\lambda_{sys}, \lambda_{ref}) = (1.00, 0.25)$。
- 用 Diffusion-NFT 优化，噪声端点 $\tau=1$，group-normalized 奖励映射 $r \in [0,1]$，损失：
  $$\mathcal{L}_{VARO} = \mathbb{E}[r\|v_\theta^+ - v\|_2^2 + (1-r)\|v_\theta^- - v\|_2^2]$$
  其中 $v_\theta^\pm = v_{samp} \pm \beta(v_\theta - v_{samp})$，$\beta=1$，EMA 混合系数 $\eta_i = \min(0.001i, 0.5)$。训练 2000 步，batch=2 clips（8 rollouts），lr=$5\times10^{-6}$，BF16 on 16 GPUs。

## 实验与结果

### 数据集与基线
- **合成基准**：REDS30、UDM10、YouHQ40（RealBasicVSR 退化）；**真实世界**：VideoLQ；**长视频**：LongVSR60（30 real + 30 AI-generated，~1000帧/视频）。
- **基线**：RealViformer、STAR、DOVE、SeedVR2、SparkVSR、SwiftVR、FlashVSR-Tiny。

### 主要结果（Table 1）
- **UDM10**：PSNR 25.94，SSIM **0.7545**（最佳），LPIPS **0.2171**，MUSIQ **65.32**（最佳），CLIP-IQA 0.4852，DOVER 0.4010（最佳）。
- **YouHQ40**：LPIPS **0.2569**（最佳），DOVER **0.7313**（最佳）。
- **VideoLQ**：MUSIQ **51.26**（最佳），DOVER **0.5488**（最佳）。
- **LongVSR60**：DOVER 0.5911（最佳），MS 0.9827；NIQE 略逊于 FlashVSR（4.14 vs 3.82），但 perceptual 指标领先。

### 效率（Table 2, 1080p / 单 A100 80G）
| 模式 | FPS | 峰值显存(GB) | 首帧延迟(s) |
|---|---|---|---|
| RelayVSR dual-endpoint (Δ=15) | **29.29** | **13.82** | 0.327 |
| RelayVSR first-keyframe | **35.89** | 13.72 | 0.183 |
| FlashVSR-Tiny | 7.80 | 24.45 | 2.830 |
| SwiftVR | 23.48 | 31.36 | 1.107 |

- 相对 FlashVSR：**吞吐提升 3.76×，显存降低 43.5%**。
- 2160p 时 dual-endpoint 达 8.23 FPS / 24.25 GB，对比 FlashVSR 1.30 FPS / 67.99 GB。

### 消融（Table 3, 4, UDM10, Δ=15）
- 双记忆 vs 单记忆：增加参考记忆 MUSIQ 56.80→63.65，增加视频记忆 DOVER 0.5110→0.5290。
- VARO：无后训练 MUSIQ 63.72 → 全 VARO 65.32，DOVER 0.5290 → 0.5448，MS 0.9830 → 0.9900；参考级单独更优 keyframe MUSIQ（64.95），系统级单独更优全帧 MUSIQ（64.86），两者结合最佳。
- 质量-效率权衡：Δ 从 5→15 FPS 18.10→29.29，LPIPS 0.2115→0.2171；Δ=15 为最佳平衡点。

### 用户研究（Table 11）
- 相对 DOVE/SeedVR2/SparkVSR/SwiftVR 均有正向整体质量 GSB（+26.7% ~ +53.6%）；相对 FlashVSR 整体质量持平（-3.3%），但内容保真 +9.0%、时序稳定性 +17.0%。

## 相关工作脉络
1. **FlashVSR（Zhuang et al., 2026）**：单步扩散 VSR + 局部稀疏注意力 + LR 条件轻量解码器；本文与其区别在于 FlashVSR 仍需大模型处理全视频 latent，RelayVSR 将大模型调用稀疏化到关键帧，通过轻量网络复用参考。
2. **SwiftVR（Yan et al., 2026）**：shifted-window attention + 轻量 autoencoder 流式 VSR；本文更强调大-小模型协作而非纯轻量设计，用参考 latent 注入生成先验。
3. **SparkVSR（Yu et al., 2026）**：基于 LR 内超分关键帧，将 sparse latent 拼接后输入大扩散模型；本文不重复调用大模型，而是用轻量网络消费关键帧 latent 实现逐帧超分。
4. **Diffusion-NFT（Zheng et al., 2026）**：在线扩散 RL，将 reward 反馈嵌入 forward-process flow matching；本文借用此框架实现 VARO，但 reward 从单模型输出扩展到跨模型协作链路。
5. **RFSR / GDPO-SR（Sun et al., 2024; Yi et al., 2026）**：图像 SR 的 reward 优化；本文将其拓展到视频场景且引入双层奖励（系统级+参考级）。
6. **PS-SR（Wu et al., 2026）**： speculative diffusion 分配大/轻量模型计算；本文思路不同，不是 speculation 而是 sparse relay，用关键帧参考贯穿整个 interval。

## 局限性与未来方向
1. **固定关键帧间隔**：未根据场景动态变化调整 Δ 和参考选择，双端点模式需有界 lookahead，难以适配快速运动或静止交替的场景。
2. **误差传播**：生成参考的纹理/颜色误差仍会在非关键帧间累积；需结合推理时的参考可靠性估计以引导记忆读取与刷新策略。
3. **显存开销**：大模型虽稀疏调用仍贡献峰值显存，当前结果在 A100 80G 验证，设备端部署需模型压缩或更小参考生成器。
4. **未来方向**：自适应关键帧调度、参考置信度感知推理、轻量化部署。

## 研究启发与可借鉴点
1. **"大模型稀疏调用 + 轻量模型复用参考"范式**：将昂贵模型的计算摊薄到多个轻量推理步骤，可迁移至图像/视频修复、去噪、风格迁移等任务，构建 large-small 协作 pipeline。
2. **双层奖励 RL 训练思路**：系统级（下游输出）+ 参考级（中间产物）的组合奖励可有效优化多模块协作链路，适用于任何存在 intermediate representation 的多阶段生成系统。
3. **持久记忆 + 滚动记忆的注意力设计**：将 reusable reference 和 recent context 分离存储、分别计算 K/V 并联合查询，兼顾效率和语义连贯性，可推广至流式视频理解/生成任务。
4. **Diffusion-NFT 在生成模型后训练中的应用**：无需重新训练整个模型，仅更新关键模块（此处为大模型 LoRA），实现以 perceptual reward 为导向的精细化优化。
5. **质量-效率权衡曲线（Δ  sweep）的量化分析**：通过系统扫参揭示关键帧间隔对 FPS、LPIPS、DOVER 的影响趋势，为工程部署提供直接的调参依据。

## 关键术语表
**Sparse Generative Relay**：大生成模型仅在稀疏关键帧上执行参考生成，轻量网络复用这些参考逐帧超分的协作机制。
**Dual-Memory Video Transformer**：含持久关键帧记忆（缓存参考 K/V）和滚动视频记忆（缓存近帧 K/V）的双流注意力结构。
**Collaboration Gap**：大模型关键帧优化目标与轻量网络最终视频质量之间的不匹配，导致关键帧误差传播累积。
**VARO (Video-Aware Reference Optimization)**：基于 Diffusion-NFT 的强化学习后训练，用系统级+参考级双层奖励优化大模型以弥合协作间隙。
**System-level Reward**：评估固定轻量网络产出的完整超分视频质量的奖励信号（含保真、感知、时序一致性）。
**Reference-level Reward**：评估解码后关键帧参考质量的奖励信号（含保真、感知、时序）。
**Diffusion-NFT**：将 reward feedback 嵌入 forward-process flow matching 的扩散模型在线强化学习方法。
**First-keyframe / Dual-endpoint 模式**：前者仅用区间左端关键帧无前瞻，后者用左右两端有界前瞻（固定播放延迟）的两种推理配置。

## 可复现要素
- **数据集**：REDS30、UDM10、YouHQ40、VideoLQ（公开）；LongVSR60（论文提供）；训练数据约 0.5M 视频 + 1M 图像（来源未明确公开链接）。
- **代码/权重**：代码已开源 https://github.com/kopperx/RelayVSR；模型权重未提及开源状态。
- **关键超参**：LoRA rank=512, α=512；Stage 1 学习率 2e-5、75K 步；Stage 2 学习率 1e-5、5K 步，λ_adv=0.1, γ_R1=1000；轻量网络 lr=1e-4、60K 步、batch=64；VARO lr=5e-6、2000 步、batch=2 clips、λ_sys=1.0、λ_ref=0.25、β=1；HR crop 704×1280（大模型）/ 1024×1536（轻量）；Δ=15 为主要设置。
- **硬件**：训练 16×A100；推理评测单 A100 80G。

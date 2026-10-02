---
title: "TexTailor-Texture-Preserving-Video-Virtual-Try-On-via-Adapti"
source: https://arxiv.org/pdf/2609.39335v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-10-02 01:53:26"
field: "视频虚拟试穿"
keywords: ["video virtual try-on", "diffusion transformer", "texture preservation", "high-resolution video generation", "adaptive conditioning", "temporal consistency"]
innovations: ["时序自适应视觉调制(TAVM)实现去噪阶段的细粒度服装特征动态调控", "帧对齐3D交叉RoPE(FAC-RoPE)消除人工时序偏移并建立稳定patch级对应", "多源交叉注意力解耦注入(MCAI)减少异构条件间的跨模态干扰"]
benchmarks: ["Eevee", "ViViD"]
---

# 论文速读：TexTailor: Texture-Preserving Video Virtual Try-On via Adaptive Garment Conditioning

## 一句话总结
本文提出 TexTailor，一种基于预训练视频 Diffusion Transformer 的高保真视频虚拟试穿框架，通过时序自适应视觉调制、帧对齐三维位置编码和多源交叉注意力解耦，在高分辨率场景下有效保持服装纹理、图案和局部结构细节，同时维持视频时序一致性。

## 研究问题与动机
- **高分辨率下细粒度服装保真度不足**：现有方法多在低分辨率数据集上开发，放大到高分辨率时服装的纹理、缝线、图案等细节容易模糊、丢失或出现时序漂移。
- **服装条件利用不充分**：多数方法在整个去噪过程中以固定方式注入服装视觉特征，未根据去噪阶段动态调整指导强度。
- **缺乏显式的位置对应建模**：服装与视频 latent 之间进行 cross-attention 时，没有显式建模两者在空间/时序上的对应关系，导致局部匹配不一致。
- **异构条件相互干扰**：文本、视觉特征和服装 latent 通常被混合注入到共享注意力空间中，可能产生跨模态干扰，削弱服装特定引导的有效性。

## 核心贡献（创新点）
1. **TAVM（时序自适应视觉调制）**：结合 timestep-conditioned AdaLN 与 token-wise gating，实现从粗结构到细纹理的阶段自适应服装引导；与已有工作的本质区别在于动态调节而非静态注入。
2. **FAC-RoPE（帧对齐的3D服装交叉旋转位置编码）**：为静态服装 key 分配与视频 query 相同的时间坐标，消除人工时序偏移并建立稳定的 patch 级对应；区别于传统固定时间坐标的做法。
3. **MCAI（多源交叉注意力注入）**：将文本、时序调制视觉特征和帧对齐服装 latent 通过并行 cross-attention 分支独立注入，减少跨模态干扰；与直接拼接异构条件的做法形成对比。
4. **Mask-aware Loss**：对服装区域赋予更强的监督信号，强调高分辨率试穿中关键区域的优化；现有工作通常对所有空间位置一视同仁。

## 方法详解
- **基础架构**：基于预训练的 Wan2.2-Fun-5B-InP 视频 Diffusion Transformer，采用 Rectified Flow 目标（flow matching），训练损失如公式 (3)：$\mathcal{L}_{\mathrm{FM}} = \mathbb{E}[w(t)\|\epsilon_\theta(z_t, t, c) - u_t\|^2_2]$。
- **TAVM**：使用 SigLIP 2 和 DINOv3 提取互补视觉 token 序列。对每个视觉流应用 timestep-conditioned AdaLN 得到 $\hat{X}$，再通过 timestep-aware gating 重新加权：$g_i = \mathcal{G}(\hat{x}_i, e_t)$，$\tilde{x}_i = g_i \cdot \hat{ x}_i$。早期阶段侧重粗结构，后期阶段聚焦细粒度纹理。
- **FAC-RoPE**：视频 query 保留原生 3D RoPE：$\widehat{Q}_v^f = \mathrm{RoPE}_{3D}(Q_v^f; f, h, w)$；服装 key 被赋予与 video query 相同的帧时间坐标：$\widehat{K}_g^f = \mathrm{RoPE}_{3D}(K_g; f, h_g, w_g)$，避免引入 $f - 0$ 的人工偏移。
- **MCAI**：将条件分解为三个独立 cross-attention 分支：文本语义、调制视觉特征、服装 latent，各自独立投影和注意力计算，减少跨模态干扰。
- **训练策略**：两阶段训练，先在较低分辨率稳定适配，再在 1088×816 上 fine-tune；每样本含 49 帧视频；LoRA 用于 self-attention 和 cross-attention 的 query 投影；新增模块全量 fine-tune。
- **Mask-aware 损失**：$\mathcal{L}_{\mathrm{mask}} = \mathbb{E}[w(t)\|M \odot (\epsilon_\theta - u_t)\|^2_2]$，最终目标 $\mathcal{L} = \mathcal{L}_{\mathrm{FM}} + \lambda \mathcal{L}_{\mathrm{mask}}$。

## 实验与结果
- **数据集**：主实验使用 Eevee（1088×816 高分辨率，服装图 2400×1800，上装 4492/500、下装 2308/250、连衣裙 1564/250 训练/测试样本）和 ViViD。
- **评估指标**：SSIM、LPIPS、VFID-I3D（VFID_I）、VFID-ResNeXt（VFID_R）、VGID（基于 DINOv2 的服装区域语义一致性）。
- **Eevee Full-shot 结果**：TexTailor VFID_R=**0.149**（最佳）、VFID_I=8.872、VGID=0.531；较 MagicTryOn 的 VFID_R 提升约 20%。
- **Eevee Close-up 结果**：TexTailor VFID_R=**0.489**（最佳）、VFID_I=10.917、VGID=**0.543**（最佳）。
- **ViViD 配对结果**：TexTailor VFID_I=**13.8754**（最佳）、VFID_R=0.2861、SSIM=0.9017、LPIPS=0.0619，GPU 显存 25.32G、推理时间 196.85s。
- **消融结论**：移除任一组件均导致性能下降；TAVM 中 SigLIP2 和 DINOv3 提供互补表征；token-wise gating 对渐进细化至关重要；FAC-RoPE 和 MCAI 各分支均贡献显著。

## 相关工作脉络
- **ViViD**（Fang et al. 2024）：扩展图像扩散模型到视频试穿，但主要在低分辨率下开发，缺乏细粒度高分辨率保持能力。
- **MagicTryOn**（Li et al. 2025b）：采用 DiT backbone，但未引入阶段自适应调制和帧对齐位置编码。
- **CatV²TON**（Chong et al. 2025）：统一时空建模的 DiT 方法，仍以拼接方式处理条件，未解决跨模态干扰问题。
- **WildVidFit**（He et al. 2024）：基于可控扩散的早期视频试穿方法，侧重粗粒度服装传输。
- **RealVVT**（Li et al. 2025c）：强调写实与时序稳定，但在高分辨率服装细节保真方面不如本文。
- **Eevee Benchmark**（Zeng et al. 2025）：本文提出的高分辨率评估基准，揭示现有方法在 close-up 场景下的细节保留短板。

## 局限性与未来方向
- **依赖辅助输入**：需要姿态、agnostic 表示和视频 mask，预处理管线较复杂；未来可向 mask-free 范式发展。
- **复杂运动场景挑战**：大形变、重度遮挡、显著视角变化仍较难处理；需进一步提升高分辨率下复杂姿态和视角变化的鲁棒性。
- **条件管线简化**：未来可探索更高效、更简化的条件注入策略。

## 研究启发与可借鉴点
- **TAVM 的阶段自适应思想可迁移**：token-wise gating + AdaLN 的组合设计可用于其他需要多尺度细节保持的视频生成任务（如视频编辑、超分）。
- **FAC-RoPE 的位置对齐策略**：将静态参考的特征 key 与动态 query 的帧坐标对齐，可推广到其他 reference-guided video generation 任务。
- **多源解耦注入设计**：MCAI 的并行 cross-attention 分支思路可用于多条件控制的图像/视频生成，减少条件间的相互干扰。
- **Mask-aware 损失**：对关键区域加权监督的思路可应用于其他需要局部高保真的生成任务。
- **两阶段训练策略**：先低分辨率适配再高分辨率 fine-tune 的策略在高分辨率视频生成中具有普适参考价值。

## 关键术语表
- **Video Virtual Try-On (VVT)**：在动态视频中保持时序一致性的虚拟试穿任务。
- **Diffusion Transformer (DiT)**：基于 Transformer 架构的扩散模型骨干网络。
- **Timestep-Adaptive Visual Modulation (TAVM)**：根据去噪时序动态调制服装视觉特征的机制。
- **Frame-Aligned 3D Cross-RoPE (FAC-RoPE)**：为服装 key 分配与视频 query 相同时间坐标的三维旋转位置编码。
- **Multi-Source Cross-Attention Injection (MCAI)**：将文本、视觉和服装 latent 通过并行分支独立注入的设计。
- **Flow Matching**：基于常速流场的扩散模型训练目标。
- **VGID**：基于 DINOv2 特征的服装区域语义一致性评估指标。
- **Eevee**：高分辨率视频虚拟试穿 benchmark，提供 close-up 场景下的细粒度评估。

## 可复现要素
- **数据集**：Eevee（公开）、ViViD（公开）。
- **代码**：论文未明确声明开源状态。
- **权重**：基于 Wan2.2-Fun-5B-InP 预训练权重；论文未声明新增权重开源。
- **关键超参**：训练分辨率两阶段（较低分辨率→1088×816）；每样本 49 帧；推理 25 步去噪；4×NVIDIA A100 (80GB) GPU。
- **LoRA**：应用于 self-attention 和 cross-attention 的 query 投影。

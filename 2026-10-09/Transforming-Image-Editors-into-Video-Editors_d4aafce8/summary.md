---
title: "Transforming-Image-Editors-into-Video-Editors"
source: https://arxiv.org/pdf/2610.11037v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-10-09 17:29:10"
field: "视频生成与编辑"
keywords: ["视频编辑", "图像到视频", "关键帧插值", "扩散模型", "指令遵循", "运动建模", "两阶段方法"]
innovations: ["两阶段解耦：组合图像编辑+运动引导插值", "锚点保持扩散：反向过程固定关键帧latent", "轻量运动编码器：仅5%参数实现有效运动保留"]
benchmarks: ["IVEBench", "VIE-Bench"]
---

# 论文速读：Transforming-Image-Editors-into-Video-Editors

## 一句话总结
本文提出 Anchor-based Video Editing (AVE)，一种两阶段视频编辑框架：先用强图像编辑器对稀疏关键帧进行组合编辑，再用运动引导的图像到视频扩散模型（Wan2.2）以编辑后的关键帧为固定锚点进行插值生成，从而将成熟的图像编辑能力低成本迁移到视频领域。

## 研究问题与动机
1. **视频编辑质量落后于图像编辑**：现代图像编辑器已具备强大的语义理解、视觉保真度和指令遵循能力，但视频编辑在视觉质量、语义一致性和指令遵循方面仍明显落后。
2. **端到端视频编辑训练成本高昂**：视频需处理更长的时空上下文，收集高质量视频编辑数据需要时序一致性、运动多样性和跨帧指令对齐，训练成本显著高于图像编辑。
3. **现有方法难以兼顾质量与效率**：逐帧编辑缺乏时序一致性导致闪烁；端到端视频编辑虽时序性好但训练昂贵且仍需大量视频特定数据。
4. **核心洞察：视频编辑的主要难点仍是图像级语义编辑**：通过实验发现，一旦获得高质量的关键帧编辑结果，剩余的视频插值任务比端到端视频编辑简单得多。

## 核心贡献（创新点）
1. **两阶段解耦设计**：将视频编辑显式分解为"组合图像编辑"和"关键帧插值"两个子问题，与端到端方法本质不同——视频编辑难题被委托给成熟的图像编辑器。
2. **锚点保持扩散（Anchor-preserving diffusion）**：在反向扩散过程中将编辑后的关键帧latent固定（clamp），仅对非锚点帧进行去噪插值，保证关键帧内容不变同时生成时序连贯的视频。
3. **轻量级运动编码器**：仅训练约710M参数（占Wan2.2总参数~5%）的6层因式时空Transformer运动编码器，通过ControlNet残差分支注入冻结的视频生成骨干网络。
4. **直接继承图像编辑器进展**：Stage-1和Stage-2完全解耦，替换更强的图像编辑器即可提升视频编辑质量，无需重新训练视频生成阶段。
5. **SOTA性能与良好平衡**：在IVEBench短/长视频子集和VIE-Bench上均取得最佳或最具竞争力的结果，尤其在运动保真度（Motion）和语义一致性上优势明显。

## 方法详解

### 整体框架
给定输入视频 $V = \{x_t\}_{t=1}^T$ 和文本编辑指令 $y$，目标是生成编辑视频 $\hat{V} = \{\hat{x}_t\}_{t=1}^T$。选取 $K$ 个关键帧索引 $\mathcal{A}$，两阶段流程为：

**Stage-1（组合图像编辑）**：$\hat{\mathbf{x}}_\mathcal{A} = \mathcal{E}(\mathbf{x}_\mathcal{A}, y)$，使用图像编辑器 $\mathcal{E}$（如BAGEL微调版或Nano Banana API）对关键帧集进行联合编辑。

**Stage-2（关键帧插值）**：$\hat{V} = \mathcal{G}(\hat{\mathbf{x}}_\mathcal{A}, \mathbf{m}(V))$，使用图像到视频扩散模型 $\mathcal{G}$，以编辑后的关键帧为固定锚点、源视频运动表示 $\mathbf{m}(V)$ 为条件生成完整视频。

### Stage-1：组合图像编辑
从私有视频编辑数据集 $(V, y, V^*)$ 中采样对齐的关键帧对，构建约1.5M多图像编辑训练样本 $(\mathbf{x}_\mathcal{A}, y, \mathbf{x}_\mathcal{A}^*)$。微调BAGEL，损失函数为：

$$\mathcal{L}_{\text{img}} = \mathbb{E}_{\mathbf{z}_\mathcal{A}^*, \epsilon, \tau}\left[\|\epsilon - \epsilon_\phi(\alpha_\tau \mathbf{z}_\mathcal{A}^* + \sigma_\tau \epsilon, \mathbf{z}_\mathcal{A}, y, \tau)\|_2^2\right]$$

其中 $\mathbf{z}_\mathcal{A} = \mathcal{V}_{\text{img}}(\mathbf{x}_\mathcal{A})$ 为图像VAE隐码。联合训练多帧确保关键帧间编辑一致性。

### Stage-2：关键帧插值
**轻量级运动编码器**：源视频经Wan2.2的3D VAE编码为 $Z \in \mathbb{R}^{T \times H \times W \times C}$，投影到运动嵌入空间：
$$U_{t,h,w}^{(0)} = W_{\text{in}}Z_{t,h,w} + e_t^{\text{temp}} + e_{h,w}^{\text{spatial}}$$
经6层因式时空Transformer（先空间自注意力、后时间自注意力）提取运动特征 $h_{\text{mot}}$，通过ControlNet残差分支注入冻结的DiT。

**锚点保持扩散**：定义二值锚点掩码 $M_\mathcal{A}$，初始潜视频：
$$X_S = M_\mathcal{A} \odot \hat{Z}_\mathcal{A} + (1 - M_\mathcal{A}) \odot \epsilon, \quad \epsilon \sim \mathcal{N}(0, I)$$
每步去噪后重新施加锚点约束：
$$X_{s-1} = M_\mathcal{A} \odot \hat{Z}_\mathcal{A} + (1 - M_\mathcal{A}) \odot \tilde{X}_{s-1}$$
最终视频 $\hat{V} = \mathcal{V}_{3D}^{\text{dec}}(X_0)$。

训练损失仅作用于非锚点帧：$\mathcal{L}_{\text{vid}} = \|(1 - M_\mathcal{A}) \odot (\hat{\epsilon} - \epsilon)\|_2^2$。

## 实验与结果

### 数据集与评估基准
- **IVEBench**（Chen et al., 2026b）：包含短/长视频子集，评估指令遵循、语义一致性、内容保真度、运动平滑度等。
- **VIE-Bench**（Mou et al., 2025）：涵盖add、swap/change、remove、style/tone transfer四类编辑任务。

### 主要结果
**IVEBench短视频子集**（Table 2）：
| 指标 | AVE | 次优（InsV2V） | 提升 |
|------|-----|----------------|------|
| Total | 70.58 | 66.68 | +3.90 |
| Quality | 82.10 | 79.58 | +2.52 |
| Ins. Comp. | 49.65 | 38.61 | +11.04 |
| Motion | 89.26 | 85.55 | +3.71 |

**IVEBench长视频子集**（Table 3）：
- Total: 68.75（SOTA），Quality: 82.15，Motion: 88.95
- 在长视频上仍保持优势，证明方法对长时序建模有效。

**VIE-Bench**（Table 4）：
- Add: 9.054，Swap/Change: 9.448（SOTA），Remove: 9.352（SOTA），Style: 9.203
- 指令遵循平均分9.539（SOTA），运动保持9.425，视频质量8.630
- 在swap/change和remove任务上显著优于开源基线，接近Kling-Omni等闭源系统。

### 消融实验
1. **Stage-1图像编辑器质量**（Table 5）：Nano Banana 2 > GPT-5.1 > Seedream 4.0 > 微调BAGEL > 原始BAGEL。视频编辑质量与图像编辑器质量强相关，验证核心假设。
2. **关键帧选择策略**（Table 6）：默认策略（PySceneDetect场景感知 + 最大间隔补全）优于均匀采样；最优关键帧数约 $\lfloor T/24 \rfloor$，过多过少均略降性能。

## 相关工作脉络
1. **InstructPix2Pix**（Brooks et al., 2023）：早期文本引导图像编辑扩散模型，AVE的Stage-1在此基础上发展，但AVE将其扩展到视频域。
2. **BAGEL**（Deng et al., 2025）：多模态统一图像编辑器，AVE以其为基础微调组合图像编辑能力。
3. **Tune-A-Video**（Wu et al., 2023）：单样本图像扩散模型调优至视频生成，AVE避免直接微调视频模型，改用两阶段解耦。
4. **VACE**（Jiang et al., 2025）：端到端视频编辑框架，需大量视频特定训练数据；AVE以极小训练成本（仅运动编码器）达到更强性能。
5. **SAMA**（Zhang et al., 2026a）：因式语义锚定与运动对齐方法，与AVE思路部分相似但AVE更强调继承图像编辑器进展。
6. **AnimateDiff**（Guo et al., 2024）：无需微调的文本到视频动画生成，AVE借鉴其轻量化时序模块思想但应用于编辑任务。

## 局限性与未来方向
1. **依赖Stage-1图像编辑器质量**：视频编辑上限受限于图像编辑器能力，若图像编辑器指令遵循或语义理解不足，视频编辑质量将显著下降。
2. **关键帧选择策略影响性能**：当前使用启发式策略（PySceneDetect + 最大间隔），对复杂动态场景或密集编辑可能不够鲁棒。
3. **关键帧数量折衷**：过少导致编辑意图表达不充分，过多压缩插值空间降低稳定性，最优策略待进一步探索。
4. **闭源API依赖**：最强实验结果使用Nano Banana API，开源场景下性能有所下降，需开发更强开源图像编辑器。
5. **未来方向**：与更强图像编辑器结合、探索自适应关键帧选择、处理更复杂运动变化（如姿态重定向）、扩展至3D视频编辑。

## 研究启发与可借鉴点
1. **任务分解范式**：将复杂视频生成任务分解为"内容生成"和"时序插值"两部分，前者委托给成熟子系统，后者用轻量模块解决，可迁移至其他多模态生成任务。
2. **锚点保持扩散技术**：在扩散过程中固定关键帧latent的clamp机制，保证指定帧内容不变的同时生成连贯序列，可用于视频补全、插帧等任务。
3. **轻量级运动建模**：仅训练5%参数的运动编码器即可有效保留源视频动力学，启示在视频生成中可将内容-运动解耦以降低训练成本。
4. **开源-闭源混合策略**：框架不绑定特定图像编辑器，可灵活选用API或开源模型，为资源受限场景提供降级方案。
5. **benchmark关联价值**：在IVEBench和VIE-Bench双基准上验证，方法对短/长视频及多种编辑类型均有效，为视频编辑研究提供统一评估视角。

## 关键术语表
- **Anchor-based Video Editing (AVE)**：本文提出的两阶段视频编辑框架，通过关键帧锚点将图像编辑能力迁移至视频域。
- **Composed Image Editing**：对多个关键帧进行联合图像编辑，保持跨帧语义一致性。
- **Anchor-preserving Diffusion**：在反向扩散过程中固定关键帧隐码，仅对非锚点帧去噪插值。
- **Lightweight Motion Encoder**：6层因式时空Transformer，从源视频提取运动特征注入冻结视频生成骨干。
- **Factorized Spatiotemporal Transformer**：先空间自注意力后时间自注意力的解耦注意力机制，高效建模时空特征。
- **ControlNet-style Residual Branch**：将运动特征以残差形式注入预训练DiT，保持生成能力同时添加运动控制。
- **Keyframe Selection**：从视频中采样稀疏关键帧的策略，影响编辑质量与插值稳定性。
- **Temporal Interpolation**：在编辑后的关键帧之间生成中间帧以恢复完整视频的任务。

## 可复现要素
- **数据集**：训练使用约1.5M多图像编辑样本（从私有视频编辑数据集构建）；评估使用公开基准IVEBench和VIE-Bench。
- **代码开源**：是，GitHub https://github.com/wangf3014/AVE
- **模型权重**：运动编码器及ControlNet分支权重开源；主干Wan2.2-I2V-A14B为预训练模型。
- **关键超参**：
  - 优化器：AdamW，lr=$1\times10^{-4}$，weight decay=0.05，$\beta_1=0.9$，$\beta_2=0.999$
  - 学习率调度：cosine schedule + linear warmup
  - 训练设备：64×A100 GPU
  - 训练样本：500K视频编辑样本（运动编码器训练）
  - 关键帧数：约$\lfloor T/24 \rfloor$
  - 可训练参数：710M（占总参数~5%）

---
title: "SCAFFOLD-THEN-INTERNALIZE-REPRESENTATION-INJECTION-FOR-DIFFU"
source: https://arxiv.org/pdf/2609.35292v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-10-01 09:35:26"
field: "扩散模型训练加速与表征学习"
keywords: ["Diffusion Transformer", "Representation Alignment", "REPA", "Scaffold-to-Internalization", "Vision Encoder", "Training Acceleration", "Image Generation"]
innovations: ["提出REPI框架，将编码器表示注入扩散Transformer并以脚手架-内化策略进行训练，与REPA反向互补", "发现K/V联合注入比V-only更适合脚手架移除与内化过程，揭示了信息流保持与外部知识吸收的平衡设计", "REPA+REPI组合在160K步达到FID 8.22，匹敌vanilla SiT 7M步的8.30，实现超43.5倍训练加速"]
benchmarks: ["ImageNet 256×256", "ImageNet 512×512", "MS-COCO text-to-image"]
---

# 论文速读：SCAFFOLD-THEN-INTERNALIZE-REPRESENTATION-INJECTION-FOR-DIFFU

## 一句话总结
论文提出了 REPI（REpresentation Injection），一种将预训练视觉编码器表示注入扩散 Transformer 并以"脚手架-内化"策略进行训练的框架，使编码器语义信息主动参与去噪过程；与 REPA 高度互补，二者结合后仅需 160K 步训练即可匹敌 vanilla SiT 的 7M 步效果，加速超 43.5×。

## 研究问题与动机
1. **现有 REPA 的方向是单向投影**：REPA 将扩散 Transformer 的隐藏状态投影到编码器空间进行对齐，论文探索反向路径——能否将编码器表示投影到扩散 Transformer 并直接参与去噪。
2. **直接替换隐藏状态会破坏信息流导致训练崩溃**：将编码器输出完全替换中间层 hidden state 会使后续 block 无法利用噪声输入，FID 高达 212.5。
3. **Oracle 设置不切实际**：若在推理时直接使用干净图像的编码器表示（Oracle），虽可在 30K 步即超越 vanilla SiT 的 7M 步效果，但标准生成流程无法访问干净图像特征。
4. **需要消除推理时对编码器的依赖**：论文提出脚手架-内化策略，训练初期用编码器表示作为临时脚手架引导学习，随后移除脚手架并通过内化损失使扩散模型自主复现该表示。

## 核心贡献（创新点）
1. **探索了与 REPA 反向且互补的知识迁移方向**：不同于 REPA 的"扩散→编码器"投影对齐，REPI 实现"编码器→扩散"的主动注入，使外部语义表示直接参与去噪计算。
2. **提出脚手架-内化训练框架（REPI）**：训练初期将编码器投影后的 K/V 临时替换扩散层的 K/V，移除脚手架后引入内化损失（stop-gradient + Frobenius 距离）引导原生表示逼近编码器表示，最终推理时移除编码器，架构不变。
3. **REPI 在多种骨干上超越 REPA 且与 REPA 高度互补**：SiT-XL/2 在 100K 步时 REPI（14.51）显著优于 REPA（19.40）；二者结合在 160K 步达到 FID 8.22，匹敌 vanilla SiT 7M 步的 8.30，提速超 43.5×。
4. **系统验证了注入目标、脚手架时长、损失权重等关键设计的选择依据**：揭示 K/V 联合注入比 V-only 更适合脚手架移除与内化（与 Oracle 结果形成有趣对比），并展示方法对各类超参的高度鲁棒性。

## 方法详解
1. **Oracle 诊断实验**：假设推理时可访问编码器表示，测试五种注入位置（Hidden state、Attention output、Q/K/V、K only、V only、K/V），V-only 表现最优（FID 4.6），但 K/V 更适合完整 REPI 框架（见消融）。
2. **表示注入（Representation Injection）**：
   - 从冻结的视觉编码器 $f$ 的最后自注意力层提取 K/V 表示 $(K_*, V_*)$。
   - 通过两个可训练线性投影层 $h_{\phi_K}, h_{\phi_V}$ 映射到扩散 Transformer 选定层的 K/V 空间：
     $$\bar{K} = h_{\phi_K}(K_*), \quad \bar{V} = h_{\phi_V}(V_*)$$
   - 替换该层的 K/V，保留原生 Q：
     $$Z_{\text{inj}} = \mathrm{softmax}\left(\frac{Q\bar{K}^\top}{\sqrt{d}}\right)\bar{V}$$
   - 保持残差连接和后续 block 的噪声输入信息流畅通。
3. **内化损失（Internalization Loss）**：
   - 脚手架移除后，鼓励原生 K/V 复现被注入的编码器表示：
     $$\mathcal{L}_{\text{int}} = \frac{1}{|K|}\|K - \mathrm{sg}[\bar{K}]\|_F^2 + \frac{1}{|V|}\|V - \mathrm{sg}[\bar{V}]\|_F^2$$
   - $\mathrm{sg}[\cdot]$ 为 stop-gradient 算子，防止投影层反向更新干扰内化过程。
4. **总损失函数**：
   $$\mathcal{L} = \mathcal{L}_{\text{velocity}} + \lambda \mathcal{L}_{\text{int}}$$
   默认 $\lambda = 2.0$（不含 REPA），与 REPA 联合时 $\lambda = 0.25$。
5. **REPI + REPA 组合策略**：
   - REPI 注入层置于 REPA 对齐层之前（推荐 layer 4，REPA 对齐层固定 layer 8），使 REPA 监督作用于已融入编码器结构的表示，避免干扰。
   - REPA 全程提供辅助对齐监督，与内化目标互补。

## 实验与结果
- **数据集**：ImageNet 256×256（主实验）、512×512、MS-COCO（文本到图像）。
- **模型**：SiT-B/L/XL、DiT-L、DiG-L；编码器默认 DINOv2-B/14，额外测试 MAE、MoCoV3、CLIP、WebSSL、I-JEPA-H、DeiT-III 共 10 种。
- **评估指标**：FID↓、sFID↓、IS↑、Precision↑、Recall↑（50K 样本，EMA 模型，遵循 ADM 协议）。
- **最强结果**（SiT-XL/2，ImageNet 256×256，无 CFG）：
  - Vanilla SiT（7M 步）：FID 8.30
  - REPA（100K 步）：FID 19.40；（400K 步）：FID 7.90
  - REPI（100K 步）：FID 14.51；（400K 步）：FID 7.21
  - **REPA + REPI（160K 步）：FID 8.22**（速度提升 >43.5×）
  - REPA + REPI（400K 步）：FID 6.33
- **有 CFG 实验**（SiT-XL/2，100K 步）：REPA+REPI 达到 FID 1.85，IS 278.4，优于所有基线。
- **文本到图像**（MM-DiT on MS-COCO，150K 步）：REPA+REPI 无 CFG FID 9.08，有 CFG FID 4.56，均优于 REPA 的 10.40/4.73。
- **通用性**：REPI 在所有 10 种编码器上均优于 REPA，且在 REPA 表现不如 vanilla SiT 的编码器（如 MAE-L、MoCoV3-L）上仍有显著提升。
- **训练开销**：REPI 相对 REPA 仅增加 3.2% 训练时间；REPA+REPI 共增加 3.9%，但 FID 相对 REPA 降低 39.3%，IS 提升 43.6%。

## 相关工作脉络
1. **REPA（Yu et al., 2025）**：将扩散 Transformer 隐藏状态投影到编码器空间对齐，本文沿相反方向（编码器→扩散）探索，形成互补机制。
2. **iREPA（Singh et al., 2026）**：强调空间结构对齐，本文与其在 Table 3 中对比；REPI 在相同步数下 IS 更高。
3. **sREPA（Xu et al., 2026）**：对齐关系几何，本文同样对比；REPI 在 SiT-XL/2 100K 步上 FID 14.51 vs sREPA 的 15.40。
4. **Stable Velocity（Yang et al., 2026）**：方差视角的 flow matching 改进，与 REPI 正交可叠加使用。
5. **U-repa（Tian et al., 2025）/ VideoREPA（Zhang et al., 2025）**：将表示对齐扩展到 U-Net 和视频生成，本文方法未来也可向此类方向推广。
6. **DiG（Zhu et al., 2024）**：门控线性注意力扩散模型，本文证明 REPI 可泛化到此类非 softmax 注意力架构（Figure 5）。

## 局限性与未来方向
1. **脚手架-内化超参仍需调优**：虽然 λ 在 1.0–4.0 区间鲁棒，但脚手架时长、注入层、λ 与 REPA 联合时的配合仍需手动设定（论文默认 20K 步脚手架、layer 8 或 layer 4）。
2. **仅验证图像生成任务**：未涉及视频生成、3D 生成等更复杂的时序/空间扩展场景。
3. **编码器依赖引入额外计算**：训练时需前向编码干净图像，虽开销小（+3.2%），但对大规模训练仍需考虑。
4. **论文自述未来方向**：向视频生成及其他生成领域扩展；探索更通用的外部表示利用方式。

## 研究启发与可借鉴点
1. **脚手架-内化范式可迁移**：任何"外部知识→内部能力"的训练模式（如教师模型蒸馏、多模态对齐）均可借鉴此"先注入引导、后移除+内化"的两阶段策略。
2. **K/V 联合注入保留信息流的精妙设计**：替换 Attention 内部的 K/V 而非整个 hidden state，既引入外部语义又保留噪声输入的 query 路由，这一设计思路可推广到其他注入场景（如文本条件、深度图引导）。
3. **Oracle 实验作为可行性验证的典范**：论文先以理想化 Oracle 设置揭示方法潜力（30K 步超越 7M 步 baseline），再推出实用版本，这种"理想化实验→工程落地"的研究路径值得参考。
4. **与既有方法（REPA）的组合策略**：通过层级顺序（REPI 在前、REPA 在后）避免互相干扰，同时利用两种监督信号互补，为方法组合提供了可复用的设计原则。
5. **跨架构通用性验证**：在 SiT/DiT/DiG 三种不同注意力机制的架构上均有效，显示方法具有广泛的适用潜力，可作为后续工作的统一基线。

## 关键术语表
**REPI（REpresentation Injection）**：将预训练视觉编码器表示注入扩散 Transformer 并内化的训练框架，与 REPA 方向相反且互补。
**脚手架-内化策略（Scaffold-to-Internalization）**：训练初期用外部表示作为临时脚手架引导模型学习，移除脚手架后通过内化损失使模型自主复现同等表示。
**REPA（Representation Alignment）**：通过将扩散 Transformer 中间隐藏状态投影到编码器空间进行对齐来加速训练的方法。
**K/V 注入（K/V Injection）**：将编码器最后注意力层的 Key/Value 投影后替换扩散层对应 K/V，保留原生 Query 以维持噪声输入信息流。
**内化损失（Internalization Loss）**：通过 stop-gradient 的 Frobenius 距离迫使扩散模型原生 K/V 逼近脚手架时期的编码器投影表示。
**Oracle 设置**：假设推理时可访问干净图像编码器表示的理想化实验设置，用于验证注入策略的可行性。
**Flow Matching**：将数据分布建模为从噪声到数据的连续轨迹，通过预测速度场进行去噪的训练范式。
**DINOv2**：微软提出的自监督视觉编码器，在本文作为主要的外部表示来源。

## 可复现要素
- **数据集**：ImageNet（公开）、MS-COCO（公开）；论文已声明遵循公开协议。
- **代码**：论文声明"Code will be available at https://jeneveuxpas.github.io/REPI"（截至发表时可能为计划中）。
- **权重**：论文未提及预训练权重开源，仅声明代码将开源。
- **关键超参**：
  - 脚手架时长：20K 步（所有实验一致）
  - 内化损失权重 λ：默认 2.0（无 REPA）/ 0.25（含 REPA）
  - 注入层：SiT-XL/2 默认 layer 8（单独 REPI）/ layer 4（REPI+REPA）
  - 投影层：单层 Linear
  - 优化器：AdamW，lr=1e-4，β₁=0.9，β₂=0.999，weight_decay=0
  - Batch size：256
  - 混合精度：fp16 + torch.compile
  - EMA：启用
  - 编码器：DINOv2-B/14（主实验）；其余编码器详见 Section 5.3
  - 采样：SDE Euler-Maruyama，NFE=250

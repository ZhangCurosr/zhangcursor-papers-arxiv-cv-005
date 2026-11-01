---
title: "Transforming-Image-Editors-into-Video-Editors"
source: https://arxiv.org/pdf/2610.11037v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-11-01 07:00:28"
field: "视频生成与编辑"
keywords: ["视频编辑", "图像编辑迁移", "两阶段框架", "关键帧插值", "扩散模型", "运动编码"]
innovations: ["将视频编辑分解为关键帧图像编辑+运动引导插值两阶段，避免端到端训练", "锚点约束扩散（Anchor Clamping）在去噪中锁定关键帧latent", "揭示视频编辑质量与图像编辑器质量强相关的规律"]
benchmarks: ["IVEBench", "VIE-Bench"]
---

# 论文速读：Transforming Image Editors into Video Editors

## 一句话总结
本文提出 AVE（Anchor-based Video Editing），一种两阶段视频编辑框架，通过将视频编辑拆解为"稀疏关键帧图像编辑"+"运动引导的关键帧插值"两个子任务，成功将一个强大的图像编辑器转化为视频编辑器，避免了昂贵且困难的端到端视频编辑训练。

## 研究问题与动机
- **视频编辑落后于图像编辑**：现代图像编辑已具备强大的语义理解、视觉保真度和指令遵循能力，但视频编辑在视觉质量、语义一致性和指令跟随方面仍明显滞后。
- **视频编辑训练成本高昂**：视频编辑需处理更长的时空上下文，且高质量视频编辑数据的收集远比图像困难（需要时间一致性、运动多样性、跨帧指令对齐）。
- **端到端范式并非最优**：现有视频编辑方法（如 VACE、Ditto、InsV2V）均需大量视频特定训练数据和计算资源，且编辑质量仍不及顶尖图像编辑器。
- **核心洞察**：视频编辑的主要难点本质上仍是 2D 图像级语义编辑，而时间维度的处理（保留/修改运动）只需极少的额外训练代价。

## 核心贡献（创新点）
- **提出 AVE 两阶段分解范式**：将视频编辑显式拆分为关键帧组合编辑与关键帧插值两个子问题，本质区别在于不追求端到端视频编辑，而是将最难部分委托给图像编辑器。
- **Stage-1 与 Stage-2 完全解耦**：只需训练轻量运动编码器（710M 参数，约占扩散模型 5%），所有其他组件保持冻结并从预训练模型初始化。
- **锚点约束扩散生成**：在去噪过程中将编辑后的关键帧latent特征锁定（clamp），仅对非锚点帧进行噪声预测与填充，确保锚点内容不变。
- **运动编码器采用因子化时空 Transformer**：6层空间-时间依次自注意力结构，通过 ControlNet 风格残差分支注入冻结的 Wan2.2 DiT。
- **揭示"图像编辑器质量强烈决定视频编辑质量"规律**：消融表明最终视频质量与 Stage-1 图像编辑器质量强相关，支持"视频编辑进步可来自更强图像编辑基础+轻量迁移"的核心主张。

## 方法详解
**整体流程**：输入视频 $V=\{x_t\}_{t=1}^T$ 和文本指令 $y$，输出编辑视频 $\hat{V}$。选取 $K$ 个关键帧索引集合 $\mathcal{A}$。

**Stage-1：组合图像编辑**
- 从视频数据构建多图像编辑训练样本 $(\mathbf{x}_\mathcal{A}, y, \mathbf{x}_\mathcal{A}^\star)$，约 1.5M 条。
- 基于 BAGEL 微调，使用标准扩散去噪目标：
$$\mathcal{L}_{\mathrm{img}} = \mathbb{E}_{\mathbf{z}_\mathcal{A}^\star, \epsilon, \tau}\left[\|\epsilon - \epsilon_\phi(\alpha_\tau \mathbf{z}_\mathcal{A}^\star + \sigma_\tau \epsilon, \mathbf{z}_\mathcal{A}, y, \tau)\|_2^2\right]$$
- 实践中默认使用 Nano Banana 2 API 进行单轮多轮对话编辑所有关键帧。

**Stage-2：关键帧插值**
- 以 Wan2.2-I2V-A14B 为骨干，3D VAE 和 DiT 完全冻结。
- **轻量运动编码器**：将视频 latent 投影到 motion embedding，经 6 层因子化时空 Transformer（先空间自注意力→后时间自注意力→MLP），输出 $h_\mathrm{mot}$，通过 ControlNet 残差分支注入 DiT。
- **锚点约束扩散**：定义二元锚点掩码 $M_\mathcal{A}$，初始 latent $X_S = M_\mathcal{A} \odot \hat{Z}_\mathcal{A} + (1-M_\mathcal{A}) \odot \epsilon$；每步去噪后重新施加锚点：$X_{s-1} = M_\mathcal{A} \odot \hat{Z}_\mathcal{A} + (1-M_\mathcal{A}) \odot \tilde{X}_{s-1}$。
- 训练损失仅计算在非锚点帧上：$\mathcal{L}_{\mathrm{vid}} = \|(1-M_\mathcal{A}) \odot (\hat{\epsilon} - \epsilon)\|_2^2$。

## 实验与结果
- **数据集**：IVEBench（短/长子集）、VIE-Bench（add/swap/remove/style 四类编辑）。
- **训练数据**：约 500K 视频编辑样本，64 张 A100 GPU，AdamW（lr=$1\times10^{-4}$，weight decay=0.05）。
- **IVEBench 短子集**：AVE 总分 70.58（previous best Ditto 66.75，提升 +3.83）；视频质量 82.10、指令合规 49.65、语义一致性 26.12、运动得分 89.26，均为最佳。
- **IVEBench 长子集**：AVE 总分 68.75（previous best Ditto 65.93，提升 +2.82）；运动得分 88.95 仍最佳。
- **VIE-Bench**：AVE 在 swap/change（9.448）、remove（9.352）、指令遵循（9.539）、保留（9.425）四项获最佳；整体质量 8.630，与最强开源基线持平。
- **消融关键发现**：① Stage-1 图像编辑器质量与最终视频质量强相关（Nano Banana 2 > GPT-5.1 > Seedream 4.0 > 微调 BAGEL > 原始 BAGEL）；② 关键帧策略鲁棒，最优预算约为 $\lfloor T/24 \rfloor$。

## 相关工作脉络
- **InstructPix2Pix / InstructEdit**：早期扩散图像编辑，本文与之互补而非竞争——不提出新图像编辑器，而是研究如何将成熟图像编辑能力迁移到视频。
- **Tune-A-Video / FateZero / TokenFlow / InstructVideo**：基于 inversion/单样本微调的早期视频编辑方法，本文相比它们无需对每个视频单独微调。
- **VACE / InsV2V / Ditto / Kiwi-Edit**：端到端视频编辑基线，依赖大量视频特定训练，本文通过两阶段分解避免了这一需求。
- **SAMA / ICVE / ReCo / OpenVE-3M**：并发工作，探索统一或 in-context  formulation，同样需要重训练；本文定位为轻量迁移方案。
- **AnimateDiff / VideoCrafter2 / VideoComposer**：图像到视频生成相关工作，本文继承其"内容-运动可分离"思想，但应用于编辑而非纯生成。
- **BAGEL / OmniGen / Step1X-Edit**：现代强图像编辑器，本文直接复用其能力作为 Stage-1，体现了"站在图像编辑肩膀上做视频编辑"的思路。

## 局限性与未来方向
- **关键帧数量需经验调优**：过多关键帧会压缩插值空间降低稳定性，过少则指令遵循不足，最优约 $\lfloor T/24 \rfloor$。
- **依赖 Stage-1 图像编辑器质量**：若图像编辑器本身弱，视频质量显著下降；无法弥补图像编辑器的固有缺陷。
- **闭源 API 依赖**：实践中最优效果来自 Nano Banana 2 API，开源场景下需依赖微调的 BAGEL，性能有差距。
- **未探索更长视频和更复杂运动编辑**：当前实验集中在指令跟随和外观/场景编辑，对极端运动变换的泛化有待验证。
- **潜在方向**：更强的图像编辑器→直接提升视频编辑上限；自适应关键帧选择策略；扩展至三维/多视角编辑。

## 研究启发与可借鉴点
- **"分解而非端到端"的设计哲学**：将复杂 3D 任务拆解为已成熟的 2D 子任务 + 轻量时序模块，可迁移至 3D 生成、多模态编辑等领域。
- **锚点约束扩散（Anchor Clamping）**：在去噪过程中锁定部分帧 latent，教模型"填充而非重绘"，是一种简洁有效的时序一致性保证机制。
- **图像质量→视频质量的强相关性发现**：启示团队可优先投资图像编辑能力，再通过轻量迁移获得视频能力，性价比远高于端到端训练。
- **因子化时空 Transformer 作为运动编码器**：先空间后时间的自注意力结构计算高效，可作为通用运动表征模块复用到其他视频生成/编辑任务。
- **与 Seedream 团队方向的结合机会**：可将 AVE 框架与 Seedream 的图像编辑能力对接，验证"图像编辑器越强、视频编辑越好"的假设。

## 关键术语表
- **AVE（Anchor-based Video Editing）**：本文提出的两阶段视频编辑框架，通过关键帧锚点将图像编辑能力迁移到视频。
- **Composed Image Editing**：多图像联合编辑，同时对多个关键帧执行一致编辑操作。
- **Anchor Clamping**：在扩散去噪过程中将关键帧的 latent 特征锁定不变，仅对非锚点帧进行去噪填充。
- **Motion Encoder**：轻量级因子化时空 Transformer，从源视频提取运动表征以指导视频插值。
- **ControlNet-style Residual Branch**：将运动特征以残差方式注入冻结的主干网络，实现条件控制而不破坏预训练权重。
- **IVEBench / VIE-Bench**：两个视频编辑评测基准，分别评估指令跟随视频编辑的短期/长期一致性及多种编辑类型。

## 可复现要素
- **数据集**：IVEBench（公开）、VIE-Bench（公开）；训练数据为私有视频编辑数据集（未公开），约 500K-1.5M 样本。
- **代码**：已开源，https://github.com/wangf3014/AVE。
- **权重**：Wan2.2-I2V-A14B 预训练权重公开；运动编码器权重随代码开源。
- **关键超参**：学习率 $1\times10^{-4}$，weight decay 0.05，$\beta_1=0.9, \beta_2=0.999$， cosine lr schedule + linear warmup，64 张 A100，混合精度训练；关键帧预算默认 $\lfloor T/24 \rfloor$；Stage-1 默认使用 Nano Banana 2 API。

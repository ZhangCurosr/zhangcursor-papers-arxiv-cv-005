---
title: "SELECTIVE-CHANNEL-RESTORATION-FOR-BACKDOORED-VISION-LANGUAGE"
source: https://arxiv.org/pdf/2609.37759v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-10-01 17:43:06"
field: "多模态大模型安全"
keywords: ["vision-language models", "backdoor defense", "post-training purification", "projection fragility", "channel restoration"]
innovations: ["发现投影脆弱性：后门VLM投影接口对微小有界扰动比干净模型更敏感，提供无触发器的通道选择信号", "提出PSR三步后训练净化框架，通过扰动探测-通道选择-预训练恢复实现零推理开销后门抑制", "PSR对三类递进式自适应攻击（PA/PA+/RA）仍保持有效防御"]
benchmarks: ["MS-COCO (image captioning)", "VQAv2 (visual question answering)"]
---

# 论文速读：SELECTIVE-CHANNEL-RESTORATION-FOR-BACKDOORED-VISION-LANGUAGE

## 一句话总结
本文提出 PSR（Perturb-Select-Restore），一种针对视觉语言模型（VLM）后门攻击的后训练净化方法，通过检测投影接口的"投影脆弱性"识别并恢复受后门污染的输出通道，在几乎不损失干净任务性能的前提下将攻击成功率降至接近零。

## 研究问题与动机
- VLM 通过参数高效微调投影接口实现多模态适配，但使用该过程中的 poisoned 数据会植入后门，触发器出现时产生攻击者指定响应。
- 现有 VLM 后门防御分为训练时净化（需干预微调过程）和推理时干预（引入 per-query 计算开销），缺乏高效的**后训练单步净化**方案。
- 通用后训练防御（Fine-Pruning、CLP 等）面向图像分类器设计，直接应用于 VLM 投影层会破坏视觉-语言对齐，损害干净任务性能。
- 作者发现后门 VLM 投影接口对微小有界扰动的敏感性显著高于干净模型（投影脆弱性），该现象可作为无触发器数据的通道选择信号。

## 核心贡献（创新点）
- **发现投影脆弱性（Projection Fragility）**：后门 VLM 投影接口在干净数据上的任务损失对微小有界扰动更敏感，为无需触发器的通道级净化提供新信号。
- **提出 PSR 通道级后训练净化框架**：通过交替优化有界扰动与通道鲁棒性分数来定位后门相关通道，再将其权重恢复至预训练值，避免对已适配参数的广泛修改。
- **跨架构/跨任务/跨攻击的一致有效性**：在 LLaVA-1.5 与 Qwen3-VL 两种架构、COCO 与 VQAv2 两个任务、六种攻击下的 ASR 均降至 ≤1.70%，且保持高干净利用率。
- **对自适应攻击具有鲁棒性**：即使攻击方知晓 PSR 机制并进行针对性训练（PA/PA+/RA），PSR 仍能保持有效的后门抑制。

## 方法详解
- **投影接口参数化**：对目标线性层 $\ell$，输出变为 $\tilde{\mathbf{u}}^{(\ell)} = \mathbf{u}^{(\ell)} \odot (\pmb{\theta}^{(\ell)} \odot (\mathbf{1} + \pmb{\delta}^{(\ell)}))$，其中 $\pmb{\theta}$ 为可学习的通道保留分数向量（∈[0,1]），$\pmb{\delta}$ 为有界辅助扰动向量（∈[-ε,ε]）。
- **内层最大化（扰动探测）**：固定 $\pmb{\theta}$，通过投影符号梯度上升寻找使干净损失增大的 $\pmb{\delta}^*$，步长为 $2ε/T_{\mathrm{PGD}}$，每步后投影到 $[-ε, ε]$ 约束。
- **外层最小化（鲁棒性分数学习）**：目标函数为 $\mathcal{L}_{\mathrm{clean}}(\pmb{\theta}, \pmb{\delta}^*; \mathcal{C}) + λ_{\mathrm{clean}} \mathcal{L}_{\mathrm{clean}}(\pmb{\theta}, \mathbf{0}; \mathcal{C}) + λ_θ \| \pmb{\theta} \|_1$，受约束于 $\pmb{\theta} ∈ [0,1]$；低分数通道即对扰动更敏感，被判定为可疑。
- **通道选择与参数恢复**：每层独立选取 Top-K 最小 $\pmb{\theta}$ 值对应的通道（恢复比例 $r$），将其权重行与偏置从预训练基模型 $\phi_0$ 恢复，其余参数保持微调后值不变。
- **超参设置**：ε=0.016，$T_{\mathrm{PGD}}=2$，$η_θ=0.06$，$λ_{\mathrm{clean}}=0.5$，$λ_θ=0.006$，$T_{\mathrm{opt}}=1000$，推荐 $r=30\%$；仅依赖 256 条干净校准数据。

## 实验与结果
- **模型**：Qwen3-VL-8B-Instruct（含 Visual Merger + DeepStack）、LLaVA-1.5-7B（两层 MLP 投影器）。
- **数据集**：MS-COCO（图像描述，CIDEr 指标）、VQAv2（视觉问答，准确率指标）。
- **攻击方法**：BadNet、Blended、ISSBA、WaNet、TrojVLM、VLOOD（目标响应均为 "you have been hacked lol"）。
- **基线方法**：Clean FT（干净数据再微调）、Random（随机通道恢复）、FP（Fine-Pruning）、CLP（基于 Lipschitz 的通道选择）。
- **主要结果（COCO）**：PSR 在所有攻击下 ASR ≤1.70%（Qwen3-VL）/ ≤1.60%（LLaVA-1.5），CU（CIDEr）保持在 120~127 区间；对比基线中 FP 对 TrojVLM 仍有 40% ASR，CLP 对 WaNet 仍有 45% ASR。
- **主要结果（VQAv2）**：PSR ASR ≤0.62%（Qwen3-VL）/ ≤1.40%（LLaVA-1.5），干净准确率保持在 73~80%。
- **消融**：恢复比 $r$ 控制安全-效用权衡（$r=100\%$ 时 ASR=0 但 CIDEr 从 ~125 降至 68.8）；两层 Merger 需联合恢复才有效；从 $\phi_0$ 恢复优于设为零。
- **自适应攻击**：即使在 PA/PA+/RA 三种强对抗训练下，PSR 仍能将多数攻击 ASR 降至接近零。

## 相关工作脉络
- **Fine-Pruning（Liu et al., 2018）**：基于干净数据激活稀疏性剪除神经元，未考虑 VLM 投影层的维度结构，PSR 扩展至通道级并配合预训练权重恢复。
- **CLP（Zheng et al., 2022）**：基于权重的 Lipschitz 近似选择通道，仅依赖 poisoned 权重无需校准数据；PSR 使用扰动探测信号，在 VLM 投影层上能更好区分后门与正常通道。
- **Neural Cleanse（Wang et al., 2019）**：检测异常触发器特征，需要反向搜索触发器模式，PSR 完全不依赖触发器先验，仅用干净数据。
- **RNP/TSBD（Li et al., 2023; Lin et al., 2024）**：基于 unlearning 信号定位后门组件，PSR 采用扰动敏感性作为更直接的 channel-wise 代理。
- **BYE/RobustIT（Rong et al., 2025; Xun et al., 2025）**：训练时后门防御，需干预微调过程；PSR 为后训练方法，无需重训练。
- **PurMM/SRD/CleanSight（Jiang et al., 2026; Xu et al., 2026; Zhang et al., 2026）**：推理时防御，引入 per-query 开销；PSR 实现零推理额外开销。

## 局限性与未来方向
- 仅评估了投影接口微调注入的后门，未覆盖视觉编码器或语言模型层后门。
- 恢复比 $r$ 需手动调节，存在安全-效用权衡，高恢复比会损失干净任务性能。
- 实验仅覆盖两种架构与两类任务（图像描述、VQA），未验证视频/多轮对话等场景。
- 需要访问预训练基模型权重 $\phi_0$ 与少量干净校准数据，在实际部署中可能受限。
- 未讨论对抗样本攻击与非触发器型后门（如模型窃取）的防护效果。

## 研究启发与可借鉴点
- **投影脆弱性信号**：将扰动敏感性作为后门通道的代理指标，可迁移至其他模态对齐模块（如音频-文本 projection）的净化。
- **内外层交替优化框架**：内层对抗扰动+外层鲁棒性学习的 MIN-MAX 范式，可作为通用通道重要性度量器，适用于模型压缩与鲁棒性分析。
- **通道级参数恢复策略**：相比全模型微调或逐神经元剪枝，恢复至预训练值能更好保留通用能力，可推广至 LoRA/Adapter 等参数高效微调结构的净化。
- **自适应攻击评估范式**：PA/PA+/RA 三层递进式对抗训练评估体系，可作为防御论文的标准安全测试协议。
- **无触发器后门检测**：PSR 完全不依赖触发器特征，对 ISSBA、VLOOD 等隐式触发器攻击同样有效，启示可探索更广泛的黑盒后门检测场景。

## 关键术语表
- **Projection Fragility（投影脆弱性）**：后门 VLM 投影接口对微小有界扰动的干净损失敏感度显著高于干净模型的现象，是 PSR 通道选择的核心信号。
- **PSR（Perturb-Select-Restore）**：三步后训练净化流程——有界扰动探测、通道鲁棒性分数学习、选定通道参数恢复至预训练值。
- **ASR（Attack Success Rate）**：被触发输入成功生成目标响应的比例，越低表示防御效果越好。
- **Clean Utility（CU）**：干净输入上的任务性能（CIDEr 或 VQA 准确率），越高表示防御对正常功能影响越小。
- **Channel-wise Robustness Score（通道鲁棒性分数 θ）**：可学习的 [0,1] 区间标量，值越小表示该通道对扰动越敏感、越可能携带后门。
- **DeepStack Merger**：Qwen3-VL 中用于多深度注入视觉信息的投影组件，PSR 同时覆盖多个 DeepStack 层。
- **Top-K smallest θ**：每层独立选取鲁棒性分数最小的前 k 个通道进行恢复，k 由恢复比例 r 决定。
- **Adaptive Attack（自适应攻击）**：攻击方知晓防御机制并在训练阶段针对性优化的后门，PSR 对其仍保持有效性。

## 可复现要素
- **数据集**：MS-COCO（公开）、VQAv2（公开），训练集 118K 图像，验证集 5K；校准集 256 条干净样本（论文未说明是否公开来源）。
- **模型**：Qwen3-VL-8B-Instruct（公开）、LLaVA-1.5-7B（公开）。
- **代码/权重**：论文未声明代码开源。
- **关键超参**：ε=0.016，$T_{\mathrm{PGD}}=2$，$η_θ=0.06$，$λ_{\mathrm{clean}}=0.5$，$λ_θ=0.006$，$T_{\mathrm{opt}}=1000$，$r=30\%$（推荐），校准集大小 256。

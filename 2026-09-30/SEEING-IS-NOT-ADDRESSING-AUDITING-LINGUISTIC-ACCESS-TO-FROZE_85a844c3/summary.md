---
title: "SEEING-IS-NOT-ADDRESSING-AUDITING-LINGUISTIC-ACCESS-TO-FROZE"
source: https://arxiv.org/pdf/2609.37230v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-10-01 17:42:54"
field: "视觉-语言模型评估与对齐"
keywords: ["vision-language models", "cross-modal retrieval", "visual discriminability", "linguistic addressability", "matched visual grounding", "compositional retrieval", "FactorAtlas"]
innovations: ["提出视觉可区分性与语言可及性的二分诊断框架", "设计目标vs均值其余的匹配视觉接地方法及几何保证", "证明全局对齐后仍存在值级特异性可及差距，匹配接地可进一步弥补"]
benchmarks: ["FactorAtlas", "Winoground", "VL-CheckList", "COCO-Facet", "Fashionpedia", "DTD", "UT-Zappos"]
---

# 论文速读：SEEING-IS-NOT-ADDRESSING-AUDITING-LINGUISTIC-ACCESS-TO-FROZE

## 一句话总结
本文揭示了冻结视觉语言模型（VLM）中"视觉可区分性"与"语言可及性"之间的系统性差距，并提出**匹配视觉接地（matched visual grounding）**方法——通过从校准图像中提取目标值相对于其他值的视觉对比方向来调整原生文本查询，从而在多个因子、骨干网络和真实图像上显著缩小这一差距。

## 研究问题与动机
- 视觉表示通常保留比语言概念化更精细的区分，VLM 共享嵌入空间中同样存在此不对称性：图像几何中可区分的视觉差异，通过原生文本接口却难以有效检索。
- 现有跨模态对齐工作（如全秩线性映射、Procrustes 对齐）侧重全局修正，但未考察**每个具体视觉区分**是否能在冻结表示中被语言接口可靠访问。
- 已有研究（如 Winoground、VL-CheckList）表明 VLM 在细粒度属性、关系绑定和组合检索上表现不足，但其根因是图像侧判别力不足还是文本侧可及性不足尚未分离诊断。
- 本文以命名视觉区分为分析单元，将"图像几何是否支持该区分"与"原生文本查询能否访问它"解耦为两个独立可测的问题。

## 核心贡献（创新点）
1. **提出视觉可区分性与语言可及性的二分框架**：以 FactorAtlas 测试集为平台，首次在同一组图像上分别用图像派生查询和原生文本查询测量同一视觉区分，揭示了二者系统性分叉。
2. **设计匹配视觉接地（matched visual grounding）方法**：对每个因子值构建目标值 vs 其余值的均值视觉对比方向 $d_v^I$，将该方向沿原生文本查询叠加（$q'_v(\alpha) = \text{norm}(q_v + \alpha d_v^I)$），并提供严格的相似性边界单调性证明（Proposition 1）。
3. **提供全面的对照实验证明增益的针对性**：原型吸引（prototype-only grounding）、非匹配方向控制、提示词汇变体、视觉信号衰减实验均表明增益来自匹配的"目标 vs 均值其余"方向而非通用吸引力或词汇替换。
4. **证明匹配接地在全局对齐后仍有残余增益**：即使在 factor-specific 全秩线性映射（macro mAP 0.716）之后，匹配接地仍可进一步提升至 0.806，说明全局对齐无法消除值级特异性差距。
5. **验证方法向组合检索与自然图像的泛化**：匹配接地使精确组合检索 R@1 从 0.357 提升至 0.711（context holdout），并在 DTD、Fashionpedia、COCO-Facet 等真实图像数据集上持续获得增益；FactorAtlas 视觉方向甚至可在零适配条件下直接复用至 Fashionpedia。

## 方法详解
**匹配视觉接地（Matched Visual Grounding）**：
- 设冻结图像嵌入为 $I(x) \in \mathbb{R}^d$，因子值集 $\mathcal{V}$，值 $v$ 的视觉支持集为 $C_v$。
- 计算目标原型与平均其余原型：$\mu_v^I = \text{norm}\!\left(\frac{1}{|C_v|}\sum_{x\in C_v}\text{norm}(I(x))\right)$，$\bar{\mu}_{-v}^I = \frac{1}{|\mathcal{V}|-1}\sum_{u\neq v}\mu_u^I$（未归一化）。
- 视觉对比方向：$\Delta_v^I = \mu_v^I - \bar{\mu}_{-v}^I$，$d_v^I = \Delta_v^I / \|\Delta_v^I\|$。
- 接地后查询：$q_v'(\alpha) = \text{norm}(q_v + \alpha d_v^I)$，其中 $\alpha \geq 0$ 为接地强度，在验证集上按 macro-mAP 选定。
- **几何保证（Proposition 1）**：设 $M_v(q) = q^\top \Delta_v^I$，$c = q_v^\top d_v^I$，则 $M_v(q_v'(\alpha)) = \|\Delta_v^I\|\frac{c+\alpha}{\sqrt{1+\alpha^2+2\alpha c}}$；当 $|c|<1$ 时该值随 $\alpha$ 严格递增；若初始差距为负（$-1<c<0$），则在 $\alpha^* = -c$ 处跨越零。

**评估协议（FactorAtlas）**：
- 23,040 张图，8 形状 × 12 色相 × 10 图案 × 24 干扰实现（12 上下文 × 2 renderer seed）。
- 4 上下文作校准，2 个作验证，6 个作 Held-out 测试，语义因子完全交叉平衡。
- 原生查询为 3 个固定提示模板嵌入的平均+归一化（如图案："a {v} surface"、"an object with a {v} pattern"、"{v}"）。

## 实验与结果
**主要数据集**：FactorAtlas（自建，程序化渲染，23,040 图像）；自然图像验证使用 DTD、Fashionpedia、COCO-Facet、UT-Zappos。

**评估基线**：原生文本检索、原型吸引（prototype-only grounding）、全秩线性全局对齐（shared/factor-specific map）、Tip-Adapter 风格缓存适配、Procrustes 闭式映射。

**最强结果（SigLIP2 Base，context holdout）**：
| 因子 | 原生 mAP | 匹配接地 mAP | 提升 |
|---|---|---|---|
| Pattern | 0.635 | **0.878** | **+0.243** |
| Hue | 0.678 | **0.854** | **+0.176** |
| Shape | 0.702 | **0.813** | **+0.111** |

**七骨干宏观（context holdout）**：Pattern 0.567→0.862（+0.295），Hue 0.619→0.786（+0.167），Shape 0.660→0.803（+0.143），21 个 backbone×factor 组合全部正向。

**精确组合检索（SigLIP2 Base，context holdout）**：R@1 0.357→0.711，R@5 0.714→0.962，mAP 0.517→0.819。

**全局对齐后残余增益**：Factor-global 对齐 mAP 0.716 → +匹配接地 0.806；共享全局对齐 0.682 → +匹配接地 0.802，所有 21 个组合均正向改善。

**自然图像迁移**：COCO-Facet 材料 0.624→0.860（+0.236），Fashionpedia 纹理 0.262→0.369（+0.107），DTD 纹理 0.526→0.760（+0.234）。

**跨数据集零适配复用**：FactorAtlas 视觉方向直接用于 Fashionpedia 图案检索（无目标域拟合），macro mAP 0.583→0.719（+0.136）。

## 相关工作脉络
- **LABCLIP (Koishigarina et al., 2026)**：通过跨文本嵌入的全局线性变换改进属性-对象对应；本文定位为值级特异性干预，证明全局对齐后仍有残余差距。
- **Tip-Adapter (Zhang et al., 2021)**：基于缓存的免训练适配；本文 Full-support cache 对照显示，虽然可获得大幅增益（0.616→0.755），但匹配接地（0.817）仍更优且无需保留校准集作为推理时缓存。
- **Factored Inference (Alshehri et al., 2026)**：因子化推理用于双编码器 VLM 组合检索；本文在此基础上证明因子级可及性改善可直接提升精确组合检索性能。
- **Visual embedding ordinal structure (Sonthalia et al., 2026; Berasi et al., 2025)**：揭示冻结图像嵌入保留组合与排序结构；本文在此基础上进一步区分"图像侧结构存在"与"文本侧可访问"为两个独立维度。
- **Global cross-modal alignment (Moayeri et al., 2023; Liang et al., 2022)**：研究全局跨模态不对齐；本文证明全局纠正无法消除值级特异性可及差距，二者互补而非替代。

## 局限性与未来方向
- 几何保证仅针对校准侧目标-均值其余相似性边界成立，不保证泛化到未见组合或组合检索（后者需实证验证）。
- 视觉方向的估计依赖于带标签校准支持；当支持极有限（n=1）时方向余弦仅 0.466，稳定性不足。
- 全局对齐仅考察了全秩线性映射一类修正，未覆盖非线性或其他对齐形式。
- 方法目前针对预定义离散因子值，未扩展到自由形式视觉区分或连续属性。
- 未来方向包括：扩展至无标签视觉结构的自动发现、探索连续属性接地、在非冻结 VLM 中研究可微版本。

## 研究启发与可借鉴点
1. **"可区分性 vs 可及性"的二分诊断思路**可迁移至任何 VLM 评估场景：对于检索失败案例，先判断是图像侧信息缺失还是文本侧访问不足，再决定干预策略。
2. **目标 vs 均值其余的对比方向构造**（$\mu_v - \bar{\mu}_{-v}$）是一种简单而有效的视觉 contrast 提取方式，比单纯原型吸引更具判别性，可推广至属性绑定、细粒度分类等任务。
3. **验证集引导的选择性接地策略**（bootstrap 置信区间筛选）值得借鉴：不是对所有值统一干预，而是基于验证证据选择性应用，以最小干预成本获取最大增益。
4. **跨数据集视觉方向复用实验**提供了低资源场景下的实用方案：在源域（如合成数据）估算视觉对比方向，零适配迁移至目标域，可大幅减少标注需求。
5. **完整消融设计范式**（方向特异性、提示鲁棒性、词汇替代、视觉信号衰减、有限支持）可作为视觉-语言模型诊断实验的参考模板。

## 关键术语表
**Visual Discriminability**：冻结图像几何中某视觉区分相对于其他值的可区分程度，由图像派生查询衡量。
**Linguistic Addressability**：原生文本查询检索具有某视觉值的图像的能力，由文本编码器输出衡量。
**Matched Visual Grounding**：将原生文本查询沿目标值 vs 均值其余值的视觉对比方向 $d_v^I$ 进行叠加调整（$q_v'(\alpha)$），以弥合语言可及性差距。
**FactorAtlas**：程序化渲染测试集，含 23,040 张图像（8 形状×12 色相×10 图案×24 干扰实现），用于解耦测量视觉可区分性与语言可及性。
**Prototype-only Grounding**：将文本查询向目标值原型$\mu_v^I$插值（$q_v^+(\lambda)$），但不构造目标-其余对比，作为匹配接地的弱对照。
**Exact Compositional Retrieval**：查询指定多个因子值（形状+色相+图案），正确图像须同时满足所有因子，按 softmax 因子似然加权组合评分。

## 可复现要素
- **数据集**：FactorAtlas 为论文自建程序化数据集；自然图像测试使用 DTD、Fashionpedia、COCO-Facet、UT-Zappos（均为公开数据集）。论文未声明 FactorAtlas 代码/数据公开。
- **代码/权重**：论文未明确声明开源代码仓库；使用的骨干网络权重为官方预训练模型（openai/clip-vit-base-patch16、google/siglip2-* 系列等）。
- **关键超参**：$\alpha \in \{0, 0.05, \ldots, 8\}$（由验证集 macro-mAP 选定）；$\lambda \in \{0, 0.0125, \ldots, 1\}$（原型吸引）；温度 $\tau \in \{0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5\}$；全局对齐学习率 $\{10^{-4}, 3\times10^{-4}, 10^{-3}\}$，AdamW，30 epoch，batch size 256。

---
title: "SEEING-IS-NOT-ADDRESSING-AUDITING-LINGUISTIC-ACCESS-TO-FROZE"
source: https://arxiv.org/pdf/2609.37230v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-10-01 17:42:43"
field: "视觉-语言模型细粒度对齐与检索"
keywords: ["vision-language models", "cross-modal retrieval", "visual discriminability", "linguistic addressability", "matched visual grounding", "compositional retrieval"]
innovations: ["提出视觉可判别性与语言可及性分离的审计框架，揭示冻结VLM中表示保留与语言访问的差距", "发明匹配视觉接地方法，通过目标对平均其余的逐值视觉对比方向增强原生文本查询", "证明全局对齐后仍存在值特定残余访问缺口，匹配接地可进一步消除"]
benchmarks: ["FactorAtlas", "DTD", "Fashionpedia", "COCO-Facet", "UT-Zappos"]
---

# 论文速读：SEEING-IS-NOT-ADDRESSING-AUDITING-LINGUISTIC-ACCESS-TO-FROZE

## 一句话总结
本文揭示了冻结视觉-语言模型中视觉可判别性与语言可及性的分离现象——某些视觉区分在图像几何中仍可判别，但原生文本查询难以有效访问；通过"匹配视觉接地"（matched visual grounding）方法利用图像侧的逐值视觉对比方向增强文本查询，在多个骨干模型和保持外设置上均显著提升检索性能。

## 研究问题与动机
- 视觉表征往往比语言概念化保留更细粒度的区分，对比训练的视觉-语言模型也存在类似不对称性：图像中表示的某些区分可通过文本接口弱访问。
- 现有工作关注跨模态对齐的整体改善（如LABCLIP等），但未从逐值层面区分视觉可判别性与语言可及性，导致无法诊断具体哪些视觉区分的访问存在缺口。
- 全 rank 线性映射等全局对齐方法虽能提升整体检索性能，但未检验其是否消除了值特定的残余访问缺口。
- 视觉-语言检索失败是否意味着图像表示中缺乏对应区分，还是一个独立的"访问"问题，尚不明确。

## 核心贡献（创新点）
1. **提出"视觉可判别性 vs 语言可及性"的审计框架**：将同一视觉属性的两个方面分开度量，区别于以往仅评估整体检索性能的工作。
2. **发明"匹配视觉接地"干预方法**：为每个视觉值构建目标对平均其余的视觉对比方向，并沿该方向调节原生文本查询，本质区别在于针对逐值视觉对比而非简单向目标原型移动。
3. **建立了几何保证（Proposition 1）**：证明沿匹配对比方向移动查询时，目标与其余平均的相似度边际不会下降。
4. **构建了FactorAtlas可控测试床**：23,040张图像在8形状×12色调×10图案×24干扰条件下完全交叉，支持严格的保持外评估协议。
5. **证明匹配接地增益的稳健性与泛化**：跨越7个骨干模型、3种因子、多种保持外设置、全局对齐残余后、组合检索以及自然图像均有效。

## 方法详解
- **匹配视觉接地核心公式**：给定冻结图像嵌入 $I(x)$，对因子值 $v$ 构建目标原型 $\mu_v^I = \text{norm}(\frac{1}{|C_v|}\sum_{x \in C_v}\text{norm}(I(x)))$ 与平均其余原型 $\bar{\mu}_{-v}^I = \frac{1}{|\mathcal{V}|-1}\sum_{u \neq v}\mu_u^I$，定义匹配视觉对比 $\Delta_v^I = \mu_v^I - \bar{\mu}_{-v}^I$ 及其单位方向 $d_v^I = \Delta_v^I / \|\Delta_v^I\|$。
- **接地查询构造**：$q_v'(\alpha) = \text{norm}(q_v + \alpha d_v^I)$，其中 $\alpha \geq 0$ 控制接地强度，通过验证集选择。
- **几何保证**：对于任意 $\alpha \geq 0$，有 $M_v(q_v'(\alpha)) \geq M_v(q_v)$，其中 $M_v(q) = q^\top \Delta_v^I$ 为相似度边际；当初始边际为负时，$\alpha^* = -c = -q_v^\top d_v^I$ 使其过零点。
- **评估协议**：4个上下文用于校准（估计视觉方向）、2个用于验证（选择 $\alpha$）、6个用于测试（保持外），语义因子完全交叉平衡，排除频率捷径。

## 实验与结果
- **数据集**：FactorAtlas（23,040张程序渲染图像），以及DTD、Fashionpedia、COCO-Facet、UT-Zappos等自然图像数据集。
- **骨干模型**：CLIP ViT-B/16、EVA02-B/16、FG-CLIP Base、SigLIP SO400M、SigLIP2 Base/Large/SO400M（共7个）。
- **主要结果（七骨干宏平均，context holdout）**：
  - Shape: mAP 0.660 → 0.803，Top-1 0.678 → 0.779
  - Hue: mAP 0.619 → 0.786，Top-1 0.599 → 0.813
  - Pattern: mAP 0.567 → 0.862，Top-1 0.598 → 0.896
- **最强结果**：SigLIP2 SO400M + Pattern: mAP 0.621 → 0.867，Top-1 0.676 → 0.914。
- **组合检索（context holdout）**：R@1从0.357提升至0.711，R@5从0.714提升至0.962，mAP从0.517提升至0.819。
- **控制实验**：prototype-only grounding持续低于matched grounding；mismatched direction sharply degrade retrieval；prompt模板敏感性测试中所有5种模板均获提升；视觉证据衰减实验中 $\gamma=0.1$ 时仍保持0.659 mAP（vs native 0.416）。
- **全局对齐后残余增益**：factor-global alignment后macro-mAP从0.716进一步提升至0.806。
- **自然图像转移**：COCO-Facet material mAP 0.624 → 0.860；Fashionpedia garment length 0.262 → 0.369。

## 相关工作脉络
- **CLIP/SigLIP等视觉-语言模型**（Radford et al., 2021; Zhai et al., 2023）：本文定位为补充其细粒度检索缺陷的审计视角，而非改进模型本身。
- **Winoground/VL-CheckList/SugarCrepe/COLA**（Thrush et al., 2022; Zhao et al., 2022; Hsieh et al., 2023; Ray et al., 2023）：这些基准揭示了VLM在细粒度属性、关系、绑定上的局限，本文从访问缺口角度重新解释此类失败。
- **LABCLIP**（Koishigarina et al., 2026）：学习跨文本嵌入的全局变换改善属性-对象对应；本文定位差异在于value-specific而非全局变换。
- **Tip-Adapter等支持基适应**（Zhang et al., 2021）：使用标记视觉示例改进下游性能；本文方法无需保留校准集作为推理时缓存即可获得更大全图库mAP增益。
- **线性跨模态对齐工作**（Moayeri et al., 2023; Liang et al., 2022）：本文证明即使在全局对齐后，值特定视觉接地仍有残余增益。

## 局限性与未来方向
- 几何保证仅局部适用于匹配的目标对平均其余边际，不保证单调的保持外检索改善。
- 访问增益依赖于视觉可判别性；当视觉区分消失时（$\gamma=0$）增益消失。
- 全局对齐测试仅覆盖全rank线性映射一族，未穷尽所有可能的全局校正方法。
- 当前分析限于预定义区分和标记视觉支持；扩展到无标签、自由形式的视觉结构利用是未来方向。
- 有限校准支持（如单样本）下方向估计不稳定（与全支持方向余弦仅0.466）。

## 研究启发与可借鉴点
1. **"可判别性 vs 可及性"分离审计思路**：可将此框架迁移到其他模态对（如音频-文本、多模态大模型内部层表示）诊断信息保留与访问的差距。
2. **匹配视觉接地作为即插即用模块**：方法仅需冻结骨干和少量校准图像，无需微调，可集成到现有VLM检索管线中提升细粒度属性检索。
3. **因子级选择性接地策略**：通过验证集Bootstrap置信区间决定哪些值需要接地，避免对已有效查询的干扰，该思想可用于自适应干预选择。
4. **跨数据集视觉方向复用**：FactorAtlas推导的方向可直接用于Fashionpedia的提升，表明受控测试床的视觉结构可迁移到真实场景。
5. **精确组合检索的提升**：因子级接地增益可传递到多条件组合检索，提示细粒度属性修正可能系统性改善VLM的组合推理能力。

## 关键术语表
**Visual discriminability**：在冻结图像几何中可靠区分某视觉值与其替代值的能力，通过图像派生查询度量。
**Linguistic addressability**：原生文本查询检索展示某视觉值的图像的有效性，反映语言接口对视觉区分的访问能力。
**Matched visual grounding**：利用目标对平均其余的视觉对比方向调节原生文本查询的干预方法，公式为 $q_v'(\alpha) = \text{norm}(q_v + \alpha d_v^I)$。
**FactorAtlas**：包含23,040张程序渲染图像的测试床，完全交叉8形状、12色调、10图案和24干扰实现。
**Matched access gain**：匹配视觉接地相对于原生文本查询在保持外图像上获得的检索性能提升。
**Target-versus-rest contrast**：目标值原型与所有其他值原型平均之间的向量差，编码该值的视觉区分信息。
**Exact compositional retrieval**：查询指定多个因子值（如形状+色调+图案），正确图像需同时匹配所有值且排在near-miss之前。
**Global alignment**：通过全rank线性映射（如 $q^G = \text{norm}(Aq)$）对文本表示进行全局变换以改善跨模态对齐。

## 可复现要素
- **数据集**：FactorAtlas由作者程序生成（附录B详述渲染参数）；自然图像使用DTD、Fashionpedia、COCO-Facet、UT-Zappos公开数据集。
- **代码/权重**：使用官方模型（openai/clip-vit-base-patch16, google/siglip2-*等），论文未明确声明开源代码仓库。
- **关键超参**：$\alpha \in \{0, 0.05, \ldots, 8\}$ 通过验证集macro-mAP选择；温度 $\tau \in \{0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5\}$；因子权重 $w_f$ 在0.1网格上选择。

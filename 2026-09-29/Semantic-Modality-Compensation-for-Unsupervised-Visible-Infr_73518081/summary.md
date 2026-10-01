---
title: "Semantic-Modality-Compensation-for-Unsupervised-Visible-Infr"
source: https://arxiv.org/pdf/2609.34294v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-10-01 03:47:50"
---

# 论文速读：Semantic-Modality-Compensation-for-Unsupervised-Visible-Infr

## 一句话总结
本文提出语义模态补偿（SMC）框架，将非配对场景下的无监督可见光-红外行人重识别（USL-VI-ReID）建模为语义补偿问题，通过在CLIP共享语义空间中解耦身份语义与模态风格并重组，为跨模态缺失身份生成可靠的语义替身，显著缓解身份重叠不全导致的跨模态监督失效。

## 研究问题与动机
- 非配对场景下可见光与红外训练集的身份对应关系往往不完整甚至完全缺失，大量身份在另一模态中无真实观测对应物。
- 现有非配对方法主要依赖视觉特征统计进行特征映射或生成，未显式分离判别身份的“内容”与模态特有的“风格”，易扭曲身份线索或继承源模态偏差。
- 独立聚类后依赖伪标签关联进行跨模态监督，在身份重叠不全时强行匹配会产生假正例，而直接拒绝又会丢失有效监督信号。
- 亟需一种不依赖真实跨模态对应、能可靠生成缺失模态替身并安全注入对比学习的监督机制。

## 核心贡献（创新点）
- 首次将非配对USL-VI-ReID中的跨模态监督缺失问题形式化为语义补偿问题，提出SMC框架，为无可靠对应关系的聚类放弃强行匹配转而生成候选语义替身。
- 引入双模态提示学习（DMP）与身份语义映射（ISM）的组合机制，在CLIP语义空间中用聚类原型派生的token表征身份、用可学习提示表征模态成像风格，实现身份内容与模态风格的显式解耦与可控重组。
- 设计跨模态语义补偿（CSC）模块，构造覆盖度得分与置信度得分双重门控，仅将高可靠性的合成替身注入补偿记忆，避免低质量生成样本污染跨模态监督。
- 在SYSU-MM01、RegDB与LLCM上系统评估四种身份不匹配程度，SMC在所有设定下均取得无监督方法最优结果，且随错配加剧性能衰减更平缓。

## 方法详解
- **整体架构**：以ADC（Augmented Dual Contrastive Learning）为基础编码器$E_\theta$，额外添加仅训练时启用的语义分支。该分支包含冻结的CLIP文本编码器$E_T$、投影器$G_{vs}$、共享身份映射器$G_{id}$与反向投影器$G_{sv}$；推理阶段直接丢弃语义分支，仅保留$E_\theta$。
- **双模态提示学习（DMP）**：每轮训练伊始用DBSCAN对两模态独立聚类得到伪标签与原型$\phi_k^m = \mathrm{Norm}\left(\frac{1}{|\mathcal{C}_k^m|}\sum_{i \in \mathcal{C}_k^m} \mathbf{f}_i^m\right)$。学习模态独立提示$\mathbf{P}^m$，拼接至模板“a photo of a [prompt] [identity tokens] person.”生成模态锚点$\mathbf{t}^m = E_T(\mathcal{T}(\mathbf{P}^m, \mathcal{O}))$。损失$\mathcal{L}_{DMP}$包含分类交叉熵与余弦边距分离项$\lambda_{sep}[\langle \mathbf{t}^v, \mathbf{t}^r\rangle - \mu]_+$，强制区分可见光与红外风格。
- **身份语义映射（ISM）**：将聚类原型映射为身份token序列$\mathbf{T}_k^m = G_{id}(\phi_k^m)$。分别用源模态提示重建与目标模态提示生成跨模态候选$\mathbf{h}_k^{m \to \bar{m}} = \mathrm{Norm}(G_{sv}(E_T(\mathcal{T}(\mathbf{P}^q, \mathbf{T}_k^m))))$。损失$\mathcal{L}_{ISM} = \lambda_{rec}\mathcal{L}_{rec} + \lambda_{cons}\mathcal{L}_{cons} + \lambda_{adv}\mathcal{L}_{adv}$，分别约束余弦重建误差、簇内样本token归一化一致性，并通过梯度反转对抗分类器剥离token中的模态信息。
- **跨模态语义补偿（CSC）**：覆盖度$s_k^m = \max_l \langle \phi_k^m, \phi_l^{\bar{m}} \rangle$衡量目标模态是否缺失对应簇；置信度$w_k^m = \frac{2 + c_{cmp,k}^m + c_{rec,k}^m}{4}$仅依赖源模态内部指标，避免与覆盖度自我矛盾。门控条件$\mathcal{G}_m = \{k \mid s_k^m < \delta_u, w_k^m > \delta_g\}$筛选候选簇，将其跨模态合成特征缓存并追加至目标模态补偿记忆$\Omega^{\bar{m}}$。CSC损失以缓存候选为唯一正样本、$\Omega^{\bar{m}}$为负样本池计算加权NCE，仅更新编码器$E_\theta$。
- **训练策略**：总损失$\mathcal{L}_{SMC} = \mathcal{L}_{ADC} + \lambda_{DMP}\mathcal{L}_{DMP} + \lambda_{ISM}\mathcal{L}_{ISM} + \lambda_{CSC}\mathcal{L}_{CSC}$。前30 epoch仅优化ADC建立稳定聚类与记忆，后30 epoch联合优化全部组件；各记忆、门控集合与缓存候选每轮边界刷新。

## 实验与结果
- **数据集与协议**：SYSU-MM01（All Search / Indoor Search）、RegDB（10折V2T & T2V）、LLCM；通过身份替换构造$\alpha \in \{0.25, 0.5, 0.75, 1.0\}$四种不匹配程度，$\alpha=1.0$时两模态训练集身份完全无重叠。
- **配对设置表现**：在SYSU-MM01上，SMC超越最新基线ITKM(M)，All Search R1提升2.96、mAP提升2.72；Indoor Search R1提升0.85、mAP提升0.92。在LLCM上，V2T R1达60.15、mAP 64.32，超越最强基线LVLM-AAM分别+7.95/+7.02；T2V保持约+7.86/+8.74提升。
- **非配对设置表现**：在所有$\alpha$值及三个数据集上均取得Uns supervised最高R1与mAP。$\alpha=1.0$时，SYSU-MM01 All Search mAP较MCL提升2.40，RegDB V2T mAP提升4.78；从$\alpha=0.25$到1.0，SMC在SYSU-MM01 All Search的mAP衰减仅9.99点，显著小于MCL（12.29点）与DLM（24.22点），鲁棒性优势明显。
- **消融实验**（$\alpha=0.5$）：ADC基线→+DMP→+ISM→+CSC逐层递增，各组件贡献稳定；CSC单步贡献最大（SYSU-MM01 All Search mAP +7.92、Indoor +6.95、RegDB V2T +6.29），验证语义补偿是性能提升的核心来源。可视化表明SMC使跨模态同类

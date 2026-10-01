---
title: "Universal-Cross-Prompt-Adversarial-Attacks-on-Promptable-Con"
source: https://arxiv.org/pdf/2609.39265v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-10-01 21:32:21"
field: "视觉基础模型对抗安全"
keywords: ["Promptable Concept Segmentation", "Universal Adversarial Perturbation", "Cross-Prompt Attack", "SAM3", "Adversarial Robustness", "Video Segmentation", "Perception Deception"]
innovations: ["提出min-max双层对抗提示优化策略以提升跨提示迁移性", "设计全局-局部感知欺骗攻击直接压制PCS模型探测器输出", "构建时序转换偏差攻击破坏帧间语义一致性与记忆指针"]
benchmarks: ["SA-CO", "YouTube-VOS", "DAVIS", "MOSE"]
---

# 论文速读：Universal-Cross-Prompt-Adversarial-Attacks-on-Promptable-Con

## 一句话总结
本文针对新兴的Promptable Concept Segmentation (PCS) 模型，提出了AdvPCS——一种通用的跨提示对抗攻击方法，能够在单一UAP下对point/box/text三种提示类型及跨视频帧实现强攻击迁移性，将SAM3系列模型在SA-CO数据集上的文本提示mIoU降至0.00%。

## 研究问题与动机
- **PCS模型的对抗鲁棒性未被探索**：SAM3引入了探测器（detector）用于开放词汇概念识别，但现有针对SAM/SAM2的对抗攻击均未覆盖这一新概念分割范式。
- **现有攻击跨提示迁移性有限**：AttackSAM、DarkSAM等方法主要针对固定提示类型设计，当提示类型变化时攻击效果大幅下降；纯编码器侧的扰动攻击（如[6,32]）在提示信息丰富时也失效。
- **时序语义一致性被忽视**：UAP-SAM2揭示了SAM2中时序一致性会削弱对抗攻击，但针对SAM3这类具有记忆指针机制的PCS模型，跨帧攻击有效性仍存挑战。
- **探测机制是关键薄弱点**：探索实验发现SAM3的分段性能下降与探测器存在置信度下降呈显著正相关，表明欺骗感知机制可有效提升攻击效果。

## 核心贡献（创新点）
1. **首次提出针对PCS模型的通用跨提示对抗攻击框架AdvPCS**：与现有仅针对PVS的方法本质不同，AdvPCS直接作用于SAM3的探测器感知模块，覆盖point/box/text三种提示模式。
2. **设计min-max双层对抗提示优化策略**：内层最大化候选提示多样性（文本语义扩展、点/框空间量化），外层最小化选取响应最高的 hardest-to-attack 提示进行联合优化，显著提升跨提示实例迁移性。
3. **提出全局-局部感知欺骗攻击**：同时压制全局存在logit（$r_i^t, r_i^p, r_i^b$）与局部特征logit（$z_i^t, z_i^p, z_i^b$），并通过交叉提示正则项（$\mathcal{I}_{reg}$）防止过拟合单一提示类型，与以往仅优化mask logits的方法形成本质区别。
4. **设计时序转换偏差攻击**：通过最大化对抗序列与良性序列在对象指针和记忆特征上的时序轨迹余弦偏差，破坏帧间语义一致性，比UAP-SAM2的纯特征差异最大化更具针对性。

## 方法详解
**整体框架**：AdvPCS生成单一UAP $\delta$（满足$\|\delta\|_p \leq \epsilon$），输入任意视频帧后使SAM3无法有效分割目标对象，总损失为：
$$\mathcal{I}_{total} = \mathcal{I}_{pa} + \mathcal{I}_{ta}$$

**Min-max双层提示优化**：
$$\min_{\|\delta\|_p \leq \epsilon} \max_{p_i^j \in \mathbb{P}} \sum_{i=1}^{N}\sum_{j\in\{t,p,b\}} \mathcal{L}(f_\theta(x_i+\delta, p_i^j), y_i)$$
- **内层最大化**：扩展候选提示集合——文本提示用LLM生成语义相似变体；点提示从前前景掩码的空间分位数生成左偏/居中/右偏点；框提示生成默认框及放大15%/30%的变体。
- **外层最小化**：基于探测器概率分数选取三类提示中响应最高的实例进行联合优化。

**全局-局部感知欺骗攻击（$\mathcal{I}_{pa}$）**：
- 全局存在弱化损失（式4）：将三种提示下的全局logit $r_i^j$ 统一压向低置信度目标$\tau$（$\tau = \log(0.01/0.99)$）。
- 局部存在弱化损失（式5）：压制局部logit $z_i^j$，对难以攻击的box提示加权$\lambda=5$。
- 交叉提示一致性正则（式6）：约束三种提示下预测概率$sigmoid(z_i^j)$的两两差异，防止过拟合。

**时序转换偏差攻击（$\mathcal{I}_{ta}$）**：
- 定义相邻帧指针转换$\Delta \mathbf{o}_i$与记忆特征转换$\Delta \mathbf{m}_i$。
- 计算对抗序列与良性序列的余弦相似度偏差$\pi_i$（式8）：
$$\pi_i = \cos(\Delta \mathbf{o}_i^{adv}, \Delta \mathbf{o}_i^{benign}) + \cos(\Delta \mathbf{m}_i^{adv}, \Delta \mathbf{m}_i^{benign})$$
- 目标是最小化该相似度，使对抗轨迹偏离良性时序轨迹。

## 实验与结果
**数据集与模型**：YouTube-VOS、DAVIS、MOSE、SA-CO四个视频分割基准；攻击SAM3、SAM3.1、EfficientSAM3。仅在SA-CO上训练UAP，跨数据集迁移测试。图像分割任务从各视频采样15帧构造测试集。

**关键结果（Table 2）**：
- **视频分割平均mIoU**：AdvPCS为18.33%，次优UAP-SAM2为29.83%，相对提升约38.5%。
- **SA-CO文本提示**：AdvPCS将mIoU降至0.00%（良性70.24%），完全消除分割性能；UAP-SAM2为1.18%。
- **图像分割平均mIoU**：AdvPCS为2.49%，远低于UAP-SAM2的11.58%。
- **跨模型迁移**（Fig.4a）：在SA-CO训练的UAP可显著降低SAM3.1和E-SAM3的文本分割性能。
- **跨提示实例迁移**（Fig.4b-d）：对五种不同提示实例均保持强攻击效果。

**消融实验**：移除任意模块（$\mathcal{I}_{global}$/$\mathcal{I}_{local}$/$\mathcal{I}_{reg}$/$\mathcal{I}_{ta}$/min-max策略）均显著降低攻击效果；$\lambda=5$为box提示最优权重；15个epoch后性能饱和；即使扰动预算仅4/255仍能大幅降低mIoU。

**防御实验**：在Spatter/Saturate数据预处理和模型剪枝（pruning ratio up to 0.6）下，对抗样本mIoU始终低于15%（接近0%），而良性样本mIoU随防御强度下降，证明AdvPCS对常见防御具有强鲁棒性。

## 相关工作脉络
1. **SAM/SAM2对抗攻击**：AttackSAM [30]利用掩码生成机制构造定向扰动，DarkSAM [38]引入混合频域-空间攻击增强跨提示迁移性，但两者均聚焦固定提示类型，未覆盖PCS的开放词汇探测机制。
2. **编码器侧提示无关攻击**：Croce & Hein [6]、Zheng et al. [32]仅扰动图像编码器特征实现prompt-agnostic攻击，但在提示信息丰富时效果衰减，AdvPCS通过直接攻击探测器弥补此不足。
3. **时序一致性利用**：UAP-SAM2 [35]发现SAM2的时序一致性削弱对抗攻击并设计跨帧攻击，但未考虑PCS中记忆指针机制对语义连贯性的更强依赖，AdvPCS的时序偏差攻击更具针对性。
4. **通用对抗扰动（UAP）方法**：传统UAP方法（如[8,34]）在分割任务中缺乏跨提示迁移性设计，AdvPCS的双层优化策略显著提升了跨提示/跨实例的泛化能力。
5. **对抗训练思想借鉴**：min-max双层优化受Madry et al. [17]对抗训练启发，但应用于提示选择而非扰动优化，是新的应用场景迁移。

## 局限性与未来方向
- **计算开销**：min-max双层优化需要进行提示探索与选择，增加额外计算成本。
- **模型适用性局限**：AdvPCS针对PCS模型的 concept-level存在预测设计，可能无法直接迁移至传统分割模型或SAM/SAM2等早期框架。
- **未来方向**：可扩展至其他具有探测机制的新型分割架构；探索更轻量的提示挖掘策略以降低计算负担；研究面向PCS的专用防御机制。

## 研究启发与可借鉴点
1. **min-max双层优化策略可迁移**：该"内层最大化多样性+外层最小化响应"的设计思路可复用于其他多模态输入场景（如多模态大模型的对抗鲁棒性评估）。
2. **感知层面攻击比特征层面攻击更有效**：对于具有显式探测器/分类头的基础模型，直接攻击高层语义输出（存在logit）比仅扰动编码器特征更能破坏下游分割任务，为后续针对类似架构的攻击设计提供新思路。
3. **时序偏差攻击的精细化设计**：UAP-SAM2仅最大化相邻对抗帧的特征差异，AdvPCS进一步利用良性序列作为参照基准，通过对比对抗/良性轨迹偏离程度实现更精准的跨帧攻击，该对比学习思路可用于视频模型的其他安全评估场景。
4. **交叉提示正则化防止过拟合**：$\mathcal{I}_{reg}$约束不同提示类型下预测概率的一致性，这一正则化策略可推广至其他需要跨模态/跨输入域泛化的对抗攻击研究。
5. **防御基准测试价值**：本文在数据预处理和模型剪枝上的防御评估，为后续构建PCS模型的标准化安全评测协议提供了参考范式。

## 关键术语表
- **Promptable Concept Segmentation (PCS)**：支持通过文本名词短语、点或框等提示进行概念级目标识别与分割的新范式，由SAM3引入。
- **Universal Adversarial Perturbation (UAP)**：单一对抗扰动，可泛化至多种输入（视频帧、不同提示类型），无需为每个样本重新生成。
- **Min-Max Bilevel Optimization**：双层优化策略，内层最大化候选提示多样性，外层最小化选取响应最高的提示进行联合对抗优化。
- **Global-Local Perception Deception**：同时压制探测器输出的全局存在logit和局部特征logit，使模型从宏观到微观均无法识别目标。
- **Temporal Transition Deviation**：通过破坏相邻帧间对象指针和记忆特征的时序一致性，使对抗扰动沿视频序列累积并放大。
- **Cross-Prompt Transferability**：对抗扰动从一种提示类型（如text）迁移至其他提示类型（如point/box）并保持攻击效果的能力。
- **Existence Confidence / Logit**：探测器输出的目标对象在当前帧中存在的置信度分数，是SAM3感知模块的核心输出。
- **Memory Pointer**：SAM3记忆模块中用于跨帧跟踪的对象表征，时序偏差攻击通过误导记忆指针破坏帧间一致性。

## 可复现要素
- **数据集**：YouTube-VOS、DAVIS、MOSE、SA-CO均为公开数据集；图像分割测试集由随机采样15帧构建。
- **代码开源**：https://github.com/alphanull-cqu/AdvPCS
- **模型**：使用SAM3、SAM3.1、EfficientSAM3官方开源模型。
- **关键超参**：扰动预算$\epsilon=10/255$，batch size=1，训练epoch=15，$\lambda=5$（box提示权重），$\tau=\log(0.01/0.99)$，随机种子固定为30。
- **硬件**：双NVIDIA A100-SXM4 GPU（80GB），Intel Xeon Silver 4210R CPU，125GB内存。
- **输入尺寸**：所有帧resize至$3\times1008\times1008$。

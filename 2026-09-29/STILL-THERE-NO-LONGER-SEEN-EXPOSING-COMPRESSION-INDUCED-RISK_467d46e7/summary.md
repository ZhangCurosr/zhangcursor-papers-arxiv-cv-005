---
title: "STILL-THERE-NO-LONGER-SEEN-EXPOSING-COMPRESSION-INDUCED-RISK"
source: https://arxiv.org/pdf/2609.35002v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-10-01 03:46:58"
field: "多模态大模型安全与鲁棒性"
keywords: ["大视觉语言模型", "视觉token压缩", "对抗攻击", "压缩特定失败", "模型鲁棒性", "效率与安全权衡"]
innovations: ["提出压缩特定失败(CSF)的配对归因框架，区分压缩引入与继承的失败", "设计CIRA攻击，仅凭vision encoder白盒即可在未知压缩配置下诱导压缩路径失败", "建立保留集分配因果影响压缩正确性、表征漂移负相关恢复效果的诊断证据"]
benchmarks: ["POPE", "TextVQA", "MME"]
---

# 论文速读：STILL THERE, NO LONGER SEEN: EXPOSING COMPRESSION-INDUCED RISK IN LARGE VISION-LANGUAGE MODELS

## 一句话总结
本文针对大视觉语言模型（LVLMs）的视觉token压缩技术，定义了"压缩特定失败"（CSF）这一配对归因概念，并提出CIRA攻击方法——仅依赖vision encoder白盒访问即可在未知压缩器和预算的情况下诱导压缩路径失败，同时保持全token推理的正确性。

## 研究问题与动机
- **压缩安全风险的归因困境**：现有鲁棒性评估多基于全token推理，无法区分失败是由压缩本身引入还是继承自底层模型，聚合准确率可能掩盖实例级别的压缩诱导错误。
- **压缩对鲁棒性的双向影响**：压缩可能改变视觉证据的选择与保留，既可能增强也可能削弱鲁棒性，但缺乏针对压缩路径的定向攻击方法。
- **部署不确定性下的攻击挑战**：实际部署中攻击者通常无法获知具体压缩器类型和压缩预算，需设计配置无关的攻击策略。
- **压缩边界作为安全组件的忽视**：现有研究未将压缩边界视为部署中的安全相关组件，缺乏全token与压缩推理的配对评估框架。

## 核心贡献（创新点）
1. **压缩风险归因框架**：提出压缩特定失败（CSF）的配对定义，通过控制实验证明保留集分配因果影响压缩正确性，且恢复效果与位移证据的表征漂移负相关。
2. **CIRA压缩诱导风险攻击**：设计仅依赖vision encoder的定向攻击，结合全局选择劫持（GSH）与隐藏证据保护（HEP），在未知压缩器和预算下诱导压缩特定失败。
3. **跨配置迁移评估**：在4种压缩器、3个数据集、4个预算及多个LVLM家族上系统评估，证明CIRA的压缩选择性行为具有跨架构泛化能力。
4. **选择稳定化防御TCS**：提出跨视图选择稳定化防御，利用平移不变性抑制CIRA，并测试自适应攻击的恢复能力。

## 方法详解
**问题设定**：
- CSF定义：对于清洁符合条件的样本集$\mathcal{S}_K$，压缩特定失败集为$\mathcal{F}_K^{\mathrm{CSF}} = \{i \in \mathcal{S}_K \mid c_i(x_i^{\mathrm{adv}}) = 1, c_{i,K}(x_i^{\mathrm{adv}}) = 0\}$，即全token推理正确但压缩后失败。

**CIRA攻击框架**：
1. **全局选择劫持（GSH）**：
   - 计算clean和adversarial的优先级分数向量$\mathbf{s}^c$和$\mathbf{s}^a$
   - 构建clean的逆优先级编码$\mathcal{R}(\mathbf{s}^c)$，使高优先级token映射到接近0的值
   - 优化目标：最大化adversarial优先级与逆clean优先级的标准化对齐$\mathcal{L}_{\mathrm{GSH}} = \mathrm{Align}_{\epsilon_s}(\mathbf{s}^a, \mathcal{R}(\mathbf{s}^c))$
   - 驱动clean高优先级token降级、低优先级token升级

2. **隐藏证据保护（HEP）**：
   - 识别跨越候选保留边界的clean高优先级token集合$\mathcal{H}_K(\delta) = \{i : r_i^c \leq K < r_i^a\}$
   - 计算每个token被隐藏的概率权重$w_i$（在均匀分布的候选预算下）
   - 优化目标：最小化位移证据的表征漂移$\mathcal{L}_{\mathrm{HEP}} = -\sum_i \tilde{w}_i d_i$，其中$d_i$为half-cosine距离
   - 使用stop-gradient冻结rank派生权重

3. **联合优化**：
   - 总损失$\mathcal{L}_{\mathrm{CIRA}} = \mathcal{L}_{\mathrm{GSH}} + \lambda \cdot \mathcal{L}_{\mathrm{HEP}}$
   - 使用投影符号梯度上升，约束$\|\delta\|_\infty \leq \epsilon$
   - 仅需vision encoder白盒访问，无需下游问题、标签、语言模型或压缩器信息

**TCS防御方法**：
- 使用4个平移视图$\mathcal{V} = \{T_{0,0}, T_{d,0}, T_{0,d}, T_{d,d}\}$，$d=7$像素
- 对齐各视图的优先级得分到参考坐标系，计算排名分位数共识
- 选择具有最高跨视图排名分位数一致性的Top-K tokens

## 实验与结果
**实验设置**：
- 模型：LLaVA-v1.5-7B（主）、Qwen3-VL-8B-Instruct、InternVL3.5-8B
- 数据集：POPE（1000样本）、TextVQA（1000样本）、MME（1000样本）
- 压缩器：VisionZip、VisPruner、PruMerge、FastV
- 预算：$K \in \{32, 64, 128, 192\}$
- 基线：VEAttack、CAGE、CAA†（更强访问参考）
- 超参：$\epsilon = 4/255$，100步优化，$\lambda = 0.8$

**主要结果**（Table 2）：
- **CIRA平均CSFR达20.35%**，显著优于VEAttack（5.49%）和CAGE（5.92%）
- **Full ASR仅6.92%**，远低于VEAttack（45.95%）和CAGE（50.20%）
- **跨压缩器泛化**：在PruMerge上平均CSFR达31.96%（CAGE仅6.28%）
- **跨预算提升**：CSFR从K=192时的12.04%提升至K=32时的28.78%
- **对比强访问基线**：CAA†平均CSFR仅3.24%，远低于CIRA的20.35%，证明保留全token正确性不足以诱导压缩特定失败

**机制分析**（Figure 3-4, Table 3）：
- GSH驱动全局优先级重分配：最高优先级八分位降级0.47-0.53，最低优先级八分位升级0.73-0.76
- HEP限制表征漂移：最高优先级组漂移仅12.4-12.7%，而低优先级组漂移26.1%
- **消融实验**：移除GSH使平均CSFR从18.22%降至3.68%；移除HEP使Full ASR从6.92%升至23.10%

**防御效果**（Table 4）：
- TCS将标准CIRA的平均CSFR从18.22%降至3.39%（相对减少81.4%）
- 自适应CIRA在TCS下恢复至12.70%，达到无防御状态的74.5%

## 相关工作脉络
1. **视觉token压缩方法**：包括基于注意力/多样性的token选择（VisPruner等）、剪枝与聚类混合方法（PruMerge等）、查询条件方法（QuietPrune等）——本文在多种压缩规则上验证攻击通用性。
2. **VLM对抗鲁棒性**：迁移攻击（Set-level guidance等）、编码器攻击（VEAttack）、原型引导灰盒攻击（PA-Attack）——本文与VEAttack对比，证明压缩特定失败的独特性。
3. **压缩鲁棒性研究**：安全感知剪枝（SAP）、鲁棒性导向剪枝——本文指出这些工作关注压缩对鲁棒性的影响方向，但未解决压缩特定失败的归因与诱导问题。
4. **压缩感知攻击**：CAGE针对未知压缩设置下预期存活的token，CAA直接操作token选择排名——本文与它们的关键区别在于无需访问压缩器配置且仅需encoder白盒。
5. **测试时防御**：提示适应、增强视图一致性——本文的TCS防御与这些方法正交，专门针对压缩边界的选择不稳定问题。

## 局限性与未来方向
- **时序与上下文依赖**：当前评估限于单图像推理，视频、多图像和多轮对话中的时序冗余和动态上下文未考虑。
- **压缩机制局限性**：诊断仅考察保留集分配和表征漂移，未涵盖自适应剪枝、学习摘要、可恢复路由等更复杂的压缩机制。
- **安全性边界**：CSF定义基于任务正确性而非响应安全性，压缩路径错误不一定构成安全对齐失败。
- **空间与位置失真**：空间破坏、位置或注意力失真等压缩诱导问题未在诊断中单独识别。
- **自适应攻击防御**：TCS可被自适应CIRA部分恢复，需探索更强的防御机制。

## 研究启发与可借鉴点
1. **配对评估框架的普适价值**：CSF的配对定义（全token vs 压缩推理）可推广至其他模型加速技术的风险评估，如量化、剪枝等。
2. **优先级重分配攻击的设计模式**：GSH通过逆优先级对齐实现选择劫持，这一思想可迁移到其他基于排序的模型组件攻击。
3. **证据保护的技术细节**：HEP使用stop-gradient冻结rank权重、按预算边际加权，这种"保护位移证据"的策略对设计选择性攻击有参考价值。
4. **跨视图稳定性防御**：TCS利用平移不变性聚合排名共识，这一思路可应用于其他对空间扰动敏感的多模态组件。
5. **可控诊断实验设计**：保留集干预的反事实实验（固定encoder状态和容量，仅交换token身份）为分离混淆因素提供了可复用的方法论。

## 关键术语表
- **压缩特定失败（CSF）**：对抗样本在全token推理下保持正确但在压缩推理下失败的现象，用于归因压缩引入的风险。
- **CIRA**：Compression-Induced Risk Attack，仅依赖vision encoder白盒访问的压缩诱导风险攻击框架。
- **全局选择劫持（GSH）**：通过最大化adversarial优先级与逆clean优先级对齐，实现token优先级的全局重分配。
- **隐藏证据保护（HEP）**：限制被位移的clean高优先级token的表征漂移，保持其在全token推理中的效用。
- **保留集分配**：压缩后保留的视觉token集合，干预实验显示其因果影响压缩正确性。
- **表征漂移**：clean与adversarial token表示的half-cosine距离，与证据恢复效果负相关。
- **Translation-Consensus Selection（TCS）**：利用多个平移视图的优先级排名共识进行稳健token选择的防御方法。
- **Selective Gap**：平均CSFR与Full ASR的差值，用于衡量攻击的压缩选择性。

## 可复现要素
- **数据集**：POPE、TextVQA、MME（公开基准，各采样1000样本对）
- **代码/权重**：论文声明代码在Github开源（具体链接见原文）
- **模型**：LLaVA-v1.5-7B、Qwen3-VL-8B-Instruct、InternVL3.5-8B（公开预训练模型）
- **关键超参**：$\epsilon = 4/255$，优化步数100，$\lambda = 0.8$，候选预算区间$[32, 192]$，TCS平移距离$d=7$像素
- **硬件**：单张NVIDIA GeForce RTX 4090 GPU
- **压缩器实现**：VisionZip、VisPruner、PruMerge、FastV（各有开源实现）

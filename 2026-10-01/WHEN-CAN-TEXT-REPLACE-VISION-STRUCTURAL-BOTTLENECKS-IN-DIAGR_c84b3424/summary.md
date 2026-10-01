---
title: "WHEN-CAN-TEXT-REPLACE-VISION-STRUCTURAL-BOTTLENECKS-IN-DIAGR"
source: https://arxiv.org/pdf/2609.39142v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-10-01 21:33:04"
field: "多模态推理与视觉语言模型评测"
keywords: ["diagram reasoning", "structured text extraction", "visual-language models", "benchmarking", "structural bottleneck", "token efficiency"]
innovations: ["分离表示充分性与获取误差的固定求解器三层梯度协议", "问题相关拓扑比全图拓扑更优预测QA正确率（Brier 0.1147 vs 0.1406）", "匹配干预证明单条相关边错误可使122B求解器准确率降至0-3.5%而无关编辑保持92-94%"]
benchmarks: ["FlowGen", "QZhou-Flowchart-QA"]
---

# 论文速读：WHEN CAN TEXT REPLACE VISION? STRUCTURAL BOTTLENECKS IN DIAGRAM REASONING

## 一句话总结
本文提出诊断协议，区分"表示是否充分"与"表示是否成功获取"两个维度，在 FlowGen 图表问答任务上证明：即使结构文本合法，若缺失答案关键边，端到端准确率仍远低于直接视觉输入（87% vs <30%），且该差距随节点数与分支深度急剧扩大。

## 研究问题与动机
- 图表常将答案编码在少量关系里；若可将这些关系以文本形式恢复，语言模型可重复使用显式表示并可能减少求解 token 数（如 DePlot、Pix2Struct 等结构化视觉接口）。
- 端到端准确率无法揭示失败原因：文本化后错误可能源于表示遗漏了问题所需信息（acquisition failure），也可能源于求解器未能利用已有信息（sufficiency failure）；全局保真度也无法反映"缺失信息是否关键"。
- 既有评测（如 MathVista、VisRes、DISSECT、OmniMapBench）多关注单点精度或问题条件式描述，未隔离"获取误差"与"表示-求解器兼容性"，也未比较"相关拓扑"与"全图拓扑"对 QA 的预测力。

## 核心贡献（创新点）
- 提出固定求解器的三层输入梯度和有效性恢复流程，将表示充分性（supplied structure utility）与表示获取（image-to-text pipeline）分离为可独立测量的指标。
- 在保留的 240 张公共 FlowGen 测试集上证实：金标准结构 QA 达 87.1%，而直接视觉与学习文本均低于 30%，两者差距随节点数与分支深度增加而翻倍（≤10 节点差 38.8pp，21–40 节点差 85.2pp）。
- 引入问题相关支撑掩码（question-relevant support mask），证明相关拓扑精确匹配比全图拓扑更优地预测 QA 正确率（Brier loss 0.1147 vs 0.1406），且匹配干预表明"错误位置"的影响远超"错误数量"（单条相关边编辑可将 122B 主解答准确率降至 0–3.5%，而匹配无关编辑保持 92–94%）。

## 方法详解
- **固定求解器输入梯度（Fixed-solver input ladder）**：给定图表图像 $I_g$、源结构 $G_g$、问题 $q$ 与参考答案 $y$，问题盲提取器 $E_b$ 在不访问 $q/y$ 的条件下生成 $z_g = E_b(I_g)$；固定求解器 $S$ 比较三种输入条件：$\hat{y}_{direct} = S(I_g, q)$、$\hat{y}_{learned} = S(z_g, q)$、$\hat{y}_{gold} = S(G_g, q)$。定义获取差距 $\Delta_{acq} = A_{learned} - A_{gold}$ 与充分性差距 $\Delta_{suff} = A_{gold} - A_{direct}$。
- **有效性恢复（Validity recovery）**：仅对 schema 无效的提取重试更大 token 预算（1,024 → 2,048 → 8,192），再经无答案归一化器与至多一次重试；已合法但语义错误的提取保持不变，验证格式障碍是否能解释差距。
- **问题相关保真度（Question-relevant fidelity）**：离线支撑掩码 $M(q, G_g)$ 针对四类问题提取相关边集合（入/出邻居、有序关系对、最短路径保守 BFS 证书），比对全图与相关子图的有向拓扑精确匹配与边 F1；使用单特征逻辑回归与 Out-of-fold Brier loss 评估预测能力。
- **匹配干预（Matched interventions）**：对源结构施加成对删除/反转/重定向编辑，相关编辑故意改变原答案语义，无关编辑保持原答案；在同一图/问题对与编辑数量 $m$ 下对比，测量错误位置敏感性与解构鲁棒性。
- **成本度量**：单次使用（$K=1$）与复用（$K=3$）下平均成本 $C_K(g) = \frac{A_g^{tok} + \sum_{q=1}^K S_{gq}^{tok}}{K}$，其中 $A_g^{tok}$ 包含所有失败尝试的 acquisition prompt 与 output tokens。

## 实验与结果
- **数据集**：240 张公共 FlowGen Diagrams renderer 图表（easy/medium/hard 各 80），生成 720 道来源-派生问题（入邻居、出邻居、源关系标签、最短路径长度），四类问题家族。
- **模型设置**：主模型 Qwen3.5-122B-A10B（MoE，~10B active parameters），提取 cap 1,024，求解 cap 8,192；BF16、temperature 0、seed 427、thinking 关闭。
- **主要结果**：
  - 金标准结构 QA 87.1%；固定学习文本 22.8%；直接视觉 27.2–28.8%；经有效性恢复后学习文本仅升至 23.5%。
  - 获取差距 $\Delta_{acq} = -64.31$pp（95% CI [−68.89, −59.58]）；充分性差距 $\Delta_{suff} = +59.24$pp。
  - 按节点数分箱：≤10 节点差距 38.8pp，11–20 节点 65.2pp，21–40 节点 85.2pp；按分支深度分箱：depth 0 差 39.3pp，depth >6 差 84.5pp，趋势显著（Bonferroni 调整后 98.33% CI 均不跨越零）。
  - 问题相关拓扑 Brier loss 0.1147 优于全图拓扑 0.1406（$\Delta = -0.0259$，CI [−0.0479, −0.0047]）。
  - 匹配干预：122B 在相关边反转 $m=1$ 下 QA 3.5%，无关反转 92.4%；相关边删除/重定向同样降至 0%，无关保持 91–94%。
  - 成本：金标准结构求解 token 比直接视觉少 81.8%；学习获取（含失败尝试）在 $K=1$ 时最昂贵（7,551 tokens/Q），复用 $K=3$ 降至 3,201，但仍无法达到可比较精度阈值。
- **对照诊断**：在受控生成流程图与布尔电路（OLD271+NEW270，共 406+135）上，有效性恢复使 QA 接近天花板（修复后 flowchart 99.5%、circuit 99.3%），说明低预算格式障碍可修复；但在公共 FlowGen 上差距存活，凸显语义获取鸿沟。

## 相关工作脉络
- **DePlot / Pix2Struct / MatCha**：图表到表格/结构的预训练与转换管线；本文对比的是"无图纯文本源结构"对求解器的效用，而非结构辅助视觉。
- **MathVista / MathVerse / VisRes**：视觉问答与视觉依赖性评测基准；MathVerse 通过变化图文信息测试视觉依赖，但缺少源派生文本参考以分离获取误差。
- **DISSECT / OmniMapBench**：DISSECT 用问题条件式描述与人类 oracle 对比五模态，但未分离"供应结构"与"获取结构"；OmniMapBench 用问题无关描述替换图像，却无源文本参考。
- **FlowGen (Shi et al., 2026)**：控制图属性与渲染风格的流程图基准，支持严格/宽松三元组保真度评测；本文在其保留测试集上扩展困难度依赖与有效性恢复诊断。
- **QZhou-Flowchart-QA**：中文流程图数据集中文本输入反而不如直接视觉，提示源结构效用具有任务/模型/容量交互性，需逐场景验证。

## 局限性与未来方向
- 确认性评估仅限 FlowGen Diagrams renderer、源生成问题与单一模型族，预训练暴露未知；不同图编码方式（实体/关系对象 vs 标签三元组 JSON 串）混淆了纯提取误差与表示-求解器兼容性。
- 部分源派生关系标签（如 connectedTo、partOf）未显式印刷于图像中，使全局差距不能完全归因为图像提取误差；可见性合格分析需单独后验。
- 离线支撑掩码非最小充分子图；干预是在 512-token 受限求解器上进行的机制研究，不估计自然获取差距中的错误比例。
- 未来可探索：(1) 统一图编码以隔离提取误差；(2) 可见性校准评测以区分"源特有约定"与"图像可观察事实"；(3) 面向答案关键边的定向提取与修复策略；(4) 多问题复用场景下的端到端成本-精度帕累托前沿。

## 研究启发与可借鉴点
- **获取-充分性分离框架**：可将相同范式迁移至图表/流程图解析、程序控制流还原、电路网表提取等"视觉→结构文本"下游任务，作为通用诊断基准。
- **问题相关保真度评估**：在构建抽取管线时，可用支撑掩码精确度（而非全局 F1）作为 early stopping 或损失函数权重，优先优化 answer-critical 子图。
- **匹配干预实验设计**：通过成对相关/无关编辑测量模型鲁棒性，适用于任何结构化推理系统的安全/可靠性审计，揭示"位置敏感性"而非仅"错误率"。
- **复用成本建模**：将 acquisition + solving token 按复用次数 $K$ 摊销，可指导工业场景中"一次性提取→多次问答"的经济性评估与容量规划。
- **可结合本团队方向**：若在文本化视觉管线中加入答案关键边优先的注意力引导（attention-guided extraction）或两阶段抽取-修复模块，有望缩小公共 FlowGen 上 64pp 的获取差距。

## 关键术语表
- **Fixed-solver input ladder**：使用同一求解器在不同输入条件（图像、学习文本、金标准文本）下运行的对照实验设置，以隔离表示质量与求解器能力。
- **Question-relevant support mask**：根据问题类型离线确定的相关边集合（邻居、有序对、最短路径 BFS 证书），用于限制保真度与预测评估范围。
- **Directed topology exact match**：在指定范围内比较节点集与有向边集是否完全一致（忽略边标签），用于衡量结构保留精度。
- **Matched intervention**：在相同图/问题上施加成对相关/无关编辑，以隔离错误位置对下游 QA 的影响而非单纯计数。
- **Validity recovery**：仅对 schema 无效的提取重试更大预算并应用无答案归一化，验证格式障碍是否足以解释端到端性能差距。
- **Acquisition gap / Sufficiency gap**：$\Delta_{acq} = A_{learned} - A_{gold}$ 衡量获取管线效用损失；$\Delta_{suff} = A_{gold} - A_{direct}$ 衡量供给了源结构后的增益。
- **Token reuse cost model**：$C_K(g) = \frac{A_g^{tok} + \sum_{q=1}^K S_{gq}^{tok}}{K}$，将单次提取成本分摊至 $K$ 个问题上的平均推理开销度量。

## 可复现要素
- **数据集**：FlowGen 公共测试集（reserved 240 charts / 720 questions），官方提供；QZhou-Flowchart-QA 公开数据集（Apache-2.0）；受控生成图表为作者自造（非公开）。
- **代码**：GitHub https://github.com/yunbeizhang/text-for-vision（论文声明）。
- **模型**：Qwen3.5-122B-A10B、Qwen3.5-27B、Qwen3.5-4B（HuggingFace 可获取）。
- **关键超参**：提取 cap 1,024；恢复预算 2,048 / 8,192；求解 cap 8,192（主实验）/ 512（干预）；BF16、temperature 0、seed 427、thinking 关闭。
- **统计设置**：10,000 次 chart-cluster bootstrap、Bonferroni 98.33% CI（主要检验）、gold floor 70%。

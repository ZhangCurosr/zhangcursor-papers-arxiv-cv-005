---
title: "SPATIAL-OPSD-SELF-IMPROVING-SPATIAL-REASON-ING-VIA-LABEL-FRE"
source: https://arxiv.org/pdf/2609.37055v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-10-01 17:44:26"
field: "空间推理与自蒸馏"
keywords: ["spatial reasoning", "on-policy distillation", "self-improvement", "vision-language model", "privileged context", "recursive training"]
innovations: ["将自动获取的空间结构转化为特权监督实现无标签在策略自蒸馏", "轮次递归自进化方案使教师-学生对齐更新并保持稳定训练目标"]
benchmarks: ["SPAR-Bench", "MindCube-tiny", "MMSI-Bench", "ViewSpatial-Bench", "VSI-Bench"]
---

# 论文速读：SPATIAL-OPSD: SELF-IMPROVING SPATIAL REASONING VIA LABEL-FREE SELF-DISTILLATION

## 一句话总结
本文提出 **Spatial-OPSD**，一种无标签的在策略自蒸馏框架，通过将自动获取的空间先验（深度、3D关系、相机几何）作为特权上下文供给教师模型，在不依赖答案标签的情况下持续自我提升 VLM 的空间推理能力。

## 研究问题与动机
- **问题**：VLM 在具身/空间落地场景中需理解深度、视角、三维关系，但现有空间推理提升方法大多依赖地面真值答案或答案衍生奖励进行监督。
- **不足**：传统方法将空间信息视为一次性训练数据，难以支撑模型的持续自我进化；同时需要任务特定标注或奖励信号，成本较高。
- **目标范式**：无需答案标签、在学生自采样轨迹上提供监督、且能随模型能力提升重复使用的训练机制。

## 核心贡献（创新点）
- **将空间结构转化为特权监督信号**：利用工具自动生成的深度、3D关系、相机几何构建教师模型的特权上下文，替代答案标签进行蒸馏，与依赖 GT 答案的 SFT/GRPO 形成本质区别。
- **提出轮次递归自进化方案**：每轮内冻结教师提供稳定目标，轮间用改进学生初始化新教师与学生，同时空间先验重新建立信息不对称；相比固定教师（无法继承改进）或逐步刷新（优化不稳定），该方法可持续推进。
- **系统验证跨架构泛化与多轮增益**：在 Qwen3-VL、Gemma-3、InternVL3.5、LLaVA-OV 四个 4B 系列上一轮训练均带来提升；从强空间专用模型出发，三轮递归训练在开源模型中取得最高平均分，并在三个基准上创最佳。
- **揭示 reverse KL 为最优蒸馏目标**：对比 forward KL 与 Jensen-Shannon 散度，reverse KL 在五个基准上整体最强、迁移最均衡。
- **证明对噪声空间先验的鲁棒性**：对特权空间数值施加高达 20% 相对误差扰动时性能仍显著优于未适配基座，说明方法对工具噪声不敏感。

## 方法详解
**框架概览**：学生策略 $\pi_\theta$ 仅接收视觉-语言输入 $x=(v,q)$；同名检查点初始化的教师 $\pi_{\bar{\theta}}$ 额外接收特权空间子图 $z=R(s,q)$，其中场景脚手架 $s=g(v,m)$ 由几何构建器从视觉输入和可选元数据生成。

**特权空间信息构建**：
- 数据源：SPAR-7M-RGBD（原生深度/相机）、VSI-590K（Grounding DINO + SAM 2 + VGGT 伪标注）、SenseNova-SI-8M（保留原生 3D 标注）。
- 场景图包含物体/相机节点及深度、位移、欧氏距离、相对位姿、跨视图投影等边。
- 质量问题过滤：仅保留通过检测置信度、可见面积、有效点数、重投影校验的几何；度量值需可靠尺度，否则转为序数/归一化关系。

**在策略空间自蒸馏损失**：
$$
\mathcal{L}_{\text{Spatial-OPSD}} = \mathbb{E}_x \mathbb{E}_{\mathbf{y}\sim\pi_{\theta_k}}\left[\frac{1}{\sum_t m_t}\sum_{t=1}^T m_t D_{\text{KL}}\!\left(p_t^S \, \| \, \mathrm{sg}[p_t^T]\right)\right]
$$
- $p_t^S = \pi_{\theta_k}(\cdot|x,y_{<t})$，$p_t^T = \pi_{\bar{\theta}}(\cdot|x,z,y_{<t})$，$m_t$ 遮蔽 prompt/padding。
- 纯蒸馏（$\lambda_{\text{KD}}=1$），无答案交叉熵或答案衍生奖励；温度 $T=1$。

**轮次递归自进化**：
- 第 $r$ 轮：$\theta_{r,0}^S \leftarrow \theta^{(r-1)}$，$\bar{\theta}_r \leftarrow \mathrm{sg}(\theta^{(r-1)})$，教师在该轮冻结。
- 训练后得到 $\theta^{(r)}$，同时初始化下一轮的教师与学生。
- 特权上下文重新建立教师-学生信息不对称，使改进可持续。

## 实验与结果
- **数据集/基线**：SPAR-Bench、MindCube-tiny、MMSI-Bench、ViewSpatial-Bench、VSI-Bench 五基准；对比 SFT、GRPO（均使用答案标签）及proprietary/open-source 空间模型。
- **4B 系列单轮结果**（Table 1）：四家族模型五项平均均提升，Qwen3-VL-4B 从 36.432 → 40.406（+3.97），SPAR-Bench 从 35.148 → 44.788（+9.64）。
- **三轮递归最强结果**（Table 2）：SenseNova-SI-Qwen3-VL-8B 从 53.54 → 54.78（+1.24），在开源模型中获最高平均，SPAR-Bench 从 40.8 → 45.8、MindCube-tiny 从 73.7 → 74.4。
- **消融**：
  - 教师刷新频率（Table 3）：逐步刷新崩溃至 0；固定教师 44.788→43.168 下降；轮次刷新 44.788→47.464→47.636 持续上升。
  - 蒸馏损失：reverse KL 整体最优。
  - 空间先验类别：privileged teacher 均被强化，student 亦超过 base。
  - 噪声鲁棒：±20% 扰动下仍显著优于基线。

## 相关工作脉络
- **MiniLLM / GKD**：解决 off-policy 中教师前缀与学生推理分布失配，本文沿用 on-policy 范式但替换监督来源为空间几何而非答案。
- **OPSD（Zhao et al., 2026; Hubotter et al., 2026）**：特权上下文蒸馏，原有工作使用解法或环境反馈，本文替换为 question-relevant 几何子图。
- **SpatialVLM（Chen et al., 2024）**：几何估计用于空间 QA 监督，但属一次性数据而非可复用的持续蒸馏源。
- **STaR / STOP / Darwin Godel Machine**：递归自改进通过答案选例或代码 scaffolds 迭代，本文则通过特权 on-policy 蒸馏更新 VLM 权重，不依赖答案标签。
- **VGGT / Depth Anything v2 / DUSt3R**：提供重建/深度工具，本文将其输出作为 teacher 特权上下文并建立质量控制。

## 局限性与未来方向
- **依赖外部工具**：当前训练需地面感知/重建工具（Grounding DINO、SAM 2、VGGT 等），部署时虽不需要，但数据构建成本高。
- **单轮单答**：每个 prompt 仅采样一条响应，可能限制对探索多样性的利用。
- **未来方向**：探索自主工具调用与模型自生成空间脚手架，迈向完全 self-evolving 的空间推理系统。

## 研究启发与可借鉴点
- **特权上下文迁移范式**：将"不可在推理时使用的辅助信息"系统性地用于蒸馏，可推广至时间/因果/物理等其他结构化先验。
- **轮次冻结-轮间刷新**：平衡教师稳定性与能力继承，是递归自改进的核心设计，值得在其他自蒸馏场景中复用。
- **质量控制流水线**：对伪几何标注进行置信度/可见面积/重投影等多维过滤，避免噪声污染蒸馏信号，可参考至其他工具增强训练。
- **与 SFT/GRPO 公平对比**：基线允许使用答案而本文不允许，仍保持竞争力，证明方法的有效性与经济性，可作为后续研究的对比标准。

## 关键术语表
- **Spatial-OPSD**：无标签的在策略特权空间自蒸馏框架，通过教师模型的特权几何上下文对学生自采样轨迹提供 token 级监督。
- **Privileged spatial scaffold**：从视觉输入与元数据构建的可复用场景图，含物体/相机节点及几何关系边。
- **On-policy self-distillation**：教师反馈在学生当前生成的轨迹上计算，避免 off-policy 前缀分布失配。
- **Round-wise recursive self-improvement**：每轮内教师冻结，轮间用改进学生同时初始化教师与学生，逐步累积能力。
- **Reverse KL distillation**：最小化学生分布相对教师分布的 KL 散度，本文中为最优蒸馏目标。
- **SPAR-Bench / MindCube-tiny / MMSI-Bench / ViewSpatial / VSI-Bench**：五个互补的空间推理评测基准，衡量深度、视角、多步空间记忆等能力。

## 可复现要素
- **数据集**：SPAR-7M-RGBD、VSI-590K、SenseNova-SI-8M（公开）；训练集为从三者各抽取的共 6.2k 条样本（论文附录 B 描述）。
- **代码/权重**：代码开源在 https://github.com/vermouth599/Spatial-OPSD；模型权重未明确声明，使用公开 4B 基座与 SenseNova-SI-Qwen3-VL-8B。
- **关键超参**：AdamW、lr=2e-6、cosine schedule、min lr=1e-8、bfloat16、FSDP2、batch=8、responses per prompt=1、temperature=1.0、reverse KL T=1、epochs per round=1、seed=42。
- **硬件**：单卡 NVIDIA H20 96GB；软件栈 PyTorch 2.8.0、CUDA 12.8、SGLang 0.5.5、lmms-eval 0.7.2。

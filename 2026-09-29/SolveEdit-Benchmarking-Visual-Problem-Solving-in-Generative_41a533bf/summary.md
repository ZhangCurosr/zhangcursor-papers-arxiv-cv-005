---
title: "SolveEdit-Benchmarking-Visual-Problem-Solving-in-Generative"
source: https://arxiv.org/pdf/2609.35504v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-10-01 03:48:48"
field: "视觉生成与编辑"
keywords: ["视觉问题解决", "图像编辑基准", "转换契约评估", "多模态规划", "生成模型评测"]
innovations: ["提出 IS/SD/RD 三 regime 划分变换信息来源", "设计原子转换契约与 SolveScore 支持多解评估", "验证推理时两阶段规划显著提升固定生成器性能"]
benchmarks: ["SolveEdit"]
---

# 论文速读：SolveEdit-Benchmarking-Visual-Problem-Solving-in-Generative

## 一句话总结
论文提出 SolveEdit，一个用于评估生成模型**视觉问题解决能力**的基准测试，包含 2,728 个场景转换案例，通过原子转换契约（Atomic Transition Contracts）和 SolveScore 指标衡量模型在无需参考输出的情况下完成目标转换并保护无关内容的能力；同时提出 SolveEdit-Plan 两阶段视觉规划器，在未修改编辑器的情况下将 GPT-Image-2 的 SolveScore 从 57.0% 提升至 71.6%。

## 研究问题与动机
1. **现有基准的局限性**：当前评估主要孤立测试感知、抽象推理、生成或显式指定的变换执行，缺乏对"目标驱动型视觉问题解决"的评估——即模型需自行推断所需变换而非直接给出。
2. **多解问题与参考依赖**：视觉问题通常存在多个有效解，而现有基准依赖单一参考输出，导致合法但不同的解被误判为错误。
3. **变换信息来源缺失**：现有工作未明确区分变换信息的来源（请求指令、场景状态、图像规则），难以诊断模型失败原因。
4. **真实场景需求**：现实世界的问题解决多为视觉化场景（如物体摆放、布局修复、路线追踪），要求模型理解场景、推断变化目标并执行变换而不扰动无关内容。

## 核心贡献（创新点）
1. **提出视觉问题解决的形式化定义与基准**：将变换信息来源作为核心评估轴，划分 IS（指令指定）、SD（状态依赖）、RD（规则依赖）三个 regime，与内容域正交。
   - *本质区别*：不同于仅测试感知或生成，本文聚焦"推断→执行→保护"的完整闭环。

2. **设计基于原子转换契约的评估框架 SolveScore**：每个案例附带 Required 和 Protected 条件，支持多解评价，分离任务完成度与意外扰动。
   - *本质区别*：突破单参考输出限制，通过 SA/RA/VQ 六维诊断视图量化模型行为。

3. **系统性诊断并验证规划干预**：发现最强模型仅达 57.0% SolveScore，主要缺陷在所需语义和关系条件；提出 SolveEdit-Plan 验证"先推断变换再生成"的有效性。
   - *本质区别*：不改编辑器参数，仅通过两阶段规划（Inspect → Resolve）提升性能，证明变换恢复是主要瓶颈。

4. **构建跨模态覆盖的大规模数据集**：2,728 案例覆盖 10 领域、54 子领域，含照片、动画、游戏、电影等多源图像。
   - *本质区别*：首次将视频生成器的终端帧纳入同一契约体系进行跨模态评估。

## 方法详解

### 1. 任务形式化与 Regime 划分
- **公式 (1)**：合法变换集合 $\mathcal{Z}^*(I, x) = \Phi(x, B(I, x), K, S(I), \mathcal{R}(I))$，其中 $x$ 为请求，$B$ 为请求-实体绑定，$K$ 为背景知识，$S(I)$ 为场景状态，$\mathcal{R}(I)$ 为图像内规则。
- **合法性判定**（公式 2）：$Y \in \mathcal{V}^*(I, x)$ 当且仅当存在 $z \in \mathcal{Z}^*(I, x)$ 使得 $\text{Realize}(Y; I, z) \land \text{Preserve}(Y; I, z)$。
- **三 Regime 定义**：
  - **IS（Instruction-Specified）**：接地请求+背景知识已确定变换。
  - **SD（State-Dependent）**：需从可观察场景状态推断缺失变量（目标、动作、目的地等）。
  - **RD（Rule-Dependent）**：需解读图像内规则（图例、约束、模式）。

### 2. 原子转换契约与 SolveScore
- **每案例存储**（公式 3）：$c = (I, x, m, \mathcal{C}^{\text{req}}, \mathcal{C}^{\text{pro}}, \mathcal{A})$，含必填条件、保护条件与审计资产。
- **六维诊断分数**（公式 4）：按角色 $r \in \{\text{req}, \text{pro}\}$ 与属性 $k \in \{\text{SA, RA, VQ}\}$ 计算：
  - **SA（Semantic Accuracy）**：实体、身份、数量、属性、文本。
  - **RA（Relational Accuracy）**：空间/结构关系（位置、顺序、连通性）。
  - **VQ（Visual Quality）**：渲染缺陷（残留痕迹、畸形几何）。
- **完成率与损伤率**（公式 5）：$R = \sum w_q a_q$，$D = \sum u_q(1-a_q)$。
- **最终得分**（公式 6）：$\text{SolveScore}_\lambda = G_{\text{quality}} \cdot \max(0, R - \lambda D)$，主协议 $\lambda = 0.5$。

### 3. SolveEdit-Plan 两阶段规划器
- **公式 (7)**：$(\hat{z}, \hat{O}, \tilde{x}) = H(I, x)$，$\hat{Y} = f(I, \tilde{x})$。
- **Inspect 阶段**：识别请求中未解决的变量，请求针对性裁剪图收集证据。
- **Resolve 阶段**：比较可行变换候选，编译为带保留义务的编辑指令，单次调用原始生成器。
- **匹配控制**：与 Generic Vision Rewrite 对比，后者无显式变换变量与候选比较。

## 实验与结果

### 数据集与评估设置
- **规模**：2,728 案例，10 领域，54 子领域；29,460 原子准则（14,831 Required + 14,629 Protected）。
- **评估模型**：9 个图像到图像模型（5 开源 + 4 商业）+ 2 个图像到视频模型。
- **评估协议**：λ=0.5，每案例独立评分后平均；视频输出按终端帧评分。

### 主要结果（Table 1）
| 模型 | R↑ | D↓ | SolveScore↑ |
|------|-----|-----|-------------|
| **GPT-Image-2** | 67.6% | 23.9% | **57.0%** |
| Seedream 5.0 Pro | 64.3% | 17.6% | 56.6% |
| Gemini 3.1 Flash Image | 66.0% | 23.1% | 55.5% |
| FLUX.2 [dev] | 31.7% | 32.3% | 22.8% |
| HunyuanVideo-1.5（视频）| 17.3% | 55.5% | 7.4% |

- **关键发现**：最强模型仅 57.0%，商业模型显著优于开源；所需视觉质量高于语义/关系准确率，表明"视觉上合理≠任务完成"。

### 诊断分析（RQ2）
- **Regime 差距**：IS→RD 下降 17.2-23.7 分，SD 差距 7.0-8.8 分。
- **失败分布**：RD 下所需语义 (-25.0) 和关系 (-13.0) 损失最大，保护维度接近中性（VQ +0.1），表明**变换恢复错误**是主要瓶颈。

### 规划干预效果（Table 4）
| 生成器 | 方法 | R↑ | D↓ | SolveScore↑ | Δ |
|--------|------|-----|-----|-------------|-----|
| GPT-Image-2 | Direct | 67.6% | 23.9% | 57.0% | — |
| GPT-Image-2 | **SolveEdit-Plan** | **75.8%** | **8.6%** | **71.6%** | **+14.6** |
| Qwen-Image-Edit-2509 | SolveEdit-Plan | 40.5% | 38.5% | 26.9% | +6.1 |
| HunyuanVideo-1.5 | SolveEdit-Plan | 23.7% | 39.7% | 14.0% | +6.6 |

- **成本**：每案例 2 次 LLM 调用（Inspect + Resolve），平均 3,207 输入 token + 635 输出 token。
- **对比控制**：SolveEdit-Plan 比 Generic Vision Rewrite 高 6.3 分（GPT-Image-2）。

## 相关工作脉络
1. **视觉推理与图像编辑**：InstructPix2Pix [3]、MagicBrush [69] 等指令编辑基准提供显式变换；本文聚焦**变换来源不确定性**，区分 IS/SD/RD regime。
2. **图像生成评估**：GenEval [16]、TIFA [24]、ClipScore [21] 侧重单图质量或对齐；本文评估**转换过程**，支持多解与保护条件。
3. **有根据的编辑评估**：I2EBench [40]、UnireEditBench [19]、Envisioning [72] 关注 grounded correctness；本文扩展至跨模态视频终帧评估，并引入原子契约分解。
4. **多模态推理代理**：MMScribe/GenArtist [54]、Visual ChatGPT [56] 使用工具链多步编辑；本文聚焦**单次生成前的精确变换实例化**，验证规划价值。
5. **视频生成基准**：VBench [26]、Video-MME [14]、TiViBench [6] 评估时间推理；本文仅用终端帧验证跨模态契约可迁移性。

## 局限性与未来方向
1. **视频生成仅评估终帧**：未检验时间一致性，横向扩展到完整时序生成仍待探索。
2. **规划器依赖 VLM 能力**：SolveEdit-Plan 使用 GPT-5.6 Sol，低成本/开源替代方案未验证。
3. **开放模型性能弱**：开源模型（FLUX.2、Qwen-Image-Edit-2509）SolveScore < 27%，差距显著。
4. **规约制定依赖人工标注**：29,460 原子准则需大量人工设计，可扩展性存疑。
5. **未探索端到端训练**：当前仅为推理时规划干预，未来可结合微调或 RLHF。

## 研究启发与可借鉴点
1. **原子契约评估范式可迁移**：将复杂任务分解为 Required/Protected 原子条件，适用于任何需"完成+保护"的生成任务（如代码生成、文档编辑）。
2. **三 Regime 划分诊断框架**：IS/SD/RD 轴可复用于其他视觉推理基准，快速定位模型能力瓶颈。
3. **规划前置的 test-time 干预策略**：SolveEdit-Plan 证明"先推断再执行"在固定生成器下显著提升性能，可推广至视频生成、3D 编辑等长程任务。
4. **六维诊断视图的设计**：SA/RA/VQ × Required/Protected 矩阵提供细粒度失败分析，值得在其他 benchmark 中采用。
5. **多解容忍评估机制**：SolveScore 避免单参考偏差，对拼图、布局设计等创意任务具有参考价值。

## 关键术语表
**SolveEdit**：视觉问题解决基准测试，通过场景转换评估生成模型推断变换、执行操作并保护无关内容的能力。
**Atomic Transition Contract**：原子转换契约，由 Required（必填）和 Protected（保护）条件组成的可检查规则集合。
**SolveScore**：综合评估指标，平衡任务完成率（R）与意外损伤率（D），公式为 $G_{\text{quality}} \cdot \max(0, R - 0.5D)$。
**Regime（IS/SD/RD）**：变换信息来源分类——指令指定、状态依赖、规则依赖，正交于内容域。
**Solution-Recovery Error**：解恢复错误，模型选择了非法变换（如错误目标、错误规则），区别于渲染失败。
**Visual-Execution Error**：视觉执行错误，模型推断正确但渲染失败（位置偏差、残留痕迹等）。
**SolveEdit-Plan**：两阶段视觉规划器，Inspect（识别变量+收集证据）→ Resolve（比较候选+编译指令），不改编辑器。
**G_quality**：质量门控，拒绝缺失、空白、严重损坏、不相关或不可用的输出。

## 可复现要素
- **数据集**：SolveEdit 基准（2,728 案例）已公开，项目页面 https://wenjieshu.github.io/SolveEdit-project-page/
- **代码**：已开源 https://github.com/WenjieShu/SolveEdit
- **模型权重**： evaluated models 需通过 API 访问（GPT-Image-2、Qwen、Seedream 等）或下载开源版本（FLUX.2、OmniGen2、BAGEL）
- **关键超参**：λ = 0.5（SolveScore 默认）；Generation retry ≤ 2 次；视频输出按终端帧评分
- **评估工具**：SAM（分割）、YOLO（检测）、GPT-5.6 Sol（VLM 评估，temperature=0）

---
title: "Simultaneous-Translation-between-Sign-Languages"
source: https://arxiv.org/pdf/2609.35608v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-10-01 03:48:34"
---

# 论文速读：Simultaneous-Translation-between-Sign-Languages

## 一句话总结
本文提出了首个直接手语到手语（S2S）的流式/同步翻译系统，将 wait-k 策略适配至 sign-token 序列，并通过测试时推理与随机多路径监督训练两种机制实现；同时引入源端计算感知延迟指标 ca-Stream-AL。在六组手语方向上，系统平均将延迟降低 38%，质量损失控制在 DTW-PA-MPJPE 增加 9% 与 BLEU-4 下降 2.1 以内。

## 研究问题与动机
- 现有直接 S2S 模型（如 Wu et al. (2026)、Inan et al. (2025)）均为离线系统，必须等待完整源手语视频片段输入后才能开始生成目标手语，无法满足直播口译、双向视频通话等实时场景需求。
- 口语同步机器翻译的 wait-k 策略可自然迁移至手语序列，但手语视频目标的每个单位具有固定的物理播放时长（chunk），传统基于目标序列中心或瞬时输出的延迟指标（AL / CA-AL）会严重低估实际流式体验延迟。
- 流式翻译引入了离线设置中不存在的词序新挑战：解码器无法重排尚未观察到的内容词，低 k 值下必须在“等待（支付延迟）”与“预判（承担错误风险）”之间权衡，而 ASL/CSL/DGS 之间存在显著的句法/词序差异。
- 上游视频→SMPL-X 姿态回归已能接近或超过实时帧率，但端到端渲染仍是延迟主因；需要统一的延迟度量框架来指导 pipeline 设计与瓶颈定位。

## 核心贡献（创新点）
- **首个直接同步 S2S 翻译系统**：形式化流式 S2S 任务，使目标手语能在源手语仍在输入时即开始渲染。与 Wu et al. (2026) 的全句离线基线本质不同，本文打破了“全源输入→全源输出”的前置约束。
- **两种 wait-k 机制**：提出在完整句子模型上直接进行测试时 wait-k 推理（TT-only），以及基于随机多路径监督（stochastic multi-path supervision）的训练型 wait-k 模型（Train+TT）。与现有前缀到前缀训练不同，本文针对手语序列远长于文本、每步需重编码前缀的显存瓶颈，设计了均匀采样 m 个目标位置的多副本损失计算策略。
- **ca-Stream-AL 延迟评估指标**：提出源端计算感知流式平均滞后指标。与经典 AL/CA-AL 相比，本指标以共享的源端时间轴（固定 chunk 间隔）为基准，显式建模目标 chunk 的物理播放时长与逐 step 计算 backlog 的累积效应，能更真实反映流式手语输出的端到端延迟体验。
- **词序与预判案例分析**：通过 CSL→DGS 典型个案揭示同步翻译中“等待 vs 预判”的内在张力，明确给出流式模型在词序对齐/错位两种场景下的行为边界，为后续量化预判分析提供范式。

## 方法详解
- **骨干网络**：沿用 Wu et al. (2026) 架构。冻结的解耦 VQ-VAE 以 25 fps SMPL-X 姿态为输入，经时间下采样因子 4 输出有效 6.25 Hz 的 sign-token（每 token 对应 4 帧，物理时长 $w_{\text{chunk}} = 160$ ms）。单 Transformer（MBART-large-cc25 初始化）共享 T2S/S2S，通过特殊 token 前缀（ASL/CSL/DGS）区分语言。
- **Wait-k 推理策略**：解码器在第 $t$ 步仅条件于当前可见的源 token 前缀 $\mathbf{x}_{\le g_k(t)}$，调度函数为 $g_k(t) = \min(k + t - 1, |\mathbf{x}|)$。由于使用双向编码器，新增源 token 会改变之前位置的隐藏状态，因此每步需对当前前缀重新运行编码器（Algorithm 1）。该策略牺牲了 $\mathcal{O}(|\mathbf{x}|^3)$ 的编码器重复计算以保留预训练双向归纳偏置，当前因渲染阶段主导延迟而可接受。
- **训练型 wait-k 模型**：直接优化前缀到前缀损失 $\mathcal{L}(\pmb{\theta}) = -\sum_{t=1}^{|\mathbf{y}|} \log p_\pmb{\theta}(y_t | \mathbf{x}_{\le z_k(t)}, \mathbf{y}_{<t})$ 在长手语序列下显存不可行。本文借鉴同时语音翻译的 sampled-prefix 策略，均匀采样 $m=5$ 个目标位置并创建 $m$ 个副本，每个副本接收对应前缀输入，仅对采样位置计算 loss，其余 mask。
- **ca-Stream-AL 指标推导**：以源端 chunk 到达时间轴为基准，理想目标进度为 $i \times r$（$r = |\mathbf{y}|/|\mathbf{x}|$），wait-k 调度进度为 $i - k$，基础 Stream-AL 为两者差的均值（Eq. 7）。引入每步计算开销 $c_i$（含 VQ-VAE 编解码、Transformer 推理、SMPL-X 渲染），通过递推 $t_i = \max(t_{i-1}, i \cdot w_{\text{chunk}}) + c_i$ 计算 wall-clock 完成时间，得到计算延迟 $\Delta_i$，最终 $\text{ca-Stream-AL} = \text{Stream-AL} + \bar{\Delta}$。上游视频→SMPL-X 回归被视为外部预处理，假设其帧率 ≥ 25 fps 时不产生累积 backlog。

## 实验与结果
- **数据集**：BT 合成集（按 Wu et al. (2026) 回译流程新构建的源-目标对）与 Strict 真人验证集（Wu et al. (2026) 公开对齐数据）。涵盖 ASL/CSL/DGS 六组双向翻译方向。
- **评估基线**：Full-sentence（$k=|\mathbf{x}|$ 离线基线）、TT-only（测试时 wait-k）、Train+TT（随机多路径训练 + 测试时 wait-k，$m=5$）。
- **主要结果**：最优折中 $k^\star$ 在 BT 集上为 7 或 9（Table 1），Strict 集上类似（Table 2）。以 CSL→DGS 为例，Train+TT 在 $k=7$ 时 ca-Stream-AL 降至 1.34 s（相比基线 4.32 s 约 3 倍降低），DTW-PA-MPJPE 仅劣化 0.19，BLEU-4 下降 2.4。跨方向平均：**延迟降低 38%，DTW-PA-MPJPE 增加 9%，BLEU-4 下降 2.1

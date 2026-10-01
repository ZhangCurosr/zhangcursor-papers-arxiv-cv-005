---
title: "S-sup-3-sup-Spectral-Null-Space-Swap-Makes-Reasoning-Models"
source: https://arxiv.org/pdf/2609.37976v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-10-01 09:41:19"
---

# 论文速读：S-sup-3-sup-Spectral-Null-Space-Swap-Makes-Reasoning-Models

## 一句话总结
本文提出无训练谱空域交换方法 S³，通过分析 Thinking 与 Non-thinking 模型权重差在 Non-thinking 主导奇异子空间中的分解，发现核心推理能力主要集中在正交补空间；该方法保留 Non-thinking 模型的主体谱方向，仅从 Thinking 模型迁移其互补分量，在不增加额外训练的前提下显著降低推理 token 消耗并维持或提升准确率。

## 研究问题与动机
- CoT/Thinking 模式大幅提升 LLM 推理能力，但伴随指数级增长的解码 token 成本，效率优化成为规模化部署的关键瓶颈。
- 现有高效推理工作多聚焦于显式预算控制、token 剪枝、自适应停止或权重全局插值，缺乏对 Thinking 模型“功能增量”在参数空间中具体分布的精细刻画。
- 核心科学问题尚未澄清：Thinking 与 Non-thinking 的权重差异究竟落在哪些方向？其中哪些差异可以安全去除而不损失推理功能？
- 先验谱方法主要在原模型的主导子空间（Dominant Subspace）内操作，忽视了正交补空间在承载功能变化方面的潜在价值。

## 核心贡献（创新点）
1. **谱域能量-功能失配发现**：首次系统量化 Thinking 后训练权重差在参数空间（Frobenius 能量）与函数空间（隐藏状态偏移）的分布不对称性，证明与 Anchor 已有方向对齐的分量虽占较大参数能量，但对实际计算影响微弱，核心功能增量集中在正交补空间。
2. **S³ 无训练非对称组合框架**：提出 Spectral Null-Space Swap (S³)，通过单一超参 $\rho$ 控制保留的奇异向量比例，实现“Non-thinking 在子空间内、Thinking 在子空间外”的精确权重拼接，区别于全局插值（MI）与冲突消解合并（TIES）。
3. **跨模态准确率-效率帕累托前沿**：在 2B–30B 稠密与 MoE 架构、文本/视觉-语言/音频推理的 28 个评估环境中，S³ 平均较 Thinking 模型减少 27.4% 推理 token，同时整体准确率提升 1.0 个百分点，建立多条训练自由组合策略的帕累托最优解。
4. **机制解释与注意力熵理论化**：引入注意力熵作为可解释性指标，揭示保留 Null-space 模型注意力更集中（$H_{\mathrm{Null}} < H_{\mathrm{Base}} < H_{\mathrm{Sub}}$），并提供基于局部最优假设的简化分析模型，从一阶/二阶梯度角度理论化子空间扰动为何增加熵而空域扰动降低熵。

## 方法详解
- **权重差分与谱投影分解**：设 $W_0$ 为 Non-thinking 权重，$W_t$ 为 Thinking 权重，差值 $\Delta W = W_t - W_0$。对 $W_0$ 作紧凑 SVD $W_0 = U \Sigma V^\top$，构造正交投影算子 $P_\mathcal{S}(X) = U(U^\top X V)V^\top$。将差分精确拆分：
  $$\Delta W = \underbrace{P_\mathcal{S}(\Delta W)}_{\Delta W_\parallel} + \underbrace{(I - P_\mathcal{S})(\Delta W)}_{\Delta W_\perp}$$
  函数空间响应通过实际前向传播计算隐藏状态偏移 $\Delta h$ 并取均方范数，避免一阶 Jacobian 近似误差。
- **保护子空间与 $\rho$ 控制**：实际取前 $k = \lceil \rho \min(m, n) \rceil$ 个主导奇异向量构成保护子空间 $\mathcal{S}_\rho \subseteq \mathcal{S}$，投影算子更新为 $P_{\mathcal{S}_\rho}$。
- **S³ 组合公式**：组合权重
  $$W_\rho = P_{\mathcal{S}_\rho}(W_0) + (I - P_{\mathcal{S}_\rho})(W_t)$$
  等价表述为从 $W_0$ 出发，丢弃 $\Delta W$ 中

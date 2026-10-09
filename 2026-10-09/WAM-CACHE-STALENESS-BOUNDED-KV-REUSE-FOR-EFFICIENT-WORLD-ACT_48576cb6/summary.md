---
title: "WAM-CACHE-STALENESS-BOUNDED-KV-REUSE-FOR-EFFICIENT-WORLD-ACT"
source: https://arxiv.org/pdf/2610.11401v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-10-09 17:29:14"
field: "机器人操作中的高效推理"
keywords: ["World Action Model", "KV caching", "robot manipulation", "inference acceleration", "training-free"]
innovations: ["首个面向 WAM 视频 DiT prefill 的免训练 KV 复用框架，无需重训", "发现策略 cross-attention 比视觉漂移更适合作为刷新信号", "引入 age bound 打破 rank-based 选择的自我引用误差累积"]
benchmarks: ["RoboTwin 2.0", "LIBERO", "AIRBOT Play 真机任务"]
---

# 论文速读：WAM-CACHE-STALENESS-BOUNDED-KV-REUSE-FOR-EFFICIENT-WORLD-ACT

## 一句话总结
论文提出 **WAM-Cache**，这是首个面向 World Action Model (WAM) 视频 DiT 预填充阶段的免训练 KV 缓存框架，通过联合"动作专家注意力"+"潜在惊喜检测"+"年龄上限"三信号选择稀疏刷新 token，在不重训练的情况下将预填充 FLOPs 降低 32–42%，仿真成功率仅下降 0.7–1.8 pp、真机下降 2.5 pp。

## 研究问题与动机
- **预填充成 WAM 主要瓶颈**：在闭环控制中，视频 DiT 需在每个 chunk 对当前观测做一次完整 prefill，生成 layerwise KV 对供 action expert 查询；当 action-denoising 步数减少（如 Fast-WAM 中 N=2），prefill 占比高达 83.5%。
- **现有免训练加速无法触及该瓶颈**：已有训练-free 缓存主要复用 denoising 步间特征或同一 frame 内静态 token，缺少对单次 video DiT prefill 的稀疏化手段。
- **"刷新变动的 token"直觉失效**：论文消融显示仅按视觉漂移刷新会导致成功率大幅下跌（比 dense baseline 低约 25 pp），即便 oracle 提前知道 ground-truth KV 漂移也无济于事。
- **核心洞察**：下游动作准确率取决于"action expert 在哪里 attention"，而非"哪里视觉上发生了漂移"，因此刷新集合的构建应优先考虑策略敏感性。

## 核心贡献（创新点）
1. **首个面向 WAM 视频 DiT prefill 的免训练 KV 缓存框架**。与 FastV/ToMe/Eventful 等直接剪枝或合并 token 的方法本质不同，WAM-Cache 保留全量 token 但仅对稀疏子集重算 prefill，架构透明、无需重训。
2. **发现"策略注意力 > 视觉漂移"的刷新准则**。即使 oracle 精确知道 KV 漂移方向也仅能恢复到 65.3%（dense 为 90.1%），而动作专家 cross-attention  Alone 即可恢复到 81.5%，揭示了稀疏刷新选择信号应以下游敏感度为核心。
3. **引入年龄上限（age bound）打破自我引用误差累积**。两种内容信号均基于 rank 排名，未被选中 token 可无限次错过刷新导致 stale error 累积；通过强制每隔 G 个 chunk 刷新一次，成功差距缩小至仅 1.8 pp。
4. **三信号 UNION 构建刷新集且互不干扰**。surprise 集与 attention 集基本不相交，union 后既捕捉动态物理变化又覆盖策略关键静止区域，且计算开销可忽略。

## 方法详解
- **缓存状态**：每个 token i 维护 (i) KV 对 {K^l[i], V^l[i]}_l；(ii) 最近刷新时刻的参考嵌入 z_{τ_i}[i]；(iii) 连续复用计数 g_i；(iv) 上一 chunk 的 action expert cross-attention 质量 p_t[i]。
- **刷新集构造**：R_t = S_t ∪ P_t ∪ A_t，其中 S_t 为 latent surprise top-k，P_t 为 action-attention top-k，A_t 为 g_i ≥ G 的强制刷新集。
- **Latent Surprise**：s_t[i] = ||z_t[i] - z_{τ_i}[i]||_2 / (0.5(||z_t[i]||_2 + ||z_{τ_i}[i]||_2) + ε)，测量当前嵌入相对参考嵌入的归一化位移，TopK 选出变化最大的 token。
- **Action-conditioned Selection**：p_t[i] 为 action expert 在第一去噪步对所有 head 在全部 L_a 层的 cross-attention 均值；利用前序 chunk 已算好的 attention map，零额外开销获取。
- **Age Bound**：A_t = {i : g_i ≥ G}，G=2 时每个 token 最多连续复用 2 个 chunk 即强制刷新，严格有界 staleness。
- **混合缓存执行**：video DiT 只对 R_t 内的 token 做前向，H_t^l[R_t] 重算后原地覆盖 K^l, V^l；queries 仅对新刷新 token 生成并 attend 全部 S_v 个 keys/values，action expert 在混合缓存上正常去噪。
- **预填充 FLOPs 分析**：C_dense ≈ 8S_v d² + 4S_v² d + 4S_v d² + 4S_v S_c d + 4S_v d d_f（自注意 + 交叉注意 + MLP），WAM-Cache 下 C(ρ_t) = ρ_t · C_dense，节省比例为 1-ρ_t。

## 实验与结果
- **仿真基准**：
  - **RoboTwin 2.0**（50 个双臂任务，clean/randomized 各 100  episodes）：在匹配 FLOPs 下，WAM-Cache 删除 41.6% prefill FLOPs 仅降 1.8 pp（clean）/ 39.3% FLOPs 降 1.3 pp（randomized）；更强 baseline FastV 同预算下降 27.4 pp，VLA-Cache 降 4.0 pp。更大预算 (b_a=0.6,b_s=0.1) 下仅降 28.2% FLOPs、成功率差距缩至 0.1 pp。
  - **LIBERO**（4 suite×10 单臂任务）：删除 32.3% FLOPs 平均成功率 95.85% vs dense 96.50%；更大预算删除 26.1% FLOPs 完全对齐 dense。
- **CUDA 延迟**：S_v=120 时 1.23×；S_v=480 时 1.47×；S_v=1080 时 1.67×，接近理论上界 1/ρ̄。
- **真实机器人**（AIRBOT Play 双臂，20 trials/task）："胡萝卜放入碗"成功率持平（90.0% vs 90.0%），"堆叠立方体"仅失败 1 次（75.0% vs 80.0%），平均 82.5% vs 85.0%，FLOPs 减 40.2%。
- **最强结果**：RoboTwin 2.0 clean 场景下，大预算 WAM-Cache 以 -28.2% FLOPs 换取 -0.1 pp 性能损失，为仿真最强性价比点。

## 相关工作脉络
- **FastV / ToMe**：基于单帧内 token 剪枝/合并，改变视觉输入表征而非复用 KV，WAM-Cache 保留全 token 语义。
- **Eventful Transformer / VLA-Cache**：复用跨帧静态 token，但复用无时间上限；WAM-Cache 通过 age bound 有界复用，且信号取自 action expert 而非语言 token。
- **DeepCache / X-Cache**：复用 denoising 步间特征，适用扩散模型内部迭代；WAM-Cache 针对的是单次 prefill 的跨 chunk 复用。
- **SnapKV / H2O**：LLM KV 缓存剪枝依据 hitter 统计或复杂度；WAM-Cache 面向视觉 token，信号为 latent surprise + action attention。
- **VLA-Cache**：结合"视觉变化"+"text attention"决定静态 token；WAM-Cache 关键区别是用 action expert 的 cross-attention 替代 text attention，并引入 age bound 防止误差无限累积。

## 局限性与未来方向
- **小 S_v 下加速落后于 FLOPs 节省**：权重加载与 kernel launch 固定开销占比大，S_v=120 时实测加速未达理论 1/ρ̄；随分辨率/视角增加差距将缩小。
- **继承注意力存在相位滞后**：Long suite 多阶段任务中 phase 切换初期动作专家 attention 尚未调整，导致部分任务精度下降。
- **年龄上限超参需调**：G=2 在多数场景表现好，但对长周期静态任务可能偏激进。
- **未来方向**：可扩展到多相机、长 observation history 等高 S_v 场景；与 diffusion step 压缩、蒸馏正交叠加；探索自适应 G/b_a/b_s 调度。

## 研究启发与可借鉴点
- **"下游敏感度优先于局部漂移"**这一设计原则可迁移至任何"编码器输出被下游模块消费"的缓存场景（如 VLM/VLA 的 visual encoder KV 复用）。
- **三信号 UNION + age bound 的组合范式**可复用于其他序列模型的跨 step/token 缓存，尤其适合存在"短期稳定 + 长期漂移"特征的视觉-动作回路。
- **复用已计算的 attention map 作为零开销信号**的思路非常实用，在 Action expert 已跑完第一去噪步后即可获得下一 chunk 的 priority map。
- **与已有加速技术（diffusion step reduction、distillation）正交**的设计哲学，意味着本方法可作为 plug-in 模块无缝集成到现有 WAM 推理管线。
- **大预算逼近 dense 但保留 28% FLOPs 节省**的工程权衡思路：在需要高精度时可选择较大 b_a 保留稳健性，在需要高吞吐时收紧预算。

## 关键术语表
- **World Action Model (WAM)**：将视频 Diffusion Transformer 作为 backbone、通过 action expert cross-attend 其 KV 输出来生成机器人动作策略的通用操控模型。
- **Prefill**：视频 DiT 对当前观测做一次完整前向传播，生成 layerwise KV 对供 action expert 查询，是整个 WAM 每 chunk 的主要计算开销。
- **Latent Surprise**：通过当前 token 嵌入与参考嵌入的归一化 L2 距离衡量"视觉漂移程度"，用于选择需要刷新的高变化 token。
- **Action-conditioned Selection**：利用 action expert 在上一 chunk 的 cross-attention 分布，挑选策略敏感度高但可能视觉变化小的 token。
- **Age Bound (G)**：每个 token 连续被复用的最大 chunk 数，超过则强制刷新，防止 stale error 无界累积。
- **Staleness-bounded KV Reuse**：在保证任意 KV 对不超过 G 个 chunk 陈旧性的前提下进行复用，兼顾缓存收益与精度。
- **Fast-WAM**：本文评测基础模型，删去 future imagination 仅靠单次 prefill + N 步 action denoising 完成控制。
- **RoboTwin 2.0 / LIBERO**：主流双/单臂机器人操作 benchmark，前者 50 任务含 clean/randomized 场景，后者含 Spatial/Object/Goal/Long 四 suite。

## 可复现要素
- **数据集**：RoboTwin 2.0、LIBERO、AIRBOT Play 真机任务（演示数据由团队 teleoperation 收集，论文未公开演示数据集本身）
- **代码/权重**：项目页面 https://dingkai0302.github.io/wam-cache/（论文声明开源情况待该页面确认）；Fast-WAM checkpoint 沿用最原始发布版本
- **关键超参**：G=2（所有实验不变）；默认预算 (b_a, b_s)=(0.4, 0.1)，大预算 (0.6, 0.1)；LIBERO 因 wrist view 占比更高取 b_a=0.5/0.6；N=2 为默认 action-denoising 步数

---
title: "Sol-H3-Recursive-Self-Improvement-for-MiniMax-H3-Inference-A"
source: https://arxiv.org/pdf/2609.35110v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-10-01 03:49:04"
---

# 论文速读：Sol-H3: Recursive Self-Improvement for MiniMax-H3 Inference Acceleration on Sol-Engine across Cloud and Edge

## 一句话总结
论文针对 33B 参数的 MiniMax-H3 音视频生成模型，提出 Sol-H3 全栈推理加速方案：通过跨分辨率两阶段生成流水线与跨 VAE 潜在适配器消除算法冗余，再经递归自改进（RSI）自动搜索算子融合与量化布局，在 8×GB200 上实现约 30× 加速（5s 视频 1.434s，3.5× 实时）并在单卡 DGX-Spark 上完成 <1 分钟的完全内存驻留生成。

## 研究问题与动机
1. **算力与延迟瓶颈**：MiniMax-H3 含 33B 参数且需 49 步迭代去噪，云侧部署受限于吞吐量与单次生成延迟，边缘侧（如 DGX-Spark）受 HBM 容量严格约束，超限额即触发 cpu-offloading 导致延迟崩溃。
2. **单类优化策略的局限性**：算法级加速（减步数、稀疏注意力）通常以损失生成质量为代价；系统级加速（算子融合、通信调度）在不改变生成契约的前提下收益有限，二者难以单独应对从云到边缘的全场景需求。
3. **多模态保真约束**：与纯视觉模型不同，H3 同时生成视频与音频，音频轨道必须在所有优化环节保持保真，无法像视觉部分那样灵活近似。
4. **跨 VAE 衔接开销**：两阶段流水线需在 H3 与 LTX-2.5 之间转移潜在表示，传统像素级 decode-reencode 往返引入额外计算与显存峰值，阻碍端到端部署。

## 核心贡献（创新点）
1. **跨分辨率两阶段生成流水线**：将 49 步全分辨率去噪重构为 4 步低分辨率全局布局 + 潜在传递 + 3 步高分辨率精修的不对称调度，从算法层面消除均匀分辨率下早期冗余计算。与已有工作的本质区别在于：本文改进的是端到端 serving 实现与跨 VAE 适配的工程闭环，而非仅提出调度概念。
2. **跨 VAE 潜在空间适配器（Latent-to-Latent Adapter）**：训练 194.76M 参数翻译模块直接映射 H3 至 LTX 潜在坐标，手递手开销从 12,748.6ms 降至 63.1ms（202× 加速），峰值内存从 13.79 GiB 降至 0.785 GiB。与朴素 VAE 往返方案的本质区别在于彻底避免 decode/reencode 两个大网络及其中间激活的驻留。
3. **Fail-closed 约束下的递归自改进（RSI）循环**：在严格"不改变生成契约"的硬约束下自动搜索 kernel fusion、内存布局、量化格式与通信调度，区分 bit-exact 布局变换与算术融合/近似模式。与 Agentic 优化系统的本质区别在于：本文 RSI 仅探索固定数学轨迹的实现变体，拒绝改变步数或分辨率等核心参数。
4. **精度保持的 LoRA Consumer Fusion**：保留 base 与 LoRA 分支独立计算，在消费者节点融合加法，避免 BF16 下 86–94% 小更新被舍入归零导致的生成轨迹偏移。与直接权重合并的本质区别在于：在零位点对齐前提下恢复 46.3% 额外运行时，而不改变生成结果。

## 方法详解
**跨分辨率两阶段生成**：原始 49 步 1344×768 去噪重设为 Stage 1（4 步 672×384×124 帧，MiniMax-H3 + FastH3 VSA DataFree LoRA 生成全局内容与主音频轨）→ latent handoff → Stage 2（3 步 1344×768，LTX-2.5 dev backbone + distilled LoRA strength=0.8 精修高频细节）。

**跨 VAE 潜在适配器**：H3（24 通道，空间压缩 16×，17 帧 chunk 编码）与 LTX（128 通道，空间压缩 32×，因果 8× 时间网格）表示差异通过固定几何对齐 + 参数化残差网络解决：线性插值特征拼接至 LTX 时间位置，空间 pixel-unshuffle 合并 $2\times2$ 邻域为通道维，产生 $24\times(1+3)\times4=384$ 通道；冻结 ridge-regression 初始化的 $1\times1\times1$ 仿射跳连 + 22 层宽度 752 残差块（空间卷积 + 深度可分离时间卷积 + gated channel MLP）。损失函数：
$$
\mathcal{L}_{\text{adapter}} = \text{MSE}(\hat{z}_L, z_L) + \alpha \cdot \text{MSE}(D_L(\hat{z}_L), D_L(z_L)), \quad \alpha=160
$$
梯度经冻结的 LTX 解码器回传，VAE 权重全程冻结。

**RSI 自动化算子优化**：在严格 fail-closed 约束下迭代搜索并验证以下四类优化：
- **通信布局**：packed QKV 存储将三次 QKV 收集合并为两次 all-to-all；stride-aware 内核一次读取写入消除双重 copy。
- **量化通信**：block-INT8（400 bytes/token/head，含 128 值字节 + 12 scale codes + 4 padding）或 FP8（128 raw E4M3 bytes）两种 wire format。
- **MXFP8 GEMM**：Transformer blocks 2–46 的注意力与 FFN 使用 E4M3 值 + 组大小 32 的 E8M0 scale， fused RMSNorm/modulation 与 SwiGLU producer 直接写入量化张量。
- **三组算子融合**：(1) residual+indexed gating+RMSNorm+modulation；(2) QK norm+partial RoPE；(3) SwiGLU split+SiLU+mul。

**稀疏注意力（Sol-Attn）**：Refinement 阶段三步的阈值 $\mu+\alpha\sigma$ 中 $\alpha$ 依次为 1.0、1.25、1.5；前两层保持 dense，prefix（文本 + 条件视频 + 音频）作为 always-sink 参与 dense cross-attention，target video 查询使用 block-sparse attention。

**VAE 并行解码**：196 个空间 tile 在 8 卡 context parallelism 下全局 batched 编译解码，峰值内存从 18.8 GiB 降至 15.6 GiB。

**边缘专属优化**：
- **AdaLN precompute**：一次性预计算全部 denoising step 的调制表（≈1.5 GiB），替代 2500 次重复投影，节省 ~24 GiB HBM。
- **Prompt caching**：Stage 2 使用离线 INT8 Gemma 编码器缓存的通用提示上下文（4K refined, cinematic detail…），在线推理不加载 Stage-2 文本编码器，释放 16.2 GiB。
- **权重量化**：FP8 H3 draft DiT（23.4 GiB）+ NVFP4 AWQ Qwen encoder（14.6 GiB）。

**Reference KV Caching**：首步完整计算 reference branch 的 K/V 并缓存，后续步骤仅对 generated tokens 计算自注意力，reference tokens 的 self-attention 完全跳过。

## 实验与结果
**数据集与评估设置**：适配器评估使用 256 个 held-out 视频（与训练/开发/更早评测集不重叠），报告 PSNR/SSIM；端到端延迟在 NVIDIA GB200（1/4/8 卡）、DGX-Spark、RTX 5090 上测试；LoRA 融合在四 GB200 上验证 byte-for-byte 一致性。

**基线**：MiniMax-H3 官方 49 步全分辨率 serving baseline（SGLang [60]）[35]。

**主要数值结果**：
- **8×GB200**（1344×768，24fps）：5s 视频 1.434s（3.5× real-time），10s 视频 3.350s，15s 视频 6.062s；相比 baseline 加速 12.73×–16.99×（Table 1）。
- **单卡 GB200**：5s 视频 6.226s，22.2× speedup over SGLang baseline。
- **DGX-Spark**（单卡）：5s 视频 56.17s，31× speedup，总内存 116.9 GiB（较 naive 两阶段 142.4 GiB 降低 17.9%），完全 HBM resident 无 cpu offload。
- **RTX 5090**：5s 视频 39.80s，26.3× speedup（prompt encoding 成为新瓶颈，占 34.1%）。
- **跨 VAE 适配器质量**：最终 checkpoint PSNR 30.618 dB / SSIM 0.8962，相较早期 checkpoint 提升 1.205 dB（95% CI [1.166, 1.245]）。
- **适配器转换开销**：63.1ms vs 完整 VAE 往返 12,748.6ms（202× 加速）；峰值内存 0.785 GiB vs 13.788 GiB。
- **LoRA Consumer Fusion**

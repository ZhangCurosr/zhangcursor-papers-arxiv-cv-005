---
title: "ViTeX-Bench-Benchmarking-High-Fidelity-Video-Scene-Text-Edit"
source: https://arxiv.org/pdf/2609.40356v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-10-01 21:32:54"
field: "视频生成与编辑"
keywords: ["video scene text editing", "video benchmark", "OCR evaluation", "temporal consistency", "diffusion video editing", "character accuracy", "glyph conditioning"]
innovations: ["首个视频场景文字编辑三轴13指标评测基准（文字正确性/时间质量/局部保留）", "基于运动对齐的glyph-video条件流实现高精度时序稳定文字渲染", "Substring Edit Distance度量与Composite确定性后处理控制协议"]
benchmarks: ["ViTeX-Bench", "ViTeX-Dataset"]
---

# 论文速读：ViTeX-Bench: Benchmarking High-Fidelity Video Scene Text Editing

## 一句话总结
本文提出了 ViTeX-Bench，首个面向视频场景文字编辑（video scene text editing）的完整评测基准，包含 387 个真实视频数据集、13 项三轴指标协议，以及开源参考编辑器 ViTeX-Edit-14B，系统性地揭示了正确文字渲染、时间一致性与场景保留三者之间的固有 trade-off。

---

## 研究问题与动机

1. **视频场景文字编辑缺乏系统性评测资源**：现有视频编辑基准（VBench、EditBoard、VEFX-Bench 等）评估指令遵循、感知质量或保留能力，但无法度量"文字是否在某帧中被正确渲染且在整个视频中保持稳定"这一核心诉求。
2. **已有图像文字编辑方法无法直接解决视频问题**：FLUX-Text、TextCtrl 等图像编辑器对每帧独立编辑，会产生笔画闪烁和漂移（glyph drift）；首帧编辑+传播方法（TextCtrl+AnyV2V）在长视频中文字会逐渐褪化或偏离。
3. **正确性、时间稳定性与局部保留难以兼顾**：实验揭示三种典型失败模式——Wan2.1-VACE 保持原文未编辑、FLUX-Text 帧级正确但时序不稳、Kling 视觉精美但文字错误，表明三者存在内在张力。
4. **视频场景文字数据集严重匮乏**：现有资源仅有少量无配对真实视频数据；本文构建了含 230 个配对训练样本 + 157 个冻结评测样本的标准化数据集。

---

## 核心贡献（创新点）

1. **ViTeX-Dataset：首个含配对编辑的真实视频场景文字编辑数据集**——387 个 720p/120 帧/24fps 真实视频，含逐帧文字区域 mask 与源/目标字符串，230 个样本通过半自动人工审核流水线生成配对编辑，157 个样本永久冻结用于评测。
2. **三轴 13 指标评估协议**——以字符级正确性（SeqAcc/CharAcc/TTS）、视觉与时间质量（Flicker/Warp/MUSIQ）、编辑局部保留（PSNR/SSIM/LPIPS/DreamSim_loc）为三个评估轴，每轴一个主指标（SeqAcc↑、Warp_c↓、DreamSim_loc↓），并引入 Pareto 比较刻画 trade-off。
3. **Substring Edit Distance 评测度量**——定义基于子串对齐的编辑距离，允许目标字符串作为候选识别结果的连续子串匹配，避免误惩罚正确文本被额外字符包围的情况。
4. **ViTeX-Edit-14B 参考编辑器**——在 Wan2.1-VACE-14B 基础上引入运动对齐的 glyph-video 条件流（目标字符结构 + 源文字轨迹透视仿射变换），在视频原生编辑器中达到最高 CharAcc（0.688）和最低 Warp_c（1.53）。
5. **Composite 确定性后处理控制协议**——提供统一的训练无关后处理包装，分离文字合成质量与背景还原效果，使不同编辑方法的评测具有可比性。

---

## 方法详解

### 数据集构建流水线

对每个源视频 $V$，构建四个资产 $(M, (s_{\text{src}}, s_{\text{tgt}}), V_{\text{clean}}, p_1^{\text{new}})$：

- **Mask $M$**：标注者在首帧用 SAM 3 关键点标注文字区域，自动传播至剩余 119 帧，经 25×25 椭圆核形态学膨胀 3 次得到带边距的膨胀 mask。
- **源/目标字符串**：Qwen3-VL-32B-Instruct 从首帧 mask 裁剪读取 $s_{\text{src}}$，建议相似长度的 $s_{\text{tgt}}$，由标注者审计。
- **干净背景视频 $V_{\text{clean}}$**：使用 removal-1.3B（Wan2.1-VACE-1.3B 微调版，类 ROSE 副作用感知训练）移除文字及其投影阴影和高光。
- **首帧目标文字 patch**：Gemini 3 Pro Image 根据目标字符串改写首帧，再经 SAM 3 二次 mask 得到 $p_1^{\text{new}} = f_1^{\text{edit}} \odot m_1^{\text{new}}$。

**两种合成策略**：
- Strategy A（Alpha 合成）：适用于静态文字区域，将 $p_1^{\text{new}}$ 逐帧 alpha 混合到 $V_{\text{clean}}$。
- Strategy B（PISCO 插入器）：适用于动态文字，使用 fine-tuned PISCO-14B 以首帧 patch 为参考插入文字，辅以 amodal completion 监督。

### 评估协议

**文字正确性**（基于源可检测帧集合 $\mathcal{D}$）：
$$\text{SeqAcc} = \frac{1}{|\mathcal{D}|}\sum_{t \in \mathcal{D}} \mathbf{1}[d_{\text{sub}}(s_{\text{tgt}}, \hat{s}_t) = 0]$$
$$\text{CharAcc} = \frac{1}{|\mathcal{D}|}\sum_{t \in \mathcal{D}} \text{Sim}(s_{\text{tgt}}, \hat{s}_t), \quad \text{Sim}(r,c) = 1 - \frac{d_{\text{sub}}(r,c)}{\max(|r|,1)}$$
$$\text{TTS} = \frac{1}{|\mathcal{P}|}\sum_{(t,t+1) \in \mathcal{P}} \mathbf{1}[\hat{s}_t = \hat{s}_{t+1}]$$
其中 $d_{\text{sub}}$ 为子串编辑距离（前缀/后缀免费），$\mathcal{P}$ 为相邻可检测帧对。

**视觉/时间质量**（全帧 scope $f$ 和文字 crop scope $c$）：
$$\text{Flicker}_S = \text{mean}_{t} \text{MAE}(x^S_{t+1}, x^S_t), \quad \text{Warp}_S = \text{mean}_{t} \text{MAE}(x^S_t, \mathcal{W}(F^{\text{src}}_{t \to t+1}, x^S_{t+1}))$$
Crop box 为固定边界框（所有帧 mask 的并集 + 16px 边距），确保 Flicker_c 和 Warp_c 测量的是 glyph 漂移而非边界框抖动。

**编辑局部保留**：构造局部预测 $\hat{f}^{\text{loc}}_t = (1-m_t) \odot \hat{f}_t + m_t \odot f_t$，对 mask 外像素计算 PSNR/SSIM/LPIPS/DreamSim（即 DreamSim_loc）。

### ViTeX-Edit-14B 架构

基于 Wan2.1-VACE-14B（DiT hidden=5120, 40 blocks），在原有文本编码器（uMT5-XXL）和 VCU 之外增加第三条条件流：

- **Glyph video $G_{\text{vid}}$ 构建**：用匹配源字体风格的 typeface 将 $s_{\text{tgt}}$ 渲染为白底黑字 glyph 图像，EasyOCR 检测首帧文字四边形，CoTracker3 跟踪 119 帧，投影仿射变换得到跟随源文字轨迹的 glyph 视频。
- **Glyph 编码器**：Frozen Wan VAE 编码 $G_{\text{vid}}$ → stride(1,2,2) patch embedding → 64 个 learnable query 进行 cross-attention pooling：
$$E_G = W_{\text{out}} \cdot \text{CrossAttn}(Q_{64}, \text{LayerNorm}(z_G))$$
- **条件 Cross-Attention**：每个 VACE block 新增零初始化残差条件层：
$$h' = h + W_o \cdot \text{FlashAttn}(W_q \text{LN}(h), W_k E_G, W_v E_G)$$
- **训练**：两阶段 Flow-Matching SFT，共 576 GPU-hours（8×H100 80GB），冻结主 DiT 和 uMT5-XXL，仅训练 VACE 分支（≈3.89B）+ glyph 编码器（≈132M）+ 条件 cross-attention。

### Composite 后处理

确定性无训练后处理，对任意编辑器通用：
1. **Annulus LAB 颜色迁移**：以 mask 周围 41×41 膨胀环带 $B_t$ 为参考，Reinhard 均值-方差迁移修正预测帧颜色。
2. **Feathered alpha（w=4px）**：基于 signed distance transform 构造平滑过渡 mask。
3. **合成**：$\hat{f}^{\text{Composite}}_t = (1-\alpha_t)f_t + \alpha_t \tilde{f}_t$。

---

## 实验与结果

### 数据集统计

| 指标 | 数值 |
|------|------|
| 总视频数 | 387（训练 230 + 评测 157） |
| 分辨率/帧率/帧数 | 1280×720, 24fps, 120 帧 |
| 字符串长度 | 源 8.0±5.7，目标 8.0±5.4（1–41 字符） |
| 字体类型 | 印刷体 23%，手写体 44%，艺术体 33% |
| 文字脚本 | Latin 150, Chinese 4, Japanese 1, Cyrillic 2 |
| 动态/静态 | 约 75% 动态，25% 静态 |
| Mask 面积比 | 0.032±0.022 |

### 主要结果（表 2）

| 方法 | SeqAcc↑ | CharAcc↑ | TTS↑ | Warp_c↓ | DreamSim_loc↓ |
|------|---------|----------|------|---------|---------------|
| Source（参考） | 0.000 | 0.317 | 0.760 | 1.27 | 0.000 |
| FLUX-Text | **0.528** | **0.737** | 0.326 | **13.01** | 0.012 |
| TextCtrl | 0.475 | 0.734 | 0.511 | 2.09 | **0.004** |
| ViTeX-Edit-14B | 0.341 | **0.688** | **0.648** | **1.53** | 0.024 |
| Wan2.1-VACE-14B | 0.000 | 0.298 | 0.689 | 1.56 | 0.007 |
| Kling Video 3.0 Omni | 0.000 | 0.208 | 0.641 | 2.90 | 0.061 |

**关键发现**：
- FLUX-Text CharAcc 最高（0.737），但 Warp_c 极大（13.01），反映帧级独立编辑的时序不稳定性。
- ViTeX-Edit-14B 在视频原生编辑器中 CharAcc 最高（0.688），且 Warp_c 最低（1.53，接近 source 的 1.27），TTS 也最高（0.648）。
- Wan2.1-VACE-14B SeqAcc=0，说明默认 mask-conditioned 视频编辑器倾向于保留源文字而非替换。
- Composite 后处理将 ViTeX-Edit-14B 的 DreamSim_loc 从 0.024 降至 0.002，PSNR_loc 从 29.08 dB 升至 42.95 dB，表明大部分局部保留误差来自背景重建而非合成区域本身。

### OCR 校准与人工评估

- 源文字 OCR 在可检测帧上 exact match = 0.851，CharAcc = 0.966。
- 方法盲态人工转录与 OCR 排名 Spearman ρ = 0.95。
- 人工评分（3 位非作者评审，70 个输出）：文本轴 Krippendorff α = 0.87，时间轴 α = 0.80，局部保留轴 α = 0.37（一致性较低）。
- Warp_c 与时间评分相关系数 −0.40，优于全帧 Warp 的 −0.20，验证了 warp_c 作为时间主指标的合理性。

### Pareto 前沿（表 11）

五项方法构成非支配集：FLUX-Text（高正确性/高 Warp）、TextCtrl（平衡）、RS-STE（低 Warp/中正确性）、ViTeX-Edit-14B（最低 Warp）、Wan2.1-VACE-14B（零正确性但最优稳定性与局部保留）。说明三轴之间存在不可同时优化的本质 trade-off。

---

## 相关工作脉络

1. **视频生成/编辑基线**：HunyuanVideo [1]、CogVideoX [2]、Wan [4]、VACE [5]、Kling Video 3.0 [6] 等大模型为视频编辑提供了骨干网络；ViTeX-Edit-14B 建立在 Wan2.1-VACE-14B 之上。
2. **图像场景文字编辑**：FLUX-Text [11]、TextCtrl [10]、RS-STE [12]、AnyText2 [9] 为代表，分别采用 FLUX backbone、prior guidance、recognition-supervised 等不同设计；本文将其作为 Family A 基线（每帧独立编辑）。
3. **首帧编辑+传播**：TextCtrl+AnyV2V [10,13] 代表 Tune-a-Video / TokenFlow / Video-P2P 等无训练传播范式的典型应用，暴露了首帧编辑在长视频中文字漂移的固有缺陷。
4. **视频文字编辑先例**：STRIVE [65] 是最接近的先驱，但仅评估编辑区域本身，无字符级正确性度量与全帧时间一致性评估；本文将其定位拓展为三轴全面评测。
5. **评测基准**：EditBoard [23]、FiVE [24]、VEFX-Bench [27]、VBench [21] 等通用视频编辑基准缺少 OCR 锚定的字符级度量；本文填补了这一空白，首次将字符正确性作为核心维度。
6. **Glyph 编码器灵感**：GlyphMastero [18] 提出显式 glyph 编码器为图像文字编辑提供笔画级引导；ViTeX-Edit-14B 将此思想延伸至视频，引入运动对齐的 glyph-video stream。

---

## 局限性与未来方向

1. **覆盖范围有限**：数据集以拉丁字母为主（150/157），中文/日文/西里尔字母仅 7 个样本；手写体和艺术体虽有代表，但密集排版、弯曲表面、严重遮挡、极端运动仍未充分覆盖。
2. **非拉丁脚本表现差距大**： pooled 7 个非拉丁样本上，除 AnyText2（CharAcc 0.295）外所有编辑器 CharAcc 均低于 0.19，多语言支持是明确短板。
3. **评测度量依赖 OCR**：OCR 准确率受脚本、字体、可读性影响；五个评测视频无可检测源帧，无法评估 SeqAcc/CharAcc。
4. **人工评估一致性不均**：局部保留维度 Krippendorff α = 0.37，说明人类对"保留质量"的判断主观性较强。
5. **Paired 训练数据的真实性存疑**：配对编辑由模型辅助流水线生成，可能包含渲染错误；首帧编辑重试率和字符串拒绝率未记录。
6. **参考编辑器组件消融未完成**：ViTeX-Edit-14B 的 glyph 条件流设计尚需受控消融，训练规模可扩展性待研究。

---

## 研究启发与可借鉴点

1. **Substring Edit Distance 的设计**：允许候选识别结果中包含目标字符串作为子串即不扣分，对 OCR 后验评估有通用参考价值，可用于任何文本识别评测场景。
2. **Composite 后处理控制的实验设计理念**：将"文字合成质量"与"背景还原质量"分离评估，通过统一的确定性后处理让不同架构方法在可比条件下竞争，是编辑类评测中值得推广的实验设计。
3. **三轴主指标 + Pareto 比较的评测哲学**：放弃单一加权分数，转而报告 Pareto 前沿，让读者自行判断 trade-off 偏好，比单一排行榜更具信息量和说服力。
4. **Warp_c 优于全帧 Warp 作为时间质量指标**：固定边界框测量 glyph 漂移而非 bounding-box jitter，避免了文字区域运动带来的混淆，这一设计可用于其他视频编辑任务的时序评估。
5. **运动对齐的 glyph-video 条件流**：将目标字符结构沿源文字轨迹做透视仿射变换，使字符几何与场景运动对齐，这一思路可迁移到其他需要精确定位的结构化内容插入任务（如徽标、水印、字幕）。

---

## 关键术语表

- **ViTeX-Bench**：首个视频场景文字编辑评测基准，含数据集、13 项三轴指标协议和开源参考编辑器。
- **Video Scene Text Editing**：在视频中对场景表面文字（招牌、白board、标签等）进行替换，同时保留周围内容、运动与相机动态。
- **SeqAcc（Sequence Accuracy）**：在源文字可检测帧上，识别结果与目标字符串完全匹配（子串距离为 0）的比例，文字正确性主指标。
- **CharAcc（Character Accuracy）**：基于子串编辑距离的相似度均值，对近似正确渲染给予部分评分，比 SeqAcc 更宽容。
- **TTS（Temporal Text Stability）**：相邻可检测帧解码字符串的一致性比例，衡量文字在时间维度上的稳定性。
- **Warp_c**：在固定文字 crop 边界框内，用 RAFT 光流将帧 t+1 反向映射到帧 t 后的像素 MAE，衡量编辑区域的时间扭曲程度。
- **DreamSim_loc**：mask 外像素区域的 DreamSim 感知距离，衡量编辑对非文字区域的破坏程度（越低越好）。
- **Composite**：确定性无训练后处理包装，通过 annulus LAB 颜色迁移 + feathered alpha 合成将预测区域与源背景融合。
- **Substring Edit Distance**：目标字符串与候选识别结果任意连续子串之间的最小编辑距离，前缀和后缀免费。
- **ViTeX-Edit-14B**：基于 Wan2.1-VACE-14B 的开源参考编辑器，引入 motion-aligned glyph-video 条件流，CharAcc 达 0.688。

---

## 可复现要素

| 要素 | 状态 |
|------|------|
| **数据集** | ViTeX-Dataset 已发布，Hugging Face: https://huggingface.co/datasets/ViTeX-Bench/ViTeX-Dataset（CC-BY-NC 4.0，训练 230 + 评测 157） |
| **评测代码** | 已开源，GitHub: https://github.com/taco-group/ViTeX-Bench（Apache-2.0） |
| **参考编辑器权重** | ViTeX-Edit-14B 权重已发布，Hugging Face: https://huggingface.co/ViTeX-Bench/ViTeX-Edit-14B（Apache-2.0） |
| **Leaderboard** | GitHub Pages: https://vitex-bench.github.io/ViTeX-Bench-Leaderboard/ |
| **训练超参** | Stage 1: 5 epochs @ 720p×49, lr=5e-5, batch=64; Stage 2: 2 epochs @ 720p×121, lr=1e-5, batch=64; AdamW WD=0.01; bf16; 8×H100 80GB |
| **推理配置** | 50-step single pass Flow-Matching |
| **关键模型** | PP-OCRv5（识别）、RAFT（光流）、MUSIQ（质量）、SAM 3（分割）、CoTracker3（追踪）、removal-1.3B（文字去除） |

---

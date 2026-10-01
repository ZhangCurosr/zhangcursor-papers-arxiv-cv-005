---
title: "SFE-VGGT-Source-Free-VGGT-Distillation-for-Event-Based-Monoc"
source: https://arxiv.org/pdf/2609.36929v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-10-01 17:44:21"
field: "事件相机几何感知"
keywords: ["Event-Based Depth Estimation", "Source-Free Distillation", "VGGT", "Cross-Modal Transfer", "Monocular Depth", "Event Camera"]
innovations: ["提出无源蒸馏框架 SFE-VGGT，以事件重建帧替代真实 RGB 作为几何教师", "设计密度感知特征蒸馏与置信度加权深度蒸馏，动态抑制重建伪影传播", "提出跨帧关系一致性损失，仅依赖学生自身时序深度排序维持几何稳定"]
benchmarks: ["EventScape", "MVSEC", "DENSE"]
---

# 论文速读：SFE-VGGT: Source-Free VGGT Distillation for Event-Based Monocular Depth Estimation

## 一句话总结
本文提出 SFE-VGGT，一种无源蒸馏框架，通过将冻结的 VGGT 几何先验迁移到事件相机域，彻底消除了对同步 RGB 图像或深度标注的依赖；在标准条件下性能逼近依赖 RGB 的 EventVGGT，并在夜间等恶劣光照场景中显著降低深度误差。

## 研究问题与动机
- **现有方法的部署瓶颈**：当前事件单目深度估计的跨模态蒸馏方法（如 Depth AnyEvent、EventDAM、EventVGGT）均严格依赖训练阶段同步的 RGB-事件配对数据或稠密深度标注，限制了多模态采集硬件受限场景的实际落地。
- **核心科学问题**：能否在不访问原始 RGB 模态的情况下，将图像基础模型的几何先验纯通过事件流蒸馏至事件深度学生模型？
- **源-free 设定引入的技术挑战**：仅从事件流重建的替代帧质量 imperfect 且具有空间不均匀性，直接均匀蒸馏会将重建伪影传播至学生网络，需设计可靠的加权机制。
- **夜间场景优势假设**：事件相机在高动态范围与低延迟上的本质优势在夜间尤为突出，探索无需 RGB 监督的蒸馏有望在该类场景中实现超越依赖图像基线的性能。

## 核心贡献（创新点）
1. **提出 SFE-VGGT 无源蒸馏框架**：利用冻结的 E2VID 将目标事件流重建为替代帧作为几何教师，完全替代真实 RGB 数据，消除对配对图像与深度 GT 的需求。
2. **设计可靠性感知蒸馏策略（DAFD + CWDD）**：通过事件密度权重与师生相对置信度动态调节特征对齐与深度监督强度，避免将重建不可靠区域的误差强制传递给学生。
3. **提出跨帧关系一致性损失（CFRC）**：仅利用替代帧的时空匹配点维持学生深度预测的时序相对顺序稳定性，无需依赖教师深度图的绝对时序一致性。
4. **验证无源几何迁移的可行性与夜间优势**：在 EventScape/MVSEC/DENSE 上证明该方法可逼近 RGB 依赖基线，且在 MVSEC 夜间序列上较 EventVGGT 降低 15.3% 的 10 m 深度误差。

## 方法详解
- **教师分支**：冻结的 E2VID 模型将连续事件流重建为 surrogate frames，输入冻结的 VGGT 骨干网络，输出教师特征 $f_{e2vid}$、深度 $d_{e2vid}$ 与置信度 $C_{e2vid}$。
- **学生分支**：同样初始化自 VGGT checkpoint，采用 LoRA 适配器（$r=16, \alpha=32$，约 1.6M 可训练参数）处理按时间窗口累积的事件表征 $I^{\mathrm{evt}}_s$，输出学生深度 $d_{\mathrm{evt}}$ 与置信度 $C_{\mathrm{evt}}$；推理时仅保留学生网络。
- **密度感知特征蒸馏（DAFD）**：计算第 $s$ 时刻第 $i$ 个 patch 的事件激活占比 $w_{s,i}=N_{s,i}/(P\times P)$ 作为空间可靠性信号，加权余弦特征对齐损失：$\mathcal{L}_{\mathrm{DAFD}}=\frac{\sum w_{s,i}[1-\cos(f_{\mathrm{evt}}^{s,i}, \mathrm{sg}[f_{e2vid}^{s,i}] )]}{\sum w_{s,i}+\epsilon}$，抑制稀疏区域的错误对齐。
- **置信度加权深度蒸馏（CWDD）**：构建师生相对置信度比值 $w_i=C_{e2vid,i}/(C_{\mathrm{evt},i}+C_{e2vid,i})$ 并做均值归一化 $\tilde{w}_i$，在 log 深度空间计算尺度不变误差 $r_i=\log d_{\mathrm{evt},i}-\mathrm{sg}[\log d_{e2vid,i}]$，结合梯度一致性项：$\mathcal{L}_{\mathrm{CWDD}}=\mathbb{E}[\tilde{w}_i r_i^2]-\lambda_{\mathrm{si}}(\mathbb{E}[\tilde{w}_i r_i])^2 + \gamma\mathbb{E}[\tilde{w}_i(|\nabla_x r_i|+|\nabla_y r_i|)]$。
- **跨帧关系一致性（CFRC）**：在替代帧上使用 SIFT+SIFT匹配与 RANSAC 剔除外点，获取可靠对应点 $(u^t_j, u^{t'}_j)$；以学生当前帧深度差符号 $r_k$ 为参考，约束下一帧对应点的深度相对顺序：$\ell_k=\mathbf{1}[r_k\neq0]\mathrm{softplus}(-r_k\Delta_k)+\mathbf{1}[r_k=0]\Delta_k^2$，最终损失 $\mathcal{L}_{\mathrm{CFRC}}$ 仅在帧对间平均，不依赖教师深度值。
- **总损失**：$\mathcal{L}=\lambda_{\mathrm{DAFD}}\mathcal{L}_{\mathrm{DAFD}}+\lambda_{\mathrm{CWDD}}\mathcal{L}_{\mathrm{CWDD}}+\lambda_{\mathrm{CFRC}}\mathcal{L}_{\mathrm{CFRC}}$，经验权重设为 $1.0, 2.0, 0.2$。

## 实验与结果
- **数据集与设置**：训练与域内测试使用 EventScape；零样本测试使用 MVSEC 与 DENSE；输入裁剪至 $252\times504$，采用 24 帧窗口，AdamW 学习率 $1\times10^{-5}$，单卡 RTX 5090 训练约 23 小时收敛。
- **EventScape 域内表现**：SFE-VGGT 在 10 m/20 m/30 m 误差分别为 0.57/0.87/1.23，与 RGB 依赖的 EventVGGT（0.54/0.79/1.06）接近；在 20 m 与 30 m 范围内优于所有监督型事件-图像融合方法（较 SRFNet 提升 48.2% 与 55.4%）。
- **MVSEC 零样本夜间优势**：在 Night1–Night3 平均误差上，SFE-VGGT 较 EventVGGT 降低 10 m 误差 15.3%、20 m 误差 4.0%，且在 Night2 取得所有蒸馏方法中最佳 20 m 结果。
- **DENSE 零样本鲁棒性**：误差为 1.04/1.44/2.35，位列第二，较 EventDAM 在 20 m/30 m 分别降低 44.6% 与 54.6%，凸显长距离几何迁移的稳健性。
- **直接 VGGT 迁移对照**：未蒸馏时直接将 VGGT 用于事件输入的 10 m 误差为 2.42，经 SFE-VGGT 蒸馏后降至 1.41（Night1），降幅达 41.7%，验证事件域对齐的必要性。
- **消融结论**：DAFD 主要贡献于 20 m/30 m 长距几何；CFRC 显著提升 10 m 近距精度与 $\delta_i$ 阈值指标；事件密度加权与相对置信度加权均为各损失中不可替代的核心组件；24 帧时序窗口综合最优。

## 相关工作脉络
- **Vision Foundation Models & Cross-Modal Distillation**：VGGT、Depth AnyEvent、EventDAM 均依赖同步 RGB 进行跨模态知识迁移；本文定位差异在于彻底移除源模态，仅用事件重建帧驱动教师。
- **Event-Based Monocular Depth Estimation**：早期监督方法（E2Depth、RAMNet、SRFNet）受限于稠密深度标注稀缺；本文与 EventVGGT 同属蒸馏范式，但突破了对真实 RGB 配对的硬性依赖。
- **Source-Free Knowledge Distillation**：现有无源蒸馏多聚焦分类/识别任务；本文将其拓展至密集几何预测，解决事件流重建伪影传播的特定难题。
- **Event-to-Video Reconstruction**：E2VID 作为预训练冻结模块提供教师输入；本文未对其微调，体现了“重建-蒸馏解耦”的轻量化设计思路。
- **Temporal Geometric Consistency**：传统方法依赖教师时序深度一致性进行约束；CFRC 改用学生自身跨帧相对顺序作为监督源，绕过教师深度噪声。

## 局限性与未来方向
- **重建质量上限约束**：教师分支依赖冻结的 E2VID，其输出质量直接决定蒸馏上界，事件空洞或剧烈运动区域仍可能引入不可逆误差。
- **重建模块未联合优化**：当前架构将事件重建与几何蒸馏解耦，重建网络无法感知下游深度任务需求，存在优化目标错位。
- **未来方向**：探索重建网络与蒸馏框架的联合训练，使 surrogate frame 生成过程自适应服务于 VGGT 几何先验的提取；同时可延伸至多相机事件阵列或动态场景 SLAM 应用。

## 研究启发与可借鉴点
- **替代帧教师 + 可靠性加权**：在无源条件下构造高质量替代监督信号，并通过数据驱动权重（密度/置信度）抑制不可靠区域传播，可作为跨模态蒸馏的通用范式。
- **相对关系一致性替代绝对监督**：CFRC 利用学生自身时序深度排序作为约束，无需教师绝对深度，有效规避了重建伪影对几何一致性的破坏。
- **轻量适配器保持基础模型先验**：仅对 VGGT 注入 LoRA 即可实现跨模态迁移，参数量极低且推理时无额外开销，适合边缘部署。
- **合成到真实的零样本验证设计**：在 EventScape 训练、MVSEC/DENSE 零样本测试的评估流程，为事件几何感知提供了可复用的泛化性 benchmark 范式。

## 关键术语表
**Source-Free Distillation**：在无原始源模态数据（如 RGB）情况下，仅利用目标域数据完成跨模态知识迁移的蒸馏范式。
**Surrogate Frame**：由事件流经预训练重建网络生成的伪图像，用作冻结教师模型的输入以提取几何先验。
**DAFD（Density-Aware Feature Distillation）**：依据局部事件激活密度对教师-学生特征对齐进行空间加权，抑制事件稀疏区域的不可靠监督。
**CWDD（Confidence-Weighted Depth Distillation）**：基于师生相对预测置信度动态调节尺度不变深度误差与梯度一致性损失。
**CFRC（Cross-Frame Relational Consistency）**：利用替代帧匹配点对约束学生网络在不同帧间的相对深度顺序一致性，不依赖教师深度值。
**VGGT**：Visual Geometry Grounded Transformer，具备多视图深度、相机位姿与轨迹联合推理能力的大型几何基础模型。
**Event Camera**：异步记录像素级亮度变化的视觉传感器，具有高动态范围、低延迟与抗运动模糊特性。
**LoRA**：Low-Rank Adaptation，通过低秩矩阵增量微调预训练骨干，以极低参数量适配下游任务。

## 可复现要素
- **数据集**：EventScape（训练/域内测试）、MVSEC（零样本夜间/日间序列）、DENSE（零样本测试）；均公开可用。
- **代码/权重**：论文未明确声明开源链接与权重托管地址。
- **关键超参**：E2VID 预训练权重公开；VGGT 主干冻结，LoRA $r=16,\alpha=32$；AdamW 学习率 $1\times10^{-5}$；输入裁剪 $252\times504$；24 帧事件窗口；损失权重 $\lambda_{\mathrm{DAFD}}=1.0,\lambda_{\mathrm{CWDD}}=2.0,\lambda_{\mathrm{CFRC}}=0.2$；训练约 23 小时（单卡 RTX 5090）。

# RELAYVSR: LARGE–SMALL MODEL COLLABORA-TION FOR EFFICIENT REAL-WORLD VIDEO SUPER-RESOLUTION

Xijun Wang<sup>1</sup> Xin Li<sup>1</sup> Zirui Lang<sup>1</sup> Suhang Yao<sup>1</sup> Haoran Li<sup>1</sup> Zhibo Chen<sup>1</sup>

<sup>1</sup>University of Science and Technology of China

## ABSTRACT

Large generative models can recover realistic detail in real-world video superresolution (VSR), but processing an entire video with them is computationally expensive. In this work, we present RelayVSR, a streaming VSR framework built on the Sparse Generative Relay mechanism. A large generative model generates reference latents for sparse keyframes, while a lightweight VSR network uses these references and low-resolution video to super-resolve every frame. The lightweight VSR network, implemented as a Dual-Memory Video Transformer, reuses keyframe information across frames and updates recent video context, supporting first-keyframe conditioning and dual-endpoint conditioning with bounded lookahead. However, errors in shared keyframes can propagate and accumulate across output frames, making keyframe quality alone an insufficient optimization target. We address this collaboration gap with Video-Aware Reference Optimization (VARO), which uses reinforcement learning to update the large generative model with two reward levels: a system-level reward evaluates videos produced by the fixed lightweight VSR network, while a reference-level reward evaluates decoded keyframe quality. VARO improves final video quality over direct joint training, and its dual-level rewards outperform a system-level reward alone. At 1080p on a single NVIDIA A100 80GB, dualendpoint RelayVSR with a 15-frame keyframe interval reaches 29.29 FPS, 13.82 GB peak GPU memory, and 0.327 s first-frame model latency, compared with 7.80 FPS, 24.447 GB, and 2.83 s for FlashVSR-Tiny. The code is available at https://github.com/kopperx/RelayVSR.

## 1 INTRODUCTION

Streaming video super-resolution (VSR) aims to produce high-quality, high-resolution video in real time from degraded low-resolution inputs (Zhuang et al., 2026). To improve output quality, recent diffusion-based VSR methods draw on pretrained video generation models. Their learned priors help synthesize realistic textures and fine details (Xie et al., 2025). However, running these large models is computationally expensive, especially at high output resolutions (Wang et al., 2026).

Recent work has reduced this cost by replacing iterative denoising with a single generative forward pass (Chen et al., 2026; Wang et al., 2026). This avoids repeatedly evaluating a large model for the same video. Further improvements target attention and decoding, two major sources of inference overhead. Local or sparse attention reduces computation over video latents, while lightweight decoders accelerate high-resolution reconstruction (Zhuang et al., 2026; Yan et al., 2026). These advances have substantially reduced both sampling cost and the overhead of each forward pass. However, even single-step inference in these approaches requires a large generative model to process spatiotemporal latents spanning the video sequence (Zhuang et al., 2026; Yan et al., 2026). Further optimizations of the generative backbone offer diminishing returns. We therefore turn to a different question: Do we need a large generative model to process every frame for high-quality video super-resolution?

In this work, we introduce RelayVSR, an efficient streaming VSR framework built on a (i) Sparse Generative Relay mechanism. A large generative model enhances only sparse keyframes, while a lightweight VSR network super-resolves every frame using the LR video and the generated reference latents. The LR video provides frame-specific content and motion cues, while the enhanced keyframes supply fine details to guide super-resolution. For each interval, RelayVSR can use the first keyframe alone to avoid input lookahead or both endpoints for richer reference guidance with bounded lookahead. The keyframe interval determines how often the large generative model updates these references and therefore its amortized cost per output frame. Fig. 1 illustrates this schedule alongside representative VSR results. At 1080p on a single NVIDIA A100 80GB, the complete system in dual-endpoint mode with a 15-frame keyframe interval runs at 29.29 FPS with 13.82 GB peak GPU memory, compared with 7.80 FPS and 24.447 GB for FlashVSR-Tiny.

![](images/569a1f527dd12cf2f0a6f7826475a6eefc00528c010c22504f4a4c8677e9613a.jpg)  
Figure 1: Visual comparison and streaming schedule of RelayVSR. Top: RelayVSR outputs on an AI-generated video and a real-world video; the 29.3 FPS badge reports RelayVSR throughput at 1080p on one NVIDIA A100 80GB in dual-endpoint mode with a 15-frame keyframe interval. Bottom left: the sparse-reference schedule for first-keyframe and dual-endpoint inference, with a fixed playback delay in the latter.

First, we adapt a pretrained video generation model (Wan et al., 2025) to frame-wise target keyframe latents, followed by one-step GAN distillation with a relativistic adversarial objective (Jolicoeur-Martineau, 2018). Second, to exploit the rich generative priors encoded in the keyframe latents, we design a lightweight (ii) Dual-Memory Video Transformer. The network super-resolves each video interval sequentially using keyframe latents from its first frame or both endpoints. Persistent keyframe and rolling video memories cache keys and values from keyframes and recent frames, respectively. Current LR tokens query both memories to combine keyframe details with local temporal context. Together with keyframe super-resolution, this design enables efficient streaming VSR while retaining the perceptual benefits of the large model’s generative priors.

However, sharing keyframe latents across frames also spreads the effects of keyframe errors. Our experiments show that texture and color errors in super-resolved keyframes propagate to nonkeyframes and accumulate during video super-resolution. Adapting the lightweight VSR network to keyframe latents produced by the large model remains insufficient to prevent this propagation. These findings expose an objective mismatch: optimizing keyframe quality alone does not account for how the lightweight network uses keyframe latents to super-resolve the full video. We term this mismatch the Collaboration Gap. Addressing it calls for optimizing keyframe super-resolution with feedback from the video produced by the lightweight network.

To address this gap, we introduce (iii) Video-Aware Reference Optimization (VARO), which trains the large model using reward feedback from the video super-resolved by the lightweight VSR network. During post-training, the system-level reward evaluates the HR videos super-resolved by the fixed lightweight network using the large model’s keyframe latents, focusing on non-keyframe quality, temporal consistency, and continuity across keyframe updates. The large model therefore learns from how its keyframe latents affect the frames that use them. A complementary reference-level reward evaluates the decoded keyframes for content fidelity and realistic detail, maintaining a direct quality objective alongside downstream video feedback. We implement VARO using Diffusion-NFT (Zheng et al., 2026), updating only the large model with the combined reward.

Experiments demonstrate that RelayVSR achieves a favorable balance between super-resolution quality and inference efficiency compared with other diffusion-based VSR methods. With reference keyframes spaced 15 frames apart, its dual-endpoint configuration processes 1080p video at 29.29 FPS on a single NVIDIA A100 80GB, with peak allocated GPU memory of 13.82 GB. Ablations further show that VARO improves final video quality over direct joint training, and that combining reference-level and system-level rewards yields better results than system-level feedback alone.

## 2 RELATED WORK

## 2.1 REAL-WORLD VIDEO SUPER-RESOLUTION

Conventional VSR methods recover high-resolution frames through temporal alignment and feature aggregation, with real-world extensions addressing complex degradations (Chan et al., 2022a; Liang et al., 2022; Chan et al., 2022b; Zhang & Yao, 2024). Diffusion-based methods further exploit image and video generative priors for detail synthesis, with temporal modeling or motion guidance to maintain consistency (Zhou et al., 2024; Yang et al., 2024; Xie et al., 2025; Wang et al., 2025; Li et al., 2025b). Their sampling cost has motivated one-step VSR methods that replace iterative denoising with a single forward pass (Chen et al., 2026; Wang et al., 2026; Sun et al., 2026; Li et al., 2025a). Beyond sampling, FlashVSR (Zhuang et al., 2026) reduces attention and decoding costs with locality-constrained sparse attention and an LR-conditioned lightweight decoder, while SwiftVR (Yan et al., 2026) combines shifted-window attention with a lightweight autoencoder for streaming inference. Stream-DiffVSR (Shiu et al., 2026) combines a distilled denoiser with motionaligned temporal guidance for causal, frame-by-frame processing. PS-SR (Wu et al., 2026) also distributes computation between models, using a large model for an initial denoising step and a lightweight model for subsequent refinements. RelayVSR instead combines sparse keyframe superresolution by a large generative model with lightweight frame-by-frame processing to enable fast, high-quality streaming VSR.

## 2.2 REFERENCE-GUIDED VIDEO SUPER-RESOLUTION

Reference-guided VSR supplements LR videos with fine details from high-quality reference frames. RefVSR and ERVSR obtain these references from an additional wide-angle camera to super-resolve ultra-wide videos (Lee et al., 2022; Kim et al., 2023). Within this multi-camera setting, RefVSR++ (Zou et al., 2025) improves temporal aggregation by separately propagating reference features and fused LR–reference features. References can also be obtained from the LR video itself through image super-resolution. DAM-VSR (Kong et al., 2025) uses such references to guide appearance in video diffusion, while the LR sequence provides motion guidance. SparkVSR (Yu et al., 2026) similarly conditions a large video diffusion model on super-resolved keyframes, combining their sparse latents with LR video latents to enable interactive VSR. In RelayVSR, generated keyframe latents instead guide a lightweight VSR network. The network retains reference information in persistent keyframe memory and incorporates recent temporal context through rolling video memory, supporting frame-by-frame super-resolution.

## 2.3 PREFERENCE OPTIMIZATION FOR DIFFUSION MODELS

Reinforcement learning and preference optimization enable diffusion models to learn from feedback beyond their original denoising objectives. DDPO (Black et al., 2024) formulates denoising as a sequential decision process and applies policy gradient optimization, while Diffusion-DPO (Wallace et al., 2024) learns directly from pairwise preferences. DiffusionNFT (Zheng et al., 2026) incorporates reward feedback into forward-process flow matching, enabling policy optimization without likelihood estimation.

![](images/aea254203d0ef13adce9ec59e52e59dc4a6b3d11466fa6f1176715eb4d3d18bb.jpg)  
Figure 2: RelayVSR framework and training. Top: the large generative model learns frame-wise target keyframe latents in Stage 1 and undergoes one-step GAN distillation in Stage 2. Bottom left: the lightweight VSR network super-resolves every frame by jointly attending to persistent keyframe and rolling video memories. Bottom right: VARO uses reinforcement learning to update the large generative model with a system-level reward on videos super-resolved by the fixed lightweight VSR network and a reference-level reward on decoded keyframes.

For image super-resolution, RFSR (Sun et al., 2024) combines perceptual reward feedback with structural constraints, while GDPO-SR (Yi et al., 2026) adapts group-based preference optimization to one-step generative models. Recent work further incorporates fidelity to the LR input: Lucid-NFT (Fei et al., 2026) combines perceptual and LR-referenced faithfulness rewards through DiffusionNFT, while OARS (Zhao et al., 2026) evaluates fidelity preservation and perceptual gain in the LR-to-SR transition. These methods optimize the quality of the SR images produced by the model being updated. In RelayVSR, the large generative model produces reference latents for sparse keyframes to guide a lightweight VSR network. The designed VARO optimizes the large generative model through reinforcement learning with two reward levels—a system-level reward and a reference-level reward—while keeping the lightweight VSR network fixed.

## 3 METHOD

Overview. RelayVSR assigns sparse keyframe super-resolution to a large generative model and uses the generated reference latents to guide a lightweight VSR network in super-resolving every frame (Fig. 2). We first describe the Sparse Generative Relay (Sec. 3.1) and the Dual-Memory Video Transformer that combines reusable references with recent video context (Sec. 3.2). Finally, Video-Aware Reference Optimization (VARO) uses reinforcement learning to update the large generative model with a system-level reward and a reference-level reward, while keeping the lightweight VSR network fixed (Sec. 3.3).

## 3.1 SPARSE GENERATIVE RELAY

Let $X = \{ x _ { t } \} _ { t = 1 } ^ { T }$ denote the LR input video and $Y = \{ y _ { t } \} _ { t = 1 } ^ { T }$ its HR training target. We select keyframes every $\Delta$ frames starting at $k _ { 1 } = 1$ and denote their ordered indices by $\bar { \boldsymbol { K } } = \{ k _ { j } \} _ { j = 1 } ^ { J } ,$ where $\Delta$ is the keyframe interval. The large generative model produces a reference latent $\hat { z } _ { j }$ for each selected frame and passes it directly to the lightweight VSR network without RGB decoding. The lightweight network uses these references and the LR video to super-resolve every frame, including those at keyframe positions.

Keyframe super-resolution. We adapt the pretrained Wan2.2 video diffusion model (Wan et al., 2025) with a learned LR projection through the two stages in Fig. 2. In Stage 1 (frame-wise latent adaptation), we independently encode each HR keyframe as $z _ { i } ^ { \star } = \mathcal { E } ( y _ { k _ { i } } )$ using the frozen VAE encoder E. We stack these targets into $Z ^ { \star }$ and fine-tune the model with LoRA using a flow-matching objective. During training, a block-causal attention mask limits temporal attention to tokens from the current keyframe and recent keyframes within a fixed window. In Stage 2, we use GAN distillation to obtain a one-step reference generator, combining a latent reconstruction loss with a relativistic adversarial (RSGAN) objective (Jolicoeur-Martineau, 2018). Appendices A.1 and B give the full objectives and training settings. At inference, the generator $G _ { \theta }$ uses key–value (KV) caching for streaming generation: $\mathsf { \bar { ( } } \hat { z } _ { j } , s _ { j } ^ { G } \mathsf { ) } = G _ { \theta } ( x _ { k _ { j } } , \epsilon _ { j } , s _ { j - 1 } ^ { G } )$ , where $\epsilon _ { j }$ is sampling noise and $\overset { \cdot \mathrm { ~  ~ \lambda ~ } } { s _ { j } ^ { G } }$ is the cache state.

Streaming collaboration. Between consecutive keyframes $k _ { j }$ and $k _ { j + 1 }$ , the lightweight VSR network super-resolves frames sequentially using the LR video and the selected reference latents. In first-keyframe mode, the network uses the left reference and causal LR context without future-frame dependency. In dual-endpoint mode, it also uses the right reference, introducing bounded input lookahead. The large generative model runs only at keyframes, with $\Delta$ controlling its invocation frequency, while the lightweight network reuses the generated references to super-resolve every frame.

## 3.2 DUAL-MEMORY VIDEO TRANSFORMER

Our lightweight VSR network, the Dual-Memory Video Transformer $F _ { \phi }$ , super-resolves each incoming LR frame using generated reference latents and information from recent frames. We use a convolutional encoder to extract features from the LR frame and a separate projection to process the reference latents. Both produce spatial tokens with the same feature dimension for the Transformer.

Dual-memory attention. To reuse keyframe information across output frames, we process each reference independently, applying self-attention among its own spatial tokens at each Transformer layer. Because this reference processing is independent of video features, we precompute its layerwise keys and values and store them in a persistent keyframe memory. Super-resolving successive frames also requires recent temporal context to complement the reusable keyframe information. We therefore maintain a rolling video memory that stores the layer-wise keys and values of recently processed video frames and is updated after each frame.

At each Transformer layer, the current frame jointly attends to the selected references, recent video history, and its own features. Let $Q _ { t }$ denote the current-frame queries, $( K _ { t } ^ { r } , V _ { t } ^ { r } )$ the key and value projections of the selected references, and $( K _ { t } ^ { v } , V _ { t } ^ { v } )$ those of the recent video history and the current frame. The attention output is

$$
a _ { t } = \mathrm { A t t n } ( Q _ { t } , [ K _ { t } ^ { r } ; K _ { t } ^ { v } ] , [ V _ { t } ^ { r } ; V _ { t } ^ { v } ] ) ,\tag{1}
$$

where $[ ; ]$ concatenates tokens. We restrict attention to local spatial neighborhoods to limit computation. Since sparse keyframes can be farther away in time than recent video frames, we use a larger neighborhood when reading references to accommodate potentially greater spatial displacement.

Super-resolution and training. The output head uses a projection and PixelShuffle to convert the final features into an RGB residual at the target resolution. Adding this residual to the bicubicupsampled LR frame gives the $4 \times$ super-resolved output. We train the network from scratch with reference latents encoded from ground-truth HR keyframes, using $L _ { 1 }$ and LPIPS losses (Zhang et al., 2018) on all output frames. During training, we randomly omit the right reference so that the same network supports first-keyframe and dual-endpoint conditioning. Appendix A.2 details the attention implementation, and Appendix B.2 lists the supervised training settings.

## 3.3 VIDEO-AWARE REFERENCE OPTIMIZATION

The lightweight VSR network is trained with reference latents encoded from ground-truth HR keyframes, whereas inference uses references generated from LR keyframes by the large generative model. Texture and color errors in these generated references can propagate and accumulate as they are reused across frames. Keyframe-level objectives do not capture these downstream effects. VARO addresses this Collaboration Gap through reinforcement learning with system-level and reference-level rewards, optimizing the large generative model for both reference and video quality while keeping the lightweight VSR network fixed.

Dual-level rewards. For each LR–HR training clip $( X , Y )$ , the one-step generator samples a causal sequence of reference latents $\hat { Z }$ . Using the streaming procedure in Sec. 3.1, the lightweight VSR network with fixed parameters $\phi ^ { \star }$ produces $\hat { Y } = F _ { \phi ^ { \star } } ( X , \hat { Z } )$ . The system-level reward $R _ { \mathrm { s y s } }$ scores the entire super-resolved video, including keyframe outputs. It combines RGB $L _ { 1 }$ fidelity across all frames, DOVER++ technical video quality (Wu et al., 2023), MUSIQ (Ke et al., 2021) on temporally spaced frames, and optical-flow warping error over consecutive output frames. The reference-level reward $R _ { \mathrm { r e f } }$ evaluates the same generated latents after decoding by the frozen VAE decoder D. It combines RGB $L _ { 1 }$ against each HR keyframe $y _ { k _ { i } }$ , MUSIQ on every decoded reference $\mathcal { D } ( \hat { z } _ { j } )$ , and optical-flow warping error between successive decoded references.

Each reference rollout receives the combined reward

$$
R _ { \mathrm { t o t a l } } = \lambda _ { \mathrm { s y s } } \widetilde { R } _ { \mathrm { s y s } } ( \hat { Y } , Y ) + \lambda _ { \mathrm { r e f } } \widetilde { R } _ { \mathrm { r e f } } ( \hat { Z } , Y _ { \mathcal { K } } ) ,\tag{2}
$$

where $Y _ { K }$ denotes the HR keyframes, the tildes denote reward-scale normalization, and the weights balance video and reference quality.

Generator optimization. We aim to maximize $\mathbb { E } _ { ( X , Y ) , \hat { Z } \sim p _ { \theta } ( \cdot | X \kappa ) } [ R _ { \mathrm { t o t a l } } ]$ using Diffusion-NFT (Zheng et al., 2026). For a sampled reference sequence $u = { \ddot { Z } } ,$ , its group-normalized reward is mapped $\mathbf { t o } r \in [ 0 , 1 ]$ . At the fixed noise endpoint $\tau = 1$ , fresh noise $\epsilon \sim \mathcal { N } ( 0 , I )$ gives the velocity target $v = \epsilon - u$ . We minimize

$$
\mathcal { L } _ { \mathrm { V A R O } } = \mathbb { E } \big [ r \| v _ { \theta } ^ { + } - v \| _ { 2 } ^ { 2 } + ( 1 - r ) \| v _ { \theta } ^ { - } - v \| _ { 2 } ^ { 2 } \big ] ,\tag{3}
$$

where $v _ { \theta } ^ { \pm } = v _ { \mathrm { s a m p } } \pm \beta ( v _ { \theta } - v _ { \mathrm { s a m p } } )$ , with fixed sampling-policy prediction $v _ { \mathrm { s a m p } } ,$ trainable prediction $v _ { \theta } .$ , and mixing coefficient β. Appendix A.3 gives the reward formulas, and Appendix B.3 lists the training settings; inference retains one-step reference generation and streaming VSR.

## 4 EXPERIMENTS

## 4.1 EXPERIMENTAL SETUP

Implementation Details. RelayVSR uses Wan2.2-TI2V-5B as its large generative model and a 31.9M-parameter lightweight VSR network; training draws on approximately 0.5M videos and 1M images. We adapt the large model with rank-512 LoRA for 75K flow-matching steps, followed by 5K steps of one-step GAN distillation. Both stages use eight-frame video clips sampled with random temporal strides and single images, with 704 × 1280 HR crops and batch sizes of 32 and 96, respectively. We use AdamW with learning rates of 2e-5 for flow matching and 1e-5 for GAN distillation. The lightweight VSR network is trained from scratch for 60K steps on 16-frame clips with 1024×1536 HR crops and a batch size of 64, using AdamW with a learning rate of 1e-4. VARO then updates the large model with DiffusionNFT while keeping the lightweight VSR network fixed. Additional training details are provided in Appendix B.

Evaluation Protocol. We assess 4× VSR on three synthetic benchmarks (REDS30, UDM10, and YouHQ40) and the real-world VideoLQ benchmark. For the synthetic sets, LR inputs are generated from HR videos with the same RealBasicVSR degradation pipeline (Chan et al., 2022b) used for training. For sustained streaming, we use LongVSR60, which contains 30 real-world and 30 AI-generated single-shot videos of approximately 1,000 frames each. The main comparison includes RealViformer (Zhang & Yao, 2024), STAR (Xie et al., 2025), DOVE (Chen et al., 2026), SeedVR2 (Wang et al., 2026), SparkVSR (Yu et al., 2026), SwiftVR (Yan et al., 2026), and FlashVSR (Zhuang et al., 2026). Throughout the paper, FlashVSR refers to its Tiny v1.1 variant. For paired synthetic videos, we measure reconstruction fidelity with PSNR and SSIM and perceptual similarity with LPIPS. We assess no-reference quality with NIQE, MUSIQ, CLIP-IQA, and DOVER on both synthetic and real-world videos. A blind pairwise user study is described in Appendix C.4.

## 4.2 COMPARISON WITH EXISTING METHODS

Quantitative Comparisons. Table 1 shows that RelayVSR maintains strong full-video quality with dual-endpoint conditioning and large-model calls spaced 15 frames apart. Across the three paired benchmarks, it surpasses FlashVSR in PSNR, SSIM, and LPIPS. It achieves the best LPIPS (0.2569) and DOVER (0.7313) on YouHQ40, the best DOVER on REDS30 (0.4010), and the best

SSIM (0.7545) and MUSIQ (65.32) on UDM10. On real-world VideoLQ, RelayVSR leads in MUSIQ (51.26) and DOVER (0.5488), while FlashVSR performs better in NIQE and CLIP-IQA. The evaluation covers complete output videos rather than keyframes alone.

Qualitative Comparisons. Fig. 3 compares RelayVSR with five generative VSR methods on two real-world videos and one AIGC video. RelayVSR delineates the horse’s face and bridle, retains recognizable facial features and license-plate characters in the second example, and resolves bicycle components and foliage in the AIGC cycling scene. To examine whether detail persists between keyframes, Fig. 4 follows a building crop between keyframes 00 and 15 through five intermediate frames. Window outlines, roof edges, and masonry remain clear at these non-keyframe positions, illustrating that sparse generative references support visual quality throughout the interval.

## 4.3 SYSTEM EFFICIENCY ANALYSIS

Inference efficiency. Table 2 shows the efficiency gains from invoking the large generative model only at sparse keyframes. On 201- frame clips at 1080p using one NVIDIA A100- 80G, RelayVSR in dual-endpoint mode reaches 29.29 FPS with 13.82 GB peak allocated memory. This is 3.76× the throughput of FlashVSR (7.80 FPS, 24.45 GB) while using 43.5% less memory; SwiftVR, the fastest baseline in the table, reaches 23.48 FPS. Switching to firstkeyframe mode raises throughput to 35.89 FPS and reduces peak memory to 13.72 GB; the first output requires 0.183 s of model computation, compared with 0.327 s in dual-endpoint mode. In a real streaming system, dual-endpoint mode must also wait for the future keyframe to arrive— output frames in that interval. output frames in that interval.

Table 2: Inference efficiency at 1080 × 1920 output resolution on one NVIDIA A100-80G using 201-frame clips. Timing excludes input wait and operations outside model inference.
<table><tr><td>Method</td><td>FPS ↑</td><td>Peak Mem. (GB)↓</td><td>First output (s)</td></tr><tr><td rowspan="3">DOVE SeedVR2-3B Stream-DiffVSR SwiftVR</td><td>0.56</td><td>41.08</td><td>356.801</td></tr><tr><td>0.79</td><td>73.89</td><td>255.199</td></tr><tr><td>0.85 23.48</td><td>25.92 31.36</td><td>0.705 1.107</td></tr><tr><td rowspan="3">FlashVSR Ours (first-keyframe)</td><td>7.80</td><td>24.45</td><td>2.830</td></tr><tr><td>35.89</td><td>13.72</td><td>0.183</td></tr><tr><td>29.29</td><td>13.82</td><td>0.327</td></tr></table>

up to 0.500 s at a 30-FPS input rate—before it can

Table 1: Quantitative comparison on synthetic and real-world VSR benchmarks. The best and second-best results are marked in red and blue, respectively.
<table><tr><td>Dataset</td><td>Metric</td><td>RealViformer</td><td>STAR</td><td>DOVE</td><td>SeedVR2</td><td>SparkVSR</td><td>SwiftVR</td><td>FlashVSR</td><td>Ours</td></tr><tr><td rowspan="6">REDS30</td><td>PSNR↑</td><td>23.32</td><td>22.14</td><td>23.28</td><td>22.17</td><td>23.29</td><td>21.33</td><td>21.41</td><td>22.77</td></tr><tr><td>SSIM↑</td><td>0.5967</td><td>0.5432</td><td>0.6103</td><td>0.5860</td><td>0.6050</td><td>0.5234</td><td>0.5399</td><td>0.5874</td></tr><tr><td>LPIPS↓</td><td>0.3043</td><td>0.4967</td><td>0.3773</td><td>0.3158</td><td>0.3909</td><td>0.3564</td><td>0.3311</td><td>0.3051</td></tr><tr><td>NIQE↓</td><td>3.0804</td><td>5.2237</td><td>4.2282</td><td>3.5157</td><td>4.4496</td><td>3.4153</td><td>2.9504</td><td>3.5398</td></tr><tr><td>MUSIQ↑</td><td>59.12</td><td>37.95</td><td>50.44</td><td>57.65</td><td>47.58</td><td>63.77</td><td>56.01</td><td>54.30</td></tr><tr><td>CLIP-IQA ↑</td><td>0.3236</td><td>0.2132</td><td>0.2904</td><td>0.3041</td><td>0.2755</td><td>0.3914</td><td>0.3160</td><td>0.3161</td></tr><tr><td rowspan="6">UDM10</td><td>DOVER↑</td><td>0.3293</td><td>0.2068</td><td>0.3337</td><td>0.3679</td><td>0.3200</td><td>0.3893</td><td>0.3477</td><td>0.4010</td></tr><tr><td>PSNR ↑</td><td>26.65</td><td>25.49</td><td>26.39</td><td>25.99</td><td>26.31</td><td>25.41</td><td>24.41</td><td>25.94</td></tr><tr><td>SSIM↑</td><td>0.7483</td><td>0.7237</td><td>0.7504</td><td>0.7313</td><td>0.7430</td><td>0.7213</td><td>0.7008</td><td>0.7545</td></tr><tr><td>LPIPS↓</td><td>0.2670</td><td>0.3677</td><td>0.2341</td><td>0.2150</td><td>0.2554</td><td>0.2640</td><td>0.2528</td><td>0.2171</td></tr><tr><td>NIQE↓ MUSIQ ↑</td><td>4.4199 56.98</td><td>7.1730</td><td>4.7713</td><td>4.6516</td><td>4.7099</td><td>4.2548</td><td>4.0032</td><td>4.3205</td></tr><tr><td>CLIP-IQA ↑</td><td>0.3705</td><td>31.50 0.2225</td><td>60.30 0.4717</td><td>58.49 0.4000</td><td>58.28 0.4687</td><td>64.55 0.4986</td><td>64.24 0.4987</td><td>65.32 0.4852</td></tr><tr><td rowspan="6">YouHQ40</td><td>DOVER↑</td><td>0.4646</td><td>0.2455</td><td>0.4996</td><td>0.5013</td><td>0.4823</td><td>0.4744</td><td>0.5507</td><td>0.5448</td></tr><tr><td>PSNR ↑</td><td>24.17</td><td>23.46</td><td>24.29</td><td>23.66</td><td></td><td>23.08</td><td>22.68</td><td></td></tr><tr><td>SSIM↑</td><td>0.6414</td><td>0.6429</td><td>0.6662</td><td>0.6555</td><td>24.21 0.6605</td><td>0.6151</td><td>0.6026</td><td>23.19 0.6340</td></tr><tr><td>LPIPS↓</td><td>0.3408</td><td>0.4545</td><td>0.2920</td><td>0.2726</td><td>0.3152</td><td>0.2912</td><td>0.2742</td><td>0.2569</td></tr><tr><td>NIQE↓</td><td>3.6506</td><td>6.9886</td><td>4.3244</td><td>4.1732</td><td>4.5953</td><td>3.2815</td><td>3.1768</td><td>3.5769</td></tr><tr><td>MUSIQ ↑</td><td>59.63</td><td>32.91</td><td>60.63</td><td>57.69</td><td>56.67</td><td>62.01</td><td>65.79</td><td>63.21</td></tr><tr><td rowspan="4"></td><td>CLIP-IQA ↑</td><td>0.3996</td><td>0.2651</td><td>0.4500</td><td>0.3970</td><td>0.4051</td><td>0.5128</td><td>0.5333</td><td>0.5182</td></tr><tr><td>DOVER↑</td><td>0.6149</td><td>0.4308</td><td>0.6649</td><td>0.6709</td><td>0.6287</td><td>0.6877</td><td>0.6916</td><td>0.7313</td></tr><tr><td>NIQE↓</td><td>4.3616</td><td>5.6624</td><td>5.0218</td><td>4.7326</td><td>5.0373</td><td>4.4435</td><td>3.9488</td><td></td></tr><tr><td>MUSIQ ↑</td><td>49.22</td><td>35.29</td><td>44.91</td><td>41.68</td><td>42.48</td><td>50.46</td><td>50.73</td><td>4.4729 51.26</td></tr><tr><td rowspan="3">VideoLQ</td><td>CLIP-IQA ↑</td><td>0.3263</td><td>0.2454</td><td>0.2942</td><td>0.2428</td><td>0.2890</td><td>0.3583</td><td>0.3623</td><td>0.3458</td></tr><tr><td>DOVER↑</td><td>0.4621</td><td>0.4135</td><td>0.4972</td><td>0.4455</td><td>0.4797</td><td>0.4957</td><td>0.5339</td><td>0.5488</td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr></table>

![](images/101a3668a70b5bd02f8dcbc407445e989b9a0218a196135d47890954f3dd06e3.jpg)

Figure 3: Qualitative VSR comparisons. The top two examples are real-world videos and the bottom example is an AIGC video. ∗ marks the keyframe sample.  
![](images/872b3bdcb0835bbfc9070b039fedfca6762843e23244e00e1739ccec0f7bb5f1.jpg)  
Figure 4: Keyframe and non-keyframe comparison on a real-world video. The ∗ symbol marks keyframes; the other columns are non-keyframes.

Quality–efficiency trade-off. Fig. 5 shows diminishing returns as the keyframe interval increases. In dual-endpoint mode, increasing ∆ from 5 to 15 raises 1080p throughput from 18.10 to 29.29 FPS while UDM10 LPIPS rises from 0.2115 to 0.2171 and DOVER falls from 0.5530 to 0.5448. Increasing it to 25 adds only 4.84 FPS, but LPIPS reaches 0.2370 and DOVER drops to 0.5140. First-keyframe mode is faster at each interval with higher LPIPS and lower DOVER; at $\Delta \ = \ 1 5 .$ , it reaches 35.89 FPS, 0.2280 LPIPS, and 0.5290 DOVER. These results support dual-endpoint $\Delta \ = \ 1 5$ for the main comparisons. Appendix C.1 reports timings across resolutions.

![](images/f665715de1f7b3b8284cbf6a5d121ba34a2264eda6005508394f215d51c7ff4e.jpg)  
Figure 5: Quality–efficiency trade-off. Left to right: ∆ = 5, 10, 15, 20, 25.

## 4.4 ABLATION STUDY

The efficiency gains above rely on each sparse reference guiding many lightweight VSR updates. We therefore examine how reference and video memories contribute to video quality, then test whether VARO makes the shared references more useful to the final video (Tables 3 and 4).

Reference and video memory. The two memory streams play complementary roles in sparse-reference VSR. Table 3 removes either reference memory or rolling video memory from the lightweight VSR network. All variants use the same training data, budget, and inference schedule; variants with references share the same pre-VARO one-step generator. Adding references to the video-memory-only variant raises MUSIQ from 56.80 to 63.72 and

Table 3: Memory ablation on UDM10 before VARO $\begin{array} { r l r } { ( \Delta } & { { } = } & { 1 5 } \end{array}$ , dual-endpoint). ✓/✗ indicates enabled/disabled memory; MS is motion smoothness.
<table><tr><td></td><td>Ref. memory Video memory MUSIQ ↑ DOVER ↑</td><td></td><td></td><td>MS↑</td></tr><tr><td>x</td><td>√</td><td>56.80</td><td>0.4760</td><td>0.9911</td></tr><tr><td>√</td><td>x</td><td>63.65</td><td>0.5110</td><td>0.9769</td></tr><tr><td>√</td><td>√</td><td>63.72</td><td>0.5290</td><td>0.9830</td></tr></table>

DOVER from 0.4760 to 0.5290, showing the contribution of generative keyframe information to output quality. Adding video memory to the reference-only variant further improves DOVER from 0.5110 to 0.5290 and MS from 0.9769 to 0.9830, while retaining similar MUSIQ (63.65 vs. 63.72). Video memory alone has the highest MS (0.9911), while dual memory yields the best MUSIQ and DOVER and improves MS over reference memory alone.

## Video-aware reference optimization.

VARO improves references according to the video they help produce. Table 4 compares reward variants initialized from the same one-step GAN generator, with the lightweight VSR network fixed. Full VARO raises all-frame MUSIQ from 63.72 to 65.32, DOVER from 0.5290 to 0.5448, and MS from 0.9830 to 0.9900 over the model without post-training. The two reward levels favor different aspects: reference-level feedback alone yields higher keyframe MUSIQ than system-level feedback (64.95 vs. 64.42), whereas system-level feedback gives

Table 4: VARO ablation on UDM10 $( \Delta = 1 5 ,$ , dualendpoint). MS is motion smoothness.
<table><tr><td rowspan="2">Variant</td><td colspan="2">MUSIQ↑</td><td rowspan="2">DOVER↑</td><td rowspan="2">MS↑</td></tr><tr><td>All</td><td>Key</td></tr><tr><td>w/o post-training</td><td>63.72</td><td>64.05</td><td>0.5290</td><td>0.9830</td></tr><tr><td>w/ reference-level reward only</td><td>64.20</td><td>64.95</td><td>0.5355</td><td>0.9838</td></tr><tr><td>w/ system-level reward only</td><td>64.86</td><td>64.42</td><td>0.5400</td><td>0.9851</td></tr><tr><td>Ours</td><td>65.32</td><td>65.18</td><td>0.5448</td><td>0.9900</td></tr></table>

stronger all-frame MUSIQ, DOVER, and MS. Combining them improves all four reported scores, including keyframe MUSIQ to 65.18. Thus, evaluating both the generated references and the final video provides a more effective training signal for large–small model collaboration.

## 5 CONCLUSION

RelayVSR makes generative real-world video super-resolution efficient by sharing sparse largemodel outputs across frames. A large generative model supplies keyframe reference latents, and a lightweight VSR network uses dual memory to carry their detail through the video while tracking recent temporal context. Video-Aware Reference Optimization (VARO) trains the large generative model using feedback on both decoded keyframes and the resulting video, making the references more useful to the lightweight VSR network. Experiments show that this collaboration delivers strong full-video quality while amortizing large-model computation across frames. RelayVSR leads the compared methods in LPIPS and DOVER on YouHQ40 and in MUSIQ and DOVER on realworld VideoLQ. At 1080p with $\Delta = 1 5$ , dual-endpoint mode reaches 29.29 model-computation FPS with 13.82 GB peak allocated GPU memory; first-keyframe mode reaches 35.89 FPS without future-frame input. These results demonstrate how reusable references and downstream optimization extend the benefits of a large generative model across an entire video. Adapting reference placement to scene dynamics is a promising direction for further improving the quality–latency trade-off.

## REFERENCES

Joshua Ainslie, James Lee-Thorp, Michiel De Jong, Yury Zemlyanskiy, Federico Lebron, and Sumit´ Sanghai. Gqa: Training generalized multi-query transformer models from multi-head checkpoints. In Proceedings ofthe 2023 conference on empirical methods in natural language processing, pp. 4895–4901, 2023.

Kevin Black, Michael Janner, Yilun Du, Ilya Kostrikov, and Sergey Levine. Training diffusion models with reinforcement learning. In International Conference on Learning Representations, volume 2024, pp. 4965–4987, 2024.

Kelvin CK Chan, Shangchen Zhou, Xiangyu Xu, and Chen Change Loy. Basicvsr++: Improving video super-resolution with enhanced propagation and alignment. In 2022 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pp. 5962–5971. IEEE, 2022a.

Kelvin CK Chan, Shangchen Zhou, Xiangyu Xu, and Chen Change Loy. Investigating tradeoffs in real-world video super-resolution. In 2022 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pp. 5952–5961. IEEE, 2022b.

Zheng Chen, Zichen Zou, Kewei Zhang, Xiongfei Su, Xin Yuan, Yong Guo, and Yulun Zhang. Dove: Efficient one-step diffusion model for real-world video super-resolution. Advances in Neural Information Processing Systems, 38:85218–85237, 2026.

Song Fei, Tian Ye, Sixiang Chen, Zhaohu Xing, Jianyu Lai, and Lei Zhu. Lucidnft: Lr-anchored multi-reward preference optimization for flow-based real-world super-resolution. arXiv preprint arXiv:2603.05947, 2026.

Ziqi Huang, Yinan He, Jiashuo Yu, Fan Zhang, Chenyang Si, Yuming Jiang, Yuanhan Zhang, Tianxing Wu, Qingyang Jin, Nattapol Chanpaisit, Yaohui Wang, Xinyuan Chen, Limin Wang, Dahua Lin, Yu Qiao, and Ziwei Liu. Vbench: Comprehensive benchmark suite for video generative models. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pp. 21807–21818, June 2024.

Alexia Jolicoeur-Martineau. The relativistic discriminator: a key element missing from standard gan. arXiv preprint arXiv:1807.00734, 2018.

Junjie Ke, Qifei Wang, Yilin Wang, Peyman Milanfar, and Feng Yang. Musiq: Multi-scale image quality transformer. In 2021 IEEE/CVF International Conference on Computer Vision (ICCV), pp. 5128–5137. IEEE, 2021.

Youngrae Kim, Jinsu Lim, Hoonhee Cho, Minji Lee, Dongman Lee, Kuk-Jin Yoon, and Ho-Jin Choi. Efficient reference-based video super-resolution (ervsr): Single reference image is all you need. In 2023 IEEE/CVF Winter Conference on Applications of Computer Vision (WACV), pp. 1828–1837. IEEE, 2023.

Zhe Kong, Le Li, Yong Zhang, Feng Gao, Shaoshu Yang, Tao Wang, Kaihao Zhang, Zhuoliang Kang, Xiaoming Wei, Guanying Chen, et al. Dam-vsr: Disentanglement of appearance and motion for video super-resolution. In Proceedings of the Special Interest Group on Computer Graphics and Interactive Techniques Conference Conference Papers, pp. 1–11, 2025.

Junyong Lee, Myeonghee Lee, Sunghyun Cho, and Seungyong Lee. Reference-based video superresolution using multi-camera video triplets. In 2022 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pp. 17803–17812. IEEE, 2022.

Hanting Li, Huaao Tang, Jianhong Han, Tianxiong Zhou, Jiulong Cui, Haizhen Xie, Yan Chen, and Jie Hu. Os-diffvsr: Towards one-step latent diffusion model for high-detailed real-world video super-resolution. arXiv preprint arXiv:2509.16507, 2025a.

Xiaohui Li, Yihao Liu, Shuo Cao, Ziyan Chen, Shaobin Zhuang, Xiangyu Chen, Yinan He, Yi Wang, and Yu Qiao. Diffvsr: Revealing an effective recipe for taming robust video super-resolution against complex degradations. In 2025 IEEE/CVF International Conference on Computer Vision (ICCV), pp. 15319–15328. IEEE, 2025b.

Jingyun Liang, Yuchen Fan, Xiaoyu Xiang, Rakesh Ranjan, Eddy Ilg, Simon Green, Jiezhang Cao, Kai Zhang, Radu Timofte, and Luc V Gool. Recurrent video restoration transformer with guided deformable attention. Advances in neural information processing systems, 35:378–393, 2022.

Hau-Shiang Shiu, Chin-Yang Lin, Zhixiang Wang, Chi-Wei Hsiao, Po-Fan Yu, Yu-Chih Chen, and Yu-Lun Liu. Stream-diffvsr: Low-latency streamable video super-resolution via auto-regressive diffusion. In European Conference on Computer Vision, pp. 315–338. Springer, 2026.

Xiaopeng Sun, Qinwei Lin, Yu Gao, Yujie Zhong, Chengjian Feng, Dengjie Li, Zheng Zhao, Jie Hu, and Lin Ma. Rfsr: Improving isr diffusion models via reward feedback learning. arXiv preprint arXiv:2412.03268, 2024.

Yujing Sun, Lingchen Sun, Shuaizheng Liu, Rongyuan Wu, Zhengqiang Zhang, and Lei Zhang. One-step diffusion for detail-rich and temporally consistent video super-resolution. Advances in Neural Information Processing Systems, 38:172821–172841, 2026.

Bram Wallace, Meihua Dang, Rafael Rafailov, Linqi Zhou, Aaron Lou, Senthil Purushwalkam, Stefano Ermon, Caiming Xiong, Shafiq Joty, and Nikhil Naik. Diffusion model alignment using direct preference optimization. In 2024 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pp. 8228–8238. IEEE, 2024.

Team Wan, Ang Wang, Baole Ai, Bin Wen, Chaojie Mao, Chen-Wei Xie, Di Chen, Feiwu Yu, Haiming Zhao, Jianxiao Yang, et al. Wan: Open and advanced large-scale video generative models. arXiv preprint arXiv:2503.20314, 2025.

Jianyi Wang, Zhijie Lin, Meng Wei, Yang Zhao, Ceyuan Yang, Chen Change Loy, and Lu Jiang. Seedvr: Seeding infinity in diffusion transformer towards generic video restoration. In 2025 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pp. 2161–2172. IEEE, 2025.

Jianyi Wang, Shanchuan Lin, Zhijie Lin, Yuxi Ren, Meng Wei, Zongsheng Yue, Shangchen Zhou, Hao Chen, Yang Zhao, Ceyuan Yang, et al. Seedvr2: One-step video restoration via diffusion adversarial post-training. In International Conference on Learning Representations, volume 2026, pp. 40987–41009, 2026.

Aiqiu Wu, Zhaofan Qiu, Ting Yao, and Tao Mei. Ps-sr: Pseudo-single-step video super-resolution via speculative diffusion. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 38218–38227, 2026.

Haoning Wu, Erli Zhang, Liang Liao, Chaofeng Chen, Jingwen Hou, Annan Wang, Wenxiu Sun, Qiong Yan, and Weisi Lin. Exploring video quality assessment on user generated contents from aesthetic and technical perspectives. In 2023 IEEE/CVF International Conference on Computer Vision (ICCV), pp. 20087–20097. IEEE, 2023.

Rui Xie, Yinhong Liu, Penghao Zhou, Chen Zhao, Jun Zhou, Kai Zhang, Zhenyu Zhang, Jian Yang, Zhenheng Yang, and Ying Tai. Star: Spatial-temporal augmentation with text-to-video models for real-world video super-resolution. In 2025 IEEE/CVF International Conference on Computer Vision (ICCV), pp. 17108–17118. IEEE, 2025.

Jiaqi Yan, Xiangyu Chen, Xinlin Zhong, Haibin Huang, Chi Zhang, Jie Liu, Jiantao Zhou, and Xuelong Li. Swiftvr: Real-time one-step generative video restoration. arXiv preprint arXiv:2606.09516, 2026.

Xi Yang, Chenhang He, Jianqi Ma, and Lei Zhang. Motion-guided latent diffusion for temporally consistent real-world video super-resolution. In European conference on computer vision, pp. 224–242. Springer, 2024.

Qiaosi Yi, Shuai Li, Rongyuan Wu, Lingchen Sun, Zhengqiang Zhang, and Lei Zhang. Gdpo-sr: Group direct preference optimization for one-step generative image super-resolution, 2026. URL https://arxiv.org/abs/2603.16769.

Jiongze Yu, Xiangbo Gao, Pooja Verlani, Akshay Gadde, Yilin Wang, Balu Adsumilli, and Zhengzhong Tu. Sparkvsr: Interactive video super-resolution via sparse keyframe propagation. In European Conference on Computer Vision, pp. 397–415. Springer, 2026.

Richard Zhang, Phillip Isola, Alexei A Efros, Eli Shechtman, and Oliver Wang. The unreasonable effectiveness of deep features as a perceptual metric. In 2018 IEEE/CVF conference on computer vision and pattern recognition, pp. 586–595. IEEE, 2018.

Yuehan Zhang and Angela Yao. Realviformer: Investigating attention for real-world video superresolution. In European conference on computer vision, pp. 412–428. Springer, 2024.

Shijie Zhao, Xuanyu Zhang, Bin Chen, Weiqi Li, Qunliang Xing, Kexin Zhang, Yan Wang, Junlin Li, Li Zhang, Jian Zhang, et al. Oars: Process-aware online alignment for generative real-world image super-resolution. arXiv preprint arXiv:2603.12811, 2026.

Kaiwen Zheng, Huayu Chen, Haotian Ye, Haoxiang Wang, Qinsheng Zhang, Kai Jiang, Hang Su, Stefano Ermon, Jun Zhu, and Ming-Yu Liu. Diffusionnft: Online diffusion reinforcement with forward process. In International Conference on Learning Representations, volume 2026, pp. 134129–134150, 2026.

Shangchen Zhou, Peiqing Yang, Jianyi Wang, Yihang Luo, and Chen Change Loy. Upscale-a-video: Temporal-consistent diffusion model for real-world video super-resolution. In 2024 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pp. 2535–2545. IEEE, 2024.

Junhao Zhuang, Shi Guo, Xin Cai, Xiaohui Li, Yihao Liu, Chun Yuan, and Tianfan Xue. Flashvsr: Towards real-time diffusion-based streaming video super resolution. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 43482–43493, 2026.

Han Zou, Masanori Suganuma, and Takayuki Okatani. Refvsr++: Exploiting reference inputs for reference-based video super-resolution. In 2025 IEEE/CVF Winter Conference on Applications of Computer Vision (WACV), pp. 2756–2765. IEEE, 2025.

## Appendix

## A ADDITIONAL METHOD DETAILS

## A.1 REFERENCE GENERATION

Flow-matching objective. For the independently encoded targets $Z ^ { \star }$ defined in Sec. 3.1, normalized diffusion time $\dot { \boldsymbol { \tau } } \in [ 0 , 1 ]$ , and Gaussian noise $\dot { \epsilon } \sim \mathcal { N } ( 0 , I )$ , we form

$$
Z _ { \tau } = ( 1 - \tau ) Z ^ { \star } + \tau \epsilon , \qquad v ^ { \star } = \epsilon - Z ^ { \star } ,\tag{4}
$$

then optimize the velocity prediction with

$$
\mathcal { L } _ { \mathrm { F M } } = \mathbb { E } _ { Z ^ { \star } , c , \tau , \epsilon } \big [ \mathrm { M S E } ( v _ { \theta } ( Z _ { \tau } , \tau , c ) , v ^ { \star } ) \big ] ,\tag{5}
$$

where c contains projected LR keyframes and their available temporal context. The block-causal attention mask restricts each keyframe to a window of H keyframes, including itself.

One-step adversarial objective. We use a relativistic adversarial objective (Jolicoeur-Martineau, 2018) in addition to latent reconstruction. Let $d _ { \mathrm { r e a l } }$ and $d _ { \mathrm { g e n } }$ be the discriminator logits and σ the sigmoid function. The two adversarial losses are

$$
\begin{array} { r } { \mathcal { L } _ { \mathrm { R S G A N } } ^ { G } = - \mathbb { E } \log \sigma ( d _ { \mathrm { g e n } } - d _ { \mathrm { r e a l } } ) , } \\ { \mathcal { L } _ { \mathrm { R S G A N } } ^ { D } = - \mathbb { E } \log \sigma ( d _ { \mathrm { r e a l } } - d _ { \mathrm { g e n } } ) . } \end{array}\tag{6}
$$

The generator and discriminator objectives are

$$
\begin{array} { r l } & { \mathcal { L } _ { G } ^ { \mathrm { 1 s t e p } } = \mathrm { M S E } ( \hat { Z } , Z ^ { \star } ) + \lambda _ { \mathrm { a d v } } \mathcal { L } _ { \mathrm { R S G A N } } ^ { G } , } \\ & { \quad \mathcal { L } _ { D } = \mathcal { L } _ { \mathrm { R S G A N } } ^ { D } + \gamma _ { \mathrm { R 1 } } \mathcal { L } _ { \mathrm { f e a t u r e - R 1 } } , } \end{array}\tag{7}
$$

The discriminator scores intermediate features of a frozen Wan feature extractor with a trainable pooling head; feature-space R1 regularizes gradients with respect to those input features.

During streaming inference, the generator attends to a local spatial neighborhood around each keyframe latent token and uses a rolling KV cache for recent keyframes; rotary position encoding (RoPE) is applied to cached keys when they are read.

## A.2 DUAL-MEMORY ATTENTION IMPLEMENTATION

Token resolution. For a spatially padded LR input of size $h \times w .$ , the LR encoder reduces the grid to $h / 4 \times w / 4$ . The frozen VAE encodes a 4× HR keyframe with spatial stride 16, giving a reference latent on the same grid. The LR and reference streams use separate input RMS normalization and type embeddings before entering the shared Transformer width.

Reference feature reuse. Reference tokens use spatial RoPE and a common reference type embedding, with no left/right endpoint role encoding. Their layer-wise K/V therefore depend on the reference content rather than its role in an interval; the cached right endpoint can serve as the next interval’s left endpoint without being encoded again.

Joint attention rules. The eligible reference, history, and current-frame K/V in Eq. (1) participate in one grouped-query attention softmax (Ainslie et al., 2023). Reference and video streams have separate K/V projections and RMS-normalized K/V features, while sharing the attention normalization, query and output projections, and SwiGLU feed-forward network. Recent video frames are close in time, so a small neighborhood focuses on nearby motion cues at low attention cost. Sparse references may be $\Delta$ frames away; a wider neighborhood lets each query search for corresponding details across larger displacements. Both neighborhoods are centered on the current query, keeping the search local rather than attending to the entire frame; at image boundaries, only valid tokens are included. The current video query has temporal RoPE position zero, while retained video frames have negative relative offsets; reference tokens use only spatial RoPE. At most $W - 1$ preceding video frames can contribute K/V to a query, with W including the current frame.

## A.3 VARO REWARD FUNCTION

Let $q \ell , m$ be a calibrated component score, oriented so that higher is better, and let $w _ { \ell , m } \geq 0$ be its weight, with $\ell = s$ for the system and $\ell = r$ for references. The system and reference rewards are

$$
R _ { \mathrm { s y s } } = \sum _ { m \in \{ L , D , M , W \} } w _ { s , m } q _ { s , m } , R _ { \mathrm { r e f } } = \sum _ { m \in \{ L , M , W \} } w _ { r , m } q _ { r , m } .\tag{8}
$$

Here L is negative RGB $L _ { 1 }$ error, D is the technical-branch DOVER++ score (Wu et al., 2023), M is MUSIQ (Ke et al., 2021), and W is negative flow-guided warping error. System scores use the emitted video; reference scores use decoded keyframes. The warping term compares consecutive emitted frames, including reference updates, or successive decoded keyframes, respectively. The two level scores enter the combined objective in Eq. (2).

## B TRAINING DETAILS

## B.1 MODEL CONFIGURATION

During base training, the Wan backbone remains frozen: the large generative model updates its LoRA adapters and LR projection, and the GAN discriminator learns a pooling head over frozen Wan features. The lightweight VSR network is trained end to end. Table 5 reports the dimensions and local attention windows of these components.

Table 5: Model configuration. Parameter counts include trainable parameters only; spatial attention windows are measured in tokens.
<table><tr><td>Setting</td><td>Value</td></tr><tr><td colspan="2">Large generative model</td></tr><tr><td>Pretrained backbone</td><td>Wan2.2-TI2V-5B</td></tr><tr><td>LoRA configuration</td><td>Rank r = 512, scaling factor α = 512</td></tr><tr><td>LR projection hidden widths</td><td>768, 3,072</td></tr><tr><td>Temporal attention window</td><td>4 keyframes</td></tr><tr><td>Spatial attention window</td><td>22 × 40</td></tr><tr><td>Trainable parameters</td><td>1.32B (LoRA and LR projection)</td></tr><tr><td colspan="2">Lightweight VSR network</td></tr><tr><td>Transformer</td><td>10 blocks, width 512</td></tr><tr><td>LR encoder channels</td><td>64, 128, 128</td></tr><tr><td>LR encoder strides</td><td>2,2,1</td></tr><tr><td>Reference input projection</td><td>1 × 1 convolution, 48 to 512 channels</td></tr><tr><td>Video attention window</td><td>2 frames, 24 × 24 spatially</td></tr><tr><td>Reference attention window</td><td>48 × 48 for self-attention and reading</td></tr><tr><td>RGB output head</td><td>Width 1,024; PixelShuffle × 16</td></tr><tr><td>Trainable parameters</td><td>31.9M</td></tr><tr><td colspan="2">GAN discriminator</td></tr><tr><td>Wan feature layers</td><td>7, 14, 21 (frozen)</td></tr><tr><td>Pooling head</td><td>4 queries, 12 heads, width 3,072</td></tr><tr><td>Trainable parameters</td><td>198.4M (pooling head)</td></tr></table>

## B.2 BASE TRAINING SETTINGS

All three runs use BF16 computation and FP32 gradient reduction. For flow matching, AdamW uses $( \beta _ { 1 } , \beta _ { 2 } ) = ( 0 . 9 , 0 . 9 5 )$ . The learning rate warms up for 100 updates from 1% of its target value and then stays constant. Diffusion times follow a logit-normal distribution with mean 0, standard deviation 1, and shift 3; text-conditioning dropout is 0.1.

For one-step GAN distillation, the adversarial and feature-space R1 weights in Eq. (7) are 0.1 and 1,000. The generator and discriminator use AdamW coefficients (0.5, 0.99) and (0, 0.99), respectively. The discriminator trains alone for the first 20 updates, warming its learning rate from 1% of the target value; the generator then trains at a constant learning rate. Feature-space R1 is applied at every discriminator update, and the selected generator uses an EMA decay of 0.999.

Table 6: Base training settings. Batch sizes are global. GAN learning rate and weight decay apply to both generator and discriminator.
<table><tr><td>Setting</td><td>Flow matching</td><td>One-step GAN</td><td>Lightweight VSR</td></tr><tr><td>Initialization</td><td>Wan2.2</td><td>Adapted generator</td><td>Random</td></tr><tr><td>Video frames per clip</td><td>8</td><td>8</td><td>16</td></tr><tr><td>HR crop</td><td> $7 0 4 \times 1 2 8 0$ </td><td> $7 0 4 \times 1 2 8 0$ </td><td> $1 0 2 4 \times 1 5 3 6$ </td></tr><tr><td>Global video / image batch</td><td> $3 2 / 9 6$ </td><td> $3 2 / 9 6$ </td><td> $6 4 / -$ </td></tr><tr><td>Updates</td><td>75K</td><td>5K</td><td>60K</td></tr><tr><td>GPUs</td><td>16</td><td>16</td><td>16</td></tr><tr><td>Optimizer</td><td> $\mathrm { A d a m W }$ </td><td> $\mathrm { A d a m W }$ </td><td> $\mathrm { A d a m W }$ </td></tr><tr><td>Learning rate</td><td> $2 \times 1 0 ^ { - 5 }$ </td><td> $1 0 ^ { - 5 }$ </td><td> $1 0 ^ { - 4 }$ </td></tr><tr><td>Weight decay</td><td> $1 0 ^ { - 4 }$ </td><td> $1 0 ^ { - 4 }$ </td><td>0</td></tr><tr><td>Maximum gradient norm</td><td>1</td><td> $\mathrm { G } ; 2 ; \mathrm { D } ; 5$ </td><td>1</td></tr></table>

The lightweight VSR network uses an all-frame loss of $L _ { 1 } \ + \ 0 . 1 \ \mathrm { L P I P S } .$ with AlexNet LPIPS (Zhang et al., 2018). The right reference is dropped with probability 0.25. AdamW uses $( \beta _ { 1 } , \beta _ { 2 } ) = ( 0 . 9 , 0 . 9 9 )$ and $\epsilon = 1 0 ^ { - 8 }$ ; the learning rate warms up for 500 updates from 1% of its target value and then stays constant. The selected network uses an EMA decay of 0.9995.

## B.3 VARO TRAINING SETTINGS

VARO trains on 46-frame clips with keyframes at positions 1, 16, 31, and 46, giving three complete intervals under dual-endpoint conditioning. For each LR–HR clip, the sampling policy draws four candidate reference sequences, each of which receives one combined reward for its full rollout. Two clips per update therefore produce eight scored rollouts.

We standardize each reward component and the system- and reference-level totals using fixed $z -$ score statistics. For each clip, we subtract the mean reward of its four candidates and divide by the global standard deviation across the update batch; clipping to [−1, 1] and rescaling gives the DiffusionNFT weight $r \in [ 0$ , 1]. After each optimizer step, the sampling policy retains a fraction $\eta _ { i }$ of its previous weights and takes $1 - \eta _ { i }$ from the updated generator, where i counts VARO updates. The level weights in Table 7 place greater emphasis on the emitted video while preserving direct feedback on decoded references.

## C ADDITIONAL RESULTS AND ANALYSIS

## C.1 EFFICIENCY ACROSS RESOLUTIONS AND REFERENCE INTERVALS

Comparison with baselines. Table 8 extends the main-paper 1080p efficiency comparison to four output resolutions. Each method processes 201 frames of 4× VSR on one NVIDIA A100-SXM4- 80GB. FPS measures the complete sequence of model CUDA-event calls after warmup; input/output handling, preprocessing, and transfers outside those calls are excluded. Peak memory is PyTorch allocated memory. Spatial tiling is enabled for baselines when needed to avoid out-of-memory errors.

Both RelayVSR modes have higher model FPS than the measured baselines at each resolution. At 2160p, dual-endpoint conditioning reaches 8.23 FPS with 24.25 GB peak allocated memory, compared with 1.302 FPS and 67.99 GB for FlashVSR and 5.773 FPS and 65.26 GB for SwiftVR. First-keyframe conditioning reaches 10.05 FPS with 23.86 GB.

Reference-interval sweep. Table 9 isolates the effect of keyframe spacing on RelayVSR throughput. It uses the same 201-frame model-computation protocol, with three timed runs per setting and both models included in each FPS value.

Table 7: VARO training settings. Batch size counts LR–HR clips. Weight tuples follow the component order in Eq. (8).
<table><tr><td>Setting</td><td>Value</td></tr><tr><td>Rollout</td><td></td></tr><tr><td>Training clip</td><td>46 frames at 704 × 1280 HR</td></tr><tr><td>Reference schedule</td><td> $\Delta = 1 5 ,$  dual-endpoint</td></tr><tr><td>Candidates per clip</td><td>4</td></tr><tr><td>Optimization</td><td></td></tr><tr><td>Optimizer and learning rate</td><td> $\mathrm { A d a m w } , 5 \times 1 0 ^ { - 6 }$ </td></tr><tr><td>Batch size</td><td>2 clips (8 rollouts)</td></tr><tr><td>Training updates</td><td>2,000</td></tr><tr><td>Hardware and precision</td><td>16 GPUs, BF16</td></tr><tr><td>Sampling-policy EMA</td><td> $\eta _ { i } = \operatorname* { m i n } ( 0 . 0 0 1 i , 0 . 5 )$ </td></tr><tr><td>DiffusionNFT mixing coefficient β</td><td>1</td></tr><tr><td>Reward configuration</td><td></td></tr><tr><td>Video MUSIQ stride</td><td>4 frames</td></tr><tr><td>System reward weights</td><td>(0.40, 0.25, 0.25, 0.10)</td></tr><tr><td>Reference reward weights</td><td>(0.50, 0.40, 0.10)</td></tr><tr><td>Level weights  $( \lambda _ { \mathrm { s y s } } , \lambda _ { \mathrm { r e f } } )$ </td><td>(1.00, 0.25)</td></tr></table>

Table 8: Inference efficiency across output resolutions. Model FPS and peak allocated memory (GB) for 201-frame 4× VSR on one A100-80G. RelayVSR uses $\Delta = 1 5$ . Best and second-best entries in each column are bold and underlined, respectively.
<table><tr><td rowspan="2">Method</td><td colspan="2">720p</td><td colspan="2">1080p</td><td colspan="2">1440p</td><td colspan="2">2160p</td></tr><tr><td>FPS ↑</td><td>GB↓</td><td>FPS↑</td><td>GB↓</td><td>FPS↑</td><td>GB↓</td><td>FPS↑</td><td>GB↓</td></tr><tr><td>RealBasicVSR</td><td>41.758</td><td>11.04</td><td>18.502</td><td>24.81</td><td>10.671</td><td>38.04</td><td>4.888</td><td>66.54</td></tr><tr><td>RealViformer</td><td>31.884</td><td>5.44</td><td>14.873</td><td>12.36</td><td>8.689</td><td>21.66</td><td>3.880</td><td>39.03</td></tr><tr><td>Stream-DiffVSR</td><td>1.759</td><td>10.24</td><td>0.849</td><td>25.92</td><td>0.468</td><td>61.67</td><td>0.169</td><td>44.39</td></tr><tr><td>FlashVSR</td><td>16.518</td><td>12.90</td><td>7.799</td><td>24.45</td><td>4.410</td><td>40.61</td><td>1.302</td><td>67.99</td></tr><tr><td>SwiftVR</td><td>51.056</td><td>21.25</td><td>23.483</td><td>31.36</td><td>13.367</td><td>40.74</td><td>5.773</td><td>65.26</td></tr><tr><td>SeedVR2-3B</td><td>2.645</td><td>75.05</td><td>0.788</td><td>73.89</td><td>0.438</td><td>74.88</td><td>0.145</td><td>61.46</td></tr><tr><td>DOVE</td><td>1.529</td><td>30.52</td><td>0.563</td><td>41.08</td><td>0.238</td><td>57.12</td><td>0.152</td><td>42.11</td></tr><tr><td>SparkVSR</td><td>1.525</td><td>30.62</td><td>0.564</td><td>41.30</td><td>0.237</td><td>57.51</td><td>0.152</td><td>42.34</td></tr><tr><td>RelayVSR (dual-endpoint)</td><td>58.320</td><td>11.92</td><td>29.290</td><td>13.82</td><td>17.590</td><td>16.47</td><td>8.230</td><td>24.25</td></tr><tr><td>RelayVSR (first-keyframe)</td><td>68.590</td><td>11.86</td><td>35.890</td><td>13.72</td><td>21.440</td><td>16.31</td><td>10.050</td><td>23.86</td></tr></table>

Increasing ∆ from 5 to 25 raises 2160p throughput from 4.91 to 9.69 FPS in dual-endpoint mode and from 5.55 to 12.39 FPS in first-keyframe mode. $\mathrm { { A t } } \Delta = 1 5$ , first-output model time at 2160p is 1.249 s for dual-endpoint and 0.690 s for first-keyframe conditioning; these times exclude input wait.

## C.2 ADDITIONAL QUALITATIVE COMPARISONS

Figure 6 follows a license-plate crop between reference keyframes 00 and 15. Using both endpoint references, RelayVSR retains recognizable plate characters and colors at intermediate frames 03, 05, 08, 10, and 13 without generating references for these frames.

Figures 7 and 8 extend the main-paper visual comparison to six further scenes. Their aligned crops show printed characters, faces, coral, fur, and a car partly obscured by foliage. These single-frame comparisons broaden the range of visible details, while Fig. 6 shows how a detail persists between reference updates.

Table 9: RelayVSR throughput across reference intervals. Model FPS for 201-frame videos at four output resolutions on one A100-80G.
<table><tr><td>Mode</td><td> $\Delta$ </td><td>720p</td><td>1080p</td><td>1440p</td><td>2160p</td></tr><tr><td>Dual-endpoint</td><td>5</td><td>36.02</td><td>18.10</td><td>10.58</td><td>4.91</td></tr><tr><td></td><td>10</td><td>50.80</td><td>25.60</td><td>15.24</td><td>7.12</td></tr><tr><td></td><td>15</td><td>58.32</td><td>29.29</td><td>17.59</td><td>8.23</td></tr><tr><td></td><td>20</td><td>64.34</td><td>32.20</td><td>19.56</td><td>9.17</td></tr><tr><td></td><td>25</td><td>68.10</td><td>34.13</td><td>20.75</td><td>9.69</td></tr><tr><td>First-keyframe</td><td>5</td><td>40.55</td><td>20.42</td><td>11.89</td><td>5.55</td></tr><tr><td></td><td>10</td><td>60.69</td><td>30.53</td><td>18.09</td><td>8.46</td></tr><tr><td></td><td>15</td><td>68.59</td><td>35.89</td><td>21.44</td><td>10.05</td></tr><tr><td></td><td>20</td><td>80.58</td><td>40.64</td><td>24.52</td><td>11.41</td></tr><tr><td></td><td>25</td><td>85.94</td><td>43.57</td><td>26.34</td><td>12.39</td></tr></table>

Table 10: LongVSR60 quality. The benchmark contains 30 real-world and 30 AI-generated videos. NIQE, MUSIQ, and CLIP-IQA use 100 uniformly sampled frames per video; SC, BC, and MS are supplementary temporal indicators. Best and second-best values are bold and underlined, respectively.
<table><tr><td colspan="5">No-reference quality</td><td colspan="3">Temporal indicators</td></tr><tr><td>Method</td><td>NIQE ↓</td><td>MUSIQ↑</td><td>CLIP-IQA ↑</td><td>DOVER ↑</td><td>SC↑</td><td>BC↑</td><td>MS ↑</td></tr><tr><td>RealViformer</td><td>5.0770</td><td>46.17</td><td>0.3819</td><td>0.5053</td><td>0.8846</td><td>0.9198</td><td>0.9848</td></tr><tr><td>Stream-DiffVSR</td><td>4.0491</td><td>54.52</td><td>0.4694</td><td>0.5704</td><td>0.8830</td><td>0.9161</td><td>0.9813</td></tr><tr><td>SwiftVR</td><td>4.2822</td><td>53.89</td><td>0.4640</td><td>0.5512</td><td>0.8831</td><td>0.9177</td><td>0.9824</td></tr><tr><td>FlashVSR</td><td>3.8160</td><td>55.15</td><td>0.4747</td><td>0.5895</td><td>0.8829</td><td>0.9144</td><td>0.9802</td></tr><tr><td>RelayVSR</td><td>4.1396</td><td>54.72</td><td>0.4701</td><td>0.5911</td><td>0.8837</td><td>0.9157</td><td>0.9827</td></tr></table>

## C.3 LONG-VIDEO QUALITY ON LONGVSR60

LongVSR60 extends evaluation to 30 real-world and 30 AI-generated single-shot videos of approximately 1,000 frames each. In addition to NIQE, MUSIQ, CLIP-IQA, and DOVER, we report subject consistency (SC), background consistency (BC), and motion smoothness (MS) following VBench (Huang et al., 2024) as supplementary temporal indicators. SC and BC measure crossframe appearance consistency using DINO and CLIP features, respectively. Motion smoothness (MS) follows the interpolation-based criterion in VBench, measuring agreement between observed frames and frames interpolated from temporal neighbors. Table 10 reports RelayVSR and selected baselines on the 60-video benchmark.

## C.4 USER STUDY

We conducted a blind pairwise study with 12 participants and 36 videos: 12 from VideoLQ and 12 each from the real-world and AI-generated subsets of LongVSR60. We use 10–15 s excerpts, or full videos when shorter; LongVSR60 excerpts are taken after full-sequence inference. RelayVSR uses dual-endpoint conditioning with $\Delta = 1 \bar { 5 }$ and is compared with DOVE, SeedVR2, SparkVSR, SwiftVR, and FlashVSR-Tiny under the main evaluation settings.

The LR input and anonymized A/B outputs were played synchronously at matched resolution and frame rate, with randomized left–right placement and trial order. Participants judged overall visual quality, fidelity to observable LR content, and temporal stability, choosing A better, similar, or B better. Unable-to-judge fidelity responses were excluded from that criterion’s score.

Following the GSB protocol (Zhuang et al., 2026), we report $1 0 0 \% \times ( G - B ) / ( G + S + B )$ where G, S, and B count better, similar, and worse judgments for RelayVSR. Results are reported in Table 11.

Table 11: User study. GSB scores (%) compare RelayVSR with each baseline. Positive scores favor RelayVSR; negative scores favor the baseline.
<table><tr><td>Baseline</td><td>Overall quality ↑</td><td>Content fidelity ↑</td><td>Temporal stability ↑</td></tr><tr><td>DOVE</td><td>+53.6</td><td>-8.9</td><td>-8.7</td></tr><tr><td>SeedVR2</td><td>+46.8</td><td>+13.5</td><td>+13.1</td></tr><tr><td>SparkVSR</td><td>+43.9</td><td>+21.9</td><td>+5.9</td></tr><tr><td>SwiftVR</td><td>+26.7</td><td>+13.2</td><td>+11.9</td></tr><tr><td>FlashVSR</td><td>-3.3</td><td>+9.0</td><td>+17.0</td></tr></table>

RelayVSR receives higher overall-quality preference than DOVE, SeedVR2, SparkVSR, and SwiftVR, with GSB scores ranging from +26.7% to +53.6%. Against FlashVSR, the near-neutral overall-quality score (-3.3%) supports comparable perceived quality, while content fidelity (+9.0%) and temporal stability (+17.0%) favor RelayVSR. These perceptual results complement the modelcomputation efficiency gains reported in Sec. 4.3.

## D LIMITATIONS AND FUTURE WORK

RelayVSR uses a fixed keyframe interval, while dual-endpoint conditioning requires bounded lookahead. Adapting keyframe placement and reference selection to content changes and latency budgets could improve the balance between video quality and computational cost. Texture or color errors in generated references can propagate across output frames. Complementing VARO with reference reliability estimation at inference could guide memory reading and reference refresh, helping the lightweight VSR network adjust its reliance on generated details. Although sparse invocation reduces average computation, the large generative model still contributes to memory use. Our efficiency results are measured on an NVIDIA A100. Model compression and smaller reference generators could support deployment on devices with tighter resource budgets.

![](images/ebe24f29d9b3ab7a60b6253c67217bfb0a9f945e0c3426a4a074ca76f0614c6b.jpg)

![](images/584b68eff391839465442a24683debd8938a2bec03aa8f7a96ac4bcf5d77a333.jpg)  
Figure 6: Detail propagation between sparse references. Left: the source frame locates the fixed crop. Right: keyframes 00 and 15 (marked ∗) and five intervening frames compare the same license plate region in the LR input, RealViformer, FlashVSR, and RelayVSR.

![](images/580e0b83e78ba7fb3a4427369c85ac8312c81869a43cbbd7c8f323330f304ac4.jpg)  
Figure 7: Additional visual comparisons I. Three examples highlight printed backdrop text and a face, underwater coral structure, and a dog’s face and fur. Left: full-frame context with marked crops. Right: aligned crops from the input and six VSR methods, including RelayVSR.

![](images/0027f1fd4b4a7226f8f6d3a9a8f5feb2449d8d1abcc32bde74237c751991c02e.jpg)  
Figure 8: Additional visual comparisons II. Aligned crops show lettering on a street sign, a squirrel, and a car partly seen through foliage. Each example includes full-frame context at left and comparisons with earlier and recent VSR methods at right.
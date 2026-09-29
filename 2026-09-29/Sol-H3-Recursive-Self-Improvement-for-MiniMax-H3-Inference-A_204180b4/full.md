# Sol-H3: Recursive Self-Improvement for MiniMax-H3 Inference Acceleration on Sol-Engine across Cloud and Edge

Yitong Li<sup>\*</sup>, Jincheng Yu<sup>\*</sup>, Junsong Chen<sup>\*</sup>, Haopeng Li<sup>\*</sup>, Shuchen Xue, Haozhe Liu, Ping Luo, Song Han, Enze Xie

Video diffusion models are rapidly scaling and exhibiting enhanced generation capabilities. Among these recent advancements, MiniMax-H3 stands out as a highly capable, production-level open-source model. However, its 33-billion parameters and multi-step iterative denoising process introduce substantial computational overhead. Consequently, their practical production is hindered by generation latency in the cloud deployment like NVIDIA-GB200, alongside strict memory limits that pose further challenges at the edge device like DGX-Spark. To address these diverse hardware bottlenecks from cloud to edge device, we present a full-stack inference pipeline that integrates efficient algorithmic design with optimized operator implementations. Algorithmically, we introduce a cross-resolution twostage generation scheduler that exploits the step-wise nature of diffusion: early low-resolution steps rapidly establish the global layout, while later high-resolution steps focus refinements of local and perceptual details. These stages are connected by a learned latent-to-latent mapping module, completely eliminating the computationally expensive VAE decode-reencode cycle for resolution transferring cross different resolutions. For operator implementation, we deploy a Recursive Self-Improvement (RSI) loop that searches kernel fusions and memory layouts, evaluating latency together with numerical agreement. These optimizations ensure our streamlined algorithmic design fully realizes its performance potential and practical latency benefits across varying hardware architectures. By combining these innovations, our pipeline achieves exceptional efficiency up to 30× speedup and 20% memory saving, delivering up to 3.5× faster-than-real-time generation on 8×GB200 cloud nodes and less than 1-min fully memory-resident execution on DGX-Spark.

Links: Github Code | Project Page

Sol-H3

≈30× speedup GB200: 6.226 s · DGX Spark: 56.17 s

17.9% less total memory DGX Spark: 142.4 → 116.9 GiB

![](images/8090e318cee6c66416c0c926dc953b37fbd171a4e273ae217cee9a7baa74d47f.jpg)  
Figure 1 | Sol-H3: two-stage video generation with recursive self-improvement. A cross-resolution two-stage inference pipeline reduces the overhead of full-resolution denoising. Recursive Self-Improvement (RSI) profiles, optimizes, and verifies implementations under restricted generation settings, producing optimized kernels and effective memory layouts. The combination of this inference algorithm design and efficient operator implementation enables MiniMax-H3 to achieve approximately 30x acceleration and reduce GPU memory consumption by nearly 20%.

## 1. Introduction

Driven by model and data scaling, video diffusion models [1, 2, 3] have achieved remarkable generative prowess, with recent open-source releases like MiniMax-H3 delivering production-grade quality [4, 5, 6, 7, 8, 9]. Despite these advances, the model’s massive 33-billion parameter and the multi-step iterative denoising process [1, 10] impose a severe computational burden. Furthermore, the specific bottlenecks encountered during inference vary drastically across different hardware platforms and deployment scenarios. In cloud environments, system throughput and generation cost act as the primary hurdles, ultimately determining the economic viability of the service. Conversely, edge deployments are tightly constrained by memory limits; exceeding the available HBM forces the system to cpu-offloading, precipitating a catastrophic collapse in inference latency. Since these hardware constraints, a single optimization strategy cannot solve both issues, making the overall optimization process much more difficult.

The acceleration of diffusion models has relied on two distinct paradigms. The first is algorithmic acceleration, which inherently alters the generation process through methods such as reducing sampling steps [11, 12, 13, 14, 15], caching denoising computation [16, 17, 18, 19, 20], implementing sparse attention mechanisms [21, 22, 23, 24, 25, 26], or modifying generation resolutions [27, 28, 29], which are often in a lossy manner. The second is system-level acceleration, which preserves the mathematical output while applying low-level system optimizations, such as kernel fusion, memory layout restructuring, and communication scheduling [30, 31, 32, 33].

In this report, we optimize the inference pipeline from two perspectives of algorithm and implementation. For algorithmic acceleration, we leverage a fundamental evolutionary property of diffusion models: early denoising steps determine the overall structural layout of the video, while later steps synthesize high-frequency fine details. Building on this, we design a cross-resolution two-stage generation pipeline tailored to maximize efficiency at each phase. The first stage operates at a low resolution to rapidly establish the global layout and structural skeleton. The second stage shifts to a high resolution to focus entirely on refining visual details. To seamlessly connect these stages, we train a feature mapping module that performs direct latent-to-latent translation, completely eliminating the latency and memory overhead of the traditional pixel-level decode-and-reencode cycle. For system-level implementation, we introduce a Recursive Self-Improvement (RSI) loop [34] to conduct iterative and verifiable optimization. Operating under strict correctness verification, this RSI loop automatically explores a massive search space encompassing kernel fusions, memory layouts, and communication boundaries. The automated search consistently uncovers and optimizes low-level compute and communication bottlenecks, delivering massive kernel-level speedups across a wide range of hardware platforms. These optimization realizes the practical latency benefits of our efficient inference pipeline across varying hardware architectures, while requiring less human effort.

Integrating these algorithmic and systematic optimizations, we build a highly efficient full-stack inference pipeline across various hardware platform. On an 8×GB200 node, Sol-H3 generates a five-second video with native audio in 1.434 seconds, approximately 3.5× faster than real-time playback (Tab. 1). On a single GB200, it achieves an end2end 6.23s generation, 22.2× speedup over the serving baseline [35]. Furthermore, on a single DGX Spark, the entire pipeline remains fully HBM resident, completing the video generation in 56.2 seconds without triggering cpu offloading [36], achieving a near 20% memory reduction more than 30× speedup .

We summarize the main contributions of this report below.

• A cross-resolution two-stage generation pipeline. Exploiting the step-wise nature of diffusion models, we allocates low-resolution earlier steps to global layout generation and high-resolution later steps to fine detail refinement, reducing the algorithmic redundancy of uniform-resolution denoising process.

• A full-stack inference implementation for MiniMax-H3. Through the co-design of algorithmic and system-level optimizations, our pipeline achieves an approximate 30× acceleration and a 20% memory reduction across cloud and edge environments. This unlocks practical deployment for MiniMax-H3.

• Automated system-level optimization via Recursive Self-Improvement (RSI). We introduce an RSI loop that searches kernel fusions, memory layouts, collective layouts, and transport precision, profiles their latency, and checks numerical agreement. We distinguish bit-exact layout transformations from arithmetic fusions and approximate execution modes, giving the reported accelerations an explicit numerical scope.

## 2. Related Work

## 2.1. Video Diffusion Models

The current landscape of generative video is defined by open, large-scale diffusion transformers. Prominent base models include the CogVideo family [8, 37], HunyuanVideo [5] and its successor HunyuanVideo 1.5 [6], as well as Wan [7], LongCat-Video [38], SANA-Video [39], Cosmos 3 [40], and JoyAI-Echo [41]. This report specifically serves MiniMax-H3 [4], a 33-billion parameter joint audio-video generator, alongside LTX-2.5 [9] and its predecessor LTX-2.3 [42], which provide second-stage refinement. Our latent handoff combines spatial upsampling with a separate learned translation between the H3 and LTX VAE representations. The joint audio-video nature of MiniMax-H3 introduces a distinct serving challenge compared to vision-only models: the audio track must survive every optimization applied to the pipeline, not merely the visual frames.

To manage the computational complexity of high-resolution generation, cascaded and multi-resolution generation schedules are well-established recipes in the literature [27, 29]. We employ a similar paradigm, utilizing a low-resolution draft from MiniMax-H3 followed by a high-resolution refinement pass with LTX-2.5. We emphasize that the lowresolution draft and high-resolution refine strategy is prior art; our contribution, detailed in Sec. 3.1, lies in the serving realization and the end-to-end execution stack rather than the conceptual schedule itself.

## 2.2. Inference Acceleration for Video Diffusion

Inference acceleration for diffusion transformers generally operates across three primary levers. The first lever is sparse attention, which mitigates the quadratic scaling of sequence length in video generation. Recent approaches include Sparse VideoGen [21] and Sparse VideoGen2 [22], trainable sparse attention (VSA) [23], Sliding Tile Attention [24], SpargeAttn [43], XAttention [26], Radial Attention [25], DraftAttention [44], and PISA [45]. Additionally, the SageAttention line of work [46, 47, 48] provides highly optimized attention backends. We leverage similar principles for our sparse attention implementation, detailed in Sec. 3.4. The second lever involves caching mechanisms and techniques to reduce the number of effective steps, thereby limiting how much of the network is evaluated per step. Methods such as TeaCache [16], EasyCache [17], Pyramid Attention Broadcast [18], TaylorSeer [19], and Cache-DiT [20] exploit temporal redundancy. These step-reduction techniques are orthogonal to our two-stage schedule; while caching reduces computation within a fixed resolution trajectory, our schedule alters the resolution trajectory itself.

The third lever encompasses quantization, token pruning, and kernel-level optimizations. Quantization techniques for diffusion models include ViDiT-Q [49], PTQ4DiT [50], Q-DiT [51], SVDQuant [52], AWQ [53], and the adoption of microscaling formats [54]. Token-level reduction strategies, such as ToMeSD [55], Astraea [56], TAPE [57], and CoReDiT [58], further compress the computational graph. These optimizations are typically deployed within the broader context of modern serving systems and sequence parallelism frameworks, including vLLM [59], SGLang [60], DeepSpeed Ulysses [61], and LongLive-2.0 [62].

Throughout this report, we lean heavily on a fundamental distinction in inference optimization: methods that preserve the generation contract versus methods that alter the schedule or the step count. Techniques such as operator fusion (discussed in Sec. 3.5), exact attention backends, and lossless collective layouts provide like-for-like runtime speedups without changing the output. Conversely, schedule modifications produce a fundamentally different artifact. While these two categories of optimization compose effectively in practice, they must not be conflated when reporting speedups, as evaluating a different generation contract is not equivalent to accelerating a fixed one.

## 2.3. Agentic and Self-Improving Optimization Systems

The general context for our recursive self-improvement loop is the rapid emergence of agentic coding and machinelearning experimentation systems. Frameworks such as SWE-agent [63], AutoCodeRover [64], Agentless [65], and OpenHands [66] automate software engineering tasks, while benchmarks and systems like AgentBench [67], MLAgent-Bench [68], The AI Scientist [69], and AI Harness Engineering [70] explore autonomous research and experimentation. More specifically, this report belongs to the lineage of agents designed to write and optimize GPU kernels. The KernelBench benchmark [71] demonstrated that frontier models perform this task poorly without iterative feedback. Subsequent systems have addressed this gap: CUDA-LLM [72] and CudaForge [73] focus on kernel generation, AccelOpt [74] introduces a self-improving agentic system that curates an optimization memory of slow-fast kernel pairs, and AlphaEvolve [75] employs an evolutionary coding agent for algorithmic discovery.

Our direct predecessor in this domain is the Sol video inference engine [34], an agent-native, full-stack acceleration framework. Sol composes caching, sparse attention, token pruning, quantization, and kernel fusion through a hierarchy of specialized skill agents and a central integrator agent. This report represents the deep application of the Sol framework to a single model family, extending the prior work by introducing a strict fail-closed contract as an explicit constraint on the optimization process.

This fail-closed constraint forms a critical positioning argument for our methodology. Unconstrained self-improving agents may inadvertently redefine the task they are optimizing—for instance, by silently reducing step counts or altering the numerical trajectory—which makes their reported performance gains difficult to attribute. Our recursive self-improvement loop is deliberately restricted: it searches realizations of a fixed generation contract (such as numerical formats, kernel boundaries, attention backends, collective layouts, and memory residency) but never alters the contract itself. Consequently, an accepted change is guaranteed to be a like-for-like reduction measured against the same baseline on the same hardware, and a rejected candidate yields a precise measurement rather than an unquantified judgment.

## 3. Full-Stack Optimization for Cloud Deployments

Given the massive computational demands of video generation, cloud environments serve as the primary deployment vehicle. To maximize serving efficiency and reduce the compute cost per generation, we restructure the original single-stage, uniform-resolution denoising process into an asymmetric two-stage generation pipeline. We further apply a Recursive Self-Improvement (RSI) loop to iteratively optimize the execution stack. This RSI process systematically eliminates redundancy across each stage, elevating computational efficiency and generation throughput.

In this section, we detail algorithm, kernel, and system optimizations for the H3 runtime and its cross-resolution extension. The applicable configurations are evaluated separately in Sec. 6. In particular, single-stage Sol-H3 on 8×GB200 in Sec. 6.2 generates a five-second video in 1.434 seconds, approximately 3.5× faster than real-time playback.

## 3.1. Cross-Resolution Generation

Cross-resolution two-stage generation. While the released pipeline runs the entire denoising trajectory at full resolution, we convert it into a two-stage efficient inference pipeline. Because early diffusion iterations establish global layout and motion while later steps synthesize high-frequency detail, executing the full trajectory at native resolution expends most compute resolving structural composition before fine features emerge. To address this, the Sol-H3 pipeline allocates spatial resolution dynamically, restructuring the conventional 49 denoising steps at full resolution (1344 × 768) into an asymmetric multi-stage schedule:

$$
\underbrace { 4 9 \mathrm { ~ s t e p s ~ a t ~ 1 3 4 4 \times 7 6 8 } } _ { \mathrm { r e l e a s e d p i e l i n e } } \quad \longrightarrow \quad \underbrace { 4 \mathrm { s t e p s ~ a t 6 7 2 \times 3 8 4 } } _ { \mathrm { S u g e 1 : c o n t e n t ~ a n d m o i o n } } + \underbrace { \mathrm { l a t e n t ~ h a n d o f f } } _ { \mathrm { \times 2 u p s a m p l e + V A E ~ t r a n s l a t i o n } } + \underbrace { 3 \mathrm { s t e p s ~ a t ~ 1 3 4 4 \times 7 6 8 } } _ { \mathrm { S u g e 2 : d e t a i l } } .
$$

Stage 1 generates core visual elements and the primary audio track using MiniMax-H3 [4] augmented with the FastH3 VSA DataFree LoRA [23]. It executes four update iterations at $6 7 2 \times 3 8 4 \times 1 2 4$ frames. Stage 2 synthesizes highfrequency detail using LTX-2.5 [9]. It executes three joint audio–video updates on the 1344 × 768 latent canvas using a resident BF16 LTX-2.5 dev backbone with distilled LoRA 450 applied at strength 0.8.

![](images/e329d192bbefbc9a60d1e3f3775527f86f9904f29c3d9ef56a9f8db87b9833e1.jpg)  
Figure 2 | System overview. We accelerates the generation process through a two-stage pipeline and a latent-to-latent connector. Stage 1 generates content at 672×384 for 4-steps; an H3 latent upscaler doubles spatial resolution, and a separate adapter translates the result into LTX latent coordinates; Stage 2 refines at 1344×768 for 3-steps.

Cross-resolution latent space adapter. The two stages use independently trained VAEs [2], so matching spatial resolution alone does not make their latents interchangeable. H3 represents video with 24 channels and spatial compression of 16×, whereas the LTX-2.5 Conv Video VAE uses 128 channels and spatial compression of 32×. A direct connection would decode the H3 latent into pixels and re-encode those pixels with the LTX VAE, adding two large networks and their intermediate activations to the handoff. We replace this round trip with a learned latent-to-latent translator. Spatial enlargement remains a separate operation: the learned H3 ×2 upscaler first raises the draft resolution, and the translator then maps that representation to LTX at the same pixel resolution.

The main alignment difficulty is temporal. H3 encodes independent 17-frame chunks with nominal temporal compression of 4×, producing nonuniform token positions across chunk boundaries; LTX uses a causal 8× temporal grid. We therefore align tokens by their physical frame positions rather than their tensor indices. A fixed front end concatenates a linearly interpolated H3 feature at each LTX time position with three slots containing the nearest-bin source tokens. Each source token is retained exactly once in the packed slots. A spatial pixel-unshuffle then moves each $2 \times 2$ neighborhood into channels, producing $2 4 \times ( 1 + 3 ) \times 4 = 3 8 4$ channels on the LTX grid. This parameter-free packing preserves the source features instead of averaging them away before learning the cross-VAE mapping.

On this aligned grid, the adapter combines a frozen $1 \times 1 \times 1$ affine skip, initialized by ridge regression, with a learned residual network. The network contains 22 blocks of width 752, each combining spatial convolution, depthwise temporal convolution, and a gated channel MLP. The complete translator has 194.76M parameters and predicts normalized 128-channel LTX latents.

Training pairs are obtained by encoding the same resized video with both frozen VAEs using their posterior modes. Let $z _ { H }$ and $z _ { L }$ denote the normalized H3 and LTX latents, � the fixed alignment, and $A _ { \theta }$ the adapter. We supervise both the latent prediction $\hat { z } _ { L } = A _ { \theta } ( G ( z _ { H } ) )$ and its decoded video:

$$
\mathcal { L } _ { \mathrm { a d a p t e r } } = \mathrm { M S E } ( \hat { z } _ { L } , z _ { L } ) + \alpha \ \mathrm { M S E } ( D _ { L } ( \hat { z } _ { L } ) , D _ { L } ( z _ { L } ) ) ,\tag{1}
$$

where $D _ { L }$ includes inverse latent normalization, the frozen LTX Conv decoder, and conversion to RGB values clamped to [0, 1]. Gradients pass through the decoder to the adapter while the VAE weights remain fixed. This decoded supervision accounts for the unequal visual effects of errors in different latent directions. The final training phases use 65,536 paired videos at $1 3 4 4 \times 7 6 8$ ; implementation and evaluation details appear in Sec. A.

At inference, the upscaler retains its prescribed input and output normalization, and the adapter output enters the refiner directly in normalized LTX coordinates. Applying LTX normalization a second time would change the input distribution. For the five-second serving configuration, temporal conversion produces 17 latent frames; the refiner consumes the first 16 to produce 121 output frames. The adapter operates only on video latents, leaving the Stage-1 audio unchanged. Removing the intermediate H3 decoder and LTX encoder reduces the handoff’s compute and memory requirements, supporting the co-resident deployment described in Sec. 4.

![](images/e5af4874391adef784821b8c6e4b34570fdd31372d49c00976e334d3c201d367.jpg)  
Both losses update only the residual adapter.

H100 conversion profile
<table><tr><td></td><td>ms</td><td>GiB</td></tr><tr><td>Full VAE</td><td>12,748.6</td><td>13.788</td></tr><tr><td>Tiny AutoEncoder</td><td>201.9</td><td>23.846</td></tr><tr><td>Adapter</td><td>63.1</td><td>0.785</td></tr><tr><td>BF16, batch 1, 192 frames, 1344 × 768. Median latency and peak allocated memory.</td><td></td><td></td></tr><tr><td>Final held-out reconstruction</td><td></td><td></td></tr><tr><td>30.618 dB</td><td>0.8962</td><td></td></tr><tr><td>PSNR / SSIM on 256 test clips.</td><td></td><td></td></tr><tr><td>Compared with the decoded LTX teacher.</td><td></td><td></td></tr></table>

Figure 3 | Cross-VAE latent translation. Left: both frozen encoders process the same source video; latent and decoded-pixel supervision train the residual translator. The spatial upscaler is separate from this training task. Right: conversion cost for an earlier checkpoint and held-out quality for the final checkpoint of the same architecture. The profile excludes the upscaler and refiner (Sec. 6.5).

## 3.2. Communication Optimization

We parallelize inference across GPUs, but communication and tensor layout transformations limit the realized speedup. Attention requires exchanges between sequence-sharded and head-sharded layouts, making the surrounding collectives a focus of the RSI optimization loop.

Optimized attention collectives. In a representative full-resolution H3 profile, layout transformations and collectives accumulated 94 ms per denoising step, alongside 341 ms of SDPA attention computation. The 94 ms comprises 42 ms of layout copies and 52 ms of communication. The baseline executes three QKV collectives and one output collective per attention layer. Retaining packed QKV storage combines the three input exchanges into one, reducing the total to two collectives without changing the unquantized payload volume. A flat (total, heads, head\_dim) layout also removes the redundant batch dimension at batch size one. However, a naive PyTorch implementation achieved only 0.953× the reference throughput, a 4.7% reduction: stack followed by a destination-major permute(...).contiguous() copied the activation volume twice before communication. The optimized strideaware kernels instead read Q, K, and V through their existing strides and write directly into the exchange buffer in one pass. The inverse head merge uses the same approach on the return path. These layout-only kernels are bit-identical to the reference permutations they replace.

Quantized communication. Under sequence parallelism [61], the packed QKV exchange and attention-output return require two all-to-all collectives per attention layer. The block-INT8 QKV format stores one token/head record as 384 quantized value bytes, twelve group-32 finite-positive UE5M3 scale codes, and four zero padding bytes, for 400 bytes in total. We evaluate two distinct output formats. The block-INT8 format stores 128 value bytes and four scale codes, padded from 132 to 144 bytes to align each record to 16 bytes; this aligned format improves the measured NCCL transfer performance on GB200 despite its larger payload. The FP8 alternative instead transmits 128 raw E4M3 bytes per token/head, without separate scale metadata or padding. All transmitted padding bytes are explicitly initialized. Both quantized wire formats are approximate and are validated separately from the bit-exact layout transformations.

## 3.3. Compute Quantization

Attention and FFN linear projections process large token matrices and are important targets for low-precision GEMM acceleration [49, 50, 51, 52, 76]. Their end-to-end benefit also depends on the cost of quantization, scale layout conversion, and surrounding data movement. We therefore evaluate the arithmetic kernels together with their producers and communication interfaces.

MXFP8 GEMM computation. Sol-H3 uses MXFP8 GEMMs for attention and FFN projections in zero-indexed transformer blocks 2–46, inclusive, on the validated SM100-family path [54]. Weights and activations use E4M3 values with one E8M0 scale for each group of 32 values along the reduction dimension. The scale buffers use the cuBLASLt-compatible SWIZZLE\_32\_4\_4 layout. Fused RMSNorm/modulation and SwiGLU producers write the E4M3 activations and swizzled scales directly, eliminating the intermediate BF16 tensor and a separate quantization pass. Boundary blocks, AdaLN projections, refiners, VAE, and text/audio encoders retain BF16 computation. The quantization is approximate.

Joint optimization with quantized communication. Sol-H3 reuses the raw E4M3 values returned by attention as inputs to the quantized output projections, supplying unit E8M0 scales. The receiver merges the rank-major heads directly into the projection’s row-major input, avoiding an intermediate BF16 tensor and subsequent dynamic quantization. Integrating this reuse with the fused activation producers removes redundant data conversions between attention, communication, and linear computation.

## 3.4. Sparse Attention

Per-step dynamic sparsity. In the refinement stage, the token count grows substantially with resolution, sharply increasing the computational cost of attention [32, 77]. We therefore employ Sol-Attn [78], a training-free block-sparse attention method [21, 22, 43, 45] that uses online thresholding to select a small subset KV blocks for computation. Specifically, for each query block, the threshold is set to � + ��, where � and � denote the mean and standard deviation of its block-level attention scores across KV blocks, and � is a tunable hyperparameter that controls sparsity. Only KV blocks whose scores exceed this threshold are selected for attention computation; a larger � raises the threshold and thus retains fewer blocks. We observe that sparsification has a greater impact on global image coherence during early, high-noise refinement steps than during later, low-noise steps. We therefore adopt a step-dependent schedule that progressively increases sparsity, setting � to 1.0, 1.25, and 1.5 for the three refinement steps, respectively. To accommodate different GPU architectures, we use cuDNN block-sparse attention (BSA) and a custom CuTe DSL kernel [30] as alternative backends, improving kernel efficiency and multi-GPU serving performance.

Safety rails and the exact prefix KV sink. In practice, we keep the first two transformer layers dense throughout denoising, as errors introduced while global structure is still forming may persist through subsequent layers. Within the remaining attention layers, we use distinct attention paths for prefix and target-video queries. MiniMax-H3 packs each sequence as [ text | conditioning video | audio | target video ], so all tokens preceding the target video form a contiguous prefix. Queries from this prefix attend to all KV tokens through dense cross-attention. The remaining target-video queries use block-sparse attention, with the prefix KV blocks serving as always-attended sinks that are evaluated exactly regardless of the routing threshold. Importantly, this protection covers the entire prefix rather than text tokens alone: the audio tokens are themselves generated, with the model predicting audio velocities at these positions. In validation, we observed severe dialogue degradation even in a case that achieved the highest visual-quality score among the compared variants, highlighting the need to preserve audio fidelity alongside visual quality. This design preserves dense connectivity between the multimodal prefix and the full sequence while retaining block-sparse attention among target-video tokens.

## 3.5. Kernel Fusion and Precision Preservation

The DiT [3] repeatedly writes wide intermediate tensors to HBM and reads them back for subsequent elementwise operations. Fusion reduces this traffic by consuming intermediates in registers. H3 combines RMSNorm, partial rotary embeddings, SwiGLU, and per-row indexed modulation, requiring adaptations to fusion kernels designed for other architectures. The RSI loop searches implementation and launch configurations, profiles their latency, and checks their numerical behavior against the corresponding reference.

Kernel Fusion. The RSI loop identifies three operator groups across the 50 H3 transformer blocks: (1) residual addition with indexed gating, RMSNorm, and indexed scale/shift modulation; (2) QK normalization with partial RoPE; and (3) SwiGLU splitting, SiLU, and multiplication. At a representative four-rank context-parallel shape of 9562 local rows and hidden width 5376, these fusions reduce repeated reads and writes of wide intermediate tensors. Pure layout transformations are bit-identical to their reference permutations. Arithmetic fusions preserve the operator structure, but changes in intermediate precision and floating-point rounding can prevent bitwise equivalence to the original eager implementation; we distinguish these from the separately validated LoRA consumer-fusion equivalence below.

Precision-preserving LoRA fusion. Weight merging removes LoRA’s extra GEMMs, but can change a few-step adapter’s numerical behavior. Small updates can round back to the original weights when � + �� is stored in BF16, even if the merge itself is computed in FP32. The real-arithmetic identity $( W + B A ) x = W x + B ( A x )$ therefore does not imply equivalence in BF16. In a controlled diagnostic of the evaluated eight-step adapter, 86.1–94.4% of nonzero weight updates in six inspected projections rounded back to their original values. These are per-projection weight statistics, not a model-wide fraction or a perceptual-quality loss; Sec. B gives the diagnostic controls.

We instead retain the base and LoRA branches and fuse their addition into the next consumer kernel. Each projection still executes the same three GEMMs with unchanged weights and reduction order. The fused consumer explicitly rounds base + � to BF16 before QK normalization, RoPE and packing, residual modulation, or SwiGLU; the final FFN consumer also retains the intermediate BF16 gate product. Preserving these rounding points removes the wide sum’s HBM write/read pair while retaining the separate-branch arithmetic. The implementation specializes to one active adapter at unit scale with BF16 linear compute, independent of the adapter’s name. Its numerical agreement and runtime are evaluated against native branches with the other acceleration settings held constant in Sec. 6.6.

## 3.6. VAE Decoding

Latent diffusion requires decoding to pixels [2]. The optimizations in this subsection concern the H3 video decoder used in single-stage H3-output paths. They are separate from the latent handoff in Sec. 3.1, which bypasses H3 decoding and passes video latents to the LTX refinement and decoding path.

Parallel decoding. Context parallelism shards the transformer stack but does not by itself partition video decoding. In the evaluated H3 configuration, each rank would otherwise decode the full video. At 1344 × 768 and 124 output frames, the H3 decoder processes seven temporal clips with 28 equal-sized spatial tiles per clip, for 196 tile decodes. Distributing these independent tiles over eight ranks reduces the recorded decode latency from 7.55 s to approximately 1.16 s while retaining the original stitch order.

Local tile batching and compilation. Sharding leaves four tile slots per rank in each temporal clip. Decoding these four tiles as one local batch, rather than launching the decoder separately for each tile, reduces latency from 1.1597 s to 0.6021 s, a 1.93× speedup in the recorded configuration. Compiling the batched decoder reaches 0.3593 s, or 3.23× relative to the 1.1597 s sharded baseline, and reduces the measured decoder peak memory from 18.8 GB to 15.6 GB. These are decode-level measurements rather than whole-pipeline speedups. Compilation is not bit-identical to eager decoding; the recorded maximum absolute deviation is approximately 0.021.

Global tile batching. A separate optimization combines tiles across all seven temporal clips. The 196 real tiles then occupy 200 execution slots, or 25 per rank, with only four padded duplicates, instead of 224 slots and 28 duplicates for per-clip scheduling. The global path invokes the batched decoder and gather once rather than seven times and restores the original tile and blend order afterward. Changing the batch shape can change the selected GEMM implementation: the recorded comparison against the per-clip path has maximum absolute deviation 0.013672 and decoder-output PSNR 75.75 dB over the [−1, 1] range. These values characterize that test, rather than a general error bound or perceptualquality guarantee. The measurements and implementation discussed here use eight ranks; they do not establish the same batching benefit on a single GPU.

## 4. Full-Stack Optimization for Edge Deployments

While cloud environments serve as the primary deployment vehicle due to massive computational demands, there is substantial demand for executing these generative models locally or on edge devices. To enable efficient local deployment, we undertake a comprehensive set of optimizations targeting strict reductions in both latency and device memory footprint. The binding constraint in this regime is device memory residency. It requires the entire working set, including the text encoder, drafting model, refiner, and VAE, must remain simultaneously resident in HBM. If the working set does not fit, the runtime must offload weights to host memory and stream them back on demand, incurring a substantial host-device communication cost that dominates request latency.

On a single DGX Spark, where the CPU and GPU share a 119.68 GiB unified memory pool, our optimizations represent the critical difference between a viable deployment and an out-of-memory failure. Specifically, a naive two-stage deployment demands 142.4 GiB, exceeding the hardware capacity by 22.7 GiB. Our optimized configuration consumes only 116.9 GiB, preserving a 2.8 GiB safety margin during sampling.

In this section, we detail the specific optimization techniques applied to achieve these latency and memory objectives, focusing on strategies such as deterministic modulation caching, prompt caching, and weight quantization. Ultimately, deploying this optimized pipeline on a single-accelerator edge environment, such as a GB10, successfully fits the complete generative stack onto the device while delivering highly responsive local inference.

## 4.1. AdaLN precompute

For MiniMax-H3, approximately 13B of the model’s 33B parameters reside in the adaln\_proj module. At inference time, this module recomputes the modulation table at every denoising step, amounting to 2500 recomputations per generated video. Because the projection’s only input is the timestep embedding, and that embedding depends solely on the sampling schedule fixed before the denoising loop, every one of those recomputations produces a value that was already knowable in advance. Consequently, this repetition is pure redundancy, with the module imposing waste along two dimensions: device memory is consumed because the wide projection weights must stay resident, and compute and bandwidth are expended regenerating the same table at each step. Our runtime precomputes the entire table once and maintains a cache indexed by (block, step). This eliminates two severe bottlenecks. First, it saves about 24 GB of device memory, as replacing the 26 GB of adaln\_proj weights with a precomputed table of ≈ 1.5 GB lowers denoiser residency from 61.7 GB to ≈ 37 GB. Second, it eliminates about 26 GB of HBM reads per step; the reference pipeline streams 520 MB of weight data across HBM per block to produce just nine output rows, incurring a bandwidth tax with minimal arithmetic intensity.

## 4.2. Prompt caching

As established in Sec. 3.1, Stage 1 has already fixed the global layout, semantic content, and motion of the generated video. The role of Stage 2 is mainly local high-frequency refinement. Therefore, Stage 2 does not require per-sample text conditioning: the sample-specific information it needs is already carried by the Stage-1 latent it receives, and a generic, prompt-independent conditioning signal suffices.

Instead of encoding the user’s prompt dynamically for Stage 2, we evaluate a single canonical generic refinement prompt once and offline, 4K, refined, high quality, cinematic detail, clean textures, natural motion. This prompt is processed through the INT8 Gemma text encoder and the INT8 dev connector, and we retain only the compact post-connector video and audio context tensors to feed to Stage 2 as a cache. Consequently, neither the Stage-2 text encoder nor its connector is ever loaded during online inference, freeing the 16.2 GiB for other modules.

## 4.3. Weight quantization

For deployment on a single DGX Spark (GB10), quantization is essential to keeping the two-stage pipeline resident within the device’s 119.68 GiB CPU–GPU unified memory pool. The edge configuration pairs an FP8 H3 draft DiT [54] with an NVFP4 AWQ Qwen prompt encoder [36, 53, 54, 76], with reported resident footprints of 23.4 GiB and 14.6 GiB, respectively. The runtime reads each layer’s quantization configuration from the checkpoint, allowing different projections to use different precision settings. Qwen encodes each incoming prompt during online inference. The INT8 Gemma encoder and connector are used only offline to construct the cached Stage-2 conditioning described in Sec. 4.2, and neither remains resident during request processing.

## 5. Caching Reference Tokens for Reference to Video Generation

In many practical deployment scenarios, generation is conditioned on user-provided reference materials. In this referenceto-video and audio (ref2av) setting, the model synthesizes output video and audio based on a set of conditioning reference images or videos. As the volume of reference materials data grows, the interaction between reference tokens and generation tokens becomes a struggler of generation latency. Thus, we propose a reference token caching strategy to further reduce latency in this setting [16, 17, 18].

Reference computation redundancy. During the standard diffusion inference process, the model executes � denoising steps. In a naive implementation, every self-attention evaluation across these � steps recomputes the reference tokens in full, with no computations skipped. However, unlike the target video tokens which are iteratively denoised, the reference latent fed into the Diffusion Transformer (DiT) [3] is a clean latent. Because no noise is added to the reference inputs, their representations vary only weakly with the timestep embedding [16, 19]. Consequently, recomputing the full self-attention for the reference tokens at every denoising step introduces potential computational redundancy.

Generation with heavy conditioning. This redundancy becomes a severe bottleneck when processing extensive reference inputs. A single generation request may supply up to 9 reference images or 3 reference videos. Under these heavy conditioning workloads, the number of reference tokens can easily exceed the number of generation tokens themselves. At this scale, the redundant recomputation of reference tokens ceases to be a marginal overhead and instead dominates the overall inference cost, leading to a significant waste of computational resources.

Reference KV caching mechanism . To eliminate this inefficiency, we introduce a reference KV caching mechanism. The core principle is to evaluate the reference branch in full only during the first DiT step. During this initial step, the key (K) and value (V) tensors for the reference tokens are computed and cached. For all subsequent denoising steps, the reference tokens are no longer recomputed; instead, their cached KV tensors are retained and reused unchanged (the cached KV is never iterated or refreshed). The subsequent steps only evaluate the generated tokens, which attend to the cached reference KV. Because the first step remains a full computation, the conditioning signal is established exactly, and only the redundant repetitions in later steps are bypassed.

We note that this caching strategy is an approximation rather than a bitwise-identical transformation. Although the input reference latent is clean, its intermediate representations naturally drift slightly across steps due to the changing timestep embeddings in the DiT blocks. However, from the qualitative comparison, we find that the faithful recomputation pipeline and caching pipeline show minor deviation, demonstrating that the caching strategy is a practical approximation method for reference to video generation especially when reference material is heavy.

![](images/c01f735839ada944919bebd5811ca23935be83c4504a1d869f5077afaec6885f.jpg)

![](images/b666ca4a4513bed2d96dad03a92d799d2ec8c3fcb34eee191529c77ced938fae.jpg)  
Figure 4 | Attention computation with reference KV cache. Left: full self-attention over all tokens, the native inference mode of MiniMax-H3 [4]. Right: after the first denoising step, image and video reference query rows are skipped (gray). Text and generation queries attend to cached reference K,V (cross-attention) and to each other (self-attention).

## 6. Experiments

## 6.1. Cross-Resolution Two-Stage Acceleration

We evaluate the two-stage cross-resolution pipeline results measured on a single NVIDIA GB200. For a 5 s workload at 1344×768 resolution, the pipeline achieves an end-to-end latency of 6.852 s, comprising 4.173 s for Stage 1 and 2.679 s for Stage 2. This represents a 22.2× speedup over the SGLang [60] baseline of 152.3 s [35]. For a 10 s workload at the same 1344×768 resolution, the end-to-end latency is 14.931 s (9.820 s for Stage 1 and 5.111 s for Stage 2), yielding a 27.7× speedup over the SGLang baseline of 414.1 s.

For edge device, the single DGX Spark deployment [36] reaches 56.17 s end to end for the same 5 s 1344×768 workload, yielding a 31× speedup over the baseline of roughly 1740.81 s. The latency share breakdown for this deployment is as follows (Fig. 5): LTX 3-step refinement accounts for 44.0%, H3 4-step generation for 35.2%, LTX VAE decode for 12.5%, H3 upscaler and VAE adapter for 3.8%, Qwen prompt encoding for 2.4%. A qualitative comparison of the generated frames is provided in Fig. 6.

The same two-stage pipeline also runs on a single consumer GPU. On one RTX 5090 the 5 s 1344 × 768 request completes in 39.80 s, against a 1045.40 s full-resolution 49-step baseline measured on the same card, a 26.3× speedup. The latency composition differs from both datacenter and DGX Spark deployments: Qwen prompt encoding is the single largest block at 34.1%, ahead of H3 4-step generation at 28.2% and LTX 3-step refinement at 19.7%, with LTX VAE decode at 5.1% and the upscaler and latent adapter at 2.8%. The text encoder dominates not because encoding is expensive but because the encoder is rebuilt for every request: of the 13.568 s attributed to this block, the NVFP4 AWQ Qwen forward pass accounts for 0.518 s and the rest is loading and teardown [53]. Unlike a one-time startup cost, this is charged on each request, because the 5090 cannot hold the encoder in GPU HBM together with both generation stages and therefore runs under CPU offload. Denoising itself is therefore no longer the limiting term on consumer hardware; prompt-encoder residency is, which is the same pressure the Stage-2 conditioning cache in Sec. 4.2 relieves on DGX Spark.

Sol-H3 (ours) — 27.7×  
Sol-H3 (ours) — 42.6×  
H3 baseline (49 step), full resolution  
![](images/fecb0669d6a1ef69c6cd0986bbf4a4ef9d1bf18c7a44ba677b5e637117e85f1b.jpg)  
Figure 5 | Latency profiles of Sol-H3 inference. The same 5 s 1344 × 768 workload on one GB200 (top), one DGX Spark (middle) and one RTX 5090 (bottom), each against a full-resolution 49-step MiniMax-H3 baseline [35, 36]. The optimized request completes in 6.226 s on GB200, 56.17 s on Spark and 39.80 s on RTX 5090, a speedup of more than 20× in every case. The profiles differ in where the time goes: Stage-1 generation dominates on GB200 at 60.6%, Stage-2 refinement dominates on Spark at 44.0%, and on RTX 5090 prompt encoding is the largest single block at 34.1%, since Qwen is loaded and released per request under CPU offload.

1344 × 768, 5 s — A man and woman exchange a tense remark on a crowded Japanese street.  
![](images/31fe2c8790597d6bb9f1155e0314e894c913f6aba50f02faaea07a4759d328a7.jpg)  
Baseline, 50 steps

![](images/78ae1ff1975cbdf033460ddf58da0dcbaf11e3ce7dc59035f734f106399cf54c.jpg)

1344 × 768, 10 s — A man speaks to a handheld camera on a quiet suburban street.  
![](images/0ecaa8fa1097598525456b302a634d7489a441d538f9b2c719fd49ba79e3cf26.jpg)  
Baseline, 50 steps

![](images/dd25949ffbe1b101baea5270fdbb3f16cd84077ba4187707cbf4a1474669c344.jpg)

1080p, 5 s — Two martial artists face one another in a bamboo forest.  
![](images/a11331f2c94663556bc20f003b0eafbd97f30dd9a93cec6d3de20b537e3fc19b.jpg)  
Baseline, 50 steps

![](images/e9227f209b3a1ab4d077a57883a8d5871c0a2a1b0ab608c673ed9fc35bface97.jpg)  
Figure 6 | Qualitative comparison between Sol-H3 and the baseline. Sol-H3(right) drafts at a lower resolution and refines back to the target resolution. While preserving scene structure, subject identity and motion continuity with minor visual differences, Sol-H3 delivers a significant speedup of more than 20× [35].

## 6.2. Sol-H3 Latency on GB200 GPUs

We evaluate Sol-H3 on one, four, and eight NVIDIA GB200 GPUs [79, 80] at 1344×768 resolution and 24 FPS, generating reference-free video with stereo audio. All nine combinations of GPU count and output duration use the same prompt, seed 20260903, and merged four-step LoRA [11, 12, 14, 81]. The evaluated system combines SOL/BSA attention, quantized QKV and attention-output communication, MXFP8 GEMMs with fused activation producers and output reuse, and the VAE optimizations described in Sec. 3.

Table 2 | Additional cost introduced by reference tokens and acceleration of reference KV caching. Left: DiT time relative to reference-free generation against the effective reference-token count. The latency increases dramatically as the reference materials growing. Right: effective reference-token counts for every condition and the speedup obtained by reference KV caching strategy, which reducing the computation overhead up to 2×.  
![](images/02ea842c7c4e00051430acca89217a33e55e7bd07638ad2205cc7af2064b895a.jpg)

<table><tr><td>Condition</td><td>Eff. ref. tokens</td><td>KV cache speedup</td></tr><tr><td>No reference</td><td>0</td><td></td></tr><tr><td>1 image</td><td>14,344</td><td>1.14×</td></tr><tr><td>2 images</td><td>28,432</td><td>1.35×</td></tr><tr><td>3 images</td><td>42,776</td><td>1.56×</td></tr><tr><td>1 video</td><td>31,792</td><td>1.41×</td></tr><tr><td>2 videos</td><td>63,584</td><td>1.92×</td></tr><tr><td>1 video + 1 image</td><td>46,136</td><td>1.65×</td></tr></table>

Table 1 | Sol-H3 latency on GB200s. Baseline of MiniMax-H3 and Sol-H3 latency for 5, 10, and 15 s outputs on NVIDIA GB200 GPUs. It delivers more significant speedups with longer duration of generated videos.
<table><tr><td>GPUs</td><td colspan="3">5 s output</td><td colspan="3">10 s output</td><td colspan="3">15 s output</td></tr><tr><td></td><td>Base H3</td><td>Sol-H3</td><td>Speedup</td><td>Base H3</td><td>Sol-H3</td><td>Speedup</td><td>Base H3</td><td>Sol-H3</td><td>Speedup</td></tr><tr><td>1</td><td>129.898</td><td>9.475</td><td>13.71×</td><td>376.942</td><td>24.309</td><td>15.51×</td><td>746.885</td><td>44.958</td><td>16.61×</td></tr><tr><td>4</td><td>35.328</td><td>2.530</td><td>13.97×</td><td>100.440</td><td>6.212</td><td>16.17×</td><td>194.930</td><td>11.471</td><td>16.99×</td></tr><tr><td>8</td><td>18.250</td><td>1.434</td><td>12.73×</td><td>50.660</td><td>3.350</td><td>15.12×</td><td>99.513</td><td>6.062</td><td>16.41×</td></tr></table>

NVIDIA GB200, 1344×768, 24 FPS. Latency in seconds; speedup = Base H3 / Sol-H3. Base H3: 49 DiT forwards; Sol-H3: 4 DiT forwards.

Each Sol-H3 latency is the median of five calls after two warmups, with four denoising evaluations verified per call. Timing covers text encoding, denoising, video/audio VAE decoding, and synchronization, excluding model loading, compilation, warmup, and MP4 encoding. Single-GPU execution uses the same SOL/BSA policy and local QKV/output quantization without distributed collectives. It uses the compiled per-tile VAE decoder; multi-GPU execution uses the compiled, globally batched distributed decoder. The Base H3 reference uses 49 denoising evaluations.

As shown in Tab. 1, Sol-H3 achieves 12.73×–16.99× speedup over Base H3 across the evaluated workloads. On eight GB200 GPUs, five-, ten-, and fifteen-second videos take 1.434 s, 3.350 s, and 6.062 s, respectively. The five-second workload is generated approximately 3.5× faster than real-time playback.

## 6.3. Reference Conditioning and KV Caching

Reference conditioning substantially increases computation overhead. As shown in Tab. 2(left), one image adds 14,344 effective reference tokens and raises the DiT time to 1.71× compared with reference-free generation process; one video adds 31,792 tokens for 2.83×, and a video together with an image adds 46,136 tokens for 3.94×. Cost is governed by the reference-token count as the reference materials grows, consistent with the quadratic growth of attention in sequence length. Reference KV caching reduces repeated reference-branch computation, yielding speedups of 1.14×–1.92×, demonstrated in Tab. 2(right). The benefit grows with the number of references within each modality.

Although the caching strategy is an approximation method instead of a lossless system optimization, the qualitative comparison demonstrates that this approximation only introduces minor perceptual differences. As shown in Fig. 7, we can see that the generated samples with caching strategy preserve scene structure, subject identity and motion continuity against the full-compute baseline and strictly follow the instruction of video-to-video editing.

Reference input  
![](images/0bf375ef684fcd908c11028b8ff9152658095b9e748d8097fb9ff79f5be290c3.jpg)  
Reference input Red chameleon crawling on a

![](images/639f0cb4d990ce36491ad1858a6cb28f8a03e360698fd1125479dec210ff862c.jpg)  
Teacher (full compute)

![](images/237a22ba29e3ee94d46d1bcf64496091e72f2b805a1754cce2e989feb57a38f5.jpg)  
Cached reference KV — 1.512×

![](images/1103a1eb300a5a318ee00ab5a179190767777992a25943bfa7a85266918f35ee.jpg)

![](images/0d39822d5fe3d23e196e16df4a25cf53cc479a55c0e8b26eff337935549426bf.jpg)

![](images/c6573519d226eaf3bd541a4f04a38b31ad9c141d70c10ae8c9ef1e0e94217852.jpg)

![](images/9feb7290b6b663c35a8c8d7fbcb2d99d8bbe6e8f5b696daaff4ec717516a4eb9.jpg)  
Cat pouncing playfully without a toy

![](images/55e7ebf81d999a9b8e4d855859d59affef522fdbf58d8e37ad24eb2de6e05b26.jpg)  
Teacher (full compute)

Cached reference KV — 1.486×  
![](images/6b4e18c8140ce7d493e1d3dc95c4a80abc4489c479d8ddc668abfea1df5d7c0a.jpg)  
Cached reference KV — 1.416×  
Figure 7 | Reference KV cache: qualitative comparison. Each row displays a single frame from the input reference video on the left, two frames from the full-compute teacher baseline in the middle, and two frames from our cached reference KV generation on the right. It demonstrates that this approximation only introduces minor visual differences while preserving visual quality and instruction alignment

## 6.4. Comparison of Few-Step LoRA Adapters

We evaluate 120 text-to-video-and-audio (T2VA) tasks and 120 reference-to-video-and-audio (Ref2VA) tasks, spanning diverse subjects, visual styles, actions, and camera movements. Ref2VA includes single- and multiple-reference inputs, direct video editing, and transfer of identity, motion, or camera behavior. Candidates are grouped by task type and number of function evaluations (NFE), with four and six candidates for T2VA at 4 and 8 NFE, respectively, and two and three candidates for Ref2VA. Exhaustive within-group pairing for every task yields 3,000 pairwise comparisons: 720, 1,800, 120, and 360 in these four groups. In addition to the adapters discussed below, the T2VA pool includes FastH3 Preview v1 LoRA [82] at 4 NFE and the FastH3 V2 full distilled-checkpoint baseline [83] at 8 NFE. For each task, the compared runs use the frozen rewritten prompt, reference assets, and the same initial video/audio noise.

We combine human assessment on a subset of pairs with VLM assessment of the full 3,000-pair pool, keeping the two sources of judgments separate. Both protocols hide model identities, randomize the A/B presentation, and allow ties. Human raters can play the generated videos with audio and provide an overall preference. For the visual-only VLM assessment, Astra with high reasoning effort receives the generation instruction, 16 uniformly sampled frames from each candidate video, and the available visual reference material; reference videos are likewise sampled into 16 frames. No audio is supplied or scored, and audio-related prompt requirements are excluded from its judgment. Each pair receives one VLM judgment in a fresh context.

Our comparisons favor Larry’s Turbo LoRA (v4, step 600) [84] and LightX2V FL2VA Turbo (4-step v1.2) [85] for T2VA at 4 NFE, and Alibaba PAI PDD Acc8 [86, 87] and the official VDN-H3 stage-dmd-step-250 adapter [88] at 8 NFE. For Ref2VA, both Alibaba PAI’s 4-NFE PDD recipe [87] and LightX2V Ref2VA Turbo (v0.1) [85] perform well at 4 NFE, while the public HyperFlow v1.0 release [89] is our preferred choice at 8 NFE.

## 6.5. Cross-VAE Latent Translation

We evaluate the adapter separately from the two-stage generator to isolate conversion cost and reconstruction fidelity. The quality evaluation uses 256 held-out test videos, disjoint from training, development, and earlier evaluation cohorts. Both the predicted latent and the paired LTX teacher latent are decoded with the same frozen Conv Video VAE. We report mean per-video PSNR over the decoded clips and SSIM over 16 uniformly sampled frames. These metrics measure agreement with the LTX reconstruction of the source video, before any Stage-2 denoising.

Table 3 | Held-out latent translation quality. Both checkpoints use the same 194.76M-parameter architecture. Metrics compare decoded adapter predictions with decoded paired LTX teacher latents on the same 256 test clips; they do not include the refiner.
<table><tr><td>Checkpoint</td><td>Latent MSE↓</td><td>PSNR (dB) ↑</td><td>SSIM↑</td></tr><tr><td>Earlier decoder-aware checkpoint</td><td>0.11058</td><td>29.413</td><td>0.8763</td></tr><tr><td>Final checkpoint</td><td>0.09783</td><td>30.618</td><td>0.8962</td></tr></table>

As shown in Tab. 3, the final adapter reaches 30.618 dB PSNR and 0.8962 SSIM. Relative to the earlier checkpoint with the same architecture, the paired PSNR gain is 1.205 dB, with a 95% bootstrap interval of [1.166, 1.245] dB; all 256 clips improve in both PSNR and SSIM. The highest-motion quartile gains 1.366 dB, compared with 1.059 dB in the lowest-motion quartile. This improvement comes from continued decoder-aware training with a larger learning rate, batch size, and exposure budget, without increasing inference-time model size. It does not isolate the contribution of each training change.

Table 4 | Isolated H3-to-LTX conversion cost. One H100 80GB, BF16, batch one, 192 source frames at 1344 × 768. Latency is the median synchronized wall time after warmup; memory is peak allocated device memory for each conversion arm. The adapter includes geometry alignment but excludes the separate ×2 upscaler.
<table><tr><td>Conversion</td><td>Params (M)</td><td>Latency (ms)</td><td>Peak (GiB)</td><td>Speedup</td></tr><tr><td>Full VAE round trip</td><td>2,742.47</td><td>12,748.6</td><td>13.788</td><td>1.0×</td></tr><tr><td>Tiny AutoEncoder round trip</td><td>23.11</td><td>201.9</td><td>23.846</td><td>63.1×</td></tr><tr><td>Latent adapter</td><td>194.76</td><td>63.1</td><td>0.785</td><td>202.0×</td></tr></table>

For conversion cost, Tab. 4 profiles a batch-one, 192-frame clip at 1344 × 768 on one H100 80GB in BF16. The adapter, including its fixed geometry transform, takes 63.1 ms versus 12,748.6 ms for H3 decoding followed by LTX encoding, a 202.0× reduction in median conversion latency. Peak allocated memory falls from 13.79 to 0.785 GiB. A Tiny AutoEncoder round trip takes 201.9 ms under the same input geometry. This profile uses an earlier checkpoint of the same 194.76M-parameter architecture, whereas Tab. 3 evaluates the final trained weights. The measurements exclude spatial upsampling, denoising, final video decoding, and audio processing; they therefore quantify the cost of the handoff module, not an end-to-end speedup or a quality-equivalent replacement for the full VAE round trip.

## 6.6. Precision-Preserving LoRA Fusion

Numerical preservation and fusion cost. We first verify that consumer fusion preserves the selected adapter’s computation. With the evaluated eight-step adapter and BF16 linear compute, consumer fusion matches native separate branches byte for byte on three 15-second samples, across all eight denoising evaluations and four GPU ranks, including the final raw video and audio tensors. Both modes retain the same other acceleration settings, including sparse attention and low-precision attention transport. This establishes agreement for the tested inputs and software environment, rather than equivalence to an entirely unaccelerated model.

Tab. 5 compares the three execution modes on four GB200 GPUs for one 1344 × 768, 362-frame workload. Native and fused branches each use five interleaved warm measurements; merged weights use three warm measurements in a separate process on the same GPU allocation. Consumer fusion reduces median pipeline time by 0.694 s (2.64%), recovering 46.3% of the native branches’ additional time relative to merged weights. It remains 0.804 s (3.24%) slower than weight merging, which produces different numerical results. Timing includes text encoding, denoising, audio/video decoding, and synchronization, while excluding model loading, warmup, and MP4 encoding. The saving includes the fused FFN gate and residual operations; the LoRA GEMMs remain unchanged.

Table 5 | LoRA precision and execution cost. Median pipeline latency for eight denoising evaluations on four GB200 GPUs. Numerical agreement is relative to native separate branches under the same other acceleration settings. Merged weights provide a speed reference with different outputs.
<table><tr><td>LoRA execution</td><td>Time (s) ↓</td><td>Bitwise agreement</td></tr><tr><td>Merged BF16 weights</td><td>24.785</td><td>No</td></tr><tr><td>Native separate branches</td><td>26.283</td><td>Reference</td></tr><tr><td>Consumer fusion</td><td>25.589</td><td>Yes (tested inputs)</td></tr></table>

Qualitative effect of weight merging. We isolate the execution mode using a fixed eight-step adapter and a 15-second animated scene in which a girl meets a dinosaur that sneezes butterflies. The three arms in Fig. 8 share the prompt, seed, actual initial noise, conditioning, sampling schedule, and other runtime optimizations. Weight merging changes the dinosaur’s appearance, the girl’s hairstyle, their relative positions, and the timing of the butterfly burst. The magnified view also shows grid-like color patterns around the butterflies in the merged-weight output, visible in the decoded frames before video encoding. Native branches and consumer fusion preserve the same generated scene: the rerun verifies byte agreement at every denoising evaluation on all four ranks, in the final decoded video and audio, and in all 362 exported frames. This example illustrates how a numerical change can alter a generation trajectory; it is not a comparison between learned adapters or a quantitative perceptual-quality evaluation.

Merged BF16 weights 8 steps  
Native separate branches 8 steps  
Consumer fusion 8 steps  
![](images/d4b98abaeb37f6e3fc6d00adfbc4e3de940e1d8dafbdbf5d3a7e6fbaee8fd467.jpg)  
10.50 s  
11.33 s  
11.33 s  
Figure 8 | LoRA execution changes the generation trajectory. The same eight-step checkpoint, prompt, seed, and initial noise are evaluated on four GB200 GPUs at 1344 × 768, with the other acceleration settings fixed. The upper rows compare identical timestamps; the bottom row magnifies the butterfly burst at the indicated times. All images come from lossless frames exported before video encoding. Native branches and consumer fusion match byte for byte in the decoded video and audio; merging the weights changes the scene and introduces grid-like patterns around the butterflies in this example.

## 7. Discussion and Conclusion

In this report, we present Sol-H3, a full-stack efficient inference pipeline for the MiniMax-H3 video-audio generator. By combining a cross-resolution two-stage generation pipeline with a kernel-level optimization stack discovered via Recursive Self-Improvement (RSI), Sol-H3 achieves unprecedented generation speeds across both cloud and edge environments.

A key takeaway from the development of Sol-H3 is the complementary relationship between human architectural insight and automated optimization. In our work, the macro-level inference framework is fundamentally driven by human design, whereas the micro-level implementation optimizations are systematically executed by the RSI loop.

Specifically, RSI is deployed to handle execution bottlenecks where the optimization space is vast but the outcomes are strictly verifiable (e.g., maintaining mathematical correctness). Its primary contributions in this pipeline include:

• Communication collectives and quantization: Jointly searching the configuration space of data compression and network transfer to eliminate distributed inference bottlenecks.

• Fusion kernel implementation: Automatically discovering and generating optimal operator groupings to minimize redundant HBM traffic.

• Sparse attention implementation: Engineering highly efficient, hardware-aware attention mechanisms to manage massive sequence lengths.

• Precision-preserving LoRA integration: Ensuring that low-rank adaptation operations are accelerated without compromising numerical fidelity or generation quality.

In contrast, the open-ended, inherently lossy, and architectural components of the pipeline require a level of paradigmshifting intuition that RSI currently cannot provide. These human-designed components include:

• The cross-resolution two-stage framework: Conceptualizing the overarching algorithmic strategy to exploit the step-wise nature of diffusion models, balancing early global layout generation with later high-resolution refinement.

• The latent adapter: Recognizing the system-level trade-offs of standard VAE decoders/encoders, and designing a custom adapter to efficiently bridge the coarse-to-fine generation stages at the cost of strict scene-identity invariance.

Based on this dichotomy, we observe that RSI is exceptionally well-suited for systematically resolving deterministic, well-bounded engineering problems where success can be mathematically verified against a baseline. However, for openended, architectural pipeline designs that necessitate lossy approximations or holistic trade-offs, human engineering remains indispensable. Ultimately, this synergy of human algorithmic design and RSI-driven kernel optimization enables faster-than-real-time high-resolution video synthesis, drastically lowering the barrier to deploying massive video generation models in production.

## Appendix

## A. Latent Adapter Training and Evaluation

Paired data and geometry. The source corpus contains 80,000 videos stored in eight archives. Each source video is resized identically before encoding with the frozen H3 and LTX-2.5 Conv VAEs. We use deterministic posterior modes and the released per-channel normalization statistics. For a 192-frame clip at $1 3 4 4 \times 7 6 8$ , the H3 tensor has shape $2 4 \times 5 7 \times 4 8 \times$ 84 in channel–time–height–width order. LTX pads the input to 193 frames and produces $1 2 8 \times 2 5 \times 2 4 \times 4 2$ . H3 token positions follow the 17-frame chunking rule, with anchors $0 , 4 , 8 , 1 2 , 1 6 , 1 7 , 2 1 , . . .$ . and the configured removal of three trailing tokens. The fixed alignment in Sec. 3.1 yields $3 8 4 \times 2 5 \times 2 4 \times 4 2$ features, retaining each H3 token in one of three temporal slots before spatial pixel-unshuffle. The skip projection maps these features to 128 channels and remains fixed after ridge-regression initialization; the residual network is trained end to end.

Optimization. The final decoder-aware optimization starts from an earlier trained adapter and uses 65,536 cached video pairs at native resolution. Two phases of 2,500 and 5,000 updates use global batch size 32, for 240,000 additional sample presentations. Each phase uses AdamW with peak learning rate $8 \times 1 0 ^ { - 5 } , \beta = ( 0 . 9 , 0 . 9 5 )$ , zero weight decay, and gradient clipping at 1.0. The respective warmup lengths are 100 and 150 updates, followed by cosine decay to 5% of the peak learning rate; the second phase restarts the optimizer from the first phase’s weights. Training uses BF16 autocast and activation checkpointing through the residual stack and the frozen decoder. The objective in Eq. (1) is evaluated on the full spatial grid, with decoded-video supervision weight � = 160. Only the adapter’s residual path is updated; the affine skip and both VAEs remain fixed. The reported final checkpoint uses the trained weights directly, without EMA averaging.

Held-out evaluation. Sample identifiers are assigned to train, validation, and test partitions by a stable hash with a 98/1/1 split. The final development and test cohorts contain 256 videos each and exclude identifiers used in earlier evaluation cohorts. Checkpoint selection uses development data. In the final evaluator, predictions and teacher reconstructions are clamped to [0, 1] and compared over their common decoded temporal extent, including the VAE padding rather than an explicit crop back to 192 frames. PSNR is computed per video and then averaged across videos; SSIM uses 16 uniformly sampled decoded frames. Motion quartiles are defined by temporal differences in the teacher latents. Paired confidence intervals use 10,000 bootstrap resamples over the same clip identifiers. These reconstruction measurements do not evaluate prompt adherence, audio quality, or the output of subsequent LTX refinement.

Conversion benchmark. All three arms in Tab. 4 consume the same cached H3 latent and output normalized LTX latents at the same pixel resolution. The full round trip uses the released H3 decoder and tiled LTX Conv encoder. The Tiny AutoEncoder arm uses TAEH3 decoding and TAELTX encoding with their parallel execution mode. Both round-trip arms use one warmup and three measured iterations; the adapter uses three warmups and 20 measured iterations. CUDA synchronization brackets wall-time measurements, and model loading and disk I/O are excluded. The profiled adapter is an earlier checkpoint with the same geometry, width, depth, and parameter count as the final quality checkpoint. Peak allocation is specific to each implementation and excludes the co-resident generative pipeline; it is not an estimate of total Spark memory usage.

## B. LoRA Precision Diagnosis and Fusion

Separating weight rounding from GEMM precision. The merge diagnostic fixes the checkpoint, prompt, initial noise and sampling schedule, and compares the first denoising evaluation on four GB200 GPUs with the other runtime optimizations disabled. With linear computation held in FP32, merging into BF16 weights gives 35.26% relative RMSE in the video prediction against the original separate-branch baseline; retaining the merged weights in FP32 reduces this to 6.24%. FP32 GEMMs alone therefore do not recover updates already lost during weight storage. Changing the separate branches themselves to FP32 gives 6.08% relative RMSE against the same baseline, showing why higher precision is also insufficient to reproduce the original mixed-precision computation exactly. These are prediction discrepancies for one input at one denoising evaluation, not perceptual-quality scores. The six-projection counts in Sec. 3.5 use FP32 reconstructions of $W + B A$ and count nonzero entries of �� for which casting the sum to BF16 returns the original � entry.

Implementation scope. The released implementation<sup>1</sup> accepts the supported public PEFT LoRA [81] and FastVideo v2 hybrid formats. Hybrid .diff/.diff\_b corrections are applied to the base parameters with FP32 accumulation before conversion to the parameter dtype; low-rank factors remain separate. Consumer fusion requires inference mode, one active scale-1 adapter, ordinary BF16 linear layers, identity LoRA dropout, and no projection hooks. Unsupported installation configurations are rejected before consumer patches are installed; unsupported runtime states fall back to native PEFT. QKV consumer fusion uses the multi-GPU Ulysses path, while single-GPU dense Q/K/V projections and projections outside the main transformer blocks retain native execution.

Execution modes. The public runtime defaults to merged. The separate and fused modes retain LoRA branches and require compute-quant=none; the latter enables the consumer kernels. This disables quantized linear compute, not the independent attention or communication optimizations. The controlled comparison in Tab. 5 explicitly selects each mode; enabling consumer fusion is not implicit in every pipeline measurement. The released regression suite covers BF16 rounding boundaries, non-contiguous inputs, indexed gates, promoted-dtype fallback, BF16/INT8 QKV packing, and row addressing beyond 2<sup>31</sup> elements. Full-pipeline byte agreement remains scoped to the tested inputs and their recorded GPU and software configurations.

Qualitative case study. The additional rerun in Fig. 8 uses seed 42, eight denoising evaluations, four GB200 GPUs, and 362 frames at 1344 × 768 and 24 FPS. Each arm performs its natural random draws and verifies the resulting tensors and generator states against the same stored reference before replay; initial conditioning and the video/audio sampling grids are also checked. The native and fused arms match in every recorded prediction and in the complete decoded video and audio. We export all frames directly to lossless 8-bit RGB PNGs before MP4 encoding and compare their pixel hashes. The full-frame rows use common zero-based indices 180 and 300. The 480 × 320 detail crops instead follow the butterfly burst at the explicitly labeled times, because merging changes its timing as well as its appearance. These audited runs include tensor hashing and frame export and are separate from the warm latency measurements in Tab. 5.

## References

[1] Jonathan Ho, Ajay Jain, and Pieter Abbeel. Denoising diffusion probabilistic models. In Advances in Neural Information Processing Systems, 2020. arXiv:2006.11239.

[2] Robin Rombach, Andreas Blattmann, Dominik Lorenz, Patrick Esser, and Björn Ommer. High-resolution image synthesis with latent diffusion models. In IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2022. arXiv:2112.10752.

[3] William Peebles and Saining Xie. Scalable diffusion models with transformers. In IEEE/CVF International Conference on Computer Vision (ICCV), 2023. arXiv:2212.09748.

[4] MiniMax AI. MiniMax-H3: A 33b audio-video generation model, 2026. Model release, https://huggingface.co/ MiniMaxAI/MiniMax-H3.

[5] Weijie Kong, Qi Tian, Zijian Zhang, Rox Min, Zuozhuo Dai, Jin Zhou, Jiangfeng Xiong, Xin Li, Bo Wu, Jianwei Zhang, et al. Hunyuanvideo: A systematic framework for large video generative models. arXiv preprint arXiv:2412.03603, 2024.

[6] Tencent Hunyuan Foundation Model Team. Hunyuanvideo 1.5 technical report, 2025. URL https://arxiv.org/abs/ 2511.18870.

[7] Team Wan, Ang Wang, Baole Ai, Bin Wen, Chaojie Mao, Chen-Wei Xie, Di Chen, Feiwu Yu, Haiming Zhao, Jianxiao Yang, et al. Wan: Open and advanced large-scale video generative models. arXiv preprint arXiv:2503.20314, 2025.

[8] Zhuoyi Yang, Jiayan Teng, Wendi Zheng, Ming Ding, Shiyu Huang, Jiazheng Xu, Yuanming Yang, Wenyi Hong, Xiaohan Zhang, Guanyu Feng, Da Yin, Yuxuan Zhang, Weihan Wang, Yean Cheng, Bin Xu, Xiaotao Gu, Yuxiao Dong, and Jie Tang. Cogvideox: Text-to-video diffusion models with an expert transformer. In International Conference on Learning Representations, 2025.

[9] Lightricks. LTX-2.5: Audio-video diffusion with a learned latent upsampler, 2026. Model release.

[10] Yaron Lipman, Ricky T. Q. Chen, Heli Ben-Hamu, Maximilian Nickel, and Matt Le. Flow matching for generative modeling. In International Conference on Learning Representations, 2023. arXiv:2210.02747.

[11] Tim Salimans and Jonathan Ho. Progressive distillation for fast sampling of diffusion models. In International Conference on Learning Representations, 2022. arXiv:2202.00512.

[12] Yang Song, Prafulla Dhariwal, Mark Chen, and Ilya Sutskever. Consistency models. In International Conference on Machine Learning, 2023. arXiv:2303.01469.

[13] Simian Luo, Yiqin Tan, Longbo Huang, Jian Li, and Hang Zhao. Latent consistency models: Synthesizing high-resolution images with few-step inference. arXiv preprint arXiv:2310.04378, 2023.

[14] Tianwei Yin, Michaël Gharbi, Richard Zhang, Eli Shechtman, Frédo Durand, William T. Freeman, and Taesung Park. One-step diffusion with distribution matching distillation. In IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2024. arXiv:2311.18828.

[15] Axel Sauer, Dominik Lorenz, Andreas Blattmann, and Robin Rombach. Adversarial diffusion distillation. arXiv preprint arXiv:2311.17042, 2023.

[16] Feng Liu, Shiwei Zhang, Xiaofeng Wang, Yujie Wei, Haonan Qiu, Yuzhong Zhao, Yingya Zhang, Qixiang Ye, and Fang Wan. Timestep embedding tells: It’s time to cache for video diffusion model. arXiv preprint arXiv:2411.19108, 2024.

[17] Xin Zhou, Dingkang Liang, Kaijin Chen, Tianrui Feng, Xiwu Chen, Hongkai Lin, Yikang Ding, Feiyang Tan, Hengshuang Zhao, and Xiang Bai. Less is enough: Training-free video diffusion acceleration via runtime-adaptive caching. arXiv preprint arXiv:2507.02860, 2025.

[18] Xuanlei Zhao, Xiaolong Jin, Kai Wang, and Yang You. Real-time video generation with pyramid attention broadcast. arXiv preprint arXiv:2408.12588, 2024.

[19] Jiacheng Liu, Chang Zou, Yuanhuiyi Lyu, Junjie Chen, and Linfeng Zhang. From reusing to forecasting: Accelerating diffusion models with taylorseers. In Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV), pages 15853–15863, October 2025.

[20] DefTruth, vipshop.com, etc. Cache-dit: A pytorch-native inference engine with cache, parallelism and quantization for diffusion transformers. https://github.com/vipshop/cache-dit.git, 2025. Open-source software. Accessed June 20, 2026.

[21] Haocheng Xi, Shuo Yang, Yilong Zhao, Chenfeng Xu, Muyang Li, Xiuyu Li, Yujun Lin, Han Cai, Jintao Zhang, Dacheng Li, et al. Sparse videogen: Accelerating video diffusion transformers with spatial-temporal sparsity. arXiv preprint arXiv:2502.01776, 2025.

[22] Shuo Yang, Haocheng Xi, Yilong Zhao, Muyang Li, Jintao Zhang, Han Cai, Yujun Lin, Xiuyu Li, Chenfeng Xu, Kelly Peng, et al. Sparse videogen2: Accelerate video generation with sparse attention via semantic-aware permutation. arXiv preprint arXiv:2505.18875, 2025.

[23] Peiyuan Zhang, Yongqi Chen, Haofeng Huang, Will Lin, Zhengzhong Liu, Ion Stoica, Eric P. Xing, and Hao Zhang. Faster video diffusion with trainable sparse attention. In Advances in Neural Information Processing Systems, 2025.

[24] Peiyuan Zhang, Yongqi Chen, Runlong Su, Hangliang Ding, Ion Stoica, Zhengzhong Liu, and Hao Zhang. Fast video generation with sliding tile attention. In Proceedings ofthe 42nd International Conference on Machine Learning, volume 267 of Proceedings ofMachine Learning Research, pages 74714–74731, 2025. URL https://proceedings.mlr.press/ v267/zhang25m.html.

[25] Xingyang Li, Muyang Li, Tianle Cai, Haocheng Xi, Shuo Yang, Yujun Lin, Lvmin Zhang, Songlin Yang, Jinbo Hu, Kelly Peng, Maneesh Agrawala, Ion Stoica, Kurt Keutzer, and Song Han. Radial attention: �(� log �) sparse attention with energy decay for long video generation. arXiv preprint arXiv:2506.19852, 2025.

[26] Ruyi Xu, Guangxuan Xiao, Haofeng Huang, Junxian Guo, and Song Han. Xattention: Block sparse attention with antidiagona scoring. In Proceedings ofthe 42nd International Conference on Machine Learning (ICML), 2025.

[27] Jonathan Ho, Chitwan Saharia, William Chan, David J. Fleet, Mohammad Norouzi, and Tim Salimans. Cascaded diffusion models for high fidelity image generation. Journal of Machine Learning Research, 23(47):1–33, 2022.

[28] Dustin Podell, Zion English, Kyle Lacey, Andreas Blattmann, Tim Dockhorn, Jonas Müller, Joe Penna, and Robin Rombach. SDXL: Improving latent diffusion models for high-resolution image synthesis. In International Conference on Learning Representations (ICLR), 2024.

[29] Yang Jin, Zhicheng Sun, Ningyuan Li, Kun Xu, Kun Xu, Hao Jiang, Nan Zhuang, Quzhe Huang, Yang Song, Yadong Mu, and Zhouchen Lin. Pyramidal flow matching for efficient video generative modeling. arXiv preprint arXiv:2410.05954, 2024.

[30] NVIDIA. CUTLASS: Efficient GEMM and epilogue fusion. https://github.com/NVIDIA/cutlass/blob/main/ media/docs/cpp/efficient\_gemm.md, 2025. Official documentation. Accessed June 20, 2026.

[31] Yujia Zhai, Chengquan Jiang, Leyuan Wang, Xiaoying Jia, Shang Zhang, Zizhong Chen, Xin Liu, and Yibo Zhu. Bytetransformer: A high-performance transformer boosted for variable-length inputs. In 2023 IEEE International Parallel and Distributed Processing Symposium (IPDPS), pages 344–355, 2023.

[32] Tri Dao, Daniel Y. Fu, Stefano Ermon, Atri Rudra, and Christopher Ré. FlashAttention: Fast and memory-efficient exact attention with IO-awareness. In Advances in Neural Information Processing Systems, 2022. arXiv:2205.14135.

[33] Han Guo, Jack Zhang, Arjun Menon, Driss Guessous, Vijay Thakkar, Yoon Kim, and Tri Dao. Coda: Rewriting transformer blocks as gemm-epilogue programs. arXiv preprint arXiv:2605.19269, 2026.

[34] Yitong Li, Junsong Chen, Haopeng Li, Haozhe Liu, Jincheng Yu, Ligeng Zhu, Ping Luo, Song Han, and Enze Xie. Sol video inference engine: Agent-native full-stack acceleration framework for efficient video generation. arXiv preprint arXiv:2606.23743, 2026.

[35] NVIDIA SANA Team. MiniMax-H3 super acceleration: Fast draft generation and high-resolution refinement. https: //nvlabs.github.io/Sana/Sol-Engine/H3-Super-Acceleration/, 2026. Project release.

[36] Sol-H3 Team. Sol-H3: Speed-of-light MiniMax-H3 on single NVIDIA DGX Spark blackwell superchip. https:// nvlabs.github.io/Sana/Sol-Engine/Sol-H3-Spark/, 2026. Single-Spark project release.

[37] Wenyi Hong, Ming Ding, Wendi Zheng, Xinghan Liu, and Jie Tang. Cogvideo: Large-scale pretraining for text-to-video generation via transformers. arXiv preprint arXiv:2205.15868, 2022.

[38] Meituan LongCat Team, Xunliang Cai, Qilong Huang, Zhuoliang Kang, Hongyu Li, Shijun Liang, Liya Ma, Siyu Ren, Xiaoming Wei, Rixu Xie, and Tong Zhang. Longcat-video technical report, 2025. URL https://arxiv.org/abs/2510.22200.

[39] Junsong Chen, Yuyang Zhao, Jincheng Yu, Ruihang Chu, Junyu Chen, Shuai Yang, Xianbang Wang, Yicheng Pan, Daquan Zhou, Huan Ling, et al. SANA-Video: Efficient video generation with block linear diffusion transformer, 2025. URL https://arxiv.org/abs/2509.24695.

[40] NVIDIA. Cosmos 3: Omnimodal world models for physical ai. arXiv preprint arXiv:2606.02800, 2026. URL https: //arxiv.org/abs/2606.02800.

[41] Echo Team @ Joy Future Academy, JD. Joyai-echo: Pushing the frontier of long video generation. Technical report, Joy Future Academy, JD, May 2026. URL https://echo-team-joy-future-academy-jd.github.io/Echo-LongVideo-Page/. Project page. Accessed June 20, 2026.

[42] Lightricks. LTX-2.3 Model Card. https://huggingface.co/Lightricks/LTX-2.3, 2026. Model checkpoint famil including ltx-2.3-22b-dev and distilled variants. Accessed June 20, 2026.

[43] Jintao Zhang, Chendong Xiang, Haofeng Huang, Jia Wei, Haocheng Xi, Jun Zhu, and Jianfei Chen. Spargeattn: Accurate sparse attention accelerating any model inference. In International Conference on Machine Learning (ICML), 2025.

[44] Xuan Shen, Chenxia Han, Yufa Zhou, Yanyue Xie, Yifan Gong, Quanyi Wang, Yiwei Wang, Yanzhi Wang, Pu Zhao, and Jiuxiang Gu. Draftattention: Fast video diffusion via low-resolution attention guidance. arXiv preprint arXiv:2505.14708, 2025.

[45] Haopeng Li, Shitong Shao, Wenliang Zhong, Zikai Zhou, Lichen Bai, Hui Xiong, and Zeke Xie. Pisa: Piecewise sparse attention is wiser for efficient diffusion transformers. arXiv preprint arXiv:2602.01077, 2026.

[46] Jintao Zhang, Jia Wei, Haofeng Huang, Pengle Zhang, Jun Zhu, and Jianfei Chen. Sageattention: Accurate 8-bit attention for plug-and-play inference acceleration. In International Conference on Learning Representations, 2025.

[47] Jintao Zhang, Haofeng Huang, Pengle Zhang, Jia Wei, Jun Zhu, and Jianfei Chen. Sageattention2: Efficient attention with thorough outlier smoothing and per-thread int4 quantization. In Proceedings ofthe 42nd International Conference on Machine Learning, volume 267 of Proceedings ofMachine Learning Research, pages 75097–75119, 2025.

[48] Jintao Zhang, Jia Wei, Haoxu Wang, Pengle Zhang, Xiaoming Xu, Haofeng Huang, Kai Jiang, Jun Zhu, and Jianfei Chen. Sageattention3: Microscaling fp4 attention for inference and an exploration of 8-bit training. arXiv preprint arXiv:2505.11594, 2025.

[49] Tianchen Zhao, Tongcheng Fang, Haofeng Huang, Rui Wan, Widyadewi Soedarmadji, Enshu Liu, Shiyao Li, Zinan Lin, Guohao Dai, Shengen Yan, Huazhong Yang, Xuefei Ning, and Yu Wang. Vidit-q: Efficient and accurate quantization of diffusion transformers for image and video generation. In International Conference on Learning Representations, 2025.

[50] Junyi Wu, Haoxuan Wang, Yuzhang Shang, Mubarak Shah, and Yan Yan. Ptq4dit: Post-training quantization for diffusion transformers. In Advances in Neural Information Processing Systems, 2024.

[51] Lei Chen, Yuan Meng, Chen Tang, Xinzhu Ma, Jingyan Jiang, Xin Wang, Zhi Wang, and Wenwu Zhu. Q-dit: Accurate post-training quantization for diffusion transformers. In Proceedings ofthe IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 28306–28315, 2025.

[52] Muyang Li, Yujun Lin, Zhekai Zhang, Tianle Cai, Xiuyu Li, Junxian Guo, Enze Xie, Chenlin Meng, Jun-Yan Zhu, and Song Han. Svdquant: Absorbing outliers by low-rank component for 4-bit diffusion models. In International Conference on Learning Representations, 2025.

[53] Ji Lin, Jiaming Tang, Haotian Tang, Shang Yang, Wei-Ming Chen, Wei-Chen Wang, Guangxuan Xiao, Xingyu Dang, Chuang Gan, and Song Han. AWQ: Activation-aware weight quantization for on-device LLM compression and acceleration. In Proceedings ofMachine Learning and Systems (MLSys), 2024.

[54] Open Compute Project. OCP microscaling formats (MX) specification. Open Compute Project specification, v1.0, 2023.

[55] Daniel Bolya and Judy Hoffman. Token merging for fast stable diffusion. CVPR Workshop on Efficient Deep Learningfor Computer Vision, 2023.

[56] Haosong Liu, Yuge Cheng, Wenxuan Miao, Zihan Liu, Aiyue Chen, Jing Lin, Yiwu Yao, Chen Chen, Jingwen Leng, Yu Feng, and Minyi Guo. Astraea: A token-wise acceleration framework for video diffusion transformers. arXiv preprint arXiv:2506.05096, 2025.

[57] Sheng Li, Yang Sui, Junhao Ran, Bo Yuan, Yue Dai, and Xulong Tang. Temporal aware pruning for efficient diffusion-based video generation. arXiv preprint arXiv:2605.17837, 2026.

[58] Zhuojin Li, Hsin-Pai Cheng, Hong Cai, Shizhong Han, and Fatih Porikli. Coredit: Spatial coherence-guided token pruning and reconstruction for efficient diffusion transformers. arXiv preprint arXiv:2605.14191, 2026.

[59] Woosuk Kwon, Zhuohan Li, Siyuan Zhuang, Ying Sheng, Lianmin Zheng, Cody Hao Yu, Joseph E. Gonzalez, Hao Zhang, and Ion Stoica. Efficient memory management for large language model serving with pagedattention. In Proceedings of the 29th Symposium on Operating Systems Principles, pages 611–626, 2023.

[60] Lianmin Zheng, Liangsheng Yin, Zhiqiang Xie, Chuyue Sun, Jeff Huang, Cody Hao Yu, Shiyi Cao, Christos Kozyrakis, Ion Stoica, Joseph Gonzalez, Clark Barrett, and Ying Sheng. Sglang: Efficient execution of structured language model programs. In Advances in Neural Information Processing Systems, 2024.

[61] Sam Ade Jacobs, Masahiro Tanaka, Chengming Zhang, Minjia Zhang, Shuaiwen Leon Song, Samyam Rajbhandari, and Yuxiong He. DeepSpeed Ulysses: System optimizations for enabling training of extreme long sequence transformer models. arXiv preprint arXiv:2309.14509, 2023.

[62] Yukang Chen, Luozhou Wang, Wei Huang, Shuai Yang, Bohan Zhang, Yicheng Xiao, Ruihang Chu, Weian Mao, Qixin Hu, Shaoteng Liu, Yuyang Zhao, Huizi Mao, Ying-Cong Chen, Enze Xie, Xiaojuan Qi, and Song Han. LongLive-2.0: An nvfp4 parallel infrastructure for long video generation, 2026. URL https://arxiv.org/abs/2605.18739.

[63] John Yang, Carlos E. Jimenez, Alexander Wettig, Kilian Lieret, Shunyu Yao, Karthik R. Narasimhan, and Ofir Press. Swe-agent: Agent-computer interfaces enable automated software engineering. In Advances in Neural Information Processing Systems, 2024.

[64] Yuntong Zhang, Haifeng Ruan, Zhiyu Fan, and Abhik Roychoudhury. Autocoderover: Autonomous program improvement. arXiv preprint arXiv:2404.05427, 2024.

[65] Chunqiu Steven Xia, Yinlin Deng, Soren Dunn, and Lingming Zhang. Agentless: Demystifying llm-based software engineering agents. arXiv preprint arXiv:2407.01489, 2024.

[66] Xingyao Wang, Boxuan Li, Yufan Song, Frank F. Xu, Xiangru Tang, Mingchen Zhuge, Jiayi Pan, Yueqi Song, Bowen Li, Jaskirat Singh, Hoang H. Tran, Fuqiang Li, Ren Ma, Mingzhang Zheng, Bill Qian, Yanjun Shao, Niklas Muennighoff, Yizhe Zhang, Binyuan Hui, Junyang Lin, Robert Brennan, Hao Peng, Heng Ji, and Graham Neubig. Openhands: An open platform for ai software developers as generalist agents. arXiv preprint arXiv:2407.16741, 2024.

[67] Xiao Liu, Hao Yu, Hanchen Zhang, Yifan Xu, Xuanyu Lei, Hanyu Lai, Yu Gu, Hangliang Ding, Kaiwen Men, Kejuan Yang, Shudan Zhang, Xiang Deng, Aohan Zeng, Zhengxiao Du, Chenhui Zhang, Sheng Shen, Tianjun Zhang, Yu Su, Huan Sun, Minlie Huang, Yuxiao Dong, and Jie Tang. Agentbench: Evaluating llms as agents. In International Conference on Learning Representations, 2024.

[68] Qian Huang, Jian Vora, Percy Liang, and Jure Leskovec. Mlagentbench: Evaluating language agents on machine learning experimentation. In Proceedings ofthe 41st International Conference on Machine Learning, volume 235 of Proceedings of Machine Learning Research, pages 20271–20309, 2024.

[69] Chris Lu, Cong Lu, Robert Tjarko Lange, Jakob Foerster, Jeff Clune, and David Ha. The ai scientist: Towards fully automated open-ended scientific discovery. arXiv preprint arXiv:2408.06292, 2024.

[70] Hailin Zhong and Shengxin Zhu. Ai harness engineering: A runtime substrate for foundation-model software agents. arXiv preprint arXiv:2605.13357, 2026.

[71] Anne Ouyang, Simon Guo, Simran Arora, Alex L. Zhang, William Hu, Christopher Ré, and Azalia Mirhoseini. Kernelbench: Can llms write efficient gpu kernels? arXiv preprint arXiv:2502.10517, 2025.

[72] Wentao Chen, Jiace Zhu, Qi Fan, Yehan Ma, and An Zou. Cuda-llm: Llms can write efficient cuda kernels. arXiv preprint arXiv:2506.09092, 2025.

[73] Zijian Zhang, Rong Wang, Shiyang Li, Yuebo Luo, Mingyi Hong, and Caiwen Ding. Cudaforge: An agent framework with hardware feedback for cuda kernel optimization. arXiv preprint arXiv:2511.01884, 2025.

[74] Genghan Zhang, Shaowei Zhu, Anjiang Wei, Zhenyu Song, Allen Nie, Zhen Jia, Nandita Vijaykumar, Yida Wang, and Kunle Olukotun. Accelopt: A self-improving llm agentic system for ai accelerator kernel optimization. arXiv preprint arXiv:2511.15915. 2025.

[75] Alexander Novikov et al. Alphaevolve: A coding agent for scientific and algorithmic discovery. arXiv preprint arXiv:2506.13131, 2025.

[76] Yitong Li, Junsong Chen, Shuchen Xue, Pengcuo Zeren, Siyuan Fu, Dinghao Yang, Yangyang Tang, Junjie Bai, Ping Luo, Song Han, and Enze Xie. Fp4 explore, bf16 train: Diffusion reinforcement learning via efficient rollout scaling, 2026. URL https://arxiv.org/abs/2604.06916.

[77] Tri Dao. FlashAttention-2: Faster attention with better parallelism and work partitioning. In International Conference on Learning Representations, 2024. arXiv:2307.08691.

[78] Haopeng Li, Yitong Li, Junsong Chen, Tian Ye, Haozhe Liu, Jincheng Yu, Duomin Wang, Ruihua Zhang, Zeke Xie, Enze Xie, and Song Han. Sol-attn: Accelerating video generation inference via on-the-fly attention sparsification. arXiv preprint arXiv:2607.24027, 2026.

[79] NVIDIA Research, Efficient AI Team. Sol-H3: Speed-of-light MiniMax-H3 on an 8× NVIDIA GB200 blackwell system. https://nvlabs.github.io/Sana/Sol-Engine/Sol-H3/, 2026. Project release.

[80] NVIDIA. Components — nvidia hgx ai factory. https://docs.nvidia.com/enterprise-referencearchitectures/hgx-ai-factory/latest/components.html, 2026. Accessed June 20, 2026.

[81] Edward J. Hu, Yelong Shen, Phillip Wallis, Zeyuan Allen-Zhu, Yuanzhi Li, Shean Wang, Lu Wang, and Weizhu Chen. LoRA: Low-rank adaptation of large language models. In International Conference on Learning Representations, 2022. arXiv:2106.09685.

[82] FastVideo Team. FastVideo-FastH3-4-step-Preview-v1-LoRA, 2026. URL https://huggingface.co/FastVideo/ FastVideo-FastH3-4-step-Preview-v1-LoRA.

[83] FastVideo Team. FastVideo-FastH3-8-Step-V2, 2026. URL https://huggingface.co/FastVideo/FastVideo-FastH3-8-Step-V2.

[84] larryvrh. MiniMax-H3 Turbo LoRA: Few-step audio-video generation, 2026. URL https://huggingface.co/ larryvrh/MiniMax-H3-Turbo-Lora.

[85] LightX2V Team. MiniMax-H3 Turbo, 2026. URL https://huggingface.co/lightx2v/Minimax-h3-Turbo.

[86] Neta Shaul, Chao Liu, Arash Vahdat, and Julius Berner. Parallel decoding distillation for fast image and video generation. arXiv preprint arXiv:2607.26004, 2026.

[87] Alibaba PAI. MiniMax-H3-Acc-LoRAs, 2026. URL https://huggingface.co/alibaba-pai/MiniMax-H3-Acc LoRAs.

[88] Haocheng Xi, Yiming Xie, Hexu Zhao, Yiwen Zhang, Michael Liu, Thomas Creavin, Kurt Keutzer, Xiuyu Li, Zhaoyang Lv, Chenfeng Xu, and Haiwen Feng. Video DeltaNet: A video-native hybrid attention for livestream video generation. arXiv preprint arXiv:2609.20744, 2026. URL https://huggingface.co/OpenVDN/vdn-minimax-h3.

[89] Video Rebirth. HyperFlow: Eight-step MiniMax-H3 generation, 2026. URL https://huggingface.co/ videorebirth/hyperflow.
# ROLLOUT-MARGINAL DISTILLATION FOR LONG-HORIZON AUTOREGRESSIVE VIDEO GENERATION

Chenjian Gao<sup>1,2∗</sup>, Zhihao Hu<sup>2∗</sup>, Jianqi Ma<sup>2</sup>, Jun Zhang<sup>2†</sup>, Weidong Zhang<sup>2</sup>, Tianfan Xue<sup>1†</sup>

<sup>1</sup>MMLab, The Chinese University of Hong Kong <sup>2</sup>Tencent AIPD <sup>∗</sup>Equal contribution <sup>†</sup>Corresponding authors

## ABSTRACT

Autoregressive (AR) video diffusion enables low-latency, streamable video generation, but prediction errors often accumulate over long rollouts. Training the generator on its own rollouts exposes it to these imperfect histories. However, existing video-level distribution matching distillation (DMD) scores the whole rollout jointly. Because a chunk is evaluated together with its past and future, its correction can favor matching artifacts in the surrounding context merely to preserve temporal consistency. To provide a clearer visual-quality signal, we introduce Rollout-Marginal Distillation (RMD). RMD retains the generated history for AR prediction but scores each chunk independently against a chunk teacher, ensuring its quality correction is not compromised by an imperfect temporal context. To compensate for the lack of temporal context in independent chunk scoring, RMD subsequently applies video-level DMD to restore temporal coherence. Extensive experiments demonstrate that RMD maintains high visual quality far beyond its training horizon and outperforms video-level DMD baselines. Code and video results are available at https://cjeen.github.io/RMD/.

![](images/7b9b824732d7ae17399d8bc2547a956431c9efaee3ecf2e503c7387eabb0776d.jpg)  
training horizon

generation beyond training

Figure 1: Long-horizon autoregressive video generation. Trained on 5-second rollouts, RMD maintains high visual quality over 500-second rollouts (100× the training horizon). Notably, generation relies solely on a fully sliding context window, without fixed anchors, additional explicit memory mechanisms, or modifications to the generator architecture.

time

![](images/2f372b1c7818029b702412d40065b69bfd85a92c254ddef0d39ef9282acbff1c.jpg)  
(c) real-score prediction under RMD

Figure 2: Comparison of real-score predictions under video-level DMD and RMD. (a) The autoregressive rollout accumulates visual artifacts. (b) Video-level DMD scores the sequence jointly, causing the real-score prediction to retain these historical artifacts. (c) RMD scores the current chunk independently, providing a clean target regardless of the degraded context.

## 1 INTRODUCTION

Due to high computational and memory costs, video diffusion models are typically trained on short clips (Wan et al., 2025; Huang et al., 2026a). However, many real-world applications demand much longer sequences, including autonomous-driving simulation, interactive and embodied world models, and physics-grounded visual simulation (Zhang et al., 2025; Bruce et al., 2024; Liu et al., 2024). Autoregressive (AR) models naturally support this by extending video generation chunk by chunk (Yin et al., 2025; Huang et al., 2026a). Specifically, the diffusion process for each new chunk conditions on previously generated frames. As the video grows, a sliding context window ensures the model only attends to the most recent history, keeping the generation cost bounded. However, this sequential process introduces an inherent challenge: each generated chunk inevitably contains minor prediction errors, which then become the context for future steps. Consequently, errors barely noticeable in a short sequence can compound and severely degrade visual quality over a long rollout.

To mitigate error accumulation, self-rollout training exposes the generator to its own imperfect predictions (Huang et al., 2026a; Zhu et al., 2026). Since a generated video rarely matches real videos perfectly, no paired ground-truth continuation exists to supervise the next steps. To overcome this lack of paired data, distribution matching distillation (DMD) is often employed (Yin et al., 2024b; 2025). Instead of requiring exact continuations, a bidirectional diffusion teacher evaluates the entire sequence jointly. It provides corrective feedback by matching a fake score tracking the student’s output against a real score representing the teacher’s high-quality prior.

However, DMD-trained models still struggle to generate very long sequences. We attribute this to a bias introduced by the video teacher’s joint supervision. As visual artifacts accumulate over a long autoregressive rollout (e.g., the severe texture and color corruption in the background foliage in Figure 2(a)), the teacher must evaluate the new chunk alongside this flawed history. Because the teacher cannot modify past frames, enforcing a clean, natural background in the current chunk would cause an abrupt visual shift. Consequently, to preserve temporal consistency, the video-level teacher often retains these accumulated errors, which manifest as non-physical ghosting artifacts, in its real-score prediction (Figure 2(b)). The temporal consistency constraint thus fundamentally limits the teacher’s ability to correct local errors.

We break this coupling with Rollout-Marginal Distillation (RMD). The generator still uses its history to produce continuations, but the teacher evaluates the generated chunk independently, without checking the history or future chunks. As shown in Figure 2(c), context-free scoring ensures the teacher provides a clean, high-fidelity real score despite the degraded history. RMD achieves this by matching the student’s chunk marginal distribution to a teacher-defined chunk distribution. To provide this supervision, we prepare the RMD teacher by adapting a video diffusion model on individual chunks extracted from videos generated by the original bidirectional model. However, because in dependent chunk supervision does not enforce cross-chunk relationships, generated videos might exhibit motion jitter between chunks. To resolve this, we subsequently refine the generator with a video-level DMD stage, explicitly smoothing these transitions to promote temporal coherence.

Extensive experiments demonstrate that RMD achieves significantly higher visual quality than video-level DMD baselines beyond the training horizon. Our quantitative evaluation covers rollouts of approximately 60 seconds (12× the 5-second training horizon). In addition, Figure 1 provides a qualitative example extending to 500 seconds (100× the training horizon). Generation relies entirely on a fully sliding context window, without fixed anchors such as frame sinks (Yang et al., 2026; Li et al., 2026), additional explicit memory mechanisms, or changes to the generator architecture.

## 2 RELATED WORK

Few-step autoregressive video generation. Causal video diffusion generates frames or blocks sequentially with KV caching for low-latency streaming. CausVid converts a pretrained bidirectional video diffusion model into a causal generator and extends DMD to few-step video generation (Yin et al., 2025). Self-Forcing further performs autoregressive rollout during training, feeding generated outputs back as context and supervising the sequence with a holistic video objective (Huang et al., 2026a). Causal Forcing instead studies initialization and uses an autoregressive teacher for ODE distillation before applying Self-Forcing DMD (Zhu et al., 2026). Causal-rCM similarly combines teacher-forced consistency training with self-forced DMD refinement (Zheng et al., 2026a). Solaris introduces Checkpointed Self Forcing, which decouples serial self-rollout from parallel differentiable replay to reduce training memory (Savva et al., 2026). Together, these methods establish causal students and on-policy training pipelines. We adopt the Solaris replay mechanism directly and focus RMD on the distribution matched during on-policy DMD.

Distribution matching distillation. DMD trains a few-step generator by minimizing a reverse KL divergence to the distribution of a pretrained diffusion model, using the difference between real and fake scores as generator supervision (Yin et al., 2024b). DMD2 removes the costly regression objective and improves stability through a two-time-scale update rule, adversarial learning, and multi-step training (Yin et al., 2024a). Subsequent work has combined distribution matching with adversarial learning (Lu et al., 2025), replaced KL-based matching with an optimal-transport objective (Wang et al., 2026), or coupled reverse-KL refinement with score-regularized consistency training (Zheng et al., 2026b). These developments primarily modify the divergence, regularization, or optimization of distribution matching. By contrast, RMD retains the DMD score-difference update but changes its output space from a joint rollout to the chunk marginal induced by autoregressive sampling.

Long rollouts and causal supervision. LongLive improves extended AR generation through streaming long-video tuning, a finite attention window, and a persistent frame sink (Yang et al., 2026). Rolling Sink studies cache maintenance beyond the training horizon and introduces a training-free cache-maintenance strategy (Li et al., 2026). More directly related to training-time supervision, OPSD-V post-trains few-step AR generators on their inference-time trajectories while supplying the teacher with a cleaner cache that can incorporate real long-video context (Liu et al., 2026). Most closely related to our analysis, Context-Matched Distillation (CMD) observes that a bidirectional full-clip teacher can use future information unavailable to a causal student (Bandyopadhyay et al., 2026). CMD addresses this issue with a causal teacher and prefix scoring under the student’s realized rollout context. RMD takes a different route. It does not construct a teacher conditional for each self-rollout history. Instead, it uses a bidirectional diffusion prior as a chunkdistribution target, scores the rollout chunks independently, and subsequently restores joint temporal structure through video-level refinement. It therefore requires neither paired continuations for generated histories nor real long-video context.

## 3 ROLLOUT-MARGINAL DISTILLATION

In this section, we present Rollout-Marginal Distillation (RMD), an approach designed to disentangle visual quality supervision from temporal context in autoregressive video generation. We first review video-level distribution matching distillation (DMD) and formalize how its joint scoring mechanism inherently couples the current chunk’s gradient to past and future frames (Section 3.1). To break this coupling, we introduce our rollout-marginal objective, which independently penalizes unrealistic chunks without conditioning on the imperfect context (Section 3.2). We then detail the training procedure in practice (Section 3.3). Finally, because independent scoring ignores crosschunk relationships, we describe a subsequent video-level refinement stage that explicitly restores temporal coherence (Section 3.4).

## 3.1 REVISITING VIDEO-LEVEL DISTRIBUTION MATCHING

To understand the necessity of rollout-marginal supervision, we first examine how video-level DMD improves a causal video generator and why its joint scoring mechanism introduces a critical flaw.

Given a text prompt $c ,$ let the causal generator $G _ { \theta }$ produce a latent rollout ${ \bf x } = ( x _ { 1 } , \dots , x _ { T } )$ , where each $x _ { i }$ is a chunk. The distribution of the generated video x can be naturally factorized as:

$$
q _ { \theta } ( \mathbf { x } \mid c ) = \prod _ { i = 1 } ^ { T } q _ { \theta } ( x _ { i } \mid x _ { < i } , c ) .\tag{1}
$$

Following the self-rollout framework (Huang et al., 2026a; Zhu et al., 2026), the generator conditions each continuation on its own generated history $x _ { < i } ,$ exposing it to the prediction errors it will encounter at inference. Because there are no ground-truth continuations paired with these imperfect generated histories, video-level distribution matching distillation (DMD) is employed to supervise the entire student rollout jointly (Yin et al., 2024b;a; 2025).

At noise level $\tau ,$ let ${ \bf x } _ { \tau } = \alpha _ { \tau } { \bf x } + \sigma _ { \tau } \epsilon$ denote a noised rollout. The video-level objective minimizes the reverse KL divergence between the noised student video distribution $q _ { \theta , \ast }$ <sub>,τ</sub> and a pretrained teacher video distribution $p _ { \tau }$

$$
\begin{array} { r } { \mathcal { L } _ { \mathrm { v i d e o } } ( \theta ) = \mathbb { E } _ { c , \tau } \Big [ D _ { \mathrm { K L } } \big ( q _ { \theta , \tau } \mid \mid p _ { \tau } \big ) \Big ] . } \end{array}\tag{2}
$$

The generator is updated using the difference between two diffusion scores:

$$
\nabla _ { \theta } \mathcal { L } _ { \mathrm { v i d e o } } \approx \mathbb { E } \Big [ w ( \tau ) J _ { \theta } ( \mathbf { x } ) ^ { \top } \big ( s _ { \mathrm { f a k e } } ( \mathbf { x } _ { \tau } , \tau , c ) - s _ { \mathrm { r e a l } } ( \mathbf { x } _ { \tau } , \tau , c ) \big ) \Big ] ,\tag{3}
$$

where $J _ { \theta } ( { \bf x } ) = \partial { \bf x } / \partial \theta$ is the generator Jacobian, $v ( \tau )$ is a noise-level weight, $s _ { \mathrm { r e a l } }$ is a frozen teacher score representing the target distribution, and $s _ { \mathrm { f a k e } }$ is an online score model trained to track the student’s distribution.

The Coupling Dilemma. Equation 3 reveals a fundamental issue in applying video-level DMD to autoregressive rollouts. Both the real and fake score networks $( s _ { \mathrm { r e a l } }$ and $s _ { \mathrm { f a k e } } )$ take the entire sequence $\mathbf { x } _ { \tau }$ as input. Consequently, the correction signal assigned to any specific chunk $x _ { i }$ is coupled with its generated past $( x _ { < i } )$ and future $( x _ { > i } )$ . If an artifact appears in the generated history, a joint score evaluation will likely penalize a sudden correction to natural colors in the current chunk, as such a transition breaks the temporal consistency of the sequence. The model is thereby forced to compromise local visual realism to accommodate an imperfect temporal context. To resolve this, we must disentangle the quality supervision of $x _ { i }$ from the rest of the rollout.

## 3.2 ROLLOUT-MARGINAL OBJECTIVE

Video-level DMD scores the entire sequence, tying each chunk’s quality correction to its compatibility with the surrounding content. RMD breaks this dependency by matching chunks from the causal rollouts independently, while the generator still uses its history for prediction.

We model the initial and continuation chunks separately to account for their distinct latent statistics. Specifically, under the video VAE, the first latent frame encodes only the initial image without temporal compression (Wan et al., 2025; Huang et al., 2026a). This gives it completely different statistical properties compared to subsequent frames. Let $q _ { \theta , 1 } ( x _ { 1 } ~ | ~ c )$ denote the initial chunk distribution. To define the continuation rollout marginal, we pool the chunks $x _ { 2 } , \ldots , x _ { T }$ across the student’s causal rollouts for prompt c:

![](images/0ed252164c98467ec3cfe10b39e0cd4730e65560be80704fa9806dc3f54dbb1e.jpg)  
Figure 3: Overview of Rollout-Marginal Training. The AR generator produces a causal rollout, sampling the initial chunk $( x _ { 1 } )$ at a random-step exit and continuation chunks $( x _ { 2 } , \dots , x _ { T } )$ at the last-step exit. Adapted score networks evaluate them independently without temporal context. The resulting gradients $( g _ { 1 } , \dots , g _ { T } )$ are backpropagated via differentiable causal replay.

$$
q _ { \theta , \mathrm { c h u n k } } ^ { ( T ) } ( x \mid c ) = \frac { 1 } { T - 1 } \sum _ { i = 2 } ^ { T } \mathbb { E } _ { x < i \sim q _ { \theta } ( x < i \mid c ) } [ q _ { \theta } ( x \mid x _ { < i } , c ) ] .\tag{4}
$$

This distribution averages over both the generated histories and the continuation positions. Sampling from it requires the causal rollout, but evaluating a chunk’s realism requires neither its history nor its rollout position.

At noise level $\tau ,$ let $p _ { 1 , \tau }$ and $p _  \mathrm { c h u n k } , $ <sub>τ</sub> denote the teacher-defined target distributions for the initial frame and continuation chunks, respectively. Weighting these two groups by their frequency in the rollout yields the RMD objective:

$$
{ \mathcal { L } } _ { \mathrm { R M D } } ( \theta ) = \mathbb { E } _ { c , \tau } \left[ { \frac { 1 } { T } } D _ { \mathrm { K L } } { \big ( } q _ { \theta , 1 , \tau } \parallel p _ { 1 , \tau } { \big ) } + { \frac { T - 1 } { T } } D _ { \mathrm { K L } } { \big ( } q _ { \theta , \mathrm { c h u n k } , \tau } ^ { ( T ) } \parallel p _ { \mathrm { c h u n k } , \tau } { \big ) } \right] .\tag{5}
$$

Crucially, the generation process remains causal, meaning each sample $x _ { i }$ is still produced conditionally from its actual on-policy history $x _ { < i }$ . However, the supervision applied to each chunk is entirely context-free. This objective simply penalizes generated chunks that are unlikely under the teacher’s chunk distribution. By doing so, it explicitly isolates visual quality supervision from the imperfect generated history. Instead of forcing the model to repeat past artifacts to maintain temporal consistency, this marginal matching ensures the teacher guides the current chunk toward high-quality targets, regardless of the flaws in the preceding frames.

## 3.3 ROLLOUT-MARGINAL TRAINING

To realize this decoupled supervision in practice, our training pipeline (Figure 3) optimizes the rollout-marginal objective by first generating the causal rollout, then evaluating the initial and continuation chunks against their distinct target distributions, and finally backpropagating the isolated gradients to the generator.

We initialize the AR generators from Causal Forcing’s causal ODE checkpoints (Zhu et al., 2026). Following the self-forcing framework (Huang et al., 2026a), at each training iteration, we generate a causal rollout of chunks $x _ { 1 } , \dots , x _ { T }$ , sampling each chunk in 4 denoising steps. To retain the temporal dynamics prior in the pretrained model, we carefully choose which denoising step to supervise for each chunk (Wu et al., 2026). Early steps in the sampling trajectory rely heavily on historical context to establish global motion, and applying independent supervision at these timesteps would disrupt the temporal consistency. Therefore, we adopt an asymmetric supervision strategy: for the initial chunk $x _ { 1 }$ , we compute the loss at a randomly selected intermediate exit step along the trajectory, whereas for the continuation chunks $x _ { 2 } , \ldots , x _ { T }$ , we apply the loss only at the final, lowest-noise step of the rollout where the network refines local visual details.

To score the sampled chunks, we use separate teacher targets for the initial frame and continuations. As discussed in Section 3.2, Wan’s causal VAE gives them different latent statistics. An unmodified video teacher would treat an isolated continuation chunk as the start of a video and expect an uncompressed first frame. We therefore adapt the Wan 14B video teacher (Wan et al., 2025) with LoRA (Hu et al., 2022) on continuation chunks extracted from videos generated by the original model. The adapted teacher supplies the continuation target $p _ { \mathrm { c h u n k } , \tau }$ in Equation 5, allowing each continuation to be scored without its generated history.

For each $x _ { i } ,$ we add scoring noise and obtain clean predictions from its teacher and fake-score networks on the same noisy input. Let $D _ { \mathrm { r e a l , : } }$ <sub>i</sub> and $D _ { \mathrm { f a k e } , i }$ denote these two predictions. Their normalized difference gives a gradient estimate $g _ { i }$ for the chunk:

$$
g _ { i } = \frac { D _ { \mathrm { f a k e } , i } - D _ { \mathrm { r e a l } , i } } { \mathrm { m e a n } \left| x _ { i } - D _ { \mathrm { r e a l } , i } \right| } .\tag{6}
$$

To propagate $g _ { i }$ to the generator parameters, we follow checkpointed self-forcing (Savva et al., 2026). The causal rollout is generated without retaining its computation graph. A differentiable replay then recomputes each supervised prediction from its sampled noisy input and generated history, allowing $g _ { i }$ to be backpropagated through the generator.

The first latent frame has different statistics from later latents. Once it leaves the sliding window, the generator must predict without that distinctive frame, a condition absent from short training rollouts that can cause flicker. Following Self-Forcing (Huang et al., 2026a), we mask attention to the initial block when replaying predictions in the latter half of the training window.

## 3.4 REFINEMENT WITH VIDEO-LEVEL DMD

Matching chunk marginals improves their appearance but does not directly constrain the transitions between them. Individually realistic chunks can still form a video with inconsistent appearance or motion. We therefore refine the AR generator with the video-level DMD objective in Equation 2.

The generator still produces causal self-rollouts, but the video teacher and online fake-score network now evaluate each complete rollout jointly. Their score difference updates the generator through the same differentiable replay, supervising motion and consistency across chunks. This stage begins from a generator already trained using the rollout-marginal objective. We use smaller learning rates to limit the changes made during refinement.

## 4 EXPERIMENTS

## 4.1 EXPERIMENTAL SETUP

Implementation Details. We distill the pretrained Wan 14B video diffusion model into a causal Wan 1.3B generator, initialized from Causal Forcing’s causal ODE checkpoints (Zhu et al., 2026). RMD is trained in two stages using a metadata set comprising 70,000 unique prompts sourced from Self Forcing (Huang et al., 2026a). Our models are trained under a strictly fixed 81-frame horizon, without performing longer rollouts during the training phase.

All trainable networks are optimized using AdamW (Loshchilov & Hutter, 2019) $( \beta _ { 1 } = 0 , \beta _ { 2 } =$ 0.999, weight decay 0.01) with a global batch size of 8. During the rollout-marginal training stage, we alternate 5 fake-score updates for every 1 generator update. When the chunk size is 1, both the generator and fake-score networks use a learning rate of $2 \times 1 0 ^ { - 5 }$ , and we train for 900 steps. When the chunk size is 3, the generator learning rate remains $2 \times 1 0 ^ { - 5 }$ , while the fake-score learning rates are reduced to $2 \times 1 0 ^ { - 7 }$ , and we train for 750 steps. In the subsequent video-level refinement stage, the generator and fake-score learning rates are reduced to $2 \times 1 0 ^ { - 6 }$ and $4 \times 1 0 ^ { - 7 }$ , respectively, and both models are trained for 800 steps.

![](images/2d705cdc866a0901225c223c7d8b259a0cd54dce66362b15690112f79be90cbd.jpg)  
Figure 4: Qualitative comparison of long-horizon generation. While baseline methods suffer from severe color saturation and structural collapse beyond their training horizon, RMD consistently preserves sharp subject details and stable scene structures.

Evaluation Protocol. We adopt the VBench-Long suite (Huang et al., 2024; 2026b) for our primary evaluation, using the complete standard set of 944 unique prompts with five generations per prompt. All videos are generated at 832 × 480 resolution and 16 FPS, and evaluated at a duration of approximately 60 seconds for both chunk sizes. The 500-second rollout in Figure 1 is a qualitative example and is not included in these aggregate metrics.

## 4.2 COMPARISON WITH BASELINES

We compare RMD with Self Forcing (Huang et al., 2026a) and Causal Forcing (Zhu et al., 2026). Both baselines share our standard sliding-window generation protocol, operating entirely without fixed frame anchors or modifications to the underlying autoregressive logic. This isolates the distillation objectives and ensures RMD introduces zero additional computational overhead during inference. We report results separately using chunk sizes of 1 and 3.

Figure 4 presents corresponding qualitative comparisons over extended rollouts. Within the 5- second training horizon, all methods generate reasonable frames. However, as generation progresses far beyond this boundary (20s, 40s, 60s), the baselines exhibit visual degradation including color saturation and structural collapse. In contrast, RMD consistently preserves clear subject details and stable scene structures throughout the 1-minute trajectory.

Table 1: Quantitative comparison on long-horizon generation. Evaluated on VBench-Long, RMD consistently outperforms video-level distillation baselines across all metrics.
<table><tr><td>Chunk size</td><td>Method</td><td>Total ↑</td><td>Quality ↑</td><td>Semantic ↑</td></tr><tr><td rowspan="3">1</td><td>Self Forcing</td><td>70.94</td><td>77.89</td><td>43.17</td></tr><tr><td>Causal Forcing</td><td>69.88</td><td>78.34</td><td>36.03</td></tr><tr><td>RMD (ours)</td><td>81.26</td><td>84.52</td><td>68.24</td></tr><tr><td rowspan="3">3</td><td>Self Forcing</td><td>77.44</td><td>81.99</td><td>59.22</td></tr><tr><td>Causal Forcing</td><td>74.88</td><td>80.53</td><td>52.27</td></tr><tr><td>RMD (ours)</td><td>81.48</td><td>85.03</td><td>67.24</td></tr></table>

![](images/e3373474625aa722a5401a19dcc7431ac28fe0ad4f004634d4cd6ef221f0b2d5.jpg)  
Figure 5: Performance over increasing rollout lengths. VBench-Long scores are evaluated at durations ranging from 10 to 60 seconds. As generation extends, RMD maintains stable visual quality and significantly mitigates the performance degradation suffered by baseline methods.

Table 1 presents the quantitative comparison on 1-minute generated videos. RMD establishes a clear advantage across both configurations. With a chunk size of 1, our method improves the total score by over 10 absolute points compared with the baselines (81.26 vs. 70.94 and 69.88), while attaining quality and semantic scores of 84.52 and 68.24, respectively. With a chunk size of 3, RMD achieves a total score of 81.48, with corresponding quality and semantic scores of 85.03 and 67.24. Together, these quantitative and qualitative results demonstrate that explicitly decoupling chunk supervision from historical context effectively prevents the degradation typical of joint video-level distillation.

## 4.3 LONG-HORIZON EXTRAPOLATION

To assess performance degradation over extended rollouts, we evaluate the same videos generated with a chunk size of 1 at durations ranging from 10 to 60 seconds. Figure 5 plots the performance dynamics over these trajectories. RMD maintains stable visual quality as the rollout extends: its quality score changes only from 84.68 at 10 seconds to 84.52 at 60 seconds, where it retains a semantic score of 68.24. In contrast, the baselines degrade steeply as prediction errors compound; for example, the total score of Self Forcing drops from 79.12 to 70.94 over the same interval. This robust extrapolation confirms that independent chunk supervision effectively mitigates the accumulation of visual artifacts.

## 4.4 ABLATION STUDIES

Effectiveness of the Rollout-Marginal Objective. Table 2 evaluates the components of RMD using a chunk size of 1. Applying video-level DMD alone yields a total score of 72.44, and qualitatively, the generation rapidly collapses into severe color artifacts (Figure 6). Optimizing exclusively for the rollout-marginal objective (Marginal only) raises the quality score to 84.90, confirming that context-free scoring effectively prevents the visual degradation.

![](images/914f74646d208c572b1cdab88af507d08616146222259e51d9d0c8b6e621d50e.jpg)  
Figure 6: Qualitative ablation results. The spatio-temporal slice is extracted along the red line. Removing the rollout-marginal objective (video DMD only) causes severe color artifacts. Removing video-level refinement (marginal only) introduces temporal flickering (jagged slice artifacts). Removing asymmetric denoising exits (random exits) destroys the temporal prior, resulting in severe temporal jitter. Full RMD achieves both high visual quality and temporal coherence.

Table 2: Quantitative ablation results. (a) Removing refinement with video DMD (marginal only) or asymmetric exits (random exits) degrades overall performance. (b) Temporal metrics confirm that this refinement is essential for temporal coherence.  
(a) Aggregate scores
<table><tr><td>Variant</td><td></td><td></td><td>Total ↑ Quality ↑ Semantic ↑</td></tr><tr><td>video DMD only</td><td>72.44</td><td>78.68</td><td>47.50</td></tr><tr><td>marginal only</td><td>80.64</td><td>84.90</td><td>63.59</td></tr><tr><td>random exits</td><td>78.29</td><td>82.48</td><td>61.56</td></tr><tr><td>RMD (full)</td><td>81.26</td><td>84.52</td><td>68.24</td></tr></table>

(b) Temporal metrics
<table><tr><td>Metric</td><td>Marginal only RMD (full)</td></tr><tr><td>Subject Cons. ↑</td><td>97.29 98.05</td></tr><tr><td>Background Cons. ↑</td><td>96.26 96.89</td></tr><tr><td>Temporal Flickering ↑</td><td>99.06 99.41</td></tr><tr><td>Motion Smoothness ↑</td><td>98.79 98.96</td></tr></table>

Effectiveness of Refinement with Video-Level DMD. While the Marginal only variant achieves high visual realism, it lacks cross-chunk constraints and falls short on temporal coherence, which is visibly manifested as jagged flickering in its spatio-temporal slice (Figure 6). The full RMD pipeline resolves this via subsequent refinement with video-level DMD, which improves subject consistency, background consistency, and motion smoothness (Table 2(b)). As shown in Figure 6, this combined approach achieves the best total and semantic scores (81.26 and 68.24, respectively) while restoring smoother, more continuous trajectories in the spatio-temporal slice.

Effectiveness of Asymmetric Denoising Exits. A variant that applies random-step exits uniformly across all chunks suffers from severe temporal jitter (Figure 6) and performs worse across all metrics, with total and quality scores of 78.29 and 82.48, respectively (Table 2(a)). This supports our asymmetric design: supervising the initial chunk across random steps preserves the pretrained temporal prior, while restricting continuation chunks to the final exit strictly refines local details.

## 5 CONCLUSION

Autoregressive video generation inherently suffers from error accumulation because joint temporal scoring forces the teacher to compromise local visual quality for sequence-level consistency. To resolve this coupling, we introduce Rollout-Marginal Distillation (RMD), which disentangles spatial realism from temporal coherence. By independently matching chunk marginals and subsequently refining global dynamics, RMD provides a clean, context-free supervision signal. Quantitative evaluations over 60-second rollouts demonstrate that RMD effectively prevents artifact accumulation beyond the training horizon, while a 500-second qualitative example illustrates its potential at sub stantially longer durations. We believe this decoupled paradigm offers a robust foundation for future long-horizon, streamable video generation.

## REFERENCES

Hmrishav Bandyopadhyay, Xuanchi Ren, Zijian Huang, Jay Zhangjie Wu, Tianshi Cao, Ruilong Li, Bryan Chu, Sanja Fidler, Yi-Zhe Song, and Zian Wang. Context-matched distillation: Teacher causality for autoregressive video distillation. arXiv preprint arXiv:2608.13391, 2026.

Jake Bruce, Michael D Dennis, Ashley Edwards, Jack Parker-Holder, Yuge Shi, Edward Hughes, Matthew Lai, Aditi Mavalankar, Richie Steigerwald, Chris Apps, Yusuf Aytar, Sarah Maria Elisabeth Bechtle, Feryal Behbahani, Stephanie C.Y. Chan, Nicolas Heess, Lucy Gonzalez, Simon Osindero, Sherjil Ozair, Scott Reed, Jingwei Zhang, Konrad Zolna, Jeff Clune, Nando de Freitas, Satinder Singh, and Tim Rocktaschel. Genie: Generative interactive environments. In¨ Forty-first International Conference on Machine Learning, 2024. URL https://openreview.net/ forum?id=bJbSbJskOS.

Edward J Hu, yelong shen, Phillip Wallis, Zeyuan Allen-Zhu, Yuanzhi Li, Shean Wang, Lu Wang, and Weizhu Chen. LoRA: Low-rank adaptation of large language models. In International Conference on Learning Representations, 2022. URL https://openreview.net/forum? id=nZeVKeeFYf9.

Xun Huang, Zhengqi Li, Guande He, Mingyuan Zhou, and Eli Shechtman. Self forcing: Bridging the train-test gap in autoregressive video diffusion. Advances in Neural Information Processing Systems, 38:167283–167308, 2026a.

Ziqi Huang, Yinan He, Jiashuo Yu, Fan Zhang, Chenyang Si, Yuming Jiang, Yuanhan Zhang, Tianxing Wu, Qingyang Jin, Nattapol Chanpaisit, Yaohui Wang, Xinyuan Chen, Limin Wang, Dahua Lin, Yu Qiao, and Ziwei Liu. VBench: Comprehensive benchmark suite for video generative models. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recogni tion, 2024.

Ziqi Huang, Fan Zhang, Xiaojie Xu, Yinan He, Jiashuo Yu, Ziyue Dong, Qianli Ma, Nattapol Chanpaisit, Chenyang Si, Yuming Jiang, Yaohui Wang, Xinyuan Chen, Ying-Cong Chen, Limin Wang, Dahua Lin, Yu Qiao, and Ziwei Liu. VBench++: Comprehensive and Versatile Benchmark Suite for Video Generative Models . IEEE Transactions on Pattern Analysis & Machine Intelligence, 48(03):3268–3285, March 2026b. ISSN 1939-3539. doi: 10.1109/TPAMI.2025.3633890. URL https://doi.ieeecomputersociety.org/10.1109/TPAMI.2025.3633890.

Haodong Li, Shaoteng Liu, Zhe Lin, and Manmohan Chandraker. Rolling sink: Bridging limitedhorizon training and open-ended testing in autoregressive video diffusion. arXiv preprint arXiv:2602.07775, 2026.

Hongyu Liu, Chun Wang, Feng Gao, Xuanhua He, Yue Ma, Ziyu Wan, Yong Zhang, Xiaoming Wei, and Qifeng Chen. Opsd-v: On-policy self-distillation for post-training few-step autoregressive video generators. arXiv preprint arXiv:2607.08766, 2026.

Shaowei Liu, Zhongzheng Ren, Saurabh Gupta, and Shenlong Wang. Physgen: Rigid-body physicsgrounded image-to-video generation. In European Conference on Computer Vision, pp. 360–378. Springer, 2024.

Ilya Loshchilov and Frank Hutter. Decoupled weight decay regularization. In International Conference on Learning Representations, 2019. URL https://openreview.net/forum?id= Bkg6RiCqY7.

Yanzuo Lu, Yuxi Ren, Xin Xia, Shanchuan Lin, Xing Wang, Xuefeng Xiao, Andy J Ma, Xiaohua Xie, and Jian-Huang Lai. Adversarial distribution matching for diffusion distillation towards efficient image and video synthesis. In 2025 IEEE/CVF International Conference on Computer Vision (ICCV), pp. 16818–16829. IEEE, 2025.

Georgy Savva, Oscar Michel, Daohan Lu, Suppakit Waiwitlikhit, Timothy Meehan, Dhairya Mishra, Srivats Poddar, Jack Lu, and Saining Xie. Solaris: Building a multiplayer video world model in minecraft. arXiv preprint arXiv:2602.22208, 2026.

Team Wan, Ang Wang, Baole Ai, Bin Wen, Chaojie Mao, Chen-Wei Xie, Di Chen, Feiwu Yu, Haiming Zhao, Jianxiao Yang, et al. Wan: Open and advanced large-scale video generative models. arXiv preprint arXiv:2503.20314, 2025.

Yutong Wang, Haiyu Zhang, Tianfan Xue, Yu Qiao, Yaohui Wang, Chang Xu, and Xinyuan Chen. Vdot: Efficient unified video creation via optimal transport distillation. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 9273–9283, 2026.

Tianhe Wu, Ruibin Li, Lei Zhang, and Kede Ma. Diversity-preserved distribution matching distillation for fast visual synthesis. In Forty-third International Conference on Machine Learning, 2026. URL https://openreview.net/forum?id=R3JzLU2qZj.

Shuai Yang, Wei Huang, Ruihang Chu, Yicheng Xiao, Yuyang Zhao, Xianbang Wang, Muyang Li, Enze Xie, Ying-Cong Chen, Yao Lu, Song Han, and Yukang Chen. Longlive: Real-time interactive long video generation. In The Fourteenth International Conference on Learning Representations, 2026. URL https://openreview.net/forum?id=nCAODkpsPJ.

Tianwei Yin, Michael Gharbi, Taesung Park, Richard Zhang, Eli Shechtman, Fredo Durand, and¨ William T Freeman. Improved distribution matching distillation for fast image synthesis. Advances in neural information processing systems, 37:47455–47487, 2024a.

Tianwei Yin, Michael Gharbi, Richard Zhang, Eli Shechtman, Fredo Durand, William T Freeman,¨ and Taesung Park. One-step diffusion with distribution matching distillation. In 2024 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pp. 6613–6623. IEEE, 2024b.

Tianwei Yin, Qiang Zhang, Richard Zhang, William T Freeman, Fredo Durand, Eli Shechtman, and Xun Huang. From slow bidirectional to fast autoregressive video diffusion models. In 2025 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pp. 22963–22974. IEEE, 2025.

Kaiwen Zhang, Zhenyu Tang, Xiaotao Hu, Xingang Pan, Xiaoyang Guo, Yuan Liu, Jingwei Huang, Li Yuan, Qian Zhang, Xiao-Xiao Long, Xun Cao, and Wei Yin. Epona: Autoregressive diffusion world model for autonomous driving. In Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV), 2025.

Kaiwen Zheng, Guande He, Min Zhao, Jintao Zhang, Huayu Chen, Jianfei Chen, Chen-Hsuan Lin, Ming-Yu Liu, Jun Zhu, and Qianli Ma. Causal-rcm: A unified teacher-forcing and self-forcing open recipe for autoregressive diffusion distillation in streaming video generation and interactive world models. arXiv preprint arXiv:2606.25473, 2026a.

Kaiwen Zheng, Yuji Wang, Qianli Ma, Huayu Chen, Jintao Zhang, Yogesh Balaji, Jianfei Chen, Ming-Yu Liu, Jun Zhu, and Qinsheng Zhang. Large scale diffusion distillation via score-regularized continuous-time consistency. In The Fourteenth International Conference on Learning Representations, 2026b. URL https://openreview.net/forum?id= 2uNlM353RI.

Hongzhou Zhu, Min Zhao, Guande He, Hang Su, Chongxuan Li, and Jun Zhu. Causal forcing: Autoregressive diffusion distillation done right for high-quality real-time interactive video generation. In Forty-third International Conference on Machine Learning, 2026. URL https: //openreview.net/forum?id=BYInOck3gr.
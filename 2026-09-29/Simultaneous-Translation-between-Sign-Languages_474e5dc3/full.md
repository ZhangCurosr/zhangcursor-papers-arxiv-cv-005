# Simultaneous Translation between Sign Languages

Zetian Wu Bowen Xie Stefan Lee Liang Huang

Oregon State University

{wuzet, xiebo, leestef, liang.huang}@oregonstate.edu

## Abstract

Deaf and hard-of-hearing (DHH) signers cannot converse in real time across different sign languages today: existing sign-to-sign translation systems run offline, requiring the full source clip before any target sign is emitted. Live use cases — e.g. broadcast interpretation and two-way video calls — instead demand simultaneous output, while the source signer is still signing. We present, to our knowledge, the first simultaneous sign → sign (S2S) translation system, with two wait-k regimes: testtime wait-k inference applied directly to a fullsentence model, and a trained wait-k model via stochastic multi-path supervision. We further introduce ca-Stream-AL, a computation-aware latency metric for streaming output. Averaged across six S2S directions on both a smaller human-verified test set and a larger synthetic S2S corpus, our streaming system achieves a 38% ca-Stream-AL reduction while staying within a 9% DTW-PA-MPJPE increase and a 2.1 BLEU-4 drop compared to the full-sentence baseline. A word-order case study probes how the streaming model handles word order mismatch between different sign languages — a consequence of simultaneous translation.

## 1 Introduction

Sign language is the primary, most natural communication medium for many Deaf and hard-ofhearing (DHH) users — which is why major broadcasts such as White House press briefings provide live ASL interpretations alongside written captions, recognizing that the visual-spatial grammar of sign carries information that written text alone cannot. Yet DHH signers across different sign-language communities cannot converse in real time today. The cascade approach, which chains a single signto-text (S2T) model with a spoken-language MT system and a text-to-sign (T2S) generator, incurs substantial latency; meanwhile, both the recent direct S2S model of Wu et al. (2026) and the earlier approach of Inan et al. (2025) remain offline, requiring the entire source clip before any target sign can be generated. This rules out the live deployments — e.g. broadcast interpretation and two-way video calls — where simultaneous S2S would matter most.

Simultaneous machine translation between spoken languages faces an analogous problem and commonly adopts wait-k policies (Ma et al., 2019), which delay generation until k source units have been observed and then proceed incrementally. We extend this principle to direct S2S translation as an end-to-end streaming pipeline, with the wait-k schedule governing the entire path from sourcesign encoding and LM inference to target-sign decoding and rendering. This setting also introduces a distinct latency challenge: unlike text tokens, generated sign chunks have non-negligible playback duration, during which new source input can arrive and subsequent chunks can be computed. Existing latency metrics therefore do not fully capture the timing experienced in streaming sign output.

We present, to our knowledge, the first simultaneous S2S translation system, adapting wait-k under both test-time and trained regimes. We further introduce ca-Stream-AL, a source-centric, computation-aware latency metric designed for streaming sign generation, accounting jointly for source arrival, model computation, and target playback where AL (Ma et al., 2019) and CA-AL (Ma et al., 2020) can underestimate latency.

Averaged across six S2S directions on both a small human-verified test set and a larger synthetic S2S corpus (Wu et al., 2026), our system achieves a 38% ca-Stream-AL reduction while staying within a 9% mean per joint position error (temporally and spatially aligned using dynamic time wrapping and Procrustes-aligned, i.e. DTW-PA-MPJPE)(Lin et al., 2023; Yu et al., 2024; Shalev-Arkushin et al., 2022; Baltatzis et al., 2024) increase and a 2.1 BLEU-4 drop relative to the full-sentence baseline.

In summary, our contributions are:

• First simultaneous direct sign → sign translation (§3): we formalize the streaming-S2S task.

• Two wait-k regimes (§3): test-time inference on a full-sentence model and a trained wait-k model via stochastic multi-path supervision.

• Ca-Stream-AL latency metric (§4): a computation-aware metric that captures fixedduration emission missed by AL and CA-AL.

• Word-order case study (§6): an analysis of how the streaming model handles word order mismatch between sign languages.

## 2 Direct Sign-to-Sign Translation

The direct S2S task takes a signed utterance in one sign language and produces a sign sequence in another within a single model, rather than cascading through sign-to-text recognition, spoken-language machine translation, and text-to-sign generation. Inan et al. (2025) present the first attempt in this setting and contribute methods for aligning multiple signed-language corpora to enable cross-lingual training; however, their generated signs achieve a back-translation BLEU-4 of 0 in two of the six translation directions, with the best direction reaching only 6.81, indicating the limited translation quality of this early system. Wu et al. (2026) extend the line with two ingredients: (i) a back-translationsynthesized sign ↔ sign parallel corpus across ASL / CSL / DGS, and (ii) a single encoder–decoder architecture trained jointly on T2S and S2S, sharing a frozen VQ-VAE sign tokenizer with SOKE (Zuo et al., 2024). Both prior systems are offline: the encoder consumes the entire source clip before any target sign is emitted, which precludes live use.

Backbone we adopt. We follow the pipeline of Wu et al. (2026) without architectural modification. A frozen decoupled VQ-VAE (Zuo et al., 2024) tokenizes 25 Hz SMPL-X poses into three streams of sign tokens (body, left hand, right hand), with temporal downsampling factor 4 → effective 6.25 Hz token rate. A single transformer, initialized from MBART-large-cc25 (Liu et al., 2020), processes both T2S and S2S samples; sign-side input uses the SOKE (Zuo et al., 2024) embedding fusion (one fused embedding per frame), text-side input uses standard subword embeddings. Source / target sign language is signaled via special-token prefixes (ASL, CSL, DGS); the same module serves both T2S and S2S tasks. At test time the decoder emits target sign tokens autoregressively; the VQ-VAE decoder reconstructs SMPL-X poses, which are rendered to an avatar in parallel with generation.

![](images/3fb2ab157a7b5978a3cbbb926e7f712a39adcc2f401e329024b7a027e0ba6f8c.jpg)  
Figure 1: Simultaneous sign → sign translation pipeline, shown for CSL→DGS (the same architecture serves all six sign → sign directions). The source video is incrementally fed to SMPL-X and tokenized by a VQ-VAE encoder into the source sign-token stream (purple, pertoken duration l); wait-k decoding (§3.1) emits target sign tokens (cyan, per-token duration l) after a k-token lag, and a VQ-VAE decoder maps them back to SMPL-X for rendering as the target (DGS) avatar. We use l=160 ms in this work (4 frames at 25 fps). The vertical dashed line marks the end of source tokens at which target signing has already begun — but under offline sign → sign no target frame would be emitted before this point.

## 3 Simultaneous Sign-to-Sign Translation

## 3.1 Wait-k Inference

Policy. Let $\mathbf { x } = \left( x _ { 1 } , \ldots , x _ { | \mathbf { x } | } \right)$ be the source signtoken stream and $\mathbf { y } ~ = ~ ( y _ { 1 } , \ldots , y _ { | \mathbf { y } | } )$ the target. Each unit $x _ { i }$ or $y _ { t }$ is a sign token corresponding to 4 source/target frames at 25 fps (=160 ms of physical signing time). For full-sentence translation, the decoder conditions on the entire source:

$$
y _ { t } = \operatorname { a r g m a x } _ { y } p _ { \mathbf { \theta } } ( y \mid \mathbf { x } , \mathbf { y } _ { < t } ) .\tag{1}
$$

For simultaneous translation, let g(t) denote the number of source tokens available when generating the t-th target token. The decoder therefore conditions only on the currently available source prefix:

$$
y _ { t } = \operatorname { a r g m a x } _ { y } p _ { \theta } \big ( y \big | \mathbf { x } _ { \le g ( t ) } , \mathbf { y } _ { < t } \big ) .\tag{2}
$$

In our wait-k setting, the read policy is

$$
g _ { k } ( t ) = \mathrm { m i n } ( k + t - 1 , | { \bf x } | ) ,\tag{3}
$$

Algorithm 1 Wait-k decoding   
Require: streaming source x (final length |x|), lag   
$k ,$ max target length $T _ { \mathrm { m a x } }$   
1: $\mathbf { y } \gets \langle \mathsf { b o s } \rangle$ $\triangleright y _ { 0 } = \langle \mathsf { b o s } \rangle$   
2: for $t = 1 , 2 , \ldots , T _ { \mathrm { m a x } }$ do   
3: $g _ { k } ( t ) \gets \operatorname* { m i n } ( k + t - 1 , | \mathbf x | )$ ▷ read   
4: h ← Encoder $( { \bf x } _ { \le g _ { k } ( t ) } )$ ▷ re-encode   
5: $y _ { t } \gets \operatorname { a r g m a x } _ { y } p _ { \pmb { \theta } } ( y \mid \mathbf { h } , \mathbf { y } _ { < t } )$ ▷ Eq. 2   
6: if $y _ { t } = \langle \mathsf { e o s } \rangle$ then   
7: break   
8: yield $y _ { t }$ ▷ incremental output   
9: $\mathbf { y }  \mathbf { y } \circ y _ { t }$

which first reads k source tokens and then reveals one additional source token for each generated target token. Once the source is exhausted, $g _ { k } ( t ) = | { \bf x } |$ and decoding continues conditioned on the full source.

Why re-encode per step. MBART’s encoder is bidirectional: for any observed source prefix, the hidden state $\mathbf { h } _ { i }$ at position i attends to all tokens in that prefix, including those after i. Consequently, when a new source token becomes available, the hidden states of earlier positions may also change and must be recomputed. A causal-mask approximation would discard the bidirectional inductive bias of the pre-trained backbone. We therefore interleave reading and writing: at each decoder step t we reveal one more source token, re-run the encoder over the current prefix $\mathbf { x } _ { \leq g _ { k } ( t ) }$ , and decode y<sub>t</sub> against the updated hidden states (Algorithm 1); $T _ { \mathrm { m a x } }$ caps the number of decoder steps. Per-step re-encoding scales encoder work as $\mathcal { O } ( | \mathbf { x } | ^ { 3 } )$ , but is acceptable in practice because end-to-end latency is currently dominated by the rendering stage (§4).

Latency unit. A source-side unit of the transformer is one sign token, which we denote as a “chunk”; each chunk includes 4 source frames, which has length $w _ { \mathrm { c h u n k } } = 4 / 2 5 ~ \mathrm { s } = 1 6 0$ ms (25 fps source timeline). AL is reported in seconds.

## 3.2 Trained Wait-k Model

Full-sentence training shows the model the full source at every target position; wait-k inference (Eq. 2) exposes position t only to a prefix of $k { + } t { - } 1$ source tokens. The prefix-to-prefix objective of Ma et al. (2019) trains θ under this wait-k conditioning:

$$
\mathcal { L } ( \pmb { \theta } ) = - \sum _ { t = 1 } ^ { | \mathbf { y } | } \log p _ { \pmb { \theta } } \big ( y _ { t } \big | \mathbf { x } _ { \le z _ { k } ( t ) } , \mathbf { y } _ { < t } \big ) .\tag{4}
$$

While Eq. 4 directly matches wait-k inference, optimizing it exhaustively is impractical for sign-video inputs. Source sign sequences are much longer than typical text sequences, and each target position requires re-encoding a different source prefix, resulting in min $( | \mathbf { x } | - k , | \mathbf { y } | )$ encoder forwards per sample. Batching these forwards exceeds GPU memory for typical sequence lengths. Following sampledprefix training in simultaneous speech translation (Zhang et al., 2023), we instead sample m target positions uniformly and create m replicas: replica j receives $\mathbf { x } _ { \leq k + t _ { j } - 1 }$ , with the loss masked at all positions except $t _ { j }$

## 4 Computational-aware Lagging for Stream: ca-Stream-AL

We introduce computational-aware stream average lagging (ca-Stream-AL), a source-centric computation-aware latency metric for streaming sign output, reported in seconds along the 25 fps source timeline (each emitted chunk spans $w _ { \mathrm { c h u n k } } = 4 / 2 5 ~ \mathrm { s = 1 6 0 }$ ms of wall-clock time). We begin by discussing the limitations of classical average lagging (AL), and develop ca-Stream-AL in two stages: a non-computation-aware formulation (Stream-AL) that establishes the sourcecentric view, followed by a computation-aware extension that incorporates per-step computation lag (ca-Stream-AL).

## 4.1 Classical Average Lagging

Classical average lagging, introduced by Ma et al. (2019), measures latency only over target steps up to the point at which the full source has been consumed. It is defined as

$$
{ \mathrm { A L } } = { \frac { 1 } { \tau } } \sum _ { t = 1 } ^ { \tau } \left[ g ( t ) - ( t - 1 ) / r \right]\tag{5}
$$

where $\begin{array} { r } { r = \frac { | \mathbf { y } | } { | \mathbf { x } | } } \end{array}$ is the target-to-source length ratio and $\tau = \operatorname* { m i n } \{ t : g ( t ) = | \mathbf { x } | \}$ is the first target step at which the full source has been consumed.

Limitations This formulation has two limitations that motivate our extensions:

1. Target-tail truncation. By definition, AL stops at τ and excludes all subsequent target steps from the latency computation (Fig. 2a). These steps can still carry nonzero lag relative to the proportional ideal and should therefore contribute to the metric. When $r \approx 1$ , the omitted tail is typically short and the resulting (b) Fix to classical AL: extending the averaging horizon from τ to the full target sequence.

![](images/1baf42d0386b59ef0b21ba1ef922336c76963b98aa969c50af71cd4ddad7825b.jpg)  
(a) Classical AL: latency not counted for tail past cut-off point.AL-fix : target-centric

![](images/5e0cc2921466f6587b3b0d53de8648c22ab9e2774d1a686ab53bd4e0a9ec1301.jpg)

![](images/be6776d334dea98e31cad54d03c53435fa7740d59ae4a26ee093f1ada73604d4.jpg)  
(c) Our Stream-AL: shared $w _ { \mathrm { c h u n k } }$ -spaced time axis.  
Figure 2: From classical AL to Stream-AL. (a) Dropping the τ cutoff captures the per-token theoretical lag of the target tail, but the metric remains target-centric and implicitly assumes the tail can be batch-emitted at the source-end instant $t = | \mathbf { x } | w _ { \mathrm { c h u n k } }$ (visualized by the targets “dropped down” to a single column). (b) We fix the target-tail truncation issue by removing the source-end cutoff, which means extending the averaging horizon from the previous cutoff point τ to the full target sequence length |y|. (c) Stream-AL lays source and target on a shared $w _ { \mathrm { c h u n k } } { = } 1 6 0$ ms time axis and sums the signed gap against the source-side proportional ideal. Every unit is a real $w _ { \mathrm { c h u n k } }$ slot of wall-clock time, so the metric converts to seconds via $\times w _ { \mathrm { c h u n k } } .$

error is small; as r increases, however, the tail can account for a substantial fraction of the target sequence, making the underestimation increasingly severe.

2. No duration for streamed target tails. The second issue arises for temporally realized outputs such as speech or sign video (i.e., in text-to-speech, speech-to-speech, or sign language translation scenarios). Unlike text, where the tail can be presented as a complete sequence once generated, a streamed output must unfold over time (at a reasonable rate) for the perceiver to consume it; its remaining tail therefore cannot be treated as if it were emitted instantaneously at the source end, nor can it be presented too fast which interferes with understanding. For example, in speechto-speech translation, substantially increasing the playback rate degrades naturalness and can disrupt comprehension (Zheng et al., 2020).

Removing the source-end cutoff. We first address the target-tail truncation by extending the averaging 9horizon from τ to the full target sequence:

$$
\mathrm { A L } _ { \mathrm { e x t } } = { \frac { 1 } { | \mathbf { y } | } } \sum _ { t = 1 } ^ { | \mathbf { y } | } \left[ g ( t ) - ( t - 1 ) / r \right] .\tag{6}
$$

Compared with classical AL, Eq. 6 retains the lag of every target step, including those emitted after the full source has been consumed (Fig. 2a). This removes the truncation error, but does not address the second limitation: the formulation remains target-centric and does not account for the physical duration over which the target tail is streamed.

## 4.2 Stream-AL

To account for the streaming duration of the target tail, we reformulate lag on a shared source-centric timeline (Fig. 2c). Rather than indexing latency by target emissions and asking how much source has been consumed, we index it by the fixed-rate source clock and ask how much target progress should have been completed at each source step. Since source chunks arrive every $w _ { \mathrm { c h u n k } }$ seconds, each source index i corresponds to a fixed wallclock interval.

At source step i, the proportional ideal has reached target position ir, whereas the wait-k schedule has reached $i - k$ . We allow $i - k < 0$ during the initial wait phase, interpreting it as a signed shortfall relative to the ideal rather than a physical number of emitted tokens. The gap $i r - ( i - k )$ therefore measures the target-progress deficit on the shared streaming timeline. When the target progresses more slowly than the proportional ideal, this deficit accumulates over time and naturally reflects the unfinished target tail at the source end.

Averaging this signed gap over the source timeline gives

$$
{ \begin{array} { r } { { \mathrm { S t r e a m } } { \mathrm { - A L } } = { \frac { 1 } { | \mathbf { x } | } } \sum _ { i = 1 } ^ { | \mathbf { x } | } { \big [ } i r - ( i - k ) { \big ] } } \\ { = k + { \frac { ( | \mathbf { x } | + 1 ) ( r - 1 ) } { 2 } } . } \end{array} }\tag{7}
$$

![](images/da2cc7cb71a7a64929a62832bed6d2a4c7cf2e45531a3110ab3c1b6b79cb56d8.jpg)  
Figure 3: In ca-Stream-AL, for source step $i ,$ the chunk arrives at $i \times w _ { \mathrm { c h u n k } }$ , but the model completes generation at $t _ { i } = \operatorname* { m a x } ( t _ { i - 1 } , i \times w _ { \mathrm { c h u n k } } ) + c _ { i }$ , where $c _ { i }$ includes VQ-VAE encoding(c<sup>(1)</sup>)/decoding(c<sup>(3)</sup>), transformer inference $( c _ { i } ^ { ( 2 ) } )$ ), and SMPL-X rendering $( c _ { i } ^ { ( 4 ) } )$ . The wallclock computation lag $\Delta _ { i } = ( t _ { i } - i \times w _ { \mathrm { c h u n k } } ) / w _ { \mathrm { c h u n k } }$ shifts the wait-k schedule from $i - k \mathrm { ~ t o ~ } i - k - \Delta _ { i }$ yielding ca-Stream-AL = Stream-AL + ∆<sup>¯</sup> .

Stream-AL is signed and may take negative values when $r \ < \ 1$ , indicating that the scheduled target progress is ahead of the proportional ideal. Since the gap in Eq. 7 is measured in target-chunk units and each target chunk spans $w _ { \mathrm { c h u n k } }$ seconds, multiplying by $w _ { \mathrm { c h u n k } }$ converts Stream-AL to seconds. Stream-AL captures only the lag induced by the streaming schedule and assumes instantaneous computation; we next account for the additional wall-clock delay introduced by inference and rendering.

## 4.3 Computation-aware extension.

Stream-AL captures the lag induced by the streaming schedule but assumes instantaneous computation. In practice, encoding, transformer inference, decoding and rendering introduce additional wallclock delay on the same shared timeline. We depict this extension in Fig. 3.

Let $c _ { i }$ denote the computation incurred at source step i, including VQ-VAE encoding and decoding, transformer inference, and SMPL-X rendering. Let $t _ { i }$ denote the wall-clock time at which step i finishes. Computation for step i can begin only after both source chunk i has arrived and the previous step has completed, giving

$$
t _ { i } = \operatorname* { m a x } ( t _ { i - 1 } , , i w _ { \mathrm { c h u n k } } ) + c _ { i } .\tag{8}
$$

Thus, any computation that exceeds the available time between source arrivals is carried forward as backlog.

We define the computation-induced delay at

source step i as

$$
\Delta _ { i } = \frac { t _ { i } - i w _ { \mathrm { c h u n k } } } { w _ { \mathrm { c h u n k } } } ,\tag{9}
$$

expressed in chunk-duration units. This delay includes both the computation of the current step and any backlog accumulated from previous steps. Since each target chunk spans the same $w _ { \mathrm { c h u n k } }$ seconds, a wall-clock delay of $\Delta _ { i } w _ { \mathrm { c h u n k } }$ corresponds to a backward displacement of $\Delta _ { i }$ target-chunk positions on the shared timeline. The nominal wait-k progress $i - k$ therefore becomes $i - k - \Delta _ { i }$

Substituting this computation-delayed progress into Eq. 7 gives

$$
\begin{array} { c } { \displaystyle \mathrm { c a - S t r e a m . A L } = \displaystyle \frac { 1 } { | \mathbf { x } | } \sum _ { i = 1 } ^ { | \mathbf { x } | } \left[ i r - ( i - k - \Delta _ { i } ) \right] } \\ { = \displaystyle \mathrm { S t r e a m . A L } + \bar { \Delta } , } \end{array}\tag{10}
$$

where $\begin{array} { r } { \bar { \Delta } = \frac { 1 } { | \mathbf { x } | } \sum _ { i = 1 } ^ { | \mathbf { x } | } \Delta _ { i } } \end{array}$ is the mean computationinduced delay. Thus, ca-Stream-AL decomposes into the schedule-induced lag measured by Stream-AL and the additional lag introduced by computation, and reduces to Stream-AL when $\Delta _ { i } \equiv 0$

## Pipeline boundary.

The full pipeline is: source video frames → SMPL-X regression into source SMPL-X motion → VQ-VAE encoding into source tokens → transformer inference from source tokens to target tokens → VQ-VAE decoding into target SMPL-X motion → SMPL-X rendering into visible avatar output. ca-Stream-AL measures latency from the source SMPL-X motion onward. Accordingly, VQ-VAE encoding and decoding, transformer inference, and the downstream SMPL-X-to-avatar rendering stage are included in the per-step computation cost $c _ { i }$

The upstream regression from raw video to SMPL-X, however, is treated as an external preprocessing stage. We assume that it operates at or above the 25 fps source rate, so that it does not accumulate a growing backlog relative to the incoming stream. This assumption is practical with modern pose regressors, several of which approach or exceed real-time throughput on contemporary GPUs. For example, OSX (Lin et al., 2023) reports 12.2 fps on V100, which scales to approximately 56 fps on H100 under the reported ∼4.6× transformer-inference speedup, while faster alternatives such as SMPLer-X (Cai et al., 2023) already exceed 25 fps on V100. Under this assumption, upstream regression contributes only a bounded processing offset rather than an accumulating delay, and is therefore excluded from ca-Stream-AL. If the regressor runs below the source frame rate, its backlog instead grows over time, in which case the reported ca-Stream-AL should be interpreted as a lower bound on the true end-to-end latency.

We retain the standard bidirectional encoder of Wu et al. (2026); under streaming inference, the available source prefix is re-encoded at each step, and this computation is included in $c _ { i }$

Relation to prior metrics. Stream-AL is related to duration-aware latency metrics such as ATD (Kano et al., 2023), but differs in its reference frame. ATD remains target-centric and measures delay with respect to corresponding input positions, whereas Stream-AL measures target progress on a shared source-time axis against the proportional ideal ir. This source-centric formulation accounts explicitly for the physical duration of the output stream and permits signed lag, including negative values when target progress is ahead of the proportional ideal.

The computation-aware extension is complementary to prior computation-aware AL formulations (Ma et al., 2020). Rather than attaching computation cost only to output emission, ca-Stream-AL models the wall-clock completion time of every streaming step. The per-step cost $c _ { i }$ therefore includes computation incurred even when no target token is emitted, such as repeated source-prefix encoding during the initial wait phase, together with decoding and rendering when output is produced. Through the recurrence in Eq. 8, any computation that exceeds the available wallclock budget is carried forward as backlog, allowing ca-Stream-AL to capture both per-step processing latency and its accumulation over the stream.

## 5 Experiments

## 5.1 Setup

Backbone. MBART-large-cc25 finetuned from the public pre-trained weights (§2); frozen VQ-VAE tokenizer from Zuo et al. (2024). We follow the public training config of Wu et al. (2026) (AdamW, lr $2 \times 1 0 ^ { - 4 }$ cosine-annealed to $1 0 ^ { - 6 }$ , 150 epochs, effective batch size 256, 2×H100).

Source frame rate. All three monolingual corpora (How2Sign, CSL-Daily, Phoenix-2014T) provide SMPL-X poses at 25 fps, and we adopt this rate for the source sign streams throughout. Combined with the VQ-VAE temporal downsampling factor of 4, one sign token spans 4 source frames = 4/25 = 0.16 s (160 ms); the effective token rate of the transformer is therefore $2 5 / 4 \ : = \ : 6 . 2 5 \ : \mathrm { H z }$ . The ca-Stream-AL reported below are in seconds along this 25 fps source timeline.

Training corpus. A freshly generated backtranslation(BT) S2S corpus built following the BT methodology of Wu et al. (2026).

Test sets. BT: held-out (text, sign) pairs run through our BT pipeline; synthetic source, gold target. This is distinct from the BT-input split of Wu et al. (2026) — same methodology, fresh built corpus. Strict: the dataset from Wu et al. (2026) (sign ↔ sign pairs collected from How2Sign, CSL-Daily, Phoenix-2014T with alignment using their text translaition).

Directions. All six $( s  s ^ { \prime } )$ with $s , s ^ { \prime } \in$ {ASL, CSL, DGS}, $s \neq s ^ { \prime } .$

Metrics. Quality: DTW-PA-MPJPE (primary, following Wu et al., 2026), computed with the fastdtw algorithm (Salvador and Chan, 2007) at radius r=2; and BLEU-4 via the Sign Language Transformer (Camgöz et al., 2020). Latency: ca-Stream-AL (§4).

Variants. Full-sentence: $k = | { \bf x } |$ baseline following Wu et al. (2026). TT-only: wait-k inference on the full-sentence checkpoint (§3.1); no retraining. Train+TT: stochastic multi-path with m=5 (§3.2).<sup>1</sup>

Latency sweep. k ∈ {1, 3, 5, 7, 9, 11, 13} source sign tokens, i.e. ≈ {0.16, 0.48, 0.80, 1.12, 1.44, 1.76, 2.08} s.

Artifacts and licenses. All upstream artifacts are used under their respective public licenses. MBART-large-cc25 (Liu et al., 2020), pyrender, and the fastdtw implementation (Salvador and Chan, 2007) are distributed under permissive open-source licenses; the decoupled VQ-VAE tokenizer (Zuo et al., 2024) and the OSX pose regressor (Lin et al., 2023) are released by their authors for research use. The three monolingual corpora (How2Sign, CSL-Daily, Phoenix-2014T) and the Strict subset (derived from Inan et al., 2025 via Wu et al., 2026) are distributed under researchonly / non-commercial terms. Our use of every artifact — as building blocks of a research system for sign-language translation — is consistent with these intended-use conditions. We will release our BT-synthesized S2S corpus and trained checkpoints under terms compatible with the strictest upstream license (research-only); downstream users should honor those restrictions, in particular the non-commercial / research-only terms of the source corpora.

<table><tr><td>Direction</td><td> $\overline { { \boldsymbol { k } ^ { \star } } }$ </td><td>AL (s) ↓</td><td>DTW↓</td><td>BLEU↑</td></tr><tr><td>ASL→CSL</td><td>7</td><td>1.97</td><td>3.74</td><td>7.5</td></tr><tr><td>CSL→ASL</td><td>9</td><td>3.71</td><td>4.79</td><td>10.4</td></tr><tr><td>ASL→DGS</td><td>7</td><td>1.15</td><td>3.85</td><td>7.4</td></tr><tr><td>DGS→ASL</td><td>9</td><td>2.41</td><td>3.55</td><td>9.0</td></tr><tr><td>CSL→DGS</td><td>7</td><td>1.34</td><td>2.88</td><td>7.5</td></tr><tr><td>DGS→CSL</td><td>9</td><td>3.05</td><td>3.27</td><td>7.4</td></tr></table>

Table 1: Best-tradeoff $k ^ { \star }$ per direction on BT under Train+TT, m=5. AL = ca-Stream-AL (seconds); DTW = DTW-PA-MPJPE; BLEU = BLEU-4 from the Sign Language Transformer evaluator. AL admits possible negative values for directions with $r < 1$

## 5.2 Quality–Latency Pareto Frontier

We sweep all six sign → sign directions; Fig. 4 shows CSL→DGS as a representative case, comparing TT-only and Train+TT (m=5) over k against the full-sentence k=|x| baseline in a $2 \times 2$ grid (rows: test sets BT and Strict; columns: DTW-PA-MPJPE and BLEU-4; x-axis: ca-Stream-AL). Across all four panels, Train+TT dominates TT-only at matched ca-Stream-AL (except for a small portion in Fig. 4c), confirming that exposing the model to source-prefix supervision recovers quality that pure test-time truncation leaves on the table. Concretely, at k=7 on CSL→DGS Train+TT reaches ca-Stream-AL = 1.34 s — a ∼ 3× reduction over the full-sentence checkpoint at 4.32 s — trailing full-sentence DTW-PA-MPJPE only by 0.19 and BLEU-4 by 2.4.

Per-direction summary. Tab. 1 and Tab. 2 report the best-tradeoff k per direction on the BT and Strict test sets respectively, with the resulting (ca-Stream-AL, DTW-PA-MPJPE, BLEU-4) under Train+TT, m=5.

## 6 Analysis: Word Order and Streaming Anticipation

Simultaneous translation surfaces a problem that the offline setting can ignore: the decoder cannot reorder content it has not yet observed. At low k, the wait-k policy can only wait (paying latency) or anticipate (risking error). This tension is as sharp for sign-to-sign as for text-to-text simultaneous MT because the canonical word orders of ASL, CSL, and DGS also differ substantially: ASL is broadly SVO with frequent topic-fronting; DGS is largely SOV, placing the verb at the clause end; CSL admits both SVO and SOV constructions and is topicprominent, and — unlike DGS — places sentential negation before the predicate it scopes over. A streaming CSL→DGS system must therefore delay clause-final material across earlier source tokens and, in the negative case, promote the adjective ahead of the negator.

<table><tr><td>Direction</td><td> $k ^ { \star }$ </td><td>AL (s) ↓</td><td>DTW↓</td><td>BLEU↑</td></tr><tr><td>ASL→CSL</td><td>7</td><td>2.88</td><td>2.53</td><td>5.5</td></tr><tr><td>CSL→ASL</td><td>9</td><td>1.78</td><td>2.09</td><td>8.5</td></tr><tr><td>ASL→DGS</td><td>7</td><td>2.11</td><td>2.02</td><td>5.3</td></tr><tr><td>DGS→ASL</td><td>9</td><td>1.79</td><td>1.52</td><td>7.6</td></tr><tr><td>CSL→DGS</td><td>7</td><td>1.29</td><td>2.39</td><td>5.6</td></tr><tr><td>DGS→CSL</td><td>9</td><td>2.94</td><td>3.72</td><td>6.3</td></tr></table>

Table 2: Best-tradeoff $k ^ { \star }$ per direction on Strict under Train+TT, m=5. Columns as in Tab. 1.

We illustrate both outcomes with two CSL→DGS picks from the Strict subset of Wu et al. (2026), decoded by Train+TT at k=3 (Fig. 5). In each panel, we show the source signing track, the gloss of each source segment, the model’s predicted target signing, and the reference DGS signing and gloss.

Good case (Fig. 5a). The CSL source glosses to “mountain / color / snow-white” (shanshàng¯ / yánsè / xuebai), a topic-comment construction in which “color” is a discourse marker introducing the descriptor. The reference DGS gloss is BERG IX SCHNEE (mountain IX snow); the indexical IX has no lexical correspondent in the source. The model emits BERG SCHNEE: it preserves the canonical mountain-then-snow order shared by both languages, decomposes the CSL compound “snowwhite” into the DGS root SCHNEE, and correctly drops the source-side “color” that does not surface in DGS. This is the regime where streaming costs little: anticipation is not required because the two canonical orders agree on the content sequence.

Bad case (Fig. 5b). The CSL source glosses to “not cold” (bu leng), with the negator first. The reference DGS gloss is KALT NEG-NICHTS (cold not): DGS places sentential negation after the predicate it scopes over. A faithful streaming policy at k=3 would have to either wait for the adjective (“cold”) to arrive before committing to anything — foregoing the latency advantage of streaming — or anticipate the adjective from a single negation sign, which is under-determined. (Noted: To avoid confusion, k in this paper always refers to sign tokens rather than the gloss unit. So wait-3 means waiting for three sign tokens when we can not even get the meaning of "cold" at the beginning.) The model takes neither path: it emits NEG-NICHTS KALT the moment the negator is observed, copying source order and violating the DGS canonical order on a two-token utterance. The miss is exactly the failure mode Ma et al. (2019) predict for short clauses whose reordering window exceeds k.

![](images/533f37d0644b8afa8509e381204dba4e8550fafdad1132e6fe181bd8e9820535.jpg)

(a) BT, DTW-PA-MPJPE (↓)  
![](images/f5b0d0b381df97d44fe41d9019733fb8bc259db5d4229068e45de99cf5158a96.jpg)

![](images/b8abaac1fcda9c1cd6e9484a156aa9429f324c88df7e0ba2650cd9bcf66e9ac0.jpg)

(c) Strict, DTW-PA-MPJPE (↓)  
(b) BT, BLEU-4 (↑)  
![](images/36a48a2b095739f4734967041c8bab39ce090b272f3e9d2d6b10df568f64376d.jpg)  
(d) Strict, BLEU-4 (↑)  
Figure 4: Quality–latency trade-off on CSL→DGS, a representative direction out of the six sign → sign directions we evaluate. Test-time wait-k (blue) and trained wait-k (orange) traced over $k \in \{ 1 , 3 , 5 , 7 , 9 , 1 1 , 1 3 \}$ ; the fullsentence k=∞ baseline (green star) is connected by a dashed segment. Rows: test set (BT, Strict); columns: quality metric (DTW-PA-MPJPE, BLEU-4); x-axis: ca-Stream-AL (seconds).

Takeaway. The two picks bracket the regime: when source and target canonical orders agree over content (Fig. 5a), small-k streaming reproduces the target sequence and even resolves source-only function words; when they disagree on a short, clause-internal swap (Fig. 5b), the wait/anticipate tradeoff is unfavorable at k=3 and the model defaults to source order. A quantitative version of this analysis — per-direction anticipation lead time on a gloss-tagged subset — is left to future work and noted in Limitations.

## 7 Related Work

Simultaneous machine translation. We adopt the prefix-to-prefix framework of Ma et al. (2019), which factorizes streaming generation as p(y<sub>t</sub> | $x _ { \leq g ( t ) } , y _ { < t } )$ for a monotonic schedule g. The waitk policy is the special case $g ( t ) = k { + } t { - } 1$ . AL (Ma et al., 2019), AP (Cho and Esipova, 2016), and DAL (Cherry and Foster, 2019) provide the classical text-MT latency framework; we follow Ma et al. (2019) in using greedy decoding (beam search is ill-defined under partial-source observation). Ma et al. (2019) is text-to-text; we generalize the framework to sign-token chunks on a multilingual sign backbone, introduce ca-Stream-AL (§4) for streaming output in which each emitted chunk has a fixed wall-clock duration, and contribute a stochastic multi-path training objective (§3.2) that makes prefix-to-prefix tractable on the bidirectional encoder of the underlying direct-S2S model. Elbayad et al. (2020) train a single model that serves multiple k at inference, sidestepping the per-k retraining cost discussed in our Limitations.

![](images/77298ff3b7a4c1fe7b45373c812222d3ca6ebf129f7a88186f6989ba766102ee.jpg)  
(b) Reordered negation: CSL neg–adj → DGS adj–neg.  
Figure 5: CSL→DGS case study (Train+TT, k=3). (a) When the canonical source and target orders align over content words, the model emits the right DGS sequence and drops the CSL topic-marker glossed “color” (yanse) that has no DGS counterpart. (b) When the two languages disagree on the position of the negator, the streaming model emits source-order NEG-NICHTS KALT as soon as the source negator (bu, “not”) arrives, instead of the canonical DGS order KALT NEG-NICHTS.

Duration-aware latency metrics. Kano et al. (2023) introduces ATD, which considers targettoken duration in delay accounting and is the closest prior metric in spirit to ca-Stream-AL. ATD is target-centric (per-target-token delay against the corresponding clipped input index) and conventionally non-negative; ca-Stream-AL is source-centric, uses the proportional diagonal $\begin{array} { r } { i \times r \ : ( r \ : = \ : \frac { | \mathbf { y } | } { | \mathbf { x } | } ) } \end{array}$ as reference, admits a closed form for wait-k, and permits negative values when $r ~ < ~ 1$ . Ma et al. (2020) extends wait-k from text-to-text to streaming speech-to-text translation and introduces computation-aware AL (CA-AL); ca-Stream-AL applies the same computation-aware idea to our source-centric formulation.

Streaming sign-language translation (sign → text). Yin et al. (2021) introduce SimulSLT, which applies wait-k to full-sentence sign-to-text translation (Camgöz et al., 2018, 2020) with a learned gloss-boundary predictor that decides when to read versus emit; their target is spoken-language text, <sup>2</sup>and latency is measured by classical (target-centric) AL on text tokens. Our setting differs along two axes: the target is rendered sign output emitted in fixed wall-clock chunks rather than text tokens at an instant — which is what motivates the sourcecentric, computation-aware ca-Stream-AL (§4) — and the source itself is a sign stream, making the present work, to our knowledge, the first streaming sign → sign system in the literature.

Text → sign (T2S) translation. T2S translation (or in some literature called sign language production) generates sign output from spoken-language text, either as continuous pose sequences with progressive Transformers (Saunders et al., 2020) or via motion-graph and GAN-based pipelines (Stoll et al., 2020). T2S translation is offline and consumes text; we generate sign tokens conditioned on a sign-language source under a streaming policy, inheriting the prefix-to-prefix latency budget rather than full-source access.

## 8 Conclusion

We presented, to our knowledge, the first simultaneous S2S translation system, closing the gap between offline S2S models and the live deployments — broadcast interpretation, two-way video calls — that require target signing to begin while the source signer is still signing. We adapted the wait-k policy to sign-token streams in two regimes — test-time inference on a full-sentence backbone and a trained wait-k model via stochastic multi-path supervision — and introduced ca-Stream-AL, a source-centric computation-aware latency metric that accounts for the fixed wall-clock duration of each emitted sign chunk, which target-centric AL and CA-AL under-report. Averaged across the six S2S directions, our streaming system cuts ca-Stream-AL by 38% while staying within a 9% DTW-PA-MPJPE increase and a 2.1 BLEU-4 drop relative to the full-sentence baseline. A word-order case study surfaced the central new challenge of going simultaneous: the decoder cannot reorder content it has not yet observed, so at low k it must trade latency for anticipation on directions whose canonical orders disagree. We hope ca-Stream-AL and the streaming-S2S task spur further work toward real-time cross-lingual sign communication.

## Limitations

Length-ratio-agnostic schedule. Both wait-k regimes (§3) use the fixed unit-slope schedule $\scriptstyle g ( t ) = k + t - 1 :$ : after the initial k-token wait, exactly one source token is revealed per emitted target. The six directions, however, have source/target length ratios far from 1 $\begin{array} { r } { \mathbf { \rho } ( r = \mathrm { \frac { | \mathbf { y } | } { | \mathbf { x } | } } \mathrm { ~ f r o m } \approx 0 . 5 2 } \end{array}$ to $\approx 2 . 4 4 ; \ S 3 . 1 )$ , so a unit-slope read/write schedule drifts linearly from the proportional ideal $\textit { i } \times \textit { r }$ — the $\frac { ( | \mathbf { x } | + 1 ) \mathbf { \bar { ( } } r - 1 ) } { 2 }$ term that dominates Stream-AL (Eq. 7) when r is far from 1. A length-ratio-aware catch-up policy (Ma et al., 2019) would instead match the read:write rate to the empirical ratio: for a direction whose source runs ≈ 1.2× the target (≈ 5 target tokens per 6 source tokens), it reveals two source tokens before one of every six writes — e.g. the fifth and sixth source tokens are read together — keeping the schedule parallel to the slope-r ideal rather than slope 1, and symmetrically writes two targets per read when the target is the longer side. We use the ratio-agnostic schedule for simplicity and direct comparability with text-MT wait-k; biasing the schedule by the measured per-direction ratio is left to future work and would tighten Stream-AL most on the highly asymmetric directions.

Single-k checkpoint. Each Train+TT model commits to one value of k at training time; serving a different latency budget at inference requires retraining. Multi-k schemes (Elbayad et al., 2020) sidestep this but at additional implementation cost — left to future work.

Stochastic multi-path hyperparameters. We use uniform position sampling with m=5 replicas for the stochastic multi-path objective (§3.2). Neither m nor the sampling distribution (e.g., geometric, early-position-biased, schedule-dependent) was tuned; both could shift the early-segment quality at low k.

Synthetic training data. Train+TT is supervised on back-translation (BT) synthetic S2S pairs (§5.1): the source side is generated by the T2S direction of the same backbone and so inherits its biases.

Whether these biases interact with the streaming objective — e.g., by encouraging source-order anticipation when the target order would differ — has not been characterized.

Qualitative anticipation analysis. §6 inspects a small number of cases. A quantitative anticipation metric $( \mathrm { e . g . , }$ source-aligned verb-emission lead time on a gloss-tagged subset) is left to future work.

No human evaluation. Quality is reported via DTW-PA-MPJPE and a downstream-evaluator BLEU-4; native-signer human evaluation of streamed output is left to future work and would be the right tool for assessing perceived choppiness or unnatural early-segment artefacts.

Rendering bottleneck. Our SMPL-X-to-avatar rendering pipeline (pyrender) runs at ≈ 30 fps, contributing ≈ 132 ms per output chunk to ca-Stream-AL— by far the dominant per-output cost, against ∼ 10 ms for the encoder forward and LM decoder step combined. Productiongrade real-time avatar renderers (game-engine or neural-rendering systems) run substantially faster; swapping in such a renderer would tighten ca-Stream-AL numerically without changing the streaming policy. A render-aware system design (e.g., overlapped render and decode) is outside this paper’s scope.

Per-step re-encoding cost. Schedule construction in §3.1 re-runs the bidirectional encoder on each successive source prefix and so scales encoder work as $\mathcal { O } ( | \mathbf { x } | ^ { 3 } )$ (matching §3.1). This is tolerable today because rendering dominates the per-chunk cost, but the constraint binds once rendering speeds up; a causal-mask approximation would amortize the cost at the price of discarding the bidirectional prior of the pre-trained backbone.

SMPL-X input assumption. We assume the upstream raw-video → SMPL-X pose-estimation front-end runs at ≥ 25 fps (matching the source frame rate); under this assumption extraction is a sub-chunk additive offset (§4), but below 25 fps a backlog accumulates linearly with |x| and our ca-Stream-AL becomes a lower bound on real userperceived latency. The OSX regressor (Lin et al., 2023) we adopt and faster alternatives such as the SMPLer-X family (Cai et al., 2023) comfortably clear 25 fps on modern GPUs (§4), but a sustained deployment over long inputs would need to verify this empirically.

Greedy decoding. Following Ma et al. (2019), we use greedy decoding because beam search is ill-defined under partial-source observation. This leaves quality on the table relative to the (nonstreaming) beam-search regime and is a shared limitation of the simultaneous-MT literature rather than specific to our work.

No external streaming-S2S baseline. To our knowledge there is no published simultaneous S2S system; we therefore compare only against our own full-sentence backbone and the test-time wait-k inference variant. Stronger external comparison will be possible once additional simultaneous S2S systems appear.

Coverage. Three sign languages (ASL / CSL / DGS); minority sign languages are not represented. Streaming systems should be deployed with care for community-specific norms.

## References

Vasileios Baltatzis, Rolandos Alexandros Potamias, Evangelos Ververas, Guanxiong Sun, Jiankang Deng, and Stefanos Zafeiriou. 2024. Neural sign actors: A diffusion model for 3d sign language production from text. In Proceedings ofthe IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 1985–1995.

Zhongang Cai, Wanqi Yin, Ailing Zeng, Chen Wei, Qingping Sun, Yanjun Wang, Hui En Pang, Haiyi Mei, Mingyuan Zhang, Lei Zhang, Chen Change Loy, Lei Yang, and Ziwei Liu. 2023. SMPLer-X: Scaling up expressive human pose and shape estimation. In Advances in Neural Information Processing Systems (NeurIPS) Datasets and Benchmarks Track.

Necati Cihan Camgöz, Simon Hadfield, Oscar Koller, Hermann Ney, and Richard Bowden. 2018. Neural sign language translation. In Proceedings ofthe IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 7784–7793.

Necati Cihan Camgöz, Oscar Koller, Simon Hadfield, and Richard Bowden. 2020. Sign language transformers: Joint end-to-end sign language recognition and translation. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 10023–10033.

Colin Cherry and George Foster. 2019. Thinking slow about latency evaluation for simultaneous machine translation. arXiv preprint arXiv:1906.00048.

Kyunghyun Cho and Masha Esipova. 2016. Can neural machine translation do simultaneous translation? arXiv preprint arXiv:1606.02012.

Maha Elbayad, Laurent Besacier, and Jakob Verbeek. 2020. Efficient wait-k models for simultaneous machine translation. In Proceedings of Interspeech 2020.

Mert Inan, Yang Zhong, Vidya Ganesh, and Malihe Alikhani. 2025. How to align multiple signed language corpora for better sign-to-sign translations? In Proceedings ofthe 2025 Conference ofthe Nations of the Americas Chapter ofthe Associationfor Computational Linguistics: Human Language Technologies (Long Papers), pages 4003–4016. Association for Computational Linguistics.

Yasumasa Kano, Katsuhito Sudoh, and Satoshi Nakamura. 2023. Average token delay: A duration-aware latency metric for simultaneous translation. In Proceedings ofInterspeech 2023.

Jing Lin, Ailing Zeng, Haoqian Wang, Lei Zhang, and Yu Li. 2023. One-stage 3d whole-body mesh recovery with component aware transformer. In Proceedings ofthe IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR).

Yinhan Liu, Jiatao Gu, Naman Goyal, Xian Li, Sergey Edunov, Marjan Ghazvininejad, Mike Lewis, and Luke Zettlemoyer. 2020. Multilingual denoising pretraining for neural machine translation. Transactions ofthe Associationfor Computational Linguistics, 8:726–742.

Mingbo Ma, Liang Huang, Hao Xiong, Renjie Zheng, Kaibo Liu, Baigong Zheng, Chuanqiang Zhang, Zhongjun He, Hairong Liu, Xing Li, Hua Wu, and Haifeng Wang. 2019. STACL: Simultaneous translation with implicit anticipation and controllable latency using prefix-to-prefix framework. In Proceedings ofthe 57th Annual Meeting ofthe Association for Computational Linguistics, pages 3025–3036. Association for Computational Linguistics.

Xutai Ma, Juan Pino, and Philipp Koehn. 2020. SimulMT to SimulST: Adapting simultaneous text translation to end-to-end simultaneous speech translation. In Proceedings of the 1st Conference of the Asia-Pacific Chapter ofthe Associationfor Computational Linguistics and the 10th International Joint Conference on Natural Language Processing, pages 582–587. Association for Computational Linguistics.

Stan Salvador and Philip Chan. 2007. Toward accurate dynamic time warping in linear time and space. Intelligent Data Analysis, 11(5):561–580.

Ben Saunders, Necati Cihan Camgöz, and Richard Bowden. 2020. Progressive transformers for end-to-end sign language production. In Proceedings of the European Conference on Computer Vision (ECCV), pages 687–705.

Rotem Shalev-Arkushin, Amit Moryossef, and Ohad Fried. 2022. Ham2pose: Animating sign language notation into pose sequences. arXiv preprint arXiv:2211.13613.

Stephanie Stoll, Necati Cihan Camgöz, Simon Hadfield, and Richard Bowden. 2020. Text2Sign: Towards sign language production using neural machine translation and generative adversarial networks. International Journal ofComputer Vision, 128:891–908.

Zetian Wu, Bowen Xie, Wuyang Meng, Milan Gautam, Stefan Lee, and Liang Huang. 2026. Direct translation between sign languages. arXiv preprint arXiv:2605.20588.

Aoxiong Yin, Zhou Zhao, Jinglin Liu, Weike Jin, Meng Zhang, Xingshan Zeng, and Xiaofei He. 2021. Simul-SLT: End-to-end simultaneous sign language translation. In Proceedings ofthe 29th ACM International Conference on Multimedia (ACM MM), pages 4118– 4127.

Zhengdi Yu, Shaoli Huang, Yongkang Cheng, and Tolga Birdal. 2024. Signavatars: A large-scale 3d sign language holistic motion dataset and benchmark. In Proceedings of the European Conference on Computer Vision (ECCV), pages 1–19.

Linlin Zhang, Kai Fan, Jiajun Bu, and Zhongqiang Huang. 2023. Training simultaneous speech translation with robust and random wait-k-tokens strategy. In Proceedings ofEMNLP, pages 7814–7831.

Renjie Zheng, Mingbo Ma, Baigong Zheng, Kaibo Liu, Jiahong Yuan, Kenneth Church, and Liang Huang. 2020. Fluent and low-latency simultaneous speechto-speech translation with self-adaptive training. In Findings ofthe Associationfor Computational Linguistics: EMNLP 2020, pages 3928–3937, Online. Association for Computational Linguistics.

Ronglai Zuo, Rolandos Alexandros Potamias, Evangelos Ververas, Jiankang Deng, and Stefanos Zafeiriou. 2024. Signs as tokens: A retrieval-enhanced multilingual sign language generator. arXiv preprint arXiv:2411.17799. To appear at ICCV 2025.
# Summarize Before Grounding: Query-Guided Chunk Condensation for Long-Video Temporal Grounding

Nanxing Hu<sup>1</sup>, Xiaoyue Duan, Qiwei Yan<sup>2</sup>, Kailin Lyu<sup>3</sup>, Jinchao Zhang<sup>∗</sup>, Guoliang Kang<sup>1∗</sup>

<sup>1</sup>Beihang University

<sup>2</sup>University of the Chinese Academy of Sciences

<sup>3</sup>Institute of automation, Chinese academy of science

## Abstract

Video temporal grounding (VTG) aims to localize the video interval corresponding to a language query. Recent large vision-language models (LVLMs) show great potential in solving such a multi-modal reasoning task. However, long videos often contain large amounts of redundant information that disturbs LVLMs to mine query-relevant evidence. Instead of dense frame sampling which incurs prohibitive training memory, previous reinforcement learning with verifiable rewards (RLVR) works typically utilize sparse sampling, which makes training feasible but may miss critical evidence. In this paper, we propose a “summarize before grounding” framework (named “SumGround”) for long-video temporal grounding. The key of SumGround is to perform query-guided chunk condensation to aggregate and retrieve query-relevant evidence. Specifically, we split the video into several chunks and perform two-level chunk condensation. First, we introduce query-guided latent summaries, which is represented as KV states of query-guided prompts, to compress redundant visual tokens into compact query-relevant chunk summaries. Furthermore, we design an associative summary retrieval scheme to rank and select chunk summaries that are most likely to contain the event interval. Both query-guided latent summary and associative summary retrieval schemes are enabled by RLVR. To reduce memory consumption, we propose a length-aware gradient gating module to selectively stop gradient back-propagated to visual tokens. Extensive experiments demonstrate that SumGround performs favorably against previous state-of-the-art methods across multiple downstream datasets, with remarkable gains on long videos.

## Introduction

Video temporal grounding (VTG) aims to localize the video interval corresponding to a language query (Gao et al. 2017; Zhang et al. 2023). In recent years, large vision-language models (LVLMs), due to their superior multi-modal reasoning capability, have been widely adopted to solve the VTG problem (Huang et al. 2024; Wang et al. 2024). However, for a long video that is quite common in practice, the queried event may occupy only a small portion, which means that a large amount of content is redundant. The redundant information may mislead LVLM, resulting in imprecise grounding boundary predictions.

Reinforcement learning with verifiable rewards (RLVR) has been widely adopted to post-train a LVLM to improve its VTG performance. Existing RLVR-based VTG methods typically rely on sparse frame sampling because retaining dense visual tokens and their gradients over long videos is prohibitively expensive. As illustrated in Figure 1(a), sparse sampling reduces the number of input frames (and thus reduces the number of tokens consumed) but may miss critical evidence relevant to query. Dense sampling covers queryrelevant content, but also contains massive redundant information. Moreover, retaining all visual embeddings and their gradients in RLVR incurs prohibitively high training memory cost. Therefore, the key challenge is to preserve queryrelevant evidence, filter out irrelevant information, and eficiently optimize the resulting long-range representation for VTG.

In this paper, we propose a “summarize before grounding” framework (named “SumGround”) for long-video temporal grounding. The key of SumGround is to perform queryguided chunk condensation to aggregate and retrieve queryrelevant evidence. As shown in Figure 1(b), we perform twolevel context condensation. First, we split the video into several chunks and process them sequentially through multiple prefill passes. For each prefill, rather than generating a textual summary or introducing additional modules, we propose using query-guided latent summary, which is represented as KV states of query-guided prompts, to compress redundant visual tokens into compact query-relevant chunk summaries. The latent summary of the current chunk attends to current chunk visual tokens and previous chunk summaries. We store the KV states of summary tokens instead of the KV states of visual tokens. Furthermore, we design an associative summary retrieval scheme to rank and select chunk summaries that are most likely to contain the event interval. In detail, during the chunk-wise prefill stage, we utilize the predicted logits of the query-association prompt to judge whether the current chunk is associated with the query. After all video chunks are processed, associative summary retrieval retains query-associated summaries and filters out unrelated ones, constructing a concentrated context for precise grounding.

For training, both query-guided latent summary and associative summary retrieval schemes are enabled by RLVR. We propose a length-aware gradient gating module to reduce memory consumption. Specifically, our gradient gating strategy keeps the forward process unchanged and selectively stops gradients back-propagated to visual tokens based on the input video length. For short videos, visual-token gradients are retained, allowing the model to learn precise visual token hidden states and the vision-to-summary capability. For long videos, visual-token gradients are blocked to save memory, while the model continues to learn how summaries are accumulated, retrieved, and exploited for temporal grounding over long temporal contexts.

![](images/52501869d2f2a03af64d798f5e5e7fd184593d4541b13105df4b55a5973dc80e.jpg)  
Figure 1: Overview of the motivation and SumGround. (a) Dense sampling retains fine-grained evidence but also introduce extensive query-irrelevant content and prohibitive training memory, whereas sparse sampling may miss key evidence. (b) SumGround processes the video chunk by chunk, compresses query-relevant evidence into sequential summary states, and retains query-associated summaries for final grounding. We gate visual-token gradients according to video length while continuing to optimize summary accumulation, retrieval, and grounding.

We instantiate RLVR with Group Relative Policy Optimization (GRPO) (Shao et al. 2024), with separate rewards for associative summary retrieval and temporal grounding. For experiments, we train LVLM on constructed training set and perform zero-shot testing on four typical VTG benchmarks, i.e., Charades-STA (Gao et al. 2017), ActivityNet-Captions (Krishna et al. 2017), TACoS (Regneri et al. 2013), and Ego4D-NLQ (Grauman et al. 2022). Experiments show that SumGround performs favorably against previous methods, with remarkable gains on the long videos. Ablations verify the efectiveness of key components of our method.

In a nutshell, our contributions are summarized as follows

• We propose a novel framework, i.e., “summarize before grounding”, to perform query-guided chunk condensation to aggregate and retrieve query-relevant evidence for long-video temporal grounding. Our framework introduces no additional parameterized modules and can be trained end-to-end by RLVR.

• We propose two-level chunk condensation techniques, including query-guided latent summary and associative summary retrieval. For training, we design GRPO rewards respectively for associative summary retrieval and temporal grounding. Further, we propose a length-aware gradient gating module to reduce memory consumption.

• Experiments demonstrate that our method performs favorably against previous state-of-the-art methods, especially on long videos. Ablations verify the efectiveness of the key components of our framework.

## Related Work

Temporal Grounding with LVLMs. Video temporal grounding aims to localize the temporal span corresponding to a natural-language query (Gao et al. 2017; Wang et al. 2024). Earlier methods typically combine pretrained video– language representations (Radford et al. 2021; Devlin et al. 2019) with dedicated temporal localization architectures, including proposal-based (Zhang et al. 2020), regressionbased (Yuan, Mei, and Zhu 2019), DETR-style (Lei, Berg, and Bansal 2021), and unified clip-level prediction models (Lin et al. 2023). Recent LVLM-based methods instead formulate grounding as generative timestamp prediction and improve temporal awareness through normalized timestamp prediction (Huang et al. 2024; Wang et al. 2024), temporal cues injected into visual tokens (Ren et al. 2024; Guo et al. 2025), or explicit textual timestamps (Yuan et al. 2025). These methods primarily improve temporal representation and timestamp decoding, but pay less attention to how queryrelevant visual evidence can be identified from long videos and organized into a focused context for precise boundary prediction.

Long-Video Processing and Temporal Evidence Selection. Scaling LVLMs to long videos is challenging because dense visual tokens quickly exhaust practical context and memory budgets. Dedicated long-video LVLMs improve efficiency through token-eficient encoders (Li, Wang, and Jia 2024; Ren et al. 2024), memory or summarization tokens (Song et al. 2024; Shu et al. 2025), and spatiotemporal compression (Shen et al. 2024). Related long-context and retrieval-based methods further reduce the efective context through chunk compression or selective retrieval (Chevalier et al. 2023; An et al. 2025; Lin et al. 2025). However, these capabilities often require dedicated modules or substantial additional training, and are not explicitly designed to retain the fine-grained evidence needed for precise temporal boundaries.

Recent long-video temporal grounding methods built on general-purpose LVLMs adopt grounded instruction tuning (Zeng et al. 2024), SFT-based coarse-to-fine localization (Li et al. 2025), or agentic localization workflows (Liu et al. 2025). Nevertheless, task-specific tuning may weaken the broader capabilities of the base LVLM, while coarse-tofine localization remains constrained by its initial proposals. These limitations motivate accumulating and selecting query-relevant evidence across the full video before predicting the final temporal boundaries.

![](images/0750a01bebfc5565fa06c67b0191480ba8710dc65effe8facee1b7b051588f29.jpg)  
Figure 2: Framework of SumGround. (a) For each video chunk $C _ { i }$ , the LVLM jointly writes a query-guided summary state $S _ { i }$ and estimates its query-association score $\Delta _ { i } ,$ , conditioned on previously retained summaries. Only the system and summary KV states are propagated to subsequent chunks, while dense visual states and the states used for association estimation are discarded. (b) After all chunks are processed, SumGround selects an anchor and retrieves a contiguous set of query-associated summaries according to their association scores. The retrieved summaries are used to generate the final temporal grounding boundaries.

RLVR for Temporal Grounding. Recent RLVR-based methods optimize LVLMs for temporal grounding with verifiable rewards derived from temporal overlap (Wang et al. 2025; Zhang et al. 2025a). This paradigm directly aligns timestamp generation with grounding quality and can better preserve the general capabilities of LVLMs (Wang et al. 2025). Nevertheless, existing RLVR methods typically optimize timestamp generation from a fixed, sparsely sampled visual context. They leave open how query-relevant evidence can be accumulated from dense long-video chunks while keeping RLVR optimization memory-eficient.

## Method

Given a video and a language query q, VTG aims to predict the temporal interval $[ \tilde { t } _ { s } , \tilde { t } _ { e } ]$ corresponding to the queried event. An LVLM-based temporal grounding process can be viewed as a multimodal prefill to gather information, followed by autoregressive decoding to generate the answer. As illustrated in Figure 2, SumGround replaces a single fullvideo prefill with sequential video chunk prefills that retain compact query-guided latent summaries and record queryassociation scores. After all video chunks are processed, the association scores are used to retrieve a focused summary context for final decoding. Length-aware gradient gating further enables RLVR training on long videos by selectively blocking visual-token gradients based on video length.

## SumGround Overview

Given a video of duration $T ,$ we divide it into $K$ consecutive video chunks $\{ C _ { i } \} _ { i = 1 } ^ { K }$ . Each chunk $C _ { i }$ contains the frames within the temporal interval $[ \tau _ { i - 1 } , \tau _ { i } )$ , where $i \in \{ 1 , . . . , K \} , \tau _ { 0 } = 0 , \bar { \tau _ { K } } = T$ . Following (Yuan et al. 2025), we insert the corresponding timestamp text after each frame.

Let $\mathcal { F } _ { \boldsymbol { \theta } } ^ { \mathrm { p r e f i l l } }$ and $\mathcal { F } _ { \theta } ^ { \mathrm { g e n } }$ denote the prefill and autoregressive decoding stages of the LVLM parameterized by θ, respectively. For the i-th prefill pass, let $X _ { i }$ denote the new input constructed for chunk $C _ { i } ,$ and let $\mathcal { H } _ { i - 1 }$ denote the KV cache retained from preceding passes. The chunk-wise prefill is

$$
\left( \operatorname { K V } _ { i } , \mathbf { z } _ { i } \right) = \mathcal { F } _ { \theta } ^ { \mathrm { p r e f i l l } } \left( \mathcal { H } _ { i - 1 } , X _ { i } \right) ,\tag{1}
$$

where $\mathrm { K V } _ { i }$ denotes the newly computed KV states of $X _ { i }$ and $\mathbf { z } _ { i }$ denotes the output logits of the current pass. After each prefill, SumGround retains the query-guided summary states from $\mathrm { K V } _ { i }$ , and derives the query-association score $\Delta _ { i }$ from $\mathbf { z } _ { i }$ . The retained query-guided summary states are used to update the KV cache from $\mathcal { H } _ { i - 1 }$ to $\mathcal { H } _ { i }$ which is passed to the prefill pass for the next chunk i+1. The association scores are collected across chunks for later retrieval. No autoregressive decoding is performed until all chunk prefills are completed.

After all K chunks have been processed, the association scores are used to retrieve a concentrated grounding context H<sup>∗</sup>. The temporal boundaries are then decoded as

$$
\left( \hat { t } _ { s } , \hat { t } _ { e } \right) = \mathcal { F } _ { \theta } ^ { \mathrm { g e n } } \left( \mathcal { H } ^ { * } , p _ { \mathrm { f m t } } \right) .\tag{2}
$$

where $p _ { \mathrm { f m t } }$ is the output format prompt.

The following subsections describe how query-guided summary states are constructed and exploited, how association scores are estimated, and how estimated association scores guide the retrieval of the final grounding context $\mathcal { H } ^ { \ast }$

## Query-Guided Latent Summary

Query-Guided Prompt KV states as latent summary. Recent analyses (Zhang et al. 2025b; Yin, Si, and Wang 2025) show that visual evidence is progressively integrated into the contextualized KV states of prompt tokens. During subsequent generation, later tokens then rely primarily on these prompt KV states, rather than relying on the original visualtoken states. This suggests that KV states of prompt tokens can naturally serve as a compact latent representation of visual evidence. Based on this observation, we append a queryguided summary prompt $p _ { \mathrm { s u m } } ( \boldsymbol { q } )$ after each video chunk, $e . g .$ , “Summarize the video clip to accurately pinpoint the event ‘chop the onions’ and determine its precise temporal interval.”, where ‘chop the onions’ is the content of query. The prompt asks the LVLM to summarize the visual evidence relevant to the queried event. Without requiring timeconsuming autoregressive textual summary generation or an additional summarization module, we directly retain the KV states of $p _ { \mathrm { s u m } } ( \boldsymbol { q } )$ as latent summaries:

$$
S _ { i } = \mathrm { K V } _ { i } \left[ p _ { \mathrm { s u m } } ( q ) \right] ,\tag{3}
$$

Although the prompt text is shared across chunks, its contextualized KV states depend on the current visual tokens and the KV states of previous summaries, allowing each $S _ { i }$ to encode diferent evidence. The ablation results in Table 2 further verify this design.

Chunk-wise summary construction. Let $p _ { s }$ denote the system prompt and $p _ { \mathrm { a s s o c } }$ denote the association prompt detailed in the next subsection. The new input for each prefill pass is

$$
X _ { i } = \left\{ \begin{array} { l l } { [ p _ { s } , C _ { 1 } , p _ { \mathrm { s u m } } ( q ) , p _ { \mathrm { a s s o c } } ] , } & { i = 1 , } \\ { [ C _ { i } , p _ { \mathrm { s u m } } ( q ) , p _ { \mathrm { a s s o c } } ] , } & { i > 1 . } \end{array} \right.\tag{4}
$$

The first pass is performed without a preceding cache, whereas each subsequent input $X _ { i }$ is prefilled conditioned on the cache $\mathcal { H } _ { i - 1 }$ retained from earlier passes.

From the resulting KV states, we retain the system-prompt states $s = \mathrm { K V _ { 1 } } [ p _ { s } ]$ ] from the first pass and the summary states $S _ { i } = \mathrm { K V } _ { i } [ p _ { \mathrm { s u m } } \bar { ( q ) } \bar { }$ ] from each pass. After processing vision chunk $C _ { i }$ , the cache becomes

$$
{ \mathcal { H } } _ { i } = \{ s , S _ { 1 } , \ldots , S _ { i } \} .\tag{5}
$$

Thus, only the system and query-guided summary states are forwarded across chunks, while the visual states are discarded. After all K chunks have been processed, SumGround obtains the summary bank $\{ S _ { i } \} _ { i = 1 } ^ { K }$ , storing compact queryrelevant evidence aggregated from the whole video.

## Associative Summary Retrieval and Grounding

Query-association estimation. Although the summaries are query-guided, summaries from less related chunks may still mislead the grounding. Rather than introducing a separate retriever or repeatedly invoking the model to refine the temporal interval, we estimate the association between chunk memory and query within the same prefill pass. Specifically, the query-association prompt $p _ { \mathrm { a s s o c } }$ asks whether the current chunk contains evidence associated with the queried event. Because $p _ { \mathrm { a s s o c } }$ follows $p _ { \mathrm { s u m } } ( q )$ under causal attention, its computation does not afect the summary states $S _ { i }$

Let $\mathbf { z } _ { i }$ denote the next-token prediction logits of the last token of $p _ { \mathrm { a s s o c } }$ . We define the query association score as the logit margin between Yes and No:

$$
\Delta _ { i } = [ \mathbf { z } _ { i } ] _ { v _ { \mathrm { y e s } } } - [ \mathbf { z } _ { i } ] _ { v _ { \mathrm { n o } } } ,\tag{6}
$$

where $v _ { \mathrm { y e s } }$ and $v _ { \mathrm { n o } }$ are the vocabulary indices of $\mathtt { Y e s }$ and No, respectively. After each prefill, the KV states of p<sub>assoc</sub> are discarded, while $\Delta _ { i }$ is recorded for retrieval.

Associative summary retrieval. We normalize the association scores across all chunks to form an anchor-selection distribution:

$$
w _ { i } = \frac { \exp ( \Delta _ { i } ) } { \sum _ { j = 1 } ^ { K } \exp ( \Delta _ { j } ) } .\tag{7}
$$

During inference, the anchor is selected as

$$
a = \arg \operatorname* { m a x } _ { i } w _ { i } ,\tag{8}
$$

whereas during RLVR training it is sampled as

$$
a \sim \mathrm { C a t e g o r i c a l } ( w _ { 1 } , \ldots , w _ { K } ) .\tag{9}
$$

The anchor identifies the chunk most likely to contain the queried event, while its full temporal extent may span adjacent chunks. We expand the retrieved interval from the anchor in both directions, including consecutive chunks with positive association scores. Let ${ \bar { \mathcal { R } } } ^ { + }$ denote the maximal contiguous chunk index set, which contains a and satisfies $\Delta _ { i } > 0 , \forall i \in \mathcal { R } ^ { + } ; \mathrm { i f } \Delta _ { a } \leq 0 ,$ , we set $\mathcal { R } ^ { + } = \mathcal { O }$ . We set a minimum retrieval size $M .$ , where $M \leq K . \operatorname { I f } \vert \mathcal { R } ^ { + } \vert \geq M .$ we set $\mathcal { R } = \mathcal { R } ^ { + }$ . Otherwise, we retrieve an anchor-centered window to form R with M consecutive chunks. Finally, only the summaries indexed by R are exposed to final grounding. Grounding with the retrieved summaries. Let $S _ { \mathcal { R } }$ denote the retrieved summaries. When $| \mathcal { R } ^ { + } | \leq 1$ , the temporal interval of queried event is likely to be small, and thus finergrained cues may be required for precise boundary prediction. We therefore additionally include the visual cache $\mathcal { C } ^ { * }$ of the chunk with the highest association score to provide finer-grained evidence; otherwise, ${ \mathcal { C } } ^ { * } = \emptyset$ . During chunkwise processing, we maintain the visual KV cache of the highest-scoring chunk seen so far, making $\mathcal { C } ^ { * }$ available for final grounding without extra forward passes. The retrieved concentrated grounding context is constructed as

$$
{ \mathcal { H } } ^ { * } = \{ s , S _ { \mathcal { R } } , { \mathcal { C } } ^ { * } \} .\tag{10}
$$

It is then passed together with the output format prompt p<sub>fmt</sub> (see Eq. (2)) to predict the temporal grounding boundaries.

Length-Aware Gradient Gating for RLVR Training Length-aware gradient gating. Although we only retain compact summary states across chunk-wise prefills, backpropagating gradients through the dense visual tokens of a long video may still incur a high memory load during training. Let N denote the number of frames in a training video, and W be the maximum length where visual-token gradients can be retained under the available memory budget.

We define the length-aware gradient gate as

$$
\gamma = \mathbb { I } [ N \leq W ] ,\tag{11}
$$

where $\mathbb { I } [ \cdot ]$ is the indicator function, i.e., when $N \leq W$ $\gamma = 1$ , otherwise, $\gamma = 0$

Let $Z _ { i }$ denote the visual-token states of chunk $C _ { i }$ . During training, we replace $Z _ { i }$ with $\widetilde { Z } _ { i }$ , which is

$$
\widetilde { Z } _ { i } = \gamma Z _ { i } + ( 1 - \gamma ) \mathrm { s g } ( Z _ { i } ) ,\tag{12}
$$

where $\operatorname { s g } ( \cdot )$ denotes stop-gradient.

As a result, for video samples with $N \leq W$ , gradients from the RLVR objective backpropagate through the visualtoken states, enabling the model to learn how query-relevant visual evidence is extracted and encoded into the retained summary states. For samples with $N > W$ , visual tokens are still used for forward summary construction but detached during backpropagation, while summary accumulation, retrieval, and grounding remain optimized.

RLVR objective. We instantiate RLVR with Group Relative Policy Optimization (GRPO). For each training instance, the current policy generates G rollouts. In rollout g, an anchor $a ^ { g }$ is sampled from the query-association distribution, the corresponding summary context is retrieved, and the model predicts a temporal interval $\hat { I } ^ { g } = [ \hat { t } _ { s } ^ { g } , \hat { t } _ { e } ^ { g } ]$

Let $I ^ { \star } ~ = ~ [ t _ { s } , t _ { e } ]$ denote the ground-truth interval. We define a soft association target for each chunk according to its temporal overlap with $I ^ { \star }$

$$
y _ { i } = { \frac { \left| \left[ \tau _ { i - 1 } , \tau _ { i } \right) \cap I ^ { \star } \right| } { \left| I ^ { \star } \right| } } ,\tag{13}
$$

where | · | denotes the temporal duration. All G rollouts contribute to association optimization. For grounding, we retain only rollouts whose retrieved context contains at least one chunk with $y _ { i } > 0$

Each rollout receives separate rewards for anchor selection and temporal grounding:

$$
r _ { \mathrm { a s s o c } } ^ { g } = y _ { a ^ { g } } , \qquad r _ { \mathrm { t g } } ^ { g } = \mathrm { I o U } ( \hat { I } ^ { g } , I ^ { \star } ) .\tag{14}
$$

The two rewards evaluate diferent actions, so we normalize their group-relative advantages separately and yield the GRPO objectives $\mathcal { I } _ { \mathrm { a s s o c } }$ and ${ \mathcal { I } } _ { \mathrm { t g } } ,$ , respectively. The final objective is

$$
\theta ^ { \star } = \arg \operatorname* { m a x } _ { \theta } \left( \mathcal { I } _ { \mathrm { t g } } + \lambda \mathcal { I } _ { \mathrm { a s s o c } } \right) ,\tag{15}
$$

where λ balances the two objectives. Details of advantage normalization, clipped policy optimization, and referencepolicy KL regularization are provided in the supplementary material.

![](images/c303074437a18b39da877a5f8a925b3409f6f8d7f8e8395e34613fcabb65b983.jpg)  
Figure 3: Peak GPU memory during training as the video length increases from 15 to 60 frames. All inputs are divided into 15-frame chunks. Full Visual Gradients retains gradients through all visual tokens, while ours applies length-aware gradient gating technique.

## Experiments

## Setup

Datasets. We construct the training set by randomly sampling 42.5K examples from datasets including NaQ (Ramakrishnan, Al-Halah, and Grauman 2023), DiDeMo (Anne Hendricks et al. 2017), QuerYD (Oncescu et al. 2021), HiRest (Zala et al. 2023), COIN (Tang et al. 2019), Momentor (Qian et al. 2024), and YouCook2 (Zhou, Xu, and Corso 2018). For video temporal grounding evaluation, we use four standard benchmarks. For short-video temporal grounding, we evaluate on Charades-STA (Gao et al. 2017), which focuses on indoor human activities, and ActivityNet-Captions (Krishna et al. 2017), which contains diverse videos with dense event annotations. For long-video temporal grounding, we use TACoS (Regneri et al. 2013), mainly consisting of cooking videos, and Ego4D-NLQ (Grauman et al. 2022), which pairs long egocentric videos with naturallanguage queries. We evaluate a single checkpoint across all four benchmarks in a zero-shot setting, without datasetspecific fine-tuning. Detailed dataset descriptions are provided in the supplementary material.

Evaluation Metrics. Following (Li et al. 2025), we report Recall@1 (R1@) at Intersection-over-Union (IoU) thresholds and mean IoU (mIoU) for evaluation. Specifically, we use IoU thresholds of 0.3 (i.e., R1@.3) and 0.5 for long-video benchmarks, and 0.5 and 0.7 for short-video benchmarks.

Implementation Details. We use Qwen2.5-VL-7B-Instruct (Bai et al. 2025) as the backbone LVLM and freeze the vision encoder during training. The minimum retrieval size M is set to 3. For GRPO optimization, each training instance uses $G = 8$ rollouts. The objective weight is set to $\lambda = 0 . 1$ . We set the gradient-gating threshold W to 42 frames and use chunk size of 42 frames (except for the last chunk, which may contain fewer than 42 frames) during inference. For short video samples with $N \leq W$ , we randomly choose the number of chunks $K \in \{ 1 , 2 , 3 \}$ . For long video samples with $N > W$ , we partition videos with each chunk of W frames. We train the model with DeepSpeed ZeRO-3 using a learning rate of $1 \times 1 0 ^ { - 6 }$ for 2 epochs on 8 NVIDIA H20 GPUs.

<table><tr><td rowspan="3">Method</td><td colspan="6">Long-video Benchmarks</td><td colspan="6">Standard-video Benchmarks</td></tr><tr><td colspan="3">Ego4D-NLQ</td><td colspan="3">TACoS</td><td colspan="3">Charades-STA</td><td colspan="3">ActivityNet-Captions</td></tr><tr><td>R1@.3</td><td>R1@.5</td><td>mIoU</td><td>R1@.3</td><td>R1@.5</td><td>mIoU</td><td>R1@.5</td><td>R1@.7</td><td>mIoU</td><td>R1@.5 R1@.7</td><td></td><td>mIoU</td></tr><tr><td colspan="9">Supervised / SFT-based grounding methods</td><td></td><td></td><td></td><td></td></tr><tr><td>UniVTG (Lin et al. 2023)</td><td>6.48</td><td>3.48</td><td>4.63</td><td>5.17</td><td>1.27</td><td>4.40</td><td>25.22</td><td>10.03</td><td>27.12</td><td>11.10</td><td>4.06</td><td>16.86</td></tr><tr><td>VTG-LLM (Guo et al. 2025)</td><td>1.71</td><td>0.46</td><td>1.36</td><td>6.87</td><td>2.92</td><td>5.27</td><td>34.11</td><td>15.81</td><td>34.93</td><td>12.32</td><td>6.74</td><td>17.86</td></tr><tr><td>TimeSuite (Zeng et al. 2024)</td><td>0.88</td><td>0.43</td><td>0.94</td><td>6.75</td><td>2.50</td><td>5.71</td><td>48.95</td><td>24.65</td><td>45.91</td><td>16.56</td><td>9.28</td><td>22.03</td></tr><tr><td>VideoMind (Liu et al. 2025)</td><td>7.20</td><td>3.70</td><td>5.40</td><td>49.50</td><td>36.20</td><td>34.40</td><td>59.10</td><td>31.20</td><td>50.20</td><td>30.30</td><td>15.70</td><td>33.30</td></tr><tr><td>UniTime (Li et al. 2025)</td><td>14.67</td><td>7.38</td><td>10.18</td><td>50.06</td><td>31.54</td><td>33.38</td><td>59.09</td><td>31.88</td><td>52.19</td><td>22.77</td><td>14.14</td><td>27.31</td></tr><tr><td colspan="9">RLVR-based grounding methods with released checkpoints</td><td></td><td></td><td></td></tr><tr><td>Time-R1 (Wang et al. 2025) TimeLens (Zhang et al. 2025a)</td><td>4.82</td><td>2.51 7.56</td><td>3.51 9.27</td><td>30.54 47.40</td><td>17.72 35.20</td><td>20.54 32.77</td><td>60.80 39.80</td><td>35.30 14.50</td><td>58.10 42.30</td><td>39.00 35.20</td><td>21.40</td><td>40.50</td></tr><tr><td colspan="9">13.48 RLVR-based grounding methods retrained under the same training data</td><td></td><td>19.70</td><td>37.30</td></tr><tr><td>Time-R1† (Wang et al. 2025)</td><td>1.95</td><td>0.63</td><td>1.68</td><td>21.49</td><td>11.05</td><td>15.20</td><td>57.74</td><td>31.53</td><td>51.19</td><td>26.39</td><td>12.39</td><td>29.99</td></tr><tr><td>TimeLens† (Zhang et al. 2025a)</td><td>10.66</td><td>5.37</td><td>7.87</td><td>44.80</td><td>28.11</td><td>29.90</td><td>66.48</td><td>34.38</td><td>54.85</td><td>36.03</td><td>18.11</td><td>38.43</td></tr><tr><td colspan="9">Ours</td><td></td><td></td><td></td></tr><tr><td>SumGround w/o. Retrieval</td><td>4.81</td><td>1.98</td><td>4.06</td><td>33.79</td><td>19.77</td><td>23.70</td><td>65.35</td><td>39.49</td><td>55.94</td><td>35.92</td><td>19.12</td><td>39.09</td></tr><tr><td>SumGround</td><td>16.42</td><td>9.51</td><td>11.83</td><td>52.96</td><td>38.07</td><td>35.93</td><td>65.91</td><td>39.57</td><td>56.10</td><td>36.70</td><td>20.72</td><td>39.54</td></tr></table>

Table 1: Comparison with previous methods on four held-out benchmarks. Ego4D-NLQ and TACoS are long-video benchmarks with average durations of 500s and 368s, respectively, while Charades-STA and ActivityNet-Captions with average durations of 29s and 118s. <sup>†</sup> denotes methods retrained with the same training data and evaluation pipeline as SumGround. Gray-shaded entries denote results reproduced or evaluated by us. We use oficially released checkpoints and codes. Other numbers are directly cited from previous papers. SumGround w/o. Retrieval means we remove the associative summary retrieval operation from our method. Best and second-best results are shown in bold and underlined, respectively.

## Comparison with Previous State-of-the-Arts

Table 1 compares SumGround with previous state-of-theart (SOTA) methods on four typical video benchmarks, including long-video benchmarks (i.e., Ego4D-NLQ and TACoS) and short-video benchmarks (i.e., Charades-STA and ActivityNet-Captions). Note that we evaluate Sum-Ground in a zero-shot setting, without fine-tuning on each evaluation benchmark. Previous methods can be roughly categorized into two groups, including supervised/SFT-based methods (e.g., UniVTG, VTG-LLM, etc. ) and RLVR-based methods (e.g., Time-R1, TimeLens). Our method falls into the category of RLVR-based methods. For previous RLVRbased methods (i.e., Time-R1, TimeLens), besides the released checkpoints, we also retrain and evaluate the models with their oficial released codes under our training data to enable a fair comparison.

From Table 1, we have three observations. First, generally speaking, the performance on long-video benchmarks is obviously lower than that on short-video benchmarks, either for SFT-based or for RLVR-based methods. This indicates that it is more challenging to locate sparse evidence in long visual contexts. Second, comparing previous RLVR-based methods to SFT-based methods, we observe that previous RLVR-based methods perform favorably against SFT-based methods on short-video benchmarks, but generally performs worse than SOTA SFT-based methods on long-video benchmarks. This suggests that without specific design, purely RLVR cannot efectively mine and exploit query-relevant evidence for grounding. Third, our SumGround performs favorably against all previous works on both short-video and longvideo benchmarks. Note that under the same training data, our method performs obviously better previous RLVR-based methods. For example, on long-video benchmarks Ego4D-NLQ and TACoS, our method outperforms TimeLens<sup>†</sup> by around 6% and 8% with R1@.3 and 4% and 6% with mIoU. Overall, the comparisons verify the superiority of our method in performing long-video temporal grounding, which benefits from our specific design to enable LVLM to efectively mine and exploit query-relevant evidence from long videos.

## Ablation Study

Efect of summary representation. Table 2(a) examines how the summary representation afects grounding. To simplify our experiment and guarantee a fair comparison, we consider single chunk and keep summary length (tokens) approximately the same. We compare direct visual grounding (“w/o. Summary”) with three summary variants: learned <summary> tokens, a query-agnostic naturallanguage summary prompt, and our query-guided summary prompt. Only our query-guided design can surpass direct visual grounding. These results show latent compression alone is insuficient and our query-guided way can efectively extract query-relevant information to support grounding.

<table><tr><td>Setting</td><td>R1@0.3</td><td>R1@0.5</td><td>R1@0.7</td><td>mIoU</td></tr><tr><td colspan="5">(a) Summary Type</td></tr><tr><td>w/o. Summary</td><td>78.09</td><td>63.01</td><td>36.42</td><td>54.46</td></tr><tr><td>&lt;summary&gt;</td><td>73.12</td><td>54.38</td><td>27.28</td><td>48.78</td></tr><tr><td>Query-Agnostic</td><td>76.96</td><td>60.56</td><td>34.81</td><td>52.93</td></tr><tr><td>Query-Guided (Ours)</td><td>81.83</td><td>69.57</td><td>44.30</td><td>58.75</td></tr><tr><td colspan="5">(b) Chunk-wise Dependency</td></tr><tr><td>Independent</td><td>72.39</td><td>56.02</td><td>30.38</td><td>49.53</td></tr><tr><td>Causal (Ours)</td><td>82.04</td><td>72.04</td><td>48.25</td><td>59.88</td></tr></table>

Table 2: Ablation of summary design on Charades-STA. All variants use approximately the same retained token budget. (a) Summary representation comparison The “w/o. Summary” directly grounds from the visual inputs without any summary operations. “<summary>” means learned <summary> tokens. “Query-Agnostic” means we use a general summary prompt which does not rely on the query content. To simplify our comparison, we treat the whole video as a single chunk. (b) Chunk-wise Dependency comparison. “Independent” means the summary of each chunk is independent on those of other chunks.

<table><tr><td>Retrieval Strategy</td><td>R1@0.3</td><td>R1@0.5</td><td>R1@0.7</td><td>mIoU</td></tr><tr><td>Random</td><td>28.32</td><td>19.57</td><td>9.75</td><td>19.46</td></tr><tr><td>Anchor Only</td><td>43.64</td><td>30.17</td><td>15.20</td><td>29.67</td></tr><tr><td>Ours</td><td>52.96</td><td>38.07</td><td>18.67</td><td>35.93</td></tr></table>

Table 3: Comparison of summary retrieval strategies on TACoS. Random retains the same number of summaries as our method but selects them at random. Anchor Only retains only the anchor summary.

Efect ofcausal accumulation. Table 2(b) demonstrates how chunk-wise dependency afects the performance. Comparing with the “Independent” way where the summary of each chunk is independent of other chunks, the causal way (ours) performs remarkably better. The results verify that the summary of current chunk should attend to the summaries of previous chunks for better information aggregation.

Efect of associative summary retrieval. Table 1 shows that without associative summary retrieval, the performance obviously drops, especially for long videos. Further, in Table 3, we compare our retrieval strategy with several variants to verify the efectiveness of our design. Among all variants, our method performs the best. All the results verify that associative summary retrieval may efectively remove redundant summaries and ease the grounding.

<table><tr><td>γ</td><td>Long Video</td><td>R1@0.3</td><td>R1@0.5</td><td>mIoU</td></tr><tr><td>1</td><td>V</td><td>OOM</td><td>00M</td><td>OOM</td></tr><tr><td>0</td><td>√</td><td>27.44</td><td>14.32</td><td>20.42</td></tr><tr><td>1</td><td>x</td><td>45.66</td><td>29.94</td><td>31.46</td></tr><tr><td> $\mathbb { I } [ N \le W ]$ </td><td>√</td><td>52.96</td><td>38.07</td><td>35.93</td></tr></table>

Table 4: Comparison of gradient-gating configurations on TACoS. γ controls whether gradients propagate through visual-token states (γ=1 and 0 means retaining and stopping visual gradients respectively), and Long Video means long videos with $N > \dot { W }$ are utilized in training. The total number of training samples is kept the same across all variants. The last row denotes length-aware gradient gating. OOM denotes out of memory.

Efect of RLVR on associative summary retrieval capability. To verify the efectiveness of RLVR on enhancing associative summary retrieval capability, we replace the learned association scores with those estimated by the base LVLM, while keeping the trained SumGround model for final grounding. This variant only achieves 30.35 mIoU on TACoS, worse than ours (35.93 mIoU), showing that RLVR improves the query-association estimation and thus improves associative summary retrieval capability.

Efect of length-aware gradient gating. Table 4 compares our length-aware gradient gating with several variants. Keeping visual gradients for all inputs $( \gamma = 1 )$ is infeasible due to OOM, whereas stopping them (γ = 0) for all inputs substantially degrades grounding, showing that visual gradients are necessary for learning efective summary. Compared with training only on short contexts, length-aware gradient gating further improves mIoU by 4.47 points. This result shows that short-context visual-to-summary learning and long-context summary optimization are complementary.

Figure 3 examines how peak GPU memory during training increases as video length increases. We compare our method to the variant which enables full visual gradients. As the length increases from 15 to 60 frames, the memory of “full visual gradients” is increased by around 84%. In contrast, our method only increases 6%. Note that at 60 frames, our method consumes less than half the memory of that variant. This comparison demonstrates the our memory eficiency.

## Conclusion

We propose a novel framework named SumGround to perform query-guided chunk condensation to aggregate and retrieve query-relevant evidence. Specifically, we split the video into several chunks and perform two-level chunk condensation, including query-guided latent summary and associative summary retrieval. Both capabilities are enabled by RLVR. To reduce memory consumption, we propose a length-aware gradient gating module to selectively stop gradient back-propagated to visual tokens. Experiments verify the efectiveness of our method. We expect our attempts may inspire future investigations on better mining query-relevant evidence for efective long video temporal grounding.

## References

An, S.; Sung, J.; Park, W.; Park, C.; and Seo, H. 2025. Lcirc: A recurrent compression approach for eficient long-form context and query dependent modeling in llms. In Proceedings of the 2025 Conference of the Nations of the Americas Chapter of the Association for Computational Linguistics: Human Language Technologies (Volume 1: Long Papers), 10431–10442.

Anne Hendricks, L.; Wang, O.; Shechtman, E.; Sivic, J.; Darrell, T.; and Russell, B. 2017. Localizing moments in video with natural language. In Proceedings of the IEEE international conference on computer vision, 5803–5812.

Bai, S.; Chen, K.; Liu, X.; Wang, J.; Ge, W.; Song, S.; Dang, K.; Wang, P.; Wang, S.; Tang, J.; et al. 2025. Qwen2. 5-VL Technical Report. arXiv e-prints, arXiv–2502.

Chevalier, A.; Wettig, A.; Ajith, A.; and Chen, D. 2023. Adapting language models to compress contexts. In Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing, 3829–3846.

Devlin, J.; Chang, M.-W.; Lee, K.; and Toutanova, K. 2019. Bert: Pre-training of deep bidirectional transformers for language understanding. In Proceedings ofthe 2019 conference ofthe North American chapter ofthe associationfor computational linguistics: human language technologies, volume 1 (long and short papers), 4171–4186.

Fu, C.; Dai, Y.; Luo, Y.; Li, L.; Ren, S.; Zhang, R.; Wang, Z.; Zhou, C.; Shen, Y.; Zhang, M.; et al. 2025. Videomme: The first-ever comprehensive evaluation benchmark of multi-modal llms in video analysis. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, 24108–24118.

Gao, J.; Sun, C.; Yang, Z.; and Nevatia, R. 2017. Tall: Temporal activity localization via language query. In Proceedings of the IEEE international conference on computer vision, 5267–5275.

Grauman, K.; Westbury, A.; Byrne, E.; Chavis, Z.; Furnari, A.; Girdhar, R.; Hamburger, J.; Jiang, H.; Liu, M.; Liu, X.; et al. 2022. Ego4d: Around the world in 3,000 hours of egocentric video. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, 18995–19012.

Guo, Y.; Liu, J.; Li, M.; Cheng, D.; Tang, X.; Sui, D.; Liu, Q.; Chen, X.; and Zhao, K. 2025. Vtg-llm: Integrating timestamp knowledge into video llms for enhanced video temporal grounding. In Proceedings ofthe AAAI Conference on Artificial Intelligence, volume 39, 3302–3310.

Huang, B.; Wang, X.; Chen, H.; Song, Z.; and Zhu, W. 2024. Vtimellm: Empower llm to grasp video moments. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 14271–14280.

Krishna, R.; Hata, K.; Ren, F.; Fei-Fei, L.; and Carlos Niebles, J. 2017. Dense-captioning events in videos. In Proceedings of the IEEE international conference on computer vision, 706–715.

Lei, J.; Berg, T. L.; and Bansal, M. 2021. Detecting moments and highlights in videos via natural language queries. Advances in Neural Information Processing Systems, 34: 11846–11858.

Li, K.; Wang, Y.; He, Y.; Li, Y.; Wang, Y.; Liu, Y.; Wang, Z.; Xu, J.; Chen, G.; Luo, P.; et al. 2024. Mvbench: A comprehensive multi-modal video understanding benchmark. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 22195–22206.

Li, Y.; Wang, C.; and Jia, J. 2024. Llama-vid: An image is worth 2 tokens in large language models. In European Conference on Computer Vision, 323–340. Springer.

Li, Z.; Di, S.; Zhai, Z.; Huang, W.; Wang, Y.; and Xie, W. 2025. Universal video temporal grounding with generative multi-modal large language models. arXiv preprint arXiv:2506.18883.

Lin, K. Q.; Zhang, P.; Chen, J.; Pramanick, S.; Gao, D.; Wang, A. J.; Yan, R.; and Shou, M. Z. 2023. Univtg: Towards unified video-language temporal grounding. In Proceedings of the IEEE/CVF international conference on computer vision, 2794–2804.

Lin, X.; Ghosh, A.; Low, B. K. H.; Shrivastava, A.; and Mohan, V. 2025. Refrag: Rethinking rag based decoding. arXiv preprint arXiv:2509.01092.

Liu, Y.; Li, S.; Liu, Y.; Wang, Y.; Ren, S.; Li, L.; Chen, S.; Sun, X.; and Hou, L. ???? Tempcompass: Do video llms really understand videos?, 2024. URL https://arxiv. org/abs/2403.00476.

Liu, Y.; Qinghong Lin, K.; Chen, C. W.; and Shou, M. Z. 2025. Videomind: A chain-of-lora agent for long video reasoning. arXiv e-prints, arXiv–2503.

Mangalam, K.; Akshulakov, R.; and Malik, J. 2023. Egoschema: A diagnostic benchmark for very long-form video language understanding. Advances in Neural Information Processing Systems, 36: 46212–46244.

Oncescu, A.-M.; Henriques, J. F.; Liu, Y.; Zisserman, A.; and Albanie, S. 2021. Queryd: A video dataset with highquality text and audio narrations. In ICASSP 2021-2021 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP), 2265–2269. IEEE.

Qian, L.; Li, J.; Wu, Y.; Ye, Y.; Fei, H.; Chua, T.-S.; Zhuang, Y.; and Tang, S. 2024. Momentor: Advancing video large language model with fine-grained temporal reasoning. arXiv preprint arXiv:2402.11435.

Radford, A.; Kim, J. W.; Hallacy, C.; Ramesh, A.; Goh, G.; Agarwal, S.; Sastry, G.; Askell, A.; Mishkin, P.; Clark, J.; et al. 2021. Learning transferable visual models from natural language supervision. In International conference on machine learning, 8748–8763. PmLR.

Ramakrishnan, S. K.; Al-Halah, Z.; and Grauman, K. 2023. Naq: Leveraging narrations as queries to supervise episodic memory. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 6694–6703.

Regneri, M.; Rohrbach, M.; Wetzel, D.; Thater, S.; Schiele, B.; and Pinkal, M. 2013. Grounding action descriptions in videos. Transactions of the Association for Computational Linguistics, 1: 25–36.

Ren, S.; Yao, L.; Li, S.; Sun, X.; and Hou, L. 2024. Timechat: A time-sensitive multimodal large language model for long video understanding. In Proceedings of the IEEE/CVF

Conference on Computer Vision and Pattern Recognition, 14313–14323.

Shao, Z.; Wang, P.; Zhu, Q.; Xu, R.; Song, J.; Bi, X.; Zhang, H.; Zhang, M.; Li, Y.; Wu, Y.; et al. 2024. Deepseekmath: Pushing the limits of mathematical reasoning in open language models. arXiv preprint arXiv:2402.03300.

Shen, X.; Xiong, Y.; Zhao, C.; Wu, L.; Chen, J.; Zhu, C.; Liu, Z.; Xiao, F.; Varadarajan, B.; Bordes, F.; et al. 2024. Longvu: Spatiotemporal adaptive compression for long video-language understanding. arXiv preprint arXiv:2410.17434.

Shu, Y.; Liu, Z.; Zhang, P.; Qin, M.; Zhou, J.; Liang, Z.; Huang, T.; and Zhao, B. 2025. Video-xl: Extra-long vision language model for hour-scale video understanding. In Proceedings of the Computer Vision and Pattern Recognition Conference, 26160–26169.

Song, E.; Chai, W.; Wang, G.; Zhang, Y.; Zhou, H.; Wu, F.; Chi, H.; Guo, X.; Ye, T.; Zhang, Y.; et al. 2024. Moviechat: From dense token to sparse memory for long video understanding. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 18221–18232.

Tang, Y.; Ding, D.; Rao, Y.; Zheng, Y.; Zhang, D.; Zhao, L.; Lu, J.; and Zhou, J. 2019. Coin: A large-scale dataset for comprehensive instructional video analysis. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 1207–1216.

Wang, Y.; Meng, X.; Liang, J.; Wang, Y.; Liu, Q.; and Zhao, D. 2024. Hawkeye: Training video-text llms for grounding text in videos. arXiv preprint arXiv:2403.10228.

Wang, Y.; Wang, Z.; Xu, B.; Du, Y.; Lin, K.; Xiao, Z.; Yue, Z.; Ju, J.; Zhang, L.; Yang, D.; et al. 2025. Time-r1: Post-training large vision language model for temporal video grounding. arXiv preprint arXiv:2503.13377.

Yin, H.; Si, G.; and Wang, Z. 2025. Lifting the veil on visual information flow in MLLMs: unlocking pathways to faster inference. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 9382–9391.

Yuan, C.; Yang, Y.; Yang, Y.; and Cheng, Z. 2025. Date: Dynamic absolute time enhancement for long video understanding. arXiv preprint arXiv:2509.09263.

Yuan, Y.; Mei, T.; and Zhu, W. 2019. To find where you talk: Temporal sentence localization in video with attention based location regression. In Proceedings of the AAAI conference on artificial intelligence, volume 33, 9159–9166.

Zala, A.; Cho, J.; Kottur, S.; Chen, X.; Oguz, B.; Mehdad, Y.; and Bansal, M. 2023. Hierarchical video-moment retrieval and step-captioning. In Proceedings ofthe IEEE/CVF Conference on Computer Vision and Pattern Recognition, 23056–23065.

Zeng, X.; Li, K.; Wang, C.; Li, X.; Jiang, T.; Yan, Z.; Li, S.; Shi, Y.; Yue, Z.; Wang, Y.; et al. 2024. Timesuite: Improving mllms for long video understanding via grounded tuning. arXiv preprint arXiv:2410.19702.

Zhang, H.; Sun, A.; Jing, W.; and Zhou, J. T. 2023. Temporal sentence grounding in videos: A survey and future directions. IEEE Transactions on Pattern Analysis and Machine Intelligence, 45(8): 10443–10465.

Zhang, J.; Wang, T.; Ge, Y.; Ge, Y.; Li, X.; Shan, Y.; and Wang, L. 2025a. Timelens: Rethinking video temporal grounding with multimodal llms. arXiv preprint arXiv:2512.14698.

Zhang, S.; Peng, H.; Fu, J.; and Luo, J. 2020. Learning 2d temporal adjacent networks for moment localization with natural language. In Proceedings of the AAAI conference on artificial intelligence, volume 34, 12870–12877.

Zhang, Z.; Yadav, S.; Han, F.; and Shutova, E. 2025b. Crossmodal information flow in multimodal large language models. In Proceedings of the Computer Vision and Pattern Recognition Conference, 19781–19791.

Zhou, L.; Xu, C.; and Corso, J. 2018. Towards automatic learning of procedures from web instructional videos. In Proceedings of the AAAI conference on artificial intelligence, volume 32.

## Appendix

## Inference procedure.

During inference, SumGround follows a deterministic summary-retrieval-and-grounding pipeline. Given a videoquery pair, it first samples timestamped frames and partitions the video into K consecutive chunks. Each chunk-wise pass jointly writes a query-guided summary state $S _ { i }$ and estimates a query-association score $\Delta _ { i }$ . Throughout chunkwise processing, SumGround retains the system KV states s, accumulates the summary states $\{ S _ { i } \} _ { i = 1 } ^ { K }$ , and maintains the local visual KV cache of the highest-scoring chunk observed so far. After all chunks are processed, it computes the query-association distribution $p _ { i } = \mathrm { s o f t m a x } ( \Delta ) _ { i }$ and selects the anchor chunk as $a = \arg \operatorname* { m a x } _ { i } p _ { i }$ . Starting from $^ { a , }$ , SumGround constructs the retrieved chunk set R using the associative summary retrieval rule defined the main paper, and concatenates the corresponding summary states as $S _ { \mathcal { R } }$ When no non-anchor chunk has positive query association, the retained local visual cache is included as $\bar { \mathcal { C } } ^ { * }$ ; otherwise, ${ \mathcal { C } } ^ { * } = \emptyset$ . Finally, the model predicts the temporal interval conditioned on $\left[ s , S _ { \mathcal { R } } , \mathcal { C } ^ { * } , p _ { \mathrm { f m t } } \right]$

## Training Objective and Procedure

## Task-Specific GRPO Objectives

We jointly optimize association estimation and temporal grounding with task-specific GRPO objectives. The two objectives difer in their action spaces, valid rollout sets, and regularization. Association optimization treats the sampled anchor chunk as a categorical action, whereas temporal grounding optimizes the generated response tokens. We apply reference-policy KL regularization only to temporal grounding.

Task-specific rollout sets. Let

$$
y ^ { + } = \{ i \mid y _ { i } > 0 \}\tag{16}
$$

denote the set of chunks that temporally overlap with the ground-truth interval.

For rollout g, let $\mathcal { R } ^ { g }$ denote the indices of all video chunks represented in its final grounding context, including the retrieved summary states and, when used, the additional local visual cache. We define the task-specific rollout sets as

$$
S _ { \mathrm { a s s o c } } = \{ 1 , . . . , G \} , \qquad S _ { \mathrm { t g } } = \{ g \mid \mathcal { V } ^ { + } \cap \mathcal { R } ^ { g } \neq \emptyset \} _ { }\tag{17}
$$

Every sampled anchor defines a valid association action, so all G rollouts are included in $S _ { \mathrm { a s s o c } } .$ . For temporal grounding, we retain only rollouts whose grounding context contains ground-truth-relevant evidence. This prevents the timestampgeneration policy from being penalized for failures caused solely by an evidence-deficient retrieved context.

Task-specific advantage normalization. For each objective $\kappa \in$ {assoc, tg}, we compute the reward mean and standard deviation over its corresponding rollout set:

$$
\mu _ { \kappa } = { \frac { 1 } { | S _ { \kappa } | } } \sum _ { g \in S _ { \kappa } } r _ { \kappa } ^ { g } ,\tag{18}
$$

$$
\sigma _ { \kappa } = \sqrt { \frac { 1 } { \left| S _ { \kappa } \right| } \sum _ { g \in S _ { \kappa } } \left( r _ { \kappa } ^ { g } - \mu _ { \kappa } \right) ^ { 2 } } .\tag{19}
$$

The group-relative advantage is

$$
{ \widehat { A } } _ { \kappa } ^ { g } = { \frac { r _ { \kappa } ^ { g } - \mu _ { \kappa } } { \sigma _ { \kappa } + \epsilon _ { \mathrm { a d v } } } } , \qquad g \in S _ { \kappa } ,\tag{20}
$$

where $\epsilon _ { \mathrm { a d v } } > 0$ is a numerical-stability constant. The normalized advantages are treated as constants during policy optimization.

When $S _ { \mathrm { t g } } = \emptyset$ , the temporal-grounding objective is omitted for that training instance. When all rewards in a rollout set are identical, the corresponding advantages are zero, and that objective produces no policy-gradient update for the instance.

Association objective. For association optimization, the action in rollout g is the sampled anchor chunk $a ^ { g }$ . Its probability under the current policy is

$$
p _ { \theta } ( a ^ { g } ) = w _ { a ^ { g } } ,\tag{21}
$$

where $w _ { i }$ is the association distribution defined in the main paper.

Let $\theta _ { \mathrm { o l d } }$ denote the rollout policy. We define the probability ratio

$$
\rho _ { \mathrm { a s s o c } } ^ { g } = \frac { p _ { \theta } ( a ^ { g } ) } { p _ { \theta _ { \mathrm { o l d } } } ( a ^ { g } ) } ,\tag{22}
$$

and its clipped counterpart

$$
\bar { \rho } _ { \mathrm { a s s o c } } ^ { g } = \mathrm { c l i p } \left( \rho _ { \mathrm { a s s o c } } ^ { g } , 1 - \epsilon _ { \mathrm { c l i p } } , 1 + \epsilon _ { \mathrm { c l i p } } \right) ,\tag{23}
$$

where $\epsilon _ { \mathrm { c l i p } }$ is the policy-ratio clipping threshold.

The association objective is

$$
\mathcal { T } _ { \mathrm { a s s o c } } = \frac { 1 } { | S _ { \mathrm { a s s o c } } | } \sum _ { g \in \mathcal { S } _ { \mathrm { a s s o c } } } \operatorname* { m i n } \left( \rho _ { \mathrm { a s s o c } } ^ { g } \widehat { A } _ { \mathrm { a s s o c } } ^ { g } , \bar { \rho } _ { \mathrm { a s s o c } } ^ { g } \widehat { A } _ { \mathrm { a s s o c } } ^ { g } \right) .\tag{24}
$$

We do not apply reference-policy KL regularization to this objective, because the anchor is a task-specific categorical action over video chunks rather than a language-generation action.

Temporal-grounding objective. For each rollout $g \in S _ { \mathrm { t g } }$ let

$$
\mathbf { o } ^ { g } = ( o _ { 1 } ^ { g } , \dots , o _ { T _ { g } } ^ { g } )\tag{25}
$$

denote the generated response containing the predicted temporal boundaries. Let

$$
H ^ { g } = [ s , S _ { \mathcal { R } ^ { g } } , \mathcal { C } ^ { g } ]\tag{26}
$$

denote the retrieved grounding context of rollout g. The token-level history is

$$
h _ { t } ^ { g } = ( H ^ { g } , p _ { \mathrm { f m t } } , o _ { < t } ^ { g } ) .\tag{27}
$$

denote the grounding context and generated prefix preceding token $o _ { t } ^ { g }$

The token-level probability ratio is

$$
\rho _ { \mathrm { t g } , t } ^ { g } = \frac { \pi _ { \theta } \left( o _ { t } ^ { g } \mid h _ { t } ^ { g } \right) } { \pi _ { \theta _ { \mathrm { o l d } } } \left( o _ { t } ^ { g } \mid h _ { t } ^ { g } \right) } ,\tag{28}
$$

with the clipped ratio

$$
\bar { \rho } _ { \mathrm { t g } , t } ^ { g } = \mathrm { c l i p } \left( \rho _ { \mathrm { t g } , t } ^ { g } , 1 - \epsilon _ { \mathrm { c l i p } } , 1 + \epsilon _ { \mathrm { c l i p } } \right) .\tag{29}
$$

Let $\pi _ { \mathrm { r e f } }$ denote the frozen reference policy. For each generated token, we use the sampled-token KL estimator

$$
d _ { \mathrm { K L } , t } ^ { g } = \exp \left( \ell _ { \mathrm { r e f } , t } ^ { g } - \ell _ { \theta , t } ^ { g } \right) - \left( \ell _ { \mathrm { r e f } , t } ^ { g } - \ell _ { \theta , t } ^ { g } \right) - 1 ,\tag{30}
$$

where

$$
\ell _ { \theta , t } ^ { g } = \log \pi _ { \theta } \left( o _ { t } ^ { g } \mid h _ { t } ^ { g } \right) , \qquad \ell _ { \mathrm { r e f } , t } ^ { g } = \log \pi _ { \mathrm { r e f } } \left( o _ { t } ^ { g } \mid h _ { t } ^ { g } \right) .
$$

The temporal-grounding objective is

(31)

$$
\begin{array} { c } { \mathcal { I } _ { \mathrm { t g } } = \displaystyle \frac { 1 } { \left| \mathcal { S } _ { \mathrm { t g } } \right| } \sum _ { g \in \mathcal { S } _ { \mathrm { t g } } } \frac { 1 } { T _ { g } } \sum _ { t = 1 } ^ { T _ { g } } { \Big [ \operatorname* { m i n } \Big ( \rho _ { \mathrm { t g } , t } ^ { g } \widehat A _ { \mathrm { t g } } ^ { g } , \bar { \rho } _ { \mathrm { t g } , t } ^ { g } \widehat A _ { \mathrm { t g } } ^ { g } \Big ) } } \\ { - \beta d _ { \mathrm { K L } , t } ^ { g } \Big ] } ,  \end{array}\tag{32}
$$

where $\beta$ controls the strength of reference-policy KL regularization.

The KL term is applied only to temporal grounding to prevent excessive drift in language generation while allowing the task-specific association distribution to adapt freely.

Final training objective. The complete objective is

$$
\theta ^ { \star } = \arg \operatorname* { m a x } _ { \theta } \left( \mathcal { I } _ { \mathrm { t g } } + \lambda \mathcal { I } _ { \mathrm { a s s o c } } \right) ,\tag{33}
$$

where λ balances temporal-grounding and association optimization.

## Training Algorithm

Algorithm 1 summarizes the training procedure of Sum-Ground. It integrates causal accumulation of query-guided summaries, associative summary retrieval, length-aware gradient gating, and dual-objective GRPO. Each chunk $C _ { i }$ contains sampled frames together with explicit timestamp tokens. For a training instance with N sampled frames, lengthaware gradient gating applies stop-gradient only to the visualtoken states when $\bar { N } > \bar { W }$ , while leaving the timestamp and prompt-token states unchanged. This operation preserves the forward computation while reducing the memory required for back-propagation through long visual contexts. The detailed reward definitions and GRPO objectives are given above.

Table 5: Statistics of temporal grounding datasets used in our experiments. “Video Len.” and “Moment Len.” denote the average video duration and average annotated temporal moment duration, respectively.
<table><tr><td>Split</td><td>Dataset</td><td>Video Len.</td><td>Moment Len.</td><td>Views</td><td>Domain</td></tr><tr><td rowspan="7">Training</td><td>NaQ (Ramakrishnan, Al-Halah, and Grauman 2023)</td><td>413s</td><td>1.1s</td><td>Ego</td><td>Open</td></tr><tr><td>DiDeMo (Anne Hendricks et al. 2017)</td><td>29s</td><td>7.5s</td><td>Exo</td><td>Open</td></tr><tr><td>QuerYD (Oncescu et al. 2021)</td><td>278s</td><td>13.6s</td><td>Ego &amp; Exo</td><td>Open</td></tr><tr><td>HiRest (Zala et al. 2023)</td><td>263s</td><td>18.9s</td><td>Ego &amp; Exo</td><td>Open</td></tr><tr><td>COIN (Tang et al. 2019)</td><td>145s</td><td>14.9s</td><td>Ego &amp; Exo</td><td>Open</td></tr><tr><td>Momentor (Qian et al. 2024)</td><td>403s</td><td>49.5s</td><td>Ego &amp; Exo</td><td>Open</td></tr><tr><td>YouCook2 (Żhou, Xu, and Corso 2018)</td><td>316s</td><td>19.7s</td><td>Ego &amp; Exo</td><td>Cooking</td></tr><tr><td rowspan="4">Benchmark</td><td>Ego4D-NLQ (Grauman et al. 2022)</td><td>500s</td><td>10.7s</td><td>Ego</td><td>Open</td></tr><tr><td>TACoS (Regneri et al. 2013)</td><td>368s</td><td>31.9s</td><td>Ego &amp; Exo</td><td>Cooking</td></tr><tr><td>Charades-STA (Gao et al. 2017)</td><td>29s</td><td>7.8s</td><td>Ego</td><td>Activity</td></tr><tr><td>ANet-Captions (Krishna et al. 2017)</td><td>118s</td><td>40.2s</td><td>Ego &amp; Exo</td><td>Activity</td></tr></table>

## Dataset Details

We provide detailed information on the training and evaluation datasets used in our experiments. The statistics of each dataset are summarized in Table 5.

Training datasets. The training set is composed of NaQ (Ramakrishnan, Al-Halah, and Grauman 2023), DiDeMo (Anne Hendricks et al. 2017), QuerYD (Oncescu et al. 2021), HiRest (Zala et al. 2023), COIN (Tang et al. 2019), Momentor (Qian et al. 2024), and YouCook2 (Zhou, Xu, and Corso 2018). NaQ provides egocentric temporal grounding annotations derived from episodic-memory queries. DiDeMo and QuerYD contain natural-language descriptions or narrations aligned with temporal moments in videos. HiRest provides hierarchical video-moment annotations with step-level descriptions. COIN and YouCook2 mainly focus on instructional and procedural videos, while Momentor contains fine-grained temporal reasoning annotations. Together, these datasets provide a mixture of short and long videos, egocentric and exocentric views, and diferent query formats.

Training data sampling. We do not use all available samples from the above training datasets. Instead, we construct the hybrid training set by randomly sampling from each dataset with a dataset-specific maximum sampling budget. Specifically, we sample 8,000 examples from DiDeMo, 5,000 from COIN, 7,000 from YouCook2, 5,000 from QuerYD, 3,500 from HiRest, 7,000 from Momentor, and 7,000 from NaQ. This sampling strategy balances datasets with diferent scales and prevents large datasets from dominating the training process.

Evaluation datasets. For short-video temporal grounding, we evaluate on Charades-STA (Gao et al. 2017) and ActivityNet-Captions (Krishna et al. 2017). Charades-STA focuses on indoor human activities, ActivityNet-Captions contains diverse activity videos with dense event captions. For long-video temporal grounding, we evaluate on TACoS (Regneri et al. 2013) and Ego4D-NLQ (Grauman et al. 2022). TACoS mainly contains cooking videos with action descriptions, while Ego4D-NLQ consists of egocentric long videos paired with natural-language queries.

Video question answering benchmarks. In addition to temporal grounding benchmarks, we evaluate the general video question answering ability of our model on MVBench (Li et al. 2024), TempCompass (Liu et al.), EgoSchema (Mangalam, Akshulakov, and Malik 2023), and VideoMME (Fu et al. 2025). These benchmarks cover diferent aspects of video understanding, including temporal reasoning, egocentric understanding, and long-video question answering. For TempCompass, we use all multiple-choice QA tasks except for the video captioning task. EgoSchema contains egocentric video clips, each approximately 3 minutes long, with temporally demanding QA pairs. VideoMME is a general video QA benchmark covering diverse domains. It contains 2.7K QA samples over videos of varied lengths, ranging from 11 seconds to 1 hour. We use the long-video split of VideoMME for evaluation.

Artifact licenses and terms of use. We use publicly available models and datasets following their released licenses and terms of use. Our experiments are conducted for research purposes only, and we do not redistribute the original videos, annotations, or model weights beyond their original access conditions.

## Evaluation Metrics

For temporal grounding evaluation, we report Recall@1 at diferent IoU thresholds and mean IoU (mIoU). Given a predicted temporal window $T _ { \mathrm { p r e d } }$ and a ground-truth temporal window $T _ { \mathrm { g t } } ^ { - }$ , the temporal Intersection-over-Union is defined as:

$$
{ \mathrm { I o U } } = { \frac { | T _ { \mathrm { p r e d } } \cap T _ { \mathrm { g t } } | } { | T _ { \mathrm { p r e d } } \cup T _ { \mathrm { g t } } | } } .\tag{34}
$$

Recall@1 at threshold $\theta ,$ denoted as R@θ, measures whether the top-1 predicted temporal window has an IoU at least θ with the ground-truth window:

$$
\mathrm { R @ } \theta = \frac { 1 } { N } \sum _ { i = 1 } ^ { N } \mathbb { I } \left[ \mathrm { I o U } \left( T _ { \mathrm { p r e d } } ^ { ( i ) } , T _ { \mathrm { g t } } ^ { ( i ) } \right) \geq \theta \right] .\tag{35}
$$

Algorithm 1: Training Procedure of SumGround   
Require: Training instance $( V , q , I ^ { \star } )$ , policy $\pi _ { \theta } ,$ rollout number   
$G ,$ and gradient threshold $W .$   
1: Sample N timestamped frames and partition them into chunks   
$\{ C _ { i } \} _ { i = 1 } ^ { K } .$   
2: Initialize summary bank $B  \varnothing , \Delta ^ { \star }  - \infty ,$ , and ${ \mathcal { C } } ^ { \star } \gets \emptyset .$   
3: for $i = 1$ to $K$ do   
4: Apply length-aware gradient gating to the visual-token states   
of $C _ { i } ,$ yielding $\widetilde { C } _ { i }$   
5: Set $X _ { i } = [ p _ { s } , \widetilde { C } _ { i } , p _ { \mathrm { s u m } } ( q ) , p _ { \mathrm { a s s o c } } ] \mathrm { i f } i = 1 ;$ otherwise set   
$X _ { i } = [ s , S _ { < i } , \widetilde { C } _ { i } , p _ { \mathrm { s u m } } ( q ) , p _ { \mathrm { a s s o c } } ] .$   
6: Prefill $X _ { i } ;$ retain $s$ when $\begin{array} { r } { \dot { a } \dot { { \bf \Delta } } = { \bf \Delta } 1 , } \end{array}$ retain $\begin{array} { r l } { S _ { i } } & { { } = } \end{array}$   
$\mathrm { K V } ( X _ { i } ) [ p _ { \mathrm { s u m } } ( q ) ] ,$ and append $S _ { i }$ to $B .$   
7: Compute $\Delta _ { i } = \dot { \ell } _ { \mathrm { y e s } } ^ { \mathrm { ( \bar { i } ) } } - \ell _ { \mathrm { n o } } ^ { \mathrm { ( \bar { i } ) } } .$   
8: if $\Delta _ { i } ^ { \phantom { * } } > \Delta ^ { \star }$ then   
9: Retain the current local visual cache as ${ \mathcal { C } } ^ { \star }$ and update   
$\Delta ^ { \star }$   
10: end if   
11: Discard the remaining chunk-wise KV states.   
12: end for   
13: Compute $w _ { i } =$ softmax $( \Delta ) _ { i }$ <sub>i</sub>.   
14: for $\bar { g ^ { } } = 1$ to G do   
15: Sample anchor $a ^ { g } \sim$ Categorical $( w _ { 1 } , \dots , w _ { K } )$   
16: Construct $( \mathcal { R } ^ { g } , \mathcal { C } ^ { g } )$ using the associative retrieval and   
auxiliary-cache rules.   
17: Generate $\hat { I } ^ { g } \sim \pi _ { \theta } ( \cdot \mid [ s , S _ { \mathcal { R } ^ { g } } , \mathcal { C } ^ { g } , p _ { \mathrm { f m t } } ] )$   
18: Compute $r _ { \mathrm { a s s o c } } ^ { g }$ and $r _ { \mathrm { t g } } ^ { g } .$   
19: end for   
20: Form $S _ { \mathrm { a s s o c } }$ and ${ \cal S } _ { \mathrm { t g } } ,$ and normalize their rewards.   
21: Compute $\mathcal { I } _ { \mathrm { a s s o c } }$ and $\mathcal { T } _ { \mathrm { t g } }$ as defined above.   
22: Update π<sub>θ</sub> by maximizing $\mathcal { T } _ { \mathrm { t g } } + \lambda \mathcal { T } _ { \mathrm { a s s o c } } .$

Mean IoU measures the average temporal overlap quality over all queries:

$$
\mathrm { m I o U } = \frac { 1 } { N } \sum _ { i = 1 } ^ { N } \mathrm { I o U } \left( T _ { \mathrm { p r e d } } ^ { ( i ) } , T _ { \mathrm { g t } } ^ { ( i ) } \right) ,\tag{36}
$$

where N is the number of evaluated queries.

## Hyperparameter Selection

To select the visual chunk size $W ^ { * }$ in inference, we evaluated $W ^ { * } \in \{ 3 5 , 4 0 , 4 2 , 4 5 \}$ using the same checkpoint and a fixed 400-example subset from each of the four temporalgrounding benchmarks. This resulted in four configurations and 2,000 evaluated examples per configuration. The corresponding macro-averaged mIoU values were 36.276, 36.924, 37.413, and 36.531, respectively. We therefore selected $W ^ { * } = 4 2$ , which is the same as the gradient-gating threshold W.

## Randomness and Reproducibility

We used a training seed of 42 and a data seed of 42. The Hugging Face training framework propagates this seed to Python, NumPy, and $\mathrm { P y }$ Torch random-number generators. The duration-aware training sampler uses a dedicated Py-Torch generator initialized with $4 2 + e$ at epoch e. The sevendataset training mixture was downsampled once using seed 12,345 and stored as a fixed cache, which was reused by all workers. The cached mixture contains 42,381 examples: 7,000 NAQ, 8,000 DiDeMo, 5,000 QuerYD, 3,381 HiRest, 5,000 COIN, 7,000 Momentor, and 7,000 YouCook2 examples.

## Computational Infrastructure

The final model was trained on one node with eight NVIDIA H20 GPUs, each with 97,871 MiB (approximately 96 GB) of device memory. Training used bfloat16 precision, FlashAttention-2, and DeepSpeed ZeRO Stage 3 with both optimizer-state and parameter ofloading to CPU memory. The per-GPU micro-batch size was one and gradient accumulation was two, giving an efective global batch size of $8 \times 1 \times 2 = 1 6$ . The preserved software environment uses Python 3.10.14, PyTorch 2.6.0 with CUDA 12.4 support, Transformers 4.50.0, TRL 0.15.1, DeepSpeed 0.14.5, Accelerate 1.4.0, FlashAttention 2.5.8, Datasets 3.3.1, and Qwen-VL-Utils 0.0.10. The distributed log records NCCL 2.21.5 built for CUDA 12.4 and CUDA driver API version 12.2. The training framework is our custom Time-R1 (Wang et al. 2025) implementation built on Transformers and TRL.

## Complete Training Configuration

We initialized from Qwen2.5-VL-7B-Instruct (Bai et al. 2025). The vision encoder was frozen and no parametereficient adapter was used. The remaining trainable parameters were optimized with AdamW using a learning rate of $1 0 ^ { - 6 }$ , betas (0.9, 0.999), epsilon $1 0 ^ { - 8 }$ , zero weight decay, and gradient-norm clipping at 1.0. We used a linear learningrate schedule without warmup. Training lasted for two epochs and 5,298 optimizer steps. Checkpoints were saved every 500 steps. Gradient checkpointing and vLLM rollout generation were disabled.

Training rollouts used stochastic decoding with temperature 1.0, ${ \mathrm { t o p } } { \cdot } p \ = \ 1 . 0 , \ { \mathrm { t o p } } { - } k \ = \ 5 0$ , a repetition penalty of 1.0, and a maximum of 32 newly generated tokens. The configured maximum prompt length was 4,500 tokens. The model used bfloat16 and FlashAttention-2 throughout training. Videos were decoded at 2 fps.

## Transfer Evaluation on the TimeLens Benchmark

To further evaluate the transferability of SumGround beyond its primary long-video setting, we report additional results under the benchmarks proposed by TimeLens (Zhang et al. 2025a). The benchmarks include Charades-TimeLens, ActivityNet-TimeLens, and QVHighlights-TimeLens, which mainly consist of short- to medium-length videos. Unlike Ego4D-NLQ and TACoS, these datasets do not primarily test the long-horizon evidence identification problem targeted by SumGround. We therefore use this evaluation to examine whether the proposed long-video summarization framework remains competitive when transferred to shorter temporal contexts.

As shown in Table ??, SumGround outperforms Time-R1 across all reported metrics on the three benchmarks. It ranks second to TimeLens on Charades-TimeLens and

<table><tr><td>Method</td><td> ${ \mathbf { } } { \mathbf { } } _ { { \mathbf { } } \mathbf { } \mathbf { } T _ { \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } } }$ </td><td>∆Mem. (GB) ↓</td><td>Latency (s) ↓</td><td>R@0.3↑</td><td>R@0.5↑</td><td>mIoU ↑</td></tr><tr><td>Uniform-Single</td><td>3,829</td><td>1.50</td><td>1.17</td><td>2.49</td><td>1.33</td><td>1.77</td></tr><tr><td>Full-Dense</td><td>51,463</td><td>19.75</td><td>17.58</td><td>2.18</td><td>1.02</td><td>1.74</td></tr><tr><td>SumGround</td><td>51,463</td><td>3.95</td><td>14.78</td><td>16.42</td><td>9.51</td><td>11.83</td></tr></table>

Table 6: Eficiency–accuracy trade-of on Ego4D-NLQ. $T _ { \mathrm { v i s } }$ denotes the average cumulative number of visual tokens processed per sample; for SumGround, it is summed over all chunk-wise passes. ∆Mem. denotes the increase in peak GPU memory relative to the approximately 15.5 GB model-only footprint measured under the same inference setup.
<table><tr><td>Dataset</td><td>Method</td><td>mIoU</td><td>R1@0.3</td><td>R1@0.5</td><td>R1@0.7</td></tr><tr><td rowspan="3">Charades-TimeLens</td><td>Time-R1</td><td>36.6</td><td>57.9</td><td>32.0</td><td>16.9</td></tr><tr><td>TimeLens</td><td>48.8</td><td>70.5</td><td>55.6</td><td>28.4</td></tr><tr><td>SumGround</td><td>41.94</td><td>62.09</td><td>40.11</td><td>20.43</td></tr><tr><td rowspan="3">ActivityNet-TimeLens</td><td>Time-R1</td><td>33.1</td><td>44.8</td><td>31.0</td><td>19.0</td></tr><tr><td>TimeLens</td><td>46.2</td><td>62.8</td><td>51.0</td><td>32.6</td></tr><tr><td>SumGround</td><td>43.92</td><td>59.71</td><td>45.88</td><td>29.71</td></tr><tr><td rowspan="3">QVHighlights-TimeLens</td><td>Time-R1</td><td>49.2</td><td>65.8</td><td>51.5</td><td>36.1</td></tr><tr><td>TimeLens</td><td>56.0</td><td>74.2</td><td>62.7</td><td>43.1</td></tr><tr><td>SumGround</td><td>56.80</td><td>73.78</td><td>61.32</td><td>43.15</td></tr></table>

Table 7: Additional evaluation on the TimeLens benchmark. Best results are shown in bold, and second-best results are underlined.

ActivityNet-TimeLens. On QVHighlights-TimeLens, Sum-Ground achieves the best mIoU of 56.80 and slightly exceeds TimeLens on R1@0.7 (43.15 versus 43.10), while remaining competitive on R1@0.3 and R1@0.5.

SumGround is specifically designed to identify queryrelevant evidence over extended videos, where dense visual content must be summarized and retrieved across multiple chunks. The shorter temporal contexts in the TimeLens benchmark place less emphasis on this long-horizon capability and consequently provide limited scope for its main advantage. Nevertheless, SumGround remains competitive without dataset-specific adaptation, indicating that its longvideo-oriented summarization and retrieval design transfers reliably to shorter videos rather than compromising performance outside its primary setting.

## Eficiency–Accuracy Trade-of under Long-Video Inference

We compare diferent long-video inference strategies in terms of cumulative visual-token processing, peak GPU memory, latency, and temporal-grounding accuracy. This analysis examines whether the advantage of SumGround comes from processing fewer visual tokens or from organizing dense visual evidence more efectively.

Experimental setting. We conduct the analysis on Ego4D-NLQ, which contains long egocentric videos and therefore provides a suitable setting for evaluating long-video inference. All experiments use bfloat16 precision, greedy decoding, and a maximum of 32 newly generated tokens, and are conducted on a single NVIDIA H20 GPU. We exclude the first 20 examples as warm-up samples and report the mean end-to-end latency over the remaining examples.

Compared methods. We compare three inference strategies:

• Uniform-Single uniformly samples a sparse set of frames from the full video such that its visual-token count approximately matches that of one SumGround chunk. All sampled frames are processed in a single forward context.

• Full-Dense uses approximately the same total visualtoken budget as SumGround, but concatenates all sampled frames into a single forward context.

• SumGround processes the same dense visual-token budget sequentially across chunks, causally accumulates query-guided summary states, and performs final grounding over associatively retrieved summaries and, when needed, a local visual cache.

Metrics. $T _ { \mathrm { v i s } }$ denotes the average cumulative number of visual tokens processed per sample. For SumGround, this quantity is summed across all chunk-wise passes and therefore measures total visual-token processing rather than the peak number of visual tokens present in a single context. ∆Mem. denotes the increase in peak GPU memory relative to the approximately 15.5 GB model-only footprint under the same inference setup. Latency denotes the mean end-to-end inference time per sample after warm-up. We report R@0.3, R@0.5, and mIoU for temporal-grounding accuracy.

Discussion. As shown in Table 6, Uniform-Single has the lowest inference cost because it processes only 3,829 visual tokens on average. However, its limited temporal coverage results in poor grounding accuracy.

Full-Dense processes 51,463 visual tokens on average, matching the cumulative visual-token budget of SumGround. Because all visual tokens are placed in a single context, it incurs substantially higher peak GPU memory. Moreover, simply increasing the number of visual tokens in a single context does not improve grounding accuracy over Uniform-Single under this setting.

SumGround processes the same cumulative number of visual tokens as Full-Dense while reducing the additional peak GPU memory to an 80.0% reduction. It also reduces the average inference latency from 17.58 s to 14.78 s. Meanwhile, SumGround improves mIoU from 1.74 to 11.83.

These results indicate that the improvement ofSumGround does not arise from processing fewer visual tokens. Instead, causal summary accumulation and associative retrieval provide a more memory-eficient and efective way to use dense visual evidence for long-video temporal grounding.

## Analysis of Associative Retrieval Coverage

We evaluate the associative retrieval stage by measuring whether the retrieved chunks cover the ground truth temporal interval before final grounding. This analysis separates retrieval quality from the final timestamp prediction.

Because each query-guided summary is causally conditioned on preceding summaries, the retrieved summary states may encode evidence originating from earlier chunks. Therefore, the following metrics measure the explicit temporal coverage of the retrieved source chunks, rather than the complete set of latent evidence available to the grounding decoder.

Evaluation protocol. For SumGround, the selected temporal region is defined as the union of the temporal intervals corresponding to the retrieved summary chunks and, when used, the source chunk of the auxiliary local visual cache.

For the pretrained Qwen2.5-VL baseline, we use the same chunk partition, query-association prompts, and associative retrieval rule as SumGround, but without task-specific RLVR post-training. This comparison isolates the improvement obtained from learning the query-association policy.

For UniTime (Li et al. 2025), we use its Stage-1 coarse proposal as the selected temporal region, since this proposal determines the input region used by its subsequent refinement stage. All coverage metrics are computed before the final grounding or refinement step.

Metrics. Let $G _ { n } = [ t _ { s , n } , t _ { e , n } ]$ denote the annotated interval of example $n ,$ and let $S _ { n }$ denote its selected temporal region, which may be a union of multiple chunk intervals. Over an evaluation set of N examples, we define

$$
\mathrm { G T - C o n t a i n } = \frac { 1 } { N } \sum _ { n = 1 } ^ { N } \mathbb { I } \left[ G _ { n } \subseteq S _ { n } \right] ,\tag{37}
$$

and

$$
\mathrm { G T \mathrm { - O v e r l a p } } = { \frac { 1 } { N } } \sum _ { n = 1 } ^ { N } \mathbb { I } \left[ G _ { n } \cap S _ { n } \neq \emptyset \right] .\tag{38}
$$

GT-Contain measures how often the explicitly retrieved source region covers the complete annotated interval, whereas GT-Overlap measures how often it intersects at least part of the annotation. Both metrics are reported as proportions between zero and one.

<table><tr><td rowspan="2">Method</td><td colspan="2">Ego4D-NLQ</td><td colspan="2">TACoS</td></tr><tr><td>GT-Contain</td><td>GT-Overlap</td><td>GT-Contain</td><td>GT-Overlap</td></tr><tr><td>Qwen2.5-VL</td><td>0.43</td><td>0.46</td><td>0.67</td><td>0.77</td></tr><tr><td>UniTime</td><td>0.43</td><td>0.50</td><td>0.72</td><td>0.89</td></tr><tr><td>SumGround</td><td>0.52</td><td>0.55</td><td>0.78</td><td>0.88</td></tr></table>

Table 8: Explicit retrieved-context coverage on long-video temporal grounding benchmarks. GT-Contain measures whether the retrieved source region fully contains the annotated interval, while GT-Overlap measures whether the two have any temporal intersection. Values are proportions, and higher is better.
<table><tr><td>Dataset</td><td>Variant</td><td>R@0.3</td><td>R@0.5</td><td>R@0.7</td><td>mIoU</td></tr><tr><td>TACoS</td><td>w/o Auxiliary Cache SumGround</td><td>47.14</td><td>29.59</td><td>12.67</td><td>31.45</td></tr><tr><td></td><td></td><td>52.96</td><td>38.07</td><td>18.67</td><td>35.93</td></tr><tr><td>Charades-STA</td><td>w/o Auxiliary Cache SumGround</td><td>77.07 79.54</td><td>61.45 65.91</td><td>34.17 39.57</td><td>52.87 56.10</td></tr></table>

Table 9: Ablation of the auxiliary local visual cache on TACoS and Charades-STA. The ablated variant sets $\boldsymbol { \mathcal { C } } = \boldsymbol { \mathcal { D } }$ throughout training and inference, while all other settings remain unchanged.

Discussion. As shown in Table 8, task-specific RLVR substantially improves the associative retrieval policy over the pretrained Qwen2.5-VL selector. On Ego4D-NLQ, Sum-Ground improves GT-Contain from 0.43 to 0.52 and GT-Overlap from 0.46 to 0.55. It also outperforms UniTime’s Stage-1 proposal on both metrics. On TACoS, SumGround achieves the highest GT-Contain of 0.78, improving over UniTime’s 0.72, while obtaining a comparable GT-Overlap of 0.88 versus 0.89. These results indicate that the learned query-association policy more frequently retrieves chunks covering the complete annotated event, rather than merely intersecting a small portion of it.

At the same time, the absolute coverage rates reveal an important remaining bottleneck. On Ego4D-NLQ, the explicitly retrieved source region fully contains the annotated interval in only 52% of the examples; the corresponding rate on TACoS is 78%. The results show that retrieval remains a meaningful system bottleneck.

## Efect of the Auxiliary Visual Cache

We further examine the contribution of the auxiliary local visual cache to the final grounding stage. While the retrieved query-guided summary states provide compact evidence accumulated across chunks, they may discard fine-grained visual and temporal cues that are useful for precise boundary prediction. The auxiliary cache is designed to complement these summaries with local visual-token states when additional fine-grained evidence is needed.

Analysis. As shown in Table 9, the auxiliary cache consistently improves performance across both datasets and all evaluation metrics. This pattern suggests that the local visual states are especially useful for refining temporal boundaries, rather than merely identifying the coarse location of the queried event. The results indicate that query-guided summary states and the auxiliary local visual cache serve complementary roles. The summary states accumulate queryrelevant evidence across chunks, whereas the auxiliary cache provides fine-grained local cues for precise boundary prediction.

![](images/b667a97d813a63738126a4fb4741e8a217fe91be8e4f0d8549fe3a7d127efb83.jpg)  
Figure 4: Illustration of prompts at both training and inference time.

## Qualitative Result

Illustration of our prompt at training and inference time. Figure 4 presents the prompts used for the temporal video grounding.

Case study on Ego4D-NLQ. We conduct a case study on Ego4D-NLQ, the most challenging benchmark in our evaluation, to better understand the failure modes of relevanceguided chunk selection. Figure 5 shows three examples where the selected anchor chunk does not overlap with the annotated ground-truth span. Case (c) is a genuine failure: although the selected chunk contains a picking action, the interacted object is not the queried bottle. In contrast, Cases (a) and (b) reveal a diferent issue. The selected anchor chunks are semantically consistent with the queries, as they also show the queried objects or events, but they are not covered by the single annotated ground-truth span. This suggests that some apparent selection failures are caused by repeated or visually similar query-relevant events outside the annotated interval. Such cases indicate that evaluation on Ego4D-NLQ may underestimate the quality of relevance-guided selection when multiple valid temporal instances exist in a long egocentric video.

![](images/33fd81cab1c01b9f11b7a1f1618c01c8e41fa94c5b02f1710a0568cbdf73a916.jpg)  
Figure 5: Case study of anchor chunk selection on Ego4D-NLQ. Orange denotes the selected anchor chunk, while green denotes the annotated ground-truth span. Case (c) is a genuine selection error, where the selected chunk contains a similar action but the wrong object. In contrast, Cases (a) and (b) show semantically relevant anchor chunks that match the query but fall outside the annotated ground-truth span, suggesting that some measured failures may reflect to repeated or under-annotated query-relevant events.
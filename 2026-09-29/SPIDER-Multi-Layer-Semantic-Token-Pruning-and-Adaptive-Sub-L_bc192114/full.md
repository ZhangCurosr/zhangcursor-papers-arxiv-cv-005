# SPIDER: Multi-Layer Semantic Token Pruning and Adaptive Sub-Layer Skipping in Multimodal Large Language Models

Tianxiang Chen, Zhentao Tan, Zi Ye, Yue Wu, Xiaobing Tu, Jinkui Ren, Xiantao Zhang, Tao Gong\*, Qi Chu, Nenghai Yu, Xipeng Qiu, Jieping Ye, Fellow, IEEE

Abstract—Multimodal Large Language Models face significant efficiency challenges that stem from two distinct yet coupled sources: data redundancy and computational redundancy. While most methods focus on data redundancy by pruning visual tokens from the output of the visual encoder or computing redundancy in LLM decoders using blockwise importance, the finer-grained inter-layer representation shifts and the distribution differences within the layers themselves have not been fully explored. In this work, we comprehensively investigate this dual-level inefficiency. We posit that intermediate layer tokens from vision encoders should be considered for effective visual token pruning, as semantic focus shifts across layers, with middle-layer tokens capturing more detailed object-centric information that deeper layers may abstract away. Furthermore, we reveal the differential contributions of Attention and FFNs across distinct LLM decoder layers. Building upon these discoveries, we propose SPIDER, a training-free framework that integrates multi-layer Semantic visual token PrunIng with an aDaptive sub-layER skipping mechanism. Experimental evaluations demonstrate that SPIDER consistently maintains strong performance across various MLLM architectures and reduction ratios. For instance, on LLaVA-NeXT-7B, SPIDER reduces FLOPs by 79% while maintaining 96% of the baseline performance.

Index Terms—Multimodal large language models, Token pruning, Layer skipping.

## I. INTRODUCTION

ULTIMODAL large language models (MLLMs) inguage models (LLMs), achieving strong performances on various complex tasks, such as image understanding [1], video understanding [2], and visual reasoning [3]. However, this integration introduces significant overhead, stemming from both data redundancy in the form of lengthy visual token sequences and computational redundancy within the largescale LLM backbone. While visual token pruning has become a dominant strategy to tackle data redundancy [4]–[6], the computational redundancy in processing the remaining tokens through the LLM decoder has received insufficient attention. In this work, we argue that data redundancy and computational redundancy can be combined, and each requires a more finegrained pruning strategy.

First, on the data redundancy front, most state-of-the-art token pruners [4]–[6] rely exclusively on the feature maps from the final layer of the vision encoder to compute token importance. However, our empirical analysis suggests that this “last-layer-only” approach suffers from a significant semantic focus shift. As illustrated in Fig. 1, while the deep layers of a Vision Transformer (ViT) excel at capturing abstract, global context, the middle layers often retain superior object-centric details essential for fine-grained tasks like counting or precise localization. By discarding middle-layer insights, present pruning methods risk eliminating critical visual fragments that are semantically “diluted” in the final layer but vital for accurate reasoning.

Second, on the computational redundancy front, existing acceleration methods primarily focus on coarse-grained blockskipping [7], [8]. These approaches indiscriminately bypass entire LLM decoder blocks, ignoring the internal heterogeneity of the decoder. Our diagnostic experiments reveal a functional divergence between sub-layers: the Attention and Feedforward Network (FFN) components contribute unequally to the refinement of visual tokens across the decoding stages. This finding motivates a more adaptive and fine-grained skipping strategy than treating blocks as monolithic units.

Based on these findings, we propose SPIDER, a novel, training-free framework to improve MLLM inference efficiency. It is built on two core mechanisms: Multi-layer semantic token pruning (MSV-Prune): This strategy uses tokens from both deep and middle layers of the visual encoder for semantic clustering and similarity computation. Adaptive sub-layer skipping (ASL-Skip): This strategy accumulates a skippability score to determine which retained visual tokens should perform layer skipping and at which layer to skip, and an offline sub-layer contribution score (SLC) to decide which specific sub-layer (Attention or FFN) to skip in subsequent LLM decoder layers for skipped tokens.

A key perspective behind SPIDER is that the utility of a visual token changes with depth across the full multimodal inference pipeline. Accordingly, MSV-Prune determines which tokens should enter the decoder at all, while ASL-Skip determines how much computation each retained token still needs as decoding progresses. SPIDER is therefore a unified utilityallocation framework rather than a simple combination of two independent acceleration modules. Our contributions are summarized as follows:

• We explore the semantic focus shift across vision encoder layers, and propose multi-layer semantic token pruning, considering middle and deep layer tokens.

![](images/968cb24d501909c2d5e74f5a79ec66521813c750b4d880639e193c380ed23476.jpg)  
Fig. 1: Visualizations to show the importance of mid-layer features for fine-grained token pruning. We highlight the top 25% most attentive tokens (in purple) from various layers of the CLIP vision encoder, together with their t-SNE embeddings. Mid-layer tokens exhibit a higher key object coverage ratio (R∗), indicating a stronger focus on crucial object-centric details compared to deeper layers, which tend to capture broader global semantics. Averaged over 100 sampled evaluation images, R<sup>∗</sup> peaks at 29.65% at layer 12, compared to 11.28% at layer 1 and 15.83% at layer 24.

• We quantify the fine-grained redundancy in attention and FFN sub-layers of MLLM decoders, and propose adaptive sub-layer skipping to decide whether a retained visual token should skip, when to skip, and skip which part of the layers.

• We propose SPIDER, a unified training-free framework that jointly addresses encoder-side token redundancy and decoder-side computational redundancy, significantly reducing inference cost across various MLLM architectures while maintaining comparable performance.

## II. RELATED WORKS

## A. Multimodal Large Language Models (MLLMs)

MLLMs have emerged as a dominant paradigm for unified vision–language understanding and generation, leveraging the strong priors of large language models (LLMs) to interpret and reason over visual inputs [9], [10]. Pioneering architectures such as LLaVA [10], BLIP-2 [11], and MiniGPT-4 [12] typically employ a frozen vision encoder (e.g., CLIP ViT) to extract image features, which are then projected into the LLM’s textual embedding space via a trainable connector. This design enables zero-shot or few-shot multimodal inference through natural language prompts, achieving remarkable performance across diverse tasks—from visual question answering to image captioning.

Beyond general-purpose benchmarks, MLLMs have been successfully specialized for domain-critical applications. For instance, EmoVerse [13] is specifically designed for affective computing, integrating emotion-aware visual reasoning through multi-task training to jointly handle sentiment analysis, emotion recognition, facial expression interpretation, and emotion cause inference. While MedTVT [14] integrates clinical multimodal data and medical knowledge graphs to enable interpretable, evidence-based multi-disease diagnosis through chain-of-evidence reasoning. These advances underscore the potential of MLLMs in high-stakes scenarios where accuracy, reliability, and contextual grounding are paramount.

However, the computational overhead of MLLMs remains a fundamental barrier to real-world deployment. High-resolution images often yield thousands of visual tokens after patch embedding, drastically inflating sequence length and memory consumption during LLM decoding [6], [15]. This issue is exacerbated in video understanding [2], [16], [17], where temporal redundancy across frames compounds token explosion. Consequently, even state-of-the-art MLLMs struggle with latency and scalability in edge or interactive settings, necessitating principled efficiency mechanisms that preserve semantic fidelity without retraining.

## B. Reducing Redundancy in MLLMs

While MLLMs have achieved remarkable success, their huge computational cost hinders scalable deployment. Existing inference optimizations primarily fall into two categories: token pruning and layer skipping.

Token pruning has gained wide attention due to the high redundancy of visual tokens, which dominate input sequences. Some efforts focused on compressing visual tokens into compact representations [18]–[20], but requiring additional training. Other training-free methods leverage text-visual attention within the LLM to identify less important visual tokens [4], [5], [21]. However, studies like Vispruner [6], VisionZip [22], VTC-CLS [23], and HiPrune [24] argue against the sole reliance on text-visual attention due to positional bias, advocating for visual cues. A common limitation across most token pruning methods is their focus on pruning from the final output, neglecting the multi-layer semantic information inherent in the vision encoder.

![](images/6a647aafc2d40bd231117b61e45fc13fa226d8348527135b60ca7bfc5ec7cdb0.jpg)  
Fig. 2: Visualization of sub-layer skipping, where we visualize either (a) skipping attention sub-layer module or (c) FFN sublayer module. Sub-layer skipping is achieved via replacing the LLM decoder block with corresponding sparse blocks, so that visual tokens are blocked in attention or FFN modules.

Layer skipping addresses the inherent layer redundancy in MLLMs, as observed by ShortV [25]. While some layer skipping approaches involve additional training [26]–[28], our focus remains on training-free methods. ShortV [25], for instance, identifies and replaces less effective layers with sparse versions where visual tokens remain frozen. However, these methods are not fine-grained enough since they often aggressively replace entire LLM layers for all visual tokens, overlooking token and sub-layer importance variance .

Different from previous works, we use the semantic information from middle and deep layer tokens for token pruning. Besides, we systematically analyze the contribution difference of attention and FFN modules in the LLM decoder and adopt a more fine-grained sub-layer skipping mechanism.

Unlike prior work that treats token pruning and layer skipping as independent problems, SPIDER addresses them as two phases of a single depth-dependent utility-allocation problem: visual token utility shifts across encoder layers motivate multilayer pruning before decoding, while the gradual semantic convergence of retained tokens during decoding motivates adaptive sub-layer skipping. Concretely, unlike previous token pruning methods, we explicitly consider the semantic differences between middle and deep layers of the vision encoder, leveraging a multi-layer semantic clustering approach to achieve more comprehensive token pruning. Furthermore, diverging from coarse-grained layer skipping strategies, SPIDER introduces a sub-layer skipping mechanism that adaptively determines whether a retained visual token should skip, when to skip, and which sub-layer to skip. The two modules are thus mutually motivated by the same depth-dependent view of token utility, rather than being independently assembled.

## III. KEY FINDINGS

## A. Semantic Shift across Vision Encoder

We visualize the semantic focus shift and the distribution of the top 25% attentive tokens across CLIP vision encoder layers in Figure 1. t-SNE embeddings across layers show a gradual shift in semantic clustering, with middle layers bridging distinct visual concepts. Notably, middle-layer tokens focus more on key object regions, yielding higher key object coverage ratios $R ^ { * }$ compared to early and deep layers. To quantify this systematically, we compute the average $R ^ { * }$ over 100 sampled evaluation images at three representative layers of the 24-layer CLIP ViT: 11.28% at layer 1 (early), 29.65% at layer 12 (middle), and 15.83% at layer 24 (deep), confirming a clear middle-layer peak consistent with the per-example results in Fig. 1. Deep layers, while capturing broader scene regions, risk missing critical object information. This suggests that effective token pruning should leverage both middle- and deep-layer features, not solely the deepest layer.

## B. Sub-Layer Redundancy in MLLMs

To quantify sub-layer redundancy for specific tokens, we introduce two sparse layers: VSkip-Attn and VSkip-FFN. In the VSkip-Attn layer, visual tokens do not function as queries, acting solely as keys and values. In the VSkip-FFN layer, the visual token input to the FFN is directly bypassed.

We propose the Sub-Layer Contribution (SLC) score to quantify a sub-layer’s impact on the model’s final prediction. The SLC is calculated as the Kullback-Leibler (KL) divergence between the output logits of the original model and a modified one where a specific sub-layer’s operation is selectively bypassed. Specifically, to measure the contribution, we simulate skipping a sub-layer (e.g., attention or FFN) for a subset of tokens X at a given layer i by preventing their hidden states from being updated by that sub-layer. We then define $\mathbf { S L C } _ { i , \mathrm { a t t n } } ^ { X }$ and $\mathbf { S L C } _ { i , \mathrm { f f n } } ^ { X }$ as the KL divergence when the self-attention or FFN sub-layers are skipped for tokens X in layer i, respectively. A lower SLC value indicates less influence on the final output, making the sub-layer a stronger skip candidate. For each layer, we identify the least impactful module by comparing the two sub-layers after the per-module normalization introduced in Eq. (5), and replace modules with the corresponding sparse layers in order of decreasing skippability.

To ensure robustness, we compute SLC offline across three diverse benchmarks. As shown in Figure 3, the resulting SLC distributions exhibit significant variance in absolute magnitude across datasets. However, the relative sub-layer rankings remain highly consistent: averaged across benchmark pairs, the top-16 most-skippable attention sub-layers overlap by 12.7/16 on LLaVA-1.5-7B and 13.3/16 on LLaVA-NeXT-7B; the corresponding FFN overlaps are $1 3 . 0 / 1 6$ and 13.0/16. This rank consistency justifies averaging the SLC across benchmarks to obtain a generalizable backbone-level skipping policy. Another key finding is that $\mathbf { S L C } _ { i , \mathrm { a t t n } } ^ { X }$ is generally lower than $\mathbf { S L C } _ { i , \mathrm { f f n } } ^ { X } .$ This suggests that for visual tokens, the attention sub-layer is often more redundant than the FFN, making it a preferable target for skipping.

![](images/6368e69502afcbe664423f8d5f715cbdc84a197e4735549aff5ea144a0abeece.jpg)  
Fig. 3: The sub-layer contribution scores (SLC) of LLaVA-1.5-7B and LLaVA-NeXT-7B. Lower SLC values indicate a weaker influence of the corresponding attention or FFN sub-layer on the specified tokens. Skipping transformations of visual tokens in such low-impact sub-layers yields minimal divergence from the original model’s output distribution, and the SLC distribution is very different across various benchmarks.

## IV. METHOD

Our training-free framework is illustrated in Figure 4. The input image first undergoes multi-layer semantic visual token pruning to reduce data redundancy, and the retained tokens are then fed to the LLM decoder for adaptive sub-layer skipping. The two components are designed to be coupled rather than independent. MSV-Prune determines which visual tokens should enter the decoder at all by removing semantically redundant tokens before decoding, while ASL-Skip further determines how much computation each retained token still needs as decoding progresses. In this way, SPIDER allocates computation across the full multimodal inference pipeline according to the evolving utility of visual tokens. Details are as follows.

## A. Multi-layer Semantic Visual Token Pruning

The retained visual tokens comprise two subsets: a fraction r of anchor tokens $\mathbf { T } _ { v } ^ { \mathrm { a n c } }$ selected via attention sorting, and complementary tokens $\mathbf { T } _ { v } ^ { \mathrm { c m p } }$ selected under the involvement of middle layer tokens into semantic clustering and sorting. Then $\mathbf { T } _ { v } ^ { \mathrm { a n c } }$ and $\mathbf { T } _ { v } ^ { \mathrm { c m p } }$ are concatenated in spatial indices order and fused with text tokens $T _ { q }$ for LLM decoding.

1) Attention-based anchor tokens: To address the high redundancy in visual inputs, we first perform a saliencybased token pruning step. Drawing inspiration from findings that visual encoder attention is a reliable indicator of patch importance [6], we use the visual encoder’s self-attention scores to select a compact set of important tokens. We average the self-attention matrix A over all heads to obtain $\mathbf { a } _ { v } \in \mathbb { R } ^ { n }$ where for CLIP-like encoders $\mathbf { a } _ { v }$ is the [CLS] row, and for encoders without [CLS] it is the mean attention each patch token receives. A dynamic threshold $\tau$ selects anchor tokens to meet a budget of $n \times R \times r$ tokens, where $R \in ( 0 , 1 )$ is the overall retention ratio and $r \in ( 0 , 1 )$ is a hyperparameter defining the proportion of anchor tokens in the retained set.

$$
\begin{array} { r l } & { \tau = \operatorname* { m i n } \{ t \mid | \{ a _ { i } ^ { v } \geq t \} | \leq n \times R \times r \} , } \\ & { \quad \mathbf { T } _ { v } ^ { \mathrm { a n c } } = \{ t _ { i } ^ { v } \in \mathbf { T } _ { v } \mid a _ { i } ^ { v } \geq \tau \} . } \end{array}\tag{1}
$$

2) Multi-layer Semantic clustering-based complementary tokens: Relying solely on foreground-centric anchor tokens $( \mathbf { T } _ { v } ^ { \mathrm { a n c } } )$ risks losing crucial background context. To mitigate this, we propose a coarse-to-fine strategy to select a set of complementary tokens $( \mathbf { T } _ { v } ^ { \mathrm { c m p } } ) . \mathbf { T } _ { v } ^ { \mathrm { c m p } }$ are obtained by clustering non-anchor tokens into K groups, computing their multi-layer similarity scores, selecting the $N _ { k }$ lowest-scoring tokens from each cluster, and aggregating them.

Coarse-grained Semantic Clustering. A naive redundancy removal on non-anchor tokens is suboptimal, as populous but uniform background regions would exhaust the selection quota, displacing smaller yet unique semantic areas. To address this, we apply K-means clustering to the high-level features $\mathbf { F } _ { v } ^ { L }$ of non-anchor tokens, partitioning them into K semantic clusters $\{ C _ { k } \} _ { k = 1 } ^ { K }$ . The total budget for complementary tokens, $N _ { \mathrm { c m p } } = n \times R \cdot ( 1 - r )$ , is then allocated proportionally to each cluster as a quota $N _ { k }$ , ensuring all semantic groups are represented.

$$
N _ { k } = \operatorname* { m a x } \left( 1 , { \mathrm { r o u n d } } \left( N _ { \mathrm { c m p } } \cdot { \frac { | C _ { k } | } { \sum _ { j = 1 } ^ { K } | C _ { j } | } } \right) \right)\tag{2}
$$

Fine-grained Multi-view Pruning. Within each semantically homogeneous cluster $C _ { k }$ , we select the most informative $N _ { k }$ tokens by leveraging hierarchical features. We observe that middle-layer features $( \mathbf { F } ^ { M } )$ capture fine-grained object-centric details, whereas last-layer features $( \mathbf { F } ^ { L } )$ encode global seman tics, as analyzed in Section III. To select a complementary set, we compute a multi-layer similarity score $\mathbf { S } _ { i j }$ for any pair of tokens (i, j). This score considers their similarity at both middle and deep layer feature levels (multi-layer intra-cluster similarity), and their similarity to the already-selected anchor tokens:

$$
\begin{array} { r l } & { \mathbf { S } _ { i j } = \underbrace { \left( \sin ( \mathbf { F } _ { i } ^ { L } , \mathbf { F } _ { j } ^ { L } ) + \sin ( \mathbf { F } _ { i } ^ { M } , \mathbf { F } _ { j } ^ { M } ) \right) } _ { \mathrm { M u l t i - l a y e r ~ I n t r a - C l u s t e r ~ S i m i l a r i t y } } + } \\ & { \underbrace { \left( \underset { p \in \mathbf { T } _ { v } ^ { \mathrm { a n c } } } { \operatorname* { m a x } } \sin ( \mathbf { F } _ { i } ^ { L } , \mathbf { F } _ { p } ^ { L } ) + \underset { p \in \mathbf { T } _ { v } ^ { \mathrm { a n c } } } { \operatorname* { m a x } } \sin ( \mathbf { F } _ { j } ^ { L } , \mathbf { F } _ { p } ^ { L } ) \right) } _ { \mathrm { S i m i l a r i t y ~ t o ~ A n c h o r ~ T o k e n s } } } \end{array}\tag{3}
$$

where sim $( \cdot , \cdot )$ is the cosine similarity. For each cluster, we compute similarity scores for all tokens, retain the $N _ { k }$ lowestscoring ones, and aggregate them across clusters to form $\mathbf { T } _ { v } ^ { \mathrm { c m p } }$

## B. Adaptive Sub-Layer Skipping

Even after token pruning, significant computational redundancy persists within the MLLM decoder. While prior work has explored coarse-grained layer skipping, this often incurs performance degradation. We observe that sub-layers (i.e., self-attention and FFN) within each decoder block contribute unequally to the final output (Fig. 3). This motivates our finegrained, adaptive sub-layer skipping policy, which dynamically bypasses less critical sub-layers for specific tokens. The decision process is guided by a hybrid mechanism, combining an online, per-token skippability score with an offline, perlayer sub-layer contribution score.

1) Online Skippability Score: To determine if and at which layer a token should perform sub-layer skipping, we compute an online skippability score $\mathbf { S } _ { s a } ( i , \ell )$ for each visual token i at each layer ℓ. It consists of two aspects:

1. Intrinsic Information Entropy $( \mathbf { E } _ { i i } ) { : }$ This measures the semantic uncertainty of a token’s hidden state $h _ { \mathrm { v } } ( i , \ell )$ We compute it as the Shannon entropy of the probability distribution $p ( i , \ell ) ~ = ~ \mathrm { S o f t m a x } ( h _ { v } ( i , \ell ) ~ \cdot ~ W _ { \mathrm { u n e m b e d } } ^ { \top } )$ , which results from projecting the hidden state into the vocabulary space V via the language model’s unembedding matrix $W _ { \mathrm { u n e m b e d } } ~ \in ~ \mathbb { R } ^ { V \times d }$ . Low entropy indicates that the token’s hidden state has semantically converged and is unlikely to benefit from further sub-layer updates, making it a candidate for skipping. We normalize the entropy to [0, 1] to get $\mathbf { E } _ { i i } ( i , \ell )$

2. Image-Text Correlation Factor $( \mathbf { F } _ { i t c } ) { : }$ This is a common factor [4], [5] that assesses a visual token’s relevance to the text query. A token weakly correlated with the text context is more skippable. We measure this as the cosine similarity between the visual token’s hidden state $h _ { \mathrm { v } } ( i , \ell )$ and the averaged text context vector $q _ { t } ( \ell )$ . The cosine similarity sim(i, ℓ) is also normalized to [0, 1] to be $\mathbf { F } _ { i t c } .$ We then define a retention score ${ \bf R } ( i , \ell ) = \bar { \bf E } _ { i i } ( \bar { i } , \ell ) + { \bf F } _ { i t c } ( i , \ell )$ . The final skippability score is its inverse:

$$
\mathbf { S } _ { s a } ( i , \ell ) = \operatorname { R e L U } \left( 1 - \mathbf { R } ( i , \ell ) \right) .\tag{4}
$$

A higher $\mathbf { S } _ { s a }$ indicates a stronger signal for skipping. Since both $\mathbf { E } _ { i i }$ and $\mathbf { F } _ { i t c }$ are independently normalized to [0, 1], their linear combination is a natural and interpretable form of evidence aggregation that requires no additional learned parameters. The ablation in Table VIII(c) confirms that both components contribute meaningfully, validating this design.

Algorithm 1 Pseudocode for SPIDER   
Require: x, E, D, R, r, T<sub>skip</sub>, w<sub>1</sub>, w<sub>2</sub>   
Ensure: y   
1: // Stage 1: Multi-layer Visual Token Pruning   
2: Extract mid/last features: $\mathbf { F } ^ { M } , \mathbf { F } ^ { L } \gets \mathcal { E } ( x )$   
3: Select anchor tokens $\mathbf { T } ^ { \mathrm { a n c } }$ (top-nRr by attention score)   
4: Cluster non-anchor tokens on $\mathbf { F } ^ { L }$ ; select representatives   
minimizing multi-layer similarity to anchors and within  
cluster redundancy   
$\bar { \mathfrak { s } } \colon \mathbf { T } _ { \mathrm { i n p u t } }  [ \mathbf { T } _ { q } ; \mathrm { c o n c a t } ( \mathbf { T } ^ { \mathrm { a n c } } , \mathbf { T } ^ { \mathrm { c m p } } ) ]$   
6: // Stage 2: Adaptive Sub-Layer Skipping   
7: Precompute $\mathrm { S L } \bar { \mathrm { C } } _ { \mathrm { A t t n / F F N } } ^ { \mathrm { N o r m } } ( \ell )$ and the per-layer skip target   
m(ℓ) ← arg max<sub>module</sub> $\dot { \mathrm { S L C } } _ { m o d u l e } ^ { \mathrm { N o r m } } ( \bar { \ell } )$   
8: $\mathrm { S } _ { s k i p } ( i ) \gets 0 ; \ : \ : \mathcal { M } \gets \emptyset ; \ : \ : h \gets \mathbf { T } _ { \mathrm { i n p u t } } \quad / /$ M: tokens in   
skip mode   
9: for $\ell = 1$ to L do   
10: if $\ell \leq L / 2$ then   
11: // decision window: accumulate evidence, admit new   
tokens   
12: for each visual token i /∈ M do   
13: Compute $\mathbf { E } _ { i i } ( { \it i } , { \it \ell } ) , \ \dot { \mathbf { F } } _ { i t c } ( { \it i } , { \boldsymbol { \ell } } ) ,$ and $\begin{array} { r l } { \mathbf { S } _ { s a } ( i , \ell ) } & { { } = } \end{array}$   
$\mathrm { R e L U } ( 1 - { \bf E } _ { i i } ( i , \ell ) - { \bf F } _ { i t c } ( i , \ell ) )$   
14: $\mathrm S _ { s k i p } ( i ) \mathrel { + } = w _ { 1 } \cdot \dot { \mathbf S } _ { s a } ( i , \ell ) + w _ { 2 } \cdot \mathrm { S L C } _ { m ( \ell ) } ^ { \mathrm { N o r m } } ( \ell )$   
15: if $\mathrm { S } _ { s k i p } ( i ) \geq T _ { \mathrm { s k i p } }$ then   
16: ${ \mathcal { M } } \gets { \mathcal { M } } \cup \{ i \}$ ; target(i) ← m(ℓ) // frozen   
at admission   
17: end if   
18: end for   
19: end if   
20: h ← Forward layer ℓ, bypassing sub-layer target(i) for   
every $i \in \mathcal { M }$   
21: end for   
22: y ← LMHead(h)   
23: return y

2) Offline Sub-Layer Contribution Score: Once a token is deemed skippable, we must decide which sub-layer (attention or FFN) to bypass. To inform this, we pre-compute a Sub-Layer Contribution (SLC) score via offline profiling. For each layer $\ell , \ \mathrm { S L C } _ { a t t n } ( \ell )$ and $\mathrm { S L C } _ { f f n } ( \ell )$ are the average KLdivergence between the original model’s output and the output when the respective sub-layer is skipped via sparse layer replacement, evaluated over multiple multimodal benchmarks. A low KL-divergence implies the module is less critical. We normalize these to obtain module-level skippability scores, where a higher score means more skippable:

$$
\begin{array} { r l r } & { } & { \mathrm { S L C } _ { m o d u l e } ^ { N o r m } ( \ell ) = 1 - \frac { \mathrm { S L C } _ { m o d u l e } ( \ell ) - \mathrm { S L C } _ { m o d u l e , m i n } } { \mathrm { S L C } _ { m o d u l e , m a x } - \mathrm { S L C } _ { m o d u l e , m i n } + \epsilon } , } \\ & { } & { m o d u l e \in \{ \mathrm { A t t n } , \mathrm { F F N } \} . \quad } \end{array}\tag{5}
$$

At each layer $\ell ,$ the sub-layer with the higher $\mathrm { S L C } ^ { N }$ orm score is designated as that layer’s skip target $m ( \ell )$ , and Table X lists $m ( \ell )$ for every decoder index. Two aspects of this designation are worth making explicit. First, the comparison is made between the two normalized scores rather than between the raw KL values, because the attention and FFN families occupy systematically different KL ranges—skipping an FFN sub-layer is far more damaging than skipping an attention sub-layer (Table III)—so a comparison of raw values would assign almost every layer to attention and forfeit the per-layer discrimination visible in Table X, where five of the 32 decoder layers of LLaVA-1.5-7B (and seven of LLaVA-NeXT-7B) are designated FFN. Second, m(ℓ) is a per-layer property, whereas the sub-layer that a given token bypasses is fixed at the layer where that token enters skip mode and is then retained for all deeper layers. Tokens admitted at different layers therefore inherit different targets, so at any single layer some tokens may bypass attention while others bypass the FFN; this is what the bypass arcs in Fig. 4 depict.

![](images/c2b846e28968314e9a80d07123ef2be2436e0c2e16962c558b27e9c70147e779.jpg)  
Fig. 4: Illustration of SPIDER. We begin by token pruning using the semantic information from the middle and last layers. The retained tokens are fed to the LLM for adaptive sub-layer skipping.

## C. Score Fusion and Cumulation

The final skipping decision integrates both online and offline scores. At each layer ℓ (up to the network’s midpoint, $L / 2 )$ we compute a fused score:

$$
\operatorname { S } _ { f u s e } ( i , \ell ) = w _ { 1 } \cdot \mathbf { S } _ { s a } ( i , \ell ) + w _ { 2 } \cdot \operatorname { S L C } ^ { N o r m } ( \ell ) ,\tag{6}
$$

where SL $\mathrm { \langle C ^ { \mathit { N o r m } } ( \ell ) \Psi \equiv \Sigma \Sigma ^ { \mathit { N o r m } } ( \ell ) }$ is the score of that layer’s designated skip target, and $w _ { 2 }$ is fixed to 1 throughout, so $w _ { 1 } / w _ { 2 }$ also sets the magnitude of $\mathrm { S } _ { f u s e } .$ . This score is accumulated as $\mathrm S _ { s k i p } ( i , \ell ) = \mathrm S _ { s k i p } ( i , \ell - 1 ) + \mathrm S _ { f u s e } ( i , \ell )$ Once $\mathrm { S } _ { s k i p } ( i , \ell )$ exceeds a threshold $T _ { \mathrm { s k i p } } ,$ token i enters a “skip mode”: its target is frozen as $\mathrm { t a r g e t } ( i ) = m ( \ell )$ and is then bypassed in every remaining layer up to L, while the accumulation itself is confined to $\ell \leq L / 2$ . Tokens not in skip mode proceed normally.

## V. EXPERIMENTS

## A. Experiment Settings

Models. We evaluate our method on LLaVA-1.5-7B [10] and the high-resolution LLaVA-NeXT-7B [29]. LLaVA-1.5 generates 576 visual tokens from 336×336 images. LLaVA-NeXT’s sub-image partitioning strategy handles flexible resolutions, yielding 2,880 tokens for our evaluation, a 5× increase.

Datasets. We evaluate on various multimodal benchmarks, including GQA [30], VQAv2 [31], MME [32], TextVQA [33], POPE [34], MMB [35] , MMVet [36], MMStar [37] and DocVQA [38]. VQAv2 [31] evaluates visual recognition via open-ended questions on 265k MSCOCO images, with adversarially balanced answers to reduce language bias. We use the test-dev set (107k image-question pairs), scored against 10 human answers. GQA [30] tests structured reasoning using scene graph–generated questions on Visual Genome images. Evaluation is based on accuracy over 12.6k test-dev pairs. TextVQA [33] requires models to read and reason about OCR text in Open Images (e.g., signs, labels). Evaluated on 5k validation pairs. POPE [34] measures object hallucination in MLLMs by querying object presence in MSCOCO images, reporting average F1 across three sampling strategies (8.9k pairs). MME [32] assesses perception (OCR, object attributes, fine-grained recognition) via 14 binary subtasks; we report the perception score on 2,374 pairs. MMB [35] offers a hierarchical multimodal evaluation with multiple-choice questions across perception and reasoning. Both English (4,377) and Chinese (4,329) versions are used. MM-Vet [36] integrates six core capabilities (e.g., OCR, spatial reasoning, math) into 16 tasks, evaluated by ChatGPT on 218 diverse samples. MMStar [37] provides a clean, human-curated benchmark of 1,500 visually dependent samples with minimal data leakage, measuring true multimodal gains across 6 capabilities and 18 axes. DocVQA [38] focuses on document understanding, requiring fine-grained textual and spatial reasoning over 50k questions on 12k+ document images.

Baselines. We benchmark SPIDER against training-free efficient MLLM methods. Token pruning methods include FastV [4], which prunes a fixed ratio R of visual tokens post-layer K based on attention scores; VTW [39], which discards all visual tokens after layer $K ;$ and VisPruner [6], which retains V visual tokens from the vision encoder. Layer skipping method includes ShortV [25], which replaces l LLM layers with its ShortV layers. Our SPIDER retains N visual tokens after the pruning stage and then allows $R _ { s }$ of them to skip certain sublayer modules. All the methods are compared at similar FLOPs reduction ratios.

To perform an ablation study on SPIDER’s components, we introduce additional specialized baselines. We evaluate our token pruning module, MSV-Prune, against other token pruners like SparseVLM [5] and VisionZip [22]. To assess the efficacy of our layer skipping module, ASL-Skip, we benchmark it against ShortV [25] and naive strategies that uniformly skip all attention or FFN modules.

TABLE I: Comparison of training-free MLLM efficiency methods. FLOPs Ratio denotes the proportion of FLOPs retained relative to the vanilla model. Best results are in bold.
<table><tr><td>Method</td><td>TFLOPs</td><td>Ratio</td><td>VQAv2</td><td>GQA</td><td>MMStar</td><td>MME</td><td>MMB</td><td>POPE</td><td>MMVet</td><td>TextVQA</td><td>DocVQA</td><td>Acc. (%)</td></tr><tr><td></td><td></td><td colspan="9">LLaVA-1.5-7B (Upper Bound, All 576 Visual Tokens)</td><td></td></tr><tr><td>Vanilla</td><td>8.5</td><td>100%</td><td>76.5</td><td>61.9</td><td>33.7</td><td>1510.7</td><td>64.1</td><td>85.9</td><td>31.1</td><td>58.2</td><td>21.5</td><td>100%</td></tr><tr><td>Approximately 55% TFLOPs</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>FastV  $( K = 2 , R = 5 0 \% )$ </td><td>4.9</td><td>58%</td><td>73.5</td><td>60.2</td><td>32.4</td><td>1475.6</td><td>64.3</td><td>84.0</td><td>29.8</td><td>57.2</td><td>17.3</td><td>95.54%</td></tr><tr><td>VTW (K = 16)</td><td>4.7</td><td>55%</td><td>66.3</td><td>55.1</td><td>32.8</td><td>1497.0</td><td>64.0</td><td>82.8</td><td>19.2</td><td>55.3</td><td>16.2</td><td>88.93%</td></tr><tr><td>ShortV ( l = 19)</td><td>4.7</td><td>55%</td><td>75.7</td><td>60.9</td><td>33.3</td><td>1503.1</td><td>64.8</td><td>86.2</td><td>27.9</td><td>55.1</td><td>17.9</td><td>96.08%</td></tr><tr><td>VisPruner (V = 288)</td><td>4.7</td><td>55%</td><td>76.3</td><td>60.8</td><td>33.3</td><td>1477.9</td><td>63.7</td><td>86.3</td><td>30.3</td><td>57.8</td><td>20.7</td><td>98.60%</td></tr><tr><td>SPIDER  $( N = 3 8 4 , R _ { s } = 5 5 \% )$ </td><td>4.8</td><td>56%</td><td>76.6</td><td>60.9</td><td>34.9</td><td>1498.5</td><td>64.9</td><td>86.6</td><td>30.7</td><td>57.9</td><td>20.9</td><td>99.87%</td></tr><tr><td>Approximately 25-30% TFLOPs</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>FastV  $( K = 2 , R = 7 5 \% )$ </td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>ShortV ( l = 31)</td><td>2.6 2.1</td><td>30% 25%</td><td>74.3 56.1</td><td>56.6</td><td>30.8</td><td>1394</td><td>62.3</td><td>79.2</td><td>30.3</td><td>56.2</td><td>16.2</td><td>92.33% 67.76%</td></tr><tr><td>VisPruner (V = 128)</td><td>2.3</td><td>27%</td><td>75.8</td><td>47.7 58.2</td><td>29.3 32.9</td><td>771.5 1461.4</td><td>56.1 62.7</td><td>58.5 84.6</td><td>17.2 28.6</td><td>35.7 57.0</td><td>9.2 18.0</td><td>95.27%</td></tr><tr><td>SPIDER (N = 144, Rs = 35%)</td><td>2.2</td><td>26%</td><td>76.0</td><td>58.6</td><td>33.7</td><td>1458.2</td><td>63.7</td><td>84.8</td><td>30.3</td><td>57.3</td><td>18.5</td><td>96.73%</td></tr><tr><td></td><td></td><td></td><td>LLaVA-NeXT-7B (Upper Bound, All 2880 Visual Tokens)</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td colspan="14">Vanilla</td></tr><tr><td></td><td>42.7</td><td>100%</td><td>80.0</td><td>62.9</td><td>37.1</td><td>1519.0</td><td>67.1</td><td>86.3</td><td>38.5</td><td>59.6</td><td>68.4</td><td>100%</td></tr><tr><td>Approximately 50% TFLOPs</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>FastV  $( K = 2 , R = 5 0 \% )$ </td><td>22.0</td><td>52%</td><td>79.5</td><td>63.0</td><td>36.5</td><td>1482.0</td><td>66.3</td><td>86.5</td><td>36.8</td><td>58.1</td><td>59.0</td><td>97.10%</td></tr><tr><td>VTW (K = 16)</td><td>21.8</td><td>51%</td><td>75.6</td><td>55.8</td><td>37.6</td><td>1518.2</td><td>67.1</td><td>84.9</td><td>18.5</td><td>57.3</td><td>58.2</td><td>89.71%</td></tr><tr><td>ShortV (l = 19)</td><td>21.6 21.8</td><td>51%</td><td>78.8</td><td>63.4</td><td>37.8</td><td>1525.1</td><td>67.2</td><td>86.9</td><td>31.7</td><td>58.3</td><td>59.8</td><td>96.67%</td></tr><tr><td>VisPruner  $( V = 1 6 0 0 )$  SPIDER (N = 1920,  $R _ { s } = 5 0 \% )$ </td><td></td><td>51%</td><td>79.9 80.2</td><td>62.5</td><td>37.3</td><td>1493.1</td><td>66.7</td><td>88.0</td><td>37.3</td><td>59.4</td><td>63.8</td><td>98.81%</td></tr><tr><td></td><td>21.9</td><td>51%</td><td></td><td>62.6</td><td>37.7</td><td>1510.5</td><td>66.6</td><td>88.2</td><td>36.6</td><td>59.7</td><td>64.6</td><td>99.11%</td></tr><tr><td>Approximately 20-25% TFLOPs</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>FastV  $( K = 2 , R = 8 9 \% )$ </td><td>8.5</td><td>20%</td><td>71.9</td><td>55.9</td><td>32.1</td><td>1282.9</td><td>53.4</td><td>71.7</td><td>25.9</td><td>55.7</td><td>43.7</td><td>81.89%</td></tr><tr><td>ShortV ( l = 29)</td><td>9.7</td><td>23%</td><td>58.6</td><td>49.7</td><td>30.4</td><td>884.5</td><td>51.2</td><td>56.6</td><td>21.5</td><td>36.5</td><td>14.1</td><td>63.56%</td></tr><tr><td>VisPruner  $( V = 6 4 0 )$ </td><td>9.1</td><td>21%</td><td>79.8</td><td>61.4</td><td>36.5</td><td>1490.8</td><td>65.2</td><td>85.9</td><td>36.7</td><td>59.3</td><td>50.6</td><td>95.49%</td></tr><tr><td>SPIDER  $( N = 7 1 0 , R _ { s } = 3 0 \% )$ </td><td>9.0</td><td>21%</td><td>79.8</td><td>61.8</td><td>37.3</td><td>1492.2</td><td>65.7</td><td>87.8</td><td>35.9</td><td>59.5</td><td>51.5</td><td>96.09%</td></tr></table>

TABLE II: Comparisons of our MSV-Prune, with other SOTA training-free token pruning methods. Best results are in bold.
<table><tr><td>Method</td><td>GQA</td><td>TextVQA</td><td>POPE</td><td>Acc. (%)</td></tr><tr><td colspan="3">Upper Bound, All 2880 Tokens (100%)</td><td></td><td></td></tr><tr><td>LLaVA-NeXT-7B</td><td>62.9</td><td>59.6</td><td>86.3</td><td>100.0%</td></tr><tr><td>Retain 320 Tokens ( ↓ 88.9%)</td><td>55.9</td><td>55.7</td><td>71.7</td><td>88.47%</td></tr><tr><td>FastV SparseVLM VisionZip VisPruner MSV-Prune (Ours)</td><td>56.5 58.1 58.4 59.0</td><td>52.4 57.6 57.6 58.1</td><td>73.5 75.0 80.4 83.3</td><td>87.64% 91.97% 94.22% 95.93%</td></tr><tr><td colspan="3">Retain 160 Tokens (↓ 94.4%)</td><td></td><td></td></tr><tr><td>FastV SparseVLM</td><td>49.8 50.2</td><td>51.9 45.1</td><td>51.7 54.6</td><td>75.39% 72.92%</td></tr><tr><td>VisionZip VisPruner</td><td>54.3 54.7</td><td>54.7 56.0</td><td>59.4 72.9</td><td>82.31% 88.46% 89.45%</td></tr></table>

Aggregate Accuracy Metric. To summarize performance across benchmarks with different native scales, we report an aggregated relative accuracy metric, denoted as Acc. (%), defined as

$$
\mathrm { A c c . ~ } ( \% ) = \frac { 1 } { \left| \mathcal { B } \right| } \sum _ { b \in \mathcal { B } } \frac { s _ { b } ^ { \mathrm { m e t h o d } } } { s _ { b } ^ { \mathrm { v a n i l l a } } } \times 1 0 0 ,\tag{7}
$$

where $\boldsymbol { B }$ denotes the set of benchmarks included in the corresponding table, $s _ { b } ^ { \mathrm { m e t h o d } }$ is the score of the evaluated method on benchmark $b ,$ and $s _ { b } ^ { \mathrm { v a n i l l a } }$ is the score of the corresponding vanilla model on the same benchmark. This formulation first normalizes each benchmark by its vanilla performance and then averages across benchmarks, making the aggregate metric comparable despite different score ranges. For MME, we directly use the reported perception score as $s _ { b }$ and normalize it by the vanilla MME perception score in the same manner as other benchmarks. All reported numbers are single-run results obtained under a fixed random seed with deterministic decoding; when comparing methods at matched compute, differences below roughly 0.3 points should therefore be read as parity rather than as a strict ordering.

TABLE III: Comparisons of our ASL-Skip with other trainingfree layer skipping/pruning methods. No visual token is pruned. Most of these methods are evaluated at around 80 % of the original TFLOPs for fair comparisons. Best results are in bold. “R” denotes TFLOPs ratio.
<table><tr><td>Method</td><td>R</td><td>MMStar</td><td>TextVQA</td><td>MME</td><td>Acc. (%)</td></tr><tr><td colspan="6">Upper Bound, All 576 Tokens (100%)</td></tr><tr><td>LLaVA-1.5-7B</td><td>100%</td><td>33.7</td><td>58.2</td><td>1510.7</td><td>100%</td></tr><tr><td>ShortV</td><td>81%</td><td>33.8</td><td>57.3</td><td>1503.3</td><td>99.42%</td></tr><tr><td>Skip All Attn</td><td>82%</td><td>34.1</td><td>51.1</td><td>1300.6</td><td>91.69%</td></tr><tr><td>Skip All FFN</td><td>38%</td><td>28.9</td><td>40.9</td><td>875.9</td><td>71.34%</td></tr><tr><td>Skip Partial FFN</td><td>81%</td><td>32.6</td><td>57.4</td><td>1483.1</td><td>97.84%</td></tr><tr><td>ASL-Skip (Ours)</td><td>80%</td><td>33.7</td><td>58.1</td><td>1504.9</td><td>99.81%</td></tr></table>

Implementation Details for Specific Architectures. For Qwen-series models with a PatchMerger, MSV-Prune extracts intermediate and final ViT hidden states and passes both through the same PatchMerger to obtain post-merger representations at two semantic depths. All importance computation and token selection are performed on these post-merger tokens, ensuring no conflict with the spatial merging structure. DeepStack feature tensors share the same token layout as the main stream by design, so the retained token indices apply to them directly. ASL-Skip operates entirely within the LLM decoder on the retained post-merger tokens and is independent of the vision encoder structure, so no additional adaptation is needed for Qwen-series models.

For LLaVA-NeXT with AnyRes, MSV-Prune is applied independently to each tile in the shared ViT feature space before the multi-modal projector. The base tile and all local tiles share the same pruning rule and token budget, with no extra quota reserved for the global tile. Per-tile masks are propagated together with the standard AnyRes reshape and unpadding operations, and the retained base-tile and localtile tokens are then concatenated into the flat visual sequence consumed by the LLM decoder.

For Qwen3-VL-8B-Instruct, the number of visual tokens is input-dependent, so SPIDER is configured by the visual retention ratio $R _ { v }$ rather than a fixed retained-token number. The two settings in Table V use $R _ { v } = 0 . 5$ and $R _ { v } = 0 . 4$ respectively, and are chosen to approximately match the compared baselines in end-to-end FLOPs. ASL-Skip is tokenadaptive rather than layer-fixed; empirically, most retained visual tokens begin skipping after the first four decoder layers, with an overall skipped-token proportion of around 70% and 60%, respectively. FLOPs are measured over the full multimodal pipeline under the same prompt and image preprocessing setup.

## B. Quantitative Results

We apply SPIDER to the classic LLaVA-1.5 and LLaVA-NeXT models and comprehensively compare against prior approaches. As shown in Table I, SPIDER consistently matches or surpasses baselines across multiple benchmarks at similar or lower FLOPs, outperforming other training-free methods and achieving the best trade-off between efficiency and performance. Notably, even under a relatively aggressive FLOPs reduction ratio to 20%, SPIDER applied on LLaVA-NeXT-7B retains 96.09% of the overall performance, whereas skiplayer-based baselines such as ShortV degrade substantially. On challenging benchmarks such as MMVet and MMStar, which demand strong spatial awareness and reasoning capabilities, SPIDER maintains competitive accuracy, underscoring its robustness under high compression.

More specifically, we ablate the two core components of SPIDER separately: MSV-Prune in Table II and ASL-Skip in Table III.

Table II shows that MSV-Prune consistently surpasses SOTA pruners across all retention ratios. This confirms that relying solely on the vision encoder’s last layer introduces a semantic focus shift where more fine-grained object-centric information is sacrificed for abstract global context. By incorporating middle-layer features as semantic anchors, MSV-Prune ensures a more holistic representation of the visual scene, which is reflected in the significantly higher scores on OCR-centric tasks like TextVQA.

Table III reveals that ASL-Skip preserves overall performance more effectively than existing layer-pruning methods at the same computational cost, underscoring the importance of fine-grained sub-layer skipping. Even under identical FLOPs ratios, non-adaptive partial FFN skipping still incurs a 2% accuracy drop compared to ASL-Skip, while skipping all FFN sub-layers leads to nearly 30% degradation. These results indicate that skipping attention sub-layers is significantly less harmful than skipping FFNs, and that adaptive skipping (ASL-Skip) is the optimal strategy.

To further demonstrate the architecture-agnostic nature of SPIDER, we extend our evaluation to Qwen2.5-VL-3B-Instruct and Qwen3-VL-8B-Instruct. As shown in Table IV and Table V, SPIDER maintains competitive overall accuracy across different TFLOPs ratios on both models. On Qwen3- VL-8B-Instruct, we further compare against more recent methods including PruneSID [40], IVC-Prune [41], iLLaVA [42], and ERASE [43]. SPIDER retains 99.06% and 98.33% of the vanilla performance at the approximately 55% and 45% TFLOPs settings, respectively. At matched compute, SPIDER is on par with the strongest recent baseline in aggregate accuracy, while leading on ChartQA and DocVQA under both settings and also obtaining the best GQA score at the 45% TFLOPs setting. These results confirm that the redundancy patterns targeted by SPIDER, namely the semantic focus shift across vision encoder layers and the sub-layer heterogeneity within the LLM decoder, are intrinsic properties of the MLLM paradigm that generalize across model families rather than being artifacts of specific architectures.

Beyond its training-free mode, SPIDER adapts to a trainingaware framework for performance enhancement. Table VI shows results after fine-tuning on LLaVA-1.5-7B using LoRA for 1 epoch with 665K instruction data, following the same training setting as LLaVA-PruMerge [20]. Notably, the training-free SPIDER already matches PruMerge+, which requires fine-tuning, under the aggregate metric of Eq. (7) (both 99.0%), leading on POPE (84.8 vs. 84.0) and TextVQA (57.4 vs. 57.1) while trailing on MMB (63.9 vs. 64.9); post-training refinement then elevates SPIDER to 99.5%.

## C. Qualitative Results

Figure 7 shows MSV-Prune retains semantically critical tokens, while ASL-Skip progressively skips less informative ones. Even at aggressive skip rates, core semantics are maintained, enabling accurate scene and object queries.

We additionally explore the extent of knowledge boundary drift caused by SPIDER via conducting a fine-grained instance-level analysis on the POPE test set using LLaVA-1.5-7B with our SPIDER method (at 56% of original FLOPs). Accuracy improves from 85.9% to 86.6%, reflecting a net gain: the number of originally incorrect predictions corrected by SPIDER exceeds the number of originally correct ones flipped to wrong by 0.7%. Qualitative examples are provided in Fig. 5. This gain stems from SPIDER’s retention of mid-layer tokens encoding object-centric regions, which enhances discriminative grounding and reduces false positives—particularly in hallucination-sensitive queries.

![](images/2da23b48c39de56aab227c8b37fb8c0f86173b20f50124225e91900606bc2b62.jpg)  
Fig. 5: Instance-level visualization of boundary shift.

![](images/e2da2e0e1ee96c7440d8296b53045124eed7f8ec8ec458967970645973b4cc33.jpg)  
Fig. 6: KL divergence comparison with magnitude pruning across LLM layers.

## D. Efficiency Analysis

Figure 8 demonstrates the efficiency gains of SPIDER across varying token reduction ratios on LLaVA-1.5-7B. As the number of retained visual tokens decreases from 576 to 128, both prefill and decode latencies are significantly reduced—prefill latency drops from 67.4 ms to 48.2 ms (a 28.5% reduction), and decode latency decreases from 23.1 ms to 20.6 ms (a 10.8% reduction). Concurrently, throughput increases from 43.3 to 45.2 tokens per second, indicating improved inference speed under lower computational load. These results highlight SPIDER’s ability to achieve substantial latency reduction and higher throughput while maintaining strong semantic fidelity, as evidenced by its consistent performance in downstream tasks even at aggressive token pruning rates. This balance between efficiency and accuracy makes SPIDER suitable for real-time vision-language applications where low-latency responses are essential.

The larger gain in prefill relative to decoding is expected: prefill computation scales directly with sequence length and is dominated by parallel matrix operations, whereas autoregressive decoding processes one token at a time and is increasingly bottlenecked by KV-cache access and memory bandwidth rather than arithmetic throughput. For ASL-Skip, the token-level dynamic sparsity introduced by our method is irregular and input-dependent, which standard dense GPU kernels cannot fully exploit without hardware-aware sparse implementations. We note this as a current practical limitation and discuss token regrouping and sparse kernel support as directions for future decode-stage acceleration.

TABLE IV: Results on Qwen2.5-VL-3B-Instruct. Best results are in bold.
<table><tr><td>Method</td><td>|MMB MMBCN</td><td></td><td>POPE</td><td> $\mathbf { S Q A } ^ { \mathrm { I M G } }$ </td><td>VizWiz</td><td>Acc. (%)</td></tr><tr><td colspan="7">Vanilla, 100% Tokens</td></tr><tr><td>Qwen2.5-VL-3B-Instruct</td><td>77.3</td><td>73.0</td><td>87.0</td><td>80.4</td><td>68.3</td><td>100 %</td></tr><tr><td colspan="7">Approximately 35 % TFLOPs</td></tr><tr><td>FastV</td><td>74.4</td><td>70.6</td><td>85.0</td><td>79.3</td><td>66.9</td><td>97.4%</td></tr><tr><td>VisionZip</td><td>74.9</td><td>69.8</td><td>85.4</td><td>80.1</td><td>67.1</td><td>97.7%</td></tr><tr><td>HiPrune</td><td>75.8</td><td>71.3</td><td>86.0</td><td>80.0</td><td>67.5</td><td>98.6%</td></tr><tr><td>SPIDER (Ours)</td><td>76.2</td><td>72.0</td><td>86.3</td><td>80.1</td><td>67.8</td><td>99.1%</td></tr><tr><td colspan="7">Approximately 25 % TFLOPs</td></tr><tr><td>FastV</td><td>72.4</td><td>69.2</td><td>82.7</td><td>79.6</td><td>66.2</td><td>95.9%</td></tr><tr><td>VisionZip</td><td>73.5</td><td>67.4</td><td>84.6</td><td>80.0</td><td>66.3</td><td>96.2%</td></tr><tr><td>HiPrune</td><td>74.0</td><td>69.3</td><td>84.7</td><td>80.3</td><td>66.5</td><td>97.1%</td></tr><tr><td>SPIDER (Ours)</td><td>75.3</td><td>69.8</td><td>85.2</td><td>79.8</td><td>67.4</td><td>97.8%</td></tr></table>

## E. Comparison with Magnitude Pruning

Prior work has shown that token pruning can induce knowledge boundary drift, which means instances once answered correctly may become incorrect after pruning, and vice versa, posing risks in applications requiring reliable responses to critical queries. To address this, we go beyond conventional benchmarks and perform an instance-level analysis of SPI-DER’s semantic fidelity on the POPE test set using LLaVA-1.5-7B at 56% of original FLOPs. Accuracy improves from 85.9% to 86.6%, a net gain of 0.7%, as SPIDER corrects more originally wrong predictions than it flips correct ones. This stems from its retention of mid-layer tokens encoding object-centric regions, enhancing discriminative grounding and reducing false positives, especially in hallucination-sensitive queries. We also compare SPIDER with magnitude pruning: for each layer ℓ and visual token i, we compute L1 norms of attention and FFN outputs, average them into layer-wise scores $M _ { \mathrm { A t t n } } ( \ell )$ and $M _ { \mathrm { F F N } } ( \ell )$ , and prune the lower-magnitude submodule to obtain a magnitude-pruned LLaVA. KL divergence between per-layer logits of the pruned and original models quantifies post-pruning distortion. As shown in Figure 6, SPI-DER achieves lower KL divergence, confirming its superior layer-skipping strategy with minimal degradation.

## F. Ablation study and analysis

Ablation on MSV-Prune (Table VII). Under a fixed token budget (1920 tokens, $r ~ = ~ 0 . 7 )$ , removing any component of MSV-Prune (semantic clustering, middle-layer tokens, or similarity scores $S _ { i j } )$ reduces accuracy, highlighting the importance of modeling semantic shifts across layers. Notably, discarding middle-layer tokens hurts fine-grained tasks like OCR (e.g., TextVQA) more severely, and using only middlelayer tokens to compute intra-cluster similarity helps extracting object-centric cues. Replacing cosine similarity in $S _ { i j }$ with MSE or KL divergence also degrades performance: MSE is sensitive to non-semantic magnitude differences [44], and KL divergence may not be the most appropriate choice here, as the layer features do not naturally form valid probability distributions. The full model (“All Equipped”) achieves the best average accuracy.

![](images/6d9abff6e04dfa8294884fc26782780bf54b6a33aefa8cb885b7db4c2b73529d.jpg)  
Fig. 7: Visualization of retained tokens by MSV-Prune and unskipped tokens by ASL-Skip; anchors in purple (r = 0.7), complementary tokens in yellow.

TABLE V: Results on Qwen3-VL-8B-Instruct. Best results are in bold.
<table><tr><td>Method</td><td>TextVQA ChartQA</td><td></td><td>DocVQA</td><td>GQA</td><td>Acc. (%)</td></tr><tr><td colspan="6">Vanilla, 100% Tokens</td></tr><tr><td>Qwen3-VL-8B-Instruct</td><td>82.98</td><td>83.16</td><td>95.75</td><td>61.88</td><td>100%</td></tr><tr><td colspan="6">Approximately 55% TFLOPs</td></tr><tr><td>PruneSID</td><td>73.46</td><td>63.56</td><td>90.51</td><td>60.08</td><td>89.15%</td></tr><tr><td>IVC-Prune</td><td>82.35</td><td>79.60</td><td>95.31</td><td>61.33</td><td>98.40%</td></tr><tr><td>iLLaVA</td><td>76.11</td><td>64.68</td><td>84.89</td><td>61.15</td><td>89.25%</td></tr><tr><td>ERASE</td><td>82.12</td><td>81.44</td><td>95.44</td><td>61.37</td><td>98.94%</td></tr><tr><td>SPIDER (Ours)</td><td>82.27</td><td>81.68</td><td>95.50</td><td>61.35</td><td>99.06%</td></tr><tr><td colspan="6">Approximately 45% TFLOPs</td></tr><tr><td>PruneSID</td><td>69.98</td><td>57.32</td><td>85.77</td><td>60.06</td><td>84.97%</td></tr><tr><td>IVC-Prune</td><td>81.74</td><td>76.48</td><td>94.97</td><td>61.11</td><td>97.11%</td></tr><tr><td>iLLaVA</td><td>72.57</td><td>61.72</td><td>79.63</td><td>60.91</td><td>85.81%</td></tr><tr><td>ERASE</td><td>81.77</td><td>79.96</td><td>95.00</td><td>61.31</td><td>98.25%</td></tr><tr><td>SPIDER (Ours)</td><td>81.82</td><td>80.03</td><td>95.16</td><td>61.33</td><td>98.33%</td></tr></table>

TABLE VI: Comparisons of training-aware modes. The TFLOPs are kept at a similar level around 30 % for fair comparison, where 1/4 of visual tokens are preserved for PruMerge+. Best results are in bold.
<table><tr><td>Method</td><td>POPE</td><td>TextVQA</td><td>MMB</td><td>Acc. (%)</td></tr><tr><td colspan="3">Upper Bound, All 576 Tokens (100%)</td></tr><tr><td>LLaVA-1.5-7B</td><td>85.9</td><td>58.2</td><td>64.1</td><td>100%</td></tr><tr><td colspan="3"></td></tr><tr><td>Approximately 30 % TFLOPs</td><td></td><td></td><td></td></tr><tr><td>+ PruMerge+ (Train)</td><td>84.0</td><td>57.1</td><td>64.9 99.0%</td></tr><tr><td>+ SPIDER (Train-Free) + SPIDER (Train)</td><td>84.8 85.0</td><td>57.4 63.9 57.6</td><td>99.0% 64.4 99.5%</td></tr></table>

![](images/2879402c1b4d9637417246a506514942c66dc506f58ea1d6eac4eef59ba16593.jpg)

![](images/71320429c269f8dc5218d569961e56b1b1170d7e97bccc54791b099ed6524f07.jpg)

![](images/9f304853d11a8cafa184394317aa2291044a52ed95e63911c42130b2a63327ba.jpg)  
Fig. 8: SPIDER efficiency on LLaVA-1.5-7B at varying token reduction ratios; key image regions relevant to queries marked by red dashed boxes.

TABLE VII: Ablation of our token pruning method, MSV-Prune, on LLaVA-NeXT-7B. Best results are in bold. ‘\*’ denotes our default setting.
<table><tr><td>Method</td><td>GQA</td><td>TextVQA POPE</td><td></td><td>Acc. (%)</td></tr><tr><td colspan="5">Upper Bound, All 2880 Tokens (100%)</td></tr><tr><td>LLaVA-NeXT-7B</td><td>62.9</td><td>59.6</td><td>86.3</td><td>100.0%</td></tr><tr><td colspan="5">Retain 1920 Tokens</td></tr><tr><td colspan="5">(a) Semantic Cluster</td></tr><tr><td>w/o Semantic Cluster Cluster Num =2 Cluster Num =8 Cluster Num =4*</td><td>61.3 61.9 61.6 62.6</td><td>59.3 59.4 59.3 59.7</td><td>87.2 87.9 87.7 88.2</td><td>99.33% 99.98% 99.68% 100.64%</td></tr><tr><td colspan="5">(b) Multi-Layer Tokens</td></tr><tr><td>w/o Middle Layer Tokens only Middle Layer Tokens w/ Middle Layer Tokens*</td><td>62.2 61.9 62.6</td><td>58.5 58.7 59.7</td><td>86.6 87.1 88.2</td><td>99.13% 99.28% 100.64%</td></tr><tr><td colspan="5">(c) Components of  $S _ { i j }$ </td></tr><tr><td colspan="5">w/o Similarity with Tanc 62.5 59.3 87.8 100.20%</td></tr><tr><td>w/o Inter-Layer Similarity</td><td>61.9 62.3</td><td>59.0 59.2</td><td>87.5 87.7</td><td>99.60%</td></tr><tr><td>cos → MSE cos → KL</td><td></td><td></td><td></td><td>99.99%</td></tr><tr><td></td><td>62.2</td><td>59.1</td><td>87.4</td><td>99.77%</td></tr><tr><td>All Equipped*</td><td>62.6</td><td>59.7</td><td>88.2</td><td>100.64%</td></tr></table>

TABLE VIII: Ablation of our sub-layer skipping method, ASL-Skip. No visual token is pruned. Best results are in bold.
<table><tr><td>Method</td><td>MMStar TextVQA</td><td>MME</td><td>Acc. (%)</td></tr><tr><td colspan="4">Upper Bound, All 576 Tokens (100%)</td></tr><tr><td>LLaVA-1.5-7B</td><td>33.7</td><td>58.2 1510.7</td><td>100%</td></tr><tr><td></td><td>(a) Components of Sfuse</td><td></td><td></td></tr><tr><td>w/o  $\mathbf { S } _ { s a }$  w/o SLC All Equipped*</td><td>33.5 32.7 33.7</td><td>57.6 1497.8 58.0 1483.6 58.1 1504.9</td><td>99.17% 98.30% 99.81%</td></tr><tr><td colspan="4">(b) Accumulation Mode of</td></tr><tr><td>w/o Accumulation w/ Accumulation*</td><td>33.4 33.7</td><td> $S _ { s k i p }$  57.8 1499.1 58.1 1504.9</td><td>99.22% 99.81%</td></tr><tr><td colspan="4">(c) Components of</td></tr><tr><td>w/o  $\mathbf { E } _ { i i }$ </td><td>33.6</td><td> $\mathbf { S } _ { s a }$  57.9 1501.0 57.7</td><td>99.52%</td></tr><tr><td>w/o  $\mathbf { F } _ { i t c }$   $\mathrm { ~ \bf ~ S ~ } _ { s a } \mathrm { ~ \bf ~ \ast ~ }$ </td><td>33.5 33.7</td><td>1498.5 58.1 1504.9</td><td>99.25% 99.81%</td></tr></table>

TABLE IX: Ablation of anchor token ratio in our token pruning method, MSV-Prune. Best results are in bold. ’\*’ denotes our default setting.
<table><tr><td>Anchor Ratio</td><td>GQA</td><td>TextVQA</td><td>POPE</td><td>Acc. (%)</td></tr><tr><td>LLaVA-1.5-7B</td><td>61.9</td><td>58.2</td><td>85.9</td><td>100.0%</td></tr><tr><td>0</td><td>59.3</td><td>54.7</td><td>83.7</td><td>95.74%</td></tr><tr><td>0.3</td><td>60.2</td><td>57.0</td><td>86.2</td><td>98.51%</td></tr><tr><td>0.5</td><td>60.6</td><td>57.5</td><td>86.4</td><td>99.09%</td></tr><tr><td>0.7*</td><td>60.9</td><td>57.9</td><td>86.6</td><td>99.56%</td></tr><tr><td>1.0</td><td>60.8</td><td>56.9</td><td>86.1</td><td>98.74%</td></tr></table>

Ablation on Anchor Token Ratio in MSV-Prune (Table IX). We perform a fine-grained ablation on the anchor token ratio to characterize its influence on model performance. The results reveal a clear non-monotonic trend: completely disabling anchor tokens (ratio = 0) incurs substantial drops in accuracy across all benchmarks (e.g., −2.6 on GQA, −3.5 on TextVQA), confirming that unstructured token removal disrupts critical visual grounding. In contrast, over-retaining anchors (ratio = 1.0) also degrades performance—particularly on TextVQA (56.9 vs. 57.9 at ratio = 0.7), suggesting that excessive token preservation hinders the model’s ability to exploit sparsity and introduces redundant computation without meaningful gains. The optimal trade-off is achieved at an anchor ratio of 0.7, which not only attains peak scores on GQA (60.9), TextVQA (57.9), and POPE (86.6), but also maintains 99.56% of the original model’s effective capacity. This indicates that a moderate yet principled selection of anchor tokens enables MSV-Prune to preserve cross-modal alignment while maximizing computational savings.

Ablation on ASL-Skip (Table VIII). When ablating $\mathbf { S } _ { f u s e }$ in (a), removing the $\mathbf { S } _ { s a }$ component forces all visual tokens to follow the offline SLC list, while omitting SLC makes skipped tokens bypass entire subsequent layers, leading to higher performance variance and reduced robustness. In (b) , the addition-based accumulation outperforms non-accumulative decisions, as tokens consistently deemed unimportant across consecutive layers are more reliably skip candidates than those judged by a single layer in isolation. In (c), both semantic uncertainty and image-text correlation are necessary for deciding if and when a token should start sub-layer skipping.

![](images/9d652728ad079d9b22848ebe32d097754908b1419f417b2a63e972dc807fa86d.jpg)

![](images/223683d43d85cf8ecc9ffc913d134a2305c7c5cd49a5ac3bde923f50721f786c.jpg)  
(a) Distribution of Skipped Visual Token Ratio  
(b) Distribution of Acc on POPE  
Fig. 9: Heatmap of token skipping proportion (a) and POPE accuracy (b) for different $T _ { s k i p }$ and $w _ { 1 } / w _ { 2 }$ settings on LLaVA-1.5-7B. No visual token is pruned.

Ablation on the Balance Between Token Skipping Ratio and Accuracy in ASL-Skip (Figure 9). We evaluate how the cumulative skip threshold $T _ { \mathrm { s k i p } }$ and weight ratio $w _ { 1 } / w _ { 2 }$ govern the balance between computational savings and output fidelity. Here, $T _ { \mathrm { s k i p } }$ sets the activation threshold for skipping (higher values suppress skipping), while $w _ { 1 } / w _ { 2 }$ balances the online skippability score $S _ { s a }$ (derived from token entropy $\mathbf { E } _ { i i }$ and image-text correlation $\mathbf { F } _ { i t c } ~ )$ and the offline normalized sub-layer redundancy score $\mathrm { S L C } ^ { \mathrm { N o r m } }$

Figure 9 (a) confirms an inverse relationship between $T _ { \mathrm { s k i p } }$ and skipping ratio: lowering $T _ { \mathrm { s k i p } } ( \mathbf { e . g . } , T _ { \mathrm { s k i p } } = 1 0 \ )$ drastically increases skipping rates, with the entire $T _ { \mathrm { s k i p } } ~ = ~ 1 0$ row skipping essentially all tokens (0.99–1.00). With w<sub>2</sub> fixed, the weight ratio sets the scale of $\mathrm { S } _ { f u s e }$ in Eq. (6), so a larger $w _ { 1 } / w _ { 2 }$ makes the accumulator cross $T _ { \mathrm { s k i p } }$ earlier: at $w _ { 1 } / w _ { 2 } = 1 0$ the skipped ratio stays at 0.99–1.00 for every $T _ { \mathrm { s k i p } }$ , whereas the small $w _ { 1 } / w _ { 2 }$ columns are the ones most sensitive to $T _ { \mathrm { s k i p } }$ . Conversely, raising $T _ { \mathrm { s k i p } } ~ ( { \bf e . g . } , T _ { \mathrm { s k i p } } = 3 0 \ )$ suppresses skipping for $w _ { 1 } / w _ { 2 } \leq 5$ , preserving nearly all tokens but yielding minimal acceleration, while the $w _ { 1 } / w _ { 2 } = 1 0$ column remains saturated.

POPE accuracy (Figure 9 (b)) directly reflects this trade-off: the most aggressive configurations lie in the $T _ { \mathrm { s k i p } } = 1 0 ~ \mathrm { r o w } ,$ whose accuracies are the lowest in the grid and bottom out at 84.00% at $w _ { 1 } / w _ { 2 } = 1 0$ , i.e. 1.9 points below the dense baseline, due to excessive skipping of discriminative features. Overly conservative settings ( $T _ { \mathrm { s k i p } } ~ = ~ 3 0 ~ )$ maintain high accuracy but sacrifice efficiency gains. Reading the two panels jointly identifies the operating point: nine configurations lie within 0.2 points of the dense LLaVA-1.5-7B POPE baseline of 85.9%, and among them $( \ T _ { \mathrm { s k i p } } = 2 0 \ , w _ { 1 } / w _ { 2 } = 3 \ )$ attains 85.70% with by far the highest skipped-token ratio, 0.41; the runner-up $( \ T _ { \mathrm { s k i p } } = 3 0 \ : , w _ { 1 } / w _ { 2 } = 5 $ ) reaches the same 85.70% but skips only 0.17. The four configurations that do reach 85.90% all have a skipped-token ratio of 0.00 in panel (a) and therefore accelerate nothing. We thus adopt $( \ T _ { \mathrm { s k i p } } = 2 0$ $w _ { 1 } / w _ { 2 } = 3 )$ , which costs 0.2 points on POPE while placing 41% of the visual tokens in skip mode. This configuration leverages $S _ { s a }$ ’s context-aware assessment to retain semantically critical tokens and $\mathrm { S L C } ^ { \mathrm { N o r m \ , } } \mathrm { s }$ precomputed redundancy profile to skip non-essential sub-layer computations.

![](images/6784f55eca4c454df2354ce140ae4a97f523404ac452792f7ff24b9e88b0834f.jpg)  
Fig. 10: Distribution of High-attention Tokens Across Vision Encoder Layers.

These results establish two principles: (1) $T _ { \mathrm { s k i p } }$ must be calibrated to avoid under-skipping (wasted efficiency) or overskipping (accuracy loss); (2) prioritizing the online score ( $w _ { 1 } / w _ { 2 } ~ > ~ 1 ~ )$ ensures skipping decisions adapt to inputspecific semantics, while $w _ { 1 } / w _ { 2 }$ must remain bounded, since at $w _ { 1 } / w _ { 2 } ~ = ~ 1 0$ the accumulator saturates and skipping becomes indiscriminate.

## G. Distribution of High-attention Visual Tokens

The visualization of top 50% high-attention tokens across all 24 layers of the CLIP vision encoder, as displayed in Figure 10, reveals a clear progression in how visual information is processed and represented hierarchically. This hierarchical transformation provides a principled basis for our middle and deep layer selection in the vision encoder: we select the $1 2 ^ { t h }$ layer (middle layer) and the $2 4 ^ { t h }$ layer (final layer) as key feature extraction points.

In the early layers (Layers 1-6), high-attention tokens are evenly distributed across the embedding space, with no strong clustering patterns. This reflects the vision encoder’s initial focus on low-level features, such as edges, textures, and simple shapes—captured at a fine-grained level. These representations are highly local and lack semantic coherence. While rich in detail, they contain too much noise and redundancy for direct use in language modeling, where global semantics and structured concepts are essential.

At Layer 12, we observe a transitional state: the highattention tokens begin to form distinct clusters while still maintaining some diversity. This indicates that the model has started combining low-level features into higher-level structures, such as object parts or spatial configurations, but without fully collapsing into coarse semantic categories. This makes it an ideal source for capturing the fine-grained objectcentric details necessary for more information-dense tasks.

TABLE X: Replaced layers for different MLLM series and parameter scales.
<table><tr><td>Model Series</td><td>Replaced Sub-Layers</td></tr><tr><td>LLaVA-1.5-7B</td><td>Attention: 25,27,28,30,23,26,22,24,21,0,20,3, 18,4,19,17,14,15,5,16,12,1,13,7,8,9,10;</td></tr><tr><td>LLaVA-NeXT-7B</td><td>FFN: 31,29,2,11,6</td></tr><tr><td></td><td>Attention: 28,29,27,30,23,24,25,22,21,26,20,19,</td></tr><tr><td></td><td>18,17,15,16,14,12,4,13,5,0,7,6,8; FFN: 31,1,2,3,11,9,10</td></tr></table>

By Layer 24, the final layer before the output, high-attention tokens exhibit strong clustering, indicating that the vision encoder has synthesized the input into coherent semantic units corresponding to major objects or scene components. The representation becomes highly abstract and globally consistent, emphasizing holistic understanding rather than fine details.

## H. Replaced Sub-Layers

For the selection of replaced layers, the SLC metric is computed based on a randomly sampled dataset comprising 150 cases, with 50 instances sampled equally from GQA, MMVet, and POPE. Layers are then replaced, either with VSkip-Attn or VSkip-FFN modules, in order of decreasing $\mathrm { S L C } ^ { N \bar { o } r m }$ of the designated sub-layer, i.e. the most skippable sub-layer first. Table X enumerates the layer IDs corresponding to the replaced components within the default SPIDER architecture.

## I. Sensitivity Analysis on Middle Layer Selection

To verify that MSV-Prune is robust to the choice of intermediate vision encoder layer, we conduct layer sweep experiments on two architectures: LLaVA-1.5-7B and Qwen3- VL-8B-Instruct, as shown in Figure 11. For LLaVA-1.5-7B (Figure 11(a)), we vary the middle layer index from 6 to 18 across the 24-layer CLIP ViT backbone under the 25-30% TFLOPs setting, and observe that both GQA and MMVet scores remain highly stable, fluctuating by less than 0.5 points throughout. For Qwen3-VL-8B-Instruct (Figure 11(b)), a similar sweep from layer 10 to 18 yields equally flat curves on TextVQA and ChartQA, with a variation of less than 0.3 points across all candidate layers. In both cases, performance peaks near the encoder midpoint (layer 12 for LLaVA-1.5- 7B and layer 14 for Qwen3-VL-8B-Instruct) and degrades only marginally at the extremes: layers that are too shallow lack sufficient semantic abstraction, while layers too close to the final output sacrifice the fine-grained object-centric details that motivate the use of intermediate features. These results confirm that MSV-Prune is not sensitive to the exact middle layer selection, and that adopting around ⌊L/2⌋ as the default intermediate layer ID provides a reliable and architectureagnostic choice that requires no dataset-specific tuning.

![](images/6bb0af164b57a3a21a7dfddebec8b5d4f61aaa5fe02cf798928e3296991c4b92.jpg)  
(a) LLaVA-1.5-7B (25-30% TFLOPs)

![](images/073e07ba57caafa6d59cf8f6fefca2618814d2057164520f18b324e13d9d453d.jpg)  
(b) Qwen3-VL-8B-Instruct (\~55% TFLOPs)  
Fig. 11: Sensitivity of MSV-Prune to the choice of intermediate vision encoder layer on (a) LLaVA-1.5-7B (25–30% TFLOPs) and (b) Qwen3-VL-8B-Instruct (∼55% TFLOPs). Performance remains stable across a wide range of layer choices on both architectures, confirming that the midpoint layer ⌊L/2⌋ is a robust and architecture-agnostic default.

![](images/447e7c73afdb73becf464e50988484c1663411ae4b941668007c350a075e4e5a.jpg)  
Fig. 12: Failure cases: red boxes highlight relevant areas; purple masks show skipped patch tokens during decoding.

## J. Stability of the offline SLC policy after token pruning.

Since ASL-Skip uses an offline SLC policy while inference is performed after MSV-Prune, we examine whether token pruning changes the SLC ranking substantially. Table XI shows that the ranking is largely preserved under the pruned token distribution: for both LLaVA-1.5-7B and LLaVA-NeXT-7B, the top-ranked attention and FFN sub-layers exhibit high overlap before and after pruning. Moreover, replacing the original dense-token SLC policy with a pruned-token SLC policy leads to only marginal differences in downstream accuracy. This suggests that the offline SLC captures a stable backbonelevel redundancy pattern that remains valid after MSV-Prune.

TABLE XI: Stability of the offline SLC policy under the pruned token distribution. We compare the layer-wise SLC ranking computed on dense visual tokens with that recomputed after MSV-Prune. We further report the average accuracy on GQA, MMVet, and POPE when ASL-Skip uses either the original dense-token SLC policy or the recomputed prunedtoken SLC policy.
<table><tr><td>Model</td><td>Retention</td><td>Attn Top-8</td><td>Attn Top-16</td><td>FFN Top-5</td><td>Dense-SLC Acc. (%)</td><td>Pruned-SLC Acc. (%)</td><td>Δ (%)</td></tr><tr><td>LLaVA-1.5-7B</td><td>385/576</td><td>7/8</td><td>14/16</td><td>5/5</td><td>99.61</td><td>99.62</td><td>+0.01</td></tr><tr><td>LLaVA-NeXT-7B</td><td>1920/2880</td><td>7/8</td><td>13/16</td><td>4/5</td><td>99.84</td><td>99.82</td><td>-0.02</td></tr></table>

## K. Failure Case Analysis

Figure 12 illustrates two representative failure cases of the SPIDER framework, revealing critical limitations in its token pruning and sublayer skipping mechanisms under challenging real-world conditions.

First, in the license plate recognition task, motion blur introduces severe degradation in image quality, particularly affecting the clarity of individual digits. As shown in the top row of Figure 12, although the input image contains the correct license plate number ”AED-632”, the retained tokens after SPIDER’s pruning process fail to preserve key character regions. Instead, the model retains ambiguous patches that resemble noise or artifacts, leading it to generate an incorrect answer: ”AH-9899999”. This demonstrates a fundamental issue: SPIDER’s attention-based token selection mechanism, while effective under clean conditions, lacks robustness against lowlevel image distortions. It tends to prioritize visually salient but semantically irrelevant regions over degraded yet contextually meaningful ones, resulting in hallucinated outputs. The model fails to maintain sufficient fidelity in regions critical for digit parsing, indicating a need for task-aware attention modulation that can adaptively protect high-stakes visual features during compression.

Second, in the chest X-ray diagnosis example, the failure arises from sub-layer skipping in clinically relevant regions. As shown in the bottom row of Figure 12, the reference answer is “Infiltration”, whereas SPIDER predicts “Pneumonia”. The skipped patch tokens, highlighted by the purple masks, overlap with the abnormal region marked by the red box, indicating that critical evidence may be insufficiently updated during decoding. This case suggests that, in medical imaging, indiscriminate compression can distort pathology attribution and lead to clinically misleading predictions. It also highlights the need for domain-aware compression strategies that incorporate anatomical priors or task-specific importance cues to better preserve diagnostically relevant regions.

To mitigate these issues, future work may explore adaptive token retention policies guided by task semantics. For example, integrating a lightweight semantic segmentation module or using anatomical priors in medical contexts could help preserve critical regions even under aggressive pruning.

## VI. CONCLUSION

We present SPIDER, a training-free framework that jointly performs multi-layer semantic visual token pruning and adaptive sub-layer skipping in MLLMs. By leveraging redundancy patterns across vision encoder layers and differentiating the roles of Attention and FFNs, SPIDER achieves substantial computational savings with minimal accuracy loss. Experiments across diverse MLLMs and benchmarks show large margins over earlier token-pruning and layer-skipping methods, and parity with the strongest recent baselines at matched compute.

## ACKNOWLEDGMENTS

This work was supported by the National Natural Science Foundation of China (No. 62121002, No. 62472396) and the Anhui Provincial Natural Science Foundation (2508085QF212).

## REFERENCES

[1] S. Bai, K. Chen, X. Liu, J. Wang, W. Ge, S. Song, K. Dang, P. Wang, S. Wang, J. Tang et al., “Qwen2. 5-vl technical report,” arXiv preprint arXiv:2502.13923, 2025.

[2] B. Lin, Y. Ye, B. Zhu, J. Cui, M. Ning, P. Jin, and L. Yuan, “Video-llava: Learning united visual representation by alignment before projection,” in Proceedings ofthe 2024 Conference on Empirical Methods in Natural Language Processing, 2024, pp. 5971–5984.

[3] S. Zhang, Q. Fang, Z. Yang, and Y. Feng, “Llava-mini: Efficient image and video large multimodal models with one vision token,” in The Thirteenth International Conference on Learning Representations.

[4] L. Chen, H. Zhao, T. Liu, S. Bai, J. Lin, C. Zhou, and B. Chang, “An image is worth 1/2 tokens after layer 2: Plug-and-play inference acceleration for large vision-language models,” in European Conference on Computer Vision. Springer, 2024, pp. 19–35.

[5] Y. Zhang, C.-K. Fan, J. Ma, W. Zheng, T. Huang, K. Cheng, D. A. Gudovskiy, T. Okuno, Y. Nakata, K. Keutzer et al., “Sparsevlm: Visual token sparsification for efficient vision-language model inference,” in Forty-second International Conference on Machine Learning.

[6] Q. Zhang, A. Cheng, M. Lu, R. Zhang, Z. Zhuo, J. Cao, S. Guo, Q. She, and S. Zhang, “Beyond text-visual attention: Exploiting visual cues for effective token pruning in vlms,” arXiv preprint arXiv:2412.01818, 2024.

[7] T. Lawson and L. Aitchison, “Learning to skip the middle layers of transformers,” arXiv preprint arXiv:2506.21103, 2025.

[8] R. Csordas, K. Irie, and J. Schmidhuber, “The neural data router: Adap-´ tive control flow in transformers improves systematic generalization,” in International Conference on Learning Representations.

[9] G. Comanici, E. Bieber, M. Schaekermann, I. Pasupat, N. Sachdeva, I. Dhillon, M. Blistein, O. Ram, D. Zhang, E. Rosen et al., “Gemini 2.5: Pushing the frontier with advanced reasoning, multimodality, long context, and next generation agentic capabilities,” arXiv preprint arXiv:2507.06261, 2025.

[10] H. Liu, C. Li, Y. Li, and Y. J. Lee, “Improved baselines with visual instruction tuning,” in Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, 2024, pp. 26 296–26 306.

[11] J. Li, D. Li, S. Savarese, and S. Hoi, “Blip-2: Bootstrapping languageimage pre-training with frozen image encoders and large language models,” in International conference on machine learning. PMLR, 2023, pp. 19 730–19 742.

[12] D. Zhu, J. Chen, X. Shen, X. Li, and M. Elhoseiny, “Minigpt-4: Enhancing vision-language understanding with advanced large language models,” arXiv preprint arXiv:2304.10592, 2023.

[13] A. Li, L. Xu, C. Ling, J. Zhang, and P. Wang, “Emoverse: Exploring multimodal large language models for sentiment and emotion understanding,” arXiv preprint arXiv:2412.08049, 2024.

[14] Y. Zhang, K. Yuan, H. Lu, Y. Yue, J. Chen, and K. Wu, “Medtvt-r1: A multimodal llm empowering medical reasoning and diagnosis,” arXiv preprint arXiv:2506.18512, 2025.

[15] Y. Zhang, Y. Liu, Z. Guo, Y. Zhang, X. Yang, X. Zhang, C. Chen, J. Song, B. Zheng, Y. Yao et al., “Llava-uhd v2: an mllm integrating high-resolution semantic pyramid via hierarchical window transformer,” arXiv preprint arXiv:2412.13871, 2024.

[16] M. Maaz, H. Rasheed, S. Khan, and F. S. Khan, “Video-chatgpt: Towards detailed video understanding via large vision and language models,” arXiv preprint arXiv:2306.05424, 2023.

[17] H. Zhang, X. Li, and L. Bing, “Video-llama: An instruction-tuned audio-visual language model for video understanding,” arXiv preprint arXiv:2306.02858, 2023.

[18] J. Chen, L. Ye, J. He, Z.-Y. Wang, D. Khashabi, and A. Yuille, “Efficient large multi-modal models via visual context compression,” Advances in Neural Information Processing Systems, vol. 37, pp. 73 986–74 007, 2024.

[19] Y. Li, C. Wang, and J. Jia, “Llama-vid: An image is worth 2 tokens in large language models,” in European Conference on Computer Vision. Springer, 2024, pp. 323–340.

[20] Y. Shang, M. Cai, B. Xu, Y. J. Lee, and Y. Yan, “Llava-prumerge: Adaptive token reduction for efficient large multimodal models,” arXiv preprint arXiv:2403.15388, 2024.

[21] L. Xing, Q. Huang, X. Dong, J. Lu, P. Zhang, Y. Zang, Y. Cao, C. He, J. Wang, F. Wu et al., “Pyramiddrop: Accelerating your large vision-language models via pyramid visual redundancy reduction,” arXiv preprint arXiv:2410.17247, 2024.

[22] S. Yang, Y. Chen, Z. Tian, C. Wang, J. Li, B. Yu, and J. Jia, “Visionzip: Longer is better but not necessary in vision language models,” in Proceedings of the Computer Vision and Pattern Recognition Conference, 2025, pp. 19 792–19 802.

[23] A. Wang, F. Sun, H. Chen, Z. Lin, J. Han, and G. Ding, “[cls] token tells everything needed for training-free efficient mllms,” arXiv preprint arXiv:2412.05819, 2024.

[24] J. Liu, F. Du, G. Zhu, N. Lian, J. Li, and B. Chen, “Hiprune: Trainingfree visual token pruning via hierarchical attention in vision-language models,” arXiv preprint arXiv:2508.00553, 2025.

[25] Q. Yuan, Q. Zhang, Y. Liu, J. Chen, Y. Lu, H. Lin, J. Zheng, X. Han, and L. Sun, “Shortv: Efficient multimodal large language models by freezing visual tokens in ineffective layers,” arXiv preprint arXiv:2504.00502, 2025.

[26] W. Zeng, Z. Huang, K. Ji, and Y. Yan, “Skip-vision: Efficient and scalable acceleration of vision-language models via adaptive token skipping,” arXiv preprint arXiv:2503.21817, 2025.

[27] W. Suo, J. Ma, M. Sun, L. Y. Wu, P. Wang, and Y. Zhang, “Pruning allrounder: Rethinking and improving inference efficiency for large vision language models,” arXiv preprint arXiv:2412.06458, 2024.

[28] M. Elhoushi, A. Shrivastava, D. Liskovich, B. Hosmer, B. Wasti, L. Lai, A. Mahmoud, B. Acun, S. Agarwal, A. Roman et al., “Layerskip: Enabling early exit inference and self-speculative decoding,” arXiv preprint arXiv:2404.16710, 2024.

[29] H. Liu, C. Li, Y. Li, B. Li, Y. Zhang, S. Shen, and Y. J. Lee, “Llavanext: Improved reasoning, ocr, and world knowledge,” 2024.

[30] D. A. Hudson and C. D. Manning, “Gqa: A new dataset for real-world visual reasoning and compositional question answering,” in Proceedings ofthe IEEE/CVF conference on computer vision and pattern recognition, 2019, pp. 6700–6709.

[31] Y. Goyal, T. Khot, D. Summers-Stay, D. Batra, and D. Parikh, “Making the v in vqa matter: Elevating the role of image understanding in visual question answering,” in Proceedings of the IEEE conference on computer vision and pattern recognition, 2017, pp. 6904–6913.

[32] Y. S. Y. Q. M. Zhang, X. L. J. Y. X. Zheng, K. L. X. S. Y. Wu, R. J. C. Fu, and P. Chen, “Mme: A comprehensive evaluation benchmark for multimodal large language models,” arXiv preprint arXiv:2306.13394, 2021.

[33] A. Singh, V. Natarajan, M. Shah, Y. Jiang, X. Chen, D. Batra, D. Parikh, and M. Rohrbach, “Towards vqa models that can read,” in Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, 2019, pp. 8317–8326.

[34] Y. Li, Y. Du, K. Zhou, J. Wang, W. X. Zhao, and J.-R. Wen, “Evaluating object hallucination in large vision-language models,” arXiv preprint arXiv:2305.10355, 2023.

[35] Y. Liu, H. Duan, Y. Zhang, B. Li, S. Zhang, W. Zhao, Y. Yuan, J. Wang, C. He, Z. Liu et al., “Mmbench: Is your multi-modal model an all-around player?” in European conference on computer vision. Springer, 2024, pp. 216–233.

[36] W. Yu, Z. Yang, L. Li, J. Wang, K. Lin, Z. Liu, X. Wang, and L. Wang, “Mm-vet: Evaluating large multimodal models for integrated capabilities,” arXiv preprint arXiv:2308.02490, 2023.

[37] L. Chen, J. Li, X. Dong, P. Zhang, Y. Zang, Z. Chen, H. Duan, J. Wang, Y. Qiao, D. Lin et al., “Are we on the right way for evaluating large vision-language models?” Advances in Neural Information Processing Systems, vol. 37, pp. 27 056–27 087, 2024.

[38] M. Mathew, D. Karatzas, and C. Jawahar, “Docvqa: A dataset for vqa on document images,” in Proceedings of the IEEE/CVF winter conference on applications of computer vision, 2021, pp. 2200–2209.

[39] Z. Lin, M. Lin, L. Lin, and R. Ji, “Boosting multimodal large language models with visual tokens withdrawal for rapid inference,” in Proceedings of the AAAI Conference on Artificial Intelligence, vol. 39, no. 5, 2025, pp. 5334–5342.

[40] Z. Fang, P. Lyu, C. Zhang, G. Lu, J. Yu, and W. Pei, “Prune redundancy, preserve essence: Vision token compression in vlms via synergistic importance-diversity,” arXiv preprint arXiv:2603.09480, 2026.

[41] Z. Sun, Y. Ma, G. Liu, Y. Chen, X. Tang, Y. Hu, and Y. Xu, “Ivcprune: Revealing the implicit visual coordinates in lvlms for vision token pruning,” arXiv preprint arXiv:2602.03060, 2026.

[42] L. Hu, L. Gao, F. Shang, L. Wan, and W. Feng, “illava: An image is worth fewer than 1/3 input tokens in large multimodal models,” arXiv preprint arXiv:2412.06263, 2024.

[43] Y. Lee, K. Min, and Y. Kim, “Erase: Eliminating redundant visual tokens via adaptive two-stage token pruning,” arXiv preprint arXiv:2605.09982, 2026.

[44] Y. Bengio, A. Courville, and P. Vincent, “Representation learning: A review and new perspectives,” IEEE transactions on pattern analysis and machine intelligence, vol. 35, no. 8, pp. 1798–1828, 2013.

![](images/369ee3a6224ac06df86f76a7a74b1e413c14d6513bad130c0f7e83de00b2534b.jpg)  
Tianxiang Chen received the B.S. degree in mathematics, physics and fundamental sciences from the University of Electronic Science and Technology of China in 2021, and the Ph.D. degree from the School of Cyber Space and Security, University of Science and Technology of China, in 2026. He is currently an algorithm expert at Alibaba Cloud and also a postdoctoral researcher in the joint industry postdoctoral program of Fudan University and Alibaba Cloud. He is a recipient of the Special Award of the President of the Chinese Academy of Sciences. His

research interests include Agentic RL, large language models, multi-model large language models and Medical AI.

Zhentao Tan received B.S. and Ph.D. degrees from the University of Science and Technology of China in 2017 and 2022, respectively. He is currently a postdoc. in the University of Science and Technology of China and Alibaba Group. His research interests include semantic segmentation, video object segmentation, image synthesis, vision transformers, lightweight models, and large language models.

![](images/5e0a55275e663a12d2deb296ae9e777fad17cc99f1e2a2db0b0a00f72531d86b.jpg)

![](images/c703c0d368d086398d0262975c5d69a704474f353c84a5110928206a22dd6e89.jpg)

![](images/57da0313f9ea94038ab08f2e3aeb52218b143c05e772cb267075e8249963c0f6.jpg)

![](images/8e954e2d348c0e5a5caec7793d40e2209b19ee80d125883e26533d159be2efff.jpg)  
several top-conference papers.

Yue Wu received a B.E. degree in electronic engineering and a Ph.D. degree in information and communication engineering from the University of Science and Technology of China (USTC), Hefei, China, in 2012 and 2017, respectively. He is currently a senior algorithm expert with the Alibaba Group, in Hangzhou, China. His research interests include multimedia, computer vision, machine learning, and data mining.

Xiaobing Tu has extensive experience in system and AI infrastructure at Samsung, Intel, and NVIDIA. From 2018 to 2021, he focused on model optimization at Alibaba Cloud’s Heterogeneous Computing Team. From 2021 to 2024, he led Kuaishou’s AI INFRA team, covering generative AI, model optimization, inference, training, and ML Ops. In 2024, he led SenseTime’s HPC and inference department for large model optimization and domestic computing platforms. He now leads the Wuying in Alibaba-Cloud LLM Team and holds over 20 patents with

Zi Ye received a Master’s degree in Applied Statistics from the University of Oxford, UK, in 2010 and a Ph.D. at Universiti Teknikal Malaysia Melaka in 2022. She was a research fellow at Trinity College Dublin, Ireland. Now she is an assistant professor at Maynooth University, Ireland. Her research interests involve Artificial Intelligence & Machine Learning.

Jinkui Ren graduated from the Special Class for the Gifted Young at Shanghai Jiao Tong University, majoring in Electronic Engineering and Communication. After obtaining his master’s degree, he worked successively at Huawei, Intel and Alibaba. He currently serves as Chief Architect of Alibaba’s Wuying and AgentBay product lines. He has extensive expertise in operating systems, virtualization, large language models and related technologies, and holds more than 60 patents.

![](images/74df99a5771f9e13c962e1159e3465d0b36965516324e64581a8768e6b1436e6.jpg)

Xiantao Zhang holds a Ph.D. & M.S. in Information Security from Wuhan University. He is President of Alibaba Cloud Wuying Division. Previously at Intel, he led Xen/KVM development and won Intel’s Highest Achievement Award. Since 2014 at Alibaba Cloud, he built the Shenlong Architecture (World Leading Scientific and Technological Achievement Award) and holds 30+ patents. He now drives Wuying’s AI-native strategy, launching AgentBay (China’s first MCP-native cloud service for AI Agents) and AgenticComputer, redefining

![](images/1430c6f35cfb83d5494e2c5c44b74ad7753ae188943566f0fde4a87f1378df32.jpg)  
cloud computing for AI-driven automation.

![](images/91dbc084c41afa77c17d7d3fb0112a9280259abb9ea1ac57eef6497b71b4f90c.jpg)

Tao Gong received the B.E. degree in electronic engineering and the Ph.D. degree in cyber science and technology from University of Science and Technology of China, Hefei, China, in 2016 and 2021, respectively. He is currently an associate research fellow at the University of Science and Technology of China. His research interests include vision and language, object perception, and AIgenerated content detection.

![](images/0da8ba89a3b6a431238ca1ac2db31ac8cbe69b410582d37fe5c5aa32b03102a7.jpg)

Qi Chu received a B.S. degree in electronic engineering and a Ph.D. degree in information and communication engineering from University of Science and Technology of China in 2014 and 2019, respectively. Currently, he is an associate research fellow at the University of Science and Technology of China. His research interests include object detection, tracking, image synthesis, and adversarial examples.

![](images/7fc3d43b71063e26e28c3b2752e0a445d4a9b1f69e463494f16be784a631c862.jpg)

Nenghai Yu is a full Professor at the University of Science and Technology of China. He is also the director of the Information Processing Center of USTC and deputy director of the academic committee of School of Information Science and Technology. He received the Ph.D. degree from USTC in 2004. He was a visiting scholar at the Institute of Production Technology, Faculty of Engineering, University of Tokyo, in 1999 and performed cooperative research as a senior visiting scholar in the Dept. of Electrical Engineering, Columbia University, from Apr. to Oct.

2008. His research focuses on image processing and video analysis, multimedia communication, media content security, internet information retrieval, data mining and content filtering, network communication, and security.

![](images/1c404352a85058fdcaf5f6a43900c204c68ce50945f520d5d077efef7a801453.jpg)

Xipeng Qiu received the B.Sc. degree and Ph.D. degrees in computer science from Fudan University, China in 2001 and 2006, respectively. Currently, he is a professor in the School of Computer Science, Fudan University, China. His research interests include natural language processing and deep learning. He has published more than 60 papers in leading international journals and conferences, including TACL, ACL, EMNLP, AAAI, and IJCAI. He received the ACL 2017 Distinguished Paper Award and the CCL 2019 Best Paper Award. He is

the author of FudanNLP, an open-source Chinese natural language processing toolkit, and the lead of the FastNLP project. He was selected for the Young Elite Scientists Sponsorship Program of the China Association for Science and Technology in 2015, received the First Prize of the Qian Weichang Award for Young Scholars in Chinese Information Processing in 2018, and was named an AI 2000 Most Influential Scholar Honorable Mention in 2020.

![](images/8cbe34240759c99a91f264ff674f48678d5e19a6c7d13ad3eb187daa61ba7e14.jpg)

Jieping Ye received a Ph.D. degree in computer science from the University of Minnesota, Twin Cities, Minnesota, in 2005. He is currently a VP of Alibaba. He is also a professor at the University of Michigan, Ann Arbor, Michigan. His research interests include Big Data, machine learning, and data mining with applications in transportation and biomedicine. He won the NSF CAREER Award, in 2010 and the 2019 Daniel H. Wagner Prize for Excellence in the Practice of Advanced Analytics and Operations. His papers have been selected for the Outstanding Student Paper at ICML, in 2004, the KDD Best Research Paper Runner Up, in 2013, and the KDD Best Student Paper Award, in 2014. He has served as a senior program committee/area chair/program committee vice chair of many conferences, including NIPS, ICML, KDD, IJCAI, ICDM, and SDM. He has served as an associate editor for the Data Mining and Knowledge Discovery and the for IEEE Transactions on Pattern Analysis and Machine Intelligence.
# REPRESENTATION DYNAMICS REVEAL SEMANTIC SALIENCY AND SIMILARITY FOR VISUAL TOKEN PRUNING IN MLLMS

Weixuan Li<sup>1</sup> Zikun Zhou<sup>1</sup> Xinyi Zhuang<sup>1</sup> Xinyan Guo<sup>1,2</sup> Rui Tian<sup>1,2</sup> Chuyao Zhang<sup>1</sup> Lin Gao<sup>1</sup>

<sup>1</sup>Harbin Institute of Technology <sup>2</sup>Shenzhen Loop Area Institute lwx20030918@gmail.com

## ABSTRACT

Multimodal large language models (MLLMs) incur high inference latency from long visual token sequences. Existing pruning methods commonly use attention maps or output features to estimate token importance or redundancy. Several recent approaches also exploit representation changes, but when and how these changes reflect foreground saliency and semantic consistency remain insufficiently understood. We analyze visual token representation dynamics across encoder depth and uncover two findings. First, the relationship between token update magnitudes and foreground saliency is layer-dependent: large token updates concentrate on foreground regions in two depth intervals, separated by several sink-dominated layers at intermediate depths. Second, similarities between token update directions better distinguish same-class from different-class tokens than those between encoder output features. Building on these findings, we propose MSDG-Prune, a training-free method that uses update magnitudes and directions to preserve salient and diverse visual information. Specifically, we group tokens by update-direction similarity and use query-weighted saliency derived from update magnitudes across a chosen depth window for group-wise token pruning. Extensive experiments across four MLLMs demonstrate the effectiveness and generalizability of MSDG-Prune. On LLaVA-NeXT, it retains 91.9% of uncompressed performance on average with only 5.6% of visual tokens, while achieving a 7.8× prefilling speedup. Code is available at https: //github.com/liweixuan-hitsz/MSDG-Prune.

## 1 INTRODUCTION

Multimodal Large Language Models (MLLMs) have demonstrated strong capabilities in visual understanding and reasoning (Achiam et al., 2023; Li et al., 2025). These models connect vision encoders to pretrained LLMs and represent each image with hundreds to thousands of visual tokens (Liu et al., 2023; Bai et al., 2025). Such long sequences increase prefilling latency and KVcache memory usage, yet contain substantial redundancy. Prior work shows that many visual tokens can be removed with little performance degradation (Bolya et al., 2022; Shang et al., 2025), making pruning a practical approach to efficient inference. The key challenge is to remove redundant tokens while preserving visual evidence needed to answer the query.

Two widely studied approaches to visual token pruning are attention-based and diversity-based methods. Attention-based methods (Chen et al., 2024; Yang et al., 2025; Liu et al., 2026) mainly derive visual saliency from attention maps inside the vision encoder or LLM. However, final-layer en coder attention can overemphasize high-norm outlier tokens with limited information (Darcet et al., 2024; Fan et al., 2026; Takezoe et al., 2026), while text-to-vision attention in LLMs can exhibit positional bias (Endo et al., 2025). These limitations make attention scores less reliable for token pruning. Diversity-based methods estimate token redundancy from cosine similarity between token representations, using high feature similarity as evidence of overlapping visual information (Alvar et al., 2025; Wen et al., 2025). Nevertheless, cosine similarity between output features of the vision encoder does not align well with semantic consistency, evidenced by the overlap between the distributions in Figure 1(c). Figure 1(a) illustrates an example: only a few retrieved tokens (blue) belong to the same class as the anchor token (red). Hence, using this similarity to infer redundancy may lead to the removal of semantically distinct visual tokens needed to answer the query.

![](images/a3a68c8f6fffe35eb7043df2c0d16eadd517c079cddf0ee6afe5069cf92767b1.jpg)  
Top 40 | Hit rate ≈ 43%  
(a) Token retrieval: OF

![](images/b03bed2a89104e1ae650e52e53847b450620d1aa982b84b1b6903c30270a640f.jpg)  
Top 40 | Hit rate ≈ 100%  
(b) Token retrieval: UD

![](images/c121531562b1a940314d0bd46f791d8fbd5cc40fa2d0635970406ca7d0e0451c.jpg)  
(c) Similarity distribution: OF

![](images/a23ac71b4b3831ee49fbd5316e7ec9b399d40622f61df33be9c7f66dc48c68d3.jpg)  
(d) Similarity distribution: UD  
Figure 1: Output features (OF) versus update directions (UD) for semantic similarity in LLaVA-1.5. Panels (a) and (b) visualize the top 40 tokens ranked by cosine similarity to the airplane anchor token (red), using OF and UD, respectively. Panels (c) and (d) show cosine similarity distributions on COCO 2017 computed using OF and UD, respectively. For each anchor token, we compute cosine similarities to other tokens and group the resulting pairs by whether they share the same semantic class as the anchor token. Blue and gray curves represent same-class and differentclass pairs, respectively. The horizontal axis indicates cosine similarity, and the vertical axis shows the percentage of pairs within each similarity bin. Compared with OF, UD shows less overlap between the two groups, indicating better semantic discrimination.

Beyond the static attention maps and token features, could the way token representations change across layers provide more reliable evidence for pruning? Recent studies have explored representation changes as pruning cues (Choi et al., 2025; Li et al., 2026). However, understanding when these changes reflect semantic saliency and similarity warrants closer examination. To this end, we analyze the magnitude and direction of visual token updates within the encoder. Our analysis reveals a stage-dependent relationship between update magnitude and foreground saliency, as shown in Figure 2. Foreground enhancement occurs in both early and late stages, interrupted by large updates concentrated on isolated sink tokens, while background responses become more prominent near the encoder output. Thus, the usefulness of update magnitude as a cue for foreground saliency depends on encoder depth. This finding motivates estimating visual saliency from representation changes within a selected foreground-enhanced window.

Beyond update magnitude, we examine whether changes in representation direction reveal semantic relationships among tokens, given the mismatch between output-feature similarity and semantic consistency. We find that similarity between update directions better distinguishes same-class from different-class token pairs than output-feature similarity, as shown in Figure 1(c) and (d). This finding motivates using update-direction similarity to form semantically coherent groups, providing a basis for preserving distinct visual information during pruning. Both findings are further supported by observations across multiple MLLMs (Appendices B.1 and B.2).

Motivated by these findings, we propose MSDG-Prune, a training-free visual token pruning method that leverages update Magnitudes for Saliency estimation and update Directions for token Grouping. Specifically, we estimate saliency within a selected foreground-enhanced window and weight it by query relevance. These query-weighted saliency scores guide budget allocation across the semantic groups and token selection within each group. This design preserves visual diversity while prioritizing salient, query-relevant tokens. Our method prunes visual tokens before LLM prefilling without extracting attention maps, reducing prefilling computation and KV-cache memory while remaining compatible with FlashAttention (Dao et al., 2022; Dao, 2024). Extensive evaluations across four MLLMs show that MSDG-Prune achieves favorable performance against state-of-the-art methods. At 11.1% token retention, it outperforms PruneSID (Fang et al., 2026) by 2.1 and 1.5 percentage points on Mini-Gemini and Qwen2.5-VL, respectively. On LLaVA-NeXT, it retains 91.9% relative performance with only 5.6% of visual tokens and achieves a 7.8× prefilling speedup over the uncompressed model. Our contributions are:

• We analyze visual token representation dynamics, revealing a depth-dependent relationship between update magnitude and foreground saliency and better semantic discrimination from update directions than output features.

![](images/f97fd0ace909c5064ddf782c462087fe83079dbe723af8ff5310b1dd02e3d4bb.jpg)  
Figure 2: Layer-wise token update magnitudes in the vision encoder of LLaVA-1.5. Each map overlays patch-token update magnitudes across a Transformer block on the input image. L00-01 denotes the change from token states entering Layer 0 to those entering Layer 1. Because LLaVA-1.5 uses the penultimate-layer output of its CLIP ViT-L/14@336 (Radford et al., 2021; Liu et al., 2024a) vision encoder as input to the visual projector, the visualization ends at L22-23 and omits L23-24. The maps illustrate depth-dependent spatial patterns in token updates, including foreground enhancement before and after the sink-dominated stage.

• Building on these findings, we propose MSDG-Prune, a training-free visual token pruning method that uses update magnitudes and directions to more effectively preserve diverse and salient visual information.

• Extensive quantitative and qualitative evaluations across four MLLMs demonstrate the effectiveness and generalizability of MSDG-Prune. On LLaVA-1.5, it retains 95.4% of uncompressed performance on average at 11.1% token retention.

## 2 RELATED WORK

We review visual token pruning methods according to their main pruning cues, including attention, representation similarity, and changes in token representations.

Attention-based methods use attention scores to guide token selection. FastV (Chen et al., 2024) and SparseVLM (Zhang et al., 2025c) use attention within the LLM to identify visual tokens for pruning. VisionZip (Yang et al., 2025) and LLaVA-PruMerge (Shang et al., 2025) combine token selection based on vision encoder attention with similarity-based merging. HiPrune (Liu et al., 2026) uses attention patterns at different encoder depths to retain complementary token types.

Similarity-based methods use relationships between token representations to reduce redundancy. ToMe (Bolya et al., 2022) merges similar tokens within the encoder, while DivPrune (Alvar et al., 2025) selects tokens by maximizing feature diversity. DART (Wen et al., 2025) uses similarity to visual and textual pivot tokens to identify redundant visual tokens within the LLM. CDPruner (Zhang et al., 2025b) combines pairwise feature similarity with instruction relevance in a determinantal point process kernel. PruneSID (Fang et al., 2026) forms groups through principal semantic component analysis, suppresses redundant tokens within each group, and allocates retention quotas according to the remaining group sizes.

Recent methods also use representation changes as pruning cues. Representation Shift (Choi et al., 2025) scores tokens from differences between the inputs and outputs of network operations, without extracting attention maps. TransPrune (Li et al., 2026) combines module-level magnitude and angular changes with instruction-guided attention for progressive pruning within the LLM. EvoCut (Lu et al., 2026) clusters update directions between adjacent encoder layers and accumulates deviations from cluster centers to rank tokens globally. These methods primarily use representation changes to construct token-level importance scores.

In MSDG-Prune, visual saliency is estimated from the update magnitude over a selected window in the late foreground-enhanced stage. Semantic groups are formed using directions of change between normalized early and late encoder representations. These groups serve as units for budget allocation and intra-group selection, both guided by visual saliency and query relevance computed from embedding similarity. The resulting selection is performed before LLM prefilling without extracting attention maps.

## 3 REPRESENTATION DYNAMICS OF VISUAL TOKENS

In this section, we analyze representation dynamics within the vision encoder to inform the design of visual token pruning methods. Specifically, we examine the magnitude and direction of visual token updates across encoder depth. Herein, we consider a vision encoder with L layers, indexed from 0 to $L - 1$ . Each layer ℓ updates the representation of visual token i from $h _ { i } ^ { \ell }$ to $h _ { i } ^ { \ell + 1 }$ , where $h _ { i } ^ { \ell } \in \mathbb { R } ^ { d _ { v } }$ and $d _ { v }$ is the hidden dimension of the encoder.

## 3.1 ANALYSIS OF TOKEN UPDATE MAGNITUDES

To gain insight into how visual token representations evolve across encoder depth, we begin with a qualitative examination of their update magnitudes. For each layer $\ell ,$ we map the update magnitudes $v _ { i } ^ { \ell } = \| h _ { i } ^ { \ell + 1 } - h _ { i } ^ { \ell } \|$ to the corresponding image regions, as illustrated in Figure 2. The spatial distribution of token update magnitudes exhibits distinct stage-wise patterns across encoder depth, which we broadly characterize as five successive stages:

Initialization stage. In the first few layers, token updates appear to primarily reflect low-level visual properties, such as local smooth or textured regions.

![](images/461b89ac9bcd252047ecb22ce9ac950550406b456faa0bbf3284c1fdbb5f202b.jpg)  
Figure 3: Layer-wise analysis of token update magnitudes in LLaVA-1.5 on COCO 2017 validation images. Top: foreground recall of the top 30% of tokens ranked by update magnitude, with random selection as the dashed baseline. Bottom: the max-to-median update magnitude ratio, max<sub>i</sub> $v _ { i } ^ { \ell } / \ l$ median $v _ { i } ^ { \ell }$ , with a dashed diagnostic threshold of 10. Both metrics are computed per image and then averaged across images. The highlighted Layers 11–12 exhibit sharp ratio peaks, marking the sink-dominated stage in the corresponding vision encoder.

Early foreground-enhanced stage. Relatively large updates become more apparent over foreground regions in the early encoder layers. This tendency emerges gradually, with no sharp boundary separating this stage from initialization.

Sink-dominated stage. At intermediate depths, large updates concentrate on a few spatially isolated sink tokens, which are high-norm patch tokens and often located in background regions. Prior work suggests that sink tokens are used for internal computation and storing global information (Darcet et al., 2024), but carry little image-specific semantic information in CLIP (Fan et al., 2026). Strong sink responses do not necessarily indicate visual saliency.

Late foreground-enhanced stage. After the sink-dominated stage, relatively large updates again extend over foreground regions. This spatially extended pattern can be observed across several layers, unlike the large updates concentrated on a few isolated sink tokens.

Background-enhanced stage. Near the encoder output, large updates increasingly occur in background regions. This shift suggests that deeper updates are not necessarily better saliency cues.

Building on these visualizations, we quantitatively examine the depth-dependent patterns on COCO 2017 (Lin et al., 2014). For each image and layer, we select the top 30% of tokens by update magnitudes and measure foreground recall against COCO instance annotations. This metric measures how much annotated foreground area is covered by the selected tokens. We also compute the concentra tion ratio max $v _ { i } ^ { \ell } /$ median $v _ { i } ^ { \ell }$ to quantify how strongly the largest update exceeds the typical token update. Both statistics are averaged across images and plotted against encoder depth in Figure 3.

The concentration ratio exhibits pronounced peaks at the middle layers, consistent with the sinkdominated patterns across all examined models. These peaks are not accompanied by increased foreground recall, indicating that exceptionally large updates do not necessarily correspond to foreground regions. In contrast, foreground recall is elevated in both early and late foreground-enhanced stages before declining near the encoder output. This depth dependence motivates using update magnitudes from an early or late foreground-enhanced stage as a training-free cue for foreground saliency.

## 3.2 ANALYSIS OF TOKEN UPDATE DIRECTIONS

We next turn to changes in representation direction across encoder depth and investigate whether tokens undergoing similar direction changes are semantically related. We compare the representation of each token at an early reference state after the initialization stage, $h _ { i } ^ { \ell _ { 0 } }$ , with its late encoder state, $h _ { i } ^ { \ell _ { d } }$ , where $\ell _ { 0 } = 2$ and $\ell _ { d } = L - 1$ . To examine direction changes independently of the norms of the early and late representations, we normalize both representations to unit length. We define the update direction as the unit direction of the displacement between these normalized states:

$$
\Delta h _ { i } = \frac { h _ { i } ^ { \ell _ { d } } } { \vert \vert h _ { i } ^ { \ell _ { d } } \vert \vert } - \frac { h _ { i } ^ { \ell _ { 0 } } } { \vert \vert h _ { i } ^ { \ell _ { 0 } } \vert \vert } , \qquad d _ { i } = \frac { \Delta h _ { i } } { \vert \vert \Delta h _ { i } \vert \vert } .\tag{1}
$$

We use cosine similarity $d _ { i } ^ { \top } d _ { j }$ to measure the alignment between token update directions and compare it with the commonly used output-feature similarity. We first qualitatively compare the spatial patterns of the two similarity measures on LLaVA-1.5, as shown in Figure 1(a) and (b). For the airplane anchor token (marked in red), output-feature similarity assigns high scores to tokens both on and outside the airplane, whereas update-direction similarity better distinguishes airplane tokens from surrounding tokens.

To examine whether this observation holds across images, we analyze token-pair similarities on COCO 2017 using COCO-Stuff semantic annotations (Caesar et al., 2018). For each anchor token, we compute cosine similarities to other labeled tokens in the same image and separate the pairs according to whether they share the semantic class with the anchor. We aggregate these scores across anchors to obtain same-class and different-class similarity distributions for output features and update directions. As shown in Figure 1(c) and (d), update-direction similarity yields less overlap between the two distributions, indicating better semantic discrimination. This finding supports update-direction similarity as a cue for semantically coherent token grouping.

Additional visualizations and quantitative results across models further support our findings on update magnitude and direction (Appendices B.1 and B.2).

## 4 MSDG-PRUNE

## 4.1 PROBLEM FORMULATION AND METHOD OVERVIEW

Problem formulation. The input image I of the MLLM is generally encoded and projected into a sequence of visual tokens $\pmb { X } \overset { \bullet } { = } [ \pmb { x } _ { 1 } , \dots , \pmb { x } _ { N } ] \in \mathbb { R } ^ { N \times D }$ , where $N$ is the number of visual tokens before pruning and $D$ is the LLM embedding dimension. Given a retention budget $B \ll N$ , the goal of visual token pruning is to select an index set ${ \mathcal { S } } \subseteq \{ 1 , \ldots , N \}$ with $| { \boldsymbol { S } } | = B$ before LLM prefilling while preserving important and diverse visual information. The retained tokens form $\widetilde { X }$ in their original order and are passed to the pretrained LLM together with the tokenized query.

Method overview. Figure 4 summarizes the training-free pruning pipeline of MSDG-Prune. We first remove sink tokens using a threshold on a designated encoder coordinate (Appendix C.1) and denote the remaining set by V. For each candidate token in V, MSDG-Prune leverages the query-weighted saliency to assess its importance for the MLLM to respond to the query. Furthermore, MSDG-Prune groups tokens by update-direction similarity and performs pruning within each group to preserve diversity among the retained tokens. Next, we detail these two core methodology designs.

![](images/251a61370db74de7c36e2b283e4eb9c2427a039376a8de50e345a157c2e2a4b2.jpg)  
Figure 4: Overview of MSDG-Prune. After sink filtering, update magnitudes over a selected foreground-enhanced window provide visual saliency scores, while update-direction similarity organizes tokens into semantic groups. Visual saliency and query relevance jointly guide budget allocation across groups and token selection within each group.

## 4.2 QUERY-WEIGHTED SALIENCY ESTIMATION

We construct a query-weighted saliency score to assess candidate tokens by combining their visual saliency with relevance to the textual query. Our analysis in Section 3.1 motivates estimating visual saliency from representation changes within the foreground-enhanced stage. For each candidate token $i \in \nu$ , we define $u _ { i } = \| h _ { i } ^ { \ell _ { e } } - h _ { i } ^ { \ell _ { s } } \|$ , where $\ell _ { s }$ and $\ell _ { e }$ delimit the selected layer window.

We leave a one-layer buffer after the sink-dominated stage and use a window of five consecutive layer updates as an empirical default. Ablations on window placement and width support this choice (Appendices C.7 and C.8).

The resulting saliency score is independent of the textual query. To account for the relevance of each visual token to the query, we weight its visual saliency score by the maximum cosine similarity between its projected embedding and the query token embeddings. Let Q denote the set of query token indices, and let $t _ { j }$ denote the input embedding of query token j. For each projected visual token ${ \mathbf { x } } _ { i } ,$ we compute its query relevance score as

$$
\alpha _ { i } = \operatorname* { m a x } _ { j \in \mathcal { Q } } \frac { { \pmb x } _ { i } ^ { \top } { \pmb t } _ { j } } { \| { \pmb x } _ { i } \| \| \pmb { t } _ { j } \| } .\tag{2}
$$

The query-weighted saliency score is then defined as $s _ { i } = \alpha _ { i } u _ { i }$

## 4.3 DIRECTION-INFORMED GROUP-WISE PRUNING

To preserve diverse visual information, we adopt the group-wise pruning mechanism, where the tokens are grouped by update-direction similarity. We then use the above query-weighted saliency to guide budget allocation across groups and select important tokens within each group.

Specifically, we partition V into K groups using spherical k-means (Hornik et al., 2012) on the update directions $d _ { i }$ defined in Equation 1. The grouping objective maximizes the sum of cosine similarities between token update directions and their assigned unit centroids:

$$
\operatorname* { m a x } _ { \{ { \mathcal C } _ { c } \} , \{ \mu _ { c } \} } \sum _ { c = 1 } ^ { K } \sum _ { i \in { \mathcal C } _ { c } } d _ { i } ^ { \top } \mu _ { c } \quad \mathrm { s . t . ~ } \| \mu _ { c } \| = 1 , \quad c = 1 , \ldots , K ,\tag{3}
$$

where $\{ \mathcal { C } _ { c } \} _ { c = 1 } ^ { K }$ partitions V into K nonempty groups, and $\mu _ { c }$ is the unit centroid of group $\mathcal { C } _ { c }$ . We initialize the centroids using deterministic farthest-point sampling and stop the iterations when a convergence criterion is met or a preset iteration limit is reached. For each group, we compute its mean query-weighted saliency score $g _ { c }$ and apply softmax across groups to obtain the target budget proportion $p _ { c } \mathbf { : }$

$$
g _ { c } = \frac { 1 } { | \mathcal { C } _ { c } | } \sum _ { i \in \mathcal { C } _ { c } } s _ { i } , \qquad p _ { c } = \frac { \exp ( g _ { c } ) } { \sum _ { r = 1 } ^ { K } \exp ( g _ { r } ) } .\tag{4}
$$

For $B \leq | \nu |$ , we allocate integer budgets $b _ { c }$ to groups according to $p _ { c } ,$ subject to $0 \leq b _ { c } \leq | \mathcal { C } _ { c } |$ and $\textstyle \sum _ { c = 1 } ^ { K } b _ { c } = B$ . Within each group ${ \mathcal { C } } _ { c } ,$ , we retain the $b _ { c }$ tokens with the highest query-weighted saliency scores $s _ { i }$ and combine their indices to form the retained set S. Finally, we arrange the selected tokens in their original order to form $\widetilde { X }$ and pass them to the LLM with the tokenized query. Algorithm 1 summarizes the complete token selection procedure.

## 5 EXPERIMENTS

Following the experimental protocol of PruneSID (Fang et al., 2026), we evaluate MSDG-Prune on LLaVA-1.5 (Liu et al., 2024a), LLaVA-NeXT (Liu et al., 2024b), Mini-Gemini (Li et al., 2025) and Qwen2.5-VL (Bai et al., 2025). These models respectively cover fixed-resolution encoding, image tiling, a dual-encoder architecture, and native dynamic resolution, allowing us to examine the effectiveness of representation dynamics across different visual processing pipelines.

We compare our method with FastV (Chen et al., 2024), SparseVLM (Zhang et al., 2025c), DART (Wen et al., 2025), DivPrune (Alvar et al., 2025), VisionZip (Yang et al., 2025), HiPrune (Liu et al., 2026), EvoCut (Lu et al., 2026), and PruneSID (Fang et al., 2026). Evaluation is conducted using LMMs-Eval (Zhang et al., 2025a) on benchmarks covering visual question answering, visual perception and reasoning, object hallucination assessment, text understanding, and high-resolution image understanding. RelAcc. is the mean of per-benchmark scores normalized by the corresponding uncompressed scores, expressed as a percentage, with missing entries excluded. Implementation settings and benchmark descriptions are provided in Appendices C.2 and D, respectively.

Table 1: Performance comparison on LLaVA-1.5-7B. Vanilla refers to the uncompressed baseline. The subscript after each method gives the publication venue.
<table><tr><td>Method GQA MMB MME POPE SQA</td></tr><tr><td>Uncompressed baseline (100%)</td><td></td><td></td><td></td><td></td><td> $\mathrm { \Delta V Q A } ^ { \mathrm { v 2 } }$ </td><td>VQAText</td><td>SEED</td><td>VizWiz</td><td>RelAcc. 100%</td></tr><tr><td>VanillaCVPR 2024 61.9 64.7 1862 85.9</td></tr><tr><td>Token retention: 33.3%</td><td></td><td></td><td></td><td>69.5</td><td>78.5</td><td>58.2</td><td>60.5</td><td>54.3</td></tr><tr><td>FastVECCV 2024</td><td>52.7</td><td>61.2</td><td>1612</td><td>64.8 67.3</td><td>67.1</td><td>52.5</td><td>57.1</td><td>50.8 89.1%</td></tr><tr><td>SparseVLMICML 2025</td><td>57.6</td><td>62.5</td><td>1721 83.6</td><td>69.1</td><td>75.6</td><td>56.1</td><td>55.8 50.5</td><td>95.2%</td></tr><tr><td>DARTEMNLP 2025</td><td>60.0</td><td>63.6</td><td>1856 82.8</td><td>69.8</td><td>76.7</td><td>57.4</td><td>51.5 54.9</td><td>97.1%</td></tr><tr><td>DivPrunecvPR 2025</td><td>60.0</td><td>62.3 1752</td><td>87.0</td><td>68.7</td><td>75.5</td><td>56.4</td><td>58.6 55.6</td><td>97.8%</td></tr><tr><td>VisionZipcVPR 2025</td><td>59.3</td><td>63.0 1783</td><td>85.3</td><td>68.9</td><td>76.8</td><td>57.3</td><td>58.5 54.1</td><td>97.8%</td></tr><tr><td>HiPruneACL 2026</td><td>59.2</td><td>62.8</td><td>1814 86.1</td><td>68.9</td><td>76.7</td><td>57.6</td><td>54.5</td><td>98.3%</td></tr><tr><td>EvoCutarXiv 2026.06</td><td>60.2</td><td>64.2</td><td>1794 86.5</td><td>68.5</td><td>76.7</td><td>57.6</td><td>57.3</td><td>97.9%</td></tr><tr><td>PruneSIDICLR 2026</td><td>60.1</td><td>63.7</td><td>1791 86.9</td><td>68.5</td><td>76.8</td><td>56.7</td><td>59.0 55.4</td><td>98.5%</td></tr><tr><td>Ours</td><td>60.1</td><td>63.6</td><td>1806 86.9</td><td>68.7</td><td>76.1</td><td>57.4</td><td>58.8 55.0</td><td>98.5%</td></tr><tr><td>Token retention: 22.2%</td></tr><tr><td>FastVECCV 2024</td><td>49.6 56.1</td><td>1490</td><td>59.6</td><td>60.2</td><td>61.8</td><td>50.6</td><td>55.9 51.3</td><td>83.9%</td></tr><tr><td>SparseVLMICML 2025</td><td>56.0</td><td>60.0</td><td>1696 80.5</td><td>67.1</td><td>73.8</td><td>54.9</td><td>53.4 51.4</td><td>92.9%</td></tr><tr><td>DARTEMNLP 2025</td><td>58.7</td><td>63.2</td><td>1840 80.1</td><td>69.1</td><td>75.9</td><td>56.4</td><td>50.5</td><td>55.3 95.9%</td></tr><tr><td>DivPrunecVPR 2025</td><td>59.2</td><td>62.3</td><td>1752 86.9</td><td>69.0</td><td>74.7</td><td>56.0</td><td>57.1 55.6</td><td>97.2%</td></tr><tr><td>VisionZipcVPR 2025</td><td>57.6</td><td>62.0</td><td>1762 83.2</td><td>68.9</td><td>75.6</td><td>56.8</td><td>57.1 54.5</td><td>96.5%</td></tr><tr><td>HiPruneACL 2026</td><td>57.3</td><td>62.2</td><td>1782 82.8</td><td>68.3</td><td>74.9</td><td>56.6</td><td>54.3</td><td>96.5%</td></tr><tr><td>EvoCutarXiv 2026.06</td><td>59.2</td><td>62.3 1790</td><td>85.7</td><td>68.7</td><td>76.2</td><td>57.1</td><td>55.2</td><td>96.6%</td></tr><tr><td>PruneSIDICLR 2026</td><td>58.8</td><td>62.1 1749</td><td>86.5</td><td>68.3</td><td>75.3</td><td>54.7</td><td>57.8 55.8</td><td>96.9%</td></tr><tr><td>Ours</td><td>58.9</td><td>62.2</td><td>1778 86.7</td><td>68.7</td><td>75.1</td><td>56.8</td><td>57.7 56.0</td><td>97.6%</td></tr><tr><td colspan="9">Token retention: 11.1%</td></tr><tr><td>FastVECCV 2024</td><td>46.1</td><td>48.0</td><td>1256</td><td>48.0 51.1</td><td>55.0</td><td>47.8</td><td>51.9</td><td>50.8 75.2%</td></tr><tr><td>SparseVLMICML 2025</td><td>52.7</td><td>56.2</td><td>1505</td><td>75.1 62.2</td><td>68.2</td><td>51.8</td><td>51.1</td><td>53.1 87.5%</td></tr><tr><td>DARTEMNLP 2025</td><td>55.9</td><td>60.6</td><td>1765</td><td>73.9 69.8</td><td>72.4</td><td>54.4</td><td>47.2</td><td>55.3 92.3%</td></tr><tr><td>DivPrunecvPR 2025</td><td>57.6</td><td>59.3</td><td>1638</td><td>85.6 68.3</td><td>72.9</td><td>55.5</td><td>55.4 57.5</td><td>95.1%</td></tr><tr><td>VisionZipcVPR 2025</td><td>55.1</td><td>60.1</td><td>1690</td><td>77.0 69.0</td><td>72.4</td><td>55.5</td><td>54.5 54.8</td><td>93.4%</td></tr><tr><td>HiPruneACL 2026</td><td>53.6</td><td>59.5</td><td>1646</td><td>73.0 68.9</td><td>69.2</td><td>54.9</td><td>54.4</td><td>91.7%</td></tr><tr><td>EvoCutarXiv 2026.06</td><td>56.6</td><td>61.8</td><td>1692</td><td>83.9 68.8</td><td>73.4</td><td>55.7</td><td>53.2</td><td>94.0%</td></tr><tr><td>PruneSIDICLR 2026</td><td>57.1</td><td>58.8</td><td>1733</td><td>83.8 67.8</td><td>73.7</td><td>54.2</td><td>56.1 56.9</td><td>95.1%</td></tr><tr><td>Ours</td><td>57.7</td><td>59.5</td><td>1708</td><td>85.6 68.4</td><td>72.9</td><td>54.9</td><td>56.1 56.3</td><td>95.4%</td></tr></table>

Table 2: Performance on LLaVA-NeXT-7B.
<table><tr><td>Method</td><td>GQA MMB</td><td>MME</td><td></td><td>POPE SQA VQAV2</td><td></td><td>SEED</td><td>RelAcc.</td></tr><tr><td>Uncompressed baseline (100%)</td><td></td><td></td><td></td><td></td><td></td><td>80.1</td><td></td></tr><tr><td>Vanilla</td><td>64.2</td><td>67.9</td><td>1842</td><td>86.4</td><td>70.2</td><td>70.2</td><td>100%</td></tr><tr><td>Token retention:</td><td></td><td>22.2%</td><td></td><td></td><td>79.1</td><td>66.7</td><td></td></tr><tr><td>VisionZip</td><td>61.3</td><td>66.3</td><td>1787</td><td>86.3</td><td>68.1</td><td></td><td>97.3%</td></tr><tr><td>PruneSID</td><td>61.6</td><td>64.2</td><td>1795</td><td>86.3 87.6</td><td>68.3 78.5 68.0 77.6</td><td>67.3</td><td>97.0%</td></tr><tr><td>Ours</td><td>63.3</td><td>64.2</td><td>1805</td><td></td><td></td><td>67.5</td><td>97.5%</td></tr><tr><td>Token retention:</td><td></td><td>11.1%</td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>VisionZip</td><td>59.3</td><td>63.1</td><td>1702</td><td>82.1</td><td>67.3</td><td>76.2 63.4</td><td>93.4%</td></tr><tr><td>PruneSID</td><td>60.5</td><td>63.0</td><td>1754</td><td>83.1</td><td>67.3</td><td>76.6 65.0</td><td>94.6%</td></tr><tr><td>Ours</td><td>61.0</td><td>63.1</td><td>1747</td><td>87.6</td><td>67.5</td><td>74.8 64.4</td><td>95.1%</td></tr><tr><td>Token retention:</td><td></td><td>:5.6%</td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>VisionZip</td><td>55.5</td><td>60.1</td><td>1630</td><td>74.8</td><td>68.3</td><td>71.4 58.3</td><td>88.5%</td></tr><tr><td>PruneSID</td><td>58.9</td><td>60.8</td><td>1704</td><td>76.9</td><td>67.1</td><td>73.8</td><td>62.5 91.4%</td></tr><tr><td>Ours</td><td>58.0</td><td>60.9</td><td>1627</td><td>87.1</td><td>68.5</td><td>71.5</td><td>60.9 91.9%</td></tr></table>

Table 3: Performance on Mini-Gemini-7B.
<table><tr><td>Method</td><td>GQA MMB</td><td></td><td></td><td></td><td>MME POPE SQA VQAV2</td><td>SEED</td><td>RelAcc.</td></tr><tr><td colspan="6">Uncompressed baseline (100%)</td><td>69.7</td><td>100%</td></tr><tr><td>Vanilla</td><td>62.4 69.3</td><td>1841</td><td>85.8</td><td>70.7</td><td>80.4</td><td></td><td></td></tr><tr><td colspan="6">Token retention: 33.3%</td><td></td><td>98.1%</td></tr><tr><td>VisionZip</td><td>60.3</td><td>68.9 1846 67.2</td><td>82.3 84.4</td><td>70.1 71.1</td><td>79.1 79.1</td><td>67.5 67.8</td><td>98.5%</td></tr><tr><td>PruneSID</td><td>61.2</td><td>1842 1884</td><td>85.4</td><td>71.5</td><td>78.0</td><td>67.4</td><td>98.8%</td></tr><tr><td>Ours</td><td>61.2</td><td>66.9</td><td></td><td></td><td></td><td></td><td></td></tr><tr><td colspan="6">Token retention: : 22.2%</td><td></td><td></td></tr><tr><td>VisionZip</td><td>58.7</td><td>68.1</td><td>1841</td><td>78.5 70.0</td><td>77.5</td><td>65.6</td><td>96.2%</td></tr><tr><td>PruneSID</td><td>60.1</td><td>66.6</td><td>1821 82.4</td><td>70.7</td><td>77.8</td><td>66.5</td><td>97.1%</td></tr><tr><td>Ours</td><td>60.5</td><td>67.4</td><td>1866 84.4</td><td>71.4</td><td>77.0</td><td>66.5</td><td>98.0%</td></tr><tr><td colspan="6">Token retention: 11.1%</td><td></td><td></td></tr><tr><td>VisionZip</td><td>55.8</td><td>65.9</td><td>1737</td><td>69.6</td><td>70.7 73.9</td><td>61.7</td><td>91.5%</td></tr><tr><td>PruneSID</td><td>58.3</td><td>63.1</td><td>1735</td><td>76.0 70.6</td><td>75.2 74.7</td><td>63.6</td><td>93.1%</td></tr><tr><td>Ours</td><td>58.5</td><td>64.9</td><td>1782</td><td>82.3</td><td>71.6</td><td>64.0</td><td>95.2%</td></tr></table>

## 5.1 MAIN RESULTS

We follow the token retention settings of prior work (Yang et al., 2025; Fang et al., 2026). Across the four models, MSDG-Prune achieves the highest or tied-highest RelAcc. among the compared methods at every reported budget.

Results on LLaVA-1.5. We first evaluate MSDG-Prune on LLaVA-1.5-7B (Liu et al., 2024a) across nine image understanding benchmarks, retaining 192, 128, and 64 of the original 576 visual tokens. As shown in Table 1, MSDG-Prune achieves RelAcc. of 98.5%, 97.6%, and 95.4%, respectively. When the budget is reduced to 64 tokens, MSDG-Prune preserves 95.4% of the baseline performance, compared with 95.1% for both PruneSID and DivPrune and 93.4% for VisionZip. These results show that representation dynamics provide effective selection cues even when 88.9% of the visual tokens are removed.

Results on LLaVA-NeXT. To evaluate pruning with longer visual sequences, we apply MSDG-Prune to LLaVA-NeXT-7B (Liu et al., 2024b), which processes multiple image crops and produces up to 2,880 visual tokens. Table 2 reports results at retention ratios of 22.2%, 11.1%, and 5.6%. Our method achieves RelAcc. of 97.5%, 95.1%, and 91.9%, respectively, exceeding both VisionZip and PruneSID at all three budgets. The advantage is particularly pronounced on POPE (Li et al., 2023b) under the smallest budget: with only 160 tokens, MSDG-Prune obtains an F1 score of 87.1, compared with 76.9 for PruneSID and 74.8 for VisionZip. This result highlights the effectiveness of the selected tokens in preserving evidence for object presence under severe compression.

Results on Mini-Gemini. We next evaluate MSDG-Prune on Mini-Gemini-7B (Li et al., 2025) to examine its applicability to a dual-encoder architecture. As shown in Table 3, MSDG-Prune achieves RelAcc. of 98.8%, 98.0%, and 95.2% at 33.3%, 22.2%, and 11.1% retention, respectively. The gains over PruneSID increase from 0.3 to 0.9 and 2.1 percentage points as the token budget decreases, while MSDG-Prune improve POPE F1 over PruneSID from 76.0 to 82.3 at 11.1% retention. The increasing advantage at smaller budgets shows that the proposed selection strategy remains effective when fewer tokens are available to represent the combined visual information.

Results on Qwen2.5-VL. To assess generalization beyond the CLIP-based models, we evaluate MSDG-Prune on Qwen2.5-VL-7B (Bai et al., 2025), which uses native dynamic resolution and spatial token merging. As reported in Table 4, MSDG-Prune achieves RelAcc. of 97.2%, 96.1%, and 92.4% at the three retention ratios, outperforming both compared methods throughout. The gains over PruneSID increase from 0.4 to 0.9 and 1.5 percentage points as retention decreases. Together with the results on the other models, these findings support the applicability of representation dynamics to visual token pruning across different encoder architectures and input-resolution schemes.

Table 4: Qwen2.5-VL-7B (HRB<sup>8K</sup>: HR-Bench 8K).
<table><tr><td>Method</td><td>GQA MMB</td><td></td><td>MME</td><td>POPE</td><td>SQA</td><td>VQAV2</td><td>HRB8K</td><td>RelAcc.</td></tr><tr><td colspan="9">Uncompressed baseline </td></tr><tr><td>Vanilla</td><td>60.9</td><td>83.9</td><td>2310</td><td>86.3</td><td>88.9</td><td>82.9</td><td>68.1</td><td>100%</td></tr><tr><td colspan="9">Token retention: 33.3%</td></tr><tr><td>VisionZip</td><td>56.6</td><td>78.9</td><td>2317</td><td>85.8</td><td>80.5</td><td>80.7</td><td>61.8</td><td>95.0%</td></tr><tr><td>PruneSID</td><td>59.8</td><td>80.9</td><td>2218</td><td>85.9</td><td>87.6</td><td>80.4</td><td>62.4</td><td>96.8%</td></tr><tr><td>Ours</td><td>58.5</td><td>82.6</td><td>2332</td><td>84.7</td><td>87.8</td><td>80.5</td><td>61.9</td><td>97.2%</td></tr><tr><td colspan="9">Token retention: 22.2%</td></tr><tr><td>VisionZip</td><td>54.6</td><td>76.8</td><td>2224</td><td>83.4</td><td>80.4</td><td>78.5</td><td>61.3</td><td>92.8%</td></tr><tr><td>PruneSID</td><td>59.0</td><td>78.0</td><td>2169</td><td>85.6</td><td>86.9</td><td>78.7</td><td>61.8</td><td>95.2%</td></tr><tr><td>Ours</td><td>57.7</td><td>81.2</td><td>2297</td><td>84.0</td><td>87.3</td><td>78.9</td><td>61.9</td><td>96.1%</td></tr><tr><td colspan="9">Token retention: 11.1%</td></tr><tr><td>VisionZip</td><td>53.2</td><td>75.8</td><td>2025</td><td>78.9</td><td>80.1</td><td>73.8</td><td>58.6</td><td>88.9%</td></tr><tr><td>PruneSID</td><td>55.8</td><td>73.9</td><td>2076</td><td>80.2</td><td>86.5</td><td>74.6</td><td>58.9</td><td>90.9%</td></tr><tr><td>Ours</td><td>55.4</td><td>78.4</td><td>2178</td><td>81.3</td><td>86.3</td><td>75.0</td><td>59.0</td><td>92.4%</td></tr></table>

Inference Efficiency. Following prior work (Fang et al., 2026; Yang et al., 2025), we measure per-sample latency on LLaVA-NeXT-7B using a single NVIDIA A800-80 GB GPU. As shown in Table 5, retaining 160 tokens reduces prefilling time from 218 ms to 27.8 ms and total inference time from 254 ms to 96 ms, yielding speedups of 7.8× and 2.6×, respectively. POPE-F1 score reaches 87.1, compared with 86.4 without pruning. Compared

Table 5: Inference latency and POPE F1 score on LLaVA-NeXT-7B. Latencies are averaged per sample using a single NVIDIA A800-80 GB GPU.
<table><tr><td>Method</td><td>Tokens</td><td>Inference Time ↓</td><td>Prefilling Time ↓</td><td>POPE (F1) ↑</td></tr><tr><td>LLaVA-NeXT</td><td>2880</td><td>254ms</td><td>218 ms</td><td>86.4</td></tr><tr><td>SparseVLM</td><td>160</td><td>199 ms</td><td>119 ms</td><td>69.3</td></tr><tr><td>VisionZip</td><td>160</td><td>84 ms</td><td>27.8ms</td><td>74.8</td></tr><tr><td>PruneSID</td><td>160</td><td>89 ms</td><td>27.8ms</td><td>76.9</td></tr><tr><td>Ours</td><td>160</td><td>96 ms</td><td>27.8 ms</td><td>87.1</td></tr></table>

with PruneSID, MSDG-Prune matches prefilling latency and gains 10.2 percentage points with 7 ms additional total latency, balancing inference efficiency and object grounding.

## 5.2 ABLATION STUDY

As shown in Table 6, we conduct component ablations on Qwen2.5-VL-7B to examine the contributions of the grouping representation, visual saliency, sink filtering, and budget allocation.

Representation and Saliency Signals. Clustering with update directions $d _ { i }$ outperforms grouping with encoder-output features at all retention ratios. Together with the analysis in Section 3.2, this supports using representation changes to form semantically coherent groups. For visual saliency, u<sub>i</sub> matches mean received attention in the final encoder layer at 33.3% retention and improves RelAcc. by 0.4 and 0.6 points at the smaller budgets. Compared with accumulated update magnitude, computing $u _ { i }$ from endpoint displacement yields gains of 0.2–1.4 points on Qwen2.5-VL, supporting net representation change as the saliency cue. The effect of the group count K is evaluated in Appendix C.6. These results support our design choices for the representation and saliency signals.

Sink Filtering and Budget Allocation. We examine the contribution of sink removal and budget allocation. Retaining sink tokens reduces RelAcc. at all retention ratios, with a drop of 0.9 percentage points at 22.2% retention. This result supports sink filtering. Instead of using group scores, allocating budgets in proportion to group size also lowers RelAcc. by 0.5–1.1 percentage points. The largest reduction occurs at 11.1% retention, highlighting the importance of allocating limited tokens according to group importance.

Complementarity of Scoring Signals. We compare random selection, global ranking by query relevance, global ranking by query-weighted visual saliency, and the full method. Query relevance alone performs below random selection at all three retention ratios, showing that similarity to the query embeddings is insufficient as a standalone selection criterion. For global token ranking, combining query relevance with visual saliency improves RelAcc. by 3.2–5.8 percentage points over query relevance alone across the three retention ratios. Group-wise pruning further improves over global ranking with the combined score by 1.1 and 1.6 points at 22.2% and 11.1% retention, respectively.

Saliency Window Endpoints. We compare saliency windows spanning five consecutive layers at different encoder depths. The default windows achieve the highest RelAcc. among the configurations in Figure 5 at both retention ratios. Using the final five encoder updates instead reduces RelAcc. by 3.7 and 5.8 percentage points on Qwen2.5-VL, and by 2.0 and 2.2 points on LLaVA-

Table 6: Component ablations and staged token selection on Qwen2.5-VL-7B. Ours is the full method; Output features uses encoder-output features for grouping. Query (global) ranks by query relevance without grouping; Query×Visual (global) ranks by the product of query relevance and visual saliency, still without grouping.
<table><tr><td rowspan="2">Method</td><td colspan="5">Retain 33.3%</td><td colspan="5">Retain 22.2%</td><td colspan="5">Retain 11.1%</td></tr><tr><td>GQA MME</td><td></td><td>POPE SQA RelAcc.</td><td></td><td></td><td>GQA MME</td><td></td><td>POPE</td><td>SQA</td><td>RelAcc.</td><td>GQA MME POPE SQA RelAcc.</td><td></td><td></td><td></td><td></td></tr><tr><td>Ours</td><td></td><td>58.5 2332</td><td>84.7 87.8</td><td></td><td>98.5%</td><td>57.7</td><td>2297</td><td>84.0 87.3</td><td></td><td>97.4%</td><td>55.4 2178</td><td></td><td>81.3 86.3</td><td></td><td>94.1%</td></tr><tr><td>Output features</td><td>58.1</td><td>2286</td><td>84.0</td><td>87.5</td><td>97.5%</td><td>56.9</td><td>2254</td><td>82.8</td><td>87.1</td><td>96.2%</td><td>54.8</td><td>2178</td><td>80.1</td><td>86.0</td><td>93.5%</td></tr><tr><td>Attention</td><td>57.9</td><td>2319</td><td>86.1</td><td>87.9</td><td>98.5%</td><td>56.9</td><td>2268</td><td>84.4</td><td>87.6</td><td>97.0%</td><td>53.8</td><td>2176</td><td>81.3</td><td>86.5</td><td>93.5%</td></tr><tr><td>Accumulated updates</td><td>58.5</td><td>2317</td><td>84.4</td><td>87.9</td><td>98.3%</td><td>56.5</td><td>2272</td><td>83.2</td><td>87.3</td><td>96.4%</td><td>54.3</td><td>2129</td><td>80.1</td><td>85.9</td><td>92.7%</td></tr><tr><td>Keep sink tokens</td><td>58.4</td><td>2330</td><td>84.7</td><td>87.7</td><td>98.4%</td><td>57.1</td><td>2264</td><td>83.1</td><td>87.2</td><td>96.5%</td><td>54.8</td><td>2143</td><td>81.2</td><td>86.3</td><td>93.5%</td></tr><tr><td>Size-based budget</td><td>58.8</td><td>2281</td><td>84.4</td><td>87.8</td><td>98.0%</td><td>57.9</td><td>2247</td><td>83.3</td><td>87.2</td><td>96.7%</td><td>55.4</td><td>2111</td><td>80.2</td><td>85.9</td><td>93.0%</td></tr><tr><td>Random</td><td>55.2</td><td>2275</td><td>82.3</td><td>87.3</td><td>95.7%</td><td>56.9</td><td>2166</td><td>81.1</td><td>86.6</td><td>94.6%</td><td>53.6</td><td>1995</td><td>77.4</td><td>85.2</td><td>90.0%</td></tr><tr><td>Query (global)</td><td>56.1</td><td>2247</td><td>80.9</td><td>87.3</td><td>95.3%</td><td>54.6</td><td>2146</td><td>77.7</td><td>86.9</td><td>92.6%</td><td>51.4</td><td>1935</td><td>71.5</td><td>85.1</td><td>86.7%</td></tr><tr><td>Query ×Visual (global)</td><td>58.3</td><td>2337</td><td>84.8</td><td>87.8</td><td>98.5%</td><td>56.6</td><td>2282</td><td>82.2</td><td>87.2</td><td>96.3%</td><td>52.4</td><td>2152</td><td>81.1</td><td>86.1</td><td>92.5%</td></tr></table>

1.5, respectively. This degradation is consistent with the increased background responses near the encoder output, supporting saliency estimation within the late foreground-enhanced stage.

![](images/71523ff4499767965618b333889ffc83b90b7ef14061b2980cb7f57b2053d367.jpg)  
(a) Qwen2.5-VL

![](images/efc3e5ff5ff37e9e9d4b4fe68ccff4171c072953677c8e9afa2e9611294aa200.jpg)  
(b) LLaVA-1.5  
Figure 5: Effect of saliency window start on RelAcc. for (a) Qwen2.5-VL-7B and (b) LLaVA-1.5-7B at 22.2% and 11.1% token retention. Window starts ℓ<sub>s</sub> are selected as representative points across the stages identified in the preceding analysis. Each window spans five consecutive encoder updates. Dashed lines mark the default starts.

## 5.3 LIMITATIONS

Because MSDG-Prune requires intermediate vision encoder states, it cannot be applied through interfaces that expose only final predictions. Moreover, generalizing our method to new backbones entails analyzing and identifying specific model-dependent components, which requires an in-depth understanding of the model architecture.

## 6 CONCLUSION

We present MSDG-Prune, a training-free visual token pruning method guided by encoder representation dynamics. It uses update directions for grouping and combines update magnitudes with query relevance for saliency. At 11.1% token retention, MSDG-Prune achieves 95.4% and 92.4% RelAcc. on LLaVA-1.5-7B and Qwen2.5-VL-7B, respectively. On LLaVA-NeXT-7B, it achieves a 7.8× prefilling speedup at 5.6% retention.

## AI USE STATEMENT

In this work, we used generative AI tools (Cursor, Codex) to generate and debug experimental code and evaluation scripts for selected benchmarks, and to edit the manuscript to improve English readability and polish presentation (e.g., phrasing, clarity, and LAT X formatting). We did not use AI tools for research design, theoretical or mathematical claims, data analysis, or interpretation of results. The authors reviewed AI-assisted text, tested AI-assisted code, and obtained all reported results by running the evaluations. The authors take responsibility for the final content.

## ETHICS STATEMENT

The research presented in this paper focuses on visual token pruning for MLLMs to improve their computational efficiency, and does not involve human subjects, personally identifiable information, or the collection of new datasets. The research process of this paper does not violate the ICLR Code of Ethics. We do not train new models; our method is an inference-time pruning technique applied to existing publicly available MLLMs. We do not foresee additional discrimination, bias, or fairness concerns beyond those already inherent in the underlying base models, nor do we expect our method to introduce new harmful generation capabilities.

## REPRODUCIBILITY STATEMENT

The base models and datasets used in our experiments are all publicly available and cited, which supports reproducibility of the evaluation setting. To preserve anonymity during double-blind review, we do not release source code at submission time; upon acceptance, we will publicly release the source code to support reproducibility.

## REFERENCES

Josh Achiam, Steven Adler, Sandhini Agarwal, Lama Ahmad, Ilge Akkaya, Florencia Leoni Aleman, Diogo Almeida, Janko Altenschmidt, Sam Altman, Shyamal Anadkat, et al. Gpt-4 technical report. arXiv preprint arXiv:2303.08774, 2023.

Saeed Ranjbar Alvar, Gursimran Singh, Mohammad Akbari, and Yong Zhang. Divprune: Diversitybased visual token pruning for large multimodal models. In 2025 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pp. 9392–9401. IEEE, 2025.

Shuai Bai, Keqin Chen, Xuejing Liu, Jialin Wang, Wenbin Ge, Sibo Song, Kai Dang, Peng Wang, Shijie Wang, Jun Tang, Humen Zhong, Yuanzhi Zhu, Mingkun Yang, Zhaohai Li, Jianqiang Wan, Pengfei Wang, Wei Ding, Zheren Fu, Yiheng Xu, Jiabo Ye, Xi Zhang, Tianbao Xie, Zesen Cheng, Hang Zhang, Zhibo Yang, Haiyang Xu, and Junyang Lin. Qwen2.5-vl technical report, 2025. URL https://arxiv.org/abs/2502.13923.

Daniel Bolya, Cheng-Yang Fu, Xiaoliang Dai, Peizhao Zhang, Christoph Feichtenhofer, and Judy Hoffman. Token merging: Your vit but faster. arXiv preprint arXiv:2210.09461, 2022.

Holger Caesar, Jasper Uijlings, and Vittorio Ferrari. Coco-stuff: Thing and stuff classes in context. In 2018 IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 1209–1218. IEEE, 2018.

Liang Chen, Haozhe Zhao, Tianyu Liu, Shuai Bai, Junyang Lin, Chang Zhou, and Baobao Chang. An image is worth 1/2 tokens after layer 2: Plug-and-play inference acceleration for large visionlanguage models. In European Conference on Computer Vision, pp. 19–35. Springer, 2024.

Joonmyung Choi, Sanghyeok Lee, Byungoh Ko, Eunseo Kim, Jihyung Kil, and Hyunwoo J Kim. Representation shift: Unifying token compression with flashattention. In 2025 IEEE/CVF International Conference on Computer Vision (ICCV), pp. 20456–20466. IEEE, 2025.

Tri Dao. Flashattention-2: Faster attention with better parallelism and work partitioning. In International Conference on Learning Representations, 2024.

Tri Dao, Dan Fu, Stefano Ermon, Atri Rudra, and Christopher Re. Flashattention: Fast and memory- ´ efficient exact attention with io-awareness. Advances in neural information processing systems, 35:16344–16359, 2022.

Timothee Darcet, Maxime Oquab, Julien Mairal, and Piotr Bojanowski. Vision transformers need´ registers. In International conference on learning representations, 2024.

Mark Endo, Xiaohan Wang, and Serena Yeung-Levy. Feather the throttle: Revisiting visual token pruning for vision-language model acceleration. In 2025 IEEE/CVF International Conference on Computer Vision (ICCV), pp. 22826–22835. IEEE, 2025.

Yingqi Fan, Junlong Tong, Anhao Zhao, and Xiaoyu Shen. What do visual tokens really encode? uncovering sparsity and redundancy in multimodal large language models. In Proceedings ofthe IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pp. 11987–11997, June 2026.

Zhengyao Fang, Pengyuan Lyu, Chengquan Zhang, Guangming Lu, Jun Yu, and Wenjie Pei. Prune redundancy, preserve essence: Vision token compression in vlms via synergistic importancediversity. In International Conference on Learning Representations, 2026.

Chaoyou Fu, Peixian Chen, Yunhang Shen, Yulei Qin, Mengdan Zhang, Xu Lin, Jinrui Yang, Xiawu Zheng, Ke Li, Xing Sun, Yunsheng Wu, Rongrong Ji, Caifeng Shan, and Ran He. Mme: A comprehensive evaluation benchmark for multimodal large language models. In D. Belgrave, C. Zhang, H. Lin, R. Pascanu, P. Koniusz, M. Ghassemi, and N. Chen (eds.), Advances in Neural Information Processing Systems, volume 38, Main Conference. Curran Associates, Inc., 2025. doi: 10.52202/085713-4899. URL https://proceedings.neurips.cc/paper\_files/paper/2025/file/ d79a27cf2772fe00be7f341efc0eb517-Paper-Datasets\_and\_Benchmarks\_ Track.pdf.

Yash Goyal, Tejas Khot, Douglas Summers-Stay, Dhruv Batra, and Devi Parikh. Making the v in vqa matter: Elevating the role of image understanding in visual question answering. In Proceedings of the IEEE conference on computer vision and pattern recognition, pp. 6904–6913, 2017.

Shuhang Gu, Andreas Lugmayr, Martin Danelljan, Manuel Fritsche, Julien Lamour, and Radu Timofte. Div8k: Diverse 8k resolution image dataset. In 2019 IEEE/CVF International Conference on Computer Vision Workshop (ICCVW), pp. 3512–3516. IEEE, 2019.

Danna Gurari, Qing Li, Abigale J Stangl, Anhong Guo, Chi Lin, Kristen Grauman, Jiebo Luo, and Jeffrey P Bigham. Vizwiz grand challenge: Answering visual questions from blind people. In 2018 IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 3608–3617. IEEE, 2018.

Kurt Hornik, Ingo Feinerer, Martin Kober, and Christian Buchta. Spherical k-means clustering. Journal ofstatistical software, 50:1–22, 2012.

Drew A Hudson and Christopher D Manning. Gqa: A new dataset for real-world visual reasoning and compositional question answering. In 2019 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pp. 6700–6709. IEEE, 2019.

Nicholas Jiang, Amil Dravid, Alexei A Efros, and Yossi Gandelsman. Vision transformers don’t need trained registers. In Advances in Neural Information Processing Systems, volume 38, pp. 56557–56595, 2025.

Ivan Krasin, Tom Duerig, Neil Alldrin, Vittorio Ferrari, Sami Abu-El-Haija, Alina Kuznetsova, Hassan Rom, Jasper Uijlings, Stefan Popov, Andreas Veit, et al. Openimages: A public dataset for large-scale multi-label and multi-class image classification. Dataset availablefrom https://github. com/openimages, 2(3):18, 2017.

Ranjay Krishna, Yuke Zhu, Oliver Groth, Justin Johnson, Kenji Hata, Joshua Kravitz, Stephanie Chen, Yannis Kalantidis, Li-Jia Li, David A Shamma, et al. Visual genome: Connecting language and vision using crowdsourced dense image annotations. Internationaljournal ofcomputer vision, 123(1):32–73, 2017.

Ao Li, Yuxiang Duan, Jinghui Zhang, Congbo Ma, Yutong Xie, Gustavo Carneiro, Mohammad Yaqub, and Hu Wang. Transprune: Token transition pruning for efficient large vision-language model. In Proceedings ofthe IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 39529–39538, 2026.

Bohao Li, Rui Wang, Guangzhi Wang, Yuying Ge, Yixiao Ge, and Ying Shan. Seed-bench: Benchmarking multimodal llms with generative comprehension. arXiv preprint arXiv:2307.16125, 2023a.

Yanwei Li, Yuechen Zhang, Chengyao Wang, Zhisheng Zhong, Yixin Chen, Ruihang Chu, Shaoteng Liu, and Jiaya Jia. Mini-gemini: Mining the potential of multi-modality vision language models. IEEE Transactions on Pattern Analysis and Machine Intelligence, 2025.

Yifan Li, Yifan Du, Kun Zhou, Jinpeng Wang, Xin Zhao, and Ji-Rong Wen. Evaluating object hallucination in large vision-language models. In Proceedings of the 2023 conference on empirical methods in natural language processing, pp. 292–305, 2023b.

Tsung-Yi Lin, Michael Maire, Serge Belongie, James Hays, Pietro Perona, Deva Ramanan, Piotr Dollar, and C Lawrence Zitnick. Microsoft coco: Common objects in context. In ´ European conference on computer vision, pp. 740–755. Springer, 2014.

Haotian Liu, Chunyuan Li, Qingyang Wu, and Yong Jae Lee. Visual instruction tuning. Advances in neural information processing systems, 36:34892–34916, 2023.

Haotian Liu, Chunyuan Li, Yuheng Li, and Yong Jae Lee. Improved baselines with visual instruction tuning. In 2024 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pp. 26286–26296, 2024a.

Haotian Liu, Chunyuan Li, Yuheng Li, Bo Li, Yuanhan Zhang, Sheng Shen, and Yong Jae Lee. Llavanext: Improved reasoning, ocr, and world knowledge, 2024b.

Jizhihui Liu, Guangdao Zhu, Feiyi Du, Niu Lian, Jun Li, Bin Chen, Weili Guan, and Yaowei Wang. Hiprune: Hierarchical attention for efficient token pruning in vision-language models. In Findings ofthe Associationfor Computational Linguistics: ACL 2026, pp. 3274–3291, 2026.

Yuan Liu, Haodong Duan, Yuanhan Zhang, Bo Li, Songyang Zhang, Wangbo Zhao, Yike Yuan, Jiaqi Wang, Conghui He, Ziwei Liu, et al. Mmbench: Is your multi-modal model an all-around player? In European conference on computer vision, pp. 216–233. Springer, 2024c.

Hongyu Lu, Feng Zhang, Wenwei Jin, Huanling Hu, Pengfei Zhang, Yao Hu, Jiawei Li, and Shikai Jiang. Evocut: Multi-layer evolution-aware visual token compression for efficient large visionlanguage models. arXiv preprint arXiv:2606.01756, 2026.

Pan Lu, Swaroop Mishra, Tanglin Xia, Liang Qiu, Kai-Wei Chang, Song-Chun Zhu, Oyvind Tafjord, Peter Clark, and Ashwin Kalyan. Learn to explain: Multimodal reasoning via thought chains for science question answering. Advances in neural information processing systems, 35:2507–2521, 2022.

Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, et al. Learning transferable visual models from natural language supervision. In International conference on machine learning, pp. 8748–8763. PmLR, 2021.

Yuzhang Shang, Mu Cai, Bingxin Xu, Yong Jae Lee, and Yan Yan. Llava-prumerge: Adaptive token reduction for efficient large multimodal models. In 2025 IEEE/CVF International Conference on Computer Vision (ICCV), pp. 22857–22867. IEEE, 2025.

Amanpreet Singh, Vivek Natarajan, Meet Shah, Yu Jiang, Xinlei Chen, Dhruv Batra, Devi Parikh, and Marcus Rohrbach. Towards vqa models that can read. In 2019 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pp. 8317–8326. IEEE, 2019.

Rinyoichi Takezoe, Yaqian Li, Zi-Hao Bo, Anzhou Hou, Mo Guang, and Kaiwen Long. Learnpruner: Rethinking attention-based token pruning in vision language models. In International Conference on Learning Representations, 2026.

Wenbin Wang, Liang Ding, Minyan Zeng, Xiabin Zhou, Li Shen, Yong Luo, Wei Yu, and Dacheng Tao. Divide, conquer and combine: A training-free framework for high-resolution image perception in multimodal large language models. In Proceedings of the AAAI Conference on Artificial Intelligence, volume 39, pp. 7907–7915, 2025.

Zichen Wen, Yifeng Gao, Shaobo Wang, Junyuan Zhang, Qintong Zhang, Weijia Li, Conghui He, and Linfeng Zhang. Stop looking for “important tokens” in multimodal language models: Duplication matters more. In Proceedings of the 2025 Conference on Empirical Methods in Natural Language Processing, pp. 9961–9980, 2025.

Senqiao Yang, Yukang Chen, Zhuotao Tian, Chengyao Wang, Jingyao Li, Bei Yu, and Jiaya Jia. Visionzip: Longer is better but not necessary in vision language models. In 2025 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pp. 19792–19802. IEEE, 2025.

Xiang Yue, Yuansheng Ni, Kai Zhang, Tianyu Zheng, Ruoqi Liu, Ge Zhang, Samuel Stevens, Dongfu Jiang, Weiming Ren, Yuxuan Sun, et al. Mmmu: A massive multi-discipline multimodal understanding and reasoning benchmark for expert agi. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pp. 9556–9567, 2024.

Kaichen Zhang, Bo Li, Peiyuan Zhang, Fanyi Pu, Joshua Adrian Cahyono, Kairui Hu, Shuai Liu, Yuanhan Zhang, Jingkang Yang, Chunyuan Li, et al. Lmms-eval: Reality check on the evaluation of large multimodal models. In Findings of the Association for Computational Linguistics: NAACL 2025, pp. 881–916, 2025a.

Qizhe Zhang, Mengzhen Liu, Lichen Li, Ming Lu, Yuan Zhang, Junwen Pan, Qi She, and Shanghang Zhang. Beyond attention or similarity: Maximizing conditional diversity for token pruning in mllms. In D. Belgrave, C. Zhang, H. Lin, R. Pascanu, P. Koniusz, M. Ghassemi, and N. Chen (eds.), Advances in Neural Information Processing Systems, volume 38, Main Conference, pp. 25438–25468. Curran Associates, Inc., 2025b. doi: 10.52202/

085713-0855. URL https://proceedings.neurips.cc/paper\_files/paper/ 2025/file/2433fec2144ccf5fea1c9c5ebdbc3924-Paper-Conference.pdf.

Yuan Zhang, Chun-Kai Fan, Junpeng Ma, Wenzhao Zheng, Tao Huang, Kuan Cheng, Denis A Gudovskiy, Tomoyuki Okuno, Yohei Nakata, Kurt Keutzer, and Shanghang Zhang. Sparsevlm: Visual token sparsification for efficient vision-language model inference. In Proceedings of the 42nd International Conference on Machine Learning, volume 267, pp. 74840–74857, 2025c.

Appendix Overview. For convenient reference, the appendix is organized as follows:

• Appendix A: Additional Algorithm Details

• Appendix C: Additional Experimental Results

• Appendix D: Details of Evaluation Benchmarks

• Appendix E: Qualitative Analysis of Retained Tokens

## A ADDITIONAL ALGORITHM DETAILS

We detail the token selection algorithm, centroid initialization, and stopping criteria, show that the grouping objective is equivalent to minimizing squared chord distance, and relate endpoint displacement to the accumulated update magnitude evaluated in Section 5.2.

Algorithm 1 MSDG-Prune Visual Token Selection   
Require: encoder states $\{ h _ { i } ^ { \ell } : 1 \le i \le N , 0 \le \ell \le L \} ,$ , projected visual tokens $\{ \pmb { x } _ { i } \} _ { i = 1 } ^ { N }$ , where $N$ is the   
number of visual tokens before pruning and $L$ is the number of encoder layers, query embeddings $\{ t _ { j } \} _ { j \in \mathcal { Q } } ,$   
where Q indexes the query tokens, budget $B ,$ group count $K , \ell _ { 0 }$ and $\ell _ { d }$ (start and end layers of the update   
direction $d _ { i } ) , \ell _ { \star }$ (peak concentration layer; Appendix ${ \mathrm { C . l } } ) , \ell _ { s }$ and $\ell _ { e }$ (start and end of the saliency window   
for the visual saliency score), diagnostic coordinate $d _ { s } ,$ threshold $\tau ,$ and integer initialization seed $s ;$   
unit $( z ) = z /$ max $\{ \| z \| , 1 0 ^ { - 1 2 } \}$ . TopId $\mathrm { x } _ { b }$ returns the indices of the b largest scores, breaking ties by   
original token order.   
1: $\mathcal { V }  \{ i : | ( h _ { i } ^ { \ell _ { \star } + 1 } ) _ { d _ { s } } | \leq \tau \}$ ; require $B \leq | \mathcal { V } | , 1 \leq K \leq | \mathcal { V } |$ , and $\mathcal { Q } \neq \mathcal { O }$   
2: for each token i ∈ V do   
3: $\Delta h _ { i } \gets \mathrm { u n i t } ( h _ { i \cdot } ^ { \ell _ { d } } ) - \mathrm { u n i t } ( h _ { i } ^ { \ell _ { 0 } } ) ; d _ { i } \gets \mathrm { u n i t } ( \Delta h _ { i } )$   
4: $u _ { i } \gets \| h _ { i } ^ { \ell _ { e } } - h _ { i } ^ { \ell _ { s } } \| ; \alpha _ { i } \gets \operatorname* { m a x } _ { j \in \mathcal { Q } } \operatorname { u n i t } ( \pmb { x } _ { i } ) ^ { \top } \operatorname { u n i t } ( \pmb { t } _ { j } )$   
5: end for   
6: Set $s _ { i } \gets \alpha _ { i } u _ { i }$ for each $i \in \mathcal V$   
7: Require max $\displaystyle i \in \mathcal { V } \left\| d _ { i } \right\| > 0 ;$ initialize centroids $\{ \mu _ { c } \} _ { c = 1 } ^ { K }$ by deterministic farthest-point sampling   
8: for $\bar { t } = 1 , \ldots ,$ 10 do   
9: Assign each $i \in \nu$ to a centroid maximizing $d _ { i } ^ { \top } \mu _ { c } ;$ set m<sub>i</sub> $ \operatorname* { m a x } _ { c } d _ { i } ^ { \top } \mu _ { c } ,$ form $\mathcal { C } _ { c } ,$ and compute   
$\boldsymbol { \mathcal { L } ^ { ( t ) } }$   
10: $\mathbf { i f } \ t > 1$ and (labels are unchanged or the relative loss change is below $1 0 ^ { - 5 } )$ then break   
11: Set $\begin{array} { r } { \mu _ { c }  \mathrm { u n i t } ( \sum _ { i \in \mathcal { C } _ { c } } d _ { i } ) } \end{array}$ for each nonempty $\mathcal { C } _ { c }$   
12: For empty groups, set $\mu _ { c }$ to distinct $d _ { i }$ in ascending $m _ { i }$ order (ties: original token order)   
13: end for; reassign tokens to the final centroids to obtain $\{ \mathcal { C } _ { c } \} _ { c = 1 } ^ { K }$   
14: Set $\begin{array} { r } { g _ { c }  \vert \mathcal { C } _ { c } \vert ^ { - 1 } \sum _ { i \in \mathcal { C } _ { c } } s _ { i } \mathrm { i f }  \mathcal { C } _ { c }  > 0 , } \end{array}$ and $g _ { c } \gets 0$ otherwise   
15: $\begin{array} { r } { p _ { c }  \exp ( g _ { c } ) / \sum _ { r = 1 } ^ { K } } \end{array}$ exp(g<sub>r</sub>) for $c = 1 , \ldots , K$   
16: $b _ { c } \gets \operatorname* { m i n } \{ | { \mathcal { C } } _ { c } | , \lfloor B p _ { c } \rfloor \}$ for $c = 1 , \ldots , K ; R \gets B - \sum _ { c = 1 } ^ { K } b _ { c }$   
17: Visit groups with $\bar { b } _ { c } < | \mathcal { C } _ { c } |$ in descending order of $B p _ { c } - \overline { { { \lfloor B p _ { c } \rfloor } } }$ (ties: lower c)   
18: While $\begin{array} { r } { \hat { R } > 0 , } \end{array}$ add one token to each visited group: $b _ { c } \gets b _ { c } \bar { + } 1 , R \gets R - 1$   
19: Visit groups in descending order of ${ p _ { c } }$ (ties: lower c); set $\Delta b _ { c } \gets \operatorname* { m i n } \{ R , | { \mathcal { C } } _ { c } | - b _ { c } \} , b _ { c } \gets b _ { c } + \Delta b _ { c }$   
$R  \mathbf { \bar { { R } } } - \Delta b _ { c }$   
20: $S _ { c } \gets \mathrm { T o p I d x } _ { \boldsymbol { b } _ { c } } ( \{ s _ { i } \} _ { i \in \mathcal { C } _ { c } } )$ for each $c = 1 , \ldots , K$   
21: $\begin{array} { r } { S \gets \mathrm { S o r t } ( \bigcup _ { c = 1 } ^ { K ^ { - } } S _ { c } ) } \end{array}$ by the original visual token indices   
22: return $\mathbf { \tilde { \mathbf { X } } } \gets [ \mathbf { \mathbf { x } } _ { i } ] _ { i \in \mathcal { S } }$

Visual token selection. Algorithm 1 summarizes the visual token selection procedure of MSDG-Prune. Update directions and visual saliency scores are computed from vision encoder states, while query relevance is measured between projected visual tokens and query embeddings in the LLM embedding space. Sink tokens are excluded from grouping, group scoring, and token selection.

Deterministic centroid initialization and stopping. We initialize the centroids by deterministic farthest-point sampling using inner products of unit update directions. Let $M = | \nu |$ , and let $v _ { 0 } , \ldots , v _ { M - 1 }$ list the candidate token indices in their original order. The integer seed s determines the first index $i _ { 1 } = v _ { s }$ <sub>mod M</sub> and centroid $\mu _ { 1 } = d _ { i _ { 1 } }$ . Each subsequent centroid is selected as

$$
i _ { c } = \arg \operatorname* { m a x } _ { i \in \mathcal { V } \backslash \{ i _ { 1 } , \ldots , i _ { c - 1 } \} } \left( 1 - \operatorname* { m a x } _ { 1 \leq r < c } d _ { i } ^ { \top } \mu _ { r } \right) , \qquad \mu _ { c } = d _ { i _ { c } } , \quad c = 2 , \ldots , K .\tag{5}
$$

This rule selects an unchosen token with the largest cosine distance to its nearest existing centroid, breaking ties by original token order. At iteration $t ,$ let $a _ { i } ^ { ( t ) }$ denote the group label of token i, and let $\mu _ { c } ^ { ( t ) }$ denote centroid $c .$ The grouping loss is the mean cosine distance to the assigned centroids,

$$
\mathcal { L } ^ { ( t ) } = \frac { 1 } { M } \sum _ { i \in \mathcal { V } } \left( 1 - d _ { i } ^ { \top } \mu _ { a _ { i } ^ { ( t ) } } ^ { ( t ) } \right) .\tag{6}
$$

Minimizing this loss is equivalent to maximizing the cosine alignment objective in Equation 3. The stopping criterion based on relative loss change is

$$
\frac { \left| \mathcal { L } ^ { \left( t - 1 \right) } - \mathcal { L } ^ { \left( t \right) } \right| } { \operatorname* { m a x } \bigl ( \left| \mathcal { L } ^ { \left( t - 1 \right) } \right| , 1 0 ^ { - 1 2 } \bigr ) } < 1 0 ^ { - 5 } .\tag{7}
$$

Each grouping run stops when the assignment labels remain unchanged, the relative loss change satisfies Equation 7, or 10 iterations have been completed. The label and loss comparisons are evaluated from the second iteration onward.

Equivalent grouping objective. In the nondegenerate case where every $d _ { i }$ and $\mu _ { c }$ has unit norm, any partition of the candidate tokens satisfies

$$
\begin{array} { c } { { \displaystyle \sum _ { c = 1 } ^ { K } \sum _ { i \in { \mathcal C } _ { c } } \| d _ { i } - \mu _ { c } \| ^ { 2 } = \displaystyle \sum _ { c = 1 } ^ { K } \sum _ { i \in { \mathcal C } _ { c } } \big ( 2 - 2 d _ { i } ^ { \top } \mu _ { c } \big ) } } \\ { { = 2 | \mathcal V | - 2 \displaystyle \sum _ { c = 1 } ^ { K } \sum _ { i \in { \mathcal C } _ { c } } d _ { i } ^ { \top } \mu _ { c } . } } \end{array}\tag{8}
$$

Thus, maximizing the objective in Equation 3 is equivalent to minimizing the total squared chord distance between unit update directions and unit centroids. The grouping objective therefore depends only on the unit update directions.

Endpoint displacement and accumulated update magnitude. Let $\delta _ { i } ^ { \ell } = h _ { i } ^ { \ell + 1 } - h _ { i } ^ { \ell }$ be the update through layer ℓ. The visual saliency score in Section 4.2 satisfies

$$
u _ { i } = \| h _ { i } ^ { \ell _ { e } } - h _ { i } ^ { \ell _ { s } } \| = \left\| \sum _ { \ell = \ell _ { s } } ^ { \ell _ { e } - 1 } \delta _ { i } ^ { \ell } \right\| \leq \sum _ { \ell = \ell _ { s } } ^ { \ell _ { e } - 1 } \| \delta _ { i } ^ { \ell } \| = \sum _ { \ell = \ell _ { s } } ^ { \ell _ { e } - 1 } v _ { i } ^ { \ell } .\tag{9}
$$

The second equality follows from telescoping, and the inequality follows from the triangle inequality. Equality holds when all nonzero updates point in the same direction. Endpoint displacement measures the representation change between the window endpoints, allowing opposing updates to cancel. The accumulated update magnitude measures the total length of the representation path within the window. The ablations in Tables 6 and 12 compare these two aggregation rules in downstream compression.

## B ADDITIONAL ANALYSIS OF REPRESENTATION DYNAMICS

## B.1 UPDATE MAGNITUDE ANALYSIS

We report layer-wise update statistics, evaluate foreground recall using visual saliency scores $u _ { i }$ computed from endpoint displacement, and visualize spatial update patterns for both vision encoders.

Cross-image consistency of the sink-dominated stage. To assess whether the sink-dominated stage generalizes beyond the visualized examples, we compute two layer-wise statistics on the COCO 2017 validation set (4,952 valid images): image-mean foreground recall when selecting the top 30% of tokens by $v _ { i } ^ { \ell } ,$ and the concentration ratio max<sub>i</sub> $v _ { i } ^ { \ell } /$ median<sub>i</sub> $v _ { i } ^ { \ell } .$ Figure 3 reports these quantities for LLaVA-1.5; Figure 6 reports the same diagnostics on LLaVA-NeXT and Qwen2.5- VL, with thresholds 5 and 10, respectively. A ratio above the dashed threshold indicates that the largest updates are isolated to a few tokens rather than distributed across the image, matching the spatial pattern of the sink-dominated stage.

On CLIP-ViT-L (Radford et al., 2021), the mean ratio exceeds the threshold only at Layers 11 and 12 and remains near 2 at other depths. Foreground recall at these layers is close to that of random selection, consistent with isolated sink updates rather than contiguous foreground change. On Qwen2.5-VL, the same isolated-token spike is confined to Layers 15–17. A much smaller rise appears near the encoder output and does not reproduce the intermediate concentration pattern. The sharpness of these image-mean peaks implies a shared sink-dominated depth within each encoder: if the peak layer varied widely across images, the average ratio would be smeared. We use the peak concentration layer $\ell _ { \star } ,$ identified in Appendix C.1, as a fixed reference for window placement within each encoder. The transition occurs around these reference layers, with some variation across images. In both encoders, the subsequent rise in foreground recall motivates us to explore this late foreground-enhanced stage for constructing visual saliency scores.

Foreground recall of fixed-width windows. We fix the window width to five updates and scan its start on the same COCO 2017 validation images for both encoders, using COCO instance annotations (Lin et al., 2014). For each start $\ell _ { s }$ , we rank eligible patches by the visual saliency score $u _ { i } = \| h _ { i } ^ { \ell _ { e } } - h _ { i } ^ { \ell _ { s } } \|$ , with $\ell _ { e } = \ell _ { s } + 5$ , and select the top 30%. For image $q ,$ let $\mathcal { T } _ { q , \ell _ { s } }$ be the selected patch set, $P _ { q , i }$ the pixel region of patch i, and $M _ { q , j }$ the mask of the j-th evaluated instance. The recall for this instance is

$$
R _ { q , j } ( \ell _ { s } ) = \frac { \sum _ { i \in \mathcal { T } _ { q , \ell _ { s } } } \vert P _ { q , i } \cap M _ { q , j } \vert } { \vert M _ { q , j } \vert } .\tag{10}
$$

We first average instance recalls within each image, then average the resulting image scores over the 4,952 images. Thus, each image contributes equally to the reported metric, and instances within an image receive equal weight.

![](images/9555b07fb60884bc9f3b5f6bea6d032b98d597c6845247bdb728d08165b5ae32.jpg)  
(a) LLaVA-NeXT

![](images/461a7112a4bbcb1bb2074549139be6bac3edceb75ba1cfd897f07ed82a938914.jpg)  
(b) Qwen2.5-VL  
Figure 6: Cross-image sink signature on COCO 2017 validation images on LLaVA-NeXT and Qwen2.5-VL. Top: image-mean layer-wise foreground recall when selecting the top 30% of tokens by $v _ { i } ^ { \ell } .$ Bottom: mean concentration ratio max<sub>i</sub> $v _ { i } ^ { \ell } / \ l$ median<sub>i</sub> $v _ { i } ^ { \ell } ;$ ; the dashed line marks the threshold (5 on LLaVA-NeXT and 10 on Qwen2.5-VL). A ratio spike at a fixed intermediate depth indicates isolated tokens with large update magnitudes, consistent with the sink-dominated stage.

![](images/d7f7b1f3f28e02d4d8fca6a0e1c4933144e90ba4d954d8219a8db8ad8754b9a5.jpg)  
Figure 7: Image-mean Foreground Recall@30% for windows of width $w = 5$ on COCO 2017 validation images. At each input-state start $\ell _ { s } ,$ patches are ranked by $\| h _ { i } ^ { \ell _ { s } + 5 } - h _ { i } ^ { \ell _ { s } } \| .$ . Horizontal dashed lines show the random selection expectation (approximately 30%); vertical dotted lines mark the default starts used in the main experiments.

CLIP images are padded to a square and resized to $3 3 6 \times 3 3 6$ , with [CLS] and padding tokens excluded from selection. For Qwen2.5-VL, we apply standard dynamic resizing and evaluate recall on 14 × 14 image patches before spatial merging. We undo the token permutation used for window attention to map each patch score back to its image location for comparison with the instance masks. Sink filtering is disabled in this diagnostic. Masks are resized with nearest-neighbor interpolation. Figure 7 reports the mean foreground recall across images when the top 30% of patches are selected by the visual saliency score $u _ { i } .$

All plotted starts follow the input-state convention of Section 3, so the default windows appear at $\ell _ { s } = 1 4$ and $\ell _ { s } \ = \ 1 9$ . On LLaVA-1.5, the default 14 → 19 window attains the highest imagemean recall, 48.1%. On Qwen2.5-VL, the default 19 → 24 window reaches 46.4%, well above the random baseline of about 30%, while the shallow $2  7$ window reaches 49.2%. The default Qwen2.5-VL window yields higher downstream RelAcc. than $2  7$ in Table 16. Foreground recall alone therefore does not determine which window works best in the full pruning pipeline, which also uses semantic grouping and query relevance. The scan evaluates window-level foreground coverage; the separate sink analysis above provides the reference for its placement.

Layer-wise update visualizations. Figures 8–12 present additional layer-wise visualizations of token update magnitudes $v _ { i } ^ { \ell }$ . Figure 8 shows the tennis scene for LLaVA-1.5 and Mini-Gemini, since both models use the same CLIP-ViT-L encoder (Radford et al., 2021). Figures 10 and 11 show the kite scene for LLaVA-NeXT and Qwen2.5-VL. Figures 9 and 12 show the tennis scene for LLaVA-NeXT and Qwen2.5-VL. Each panel maps update magnitudes to their corresponding spatial positions in the input image; a label $\ell \stackrel { - } { \to } \ell + 1$ denotes the update through Layer ℓ. These examples illustrate how the spatial distribution of updates varies across encoder depth for different images and encoders.

![](images/6e8b11e3e989833a1e25f5b403acf5833022ddba79ead82cfa13537542ef5809.jpg)  
Figure 8: Spatial maps of update magnitudes $v _ { i } ^ { \ell }$ across LLaVA-1.5 and Mini-Gemini vision encoder layers for the tennis scene.

![](images/87b8ed30a59a68a0c76fa05ec5598e10009e60f6eb92814405da2c81c43fd7fa.jpg)  
Figure 9: Spatial maps of update magnitudes $v _ { i } ^ { \ell }$ across LLaVA-NeXT vision encoder layers for the tennis scene.

![](images/f788247c5a1a44d34167415b39c6a77e0ed76354def969300e9e823e72b15755.jpg)  
Figure 10: Spatial maps of update magnitudes v<sup>ℓ</sup> across LLaVA-NeXT vision encoder layers for the kite scene.

![](images/148a3ba4ad00262c3a56470086224d4ca2c05e2db747c51551968821e20f1927.jpg)  
Figure 11: Spatial maps of update magnitudes v<sup>ℓ</sup><sub>i</sub> across Qwen2.5-VL vision encoder layers for the kite scene.

![](images/32dbc879daadcb91d55588a4c82dd010fc82aeb4f3bc64b45680f5350049007b.jpg)  
Figure 12: Spatial maps of update magnitudes $v _ { i } ^ { \ell }$ across Qwen2.5-VL vision encoder layers for the tennis scene.

## B.2 UPDATE DIRECTION ANALYSIS

We compare update directions with encoder output features by semantic token retrieval on the COCO 2017 validation set (Lin et al., 2014), using COCO-Stuff labels (Caesar et al., 2018). For each anchor, other labeled tokens in the same image are ranked by cosine similarity, and retrieval of the anchor class is scored.

![](images/3ebec6a8c3db754fed34d69c873d610820652df691b8b7dd281a835a1427344a.jpg)  
(a) LLaVA-1.5

![](images/011242b23fe93e9bf8c34a7452b206ad1c1ce7444ee3e0887599881c4bd68d9f.jpg)  
(b) Qwen2.5-VL

Figure 13: Precision versus recall for semantic token retrieval with interior anchors on (a) LLaVA-1.5 and (b) Qwen2.5-VL. The AP values shown in the legend are averaged over anchors.  
![](images/99925a53e5d2377acf0960a82164a7200e94d359c9ff05c9bf475d9fa3db0314.jpg)  
(a) LLaVA-1.5  
output features

![](images/b6df054a540e0297fc315870eff7415c2ae8edfa270bb02d6d70dfadae608fed.jpg)  
(b) LLaVA-1.5  
update directions

![](images/6cbe4d447646707932cba66d3bc4fb9efd1618c4dab1badfa7cede494d7c7af4.jpg)  
(c) Qwen2.5-VL  
output features

![](images/f63197ff7f537cbc66123efd1695859177eaca40f127e450dbc800286cfb96ce.jpg)  
(d) Qwen2.5-VL update directions  
Figure 14: Cosine similarity distributions for same-class (blue) and different-class (gray) pairs on LLaVA-1.5 (a, b) and Qwen2.5-VL (c, d). Within each model, the left panel uses output features and the right panel uses update directions. Panels (a) and (b) correspond to Figure 1.

Protocol. Images are encoded at 336×336 with the CLIP-ViT-L/14 encoder in $\mathrm { L L a V A – 1 } . 5 ( 2 4 \times 2 4$ tokens) (Liu et al., 2024a; Radford et al., 2021). A token is assigned to class c $\mathrm { a t } \geq 5 0 \%$ patch coverage and marked interior $\mathrm { a t } \geq 7 0 \%$ We keep image–class pairs with at least eight labeled tokens and select as the anchor the interior token that maximizes boundary distance times coverage, or the highest-coverage labeled token if none is interior. Same-class tokens are positives, differentclass tokens are negatives, and unlabeled tokens are excluded. This yields 25,158 anchors on 4,952 images for LLaVA-1.5 and 23,855 anchors on 4,949 images for Qwen2.5-VL (Bai et al., 2025).

Features and metrics. Let $h _ { i } ^ { o }$ denote the encoder output feature of token i: the final-layer feature for Qwen2.5-VL and the penultimate-layer feature for the CLIP ViT-L/14 encoder in LLaVA-1.5. We compare $h _ { i } ^ { o }$ with unit update directions $d _ { i }$ (Equation 1; $\ell _ { 0 } = 2 , \ell _ { d } = L - 1 )$ using cosine similarity and the same candidate tokens. Average precision (AP) is computed per anchor and then averaged over anchors, images, and classes; the image mean is the primary metric, with standard deviations over the same units.

Results. On LLaVA-1.5 (Table 7), update directions raise mean AP over images from 0.543 to 0.649, with gains on 4,702/4,952 images and 162/171 classes (Table 8); stuff classes improve more than thing classes (0.114 vs. 0.043). Mean AP over anchors rises from 0.508 to 0.613 on LLaVA-1.5 and from 0.458 to 0.572 on Qwen2.5-VL (Figure 13). Figure 15 shows fewer retrieved tokens from different classes when using update directions.

Table 7: Semantic token retrieval AP on the COCO 2017 validation set for LLaVA-1.5. Entries report mean AP over anchors, images, or classes; standard deviations are also reported for image and class means. The mean over images is the primary metric; ∆ is direction minus output. Values are rounded to three decimal places.
<table><tr><td>Averaging</td><td>Output (AP ↑)</td><td>Direction (AP ↑)</td><td>∆</td></tr><tr><td>Mean over anchors</td><td>0.508</td><td>0.613</td><td>0.105</td></tr><tr><td>Mean over images</td><td>0.543±0.136</td><td>0.649±0.130</td><td>0.106</td></tr><tr><td>Mean over classes</td><td>0.532±0.108</td><td>0.612±0.094</td><td>0.081</td></tr></table>

![](images/76daac326bc4f099e87e1ec590bb22f66bfd4fe463dc37e1258e559c79ab452a.jpg)  
Figure 15: Semantic token retrieval examples. From left to right: the input image; the class mask with anchor a; candidates $P _ { a }$ (same class) and $N _ { a }$ (different classes); and top-ranked tokens by cosine similarity to a using output features and update directions. The last two panels each show k tokens, where k is the number of interior tokens in the class mask. “Interior $\boldsymbol { a } ^ { \prime \prime }$ indicates at least 70% coverage of the anchor patch by its class. Candidates exclude the anchor and unlabeled tokens. Red boxes mark a; blue and gray denote same-class and different-class tokens, respectively.

Table 8: AP for each class on the COCO 2017 validation set: thing classes (top) and stuff classes (bottom). n is the number of anchors for the class, with one anchor per eligible image. Out. and Dir. denote mean AP using cosine similarity of output features and update directions; ∆ is Dir. minus Out. Negative values of ∆ are highlighted in red, indicating classes for which update directions retrieve same-class tokens less accurately than output features.  
(a) Thing classes
<table><tr><td>Class</td><td>n</td><td>Out.</td><td>Dir.</td><td>∆ Class</td><td>n</td><td>Out.</td><td>Dir.</td><td>∆</td><td>Class</td><td>n Out.</td><td></td><td>Dir.</td></tr><tr><td>airplane</td><td>72 0.555</td><td></td><td>0.610 +0.055</td><td>dining table</td><td>405</td><td>0.536</td><td>0.619 +0.083</td><td>sandwich</td><td></td><td>61 0.681</td><td>0.718</td><td>+0.038</td></tr><tr><td>apple</td><td></td><td>350.615 0.650 +0.035</td><td></td><td>dog</td><td></td><td></td><td>115 0.702 0.746+0.044</td><td>scissors</td><td></td><td></td><td>140.574 0.593+0.019</td><td></td></tr><tr><td>backpack</td><td></td><td>30 0.694 0.718 +0.025</td><td></td><td>donut</td><td></td><td></td><td>40 0.738 0.745 +0.008</td><td>sheep</td><td></td><td></td><td>52 0.656 0.701 +0.045</td><td></td></tr><tr><td>banana</td><td></td><td>63 0.652 0.732 +0.080</td><td></td><td>elephant</td><td></td><td></td><td>86 0.735 0.795 +0.060</td><td>sink</td><td></td><td></td><td>87 0.472 0.539 +0.067</td><td></td></tr><tr><td>baseball bat</td><td></td><td>7 0.618 0.605-0.014</td><td></td><td>fire hydrant</td><td></td><td></td><td>49 0.620 0.679 +0.059</td><td></td><td>skateboard</td><td></td><td>32 0.584 0.610 +0.025</td><td></td></tr><tr><td>baseball glove</td><td></td><td>7 0.762 0.777 +0.014</td><td></td><td>fork</td><td></td><td></td><td>27 0.511 0.533 +0.022</td><td>skis</td><td></td><td></td><td>10 0.352 0.368 +0.017</td><td></td></tr><tr><td>bear</td><td></td><td>440.6640.739+0.075</td><td></td><td>frisbee</td><td></td><td></td><td>140.5480.658+0.110</td><td></td><td>snowboard</td><td></td><td>130.3430.445+0.102</td><td></td></tr><tr><td>bed</td><td>139 0.625 0.692 +0.067</td><td></td><td></td><td>giraffe</td><td></td><td></td><td>940.677 0.720 +0.043</td><td>spoon</td><td></td><td></td><td>19 0.394 0.417 +0.023</td><td></td></tr><tr><td>bench</td><td>1250.4810.556+0.075</td><td></td><td></td><td>hair drier</td><td></td><td></td><td>3 0.601 0.670 +0.069</td><td></td><td>sports ball</td><td></td><td>6 0.676 0.739 +0.063</td><td></td></tr><tr><td>bicycle</td><td>610.5670.578+0.012</td><td></td><td></td><td>handbag</td><td></td><td></td><td>470.5280.566+0.038</td><td></td><td>stop sign</td><td></td><td>32 0.793 0.899 +0.107</td><td></td></tr><tr><td>bird</td><td>56 0.712 0.742 +0.030</td><td></td><td></td><td>horse</td><td></td><td></td><td>99 0.627 0.711 +0.084</td><td>suitcase</td><td></td><td></td><td>59 0.588 0.680 +0.093</td><td></td></tr><tr><td>boat</td><td>77 0.605 0.633 +0.028 74 0.568 0.612 +0.044</td><td></td><td></td><td>hot dog</td><td></td><td></td><td>31 0.688 0.718 +0.030 70 0.652 0.715 +0.063</td><td></td><td>surfboard</td><td></td><td>46 0.493 0.609 +0.116</td><td></td></tr><tr><td>book</td><td>96 0.618 0.587-0.030</td><td></td><td></td><td>keyboard kite</td><td></td><td></td><td></td><td></td><td>teddy bear</td><td></td><td>74 0.674 0.705 +0.031</td><td></td></tr><tr><td>bottle bowl</td><td>1480.5610.595+0.035</td><td></td><td></td><td>knife</td><td></td><td>15 0.571 0.537-0.034</td><td>30 0.648 0.698 +0.050</td><td>tie</td><td>tennis racket</td><td></td><td>350.4750.502+0.027</td><td></td></tr><tr><td></td><td>42 0.604 0.666 +0.062</td><td></td><td></td><td>laptop</td><td></td><td>127 0.540 0.596 +0.057</td><td></td><td></td><td></td><td></td><td>9 0.502 0.562 +0.060</td><td></td></tr><tr><td>broccoli bus</td><td>151 0.630 0.621 -0.010</td><td></td><td></td><td>microwave</td><td></td><td>24 0.535 0.565 +0.030</td><td></td><td>toaster toilet</td><td></td><td></td><td>3 0.734 0.735 +0.002</td><td></td></tr><tr><td>cake</td><td>850.6340.662+0.027</td><td></td><td></td><td>motorcycle</td><td></td><td>117 0.723 0.728 +0.005</td><td></td><td></td><td>toothbrush</td><td></td><td>123 0.485 0.557 +0.072</td><td></td></tr><tr><td></td><td>202 0.668 0.658-0.010</td><td></td><td></td><td>mouse</td><td></td><td>13 0.638 0.697 +0.060</td><td></td><td></td><td>traffic light</td><td></td><td>40.5980.584-0.014</td><td></td></tr><tr><td>car</td><td>41 0.695 0.761 +0.065</td><td></td><td></td><td>orange</td><td></td><td>48 0.619 0.716 +0.097</td><td></td><td>train</td><td></td><td></td><td>33 0.595 0.639 +0.044</td><td></td></tr><tr><td>carrot</td><td>152 0.701 0.773+0.072</td><td></td><td></td><td>oven</td><td></td><td>94 0.539 0.572 +0.033</td><td></td><td>truck</td><td></td><td>138 0.656 0.663 +0.007</td><td>154 0.663 0.687 +0.024</td><td></td></tr><tr><td>cat</td><td>33 0.650 0.645-0.005</td><td></td><td></td><td>parking meter</td><td></td><td>26 0.552 0.631+0.080</td><td></td><td>tv</td><td></td><td>149 0.607 0.646+0.038</td><td></td><td></td></tr><tr><td>cell phone</td><td>331 0.451 0.509 +0.059</td><td></td><td></td><td>person</td><td></td><td>20370.6040.626+0.021</td><td></td><td></td><td>umbrella</td><td>116 0.605 0.712 +0.108</td><td></td><td></td></tr><tr><td>chair clock</td><td></td><td>52 0.625 0.656 +0.032</td><td></td><td>pizza</td><td></td><td>105 0.710 0.752 +0.042</td><td></td><td></td><td>vase</td><td>43 0.567 0.579 +0.013</td><td></td><td></td></tr><tr><td>couch</td><td>169 0.522 0.613 +0.091</td><td></td><td></td><td>potted plant</td><td></td><td>79 0.568 0.620 +0.052</td><td></td><td></td><td>wine glass</td><td>36 0.412 0.425 +0.012</td><td></td><td></td></tr><tr><td>COW</td><td></td><td>69 0.671 0.741 +0.070</td><td></td><td>refrigerator</td><td></td><td>910.4640.554 +0.089</td><td></td><td></td><td>zebra</td><td>780.7450.791+0.046</td><td></td><td></td></tr><tr><td>cup</td><td>139 0.576 0.588+0.012</td><td></td><td></td><td>remote</td><td></td><td>17 0.663 0.652-0.012</td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr></table>

(b) Stuff classes
<table><tr><td>Class</td><td>n</td><td>Out.</td><td>Dir.</td><td>∆ Class</td><td></td><td>n Out.</td><td>Dir.</td><td></td><td>△ Class</td><td>n</td><td>Out.</td><td>Dir.</td><td>∆</td></tr><tr><td>banner</td><td>60 0.563</td><td></td><td>0.648</td><td>+0.085</td><td>furniture-other</td><td>539 0.402</td><td>0.475</td><td>+0.073</td><td>salad</td><td></td><td>11 0.713</td><td>0.702</td><td>-0.010</td></tr><tr><td>blanket</td><td></td><td>46 0.451 0.559 +0.108</td><td></td><td></td><td>grass</td><td>734 0.503 0.682 +0.179</td><td></td><td></td><td>sand</td><td></td><td></td><td>183 0.498 0.678 +0.180</td><td></td></tr><tr><td>branch</td><td></td><td>46 0.468 0.555 +0.087</td><td></td><td></td><td>gravel</td><td>84 0.358 0.485 +0.127</td><td></td><td></td><td>sea</td><td></td><td></td><td>261 0.671 0.744 +0.073</td><td></td></tr><tr><td>bridge</td><td></td><td>54 0.474 0.551+0.077</td><td></td><td></td><td>ground-other</td><td>113 0.422 0.589 +0.167</td><td></td><td></td><td>shelf</td><td></td><td></td><td>155 0.367 0.447 +0.080</td><td></td></tr><tr><td>building-other 744 0.578 0.619 +0.041</td><td></td><td></td><td></td><td>hill</td><td></td><td>71 0.508 0.643 +0.135</td><td></td><td></td><td>sky-other</td><td></td><td></td><td>1080 0.474 0.801 +0.327</td><td></td></tr><tr><td>bush</td><td>274 0.451 0.564 +0.113</td><td></td><td></td><td></td><td>house</td><td>185 0.539 0.581 +0.042</td><td></td><td></td><td>skyscraper</td><td></td><td></td><td>59 0.577 0.638 +0.061</td><td></td></tr><tr><td>cabinet</td><td></td><td>254 0.412 0.534 +0.123</td><td></td><td></td><td>leaves</td><td>79 0.550 0.630 +0.080</td><td></td><td></td><td>snow</td><td></td><td></td><td>174 0.651 0.795 +0.144</td><td></td></tr><tr><td>cage</td><td></td><td></td><td>870.455 0.554 +0.099</td><td>light</td><td></td><td>59 0.459 0.551 +0.092</td><td></td><td></td><td>solid-other</td><td></td><td></td><td>19 0.476 0.570 +0.094</td><td></td></tr><tr><td>cardboard</td><td>105 0.538 0.607 +0.069</td><td></td><td></td><td>mat</td><td></td><td>150.3640.534 +0.170</td><td></td><td></td><td>stairs</td><td></td><td></td><td>400.450 0.546+0.096</td><td></td></tr><tr><td>carpet</td><td>128 0.390 0.552 +0.162</td><td>320 0.416 0.576 +0.160</td><td></td><td>metal</td><td>mirror-stuff</td><td>582 0.412 0.483 +0.071 102 0.365 0.396 +0.031</td><td></td><td></td><td>stone</td><td></td><td></td><td>34 0.507 0.590 +0.082</td><td></td></tr><tr><td>ceiling-other</td><td></td><td>9 0.430 0.626 +0.196</td><td></td><td></td><td>moss</td><td>7 0.436 0.578 +0.142</td><td></td><td></td><td>straw structural-other</td><td></td><td></td><td>480.443 0.577+0.134</td><td></td></tr><tr><td>ceiling-tile</td><td></td><td>30 0.503 0.578 +0.076</td><td></td><td></td><td>mountain</td><td>143 0.511 0.621 +0.110</td><td></td><td></td><td>table</td><td></td><td></td><td>90 0.433 0.518 +0.085</td><td></td></tr><tr><td>cloth clothes</td><td>191 0.445 0.502 +0.056</td><td></td><td></td><td>mud</td><td></td><td>11 0.522 0.635+0.113</td><td></td><td></td><td>tent</td><td></td><td></td><td>383 0.416 0.533 +0.117</td><td></td></tr><tr><td>clouds</td><td>322 0.537 0.799 +0.262</td><td></td><td></td><td></td><td>napkin</td><td>32 0.534 0.608 +0.074</td><td></td><td></td><td>textile-other</td><td></td><td></td><td>37 0.476 0.538 +0.062</td><td></td></tr><tr><td>counter</td><td>95 0.332 0.427 +0.095</td><td></td><td></td><td>net</td><td></td><td>39 0.373 0.453 +0.080</td><td></td><td></td><td>towel</td><td></td><td></td><td>200 0.491 0.578 +0.087</td><td></td></tr><tr><td>cupboard</td><td></td><td>16 0.357 0.533 +0.176</td><td></td><td></td><td>paper</td><td>196 0.501 0.560 +0.060</td><td></td><td></td><td>tree</td><td></td><td></td><td>39 0.592 0.610 +0.017</td><td></td></tr><tr><td>curtain</td><td>170 0.491 0.600 +0.110</td><td></td><td></td><td></td><td>pavement</td><td>641 0.402 0.548 +0.146</td><td></td><td></td><td>vegetable</td><td></td><td>49 0.561 0.624 +0.062</td><td>1277 0.554 0.666 +0.113</td><td></td></tr><tr><td>desk-stuff</td><td>114 0.414 0.524 +0.110</td><td></td><td></td><td></td><td>pillow</td><td>13 0.569 0.717 +0.149</td><td></td><td></td><td>wall-brick</td><td></td><td>137 0.463 0.584 +0.121</td><td></td><td></td></tr><tr><td>dirt</td><td>321 0.425 0.573 +0.148</td><td></td><td></td><td></td><td>plant-other</td><td>166 0.488 0.586 +0.098</td><td></td><td></td><td>wall-concrete</td><td></td><td>1086 0.437 0.616 +0.179</td><td></td><td></td></tr><tr><td>door-stuff</td><td>246 0.3740.504 +0.130</td><td></td><td></td><td></td><td>plastic</td><td>265 0.468 0.512 +0.044</td><td></td><td></td><td>wall-other</td><td></td><td>451 0.414 0.558 +0.144</td><td></td><td></td></tr><tr><td>fence</td><td>250 0.482 0.573 +0.091</td><td></td><td></td><td></td><td>platform</td><td>75 0.390 0.548 +0.158</td><td></td><td></td><td>wall-panel</td><td></td><td>127 0.419 0.586 +0.167</td><td></td><td></td></tr><tr><td>floor-marble</td><td>41 0.356 0.521 +0.165</td><td></td><td></td><td></td><td>playingfield</td><td>217 0.608 0.809 +0.201</td><td></td><td></td><td>wall-stone</td><td></td><td>59 0.520 0.599 +0.079</td><td></td><td></td></tr><tr><td>floor-other</td><td>253 0.345 0.522 +0.176</td><td></td><td></td><td></td><td>railing</td><td>31 0.460 0.530 +0.070</td><td></td><td></td><td>wall-tile</td><td></td><td>194 0.460 0.566 +0.106</td><td></td><td></td></tr><tr><td>floor-stone</td><td>51 0.430 0.585 +0.155</td><td></td><td></td><td></td><td>railroad</td><td>100 0.414 0.491 +0.077</td><td></td><td></td><td>wall-wood</td><td></td><td>2250.4060.551+0.144</td><td></td><td></td></tr><tr><td>floor-tile</td><td>2580.3950.521+0.126</td><td></td><td></td><td></td><td>river</td><td>79 0.570 0.708 +0.138</td><td></td><td></td><td>water-other</td><td></td><td>65 0.518 0.622 +0.104</td><td></td><td></td></tr><tr><td>floor-wood</td><td>193 0.422 0.579 +0.157</td><td></td><td></td><td></td><td>road</td><td>5380.416 0.584 +0.168</td><td></td><td></td><td>waterdrops</td><td></td><td>3 0.237 0.270 +0.033</td><td></td><td></td></tr><tr><td>flower</td><td>50 0.5750.606 +0.031</td><td></td><td></td><td></td><td>rock</td><td>81 0.547 0.635 +0.087</td><td></td><td></td><td>window-blind</td><td></td><td>73 0.462 0.625 +0.163</td><td></td><td></td></tr><tr><td>fog</td><td>102 0.452 0.784 +0.332</td><td></td><td></td><td></td><td>roof</td><td>76 0.468 0.580 +0.112</td><td></td><td></td><td>window-other</td><td></td><td>331 0.499 0.600 +0.101</td><td></td><td></td></tr><tr><td>food-other</td><td>130 0.587 0.608 +0.021</td><td></td><td></td><td>rug</td><td></td><td>66 0.460 0.591 +0.131</td><td></td><td></td><td>wood</td><td></td><td>97 0.460 0.553 +0.093</td><td></td><td></td></tr><tr><td>fruit</td><td></td><td>41 0.502 0.541 +0.039</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr></table>

## C ADDITIONAL EXPERIMENTAL RESULTS

This section describes sink detection, compression settings, and evaluation metrics, followed by results on larger models and component ablations.

## C.1 SINK DETECTION CONFIGURATION

For each encoder, we locate the peak of the image-averaged max-to-median update magnitude ratio:

$$
\bar { r } ^ { \ell } = \frac { 1 } { | \mathcal { D } | } \sum _ { I \in \mathcal { D } } \frac { \operatorname* { m a x } _ { i } v _ { i } ^ { \ell } ( I ) } { \operatorname* { m e d i a n } _ { i } v _ { i } ^ { \ell } ( I ) } , \qquad \ell _ { \mathrm { p e a k } } = \arg \operatorname* { m a x } _ { \ell } \bar { r } ^ { \ell } ,\tag{11}
$$

where $\mathcal { D }$ contains the analysis images and $v _ { i } ^ { \ell } ( I ) = \| h _ { i } ^ { \ell + 1 } ( I ) - h _ { i } ^ { \ell } ( I ) \|$ . We set $\ell _ { \star } : = \ell _ { \mathrm { p e a k } }$ and inspect $h _ { i } ^ { \ell _ { \mathrm { p e a k } } + 1 }$ , the output of the peak layer.

Motivated by the observation that sparse activation coordinates can drive high-norm outlier tokens (Jiang et al., 2025), we examine the activation entropy of the ten highest-norm tokens at this state and identify a diagnostic coordinate $d _ { s }$ from the concentrated activations of low-entropy candidates. We use $d _ { s } = 6 5 0$ for CLIP-ViT-L and $d _ { s } = 8 4 9$ for the Qwen2.5-VL vision encoder, with an empirical threshold $\tau = 5 0$ . At inference, tokens with absolute activation above τ are excluded:

$$
\mathcal { V } = \left\{ i : \left| \left( h _ { i } ^ { \ell _ { \star } + 1 } \right) _ { d _ { s } } \right| \leq \tau \right\} .\tag{12}
$$

These detection settings remain fixed across datasets, images, and token budgets; norm ranking and entropy analysis are performed only when configuring the encoder.

## C.2 EXPERIMENTAL SETTINGS AND EVALUATION METRICS

Compression settings. We form $\Delta { h } _ { i }$ as in Equation 1, normalize it to unit length, and group the resulting directions with spherical k-means (Hornik et al., 2012). Centroids are initialized by deterministic farthest-point sampling as in Equation 5, with the initialization seed fixed $\mathrm { ~ t o ~ } s = 0$ in all experiments. Unless otherwise stated, each grouping run stops when the assignment labels remain unchanged, the relative change in grouping loss is below $1 0 ^ { - 5 }$ , or 10 iterations have been completed. We use $K = 2 0$ for all models in the main experiments and evaluate the effect of the group count in Appendix C.6. The default grouping endpoints are $\ell _ { 0 } = 2$ and $\ell _ { d } = L - 1$ . The saliency windows are $1 4  1 9$ for CLIP-ViT-L and $1 9  2 4$ for the Qwen2.5-VL vision encoder, following the layer convention in Section 4.2. For a given encoder, these settings are held fixed across datasets, images, and token budgets. Unless otherwise stated, all experiments are run on a single NVIDIA RTX PRO 6000-96 GB GPU.

Qwen2.5-VL adaptation and budget convention. For Qwen2.5-VL (Bai et al., 2025), update directions and visual saliency scores are computed separately on the original encoder patches before aggregation. These signals are then aggregated over each set of four patches combined by the spatial merger and aligned with the corresponding visual token passed to the LLM. If any of these four patches is identified as a sink, the corresponding visual token is excluded from the candidate set. The visual token count N and target budget B refer to tokens after merging. The reported retention ratio $B / N$ uses the visual token count before sink removal as its denominator.

Mini-Gemini adaptation. For Mini-Gemini (Li et al., 2025), visual tokens come from the lowresolution CLIP branch and follow the same settings as in LLaVA-1.5.

Evaluation metrics. We use LMMs-Eval (Zhang et al., 2025a) for downstream evaluation. RelAcc. is the mean relative benchmark score: each score is divided by the corresponding uncompressed score, and the resulting ratios are averaged across the benchmarks in the table and expressed as a percentage.

## C.3 EVALUATION ON LARGER MODELS

We extend the evaluation from the 7B models in the main text to LLaVA-1.5-13B (Liu et al., 2024a), LLaVA-NeXT-13B (Liu et al., 2024b), and Qwen2.5-VL-32B (Bai et al., 2025). These larger models share the vision encoder architectures of their 7B counterparts, allowing us to assess the method at increased scale within the same model families.

LLaVA-1.5-13B encodes each image into 576 visual tokens. Following VisionZip (Yang et al., 2025) and PruneSID (Fang et al., 2026), we retain 192, 128, and 64 tokens. In Table 9, MSDG-Prune achieves the highest RelAcc. at all three budgets, with scores of 98.3%, 97.4%, and 96.3%. The gains over the stronger baseline at each budget are 0.6, 0.1, and 1.3 percentage points, respectively. LLaVA-NeXT-13B produces up to 2,880 visual tokens; we retain 640, 320, and 160 tokens, matching the budgets used for the 7B model. Table 10 compares MSDG-Prune with PruneSID. Our method exceeds PruneSID by 1.0 and 0.3 percentage points at 640 and 160 tokens and matches it at 94.0% at 320 tokens.

For Qwen2.5-VL-32B, we report retention ratios because the visual token count depends on the input, consistent with Table 4. As reported in Table 11, MSDG-Prune achieves RelAcc. of 96.6%, 94.7%, and 89.8% at retention ratios of 33.3%, 22.2%, and 11.1%, respectively. Its advantage over the stronger baseline at each ratio is 0.4, 2.3, and 2.8 percentage points, increasing as retention decreases. Together, these results support the generalization of MSDG-Prune to larger models across different multimodal architectures.

Table 9: Performance comparison on LLaVA-1.5-13B. RelAcc. averages the relative scores across the nine listed benchmarks.
<table><tr><td colspan="17">Method GQA MMB MME POPE SQA  $\mathrm { V Q A } ^ { \mathrm { v 2 } }$   $\mathrm { V Q A } ^ { \mathrm { T e x t } }$  SEED VizWiz RelAcc.</td></tr><tr><td colspan="11">Uncompressed baseline (100%) 1818 85.9 72.8 80.0 61.3</td><td colspan="4">56.6</td></tr><tr><td colspan="11">Vanilla 63.2 67.7</td><td colspan="4">66.9</td></tr><tr><td colspan="11">Token retention: 33.3%</td><td colspan="4"></td></tr><tr><td colspan="11"></td><td colspan="4"></td></tr><tr><td>VisionZip</td><td>59.6 59.6</td><td>65.9 65.9</td><td>1770</td><td></td><td>86.4</td><td>72.8</td><td>78.0</td><td>58.6</td><td></td><td>65.2</td><td>54.9</td><td>97.5%</td></tr><tr><td colspan="4">PruneSID Ours 60.1</td><td>1770</td><td>86.4 86.1</td><td>72.8</td><td></td><td>78.0</td><td>58.6</td><td>65.2</td><td>56.0</td><td>97.7%</td></tr><tr><td colspan="4">67.4 22.2%</td><td>1801</td><td></td><td>73.7</td><td>76.8</td><td></td><td>59.4</td><td>65.3</td><td>56.0</td><td>98.3%</td></tr><tr><td colspan="4">Token retention: 57.9</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td colspan="4">VisionZip</td><td>1743</td><td>85.2</td><td>74.0</td><td></td><td>76.8</td><td>58.7</td><td>63.8</td><td>55.0</td><td>96.8%</td></tr><tr><td colspan="4">PruneSID 58.9 Ours</td><td>1811</td><td>85.9 86.6</td><td>73.1</td><td></td><td>76.7 75.9</td><td>57.5</td><td>64.1</td><td>56.8</td><td>97.3%</td></tr><tr><td colspan="4">59.2</td><td>1783</td><td></td><td>73.3</td><td></td><td></td><td>58.8</td><td>64.2</td><td>56.1</td><td>97.4%</td></tr><tr><td colspan="4">Token retention:</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>VisionZip</td><td>56.2</td><td>11.1% 64.9</td><td>1676</td><td></td><td>76.0</td><td>74.4</td><td>73.7</td><td>57.4</td><td></td><td>60.4</td><td>55.9</td><td>93.6%</td></tr><tr><td>PruneSID</td><td>57.8</td><td>63.8</td><td>1711</td><td></td><td>82.0</td><td>71.8</td><td>75.2</td><td>56.3</td><td></td><td>62.8</td><td>57.3</td><td>95.0%</td></tr><tr><td>Ours</td><td>58.4</td><td>64.6</td><td>1792</td><td></td><td>85.3</td><td>72.4</td><td>74.1</td><td></td><td>57.5</td><td>62.4</td><td>57.5</td><td>96.3%</td></tr></table>

Table 10: Performance comparison on LLaVA-NeXT-13B.
<table><tr><td>Method</td><td>GQA MMB</td><td></td><td>MME</td><td>POPE</td><td>SQA</td><td> $\mathrm { \Delta V Q A } ^ { \mathrm { v 2 } }$ </td><td> $\mathrm { V Q A } ^ { \mathrm { T e x t } }$ </td><td>SEED</td><td></td><td>VizWiz RelAcc.</td></tr><tr><td colspan="11">Uncompressed baseline (100%)</td></tr><tr><td>Vanilla</td><td>65.4</td><td>70.0</td><td>1858</td><td>86.2</td><td>73.5</td><td>81.8</td><td>64.3</td><td>71.9</td><td>64.0</td><td>100%</td></tr><tr><td>Token retention: 22.2%</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>PruneSID</td><td>62.4</td><td>67.0</td><td>1817</td><td>85.6</td><td>70.1</td><td>79.1</td><td>60.2</td><td>68.5</td><td>60.2</td><td>95.9%</td></tr><tr><td>Ours</td><td>63.6</td><td>66.6</td><td>1820</td><td>87.0</td><td>72.8</td><td>78.6</td><td>62.0</td><td>68.5</td><td>60.1</td><td>96.9%</td></tr><tr><td>Token retention:</td><td></td><td>11.1%</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>PruneSID</td><td>61.5</td><td>65.4</td><td>1810</td><td>82.7</td><td>70.3</td><td>76.9</td><td>58.5</td><td>66.8</td><td>58.5</td><td>94.0%</td></tr><tr><td>Ours</td><td>61.3</td><td>64.5</td><td>1753</td><td>86.9</td><td>73.1</td><td>75.6</td><td>60.1</td><td>64.9</td><td>57.0</td><td>94.0%</td></tr><tr><td colspan="9">Token retention: 5.6%</td><td></td><td></td></tr><tr><td>PruneSID</td><td>59.5</td><td>65.5</td><td>1715</td><td>77.8</td><td>69.1</td><td>73.6</td><td>56.7</td><td>64.1</td><td>56.7</td><td>90.8%</td></tr><tr><td>Ours</td><td>58.5</td><td>63.3</td><td>1712</td><td>85.7</td><td>72.5</td><td>72.9</td><td>56.3</td><td>62.2</td><td>55.8</td><td>91.1%</td></tr></table>

## C.4 GROUPING, SALIENCY SCORING, AND UPDATE AGGREGATION ON LLAVA-1.5

Table 12 compares three alternatives on LLaVA-1.5-7B, changing one component at a time. Grouping. Grouping by update directions $d _ { i }$ improves RelAcc. over encoder output features $h _ { i } ^ { o }$ by 0.7, 0.9, and 0.7 percentage points at the three retention ratios. Together with the Qwen2.5- VL results in Table 6, these consistent downstream gains support update directions as a grouping representation.

Saliency scoring. The visual saliency score $u _ { i } .$ , computed from endpoint displacement, outperforms output-layer [CLS] attention by 1.4, 1.5, and 2.2 percentage points in RelAcc., supporting update magnitudes for visual saliency estimation.

Update aggregation. Accumulated update magnitude matches or exceeds endpoint displacement by up to 0.3 points on LLaVA-1.5, while endpoint displacement performs better on Qwen2.5-VL (Table 6). Endpoint displacement computes each score from two endpoint states with a single difference and norm, avoiding calculations over every intermediate transition. These results and the

Table 11: Performance comparison on Qwen2.5-VL-32B.
<table><tr><td>Method</td><td>GQA</td><td>MMB</td><td>MME</td><td>POPE</td><td>SQA</td><td>MMMU</td><td> $\mathrm { V Q A } ^ { \mathrm { T e x t } }$ </td><td>RelAcc.</td></tr><tr><td colspan="7">Uncompressed baseline (100%)</td><td></td><td></td></tr><tr><td>Vanilla</td><td>59.3</td><td>86.3</td><td>2428</td><td>84.3</td><td>91.5</td><td>61.1</td><td>76.5</td><td>100%</td></tr><tr><td colspan="7">Token retention: 33.3%</td><td></td><td></td></tr><tr><td>VisionZip</td><td>57.3</td><td>85.1</td><td>2307</td><td>82.1</td><td>88.9</td><td>57.7</td><td>72.1</td><td>96.2%</td></tr><tr><td>PruneSID</td><td>57.0</td><td>81.6</td><td>2209</td><td>82.7</td><td>85.4</td><td>57.6</td><td>70.6</td><td>94.2%</td></tr><tr><td>Ours</td><td>56.4</td><td>84.4</td><td>2396</td><td>81.6</td><td>89.1</td><td>58.3</td><td>72.9</td><td>96.6%</td></tr><tr><td colspan="7">Token retention: 22.2%</td><td></td><td></td></tr><tr><td>VisionZip</td><td>55.6</td><td>82.3</td><td>2173</td><td>79.8</td><td>87.0</td><td>55.3</td><td>67.0</td><td>92.4%</td></tr><tr><td>PruneSID</td><td>55.9</td><td>80.8</td><td>2158</td><td>82.1</td><td>83.6</td><td>55.8</td><td>66.9</td><td>92.0%</td></tr><tr><td>Ours</td><td>55.2</td><td>81.4</td><td>2356</td><td>80.1</td><td>87.5</td><td>58.3</td><td>70.5</td><td>94.7%</td></tr><tr><td colspan="7">Token retention: 11.1%</td><td></td><td></td></tr><tr><td>VisionZip</td><td>51.8</td><td>77.4</td><td>1862</td><td>71.4</td><td>82.5</td><td>56.4</td><td>57.8</td><td>85.2%</td></tr><tr><td>PruneSID</td><td>52.6</td><td>73.9</td><td>2076</td><td>75.0</td><td>81.3</td><td>55.8</td><td>61.3</td><td>87.0%</td></tr><tr><td>Ours</td><td>52.5</td><td>77.3</td><td>2175</td><td>78.3</td><td>82.5</td><td>56.1</td><td>65.7</td><td>89.8%</td></tr></table>

reduced computation and state-access requirements support endpoint displacement as the shared default.

Table 12: Grouping, saliency scoring, and update aggregation on LLaVA-1.5-7B. Ours uses unit update directions and endpoint displacement. Output features replaces the grouping representation with $h _ { i } ^ { o }$ . Attention replaces the visual saliency score u with [CLS] attention from the output layer. Accumulated replaces endpoint displacement with accumulated update magnitude.
<table><tr><td rowspan="2">Method</td><td colspan="5">Retain 33.3%</td><td colspan="5">Retain 22.2%</td><td colspan="5">Retain 11.1%</td></tr><tr><td>GQA</td><td>MME</td><td>POPE</td><td>SQA</td><td></td><td>RelAcc.</td><td>GQA MME</td><td>POPE</td><td>SQA</td><td>RelAcc.</td><td>GQA</td><td>MME</td><td>POPE</td><td></td><td>SQA RelAcc.</td></tr><tr><td>Ours</td><td>60.1</td><td>1806</td><td>86.9</td><td>68.7</td><td>98.5%</td><td></td><td>58.9</td><td>1778</td><td>86.7 68.7</td><td></td><td>97.6%</td><td>57.7</td><td>1708</td><td>85.6</td><td>68.4 95.8%</td></tr><tr><td>Grouping Output features</td><td>60.0</td><td>1771</td><td>86.7</td><td>68.4</td><td>97.8%</td><td></td><td>58.9 1712</td><td></td><td>86.6 68.7</td><td>96.7%</td><td>57.8</td><td>1680</td><td>84.4</td><td>68.4</td><td>95.1%</td></tr><tr><td>Saliency scoring</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Attention</td><td>58.7</td><td>1795</td><td>84.3</td><td>68.7</td><td>97.1%</td><td></td><td>57.9</td><td>1775</td><td>83.3 68.4</td><td>96.1%</td><td>56.4</td><td>1675</td><td></td><td>81.2 68.7</td><td>93.6%</td></tr><tr><td>Update aggregation</td><td></td><td>1800</td><td>87.0</td><td>69.1</td><td>98.6%</td><td></td><td>59.1 1770</td><td></td><td>87.0 68.4</td><td>97.6%</td><td>57.8</td><td>1719</td><td></td><td>68.0</td><td>96.1%</td></tr></table>

## C.5 GROUPING ENDPOINTS

We vary the early endpoint $\ell _ { 0 }$ and the late endpoint $\ell _ { d }$ used to compute update directions. Table 13 varies $\ell _ { 0 }$ with $\ell _ { d } = L - 1$ , while Table 14 varies $\ell _ { d }$ with $\ell _ { 0 } = 2$ . Here, $h ^ { \dot { L } - 1 }$ is the output of Layer $L - 2 .$ . Using the encoder input $( \ell _ { 0 } = 0 )$ or final output $( \ell _ { d } = L )$ reduces mean RelAcc. by 0.5 and 0.7 percentage points relative to the default pair $( 2 , L - 1 )$ . In contrast, varying $\ell _ { 0 }$ within {1, 2, 3} or $\ell _ { d }$ within $\{ L - 3 , L - 2 , L - 1 \}$ changes mean RelAcc. by at most 0.3 points. These results support the default endpoints and show limited sensitivity to nearby choices.

Table 13: Ablation of the early endpoint $\ell _ { 0 }$ on Qwen2.5-VL-7B. Avg. is the mean RelAcc. over the three retention ratios.
<table><tr><td></td><td colspan="6">Retain 33.3%</td><td colspan="6">Retain 22.2%</td><td colspan="7">Retain 11.1%</td><td></td></tr><tr><td> $\ell _ { 0 }$ </td><td>GQA MMB MME POPE SQA RelAcc. GQA MMB MME POPE SQA RelAcc. GQA MMB MME POPE SQA RelAcc. Avg.</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>0</td><td>58.1</td><td>81.7</td><td>2323</td><td>84.9</td><td>88.0</td><td>98.2%</td><td>57.0</td><td>80.2</td><td>2300</td><td>83.7</td><td>87.1</td><td>96.8%</td><td>54.5</td><td>77.6</td><td>2165</td><td>81.3</td><td>86.4</td><td></td><td>93.4%</td><td>96.1%</td></tr><tr><td>1</td><td>58.2</td><td>82.0</td><td>2318</td><td>84.7</td><td>87.7</td><td>98.1%</td><td>57.3</td><td>80.8</td><td>2290</td><td>84.1</td><td>87.3</td><td>97.1%</td><td>54.8</td><td>77.7</td><td></td><td>2212</td><td>81.4</td><td>86.3</td><td>94.0%</td><td>96.4%</td></tr><tr><td>2</td><td>58.5</td><td>82.6</td><td>2332</td><td>84.7</td><td>87.8</td><td>98.5%</td><td>57.7</td><td>81.2</td><td>2297</td><td>84.0</td><td>87.3</td><td>97.3%</td><td>55.4</td><td>78.4</td><td>2178</td><td></td><td>81.3</td><td>86.3</td><td>94.0% 96.6%</td><td></td></tr><tr><td>3</td><td>58.2</td><td>82.0</td><td>2329</td><td>85.0</td><td>87.8</td><td>98.3%</td><td>57.5</td><td>80.7</td><td>2294</td><td>84.6</td><td></td><td>87.0 97.1%</td><td>55.4</td><td>77.1</td><td>2161</td><td></td><td>81.3</td><td>86.4 93.6% 96.3%</td><td></td><td></td></tr></table>

Table 14: Ablation of the late endpoint $\ell _ { d }$ on Qwen2.5-VL-7B.
<table><tr><td></td><td colspan="6">Retain 33.3%</td><td colspan="6">Retain 22.2%</td><td colspan="6">Retain 11.1%</td><td></td></tr><tr><td> $\ell _ { d }$ </td><td colspan="6">GQA MMB MME POPE SQA RelAcc.</td><td colspan="6"></td><td colspan="6">GQA MMB MME POPE SQA RelAcc. GQA MMB MME POPE SQA RelAcc. Avg.</td><td></td></tr><tr><td> $L$ </td><td>58.1</td><td>82.2</td><td>2340</td><td>84.3</td><td>87.8</td><td>98.2%</td><td>56.8</td><td>81.8</td><td>2276</td><td>82.5</td><td>87.5</td><td>96.7%</td><td>53.9</td><td>78.1</td><td>2143</td><td>79.7</td><td>86.3</td><td></td><td></td></tr><tr><td>L-1</td><td>58.5</td><td>82.6</td><td>2332</td><td>84.7</td><td>87.8</td><td>98.5%</td><td>57.7</td><td>81.2</td><td>2297</td><td>84.0</td><td>87.3</td><td>97.3%</td><td>55.4</td><td>78.4</td><td>2178</td><td>81.3</td><td>86.3</td><td>92.8% 94.0%</td><td>95.9% 96.6%</td></tr><tr><td>L-2</td><td>58.4</td><td>82.6</td><td>2328</td><td>85.3</td><td>87.9</td><td>98.6%</td><td>57.5</td><td>81.1</td><td>2283</td><td>84.7</td><td>87.3</td><td>97.2%</td><td>55.3</td><td>78.2</td><td>2192</td><td>81.4</td><td>85.9</td><td>93.9%</td><td>96.6%</td></tr><tr><td>L-3</td><td>58.4</td><td>82.0</td><td>2340</td><td>85.2</td><td>87.7</td><td>98.5%</td><td>57.3</td><td>80.6</td><td>2295</td><td>84.3</td><td>87.2</td><td>97.1%</td><td>55.0</td><td>77.3</td><td>2145</td><td>81.6</td><td>86.0</td><td>93.3%</td><td>96.3%</td></tr></table>

## C.6 NUMBER OF GROUPS K

The group count K controls the granularity of the partition used for budget allocation. In Table 15, mean RelAcc. increases from 94.7% to 96.6% as K grows from 8 through 12 and 16 to 20. Increasing K further to 24 or 28 leaves mean performance at 96.6% or 96.5%, respectively, indicating saturation around $K = 2 0$ . This trend, together with the highest RelAcc. at 11.1% retention, supports $K = 2 0$ as the default group count.

Table 15: Ablation of the group count K on Qwen2.5-VL.
<table><tr><td></td><td colspan="6">Retain 33.3%</td><td colspan="6">Retain 22.2%</td><td colspan="6">Retain 11.1%</td><td></td></tr><tr><td>K</td><td>GQA MMB MME POPE SQA RelAcc.</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td>GQA MMB MME POPE SQA RelAcc.</td><td></td><td></td><td></td><td>GQA MMB MME POPE SQA RelAcc.</td><td></td><td></td><td></td><td></td><td></td><td>Avg.</td></tr><tr><td>8</td><td>57.6</td><td>81.2</td><td>2276</td><td>83.3</td><td>87.6</td><td>97.0%</td><td>56.2</td><td>79.8</td><td>2263</td><td>81.8</td><td>86.7</td><td>95.5%</td><td>53.7</td><td>76.7</td><td>2105</td><td>78.8</td><td>85.3</td><td>91.6%</td><td>94.7%</td></tr><tr><td>12</td><td>57.8</td><td>80.8</td><td>2286</td><td>83.6</td><td>87.7</td><td>97.1%</td><td>56.2</td><td>79.7</td><td>2249</td><td>82.7</td><td>86.8</td><td>95.6%</td><td>54.4</td><td>75.8</td><td>2131</td><td>80.0</td><td>85.6</td><td>92.2%</td><td>95.0%</td></tr><tr><td>16</td><td>58.0</td><td>81.3</td><td>2302</td><td>84.4</td><td>87.8</td><td>97.7%</td><td>56.8</td><td>79.7</td><td>2230</td><td>82.5</td><td>87.2</td><td>95.7%</td><td>54.6</td><td>76.6</td><td>2130</td><td>80.8</td><td>85.7</td><td>92.6%</td><td>95.3%</td></tr><tr><td>20</td><td>58.5</td><td>82.6</td><td>2332</td><td>84.7</td><td>87.8</td><td>98.5%</td><td>57.7</td><td>81.2</td><td>2297</td><td>84.0</td><td>87.3</td><td>97.3%</td><td>55.4</td><td>78.4</td><td>2178</td><td>81.3</td><td>86.3</td><td>94.0%</td><td>96.6%</td></tr><tr><td>24</td><td>58.5</td><td>81.9</td><td>2361</td><td>85.3</td><td>87.8</td><td>98.7%</td><td>57.5</td><td>81.1</td><td>2311</td><td>83.8</td><td>87.3</td><td>97.3%</td><td>55.3</td><td>78.0</td><td>2170</td><td>81.4</td><td>86.3</td><td>93.8%</td><td>96.6%</td></tr><tr><td>28</td><td>58.8</td><td>82.3</td><td>2331</td><td>85.0</td><td>87.9</td><td>98.6%</td><td>57.4</td><td>80.3</td><td>2298</td><td>84.3</td><td>87.3</td><td>97.1%</td><td>55.5</td><td>77.9</td><td>2174</td><td>81.4</td><td>86.5</td><td>93.9%</td><td>96.5%</td></tr></table>

## C.7 SALIENCY WINDOW ENDPOINTS

Guided by the layer-wise stage analysis, we compare a small set of representative windows at different depths, including intervals that overlap or cross the sink-dominated stage. We vary only the saliency window endpoints, with sink filtering enabled and all other components fixed. An entry a→b measures the displacement between $h ^ { a }$ and $h ^ { b }$ over $b - a$ layer updates. Most windows cover five updates; 15→18 on Qwen2.5-VL and $1 2 {  } 1 8$ on LLaVA-1.5 additionally probe the sink-dominated stage with different widths. The default windows follow the stage-based rule in Section 4.2.

Tables 16 and 17 report the results; RelAcc. is the mean relative score across GQA, MME, POPE, and SQA. On both models, the default windows in the late foreground-enhanced stage outperform shallow windows and attain the highest RelAcc. among the tested configurations at both retention ratios. Using the final five encoder updates instead reduces RelAcc. by 3.7 and 5.8 percentage points on Qwen2.5-VL and by 2.0 and 2.2 points on LLaVA-1.5 at 22.2% and 11.1% retention, respectively. The gains at the default windows and the decline near the encoder output show the depth dependence of visual saliency estimation and support the window placement for each encoder.

## C.8 SALIENCY WINDOW WIDTH

To test sensitivity to the empirical five-layer width, we fix the start at Layer 19 on Qwen2.5-VL and vary the window width $w = \ell _ { e } - \ell _ { s } ,$ keeping all other components unchanged. The default $w = 5$ corresponds to 19→24; w = 3 and $w = 7$ give 19→22 and 19→26. Table 18 reports results at 22.2% and 11.1% retention. The default width attains the highest RelAcc. at 22.2% retention. At 11.1% retention, $w = 3$ is 0.2 points higher than the default, while $w = 7$ is 0.4 points lower. Across $w \in \{ 3 , 5 , 7 \}$ , the RelAcc. range is 0.4 and 0.6 percentage points at the two retention ratios, respectively. These small variations indicate limited sensitivity to window width over the tested range. We use $w = 5$ as a shared empirical default throughout our experiments.

## C.9 SIGNAL ASSIGNMENT

The full method uses the query-weighted saliency score $s _ { i } = \alpha _ { i } u _ { i }$ for both budget allocation across groups and token ranking within each group. Table 19 compares two variants: (i) the query relevance score $\alpha _ { i }$ for allocation and the visual saliency score $u _ { i }$ for ranking, and (ii) $u _ { i }$ for allocation and $\alpha _ { i }$ for ranking. Grouping and the remaining pipeline are held fixed. Using query relevance for allocation and visual saliency for ranking exceeds the reverse assignment by 2.5 and 2.7 percentage points and remains within 0.1 point of the full method. These results support using query relevance for budget allocation across groups and visual saliency for token selection within each group.

Table 16: Saliency windows on Qwen2.5-VL-7B.
<table><tr><td>Endpoints GQA MME</td></tr><tr><td>POPE SQA RelAcc. Uncompressed baseline (100%)</td></tr><tr><td>Vanilla 60.9 2310 86.3 88.9 100%</td></tr><tr><td>Token retention:  $2 2 . 2 \%$   $_ { 0  5 }$  57.2 2269 84.0 87.3 96.9%</td></tr><tr><td> $^ { 2 \to 7 }$  57.2 2273 84.2 87.4 97.1%</td></tr><tr><td> $_ { 7 \to 1 2 }$  57.1 2268 84.0 87.3 96.9%</td></tr><tr><td> $1 2 \to 1 7$  57.7 2269 83.9 87.3 97.1% 57.5 2290 84.0 97.2%</td></tr><tr><td> $1 5 \to 1 8$  87.1  $1 5 \to 2 0$  57.7 2291 84.0 87.2 97.3%</td></tr><tr><td> $\mathrm { D e f a u l t 1 9 } {  } 2 4 $  57.7 2297 84.0 87.3 97.4%</td></tr><tr><td> $2 4  2 9$  57.2 2272 84.0 87.3 97.0%</td></tr><tr><td> $2 7  3 2$  54.1 2245 79.6 86.0 93.7%</td></tr><tr><td></td></tr><tr><td>Token retention: 11.1%</td></tr><tr><td>0→5 54.3 2134 80.9 86.3 93.1%</td></tr><tr><td>2→7 54.7 2159 80.9 86.6 93.6%</td></tr><tr><td> $_ { 7 \to 1 2 }$  54.6 2172 81.1 86.7 93.8%</td></tr><tr><td> $_ { 1 2  1 7 }$  54.9 2138 81.2 86.3 93.5%</td></tr><tr><td> $1 5 \to 1 8$  55.0 2123 81.5 86.2 93.4%</td></tr><tr><td> $1 5 \to 2 0$  55.3 2150 81.5 86.3 93.8%</td></tr><tr><td> $\mathrm { D e f a u l t 1 9 } {  } 2 4 $  55.4 2178 81.3 86.3 94.1%</td></tr><tr><td></td></tr><tr><td> $2 4  2 9$  54.5 2207 81.3 86.1 94.0%</td></tr><tr><td> $2 7  3 2$  51.3 2018 74.2 85.1 88.3%</td></tr></table>

Table 17: Saliency windows on LLaVA-1.5-7B.
<table><tr><td>Endpoints GQA MME</td></tr><tr><td>POPE SQA RelAcc. Uncompressed baseline (100%) Vanilla 61.9 1862 85.9 69.5 100%</td></tr><tr><td>Token retention:  $2 2 . 2 \%$ </td></tr><tr><td> $_ { 0  5 }$  58.4 1691 86.5 68.1 96.0%</td></tr><tr><td> $^ { 2 \to 7 }$  58.4 1690 86.3 68.5 96.0%</td></tr><tr><td> $_ { 7 \to 1 2 }$  58.5 1744 86.0 68.3 96.6%</td></tr><tr><td> $1 2 \to 1 7$  59.2 1762 86.8 68.5 97.5%</td></tr><tr><td> $1 2 \to 1 8$  59.3 1752 86.7 68.5 97.3%</td></tr><tr><td> $\mathrm { D e f a u l t } ~ 1 4 \substack {  } 1 9$  58.9 1778 86.7 68.7 97.6%</td></tr><tr><td> $1 9  2 4$  57.5 1710 85.7 68.1 95.6%</td></tr><tr><td>Token retention: 11.1%</td></tr><tr><td> $_ { 0  5 }$  56.6 1687 85.6 67.9 94.8%</td></tr><tr><td> $^ { 2 \to 7 }$  56.6 1655 85.0 68.8 94.6%</td></tr><tr><td> $7  \mathrm { i } 2$  56.5 1647 85.8 68.0 94.4%</td></tr><tr><td> $1 2 \to 1 7$  57.7 1689 85.5 68.6 95.5%</td></tr><tr><td> $1 2 \to 1 8$  57.6 1687 85.4 68.0 95.2%</td></tr><tr><td> $\mathrm { D e f a u l t } ~ 1 4 \substack {  } 1 9$  57.7 1708 85.6 68.4 95.8%</td></tr><tr><td> $1 9  2 4$  54.9 1658 84.5 68.2 93.6%</td></tr></table>

Table 18: Saliency window width ablation on Qwen2.5-VL-7B with the start fixed at Layer 19 and all other components unchanged. An interval $a {  } b$ covers updates from Layer a through Layer $b - 1 ,$ so varying b changes the width while preserving the starting layer.
<table><tr><td>Window</td><td>GQA MME</td><td>POPE</td><td>SQA</td><td>RelAcc.</td></tr><tr><td>Uncompressed baseline (100%) Vanilla 60.9</td><td>2310</td><td>86.3</td><td>88.9</td><td>100%</td></tr><tr><td>Token retention: 22.2%</td><td></td><td></td><td></td><td></td></tr><tr><td>19→22 (w=3)</td><td>57.8</td><td>2252</td><td>84.2 87.1</td><td>97.0%</td></tr><tr><td> $1 9 \to 2 4 ( w = 5 )$ </td><td>57.7</td><td>2297</td><td>84.0 87.3</td><td>97.4%</td></tr><tr><td> $1 9 \substack {  } 2 6 ( w = 7 )$ </td><td>57.6</td><td>2273</td><td>84.0 87.4</td><td>97.2%</td></tr><tr><td>Token retention: 11.1%</td><td></td><td></td><td></td><td></td></tr><tr><td> $1 9 \substack {  } 2 2 ( w = 3 )$ </td><td>55.6</td><td>2182</td><td>81.3 86.6</td><td>94.3%</td></tr><tr><td> $1 9 \to 2 4 ( w = 5 )$ </td><td>55.4</td><td>2178</td><td>81.3</td><td>86.3 94.1%</td></tr><tr><td> $1 9 \substack {  } 2 6 ( w = 7 )$ </td><td>55.2</td><td>2147</td><td>81.0</td><td>86.4 93.7%</td></tr></table>

Table 19: Ablation of signal assignment on Qwen2.5-VL-7B. Allocation and ranking refer to budget allocation across groups and token selection within each group, respectively.
<table><tr><td>Method</td><td>GQA</td><td>MME</td><td>POPE</td><td>SQA</td><td>RelAcc.</td></tr><tr><td colspan="6">Token retention: 22.2%</td></tr><tr><td>α Alloc., u Rank</td><td>57.4</td><td>2307</td><td>84.2</td><td>87.1</td><td>97.4%</td></tr><tr><td>u Alloc., α Rank</td><td>55.0</td><td>2264</td><td>81.1</td><td>86.6</td><td>94.9%</td></tr><tr><td>Ours</td><td>57.7</td><td>2297</td><td>84.0</td><td>87.3</td><td>97.4%</td></tr><tr><td colspan="6">Token retention: 11.1%</td></tr><tr><td>α Alloc., u Rank</td><td>55.0</td><td>2166</td><td>81.5</td><td>86.5</td><td>94.0%</td></tr><tr><td>u Alloc., α Rank</td><td>52.7</td><td>2113</td><td>78.0</td><td>86.1</td><td>91.3%</td></tr><tr><td>Ours</td><td>55.4</td><td>2178</td><td>81.3</td><td>86.3</td><td>94.1%</td></tr></table>

## D DETAILS OF EVALUATION BENCHMARKS

We summarize the tasks covered by the evaluation benchmarks and define their notation in the result tables.

GQA (Hudson & Manning, 2019). GQA evaluates compositional visual reasoning through questions about objects, attributes, and spatial relations. Its questions are constructed from Visual Genome scene graphs (Krishna et al., 2017).

MMBench (Liu et al., 2024c). MMBench evaluates visual perception and reasoning through questions with multiple answer choices in English and Chinese, covering 20 ability dimensions. We use the English split and denote it as MMB in the tables.

MME (Fu et al., 2025). MME evaluates MLLMs across 14 perception and cognition tasks, including object recognition, spatial reasoning, text recognition, and knowledge reasoning. In our tables, MME refers to the full perception and cognition suite.

POPE (Li et al., 2023b). POPE assesses object hallucination through questions about whether specified objects are present in an image. We use the configuration based on MS COCO images (Lin et al., 2014). We report the average F1 score over the adversarial, random, and popular settings.

ScienceQA (Lu et al., 2022). ScienceQA evaluates reasoning over questions spanning natural science, social science, and language science. Questions may include image or text context. We denote the benchmark as SQA in the tables.

VQAv2 (Goyal et al., 2017). VQAv2 is a visual question answering benchmark containing 204,721 images from MS COCO (Lin et al., 2014) and 1,105,904 questions. It includes complementary image pairs with different answers to reduce reliance on language priors. Each question is associated with ten human answers. We denote the benchmark as $\mathrm { V Q } \mathrm { \bar { A } } ^ { \mathrm { v } 2 }$ in the tables.

TextVQA (Singh et al., 2019). TextVQA evaluates reading and reasoning over text in natural images. It contains 45,336 questions associated with 28,408 images from the Open Images dataset (Krasin et al., 2017). We denote the benchmark as $\mathrm { V Q A } ^ { \mathrm { T e x t } }$ in the tables.

MMMU (Yue et al., 2024). MMMU evaluates multimodal knowledge and reasoning at the college level across disciplines such as art, business, science, and engineering.

SEED-Bench (Li et al., 2023a). SEED-Bench evaluates visual understanding through questions with multiple answer choices and human annotations. It includes tasks based on static images and videos; SEED<sup>I</sup> denotes the image split used in all our result tables.

VizWiz (Gurari et al., 2018). VizWiz contains photographs and questions collected from visually impaired users. It includes images of varying quality and questions that cannot always be answered from the available visual content.

HR-Bench 8K (Wang et al., 2025). HR-Bench 8K is the subset of HR-Bench constructed from 8K images in DIV8K (Gu et al., 2019). It evaluates the ability to recover fine visual details from images at 8K resolution. We denote the benchmark as $\mathrm { H R B ^ { 8 K } }$ in the tables.

## E QUALITATIVE ANALYSIS OF RETAINED TOKENS

We examine the spatial distribution of retained tokens to better understand the behavior of MSDG-Prune under compression. We first analyze two cases where pruning changes a correct answer into an incorrect one, then compare token selection across MSDG-Prune, VisionZip, and PruneSID.

## E.1 FAILURE CASES

Figure 16 shows two GQA examples on Qwen2.5-VL-7B at a target token retention ratio of 11.1%. The uncompressed model answers both questions correctly, while MSDG-Prune fails after pruning. In the kitchen example, the bananas occupy a small region beneath the microwave and receive limited coverage from the retained tokens. The pruned model answers “none” instead of “bananas,” illustrating a failure to identify a small object required by the query.

In the tennis example, the pruned model incorrectly answers that the shorts and shoes have the same color. Although parts of both objects remain represented by retained patches, the model fails to distinguish their colors. These cases highlight the difficulty of preserving sufficient evidence for small-object recognition and attribute comparison within a limited token budget, even when selection incorporates query relevance.

Q: What fruits are beneath the microwave?  
![](images/9344ed445b8495a8b4a7d8d3f7ba678dd42c48265c18ea8ec5b11863373271b0.jpg)  
GT: Bananas

![](images/fc8daa3b1c060bec15c462aa9ef12d133eb230e4452740976366b04ba0774fc8.jpg)  
Pred: None

Q: Do the shorts and the shoes have the same color?  
![](images/871e3a4ed9241acab1ba8df61e1ad41327447cf5e21bf7a1ba1c648077027f52.jpg)  
GT: No

![](images/7388f8ead169f99da2894656de9fdd4ae565924a80f868d993ab43e1b6d385a0.jpg)  
Pred: Yes

Figure 16: Failure cases of MSDG-Prune on Qwen2.5-VL-7B at 11.1% target token retention. Each case pairs the original image (left) with the retained-token visualization (right). GT denotes the ground-truth answer and Pred denotes the answer after pruning; the uncompressed model answers both questions correctly. Discarded patches are covered by a translucent gray mask.

## E.2 COMPARISON OF RETAINED TOKENS

Figure 17 compares the tokens retained by VisionZip, PruneSID, and MSDG-Prune for the same images and queries. The examples cover object presence, text reading, spatial relations, and flowchart reasoning at 11.1% token retention. For the laptop and flowchart queries, MSDG-Prune retains patches over portions of the screen content and program structure. For the spatial query, retained patches cover both the lamp and computer regions. These examples illustrate the coverage of queried content across natural images and diagrams, complementing the quantitative benchmark results.

![](images/572fcb111af1ace4bcdeee9101638c0325ac0ef73ff14d6d3c7595b2db49b535.jpg)  
Figure 17: Qualitative comparison at a retention ratio of 11.1%. Columns show the original image and the tokens retained by VisionZip, PruneSID, and MSDG-Prune, with the corresponding query below each example. Discarded patches are covered by a translucent gray mask.
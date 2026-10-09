# V-CoLA: Vision Token Compression with Linear Attention

Hao Jiang<sup>1,\*</sup> Yiru Mao<sup>1,\*</sup>

Tianpeng Bu<sup>1</sup> Hao Zhou<sup>1</sup> Hongtao Duan<sup>1</sup> Wang Jing<sup>1</sup> Bowen Xu<sup>1</sup> Xin Chen<sup>1</sup> Lulu Hu<sup>1</sup> Bin Yang<sup>1</sup> Yongliang Tao<sup>1,†</sup> Minying Zhang<sup>1,†</sup>

<sup>1</sup>Alibaba Cloud Computing, Alibaba Group

<sup>\*</sup>Equal contribution. <sup>†</sup>Corresponding author.

suanshi.tyl@alibaba-inc.com minying.zmy@alibaba-inc.com

## Abstract

Vision-language models (VLMs) have demonstrated impressive capabilities but suffer from substantial computational overhead, as vision tokens dominate the input sequence. This motivates vision token compression as a key direction to alleviate the burden. However, with the emergence of hybrid architectures incorporating linear attention (e.g., Qwen3.5), prior methods designed for softmax attention struggle to generalize. Our analysis reveals that both attention- and similarity-based approaches suffer notable performance degradation, underscoring the urgent need for compression methods tailored to this regime. To this end, we propose V-CoLA, an efficient training-free token compression framework specifically designed for linear attention. V-CoLA introduces a novel uniqueness-aware importance criterion for identifying critical vision tokens, coupled with an adaptive token merging strategy that performs compression. All components are optimized at the implementation level to remain compatible with the chunk-wise parallelism of linear attention, ensuring strong practical value. Extensive experiments across multiple benchmarks demonstrate the superiority of V-CoLA: it achieves 99.5% of the original performance with only 50.0% of vision tokens, and over 88.0% with as few as 12.5%, while delivering a 1.86× to 6.15× prefill speedup.

## 1 Introduction

Vision-language models (VLMs) have demonstrated remarkable performance in multi-modal understanding and reasoning (Bai et al., 2025; Singh et al., 2025; Wang et al., 2025b). Nevertheless, it is widely acknowledged that the number of vision tokens far exceeds that of linguistic tokens (Vasu et al., 2025; Chen et al., 2024a), imposing substantial computational and memory burdens. Recent advances in vision token compression (Shao et al., 2025; Tao et al., 2025b; Yang et al., 2025b; Bolya et al., 2022) have significantly improved computational efficiency and inference speed by reducing the number of vision tokens.

![](images/d57c4ce9a73e39ed2f1b22bc1ee99e65b3d8298bf89b47e917c3ebaddd3bc9c5.jpg)  
Figure 1: Comparison of token selection. FastV suffers from attention bias, missing the digits at the top of the image. DART selects pivot tokens from only sparse semantic regions, leading to insufficient awareness of some primary objects. Our method successfully identifies all main regions and provides an accurate response.

However, most of these methods either explicitly rely on softmax-attention scores (Xing et al., 2024; Hu et al., 2025) or are evaluated solely on softmax-attention models (Zhang et al., 2025; Yang et al., 2025a). With the rise of hybrid architectures incorporating linear attention (Qwen Team, 2026; Team et al., 2025; Tao et al., 2025a), the crossarchitecture transfer of these methods has become significantly challenging, calling for new compression algorithms tailored to them.

We conduct systematic experiments and find that both attention- and similarity-based compression methods fail to surpass even a random baseline when moving to hybrid architectures, and produce suboptimal results (Figure 1). Our theoretical analysis attributes this structural failure to an information-theoretic ceiling imposed by the bounded recurrent state of linear attention, which disrupts both the shallow-layer attention concentration exploited by attention-based methods and the token distinguishability required by similaritybased methods. Yet, it also reveals that linear attention’s selective retention itself encodes an intrinsic token-importance signal.

Inspired by this, we explore the intrinsic indicators of token importance inherent in linear attention, and propose Vision token Compression with Linear Attention (V-CoLA), a novel efficient token compression framework compatible with hybrid VLMs incorporated with linear attention. Our method mainly evolves two components: (1) an uniqueness-aware importance criterion that jointly considers each token’s long-term contribution to the final state and its short-term uniqueness with respect to historical context, enabling the indicating of important tokens while avoiding redundancy, (2) an adaptive chunk-wise merging strategy that dynamically adjusts chunk sizes and performs token merging within each chunk, thereby adaptively regulating the local compression ratio and reducing information loss in critical regions. Moreover, we optimize all computations at the implementation level, enabling compatibility with linear attention acceleration algorithms through chunk-wise parallelism.

We conduct comprehensive evaluations using Qwen3.5 (Qwen Team, 2026) against state-of-theart baselines. The experimental results demonstrate that our method significantly outperforms these baselines. Even when preserving 50% of the tokens after the first layer, V-CoLA achieves 99.5% of the average performance; when preserving only 12.5% of the tokens, it still preserves over 88.0% of the average performance. Benefiting from the implementation-level optimizations in our codebase, V-CoLA achieves a runtime speedup of 1.86× to 6.15×, while the additional runtime overhead associated with pruning accounts for only 1.6% of the forward-pass cost.

In summary, our main contributions are:

• We systematically analyze the structural failure of existing vision token compression methods under hybrid architectures with linear attention.

• We propose V-CoLA, a novel vision token compression framework compatible to linear attention, which achieves superior performance over state-of-the-art baselines.

• We further optimize V-CoLA for full compatibility with the chunk-wise parallelism of linear attention, achieving significant inference acceleration with marginal computational overhead.

<table><tr><td rowspan=7 colspan=1>1050-5-10-120</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td rowspan=2 colspan=1></td><td rowspan=2 colspan=1></td><td rowspan=2 colspan=1></td><td rowspan=2 colspan=1></td><td rowspan=2 colspan=1></td><td></td><td></td><td></td></tr><tr><td rowspan=1 colspan=1>Outperforms</td><td rowspan=1 colspan=1>Random</td><td rowspan=1 colspan=1>Baseline</td></tr><tr><td rowspan=1 colspan=1>MME</td><td rowspan=1 colspan=1>MMB</td><td rowspan=1 colspan=1>GQA</td><td rowspan=1 colspan=1> $\mathrm { ‰ }$ </td><td rowspan=1 colspan=1>●</td><td rowspan=1 colspan=1>POPE</td><td rowspan=1 colspan=1>è</td><td rowspan=1 colspan=1>MMStar</td></tr><tr><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>2</td><td rowspan=1 colspan=1>2</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>T-VQA</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>VizWiz</td><td rowspan=1 colspan=1></td></tr><tr><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>▲</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td></tr><tr><td rowspan=1 colspan=1>▲</td><td rowspan=1 colspan=1>Fast</td><td rowspan=1 colspan=1>V  ▲ Da</td><td rowspan=1 colspan=1>rt</td><td rowspan=1 colspan=1>Underp</td><td rowspan=1 colspan=1>erforms Random Baseline</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td></tr></table>

Figure 2: FastV (attention-based) vs. DART (similaritybased) vs. random baseline on Qwen3.5-9B.

## 2 Analysis

## 2.1 Preliminary

Softmax Attention. Given a query $\pmb q _ { t }$ , standard softmax attention computes its output by attending over all preceding key-value pairs:

$$
{ \pmb o } _ { t } = \sum _ { i = 1 } ^ { t } \alpha _ { t , i } { \pmb v } _ { i } , \ \alpha _ { t , i } = \frac { \exp ( { \pmb q } _ { t } ^ { \top } { \pmb k } _ { i } ) } { \sum _ { j = 1 } ^ { t } \exp ( { \pmb q } _ { t } ^ { \top } { \pmb k } _ { j } ) } ,\tag{1}
$$

where attention weight $\alpha _ { t , i }$ provides an explicit, token-level measure of $\omega _ { t } { ' } s$ reliance on $v _ { i } .$ While this formulation preserves fine-grained access to every past token, computing the attention map over full sequence incurs quadratic time and memory cost, making it expensive for long contexts.

Linear Attention. To alleviate the quadratic cost of softmax attention, linear-attention variants replace the softmax kernel with a recurrent state $\bar { \boldsymbol { S } } _ { t } \in \mathbb { R } ^ { d \times d }$ that compresses the entire history into a fixed-size memory. Among recent advances, Gated DeltaNet (Yang et al., 2024) has attracted significant attention and been adopted by popular hybrid VLMs such as Qwen3.5 (Qwen Team, 2026) and InfiniteVL (Tao et al., 2025a). It augments the state recurrence with selective writing, erasing, and forgetting operations:

$$
\begin{array} { r } { \mathbf { S } _ { t } = \mathbf { S } _ { t - 1 } \big ( \alpha _ { t } ( \pmb { I } - \beta _ { t } \pmb { k } _ { t } \pmb { k } _ { t } ^ { \top } ) \big ) + \beta _ { t } \pmb { v } _ { t } \pmb { k } _ { t } ^ { \top } , } \end{array}\tag{2}
$$

where ${ \pmb { \alpha } } _ { t } ~ \in ~ ( 0 , 1 )$ is a decay gate, $\beta _ { t } ~ \in ~ ( 0 , 1 )$ controls the strength of state updating, and $k _ { t }$ are ℓ<sub>2</sub>-normalized. The update rule can be interpreted as one step of online gradient descent on a pertoken reconstruction objective:

$$
\begin{array} { r } { \mathscr { L } _ { t } ( { \bf S } ) = \frac { 1 } { 2 } \| { \bf S } { \bf k } _ { t } - { \bf v } _ { t } \| ^ { 2 } , } \end{array}\tag{3}
$$

Under this view, the recurrent state S acts as an associative memory that is continually optimized to map each key $k _ { t }$ to its corresponding value ${ \mathbf { } } v _ { t }$ Retrieval at query time is then carried out implicitly through the matrix-vector product $o _ { t } = \mathbf { S } _ { t } \mathbf { { q } } _ { t } .$

![](images/ebe881f56d5642b2cc4fd646c1dd97a57a504e428ec4a7ce23877fc341b1674a.jpg)  
Figure 3: Average proportion of attention scores allocated to vision tokens across different layers in Qwen3- VL-8B and Qwen3.5-9B.

## 2.2 Failure of Existing Methods

We evaluate the performance of both attention- and similarity-based methods when applied to hybrid architectures across multiple benchmarks. Shown in Figure 2, they surprisingly exhibit virtually no advantage over the random token pruning baseline. To further investigate the failure mechanisms, we perform an in-depth analysis below.

Figure 3 compares the attention distributions of softmax attention layers at various depths in Qwen3-VL and Qwen3.5. The results illustrate that the vision token attention efficiency of Qwen3- VL declines rapidly in the shallow layers, whereas Qwen3.5 continues to attend to visual content in the intermediate layers and only drops to a low level in the deeper layers. This suggests that leveraging solely on shallow-layer attention may be insufficient to identify all important vision tokens in hybrid architectures. Moreover, we evaluate DART’s redundant token selection in Figure 4. The results show that even with fixed pivot tokens, the redundant tokens identified on Qwen3.5 become markedly less reliable. This suggests that the discriminability of latent feature similarity of hybrid architectures has been fundamentally altered.

The failure of these two representative categories of methods highlights the inherent challenge of transferring previous token compression strategies to hybrid architectures. And these findings indicate that effective token compression for hybrid VLMs requires importance signals that go beyond softmax attention maps and feature similarity, and instead exploit cues intrinsic to the hybrid attention design itself.

## 2.3 Information Ceiling of Linear Attention

From the associative-memory view, softmax attention keeps every token independently addressable, whereas Gated DeltaNet compresses the entire history into a single fixed-capacity state $\mathbf { S } _ { t } \in \mathbb { R } ^ { d \times d }$ raising an information-preservation problem.

![](images/62c6b6209b164c6562bf22dd5d7daffec3f0469e484e85edb817f206eb375cb4.jpg)  
Figure 4: Comparison of similarity-based token selection between Qwen3-VL-8B and Qwen3.5-9B.

Let $\mathcal { M } = \{ ( \boldsymbol { k } _ { i } , \boldsymbol { v } _ { i } ) \} _ { i = 1 } ^ { L }$ be drawn from a joint distribution with per-token entropy $h ( \pmb { k } , \pmb { v } )$ . Treating $\mathbf { S } _ { L }$ as a lossy encoder of the sequence and letting c denote the bit-precision of each entry in $\mathbf { S } _ { L }$ , the data processing inequality (Cover, 1999) gives $I \big ( { \mathcal { M } } ; { \mathbf { S } } _ { L } \big ) \ \le \ H ( { \mathbf { S } } _ { L } ) = c \cdot d ^ { 2 } = { \mathcal { O } } ( d ^ { 2 } )$ whereas the source information satisfies $H ( \mathcal { M } ) =$ $L \cdot h ( k , v ) = \mathcal { O } ( L d )$ . Combining the two yields a lower bound on the information lost by the state:

$$
\begin{array} { c } { { H \big ( \mathcal { M } \mid \mathbf { S } _ { L } \big ) = H ( \mathcal { M } ) - I \big ( \mathcal { M } ; \mathbf { S } _ { L } \big ) } } \\ { { \geq H ( \mathcal { M } ) - H ( \mathbf { S } _ { L } ) = \Omega ( L d - d ^ { 2 } ) , } } \end{array}\tag{4}
$$

which grows linearly in L once $L \ > \ d .$ In VLMs, where high-resolution multimodal inputs often yield $L \gg d ,$ this information ceiling becomes particularly pronounced (Arora et al., 2024; Jelassi et al., 2024) and carries direct implications for token compression in hybrid architectures. From another perspective, the bounded state compels the gates in Equation 2 to update selectively, which implies that model’s retention preference during state recurrence inherently encodes token importance (Park et al., 2025), providing an intrinsic signal that we exploit in Section 3.1.

As this qualitative argument relies only on the data-processing inequality and an L-independent finite state capacity, it is independent of the specific update rule and extends to bounded-state recurrent backbones in general (Dao and Gu, 2024; Peng et al., 2023).

## 3 Method

In Section 2, we analyze the failure of transferring typical attention-based and similarity-based methods to identify token importance into hybrid architectures, and the information ceiling of linear attention. Experimental results show different behavioral patterns in the state recurrence and pose challenges for token importance estimation and compression methods. To address this, shown in Figure 5, we propose a novel training-free token compression framework. Specifically, we first uncover the inherent capability of linear attention to indicate token importance in Section 3.1. Building upon this insight, we further introduce a uniqueness-aware token importance in Section 3.2, and develop an adaptive chunk-wise token merging strategy in Section 3.3.

![](images/e6c889599ab70598288074f239caf4d7bf18473da10ed2f8756ed5d612530501.jpg)  
Figure 5: Overview of our method. (left) Original hybrid architectures. (right) Forwarding with V-CoLA, equipped with token-wise importance criterion and adaptive chunk-wise token merging.

## 3.1 Linear Attention as an Intrinsic Indicator

The analysis in Section 2.3 reveals that the update process of the bounded recurrent state inherently encodes token importance, which motivates us to derive it directly from the state recurrence.

Although pioneering work (Park et al., 2025) has attempted to quantify token importance through the "magnitude of state update", such a strategy considers only a single dimension of the recurrence and lacks a measure of long-term information retention. According to Equation 3, the objective of state recurrence is to write the paired key-value information into the state such that it can be fully retrieved. We adopt this reconstruction error as a straightforward criterion, and quantify the retention rate of visual information written into the final state of the model across varying vision token lengths and model depths. We also record the Pearson correlation coefficient of the retention rate rankings between adjacent layers.

As shown in Figure 6, we have several key observations: (1) The retention rate is relatively high in shallow layers and progressively declines with depth, dropping below 20% in the deeper layers. This indicates that visual information is heavily involved in shallow-layer reasoning, while becoming increasingly marginalized in deeper layers. (2) The retention rate also decreases as the sequence length grows, which aligns with the information ceiling discussed in Section 2.3. Given the limited state size, this suggests that the recurrent state is already under pressure in retaining information, forcing the model to preserve tokens in a selective manner. (3) The Pearson correlation of retention rate rankings between adjacent layers remains high in shallow layers and declines in deeper layers alongside the fading of visual information. This implies that during the visual reasoning stage, different layers exhibit consistent preferences in selecting critical tokens.

![](images/79387b0059a68f1eee7647ab311d3054726d39a149ba3d86d642f4d5885283af.jpg)  
Figure 6: Layer-wise retention rate (blue) and importance correlation (red) on Qwen3.5-9B.

The above observations offer important insights: the state updates of linear attention inherently possess a capability for indicating token importance, and we can leverage the selective retention behavior of linear attention in the shallow layers to filter out important or redundant tokens.

## 3.2 Uniqueness-aware Token Importance

As discussed in Section 3.1, the reconstruction objective of linear attention provides a natural lens for measuring token importance: a low reconstruction error indicates that the token’s information is faithfully preserved in the state. Building on this insight, follow-up work (Peng et al., 2025) further posits that state updates ought to account for both current and historical information, proposing a more comprehensive objective:

$$
\mathcal { L } _ { t } ( \mathbf { S } ) = \lambda \cdot \| S \| _ { F } ^ { 2 } + \sum _ { i = 1 } ^ { t } \eta _ { i } \cdot \| \mathbf { S } \pmb { k } _ { i } - \pmb { v } _ { i } \| ^ { 2 } .\tag{5}
$$

This suggests that an important token should be accurately encoded in the state S, while the degree of reconstruction error reflects whether the token has contributed to the final state. Inspired by this, we propose a token importance criterion from the perspective of memory reconstruction:

$$
\mathcal { R } _ { t } ^ { \mathrm { i m p } } = 1 - \psi ( \mathbf { S } _ { L } \pmb { k } _ { t } , \pmb { v } _ { t } ) , \quad t \in [ 1 , L ] ,\tag{6}
$$

where $S _ { L }$ is the final state of the sequence with length $L ,$ and $\psi ( \mathbf { a } , \mathbf { b } ) ~ = ~ \left( 1 - \cos ( \mathbf { a } , \mathbf { b } ) \right) / 2 ~ \in$ [0, 1] denotes the normalized cosine distance.

However, filtering based solely on token-wise importance suffers from a bias toward salient regions, thereby causing the model to select a large number of redundant tokens (Wen et al., 2025). Moreover, focusing on the final state may cause certain important long-term tokens to be overlooked, as their contributions may be attenuated by the accumulated gated decaying in Equation 2, or chooses to write information and pass it directly to the next layer for processing, rather than continuously propagating along the sequence to $\mathbf { S } _ { L }$

To overcome this, we introduce token’s shortterm contribution to the state as a complement to Equation 6. In the recurrent update of the state, each token writes its information into the state and optimize the objective in Equation 3. As a result, when subsequent redundant tokens further update the state, the gradient of the optimization objective decreases, which in turn suppresses the magnitude of their influence on the state. Therefore, the degree of state change can be used to measure redundancy and serve as an indicator of uniqueness. To assess this change more robustly, we use a set of pseudo queries $\bar { Q _ { t } ^ { * } } \in \mathbb { R } ^ { w \times d }$ as probes for querying instead of comparing state after/before updating, and introduce following metric:

$$
\mathcal { R } _ { t } ^ { \mathrm { u n i } } = \psi ( \mathbf { S } _ { t } { \mathbf { Q } _ { t } ^ { \ast } } ^ { \top } , \mathbf { S } _ { t - 1 } { \mathbf { Q } _ { t } ^ { \ast } } ^ { \top } ) , \quad t \in [ 1 , L ] ,\tag{7}
$$

where we choose $Q _ { t } ^ { * } = Q _ { [ t + w _ { 1 } : t + w _ { 2 } ) }$ as the nearest queries from token t, which will be affected in recent state recurrent updates, with $w _ { 1 } \leq 0$ and $w _ { 2 } > 0$ denoting the signed left and right boundary offsets, respectively. Each of the $w = w _ { 2 } - w _ { 1 }$ probes reads one value vector from the state before and after token t is absorbed, and then we average ψ over the probes.

Overall, we define the final importance score in a weighted form:

$$
\mathcal { R } _ { t } = ( 1 - \lambda ) \cdot \mathcal { R } _ { t } ^ { \mathrm { i m p } } + \lambda \cdot \mathcal { R } _ { t } ^ { \mathrm { u n i } } ,\tag{8}
$$

where λ denotes the weight of $\mathcal { R } _ { t } ^ { \mathrm { u n i } }$ . As both terms are ψ-based distances over value-space readouts, $\mathcal { R } _ { t }$ itself remains a normalized measure in [0, 1], consistently interpreted within the value space.

Notably, we integrate the importance computation into the chunk-wise parallelism of linear attention, incurring minimal additional runtime overhead. The details are provided in the Appendix A.

## 3.3 Adaptive Chunk-wise Token Merging

Previous approaches primarily remain tokens with top-k importance scores (Yang et al., 2025a; Zhao et al., 2025; Chen et al., 2024a). Recent analyses (Omri et al., 2025) indicate that, compared with token pruning, token aggregation exhibits more robust performance. Inspired by this, various token aggregation strategies have been proposed, $e . g .$ clustering-based aggregation (Shen et al., 2025; Shao et al., 2025) and bipartite matching aggregation (Wang et al., 2025a). However, these methods do not consider distribution of importance, therefore fail to adaptively allocate compression rate across varying semantic regions.

Based on these, we propose an adaptive token merging strategy that automatically partitions tokens into importance-aligned chunks according to their importance distribution density, and performs token merging within each chunk. Given token sequence $o _ { t } .$ , remain rate $\gamma$ and importance $\mathcal { R } _ { t }$ we partition the token sequence into $L ^ { \prime } = L \times \gamma$ chunks such that the sum of importance scores within each chunk is close. Specifically, we define chunks as $\mathcal { C } _ { i } = o _ { [ t _ { i } : t _ { i + 1 } ] } , 1 = t _ { 1 } < \cdot \cdot \cdot < t _ { L ^ { \prime } } = L ,$ the objective can be formulated as:

$$
\{ \mathcal C _ { i } ^ { * } \} _ { i = 1 } ^ { L ^ { \prime } } = \arg \operatorname* { m i n } _ { \{ \mathcal C _ { i } \} } \sum _ { i = 1 } ^ { L ^ { \prime } } \left| \bar { \mathcal R } - \sum _ { t = t _ { i } } ^ { t _ { i + 1 } } \mathcal R _ { t } \right| ,\tag{9}
$$

where $\begin{array} { r } { \bar { { \mathcal R } } = \frac { 1 } { L ^ { \prime } } \sum _ { t = 1 } ^ { L } { \mathcal R } _ { t } } \end{array}$ is expected importance mass per chunk. In this way, regions with dense importance are assigned finer-grained chunks, while less informative regions are grouped into coarser chunks, yielding an importance-aware partition of the token sequence. Then, we perform token merging independently within each chunk and get compressed token sequence $o _ { i } ^ { \prime }$ by:

$$
o _ { i } ^ { \prime } = \sum _ { t = t _ { i } } ^ { t _ { i + 1 } } w _ { t } o _ { t } , ~ w _ { t } = \frac { \exp ( \mathcal { R } _ { t } / \tau ) } { \sum _ { k = t _ { i } } ^ { t _ { i + 1 } } \exp ( \mathcal { R } _ { k } / \tau ) } ,\tag{10}
$$

where $\tau$ is a temperature parameter that controls the smoothness of token merging, which is set as 1.0 in our experiments. After that, $o _ { t } ^ { \prime }$ is further propagated through the subsequent layers.

In Appendix A, we present optimized implementation of the proposed algorithm, including efficient chunk partitioning and token merging via matrix parallel operators, which achieves negligible runtime latency in practice.

## 3.4 Early Exiting of Vision Tokens

As suggested by previous work (Wu et al., 2026) that deeper layers tend to perform visionindependent reasoning, we observe a similar pattern in the hybrid architecture: the proportion of important visual tokens sharply decreases in later layers, as shown in Figure 6. To further validate this observation, we conduct a perplexity (PPL) analysis by systematically removing all vision tokens at each layer and measuring the resulting PPL relative to the full-token baseline:

![](images/75197844a45ee5341304d301a5617f0ebefa5f33ac81b7b8af06fae9522be45d.jpg)

where the PPL drops sharply and plateaus after layer-24 of the model, corroborating the importance pattern observed that vision tokens become increasingly negligible in deep layers. We also report the MME benchmark results that discarding vision tokens beyond different layer depths:
<table><tr><td>Exit Layer</td><td>1</td><td>12</td><td>24</td><td>32 (N/A)</td></tr><tr><td>MME</td><td>967.3</td><td>2169.5</td><td>2401.6</td><td>2398.2</td></tr><tr><td>MMB</td><td>22.6</td><td>80.4</td><td>85.2</td><td>85.6</td></tr><tr><td>SQA</td><td>83.7</td><td>90.6</td><td>92.3</td><td>92.2</td></tr></table>

which demonstrates that the model can fully preserve its multimodal reasoning performance even without vision tokens in the deep layers. Based on these, we propose a vision token early-exit mechanism that discards all vision tokens in the deep layers of the model, which further reducing the computational burden.

## 4 Experiments

## 4.1 Setup

Models and Benchmarks. Our experiments are mainly conducted on Qwen3.5-9B (Qwen Team, 2026), which is a popular hybrid model family with linear attention. To validate the generalizability of the proposed method, we also include Qwen3.5- 27B and InfiniteVL (Tao et al., 2025a) in additional experiments. All experiments are performed on NVIDIA A100 GPUs.

We conduct experiments on comprehensive benchmarks: MME (Fu et al., 2023), MMB (Liu et al., 2024a), GQA (Hudson and Manning, 2019), ScienceQA (Lu et al., 2022), TextVQA (Singh et al., 2019), POPE (Li et al., 2023b), VizWiz (Gurari et al., 2018) and MMStar (Chen et al., 2024b). We employ LMMs-Eval (Zhang et al., 2024a) for benchmark evaluation.

Hyperparameters. In our main experiments, importance-based token compression is applied by default at the end of layer-1 (whereas applied at the first softmax-attention layer for attention-scorebased methods), and the early exit in V-CoLA is set to layer-24. To ensure a fair comparison, we align the average token remain rates across different methods in our experiments. We set distance function $\psi ( \cdot , \cdot )$ as cosine distance. The window parameters $( w _ { 1 } , w _ { 2 } )$ in Equation 7 are set as (0, 4) and the weight λ in Equation 8 is set as 0.1.

## 4.2 Main Results

Comparison with State-of-the-art Methods. We select several advanced and representative methods for comparison: (1) FastV (Chen et al., 2024a), (2) SparseVLM (Zhang et al., 2024b), (3) DART (Wen et al., 2025), (4) VisionZip (Yang et al., 2025b), (5) DTP (Park et al., 2025). Notably, DTP is also specifically designed for the Linear Attention architecture. As shown in Table 1, our method consistently outperforms the compared approaches across different benchmarks under various compression rates.

Efficiency of V-CoLA. To demonstrate the practical performance of our method in real-world applications, we compare the actual multimodal inference latency under varying remain rates. As shown in Table 2, V-CoLA achieves consistent reductions in prefill latency compared to the original model, attaining a speedup from 1.86× to 6.15×. Furthermore, we report the runtime of the main components in Table 3. Notably, our implementation of the Extended Gated DeltaRule maintains comparable latency to the original operator under single-query inference, while substantially outperforming it under 4-query inference, achieving a 3.85× speedup. Owing to our efficient implementation, the Adaptive Token Merging incurs only

Table 1: We compare our method with state-of-the-art approaches using Qwen3.5-9B. The best results are bold.
<table><tr><td colspan="7"></td></tr><tr><td>Method</td><td>ME MMB Upper Bound, 880 Tokens (100%)</td><td>GA</td><td>VOA</td><td>TΛA</td><td>POE</td><td>iz</td><td>Mttar</td><td>(%)SAV</td></tr><tr><td colspan="7"></td></tr><tr><td>Qwen3.5-9B 2398.2</td><td>85.6 61.1</td><td></td><td>92.2</td><td>83.2</td><td>89.9</td><td>69.2</td><td>49.3</td><td>100.0</td></tr><tr><td colspan="7">Remain 440 Tokens in Average (↓ 50.0%)</td></tr><tr><td>FastV (Chen et al., 2024a)</td><td>2351.6 84.8</td><td>60.4</td><td>92.1</td><td>79.5</td><td>88.8</td><td>67.6</td><td>45.3</td><td>97.5</td></tr><tr><td>SparseVLM (Zhang et al., 2024b)</td><td>2333.1 84.7</td><td>60.1</td><td>92.2</td><td>80.1</td><td>88.7</td><td>67.5</td><td>45.6</td><td>97.5</td></tr><tr><td>DART (Wen et al., 2025)</td><td>2220.4 83.3</td><td>59.6</td><td>91.7</td><td>69.3</td><td>88.9</td><td>67.0</td><td>48.3</td><td>95.5</td></tr><tr><td>VisionZip (Yang et al., 2025b) DTP (Park et al., 2025)</td><td>2376.5 84.7</td><td>60.7</td><td>92.6</td><td>80.6</td><td>89.4</td><td>68.0</td><td>48.9</td><td>99.0</td></tr><tr><td>V-CoLA (Ours) 2398.5</td><td>2197.2 81.6</td><td>58.7 61.0</td><td>91.0 92.7</td><td>71.5 80.6</td><td>87.3 89.8</td><td>67.6 68.3</td><td>45.0</td><td>94.2</td></tr><tr><td>Remain 220 Tokens in Average (↓ 75.0%)</td><td>85.0</td><td></td><td></td><td></td><td></td><td></td><td>49.6</td><td>99.5</td></tr><tr><td colspan="7"></td></tr><tr><td rowspan="5">FastV (Chen et al., 2024a) SparseVLM (Zhang et al., 2024b) DART (Wen et al., 2025)</td></tr><tr><td>2219.8 83.7</td><td>57.4</td><td>90.4</td><td>58.9</td><td>85.7</td><td>64.4</td><td>42.4</td></tr><tr><td>2252.8 83.5</td><td>55.1</td><td>89.9</td><td>61.8</td><td>85.4</td><td>63.5</td><td>43.7</td><td>90.9 91.1</td></tr><tr><td>2025.8</td><td>79.7</td><td>55.6</td><td>89.9</td><td>52.3</td><td>86.2</td><td>65.5</td><td>43.3 88.4</td></tr><tr><td>VisionZip (Yang et al., 2025b) 2247.1</td><td>83.6 59.1</td><td>92.1</td><td></td><td>65.7</td><td>88.0</td><td>66.1 47.0</td><td>94.5</td></tr><tr><td>DTP (Park et al., 2025) 1975.3</td><td>76.2</td><td>53.9</td><td>87.9</td><td>58.6</td><td>79.7</td><td>66.7</td><td>39.6</td><td>86.3</td></tr><tr><td>V-CoLA (Ours) 2268.7</td><td>84.4</td><td>60.1</td><td>92.4</td><td>63.4</td><td>88.9</td><td>66.7</td><td>47.2</td><td>94.8</td></tr><tr><td></td><td>Remain 110 Tokens in Average (↓ 87.5%)</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>FastV (Chen et al., 2024a) 1387.6</td><td>56.5</td><td>43.0</td><td>84.2</td><td>14.6</td><td>61.4</td><td>57.0</td><td>28.3</td><td>63.9</td></tr><tr><td>SparseVLM (Zhang et al., 2024b) 1622.4</td><td>68.1</td><td>43.7</td><td>85.3</td><td>16.5</td><td>69.5</td><td>57.1</td><td>33.0</td><td>69.7</td></tr><tr><td>DART (Wen et al., 2025)</td><td>1792.4 71.3</td><td>47.6</td><td>87.8</td><td>38.2</td><td>81.4</td><td>61.7</td><td>37.9</td><td>79.2</td></tr><tr><td>VisionZip (Yang et al., 2025b)</td><td>2036.0 77.3</td><td>56.7</td><td>90.8</td><td>45.7</td><td>84.1</td><td>65.3</td><td>40.3</td><td>86.4</td></tr><tr><td>DTP (Park et al., 2025)</td><td>1721.4 64.7</td><td>47.6</td><td>85.2</td><td>45.9</td><td>68.2</td><td>63.2</td><td>32.4</td><td>75.7</td></tr><tr><td>V-CoLA (Ours)</td><td>2012.0 77.9</td><td>57.5</td><td>90.9</td><td>46.0</td><td>88.3</td><td>68.3</td><td>41.4</td><td>88.0</td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr></table>

Table 2: Comparison of wall-clock runtime (ms) of the language model prefilling across various vision token numbers and different remain rates.
<table><tr><td># V-Tokens</td><td>100%</td><td>50.0%</td><td>25.0%</td><td>12.5%</td></tr><tr><td>2K</td><td>269</td><td> $1 4 5 \left( 1 . 8 6 \times \right)$ </td><td> $8 8 \ : ( 3 . 0 6 \times )$ </td><td> $6 1 \ : ( 4 . 4 1 \times )$ </td></tr><tr><td>4K</td><td>511</td><td> $2 7 3 \ : ( 1 . 8 7 \times )$ </td><td> $1 5 3 \ : ( 3 . 3 4 \times )$ </td><td> $9 4 \ : ( 5 . 4 4 \times )$ </td></tr><tr><td>8K</td><td>1034</td><td> $5 4 2 ( 1 . 9 1 \times )$ </td><td> $2 8 9 \left( 3 . 5 8 \times \right)$ </td><td> $1 6 8 ( 6 . 1 5 \times )$ </td></tr></table>

Table 3: Comparison of wall-clock runtime across different components.
<table><tr><td>Component</td><td># Tokens</td><td>Runtime (ms)</td></tr><tr><td>Softmax Attention</td><td> $8 \mathrm { K } \times 1$ </td><td>68.5</td></tr><tr><td rowspan="2">Gated DeltaRule</td><td> $\mathbf { \bar { 8 K } } \mathbf { \bar { \times } } \mathbf { \bar { 1 } }$ </td><td>34.1</td></tr><tr><td> $8 \mathrm { K } \times 4$ </td><td>135.1</td></tr><tr><td rowspan="2">Extended Gated DeltaRule</td><td> $\bar { 8 \bf K } \bar { \times } \bar { 1 }$ </td><td>34.2</td></tr><tr><td>1  $8 \mathrm { K } \times 4$ </td><td> $3 5 . 1 \ : ( 3 . 8 5 \times )$ </td></tr><tr><td>Adaptive Token Merging</td><td>」  $\bar { 8 \bf K } \bar { \times } \bar { 1 }$ </td><td>0.86</td></tr></table>

Table 4: Ablation of parameter λ in $\mathcal { R } _ { t }$ calculation.
<table><tr><td>λ</td><td>MME</td><td>SQA</td><td>POPE</td><td>MMStar</td></tr><tr><td> $0 . 0 0 ( \mathrm { o n l y } \ \mathcal { R } _ { t } ^ { \mathrm { i m p } } )$ </td><td>2360.5</td><td>92.3</td><td>89.5</td><td>48.9</td></tr><tr><td>0.05</td><td>2388.0</td><td>92.5</td><td>89.6</td><td>49.6</td></tr><tr><td>0.10</td><td>2398.5</td><td>92.7</td><td>89.8</td><td>49.8</td></tr><tr><td>0.90</td><td>2333.1</td><td>91.7</td><td>87.3</td><td>48.0</td></tr></table>

0.86ms of overhead, accounting for approximately 0.1% of the total model prefill cost. This highlights the deployment friendliness of V-CoLA in practical applications. We also provide FLOPs analysis in Appendix C and an end-to-end latency breakdown in Appendix D.

## 4.3 More Analysis

Ablation of λ. The parameter λ controls the weight of $\mathcal { R } _ { t } ^ { \mathrm { u n i } }$ in Equation 8. We compare different λ in Table 4, the results show that $\lambda = 0 . 1$ produces better performance.

Ablation of $( w _ { 1 } , w _ { 2 } )$ . We also compare different setting of window parameters $( w _ { 1 } , w _ { 2 } )$ in Equation 7, where we find that expanding a single pseudo query into a window of length 4 enhances robustness and improves performance, with forward window selection proving more advantageous than backward selection. However, further increasing the number of queries beyond this point yields no additional performance gains.

Table 5: Ablation of window parameters $( w _ { 1 } , w _ { 2 } )$
<table><tr><td> $( w _ { 1 } , w _ { 2 } )$ </td><td>MME</td><td>SQA</td><td>POPE</td><td>MMStar</td></tr><tr><td>(0,1)</td><td>2387.6</td><td>92.5</td><td>89.3</td><td>49.1</td></tr><tr><td>(0,4)</td><td>2398.5</td><td>92.7</td><td>89.8</td><td>49.8</td></tr><tr><td>(−4,0)</td><td>2376.4</td><td>92.4</td><td>87.9</td><td>49.3</td></tr><tr><td>(−4,4)</td><td>2395.2</td><td>92.6</td><td>89.9</td><td>49.5</td></tr></table>

Table 6: Ablation of main components.
<table><tr><td>Method</td><td>MME</td><td>SQA</td><td>POPE</td><td>MMStar</td></tr><tr><td colspan="5">↓ 75.0% Vision Tokens</td></tr><tr><td>V-CoLA - w/o Ada. Merging</td><td>2268.7</td><td>92.4</td><td>88.9</td><td>47.2</td></tr><tr><td>- w/o Early Exit</td><td>2197.4 2235.1</td><td>91.5 91.9</td><td>88.4 88.2</td><td>44.3 45.6</td></tr><tr><td colspan="3">↓ 87.5% Vision Tokens</td><td></td><td></td></tr><tr><td>V-CoLA</td><td>2012.0</td><td>90.9</td><td>88.3</td><td>41.4</td></tr><tr><td>- w/o Ada. Merging</td><td>1913.4</td><td>89.7</td><td>86.2</td><td>36.5</td></tr><tr><td>- w/o Early Exit</td><td>1970.3</td><td>89.4</td><td>87.6</td><td>38.1</td></tr></table>

Hyperparameter sensitivity. Tables 4 and 5 vary smoothly rather than sharply, where $\lambda \ \in$ [0.05, 0.10] performs comparably and only an extreme value degrades notably, and setting (w<sub>1</sub>, w<sub>2</sub>) to (0, 4) and (−4, 4) yields closely matching results. Moreover, we adopt an identical configuration for Qwen3.5-9B, Qwen3.5-27B and InfiniteVL in Tables 1 and 7 and obtain the best results on all three, indicating that our default setting transfers across model scales and architecture families without per-model tuning.

Ablation of main components. We additionally ablate two key components: adaptive token merging and early exit of vision tokens. As shown in Table 6, removing either component leads to performance drops. This confirms the complementary roles of both components: adaptive merging provides a more comprehensive way to aggregate token information than hard pruning, while early exit enables us to reallocate the redundant token budget in deeper layers toward shallow-layer visual understanding. Notably, by tuning the temperature coefficient, our merging strategy can also emulate hard pruning, affording users greater flexibility for customization.

Generalizability of V-CoLA. As a supplementary analysis, we extend comparisons to different model scales (Qwen3.5-27B) and model families (InfiniteVL). As shown in Table 7, V-CoLA consistently achieves leading performance across models and benchmarks, demonstrating the strong generalization of our method. Evaluations on long-form multi-turn dialogue and small-object recognition are further provided in Appendix D.

Table 7: We extend comparison with state-of-the-art approaches to include Qwen3.5-27B and InfiniteVL, with fixed ↓ 75.0% token compression.
<table><tr><td>Method</td><td>MME</td><td>SQA</td><td>POPE</td><td>MMStar</td></tr><tr><td>Qwen3.5-27B</td><td>2517.4</td><td>97.0</td><td>90.5</td><td>58.1</td></tr><tr><td>- DART</td><td>2173.0</td><td>91.9</td><td>82.4</td><td>44.5</td></tr><tr><td>- VisionZip</td><td>2405.2</td><td>96.0</td><td>88.4</td><td>53.3</td></tr><tr><td>- DTP</td><td>1798.2</td><td>89.2</td><td>70.2</td><td>43.1</td></tr><tr><td>- V-CoLA (Ours)</td><td>2414.5</td><td>96.1</td><td>88.8</td><td>53.7</td></tr><tr><td>InfiniteVL</td><td>1998.9</td><td>86.0</td><td>87.9</td><td>53.7</td></tr><tr><td>- DART</td><td>1885.5</td><td>84.7</td><td>85.0</td><td>49.0</td></tr><tr><td>- VisionZip</td><td>1883.8</td><td>84.8</td><td>86.8</td><td>48.4</td></tr><tr><td>- DTP</td><td>1663.8</td><td>83.0</td><td>82.0</td><td>44.1</td></tr><tr><td>- V-CoLA (Ours)</td><td>1893.1</td><td>85.8</td><td>86.3</td><td>49.3</td></tr></table>

Together with Table 1, our evaluation thus covers representative hybrid linear-attention VLMs across two model scales and two architecture families. Beyond these models, both criteria remain well defined for recurrent backbones with compatible matrix-state readouts, including updates of the general form $\mathbf { S } _ { t } = \mathbf { A } _ { t } \mathbf { S } _ { t - 1 } + \mathbf { B } _ { t } { v } _ { t } { k } _ { t } ^ { \top }$ , and extending the evaluation to such backboned models is a natural direction for future work.

## 5 Related Work

## 5.1 Visual Token Compression for MLLMs

Since visual tokens often dominate the input sequence in MLLMs (Chen et al., 2024a; Yang et al., 2025b), token compression has emerged as a key direction for reducing computational cost. Existing methods fall into two categories. Attention-based methods prune tokens via softmax attention scores, either in a single step (Chen et al., 2024a), progressively across layers (Xing et al., 2024), or with textguided sparsification and feature recycling (Zhang et al., 2024b; Yang et al., 2025b; Han et al., 2026). Similarity-based methods provide attention-free alternatives through bipartite matching (Bolya et al., 2022; Chai et al., 2024), feature-spatial similarity (Yang et al., 2025a), or diversity-aware selection (Wen et al., 2025). However, the former relies on softmax attention scores, which are sparsely available and exhibit different distributional properties in hybrid linear-attention architectures. The latter assumes that hidden representations preserve reliable similarity structure, an assumption challenged by the state-recurrent updates in linear attention, where outputs are queried from a shared compressed state. While pioneering work (Park et al., 2025) have explored token compression for mamba-based architectures, it merely leverage intermediate output for importance ranking with suboptimal performance.

A parallel line of work reduces tokens by grouping rather than scoring, e.g., the Visual-Word Tokenizer (Gee et al., 2024) merges patches into visual words according to intra- and inter-image statistics. Such grouping is performed at the encoder stage from static patch-space statistics, and is therefore complementary to our importance signal derived from state recurrence inside the backbone.

## 5.2 Linear Attention Architectures

Beyond the simple recurrent accumulation of early linear attention (Katharopoulos et al., 2020), recent work introduce more expressive updating paradigms. DeltaNet (Schlag et al., 2021) erases existing associations along the current key direction before writing new ones, and Gated DeltaNet (Yang et al., 2024) further adds per-dimension decay gates for selective forgetting. More broadly, the same principle underlies state-space and recurrent alternatives such as Mamba-2 (Dao and Gu, 2024) and RWKV (Peng et al., 2023), which likewise compress the history into a bounded state under data-dependent gating. These designs underpin hybrid architectures such as Qwen3.5 (Qwen Team, 2026) and Kimi-Linear (Team et al., 2025), which interleave linear recurrence with softmax attention, and have been extended to multimodal settings (Hou et al., 2025; Li et al., 2025; Tao et al., 2025a). Nevertheless, token compression tailored to linearattention VLMs remains largely unexplored, motivating architecture-aware importance estimation.

## 6 Conclusion

We present V-CoLA, a training-free vision token compression framework tailored to hybrid VLMs with linear attention. Motivated by the observation that previous token compression method fail to transfer to hybrid architectures, V-CoLA estimates token importance from the intrinsic dynamics of linear attention, and further incorporates an adaptive chunk-wise merging strategy and an early-exit mechanism for vision tokens. With implementation-level optimizations compatible with chunk-wise parallelism, V-CoLA delivers state-of-the-art performance across diverse benchmarks while substantially improving inference efficiency.

## 7 Limitations

While V-CoLA demonstrates strong empirical performance and efficiency gains, we acknowledge several limitations of the current study, which also point to promising directions for future work.

Architectural coverage. Our evaluation primarily focuses on hybrid VLMs that interleave linearattention and softmax-attention layers, such as Qwen3.5 and InfiniteVL. While this family represents one of the most widely adopted designs in recent linear-attention VLMs, other architectural variants, including purely Mamba-based or fully linear-attention VLMs, are not yet covered in our experiments. Extending V-CoLA to such architectures may require revisiting the interaction between our uniqueness-aware importance criterion and the absence of periodic softmax-attention layers, which we leave for future exploration.

Scope of linear-attention formulations. Our analysis and method design are mainly grounded in the Gated DeltaNet formulation, which underlies several recent hybrid VLMs. Although the reconstruction-based perspective we adopt is fairly general and may extend to other recurrent state update rules (e.g., other gated linear attention variants), a more systematic study across a broader range of linear-attention formulations would be valuable for assessing the generality of the proposed criterion.

## 8 Ethical Considerations

This work focuses on improving the inference efficiency of vision-language models through a training-free token compression framework. We conduct no user studies or user-facing deployment, and no human-subject experiments are involved. All experiments are performed on publicly available benchmarks and pretrained models, following their intended research use. These resources contain no intended personally identifiable or sensitive information. Since V-CoLA is training-free and operates on top of existing VLMs, it does not introduce additional biases beyond those already present in the underlying models.

## References

Simran Arora, Sabri Eyuboglu, Michael Zhang, Aman Timalsina, Silas Alberti, Dylan Zinsley, James Zou, Atri Rudra, and Christopher Ré. 2024. Simple linear attention language models balance the recall-throughput tradeoff. arXiv preprint arXiv:2402.18668.

Shuai Bai, Yuxuan Cai, Ruizhe Chen, Keqin Chen, Xionghui Chen, Zesen Cheng, Lianghao Deng, Wei Ding, Chang Gao, Chunjiang Ge, and 1 others. 2025. Qwen3-vl technical report. arXiv preprint arXiv:2511.21631.

Daniel Bolya, Cheng-Yang Fu, Xiaoliang Dai, Peizhao Zhang, Christoph Feichtenhofer, and Judy Hoffman. 2022. Token merging: Your vit but faster. arXiv preprint arXiv:2210.09461.

Wenhao Chai, Enxin Song, Yilun Du, Chenlin Meng, Vashisht Madhavan, Omer Bar-Tal, Jenq-Neng Hwang, Saining Xie, and Christopher D Manning. 2024. Auroracap: Efficient, performant video detailed captioning and a new benchmark. arXiv preprint arXiv:2410.03051.

Liang Chen, Haozhe Zhao, Tianyu Liu, Shuai Bai, Junyang Lin, Chang Zhou, and Baobao Chang. 2024a. An image is worth 1/2 tokens after layer 2: Plug-andplay inference acceleration for large vision-language models. In European Conference on Computer Vision, pages 19–35. Springer.

Lin Chen, Jinsong Li, Xiaoyi Dong, Pan Zhang, Yuhang Zang, Zehui Chen, Haodong Duan, Jiaqi Wang, Yu Qiao, Dahua Lin, and 1 others. 2024b. Are we on the right way for evaluating large vision-language models? Advances in Neural Information Processing Systems, 37:27056–27087.

Thomas M Cover. 1999. Elements of information theory. John Wiley & Sons.

Tri Dao and Albert Gu. 2024. Transformers are ssms: Generalized models and efficient algorithms through structured state space duality. arXiv preprint arXiv:2405.21060.

Chaoyou Fu, Peixian Chen, Yunhang Shen, Yulei Qin, Mengdan Zhang, Xu Lin, Jinrui Yang, Xiawu Zheng, Ke Li, Xing Sun, and 1 others. 2023. Mme: A comprehensive evaluation benchmark for multimodal large language models. arXiv preprint arXiv:2306.13394.

Leonidas Gee, Wing Yan Li, Viktoriia Sharmanska, and Novi Quadrianto. 2024. Visual-word tokenizer: Beyond fixed sets of tokens in vision transformers. arXiv preprint arXiv:2411.15397.

Danna Gurari, Qing Li, Abigale J. Stangl, Anhong Guo, Chi Lin, Kristen Grauman, Jiebo Luo, and Jeffrey P. Bigham. 2018. Vizwiz grand challenge: Answering visual questions from blind people. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR).

Yuhang Han, Xuyang Liu, Zihan Zhang, Pengxiang Ding, Junjie Chen, Honggang Chen, Donglin Wang, Qingsen Yan, and Siteng Huang. 2026. Filter, correlate, compress: Training-free token reduction for mllm acceleration. In Proceedings ofthe AAAI Conference on Artificial Intelligence, volume 40, pages 4601–4609.

Haowen Hou, Peigen Zeng, Fei Ma, and Fei Richard Yu. 2025. Visualrwkv: Exploring recurrent neural networks for visual language models. In Proceedings of the 31st International Conference on Computational Linguistics, pages 10423–10434.

Lianyu Hu, Fanhua Shang, Wei Feng, and Liang Wan. 2025. Lightvlm: Acceleraing large multimodal models with pyramid token merging and kv cache compression. arXiv preprint arXiv:2509.00419.

Drew A Hudson and Christopher D Manning. 2019. Gqa: A new dataset for real-world visual reasoning and compositional question answering. In Proceed ings of the IEEE/CVF conference on computer vision and pattern recognition, pages 6700–6709.

Samy Jelassi, David Brandfonbrener, Sham M Kakade, and Eran Malach. 2024. Repeat after me: Transformers are better than state space models at copying. arXiv preprint arXiv:2402.01032.

Angelos Katharopoulos, Apoorv Vyas, Nikolaos Pappas, and François Fleuret. 2020. Transformers are rnns: Fast autoregressive transformers with linear attention. In International conference on machine learning, pages 5156–5165. PMLR.

Bohao Li, Rui Wang, Guangzhi Wang, Yuying Ge, Yixiao Ge, and Ying Shan. 2023a. Seed-bench: Benchmarking multimodal llms with generative comprehension. arXiv preprint arXiv:2307.16125.

Yifan Li, Yifan Du, Kun Zhou, Jinpeng Wang, Xin Zhao, and Ji-Rong Wen. 2023b. Evaluating object hallucination in large vision-language models. In Proceedings of the 2023 conference on empirical methods in natural language processing, pages 292– 305.

Yingyue Li, Bencheng Liao, Wenyu Liu, and Xinggang Wang. 2025. Matvlm: Hybrid mamba-transformer for efficient vision-language modeling. In Proceedings ofthe IEEE/CVF International Conference on Computer Vision, pages 20878–20888.

Yuan Liu, Haodong Duan, Yuanhan Zhang, Bo Li, Songyang Zhang, Wangbo Zhao, Yike Yuan, Jiaqi Wang, Conghui He, Ziwei Liu, and 1 others. 2024a. Mmbench: Is your multi-modal model an all-around player? In European conference on computer vision, pages 216–233. Springer.

Ziyu Liu, Tao Chu, Yuhang Zang, Xilin Wei, Xiaoyi Dong, Pan Zhang, Zijian Liang, Yuanjun Xiong, Yu Qiao, Dahua Lin, and 1 others. 2024b. Mmdu: A

multi-turn multi-image dialog understanding benchmark and instruction-tuning dataset for lvlms. Advances in Neural Information Processing Systems, 37:8698–8733.

Pan Lu, Swaroop Mishra, Tanglin Xia, Liang Qiu, Kai-Wei Chang, Song-Chun Zhu, Oyvind Tafjord, Peter Clark, and Ashwin Kalyan. 2022. Learn to explain: Multimodal reasoning via thought chains for science question answering. Advances in neural information processing systems, 35:2507–2521.

Yasmine Omri, Parth Shroff, and Thierry Tambe. 2025. Token sequence compression for efficient multimodal computing. arXiv preprint arXiv:2504.17892.

Seong Yeol Park, Min Jung Kwon, Xianghua Piao, and Yeong Hyeon Gu. 2025. Dtp: Delta-guided two stage pruning for mamba-based multimodal large language models. In The Fourteenth International Conference on Learning Representations.

Bo Peng, Eric Alcaide, Quentin Anthony, Alon Albalak, Samuel Arcadinho, Stella Biderman, Huanqi Cao, Xin Cheng, Michael Chung, Leon Derczynski, and 1 others. 2023. Rwkv: Reinventing rnns for the transformer era. In Findings of the association for computational linguistics: EMNLP 2023, pages 14048– 14077.

Liangzu Peng, Aditya Chattopadhyay, Luca Zancato, Elvis Nunez, Wei Xia, and Stefano Soatto. 2025. Gated kalmanet: A fading memory layer through test-time ridge regression. arXiv preprint arXiv:2511.21016.

Qwen Team. 2026. Qwen3.5: Towards native multimodal agents.

Imanol Schlag, Kazuki Irie, and Jürgen Schmidhuber. 2021. Linear transformers are secretly fast weight programmers. In International conference on machine learning, pages 9355–9366. PMLR.

Kele Shao, Keda Tao, Can Qin, Haoxuan You, Yang Sui, and Huan Wang. 2025. Holitom: Holistic token merging for fast video large language models. In NeurIPS.

Leqi Shen, Guoqiang Gong, Tao He, Yifeng Zhang, pengzhang liu, Sicheng Zhao, and Guiguang Ding. 2025. FastVID: Dynamic density pruning for fast video large language models. In The Thirty-ninth Annual Conference on Neural Information Processing Systems.

Aaditya Singh, Adam Fry, Adam Perelman, Adam Tart, Adi Ganesh, Ahmed El-Kishky, Aidan McLaughlin, Aiden Low, AJ Ostrow, Akhila Ananthram, and 1 others. 2025. Openai gpt-5 system card. arXiv preprint arXiv:2601.03267.

Amanpreet Singh, Vivek Natarajan, Meet Shah, Yu Jiang, Xinlei Chen, Dhruv Batra, Devi Parikh, and Marcus Rohrbach. 2019. Towards vqa models

that can read. In Proceedings ofthe IEEE/CVF conference on computer vision and pattern recognition, pages 8317–8326.

Hongyuan Tao, Bencheng Liao, Shaoyu Chen, Haoran Yin, Qian Zhang, Wenyu Liu, and Xinggang Wang. 2025a. Infinitevl: Synergizing linear and sparse attention for highly-efficient, unlimited-input visionlanguage models. arXiv preprint arXiv:2512.08829.

Keda Tao, Can Qin, Haoxuan You, Yang Sui, and Huan Wang. 2025b. Dycoke: Dynamic compression of tokens for fast video large language models. In Proceedings of the Computer Vision and Pattern Recognition Conference, pages 18992–19001.

Kimi Team, Yu Zhang, Zongyu Lin, Xingcheng Yao, Jiaxi Hu, Fanqing Meng, Chengyin Liu, Xin Men, Songlin Yang, Zhiyuan Li, and 1 others. 2025. Kimi linear: An expressive, efficient attention architecture. arXiv preprint arXiv:2510.26692.

Pavan Kumar Anasosalu Vasu, Fartash Faghri, Chun-Liang Li, Cem Koc, Nate True, Albert Antony, Gokula Santhanam, James Gabriel, Peter Grasch, Oncel Tuzel, and 1 others. 2025. Fastvlm: Efficient vision encoding for vision language models. In Proceedings of the Computer Vision and Pattern Recognition Conference, pages 19769–19780.

Haicheng Wang, Zhemeng Yu, Gabriele Spadaro, Chen Ju, Victor Quétu, Shuai Xiao, and Enzo Tartaglione. 2025a. Folder: Accelerating multi-modal large language models with enhanced performance. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 23614–23625.

Weiyun Wang, Zhangwei Gao, Lixin Gu, Hengjun Pu, Long Cui, Xingguang Wei, Zhaoyang Liu, Linglin Jing, Shenglong Ye, Jie Shao, and 1 others. 2025b. Internvl3. 5: Advancing open-source multimodal models in versatility, reasoning, and efficiency. arXiv preprint arXiv:2508.18265.

Zichen Wen, Yifeng Gao, Shaobo Wang, Junyuan Zhang, Qintong Zhang, Weijia Li, Conghui He, and Linfeng Zhang. 2025. Stop looking for “important tokens” in multimodal language models: Duplication matters more. In Proceedings of the 2025 Conference on Empirical Methods in Natural Language Processing, pages 9972–9991.

Hao Wu, Yingqi Fan, Jinyang Dai, Junlong Tong, Yunpu Ma, and Xiaoyu Shen. 2026. Hidrop: Hierarchical vision token reduction in mllms via late injection, concave pyramid pruning, and early exit. Preprint, arXiv:2602.23699.

Penghao Wu and Saining Xie. 2024. V\*: Guided visual search as a core mechanism in multimodal llms. In 2024 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 13084–13094. IEEE.

Long Xing, Qidong Huang, Xiaoyi Dong, Jiajie Lu, Pan Zhang, Yuhang Zang, Yuhang Cao, Conghui He, Jiaqi

Wang, Feng Wu, and 1 others. 2024. Pyramiddrop: Accelerating your large vision-language models via pyramid visual redundancy reduction. arXiv preprint arXiv:2410.17247.

Cheng Yang, Yang Sui, Jinqi Xiao, Lingyi Huang, Yu Gong, Chendi Li, Jinghua Yan, Yu Bai, Ponnuswamy Sadayappan, Xia Hu, and 1 others. 2025a. Topv: Compatible token pruning with inference time optimization for fast and low-memory multimodal vision language model. In Proceedings ofthe Computer Vision and Pattern Recognition Conference, pages 19803–19813.

Senqiao Yang, Yukang Chen, Zhuotao Tian, Chengyao Wang, Jingyao Li, Bei Yu, and Jiaya Jia. 2025b. Visionzip: Longer is better but not necessary in vision language models. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 19792–19802.

Songlin Yang, Jan Kautz, and Ali Hatamizadeh. 2024. Gated delta networks: Improving mamba2 with delta rule. arXiv preprint arXiv:2412.06464.

Quan-Sheng Zeng, Yunheng Li, Qilong Wang, Peng-Tao Jiang, Zuxuan Wu, Ming-Ming Cheng, and Qibin Hou. 2025. A glimpse to compress: Dynamic visual token pruning for large vision-language models. arXiv preprint arXiv:2508.01548.

Hongzhi Zhang, Jingyuan Zhang, Xingguang Ji, Qi Wang, and Fuzheng Zhang. 2025. Dyntok: Dynamic compression of visual tokens for efficient and effective video understanding. arXiv preprint arXiv:2506.03990.

Kaichen Zhang, Bo Li, Peiyuan Zhang, Fanyi Pu, Joshua Adrian Cahyono, Kairui Hu, Shuai Liu, Yuanhan Zhang, Jingkang Yang, Chunyuan Li, and Ziwei Liu. 2024a. Lmms-eval: Reality check on the evaluation of large multimodal models. Preprint, arXiv:2407.12772.

Yuan Zhang, Chun-Kai Fan, Junpeng Ma, Wenzhao Zheng, Tao Huang, Kuan Cheng, Denis Gudovskiy, Tomoyuki Okuno, Yohei Nakata, Kurt Keutzer, and 1 others. 2024b. Sparsevlm: Visual token sparsification for efficient vision-language model inference. arXiv preprint arXiv:2410.04417.

Shiyu Zhao, Zhenting Wang, Felix Juefei-Xu, Xide Xia, Miao Liu, Xiaofang Wang, Mingfu Liang, Ning Zhang, Dimitris N Metaxas, and Licheng Yu. 2025. Accelerating multimodal large language models by searching optimal vision token reduction. In Proceedings of the Computer Vision and Pattern Recognition Conference, pages 29869–29879.

## A V-CoLA Implementation Parallelism

Making token compression compatible with existing inference acceleration techniques has long remained highly challenging (Hu et al., 2025; Yang et al., 2025a; Zeng et al., 2025). In this section, we present a parallel implementation of the main components introduced above, aiming to maximize compatibility with linear-attention acceleration algorithms (Team et al., 2025; Yang et al., 2024) while incurring minimal additional overhead.

Token importance $\mathcal { R } _ { t } ^ { \mathbf { i m p } }$ . Since the computation of $\mathcal { R } _ { t } ^ { \mathrm { i m p } }$ depends only on the original input and the final state $S _ { L }$ , it is naturally compatible with various forms of chunk-wise parallelism while maintaining high hardware efficiency.

Token uniqueness $\mathcal { R } _ { t } ^ { \mathbf { u n i } }$ Existing linearattention acceleration primarily rely on chunk-wise parallel forms (Team et al., 2025; Yang et al., 2024), which strike a balance between computational efficiency and hardware parallelism by combining inter-chunk recurrence with intra-chunk parallelism. However, this leaves a challenge: under chunk-wise parallelism, the state cannot be accessed at arbitrary time steps, and the standard forward pass of linear attention permits only a single sequence of $( q , k , v )$ inputs.

As shown in Figure 7 (left), we first extend Gated DeltaRule for multiple queries injection. Since the injected queries are only involved in computing outputs and do not participate in state recurrence, the increase in computation is kept relatively low. After that, shown in Figure 7 (right), We construct pseudo queries $Q _ { t } ^ { * }$ from input keys with shift $\delta \in$ $[ w _ { 1 } , w _ { 2 } )$ , and perform single forward to get pseudo output matrix $\mathbf { U } _ { t , \delta } ^ { * }$ . Then, we can re-shift the $\mathbf { U } _ { t , \delta } ^ { * }$ to calculate result:

$$
\mathcal { R } _ { t } ^ { \mathrm { u n i } } = \psi ( \mathbf { U } _ { t , \delta } ^ { * } , \mathbf { U } _ { t - 1 , \delta + 1 } ^ { * } ) .\tag{11}
$$

Adaptive Chunk-wise Token Merging . Since the partitioning of the input according to importance yields a large number of chunks with varying sizes, and each chunk requires independent weight normalization for weighted aggregation, this process poses challenges to parallelization. To address this issue, as shown in Figure 8, we regard the cumulative importance curve as a continuous function and partition the importance into equal intervals according to the target number of chunks to determine the ideal chunk boundaries. We then select the actual chunk boundaries as close as possible to these ideal positions, so that the total importance of each chunk is as balanced as possible while remaining statistically unbiased. The entire process is based on parallel matrix computation, thereby maximizing GPU computational efficiency.

## B Evaluation Benchmarks

Our experimental evaluation is performed on a diverse set of benchmarks. For clarity, we provide a brief overview of these benchmarks:

MME (Fu et al., 2023) evaluates the perception and cognition abilities of multimodal large language models across 14 subtasks. It adopts manually designed yes/no question pairs, where each image is paired with two questions expecting opposite yes/no answers.

MMBench (MMB) (Liu et al., 2024a) is a systematically designed objective benchmark that evaluates multimodal models through approximately 3,000 multiple-choice questions. It covers a hierarchical ability taxonomy spanning perception and reasoning, and introduces the CircularEval strategy where a model must answer correctly across all circularly shifted versions of the same question to be considered successful and produce more robust evaluation outcomes.

GQA (Hudson and Manning, 2019) is a largescale dataset for visual reasoning and compositional question answering, containing over 22M questions generated from Visual Genome scene graphs. It probes diverse reasoning skills such as spatial understanding, multi-step inference, and compositional generalization, and provides additional metrics for consistency, validity and plausibility.

ScienceQA (SQA) (Lu et al., 2022) is a multimodal science question answering benchmark comprising 21,208 multiple-choice questions collected from elementary and high school curricula. It spans natural science, language science, and social science, with questions accompanied by multimodal context and chain-of-thought annotations.

TextVQA (T-VQA) (Singh et al., 2019) challenges models to recognize and comprehend textual content embedded in natural images. It contains 45,336 questions over 28,408 images sourced from Open Images, where answering correctly depends on recognizing and understanding the textual content in the scene.

POPE (Li et al., 2023b) is a polling-based evaluation protocol that assesses the tendency of visionlanguage models to hallucinate objects. It reformulates hallucination assessment as binary yes/no questions about whether a specific object exists in the image, employing random, popular, and adversarial sampling strategies for selecting negative objects.

![](images/a9f495ff189d795b2e86bce9dceafef7ef488a9b2adc1deade1a3dd0ba4c434f.jpg)  
Figure 7: Parallelism algorithm of token uniqueness. (left) We extend Gated DeltaNet with bypass input and batched queries, it enables injecting additional queries when computing output. (right) We use shifted keys as pseudo queries, and calculate token uniqueness on output matrix U .

![](images/addd6bf0f80c399656530c5d4e577df456d4db22499310dc5181b35cf6d08873.jpg)  
Figure 8: Efficient implementation of adaptive token merging.

SeedBench (SEED) (Li et al., 2023a) is a largescale benchmark consisting of 19K multiple-choice questions with human-annotated answers, designed to evaluate the generative comprehension of multimodal LLMs. It spans 12 evaluation dimensions covering both image and video understanding, including spatial relation recognition and instance interaction.

VizWiz (Gurari et al., 2018) is a VQA dataset collected from blind people who take photos with smartphones and ask spoken questions about their surroundings. It comprises over 31K visual questions and presents unique real-world challenges including poor image quality and a significant fraction of unanswerable questions.

MMStar (Chen et al., 2024b) is a visionindispensable multimodal benchmark with 1,500 carefully curated samples across 6 capabilities and 18 detailed axes. It is specifically designed to eliminate data leakage and ensure that questions cannot be solved through text-only reasoning without actually perceiving the image.

MMDU (Liu et al., 2024b) is a multi-turn, multiimage dialogue understanding benchmark, whose samples contain up to 20 images, 27 turns and 18K multimodal tokens. Responses are scored along six dimensions, making it suitable for assessing long-form generation quality.

V\*-Bench (Wu and Xie, 2024) contains 191 samples targeting easily overlooked small objects in high-resolution, visually crowded scenes, split into direct attribute recognition and relative spatial reasoning subsets.

## C FLOPs Analysis and Theoretical Speedup

We analyze the computational cost of Qwen3.5’s 3:1 hybrid architecture, in which every 4 consecutive transformer layers comprise 3 Gated DeltaNet (linear-attention) layers and 1 gated softmax-attention layer (with GQA and a gated q-projection), and derive the theoretical speedup from vision-token reduction.

## C.1 Per-Layer Computational Cost

Consider a transformer with hidden dimension $d ,$ FFN intermediate dimension $d _ { f }$ , per-head dimension $d _ { h }$ , and input sequence length N. Each layer comprises a token mixer (attention) followed by a SwiGLU MLP. We split the per-layer FLOPs into a part linear in N (projections, MLP, and the linear-attention recurrence) and a part quadratic in N (softmax attention only).

Projections and MLP (linear in N). The SwiGLU MLP contributes $6 N d d _ { f }$ FLOPs. Both layer types apply analogous QKV-style and output projections, though with different widths:

• Softmax layer: a gated q-projection of width $2 n _ { h } d _ { h }$ , GQA k, v-projections of width $n _ { k v } d _ { h }$ each, and an output projection of width $n _ { h } d _ { h }$

• Linear layer (Gated DeltaNet): a fused QKV projection of width $2 d _ { k } + d _ { v }$ , an output-gate z of width $d _ { v } .$ , and an output projection of width d (plus a small depthwise conv1d and tiny a, b projections, all $O ( N d ) )$ .

For the Qwen3.5 default config $( n _ { h } d _ { h } = d _ { v } = d _ { \colon }$ $n _ { k v } = n _ { h } / 4 , d _ { k } = d / 2 )$ , the total projection output width is ≈ 4d for both types (softmax: 3.5d; linear: 4d). We therefore approximate the per-layer nonattention cost uniformly as

$$
\mathcal { F } _ { \mathrm { n o n - a t t n } } ( N ) \approx 2 N d ( 4 d + 3 d _ { f } ) ,\tag{12}
$$

where the factor of 2 accounts for fused multiplyadd operations.

Token-mixer compute. The attention computation itself differs between the two layer types:

• Softmax Attention. The $Q K ^ { \top }$ product and the Attn·V aggregation each cost $2 N ^ { 2 } n _ { h } d _ { h }$ FLOPs:

$$
\mathcal { F } _ { \mathrm { s o f t } } ( N ) = 4 N ^ { 2 } n _ { h } d _ { h } \approx 4 N ^ { 2 } d .\tag{13}
$$

• Linear Attention (Gated DeltaNet). The recurrent update in Eq. 2 costs ≈ $c d _ { k } d _ { v }$ FLOPs per head per token; aggregated over $n _ { v }$ value heads (keys/queries are replicated to match):

$$
\mathcal { F } _ { \mathrm { l i n } } ( N ) = c \cdot N n _ { v } d _ { k } d _ { v } \approx c \cdot N d d _ { h } ,\tag{14}
$$

with $c \approx 8$ obtained by counting the operations in Eq. 2 per head per token: the matrix-vector product $S _ { t - 1 } ^ { \top } k _ { t } \left( 2 d _ { k } d _ { v } \mathrm { F L O P s } \right)$ , the outer product $\beta _ { t } ( v _ { t } - S _ { t - 1 } ^ { \top } k _ { t } ) \ : k _ { t } ^ { \top } \ : ( d _ { k } d _ { v } )$ , the gated combination $\alpha _ { t } S _ { t - 1 } + \mathbf { \Omega } ( \cdot ) \ ( 2 d _ { k } d _ { v } )$ , and the readout $o _ { t } = S _ { t } ^ { \top } q _ { t } ( 2 d _ { k } d _ { v } )$ , totaling $\approx 7 d _ { k } d _ { v }$ ; we round up to 8 to cover the qk-L2 normalization and the small $\alpha , \beta$ activations. The approximation $\mathcal { F } _ { \mathrm { l i n } } ~ \approx ~ c N d d _ { h }$ uses $n _ { v } d _ { v } \ \approx \ d$ and $d _ { k } = d _ { v } = d _ { h }$ (Qwen3.5 default).

Thus $\mathcal { F } _ { \mathrm { s o f t } }$ is quadratic in N whereas $\mathcal { F } _ { \mathrm { l i n } }$ is linear, crossing over at $N = c \cdot d _ { h } ( \mathbf { e . g . } , N \approx 1 0 2 4$ for $d _ { h } = 1 2 8 )$

## C.2 Total FLOPs for 3:1 Hybrid Architecture

In the 3:1 hybrid architecture (e.g., Qwen3.5, InfiniteVL), every 4 consecutive layers comprise 3 linear and 1 softmax attention layers. For a model with L layers, the total FLOPs are:

$$
\begin{array} { r } { \mathcal { F } _ { \mathrm { h y b r i d } } ( N ) = \mathcal { L } \Big [ \underbrace { 2 N d ( 4 d + 3 d _ { f } ) + \frac { 3 } { 4 } c N d d _ { h } } _ { \mathrm { l i n e a r ~ i n ~ } N } } \\ { + \underbrace { N ^ { 2 } d } _ { \mathrm { q u a d r a t i c ~ i n ~ } N } \Big ] . } \end{array}\tag{15}
$$

Letting $a = 2 d ( 4 d + 3 d _ { f } ) + { \textstyle { \frac { 3 } { 4 } } }$ c d $d _ { h }$ and $b = d ,$ this simplifies to:

$$
\mathcal { F } _ { \mathrm { h y b r i d } } ( N ) = \mathcal { L } \big ( a N + b N ^ { 2 } \big ) .\tag{16}
$$

## C.3 Theoretical Speedup from Token Reduction

Suppose vision token compression is applied at layer ${ { l } _ { c } } ,$ reducing $N _ { v }$ tokens to $N _ { v } ^ { \prime } = r N _ { v }$ with retention ratio $r \in ( 0 , 1 )$ . The effective sequence length for the remaining $\mathcal { L } - l _ { c }$ layers becomes $N ^ { \prime } = N _ { t } + r N _ { v }$ . Since vision tokens dominate in VLMs $( N _ { v } \gg N _ { t } )$ , we have $N ^ { \prime } \approx r N$ , yielding a per-layer speedup:

$$
{ \boldsymbol { S } } (  { \boldsymbol { r } } ) \approx \frac { 1 } { r } \cdot \frac {  { \boldsymbol { a } } + b  { \boldsymbol { N } } } {  { \boldsymbol { a } } + b r  { \boldsymbol { N } } } .\tag{17}
$$

This reveals two regimes: $S ( r )  1 / r$ when $N \ll$ $a / b$ (linear-dominated) and $S ( r ) \to 1 / r ^ { 2 }$ when $N \gg a / b$ (quadratic-dominated), so the actual speedup interpolates between $1 / r$ and $1 / r ^ { 2 }$

General end-to-end speedup. Let $\rho = ( \mathcal { L } -$ $l _ { c } ) / \mathcal { L }$ denote the fraction of compressed layers, and $\gamma = b N / ( a + b N )$ the relative weight of the quadratic (softmax) cost per layer. The end-to-end speedup admits a compact closed form:

$$
S _ { \mathrm { t o t a l } } ( r ) = \frac { 1 } { ( 1 - \rho ) + \rho r \left[ ( 1 - \gamma ) + \gamma r \right] } ,\tag{18}
$$

which recovers $1 / r$ when $\gamma \quad  \quad 0$ (lineardominated) and $1 / r ^ { 2 }$ when $\gamma \  \ 1$ (quadraticdominated), with $\rho$ controlling how many layers benefit from compression.

As shown in Table 2, our empirical speedups closely match Eq. 18: at $r = 0 . 5$ with 8K vision tokens we obtain 1.91× (theory $\approx 2 . 0 \times )$ , and at $r = 0 . 1 2 5 , 6 . 1 5 \times ( \mathrm { t h e o r y } \approx 7 . 0 { \times } )$

## D Additional Evaluations

Long-form multi-turn dialogue. To verify that compression does not harm long-form generation, we evaluate Qwen3.5-9B on MMDU (Liu et al., 2024b), a multi-turn multi-image dialogue benchmark with up to 20 images, 27 turns and 18K multimodal tokens per sample. As shown in Table 8, V-CoLA preserves dialogue quality across all six dimensions, remaining within 1.4 points of the full model even at a 12.5% token budget.

Table 8: MMDU results on Qwen3.5-9B under varying average vision token retention rates. All dimensions are scored by the official protocol (higher is better).
<table><tr><td>Vision Tokens</td><td>100%</td><td>50.0%</td><td>25.0%</td><td>12.5%</td></tr><tr><td>Creativity</td><td>61.0</td><td>61.0</td><td>60.6</td><td>60.0</td></tr><tr><td>Richness</td><td>66.4</td><td>66.7</td><td>66.0</td><td>65.2</td></tr><tr><td>Visual Perception</td><td>62.2</td><td>62.8</td><td>61.7</td><td>61.2</td></tr><tr><td>Logical Coherence</td><td>77.5</td><td>77.9</td><td>77.3</td><td>76.7</td></tr><tr><td>Answer Accuracy</td><td>70.2</td><td>70.7</td><td>69.7</td><td>68.8</td></tr><tr><td>Image Relationship</td><td>63.1</td><td>64.0</td><td>63.0</td><td>62.2</td></tr><tr><td>Overall</td><td>66.6</td><td>67.0</td><td>66.0</td><td>65.4</td></tr></table>

Fine-grained small-object recognition. Since early exiting removes vision tokens in deep layers, we further examine tasks that rely on fine-grained visual evidence by evaluating all 191 samples of V\*-Bench (Wu and Xie, 2024), which targets easily overlooked objects in crowded high-resolution scenes. As reported in Table 9, exiting at layer-24 stays within 1.05 points of the full model while exiting at layer-12 loses 2.62 points, confirming the value of intermediate-layer visual access. Under a matched 50% budget, early exiting reallocates the deep-layer token budget to shallow layers and improves overall accuracy by 2.09 points.

Table 9: V\*-Bench accuracy on Qwen3.5-9B. “Pre-exit” denotes the average vision token retention rate before the exit layer.
<table><tr><td>Setting</td><td>Exit</td><td>Pre-exit</td><td>Direct</td><td>Relative</td><td>Overall</td></tr><tr><td>Early exit only</td><td>12</td><td>100%</td><td>70.43</td><td>77.63</td><td>73.30</td></tr><tr><td>Early exit only</td><td>24</td><td>100%</td><td>71.30</td><td>80.26</td><td>74.87</td></tr><tr><td>Original</td><td>32</td><td>100%</td><td>73.04</td><td>80.26</td><td>75.92</td></tr><tr><td>V-CoLA w/o exit</td><td>32</td><td>48.39%</td><td>66.96</td><td>78.95</td><td>71.73</td></tr><tr><td>V-CoLA w/ exit</td><td>24</td><td>65.22%</td><td>71.30</td><td>77.63</td><td>73.82</td></tr></table>

End-to-end latency. Practical VLM inference also involves vision encoding, compression overhead and decoding, so we complement the prefill results in Table 2 with an end-to-end (E2E) profile of an INT4 TensorRT build of Qwen3.5-9B. As shown in Table 10, V-CoLA reduces time-to-firsttoken (TTFT) by 38.6–64.6% and E2E latency by up to 1.733×, which is particularly useful for firstresponse-sensitive applications. Since compression targets prefill, the attainable E2E gain scales with the prefill share of the total cost, and is thus smaller on an unoptimized PyTorch backend where prefill accounts for 29.4% and decoding for 58.8% of the E2E latency.

Table 10: End-to-end latency (ms) of INT4 TensorRT Qwen3.5-9B on Jetson Thor with 2K visual input tokens and 32 output tokens. Overhead denotes V-CoLA’s compression cost, which is already included in prefill.
<table><tr><td>Retention</td><td>ViT</td><td>Overhead</td><td>Prefll</td><td>TTFT</td><td>Decode</td><td>E2E</td><td>Speedup</td></tr><tr><td>100%</td><td>243.3</td><td>=</td><td>1233.0</td><td>1476.3</td><td>778.6</td><td>2254.9</td><td>1.000×</td></tr><tr><td>50.0%</td><td>243.3</td><td>0.47</td><td>662.9</td><td>906.2</td><td>778.6</td><td>1684.8</td><td>1.338×</td></tr><tr><td>25.0%</td><td>243.3</td><td>0.48</td><td>402.9</td><td>646.2</td><td>778.6</td><td>1424.8</td><td>1.583×</td></tr><tr><td>12.5%</td><td>243.3</td><td>0.46</td><td>279.6</td><td>522.9</td><td>778.6</td><td>1301.5</td><td>1.733×</td></tr></table>
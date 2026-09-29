# ReSS: Residual-Restoring Sparse Attention for 3D Vision Transformers

Yongsung Kim<sup>1</sup> Jaehoon Lee<sup>1</sup> Minjun Park<sup>1</sup> Wooseok Song<sup>2</sup> Hun Hwangbo<sup>1</sup> Sungroh Yoon<sup>1,2,3†</sup>

<sup>1</sup>IPAI, <sup>2</sup>ECE, <sup>3</sup>AIIS, ASRI, INMC, ISRC

Seoul National University

{libary753, jhcaptain7, minjunpark, cody1129, genchiprofac, sryoon}@snu.ac.kr

## Abstract

3D vision transformers such as VGGT predict camera poses and scene geometryfrom multi-view images in a singleforward pass, but their global attention over all concatenated view tokens dominates computation as the number ofviews grows. To reduce this cost, SparseVGGT and HeSS sparsify attention at the block level, and both retain blocks with high attention probability. However, we observe that attention probability poorly predicts how much the model’s behavior actually changes when a block is removed, and we show that this mismatch is why performance collapses as sparsity increases. In this paper, we propose ReSS (ReSidual-ReStoring Sparse Attention), which recasts block selection from a problem of maximizing the retained attention mass to one of minimizing the drift that sparsification leaves in the residual stream. We introduce a drift score that quantifies how much each block shifts the residual, and, since the drift of a drop set depends on the directions of the contribution vectors rather than on their magnitudes alone, an iterative residual restoration procedure that refines the drop set as a whole. Across three backbones and five datasets, ReSS preserves dense performance better than prior methods at matched sparsity. Two further results support drift as the quantity that governs the cost of sparsification: maximizing drift degrades performance faster than random selection, and plotted against realized drift instead of sparsity, all methods fall approximately onto a single curve. Code is available at https://github.com/libary753/ReSS.

## 1. Introduction

3D vision transformers (3D ViTs) such as VGGT [24], $\pi ^ { 3 }$ [26], and DepthAnything3 [13] predict camera poses and scene geometry from multi-view images in a single forward pass. This capability comes from global attention over the concatenated tokens of all views. For the same reason, however, the attention cost grows quadratically with the number of views and dominates the overall computation precisely when many views are processed at once.

![](images/593166a2afe0cc5b8bd1be8b15aada0c566acbd47d33c638f8ef991928812aa4.jpg)  
Figure 1. Sparse selection by residual restoration. Attention writes its output into the residual stream, and sparsification shifts what it writes; that shift is the residual drift. ReSS restores it by selecting the block set that minimizes this drift.

A common remedy is sparse attention: only a small fraction of query–key pairs contribute meaningfully to the output, so the rest can be skipped with little change; for practical speedups, the skipping is done at the block level. The question is then how to identify the blocks to keep. For 3D ViTs, SparseVGGT [23] first applies block-sparse attention on a pooled block-level attention map, and HeSS [11] adds head-wise budget allocation. Both select them based on attention-probability magnitude, and both degrade rapidly as sparsity increases.

We trace this degradation to probability-based selection itself. When we directly measure how much the attention output changes upon removing a block, attention probability turns out to be a poor predictor of the ranking. The rank correlation fluctuates widely across layers and heads, dropping to near-random in later layers (Sec. 3). In other words, existing methods decide what to keep without looking at the very quantity they aim to preserve: what attention writes into the residual stream. We call the shift that sparsification leaves there the residual drift (Fig. 1).

We propose ReSS (ReSidual-ReStoring Sparse Attention). ReSS recasts block selection as a set selection problem: find the block set that minimizes this drift. We introduce a drift score that quantifies how much removing each block shifts the residual. However, the drift of a drop set depends on the directions of the contribution vectors, not only on their magnitudes, so we pair the score with an iterative residual restoration procedure that refines the drop set as a whole. Each restoration round is fused into a single kernel to suppress its overhead.

Across three backbones and five datasets, and against baselines that include selection criteria ported from longcontext LLM prefill and video diffusion, ReSS preserves dense performance best at matched sparsity. Moreover, maximizing drift by flipping the sign of the objective collapses performance faster than random selection. Re-plotted against the realized drift of each method rather than sparsity, performance falls approximately onto a single curve across methods.

Our contributions are as follows.

• We show that attention probability poorly predicts the output change caused by block removal, and recast block selection as minimizing residual drift over block sets.

• We introduce a drift score for single-block removal and iterative residual restoration that reduces the cumulative drift of the drop set.

• Across three 3D ViT backbones and five datasets, our method best preserves dense performance, in comparisons that include sparse attention methods designed for LLMs and video diffusion as well as for 3D ViTs.

• A sign-flipped control experiment and the collapse of performance onto a single curve as a function of drift support drift as the quantity that determines the cost of sparsification.

## 2. Related Work

Accelerating 3D Vision Transformers Recent 3D ViTs replace multi-stage geometry pipelines [15] with a single feed-forward transformer that regresses camera poses and scene geometry from multi-view images [13, 24–26], at the cost of global attention over the concatenated tokens of all views. Token merging and pruning [2, 4, 17, 19] reduce the number of tokens entering attention, and architectural approaches replace attention with linear-time recurrence [3, 33] or chunked processing [5] at the cost of retraining or pipeline changes; sparse attention instead reduces the attention call itself, runs on released weights, and is complementary to token reduction.

Sparse Attention Block-sparse attention, developed for long-context LLM prefill [10, 12, 20, 28, 31] and extended to video diffusion [27, 29, 32], computes exact attention over a selected subset of key blocks, and selection has relied on fixed structural patterns [10, 27], attention probability mass [12, 20, 28, 31], or the dispersion of attention logits within a block [32]. For 3D ViTs, SparseVGGT [23] keeps blocks with large mass on a pooled block-level attention map via top-k and top-p criteria, and HeSS [11] redistributes block budgets across heads. Whatever the domain, these criteria are statistics of the attention call itself.

![](images/522790c8940c2669b4232c14985546d8e3202f5f9fb2cb27a859e88c0195ad14.jpg)

![](images/ac2dd010f4481e901140b65fd2e4a01433290d796ba5de2a24494e1fde5949d7.jpg)  
Figure 2. Rank agreement with masking-induced residual changes. (Left) Rank of each selection score (vertical) against the rank of the actual residual change caused by masking each block (horizontal); the diagonal indicates perfect agreement. (Right) Rank correlation across global attention layers.

Output-Aware KV Cache Eviction To our knowledge, the closest attempts to score removal by its effect on the output come from KV cache eviction, where recent methods rank each token by the change its eviction induces in the attention output [6–9]. These scores remain per-token and stop at the attention output; ReSS minimizes the drift of the drop set as a whole, measured in the residual stream.

## 3. Motivation

Block-Sparse Attention for 3D ViTs For 3D ViTs, SparseVGGT [23] and HeSS [11] adopt block-sparse attention, which proceeds in three steps: (1) queries and keys are partitioned into fixed-size blocks, and mean pooling within each block yields a coarse block-level attention map; (2) for each query block and head, key blocks are selected based on this map, up to a per-head budget $c _ { h } ;$ (3) exact attention is computed only over the selected blocks, with softmax renormalized among them, so the output behaves as if the unselected blocks never existed. Special tokens such as camera and register tokens aggregate global information and are excluded from sparsification [23]. The block layout of step (1) also affects the approximation [29], but given a layout, its quality is governed by step (2): which blocks to keep. This selection criterion is the subject of this paper.

The Probability Assumption Existing methods, including SparseVGGT and HeSS, uniformly select blocks by attention probability. This choice carries an implicit assumption: the less attention a block receives, the less the model’s output changes when it is removed. The assumption is typically taken for granted, and the criterion has rarely been explored beyond probability.

![](images/96859ccb392767bd7303bb5e01e093c1645c74d21cf38af730454d813f87d9ea.jpg)  
Figure 3. Overview of ReSS. (a) Masking a key block shifts what attention writes into the residual stream. (b) The drift score measures this shift per block; the $c _ { h }$ blocks with the highest scores form the initial selected set. (c) Iterative restoration then reduces the drift of the drop set as a whole, swapping blocks while preserving the budget.

Testing the Probability Assumption We put this assumption to the test by directly measuring the change that masking a single block induces. For each candidate block, we run two forward passes, one with dense attention and one with only that block masked, and take the difference between the resulting attention outputs after the output projection. The magnitude of this difference, a Euclidean norm over the query tokens of the block, is our reference: it is the change that masking actually leaves in the residual stream, measured at token resolution, and its definition involves no selection criterion. We then compare the ranking assigned by attention probability against the ranking induced by this reference, computing the Spearman rank correlation for every (query block, head) pair on six DTU [1] scenes with twelve views each.

Figure 2 shows that attention probability fails to consistently predict this ranking. At layer 18, head 0, its rank correlation is only 0.29; across layers, the correlation fluctuates widely and approaches or even falls below random selection in later layers. Attention probability is thus an incomplete proxy for block importance.

The Right Objective Since probability fails to predict the change, we take the change itself as the objective. Block masking changes only what the attention sublayer adds to the residual stream; the entire cost of sparsification is a shift in this contribution. We call this shift the residual drift. Writing the pre-LN structure as ${ \boldsymbol { x } } _ { \ell + 1 } = { \boldsymbol { x } } _ { \ell } + { f } _ { \ell } ( { \boldsymbol { x } } _ { \ell } )$ , with $x _ { \ell }$ the input to layer ℓ and $f _ { \ell }$ its sublayer, the drift injected at layer ℓ propagates as a change in the input to every subsequent layer. To keep the output of the sparse model close to that of the dense model, block selection should therefore minimize the residual drift. This is the selection objective of this paper.

## 4. Method

We propose ReSS (Residual-Restoring Sparse Attention), a block selection method that minimizes residual drift. The overall pipeline is shown in Fig. 3.

## 4.1. Problem Formulation

$P _ { i j } ^ { h }$ denotes the pooled attention probability from query block i to key block j in head h, and $V _ { j } ^ { h }$ the corresponding pooled value. Let B be the set of all key blocks, over which the pooled attention map is normalized, so that the dense pooled attention output is

$$
\bar { v } _ { i } ^ { h } = \sum _ { j \in { \cal { B } } } { \cal P } _ { i j } ^ { h } V _ { j } ^ { h } .\tag{1}
$$

Blocks containing special tokens are always kept (Sec. 3); the remaining blocks, consisting only of patch keys, form the candidate set $\mathcal { P } \subseteq B$

For each query block i and head h, we formulate block selection as choosing a drop set $\mathcal { D } _ { i } ^ { h } \subseteq \mathcal { P }$ , whose complement $S _ { i } ^ { h } = \mathcal { P } \setminus \mathcal { D } _ { i } ^ { h }$ is the selected set:

$$
\operatorname* { m i n } _ { \mathcal { D } _ { i } ^ { h } \subseteq \mathcal { P } , \mathcal { | D } _ { i } ^ { h } | = | \mathcal { P } | - c _ { h } } \ J _ { i } ^ { h } ( \mathcal { D } _ { i } ^ { h } ) ,\tag{2}
$$

where $J _ { i } ^ { h }$ is the squared magnitude of the drift that the attention sublayer leaves in the residual stream when $\mathcal { D } _ { i } ^ { h }$ is removed, and $c _ { h }$ is the per-head budget; the constraint gives $| S _ { i } ^ { h } | = c _ { h }$ . The problem is solved independently for each (query block, head) pair.

Equation (2) is a combinatorial optimization over subsets of P, whose search space grows exponentially with the number of candidates; solving it exactly within a forward pass at every layer is infeasible. ReSS approximates it in two stages: Sec. 4.2 constructs an initial selected set from a per-block drift score (Fig. 3b), and Sec. 4.3 directly lowers $J _ { i } ^ { h }$ through iterative restoration (Fig. 3c).

## 4.2. Block Drift Score

Drift induced by block masking. We first consider masking a single key block $j .$ Since softmax is renormalized over the remaining blocks, the attention output after masking is

$$
\tilde { v } _ { i j } ^ { h } = \sum _ { k \in \mathcal { B } , k \neq j } \frac { P _ { i k } ^ { h } } { 1 - P _ { i j } ^ { h } } V _ { k } ^ { h } = \frac { \bar { v } _ { i } ^ { h } - P _ { i j } ^ { h } V _ { j } ^ { h } } { 1 - P _ { i j } ^ { h } } ,\tag{3}
$$

and the resulting output change is exactly

$$
\tilde { v } _ { i j } ^ { h } - \bar { v } _ { i } ^ { h } = \frac { P _ { i j } ^ { h } } { 1 - P _ { i j } ^ { h } } \left( \bar { v } _ { i } ^ { h } - V _ { j } ^ { h } \right) .\tag{4}
$$

This change can be decomposed into three factors: the deviation $e _ { i j } ^ { h } = { \bar { v } } _ { i } ^ { h } - V _ { j } ^ { h }$ of the block’s value from the dense output, the attention probability $P _ { i j } ^ { h }$ , and the factor $1 / ( 1 - P _ { i j } ^ { h } )$ from softmax renormalization. For the subsequent development, we set the renormalization factor aside and define the unnormalized drift contribution

$$
\begin{array} { r } { d _ { i j } ^ { h } = P _ { i j } ^ { h } e _ { i j } ^ { h } . } \end{array}\tag{5}
$$

Mapping to the residual space. The attention output enters the residual stream through the output projection, so the impact of masking must be measured after $\bar { W } _ { O } ^ { h } \in \mathbb { R } ^ { d \times d _ { h } }$ the slice of the output projection for head h, with residual dimension d and head dimension $d _ { h }$ . We define the unnormalized block drift score

$$
s _ { i j } ^ { h } = \left\| \boldsymbol W _ { O } ^ { h } \boldsymbol d _ { i j } ^ { h } \right\| _ { 2 } = P _ { i j } ^ { h } \left\| \boldsymbol W _ { O } ^ { h } \boldsymbol e _ { i j } ^ { h } \right\| _ { 2 } .\tag{6}
$$

For selection, we restore the renormalization factor and keep the $c _ { h }$ blocks with the largest exact single-block drift magnitude,

$$
\frac { s _ { i j } ^ { h } } { \operatorname* { m a x } ( 1 - P _ { i j } ^ { h } , \epsilon ) } , \qquad \epsilon = 1 0 ^ { - 4 } .\tag{7}
$$

Efficient score computation. Computing Eq. (6) directly materializes $W _ { O } ^ { h } e _ { i j } ^ { h }$ for every head and query–key block pair, a prohibitive intermediate of size $( n _ { h } , q _ { \mathrm { b l k } } , k _ { \mathrm { b l k } } , d )$ over $n _ { h }$ heads, $q _ { \mathrm { b l k } }$ query blocks, and $k _ { \mathrm { b l k } }$ key blocks. We instead precompute the per-head Gram matrix

$$
M _ { h } = ( W _ { O } ^ { h } ) ^ { \top } W _ { O } ^ { h } \in \mathbb { R } ^ { d _ { h } \times d _ { h } }\tag{8}
$$

once at model load time and expand the squared projected deviation as

$$
\begin{array} { r l } & { q _ { i j } ^ { h } = \left\| \boldsymbol { W } _ { O } ^ { h } \boldsymbol { e } _ { i j } ^ { h } \right\| _ { 2 } ^ { 2 } = ( \boldsymbol { e } _ { i j } ^ { h } ) ^ { \top } M _ { h } \boldsymbol { e } _ { i j } ^ { h } } \\ & { \quad \quad = ( \bar { \boldsymbol { v } } _ { i } ^ { h } ) ^ { \top } M _ { h } \bar { \boldsymbol { v } } _ { i } ^ { h } - 2 ( \bar { \boldsymbol { v } } _ { i } ^ { h } ) ^ { \top } M _ { h } \boldsymbol { V } _ { j } ^ { h } + \boldsymbol { V } _ { j } ^ { h } ^ { \top } M _ { h } \boldsymbol { V } _ { j } ^ { h } , } \end{array}\tag{9}
$$

three low-dimensional contractions with no per-pair vector intermediate; $s _ { i j } ^ { h } = P _ { i j } ^ { h } \sqrt { q _ { i j } ^ { h } }$ then recovers Eq. (6) exactly.

## 4.3. Iterative Residual Restoration

The initial selection of Eq. (7) evaluates each block independently. However, the drift of removing multiple blocks is determined by the sum of their contribution vectors, in which opposing directions partially offset. ReSS therefore refines the initial selection iteratively, lowering the drift of the drop set as a whole (Fig. 3c).

Net drift of the drop set. Let D be the current drop set. We define the net drift and the remaining attention probability mass as

$$
g _ { i } ^ { h } = \sum _ { j \in \mathcal { D } } d _ { i j } ^ { h } , \qquad K _ { i } ^ { h } = 1 - \sum _ { j \in \mathcal { D } } { P } _ { i j } ^ { h } .\tag{10}
$$

Since $\mathcal { D } \subseteq \mathcal { P }$ and the blocks in $B \setminus \mathcal { P }$ are always kept, $K _ { i } ^ { h }$ stays away from zero. The attention output after masking all blocks in $\mathcal { D }$ is

$$
\tilde { v } _ { i } ^ { h } = \bar { v } _ { i } ^ { h } + \frac { g _ { i } ^ { h } } { K _ { i } ^ { h } } ,\tag{11}
$$

and the resulting squared residual drift is

$$
J _ { i } ^ { h } ( { \mathcal { D } } ) = \left\|  W _ { O } ^ { h } \frac { g _ { i } ^ { h } } { K _ { i } ^ { h } } \right\| _ { 2 } ^ { 2 } = \frac { ( g _ { i } ^ { h } ) ^ { \top } M _ { h } g _ { i } ^ { h } } { ( K _ { i } ^ { h } ) ^ { 2 } } .\tag{12}
$$

This is the objective of Eq. (2). We denote its numerator by $G _ { i } ^ { h } \ = \ ( g _ { i } ^ { h } ) ^ { \top } M _ { h } g _ { i } ^ { h }$ , the squared magnitude of the net contribution measured in the residual space. For $\mathcal { D } = \{ j \}$ Eq. (12) reduces to the square of the ranking key in Eq. (7); the initial selection and the iterative refinement thus minimize the same objective at different set sizes, rather than two different ones.

Budget-neutral swaps. Starting from the initial drop set, ReSS repeats budget-neutral swaps: each round first restores the blocks that most reduce the current net drift, then drops an equal number of blocks to preserve the budget.

Restoring a block $k \in \mathcal { D }$ updates

$$
g _ { i } ^ { h }  g _ { i } ^ { h } - d _ { i k } ^ { h } , \qquad K _ { i } ^ { h }  K _ { i } ^ { h } + P _ { i k } ^ { h } ,\tag{13}
$$

and changes the numerator of Eq. (12) by

$$
\Delta G _ { k } ^ { + } = - 2 ( d _ { i k } ^ { h } ) ^ { \top } M _ { h } g _ { i } ^ { h } + ( s _ { i k } ^ { h } ) ^ { 2 } .\tag{14}
$$

Conversely, dropping a kept block $k \notin$ D changes it by

$$
\Delta G _ { k } ^ { - } = 2 ( d _ { i k } ^ { h } ) ^ { \top } M _ { h } g _ { i } ^ { h } + ( s _ { i k } ^ { h } ) ^ { 2 } .\tag{15}
$$

Denoting the change in $G _ { i } ^ { h }$ from one update by $\Delta G$ and that in $K _ { i } ^ { h }$ by $u \left( u \right) = + P _ { i k } ^ { h }$ for a restore, $- P _ { i k } ^ { h }$ for a drop), the exact decrease of the objective is

$$
\Delta J = \frac { G _ { i } ^ { h } } { ( K _ { i } ^ { h } ) ^ { 2 } } - \frac { G _ { i } ^ { h } + \Delta G } { ( K _ { i } ^ { h } + u ) ^ { 2 } } .\tag{16}
$$

![](images/a0cc780bf29eb52a2af0f54a6a85cd43494ea7b525266563c537ec2351ac47c3.jpg)  
Figure 4. Qualitative results. Two DTU scans reconstructed at five sparsity levels, on VGGT [24] and π<sup>3</sup> [26]. Columns are the measured sparsity, matched across methods. Points are drawn in green where their distance to the ground truth exceeds 5 mm, and each panel reports the fraction of such points.

In each round, we select the blocks to restore and, in compensation, the blocks to drop using Eq. (16).

The swap budget is halved every round. Setting the total budget to a fraction $\rho$ of the keep budget, round r performs

$$
m _ { r } = \rho 2 ^ { - ( r + 1 ) } c _ { h } , \qquad r = 0 , \ldots , R - 1 ,\tag{17}
$$

swaps, for a total of $\rho ( 1 - 2 ^ { - R } ) c _ { h }$ over R rounds. We use $\rho = 0 . 5$ and $R = 5 $ , giving 0.484 $c _ { h }$ swaps in total. The budget is proportional to the keep budget $c _ { h }$ rather than the drop set size: once sparsity exceeds $2 / 3 , | \mathcal { D } | > 2 c _ { h }$ , and a budget proportional to $| \mathcal D |$ could replace the entire kept set. Throughout the iteration, ReSS tracks the drop set with the smallest objective Eq. (12) observed so far and returns it as the final selection. The objective after refinement therefore never exceeds that of the initial selection.

Fused kernel implementation. Each restoration round launches many small operations over $( n _ { h } , q _ { \mathrm { b l k } } , k _ { \mathrm { b l k } } )$ tensors; each is cheap in FLOPs but incurs a separate round trip to memory, so the round is bound by memory bandwidth rather than computation. We therefore fuse the round body into a single Triton [21] kernel: each query row is processed in registers in one sweep, with no intermediate tensor written to memory and the same selection as the unfused path, bit for bit. Its effect on the selection cost is reported in Tab. 1, and implementation details are in Sec. E.

## 4.4. Pipeline Configuration

Sparsity control. We control sparsity with the top-k budget alone. Unlike a CDF threshold, whose keep count varies with the data, a fixed budget makes the realized sparsity match its target; removing the threshold has no consistent effect on accuracy (Sec. 6.2), and all experiments are compared at matched measured sparsity.

Head-wise budget allocation. ReSS assumes the perhead budgets $\left\{ c _ { h } \right\}$ as given and is compatible with any allocation scheme. Unless noted otherwise, we adopt the headwise allocation of HeSS [11] (see Sec. D); to isolate its effect, we also evaluate a ReSS variant with uniform budgets.

## 5. Experiments

## 5.1. Experimental Setup

Backbones. We evaluate on three feed-forward 3D ViTs: VGGT [24], $\pi ^ { 3 }$ [26], and DepthAnything3 [13]. All three alternate per-frame attention with global attention over the concatenated tokens of every view, and ReSS sparsifies only the global layers: 24 of 48 in VGGT, 18 of 36 in $\pi ^ { 3 }$ , and 14 of 40 in DepthAnything3. None is retrained; ReSS runs on the released weights. The main text reports VGGT and $\pi ^ { 3 }$ DepthAnything3 results are in Sec. B.

![](images/c89e4e0f2a2ff060ee0c548781c2ff98d7a3b7ffdccf40af9c4353d15f0d2fc9.jpg)  
Figure 5. Quantitative results. Reconstruction (recon) and camera pose estimation (pose) across sparsity levels, on the VGGT [24] (top) and $\pi ^ { 3 }$ [26] (bottom) backbones. Dashed lines denote dense attention; arrows mark whether higher (↑) or lower (↓) is better. As sparsity grows, ReSS preserves dense performance better than SparseVGGT [23] and HeSS [11].

Baselines. Our main baselines are SparseVGGT [23] and HeSS [11], the block-sparse attention methods proposed for 3D ViTs. We additionally port six sparse attention methods from other domains to VGGT: XAttention [28], SpargeAttn [31], and FlexPrefill [12] from LLM long-context prefill, and SVG [27], SVG2 [29], and SVG-EAR [32] from video diffusion. The LLM methods share our block configuration and kernel, differing only in the selection criterion; the video methods keep their own block layouts (Sec. F, Sec. G).

Benchmarks. We measure 3D geometry performance on DA3-Bench, the evaluation pipeline of DepthAnything3 [13], which covers HiRoom [13], 7-Scenes [18], DTU [1], ETH3D [16], and ScanNet++ [30], evaluating camera pose estimation and multi-view geometry on each. Each scene is evaluated on all of its frames, subsampled to at most 100: HiRoom has 10–23 views, ETH3D 14– 76, DTU 49, and ScanNet++ and 7-Scenes reach the cap. To measure the degradation precisely across sparsity levels, we add an alignment step based on dense point correspondences to the evaluation pipeline (Sec. C).

Metrics. Multi-view geometry is reported as Chamfer distance (↓) on DTU and F1 score (↑) elsewhere, with distance threshold $\tau = 0 . 2 5$ m on ETH3D and 0.05 m on the other datasets. Camera pose is reported as AUC@30<sup>◦</sup> (↑) over relative rotation and translation errors. Exact definitions follow DA3-Bench [13].

## 5.2. Qualitative Results

Figure 4 shows reconstructions of two DTU scans at five sparsity levels, with points more than 5 mm from the ground truth drawn in green. As sparsity grows, the baselines progressively lose the scene structure, while ReSS keeps it largely intact at every level. The contrast is sharpest at the highest sparsity on $\pi ^ { 3 } { \mathrm { : } }$ on the right scan, errors spread across the scene for SparseVGGT and HeSS, with 23.32% and 23.21% of points over the threshold, while ReSS stays at 2.25%. The reconstructions thus make visible the gap that Sec. 5.3 quantifies.

## 5.3. Quantitative Results

Figure 5 shows performance across sparsity levels; dashed lines mark dense performance. All methods match dense at low sparsity, but as compression grows, SparseVGGT and HeSS depart from the dashed lines while ReSS stays on them up to sparsity 0.4. The gap is widest at the highest operating point: on $\pi ^ { 3 } / \mathrm { D T U }$ , the baselines’ Chamfer distance exceeds twice the dense value while ReSS remains near dense (1.067 vs. 0.838), and on π<sup>3</sup>/HiRoom, F1 falls to 0.378 and 0.424 while ReSS keeps 0.850 (dense 0.908). Pose degrades more gently but in the same order. At the highest operating point, ReSS is best in both reconstruction and pose in all ten backbone–dataset combinations, and a similar trend holds on DepthAnything3 (Sec. B.2).

![](images/178fd12024482b7bcbba641161f4399050e016e064f0b223b589e89fd894b9bc.jpg)  
Figure 6. Quality against latency. Each point is one operating point on ScanNet++ with the VGGT backbone; the dashed line marks dense attention. Latency covers the full forward pass including all selection overhead. LLM criteria reach dense quality only at dense latency or beyond; video methods lose either quality (SVG) or speed (SVG2, SVG-EAR).

## 5.4. Sparse Attention from Other Domains

Fig. 6 compares the six ported methods against ReSS in the latency–quality plane. The three LLM criteria trace the same shape: all collapse in the compression regime, where speed gains actually appear. For XAttention and FlexPrefill the masks show why: each sizes its mask by a threshold on attention mass alone, so at matched sparsity the loss concentrates on a few rows rather than spreading evenly (Sec. F).

Video diffusion appears the closer neighbor, as both consume long token sequences from many images, yet the transfer fails on two counts. First, the assumptions do not carry: the fixed bands of SVG assume adjacent frames share nearly the same scene, but adjacent views are much farther apart even on ScanNet++, where the view order follows a capture trajectory, and SVG reaches dense quality at no operating point. Second, the costs do not amortize: the clustering of SVG2 and SVG-EAR carries over, and SVG-EAR retains dense quality even at high sparsity, yet both run slower than dense attention, since re-planning the block layout at every layer is a fixed cost that diffusion spreads over tens of denoising steps and a single forward pass pays in full (Sec. G).

## 6. Analysis

## 6.1. Does Drift Explain the Cost of Sparsification?

We examine the premise of ReSS from three directions: whether ReSS actually reduces drift, whether reducing drift preserves quality, and what reducing it costs.

![](images/93f59e831a75dc88c1e276132d9879f309f0044001c4556cb12a901b31e171be.jpg)  
Figure 7. Number of restoration rounds. Quality saturates after three to four rounds, while the realized drift keeps decreasing beyond that point. The grey line marks the setting we adopt. Dashed line denotes dense attention.

![](images/af5bf2a89669ce3dabc8eb0a34a6254d062b1e7944ba498fbae77cea1233dae5.jpg)  
Figure 8. Drift minimization vs maximization. With the budget and sparsity fixed, we flip the sign of the objective, either only in the restoration stage (swap only) or from the initial selection (full). Maximizing drift collapses performance faster than random selection, supporting drift as a valid selection criterion.

Does ReSS reduce drift? Figure 7 varies the number of restoration rounds from 0 to 10 with all else fixed. The realized drift J falls by more than an order of magnitude from the initial selection, confirming that the iterative refinement optimizes what it claims to.

Does reducing drift preserve quality? Figure 8 flips the sign of the objective with the budget and sparsity fixed: maximizing drift collapses performance even at mild sparsity, far below random selection. Figure 9 plots quality against the realized drift of each selection, and all methods land on nearly the same curve, the drift-maximizing variants included. Methods that differ widely at equal sparsity

Table 1. Cost analysis. Selection cost at 100 views on ScanNet++, sparsity 0.745. Parenthesized: overhead over SparseVGGT [23]. RTX 4090, bfloat16.
<table><tr><td></td><td>Latency (s)</td><td>Scoring mem. Peak mem. (MiB)</td><td>(GiB)</td></tr><tr><td>SparseVGGT</td><td>7.58</td><td></td><td></td></tr><tr><td>ReSS (full)</td><td>7.82 (+0.24)</td><td>205</td><td>16.4</td></tr><tr><td>w/o fused kernel 8.84 (+1.26)</td><td></td><td>205</td><td>16.4</td></tr><tr><td>w/o expansion</td><td>8.48 (+0.90)</td><td>7760</td><td>22.4</td></tr><tr><td>w/o cached  $M _ { h }$ </td><td>7.87(+0.29)</td><td>278</td><td>16.4</td></tr></table>

![](images/23922072b7a6c89c2d66bd6576c8112e6b9d24f1a9d4a98fa5510df9d1f5d519.jpg)  
Figure 9. Drift determines quality. Performance plotted against the realized drift J of each selection instead of sparsity, on the VGGT [24] backbone over DTU and ETH3D. Different methods, including the drift-maximizing variants, collapse onto a single curve.

nearly coincide at equal drift.

Together, the two results say more than that minimizing drift works: performance tracks the drift a selection leaves, regardless of how the selection is made.

What does reducing drift cost? With R = 5 rounds, the entire selection path, scoring and restoration included, adds 0.24 s over SparseVGGT and 205 MiB of scoring memory at sparsity 0.745. Five rounds suffice: quality saturates after three to four rounds while each additional round adds a near-constant latency (Fig. 7). The cost stays this low because of two reformulations. Expanding the quadratic form in Eq. (9) removes the per-pair $d _ { h } .$ -dimensional intermediate, which otherwise grows the scoring allocation 38× and raises peak memory from 16.4 to 22.4 GiB. Keeping each round in registers with a fused kernel accounts for the latency: without it, the overhead grows about 5×, to +1.26 s. Table 1 also reports the smaller saving from caching $M _ { h }$

Table 2. Ablation study. Components are removed (−) or added (+) cumulatively, one per row, from ReSS down to Sparse-VGGT [23]; the last row coincides with SparseVGGT. VGGT [24] backbone, all rows matched at the highest operating point.
<table><tr><td rowspan="2"></td><td colspan="2">Reconstruction</td><td colspan="2">Pose (AUC@30°)</td></tr><tr><td>DTU↓</td><td>ETH3D↑</td><td>DTU↑</td><td>ETH3D↑</td></tr><tr><td>ReSS (full)</td><td>0.791</td><td>0.743</td><td>0.996</td><td>0.795</td></tr><tr><td>– head budget</td><td>0.983</td><td>0.696</td><td>0.980</td><td>0.762</td></tr><tr><td>— restoration</td><td>1.151</td><td>0.582</td><td>0.980</td><td>0.702</td></tr><tr><td>– drift score</td><td>1.481</td><td>0.581</td><td>0.958</td><td>0.643</td></tr><tr><td>+ CDF threshold</td><td>1.338</td><td>0.538</td><td>0.963</td><td>0.623</td></tr></table>

## 6.2. Ablation Study

Table 2 strips ReSS down to SparseVGGT one component at a time, cumulatively, so that the last row coincides with SparseVGGT; all rows are matched at a measured sparsity of 0.73 on the VGGT backbone.

Each row tests one claim. Taking away head-wise budget allocation costs a consistent margin on all four metrics, showing that the criterion benefits from, but does not depend on, how the budget is distributed across heads. Removing restoration costs the largest margin on ETH3D, in both reconstruction and pose, which follows from its premise: the drift of a drop set is the sum of contribution vectors, so independent scoring leaves cancellation that only set-level refinement recovers (Sec. 4.3). Falling back from the drift score to probability selection costs the largest margin on DTU, in both reconstruction and pose, without changing how many blocks are kept, only which (Sec. 4.2). Restoring the CDF threshold on top changes results in neither direction consistently, confirming it can be dropped for simpler sparsity control without paying in quality (Sec. 4.4).

Every component contributes, including the two that embody our claim: drift scoring and set-level restoration.

## 7. Conclusion

We recast block selection in 3D vision transformers, from keeping blocks with high attention probability to minimizing the drift that sparsification leaves in the residual stream. ReSS realizes this with a drift score for single-block removal and iterative residual restoration that reduces the cumulative drift of the drop set. Across three backbones and five datasets, ReSS preserves dense performance better than prior methods at matched sparsity; flipping the sign of the objective shows that this gain comes from the drift criterion itself, and performance across methods collapses onto a single curve as a function of the drift each leaves. Sparsity says how much attention is skipped; the drift left in the residual stream says what that skipping costs, and selection should be designed, and perhaps evaluated, in that currency.

## Acknowledgements

This work was supported by Samsung Electronics Co., Ltd [No. IO260120-15267-01]; Information & communications Technology Planning & Evaluation (IITP) grant funded by the Korea government (MSIT) [No. RS-2021- II211343; RS-2022-II220959; RS-2025-02263754; Artificial Intelligence Graduate School Program (Seoul National University)]; the National Research Foundation of Korea (NRF) grant funded by MSIT [No. 2022R1A3B1077720; 2022R1A5A7083908] and the BK21 Four program of the Education and Research Program for Future ICT Pioneers, SNU in 2026. This research was also conducted as part of the Sovereign AI Foundation Model Project (Data Track), organized by MSIT and supported by the National Information Society Agency (NIA) of Korea [No. 2025-AI Datawi43].

## References

[1] Henrik Aanæs, Rasmus Ramsbøl Jensen, George Vogiatzis, Engin Tola, and Anders Bjorholm Dahl. Large-scale data for multiple-view stereopsis. International Journal ofComputer Vision, 120(2):153–168, 2016. 3, 6, 1

[2] Daniel Bolya, Cheng-Yang Fu, Xiaoliang Dai, Peizhao Zhang, Christoph Feichtenhofer, and Judy Hoffman. Token merging: Your ViT but faster. In International Conference on Learning Representations, 2023. 2

[3] Xingyu Chen, Yue Chen, Yuliang Xiu, Andreas Geiger, and Anpei Chen. TTT3R: 3D reconstruction as test-time training. In International Conference on Learning Representations, 2026. 2

[4] Yutian Chen, Yuheng Qiu, Ruogu Li, Jay Patrikar, and Sebastian Scherer. Co-Me: Confidence guided token merging for visual geometric transformers. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 14590–14599, 2026. 2

[5] Kai Deng, Zexin Ti, Jiawei Xu, Jian Yang, and Jin Xie. VGGT-Long: Chunk it, loop it, align it–pushing VGGT’s limits on kilometer-scale long RGB sequences. arXiv preprint arXiv:2507.16443, 2025. 2

[6] Yuan Feng, Junlin Lv, Haoyu Guo, Yukun Cao, S Kevin Zhou, and Xike Xie. CriticalKV: Optimizing KV cache eviction from an output perturbation perspective. In Forty-third International Conference on Machine Learning, 2026. 2

[7] Raghavv Goel, Junyoung Park, Mukul Gagrani, Dalton Jones, Matthew J Morse, Matthew Harper Langston, Christopher Lott, and Mingu Lee. CAOTE: Optimizing KV cache memory through attention output error-based token eviction. In ICLR 2026 Workshop on Memoryfor LLM-Based Agentic Systems, 2026.

[8] Yuzhe Gu, Xiyu Liang, Jiaojiao Zhao, and Enmao Diao. OB-Cache: Optimal brain KV cache pruning for efficient longcontext LLM inference. arXiv preprint arXiv:2510.07651, 2025.

[9] Zhiyu Guo, Hidetaka Kamigaito, and Taro Watanabe. Attention score is not all you need for token importance indicator in KV cache reduction: Value also matters. In Proceedings of

the 2024 Conference on Empirical Methods in Natural Language Processing, pages 21158–21166, 2024. 2

[10] Huiqiang Jiang, Yucheng Li, Chengruidong Zhang, Qianhu Wu, Xufang Luo, Surin Ahn, Zhenhua Han, Amir H Abdi, Dongsheng Li, Chin-Yew Lin, et al. MInference 1.0: Accelerating pre-filling for long-context LLMs via dynamic sparse attention. Advances in Neural Information Processing Sys tems, 37:52481–52515, 2024. 2

[11] Yongsung Kim, Wooseok Song, Jaihyun Lew, Hun Hwangbo, Jaehoon Lee, and Sungroh Yoon. HeSS: Head sensitivity score for sparsity redistribution in VGGT. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2026. 1, 2, 5, 6

[12] Xunhao Lai, Jianqiao Lu, Yao Luo, Yiyuan Ma, and Xun Zhou. FlexPrefill: A context-aware sparse attention mechanism for efficient long-sequence inference. In The Thirteenth International Conference on Learning Representa tions, 2025. 2, 6, 4, 5

[13] Haotong Lin, Sili Chen, Jun Hao Liew, Donny Y. Chen, Zhenyu Li, Yang Zhao, Sida Peng, Hengkai Guo, Xiaowei Zhou, Guang Shi, Jiashi Feng, and Bingyi Kang. Depth Anything 3: Recovering the visual space from any views. In The Fourteenth International Conference on Learning Represen tations, 2026. 1, 2, 5, 6

[14] Jeremy Reizenstein, Roman Shapovalov, Philipp Henzler, Luca Sbordone, Patrick Labatut, and David Novotny. Common objects in 3D: Large-scale learning and evaluation of real-life 3D category reconstruction. In International Conference on Computer Vision, 2021. 2

[15] Johannes L Schonberger and Jan-Michael Frahm. Structure-¨ from-motion revisited. In Proceedings of the IEEE conference on computer vision and pattern recognition, pages 4104–4113, 2016. 2

[16] Thomas Schops, Johannes L Sch¨ onberger, Silvano Galliani,¨ Torsten Sattler, Konrad Schindler, Marc Pollefeys, and An dreas Geiger. A multi-view stereo benchmark with highresolution images and multi-camera videos. In Proceedings ofthe IEEE conference on computer vision and pattern recognition, pages 3260–3269, 2017. 6, 1

[17] You Shen, Zhipeng Zhang, Yansong Qu, Xiawu Zheng, Jiay Ji, Shengchuan Zhang, and Liujuan Cao. FastVGGT: Fast visual geometry transformer. In International Conference on Learning Representations, 2026. 2

[18] Jamie Shotton, Ben Glocker, Christopher Zach, Shahram Izadi, Antonio Criminisi, and Andrew Fitzgibbon. Scene coordinate regression forests for camera relocalization in RGB-D images. In Proceedings of the IEEE conference on computer vision and pattern recognition, pages 2930–2937, 2013. 6

[19] Zhijian Shu, Cheng Lin, Tao Xie, Wei Yin, Ben Li, Zhiyuan Pu, Weize Li, Yao Yao, Xun Cao, Xiaoyang Guo, et al. LiteVGGT: Boosting vanilla VGGT via geometry-aware cached token merging. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 36422–36432, 2026. 2

[20] Jiaming Tang, Yilong Zhao, Kan Zhu, Guangxuan Xiao, Baris Kasikci, and Song Han. Quest: Query-aware sparsity

for efficient long-context LLM inference. In International Conference on Machine Learning (ICML), 2024. 2

[21] Philippe Tillet, Hsiang-Tsung Kung, and David Cox. Triton: an intermediate language and compiler for tiled neural network computations. In Proceedings of the 3rd ACM SIGPLAN International Workshop on Machine Learning and Programming Languages, pages 10–19, 2019. 5

[22] Shinji Umeyama. Least-squares estimation of transformation parameters between two point patterns. IEEE Transactions on pattern analysis and machine intelligence, 13(4): 376–380, 1991. 1

[23] Chung-Shien Brian Wang, Christian Schmidt, Jens Piekenbrinck, and Bastian Leibe. Block-sparse global attention for efficient multi-view geometry transformers. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 14546–14555, 2026. 1, 2, 6, 8, 5

[24] Jianyuan Wang, Minghao Chen, Nikita Karaev, Andrea Vedaldi, Christian Rupprecht, and David Novotny. VGGT: Visual geometry grounded transformer. In 2025 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 5294–5306. IEEE, 2025. 1, 2, 5, 6, 8, 4

[25] Shuzhe Wang, Vincent Leroy, Yohann Cabon, Boris Chidlovskii, and Jerome Revaud. DUSt3R: Geometric 3D vision made easy. In 2024 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 20697– 20709. IEEE, 2024.

[26] Yifan Wang, Jianjun Zhou, Haoyi Zhu, Wenzheng Chang, Yang Zhou, Zizun Li, Junyi Chen, Jiangmiao Pang, Chunhua Shen, and Tong He. π<sup>3</sup>: Permutation-equivariant visual geometry learning. In International Conference on Learning Representations, 2026. 1, 2, 5, 6

[27] Haocheng Xi, Shuo Yang, Yilong Zhao, Chenfeng Xu, Muyang Li, Xiuyu Li, Yujun Lin, Han Cai, Jintao Zhang, Dacheng Li, Jianfei Chen, Ion Stoica, Kurt Keutzer, and Song Han. Sparse Video-Gen: Accelerating video diffusion transformers with spatial-temporal sparsity. In Forty-second International Conference on Machine Learning, 2025. 2, 6

[28] Ruyi Xu, Guangxuan Xiao, Haofeng Huang, Junxian Guo, and Song Han. XAttention: Block sparse attention with antidiagonal scoring. In Proceedings of the 42nd International Conference on Machine Learning (ICML), 2025. 2, 6, 4, 5

[29] Shuo Yang, Haocheng Xi, Yilong Zhao, Muyang Li, Jintao Zhang, Han Cai, Yujun Lin, Xiuyu Li, Chenfeng Xu, Kelly Peng, et al. Sparse VideoGen2: Accelerate video generation with sparse attention via semantic-aware permutation. Advances in Neural Information Processing Systems, 38:96965–96991, 2025. 2, 6

[30] Chandan Yeshwanth, Yueh-Cheng Liu, Matthias Nießner, and Angela Dai. ScanNet++: A high-fidelity dataset of 3D indoor scenes. In 2023 IEEE/CVF International Conference on Computer Vision (ICCV), pages 12–22. IEEE, 2023. 6, 1

[31] Jintao Zhang, Chendong Xiang, Haofeng Huang, Jia Wei, Haocheng Xi, Jun Zhu, and Jianfei Chen. SpargeAttn: Accurate sparse attention accelerating any model inference. In International Conference on Machine Learning (ICML), 2025. 2, 6, 4, 5

[32] Xuanyi Zhou, Qiuyang Mang, Shuo Yang, Haocheng Xi, Jintao Zhang, Huanzhi Mao, Joseph E Gonzalez, Kurt Keutzer,

Ion Stoica, and Alvin Cheung. SVG-EAR: Parameter-free linear compensation for sparse video generation via erroraware routing. arXiv preprint arXiv:2603.08982, 2026. 2, 6

[33] Dong Zhuo, Wenzhao Zheng, Jiahe Guo, Yuqi Wu, Jie Zhou, and Jiwen Lu. Streaming visual geometry transformer. In In ternational Conference on Learning Representations, 2026. 2

# ReSS: Residual-Restoring Sparse Attention for 3D Vision Transformers

Supplementary Material

## A. Limitation

The objective of ReSS is to minimize drift from the dense computation, and this takes the dense output as an attainable ceiling. The experiments show that the premise does not always hold (Sec. B.2). In some cases sparsification improves on dense, which means that the drift introduced at the selection step can act, as it passes through the subsequent layers of the model, in a direction that raises quality. The current formulation addresses only the magnitude of drift and does not ask whether its direction is beneficial or harmful, so ReSS stays near the dense result by design and forgoes these gains. Distinguishing beneficial from harmful drift at selection time, that is, sparsification that improves on the dense computation rather than restoring it, is worth pursuing as future work.

## B. Additional Experimental Results

## B.1. Additional Results for VGGT [24] and $\pi ^ { 3 }$

Tab. S1 gives the absolute numbers behind Fig. 5 at three of its five operating points. Fig. S3 and Fig. S4 extend the qualitative comparison of Fig. 4 to two further DTU [1] scans and to ETH3D [16], HiRoom [13] and ScanNet++ [30].

## B.2. DepthAnything3

Fig. S1 and Tab. S2 report the same sweep on DepthAnything3 [13]: the same five datasets, the same five operating points, the same sparsity-matching procedure. Sparsification is applied to the 14 of its 40 blocks that perform cross-view attention.

The trend is the one in the main paper, with smaller margins between methods than on the other two backbones. At the highest operating point, ReSS gives the best reconstruction on every dataset except ETH3D, where HeSS is marginally ahead. At lower sparsity the methods lie within a small margin of one another, and where one pulls ahead, it is often by scoring above dense, that is, sparsification happens to improve on the dense computation. The one clear exception is HiRoom at intermediate sparsity, where SparseVGGT stays closer to dense than ReSS. ReSS minimizes drift away from the dense computation, so it stays near the dense result and does not gain from departures that improve on it; Sec. A takes up this limitation.

## B.3. The Contribution of Track-best

The objective does not necessarily decrease monotonically over restoration rounds, so track-best keeps the best drop set seen so far instead of the last iterate. Tab. S3 measures its contribution at two round counts. At 5 rounds it improves Chamfer from 0.8403 to 0.7910; at 10 rounds it moves the result by only 0.002. Five rounds with track-best already match ten without it (0.7910 against 0.7952): given enough rounds the last iterate is itself a good set, and remembering the best one has little left to add. Track-best therefore does not raise the attainable quality; it reaches the same quality in fewer rounds.

## B.4. Reproducibility

Running the same configuration twice on the same GPU gives bit-identical results. On DTU with VGGT at the highest operating point, all four arms (dense, SparseVGGT, HeSS, ReSS) gave |∆CD| = 0 and |∆AUC| = 0. Nothing in the selection, the forward pass, or the scoring varies from run to run. Every quality number in this paper was measured on a single GPU model (NVIDIA L40S); every latency number, on an RTX 4090.

## C. Dense Alignment for Precise Evaluation

DA3-Bench [13] aligns the predicted geometry to the ground truth with a Sim(3) transform, estimated by a Umeyama fit [22] to the camera centers of the scene. This gives only as many samples as there are cameras (49 on DTU [1]): as sparsity grows and a few poses go badly wrong, those few dominate the fit, and the measured degradation mixes degraded geometry with failed alignment, changing the shape of the performance curve over sparsity.

We therefore estimate the transform from pixel-wise depth correspondences. Unprojecting the predicted and ground-truth depths, with validity masks from the backbone’s own confidence threshold, makes each pixel its own correspondence, yielding tens of millions of pairs per scene and diluting the influence of a few wrong poses accordingly. We run the Umeyama fit on these pairs, trim the top quantile of residuals once, and refit, which keeps regions of collapsed geometry from dragging the global transform. The procedure is applied identically to every method and every sparsity level, and the same alignment enters the head importance calibration of HeSS (Sec. D): the entire paper uses one definition of alignment.

## D. HeSS Implementation

ReSS takes the per-head budgets $\left\{ c _ { h } \right\}$ as given and, for fairness of comparison, uses the budget allocation of HeSS [11]. We recompute the allocation for all three backbones with a single procedure following the description in the HeSS paper; since HeSS and ReSS share the result, differences in allocation do not enter the comparison.

Table S1. Absolute numbers behind Fig. 5 of the main paper. Three of the five operating points, showing the reconstruction metric (Chamfer on DTU, F1 elsewhere). The best value within each compression group is in bold. Sparsity is measured, not requested.
<table><tr><td></td><td></td><td>dense</td><td colspan="3"> $s { = } 0 . 0 7$ </td><td colspan="3"> $s { = } 0 . 3 5$ </td><td colspan="3"> $s { = } 0 . 7 3$ </td></tr><tr><td></td><td></td><td></td><td>SparseVGGT</td><td>HeSS</td><td>Ours</td><td>SparseVGGT</td><td>HeSS</td><td>Ours</td><td>SparseVGGT</td><td>HeSS</td><td>Ours</td></tr><tr><td>VGGT</td><td>DTU↓</td><td>0.588</td><td>0.597</td><td>0.590</td><td>0.593</td><td>0.727</td><td>0.718</td><td>0.623</td><td>1.338</td><td>0.982</td><td>0.791</td></tr><tr><td>VGGT</td><td>ETH3D↑</td><td>0.756</td><td>0.754</td><td>0.756</td><td>0.751</td><td>0.739</td><td>0.729</td><td>0.768</td><td>0.538</td><td>0.600</td><td>0.743</td></tr><tr><td>VGGT</td><td>HiRoom ↑</td><td>0.790</td><td>0.781</td><td>0.773</td><td>0.793</td><td>0.738</td><td>0.766</td><td>0.802</td><td>0.472</td><td>0.575</td><td>0.755</td></tr><tr><td>VGGT</td><td>ScanNet++↑</td><td>0.676</td><td>0.674</td><td>0.681</td><td>0.678</td><td>0.623</td><td>0.629</td><td>0.676</td><td>0.508</td><td>0.527</td><td>0.626</td></tr><tr><td>VGGT</td><td>7-Scenes ↑</td><td>0.560</td><td>0.560</td><td>0.563</td><td>0.560</td><td>0.553</td><td>0.561</td><td>0.561</td><td>0.525</td><td>0.535</td><td>0.555</td></tr><tr><td> $\pi ^ { 3 }$ </td><td>DTU↓</td><td>0.838</td><td>0.858</td><td>0.827</td><td>0.839</td><td>1.192</td><td>1.079</td><td>0.857</td><td>2.221</td><td>2.227</td><td>1.067</td></tr><tr><td> $\pi ^ { 3 }$ </td><td>ETH3D ↑</td><td>0.846</td><td>0.845</td><td>0.847</td><td>0.849</td><td>0.810</td><td>0.841</td><td>0.850</td><td>0.647</td><td>0.649</td><td>0.842</td></tr><tr><td> $\pi ^ { 3 }$ </td><td>HiRoom ↑</td><td>0.908</td><td>0.905</td><td>0.909</td><td>0.909</td><td>0.658</td><td>0.855</td><td>0.912</td><td>0.378</td><td>0.424</td><td>0.850</td></tr><tr><td> $\pi ^ { 3 }$ </td><td>ScanNet++↑</td><td>0.774</td><td>0.767</td><td>0.771</td><td>0.771</td><td>0.602</td><td>0.678</td><td>0.763</td><td>0.381</td><td>0.400</td><td>0.711</td></tr><tr><td> $\pi ^ { 3 }$ </td><td>7-Scenes ↑</td><td>0.586</td><td>0.588</td><td>0.585</td><td>0.587</td><td>0.592</td><td>0.579</td><td>0.584</td><td>0.526</td><td>0.491</td><td>0.576</td></tr></table>

Table S2. Absolute numbers for DepthAnything3. The same runs as Fig. S1. Read as in Tab. S1.
<table><tr><td></td><td></td><td>dense</td><td colspan="3">s=0.10</td><td colspan="3"> $s { = } 0 . 4 8$ </td><td colspan="3"> $s { = } 0 . 8 0$ </td></tr><tr><td></td><td></td><td></td><td>SparseVGGT</td><td>HeSS</td><td>Ours</td><td>SparseVGGT</td><td>HeSS</td><td>Ours</td><td>SparseVGGT</td><td>HeSS</td><td>Ours</td></tr><tr><td>DA3</td><td>DTU↓</td><td>0.850</td><td>0.852</td><td>0.875</td><td>0.854</td><td>0.985</td><td>1.464</td><td>0.901</td><td>1.625</td><td>2.205</td><td>1.195</td></tr><tr><td>DA3</td><td>ETH3D ↑</td><td>0.853</td><td>0.857</td><td>0.848</td><td>0.851</td><td>0.847</td><td>0.851</td><td>0.854</td><td>0.786</td><td>0.835</td><td>0.832</td></tr><tr><td>DA3</td><td>HiRoom ↑</td><td>0.894</td><td>0.892</td><td>0.893</td><td>0.891</td><td>0.877</td><td>0.846</td><td>0.855</td><td>0.760</td><td>0.718</td><td>0.796</td></tr><tr><td>DA3</td><td>ScanNet++ ↑</td><td>0.772</td><td>0.772</td><td>0.772</td><td>0.772</td><td>0.768</td><td>0.766</td><td>0.771</td><td>0.743</td><td>0.755</td><td>0.771</td></tr><tr><td>DA3</td><td>7-Scenes ↑</td><td>0.569</td><td>0.570</td><td>0.569</td><td>0.570</td><td>0.572</td><td>0.572</td><td>0.570</td><td>0.564</td><td>0.571</td><td>0.572</td></tr></table>

Table S3. Track-best on and off. DTU, VGGT, 22 scans, highest operating point. All four arms have the same measured sparsity, so the rows are directly comparable.
<table><tr><td>rounds</td><td>track-best</td><td>sparsity</td><td>CD↓</td><td> $\mathbf { A U C } @ 3 0 ^ { \circ } \uparrow$ </td></tr><tr><td>5</td><td>一</td><td>0.7274</td><td>0.8403</td><td>0.9939</td></tr><tr><td>5</td><td> $\checkmark$ </td><td>0.7274</td><td>0.7910</td><td>0.9958</td></tr><tr><td>10</td><td></td><td>0.7274</td><td>0.7952</td><td>0.9950</td></tr><tr><td>10</td><td> $\checkmark$ </td><td>0.7274</td><td>0.7934</td><td>0.9964</td></tr></table>

Procedure. HeSS measures the importance of each head by the trace of the Fisher information of that head’s QKV weight gradients, averaged over a calibration set. Following the original procedure, we keep the CO3Dv2 [14] dev split, 20 views, and per-scene averaging, accumulate $\operatorname { t r } ( g g ^ { \top } ) =$ $\| g \| ^ { 2 }$ , and compute the loss under the project-wide pixelwise alignment (Sec. C). The two losses (point cloud and camera position) each yield one table, combined as

$$
s = \lambda \cdot s _ { \mathrm { c a m } } + ( 1 - \lambda ) \cdot s _ { \mathrm { p c } } .
$$

The head scores are turned into $\left\{ c _ { h } \right\}$ by proportional allocation with iterative clipping at the cap; the allocation is fixed after calibration and adds no inference-time cost.

Table S4. λ sweep for DepthAnything3. 22 DTU scenes and 20 ScanNet++ scenes at the highest compression operating point; measured sparsity 0.7963 on DTU and 0.7946 on ScanNet++, identical across the three settings. The best value is in bold.
<table><tr><td></td><td colspan="3"></td><td>ScanNet++</td></tr><tr><td> $\lambda$ </td><td> $\mathrm { C D \downarrow }$ </td><td>DTU  $\mathbf { A U C @ 3 0 ^ { \circ } } \uparrow$ </td><td> $\mathbf { A U C @ 5 ^ { \circ } } \uparrow$ </td><td>F1↑</td></tr><tr><td>0.0</td><td>2.3221</td><td>0.9425</td><td>0.7137</td><td>0.7204</td></tr><tr><td>0.5</td><td>2.1541</td><td>0.9729</td><td>0.8390</td><td>0.7416</td></tr><tr><td>1.0</td><td>2.2054</td><td>0.9774</td><td>0.8650</td><td>0.7551</td></tr></table>

λ differs across backbones. Which loss carries head importance depends on the backbone. For VGGT and $\pi ^ { 3 }$ we use the values reported by HeSS [11]: λ = 0.5 (Sec. S4.1) and $\lambda ~ = ~ 0 . 0$ (Sec. S4.5). HeSS reports no value for DepthAnything3, so we sweep $\lambda \in \{ 0 . 0 , 0 . 5 , 1 . 0 \}$ as HeSS does for $\pi ^ { 3 }$ and take $\lambda = 1 . 0$ , which gives the best DTU pose and ScanNet++ F1 (Tab. S4). DTU Chamfer slightly favors $\lambda \ = \ 0 . 5 ,$ , but ReSS stays ahead of HeSS on both datasets at either value (Tab. S2).

## E. Fused Kernel Implementation

One restoration round consists of computing the restore/drop gains Eqs. (14) and (15), masking against the inside and outside of the drop set, exact top-m selection for restores and for drops, and updating the drop set and the scalars G and K. Each is a cheap operation over an $( n _ { h } , q _ { \mathrm { b l k } } , k _ { \mathrm { b l k } } )$ tensor, but each incurs its own round trip to memory, so the round is bound by bandwidth rather than computation. We therefore fuse all of the above into a single Triton kernel. Each query row is processed in registers in one sweep, from gain computation to the drop-set update, with no intermediate tensor written to memory and only the new drop mask written out. The GEMM behind the innerproduct term of the gains stays outside the kernel, taking the cuBLAS output as is. The effect of fusion is reported in the w/o fused kernel row of Tab. 1.

![](images/0df1bfc31470e6e3dc710e82e90e0070e395c3ffb75fd67bd6187af9c4abeeac.jpg)  
Figure S1. DepthAnything3 on the five DA3-Bench datasets. Read as in Fig. 5: the vertical axis is the metric in its own units with the dashed line at dense, and the horizontal axis is measured sparsity. The trend matches the two backbones in the main paper, but the margins between methods are smaller. On 7-Scenes the methods differ only slightly, so that axis is magnified accordingly

The kernel produces bit-for-bit the same selection as the unfused path: the GEMM is not moved, so no accumulation order changes; the elementwise operations round at the same points as the unfused path; and the top-m is exact, with no approximation. This identity holds within the supported configurations; outside them, the implementation falls back to the unfused path. The cap $k _ { \mathrm { b l k } } \le 4 0 9 6$ comes from holding one row in registers, corresponds to about 253 views, and is conservative; every experiment in this paper is within this range.

## F. Porting LLM Sparse Attention Baselines

Common conditions. The three methods replace only the mask-construction step and share everything else with our method: the same grid of query blocks of 128 and key blocks of 64, the same block-sparse kernel, the same sink rule that always keeps the trailing block holding the special tokens, the same floor of at least two key blocks per query row that applies to every method in the paper, and the same definition of measured sparsity. The always-keep rules that each method defines in its own paper are kept on top. All three set their budget through a threshold, so for each of our grid points we pre-measure, on a subset of scenes, the threshold whose measured sparsity comes closest, and fix it; the maximum residual from the target sparsity is 0.023.

All three methods are designed for causal attention, whereas the global attention of 3D ViTs is bidirectional. We run all three without a causal mask and reinterpret every causality-dependent rule so that it keeps its intent under bidirectional attention. The antidiagonal score of XAttention is computed per block pair and does not depend on token order, and SpargeAttn calls the non-causal path already present in its official implementation. FlexPrefill has three causality-dependent rules, all listed in Tab. S5.

Porting decisions and their direction. All three methods are evaluated under a condition their papers do not cover: bidirectional sequences at the 10<sup>5</sup>-token scale. Transferred literally, two of the three either fail to compress at all or operate on biased statistics. Table S5 lists every decision: each one favors the baseline, preserves the original rule’s intent, or, in one case, is marked as having no established direction. The gaps in Fig. S2 and Tab. S6 are therefore unlikely to be artifacts of the port.

Observation. XAttention and FlexPrefill share one property: each sizes its mask by a threshold on attention probability mass alone, per query-block row for XAttention and per head over the whole block map for FlexPrefill, so beyond the common two-block floor, no row is guaranteed a budget. SparseVGGT instead fixes the count per row through its top-k rule. On one scene, we measure the attention mass actually captured by masks of equal measured sparsity. XAttention matches SparseVGGT on average (0.820 vs. 0.819) but falls to 0.655 vs. 0.753 on the bottom 5% of rows, so the loss concentrates on a few rows instead of spreading across queries.

## G. Porting Video Sparse Attention Baselines

Porting strategy. Unlike the LLM family, the cost of this family lies in its variable block layouts and their mask construction, so we use the upstream kernel and harness exactly as released. The dense-step schedule has no counterpart in a single forward pass; the k-means warm start and the permutation are all kept as is, hyperparameters are the upstream defaults without retuning, and the only knob we sweep is the one each method exposes in the upstream scripts (--sparsity for SVG, --top p kmeans for SVG2 and SVG-EAR). The one exception is the protected tokens: the block holding the camera/register tokens is placed in the upstream slot for context tokens and always kept, the same rule as for every other baseline in the paper.

Table S5. Porting decisions. Each row is a point where the original rule fails on 3D ViTs, together with our treatment; the parenthetical states whom the treatment favors, where this can be established. The favorable decisions are cases where, without the treatment, the baseline either could not be evaluated at all or would be penalized as an artifact of the port.
<table><tr><td>Rule in the original paper</td><td>Why it fails on 3D ViTs</td><td>Our treatment</td></tr><tr><td>Causal attention</td><td>Global attention is bidirectional, with no query-key ordering constraint.</td><td>Run without a causal mask; order-dependent rules are reinterpreted to keep their intent (neutral: pre- serves the rule&#x27;s intent).</td></tr><tr><td>gate  $\tau _ { \mathrm { s i m } } = 0 . 6$ </td><td>SpargeAttn [31] similarity Nearly every VGGT block is judged self-dissimilar Disable the gate and sweep the cumulative thresh- and reverted to dense: measured sparsity 0.006 at old alone (favors the baseline: no compression curve the default cumulative threshold of the official im- exists otherwise). Without the gate, the selection ef- plementation (0.98), and only 0.062 even at 0.1.</td><td>fectively reduces to the SparseVGGT rule of a cumu- lative threshold on pooled query/key inner products.</td></tr><tr><td>FlexPrefill [12] represen-] tative queries = last 128</td><td>tion they are a biased sample covering part of the last frame.</td><td>In a causal LM those queries are the only ones that Sample uniformly across the sequence (favors the have seen the full context; under bidirectional atten- baseline: the literal rule gives a worse estimate).</td></tr><tr><td>FlexPrefill [12] slash lines = constant offsets  $q - k$ </td><td>In a causal LM only non-negative offsets exist, so Search the full signed range – the slash family is the lower triangle; under bidirec- (neutral: preserves the rule&#x27;s intent). tional attention both signs carry mass, and searching the causal range alone would discard half of the can- didate lines.</td><td> $\cdot ( T _ { k } - 1 ) \dots ( T _ { q } - 1 )$ </td></tr><tr><td>of each query block</td><td>FlexPrefill [12] always- In a causal LM the last key block is the diagonal Keep the first block and the diagonal block, i.e., the keep = first/last key block block; under bidirectional attention it merely points blocks the rule pointed at in a causal LM (neutral: at the end of the sequence, and the rule loses its in- preserves the rule&#x27;s intent).</td><td></td></tr><tr><td>cal/slash line granularity</td><td>FlexPrefill [12] verti- The paper&#x27;s Algorithm 3 defines the lines at to- Select at token resolution, following the paper, and ken resolution, whereas the released implementation keep a key block if a selected line passes through sum-pools the vertical and slash scores into blocks it (direction not established: rasterizing adds blocks of 128 before selecting; neither granularity is the 64- relative to a token-exact mask, whereas the released wide key block of the shared kernel.</td><td>implementation keeps coarser, wider blocks).</td></tr><tr><td>XAttention [28] threshold At T</td><td> $1 0 ^ { 5 }$  tokens the paper&#x27;s  $\tau = 0 . 9$  sponds to sparsity 0.39, so values near it never reach grid (favors the baseline: with the original range, the the low-compression regime.</td><td>already corre- Extend the sweep up to  $\tau = 0 . 9 9 9 9$  to cover our full low-compression points are empty and no compari- son is possible).</td></tr></table>

![](images/bcd80df3f9cfb356bfbb63ef26da3ca86dd35e6c79297a4533f4c7b6f37c9853.jpg)  
Figure S2. Comparison with sparse attention for LLMs. DTU, VGGT [24] backbone. Dashed line denotes dense attention. ReSS preserves dense performance better than the three methods ported from LLM sparse attention.

Choice of SVG variant. The upstream implementation ships a separate mask definition per model family; of these, the Wan and HunyuanVideo variants are the candidates, and we use the latter. All variants share one formula that converts a target density into band widths after subtracting the cost of the always-kept context tokens; the formula is derived for the HunyuanVideo mask shape, and the Wan family, which has no context tokens, shares it only because that term vanishes. Measured elementwise on the VGGT/ScanNet++ shapes, the HunyuanVideo variant also hits the target density more accurately.

Table S6. Comparison with sparse attention for LLMs. DTU, VGGT [24] backbone, 22 scans, at two matched sparsity levels. The last block lists three methods proposed for LLM long-context prefill, ported onto the same block grid and the same sparse kernel as ReSS, so that only the block selection criterion differs.
<table><tr><td rowspan="2">Sparsity</td><td colspan="2">Chamfer ↓</td><td colspan="2">Pose AUC@30°↑</td></tr><tr><td>0.35</td><td>0.73</td><td>0.35</td><td>0.73</td></tr><tr><td>VGGT (dense)</td><td>0.588</td><td>0.588</td><td>0.999</td><td>0.999</td></tr><tr><td rowspan="2">SparseVGGT [23] HeSS [11]</td><td>0.727</td><td>1.338</td><td>0.991</td><td>0.963</td></tr><tr><td>0.718</td><td>0.982</td><td>0.995</td><td>0.982</td></tr><tr><td>XAttention [28]</td><td>1.663</td><td>1.961</td><td>0.959</td><td>0.915</td></tr><tr><td>SpargeAttn [31]</td><td>0.781</td><td>1.964</td><td>0.989</td><td>0.903</td></tr><tr><td>FlexPrefill [12]</td><td>1.039</td><td>1.957</td><td>0.984</td><td>0.942</td></tr><tr><td>ReSS (Ours)</td><td>0.623</td><td>0.791</td><td>0.999</td><td>0.996</td></tr></table>

Table S7. Latency against the k-means iteration count. VGGT, 100 views, RTX 4090. The SVG2 row is a linear fit over the sweep; dense and SparseVGGT do not depend on the iteration count, which confirms that the sweep touches nothing but k-means.
<table><tr><td></td><td>t (s)</td><td>at 50 iter</td></tr><tr><td>dense</td><td>10.47 (constant)</td><td>10.47</td></tr><tr><td>SparseVGGT</td><td>7.58 (constant)</td><td>7.58</td></tr><tr><td>SVG2</td><td>12.40 + 0.199 · iter</td><td>22.35</td></tr></table>

The k-means cost does not amortize. The upstream scripts (--zero step kmeans init) run 50 iterations of k-means at the first denoising step of each layer, which is still a dense step, and 2 warm-started iterations at each of the remaining 49 steps, dense or sparse, an average of 2.96 iterations per step. VGGT has a single forward pass, so every global layer pays the full 50 iterations of cold kmeans. The speedup of this family presupposes centroid reuse across denoising steps, and a single-pass reconstructor has no denominator to divide by. To confirm this, we sweep the iteration count directly and measure latency (Tab. S7): the cost is exactly linear in the iteration count. The point is the intercept: applying the upstream amortized rate (2.96 iterations) gives 13.0 s, and even with k-means entirely free the latency is 12.40 s, both above the dense 10.47 s. The cost does not sit in k-means alone but in re-planning the variable block layout at every layer, and no adjustment of the iteration count closes this gap.

![](images/db397256f94f962595c353448ef22287e30320ebaac6865eeebf29dc36c686a8.jpg)  
Figure S3. Qualitative results on DTU and ETH3D (VGGT). Each block is one scene; rows are the methods and columns the measured sparsity, matched across methods. A point is drawn in green when its distance to the ground truth exceeds a per-benchmark threshold (5 mm on DTU, and the F1 threshold of 0.25 m on ETH3D), and each panel reports the fraction of points above it.

![](images/6d2184a9b349716ca6c300af9169b4909709acd9596e9ac560213b79cc5a9321.jpg)  
Figure S4. Qualitative results on HiRoom and ScanNet++ (VGGT). Read as in Fig. S3. The accuracy threshold is 0.05 m on both benchmarks.
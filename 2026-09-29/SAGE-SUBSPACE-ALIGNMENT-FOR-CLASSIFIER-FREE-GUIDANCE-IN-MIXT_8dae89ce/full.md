# SAGE: SUBSPACE ALIGNMENT FOR CLASSIFIER-FREE GUIDANCE IN MIXTURE-OF-EXPERTS DIFFU-SION MODELS

Boyu Zhang<sup>1</sup>, Yangming Cheng<sup>1</sup>, Ning Zhang<sup>1</sup>, Pengfei Liu<sup>1</sup>, Weijie Li<sup>1</sup>, Yifan Gao<sup>1</sup>, Hangyu Li<sup>1</sup>, Litong Gong<sup>1,∗</sup>

<sup>1</sup>Alibaba Token Hub, Alibaba Group

## ABSTRACT

Diffusion Transformers with Mixture-of-Experts (MoE) routing are a leading recipe for scaling generative models. Classifier-Free Guidance (CFG) is essential for generation quality, yet excessively high guidance scales trigger collapse. We identify a previously unreported failure mode in their combination: the two CFG branches route independently, so their realized activations occupy different subspaces. The unconditional write then leaves the conditional subspace, and CFG amplifies that residual linearly in the guidance scale. We propose SAGE, a training-time regularizer that aligns unconditional MoE activations to the conditional subspace without restricting routing diversity, at zero inference cost. Toy experiments show that SAGE dramatically suppresses extreme drift by 9.2×. When scaled to a 1B-parameter text-to-image model, SAGE significantly improves generation quality, delivering a 9.3% boost in peak DPG-Bench performance. Extensive experiments demonstrate that SAGE consistently outperforms the baseline.

## 1 INTRODUCTION

Diffusion Transformers (DiTs) (Peebles & Xie, 2023) and their flow-matching counterparts (Lipman et al., 2023; Liu et al., 2023) have become the dominant paradigm for high-fidelity generation (Seedance, 2026; Happyhorse, 2026). Following the success of sparse experts in language modeling (Fedus et al., 2022; Lepikhin et al., 2021), recent work scales DiTs via Mixture-of-Experts (MoE) feed-forward blocks, including DiT-MoE (Fei et al., 2024), EC-DiT (Sun et al., 2025), Race-DiT (Yuan et al., 2025), and others (Cheng et al., 2025; Zheng et al., 2025). In parallel, Classifier-Free Guidance (CFG) (Ho & Salimans, 2022) remains the indispensable inference knob that trades diversity for fidelity by linearly combining conditional and unconditional velocity fields. Virtually every modern text-to-image system ships with CFG enabled, and practitioners routinely tune the guidance scale s to reach the operating point they want. These two ingredients, sparse experts for capacity and CFG for controllability, are therefore combined in almost every large-scale generative model being built today.

Despite the popularity of both MoE scaling and CFG conditioning, we find that their combination harbors a previously overlooked pathology. When CFG is applied to an MoE-DiT, sample quality degrades far more severely than in a parameter-matched dense model as the guidance scale increases. Because the conditional and unconditional forward passes produce different hidden states, the routers select different expert subsets for the two branches, and the CFG linear combination mixes outputs from incompatible subspaces, injecting a spurious error that grows with the guidance scale. This routing divergence is intrinsic to MoE and vanishes identically in dense architectures.

We trace this failure to a geometric mechanism. Different hidden states $h _ { c } \neq h _ { u }$ cause the Top-k routers to select different expert subsets $S _ { c } \neq S _ { u }$ . Write $D _ { \mathrm { r o u t e } }$ for this routing misalignment, and let $\gamma _ { S _ { c } }$ be the subspace spanned by the realized conditional activations. CFG then injects a leak $\delta ( s )$ outside $\nu _ { S _ { c } }$ that grows with the guidance scale s. In a dense model, $D _ { \mathrm { r o u t e } } \equiv 0$ , so the failure is MoE-intrinsic.

The mechanism suggests an obvious fix: forcing the two branches onto the same routes, for instance by penalizing the divergence between their routing distributions. We find this approach to be fundamentally flawed. It removes the mismatch, but it also eliminates the specialization that motivates MoE. Experiments confirm that constraining routing freedom degrades performance.

Our starting point is that CFG is closed in any common subspace: if both writes lie in a span V, then so does $v _ { \mathrm { c f g } }$ , regardless of which experts produced them. It is therefore enough to align the unconditional activations to the conditional subspace $\nu _ { S _ { c } }$ , while leaving the routers free. This observation motivates SAGE, Subspace Alignment for CFG in MoE Diffusion Models, a trainingtime regularizer that penalizes only the component of the unconditional MoE output that lies outside the stacked conditional activations. SAGE requires no architectural change, adds one loss during training, preserves routing diversity, and incurs zero inference cost.

Empirically, the gains are substantial. In toy experiments, SAGE cuts the worst-case drift at high guidance by 9.2× and largely restores mode purity where the baseline has collapsed. We further scale SAGE up to a 1B-parameter MoE-DiT on the text-to-image task, where it improves the peak DPG-Bench score by 9.3% and keeps that quality as guidance grows. The same pattern holds on GenEval, where SAGE lifts the peak score by 2.7 pp, and the qualitative comparisons show coherent structure at high guidance where the baseline has already broken down.

We summarize our contributions as follows:

• We uncover a novel failure mode specific to MoE-DiTs, termed routing-induced subspace leakage: routing misalignment forces conditional and unconditional branches into disjoint subspaces, which CFG subsequently amplifies linearly with the guidance scale. We show this structural leak is unique to sparse models and vanishes in dense architectures.

• We propose SAGE, a training-time regularizer that aligns unconditional activations back into the conditional subspace while maintaining expert routing diversity and incurring zero inference overhead. Crucially, we demonstrate that the intuitive alternative of forcing hard routing agreement is markedly sub-optimal.

• Across extensive experiments on large-scale text-to-image MoE-DiTs, SAGE consistently outperforms baseline models across various CFG scales. Controlled toy experiments and ablation studies validate the underlying mechanism and design choices of SAGE.

## 2 RELATED WORK

## 2.1 MIXTURE-OF-EXPERTS IN DIFFUSION MODELS

Sparse mixture-of-experts (MoE) architectures (Lepikhin et al., 2021; Fedus et al., 2022) have emerged as a dominant recipe for scaling diffusion transformers (Peebles & Xie, 2023) beyond the capacity of dense feed-forward networks (FFNs). RAPHAEL (Xue et al., 2023) pioneered the use of space-MoE and time-MoE layers for text-to-image diffusion, demonstrating that routing along spatial and temporal axes yields strong artistic generation. DiT-MoE (Fei et al., 2024) scales sparse DiTs to 16.5B parameters and achieves state-of-the-art performance. EC-DiT (Sun et al., 2025) replaces token-choice routing with expert-choice routing (Zhou et al., 2022) to allocate compute adap tively across image patches, while Diff-MoE (Cheng et al., 2025) and Race-DiT (Yuan et al., 2025) further explore time-aware and space-adaptive expert selection and flexible per-token expert counts. Dense2MoE (Zheng et al., 2025) takes a post-hoc route, restructuring pre-trained FLUX-style (Black Forest Labs, 2024) dense DiTs into sparse MoE backbones with 60% fewer activated parameters. Almost all of these works evaluate their models with classifier-free guidance (Ho & Salimans, 2022), yet none of them study how MoE routing interacts with the dual conditional/unconditional forward passes that CFG requires. Most recently, ProMoE (Wei et al., 2026) partitions tokens into conditional and unconditional subsets via a hard first-stage router and assigning dedicated experts to the unconditional branch. While this design provides partial relief analogous to that offered by the shared-expert mechanism analyzed in section 3.2, it does not align the realized activations of the remaining routed experts and sacrifices one expert’s capacity for conditional generation.

## 2.2 CLASSIFIER-FREE GUIDANCE

Classifier-free guidance has become the mainstream inference-time mechanism for conditional diffusion, as it improves generation results and enables better semantic control. It relies on jointly trained conditional and unconditional predictions, which are combined as $\left( 1 + s \right) v _ { c } - s v _ { u }$ at inference. Improving CFG has motivated a line of corrective methods (Hong et al., 2023; Zheng & Lan, 2024; Ahn et al., 2024; Kynka¨anniemi et al., 2024; Sadat et al., 2025b). GLIDE (Nichol et al.,¨ 2022) empirically compared CFG to CLIP guidance (Radford et al., 2021) for text-to-image synthesis, and subsequent work has proposed CFG rescaling, dynamic schedules, and spatial-inconsistency analyses to mitigate artifacts at high s (Shen et al., 2024). CADS (Sadat et al., 2024) anneals the conditioning to restore diversity, APG (Sadat et al., 2025a) decomposes the guidance term to remove the oversaturation and artifacts that appear at large scales, and CFG-Zero\* (Fan et al., 2025) rescales and zero-initializes the guidance to correct inaccurate velocity estimates. CFG++ (Chung et al., 2025) attributes guidance failures to iterates drifting off the data manifold and addresses this by reformulating CFG as a manifold-constrained inverse problem solved at sampling time. Another line of work replaces the null-condition branch altogether: PAG (Ahn et al., 2024) perturbs self-attention to synthesize a deliberately degraded prediction as the guidance baseline, in the spirit of self-attention guidance (Hong et al., 2023), and Karras et al. (2024) propose autoguidance, which improves performance by additionally training an auxiliary model. Li et al. (2025) investigates how guidance amplifies the top singular directions of the conditional score. Our analysis complements theirs by revealing a qualitatively different failure mode unique to sparse architectures: routing misalignment places the two CFG branches in different realized subspaces, and CFG then scales the residual of the unconditional write outside the conditional subspace. Unlike autoguidance or guidance-distillation approaches (Meng et al., 2023; Jensen & Sadat, 2025), which modify or bypass the unconditional branch, we resolve this structural mismatch during training via an architecture-aware subspace constraint that incurs zero inference cost.

## 3 SAGE: SUBSPACE ALIGNMENT FOR CLASSIFIER-FREE GUIDANCE

## 3.1 PRELIMINARIES

Flow Matching. Flow matching (Lipman et al., 2023; Liu et al., 2023) learns a velocity field $v _ { \theta } ( x , t , c ) : \mathbb { R } ^ { d } \times [ 0 , 1 ] \times \mathcal { C }  \overline { { \mathbb { R } ^ { d } } }$ transporting $\mathcal { N } ( 0 , I _ { d } )$ to $p _ { \mathrm { d a t a } } ( \cdot \mid c )$ . With the flow interpolant $x _ { t } = t x _ { 1 } + \left( 1 - t \right) \varepsilon$ , the loss is

$$
\begin{array} { r } { \mathcal { L } _ { \mathrm { F M } } ( \theta ) = \mathbb { E } _ { x _ { 1 } , \varepsilon , t , c } \Big [ \big | | v _ { \theta } ( x _ { t } , t , c ) - ( x _ { 1 } - \varepsilon ) \big | \big | _ { 2 } ^ { 2 } \Big ] . } \end{array}\tag{1}
$$

Classifier-Free Guidance. Classifier-free guidance (Ho & Salimans, 2022) uses a single model to handle both conditional and unconditional regimes by replacing c with a null token $\mathcal { D }$ at random during training. At inference, the two branches are combined as

$$
\begin{array} { r } { v _ { \mathrm { c f g } } ( x , t , c , s ) = ( 1 + s ) v _ { \theta } ( x , t , c ) - s v _ { \theta } ( x , t , \emptyset ) . } \end{array}\tag{2}
$$

MoE Layer. An MoE layer (Lepikhin et al., 2021; Fedus et al., 2022) replaces the FFN with N parallel experts and a sparse router. Given a hidden state $\boldsymbol { h } \in \mathbb { R } ^ { d _ { h } }$ , the router selects a top-k active set $S ( h ) \ \bar { \subseteq } \ \{ 1 , \dots , N \}$ with gating weights $g _ { i } ( h )$ , and each expert $E _ { i }$ is a two-layer FFN with output projection $W _ { 2 } ^ { ( i ) } \in \mathbb { R } ^ { d _ { h } \times d _ { f } }$

$$
f _ { \mathrm { M o E } } ( h ) = \sum _ { i \in S ( h ) } g _ { i } ( h ) E _ { i } ( h ) .\tag{3}
$$

## 3.2 ROUTING MISALIGNMENT AND SUBSPACE ALIGNMENT

We write $v _ { c } = f _ { \mathrm { M o E } } ( h _ { c } )$ and $v _ { u } = f _ { \mathrm { M o E } } ( h _ { u } )$ for the MoE outputs under condition c and the null token ${ \cal { O } } .$ , respectively, with active sets $S _ { c }$ and $S _ { u }$ . Because the conditioning signal is mixed into earlier layers, $h _ { c } \neq h _ { u }$ , so the routers select different experts: $S _ { c } \ne S _ { u }$ in general. We quantify this with

$$
D _ { \mathrm { r o u t e } } : = 1 - \frac { \left| S _ { c } \cap S _ { u } \right| } { k } \in [ 0 , 1 ] .\tag{4}
$$

![](images/b3faa564e2436101ba3e12c5430da9d88cbf206377b6f0d3bda181dbc5f08a00.jpg)  
Figure 1: Subspace alignment. Top: Conditional and unconditional hidden states $h _ { c }$ and $h _ { u }$ are routed independently, so the realized activations occupy different subspaces. (a) Standard CFG injects a leak $\delta ( s )$ . (b) SAGE uses thin QR of the stacked conditional outputs $F _ { c }$ to yield an orthonormal basis $Q _ { c }$ of $\nu _ { S _ { c } }$ . Projecting $F _ { u }$ onto this subspace and penalizing the residual with $\mathcal { L } _ { \mathrm { s a g e } }$ enforces $v _ { u } \in \mathcal { V } _ { S _ { c } }$ at training time.

As illustrated in fig. 1 (a), $v _ { c }$ and $v _ { u }$ live in different subspaces, so the CFG combination (2) mixes incompatible directions. Decomposing this combination into shared and exclusive parts gives:

$$
\begin{array} { r l } & { v _ { \mathrm { c f g } } = \underset { \underset { i \in S _ { c } \cap S _ { u } } { \underbrace { \sum } } } { \sum } \left[ \left( 1 + s \right) g _ { i } ^ { c } E _ { i } ( h _ { c } ) - s \underset { j \in S _ { u } \setminus S _ { c } } { \sum } g _ { j } ^ { u } E _ { i } ( h _ { u } ) \right] } \\ & { ~ + ~ ( 1 + s ) \underset { \underset { i \in S _ { c } \setminus S _ { u } } { \sum } } { \sum } g _ { i } ^ { c } E _ { i } ( h _ { c } ) ~ - ~ s \underset { j \in S _ { u } \setminus S _ { c } } { \sum } ~ g _ { j } ^ { u } E _ { j } ( h _ { u } ) . } \end{array}\tag{5}
$$

The exclusive unconditional term is a write that the conditional branch did not produce in this pass. Let $\nu _ { S _ { c } }$ be the rank-m section of stacked conditional activations defined below, with projector $P _ { c }$ and define the leak

$$
\delta ( s ) : = - s ( I - P _ { c } ) v _ { u } .\tag{6}
$$

Then $v _ { \mathrm { c f g } } = ( 1 + s ) v _ { c } - s P _ { c } v _ { u } + \delta ( s )$ , and $\| \delta ( s ) \| = s \| ( I - P _ { c } ) v _ { u } \|$ . The leak is exactly linear in s and vanishes if and only if $v _ { u } \in \mathcal { V } _ { S _ { c } }$ . Exclusive experts are the MoE-specific source of this residual (Appendix A); a dense FFN has $D _ { \mathrm { r o u t e } } \equiv 0$ and no exclusive experts.

If $v _ { c } , v _ { u } \in \mathcal { V }$ for any subspace $\nu ,$ then $v _ { \mathrm { c f g } } \in \mathcal { V } ;$ CFG is closed. We need neither $S _ { c } = S _ { u }$ nor a shared global basis. As in fig. 1, it is enough that $v _ { u }$ lie in the conditional subspace. SAGE therefore penalizes only the component of the unconditional write outside $\nu _ { S _ { c } }$

$$
\begin{array} { r } { \mathcal { L } _ { \mathrm { s a g e } } ( \theta ) : = \mathbb { E } \left[ \left. ( I - P _ { c } ) f _ { \mathrm { M o E } } ( h _ { u } ) \right. _ { 2 } ^ { 2 } \right] . } \end{array}\tag{7}
$$

In a minibatch of B samples with T tokens per sample, stack the layer outputs into $F _ { c } , F _ { u } \in \mathbb { R } ^ { d _ { h } \times n }$ with $n = B T$ . Full-span QR of $F _ { c }$ is vacuous whenever $n \geq d _ { h }$ . We therefore draw a column subset Ω of size

$$
m = \mathrm { m i n } \big ( n , \lfloor d _ { h } / k _ { \mathrm { s p l i t } } \rfloor \big ) ,\tag{8}
$$

where the integer $k _ { \mathrm { s p l i t } }$ is chosen so that $\nu _ { S _ { c } }$ remains a proper subspace whenever n is large. Stopgradient $F _ { c }$ and take a thin QR of the section $F _ { c , \Omega } = \hat { F } _ { c } R \colon$

$$
\mathcal { L } _ { \mathrm { s a g e } } = \frac { 1 } { \Vert F _ { u } \Vert _ { F } ^ { 2 } } \big \Vert F _ { u } - \hat { F } _ { c } \hat { F } _ { c } ^ { \top } F _ { u } \big \Vert _ { F } ^ { 2 } .\tag{9}
$$

Algorithm 1 MoE-DiT Training with SAGE   
Require: Dataset D, MoE-DiT $v _ { \theta } ,$ , SAGE weight λ, CFG dropout $p _ { \mathrm { d r o p } } ,$ split factor $k _ { \mathrm { s p l i t } }$   
1: for each training iteration do   
2: Sample minibatch $\{ ( x _ { 1 } , c ) \} \sim \mathcal { D } , t \sim \mathcal { U } [ 0 , 1 ] , \varepsilon \sim \mathcal { N } ( 0 , I )$   
3: $x _ { t }  t x _ { 1 } + ( 1 - t ) \varepsilon ; \quad y  x _ { 1 } - \varepsilon$   
4: Conditional pass: $v _ { c } , \{ f _ { \mathrm { M o E } } ^ { ( \ell ) } ( h _ { c } ) \} _ { \ell } \gets \mathrm { F o r w a r d } ( v _ { \theta } , x _ { t } , t , c )$   
5: Unconditional pass: $v _ { u } , \{ f _ { \mathrm { M o E } } ^ { ( \ell ) } ( h _ { u } ) \} _ { \ell } \gets$ Forward $\left( { \boldsymbol { v } } _ { \theta } , { \boldsymbol { x } } _ { t } , t , { \boldsymbol { \emptyset } } \right)$   
6: Draw $u \sim$ Bernoulli $( p _ { \mathrm { d r o p } } ) ;$ $\overline { { \mathcal { L } } } _ { \mathrm { F M } }  \| v _ { u } - y \| ^ { 2 }$ if u else $\| v _ { c } - y \| ^ { 2 }$   
7: for each MoE layer ℓ do   
8: $\begin{array} { r } { F _ { c } \gets \mathrm { s t a c k } ( f _ { \mathrm { M o E } } ^ { ( \ell ) } ( h _ { c } ) ) ; \quad F _ { u } \gets \mathrm { s t a c k } ( f _ { \mathrm { M o E } } ^ { ( \ell ) } ( h _ { u } ) ) \left\{ F _ { c } , F _ { u } \in \mathbb { R } ^ { d _ { h } \times n } , n = B T \right\} } \end{array}$   
9: $m \gets \operatorname* { m i n } ( n , \lfloor d _ { h } / k _ { \mathrm { s p l i t } } \rfloor )$ ; sample m columns Ω; $\hat { F } _ { c } , \tiny - \mathrm { ~ t h i n Q R } ( \mathrm { s g } ( F _ { c , \Omega } ) )$   
10: $\begin{array} { r } { \mathcal { L } _ { \mathrm { s a g e } } ^ { ( \ell ) }  \frac { 1 } { \Vert F _ { u } \Vert _ { F } ^ { 2 } } \Vert F _ { u } - \hat { F } _ { c } \hat { F } _ { c } ^ { \top } F _ { u } \Vert _ { F } ^ { 2 } } \end{array}$   
11: end for   
12: $\begin{array} { r } { \mathcal { L } _ { \mathrm { t o t a l } }  \mathcal { L } _ { \mathrm { F M } } + \lambda \sum _ { \ell } \mathcal { L } _ { \mathrm { s a g e } } ^ { ( \ell ) } } \end{array}$   
13: $\theta  \theta - \mathrm { l r } \cdot \nabla _ { \theta } \mathcal { L } _ { \mathrm { t o t a l } }$   
14: end for

The cost is $\mathcal { O } ( d _ { h } m ^ { 2 } + d _ { h } m n )$ per layer, and the total objective is

$$
\mathcal { L } _ { \mathrm { t o t a l } } ( \theta ) = \mathcal { L } _ { \mathrm { F M } } ( \theta ) + \lambda \sum _ { \ell } \mathcal { L } _ { \mathrm { s a g e } } ^ { ( \ell ) } ( \theta ) , \qquad \lambda > 0 .\tag{10}
$$

The batch leak energy is then $s \varepsilon \| F _ { u } \| _ { F }$ (theorem A.3), where $\varepsilon$ is the training residual. SAGE needs a paired unconditional forward on the same $x _ { t }$ in order to form $( F _ { c } , F _ { u } )$ . The flow-matching term still follows standard CFG dropout: with probability $p _ { \mathrm { d r o p } }$ one supervises $v _ { u }$ , otherwise $v _ { c } .$ Inference is unchanged. The procedure is algorithm 1.

## 4 EXPERIMENTS

We validate SAGE in two complementary settings. First, we use a 2-D Gaussian mixture toy experiment (section 4.1) to empirically verify the leak arising from CFG in MoE-DiTs. In this setting, the leak $\lVert \delta ( s ) \rVert$ is directly measurable, and the true likelihood is available in closed form. Subsequently, we demonstrate the effectiveness of SAGE on a large-scale, 1B-parameter MoE-DiT for text-toimage generation (section 4.2), showing that our method successfully scales to high-dimensional data.

## 4.1 TOY EXPERIMENT

We construct a 2-D class-conditional Gaussian mixture with $K { = } 8$ isotropic modes $( r { = } 5 , \sigma { = } 0 . 3 ;$ 80,000 training samples). The backbone is a flow-matching (Lipman et al., 2023) multilayer perceptron (MLP) with a single MoE layer (N=8 experts, $\mathrm { t o p } \mathrm { - } \bar { k } \mathrm { = } 2 , \bar { d } _ { h } \mathrm { = } 6 4 , d _ { f } \mathrm { = } 1 6 ) .$ . A dense FFN of width 32 (matching the number of active parameters) serves as a control. With T=1, SAGE uses every available conditional write, so $k _ { \mathrm { s p l i t } }$ is not needed. We compare three configurations: Dense, MoE, and MoE + SAGE (λ=1), all trained with AdamW (Loshchilov & Hutter, 2019) for 20,000 steps and sampled using a 50-step Euler ODE solver.

In this toy setup, conditional generation corresponds to class-specific point clouds, while the unconditional counterpart spans the remaining empty regions. Thus, applying CFG essentially concentrates the points of each class toward their respective centers. In theory, a larger guidance scale should drive the empirical cluster centers closer to the ground-truth centers. However, due to the inherent CFG bias, an excessively large guidance scale often causes the generated clusters to deviate from the ground-truth centers.

Figure 2 visualizes inference results for each model across four CFG scales, with per-class excess drift $\Delta d$ relative to $s { = } 1$ shown as bar charts below each panel; the annotation in each bar chart reports the maximum $\Delta d .$ . Table 1 quantifies two complementary metrics: the mean $\Delta d$ averaged over all eight classes (with the per-class maximum in parentheses, matching the bar-chart annotations in fig. 2) and mode purity. At s=7, the maximum per-class excess drift for MoE reaches 3.9, over 1.9× that of Dense, while MoE + SAGE stays at only 1.0. At s=15, the gap widens dramatically: the maximum per-class excess drift for MoE reaches 12.9, while MoE + SAGE remains at 1.4 (9.2× down). Mode purity tells the same story: MoE collapses to 0.121 at s=7 while MoE + SAGE maintains 0.781 (a 6.5× improvement). These results confirm that the subspace leak from routing misalignment is MoE-specific, and that SAGE effectively neutralizes it.

![](images/d87b5469970c3fdd44f95aa637f2e56855bba51d9e7d49ca083f0059cf0791f8.jpg)  
Figure 2: Toy experiment sample grids for Dense, MoE, and MoE+SAGE across $s \in \{ 1 , 4 , 7 , 1 5 \}$ with per-class center drift $( \Delta d ;$ lower is better). The baseline MoE exhibits increasing drift at large s; SAGE keeps the drift substantially smaller.

Table 1: Quantitative results on the toy experiment. Excess drift ∆d reports the mean perclass distance increase relative to s=1 (matching the bar charts in fig. 2); the per-class maximum is shown in parentheses for cross-reference with the bar-chart annotations. Mode purity (higher is better) measures separation across the eight modes. MoE drift explodes at high guidance while MoE + SAGE stays close to Dense.
<table><tr><td></td><td colspan="3">Excess Drift ∆d (↓)</td><td colspan="5">Mode Purity (↑)</td></tr><tr><td>Method</td><td>s=4</td><td>s=7</td><td>s=15</td><td>s=1</td><td>s=2</td><td>s=4</td><td> $s { = } 7$ </td><td>s=15</td></tr><tr><td>Dense</td><td>0.33 (0.9)</td><td>0.50 (2.0)</td><td>1.06 (4.3)</td><td>.993</td><td>.968</td><td>.794</td><td>.586</td><td>.501</td></tr><tr><td>MoE</td><td>0.72 (0.9)</td><td>1.83 (3.9)</td><td>5.59 (12.9)</td><td>.989</td><td>.929</td><td>.394</td><td>.121</td><td>.125</td></tr><tr><td>MoE + SAGE</td><td>0.34 (0.6)</td><td>0.39 (1.0)</td><td>0.39 (1.4)</td><td>.997</td><td>.989</td><td>.917</td><td>.781</td><td>.666</td></tr></table>

## 4.2 LARGE-SCALE TEXT-TO-IMAGE GENERATION

## 4.2.1 SETUP

We train a 1B-parameter MoE-DiT consisting of 12 Transformer layers (Vaswani et al., 2017) with hidden dimension $d _ { h } { = } 1 5 3 6$ and 12 attention heads (128 dimensions per head). Each layer contains 8 routed experts and 1 shared expert, with top-2 routing applied to the routed experts. The intermediate dimensions of the routed and shared experts are 1024 and 2048, respectively. SAGE is applied with $k _ { \mathrm { s p l i t } } { = } 2$ . Text conditioning is provided by a frozen Qwen2.5-VL-7B-Instruct encoder (Bai et al., 2025).

Table 2: Overall scores on DPG-Bench and GenEval across CFG scales. SAGE consistently outperforms the Baseline at all $s \geq 3$ and degrades more gracefully under high guidance, while the KL routing constraint hurts performance across the board. ∆ (peak→10) measures the change from the peak score to the score at s=10; $\Delta _ { \mathrm { b a s e } }$ shows the difference from the Baseline’s peak score. Bold denotes the best score.
<table><tr><td rowspan="2">Benchmark</td><td rowspan="2">Method</td><td colspan="5">CFG guidance scale s</td><td rowspan="2"></td><td rowspan="2"> $\Delta _ { \mathrm { b a s e } }$ </td></tr><tr><td>1</td><td>3</td><td>5</td><td>7</td><td>10  $\Delta ( \mathrm { p e a k } {  } 1 0 )$ </td></tr><tr><td rowspan="3">DPG-Bench</td><td>Baseline</td><td>54.77</td><td>61.76</td><td>56.33</td><td>52.38</td><td>48.56</td><td>-13.20</td><td></td></tr><tr><td>SAGE (λ=1)</td><td>52.30</td><td>67.49</td><td>65.63</td><td>63.36</td><td>59.55</td><td>-7.94</td><td>+5.73</td></tr><tr><td> ${ \mathrm { K L } } \left( \lambda _ { \mathrm { k l } } { = } 1 \right)$ </td><td>40.95</td><td>50.65</td><td>46.82</td><td>44.38</td><td>41.08</td><td>-9.57</td><td>-11.11</td></tr><tr><td rowspan="3">GenEval</td><td>Baseline</td><td>30.3</td><td>57.1</td><td>55.5</td><td>53.2</td><td>47.1</td><td>-10.0</td><td>一</td></tr><tr><td> $\mathbf { S A G E } \left( \lambda { = } 1 \right)$ </td><td>26.5</td><td>59.8</td><td>59.3</td><td>57.1</td><td>52.5</td><td>-7.3</td><td>+2.7</td></tr><tr><td> ${ \mathrm { K L } } \left( \lambda _ { \mathrm { k l } } { = } 1 \right)$ </td><td>13.8</td><td>36.0</td><td>34.5</td><td>32.2</td><td>29.5</td><td>-6.5</td><td>-21.1</td></tr></table>

![](images/bc64a6ef6b1e082e8caa4e0e4c6cee9bd4e5ddd1f26329e960178b84ee233f16.jpg)  
Figure 3: Performance improvements of SAGE across CFG scales. SAGE (red) maintains high performance across the full guidance range, while the Baseline (blue) peaks at s=3 and declines sharply. The KL routing constraint (gray) consistently underperforms both methods, confirming that naive distributional alignment of routing decisions is counterproductive.

We curate a 100M-image subset of LAION-5B (Schuhmann et al., 2022), re-caption it with Qwen2.5-VL 72B, and resize and crop all images to a resolution of approximately $2 5 6 \times 2 5 6$ . We use AdamW with a learning rate of $1 \dot { 0 } ^ { - 4 }$ , a constant learning-rate schedule following a 1,000-step warmup, and a global batch size of 512 on 32 NVIDIA H200 GPUs. We set $p _ { \mathrm { d r o p } } { = } 0 . 1$ , the MoE auxiliary load-balancing loss coefficient (Fedus et al., 2022) to $\alpha _ { \mathrm { a u x } } { = } 0 . 0 0 1$ , and the router z-loss coefficient (Zoph et al., 2022) to $\alpha _ { z } { = } 0 . 0 0 1$ . Three configurations are compared: Baseline $( \lambda { = } 0 )$ SAGE (λ=1), and KL Routing Constraint $( \lambda _ { \mathrm { k l } } { = } 1 )$ , which penalizes the KL divergence between the conditional and unconditional routing distributions to force $S _ { c } \approx S _ { u }$ . All models are trained for the same 200,000 steps to ensure a fair comparison.

We use two benchmarks: (a) DPG-Bench (Hu et al., 2024), evaluated with an mPLUG backbone (Li et al., 2022), for which we report the overall score and a per-category breakdown; and (b) GenEval (Ghosh et al., 2023), with 553 prompts and 4 seeds per prompt. CFG scales are swept over s ∈ {1.0, 3.0, 5.0, 7.0, 10.0}.

## 4.2.2 MAIN RESULTS

Quantitative comparison. The ordering SAGE > Baseline > KL holds consistently at all $s \geq 3$ on both benchmarks (table 2 and fig. 3). SAGE significantly enhances generation quality, boosting peak DPG-Bench performance by 9.3%. Meanwhile, SAGE’s robustness is particularly notable: the DPG-Bench score drops by only 7.94 points from its peak to $s { = } 1 0$ , versus 13.20 for the Baseline; moreover, SAGE at $s { = } 7$ still exceeds the Baseline’s peak. The DPG-Bench evaluation protocol and category-level breakdown are provided in Appendix B.1. On GenEval, the pattern is the same: SAGE retains a +5.4 pp advantage at s=10 (52.5% vs. 47.1%) and loses only 7.3 pp from its peak, compared with 10.0 pp for the Baseline. The largest per-task improvement is on color-attribute binding (+11.5 pp; full breakdown in Appendix B.2), a compositional task that relies heavily on precise conditional control, exactly where a write outside the conditional subspace is most damaging. To confirm that the CFG-robustness benefit of SAGE also holds for distributional image fidelity, we report FID (Heusel et al., 2017) across CFG scales in Appendix B.5.

![](images/2df7ee97ad3cc5acf91ee926df054d89b105abaca241bc6fdfdfc5d7091f09ba.jpg)  
Figure 4: Qualitative comparison across CFG scales. Each panel shows the same prompt rendered by Baseline (top) and SAGE (bottom) at s ∈ {1, 3, 7, 10, 15} from left to right. At s=10 and 15, the Baseline exhibits over-saturation, white patches, and structural collapse, while SAGE preserves coherent composition and realistic textures.

The KL routing constraint, which penalizes $D _ { \mathrm { K L } } ( \pi _ { c } \Vert \pi _ { u } )$ to force consistent routing, is counterproductive on every metric. By collapsing the routing distributions, it destroys expert specialization: the model can no longer assign different experts to different semantic attributes, resulting in a −21 pp drop on GenEval relative to the Baseline at $s { = } 3$ . This provides strong empirical evidence for the theoretical insight of section 3.2: the correct fix is subspace alignment, not routing agreement. In contrast to the KL constraint, SAGE leaves the router essentially untouched: its conditional and unconditional branches stay as routing-diverse as the Baseline at every guidance scale, so subspace alignment is achieved without sacrificing expert specialization (Appendix B.4).

Qualitative comparison. Figure 4 presents several prompts across the full CFG sweep. At moderate guidance $( s \leq 7 )$ , both methods produce plausible images; as s increases to 10 and 15, the Baseline develops over-saturated colors, white-out patches, and structural repetition, all symptomatic of the subspace leak $\delta ( s )$ in (6). SAGE preserves coherent composition, fine-grained texture, and realistic color balance even at s=15. The quantitative and qualitative trends therefore agree: the benefit of SAGE is not a benchmark artifact but a visible improvement in image structure at high guidance.

## 4.2.3 ABLATION STUDY

Keeping the setup of section 4.2.1 fixed and evaluating at guidance scales $s \in \{ 3 , 5 \}$ , we ablate the two most consequential design choices of SAGE: when the SAGE loss is applied and how strongly it is weighted. For both ablations, each configuration is evaluated on DPG-Bench and GenEval at each scale. Two further ablations, covering how often the loss needs to be activated and how much of its effect a shared expert already provides for free, are deferred to Appendix B.3 and use the same two guidance scales.

Table 3: Training timing and loss weight analysis. Left: comparison of from-scratch SAGE and post-hoc fine-tuning of a pre-trained baseline for 5k or 10k steps, with λ=1 for all SAGE variants. Right: effect of the SAGE loss weight λ; the default λ=1 performs best among the tested values on both benchmarks at both guidance scales.
<table><tr><td rowspan="2">Method</td><td colspan="2"> $s { = } 3$ </td><td colspan="2">s=5</td></tr><tr><td>DPG</td><td>GenEval</td><td>DPG</td><td>GenEval</td></tr><tr><td>Baseline</td><td>61.76</td><td>57.1</td><td>56.33</td><td>55.5</td></tr><tr><td>SAGE (from scratch)</td><td>67.49</td><td>59.8</td><td>65.63</td><td>59.3</td></tr><tr><td>Post-hoc SAGE (5k)</td><td>61.32</td><td>56.3</td><td>56.23</td><td>54.8</td></tr><tr><td>Post-hoc SAGE (10k)</td><td>60.20</td><td>55.3</td><td>55.73</td><td>53.5</td></tr></table>

<table><tr><td rowspan="2">λ</td><td colspan="2"> $s { = } 3$ </td><td colspan="2">s=5</td></tr><tr><td>DPG</td><td>GenEval</td><td>DPG</td><td>GenEval</td></tr><tr><td>0.565.31</td><td>58.8</td><td>62.07</td><td>57.8</td></tr><tr><td>1 67.49</td><td>59.8</td><td>65.63</td><td>59.3</td></tr><tr><td>2.064.85</td><td>58.6</td><td>63.33</td><td>58.2</td></tr><tr><td>5.0 59.94</td><td>56.2</td><td>58.69</td><td>55.8</td></tr></table>

Training timing. A natural question is whether SAGE can be applied post-hoc to an alreadytrained baseline by fine-tuning with $\mathcal { L } _ { \mathrm { s a g e } }$ for a small number of additional steps. Table 3 compares SAGE trained from scratch with post-hoc fine-tuning for 5k and 10k steps, all with λ=1. Applying SAGE from the start yields better performance, whereas post-hoc fine-tuning fails to improve on the baseline. We hypothesize that once the two branches’ activation clouds have specialized apart during pre-training, a short SAGE fine-tune cannot pull $F _ { u }$ into the conditional section without harming the flow-matching fit: the residual range $F _ { u } ) \nsubseteq { \mathcal { V } } _ { S _ { c } }$ has already crystallized, and rotating the writes into containment requires training from scratch.

Loss weight sensitivity. To assess sensitivity to the loss weight at scale, we vary $\lambda \in$ {0.5, 1, 2.0, 5.0}. table 3 shows that the default weight λ=1 performs best on both benchmarks at both guidance scales. Relative to the baseline in the left panel, reducing the weight to 0.5 yields only small gains, while increasing it to 2.0 or 5.0 lowers all four scores below the baseline. This peaked response is consistent with a trade-off between subspace alignment and flow-matching fit: an excessively large SAGE loss weight may restrict the representations needed for accurate velocity prediction. theorem A.3 identifies leak energy with the SAGE residual; it does not imply that benchmark scores improve monotonically with λ.

Other ablations. Appendix B.3 reports further ablations on activation frequency and the shared expert, together with a comparison against ProMoE (Wei et al., 2026), at $s \in \{ 3 , 5 \}$ . Activating SAGE every second or fifth training step retains most of the gains from applying it every step, indicating that continuous activation is unnecessary. Removing the shared expert lowers baseline performance, but SAGE still improves both benchmarks, supporting the view that a shared expert alone does not replace explicit subspace alignment. ProMoE improves over the baseline but remains below the default SAGE configuration at both guidance scales.

## 5 CONCLUSION AND FUTURE WORK

We identified a subspace leak in MoE diffusion models. Routing misalignment lets the two CFG branches realize different subspaces; exclusive experts let the unconditional write leave the conditional subspace, and CFG amplifies that residual linearly in s. We proposed SAGE, which penalizes the component of the unconditional MoE write outside $\gamma _ { S _ { c } }$ at training time, requires no architectural change, and incurs zero inference cost. From a controlled toy microscope to a 1B text-to-image model, SAGE consistently improves CFG stability.

Our large-scale evaluation covers models up to the 1B-parameter scale. Scaling SAGE to 100B+ models, analyzing how errors compound across layers in deeper architectures, and exploring video generation remain important directions for future work. Furthermore, CFG mechanisms in most generative models are considerably more complex; for instance, text-to-audio-video (T2AV) and reference-to-video (R2V) tasks often employ double CFG and offer promising settings for future investigation.

## REFERENCES

Donghoon Ahn, Hyoungwon Cho, Jaewon Min, Wooseok Jang, Jungwoo Kim, SeonHwa Kim, Hyun Hee Park, Kyong Hwan Jin, and Seungryong Kim. Self-rectifying diffusion sampling with perturbed-attention guidance. In Proceedings of the European Conference on Computer Vision, 2024.

Shuai Bai, Keqin Chen, Xuejing Liu, Jialin Wang, Wenbin Ge, Sibo Song, Kai Dang, Peng Wang, Shijie Wang, Jun Tang, Humen Zhong, Yuanzhi Zhu, Mingkun Yang, Zhaohai Li, Jianqiang Wan, Pengfei Wang, Wei Ding, Zheren Fu, Yiheng Xu, Jiabo Ye, Xi Zhang, Tianbao Xie, Zesen Cheng, Hang Zhang, Zhibo Yang, Haiyang Xu, and Junyang Lin. Qwen2.5-VL technical report. arXiv preprint arXiv:2502.13923, 2025.

Black Forest Labs. Announcing Black Forest Labs, 2024. URL https://bfl.ai/ announcing-black-forest-labs/.

Kun Cheng, Xiao He, Lei Yu, Zhijun Tu, Mingrui Zhu, Nannan Wang, Xinbo Gao, and Jie Hu. Diff-MoE: Diffusion transformer with time-aware and space-adaptive experts. In Proceedings of the International Conference on Machine Learning, 2025.

Hyungjin Chung, Jeongsol Kim, Geon Yeong Park, Hyelin Nam, and Jong Chul Ye. CFG++: Manifold-constrained classifier free guidance for diffusion models. In International Conference on Learning Representations, 2025.

Weichen Fan, Chenyang Si, Ziqi Liu, and Ziwei Liu. CFG-Zero\*: Improved classifier-free guidance for flow matching models. arXiv preprint arXiv:2503.18886, 2025.

William Fedus, Barret Zoph, and Noam Shazeer. Switch transformers: Scaling to trillion parameter models with simple and efficient sparsity. Journal ofMachine Learning Research, 2022.

Zhengcong Fei, Mingyuan Fan, Changqian Yu, Debang Li, and Junshi Huang. Scaling diffusion transformers to 16 billion parameters. arXiv preprint arXiv:2407.11633, 2024.

Dhruba Ghosh, Hannaneh Hajishirzi, and Ludwig Schmidt. GenEval: An object-focused framework for evaluating text-to-image alignment. In Advances in Neural Information Processing Systems, 2023.

Team Happyhorse. Happyhorse 1.0, 2026. URL https://www.happyhorse.com/.

Martin Heusel, Hubert Ramsauer, Thomas Unterthiner, Bernhard Nessler, and Sepp Hochreiter. GANs trained by a two time-scale update rule converge to a local Nash equilibrium. In Advances in Neural Information Processing Systems, 2017.

Jonathan Ho and Tim Salimans. Classifier-free diffusion guidance. arXiv preprint arXiv:2207.12598, 2022.

Susung Hong, Gyuseong Lee, Wooseok Jang, and Seungryong Kim. Improving sample quality of diffusion models using self-attention guidance. In Proceedings of the IEEE/CVF International Conference on Computer Vision, 2023.

Xiwei Hu, Rui Wang, Yixiao Fang, Bin Fu, Pei Cheng, and Gang Yu. ELLA: Equip diffusion models with LLM for enhanced semantic alignment. arXiv preprint arXiv:2403.05135, 2024.

Cristian Perez Jensen and Seyedmorteza Sadat. Efficient distillation of classifier-free guidance using adapters. Transactions on Machine Learning Research, 2025.

Tero Karras, Miika Aittala, Tuomas Kynka¨anniemi, Jaakko Lehtinen, Timo Aila, and Samuli Laine.¨ Guiding a diffusion model with a bad version of itself. In Advances in Neural Information Processing Systems, 2024.

Tuomas Kynka¨anniemi, Miika Aittala, Tero Karras, Samuli Laine, Timo Aila, and Jaakko Lehtinen.¨ Applying guidance in a limited interval improves sample and distribution quality in diffusion models. In Advances in Neural Information Processing Systems, 2024.

Dmitry Lepikhin, HyoukJoong Lee, Yuanzhong Xu, Dehao Chen, Orhan Firat, Yanping Huang, Maxim Krikun, Noam Shazeer, and Zhifeng Chen. GShard: Scaling giant models with conditional computation and automatic sharding. In International Conference on Learning Representations, 2021.

Chenliang Li, Haiyang Xu, Junfeng Tian, Wei Wang, Ming Yan, Bin Bi, Jiabo Ye, He Chen, Guohai Xu, Zheng Cao, Ji Zhang, Songfang Huang, Fei Huang, Jingren Zhou, and Luo Si. mPLUG: Effective and efficient vision-language learning by cross-modal skip-connections. In Proceedings of the 2022 Conference on Empirical Methods in Natural Language Processing, 2022.

Xiang Li, Rongrong Wang, and Qing Qu. Towards understanding the mechanisms of classifier-free guidance. In Advances in Neural Information Processing Systems, 2025.

Yaron Lipman, Ricky T. Q. Chen, Heli Ben-Hamu, Maximilian Nickel, and Matt Le. Flow matching for generative modeling. In International Conference on Learning Representations, 2023.

Xingchao Liu, Chengyue Gong, and Qiang Liu. Flow straight and fast: Learning to generate and transfer data with rectified flow. In International Conference on Learning Representations, 2023.

Ilya Loshchilov and Frank Hutter. Decoupled weight decay regularization. In International Conference on Learning Representations, 2019.

Chenlin Meng, Robin Rombach, Ruiqi Gao, Diederik P. Kingma, Stefano Ermon, Jonathan Ho, and Tim Salimans. On distillation of guided diffusion models. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2023.

Alex Nichol, Prafulla Dhariwal, Aditya Ramesh, Pranav Shyam, Pamela Mishkin, Bob McGrew, Ilya Sutskever, and Mark Chen. GLIDE: Towards photorealistic image generation and editing with text-guided diffusion models. In Proceedings of the International Conference on Machine Learning, 2022.

William Peebles and Saining Xie. Scalable diffusion models with transformers. In Proceedings of the IEEE/CVF International Conference on Computer Vision, 2023.

Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, Gretchen Krueger, and Ilya Sutskever. Learning transferable visual models from natural language supervision. In Proceedings ofthe International Conference on Machine Learning, 2021.

Seyedmorteza Sadat, Jakob Buhmann, Derek Bradley, Otmar Hilliges, and Romann M. Weber. CADS: Unleashing the diversity of diffusion models through condition-annealed sampling. In International Conference on Learning Representations, 2024.

Seyedmorteza Sadat, Otmar Hilliges, and Romann M. Weber. Eliminating oversaturation and artifacts of high guidance scales in diffusion models. In International Conference on Learning Representations, 2025a.

Seyedmorteza Sadat, Manuel Kansy, Otmar Hilliges, and Romann M. Weber. No training, no problem: Rethinking classifier-free guidance for diffusion models. In International Conference on Learning Representations, 2025b.

Christoph Schuhmann, Romain Beaumont, Richard Vencu, Cade Gordon, Ross Wightman, Mehdi Cherti, Theo Coombes, Aarush Katta, Clayton Mullis, Mitchell Wortsman, Patrick Schramowski, Srivatsa Kundurthy, Katherine Crowson, Ludwig Schmidt, Robert Kaczmarczyk, and Jenia Jitsev. LAION-5B: An open large-scale dataset for training next generation image-text models. In Advances in Neural Information Processing Systems, 2022.

Team Seedance. Seedance 2.0: Advancing video generation for world complexity, 2026. URL https://arxiv.org/abs/2604.14148.

Dazhong Shen, Guanglu Song, Zeyue Xue, Fu-Yun Wang, and Yu Liu. Rethinking the spatial inconsistency in classifier-free diffusion guidance. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2024.

Haotian Sun, Tao Lei, Bowen Zhang, Yanghao Li, Haoshuo Huang, Ruoming Pang, Bo Dai, and Nan Du. EC-DIT: Scaling diffusion transformers with adaptive expert-choice routing. In International Conference on Learning Representations, 2025.

Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez, Łukasz Kaiser, and Illia Polosukhin. Attention is all you need. In Advances in Neural Information Processing Systems, 2017.

Yujie Wei, Shiwei Zhang, Hangjie Yuan, Yujin Han, Zhekai Chen, Jiayu Wang, Difan Zou, Xihui Liu, Yingya Zhang, Yu Liu, and Hongming Shan. Routing matters in MoE: Scaling diffusion transformers with explicit routing guidance. In International Conference on Learning Representations, 2026.

Zeyue Xue, Guanglu Song, Qiushan Guo, Boxiao Liu, Zhuofan Zong, Yu Liu, and Ping Luo. RAPHAEL: Text-to-image generation via large mixture of diffusion paths. In Advances in Neural Information Processing Systems, 2023.

Yike Yuan, Ziyu Wang, Zihao Huang, Defa Zhu, Xun Zhou, Jingyi Yu, and Qiyang Min. Expert race: A flexible routing strategy for scaling diffusion transformer with mixture of experts. In Proceedings ofthe International Conference on Machine Learning, 2025.

Candi Zheng and Yuan Lan. Characteristic guidance: Non-linear correction for diffusion model at large guidance scale. In Proceedings ofthe International Conference on Machine Learning, 2024.

Youwei Zheng, Yuxi Ren, Xin Xia, Xuefeng Xiao, and Xiaohua Xie. Dense2MoE: Restructuring diffusion transformer to MoE for efficient text-to-image generation. In Proceedings of the IEEE/CVF International Conference on Computer Vision, 2025.

Yanqi Zhou, Tao Lei, Hanxiao Liu, Nan Du, Yanping Huang, Vincent Zhao, Andrew M Dai, Quoc V Le, James Laudon, et al. Mixture-of-experts with expert choice routing. In Advances in Neural Information Processing Systems, 2022.

Barret Zoph, Irwan Bello, Sameer Kumar, Nan Du, Yanping Huang, Jeff Dean, Noam Shazeer, and William Fedus. ST-MoE: Designing stable and transferable sparse expert models. arXiv preprint arXiv:2202.08906, 2022.

## A THEORETICAL ANALYSIS

This section records the identities used in section 3 of the main paper. CFG would mix the layer writes $v _ { c } = f _ { \mathrm { M o E } } ( h _ { c } )$ and $v _ { u } = f _ { \mathrm { M o E } } ( h _ { u } )$ . Notation follows the main text: $S _ { c } = \operatorname { s u p p } g ( h _ { c } )$ $S _ { u } = { \mathrm { s u p p ~ } } g ( h _ { u } ) , D _ { \mathrm { r o u t e } } = 1 - | S _ { c } \cap S _ { u } | / k$ , and

$$
E _ { i } ( h ) = W _ { 2 } ^ { ( i ) } \sigma ( W _ { 1 } ^ { ( i ) } h + b _ { 1 } ^ { ( i ) } ) + b _ { 2 } ^ { ( i ) } , \quad W _ { 2 } ^ { ( i ) } \in \mathbb { R } ^ { d _ { h } \times d _ { f } } .\tag{11}
$$

When a shared expert is present it is included in $f _ { \mathrm { M o E } }$ and in the stacked activations below.

## A.1 INDEPENDENT ROUTING, DIFFERENT SUBSPACES

The two CFG branches route independently, so $S _ { c }$ and $S _ { u }$ need not coincide. Stack the minibatch writes into $F _ { c } , F _ { u } \in \mathbb { R } ^ { d _ { h } \times n } , n = \mathbf { \dot { B } } T$ . Let Ω be a set of $m = \mathrm { m i n } ( n , \lfloor d _ { h } / k _ { \mathrm { s p l i t } } \rfloor )$ ) column indices (Eq. (8) of the main paper), and define the conditional activation subspace

$$
\mathcal V _ { S _ { c } } : = \mathrm { s p a n } ( F _ { c , \Omega } ) , \qquad P _ { c } : = \hat { F } _ { c } \hat { F } _ { c } ^ { \top } ,\tag{12}
$$

where $\hat { F } _ { c }$ is an orthonormal basis of $\nu _ { S _ { c } }$ from thin QR of $F _ { c , \Omega }$ (stop-gradient on $F _ { c } ,$ as in Eq. (9) of the main paper). Then dim $\begin{array} { r } { \partial _ { S _ { c } } \ \le \ m , } \end{array}$ , so $\gamma _ { S _ { c } }$ is a proper subspace whenever $m \textless d _ { h }$ . It is not claimed to contain every column of $F _ { c }$ when $n > m$ . The object SAGE uses is this realized subspace $\gamma _ { S _ { c } }$

Lemma A.1 (CFG closure). $I f v _ { c } , v _ { u } \in \mathcal { A } _ { \cdot }$ for any subspace $\mathcal { A } \subseteq \mathbb { R } ^ { d _ { h } }$ , then $v _ { \mathrm { c f g } } = ( 1 + s ) v _ { c } - s v _ { u } \in$ $\mathcal { A } .$

Definition A.2 (Subspace leak). Let $v _ { \mathrm { i d e a l } } : = ( 1 + s ) v _ { c } - s P _ { c } v _ { u }$ . The leak is the difference

$$
\delta ( s ) : = v _ { \mathrm { c f g } } - v _ { \mathrm { i d e a l } } = - s ( I - P _ { c } ) v _ { u } .\tag{13}
$$

Equivalently,

$$
v _ { \mathrm { c f g } } = ( 1 + s ) v _ { c } - s P _ { c } v _ { u } + \delta ( s ) , \qquad \| \delta ( s ) \| = s \| ( I - P _ { c } ) v _ { u } \| .\tag{14}
$$

So $\delta ( s ) = 0$ if and only i ${ \mathrm { ~ f ~ } } v _ { u } \in \mathcal { V } _ { S _ { c } }$ , and the leak, when present, is exactly linear in s. Independent routing makes $v _ { u } \notin \mathcal { V } _ { S _ { c } }$ typical: the unconditional branch can write in directions the conditional activations did not realize. SAGE drives $\delta  0 ;$ ; it does not claim $v _ { \mathrm { c f g } } \in \mathcal { V } _ { S }$ when $n > m$ , because tokens of $F _ { c }$ outside Ω need not lie in $\nu _ { S _ { c } }$ . Closure (theorem $\mathbf { A . } 1 )$ is the motivation for aligning $v _ { u }$ to a realized conditional subspace, not a claim that the implemented section contains the whole batch.

## A.2 WHAT SAGE MINIMIZES

The implemented loss Eq. (9) of the main paper is the relative energy of $F _ { u }$ outside $\nu _ { S _ { c } }$ . Writing ∆ for the matrix whose columns are the per-token leaks (13), the identity

$$
\Delta = - s ( I - P _ { c } ) F _ { u } , \qquad \| \Delta \| _ { F } = s \| ( I - P _ { c } ) F _ { u } \| _ { F }\tag{15}
$$

holds at every parameter value. With

$$
\mathcal { L } _ { \mathrm { s a g e } } = \frac { 1 } { \Vert F _ { u } \Vert _ { F } ^ { 2 } } \Vert ( I - P _ { c } ) F _ { u } \Vert _ { F } ^ { 2 } ,\tag{16}
$$

one has $\lVert \Delta \rVert _ { F } = s \sqrt { \mathcal { L } _ { \mathrm { s a g e } } } \left. F _ { u } \right. _ { F } .$

Proposition A.3 (SAGE controls leak energy). At any parameter value, $\| \Delta \| _ { F } = s \| ( I - P _ { c } ) F _ { u } \| _ { F }$ In particular, $i f \mathcal { L } _ { \mathrm { s a g e } } ( \theta ^ { * } ) = \varepsilon ^ { 2 }$ , then

$$
\| \Delta \| _ { F } = s \varepsilon \| F _ { u } \| _ { F } .\tag{17}
$$

No Markov bound and no $\lambda ^ { - 1 / 2 }$ rate are claimed: λ only trades $\mathcal { L } _ { \mathrm { s a g e } }$ against $\mathcal { L } _ { \mathrm { F M } }$ during optimization (section 4.2.3 of the main paper). Equation (15) is the identity that Eq. (9) of the main paper drives to zero.

## A.3 EXCLUSIVE EXPERTS

Split $v _ { u } = v _ { u } ^ { \mathrm { s h } } + v _ { u } ^ { \mathrm { e x } }$ into experts in $S _ { c } \cap S _ { u }$ and in $S _ { u } \setminus S _ { c }$

$$
v _ { u } ^ { \mathrm { e x } } = \sum _ { j \in S _ { u } \backslash S _ { c } } g _ { j } ^ { u } E _ { j } ( h _ { u } ) .\tag{18}
$$

This extra write is MoE-specific: a dense FFN has $S _ { c } = S _ { u } = \{ 1 \}$ , hence $D _ { \mathrm { r o u t e } } \equiv 0$ and $v _ { u } ^ { \mathrm { e x } } = 0$ The split does not bound $\| ( I - P _ { c } ) v _ { u } \|$ . Shared-set writes $E _ { i } ( h _ { u } )$ need not lie in $\gamma _ { S _ { c } }$ (which is spanned by $E _ { i } ( h _ { c } )$ and other tokens), and exclusive writes need not be orthogonal to $\nu _ { S _ { c } }$ . Independent routing only makes a residual outside $\nu _ { S _ { c } }$ typical. Dense CFG can still fail for other reasons; SAGE targets the component of $v _ { u }$ outside $\nu _ { S _ { c } }$

## A.4 SHARED EXPERT

A shared expert $E _ { 0 }$ is evaluated on both branches and does not enter $D _ { \mathrm { r o u t e } } .$ . It reduces exclusive routed mass in $v _ { u } ^ { \mathrm { e x } }$ , but it does not put $v _ { u }$ in $\mathcal { V } _ { S _ { c } } \colon E _ { 0 } \left( h _ { u } \right)$ still differs from $E _ { 0 } ( h _ { c } )$ , and exclusive routed writes remain. Our 1B implementation applies Eq. (9) of the main paper to the full MoE output (shared plus routed). Both experiments minimize the same loss: the toy model has $T = 1$ and uses every available conditional write, so $k _ { \mathrm { s p l i t } }$ is not needed; the 1B model has $n = B T \gg d _ { h }$ and uses $k _ { \mathrm { s p l i t } } { = } 2$ , so only the rank-m section is a proper subspace.

## B ADDITIONAL EXPERIMENTAL RESULTS

## B.1 DPG-BENCH EVALUATION AND CATEGORY BREAKDOWN

We evaluate the Baseline, SAGE (λ=1), and the KL routing constraint $( \lambda _ { \mathrm { k l } } = 1 )$ on DPG-Bench (Hu et al., 2024) at CFG scales (Ho & Salimans, 2022) $s \in \{ 1 , 3 , 5 , 7 , 1 0 \}$ , using mPLUG (Li et al., 2022) for visual question answering $\mathrm { ( V Q A ) }$ . For each prompt, we generate four $3 2 0 \times 3 2 0$ images using 50 sampling steps.

Table 4 reports the five top-level categories: Global, Entity, Attribute, Relation, and Other. SAGE exceeds the Baseline in all five categories at every tested $s \geq 3$ . At s=3, the largest gains are in Other (+6.0 pp) and Entity (+4.7 pp); at s=10, the gains reach +10.8 pp for Entity and +8.1 pp for Relation. Thus, the advantage under stronger guidance extends across object presence, attributes, and relations, rather than being confined to one category.

Table 5 gives all 13 second-level categories at $s { = } 3$ and $s { = } 5$ . SAGE improves over the Baseline in every subcategory at both scales. The gains in counting are +6.4 and +12.4 pp, respectively, while whole-entity presence improves by +5.1 and $+ 8 . 6 \mathrm { p p }$ . Size, texture, and spatial relations also improve at both scales, showing that the benefits cover multiple aspects of compositional generation.

## B.2 GENEVAL PER-TASK BREAKDOWN

Table 6 reports the full per-task breakdown on GenEval (Ghosh et al., 2023) for all three methods at every guidance scale $s \in \{ 1 , 3 , 5 , 7 , 1 0 \}$ ; the “Overall” column is the mean over the six tasks and reproduces the GenEval rows of table 2 of the main paper. Focusing on the shared operating point $s { = } 3$ , SAGE improves four of the six categories, with the largest gain by far in color-attribute binding (+11.5 pp), followed by two-object composition (+3.5 pp) and single-object presence (+3.4 pp); it shows small decreases in counting $( - 1 . 9 \mathrm { p p } )$ and colors $( - 2 . 4 \mathrm { p p } )$ . The gains therefore concentrate exactly on the compositional categories that depend on precise conditional control, which is where a write outside the conditional subspace is most damaging. Across the whole sweep, SAGE leads the Baseline on the Overall score at every $s \geq 3 ,$ while the KL routing constraint degrades every category at every scale, consistent with the loss of expert specialization discussed in section 4.2.2 of the main paper.

## B.3 ADDITIONAL ABLATIONS

The two ablations below complement those reported in section 4.2.3 of the main paper, use the same setup as in section 4.2.1 of the main paper, and are likewise evaluated at guidance scales $s \in \{ 3 , 5 \}$

Table 4: DPG-Bench category scores (%) across CFG scales. Category scores aggregate raw per-question answers over all four images per prompt. Overall uses dependency-aware scoring and matches table 2 of the main paper; it is not the category average. Higher is better.
<table><tr><td>Method</td><td>S</td><td>Global</td><td>Entity</td><td>Attribute</td><td>Relation</td><td>Other</td><td>Overall</td></tr><tr><td rowspan="5">Baseline</td><td>1</td><td>70.06</td><td>68.04</td><td>73.16</td><td>84.98</td><td>43.00</td><td>54.77</td></tr><tr><td>3</td><td>73.10</td><td>73.87</td><td>81.00</td><td>85.71</td><td>57.40</td><td>61.76</td></tr><tr><td>5</td><td>65.43</td><td>69.66</td><td>77.44</td><td>82.28</td><td>54.60</td><td>56.33</td></tr><tr><td>7</td><td>61.02</td><td>66.10</td><td>75.00</td><td>79.96</td><td>52.00</td><td>52.38</td></tr><tr><td>10</td><td>59.65</td><td>62.59</td><td>71.66</td><td>77.90</td><td>50.60</td><td>48.56</td></tr><tr><td rowspan="5">SAGE (λ=1)</td><td>1</td><td>69.45</td><td>65.94</td><td>71.04</td><td>84.91</td><td>42.30</td><td>52.30</td></tr><tr><td>3</td><td>76.44</td><td>78.58</td><td>83.36</td><td>88.65</td><td>63.40</td><td>67.49</td></tr><tr><td>5</td><td>73.40</td><td>77.70</td><td>81.54</td><td>86.95</td><td>65.10</td><td>65.63</td></tr><tr><td>7</td><td>70.44</td><td>76.10</td><td>80.20</td><td>86.60</td><td>63.90</td><td>63.36</td></tr><tr><td>10</td><td>66.49</td><td>73.34</td><td>77.21</td><td>85.96</td><td>62.00</td><td>59.55</td></tr><tr><td rowspan="5"> ${ \mathrm { K L } } \left( \lambda _ { \mathrm { k l } } { = } 1 \right)$ </td><td>1</td><td>63.30</td><td>55.26</td><td>66.73</td><td>78.74</td><td>29.30</td><td>40.95</td></tr><tr><td>3</td><td>67.63</td><td>64.66</td><td>77.45</td><td>80.32</td><td>48.90</td><td>50.65</td></tr><tr><td>5</td><td>64.44</td><td>61.86</td><td>74.45</td><td>77.10</td><td>45.60</td><td>46.82</td></tr><tr><td>7</td><td>60.56</td><td>59.56</td><td>72.40</td><td>74.76</td><td>44.10</td><td>44.38</td></tr><tr><td>10</td><td>59.80</td><td>56.13</td><td>69.33</td><td>73.60</td><td>42.20</td><td>41.08</td></tr></table>

Table 5: Fine-grained DPG-Bench scores (%) at $s \in \{ 3 , 5 \}$ . All 13 second-level categories use the same four-image, pre-dependency aggregation as table 4. SAGE and KL use λ=1 and $\lambda _ { \mathrm { k l } } { = } 1$ respectively. Higher is better.
<table><tr><td></td><td></td><td colspan="3">s=3</td><td colspan="3">s=5</td></tr><tr><td>Category</td><td>Subcategory</td><td>Baseline</td><td>SAGE</td><td>KL</td><td>Baseline</td><td>SAGE</td><td>KL</td></tr><tr><td>Global</td><td>一</td><td>73.10</td><td>76.44</td><td>67.63</td><td>65.43</td><td>73.40</td><td>64.44</td></tr><tr><td rowspan="3">Entity</td><td>Whole</td><td>73.89</td><td>79.03</td><td>63.72</td><td>69.43</td><td>78.02</td><td>60.59</td></tr><tr><td>Part</td><td>77.25</td><td>80.08</td><td>71.58</td><td>75.44</td><td>81.01</td><td>71.09</td></tr><tr><td>State</td><td>71.91</td><td>75.53</td><td>65.63</td><td>67.66</td><td>74.29</td><td>63.20</td></tr><tr><td rowspan="5">Attribute</td><td>Color</td><td>87.90</td><td>88.52</td><td>84.71</td><td>85.32</td><td>87.87</td><td>82.35</td></tr><tr><td>Shape</td><td>78.93</td><td>80.57</td><td>72.93</td><td>79.59</td><td>80.57</td><td>78.17</td></tr><tr><td>Size</td><td>62.71</td><td>66.32</td><td>55.58</td><td>56.20</td><td>64.57</td><td>52.38</td></tr><tr><td>Texture</td><td>77.59</td><td>81.61</td><td>75.29</td><td>73.15</td><td>78.60</td><td>70.46</td></tr><tr><td>Other</td><td>77.47</td><td>80.43</td><td>72.39</td><td>73.00</td><td>77.72</td><td>69.26</td></tr><tr><td rowspan="2">Relation</td><td>Spatial</td><td>86.10</td><td>89.17</td><td>80.72</td><td>82.82</td><td>87.46</td><td>77.67</td></tr><tr><td>Non-spatial</td><td>79.72</td><td>80.66</td><td>74.21</td><td>74.06</td><td>79.25</td><td>68.40</td></tr><tr><td rowspan="2">Other</td><td>Count</td><td>53.88</td><td>60.25</td><td>43.38</td><td>49.63</td><td>62.00</td><td>39.13</td></tr><tr><td>Text</td><td>71.50</td><td>76.00</td><td>71.00</td><td>74.50</td><td>77.50</td><td>71.50</td></tr></table>

Both the activation-frequency ablation (table 7) and the shared-expert and ProMoE (Wei et al., 2026) comparison (table 8) report DPG-Bench and GenEval scores at each scale.

Activation frequency. We investigate whether SAGE needs to be active at every training step or whether intermittent activation suffices. Because $\mathcal { L } _ { \mathrm { s a g e } }$ is a minibatch subspace constraint rather than a per-token identity, it can be amortized across steps: intermittent application still reshapes how the two branches write. The table shows that activating SAGE every second or every fifth step retains nearly all of the from-scratch gain, which is consistent with a batch-level section rather than a per-token constraint. Table 7 compares activation at every step with activation at every second and every fifth step.

Shared expert as an implicit bridge. Two factors explain why existing MoE-DiTs (Fei et al., 2024) produce acceptable images despite the subspace leak. The first is that moderate CFG hides the problem: at typical operating points (s=3 to 5), the leak is present but not catastrophic, so the degradation without SAGE is tolerable, even though the Baseline still benefits substantially from SAGE at these scales (+5.73 DPG at s=3; table 2 of the main paper); the gap widens dramatically at s=7 to 10, which matters increasingly as applications demand high guidance (double-CFG, style transfer, compositional prompting). The second is that the shared expert is a dense-style path evaluated on both branches, which reduces exclusive routed mass (section A.4) without putting $v _ { u }$ in $\nu _ { S _ { c } }$ . We isolate this effect in table 8 by removing the shared expert entirely and by comparing with ProMoE (which fixes one expert for the unconditional branch).

Table 6: GenEval per-task accuracy (%) across CFG scales. Per-category GenEval scores for all three methods at every guidance scale $s ;$ the “Overall” column is the mean over the six tasks and matches the GenEval rows of table 2 of the main paper.
<table><tr><td>Method</td><td>S</td><td>Single Obj.</td><td>Two Obj.</td><td>Counting</td><td>Colors</td><td>Position</td><td>Color Attr.</td><td>Overall</td></tr><tr><td rowspan="5">Baseline</td><td>1</td><td>50.0</td><td>18.2</td><td>20.3</td><td>44.1</td><td>25.3</td><td>23.8</td><td>30.3</td></tr><tr><td>3</td><td>79.4</td><td>46.0</td><td>40.0</td><td>79.3</td><td>51.5</td><td>46.8</td><td>57.1</td></tr><tr><td>5</td><td>82.5</td><td>43.9</td><td>40.3</td><td>75.8</td><td>48.8</td><td>41.8</td><td>55.5</td></tr><tr><td>7</td><td>79.4</td><td>43.9</td><td>40.3</td><td>68.6</td><td>49.8</td><td>37.3</td><td>53.2</td></tr><tr><td>10</td><td>73.1</td><td>42.4</td><td>33.1</td><td>58.2</td><td>42.5</td><td>33.3</td><td>47.1</td></tr><tr><td rowspan="5">SAGE (λ=1)</td><td>1</td><td>42.5</td><td>17.2</td><td>18.1</td><td>36.7</td><td>20.3</td><td>24.0</td><td>26.5</td></tr><tr><td>3</td><td>82.8</td><td>49.5</td><td>38.1</td><td>76.9</td><td>53.3</td><td>58.3</td><td>59.8</td></tr><tr><td>5</td><td>80.0</td><td>48.7</td><td>45.3</td><td>75.3</td><td>52.8</td><td>54.0</td><td>59.3</td></tr><tr><td>7</td><td>76.9</td><td>50.8</td><td>39.4</td><td>72.3</td><td>53.5</td><td>49.5</td><td>57.1</td></tr><tr><td>10</td><td>74.7</td><td>51.3</td><td>39.4</td><td>63.3</td><td>46.3</td><td>40.3</td><td>52.5</td></tr><tr><td rowspan="5">KL (λkl=1)</td><td>1</td><td>23.8</td><td>6.8</td><td>10.6</td><td>23.4</td><td>8.0</td><td>10.3</td><td>13.8</td></tr><tr><td>3</td><td>60.3</td><td>23.5</td><td>23.8</td><td>55.6</td><td>27.3</td><td>25.5</td><td>36.0</td></tr><tr><td>5</td><td>55.0</td><td>24.2</td><td>27.8</td><td>49.2</td><td>28.0</td><td>23.0</td><td>34.5</td></tr><tr><td>7</td><td>55.6</td><td>20.2</td><td>25.9</td><td>44.1</td><td>29.0</td><td>18.5</td><td>32.2</td></tr><tr><td>10</td><td>56.3</td><td>19.2</td><td>21.9</td><td>39.9</td><td>23.8</td><td>16.3</td><td>29.5</td></tr></table>

Table 7: Ablation: SAGE activation frequency $( s \in \{ 3 , 5 \} )$ . DPG-Bench and GenEval scores for SAGE activation at every step, every second step, and every fifth step.
<table><tr><td></td><td colspan="2">s=3</td><td colspan="2">s=5</td></tr><tr><td>Interval</td><td>DPG</td><td>GenEval</td><td>DPG</td><td>GenEval</td></tr><tr><td>Every step</td><td>67.49</td><td>59.8</td><td>65.63</td><td>59.3</td></tr><tr><td>Every 2 steps</td><td>67.21</td><td>59.9</td><td>65.84</td><td>59.1</td></tr><tr><td>Every 5 steps</td><td>66.73</td><td>59.4</td><td>65.12</td><td>58.8</td></tr></table>

Table 8: Ablation: shared expert and ProMoE $( s \in \{ 3 , 5 \} )$ . Comparison of MoE configurations with and without SAGE against ProMoE at both guidance scales.
<table><tr><td></td><td colspan="2">s=3</td><td colspan="2">s=5</td></tr><tr><td>Configuration</td><td>DPG</td><td>GenEval</td><td>DPG</td><td>GenEval</td></tr><tr><td>8E + 1 shared, no SAGE</td><td>61.76</td><td>57.1</td><td>56.33</td><td>55.5</td></tr><tr><td>8E + 1 shared, SAGE (λ=1)</td><td>67.49</td><td>59.8</td><td>65.63</td><td>59.3</td></tr><tr><td>8E, no shared, no SAGE</td><td>57.82</td><td>53.0</td><td>52.41</td><td>51.4</td></tr><tr><td>8E, no shared, SAGE (λ=1)</td><td>64.05</td><td>56.4</td><td>62.18</td><td>55.6</td></tr><tr><td>ProMoE</td><td>63.12</td><td>58.4</td><td>57.95</td><td>56.8</td></tr></table>

The results show the expected pattern: removing the shared expert worsens baseline performance (more exclusive routed mass is exposed), while adding SAGE still recovers most of the gap. Pro-MoE, which reserves one expert for the unconditional branch and thus plays a role analogous to the shared expert without aligning the remaining routed activations, provides partial relief but does not match SAGE.

## B.4 SAGE PRESERVES ROUTING DIVERSITY

![](images/023bda0aee2e1536ee3b5b87e647308e40c5cfd5de0c3c9596f1412f853645dd.jpg)

![](images/7c743b71e5ceaaf1e28bb21610074469e495f7fa14e50292868231eabab91ea9.jpg)

![](images/da6d9adcf1a4170b4e7bc057bb67eaaeaa6ac779953937eb130c01663ca0454a.jpg)

![](images/89cdd9902c5a1364ba40d1246745250a22c0d87cdd785555ec0292483068f51e.jpg)

Figure 5: Conditional vs. unconditional router similarity across CFG scales. SAGE (red) tracks the Baseline (blue) almost exactly on all four measures, whereas the KL routing constraint (gray, dashed) forces the two branches to agree (higher Jaccard/agreement, near-zero Jensen–Shannon (JS) divergence). Higher values indicate greater similarity for Top-2 Jaccard, Top-1 agreement, and routing-probability cosine similarity; lower values indicate greater similarity for JS divergence. Error bars are standard errors over 500 prompts and are smaller than the markers.  
![](images/6e988db71369c1579a11385d194ccffc07519072240a711aa65c560cecd007f2.jpg)  
Figure 6: Conditional minus unconditional expert load per layer and expert at s=3, averaged over 500 prompts and shown on a shared color scale. Baseline and SAGE exhibit comparably large conditional/unconditional load differences (routing diversity preserved), whereas the KL constraint suppresses them almost entirely (routing forced to agree).

A central design goal of SAGE is to fix the subspace leak without touching the router: unlike the KL routing constraint, which explicitly forces the two branches to route alike, SAGE only aligns unconditional activations to the conditional subspace (section 3.2 of the main paper). Here we verify empirically that SAGE indeed leaves routing behavior essentially unchanged. Using the 1B MoE DiT of section 4.2.1 of the main paper, we run each model on 500 prompts across four guidance scales and record, at every denoising step and layer, the conditional and unconditional Top-2 expert selections and routing distributions. We evaluated various values for $\lambda _ { \mathrm { k l } }$ (0.01, 0.1, and 1) and observed consistent results across all settings. Decreasing $\lambda _ { \mathrm { k l } }$ did not yield improvements, as the routing health and benchmark scores still fell short of the baseline.

Conditional/unconditional routing similarity is unchanged. Figure 5 reports four similarity measures between the conditional and unconditional branches as a function of the guidance scale. Across all scales, SAGE is statistically indistinguishable from the Baseline on every measure: Top-2 Jaccard, Top-1 agreement, and routing-probability cosine similarity all coincide within noise, and the JS divergence between the two routing distributions is, if anything, marginally higher for SAGE than for the Baseline, indicating that SAGE does not push the branches toward common experts. The KL constraint behaves oppositely, driving Jaccard and agreement up toward 0.95 and the JS divergence down toward zero, that is, collapsing the two branches onto nearly identical routes.

Per-expert load divergence is preserved. Figure 6 visualizes the per-layer, per-expert difference between the conditional and unconditional expert loads at $s { = } 3 ,$ on a shared color scale. The Baseline and SAGE panels are equally vivid, with mean absolute load differences of 0.0144 and 0.0148, respectively, confirming that the two branches continue to recruit visibly different experts under SAGE. By contrast, the KL panel is almost uniformly white (mean absolute difference 0.0034, roughly 4× smaller), showing that routing agreement has been enforced at the cost of expert specialization. Together, the two figures confirm the claim of section 4.2.2 of the main paper: SAGE reduces the residual outside $\nu _ { S _ { c } }$ while leaving routing diversity intact, which is precisely why it avoids the specialization collapse that makes the KL constraint counterproductive.

Table 9: FID across CFG scales. Lower is better. $\Delta$ is Baseline − SAGE, positive values indicate a SAGE improvement.
<table><tr><td>Method</td><td> $s { = } 1$ </td><td> $s { = } 3$ </td><td> $s { = } 5$ </td><td> $s { = } 7$ </td><td> $s { = } 1 0$ </td></tr><tr><td>Baseline</td><td>13.90</td><td>18.54</td><td>26.11</td><td>33.11</td><td>45.38</td></tr><tr><td>SAGE (λ=1)</td><td>13.18</td><td>14.80</td><td>20.28</td><td>25.54</td><td>37.12</td></tr><tr><td>∆ (Base - SAGE)</td><td>+0.72</td><td>+3.74</td><td>+5.83</td><td>+7.57</td><td>+8.26</td></tr></table>

## B.5 FID ACROSS CFG SCALES

To confirm that the CFG-robustness benefit of SAGE also holds for distributional image fidelity, we report the Frechet Inception Distance (FID) (Heusel et al., 2017) as a function of the guidance scale´ $s \in \{ 1 , 3 , 5 , 7 , 1 0 \}$

We sample a fixed set of 10,000 prompts from the in-domain T2I test set and pair each prompt with its ground-truth image to form an in-domain reference set. Reference images are decoded, EXIForiented, center-cropped to a square, and bicubic-resized to 320 × 320 to match the generation resolution; each model then generates one 320 × 320 image per prompt with 50 sampling steps and a fixed per-prompt seed. FID is computed from standard Inception-V3 features (2048-d pool3, bicubic resize to 299), using the identical 10,000 prompt IDs for the reference set and for every model and scale, so all conditions are strictly paired. The Baseline and SAGE checkpoints are trained 200,000 steps.

Table 9 mirrors the trend of the alignment metrics. At s=1 (no guidance) SAGE performs better than baseline (13.90 vs. 13.18). As the guidance scale grows, the Baseline FID degrades steeply (+31.5 from s=1 to s=10), whereas SAGE degrades markedly more slowly (+23.94); SAGE therefore attains a better FID at every tested scale.

## C TRAINING DYNAMICS AND HEALTH

We further compare the optimization dynamics of the Baseline and SAGE over a 200,000-step training schedule. The purpose of this analysis is twofold: to verify that the additional alignment objective does not compromise the primary flow-matching objective or the health of MoE routing, and to confirm that it directly reduces the subspace residual targeted by SAGE.

Figure 7(a) reports the primary flow-matching objective $\mathcal { L } _ { \mathrm { F M } }$ defined in Eq. (1) of the main paper. Both methods exhibit the same rapid initial decrease and converge to nearly identical values. Because the two curves overlap at the scale of the full plot, we additionally show a 4,000-step magnified interval. The zoomed view confirms that their differences remain negligible.

Figure 7(b) shows the standard MoE auxiliary loss used to encourage balanced expert utilization. A lower and stable value indicates that routing does not concentrate excessively on a small subset of experts. The two methods follow closely matched trajectories throughout training, with SAGE maintaining a slightly lower auxiliary loss than the Baseline. This indicates that SAGE does not interfere with the existing load-balancing mechanism or induce routing collapse.

Dead expert ratio is measured in Figure 7(c). Let E denote the number of experts, N the total number of routed tokens in the current batch, and $C _ { e }$ the number of tokens assigned to expert e. We classify expert e as dead when its load is below 10% of the uniform expected load $N / \dot { E }$ , and

![](images/75dcc817164470b86214cc936fdc53ed7991a482953026b18d437bd749d4e1f5.jpg)

![](images/03579681c44375796ef8eee55d45bdf20a7f7626b5fb8ff7c24fd65e828254e1.jpg)

![](images/3e81b2622c486ebf04337a787ac40c51516555031d092e479bad7cdba6c8fb94.jpg)

![](images/f087630a59ed0067b2c55c3f987a61a8399c4c985d620886b847b5c4f7f82ad1.jpg)

![](images/dc57b65b47c193d82a4ac8b8bd606b9fa7a3d9708f9be0df332d3eb329b0536a.jpg)

![](images/dd82f7450277737d6c257798fc6a8435e81f6d73f1b50271d9f51658c43bc53e.jpg)  
Figure 7: Training dynamics of the Baseline and SAGE. (a) Flow-matching MSE loss; the inset magnifies a 4,000-step interval and shows that the two runs remain closely matched. (b) MoE load-balancing auxiliary loss. (c) Percentage of dead experts. (d–f) Mean, per-layer maximum, and per-layer minimum of the subspace residual, respectively. SAGE preserves the primary optimization and routing-health statistics while consistently reducing the alignment residual.

compute

$$
\mathrm { d e a d \mathrm { \mathrm { - } e x p e r t { - } p c t { = } } } \frac { 1 } { E } \sum _ { e = 1 } ^ { E } \mathbb { I } \bigg ( C _ { e } < 0 . 1 \frac { N } { E } \bigg ) ,\tag{19}
$$

where I(·) is the indicator function. For both methods, the dead-expert percentage remains below 5% and decreases as training proceeds. SAGE closely tracks the Baseline and does not increase the number of under-utilized experts, further confirming that its alignment constraint does not waste MoE capacity.

Figure 7(d–f) report three complementary summaries of the layer-wise alignment residual: its mean across MoE layers, its maximum, and its minimum. The Baseline residual grows over training in both the mean and worst-layer views, indicating that unconditional writes drift outside the conditional subspace $\gamma _ { S _ { c } }$ as experts specialize. In contrast, SAGE rapidly decreases the residual and keeps its mean, maximum, and minimum close to zero throughout training. The per-layer maximum is especially informative because it rules out the possibility that a small set of poorly aligned layers is hidden by averaging.
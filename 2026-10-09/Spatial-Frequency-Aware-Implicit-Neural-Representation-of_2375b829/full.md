# Spatial-Frequency-Aware Implicit Neural Representation of

Multidimensional Signals via MLP-KAN Fusion

Wen Yan, Ligen Shi, Jun Qiu, Haimiao Zhang,Member, IEEE, Lina Wu, Member, IEEE and Chang Liu\*, Member, IEEE

Abstract—Implicit Neural Representations (INRs) have emerged as a compelling paradigm for modeling multidimensional signals by mapping continuous coordinates to signal values. However, Multi-Layer Perceptrons (MLP)-based INRs inherently sufer from spectral bias, which favors low-frequency components and suppresses the reconstruction of essential highfrequency details. While existing techniques, such as Fourier feature mappings, mitigate this issue, they often rely on sensitive manual tuning and are prone to spectral artifacts. In this paper, we propose a spatial-frequency-aware INR framework that combines an MLP branch with a Kolmogorov-Arnold network (KAN) branch for complementary frequency-oriented modeling. The MLP branch provides a low-frequencyoriented representation of smooth structures, whereas the KAN branch complements localized variations and fine details. To coordinate the two branches, we integrate the discrete wavelet transform (DWT) and inverse discrete wavelet transform (IDWT) into the output fusion stage. The outputs of the two branches are decomposed into wavelet coeficients, and the corresponding coeficients are additively fused before inverse wavelet reconstruction. A wavelet-domain band-separation regularization further penalizes highfrequency responses in the MLP branch and lowfrequency responses in the KAN branch, thereby encouraging complementary frequency-oriented behavior. Experiments on 1D signals, 2D images, 3D volumes and signed distance functions, videos, and 4D lightfields demonstrate the applicability of the proposed representation across the evaluated signal modalities. Results demonstrate improved reconstruction fidelity across the evaluated signal modalities.

Index Terms—Implicit neural representations, Kolmogorov–Arnold Networks, frequency-aware modeling, wavelet decomposition, multidimensional signal reconstruction.

## I. Introduction

MPLICIT Neural Representations (INRs) have I emerged as a powerful paradigm for signal representation, replacing discrete grid-based tensors with coordinate-based neural functions [1], [2]. By parameterizing a signal as a mapping from spatial, temporal, or other coordinates to its corresponding values, INR enables continuous-coordinate querying, compact parameterization, and diferentiable signal modeling [3]. These properties have led to substantial advances in 3D surface modeling [4], [5], novel-view synthesis [6], [7], shape modeling [8], and video representation [9], [10]. Moreover, INRs have shown considerable potential for multidimensional signal reconstruction and inverse problems [11], including light-field (LF) representation and reconstruction [12], [13], disparity estimation [14], infrared and visible image fusion [15], computed tomography reconstruction [16], [17], and ring artifact removal [18]. These studies demonstrate the flexibility of INRs for representing and reconstructing signals with diferent dimensions and modalities.

Despite their advantages, MLP-based INRs inherently exhibit spectral bias [19], [20], whereby low-frequency components are learned preferentially, while high-frequency variations are more dificult to capture. Existing methods primarily attempt to overcome this behavior through coordinate encoding, activation-function design, or hybrid explicit–implicit representations. Fourier feature mappings [21] embed coordinates into a higher-frequency space, but their performance depends on the selected encoding frequencies and dimensions. SIREN [22] employs periodic activation functions, but requires carefully designed initialization and network configurations. Hybrid explicit–implicit representations, such as InstantNGP [23] and DINER [24], augment coordinate networks with trainable embeddings and introduce additional signaldependent storage overhead. Although these approaches improve high-frequency representation, they primarily ad dress spectral bias by modifying the input representation or the MLP-based coordinate network itself. Taking a diferent perspective on spectral bias, we retain and exploit the low-frequency-oriented behavior of MLP and investigate a heterogeneous representation in which diferent network structures exhibit complementary frequency-oriented modeling behaviors.

As illustrated in Fig. 1, the proposed framework combines an MLP branch and a KAN branch to exploit their complementary representation characteristics. The MLP branch provides a low-frequency-oriented representation for smooth signal components, consistent with the spectral bias of conventional MLP. This characteristic is well matched to the spectral distribution of many natural signals, in which low-frequency components account for most of the signal energy and determine the global structure, whereas high-frequency components are relatively sparse but encode perceptually important edges, textures, and localized variations. In contrast to MLP, KAN [25] parameterize network edges using learnable univariate functions, which are represented by spline basis functions in the standard spline-based implementation [26], providing greater local flexibility for representing nonlinear variations. Recent analyses further indicate that KAN can exhibit a weaker low-frequency bias under appropriate configurations [27]. Accordingly, we retain the MLP branch for the dominant smooth structures and use the KAN branch as a complementary representation pathway with relatively stronger responses to localized variations and fine-scale details. The two branches are not restricted to disjoint frequency ranges, and both contribute to the final reconstruction.

![](images/c94a02c6ce49f8022bfea323a8a2b72e0137ce8b263798f184b3df0ccb4a63ed.jpg)  
Fig. 1: Motivation of the proposed spatial–frequency-aware MLP–KAN framework. The MLP branch provides a lowfrequency-oriented representation of smooth structures, and the KAN branch provides complementary responses to localized variations and fine-scale details. DWT exposes the multiscale responses of both branches. Corresponding wavelet coeficients are combined through additive wavelet-domain fusion, and band-separation regularization encourages complementary frequency behavior between the two branches.

To coordinate the two representation pathways, we introduce wavelet-domain fusion and spectral regularization. Wavelet transforms provide localized multiresolution analysis of non-stationary signals. Based on this property, the outputs of the two branches are decomposed by DWT [28]. The corresponding wavelet coeficients generated by the MLP and KAN branches are then combined by elementwise addition, followed by IDWT reconstruction. This additive fusion allows both branches to contribute to each wavelet band and preserves the flexibility of the dualbranch representation.

Because wavelet decomposition alone does not guarantee complementary frequency responses between the two branches, we further introduce a band-separation loss that penalizes the high-frequency responses of the MLP branch and the low-frequency responses of the KAN branch.

The resulting spectral constraint strengthens the lowfrequency tendency of the MLP branch and increases the relative detail response of the KAN branch. The corresponding wavelet coeficients remain additively fused, so the regularization controls the frequency tendency of the branches without enforcing exclusive subband assignment. In this manner, the proposed framework alleviates the high-frequency fitting limitation of MLP-based INRs without discarding their advantageous low-frequency inductive bias, while preserving contributions from both branches to the final reconstruction.

In summary, the main contributions of this work are as follows:

• We propose a spatial–frequency-aware INR framework based on an MLP-KAN dual-branch architecture. The MLP branch provides a low-frequencyoriented representation of smooth components, and the KAN branch supplies complementary responses to localized and fine-scale variations. Their heterogeneous representation characteristics are jointly exploited for multidimensional signal fitting.

• We introduce a wavelet-domain coordination mechanism based on additive coeficient fusion and branchspecific band-separation regularization. Corresponding DWT coeficients from the two branches are additively fused before inverse reconstruction, and the spectral regularization suppresses high-frequency responses in the MLP branch and low-frequency responses in the KAN branch to promote complementary frequency-oriented behavior.

• We systematically evaluate the proposed framework on diverse multidimensional signal representation tasks, including 1D signals, 2D images, 3D volumes and signed distance functions, spatiotemporal videos, and 4D LFs.

## II. Related Work

This section reviews three lines of work most relevant to our method: implicit neural representations, Kolmogorov-Arnold networks, and wavelet-based multiresolution analysis.

## A. Implicit Neural Representation

To improve high-frequency modeling, existing INRs modify coordinate encodings, activation functions, or network structures. Fourier features and Random Fourier Features [21] map coordinates to higher-frequency spaces, while SIREN [22], GAUSS [29], WIRE [30], and SINC [31] use sinusoidal, Gaussian, Gabor-wavelet, or sinc activations. FINER [32], [33] adapts activation frequencies, INCODE [34] modulates periodic activations, MFN [35] employs multiplicative filters, and FR [36] uses Fourier reparameterization. Recent multidimensional INRs further incorporate semantic structure [37] or learnable latent frequency spaces [38]. Other recent approaches explicitly consider cross-frequency representation [39] or joint frequency and locality information [40]. Continuous tensor representations have also been developed to exploit structural priors for high-dimensional signal recovery [41]. Our framework instead coordinates heterogeneous MLP/KAN representations through wavelet-domain fusion and branchspecific spectral regularization.

## B. Advances in Kolmogorov-Arnold Networks

Inspired by the Kolmogorov-Arnold representation theorem, KAN [25] replaces the fixed activations used in MLP with learnable univariate functions on network edges. WaveKAN [42] combines adaptive wavelet decomposition with KAN for long-term time-series forecasting, using wavelet decomposition to separate low-frequency trends and high-frequency details before subsequent modeling and fusion. In addition, numerous variants based on KAN exist, such as GKAN [43], RKAN [44], KKAN [45], and FC-KAN [46]. In this work, we adopt the standard splinebased KAN formulation to focus on the complementary representation behavior between heterogeneous MLP and KAN branches. The standard formulation provides a representative and relatively simple KAN architecture for evaluating this heterogeneous dual-branch design.

## C. Wavelet-based Multiresolution Analysis

The DWT decomposes signals into approximation and detail subbands for multiresolution analysis [47], [48]. Wavelet and wavelet-frame representations have been widely studied in image restoration and related inverse problems [49]. Wavelet-based representations have also been incorporated into INRs [30]. Compared with global Fourier bases, wavelets provide localized spatial– frequency analysis, which is beneficial for representing non-stationary structures such as edges, textures, and local variations. Adaptive WPE [50] further incorporates wavelet information into coordinate encoding, whereas our framework applies DWT to branch outputs for coeficient fusion and band-separation regularization. However, applying DWT alone does not automatically lead to frequency specialization between diferent neural branches. Since DWT is a linear transform, directly decomposing branch outputs and recombining the corresponding subbands does not necessarily change the modeling behavior unless additional constraints are introduced. This observation motivates the use of wavelet-domain regularization to explicitly measure and constrain the frequency responses of diferent branches.

![](images/a9ae7ce64a62f847557c403dbe5ebadec1f67c27db532eabccbf27b2573baaac.jpg)  
Fig. 2: Architecture of the proposed method. The input coordinates are processed by parallel MLP and KAN branches that provide complementary frequency-oriented representations. Their outputs are decomposed by DWT, corresponding wavelet coeficients are additively fused, and the fused coeficients are reconstructed by IDWT. A band-separation regularization encourages complementary frequency responses between the two branches.

## III. Method

## A. Architecture Overview

We propose a spatial-frequency-aware INR framework based on an MLP-KAN dual-branch architecture with wavelet-domain fusion and regularization, as shown in Fig. 2.

The proposed framework maps continuous coordinates to signal values through two parallel branches. The MLP branch provides a low-frequency-oriented representation, whereas the KAN branch provides a complementary representation with relatively stronger responses to localized variations and fine-scale details. The outputs of the two branches are decomposed into wavelet coeficients by DWT, and the corresponding coeficients are combined through element-wise addition. The fused coeficients are subsequently reconstructed by IDWT.

Because additive fusion does not impose a hard frequency assignment, both branches can contribute to the reconstructed signal. We therefore introduce a waveletdomain band-separation regularization to encourage complementary frequency behavior by suppressing highfrequency responses in the MLP branch and low-frequency responses in the KAN branch.

## B. Dual-Branch Spectral Decomposition

Given an input coordinate $\textbf { x } \in \mathbb { R } ^ { d _ { \mathrm { i n } } }$ , the proposed framework employs two coordinate networks to parameterize complementary spectral representations. After waveletdomain fusion and IDWT reconstruction, the final prediction is expressed as the sum of a low-frequency component and a high-frequency component:

$$
\hat { \mathbf { I } } = \hat { \mathbf { I } } _ { \mathrm { l o w } } + \hat { \mathbf { I } } _ { \mathrm { h i g h } } ,\tag{1}
$$

where $\hat { \mathbf { I } } _ { \mathrm { l o w } }$ and $\hat { \bf I } _ { \mathrm { h i g h } }$ denote the approximation-band and detail-band components of the fused reconstruction, respectively. Under the band-separation constraint, these components are encouraged to be dominated by the MLP and KAN branches, respectively.

1) MLP Branch for Global Structure: The MLP branch parameterizes the smooth global structure of the signal. Let $\mathbf { h } _ { 0 } = \mathbf { x } .$ . For an L<sub>M</sub>-layer MLP, the hidden features are computed as

$$
\begin{array} { r } { \mathbf { h } _ { l } = \sigma ( \mathbf { W } _ { l } \mathbf { h } _ { l - 1 } + \mathbf { b } _ { l } ) , \quad l = 1 , \ldots , L _ { M } - 1 , } \end{array}\tag{2}
$$

where $\sigma ( \cdot )$ denotes a fixed nonlinear activation function. The output of the MLP branch is defined by

$$
f _ { \boldsymbol { \theta } _ { M } } ( \mathbf { x } ) = \mathbf { W } _ { L _ { M } } \mathbf { h } _ { L _ { M } - 1 } + \mathbf { b } _ { L _ { M } } ,\tag{3}
$$

where $\boldsymbol { \theta } _ { M } = \{ \mathbf { W } _ { l } , \mathbf { b } _ { l } \} _ { l = 1 } ^ { L _ { M } }$ contains the learnable parameters of the MLP branch. The output $f _ { \boldsymbol { \theta } _ { M } } ( \mathbf { x } )$ serves as a low-frequency-oriented representation and is encouraged to capture slowly varying global structures.

2) KAN Branch for Localized Details: The KAN branch parameterizes localized signal variations using learnable univariate functions placed on network edges. Let $\mathbf { z } _ { 0 } = \mathbf { x }$ For the ℓ-th KAN layer, the $q \mathrm { - }$ -th output feature is computed as

$$
z _ { \ell + 1 , q } = \sum _ { p = 1 } ^ { d _ { \ell } } \phi _ { \ell , q , p } ( z _ { \ell , p } ) , \quad q = 1 , \ldots , d _ { \ell + 1 } ,\tag{4}
$$

where $d _ { \ell }$ is the feature dimension and $\phi _ { \ell , q , p } ( \cdot )$ is the learnable function on the edge from input $p$ to output $q .$

In the spline-based implementation, each edge function is parameterized as

$$
\phi _ { \ell , q , p } ( t ) = w _ { \ell , q , p } b ( t ) + \sum _ { r = 1 } ^ { R } c _ { \ell , q , p , r } B _ { r } ( t ) ,\tag{5}
$$

where $b ( \cdot )$ is the base function, $B _ { r } ( \cdot )$ is the r-th B-spline basis, and $w _ { \ell , q , p }$ and $c \ell , q , p , r$ are learnable coeficients. $B _ { r } ( \cdot )$ denotes the r-th B-spline basis function of degree k. The spline grid size controls the number of knot intervals used to construct the basis functions. The degree k and grid size reported in the experiments therefore determine the local spline order and resolution of each KAN edge function, respectively.

After recursively applying Eq. (4) over $L _ { K }$ layers, the output of the KAN branch is defined as

$$
g _ { \theta _ { K } } ( \mathbf { x } ) = \mathbf { z } _ { L _ { K } } ,\tag{6}
$$

where $\theta _ { K }$ collects all learnable parameters of the KAN branch. The output $g _ { \theta _ { K } } ( \mathbf { x } )$ serves as a complementary representation with relatively stronger responses to localized rapid variations, edges, textures, and fine-scale details.

![](images/82694aad9a8ac687b915a9ca77433fa4ddec4fa79ce73af47d444b7f23bd4cb9.jpg)  
Fig. 3: Schematic illustration of wavelet-domain decomposition and reconstruction. For a 2D signal, DWT decomposes the representation into the LL, LH, HL, and HH subbands. Corresponding MLP and KAN wavelet coeficients are additively fused before IDWT reconstruction, while band-separation regularization encourages complementary frequency-oriented responses.

## C. Wavelet-Domain Fusion and Reconstruction

Let $\mathcal { W } _ { \psi } ( \cdot )$ and $\mathcal { W } _ { \psi } ^ { - 1 } ( \cdot )$ denote DWT and IDWT with wavelet basis ψ. We use a J-level 1D Haar (db1) transform along sampled coordinate-axis lines; Eqs. (8)–(13) use a one-level 2D formulation for clarity.

Given a discrete coordinate grid $\mathcal { X } = \{ \mathbf { x } _ { n } \} _ { n = 1 } ^ { N }$ , the two coordinate networks are evaluated to obtain

$$
{ \bf F } _ { M } = f _ { \theta _ { M } } ( { \boldsymbol { \chi } } ) , \qquad { \bf F } _ { K } = g _ { \theta _ { K } } ( { \boldsymbol { \chi } } ) ,\tag{7}
$$

where ${ \bf F } _ { M }$ and ${ \bf F } _ { K }$ denote the signal tensors generated by the MLP and KAN branches, respectively.

The MLP branch output is decomposed as

$$
\left\{ Y _ { L L } ^ { M } , Y _ { L H } ^ { M } , Y _ { H L } ^ { M } , Y _ { H H } ^ { M } \right\} = { \mathcal W } _ { \psi } ( { \bf F } _ { M } ) ,\tag{8}
$$

and the KAN branch output is decomposed as

$$
\left\{ Y _ { L L } ^ { K } , Y _ { L H } ^ { K } , Y _ { H L } ^ { K } , Y _ { H H } ^ { K } \right\} = { \mathcal { W } } _ { \psi } ( \mathbf { F } _ { K } ) .\tag{9}
$$

Here, $Y _ { L L } ^ { ( \cdot ) }$ denotes the low-frequency approximation subband, whereas $Y _ { L H } ^ { ( \cdot ) } , \ Y _ { H L } ^ { ( \cdot ) }$ , and $Y _ { H H } ^ { ( \cdot ) }$ denote the three directional high-frequency detail subbands.

The corresponding coeficients are additively fused as $\widetilde { Y } _ { b } ^ { F } = Y _ { b } ^ { M } + \mathbf { \widetilde { Y } } _ { b } ^ { K } , b \in \{ L L , L H , H L , H H \}$ , allowing both branches to contribute to each band. Band-separation regularization encourages the branch responses toward the following idealized limiting state:

$$
\left\{ \begin{array} { l l } { \widetilde { Y } _ { L L } ^ { F } = Y _ { L L } ^ { M } , } \\ { \widetilde { Y } _ { L H } ^ { F } = Y _ { L H } ^ { K } , } \\ { \widetilde { Y } _ { H L } ^ { F } = Y _ { H L } ^ { K } , } \\ { \widetilde { Y } _ { H H } ^ { F } = Y _ { H H } ^ { K } . } \end{array} \right.\tag{10}
$$

Eq. (10) represents the limiting case in which the highfrequency responses of the MLP branch and the lowfrequency response of the KAN branch are fully suppressed.

The final prediction is reconstructed from the fused subbands through IDWT:

$$
\begin{array} { r l } & { \hat { \mathbf { I } } = \hat { f } _ { \theta _ { M } , \theta _ { K } } ( \pmb { \chi } ) } \\ & { \quad = \mathcal { W } _ { \psi } ^ { - 1 } \left( \widetilde { Y } _ { L L } ^ { F } , \widetilde { Y } _ { L H } ^ { F } , \widetilde { Y } _ { H L } ^ { F } , \widetilde { Y } _ { H H } ^ { F } \right) . } \end{array}\tag{11}
$$

To characterize the approximation and detail components of the actual fused reconstruction, we define:

$$
\begin{array} { r } { \hat { \bf \cal I } _ { \mathrm { l o w } } = { \mathcal W } _ { \psi } ^ { - 1 } \left( \widetilde { Y } _ { L L } ^ { F } , \mathbf { 0 } , \mathbf { 0 } , \mathbf { 0 } \right) , } \end{array}\tag{12}
$$

and

$$
\hat { \bf I } _ { \mathrm { h i g h } } = { \mathcal W } _ { \psi } ^ { - 1 } \left( { \bf 0 } , \widetilde { Y } _ { L H } ^ { F } , \widetilde { Y } _ { H L } ^ { F } , \widetilde { Y } _ { H H } ^ { F } \right) .\tag{13}
$$

Here, 0 is introduced only to isolate the approximation or detail component during inverse reconstruction; it does not imply that the corresponding coeficients of the actual fused representation are zero.

For multidimensional signals, we sample complete coordinate-axis lines and apply a J-level 1D DWT/IDWT along each line rather than constructing a full Ddimensional wavelet tensor. Approximation coeficients are recursively decomposed and all detail coeficients are retained. For axis d, the fused coeficients are $\widetilde { Y } _ { l } ^ { F , d } ~ =$ $Y _ { l } ^ { M , d } + Y _ { l } ^ { K , d }$ and $\widetilde { Y } _ { h , j } ^ { F , d } \ : = \ : Y _ { h , j } ^ { M , d } \ : + \ : Y _ { h , j } ^ { K , d } , \ : j \ : = \ : 1 , \ldots , J .$ The same fusion and band-separation regularization are applied along every sampled axis, including horizontal and vertical lines in 2D and their extensions to higher dimensions.

## D. Objective Function and Spectral Constraints

For direct signal representation, reconstruction fidelity is measured by

$$
\mathcal { L } _ { \mathrm { r e c o n } } = \frac { 1 } { N } \left. \hat { \mathbf { I } } - \mathbf { I } \right. _ { F } ^ { 2 } ,\tag{14}
$$

where <sup>ˆ</sup>I and I denote the reconstructed and ground-truth signals, respectively, N denotes the total number of scalar elements in the signal tensor, and $\| \cdot \| _ { F }$ denotes the Frobenius norm.

Because additive fusion alone does not induce complementary frequency-oriented responses between the two branches, the two branches may retain responses outside their intended frequency tendencies. We therefore introduce a wavelet-domain band-separation loss to encourage the idealized limiting state in Eq. (10) by suppressing the high-frequency responses of the MLP branch and the lowfrequency response of the KAN branch. For consistency with the one-level 2D formulation in Eqs. (8)–(13), let $B _ { H }$ denote the set of high-frequency detail subbands. The band-separation loss is defined as

$$
\mathcal { L } _ { \mathrm { b a n d } } = \alpha \sum _ { b \in \mathcal { B } _ { H } } \frac { 1 } { N _ { b } } \left. Y _ { b } ^ { M } \right. _ { F } ^ { 2 } + ( 1 - \alpha ) \frac { 1 } { N _ { L L } } \left. Y _ { L L } ^ { K } \right. _ { F } ^ { 2 } ,\tag{15}
$$

where $Y _ { b } ^ { M }$ denotes a high-frequency detail subband generated by the MLP branch, $Y _ { L L } ^ { \bar { K } }$ denotes the low-frequency approximation subband generated by the KAN branch, and $N _ { b }$ and $N _ { L L }$ are the corresponding numbers of wavelet coeficients. The parameter $\alpha \in [ 0 , 1 ]$ controls the relative weighting between the aggregated MLP high-frequency penalty and the KAN low-frequency penalty, and $\begin{array} { r } { B _ { H } = } \end{array}$ $\{ L H , H L , H H \}$ for the one-level 2D formulation. For the actual J-level line-wise implementation described above, the same penalty is extended to the corresponding approximation and detail coeficients along each sampled axis. By penalizing the MLP high-frequency responses and the KAN low-frequency response, $\mathcal { L } _ { \mathrm { b a n d } }$ promotes complementary frequency responses between the two branches without imposing hard subband assignment.

The overall objective for direct signal representation is formulated as

$$
\begin{array} { r } { \mathcal { L } = \mathcal { L } _ { \mathrm { r e c o n } } + \lambda _ { \mathrm { b a n d } } \mathcal { L } _ { \mathrm { b a n d } } , } \end{array}\tag{16}
$$

where $\lambda _ { \mathrm { b a n d } } \geq 0$ controls the strength of the spectral constraint. This objective encourages a stronger low-frequency tendency in the MLP branch and relatively stronger detail responses in the KAN branch while preserving additive coeficient fusion.

## E. Inverse Problem Formulation

The proposed representation can also be incorporated into inverse problems. Let A denote a known forward operator that maps a latent signal I to the observation domain. The measurement model is expressed as

$$
\mathbf { y } = \mathcal { A } ( \mathbf { I } ) + \epsilon ,\tag{17}
$$

where y denotes the observed data and ϵ represents measurement noise.

Given the reconstructed signal <sup>ˆ</sup>I produced by the proposed MLP-KAN representation, the corresponding datafidelity loss is defined as

$$
\mathcal { L } _ { \mathrm { d a t a } } = \frac { 1 } { N _ { y } } \left. \mathbf { \nabla } A ( \hat { \mathbf { I } } ) - \mathbf { y } \right. _ { 2 } ^ { 2 } ,\tag{18}
$$

where $N _ { y }$ is the number of observed measurements. Direct signal representation is a special case of Eq. (18), in which A is the identity operator and $\mathbf y = \mathbf I$

For inverse reconstruction, the network parameters are estimated by

$$
\begin{array} { r } { \left( \theta _ { M } ^ { * } , \theta _ { K } ^ { * } \right) = \underset { \theta _ { M } , \theta _ { K } } { \arg \operatorname* { m i n } } \left[ \mathcal { L } _ { \mathrm { d a t a } } + \lambda _ { \mathrm { b a n d } } \mathcal { L } _ { \mathrm { b a n d } } + \lambda _ { \mathrm { T V } } \mathcal { R } _ { \mathrm { T V } } ( \hat { \bf I } ) \right] , } \end{array}\tag{19}
$$

where $\lambda _ { \mathrm { b a n d } } ~ \geq ~ 0$ and $\lambda _ { \mathrm { T V } } ~ \ge ~ 0$ control the spectral constraint and spatial regularization, respectively. The reconstructed latent signal is subsequently obtained by evaluating the proposed representation using the optimized network parameters:

$$
\hat { \mathbf { I } } ^ { * } = \hat { f } _ { \boldsymbol { \theta } _ { M } ^ { * } , \boldsymbol { \theta } _ { K } ^ { * } } ( \mathcal { X } ) .\tag{20}
$$

For regularly sampled multidimensional signals, the anisotropic total variation regularizer is defined as

$$
\mathcal { R } _ { \mathrm { T V } } ( { \hat { \mathbf { I } } } ) = \sum _ { d = 1 } ^ { D } \left\| \mathbf { D } _ { d } { \hat { \mathbf { I } } } \right\| _ { 1 } ,\tag{21}
$$

where D is the signal dimensionality and $\mathbf { D } _ { d }$ denotes the discrete diference operator along the d-th dimension. The

$\ell _ { 1 } .$ -norm promotes sparsity of the signal gradients, thereby suppressing noise while preserving sharp structural tran sitions.

## IV. Experiments

## A. Experimental Setup

We evaluate per-instance fitting on 1D square waves and audio, 2D images, 3D volumes and SDFs, videos, and 4D LFs. These tasks encompass piecewise-constant signals with abrupt transitions, oscillatory temporal signals, natural images with diverse texture distributions, volumetric intensity fields, implicit geometric surfaces, dynamic visual content, and spatial–angular LF data. The 1D experiments use the original KAN codebase, whereas the 2D–4D experiments use custom CUDA KAN layers integrated with PyTorch 2.0.0 and CUDA 11.3 on a single NVIDIA RTX 4090 GPU. All methods are optimized with AdamW. The 1D, 2D, and 4D tasks run for 5,000 iterations, while the 3D volume, SDF, and video tasks run for 20,000 iterations. Within each task, all methods use identical data, preprocessing, sampling, iteration budgets, and evaluation protocols, with method-specific settings following their oficial implementations.For the full model, $\lambda _ { \mathrm { b a n d } } = 1 0 ^ { - 3 }$ and $\alpha = 0 . 5 ; \lambda _ { \mathrm { b a n d } } = 0$ only in no-band ablations. Network parameters are optimized separately for each target signal, with audio fitted segment by segment as described in Section IV-C.

## B. Parameterization of 1D Square-Wave Signals

1) Datasets: Training uses 128 uniformly sampled square-wave points, while evaluation uses 256 points over the same domain to test reconstruction at additional coordinates not observed during optimization.

2) Experimental Details: All methods are optimized for 5,000 iterations with $\lambda _ { \mathrm { b a n d } } = 1 0 ^ { - 3 } , \alpha = 0 . 5 , J = 3 ,$ and the Haar wavelet.

3) Experimental Results: Figs. 4 and 5 present the square-wave reconstruction results on the original and denser coordinate grids. Square-wave signals contain piecewise-constant regions and abrupt transitions, providing a simple test of both smooth-region fitting and sharptransition reconstruction. Several comparison methods exhibit transition blurring, oscillatory artifacts, or ampli tude deviations near discontinuities, whereas the proposed method better preserves piecewise-constant regions and sharp transitions. On the denser coordinate grid, which contains additional coordinates not used during optimization, the proposed method also maintains more stable signal values and sharper transition profiles.

![](images/b4ca560bd55ba6b0778bd944e75c8db1e659b68dc3b001753fa3731c652e6690.jpg)  
Fig. 4: Qualitative comparison of the 1D square-wave reconstruction results obtained by nine representation methods. The reference signal and the predictions of all methods are evaluated on the same coordinate grid.

![](images/2333c56dcd668aa8a41262069d1191b0ec3ad2efe5f0a7aea5554358ce9516ce.jpg)  
Fig. 5: Dense-grid evaluation of the square-wave representation results. All methods were optimized using 128 uniformly sampled coordinates and subsequently evaluated at 256 uniformly sampled coordinates over the same signal domain.

## C. Parameterization of 1D Audio Signals

1) Datasets: For the 1D audio-fitting experiments, we used the same target audio clips as those used in SIREN [22], namely “bach” and “counting.” Their durations are 6.99 s and 12 s, respectively. Both clips are stored in WAV format at a sampling rate of 44.1 kHz. Each audio waveform was represented as a mapping from a normalized temporal coordinate to the corresponding signal amplitude.

2) Experimental Details: The audio experiments followed the optimization settings used in the 1D squarewave experiments. To reduce the dificulty of fitting long audio sequences and enable a consistent comparison among diferent representation methods, each audio waveform was divided into nonoverlapping 0.5-s segments. Each segment was independently parameterized using its corresponding temporal coordinates and amplitude values. After optimization, the reconstructed segments were concatenated in chronological order to obtain the complete audio waveform. All comparison methods used the same segmentation strategy and experimental settings.

3) Experimental Results: Figs. 6 and 7 compare the audio fitting performance of the proposed method with that of eight representative coordinate-based methods. PSNR was used to quantify the sample-wise discrepancy between the reconstructed and reference audio waveforms. The two audio clips contain both slowly varying waveform structures and rapid local oscillations, thereby requiring the representation model to characterize signal components over diferent frequency ranges.

As shown in Figs. 6 and 7, several comparison methods produce noticeable reconstruction deviations in waveform regions containing rapid amplitude variations. In contrast, the proposed method more closely follows the reference waveform in both slowly varying and rapidly oscillating regions. The proposed method also achieves the highest PSNR among the compared methods on both audio clips, indicating improved sample-level reconstruction fidelity for 1D audio signals.

![](images/b959925155cc37961c7adbe787d51260c180a5619debdefb3645e20588462c8c.jpg)  
Fig. 6: Comparison of the 1D audio fitting results obtained by nine representation methods on the “bach” audio clip.

![](images/ff6bd6a17654087dbf316b5172b437ff27839691d3858d54fc5020f4691a6580.jpg)  
Fig. 7: Comparison of the 1D audio fitting results obtained by nine representation methods on the “counting” audio clip.

## D. Parameterization of 2D Images

1) Datasets: We use the first 700 images from the DIV2K Train Data (HR images) subset [51] for the 2D image parameterization experiments. All selected images are uniformly resized to 512 × 512 before training and evaluation. The selected images contain diverse scenes and exhibit diferent levels of structural and textural complexity. DIV2K is widely used for evaluating image reconstruction and super-resolution methods.

2) Experimental Details: All images are resized to 512×   
512 and evaluated using PSNR, SSIM, and LPIPS. The

MLP uses three hidden layers of width 64, while the KAN widths are 128-256-128 with k = 3, grid size 64, and J = 3. This accuracy–eficiency configuration is used in Table VII and subsequent multidimensional experiments. Additional network settings are evaluated in the hyperparameter sensitivity analysis in Section V-A.

3) Experimental Results: Fig. 8 shows that the proposed method better preserves fine textures and structural boundaries across images with diferent frequency characteristics. Table I further shows the best average PSNR, SSIM, and LPIPS over 700 images; its PSNR of 45.72 dB exceeds the second-best FINER by 6.36 dB.

## E. Parameterization of 3D Volumes, SDFs, and Videos

1) Datasets: For volumetric fitting, we use the Head [52] and Standard volumes [53]. SDF experiments use a synthetic Sphere and the Armadillo and Dragon models from the Stanford 3D Scanning Repository<sup>1</sup>, covering smooth and geometrically complex surfaces. Video experiments use the BALL, COLORS, and PARAL-LAX sequences from the MSU Video Compression Super-

![](images/4712c2219b2fc7be4f426690550aed7442b0c17dd33799a37cfa4e794008ed2c.jpg)  
(a)  
(b)  
(c)  
(d)  
(f)  
(e)  
Fig. 8: Qualitative comparison of nine image representation methods on six representative DIV2K images with diferent levels of texture complexity. Subfigures (a) and (b) correspond to images with complex and dense textures, (c) and (d) correspond to images with relatively sparse textures, and (e) and (f) correspond to images containing both textured and smooth regions.

Resolution Benchmark [54], with each sequence represented as a mapping from (x, y, t) to RGB.

2) Experimental Details: The Head and Standard volumes have resolutions of 256×256×256 and $1 8 1 \times 2 1 7 \times 1 8 1$ respectively. Video frames are resized to 480×270, with the first 100 frames used for fitting and evaluation. Volumetric,

![](images/49b1726c32e57a45c236fe43c3535dea07f504857ad1e4c7ac871f5714a42dbf.jpg)  
Fig. 9: Qualitative and quantitative comparison of volumetric reconstruction results on the Head and Standard datasets. Orthogonal cross-sections along the three spatial dimensions are shown, with the corresponding PSNR and SSIM values annotated for each method.

TABLE I: Image fitting performance on the DIV2K subset at a resolution of $5 1 2 \times 5 1 2$ . Results are reported as mean ± standard deviation.
<table><tr><td>Method</td><td>PSNR (dB) ↑</td><td>SSIM ↑</td><td>LPIPS ↓</td></tr><tr><td>FINER</td><td> $3 9 . 3 6 \pm 3 . 8 3 7$ </td><td> $0 . 9 7 2 6 \pm 0 . 0 1 5 1$ </td><td> $0 . 0 6 1 4 \pm 0 . 0 3 3 8$ </td></tr><tr><td>MFN</td><td> $3 1 . 2 4 \pm 4 . 3 3 4$ </td><td> $0 . 8 8 8 7 \pm 0 . 0 1 1 0$ </td><td> $0 . 2 2 7 4 \pm 0 . 0 1 2 1$ </td></tr><tr><td>FR</td><td> $3 4 . 7 2 \pm 3 . 0 1 0$ </td><td> $0 . 9 3 1 6 \pm 0 . 0 3 8 9$ </td><td> $0 . 1 3 0 8 \pm 0 . 0 6 2 5$ </td></tr><tr><td>MLP-PE</td><td> $2 9 . 9 6 \pm 3 . 4 5 8$ </td><td> $0 . 8 8 1 9 \pm 0 . 0 4 3 4$ </td><td> $0 . 2 3 7 4 \pm 0 . 0 4 4 4$ </td></tr><tr><td>SIREN</td><td> $3 7 . 4 8 \pm 3 . 2 9 1$ </td><td> $0 . 9 7 8 1 \pm 0 . 0 0 9 9$ </td><td> $0 . 0 5 2 7 \pm 0 . 0 2 5 3$ </td></tr><tr><td>WIRE</td><td> $3 3 . 1 4 \pm 2 . 8 5 8$ </td><td> $0 . 8 7 4 1 \pm 0 . 0 4 8 1$ </td><td> $0 . 2 2 8 9 \pm 0 . 0 6 8 6$ </td></tr><tr><td>KAN</td><td> $2 6 . 8 9 \pm 2 . 7 8 2$ </td><td> $0 . 8 1 0 6 \pm 0 . 0 7 3 2$ </td><td> $0 . 3 2 1 0 \pm 0 . 0 6 9 4$ </td></tr><tr><td>BACON</td><td> $3 0 . 8 1 \pm 2 . 5 7 1$ </td><td> $0 . 8 9 2 0 \pm 0 . 0 3 1 9$ </td><td> $0 . 2 0 4 2 \pm 0 . 0 5 9 0$ </td></tr><tr><td>Ours</td><td> $\mathbf { 4 5 . 7 2 \pm 2 . 3 4 9 }$ </td><td> $\mathbf { 0 . 9 9 1 2 \pm 0 . 0 0 3 0 }$ </td><td> $\mathbf { 0 . 0 0 3 0 \pm 0 . 0 0 2 0 }$ </td></tr></table>

SDF, and video experiments follow the configuration in Section IV-D and are optimized for 20,000 iterations. Volumes and videos are evaluated using the corresponding PSNR/SSIM/LPIPS metrics, while SDF reconstruction is evaluated using Chamfer distance and IoU.

## 3) Experimental Results:

a) Volumetric representation: Fig. 9 compares the reconstructed cross-sections of the Head and Standard volumes. The competing methods exhibit diferent degrees of smoothing and structural deviation near tissue boundaries and small anatomical regions. The proposed method preserves the principal anatomical structures while producing clearer boundaries and fewer local reconstruction artifacts. These results demonstrate that the proposed representation can model both slowly varying volumetric

![](images/db22190507cec881b109fb8467af3d70352f29cf0692e2753871efb503d24a06.jpg)  
Fig. 10: Qualitative comparison of the reconstructed surfaces obtained by diferent representation methods on Sphere, Armadillo, and Dragon.

TABLE II: Quantitative comparison on the Sphere, Armadillo, and Dragon SDF datasets. Results are reported as mean ± standard deviation over three runs. The best result for each metric is shown in bold.
<table><tr><td rowspan="2">Method</td><td colspan="2"> ${ \mathrm { S p h e r e } }$ </td><td colspan="2">Armadillo</td><td colspan="2">Dragon</td></tr><tr><td>Chamfer ↓</td><td>IoU ↑</td><td>Chamfer ↓</td><td>IoU ↑</td><td>Chamfer ↓</td><td>IoU ↑</td></tr><tr><td>FINER</td><td> $1 . 7 2 1 \times 1 0 ^ { - 4 } \pm 2 . 5 2 \times 1 0 ^ { - 7 }$ </td><td> $9 . 9 6 4 \times 1 0 ^ { - 1 } \pm 3 . 6 1 \times 1 0 ^ { - 4 }$ </td><td> $9 . 6 3 0 \times 1 0 ^ { - 6 } \pm 4 . 8 6 \times 1 0 ^ { - 7 }$ </td><td> $9 . 7 8 9 \times 1 0 ^ { - 1 } \pm 1 . 1 9 \times 1 0 ^ { - 3 }$ </td><td> $7 . 9 4 1 \times 1 0 ^ { - 6 } \pm 5 . 6 6 \times 1 0 ^ { - 7 }$ </td><td> $9 . 5 8 3 \times 1 0 ^ { - 1 } \pm 7 . 0 2 \times 1 0 ^ { - 4 }$ </td></tr><tr><td>MFN</td><td> $1 . 7 3 5 \times 1 0 ^ { - 4 } \pm 7 . 3 7 \times 1 0 ^ { - 7 }$ </td><td> $9 . 9 6 7 \times 1 0 ^ { - 1 } \pm 3 . 0 6 \times 1 0 ^ { - 4 }$ </td><td> $8 . 1 9 2 \times 1 0 ^ { - 6 } \pm 3 . 0 0 \times 1 0 ^ { - 8 }$ </td><td> $9 . 9 2 4 \times 1 0 ^ { - 1 } \pm 3 . 7 9 \times 1 0 ^ { - 4 }$ </td><td> $6 . 5 5 6 \times 1 0 ^ { - 6 } \pm 2 . 7 6 \times 1 0 ^ { - 8 }$ </td><td> $9 . 7 5 8 \times 1 0 ^ { - 1 } \pm 6 . 0 0 \times 1 0 ^ { - 4 }$ </td></tr><tr><td>FR</td><td> $1 . 7 3 0 \times 1 0 ^ { - 4 } \pm 3 . 5 1 \times 1 0 ^ { - 7 }$ </td><td> $9 . 9 7 0 \times 1 0 ^ { - 1 } \pm 3 . 5 1 \times 1 0 ^ { - 4 }$ </td><td> $9 . 3 3 3 \times 1 0 ^ { - 6 } \pm 1 . 3 5 \times 1 0 ^ { - 7 }$ </td><td> $9 . 8 1 4 \times 1 0 ^ { - 1 } \pm 3 . 0 6 \times 1 0 ^ { - 4 }$ </td><td> $8 . 0 7 6 \times 1 0 ^ { - 6 } \pm 3 . 5 5 \times 1 0 ^ { - 7 }$ </td><td> $9 . 5 7 0 \times 1 0 ^ { - 1 } \pm 3 . 5 5 \times 1 0 ^ { - 3 }$ </td></tr><tr><td>MLP-PE</td><td> $1 . 7 3 9 \times 1 0 ^ { - 4 } \pm 2 . 5 2 \times 1 0 ^ { - 7 }$ </td><td> $9 . 9 3 1 \times 1 0 ^ { - 1 } \pm 6 . 0 0 \times 1 0 ^ { - 4 }$ </td><td> $4 . 6 9 7 \times 1 0 ^ { - 5 } \pm 6 . 4 6 \times 1 0 ^ { - 5 }$ </td><td>9.514 × 10−1  $\pm 5 . 2 0 \times 1 0 ^ { - 2 }$ </td><td> $7 . 4 0 6 \times 1 0 ^ { - 6 } \pm 2 . 5 0 \times 1 0 ^ { - 7 }$ </td><td> $9 . 6 2 8 \times 1 0 ^ { - 1 } \pm 2 . 0 3 \times 1 0 ^ { - 3 }$ </td></tr><tr><td>SIREN</td><td> $1 . 7 2 2 \times 1 0 ^ { - 4 } \pm 5 . 6 9 \times 1 0 ^ { - 7 }$ </td><td> $9 . 9 5 9 \times 1 0 ^ { - 1 } \pm 7 . 5 7 \times 1 0 ^ { - 4 }$ </td><td> $1 . 0 3 2 \times { 1 0 ^ { - 5 } } \pm 4 . 8 2 \times { 1 0 ^ { - 7 } }$ </td><td> $9 . 7 6 1 \times 1 0 ^ { - 1 } \pm 4 . 9 3 \times 1 0 ^ { - 4 }$ </td><td> $9 . 4 3 8 \times 1 0 ^ { - 6 } \pm 1 . 0 8 \times 1 0 ^ { - 6 }$ </td><td> $9 . 4 8 0 \times 1 0 ^ { - 1 } \pm 8 . 0 5 \times 1 0 ^ { - 3 }$ </td></tr><tr><td>WIRE</td><td> $2 . 2 3 0 \times 1 0 ^ { - 4 } \pm 5 . 5 7 \times 1 0 ^ { - 7 }$ </td><td> $9 . 7 4 2 \times 1 0 ^ { - 1 } \pm 4 . 5 8 \times 1 0 ^ { - 4 }$ </td><td> $9 . 5 5 5 \times 1 0 ^ { - 6 } \pm 1 . 1 0 \times 1 0 ^ { - 8 }$ </td><td> $9 . 9 2 0 \times 1 0 ^ { - 1 } \pm 6 . 0 3 \times 1 0 ^ { - 4 }$ </td><td> $1 . 5 1 5 \times 1 0 ^ { - 5 } \pm 9 . 2 9 \times 1 0 ^ { - 8 }$ </td><td> $9 . 6 2 4 \times 1 0 ^ { - 1 } \pm 1 . 5 4 \times 1 0 ^ { - 3 }$ </td></tr><tr><td>KAN</td><td> $1 . 7 2 6 \times 1 0 ^ { - 4 } \pm 6 . 0 8 \times 1 0 ^ { - 7 }$ </td><td> $9 . 9 7 5 \times 1 0 ^ { - 1 } \pm 4 . 0 0 \times 1 0 ^ { - 4 }$ </td><td> $1 . 6 7 0 \times 1 0 ^ { - 5 } \pm 6 . 7 1 \times 1 0 ^ { - 7 }$ </td><td>9.617 × 10−1 ± 6.03 × 10−4</td><td> $3 . 3 2 8 \times 1 0 ^ { - 5 } \pm 5 . 8 7 \times 1 0 ^ { - 6 }$ </td><td> $8 . 7 3 7 \times 1 0 ^ { - 1 } \pm 1 . 2 5 \times 1 0 ^ { - 2 }$ </td></tr><tr><td>BACON</td><td> $1 . 7 4 1 \times 1 0 ^ { - 4 } \pm 3 . 5 1 \times 1 0 ^ { - 7 }$ </td><td> $9 . 9 5 3 \times 1 0 ^ { - 1 } \pm 4 . 7 3 \times 1 0 ^ { - 4 }$ </td><td> $8 . 2 0 5 \times 1 0 ^ { - 6 } \pm 1 . 7 1 \times 1 0 ^ { - 8 }$ </td><td> $9 . 9 1 3 \times 1 0 ^ { - 1 } \pm 3 . 2 1 \times 1 0 ^ { - 4 }$ </td><td> $6 . 5 6 4 \times 1 0 ^ { - 6 } \pm 6 . 0 2 \times 1 0 ^ { - 8 }$ </td><td> $9 . 7 4 9 \times 1 0 ^ { - 1 } \pm 5 . 1 3 \times 1 0 ^ { - 4 }$ </td></tr><tr><td>Ours</td><td> $\mathbf { 1 . 7 1 2 \times 1 0 ^ { - 4 } } \pm { \bf 6 . 5 1 \times 1 0 ^ { - 7 } }$ </td><td> $\mathbf { 9 . 9 8 7 \times 1 0 ^ { - 1 } } \pm \mathbf { 3 . 2 1 } \times \mathbf { 1 0 ^ { - 4 } }$ </td><td> $\mathbf { 8 . 0 6 4 \times 1 0 ^ { - 6 } \pm 1 . 0 6 \times 1 0 ^ { - 8 } }$ </td><td> $\mathbf { 9 . 9 4 3 \times 1 0 ^ { - 1 } } \pm \mathbf { 2 . 0 7 } \times \mathbf { 1 0 ^ { - 3 } }$ </td><td> $\mathbf { 6 . 3 0 9 \times 1 0 ^ { - 6 } \pm 4 . 7 1 \times 1 0 ^ { - 8 } }$ </td><td> $\mathbf { 9 . 7 8 2 \times 1 0 ^ { - 1 } \pm 1 . 8 3 \times 1 0 ^ { - 3 } }$ </td></tr></table>

TABLE III: Quantitative comparison on the BALL, COLORS, and PARALLAX video sequences. Results are reported as mean ± standard deviation over the evaluated frames. The best result for each metric is shown in bold.
<table><tr><td rowspan="2">Method</td><td colspan="3">BALL</td><td colspan="3">COLORS</td><td colspan="3">PARALLAX</td></tr><tr><td>PSNR (dB) ↑</td><td>SSIM ↑</td><td>LPIPS ↓</td><td> $\mathrm { P S N R } \ ( \mathrm { d B } ) \uparrow$ </td><td>SSIM ↑</td><td>LPIPS ↓</td><td> $\mathrm { P S N R } \ ( \mathrm { d B } ) \uparrow$ </td><td>SSIM ↑</td><td>LPIPS ↓</td></tr><tr><td>BACON</td><td> $2 7 . 5 7 \pm 0 . 9 1$ </td><td> $0 . 6 8 1 1 \pm 0 . 0 1 8 1$ </td><td> $0 . 4 8 2 7 \pm 0 . 0 1 6 2$ </td><td> $2 2 . 7 1 \pm 0 . 5 9$ </td><td> $0 . 5 0 6 6 \pm 0 . 0 0 8 3$ </td><td> $0 . 5 8 8 8 \pm 0 . 0 0 4 9$ </td><td> $1 9 . 4 7 \pm 0 . 0 8$ </td><td> $0 . 3 4 4 5 \pm 0 . 0 3 9 0$ </td><td> $0 . 6 6 0 1 \pm 0 . 0 1 4 0$ </td></tr><tr><td>FINER</td><td> $3 6 . 9 9 \pm 0 . 4 9$ </td><td> $0 . 9 5 0 8 \pm 0 . 0 3 1 5$ </td><td> $0 . 1 8 4 7 \pm 0 . 0 1 0 8$ </td><td> $3 0 . 3 4 \pm 0 . 1 4$ </td><td> $0 . 9 1 3 9 \pm 0 . 0 1 7 4$ </td><td>0.2425 ± 0.0226</td><td> $2 7 . 0 1 \pm 0 . 2 7$ </td><td> $0 . 7 8 5 3 \pm 0 . 0 4 0 9$ </td><td> $0 . 3 5 3 6 \pm 0 . 0 1 6 9$ </td></tr><tr><td>FR</td><td> $2 9 . 5 2 \pm 0 . 2 3$ </td><td> $0 . 8 4 4 4 \pm 0 . 0 1 3 0$ </td><td> $0 . 4 3 2 0 \pm 0 . 0 2 1 8$ </td><td> $2 5 . 9 3 \pm 0 . 6 2$ </td><td> $0 . 8 1 9 1 \pm 0 . 0 0 3 0$ </td><td> $0 . 3 7 9 9 \pm 0 . 0 2 9 0$ </td><td> $2 6 . 5 3 \pm 0 . 2 8$ </td><td> $0 . 8 0 7 2 \pm 0 . 0 3 3 6$ </td><td> $0 . 3 1 2 4 \pm 0 . 0 1 8 8$ </td></tr><tr><td>KAN</td><td> $3 2 . 8 9 \pm 0 . 3 4$ </td><td> $0 . 9 3 9 0 \pm 0 . 0 2 0 3$ </td><td> $0 . 1 6 7 0 \pm 0 . 0 2 2 0$ </td><td> $2 7 . 0 8 \pm 0 . 5 0$ </td><td> $0 . 9 0 2 5 \pm 0 . 0 0 8 5$ </td><td> $0 . 2 1 5 8 \pm 0 . 0 1 1 3$ </td><td> $2 2 . 3 1 \pm 0 . 1 6$ </td><td> $0 . 7 1 6 7 \pm 0 . 0 3 1 4$ </td><td> $0 . 4 4 1 3 \pm 0 . 0 1 0 4$ </td></tr><tr><td>MFN</td><td> $2 8 . 0 8 \pm 0 . 9 5$ </td><td> $0 . 7 9 5 0 \pm 0 . 0 0 8 2$ </td><td>0.4168 ± 0.0100</td><td> $2 4 . 6 0 \pm 0 . 1 9$ </td><td> $0 . 7 2 9 1 \pm 0 . 0 2 3 8$ </td><td>0.4671 ± 0.0146</td><td> $2 3 . 2 6 \pm 0 . 3 6$ </td><td> $0 . 6 4 7 5 \pm 0 . 0 1 0 2$ </td><td> $0 . 4 6 9 7 \pm 0 . 0 0 6 1$ </td></tr><tr><td>MLP-PE</td><td> $3 8 . 8 9 \pm 0 . 8 6$ </td><td> $0 . 9 7 8 1 \pm 0 . 0 2 8 2$ </td><td> $0 . 1 0 5 7 \pm 0 . 0 0 5 0$ </td><td> $3 1 . 1 1 \pm 0 . 2 4$ </td><td> $0 . 9 4 6 3 \pm 0 . 0 4 4 8$ </td><td> $0 . 1 6 5 7 \pm 0 . 0 1 9 6$ </td><td> $2 3 . 6 3 \pm 0 . 2 6$ </td><td> $0 . 7 0 7 8 \pm 0 . 0 1 2 2$ </td><td> $0 . 4 5 6 7 \pm 0 . 0 1 7 4$ </td></tr><tr><td>SIREN</td><td> $3 5 . 0 4 \pm 0 . 3 5$ </td><td> $0 . 9 5 4 6 \pm 0 . 0 1 1 1$ </td><td> $0 . 1 3 3 3 \pm 0 . 0 2 8 0$ </td><td> $2 9 . 1 3 \pm 0 . 2 5$ </td><td> $0 . 9 3 7 4 \pm 0 . 0 2 1 6$ </td><td> $0 . 1 3 7 5 \pm 0 . 0 0 6 3$ </td><td> $2 6 . 1 4 \pm 0 . 4 7$ </td><td> $\mathbf { 0 . 8 5 4 5 \pm 0 . 0 3 0 8 }$ </td><td> $0 . 2 6 6 5 \pm 0 . 0 1 3 0$ </td></tr><tr><td>WIRE</td><td> $2 9 . 0 6 \pm 0 . 4 0$ </td><td> $0 . 8 3 1 8 \pm 0 . 0 4 4 7$ </td><td> $0 . 4 1 2 6 \pm 0 . 0 2 2 5$ </td><td> $2 4 . 2 6 \pm 0 . 2 2$ </td><td> $0 . 7 5 9 8 \pm 0 . 0 4 6 0$ </td><td>0.4539 ± 0.0167</td><td> $2 3 . 3 6 \pm 0 . 1 0$ </td><td> $0 . 6 7 4 0 \pm 0 . 0 3 7 5$ </td><td> $0 . 4 7 1 0 \pm 0 . 0 2 3 2$ </td></tr><tr><td>Ours</td><td> ${ \bf 4 3 . 7 7 \pm 0 . 5 3 }$ </td><td> $\mathbf { 0 . 9 8 8 3 \pm 0 . 0 0 1 8 }$ </td><td> $\mathbf { 0 . 0 0 6 9 \pm 0 . 0 0 0 9 }$ </td><td> $\mathbf { 3 2 . 7 7 \pm 0 . 8 7 }$ </td><td> $\mathbf { 0 . 9 5 2 3 \pm 0 . 0 1 4 6 }$ </td><td> $\mathbf { 0 . 0 3 6 9 \pm 0 . 0 0 4 4 }$ </td><td> ${ \bf 2 7 . 4 2 \pm 0 . 1 0 }$ </td><td> $0 . 7 7 4 4 \pm 0 . 0 1 0 7$ </td><td> $\mathbf { 0 . 2 2 1 1 \pm 0 . 0 0 9 1 }$ </td></tr></table>

b) SDF representation: Fig. 10 and Table II show that the proposed method achieves the lowest mean Chamfer distance and highest mean IoU on all three shapes, indicating consistent reconstruction of both smooth and geometrically complex surfaces.

![](images/e200ce254fbe95bcfb64747ac23f3ec53474b27b2e00fcd52c6df9053e16f9ea.jpg)  
Fig. 11: Qualitative comparison of video reconstruction results on the BALL, COLORS, and PARALLAX sequences.

c) Video representation: Fig. 11 and Table III show that the proposed method achieves the best result in eight of the nine reported metrics, including the highest PSNR and lowest LPIPS on all three sequences. The only exception is SSIM on PARALLAX, where SIREN achieves 0.8545 compared with 0.7744 for the proposed method.

## F. Parameterization of 4D Light Fields

1) Datasets: Four scenes from the HCI 4D LF Benchmark [55] are used. Each scene contains 9 × 9 regularly arranged subaperture images, and all 81 views are used for fitting and evaluation.

2) Experimental Details: Each LF is represented as $( x , y , u , v ) \mapsto R G B$ . During optimization, spatial patches and their corresponding 4D coordinates are randomly sampled across angular views. All methods use identical views, sampling, and optimization conditions, follow the settings in Section IV-A, and are trained for 5,000 iterations. PSNR, SSIM, and LPIPS are averaged over all reconstructed subaperture views.

3) Experimental Results: Fig. 12 shows that the proposed method better preserves scene structures and local textures in the representative subaperture views. Table IV shows that the proposed method achieves the highest

BACON

Ours

MFN

FR  
MLP\_PE  
SIREN  
WIRE  
KAN  
![](images/66e06d929524f526b6ed3ed8373b36b61b0ef036f416d61eb396fc880428a4cc.jpg)  
Fig. 12: Qualitative comparison of the 4D LF reconstruction results obtained by diferent INR-based methods on four representative HCI scenes. Representative subaperture views and enlarged local regions are shown.

TABLE IV: Quantitative comparison of diferent implicit neural representation methods on four HCI LF scenes. Results are reported as mean ± standard deviation. The best result for each metric is shown in bold.
<table><tr><td>Method</td><td>PSNR (dB) ↑</td><td>SSIM ↑</td><td>LPIPS ↓</td></tr><tr><td>BACON</td><td> $2 5 . 5 1 \pm 1 . 1 3 0$ </td><td> $0 . 6 5 6 8 \pm 0 . 0 6 9 4$ </td><td> $0 . 5 7 2 1 \pm 0 . 0 8 2 1$ </td></tr><tr><td>FINER</td><td> $2 4 . 4 2 \pm 1 . 8 7 2$ </td><td> $0 . 5 7 9 6 \pm 0 . 0 9 5 6$ </td><td> $0 . 4 6 7 2 \pm 0 . 0 8 0 9$ </td></tr><tr><td>FR</td><td> $2 7 . 5 7 \pm 1 . 1 6 0$ </td><td> $0 . 7 2 4 6 \pm 0 . 0 7 5 3$ </td><td> $0 . 4 3 8 2 \pm 0 . 1 0 5 6$ </td></tr><tr><td>KAN</td><td> $2 7 . 9 8 \pm 1 . 2 5 9$ </td><td> $0 . 7 3 8 2 \pm 0 . 0 6 7 3$ </td><td> $0 . 3 6 4 4 \pm 0 . 1 0 3 4$ </td></tr><tr><td>MFN</td><td> $2 5 . 6 8 \pm 1 . 0 6 5$ </td><td> $0 . 6 5 5 9 \pm 0 . 0 7 4 1$ </td><td> $0 . 5 3 4 4 \pm 0 . 0 9 7 6$ </td></tr><tr><td>MLP-PE</td><td> $2 8 . 8 9 \pm 1 . 5 9 0$ </td><td> $0 . 7 5 6 8 \pm 0 . 0 7 3 7$ </td><td> $\mathbf { 0 . 2 8 0 3 \pm 0 . 0 9 4 0 }$ </td></tr><tr><td>SIREN</td><td> $2 8 . 6 0 \pm 1 . 2 1 8$ </td><td> $0 . 7 5 7 6 \pm 0 . 0 7 1 4$ </td><td> $0 . 3 8 0 5 \pm 0 . 0 1 0 7$ </td></tr><tr><td>WIRE</td><td> $2 3 . 4 3 \pm 1 . 2 7 1$ </td><td> $0 . 5 3 5 3 \pm 0 . 0 4 4 6$ </td><td> $0 . 6 1 1 2 \pm 0 . 0 2 2 4$ </td></tr><tr><td>Ours</td><td> $\mathbf { 2 9 . 0 6 \pm 2 . 5 3 3 }$ </td><td> $\mathbf { 0 . 8 1 0 2 \pm 0 . 0 6 9 7 }$ </td><td> $0 . 3 6 0 7 \pm 0 . 0 7 9 0$ </td></tr></table>

average PSNR and SSIM over the four scenes, while MLP-PE achieves the lowest LPIPS. The proposed method improves the average PSNR by 0.17 dB over MLP-PE, which achieves the second-highest PSNR.

## G. Computational Eficiency Analysis

Table V compares parameter count, FLOPs, convergence time, and peak GPU memory using a batch size of 2048. Convergence time denotes the elapsed optimization time required to reach 20 dB for the 1D and 2D tasks,

24 dB for the 3D voxel task, and 18 dB for the 4D LF task. Because each INR is optimized independently for a target signal, we emphasize optimization convergence time rather than inference time. The proposed framework generally uses more parameters than the single-branch baselines and introduces additional computational cost in the higherdimensional settings, reflecting an accuracy–computation trade-of. The standalone KAN baseline uses the original PyTorch implementation, whereas the proposed 2D–4D model uses custom CUDA KAN layers. Therefore, the reported convergence times involving KAN should be interpreted as implementation-dependent results.

## V. Ablation Experiments

## A. Ablation on Network Hyperparameters

We evaluate the sensitivity to KAN grid size, B-spline degree, KAN architecture, and MLP architecture on the first 100 DIV2K images at $5 1 2 \times 5 1 2$ . Each block in Table VI is conducted as an independent hyperparameter sweep. Within each block, only the factor under study is varied while the remaining settings are fixed. The nominal settings are MLP $3 \times 2 5 6$ , KAN $3 \times 1 2 8$ , k = 3, and grid size 64. These nominal settings are used only for the independent sensitivity sweeps and are distinct from the accuracy–eficiency configuration adopted in the main experiments. The KAN-architecture sweep uses our CUDAfused implementation and is therefore mainly interpreted through within-block comparisons.

The KAN branch is highly sensitive to its architecture, while increasing network depth or width does not nec essarily improve reconstruction performance. In contrast, the MLP branch remains relatively stable across diferent depths and widths. We further evaluate the sensitivity to the wavelet decomposition level J and basis on the first 100 DIV2K images at $5 1 2 \times 5 1 2$ . Table VII reports the results under the accuracy–eficiency configuration described in Section IV-D. The evaluated settings exhibit only minor performance variations. Therefore, we adopt $J = 3$ and db1 as the wavelet configuration in the main experiments.

## B. Ablation on Band-Separation Regularization

To isolate the efect of the band-separation regularization, we compare the full model with a variant denoted $\mathrm { W / o }$ Band Loss on 1D audio, 2D image, 3D volume, and 4D LF fitting tasks. Both variants retain the same DWT/IDWT operations and additive wavelet-domain fusion, while $\mathcal { L } _ { \mathrm { b a n d } }$ is removed only in the ablated variant. All other network and optimization settings remain unchanged. As shown in Fig. 13, removing $\mathcal { L } _ { \mathrm { b a n d } }$ consistently degrades reconstruction quality across the evaluated tasks, demonstrating its contribution to frequency-oriented coordination and reconstruction fidelity.

To further examine the interaction between the fusion rule and band-separation regularization, we conduct a controlled study on 100 DIV2K images using four variants: additive fusion with and without ${ \mathcal { L } } _ { \mathrm { b a n d } } ,$ and fixed

TABLE V: Computational eficiency comparison across 1D–4D signal-fitting tasks using a batch size of 2048.
<table><tr><td>Task</td><td>Method</td><td>Params.</td><td>FLOPs</td><td>Time (s)</td><td>Mem. (MiB)</td></tr><tr><td rowspan="7">ue uve</td><td>FINER</td><td>198K</td><td>0.85G</td><td>0.4</td><td>46.5</td></tr><tr><td>MFN</td><td>201K</td><td>0.91G</td><td>2.0</td><td>74.8</td></tr><tr><td>FR</td><td>1.57M</td><td>1.65G</td><td>0.5</td><td>61.3</td></tr><tr><td>MLP-PE</td><td>332K</td><td>1.42G</td><td>0.8</td><td>48.4</td></tr><tr><td>SIREN</td><td>329K</td><td>1.41G</td><td>0.3</td><td>48.3</td></tr><tr><td>WIRE</td><td>264K</td><td>2.27G</td><td>0.5</td><td>64.3</td></tr><tr><td>KAN</td><td>869K</td><td>3.95G</td><td>9.2</td><td>1261.4</td></tr><tr><td>1D</td><td>BACON</td><td>267K</td><td>1.13G</td><td>1.4</td><td>33.0</td></tr><tr><td rowspan="8">iage D</td><td>Ours</td><td>1.97M</td><td>0.55G</td><td>1.7</td><td>207.6</td></tr><tr><td>FINER</td><td>198K</td><td>0.86G</td><td>0.9</td><td>46.1</td></tr><tr><td>MFN FR</td><td>201K</td><td>0.26G</td><td>7.9</td><td>46.8</td></tr><tr><td>MLP-PE</td><td>1.57M</td><td>1.66G</td><td>1.9</td><td>63.1</td></tr><tr><td></td><td>332K 329K</td><td>0.88G</td><td>20.9</td><td>40.9</td></tr><tr><td>SIREN WIRE</td><td>264K</td><td>1.13G 2.86G</td><td>3.8</td><td>45.3</td></tr><tr><td>KAN</td><td>136K</td><td>0.69G</td><td>18.1</td><td>66.1</td></tr><tr><td>BACON</td><td>405K</td><td>1.72G</td><td>17.3 1.6</td><td>390.8 80.7</td></tr><tr><td></td><td>Ours</td><td>4.13M</td><td>20.51G</td><td>27.7</td><td>108.6</td></tr><tr><td rowspan="10"> el</td><td>FINER</td><td>264K</td><td>1.08G</td><td>25.7</td><td>101.3</td></tr><tr><td>MFN</td><td>274K</td><td>1.12G</td><td>334.1</td><td>101.3</td></tr><tr><td>FR</td><td>2.10M</td><td>2.15G</td><td>26.6</td><td>112.3</td></tr><tr><td>MLP-PE</td><td>180K</td><td>0.86G</td><td>186.9</td><td>97.1</td></tr><tr><td>SIREN</td><td>264K</td><td>1.07G</td><td>15.9</td><td>97.3</td></tr><tr><td>WIRE</td><td>265K</td><td>0.53G</td><td>269.0</td><td>111.0</td></tr><tr><td>KAN</td><td>449K</td><td>5.13G</td><td>349.2</td><td>608.5</td></tr><tr><td>BACON</td><td>529K</td><td>1.09G</td><td>223.5</td><td>101.6</td></tr><tr><td>Ours</td><td>4.56M</td><td>19.09G</td><td>136.6</td><td>954.8</td></tr><tr><td></td><td></td><td></td><td></td><td></td></tr><tr><td rowspan="8">4DD</td><td>FINER MFN</td><td>199K</td><td>0.86G</td><td>1.1</td><td>56.64</td></tr><tr><td>FR</td><td>228K</td><td>0.94G</td><td>24.5</td><td>75.2</td></tr><tr><td>MLP-PE</td><td>1.58M 363K</td><td>1.66G</td><td>1.4</td><td>61.4</td></tr><tr><td>SIREN</td><td>256K</td><td>0.90G</td><td>1.0</td><td>33.8</td></tr><tr><td>WIRE</td><td></td><td>1.14G</td><td>1.7</td><td>43.6</td></tr><tr><td></td><td>266K</td><td>27.10G</td><td>47.5</td><td>76.4</td></tr><tr><td>KAN</td><td>142K</td><td>3.65G</td><td>37.6</td><td>355.0</td></tr><tr><td>BACON</td><td>266K</td><td>1.73G</td><td>2.2</td><td>77.2</td></tr><tr><td>Ours</td><td>4.42M</td><td>11.43G</td><td>47.1</td><td>65.6</td></tr></table>

TABLE VI: Hyperparameter sensitivity analysis on the first 100 DIV2K images at $5 1 2 \times 5 1 2$ . Each block represents an independent hyperparameter sweep. The nominal settings are MLP $3 \times 2 5 6$ , KAN $3 \times 1 2 8$ $k = 3$ , and grid size 64.
<table><tr><td></td><td>Setting</td><td>PSNR (dB) ↑</td><td>SSIM ↑</td><td>LPIPS ↓</td></tr><tr><td rowspan="4">Grid</td><td>32</td><td> $3 9 . 9 8 \pm 3 . 6 4$ </td><td> $0 . 9 7 8 5 \pm 0 . 0 0 9 0$ </td><td> $0 . 0 0 7 6 \pm 0 . 0 0 5 7$ </td></tr><tr><td>64</td><td> ${ \bf 5 0 . 7 7 \pm 3 . 4 6 }$ </td><td> $\mathbf { 0 . 9 9 7 4 \pm 0 . 0 0 1 0 }$ </td><td> $\mathbf { 0 . 0 0 0 5 \pm 0 . 0 0 0 3 }$ </td></tr><tr><td>128</td><td> $4 2 . 8 8 \pm 5 . 0 6$ </td><td> $0 . 9 8 1 4 \pm 0 . 0 0 9 4$ </td><td> $0 . 0 0 7 2 \pm 0 . 0 0 7 9$ </td></tr><tr><td> $k = 1$ </td><td> $4 7 . 1 3 \pm 2 . 5 2$ </td><td> $0 . 9 9 3 8 \pm 0 . 0 0 4 2$ </td><td> $0 . 0 0 1 5 \pm 0 . 0 0 1 2$ </td></tr><tr><td rowspan="4">Deree</td><td> $k = 2$ </td><td> $4 9 . 9 2 \pm 2 . 2 8$ </td><td> $0 . 9 9 7 0 \pm 0 . 0 0 1 4$ </td><td> $0 . 0 0 0 6 \pm 0 . 0 0 0 4$ </td></tr><tr><td> $k = 3$ </td><td> ${ \bf 5 0 . 9 0 \pm 3 . 3 5 }$ </td><td> $\mathbf { 0 . 9 9 7 5 \pm 0 . 0 0 0 8 }$ </td><td> $\mathbf { 0 . 0 0 0 5 \pm 0 . 0 0 0 4 }$ </td></tr><tr><td> $1 \times 6 4$ </td><td> $2 2 . 8 9 \pm 2 . 7 3$ </td><td> $0 . 5 5 7 9 \pm 0 . 0 9 5 0$ </td><td> $0 . 5 5 6 6 \pm 0 . 0 5 9 0$ </td></tr><tr><td> $2 \times 6 4$ </td><td> $2 8 . 8 6 \pm 3 . 1 1$ </td><td> $0 . 8 0 8 1 \pm 0 . 0 4 6 0$ </td><td> $0 . 2 0 3 4 \pm 0 . 0 5 3 0$ </td></tr><tr><td rowspan="10">(M×T)NM</td><td> $3 \times 6 4$ </td><td> $3 5 . 2 9 \pm 3 . 8 8$ </td><td> $0 . 9 3 9 4 \pm 0 . 0 2 4 0$ </td><td> $0 . 0 4 8 0 \pm 0 . 0 2 5 0$ </td></tr><tr><td> $4 \times 6 4$ </td><td> $3 9 . 2 4 \pm 3 . 7 5$ </td><td> $0 . 9 7 0 1 \pm 0 . 0 1 2 0$ </td><td> $0 . 0 2 0 3 \pm 0 . 0 4 8 0$ </td></tr><tr><td> $1 \times 1 2 8$ </td><td> $3 0 . 2 1 \pm 3 . 5 7$ </td><td> $0 . 8 5 4 0 \pm 0 . 0 4 9 0$ </td><td> $0 . 1 2 9 9 \pm 0 . 0 3 8 0$ </td></tr><tr><td> $2 \times 1 2 8$ </td><td> $3 1 . 7 4 \pm 2 . 7 1$ </td><td> $0 . 9 3 4 5 \pm 0 . 0 0 1 0$ </td><td> $0 . 0 0 8 0 \pm 0 . 0 1 3 1$ </td></tr><tr><td> $3 \times 1 2 8$ </td><td> $4 5 . 5 9 \pm 4 . 8 4$ </td><td> $0 . 9 9 1 6 \pm 0 . 0 0 6 0$ </td><td> $0 . 0 0 5 6 \pm 0 . 0 0 7 0$ </td></tr><tr><td> $4 \times 1 2 8$ </td><td> $4 7 . 0 8 \pm 5 . 5 5$ </td><td> $0 . 9 9 2 7 \pm 0 . 0 0 5 0$ </td><td> $0 . 0 0 2 9 \pm 0 . 0 0 7 0$ </td></tr><tr><td> $1 \times 2 5 6$ </td><td> $2 4 . 0 8 \pm 1 . 1 1$ </td><td> $0 . 6 8 7 5 \pm 0 . 1 0 3 0$ </td><td> $0 . 4 1 4 3 \pm 0 . 0 7 8 0$ </td></tr><tr><td> $2 \times 2 5 6$ </td><td> $4 9 . 6 6 \pm 6 . 0 2$ </td><td> $0 . 9 9 3 9 \pm 0 . 0 0 5 0$ </td><td> $0 . 0 0 0 6 \pm 0 . 0 0 3 0$ </td></tr><tr><td> $3 \times 2 5 6$ </td><td> ${ \bf 6 6 . 2 2 \pm 5 . 2 8 }$ </td><td> $\mathbf { 0 . 9 9 9 8 \pm 0 . 0 0 0 3 }$ </td><td> $\mathbf { 0 . 0 0 0 0 4 } \pm \mathbf { 0 . 0 0 0 1 }$ </td></tr><tr><td> $4 \times 2 5 6$ </td><td> $5 6 . 5 1 \pm 4 . 0 2$ </td><td> $0 . 9 9 8 5 \pm 0 . 0 0 1 0$ </td><td> $0 . 0 0 0 6 \pm 0 . 0 0 1 0$ </td></tr><tr><td rowspan="19">(×)</td><td></td><td></td><td></td><td></td></tr><tr><td> $1 \times 6 4$   $2 \times 6 4$ </td><td> $4 2 . 6 2 \pm 1 . 9 7$   $5 0 . 2 5 \pm 2 . 1 5$ </td><td> $0 . 9 9 2 0 \pm 0 . 0 0 1 3$   $0 . 9 9 7 2 \pm 0 . 0 0 0 9$ </td><td> $0 . 0 0 1 0 \pm 0 . 0 0 0 3$   $0 . 0 0 0 5 \pm 0 . 0 0 0 3$ </td></tr><tr><td> $3 \times 6 4$ </td><td> $5 0 . 8 9 \pm 3 . 3 9$ </td><td> $0 . 9 9 7 4 \pm 0 . 0 0 0 8$ </td><td> $0 . 0 0 0 4 \pm 0 . 0 0 0 3$ </td></tr><tr><td> $4 \times 6 4$ </td><td> $5 1 . 0 3 \pm 3 . 3 8$ </td><td> $0 . 9 9 7 5 \pm 0 . 0 0 0 7$ </td><td> $0 . 0 0 0 4 \pm 0 . 0 0 0 4$ </td></tr><tr><td> $1 \times 1 2 8$ </td><td> $4 4 . 3 4 \pm 1 . 6 9$ </td><td> $0 . 9 9 2 3 \pm 0 . 0 0 0 2$ </td><td> $0 . 0 0 1 8 \pm 0 . 0 0 0 4$ </td></tr><tr><td> $2 \times 1 2 8$ </td><td> $5 0 . 8 7 \pm 3 . 1 6$ </td><td> $0 . 9 9 7 5 \pm 0 . 0 0 0 6$ </td><td> $0 . 0 0 0 5 \pm 0 . 0 0 0 5$ </td></tr><tr><td> $3 \times 1 2 8$ </td><td> $5 0 . 6 4 \pm 3 . 2 0$ </td><td> $0 . 9 9 7 3 \pm 0 . 0 0 0 8$ </td><td> $0 . 0 0 0 5 \pm 0 . 0 0 0 5$ </td></tr><tr><td> $4 \times 1 2 8$ </td><td> $5 1 . 2 1 \pm 3 . 5 6$ </td><td> $0 . 9 9 7 5 \pm 0 . 0 0 0 8$ </td><td> $0 . 0 0 0 4 \pm 0 . 0 0 0 4$ </td></tr><tr><td>1 × 256</td><td> $4 4 . 5 0 \pm 2 . 2 3 $ </td><td> $0 . 9 9 2 5 \pm 0 . 0 0 0 9$ </td><td> $0 . 0 0 0 9 \pm 0 . 0 0 0 4$ </td></tr><tr><td> $2 \times 2 5 6$ </td><td> $5 0 . 8 9 \pm 3 . 4 1$ </td><td> $0 . 9 9 7 4 \pm 0 . 0 0 0 7$ </td><td> $0 . 0 0 0 5 \pm 0 . 0 0 0 3$ </td></tr><tr><td> $3 \times 2 5 6$ </td><td> $5 0 . 5 2 \pm 2 . 3 1$ </td><td> $0 . 9 9 7 4 \pm 0 . 0 0 0 8$ </td><td> $0 . 0 0 0 4 \pm 0 . 0 0 0 3$ </td></tr><tr><td></td><td></td><td></td><td></td></tr><tr><td> $4 \times 2 5 6$ </td><td> ${ \bf 5 1 . 2 9 \pm 4 . 0 9 }$ </td><td> $\mathbf { 0 . 9 9 7 5 \pm 0 . 0 0 0 6 }$ </td><td> $\mathbf { 0 . 0 0 0 4 } \pm \mathbf { 0 . 0 0 0 2 }$ </td></tr></table>

routing with and without $\mathcal { L } _ { \mathrm { b a n d } }$ . All other settings are kept identical. Additive + Band corresponds to the full model, whereas Additive + No Band uses the same 2D setting as the $\mathrm { W / o }$ Band variant in Fig. 13. The fixed-routing variants are introduced only as controlled baselines that explicitly impose the ideal frequency-selective assignment described in Eq. (10), rather than as the forward mechanism of the proposed model.

As shown in Table VIII, $\mathcal { L } _ { \mathrm { b a n d } }$ increases the KAN highfrequency proportion and improves reconstruction quality under both fusion strategies; under additive fusion, it also further suppresses the already small MLP high-frequency response. Fixed routing yields a substantially larger KAN high-frequency proportion but markedly lower reconstruction accuracy due to the restricted coeficient assignment, indicating that hard frequency assignment is overly restrictive. Additive fusion with band-separation regularization therefore provides a better balance between frequencyoriented coordination and reconstruction flexibility.

![](images/28c9a50a1aefb2e860131319465090b0c688c2e92c473a2afad8d54a948144b4.jpg)  
Fig. 13: Ablation results for the band-separation regularization on 1D audio, 2D images, 3D volumetric data, and 4D LFs. The full model is compared with a variant in which $\mathcal { L } _ { \mathrm { b a n d } }$ is removed while DWT/IDWT and additive wavelet domain fusion are retained. Quantitative metrics are reported in the corresponding subfigures.

## C. Parameter-Controlled Comparison

To investigate whether the performance improvement mainly comes from the increased model capacity, we conduct an additional parameter-controlled experiment. Specifically, the compared methods are adjusted toward a parameter scale comparable to that of our model (4.13M), subject to their architecture-specific parameterization constraints. Detailed network configurations are provided in the supplementary material.

The results are summarized in Table IX. Under this parameter-controlled setting, our method still achieves strong reconstruction performance, indicating that the observed improvement cannot be explained solely by a larger trainable-parameter budget.

## VI. Discussion

The experiments on 1D square-wave and audio signals, 2D images, 3D volumes, SDFs, videos, and 4D LFs demonstrate the applicability of the proposed framework to signals with diferent dimensionalities and frequency characteristics. The MLP-KAN architecture provides two heterogeneous representation pathways, while the wavelet module exposes and coordinates their frequency responses through DWT, additive coeficient fusion, and bandseparation regularization. The resulting representation is frequency-oriented: the MLP branch exhibits a stronger low-frequency tendency, while the KAN branch provides relatively stronger responses to localized variations and detail components. Both branches remain involved in the reconstruction because the corresponding wavelet coeficients are fused additively.

The ablation experiments further clarify the roles of the fusion strategy and spectral regularization. Removing the band-separation loss consistently reduces reconstruction quality across the evaluated 1D, 2D, 3D, and 4D tasks. The branch-wise wavelet-energy analysis also shows that the regularization increases the relative high-frequency response of the KAN branch and further suppresses the small high-frequency response of the MLP branch. The KAN branch still contains substantial low-frequency energy, indicating a frequency-oriented tendency rather than complete spectral separation. Fixed routing produces a stronger concentration of KAN energy in the detail bands, but leads to substantially lower reconstruction accuracy. These observations indicate that hard frequency assignment is too restrictive for the current dual-branch representation, while additive fusion with band-separation regularization provides a better balance between frequency coordination and reconstruction flexibility. The parametercontrolled comparison further shows that the reconstruction advantage cannot be attributed solely to a larger parameter budget, since the proposed method maintains strong performance when the compared methods are adjusted toward a parameter scale comparable to that of the proposed model.

TABLE VII: Sensitivity analysis of the wavelet level J and Daubechies basis on the first 100 DIV2K images at $5 1 2 \times 5 1 2$ . The adopted configuration uses an MLP with three hidden layers of width 64 and a KAN with widths 128-256-128, k = 3, and grid size 64.
<table><tr><td>Setting</td><td>PSNR (dB) ↑</td><td>SSIM ↑</td><td>LPIPS↓</td></tr><tr><td>J = 1, db1 J = 1, db2</td><td> $4 5 . 1 1 6 0 \pm 2 . 3 5 7 4$   $0 . 9 9 1 0 \pm 0 . 0 0 3 6$   $4 5 . 1 4 6 3 \pm 2 . 3 6 1 0$   $0 . 9 9 1 1 \pm 0 . 0 0 3 0$   $4 5 . 1 3 9 0 \pm 2 . 3 7 3 0$ </td></tr><tr><td colspan="2">J = 1, db3  $0 . 9 9 1 1 \pm 0 . 0 0 3 1$   $0 . 0 0 3 0 \pm 0 . 0 0 2 2$  J = 1, db4  $4 5 . 1 0 6 8 \pm 2 . 3 4 1 5$   $0 . 9 9 1 0 \pm 0 . 0 0 3 0$   $0 . 0 0 3 0 \pm 0 . 0 0 2 1$   $0 . 0 0 3 0 \pm 0 . 0 0 1 6$ </td></tr><tr><td colspan="2">J = 2, db1  $4 5 . 1 5 2 7 \pm 2 . 3 7 0 4$   $0 . 9 9 1 1 \pm 0 . 0 0 3 1$ </td></tr><tr><td colspan="2">J = 2, db2  $4 5 . 1 2 2 6 \pm 2 . 3 3 1 8$   $0 . 9 9 1 0 \pm 0 . 0 0 3 1$   $0 . 0 0 3 1 \pm 0 . 0 0 1 6$ </td></tr><tr><td colspan="2">J = 2, db3  $4 5 . 1 6 6 7 \pm 2 . 3 6 0 0$   $0 . 9 9 1 1 \pm 0 . 0 0 3 1$   $0 . 0 0 3 1 \pm 0 . 0 0 1 7$  J = 2, db4  $4 5 . 1 4 7 3 \pm 2 . 3 0 3 3$   $0 . 9 9 1 0 \pm 0 . 0 0 3 1$   $0 . 0 0 3 1 \pm 0 . 0 0 1 7$   $0 . 0 0 3 2 \pm 0 . 0 0 2 0$ </td></tr><tr><td colspan="2">J = 3, db1  $4 5 . 1 6 8 7 \pm 2 . 3 3 2$   $0 . 9 9 1 1 \pm 0 . 0 0 3 0$  J = 3, db2  $4 5 . 1 6 9 8 \pm 2 . 3 7 0$   $0 . 9 9 1 0 \pm 0 . 0 0 3 0$  J = 3, db3  $4 5 . 1 7 4 7 \pm 2 . 3 0 2$   $0 . 9 9 1 1 \pm 0 . 0 0 3 0$ </td></tr></table>

The wavelet decomposition level, KAN grid size, spline degree, and branch architectures remain dependent on the signal dimensionality and data characteristics. The framework also involves a trade-of between representation accuracy, model complexity, and computational eficiency. Future work will investigate lighter heterogeneous branch designs, adaptive frequency-domain coordination, and more eficient CUDA implementations to improve the accuracy–eficiency balance while preserving reconstruction fidelity.

## VII. Conclusion

We present a spatial–frequency-aware implicit neural representation framework that integrates MLP and KAN branches through wavelet-domain decomposition and additive fusion, together with band-separation regularization. The proposed design exploits the diferent frequency tendencies of the two branches while allowing both to contribute to the reconstructed signal. The spectral regularization further promotes complementary branch responses without imposing exclusive subband assignment.

Experiments on 1D signals, 2D images, 3D volumes and SDFs, videos, and 4D LFs demonstrate its efectiveness across diverse signal dimensions and modalities. Controlled ablations verify the contribution of the band separation regularization and show that additive fusion provides higher reconstruction fidelity than fixed routing under the evaluated settings. Parameter-controlled experiments further indicate that the performance improvement is not explained solely by model size. These results support wavelet-domain coordination of heterogeneous neural representations as an efective approach for multidimensional implicit signal fitting.

## Acknowledgment

The authors thank doctoral researchers Dingbo Hou and Yulin Han and engineers Yao Guo and Lin Nai from the Institute of Computational Imaging, Beijing Information Science and Technology University, for their assistance with the comparative experiments.

## References

[1] Y. Xie, T. Takikawa, S. Saito, O. Litany, S. Yan, N. Khan, F. Tombari, J. Tompkin, V. Sitzmann, and S. Sridhar, “Neural fields in visual computing and beyond,” in Computer Graphics Forum, vol. 41, no. 2. Wiley Online Library, 2022, pp. 641–676.

[2] A. Essakine, Y. Cheng, C.-W. Cheng, L. Zhang, Z. Deng, L. Zhu, C.-B. Sch¨onlieb, and A. I. Aviles-Rivero, “Where do we stand with implicit neural representations? a technical and performance survey,” Transactions on Machine Learning Research, 2025, survey Certification. [Online]. Available: https: //openreview.net/forum?id=QTsJXSvAI2

[3] Y. Luo, X. Zhao, and D. Meng, “Continuous representation methods, theories, and applications: An overview and perspective,” Science China Information Sciences, vol. 69, no. 5, 2026.

[4] J. J. Park, P. Florence, J. Straub, R. Newcombe, and S. Lovegrove, “Deepsdf: Learning continuous signed distance functions for shape representation,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2019, pp. 165–174.

[5] C. Yang, Y. Zhou, G. Wei, L. Ma, J. Hou, Y. Liu, and W. Wang, “Monge-ampere regularization for learning arbitrary shapes from point clouds,” IEEE Transactions on Pattern Analysis and Machine Intelligence, vol. 47, no. 8, pp. 6809–6822, 2025.

[6] B. Mildenhall, P. P. Srinivasan, M. Tancik, J. T. Barron, R. Ramamoorthi, and R. Ng, “Nerf: Representing scenes as neural radiance fields for view synthesis,” in European Conference on Computer Vision, 2020.

[7] J. T. Barron, B. Mildenhall, M. Tancik, P. Hedman, R. Martin-Brualla, and P. P. Srinivasan, “Mip-nerf: A multiscale representation for anti-aliasing neural radiance fields,” in 2021 IEEE/CVF International Conference on Computer Vision (ICCV). IEEE, 2021, pp. 5835–5844.

[8] F. Wu, B. Hu, and S. Z. Li, “Generalized implicit neural representations for dynamic molecular surface modeling,” in Proceedings of the AAAI Conference on Artificial Intelligence, vol. 39, no. 1, 2025, pp. 877–885.

[9] G. Gao, H. M. Kwan, F. Zhang, and D. Bull, “Pnvc: Towards practical inr-based video compression,” in Proceedings of the AAAI Conference on Artificial Intelligence, vol. 39, no. 3, 2025, pp. 3068–3076.

[10] C. Zhu, G. Lu, B. He, R. Xie, and L. Song, “Implicit-explicit integrated representations for multi-view video compression,” IEEE Transactions on Image Processing, vol. 34, pp. 1106–1118, 2025.

[11] Z. Zhang, L. Zhang, Y. Cheng, Z. Wang, F. Wang, H. Zhang, Y. Yang, Y. Wu, J. Huang, A. I. Aviles-Rivero et al., “From coarse to continuous: Progressive refinement implicit neural representation for motion-robust anisotropic mri reconstruction,” IEEE Transactions on Image Processing, vol. 35, pp. 3550–3565, 2026.

TABLE VIII: Controlled analysis of the fusion strategy and band separation regularization on 100 DIV2K images. $\mathrm { M L P - L / H }$ and KAN-L/H denote the low- and high-frequency energy proportions normalized within the corresponding branch. Results are reported as mean ± standard deviation.
<table><tr><td rowspan="2">Variant</td><td colspan="3">Reconstruction Performance</td><td colspan="4">Branch-wise Wavelet Energy (%)</td></tr><tr><td>PSNR (dB) ↑</td><td>SSIM ↑</td><td>LPIPS ↓</td><td>MLP-L</td><td>MLP-H</td><td>KAN-L</td><td>KAN-H</td></tr><tr><td> $\mathbf { A d d i t i v e } + \mathbf { B a n d }$ </td><td> $\mathbf { 4 4 . 6 0 { \pm } 2 . 6 5 }$ </td><td> $\mathbf { 0 . 9 9 0 8 { \scriptstyle \pm 0 . 0 0 2 7 } }$ </td><td>0.0031±0.0017</td><td> $( 9 9 . 9 9 9 9 3 \pm 3 . 0 ) \times 1 0 ^ { - 5 }$ </td><td> $( 7 . 5 \pm 3 . 0 ) \times 1 0 ^ { - 5 }$ </td><td> $8 4 . 3 4 { \pm } 6 . 8 0 $ </td><td> $1 5 . 6 7 { \scriptstyle \pm 6 . 8 0 }$ </td></tr><tr><td> $\mathrm { A d d i t i v e } + \mathrm { N o \ B a n d }$ </td><td> $2 8 . 9 3 { \pm } 2 . 7 9$ </td><td> $0 . 6 9 1 4 { \scriptstyle \pm 0 . 0 0 2 7 }$ </td><td> $0 . 0 3 9 7 { \scriptstyle \pm 0 . 0 0 1 5 }$ </td><td> $( 9 9 . 9 9 9 8 1 \pm 8 . 0 ) \times 1 0 ^ { - 5 }$ </td><td> $( 1 . 8 6 \pm 0 . 8 0 ) \times 1 0 ^ { - 4 }$ </td><td> $9 6 . 0 0 { \scriptstyle \pm 2 . 9 0 }$ </td><td> $4 . 0 0 { \scriptstyle \pm 2 . 9 0 }$ </td></tr><tr><td> $\mathrm { F i x e d } + \mathrm { B a n d }$ </td><td> $1 9 . 2 1 { \pm } 1 . 6 4 $ </td><td> $0 . 8 6 2 8 { \scriptstyle \pm 0 . 0 4 5 1 }$ </td><td> $0 . 1 7 5 8 { \scriptstyle \pm 0 . 0 6 5 5 }$ </td><td> $( 9 9 . 9 9 7 2 2 \pm 1 . 2 ) \times 1 0 ^ { - 3 }$ </td><td> $( 2 . 7 8 \pm 1 . 2 0 ) \times 1 0 ^ { - 3 }$ </td><td> $5 2 . 6 3 { \pm } 1 2 . 4 0 $ </td><td> $4 7 . 3 7 { \pm } 1 2 . 4 0 $ </td></tr><tr><td> $\mathrm { F i x e d } + \mathrm { N o } \ \mathrm { B a n d }$ </td><td> $1 7 . 3 5 { \pm } 2 . 0 0$ </td><td> $0 . 7 5 6 0 { \scriptstyle \pm 0 . 1 5 4 1 }$ </td><td> $0 . 2 6 3 5 { \pm } 0 . 1 5 4 2$ </td><td> $( 9 9 . 9 9 7 7 9 { \scriptstyle \pm 1 . 0 ) \times 1 0 ^ { - 3 } }$ </td><td> $( 2 . 2 1 \pm 1 . 0 0 ) \times 1 0 ^ { - 3 }$ </td><td> $7 2 . 0 2 { \pm } 9 . 1 0 $ </td><td> $2 7 . 9 8 { \pm } 9 . 1 0 $ </td></tr></table>

TABLE IX: Parameter-controlled comparison of image fitting performance on the DIV2K dataset. The compared methods are adjusted toward a trainable-parameter scale comparable to that of the proposed method (4.13M), subject to their architecture-specific parameterization constraints. Results are reported as mean ± standard deviation.
<table><tr><td>Method</td><td>PSNR (dB) ↑</td><td>SSIM ↑</td><td>LPIPS↓</td></tr><tr><td>FINER</td><td> $4 1 . 6 5 \pm 2 . 7 4 6$ </td><td> $0 . 9 8 5 3 \pm 0 . 0 0 9 0$ </td><td> $0 . 0 2 3 4 \pm 0 . 0 2 0 7$ </td></tr><tr><td>MFN</td><td> $3 0 . 1 9 \pm 3 . 0 2 7$ </td><td> $0 . 9 1 9 3 \pm 0 . 0 2 7 8$ </td><td> $0 . 1 5 3 1 \pm 0 . 0 4 0 2$ </td></tr><tr><td>FR</td><td> $4 1 . 6 6 \pm 3 . 8 4 8$ </td><td> $0 . 9 7 2 6 \pm 0 . 0 3 2 0$ </td><td> $0 . 0 4 7 7 \pm 0 . 0 5 5 9$ </td></tr><tr><td>MLP-PE</td><td> $2 9 . 2 4 \pm 1 . 8 2 1$ </td><td> $0 . 8 0 8 5 \pm 0 . 0 3 8 1$ </td><td> $0 . 2 8 8 5 \pm 0 . 0 6 2 6$ </td></tr><tr><td>SIREN</td><td> $4 0 . 7 4 \pm 2 . 6 6 2$ </td><td> $0 . 9 9 1 0 \pm 0 . 0 0 2 4$ </td><td> $0 . 0 2 0 9 \pm 0 . 0 1 1 1$ </td></tr><tr><td>WIRE</td><td> $3 9 . 2 2 \pm 6 . 8 8 7$ </td><td> $0 . 9 5 8 5 \pm 0 . 0 6 6 6$ </td><td> $0 . 0 4 4 3 \pm 0 . 0 4 7 4$ </td></tr><tr><td>KAN</td><td> $2 8 . 3 6 \pm 3 . 1 6 3$ </td><td> $0 . 8 7 9 1 \pm 0 . 0 5 3 0$ </td><td> $0 . 2 3 3 6 \pm 0 . 0 6 9 8$ </td></tr><tr><td>BACON</td><td> $3 9 . 3 2 \pm 2 . 1 4 6$ </td><td> $0 . 9 8 0 3 \pm 0 . 0 0 8 6$ </td><td> $0 . 0 5 1 0 \pm 0 . 0 3 7 0$ </td></tr><tr><td>Ours</td><td> $\mathbf { 4 5 . 7 2 \pm 2 . 3 4 9 }$ </td><td> $\mathbf { 0 . 9 9 1 2 \pm 0 . 0 0 3 0 }$ </td><td> $\mathbf { 0 . 0 0 3 0 \pm 0 . 0 0 2 0 }$ </td></tr></table>

[12] C. Liu, L. Shi, L. Wu, K. Yu, and J. Qiu, “Neural light field representation and reconstruction based on a ray displacement field,” Optics Express, vol. 34, no. 8, pp. 15 468–15 483, Apr 2026.

[13] R. Cong, H. Sheng, Y. Wang, D. Yang, Z. Cui, W. Lyv, Y. Zhang, and W. Ke, “Learning three-domain implicit image function for arbitrary-scale light field super-resolution,” IEEE Transactions on Pattern Analysis and Machine Intelligence, 2026.

[14] L. Shi, C. Liu, X. Zhao, and J. Qiu, “Iterative approach to reconstructing neural disparity fields from light-field data,” IEEE Transactions on Computational Imaging, vol. 11, pp. 410–420, 2025.

[15] S. Sun, L. Shi, J. Qiu, H. Zhang, L. Wu, and C. Liu, “Multiresolution infrared and visible image fusion via implicit neural representations,” IEEE Transactions on Instrumentation and Measurement, vol. 75, pp. 5 017 215–5 017 215, 2026.

[16] Q. Zhou, Y. Ye, and Z. Cai, “Spatiotemporal-aware neural fields for dynamic ct reconstruction,” in Proceedings of the AAAI Conference on Artificial Intelligence, vol. 39, no. 10, 2025, pp. 10 834–10 842.

[17] Z. Shen, H. Zhang, B. Dong, J. Qiu, Y. Li, and Z. Cui, “Incomplete data multisource static computed tomography reconstruction with difusion priors and implicit neural representation,” SIAM Journal on Imaging Sciences, vol. 19, no. 2, pp. 1174– 1206, 2026.

[18] L. Shi, X. Jiang, Y. Liu, C. Liu, P. Yang, S. Guo, and X. Zhao, “Ring artifacts removal based on implicit neural representation of sinogram data,” IEEE Transactions on Image Processing, vol. 34, pp. 4080–4091, 2025.

[19] N. Rahaman, A. Baratin, D. Arpit, F. Draxler, M. Lin, F. Hamprecht, Y. Bengio, and A. Courville, “On the spectral bias of neural networks,” in International Conference on Machine Learning. PMLR, 2019, pp. 5301–5310.

[20] A. Bietti and J. Mairal, “On the inductive bias of neural tangent

kernels,” Advances in Neural Information Processing Systems, vol. 32, 2019.

[21] M. Tancik, P. Srinivasan, B. Mildenhall, S. Fridovich-Keil, N. Raghavan, U. Singhal, R. Ramamoorthi, J. Barron, and R. Ng, “Fourier features let networks learn high frequency functions in low dimensional domains,” Advances in Neural Information Processing Systems, vol. 33, pp. 7537–7547, 2020.

[22] V. Sitzmann, J. Martel, A. Bergman, D. Lindell, and G. Wetzstein, “Implicit neural representations with periodic activation functions,” Advances in Neural Information Processing systems, vol. 33, pp. 7462–7473, 2020.

[23] T. M¨uller, A. Evans, C. Schied, and A. Keller, “Instant neural graphics primitives with a multiresolution hash encoding,” ACM Transactions on Graphics (TOG), vol. 41, no. 4, pp. 1–15, 2022.

[24] S. Xie, H. Zhu, Z. Liu, Q. Zhang, Y. Zhou, X. Cao, and Z. Ma, “Diner: Disorder-invariant implicit neural representation,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2023, pp. 6143–6152.

[25] Z. Liu, Y. Wang, S. Vaidya, F. Ruehle, J. Halverson, M. Soljacic, T. Hou, and M. Tegmark, “Kan: Kolmogorov–arnold networks,” in International conference on learning representations, vol. 2025, 2025, pp. 70 367–70 413.

[26] M. Unser, A. Aldroubi, and M. Eden, “B-spline signal processing. part i—theory,” IEEE Transactions on Signal Processing, vol. 41, no. 2, pp. 821–833, 1993.

[27] Y. Wang, J. Siegel, Z. Liu, and T. Hou, “On the expressiveness and spectral bias of kans,” in International Conference on Learning Representations, vol. 2025, 2025, pp. 27 492–27 511.

[28] Y.-N. Zhang, Q. Li, X. Wu, N. Mu, X. Li, and L. Shen, “A wavelet-guided deep unfolding network for single image reflection removal,” IEEE Transactions on Image Processing, vol. 34, pp. 4040–4051, 2025.

[29] S. Ramasinghe and S. Lucey, “Beyond periodicity: Towards a unifying framework for activations in coordinate-mlps,” in European Conference on Computer Vision. Springer, 2022, pp. 142–158.

[30] V. Saragadam, D. LeJeune, J. Tan, G. Balakrishnan, A. Veeraraghavan, and R. G. Baraniuk, “Wire: Wavelet implicit neural representations,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2023, pp. 18 507– 18 516.

[31] H. Saratchandran, S. Ramasinghe, V. Shevchenko, A. Long, and S. Lucey, “A sampling theory perspective on activations for implicit neural representations,” in Proceedings of the 41st International Conference on Machine Learning, ser. Proceedings of Machine Learning Research, vol. 235. PMLR, 2024, pp. 43 422–43 444.

[32] Z. Liu, H. Zhu, Q. Zhang, J. Fu, W. Deng, Z. Ma, Y. Guo, and X. Cao, “Finer: Flexible spectral-bias tuning in implicit neural representation by variable-periodic activation functions,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2024, pp. 2713–2722.

[33] H. Zhu, Z. Liu, Q. Zhang, J. Fu, W. Deng, Z. Ma, Y. Guo, and X. Cao, “Finer++: Building a family of variable-periodic functions for activating implicit neural representation,” IEEE Transactions on Pattern Analysis and Machine Intelligence, 2026.

[34] A. Kazerouni, R. Azad, A. Hosseini, D. Merhof, and U. Bagci, “Incode: Implicit neural conditioning with prior knowledge embeddings,” in Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision, 2024, pp. 1298–1307.

[35] R. Fathony, A. K. Sahu, D. Willmott, and J. Z. Kolter, “Multiplicative filter networks,” in International Conference on Learning Representations, 2020.

[36] K. Shi, X. Zhou, and S. Gu, “Improved implicit neural representation with fourier reparameterized training,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2024, pp. 25 985–25 994.

[37] J. Li, X. Zhao, J. Wang, C. Wang, and M. Wang, “Superpixelinformed implicit neural representation for multi-dimensional data,” in European Conference on Computer Vision. Springer, 2024, pp. 258–276.

[38] T.-W. Zhou, X.-L. Zhao, W.-H. Wu, J.-L. Wang, and Y.-S. Luo, “Frequency-aware implicit neural representation for multidimensional data recovery,” IEEE Transactions on Circuits and Systems for Video Technology, 2025.

[39] C. Yu, Y. Luo, K. Ye, X. Zhao, and D. Meng, “Cross-frequency implicit neural representation with self-evolving parameters,” IEEE Transactions on Pattern Analysis and Machine Intelligence, vol. 48, no. 8, pp. 9467–9485, 2026.

[40] S. Ko, S. Youn, D. Kye, K. Min, C. Eom, and J. Oh, “Flair: Frequency- and locality-aware implicit neural representations,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2026, pp. 4877–4887.

[41] Y. Luo, X. Zhao, Z. Li, M. K. Ng, and D. Meng, “Lowrank tensor function representation for multi-dimensional data recovery,” IEEE Transactions on Pattern Analysis and Machine Intelligence, vol. 46, no. 5, pp. 3351–3369, 2024.

2024. [Online]. Available: https://papers.bmvc2024.org/0959. pdf

[42] Y. Liu, F. Yan, P. Xu, C. Liu, and L. Wang, “Wavekan: Adaptive wavelet decomposition and kolmogorov-arnold network for longterm time series forecasting,” Neurocomputing, p. 133361, 2026.

[43] T. Liu, J. Xu, T. Lei, Y. Wang, X. Du, and Z. Lv, “Dgkan: Dual-branch graph kolmogorov-arnold network for unsupervised multimodal change detection,” in Proceedings of the AAAI Conference on Artificial Intelligence, vol. 40, no. 9, 2026, pp. 7278–7286.

[55] K. Honauer, O. Johannsen, D. Kondermann, and B. Goldluecke, “A dataset and evaluation methodology for depth estimation on 4d light fields,” in Asian Conference on Computer Vision. Springer, 2016, pp. 19–34.

[44] A. A. Aghaei, M. Hosseinzadeh, and K. Parand, “rkan: Rational kolmogorov-arnold networks,” Neural Networks, p. 108888, 2026.

[45] J. D. Toscano, L.-L. Wang, and G. E. Karniadakis, “Kkans: K˚urkov´a-kolmogorov-arnold networks and their learning dynamics,” Neural Networks, vol. 191, p. 107831, 2025.

[47] C.-G. Lucas and J. Gilles, “Multidimensional empirical wavelet transform,” SIAM Journal on Imaging Sciences, vol. 18, no. 2, pp. 906–935, 2025.

[48] C. Su, Y. Liu, P. Yang, L. Shi, and X. Zhao, “Ring artifacts correction based on global–local feature interaction guidance in the projection domain,” IEEE Transactions on Instrumentation and Measurement, vol. 74, pp. 1–14, 2025.

[46] H.-T. Ta, D.-Q. Thai, A. B. S. Rahman, G. Sidorov, and A. Gelbukh, “Fc-kan: Function combinations in kolmogorovarnold networks,” Information Sciences, vol. 736, p. 123103, 2026.

[49] J.-F. Cai, B. Dong, S. Osher, and Z. Shen, “Image restoration: total variation, wavelet frames, and beyond,” Journal of the American Mathematical Society, vol. 25, no. 4, pp. 1033–1089, 2012.

[50] H. Zhao, Z. Gao, Y. Wang, R. Xiong, and Y. Zhang, “Adaptive wavelet-positional encoding for high-frequency information learning in implicit neural representation,” in Proceedings of the AAAI Conference on Artificial Intelligence, vol. 39, no. 10, 2025, pp. 10 430–10 438.

[51] E. Agustsson and R. Timofte, “Ntire 2017 challenge on single image super-resolution: Dataset and study,” in Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition Workshops, 2017, pp. 126–135.

[52] A. Biguri, M. Dosanjh, S. Hancock, and M. Soleimani, “Tigre: a matlab-gpu toolbox for cbct image reconstruction,” Biomedical Physics and Engineering Express, vol. 2, no. 5, p. 055010, 2016.

[53] C. J. Holmes, R. Hoge, D. L. Collins, R. P. Woods, A. W. Toga, and A. C. Evans, “Enhancement of mr images using registration for signal averaging,” Journal of Computer Assisted Tomography, vol. 22, no. 2, pp. 324–333, 1998.

[54] E. Bogatyrev, I. Molodetskikh, and D. S. Vatolin, “Sr+codec: a benchmark of super-resolution for video compression bitrate reduction,” in 35th British Machine Vision Conference 2024, BMVC 2024, Glasgow, UK, November 25-28, 2024. BMVA,
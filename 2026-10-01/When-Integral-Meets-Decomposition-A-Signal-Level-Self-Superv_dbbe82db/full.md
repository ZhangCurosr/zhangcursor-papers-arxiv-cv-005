# When Integral Meets Decomposition: A Signal-Level Self-Supervised Feature Decompose Paradigm for Multi-Modal Image Fusion

Zeyu Wang<sup>1</sup> Jiayu Wang<sup>1,</sup> <sup>2</sup> Haiyu Song<sup>1,</sup> Haoran Duan<sup>3,</sup>

<sup>1</sup>College of Computer Science and Engineering, Dalian Minzu University, Dalian, China

<sup>2</sup>School of Artificial Intelligence and Robotics, Hunan University, Hunan, China

<sup>3</sup> Department of Automation, Tsinghua University, Beijing, China

{wangzeyu, shy}@dlnu.edu.cn jiayuwang@hnu.edu.cn haoran.duan@ieee.org

## Abstract

Multimodal image fusion (MMIF) aims to integrate complementary information from different modalities into a high-quality fused image and support downstream tasks. Recently, feature decomposition has become an important paradigm by separating source images into common and modality-specific unique features. However, existing methods lack clear supervision because ground-truth (GT) decomposition feature maps are unavailable. They usually combine multiple image-level metrics as losses, which are inherently incomplete and may conflict since each pixel couples attributes such as texture, edge, and contour. To address this, we propose a 1D signal-level self-supervised feature decomposition paradigm. Our core insight is to reformulate feature decomposition from unclear 2D image-level supervision into an integral-driven 1D signal-level optimization problem. This objective-level reformulation uses the 1D signal form to compute the integral constraint. The decomposer is optimized by the integral area between common and original signals, enabling more stable optimization with a clear optimization objective. Our model follows a two-stage SSL framework. Stage I designs dual pretext tasks for integral-driven decomposition at the signal level and structurepreserving reconstruction at the image level. Stage II fuses unique features and combines them with common features to reconstruct the fused image. Experiments on representative MMIF tasks show state-of-the-art (SOTA) performance. Code: github.com/Wangjiayu0512/SIDFusion.

## 1 Introduction

Multimodal image fusion (MMIF) integrates complementary information from different modalities to produce fused images [28, 30, 15, 7]. Typical tasks include medical image fusion (MIF) and infrared–visible fusion (VIF) [16, 49, 41, 18], where modalities capture complementary cues, such as thermal radiation in infrared images and textures in visible images [11]. MMIF also benefits downstream tasks, including medical image segmentation [19] and object detection [17, 20].

Currently, feature decomposition paradigm has become a key method [12]. It decomposes source images into two complementary feature types: common features representing shared scene information and unique features reflecting modality-specific details. This strategy facilitates the interaction and fusion of modality-specific information while preserving complete scene structure.

However, existing 2D image-level methods face a key challenge that decomposer training heavily depends on handcrafted or empirical loss design caused by the lack of ground truth (GT) common/unique feature maps as supervision. Existing methods have to empirically embed multiple metrics into the loss function [46, 1, 2, 5, 6, 10]. It is well known that each image pixel is a coupling of multiple visual attributes, such as edges, textures, contrast, light and contour. Obviously, it is almost impossible to incorporate all metrics that describe specific visual attributes into the loss, which makes an intrinsic difficulty. More importantly, we have no prior knowledge of whether these metrics are mutually conflicting, nor can we determine which combinations lead to better performance. This raises a central question: How can we construct an effective and reliable supervisory objective?

![](images/ee934afb04b6ed04f5c56d2df39025b3a4cc61732230eeddc4db7b8f8f384268.jpg)  
Figure 1: Existing decomposition paradigms vs Ours. We introduce signal-level decomposition and image-level reconstruction pretext tasks for accurate decomposition and high-quality fusion.

To address this, we propose a 1D signal-level self-supervised feature decomposition paradigm. Our core insight is to reformulate feature decomposition from an unclear supervision problem requiring simultaneous optimization of multiple visual attributes into an integral-driven mathematical problem. This reformulation shifts the focus from architectural design to the construction of a clear supervisory objective for feature decomposition. Accordingly, we design a tailored loss. In this formulation, one only needs to optimize the integral area between the common and source signals. It avoids conflicts from metric-mixing losses and enables disentanglement of coupled information. Under this integral-driven objective, the decomposer achieves more stable and accurate optimization.

Holistically, our model follows a two-stage self-supervised learning (SSL) framework [27]. In Stage I, we design dual pretext tasks at both the signal- and image-level. For decomposition, motivated by the reformulated problem, we design a novel signal-level pretext task. Features from source images are transformed into 1D signals and decomposed into low- and high-frequency components via DWT [24]. The network then acts as an integral solver, estimating common features from paired low-frequency bands and extracting unique features from high-frequency bands of each modality. Paired source signals directly provide pseudo labels for strict mathematical constraints. For reconstruction, we return to the image-level, where original images supervise spatial structure preservation. In Stage II, we first fuse the unique feature signals, then combine them with the common signals, and finally apply IDWT to reconstruct fused images with complete information. Experimental results demonstrate that our method achieves state-of-the-art (SOTA) performance, validating the proposed paradigm.

## Our contributions are summarized as follows:

• We propose a novel self-supervised feature decomposition paradigm at 1D signal-level to enable accurate feature decomposition.

• We reformulate the feature decomposition problem from an image-level supervision problem with unclear objectives into a signal-level integral optimization problem with an explicit mathematical objective. This clear optimization target reduces the ambiguity caused by empirical metric-mixture losses and enables more accurate feature decomposition.

• We design complementary dual pretext tasks at both levels. The signal-level task offers strong mathematical constraints for decomposition, while the image-level task ensures reliable spatial structures. They achieve both precise decomposition and faithful reconstruction.

## 2 Related Work

Feature Decomposition-based MMIF. Feature decomposition-based fusion methods decompose multimodal images into two complementary types of features: common features and unique features. This strategy not only promotes sufficient interaction and fusion of cross-modal complementary information, but also helps preserve the overall scene structure. Several methods adopt this idea. For instance, DIDFuse [48] designs a dedicated feature decomposition module to explicitly separate common and unique features in an end-to-end manner; FD-Fuse [5] imposes indirect constraints on the decomposition process via pretext tasks; CU-Net [6] formulates feature decomposition as a mathematical optimization problem and uses sparse convolutions to solve a hand-crafted objective function. Furthermore, external prior knowledge from large language models is used to guide the decomposition process. MTG-Fusion [39] leverages a large language model to generate multiple descriptive texts and uses their embeddings to guide visual feature decomposition. However, since GT common and unique features are unavailable, these decomposers still lack reliable supervision. We address this issue by reformulating feature decomposition over signalized features as an integraldriven optimization problem, which provides an explicit objective for accurate feature separation.

Self-Supervised Learning. Self-supervised learning constructs pretext tasks to generate supervision from unlabeled data, usually through pretraining and downstream adaptation [51]. Due to the lack of GT in MMIF, many fusion methods adopt SSL strategies. DeFusion [12] learns cross-modal semantic relations via masked image restoration. SMFuse [21] adopts a self-supervised scheme to generate accurate masks for multi-focus image fusion. Wang et al. use image super-resolution as a pretext task, treating high-resolution images as pseudo labels for low-resolution inputs [37]. EMMA [47] exploits image equivariance and generates supervision signals from geometrically transformed natural images. However, these methods are not specifically designed for the feature decomposition paradigm and cannot fundamentally resolve the absence of GT for common and unique features. Therefore, we design an SSL paradigm with dual pretext tasks. We transform the feature decomposition problem into an integral solving problem defined at the 1D signal-level, which naturally yields a pretext task with a clear mathematical objective, using 1D signals of the source images to provide direct supervision. At the image-level, we use the source images as supervision for a reconstruction task, thus providing comprehensive supervision for the entire feature decomposition process.

Signal Modeling-based Image Fusion. Recently, SigFusion [36] proposes a unified self-supervised paradigm from the signal-level perspective, integrating dataset synthesis and image fusion into one framework. However, it is fundamentally different from our work. SigFusion uses signallevel modeling mainly to learn modality-specific signal distributions and synthesize large-scale training data, aiming to alleviate the lack of realistic training pairs. Our method uses signalization to reformulate the decomposition problem itself, focusing on the lack of direct supervision for common and unique features and providing the decomposer with an explicit integral-driven objective.

## 3 Method

This section describes the workflow of our proposed model. As shown in Fig. 2, our model consists of an encoder $\zeta ,$ a signal-level feature decomposer $S D ( \cdot )$ , an MLP based feature fusion module δ and a decoder $\xi .$ Both the encoder and decoder are composed of Restormer blocks [46]. Following the self-supervised learning (SSL) paradigm, our model adopts a two-stage paradigm. Stage I performs signal-level decomposition and reconstruction pretraining to obtain accurate common and unique features. Stage II conducts cross-modal fusion on the decomposed unique features and combines them with the common features to generate the final fused image.

## 3.1 Problem Formulation and Modeling

In MMIF, mainstream methods based on feature decomposition paradigm typically follow a two-stage pipeline. Given a pair of source images $( I _ { A } , I _ { B } )$ , a decomposer $D ( \cdot )$ first extracts the common feature C and unique features $( { \cal U } _ { A } , { \cal U } _ { B } )$

$$
C , U _ { A } , U _ { B } = D ( I _ { A } , I _ { B } ) .\tag{1}
$$

Then, these features are fused to obtain fused image $I _ { f }$

$$
I _ { f } = F ( C , U _ { A } , U _ { B } ) .\tag{2}
$$

![](images/14b7df97d203432f206881a5c34bb5abd4c105ffb495eaafd04516533d939362.jpg)  
Figure 2: Schematic of the proposed model. It adopts a two-stage self-supervised paradigm. Stage I: feature decomposition via 1D signal-level and 2D image-level pretext tasks; Stage II: fuses the decomposed features to reconstruct the final fused image.

However, these methods suffer from a fundamental limitation in supervision. Because ground-truth common and unique feature maps are unavailable, the training objective is inevitably constructed in a handcrafted or empirical manner, typically by mixing multiple attribute metrics. Yet each pixel couples diverse visual attributes such as edges, textures and contours, making it impractical to cover all attributes with metrics. Consequently, the final objective is often incomplete and may even introduce conflicts, which prevents accurate and stable decomposition. Therefore, it is crucial to design a decomposition paradigm with a clear and reliable supervisory objective.

Reformulating the Feature Decomposition Problem. To address these issues, we propose a signallevel feature decomposition paradigm that reformulates the decomposition problem from an ill-posed 2D image-level problem into a well-defined optimization problem in the 1D signal domain.

Given a source image pair $( I _ { A } , I _ { B } )$ and their 1D signal representations $( s _ { A } ( l ) , s _ { B } ( l ) )$ , we optimize the common signal $c _ { s } ( l )$ by minimizing the accumulated integral area induced by its deviation from the two original source signals:

$$
c _ { s } ^ { * } ( l ) = \arg \operatorname* { m i n } _ { c _ { s } } \int { \left( c _ { s } ( l ) - s _ { A } ( l ) \right) ^ { 2 } } d l + \int { \left( c _ { s } ( l ) - s _ { B } ( l ) \right) ^ { 2 } } d l .\tag{3}
$$

In this formulation, the common signal is encouraged to lie close to both source signals in the integral sense, so that it can capture the component consistently supported by the two modalities. Compared

with image-level metric mixing, this objective provides a more direct criterion for common feature estimation and reduces the influence of local fluctuations or modality-specific disturbances.

## 3.2 Stage I: Self-supervised Feature Decomposition with Dual Pretext Tasks

In Stage I, we design two complementary pretext tasks for self-supervised common and unique feature extraction. The signal-level task provides an explicit objective for feature decomposition, while the image-level task preserves spatial structures through reconstruction. Procedure is as follows.

## Feature Decomposition based on the Signal-level Pretext Task.

Given a pair of source images $I _ { A } , I _ { B }$ , we first extract high-level feature maps via an encoder ζ:

$$
F _ { A } , F _ { B } = \zeta ( I _ { A } ) , \zeta ( I _ { B } ) ,\tag{4}
$$

These features are then transformed into 1D signal-level representations:

$$
s _ { A } ( l ) , s _ { B } ( l ) = \mathcal { H } \mathopen { } \mathclose \bgroup \left( F _ { A } , F _ { B } \aftergroup \egroup \right) ,\tag{5}
$$

where H denotes the transformation with a $1 \times 1$ convolution followed by flattening and l indexes the 1D positions corresponding to spatial locations. To enable more fine-grained feature decomposition, we decompose each signal into low- and high-frequency sub-bands via DWT 1D:

$$
L _ { m } ( l ) , \{ H _ { m } ^ { i } ( l ) \} _ { i = 1 } ^ { N } = \mathrm { D W T } \big ( s _ { m } ( l ) \big ) , \quad m \in \{ A , B \} .\tag{6}
$$

Low-frequency components mainly encode global structure shared across modalities, while highfrequency components capture unique details such as textures and edges.

Based on these characteristics, we design a signal decomposer $S D ( \cdot )$ with two paths: the commonpath takes the low-frequency pair $( L _ { A } ( \bar { l } ) , L _ { B } \bar { ( } l ) )$ as input and estimates the common signal $c _ { s } ( l )$

$$
c _ { s } ( l ) = S D \big ( L _ { A } ( l ) , L _ { B } ( l ) \big ) .\tag{7}
$$

The unique-path takes high-frequency sub-bands of each modality to estimate unique signals $u _ { s ^ { A } } ( l )$ and $u _ { s ^ { B } } ( l )$

$$
u _ { s ^ { A } } ( l ) , u _ { s ^ { B } } ( l ) = S D \big ( \{ H _ { A } ^ { i } ( l ) \} _ { i = 1 } ^ { N } , \{ H _ { B } ^ { i } ( l ) \} _ { i = 1 } ^ { N } \big ) .\tag{8}
$$

To optimize the common feature, we adopt a tailored integral-driven loss ${ \mathcal { L } } _ { \mathrm { d e c } } .$

$$
\mathcal { L } _ { \mathrm { d e c } } = \int { \left( c _ { s } ( l ) - s _ { A } ( l ) \right) ^ { 2 } d l } + \int { \left( c _ { s } ( l ) - s _ { B } ( l ) \right) ^ { 2 } d l } ,\tag{9}
$$

where the 1D signals $s _ { A } ( l )$ and $s _ { B } ( l )$ serve as pseudo labels. This loss enforces global consistency along the entire signal, suppressing local noise and guiding accurate estimation of $c _ { s } ( l )$

## Image Reconstruction based on the Image-level Pretext Task.

The reconstruction process mirrors the decomposition and uses the decomposed features to recover the original images, providing an image-level constraint.

First, $c _ { s } ( l )$ and $u _ { s ^ { m } } ( l )$ are fed into the inverse DWT, denoted as $\mathrm { D W T } ^ { T }$ , to reconstruct the 1D signal:

$$
\begin{array} { r } { \hat { s } _ { m } ( l ) = \mathrm { D W T } ^ { T } \big ( c _ { s } ( l ) , u _ { s ^ { m } } ( l ) \big ) , \quad m \in \{ A , B \} . } \end{array}\tag{10}
$$

Then, the reconstructed signals are mapped back to 2D feature maps by the inverse transformation $\mathcal { H } ^ { T }$ and reconstructed to the source images by a decoder $\xi \colon$

$$
\hat { I } _ { m } = \xi \big ( \mathcal { H } ^ { T } ( \hat { s } _ { m } ( l ) ) \big ) , \quad m \in \{ A , B \} .\tag{11}
$$

In this reconstruction process, we employ $I _ { A }$ and $I _ { B }$ act as pseudo labels to preserve spatial structure:

$$
\mathcal { L } _ { \mathrm { r e c } } = \mathrm { M S E } ( \hat { I } _ { A } , I _ { A } ) + \mathrm { M S E } ( \hat { I } _ { B } , I _ { B } ) ,\tag{12}
$$

Loss Function of Stage I. The overall loss function of Stage I is defined as Eq. 13:

$$
{ \mathcal { L } } _ { \mathrm { s t a g e 1 } } = \alpha { \mathcal { L } } _ { \mathrm { d e c } } + \beta { \mathcal { L } } _ { \mathrm { r e c } } ,\tag{13}
$$

where $\alpha$ and $\beta$ are hyperparameters. Through these dual-pretext tasks, Stage I provides a clear optimization target for decomposition at 1D signal-level and enforces structural correctness in 2D image-level, producing accurate common and unique features for the subsequent fusion stage.

![](images/c77db7c16222ab50658c037abf62bae2fae3587973841ad4fe3fc190357ca2b9.jpg)  
Figure 3: Qualitative comparison of various fusion models.

## 3.3 Stage II: Feature Decomposition-based Image Fusion

In Stage II, our goal is to generate a high-quality fused image based on the accurate decomposed features obtained from Stage I. Training and inference processes are as follows.

Training. We freeze the encoder ζ and the signal decomposer $_ { S D }$ pretrained in Stage I. Given a source image pair $( I _ { A } , I _ { B } )$ , we obtain the common signal $c _ { s } ( l )$ and unique signals $u _ { s ^ { A } } ( l )$ and $u _ { s ^ { B } } ( l )$ via the Stage I modules. We then fuse the unique features via an MLP-based feature fusion module δ:

$$
u _ { \mathrm { f } } ( l ) = \delta \big ( u _ { s ^ { A } } ( l ) , u _ { s ^ { B } } ( l ) \big ) ,\tag{14}
$$

Next, the fused specific signal $u _ { \mathrm { f } } ( l )$ and the common signal $c _ { s } ( l )$ are passed through the inverse DWT to obtain the fused 1D signal:

$$
f _ { s } ( l ) = \mathrm { D W T } ^ { \mathrm { T } } \bigl ( c _ { s } ( l ) , u _ { \mathrm { f } } ( l ) \bigr ) .\tag{15}
$$

This signal is mapped back to a 2D feature map by $\mathcal { H } ^ { T }$ and decoded to the fused image:

$$
I _ { f } = \xi ( \mathcal { H } ^ { T } \big ( f _ { s } ( l ) \big ) ) .\tag{16}
$$

Loss Function of Stage II. Following mainstream practice, we use the source images as supervision to guide fusion. The Stage II loss is defined as:

$$
\begin{array} { r } { \mathcal { L } _ { \mathrm { s t a g e 2 } } = \gamma \mathcal { L } _ { \mathrm { S S I M } } ( I _ { f } ; I _ { A } , I _ { B } ) + \lambda \mathcal { L } _ { L 1 } ( I _ { f } ; I _ { A } , I _ { B } ) + \omega \mathcal { L } _ { \mathrm { G r a d } } ( I _ { f } ; I _ { A } , I _ { B } ) , } \end{array}\tag{17}
$$

where $\mathcal { L } _ { \mathrm { S S I M } }$ is the SSIM-based structural similarity loss, $\mathcal { L } _ { L 1 }$ is the spatial $L _ { 1 }$ loss, ${ \mathcal { L } } _ { \mathrm { G r a d } }$ is the gradient-domain $L _ { 1 }$ loss, and $\gamma , \lambda ,$ ω are hyperparameters.

Inference. During inference, only the Stage II fusion pipeline is executed. Given a registered source image pair $( I _ { A } , I _ { B } )$ , the model $\mathcal { G }$ directly produces the fused image $I _ { f }$ in a single forward pass:

$$
I _ { f } = { \mathcal { G } } ( I _ { A } , I _ { B } ) .\tag{18}
$$

## 4 Experimental Results

## 4.1 Setting

We adopt the AdamW optimizer with an initial learning rate of $3 \times 1 0 ^ { - 4 }$ , batch size $^ { 4 , }$ and no weight decay. The learning rate is halved every 20 epochs. All experiments are conducted on Ubuntu 22.04 with an Intel i9-14900K CPU, RTX 4090 GPU, 64GB RAM, and PyTorch 2.3.0. In Stage I, we set α = 1, β = 2 for 200 epochs. In Stage $\operatorname { I I } , \gamma , \lambda$ and ω are set as 1, 1 and 15.6 for 200 epochs, respectively.

Table 1: Quantitative comparison of various fusion models on the VIF and MIF tasks. Best is bold; second best is underlined.
<table><tr><td colspan="2">VIF Task</td><td colspan="4"></td><td colspan="4">M⁵FD</td><td colspan="3"></td><td colspan="4">MSRS</td><td colspan="7">TNO</td></tr><tr><td>Methods CDDFuse</td><td>Pub/Year</td><td>QM11</td><td>QNICE</td><td>QP↑ 0.4554</td><td>0.4746 QcB↑</td><td>MI↑</td><td>VIF</td><td>0.8698 QY↑</td><td>QMI↑ 0.7576</td><td>QNICE1 0.8229</td><td>QP↑ 0.5399</td><td>QcB1</td><td>MI↑</td><td>0.5106 VIF</td><td>QY↑</td><td>QM11 0.4654</td><td>QNICE1</td><td>QP↑</td><td>QcB1</td><td>MI↑</td><td>VIF</td><td>QY↑</td></tr><tr><td></td><td>CVPR 23</td><td>0.5741</td><td>0.8123</td><td></td><td></td><td>3.8730</td><td>0.4410</td><td></td><td></td><td></td><td></td><td>0.5669</td><td>4.9089</td><td></td><td>0.8267</td><td></td><td>0.8090 0.8062</td><td>0.3789</td><td>0.4504</td><td>3.0691</td><td>0.4428</td><td>0.7874</td></tr><tr><td>LRRNet</td><td>TPAMI 23</td><td>0.6353</td><td>0.8073</td><td>0.3709</td><td>0.4346</td><td>2.8124</td><td>0.3550</td><td>0.7209</td><td>0.6608</td><td>0.8082</td><td>0.3372</td><td>0.3928</td><td>2.8662</td><td>0.2933</td><td>0.5067</td><td>0.3686</td><td></td><td>0.2303</td><td>0.4831</td><td>2.3868</td><td>0.3685 0.7011</td></tr><tr><td>EMMA</td><td>CVPR 24</td><td>0.5564</td><td>0.8118</td><td>0.4602</td><td>0.4775</td><td>3.7712</td><td>0.4690</td><td>0.7979</td><td>0.6413</td><td>0.8161</td><td>0.4961</td><td>0.5432</td><td>4.1637</td><td>0.5230</td><td>0.7840</td><td>0.4356</td><td>0.8081</td><td>0.3550</td><td>0.5112</td><td>2.9089 0.4847</td><td>0.8072</td></tr><tr><td>TC-MoA</td><td>CVPR 24</td><td>0.5011</td><td>0.8096</td><td>0.5133</td><td>0.4934</td><td>3.3330</td><td>0.4603</td><td>0.8538</td><td>0.7446</td><td>0.8119</td><td>0.4833</td><td>0.5558</td><td>3.4925</td><td>0.4924</td><td>0.8825</td><td>0.4068</td><td>0.8071</td><td>0.4020 0.5124</td><td>2.5747</td><td>0.4312</td><td>0.8664</td></tr><tr><td>Text-Difuse</td><td>NeurIPS 24</td><td>0.4262</td><td>0.8057</td><td>0.4323</td><td>0.3398</td><td>2.0535</td><td>0.1574</td><td>0.2566</td><td>0.7252</td><td>0.8032</td><td>0.5378 0.3660</td><td>1.3095</td><td></td><td>0.1570 0.2122</td><td>0.2977</td><td>0.8047</td><td>0.1167</td><td>0.4296</td><td>1.8773</td><td>0.2638</td><td>0.4410</td></tr><tr><td>DCEvo</td><td>CVPR 25</td><td>0.6085</td><td>0.8135</td><td>0.4692</td><td>0.4801</td><td>4.0074</td><td>0.4485</td><td>0.8987</td><td>0.6246</td><td>0.8170</td><td>0.4967 0.5627</td><td>4.0493</td><td></td><td>0.5107 0.8650</td><td>0.5739</td><td>0.8125</td><td>0.4280</td><td>0.5089</td><td>3.6054</td><td>0.4605 0.3963</td><td>0.9060</td></tr><tr><td>SAGE</td><td>CVPR 25</td><td>0.4566</td><td>0.8080</td><td>0.4338</td><td>0.4647</td><td>2.8508</td><td>0.4257</td><td>0.8324</td><td>0.5153</td><td>0.8101</td><td>0.4348</td><td>0.4775</td><td>2.9895</td><td>0.3455 0.7732</td><td>0.3751</td><td>0.8060</td><td>0.3323</td><td>0.4715</td><td>2.2033</td><td></td><td>0.7855</td></tr><tr><td>TD-Fusion</td><td>CVPR 25</td><td>0.4494</td><td>0.8081</td><td>0.4987</td><td>0.5108</td><td>3.0290</td><td>0.4629</td><td>0.8519</td><td>0.7462</td><td>0.8082</td><td>0.4170</td><td>0.5268</td><td>2.8434</td><td>0.4407</td><td>0.7922 0.3640</td><td>0.8062</td><td>0.3512</td><td>0.5288</td><td>2.4507</td><td>0.4317</td><td>0.8183</td></tr><tr><td>Omni-fuse</td><td>TPAMI 25</td><td>0.4882</td><td>0.8092</td><td>0.2770</td><td>0.4776</td><td>3.2075</td><td>0.4432</td><td>0.6543</td><td>0.7208</td><td>0.8079</td><td>0.5174</td><td>0.4211</td><td>2.6641</td><td>0.4121</td><td>0.4927 0.3814</td><td>0.8062</td><td>0.1908</td><td>0.4374</td><td>2.2871</td><td>0.3933</td><td>0.6406</td></tr><tr><td>C2RF</td><td>IJCV 25</td><td>0.4082 0.7123</td><td>0.8075</td><td>0.4989</td><td>0.3882</td><td>2.4979</td><td>0.2241</td><td>0.4293</td><td>0.6928</td><td>0.8039</td><td>0.4922</td><td>0.4246</td><td>1.7292</td><td>0.1589</td><td>0.3892 0.3535</td><td>0.8059</td><td>0.1847</td><td>0.4362</td><td>2.0637</td><td>0.2975</td><td>0.6155</td></tr><tr><td>SigFusion Ours</td><td>AAAI 26</td><td></td><td>0.8186</td><td>0.4665</td><td>0.4796</td><td>4.5751</td><td>0.4478</td><td>0.9239</td><td>0.8034</td><td>0.8256</td><td>0.5492 0.5643</td><td></td><td>4.9123</td><td>0.4432</td><td>0.8772 0.5296</td><td>0.8114</td><td>0.3936</td><td>0.5064</td><td>3.2423</td><td>0.4455</td><td>0.8911</td></tr><tr><td></td><td></td><td>0.6608</td><td>0.8209</td><td>0.5735</td><td>0.4998</td><td>4.1050</td><td>0.4694</td><td>0.8988</td><td>0.8405</td><td>0.8290</td><td>0.5483</td><td>0.5798</td><td>5.4547</td><td>0.5253</td><td>0.8936 0.5711</td><td>0.8128</td><td>0.4318</td><td>0.5346</td><td>3.8015</td><td>0.4768</td><td>0.8450</td></tr><tr><td>MIF Methods</td><td>Task</td><td></td><td></td><td></td><td>MRI-CT</td><td></td><td></td><td></td><td></td><td></td><td>MRI-PET</td><td></td><td></td><td></td><td></td><td></td><td></td><td>MRI-SPECT</td><td></td><td></td><td></td></tr><tr><td>CDDFuse</td><td>Pub/Year CVPR 23</td><td>QM11</td><td>QNICE↑</td><td>QP1</td><td>QCB↑</td><td>MI1</td><td>VIFp↑</td><td>QY1</td><td>QMI↑</td><td>QNICE QP1</td><td>QCB1</td><td>MI↑</td><td>VIFp</td><td>QY</td><td>QM1</td><td>QNICE1</td><td>QP↑</td><td>QcB↑</td><td>MI↑</td><td>VIFp</td><td>QY↑</td></tr><tr><td></td><td>TPAMI 23</td><td>0.8403 0.6273</td><td>0.8098 0.8066</td><td>0.4122 0.2073</td><td>0.6911</td><td>3.7522</td><td>0.3735</td><td>0.9047 0.7856</td><td>0.8064</td><td>0.4448</td><td>0.7213</td><td>2.7359</td><td>0.3872</td><td>0.9125</td><td>0.8778</td><td>0.8068</td><td>0.6289</td><td>0.7305</td><td>2.7296</td><td>0.4480</td><td>0.2763 0.9118</td></tr><tr><td>LRRNet</td><td>CVPR 24</td><td>0.7032</td><td>0.8078</td><td>0.3160</td><td>0.2205 0.5627</td><td>2.7570</td><td>0.2295</td><td>0.3285 0.6427</td><td>0.8051</td><td>0.3177 0.3809</td><td>0.2129</td><td>3.6627 2.4481</td><td>0.2807 0.3577</td><td>0.3122 0.7433</td><td>0.7355</td><td>0.8056</td><td>0.3580</td><td>0.1839</td><td>3.8263</td><td>0.2607</td><td>0.6633</td></tr><tr><td>EMMA</td><td>CVPR 24</td><td>0.7279</td><td>0.8080</td><td>0.3686</td><td>0.5838</td><td>3.2790 3.3169</td><td>0.3575 0.3751</td><td>0.7590 0.7782</td><td>0.6648</td><td>0.8055 0.4692</td><td>0.5606 0.6129</td><td>2.4498</td><td>0.3900</td><td>0.8394</td><td>0.7086 0.7397</td><td>0.8057 0.8059</td><td>0.3368</td><td>0.4834 0.5426</td><td>2.3611 2.3331</td><td>0.3660</td><td>0.7659</td></tr><tr><td>TC-MoA</td><td>NeurIPS 24</td><td>0.6456</td><td>0.8075</td><td>0.2643</td><td>0.3239</td><td>3.1594</td><td>0.3150</td><td>0.3891</td><td>0.6758 0.5235</td><td>0.8055 0.8051</td><td>0.2606</td><td>2.3310</td><td>0.2973</td><td>0.2630</td><td>0.5638</td><td>0.8053</td><td>0.5564 0.3649</td><td>0.1976</td><td>2.2253</td><td>0.4422 0.3261</td><td>0.2550</td></tr><tr><td>Text-Difuse CCF</td><td>NeurIPS 24</td><td>0.7361 0.6929</td><td>0.8079 0.8079</td><td>0.2825 0.3229</td><td>0.3479</td><td>3.2844</td><td>0.2968 0.3435</td><td>0.4329</td><td>0.6439</td><td>0.3118 0.8050 0.2733</td><td>0.2637</td><td>2.4926</td></table>

Table 2: Quantitative results of ablation experiments on VIF and MIF tasks.
<table><tr><td colspan="2"></td><td colspan="8">Task</td><td colspan="8">Task</td></tr><tr><td>Task | Case </td><td>Configurations w/o signal-level Dec.</td><td>|QM1↑ 0.6282</td><td>QNICE↑ 0.8181</td><td>0.5350</td><td>QP↑ QCB↑</td><td>0.4716</td><td>MI↑ 3.8631</td><td>VIFp↑ 0.4391</td><td>QY↑ Task | Case 0.8729</td><td></td><td>Configurations</td><td>| QM1↑ 0.9012</td><td>QNICE↑ 0.8071</td><td>QP↑ 0.3627</td><td>QcB↑ 0.6715</td><td>MI↑ 3.6129</td><td>VIFp↑ 0.3689</td><td>QY↑ 0.9127</td></tr><tr><td rowspan="8">I VIF</td><td>w/o DWT</td><td>0.6425</td><td>0.8192</td><td></td><td>0.5551 0.4860</td><td></td><td>3.9812</td><td>0.4534</td><td>0.8847</td><td rowspan="5">I</td><td rowspan="5">w/o signal-level Dec. w/o DWT</td><td colspan="9">0.9185</td></tr><tr><td>w/ IDWT ← learnable Rec.</td><td>0.6531 0.6380</td><td>0.8198</td><td>0.5631</td><td>0.4925</td><td>4.0200</td><td>0.8911</td><td rowspan="5"></td><td rowspan="5">w/ IDWT ← learnable Rec.</td><td colspan="9">0.9253</td></tr><tr><td>w/ signal- ← image-level Dec.</td><td>0.8185</td><td>0.5460</td><td>0.4793</td><td>3.9022</td><td>0.4594 0.4468</td><td>0.8781</td><td rowspan="3">w/ signal- ← image-level Dec.</td><td>0.9108</td><td rowspan="3">0.8096 0.8086</td><td>0.3928 0.3755 0.4034</td><td rowspan="3">0.6905 0.6810</td><td rowspan="3">3.9025 3.7420 4.0184</td><td rowspan="3">0.3954 0.3798 0.4034</td><td rowspan="3">0.9361 0.9275 0.9439</td></tr><tr><td>Default setups (ours)</td><td>0.6608</td><td>0.8209</td><td>0.4998</td><td>4.1050</td><td>0.4694</td><td>Default setups (ours)</td><td>0.9369</td><td>0.8108</td><td>0.7008</td></tr><tr><td>w/o signal-level pretext task</td><td></td><td></td><td>0.5735</td><td></td><td></td><td>0.8988 MIF</td><td></td><td></td><td></td></tr><tr><td>w/o image-level pretext task</td><td>0.6315 0.6495</td><td>0.8188 0.8195</td><td>0.5420 0.5525</td><td>0.4868 0.4820</td><td>3.9033 3.9800</td><td>0.4512 0.4553</td><td>0.8799 0.8850</td><td rowspan="3">w/o signal-level pretext task w/o image-level pretext task</td><td rowspan="3"></td><td>0.9144 0.9228 0.9202</td><td>0.8093 0.8294 0.8091</td><td rowspan="3">0.3786 0.3866 0.3814</td><td rowspan="3">0.6869 0.6765 0.6851</td><td rowspan="3">3.7562 3.8041 3.7820</td><td rowspan="3">0.3831 0.3912 0.3886</td><td rowspan="3">0.9293 0.9335 0.9318 0.4034 0.9439</td></tr><tr><td>ⅡI</td><td>w/ Ldec ← L1 Default setups (ours)</td><td>0.6463 0.6608</td><td>0.8195 0.8209</td><td>0.5482 0.5735</td><td>0.4817 0.4998</td><td>3.9521 0.4520</td><td>0.8832 0.8988</td><td>w/ Ldec ← L1 Default setups (ours)</td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td><td>4.1050</td><td>0.4694</td><td></td><td></td><td></td><td>0.9369</td><td>0.8108</td></tr></table>

## 4.2 Datasets and Metrics

Datasets. For VIF, the training set consists of 400 image pairs from the MSRS[22]. Testing is conducted on M<sup>3</sup>FD (100 pairs), TNO (25 pairs), and MSRS (361 pairs). For MIF, we construct the training set from 453 pairs of registered medical images from the Harvard Medical website [32]. For testing, we select 21 MRI–CT pairs [46], 42 MRI–PET pairs [46], and 73 MRI–SPECT pairs [46]. Metrics. We adopt widely used image quality metrics following mainstream practice, including Q<sub>MI</sub> [25], Q [35], Q [42], Q [38], MI [25], $V I F _ { P }$ [26], and $Q _ { Y }$ [43].

## 4.3 Comparison with SOTA

Comparison Algorithms. For VIF task, we select CDDFuse [46], LRRNet [9], EMMA [47], TC-MoA [50], Text-Difuse [44], DCEvo [17], SAGE [40], TDFusion [1], Omni-Fuse [45], C2RF [31] and SigFusion[36]. For MIF task, we select CDDFuse, LRRNet, EMMA, TC-MoA, Text-Difuse, CCF [4], BSA-Fusion [8], Mask-Difuser [29], MTG-Fusion [39], C2RF [31] and SigFusion[36]. Qualitative Comparison. Fig. 3 shows the qualitative comparison results of 12 fusion models on four test sets. More results are shown in Appendix D.1. For VIF task, the first 2 groups show results on TNO [34] and M<sup>3</sup>FD [14]. Benefiting from the 1D signal-level decomposition, our fused images better preserve both the common structure and modality-specific cues. For MIF task, the last two groups are the MRI-CT and MRI-PET datasets results. Our dual-pretext task design provides accurate and comprehensive supervision for both levels, yielding fused images with improved color fidelity. Quantitative Comparison. Table 1 reports the results of various models on seven metrics across VIF and MIF. Our model achieves superior results in most metrics. We attribute these gains to the signal-level decomposer and dual-pretext design, which provide clear supervision for the common component and enhance cross-modal interaction, yielding a high-quality representation for fusion.

## 4.4 Ablation Study

To comprehensively verify the effectiveness of each key component, we conduct a systematic ablation study. More results are in Appendix F.

Effect of Signal-Level Decomposer. Our signal-level decomposer transfers the decomposition problem into 1D signal-level. To assess its impact, we conduct the following ablations: (1) removing the signal-level decomposer(Dec.); (2) removing frequency decomposition; (3) replacing IDWT with a learnable reconstructor(Rec.); (4) substituting an image-level decomposer; (5) default setup. The results are presented in Table 2 Case I, which demonstrate the effectiveness of our paradigm.

![](images/bb3c065525f852c8b1acc4df775f310f85ccb2912a3ddd77d373313bb595ac91.jpg)

Figure 4: Visualization of intermediate 1D signals. The learned common signal $c _ { s } ( l )$ follows the shared trend of the source signals $s _ { A } ( l )$ and $s _ { B } ( l )$  
![](images/1b8eec539b90504b621ed52ee0feb93a3b179f17735943606f39623d827b2100.jpg)  
Figure 5: Qualitative results on downstream tasks. Left: object detection performance using fused images generated by different fusion methods. Right: medical image segmentation. Our fused images provide more accurate results.

Effect of Dual Pretext Tasks. To evaluate the dual pretext tasks at both levels, we perform the following ablations: (1) removing the signal-level pretext task; (2) removing the image-level pretext task; (3) replacing the proposed $L _ { d e c }$ with vanilla $\bar { L } _ { 1 }$ only; (4) default configuration. The results are shown in Table 2 Case II. Apparently, removing either pretext task degrades performance, highlighting the necessity to provide decomposer a clear supervision.

Sensitivity to Key Hyperparameters. We analyze the sensitivity of key hyperparameters. Full analysis and ablation results are provided in Appendix F.

## 4.5 Signal-Level Feature Visualizations

To further validate the proposed signal-level formulation, we visualize the source signals $s _ { A } ( l )$ and $s _ { B } ( l )$ and the learned common signal $c _ { s } ( l )$ in Fig. 4. The red and green curves denote the two source signals, while the blue curve denotes the estimated common signal. It can be observed that $c _ { s } ( l )$ generally lies between $s _ { A } ( l )$ and $s _ { B } ( l )$ and follows their shared trend, indicating that it captures the component consistently shared by both modalities.

## 4.6 Downstream Tasks

Object Detection for VIF Task. We evaluate detection on fused images produced by various VIF models. A pre-trained detector (YOLOv12) [33] is used to ensure fairness. Qualitative results are given in Fig. 5 (left) and quantitative results see Appendix Table 4. Compared with other fusion methods, our model achieves stronger detection performance, suggesting that it better preserves taskrelevant complementary cues and provides more favorable inputs for downstream object detection.

![](images/d05f8f2c114fc91801632510a4f2444d630e986ab2dcb2f53f044f341e2d1162.jpg)

Figure 6: Broader impact of our paradigm. Refined variants (denoted by “\*”) obtain fused images with clearer structures and more salient target details, as highlighted by the red and yellow circles.  
![](images/9900d3242148cd8ffaa1e516ebdbe03c522ae0c3775c26f8f7fb0506a028f496.jpg)  
Figure 7: Qualitative results on multi-focus image fusion, showing that the proposed signal-level decomposition paradigm can be naturally transferred to other image fusion tasks.

Medical Image Segmentation for MIF Task. We evaluate medical image segmentation using UniverSeg [3]. Qualitative results are given in Fig. 5 (right) and quantitative results see Appendix Table 5. Compared with single-modality inputs, fused images provide more informative representations for segmentation, leading to predictions closer to the ground truth. Our method also outperforms other fusion models, indicating that the proposed decomposition strategy helps preserve common anatomical or lesion structures while retaining modality-specific details for boundary delineation.

## 4.7 Broader Impact

Refining decomposition-based MMIF paradigms. Our signal-level decomposition paradigm provides a clearer objective for common and unique feature separation and can be plugged into existing decomposition-based MMIF models. To validate this, we insert it into DIDFuse, CDDFuse, FD-Fuse [5], and C2RF without changing their default settings. As shown in Fig. 6, the enhanced models preserve more complete structures and richer details. Quantitative results in Appendix Table 6 further confirm its effectiveness in improving existing decomposition-based MMIF methods.

Extension to other image fusion tasks. Our paradigm is not limited to MMIF. We further select multi-focus image fusion as a representative task, where common scene structures and source-specific focused details need to be separated. As shown in Fig. 7, the proposed paradigm can also produce clearer focus boundaries and more faithful structural details on this task. Detailed quantitative results are provided in the Appendix Table 7, suggesting its applicability to broader image fusion problems.

## 4.8 Model Complexity and Overhead Analysis

We analyze the computational overhead including parameters, GFLOPs, and inference time. Results are in Appendix Table 9. Our model remains computationally competitive. Although the DWT operation introduces a slight additional overhead, the overall complexity is still acceptable.

## 5 Conclusion

In this paper, we presented a signal-level feature decomposition paradigm that reformulates feature decomposition into an integral-driven optimization at the 1D signal-level. This reformulation yields a matched loss, providing a clear learning objective. Based on this, our two-stage SSL pipeline first performs a decomposition and reconstruction in Stage I, with dual pretext tasks at both levels supervising the entire decomposition and reconstruction process. Stage II then carries out fusion. Extensive experiments show advanced performance, validating the effectiveness of our paradigm.

## Acknowledgments

This work was supported by the National Natural Science Foundation of China [No.62401097, 62601484]; Fundamental Research Funds for Central Universities, Dalian Minzu University [No.0854- 53]; Liaoning Province Applied Basic Research Program [2026JH2/101300163]; Liaoning Province Science and Technology Joint Plan (2024JH2/102600113).

## References

[1] H. Bai, J. Zhang, Z. Zhao, Y. Wu, L. Deng, Y. Cui, T. Feng, and S. Xu. Task-driven image fusion with learnable fusion loss. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 7457–7468, 2025.

[2] H. Bai, Z. Zhao, J. Zhang, Y. Wu, L. Deng, Y. Cui, B. Jiang, and S. Xu. Refusion: Learning image fusion from reconstruction with learnable loss via meta-learning. International Journal of Computer Vision, 133(5):2547–2567, 2025.

[3] V. I. Butoi\*, J. J. G. Ortiz\*, T. Ma, M. R. Sabuncu, J. Guttag, and A. V. Dalca. Universeg: Universal medical image segmentation. International Conference on Computer Vision, 2023.

[4] B. Cao, X. Xu, P. Zhu, Q. Wang, and Q. Hu. Conditional controllable image fusion. Advances in Neural Information Processing Systems, 37:120311–120335, 2024.

[5] M. Cheng, H. Huang, X. Liu, H. Mo, S. Wu, and X. Zhao. Fdfuse: Infrared and visible image fusion based on feature decomposition. IEEE Transactions on Instrumentation and Measurement, 2025.

[6] X. Deng and P. L. Dragotti. Deep convolutional neural network for multi-modal image restoration and fusion. In IEEE Transactions on Pattern Analysis and Machine Intelligence (PAMI), 2020.

[7] S. Karim, G. Tong, J. Li, A. Qadir, U. Farooq, and Y. Yu. Current advances and future perspectives of image fusion: A comprehensive review. Information Fusion, 90:185–217, 2023.

[8] H. Li, D. Su, Q. Cai, and Y. Zhang. Bsafusion: A bidirectional stepwise feature alignment network for unaligned medical image fusion. In Proceedings of the AAAI Conference on Artificial Intelligence, volume 39, pages 4725–4733, 2025.

[9] H. Li, T. Xu, X.-J. Wu, J. Lu, and J. Kittler. Lrrnet: A novel representation learning guided fusion network for infrared and visible images. IEEE transactions on pattern analysis and machine intelligence, 45(9):11040–11052, 2023.

[10] W. Li, B. Li, H. Song, P. Wang, and Z. Wang. Infrared and visible image fusion via iterative feature decomposition and deep balanced fusion. Pattern Recognition, 174:113022, 2026.

[11] X. Li, Z. Wang, Y. Zou, Z. Chen, J. Ma, Z. Jiang, L. Ma, and J. Liu. Difiisr: A diffusion model with gradient guidance for infrared image super-resolution. In Proceedings ofthe Computer Vision and Pattern Recognition Conference, pages 7534–7544, 2025.

[12] P. Liang, J. Jiang, X. Liu, and J. Ma. Fusion from decomposition: A self-supervised decomposition approach for image fusion. In European conference on computer vision, pages 719–735. Springer, 2022.

[13] T.-Y. Lin, M. Maire, S. Belongie, J. Hays, P. Perona, D. Ramanan, P. Dollár, and C. L. Zitnick. Microsoft coco: Common objects in context. In European Conference on Computer Vision (ECCV), pages 740–755, 2014.

[14] J. Liu, X. Fan, Z. Huang, G. Wu, R. Liu, W. Zhong, and Z. Luo. Target-aware dual adversarial learning and a multi-scenario multi-modality benchmark to fuse infrared and visible for object detection. In Proceedings ofthe IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 5802–5811, 2022.

[15] J. Liu, X. Li, Z. Wang, Z. Jiang, W. Zhong, W. Fan, and B. Xu. Promptfusion: Harmonized semantic prompt learning for infrared and visible image fusion. IEEE/CAA Journal ofAutomatica Sinica, 2024.

[16] J. Liu, G. Wu, Z. Liu, D. Wang, Z. Jiang, L. Ma, W. Zhong, X. Fan, and R. Liu. Infrared and visible image fusion: From data compatibility to task adaption. IEEE Transactions on Pattern Analysis and Machine Intelligence, 47(4):2349–2369, 2025.

[17] J. Liu, B. Zhang, Q. Mei, X. Li, Y. Zou, Z. Jiang, L. Ma, R. Liu, and X. Fan. Dcevo: Discriminative cross-dimensional evolutionary learning for infrared and visible image fusion. In Proceedings of the Computer Vision and Pattern Recognition Conference, pages 2226–2235, 2025.

[18] R. Liu, Z. Liu, J. Liu, X. Fan, and Z. Luo. A task-guided, implicitly-searched and meta-initialized deep model for image fusion. IEEE Transactions on Pattern Analysis and Machine Intelligence, 46(10):6594– 6609, 2024.

[19] Y. Liu, Y. Shi, F. Mu, J. Cheng, C. Li, and X. Chen. Multimodal mri volumetric data fusion with convolutional neural networks. IEEE Transactions on Instrumentation and Measurement, 71:1–15, 2022.

[20] Z. Liu, J. Liu, G. Wu, L. Ma, X. Fan, and R. Liu. Bi-level dynamic learning for jointly multi-modality image fusion and beyond. arXiv preprint arXiv:2305.06720, 2023.

[21] J. Ma, Z. Le, X. Tian, and J. Jiang. Smfuse: Multi-focus image fusion via self-supervised mask-optimization. IEEE Transactions on Computational Imaging, 7:309–320, 2021.

[22] J. Ma, L. Tang, F. Fan, J. Huang, X. Mei, and Y. Ma. Swinfusion: Cross-domain long-range learning for general image fusion via swin transformer. IEEE/CAA Journal ofAutomatica Sinica, 9(7):1200–1217, 2022.

[23] F. Milletari, N. Navab, and S.-A. Ahmadi. V-net: Fully convolutional neural networks for volumetric medical image segmentation. In 2016 fourth international conference on 3D vision (3DV), pages 565–571. Ieee, 2016.

[24] M. Nechyba. Introduction to the discrete wavelet transform (dwt). University of Florida, February, 2004.

[25] G. Qu, D. Zhang, and P. Yan. Information measure for performance of image fusion. Electronics letters, 38(7):313–315, 2002.

[26] H. R. Sheikh and A. C. Bovik. Image information and visual quality. IEEE Transactions on image processing, 15(2):430–444, 2006.

[27] O. Siméoni, H. V. Vo, M. Seitzer, F. Baldassarre, M. Oquab, C. Jose, V. Khalidov, M. Szafraniec, S. Yi, M. Ramamonjisoa, F. Massa, D. Haziza, L. Wehrstedt, J. Wang, T. Darcet, T. Moutakanni, L. Sentana, C. Roberts, A. Vedaldi, J. Tolan, J. Brandt, C. Couprie, J. Mairal, H. Jégou, P. Labatut, and P. Bojanowski. DINOv3, 2025.

[28] L. Tang, Y. Deng, X. Yi, Q. Yan, Y. Yuan, and J. Ma. Drmf: Degradation-robust multi-modal image fusion via composable diffusion prior. In Proceedings of the ACM International Conference on Multimedia, pages 8546–8555, 2024.

[29] L. Tang, C. Li, and J. Ma. Mask-difuser: A masked diffusion model for unified unsupervised image fusion. IEEE Transactions on Pattern Analysis and Machine Intelligence, pages 1–18, 2025.

[30] L. Tang, Y. Wang, Z. Cai, J. Jiang, and J. Ma. Controlfusion: A controllable image fusion framework with language-vision degradation prompts. Advances in Neural Information Processing Systems, 2025.

[31] L. Tang, Q. Yan, X. Xiang, L. Fang, and J. Ma. C2rf: Bridging multi-modal image registration and fusion via commonality mining and contrastive learning. International Journal of Computer Vision, 133:5262–5280, 2025.

[32] W. Tang, F. He, Y. Liu, and Y. Duan. Matr: Multimodal medical image fusion via multiscale adaptive transformer. IEEE Transactions on Image Processing, 31:5134–5149, 2022.

[33] Y. Tian, Q. Ye, and D. Doermann. Yolov12: Attention-centric real-time object detectors. arXiv preprint arXiv:2502.12524, 2025.

[34] A. Toet. The tno multiband image data collection. Data in Brief, 15:249–251, 2017.

[35] Q. Wang, Y. Shen, and J. Q. Zhang. A nonlinear correlation measure for multivariable data set. Physica D: Nonlinear Phenomena, 200(3-4):287–295, 2005.

[36] Z. Wang, J. Feng, J. Wang, P. Wang, and H. Song. Sigfusion: Unified signal-level self-supervised learning paradigm for image fusion. Proceedings of the AAAI Conference on Artificial Intelligence, 40(12):10385–10393, Mar. 2026.

[37] Z. Wang, X. Li, H. Duan, and X. Zhang. A self-supervised residual feature learning model for multifocus image fusion. IEEE Transactions on Image Processing, 31:4527–4542, 2022.

[38] Z. Wang, J. Zhang, H. Song, M. Ge, J. Wang, and H. Duan. Highlight what you want: Weakly-supervised instance-level controllable infrared-visible image fusion. In Proceedings ofthe IEEE/CVF International Conference on Computer Vision, pages 12637–12647, 2025.

[39] Z. Wang, L. Zhao, J. Zhang, R. Song, H. Song, J. Meng, and S. Wang. Multi-text guidance is important: Multi-modality image fusion via large generative vision-language model. International Journal of Computer Vision, pages 1–23, 2025.

[40] G. Wu, H. Liu, H. Fu, Y. Peng, J. Liu, X. Fan, and R. Liu. Every sam drop counts: Embracing semantic priors for multi-modality image fusion and beyond. In Proceedings of the Computer Vision and Pattern Recognition Conference, pages 17882–17891, 2025.

[41] X. Wu, Z.-H. Cao, T.-Z. Huang, L.-J. Deng, J. Chanussot, and G. Vivone. Fully-connected transformer for multi-source image fusion. IEEE Transactions on Pattern Analysis and Machine Intelligence, 47(3):2071– 2088, 2025.

[42] C. S. Xydeas and V. Petrovic. Objective image fusion performance measure. Electronics letters, 36(4):308– 309, 2000.

[43] C. Yang, J.-Q. Zhang, X.-R. Wang, and X. Liu. A novel similarity based quality metric for image fusion. Information Fusion, 9(2):156–160, 2008.

[44] H. Zhang, L. Cao, and J. Ma. Text-difuse: An interactive multi-modal image fusion framework based on text-modulated diffusion model. Advances in Neural Information Processing Systems, 37:39552–39572, 2024.

[45] H. Zhang, L. Cao, X. Zuo, Z. Shao, and J. Ma. Omnifuse: Composite degradation-robust image fusion with language-driven semantics. IEEE Transactions on Pattern Analysis and Machine Intelligence, 47(9):7577– 7595, 2025.

[46] Z. Zhao, H. Bai, J. Zhang, Y. Zhang, S. Xu, Z. Lin, R. Timofte, and L. Van Gool. Cddfuse: Correlationdriven dual-branch feature decomposition for multi-modality image fusion. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 5906–5916, 2023.

[47] Z. Zhao, H. Bai, J. Zhang, Y. Zhang, K. Zhang, S. Xu, D. Chen, R. Timofte, and L. Van Gool. Equivariant multi-modality image fusion. In Proceedings ofthe IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 25912–25921, 2024.

[48] Z. Zhao, S. Xu, C. Zhang, J. Liu, P. Li, and J. Zhang. Didfuse: Deep image decomposition for infrared and visible image fusion. arXiv preprint arXiv:2003.09210, 2020.

[49] M. Zhou, J. Huang, K. Yan, D. Hong, X. Jia, J. Chanussot, and C. Li. A general spatial-frequency learning framework for multimodal image fusion. IEEE Transactions on Pattern Analysis and Machine Intelligence, 47(7):5281–5298, 2025.

[50] P. Zhu, Y. Sun, B. Cao, and Q. Hu. Task-customized mixture of adapters for general image fusion. In Proceedings ofthe IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 7099–7108, June 2024.

[51] Y. Zong, O. M. Aodha, and T. M. Hospedales. Self-supervised multimodal learning: A survey. IEEE Transactions on Pattern Analysis and Machine Intelligence, 47(7):5299–5318, 2025.

[52] K. H. Zou, S. K. Warfield, A. Bharatha, C. M. Tempany, M. R. Kaus, S. J. Haker, W. M. Wells, F. A. Jolesz, and R. Kikinis. Statistical validation of image segmentation quality based on a spatial overlap index1: scientific reports. Academic Radiology, 11(2):178–189, 2004.

## A Motivation of the Signal-level Formulation

The core motivation of adopting the signal-level formulation is to redefine the optimization objective for feature decomposition. For decomposition-based MMIF methods, GT feature maps for common and unique features are unavailable. Existing methods usually rely on a combination of image-level metrics, such as intensity and gradient, to indirectly constrain feature decomposition. However, these metrics only describe partial image attributes and are difficult to directly guide the decomposer to explicitly learn common feature and unique features. Therefore, the feature decomposition process still suffers from an unclear supervision target.

We reformulate this supervision-ambiguous feature decomposition problem as a signal-level integral optimization problem. The integral objective describes the global accumulated relation along the signal sequence, making the 1D signal space a natural domain for this optimization form. In this domain, the common signal can be constrained by the integral area between itself and the two source signals, which provides the decomposer with an explicit, computable, and decompositionoriented learning target. In this way, the signalized representation becomes a key form to support the integral-driven objective, giving the decomposer a clearer optimization objective.

Please note that, the signal-level formulation does not discard spatial information. The 1D signal is mainly used to construct a clearer decomposition objective, while the decomposed signals are mapped back to 2D feature maps and further constrained by the image-level reconstruction task. Therefore, the signal-level and image-level tasks are complementary: the former provides an explicit objective for common and unique feature separation, while the latter preserves spatial structural consistency during reconstruction.

## B Ambiguous Supervision of Pixel-Level Metric-Mixture Paradigm.

In current decomposition-based MMIF methods, ground-truth common and unique feature maps are unavailable. As a result, existing methods usually constrain the decomposition process by imposing losses between the learned representation and the source images. However, each image pixel does not correspond to a single visual property. Instead, it simultaneously couples multiple attributes, such as intensity, contrast, edge, texture, contour, structure, and salient modality-dependent responses. To make this issue more explicit, we summarize representative visual attributes coupled in image pixels and their commonly used constraints in Table 3.

Table 3: Representative visual attributes coupled in image pixels and their commonly used constraints in MMIF.
<table><tr><td>Visual Attribute</td><td>Pixel-level Manifestation</td><td>Commonly Used Constraints</td></tr><tr><td>Intensity / Brightness</td><td>Absolute pixel magnitude, local lumi- nance</td><td> $L _ { 1 }$  loss,  $L _ { 2 }$  loss, reconstruction loss</td></tr><tr><td>Contrast</td><td>Relative intensity difference between a pixel and its local neighborhood, or be-</td><td>SSIM contrast term, local contrast loss, entropy-based or information-based constraints</td></tr><tr><td>Edge</td><td>tween foreground and background Sharp local intensity variation around ob- ject boundaries and structural transitions</td><td>Gradient loss, Sobel loss, Laplacian loss</td></tr><tr><td>Texture</td><td>fine details, and high-frequency responses</td><td>Gradient loss, frequency-domain loss</td></tr><tr><td>Boundary</td><td>Continuous object outlines and region boundaries</td><td>SSIM loss, gradient loss, boundary-aware loss</td></tr><tr><td>Structural Layout</td><td>Spatial arrangement of objects, organs,</td><td>SSIM loss, perceptual loss</td></tr><tr><td>Color</td><td>roads, targets, or anatomical regions Channel-wise appearance, pseudo-color distribution, and modality-related color</td><td>Color consistency loss, histogram loss</td></tr></table>

As shown in Table 3, a single pixel simultaneously reflects multiple visual attributes, whereas each loss term only constrains part of them from a specific perspective. Obviously, it is almost impossible to embed all these attribute-related constraints into a single loss formulation. More importantly, there is no prior knowledge about which combination of these metrics is optimal for feature decomposition. Different losses may even impose potentially conflicting optimization preferences. Therefore, existing supervision based on empirical metric mixtures is inherently incomplete and ambiguous. This is exactly why we reformulate feature decomposition into a signal-level integral optimization problem, which provides the decomposer with a clearer objective.

Algorithm 1 Stage I: Self-supervised signal-level feature decomposition with dual pretext tasks   
Input: Training set $\overline { { \mathcal { D } = \{ ( I _ { A } ^ { k } , I _ { B } ^ { k } ) \} _ { k = 1 } ^ { N } } }$   
Modules: Encoder $\zeta ,$ signal-level decomposer $S D _ { \mathbf { \Omega } }$ , decoder $\xi$   
Hyper-parameters: $\alpha , \beta$ for Stage I.   
1: Initialize parameters of $\zeta , S D , \xi$   
2: for epoch = 1 to $E _ { 1 }$ do   
3: for each mini-batch $( I _ { A } , I _ { B } ) \subset \mathcal { D }$ do   
4: $F _ { A } , F _ { B } = \zeta ( I _ { A } ) \overset { \cdot } { , } \zeta ( I _ { B } )$   
5: $s _ { A } ( l ) , s _ { B } ( \bar { l } ) \stackrel { . } { = } \bar { H } ( \dot { F } _ { A } , \stackrel { . } { F _ { B } } )$   
6: $L _ { A } ( l ) , \{ \dot { H _ { A } ^ { i } } ( l ) \} _ { i = 1 } ^ { N } = \mathrm { D W T } ( s _ { A } ( l ) )$   
7: $L _ { B } ( l ) , \{ H _ { B } ^ { i } ( l ) \} _ { i = 1 } ^ { N } = \mathrm { D W T } ( s _ { B } ( l ) )$   
8: // Signal-level decomposer: common & unique signals   
9: $c _ { s } ( \bar { l } ) = \underline { { S } } D _ { \mathrm { c o m m o n } } ( \bar { L _ { A } } ( l ) , L _ { B } ( l ) )$   
10: $u _ { s } ^ { A } ( \bar { l } ) , u _ { s } ^ { B } ( \bar { l } ) = \bar { S } \bar { D } _ { \mathrm { s p e c i f i c } } ( \{ \bar { H } _ { A } ^ { i } ( \bar { l } ) \} , \{ H _ { B } ^ { i } ( \bar { l } ) \} )$   
11: // Signal-level pretext: common-signal optimization with $L _ { \mathrm { d e c } }$   
12: $L _ { \mathrm { d e c } } =$ IntegralSquaredError $( c _ { s } ( l ) , s _ { A } ( l ) , s _ { B } ( l ) )$   
13: // Image-level pretext: reconstruct each source image   
14: $\hat { s } _ { A } ( l ) = \mathrm { D W } \bar { \mathrm { T } } ^ { - 1 } ( c _ { s } ( l ) , u _ { s } ^ { A } ( l ) )$   
15: $\hat { s } _ { B } ( \boldsymbol { l } ) = \mathrm { D W T } ^ { - 1 } ( c _ { s } ( \boldsymbol { l } ) , u _ { s } ^ { B } ( \boldsymbol { l } ) )$   
16: $\hat { F } _ { A } = H ^ { \top } ( \hat { s } _ { A } ( l ) ) , \quad \hat { F } _ { B } = H ^ { \top } ( \hat { s } _ { B } ( l ) )$   
17: $\hat { I } _ { A } = \xi ( \hat { F } _ { A } ) , \quad \hat { I } _ { B } = \xi ( \hat { F } _ { B } )$   
18: $L _ { \mathrm { r e c } } = \mathrm { M S E } ( \hat { I } _ { A } , I _ { A } ) + \mathrm { M S E } ( \hat { I } _ { B } , I _ { B } )$   
19: // Dual-pretext lossfor decomposer   
20: $L _ { \mathrm { s t a g e 1 } } = \alpha L _ { \mathrm { d e c } } + \beta L _ { \mathrm { r e c } }$   
21: Update parameters of $\zeta , S D ,$ ξ by back-propagating $L _ { \mathrm { s t a g e l } }$   
22: end for   
23: end for   
24: Output: Pretrained encoder ζ and signal-level decomposer SD

## C Two-Stage Training and Inference Pipeline

For clarity, we summarize the overall pipeline of the proposed method in a two-stage manner, as shown in Algorithm 1 and Algorithm 2.

## C.1 Stage I: Self-supervised signal-level decomposition.

In the first stage (Algorithm 1), we focus on learning a stable signal-level decomposer in a fully self-supervised fashion. Given paired source images $( I _ { A } , I _ { B } )$ , we first extract high-level features via the encoder $\zeta ,$ and then project the 2D feature maps into 1D signals. These 1D signals are decomposed by a 1D DWT into low- and high-frequency components, which are further processed by the signal-level decomposer SD to obtain a common signal $c _ { s } ( l )$ and modality-specific unique signals $\bar { u _ { s } ^ { A } } ( l )$ and $u _ { s } ^ { B } ( l )$ . To train SD, we introduce a dual-pretext objective: a signal-level loss $L _ { \mathrm { d e c } }$ that directly supervises the decomposition in the 1D domain, and an image-level reconstruction loss $L _ { \mathrm { r e c } }$ that reconstructs each source image via inverse DWT and the decoder $\xi .$ . The decomposer and encoder are optimized jointly with the combined loss $L _ { \mathrm { s t a g e 1 } } = \alpha L _ { \mathrm { d e c } } + \beta L _ { \mathrm { r e c } }$ , yielding a pretrained encoder–decomposer pair that can reliably separate common and modality-specific information.

## C.2 Stage II: Fusion based on signal-level decomposition.

In the second stage (Algorithm 2), we freeze the encoder ζ and decomposer SD learned in Stage I, and only train the fusion head δ together with the decoder ξ. For each training pair, we reuse the signal-level decomposition to obtain the common signal $c _ { s } ( l )$ and unique signals $u _ { s } A ( l )$ and $u _ { s } B ( l )$ and then feed the unique signals into the learnable fusion module δ to produce a fused unique signal $u _ { f } ( l )$ . Combining $c _ { s } ( \bar { l } )$ and $u _ { f } ( l )$ through inverse DWT gives a fused 1D signal, which is mapped back to 2D feature space and decoded into the fused image $I _ { f }$ . The fusion-specific loss $\mathcal { L }$ is computed between $I _ { f }$ and the source images, encouraging the fused output to preserve structural details, intensity information, and gradient cues from both modalities. After training, inference reduces to a single forward pass through this Stage II pipeline, i.e., $I _ { f } = G ( I _ { A } , I _ { B } )$ , for both MIF and VIF tasks.

Algorithm 2 Stage II: Fusion based on signal-level decomposition   
Require: Training set $\overline { { \mathcal { D } = \{ ( I _ { A } ^ { k } , I _ { B } ^ { k } ) \} _ { k = 1 } ^ { N } / } }$ registered multimodal image pairs   
Require: Pretrained encoder ζ and signal-level decomposer SD from Stage I //fixedfeature extractor   
and decomposer   
Ensure: Fused image generator $G ( I _ { A } , I _ { B } )$ //for both MIF and VIF tasks   
1: Freeze parameters of $\zeta$ and $S D / /$ use Stage I as a fixed decomposer   
2: Initialize fusion module δ and decoder $\xi \ddot { / / }$ learnablefusion head and image reconstructor   
3: for each training epoch do   
4: for each mini-batch $( I _ { A } , I _ { B } ) \subset \mathcal { D }$ do   
5: $F _ { A } , F _ { B } = \zeta ( I _ { A } ) \overset { \cdot } { , } \zeta ( I _ { B } )$   
6: $s _ { A } ( l ) , s _ { B } ( \bar { l } ) \stackrel { } { = } \bar { H } ( \dot { F } _ { A } , \stackrel { } { F _ { B } } )$   
7: $L _ { A } ( l ) , \{ \dot { H _ { A } ^ { i } } ( l ) \} = \mathrm { D W T } ( s _ { A } ( l ) )$   
8: $L _ { B } ( l ) , \{ H _ { B } ^ { i } ( l ) \} = \mathrm { D W T } ( s _ { B } ( l ) )$   
9: $c _ { s } ( \tilde { l } ) = \dot { S } \tilde { D } _ { \mathrm { c o m m o n } } ( L _ { A } ( l ) , L _ { B } ( \tilde { l } ) )$   
10: $u _ { s } ^ { A } ( \acute { l } ) , u _ { s } ^ { B } ( l ) = \bar { S } \bar { D } _ { \mathrm { s p e c i f i c } } ( \{ \bar { H } _ { A } ^ { i } ( \acute { l } ) \} , \{ H _ { B } ^ { i } ( \acute { l } ) \} ) /$ / modality-specific unique signals   
11: $u _ { f } ( l ) = \delta ( u _ { s } ^ { A } ( l ) , u _ { s } ^ { \bar { B ( } } l ) ) / / $ fuse unique signals by the learnablefusion head   
12: $f _ { s } ( l ) = \mathrm { D W T } ^ { - 1 } ( c _ { s } ( l ) , u _ { f } ( l ) )$ // reconstruct fused 1D signal with inverse DWT   
13: $\begin{array} { r } { F _ { f } = H ^ { \top } ( f _ { s } ( l ) ) } \end{array}$   
14: $\dot { I _ { f } } = \xi ( F _ { f } ) / /$ decodefusedfeature intofused image   
15: $\dot { L _ { \mathrm { f u s e } } } \stackrel { \sim } { = } \dot { \mathcal { L } } ( I _ { f } ; I _ { A } , I _ { B } )$   
16: Update parameters of δ and $\xi$ using gradient of $L _ { \mathrm { f u s e } }$ // optimizefusion head and decoder   
17: end for   
18: end for   
19: $I _ { f } = G ( I _ { A } , I _ { B } )$ for a given test pair $( I _ { A } , I _ { B } )$ // inference: one forward pass through the Stage II   
pipeline

## D Extended Experimental Results and Visualizations

We adopt the same evaluation as in the main paper. Following mainstream practice, we use a set of widely used image quality metrics, including $Q _ { M I }$ [25], Q [35], Q [42], Q [38], MI [25], VIF<sub>P</sub> [26], and $Q _ { Y }$ [43]. For fair comparison, we also keep the same state-of-the-art baselines as in the main paper: for the VIF task we consider CDDFuse [46], LRRNet [9], EMMA [47], TC-MoA [50], Text-Difuse [44], SigFusion [36], DCEvo [17], SAGE [40], TDFusion [1], Omni-Fuse [45], and C2RF [31]; for the MIF task we adopt CDDFuse [46], LRRNet [9], EMMA [47], TC-MoA [50], Text-Difuse [44], CCF [4], SigFusion [36], BSA-Fusion [8], Mask-Difuser [29], MTG-Fusion [39], and C2RF [31]. Due to page constraints in the main paper, only a subset of results can be shown there; this section reports extended quantitative results and richer visualizations across all datasets and modalities to more comprehensively demonstrate the behavior and advantages of our method.

## D.1 Full Qualitative Comparisons

We provide visual comparisons against all competing methods on six datasets. For each dataset, we show the source images, the fused results generated by representative competitors, and the output of our method. The extended visualizations are presented in Fig. 8, where the first three groups correspond to MIF cases (MRI-CT, MRI-PET, MRI-SPECT) and the last three groups correspond to VIF cases (TNO, $\mathbf { M } ^ { \mathrm { 3 } } \mathbf { F D }$ , MSRS). Within each row, the first two columns display the source image pairs, while the remaining columns show the fused results of different comparison methods, with our fusion result placed in the rightmost column.

Benefiting from the proposed 1D signal-level decomposer and the integral-driven loss $L _ { \mathrm { d e c } }$ , our method yields fused images that retain both the global structures and modality-specific details. On the MIF datasets, our results preserve sharp boundaries, bone regions and texture from MRI and CT, while maintaining the functional patterns of PET and SPECT without color distortion. On the VIF datasets, our fused images simultaneously preserve the fine textures and contrast of the visible modality and the salient thermal targets from infrared, leading to clearer edges, more stable illumination than competing methods. These visual results are consistent with our signal-level loss which provides a clearer optimization objective with the dual pretext tasks, jointly enabling more accurate feature decomposition and, consequently, more advanced fusion results across all six datasets.

![](images/b6278704f47949167a49eab722890ec106345d9e0e396f50b6a288ab49204606.jpg)  
Figure 8: Full qualitative comparisons on six datasets. The top three groups show MIF task and the bottom three groups show VIF cases. In each row, the first two columns are the source image pairs and the remaining columns are fused results of different methods, with our method in the rightmost column, yielding sharper structures and richer complementary details.

## D.2 Downstream Task Evaluation

Object Detection for VIF Task. To evaluate whether the fused images can better support downstream visual understanding, we conduct object detection on the VIF task. Specifically, fused images generated by different VIF methods are fed into the same pre-trained YOLOv12 detector [33], without modifying the detector or fine-tuning it on any specific fusion result. This setting ensures that the comparison mainly reflects the influence of different fusion outputs on detection performance.

As shown in Fig. 5(left), the detection results produced from our fused images are generally more complete and accurate than those from other fusion methods. In challenging scenes, competing methods may miss small targets, generate inaccurate bounding boxes, or confuse objects with weak contrast. In contrast, our method better preserves complementary infrared and visible cues, which helps the detector localize objects more reliably. Table 4 reports the per-class AP [13] and mAP results. Compared with other VIF methods, our method achieves the best overall mAP and competitive

Table 4: AP for object detection on source images and fused images from different methods. mAP is the mean of the six category APs.
<table><tr><td>Method</td><td>People</td><td>Car</td><td>Bus</td><td>Motorcycle</td><td>Lamp</td><td>Truck</td><td>mAP</td></tr><tr><td>IR</td><td>0.512</td><td>0.581</td><td>0.447</td><td>0.468</td><td>0.403</td><td>0.452</td><td>0.477</td></tr><tr><td>VIS</td><td>0.563</td><td>0.638</td><td>0.483</td><td>0.521</td><td>0.436</td><td>0.494</td><td>0.522</td></tr><tr><td>CDDFuse</td><td>0.662</td><td>0.726</td><td>0.618</td><td>0.634</td><td>0.573</td><td>0.602</td><td>0.636</td></tr><tr><td>LRRNet</td><td>0.641</td><td>0.707</td><td>0.597</td><td>0.612</td><td>0.551</td><td>0.582</td><td>0.615</td></tr><tr><td>EMMA</td><td>0.651</td><td>0.719</td><td>0.612</td><td>0.631</td><td>0.562</td><td>0.593</td><td>0.628</td></tr><tr><td>TC-MoA</td><td>0.642</td><td>0.715</td><td>0.606</td><td>0.623</td><td>0.559</td><td>0.601</td><td>0.624</td></tr><tr><td>Text-DiFuse C2RF</td><td>0.598</td><td>0.689</td><td>0.542 0.553</td><td>0.588</td><td>0.521</td><td>0.552</td><td>0.582</td></tr><tr><td>SigFusion</td><td>0.611</td><td>0.701 0.763</td><td>0.653</td><td>0.599</td><td>0.515</td><td>0.561</td><td>0.590</td></tr><tr><td>DCEvo</td><td>0.688</td><td></td><td>0.661</td><td>0.649</td><td>0.579</td><td>0.622</td><td>0.655</td></tr><tr><td>SAGE</td><td>0.689</td><td>0.735</td><td>0.571</td><td>0.653</td><td>0.583</td><td>0.612</td><td>0.655</td></tr><tr><td></td><td>0.629</td><td>0.703</td><td>0.632</td><td>0.609</td><td>0.542</td><td>0.571</td><td>0.604</td></tr><tr><td>TD-Fusion</td><td>0.661</td><td>0.731</td><td>0.601</td><td>0.643</td><td>0.583</td><td>0.611</td><td>0.643</td></tr><tr><td>Omni-Fuse</td><td>0.649</td><td>0.721</td><td></td><td>0.621</td><td>0.571</td><td>0.603</td><td>0.628</td></tr><tr><td>Ours</td><td>0.691</td><td>0.737</td><td>0.658</td><td>0.651</td><td>0.577</td><td>0.623</td><td>0.656</td></tr></table>

Table 5: Quantitative comparison for medical image segmentation.
<table><tr><td>Metrics</td><td>T1 Flair</td><td>CDDFuse</td><td></td><td>LRRNet</td><td>EMMA</td><td>TC-MoA</td><td>Text-Diffuse</td></tr><tr><td>IoU</td><td>0.615</td><td>0.657</td><td>0.762</td><td>0.731</td><td>0.743</td><td>0.689</td><td>0.685</td></tr><tr><td>Dice</td><td>0.607</td><td>0.727</td><td>0.850</td><td>0.836</td><td>0.841</td><td>0.829</td><td>0.751</td></tr><tr><td>Metrics</td><td>CCF</td><td>SigFusion</td><td>BSAFusion</td><td>Mask-Difuser</td><td>MTG-Fusion</td><td>C2RF</td><td>Ours</td></tr><tr><td>IoU</td><td>0.661</td><td>0.735</td><td>0.745</td><td>0.723</td><td>0.775</td><td>0.772</td><td>0.790</td></tr><tr><td>Dice</td><td>0.752</td><td>0.770</td><td>0.832</td><td>0.829</td><td>0.846</td><td>0.848</td><td>0.856</td></tr></table>

AP on most object categories, demonstrating that the proposed fusion strategy can provide more task-favorable inputs for downstream detection.

Medical Image Segmentation for MIF Task. We further evaluate the downstream utility of our method on medical image segmentation. In this experiment, the fused images generated by different MIF methods are used as inputs to UniverSeg [3], and the segmentation results are compared with the ground-truth masks. This evaluation examines whether the fused images can preserve lesion-related structures and boundary information that are useful for medical analysis.

As shown in Fig. 5(right), segmentation results based on our fused images are closer to the ground truth, especially around lesion boundaries and structurally complex regions. Other fusion methods may produce incomplete lesion regions or less accurate boundaries due to insufficient preservation of cross-modal complementary information. In contrast, our method better maintains common anatomical structures while retaining modality-specific details, which provides more discriminative cues for segmentation. Table 5 reports the IoU [52] and Dice [23] results. Our method obtains the best performance among the compared fusion methods, confirming that the proposed decomposition strategy can improve not only visual fusion quality but also downstream medical segmentation performance.

## D.3 Visualization of Common and Unique Features

To better understand our feature decomposition mechanism, we compare the estimated common and unique features of our method with those of several representative decomposition-based baselines. For each source pair $( I _ { A } , I _ { B } )$ , we visualize the common feature maps and the corresponding modalityunique feature maps.

As shown in Fig. 9, existing image-level decomposers typically produce two common features for each modality. In practice, these two features are not strictly shared as they often contain modalityspecific patterns, which contradicts the assumption that there should be a single common component underlying both modalities. In contrast, our signal-level decomposer explicitly models a single common feature map $C _ { A B }$ together with two unique feature maps for $I _ { A }$ and $I _ { B }$ , leading to a more reasonable and identifiable decomposition form.

![](images/be79475969550d8d8cccdb850dfa28cf0dee2b59ed4fdcf406701d86e5a78506.jpg)  
Figure 9: Visualization of common and unique feature maps produced by different decomposition strategies. For each source pair, we display the estimated common features and the modality-unique features. Compared with existing methods that generate two inconsistent common features, our signal-level decomposer yields a single common map and better-separated unique maps.

The visualizations further reveal that, for our method, the common feature mainly captures shared semantic content and global structures, while the unique feature concentrate on modality-specific cues. These observations align with our signal-level formulation and support that the proposed decomposer separates common and modality-specific information in a more structured and disentangled way.

## D.4 t-SNE Visualization of Frequency and Decomposed Features

To further analyze the relationship between frequency components and the learned decomposed features, we visualize their feature distributions using t-SNE. Specifically, we project the lowfrequency components, high-frequency components, learned common features, and modality-specific unique features into a two-dimensional space. As shown in Fig. 10, the learned common features are mainly distributed around the low-frequency components of both modalities, indicating that lowfrequency signals contain more modality-shared structural information. In contrast, the learned unique features are closer to the corresponding high-frequency components and form more modality-specific distributions. This observation is consistent with our design motivation that common information can be better extracted from low-frequency components, while modality-specific details are more related to high-frequency components.

These results provide intuitive evidence for the proposed decomposition strategy. By exploiting the complementary roles of low- and high-frequency components, our method can more effectively separate common structures and modality-specific details.

## D.5 Visualization of Training Curves

To analyze the optimization behavior of the proposed signal-level formulation, we compare it with a conventional 2D image-domain mixed-loss strategy. As shown in Fig. 11, our signal-level objective converges faster and reaches a lower final loss, indicating a clearer and more stable optimization target.

We further visualize the similarity curves during training, including SSIM, Pearson correlation, and cosine similarity. Compared with the 2D mixed-loss strategy, our method shows higher and more stable similarity between the learned common features and source signals. These results provide intuitive evidence that the proposed signal-level objective can better guide common feature learning.

t-SNE visualization  
![](images/96feca434b02844cc11d48cb0c01003c067c1f8387d74d2db36d018d21837f07.jpg)  
Figure 10: t-SNE visualization of low-/high-frequency components and learned decomposed features. The common features are close to the low-frequency components of both modalities, while the unique features are closer to their corresponding high-frequency components, supporting the design of extracting common information from low frequencies and unique information from high frequencies.

![](images/f3e5a66895fbb13b53d5f7d4d9e2851ea6d00072308b387192b0cf5c711014de.jpg)  
(a) Loss Curve

![](images/e4e8e27d3f43d34fc9b19f977a9d8d7dc18330b73329e68520900429d8fac7c0.jpg)  
(b) SSIM Curve

![](images/5551e3cb3b091fc6873be24149038cdb8896f8ccb2453f5266021f81a34e1d51.jpg)  
(c) Pearson Curve

![](images/ab570c5c3748b7beabc2a7a56c9808448d1a74412347125640e48e870e7dc0a0.jpg)  
(d) Cosine Curve  
Figure 11: Visualization of training curves. We compare the proposed signal-level objective with a conventional 2D image-domain mixed-loss strategy in terms of training loss and feature similarity, including SSIM, Pearson correlation, and cosine similarity. The results show that our signal-level objective converges faster and achieves a lower final loss, while maintaining higher and more stable similarity between the learned common features and source signals.

## E Additional Results for Broader Impact

In this section, we provide the detailed quantitative results corresponding to the broader impact analysis in the main paper. We mainly consider two aspects: refining existing decomposition-based MMIF paradigms and extending the proposed signal-level decomposition paradigm to other image fusion tasks.

## E.1 Refining Decomposition-based MMIF Paradigms

To further validate the compatibility of the proposed signal-level decomposition paradigm, we insert it into several representative decomposition-based MMIF models. For a fair comparison, we keep the original training strategies, network settings, and evaluation protocols of these models unchanged, and only replace or enhance their feature decomposition process with our signal-level decomposition paradigm. The enhanced variants are denoted by “<sup>∗</sup>”.

Table 6 reports the quantitative results. Compared with the original models, the enhanced variants achieve consistent improvements on most evaluation metrics. These results indicate that the proposed paradigm is not limited to our specific network architecture. Instead, by providing a clearer objective for separating common and unique features, it can serve as a generally effective refinement for existing decomposition-based MMIF methods.

Table 6: The refinement effects results of our paradigm. Original vs. upgraded models (\* denotes our paradigm inserted).
<table><tr><td>Method</td><td> $\overline { { Q _ { M I } \mathcal { \Lambda } } }$ </td><td> $\overline { { Q _ { N I C E } \mathrm { \uparrow } } }$ </td><td> $\overline { { Q _ { P } \uparrow } }$ </td><td> $\overline { { Q _ { C B } \mathrm { \uparrow } } }$ </td><td>MI↑</td><td> $\overline { { V I F _ { p } \ : \uparrow } }$ </td><td> $\overline { { Q _ { Y } \mathrm { \uparrow } } }$ </td></tr><tr><td>DIDFuse DIDFuse*</td><td>0.6130 0.7466</td><td>0.8083 0.8196</td><td>0.3570 0.4198</td><td>0.4830 0.5299</td><td>3.2370 3.9038</td><td>0.3610 0.4596</td><td>0.9020 0.9101</td></tr><tr><td>Gain CDDFuse</td><td>21.8%↑ 0.6610</td><td>1.4%↑ 0.8107</td><td>17.6%↑ 0.4010</td><td>9.7%↑ 0.6940</td><td>20.6%↑ 3.9820</td><td>27.3%↑ 0.3990</td><td>0.9%↑ 0.9380</td></tr><tr><td>CDDFuse* Gain</td><td>0.7582 14.7%↑</td><td>0.8277 2.1%↑</td><td>0.4523 12.8%↑</td><td>0.7051 1.6%↑</td><td>4.3563 9.4%↑</td><td>0.4736 18.7%↑</td><td>0.9446 0.7%↑</td></tr><tr><td>FDFuse</td><td>0.6420</td><td>0.8086</td><td>0.3890</td><td>0.6860</td><td>3.9050</td><td>0.3920</td><td>0.9330</td></tr><tr><td>FDFuse*</td><td>0.8044</td><td>0.8215</td><td>0.4820</td><td>0.7320</td><td>4.5532</td><td>0.5088</td><td>0.9433</td></tr><tr><td>Gain</td><td>25.3%↑</td><td>1.6%↑</td><td>23.9%↑</td><td>6.7%↑</td><td></td><td></td><td></td></tr><tr><td>C2RF</td><td>0.6360</td><td></td><td></td><td></td><td>16.6%↑</td><td>29.8%↑</td><td>1.1%↑</td></tr><tr><td>C2RF*</td><td></td><td>0.8071</td><td>0.3660</td><td>0.6740</td><td>3.8510</td><td>0.3730</td><td>0.9290</td></tr><tr><td></td><td>0.8243</td><td>0.8224</td><td>0.4663</td><td>0.7535</td><td>4.8022</td><td>0.4566</td><td>0.9383</td></tr><tr><td>Gain</td><td>29.6%↑</td><td>1.9%↑</td><td>27.4%↑</td><td>11.8%↑</td><td>24.7%↑</td><td>22.4%↑</td><td>1.0%↑</td></tr></table>

Table 7: Quantitative comparison on the MFIF task against 9 competing methods using 6 evaluation metrics on the LYTRO and MFFW datasets. Although MFIF may provide fused ground truth for evaluation, the decomposition of common and unique features still lacks direct supervision. Our method remains applicable in this setting and achieves strong overall performance.
<table><tr><td rowspan="2">Method</td><td rowspan="2">Pub/Year</td><td colspan="6">Dataset: LYTRO</td><td colspan="6">Dataset: MFFW</td></tr><tr><td> $Q _ { G } \uparrow$ </td><td>QM↑</td><td> $Q _ { P } \uparrow$ </td><td>MI↑</td><td>SD↑</td><td>VIFF↑</td><td> $Q _ { G } \uparrow$ </td><td> $Q _ { M }$  ←</td><td>QP↑</td><td>MI↑</td><td>SD↑</td><td>VIFF↑</td></tr><tr><td>CUNet</td><td>TPAMI 20</td><td>0.526</td><td>0.553</td><td>0.696</td><td>5.441</td><td>58.70</td><td>1.022</td><td>0.482</td><td>0.455</td><td>0.552</td><td>4.593</td><td>56.33</td><td>0.847</td></tr><tr><td>U2Fusion</td><td>TPAMI 20</td><td>0.580</td><td>0.480</td><td>0.742</td><td>5.677</td><td>58.37</td><td>1.086</td><td>0.537</td><td>0.405</td><td>0.611</td><td>4.876</td><td>55.26</td><td>0.825</td></tr><tr><td>DeFusion</td><td>ECCV 22</td><td>0.455</td><td>0.325</td><td>0.660</td><td>5.984</td><td>54.39</td><td>1.028</td><td>0.418</td><td>0.296</td><td>0.518</td><td>5.137</td><td>51.55</td><td>0.876</td></tr><tr><td>DIFNet</td><td>CVPR 22</td><td>0.437</td><td>0.325</td><td>0.688</td><td>5.774</td><td>49.67</td><td>1.032</td><td>0.422</td><td>0.309</td><td>0.577</td><td>4.867</td><td>46.66</td><td>0.890</td></tr><tr><td>FusionDiff</td><td>ESWA 23</td><td>0.629</td><td>0.821</td><td>0.783</td><td>6.554</td><td>56.13</td><td>1.188</td><td>0.545</td><td>0.602</td><td>0.659</td><td>5.334</td><td>53.27</td><td>0.993</td></tr><tr><td>MGDN</td><td>ACMMM 23</td><td>0.662</td><td>0.901</td><td>0.810</td><td>6.655</td><td>56.88</td><td>1.226</td><td>0.606</td><td>0.650</td><td>0.663</td><td>5.558</td><td>54.47</td><td>1.020</td></tr><tr><td>ZMFF</td><td>INFFUS 23</td><td>0.631</td><td>0.600</td><td>0.785</td><td>6.235</td><td>57.06</td><td>1.175</td><td>0.552</td><td>0.487</td><td>0.635</td><td>5.092</td><td>54.38</td><td>0.990</td></tr><tr><td>DeepM2CDL</td><td>TPAMI 24</td><td>0.639</td><td>1.004</td><td>0.810</td><td>6.441</td><td>58.05</td><td>1.267</td><td>0.582</td><td>0.805</td><td>0.679</td><td>5.372</td><td>56.01</td><td>1.064</td></tr><tr><td>FILM</td><td>ICML 24</td><td>0.619</td><td>0.567</td><td>0.782</td><td>6.758</td><td>59.15</td><td>1.283</td><td>0.498</td><td>0.436</td><td>0.544</td><td>5.254</td><td>57.10</td><td>0.919</td></tr><tr><td>Ours</td><td></td><td>0.659</td><td>1.031</td><td>0.811</td><td>6.733</td><td>58.66</td><td>1.292</td><td>0.611</td><td>0.994</td><td>0.682</td><td>6.002</td><td>56.18</td><td>1.112</td></tr></table>

## E.2 Extension to Multi-focus Image Fusion

We further examine whether the proposed signal-level decomposition paradigm can be transferred to other image fusion tasks beyond MMIF. In this section, we select multi-focus image fusion as a representative task. Different from MMIF, multi-focus image fusion aims to integrate multiple images of the same scene with different focus regions. Although the imaging setting is different, this task also requires the model to preserve common scene structures while extracting source-specific focused details. Therefore, multi-focus image fusion provides a suitable testbed for evaluating the broader applicability of our decomposition paradigm.

Table 7 reports the quantitative results. The proposed paradigm achieves competitive or superior performance across multiple metrics. These results are consistent with the qualitative observations in the main paper, where our method produces clearer focus boundaries and more faithful structural details. This further demonstrates that the signal-level decomposition paradigm can be naturally extended to other image fusion tasks and has potential value for more general fusion problems.

## F Extended Ablation Study

To comprehensively verify the effectiveness of each key component, we conduct an extended ablation study on all six datasets for both VIF and MIF tasks. The quantitative results of all variants are summarized in Table 8, and representative qualitative comparisons for typical cases are visualized in Fig. 12, complementing the main-paper results.

Table 8: Quantitative results of multiple ablation experiments for VIF and MIF tasks on six datasets.
<table><tr><td rowspan="2">Configuration settings</td><td rowspan="2" colspan="2">QM1↑ QNICE↑</td><td colspan="6">VIF task QcB↑</td><td rowspan="2" colspan="2"></td><td colspan="4">MIF task</td><td rowspan="2">Qr↑</td></tr><tr><td>QP↑</td><td></td><td></td><td>VIFp↑</td><td>Qr↑</td><td>QM1↑ QNICE↑</td><td>QP↑ QCB↑</td><td></td><td>MI↑</td><td>VIFp↑</td></tr><tr><td colspan="10"></td><td colspan="7">MIF: MRI-CT</td></tr><tr><td>w/o signal-level decomposer</td><td>0.6282</td><td>0.8181</td><td>0.5350</td><td>0.4716</td><td>3.8631</td><td>0.4391</td><td>0.8729</td><td></td><td>0.9012</td><td>0.8071</td><td>0.3627</td><td>0.6715</td><td>3.6129</td><td>0.3689</td><td>0.9127</td></tr><tr><td>w/o DWT</td><td>0.6425</td><td>0.8192</td><td>0.5551</td><td>0.4860</td><td>3.9812</td><td>0.4534</td><td>0.8847</td><td></td><td>0.9185</td><td>0.8090</td><td>0.3825</td><td>0.6880</td><td>3.8310</td><td>0.3897</td><td>0.9312</td></tr><tr><td>w/ IDWT ← learnable reconstructor</td><td>0.6531</td><td>0.8198</td><td>0.5631</td><td>0.4925</td><td>4.0200</td><td>0.4594</td><td>0.8911</td><td></td><td>0.9253 0.8096</td><td></td><td>0.3928</td><td>0.6905</td><td>3.9025</td><td>0.3954</td><td>0.9361</td></tr><tr><td>w/ signal-level ← image-level decomposer</td><td>0.6380</td><td>0.8185</td><td>0.5460</td><td>0.4793</td><td>3.9022</td><td>0.4468</td><td>0.8781</td><td></td><td>0.9108</td><td>0.8086</td><td>0.3755</td><td>0.6810</td><td>3.7420</td><td>0.3798</td><td>0.9275</td></tr><tr><td>Default setups (ours)</td><td>0.6608</td><td>0.8209</td><td>0.5735</td><td>0.4998</td><td>4.1050</td><td>0.4694</td><td>0.8988</td><td></td><td>0.9369</td><td>0.8108</td><td>0.4034</td><td>0.7008</td><td>4.0184</td><td>0.4034</td><td>0.9439</td></tr><tr><td></td><td></td><td></td><td></td><td>VIF: MSRS</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td>MIF: MRI-PET</td><td></td><td></td><td></td></tr><tr><td>w/o signal-level decomposer</td><td>0.8154</td><td>0.8230</td><td>0.5112</td><td>0.4667</td><td>4.8120</td><td>0.4932</td><td></td><td>0.8791</td><td>0.8022</td><td>0.8053</td><td>0.4613</td><td>0.4442</td><td>2.8931</td><td>0.4351</td><td>0.5620</td></tr><tr><td>w/o DWT</td><td>0.8260</td><td>0.8268</td><td>0.5304</td><td>0.5551</td><td>5.0921</td><td>0.5085</td><td></td><td>0.8897</td><td>0.8191</td><td>0.8061</td><td>0.4780</td><td>0.4557</td><td>2.9874</td><td>0.4467</td><td>0.5734</td></tr><tr><td>w/ IDWT ← learnable reconstructor</td><td>0.8335</td><td>0.8279</td><td>0.5411</td><td>0.5645</td><td>5.1982</td><td>0.5166</td><td></td><td>0.8935</td><td>0.8318</td><td>0.8070</td><td>0.4876</td><td>0.4621</td><td>3.0315</td><td>0.4520</td><td>0.5826</td></tr><tr><td>w/ signal-level ← image-level decomposer</td><td>0.8213</td><td>0.8250</td><td>0.5283</td><td>0.5512</td><td>5.0455</td><td>0.5033</td><td></td><td>0.8860</td><td>0.8244</td><td>0.8066</td><td>0.4762</td><td>0.4520</td><td>2.9634</td><td>0.4447</td><td>0.5692</td></tr><tr><td>Default setups (ours)</td><td>0.8405</td><td>0.8290</td><td>0.5483</td><td>0.5798</td><td>5.4547</td><td>0.5253</td><td></td><td>0.8936</td><td>0.8843</td><td>0.8074</td><td>0.4967</td><td>0.4740</td><td>3.0547</td><td>0.4577</td><td>0.5808</td></tr><tr><td></td><td></td><td></td><td></td><td>VIF: TNO</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td>MIF: MRI-SPECT</td><td></td><td></td><td></td></tr><tr><td>w/o signal-level decomposer</td><td>0.5410</td><td>0.8105</td><td>0.4023</td><td>0.5115</td><td>3.7412</td><td>0.4544</td><td></td><td>0.8322</td><td>0.9621</td><td>0.8072</td><td>0.5260</td><td>0.6521</td><td>2.9435</td><td>0.4410</td><td>0.8998</td></tr><tr><td>w/o DWT</td><td>0.5567</td><td>0.8116</td><td>0.4195</td><td>0.5237</td><td>3.7952</td><td>0.4658</td><td>0.8390</td><td></td><td>0.9699</td><td>0.8079</td><td>0.5361</td><td>0.6615</td><td>2.9857</td><td>0.4478</td><td>0.9047</td></tr><tr><td>w/ IDWT ← learnable reconstructor w/ signal-level ← image-level decomposer</td><td>0.5630 0.5489</td><td>0.8121 0.8113</td><td>0.4264 0.4152</td><td>0.5298</td><td>3.8434</td><td>0.4722</td><td>0.8424</td><td></td><td>0.9755</td><td>0.8080</td><td>0.5445</td><td>0.6659</td><td>3.0213</td><td>0.4511</td><td>0.9101</td></tr><tr><td>Default setups (ours)</td><td></td><td>0.8128</td><td></td><td>0.5189</td><td>3.7680</td><td>0.4607</td><td>0.8361</td><td></td><td>0.9682</td><td>0.8075</td><td>0.5324</td><td>0.6582</td><td>2.9880</td><td>0.4482</td><td>0.9036</td></tr><tr><td></td><td>0.5711</td><td></td><td>0.4318</td><td>0.5346</td><td>3.8015</td><td>II. Effect of Dual Pretext Tasks</td><td>0.4768</td><td>0.8450</td><td>0.9778</td><td>0.8083</td><td>0.5507</td><td>0.6697</td><td>3.0655</td><td>0.4520</td><td>0.9149</td></tr><tr><td colspan="10"></td><td colspan="7"></td></tr><tr><td>Configuration settings</td><td>QM1↑</td><td></td><td>QNICE↑</td><td>VIF task</td><td></td><td></td><td></td><td></td><td></td><td></td><td>QP↑</td><td>MIF task</td><td></td><td></td><td></td></tr><tr><td></td><td></td><td></td><td>QP↑</td><td> $\frac { Q _ { C B } \uparrow } { W I F ; M ^ { 3 } F D }$ </td><td>MI↑</td><td>VIFp↑</td><td></td><td>Qr↑</td><td>QM1↑</td><td>QNICE↑</td><td></td><td>QCB↑ MIF: MRI-CT</td><td>MI↑</td><td>VIFp↑</td><td>Qr↑</td></tr><tr><td>w/o signal-level pretext task</td><td>0.6315</td><td>0.8188</td><td>0.5420</td><td>0.4868</td><td>3.9033</td><td>0.4512</td><td></td><td>0.8799</td><td>0.9144</td><td>0.8093</td><td>0.3786</td><td>0.6869</td><td>3.7562</td><td>0.3831</td><td>0.9293</td></tr><tr><td>w/o image-level pretext task</td><td>0.6495</td><td>0.8195</td><td>0.5525</td><td>0.4820</td><td>3.9800</td><td>0.4553</td><td></td><td>0.8850</td><td>0.9228</td><td>0.8294</td><td>0.3866</td><td>0.6765</td><td>3.8041</td><td>0.3912</td><td>0.9335</td></tr><tr><td>w/ Ldec← L1</td><td>0.6463</td><td>0.8195</td><td>0.5482</td><td>0.4817</td><td>3.9521</td><td>0.4520</td><td></td><td>0.8832</td><td>0.9202</td><td>0.8091</td><td>0.3814</td><td>0.6851</td><td>3.7820</td><td>0.3886</td><td>0.9318</td></tr><tr><td>Default setups (ours)</td><td>0.6608</td><td>0.8209</td><td>0.5735</td><td>0.4998</td><td>4.1050</td><td>0.4694</td><td></td><td>0.8988</td><td>0.9369</td><td>0.8108</td><td>0.4034</td><td>0.7008</td><td>4.0184</td><td>0.4034</td><td>0.9439</td></tr><tr><td></td><td></td><td></td><td></td><td>VIF: MSRS</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td>MIF: MRI-PET 0.4604 2.9634</td><td></td><td>0.4461</td><td>0.5742</td></tr><tr><td>w/o signal-level pretext task w/o image-level pretext task</td><td>0.8128 0.8467</td><td>0.8220 0.8265</td><td>0.5287 0.5361</td><td>0.5652 0.5661</td></table>

Effect of Signal-Level Decomposer. Our signal-level decomposer transfers the decomposition problem into 1D signal-level, effectively disentangling image-level coupling and enabling more accurate separation of features. To assess its impact, we conduct the following ablations: (1) removing the signal-level decomposer; (2) removing frequency decomposition; (3) replacing IDWT with a learnable reconstructor; (4) substituting an image-level decomposer; (5) default setup. The corresponding quantitative results are given in the upper part of Table 8. Across all six datasets and seven metrics, the default setup consistently achieves the advanced performance, while removing or weakening any part of the signal-level decomposer leads to clear drops, which further demonstrates the effectiveness of our paradigm. In Fig. 12 (left), we visualize typical VIF and MIF examples under different configurations: the variants without a complete signal-level decomposer tend to produce blurred structures, noisy background or loss of weak details, whereas the default setup preserves sharper organ boundaries, clearer edges and richer complementary cues from both modalities.

Effect of Dual Pretext Tasks. To evaluate the dual pretext tasks at both levels, we perform the following ablations: (1) removing the signal-level pretext task; (2) removing the image-level pretext task; (3) replacing the proposed $L _ { \mathrm { d e c } }$ with vanilla $L _ { 1 }$ only; (4) default configuration. The results are reported in the lower part of Table 8. Removing either pretext task consistently degrades performance on all datasets, and using a naive $L _ { 1 }$ loss also weakens the results, highlighting the necessity to provide the decomposer a clear and properly designed supervision. These tendencies are also clearly reflected in Fig. 12 (right), where the default configuration yields the most faithful results.

Sensitivity to Key Hyperparameters. We further analyze the sensitivity of key hyperparameters and design choices. The corresponding curves and additional ablation plots are provided in Fig. 14 of the Appendix.

## G Reproducibility Verification

We further conduct robustness checks with respect to random initialization. Keeping all architectural settings unchanged, we rerun our model with multiple random seeds. As shown in Fig. 13, the results under different seeds are tightly clustered, and the quantitative metrics exhibit small standard deviations, indicating that our method is stable.

Effect of Signal-Level Decomposer  
![](images/bcad1c159ae2f38541d93a9de9bd75c3e7742cbe58d566fed4f3e7c936cb2390.jpg)

![](images/9ca35aa2a60884681ce4e5cac9c30efad8462d2fab6921d2879f76ef312fcee1.jpg)  
Default

Effect of Dual Pretext Tasks  
![](images/91d3656556ae852d7b97ef179a86ad70647ca1ff4e27ffde5d77a57dcc41ebe0.jpg)  
w/o signal-level w/o image-level pretext task pretext task w/ Ldec← L1

Figure 12: Visualization of ablation results for different configurations on representative VIF and MIF cases.  
![](images/3a4808bc92c2136365fc31cd6cd09f10c1eb89dfa8ba5316dcb1c2d9bf0c20cf.jpg)  
Figure 13: Reproducibility verification under different random seeds. For each metric, markers denote independent runs with different initializations, and their tight clustering with small standard deviations indicates that our method is robust to random initialization.

## H Model Complexity and Overhead Analysis

We analyze the computational overhead, including the number of parameters, GFLOPs, and inference time as shown in Table 9. Compared with representative baselines, our model remains computationally competitive. Although the use of DWT introduces some additional overhead, the resulting performance gains make this cost well justified.

## I Limitations

While our algorithm demonstrates promising performance, it still has several limitations. First, our method currently relies on pre-registered image pairs to achieve optimal fusion results. This requirement, which is also shared by many mainstream image fusion models, may restrict its applicability in scenarios where accurate pre-registration is challenging or expensive. Second, since our method operates in the signal domain, larger images, when flattened, produce longer 1D signals and thus lead to relatively slower processing. This indicates that image size has an impact on the computational efficiency of our algorithm, which suggests a promising direction for future optimization.

![](images/45c55d7efb1d1c5cd021b9888018aa1a518c58a809c5aee95b612e452ae569cc.jpg)  
Figure 14: Sensitivity to key hyperparameters.

Table 9: Comparison of runtime, computational cost (GFLOPs), and model size (Params) on VIF and MIF tasks. All values are averaged per image.
<table><tr><td colspan="4">VIF</td><td colspan="4">MIF</td></tr><tr><td>Method</td><td>Time (s)</td><td>GFLOPs (G)</td><td>Params (M)</td><td>Method</td><td>Time (s)</td><td>GFLOPs (G)</td><td>Params (M)</td></tr><tr><td>CDDFuse</td><td>0.006</td><td>116.851</td><td>1.186</td><td>CDDFuse</td><td>0.025</td><td>116.851</td><td>1.186</td></tr><tr><td>LRRNet</td><td>0.001</td><td>0.001</td><td>0.049</td><td>LRRNet</td><td>0.001</td><td>0.001</td><td>0.049</td></tr><tr><td>EMMA</td><td>0.058</td><td>8.861</td><td>1.516</td><td>EMMA</td><td>0.009</td><td>8.861</td><td>1.516</td></tr><tr><td>TC-MoA</td><td>0.543</td><td>61.000</td><td>340.580</td><td>TC-MoA</td><td>0.161</td><td>61.000</td><td>340.580</td></tr><tr><td>Text-Difuse</td><td>23.818</td><td>18513</td><td>119.460</td><td>Text-Difuse</td><td>24.424</td><td>2742.5</td><td>119.460</td></tr><tr><td>SigFusion</td><td>0.533</td><td>41.761</td><td>0.177</td><td>CCF</td><td>31.876</td><td>1114.000</td><td>552.810</td></tr><tr><td>DČEvo</td><td>0.521</td><td>15.000</td><td>2.000</td><td>SigFusion</td><td>0.037</td><td>41.761</td><td>0.177</td></tr><tr><td>SAGE</td><td>0.020</td><td>29.285</td><td>0.136</td><td>BSA-Fusion</td><td>0.127</td><td>13.437</td><td>9.694</td></tr><tr><td>TD-Fusion</td><td>0.053</td><td>3.886</td><td>0.059</td><td>Mask-Difuser</td><td>0.503</td><td>995.706</td><td>171.262</td></tr><tr><td>Omni-Fuse</td><td>2.614</td><td>172.923</td><td>78.320</td><td>MTG-Fusion</td><td>0.053</td><td>121.411</td><td>4.273</td></tr><tr><td>C2RF</td><td>0.089</td><td>123.869</td><td>1.325</td><td>C2RF</td><td>0.112</td><td>123.869</td><td>1.325</td></tr><tr><td>Ours</td><td>0.059</td><td>4.441</td><td>0.610</td><td>Ours</td><td>0.011</td><td>4.441</td><td>0.610</td></tr></table>
# Role-Guided MOE for Encoder-Level Pathology Representation Learning in WSI Classification

Xinyu Ma, Xing Yang, Hongtao Jin, Guoquan Zhang, Shijie Zhang, Yu Zhang, and Xitong Li

Abstract— Whole slide image (WSI) classification is a fundamental task in computational pathology, where the quality of patch representations directly affects downstream aggregation and determines the discriminability of slide-level predictions. Pathology foundation models have recently been widely adopted as frozen feature extractors for WSI classification; however, their fixed encoders may produce patch representations that are insufficiently adapted to target-specific tissue patterns and discriminative cues. Fine-tuning the encoder can improve target adaptation, but it introduces a practical dilemma between pathology-specific representation capacity and adaptation efficiency, particularly in data-scarce pathology settings. To address these challenges, we propose a pathology roleguided mixture-of-experts feed-forward network (MoE-FFN) framework for efficient encoder-level representation learning. We innovatively design a two-stage training paradigm to establish and adapt pathology-aware expert specialization. In source-domain expert initialization, pathologyspecific priors are distilled from a frozen Virchow2 teacher into a lightweight DINOv2-small student, while role prototypes serve as weak pathological anchors to encourage distinct expert functions. MoE-FFN blocks are introduced into selected high-level transformer layers to provide transformation diversity for heterogeneous pathological patterns. In target-domain adaptation, the initialized experts are refined through asymmetric prototype-guided optimization, which enhances task-relevant positive evidence and separates confusable hard negatives. The resulting encoder extracts offline patch representations that can be directly integrated with standard MIL aggregators for slide-level classification. Experiments on the public BRACS dataset and a private PAROTID WSI dataset across five representative backbones demonstrate consistent improvements over the strongest baseline.

Index Terms— Computational pathology, whole slide image classification, pathology foundation models, encoder optimization, mixture of experts, multiple instance learning.

## I. INTRODUCTION

W <sup>HOLE</sup> <sup>slide</sup> <sup>images</sup> <sup>(WSIs)</sup> <sup>are</sup> <sup>a</sup> <sup>cornerstone</sup> <sup>of</sup> <sup>digi-</sup> tal pathology, providing gigapixel-scale representations that capture diagnostically relevant morphological and structural information [1]. Accurate WSI classification is therefore crucial for assisting pathologists in disease diagnosis, staging, and prognostic assessment. Owing to their extremely large image size, mainstream WSI classification methods typically follow the multiple instance learning (MIL) paradigm, in which a slide is divided into patches, patch representations are extracted by a feature encoder, and a MIL aggregator predicts the slide-level label from the resulting patch set [1]– [3]. Under this paradigm, the quality of patch representations critically affects downstream aggregation and determines the discriminability of slide-level predictions [3], [4].

Recent advances in pathology foundation models have substantially improved representation learning for histopathological images [5]–[7]. Large-scale pretrained models, such as UNI [5] and the Virchow series [6], are trained on extensive collections of histopathology images and have demonstrated strong capability in encoding pathological semantics. Consequently, using these models as frozen feature extractors has become a widely adopted strategy for WSI classification tasks [4]–[6]. However, since the encoder remains fixed during downstream training, the extracted patch representations may not be sufficiently adapted to a specific target dataset, particularly its task-relevant tissue patterns and discriminative characteristics [8], [9]. This limitation is particularly pronounced in data-scarce pathology domains that are underrepresented in pretraining data or exhibit substantial distribution shifts [6], [10], [11]. In such cases, the MIL head primarily optimizes slide-level aggregation over fixed patch representations, while the overall classification performance may remain limited by insufficient encoder-level adaptation [12]. These observations highlight the necessity for effective encoder adaptation to learn target-aware and pathology-discriminative patch representations for WSI classification.

Current encoder fine-tuning strategies for WSI classification face two main limitations. First, fine-tuning pathology foundation models is computationally intensive due to their large parameter scale and patch-intensive WSI training, while limited target-domain data may further increase the risk of overfitting [1], [5], [6], [13]. In contrast, lightweight vision transformers such as DINOv2-small [14] are easier to optimize and adapt, but lack the pathology-specific pretraining of foundation models [5], [6]. This creates a practical dilemma between pathology-specific representation capacity and adaptation efficiency. Moreover, standard encoders apply shared transformations to all patch tokens, which may be insufficient to capture heterogeneous tissue patterns requiring distinct feature mappings. This motivates model architectures that explicitly promote transformation diversity.

To this end, we propose a pathology role-guided mixtureof-experts (MoE) adaptation framework for efficient encoderlevel representation learning in low-resource pathology settings. During the first stage, we develop a novel sourcedomain distillation strategy using multi-cancer data to learn general pathology-aware representations. A frozen Virchow2 encoder [6] serves as the teacher to transfer pathology-specific priors through knowledge distillation [15], while DINOv2- small [14] is adopted as a lightweight student backbone. We further replace selected high-level feed-forward network (FFN) layers of the student encoder with MoE-based FFN modules, termed MoE-FFNs. Once optimized, the MoE-FFN blocks can serve as a transferable module across transformer-based encoders through lightweight dimensional alignment, without updating the entire backbone. To encourage meaningful expert specialization, we introduce role prototypes as weak pathological anchors, inspired by prototype-based representation learning [16]. These prototypes guide experts toward distinct tissue-related feature subspaces while discouraging redundant or pathology-irrelevant transformations [17], [18]. During the second stage, the initialized experts are refined through asymmetric prototype-guided optimization, which aligns positive patches with relevant role prototypes while separating confusable hard negatives from their nearest prototypes. After twostage training, the MoE-based encoder extracts patch representations offline, which are then aggregated by standard MIL models for slide-level classification. The main contributions of this work are summarized as follows:

1) We design a plug-and-play MoE-FFN module with role-prototype guidance to enable the encoder to learn specialized transformations for heterogeneous pathological patterns. It can replace selected FFN layers in transformer-based encoders while keeping the remaining backbone frozen, and using lightweight projections for dimensional alignment.

2) We develop a two-stage training strategy in which source-domain training enables the MoE-based encoder to learn pathology-aware representations and establish initial expert specialization, while target-domain finetuning adapts these representations to distinguish taskrelevant positive patterns from confusable negatives.

3) We validate the effectiveness of the proposed method on both public and private WSI datasets, demonstrating consistent performance gains and interpretable representation improvement across diverse settings.

## II. RELATED WORK

MIL for WSI Classification. Due to the gigapixel resolution of WSIs, most WSI classification methods follow the MIL paradigm [1], [2]. Early MIL methods used mean or max pooling, while ABMIL introduced attention weights to identify discriminative instances [2]. CLAM further used class-specific attention and instance-level clustering [4], and TransMIL modeled correlations among patches with selfattention [3]. Although MIL aggregators have been widely developed, their performance still depends heavily on patchlevel representations. When frozen encoder features are not well adapted to the target task, improving only the aggregation module may still remain limited by input feature quality [12].

Foundation Models for Patch Representation. Pathology foundation models have advanced patch representation learning by pretraining on large-scale histopathology images and capturing rich tissue morphology, cellular structures, and pathological semantics [5], [6], [19]. Representative models include H&E-based encoders such as UNI [5] and Virchow [6], vision-language models such as CONCH [7], and recent patchlevel, slide-level, or multimodal models such as CTransPath, GigaPath, and TITAN [19]–[21]. In WSI classification, these models are often used as frozen feature encoders to reduce computational cost and overfitting risk on small datasets [5], [6]. However, fixed encoder representations may be insufficiently adapted to target specific tissue composition, discriminative cues, and confusing patterns, especially under finegrained lesions or distribution shifts [12], [22]. This motivates lightweight and stable optimization of encoder representations for WSI classification.

Encoder Optimization. A direct way to optimize encoders is full or partial fine-tuning, but this is costly for large pathology foundation models and may overfit on limited WSI data. Parameter-efficient fine-tuning (PEFT) methods provide lightweight alternatives, including adapters [23] and LoRA [24]. While widely studied in general vision and language tasks, these methods do not explicitly model tissue heterogeneity and diverse local morphology in pathology images. Pathology-specific methods have also been proposed. R<sup>2</sup>T re-embeds extracted patch features to improve the use of frozen pathology foundation model features in WSI tasks [12]. HiAdapter introduces stain-invariant and morphology-aware adapters with a pathology-prototypical contrastive loss for generalization to unseen cancers and stains [22]. Different from these fine-tuning, PEFT, and post-encoder refinement methods, this work focuses on token transformation inside the encoder, where expert specialization occurs during patch representation formation.

Mixture of Experts in Vision and Pathology. MoE improves model capacity through conditional computation by routing inputs to different expert modules [25]. In transformers, MoE is commonly introduced by replacing the FFNs with multiple expert FFNs and sparse routing, as in sparse-gated MoE and V-MoE [26], [27]. However, MoE training often faces expert collapse, load imbalance, routing instability, and unclear expert specialization [26], [27]. These issues are particularly relevant to pathology images, which contain strong tissue heterogeneity and diverse local morphological patterns [28], [29]. MoE has also been used in computational pathology, mainly in WSI-level MIL aggregation, multi-task prediction, and multi-modal diagnosis. PAMoE proposes a plug-and-play pathology-aware MoE module for tissue-specific feature modeling in MIL pipelines [28]. M4 introduces multi-gate MoE into histopathology multitask MIL for predicting multiple genetic mutations from a single WSI [29]. Gated MoE has been used for multi-modal pathology diagnosis by combining WSI and flow cytometry information [30]. These studies show the value of multiexpert structures for modeling tissue heterogeneity, multi-task correlations, or multi-modal evidence. However, representative pathology MoE methods mainly operate after patch features have been extracted [28]–[30]. In contrast, this work introduces MoE into high-level transformer FFN transformation for encoder-level representation formation, making it complementary to MIL-level MoE methods.

## III. METHOD

## A. Overview

To transfer pathology-aware knowledge to a lightweight MoE-based encoder and establish expert-specific transformations for downstream adaptation, we design a role-guided MoE framework, as illustrated in Fig. 1. The framework consists of two stages of encoder-level representation learning, namely source-domain expert initialization and target-domain finetuning, followed by offline WSI classification. The source stage initializes the MoE parameters using source-domain data covering multiple cancer types, enabling the model to capture general pathological patterns and establish initial expert differentiation. The target stage fine-tunes these parameters on each downstream dataset to learn task-specific patch representations. After fine-tuning, the fixed MoE-based encoder extracts patch features, and the extracted features are aggregated by a standard MIL model for slide-level classification.

1) Problem Formulation: Let D denote the data used for encoder representation learning, and let $F _ { \theta , \phi }$ denote the MoE-based encoder, with frozen backbone parameters $\theta$ and trainable MoE-related parameters ϕ. Formally, the encoder representation learning problem aims to preserve pathological semantic priors while improving discrimination for the target WSI classification task. The optimal parameters are learned through the overall representation learning objective ${ \mathcal { L } } _ { \mathrm { r e p } } ;$

$$
\phi ^ { * } = \operatorname* { m i n } _ { \phi } \mathcal { L } _ { \mathrm { r e p } } \left( F _ { \theta , \phi } ; \mathcal { D } \right) .\tag{1}
$$

Given the i-th WSI $S _ { i }$ , it is divided into a patch bag $X _ { i } =$ $\{ x _ { i , j } \} _ { j = 1 } ^ { M _ { i } }$ containing $M _ { i }$ non-overlapping patches. After representation learning, each patch is encoded as

$$
z _ { i , j } = F _ { \theta , \phi ^ { * } } ( x _ { i , j } ) , \qquad j = 1 , \ldots , M _ { i } .\tag{2}
$$

After representation learning, the encoder is fixed, and WSI classification is optimized over the MIL model $G _ { \eta }$

$$
\operatorname* { m i n } _ { \eta } \sum _ { i = 1 } ^ { N } \mathcal { L } _ { \mathrm { c l s } } \left( G _ { \eta } \left( \{ z _ { i , j } \} _ { j = 1 } ^ { M _ { i } } \right) , y _ { i } \right) .\tag{3}
$$

## B. Source-Domain Expert Initialization

Source-domain expert initialization aims to build a pathology-aware expert structure before target-specific finetuning. By transferring general pathological priors into the MoE-FFN blocks and encouraging differentiated expert transformations, this stage provides a stable initialization for subsequent adaptation to data-limited target WSI tasks.

## 1) MoE-FFN Encoder Architecture:

a) MoE-FFN Integration and Expert Composition: Standard FFNs use a single shared transformation for all tokens, which limits their ability to model heterogeneous tissue patterns. To overcome this limitation, we introduce MoE-FFNs that provide expert-specific transformations with conditional token routing. To exploit this design while preserving the pretrained representation capacity, we replace the original FFNs in selected highlevel transformer blocks $\mathcal { L } _ { \mathrm { m o e } }$ with MoE-FFNs and keep the remaining backbone parameters frozen. We focus on high-level blocks because their representations are more likely to encode higher-order pathological morphology and tissue architecture, whereas earlier blocks mainly capture generic visual cues such as color, edges, and local textures. Accordingly, only the MoE-related components in $\mathcal { L } _ { \mathrm { m o e } }$ , including expert FFNs, routing and normalization parameters, are optimized, focusing trainable capacity on high-level semantic transformation.

Each MoE-FFN contains one always-active shared expert for common feature transformation and K routed experts that provide specialized transformations for heterogeneous tissue patterns according to token-expert compatibility. For token $h _ { i } ^ { l }$ in the l-th MoE layer, the output is formulated as

$$
\mathrm { M o E } ^ { l } ( h _ { i } ^ { l } ) = \alpha E _ { s } ^ { l } ( h _ { i } ^ { l } ) + \sum _ { k = 1 } ^ { K } g _ { i k } ^ { l } E _ { k } ^ { l } ( h _ { i } ^ { l } ) ,\tag{4}
$$

where $E _ { s } ^ { l }$ and $E _ { k } ^ { l }$ denote the shared expert and the k-th routed expert, respectively; α controls the contribution of the shared expert, and $g _ { i k } ^ { l }$ denotes the sparse routing weight defined in the following Routing Strategy paragraph.

b) Routing Strategy: Since a patch token may encode multiple tissue patterns, we adopt a top-any routing strategy [31] that dynamically activates at most K experts according to token-level complexity. For token $h _ { i } ^ { l }$ in the l-th MoE layer, each routed expert $E _ { k } ^ { \bar { l } }$ is represented by a learnable routing vector $v _ { k } ^ { l }$ . Let $s _ { i k } ^ { l }$ denote their cosine similarity score with a learnable scale. Each expert additionally has a learnable threshold $\tau _ { k } ^ { l } .$ , and the adjusted routing score is defined as

$$
q _ { i k } ^ { l } = s _ { i k } ^ { l } - \tau _ { k } ^ { l } .\tag{5}
$$

Let $k _ { i , a } ^ { l }$ denote the expert with the a-th highest routing score for token i. We first keep the $\mathrm { \ t o p { - } } A _ { \mathrm { m a x } }$ candidate experts and then activate only those with sufficiently competitive responses:

$$
\mathcal { K } _ { i } ^ { l } = \{ k _ { i , a } ^ { l } \} _ { a = 1 } ^ { A _ { \mathrm { m a x } } } ,\tag{6}
$$

$$
\mathcal { S } _ { i } ^ { l } = \{ k _ { i , 1 } ^ { l } \} \cup \left\{ k _ { i , a } ^ { l } \in \mathcal { K } _ { i } ^ { l } \Big | a > 1 , \ q _ { i , k _ { i , a } ^ { l } } ^ { l } \geq \rho q _ { i , k _ { i , 1 } ^ { l } } ^ { l } \right\} .\tag{7}
$$

where $\rho$ specifies the minimum relative score required for activating an additional expert. Consequently, $| S _ { i } ^ { l } | \in$ $\{ 1 , \ldots , A _ { \mathrm { m a x } } \}$ , bounding computation while allowing adaptive routing. The routing weights are normalized as:

$$
\{ g _ { i k } ^ { l } \} _ { k \in \mathcal { S } _ { i } ^ { l } } = \mathrm { s o f t m a x } \left( \{ q _ { i k } ^ { l } \} _ { k \in \mathcal { S } _ { i } ^ { l } } \right) , \quad g _ { i k } ^ { l } = 0 \mathrm { ~ f o r ~ } k \notin \mathcal { S } _ { i } ^ { l } .\tag{8}
$$

Eq. (4) uses the shared expert and selected routed experts.

![](images/436ddb0cee13102b023903e2ab346975a4dd90c704868abbac571f95085a7c4b.jpg)  
Fig. 1. Overview of the proposed role-guided MoE framework. (i) Source-domain expert initialization learns pathology-aware representations and initial expert specialization using a frozen Virchow2 teacher, source role prototypes, and MoE regularization. (ii) Target-domain fine-tuning refines the MoE-related parameters by scoring selected positive and hard-negative patches against target role prototypes. (iii) Offline WSI classification fixes the fine-tuned MoE-based encoder for patch representation extraction and MIL-based slide classification.

c) Plug-and-Play MoE Module and Backbone Compatibility: The two selected high-level transformer blocks equipped with MoE-FFNs constitute a plug-and-play MoE module. Once encoder-level optimization is completed, the resulting module can replace the corresponding high-level blocks in another transformer-based backbone. For backbones with different hidden dimensions, lightweight input and output projections align the backbone features with the module space. This modular design enables cross-backbone transfer without redesign.

2) Source Role Prototype Construction: To guide initial expert specialization, we construct a source role prototype bank in the teacher feature space, where each prototype serves as a weak pathological anchor for a recurring tissue pattern in the source data. During source-domain training, tokens routed to each role-guided expert are encouraged to align with the corresponding prototype, promoting expert differentiation without assigning fixed tissue identities.

Since tissue-level annotations are typically unavailable in WSI datasets, role-specific candidate patches cannot be directly identified from annotated regions. We therefore employ CONCH [7], a pathology vision-language model with strong image-text alignment, to assign patch-level labels and relevance scores. For the source data, we define a prompt bank $\tau ^ { \mathrm { s r c } }$ containing R tissue roles. Given a source patch $x _ { n } ^ { \mathrm { s r c } }$ , CONCH produces a predicted role label $\hat { r } _ { n } ^ { \mathrm { s r c } }$ and the corresponding confidence score $s _ { n } ^ { \mathrm { s r c } }$

$$
\left( \hat { r } _ { n } ^ { \mathrm { s r c } } , s _ { n } ^ { \mathrm { s r c } } \right) = \mathrm { C O N C H } \left( x _ { n } ^ { \mathrm { s r c } } ; T ^ { \mathrm { s r c } } \right) .\tag{9}
$$

For each role, patches are ranked according to their confidence scores $s _ { n } ^ { \mathrm { s r c } }$ , and only the top-ranked fraction is retained to reduce the risk of introducing inaccurate pathological priors into prototype construction. The selected patches are further balanced across source organs to prevent a single organ from dominating the resulting prototype. The candidate set for role r is denoted by $\mathcal { H } _ { r } ^ { \mathrm { s r c } }$

For each candidate patch $x _ { n } ^ { \mathrm { s r c } } ~ \in ~ \mathcal { H } _ { r } ^ { \mathrm { s r c } }$ , let $t _ { n , j } ^ { \mathrm { T } }$ denote its j-th patch-token feature extracted from layer ℓ of the frozen teacher encoder. The patch-level feature $f _ { n } ^ { \mathrm { T } }$ is obtained by mean-pooling its J patch-token features: $\begin{array} { r l } { f _ { n } ^ { \mathrm { T } } } & { { } = } \end{array}$ MeanPool $( \{ t _ { n , j } ^ { \mathrm { T } } \} _ { j = 1 } ^ { J } )$ .

To reduce the influence of noisy or morphologically atypical candidates, the L2-normalized teacher features $\{ f _ { n } ^ { \mathrm { T } } \}$ of each role are clustered using K-means. Let $\boldsymbol { \mathcal { T } } _ { r } ^ { * }$ denote the indices of the largest cluster containing at least $m _ { \mathrm { m i n } }$ candidates. The source prototype of role r is then computed as

$$
p _ { r } ^ { \mathrm { s r c } } = \frac { 1 } { | \mathcal { T } _ { r } ^ { * } | } \sum _ { n \in \mathcal { T } _ { r } ^ { * } } f _ { n } ^ { \mathrm { T } } .\tag{10}
$$

When no valid cluster is available, all candidate features of that role are averaged. Then, the source role prototype bank is defined as

$$
P ^ { \mathrm { s r c } } = \{ p _ { r } ^ { \mathrm { s r c } } \} _ { r = 1 } ^ { R } .\tag{11}
$$

3) Source-Domain Training Objectives: To impose sourcedomain supervision on the MoE-based encoder, we use DINOv2-small [14] as a compact student encoder and a frozen Virchow2 encoder [6] as the pathology-aware teacher. The teacher also defines the feature space of the source role prototype bank $P ^ { \mathrm { s r c } }$ . To resolve the teacher-student dimension mismatch, a projection head maps student features to the teacher feature space before supervision. The objective combines teacher-student distillation, role prototype supervision, and MoE auxiliary losses.

a) Teacher-Student Distillation Loss: To transfer pathologyaware global (i.e., CLS tokens) and patch-level (i.e., patch tokens) representations, we align the final-layer token features of the student encoder and the frozen teacher, as features from this layer directly form the encoder output, while the student features also capture the representation changes produced by the optimized MoE blocks. Let $z _ { \mathrm { c l s } } ^ { \mathrm { S } }$ and $\{ z _ { i } ^ { \mathrm { S } } \} _ { i = 1 } ^ { N }$ denote the student CLS and patch tokens, and le $z _ { \mathrm { c l s } } ^ { \mathrm { T } }$ and $\{ z _ { i } ^ { \mathrm { T } } \} _ { i = 1 } ^ { N }$ denote the corresponding teacher tokens. The projection head $W _ { p }$ maps both $z _ { \mathrm { c l s } } ^ { \mathrm { T } }$ and $\{ z _ { i } ^ { \mathrm { T } } \} _ { i = 1 } ^ { N }$ into the teacher feature space. The projected student tokens are denoted by $\tilde { z } ^ { \mathrm { S } }$

Then, we align the student and teacher CLS tokens using the cosine distance loss $\mathcal { L } _ { \mathrm { c l s } } = 1 - \cos ( \tilde { z } _ { \mathrm { c l s } } ^ { \mathrm { S } } , z _ { \mathrm { c l s } } ^ { \mathrm { T } } )$ to preserve global semantic consistency.

To encourage the student to infer pathology-aware tissue representations from surrounding context rather than relying only on directly visible patch content, we apply block masking to the student tokens while the teacher processes the complete image. Let M and U denote the masked and unmasked patchtoken index sets, respectively, with alignment losses

$$
\mathcal { L } _ { \mathrm { m a s k } } = \frac { 1 } { \left| \mathcal { M } \right| } \sum _ { i \in \mathcal { M } } \mathrm { S m o o t h L 1 } \left( \tilde { z } _ { i } ^ { \mathrm { S } } , z _ { i } ^ { \mathrm { T } } \right) ,\tag{12}
$$

$$
\mathcal { L } _ { \mathrm { u n m a s k } } = \frac { 1 } { \left| \mathcal { U } \right| } \sum _ { i \in \mathcal { U } } \mathrm { S m o o t h L 1 } \left( \tilde { z } _ { i } ^ { \mathrm { S } } , z _ { i } ^ { \mathrm { T } } \right) .\tag{13}
$$

Beyond individual token alignment, we randomly sample token pairs Ω. For each pair in Ω, we encourage the difference between the two student token features to approximate that between the corresponding teacher token features:

$$
\mathcal { L } _ { \mathrm { r e l } } = \frac { 1 } { \left| \Omega \right| } \sum _ { ( i , j ) \in \Omega } \left. \left( \tilde { z } _ { i } ^ { \mathrm { S } } - \tilde { z } _ { j } ^ { \mathrm { S } } \right) - \left( z _ { i } ^ { \mathrm { T } } - z _ { j } ^ { \mathrm { T } } \right) \right. _ { 2 } ^ { 2 } .\tag{14}
$$

The complete distillation objective is

$$
\mathcal { L } _ { \mathrm { d i s t i l l } } = \lambda _ { \mathrm { c l s } } \mathcal { L } _ { \mathrm { c l s } } + \lambda _ { \mathrm { m a s k } } \mathcal { L } _ { \mathrm { m a s k } } + \lambda _ { \mathrm { u n m a s k } } \mathcal { L } _ { \mathrm { u n m a s k } } + \lambda _ { \mathrm { r e l } } \mathcal { L } _ { \mathrm { r e l } } ,\tag{15}
$$

b) Role Prototype Loss: To promote pathology-relevant expert differentiation, we associate R of the K routed experts with the R source role prototypes and retain one unconstrained free expert, such that $K = R + 1$ . The free expert captures patterns beyond the predefined roles.

As illustrated in Fig. 2, role supervision is applied to a token from the last block in $\mathcal { L } _ { \mathrm { m o e } }$ only when a role-guided expert receives the highest routing weight for that token. After projection into the teacher feature space, the similarity score $s _ { i , r }$ between token i and source role prototype r is computed as their cosine similarity. The role prototype probabilities are computed as $\mathbf { a } _ { i } = \mathrm { s o f t m a x } ( \mathbf { s } _ { i } / \tau )$ , where τ is the temperature.

To reduce supervision noise, we retain only tokens with confident prototype assignments, determined by the highest prototype probability and its margin over the second highest. Let $\mathcal { V } _ { r } ^ { \mathrm { s r c } }$ denote the high-confidence tokens dominated by the role-guided expert associated with role r.

The role target loss aligns each selected token with its corresponding role prototype:

$$
\mathcal { L } _ { \mathrm { t a r g e t } } ^ { ( r ) } = \mathbb { E } _ { i \in \mathcal { V } _ { r } ^ { \mathrm { s r c } } } \left[ - \log a _ { i , r } \right] .\tag{16}
$$

![](images/3bea0f8a875b38f9ac5b4ea05e4f236bba665626514d0a221225d0f96ff898ea.jpg)  
Fig. 2. Role prototype-guided supervision. Left: candidate selection and prototype construction. Right: token alignment with role prototypes.

The role attraction loss pulls each selected token toward its corresponding role prototype:

$$
\mathcal { L } _ { \mathrm { a t t r } } ^ { ( r ) } = \mathbb { E } _ { i \in \mathcal { V } _ { r } ^ { \mathrm { s r c } } } \left[ 1 - s _ { i , r } \right] .\tag{17}
$$

Let $s _ { i , \lnot r } = \operatorname* { m a x } _ { q \ne r } s _ { i , q }$ denote the highest similarity between token i and all role prototypes except its corresponding prototype r. The role separation loss requires the similarity to prototype r to exceed this value by a margin m:

$$
\mathcal { L } _ { \mathrm { s e p } } ^ { ( r ) } = \frac { 1 } { \vert \mathcal { V } _ { r } ^ { \mathrm { s r c } } \vert } \sum _ { i \in \mathcal { V } _ { r } ^ { \mathrm { s r c } } } \operatorname* { m a x } \left( 0 , m + s _ { i , \lnot r } - s _ { i , r } \right) .\tag{18}
$$

Let $w _ { r } = | \mathcal { V } _ { r } ^ { \mathrm { s r c } } | / \textstyle \sum _ { q = 1 } ^ { R } | \mathcal { V } _ { q } ^ { \mathrm { s r c } } |$ denote the proportion of highconfidence tokens associated with role r. The complete source role prototype loss is

$$
\mathcal { L } _ { \mathrm { r o l e } } ^ { \mathrm { s r c } } = \sum _ { r = 1 } ^ { R } w _ { r } \left( \lambda _ { t , r } \mathcal { L } _ { \mathrm { t a r g e t } } ^ { ( r ) } + \lambda _ { a , r } \mathcal { L } _ { \mathrm { a t t r } } ^ { ( r ) } + \lambda _ { s , r } \mathcal { L } _ { \mathrm { s e p } } ^ { ( r ) } \right) .\tag{19}
$$

c) MoE Auxiliary Loss: To stabilize expert routing, we use four auxiliary losses. Following the principle in sparse MoE models [26], [32], the load balancing loss $\mathcal { L } _ { \mathrm { b a l } }$ promotes balanced token assignment and routing weights across routed experts. Following DynMoE [31], the diversity loss ${ \mathcal { L } } _ { \mathrm { d i v } }$ separates the normalized routing vectors. We further introduce two losses to promote reliable and sparse expert routing. The coverage loss ensures that each token receives a strong response from at least one routed expert. The sparsity loss controls routing complexity by constraining the average number of activated experts per token:

$$
\mathcal { L } _ { \mathrm { c o v } } = \frac { 1 } { | \mathcal { T } | } \sum _ { i \in \mathcal { T } } \operatorname* { m a x } ( 0 , c - \operatorname* { m a x } _ { k } s _ { i , k } ) ,\tag{20}
$$

$$
\begin{array} { r } { \mathcal { L } _ { \mathrm { s p r } } = \left( \mathbb { E } _ { i \in \mathcal { T } } [ a _ { i } ] - a ^ { * } \right) ^ { 2 } . } \end{array}\tag{21}
$$

where c is the minimum routing score threshold, and $a ^ { * }$ is the target average number of activated experts per token.

The four auxiliary terms are computed for each MoE layer and then averaged across all selected MoE layers. The resulting MoE auxiliary loss is

$$
\mathcal { L } _ { \mathrm { a u x } } = \lambda _ { \mathrm { d i v } } \mathcal { L } _ { \mathrm { d i v } } + \lambda _ { \mathrm { b a l } } \mathcal { L } _ { \mathrm { b a l } } + \lambda _ { \mathrm { c o v } } \mathcal { L } _ { \mathrm { c o v } } + \lambda _ { \mathrm { s p r } } \mathcal { L } _ { \mathrm { s p r } } .\tag{22}
$$

d) Overall Source Objective: The source-domain training jointly transfers pathology-aware representations from the teacher, promotes role-guided expert differentiation, and stabilizes expert routing. The overall source objective is

$$
\mathcal { L } _ { s r c } = \lambda _ { d i s t i l l } \mathcal { L } _ { d i s t i l l } + \lambda _ { r o l e } ^ { s r c } \mathcal { L } _ { r o l e } ^ { s r c } + \lambda _ { a u x } \mathcal { L } _ { a u x } .\tag{23}
$$

where the three coefficients balance the distillation, role prototype, and MoE auxiliary losses.

## C. Target-Domain Fine-Tuning

The source-initialized experts may not fully match the tissue composition and discriminative cues of a specific downstream task, especially when positive patterns resemble confusing negative regions. Starting from the source-initialized encoder, we construct a task-specific target role prototype bank $P ^ { \mathrm { t g t } }$ for each downstream dataset. We then further optimize the MoErelated parameters to enhance target-specific discrimination while preserving the source-initialized representations.

1) Target Role Prototype Construction: Following the procedure described in Sec. III-B.2, we construct a task-specific target role prototype bank

$$
P ^ { \mathrm { t g t } } = \left\{ p _ { r } ^ { \mathrm { t g t } } \right\} _ { r = 1 } ^ { R }\tag{24}
$$

from representative patches in the corresponding target dataset.

We designate the prototype corresponding to the tissue pattern indicative of the positive class as $r ^ { + }$ , and define $\mathcal { R } _ { \mathrm { c o m p } }$ as the set of all remaining role prototypes. These remaining prototypes serve as competing references to the positive role.

2) Positive and Hard-Negative Candidate Selection: Since slide-level labels provide no patch-level localization, applying role supervision to all patches may introduce noise. We therefore mine positive and hard-negative patches for targetdomain fine-tuning.

Using the same projection and similarity computation as in source-domain role supervision, we obtain the similarity $s _ { i } ^ { r }$ and role probability for each target patch i and target prototype r. We use the positive role gap to measure how strongly patch i favors the positive role over the competing roles:

$$
g _ { i } = s _ { i } ^ { r ^ { + } } - \operatorname* { m a x } _ { r \in \mathscr { R } _ { \mathrm { c o m p } } } s _ { i } ^ { r } .\tag{25}
$$

Let $B ^ { + }$ and $B ^ { - }$ denote the sets of positive and negative slides, respectively. For $b \in B ^ { + }$ , let $\Omega _ { b } ^ { + }$ contain patches with sufficiently high positive-role probability and role gap. For $b \in$ $B ^ { - }$ , let $\Omega _ { b } ^ { - }$ contain patches with strong positive-role responses but limited neighborhood support. Let $q _ { i } ^ { \mathrm { r o l e } }$ measure direct positive-role evidence, $q _ { i } ^ { \mathrm { c t x } }$ measure neighborhood-supported positive evidence, and $q _ { i } ^ { \mathrm { h a r d } }$ measure the likelihood that a negative patch is confused with the positive pattern. The initial candidate pools, which define the search space for targetdomain fine-tuning, are constructed as

$$
\mathcal { C } _ { b } ^ { + } = \mathrm { T o p K } _ { i \in \Omega _ { b } ^ { + } } \left( q _ { i } ^ { \mathrm { r o l e } } \right) \cup \mathrm { T o p K } _ { i \in \Omega _ { b } ^ { + } } \left( q _ { i } ^ { \mathrm { c t x } } \right) , \quad b \in \mathcal { B } ^ { + } ,\tag{26}
$$

$$
\mathcal { C } _ { b } ^ { - } = \mathrm { T o p K } _ { i \in \Omega _ { b } ^ { - } } \left( q _ { i } ^ { \mathrm { h a r d } } \right) , \quad b \in \mathcal { B } ^ { - } .\tag{27}
$$

3) Target-Domain Training Objectives: Target-domain finetuning combines patch-level discrimination, slide-level consistency, and representation preservation. For each slide $b ,$ let $\mathcal { C } _ { b }$ denote its pre-constructed candidate pool, where $\mathcal { C } _ { b } = \mathcal { C } _ { b } ^ { + }$ for a positive slide and $\mathcal { C } _ { b } = \mathcal { C } _ { b } ^ { - }$ for a negative slide. During fine-tuning, the current encoder re-scores the patches in $\mathcal { C } _ { b }$ and dynamically selects a subset $\boldsymbol { S } _ { b } \subseteq \boldsymbol { \mathcal { C } } _ { b }$

a) Asymmetric Role Loss: The asymmetric role loss increases the positive role gap for positive candidates while suppressing it for hard-negative candidates. Let $S ^ { + }$ and $S ^ { - }$ denote the selected candidates from positive and negative slides, respectively. Let $m _ { \mathrm { p o s } }$ and $- m _ { \mathrm { n e g } }$ denote the desired lower and upper bounds of the positive role $\mathrm { g a p }$ for positive and hard-negative candidates. Let $\omega _ { i }$ denote the confidence weight derived from the role response and neighborhood context of candidate i. The loss is defined as

$$
\begin{array} { r } { \mathcal { L } _ { \mathrm { a s y m } } = \frac { \sum _ { i \in \mathcal { S } ^ { + } } \omega _ { i } \operatorname* { m a x } \left( 0 , m _ { \mathrm { p o s } } - g _ { i } \right) } { \sum _ { i \in \mathcal { S } ^ { + } } \omega _ { i } } } \\ { + \frac { \sum _ { i \in \mathcal { S } ^ { - } } \omega _ { i } \operatorname* { m a x } \left( 0 , m _ { \mathrm { n e g } } + g _ { i } \right) } { \sum _ { i \in \mathcal { S } ^ { - } } \omega _ { i } } . } \end{array}\tag{28}
$$

b) Pairwise Ranking Loss: The ranking loss encourages positive candidates to have larger role gaps than negative candidates. Let $\bar { g } _ { K } ^ { + }$ and $\bar { g } _ { K } ^ { - }$ denote the mean top-K role gaps of the selected positive and negative candidates, and let $m _ { \mathrm { r a n k } }$ denote the ranking margin. The loss is defined as

$$
\mathcal { L } _ { \mathrm { r a n k } } = \operatorname* { m a x } \left( 0 , m _ { \mathrm { r a n k } } - \bar { g } _ { K } ^ { + } + \bar { g } _ { K } ^ { - } \right) .\tag{29}
$$

c) Slide-Level Proxy Loss: The slide-level proxy loss aligns the strongest patch-level evidence with the corresponding slide label. For each slide $b ,$ we rank the candidates in $\boldsymbol { S _ { b } }$ by their role gaps and retain the top $K _ { b }$ candidates as ${ \mathcal { T } } _ { b } .$ , where $K _ { b } =$ min(K, $\left| S _ { b } \right| )$ . Let $\bar { g } _ { b }$ denote the mean role gap of the $\mathrm { t o p } { - } K _ { b }$ candidates in slide b. Let $y _ { b }$ denote the slide label and $B$ the number of slides. The loss is defined as

$$
\mathcal { L } _ { \mathrm { p r o x y } } = \frac { 1 } { B } \sum _ { b = 1 } ^ { B } \ell _ { \mathrm { B C E } } \left( \boldsymbol { \sigma } ( \bar { g } _ { b } ) , \boldsymbol { y } _ { b } \right) .\tag{30}
$$

d) Feature Preservation Loss: The feature preservation loss limits representation drift during target-domain fine-tuning. For each patch in the mined candidate pool, it minimizes the cosine distance between the representation $h _ { b , i } ^ { \mathrm { t g t } }$ produced by the current encoder and the reference representation $h _ { b , i } ^ { \mathrm { s r c } }$ produced by the frozen source-initialized encoder:

$$
\mathcal { L } _ { \mathrm { p r e s } } = \mathbb { E } _ { b } \mathbb { E } _ { i \in \mathcal { C } _ { b } } \left[ 1 - \cos \left( h _ { b , i } ^ { \mathrm { t g t } } , h _ { b , i } ^ { \mathrm { s r c } } \right) \right] .\tag{31}
$$

e) Overall Target Objective: The target-domain objective jointly improves patch-level role discrimination, preserves slide-level label consistency, and limits representation drift from the source-initialized encoder. The overall target-domain fine-tuning objective is

$$
\begin{array} { r } { \mathcal { L } _ { \mathrm { t g t } } = \lambda _ { \mathrm { a s y m } } \mathcal { L } _ { \mathrm { a s y m } } + \lambda _ { \mathrm { r a n k } } \mathcal { L } _ { \mathrm { r a n k } } + \lambda _ { \mathrm { p r o x y } } \mathcal { L } _ { \mathrm { p r o x y } } + \lambda _ { \mathrm { p r e s } } \mathcal { L } _ { \mathrm { p r e s } } . } \end{array}\tag{32}
$$

## IV. EXPERIMENTS AND RESULTS

## A. Experimental Setup

1) Datasets: The study uses source-domain data for expert initialization and two target-domain WSI datasets for downstream evaluation. Source-domain training includes diagnostic H&E-stained WSIs from TCGA-BRCA, TCGA-COAD, TCGA-KIRC, and TCGA-LUAD [33]. To reduce cancer-type imbalance, we sampled equal numbers of tumor and non-tumor slides from each cohort. HISTAI-SPIDER, a public multiorgan patch dataset, was additionally included to increase the diversity of tissue patterns during source-domain training [34].

Target-domain evaluation was conducted on the public BRACS dataset and a private PAROTID dataset. For BRACS [35], we followed the official WSI-level split and used the Group BT versus Group AT binary classification task, where Group BT denotes benign tumors and Group AT denotes atypical tumors. This task involves fine-grained breast lesions with confusing morphological patterns.

The private PAROTID dataset contains 195 H&E-stained WSIs of parotid tumors collected from Shenzhen People’s Hospital, Guangdong Province, for benign versus malignant classification. Slide-level labels were obtained from pathological diagnoses and reviewed by experienced pathologists, and all WSIs were de-identified before analysis. The dataset was divided into training, validation, and test sets at the patient level with a ratio of 0.70, 0.15, and 0.15, ensuring that all slides from the same patient remained in the same split.

2) Evaluation Metrics: For all datasets, we use the Area Under Curve (AUC) and macro F1-score (F1) to evaluate classification performance. All methods are evaluated using the same data split and downstream training protocol. Each experiment is repeated with five random seeds, and the mean and standard deviation are reported.

3) Baselines and Implementation Details: We conduct experiments across five backbones, including UNI [5], UNI2- h [5], Virchow2 [6], DINOv2-small [14], and OpenCLIP ViT-B/16 [36], and two MIL aggregators, ABMIL [2] and TransMIL [3]. Under each backbone-MIL setting, we compare the proposed method with four baselines: (1) Frozen, which directly uses pretrained encoder for offline feature extraction without encoder updates; (2) Partial FT [13], which fine-tunes selected high-level encoder blocks; (3) LoRA [24], which applies low-rank adaptation with a limited number of trainable parameters; and (4) $\mathsf { R } ^ { 2 } \mathsf { T }$ [12], which re-embeds frozen patch features through a feature refinement module.

For the main DINOv2-small configuration, the FFNs in the 10th and 11th transformer blocks are replaced with MoE-FFNs. Each MoE-FFN contains four routed experts and one shared expert, with up to two routed experts activated for each token. AdamW is used as the optimizer for all encoder-level training stages. Source-domain initialization is performed for 15 epochs with a weight decay of 0.05; the learning rate is initialized at $1 \times 1 0 ^ { - 4 }$ and reduced to $5 \times 1 0 ^ { - 5 }$ for the final 5 epochs. Target-domain fine-tuning is performed for 10 epochs with a learning rate of $2 \times 1 0 ^ { - 5 }$ and a weight decay of $1 \times 1 0 ^ { - 4 }$

For downstream WSI classification, each encoder extracts offline features from 1024 patches per WSI, which are subsequently aggregated by ABMIL or TransMIL. The hyperparameters of Partial FT and LoRA were selected through validation experiments, and the best-performing configurations were used for comparison. $\mathsf { R } ^ { 2 } \mathsf { T }$ was implemented using the recommended settings from the original paper. Under the same backbone-MIL setting, all encoder variants use identical downstream architectures and training protocols.

## B. Downstream WSI Classification Results

Table I reports the AUC and F1 scores on PAROTID and BRACS across all backbone-MIL settings. Overall, our method achieves the best F1 score in 19 of the 20 settings and ranks first or second in AUC in 19 settings, showing consistent improvements across backbones and MIL configurations.

On PAROTID, where only limited target-domain training data are available, our method obtains the best F1 score in nine of the ten settings, indicating stable improvement in a low-data setting. The only exception is DINOv2-small with ABMIL, where Partial FT achieves an F1 score of 0.774 compared with 0.763 for our method. Nevertheless, our method achieves the highest AUC of 0.890 in this setting, improving upon Partial FT by 0.058. In comparison, the relative performance of the baseline strategies varies more noticeably across backbones. On BRACS, our method achieves the best F1 score in all ten settings. The largest improvement is observed for UNI2- h with ABMIL, where the F1 score increases from 0.612 for the best competing method to 0.722 for our method. These results suggest that the proposed method produces more discriminative patch representations for the fine-grained classification of benign and atypical lesions.

Finally, the similar performance gains observed with both ABMIL and TransMIL suggest that the improvements mainly arise from enhanced patch representations rather than dependence on a particular MIL model.

## C. Ablation Studies

To analyze the contribution of each component, we conduct ablation studies on UNI, as shown in Table II.

MoE architecture. Adding a randomly initialized MoE improves the F1 score of Frozen UNI from 0.725 to 0.791 on PAROTID and from 0.546 to 0.585 on BRACS. This result shows that the additional expert branches provide useful task-specific transformation capacity. However, random MoE remains below the full model by 0.069 and 0.108 in F1 on the two datasets, respectively, indicating that the MoE architecture alone does not account for the overall improvement.

Distillation. The full model achieves the highest AUC and F1 score on both datasets. Among the ablated variants, removing distillation causes the largest performance reduction.

TABLE I  
MAIN COMPARISON ACROSS BACKBONES AND MIL AGGREGATORS. AUC AND F1 ARE REPORTED FOR EACH DATASET. BOLD AND UNDERLINED VALUES DENOTE THE BEST AND SECOND-BEST RESULTS WITHIN EACH BACKBONE-AGGREGATOR SETTING.
<table><tr><td rowspan="3">Backbone Method</td><td rowspan="3"></td><td colspan="4">ABMIL</td><td colspan="4">TransMIL</td></tr><tr><td colspan="2">PAROTID</td><td colspan="2">BRACS</td><td colspan="2">PAROTID</td><td colspan="2">BRACS</td></tr><tr><td>AUC</td><td>F1</td><td>AUC</td><td>F1</td><td>AUC</td><td>F1</td><td>AUC</td><td>F1</td></tr><tr><td rowspan="5">UNI</td><td>Frozen</td><td> $0 . 8 9 8 \pm 0 . 0 3 1$ </td><td> $0 . 7 2 5 \pm 0 . 0 9 7$ </td><td> $0 . 6 4 9 \pm 0 . 0 6 8$ </td><td> $0 . 5 4 6 \pm 0 . 1 4 3$ </td><td> $0 . 9 5 7 \pm 0 . 0 1 9$ </td><td> $0 . 8 7 5 \pm 0 . 0 3 0$ </td><td>0.681 ± 0.054</td><td>0.627 ± 0.091</td></tr><tr><td>Partial FT</td><td> $0 . 9 0 3 \pm 0 . 0 5 9$ </td><td> $0 . 8 1 9 \pm 0 . 0 3 8$ </td><td> $0 . 5 1 4 \pm 0 . 0 0 8$ </td><td> $0 . 5 4 0 \pm 0 . 0 3 6$ </td><td> $0 . 9 6 7 \pm 0 . 0 1 8$ </td><td> $0 . 8 5 3 \pm 0 . 0 8 5$ </td><td> $0 . 4 5 6 \pm 0 . 0 5 7$ </td><td>0.380 ± 0.129</td></tr><tr><td>LoRA</td><td> $0 . 9 2 5 \pm 0 . 0 2 9$ </td><td> $\underline { { 0 . 8 2 2 \pm 0 . 0 4 3 } }$ </td><td> $0 . 6 5 1 \pm 0 . 0 6 8$ </td><td> $0 . 5 5 1 \pm 0 . 1 4 2$ </td><td> $\underline { { 0 . 9 7 0 \pm 0 . 0 1 8 } }$ </td><td> $0 . 8 4 5 \pm 0 . 0 6 3$ </td><td> $0 . 6 2 5 \pm 0 . 0 6 6$ </td><td>0.585 ± 0.049</td></tr><tr><td>R2T</td><td> $\mathbf { 0 . 9 6 2 \pm 0 . 0 0 5 }$ </td><td> $0 . 7 8 7 \pm 0 . 0 1 9$ </td><td> $0 . 7 3 0 \pm 0 . 0 0 8$ </td><td> $0 . 6 7 4 \pm 0 . 0 1 4$ </td><td> $0 . 9 6 1 \pm 0 . 0 1 6$ </td><td> $0 . 9 0 2 \pm 0 . 0 1 5$ </td><td> $0 . 7 0 0 \pm 0 . 0 2 7$ </td><td> $0 . 6 0 8 \pm 0 . 0 2 4$ </td></tr><tr><td>Ours</td><td> $0 . 9 5 4 \pm 0 . 0 2 6$ </td><td></td><td>0.860 ± 0.034 0.765 ± 0.026</td><td></td><td></td><td>0.693 ± 0.045 0.982 ± 0.010 0.933 ± 0.015</td><td>0.720 ± 0.030</td><td>0.636 ± 0.066</td></tr><tr><td rowspan="6">UNI2-h</td><td>Frozen</td><td> $0 . 9 6 4 \pm 0 . 0 1 5$ </td><td> $0 . 8 7 1 \pm 0 . 0 7 8$ </td><td> $0 . 7 5 1 \pm 0 . 0 1 8$ </td><td> $0 . 6 1 2 \pm 0 . 0 3 1$ </td><td> $0 . 9 7 0 \pm 0 . 0 2 0$ </td><td> $0 . 8 4 0 \pm 0 . 0 2 8$ </td><td> $0 . 6 3 9 \pm 0 . 0 4 1$ </td><td> $\underline { { 0 . 5 2 6 \pm 0 . 0 5 1 } }$ </td></tr><tr><td>Partial FT</td><td> $0 . 9 7 8 \pm 0 . 0 1 2$ </td><td> $0 . 9 3 9 \pm 0 . 0 3 7$ </td><td> $0 . 7 5 1 \pm 0 . 0 1 9$ </td><td>0.605 ± 0.034</td><td> $0 . 9 7 1 \pm 0 . 0 0 5$ </td><td> $0 . 8 4 2 \pm 0 . 0 7 6$ </td><td> $0 . 5 7 3 \pm 0 . 0 7 3$ </td><td>0.345 ± 0.250</td></tr><tr><td>LoRA</td><td> $\underline { { 0 . 9 7 9 \pm 0 . 0 1 2 } }$ </td><td> $\overline { { 0 . 9 3 3 \pm 0 . 0 3 2 } }$ </td><td> $0 . 7 5 2 \pm 0 . 0 1 9$ </td><td>0.612 ± 0.031</td><td> $\underline { { 0 . 9 8 0 \pm 0 . 0 1 8 } }$ </td><td> $\underline { { 0 . 8 9 6 \pm 0 . 0 5 2 } }$ </td><td> $0 . 5 8 2 \pm 0 . 0 5 6$ </td><td>0.365 ± 0.170</td></tr><tr><td> $\mathrm { R } ^ { 2 } \mathrm { T }$ </td><td> $0 . 9 6 1 \pm 0 . 0 0 1$ </td><td> $0 . 9 1 3 \pm 0 . 0 1 3$ </td><td> $0 . 7 5 5 \pm 0 . 0 1 9$ </td><td> $0 . 5 4 8 \pm 0 . 0 3 3$ </td><td> $0 . 9 6 3 \pm 0 . 0 0 2$ </td><td>0.788 ± 0.000</td><td>0.716 ± 0.080</td><td> $0 . 4 8 5 \pm 0 . 1 8 4$ </td></tr><tr><td>Ours</td><td>0.985 ± 0.008</td><td>0.940 ± 0.014</td><td>0.776 ± 0.015</td><td></td><td>0.722 ± 0.026 0.983 ± 0.011</td><td>0.946 ± 0.018</td><td>0.711 ± 0.033</td><td>0.595 ± 0.056</td></tr><tr><td>Frozen</td><td>0.936 ± 0.043</td><td>0.760 ± 0.062</td><td>0.785 ± 0.033</td><td>0.669 ± 0.059</td><td>0.930 ± 0.030</td><td>0.839 ± 0.114</td><td></td><td>0.604 ± 0.063</td></tr><tr><td rowspan="5">Virchow2</td><td>Partial FT</td><td>0.667 ± 0.005</td><td>0.608 ± 0.082</td><td>0.786 ± 0.030</td><td>0.671 ± 0.051</td><td>0.609 ± 0.041</td><td>0.490 ± 0.183</td><td> $0 . 7 0 8 \pm 0 . 0 7 7$  0.692 ± 0.058</td><td>0.591 ± 0.069</td></tr><tr><td>LoRA</td><td>0.914 ± 0.081</td><td>0.768 ± 0.086</td><td>0.786 ± 0.032</td><td>0.677 ± 0.049</td><td>0.963 ± 0.019</td><td>0.836 ± 0.128</td><td>0.741 ± 0.033</td><td>0.622 ± 0.046</td></tr><tr><td> $\mathrm { R ^ { 2 } T }$ </td><td>0.932 ± 0.011</td><td>0.737 ± 0.007</td><td>0.745 ± 0.021</td><td> $\overline { { 0 . 5 8 1 \pm 0 . 0 3 7 } }$ </td><td>0.923 ± 0.014</td><td>0.774 ± 0.054</td><td></td><td>0.614 ± 0.102</td></tr><tr><td>Ours</td><td>0.970 ± 0.014</td><td>0.853 ± 0.018</td><td>0.782 ± 0.027</td><td>0.683 ± 0.015</td><td>0.969 ± 0.011</td><td>0.884 ± 0.026</td><td> $0 . 7 2 8 \pm 0 . 0 2 9$  0.737 ± 0.024</td><td>0.639 ± 0.037</td></tr><tr><td>Frozen</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td rowspan="5">DINOv2-s LoRA</td><td>Partial FT</td><td>0.833 ± 0.054 0.832 ± 0.049</td><td>0.665 ± 0.100 0.774 ± 0.038</td><td> $0 . 4 5 8 \pm 0 . 0 4 3$  0.447 ± 0.018</td><td> $0 . 3 2 3 \pm 0 . 1 0 4$   $0 . 3 2 2 \pm 0 . 1 8 2$ </td><td>0.782 ± 0.061 0.826 ± 0.032</td><td> $0 . 5 3 3 \pm 0 . 2 4 4$  0.681 ± 0.055</td><td> $0 . 4 5 9 \pm 0 . 0 7 4$  0.496 ± 0.092</td><td> $0 . 4 2 6 \pm 0 . 1 1 2$  0.419 ± 0.157</td></tr><tr><td></td><td>0.835 ± 0.037</td><td>0.683 ± 0.126</td><td> $0 . 4 8 0 \pm 0 . 0 9 9$ </td><td> $0 . 0 0 0 \pm 0 . 0 0 0$ </td><td>0.829 ± 0.055</td><td>0.670 ± 0.104</td><td>0.528 ± 0.070</td><td>0.000 ± 0.000</td></tr><tr><td>R2T</td><td>0.878 ± 0.015</td><td>0.762 ± 0.037</td><td> $\overline { { 0 . 4 7 4 \pm 0 . 0 3 5 } }$ </td><td> $\underline { { 0 . 4 0 5 \pm 0 . 0 2 5 } }$ </td><td>0.853 ± 0.022</td><td>0.701 ± 0.053</td><td>0.458 ± 0.020</td><td>0.390 ± 0.125</td></tr><tr><td>Ours</td><td>0.890 ± 0.011</td><td>0.763 ± 0.014</td><td>0.530 ± 0.0150.484 ± 0.097</td><td></td><td>0.802 ± 0.041</td><td>0.712 ± 0.045</td><td>0.607 ± 0.026</td><td>0.459 ± 0.042</td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td rowspan="5">OpenCLIP LoRA</td><td>Frozen</td><td> $0 . 8 9 9 \pm 0 . 0 5 0$ </td><td>0.810 ± 0.074</td><td> $\underline { { 0 . 4 3 4 } } \pm 0 . 0 2 3$ </td><td> $0 . 4 0 4 \pm 0 . 0 5 6$ </td><td>0.895 ± 0.009</td><td>0.760 ± 0.107</td><td>0.413 ± 0.026</td><td>0.330 ± 0.199</td></tr><tr><td>Partial FT</td><td>0.907 ± 0.049</td><td>0.822 ± 0.068</td><td>0.426 ± 0.018</td><td> $0 . 4 0 4 \pm 0 . 0 6 4$ </td><td>0.897 ± 0.011</td><td>0.763 ± 0.110</td><td>0.452 ± 0.074</td><td>0.328 ± 0.097</td></tr><tr><td></td><td>0.909 ± 0.046</td><td> $\overline { { 0 . 8 2 2 \pm 0 . 0 6 8 } }$ </td><td> $0 . 4 3 2 \pm 0 . 0 2 3$ </td><td> $0 . 3 4 7 \pm 0 . 1 1 3$ </td><td>0.899 ± 0.014</td><td>0.758 ± 0.105</td><td>0.435 ± 0.035</td><td>0.413 ± 0.146</td></tr><tr><td> $\mathrm { R ^ { 2 } T }$ </td><td>0.909 ± 0.008</td><td>0.779 ± 0.071</td><td>0.421 ± 0.032</td><td> $0 . 4 1 3 \pm 0 . 0 5 2$ </td><td>0.916 ± 0.004</td><td>0.802 ± 0.033</td><td>0.409 ± 0.033</td><td>0.377 ± 0.085</td></tr><tr><td>Ours</td><td>0.911 ± 0.019</td><td></td><td>0.833 ± 0.013 0.443 ± 0.011</td><td>0.448 ± 0.053</td><td>0.928 ± 0.016</td><td>0.856 ± 0.026</td><td>0.486 ± 0.035</td><td>0.478 ± 0.039</td></tr></table>

Compared with the full model, the AUC decreases by 0.023 on PAROTID and 0.056 on BRACS, while the F1 score decreases by 0.060 and 0.054, respectively. This result supports the importance of transferring pathology-aware representation priors from the teacher encoder.

Role prototype guidance. Removing role prototype guidance decreases AUC and F1 on both datasets, with a more evident reduction in F1. This degradation indicates that role guidance provides information beyond the MoE structure and distillation by encouraging distinct expert transformations associated with task-relevant tissue patterns.

21-22 on both datasets, followed by a decline at layers 22- 23. Based on this trend, we adopt a relative insertion rule across backbones, placing the MoE-FFNs in selected highlevel blocks while leaving the final block unchanged.

Target-domain fine-tuning. On PAROTID, removing target-domain fine-tuning leaves the AUC unchanged at 0.954 but reduces F1 from 0.860 to 0.800. The unchanged AUC suggests that source-domain initialization already provides a useful overall ranking of positive and negative slides, whereas target-domain fine-tuning improves the final classification decisions under the target data distribution. Its smaller effect on BRACS suggests that the source-initialized representations transfer more directly to this dataset, while further targetspecific optimization is more beneficial for PAROTID.

Shared-expert weight. BRACS performs best at $\alpha = 0 ,$ whereas PAROTID reaches its highest F1 at α = 0.05. Since α = 0.05 substantially improves PAROTID while maintaining competitive performance on BRACS, it is selected as the default trade-off across the two datasets.

We conduct hyperparameter analysis using UNI with AB-MIL by varying one factor at a time while keeping the remaining settings fixed. The F1 results are shown in Fig. 3.

Number of routed experts. When varying the number of routed experts, one routed expert is kept free, while the remaining experts are guided by role prototypes. F1 improves when the number of routed experts increases from three to four but decreases with five or six experts. We therefore use four routed experts as the default setting.

Maximum activated routed experts. The performance is relatively stable when the maximum number of activated routed experts is set to one or two. BRACS slightly favors one expert, whereas PAROTID achieves its best performance with two. Increasing the upper bound to three or four provides no further gains. We therefore use two as the default to retain adaptive expert collaboration while maintaining stable performance across both datasets.

## D. Hyperparameter Analysis

## E. Representation and Downstream Evidence Analysis

MoE-FFN insertion layers. On UNI, performance generally improves as the MoE-FFNs are moved from earlier to higher transformer blocks, reaching the highest F1 at layers

To examine how the proposed method changes patch representations, we visualize features from the last MoE layer (i.e., the 11th block of DINOv2-small) using t-SNE. Frozen DINOv2-small features are extracted from the same block for comparison. As shown in Fig. 4, the frozen features are broadly dispersed with substantial overlap between clusters. In contrast, the features produced by the MoE-based encoder form more compact groups with clearer separation, indicating a more structured representation space. Panels (B) and (C) show the same MoE feature embedding colored by token cluster and routed expert, respectively. Several clusters are dominated by particular experts, while some regions contain contributions from multiple experts. This pattern suggests that the routing mechanism organizes token features into expert-associated subspaces without forcing a strict one-to-one correspondence between experts and tissue patterns. Together, these observations provide qualitative evidence that the MoE encoder reorganizes high-level token representations and promotes complementary expert specialization.

TABLE II  
ABLATION STUDY OF THE PROPOSED ROLE-GUIDED MOE ENCODER ADAPTATION FRAMEWORK USING UNI AS THE BACKBONE.
<table><tr><td rowspan="2">Method</td><td rowspan="2">MoE</td><td rowspan="2">Distill.</td><td rowspan="2">Role proto.</td><td rowspan="2">Target FT</td><td colspan="2">PAROTID</td><td colspan="2">BRACS</td></tr><tr><td>AUC</td><td>F1</td><td>AUC</td><td>F1</td></tr><tr><td>Frozen UNI</td><td></td><td></td><td></td><td></td><td> $0 . 8 9 8 \pm 0 . 0 3 1$ </td><td> $0 . 7 2 5 \pm 0 . 0 9 7$ </td><td> $0 . 6 4 9 \pm 0 . 0 6 8$ </td><td> $0 . 5 4 6 \pm 0 . 1 4 3$ </td></tr><tr><td>UNI + random MoE</td><td>√</td><td></td><td></td><td></td><td> $0 . 9 2 8 \pm 0 . 0 2 8$ </td><td> $0 . 7 9 1 \pm 0 . 0 3 6$ </td><td> $0 . 6 8 7 \pm 0 . 1 1 1$ </td><td> $0 . 5 8 5 \pm 0 . 1 3 4$ </td></tr><tr><td>Ours w/o distillation</td><td>√</td><td></td><td>√</td><td>√</td><td> $0 . 9 3 1 \pm 0 . 0 4 1$ </td><td> $0 . 8 0 0 \pm 0 . 0 4 8$ </td><td> $0 . 7 0 9 \pm 0 . 0 8 3$ </td><td> $0 . 6 3 9 \pm 0 . 1 0 4$ </td></tr><tr><td>Ours w/o role prototype</td><td>√</td><td>√</td><td></td><td>√</td><td> $0 . 9 4 9 \pm 0 . 0 2 9$ </td><td> $\overline { { 0 . 8 0 0 \pm 0 . 0 8 7 } }$ </td><td> $0 . 7 5 3 \pm 0 . 0 3 5$ </td><td> $0 . 6 8 5 \pm 0 . 0 3 5$ </td></tr><tr><td>Ours w/o target fine-tuning</td><td>√</td><td>√</td><td>√</td><td></td><td> $\underline { { 0 . 9 5 4 \pm 0 . 0 2 7 } }$ </td><td> $0 . 8 0 0 \pm 0 . 0 8 7$ </td><td> $0 . 7 5 9 \pm 0 . 0 3 8$ </td><td> $\underline { { 0 . 6 9 0 \pm 0 . 0 3 6 } }$ </td></tr><tr><td>Ours (full)</td><td>√</td><td>√</td><td>√</td><td>√</td><td> $\mathbf { 0 . 9 5 4 \pm 0 . 0 2 6 }$ </td><td> $\mathbf { 0 . 8 6 0 \pm 0 . 0 3 4 }$ </td><td> $\mathbf { 0 . 7 6 5 \pm 0 . 0 2 6 }$ </td><td> $\mathbf { 0 . 6 9 3 \pm 0 . 0 4 5 }$ </td></tr></table>

![](images/d12632c59ca97bfad752c7c4c5d4b5a7cbc96e3440d7e9d986ea09e5dc322d6f.jpg)

![](images/5c65a3f0891e2f30e55b72ff8db8a4ee67240622ba7275696be64a5599cf2720.jpg)

![](images/eb272ea21b75331db1ce93523615cea517ca98ca5a391efbf39029bdc19785d2.jpg)  
Fig. 3. F1-based hyperparameter analysis of the proposed method.

![](images/55a957562fafc8d4ca026edea307e2318b67be3994b93c653c546e352fae5127.jpg)

![](images/32c25ba761708a10bb0bd5f66df5fe991391eebb72def75d3fb065e6cf4f598a.jpg)  
Fig. 4. t-SNE visualization of frozen and MoE-based DINOv2-small features, colored by clusters in (A,B) and expert assignments in (C).

Fig. 5 provides a morphological interpretation of the expert preferences. The expert assignment map shows spatially heterogeneous routing across the WSI. Among the representative high-response patches, Expert 0 is mainly associated with non-tumor and background regions, including fibroadipose tissue, whereas Experts 1-3 respond predominantly to tumorrich regions with different local architectures, such as solid and gland-like patterns. These observations show that different experts emphasize different morphological patterns, supporting complementary specialization within the MoE encoder.

![](images/ed24b14d68908ab96b22063c9566a693f25d16cd6937a101f59643305d370c9d.jpg)  
Fig. 5. Morphological interpretation of expert preferences based on patch-level expert composition. (A) WSI F24-0328A01 H01 from the PAROTID dataset. (B) Expert assignment map and zoomed ROI. (C) Representative high-response patches for each expert.

![](images/f5f1ba3b815397e7c537bc015822c003189aa8b76b1b21b0d338b4149d5c4c76.jpg)  
Fig. 6. Downstream evidence shift after MoE optimization. ABMIL attention maps and top-attended patches show changes in slide-leve evidence selection.

We further examine whether the differences in encoder representations affect the evidence selected by the downstream MIL model. As shown in Fig. 6, the MoE-based features alter both the ABMIL attention distribution and the resulting high-attention patches. In the positive rescue case, attention is shifted toward tumor-rich regions, increasing the predicted positive probability from 0.447 to 0.584. In the false-positive suppression case, the selected evidence becomes more consistent with benign tissue patterns, reducing the positive probability from 0.802 to 0.269. These examples show that changes in patch representations propagate to the MIL attention mechanism and influence which tissue regions are used for slide-level prediction.

## V. DISCUSSION AND CONCLUSION

In MIL-based WSI classification, downstream performance is often constrained by frozen patch representations, while direct encoder fine-tuning may overfit on limited WSI data, and standard shared transformations may be insufficient for heterogeneous tissue patterns. To address these limitations, we propose a pathology role-guided MoE-FFN framework for efficient encoder-level representation learning. The framework introduces transformation diversity into selected highlevel transformer blocks, transfers pathology-aware priors through teacher-student distillation, and uses role prototypes to promote initial expert specialization. Target-domain finetuning further refines the initialized experts with asymmetric prototype-guided optimization, improving discrimination between task-relevant positive patterns and confusable hard negatives. As a plug-and-play module, the optimized MoE-FFN blocks can be transferred across transformer-based backbones. Experiments on the public BRACS dataset and the private PAROTID dataset demonstrate consistent improvements across five backbones and two MIL aggregators. Representation analyses further show a more structured token feature space, distinct expert preferences for tissue morphology, and shifted MIL evidence selection. These results suggest that the proposed framework improves target-aware patch representation quality and benefits slide-level WSI classification.

## REFERENCES

[1] G. Campanella et al., “Clinical-grade computational pathology using weakly supervised deep learning on whole slide images,” Nature medicine, vol. 25, no. 8, pp. 1301–1309, 2019.

[2] M. Ilse, J. Tomczak, and M. Welling, “Attention-based deep multiple instance learning,” in International conference on machine learning. PMLR, 2018, pp. 2127–2136.

[3] Z. Shao et al., “Transmil: Transformer based correlated multiple instance learning for whole slide image classification,” Advances in neural information processing systems, vol. 34, pp. 2136–2147, 2021.

[4] M. Y. Lu, D. F. Williamson, T. Y. Chen, R. J. Chen, M. Barbieri, and F. Mahmood, “Data-efficient and weakly supervised computational pathology on whole-slide images,” Nature biomedical engineering, vol. 5, no. 6, pp. 555–570, 2021.

[5] R. J. Chen et al., “Towards a general-purpose foundation model for computational pathology,” Nature medicine, vol. 30, no. 3, pp. 850–862, 2024.

[6] E. Vorontsov et al., “A foundation model for clinical-grade computational pathology and rare cancers detection,” Nature medicine, vol. 30, no. 10, pp. 2924–2935, 2024.

[7] M. Y. Lu et al., “A visual-language foundation model for computational pathology,” Nature medicine, vol. 30, no. 3, pp. 863–874, 2024.

[8] Y. Huang, W. Zhao, Y. Chen, Y. Fu, and L. Yu, “Free lunch in pathology foundation model: Task-specific model adaptation with concept-guided feature enhancement,” Advances in Neural Information Processing Systems, vol. 37, pp. 79 963–79 995, 2024.

[9] J. Lu, F. Yan, X. Zhang, Y. Gao, and S. Zhang, “Pathotune: Adapting visual foundation model to pathological specialists,” in International Conference on Medical Image Computing and Computer-Assisted Intervention. Springer, 2024, pp. 395–406.

[10] M. Kang, H. Song, S. Park, D. Yoo, and S. Pereira, “Benchmarking selfsupervised learning on diverse pathology datasets,” in Proceedings ofthe IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2023, pp. 3344–3354.

[11] F. M. Howard et al., “The impact of site-specific digital histology signatures on deep learning model accuracy and bias,” Nature communications, vol. 12, no. 1, p. 4423, 2021.

[12] W. Tang, F. Zhou, S. Huang, X. Zhu, Y. Zhang, and B. Liu, “Feature re-embedding: Towards foundation model-level performance in computational pathology,” in Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, 2024, pp. 11 343–11 352.

[13] J. Lee, J. Lim, K. Byeon, and J. T. Kwak, “Benchmarking pathology foundation models: Adaptation strategies and scenarios,” Computers in Biology and Medicine, vol. 190, p. 110031, 2025.

[14] M. Oquab et al., “Dinov2: Learning robust visual features without supervision,” Transactions on Machine Learning Research Journal, 2024.

[15] H. Touvron, M. Cord, M. Douze, F. Massa, A. Sablayrolles, and H. Jegou, “Training data-efficient image transformers & distillation´ through attention,” in International conference on machine learning. PMLR, 2021, pp. 10 347–10 357.

[16] J. Snell, K. Swersky, and R. Zemel, “Prototypical networks for few-shot learning,” Advances in neural information processing systems, vol. 30, 2017.

[17] Z. Chi et al., “On the representation collapse of sparse mixture of experts,” Advances in Neural Information Processing Systems, vol. 35, pp. 34 600–34 613, 2022.

[18] D. Dai et al., “Deepseekmoe: Towards ultimate expert specialization in mixture-of-experts language models,” in Proceedings ofthe 62nd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), 2024, pp. 1280–1297.

[19] X. Wang et al., “Transformer-based unsupervised contrastive learning for histopathological image classification,” Medical image analysis, vol. 81, p. 102559, 2022.

[20] H. Xu et al., “A whole-slide foundation model for digital pathology from real-world data,” Nature, vol. 630, no. 8015, pp. 181–188, 2024.

[21] T. Ding et al., “A multimodal whole-slide foundation model for pathology,” Nature medicine, pp. 1–13, 2025.

[22] Q. Liu, P. Xie, Z. Dai, and X. Bai, “Hiadapter: Histopathology-induced adapter for pathology foundation models,” IEEE Transactions on Medical Imaging, pp. 1–1, 05 2026.

[23] N. Houlsby et al., “Parameter-efficient transfer learning for nlp,” in International conference on machine learning. PMLR, 2019, pp. 2790– 2799.

[24] E. J. Hu et al., “Lora: Low-rank adaptation of large language models,” in The Tenth International Conference on Learning Representations, ICLR 2022, Virtual Event, April 25-29, 2022, 2022.

[25] R. A. Jacobs, M. I. Jordan, S. J. Nowlan, and G. E. Hinton, “Adaptive mixtures of local experts,” Neural computation, vol. 3, no. 1, pp. 79–87, 1991.

[26] W. Fedus, B. Zoph, and N. Shazeer, “Switch transformers: Scaling to trillion parameter models with simple and efficient sparsity,” Journal of Machine Learning Research, vol. 23, no. 120, pp. 1–39, 2022.

[27] C. Riquelme et al., “Scaling vision with sparse mixture of experts,” Advances in Neural Information Processing Systems, vol. 34, pp. 8583– 8595, 2021.

[28] J. Wu et al., “Learning heterogeneous tissues with mixture of experts for gigapixel whole slide images,” in Proceedings of the Computer Vision and Pattern Recognition Conference, 2025, pp. 5144–5153.

[29] J. Li et al., “M4: Multi-proxy multi-gate mixture of experts network for multiple instance learning in histopathology image analysis,” Medical Image Analysis, vol. 103, p. 103561, 2025.

[30] N. Hashimoto et al., “Multimodal gated mixture of experts using whole slide image and flow cytometry for multiple instance learning classification of lymphoma,” Journal of Pathology Informatics, vol. 15, p. 100359, 2024.

[31] Y. Guo, Z. Cheng, X. Tang, Z. Tu, and T. Lin, “Dynamic mixture of experts: An auto-tuning approach for efficient transformer models,” in International Conference on Learning Representations, vol. 2025, 2025, pp. 79 643–79 672.

[32] N. Shazeer et al., “Outrageously large neural networks: The sparselygated mixture-of-experts layer,” in International Conference on Learning Representations, 2017.

[33] J. N. Weinstein et al., “The cancer genome atlas pan-cancer analysis project,” Nature genetics, vol. 45, no. 10, pp. 1113–1120, 2013.

[34] D. Nechaev, A. Pchelnikov, and E. Ivanova, “Spider: a comprehensive multi-organ supervised pathology dataset and baseline models,” arXiv preprint arXiv:2503.02876, 2025.

[35] N. Brancati et al., “Bracs: A dataset for breast carcinoma subtyping in h&e histology images,” Database, vol. 2022, p. baac093, 2022.

[36] M. Cherti et al., “Reproducible scaling laws for contrastive languageimage learning,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2023, pp. 2818–2829.
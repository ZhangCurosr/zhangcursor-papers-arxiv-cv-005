# Scaling Full Conformal Image Classifiers

Julio Silva-Rodríguez Computer Vision Lab, ETH Zürich jusilva@ethz.ch

Ender Konukoglu Computer Vision Lab, ETH Zürich

## Abstract

Conformal prediction provides set-valued predictions with distribution-free coverage guarantees, making it attractive for high-stakes image classification. However, split conformal prediction is data-inefficient, while full conformal prediction (FCP), despite its stronger statistical efficiency, is computationally prohibitive at scale because it requires candidate-specific model refits at test time. We address this limitation by leveraging zero-shot vision-language models (VLMs) to guide scalable FCP in large label spaces. We introduce Targeted Full Conformal Prediction (T-FCP), which uses a lightweight inductive conformal predictor to prune unlikely labels and applies FCP only to the remaining candidates, reducing computation while retaining the formal guarantee of the combined conformal procedure. We further propose Stabilized Online LDA (SO-LDA), an efficient VLM adaptation solver based on rank-one inverse-covariance updates. Across multiple benchmarks, including ImageNet, T-FCP enables practical full-conformal image classification with modest test-time overhead, yielding efficient prediction sets and more stable empirical coverage than split conformal alternatives. Code is available: T-FCP

## 1 Introduction

In image classification, an input image $\mathbf { x } \in \mathcal { X }$ is mapped to a label space $\mathcal { V } = \{ 1 , \ldots , C \}$ via a trained model $\pi _ { \theta } : \mathcal { X }  \Delta _ { C } .$ , where $\Delta _ { C }$ denotes the probability simplex, and θ the parameters. Predictions are typically turned into decisions by selecting the label with the highest score. However, such a procedure does not provide information about model certainty, so potential errors or uncertain cases might remain unnoticed. While modern neural networks are increasingly adopted in high-stakes computer vision applications, e.g., medical imaging [32] or autonomous driving [30], there is a growing need to provide users with trustworthy operational decisions with appropriate guarantees.

Conformal prediction (CP) [54, 1, 43] is a machine learning framework that creates set-valued predictions with formal, distribution-free coverage guarantees. In the classification setting, CP constructs a prediction set ${ \mathcal { C } } ( \mathbf { x } ) \subseteq { \mathcal { D } }$ through a set-valued mapping $\mathcal { C } : \mathcal { X }  2 ^ { \mathcal { Y } }$ , such that the following marginal coverage property holds:

$$
\begin{array} { r } { \mathbb { P } \big ( Y \in \mathcal { C } ( \mathbf { x } ) \big ) \geq 1 - \alpha , } \end{array}\tag{1}
$$

where $\alpha \in ( 0 , 1 )$ is a user-specified error rate. This guarantee ensures, under mild exchangeability assumptions, that the true label lies within the prediction set with probability at least 1 − α under the target data distribution. Importantly, the size of $\mathcal C (  { \mathbf { x } } )$ adapts to the uncertainty of the prediction, such that inputs more ambiguous according to the model typically yield larger sets

The construction of C is data-driven, as it relies on a labeled calibration dataset, $\mathcal { D } _ { N } = \{ ( \mathbf { x } _ { i } , y _ { i } ) \} _ { i = 1 } ^ { N }$ Specifically, CP requires defining a nonconformity score function $\mathcal { S } : ( \mathcal { X } \times \mathcal { Y } ) \times \theta _ { \mathcal { D } _ { * } }  \mathbb { R }$ , where $\begin{array} { r } { S ( ( \mathbf { x } , y ) ; \theta _ { \mathcal { D } _ { * } } ) } \end{array}$ ) quantifies how atypical a candidate label y is for a given input x relative to a model trained on a reference dataset $\mathcal { D } _ { * } . \bar { \mathrm { C P } }$ uses the calibration data to determine a rejection threshold via an empirical quantile search on the observed score distribution $s _ { i } = S ( ( \mathbf x _ { i } , y _ { i } ) ; \theta _ { \mathcal D _ { * } } )$ for $i = 1 , \ldots , N ;$

$$
\hat { q } _ { \alpha } = \operatorname { Q u a n t i l e } ( \mathcal { D } _ { N } ; S ; \alpha ) = \operatorname* { i n f } \left\{ s \in \mathbb { R } : \frac { 1 } { N } \sum _ { i = 1 } ^ { N } \mathbf { 1 } \{ s _ { i } \leq s \} \geq \frac { \lceil ( N + 1 ) ( 1 - \alpha ) \rceil } { N } \right\} ,\tag{2}
$$

which corresponds to a finite-sample corrected (1 − α)-quantile of the calibration nonconformity scores, and is used for a new test input, $\mathbf { x } _ { N + 1 }$ , for constructing the prediction set:

$$
\mathcal { C } ( \mathbf { x } _ { N + 1 } ) = \{ y \in \mathcal { V } : { S ( ( \mathbf { x } _ { N + 1 } , y ) ; \theta _ { \mathcal { D } _ { * } } ) } \leq \hat { q } _ { \alpha } \} .\tag{3}
$$

By definition, this set contains all labels whose nonconformity scores are sufficiently small, and therefore conform to the calibration distribution.

Designing valid scoring functions. Several choices of S exist in the literature, which in turn influence the structure of the resulting prediction sets, typically building upon predicted class probabilities. As an illustrative example, one may define $\bar { S ( \mathbf { x } , y ; \theta ) } = 1 - \bar { \pi _ { \theta } ( \mathbf { x } ) } _ { y }$ , which assigns small scores to labels with high predicted probability and larger scores to less likely alternatives. A key formal requirement for conformal validity, however, is that the resulting nonconformity scores are exchangeable when evaluated on the calibration and test samples. Formally, consider the augmented dataset, $\mathcal { D } _ { N + 1 } = \{ ( { \bf x } _ { i } , y _ { i } ) \} _ { i = 1 } ^ { N + 1 }$ , assumed to be exchangeable. Let $s _ { i } = \mathcal { S } ( \mathbf { x } _ { i } , y _ { i } ; \boldsymbol { \theta } )$ denote the corresponding scores. Then, for any permutation σ of $\{ 1 , \bar { , } . . . , N + 1 \}$ , it holds that

$$
\begin{array} { r } { \big ( s _ { 1 } , \ldots , s _ { N + 1 } \big ) \stackrel { d } { = } \big ( s _ { \sigma ( 1 ) } , \ldots , s _ { \sigma ( N + 1 ) } \big ) , } \end{array}\tag{4}
$$

i.e., the joint distribution of the scores is permutation-invariant. This condition ensures that the test score is statistically indistinguishable from calibration scores, yielding the marginal coverage guarantee in Eq. (1). In practice, Inductive Conformal Prediction (ICP) [37, 54] enforces this requirement by learning the model parameters θ on data that is disjoint from the calibration set, thereby ensuring that the scores $\{ s _ { i } \} _ { i = 1 } ^ { \mathbf { \bar { N } } + 1 }$ are computed using a fixed, calibration data-independent model. When the calibration set is obtained by splitting the available labeled data, this procedure is commonly referred to as Split Conformal Prediction (SCP) [37, 54, 44, 53].

The data-efficiency challenge. Despite the SCP being the most common procedure for image classification, it is also inherently data-inefficient, as only a fraction of the available data contributes to training and to the empirical quantile estimation. This limitation is nontrivial: the marginal coverage guarantee holds only in expectation over the calibration samples, introducing finite-sample variability (i.e., a beta-binomial distribution on α and N [53, 21, 34]) that may lead to noticeable deviations from the target coverage when N is small. For instance, a user specifying $\alpha = 0 . 1$ may observe empirical coverage closer to 85% instead of 90% in practice.

Full Conformal Prediction (FCP) [43] is, in contrast, a more appealing alternative in terms of data efficiency, as it leverages the entire available calibration labeled data for both model fitting and quantile estimation without splits. The key idea is to maintain the score exchangeability requirement in Eq. 4 by ensuring that the fitting function is symmetric with respect to the extended dataset, $\mathcal { D } _ { N + 1 }$ which means that for all $( { \bf x } _ { i } , y _ { i } ) \in { \mathcal { D } } _ { N + 1 }$ , and any permutation σ:

$$
\begin{array} { r } { S ( ( \mathbf { x } _ { i } , y _ { i } ) ; \theta _ { \mathcal { D } _ { N + 1 } } ) = S ( ( \mathbf { x } _ { i } , y _ { i } ) ; \theta _ { \sigma ( \mathcal { D } _ { N + 1 } ) } ) . } \end{array}\tag{5}
$$

Since $y _ { N + 1 }$ is unknown, the FCP procedure requires testing the conformity of each new test sample x and candidate label $y \in \mathcal { V }$ . Each augmented dataset $\mathcal { D } _ { N + 1 } ^ { y } = \{ ( \mathbf { x } _ { 1 } , y _ { 1 } ) , \dotsc , ( \mathbf { x } _ { N } , y _ { N } ) , ( \mathbf { x } _ { N + 1 } , y ) \}$ is used to fit the parameters of the scoring function, $\theta _ { { D _ { N + 1 } ^ { y } } } .$ , which produces its respective nonconformity scores, $s _ { i } ^ { y } = { \cal S } ( ( { \bf x } _ { i } , y _ { i } ) ; \theta _ { \mathcal { D } _ { N + 1 } ^ { y } } )$ . The (1-α)-quantile using such score function is obtained from the same calibration set, $\hat { q } _ { \alpha } ^ { y }$ , and is used as a rejection criterion to construct valid conformal sets:

$$
\mathcal { C } _ { \mathrm { F C P } } ( \mathbf { x } _ { N + 1 } ) = \{ y \in \mathcal { Y } : { S ( ( \mathbf { x } _ { N + 1 } , y ) ; \theta _ { \mathcal { D } _ { N + 1 } ^ { y } } ) } \leq \hat { q } _ { \alpha } ^ { y } \} .\tag{6}
$$

Despite being more data-efficient, the FCP procedure creates two computational bottlenecks at test time: (B1) testing each potential label for an incoming test point, and (B2) the cost of training each classifier during testing. Consequently, especially considering standard large-scale computer vision benchmarks, e.g., ImageNet, where $\bar { C } = \bar { 1 } 0 0 0$ , FCP has been systematically neglected in favor of SCP, at the cost of assuming larger annotation efforts or wider coverage distributions.

Contributions. In this work, we aim to enable the construction of full conformal image classifiers at scale. Our main observation is that in a large label space, despite not being intrinsically ordered, many classes may be weakly related to each other, e.g., “husky” and “toaster”, while some groups of categories might be more similar, e.g., “husky”, “malamute”, “Samoyed”, or “Greater Swiss Mountain dog” (real examplesfrom ImageNet). Therefore, if an initial proxy were available to guide the FCP procedure toward the most likely labels for a new test image, the total runtime could be significantly reduced. Hence, we propose exploiting the open-vocabulary capabilities of pre-trained vision-language models (VLMs), such as CLIP [39]. These provide zero-shot class prototypes via text descriptions for the target label domain, which, despite not being specialized in the downstream task, serve as proxies in the model’s embedding space to discard non-similar categories that are not considered in the FCP procedure for each test image. Technically, our contributions are:

• We formalize Targeted Full Conformal Prediction (T-FCP), a procedure that only requires running FCP on a subset of the label space (B1), defined by an ICP procedure guided by zero-shot class prototypes with provable coverage guarantees without requiring splits on calibration data.

• To address the cost of multiple model fits (B2), we introduce SO-LDA, an online Linear Discriminant Analysis solver for VLM adaptation that performs stable incremental class-prototype updates together with rank-one inverse-covariance updates. SO-LDA achieves performance close to expensive gradient-based solvers while being orders of magnitude faster.

Extensive experiments demonstrate that our framework enables full conformal procedures on largescale datasets with a modest latency overhead (e.g., ∼ 30 ms/image for ImageNet), yielding efficient sets and more stable coverage distributions than split conformal procedures.

## 2 Related Work

Full conformal prediction. Recent work on full conformal prediction has focused on reducing its computational burden, e.g., by limiting the candidate label space evaluated at test time. In regression, this has been achieved by exploiting the natural ordering of the response space, either through discretization [28, 27] or homotopy/continuation methods that track the solution path of the fitted model as the candidate response varies [27, 26, 17, 29]. Classification problems, however, do not directly admit these strategies. Existing work for classification has instead focused on online classifiers with decoupled test-time updates, including exact permutation-invariant updates for kNN and SVMs [6], as well as inexact updates for gradient-based solvers via influence functions [35].

Conformal prediction for image classification. Most neural-network-based image classification works follow SCP [11, 51, 2, 9, 20, 5, 10], improving either training objectives for set efficiency [11, 51] or nonconformity scores for adaptiveness, class-conditional coverage, and robustness [2, 5, 9, 10]. In contrast, calibration-data efficiency and empirical coverage variability have received less attention, as these methods typically rely on in-domain training data and separate calibration splits.

Conformal prediction for zero-shot models. Recent works have explored split conformal prediction for vision foundation models [13] and, more specifically, vision-language models (VLMs) [47]. Several methods improve VLM adaptation within conformal procedures using unsupervised transductive adaptation [47, 46, 4]. Closer to our work, [49] explored full conformal classifiers using closed-form linear solvers in the VLM embedding space. However, [49] focused on small-scale datasets (C < 50). As we show later, their proposed kNN-style solver, SS-Text, does not scale well to large label spaces and fine-grained classification tasks.

Combining conformal predictors. Related to our work are methods that combine or cascade multiple split conformal procedures [14, 52, 42]. For example, conformal cascades progressively prune candidate sets obtained with different nonconformity scores [14]. Other works derive combined coverage guarantees, either through multiplicative bounds under independence assumptions [52], or by allocating individual error rates using common corrections such as Bonferroni [52, 42]. These approaches have been used to achieve goals such as label-dependent guarantees for object detection [52] and joint guarantees across multiple outputs [42].

## 3 Methods

## 3.1 Background on zero-shot image classifiers

Let $f _ { \theta } : \mathcal { X } \to \mathbb { R } ^ { F }$ denote a visual encoder that maps an input image x to a feature vector v = $f _ { \theta } ( \bar { \mathbf { x } } ) \in \mathbb { R } ^ { F }$ . We consider contrastive vision-language models (VLMs), following CLIP [39], as our zero-shot models. These provide class prototypes from class descriptions $( \mathrm { e . g . , \mathrm { ~ } } ^ { 6 6 } \mathrm { a n }$ image of a [class name]”), via its text encoder $g _ { \phi } .$ , producing textual embeddings $\mathbf { t } _ { y } = g _ { \phi } \mathbf { \bar { ( p r o m p t } { } _ { \psi } ) } ^ { - } \in \mathbb { R } ^ { F }$ . These embeddings can be directly used as class prototypes, i.e., $\mathbf { W } ^ { 0 } = ( \mathbf { t } _ { y } ) _ { y = 1 } ^ { C }$ , by measuring the similarity between $\ell _ { 2 }$ -normalized image embeddings and class prototypes, yielding class probabilities:

$$
\pi _ { \mathbf { W } ^ { 0 } } ( \mathbf { x } _ { i } ) = ( p _ { i , y } ^ { 0 } ) _ { y = 1 } ^ { C } \in \Delta _ { C } , \quad p _ { i , y } ^ { 0 } = \frac { \exp { \left( \mathbf { v } _ { i } ^ { \top } \mathbf { w } _ { y } ^ { 0 } / \tau \right) } } { \sum _ { j \in \mathcal { Y } } \exp { \left( \mathbf { v } _ { i } ^ { \top } \mathbf { w } _ { j } ^ { 0 } / \tau \right) } } ,\tag{7}
$$

## 3.2 Targeted Full Conformal Prediction

First, we aim to leverage zero-shot models to address the first bottleneck in full conformal prediction procedures (B1): the need to iterate over the entire label space for each incoming test sample.

Combining conformal procedures. Our approach is based on the observation that multiple valid conformal predictors can be combined while maintaining statistical guarantees. Let $\mathcal { C } _ { 1 }$ and $\mathcal { C } _ { 2 }$ be two conformal predictors constructed using error levels $\alpha _ { 1 }$ and $\alpha _ { 2 }$ , respectively. That is, for a test point $( \mathbf { x } , Y )$ , and under the standard exchangeability assumptions:

$$
\begin{array} { r } { \mathbb { P } ( Y \in \mathcal { C } _ { 1 } ( \mathbf { x } ) ) \geq 1 - \alpha _ { 1 } , \quad \mathbb { P } ( Y \in \mathcal { C } _ { 2 } ( \mathbf { x } ) ) \geq 1 - \alpha _ { 2 } . } \end{array}\tag{8}
$$

We define a combined prediction set as the intersection

$$
\mathcal { C } _ { 1 \cap 2 } ( \mathbf { x } ) = \mathcal { C } _ { 1 } ( \mathbf { x } ) \cap \mathcal { C } _ { 2 } ( \mathbf { x } ) .\tag{9}
$$

Proposition 1. The prediction set $\mathcal { C } _ { 1 \cap 2 } ( \mathbf { x } )$ satisfies

$$
\begin{array} { r } { \mathbb { P } \big ( Y \in \mathcal { C } _ { 1 \cap 2 } ( \mathbf { x } ) \big ) \geq 1 - ( \alpha _ { 1 } + \alpha _ { 2 } ) . } \end{array}\tag{10}
$$

The proof follows directly from De Morgan’s law and the union bound, and is given in Appendix B. Remark 1 (Tightness and conservativeness). The bound in Proposition 1 is a worst-case guarantee and may be conservative in practice. It becomes tight when the two miscoverage events are disjoint and have probabilities $\alpha _ { 1 }$ and $\alpha _ { 2 }$ . When the error events overlap, the intersection can have coverage higher than $1 - \left( \alpha _ { 1 } + \alpha _ { 2 } \right)$ . Ifthe individual miscoverage probabilities are exactly $\alpha _ { 1 }$ and $\alpha _ { 2 } ,$ the coverage ofthe intersection lies between $1 - \left( \alpha _ { 1 } + \alpha _ { 2 } \right)$ and $1 - \operatorname* { m a x } \{ \alpha _ { 1 } , \alpha _ { 2 } \}$

Targeted conformal prediction via zero-shot models. Let us now introduce a labeled calibration set consisting of embedding representations, $\mathcal { D } _ { N } = \{ ( \mathbf { v } _ { i } , y _ { i } ) \} _ { i = 1 } ^ { N }$ . A set of zero-shot linear weights, $\mathbf { W } ^ { 0 }$ , is presented, and the goal is to create conformal sets with coverage guarantees from new incoming test images, using their embedding, $\mathbf { v } _ { N + 1 }$ . Let us now define an ICP procedure, $\mathcal { C } _ { \mathrm { I C P } }$ whose nonconformity score function leverages the zero-shot weights, $S ( \mathbf { x } , y ; \mathbf { W } ^ { 0 } )$ , and produces conformal sets with an error rate of $\alpha _ { \mathrm { I C P } }$ using the whole calibration set. Also, let us denote an FCP procedure, $\mathcal { C } _ { \mathrm { F C P } } .$ , that uses a more expensive scoring function by fitting the classifier on the extended dataset $\begin{array} { r } { \mathcal { S } ( ( \mathbf { x } , y ) ; \mathbf { W } ^ { \mathcal { D } _ { N + 1 } ^ { y } } ) } \end{array}$ , operating at an error level $\alpha _ { \mathrm { { F C P } } }$ . We define the Targeted Full Conformal Prediction (T-FCP) sets as the intersection of both procedures, i.e., ${ \mathcal { C } } _ { \mathrm { T - F C P } } = { \mathcal { C } } _ { \mathrm { I C P \cap F C P } }$ , such that, according to Proposition 1, the following marginal coverage property holds:

$$
\mathbb { P } \big ( Y \in \mathcal { C } _ { \mathrm { T - F C P } } ( \mathbf { x } ) \big ) \geq 1 - \big ( \alpha _ { I C P } + \alpha _ { F C P } \big ) .\tag{11}
$$

By construction, any label excluded by $\mathcal { C } _ { \mathrm { I C P } }$ for a test sample cannot appear in $\mathcal { C } _ { \mathrm { T - F C P } }$ . Therefore, we can use $\mathcal { C } _ { \mathrm { I C P } }$ as a pruning stage of highly unlikely labels to be included in the conformal set, and only run the expensive FCP procedure, $\mathcal { C } _ { \mathrm { F C P } } .$ , on the surviving, i.e., target labels, while maintaining theoretical guarantees. Specifically, for a given target error rate α, we can set $\alpha _ { F C P } = \alpha - \alpha _ { I C P }$ The parameter $\alpha _ { \mathrm { I C P } }$ controls a computational-statistical tradeoff: larger values typically yield smaller ICP candidate sets and therefore lower FCP cost, but allocate less error budget to the full conformal stage. In our experiments, we find that $\alpha _ { \mathrm { I C P } } = 0 . 5 \%$ already provides substantial computational efficiency gains while keeping the additional conservativeness limited in practice.

## 3.3 SO-LDA: Stabilized Online LDA

Second, we exploit the capabilities of zero-shot models to alleviate the second bottleneck in full conformal prediction (B2): the computational cost of fitting the classifier for each extended dataset.

Transfer learning. Instead of training a deep network for each extended dataset, we leverage the rich embedding representation of pre-trained vision-language models, similarly to [49]. Given a dataset with labeled visual examples, and its corresponding embeddings, $\mathcal { D } _ { N } = \mathbf { \bar { \{ } }  ( \mathbf { v } _ { i } , y _ { i } ) \} _ { i = 1 } ^ { N }$ , a common strategy is to learn task-specific linear class prototypes ${ \bf W } ^ { D _ { N } }$ , that replace its zero-shot counterpart in Eq. 7 to produce softmax class scores.

Online LDA. While several linear probe solvers have been proposed in the literature [16, 45, 56], gradient-descent-based minimization of softmax cross-entropy is generally the best-performing solution [31, 45]. However, its computational cost $( \mathcal { O } ( T N F \tilde { C } )$ , with $T$ the number of iterations) makes it infeasible for full conformal prediction. We therefore turn our attention to training-free solvers [59, 56, 49, 48], particularly those that enable decoupled updates between calibration samples and new incoming test points. Specifically, we build upon the Linear Discriminant Analysis (LDA) solver [15], proposed in [56] for VLMs, which models class-conditional feature distributions as Gaussian with a shared covariance matrix. In its simplified form, by dropping the bias term for consistency with Eq.7, the classifier weights are given by:

$$
\mathbf { w } _ { y } = \pmb { \Sigma } ^ { - 1 } \mu _ { y }\tag{12}
$$

where $\mu _ { y }$ are the class centers, and $\Sigma ^ { - 1 }$ the inverse covariance matrix.

Given a labeled set $\mathcal { D } _ { N }$ , and let $\begin{array} { r } { N _ { y } = \sum _ { i \in \mathcal { D } _ { N } } { \bf 1 } \{ y _ { i } = y \} } \end{array}$ denote the number of samples for each category, then the model parameters can be estimated as follows:

$$
\mu _ { y } _ { \left( N \right) } = \frac { 1 } { N _ { y } } \sum _ { i \in \mathcal { D } _ { N } } \mathbf { 1 } \{ y _ { i } = y \} \mathbf { v } _ { i } , \quad \Sigma _ { \left( N \right) } ^ { - 1 } = \mathbf { S } _ { \left( N \right) } ^ { - 1 } = ( \frac { 1 } { N } \sum _ { i \in \mathcal { D } _ { N } } \mathbf { z } _ { i } \mathbf { z } _ { i } ^ { \top } ) ^ { - 1 } ,\tag{13}
$$

where $\mathbf { z } _ { i } = \mathbf { v } _ { i } - \mu _ { y = y _ { i } } ( _ { N } )$ are the class-centered features. To avoid numerical instabilities, we use diagonal-loading stabilization, ${ \bf S } _ { ( N ) } ^ { \mathrm { r e g } } = { \bf S } _ { ( N ) } + \lambda _ { \mathrm { R E G } } \mathrm { D i a g } ( { \bf S } _ { ( N ) } )$ , following the common use of shrinkage for high-dimensional precision estimation [24, 56]. The loading term is estimated once from the calibration covariance and kept fixed during candidate augmentations, preserving efficient rank-one updates. Appendix C compares it with fixed-ridge and full recomputation variants.

Given a new incoming sample, $( v _ { N + 1 } , y _ { N + 1 } )$ , the model parameters can be updated [25], instead of being recalculated again. Particularly, for the covariance matrix updates, we can consider the Sherman–Morrison formula [18] to compute the inverse of its rank-one update efficiently:

$$
\mu _ { y } _ { ( N + 1 ) } = \frac { N _ { y } \cdot \mu _ { y } _ { ( N ) } + \mathbf { v } _ { N + 1 } \mathbf { 1 } \{ y _ { N + 1 } = y \} } { N _ { y } + \mathbf { 1 } \{ y _ { N + 1 } = y \} } ,\tag{14}
$$

$$
\pmb { \Sigma } _ { ( N + 1 ) } ^ { - 1 } = \frac { N + 1 } { N } \left( \pmb { \Sigma } _ { ( N ) } ^ { - 1 } - \frac { \pmb { \Sigma } _ { ( N ) } ^ { - 1 } \pmb { \mathrm { z } } _ { N + 1 } \pmb { \mathrm { z } } _ { N + 1 } ^ { \top } \pmb { \Sigma } _ { ( N ) } ^ { - 1 } } { N + \pmb { \mathrm { z } } _ { N + 1 } ^ { \top } \pmb { \Sigma } _ { ( N ) } ^ { - 1 } \pmb { \mathrm { z } } _ { N + 1 } } \right) .\tag{15}
$$

In full conformal procedures, the parameters are first estimated from the calibration data and then updated for each augmented dataset $\mathcal { D } _ { N + 1 } ^ { y }$ . This reduces the per-label update cost to $\mathcal { O } ( F ^ { 2 } )$ avoiding repeated matrix inversion $( \mathcal { O } ( F ^ { 3 } ) )$ and enabling scalable conformal inference compared to recomputing the LDA solution from scratch $( \mathcal { O } ( N F ^ { 2 } + \mathbf { \bar { \it F } } ^ { 3 } ) )$ .

Enabling stable online updates through zero-shot prototypes. A practical difficulty of online LDA in the full conformal setting is that the residual vectors $\mathbf { z } _ { i }$ for samples in $\mathcal { D } _ { N }$ are computed using statistics estimated before the candidate test sample is observed, whereas the test sample itself is evaluated under updated statistics. This creates an asymmetry between the approximate online update (Eq. 15) and a full refit on the augmented dataset. To mitigate this issue, we introduce Stabilized Online LDA (SO-LDA), including the following adjustments:

$$
\mathbf { z } _ { i } = \mathbf { v } _ { i } - \mathbf { W } _ { y _ { i } } ^ { ( 0 ) } ,
$$

(16)

$$
\mu _ { y } ^ { \mathrm { S O - L D A } } = \mu _ { y } ^ { ( N + 1 ) } + \lambda _ { \mathrm { T E X T } } \mathbf { W } _ { y } ^ { ( 0 ) } .\tag{17}
$$

According to $\operatorname { E q } .$ . 16, the feature-centering anchors used for covariance estimation no longer depend on the calibration samples, but on the zero-shot prototypes. Therefore, the residual centering rule remains invariant under candidate augmentation and avoids the asymmetry induced by recomputing visual class centers differently for calibration and test samples. Additionally, following standard practices in VLM adaptation [45, 48, 49], Eq. 17 introduces a text-guided prior (weighted by $\lambda _ { \mathrm { T E X T } } )$ on the updated class means by biasing them toward the textual prototypes before $\ell _ { 2 } \cdot$ -normalization and score prediction in Eq. 12. This improves stability, especially in low-data regimes.

## 4 Experiments

## 4.1 Setup

Datasets. We evaluate the proposed procedure in the standard 11 datasets used for CLIP zero- and few-shot transfer evaluation [45, 56]. These are: ImageNet [8], SUN397 [57], FGVCAircraft [33], EuroSAT [19], StanfordCars [23], Food101 [3], OxfordPets [38], Flowers102 [36], Caltech101 [12], DTD [7], and UCF101 [50]. These gather a heterogeneous set of general, fine-grained, and specialized domain classification tasks, several of which have hundreds of categories. We refer to Appendix D for specific details on the number of categories and tasks.

Calibration sets. The corresponding test partition from each dataset is used in our conformal experiments to produce disjoint calibration and test subsets. Specifically, samples of size $N = C \times K$ with K the so-called number of shots in the transfer learning literature [60, 16, 45], are retrieved for calibration. By default, we set $K \ : = \ : 1 6$ , a standard data regime in transfer learning from VLMs [60, 16, 45], and present ablation studies with smaller sample sizes. Note that, in contrast to the standard literature on transfer learning from VLMs, which samples label-balanced adaptation sets, we follow the practice in conformal prediction [47, 46] that maintains the dataset’s label-marginal distribution in the calibration data to avoid breaking exchangeability assumptions. All experiments are repeated 50 times using different random seeds for sampling the calibration data.

Zero-shot models. CLIP [39], specifically ViT-B/16 backbone, is used as a zero-shot model to evaluate the proposed conformal procedure. We also present generalization studies to other CLIP and MetaCLIP [58] backbones. The text encoder from each model is used to produce zero-shot class-wise prototypes for each downstream category by using standard templates and category names [60, 16, 45], specifically the same prompts as in [47].

Implementation details. Conformal sets are created at error rates $\alpha \in \{ 0 . 1 0 , 0 . 0 5 \}$ . For T-FCP, we set α<sub>ICP</sub> = 0.5%. For SO-LDA, the hyperparameters are fixed to $\lambda _ { \mathrm { T E X T } } = 1$ and $\lambda _ { \mathrm { R E G } } = 1 0$

Baselines. First, we compare our conformal procedure with standard ones in the literature:

• Inductive conformal prediction (ICP): uses the whole calibration sample for quantile search, and computes the nonconformity score using the zero-shot scores (ZS) without adaptation.

• Split conformal prediction (SCP): splits the calibration data into two equally-sized subsets, one for adaptation and the other for conformal quantile search. Since SCP is inductive, there is no computational restriction during the adaptation stage; therefore, we used standard softmax cross-entropy minimization via iterative gradient descent (GD). Training details are in Appendix E.

• Full conformal prediction (FCP): performs the full conformal procedure over the whole label space. Adaptation is performed using our proposed SO-LDA solver.

Second, we include ICP procedures that perform unsupervised transductive adaptation (ICP-T), recently proposed for VLMs [47, 46, 4]. Training details are in Appendix E.

Nonconformity scores. We use LAC [41] as the default score, as it is low-latency and commonly used in FCP [6, 35]. APS [40] is explored in Appendix F.3. More adaptive scores, such as RAPS [2] or ClusterCP [9], require sorting or additional hyperparameters, which introduces extra computation and, in some cases, additional data splits. Since our focus is on data-efficient and scalable FCP, we leave the integration of adaptive nonconformity scores into large-scale FCP to future work.

Metrics. We report accuracy and standard conformal metrics. Empirical coverage is characterized by its average and deviation (2σ) across seeds, together with the percentage of runs achieving coverage at least 1 − α − 0.005 (%Valid), being 0.005 a tolerance. Set efficiency is measured by the mean and median set size, and by the percentage of singleton predictions (%Sing.), as in [5].

## 4.2 Main results

Standard conformal procedures. Table 1 provides the results obtained with the different conformal procedures, and Figure 1 (a) illustrates the empirical coverage distribution for the main conformal alternatives. 1 In terms of coverage, it is worth noting that, despite all procedures achieving the target average coverage, SCP’s coverage deviation across samples is 20% relatively larger than ICP’s, due to the data split. This is not the case for FCP procedures, which provide narrower coverage distributions, as in ICP. Notably, T-FCP achieves the largest proportion of experiments with valid coverage among these procedures: 81.8% and 88.7% for $\alpha = 0 . 1 0$ and $\alpha = 0 . 0 5$ , respectively. This is explained by the theoretically expected slightly over-coverage produced by the union-bound error rate design, $\mathrm { i . e . , } \alpha _ { \mathrm { I C P } } .$ , for early label pruning. 2 Regarding set efficiency, ICP produces the largest set sizes, since it does not benefit from transfer learning with labeled data. SCP and FCP provide similar mean set sizes. However, the median indicates that FCP produces set sizes that are ∼ 10% smaller (more efficient) than SCP, while its number of singletons (better on the discriminative aspect) is larger by a considerable margin. We explain these mixed observations in Figure 1 (b), showing that FCP using SO-LDA presents longer tails in the set size distribution, i.e., larger sets on uncertain inputs. This is explained by the design of the FCP, which inflates the conformity of each image-label candidate by training in the extended dataset, and the differences in optimization objectives between solvers. Finally, T-FCP provides slightly more conservative (larger) sets than FCP, with the advantage of enabling its real-time application on large-scale datasets, as discussed later. Still, T-FCP improves the median set size and singleton rate relative to SCP.

Table 1: Comparison of conformal procedures atop CLIP ViT-B/16 using $N = C \times 1 6$ . Results are averaged across 11 datasets, and per-dataset results are in Appendix F.2.
<table><tr><td rowspan="2"></td><td rowspan="2"></td><td rowspan="2"></td><td colspan="3"> $\mathrm { C o v . } \ ( \alpha = 0 . 1 0 )$ </td><td colspan="3">Set size</td><td colspan="3"> $\mathrm { C o v . } \ ( \alpha = 0 . 0 5 )$ </td><td colspan="3">Set size</td></tr><tr><td>|Acc. |Avg.</td><td>2σ</td><td>%Valid</td><td>Mean</td><td>Med.</td><td>%Sing.</td><td>Avg.</td><td>2σ</td><td></td><td>%Valid Mean</td><td>Med.</td><td>%Sing.</td></tr><tr><td>ICP</td><td>ZS</td><td>65.7</td><td>90.1 2.0</td><td></td><td>77.8</td><td>4.8</td><td>4.5</td><td>37.3</td><td>95.2</td><td>1.5</td><td>84.0</td><td>7.9</td><td>7.3</td><td>28.2</td></tr><tr><td>ICP-T</td><td>OT [47]</td><td>68.6</td><td>90.6 2.0</td><td></td><td>82.3</td><td>3.9</td><td>3.7</td><td>42.4</td><td>95.4</td><td>1.4</td><td>86.8</td><td>6.1</td><td>5.6</td><td>32.6</td></tr><tr><td>ICP-T</td><td>TIM [46]</td><td>71.5</td><td>90.62.0</td><td></td><td>80.5</td><td>3.6</td><td>3.5</td><td>45.2</td><td>95.4</td><td>1.4</td><td>87.2</td><td>5.6</td><td>5.3</td><td>34.3</td></tr><tr><td>SCP</td><td>GD</td><td>|79.9|</td><td></td><td>|90.7 2.7</td><td>77.1</td><td>2.2</td><td>2.1</td><td>64.0</td><td>|95.5</td><td>1.9</td><td>79.5</td><td>3.3</td><td>3.0</td><td>51.9</td></tr><tr><td>FCP</td><td>SO-LDA</td><td>80.3</td><td>90.12.2</td><td></td><td>77.6</td><td>2.2</td><td>1.8</td><td>69.3</td><td>95.2</td><td>21.6</td><td>80.5</td><td>3.3</td><td>2.6</td><td>57.4</td></tr><tr><td>T-FCP</td><td>SO-LDA</td><td>80.3</td><td>90.7 2.1</td><td></td><td>81.8</td><td>2.3</td><td>1.9</td><td>68.1</td><td>95.6</td><td>1.5</td><td>88.7</td><td>3.5</td><td>2.7</td><td>56.0</td></tr></table>

![](images/3ff010f26232056c66ea235da359d59a531889f08656dcc2bbf030fdbd68f58b.jpg)

![](images/c168ede4848b9bab8f0d564e9e0dcc469710a81427bc24b8eceb1e8f371d13a6.jpg)  
(a) Coverage empirical distribution.

![](images/758db1219c06657f12875875357b31df475f2ccccca7a4ff5aa5d92fef25a554.jpg)

![](images/22311ad09256999b810f23f4f1ee5306d2960b17d4e7b99ce54110dc137491f9.jpg)  
(b) Set size distribution.  
Figure 1: Analysis on empirical coverage and set size distribution on representative datasets. Results using CLIP ViT-B/16 as backbone, $N = C \times 1 6$ samples for calibration.

Comparison against transductive adaptation. Recently explored unsupervised transductive solvers (ICP-T in Table 1) improve over vanilla ICP in accuracy and set efficiency while maintaining the same calibration sample and therefore similar dispersion in coverage distribution. However, their unsupervised adaptation underperforms SCP in accuracy and therefore yields larger conformal sets.

Computational efficiency. Figure 2(a) showcases the latency scaling, in images per second, for full conformal procedures with the number of categories. Our experiments estimate that running the full conformal procedure using the iterative GD solver on ImageNet would require ∼ 500 s per image, which is infeasible, as generally assumed in the literature. By using our SO-LDA solver, such latency is dramatically reduced to ∼ 0.33 s, and is further decreased to ∼ 30 ms when pruning unlikely labels via our proposed T-FCP procedure. Regarding GPU usage, we parallelized the number of tested labels in the FCP procedure to 100, which maintains, as depicted in Figure 2(b), a top-up peak-GPU memory consumption of ∼ 16 Gb on the largest dataset, i.e., ImageNet. However, as shown in Figure 2(b), such a limit can be decreased without sacrificing much latency.

## 4.3 In-depth studies

Label pruning stage. Figure 3 (a) provides a visualization of the pruning capabilities of the inductive stage of our T-FCP procedure across datasets. Figure 3 (a, left) shows that most categories are discarded with small α , specifically the elbow is found near 1% of the allowed error rate. Therefore, most computational efficiency gains can be achieved without sacrificing much of the error-rate budget for the full conformal stage. Figure 3 (a, center) shows a correlation between the zero-shot performance and the pruning capabilities; thus, T-FCP will be more computationally efficient the stronger the zero-shot model initially is. However, upon a closer look to Figure 3 (a, center and right), large-scale datasets, such as SUN397 ( ) or ImageNet ( ), present larger pruning ratios than their zero-shot accuracy (nearly 60 − 70%) would imply. These results suggest that our T-FCP procedure is especially effective on the largest-scale datasets, where inter-class heterogeneities are greater and therefore categories with unrelated visual content are more easily pruned.

Zero-shot acc. (%)  
![](images/7187fe610e48204a1e62ddd8606522fa3efd0c34f5e5b061a5ef73677fd44321.jpg)

<table><tr><td>Dataset</td><td>C /Iter</td><td>P-GPU</td><td>imgs / s</td></tr><tr><td rowspan="3">DTD (C = 37)</td><td>25</td><td>106.3 MB</td><td>393</td></tr><tr><td>50</td><td>181.0 MB</td><td>745</td></tr><tr><td>100</td><td>181.1 MB</td><td>746</td></tr><tr><td rowspan="3">ImageNet (C = 1K)</td><td>25</td><td>4.1 GB</td><td>27.7</td></tr><tr><td>50 100</td><td>8.2 GB</td><td>29.4</td></tr><tr><td></td><td>16.2 GB</td><td>30.3</td></tr></table>

(a) Latency scaling with number of classes.  
(b) Label parallelization (C/Iter) in T-FCP.  
Figure 2: Computational efficiency analysis. Results using ViT-B/16, $N = C \times 1 6$ and $\alpha = 0 . 0 5$ All our experiments were conducted on a single NVIDIA A100-PCIE-40GB.

Conservativeness and computational feasibility trade-off. In Figure 3 (b) we explore the effect of pruning at different α on latency gains, as well as the set efficiency loss. As theoretically expected, larger α provide faster runtimes in the FCP stage according to Figure 3 (b, left). In terms of set efficiency, i.e, conservativeness, the produced set distribution is bounded between the ICP and FCP, as shown in Figure 3 (b, right), providing a user with control over the computational feasibility and set efficiency sacrifice via α . If the latter is a priority, then T-FCP provides gains over SCP only when using $\alpha _ { \mathrm { I C P } } < 2 . 5 \%$ . On the other hand, if coverage consistency is the main desideratum, all T-FCP configurations would yield a smaller variance in the finite-sample coverage distribution, similar to ICP, and therefore the decision on how to set α<sub>ICP</sub> is constrained only by the available latency margin.

![](images/048ebceb47c8f89c3c2f4cc2b2ee66e85d72a5a8e0ce389f368e87723eabfb0d.jpg)

![](images/897dcc372425cdb62768a64a5a0ae96f9c6b777170440134d172aa8ac8cd10ed.jpg)  
(a) Pruning control at ICP stage (results per dataset).

![](images/6f553269b3f11b2638333e6395c0f754f2cf2ca118354d15d6ebe7a48db3179f.jpg)

![](images/4906cf6ed1089583b9f2a85887e3c3fc8bc10b124933e61e0f2f51ce6cb7d0d7.jpg)

![](images/544b55d8445ba6a9be22e2df060ae92da367af73a63f44810df00f06dd01efed.jpg)  
(b) Latency vs. convervativeness.  
Figure 3: Ablation studies on T-FCP. Results using CLIP ViT-B/16 as backbone, $N = C \times 1 6$ samples for calibration, and conformal sets created at α = 0.1 error rate.

Robustness to smaller calibration samples. Figure 4 (a-d) tests the robustness of the different conformal procedures under smaller calibration sets. Focusing on the empirical coverage dispersion Figure 4 (b), as theoretically expected, all procedures reduce it as more calibration data become available. In this context, T-FCP enables more consistent coverage than SCP across all data regimes, while providing consistently more efficient sets (Figure 4 (c-d)). In the extreme low-data regime, i.e., $N = C \times 4$ , we observe a slight instability of the full conformal procedure, Figure 4 (a), which provides slightly lower marginal coverage than desired, an empirical instability already observed in prior work [35]. However, we would like to point out that the large variance of the SCP procedure (∼ ±4.5) makes its realized coverage also less stable in this regime.

SO-LDA configuration and performance. Figure 4 (e) compares the discriminative performance of the different relevant linear solvers across increasing data regimes. First, it is worth noting that methods that rely solely on class centers, e.g., Simple-shot (SS) [55, 49], despite their computational efficiency, fall short in performance compared to the best performing method, i.e., using iterative gradient-descent updates (GD), which makes them uncompetitive for full conformal procedures compared to SCP. Our proposed stabilized LDA solver with zero-shot guided feature centering in Eq. 16 consistently improves the performance of the learned prototypes compared to vanilla LDA [56].

We attribute this behavior to the fact that visual examples within a category may be highly correlated in a specific visual domain, whereas textual information can improve the estimation of the main directions of dispersion in class distributions. Notably, SO-LDA reached performances closer to the expensive iterative GD solver, just ∼ 1.5% below. The constraint that learned prototypes remain close to their initial configurations in Eq. 17 helped stabilize performance gains in lower data regimes, as already observed in prior work on transfer learning from VLMs [22, 31, 45, 48].

![](images/877288c6bdce6570c1696db040a2e72e20fccba9c0b0ddc2ddabf9fb13146aeb.jpg)  
(a)

![](images/e4bd96ca29cf3210b49f516f9340658480d8865622cc81112aeb4b83aae46990.jpg)  
(b)

![](images/8fdfc896adc9e9d038ec062ec9d7aa0445e06c3b4dd2d5b0d704f99bf8ee4f8a.jpg)  
(c)

![](images/076be2ddcf55a4de01b7d3cd3272a8a68923aa1cc6fdebde3b03bfb8adcf77b2.jpg)  
(d)

![](images/22ae8e2cb5033272891fed496c6d64193043a428fa2aca06a400ff7cac2dd373.jpg)  
(e)  
Figure 4: Studies at lower data regimes: (a-d) conformal procedures, and (e) few-shot solvers. Results using CLIP ViT-B/16 as backbone, and conformal sets at α = 0.1, averaged for 11 datasets.

Performance atop other VLMs. We evaluate the conformal procedures on additional backbones, including CLIP [39] and MetaCLIP [58], in Figure 5. The merits of T-FCP hold across all backbones, providing consistently more concentrated empirical coverage distributions than SCP, with variances ∼ 30% smaller, while maintaining valid coverage. Moreover, predictive accuracy and set efficiency are consistently improved, especially compared to ICP. This is even the case for larger-scale backbones, e.g., MetaCLIP ViT-L/14, for which the median set size is reduced by 31%, without sacrificing finite-sample coverage stability. Despite the strong reliance of T-FCP and our SO-LDA solver on zero-shot prototypes, performance remains robust also in smaller backbones, e.g., ViT-B/32.

![](images/27140c60a05840282f03706616ee273c899732b4a50d0891b6b8be64836a056c.jpg)

![](images/4538bb256e8c52e44d1cdf6971ba84f702a5ea9ee3565c7a1442c4756d85ca69.jpg)

![](images/8b80ab78d5e5ddb394d943f31e086e39106a3466a9204cbe99488350347bea0a.jpg)

![](images/c92cb7a08f688334500a5f7a5d8364cfcc8699145ce5bb40120b9877f1c22fc4.jpg)  
Figure 5: Generalization atop other CLIP [39] and MetaCLIP [58] backbones of conformal procedures. Results using N = C × 16 samples and α = 0.1, averaged across 11 datasets.

## 5 Discussion

We presented T-FCP and SO-LDA, enabling full conformal prediction on large-scale image datasets by leveraging zero-shot VLMs. Our experiments highlight that data-efficient procedures yield more stable empirical coverage than split-based alternatives and underscore the importance of data efficiency in conformal image classification.

Limitations. The proposed linear solver is still less performant than iterative gradient-based methods. Although this gap is partially compensated for in FCP by fitting on each candidate-augmented dataset, we cannot guarantee that the resulting prediction sets will always be more efficient than those from SCP, which can use more expensive solvers during its adaptation stage. A further limitation is that our main SO-LDA implementation uses diagonal loading to stabilize the precision matrix, thereby approximating recomputing a fully regularized LDA estimator for each augmented dataset. Therefore, although T-FCP offers theoretically valid coverage, the same is not true when it is combined with our SO-LDA solver. In practice, ablations in Appendix C show negligible differences against exact implementations. Finally, the coverage guarantee of T-FCP is based on an additive union-bound allocation across the pruning and full-conformal stages. Consequently, the resulting sets may be slightly more conservative than FCP’s. Hence, FCP may be preferred for smaller-scale tasks, e.g., those involving tens or a few hundred classes, especially when coupled with our online LDA solver.

## References

[1] Vladimir Vapnik Alex Gammerman, Volodya Vovk. Learning by transduction. In Conference on Uncertainty in Artificial Intelligence, pages 148–156, 1998.

[2] Anastasios Nikolas Angelopoulos, Stephen Bates, Michael Jordan, and Jitendra Malik. Uncertainty sets for image classifiers using conformal prediction. In International Conference on Learning Representations, pages 1–17, 2020.

[3] Lukas Bossard, Matthieu Guillaumin, and Luc Van Gool. Food-101 – mining discriminative components with random forests. In European Conference on Computer Vision, 2014.

[4] Behzad Bozorgtabar, Dwarikanath Mahapatra, Sudipta Roy, Imran Razzak Muzammal Naseer, and Zongyuan Ge. LATA: Laplacian-Assisted Transductive Adaptation for Conformal Uncertainty in Medical VLMs . In Proceedings of the Computer Vision and Pattern Recognition Conference, 2026.

[5] Margarida M Campos, João Cálem, Sophia Sklaviadis, Mario A. T. Figueiredo, and Andre Martins. Sparse activations as conformal predictors. In Yingzhen Li, Stephan Mandt, Shipra Agrawal, and Emtiyaz Khan, editors, Proceedings of International Conference on Artificial Intelligence and Statistics, volume 258, pages 2674–2682, 2025.

[6] Giovanni Cherubin, Konstantinos Chatzikokolakis, and Martin Jaggi. Exact optimization of conformal predictors via incremental and decremental learning. In International Conference on Machine Learning, pages 1836–1845. PMLR, 2021.

[7] Mircea Cimpoi, Subhransu Maji, Iasonas Kokkinos, Sammy Mohamed, and Andrea Vedaldi. Describing textures in the wild. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 3606–3613, 2014.

[8] Jia Deng, Wei Dong, Richard Socher, Li-Jia Li, Kai Li, and Li Fei-Fei. Imagenet: A large-scale hierarchical image database. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 248–255, 2009.

[9] Tiffany Ding, Anastasios Angelopoulos, Stephen Bates, Michael Jordan, and Ryan J Tibshirani. Class-conditional conformal prediction with many classes. In Advances in Neural Information Processing Systems, volume 36, pages 64555–64576, 2023.

[10] Tiffany Ding, Jean-Baptiste Fermanian, and Joseph Salmon. Conformal prediction for longtailed classification. In The Fourteenth International Conference on Learning Representations, 2026.

[11] Bat-Sheva Einbinder, Yaniv Romano, Matteo Sesia, and Yanfei Zhou. Training uncertaintyaware classifiers with conformalized deep learning. In Advances in Neural Information Processing Systems (NeurIPS), 2022.

[12] Li Fei-Fei, R. Fergus, and P. Perona. Learning generative visual models from few training examples: An incremental bayesian approach tested on 101 object categories. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition Worskshops, pages 178–178, 2004.

[13] Leo Fillioux, Julio Silva-Rodriguez, Ismail Ben Ayed, Paul-Henry Cournede, Maria Vakalopoulou, Stergios Christodoulidis, and Jose Dolz. Are foundation models for computer vision good conformal predictors? Transactions on Machine Learning Research, 2026.

[14] Adam Fisch, Tal Schuster, Tommi Jaakkola, and Regina Barzilay. Efficient conformal prediction via cascaded inference with expanded admission. In Proceedings ofThe Tenth International Conference on Learning Representations, 2021.

[15] R. A. Fisher. The use of multiple measurements in taxonomic problems. Annals ofEugenics, 7(2):179–188, 1936.

[16] Peng Gao, Shijie Geng, Renrui Zhang, Teli Ma, Rongyao Fang, Yongfeng Zhang, Hongsheng Li, and Yu Qiao. Clip-adapter: Better vision-language models with feature adapters. International Journal ofComputer Vision, 2023.

[17] Etash Kumar Guha, Eugene Ndiaye, and Xiaoming Huo. Conformalization of sparse generalized linear models. In International Conference on Machine Learning, pages 11871–11887. PMLR, 2023.

[18] William W Hager. Updating the inverse of a matrix. SIAM Review, 31(2):221–239, 1989.

[19] Patrick Helber, Benjamin Bischke, Andreas Dengel, and Damian Borth. Introducing eurosat: A novel dataset and deep learning benchmark for land use and land cover classification. In IEEE International Geoscience and Remote Sensing Symposium, pages 3606–3613, 2018.

[20] Jianguo Huang, Huajun Xi, Linjun Zhang, Huaxiu Yao, Yue Qiu, and Hongxin Wei. Conformal prediction for deep classifier via label ranking. In International Conference on Machine Learning, volume 235, pages 20331–20347, 2024.

[21] Roel Hulsman. Distribution-free finite-sample guarantees and split conformal prediction. arXiv preprint arXiv:2210.14735, 2022.

[22] Muhammad Uzair Khattak, Syed Talal Wasim, Muzammal Naseer, Salman Khan, Ming-Hsuan Yang, and Fahad Shahbaz Khan. Self-regulating prompts: Foundational model adaptation without forgetting. In Proceedings ofthe IEEE/CVF International Conference on Computer Vision, pages 15190–15200, 2023.

[23] Jonathan Krause, Michael Stark, Jia Deng, and Li Fei-Fei. 3d object representations for finegrained categorization. In Proceedings of the IEEE International Conference on Computer Vision Workshops, pages 554–561, 2013.

[24] Tatsuya Kubokawa and Muni S Srivastava. Estimation of the precision matrix of a singular wishart distribution and its application in high-dimensional data. Journal of multivariate Analysis, 99(9):1906–1928, 2008.

[25] Ludmila I. Kuncheva and Catrin O. Plumpton. Adaptive learning rate for online linear discriminant classifiers. In Structural, Syntactic, and Statistical Pattern Recognition, pages 510–519, 2008.

[26] Jing Lei. Fast exact conformalization of the lasso using piecewise linear homotopy. Biometrika, 106(4):749–764, 2019.

[27] Jing Lei, Max G’Sell, Alessandro Rinaldo, Ryan J. Tibshirani, and Larry Wasserman. Distribution-free predictive inference for regression. Journal of the American Statistical Association, 113(523):1094–1111, 2018.

[28] Jing Lei, James Robins, and Larry Wasserman. Distribution-free prediction sets. Journal ofthe American Statistical Association, 108(501):278–287, 2013.

[29] Diyang Li. Generalized fast exact conformalization. In Neural Information Processing Systems, 2024.

[30] Xiwen Liang, Yangxin Wu, Jianhua Han, Hang Xu, Chunjing Xu, and Xiaodan Liang. Effective adaptation in multi-task co-training for unified autonomous driving. Advances in Neural Information Processing Systems (NeurIPS), 35:19645–19658, 2022.

[31] Zhiqiu Lin, Samuel Yu, Zhiyi Kuang, Deepak Pathak, and Deva Ramanan. Multimodality helps unimodality: Cross-modal few-shot learning with multimodal models. In Proceedings ofthe IEEE/CVF conference on computer vision and pattern recognition, pages 19325–19337, 2023.

[32] Geert Litjens, Thijs Kooi, Babak Ehteshami Bejnordi, Arnaud Arindra Adiyoso Setio, Francesco Ciompi, Mohsen Ghafoorian, Jeroen AWM van der Laak, Bram van Ginneken, and Clara I Sánchez. A survey on deep learning in medical image analysis. Medical image analysis, 42:60–88, 2017.

[33] S. Maji, J. Kannala, E. Rahtu, M. Blaschko, and A. Vedaldi. Fine-grained visual classification of aircraft. In arXiv preprint arXiv:1306.5151, 2013.

[34] F Marques and C Paulo. Universal distribution of the empirical coverage in split conformal prediction. Statistics & Probability Letters, 219(C), 2025.

[35] Javier Abad Martinez, Umang Bhatt, Adrian Weller, and Giovanni Cherubin. Approximating full conformal prediction at scale via influence functions. In Proceedings of the AAAI Conference on Artificial Intelligence, volume 37, pages 6631–6639, 2023.

[36] Maria-Elena Nilsback and Andrew Zisserman. Automated flower classification over a large number of classes. In Indian Conference on Computer Vision, Graphics and Image Processing, 2008.

[37] Harris Papadopoulos, Kostas Proedrou, Volodya Vovk, and Alex Gammerman. Inductive confidence machines for regression. In European Conference on Machine Learning (ECML), pages 345–356, 2002.

[38] Omkar M Parkhi, Andrea Vedaldi, Andrew Zisserman, and CV Jawahar. Cats and dogs. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, page 3498–3505, 2012.

[39] Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, et al. Learning transferable visual models from natural language supervision. In International Conference on Machine Learning, pages 8748–8763, 2021.

[40] Yaniv Romano, Matteo Sesia, and Emmanuel Candes. Classification with valid and adaptive coverage. In Advances in Neural Information Processing Systems, volume 33, pages 3581–3591, 2020.

[41] Mauricio Sadinle, Jing Lei, and Larry Wasserman. Least ambiguous set-valued classifiers with bounded error levels. Journal ofthe American Statistical Association, 114(525):223–234, 2019.

[42] Sara Sangalli, Gary Sarwin, Ertunc Erdil, Carlo Serra, Alessandro Carretta, Victor Staartjes, and Ender Konukoglu. Conformal forecasting for surgical instrument trajectory . In proceedings of Medical Image Computing and Computer Assisted Intervention – MICCAI 2025, volume LNCS 15968. Springer Nature Switzerland, September 2025.

[43] Craig Saunders, Alexander Gammerman, and Volodya Vovk. Transduction with confidence and credibility. In International Joint Conference on Artificial Intelligence (IJCAI), pages 722–726, 1999.

[44] Glenn Shafer and Vladimir Vovk. A tutorial on conformal prediction. Journal of Machine Learning Research, 9(3), 2008.

[45] Julio Silva-Rodríguez, Sina Hajimiri, Ismail Ben Ayed, and Jose Dolz. A closer look at the fewshot adaptation of large vision-language models. In Proceedings ofthe IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 23681–23690, 2024.

[46] Julio Silva-Rodríguez, Ismail Ben Ayed, and Jose Dolz. Trustworthy Few-Shot Transfer of Medical VLMs through Split Conformal Prediction . In Proceedings of Medical Image Computing and Computer Assisted Intervention, volume LNCS 15966, pages 658–668, 2025.

[47] Julio Silva-Rodríguez, Ismail Ben Ayed, and Jose Dolz. Conformal prediction for zero-shot models. In Proceedings ofthe Computer Vision and Pattern Recognition Conference, pages 19931–19941, 2025.

[48] Julio Silva-Rodríguez et al. Few-shot, now for real: Medical vlms adaptation without balanced sets or validation. In Proceedings of Medical Image Computing and Computer Assisted Intervention, volume LNCS 15966, 2025.

[49] Julio Silva-Rodríguez et al. Full conformal adaptation of medical vision-language models. In International Conference on Information Processing in Medical Imaging, pages 278–293, 2025.

[50] Khurram Soomro, Amir Roshan Zamir, and Mubarak Shah. Ucf101: A dataset of 101 human actions classes from videos in the wild. In arXiv preprint arXiv:1212.0402, 2012.

[51] David Stutz, Krishnamurthy Dj Dvijotham, Ali Taylan Cemgil, and Arnaud Doucet. Learning optimal conformal classifiers. In International Conference on Learning Representations (ICLR), 2022.

[52] Alexander Timans, Christoph-Nikolas Straehle, Kaspar Sakmann, and Eric Nalisnick. Adaptive bounding box uncertainties via two-step conformal prediction. In European Conference on Computer Vision, pages 363–398. Springer, 2024.

[53] Vladimir Vovk. Conditional validity of inductive conformal predictors. In Proceedings ofthe Asian Conference on Machine Learning, volume 25, pages 475–490, 2012.

[54] Vladimir Vovk, Alex Gammerman, and Glenn Shafer. Algorithmic Learning in a Random World. Springer, 01 2005.

[55] Yan Wang et al. Simpleshot: Revisiting nearest-neighbor classification for few-shot learning. In arXiv preprint arXiv:1911.04623, 2019.

[56] Zhengbo Wang, Jian Liang, Lijun Sheng, Ran He, Zilei Wang, and Tieniu Tan. A hard-to-beat baseline for training-free CLIP-based adaptation. In International Conference on Learning Representations, 2024.

[57] Jianxiong Xiao, James Hays, Krista A. Ehinger, Aude Oliva, and Antonio Torralba. Sun database: Large-scale scene recognition from abbey to zoo. In Proceedings ofthe IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 3485–3492, 2010.

[58] Hu Xu, Saining Xie, Xiaoqing Tan, Po-Yao Huang, Russell Howes, Vasu Sharma, Shang-Wen Li, Gargi Ghosh, Luke Zettlemoyer, and Christoph Feichtenhofer. Demystifying CLIP data. In International Conference on Learning Representations, 2024.

[59] Renrui Zhang, Rongyao Fang, Wei Zhang, Peng Gao, Kunchang Li, Jifeng Dai, Yu Qiao, and Hongsheng Li. Tip-adapter: Training-free clip-adapter for better vision-language modeling. In European Conference on Computer Vision, pages 1–19, 2022.

[60] Kaiyang Zhou, Jingkang Yang, Chen Change Loy, and Ziwei Liu. Learning to prompt for vision-language models. International Journal ofComputer Vision, 2022.

□

## A Broader Impacts

Broader impacts. Conformal prediction is often motivated by providing theoretical guarantees in high-stakes deployment. However, it should be interpreted with care. Marginal coverage does not imply conditional or subgroup coverage, and guarantees need not hold under distribution shift. Moreover, finite-sample empirical coverage can fall below the nominal level in a real-world deployment, especially with data-inefficient pipelines. In this work, we directly address the latter use case, enabling more stable conformal image classifiers via full conformal prediction. We do not anticipate negative societal impacts from our work beyond those of standard conformal prediction, and we identify no domain-specific risks requiring special attention.

## B Proofs

Proof of Proposition 1. According to Eq 9:

$$
\mathbb { P } \big ( Y \notin \mathcal { C } _ { 1 \cap 2 } ( \mathbf { x } ) \big ) = \mathbb { P } \big ( Y \notin \mathcal { C } _ { 1 } ( \mathbf { x } ) \cup Y \notin \mathcal { C } _ { 2 } ( \mathbf { x } ) \big ) .\tag{18}
$$

By the union bound,

$$
\mathbb { P } \big ( Y \notin \mathcal { C } _ { 1 \cap 2 } ( \mathbf { x } ) \big ) \le \mathbb { P } \big ( Y \notin \mathcal { C } _ { 1 } ( \mathbf { x } ) \big ) + \mathbb { P } \big ( Y \notin \mathcal { C } _ { 2 } ( \mathbf { x } ) \big ) .\tag{19}
$$

Using the marginal coverage guarantees for each individual procedure in Eq. 8,

$$
\begin{array} { r } { \mathbb { P } \big ( Y \notin \mathcal { C } _ { 1 \cap 2 } ( \mathbf { x } ) \big ) \le \alpha _ { 1 } + \alpha _ { 2 } , } \end{array}\tag{20}
$$

which implies the result.

## C Stability of SO-LDA

In this section, we further analyze SO-LDA, particularly the implications of the mild asymmetry introduced by the diagonal covariance matrix regularization used in the main implementation, whose diagonal is estimated only from calibration data to enable efficient rank-one updates. We compare such a solution with different configurations:

• Ridge: the regularization is imposed via a fixed identity loading, ${ \bf S } _ { ( N ) } ^ { \mathrm { r e g } } = { \bf S } _ { ( N ) } + \lambda$ <sub>RIDGE</sub>I, which is compatible with permutation-invariant rank-one updates under fixed regularization.

• Non-online: for each candidate sample-label pair, it recomputes the class means and the covariance matrix from the full augmented dataset, then directly inverts the regularized covariance matrix. This solver increases the per-label computational cost to $\mathcal { O } ( N F ^ { 2 } + { \bf \bar { \it F } } ^ { 3 } )$

• Unstable: does not use zero-shot prototypes for the residuals estimation as in Eq. 16. Hence, the centered features are calculated asymmetrically between calibration and test data.

Comparative performance on different data regimes is presented in Table 2. The results suggest that the dominant source of instability is the asymmetric centering of the residuals used for covariance estimation, with the Unstable configuration yielding the largest marginal coverage gap and lower ratio of valid occurrences. The fixed-Ridge configuration is the cleanest variant with respect to symmetric online updates. However, its discriminative performance is lower than that of our diagonal regularization in SO-LDA. Moreover, both exhibit a similar coverage distribution, suggesting that the asymmetry introduced by our implementation choice is negligible. Also, the performance levels achieved by SO-LDA are similar to those obtained with more expensive, exact Non-online updates.

The under-coverage for $K = 4$ is observed in both online and exact recomputation variants, suggesting it is not primarily caused by the diagonal-loading approximation. As this instability in exact FCP also appears in prior work [35] (e.g., see their results at their Figure 4(d)), we leave a complete characterization of this regime to future work.

## D Datasets

A summary of the datasets, number of classes, tasks, and number of samples in the used test partition is depicted in Table 3.

Table 2: Study of the SO-LDA solver configurations under T-FCP. Average results for 10 datasets. In contrast to the main manuscript, these studies use 20 random seeds and exclude ImageNet.
<table><tr><td rowspan="2"></td><td rowspan="2"></td><td colspan="3">Cov.  $( \alpha = 0 . 1 0 )$ </td><td colspan="2">Set size</td><td rowspan="2"></td><td rowspan="2"></td><td colspan="2"> $\mathrm { C o v . } \ ( \alpha = 0 . 1 0 )$ </td><td colspan="2"></td><td rowspan="2">Set size</td></tr><tr><td>Acc. Avg.</td><td>2σ</td><td></td><td>%Valid Mean %Sing.</td><td></td><td>Acc. Avg.</td><td>2σ</td><td>%Valid Mean %Sing.</td><td></td></tr><tr><td>SO-LDA</td><td>81.0</td><td>90.8 2.1</td><td></td><td>80.0</td><td>2.3</td><td>70.3</td><td>SO-LDA</td><td>76.4</td><td>88.2 4.2</td><td></td><td>27.0</td><td>3.1</td><td>61.6</td></tr><tr><td>Non-online</td><td>81.0</td><td>90.8 2.1</td><td></td><td>80.0</td><td>2.3</td><td>70.2</td><td>Non-online</td><td>76.4</td><td>88.3 4.2</td><td></td><td>27.5</td><td>3.2</td><td>61.3</td></tr><tr><td>Ridge</td><td>78.8</td><td>90.7 2.3</td><td></td><td>76.0</td><td>2.4</td><td>64.6</td><td>Ridge</td><td>74.1</td><td>88.04.3</td><td></td><td>26.0</td><td>2.9</td><td>58.4</td></tr><tr><td>Unstable</td><td></td><td>78.8 90.4 2.3</td><td></td><td>67.5</td><td>2.6</td><td>62.3</td><td>Unstable</td><td></td><td>70.1 86.0 5.3</td><td></td><td>13.0</td><td>4.6</td><td>45.9</td></tr><tr><td></td><td>(a)</td><td></td><td></td><td> $\overline { { N = C \times 1 6 } }$ </td><td></td><td></td><td></td><td>(b)</td><td></td><td> $\overline { { N = C \times 4 } }$ </td><td></td><td></td><td></td></tr></table>

Table 3: Datasets overview. “b"/“i" denotes balanced/imbalanced.
<table><tr><td>Dataset</td><td>Classes</td><td>Train</td><td>Splits Val</td><td>Test</td><td>b/i</td><td>Task description</td></tr><tr><td>EuroSAT [19]</td><td>10</td><td>13,500</td><td>5,400</td><td>8,100</td><td>i</td><td>Satellite image classification.</td></tr><tr><td>OxfordPets [38]</td><td>37</td><td>2,944</td><td>736</td><td>3,669</td><td>i</td><td>Pets classification.</td></tr><tr><td>DTD [7]</td><td>47</td><td>2,820</td><td>1,128</td><td>1,692</td><td>b</td><td>Textures classification.</td></tr><tr><td>FGVCAircraft [33]</td><td>100</td><td>3,334</td><td>3,333</td><td>3,333</td><td>i</td><td>Aircraft classification.</td></tr><tr><td>Caltech101 [12]</td><td>100</td><td>4,128</td><td>1,649</td><td>2,465</td><td>i</td><td>Natural objects classification.</td></tr><tr><td>UCF101 [50]</td><td>101</td><td>7,639</td><td>1,898</td><td>3,783</td><td>i</td><td>Action recognition.</td></tr><tr><td>Food101 [3]</td><td>101</td><td>50,500</td><td>20,200</td><td>30,300</td><td>b</td><td>Foods classification.</td></tr><tr><td>Flowers102 [36]</td><td>102</td><td>4,093</td><td>1,633</td><td>2,463</td><td>u</td><td>Flowers classification.</td></tr><tr><td>StanfordCars [23]</td><td>196</td><td>6,509</td><td>1,635</td><td>8,041</td><td>i</td><td>Cars classification.</td></tr><tr><td>SUN397 [57]</td><td>397</td><td>15,880</td><td>3,970</td><td>19,850</td><td>b</td><td>Scenes classification.</td></tr><tr><td>ImageNet [8]</td><td>1,000</td><td>1.28M</td><td></td><td>50,000</td><td>b</td><td>Natural objects recognition.</td></tr></table>

## E Baselines

Gradient-descent linear probe (GD). We followed training details similar to [45], which remain fixed for all datasets. The learned class weights are initialized with zero-shot prototypes. Full-batch gradient descent is performed for 300 iterations, minimizing cross-entropy. SGD is used as the optimizer with momentum 0.9. A cosine-annealing scheduler with a base learning rate of 0.1 is applied to ensure a small step size and proper convergence.

ICP-T via Conf-OT [47]. The logits are extracted using the zero-shot prototypes, and label-marginal distributions are estimated from calibration data. The Optimal Transport solver is applied to the joint calibration and test sets, with the entropic weight τ = 1 and $T _ { \mathrm { O T } } = 3 .$ , as default values in [47].

ICP-T via TIM [46]. We use the Transductive Information Maximization solver with regularization on the expected label-marginal distribution, estimated from calibration labels. Training is carried out on the joint calibration and test samples. The relative weight λ is set to 1.0, as in [46]. Similarly to GD, full-batch gradient descent is performed for 300 iterations, in this case optimizing the mutual information criteria. Other details are as in GD, since they provided satisfactory convergence.

## F Detailed Results

## F.1 Additional configurations

We report the performance of the proposed solver SO-LDA for split conformal prediction in Table 4.

Table 4: Comparison of conformal procedures atop CLIP ViT-B/16 using $N = C \times 1 6$ . This table reports the results using configurations not included in Table 1.
<table><tr><td rowspan="2"></td><td rowspan="2"></td><td colspan="3">Cov. (α = 0.10)</td><td colspan="3">Set size</td><td colspan="3">Cov. (α = 0.05)</td><td colspan="3">Set size</td></tr><tr><td></td><td>|Acc. |Avg. 2σ</td><td></td><td></td><td></td><td></td><td></td><td>σ %Valid Mean Med. %Sing. |Avg. 2σ</td><td> %Valid Mean Med. %Sing.</td><td></td><td></td><td></td></tr><tr><td>SCP SO-LDA</td><td></td><td></td><td>|79.5 | 90.8 3.5</td><td>71.3</td><td>2.7</td><td>2.3</td><td>69.4</td><td></td><td>|95.62.4</td><td>77.0</td><td>4.1</td><td>3.3</td><td>58.4</td></tr></table>

## F.2 Table 1: detailed results per dataset

The detailed results of the conformal procedures’ performance across datasets are reported in Table 5.  
Table 5: Per-dataset conformal prediction results of average performances of the main conformal procedures reported in Table 1, using CLIP ViT-B/16.
<table><tr><td colspan="3"></td><td colspan="3">Cov. (α = 0.10)</td><td colspan="3">Set size</td><td colspan="3">Cov. (α = 0.05)</td><td colspan="3">Set size</td></tr><tr><td>ICP</td><td>ZS</td><td>48.4</td><td>|Acc. |Avg.</td><td>2σ |90.4 6.1</td><td>62.0</td><td>4.2</td><td>4.4</td><td>%Valid Mean Med. %Sing. 9.4</td><td></td><td>|Avg. 2σ |95.64.8</td><td>66.0</td><td>5.2</td><td>%Valid Mean Med. 5.3</td><td>%Sing. 6.6</td></tr><tr><td>EoSAT ICP Pets</td><td>SCP GD T-FCP ZS SCP GD</td><td>SO-LDA</td><td>83.4 83.0 88.8 92.3</td><td>89.86.5 90.2 5.3 89.93.2 90.0 4.9</td><td></td><td>50.0 58.0 54.0 65.0</td><td>1.4 1.1 1.4 1.0 1.0 1.0 1.0 1.0</td><td>73.4 71.4 92.9 94.8</td><td></td><td>96.1 4.7 95.93.9 95.1 2.0 95.2 3.3</td><td>76.0 90.0 68.0 66.0</td><td>2.2 2.0 1.2 1.1</td><td>1.8 1.9 1.0 1.0</td><td>43.8 42.0 80.4</td></tr><tr><td>ICP DTD SCP</td><td>T-FCP ZS GD</td><td>SO-LDA</td><td>92.8 43.4 66.6</td><td>90.32.7 |89.92.9 90.24.4</td><td>70.0 66.0 55.0</td><td>1.0 10.8 3.4</td><td>1.0 11.1 2.8</td><td>93.6 6.75 28.3</td><td>95.6</td><td>2.2 |95.01.9 95.1 2.6</td><td>82.0 72.0 72.0</td><td>1.1 17.2 6.2</td><td>1.0 18.6 5.3</td><td>91.0 91.0 4.3 16.7</td></tr><tr><td>Airat ICP</td><td>T-FCP ZS SCP GD</td><td>SO-LDA</td><td>69.7 |24.8 41.4</td><td>90.32.9 |90.01.3 90.1 2.3</td><td>66.0 76.0 65.0</td><td>3.2 18.4 8.8</td><td>2.0 18.1 8.8</td><td>43.2 0.8 3.7</td><td></td><td>95.0 2.0 95.2 1.2 95.01.8</td><td>70.0 90.0 70.0</td><td>6.1 28.6 12.7</td><td>3.4 28.0 12.7</td><td>29.7 0.1 2.1</td></tr><tr><td>Catech</td><td>T-FCP ICP ZS SCP GD</td><td>SO-LDA</td><td>39.6 |95.1 97.8</td><td>90.2 2.1 90.41.6 94.91.4</td><td>70.0 88.0 100.0</td><td>9.4 0.9 1.0</td><td>8.9 1.0 1.0</td><td>7.2 93.4 95.5</td><td></td><td>95.1 1.5 |96.21.0 97.8 0.9</td><td>78.0 100.0 100.0</td><td>13.1 1.0 1.0</td><td>12.5 1.0 1.0</td><td>5.9 94.0 99.0</td></tr><tr><td>UC101 ICP SCP</td><td>T-FCP ZS GD</td><td>SO-LDA</td><td>97.9 |67.6 89.3</td><td>94.1 0.9 90.1 1.7 90.3 3.0</td><td>100.0 74.0 70.0</td><td>1.0 2.9 1.0</td><td>1.0 2.0 1.0</td><td>94.7 29.3 90.5</td><td></td><td>97.3 0.9 95.11.4 95.1 1.8</td><td>100.0 80.0 68.0</td><td>1.0 5.1 1.3</td><td>1.0 3.9 1.0</td><td>98.4 16.4 75.1</td></tr><tr><td>Fo0o01 ICP SCP</td><td>T-FCP ZS GD</td><td>SO-LDA</td><td>89.6 86.0 86.3</td><td>90.5 1.9 |90.1 2.0 90.02.9</td><td>86.0 72.0 60.0</td><td>1.0 1.1 1.1</td><td>1.0 1.0 1.0</td><td>94.8 85.1 86.8</td><td></td><td>95.41.4 95.1 1.4 94.82.0</td><td>86.0 72.0 60.0</td><td>1.3 1.6 1.5</td><td>1.0 1.0 1.0</td><td>79.2 63.4 66.5</td></tr><tr><td>FLoers ICP SCP</td><td>T-FCP ZS GD</td><td>P SO-LDA</td><td>86.4 |71.5 96.5</td><td>90.4 1.9 |90.3 0.8 92.22.1</td><td></td><td>80.0 98.0 100.0</td><td>1.2 1.0 4.9 3.9 0.9 1.0</td><td>87.8 19.7 93.0</td><td></td><td>95.4 1.6 |95.2 0.8 96.31.4</td><td>90.0 96.0 98.0</td><td>1.7 11.1 1.0</td><td>1.0 9.4 1.0</td><td>68.3 5.67 96.9</td></tr><tr><td>ICP Cars SCP</td><td>T-FCP ZS GD T-FCP SO-LDA</td><td>SO-LDA</td><td>95.7 |65.5 78.6</td><td>90.9 1.6 |90.1 1.0 90.3 1.8</td><td></td><td>96.0 86.0 2.4 90.0 90.0</td><td>0.9 1.0 2.0 1.5 1.0 1.4 1.0</td><td>92.9 27.6 59.3 65.4</td><td></td><td>95.81.1 |95.2 0.9 95.21.1 95.3 0.9</td><td>100.0 88.0 86.0 96.0</td><td>1.0 3.4 2.0 1.9</td><td>1.0 3.0 2.0 2.0</td><td>96.1 17.3 38.3 45.9</td></tr><tr><td>SUN397 ICP</td><td>ZS SCP GD T-FCP SO-LDA</td><td></td><td>80.8 |62.6 73.7 74.3</td><td>90.4 1.2 |90.2 1.0 90.01.5 90.31.4</td><td></td><td>90.0 80.0 84.0</td><td>3.5 3.0 1.9 2.0 2.0 1.2</td><td>16.2 40.8 51.3</td><td></td><td>|95.1 0.7 95.1 1.1 95.31.0</td><td>98.0 84.0 94.0</td><td>6.6 3.2 3.4</td><td>5.3 2.9 2.0</td><td>7.1 22.0 31.2</td></tr><tr><td>Imaggeet ICP SCP</td><td>ZS GD T-FCP SO-LDA</td><td></td><td>|68.7 72.5 72.9</td><td>|90.00.7 90.00.8 90.30.7</td><td></td><td>90.0 85.0 100.0</td><td>2.8 2.3 2.5</td><td>2.0 28.9 2.0 36.9 2.0 46.7</td><td></td><td>|95.00.6 95.00.5 95.4 0.4</td><td>94.0 94.0 100.0</td><td>5.5 4.2 5.7</td><td>4.0 3.0 3.0</td><td>15.0 20.1 27.9</td></tr></table>

## F.3 Results using APS

We report results using APS [40] as the non-conformity score in Table 6. They demonstrate that the gains in empirical coverage stability are not specific to LAC, but also generalize to APS: T-FCP reports smaller coverage dispersion and a larger ratio of realizations with satisfied coverage than ICP and SCP. We also incorporate class-conditional average coverage gap (CCV), the reference metric used for measuring adaptiveness [9], where T-FCP remains consistent.

It is worth noting that adaptive scores such as APS/RAPS require ranking or sorting over the label space. This adds non-negligible cost inside the candidate-wise FCP loop, which conflicts with our scalability focus. For example, for ImageNet, T-FCP [α = 0.5%, α = 9.5%] using LAC has a latency of nearly 30 ms/image, and it increases to nearly 220 ms/image using APS, which makes it currently less appealing to use for the largest-scale datasets for T-FCP and unfeasible for the more expensive vanilla FCP.

Table 6: Results using APS. SO-LDA is used as solver, with $N = C \times 1 6$ . Average results for 10 datasets. These studies use 20 random seeds and exclude ImageNet.
<table><tr><td rowspan="2"></td><td colspan="4"> $\mathrm { C o v . } \ ( \alpha = 0 . 1 0 )$ </td><td colspan="3">Set size</td><td colspan="4"> $\mathrm { C o v . } \ ( \alpha = 0 . 0 5 )$ </td><td colspan="3">Set size</td></tr><tr><td>|Avg.</td><td>2σ</td><td>%Valid</td><td>CCV</td><td></td><td>Mean Med.</td><td>%Sing. |</td><td>|Avg.</td><td>2σ</td><td>%Valid</td><td>CCV</td><td>Mean</td><td></td><td>Med. %Sing.</td></tr><tr><td>ICP</td><td></td><td>90.42.2</td><td>81.5</td><td>8.2</td><td>6.3</td><td>4.9</td><td>34.4</td><td>95.3</td><td>1.5</td><td>85.0</td><td>5.0</td><td>10.1</td><td>8.0</td><td>27.1</td></tr><tr><td>SCP</td><td></td><td>90.1 3.1</td><td>67.5</td><td>6.3</td><td>3.0</td><td>2.2</td><td>54.0</td><td>95.1</td><td>2.4</td><td>70.0</td><td>5.0</td><td>4.3</td><td>3.1</td><td>46.9</td></tr><tr><td>T-FCP</td><td></td><td>90.4 1.7</td><td>84.0</td><td>6.4</td><td>3.3</td><td>2.3</td><td>52.6</td><td>95.3</td><td>1.4</td><td>89.0</td><td>4.9</td><td>4.5</td><td>2.9</td><td>46.9</td></tr></table>

## F.4 Extension to unimodal models

The T-FCP construction is model-agnostic in principle: any valid low-cost conformal predictor could be used as the pruning stage, and any valid or efficient FCP scoring rule could be used as the second stage. Beyond VLMs, T-FCP can therefore be explored for unimodal models as long as an initial pruning proxy exists, e.g., a pretrained classifier over the target label space.

To test this, we next transfer a standard ResNet-50 model pretrained on ImageNet to ImageNet-V2 (accuracy 69.1%), since the two datasets share the same label space. T-FCP $\bar { [ \alpha } _ { \mathrm { I C P } } = 0 . 5 \% , \ \alpha _ { \mathrm { F C P } } =$ 4.5%] early prunes 65% of the labels, and provides a more stable empirical coverage distribution than split-conformal alternatives. Specific results are presented below, in Table 7.

Table 7: Conformal prediction results on ImageNet-V2 using ResNet-50 pre-trained on ImageNet, LAC, and 8 labeled samples per class as calibration data. Results across 20 seeds.
<table><tr><td rowspan="2"></td><td colspan="3">Cov. (α = 0.05)</td><td>Set size</td></tr><tr><td>1  $\operatorname { A v g } .$ </td><td> $2 \sigma$ </td><td>%Valid</td><td>Mean</td></tr><tr><td>ICP</td><td>95.2</td><td>1.3</td><td>80.0</td><td>12.3</td></tr><tr><td>SCP</td><td>95.0</td><td>1.2</td><td>70.0</td><td>11.7</td></tr><tr><td>T-FCP</td><td>95.1</td><td>1.2</td><td>80.0</td><td>10.6</td></tr></table>
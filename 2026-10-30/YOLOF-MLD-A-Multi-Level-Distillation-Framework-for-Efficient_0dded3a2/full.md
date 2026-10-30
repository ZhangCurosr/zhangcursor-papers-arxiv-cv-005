# YOLOF-MLD: A Multi-Level Distillation Framework for Efficient Object Detection

Xiaolong Wu<sup>1</sup>, Liming Zheng<sup>1</sup>, Zitong Liu<sup>2</sup>, Yue Zheng<sup>1</sup>

<sup>1</sup> The Chinese University of Hong Kong (Shenzhen), Shenzhen, Guangdong, 518172, P.R. China

<sup>2</sup>The Chinese University of Hong Kong, Shatin, New Territories, Hong Kong SAR, China

Email: 224010119@link.cuhk.edu.cn, 125090999@link.cuhk.edu.cn, 1155248030@link.cuhk.edu.hk, zhengyue@cuhk.edu.cn

Abstract—Efficient object detection on resource-constrained edge devices is essential for safety-critical applications such as fire-smoke monitoring. However, current lightweight frameworks, such as YOLO26, still rely on the parameter-dense multi-branch Feature Pyramid Network (FPN), preventing full exploitation of their lightweight advantage. Replacing FPN with YOLOF’s single-input-single-output (SiSo) neck is a promising solution but causes substantial accuracy loss. To alleviate this, we propose YOLOF-MLD, a CPU-efficient multi-level distillation (MLD) framework that combines a YOLO26n backbone, a SiSo neck, and MLD. It aligns features at the backbone via sliced-semantics matching, neck with receptive-field matching, and response levels. On D-Fire and Annotated Fire-Smoke (AFS) datasets, YOLOF-MLD reduces the parameter count of YOLO26n by 35.7% and achieves a 62.3% speedup on Raspberry Pi 5, attaining 71.6% and 59.9% mAP50, respectively, compared with 75.5% and 60.0% for YOLO26n. These results demonstrate a competitive accuracy–latency trade-off for edge deployment. Our code is available at https://github.com/XiaolongWU111/YOLOF---MLD/ blob/main/README.md.

Index Terms—Lightweight object detection, edge deployment, knowledge distillation, fire-smoke detection

## I. INTRODUCTION

Edge-based visual detection is increasingly required in safety-critical applications such as fire-smoke monitoring and traffic surveillance. Deploying detectors on edge platforms is often constrained by limited computational capability, memory bandwidth, and power budgets. Despite these constraints, models must maintain high accuracy and timely alarms, as missed detections or delayed responses may endanger human lives and cause substantial economic losses. Therefore, developing lightweight, high-performance, and low-latency object detectors has become a critical requirement.

To address this, lightweight detectors have been proposed, which mainly follow two strategies. The first is efficient architecture design (e.g., MobileNet-SSDv2 [1], NanoDet-Plus [2], PP-PicoDet [3], and tiny YOLO variants [4]), which reduces computation but sacrifices representation capability. The second is knowledge distillation (e.g., GID [5], SKD [6], CrossKD [7], and CanKD [8]), which recovers accuracy without architectural changes, but struggles with large teacherstudent feature discrepancies.

To the best of our knowledge, YOLO26 [9] achieves the best accuracy–speed trade-off through its optimized architecture design. Nevertheless, YOLO26 [9] still relies on a multibranch feature pyramid network (FPN) [13] neck, which utilizes up-sampling, concatenation, and extra convolutions to fuse multi-level features, complicating the topological structure and increasing inference latency.

Replacing FPN [13] with the YOLOF’s [10] single-inputsingle-output (SiSo) neck can significantly reduce redundant computation and boost inference speed. However, directly coupling the SiSo [10] neck with YOLO26’s lightweight backbone and framework is non-trivial. Our empirical observations show that this direct replacement causes a drastic accuracy drop, as the backbone is too lightweight to supply sufficient feature representation, violating YOLOF’s “one-level feature carries enough context” assumption. Furthermore, the conventional YOLOF neck relies on naive dilated convolutions, which lacks the flexibility to adapt to targets with varying scales and irregular shapes, hindering effective context aggregation at the neck level.

To bridge this gap, we propose YOLOF-MLD, a multilevel distillation (MLD) framework tailored for SiSo neck based lightweight detector. By progressively aligning features across the backbone, neck and response levels, our method effectively transfers the rich, multi-scale contextual knowledge of a complex FPN teacher to a highly efficient SiSo student. Our contributions are threefold:

1) We propose YOLOF-MLD, a lightweight object detector that replaces the computational expensive multi-branch FPN [13] neck with an optimized SiSo neck, significantly reducing inference latency on edge devices.

2) We design an MLD framework comprising sliced semantics matching (SSM), receptive field matching (RFM), and response knowledge distillation (RKD) to effectively compensate for the representational loss caused by the architectural simplification.

3) We extensively evaluate the approach on two challenging fire-smoke benchmarks: D-Fire [14] and the annotated fire-smoke (AFS) dataset [15] on CPU platforms. Compared to the YOLO26n [9] baseline, our YOLOF-MLD reduces the parameter count by 35.7% and achieves relative speedups of 62.3% and 76.2% on Raspberry Pi 5 and a 4-thread Intel CPU, respectively, while preserving nearly identical accuracy on AFS (59.9% vs. 60.0% mAP50). Despite using more parameters, it also achieves more than 20% relative speedups than YOLOXnano [11] and NanoDet-plus-m [2] on both platforms.

## II. PRELIMINARY

## A. FPN Neck [13]

Multi-scale feature fusion is essential for detecting objects of varying sizes. The mainstream FPN utilizes a multi-branch topology, where high-level semantic features are progressively up-sampled and laterally concatenated with low-level features. Although effective, this architecture introduces complex memory access patterns and significant computational overhead. On resource-constrained edge CPUs, this multi-branch topology severely bottlenecks inference latency.

## B. YOLOF’s Dilated SiSo Neck [10]

To eliminate the FPN bottleneck, YOLOF proposes a highly streamlined SiSo neck. Instead of fusing multiple scales, the SiSo neck takes only the deepest feature map from the backbone and processes it through sequentially stacked dilated convolution blocks. By introducing gaps between convolution weights, the SiSo neck expands the receptive field to capture global context without the latency penalty of up-sampling and concatenation.

## III. METHODOLOGY

This section details the proposed MLD framework, as illustrated in Fig. 1. The teacher consists of a backbone $B ^ { T } ( \cdot )$ and employs a multi-branch FPN [13] neck $N ^ { T } ( \cdot )$ that generates three outputs with varying receptive fields. In contrast, the student comprises a backbone $B ^ { \bar { S _ { \left( . \right) } } }$ and utilizes a lightweight, single-branch SiSo neck $N ^ { S } ( \cdot )$ comprising three concatenated dilated convolutions. To bridge the resulting representational gap, we propose to align the networks progressively in backbone, neck, and response levels.

Throughout the paper, superscript <sup>T</sup> and <sup>S</sup> denote teacherand student- related variables, respectively, while c and d denote the channel and spatial dimensions of feature maps.

![](images/121da2719293c18b10a5ee5dcb0af5a68ca7f94ea5c3c17c3cd92cecf1d3b319.jpg)  
Fig. 1: Framework of YOLOF-MLD.

## A. Backbone-level Sliced-semantics Matching (SSM)

Conventional feature distillation typically aligns teacher and student features at corresponding depths. Although teacher’s deep layer features provide strong semantic supervision, they have also undergone repeated downsampling and abstraction, and may therefore contain limited fine-grained spatial details. Such details are particular valuable for compensating the representation loss of a compact student. In contrast, the teacher’s early-stage features preserve higher-resolution local structures and appearance cues, which motivates us to use an early-stage teacher feature $\boldsymbol { f } ^ { T } \in \mathbb { R } ^ { c ^ { T } \times d ^ { T } }$ to supervise the student’s late-stage feature $\mathbf { \boldsymbol { f } } ^ { S } \in \mathbb { R } ^ { c ^ { S } \times d ^ { S } }$ , enabling the final student representation to retain fine-grained information beyond conventional same-stage distillation.

However, directly aligning $f ^ { T }$ and $f ^ { S }$ is problematic. The teacher’s early-stage feature contains massive, highdimensional raw information that easily overwhelms the student if forced into a rigid, element-wise alignment. Besides, $f ^ { T }$ and $f ^ { S }$ differs significantly in both spatial dimensionality and feature distribution. Classical methods [8] [12] attempt to bridge this gap by using parameterized projection layers. However, these methods either suffer from the teacher’s knowledge loss during the projection process, or induce the student model to rely too much on the projection branch, reducing the gain of distillation.

To overcome this dimension-distribution mismatch without relying on lossy projections, we propose SSM, which is an optimal transport (OT) inspired distillation. Specifically, given $f ^ { \dot { T } }$ and $f ^ { S }$ , we first slice them along the channel dimension: $\boldsymbol { f } ^ { T } = \left\{ \boldsymbol { f } _ { i } ^ { T } \right\} _ { i = 1 } ^ { c ^ { T } } , \boldsymbol { f } ^ { S } = \left\{ \boldsymbol { f } _ { j } ^ { S } \right\} _ { i = 1 } ^ { c ^ { S } }$ . Since the two sliced features differ in spatial dimension, we utilize a lightweight projector $P ( \cdot )$ to align the student’s spatial resolution to the teacher’s, yielding $f _ { j } ^ { \breve { S ^ { \prime } } } = { \cal P } ( f _ { j } ^ { S } ) \in \mathbb { R } ^ { d ^ { \hat { T } } }$ , where $P ( \cdot )$ is composed only of depth-wise convolutions. This architectural constraint limits the projector’s capacity, effectively preventing the student from over-relying on the auxiliary branch.

Using these aligned features, we establish the matching relationship (i.e., the transport plan) via a joint distribution $\pi ( f _ { j } ^ { S } , f _ { i } ^ { T } ) \dot { = } \pi ( f _ { i } ^ { T } | f _ { j } ^ { S } ) \times \dot { P r } ( f _ { j } ^ { \hat { S } } )$ . Since the exact conditional matching probability $\pi ( f _ { i } ^ { T } | f _ { j } ^ { S } )$ is unknown and computationally expensive to strictly solve, we approximate it using a Softmax operation over the cosine similarities between slices. Assuming a uniform marginal distribution for the student slices $P r ( f _ { j } ^ { S } ) \stackrel { } { = } 1 / c ^ { S }$ , the joint distribution is then approximated as:

$$
\begin{array} { r l } & { \pi ( f _ { j } ^ { S } , f _ { i } ^ { T } ) = \pi ( f _ { i } ^ { T } | f _ { j } ^ { S } ) \cdot \frac { 1 } { c ^ { S } } } \\ & { \approx \frac { \exp ( \langle f _ { j } ^ { S ^ { \prime } } , f _ { i } ^ { T } \rangle / \tau _ { 1 } ) } { \sum _ { k = 1 } ^ { c ^ { T } } \exp ( \langle f _ { j } ^ { S ^ { \prime } } , f _ { k } ^ { T } \rangle / \tau _ { 1 } ) } \cdot \frac { 1 } { c ^ { S } } } \end{array}\tag{1}
$$

where $\tau _ { 1 }$ is the temperature. To encourage confident and precise semantic matching, we optimize the projector $P ( \cdot )$ via joint entropy minimization:

$$
\begin{array} { r l } & { J _ { p r o } = H \big ( \pi ( f _ { j } ^ { S } , f _ { i } ^ { T } ) \big ) } \\ & { \qquad = - \Sigma _ { i } \Sigma _ { j } \pi ( f _ { j } ^ { S } , f _ { i } ^ { T } ) \cdot \log \big ( \pi ( f _ { j } ^ { S } , f _ { i } ^ { T } ) \big ) } \end{array}\tag{2}
$$

By minimizing $J _ { p r o }$ , the projector is forced to discover unambiguous fine-grained co-relationship $\pi ( f _ { j } ^ { S } , f _ { i } ^ { T } )$ between sliced semantics. Consequently, for each student slice $f _ { j } ^ { S }$ , we can identify its most similar teacher slice $f _ { i } ^ { T }$ by maximizing the joint distribution:

$$
f _ { i } ^ { T } = \arg \operatorname* { m a x } _ { f _ { k } ^ { T } } \pi ( f _ { j } ^ { S } , f _ { k } ^ { T } )\tag{3}
$$

The distillation objective is then formulated to minimize the distance between these optimally matched pairs, which approximates the semi-relaxed Kantorovich OT problem [16]:

$$
J _ { S S M } \approx \operatorname* { i n f } _ { \pi \in \Pi } \Sigma _ { j = 1 } ^ { c ^ { S } } \Sigma _ { i = 1 } ^ { c ^ { T } } \Vert f _ { j } ^ { S ^ { \prime } } - f _ { i } ^ { T } \Vert _ { 2 } ^ { 2 } \cdot \pi ( f _ { j } ^ { S } , f _ { i } ^ { T } )\tag{4}
$$

where Π denotes the complete set of joint distributions. This OT-inspired formulation provides a highly fine-grained optimization objective without imposing rigid point-to-point constraints.

However, since the teacher’s early-stage features inevitably contain background noise irrelevant to the target task, directly applying Eq. 4 is sub-optimal. To alleviate this, we introduce a spatial attention mask $M ^ { T } \in \mathbb { R } ^ { d ^ { T } }$ with mean value normalization $V ( \cdot )$ to filter out the unrelated noise by averaging the teacher’s channels: $M ( f ^ { T } ) = \Sigma _ { i } f _ { i } ^ { T } / c ^ { T }$ . The final spatiallyweighted SSM objective becomes:

$$
J _ { S S M } \approx \underset { \pi \in \Pi } { \operatorname* { i n f } } \Sigma _ { j } ^ { c ^ { S } } \Sigma _ { i } ^ { c ^ { T } } | | f _ { j } ^ { S ^ { \prime } } - f _ { i } ^ { T } | | _ { 2 } ^ { 2 } \cdot V ( M ( f ^ { T } ) ) \cdot \pi ( f _ { j } ^ { S } , f _ { i } ^ { T } ) .\tag{5}
$$

## B. Neck-level Receptive-field Matching (RFM)

Formally, the teacher’s FPN neck generates three multiscale output features $O _ { 1 } ^ { T } , O _ { 2 } ^ { T } , O _ { 3 } ^ { T }$ . These features naturally capture contexts with progressively larger receptive fields, i.e., $\tilde { R ( { O } _ { 1 } ^ { T } ) } < R ( { O } _ { 2 } ^ { T } ) < \tilde { R } ( \bar { O _ { 3 } ^ { T } } )$ , where $R ( \cdot )$ is the receptive field measure function. In contrast, the student relies on a SiSo neck $N ^ { S } ( \cdot )$ , which sequentially stacks three convolution blocks: $O _ { 1 } ^ { S } = N _ { 1 } ^ { S } ( f ^ { S } ) , \bar { O _ { 2 } ^ { S } } = N _ { 2 } ^ { \bar { S } } ( O _ { 1 } ^ { S } )$ and $O _ { 3 } ^ { S } = N _ { 3 } ^ { S } ( O _ { 2 } ^ { S } )$ ).

The baseline SiSo design (e.g., YOLOF) relies on standard dilated convolutions. By introducing gaps between convolution weights, dilated convolutions successfully expand the receptive field, achieving $R ( O _ { 1 } ^ { S } ) \ < \ R ( O _ { 2 } ^ { S } ) \ < \ \bar { R } ( O _ { 3 } ^ { S } )$ , without adding computational overhead. However, their sampling grids remain strictly rigid and uniformly distributed, hindering the network’s ability to model targets with extreme scale variations and irregular geometric deformations.

To alleviate this, we propose RFM distillation to guide the student’s stacked dilated blocks to mimic the FPN’s aggregation mechanism. We accomplish this in a divide-andconquer manner by strictly aligning features with comparable receptive-field scales:

$$
J _ { R F M } = \Sigma _ { i = 1 } ^ { 3 } | | O _ { i } ^ { T } - P _ { i } ( O _ { i } ^ { S } ) | | _ { 2 } ^ { 2 }\tag{6}
$$

where $P _ { i } ( \cdot )$ denotes projectors to match the feature dimensions. By minimizing $J _ { R F M }$ , the adaptive SiSo neck implicitly acquires the rich multi-scale fusion capability of the FPN, while circumventing the inference latency of a complex multibranch topology.

## C. Response-level Knowledge Distillation (RKD)

Beyond intermediate feature alignment, we introduce RKD to directly utilize the teacher’s predictions to supervise the student. Specifically, let $\{ I _ { i } ^ { T } ~ = ~ ( b _ { i } ^ { T } , z _ { i } ^ { T } ) | _ { i = 1 } ^ { n ^ { T } } \}$ and $\{ I _ { j } ^ { S } ~ =$ $( b _ { j } ^ { S } , z _ { j } ^ { S } ) | _ { j = 1 } ^ { n ^ { S } } \}$ denote the final predictions of the teacher and the student, respectively, where b represents the bounding box coordinates, z is the classification logits, and n is the number of predicted boxes.

For every student’s bounding box, we first find the corresponding teacher’s bounding box via the assignment rule $A ( \cdot )$ of YOLO26 [9], i.e, $\{ I _ { j } ^ { T } = ( b _ { A ( j ) } ^ { T } , z _ { A ( j ) } ^ { T } ) | _ { j = 1 } ^ { n ^ { S ^ { - } } } \}$ . Then we align the two matched predictions. In detail, for bounding box alignment, we utilize IoU loss, and for probability distribution alignment, we utilize KL divergence with temperature τ<sub>2</sub>, i.e.,

$$
\begin{array} { r l } & { J _ { \mathrm { R K D } } = \boldsymbol { \Sigma } _ { j = 1 } ^ { n ^ { S } } \biggl [ 1 - I o U ( b _ { A ( j ) } ^ { T } , b _ { j } ^ { S } ) } \\ & { ~ + \tau _ { 2 } ^ { 2 } \cdot \mathrm { K L } ( z _ { j } ^ { S } / \tau _ { 2 } \parallel z _ { A ( j ) } ^ { T } / \tau _ { 2 } ) \biggr ] , } \end{array}\tag{7}
$$

## D. Online Distillation and Overall Objective

Finally, we train the entire MLD framework using an online distillation paradigm, optimizing both models in parallel. The total loss is:

$$
\begin{array} { r l } & { J _ { t o t a l } = J _ { t a s k } ^ { S } + J _ { t a s k } ^ { T } + ( \lambda _ { p r o } J _ { p r o } + \lambda _ { S S M } J _ { S S M } } \\ & { ~ + \lambda _ { R F M } J _ { R F M } + \lambda _ { R K D } J _ { R K D } ) \cdot s } \end{array}\tag{8}
$$

where $J _ { t a s k } ^ { S }$ and $J _ { t a s k } ^ { T }$ denote the standard detection losses for student and teacher using ground-truth hard labels. The hyperparameters $\lambda _ { p r o } , \lambda _ { S S M } , \lambda _ { R F M }$ and $\lambda _ { R K D }$ are empirically set to 0.01, 0.01, 0.008, 0.12 by default to balance the MLD terms, s denotes batch size.

## IV. EXPERIMENTS

## A. Experiment Settings

Datasets. We evaluated our method on 2 datasets: D-Fire [14], a large-scale dataset comprising 21,527 images of fire and smoke, and Annotated fire-smoke dataset (AFS) [15], which is composed of 11027 images of fire and smoke. Both of them effectively simulate harsh real-world environment demanding efficient edge deployment.

Implementation Details. We choose YOLO26n as the backbone of both teacher and student, and aim to adapt the FPN-based YOLO26n baseline for efficient edge deployment. The temperatures are empirically set to $\tau _ { 1 } = 0 . 0 7$ for SSM and $\tau _ { 2 } = 3$ for RKD. To evaluate inference efficiency across different CPU-based edge settings, we benchmark the models on a Raspberry Pi 5 (Model B Rev 1.1) and an Intel(R) Core(TM) i7-10750H. The latter represents a higher-performance edge gateway. Inference on both platforms is restricted to four CPU threads to emulate constrained per-stream compute resources. Models are trained in Pytorch [18], exported and executed via ONNX [17] Runtime with a batch size of 1.

## B. Performance and Comparison

To validate the effectiveness of our proposed framework, we compare it against two groups of baselines:

• Lightweight Detectors: Tiny-YOLOv3 [4], NanoDet-m-0.5x [2], NanoDet-Plus-m [2], and YOLOX-Nano [11].

• Distillation Methods: GID [5], SKD [6], GKD [12], CrossKD [7], and CanKD [8].

TABLE I: Comparisons with Lightweight Object Detectors
<table><tr><td rowspan="2">Model</td><td rowspan="2">Param.</td><td colspan="2">FPS (Avg)</td><td colspan="2">D-Fire [14]</td><td colspan="2">AFS [16]</td></tr><tr><td>Raspberry Pi Intel CPU</td><td></td><td>mAP50 mAP50-95</td><td></td><td></td><td>mAP50 mAP50-95</td></tr><tr><td>TN-YOLOv3 [4]</td><td>12.13M</td><td>3.79</td><td>15.81</td><td>71.5</td><td>37.1</td><td>54.7</td><td>28.3</td></tr><tr><td>NanoDet-m-0.5x [2]</td><td>0.26M</td><td>26.23</td><td>97.67</td><td>70.8</td><td>33.5</td><td>56.2</td><td>27.2</td></tr><tr><td>NanoDet-plus-m [2]</td><td>1.17M</td><td>9.60</td><td>40.67</td><td>77.8</td><td>43.9</td><td>56.4</td><td>32.1</td></tr><tr><td>YOLOX-nano [11]</td><td>0.90M</td><td>10.06</td><td>49.58</td><td>72.2</td><td>41.1</td><td>57.8</td><td>32.0</td></tr><tr><td>YOLO26n [9]</td><td>2.38M</td><td>7.86</td><td>33.92</td><td>75.5</td><td>43.4</td><td>60.0</td><td>33.5</td></tr><tr><td>YOLOF-MLD (Ours)</td><td>1.53M</td><td>12.76</td><td>59.78</td><td>71.6</td><td>38.8</td><td>59.9</td><td>32.6</td></tr></table>

Note: FPS (Avg) is the average FPS across both datasets, excluding pre-processing and post-processing (e.g., non-max suppression).

1) Comparison with lightweight detectors: Baseline lightweight detectors are trained using their default settings and official codes, and we set the maximum input image size as 640 under their different multi-scale training strategy. For inference evaluation, we use an input resolution of 640 × 640 and report the frame-per-second (FPS) on the Raspberry Pi and the Intel CPU platforms, measured from model-forward latency only and excluding pre- and post-processing. For detection accuracy, we adopt the standard mean average precision (mAP) metrics: mAP@0.5 and mAP@0.5-0.95, which are denoted as mAP50 and mAP50-95 for brevity.

Table I compares YOLOF-MLD against existing lightweight detectors across two datasets. Compared with the original YOLO26n baseline, our method reduce the parameter count from 2.38M to 1.53M and achieves a 62.34% relative speedup (7.86 to 12.76 FPS) on the Raspberry Pi 5 and 76.2% relative speedup (33.92 to 59.78 FPS) on Intel CPU. This acceleration introduces an accuracy trade-off on D-Fire (mAP50/mAP50-95: 75.5%/43.4% to 71.6%/38.8%), while nearly preserving the baseline accuracy on AFS (60.0%/33.5% vs. 59.9%/32.6%). Compared with the ultralightweight NanoDet-m-0.5x, YOLOF-MLD prioritizes detection accuracy over maximum throughput, achieving higher mAP50/mAP50-95 on both datasets. Moreover, despite having more parameters, YOLOF-MLD outperforms YOLOX-nano and NanoDet-plus-m in inference speed, achieving relative speed up of 26.8%/20.57% and 32.9%/46.99% on Raspberry Pi 5/Intel CPU, respectively. These results suggest that parameter count alone does not characterize practical CPU inference efficiency, and a streamlined SiSo topology can offer favorable balance between detection accuracy and deployment speed.

2) Comparison with distillation methods: For all distillation-related experiments, we adopt a unified training configuration for fair comparison. Specifically, the baseline and competing KD methods (GID, SKD, GKD, CrossKD, and CanKD) are trained for 200 epochs with a batch size of 32 using the MuSGD optimizer, an initial learning rate of 0.01 (for CanKD [8], the learning rate is set as 0.001 for numerical stability), and a linear decay schedule. All methods use the same student architecture, consisting of the YOLO26n [9] backbone and YOLOF [10]’s dilated SiSo neck, and follow the original YOLO26 label-assignment rule.

As shown in Table II, most general-purpose methods provide weak improvements on the baselines and may even cause negative transfer; for example, SKD improves the baseline by

TABLE II: Comparison of Distillation and Ablation Study
<table><tr><td rowspan="2">Method</td><td colspan="2">D-Fire [14]</td><td colspan="2">AFS [15]</td></tr><tr><td>mAP50</td><td>mAP50-95</td><td>mAP50</td><td>mAP50-95</td></tr><tr><td>Baseline1</td><td>69.8</td><td>37.8</td><td>58.8</td><td>31.6</td></tr><tr><td>GID [5]</td><td></td><td>70.0(+0.2) 37.8 (+0.0) 58.2 (-0.6) 31.7 (+0.1)</td><td></td><td></td></tr><tr><td>SKD [6]</td><td></td><td></td><td>70.3 (+0.5) 37.3 (-0.5) 60.3 (+1.5) 32.4 (+0.8)</td><td></td></tr><tr><td>GKD [12]</td><td></td><td>69.5 (-0.3) 36.4 (-1.4) 55.8 (-3.0) 29.5 (-2.1)</td><td></td><td></td></tr><tr><td>CrossKD [7]</td><td></td><td></td><td>70.2 (+0.4) 38.0 (+0.2) 59.2 (+0.4) 31.7 (+0.1)</td><td></td></tr><tr><td>CanKD [8]</td><td></td><td></td><td>70.4 (+0.6) 38.0 (+0.2) 59.5 (+0.7) 31.9 (+0.3)</td><td></td></tr><tr><td>+ Uniform Matching [10]2</td><td></td><td>65.9(-3.9) 34.9 (-2.9) 58.6(-0.2)</td><td></td><td>30.0(-1.6)</td></tr><tr><td>+ RFM</td><td></td><td></td><td>70.2 (+0.4) 38.1 (+0.3) 58.4 (-0.4) 31.8 (+0.2)</td><td></td></tr><tr><td>+ RFM + SSM</td><td></td><td></td><td>70.5 (+0.7) 37.8 (+0.0) 59.7 (+0.9) 32.2 (+0.6)</td><td></td></tr><tr><td>+ RFM + SSM + RKD(ours) 71.6 (+1.8) 38.8 (+1.0) 59.9 (+1.1) 32.6 (+1.0)</td><td></td><td></td><td></td><td></td></tr></table>

Note: <sup>1</sup>Student without distillation; <sup>2</sup> Top k=4.

1.5 percentage points of mAP50 on AFS but decreases by 0.5 percentage points of mAP50-95 on D-Fire. In contrast, our MLD strategy consistently improves the baseline by 1.8/1.0 mAP50/mAP50-95 on D-Fire and 1.1/1.0 on AFS. Notably, it also outperforms the strongest competitor (CanKD) by a significant margin of +0.8 and +0.7 mAP50-95 on D-Fire and AFS, respectively, demonstrating the effectiveness of the proposed multi-level transfer for the FPN-to-SiSo setting.

## C. Ablation Study

We conduct ablations on the proposed YOLOF-MLD framework. The baseline consists of the YOLO26n backbone and the SiSo neck, trained without distillation while retaining the original YOLO26n dynamic label-assignment rule. We first replace this rule with YOLOF’s Uniform Matching [10], which encourages a more balanced positive-negative label assignment ratio to compensate for the accuracy drop caused by replacing FPN with dilated SiSo neck. However, it reduces overall performance on D-Fire datasets by up to 3.9 mAP50 and on AFS datasets by up to 1.6 mAP50-95, possibly because sparse prediction locations on low-resolution SiSo feature maps may yield low-quality positive samples with substantial background content. We then add RFM, SSM, and RKD individually to the baseline. As shown in Table II, every sub-method continues to improve the baselines; combining all components yields a cumulative gain of 1.8/1.1 mAP50 and 1.1/1.0 mAP50-95.

## D. Conclusion and Future Works

In conclusion, we presented YOLOF-MLD, a CPU-efficient object detector that combines a YOLO26n backbone with a lightweight SiSo neck derived from YOLOF. By incorporating multi-level knowledge distillation, the proposed framework mitigates the accuracy degradation caused by replacing the multi-branch FPN neck. Experiments on two fire-smoke detection datasets demonstrate that YOLOF-MLD substantially improves the inference efficiency of the YOLO26n baseline on both Raspberry Pi and thread-constrained Intel CPU, while maintaining competitive detection accuracy. In future work, we will further investigate backbone optimization and its integration with the proposed MLD framework to reduce remaining computational redundancy.

## REFERENCES

[1] Y.-C. Chiu, C.-Y. Tsai, M.-D. Ruan, G.-Y. Shen, and T.-T. Lee, “Mobilenet-SSDv2: An improved object detection model for embedded systems,” in 2020 International conference on system science and engineering (ICSSE), 2020, pp. 1–5.

[2] RangiLyu, “NanoDet-Plus: Super fast and high accuracy lightweight anchor-free object detection model,” 2021. [Online]. Available: https://github.com/RangiLyu/nanodet

[3] G. Yu et al., “PP-PicoDet: A better real-time object detector on mobile devices,” arXiv preprint arXiv:2111.00902, 2021.

[4] P. Adarsh, P. Rathi, and M. Kumar, “YOLO v3-Tiny: Object Detection and Recognition using one stage improved model,” in 2020 6th international conference on advanced computing and communication systems (ICACCS), 2020, pp. 687–694.

[5] X. Dai et al., “General instance distillation for object detection,” in 2021 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), IEEE, 2021, pp. 7838–7847.

[6] P. De Rijk, L. Schneider, M. Cordts, and D. Gavrila, “Structural knowledge distillation for object detection,” Advances in Neural Information Processing Systems, vol. 35, pp. 3858–3870, 2022.

[7] J. Wang, Y. Chen, Z. Zheng, X. Li, M.-M. Cheng, and Q. Hou, “CrossKD: Cross-head knowledge distillation for object detection,” in 2024 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), IEEE, 2024, pp. 16520–16530.

[8] S. Sun and W. Ohyama, “CanKD: Cross-Attention-based Non-local operation for Feature-based Knowledge Distillation,” in 2026 IEEE/CVF Winter Conference on Applications of Computer Vision (WACV), IEEE, 2026, pp. 8606–8616.

[9] G. Jocher, J. Qiu, M. Liu, S. Lyu, F. C. Akyon, and M. E. Kalfaoglu, “Ultralytics YOLO26: Unified real-time end-to-end vision models,” arXiv preprint arXiv:2606.03748, 2026.

[10] Q. Chen, Y. Wang, T. Yang, X. Zhang, J. Cheng, and J. Sun, “You only look one-level feature,” in 2021 IEEE/CVF conference on computer vision and pattern recognition (CVPR), 2021, pp. 13034–13043.

[11] Z. Ge, S. Liu, F. Wang, Z. Li, and J. Sun, “YOLOX: Exceeding YOLO series in 2021,” arXiv preprint arXiv:2107.08430, 2021.

[12] Q. Lan and Q. Tian, “Gradient-guided knowledge distillation for object detectors,” in 2024 IEEE/CVF Winter Conference on Applications of Computer Vision (WACV), 2024, pp. 423–432.

[13] T.-Y. Lin, P. Dollar, R. Girshick, K. He, B. Hariharan, and S. Belongie,´ “Feature pyramid networks for object detection,” in 2017 IEEE conference on computer vision and pattern recognition (CVPR), 2017, pp. 936–944.

[14] P. V. A. Venancio, A. C. Lisboa, and A. V. Barbosa, “An automatic fire detection system based on deep convolutional neural networks for low-power, resource-constrained devices,” Neural Computing and Applications, vol. 34, no. 18, pp. 15349–15368, 2022.

[15] S. Partheepan, “Annotated Fire -Smoke Image Dataset for fire detection Using YOLO,” 4 2025, doi: 10.25946/28747046.v1.

[16] C. Villani, “Optimal transport: old and new,” vol. 338, 2009.

[17] O. R. developers, “ONNX Runtime.” Accessed: Oct. 22, 2026. [Online]. Available: https://onnxruntime.ai/

[18] A. Paszke et al., “Pytorch: An imperative style, high-performance deep learning library,” Advances in neural information processing systems, vol. 32, 2019.
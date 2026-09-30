# STRUCTURED VISUAL TARGET LEARNING FOR CROSS-SUBJECT EEG-TO-IMAGE RETRIEVAL

Salini Yadav<sup>1</sup>, Taveena Lotey<sup>1</sup>, Mickael Coustaty ¨ <sup>2</sup>, Pravendra Singh<sup>1</sup>, Partha Pratim Roy<sup>3</sup>

<sup>1</sup>Indian Institute of Technology Roorkee, India <sup>2</sup>University of La Rochelle, France <sup>3</sup>Indian Institute of Technology (ISM) Dhanbad, India

## ABSTRACT

Cross-subject EEG-to-image retrieval requires a neural representation trained on source subjects to remain aligned with a visual embedding space for an unseen subject. Whereas existing methods primarily focus on the EEG side, we address this problem from the perspective of the visual target. Our approach preserves the spatial information of the Perception Encoder, converts its patch grid into a compact set of learned visual views, and aggregates them for each image with a block-structured, content-dependent router. The target is learned jointly with the EEG encoder through contrastive learning with MMD regularization across source subjects. For deployment, we propose a training-free representation refinement that aligns frozen embeddings without updating either encoder. Under leaveone-subject-out evaluation on THINGS-EEG2, the structured target achieves 35.3%/65.6% Top-1/Top-5 accuracy, the best among compared methods. Refinement raises this to 48.1%/77.1%, an 18.5% Top-1 gain over the strongest compared method, improving all ten held-out subjects <sup>1</sup>.

Index Terms— EEG decoding, cross-subject generalization, zero-shot EEG-to-image retrieval, visual representation learning, contrastive learning

## 1. INTRODUCTION

Visual decoding from EEG aims to recover perceived visual content from non-invasive neural recordings [1, 2]. EEG-to-image retrieval systems map neural responses and candidate images into a shared embedding space, enabling zero-shot recognition of concepts never seen as output classes during training [2, 3]. On the THINGS-EEG2 benchmark [1], progress depends on both the EEG encoder and the visual target it is trained to match [2, 4, 5].

A practically important but considerably harder setting is crosssubject retrieval. Under leave-one-subject-out (LOSO) evaluation, the model is trained on several subjects and tested on one whose recordings are never observed [6]. Because EEG responses to the same image vary substantially across individuals, a mapping learned from source subjects often transfers poorly to a new one. Differences in head anatomy, electrode placement, and individual neural response patterns shift the EEG feature distribution, so an encoder that performs well on source subjects may produce embeddings that no longer align with the visual space for the target subject [7]. Closing this gap is essential for deployment, since subject-specific labeled calibration is costly in practice.

Prior work has addressed this problem mainly from the neural side. NICE established contrastive EEG-to-image retrieval on THINGS-EEG2 with frozen visual features [3], and later methods improved the EEG encoder, multimodal supervision, or the choice of visual features [2, 8, 5, 9, 10, 11]. SAMGA introduced subjectaware multi-granularity alignment for zero-shot EEG-to-image retrieval [4]. Classical cross-subject BCI methods, such as Euclidean alignment, hyperalignment, and Riemannian Procrustes analysis, instead align the neural data directly [12, 13, 14], while test-time calibration adapts frozen models using unlabeled target-subject data [15]. In most of these approaches, however, the visual side is represented by a single global image embedding shared by all subjects.

We argue that this visual target is itself a bottleneck for crosssubject generalization. A global embedding compresses an image into one vector, so every subject’s EEG must be mapped onto the same summary, even though different individuals’ responses may reflect different aspects of the image. Modern vision encoders retain far richer information: the Perception Encoder (PE) shows that intermediate spatial representations carry distributed, complementary visual information not confined to the final output [16]. We hypothesize that supervising EEG with several complementary image views provides the encoder with multiple ways to match a new subject’s response, thereby enabling better transfer across individuals.

We therefore introduce a structured multi-view visual target constructed from PE spatial features. Learned queries [17, 18] pool the spatial feature map into multiple complementary views. These views are then grouped and aggregated by a block-structured, content-dependent router [19, 20], so that different subsets of spatial information contribute depending on the image. The visual target is learned jointly with the EEG encoder through EEG–vision contrastive learning and is combined with a two-stage schedule that uses MMD regularization [21] to reduce distributional differences among source subjects. For unseen subjects, we further introduce a post hoc representation refinement that requires no target-subject labels or additional training. It aligns the frozen EEG and image embeddings through moment matching, CSLS-based pseudo-correspondences [22], and regularized orthogonal alignment. This step targets the residual distribution shift that remains after training, which trainingtime regularization cannot fully remove because target-subject data are unavailable during training. Since the refinement uses the unlabeled test embeddings of the held-out subject collectively (i.e., it is transductive), we report it separately.

Our contributions are:

• a structured multi-view visual target that converts spatially distributed PE features into complementary supervision views for cross-subject EEG decoding;

• a content-dependent block routing strategy that adaptively ag-

![](images/1fbeee16759ef40e42b636347557eec64f81cf21e441568647c20ee5f4de1c96.jpg)  
Fig. 1. Overview of the proposed cross-subject EEG-to-image retrieval framework. The trainable EEG encoder maps EEG signals into the shared embedding space, while a frozen pre-trained vision encoder produces spatial visual tokens that are converted into multiple visual views using learned multi-view pooling and fused by the block attention–residual router. EEG and image embeddings are aligned through a two-stage objective, combining contrastive learning with MMD-based distribution alignment in Stage I and contrastive learning alone in Stage II.

gregates these views for each image;

• a training-free representation refinement for unseen subjects. In ten-subject LOSO experiments on THINGS-EEG2, the structured target achieves the highest average retrieval accuracy among the methods reported in Table 1, while the separate representation-refinement stage further raises Top-1 accuracy to 48.1%.

## 2. METHOD

## 2.1. Problem Setup

We address cross-subject EEG-to-image retrieval under a leave-onesubject-out (LOSO) protocol. Let

$$
\mathcal { D } = \{ ( x _ { i } , I _ { i } , s _ { i } ) \} _ { i = 1 } ^ { N } ,\tag{1}
$$

where $\boldsymbol { x } _ { i } ~ \in ~ \mathbb { R } ^ { C \times T }$ denotes an EEG trial, $I _ { i }$ is the corresponding image, and $s _ { i }$ denotes the subject identity. We learn an EEG encoder $f _ { \theta }$ that maps neural responses into a shared embedding space with a frozen visual encoder. During training, nine subjects are used as source subjects, while the held-out subject is completely unseen. At inference, the learned EEG encoder is applied without target-subject fine-tuning.

Rather than using a single global image descriptor for supervision, we construct a structured multi-view visual target from the spatial representation produced by the frozen visual encoder. The detailed architecture is depicted in Fig. 1.

## 2.2. Structured Multi-View Visual Target

Let the spatial patch representation of an image be denoted by $P \in$ $\mathbb { R } ^ { T \times D }$ .To preserve complementary spatial information, we introduce K learnable queries $Q \in \mathbb { R } ^ { \check { K } \times \check { D } }$ and extract multiple visual views through cross-attention:

$$
V = \mathrm { M H A } \left( Q , \mathrm { L N } ( P ) , \mathrm { L N } ( P ) \right) , \qquad V \in \mathbb { R } ^ { K \times D } .\tag{2}
$$

where P denotes the spatial patch features, Q the learnable queries, LN Layer Normalization, and MHA multi-head cross-attention. Each query learns a distinct attention pattern over the spatial tokens, producing multiple task-specific views instead of collapsing the entire feature map into a single global vector. The extracted views are subsequently projected into the common EEG–vision embedding space:

$$
z _ { k } = f _ { k } ( V _ { k } ) , \qquad k = 1 , \ldots , K .\tag{3}
$$

We then adapt block attention-residual routing to aggregate these visual views into a structured supervision target. This adaptation is designed specifically to construct visual supervision for cross-subject EEG representation learning, rather than to combine transformer-layer residuals.

## 2.3. Block Attention Residual View Routing

The learned visual views are hierarchically aggregated through block attention-residual routing to form a structured visual supervision target for cross-subject EEG–vision learning [19]. The K views are divided into B equal-sized blocks. Within each block, a learnable query $q _ { v }$ computes content-dependent attention weights:

$$
a _ { b , k } = \mathrm { s o f t m a x } _ { k \in \mathcal { B } _ { b } } \left( \frac { q _ { v } ^ { \top } \mathrm { L N } ( z _ { k } ) } { \tau } \right) ,\tag{4}
$$

where $B _ { b }$ denotes the set of views in the b-th block. The block representation is

$$
h _ { b } = \sum _ { k \in \mathcal { B } _ { b } } a _ { b , k } z _ { k } .\tag{5}
$$

A second learnable query q<sub>B</sub> assigns content-dependent weights to the block representations:

$$
w _ { b } = \mathrm { s o f t m a x } _ { b } \left( { \frac { q _ { B } ^ { \top } \mathrm { L N } ( h _ { b } ) } { \tau } } \right) .\tag{6}
$$

The final view weight is

$$
\omega _ { b , k } = w _ { b } a _ { b , k } .\tag{7}
$$

The structured visual target is then obtained as

$$
z _ { I } = g \left( \sum _ { b = 1 } ^ { B } \sum _ { k \in \mathcal { B } _ { b } } \omega _ { b , k } z _ { k } \right) .\tag{8}
$$

where $g ( \cdot )$ denotes the final projection into the shared EEG– vision embedding space.

Table 1. Cross-subject zero-shot EEG-to-image retrieval on THINGS-EEG2 under leave-one-subject-out evaluation. Results are reported as Top-1 / Top-5 accuracy (%).
<table><tr><td>Method</td><td>Sub-01</td><td>Sub-02</td><td>Sub-03</td><td>Sub-04</td><td>Sub-05</td><td>Sub-06</td><td>Sub-07</td><td>Sub-08</td><td>Sub-09</td><td>Sub-10</td><td>Avg.</td></tr><tr><td>NICE [3]</td><td>7.6/22.8</td><td>5.9/20.5</td><td>6.0/22.3</td><td>6.3/20.7</td><td>4.4/18.3</td><td>5.6/22.2</td><td>5.6/19.7</td><td>6.3/22.0</td><td>5.7/17.6</td><td>8.4/28.3</td><td>6.2/21.4</td></tr><tr><td>ATM [2]</td><td>10.5/26.8</td><td>7.1/24.8</td><td>11.9/33.8</td><td>14.7/39.4</td><td>7.0/23.9</td><td>11.1/35.8</td><td>16.1/43.5</td><td>15.0/40.3</td><td>4.9/22.7</td><td>20.5/46.5</td><td>11.9/33.8</td></tr><tr><td>UBP [9]</td><td>11.5/29.7</td><td>15.5/40.0</td><td>9.8/27.0</td><td>13.0/32.3</td><td>8.8/33.8</td><td>11.7/31.0</td><td>10.2/23.8</td><td>12.2/32.2</td><td>15.5/40.5</td><td>16.0/43.5</td><td>12.4/33.4</td></tr><tr><td>NeuroBridge [10]</td><td>23.2/52.4</td><td>21.2/49.3</td><td>13.2/36.5</td><td>17.0/45.3</td><td>14.5/37.7</td><td>25.0/55.0</td><td>15.3/45.1</td><td>20.1/44.9</td><td>13.7/36.5</td><td>27.2/56.3</td><td>19.0/45.9</td></tr><tr><td>Shallow Alignment [11]</td><td>24.6/54.7</td><td>31.3/61.5</td><td>11.4/31.1</td><td>19.9/48.8</td><td>19.0/45.5</td><td>24.1/49.8</td><td>18.6/51.6</td><td>17.6/46.7</td><td>23.3/54.9</td><td>34.6/63.2</td><td>22.4/50.8</td></tr><tr><td>SAMGA*[4]</td><td>36.0/58.5</td><td>37.5/68.5</td><td>21.0/43.0</td><td>29.0/56.0</td><td>21.0/51.0</td><td>32.0/62.0</td><td>25.5/55.5</td><td>26.0/50.0</td><td>24.0/51.0</td><td>43.5/72.5</td><td>29.6/56.8</td></tr><tr><td>Ours</td><td>41.5/72.0</td><td>38.5/73.0</td><td>24.5/51.0</td><td>39.0/71.5</td><td>32.5/61.0</td><td>35.5/64.5</td><td>33.5/69.0</td><td>32.5/58.0</td><td>27.5/55.5</td><td>48.0/80.5</td><td>35.3/65.6</td></tr><tr><td>Ours†</td><td>56.0/84.0</td><td>58.0/84.0</td><td>41.5/72.5</td><td>44.0/75.5</td><td>44.5/74.5</td><td>49.5/76.0</td><td>48.0/78.0</td><td>37.0/63.0</td><td>43.5/76.0</td><td>59.0/87.5</td><td>48.1/77.1</td></tr></table>

<sup>∗</sup>Reproduced using the official implementation under the same experimental settings as the proposed method. <sup>†</sup>Transductive: uses unlabeled target-subject embeddings without labels or encoder updates for representation refinement.

In our implementation, $K \ = \ 1 2$ visual views are organized into B = 4 blocks. This two-level aggregation selectively integrates complementary visual views while avoiding a single flat softmax over all views. The key contribution is the adaptation of block attention-residual routing to structured visual target construction for cross-subject EEG–vision representation learning.

## 2.4. Training Objective

We optimize the EEG and visual projection components using a temperature-scaled contrastive EEG–image objective. To improve cross-subject consistency, training uses a two-stage optimization schedule. In Stage I, the contrastive objective is jointly optimized with an MMD-based regularization term that reduces distributional differences among source subjects. The MMD contribution is progressively reduced during this stage. In Stage II, the shared projection layer is frozen and the optimization focuses on EEG–vision contrastive alignment with a reduced learning rate.

This training strategy encourages subject-invariant EEG representations while preserving the semantic alignment required for image retrieval. For the representation-refinement setting, the trained encoders are kept frozen and refinement is performed post hoc on the held-out subject using only the unordered EEG and image embeddings, without target-subject labels or additional model training. The procedure first performs per-dimension moment matching, followed by pseudo-correspondence estimation using CSLS and mutual nearest-neighbor landmarks [22]. We use 64 landmarks and confidence-based weighting for the estimated correspondences, followed by regularized orthogonal alignment with a regularization coefficient $\rho = 0 . 1$ . The resulting aligned representations are finally evaluated using CSLS-based retrieval. This post hoc procedure is referred to as representation refinement and is evaluated separately from the trained model’s direct output.

## 3. EXPERIMENTS

## 3.1. Dataset Details

We evaluate on the THINGS-EEG2 dataset [1], which contains EEG recordings from 10 subjects collected using a rapid serial visual presentation (RSVP) paradigm. The training set comprises 1,654 object concepts, with 10 images per concept and 4 repetitions per image, resulting in 16,540 images and 66,160 EEG trials per subject. The test set contains 200 previously unseen concepts, with one image per concept repeated 80 times, forming a shared 200-image retrieval gallery. EEG recordings consist of 63 channels and are downsampled from 1,000 Hz to 250 Hz for modeling. We follow a strict tenfold leave-one-subject-out (LOSO) protocol, where nine subjects are used for training and the remaining subject is held out entirely for testing. We report 200-way zero-shot Top-1 and Top-5 retrieval accuracy.

## 3.2. Evaluation Details

For each EEG trial, the learned EEG encoder produces an embedding

$$
z _ { E } = f _ { s } \left( f _ { p } \left( f _ { \theta } ( x _ { t } ) \right) \right) ,\tag{9}
$$

where $f _ { \theta }$ denotes the EEG encoder and $f _ { p } , f _ { s }$ denote the projection modules. Each candidate image is represented by its corresponding structured visual target, forming the retrieval gallery

$$
\mathcal { G } = \left\{ z _ { I } ^ { ( 1 ) } , z _ { I } ^ { ( 2 ) } , \dots , z _ { I } ^ { ( M ) } \right\} .\tag{10}
$$

Retrieval is performed by ranking the gallery according to cosine similarity,

$$
S ( z _ { E } , z _ { I } ) = \frac { z _ { E } ^ { \top } z _ { I } } { \| z _ { E } \| _ { 2 } \| z _ { I } \| _ { 2 } } .\tag{11}
$$

The highest-scoring image determines Top-1 retrieval, while the five highest-scoring images determine Top-5 retrieval.

## 3.3. Implementation Details

All experiments are conducted on a Linux-based system equipped with three NVIDIA RTX A6000 GPUs, each providing 48 GB of GPU memory. The EEG signal is encoded using a TSConvbased neural encoder followed by a linear projection into a 512- dimensional shared EEG–vision embedding space [3]. The encoder operates on a 0–250 ms time window, with smoothing-based EEG augmentation enabled during training, while the visual encoder (Perception Encoder PE-Core-G14 [16]) is kept frozen. We extract K = 12 visual views from the spatial visual representation and organize them into B = 4 blocks for hierarchical block attention-residual routing. The router uses a temperature of 0.7 and a subject-bias dropout rate of 0.3, with EEG-conditioned routing disabled.

Training uses a batch size of 1024 and an initial learning rate of $1 \times 1 0 ^ { - 4 }$ for up to 50 epochs. The contrastive temperature is fixed at $0 . 0 7 \cdot$ , with $\alpha = \beta = 1 . 0$ for the alignment objectives. Image features are $\ell _ { 2 }$ -normalized before alignment. Following the two-stage optimization described in Section 2.4, Stage I runs for 20 epochs and jointly optimizes the contrastive and distribution-alignment objectives, with the distribution-alignment weight reduced from 0.9 to

![](images/7001dbf5436e3a2af723f9d9ab9589abd2e9d1cab52402121814ce17b4588d83.jpg)

Fig. 2. Subject-wise Top-1 retrieval accuracy on THINGS-EEG2 under leave-one-subject-out evaluation. Results are reported for each of the ten held-out subjects across all compared methods.  
![](images/df7f1aa3337fef286a7a0b3358062dad03d087a892bf0e3117faf18f866d6bde.jpg)  
Fig. 3. Average cross-subject Top-1 retrieval accuracy on THINGS-EEG2 under leave-one-subject-out evaluation. Results are averaged across the ten held-out subjects for all compared methods.

0.5. In Stage II, the shared projection component is frozen and contrastive training continues with a reduced learning rate of $5 \times 1 0 ^ { - 5 }$ Early stopping is applied with respect to the validation set with a patience of 10 epochs. The EEG feature dimension before projection is 1024, and the intermediate visual feature dimension is 1024.

For cross-subject evaluation, we follow a leave-one-subject-out protocol on THINGS-EEG2, using nine subjects for source training and the remaining subject for evaluation. All experiments use a fixed random seed of 2025. The frozen visual representation uses the specified spatial feature representation with visual layers 20, 24, 28, 32, and 36 as the source features for multi-view target construction.

## 3.4. Results and Discussion

Comparison with prior methods. Table 1 reports LOSO retrieval accuracy for all ten held-out subjects. Without any target-subject adaptation, the proposed structured visual target achieves 35.3% Top-1 and 65.6% Top-5 accuracy on average, the highest among the compared methods. For a controlled comparison with the strongest baseline, we reproduced SAMGA [4] using its official implementation under our experimental settings (SAMGA<sup>∗</sup>). Under identical settings, our model improves the average accuracy by 5.7 points in Top-1 and 8.8 points in Top-5, and outperforms SAMGA on all ten held-out subjects for both metrics. Fig. 2 shows this comparison per subject. All methods follow a similar difficulty profile: Sub-10 is the easiest held-out subject for every method, and Sub-03 is among the hardest for the stronger methods, reflecting subject-dependent signal quality. Nevertheless, our model stays above SAMGA for every subject, and the refined model is the best for every subject. As summarized in Fig. 3, the average Top-1 accuracy rises from 29.6% for SAMGA<sup>∗</sup> to 35.3% with the structured target and to 48.1% with refinement. This suggests that supervising EEG with multiple learned visual views, rather than a single global image embedding, yields representations that transfer better to unseen subjects.

![](images/0fdbe55730e754b715aca9716ad9858fd87fa8e9919b6c118da49353ec360284.jpg)  
Fig. 4. Subject-wise effect of representation refinement on crosssubject EEG-to-image retrieval. (a) Top-1 and Top-5 retrieval accuracy before and after refinement for each held-out subject. (b) Subject-wise Top-1 and Top-5 accuracy gains, reported in percentage points (pp).

Representation refinement. Applying the label-free refinement to the frozen embeddings raises the average accuracy from 35.3% to 48.1% Top-1 and from 65.6% to 77.1% Top-5. Because the refinement uses the unlabeled test embeddings of the held-out subject collectively, we report it separately from the direct model output. As shown in Fig. 4, refinement improves both Top-1 and Top-5 accuracy for every held-out subject, with Top-1 gains ranging from 4.5 percentage points (pp) for Subject 8 to 19.5 pp for Subject 2 (mean 12.8 pp). The gain is only weakly related to a subject’s accuracy before refinement (Pearson r = −0.23 across subjects), suggesting that the benefit depends on the subject-specific structure of the embeddings rather than on how poorly the subject is initially decoded.

Discussion. The two components act at different stages. The structured target shapes the EEG representation during source training, whereas refinement aligns the resulting embeddings to an unseen subject at deployment without labels or encoder updates. The large refinement gains indicate that, for new subjects, the source-trained embeddings retain stimulus-related structure that is misaligned rather than lost. A limitation is that refinement currently uses all test trials of the held-out subject; its behavior with smaller target batches remains to be characterized.

## 4. CONCLUSION

We presented a structured visual target learning framework for crosssubject EEG-to-image retrieval. The proposed approach preserves spatial information from the frozen Perception Encoder, converts it into multiple learned visual views, and adaptively aggregates them through block-structured, content-dependent routing. MMD-based regularization further promotes cross-subject consistency during training. We also introduced a training-free representation refinement stage that aligns frozen EEG and visual embeddings for an unseen subject without target-subject labels or encoder updates. Future work will explore refinement with limited target-subject data and alternative visual backbones.

## 5. REFERENCES

[1] Alessandro T Gifford, Kshitij Dwivedi, Gemma Roig, and Radoslaw M Cichy, “A large and rich eeg dataset for modeling human visual object recognition,” NeuroImage, vol. 264, pp. 119754, 2022.

[2] Dongyang Li, Chen Wei, Shiying Li, Jiachen Zou, Haoyang Qin, and Quanying Liu, “Visual decoding and reconstruction via eeg embeddings with guided diffusion,” arXiv preprint arXiv:2403.07721, 2024.

[3] Yonghao Song, Bingchuan Liu, Xiang Li, Nanlin Shi, Yijun Wang, and Xiaorong Gao, “Decoding natural images from eeg for object recognition,” in International conference on learning representations, 2024, vol. 2024, pp. 47648–47665.

[4] Lin Jiang, Qingshan She, Jiale Xu, Haiqi Xu, Duanpo Wu, and Zhenzhong Kuang, “Subject-aware multi-granularity alignment for zero-shot eeg-to-image retrieval,” arXiv preprint arXiv:2604.17782, 2026.

[5] Jiyuan Wang, Li Zhang, Haipeng Lin, Qile Liu, Gan Huang, Ziyu Li, Zhen Liang, and Xia Wu, “Neuroclip: Brain-inspired prompt tuning for eeg-to-image multimodal contrastive learning,” arXiv preprint arXiv:2511.09250, 2025.

[6] Taida Li, Yujun Yan, Fei Dou, Wenzhan Song, and Xiang Zhang, “Cross-subject generalization for eeg decoding: a survey of deep learning methods,” Progress in Biomedical Engineering, vol. 8, no. 2, pp. 022013, 2026.

[7] Simanto Saha and Mathias Baumert, “Intra-and inter-subject variability in eeg-based sensorimotor brain computer interface: a review,” Frontiers in computational neuroscience, vol. 13, pp. 87, 2020.

[8] Jingyi Tang, Shuai Jiang, Fei Su, and Zhicheng Zhao, “Aligning what eeg can see: Structural representations for brainvision matching,” arXiv preprint arXiv:2603.07077, 2026.

[9] Haitao Wu, Qing Li, Changqing Zhang, Zhen He, and Xiaomin Ying, “Bridging the vision-brain gap with an uncertaintyaware blur prior,” in 2025 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2025, pp. 2246–2257.

[10] Wenjiang Zhang, Sifeng Wang, Yuwei Su, Xinyu Li, Chen Zhang, and Suyu Zhong, “Neurobridge: Bio-inspired selfsupervised eeg-to-image decoding via cognitive priors and bidirectional semantic alignment,” in Proceedings of the AAAI Conference on Artificial Intelligence, 2026, vol. 40, pp. 18028– 18036.

[11] Yang Du, Siyuan Dai, Yonghao Song, Paul M Thompson, Haoteng Tang, and Liang Zhan, “Deep models, shallow alignment: Uncovering the granularity mismatch in neural decoding,” arXiv preprint arXiv:2601.21948, 2026.

[12] He He and Dongrui Wu, “Transfer learning for brain–computer interfaces: A euclidean space data alignment approach,” IEEE Transactions on Biomedical Engineering, vol. 67, no. 2, pp. 399–410, 2019.

[13] James V Haxby, J Swaroop Guntupalli, Andrew C Connolly, Yaroslav O Halchenko, Bryan R Conroy, M Ida Gobbini, Michael Hanke, and Peter J Ramadge, “A common, highdimensional model of the representational space in human ventral temporal cortex,” Neuron, vol. 72, no. 2, pp. 404–416, 2011.

[14] Pedro Luiz Coelho Rodrigues, Christian Jutten, and Marco Congedo, “Riemannian procrustes analysis: transfer learning for brain–computer interfaces,” IEEE Transactions on Biomedical Engineering, vol. 66, no. 8, pp. 2390–2401, 2018.

[15] Qunjie Huang and Weina Zhu, “Sattc: Structure-aware labelfree test-time calibration for cross-subject eeg-to-image retrieval,” arXiv preprint arXiv:2603.20738, 2026.

[16] Daniel Bolya, Po-Yao Huang, Peize Sun, Jang Hyun Cho, Andrea Madotto, Chen Wei, Tengyu Ma, Jiale Zhi, Jathushan Rajasegaran, Hanoona Bangalath, et al., “Perception encoder: The best visual embeddings are not at the output of the network,” Advances in Neural Information Processing Systems, vol. 38, pp. 60884–60937, 2026.

[17] Junnan Li, Dongxu Li, Silvio Savarese, and Steven Hoi, “BLIP-2: Bootstrapping language-image pre-training with frozen image encoders and large language models,” in Proceedings of the 40th International Conference on Machine Learning (ICML), 2023, vol. 202 of Proceedings of Machine Learning Research, pp. 19730–19742.

[18] Andrew Jaegle, Felix Gimeno, Andrew Brock, Andrew Zisserman, Oriol Vinyals, and Joao Carreira, “Perceiver: General˜ perception with iterative attention,” in Proceedings ofthe 38th International Conference on Machine Learning (ICML), 2021, vol. 139 of Proceedings of Machine Learning Research, pp. 4651–4664.

[19] Kimi Team, Guangyu Chen, Yu Zhang, Jianlin Su, Weixin Xu, Siyuan Pan, Yaoyu Wang, Yucheng Wang, Guanduo Chen, Bohong Yin, et al., “Attention residuals,” arXiv preprint arXiv:2603.15031, 2026.

[20] Noam Shazeer, Azalia Mirhoseini, Krzysztof Maziarz, Andy Davis, Quoc Le, Geoffrey Hinton, and Jeff Dean, “Outrageously large neural networks: The sparsely-gated mixture-ofexperts layer,” in International Conference on Learning Representations (ICLR), 2017.

[21] Arthur Gretton, Karsten M. Borgwardt, Malte J. Rasch, Bernhard Scholkopf, and Alexander Smola, “A kernel two-sample¨ test,” Journal ofMachine Learning Research, vol. 13, no. 25, pp. 723–773, 2012.

[22] Guillaume Lample, Alexis Conneau, Marc’Aurelio Ranzato, Ludovic Denoyer, and Herve J´ egou, “Word translation without´ parallel data,” in International conference on learning representations, 2018.
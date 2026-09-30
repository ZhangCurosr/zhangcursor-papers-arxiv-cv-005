# SFE-VGGT: Source-Free VGGT Distillation for Event-Based Monocular Depth Estimation

Thai Duy Nguyen<sup>1</sup> and Addison Lin Wang<sup>1,∗</sup>

![](images/a17fc3e49bafa10698bbe4207a60678c26b48e9023a177e761e8460de5dba54d.jpg)

![](images/5131652ef2139a2f8b43eac4307bc6b0bac0cc81f7ac7057d2ae1d1555b6d531.jpg)  
Fig. 1. Our method (SFE-VGGT) distills the geometric priors of VGGT to the event domain in a source-free manner. Despite requiring no paired RGB observations, our method achieves performance comparable to EventVGGT, which is trained with paired RGB-event data. Under challenging nighttime conditions on MVSEC, our method achieves lower 10 m depth error, demonstrating that effective transfer of VGGT’s geometric priors to the event domain is possible without paired RGB supervision. (Left: Qualitative results on EventScape dataset. Right: Quantiative results on real-world MVSEC dataset.)

Abstract— Recent event-based depth estimation methods successfully transfer geometric priors from vision foundation models via cross-modal distillation. However, their reliance on synchronized RGB-event pairs or depth annotations during training severely restricts practical deployment. To overcome this bottleneck, we propose SFE-VGGT, a novel source-free framework that distills the geometric priors of VGGT to the event domain without any paired RGB observations. Our core idea is to reconstruct surrogate frames directly from the target event stream to act as a frozen geometric teacher, entirely eliminating the need for genuine source RGB data. Crucially, as these surrogate frames inherently yield imperfect and spatially varying supervision, directly distilling from them propagates artifacts. To resolve this, we introduce a novel reliability-aware distillation strategy. This includes Density-Aware Feature Distillation to emphasize informative event regions, and Confidence-Weighted Depth Distillation to dynamically regulate supervision based on relative teacher-student prediction confidence. Meanwhile, we propose a Cross-Frame Relational Consistency loss that enforces temporal geometric stability using reliable interframe correspondences, bypassing the need for temporallyconsistent teacher’s depth. Extensive experiments demonstrate that, despite source-free, our SFE-VGGT closely matches the accuracy of RGB-dependent baselines under standard conditions, and significantly surpasses them in challenging nighttime scenarios. Across MVSEC nighttime sequences, SFE-VGGT reduces the average 10 m depth error by 15.3% compared with EventVGGT. Moreover, our method exhibits robust zero-shot generalization across real-world datasets, proving that highly effective geometric priors can be transferred to event cameras using strictly source-free supervision.

## I. INTRODUCTION

Event cameras asynchronously measure changes in scene brightness, providing high temporal resolution, high dynamic range, and low latency [1]. These properties make them particularly attractive for geometric perception in scenarios involving rapid motion or challenging illumination, where conventional frame-based cameras typically suffer from motion blur, saturation, or insufficient exposure. Among eventbased perception tasks, monocular depth estimation is critical for applications like autonomous navigation and robotic perception. However, learning dense depth directly from events remains difficult because event streams are sparse and fundamentally different from images. Such sparsity and spatial variation are also encountered in prior event-based depth methods [2]–[4]. Moreover, large-scale datasets with dense metric depth annotations are expensive and scarce, constraining early supervised approaches [5] and eventimage fusion methods [2], [6], [7] to task-specific datasets.

Recent vision foundation models provide a compelling alternative to direct supervision. Large-scale image-based models encapsulate strong semantic and geometric priors that can be transferred to event representations through cross-modal knowledge distillation. Methods such as Depth AnyEvent [3] and EventDAM [4] successfully exploit imagebased depth foundation models to supervise event-based students without requiring metric depth annotations. More recently, EventVGGT [8] demonstrated that the powerful multi-view geometric priors of VGGT [9] can be transferred to temporally ordered event sequences, substantially improving both depth accuracy and temporal consistency.

Crucially, however, eliminating the need for depth annotations does not eliminate the dependence on the original image modality. Existing distillation-based approaches strictly assume that synchronized or paired RGB observations are available alongside the event stream during training [3], [4], [8]. This assumption severely restricts practical deployment, as it necessitates a multi-modal acquisition setup with calibrated and temporally aligned cameras. Consequently, training is confined to datasets where the source image modality was recorded. This critical limitation raises a fundamental research question: can the geometric priors of an imagebasedfoundation model like VGGT be distilled into an eventbased model using strictly source-free supervision, without any access to the original RGB observations?

To this end, we propose a novel source-free distillation framework, called SFE-VGGT. Our core idea is to reconstruct surrogate frames directly from the target event stream to act as a frozen geometric teacher, entirely eliminating the need for genuine source RGB data. Specifically, we utilize a frozen recurrent event-to-video reconstruction network [10] to convert events into surrogate frames. These frames are subsequently processed by a frozen VGGT [9], exposing its pretrained geometric priors purely from event-derived data. Concurrently, a student initialized from the same VGGT backbone processes frame-like event representations, adapting via lightweight trainable parameters. At inference time, only the student model is retained.

However, this source-free formulation introduces a critical technical challenge: event reconstruction is inherently imperfect, yielding spatially varying and sometimes unreliable supervision. Directly forcing a student network to reproduce all teacher features and depths from these reconstructed frames inevitably propagates artifacts. To resolve this, we introduce a reliability-aware distillation strategy comprising three main technical contributions. First, Density-Aware Feature Distillation (DAFD) weights feature alignment according to patchlevel event activity, ensuring that regions lacking sufficient event evidence do not dominate the transfer process. Second, Confidence-Weighted Depth Distillation (CWDD) performs scale-invariant depth transfer while dynamically regulating pixel-wise supervision based on the relative confidence between the teacher and student. Finally, we formulate a Cross-Frame Relational Consistency (CFRC) loss that leverages reliable inter-frame correspondences from the reconstructed frames to preserve the student’s relative depth ordering across time, bypassing the need to directly distill temporal changes from potentially noisy teacher depth maps.

Extensive experiments on the EventScape [11], MVSEC [12], and DENSE [13] datasets demonstrate the effectiveness of this approach. Our findings show that the proposed framework successfully retains the formidable geometric capabilities of VGGT despite operating in a restrictive, source-free setting. Under standard evaluation conditions, SFE-VGGT closely matches the performance of the RGBdependent EventVGGT baseline. More notably, as depicted in Fig. 1, our source-free approach significantly outperforms EventVGGT on challenging nighttime MVSEC sequences, where the inherent advantages of event sensing are most pronounced. In particular, SFE-VGGT reduces the average 10 m and 20 m depth errors by 15.3% and 4.0%, respectively. Moreover, the model exhibits robust zero-shot generalization from synthetic EventScape data to unseen real-world sequences, decisively proving that access to the original RGB modality is not a strict prerequisite for transferring highly effective geometric priors to event cameras.

In summary, our main contributions are four-fold: (I) We introduce SFE-VGGT, a novel source-free distillation framework for event-based monocular depth estimation that eliminates the need for paired RGB data or metric depth labels. (II) We propose a reliability-aware distillation strategy (including DAFD and CWDD) to dynamically suppress uncertain regions arising from imperfect surrogate frames. (III) We propose a CFRC loss that enforces temporal depth stability using reliable reconstructed-frame correspondences, without relying on absolute VGGT depth maps. (IV) Our source-free method achieves on-par performance with the RGB-dependent methods and significantly better performance in nighttime scenarios.

## II. RELATED WORKS

a) Vision foundation models: Large-scale pretraining has produced highly transferable visual representations through image-text supervision, self-supervised learning, and masked modeling [14]–[17]. Foundation models have subsequently demonstrated strong generalization for segmentation and monocular geometry [18], [19], and their priors are increasingly being adapted to event-based perception [3], [4], [20], [21]. Among geometric foundation models, VGGT [9] jointly reasons about depth, cameras, point maps, and track across multiple views. EventVGGT [8] shows that these multi-view priors can be transferred to event sequences when synchronized RGB data are available. Our work considers the complementary source-free setting, where the teacher observation is reconstructed solely from events and used for reliability-aware geometric distillation.

b) Event-based monocular depth estimation: Event cameras provide high temporal resolution and dynamic range, making them attractive for geometric perception under fast motion and challenging illumination [1]. Early event-only methods estimate dense depth using recurrent or transformer-based architectures [5], [22]–[24], while another line of work combines events with conventional frames to exploit complementary temporal and photometric information [2], [6], [7], [25], [26]. Despite their strong performance, supervised approaches are limited by the scarcity of dense event-depth annotations. Recent methods therefore transfer knowledge from pretrained image models [3], [4], [8]. Depth AnyEvent [3] and EventDAM [4] distill image-domain depth priors into event representations, while EventVGGT [8] further transfers the geometric priors of VGGT to temporally ordered event sequences. However, these approaches still rely on paired or synchronized RGB observations during distillation. Our method instead derives the teacher view directly from the event stream.

![](images/b6f84d32593daf181dd62b3893c58b37c0f46934c6ffe2c8ca7ddcd43c8908ac.jpg)  
Fig. 2. Overview of the proposed SFE-VGGT framework. During training, the event data is processed by a pretrained E2VID model to construct surrogate frames for the frozen VGGT teacher, while the student directly processes event representations. Knowledge is transferred through (a) Density Aware Feature Distillation (DAFD), which weights patch-level feature alignment by event density, (b) Confidence-Weighted Depth Distillation (CWDD) which weights depth supervision using relative prediction confidence, and (c) Cross-Frame Relational Consistency (CFRC), which enforces consistent depth relations across matched points between frames. During inference, only the event-based student is retained.

c) Source-free knowledge distillation: Knowledge distillation transfers predictions or representations from a pretrained teacher to a student [27], [28], and has been extended across diverse modalities for representation learning, temporal understanding, and 3D perception [29]–[31]. Sourcefree distillation further removes access to the original taskrelevant source data [32]–[34], with recent works extending this setting to cross-modal and event-based transfer under limited or unavailable source observations [35]–[37]. These approaches aim to preserve transferable knowledge despite the absence of the original source modality, while addressing the distribution and representation gaps between heterogeneous sensing domains. However, unlike prior works that primarily focus on recognition tasks, our method targets dense geometric prediction by distilling the geometric priors of a frozen VGGT teacher to an event-based depth student.

## III. METHODOLOGY

## A. Overview and Problem Formulation

Framework overview. We propose a source-free framework for event-based depth estimation that transfers geometric priors from a pretrained VGGT model [9] to the event domain without paired RGB observations. As depicted in Fig. 2, our method reconstructs intensity frames from the event stream using a pretrained E2VID model [10] and feeds them to the teacher branch, while the student directly processes event representations. Both branches are initialized from the same pretrained VGGT checkpoint, and no paired RGB data are required. Both branches share the same pretrained VGGT initialization, and training requires no paired RGB data.

Since the teacher operates on reconstructed frames, its supervision can be affected by reconstruction artifacts and spatial uncertainty. Directly enforcing uniform agreement may propagate unreliable teacher signals to the student. To make the transfer more selective, we use three complementary objectives: Density-Aware Feature Distillation (Sec. III-B), Confidence-Weighted Depth Distillation (Sec. III-C), and Cross-Frame Relational Consistency (Sec. III-D).

Event-based Monocular Depth Estimation. Event cameras encode visual changes as a stream of asynchronous events. Each event is represented as $\boldsymbol { e } _ { i } ~ = ~ \left( x _ { i } , y _ { i } , t _ { i } , p _ { i } \right)$ , where $( x _ { i } , y _ { i } )$ denotes the pixel location, $t _ { i }$ is the timestamp, and $p _ { i } \in \{ - 1 , + 1 \}$ indicates the polarity. To enable processing by the frame-based VGGT architecture, we partition the continuous event stream into S consecutive temporal windows and accumulate them into a sequence of event representations $\mathbf { I } ^ { \mathrm { e v t } } = \{ I _ { s } ^ { \mathrm { e v t } } \} _ { s = 1 } ^ { S }$ , where $I _ { s } ^ { \mathrm { e v t } } \in \mathbb { R } ^ { 3 \times H \times W }$ . The resulting sequence is used as input to the event-based depth model.

Following VGGT [9], each event frame is partitioned into non-overlapping patches and embedded into patch tokens. The tokens from the S frames are processed by the VGGT backbone, yielding event features $\dot { \boldsymbol { f } } _ { \mathrm { e v t } } \in \mathbb { R } ^ { S \times \mathbf { \bar { K } } \times C }$ , where K and C denote the number of patch tokens and feature dimension, respectively. The prediction head outputs perframe depth prediction $d _ { \mathrm { e v t } }$ and confidence $C _ { \mathrm { e v t } }$

## B. Density-Aware Feature Distillation (DAFD)

Insight: Event observations are inherently sparse and spatially non-uniform, such that different regions provide substantially different amounts of visual evidence. This affects both sides of our distillation framework. For the student, features extracted from event-active regions are better supported by the input than those from weakly observed patches. Likewise, the surrogate frames used by the teacher are also less reliable in regions where insufficient events are available for reconstruction. Therefore, uniformly aligning all feature tokens may overemphasize regions where either branch is weakly supported by the observed events.

To account for this, we use the local event density as a reliability signal for feature distillation. As shown in Fig.2(a), for the i-th patch of size $P \times P$ at timestep s, we define

$$
w _ { s , i } = \frac { N _ { s , i } } { P \times P } ,\tag{1}
$$

where $N _ { s , i }$ denotes the number of spatial locations activated by at least one event. Hence, $w _ { s , i } \in [ 0 , 1 ]$ measures the event support of the corresponding patch token. We directly associate this weight with the feature token extracted from the same spatial patch, such that a denser patch receives stronger feature-level supervision, while a sparse patch contributes less to the distillation objective. Given student and teacher features $f _ { \mathrm { e v t } } , f _ { \mathrm { e 2 v i d } } ~ \in ~ \bar { \mathbb { R } } ^ { S \times K _ { f } \times C }$ , we define the densityaware feature distillation loss as

$$
\mathcal { L } _ { \mathrm { { D A F D } } } = \frac { \sum _ { s , i } w _ { s , i } \left[ 1 - \cos \left( f _ { \mathrm { { e v t } } } ^ { s , i } , \mathrm { { s g } } [ f _ { \mathrm { { e 2 v i d } } } ^ { s , i } ] \right) \right] } { \sum _ { s , i } w _ { s , i } + \epsilon } ,\tag{2}
$$

where $\mathrm { s g } [ \cdot ]$ denotes stop-gradient. The density weighting places greater emphasis on regions where both the event representation and the reconstructed teacher observation are better supported by the events, while reducing the influence of potentially unreliable alignment in sparse regions. Normalizing by the accumulated density prevents variations in event activity from affecting the scale of the objective.

## C. Confidence-Weighted Depth Distillation (CWDD)

Insight: Feature alignment alone does not directly constrain the predicted geometry. We therefore further distill the teacher depth into the event student. However, the teacher prediction is obtained from E2VID frames may inherit errors from uncertain reconstruction regions. Uniform depth supervision could therefore force the student to reproduce unreliable teacher predictions. Since both branches jointly predict depth and confidence, we use their relative confidence to adapt the supervision strength at each pixel.

As shown in Fig.2(b), for each pixel i, we define the confidence weight as

$$
w _ { i } = \frac { \mathrm { s g } [ C _ { \mathrm { e 2 v i d } , i } ] } { \mathrm { s g } [ C _ { \mathrm { e v t } , i } ] + \mathrm { s g } [ C _ { \mathrm { e 2 v i d } , i } ] + \epsilon } , \qquad \tilde { w } _ { i } = \frac { w _ { i } } { \mathrm { m e a n } _ { j } ( w _ { j } ) } ,
$$

where $\mathrm { s g } [ \cdot ]$ denotes stop-gradient. The ratio reflects the teacher confidence relative to that of the student. When $C _ { \mathrm { e 2 v i d } } \gg C _ { \mathrm { e v t } } , w _ { i }  1$ and the teacher provides stronger supervision; when $C _ { \mathrm { e v t } } \gg C _ { \mathrm { e 2 v i d } } , w _ { i }  0$ and the teacher contribution is suppressed. Comparable confidences yield $w _ { i } \approx 0 . 5$ . The bounded ratio prevents high-confidence predictions from dominating the loss, while mean normalization stabilizes its magnitude across samples.

Since monocular depth is ambiguous up to a global scale, we perform distillation in log-depth space using a scaleinvariant objective [38]. Let

$$
r _ { i } = \log d _ { \mathrm { e v t } , i } - { \mathrm { s g } } [ \log d _ { \mathrm { e 2 v i d } , i } ] .\tag{4}
$$

The confidence-weighted scale-invariant term is

$$
\mathcal { L } _ { \mathrm { s i } } = \mathbb { E } _ { i } \left[ \tilde { w } _ { i } r _ { i } ^ { 2 } \right] - \lambda _ { \mathrm { s i } } \left( \mathbb { E } _ { i } [ \tilde { w } _ { i } r _ { i } ] \right) ^ { 2 } .\tag{5}
$$

To additionally preserve local depth structure, we introduce a gradient consistency term,

$$
\mathcal { L } _ { \mathrm { g r a d } } = \frac { 1 } { 2 } \mathbb { E } _ { i } \left[ \tilde { w } _ { i } \left( \left| \nabla _ { x } r _ { i } \right| + \left| \nabla _ { y } r _ { i } \right| \right) \right] .\tag{6}
$$

The final depth distillation objective is

$$
\mathcal { L } _ { \mathrm { C W D D } } = \mathcal { L } _ { \mathrm { s i } } + \gamma \mathcal { L } _ { \mathrm { g r a d } } .\tag{7}
$$

This objective transfers both relative depth values and local geometric structure while reducing the influence of uncertain teacher predictions.

## D. Cross-Frame Relational Consistency (CFRC)

Insight: The preceding objectives supervise feature and depth predictions but do not explicitly constrain temporal relations. We therefore use the reconstructed frames to establish reliable cross-frame correspondences, while deriving relational supervision from the student’s depth predictions.

As depicted in $\mathrm { F i g . 2 ( c ) }$ , for a frame pair $( t , t ^ { \prime } )$ , SIFT features [39] are matched on the reconstructed frames and geometric outliers are removed using RANSAC [40]. Let $( u _ { j } ^ { t } , u _ { j } ^ { t ^ { \prime } } )$ denote a valid correspondence. From these matches, we form pairs of correspondence tracks $( a _ { k } , b _ { k } )$ and use the student’s depth ordering in frame t as a detached reference:

$$
r _ { k } = \mathrm { s i g n } \left( \mathrm { s g } \left[ d _ { \mathrm { e v t } } ^ { t } ( u _ { a _ { k } } ^ { t } ) - d _ { \mathrm { e v t } } ^ { t } ( u _ { b _ { k } } ^ { t } ) \right] \right) .\tag{8}
$$

The corresponding depth difference in frame t<sup>′</sup> is

$$
\Delta _ { k } = d _ { \mathrm { e v t } } ^ { t ^ { \prime } } ( u _ { a _ { k } } ^ { t ^ { \prime } } ) - d _ { \mathrm { e v t } } ^ { t ^ { \prime } } ( u _ { b _ { k } } ^ { t ^ { \prime } } ) .\tag{9}
$$

For each relational pair, we define

$$
\ell _ { k } = { \bf 1 } [ r _ { k } \neq 0 ] \mathrm { s o f t p l u s } ( - r _ { k } \Delta _ { k } ) + { \bf 1 } [ r _ { k } = 0 ] \Delta _ { k } ^ { 2 } .\tag{10}
$$

Let $\tau$ denote the set of frame pairs and $\boldsymbol { \mathcal { K } } _ { t , t ^ { \prime } }$ the relational pairs associated with $( t , t ^ { \prime } )$ . The final objective is

$$
\mathcal { L } _ { \mathrm { C F R C } } = \frac { 1 } { | \mathcal { T } | } \sum _ { ( t , t ^ { \prime } ) \in \mathcal { T } } \frac { 1 } { | \mathscr { K } _ { t , t ^ { \prime } } | } \sum _ { k \in \mathscr { K } _ { t , t ^ { \prime } } } \ell _ { k } .\tag{11}
$$

For $r _ { k } ~ = ~ + 1 ~ \mathrm { o r } ~ - 1$ , the loss encourages the same relative depth ordering to be preserved in frame $t ^ { \prime } ,$ while the zero case preserves a near-equal relation. The loss is first averaged within each frame pair and then across frame pairs, preventing pairs with more matches from dominating the objective. Notably, the teacher provides only correspondence geometry; no teacher depth values are used in this loss.

Total objective. The final training objective combines the three proposed objectives as

$$
\begin{array} { r } { \mathcal { L } = \lambda _ { \mathrm { D A F D } } \mathcal { L } _ { \mathrm { D A F D } } + \lambda _ { \mathrm { C W D D } } \mathcal { L } _ { \mathrm { C W D D } } + \lambda _ { \mathrm { C F R C } } \mathcal { L } _ { \mathrm { C F R C } } , } \end{array}\tag{12}
$$

where $\lambda _ { \mathrm { D A F D } } , \lambda _ { \mathrm { C W D D } }$ , and $\lambda _ { \mathrm { C F R C } }$ balance feature-level transfer, depth-level geometric supervision, and cross-frame relational consistency, respectively. Empirically, we set $\lambda _ { \mathrm { D A F D } } = 1 . 0 , \lambda _ { \mathrm { C W D D } } = 2 . 0 ,$ and $\lambda _ { \mathrm { C F R C } } = 0 . 2 .$

## IV. EXPERIMENTS

## A. Experimental Settings

Datasets. Following the evaluation protocol of prior works [2], [4], [8], we use EventScape [11] as the primary training dataset due to its large scale. Subsequently, its test set is used for in-domain evaluation. For zero-shot evaluation, models trained on EventScape are directly tested on the unseen MVSEC [12] and DENSE [13] datasets without any finetuning or adaptation.

Evaluation metrics. We report mean absolute depth error at mutiple cut-off ranges of 10 m, 20 m, and 30 m. Additionally, percentage metrics $\delta _ { i }$ at $i \in \{ 1 . 2 5 , 1 . 2 5 ^ { 2 } , 1 . 2 5 ^ { 3 } \}$ are reported where applicable.

Implementation details. Following prior works [2], [4], [8], event and E2VID-reconstructed inputs are center-cropped to $( 2 5 2 \times 5 0 4 )$ . For event reconstruction, we use the publicly released pretrained E2VID model [10]. Aligning with the VGGT [9] training protocol, sky regions are masked out to exclude invalid depth values. For comparison with EventVGGT [8], we adapt the frozen VGGT backbone using a similar LoRA configuration $( r = 1 6 , \alpha = 3 2 )$ , resulting in approximately 1.6 million trainable parameters. The model is optimized with AdamW at a learning rate of $1 \times 1 0 ^ { - 5 }$ Training uses 24-frame windows on a single NVIDIA RTX 5090, converging in roughly 23 hours.

## B. Experiment Results

1) Results on EventScape dataset: As shown in Tab. I, SFE-VGGT remains highly competitive across all evaluation ranges despite operating in a source-free setting. At 10 m, it achieves comparable performance to EventVGGT [8], while remaining close to the strongest supervised and distillation-based baselines. The advantage becomes more evident as the evaluation range increases. SFE-VGGT achieves the second-best overall performance at both 20 m and 30 m, outperforming all supervised event-image methods at these two ranges. Compared with EventDAM [4], the errors are reduced by 42.8% and 46.5% at 20 m and 30 m, respectively. It also improves over the strongest supervised event-image baseline (SRFNet [2]) at these ranges by 48.2% and 55.4%, showing that the transferred VGGT priors remain effective for medium- and long-range geometry. Although EventVGGT retains the best overall accuracy, SFE-VGGT achieves competitive depth estimation without requiring paired RGB observations during distillation. This result is particularly noteworthy because SFE-VGGT is the only source-free approach among the compared distillation methods, demonstrating that strong geometric transfer can be retained even when the original RGB modality is unavailable. Qualitative results are shown in Fig. 1.

TABLE I  
MEAN ABSOLUTE DEPTH ERROR ON EVENTSCAPE IN METERS. E DENOTES EVENT-ONLY INPUT AND E+I DENOTES EVENT AND RGB IMAGE INPUT. “DEPTH GT” INDICATES WHETHER METRIC GROUND-TRUTH DEPTH VALUES ARE USED AS TRAINING SUPERVISION. BEST AND SECOND-BEST RESULTS ARE HIGHLIGHTED.
<table><tr><td rowspan="2">Method</td><td colspan="2">Training</td><td rowspan="2">Inference </td><td rowspan="2"> $1 0 \mathrm { m } \downarrow$ </td><td rowspan="2">20m↓</td><td rowspan="2">30m↓</td></tr><tr><td>Source-free</td><td>Depth GT</td></tr><tr><td>E2Depth [5]</td><td></td><td>√</td><td>E</td><td>1.79</td><td>5.35</td><td>8.31</td></tr><tr><td>RAMNet [6]</td><td></td><td>√</td><td>E+I</td><td>0.81</td><td>2.26</td><td>3.58</td></tr><tr><td>HMNet [25]</td><td></td><td>√</td><td>E+I</td><td>0.55</td><td>1.80</td><td>3.27</td></tr><tr><td>ER-F2D [7]</td><td>一</td><td>√</td><td>E+I</td><td>0.67</td><td>1.69</td><td>2.81</td></tr><tr><td>SRFNet [2]</td><td></td><td>√</td><td>E+I</td><td>1.27</td><td>1.68</td><td>2.76</td></tr><tr><td>EventDAM [4]</td><td>×</td><td>×</td><td>E</td><td>0.56</td><td>1.52</td><td>2.30</td></tr><tr><td>EventVGGT [8]</td><td>×</td><td>X</td><td>E</td><td>0.54</td><td>0.79</td><td>1.06</td></tr><tr><td>SFE-VGGT</td><td>√</td><td>X</td><td>E</td><td>0.57</td><td>0.87</td><td>1.23</td></tr></table>

2) Zero-shot results on MVSEC dataset: The evaluation on MVSEC [12] examines whether the geometric priors learned from synthetic EventScape transfer to real-world event streams without finetuning. Despite the substantial domain shift in scene appearance, motion, and event statistics, SFE-VGGT consistently ranks among the leading distillation-based methods across the evaluated sequences and depth ranges, as shown in Tab. II. Its advantage is particularly evident under nighttime conditions. At 10 m, SFE-VGGT consistently outperforms EventVGGT [8] across Night1–Night3 (see Fig. 1). Averaged over these three sequences, it reduces the error by 15.3% at 10 m and 4.0% at 20 m relative to EventVGGT, while also achieving the best overall 20 m result on Night2. At the farther 30 m range, EventVGGT remains stronger, although SFE-VGGT retains second-best performance among the distillation methods on Night2 and Night3. On Day1, SFE-VGGT also ranks second among the distillation-based approaches across all three ranges. Overall, the results demonstrate strong syntheticto-real generalization, with the clearest gains appearing in near- and medium-range nighttime depth estimation. The strong nighttime performance further suggests that sourcefree event-based transfer may be particularly beneficial under challenging illumination, where conventional RGB observations can become degraded. Qualitative results on Day1, Night1, and Night2 are shown in Fig. 3.

3) Zero-shot results on DENSE dataset: To further assess robustness under domain shift, we evaluate the EventScapetrained model on the unseen DENSE dataset [13]. As shown in Tab. III, SFE-VGGT achieves the second-best performance across all three evaluation ranges, outperforming Event-DAM [4] and the supervised RAMNet [6] and SRFNet [2] baselines. Compared with EventDAM, the advantage becomes more pronounced with depth, with 44.6% and 54.6% lower errors at 20 m and 30 m, respectively. This wideningy margin highlights the robustness under a substantial domain shift of the transferred geometry.

RGB  
Event Input  
TABLE II  
MEAN ABSOLUTE DEPTH ERROR ON MVSEC IN METERS.  
E DENOTES EVENT-ONLY INPUT AND E+I DENOTES EVENT AND RGB IMAGE INPUT. “DEPTH GT” INDICATES WHETHER METRIC GROUND-TRUTH DEPTH VALUES ARE USED AS TRAINING SUPERVISION. AMONG DISTILLATION-BASED METHODS, THE BEST AND SECOND-BEST RESULTS ARE HIGHLIGHTED. OVERALL BEST RESULTS ACROSS ALL METHODS ARE SHOWN IN BOLD.
<table><tr><td rowspan="2">Method</td><td colspan="2">Training</td><td>Inference |</td><td colspan="2">Night1</td><td></td><td colspan="3">Night2</td><td colspan="3">Night3</td><td colspan="3">Day1</td></tr><tr><td>Source-free</td><td>Depth GT</td><td>Input</td><td>10m ↓</td><td>20m ↓</td><td>30m↓|</td><td>10m↓</td><td>20m↓</td><td> $3 0 \mathrm { m } \downarrow |$ </td><td>10m↓</td><td>20m↓</td><td>30m↓|</td><td>10m↓</td><td>20m↓</td><td>30m↓</td></tr><tr><td>E2Depth [5]</td><td></td><td>√</td><td>E</td><td>3.38</td><td>3.82</td><td>4.46</td><td>1.67</td><td>2.63</td><td>3.58</td><td>1.42</td><td>2.33</td><td>3.18</td><td>1.67</td><td>2.64</td><td>3.13</td></tr><tr><td>RAMNet [6]</td><td></td><td>√</td><td>E+I</td><td>2.50</td><td>3.19</td><td>3.82</td><td>1.21</td><td>2.31</td><td>3.28</td><td>1.01</td><td>2.34</td><td>3.43</td><td>1.39</td><td>2.17</td><td>2.76</td></tr><tr><td>EvT+ [22]</td><td></td><td>√</td><td>E+I</td><td>1.45</td><td>2.10</td><td>2.88</td><td>1.48</td><td>2.13</td><td>2.90</td><td>1.38</td><td>2.03</td><td>2.77</td><td>1.24</td><td>1.91</td><td>2.36</td></tr><tr><td>HMNet [25]</td><td></td><td>√</td><td>E+I</td><td>1.50</td><td>2.48</td><td>3.19</td><td>1.36</td><td>2.25</td><td>2.96</td><td>1.27</td><td>2.17</td><td>2.86</td><td>1.22</td><td>2.21</td><td>2.68</td></tr><tr><td>ER-F2D [7]</td><td></td><td>√</td><td>E+I</td><td>1.58</td><td>2.24</td><td>2.78</td><td>1.54</td><td>2.23</td><td>2.95</td><td>1.24</td><td>1.96</td><td>2.81</td><td>1.34</td><td>2.25</td><td>2.62</td></tr><tr><td>SRFNet [2]</td><td></td><td>√</td><td>E+I</td><td>1.26</td><td>1.95</td><td>3.01</td><td>1.19</td><td>2.13</td><td>3.22</td><td>1.01</td><td>2.12</td><td>3.52</td><td>0.96</td><td>1.77</td><td>2.37</td></tr><tr><td>EReFormer [23]</td><td></td><td>√</td><td>E+I</td><td>1.52</td><td>2.28</td><td>2.98</td><td>1.40</td><td>2.12</td><td>2.66</td><td>1.32</td><td>2.04</td><td>2.68</td><td>1.29</td><td>2.14</td><td>2.59</td></tr><tr><td>EventDAM [4]</td><td>×</td><td>×</td><td>E</td><td>1.39</td><td>2.10</td><td>3.25</td><td>1.43</td><td>2.18</td><td>3.22</td><td>1.44</td><td>2.16</td><td>3.22</td><td>1.12</td><td>1.79</td><td>2.69</td></tr><tr><td>DepthAnyEvent [3]</td><td>×</td><td>×</td><td>E</td><td>1.87</td><td>2.27</td><td>2.81</td><td>1.99</td><td>2.40</td><td>2.86</td><td>2.05</td><td>2.49</td><td>2.97</td><td>1.50</td><td>1.97</td><td>2.40</td></tr><tr><td>EventVGGT [8]</td><td>×</td><td>x</td><td>E</td><td>1.67</td><td>2.02</td><td>2.61</td><td>1.64</td><td>2.03</td><td>2.48</td><td>1.74</td><td>2.15</td><td>2.64</td><td>0.96</td><td>1.33</td><td>1.63</td></tr><tr><td>SFE-VGGT</td><td>√</td><td>×</td><td>E</td><td>1.42</td><td>2.04</td><td>2.92</td><td>1.36</td><td>1.90</td><td>2.62</td><td>1.50</td><td>2.01</td><td>2.89</td><td>1.00</td><td>1.53</td><td>2.14</td></tr></table>

Depth GT  
EventVGGT  
![](images/0a3fab44cc60a9988c9e76d2d617e18e9c8078263e035844fc37d74aa5485911.jpg)  
SFE-VGGT  
Fig. 3. Qualitative results on MVSEC dataset.  
TABLE III

ZERO-SHOT DEPTH ESTIMATION ON DENSE.  
MEAN ABSOLUTE DEPTH ERROR IS REPORTED IN METERS.
<table><tr><td>Method</td><td>Input</td><td>10m↓</td><td>20m ↓</td><td>30m↓</td></tr><tr><td>RAMNet [6]</td><td>E+I</td><td>2.62</td><td>11.26</td><td>19.11</td></tr><tr><td>SRFNet [2]</td><td>E+I</td><td>1.50</td><td>3.57</td><td>6.12</td></tr><tr><td>EventDAM [4]</td><td>E</td><td>1.20</td><td>2.60</td><td>5.18</td></tr><tr><td>EventVGGT [8]</td><td>E</td><td>0.54</td><td>0.89</td><td>1.33</td></tr><tr><td>SFE-VGGT</td><td>E</td><td>1.04</td><td>1.44</td><td>2.35</td></tr></table>

4) Direct VGGT transfer on MVSEC Night1 scene: This experiment examines whether pretrained VGGT [9] can be directly applied to event-based depth estimation without event-domain distillation. As shown in Tab. IV, SFE-VGGT substantially improves over direct VGGT with event inputs, reducing the error by 41.7%, 24.7%, and 15.1% at 10 m, 20 m, and 30 m, respectively. It also achieves the best 10 m result and remains close to EventVGGT [8] at 20 m. Overall, these results highlight the importance of event-domain distillation for effectively exploiting VGGT’s geometric priors. Qualitative results are shown in Fig. 3 (middle row).

TABLE IV  
DIRECT VGGT TRANSFER ANALYSIS ON MVSEC NIGHT1.  
MEAN ABSOLUTE DEPTH ERROR IS REPORTED IN METERS.
<table><tr><td>Method</td><td>Input</td><td>10m↓</td><td>20m↓</td><td>30m↓|  $\delta _ { 1 }$ </td><td>↑  $\delta _ { 2 }$ </td><td>↑  $\delta _ { 3 }$  个</td></tr><tr><td>VGGT [9]</td><td>E</td><td>2.42</td><td>2.71</td><td>3.44</td><td>0.42 0.69</td><td>0.84</td></tr><tr><td>VGGT [9]</td><td>I</td><td>2.31</td><td>2.68</td><td>3.33</td><td>0.41 0.71</td><td>0.88</td></tr><tr><td>EventVGGT [8]</td><td>E</td><td>1.67</td><td>2.02</td><td>2.61</td><td>0.57 0.82</td><td>0.93</td></tr><tr><td>SFE-VGGT</td><td>E</td><td>1.41</td><td>2.04</td><td>2.92</td><td>0.48 0.71</td><td>0.83</td></tr></table>

TABLE V

ABLATION OF THE PROPOSED TRAINING OBJECTIVES.
<table><tr><td>LDAFD</td><td>LCWDD</td><td>LCFRC</td><td>10m ↓</td><td>20m↓</td><td>30m↓</td><td>δ1↑</td><td> $\delta _ { 2 }$  ↑</td><td>δ3 ↑</td></tr><tr><td rowspan="3">L</td><td>√</td><td></td><td>0.620</td><td>0.951</td><td>1.337</td><td>0.740</td><td>0.880</td><td>0.948</td></tr><tr><td>√</td><td></td><td>0.622</td><td>0.947</td><td>1.327</td><td>0.739</td><td>0.875</td><td>0.942</td></tr><tr><td>√</td><td>L</td><td>0.604</td><td>0.948</td><td>1.336</td><td>0.752</td><td>0.887</td><td>0.947</td></tr><tr><td>√</td><td>√</td><td>√</td><td>0.617</td><td>0.941</td><td>1.322</td><td>0.743</td><td>0.880</td><td>0.948</td></tr></table>

## C. Ablation Studies

We conduct all ablation experiments using the Town2 split of EventScape for training and evaluate on the full EventScape test set. We examine three aspects of the proposed framework: the contribution of each training objective, the effectiveness of the key components introduced within these objectives, and the influence of input sequence length.

TABLE VI  
ABLATION OF KEY COMPONENTS WITHIN THE PROPOSED OBJECTIVES.
<table><tr><td>Settings</td><td>10m↓</td><td>20m↓</td><td>30m↓</td></tr><tr><td>w/o event-density weighting  $( \mathcal { L } _ { D A F D } )$ </td><td>0.633</td><td>0.965</td><td>1.347</td></tr><tr><td>w/o relative-confidence weighting  $( \mathcal { L } _ { C W D D } )$ </td><td>0.631</td><td>0.954</td><td>1.334</td></tr><tr><td>w/o RANSAC  $( \mathcal { L } _ { C F R C } )$ </td><td>0.630</td><td>0.944</td><td>1.325</td></tr><tr><td>Full model (all activated)</td><td>0.617</td><td>0.941</td><td>1.322</td></tr></table>

TABLE VII
<table><tr><td colspan="7">EFFECT OF INPUT SEQUENCE LENGTH.</td></tr><tr><td>Frames</td><td>10m↓</td><td>20m↓</td><td>30m↓</td><td>δ1↑</td><td>δ2 ↑</td><td> $\delta _ { 3 } \uparrow$ </td></tr><tr><td>3</td><td>0.666</td><td>1.034</td><td>1.510</td><td>0.693</td><td>0.834</td><td>0.911</td></tr><tr><td>6</td><td>0.681</td><td>1.025</td><td>1.471</td><td>0.706</td><td>0.847</td><td>0.921</td></tr><tr><td>12</td><td>0.677</td><td>1.007</td><td>1.424</td><td>0.725</td><td>0.866</td><td>0.937</td></tr><tr><td>24</td><td>0.617</td><td>0.941</td><td>1.322</td><td>0.743</td><td>0.880</td><td>0.943</td></tr></table>

1) Effect of the Distillation Objectives: Tab. V evaluates the contribution of the three training objectives. CWDD is retained in all variants as the primary depth-distillation objective, since removing it leads to severe performance degradation. Adding DAFD mainly benefits the 20 m and 30 m ranges, suggesting that reliability-aware feature transfer is more useful for preserving geometry at larger depths. In contrast, CFRC gives the strongest 10 m result and improves the stricter δ metrics, indicating that cross-frame relational supervision is particularly effective for near-range depth consistency. Combining all three objectives yields the most balanced performance across both depth ranges and threshold-based metrics, supporting the complementary roles of DAFD and CFRC alongside CWDD. Since all variants share the same VGGT initialization and retain CWDD under a reduced training budget, the overall trends are more informative than small differences in individual scores.

2) Effect of Key Components in the Proposed Objectives: Tab. VI evaluates the key mechanisms within DAFD, CWDD, and CFRC. Removing event-density weighting causes the most consistent degradation, particularly at the 20 m and 30 m ranges, supporting its role in suppressing unreliable feature transfer from sparsely observed regions. Relative-confidence weighting also improves performance across all ranges, indicating the benefit of selectively regulating depth supervision. Removing RANSAC leads to a smaller but consistent degradation, showing that filtering unreliable correspondences further improves the crossframe constraint. Overall, the full configuration performs best across all depth ranges, confirming that each reliability mech anism contributes to the proposed distillation framework.

3) Effect of Input Sequence Length: Tab. VII studies the influence of temporal context on depth estimation. Increasing the sequence length consistently improves the 20 m and 30 m errors as well as all δ metrics, while the 10 m error shows only minor variation for shorter sequences. The 24-frame setting achieves the best performance across all reported metrics, indicating that longer temporal windows provide richer cross-frame geometric cues for the VGGT backbone. Overall, a longer input sequence leads to more reliable depth estimation, with the clearest benefit appearing at larger depth ranges and in the threshold-based metrics.

## V. CONCLUSION

In this paper, we presented SFE-VGGT, a source-free distillation framework for transferring VGGT’s geometric priors to event-based depth estimation without paired RGB observations or metric depth supervision. By using E2VIDreconstructed frames as teacher inputs, SFE-VGGT removes the need for paired RGB observations during distillation. To address the imperfect supervision introduced by event reconstruction, we propose Density-Aware Feature Distillation (DAFD), Confidence-Weighted Depth Distillation (CWDD), and Cross-Frame Relational Consistency (CFRC) for reliable feature, depth, and temporal geometric transfer. Extensive experiments on EventScape, MVSEC, and DENSE show that SFE-VGGT remains competitive with RGB-assisted distillation methods, performs particularly well under challenging lighting conditions, where conventional RGB observations are often degraded, and exhibits strong zero-shot generalization. These results demonstrate the feasibility of transferring geometric foundation-model priors to event-based modality without relying on synchronized RGB training data.

Limitations and Future Work. The current framework relies on a frozen E2VID model to construct the teacher observations, and its performance can therefore be influenced by the quality of the reconstructed frames. While the proposed reliability-aware objectives reduce the impact of uncertain reconstruction regions, the reconstruction module itself is not adapted to the downstream geometric task. Future work will explore joint optimization of the reconstruction and distillation framework, allowing the reconstructed representation to better support the extraction of geometric priors from VGGT while maintaining stable training.

## REFERENCES

[1] G. Gallego, T. Delbruck, G. Orchard, C. Bartolozzi, B. Taba, A. Censi, S. Leutenegger, A. J. Davison, J. Conradt, K. Daniilidis, and D. Scaramuzza, “Event-based vision: A survey,” IEEE Transactions on Pattern Analysis and Machine Intelligence, vol. 44, no. 1, pp. 154–180, 2022.

[2] T. Pan, Z. Cao, and L. Wang, “Srfnet: Monocular depth estimation with fine-grained structure via spatial reliability-oriented fusion of frames and events,” in 2024 IEEE International Conference on Robotics and Automation (ICRA), pp. 10695–10702, IEEE, 2024.

[3] L. Bartolomei, E. Mannocci, F. Tosi, M. Poggi, and S. Mattoccia, “Depth anyevent: A cross-modal distillation paradigm for event-based monocular depth estimation,” in Proceedings of the IEEE/CVF International Conference on Computer Vision, pp. 19669–19678, 2025.

[4] J. Zhu, T. Pan, Z. Cao, Y. Liu, J. T. Kwok, and H. Xiong, “Depth any event stream: Enhancing event-based monocular depth estimation via dense-to-sparse distillation,” in Proceedings of the IEEE/CVF International Conference on Computer Vision, pp. 5146–5155, 2025.

[5] J. Hidalgo-Carrio, D. Gehrig, and D. Scaramuzza, “Learning monoc-´ ular dense depth from events,” in 2020 International Conference on 3D Vision (3DV), pp. 534–542, IEEE, 2020.

[6] D. Gehrig, M. Ruegg, M. Gehrig, J. Hidalgo-Carri ¨ o, and D. Scara- ´ muzza, “Combining events and frames using recurrent asynchronous multimodal networks for monocular depth prediction,” IEEE Robotics and Automation Letters, vol. 6, no. 2, pp. 2822–2829, 2021.

[7] A. Devulapally, M. F. F. Khan, S. Advani, and V. Narayanan, “Multimodal fusion of event and rgb for monocular depth estimation using a unified transformer-based architecture,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition Workshops, pp. 2081–2089, 2024.

[8] Y. Ren, J. Zhu, K. Chen, Z. Li, J. Ou, Z. Cao, T. Hua, P. Shi, Y. Fu, W. Zhao, and H. Xiong, “Eventvggt: Exploring cross-modal distillation for consistent event-based depth estimation,” arXiv preprint arXiv:2603.09385, 2026.

[9] J. Wang, M. Chen, N. Karaev, A. Vedaldi, C. Rupprecht, and D. Novotny, “Vggt: Visual geometry grounded transformer,” in Proceedings ofthe IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 5294–5306, 2025.

[10] H. Rebecq, R. Ranftl, V. Koltun, and D. Scaramuzza, “High speed and high dynamic range video with an event camera,” IEEE transactions on pattern analysis and machine intelligence, vol. 43, no. 6, pp. 1964– 1980, 2019.

[11] D. Gehrig, M. Ruegg, M. Gehrig, J. Hidalgo-Carri¨ o, and D. Scara-´ muzza, “Combining events and frames using recurrent asynchronous multimodal networks for monocular depth prediction,” IEEE Robotics and Automation Letters, vol. 6, no. 2, pp. 2822–2829, 2021.

[12] A. Z. Zhu, D. Thakur, T. Ozaslan, B. Pfrommer, V. Kumar, and K. Daniilidis, “The multivehicle stereo event camera dataset: An event camera dataset for 3d perception,” IEEE Robotics and Automation Letters, vol. 3, no. 3, pp. 2032–2039, 2018.

[13] J. Hidalgo-Carrio, D. Gehrig, and D. Scaramuzza, “Learning monoc-´ ular dense depth from events,” in 2020 International Conference on 3D Vision (3DV), pp. 534–542, IEEE, 2020.

[14] A. Radford, J. W. Kim, C. Hallacy, A. Ramesh, G. Goh, S. Agarwal, G. Sastry, A. Askell, P. Mishkin, J. Clark, G. Krueger, and I. Sutskever, “Learning transferable visual models from natural language supervision,” in Proceedings of the 38th International Conference on Machine Learning, vol. 139 of Proceedings of Machine Learning Research, pp. 8748–8763, PMLR, 2021.

[15] M. Caron, H. Touvron, I. Misra, H. Jegou, J. Mairal, P. Bojanowski,´ and A. Joulin, “Emerging properties in self-supervised vision transformers,” in Proceedings of the IEEE/CVF International Conference on Computer Vision, pp. 9650–9660, 2021.

[16] K. He, X. Chen, S. Xie, Y. Li, P. Dollar, and R. Girshick, “Masked´ autoencoders are scalable vision learners,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 16000–16009, 2022.

[17] M. Oquab, T. Darcet, T. Moutakanni, H. V. Vo, M. Szafraniec, V. Khalidov, P. Fernandez, D. Haziza, F. Massa, A. El-Nouby, M. Assran, N. Ballas, W. Galuba, R. Howes, P.-Y. Huang, S.-W. Li, I. Misra, M. Rabbat, V. Sharma, G. Synnaeve, H. Xu, H. Jegou,´ J. Mairal, P. Labatut, A. Joulin, and P. Bojanowski, “Dinov2: Learning robust visual features without supervision,” Transactions on Machine Learning Research, 2024.

[18] A. Kirillov, E. Mintun, N. Ravi, H. Mao, C. Rolland, L. Gustafson, T. Xiao, S. Whitehead, A. C. Berg, W.-Y. Lo, P. Dollar, and´ R. Girshick, “Segment anything,” in Proceedings of the IEEE/CVF International Conference on Computer Vision, pp. 4015–4026, 2023.

[19] L. Yang, B. Kang, Z. Huang, Z. Zhao, X. Xu, J. Feng, and H. Zhao, “Depth anything v2,” in Advances in Neural Information Processing Systems, vol. 37, 2024.

[20] Z. Chen, Z. Zhu, Y. Zhang, J. Hou, G. Shi, and J. Wu, “Segment any event streams via weighted adaptation of pivotal tokens,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 3890–3900, 2024.

[21] L. Kong, Y. Liu, L. X. Ng, B. R. Cottereau, and W. T. Ooi, “Openess: Event-based semantic scene understanding with open vocabularies,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 15686–15698, 2024.

[22] A. Sabater, L. Montesano, and A. C. Murillo, “Event transformer<sup>+</sup>: A multi-purpose solution for efficient event data processing,” IEEE Transactions on Pattern Analysis and Machine Intelligence, vol. 45, no. 12, pp. 16013–16020, 2023.

[23] X. Liu, J. Li, J. Shi, X. Fan, Y. Tian, and D. Zhao, “Event-based monocular depth estimation with recurrent transformers,” IEEE Transactions on Circuits and Systems for Video Technology, vol. 34, no. 8, pp. 7417–7429, 2024.

[24] P. Shi, J. Peng, J. Qiu, X. Ju, F. P. W. Lo, and B. Lo, “Even: An eventbased framework for monocular depth estimation at adverse night conditions,” in 2023 IEEE International Conference on Robotics and Biomimetics (ROBIO), pp. 1–7, IEEE, 2023.

[25] R. Hamaguchi, Y. Furukawa, M. Onishi, and K. Sakurada, “Hierarchical neural memory network for low latency event processing,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 22867–22876, 2023.

[26] X. Liu, X. Fan, J. Li, D. Li, W. Zhang, Z. Ma, and Y. Tian, “Highrate monocular depth estimation via cross frame-rate collaboration of frames and events,” International Journal of Computer Vision, vol. 133, no. 10, pp. 7332–7351, 2025.

[27] G. Hinton, O. Vinyals, and J. Dean, “Distilling the knowledge in a neural network,” arXiv preprint arXiv:1503.02531, 2015.

[28] L. Wang and K.-J. Yoon, “Knowledge distillation and student-teacher learning for visual intelligence: A review and new outlooks,” IEEE transactions on pattern analysis and machine intelligence, vol. 44, no. 6, pp. 3048–3068, 2021.

[29] S. Gupta, J. Hoffman, and J. Malik, “Cross modal distillation for supervision transfer,” in Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition, pp. 2827–2836, 2016.

[30] P. Lee, T. Kim, M. Shim, D. Wee, and H. Byun, “Decomposed cross-modal distillation for rgb-based temporal action detection,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 2373–2383, 2023.

[31] S. Jang, D. U. Jo, S. J. Hwang, D. Lee, and D. Ji, “Stxd: Structural and temporal cross-modal distillation for multi-view 3d object detection,” in Advances in Neural Information Processing Systems, vol. 36, 2023.

[32] J. Liang, D. Hu, and J. Feng, “Do we really need to access the source data? source hypothesis transfer for unsupervised domain adaptation,” in Proceedings of the 37th International Conference on Machine Learning, vol. 119 of Proceedings of Machine Learning Research, pp. 6028–6039, PMLR, 2020.

[33] S. Yang, Y. Wang, J. van de Weijer, L. Herranz, and S. Jui, “Exploiting the intrinsic neighborhood structure for source-free domain adaptation,” in Advances in Neural Information Processing Systems, vol. 34, 2021.

[34] N. Ding, Y. Xu, Y. Tang, C. Xu, Y. Wang, and D. Tao, “Source-free domain adaptation via distribution estimation,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 7212–7222, 2022.

[35] S. M. Ahmed, S. Lohit, K.-C. Peng, M. J. Jones, and A. K. Roy-Chowdhury, “Cross-modal knowledge transfer without task-relevant source data,” in European Conference on Computer Vision, pp. 111– 127, Springer, 2022.

[36] J. Zhu, Y. Chen, and L. Wang, “Source-free cross-modal knowledge transfer by unleashing the potential of task-irrelevant data,” IEEE Transactions on Image Processing, vol. 34, pp. 2840–2852, 2025.

[37] X. Zheng and L. Wang, “Eventdance: Unsupervised source-free crossmodal adaptation for event-based object recognition,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 17448–17458, 2024.

[38] D. Eigen, C. Puhrsch, and R. Fergus, “Depth map prediction from a single image using a multi-scale deep network,” Advances in neural information processing systems, vol. 27, 2014.

[39] D. G. Lowe, “Distinctive image features from scale-invariant keypoints,” International journal of computer vision, vol. 60, no. 2, pp. 91–110, 2004.

[40] M. A. Fischler and R. C. Bolles, “Random sample consensus: a paradigm for model fitting with applications to image analysis and automated cartography,” Communications of the ACM, vol. 24, no. 6, pp. 381–395, 1981.
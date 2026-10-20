# XGLASS: X-Ray, Gaze and Language Assisted Slice Sequencing for CT Volumes

Michael Glockler and Timo Ropinski¨

Ulm University, Germany

{michael.gloeckler, timo.ropinski}@uni-ulm.de

## Abstract

Vision-language models (VLMs) achieve remarkable performance on 2D medical images, such as X-rays or pathology images, while 3D VLMs that directly process CT/MRI volumes lag behind due to limited 3D training data and the predominance of 2D foundation architectures. To leverage powerful 2D VLMs for analyzing medical volumes, which contain hundreds of slices, it is therefore necessary to focus only on those few slices, which are diagnostically relevant. To enable such a prioritization, we propose XGLASS, a method that transfers medical expertise from 2D X-rays to 3D CT for task-aware slice prioritization. XGLASS uses disease-conditioned importance maps derived from language-aligned radiologist gaze on X-rays and spatially aligns them with coronal CT projections, bringing gaze priors and CT content into a shared 2D representation. The fused gaze-prior/CT projection is then used to predict a disease-specific ordering of axial slices, prioritizing diagnostically relevant slices. XGLASS significantly improves budgeted retrieval over 2D and 3D-native baselines, achieving a top-10 hit rate up to 75.9% for the best-performing disease. In practice, this means that after inspecting only the top 10 slices (4.17% of a 240-slice scan), XGLASS identifies a diagnostically relevant slice in more than 75% ofcases. For downstream slice-wisefinding classification, an XGLASS-selected 30-slice candidate set reduces model calls by 87.5% while improving F1 by 11.3 percentage points over all-slice inference, 4.2 points over the strongest 2D selector, and remains competitive with 3Dnative selectors while requiring up to 130.8× less time and 10.9× less GPU memory perfinding, supporting its use as a lightweight slice selectorfor volumetric CT analysis. The code and pretrained checkpoint are available on GitHub.

## 1. Introduction

Vision-language models (VLMs) have recently enabled remarkable progress in computer vision, in tasks such as zero-/few-shot recognition, retrieval, and visual question answering (VQA), largely driven by the abundance of real-world image-text pairs [1, 2, 8, 33, 37, 38]. Analogously, the release of large paired medical datasets [7, 17, 18] has catalyzed medical adaptations, leading to CLIP-style medical contrastive models [25, 42, 46, 47, 49] and medica VLMs [24, 31, 35] capable of report generation, diagnosis, and medical VQA. Alongside these models, a growing ecosystem of 2D medical datasets has emerged, such as closed- and open-ended VQA benchmarks targeting medical questions [15, 23, 26, 48]. However, the majority of these resources contains 2D medical images, such as xray or pathology slides. When CT or MRI appear in those datasets, they are typically represented by single, diagnostically relevant slices that were manually extracted by experts, rather than complete 3D volumes [15, 23, 26]. This results in a mismatch between dataset availability and diag nostic requirements, since core diagnostic modalities, such as CT or MRI, are inherently 3D. Applying 2D VLMs directly to complete CT volumes would require processing every axial slice independently, which is computationally expensive and often inefficient because most slices are redundant or irrelevant to a queried finding. In practice, downstream applications such as slice-wise VLM screening, computer-aided detection (CAD), and 2D dataset curation benefit from compact disease-conditioned candidate sets rather than exhaustive all-slice inference [41]. Motivated by this gap between predominantly 2D VLMs and the 3D nature of clinical imaging, we propose X-Ray, Gaze and Language Assisted Slice Sequencing (XGLASS), a lightweight method that transfers radiologist gaze-language priors learned on chest X-rays (CXRs) to rank axial CT slices by diagnostic relevance in a disease-conditioned manner. XGLASS effectively regresses a continuous, 240- dimensional relevance profile over axial slices, enabling relevance ranking, top-k slice extraction, and importancebased traversal for retrieval or VLM conditioning. Empirically, XGLASS improves budgeted slice retrieval over 2D and 3D-native baselines, reaching 75.9% Hit@10 for the best-performing disease, while downstream VLM screening shows that XGLASS-selected candidate sets reduce mode calls by 87.5% and improve F1 by up to 11.3 points over all-slice inference.

To this end, we make the following contributions:

• We introduce disease-conditioned importance maps from medical expert knowledge collected through x-ray eyetracking data and synchronized speech recordings.

• We propose XGLASS, a lightweight 2D-to-3D crossmodal model that transfers these priors to CT projections and predicts disease-specific axial slice relevance scores.

• We evaluate XGLASS against 2D and 3D-native baselines, showing significantly better budgeted retrieval, substantially lower inference cost, and downstream F1 gains while reducing model calls by 87.5%.

## 2. Related Work

Eye-tracking in medical imaging. Radiologist eyetracking provides implicit supervision of diagnostic search behavior and has been used to study visual expertise, diagnostic performance, and attention allocation [12, 20, 22, 39]. Recent CXR datasets such as REFLACX and MIMIC-Eye link gaze with report transcripts [5, 16, 19], enabling gaze-guided localization, attention steering, and multimodal alignment [4, 21, 28]. In contrast, eye-tracking data for 3D CT remain scarce: CT-Scangaze [32] provides 3D scanpaths with dictation transcripts, but its limited number of only 57 distinct scanpath–report pairs makes it unsuitable for deriving stable disease-specific importance maps.

Slice selection for 3D medical imaging. CT and MRI interpretation requires scrolling through volumetric slices, making slice selection useful for reducing redundancy and enabling 2D models to process 3D data. Vote-MI [41] selects representative slices from 2D features, but remains unsupervised, not disease-conditioned, and requires encoding all slices before selection, increasing computational cost. Although CT reading differs from CXR interpretation through slice drilling and z-axis-ordered search [12], our CXR–CT gaze-prior comparison in Supplementary Sec. B.4 shows correlated superior–inferior disease-attention profiles across modalities $( r { = } 0 . 6 5 { - } 0 . 9 6 )$ , supporting XGLASS’s transfer of 2D gaze priors for CT axial slice sequencing.

## 3. Method

Based on x-ray, gaze, and language input, XGLASS learns $f ( P , G ) \to { \hat { y } } ,$ which processes a coronal CT intensity projection $P ,$ enriched with a disease-specific gaze prior $G ,$ in the form of an importance map. For clarity, the overall pipeline is illustrated in Figure 1. Based on $P$ and $G , f$ predicts a continuous importance vector $\hat { y }$ of length $Z _ { \mathrm { t a r g e t } } .$ Each entry of this vector represents the diagnostic relevance of a single axial slice with respect to the disease conditioned through the importance map. By learning $f ( P , G )$ with a ViT-based regression model, we enable effective diseasespecific sequencing of the CT volume, from which $P$ is obtained. In the following subsections, we first describe how to generate $G ,$ then how to project a CT volume to obtain $P ,$ and finally how we learn $f .$

## 3.1. Disease-Conditioned Importance Maps

A central component of XGLASS is the construction of disease-specific 2D importance maps G that capture how expert radiologists navigate through chest X-rays. To obtain these priors, we leverage the eye-tracking data paired with dictated report texts, available in the REFLACX dataset [5]. For disease filtering for each report transcript, we use custom regular expressions similar to SARLE [11] to identify disease-specific mentions in the spoken text. The timestamps associated with these textual matches define a temporal window during which the radiologist was verbally describing the relevant findings. Prior work in psycholinguistics shows that speakers’ gaze reliably anticipates verbal articulation by more than one second, motivating the inclusion of a short pre-speech buffer on the order of 1–1.5 s [14]. Given the cognitive load and visual search demands of CT interpretation, we included a slightly longer pre-speech buffer of 2 s to ensure we capture all relevant fixation points. Formally, the gaze pooling process is described as follows, for a spoken mention with start time $t _ { \mathrm { s t a r t } }$ and end time $t _ { \mathrm { e n d } } .$ we define the effective gaze window as $[ t _ { \mathrm { s t a r t } } - 2 \mathrm { s } , ~ t _ { \mathrm { e n d } } ]$ All gaze fixations that fall into this window are then extracted, ensuring that only gaze from the relevant diagnostic moment contributes to the disease-conditioned importance map. To obtain a consistent spatial representation, we normalize each fixation with respect to the REFLACX chest bounding box $\boldsymbol { B } _ { \mathrm { c h e s t } } = ( x _ { 0 } , y _ { 0 } , w , h )$ and discard fixations falling outside $B _ { \mathrm { c h e s t } }$ (potentially resulting from tracker drift or blinks). The box-normalized coordinates are

$$
\tilde { x } = \frac { x - x _ { 0 } } { w } , \qquad \tilde { y } = \frac { y - y _ { 0 } } { h } ,
$$

so that $( \tilde { x } , \tilde { y } ) \in [ 0 , 1 ] ^ { 2 }$ . We then aggregate the normalized fixations across studies for the same disease category and smooth them by placing a Gaussian at each fixation to obtain a continuous density map [43]. Since REFLACX contains frontal CXRs acquired in both AP and PA views, we additionally analyze whether the resulting gaze priors are sensitive to view-specific acquisition geometry. In Sec. B.3 of the supplementary material, AP- and PA-specific gaze maps show only minor differences in their superior–inferior profiles; therefore, we aggregate both views to maximize the number of gaze samples available per disease. The resulting disease-conditioned importance maps capture the aggregated spatial attention pattern for each finding and serve as weak expert annotation priors that are subsequently fused with coronal CT intensity projections to create the final input for XGLASS.

![](images/b387540c6faade8866142111cd1bcbc2b9d97c998c6308f212f16a188d7f0dec.jpg)  
Figure 1. Overview of the XGLASS Pipeline. Given a chest CT and chest X-ray eye-tracking data with matching report transcripts as input, we first normalize CT volumes and compute coronal CT intensity projections, while also matching gaze and disease mentions to derive disease-conditioned importance maps. These importance maps are spatially aligned to the CT projection and fused via alpha blending. The gaze-enriched intensity projection is input to the ViT-based regression head trained with MSE to predict a slice-importance vector. For inference, the model uses the precomputed importance map for the desired disease fused with the CT intensity projection and outputs a disease-specific relevance ordering of axial slices, ranking diagnostically important slices at the top.

## 3.2. CT Volume Projection

Our method takes two complementary sources of supervision: pre-computed, disease-specific 2D gaze importance maps derived from radiologist eye-tracking on chest Xrays using the REFLACX dataset [5], and a volumetric CT dataset that shares matching disease findings and therefore ReXGroundingCT [3] is to our knowledge the largest CT dataset that fulfills our requirements. To project a given CT volume into the disease-conditioned importance map domain, we first need to perform an intensity projection followed by a projection window normalization.

Intensity projection: CT volume projections enable us to project a given 3D volume into a 2D image, which in case of a coronal projection, is aligned with our diseaseconditioned importance maps. Formally, let $f ( \mathbf { x } )$ denote the volumetric attenuation at location $\mathbf { x } \in \mathbb { R } ^ { 3 }$ . A projection along an imaging direction d samples parallel-beams $\mathbf { x } ( u , v , t ) = \mathbf { o } ( u , v ) + t$ d and applies an aggregation operator A to the line signal $f ( \mathbf { x } ( u , v , t ) )$ :

$$
\mathcal { P } _ { A } ( u , v ) ~ = ~ \mathcal { A } \big ( \{ ~ f ( \mathbf { o } ( u , v ) + t \mathbf { d } ) ~ \} _ { t \in [ 0 , L ] } \big ) .\tag{1}
$$

Because the coronal projection preserves the superior– inferior coordinate indexing axial slices, aggregating anterior–posterior depth retains the positional information required for axial slice selection. Accordingly, d follows the anterior–posterior axis, while different choices of A yield standard projection types. To simulate real-world x-ray projection $\scriptstyle A _ { B , C }$ would resemble the Beer-Lambert law [36], which assumes that x-rays are only attenuated but not reflected at x:

$$
\operatorname { A } _ { \mathrm { B L } } ( u , v ) \ = \ \exp ( - \kappa \int _ { 0 } ^ { L } f ( \mathbf { o } ( u , v ) + t \mathbf { d } ) \mathrm { d } t ) ,\tag{2}
$$

whereby κ is the constant extinction coefficient, which models absorption and out-scattering. An average-intensity projection (AIP) in contrast uses the line integral normalized by its length:

$$
\mathrm { A } _ { \mathrm { I P } } ( u , v ) \ = \ \frac { 1 } { L } \int _ { 0 } ^ { L } f ( \mathbf { o } ( u , v ) + t \mathbf { d } ) \mathrm { d } t .\tag{3}
$$

In the field of radiology, additionally, the maximumintensity projection (MIP), defined as $\begin{array} { r l } { \mathrm { A _ { M I P } } ( u , v ) } & { { } = } \end{array}$ $\operatorname* { m a x } _ { t \in [ 0 , L ] } f ( \mathbf { o } ( u , v ) + t \mathbf { d } )$ , plays an important role. However, MIPs emphasize very high-density structures (e.g. bones) and de-emphasize soft-tissue pathology, which conflicts with the intensity ranges in which our target findings occur. Therefore, throughout this paper, we solely focus on the Beer-Lambert projection and the AIP. We have ablated both (see Section 4.4.2), whereby we compare three standard CT windowings using AIP, i.e., lung, soft-tissue, and a wide window.

Projection window normalization. To ensure the described intensity projections are aligned with the obtained disease-conditioned importance maps, anatomy normalization needs to be applied. During this normalization, our training set of 3,142 CT volumes was resampled to a uniform isotropic spacing of $1 . 2 5 \times 1 . 2 5 \times 1 . 2 5 \mathrm { m m }$ While this standardizes resolution, it does not reconcile variability in scan coverage along the Z-axis, which is critical since XGLASS 2D-to-3D matching expects a fixed amount of axial slices. To avoid the anatomical distortion introduced by global resizing (i.e., stretching or squashing), we adopt a fixed-size cropping strategy and define a target axial block size $Z _ { \mathrm { t a r g e t } }$ to be extracted from each volume. To further minimize exclusions, we sweep $Z _ { \mathrm { t a r g e t } }$ to find an optimal setting. A CT volume $V _ { i }$ is included only if two constraints hold: (1) its total axial slice count $Z _ { \mathrm { t o t a l } , i }$ <sub>i</sub> (at 1.25 mm) is large enough to contain the block, and (2) its pre-computed chest bounding-box height $Z _ { \mathrm { b b o x . } }$ <sub>,i</sub> (at 1.25 mm) fits within the block.

$$
V _ { i } \in \mathrm { C o h o r t } _ { \mathrm { f i n a l } } \ \Leftrightarrow \ \left( Z _ { \mathrm { t o t a l } , i } \geq Z _ { \mathrm { t a r g e t } } \right) \wedge \left( Z _ { \mathrm { b b o x } , i } \leq Z _ { \mathrm { t a r g e t } } \right)
$$

We tested integer values $Z _ { \mathrm { t a r g e t } } \in \ [ 1 2 8 , 5 1 2 ]$ and found an optimal value of $Z _ { \mathrm { t a r g e t } } = 2 4 0$ slices. This choice minimized exclusions by balancing chest-bbox inclusion (median $Z _ { \mathrm { b b o x } , i } ~ = ~ 2 0 5 )$ against total volume length (median $Z _ { \mathrm { t o t a l } , i } = 2 6 9 )$ . As a result, the final cohort comprises 2,078 unique CTs, with 1,064 excluded for not meeting the constraints. For all included CTs, we then extract a fixed axial block of 240 slices (1.25 mm spacing) centered on the chest bbox, yielding a standardized ReXGroundingCT subset with a consistent 240-slice axial size across scans. Further dataset statistics and category-selection details are provided in Sec. C of the supplementary material.

Ground-truth slice-area vector extraction. A key advantage of XGLASS is that it does not necessarily require full 3D segmentation masks: it can also be trained using only the start and end slices of a finding, as demonstrated by the SE-Binary and SE-Smooth ablations in Sec. 4.4.1. When segmentation masks are available, as in ReXGroundingCT, we use the more detailed Slice-Area target by deriving perslice relevance scores from the 3D ground-truth masks. Therefore each mask is converted into a 240-dimensional disease-area vector over the fixed axial block. Let A<sub>i</sub> denote the disease area on slice i, with $A _ { i } = 0$ when the finding is absent. We compute

$$
M = \operatorname* { m a x } _ { j = 1 , \ldots , 2 4 0 } A _ { j } ,
$$

and define the target vector $\mathbf { y } ~ = ~ ( y _ { 1 } , \dots , y _ { 2 4 0 } )$ by maxnormalization:

$$
y _ { i } = { \left\{ \begin{array} { l l } { A _ { i } / M , } & { { \mathrm { i f ~ } } M > 0 , } \\ { 0 , } & { { \mathrm { i f ~ } } M = 0 , } \end{array} \right. } \qquad i = 1 , \ldots , 2 4 0 .
$$

Thus, the slice with the largest disease extent receives $y _ { i } =$ 1, absent slices remain at 0, and intermediate slices receive values proportional to their disease area.

## 3.3. 2D-3D Cross-Modal Slice Sequencing

Our model builds on a ViT-B/16 backbone with AugReg ImageNet-21k pretraining [10], which processes the fused representation of the disease-conditioned importance map G and projected CT volume P. To adapt the network from classification to slice-importance regression, we replace the original classification head with a lightweight twolayer MLP that outputs a fixed-length vector $\hat { y } ~ \in ~ \mathbb { R } ^ { Z _ { \mathrm { t a r g e t } } }$ $( Z _ { \mathrm { t a r g e t } } = 2 4 0 )$ . A final sigmoid activation constrains all predictions to the range [0, 1], matching the normalized ground-truth slice–area profiles. The model receives a 3- channel $2 2 4 \times 2 2 4$ image initialized with a coronal CT intensity projection P, replicated across all three input channels. The corresponding disease-specific importance map G is then spatially scaled to the chest bounding-box region in the projection and blended into the red channel with an opacity coefficient α, while the green and blue channels remain unchanged. This channel-wise injection allows the ViT to process the anatomical projection and gaze-derived prior jointly in a single forward pass, enabling the model to estimate slice relevance directly from the fused representation. During training, the model obtains as input the fused G and P image, and learns to regress the groundtruth slice–area vector derived in Sec. 3.2. For each input image, the network outputs $\hat { y } \in [ 0 , 1 ] ^ { 2 4 0 }$ and is optimized using MSE loss between yˆ and the corresponding normalized ground-truth vector y. We train the full model end-toend using the AdamW optimizer with a differential learning rate schedule, assigning a higher learning rate to the regression head $( 1 \times 1 0 ^ { - 4 } )$ and a lower rate to the ViT backbone $( 1 \times 1 0 ^ { - 5 } )$ . Training is performed with a batch size of 16. This setup preserves the pretrained backbone’s representational capacity while allowing the head to adapt to the sliceimportance regression task.

## 4. Results

We evaluate XGLASS on a held-out test set (split details in Supp. Sec. C.2) to answer the following questions: (i) How does XGLASS perform compared to existing 2D and 3D slice selection baselines such as Vote-MI [41], BioMed-CLIP [46], Merlin [6], 3D U-Net [9] and heuristic approaches (ii) does injecting 2D gaze–language priors improve 3D slice sequencing, and (iii) does this translate into practically useful retrieval of diagnostically relevant slices?

## 4.1. Evaluation Metrics

We used different metrics to assess how well XGLASS orders all axial slices and how useful this ordering is for retrieval in potential downstream tasks. Let $Z { = } 2 4 0$ be the number of axial slices, $y \in [ 0 , 1 ] ^ { Z }$ the ground-truth slicearea vector, and $\hat { y } \in [ 0 , 1 ] ^ { Z }$ the predicted importance scores and a slice is positive if $y _ { i } > 0 .$ . Unless otherwise noted, all metrics are computed per study and then macro-averaged. For overall ordering quality, we use ROC-AUC and Spearman’s rank correlation $\rho$ [34] between y and yˆ over all Z slices, which directly evaluates agreement of the complete 240-slice ordering from most to least relevant. For practical analysis of performance on downstream tasks, we measure the hit rate at K (Hit@K) by sorting slices in descending yˆ and checking whether any positive slice appears among the first K predicted slices. Further, we also calculate the average recall at K (AR@K), indicating the fraction of all GT positive slices that are represented within the K predictions.

Table 1. Main Results. Metrics are reported as point estimate with 95% confidence interval below. Intervals are computed using the non parametric percentile bootstrap [13] over the test set with 10,000 resamples. The last rows report paired absolute differences in percentage points between XGLASS and the strongest 2D or 3D baseline for each metric; bold indicates statistical significance.
<table><tr><td>Method</td><td>Hit@1</td><td>Hit@3</td><td>Hit@5</td><td>Hit@10</td><td>AR@10</td><td>AUC</td><td>Spear. (ρ)</td></tr><tr><td>Random</td><td>0.101 [0.070, 0.136]</td><td>0.250 [0.203, 0.298]</td><td>0.342 [0.291, 0.396]</td><td>0.491 [0.437, 0.547]</td><td>0.043 [0.035, 0.051]</td><td>0.504 [0.497, 0.511]</td><td>0.003 [-0.004, 0.010]</td></tr><tr><td>Gaussian Centered</td><td>0.190 [0.149, 0.234]</td><td>0.237 [0.190, 0.285]</td><td>0.278 [0.231, 0.329]</td><td>0.323 [0.272, 0.377]</td><td>0.082 [0.064, 0.101]</td><td>0.718 [0.698, 0.738]</td><td>0.208 [0.185, 0.233]</td></tr><tr><td>Gaze prior only</td><td>0.158 [0.120, 0.199]</td><td>0.177 [0.136, 0.222]</td><td>0.209 [0.165, 0.253]</td><td>0.263 [0.215, 0.313]</td><td>0.068 [0.051, 0.088]</td><td>0.638 [0.615, 0.661]</td><td>0.163 [0.141, 0.186]</td></tr><tr><td>Vote-MI [41]</td><td>0.044 [0.022, 0.070]</td><td>0.190 [0.146, 0.234] 0.282</td><td>0.339 [0.288, 0.392] 0.339</td><td>0.465 [0.411, 0.519] 0.443</td><td>0.033 [0.027, 0.038]</td><td>0.472 [0.454, 0.491]</td><td>-0.025 [-0.036, -0.015]</td></tr><tr><td>BioMedCLIP [46]</td><td>0.212 [0.168, 0.256] 0.231</td><td>[0.234, 0.332] 0.275</td><td>[0.288, 0.392] 0.301</td><td>[0.389, 0.500] 0.361</td><td>0.070 [0.058, 0.083] 0.089</td><td>0.658 [0.627, 0.688]</td><td>0.164 [0.142, 0.187] 0.228</td></tr><tr><td>3D U-Net [9]</td><td>[0.187, 0.278] 0.247</td><td>[0.225, 0.326] 0.297</td><td>[0.250, 0.351] 0.320</td><td>[0.307, 0.415] 0.399</td><td>[0.071, 0.107]</td><td>0.785 [0.763, 0.806]</td><td>[0.204, 0.253]</td></tr><tr><td>3D Merlin + MLP [6]</td><td>[0.199, 0.294]</td><td>[0.247, 0.348] 0.288</td><td>[0.269, 0.370] 0.342</td><td>[0.345, 0.453]</td><td>0.094 [0.075, 0.114]</td><td>0.779 [0.755, 0.800]</td><td>0.224 [0.201, 0.249]</td></tr><tr><td>Ours (w/o gaze cues)</td><td>0.199 [0.155, 0.244] 0.228</td><td>[0.237, 0.339] 0.339</td><td>[0.288, 0.396] 0.421</td><td>0.468 [0.415, 0.525] 0.563</td><td>0.077 [0.065, 0.091] 0.091</td><td>0.768 [0.745, 0.791]</td><td>0.212 [0.189, 0.235]</td></tr><tr><td>Ours (w/ gaze cues)</td><td>[0.180, 0.275]</td><td>[0.288, 0.392]</td><td>[0.367, 0.475]</td><td>[0.509, 0.617]</td><td>[0.077, 0.106]</td><td>0.781 [0.760, 0.801]</td><td>0.224 [0.202, 0.246]</td></tr><tr><td>best 2D baseline ∆ vs. best 3D baseline</td><td>+1.6% -1.9%</td><td>+5.7% +4.1%</td><td>+7.9% +10.1%</td><td>+7.3% +16.5%</td><td>+0.9% -0.3%</td><td>+6.3% -0.5%</td><td>+1.5% -0.5%</td></tr></table>

![](images/79bebde663456f06ffca852949cdc4f93f07cb656bce9bbb05d0012ddc68a878.jpg)  
Figure 2. Average Ground-Truth (GT) Coverage. For each CT, we compute the fraction of all GT slices recovered within the top-K predictions. XGLASS outperforms the 2D baselines and compete with the 3D baselines despite substantially less compute.

In summary, higher $\rho$ and ROC-AUC indicate better global ranking alignment, higher Hit@K shows that a GT slice is in the K predictions, and AR@K reflects the percentage amount of all GT slices inside K predictions.

## 4.2. Baselines

We benchmark XGLASS against the state-of-the-art slice selection method Vote-MI [41], a BioMedCLIP-based slicewise retrieval baseline [46], heuristic and prior-only baselines, two 3D-native baselines [6, 9], and a gaze-free variant of our model. Only the gaze-prior baseline and gaze-guided XGLASS use REFLACX gaze information; BioMedCLIP, 3D Merlin + MLP, 3D U-Net, and XGLASS are diseaseconditioned, while XGLASS (w/out gaze), Vote-MI, Gaussian Centered, and Random are disease-agnostic. Gaze prior only ranks slices using only the disease-specific gazederived superior–inferior profile, without patient-specific CT input, serving as a control for static anatomical prior effects. Vote-MI was originally proposed for brain MRI; we follow its preprocessing and two-stage slice-selection procedure, but adapt it to thoracic CT by replacing the original feature extractor with MedVAE [40], a medical foundation model, and convert the selected representatives into continuous slice scores for fair metric-based comparison. BioMedCLIP is used as a slice-wise image–text retrieval baseline. For each CT-condition pair, we encode all axial slices with the BioMedCLIP image encoder and the queried disease text with the text encoder, and then rank slices by image–text cosine similarity to obtain a diseaseconditioned 240-dimensional relevance score vector. Full prompt and implementation details are provided in Supplementary Sec. G.1. For the 3D-native model comparisons, 3D Merlin + MLP uses a frozen Merlin volumetric encoder [6] followed by a disease-conditioned MLP trained to regress the same 240-dimensional slice-relevance target as XGLASS, ands 3D U-Net [9] uses a 3D U-Net encoder trained from scratch on ReXGroundingCT with a sliceregression head for the same targets. Further implementation details for both 3D baselines are provided in Supplementary Sec. G.2. XGLASS (w/o gaze cues) uses the identical backbone, projections, loss, and training as XGLASS but disables the gaze blend (α=0), isolating the architecture without disease-specific importance priors. Gaussian Centered places a Gaussian bump over slice indices with fractional width $\sigma _ { \mathrm { f r a c } } ,$ yielding higher scores near the midthorax, and Random draws an independent score for each slice from Uniform[0, 1] to reflect chance performance. All methods output a 240-dimensional score vector per study and are evaluated with the same metrics (ROC-AUC, Spearman’s $\rho ,$ Hit@K, and AR@K), as described in Section 4.1.

![](images/1284702b76c18fcf97f5b950aa9c7400289c4e5e0264b83c96a2a0e12e6b5f37.jpg)  
Figure 3. Slices until first Ground-Truth (GT). Boxplots show the distribution of slices required to reach the first GT slice (median, mean, 25–75% IQR, and 5–95% whiskers). XGLASS reaches a relevant slice earlier and with lower variability than the baselines.

Table 2. Disease-Specific Results. The “Overall” row corresponds to the “Ours (w/ gaze cues)” model from Table 1. Counts (N) are shown in parentheses. We report hit-rate metrics Hit@K, the averaged recall at 10 (AR@10), AUC and rank correlation (Spearman’s $\rho ) .$
<table><tr><td>Category (N)</td><td>Hit@1</td><td>Hit@3</td><td>Hit@5</td><td>Hit@10</td><td>AR@10</td><td>AUC</td><td>Spear.  $( \rho )$ </td></tr><tr><td>Atelectasis (39)</td><td>0.205</td><td>0.385</td><td>0.436</td><td>0.564</td><td>0.075</td><td>0.773</td><td>0.265</td></tr><tr><td>Consolidation (29)</td><td>0.483</td><td>0.586</td><td>0.655</td><td>0.759</td><td>0.114</td><td>0.810</td><td>0.386</td></tr><tr><td>Groundglass opacity (95)</td><td>0.284</td><td>0.400</td><td>0.516</td><td>0.674</td><td>0.114</td><td>0.728</td><td>0.272</td></tr><tr><td>High lung volume (29)</td><td>0.172</td><td>0.241</td><td>0.276</td><td>0.414</td><td>0.055</td><td>0.657</td><td>0.161</td></tr><tr><td>Lung nodule or mass (112)</td><td>0.107</td><td>0.205</td><td>0.286</td><td>0.446</td><td>0.074</td><td>0.722</td><td>0.129</td></tr><tr><td>Pleural abnormality (12)</td><td>0.500</td><td>0.583</td><td>0.667</td><td>0.667</td><td>0.139</td><td>0.779</td><td>0.345</td></tr><tr><td>Overall (316)</td><td>0.228</td><td>0.339</td><td>0.421</td><td>0.563</td><td>0.091</td><td>0.781</td><td>0.224</td></tr></table>

## 4.3. Overall Performance

Table 1 shows that gaze-guided XGLASS outperforms the 2D baselines across all reported metrics, with paired bootstrap intervals indicating significant gains in Hit@5, Hit@10, AUC, and Spearman’s $\rho .$ The native 3D baselines are competitive on global ranking metrics, with 3D U-Net achieving the highest AUC and Spearman’s $\rho$ point estimates and 3D Merlin achieving the highest Hit@1 and AR@10 point estimates. Despite operating on the full CT volume, the 3D baselines do not translate their small nonsignificant advantages in Hit@1 and global ranking into reliable slice retrieval and XGLASS remains significantly better on Hit@5 and Hit@10, the most relevant metrics for prioritizing a compact inspection set. Figure 3 further shows that XGLASS reaches the first diagnostically relevant slice earlier and more consistently than all baselines. In the coverage analysis (Figure 2), the 3D baselines become competitive, but together with their later first-hit distributions in Figure 3, it suggests that the 3D models recover dense relevant regions once the correct location is found, but are less reliable at finding that location in every case. A case-level Miss@10 analysis supports this interpretation: Merlin and 3D U-Net miss the early inspection budget in 190/316 and 202/316 cases, respectively, compared with only 138/316 missed cases for XGLASS, indicating that their slightly higher Hit@1, AR@10, and global-ranking point estimates do not translate into more reliable compact-budget retrieval. For clinical use cases, reliably surfacing at least one diagnostic slice within the inspected budget is more valuable than dense coverage only after successful localization, making XGLASS more suitable for practical slice selection, especially since it also requires 34.8× less inference time and 6.5× less GPU memory than Merlin, and 130.8× less time and 10.9× less memory than U-Net.

## 4.3.1. Disease-specific Performance

The per-category trends in Table 2 align with typical anatomical distributions. Consolidation and Pleural abnormalities are often gravity-dependent and cluster in posterior lower lobes, producing a relatively narrow and predictable band along the z-axis. This yields the highest AUC/ρ and strong Hit@K performance. Ground-glass opacities are frequently diffuse and extend across adjacent slices, which lifts mid- to higher-K retrieval (e.g., Hit@10) but softens global ranking metrics. In contrast, lung nodules/masses are small, sparse, and spatially unpredictable along the $z -$ axis, which reduces the chance that the very top-ranked slices contain a hit and depresses global rank agreement. Thus, lung nodules are relatively less suited to our approach due to their small size and variable location, characteristics that also make them challenging for slice-selection methods in general. When evaluated only on the lung-nodule subset, XGLASS achieves the strongest Hit@K and AUC, showing that despite performing worse than on other diseases, it still provides the best slice prioritization (Supplementary Sec. D). Taken together, these findings indicate that XGLASS learns disease-specific z-axis priors similar to radiologist behaviour, prioritizes anatomically informative regions, and therefore performs best for conditions with consistent spatial distributions such as consolidations.

![](images/ececac52387e796ba7b23f3bec121a22e80070463fa3a074e9ebe68ab7950f04.jpg)  
Figure 4. Downstream slice-wise VLM screening under XGLASS budgets. For each CT-condition pair, XGLASS selects the Top-K axial slices and the downstream model is evaluated only on this subset. Precision, recall, F1, and runtime are reported for MedGemma 4B, Lingshu 7B, and BioMedCLIP under the same single-GPU setting. Smaller XGLASS budgets consistently improve precision and F1, while reducing runtime by decreasing the number of inspected slices. Comparison to other slice selectors are in Supplementary Sec. G.4.

## 4.3.2. Computational Efficiency

XGLASS contains approximately 86M parameters and is trained on a single 2D projection per CT study. On an NVIDIA RTX 4080, one training epoch over 2,512 projections with batch size 16 takes 62.2 s, corresponding to roughly 40 samples/s, with approximately 3 GB peak GPU memory. Early stopping on validation loss consistently selects checkpoints after 6–8 epochs, yielding a complete training run in under 10 minutes. At inference time, XGLASS processes a full CT study, including intensity projection, in approximately 0.025 s per finding (0.150 s for all findings) with an estimated cost of 33.7 GFLOPs and 0.34 GB memory. In comparison, Vote-MI requires 7.51 s and approximately 116.33 TFLOPs per CT, while slice-wise BioMedCLIP requires 1.40 s and approximately 8.09 TFLOPs per CT. The 3D Merlin and U-Net baselines require 0.87 s and 3.27 s per finding, respectively, with 2.21 GB and 3.72 GB peak inference memory, making them 34.8×/130.8× slower and 6.5×/10.9× more memoryintensive than XGLASS. Overall, XGLASS is orders of magnitude cheaper than slice-wise 2D baselines and substantially faster and more memory-efficient than the 3D baselines, making local retraining on commodity GPUs feasible for institutions with private clinical data.

## 4.3.3. Downstream Slice-wise VLM Screening

We evaluate XGLASS as a disease-conditioned pre-filter for slice-wise VLM screening with 2D VLMs. For each CT-condition pair, XGLASS ranks all axial slices by relevance to the queried finding, and the downstream model is evaluated only on the Top-K selected slices. Metrics are computed over the inspected slices and compared against applying the same model to all 240 slices. Prompts, detailed inference settings and a matched-budget comparison to other slice selection methods are provided in Supplementary Sec. G. As shown in Fig. 4, smaller XGLASS budgets consistently improve precision across MedGemma 4B [35], Lingshu 7B [44], and BioMedCLIP [46], indicating that XGLASS determines disease-enriched slice subsets. This also improves F1 over inspected slices at small budgets: Top-30 improves F1 from 0.288 to 0.414 for MedGemma, from 0.327 to 0.441 for Lingshu, and from 0.235 to 0.316 for BioMedCLIP. Compared with other sliceselected subsets, XGLASS improves average Top-30 F1 over the strongest 2D selector from 0.338 to 0.390, while remaining competitive to the strongest 3D selector. Evaluating fewer slices also reduces downstream runtime, with Top-30 requiring only 12.5% of the slice-wise model calls compared with all-slice inference. This supports XGLASS as a fast pre-selection step that enriches disease-relevant slices before costly slice-wise VLM review, 2D dataset curation or similar applications.

## 4.4. Ablation Studies

We perform targeted ablations to isolate the effect of the main XGLASS design choices. Hyperparameter choices were made using the validation split resulting in ViT-B/16 backbone, AIP-wide CT projection, gaze blending strength of $\alpha { = } 0 . 7 5$ , and training using MSE. The corresponding validation results and selection criteria are reported in Supplementary Sec. E. All ablations tables modify only one factor at a time, and are reported on the held-out test set. Additional backbone, multi-channel-fusion, gaze-prior perturbations, loss ablation, and gaze prior-specificity ablations are provided in Supplementary Sec. F.

## 4.4.1. Weak Supervision.

Our best training target uses segmentation-derived slicearea scores, but XGLASS can also be trained from substantially weaker annotations. We evaluate three reducedsupervision targets: SE-Binary, which marks all slices between annotated start and end positions as positive; SE-Smooth, which additionally applies Gaussian smoothing to this interval; and Rel-Rank, which uses relative slice ranking instead of dense area scores. As shown in Table 3, reduced-supervision targets remain competitive, highlighting XGLASS’s ability to be trained from weak annotations.

Table 3. Supervision Target Ablation. Comparison of dense Slice-Area supervision with only start/end slice annotations.
<table><tr><td>Target</td><td>AUC</td><td>Spear.</td><td>Hit@1</td><td>Hit@5</td><td>Hit@10</td></tr><tr><td>SE-Binary</td><td>0.791</td><td>0.223</td><td>0.168</td><td>0.364</td><td>0.490</td></tr><tr><td>SE-Smooth</td><td>0.792</td><td>0.227</td><td>0.220</td><td>0.413</td><td>0.524</td></tr><tr><td>Rel-Rank</td><td>0.793</td><td>0.212</td><td>0.187</td><td>0.417</td><td>0.524</td></tr><tr><td>Slice-Area</td><td>0.781</td><td>0.224</td><td>0.228</td><td>0.421</td><td>0.563</td></tr></table>

## 4.4.2. Volume Projection Ablation

The results in Table 4 show a trade-off between global ranking and early retrieval utility. Beer–Lambert achieves the highest AUC (0.795) and Spearman’s ρ (0.232), and it edges out others at Hit@5 (0.427), indicating strong overall separability and mid-K retrieval. In contrast, AIP (wide) delivers the best utility-centric performance with the top Hit@1 (0.228), Hit@10 (0.563), and near-best Hit@5 (0.421), suggesting that preserving a broad intensity range helps rank diagnostically relevant slices.

Table 4. CT Projection Ablation. Comparison of Beer–Lambert and AIP projections with lung, soft-tissue, and wide CT window.
<table><tr><td>Projection</td><td colspan="3">AUC Spear. Hit@1 Hit@5 Hit@10</td></tr><tr><td>AIP (lung)</td><td>0.784 0.222</td><td>0.206 0.361</td><td>0.462</td></tr><tr><td>Beer-Lambert</td><td>0.795 0.232</td><td>0.196 0.427</td><td>0.535</td></tr><tr><td>AIP (soft tissue)</td><td>0.783 0.223</td><td>0.228 0.383</td><td>0.519</td></tr><tr><td>AIP (wide)</td><td>0.781 0.224</td><td>0.228</td><td>0.421 0.563</td></tr></table>

## 4.4.3. Gaze Fusion Ablation

As described in Section 3.1, XGLASS fuses the diseasespecific gaze prior with the coronal CT projection by alphablending the importance map into one RGB input channel. Table 5 evaluates both the blending strength and the injection channel. Adding gaze guidance consistently improves over the gaze-free setting $( \alpha { = } 0 )$ , confirming that the disease-conditioned prior provides useful localization information. While α=1.0 achieves the highest global AUC and Spearman’s $\rho ,$ but it partially suppresses CT intensity cues lowering retrieval metrics. In contrast, α=0.75 gives the best early retrieval performance, reaching Hit@10 of 0.563. Among single-channel injection variants, red-channel fusion performs best, outperforming green and blue injection on all metrics. Additional multi-RGB-channel fusion results are reported in the Supplementary Material in Sec. F.1.

Table 5. Gaze fusion ablation. Effect of blending opacity α and RGB injection channel for the disease-specific importance map.
<table><tr><td>Channel α</td><td colspan="3">AUC Spear. Hit@1 Hit@5 Hit@10</td></tr><tr><td>Red</td><td>0.00 0.768 0.212 0.25 50.787 0.221 0.50 0.785 0.220 1.00 0.797 0.227 0.75 0.781 0.224</td><td>0.199 0.342 0.187 0.392 0.196 0.418 0.199 0.415 0.228 0.421</td><td>0.468 0.506 0.509 0.528 0.563</td></tr><tr><td>Green Blue</td><td>0.75 0.754 0.212 0.75 0.761 0.209</td><td>0.193 0.345 0.190 0.418</td><td>0.459 0.519</td></tr></table>

## 5. Conclusion and Limitations

We presented XGLASS, a lightweight disease-conditioned model that transfers aggregated CXR gaze-language priors to CT for relevance-ordered axial slice retrieval. XGLASS significantly improves Hit@5 and Hit@10 over the strongest evaluated 2D and 3D baselines while remaining computationally efficient. Downstream experiments further show that XGLASS enriches compact diseaserelevant candidate sets, improves downstream classification performance, and reduces slice-wise model calls. Beyond slice selection, XGLASS illustrates how heterogeneous 2D supervision can help bridge the gap between 2D foundation models and 3D medical image analysis. Nevertheless, several limitations remain. Despite surveying publicly available CT datasets, ReXGroundingCT is the only dataset, that overlaps with the 14 REFLACX-derived disease priors, yielding only to one dataset and six shared thoracic findings for evaluation. Our downstream screening experiments provide an initial demonstration of XGLASS utility and not direct use cases; broader applications such as VQA, report generation, retrieval, and localization remain future work.

## Acknowledgments

This work was funded by the Deutsche Forschungsgemeinschaft (DFG, German Research Foundation) – GRK 3012 – 520750254 (https://kemai.uni-ulm.de/).

## References

[1] Josh Achiam, Steven Adler, Sandhini Agarwal, et al. Gpt-4 technical report. arXiv, 2023. 1

[2] Jean-Baptiste Alayrac, Jeff Donahue, Pauline Luc, et al. Flamingo: a visual language model for few-shot learning. Adv. Neural Inf. Process. Syst., 35:23716–23736, 2022. 1

[3] Mohammed Baharoon, Luyang Luo, Michael Moritz, et al. Rexgroundingct: A 3d chest ct dataset for segmentation of findings from free-text reports. arXiv, 2025. 3, 12

[4] Ricardo Bigolin Lanfredi, Joyce D Schroeder, and Tolga Tasdizen. Localization supervision of chest x-ray classifiers using label-specific eye-tracking annotation. Frontiers in radiology, 3:1088068, 2023. 2

[5] Ricardo Bigolin Lanfredi et al. Reflacx, a dataset of reports and eye-tracking data for localization of abnormalities in chest x-rays. Scientific data, 9(1):350, 2022. 2, 3, 11

[6] Louis Blankemeier, Joseph Paul Cohen, Ashwin Kumar, Dave Van Veen, Syed Jamal Safdar Gardezi, Magdalini Paschali, Zhihong Chen, Jean-Benoit Delbrouck, Eduardo Reis, Cesar Truyts, et al. Merlin: A vision language foundation model for 3d computed tomography. Research square, pages rs–3, 2024. 4, 5, 6, 15, 17

[7] Aurelia Bustos et al. Padchest: A large chest x-ray image dataset with multi-label annotated reports. Medical image analysis, 66:101797, 2020. 1

[8] Xi Chen, Xiao Wang, Soravit Changpinyo, et al. Pali: A jointly-scaled multilingual language-image model. arXiv, 2022. 1

[9] Ozg <sup>¨</sup> un C¸ ic¸ek, Ahmed Abdulkadir, Soeren S Lienkamp,¨ Thomas Brox, and Olaf Ronneberger. 3d u-net: learning dense volumetric segmentation from sparse annotation. In International conference on medical image computing and computer-assisted intervention, pages 424–432. Springer, 2016. 4, 5, 6, 15

[10] Alexey Dosovitskiy, Lucas Beyer, Alexander Kolesnikov, et al. An image is worth 16x16 words: Transformers for image recognition at scale. ICLR, 2021. 4, 14

[11] Rachel Lea Draelos, David Dov, Maciej A Mazurowski, Joseph Y Lo, Ricardo Henao, Geoffrey D Rubin, and Lawrence Carin. Machine-learning-based multiple abnormality prediction with large-scale chest computed tomography volumes. Medical image analysis, 67:101857, 2021. 2

[12] Trafton Drew, Melissa Le-Hoa Vo, Alex Olwal, Francine Jacobson, Steven E Seltzer, and Jeremy M Wolfe. Scanners and drillers: characterizing expert visual search through volumetric images. Journal ofvision, 13(10):3–3, 2013. 2

[13] Bradley Efron. Bootstrap methods: another look at the jackknife. In Breakthroughs in statistics: Methodology and distribution, pages 569–593. Springer, 1992. 5

[14] Zenzi M Griffin and Kathryn Bock. What the eyes say about speaking. Psychological science, 11(4):274–279, 2000. 2

[15] Xuehai He, Yichen Zhang, Luntian Mou, Eric Xing, and Pengtao Xie. Pathvqa: 30000+ questions for medical visual question answering. arXiv, 2020. 1

[16] Chihcheng Hsieh, Chun Ouyang, et al. Mimic-eye: Integrating mimic datasets with reflacx and eye gaze for multimodal deep learning applications. PhysioNet (version 1.0. 0), 2023. 2

[17] Jeremy Irvin, Pranav Rajpurkar, Michael Ko, et al. Chexpert: A large chest radiograph dataset with uncertainty labels and expert comparison. In Proc. of the AAAI, pages 590–597, 2019. 1

[18] Alistair EW Johnson, Tom J Pollard, Nathaniel R Greenbaum, et al. Mimic-cxr-jpg, a large publicly available database of labeled chest radiographs. arXiv, 2019. 1

[19] Alexandros Karargyris, Satyananda Kashyap, Ismini Lourentzou, Joy Wu, Matthew Tong, Arjun Sharma, Shafiq Abedin, David Beymer, Vandana Mukherjee, Elizabeth Krupinski, and Mehdi Moradi. Eye Gaze Data for Chest X-rays (version 1.0.0), 2020. 2

[20] Linda C Kelahan, Allan Fong, Joseph Blumenthal, Swaminathan Kandaswamy, Raj M Ratwani, and Ross W Filice. The radiologist’s gaze: mapping three-dimensional visual search in computed tomography of the abdomen and pelvis. Journal ofDigital Imaging, 32(2):234–240, 2019. 2

[21] Yunsoo Kim, Jinge Wu, Yusuf Abdulle, Yue Gao, and Honghan Wu. Enhancing human-computer interaction in chest x-ray analysis using vision and language model with eye gaze patterns. In International Conference on Medical Image Computing and Computer-Assisted Intervention, pages 184–194. Springer, 2024. 2

[22] Harold L Kundel, Calvin F Nodine, Elizabeth A Krupinski, and Claudia Mello-Thoms. Using gaze-tracking data and mixture distribution analysis to support a holistic model for the detection of cancers on mammograms. Academic radiol ogy, 15(7):881–886, 2008. 2

[23] Jason J Lau, Soumya Gayen, et al. A dataset of clinically generated visual questions and answers about radiology im ages. Scientific data, 5(1):1–10, 2018. 1

[24] Chunyuan Li, Cliff Wong, et al. Llava-med: Training a large language-and-vision assistant for biomedicine in one day. Adv. Neural Inf. Process. Syst., 36:28541–28564, 2023. 1

[25] Weixiong Lin, Ziheng Zhao, et al. Pmc-clip: Contrastive language-image pre-training using biomedical documents. In Proc. ofMICCAI, pages 525–536. Springer, 2023. 1

[26] Bo Liu, Li-Ming Zhan, Li Xu, Lin Ma, Yan Yang, and Xiao-Ming Wu. Slake: A semantically-labeled knowledge enhanced dataset for medical visual question answering. In Proc. IEEE Int. Symp. Biomed. Imaging (ISBI), pages 1650– 1654. IEEE, 2021. 1

[27] Ze Liu, Yutong Lin, Yue Cao, et al. Swin transformer: Hier archical vision transformer using shifted windows. In Proc. ofICCV, 2021. 14

[28] Chong Ma, Hanqi Jiang, Wenting Chen, Yiwei Li, Zihao Wu, Xiaowei Yu, Zhengliang Liu, Lei Guo, Dajiang Zhu, Tuo Zhang, et al. Eye-gaze guided multi-modal alignment for medical representation learning. Advances in Neural Infor mation Processing Systems, 37:6126–6153, 2024. 2

[29] Xueyan Mei, Zelong Liu, Philip M Robson, et al. Radimagenet: an open radiologic deep learning research dataset for effective transfer learning. Radiology: Artificial Intelligence, 4(5):e210315, 2022. 15

[30] Cristian Minoccheri, Emily Wittrup, Kayvan Najarian, and Ryan Stidham. Representation geometry shapes task performance in vision-language modeling for ct enterography. arXiv preprint arXiv:2604.13021, 2026. 18

[31] Andriy Myronenko, Dong Yang, Baris Turkbey, et al. Reasoning visual language model for chest x-ray analysis. arXiv, 2025. 1

[32] Trong Thang Pham, Akash Awasthi, Saba Khan, et al. Ctscangaze: A dataset and baselines for 3d volumetric scanpath modeling. In Proc. of ICCV, pages 21732–21743, 2025. 2, 11

[33] Alec Radford, Jong Wook Kim, Chris Hallacy, et al. Learning transferable visual models from natural language supervision. In Proc. ofICML, pages 8748–8763, 2021. 1

[34] Philip Sedgwick. Spearman’s rank correlation coefficient. Bmj, 349, 2014. 4

[35] Andrew Sellergren, Sahar Kazemzadeh, et al. Medgemma technical report. arXiv, 2025. 1, 7, 18

[36] Donald F Swinehart. The beer-lambert law. Journal ofchemical education, 39(7):333, 1962. 3

[37] Gemma Team, Aishwarya Kamath, Johan Ferret, et al. Gemma 3 technical report. arXiv, 2025. 1

[38] Hugo Touvron, Thibaut Lavril, Gautier Izacard, et al. Llama: Open and efficient foundation language models. arXiv, 2023. 1

[39] A Van der Gijp, CJ Ravesloot, H Jarodzka, et al. How visual search relates to visual diagnostic performance: a narrative systematic review of eye-tracking research in radiology. Advances in Health Sciences Education, 22(3):765–787, 2017. 2

[40] Maya Varma, Ashwin Kumar, Rogier van der Sluijs, et al. Medvae: Efficient automated interpretation of medical images with large-scale generalizable autoencoders, 2025. 6

[41] Yuli Wang, Yuwei Dai, Craig Jones, et al. Enhancing visionlanguage models for medical imaging: bridging the 3d gap with innovative slice selection. Adv. Neural Inf. Process. Syst., 37:99947–99964, 2024. 1, 2, 4, 5, 15

[42] Zifeng Wang, Zhenbang Wu, Dinesh Agarwal, and Jimeng Sun. Medclip: Contrastive learning from unpaired medical images and text. In Proc. ofEMNLP, page 3876, 2022. 1

[43] David S Wooding. Fixation maps: quantifying eyemovement traces. In Proceedings ofthe 2002 symposium on Eye tracking research & applications, pages 31–36, 2002. 2

[44] Weiwen Xu, Hou Pong Chan, Long Li, Mahani Aljunied, Ruifeng Yuan, Jianyu Wang, Chenghao Xiao, Guizhen Chen, Chaoqun Liu, Zhaodonghui Li, et al. Lingshu: A generalist foundation model for unified multimodal medical understanding and reasoning. arXiv preprint arXiv:2506.07044, 2025. 7, 18

[45] Luca Zedda et al. Radio dino: A foundation model for advanced radiomics and ai-driven medical imaging analysis. Comput. Biol. Med., 195:110583, 2025. 15

[46] Sheng Zhang, Yanbo Xu, Naoto Usuyama, Hanwen Xu, Jaspreet Bagga, Robert Tinn, Sam Preston, Rajesh Rao, Mu Wei, Naveen Valluri, et al. Biomedclip: a multimodal biomedical foundation model pretrained from fifteen million scientific image-text pairs. arXiv preprint arXiv:2303.00915, 2023. 1, 4, 5, 7, 15, 17

[47] Sheng Zhang, Yanbo Xu, Naoto Usuyama, et al. A multimodal biomedical foundation model trained from fifteen mil lion image–text pairs. NEJM AI, 2(1):AIoa2400640, 2025. 1

[48] Xiaoman Zhang, Chaoyi Wu, Ziheng Zhao, et al. Pmc-vqa: Visual instruction tuning for medical visual question answering. arXiv, 2023. 1

[49] Yuhao Zhang, Hang Jiang, et al. Contrastive learning of medical visual representations from paired images and text. In Proc. of MLHC, pages 2–25. PMLR, 2022. 1

# XGLASS: X-Ray, Gaze and Language Assisted Slice Sequencing for CT Volumes

Supplementary Material

## A. Qualitative Results

We complement the quantitative metrics with a qualitative analysis of slice selection across six chest CT findings. Figure 6 visualizes the axial slice indices (z) selected by a Random baseline, Vote-MI, and XGLASS. Across categories, XGLASS not only hits a ground-truth slice earlier, but also concentrates its selections within the annotated region, yielding denser and more contiguous coverage than competing methods. These examples illustrate that our model prioritizes clinically meaningful slices rather than scattered or isolated hits, aligning with the intended notion of regionlevel localization.

## B. Disease-Conditioned Importance Maps

## B.1. Text–Gaze Alignment

To construct disease-conditioned 2D importance maps from REFLACX [5], we jointly process the word-level transcripts and synchronized gaze recordings. For each target disease, we first detect positive (non-negated) mentions in the transcript using label-specific regular expressions together with clause-level negation handling, as described in Section 3.1. Each detected mention defines a speech window $[ t _ { \mathrm { s t a r t } } - \tau _ { \mathrm { p r e } } , t _ { \mathrm { e n d } } ]$ with a short pre-speech buffer $\tau _ { \mathrm { p r e } }$ of 2 seconds to account for anticipatory gaze. We then collect all gaze samples whose timestamps fall inside this window and project their screen coordinates onto the corresponding chest X-ray. Following the REFLACX protocol, we normalize coordinates to the chest bounding box and discard samples that fall outside this region (e.g., blink artefacts or fixations off the image). Figure 5 shows qualitative examples of the resulting text–gaze alignment for the six disease categories used in this work.

## B.2. Extracted Gaze Importance Maps

Using label-specific regular expressions and clause-level negation on the REFLACX reports, we defined 14 non–device pathology categories and, for each of them, extracted a disease-specific importance map by aggregating gaze samples associated with positive mentions across all matching sessions. Figure 7 summarizes how many gaze–report pairs contribute to each category.

## B.3. AP–PA Robustness of CXR Gaze Priors

Because REFLACX is derived from frontal chest radiographs, one potential concern is that the resulting gaze priors may depend on acquisition view. AP and PA radiographs can differ in apparent anatomy due to projection geometry, magnification, and patient positioning. We therefore stratify REFLACX sessions by MIMIC-CXR view position and construct separate AP-only and PA-only diseaseconditioned gaze maps. Figures 12 and 13 compare the resulting maps for all six target findings. Despite viewdependent differences in the 2D heatmaps, the superior– inferior profiles remain similar across AP and PA views, with correlations ranging from r=0.53 to 0.96. Since this axis is the one used for axial slice ordering, these results support combining AP and PA sessions to obtain more stable disease-conditioned gaze priors for XGLASS.

## B.4. CXR–CT Gaze Alignment

To further assess whether CXR-derived gaze priors capture transferable anatomical search patterns, we compare them with CT-ScanGaze [32], a CT eye-tracking dataset with timestamped dictation. CT-ScanGaze contains 57 unique CT cases, but several scans have inconsistent orientation metadata or cover non-chest anatomy. We therefore performed visual quality control and excluded cases with incorrect projection geometry or non-thoracic coverage. For the remaining cases, we constructed disease-conditioned CT gaze maps analogously to the REFLACX priors: report sentences were matched to gaze by timestamp, using the same 2 s pre-speech buffer, and fixations were aggregated for the corresponding finding. These CT-ScanGaze maps provide insight into CT gaze patterns but are substantially sparser than REFLACX, with 6–23 CT cases and 244–1,215 fixations per finding (mean: 12.8 cases and 731 fixations). Figures 14–15 compare the resulting CT-ScanGaze maps with the REFLACX CXR priors. The 2D heatmaps differ visibly, reflecting different reading behavior in CT: radiologists often fixate a local region and scroll through adjacent axial slices, rather than scanning broadly across the frontal image as in CXR. However, the superior–inferior gaze-density profiles remain aligned across modalities, with correlations ranging from 0.65 to 0.96 for the analyzed findings. This indicates that although CT and CXR differ in traversal order and fixation dynamics, radiologists still attend to similar disease-relevant anatomical regions. Thus, the CXRderived gaze priors encode non-arbitrary, disease-specific search patterns that can be transferred to CT and used as additional input for axial slice selection.

## C. CT Dataset and Volume Projections

## C.1. Dataset Selection

To evaluate how effectively our disease-specific eyetracking priors transfer from CXR to CT, we require a CT dataset that (i) provides report text and findings aligned with individual volumes, and (ii) contains findings that can be reliably matched to our REFLACX-derived diseasespecific importance maps. As discussed in Section 3.2, ReXGroundingCT [3] is, to our knowledge, the only large-scale CT dataset that satisfies both criteria. Because this dataset will also be used to train XGLASS, each disease category must be supported by a sufficient number of CT studies. We therefore exclude any finding category with fewer than 100 matching CTs. As shown in Figure 7 and Figure 8, only six of the 14 REFLACX-defined categories reach this threshold in our standardized 240-slice

![](images/6ebf6b42f68e9ce8ccf74768e7a939943d79788fb66079a2712fb5b9bcf493cc.jpg)  
Figure 5. Text–gaze alignment for disease phrases. Examples of text–gaze alignment used to construct disease-conditioned importance maps for the six target findings: Atelectasis, Consolidation, Groundglass opacity, Lung nodule or mass, High lung volume, and Pleural abnormality. Each panel shows a REFLACX chest X-ray with the chest bounding box (green dashed line), disease-specific annotation ellipses (black), and gaze trajectories for the words in the corresponding disease phrase (color-coded by word with the pre-speech buffer in orange). Fixations that fall outside the chest bounding box, visible for the High lung volume and Groundglass opacity examples, are removed during preprocessing as described in Section 3.1.

ReXGroundingCT subset introduced in Section 3.2, and we consequently restrict our experiments to these six categories.

## C.2. Training Dataset and Splits

We use an 80/10/10 CT-volume-level train–validation– test split of the standardized 240-slice ReXGroundingCT subset, with no volume overlap between partitions and all findings from each CT retained in the same split. This yields 2,514/306/316 CT–finding cases for training/validation/testing, respectively, totaling 3,136 cases across six categories: Lung nodule or mass (1,096), Groundglass opacity (925), Atelectasis (412), High lung volume (301), Consolidation (284), and Pleural abnormality (118), as shown in Fig. 8. The disease distribution in our ReXGroundingCT subset is markedly imbalanced across our six categories. However the best performing model did not apply class-balancing or reweighting (e.g., oversampling, inverse-frequency sampling, focal/weighted losses), because the most prevalent class, lung nodule or mass, is also the hardest to localize due to its small size, variable location, and sparse slice extent. This choice is supported by the disease-specific results in Table 2 of the main paper, where XGLASS improves across all disease categories, and by the loss-function ablation in Sec. F.5, where reweighted objectives do not improve small-finding and overall retrieval performance.

![](images/7fd638ad2bea5470673a58973bfc3419682071db8326970649692ec1be5f828c.jpg)  
Figure 6. Qualitative Results. Slice retrieval across six chest CT findings: Atelectasis, Consolidation, Ground-glass Opacity, Lung Nodule/Mass, High Lung Volume, and Pleural Abnormality. For each category, we show the axial slice indices (z) predicted by a Random baseline, Vote-MI, and our XGLASS selector. XGLASS not only identifies a ground-truth slice (green border) earlier, but also provides broader coverage of the ground-truth region by retrieving multiple slices that fall within it, rather than a single correct slice.

![](images/33fa462d1d290b193f1164ae906cd019aff17ca36600b4fcc675cc2d252266fe.jpg)  
Figure 7. REFLACX sessions per disease category. Number of REFLACX sessions contributing to each disease-conditioned gaze heatmap. Bars show the number of matching gaze–report pairs for each category defined in REFLACX. Categories highlighted in blue correspond to the six findings used in XGLASS, whereas orange bars denote additional categories for which we obtained gaze heatmaps but did not use them due to the lack of corresponding findings in the CT dataset.

![](images/ba8bc5bd50ed541fdaa8ec899e97644c8fbc98dd0417db09aa4170a9861a0b51.jpg)  
Figure 8. CT disease category distribution. Counts of CT– finding cases for the six ReXGroundingCT categories selected for XGLASS, stacked by train/validation/test split. These categories were retained because they overlap with the REFLACX-derived gaze priors and provide sufficient CT cases for training and evaluation.

## C.3. CT Volume Projections

For XGLASS, each 3D chest CT volume is converted into a 2D coronal intensity projection that lives in the same domain as our disease-conditioned importance maps, as explained in Section 3.2. Formally, we project along the anterior–posterior axis and aggregate attenuation values along each ray using an operator A (Beer–Lambert or averageintensity), yielding a coronal image $P _ { A } ( u , v )$ for each volume. In the main experiments we focus on two projection families: (i) a physically motivated Beer-Lambert projection that mimics x-ray transmission, and (ii) averageintensity projections (AIP) combined with standard CT windowings. For AIP we consider three clinical window settings: a lung window, a soft-tissue window, and a wide window spanning a broad Hounsfield range. As discussed in Section 3.2 and ablated in Section 4.4.2, Beer-Lambert provides slightly better global ranking (AUC, Spearman), whereas AIP with a wide window offers the best early retrieval utility (Hit@K). Figure 9 illustrates how a single CT scan appears under these different projection choices, highlighting how windowing and projection type emphasize complementary structures (e.g., parenchyma in the lung window vs. mediastinum and pleural surfaces under broader windows).

## C.4. Gaze-Prior Blending Strength

In XGLASS, the 2D CT projection and the disease-specific REFLACX gaze importance map are combined by injecting the normalized gaze heatmap $H _ { d } ( u , v )$ into the red channel inside the chest bounding box using a blending coefficient $\alpha \in [ 0 , 1 ]$ . Here, $\alpha = 0$ corresponds to the CT-only input, whereas larger values increase the contribution of the disease-conditioned gaze prior in the fused representation. Figure 11 illustrates the qualitative effect of varying α for representative CT studies. The quantitative test-set ablation of α is reported in Sec. 4.4.3 of the main paper. The final value α=0.75 was selected on the validation split, as described in Sec. E.

## D. Lung Nodule Baseline Comparison

Lung nodules/masses are the hardest evaluated category because they are small, sparse, and spatially variable. However, this relative weakness does not indicate near-random performance. Across all 112 nodule cases, XGLASS achieves the best Hit@K and AUC among all 2D and 3D baselines, showing that it more reliably retrieves at least one relevant slice early. The 3D U-Net obtains slightly higher AR@10 and Spearman’s ρ, but performs substantially worse on Hit@5, Hit@10, and Hit@20 (Table 6).

## E. Hyperparameter Selection

We selected the model’s hyperparameters such as the visual backbone, CT projection, gaze-blending coefficient α, loss function, and gaze-injection channel using the validation split. All optimization hyperparameters were kept fixed across configurations: AdamW with learning rates of $1 \times 1 0 ^ { - 4 }$ for the regression head and $1 \times 1 0 ^ { - 5 }$ for the ViT backbone, and a batch size of 16. Since XGLASS targets slice retrieval across different inspection budgets, selection was based on the mean of Hit@5, Hit@10, and Hit@20, denoted as Hit in Table 7. We exclude Hit@1 and Hit@3 from the selection criterion because these very small retrieval budgets are more sensitive to single-slice rank changes. When Hit was tied, AUC and Spearman’s ρ were considered as secondary ranking criteria. ViT-B/16, AIP-wide, and $\alpha { = } 0 . 7 5$ achieved the highest Hit within their respective ablations. For the gaze-injection channel, red and blue achieve the same Hit. Red yields a slightly higher Spearman correlation (0.250 vs. 0.243) while differing only marginally in AUC (0.796 vs. 0.801). Given the similar slice retrieval performance and the ranking objective of XGLASS, we therefore selected red-channel injection for the final model. In the loss-function ablation, unweighted MSE achieves a higher Hit than binary focal loss and lesion-balanced MSE, while preserving competitive AUC and Spearman correlation; we therefore selected the unweighted MSE as the final training objective. The resulting configuration, ViT-B/16, AIP-wide, α=0.75, redchannel gaze injection, and unweighted MSE, was fixed and used for the test-set ablations in Sec. 4.4 and Sec. F.

## F. Extended Ablations

## F.1. Gaze-Injection Channel Combinations

We additionally evaluate every RGB-channel combination for injecting the gaze prior at the selected blending strength α=0.75 on the held-out test set. Table 8 shows that the two strongest configurations both include the red channel: magenta (R+B) performs best for AUC, Spearman correlation, Hit@1, and Hit@5, while red-only achieves similar overall performance and the highest Hit@10 and Hit@20. Red-only was selected on the validation split, where it tied with blue for Hit and achieved the higher Spearman correlation. The test-set results further show that red-only remains particularly strong at the larger retrieval budgets.

## F.2. XGLASS Backbone

Different backbones encode different inductive biases (e.g., local vs. global receptive fields, domain pretraining). We test a ViT-B/16 backbone (patch size 16, input 224×224) with AugReg ImageNet-21k pretraining [10], a Swin transformer with shifted windows (patch size 4, window 7) at 224×224, pretrained on ImageNet-22k [27], and

![](images/18dbc37c0e18c3b5e5243f6ae1c79521338a0cbc408506300cb1beb755a815a8.jpg)  
Figure 9. CT volume projections in XGLASS. Example coronal slice projections of the same chest CT under different projection methods and windowing settings. AIP (lung) emphasizes parenchymal findings and saturates mediastinal structures, AIP (soft) suppresses low-attenuation lung parenchyma, so bony structures dominate the appearance, AIP (wide) exposes a broader range of intensities, and the Beer–Lambert projection yields a more radiograph-like appearance with stronger contrast in high-attenuation structures.

Table 6. Lung-nodule-specific baseline comparison. Results on all 112 lung nodule/mass test cases, the most spatially sparse and variable finding category. XGLASS achieves the strongest Hit@K and AUC compared to the other 2D and 3D baselines, indicating more reliable early retrieval despite the difficulty of this disease.
<table><tr><td>Method</td><td>Hit@1</td><td>Hit@3</td><td>Hit@5</td><td>Hit@10</td><td>Hit@20</td><td>AR@10</td><td>AUC</td><td>Spearman (ρ)</td></tr><tr><td>Random</td><td>0.080</td><td>0.125</td><td>0.179</td><td>0.286</td><td>0.446</td><td>0.038</td><td>0.491</td><td>-0.007</td></tr><tr><td>Vote-MI [41]</td><td>0.018</td><td>0.107</td><td>0.179</td><td>0.313</td><td>0.545</td><td>0.035</td><td>0.478</td><td>-0.017</td></tr><tr><td>BioMedCLIP [46]</td><td>0.080</td><td>0.143</td><td>0.214</td><td>0.286</td><td>0.455</td><td>0.062</td><td>0.585</td><td>0.093</td></tr><tr><td>3D Merlin + MLP [6]</td><td>0.080</td><td>0.107</td><td>0.143</td><td>0.259</td><td>0.420</td><td>0.057</td><td>0.688</td><td>0.125</td></tr><tr><td>3D U-Net [9]</td><td>0.080</td><td>0.143</td><td>0.161</td><td>0.232</td><td>0.384</td><td>0.079</td><td>0.698</td><td>0.132</td></tr><tr><td>XGLASS</td><td>0.107</td><td>0.205</td><td>0.286</td><td>0.446</td><td>0.607</td><td>0.074</td><td>0.722</td><td>0.129</td></tr></table>

RadioDino-s8 [45], a DINO-style self-supervised ViT pretrained on RadImageNet [29] as a domain-adapted medical backbone. Holding inputs fixed, this ablation isolates how architectural choices and pretraining affect the model’s ability to learn disease-aware slice-importance scores from our importance map enhanced CT intensity projections. The results in Table 9 show that while Swin-Tiny attains the highest AUC and RadioDino-s8 slightly leads on Spearman’s $\rho ,$ the validation-selected ViT-B/16 backbone delivers the best diagnostically relevant slice retrieval performance, with higher Hit@K across K=1, 5, 10.

## F.3. Gaze-Only and CT-Derived Prior Ablations

We further analyze the role of disease-conditioned priors in XGLASS by comparing transferred CXR gaze priors with empirical CT abnormality-location priors. First, we evaluate the CXR gaze prior alone by averaging each diseasespecific gaze map along the left–right axis, interpolating the resulting superior–inferior profile to 240 axial positions, and ranking slices by profile intensity. Second, we construct an empirical CT location prior using only train+val labels by averaging slice-presence vectors for each disease category and applying the resulting fixed category-specific prior to all corresponding test cases. Third, we replace the gaze map in XGLASS with this CT-derived prior converted into a synthetic heatmap and fuse it with the CT projection in the same way as the gaze heatmap. Since this CT prior is computed from train+val ground-truth slice-presence targets from the same CT dataset distribution, it represents an empirical average lesion-location prior for each abnormality. In contrast, the gaze prior is transferred from CXR eye-tracking and does not use CT lesion-location statistics; instead, it captures radiologist visual search patterns that include both observed abnormality regions and plausible candidate regions inspected for each finding. In contrast, the CT-derived prior is less reliable when abnormalities occur in rare locations, whereas the broader radiologist search patterns encoded by the gaze prior can still overlap these regions, supporting the use of gaze over CT derived location priors. Table 10 shows that CT location statistics are informative: the CT location prior alone achieves the best Hit@1 and Spearman correlation. However, the full gaze-guided XGLASS model achieves the best AUC and performs substantially better for utility-centric top-K retrieval metrics, improving Hit@10 from 0.453 to 0.563 and Hit@20 from 0.522 to 0.693 compared with the CT location prior alone.

Table 7. Validation-based hyperparameter selection. All values are computed on the validation split. Hit denotes the mean of Hit@5, Hit@10, and Hit@20. Gray rows indicate the settings selected for the final model; bold values indicate the best performance within each ablation.
<table><tr><td>Setting</td><td>Hit@5</td><td>Hit@10</td><td>Hit@20</td><td>Hit</td><td>AUC</td><td>ρ</td></tr><tr><td colspan="7">Backbone</td></tr><tr><td>ViT-B/16</td><td>0.441</td><td>0.559</td><td>0.709</td><td>0.570</td><td>0.796</td><td>0.250</td></tr><tr><td>RadioDino-s8</td><td>0.435</td><td>0.539</td><td>0.644</td><td>0.539</td><td>0.804</td><td>0.256</td></tr><tr><td>Swin-Tiny</td><td>0.405</td><td>0.464</td><td>0.578</td><td>0.482</td><td>0.798</td><td>0.244</td></tr><tr><td colspan="7">CT projection</td></tr><tr><td>AIP-wide</td><td>0.441</td><td>0.559</td><td>0.709</td><td>0.570</td><td>0.796</td><td>0.250</td></tr><tr><td>AIP-soft</td><td>0.395</td><td>0.572</td><td>0.654</td><td>0.540</td><td>0.798</td><td>0.245</td></tr><tr><td>AIP-lung</td><td>0.431</td><td>0.510</td><td>0.618</td><td>0.520</td><td>0.794</td><td>0.242</td></tr><tr><td>Beer-Lambert</td><td>0.428</td><td>0.549</td><td>0.647</td><td>0.541</td><td>0.796</td><td>0.246</td></tr><tr><td colspan="7">Gaze blending coefficient α</td></tr><tr><td>0.00</td><td>0.395</td><td>0.500</td><td>0.578</td><td>0.491</td><td>0.771</td><td>0.236</td></tr><tr><td>0.25</td><td>0.422</td><td>0.549</td><td>0.641</td><td>0.537</td><td>0.802</td><td>0.246</td></tr><tr><td>0.50</td><td>0.412</td><td>0.539</td><td>0.693</td><td>0.548</td><td>0.797</td><td>0.248</td></tr><tr><td>0.75</td><td>0.441</td><td>0.559</td><td>0.709</td><td>0.570</td><td>0.796</td><td>0.250</td></tr><tr><td>1.00</td><td>0.428</td><td>0.513</td><td>0.657</td><td>0.533</td><td>0.803</td><td>0.248</td></tr><tr><td colspan="7">Gaze-injection channel</td></tr><tr><td>R</td><td>0.441</td><td>0.559</td><td>0.709</td><td>0.570</td><td>0.796</td><td>0.250</td></tr><tr><td>B</td><td>0.451</td><td>0.575</td><td>0.683</td><td>0.570</td><td>0.801</td><td>0.243</td></tr><tr><td>R+B</td><td>0.412</td><td>0.539</td><td>0.683</td><td>0.545</td><td>0.804</td><td>0.248</td></tr><tr><td>G</td><td>0.415</td><td>0.536</td><td>0.650</td><td>0.534</td><td>0.806</td><td>0.243</td></tr><tr><td>G+B</td><td>0.422</td><td>0.510</td><td>0.650</td><td>0.527</td><td>0.798</td><td>0.241</td></tr><tr><td>R+G</td><td>0.435</td><td>0.520</td><td>0.624</td><td>0.526</td><td>0.800</td><td>0.248</td></tr><tr><td colspan="7">Loss function</td></tr><tr><td>Area MSE</td><td>0.441</td><td>0.559</td><td>0.709</td><td>0.570</td><td>0.796</td><td>0.249</td></tr><tr><td>Binary focal</td><td>0.418</td><td>0.582</td><td>0.683</td><td>0.561</td><td>0.804</td><td>0.250</td></tr><tr><td>Slice-bal. MSE</td><td>0.389</td><td>0.536</td><td>0.690</td><td>0.538</td><td>0.726</td><td>0.229</td></tr><tr><td>Size-wt. MSE</td><td>0.405</td><td>0.520</td><td>0.670</td><td>0.532</td><td>0.790</td><td>0.235</td></tr></table>

Table 8. Channel Injection Strategy Ablation. Gaze-prior injection into individual RGB channels and their combinations.
<table><tr><td>Channels</td><td>AUC</td><td>Spearman</td><td>Hit@1</td><td>Hit@5</td><td>Hit@10</td><td>Hit@20</td></tr><tr><td>R</td><td>0.781</td><td>0.224</td><td>0.228</td><td>0.421</td><td>0.563</td><td>0.693</td></tr><tr><td>G</td><td>0.754</td><td>0.212</td><td>0.193</td><td>0.345</td><td>0.459</td><td>0.582</td></tr><tr><td>B</td><td>0.761</td><td>0.209</td><td>0.190</td><td>0.418</td><td>0.519</td><td>0.655</td></tr><tr><td>R+G</td><td>0.774</td><td>0.220</td><td>0.184</td><td>0.383</td><td>0.503</td><td>0.639</td></tr><tr><td>R+B</td><td>0.784</td><td>0.226</td><td>0.241</td><td>0.443</td><td>0.547</td><td>0.661</td></tr><tr><td>G+B</td><td>0.749</td><td>0.212</td><td>0.222</td><td>0.367</td><td>0.468</td><td>0.617</td></tr><tr><td>RGB</td><td>0.761</td><td>0.196</td><td>0.192</td><td>0.364</td><td>0.465</td><td>0.594</td></tr></table>

Injecting the CT-derived prior into XGLASS also improves over CT-only input at Hit@10 (0.535 vs. 0.468). Nevertheless, gaze-guided XGLASS remains stronger at Hit@10 and Hit@20, indicating that the transferred radiologist gaze prior provides complementary information beyond average CT lesion-frequency statistics, which improves the slice selection performance especially for higher K. We additionally stratify test samples by their typicality under the train+val CT location prior, defined as the average CTprior score over the ground-truth positive slices of each test sample. In low-typicality cases, where the empirica CT prior assigns low probability to the true lesion region, the CT location prior drops sharply to Hit@10 of 0.151. In contrast, the CXR gaze prior alone remains stronger at Hit@10 of 0.330, while full XGLASS reaches Hit@10 of 0.358 and Hit@20 of 0.528. This suggests that empirical CT priors primarily capture common dataset-specific lesion locations, whereas radiologist gaze priors cover broader disease-specific search regions. Combined with patientspecific CT projection evidence, this leads to more robust disease-conditioned slice prioritization than static location priors alone.

Table 9. Backbone Ablation. Under AIP with wide window projection and α=0.75.
<table><tr><td>Backbone</td><td>AUC Spear. Hit@1 Hit@5 Hit@10</td><td></td><td></td><td></td></tr><tr><td>RadioDino-s8</td><td>0.777 0.227</td><td>0.180</td><td>0.361</td><td>0.487</td></tr><tr><td>Swin-Tiny</td><td>0.784 0.222</td><td>0.203</td><td>0.358</td><td>0.472</td></tr><tr><td>ViT-B/16</td><td>0.781 0.224</td><td>0.228</td><td>0.421</td><td>0.563</td></tr></table>

Table 10. Prior-Only and Prior-Injection Ablations. Compari son of static CXR gaze priors, empirical CT location priors, CTonly XGLASS, CT-prior injection, and full gaze-guided XGLASS.
<table><tr><td>Method</td><td>AUC</td><td>Spear.</td><td>Hit@1</td><td>Hit@5</td><td>Hit@10</td><td>Hit@20</td></tr><tr><td>Gaze prior only</td><td>0.638</td><td>0.163</td><td>0.158</td><td>0.209</td><td>0.263</td><td>0.358</td></tr><tr><td>CT location prior only</td><td>0.734</td><td>0.229</td><td>0.250</td><td>0.329</td><td>0.453</td><td>0.522</td></tr><tr><td>XGLASS w/o gaze cues</td><td>0.768</td><td>0.212</td><td>0.199</td><td>0.342</td><td>0.468</td><td>0.619</td></tr><tr><td>XGLASS + CT loc. prior</td><td>0.722</td><td>0.199</td><td>0.199</td><td>0.424</td><td>0.535</td><td>0.658</td></tr><tr><td>XGLASS w/ gaze cues</td><td>0.781</td><td>0.224</td><td>0.228</td><td>0.421</td><td>0.563</td><td>0.693</td></tr></table>

## F.4. Robustness to Gaze-Prior Perturbations

To test whether XGLASS depends on exact pixel-level alignment of the transferred gaze prior, we first measured coarse anatomical alignment between CXR-derived priors and CT projections using lung and heart layouts, obtaining a lungs+heart centroid error of 16.2 px and an average symmetric surface distance of 11.6 px; for lungs alone, the corresponding errors were 8.8 px and 8.0 px. We then perturb the disease-conditioned heatmap at test time while keeping the CT projection unchanged. Table 11 shows that random spatial shifts, additive noise, and left–right heatmap flips only mildly affect performance, supporting that XGLASS uses the gaze map as a broad anatomical search prior, that is robust to minor perturbations, rather than an exact lesionlocation mask.

Table 11. Robustness to Gaze-Prior Perturbations. Test-time perturbations of the disease-conditioned gaze map.
<table><tr><td>Condition</td><td>AUC</td><td>Spear.</td><td>Hit@1</td><td>Hit@5</td><td>Hit@10</td></tr><tr><td>Baseline</td><td>0.781</td><td>0.224</td><td>0.228</td><td>0.421</td><td>0.563</td></tr><tr><td>Random shift ±16 px</td><td>0.778</td><td>0.221</td><td>0.206</td><td>0.405</td><td>0.528</td></tr><tr><td>Noise σ=0.1</td><td>0.775</td><td>0.220</td><td>0.215</td><td>0.427</td><td>0.557</td></tr><tr><td>Left-right flip</td><td>0.786</td><td>0.224</td><td>0.203</td><td>0.396</td><td>0.563</td></tr></table>

## F.5. Loss Function Ablation

Abnormalities vary substantially in spatial extent, e.g., consolidations often cover larger regions than lung nodules. We therefore compare four training objectives while keeping all other settings fixed: MSE regresses the normalized slicearea target, Focal treats each slice as positive or negative and emphasizes difficult cases, Slice-bal. MSE gives equal aggregate weight to positive and negative slices within each CT, preventing scans with many positive slices from dominating the loss, and Size-wt. MSE inversely weights positive-slice errors by the total lesion size in each CT, reducing the contribution of spatially extensive lesions. Table 12 shows that MSE gives the strongest overall retrievalranking trade-off, with the best Hit@1 and Hit@10 and tied-best Hit@5. We further stratify the same ablation by finding extent. Test cases are split into small, medium, and large tertiles by total ground-truth slice-area mass, corresponding to average extents of 5.6, 14.7, and 52.4 positive slices, respectively.

Table 12. Loss-function ablation. We compare Area MSE with binary and lesion-size-aware alternatives.
<table><tr><td>Loss</td><td>AUC Spear. Hit@1 Hit@5 Hit@10</td><td></td></tr><tr><td>MSE 0.781 0.224</td><td>0.228 0.421</td><td>0.563</td></tr><tr><td>Focal 0.790</td><td>0.218 0.180 0.392</td><td>0.516</td></tr><tr><td>Slice-bal. MSE 0.709</td><td>0.202 0.206 0.421</td><td>0.519</td></tr><tr><td>Size-wt. MSE 0.785</td><td>0.221 0.215 0.411</td><td>0.519</td></tr></table>

Table 13 shows that MSE is strongest for small sparse findings across retrieval metrics, whereas Focal and Slice-bal. MSE mainly improve medium or large findings. Although the reweighted objectives were intended to improve smalllesion retrieval, they appear to amplify the instability of very sparse positive targets: for small findings, only a few slices are labeled relevant and the corresponding projection evidence is often weak, so upweighting these slices can emphasize case-specific noise. In contrast, MSE preserves the continuous slice-area target and provides a smoother ranking signal, making it the most stable default objective.

Table 13. Loss retrieval metrics by finding size. Test cases are grouped into tertiles by total ground-truth slice-area mass. We report retrieval metrics within each size group.
<table><tr><td>Loss</td><td>Size</td><td>Hit@1</td><td>Hit@3</td><td>Hit@5</td><td>Hit@10</td><td>AR@10</td></tr><tr><td></td><td>Small</td><td>0.057</td><td>0.123</td><td>0.189</td><td>0.340</td><td>0.103</td></tr><tr><td>MSE</td><td>Medium</td><td>0.114</td><td>0.276</td><td>0.352</td><td>0.543</td><td>0.079</td></tr><tr><td></td><td>Large</td><td>0.514</td><td>0.619</td><td>0.724</td><td>0.810</td><td>0.090</td></tr><tr><td>Focal</td><td>Small</td><td>0.019</td><td>0.075</td><td>0.104</td><td>0.226</td><td>0.065</td></tr><tr><td></td><td>Medium</td><td>0.162</td><td>0.352</td><td>0.476</td><td>0.581</td><td>0.108</td></tr><tr><td></td><td>Large</td><td>0.438</td><td>0.600</td><td>0.667</td><td>0.752</td><td>0.090</td></tr><tr><td></td><td>Small</td><td>0.047</td><td>0.113</td><td>0.142</td><td>0.208</td><td>0.073</td></tr><tr><td>Slice-bal. MSE Medium</td><td></td><td>0.105</td><td>0.257</td><td>0.352</td><td>0.486</td><td>0.077</td></tr><tr><td></td><td>Large</td><td>0.467</td><td>0.657</td><td>0.771</td><td>0.867</td><td>0.087</td></tr><tr><td></td><td>Small</td><td>0.028</td><td>0.104</td><td>0.151</td><td>0.264</td><td>0.085</td></tr><tr><td>Size-wt. MSE</td><td>Medium</td><td>0.114</td><td>0.257</td><td>0.362</td><td>0.505</td><td>0.087</td></tr><tr><td></td><td>Large</td><td>0.505</td><td>0.648</td><td>0.724</td><td>0.790</td><td>0.089</td></tr></table>

## G. Additional Implementation Details

## G.1. BioMedCLIP Slice Retrieval

We use BioMedCLIP [46] as a zero-shot slice-wise image– text retrieval baseline. We follow its official evaluation pipeline by rendering each axial CT slice with a wide window (center 0 HU, width 2000 HU), converted to RGB, and encoded independently. Finding names are normalized where appropriate (e.g., “lung nodule or mass” to “pulmonary nodule” and “groundglass opacity” to “groundglass opacity”). To reduce sensitivity to a single prompt formulation, we average the cosine similarities obtained from the following five templates, where {finding} denotes the normalized finding name:

1. The radiological finding is   
{finding}.

2. This axial CT image shows {finding}.

3. The CT slice is positive for   
{finding}.

4. There is {finding} in this CT slice.

5. A finding of {finding} is present on   
this CT image.

Slices are ranked in descending order of the ensembleaveraged similarity. For the downstream binary-screening analysis, one global similarity threshold (0.3751) is selected on the all-slice BioMedCLIP baseline by maximizing F1 and then applied unchanged to every XGLASS Top-K subset.

## G.2. 3D Baseline Implementation Details

For the 3D Merlin + MLP baseline, we follow the CT preparation used by Merlin [6]: full CT volumes are resampled to 1.5×1.5×3.0 mm spacing and resized to 160×224×224 before encoding. The Merlin encoder is kept frozen and used to precompute one 2048-dimensional volume embedding per CT. We then train a disease-conditioned MLP on ReXGroundingCT to regress the same 240-dimensional slice-relevance target as XGLASS. The MLP uses a 32- dimensional disease embedding concatenated with the Merlin feature, followed by two 512-unit hidden layers with ReLU activations and dropout 0.1, and a final linear layer producing 240 slice scores. The MLP is trained with AdamW, learning rate $1 0 ^ { - 3 }$ , weight decay $1 0 ^ { - 4 } .$ , batch size 64, and the best checkpoint is selected on the validation loss. Including embedding precomputation, the full Merlin baseline required approximately 6 hours on an A100 GPU. For the 3D U-Net baseline, we train a compact 3D U-Net from scratch on ReXGroundingCT. Full CT volumes are resampled to 1.25 mm isotropic spacing and resized to 160×224×224. The encoder uses 3D convolutional blocks with base width 16 and instance normalization. To obtain slice scores, the final feature maps are averaged over the in-plane dimensions, interpolated to 240 axial positions, concatenated with a 32-dimensional disease embedding, and passed through a slice-regression head. The model is trained with AdamW, learning rate $1 0 ^ { - 4 }$ , weight decay $1 0 ^ { - 4 }$ , batch size 16, dropout 0.1, and the best checkpoint is selected on the validation loss and Training to the selected checkpoint required approximately 22 hours on an A100 GPU. Both 3D baselines are therefore substantially more compute-intensive to train than XGLASS, which trains in under 10 minutes on a single RTX 4080.

![](images/9513df713b0ff478b4754dcc11f043dec63e8ce24f9c2662c11ec340dd19fee7.jpg)  
Figure 10. Matched-budget downstream screening across slice selectors. For each CT–finding pair, different selectors choose the Top-K inspected axial slices under the same slice budget. XGLASS is compared with BioMedCLIP retrieval, Vote-MI, 3D u-Net, 3D Merlin and Random selection across MedGemma 4B, Lingshu 7B, and BioMedCLIP downstream screening.

## G.3. Slice-wise Medical VLM Screening

We evaluate google/medgemma-4b-it [35] and lingshu-medical-mllm/Lingshu-7B [44] using the same disease-conditioned binary screening task. Motivated by CT vision-language transfer work that maps complementary HU windows to RGB channels to expose different tissue contrasts [30], the main-paper experiments use a pseudo-RGB representation with wide/lung, soft-tissue, and mediastinal/fluid windows assigned to separate channels. As an input-rendering sensitivity analysis, we additionally evaluate a standard wide-window representation (−1024 to 1024 HU) and replicated across all RGB channels. This alternative setting shows the same trend: XGLASS Top-K preselection improves F1 over all-slice screening, with the strongest gains at small budgets (e.g., Top-30 F1 of 0.414 for MedGemma and 0.441 for Lingshu). Thus, the downstream benefit of XGLASS is consistent across CT slice renderings. Both models receive the same system and user instructions:

System: You are an expert thoracic radiologist. Evaluate only the queried finding on one axial chest CT slice.

User: Are there clear visual signs of {finding} in this axial CT slice? Ignore other abnormalities. If the queried finding is not visible in this slice, answer no. Return only one short line, with no explanation: Answer: yes or Answer: no.

We use deterministic decoding and parse the response as yes, no, or unknown; no unknown responses occurred in the reported MedGemma and Lingshu evaluation. MedGemma is evaluated in bfloat16 without quantization, while Lingshu is loaded with 4-bit NF4 quantization and bfloat16 computation due to memory requirements. Predictions are generated independently for every slice and cached, allowing identical predictions to be evaluated on either all 240 slices or XGLASS-selected Top-K subsets. Precision, recall, and F1 are computed only over the inspected slices.

## G.4. Downstream Slice Selection Comparison

We further analyze how slice selection affects downstream screening when the number of inspected slices is fixed. For each CT–finding pair, we vary only the ranking used to choose the Top-K inspected slices, while keeping the downstream slice-wise predictions unchanged. This isolates the effect of slice selection from stochasticity or repeated downstream model calls. We compare XGLASS with Random selection, Vote-MI, BioMedCLIP retrieval, and the two 3Dnative selectors, 3D Merlin + MLP and 3D U-Net. BioMed-CLIP retrieval is evaluated as a selector for MedGemma and Lingshu, but omitted from the BioMedCLIP downstream panel to avoid selecting and classifying slices with the same image–text similarity model. Figure 10 shows that selector choice strongly affects low-budget screening, especially at $K { = } 3 0$ and $K { = } 6 0$ XGLASS consistently outperforms the other 2D selectors under matched slice budgets, while the 3D selectors are competitive and sometimes achieve the highest downstream F1 for specific model–budget combinations. Notably, XGLASS remains close to these volumetric selectors despite using only a lightweight 2D projection, and outperforms them for Lingshu at K=30, 60, and 180. This suggests that the disease-conditioned XGLASS ranking transfers effectively to downstream screening, narrowing much of the gap to more expensive 3D-native selection.

## G.5. Downstream Computational Cost

Applying the downstream models to all 240 slices of one CT requires approximately 1.40 s for BioMedCLIP, 154.91 s for MedGemma, and 267.83 s for Lingshu on a single NVIDIA RTX 4080. XGLASS requires only 0.025 s per finding (0.150 s per CT) to rank the complete volume and allows the downstream model to inspect a smaller, diseaseenriched subset. At Top-30, including XGLASS preselection, the corresponding per-CT runtimes decrease to approximately 0.20 s, 19.39 s, and 33.51 s, respectively. Thus, XGLASS reduces downstream runtime by 85.7–87.5% while simultaneously improving F1 at the smaller slice budget, showing that preselection improves both computational efficiency and downstream screening performance.

![](images/527a2edaa6e236befb1a639f4739d9168297c9c1f205df1f36ccafa9be4dc4f6.jpg)  
Figure 11. Effect of the gaze–heatmap blending strength α on the AIP-wide CT projections. Rows correspond to the six disease categories used in XGLASS (one representative CT per category), and columns sweep the blending coefficient from left to right: α ∈ {0.0, 0.25, 0.5, 0.75, 1.0}. The leftmost column (α = 0) shows the raw CT projection without gaze information, while increasing α progressively injects the REFLACX-derived gaze importance map into the red channel within the chest bounding box.

![](images/c58a7de3e758850ed72811f4699bf50112fc938bb9ea8035af4cde6cfe397894.jpg)

![](images/1edaa14e07457f26fde3f80b62c868e71465358e7fda1a718ae714f75fb24025.jpg)

![](images/3c0f71c35743393ab5e0e9fbece5a5b412c7a80808beebf09bfabe4eab518ec6.jpg)

![](images/c04694493ab0c2a9d6947249776154af1eb9c014d77c38e81d1616cfffb87fc4.jpg)  
Figure 12. AP–PA comparison of disease-conditioned gaze priors, part 1. For each finding, we construct separate AP-only and PA-only REFLACX gaze heatmaps and compare their superior–inferior profiles. The superior–inferior axis encodes the axial slices and therefore it is most relevant for XGLASS.

![](images/7123df29f2748fad520d83afaee9e653c1b452a96a4d3c83ac4703592e1bd713.jpg)

High lung volume / emphysema (AP n=13, PA n=14)  
![](images/ad85179a5fb22b878abe0fbc545ebddb78428784654b78d92aa1574432561e25.jpg)  
Lung nodule or mass (AP n=62, PA n=48)

![](images/5922e0e1c9ad1f081e0b20167f9516ea567ffdd64dc410211e6a96fc51c72d99.jpg)

![](images/331198622422c4b5349abf7f42cfb696dab29d9be739e2b84ccde8e6bb815e86.jpg)

![](images/2ad476d70e61cc1714f51720040099bc19794511444a70165ed42b3a00274e31.jpg)  
Pleural abnormality (AP n=564, PA n=176)

![](images/3b22ecdbcf12c1c487ed57748d9abce9e1aa7e10752d8e02312e6ffca7c37fc3.jpg)

![](images/facffb4cce2af326445a66f366108d6e6feb1c3f12da5c2854178cd7c53eedc6.jpg)

![](images/e56be2e16b1d9cee1c87b18da1c106a4db80b8a2f042cbe5b62e7b386a6e6de6.jpg)

![](images/60f4951d077ddf8f5ef540b130146c69828d6827ce05df3f45587e70eba70457.jpg)  
Figure 13. AP–PA comparison of disease-conditioned gaze priors, part 2. Continuation of the AP–PA comparison for the remaining findings. Similar superior–inferior profiles across views support combining AP and PA sessions into more stable disease-conditioned gaze priors.

![](images/cca695b397406d161b38c9f1b269014a6d8dcdbbdd73c7cd29e67fb55d7d978b.jpg)  
Figure 14. CXR–CT comparison of disease-conditioned gaze priors, part 1. We compare REFLACX CXR-derived gaze priors with sentence-matched CT-ScanGaze fixation heatmaps. Although CT reading follows different fixation dynamics, the superior–inferior profiles show similar coarse disease-specific attention patterns, supporting the idea to transfer CXR gaze priors to CTs.

![](images/510b79b9cfd1ef5fa39aec01067f88898087715d2e4963ee24804b0944a29743.jpg)  
High lung volume / emphysema (n=8, fix=720)

![](images/523a874580a26ac55fa708695315a91248a28726e18d925240071af7dc15a07a.jpg)

![](images/c243a351e5bc3b2500ef6a26b4431db0d28c7baf426d4f965660fdc064121da7.jpg)  
Figure 15. CXR–CT comparison of disease-conditioned gaze priors, part 2. Continuation of the CXR–CT comparison for the remaining findings. The aligned superior–inferior profiles support transferring CXR-derived disease-specific search priors to CT slice selection.
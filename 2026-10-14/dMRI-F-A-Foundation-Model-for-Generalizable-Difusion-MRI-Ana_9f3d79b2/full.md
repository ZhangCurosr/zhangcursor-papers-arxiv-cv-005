# dMRI-F: A Foundation Model for Generalizable Difusion MRI Analysis from Sparse Acquisitions

Xi Zhu<sup>1</sup>, Yijie Li<sup>1</sup>, Xiaofan Wang<sup>1</sup>, Wei Zhang<sup>1</sup>, Huafu Chen<sup>1</sup>, Ce Zhu<sup>1</sup>, Fan Zhang<sup>1,\*</sup>

<sup>1</sup>University of Electronic Science and Technology of China.

\*Corresponding author(s). E-mail(s): fan.zhang@uestc.edu.cn;

## Abstract

Difusion MRI (dMRI) probes water difusion within the brain, providing unique information about tissue microstructure and white matter connectivity. High-quality dMRI enables advanced and accurate analyses but demands dense acquisitions, resulting in long scan times and hardware constraints that are often infeasible. Further, dMRI requires dedicated computational methods to support diverse analysis tasks (e.g., parcellation, registration, and synthesis) for both neuroscientific and clinical applications. Here, we present dMRI-F, a foundation model that unifies dMRI data enhancement and downstream tasks across diverse acquisitions and populations, while requiring only minimal acquisition input. dMRI-F is pre-trained with a q-space masked reconstruction task: predicting densely sampled multi-shell data from only six volumes. It learns generalizable latent feature representations of dMRI data, which serve as a core foundation for adapting to diverse downstream tasks, including brain parcellation, brain registration, and cross-modal synthesis. Across nine independently acquired datasets, we demonstrate that dMRI-F outperforms state-of-the-art methods with robust generalization across acquisitions, populations, and tasks. Our results demonstrate dMRI-F’s versatility as a general-purpose framework that enables diverse dMRI applications from low-cost, sparse acquisitions.

Difusion magnetic resonance imaging (dMRI) is an advanced, non-invasive imaging technique that can uniquely measure the random motion of water molecules within the brain, enabling in vivo characterization of tissue microstructure<sup>1</sup> and structural connectivity <sup>2</sup>. In dMRI, the acquired data, known as difusion-weighted images (DWIs), are collected by sampling q-space, where each sampled location corresponds to a DWI volume acquired with a specific gradient direction (i.e., b-vector) and strength (i.e., b-value). From the DWIs, biophysical models can be fit to derive microstructural measures (e.g., fractional anisotropy, FA) <sup>3</sup>, and fiber tractography can be performed to map the brain’s structural connectivity <sup>4</sup>. Leveraging these properties, dMRI has found broad utility in both neuroscientific research and clinical practice, with applications spanning brain development <sup>5</sup>, aging<sup>6</sup>, neuroplasticity<sup>7</sup>, and cognitive function<sup>8</sup>, as well as lesion characterization <sup>9</sup>, disease monitoring<sup>10</sup>, and neurosurgical planning<sup>11</sup>.

A key demand in dMRI is acquiring high-quality DWI data, combining dense q-space sampling, high spatial resolution, and robust signal fidelity, to improve the accuracy of modeling subtle microstructural features and complex fiber configurations. However, acquiring such high-quality data entails prolonged scan times and substantial hardware demands. The Human Connectome Project Young Adult (HCP-YA) project <sup>12</sup> exemplifies this trade-of: its dMRI protocol achieves high 1.25 mm isotropic resolution, dense 270 difusion directions across three diferent shells, and multiple phase-encoding acquisitions for artifact correction, but requires customized scanners and nearly one hour of scanning time per subject. While the HCP-YA data have proven immensely valuable to the neuroimaging and neuroscientific community <sup>13–15</sup>, this level of quality remains impractical for most routine studies, which typically acquire single-shell acquisitions with ˜2 mm isotropic resolution and far fewer gradient directions—often as few as 6, the minimum required for conventional difusion tensor modeling <sup>16</sup>. Consequently, practical dMRI remains quality-limited, necessitating eficient DWI enhancement methods that operate under routine acquisition constraints.

Deep learning has emerged as a promising approach for enhancing dMRI quality by inferring high-quality DWI data from sparse acquisitions. Existing methods generally fall into three major categories based on their primary objectives. The first focuses on angular super-resolution, where densely sampled difusion signals encompassing a large number of gradient directions across multiple b-values are inferred from sparsely acquired q-space data<sup>17–19</sup>. The second category emphasizes spatial super-resolution, aiming at enhancing the spatial resolution of DWI data by generating fine-grained structural details from low-resolution scans <sup>20–22</sup>. The third category addresses denoising, with the goal of reducing imaging noise to improve the fidelity of difusion signals <sup>23,24</sup>. Overall, these methods aim to capture the complex relationships inherent to the high-dimensional dMRI signal space—spanning both spatial (3D) and difusion-encoding (q-space) dimensions—to enable prediction of enhanced signal data. However, despite their success, most existing methods process the data as independent DWI slices, limiting their ability to exploit its full high-dimensional spatial continuity. Furthermore, these methods are typically tailored to specific enhancement objectives and acquisition protocols, leaving a strong need for a unified framework for broad applicability.

Beyond data-quality enhancement, dMRI also demands accurate and eficient methods for various computational tasks. Unlike traditional anatomical MRI (e.g., T1w), dMRI uniquely characterizes water difusion using multi-directional, multi-shell acquisitions. The resulting large number of image volumes (DWIs) creates substantial computational challenges and requires dedicated analytical methods. For example, in brain parcellation, recent studies have sought to perform the task directly on DWIs by extracting tissue-informative microstructural contrasts, thereby bypassing the inter-modality registration required by conventional T1w-based approaches <sup>25–27</sup>. Similarly, in brain registration, an emerging trend is to incorporate multiple complementary dMRI-derived maps that provide distinct tissue and fiber orientation information to guide image alignment <sup>28–30</sup>. Furthermore, cross-modal synthesis methods generate T1w images directly from DWIs when native anatomical images are unavailable or cannot be reliably aligned with difusion data <sup>31,32</sup>. Across these distinct tasks, the common goal is to extract anatomically informative features directly from dMRI. Yet, the pursuit of feature representations that are broadly applicable across various tasks remains underexplored.

Another demand in dMRI is generalizable methods across diverse acquisitions and populations. dMRI is known to be highly sensitive to acquisition protocols, scanner types, and parameter settings, which often undermines a method’s generalizability across studies <sup>33,34</sup>. In related fields in neuroimaging, one common strategy to improve generalizability is to train on large-scale pooled datasets that increase data variability <sup>35,36</sup>—though this typically requires substantial data curation eforts and high computational costs. In dMRI, an emerging alternative is to leverage high-quality reference datasets (e.g., HCP-YA) that capture a broad distribution of difusion properties, with the aim of generalizing to clinically typical acquisitions and populations. For example, the strategy of training on HCP-YA data and applying to other datasets has proven successful across various tasks, including brain segmentation <sup>37</sup> and tract parcellation <sup>38–40</sup>, with demonstrated generalizability across datasets spanning diverse genders, ages, and health conditions. Central to this strategy is the extraction of informative contrasts that provide compact yet discriminative representations—essential for achieving robust generalizability.

Here, we introduce dMRI-F, a foundation model for dMRI that jointly performs DWI enhancement and learns generalizable latent feature representations, enabling diverse downstream computational tasks across varying acquisition protocols and populations. dMRI-F employs a self-supervised learning (SSL) framework, pre-trained through a q-space masked-reconstruction task that predicts densely sampled multi-shell DWIs from sparse singleshell inputs (Fig. 1a). Architecturally, the model comprises an anatomy-aware encoder that learns latent features reflecting the underlying anatomical structures from input volumes, and a coordinate-conditioned decoder that predicts unseen volumes by jointly modeling these features with DWI spatial and q-space coordinates (Fig. 1b). By generating enhanced, densely sampled data, dMRI-F enables advanced analysis of tissue microstructure and white-matter connectivity from sparse inputs of only six volumes (Fig. 1c). Furthermore, by fine-tuning the learned latent features, dMRI-F enables generalizable adaptation to diverse downstream tasks, including brain parcellation, registration, and cross-modal synthesis (Fig. 1d). Using high-quality data from only 200 HCP-YA subjects for training, dMRI-F achieves robust generalization for DWI enhancement and downstream tasks across various acquisitions and populations (1,882 unseen subjects from the high-quality HCP-YA test set and eight independent test datasets with clinically typical acquisitions) (Fig. 1e). Together, our results demonstrate dMRI-F as a versatile, general-purpose computational framework for dMRI, ofering a solid foundation for diverse neuroscientific and clinical applications with minimal acquisition requirements.

## Results

## Overview

We demonstrate dMRI-F from the following aspects. First, we present dMRI-F’s advantages in DWI enhancement to predict densely sampled data from sparse, low-b-value inputs. These include: (1) a benchmark comparison with state-of-the-art (SOTA) methods on high-quality HCP-YA data; (2) a generalization assessment across eight independent test datasets from diverse acquisition protocols and populations; and (3) a fidelity evaluation of the enhanced DWIs for preserving white matter connectivity and population tissue microstructure. Next, we showcase dMRI-F’s capability for downstream computational tasks by fine-tuning task-specific decoders on the learned latent representations. These include: (1) brain parcellation, which subdivides DWIs into anatomically segregated brain regions; (2) brain registration, which aligns DWI volumes across diferent a, Pretraining Task: Extending the principle of conventional spatial masked reconstruction, dMRI-F performs q-space masked reconstruction to predict 270 densely sampled multi-shell DWIs across $\mathrm { b } = 1 , 0 0 0 , 2 , 0 0 0$ and $\mathrm { 3 , \bar { 0 0 0 } s / m m ^ { 2 } }$ using only six single-shell volumes at $\mathrm { b } = 1 , 0 0 0 \mathrm { s } / \mathrm { m m } ^ { 2 }$ as input. $\mathbf { b } ,$ Network Architecture: dMRI-F includes an anatomy-aware encoder that extracts latent feature representations describing the underlying anatomical structures from the sparse DWI inputs, and a Neural Radiance Fields (NeRF)-inspired coordinate-conditioned decoder that models the relationships between the latent features and the corresponding spatial and q-space coordinates to predict the target DWI signals. c, DWI Enhancement: The predicted densely sampled DWI data enable advanced analysis of white-matter connectivity and tissue microstructure— analyses that are not feasible from the original six-volume input alone. d, Downstream Task Adaption: The learned latent feature representations support generalized adaptation to diverse downstream tasks, including brain parcellation, brain registration, and cross-modal synthesis. e, Training and Evaluation Data: dMRI-F was pre-trained on high-quality dMRI data from 200 HCP-YA subjects and evaluated on 1,882 unseen subjects, comprising the HCP-YA test set and eight independent datasets with clinically typical acquisitions.

![](images/1ca30445bd3091292e36263868432e948dda33740afa342cdfe0f92b05bf37d4.jpg)

![](images/fe3c2f07632f052160649f24709d436626029efbbd81310538c0526fd47daabf.jpg)

![](images/b684feccbc0ef1a45d47e8bc12afe6f1bd47eadf7da358ad691a171b75ec30f2.jpg)

![](images/bced879f0cc2bdd3f26699509f438ce9e6c24922187b7df41cd4e9b159fb78ae.jpg)

![](images/3f76f9caaa22ff63d1a1c9bffc5df985b46230e06005b73326a2f447ac4096bd.jpg)  
Fig. 1 — Overview of dMRI-F and the study design.

subjects; and (3) cross-modal synthesis, which generates anatomical T1w images from DWIs. For each task, we benchmarked dMRI-F against representative SOTA methods and also assessed generalizability on both the HCP-YA and the test datasets with clinically typical acquisitions. Finally, we assess our method design, including: (1) an ablation study evaluating the network architecture and major parameters (see “Extended Data Table 1”); (2) an analysis of the efect of training subject size (see “Extended Data Fig. 1”); and (3) an evaluation of model performance under diferent numbers of input difusion directions (see “Extended Data Fig. 2”).

## dMRI-F Robustly Predicts Dense DWI Data from Sparse Inputs Across Acquisitions and Populations

## Superior DWI Enhancement Performance on Benchmark HCP-YA Data

We first evaluated dMRI-F’s ability on our pretraining DWI enhancement task, i.e., predict densely sampled multi-shell data (270 DWI volumes across b = 1,000, 2,000, and $\mathrm { 3 , 0 0 0 s / m m ^ { 2 } ) }$ from sparse, low-b-value input (6 volumes at $\mathrm { b } = 1 , 0 0 0 \mathrm { s } / \mathrm { m m } ^ { 2 } )$ . This was conducted on a benchmark test set of 200 HCP-YA subjects. We compared dMRI-F with four SOTA methods: (1) SH-Inter<sup>41</sup>, (2) Q-GAN<sup>18</sup>, (3) Q-MMTN<sup>17</sup>, and (4) PCConv<sup>19</sup>. Briefly, SH-Inter performs interpolation in the spherical harmonics domain, Q-GAN and Q-MMTN learn sparseto-dense DWI mappings with additional T1w and T2w inputs, whereas PCConv learns a direct mapping from only DWI data (see Supplementary Table 1 for details). Performance was evaluated against the full DWI data as reference, at two levels: (1) DWI signals at each b value, and (2) microstructure measures (including fractional anisotropy; FA, mean kurtosis; MK, etc.) derived for difusion kurtosis (DK) modeling <sup>42</sup>. We used three metrics: Normalized Root Mean Square Error (NRMSE) for voxel-wise error, Peak Signal-to-Noise Ratio (PSNR) for peak error derived from MSE, and Structural Similarity Index Measure (SSIM) for contrast and structural similarity.

![](images/9e35156caef2ae507b9bd6955af42cf5f566919a8d9bf6764bf5f606878f5c33.jpg)  
Fig. 2 — dMRI-F achieves superior data enhancement performance in both DWI signal prediction and microstructural measure estimation in the benchmark HCP-YA data.  
a, Quantitative comparisons between dMRI-F and the SOTA DWI enhancement methods (SH-Inter, Q-GAN, Q-MMTN, and PCConv) at both the DWI signal (at b = 1,000, 2,000, and 3,000 s/mm<sup>2</sup>) and derived microstructure measure (FA and MK) levels. b, Visualization of example DWIs, FA maps, and MK maps derived from each method, together with the corresponding reference images from the full DWI data. Yellow boxes and zoomed views highlight example local regions. FA Dif and MK Dif show absolute errors of each method relative to the reference. Note: For all box plots, the center line denotes the median, the box spans the interquartile range, and the whiskers extend to 1.5 times the interquartile range. White diagonal crosses denote the mean, whereas plus signs mark outliers beyond the whiskers. Significance is shown for the comparisons marked by brackets: \*p $\textit { i } 0 . 0 5 , \ { } ^ { * * } p \ i \ 0 . 0 1$ , and $^ { * * * } p \textit { j } \theta . 0 0 1$

dMRI-F outperformed each SOTA method at both evaluation levels (Fig. 2a). For the DWI signals, dMRI F achieved statistically significant improvements over PCConv, the strongest compared method, with higher PSNR and SSIM and lower NRMSE. This gain was also evident at the derived microstructure measures, where dMRI-F consistently outperformed all competing methods in the FA and MK maps (see Supplementary Table 2 for results of additional maps) and each metric. Notably, dMRI-F maintained its advantage across all three difusion shells, with particularly strong performance at the higher b-value shells. This more accurate recovery of high-b-value signals was reflected in the pronounced gains observed for the kurtosis-related MK maps. Visua comparisons showed that the predicted DWIs from dMRI-F most closely matched the reference across all three shells, with better-preserved tissue boundaries and structural details (Fig. 2b and Supplementary Fig. 1). The FA and MK maps derived from dMRI-F likewise exhibited clearer tissue contrast and fewer local artifacts, together with visibly lower residual errors than those of the compared methods.

## Robust Generalization for DWI Enhancement Across Independently Acquired Test Datasets

To assess whether dMRI-F generalizes beyond the HCP-YA training data under clinically realistic conditions, we evaluated the trained DWI enhancement model on eight test datasets comprising 1,682 subjects (see Methods and Supplementary Table 3 for demographic and acquisition details). These cohorts spanned a wide age range from 10 to 95 years, included both healthy and multiple disease conditions, and were acquired across multiple scanner manufacturers and protocols covering difusion shells at 1,000, 1,500, or $\mathrm { 2 , 0 0 0 s / m m ^ { 2 } }$ . In contrast to the high-quality HCP-YA training set, these test datasets had lower spatial resolution, higher noise, and substantial cross-site image-quality variation, providing a stringent test of dMRI-F’s generalizability across diverse data conditions. We compared dMRI-F against PCConv—the top-performing method in the preceding benchmark evaluation—under identical conditions for predicting the densely sampled DWI data from sparse inputs of six volumes.

Across the eight test datasets, dMRI-F significantly outperformed PCConv across all quantitative metrics, at both the DWI signal and derived microstructural measure levels (Fig. 3a). Notably, the improvement was particularly evident on the ABIDE dataset, which was acquired at $\mathrm { b } = 1 { , } 5 0 0 \mathrm { s } / \mathrm { m m } ^ { 2 }$ , a shell not observed during model training on HCP-YA. Qualitatively, despite substantial variation in input quality, the DWIs and FA images derived from dMRI-F were visually highly consistent with the reference data. The zoomed images reveal that dMRI-F better recovered tissue boundaries and local anatomical patterns, whereas PCConv often produced smoother predictions with less distinct anatomical contrast (Fig. 3b). Collectively, these results showed that dMRI-F generalized robustly across heterogeneous imaging conditions and populations, with the ability to extrapolate to difusion shells beyond those encountered during training.

## High Fidelity of Predicted Data for Preserving White Matter Connectivity and Population Tissue Microstructure

We performed two post hoc analyses—fiber tractography and population-level microstructure comparison—to assess whether the enhanced DWI data preserve information essential for downstream applications.

For the tractography analysis, we processed the nine evaluation datasets (HCP-YA and eight test datasets) through a standard pipeline: fiber orientation distribution (FOD) estimation and whole-brain tractography (using MRtrix<sup>43</sup>), followed by automated atlas-based tracts identification (using WhiteMatterAnalysis with the ORG atlas<sup>15,44</sup>). This was applied to the predicted DWI data from our dMRI-F method and PCConv, as well as the reference full DWI data. Results showed that dMRI-F achieved significantly more accurate FOD estimation across all datasets (measured using the angular correlation coeficient), producing FOD maps visually similar to the reference (Fig. 4a). Visual comparisons of the derived fiber tracts (the corticospinal tract in Fig. 4a and additional tracts in Supplementary Fig. 2) further demonstrated that dMRI-F tractography more closely resembled the reference, whereas PCConv frequently yielded sparser trajectories and, in some cases, failed to reconstruct the fiber pathways.

For the microstructure analysis, building on the experiments demonstrating that the derived microstructural estimates were highly consistent with those obtained from the full DWI (see Fig. 3 in the above experiment), we further conducted two population-level comparisons: a sex comparison (female vs. male) and a disease comparison (Alzheimer’s disease, AD vs. normal controls, NC). For the sex comparison, we included 20 agematched female–male pairs from the HCP-YA testing cohort, computed the mean fractional anisotropy (FA) across the whole white-matter skeleton—an interpretable marker of sex-related white-matter organization <sup>45</sup>— and compared it between groups using a two-sided t-test. For the disease comparison, we included 20 age-matched AD–NC pairs from the ADNI<sup>46</sup> dataset and compared mean skeleton MD (a highly sensitive measure for detecting AD-related white-matter abnormalities <sup>47</sup>) between the two groups. We observed significantly higher FA in females in the sex comparison and significantly higher MD in the AD group in the disease comparison, with the detected group diferences being highly consistent between the predicted and full-reference data (Fig. 4b).

Overall, these results suggest that dMRI-F-enhanced DWIs largely preserve fidelity for FOD estimation and enable anatomically reliable tractography across varied acquisition conditions. Moreover, the enhanced data appear to retain known population-level microstructure diferences, consistent with those observed in the full reference data.

![](images/45812693fb397d2084cbb6fe4de4ae9dee7f6a8940a01973e31c53afc06ecd88.jpg)  
Fig. 3 — dMRI-F enables robust generalizations for DWI enhancement across acquisition protocols and populations.  
a, Quantitative comparison of dMRI-F and PCConv for DWI signal prediction and microstructural estimation across the multiple testing datasets with clinically typical acquisitions. b, Visualization of the predicted DWIs and corresponding derived FA maps in each test dataset. Separate ADNI examples are shown for the b = 1,000 and $\mathrm { 2 , 0 0 0 s / m m ^ { 2 } }$ shells, and ABIDE represents an unseen $\mathrm { b } = 1 { , } 5 0 0 \mathrm { s } / \mathrm { m m } ^ { 2 }$ acquisition. Red boxes and enlarged views highlight local tissue boundaries and anatomical details. Note: For all box plots, the center line denotes the median, the box spans the interquartile range, and the whiskers extend to 1.5 times the interquartile range. White diagonal crosses denote the mean, whereas plus signs mark outliers beyond the whiskers. Significance is shown for the comparisons marked by brackets: ${ } ^ { * } p \ i \ 0 . 0 5 , { } ^ { * * } p \ i \ 0 . 0 1 .$ , and $^ { * * * } p \textit { j } \theta . 0 0 1$

![](images/5497cfc3fb1f80205bdd389f51317ff5ec5f51e5923dfe7e53466728948fb265.jpg)

![](images/308e78d0728b0e6c85905e7eb37b64927ab5c6f1edfe2875e5bfead7983c8b89.jpg)

![](images/a926caa0580e4b53e6856572c1fa1514282b1ba32dd6997348a1ee4d873404ac.jpg)  
Fig. 4 — dMRI-F’s predicted DWIs preserve fiber-orientation information for tractography and maintain microstructure information for population-level comparisons.  
a, Quantitative comparison of FOD estimation accuracy between dMRI-F and PCConv, with visualization of example FOD maps and derived fiber tracts from both methods alongside the reference data. b, Populationlevel microstructure analyses in HCP-YA for sex diference comparison (left) and ADNI for disease diference comparison (right). Note: For all box plots, the center line denotes the median, the box spans the interquartile range, and the whiskers extend to 1.5 times the interquartile range. White diagonal crosses denote the mean, whereas plus signs mark outliers beyond the whiskers. Significance is shown for comparisons marked by brackets: $^ { * } p \textit { j } 0 . 0 5 , ^ { * * } p \textit { j } 0 . 0 1 , \mathrm { a n d } ^ { * * * } p \textit { j } 0 . 0 0 1$

![](images/a5246256628c0dbcc83d346fa6daab984fed112e0db42578724d700681de1ea9.jpg)  
Fig. 5 — dMRI-F learns consistent feature representations across datasets.  
Visualization of feature maps obtained from the pretrained dMRI-F encoder for example subjects from each evaluation dataset. The top row shows one DWI volume from each subject’s input for reference. For visualization purposes, the first three principal components obtained by applying principal component analysis (PCA) to the features are shown for each subject. Despite substantial diferences in acquisition protocols, image quality and study populations, each component consistently emphasizes similar tissue contrasts.

## dMRI-F Learned Feature Representations Enable Adaptation to Diverse Downstream Tasks

The dMRI-F encoder enabled extraction of generalizable latent features across subjects, highlighting corresponding anatomical patterns and tissue boundaries. As illustrated in Fig. 5, corresponding tissue structures showed visually consistent contrasts in the principal components of the features across the example subjects, e.g., ventricular regions appearing dark in component 1, deep white matter appearing dark in component 2, and the white/gray matter interface appearing bright in component 3. This consistency suggested that the features learned during pretraining captured stable structural information, providing a transferable basis for downstream computational dMRI tasks.

In subsequent experiments, we evaluated downstream task adaptation by fine-tuning task-specific decoders using the learned latent features for three distinct applications: (1) brain parcellation, which subdivides DWIs into anatomically segregated regions; (2) brain registration, which aligns DWI volumes across diferent subjects; and (3) cross-modal synthesis, which generates anatomical T1w images from DWIs. For each downstream task, from an input with six DWI volumes, the pre-trained dMRI-F encoder was employed to extract latent features, and a task-specific decoder was trained on a subset of 50 HCP-YA training subjects (see Methods). In the following, we refer to these methods as dMRI-F-Parc, dMRI-F-Reg, and dMRI-F-Syn, respectively. Evaluation of each method was performed on both the high-quality HCP-YA data and the eight clinically typical test datasets, comparing their performance against representative SOTA methods in each task.

## Brain Parcellation into Anatomically Segregated Regions

Brain parcellation is important in dMRI for region-specific quantification and anatomically accurate tractography. Parcellation directly from DWI bypasses the need for additional anatomical scans, avoids error-prone inter-modality registration, and accelerates processing time <sup>26,27</sup>. Our proposed dMRI-F-Parc leverages learned feature representations to perform fine-scale parcellation of 101 anatomical regions corresponding to the Desikan-Killiany (DK) parcellation <sup>48</sup> (see Methods: Brain Parcellation). We compared with two SOTA methods: (1) FastSurfer<sup>49</sup>, which performs parcellation from T1w images and registers the parcellation to DWI; and (2) DDParcel<sup>27</sup>, which uses dMRI-derived microstructure measures as inputs for parcellation (See Methods and Supplementary Table 4 for details).

As shown in Fig. 6, dMRI-F-Parc demonstrated consistent advantages across both the HCP-YA and clini cally typical test datasets. In the HCP-YA data, it achieved the highest Dice scores relative to the ground-truth parcellation across cortical, subcortical, and white matter regions, significantly outperforming the compared methods (Fig. 6a). This improvement was most pronounced in cortical regions—notoriously dificult due to their complex cortical folding—where dMRI-F-Parc significantly outperformed the second-ranking DDParcel. Visual comparisons showed that dMRI-F-Parc parcellations more closely matched the ground-truth anatomical boundaries, particularly in the cortical regions (Fig. 6b). In the eight clinically acquired test datasets, dMRI-F-Parc showed robust generalization, achieving significantly lower relative standard deviation (RSD) of FA (a measure of regional parcellation homogeneity <sup>27</sup>) than DDParcel in each dataset (Fig. 6c). Visual inspection supported these quantitative findings: dMRI-F-Parc produced regional tissue boundaries that were less displaced and better aligned with visible anatomical structures across heterogeneous datasets (Fig. 6d). Together, these results suggested that dMRI-F preserved parcellation-relevant anatomical information and enabled robust regional delineation directly from DWI data across various datasets.

a.  
![](images/700b258bc6f21b1294b29184ab7a95f8ba0d04e0585985d22ba866fbd28580bd.jpg)

![](images/56843f34c8154809dc0efbf78a1a30d0b620c32542a4e7776ad0e43bdc52c0dc.jpg)

![](images/8f7a942dfa7ddc206d7ca79f50e9b0c2e6926cda2e474a1173335e23258be2de.jpg)

![](images/6f5a77619b02b2135f7bd5792dc36cd035eb1d405a8ae220bb57d92f1607a10c.jpg)

![](images/c90cfd1ad6004be54ba94b8638cb2a2c272814ece51c737bd90b3fa4d1b0b953.jpg)  
Fig. 6 — dMRI-F-Parc enables accurate and generalizable brain parcellation directly from difusion MRI.

a, Dice coeficients for cortical, subcortical, and white-matter parcellation against ground truth on the HCP-YA test set are compared across DDParcel, FastSurfer, and dMRI-F-Parc. dMRI-F-Parc achieved significantly higher Dice scores than DDParcel and FastSurfer across all region groups, with the largest gain in cortical regions. b, Visualization of parcellation results from example subjects for each method, shown alongside the ground-truth reference, with enlarged views highlighting representative regions. dMRI-F-Parc more closely matched the tissue boundaries as appearing on the reference, particularly in cortical folding regions. Red arrows highlight representative regions where dMRI-F-Parc provides more accurate anatomical parcellation than the compared methods. $\mathbf { c } ,$ Quantitative comparison on the eight clinical test datasets, where dMRI-F-Parc achieved significantly lower RSD than DDParcel across all datasets. d, Visual comparison of parcellation results on example subjects across datasets. The enlarged views show representative parcels, in which dMRI-F-Parc exhibits close visual alignment with the anatomical structures visible in the background FA image. Red arrows highlight representative regions where dMRI-F-Parc provides more accurate anatomical parcellation than DDParcel. Note: For all box plots, the center line denotes the median, the box spans the interquartile range, and the whiskers extend to 1.5 times the interquartile range. White diagonal crosses denote the mean, whereas plus signs mark outliers beyond the whiskers. Bars show the cohort mean, and error bars indicate the standard deviation. Significance is shown for the comparisons marked by brackets: ${ } ^ { * } p \ i \ 0 . 0 5 , { } ^ { * * } p \ i \ 0 . 0 1 .$ and $^ { * * * } p \textit { j } \theta . 0 0 1$

![](images/f21e5549f321e7611563a21b17c740df07bb3d55a610fd2e7124b09c8d0903ce.jpg)  
Fig. 7 — dMRI-F-Reg improves inter-subject difusion MRI registration across acquisition settings.  
a, Performance on registering the corresponding anatomical structures on the HCP-YA test set. dMRI-F-Reg achieved significantly higher Dice scores than the compared methods across all region groups. b, Visualization of the registration results on an example moving and target HCP-YA subject pair. The enlarged views show the representative pallidum structure warped by each method (red) overlaid on the target (yellow), where dMRI-F-Reg yielded the closest spatial alignment. c, Performance on registering fiber tracts on the HCP-YA test set, where dMRI-F-Reg achieved the highest tract spatial Dice overlap. Visualization of an example corpus callosum (CC) tract after registration (in red) shows the closest spatial correspondence with the target tracts (in yellow). d, NCC between warped and fixed b0 images across the eight test datasets. dMRI-F-Reg achieved significantly higher NCC than SynthMorph in every dataset. e, Visualization of example moving, target, and warped b0 images. Enlarged views show improved local alignment with dMRI-F-Reg, and the arrows highlight representative regions where dMRI-F-Reg achieves closer anatomical alignment with the target images. Note: For all box plots, the center line denotes the median, the box spans the interquartile range, and the whiskers extend to 1.5 times the interquartile range. White diagonal crosses denote the mean, whereas plus signs mark outliers beyond the whiskers. Bars show the cohort mean, and error bars indicate the standard deviation. Significance is shown for the comparisons marked by brackets: \*p ¡ 0.05, \*\*p ¡ 0.01, and \*\*\*p ¡ 0.001.

## Brain Registration Between Subjects

Brain registration across subjects is a critical step in applications such as population-based analyses in health and disease and the construction of brain atlases. In DWI registration, the goal is to spatially align brain anatomical structures while preserving the consistency of local fiber orientations with the underlying white matter tract anatomy. Our proposed dMRI-F-Reg addresses this by training a registration model that takes learned features from subject pairs (moving and fixed) and predicts deformable fields to warp the moving DWI data into the fixed space (see Methods: Brain Registration). We compared with two SOTA methods: (1) VoxelMorph <sup>50</sup>, which is a widely used unsupervised framework to learn deformation fields from input image pairs; and (2) SynthMorph <sup>51</sup>, which is a synthesis-based registration method that generalizes robustly across contrasts, resolutions, and anatomies (See Methods and Supplementary Table 5 for details).

Quantitatively, dMRI-F-Reg achieved the highest Dice coeficient across cortical, subcortical, and white matter regions (Fig. 7a). Visual comparison on example HCP-YA subjects (Fig. 7b) shows that dMRI-F-Reg produced superior anatomical alignment, with warped anatomical regions more closely matching the fixed space than those from the compared method. The largest improvement was observed in the cortical regions, a particularly challenging area due to its highly folded geometry and substantial inter-subject variability. Furthermore, when warping fiber tracts (derived using TractSeg <sup>40</sup>), dMRI-F-Reg produced the greatest spatial overlap (Fig. 7c), demonstrating more accurate alignment of tract-specific white-matter anatomy. Across all eight testing datasets, dMRI-F-Reg demonstrated statistically robust and spatially precise registration, achieving significantly higher Normalized Cross-Correlation (NCC) scores than SynthMorph (Fig. 7d). The qualitative improvements at tissue boundaries (Fig. 7e) indicate that dMRI-F-Reg better preserves structural similarity, likely due to its ability to model complex, non-linear deformations without over-smoothing—a common limitation in learning-based approaches like SynthMorph.

Cross-Modal Synthesis of Anatomical T1w Images

Brain-ID  
d.  
a.  
![](images/0746430f73f863f43d7dea2be157cffe4b6bc59ecfd6279e36ef61ea4a3a4e3b.jpg)

![](images/cfd82d93d3ffddb4fad0ff5cf4f69d88b9995fbe3f0f540f8aae74c6f251c878.jpg)

![](images/56a58af6c0316893c45ef4a0827cb6d84ffd103ffcf61e93d5eb72962e00b4ae.jpg)

![](images/96b81cc82498a8a80775847aa6256777f7291b0d62e9bc921e037cf939b85f1f.jpg)

![](images/7a145ffbeb7c5131a80126e2a8af67a03033a2fda4cebbfd682a04a054f143bc.jpg)

![](images/9b7f12a8e7acb4bde5681c26b25937b0f6e48e8fab6ee5f4f69a3b9f3b860930.jpg)

![](images/85e01b6d4c2705a0f65098f41c7a7307194c1e02c6b67020bd41b458a267ced9.jpg)  
Fig. 8 — dMRI-F-Syn enables anatomically faithful synthesis of T1w images from difusion MRI. a, SSIM, PSNR, and NRMSE between synthesized and acquired T1w images in the HCP-YA test set. dMRI-F-Syn achieved significantly better results in each metric compared to Brain-ID and DeepAnat. b, Absolute cortical-thickness diferences between surfaces reconstructed from the synthesized and the real T1w images in the HCP-YA data. Lateral, medial and ventral views of both hemispheres are shown. c, Mean absolute cortical-thickness diferences between surfaces reconstructed from synthesized and acquired T1w images across the seven testing datasets. Bars represent the mean, and error bars represent the standard deviation. Missing DeepAnat results indicate datasets in which abnormal synthesized-image intensities prevented successful surface reconstruction. d, Representative T1w images synthesized by DeepAnat, Brain-ID, and dMRI-F-Syn across eight test datasets. Red boxes and enlarged views highlight local tissue boundaries and anatomical details. The corresponding reference FA maps are shown for anatomical localization. Note: For all box plots, the center line denotes the median, the box spans the interquartile range, and the whiskers extend to 1.5 times the interquartile range. White diagonal crosses denote the mean, whereas plus signs mark outliers beyond the whiskers. Bars show the cohort mean, and error bars indicate the standard deviation. Significance is shown for the comparisons marked by brackets: $^ { * } p \ i \ 0 . 0 5 , ^ { * * } p \ i \ 0 . 0 1 .$ , and \*\*\*p ¡ 0.001.

Cross-modal synthesis serves as an important tool for enriching dMRI with supplementary anatomical context, thereby supporting comprehensive neuroimaging analyses. While dMRI ofers distinct information about water molecule difusion properties, it lacks the explicit anatomical details as provided by T1w images. The proposed dMRI-F-Syn addresses this by learning mappings from the DWI-derived feature representations to the corresponding T1w data, enabling the prediction of synthetic T1w images from new DWI data (see Methods: Cross-modal Synthesis). We compared with two SOTA methods: (1) Brain-ID <sup>31</sup>, which learns contrast-agnostic anatomical representations to synthesize T1w images for any imaging input (b0 is used in our experiment); (2) DeepAnat<sup>32</sup>, which synthesizes T1w images directly from DWI using a U-Net and a hybrid generative adversarial network (see Supplementary Table 6 for details).

On the HCP-YA data, dMRI-F-Syn demonstrated superior T1w synthesis performance, yielding significantly higher SSIM, higher PSNR, and lower NRMSE against the ground-truth T1w images, than all compared methods (Fig. 8a). Moreover, the synthetic T1w images generated by dMRI-F-Syn enabled accurate cortical morphology estimation, as evidenced by significantly reduced cortical thickness diferences relative to the ground-truth (Fig. 8b). Generalization across the seven test datasets was then assessed quantitatively by comparing cortical morphometry between the synthesized and acquired T1w images. dMRI-F-Syn achieved the lowest mean cortical-thickness diference in every dataset (Fig. 8c). (Note that DeepAnat failed to produce cortical surface reconstructions in several datasets due to abnormal intensity values, though it achieved relatively small cortical-thickness diferences in the datasets where it succeeded.) Consistent with these quantitative findings, visual inspection showed that dMRI-F-Syn produced T1w images with sharper tissue boundaries and better-preserved local anatomical features (Fig. 8d). Collectively, these results demonstrate that dMRI-F-Syn efectively synthesizes T1w images that preserve anatomical information critical for both voxel-level reconstruction and downstream cortical-thickness estimation, even when applied to heterogeneous acquisitions with lower image quality.

## Outlook and Conclusion

This work presents dMRI-F, which learns generalizable representations from sparse DWI inputs to support data enhancement and downstream analyses across heterogeneous acquisitions and populations. The enhanced DWIs could broaden the use of sparsely sampled datasets for studying structural connectivity and tissue microstructure across development, aging and disease. The ability to generate DWIs at arbitrary difusion directions across multi shells could also support the training of other deep learning models with diferent sampling requirements. Beyond data enhancement, the learned representations could reduce reliance on additional anatomical scans and cross-modal registration in downstream analyses. These capabilities could enable more comprehensive difusion imaging analyses from shorter acquisitions, potentially reducing patient burden and improving accessibility. Building on these capabilities, future work could extend dMRI-F to populations with tumours and other lesions, with particular attention to the preservation of pathological anatomy. Further studies could also explore the potential of the learned representations for predicting disease status, severity and treatment response, extending their use beyond the parcellation, registration and synthesis tasks evaluated here. Finally, integration with scanner reconstruction workflows and prospective evaluation using shortened acquisition protocols could help establish the reliability and practical value of dMRI-F in clinical settings.

## Methods

The proposed dMRI-F is a foundation model for dMRI that performs DWI enhancement while supporting diverse downstream tasks across acquisition protocols and populations. It employs the popular self-supervised learning (SSL) framework with a Masked Autoencoder (MAE) <sup>52</sup> design. SSL has emerged as a powerful paradigm for training foundation models by transforming unlabeled data into supervised learning signals through carefully designed pretext objectives <sup>53–55</sup>. By exploiting the intrinsic structure of unlabeled data, SSL pre-training learns generalizable features that can be efectively transferred to a wide range of downstream tasks <sup>56,57</sup>. Among various SSL strategies, masked reconstruction, as exemplified by $\mathrm { M A E ^ { 5 2 } }$ , has proven particularly efective. Its success relies on two closely related components: a pre-training task that defines what information should be hidden and recovered, and a model architecture capable of exploiting the remaining observations to infer the missing content. Further, the learned representations can be transferred to other computational tasks beyond pre-training. In the rest, we first describe the design of our pre-training task and the model architecture used to solve it, followed by the fine-tuning process for adapting the learned latent features to downstream tasks.

## Pretraining for DWI Enhancement

## Pretraining Task

The goal of our pre-training is to reconstruct missing DWI volumes from sparse input, thereby enabling efective DWI enhancement. In natural images and standard 3D medical images, reconstruction is commonly achieved by masking image patches or volumetric blocks. dMRI, however, presents a diferent setting: the acquired data form a 4D volume spanning three spatial dimensions and an additional difusion-encoding dimension, where each DWI volume corresponds to a specific q-space sampling point determined by the gradient direction (b-vector) and strength (b-value) (see Fig. 1a). Following the masked reconstruction principle, we formulate the pre-training task as angular masked reconstruction, which recovers unobserved difusion signals from sparse observations.

Specifically, let $X \in \mathbb { R } ^ { H \times W \times D \times G }$ denote the complete set of DWI signals acquired at non-zero b-values, where H, W and D are the spatial dimensions, and G is the difusion-encoding dimension (i.e., the total number of acquired DWI volumes across all gradient directions and b-value shells). The b0 volumes are excluded from both the pretraining inputs and reconstruction targets but are retained for subsequent analyses, including microstructural estimation, FOD estimation, and tractography. We define a masking operator M that selects a sparse subset of q-space coordinates $q _ { s } .$ , yielding the q-space subsampled input volumes ${ \overline { { X } } } = M ( X )$ . The pre-training task is then defined as:

$$
\hat { X } ( q _ { t } ) = F _ { \theta } ( q _ { s } , \overline { { X } } , q _ { t } ) ,
$$

where $q _ { t }$ denotes the target q-space coordinate, and $F _ { \theta }$ learns to predict the missing DWIs from the sparse input. In our pre-training setting, the input $\overline { { X } }$ contains only 6 arbitrarily selected DWI volumes at $\mathrm { b } = 1 , 0 0 0 \mathrm { s } / \mathrm { m m } ^ { 2 }$ while the reconstruction target comprises the full DWI data, including 270 DWIs across $\mathrm { b } = 1 , 0 0 0 , 2 , 0 0 0$ , and $\mathrm { 3 , 0 0 0 s / m m ^ { 2 } }$

## Network Architecture

To solve the pre-training task, we design an SSL network based on the MAE architecture to recover the masked DWIs. In conventional spatial masked reconstruction tasks, the objective is typically to recover masked patches from the observed context by exploiting spatial dependencies within the image. Our angular masked reconstruction task, however, requires modeling additional context, because DWIs acquired under diferent q-space locations are not independent but rather reflect the same underlying tissue anatomy. Recovering the masked DWIs thus requires jointly modeling the relationships among spatial coordinates, underlying tissue anatomy, and q-space coordinates, motivating an architecture that integrates anatomical context with spatial and q-space coordinates.

Implicit Neural Representations $\mathrm { ( I N R s ) ^ { 5 8 } }$ are particularly suited to this requirement. INRs are a class of neural networks that learn continuous, coordinate-based mappings from input coordinates to output signals, enabling flexible and high-fidelity representation of complex data. Neural radiance fields (NeRF) <sup>59</sup> is a landmark implementation of the INR paradigm, where an INR is trained to represent a 3D scene as a continuous function of spatial coordinates and viewing directions, enabling scene prediction from novel viewpoints. The design of NeRF inspires our approach to modeling DWI signals as continuous functions of spatial and q-space coordinates. In related work, recent dMRI studies have demonstrated the success of INRs for continuously modeling fiber orientation distributions and data super-resolution <sup>60–63</sup>. However, these approaches typically require subject-specific optimization and are not designed for direct transfer across acquisition protocols or heterogeneous datasets. In our study, we developed dMRI-F as a unified pretrained framework that takes sparsely sampled DWI volumes as inputs and reconstructs complete DWI volumes at unobserved q-space locations. Through q-space masked reconstruction over a large cohort of training subjects, its encoder learns to extract structural representations consistently from unseen subjects and heterogeneous datasets without subject-specific optimization.

Specifically, we adopt an encoder–decoder architecture that learns a coordinate-conditioned mapping across subjects. Given the input DWI data X consisting of K volumes, the encoder $E _ { \phi }$ extracts a structural feature field Φ that captures anatomical information shared across DWIs. At each spatial location $x ,$ the implicit mapping network within the decoder models the relationship among the local structural feature $\Phi ( x )$ , the $K$ corresponding q-space coordinates $\eta ( q _ { s } ) _ { s = 1 } ^ { K } .$ , and the target q-space coordinate $q _ { t }$ . This formulation allows the model to adapt its prediction to subject-specific anatomy through the encoded feature field while avoiding persubject optimization. The implicit mapping module $G _ { \psi }$ then predicts a voxel-wise latent code for each target spatial and q-space coordinate:

$$
\begin{array} { r } { z ( x , q _ { t } ) = G _ { \psi } ( \gamma ( x ) , \eta ( q _ { t } ) ; \Phi ( x ) , \eta ( q _ { s } ) _ { s = 1 } ^ { K } ) , } \end{array}
$$

where $\gamma ( \cdot )$ and $\eta ( \cdot )$ denote the spatial and angular coordinate embeddings, respectively.

For the angular coordinate, we encode the queried difusion direction using a real spherical-harmonics (SH) expansion:

$$
\eta ( q ) = \mathrm { S H } _ { L } ( q ) = [ Y _ { 0 , 0 } ^ { \mathrm { r e a l } } ( q ) , Y _ { 1 , - 1 } ^ { \mathrm { r e a l } } ( q ) , . . . , Y _ { l , m } ^ { \mathrm { r e a l } } ( q ) , Y _ { L - 1 , L - 1 } ^ { \mathrm { r e a l } } ( q ) ] ,
$$

where $Y _ { l , m } ^ { \mathrm { r e a l } } ( q )$ denotes the real SH basis function. We set $L \ = \ 4$ , which provides a compact and smooth representation of angular variation on the sphere. For the spatial coordinate, we use the global voxel coordinate x in the DWI volume, normalize each dimension to $[ - 1 , 1 ]$ , and apply a sinusoidal positional encoding:

$$
\gamma ( x ) = \mathrm { S I N } ( x ) = [ x , \sin ( 2 ^ { 0 } \pi x ) , \cos ( 2 ^ { 0 } \pi x ) , . . . , \sin ( 2 ^ { N - 1 } \pi x ) , \cos ( 2 ^ { N - 1 } \pi x ) ] ,
$$

where N denotes the number of spatial frequency bands.

The resulting latent codes are assembled and refined by a volumetric decoder to reconstruct the highresolution DWI at the target q-space location.

## Pretraining Loss

To preserve fine-grained image details, the model is optimized using a combination of a reconstruction loss and a perceptual loss:

$$
\begin{array} { r } { L = L _ { 1 } ^ { \mathrm { D W I } } + \lambda L _ { \mathrm { p e r c } } ^ { \mathrm { D W I } } , } \end{array}
$$

where λ denotes the weighting factor for the perceptual loss and is set to 0.01. The voxel-wise reconstruction loss is computed as:

$$
L _ { 1 } ^ { \mathrm { D W I } } = \| \hat { X } _ { q _ { t } } - X _ { q _ { t } } \| _ { 1 } ,
$$

where $\hat { X } _ { q _ { t } }$ and $X _ { q _ { t } }$ denote the reconstructed and reference DWIs at the target q-space coordinate q<sub>t</sub>, respectively. The perceptual loss is defined as

$$
L _ { \mathrm { p e r c } } ^ { \mathrm { D W I } } = \sum _ { l \in S } \lVert \sigma _ { l } ( \hat { X } _ { q _ { t } } ) - \sigma _ { l } ( X _ { q _ { t } } ) \rVert _ { 1 } ,
$$

where $\sigma _ { l } ( \cdot )$ denotes the feature representation extracted at layer l, and S is the set of selected feature layers.

## Fine-Tuning for Downstream Tasks

Following q-space masked reconstruction pre-training, we adapted the learned feature representations to three downstream tasks: brain parcellation, brain registration, and cross-modal synthesis. For each task, the pretrained encoder served as a feature extractor. At each training iteration, six DWI volumes were randomly selected from the $\mathrm { b } = 1 , 0 0 0 \mathrm { s } / \mathrm { m m } ^ { 2 }$ shell of each subject and used as encoder inputs. A task-specific decoder was then fine-tuned on a subset of 50 HCP-YA subjects randomly selected from the pre-training set. For the decoder, we adopted a U-Net architecture optimized with a task-specific loss function. To evaluate performance, we compared against SOTA methods in each task, on the high-quality HCP-YA dataset and the eight independently acquired test datasets with clinically typical acquisitions.

## Brain Parcellation

For brain parcellation, the goal was to predict a voxel-wise anatomical label from the input DWI volumes. In our study, we trained a parcellation decoder that mapped the extracted features to a parcellation comprising 101 anatomical labels corresponding to the FreeSurfer Desikan–Killiany (DK) parcellation <sup>48</sup>. The decoder was trained using a combination of cross-entropy and Dice losses:

$$
L _ { \mathrm { P a r c } } = L _ { \mathrm { C E } } + L _ { \mathrm { D i c e } } ,
$$

where the cross-entropy term $L _ { \mathrm { C E } }$ provided voxel-wise supervision for anatomical classification, and the Dice term $L _ { \mathrm { D i c e } }$ promoted spatial overlap between the predicted and reference regions while reducing the influence of class imbalance.

We compared the SOTA brain parcellation methods, FastSurfer <sup>49</sup> and DDParcel<sup>27</sup>. FastSurfer aggregates information from neighboring T1w slices and combines predictions across three anatomical planes to generate whole-brain parcellations. DDParcel integrates complementary microstructural information from difusionderived parameter maps at multiple levels of the network to generate high-quality anatomical parcellations. For evaluation, we adopted strategies from a previous DWI parcellation study <sup>27</sup>. For HCP-YA, the advanced acquisition protocol supported accurate registration between anatomical T1w images and DWIs. We therefore used FreeSurfer parcellations derived from T1w images and registered to DWI space as the reference, and quantified performance using Dice. For the testing datasets with clinically typical acquisitions, lower image quality and less reliable cross-modality registration prevented the construction of comparably accurate reference parcellations in DWI space. We therefore assessed regional homogeneity using the relative standard deviation (RSD) of FA, defined as the within-region standard deviation divided by its mean. Lower RSD values indicate greater anatomical and microstructural coherence within the predicted regions.

## Brain Registration

For brain registration, the goal was to estimate a non-linear deformation field aligning the DWI data of a moving subject to a fixed subject. To estimate this field, the features extracted from the moving and fixed subjects were jointly processed by a registration network. The predicted deformation field was then applied to the entire moving-subject DWI series to warp it into fixed-subject space. The network was optimized using image-similarity and deformation-smoothness losses:

$$
L _ { \mathrm { R e g } } = L _ { \mathrm { M S E } } + L _ { \mathrm { g r a d } } ,
$$

where the MSE term $L _ { \mathrm { M S E } }$ promotes intensity agreement between the warped moving and fixed b0 images, and the gradient term $L _ { \mathrm { g r a d } }$ constrains spatial variations in the displacement field to encourage smooth and anatomically plausible deformations.

Registration performance was evaluated against the SOTA registration methods: VoxelMorph <sup>50</sup> and Synth-Morph <sup>51</sup>. VoxelMorph learns a direct mapping from image pairs to dense deformation fields using a convolutional neural network, enabling rapid deformable registration in a single forward pass. SynthMorph learns registration from images synthesized with diverse contrasts and deformations, using a label-overlap objective to promote anatomical alignment independent of image intensities. This training strategy supports registration across MRI contrasts, including those not encountered during training. Registration performance was evaluated at complementary anatomical and white-matter tract levels. For HCP-YA, FreeSurfer-derived cortical, subcortical, and white-matter parcellations from each moving subject were warped into fixed-subject space using the deformation field predicted by each method. Their spatial overlap with the corresponding fixed-subject parcellations was quantified using the Dice coeficient. To assess tract alignment across the complete difusion acquisition, each deformation field was also applied to all moving-subject FOD volumes. TractSeg was then applied to the warped moving-subject FODs and the fixed-subject FODs, and the resulting tract segmentations were compared using the same metrics. For the eight test datasets, registration performance was assessed using normalized cross-correlation (NCC) between the warped moving and fixed b0 images, complemented by visual inspection of local anatomical correspondence.

## Cross-Modal Synthesis

In cross-modal synthesis, anatomical T1-weighted (T1w) images were predicted directly from DWI volumes. The synthesis decoder mapped the extracted features to the corresponding spatially aligned T1w image and was optimized using a combination of voxel-wise and perceptual losses:

$$
L _ { \mathrm { S y n } } = L _ { 1 } ^ { \mathrm { T 1 w } } + \lambda L _ { \mathrm { p e r c } } ^ { \mathrm { T 1 w } } ,
$$

where λ is set to 0.01. The $L _ { 1 } ^ { \mathrm { T 1 w } }$ term enforces voxel-wise intensity agreement between the synthesized and reference T1-weighted images, whereas the perceptual term promotes the preservation of higher-level anatomical features.

For T1w image synthesis, we compared with the Brain-ID <sup>31</sup> and DeepAnat<sup>32</sup>. Brain-ID learns contrastagnostic anatomical representations from synthetic inputs with varying contrasts, resolutions, and corruption levels, supporting T1w synthesis through a lightweight task-specific output layer. In our comparison, b0 images were used as the input. DeepAnat synthesizes T1w images directly from difusion data using generative adversarial network architectures. The synthesized images retain spatial correspondence with the difusion data, supporting anatomical segmentation. For HCP-YA, accurately aligned T1w reference images enabled voxel-wise evaluation of the three methods using SSIM, PSNR and NRMSE. Preservation of cortical morphology was further assessed by applying FastSurfer to the synthesized T1w images and comparing the resulting cortical thickness maps with the FreeSurfer-derived HCP-YA references. Cortical thickness diferences were also visualized for a representative subject. For the independent datasets, inconsistent alignment between the acquired T1w images and DWIs precluded reliable voxel-wise comparison. We therefore applied FastSurfer to both the synthesized and acquired T1w images and compared the resulting cortical thickness estimates to assess anatomical fidelity at the cortical-surface level.

## Datasets and Implementation

## Training Data

We selected the HCP-YA data as the pre-training dataset and included 200 young healthy adults (age, $2 8 . 8 4 \pm 3 . 5 1$ years; 118 females and 82 males). Of these, 180 subjects were used for training and 20 for validation (see Extended Data Fig. 1 for the ablation study on training data size). Imaging was performed on a customized Siemens 3 T Connectome Skyra scanner using the high-quality HCP acquisition protocol. The dMRI data were acquired with TE = 89.5 ms, TR = 5,520 ms and a 1.25-mm isotropic voxel size. Each subject had 288 volumes, comprising 18 b0 images and 270 DWIs distributed equally across three shells at b = 1,000, 2,000, and $\mathrm { 3 , 0 0 0 s / m m ^ { 2 } }$ . Images were acquired with both left-to-right and right-to-left phase encoding to facilitate correction of echo-planar imaging distortions. The $\mathrm { b } = 1 , 0 0 0 \mathrm { s } / \mathrm { m m } ^ { 2 }$ shell was used to construct the sparse input, reflecting its broad availability in conventional and legacy dMRI protocols, whereas the complete multi-shell acquisition served as the prediction target. The corresponding T1w images were acquired with TE = 2.14 ms, TR = 2,400 ms and a 0.7-mm isotropic voxel size. We used data processed with the HCP minimal preprocessing pipelines, which corrected EPI and eddy-current distortions, subject motion and gradient nonlinearities, and registered the dMRI data to the native T1w structural space at 1.25-mm isotropic resolution. This processing provided precise within-subject alignment between the dMRI and T1-weighted images. The minimally preprocessed DWIs were further denoised using $\mathrm { P a t c h 2 S e l f ^ { 2 4 } }$ to suppress residual noise and improve the clarity of anatomical structures.

## Testing Data

To evaluate the generalizability of dMRI-F beyond the pre-training data, we used the HCP-YA test set together with eight test datasets: ABIDE <sup>64</sup>, ADNI<sup>46</sup>, ADNI-DOD<sup>65</sup>, HABS<sup>66</sup>, ICBM<sup>67</sup>, SCAN<sup>68</sup>, NIFD<sup>69</sup>, and PPMI<sup>70</sup>. Together, the HCP-YA test set and the eight test datasets span a broad age range and include healthy participants as well as individuals with neurodevelopmental, neurodegenerative, psychiatric, and trauma-related conditions, with data acquired using diverse scanners and difusion protocols. Detailed sample sizes, demographic and diagnostic characteristics, scanner information, and acquisition parameters are provided in Supplementary Table 3. The original studies were approved by their respective institutional review boards or ethics committees, and all de-identified data were used in accordance with the corresponding data-use agreements. Participants in the HCP-YA test set and the eight test datasets did not overlap with those used for pre-training. Data from the HCP-YA test set were processed using the same HCP minimal preprocessing pipelines <sup>71</sup> described above. For the test datasets available in DICOM format, the images were converted to NIfTI while preserving the associated b-values and difusion gradient directions. All dMRI data from the eight test datasets underwent correction of motion- and eddy-current-induced distortions using DTIPrep (v1.2.11) <sup>72</sup>. Brain masks were generated using CNN-Difusion-MRIBrain-Segmentation (v0.3) <sup>73</sup>. Patch2Self was then applied to the complete difusion series to generate the denoised reference data, whereas the six sparse input DWIs were selected from the corresponding undenoised acquisitions. Finally, all dMRI data from the eight test datasets underwent visual quality assessment.

## Implementation

The input DWIs were min–max normalized using the intensities across DWIs for each subject. During pretraining, we randomly reduced the spatial resolution of the input DWIs as data augmentation to promote the learning of more generalizable representations. All models except the registration model received patches of 64 $\times ~ 6 4 ~ \times ~ 6 4$ voxels as input. For registration, the extracted feature patches were first reassembled into wholebrain feature volumes to preserve spatial continuity, then processed by the registration network. The structural encoder, volumetric decoder, and task-specific decoders for all downstream tasks were implemented using 3D U-Net architectures. The learning rate was set to $1 \times 1 0 ^ { - 4 }$ . All models were implemented in PyTorch, and training and inference were performed on NVIDIA RTX 3090 GPUs.

## Data Availability

The data used in this study are available from the Human Connectome Project Young Adult (HCP-YA; https://www.humanconnectome.org/study/hcp-young-adult/data-releases), Autism Brain Imaging Data Exchange (ABIDE; https://fcon 1000.projects.nitrc.org/indi/abide/), Alzheimer’s Disease Neuroimaging Initiative (ADNI; https://adni.loni.usc.edu/data-samples/adnidata/), Department of Defense Alzheimer’s Disease Neuroimaging Initiative (DOD ADNI; https://ida.loni.usc.edu/collaboration/access/appApply.jsp?project=ADNIDOD), Harvard Aging Brain Study (HABS; https://www.synapse.org/habs), International Consortium for Brain Mapping (ICBM; https://ida.loni.usc.edu/collaboration/access/appApply.jsp?project=ICBM), Standardized Centralized Alzheimer’s and Related Dementias Neuroimaging initiative (SCAN; https://scan.naccdata.org), Neuroimaging in Frontotemporal Dementia study (NIFD; https://ida.loni.usc.edu/collaboration/access/appApply.jsp?project=NIFD), and Parkinson’s Progression Markers Initiative (PPMI; https://www.ppmi-info.org/access-data-specimens/download-data/). Access to controlled datasets is subject to registration, approval, and the respective data-use agreements. No new datasets were generated in this study.

## Code Availability

The code for pretraining, fine-tuning, and evaluation of the foundation model is available in GitHub at https://github.com/XiZhu-UE/dMRI-F.

## References

[1] Basser, P. J., Mattiello, J. & LeBihan, D. MR difusion tensor spectroscopy and imaging. Biophys. J. 66, 259–267 (1994).

[2] Basser, P. J., Pajevic, S., Pierpaoli, C., Duda, J. & Aldroubi, A. In vivo fiber tractography using DT-MRI data. Magn. Reson. Med. 44, 625–632 (2000).

[3] Alexander, D. C., Dyrby, T. B., Nilsson, M. & Zhang, H. Imaging brain microstructure with difusion MRI: practicality and applications. NMR Biomed. 32, e3841 (2019).

[4] Zhang, F. et al. Quantitative mapping of the brain’s structural connectivity using difusion MRI tractography: A review. Neuroimage 249, 118870 (2022).

[5] Lebel, C. & Beaulieu, C. Longitudinal development of human brain wiring continues from childhood into adulthood. J. Neurosci. 31, 10937–10947 (2011).

[6] Yeatman, J. D., Wandell, B. A. & Mezer, A. A. Lifespan maturation and degeneration of human brain white matter. Nat. Commun. 5, 4932 (2014).

[7] Scholz, J., Klein, M. C., Behrens, T. E. J. & Johansen-Berg, H. Training induces changes in white-matter architecture. Nat. Neurosci. 12, 1370–1371 (2009).

[8] Penke, L. et al. Brain white matter tract integrity as a neural foundation for general intelligence. Mol. Psychiatry 17, 1026–1030 (2012).

[9] Schaefer, P. W., Grant, P. E. & Gonzalez, R. G. Difusion-weighted MR imaging of the brain. Radiology 217, 331–345 (2000).

[10] Ciccarelli, O., Catani, M., Johansen-Berg, H., Clark, C. & Thompson, A. Difusion-based tractography in neurological disorders: concepts, applications, and future developments. Lancet Neurol. 7, 715–727 (2008).

[11] Essayed, W. I. et al. White matter tractography for neurosurgical planning: A topography-based review of the current state of the art. NeuroImage Clin. 15, 659–672 (2017).

[12] Van Essen, D. C. et al. The WU-Minn Human Connectome Project: an overview. Neuroimage 80, 62–79 (2013).

[13] Yeh, F.-C. Population-based tract-to-region connectome of the human brain and its hierarchical topology. Nat. Commun. 13, 4933 (2022).

[14] Kim, M. E. et al. White matter micro- and macrostructure brain charts for the human lifespan. Nature 655, 979–989 (2026).

[15] Zhang, F. et al. An anatomically curated fiber clustering white matter atlas for consistent white matter tract parcellation across the lifespan. Neuroimage 179, 429–447 (2018).

[16] O’Donnell, L. J. & Westin, C.-F. An introduction to difusion tensor image analysis. Neurosurg. Clin. N. Am. 22, 185–96, viii (2011).

[17] Zhu, P., Fu, Y., Chen, N. & Qiu, A. Q-space guided multi-modal translation network for difusion-weighted image synthesis. IEEE Trans. Med. Imaging 45, 1167–1178 (2026).

[18] Ren, M., Kim, H., Dey, N. & Gerig, G. Q-space conditioned translation networks for directional synthesis of difusion weighted images from multi-modal structural MRI. Med. Image Comput. Comput. Assist. Interv. 12907, 530–540 (2021).

[19] Lyon, M., Armitage, P. & Alvarez, M. A. Spatio-angular convolutions for super-resolution in difusion MRI.<sup>´</sup> Adv. Neural Inf. Process. Syst. 36, 12457–12475 (2023).

[20] Qin, Y. et al. Multimodal super-resolved q-space deep learning. Med. Image Anal. 71, 102085 (2021).

[21] Tian, Q. et al. SRDTI: deep learning-based super-resolution for difusion tensor MRI. arXiv:2102.09069 (2021).

[22] Tanno, R. et al. Bayesian image quality transfer with CNNs: exploring uncertainty in dMRI superresolution. In Medical Image Computing and Computer Assisted Intervention–MICCAI 2017, Lecture Notes in Computer Science 10433, 611–619 (2017).

[23] Veraart, J. et al. Denoising of difusion MRI using random matrix theory. Neuroimage 142, 394–406 (2016).

[24] Fadnavis, S., Batson, J. & Garyfallidis, E. Patch2Self: denoising difusion MRI with self-supervised learning. Adv. Neural Inf. Process. Syst. 33 (2020).

[25] Karimi, D. et al. Detailed delineation of the fetal brain in difusion MRI via multi-task learning. IEEE Trans. Med. Imaging 45, 1208–1222 (2026).

[26] Theaud, G. et al. DORIS: a difusion MRI-based 10 tissue class deep learning segmentation algorithm tailored to improve anatomically-constrained tractography. Front. Neuroimaging 1, 917806 (2022).

[27] Zhang, F. et al. DDParcel: deep learning anatomical brain parcellation from difusion MRI. IEEE Trans. Med. Imaging 43, 1191–1202 (2024).

[28] Wang, Y. et al. Difusion tensor magnetic resonance image registration based on Parallel Dual-channel VoxelMorph. IEEE J. Biomed. Health Inform. 30, 6018–6029 (2026).

[29] Bouza, J. J., Yang, C.-H. & Vemuri, B. C. Geometric deep learning for unsupervised registration of difusion magnetic resonance images. Inf. Process. Med. Imaging 13939, 563–575 (2023).

[30] Zhang, F., Wells, W. M. & O’Donnell, L. J. Deep difusion MRI registration (DDMReg): a deep learning method for difusion MRI registration. IEEE Trans. Med. Imaging 41, 1454–1467 (2022).

[31] Liu, P., Puonti, O., Hu, X., Alexander, D. C. & Iglesias, J. E. Brain-ID: learning contrast-agnostic anatomical representations for brain imaging. In Computer Vision–ECCV 2024, Lecture Notes in Computer Science 15070, 322–340 (2024).

[32] Li, Z. et al. Difusion MRI data analysis assisted by deep learning synthesized anatomical images (DeepAnat). Med. Image Anal. 86, 102744 (2023).

[33] Tax, C. M. et al. Cross-scanner and cross-protocol difusion MRI data harmonisation: A benchmark database and evaluation of algorithms. Neuroimage 195, 285–299 (2019).

[34] Cetin Karayumak, S. et al. Retrospective harmonization of multi-site difusion MRI data acquired with diferent acquisition parameters. Neuroimage 184, 180–200 (2019).

[35] Ghamizi, S. et al. Foundation models for brain imaging: A systematic review. Neuroimage 333, 121877 (2026).

[36] Sun, Y., Wang, L., Li, G., Lin, W. & Wang, L. A foundation model for enhancing magnetic resonance images and downstream segmentation, registration and diagnostic tasks. Nat. Biomed. Eng. 9, 521–538 (2025).

[37] Li, C. et al. DDSurfer: A weakly-supervised dual-stream deep learning framework for cortical surface reconstruction from difusion MRI. Adv. Sci. (Weinh.) e76596 (2026).

[38] Xue, T. et al. Superficial white matter analysis: An eficient point-cloud-based deep learning framework with supervised contrastive learning for consistent tractography parcellation across populations and dMRI acquisitions. Med. Image Anal. 85, 102759 (2023).

[39] Zhang, F. et al. Deep white matter analysis (DeepWMA): Fast and consistent tractography segmentation. Med. Image Anal. 65, 101761 (2020).

[40] Wasserthal, J., Neher, P. & Maier-Hein, K. H. TractSeg: fast and accurate white matter tract segmentation. Neuroimage 183, 239–253 (2018).

[41] Lyon, M., Armitage, P. & Alvarez, M. A. Angular super-resolution in difusion MRI with a 3D recurrent<sup>´</sup> convolutional autoencoder. Proc. Mach. Learn. Res. 172, 834–846 (2022).

[42] Jensen, J. H., Helpern, J. A., Ramani, A., Lu, H. & Kaczynski, K. Difusional kurtosis imaging: the quantification of non-Gaussian water difusion by means of magnetic resonance imaging. Magn. Reson. Med. 53, 1432–1440 (2005).

[43] Tournier, J.-D. et al. MRtrix3: A fast, flexible and open software framework for medical image processing and visualisation. Neuroimage 202, 116137 (2019).

[44] O’Donnell, L. J. & Westin, C.-F. Automatic tractography segmentation using a high-dimensional white matter atlas. IEEE Trans. Med. Imaging 26, 1562–1575 (2007).

[45] Kochunov, P. et al. Heritability of fractional anisotropy in human white matter: a comparison of Human Connectome Project and ENIGMA-DTI data. Neuroimage 111, 300–311 (2015).

[46] Veitch, D. P. et al. Using the Alzheimer’s Disease Neuroimaging Initiative to improve early detection, diagnosis, and treatment of Alzheimer’s disease. Alzheimers. Dement. 18, 824–857 (2022).

[47] Nir, T. M. et al. Efectiveness of regional DTI measures in distinguishing Alzheimer’s disease, MCI, and normal aging. NeuroImage Clin. 3, 180–195 (2013).

[48] Desikan, R. S. et al. An automated labeling system for subdividing the human cerebral cortex on MRI scans into gyral based regions of interest. Neuroimage 31, 968–980 (2006).

[49] Henschel, L. et al. FastSurfer - A fast and accurate deep learning based neuroimaging pipeline. Neuroimage 219, 117012 (2020).

[50] Balakrishnan, G., Zhao, A., Sabuncu, M. R., Guttag, J. & Dalca, A. V. VoxelMorph: A learning framework for deformable medical image registration. IEEE Trans. Med. Imaging 38, 1788–1800 (2019).

[51] Hofmann, M. et al. SynthMorph: Learning contrast-invariant registration without acquired images. IEEE Trans. Med. Imaging 41, 543–558 (2022).

[52] He, K. et al. Masked Autoencoders Are Scalable Vision Learners. in 2022 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR) 15979–15988 (2022).

[53] Bommasani, R. et al. On the opportunities and risks of foundation models. arXiv:2108.07258 (2021).

[54] Jing, L. & Tian, Y. Self-supervised visual feature learning with deep neural networks: A survey. IEEE Trans. Pattern Anal. Mach. Intell. 43, 4037–4058 (2021).

[55] Gui, J. et al. A survey on self-supervised learning: Algorithms, applications, and future trends. IEEE Trans. Pattern Anal. Mach. Intell. 46, 9052–9071 (2024).

[56] He, K., Fan, H., Wu, Y., Xie, S. & Girshick, R. Momentum contrast for unsupervised visual representation learning. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 9729–9738 (2020).

[57] Chen, T., Kornblith, S., Norouzi, M. & Hinton, G. A simple framework for contrastive learning of visual representations. Proc. Mach. Learn. Res. 119, 1597–1607 (2020).

[58] Sitzmann, V., Martel, J. N. P., Bergman, A. W., Lindell, D. B. & Wetzstein, G. Implicit neural representations with periodic activation functions. Adv. Neural Inf. Process. Syst. 33, 7462–7473 (2020).

[59] Mildenhall, B. et al. NeRF: representing scenes as neural radiance fields for view synthesis. Commun. ACM 65, 99–106 (2022).

[60] Ghandoura, A., Zakour, M., Consagra, W. & Rathi, Y. Prior-guided implicit neural representations for single-subject difusion MRI super-resolution. arXiv:2609.00981 (2026).

[61] Zeng, T. et al. MINeR: Direction-modulated implicit neural representation enables ultrafast multi-shell difusion MRI. Med. Image Anal. 113, 104205 (2026).

[62] Hendriks, T., Consagra, W., Vilanova, A., Rathi, Y. & Chamberland, M. SIINR: structurally informed implicit neural representations for super-resolution with uncertainty quantification of clinical quality difusion MRI datasets. arXiv:2607.19943 (2026).

[63] Hendriks, T., Vilanova, A. & Chamberland, M. Implicit neural representation of multi-shell constrained spherical deconvolution for continuous modeling of difusion MRI. Imaging Neurosci. (Camb.) 3, imag a 00501 (2025).

[64] Di Martino, A. et al. The autism brain imaging data exchange: towards a large-scale evaluation of the intrinsic brain architecture in autism. Mol. Psychiatry 19, 659–667 (2014).

[65] Weiner, M. W. et al. Efects of traumatic brain injury and posttraumatic stress disorder on Alzheimer’s disease in veterans, using the Alzheimer’s Disease Neuroimaging Initiative. Alzheimers. Dement. 10, S226–S235 (2014).

[66] Dagley, A. et al. Harvard Aging Brain Study: Dataset and accessibility. Neuroimage 144, 255–258 (2017).

[67] Mazziotta, J. et al. A probabilistic atlas and reference system for the human brain: International Consortium for Brain Mapping (ICBM). Philos. Trans. R. Soc. Lond. B Biol. Sci. 356, 1293–1322 (2001).

[68] Standardized Centralized Alzheimer’s and Related Dementias Neuroimaging (SCAN). https://scan.naccdata.org.

[69] Stafaroni, A. M. et al. Longitudinal multimodal imaging and clinical endpoints for frontotemporal dementia clinical trials. Brain 142, 443–459 (2019).

[70] Marek, K. et al. The Parkinson’s Progression Markers Initiative (PPMI): establishing a PD biomarker cohort. Ann. Clin. Transl. Neurol. 5, 1460–1477 (2018).

[71] Glasser, M. F. et al. The minimal preprocessing pipelines for the Human Connectome Project. Neuroimage 80, 105–124 (2013).

[72] Oguz, I. et al. DTIPrep: quality control of difusion-weighted images. Front. Neuroinform. 8, 4 (2014).

[73] Cetin-Karayumak, S. et al. Harmonized difusion MRI data and white matter measures from the Adolescent Brain Cognitive Development Study. Sci. Data 11, 249 (2024).

## Acknowledgment

The NACC database is funded by NIA/NIH Grant U24 AG072122. SCAN is a multi-institutional project that was funded as a U24 grant (AG067418) by the National Institute on Aging in May 2020. Data collected by SCAN and shared by NACC are contributed by the NIA-funded ADRCs as follows:

Arizona Alzheimer’s Center - P30 AG072980 (PI: Eric Reiman, MD); R01 AG069453 (PI: Eric Reiman (contact), MD); P30 AG019610 (PI: Eric Reiman, MD); and the State of Arizona which provided additiona funding supporting our center; Boston University - P30 AG013846 (PI Neil Kowall MD); Cleveland ADRC - P30 AG062428 (James Leverenz, MD); Cleveland Clinic, Las Vegas - P20AG068053; Columbia - P50 AG008702 (PI Scott Small MD); Duke/UNC ADRC - P30 AG072958; Emory University - P30AG066511 (PI Levey Allan, MD, PhD); Indiana University - R01 AG19771 (PI Andrew Saykin, PsyD); P30 AG10133 (PI Andrew Saykin, PsyD); P30 AG072976 (PI Andrew Saykin, PsyD); R01 AG061788 (PI Shannon Risacher, PhD); R01 AG053993 (PI Yu-Chien Wu, MD, PhD); U01 AG057195 (PI Liana Apostolova, MD); U19 AG063911 (PI Bradley Boeve, MD); and the Indiana University Department of Radiology and Imaging Sciences; Johns Hopkins - P30 AG066507 (PI Marilyn Albert, Phd.); Mayo Clinic - P50 AG016574 (PI Ronald Petersen MD PhD); Mount Sinai - P30 AG066514 (PI Mary Sano, PhD); R01 AG054110 (PI Trey Hedden, PhD); R01 AG053509 (PI Trey Hedden, PhD); New York University - P30AG066512-01S2 (PI Thomas Wisniewski, MD); R01AG056031 (PI Ricardo Osorio, MD); R01AG056531 (PIs Ricardo Osorio, MD; Girardin Jean-Louis, PhD); Northwestern University - P30 AG013854 (PI Robert Vassar PhD); R01 AG045571 (PI Emily Rogalski, PhD); R56 AG045571, (PI Emily Rogalski, PhD); R01 AG067781, (PI Emily Rogalski, PhD); U19 AG073153, (PI Emily Rogalski, PhD); R01 DC008552, (M.-Marsel Mesulam, MD); R01 AG077444, (PIs M.-Marsel Mesulam, MD, Emily Rogalski, PhD); R01 NS075075 (PI Emily Rogalski, PhD); R01 AG056258 (PI Emily Rogalski, PhD); Oregon Health and Science University - P30 AG066518 (PIs Lisa Silbert, MD, Kevin Duf, PhD); R56 AG074321 (PI Jefrey Kaye, MD); Rush University - P30 AG010161 (PI David Bennett MD); Stanford - P30AG066515; P50 AG047366 (PI Victor Henderson MD MS); University of Alabama, Birmingham - P20; University of California, Davis - P30 AG10129 (PI Charles DeCarli, MD); P30 AG072972 (PI Charles DeCarli, MD); University of California, Irvine - P50 AG016573 (PI Frank LaFerla PhD); University of California, San Diego - P30AG062429 (PI James Brewer, MD, PhD); University of California, San Francisco - P30 AG062422 (Rabinovici, Gil D., MD); University of Kansas - P30 AG035982 (Russell Swerdlow, MD); University of Kentucky - P30 AG028283-15S1 (PIs Linda Van Eldik, PhD and Brian Gold, PhD); University of Michigan ADRC - P30AG053760 (PI Henry Paulson, MD, PhD) P30AG072931 (PI Henry Paulson, MD, PhD) Cure Alzheimer’s Fund 200775 - (PI Henry Paulson, MD, PhD) U19 NS120384 (PI Charles DeCarli, MD, University of Michigan Site PI Henry Paulson, MD, PhD) R01 AG068338 (MPI Bruno Giordani, PhD, Carol Persad, PhD, Yi Murphey, PhD) S10OD026738-01 (PI Douglas Noll, PhD) R01 AG058724 (PI Benjamin Hampstead, PhD) R35 AG072262 (PI Benjamin Hampstead, PhD) W81XWH2110743 (PI Benjamin Hampstead, PhD) R01 AG073235 (PI Nancy Chiaravalloti, University of Michigan Site PI Benjamin Hampstead, PhD) 1I01RX001534 (PI Benjamin Hampstead, PhD) IRX001381 (PI Benjamin Hampstead, PhD); University of New Mexico - P20 AG068077 (Gary Rosenberg, MD); University of Pennsylvania - State of PA project 2019NF4100087335 (PI David Wolk, MD); Rooney Family Research Fund (PI David Wolk, MD); R01 AG055005 (PI David Wolk, MD); University of Pittsburgh - P50 AG005133 (PI Oscar Lopez MD); University of Southern California - P50 AG005142 (PI Helena Chui MD); University of Washington - P50 AG005136 (PI Thomas Grabowski MD); University of Wisconsin - P50 AG033514 (PI Sanjay Asthana MD FRCP); Vanderbilt University - P20 AG068082; Wake Forest - P30AG072947 (PI Suzanne Craft, PhD); Washington University, St. Louis - P01 AG03991 (PI John Morris MD); P01 AG026276 (PI John Morris MD); P20 MH071616 (PI Dan Marcus); P30 AG066444 (PI John Morris MD); P30 NS098577 (PI Dan Marcus); R01 AG021910 (PI Randy Buckner); R01 AG043434 (PI Catherine Roe); R01 EB009352 (PI Dan Marcus); UL1 TR000448 (PI Brad Evanof); U24 RR021382 (PI Bruce Rosen); Avid Radiopharmaceuticals / Eli Lilly; Yale - P50 AG047270 (PI Stephen Strittmatter MD PhD); R01AG052560 (MPI: Christopher van Dyck, MD; Richard Carson, PhD); R01AG062276 (PI: Christopher van Dyck, MD); Florida - P30AG066506-03 (PI Glenn Smith, PhD); P50 AG047266 (PI Todd Golde MD PhD)

## Funding

This work is in part supported by the National Natural Science Foundation of China (No. 62371107) and the National Key R&D Program of China (No. 2023YFE0118600).

## Author Contribution

Xi Zhu contributed to conceptualization, data curation, investigation, methodology, supervision, validation, visualization, and writing of the original draft, and reviewed and edited the manuscript. Yijie Li contributed to data curation and reviewed and edited the manuscript. Xiaofan Wang, Wei Zhang, Huafu Chen and Ce Zhu reviewed and edited the manuscript. Fan Zhang contributed to methodology, project administration, supervision, and writing of the original draft, and reviewed and edited the manuscript.

## Competing Interests

The authors declare no competing interests.

## Extended Data

Extended Data Table 1 — Ablation study of spherical harmonics encoding and the volumetric decoder.
<table><tr><td>SH-Encoding</td><td>Decoder</td><td>SSIM↑</td><td>PSNR↑</td><td>NRMSE↓</td></tr><tr><td>√</td><td>–</td><td> $0 . 9 7 6 8 { \scriptstyle \pm 0 . 0 0 5 6 }$ </td><td> $3 5 . 8 3 4 5 { \pm } 1 . 6 5 1 $ </td><td> $0 . 0 1 6 4 { \pm } 0 . 0 0 3 1$ </td></tr><tr><td></td><td>√</td><td> $0 . 9 7 2 1 { \scriptstyle \pm 0 . 0 0 7 1 }$ </td><td> $3 4 . 8 7 1 5 { \pm } 1 . 3 4 4$ </td><td> $0 . 0 1 8 3 { \pm } 0 . 0 0 2 8$ </td></tr><tr><td>√</td><td>√</td><td> $\mathbf { 0 . 9 8 2 7 { \pm } 0 . 0 0 2 0 }$ </td><td> $\mathbf { 3 7 . 1 9 5 9 { \pm 0 . 4 1 0 } }$ </td><td> $\mathbf { 0 . 0 1 3 8 7 { \pm 0 . 0 0 6 } }$ </td></tr></table>

DWI prediction performance on the HCP-YA test set was evaluated after separately removing the SH encoding or the volumetric decoder. In the variant without SH encoding, q-space coordinates were passed directly to the decoder without SH transformation. In the variant without the volumetric decoder, the decoder’s implicit mapping network directly predicted voxel-wise signal intensities to form the target DWI, rather than generating latent codes for subsequent volumetric decoding. The complete model achieved higher SSIM and PSNR and lower NRMSE than either ablated variant, supporting the contribution of both components to DWI prediction accuracy.

![](images/adc6b31e4e63e889f894a43fdf98c7d17e71e324d468ba8b5697eb6e6216a972.jpg)

![](images/ea8bb1cdc0bc907a826b5a1e448850616b80591b1a292d1b68a2bb76657a52af.jpg)

![](images/b7b53bce1351b9d9dfd5c872680bfafa6577796797e7b2ff191dcdfa327847f4.jpg)  
Extended Data Fig. 1 — Efect of training data size on DWI enhancement performance. To examine how the amount of training data afects DWI enhancement, we varied the training data size from 100 to 200 and 500 subjects and evaluated the resulting dMRI-F models on the HCP-YA test set. Performance improved with larger training datasets, as reflected by higher SSIM and PSNR and lower NRMSE. For all box plots, the center line denotes the median, the box spans the interquartile range, and the whiskers extend to the most extreme values within 1.5 times the interquartile range of the box. White crosses indicate means, connected by black lines to illustrate trends across training cohort sizes; plus signs denote outliers beyond the whiskers.

![](images/ffe3d70497cdb14d1cd035d0f7fd7e512d7542226fd023dc5e858a49f9e8a106.jpg)

![](images/d7e8d91051b6a013bc842e6cd04c267e542918210d1573d5cefb55c8410c0ba4.jpg)

![](images/ba950624c345600ec784f5997f71de4caabcfcd035152ca22e525e4503f16f5d.jpg)  
Extended Data Fig. 2 — Efect of the number of input difusion directions on DWI prediction performance.

Separate dMRI-F models were trained with 6, 10, 20, or 25 input DWI volumes to assess how the number of sampled difusion directions afects model prediction accuracy. Each model was evaluated on the HCP-YA test set using the same number of input volumes as during training. Enhancement performance generally improved as the number of input directions increased. For all box plots, the center line denotes the median, the box spans the interquartile range, and the whiskers extend to the most extreme values within 1.5 times the interquartile range of the box. White crosses indicate means, connected by black lines to illustrate trends across input direction counts; plus signs denote outliers beyond the whiskers.
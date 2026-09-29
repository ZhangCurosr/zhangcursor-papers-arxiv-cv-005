# SCALING VERSATILE 3D ASSETS EDITING WITH A MILLION-SCALE DATASET

Badi Li<sup>1,2,4</sup>, Tianxin Huang<sup>1</sup>, Yu Zhou<sup>3</sup>, Wei-Shi Zheng<sup>2,4</sup>, Yi Ma<sup>1,2</sup>, Shenghua Gao<sup>1,2∗</sup>

<sup>1</sup> The University of Hong Kong <sup>2</sup> Shenzhen Loop Area Institute <sup>3</sup> Shanghai Innovation Institute <sup>4</sup> Sun Yat-Sen University

 Pro<sub>j</sub>ect Pa<sub>g</sub>e § Code § Evaluation Model Dataset Benchmark

![](images/0de44f8da8190b76e531f73ace8e5e7fcd75d80235f00164a356d8425b682cdb.jpg)  
Figure 1: Given a 3D asset and an editing instruction, our method edits the asset accordingly while preserving unrelated regions.

## ABSTRACT

Although recent 3D generative models produce increasingly realistic assets, controllable 3D asset editing remains challenging. Existing methods are limited by scarce training data, insufficient source-aware modeling, and a lack of practical evaluation protocols. To address these limitations, we present Alchemy3D, a unified framework for training and evaluating versatile 3D asset editors that covers data construction, model architecture, and benchmark evaluation. Specifically, we curate Alchemy3D-1M, a large-scale 3D editing dataset containing 1.25M assets and 1.38M editing pairs across seven editing types. On this data, we train a family of generative flow models for general-purpose 3D asset editing. The model family supports image- and text-conditioned editing, few-step inference, and transfer to multi-view 3D part segmentation. We further introduce GEdit3D-Bench, a large-scale, open-world benchmark with a multi-dimensional evaluation protocol. Across existing and newly introduced benchmarks, our method outperforms prior methods on most metrics of editing fidelity, source preservation, and visual quality.

## 1 INTRODUCTION

With the development of diffusion and flow-matching models (Song & Ermon, 2019; Song et al., 2021; Ho et al., 2020; Lipman et al., 2023), recent 3D generative models (Xiang et al., 2025; 2026; Hunyuan3D et al., 2025) can produce high-quality 3D assets from text or image prompts. However, general-purpose 3D editing remains comparatively underdeveloped because it requires fine-grained spatial awareness. An editing model must not only execute a localized or global modification according to the instruction, but also keep irrelevant regions unchanged.

Compared to generation methods (Xiang et al., 2025; 2026; Hunyuan3D et al., 2025), existing 3D editing models are typically constrained by limitations in training data, model design, and evaluation protocols. First, achieving region-specific revisions requires paired data before and after editing, which is difficult to acquire. Existing 3D editing datasets are limited in scale, diversity, quality, and editing types (Ma et al., 2025; Xia et al., 2025; Weng et al., 2026). Second, recent 3D editors (Ma et al., 2025; Weng et al., 2026) inject the source asset through a ControlNet-style (Zhang et al., 2023) branch, which limits the interaction between source and target and may provide insufficient fine-grained guidance from the source. Finally, existing editing benchmarks are often relatively small, drawn from the same distribution as the training data, or evaluated by similarity to a single automatically generated target (Li et al., 2026b; Zhou et al., 2026b; Ye et al., 2025a; Xia et al., 2025; Weng et al., 2026), which restricts evaluation to in-distribution settings and fails to capture the one-to-many nature of generative editing.

To address these challenges, we present Alchemy3D, an integrated 3D editing framework comprising scalable data construction, model architecture design, and benchmark evaluation. For data construction, we select seven representative editing types (addition, removal, replacement, animation, local appearance, global appearance, and segmentation) and design type-specific pipelines to create editing pairs at scale. Candidate pairs are filtered with vision-language models (Qwen Team, 2026b; Bai et al., 2025), yielding 1.38 million editing pairs.

For the model architecture, we concatenate the tokens of the source asset and the edited target asset so that they interact through self-attention, while text or image instructions are injected through cross-attention. To improve inference efficiency, we further adopt a few-step distillation procedure based on MeanFlow (Geng et al., 2025) and DMD (Yin et al., 2024b;a). With this architecture and Alchemy3D-1M, we train a family of Alchemy3D model variants for different editing scenarios.

For evaluation, we construct GEdit3D-Bench, a benchmark built from assets that are independent of our training data. It combines newly synthesized 3D content with recently released real-world assets from Sketchfab<sup>1</sup> to mitigate in-distribution evaluation. Editing instructions are written by a vision-language model, target images are produced by an image editing model (Cao et al., 2025), and all samples are curated through automated and human filtering. Rather than measuring similarity to a single generated target, GEdit3D-Bench jointly assesses view quality, reference alignment, and Multimodal Large Language Model (MLLM)-based scores, providing a more comprehensive evaluation of 3D editors.

Our contributions are summarized as follows:

• To expand the scope of versatile 3D editing, we propose a scalable data construction pipeline spanning 7 diverse editing tasks, yielding Alchemy3D-1M, a dataset of 1.38 million high-quality editing pairs covering diverse assets.

• Instead of introducing source asset features through a separate branch, we fuse tokens from the source and edited target models, enabling sufficient interaction between them. We further propose a distillation pipeline that enables efficient, few-step inference for the 3D editing model. Building on this architecture and Alchemy3D-1M, we train a series of Alchemy3D model variants tailored to different editing use cases.

• To enable a more comprehensive and fair evaluation of 3D editors, we introduce GEdit3D-Bench, a large-scale, multi-dimensional benchmark independent of our training data.

• Extensive comparisons on multiple benchmarks confirm that our model achieves significant improvements over existing 3D editing methods, with the distilled model runs about 4× faster than the base model while retaining comparable quality.

Table 1: Comparison of existing 3D asset editing datasets. Our Alchemy3D-1M incorporates more diverse editing types and is more than 10× larger in scale than existing related datasets.
<table><tr><td>Dataset</td><td>Scale PBR</td><td></td><td colspan="5">Editing Types</td></tr><tr><td></td><td></td><td></td><td>Add Remove Replace Appearance Animation Segmentation</td><td></td><td></td><td></td><td></td></tr><tr><td>Steer3D</td><td>100K</td><td>X</td><td></td><td></td><td></td><td>X</td><td>X</td></tr><tr><td>Nano3D-100K (ICLR&#x27;26)</td><td>100K</td><td>X</td><td></td><td></td><td></td><td>X</td><td>X</td></tr><tr><td>3DEditVerse (ICML&#x27;26)</td><td>116K</td><td>X</td><td></td><td></td><td></td><td>X</td><td>x</td></tr><tr><td>PxForm (SIGGRAPH Asia&#x27;26)</td><td>102K</td><td>X</td><td></td><td></td><td></td><td>X</td><td>X</td></tr><tr><td>Alchemy3D-1M</td><td>1.38M</td><td></td><td></td><td></td><td></td><td></td><td></td></tr></table>

## 2 RELATED WORK

## 2.1 NATIVE 3D GENERATION

3D generative models synthesize representations such as point clouds (Luo & Hu, 2021; Nichol et al., 2022), neural fields (Muller et al., 2023), triplanes (Wang et al., 2023), and 3D Gaussian¨ splats (He et al., 2024). TRELLIS (Xiang et al., 2025) introduced a structured latent space that supports meshes, NeRFs (Mildenhall et al., 2021), and 3D Gaussian splats (Kerbl et al., 2023) through a unified VAE (Kingma & Welling, 2013). TRELLIS.2 (Xiang et al., 2026) subsequently introduced the compact O-Voxel representation and decomposed asset generation into sparse-structure, shape, and PBR-material stages. Alchemy3D adopts this representation and transforms the generation pipeline into a source-conditioned editor.

## 2.2 3D ASSET EDITING

Early 3D editing methods rely on optimization, either reconstructing assets from images modified by a 2D editor (Haque et al., 2023) or applying score-distillation sampling (Poole et al., 2023) to optimize a 3D representation (Sella et al., 2023; Li et al., 2024; Palandra et al., 2024). Subsequent systems edit multi-view renderings or videos and reconstruct an asset from the modified observations; agentic variants primarily automate view selection (Qi et al., 2024; Huang et al., 2025; Qu et al., 2025; Zheng et al., 2025). Although flexible, these pipelines are computationally expensive and prone to cross-view inconsistency.

Feed-forward approaches improve efficiency by adapting techniques from zero-shot image editing. VoxHammer (Li et al., 2026b) combines TRELLIS (Xiang et al., 2025) with RF-Inversion (Rout et al., 2024) and attention manipulation (Wang et al., 2025a), while Nano3D (Ye et al., 2025a) integrates TRELLIS with FlowEdit (Kulikov et al., 2025). Their reliance on training-free image-editing mechanisms, however, limits robustness on complex 3D transformations. Steer3D (Ma et al., 2025) and 3DEditFormer (Xia et al., 2025) instead train 3D editors on constructed paired datasets, demonstrating the value of task-specific supervision but remaining constrained by data scale and coverage. Concurrently, PartFlow (Weng et al., 2026) constructs editing pairs from part-segmentation datasets and is the closest prior setting to ours. Alchemy3D extends this direction with substantially larger and broader dataset, a unified model family, and open-world evaluation.

## 3 ALCHEMY3D-1M DATASET

As shown in Table 1, existing datasets for 3D asset editing remain limited in both scale and editing types. Among them, Steer3D (Ma et al., 2025) independently reconstructs the before and after assets from corresponding image pairs, which can lead to poor identity preservation. Nano3D-100K and 3DEditVerse (Ye et al., 2025a; Xia et al., 2025) focus primarily on structural edits, such as addition, removal, and replacement. PxForm, in contrast, curates editing pairs from labeled 3D part datasets (Dong et al., 2025), limiting the diversity and complexity of the assets. In contrast, Alchemy3D-1M, to the best of our knowledge, is the first million-scale dataset for 3D asset editing.

The construction of Alchemy3D-1M is divided into three distinct categories. For structural editing data which cover addition, removal, replacement, as well as local and global appearance modifications, the dataset generation follows a structured three-stage pipeline consisting of preparation, construction, and filtering. (1) In the data preparation stage, source images are rendered from public 3D datasets (Deitke et al., 2023; Fu et al., 2021; Chang et al., 2015; Collins et al., 2022; Zhang et al., 2025; Khanna et al., 2024) or generated using a text-to-image model (Labs, 2025). The paired 3D source assets are re-generated with TRELLIS.2 (Xiang et al., 2026) according to the source images, where the sampling trajectories are stored for subsequent operations. Then, we use Qwen3-VL (Bai et al., 2025) to write an editing text instruction for these source images and introduce FLUX.2- Dev-Turbo to generate target edited images. (2) During data construction, for removal and addition types, we detect (Jiang et al., 2026) and segment (Carion et al., 2025) 2D edit-region masks based on the text instruction and the rendered views from reconstructed assets. These masks are then lifted into 3D using SegViGen (Li et al., 2026a). For replacement type, the 3d editing mask is instead estimated by training-free editing method (Kulikov et al., 2025). Guided by these 3D masks, trajectory-aware 3D inpainting modifies the target region while maintaining the original diffusion trajectories in unedited areas to produce the edited 3D model. (3) Finally, in the filtering stage, the generated 3D and image editing pairs are checked and recaptioned with Qwen3.6-27B (Qwen Team, 2026b), where the low quality assets are discarded.

![](images/b34ff46be6ff79cf0a6a6a511648f3b9b8de8b709a6d81ff782ec1d1bda68b73.jpg)  
Figure 2: Statistics of Alchemy3D-1M dataset. Alchemy3D-1M includes 1.25M unique assets and 1.38 M editing pairs, across 7 different editing types.

For animation data, we sample pairs of frames from motion sequences in existing animation datasets (Deitke et al., 2023; Zhang et al., 2025; Li et al., 2021; Inc., 2025) and articulation datasets (Iliash et al., 2026). For articulated assets without explicit motion sequences, we simulate motions with a physics simulator (Xiang et al., 2020). We then compute inter-frame feature similarities with DINOv3 (Simeoni et al., 2025) and discard pairs without noticeable motion.´

For segmentation data, we collect assets from 3D part segmentation datasets (Mo et al., 2019; Wang et al., 2025b; Ding et al., 2025) and apply a custom color assignment algorithm to colorize individual parts. Rendered images of these colorized assets then serve as visual instructions to guide the segmentation. The detailed construction pipeline is provided in Appendix A.2.

As shown in Table 1 and Fig. 2, Alchemy3D-1M contains 1,249,862 unique 3D assets and 1,382,596 annotated editing pairs spanning seven editing categories: addition, removal, replacement, local appearance, global appearance, animation, and segmentation. For the training set, we downsample the animation pairs to 200k to balance the distribution across editing types and reserve the segmentation pairs exclusively for downstream evaluation. This yields a final training set of 825,040 editing pairs for unified training. Compared to existing 3D editing datasets, Alchemy3D-1M is substantially larger, more diverse, and covers a broader spectrum of editing tasks, serving as a valuable asset for the 3D vision community.

## 4 ALCHEMY3D MODEL

## 4.1 ARCHITECTURE.

3D asset editing is inherently a generative task: one text or image instruction can yield multiple valid edited assets. While current editing approaches (Xia et al., 2025; Weng et al., 2026; Zhou et al., 2026b; Ma et al., 2025) train generative models to modify 3D assets, they treat the source asset strictly as external guidance via an auxiliary branch such as ControlNet (Zhang et al., 2023), which may miss crucial cross-feature interactions between the source and target assets.

![](images/54c127ab440bd53c71888e8ff0256fee746d2ef823fc10ef77764985d287cf98.jpg)  
Figure 3: Illustration of Alchemy3D model. Following Trellis.2, we design a three-stage editing pipeline that sequentially edits sparse structure (voxel occupancy), geometry (fine-grained surface), and material (PBR attributes). Each stage comprises hybrid attention blocks that denoise noisy target tokens by attending to source tokens from the source asset alongside condition tokens from image/text instructions. For clarity, we display decoded outputs from denoised tokens at each stage.

As illustrated in Fig. 3, rather than introducing an auxiliary branch to inject source features, we directly incorporate tokens encoded from both the source asset and the noisy target state into the hybrid attention blocks, enabling in-depth feature interaction. Following Trellis.2 (Xiang et al., 2026), we organize these attention blocks into a three-stage flow transformer that sequentially edits sparse structure (voxel occupancy), geometry (fine-grained surface), and materials (PBR attributes).

The requested edit is specified by an external condition, which can be either a text instruction or an edited image instruction. To support diverse usage scenarios, we adopt multiple encoders to process different conditional signals. For text-conditioned editing, we encode the text instruction with Qwen3.5-2B (Qwen Team, 2026a) followed by a lightweight trainable projector, and the resulting features provide semantic guidance for specific editing operations. For image-conditioned editing, we use DINOv3 (Simeoni et al., 2025) to extract the condition tokens. Since DINO focuses on´ extracting semantic features instead of low-level vision features, we optionally trained a separate texture editing model with FLUX.2 encoder (Labs, 2025) followed by a lightweight projector for the material editing stage, which can help produce better color and texture fidelity. Comparison between DINOv3 and FLUX.2 encoder is presented in Sec. 6.2.

## 4.2 TRAINING OBJECTIVES.

Flow Matching. During training, a pair of source and target assets is sampled and encoded into the corresponding latent representations, which can be sparse voxel, geometry, or material latents depending on the stage. The target latent $X _ { 0 }$ is then interpolated with Gaussian noise according to $X _ { t } ^ { ' } = ( 1 - t ) X _ { 0 } + t \breve { X } _ { 1 }$ , and concatenated token-wise with the source latent $X ^ { \prime }$ . The generative flow model v<sub>θ</sub> predicts the velocity field from the concatenated input tokens $[ X ^ { \prime } X _ { t } ]$ , conditioned on the image or text instruction c and the sampled timestep t. We then train the model with the standard optimal transport flow-matching objective (Lipman et al., 2023):

$$
\mathcal { L } _ { \mathrm { F M } } = \mathbb { E } _ { t , X _ { t } \sim p _ { t } } \left[ | | ( X _ { 1 } - X _ { 0 } ) - v _ { \theta } ( X ^ { \prime } , X _ { t } , c , t ) | | _ { 2 } ^ { 2 } \right] ,\tag{1}
$$

Few-Step Distillation. While effective, concatenating source and target tokens roughly doubles the sequence length and consequently increases the computational cost of attention. We recover this efficiency through a two-stage distillation procedure inspired by few-step video distillation (Gu et al., 2026). In the first stage, we train a continuous-step flow map following MeanFlow (Geng

et al., 2025), with the objective:

$$
\begin{array} { r } { \mathcal { L } _ { \mathrm { M F } } ( \theta ) = \mathbb { E } [ | | u _ { \theta } ( X ^ { \prime } , X _ { t } , c , r , t ) - \mathrm { s g } ( u _ { \mathrm { t g t } } ) | | _ { 2 } ^ { 2 } ] , } \\ { u _ { \mathrm { t g t } } = v ( X _ { t } , t ) - ( t - r ) \frac { d u _ { \theta } ( X ^ { \prime } , X _ { t } , c , r , t ) } { d t } , } \end{array}\tag{2}
$$

where the flow map $u _ { \theta } .$ , which differs slightly from the pretrained flow model, predicts the target flow conditioned on an additional timestep $^ { r } \cdot$ Here, $\operatorname { s g } ( \cdot )$ denotes the stop-gradient operation, and v denotes the pretrained flow model defined in Eq. 1. Following the transition model (Wang et al., 2026), we approximate the time derivative as

$$
\frac { d u _ { \theta } ( X ^ { \prime } , X _ { t } , c , r , t ) } { d t } \approx \frac { u _ { \theta } ( X ^ { \prime } , X _ { t + \Delta t } , c , r , t + \Delta t ) - u _ { \theta } ( X ^ { \prime } , X _ { t - \Delta t } , c , r , t - \Delta t ) } { 2 \Delta t } .\tag{3}
$$

Starting from this flow map, we then perform on-policy distribution matching distillation following DMD (Yin et al., 2024b;a). Given an initial Gaussian noise $X _ { 1 } \sim \mathcal { N } ( 0 , I )$ , the student model first approximates a clean sample through its inference trajectory $\hat { X _ { 0 } } = f _ { \theta } ( X _ { 1 } )$ , which is then re-noised at a sampled timestep t as $X _ { t } = ( 1 - t ) \hat { X _ { 0 } } + t \epsilon$ , where $\epsilon \sim \mathcal { N } ( 0 , I )$ . The DMD gradient is given by

$$
\nabla _ { \theta } \mathcal { L } _ { \mathrm { D M D } } = - \mathbb { E } _ { t , X _ { 1 } } \left[ \left( s _ { \mathrm { r e a l } } ( X _ { t } , t ) - s _ { \mathrm { f a k e } } ( X _ { t } , t ) \right) \frac { \partial f _ { \theta } ( X _ { 1 } ) } { \partial \theta } \right] ,\tag{4}
$$

where $s _ { \mathrm { r e a l } }$ and $s _ { \mathrm { f a k e } }$ denote the score functions of the pretrained teacher and the student-generated distributions, respectively. We replace the adversarial loss in DMD2 Yin et al. (2024a) with the MeanFlow objective in Eq. 2. Specifically, we optimize

$$
\begin{array} { r } { \mathcal { L } _ { \mathrm { o n - p o l i c y } } = \mathcal { L } _ { \mathrm { D M D } } + \mathcal { L } _ { \mathrm { M F } } . } \end{array}\tag{5}
$$

Throughout the few-step distillation process, only the attached LoRA modules are trainable.

## 5 GEDIT3D-BENCH

Existing benchmarks for 3D asset editing have several fundamental limitations. Edit3D-Bench, Eval3DEdit, and TANGOEdit (Li et al., 2026b; Zhou et al., 2026b; Lim et al., 2026) each collect only about 100 assets from existing 3D datasets (Deitke et al., 2023; Downs et al., 2022; Yang et al., 2024), which is too small for a comprehensive evaluation. Another line of work, including Steer3D, Nano3D, 3DEditFormer, and PartFlow (Ma et al., 2025; Ye et al., 2025a; Xia et al., 2025; Weng et al., 2026), builds test sets by splitting the training data of the corresponding method, which risks leakage and overfitting to that specific distribution. The protocols are also narrow. The first line of works evaluates only CLIP or DINO similarity between rendered views and the input image. The second provides a low-quality ground-truth asset and measures alignment to it, which overlooks the quality of that reference and the one-to-many nature of generative editing. GEdit3D-Bench instead provides large-scale, open-world data and evaluates complementary aspects of editing quality.

Construction.Each sample contains a source asset ${ \mathcal { A } } ,$ source and target captions $\mathcal { C } _ { \mathrm { s r c } }$ and $\mathcal { C } _ { \mathrm { t g t } }$ , a source rendered image $\bar { \mathcal { T } } _ { \mathrm { s r c } } .$ , an editing instruction I, and a target edited image $\mathcal { T } _ { \mathrm { t g t } }$ . We first assemble synthetic assets generated with Hunyuan3D V3.1 and recently released Sketchfab assets from diverse categories. Then, we caption multi-view renderings, generate instructions for six editing types, and edit one source view to produce the visual target. Finally, automated checks and human review will be introduced to remove inconsistent or low-quality samples and generate target captions for text-based evaluation. More details can be found in Appendix A.3.

Metrics.Given an edited asset ${ \mathcal { A } } ^ { \prime } ,$ , we evaluate its multi-view renderings without treating a generated 3D target as ground truth. View Quality averages aesthetic-predictor-v2.5, MANIQA (Yang et al., 2022), and MUSIQ (Ke et al., 2021) scores across views (Chen & Mo, 2022). Reference Alignment comprises image alignment $A _ { I }$ and text alignment $A _ { T }$ , instantiated with complementary encoders including EVA-CLIP (Sun et al., 2023), SigLIP (Zhai et al., 2023), BLIP (Li et al., 2022), DI-NOv3 (Simeoni et al., 2025), and Uni3D (Zhou et al., 2024), where applicable. Following common´ practice in image and video evaluation (Zhou et al., 2026a; Liang et al., 2026; Jiang et al., 2024; Ye et al., 2025b), we employ MLLM (Multimodal Large Language Model) as judge: Success Rate (SR) is a binary measure of whether the requested edit succeeds, while Instruction Following (IF), Identity Preservation (IP), and Visual Quality (VQ) are scored from 1 to 100.

![](images/fb56afebd28e2eccc8bdb3e3e15a1606809e807faad041d8dcca3f6cb87e00e3.jpg)  
Figure 4: Qualitative comparison between Alchemy3D and baselines. We show only addition, removal, and replacement, the editing types supported by all baselines.

## 6 EXPERIMENTS

## 6.1 EVALUATION SETTING.

We evaluate five variants of our model: (1) Alchemy3D, our default image-conditioned editor based on the DINOv3 encoder; (2) Alchemy3D-Flux, which uses the FLUX.2 image encoder to improve color and texture fidelity; (3) Alchemy3D-Instruct, which uses Qwen3.5-2B (Qwen Team, 2026a) to support text-instruction input; (4) Alchemy3D-Turbo, a few-step image-conditioned editor distilled from Alchemy3D; and (5) Alchemy3D-Segment, which is fine-tuned from Alchemy3D for multi-view-guided 3D part segmentation.

For baselines and benchmarks, We compare image-conditioned Alchemy3D with Nano3D (Ye et al., 2025a), 3DEditFormer (Xia et al., 2025), and PartFlow (Weng et al., 2026). For instructionconditioned editing, we compare with Steer3D (Ma et al., 2025), In this work, we adopt GEdit3D-Bench (Sec. 5) as our primary evaluation benchmark. For a more comprehensive comparison, we additionally evaluate on existing datasets, including Eval3DEdit (Zhou et al., 2026b), 3DEditVerse (Xia et al., 2025), Nano3D-100K (Ye et al., 2025a), and the Edit3D-Bench introduced by VoxHammer (Li et al., 2026b). Since Nano3D does not release an official test split, we randomly sample 1.5K pairs from its data for evaluation. For text-conditioned editors, we conduct comparisons on GEdit3D-Bench and on the Edit3D-Bench introduced by Steer3D.

Unless otherwise stated, MLLM scores use Gemini-3.8-Flash.<sup>2</sup> On GEdit3D-Bench, a random 400- sample subset is used for MLLM scoring and the user study. Alchemy3D and Alchemy3D-Instruct are sampled with 12 steps and classifier-free guidance on sparse structure and shape. Alchemy3D-Turbo uses 3 steps, with guidance fused during training. Please refer to Sec. A.4.1 for more details.

## 6.2 EXPERIMENTAL RESULTS

Image-conditioned editing. As demonstrated in Tables 2 and 3, Alchemy3D and Alchemy3D-Turbo outperform existing methods across both alignment and MLLM scores, with human preference for our methods significantly exceeding that of the baselines. More details on the human user study are provided in Sec. A.4.1. Qualitative comparisons in Fig. 4 indicate our methods faithfully execute the requested changes while well preserving the identity of the source asset. In terms of sampling efficiency, Alchemy3D-Turbo requires 1.231 seconds, making it faster than all baseline methods. Although PartFlow uses a lower-complexity ControlNet-style architecture, it samples with 50 steps by default and therefore incurs a substantial inference cost. The remaining per-type results are reported in Table 10 and Table 9. As a side comparison, we also report image alignment scores on 3DEditVerse, the Edit3D-Bench introduced by VoxHammer, and Nano3D-100K in Table 4. Alchemy3D and Alchemy3D-Turbo still outperform the baselines.

![](images/da56915610371abb8a7b7f5cb73e1ff3ea0961a49257053e9b7afcab491eb17a.jpg)  
Figure 5: (a) Examples from Alchemy3D-Instruct. (b) Segmentation results from Alchemy3D-Segment, where the segmentation criterion (e.g., semantic or instance) is controlled by the image instruction. (c) Comparison between Alchemy3D and Alchemy3D-Flux.

Table 2: Quantitative results on GEdit3D-Bench for addition, removal, and replacement. Per-type results are reported in Table 9. Human preference scores do not sum to 100% because participants could also select a “tie” option when neither result was clearly better. Sampling time is measured on a single NVIDIA H200 GPU and averaged over 10 examples.
<table><tr><td></td><td colspan="2">View Qual.</td><td colspan="3">Ref. Alignment</td><td colspan="4">MLLM</td><td colspan="2">Human Time</td></tr><tr><td>Method</td><td></td><td>Aes. ↑ MUSIQ ↑ AC</td><td>CLIP</td><td>↑ACLIP</td><td>↑ADINO</td><td>SR↑</td><td>VQ↑</td><td>IF↑</td><td>IP↑</td><td>Pref.</td><td>(s) ↓</td></tr><tr><td>Nano3D</td><td>4.40</td><td>72.87</td><td>81.59</td><td>65.10</td><td>69.15</td><td>51.14%</td><td>70.16 53.84</td><td></td><td>72.09</td><td>5.1%</td><td>8.629</td></tr><tr><td>3DEditFormer</td><td>4.34</td><td>70.87</td><td>81.83</td><td>63.19</td><td>68.07</td><td>58.90%</td><td>66.31</td><td></td><td>59.79 63.39</td><td>5.3%</td><td>3.262</td></tr><tr><td>PartFlow</td><td>4.31</td><td>70.96</td><td>81.25</td><td>62.24</td><td>67.78</td><td>55.50%</td><td></td><td></td><td>65.27 56.20 66.60</td><td>6.5%</td><td>7.184</td></tr><tr><td>Alchemy3D-Turbo</td><td>4.58</td><td>72.94</td><td>82.36</td><td>68.31</td><td>72.88</td><td>88.18%</td><td></td><td></td><td>69.91 70.71 73.22</td><td></td><td>1.231</td></tr><tr><td>Alchemy3D</td><td>4.43</td><td>72.36</td><td>84.91</td><td>67.47</td><td>73.18</td><td>89.55%</td><td></td><td></td><td>72.74 71.38 76.19</td><td>70.7%5.202</td><td></td></tr></table>

Instruction-conditioned editing. In this section, we compare text-conditioned editing between Alchemy3D-Instruct and Steer3D (Ma et al., 2025). Since Steer3D (Ma et al., 2025) provides neither text captions nor image instructions, we evaluate editing performance using the aesthetic score for visual quality, together with the MLLM-based metrics SR, IF, and IP described in Sec. 5. As shown in Table 6, Alchemy3D-Instruct, trained across all editing types, outperforms Steer3D’s typespecific models on both GEdit3D-Bench and Steer3D’s Edit3D-Bench. Examples of Alchemy3D-Instruct are shown in Fig. 5(a). Interestingly, the text-conditioned model underperforms the image conditioned one, likely due to the ambiguity of text instructions. Motivated by this observation, we explore an agentic text-conditioned editing pipeline that automatically selects a 2D view, edits it, and then invokes the image-conditioned Alchemy3D. A detailed analysis is presented in Sec. A.5.1.

Ablation on Encoder. As shown in Fig. 5(c), replacing DINOv3 with the FLUX.2 encoder improves color and style fidelity, while reducing local editing quality. Though the standard performances of the two models are close (Table 5), Alchemy3D achieves much better performance when adapting to part segmentation with limited data (Table 7). We therefore retain Alchemy3D as our default foundation and release Alchemy3D-Flux for appearance-critical use cases.

Downstream part segmentation. To validate whether the priors acquired through unified editing transfer to downstream applications, we adapt Alchemy3D using LoRA (Hu et al., 2022) to create Alchemy3D-Segment, a model designed for 3D part segmentation guided by 1 to 8 segmented views. Results on PartObjaverse-Tiny (Yang et al., 2024) (Table 7) show that Alchemy3D-Segment substantially outperforms prior baselines and initialization variants trained from scratch or from TRELLIS.2. These improvements demonstrate that unified 3D editing serves as a powerful pretext

Table 3: Quantitative results on Eval3DEdit. Since Eval3DEdit originally only evaluates CLIP similarities, we additionally report aesthetic score for view quality and SR for MLLM rating.
<table><tr><td></td><td colspan="4">Add / Remove / Replace</td><td colspan="4">Action</td><td colspan="4">Style</td></tr><tr><td>Method</td><td>Aes.</td><td>AI</td><td> $A _ { T }$ </td><td>SR</td><td>Aes.</td><td>AI</td><td>AT</td><td>SR</td><td>Aes.</td><td> $A _ { I }$ </td><td> $A _ { T }$ </td><td>SR</td></tr><tr><td>Nano3D</td><td>4.60</td><td>82.31</td><td>49.43</td><td>40.00%</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>3DEditFormer</td><td>4.55</td><td>81.78</td><td>49.24</td><td>70.97%</td><td></td><td>4.0675.8044.45</td><td></td><td>20.00%</td><td></td><td></td><td></td><td></td></tr><tr><td>PartFlow</td><td>4.51</td><td>81.59</td><td>49.32</td><td>67.86%</td><td>一</td><td>一</td><td></td><td></td><td>4.58</td><td>78.10</td><td>50.68</td><td>40.00%</td></tr><tr><td>Alchemy3D-Turbo</td><td>4.64</td><td>82.67</td><td>52.56</td><td>90.00%</td><td>4.33</td><td>79.96</td><td>51.46</td><td>90.00%</td><td>4.61</td><td>81.15</td><td>52.51</td><td>100.00%</td></tr><tr><td>Alchemy3D</td><td>4.60</td><td>83.77</td><td>52.10</td><td>83.87%</td><td>4.29</td><td>80.84</td><td>51.15</td><td>80.00%</td><td>4.64</td><td>82.40</td><td>54.41</td><td>90.00%</td></tr></table>

Table 4: Comparison on other benchmark data.
<table><tr><td rowspan=1 colspan=2>Method           $A _ { I } ^ { \mathrm { C L I P } } \uparrow$   $A _ { I } ^ { \mathrm { S i g L I P } } \uparrow$   $A _ { I } ^ { \mathrm { D I N O \ . } }$ ← $A _ { I } ^ { \mathrm { U n i 3 D . } }$ 个</td></tr><tr><td rowspan=1 colspan=2>3DEditVerse</td></tr><tr><td rowspan=2 colspan=2>3DEditFormer      69.18   79.18   64.60   36.76PartFlow          68.08</td></tr><tr><td rowspan=1 colspan=1>8.08 78.60 63.74 36.46</td></tr><tr><td rowspan=1 colspan=2>Alchemy3D       73.10   82.18   72.20   37.41</td></tr><tr><td rowspan=1 colspan=2>Edit3D-Bench (VoxHammer)</td></tr><tr><td rowspan=2 colspan=2>3DEditFormer      76.91   84.99   66.90   38.33PartFlow          77.55   85.95   69.02   38.93Alchemy3D-Turbo 79.90   87.28   72.91   38.25Alchemy3D       80.91   88.44   75.04   39.06</td></tr><tr><td rowspan=1 colspan=1>85.9569.0238.93</td></tr><tr><td rowspan=1 colspan=2>Nano3D-100K</td></tr><tr><td rowspan=1 colspan=2>3DEditFormer      66.30   78.06   60.38   36.57</td></tr><tr><td rowspan=1 colspan=2>PartFlow          65.42   78.61   61.18   36.20</td></tr><tr><td rowspan=1 colspan=2>Alchemy3D-Turbo 69.86   80.81   66.58   37.04Alchemy3D       72.22   82.37   71.11   38.06</td></tr></table>

Table 6: Quantitative comparison between Steer3D and Alchemy3D-Instruct on GEdit3D-Bench and Edit3D-Bench (Steer3D).

Table 5: Comparison between Alchemy3D and Alchemy3D-Flux on GEdit3D-Bench.
<table><tr><td>Model</td><td>Aes. ↑</td><td> $A _ { I } ^ { \mathrm { C L I P } } \uparrow$ </td><td> $A _ { T } ^ { \mathrm { C L I P } } \uparrow$ </td><td> $A _ { I } ^ { \mathrm { D I N O } } \uparrow$ </td></tr><tr><td>Alchemy3D-Flux</td><td>4.420</td><td>83.43</td><td>66.36</td><td>72.26</td></tr><tr><td>Alchemy3D</td><td>4.419</td><td>83.53</td><td>66.36</td><td>72.52</td></tr></table>

Table 7: Quantitative results for semantic part segmentation on PartObjaverse-Tiny. Trial experiments compare different strategies for training single-view conditioned segmentation.

<table><tr><td>Method</td><td>Aes.</td><td>SR</td><td>IF</td><td>IP</td></tr><tr><td>GEdit3D-Bench</td><td></td><td></td><td></td><td></td></tr><tr><td>Steer3D Alchemy3D-Instruct</td><td>3.84 4.60</td><td>12.37% 45.67%</td><td>38.55 35.28 51.81</td><td>72.56</td></tr><tr><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Edit3D-Bench (Steer3D) Steer3D</td><td></td><td></td><td></td><td></td></tr><tr><td>Alchemy3D-Instruct 3.9449.60%</td><td></td><td>3.78 33.73% 38.69 53.94</td><td>67.81</td><td>71.76</td></tr></table>

<table><tr><td rowspan=1 colspan=1>Method                                   mIoU ↑</td></tr><tr><td rowspan=1 colspan=1>Trial Experiments</td></tr><tr><td rowspan=2 colspan=1>From Scratch (1-view)                       28.54From TRELLIS.2 (1-view)                   58.68From Alchemy3D (1-view)                   60.08From Alchemy3D-Flux (LoRA+1-view)      52.55From Alchemy3D (LoRA+1-view)           61.10</td></tr><tr><td rowspan=1 colspan=1>58.68</td></tr><tr><td rowspan=1 colspan=1>Baselines</td></tr><tr><td rowspan=1 colspan=1>Find3D                                      19.35PartField                                     52.01P3-SAM                                     46.58SegViGen                                   53.49</td></tr><tr><td rowspan=1 colspan=1>Ours</td></tr><tr><td rowspan=2 colspan=1>Alchemy3D-Segment(1 to 8-views)(1-view)                                   63.1065.23(4-view)                                   66.98(8-view)                                   68.70</td></tr><tr><td rowspan=1 colspan=1></td></tr></table>

task for downstream part understanding. Interestingly, training Alchemy3D-Segment with variable 1-to-8 view inputs yields better 1-view evaluation performance than training on single views alone (Alchemy3D LoRA+1-view), highlighting the value of multi-view data augmentation.

## 7 CONCLUSION

In this work, we present Alchemy3D, a unified framework spanning scalable data construction, model architecture, and benchmark evaluation for versatile 3D editing. We construct Alchemy3D-1M, a dataset of 1.38 million editing pairs across seven editing types, and introduce an editing architecture that fuses tokens from source and edited target 3D representations via self-attention, with text/image instructions injected via cross-attention. Together with a few-step distillation procedure, this architecture yields a series of Alchemy3D model variants tailored to different editing use cases. To enable fair, comprehensive evaluation, we also introduce GEdit3D-Bench, a large-scale, multidimensional benchmark entirely independent of our training data. Extensive experiments show that our models achieve significant improvements over existing 3D editing methods, with the distilled variant achieving a 4× acceleration while maintaining superior performance.

## 8 AI USE STATEMENT

In this work, we used generative AI tools to edit the manuscript for readability and to assist with writing and debugging code. We have reviewed all AI-assisted work. Two authors checked the revised text, and the AI-assisted code was tested for correctness. We take responsibility for the final content of this work, including text, claims, or artifacts produced with the aid of generative AI.

## REFERENCES

Niket Agarwal, Arslan Ali, Jon Allen, Martin Antolini, Adeline Aubame, Alisson Azzolini, Junjie Bai, Maciej Bala, Yogesh Balaji, Josh Bapst, et al. Cosmos 3: Omnimodal world models for physical ai. arXiv preprint arXiv:2606.02800, 2026.

Shuai Bai, Yuxuan Cai, Ruizhe Chen, Keqin Chen, Xionghui Chen, Zesen Cheng, Lianghao Deng, Wei Ding, Chang Gao, Chunjiang Ge, Wenbin Ge, Zhifang Guo, Qidong Huang, Jie Huang, Fei Huang, Binyuan Hui, Shutong Jiang, Zhaohai Li, Mingsheng Li, Mei Li, Kaixin Li, Zicheng Lin, Junyang Lin, Xuejing Liu, Jiawei Liu, Chenglong Liu, Yang Liu, Dayiheng Liu, Shixuan Liu, Dunjie Lu, Ruilin Luo, Chenxu Lv, Rui Men, Lingchen Meng, Xuancheng Ren, Xingzhang Ren, Sibo Song, Yuchong Sun, Jun Tang, Jianhong Tu, Jianqiang Wan, Peng Wang, Pengfei Wang, Qiuyue Wang, Yuxuan Wang, Tianbao Xie, Yiheng Xu, Haiyang Xu, Jin Xu, Zhibo Yang, Mingkun Yang, Jianxin Yang, An Yang, Bowen Yu, Fei Zhang, Hang Zhang, Xi Zhang, Bo Zheng, Humen Zhong, Jingren Zhou, Fan Zhou, Jing Zhou, Yuanzhi Zhu, and Ke Zhu. Qwen3-vl technical report. arXiv preprint arXiv:2511.21631, 2025.

Siyu Cao, Hangting Chen, Peng Chen, Yiji Cheng, Yutao Cui, Xinchi Deng, Ying Dong, Kipper Gong, Tianpeng Gu, Xiusen Gu, et al. Hunyuanimage 3.0 technical report. arXiv preprint arXiv:2509.23951, 2025.

Nicolas Carion, Laura Gustafson, Yuan-Ting Hu, Shoubhik Debnath, Ronghang Hu, Didac Suris, Chaitanya Ryali, Kalyan Vasudev Alwala, Haitham Khedr, Andrew Huang, Jie Lei, Tengyu Ma, Baishan Guo, Arpit Kalla, Markus Marks, Joseph Greer, Meng Wang, Peize Sun, Roman Radle,¨ Triantafyllos Afouras, Effrosyni Mavroudi, Katherine Xu, Tsung-Han Wu, Yu Zhou, Liliane Momeni, Rishi Hazra, Shuangrui Ding, Sagar Vaze, Francois Porcher, Feng Li, Siyuan Li, Aishwarya Kamath, Ho Kei Cheng, Piotr Dollar, Nikhila Ravi, Kate Saenko, Pengchuan Zhang, and ´ Christoph Feichtenhofer. SAM 3: Segment anything with concepts. 2025.

Angel X. Chang, Thomas A. Funkhouser, Leonidas J. Guibas, Pat Hanrahan, Qi-Xing Huang, Zimo Li, Silvio Savarese, Manolis Savva, Shuran Song, Hao Su, Jianxiong Xiao, Li Yi, and Fisher Yu. Shapenet: An information-rich 3d model repository. 2015.

Chaofeng Chen and Jiadi Mo. IQA-PyTorch: Pytorch toolbox for image quality assessment. [Online]. Available: https://github.com/chaofengc/IQA-PyTorch, 2022.

Jasmine Collins, Shubham Goel, Kenan Deng, Achleshwar Luthra, Leon Xu, Erhan Gundogdu, Xi Zhang, Tomas F Yago Vicente, Thomas Dideriksen, Himanshu Arora, et al. Abo: Dataset and benchmarks for real-world 3d object understanding. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, 2022.

Matt Deitke, Ruoshi Liu, Matthew Wallingford, Huong Ngo, Oscar Michel, Aditya Kusupati, Alan Fan, Christian Laforte, Vikram Voleti, Samir Yitzhak Gadre, et al. Objaverse-xl: A universe of 10m+ 3d objects. Advances in Neural Information Processing Systems, 2023.

Lihe Ding, Shaocong Dong, Yaokun Li, Chenjian Gao, Xiao Chen, Rui Han, Yihao Kuang, Hong Zhang, Bo Huang, Zhanpeng Huang, et al. Fullpart: Generating each 3d part at full resolution. arXiv preprint arXiv:2510.26140, 2025.

Shaocong Dong, Lihe Ding, Xiao Chen, Yaokun Li, Yuxin Wang, Yucheng Wang, Qi Wang, Jaehyeok Kim, Chenjian Gao, Zhanpeng Huang, et al. From one to more: Contextual part latents for 3d generation. In 2025 IEEE/CVF International Conference on Computer Vision (ICCV), 2025.

Laura Downs, Anthony Francis, Nate Koenig, Brandon Kinman, Ryan Hickman, Krista Reymann, Thomas B McHugh, and Vincent Vanhoucke. Google scanned objects: A high-quality dataset of 3d scanned household items. In 2022 International Conference on Robotics and Automation (ICRA), 2022.

Haoqiang Fan, Hao Su, and Leonidas J. Guibas. A point set generation network for 3d object reconstruction from a single image. In 2017 IEEE Conference on Computer Vision and Pattern Recognition, CVPR 2017, Honolulu, HI, USA, July 21-26, 2017, 2017.

Huan Fu, Rongfei Jia, Lin Gao, Mingming Gong, Binqiang Zhao, Steve Maybank, and Dacheng Tao. 3d-future: 3d furniture shape with texture. International Journal ofComputer Vision, 2021.

Zhengyang Geng, Mingyang Deng, Xingjian Bai, Zico Kolter, and Kaiming He. Mean flows for one-step generative modeling. Advances in Neural Information Processing Systems, 2025.

Georgia Gkioxari, Justin Johnson, and Jitendra Malik. Mesh R-CNN. In 2019 IEEE/CVF International Conference on Computer Vision, ICCV 2019, Seoul, Korea (South), October 27 - November 2, 2019, 2019.

Yuchao Gu, Guian Fang, Yuxin Jiang, Weijia Mao, Song Han, Han Cai, and Mike Zheng Shou. Anyflow: Any-step video diffusion model with on-policy flow map distillation, 2026.

Ayaan Haque, Matthew Tancik, Alexei A Efros, Aleksander Holynski, and Angjoo Kanazawa. Instruct-nerf2nerf: Editing 3d scenes with instructions. In Proceedings of the IEEE/CVF international conference on computer vision, 2023.

Xianglong He, Junyi Chen, Sida Peng, Di Huang, Yangguang Li, Xiaoshui Huang, Chun Yuan, Wanli Ouyang, and Tong He. Gvgen: Text-to-3d generation with volumetric representation. In European Conference on Computer Vision, 2024.

Jonathan Ho, Ajay Jain, and Pieter Abbeel. Denoising diffusion probabilistic models. In Advances in Neural Information Processing Systems 33, 2020.

Edward J. Hu, Yelong Shen, Phillip Wallis, Zeyuan Allen-Zhu, Yuanzhi Li, Shean Wang, Lu Wang, and Weizhu Chen. Lora: Low-rank adaptation of large language models. In The Tenth International Conference on Learning Representations, ICLR 2022, Virtual Event, April 25-29, 2022, 2022.

Junchao Huang, Xinting Hu, Shaoshuai Shi, Zhuotao Tian, and Li Jiang. Edit360: 2d image edits to 3d assets from any angle. In Proceedings ofthe IEEE/CVF International Conference on Computer Vision, 2025.

Ziqi Huang, Yinan He, Jiashuo Yu, Fan Zhang, Chenyang Si, Yuming Jiang, Yuanhan Zhang, Tianxing Wu, Qingyang Jin, Nattapol Chanpaisit, et al. Vbench: Comprehensive benchmark suite for video generative models. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2024.

Team Hunyuan3D, Shuhui Yang, Mingxin Yang, Yifei Feng, Xin Huang, Sheng Zhang, Zebin He, Di Luo, Haolin Liu, Yunfei Zhao, et al. Hunyuan3d 2.1: From images to high-fidelity 3d assets with production-ready pbr material. arXiv preprint arXiv:2506.15442, 2025.

Denys Iliash, Jiayi Liu, Egor Fokin, Qirui Wu, Ali Mahdavi-Amiri, Manolis Savva, and Angel X. Chang. Artiverse: A diverse and physically grounded dataset for articulated objects. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2026.

Adobe Systems Inc. Mixamo. https://www.mixamo.com/, 2025.

Robert A Jacobs, Michael I Jordan, Steven J Nowlan, and Geoffrey E Hinton. Adaptive mixtures of local experts. Neural computation, 1991.

Dongfu Jiang, Max Ku, Tianle Li, Yuansheng Ni, Shizhuo Sun, Rongqi Fan, and Wenhu Chen. Genai arena: An open evaluation platform for generative models. Advances in Neural Information Processing Systems, 2024.

Qing Jiang, Junan Huo, Xingyu Chen, Yuda Xiong, Zhaoyang Zeng, Yihao Chen, Tianhe Ren, Junzhi Yu, and Lei Zhang. Detect anything via next point prediction. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2026.

Junjie Ke, Qifei Wang, Yilin Wang, Peyman Milanfar, and Feng Yang. Musiq: Multi-scale image quality transformer, 2021. URL https://arxiv.org/abs/2108.05997.

Bernhard Kerbl, Georgios Kopanas, Thomas Leimkuehler, and George Drettakis. 3d gaussian splat ting for real-time radiance field rendering. ACM Trans. Graph., 2023.

Mukul Khanna, Yongsen Mao, Hanxiao Jiang, Sanjay Haresh, Brennan Shacklett, Dhruv Batra, Alexander Clegg, Eric Undersander, Angel X Chang, and Manolis Savva. Habitat synthetic scenes dataset (hssd-200): An analysis of 3d scene scale and realism tradeoffs for objectgoal navigation. In Proceedings ofthe IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2024.

Diederik P Kingma and Max Welling. Auto-encoding variational bayes. arXiv preprint arXiv:1312.6114, 2013.

Arno Knapitsch, Jaesik Park, Qian-Yi Zhou, and Vladlen Koltun. Tanks and temples: benchmarking large-scale scene reconstruction. ACM Trans. Graph., 2017.

Vladimir Kulikov, Matan Kleiner, Inbar Huberman-Spiegelglas, and Tomer Michaeli. Flowedit: Inversion-free text-based editing using pre-trained flow models. In Proceedings of the IEEE/CVF International Conference on Computer Vision, 2025.

Black Forest Labs. FLUX.2: Frontier Visual Intelligence. https://bfl.ai/blog/flux-2, 2025.

Junnan Li, Dongxu Li, Caiming Xiong, and Steven Hoi. Blip: Bootstrapping language-image pretraining for unified vision-language understanding and generation. In International conference on machine learning. PMLR, 2022.

Lin Li, Haoran Feng, Zehuan Huang, Haohua Chen, Wenbo Nie, Shaohua Hou, Keqing Fan, Pan Hu, Sheng Wang, Buyu Li, et al. Segvigen: Repurposing 3d generative model for part segmentation. arXiv preprint arXiv:2603.16869, 2026a.

Lin Li, Zehuan Huang, Haoran Feng, Gengxiong Zhuang, Rui Chen, Chunchao Guo, and Lu Sheng. Voxhammer: Training-free precise and coherent 3d editing in native 3d space. In 2026 Interna tional Conference on 3D Vision (3DV), 2026b.

Yang Li, Hikari Takehara, Takafumi Taketomi, Bo Zheng, and Matthias Nießner. 4dcomplete: Nonrigid motion estimation beyond the observable surface. In Proceedings of the IEEE/CVF International Conference on Computer Vision, 2021.

Yuhan Li, Yishun Dou, Yue Shi, Yu Lei, Xuanhong Chen, Yi Zhang, Peng Zhou, and Bingbing Ni. Focaldreamer: Text-driven 3d editing via focal-fusion assembly. In Proceedings of the AAAI conference on artificial intelligence, 2024.

Hanwen Liang, Yuyang Yin, Dejia Xu, Hanxue Liang, Zhangyang Wang, Konstantinos N. Plataniotis, Yao Zhao, and Yunchao Wei. Diffusion4d: Fast spatial-temporal consistent 4d generation via video diffusion models. In Advances in Neural Information Processing Systems 37: Annual Conference on Neural Information Processing Systems 2024, NeurIPS 2024, Vancouver, BC, Canada, December 10 - 15, 2024, 2024.

Sen Liang, Cong Wang, Zhentao Yu, Fengbin Guan, Zhengguang Zhou, Teng Hu, Youliang Zhang, Yuan Zhou, Xin Li, Qinglin Lu, et al. Goku: A million-scale universal dataset and benchmark for instruction-based video editing. arXiv preprint arXiv:2606.30599, 2026.

Siwoo Lim, Sunjae Yoon, Gwanhyeong Koo, Hyeonseo Yun, and Chang D Yoo. Tango: Trainingfree 3d editing via tangent-space guidance and optimization. arXiv preprint arXiv:2607.14927, 2026.

Yaron Lipman, Ricky T. Q. Chen, Heli Ben-Hamu, Maximilian Nickel, and Matthew Le. Flow matching for generative modeling. In The Eleventh International Conference on Learning Representations, 2023.

Shitong Luo and Wei Hu. Diffusion probabilistic models for 3d point cloud generation. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, 2021.

Ziqi Ma, Hongqiao Chen, Yisong Yue, and Georgia Gkioxari. Feedforward 3d editing via textsteerable image-to-3d. arXiv preprint arXiv:2512.13678, 2025.

Ben Mildenhall, Pratul P Srinivasan, Matthew Tancik, Jonathan T Barron, Ravi Ramamoorthi, and Ren Ng. Nerf: Representing scenes as neural radiance fields for view synthesis. Communications ofthe ACM, 2021.

Kaichun Mo, Shilin Zhu, Angel X Chang, Li Yi, Subarna Tripathi, Leonidas J Guibas, and Hao Su. Partnet: A large-scale benchmark for fine-grained and hierarchical part-level 3d object understanding. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, 2019.

Norman Muller, Yawar Siddiqui, Lorenzo Porzi, Samuel Rota Bulo, Peter Kontschieder, and¨ Matthias Nießner. Diffrf: Rendering-guided 3d radiance field diffusion. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, 2023.

Alex Nichol, Heewoo Jun, Prafulla Dhariwal, Pamela Mishkin, and Mark Chen. Point-e: A system for generating 3d point clouds from complex prompts. arXiv preprint arXiv:2212.08751, 2022.

Francesco Palandra, Andrea Sanchietti, Daniele Baieri, and Emanuele Rodola. Gsedit: Efficient text-guided editing of 3d objects via gaussian splatting. arXiv preprint arXiv:2403.05154, 2024.

Ben Poole, Ajay Jain, Jonathan T. Barron, and Ben Mildenhall. Dreamfusion: Text-to-3d using 2d diffusion. In The Eleventh International Conference on Learning Representations, 2023.

Zhangyang Qi, Yunhan Yang, Mengchen Zhang, Long Xing, Xiaoyang Wu, Tong Wu, Dahua Lin, Xihui Liu, Jiaqi Wang, and Hengshuang Zhao. Tailor3d: Customized 3d assets editing and generation with dual-side images. CoRR, 2024.

Huaizhi Qu, Ruichen Zhang, Shuqing Luo, Luchao Qi, Zhihao Zhang, Xiaoming Liu, Roni Sengupta, and Tianlong Chen. Editcast3d: Single-frame-guided 3d editing with video propagation and view selection. arXiv preprint arXiv:2510.13652, 2025.

Qwen Team. Qwen3.5: Towards native multimodal agents, February 2026a. URL https:// qwen.ai/blog?id=qwen3.5.

Qwen Team, 2026b. URL https://qwen.ai/blog?id=qwen3.6-27b.

Nikhila Ravi, Valentin Gabeur, Yuan-Ting Hu, Ronghang Hu, Chaitanya Ryali, Tengyu Ma, Haitham Khedr, Roman Radle, Chloe Rolland, Laura Gustafson, Eric Mintun, Junting Pan, Kalyan Va-¨ sudev Alwala, Nicolas Carion, Chao-Yuan Wu, Ross Girshick, Piotr Dollar, and Christoph Fe-´ ichtenhofer. Sam 2: Segment anything in images and videos. arXiv preprint arXiv:2408.00714, 2024. URL https://arxiv.org/abs/2408.00714.

Litu Rout, Yujia Chen, Nataniel Ruiz, Constantine Caramanis, Sanjay Shakkottai, and Wen-Sheng Chu. Semantic image inversion and editing using rectified stochastic differential equations. arXiv preprint arXiv:2410.10792, 2024.

Etai Sella, Gal Fiebelman, Peter Hedman, and Hadar Averbuch-Elor. Vox-e: Text-guided voxel editing of 3d objects. In Proceedings of the IEEE/CVF international conference on computer vision, 2023.

Oriane Simeoni, Huy V Vo, Maximilian Seitzer, Federico Baldassarre, Maxime Oquab, Cijo Jose,´ Vasil Khalidov, Marc Szafraniec, Seungeun Yi, Michael Ramamonjisoa, et al. Dinov3.¨ arXiv preprint arXiv:2508.10104, 2025.

Yang Song and Stefano Ermon. Generative modeling by estimating gradients of the data distribution. In Advances in Neural Information Processing Systems, 2019.

Yang Song, Jascha Sohl-Dickstein, Diederik P. Kingma, Abhishek Kumar, Stefano Ermon, and Ben Poole. Score-based generative modeling through stochastic differential equations. In 9th International Conference on Learning Representations, 2021.

Quan Sun, Yuxin Fang, Ledell Wu, Xinlong Wang, and Yue Cao. Eva-clip: Improved training techniques for clip at scale. arXiv preprint arXiv:2303.15389, 2023.

Jiangshan Wang, Junfu Pu, Zhongang Qi, Jiayi Guo, Yue Ma, Nisha Huang, Yuxin Chen, Xiu Li, and Ying Shan. Taming rectified flow for inversion and editing. In International Conference on Machine Learning, 2025a.

Penghao Wang, Yiyang He, Xin Lv, Yukai Zhou, Lan Xu, Jingyi Yu, and Jiayuan Gu. Partnext: A next-generation dataset for fine-grained and hierarchical 3d part understanding. In Advances in Neural Information Processing Systems 38: Annual Conference on Neural Information Processing Systems 2025, NeurIPS 2025, San Diego,CA, USA, December 2-7, 2025 / Mexico City, Mexico, November 30 - December5, 2025, 2025b.

Tengfei Wang, Bo Zhang, Ting Zhang, Shuyang Gu, Jianmin Bao, Tadas Baltrusaitis, Jingjing Shen, Dong Chen, Fang Wen, Qifeng Chen, et al. Rodin: A generative model for sculpting 3d digital avatars using diffusion. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, 2023.

Zhou Wang, Alan C. Bovik, Hamid R. Sheikh, and Eero P. Simoncelli. Image quality assessment: from error visibility to structural similarity. IEEE Trans. Image Process., 2004.

Zidong Wang, Yiyuan Zhang, Xiaoyu Yue, Xiangyu Yue, Yangguang Li, Wanli Ouyang, and Lei Bai. Transition models: Rethinking the generative learning objective. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2026.

Jiawei Weng, Saining Zhang, Zhenxin Diao, Peishuo Li, Henghaofan Zhang, Junhao Chen, and Hao Zhao. Feedforward 3d editing learns from semantic-part transformation. arXiv preprint arXiv:2605.27351, 2026.

Ruihao Xia, Yang Tang, and Pan Zhou. Towards scalable and consistent 3d editing. arXiv preprint arXiv:2510.02994, 2025.

Fanbo Xiang, Yuzhe Qin, Kaichun Mo, Yikuan Xia, Hao Zhu, Fangchen Liu, Minghua Liu, Hanxiao Jiang, Yifu Yuan, He Wang, Li Yi, Angel X. Chang, Leonidas J. Guibas, and Hao Su. SAPIEN: A simulated part-based interactive environment. In The IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2020.

Jianfeng Xiang, Zelong Lv, Sicheng Xu, Yu Deng, Ruicheng Wang, Bowen Zhang, Dong Chen, Xin Tong, and Jiaolong Yang. Structured 3d latents for scalable and versatile 3d generation. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2025.

Jianfeng Xiang, Xiaoxue Chen, Sicheng Xu, Ruicheng Wang, Zelong Lv, Yu Deng, Hongyuan Zhu, Yue Dong, Hao Zhao, Nicholas Jing Yuan, et al. Native and compact structured latents for 3d generation. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2026.

Sidi Yang, Tianhe Wu, Shuwei Shi, Shanshan Lao, Yuan Gong, Mingdeng Cao, Jiahao Wang, and Yujiu Yang. Maniqa: Multi-dimension attention network for no-reference image quality assessment. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2022.

Yunhan Yang, Yukun Huang, Yuan-Chen Guo, Liangjun Lu, Xiaoyang Wu, Edmund Y Lam, Yan-Pei Cao, and Xihui Liu. Sampart3d: Segment any part in 3d objects. arXiv preprint arXiv:2411.07184, 2024.

Junliang Ye, Shenghao Xie, Ruowen Zhao, Zhengyi Wang, Hongyu Yan, Wenqiang Zu, Lei Ma, and Jun Zhu. Nano3d: A training-free approach for efficient 3d editing without masks. arXiv preprint arXiv:2510.15019, 2025a.

Yang Ye, Xianyi He, Zongjian Li, Bin Lin, Shenghai Yuan, Zhiyuan Yan, Bohan Hou, and Li Yuan. Imgedit: A unified image editing dataset and benchmark. In Advances in Neural Information Processing Systems 38: Annual Conference on Neural Information Processing Systems 2025, NeurIPS 2025, San Diego, CA, USA, December 2-7, 2025 / Mexico City, Mexico, November 30 - December 5, 2025, 2025b.

Tianwei Yin, Michael Gharbi, Taesung Park, Richard Zhang, Eli Shechtman, Fr¨ edo Durand, and´ Bill Freeman. Improved distribution matching distillation for fast image synthesis. In Advances in Neural Information Processing Systems 37: Annual Conference on Neural Information Processing Systems 2024, NeurIPS 2024, Vancouver, BC, Canada, December 10 - 15, 2024, 2024a.

Tianwei Yin, Michael Gharbi, Richard Zhang, Eli Shechtman, Fr ¨ edo Durand, William T. Freeman,´ and Taesung Park. One-step diffusion with distribution matching distillation. In IEEE/CVF Conference on Computer Vision and Pattern Recognition, CVPR 2024, Seattle, WA, USA, June 16-22, 2024, 2024b.

Qifan Yu, Wei Chow, Zhongqi Yue, Kaihang Pan, Yang Wu, Xiaoyang Wan, Juncheng Li, Siliang Tang, Hanwang Zhang, and Yueting Zhuang. Anyedit: Mastering unified high-quality image editing for any idea. In Proceedings of the Computer Vision and Pattern Recognition Conference, 2025.

Xiaohua Zhai, Basil Mustafa, Alexander Kolesnikov, and Lucas Beyer. Sigmoid loss for language image pre-training. In Proceedings of the IEEE/CVF international conference on computer vision, 2023.

Lvmin Zhang, Anyi Rao, and Maneesh Agrawala. Adding conditional control to text-to-image diffusion models. In 2023 IEEE/CVF International Conference on Computer Vision (ICCV), 2023.

Richard Zhang, Phillip Isola, Alexei A. Efros, Eli Shechtman, and Oliver Wang. The unreasonable effectiveness of deep features as a perceptual metric. In 2018 IEEE Conference on Computer Vision and Pattern Recognition, CVPR 2018, Salt Lake City, UT, USA, June 18-22, 2018, 2018.

Yibo Zhang, Li Zhang, Rui Ma, and Nan Cao. Texverse: A universe of 3d objects with highresolution textures. arXiv preprint arXiv:2508.10868, 2025.

Yang Zheng, Mengqi Huang, Nan Chen, and Zhendong Mao. Pro3d-editor: A progressive-views perspective for consistent and precise 3d editing. Advances in Neural Information Processing Systems, 2025.

Junsheng Zhou, Jinsheng Wang, Baorui Ma, Yu-Shen Liu, Tiejun Huang, and Xinlong Wang. Uni3d: Exploring unified 3d representation at scale. In International Conference on Learning Representations, 2024.

Yu Zhou, Xiaoyan Yang, Bojia Zi, Lihan Zhang, Ruijie Sun, Weishi Zheng, Haibin Huang, Chi Zhang, and Xuelong Li. Point2insert: Video object insertion via sparse point guidance. arXiv preprint arXiv:2602.04167, 2026a.

Zhenglin Zhou, Fan Ma, Chengzhuo Gui, Xiaobo Xia, Hehe Fan, Yi Yang, and Tat-Seng Chua. Anchorflow: Training-free 3d editing via latent anchor-aligned flows. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2026b.

## A APPENDIX

## A.1 MODEL TRAINING DETAILS AND COMPUTE REPORT

## A.1.1 HYPERPARAMETERS

LoRA. For Alchemy3D-Segment, we attach a LoRA module to the query, key, value, and output projections of each transformer block, with rank 96. For Alchemy3D-Turbo, in addition to these projection layers, we attach LoRA modules to the FFN, AdaLN modulation, input and output layers, and the two time embedders, with rank 256.

Classifier-free Guidance. Both MeanFlow (Geng et al., 2025) and DMD (Yin et al., 2024b;a) fuse classifier-free guidance into the model during training. We follow most of the guidance settings used when sampling the pretrained generator. We found, however, that a MeanFlow model with guidance disabled, as used in the original sampler, produced noisy textures. We therefore set the guidance strength to 3.0 when training the third-stage PBR editing model.

## A.1.2 TRAINING COMPUTE REPORT

We report the core training configurations and compute resources in Tab. 8 to facilitate reproduction.

Table 8: Training configurations of our models. Stage 1 refers to sparse-structure editing, stage 2 to Shape SLat editing, and stage 3 to PBR SLat editing.
<table><tr><td>Model Name</td><td>Stage Initialization</td><td>Steps</td><td>Training Batch Size</td><td># GPUs (NVIDIA H200) Time (hours)</td><td>Training</td></tr><tr><td>Alchemy3D</td><td>1 TRELLIS.2</td><td></td><td>62.5K 192</td><td>16</td><td>~83.33</td></tr><tr><td>Alchemy3D</td><td>2 TRELLIS.2</td><td>65K</td><td>128</td><td>16</td><td>~133.61</td></tr><tr><td>Alchemy3D</td><td>3 TRELLIS.2</td><td>65K</td><td>128</td><td>16</td><td>~162.50</td></tr><tr><td>Alchemy3D-Flux</td><td>3 TRELLIS.2-Flux</td><td>50K</td><td>128</td><td>16</td><td>~125</td></tr><tr><td>Alchemy3D-Instruct</td><td>1 TRELLIS.2-Instruct</td><td>75K</td><td>192</td><td>16</td><td>~105.83</td></tr><tr><td>Alchemy3D-Instruct</td><td>2 TRELLIS.2-Instruct</td><td>40K</td><td>128</td><td>16</td><td>~100</td></tr><tr><td>Alchemy3D-Instruct</td><td>3 TRELLIS.2-Instruct</td><td>60K</td><td>96</td><td>16</td><td>~100</td></tr><tr><td>Alchemy3D-Segment (Trial)</td><td>3 Alchemy3D</td><td>10K</td><td>64</td><td>8</td><td>~10.83</td></tr><tr><td>Alchemy3D-Segment</td><td>3 Alchemy3D-Segment (Trial)</td><td>20K</td><td>64</td><td>8</td><td>~28.89</td></tr><tr><td>Alchemy3D-MF</td><td>1 Alchemy3D</td><td>15K</td><td>288</td><td>24</td><td>~74.44</td></tr><tr><td>Alchemy3D-MF</td><td>2 Alchemy3D</td><td>15K</td><td>192</td><td>24</td><td>~65.83</td></tr><tr><td>Alchemy3D-MF</td><td>3 Alchemy3D</td><td>10K</td><td>192</td><td>24</td><td>~55.83</td></tr><tr><td>Alchemy3D-Turbo</td><td>1 Alchemy3D-MF</td><td>12K</td><td>288</td><td>24</td><td>~56.67</td></tr><tr><td>Alchemy3D-Turbo</td><td>2 Alchemy3D-MF</td><td>12K</td><td>96</td><td>24</td><td>~37.78</td></tr><tr><td>Alchemy3D-Turbo</td><td>3 Alchemy3D-MF</td><td>14K</td><td>96</td><td>24</td><td>~43.33</td></tr></table>

## A.2 ALCHEMY3D-1M CONSTRUCTION PIPELINE

In this section, we describe the construction pipeline of Alchemy3D-1M (Sec. 3). It has three stages: asset preparation, type-specific construction, and data filtering with condition generation.

## A.2.1 STAGE 1: ASSET PREPARATION

Animation. For animation editing, we collect assets with temporal motion from multiple public datasets, including the animated subset of Objaverse-XL (Deitke et al., 2023) filtered by Diffu sion4D (Liang et al., 2024), TexVerse (Zhang et al., 2025), and DeformingThings4D (Li et al., 2021). To further diversify character-motion combinations, we randomly pair rigged characters from Mixamo (Inc., 2025) with different motion sequences. We additionally incorporate articulated assets from ArtiVerse (Iliash et al., 2026) and PartNet-Mobility (Mo et al., 2019).

Segmentation. For part segmentation, we collect assets with dense semantic part annotations from PartNet (Mo et al., 2019), PartNeXT (Wang et al., 2025b), and PartVerseXL (Ding et al., 2025). These annotations are directly used to construct the corresponding segmentation supervision.

Other editing types. For the remaining five edit categories, namely addition, removal, replacement, local appearance editing, and global appearance editing, we adopt a shared preparation pipeline. We first construct a large pool of source images by rendering assets from existing 3D datasets (Deitke et al., 2023; Fu et al., 2021; Chang et al., 2015; Collins et al., 2022; Zhang et al., 2025; Khanna et al., 2024), together with internally generated asset-centric images synthesized by a text-to-image foundation model (Labs, 2025). Given each source image, Qwen3-VL (Bai et al., 2025) generates edit instructions according to edit-type-specific prompting templates. The source image and generated instruction are then provided to FLUX.2-Dev-Turbo (Labs, 2025) to synthesize the corresponding edited target image. The generated source-target image pairs are subsequently verified using Qwen3-VL to remove semantically inconsistent or low-quality edits.

For each retained source image, we reconstruct the corresponding source 3D asset using TREL-LIS.2 (Xiang et al., 2026). To ensure the quality of the reconstructed assets, we render multiple views and employ Qwen3-VL to assess their geometry and appearance, discarding assets with noticeable reconstruction artifacts or quality issues. For the accepted source assets, we additionally preserve the intermediate diffusion trajectories, which are later reused for trajectory-aware 3D inpainting to preserve the unedited regions during editing.

## A.2.2 STAGE 2: TYPE-SPECIFIC CONSTRUCTION PIPELINE

Animation. For assets with existing motion sequences, we uniformly sample keyframes at fixed temporal intervals and randomly pair different keyframes of the same asset to construct animation editing pairs. For articulated assets, we use SAPIEN (Xiang et al., 2020) to simulate joint interactions for each annotated joint, obtain the resulting motion trajectories, and similarly sample and randomly pair keyframes from the same asset.

Segmentation. For each asset with K annotated parts, we construct a K-color palette with wellseparated hues. We first randomly sample the initial hue $h _ { 0 }$ and generate the remaining hues using the golden-ratio increment:

$$
\begin{array} { c } { { h _ { 0 } \sim \mathcal { U } \left[ 0 , 1 \right) , } } \\ { { h _ { i + 1 } = \left( h _ { i } + \phi \right) \mod 1 , } } \end{array}\tag{6}
$$

where $\textstyle \phi = { \frac { { \sqrt { 5 } } - 1 } { 2 } } \approx 0 . 6 1 8$ is the golden-ratio increment. For each part, the saturation and value are independently sampled as $s _ { i } \sim \mathcal { U } \left[ 0 . 7 , 0 . 9 \right]$ and $v _ { i } \sim \mathcal { U } \left[ 0 . 8 , 0 . 9 5 \right]$ ]. The resulting HSV colors are converted to RGB and assigned to mesh faces according to their annotated part labels, producing the corresponding part segmentation representation.

Addition/Removal. Addition and removal are constructed using a shared segment-and-remove pipeline. We independently construct image-editing vocabularies for the two editing types rather than deriving one exclusively by reversing the other, thereby avoiding the limited diversity and potential distributional bias introduced by relying on a single editing direction.

For each image pair, we reconstruct the asset in the state where the edited object is present using TRELLIS.2. For removal, this reconstructed asset directly serves as the source asset. For addition, it instead serves as the target asset, and the resulting pair is reversed after constructing its object-absent counterpart.

We render multiple views of the reconstructed asset and employ Rex-Omni (Jiang et al., 2026) to identify the object to be removed. SAM3 (Carion et al., 2025) is subsequently used to obtain dense 2D segmentations. Among all rendered views, we select the view that jointly maximizes the visible area and the number of detected instances of the segmented object, thereby retaining as many target components as possible. The selected rendering is then converted into a two-color map, where the segmented object instances are colored white and the remaining objects are colored dark gray. Together with the corresponding 3D asset, this map is provided to SegViGen (Li et al., 2026a) to recover the editable region in 3D. We further apply binary clustering to the predicted segmentation, assuming two clusters corresponding to the editable and unedited regions, and use the resulting cluster assignment to construct the final binary 3D mask.

Given the recovered 3D mask, we construct the object-absent counterpart through trajectory-aware 3D inpainting. Instead of re-noising the reconstructed asset during sampling, we directly reuse the diffusion trajectory recorded when generating the original asset. At each sampling step, the latent in the unedited region is replaced with the corresponding latent from the stored trajectory:

$$
\begin{array} { r l } & { \tilde { z } _ { t } = x _ { t } \odot ( 1 - M ) + z _ { t } ^ { \prime } \odot M , } \\ & { z _ { t - \Delta t } ^ { \prime } = \tilde { z } _ { t } - \Delta t v _ { \theta } ( \tilde { z } _ { t } , t , c ) , } \end{array}\tag{7}
$$

where $\Delta t > 0$ is the sampling interval, M is the binary 3D mask with $M = 1$ indicating the editable $\mathrm { r e g i o n } , x _ { t }$ is the latent from the stored diffusion trajectory of the original asset at timestep t, and $z _ { t } ^ { \prime }$ is the current latent generated under the edited condition c. The mask is projected to the corresponding representations, and the same procedure is applied throughout the sparse-structure, Shape SLat, and PBR SLat generation stages. Consequently, the unedited regions remain explicitly constrained by the original generation trajectory throughout the reconstruction process.

Replacement. Unlike addition and removal, replacement edits simultaneously involve the removal of an existing object and the introduction of a new one, making the edited region more challenging to localize through direct segmentation. We therefore estimate the editable region by comparing the sparse structures before and after editing.

Specifically, we employ FlowEdit (Kulikov et al., 2025) to approximately transform the source sparse structure conditioned on the edited image. The spatial differences between the resulting structure and the original source structure are used to estimate the editable region, from which we construct a binary 3D mask. Importantly, FlowEdit is used only for estimating the spatial extent of the edit rather than generating the final edited asset. The estimated mask is subsequently used in the trajectory-aware 3D inpainting procedure described above to generate the final geometry and appearance while preserving the unedited regions of the source asset.

Local Appearance. For local appearance editing, we adopt the same 3D mask construction procedure as in addition and removal to localize the edited region. We preserve the sparse structure of the source asset and perform mask-guided 3D inpainting only during the Shape SLat and PBR SLat generation stages. We do not hold the Shape SLat fixed, because appearance modifications, such as material or style changes, may also require adjustments to fine geometric details while preserving the overall structure of the asset.

Global Appearance. For global appearance editing, no spatial mask is required. We preserve the sparse structure of the source asset and regenerate the Shape SLat and PBR SLat conditioned on the edited image, allowing the appearance of the asset to be globally modified while maintaining its overall structure.

## A.2.3 STAGE 3: DATA FILTERING AND CONDITION GENERATION

The preceding stages produce candidate editing pairs, which may still contain reconstruction artifacts, insufficient editing changes, or other quality issues. We therefore design edit-type-specific filtering procedures to remove undesirable samples and improve the overall quality of Alchemy3D-1M. For the retained samples, we further generate editing instructions and image-based conditions for subsequent model training.

Animation. To remove samples with insufficient motion changes, we render eight fixed circular views of both the source and target assets. For each corresponding view, we compute the cosine similarity between their DINOv3 (Simeoni et al., 2025) features:´

$$
s _ { v } = \cos ( f _ { \mathrm { D I N O } } ( R _ { v } ( X _ { s } ) ) , f _ { \mathrm { D I N O } } ( R _ { v } ( X _ { t } ) ) ) , \quad v = 1 , \ldots , 8 .\tag{8}
$$

We retain a pair only if

$$
\operatorname* { m i n } _ { v = 1 , \dots , 8 } s _ { v } < \tau ,\tag{9}
$$

where $\tau$ is a predefined similarity threshold. This filtering removes pairs for which the motion change is insufficiently apparent from all considered viewpoints. For each retained pair, the eight source and target renderings are concatenated into a grid and provided to Qwen3.6-27B (Qwen Team, 2026b) to generate editing instructions describing the motion changes. To increase linguistic diversity, we generate 3–5 instructions for each pair using different phrasings while preserving the same editing semantics. For image-conditioned editing, we additionally render eight random views of the target asset as image conditions.

![](images/cca9ee9a016816dae744faec1e64df71523ec3f25b347f3d514298f28bdfeebd.jpg)  
Figure 6: Overview of the construction pipeline and evaluation dimensions of GEdit3D-Bench.

Segmentation. For segmentation editing, we render eight random views of the part-colored assets using flat colors rather than conventional lighting. This rendering scheme is designed to mimic practical 2D segmentation-map conditions, where the regions predicted by a 2D segmentation model (Ravi et al., 2024; Carion et al., 2025) can be represented as manually assigned flat colors. The resulting multi-view renderings are used as image conditions for segmentation editing.

Other editing types. For addition, removal, replacement, local appearance, and global appearance editing, we render multiple views of each editing pair and concatenate them into grid images for quality filtering and recaptioning with Qwen3.6-27B (Qwen Team, 2026b). For addition, removal, replacement, and local appearance editing, we additionally render 16 random views of each target asset and use Qwen3.6-27B to determine whether the edited region is visible in each view. Views in which the edited region is not sufficiently visible are discarded. For global appearance editing, where the change is not spatially localized, we render eight random views instead.

## A.3 GEDIT3D-BENCH

## A.3.1 LIMITATIONS OF EXISTING BENCHMARKS

The quality of a benchmark and its evaluation protocol can strongly influence the development of a research field. A meaningful benchmark should provide a fair, reliable, and realistic assessment of model capabilities while remaining aligned with practical applications. However, existing benchmarks for 3D asset editing still suffer from several fundamental limitations.

Limited Scale and Diversity. Existing benchmarks typically evaluate models on a relatively small number of manually curated assets. For example, Edit3D-Bench (Li et al., 2026b) selects only 100 assets from GSO (Downs et al., 2022) and PartObjectVerse-Tiny (Yang et al., 2024). Similarly, Eval3DEdit (Zhou et al., 2026b) and TANGOEdit (Lim et al., 2026) curate approximately 100 assets from datasets including Objaverse-XL (Deitke et al., 2023) and GSO (Downs et al., 2022). While these benchmarks provide valuable initial evaluations, their limited scale restricts the coverage of asset categories, appearances, geometric structures, and editing scenarios, making it difficult to assess the robustness of 3D editing models in diverse real-world settings.

Low-Quality Ground Truth and In-distribution Evaluation. Another line of work (Ma et al., 2025; Ye et al., 2025a; Xia et al., 2025; Weng et al., 2026) constructs evaluation sets by splitting the training data of the corresponding method into training and test subsets. Models are then evaluated by measuring the similarity between generated outputs and the provided edited 3D assets. This protocol has several limitations. First, existing 3D editing datasets contain annotation and generation errors, including Alchemy3D-1M (Sec. 3). Treating generated or reconstructed assets as ground truth can therefore pass those artifacts into the score. Second, 3D asset editing is a conditional generation task with multiple valid solutions, so similarity to a single reference is an incomplete measure of editing quality. Third, a random split of the same distribution used for training measures indistribution fit. A model may score well by matching dataset-specific patterns without generalizing to unseen assets and edits.

We provide visualizations of samples for the Edit3D-Bench proposed by Steer3D in Fig. 11, for Nano3D-100K in Fig. 12, and for 3DEditVerse in Fig. 13.

Insufficient Evaluation Dimensions. Existing evaluation protocols also provide limited coverage of the diverse requirements of 3D asset editing. In particular, measuring similarity to an edited reference alone does not fully characterize whether a model successfully executes the requested modifi cation, preserves the identity and irrelevant attributes of the source asset, or produces a perceptually high-quality result. A comprehensive benchmark should therefore jointly assess multiple aspects of editing quality, including editing success, instruction following, source identity preservation, visual quality, and semantic alignment with the desired edit.

## A.3.2 CONSTRUCTION PIPELINE

Each sample in GEdit3D-Bench consists of a source 3D asset, a rendered source image, a naturallanguage editing instruction, and a corresponding target edited image, along with the text captions for source and target assets. Given a source asset A and an editing instruction I or edited image I<sub>tgt</sub>, a 3D editing model is expected to generate an edited asset A<sup>′</sup> that satisfies the requested modification while preserving the identity and irrelevant attributes of the source asset. The construction pipeline consists of four stages.

3D Asset Synthesis and Collection. We first define 21 high-level asset categories covering a broad range of semantic domains, including characters, animals, vehicles, furniture, architecture, and daily objects. For each category, Gemini-3.5-Flash<sup>3</sup> generates diverse asset descriptions, which are subsequently converted into high-quality images using Cosmos3-Super-Text2Image (Agarwal et al., 2026). These images are then transformed into 3D assets using Hunyuan3D V3.1. To further improve the diversity and realism of the benchmark, we additionally collect real-world 3D assets from Sketchfab. During collection, we consider semantic categories, user engagement, and release timestamps. By prioritizing recently released assets, we reduce potential overlap with existing large-scale 3D generation datasets, such as Objaverse-XL (Deitke et al., 2023) and TexVerse (Zhang et al., 2025). Combining generated and real-world assets enables GEdit3D-Bench to cover a broad spectrum of use cases, object categories, visual styles, and geometric structures.

3D Asset Captioning. For each collected asset, we render multiple predefined viewpoints using Blender and concatenate them into multi-view image grids. These rendered views are provided to Gemini-3.5-Flash to generate detailed semantic descriptions of the assets. The resulting captions describe object categories, appearances, materials, structures, and distinctive characteristics, and serve as semantic references for subsequent instruction generation.

Editing Instruction and Target Image Generation. Given each source asset, we randomly sample rendered viewpoints and provide them together with the source captions to Gemini-3.5-Flash. Using category-specific prompting templates, Gemini-3.5-Flash generates diverse editing instructions covering addition, removal, replacement, local and global appearance modification, and animation. For each instruction, HunyuanImage3.0-Instruct (Cao et al., 2025) performs image-level editing on the rendered source views to generate corresponding target images. These edited images provide visual references for evaluating whether a 3D editing model correctly interprets and executes the requested modification.

Multi-stage Quality Curation. To ensure the reliability of benchmark samples, each editing triplet, consisting of a source rendering, editing instruction, and edited image, undergoes multi-stage quality verification. Specifically, we employ Gemini-3.5-Flash, GPT-5.6-Sol, and human experts to filter samples with incorrect semantics, unrealistic modifications, inconsistent object identities, or low-quality editing results. After quality control, we randomly sample the remaining high-quality triplets to form the final benchmark set. Finally, source captions and editing instructions are provided to GPT-5.6-Sol to generate target captions describing the expected edited assets.

## A.3.3 BENCHMARK STATISTICS

![](images/bd72abb5f0cba362bcd30b5db0fe6e7a0fd6b1f5192118d34e81b44f60830cf6.jpg)

![](images/fd80d536e85040b74bdaa5244388d1c6784783a2d472367dc6c3a3175e320213.jpg)

![](images/597a5889f5297f3dbcaccc42acb40e1943fdcaa165c7130d460c5e39119ea2bf.jpg)  
Figure 7: Statistics of GEdit3D-Bench. The left panel shows the number of samples per editing type, the middle panel shows the distribution of instruction length, and the right panel visualizes instruction embeddings from Qwen3-Embedding-8B.

As shown in Fig. 7, GEdit3D-Bench contains 400 samples for addition, removal, and local appearance, and 300 samples for replacement, animation, and global appearance. The editing instructions are about a dozen words long, as in our training set, and describe short atomic edits. We embed these instructions with Qwen3-Embedding-8B<sup>4</sup> and visualize them with UMAP in the right panel of Fig. 7. The embedding distribution indicates that the instructions cover diverse edits.

## A.3.4 EVALUATION PROTOCOL DETAILS

View Quality Assessment. To assess the perceptual quality of edited assets, we render multiple views and compute image-quality scores for each view. Specifically, we employ the aestheticpredictor-v2.5<sup>5</sup> and established image quality assessment models, including MANIQA (Yang et al., 2022) and MUSIQ (Ke et al., 2021), implemented through pyiqa (Chen & Mo, 2022). Scores are averaged across the rendered views to obtain an overall view-quality assessment.

Reference Alignment. Pretrained encoders are widely used to evaluate semantic alignment between generated results and desired targets. We consider image alignment A<sub>I</sub> and text alignment A , instantiated with EVA-CLIP (Sun et al., 2023), SigLIP (Zhai et al., 2023), BLIP (Li et al., 2022), DINOv3 (Simeoni et al., 2025), and Uni3D (Zhou et al., 2024), where applicable.´

MLLM-based Evaluation. Recent benchmarks for video and image generation and editing (Zhou et al., 2026a; Liang et al., 2026; Ye et al., 2025b; Huang et al., 2024) have increasingly adopted modern multimodal large language models (MLLMs) for evaluation due to their strong visual understanding and reasoning capabilities. Following this direction, we instruct MLLMs to evaluate each editing result along four complementary dimensions: Success Rate (SR), Instruction Following (IF), Identity Preservation (IP), and Visual Quality (VQ). SR is a binary judgment of whether the requested edit is successfully achieved, while IF, IP, and VQ are scored on a 1–100 scale. Specifically, IF measures how faithfully the result follows the editing instruction, IP measures whether the identity and irrelevant properties of the source asset are preserved, and VQ measures the perceptual quality and plausibility of the resulting asset. SR and VQ use a shared system prompt. IF and IP use an edit-type-specific prompt, because the editing types differ substantially in granularity.

## A.4 MORE EVALUATION DETAILS AND RESULTS

## A.4.1 DETAILS OF EVALUATION SETTING

Main experiments. Unless otherwise stated, the CLIP model used in evaluation is EVA-CLIP-18B<sup>6</sup>, the SigLIP model is siglip2-giant<sup>7</sup>, the DINO model is DINOv3 ViT-L<sup>8</sup>, and the Uni3D model is uni3d-giant<sup>9</sup>. For reference alignment, each asset is rendered from ten views with camera radius 1.8 and a 49<sup>◦</sup> field of view. Eight views use a pitch of 10<sup>◦</sup> and yaw angles spaced by 45<sup>◦</sup>. The other two are front views, with yaw fixed to the front and pitch set to +30<sup>◦</sup> and −30<sup>◦</sup>. MLLM scores use the same random 400-sample subset of GEdit3D-Bench as the user study.

Instruction-driven editing by Alchemy3D. Noisy instruction captions make Alchemy3D-Instruct weaker than image-conditioned Alchemy3D, a gap also observed in text-conditioned 3D generation (Xiang et al., 2025). We therefore design a complementary pipeline that uses imageconditioned Alchemy3D for instruction-driven editing. Given a source asset and an editing instruction, we render 16 random views and ask a VLM to select the view best suited to the requested edit. A 2D image editor modifies that view, after which Alchemy3D edits the source asset using the modified image as its reference. We use Qwen3.8-27B<sup>10</sup> for view selection and FLUX.2-Klein-9B<sup>11</sup> for image editing.

User Study. To evaluate the methods in a realistic setting, we conducted a user study with 24 participants from nine institutions. We used the same subset as for MLLM scoring and randomly assigned 50 tasks to each participant. For each task, the source asset and outputs from all methods were loaded into Google Model Viewer<sup>12</sup>, allowing participants to inspect each asset interactively. Participants selected the best result or chose “Hard to Select”, which we counted as “Cannot Decide”. An example page is shown in Fig. 8.

![](images/f0939713a2e9283ddf4afbfc75ad6b71200bfcd982ca78d66f6b8313b478f838.jpg)  
Figure 8: Example page from the user study.

## A.5 MORE EXPERIMENTAL RESULTS

Fig. 9 reports user-selection rates across all six editing types. Alchemy3D receives the highest preference for all editing types: addition (69.2%), removal (71.8%), replacement (73.1%), animation (78.3%), local appearance (44.3%), and global appearance (39.8%). The “Cannot Decide” rate

![](images/0f9cb0172ea9d0ab18e75432ddc1df887f08832dea738f10c888e772075c2d59.jpg)  
Figure 9: Results of the user study.

Table 9: Quantitative comparison on GEdit3D-Bench. Best results are in bold and second-best are underlined.
<table><tr><td></td><td colspan="3">View Quality</td><td colspan="6">Ref. Alignment</td><td colspan="3">MLLM</td></tr><tr><td>Method</td><td></td><td>Aes. ↑ MANIQA ↑ MUSIQ ↑</td><td></td><td>ACLIP ↑</td><td>↑</td><td></td><td>↑ ADINO ↑</td><td>AUni3D ↑</td><td>↑</td><td>SR↑</td><td>VQ↑</td><td>IF↑IP↑</td></tr><tr><td colspan="9">add</td></tr><tr><td>Nano3D</td><td>4.40</td><td>0.552</td><td>73.04</td><td>80.11</td><td>65.34</td><td>86.96</td><td>66.70</td><td>36.48</td><td>80.70</td><td>56.2%</td><td>70.1</td><td>55.3 72.5</td></tr><tr><td>3DEditFormer</td><td>4.36</td><td>0.537</td><td>70.91</td><td>79.98</td><td>63.44</td><td>86.17</td><td>65.01</td><td>36.06</td><td>79.55</td><td>53.8%</td><td>63.8</td><td>56.1 59.4</td></tr><tr><td>PartFlow</td><td>4.32</td><td>0.543</td><td>70.98</td><td>79.89</td><td>62.74</td><td>86.50</td><td>65.20</td><td>36.21</td><td>79.34</td><td>57.5%</td><td>63.4</td><td>56.5 65.8</td></tr><tr><td>Alchemy3D-Turbo</td><td>4.59</td><td>0.541</td><td>73.49</td><td>82.01</td><td>69.66</td><td>88.59</td><td>70.80</td><td>36.24</td><td>80.32</td><td>85.0%</td><td>68.6</td><td>68.3 71.8</td></tr><tr><td>Alchemy3D</td><td>4.47</td><td>0.578</td><td>72.94</td><td>84.84</td><td>68.66</td><td>90.28</td><td>71.63</td><td>36.87</td><td>81.31</td><td>86.2%</td><td>72.2 69.8</td><td>76.3</td></tr><tr><td colspan="9">remove</td><td colspan="3"></td></tr><tr><td>Nano3D</td><td>4.39</td><td>0.553</td><td>72.44</td><td>82.53</td><td>64.12</td><td>89.28</td><td>70.39</td><td>37.32</td><td>81.05</td><td>41.2%</td><td>70.3</td><td>48.8 70.5</td></tr><tr><td>3DEditFormer</td><td>4.33</td><td>0.548</td><td>70.69</td><td>83.27</td><td>61.80</td><td>89.20</td><td>69.93</td><td>37.06</td><td>79.63</td><td>52.5%</td><td>68.3 58.0</td><td>65.9</td></tr><tr><td>PartFlow</td><td>4.30</td><td>0.550</td><td>70.73</td><td>82.58</td><td>61.05</td><td>89.08</td><td>69.25</td><td>36.97</td><td>79.44</td><td>50.0%</td><td>66.1</td><td>53.5 65.1</td></tr><tr><td>Alchemy3D-Turbo</td><td>4.56</td><td>0.535</td><td>72.26</td><td>82.19</td><td>66.77</td><td>89.54</td><td>74.28</td><td>36.66</td><td>80.61</td><td>90.0%</td><td>70.7</td><td>71.7 73.9</td></tr><tr><td>Alchemy3D</td><td>4.41</td><td>0.572</td><td>71.64</td><td>84.85</td><td>65.93</td><td>91.23</td><td>73.67</td><td>37.15</td><td>81.24</td><td>90.0%</td><td>73.0</td><td>72.8 75.6</td></tr><tr><td colspan="9">replace</td><td colspan="3"></td></tr><tr><td>Nano3D</td><td>4.41</td><td>0.557</td><td>73.20</td><td>82.29</td><td>66.16</td><td>88.28</td><td>70.75</td><td>37.69</td><td>82.91</td><td>57.6%</td><td>70.0</td><td>58.7 73.7</td></tr><tr><td>3DEditFormer</td><td>4.34</td><td>0.545</td><td>71.06</td><td>82.38</td><td>64.77</td><td>88.19</td><td>69.66</td><td>37.68</td><td>82.83</td><td>74.6%</td><td>67.0</td><td>67.2 65.4</td></tr><tr><td>PartFlow</td><td>4.32</td><td>0.548</td><td>71.23</td><td>81.29</td><td>63.28</td><td>88.14</td><td>69.28</td><td>37.32</td><td>81.32</td><td>60.3%</td><td>66.7</td><td>59.6 69.7</td></tr><tr><td>Alchemy3D-Turbo</td><td>4.58</td><td>0.540</td><td>73.11</td><td>83.05</td><td>68.91</td><td>89.62</td><td>73.80</td><td>37.13</td><td>82.36</td><td>90.0%</td><td>70.6</td><td>72.6 74.2</td></tr><tr><td>Alchemy3D</td><td>4.43</td><td>0.578</td><td>72.53</td><td>85.10</td><td>68.22</td><td>90.67</td><td>74.60</td><td>37.41</td><td>82.99</td><td>93.3%</td><td>73.2</td><td>71.5 76.8</td></tr><tr><td colspan="10"></td><td colspan="3"></td></tr><tr><td>local_appearance</td><td>4.32</td><td>0.552</td><td>70.53</td><td>81.67</td><td>61.36</td><td>89.21</td><td>71.31</td><td>36.76</td><td>79.42</td><td>38.8%</td><td>69.7</td><td>45.2 68.5</td></tr><tr><td>PartFlow Alchemy3D-Turbo</td><td>4.60</td><td>0.541</td><td>72.93</td><td>81.30</td><td>66.73</td><td>89.30</td><td>73.60</td><td>36.45</td><td>80.43</td><td>61.3%</td><td>70.0</td><td>57.4 75.0</td></tr><tr><td>Alchemy3D</td><td>4.44</td><td>0.579</td><td>71.97</td><td>83.45</td><td>65.56</td><td>90.44</td><td>72.78</td><td>36.64</td><td>80.88</td><td>54.4%</td><td>72.4</td><td>53.4 76.0</td></tr><tr><td colspan="10"></td><td></td><td></td><td></td></tr><tr><td>global_appearance</td><td>4.36</td><td>0.542</td><td>70.14</td><td>72.04</td><td>55.31</td><td>82.01</td><td>64.38</td><td>34.20</td><td>73.50</td><td>8.3%</td><td>68.7</td><td>36.1 75.0</td></tr><tr><td>PartFlow Alchemy3D-Turbo</td><td>4.56</td><td>0.502</td><td>71.20</td><td>74.57</td><td>63.17</td><td>83.62</td><td>67.09</td><td>34.51</td><td>75.35</td><td>50.0%</td><td>68.3</td><td>53.2 76.8</td></tr><tr><td>Alchemy3D</td><td>4.45</td><td>0.565</td><td>71.05</td><td>75.72</td><td>60.79</td><td>84.46</td><td>67.82</td><td>34.36</td><td>74.99</td><td>28.3%</td><td>72.7 43.7</td><td>77.7</td></tr><tr><td colspan="10">animation</td><td colspan="3"></td></tr><tr><td>3DEditFormer</td><td></td><td>0.535</td><td>70.59</td><td>83.83</td><td>65.88</td><td>88.91</td><td>70.60</td><td>36.79</td><td>79.38</td><td>49.2%</td><td>63.9</td><td></td></tr><tr><td>Alchemy3D-Turbo</td><td>4.19 4.42</td><td>0.533</td><td>72.78</td><td>83.62</td><td>69.97</td><td>89.96</td><td>73.88</td><td>36.44</td><td>80.59</td><td>68.3%</td><td>64.8</td><td>54.9 61.6 65.1 69.2</td></tr><tr><td>Alchemy3D</td><td>4.31</td><td>0.567</td><td>72.34</td><td>86.41</td><td>69.29</td><td>91.44</td><td>75.23</td><td>37.31</td><td>82.02</td><td>66.7%</td><td>70.8</td><td>66.6 72.8</td></tr></table>

increases from 8.1–15.1% for structural and animation edits to 36.1% and 42.4% for local and global appearance edits, respectively, indicating greater ambiguity in evaluating appearance changes.

## A.5.1 IMAGE-CONDITIONED EDITING

We first report complete image-conditioned results on Eval3DEdit and GEdit3D-Bench, followed by evaluations on 3DEditVerse, the Edit3D-Bench of VoxHammer, and Nano3D-100K. Nano3D supports only addition, removal, and replacement. 3DEditFormer does not support appearance edits, and PartFlow does not support animation edits. We evaluate each baseline only on the editing types it supports.

On addition, removal, and replacement in Table 9, Alchemy3D and Alchemy3D-Turbo lead in success rate, instruction following, and most alignment scores. Removal is the clearest case: both of our models reach 90% success, against 41.2% for Nano3D and about 50% for 3DEditFormer and PartFlow. Animation follows the same ranking. Alchemy3D-Turbo reaches 68.3% success and Alchemy3D 66.7%, against 49.2% for 3DEditFormer, and Alchemy3D is also highest on image alignment and identity preservation.

Appearance edits remain weaker than structural edits. PartFlow reaches 38.8% success on local appearance and 8.3% on global appearance. Alchemy3D-Turbo reaches 61.3% and 50.0%, and Alchemy3D reaches 54.4% and 28.3%, while both preserve identity better than PartFlow. We attribute the remaining gap to task conflict under joint training: most training pairs modify both geometry and appearance, so appearance-only edits conflict with the majority of the training distribution (Sec. A.6).

Table 10: Quantitative comparison on Eval3DEdit. Best results are in bold and second-best are underlined.
<table><tr><td></td><td colspan="3">View Quality</td><td colspan="6">Ref. Alignment</td><td colspan="3">MLLM Score</td></tr><tr><td>Method</td><td></td><td> $\mathrm { A c s . \ : \ : \uparrow \ : \ : M A N I Q A \uparrow \ : \ : M U S I Q \dag \ : \ : } A _ { I } ^ { \mathrm { C L I P } } \dag \ : \ : A _ { T } ^ { \mathrm { C L I P } } \dag \ : \ : A _ { I } ^ { \mathrm { S i g L I P } } \dag \ : \ : A _ { I } ^ { \mathrm { D I N O } } \dag \ : \ : A _ { I } ^ { \mathrm { U n i 3 D } } \dag \ : \ : A _ { T } ^ { \mathrm { U n i 3 D } } \dag \ : \uparrow \ : A _ { T } ^ { \mathrm { U n i 3 D } } \dag \ : \uparrow$ </td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td>SR↑</td><td>VQ↑ IF↑IP↑</td><td></td></tr><tr><td colspan="9">add</td></tr><tr><td>Nano3D</td><td>4.65</td><td>0.547</td><td>72.09</td><td>80.58</td><td>46.70</td><td>87.89</td><td>70.46</td><td>37.94</td><td>61.59</td><td>27.3%</td><td>68.9</td><td>38.3 66.9</td></tr><tr><td>3DEditFormer</td><td>4.57</td><td>0.523</td><td>72.05</td><td>79.49</td><td>45.56</td><td>86.72</td><td>67.06</td><td>37.29</td><td>59.88</td><td>45.5%</td><td>66.2</td><td>45.8 62.3</td></tr><tr><td>PartFlow</td><td>4.60</td><td>0.549</td><td>72.25</td><td>80.63</td><td>48.47</td><td>87.37</td><td>68.44</td><td>36.72</td><td>60.96</td><td>62.5%</td><td>63.0</td><td>55.0 69.9</td></tr><tr><td>Alchemy3D-Turbo</td><td>4.75</td><td>0.537</td><td>74.80</td><td>81.99</td><td>53.05</td><td>88.93</td><td>74.61</td><td>38.80</td><td>66.20</td><td>80.0%</td><td>68.4 64.5</td><td>77.0</td></tr><tr><td>Alchemy3D</td><td>4.68</td><td>0.559</td><td>74.08</td><td>83.21</td><td>53.06</td><td>89.70</td><td>74.89</td><td>38.86</td><td>66.28</td><td>81.8%</td><td>74.1 63.7</td><td>72.5</td></tr><tr><td colspan="10">remove</td><td colspan="3"></td></tr><tr><td>Nano3D</td><td>4.65</td><td>0.543</td><td>72.37</td><td>84.27</td><td>49.12</td><td>90.48</td><td>75.47</td><td>40.90</td><td>62.51</td><td>20.0%</td><td>66.2</td><td>37.8 74.6</td></tr><tr><td>3DEditFormer</td><td>4.57</td><td>0.518</td><td>71.46</td><td>82.94</td><td>47.98</td><td>89.35</td><td>73.31</td><td>39.99</td><td>58.14</td><td>70.0%</td><td>65.0 60.7</td><td>67.8</td></tr><tr><td>PartFlow</td><td>4.49</td><td>0.531</td><td>71.45</td><td>82.45</td><td>47.10</td><td>89.05</td><td>70.25</td><td>39.91</td><td>58.71</td><td>80.0%</td><td>62.9</td><td>66.2 69.9</td></tr><tr><td>Alchemy3D-Turbo</td><td>4.64</td><td>0.554</td><td>74.40</td><td>83.09</td><td>47.09</td><td>89.93</td><td>76.70</td><td>41.06</td><td>60.66</td><td>90.0%</td><td>69.5</td><td>70.6 77.1</td></tr><tr><td>Alchemy3D</td><td>4.59</td><td>0.560</td><td>72.73</td><td>84.05</td><td>46.69</td><td>90.94</td><td>78.12</td><td>40.95</td><td>59.61</td><td>80.0%</td><td>69.3</td><td>71.0 72.0</td></tr><tr><td colspan="10">replace</td><td colspan="3"></td></tr><tr><td>Nano3D</td><td>4.51</td><td>0.542</td><td>72.20</td><td>82.06</td><td>52.64</td><td>88.20</td><td>73.65</td><td>38.73</td><td>67.49</td><td>77.8%</td><td>69.0</td><td>64.3 74.8</td></tr><tr><td>3DEditFormer</td><td>4.50</td><td>0.535</td><td>71.93</td><td>82.90</td><td>54.18</td><td>88.49</td><td>72.89</td><td>38.66</td><td>66.76</td><td>100.0%</td><td>65.1 74.0</td><td>73.6</td></tr><tr><td>PartFlow</td><td>4.43</td><td>0.542</td><td>72.44</td><td>81.70</td><td>52.37</td><td>88.28</td><td>72.77</td><td>38.77</td><td>68.68</td><td>60.0%</td><td>67.1</td><td>58.7 71.9</td></tr><tr><td>Alchemy3D-Turbo</td><td>4.54</td><td>0.536</td><td>73.94</td><td>82.95</td><td>57.54</td><td>88.84</td><td>74.72</td><td>39.62</td><td>71.84</td><td>100.0%</td><td>71.2 73.2</td><td>76.4</td></tr><tr><td>Alchemy3D</td><td>4.54</td><td>0.553</td><td>73.46</td><td>84.04</td><td>56.56</td><td>89.75</td><td>77.63</td><td>39.96</td><td>71.72</td><td>90.0%</td><td>69.5</td><td>73.5</td></tr><tr><td colspan="10">style</td><td colspan="3">67.0</td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>3DEditFormer PartFlow</td><td>4.61 4.58</td><td>0.507 0.532</td><td>70.84 71.63</td><td>81.05 78.10</td><td>51.72 50.68</td><td>88.57 86.29</td><td>67.24 64.58</td><td>39.17 37.17</td><td>66.35 65.70</td><td>90.0% 40.0%</td><td>67.2 61.7</td><td></td></tr><tr><td>Alchemy3D-Turbo</td><td>4.61</td><td>0.502</td><td>71.32</td><td>81.15</td><td>52.51</td><td>87.74</td><td>71.97</td><td>38.23</td><td>65.78</td><td>100.0%</td><td>65.1</td><td></td></tr><tr><td>Alchemy3D</td><td>4.64</td><td>0.552</td><td>72.53</td><td>82.40</td><td>54.41</td><td>89.89</td><td>73.34</td><td>38.16</td><td>67.47</td><td>90.0%</td><td>71.2 一 =</td><td></td></tr><tr><td colspan="10">action</td><td colspan="3"></td></tr><tr><td>3DEditFormer</td><td>4.06</td><td>0.509</td><td>70.32</td><td>75.80</td><td>44.45</td><td>83.56</td><td>65.69</td><td>32.86</td><td>54.34</td><td>20.0%</td><td>55.2</td><td></td></tr><tr><td>Alchemy3D-Turbo</td><td>4.33</td><td>0.513</td><td>73.58</td><td>79.96</td><td>51.46</td><td>86.85</td><td>73.76</td><td>36.85</td><td>63.89</td><td>90.0% 64.4</td><td>70.5</td><td>40.057.5 69.5</td></tr><tr><td>Alchemy3D</td><td>4.29</td><td>0.537</td><td>72.83</td><td>80.84</td><td>51.15</td><td>88.09</td><td>76.49</td><td>37.01</td><td>66.34</td><td>80.0% 70.8</td><td>70.5</td><td>75.6</td></tr></table>

Table 11: Image-conditioned comparison on 3DEditVerse. Shaded columns measure alignment with the provided reconstructed target and are not treated as primary metrics.
<table><tr><td></td><td colspan="3">View Quality</td><td colspan="4">Ref. Alignment</td><td colspan="6">Target Alignment</td></tr><tr><td>Method</td><td>Aes. ↑</td><td>MANIQA↑</td><td>MUSIQ ↑</td><td></td><td></td><td></td><td> $\overline { { { A _ { I } ^ { \mathrm { C L I P } } \uparrow \ A _ { I } ^ { \mathrm { S i g L I P } } \uparrow \ A _ { I } ^ { \mathrm { D I N O } } \uparrow \ A _ { I } ^ { \mathrm { U n i 3 D } } \uparrow } } }$ </td><td>LPIPS ↓</td><td>SSIM↑</td><td>PSNR ↑</td><td>CD↓</td><td>F1 ↑</td><td>NC↑</td></tr><tr><td>3DEditFormer</td><td>4.191</td><td>0.533</td><td>68.96</td><td>69.18</td><td>79.18</td><td>64.60</td><td>36.76</td><td>0.0797</td><td>0.9112</td><td>23.81</td><td>13.36</td><td>72.18</td><td>0.852</td></tr><tr><td>PartFlow</td><td>4.145</td><td>0.537</td><td>69.08</td><td>68.08</td><td>78.60</td><td>63.74</td><td>36.46</td><td>0.0786</td><td>0.9148</td><td>23.75</td><td>17.65</td><td>66.92</td><td>0.844</td></tr><tr><td>Alchemy3D-Turbo</td><td>4.239</td><td>0.524</td><td>71.43</td><td>72.10</td><td>82.05</td><td>69.44</td><td>36.36</td><td>0.1016</td><td>0.8967</td><td>20.65</td><td>22.71</td><td>50.28</td><td>0.773</td></tr><tr><td>Alchemy3D</td><td>4.213</td><td>0.558</td><td>70.78</td><td>73.10</td><td>82.18</td><td>72.20</td><td>37.41</td><td>0.1000</td><td>0.8961</td><td>22.06</td><td>24.48</td><td>44.15</td><td>0.757</td></tr></table>

Table 10 separates Eval3DEdit by editing type. On addition, Alchemy3D reaches 81.8% success, against 27.3% for Nano3D, 45.5% for 3DEditFormer, and 62.5% for PartFlow. On removal, Nano3D keeps a high CLIP score but only 20% success, so similarity to the source is not the same as completing the edit. Replacement is the closest comparison: 3DEditFormer and Alchemy3D-Turbo both reach 100% success, while Alchemy3D is slightly lower at 90% and remains stronger on image alignment. On style, PartFlow falls to 40% success, whereas Alchemy3D-Turbo reaches 100%. On action, 3DEditFormer succeeds on 20% of cases, against 90% for Alchemy3D-Turbo and 80% fo Alchemy3D.

We next evaluate on 3DEditVerse (Xia et al., 2025), the Edit3D-Bench of VoxHammer (Li et al., 2026b), and Nano3D-100K (Ye et al., 2025a). Nano3D does not release an official test list, so we randomly sample 1.5K pairs. View quality and reference alignment follow Sec. 5 and do not use a reconstructed 3D target. For completeness, we also report target alignment: perceptual similarity to rendered views of the provided target asset through LPIPS (Zhang et al., 2018), SSIM (Wang et al., 2004), and PSNR, and geometric similarity to its mesh through Chamfer distance (Fan et al., 2017), F-score (Knapitsch et al., 2017), and normal consistency (Gkioxari et al., 2019). These columns are shaded. As discussed in Sec. 5, a reconstructed asset is not an independent reference, and similarity to one such asset ignores the one-to-many nature of editing. Alchemy3D performs better on the unshaded metrics but worse on the shaded metrics, consistent with prior methods being trained or selected against the same reconstructed targets.

Table 12: Image-conditioned comparison on Edit3D-Bench (VoxHammer). Target-alignment metrics are not available for this benchmark.
<table><tr><td></td><td colspan="3">View Quality</td><td colspan="4">Ref. Alignment</td></tr><tr><td>Method</td><td>Aes. ↑</td><td>MANIQA↑</td><td>MUSIQ ↑</td><td> $A _ { I } ^ { \mathrm { C L I P } }$  ←</td><td> $A _ { I } ^ { \mathrm { S i g L I P } } \uparrow$ </td><td> $A _ { I } ^ { \mathrm { D I N O } }$  ←</td><td> $A _ { I } ^ { \mathrm { U n i 3 D } } \uparrow$ </td></tr><tr><td>3DEditFormer</td><td>4.099</td><td>0.553</td><td>73.72</td><td>76.91</td><td>84.99</td><td>66.90</td><td>38.33</td></tr><tr><td>PartFlow</td><td>4.112</td><td>0.568</td><td>74.25</td><td>77.55</td><td>85.95</td><td>69.02</td><td>38.93</td></tr><tr><td>Alchemy3D-Turbo</td><td>4.124</td><td>0.553</td><td>73.48</td><td>79.90</td><td>87.28</td><td>72.91</td><td>38.25</td></tr><tr><td>Alchemy3D</td><td>4.207</td><td>0.580</td><td>74.61</td><td>80.91</td><td>88.44</td><td>75.04</td><td>39.06</td></tr></table>

Table 13: Image-conditioned comparison on Nano3D-100K. Shaded columns measure alignment with the provided reconstructed target and are not treated as primary metrics.
<table><tr><td></td><td colspan="3">View Quality</td><td colspan="4">Ref. Alignment</td><td colspan="6">Target Alignment</td></tr><tr><td>Method</td><td>Aes. ↑</td><td>MANIQA↑</td><td>MUSIQ ↑</td><td> $A _ { I } ^ { \mathrm { C L I P } } \cdot$  1</td><td> $A _ { I } ^ { \mathrm { S i g L I P } } ~ .$  ←</td><td> $A _ { I } ^ { \mathrm { D I N O } } \uparrow$ </td><td> $A _ { I } ^ { \mathrm { U n i 3 D } } \uparrow$ </td><td>LPIPS ↓</td><td>SSIM↑</td><td>PSNR ↑</td><td>CD↓</td><td>F1 ↑</td><td>NC↑</td></tr><tr><td>3DEditFormer</td><td>4.228</td><td>0.500</td><td>71.27</td><td>66.30</td><td>78.06</td><td>60.38</td><td>36.57</td><td>0.1084</td><td>0.8740</td><td>19.92</td><td>15.21</td><td>69.88</td><td>0.814</td></tr><tr><td>PartFlow</td><td>4.192</td><td>0.509</td><td>71.93</td><td>65.42</td><td>78.61</td><td>61.18</td><td>36.20</td><td>0.0970</td><td>0.8887</td><td>20.79</td><td>17.58</td><td>69.43</td><td>0.823</td></tr><tr><td>Alchemy3D-Turbo</td><td>4.266</td><td>0.487</td><td>71.24</td><td>69.86</td><td>80.81</td><td>66.58</td><td>37.04</td><td>0.1296</td><td>0.8603</td><td>18.79</td><td>24.99</td><td>49.81</td><td>0.726</td></tr><tr><td>Alchemy3D</td><td>4.304</td><td>0.539</td><td>73.27</td><td>72.22</td><td>82.37</td><td>71.11</td><td>38.06</td><td>0.1327</td><td>0.8572</td><td>18.40</td><td>29.12</td><td>42.00</td><td>0.699</td></tr></table>

Across Tables 11, 12, and 13, Alchemy3D ranks first and Alchemy3D-Turbo usually second on the unshaded metrics.

Instruction-conditioned editing. Table 14 separates direct instruction editing from the agentic pipeline in Sec. A.4.1. Alchemy3D-Instruct outperforms Steer3D on every editing type that Steer3D supports. The margin is largest on addition, where Steer3D has zero success, and on identity preservation, where Steer3D remains near 30–50 while Alchemy3D-Instruct remains above 65. The agentic variant edits one selected view and then applies image-conditioned Alchemy3D, which further raises success: 75.0% versus 57.5% on addition, 87.1% versus 78.1% on replacement, 52.5% versus 30.0% on local appearance, 50.0% versus 6.7% on global appearance, and 70.0% versus 46.7% on animation. Global appearance remains the weakest setting for the instruction model, consistent with noise in the recaptioned instructions and with the task conflict in Sec. A.6. Identity preservation is sometimes higher for Alchemy3D-Instruct than for the agentic pipeline, because the intermediate 2D edit changes the asset more strongly.

Multi-turn Editing. Fig. 10 illustrates multi-turn, long-horizon editing with Alchemy3D. A language model (Qwen3.8-27B) decomposes a complex instruction into short atomic edits. The agentic pipeline in Sec. A.4, which selects a view, edits that image, and then edits the 3D asset, is applied recursively to each atomic instruction.

## A.6 LIMITATIONS AND FUTURE WORK

Imperfect verification. Although Alchemy3D-1M (Sec. 3) is a million-scale 3D asset editing dataset, and training Alchemy3D on it supports its utility, several limitations remain. As with prior datasets in image, video, and 3D asset editing (Yu et al., 2025; Liang et al., 2026; Ye et al., 2025a; Xia et al., 2025; Weng et al., 2026) that verify and caption data automatically with vision-language models, our dataset may contain low-quality samples because current VLMs are imperfect. For example, they can struggle to distinguish the left and right sides of an asset, and may accept lowquality samples as valid training pairs.

Task conflict. In this work, we follow TRELLIS.2 (Xiang et al., 2026) in adopting a dense crossattention transformer architecture. However, during training, we found that the goals and granularities of different editing types are not always mutually beneficial when trained together. For example, local and global appearance changes are learned poorly when trained jointly with a large number of geometry-changing edits. A mixture-of-experts model (Jacobs et al., 1991) may separate editing types that operate at different granularities.

Table 14: Instruction-driven comparison on GEdit3D-Bench. Steer3D does not support replacement or animation edits. <sup>†</sup> Alchemy3D is evaluated with the agentic view selection and image editing described in Sec. A.4.1.
<table><tr><td rowspan="2">Method</td><td colspan="3">View Quality</td><td colspan="2">Ref. Alignment</td><td colspan="3">MLLM</td></tr><tr><td></td><td>Aes. ↑ MANIQA ↑ MUSIQ ↑</td><td></td><td> $A _ { T } ^ { \mathrm { C L I P } } \uparrow$ </td><td> $A _ { I } ^ { \mathrm { U n i 3 D . } }$ </td><td>SR↑</td><td>VQ↑</td><td>IF↑ IP↑</td></tr><tr><td colspan="9">Add</td></tr><tr><td>Steer3D</td><td>3.810</td><td>0.522</td><td>69.39</td><td>46.47</td><td>63.38</td><td>0.0</td><td>47.9</td><td>25.7 30.2</td></tr><tr><td>Alchemy3D-Instruct</td><td>4.602</td><td>0.563</td><td>74.43</td><td>68.34</td><td>80.69</td><td>57.5</td><td>69.6 57.9</td><td>75.5</td></tr><tr><td>Alchemy3D†</td><td>4.604</td><td>0.566</td><td>74.48</td><td>68.33</td><td>78.76</td><td>75.0</td><td>72.4 65.6</td><td>69.3</td></tr><tr><td colspan="9">Remove</td></tr><tr><td>Steer3D</td><td>3.827</td><td>0.523</td><td>69.18</td><td>46.12</td><td>61.84</td><td>8.8</td><td>53.0</td><td>41.7 32.9</td></tr><tr><td>Alchemy3D-Instruct</td><td>4.537</td><td>0.549</td><td>72.94</td><td>65.15</td><td>78.79</td><td>75.0</td><td>66.8 64.9</td><td>68.3</td></tr><tr><td>Alchemy3D†</td><td>4.590</td><td>0.557</td><td>73.11</td><td>66.34</td><td>80.69</td><td>72.5</td><td>70.9</td><td>65.9 69.8</td></tr><tr><td colspan="9">Replace</td></tr><tr><td>Alchemy3D-Instruct 4.443</td><td></td><td>0.544</td><td>74.04</td><td>65.99</td><td>79.28</td><td>78.1</td><td>67.5</td><td>67.1 66.2</td></tr><tr><td>Alchemy3D†</td><td>4.564</td><td>0.564</td><td>73.98</td><td>67.78</td><td>81.85</td><td>87.1</td><td>70.8</td><td>71.2 68.8</td></tr><tr><td colspan="9">Local Appearance</td></tr><tr><td>Steer3D</td><td>3.840</td><td>0.528</td><td>69.54</td><td>50.79</td><td>69.64</td><td>23.8</td><td>55.1 43.1</td><td>32.3</td></tr><tr><td>Alchemy3D-Instruct</td><td>4.639</td><td>0.558</td><td>73.51</td><td>65.10</td><td>79.75</td><td>30.0</td><td>71.0 42.0</td><td>72.0</td></tr><tr><td>Alchemy3D†</td><td>4.604</td><td>0.563</td><td>73.52</td><td>65.72</td><td>80.03</td><td>52.5</td><td>71.2 53.2</td><td>68.6</td></tr><tr><td colspan="9">Global Appearance</td></tr><tr><td>Steer3D</td><td>3.882</td><td>0.523</td><td>68.74</td><td>49.79</td><td>68.78</td><td>18.3</td><td>53.3</td><td>45.3 49.1</td></tr><tr><td>Alchemy3D-Instruct</td><td>4.605</td><td>0.538</td><td>72.88</td><td>57.00</td><td>71.86</td><td>6.7</td><td>67.6</td><td>33.4 77.6</td></tr><tr><td>Alchemy3D†</td><td>4.580</td><td>0.540</td><td>72.40</td><td>62.23</td><td>75.22</td><td>50.0</td><td>71.2 51.8</td><td>75.5</td></tr><tr><td colspan="9">Animation</td></tr><tr><td>Alchemy3D-Instruct</td><td>4.431</td><td>0.549</td><td>73.43</td><td>69.20</td><td>79.33</td><td>46.7</td><td>64.4 56.2</td><td>65.7</td></tr><tr><td>Alchemy3D†</td><td>4.487</td><td>0.551</td><td>73.47</td><td>69.94</td><td>81.18</td><td>70.0</td><td>67.7 62.7</td><td>70.6</td></tr></table>

![](images/375700af9f7fcb958b2bfa96b2eacb99f5964e7490b289472b33860d4fc40640.jpg)  
Figure 10: Examples of multi-turn, long-horizon 3D asset editing.

![](images/fc7b4868e00db3872399f0dc970b008a6759b04d43f49a3237763aaadb14f79f.jpg)  
Figure 11: Examples from the Edit3D-Bench of Steer3D.

Source Asset

Source Image

Edit Instruction

Add a miniature saddle of fused, jagged swords to its back.

![](images/18486080cac5fe02e8ad79c3c99144151ae630fa39c920239085b2e8d670dd02.jpg)  
Add a tall, pointed hood with sharp, ribbed cathedral arches to the head.

Add a sharp, faceted polyhedral scarab beetle at the base.

Add a jagged wooden mallet tremolo arm wrapped in thick, physical bandage layers.

Add a   
mechanical tail with sprouting, multi-layered   
geometric   
flower-shaped protrusions.   
Replace the   
body with a   
layered,   
spiraling   
nautilus shell   
geometry.

Target Asset

Replace the body with a layered, spiraling nautilus shell geometry.

Add a heavy chiseled jade pedestal base with deep  
carved   
geometric   
basins.

Target Image

![](images/0560106deb36e3bd5243524180de390c3d96f578dc0f7ed06b8f99eb2770d690.jpg)  
Figure 12: Examples from Nano3D-100K. Nano3D did not release its test set, so this figure uses the 1.5K-pair subset described in Sec. A.5.1.

![](images/46e27f94a0720982d068fab1cc5c1e68c420e14b573a525923d06f594e7d9d8b.jpg)  
Figure 13: Examples from the 3DEditVerse test set.

Source

Edited Image

Nano3D

![](images/9a3ef8994124c875c03ce494f26dfb1aa28ef34898b4409c9dec2b9629ecccb4.jpg)  
Figure 14: Qualitative comparison on addition, removal, and replacement edits.

Source

Edited Image

3DEditFormer

Alchemy3D

![](images/4d1c01f1cdf8ef27a07dcc21c3edd9ca4f04a038840785f6c2569b7f6dcbbff2.jpg)  
Figure 15: Qualitative comparison on animation edits.

Source

Edited Image

PartFlow

Alchemy3D

![](images/eaea4b4c23e8e0b8d122b2b1891bf479edfaf0b37a8b0dd90f50a83888616c4f.jpg)  
Figure 16: Qualitative comparison on appearance edits.
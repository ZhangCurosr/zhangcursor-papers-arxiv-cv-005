# mniDex: Scaling Dexterous Hand Grasping to Diverse Cluttered Scenes

Naiyu Fang<sup>1,2,3∗</sup> Zhongjin Luo<sup>1,2,3∗</sup> <sup>†</sup> Yuxin Mo<sup>1,4</sup> Siyuan Huang<sup>1</sup> <sup>†</sup> Jianbo Liu<sup>1</sup> Yufei Liu<sup>1,5</sup> Zheyuan Zhou<sup>6</sup> Chenkai Jin<sup>1,7</sup> Xiaogang Wang<sup>1,2</sup> Hongsheng Li<sup>2</sup> <sup>B</sup>

<sup>1</sup>ACE Robotics <sup>2</sup>CUHK, MMLab <sup>3</sup>CPII under InnoHK <sup>4</sup>Tongji University

<sup>5</sup>Shanghai Jiao Tong University <sup>6</sup>Zhejiang University <sup>7</sup>Nanyang Technological University

<sup>∗</sup>Equal contribution <sup>†</sup>Project lead <sup>B</sup>Corresponding author

Dexterous grasping is the foundational primitive in embodied AI, demanding massive data to train robust models. As real-world data collection is expensive, simulation has become the mainstream paradigm. Yet, while cluttered scenes best reflect real-world applications, learning to grasp within them is bottlenecked by a critical scarcity of large-scale data. To resolve this, we curate high-quality 3D objects and supporting bases, proposing a scalable seed-and-filter strategy that bypasses sluggish scene-level optimization. This yields an unprecedented benchmark comprising over 2.6 million scenes and 0.4B scene-specific grasp ground truths, featuring diverse realistic layouts paired with rich semantic and geometric observations. Furthermore, we introduce the OmniDex model to overcome the grasp multimodality and last-millimeter precision errors plaguing current generative models. By coupling Soft Winner-Takes-All learning with human-inspired physical constraints during training, and utilizing physics-driven ranking, our approach achieves robust dexterous grasping without the latency of post-optimization. Experimental results show that OmniDex model achieves state-of-the-art performance and strong generalization across diverse scenes, views, and unseen objects.

Website: https://aceroboticsdex.github.io/OmniDex/ Date: September 2026

![](images/ed43e29576e90b577566e491601098c476d4f433b53117de604fee7ffc93afa8.jpg)  
Figure 1 Overview of OmniDex, the largest benchmark for dexterous grasping in cluttered scenes to date. Left: Multi-view RGB-D and instruction masks paired with scene-specific grasp poses. Right: A scalable seed-and-filter strategy enables unprecedented scale (2.6M scenes, 0.4B grasp ground truths), encompassing diverse objects, supporting bases, and realistic layouts.

## 1 Introduction

Dexterous hands mark a critical milestone in embodied artificial intelligence (AI) [1]. Unlike parallel-jaw grippers [2, 3], they offer the kinematic flexibility and fine-grained manipulation essential for complex, human-centric tasks. At the core of this capability lies grasping—the foundational primitive upon which all advanced interactions are built. For an intelligent agent, predicting a valid grasp pose acts as the "key frame" that dictates the success of all subsequent tasks. However, training grasp pose generation models is severely bottlenecked by a profound scarcity of high-quality data. Since collecting high degree-of-freedom (DoF) grasp poses in the real world is expensive and unscalable [4], simulation-based data synthesis [5, 6] has emerged as the mainstream paradigm. To drive dexterous grasping task to ward greater robustness and real-world applicability, this paper introduces a large-scale simulation benchmark targeting cluttered scenes, coupled with a novel learning framework.

Real-world dexterous grasping is inherently messy and cluttered. Yet, the standard setting in prior dexterous grasping literature oversimplifies the problem by isolating a single object in floating space or on a clear tabletop [7, 8]. Recent efforts tackle cluttered scenes [9, 10]; however, robust performance in complex real-world settings remains an open problem due to three underlying challenges. First, current approaches rely on depth-based pseudo point clouds as their primary perceptual input. This forces models to focus on geometric shapes at the expense of semantic awareness. Consequently, agents struggle with target object localization in cluttered scenes (e.g., distinguishing geometrically identical Coke and Sprite cans), since their geometries are nearly identical. Second, existing benchmark scales are critically constrained by computationally expensive data generation pipelines. Scene-level optimization for grasp ground truths [9] incurs prohibitive time costs, limiting existing cluttered-scene benchmarks to only 0.4K–8K scenes (e.g., DexGrasp Net 2.0: 8K; Tab. 1). Yet, massive scale is indispensable. Unlike single-object floating setups, cluttered scenes feature complex spatial dynamics among object. Thus, models require a large-scale dataset to effectively learn these spatial relationships and master scene-specific grasping, such as avoiding adjacent items or table penetration. Finally, prior works neglect diverse supporting bases and spatial layouts, compromising robustness and real-world generalization.

Beyond the data bottleneck, existing dexterous grasping generation models face critical algorithmic hurdles when mapping perceptual inputs to high-DoF kinematic space. Dexterous grasping is a multimodal problem, as an object affords numerous viable grasp poses. To capture this complex distribution, data-driven approaches rely heavily on intermediate representations, such as affordance maps [9, 11], rendering the generation pipeline inherently non-end-toend. However, pure data-driven methods struggle with critical last-millimeter precision errors. A mere millimeter of deviation translates directly to catastrophic contact failures in simulation. To compensate for this inherent inaccuracy, current methods resort to costly post-optimization or test-time fine-tuning [12, 10], severely compromising inference efficiency.

To address these data and algorithmic bottlenecks, we introduce a holistic paradigm comprising a large-scale cluttered scene benchmark and a novel end-to-end generation framework. To bypass the prohibitive computational cost of scene-level optimization, our OmniDex benchmark employs a scalable seed-and-filter strategy. We first establish a reusable library of free-space seed grasps for single objects, then filter them through collision constraints in cluttered scenes within gravity-stabilized simulations. Leveraging high-quality 3D object assets and high-fidelity rendering, this pipeline yields multi-modal RGB-D observations from multiple fixed camera views, aligned with scene-aware grasp ground truths. Ultimately, our OmniDex benchmark delivers unprecedented scale: over 2.6 million scenes and 0.4B scene-specific grasp pose labels, covering diverse objects, supporting bases, and realistic cluttered layouts. We hope this benchmark will spur a new wave of research in cluttered scenes dexterous grasping. Empowered by this benchmark, OmniDex model fundamentally avoids the pitfalls of non-end-to-end intermediate representations and sluggish test-time fine-tuning. Instead, we internalize physical dynamics directly into training. By coupling a Soft Winner-Takes-All learning to capture grasp multimodality with a human-inspired physics-constrained loss—which enforces direct hand-object interaction—we effectively bridge the critical last-millimeter precision gap to ensure close and stable grasping. During inference, the optimal grasp pose is rapidly identified via a physical ranking mechanism, achieving both exceptional fidelity and real-time efficiency.

• We introduce a massive cluttered-scene OmniDex benchmark via a scalable seed-and-filter paradigm, yielding an unprecedented dataset of over 2.6 million realistic scenes and 0.4B scene-specific validated grasp ground truths, featuring diverse supporting bases and rich multi-modal observations.

• We propose OmniDex model, a novel end-to-end grasp generation framework that couples Soft Winner-Takes-All learning with human-inspired physical constraints to capture grasp multimodality and bridge the last-millimeter preci-• Experimental results show that OmniDex achieves state-of-the-art performance on the OmniDex benchmark. It generalizes well across scene subsets, supporting bases, input views, and unseen objects, demonstrating robustness in realistic cluttered scenes.

Table 1 Quantitative comparison of simulation-based dexterous grasping benchmarks. Our dataset features the large scale of scenes and objects, emphasizing diverse layouts in clutter.
<table><tr><td>Method</td><td># Scenes</td><td>Setting</td><td># Objects</td><td>Workspace</td><td>Modality</td></tr><tr><td>Dex1B [5]</td><td>一</td><td>Single</td><td>6K</td><td></td><td>Depth</td></tr><tr><td>AffordDexGrasp [12]</td><td>2K</td><td>Single</td><td>1K</td><td>Table</td><td>RGB-D</td></tr><tr><td>DDGC [13]</td><td>0.4K</td><td>Cluttered</td><td>0.3K</td><td>Table</td><td>RGB-D</td></tr><tr><td>DexGraspNet 2.0 [9]</td><td>8K</td><td>Cluttered</td><td>1K</td><td>Table</td><td>Depth</td></tr><tr><td>ClutterDexGrasp [14]</td><td>1K</td><td>Cluttered</td><td>2K</td><td>Table</td><td>Depth</td></tr><tr><td>Ours</td><td>2.6M</td><td>Cluttered</td><td>6K</td><td>Table, Box, Shelf</td><td>RGB-D</td></tr></table>

sion gap, with physics-driven ranking for robust inference.

## 2 Related Works

## 2.1 Dexterous Grasping Datasets and Benchmarks

Real-world dexterous grasping datasets largely rely on human teleoperation [4, 15, 16, 17]. While capturing humanlike priors, this labor-intensive paradigm limits dataset scale and object diversity. Consequently, the community shifted to simulation-based synthesis for isolated objects. Pioneering works [7, 18, 6, 19] leveraged optimization and physical priors to generate massive single-object grasps, culminating in billion-scale datasets like Dex1B [5]. Despite this impressive scale, these single-object benchmarks fail to model the spatial dynamics of real-world clutter. While re cent efforts have introduced cluttered-scene benchmarks [9, 14] to bridge this reality gap, they remain bottlenecked by costly per-scene optimization. In stark contrast, our efficient seed-and-filter strategy entirely bypasses this sluggish optimization, yielding an unprecedented 2.6 million complex scenes and 0.4B physically validated ground truth, thereby establishing a new benchmark for diversity and scale in cluttered grasping.

## 2.2 Dexterous Grasp Generation Methods

Learning-based generative models promise real-time dexterous grasp generation, yet mapping visual inputs to high-DoF kinematics is fundamentally plagued by grasp multimodality. To capture this one-to-many manifold, data-driven methods typically rely on heuristic proxies—such as contact maps [8, 10] or affordance fields [9, 20], making the pipeline non-end-to-end. Furthermore, these models struggle with the last-millimeter precision, where minute devi ations cause catastrophic contact failures. To mitigate this, recent frameworks [12, 11] use computationally heavy test-time optimization [21], compromising real-time efficiency. While some attempts internalize physical priors during training to bypass this latency [18], their geometric penalties remain insufficient for dense clutters. In stark contrast, our framework resolves these bottlenecks: we capture the multimodal distribution via a Soft Winner-Takes-All objective, and directly bridge the precision gap using a human-inspired physical constraint during training, coupled with a physical ranking during inference.

## 3 Benchmark

## 3.1 Overview of Data Generation

Our data generation pipeline targets realistic cluttered scenes while producing two tightly coupled resources: (i) state observation representations; (ii) instruction-conditioned dexterous hand grasp pose labels. A central difficulty arises from the high DoF of dexterous hands: we must generate grasps that are not only physically feasible but also executable in cluttered scenes. Optimization-based methods have proven effective for dexterous hand grasp generation for a single object and can be applied in tabletop settings. However, the tabletop setting is inherently instance-specific, as each run is anchored to a particular object–table configuration. Moreover, its computational cost is prohibitive (e.g., optimizing

![](images/4d88ad5297dcf8958b68fdf419c420b2ea78615ab6fd94672df84f8afec52795.jpg)  
Figure 2 Seed-and-filter generation pipeline. Our two-stage framework generates scene-valid grasps by: (1) constructing a reusable seed grasp library in free space, and (2) simulating cluttered scenes in Isaac Gym for collision filtering, followed by multi-modal rendering in Blender.

500 initializations for one object takes ∼ 15 minutes on an A800), making per-object, per-scene optimization infeasible for large-scale cluttered scene benchmark construction.

As shown in Fig. 2, to enable scalable generation, we adopt a two-stage seed-and-filter strategy. First, we run optimization for each object in free space to produce a reusable library of seed grasp poses sampled from diverse directions. Next, we simulate cluttered scenes in Isaac Gym by dropping objects onto supporting bases. Once dynamic stability is achieved, we transform free-space seed grasp poses into scene-specific candidates and retain those that satisfy a conser vative direct-retrieval criterion. Specifically, accepted candidates must establish target contact while avoiding support surfaces and clearly disruptive interactions with neighboring objects. This protocol is designed to isolate direct target extraction in cluttered scene, rather than rearrangement-mediated retrieval. To simulate the visual scene observations available to learning agents, we replay the cluttered scene geometry in Blender to build a visual database, including RGB images, depth maps, semantic segmentation, and camera parameters. The segmentation taxonomy is defined within each scene and used to generate target-object masks as visual instructions for grasping. In summary, the visual database provides diverse model inputs, while the geometry database provides scene-valid grasp ground truths for both model supervision and evaluation. Qualitative visualizations are in Appendix F.1.

## 3.2 3D Assets

Grasped Objects. We curate graspable object assets from OmniObject3D [22], ShapeNetCore v2 [23], and Google Scanned Objects [24] datasets. The composition of the curated assets, including instance counts and category-wise distributions, is detailed in Fig. 3 (a). Our selection follows a staged protocol: we first use their semantic annotations to exclude categories incompatible with dexterous hand grasping, then enforce inter-class balancing to avoid long tailed bias, and finally remove instances with defective meshes or incomplete textures. Each asset is processed into two scale-aligned representations: a physics-ready model generated via manifold–normalization–decomposition [7] and a high-fidelity visual model, both sharing an identical normalization to ensure spatial consistency. In total, we provide a database of over 6,000 high-quality objects, spanning diverse categories such as cellphones, toys, and shoes. All selected instances are manually inspected to ensure they are suitable for the dexterous grasping task.

Supporting Base. As illustrated in Fig. 3 (b), our OmniDex benchmark incorporates tables, open-top boxes, and shelves to diversify object placement in cluttered scenes. Specifically, our asset library comprises 244 tables from ABO [25], 79 open-top boxes sourced from both ABO and Isaac Lab [26], and 21 shelves from ShapeNetCore v2 [23].

## 3.3 Scene-Specific Ground-Truth Grasp Pose Generation

Following the seed-and-filter strategy in Sec. 3.1, we first optimize 4,000 candidates per object using the force-closure estimator [7], retaining those that pass a simulated lift test as seed grasp poses. To construct cluttered scenes, a random number of N objects are dropped from stochastic heights, spatially confined within the horizontal bounds of the supporting base. We adopt subset-specific scene generation rules to induce distinct clutter regimes: random tabletop clutter for $\mathcal { D } _ { \mathrm { s t d } }$ and $\mathcal { D } _ { \operatorname* { m i x } }$ , dense confined stacking for $\mathcal { D } _ { \mathrm { { b o x } } } .$ , spaced identical-object layouts for ${ \mathcal { D } } _ { \mathrm { g r i d } } .$ , and layer-wise arrangements for $\mathcal { D } _ { \mathrm { s h e l f } }$ . Please see Appendix A.2 for more details.

Simulation terminates once all objects reach stable resting conditions or a fixed rollout cutoff is reached. The resulting settled scene yields stable object poses $\mathbf { o } = ( \mathbf { t } _ { \mathrm { o b j } } , \mathbf { R } _ { \mathrm { o b j } } )$ , where $\mathbf { t } _ { \mathrm { o b j } } \in \mathbb { R } ^ { 3 }$ and $\mathbf { R } _ { \mathrm { o b j } } \in S O \left( 3 \right)$ denote the object translation and orientation, respectively. We then apply the object transformations to the selected seed grasps to derive scene-specific poses $\mathbf { g } = ( \mathbf { t } _ { \mathrm { d e x } } , \mathbf { R } _ { \mathrm { d e x } } , \theta )$ . Here, $\mathbf { t } _ { \mathrm { d e x } } \in \mathbb { R } ^ { 3 }$ and $\mathbf { R } _ { \mathrm { d e x } } \in S O \left( 3 \right)$ denote the wrist translation and orien tation, respectively, while θ represents joint angles. The complete scene configuration is thus denoted as $\mathbf { s } = ( \mathbf { o } , \mathbf { g } )$ To mitigate artifacts from physical-visual mesh discrepancies and incomplete simulation convergence, we perform a sanity check by replaying each scene with visual models. Please see Appendix A.1 for more details.

![](images/5853ae3545ddde6ab50ffdf516fb4cf3191a09b72f82d8d24f519c77082e1874.jpg)  
Figure 3 Dataset statistics and visualization of 3D assets. (a): Instance counts and category-wise distributions of the curated graspable objects, highlighting the top 50 semantic classes. (b): Visual examples of the assets, which range from diverse grasped objects (medicine bottles, boxes, apples) to realistic supporting bases (tables, open-top boxes, shelves).

Table 2 Benchmark diversity and evaluation protocol across OmniDex subsets. Each subset is organized with a large training split and two decoupled test tracks to evaluate generalization.
<table><tr><td rowspan="2">Subset</td><td rowspan="2">Obj. Div.</td><td rowspan="2">Scene Obj. Comp.</td><td rowspan="2">Sup. Base Type</td><td rowspan="2">Sup. Base Div.</td><td rowspan="2">Density</td><td rowspan="2">Scene Num.</td><td colspan="3">Generalization Split</td></tr><tr><td>Train</td><td>Seen</td><td>Unseen</td></tr><tr><td> $2 \mathcal { D } _ { \mathrm { s t d } }$ </td><td>√</td><td>Multi-category</td><td>Table</td><td>x</td><td>0.1 / 0.2 / 0.3</td><td>1.14M</td><td>919K</td><td>62K</td><td>163K</td></tr><tr><td> $\mathcal { D } _ { \mathrm { m i x } }$ </td><td>√</td><td>Multi-category</td><td>Table</td><td>√</td><td>0.1 / 0.2 / 0.3</td><td>0.94M</td><td>706K</td><td>80K</td><td>152K</td></tr><tr><td> $\mathcal { D } _ { \mathrm { b o x } }$ </td><td>√</td><td>Multi-category</td><td>Box</td><td>√</td><td>0.9</td><td>0.14M</td><td>112K</td><td>12K</td><td>16K</td></tr><tr><td> $\mathcal { D } _ { \mathrm { g r i d } }$ </td><td>√</td><td>Single-category</td><td>Table</td><td>x</td><td></td><td>0.28M</td><td>246K</td><td>11K</td><td>24K</td></tr><tr><td> $\mathcal { D } _ { \mathrm { s h e l f } }$ </td><td>√</td><td>Multi-category</td><td>Shelf</td><td>√</td><td></td><td>0.14M</td><td>107K</td><td>11K</td><td>28K</td></tr></table>

## 3.4 Diversity and Protocol

Modal Diversity. As discussed in Sec. 1, geometry-only input is inadequate for instruction-guided grasping in cluttered scenes. We therefore provide aligned RGB, depth, instruction masks, and camera parameters under a subset-aware multi-view protocol. We use one ego-centric view and four auxiliary views by default, while adapting the camera layout to each subset: tabletop scenes use standard corner views, box scenes tighten the rig to counter boundary occlusion, and shelf scenes adopt layer-aligned frontal views to maintain visibility under severe vertical blockage. This makes perception a first-class benchmark component. Please see Appendix A.2 for details.

Task Diversity. To align with the core objective of dexterous grasping, our OmniDex benchmark introduces diversity across: grasped object and supporting base. These factors directly dictate grasp feasibility and perceptual difficulty. As illustrated in Table 2, following prior works [5, 11], we provide $\mathcal { D } _ { \mathrm { s t d } }$ with 1.14M scenes where the supporting base remains fixed, varying only the target object. This design isolates the evaluation of policy transferability across diverse geometries and appearances. To evaluate grasping in more unstructured environments, we include $\mathcal { D } _ { \mathrm { m i x } }$ with 0.94M scenes where both the target objects and the supporting bases vary simultaneously.

Real Scene Diversity. To approximate real-scene applications, we introduce three specialized $\mathcal { D } _ { \mathrm { b o x } } , \mathcal { D } _ { \mathrm { g r i d } }$ , and $\mathcal { D } _ { \mathrm { s h e l f } }$ As illustrated in Fig. 1, $\mathcal { D } _ { \mathrm { b o x } }$ models parcel sorting, where objects are densely packed both vertically and horizon tally within confined containers, mimicking carton or express package handling. $\mathcal { D } _ { \mathrm { g r i d } }$ simulates warehouse sorting environments featuring multiple identical objects uniformly distributed on a table. To ensure stable post-drop config urations in this subset, we restrict our asset pool to objects capable of remaining upright after falling (e.g., bottles). Finally, $\mathcal { D } _ { \mathrm { s h e l f } }$ targets SKU-level retail scenarios where products are systematically arranged on shelves; for this, we specifically curate retail-style items (e.g., noodle cups) from complete object dataset.

![](images/d38ae06bcbefb668813e22920d3d097dcd723c280d71a7671cda46b7be091073.jpg)  
Figure 4 Architecture of the OmniDex Model. Condition embeddings are transformed from RGB-D inputs, instruction masks, and scene parameters. During training, a denoising network adds identical noise to M grasp poses, optimized jointly via a Soft Winner-Takes-All loss (for multimodal distributions) and a physics-aware loss (for stable contact). During inference, the network generates M candidates, followed by a physics-based ranking module to select the optimal grasp.

Benchmark Protocol. As summarized in Table 2, each subset is partitioned into a large training split and two test tracks: Seen Objects in Unseen Layouts and Unseen Objects in Unseen Layouts. The former isolates spatial reasoning under novel clutter configurations while controlling for object familiarity, whereas the latter probes zero-shot transfer to unseen geometries under equally novel layouts. This protocol cleanly separates layout novelty from object novelty, yielding a benchmark that measures robust scene-level generalization in cluttered dexterous grasping.

## 3.5 Metrics

Aligned with our OmniDex benchmark protocol, we evaluate model performance along two dimensions: physical grasp stability and protocol-consistent scene interaction under direct retrieval. Following [7], a grasp is deemed physically valid if the target object securely withstands applied gravity with the supporting surface removed. We quantify this via three Success Rate (SR) variants based on the tested gravitational directions: the actual scene gravity $( \mathrm { S R } _ { \mathrm { s c e n e } } ,$ , our primary metric), all six canonical directions $( \mathrm { S R _ { a n y } } )$ , or at least one canonical direction $\mathrm { ( S R _ { o n e } ) }$ . To measure consistency with our selection protocol, we further report the Collision-Free Rate (CFR), defined as the percentage of generated grasps that avoid support-surface collisions and non-target contacts under the simulator.

## 4 Methodology

## 4.1 Overview

For clarity, we present the ego-centric single-view formulation here. We aim to generate a valid grasp pose $\hat { \bf g }$ for a target object in a cluttered scene, given an RGB image $\mathbf { I } \in \mathbb { R } ^ { H \times W \times 3 }$ , a depth map $\mathbf { \bar { D } } \in \mathbb { R } ^ { H \times W \times 1 }$ , camera parameters {K, T}, the table height H, and an instruction mask $\mathbf { M } \in \left\{ 0 , 1 \right\} ^ { H \times W }$ . Here, the instruction mask explicitly specifies the target object within the clutter. K and T denote the intrinsic and extrinsic camera matrices, respectively, while H and W denote the image height and width. Our method is formulated as training a diffusion-based neural network to generate a high-DoF dexterous grasp pose conditioned on scene appearance, geometry, and target identity: $\hat { \bf g } =$ Θ (I, D, M, K, T, H). The multi-view extension is provided in Appendix C.2.

The overview of the OmniDex model is illustrated in Fig. 4. In the feature encoder module, we extract features from the camera image, depth map, and instruction mask, and aggregate them into a unified perception feature for the cluttered scene. It is subsequently combined with the scene parameter features to yield the condition embedding (Sec. 4.2). In the training phase, we apply a Soft Winner-Takes-All (WTA) loss to address the inherently multimodal distribution of grasp poses, alongside a novel physics-constrained loss designed to enforce close and stable picking (Sec. 4.3). In the inference phase, the model generates multiple grasp pose candidates, from which the optimal grasp is identified via a physical ranking and selection (Sec. 4.4).

## 4.2 Multimodal Scene-Aware Condition Embedding

To generate accurate and stable dexterous grasps in cluttered scenes, the model requires a comprehensive understanding of both the semantic and geometric structure of the workspace. Under the ego-centric single-view setting described above, we construct a multimodal, scene-aware condition embedding by fusing the RGB image, depth map, instruction mask, and scene parameters into a unified tokenized representation.

Given the ego-centric RGB image, we use a pre-trained DINOv2 [27] backbone to extract patch-level semantic features $\mathbf { F } _ { \mathrm { i m g } } \in \mathbb { R } ^ { N \times d }$ , where N is the number of patches and d is the embedding dimension. We derive geometric features $\breve { \mathbf { F } } _ { \mathrm { d e p t h } } \in \mathbb { R } ^ { N \times d }$ from the depth map via a lightweight adapter, and encode the instruction mask into mask features $\mathbf { F } _ { \mathrm { m a s k } } \in \mathbb { R } ^ { N \times d }$ . We then fuse semantic, geometric, and mask features to obtain a scene-aware representation $\mathbf { F } _ { \mathrm { s c e n e } } \in \mathbb { R } ^ { N \times d }$

To anchor grasp generation in a consistent global frame, we additionally encode the camera parameters and table height into a parameter token $\mathbf { f } _ { \mathrm { p a r a } } \in \mathbb { R } ^ { 1 \times d }$ . Appending $\mathbf { f } _ { \mathrm { p a r a } }$ to $\mathbf { F } _ { \mathrm { s c e n e } }$ yields the final condition embedding $\mathbf { c } \in \mathbb { R } ^ { ( N + 1 ) \times d }$ Implementation details are deferred to Appendix C.1.

## 4.3 Physically-Constrained Denoising for Multimodal Grasp Generation

Cluttered-scene grasping is intrinsically multimodal: under the same scene observation, a target object may admit multiple valid grasp modes. Deterministic supervision tends to average across these modes, often producing implausible intermediate poses. Inspired by winner-takes-all learning [28], we train the diffusion model with a soft WTA-style objective that preserves this one-to-many structure directly in grasping space. Specifically, we sample a set of M valid grasp poses for a target object and inject the same Gaussian noise ε into all M poses at diffusion step t. The network is then trained to explain the corrupted sample through the most compatible grasp mode rather than their average. Here, M serves as a finite hypothesis budget for approximating the dominant modes of the conditional grasp distribution. The resulting Soft-WTA loss is

$$
\mathcal { L } _ { \mathrm { W T A } } = \mathbb { E } _ { t , \varepsilon } \Big [ - \tau \log \sum _ { i = 1 } ^ { M } \exp \big ( - \| \varepsilon - \hat { \varepsilon } _ { i } \| ^ { 2 } / \tau \big ) \Big ] ,\tag{1}
$$

where $\hat { \varepsilon } _ { i }$ is the predicted noise for the i-th grasp hypothesis and τ is a temperature parameter. This objective allows the model to represent diverse grasp modes without introducing handcrafted intermediate representations.

Purely data-driven denoising, however, is often insufficient for dexterous grasping, where millimeter-scale errors can break contact or induce unstable force closure. We therefore augment training with a physics-constrained objective defined on a differentiable dexterous hand simulator:

$$
{ \mathcal { L } } _ { \mathrm { t o t a l } } = \lambda _ { \mathrm { W T A } } { \mathcal { L } } _ { \mathrm { W T A } } + \lambda _ { \mathrm { p h y s } } ( { \mathcal { L } } _ { \mathrm { t h u m b } } + { \mathcal { L } } _ { \mathrm { o p p } } + { \mathcal { L } } _ { \mathrm { S D F } } + { \mathcal { L } } _ { \mathrm { c l o s u r e } } ) .\tag{2}
$$

Here, ${ \mathcal { L } } _ { \mathrm { t h u m b } }$ and $\mathcal { L } _ { \mathrm { o p p } }$ encourage stable contact on the thumb and opposing fingers, $\mathcal { L } _ { \mathrm { c l o s u r e } }$ promotes an antipodal grasp configuration, and $\mathcal { L } _ { \mathrm { S D F } }$ penalizes hand-object interpenetration. Full definitions are provided in Appendix C.3.

## 4.4 Physics-Driven Ranking and Selection

In the inference stage, a single sampled pose may exhibit geometric misalignment due to the inherent stochasticity of generative models. To enable robust execution, we introduce a parallel generation and physics-driven ranking paradigm. Specifically, given the same condition embedding c, we simultaneously sample M independent Gaussian noises to

decode candidate grasp poses $\{ \hat { \bf g } _ { i } \} _ { i = 1 } ^ { M }$ . These candidates are evaluated in the dexterous hand simulator to compute a comprehensive physical score $S ( \hat { \mathbf { g } } )$ , assessing kinematic closure while strictly penalizing physical violations:

$$
S ( \hat { \mathbf { g } } ) = S _ { \mathrm { k i n } } ( \hat { \mathbf { g } } ) - R _ { \mathrm { c o n } } ( \hat { \mathbf { g } } ) + \Omega _ { \mathrm { p h y } } ( \hat { \mathbf { g } } )\tag{3}
$$

The primary metric $S _ { \mathrm { k i n } } ( \hat { \bf g } )$ evaluates force closure stability via finger-to-object distances as Sec. 4.3. The remaining terms act as essential constraint checks: the contact reward $R _ { \mathrm { c o n } } ( \hat { \bf g } )$ encourages multi-finger engagement by rewarding additional fingertip contacts, while the safety penalty $\Omega _ { \mathrm { p h y } }$ linearly penalizes slight interpenetrations and heavily rejects severe collisions or missing thumb-opposition. Ultimately, we filter out physically invalid modes to output the optimal grasp pose.

## 5 Experiment

## 5.1 Implementation Details

We instantiate our framework on the Shadow Dexterous Hand [29], leveraging its high-dimensional action space (24 DoF) to rigorously validate our model’s generative capacity. All subsets of our proposed benchmark adopt an 8/2 split for the train and test sets. The test sets are decoupled into two distinct tracks: Seen objects in unseen clustered scene; unseen objects (see Appendix B). Please refer to Appendix D for more details.

## 5.2 Main Results

As shown in Table 3, our method establishes the state-of-the-art in cluttered-scene dexterous grasping. All baselines in Table 3 are retrained on the same OmniDex training split, rather than evaluated zero-shot. On $\mathcal { D } _ { \mathrm { s t d } }$ with ego-view input, OmniDex achieves 55.80 $\mathrm { S R } _ { \mathrm { s c e n e } }$ and 84.12 CFR, surpassing DexGraspVLA by 14.81 and 2.68 points, respectively. This advantage also holds for unseen objects, where OmniDex attains 59.74 $\mathrm { S R } _ { \mathrm { s c e n e } }$ , demonstrating generalization beyond the training set. Across subsets, our method remains effective across scene layouts and support types, and reaches particularly strong performance on $\mathcal { D } _ { \mathrm { g r i d } }$ , indicating reliable use of structured spatial cues. Overall, these results verify that the OmniDex model delivers superior accuracy, robustness, and cross-setting generalization for realistic cluttered-scene grasping. Qualitative results are shown in Appendix F.2.

Table 3 Main Results on the OmniDex Benchmark. We evaluate our method against baselines across diverse scene subsets and compare with other methods on $\mathcal { D } _ { \mathrm { s t d } } .$ . \* denotes the use of the finger-force trick [7]
<table><tr><td rowspan="2">Method</td><td rowspan="2">Subset</td><td rowspan="2">Input View</td><td colspan="7">Seen Objects</td><td colspan="7">Unseen Objects</td></tr><tr><td> $\mathrm { S R } _ { \mathrm { s c e n e } }$ </td><td> ${ \mathrm { S R } } _ { \mathrm { s c e n e } } ^ { * }$ </td><td> $\mathrm { S R _ { a n y } }$ </td><td> $\mathrm { S R _ { a n y } ^ { * } }$ </td><td> $\mathrm { S R _ { o n e } }$ </td><td> $\mathrm { S R _ { o n e } ^ { * } }$ </td><td>CFR</td><td> $\mathrm { S R } _ { \mathrm { s c e n e } }$ </td><td> $\mathrm { S R } _ { \mathrm { s c e n e } } ^ { * }$ </td><td> $\mathrm { S R _ { a n y } }$ </td><td> $\mathrm { S R _ { a n y } ^ { * } }$ </td><td> $\mathrm { S R _ { o n e } }$ </td><td> $\mathrm { S R _ { o n e } ^ { * } }$ </td><td>CFR</td></tr><tr><td rowspan="10">Ours</td><td> $\mathcal { D } _ { \mathrm { s t d } }$ </td><td>Ego</td><td>33.60</td><td>55.80</td><td>18.87</td><td>36.79</td><td>65.15</td><td>79.35</td><td>84.12</td><td>40.15</td><td>59.74</td><td>27.18</td><td>40.77</td><td>65.54</td><td>81.77</td><td>84.61</td></tr><tr><td> $\mathcal { D } _ { \operatorname* { m i x } }$ </td><td>Ego</td><td>32.05</td><td>52.21</td><td>17.12</td><td>34.04</td><td>58.51</td><td>74.52</td><td>73.12</td><td>37.90</td><td>55.10</td><td>24.13</td><td>35.19</td><td>62.41</td><td>78.02</td><td>83.22</td></tr><tr><td> $\mathcal { D } _ { \mathrm { { b o x } } }$ </td><td>Ego</td><td>32.34</td><td>35.83</td><td>17.06</td><td>19.79</td><td>55.15</td><td>56.52</td><td>99.12</td><td>33.48</td><td>36.50</td><td>17.80</td><td>20.17</td><td>56.77</td><td>58.11</td><td>95.72</td></tr><tr><td> $\mathcal { D } _ { \mathrm { g r i d } }$ </td><td>Ego</td><td>55.01</td><td>79.60</td><td>33.68</td><td>55.02</td><td>91.91</td><td>93.70</td><td>95.46</td><td>43.11</td><td>68.05</td><td>33.97</td><td>52.67</td><td>87.50</td><td>92.09</td><td>94.86</td></tr><tr><td> $\bar { \mathcal { D } _ { \mathrm { s h e l f } } }$ </td><td>Ego</td><td>29.49</td><td>39.46</td><td>18.31</td><td>25.25</td><td>55.17</td><td>60.57</td><td>81.95</td><td>14.96</td><td>24.08</td><td>11.58</td><td>15.08</td><td>34.26</td><td>37.89</td><td>63.57</td></tr><tr><td> $\mathcal { D } _ { \mathrm { s t d } }$ </td><td>Multi</td><td>37.35</td><td>60.65</td><td>22.18</td><td>41.18</td><td>68.01</td><td>82.60</td><td>85.11</td><td>45.05</td><td>64.08</td><td>30.79</td><td>43.76</td><td>68.48</td><td>84.50</td><td>87.24</td></tr><tr><td> $\mathcal { D } _ { \operatorname* { m i x } }$ </td><td>Multi</td><td>35.41</td><td>56.02</td><td>20.54</td><td>38.31</td><td>62.47</td><td>78.06</td><td>75.31</td><td>42.56</td><td>60.34</td><td>28.36</td><td>40.12</td><td>65.83</td><td>82.61</td><td>85.04</td></tr><tr><td> $\mathcal { D } _ { \mathrm { { b o x } } }$ </td><td>Multi</td><td>27.89</td><td>28.52</td><td>19.11</td><td>19.53</td><td>41.20</td><td>42.06</td><td>97.91</td><td>31.14</td><td>31.71</td><td>21.17</td><td>21.79</td><td>43.33</td><td>44.30</td><td>97.64</td></tr><tr><td> $\mathcal { D } _ { \mathrm { g r i d } }$ </td><td>Multi</td><td>60.37</td><td>84.28</td><td>38.44</td><td>58.95</td><td>95.14</td><td>96.52</td><td>96.49</td><td>46.68</td><td>73.14</td><td>37.42</td><td>56.82</td><td>91.21</td><td>95.31</td><td>95.71</td></tr><tr><td> $\mathcal { D } _ { \mathrm { s h e l f } }$ </td><td>Multi</td><td>34.06</td><td>44.63</td><td>22.27</td><td>29.61</td><td>59.88</td><td>64.31</td><td>83.42</td><td>18.21</td><td>28.86</td><td>15.24</td><td>19.37</td><td>39.54</td><td>42.21</td><td>65.29</td></tr><tr><td>DexGraspVLA [30]</td><td> $\mathcal { D } _ { \mathrm { s t d } }$ </td><td>Ego</td><td>24.30</td><td>40.99</td><td>11.87</td><td>24.42</td><td>53.83</td><td>67.05</td><td>81.44</td><td>39.42</td><td>54.97</td><td>29.95</td><td>41.02</td><td>54.89</td><td>71.55</td><td>42.86</td></tr><tr><td>DexGraspNet 2.0 [9]</td><td> $\mathcal { D } _ { \mathrm { s t d } }$ </td><td>Ego</td><td>28.03</td><td>43.64</td><td>13.16</td><td>25.59</td><td>56.04</td><td>68.47</td><td>81.06</td><td>32.40</td><td>47.60</td><td>18.83</td><td>27.39</td><td>58.77</td><td>73.73</td><td>85.15</td></tr><tr><td>Grasp-as-You-Say [31]</td><td> $\mathcal { D } _ { \mathrm { s t d } }$ </td><td>Ego</td><td>0.24</td><td>1.68</td><td>0.09</td><td>0.15</td><td>20.91</td><td>27.45</td><td>38.57</td><td>0.44</td><td>2.55</td><td>0.10</td><td>0.42</td><td>19.46</td><td>25.57</td><td>64.25</td></tr></table>

## 5.3 Ablation Studies

We evaluate our procedures on a representative mini-split curated from $\mathcal { D } _ { \mathrm { s t d } }$ (see Appendix B). By default, baselines operate on a single ego-centric view. We use $\mathrm { S R } _ { \mathrm { s c e n e } }$ as the principal metric to quantify grasp stability. More ablation studies can be seen in Appendix E.

Effect of Benchmark Scale. To justify the massive scale of our proposed OmniDex benchmark, we train our model on incrementally larger fractions of the training set, and evaluate it on the full test set. As Table 4 shows, performance scales strictly with data volume. Scaling explicitly enhances the model’s capacity to reason about complex inter-object physical interactions in cluttered layouts; notably, CFR surges from 81.54% (20% subset) to 83.66% (100% subset). This significant improvement in stable grasping validates the necessity of our large-scale data generation pipeline.

Table 4 Quantitative results when trained on incrementally larger fractions of training dataset.
<table><tr><td>Metrics</td><td>20%</td><td>40%</td><td>60%</td><td>80%</td><td>100%</td></tr><tr><td> $\mathrm { S R } _ { \mathrm { s c e n e } }$ </td><td>40.50</td><td>41.04</td><td>43.84</td><td>44.32</td><td>44.41</td></tr><tr><td>CFR</td><td>81.54</td><td>82.64</td><td>82.13</td><td>84.15</td><td>83.66</td></tr></table>

Effect of Input Modalities. We analyze the role of various inputs as shown in Table 5. Depth provides the foundational shape cognition. RGB supplies scene semantics and, more crucially, localizes the target object within the cluster. Camera parameters ensure accurate geometric perception. Multi-view inputs yield significant performance gains by enabling comprehensive scene understanding. Table height serves as a vital spatial anchor; since the coordinate origin lies at the table center, it aligns the depth-derived geometry with grasp ground truths to explicitly dictate precise grasp locations.

Table 5 Quantitative ablation on various input conditions. The highlighted row represents baseline.
<table><tr><td>RGB</td><td>Depth</td><td>Cam. Para.</td><td>Table Height</td><td>Multi-view</td><td> $\mathrm { S R } _ { \mathrm { s c e n e } }$ </td><td>CFR</td></tr><tr><td>√</td><td></td><td>√</td><td>√</td><td></td><td>31.89</td><td>79.97</td></tr><tr><td></td><td>√</td><td>√</td><td>√</td><td></td><td>36.80</td><td>82.81</td></tr><tr><td>√</td><td>√</td><td></td><td>√</td><td></td><td>43.68</td><td>84.09</td></tr><tr><td>√</td><td>√</td><td>√</td><td></td><td></td><td>40.64</td><td>84.04</td></tr><tr><td>√</td><td>√</td><td>√</td><td>√</td><td>√</td><td>49.87</td><td>82.11</td></tr><tr><td>√</td><td>√</td><td>√</td><td>√</td><td></td><td>44.41</td><td>83.66</td></tr></table>

Effect of Model Components. Table 6 details our architectural ablations. Standard denoising loss traps generation in an averaging effect across grasp hypotheses. Conversely, ${ \mathcal { L } } _ { \mathrm { W T A } }$ resolves this multi-modal manifold issue, substantially boosting ${ \mathrm { S R } } _ { \mathrm { s c e n e } } .$ Training-phase physical constraints are vital for overcoming the critical last-millimeter contact challenge. At inference, our physics-driven ranking drastically outperforms naive random sampling, elevating $\mathrm { S R } _ { \mathrm { s c e n e } }$ from 30% to 40%. Test-time optimization yields limited gains, bottlenecked by inherent instability and depth-backprojection occlusions (see details in Appendix E).

Table 6 Component-wise ablation study evaluated on both training and inference phases. TTP denotes Test-Time Optimization.
<table><tr><td colspan="5">Training Phase</td><td colspan="3">Inference Phase</td><td colspan="2">Metrics</td></tr><tr><td> $\mathcal { L } _ { \mathrm { s t a n d } }$ </td><td>LWTA</td><td></td><td>LSDF Lthumb, Lopp</td><td>Lclosure</td><td>Naive Samp.</td><td>Rank.</td><td>TTP</td><td> ${ \mathrm { S R } } _ { \mathrm { s c e n e } }$ </td><td>CFR</td></tr><tr><td>V</td><td></td><td></td><td></td><td></td><td></td><td>√</td><td></td><td>37.79</td><td>81.37</td></tr><tr><td></td><td>√</td><td></td><td></td><td></td><td></td><td>√</td><td></td><td>38.07</td><td>81.31</td></tr><tr><td></td><td>√</td><td>√</td><td></td><td></td><td></td><td>√</td><td></td><td>41.93</td><td>84.06</td></tr><tr><td></td><td>√</td><td>√</td><td>√</td><td></td><td></td><td>√</td><td></td><td>43.19</td><td>84.40</td></tr><tr><td></td><td>√</td><td>√</td><td>√</td><td>√</td><td>5</td><td></td><td></td><td>36.08</td><td>79.71</td></tr><tr><td></td><td>√</td><td>√</td><td>√</td><td>√</td><td></td><td></td><td>√</td><td>36.16</td><td>2.56</td></tr><tr><td></td><td>√</td><td>√</td><td>√</td><td>√</td><td></td><td>y</td><td></td><td>44.41</td><td>83.66</td></tr></table>

## 6 Conclusion

In this paper, we presented OmniDex to tackle the data and algorithmic challenges of dexterous grasping in cluttered scenes. Our OmniDex benchmark leverages a scalable seed-and-filter strategy to deliver over 2.6 million scenes and 0.4 billion validated grasp ground truths, complete with rich multi-modal observations. Built upon this, our novel generation framework couples Soft Winner-Takes-All learning with human-inspired physical constraints to effectively capture grasp multimodality and bridge the crucial last-millimeter precision gap. Experimental results confirm that our OmniDex model achieves state-of-the-art performance on the proposed benchmark, significantly outperforming existing baselines on standard cluttered scenes.

## References

[1] J. Duan, S. Yu, H. L. Tan, H. Zhu, and C. Tan. A survey of embodied ai: From simulators to research tasks. IEEE Trans. Emerg. Top. Comput. Intell., 6(2):230–244, 2022.

[2] H.-S. Fang, C. Wang, M. Gou, and C. Lu. Graspnet-1billion: A large-scale benchmark for general object grasping. In IEEE Conf. Comput. Vis. Pattern Recog., pages 11444–11453, 2020.

[3] H.-S. Fang, C. Wang, H. Fang, M. Gou, J. Liu, H. Yan, W. Liu, Y. Xie, and C. Lu. Anygrasp: Robust and efficient grasp perception in spatial and temporal domains. IEEE Trans. Robot., 39(5):3929–3945, 2023.

[4] Y. Liu, Y. Yang, Y. Wang, X. Wu, J. Wang, Y. Yao, S. Schwertfeger, S. Yang, W. Wang, J. Yu, et al. Realdex: towards human-like grasping for robotic dexterous hand. In Proc. Int. Joint Conf. Artif. Intell., pages 6859–6867, 2024.

[5] J. Ye, K. Wang, C. Yuan, R. Yang, Y. Li, J. Zhu, Y. Qin, X. Zou, and X. Wang. Dex1b: Learning with 1b demonstrations for dexterous manipulation. In RSS Workshop on Dexterous Manipulation, 2025.

[6] J. Chen, Y. Ke, L. Peng, and H. Wang. Dexonomy: Synthesizing all dexterous grasp types in a grasp taxonomy. In RSS Workshop on Dexterous Manipulation, 2025.

[7] R. Wang, J. Zhang, J. Chen, Y. Xu, P. Li, T. Liu, and H. Wang. Dexgraspnet: A large-scale robotic dexterous grasp dataset for general objects based on simulation. In IEEE Int. Conf. Robot. Autom., pages 11359–11366. IEEE, 2023.

[8] Y. Xu, W. Wan, J. Zhang, H. Liu, Z. Shan, H. Shen, R. Wang, H. Geng, Y. Weng, J. Chen, et al. Unidexgrasp: Universal robotic dexterous grasping via learning diverse proposal generation and goal-conditioned policy. In IEEE Conf. Comput. Vis. Pattern Recog., pages 4737–4746, 2023.

[9] J. Zhang, H. Liu, D. Li, X. Yu, H. Geng, Y. Ding, J. Chen, and H. Wang. Dexgraspnet 2.0: Learning generative dexterous grasping in large-scale synthetic cluttered scenes. In Conf. Robot. Learn., 2024.

[10] J. Zhang, Z. Ma, T. Wu, Z. Chen, and H. Dong. Cadgrasp: Learning contact and collision aware general dexterous grasping in cluttered scenes. In Adv. Neural Inf. Process. Syst., 2025.

[11] J. He, D. Li, X. Yu, Z. Qi, W. Zhang, J. Chen, Z. Zhang, Z. Zhang, L. Yi, and H. Wang. Dexvlg: Dexterous vision-languagegrasp model at scale. In Int. Conf. Comput. Vis., pages 14248–14258, 2025.

[12] Y.-L. Wei, M. Lin, Y. Lin, J.-J. Jiang, X.-M. Wu, L.-A. Zeng, and W.-S. Zheng. Afforddexgrasp: Open-set language-guided dexterous grasp with generalizable-instructive affordance. In Int. Conf. Comput. Vis., pages 11818–11828, 2025.

[13] J. Lundell, F. Verdoja, and V. Kyrki. Ddgc: Generative deep dexterous grasping in clutter. IEEE Robot. Autom. Lett., 6(4): 6899–6906, 2021.

[14] Z. Chen, Q. Yan, Y. Chen, T. Wu, J. Zhang, Z. Ding, J. Li, Y. Yang, and H. Dong. Clutterdexgrasp: A sim-to-real system for general dexterous grasping in cluttered scenes. In Conf. Robot. Learn., pages 885–905. PMLR, 2025.

[15] Y. Wang, J. Ye, C. Xiao, Y. Zhong, H. Tao, H. Yu, Y. Liu, J. Yu, and Y. Ma. Dexh2r: A benchmark for dynamic dexterous grasping in human-to-robot handover. In Int. Conf. Comput. Vis., pages 12702–12712, 2025.

[16] M. Xu, H. Zhang, Y. Hou, Z. Xu, L. Fan, M. Veloso, and S. Song. Dexumi: Using human hand as the universal manipulation interface for dexterous manipulation. In Workshop on Human to Robot, 2025.

[17] J. Hang, X. Lin, T. Zhu, X. Li, R. Wu, X. Ma, and Y. Sun. Dexfuncgrasp: A robotic dexterous functional grasp dataset constructed from a cost-effective real-simulation annotation system. In AAAI, volume 38, pages 10306–10313, 2024.

[18] Y. Zhong, Q. Jiang, J. Yu, and Y. Ma. Dexgrasp anything: Towards universal robotic dexterous grasping with physics awareness. In IEEE Conf. Comput. Vis. Pattern Recog., pages 22584–22594, 2025.

[19] W. Wan, H. Geng, Y. Liu, Z. Shan, Y. Yang, L. Yi, and H. Wang. Unidexgrasp++: Improving dexterous grasping policy learning via geometry-aware curriculum and iterative generalist-specialist learning. In Int. Conf. Comput. Vis., pages 3891–3902, 2023.

[20] Y. Qin, B. Huang, Z.-H. Yin, H. Su, and X. Wang. Dexpoint: Generalizable point cloud reinforcement learning for sim-to-real dexterous manipulation. In Conf. Robot. Learn., pages 594–605. PMLR, 2023.

[21] J. Chen, Y. Ke, and H. Wang. Bodex: Scalable and efficient robotic dexterous grasp synthesis using bilevel optimization. In IEEE Int. Conf. Robot. Autom., pages 01–08. IEEE, 2025.

[22] T. Wu, J. Zhang, X. Fu, Y. Wang, J. Ren, L. Pan, W. Wu, L. Yang, J. Wang, C. Qian, et al. Omniobject3d: Large-vocabulary 3d object dataset for realistic perception, reconstruction and generation. In IEEE Conf. Comput. Vis. Pattern Recog., pages 803–814, 2023.

[23] A. X. Chang, T. Funkhouser, L. Guibas, P. Hanrahan, Q. Huang, Z. Li, S. Savarese, M. Savva, S. Song, H. Su, et al. Shapenet: An information-rich 3d model repository. arXiv preprint arXiv:1512.03012, 2015.

[24] L. Downs, A. Francis, N. Koenig, B. Kinman, R. Hickman, K. Reymann, T. B. McHugh, and V. Vanhoucke. Google scanned objects: A high-quality dataset of 3d scanned household items. In IEEE Int. Conf. Robot. Autom., pages 2553–2560, 2022.

[25] J. Collins, S. Goel, K. Deng, A. Luthra, L. Xu, E. Gundogdu, X. Zhang, T. F. Y. Vicente, T. Dideriksen, H. Arora, et al. Abo: Dataset and benchmarks for real-world 3d object understanding. In IEEE Conf. Comput. Vis. Pattern Recog., pages 21126–21136, 2022.

[26] M. Mittal, P. Roth, J. Tigue, A. Richard, O. Zhang, P. Du, A. Serrano-Munoz, X. Yao, R. Zurbrügg, N. Rudin, et al. Isaac lab: A gpu-accelerated simulation framework for multi-modal robot learning. arXiv preprint arXiv:2511.04831, 2025.

[27] M. Oquab, T. Darcet, T. Moutakanni, H. Vo, M. Szafraniec, V. Khalidov, P. Fernandez, D. Haziza, F. Massa, A. El-Nouby, et al. Dinov2: Learning robust visual features without supervision. Trans. Mach. Learn. Res., 2024.

[28] C. Rupprecht, I. Laina, R. DiPietro, M. Baust, F. Tombari, N. Navab, and G. D. Hager. Learning in an uncertain world: Representing ambiguity through multiple hypotheses. In Int. Conf. Comput. Vis., pages 3591–3600, 2017.

[29] Shadow Robot Company. Shadow dexterous hand. https://www.shadowrobot.com/dexterous-hand/, 2005.

[30] Y. Zhong, X. Huang, R. Li, C. Zhang, Z. Chen, T. Guan, F. Zeng, K. N. Lui, Y. Ye, Y. Liang, et al. Dexgraspvla: A vision language-action framework towards general dexterous grasping. In AAAI, volume 40, pages 18836–18844, 2026.

[31] Y.-L. Wei, J.-J. Jiang, C. Xing, X.-T. Tan, X.-M. Wu, H. Li, M. Cutkosky, and W.-S. Zheng. Grasp as you say: Languageguided dexterous grasp generation. volume 37, pages 46881–46907, 2024.

[32] L. Keselman, J. Iselin Woodfill, A. Grunnet-Jepsen, and A. Bhowmik. Intel realsense stereoscopic depth cameras. In IEEE Conf. Comput. Vis. Pattern Recog., pages 1–10, 2017.

[33] I. Loshchilov and F. Hutter. Decoupled weight decay regularization. arXiv preprint arXiv:1711.05101, 2017.

[34] Tianji Intelligent Systems. Marvin m6s long (ccs-920) force-controlled humanoid arm, 2026. Product documentation.

[35] BrainCo. Revo2 bionic dexterous hand, 2026. Product documentation.

## A Benchmark Generation Pipeline Details

## A.1 Cluttered Scene Geometry Database Generation

Supporting Base Asset Setting. All supporting structures are canonicalized to a gravity-aligned orientation. We standardize their local coordinate frames by placing the origin at the intersection of the gravity axis and the ground plane, facilitating straightforward base anchoring within the simulation environment.

Simulation Setup and Physical Properties. We leverage Isaac Gym4 as our core physics engine to simulate clustered scene. To ensure stable and uniform dynamic behaviors across varied shapes, all object instances are assigned universally consistent physical properties: a density of 500 kg/m³, coupled with linear and angular damping factors of 1.0 and 0.3, respectively.

Subset-Specific Initialization Strategies. To systematically decouple the generalization evaluation, we employ tailored spatial initialization logic for each subset:

$\bullet \mathcal { D } _ { \mathrm { s t d } }$ and $\mathcal { D } _ { \operatorname* { m i x } }$ : We randomly sample 2 to 7 unique object instances. Objects are dropped from a height of 0.5m above the table and their dropping horizontal spawn coordinates are tightly bounded within the table’s footprint.

$\bullet D _ { \mathrm { b o x } }$ : 2 to 10 unique instances are initialized 0.4m above the top rim of the box. The horizontal dropping area is deliberately constricted to a 0.8 scale of the box’s extent, naturally inducing dense and complex clustered stacking.

$\bullet \mathcal { D } _ { \mathrm { g r i d } }$ : We replicate 4 to 10 identical instances. To simulate structured arrays, objects are uniformly distributed within a 0.8 scaled tabletop area. Layouts are dynamically computed utilizing object bounding boxes, enforcing a strict minimum inter-instance spacing of 5cm.

$\mathbf { \nabla } \cdot \mathcal { D } _ { \mathrm { s h e l f } } .$ Based on explicitly annotated layer heights, objects are uniformly arranged along each shelf layer using their bounding boxes, maintaining a minimum spacing of 4cm.

Dynamic Resting Conditions. The simulation evolves until the scene reaches a physically stable resting state. We define termination by two rigorous criteria: (1) kinematic energy dissipation, where the linear and angular velocities of all objects fall below 0.03 m/s and 0.10 rad/s, respectively; or (2) reaching a hard cutoff at 500 simulation steps to prevent infinite oscillation.

Rigorous Geometric Validation. To completely eliminate physically anomalous artifacts commonly found in automated generation pipelines, we implement a strict post-simulation geometric sanity check. A generated scene is systematically rejected unless it satisfies the following tolerance thresholds:

•Sinking & Floating: The maximum allowed vertical deviation (embedding into the base or hovering above it) is strictly bounded to 3mm.

•Inter-object Penetration: The maximum permissible mutual penetration depth between any two clustered objects is capped at 2mm.

Tolerance in Contact Filtering. Our filtering is conservative but not perfectly zero-tolerance at the contact level. In practice, small numerical overlaps may arise from mesh decomposition, contact discretization, and simulator tolerances. We therefore allow a limited penetration margin during filtering and reject only clearly invalid grasp poses. In the current implementation, object–support penetration is tolerated up to 2cm before rejection, while shallower overlaps are retained to reduce false negatives caused by imperfect collision geometry.

## A.2 Cluttered Scene Visual Database Generation

We design a multi-camera rig setup that mimics typical real-world robotic workspaces. We deploy five fixed Intel RealSense D455 sensors [32]. To maximize the visual coverage of complex object interactions and mitigate severe occlusions, the camera poses are adaptively configured based on the subset:

$\mathbf { \partial } \cdot \mathcal { D } _ { \mathrm { s t d } } , \mathcal { D } _ { \mathrm { m i x } } , \mathcal { D } _ { \mathrm { g r i d } } \colon$ For scenes without pronounced boundary occlusions, the primary ego-centric camera is positioned 0.6m above the global coordinate origin and retracted 0.3m outward from the table edge. Auxiliary multi-view cameras are symmetrically deployed 0.25m outward from the table corners. To guarantee optimal object framing, the optical axes of all cameras are explicitly oriented towards a focal point 0.15m below the spatial center.

$\bullet D _ { \mathrm { b o x } }$ : The high physical boundaries of the container in the box subset introduce severe peripheral occlusions. To counteract this, we dynamically tighten the camera rig. The ego-centric camera and the auxiliary corner cameras are moved closer, fixed at a uniform 0.2m outward offset from the table edges and corners.

$\mathbf { \nabla } \cdot \mathcal { D } _ { \mathrm { s h e l f } } \mathbf { \dot { z } }$ The multi-layered geometry of shelf imposes severe vertical visibility constraints, rendering standard top-down or steep perspectives entirely ineffective due to overhanging structures. We deploy a specialized 5-camera rig driven by a dynamic, layer-wise elevation-tracking strategy. The z-axis coordinates of all five cameras and their focal targets are aligned with the height of shelf, establishing a zero-pitch, horizon-aligned viewing frustum. The primary ego-centric camera is positioned squarely in front of the shelf, with its standoff distance adaptively scaled by a factor of 1.2 of the shelf’s spatial diagonal. Simultaneously, the four auxiliary cameras are deployed radially outward from the top-surface corners, pushed along the diagonal axes by a scaling factor of 0.8.

## B Benchmark Split and Density Details

To comprehensively assess grasping stability under varying degrees of physical entanglement, we deliberately modulate the cluster density, particularly for the $\mathcal { D } _ { \mathrm { s t d } }$ and $\mathcal { D } _ { \operatorname* { m i x } }$ subsets. This is explicitly achieved by constraining the object dropping area during simulation as a scalar ratio of the full tabletop extent. Specifically, we synthesize scenes across three distinct density tiers (ratios of 0.1, 0.2, and 0.3), with each density level contributing approximately 0.6M scenes to the overarching benchmark.

For the ablation mini-split, we uniformly sample ∼10% of the training and testing scenes across all density tiers, yielding a diverse subset of roughly 90K scenes. This highly challenging spatial configuration allows us to rigorously benchmark the geometric reasoning capabilities of individual model components under dense clutter.

The asset pool is finite and fixed. To maximize the diversity of the training set, we intentionally assigned more challenging object assets to the seen objects split. Consequently, the remaining unseen Objects split is relatively simpler on average, and the two tracks are not difficulty-balanced.

## C OmniDex Model Details

## C.1 Single-View Condition Embedding

For the ego-centric single-view setting, we extract patch-level semantic features from the RGB image using a pretrained DINOv2 [27] backbone, yielding $\mathbf { F } _ { \mathrm { i m g } } \in \mathbb { R } ^ { \bar { N } \times d }$ , where N is the number of patches and d is the embedding dimension. To encode scene geometry, we back-project the depth map into a 3D coordinate map $\mathbf { X } \in \mathbb { R } ^ { H \times W \times 3 }$

$$
\mathbf { X } \left( u , v \right) = \mathbf { D } \left( u , v \right) \mathbf { K } ^ { - 1 } [ u , v , 1 ] ^ { T } ,\tag{4}
$$

where $( u , v )$ denotes the 2D pixel coordinate. We then use a lightweight adapter to map the XYZ representation into the DINOv2 input space, producing geometric features $\mathbf { F } _ { \mathrm { d e p t h } } \breve { \in } \mathbb { R } ^ { N \times d }$

To isolate the target object from clutter, the instruction mask is encoded into mask features $\mathbf { F } _ { \mathrm { m a s k } } \in \mathbb { R } ^ { N \times d }$ . We concatenate semantic, geometric, and mask features along the channel dimension and fuse them into a scene-aware representation $\mathbf { F } _ { \mathrm { s c e n e } } \in \mathbb { R } ^ { N \times d }$

To preserve a consistent global reference for grasp generation, we additionally encode the camera parameters and table height. Specifically, we compute the projection matrix $\mathbf { P } = \mathbf { K } \mathbf { T }$ and encode P together with H via MLPs to obtain a parameter token $\mathbf { f } _ { \mathrm { p a r a } } \in \mathbb { R } ^ { 1 \times d }$ . Appending $\mathbf { f } _ { \mathrm { p a r a } }$ to $\mathbf { F } _ { \mathrm { s c e n e } }$ yields the final condition embedding $\mathbf { c } \in \mathbb { R } ^ { ( N + 1 ) \times d }$

## C.2 Multi-View Input Fusion

To effectively process extreme spatial occlusions, we introduce a hybrid and physics-aware encoder for multi-view observation. Instead of relying on a monolithic black-box transformer to indiscriminately mix multi-view input, our architecture deliberately decouples the fusion paradigm into two distinct pathways: a data-driven semantic early RGB fusion, a rigorous geometry-aware depth reprojection, and camera relationship tokens.

Data-Driven Early RGB Fusion. Traditional multi-view encoders often suffer from computational overhead when processing independent visual tokens across views. To ensure efficiency without losing critical textural semantics, we concatenate the multi-view RGB, depth, and instance masks, feeding them into a lightweight convolutional adapter. This yields an inter-view alpha map via a softmax operation across the view dimension V. The RGB inputs are then aggregated using this learned alpha map, effectively allowing the network to dynamically stitch unoccluded visual patches from auxiliary cameras into a cohesive representation.

Rigorous Geometry-Aware Depth Reprojection. Unlike textural features, depth represents strict spatial geometry that cannot be naively interpolated. To capture the 3D manifold of the cluttered scene, we explicitly route the multiview depth maps to the primary anchor camera using strict camera parameters. To maintain computational tractability, this point-cloud inversion is performed utilizing a hardware-accelerated Z-buffer scattering mechanism. The physically fused depth map is subsequently converted into a normalized XYZ spatial map.

Camera Relationship Tokens. To supplement the dense visual features, our encoder explicitly injects global contextual priors. The camera poses are tokenized and processed through a View Relation Transformer to capture the spatial topology of the multi-camera rig. These auxiliary tokens are concatenated with the visual features to form the final comprehensive observation sequence for the downstream generative policy.

## C.3 Physically-Constrained Loss

We first back-project the depth map and filter it utilizing the instruction mask to extract a partial 3D point cloud of the targeted object, denoted as $\mathcal { P } _ { \mathrm { o b j } }$ . Simultaneously, we obtain the predicted grasp pose $\widehat { \mathbf { g } }$ from the denoised output and feed it into the differentiable dexterous hand simulator to compute the predefined contact points on the hand surface [7], denoted as $\mathcal { P } _ { \mathrm { c o n } }$ . For any point $p \in \mathcal { P } _ { \mathrm { c o n } }$ , its minimal distance to the object point cloud is defined as $d ( p ) = \mathrm { m i n } _ { q \in \mathcal { P } _ { \mathrm { o b j } } } \| p - q \| _ { 2 }$

Effective grasping strictly requires a stable kinematic clamping force. Morphologically, the thumb plays a dominant role in opposing the other fingers. We explicitly enforce this by partitioning the contact points into a thumb set $\mathcal { P } _ { \mathrm { t h u m b } }$ and an opposing fingers set $\mathcal { P } _ { \mathrm { o p p } }$ . Specifically, we apply a thumb contact loss as:

$$
\mathcal { L } _ { \mathrm { t h u m b } } = \operatorname* { m a x } \left( \frac { 1 } { | \mathcal { P } _ { \mathrm { t h u m b } } | } \sum _ { p \in \mathcal { P } _ { \mathrm { t h u m b } } } d ( p ) - \epsilon _ { \mathrm { c o n t a c t } } , 0 \right)\tag{5}
$$

where $\epsilon _ { \mathrm { c o n t a c t } }$ is a contact tolerance margin.

For the opposing fingers, forming a stable force closure often only requires a partial subset to make contact. Therefore, we apply an identical loss formulation for $\mathcal { L } _ { \mathrm { o p p } } .$ , but compute the average distance over exclusively the $K$ closest points within $\mathcal { P } _ { \mathrm { o p p } }$

Beyond simple proximity, a stable grasp necessitates a valid force closure, requiring the fingers to approach the object from opposing directions. To explicitly encourage this antipodal grasping pattern, we introduce a force closure loss as:

$$
\mathcal { L } _ { \mathrm { c l o s u r e } } = \operatorname* { m a x } \left( \cos { \left( v _ { \mathrm { t h u m b } } , v _ { \mathrm { o p p } } \right) } + \epsilon _ { \mathrm { c l o s u r e } } , 0 \right)\tag{6}
$$

where $v _ { \mathrm { t h u m b } } , v _ { \mathrm { o p p } }$ are approach vectors from finger to object center, $\cos ( \cdot , \cdot )$ denotes the cosine similarity, and $\epsilon _ { \mathrm { c l o s u r e } }$ is a tolerance margin.

Finally, to prevent physically impossible interpenetrations between the hand and the target object, we incorporate a Signed Distance Field (SDF) penetration loss $\mathcal { L } _ { \mathrm { S D F } }$ . Let $\Phi _ { \mathrm { o b j } } ( \cdot )$ denote the SDF of the object, where negative values indicate spatial locations strictly inside the object’s geometry. We penalize any hand mesh vertices $\mathcal { P } _ { \mathrm { h a n d } }$ that penetrate the object as follows:

$$
\mathcal { L } _ { \mathrm { S D F } } = \frac { 1 } { | \mathcal { P } _ { \mathrm { h a n d } } | } \sum _ { p \in \mathcal { P } _ { \mathrm { h a n d } } } \operatorname* { m a x } ( - \Phi _ { \mathrm { o b j } } ( p ) , 0 )\tag{7}
$$

where $\mathcal { P } _ { \mathrm { h a n d } }$ represents the set of all vertices on the predicted hand mesh. This term effectively repels the hand fingers and palm from intersecting with the object’s interior volume.

## D Supplementary Implementation Details

All experiments are conducted on NVIDIA A800 GPUs. The generation of the cluster scene geometry database and the cluster scene visual database consumed about 5K and 17K GPU hours, respectively. The AdamW optimizer [33] is used with an initial learning rate of $e ^ { - 4 }$ and a weight decay of $e ^ { - 4 }$ , All training durations are set to 25 epochs. For both the STA loss optimization during training and the physics-driven ranking strategy during inference, we uniformly set the number of parallel grasp hypotheses to $M = 6 4$ . We define loss weights $\lambda _ { \mathrm { W T A } } = 1$ and $\lambda _ { \mathrm { p h y s } } = 1 0 0$ , respectively. Within the physical constraint formulation, the relative weighting among the sub-components is specifically calibrated as $\mathcal { L } _ { \mathrm { t h u m b } } : \mathcal { L } _ { \mathrm { o p p } } : \mathcal { L } _ { \mathrm { S D F } } : \mathcal { L } _ { \mathrm { c l o s u r e } } = 5 : 1 : 0 . 2 : 2$ . Notably, we assign a substantially higher penalty weight to the thumb, explicitly reflecting its indispensable kinematic role in establishing robust force closure. Imposing strict physical penalties on an uninitialized network often leads to severe learning instability. To mitigate this, we employ a delayed linear warm-up scheduling for the physical constraints. Specifically, the physical loss term is entirely deactivated $( \lambda _ { \mathrm { p h y s } } ~ = ~ 0 )$ during the initial 10 epochs, allowing the generative network to first learn a stable grasp manifold. Subsequently, $\lambda _ { \mathrm { p h y s } }$ is linearly ramped up to its maximum target value of 100 between epochs 10 and 20, ensuring a smooth transition towards physically compliant grasp synthesis.

Our benchmark is constructed from third-party assets obtained from their official releases, including OmniObject3D [22], ShapeNetCore v2 [23], Google Scanned Objects [24], ABO [25], and Isaac Lab [26]. We credit these original resources in the paper and use them in accordance with their respective licenses and terms of use. In particular, OmniObject3D and Google Scanned Objects are released under CC BY 4.0, ABO is released under CC BY 4.0, ShapeNetCore v2 is used under the ShapeNet Terms of Use for non-commercial research, and Isaac Lab is distributed under the BSD-3-Clause license.

## E Supplementary Ablation Study

## E.1 Inference Ablation

To evaluate the efficiency-performance trade-off during inference, we ablate M, the number of parallel generated grasp hypotheses, and comparison with TTP. For a rigorously fair comparison, the TTP baseline is configured to optimize the exact same physical loss objectives utilized in the training phase, executing for 200 steps per grasp. Quantitative results are detailed in Table D1.

Table D1 Effect of parallel hypothesis generation vs. Test-Time Optimization.
<table><tr><td rowspan="2">Metric</td><td colspan="7">Parallel Hypotheses (M)</td><td rowspan="2">TTP</td></tr><tr><td>1</td><td>2</td><td>4</td><td>8</td><td>16</td><td>32</td><td>64</td></tr><tr><td> $\mathrm { S R } _ { \mathrm { s c e n e } }$ </td><td>36.08</td><td>38.02</td><td>41.06</td><td>42.47</td><td>43.41</td><td>44.37</td><td>44.41</td><td>36.16</td></tr><tr><td>CFR</td><td>79.71</td><td>80.81</td><td>81.58</td><td>82.26</td><td>82.62</td><td>83.09</td><td>83.66</td><td>2.56</td></tr><tr><td>Inference Time (ms)</td><td>219.14</td><td>233.35</td><td>266.39</td><td>325.20</td><td>543.93</td><td>948.33</td><td>1598.41</td><td>6044.60</td></tr></table>

As observed, scaling the hypothesis pool M consistently improves grasping success rates with only a negligible sacrifice in inference latency, validating the high efficiency of our parallel generation paradigm. In stark contrast, TTP incurs prohibitive computational overhead and suffers from severe optimization instability. We empirically find that the TTP trajectory frequently diverges from physically viable grasp manifolds, ultimately degrading both execution speed and overall performance.

## E.2 Failure Analysis

We analyze OmniDex failures on the $\mathcal { D } _ { \mathrm { s t d } }$ mini-split. As summarized in Table D2, among grasps failing ${ \mathrm { S R } } _ { \mathrm { s c e n e } } ,$ 4.29% are caused by insufficient contact and 95.71% by stability failure after contact is established. Among grasps

failing CFR, 61.26% involve supporting-base collisions and 38.74% involve non-target-object collisions. These results indicate that post-contact grasp stability and collision avoidance in clutter are the dominant remaining challenges.

Table D2 Failure breakdown on the $\mathcal { D } _ { \mathrm { s t d } }$ mini-split.
<table><tr><td colspan="2"> $\mathrm { S R } _ { \mathrm { s c e n e } }$  failures</td><td colspan="2">CFR failures</td></tr><tr><td>Failure type</td><td>Rate</td><td>Failure type</td><td>Rate</td></tr><tr><td>Insufficient contact</td><td>4.29%</td><td>Supporting-base collision</td><td>61.26%</td></tr><tr><td>Stability failure</td><td>95.71%</td><td>Non-target-object collision</td><td>38.74%</td></tr></table>

## E.3 IK Feasibility Audit

To assess the kinematic feasibility of successful predicted grasps, we evaluate the $\mathcal { D } _ { \mathrm { s t d } }$ mini-split using a Franka arm placed on the ego-view side of the table. We consider five representative arm-base configurations, where the rear offset denotes the horizontal distance from the table edge and the height offset denotes the vertical displacement above the tabletop. For each grasp passing $\mathrm { S R } _ { \mathrm { s c e n e } }$ , we transform the end-effector pose into the arm-base frame and solve IK subject to the Franka joint limits. IK feasibility ranges from 83.75% to 87.98% across the five configurations, indicating that most generated grasps are kinematically reachable by the robot arm. The detailed results are shown in Table D3.

Table D3 IK feasibility under five representative Franka arm placements on the $\mathcal { D } _ { \mathrm { s t d } }$ mini-split.
<table><tr><td>Rear offset (cm)</td><td>Height offset (cm)</td><td>IK feasibility (%)</td></tr><tr><td>0.0</td><td>+2.5</td><td>87.98</td></tr><tr><td>0.0</td><td>+5.0</td><td>87.94</td></tr><tr><td>2.5</td><td>+5.0</td><td>86.68</td></tr><tr><td>5.0</td><td>+2.5</td><td>85.63</td></tr><tr><td>7.5</td><td>+5.0</td><td>83.75</td></tr></table>

## F Supplementary Visual Results

## F.1 OmniDex Benchmark Visual Results

To comprehensively demonstrate the high fidelity and multi-modal richness of our proposed OmniDex benchmark, we present extensive visual galleries for each subset in Figures E1 through E5. For every representative scene, we visualize the complete suite of aligned multi-modal sensory inputs: multi-view camera images, depth maps, instruction masks, and the physically-validated dexterous grasp ground truths. These visualizations collectively underscore the extreme spatial complexity of the cluttered layouts and the rigorous alignment of our grasping label across diverse visual modalities.

## F.2 OmniDex Model Visual Results

We present qualitative results of OmniDex model across diverse benchmark subsets in Fig. E6. The predicted grasps remain stable and semantically consistent under substantially different scene layouts. Notably, OmniDex is able to localize the target object and generate plausible grasping poses even in the presence of tight inter-object spacing and strong geometric ambiguity among similar instances. These visualizations further support our quantitative findings, showing that the proposed multimodal conditioning and physics-aware generation scheme generalize well across vary ing scene structures and grasping difficulties.

## F.3 Real-Robot Demonstration

We conducted a small qualitative real-robot study using a Tianji Marvin M6S arm [34] and a BrainCo Revo2 hand [35], with the camera placement approximately matching the simulated setup. OmniDex was retrained on a hardware-

![](images/5f0defe2753771fc3b034bc7b0dc51da3ca91f868409441a416fe8978199a40c.jpg)  
Figure E1 Qualitative visualization of the $\mathcal { D } _ { \mathrm { s t d } }$ subset.

![](images/dfa335a6872d552fd9d8d8f07b5752e00ef54d49338f3da97cc12bc509779d10.jpg)  
Figure E2 Qualitative visualization of the $\mathcal { D } _ { \operatorname* { m i x } }$ subset.

![](images/d13cde19bdce25686f95e95d6c1d4b3c4de1f0f037e56d7eff8e9200b8811a47.jpg)  
Figure E3 Qualitative visualization of the $\mathcal { D } _ { \mathrm { { b o x } } }$ subset.

![](images/eaea11e44bc3db0abd6994ac733c868eac0728bacd5f1371b540a2a76674774e.jpg)  
Figure E4 Qualitative visualization of the $\mathcal { D } _ { \mathrm { g r i d } }$ subset.

![](images/f1aeed445dfbd8a7c494130f6acfe671da3917609ac5040b34762346dc035261.jpg)  
Figure E5 Qualitative visualization of the $\mathcal { D } _ { \mathrm { s h e l f } }$ subset.

![](images/c7a4ff7ef3023c62b282ac37aa3ab18c7b212137e0e589a4a06eb6f99ded09cb.jpg)  
Figure E6 Qualitative results of OmniDex model across diverse benchmark subsets.

matched mini-split, and SAM generated the target masks. As shown in Fig. E7, OmniDex conducted representative grasping trials on an apple, a mangosteen, and a cup in cluttered scenes.

![](images/cae04a51ed764e6564d43c65a721b561cc0eb1f931447cc6a2855a350cc7863a.jpg)  
Figure E7 Representative real-robot grasping trials in cluttered scenes.

## G Limitation

Despite the scale and diversity of the proposed benchmark, both the benchmark construction and the main evaluation are conducted in simulation. Although we incorporate diverse supporting bases, multi-view RGB-D observations, and physics-based validation to better approximate realistic cluttered grasping, a non-negligible sim-to-real gap may still remain due to unmodeled factors such as sensor noise, calibration error, contact parameter mismatch, and actuation uncertainty. While our seed-and-filter pipeline is substantially more scalable than scene-level optimization and our inference is more efficient than test-time optimization, the overall system still requires considerable computational resources for large-scale data generation and model training, constructing the full benchmark and training the model in volve substantial GPU cost. Addressing real-world transfer and further reducing the resource requirement are important directions for future work.
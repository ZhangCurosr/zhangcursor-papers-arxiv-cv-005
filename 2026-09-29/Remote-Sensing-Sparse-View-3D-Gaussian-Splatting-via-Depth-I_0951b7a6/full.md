# Remote Sensing Sparse-View 3D Gaussian Splatting via Depth Image-Based Rendering

Jiaming Kang, Zhengxia Zou, and Zhenwei Shi<sup>⋆</sup> Beihang University

Abstract—Remote sensing novel view synthesis under sparse observations remains challenging due to insufficient geometric constraints and limited cross-view supervision. Existing Neural Radiance Fields (NeRF) and 3D Gaussian Splatting (3DGS) methods are prone to overfitting and face challenges of depth ambiguities, missing cross-view information, and insufficient constraints in under-observed regions. To address these challenges, we propose DIBR-GS, a neural Gaussian Splatting framework that exploits Depth Image-Based Rendering (DIBR) to generate pseudo views for cross-view consistency supervision. Specifically, reliable geometric initialization is constructed by aligning monocular depth priors with sparse SfM reconstruction, and crossview appearance priors are incorporated into neural Gaussian representations to enhance appearance modeling under sparse observations. Furthermore, we introduce a progressive DIBRbased pseudo-view supervision strategy to provide additional geometric and appearance constraints, enabling more complete reconstruction of weakly observed regions. In addition, a heightconstrained anchor growth strategy is designed to suppress unreasonable Gaussian expansion. Experiments demonstrate that the proposed method achieves superior performance over existing approaches when training with only 3 input views. Compared with the previous best-performing method, it improves PSNR by 6.83 dB, with relative gains of 14% in SSIM and 60% in LPIPS, while maintaining competitive computational efficiency. Our code is available at https://github.com/kanehub/DIBR-GS

Index Terms—Sparse-view, novel view synthesis, remote sensing, Gaussian Splatting, depth image-based rendering.

## I. INTRODUCTION

Novel view synthesis aims to render images at arbitrary new viewpoints, given a set of images and their camera poses. In remote sensing, novel view synthesis provides a complementary way to enhance 3D scene understanding beyond conventional observations [1]–[5]. Consequently, it plays a crucial role in applications such as urban 3D reconstruction, disaster assessment, and environmental monitoring [6]–[9].

With the rapid development of deep learning, neural rendering has become the mainstream for novel view synthesis [10], [11]. Neural Radiance Fields (NeRF) [12]–[15] employs an implicit neural network to continuously model scene geometry and appearance, performing well in terms of view consistency and detail representation. However, its reliance on dense per-ray sampling incurs high computational costs. Conversely, 3D Gaussian Splatting (3DGS) [16]–[19] has attracted widespread attention due to its real-time rendering capabilities. By optimizing explicit Gaussian primitives and leveraging tile-based rasterization, it significantly reduces both training and inference overhead while maintaining high-quality rendering results.

Despite these advances, existing neural rendering methods still heavily rely on dense multi-view observations. Under sparse-view conditions, the lack of sufficient multi-view constraints often leads to overfitting. This challenge becomes even more prominent in remote sensing scenes. Due to limitations such as satellite revisit cycles, drone flight paths, and occlusions, it is difficult to capture dense multi-view images. In extreme cases, only 3 to 5 viewpoints are available, which is far fewer than the hundreds of densely sampled multi-view inputs typically used. This observation scarcity leaves large regions of the viewpoint space weakly constrained, resulting in ambiguous geometry, incomplete structures, and inconsistent appearance during novel-view synthesis.

![](images/2b3139a271eaf4dbf69995dc6e89b5d2b0a219f6bb008107812aca89ae7eccc6.jpg)  
Fig. 1. Visual and quantitative comparison on the LEVIR-NVS dataset with 3 input views. The proposed method produces more complete structures and finer details compared with existing approaches, especially in weakly observed regions. The radar chart summarizes the performance comparison in terms of reconstruction quality and rendering efficiency. For visualization, we report the negative LPIPS, AVGE, and 1 + log(FPS) values to enable unified comparison.

For sparse-view inputs, NeRF-based methods typically rely on regularization [20]–[22], semantic priors [23], [24], or depth supervision [25], [26]. However, these strategies are mainly developed for object-centric scenes with relatively diverse viewpoint coverage, and their effectiveness remains limited for remote sensing scenes, which contain complex land-cover structures observed from constrained overhead viewpoints. Meanwhile, 3DGS-based approaches attempt to enhance sparse-view performance by incorporating depth priors [27]–[29], co-regularization [30], or structural pruning [31]. Nevertheless, depth ambiguities in remote sensing images can compromise reconstruction accuracy, and heuristic optimization strategies struggle to capture complex scene geometry and maintain cross-view consistency.

To this end, we identify the fundamental challenge of sparse-view novel view synthesis in remote sensing as how to effectively exploit various prior information, including geometry, appearance, and structural relationships, from limited observations. Existing methods are largely constrained by the unreliability and insufficient utilization of such priors, which can be manifested in three aspects. First, geometric priors derived from monocular depth estimation suffer from scale ambiguity and structural distortions in remote sensing images, making them unreliable for accurate supervision or initialization. Second, existing approaches lack effective mechanisms to jointly exploit appearance and geometric cues across multiple views, leading to inconsistent reconstruction of scene structures and textures. Third, the lack of effective supervision for unobserved regions, where observations are limited due to occlusions or viewpoint constraints, often leads to incomplete geometry recovery and degraded reconstruction quality.

To address the above challenges, we propose DIBR-GS, a geometry-guided neural Gaussian splatting framework. It enhances sparse-view novel view synthesis in remote sensing scenes by unifying initialization, feature representation, and unobserved-view constraints within a depth image-based rendering (DIBR) [32] framework, which effectively exploits cross-view geometric and appearance priors. Specifically, we first design a point cloud initialization strategy tailored for monocular depth estimation. An affine transformation is introduced to alleviate scale ambiguity of monocular depth, while multi-view fusion is employed to reduce geometric misalignment and obtain reliable initial anchors. Then, following the image-based rendering (IBR) [33] framework, crossview geometric and appearance features are incorporated into the feature representation of neural Gaussians, improving the utilization of limited prior information. More importantly, we introduce a progressive pseudo-view synthesis strategy via DIBR to propagate depth priors from input images to unobserved viewpoints, which effectively promotes multiview consistency during Gaussian optimization. To further improve robustness for near-nadir remote sensing scenes, we incorporate a height-constrained anchor growth strategy based on aerial scene priors to suppress floating artifacts.

Ultimately, our proposed method achieves high-fidelity novel view synthesis from sparse-view inputs in remote sensing scenes while preserving real-time rendering capability. Fig. 1 presents the visual and quantitative comparison results.

In summary, our main contributions are:

1) We propose DIBR-GS, which integrates depth-based image rendering with Gaussian Splatting to effectively exploit geometric and appearance priors for remote sensing sparse view synthesis.

2) We introduce a progressive pseudo view synthesis strategy that propagates depth priors from observed views to unobserved viewpoints, providing additional multi-view consistency constraints during Gaussian optimization. This strategy effectively alleviates incomplete geometry and structural degradation in under-observed regions.

3) Experiments on the LEVIR-NVS [1] dataset demonstrate that the proposed method consistently outperforms stateof-the-art approaches for remote sensing sparse view synthesis, achieving a favorable balance between novelview rendering quality and computational efficiency.

## II. RELATED WORK

## A. Neural Scene Representations

Neural scene representation refers to using neural networks to approximate the surface or volumetric representation function, and integrating classical rendering principles to achieve a continuous and compact 3D representation. Based on the underlying structural form, these methods can be categorized into implicit, explicit, and hybrid representations [11]. Neural Radiance Fields [12], [34] serves as a classical implicit approach, which maps spatial coordinates to color and volume density via a multilayer perceptron, enabling high-fidelity novel view synthesis. However, the training efficiency is limited due to the dense sampling and per-ray inference. Hybrid representations [35]–[37] incorporate explicit geometry to reduce computational costs, while maintaining modeling capabilities through lightweight decoding networks. Plenoxels [36] store spherical harmonic coefficients and volume density within a sparse voxel grid to enable efficient queries. Point-NeRF [38] embeds neural features within point clouds, which accelerates convergence and provides explicit editing capabilities. Alternative methods encode scene information using multi-plane images [39], [40] or triplane representations [41] to avoid the redundancy associated with volumetric approaches.

Recently, explicit representations such as 3D Gaussian Splatting [16] represent scenes with anisotropic Gaussian primitives and rasterization, achieving high-fidelity reconstruction and real-time rendering. 2DGS [17] introduces surface Gaussian primitives and constrains them to align closely with object surfaces to improve geometric accuracy. Scaffold-GS [42] introduces an anchor-based hierarchical structure that dynamically generates and organizes neural Gaussians utilizing a sparse voxel grid, addressing issues of structural redundancy and loose organization. Although these approaches resolve the trade-offs among rendering speed, geometric quality, and storage efficiency, the training process still requires massive amounts of multi-view images to ensure convergence.

## B. Novel View Synthesis with Sparse Views

Novel view synthesis aims to render images from unobserved viewpoints based on input images and their corresponding poses. Traditional methods require hundreds of views to provide dense supervision, which is impractical for remote sensing scenarios because data acquisition is often constrained by flight path or fuel and collected images are typically sparse. Current research primarily improves sparse-view performance on object-level and synthetic scenes. Early methods introduce regularization constraints to mitigate overfitting in neural radiance fields under sparse views [20]–[23]. For instance, RegNeRF [20] employs local depth smoothness to eliminate irregular artifacts. DietNeRF [23] enforces cross-view semantic consistency. FreeNeRF [21] designs a frequency regularization strategy to achieve progressive learning from low to high frequencies. DS-NeRF [25] utilizes sparse point clouds as depth supervision, providing effective explicit constraints for scene geometry. Other approaches [33], [43], [44] explore cross-scene pretraining strategies to learn general scene representations and view synthesis priors from large-scale data, enhancing few-shot performance and generalization.

![](images/bf0d25af5f9a9acee590a5da8f2f82d459728ddd4d09a281ebce4f8016a15539.jpg)  
Fig. 2. Overview of the proposed DIBR-GS framework. The framework consists of a DIBR branch and a neural Gaussian branch. In the DIBR branch monocular depth priors are aligned with sparse SfM reconstruction through affine transformation and multi-view fusion to construct a reliable geometric initialization. Cross-view appearance features are extracted following the image-based rendering paradigm, and progressive DIBR is further employed t generate pseudo views that provide additional supervision. In the neural Gaussian branch, Gaussian anchors are initialized from the dense point cloud, while the anchor features and cross-view appearance features are adaptively integrated to predict Gaussian attributes. The rendered training views and pseudo view are jointly optimized, enabling more complete and consistent reconstruction under sparse-view remote sensing scenes.

3DGS also suffers performance degradation under sparse viewpoints. To address this, some methods incorporate depth priors as auxiliary supervision [27]–[29]. FSGS [27] guides the initialization and densification of Gaussian primitives using monocular depth. By distributing Gaussian points more appropriately in space, FSGS compensates for holes in sparse initial SfM point clouds, enabling coherent scene reconstruction even with very few samples. Similarly, DNGaussian [28] employs hard and soft depth regularization losses to enforce consistency between predicted depth and monocular depth, resolving common depth ambiguities under sparse views. Other methods focus on regularization strategies to suppress artifacts and optimize Gaussian distributions [30], [31], [45], [46]. DropGaussian [31] identifies and removes redundant primitives that cause visual artifacts by analyzing cumulative opacity and spatial distribution, resulting in a more compact scene representation. CoR-GS [30] jointly trains two Gaussian fields and utilizes their inconsistencies to suppress inaccurate reconstructions, improving view synthesis quality. Nevertheless, these methods still face challenges in remote sensing scenarios. Monocular depth exhibits inherent scale ambiguity, making depth constraints unreliable. Moreover, manually designed heuristic constraints struggle to handle the complex structures present in remote sensing scenes.

## III. METHOD

We propose DIBR-GS, a geometry-guided neural Gaussian framework. Inspired by the DIBR, we explore how to effectively leverage reliable geometric and appearance priors through three key stages: geometry initialization, feature representation, and optimization constraints. The overall framework is illustrated in Fig. 2. First, monocular depth priors are aligned with SfM points and fused across multiple views to construct reliable initial Gaussian anchors, reducing geometric ambiguity under sparse inputs. Then, cross-view reference features are incorporated into neural Gaussian representations to provide complementary appearance cues for sparse-view optimization. Finally, unlike conventional DIBR that aims at producing visually complete images, we utilize progressive DIBR to establish additional supervision in unobserved regions and viewpoints. The generated pseudo views implicitly incorporate geometric information derived from depth priors and enhance multiview consistency during optimization. Besides, considering the inherent height distribution prior of remote sensing scenes, we further introduce a height-constrained anchor growth strategy to constrain unreasonable Gaussian expansion.

## A. Preliminary

1) 3DGS: 3DGS [16] represents a scene using a set of anisotropic Gaussian primitives. Given the mean $\pmb { \mu } \in \mathbb { R } ^ { 3 }$ and

covariance matrix $\pmb { \Sigma } \in \mathbb { R } ^ { 3 \times 3 }$ , each Gaussian is defined as

$$
G ( \mathbf { x } ) = e ^ { - \frac { 1 } { 2 } ( \mathbf { x } - \pmb { \mu } ) ^ { T } \pmb { \Sigma } ^ { - 1 } ( \mathbf { x } - \pmb { \mu } ) } .\tag{1}
$$

To ensure positive semi-definiteness, the covariance matrix is parameterized as $\mathbf { \Sigma } \mathbf { \Sigma } = \mathbf { \Psi } \mathbf { R } \mathbf { S } \mathbf { S } ^ { T } \mathbf { R } ^ { T }$ , where R and $\mathbf { S }$ are represented by a quaternion $\mathbf { q _ { \lambda } } \in \mathbb { R } ^ { 4 }$ and a scaling vector $\mathbf { s } \in \mathbb { R } ^ { 3 }$ , respectively. Each Gaussian is further associated with a color c and opacity $\alpha .$

During rendering, 3D Gaussians are projected onto the image plane and blended in depth order through differentiable rasterization. The pixel color is computed by

$$
C = \sum _ { i = 1 } ^ { N } \mathbf { c } _ { i } \sigma _ { i } \prod _ { j = 1 } ^ { i - 1 } ( 1 - \sigma _ { j } ) , \quad \sigma _ { i } = \alpha _ { i } G _ { i } ^ { \prime } ( \mathbf { x } ) ,\tag{2}
$$

2) Neural Gaussian: Scaffold-GS [42] introduces anchorbased neural Gaussians for compact scene representation. Each anchor controls k Gaussians, whose centers are generated by

$$
\left\{ \pmb { \mu } _ { 1 } , \dots , \pmb { \mu } _ { k } \right\} = \mathbf { x } _ { v } + \left\{ \pmb { \mathcal { O } } _ { 1 } , \dots , \pmb { \mathcal { O } } _ { k } \right\} \cdot \boldsymbol { l } _ { v } ,\tag{3}
$$

where $\mathbf { x } _ { v }$ is the anchor position, $l _ { v }$ is a scaling factor, and $\mathcal { O } _ { k } \in \mathbb { R } ^ { 3 }$ denotes the learned offset.

The Gaussian attributes $\{ \alpha _ { i } , \mathbf { q } _ { i } , \mathbf { s } _ { i } , \mathbf { c } _ { i } \}$ are decoded from the anchor feature $\mathbf { f } _ { s } \in \mathbb { R } ^ { 3 2 }$ , the viewing direction $\mathbf { d } _ { v c } .$ , and the camera-relative distance $\delta _ { v c }$ using lightweight MLPs. For example, opacity is predicted as

$$
\{ \alpha _ { 1 } , \dots , \alpha _ { k } \} = F _ { \alpha } ( \mathbf { f } _ { s } , \delta _ { v c } , \mathbf { d } _ { v c } ) ,\tag{4}
$$

while quaternion, scale, and color are generated by $F _ { q } , F _ { s }$ and $F _ { c } ,$ respectively.

During training, anchors are adaptively grown according to Gaussian gradients and pruned when their opacity remains low, enabling efficient coverage of scene structures.

## B. Initialization with Dense Depth Priors

Point cloud initialization plays an important role in sparseview 3D Gaussian optimization, as inaccurate geometric initialization may lead to incomplete structures and unstable optimization. However, SfM reconstruction from extremely sparse views usually produces insufficient points, limiting the ability to capture scene structures. To alleviate this issue, we exploit monocular depth estimation as an auxiliary geometric prior and align it with SfM geometry to construct a denser and more reliable initialization. Given an input image I, the original monocular depth map can be formulated as $\mathbf { D } = F _ { \theta } ( \mathbf { I } )$ . Since monocular depth estimation only provides relative depth, we align it with sparse SfM points through an affine transformation:

$$
\hat { \mathbf { D } } = a \cdot \mathbf { D } + b .\tag{5}
$$

The scaling and translation coefficients a, b are solved using the least squares method:

$$
\hat { a } , \hat { b } = \arg \operatorname* { m i n } _ { a , b } \sum _ { p \in \Omega _ { \mathrm { s f m } } } \left( \hat { \mathbf { D } } ( p ; a , b ) - \mathbf { D } _ { \mathrm { s f m } } ( p ) \right) ^ { 2 } ,\tag{6}
$$

where $\boldsymbol { p } = \left( u , v \right)$ represents the projected coordinates of the point clouds on the image, and $\bf { D } _ { \mathrm { { s f m } } }$ is the depth of the sparse SfM point clouds. Subsequently, we use the depth map to back-project image pixels into the 3D space of the world coordinate system:

$$
\mathbf { P } = \mathbf { R } _ { i } \left( \hat { \mathbf { D } } _ { i } ( u , v ) \mathbf { K } _ { i } ^ { - 1 } \mathbf { p } \right) + \mathbf { t } _ { i } ,\tag{7}
$$

where $\mathbf { P } = ( x , y , z , 1 ) ^ { \top }$ represents the homogeneous coordinates in the world coordinate system, $\mathbf { K } _ { i }$ denotes the intrinsics of view i, $\mathbf { R } _ { i }$ and $\mathbf { t } _ { i }$ represent the rotation and translation matrix in the extrinsics, and $\mathbf { p } ~ = ~ ( u , v , 1 ) ^ { \top }$ represents the homogeneous pixel coordinates.

After back-projecting depth maps from multiple views, we fuse the generated points to obtain an initial dense representation. However, due to inherent errors in monocular depth estimation, the fused point cloud often contains numerous mismatches. In remote sensing scenarios, such errors typically manifest as layered stacking of point clouds at different heights, which clearly contradicts reality. For nearnadir remote sensing scenes, dominant structures can often be approximated by a single surface distribution along the vertical direction. We therefore exploit this domain prior to identify obvious depth outliers while preserving the major scene structure.

Leveraging this prior knowledge, we can effectively identify and remove inaccurate point clouds.

We first partition the point clouds into grids on the XY plane at a fixed resolution, with each grid denoted as $G _ { k }$ . The height set of point clouds within each grid is denoted as:

$$
Z _ { k } = \{ z _ { i } | ( x _ { i } , y _ { i } ) \in G _ { k } \}\tag{8}
$$

We perform a dip test [47] on each grid. If the height distribution significantly deviates from the hypothesis of unimodality, we consider that the region may contain multiple vertically overlapping surfaces and unreliable points. To effectively evaluate the overlapping relationships, we construct a KD-tree [48] within the grid to identify neighboring point pairs. Two points $P _ { i }$ and $P _ { j }$ are considered overlapping if they satisfy the following criterion:

$$
\mathbb { I } _ { \mathrm { o v l p } } ( P _ { i } , P _ { j } ) = \left\{ { \begin{array} { l l } { 1 , } & { { \mathrm { i f ~ } } \| ( x _ { i } , y _ { i } ) - ( x _ { j } , y _ { j } ) \| _ { 2 } < \varepsilon _ { x y } } \\ & { \wedge | z _ { i } - z _ { j } | > \varepsilon _ { z } } \\ { 0 , } & { { \mathrm { o t h e r w i s e } } , } \end{array} } \right.\tag{9}
$$

where $\varepsilon _ { x y }$ and $\varepsilon _ { z }$ represent the thresholds for horizontal and vertical distances, respectively.

Previous studies [49] have revealed that sparse-view Gaussian optimization often suffers from a typical failure mode where Gaussians tend to accumulate toward the cameras, resulting in overfitting to input views. Therefore, when multiple redundant surfaces are generated during depth fusion, we retain the points with lower height values as initialization anchors. For near-nadir remote sensing scenes, lower-height points generally correspond to surfaces farther from the camera, which provides a more stable initialization and mitigates the camera-oriented drift of optimized Gaussians.

## C. IBR-Based Neural Gaussian Representation

To enhance modeling under sparse observations, we incorporate cross-view reference features into neural Gaussian representations. The proposed representation exploits complementary appearance cues from multiple observations following an image-based rendering paradigm. However, due to occlusions and inaccurate matches in sparse views, reference features may contain unreliable information. Therefore, we further introduce an adaptive modulation mechanism to control the contribution of cross-view features during Gaussian optimization.

1) IBR Feature Extraction: Given an input image I, we adopt a pretrained DINO [50] encoder to extract the local features pyramid $\{ \mathbf { F } _ { r } ^ { \ell } \} _ { \ell = 1 } ^ { L }$ and the ViT tokens $\mathbf { F } _ { t }$ that contain global semantics and long-range dependencies. We subsequently introduce a learnable projection module $\mathcal { P }$ to map both into the same dimension and fuse them into a single feature map, thereby modeling the complementary relationship between local texture and global semantics within a unified feature space:

$$
\tilde { \mathbf { F } } _ { r } ^ { \ell } = \mathrm { U p } ( \mathcal { P } _ { \ell } ( \mathbf { F } _ { r } ^ { \ell } ) ) ,\tag{10}
$$

$$
\tilde { \mathbf { F } } _ { t } = \operatorname { B r o a d c a s t } ( \mathcal { P } _ { t } ( \mathbf { F } _ { t } ) ) ,\tag{11}
$$

$$
\mathbf { F } _ { v } = \tilde { \mathbf { F } } _ { t } + \sum _ { \ell = 1 } ^ { L } \tilde { \mathbf { F } } _ { r } ^ { \ell } ,\tag{12}
$$

where $\mathrm { U p } ( \cdot )$ represents upsampling and Broadcast(·) expands patch-level features to pixel space by repeating.

For anchor point with center $\boldsymbol { \mu } _ { w } ~ = ~ ( x _ { w } , y _ { w } , z _ { w } ) ^ { \top }$ , its projection coordinate in the i-th view is given as:

$$
\begin{array} { r l r } & { \boldsymbol { \mu } _ { c } = \mathbf { E } _ { i } [ \boldsymbol { \mu } _ { w } ; 1 ] , } & \\ & { \mathbf { p } _ { i } = \displaystyle \frac { 1 } { z _ { c } } \mathbf { K } _ { i } \boldsymbol { \mu } _ { c } , } & \end{array}\tag{13}
$$

where ${ \pmb \mu } _ { c } ~ = ~ ( x _ { c } , y _ { c } , z _ { c } ) ^ { \top }$ and $\mathbf { E } _ { i }$ represents the extrinsic matrix from world to camera. Then we extract the feature vector from the feature map according to the projection coordinate:

$$
\mathbf { f } _ { v , i } = \Psi \left( \mathbf { F } _ { v } , \mathbf { p } _ { i } \right) ,\tag{14}
$$

where $\Psi ( \cdot )$ denotes bilinear sampling. By aggregating the sampled vectors across all views, we derive the final IBR feature for the anchor:

$$
\mathbf { f } _ { v } = \frac { 1 } { | \mathcal { V } | } \sum _ { i \in \mathcal { V } } \mathbf { f } _ { v , i } .\tag{15}
$$

2) Adaptive Feature Fusion: Given that the IBR features under sparse views are susceptible to occlusion or matching errors, the direct fusion of reference features is prone to introducing noise. To address this, we design a modulation network $\mathcal { E } _ { \mathrm { m o d } }$ that adaptively predicts channel-wise modulation weights to regulate the contribution of cross-view features:

$$
\begin{array} { r } { \mathbf { g } = \mathcal { E } _ { \mathrm { m o d } } ( \mathbf { f } _ { v } ) . } \end{array}\tag{16}
$$

The fused feature of the anchor point is expressed as:

$$
\hat { \mathbf { f } } _ { s } = \mathbf { f } _ { s } + \mathbf { g } \odot \mathcal { E } _ { \mathrm { r e f } } ( \mathbf { f } _ { v } ) ,\tag{17}
$$

where $\mathbf { f } _ { s }$ denotes the original neural Gaussian feature, ${ \mathcal E } _ { \mathrm { r e f } }$ represents the feature projection network, and ⊙ indicates channel-wise multiplication. Through the adaptive feature fusion strategy, information derived from IBR is fused into the neural Gaussian representation in a controlled and stable manner, reducing the influence of unreliable reference features while preserving useful prior information for sparse-view reconstruction.

## D. Height-constrained Anchor Growth

A typical failure mode in sparse-view novel view synthesis is the emergence of floating artifacts around cameras, which mainly originate from regions with insufficient observation constraints [21]. Due to the limited multi-view overlap, Gaussian optimization may explain uncertain regions by increasing primitive density near the observed viewpoints, resulting in unstable geometry and noticeable artifacts. Although depthbased regularization strategies [27], [28] have been explored to alleviate this issue, they usually require reliable depth supervision. In this work, we exploit the inherent characteristics of remote sensing scenes and introduce a lightweight geometryaware regularization strategy to constrain unreasonable anchor growth.

During optimization, the model computes the average gradient of neural Gaussians within each local spatial neighborhood, and regions with large gradients are identified as important areas where new anchors are added. Our method specifically aims to identify potential floater Gaussians and prevent wrong anchor growth in these regions. A key characteristic in remote sensing scenarios is that many aerial images are captured from a nadir viewpoint, where a point’s height is negatively correlated with its distance to the camera. This property allows us to detect Gaussians that are anomalously close to the camera based on their height. Therefore, we construct a coarse height map from the initial dense point cloud to identify Gaussians that may cause camera-oriented expansion. Instead of enforcing a strict geometric supervision, the height map serves as a soft criterion to prevent unreasonable anchor growth while preserving valid scene structures. Specifically, we partition the XY plane into grids and estimate a relaxed upper bound on height within each grid, which is used to suppress anchor growth only for Gaussians that clearly fall outside the valid height as shown in Fig. 3(b). The height map is formulated as:

$$
\mathcal { H } ( x , y ) = \rho \left( \{ z _ { n } \mid P _ { n } \in \mathcal { P } _ { i , j } , \left\lfloor \frac { x _ { n } } { s } \right\rfloor = i , \left\lfloor \frac { y _ { n } } { s } \right\rfloor = j \} \right)\tag{18}
$$

where $\rho ( \cdot )$ denotes the height aggregation operator, ⌊·⌋ indicates the floor operation, and s represents the grid size.

Based on this design, the new anchor growth criterion can be defined as:

$$
\mathbb { I } _ { \mathrm { g r o w } } ( P ) = \left\{ { \begin{array} { l l } { 1 , } & { { \mathrm { i f ~ } } \nabla _ { g } > \tau _ { g } \wedge z \leq \mathcal { H } ( x , y ) } \\ { 0 , } & { { \mathrm { o t h e r w i s e } } . } \end{array} } \right.\tag{19}
$$

![](images/c528ed8290d7efba7a86600d13b06ae906a2c73b98746cfacd2adceeebaa23ef.jpg)

![](images/a58ae698bd8d4223c3f4c52730b47300d0477857ac190a5a0c4e5a1f541771c7.jpg)  
Fig. 3. Illustration of anchor growth. (a) Anchor distribution after growth. The original anchors contain numerous floating artifacts and deviate significantly from the initial point cloud, while the Gaussians become more compact and precise with the constraint. (b) The proposed height-constrained anchor growth strategy, where only neural Gaussians whose heights fall within the predefined bounds are considered as candidate new anchors.

where $\nabla _ { g }$ denotes the averaged gradient and $\tau _ { g }$ is a pre-defined threshold.

The proposed regularization reduces floating artifacts and encourages a more compact Gaussian distribution, as illustrated in Fig. 3(a).

## E. Pseudo View Supervision through Progressive DIBR

Another critical challenge in sparse-view novel view synthesis is the lack of effective supervision for unobserved regions. Due to limited viewpoints and occlusions, some regions in novel views may not be covered by any input observations, resulting in insufficient Gaussian optimization and severe artifacts such as holes or incomplete structures. Prior studies [16], [38], [51] have shown that Gaussians struggle to model areas beyond the boundaries defined by the initial point cloud unless they are explicitly encouraged to expand outward. However, existing methods [51] cannot precisely identify true boundary regions and may introduce redundant primitives.

To address this issue, we propose to synthesize pseudo views through progressive DIBR to incorporate more prior knowledge from pretrained model. The synthesized pseudo views provide complementary supervision signals, encouraging Gaussian optimization to recover missing structures while maintaining multi-view consistency.

We sample intermediate viewpoints along the acquisition trajectory of the input views to generate additional observations between existing cameras. For each target view $j ,$ we obtain depth map $\mathbf { D } _ { j } ^ { \mathrm { p s e } }$ by performing depth completion on the dense point cloud. Given target view pixel $\mathbf { p } _ { j }$ , we compute the corresponding coordinate $\mathbf { p } _ { j  i }$ in the source view through depth warping:

$$
{ \bf p } _ { j  i } = { \bf K } _ { i } { \bf T } _ { j  i } { \bf D } _ { j } ^ { \mathrm { p s e } } ( u , v ) { \bf K } _ { j } ^ { - 1 } { \bf p } _ { j } ,\tag{20}
$$

where $\mathbf { T } _ { j  i }$ denotes the transformation matrix from view $j$ to view i. Then we sample from the source-view images $\mathbf { I } _ { i } ^ { \mathrm { g t } }$ to synthesize the pseudo-view image $\mathbf { I } _ { j } ^ { \mathrm { p s e } }$ :

$$
\mathbf { I } _ { j } ^ { \mathrm { p s e } } ( \mathbf { p } _ { j } ) = \mathrm { S a m p l e } ( \mathbf { I } _ { i } ^ { \mathrm { g t } } ; \mathbf { p } _ { j  i } ) ,\tag{21}
$$

where Sample(·) denotes the pixel sampling operation. Due to spatial occlusions, it is not possible to obtain all pixels of the target view in a single step. We therefore design a progressive synthesis strategy that integrates images from multiple input views. Specifically, we compute the relative distances between the target view and all input views and sort them accordingly. The pseudo view is filled sequentially from near to far using depth-based warping, and any remaining unfilled areas are finally completed using depth warping with padding. Fig. 9 illustrates the detailed synthesis process. This progressive strategy follows the intuition that nearby source views generally provide more reliable correspondence with fewer occlusions. By gradually integrating complementary observations, the synthesized views reduce artifacts caused by unreliable warping.

## F. Optimization

We adopt an end-to-end optimization strategy that jointly trains the neural Gaussian representation and its associated modules. The overall objective is formulated as a pixel-wise reconstruction loss. During training, we jointly optimize three groups of parameters: (i) the learnable attributes of anchor Gaussians, (ii) the neural Gaussian decoder MLP, and (iii) the projection and fusion networks for IBR features. Besides, we adopt a scheduled supervision strategy. During the early training stage, only ground truth images guide the model to reconstruct the global scene structure and establish stable geometry and appearance. Then pseudo-view supervision is introduced periodically to enforce multi-view consistency and eliminate holes. Finally, pseudo-view supervision is disabled to focus on optimizing fine-grained details.

The final loss function can be expressed as:

$$
\begin{array} { r } { \mathcal { L } = ( 1 - \lambda ) \mathcal { L } _ { 1 } ( \hat { \bf I } , { \bf I } ) + \lambda \mathcal { L } _ { \mathrm { D - S S I M } } ( \hat { \bf I } , { \bf I } ) , } \end{array}\tag{22}
$$

where <sup>ˆ</sup>I is the rendered image and I is the corresponding ground-truth or pseudo-view image. We set $\lambda = 0 . 2$ in our experiments.

## IV. EXPERIMENT

## A. Experimental Setup

1) Dataset: We evaluate our method on LEVIR-NVS [1], a public remote sensing NVS dataset. It comprises 16 scenes derived from Google Earth, simulating aerial imaging viewpoints and covering diverse land-cover types, including mountainous terrain, urban areas, villages, and building complexes. Each scene contains 21 multi-view images at a resolution of $5 1 2 \times 5 1 2$ . To evaluate performance under sparse-view settings, we adopt the data split protocol of TriDF [52], uniformly selecting 3 images per scene for training and using the remaining views for evaluation.

2) Baseline and Metrics: We compare our proposed method with state-of-the-art approaches, including NeRF-based methods such as RegNeRF [20] and FreeNeRF [21], as well as 3DGS–based methods including FSGS [27], DropGaussian [31], CoR-GS [30] and the vanilla 3DGS [16]. In addition, we compare against advanced methods specifically designed for remote sensing scenarios, including MPNeRF [4] and TriDF [52]. For all compared methods, we follow the same experimental settings and training protocols to ensure a fair comparison. Following prior work, we report PSNR, SSIM, LPIPS and AVGE metric [20] to evaluate the rendering quality.

![](images/cade227752b096982ab90b89c45499271e96ab53ac0ad364a557f9b256ba1d66.jpg)  
TriDF

![](images/01dcb8d5ac8c24b6346cb1ea535ec8c45a0ad28de68dda1f915d8952a7b80038.jpg)  
FSGS

![](images/198435f4ee8f5e8457350ef9f06a878519dcfa7bcf5255a93ddd7e3e612544ef.jpg)  
CoR-GS  
Fig. 4. Visual comparison on LEVIR-NVS Dataset.

## B. Implementation Details

1) Architecture Details: We use the frozen Depth Anythingv2 [53] model to extract monocular depth maps from the training images, and then invert the disparity to obtain depth. Next, we perform affine transformation using the sparse point clouds estimated by COLMAP [54] to obtain dense depth maps with real-world scale. To eliminate overlapping regions in the point cloud, we reproject image pixels into 3D space in the world coordinate system and uniformly partition the XY plane into grids, where we compute the height distribution of the point cloud within each grid. When the significance level of the dip test is below 0.05, we consider the grid to contain overlap. The horizontal and vertical distance thresholds $\varepsilon _ { x y }$ and $\varepsilon _ { z }$ for redundant point pairs are set to 0.1 and 1.0, respectively. After obtaining the non-overlapping dense point cloud, we perform voxel downsampling with a voxel size of 1.0, and then merge it with the point cloud acquired via SfM for the initialization of neural Gaussians.

![](images/f05b87c1a3d9daeb8d98a10b821c489677ee149ef4d78cf0e77449d915ffb6c2.jpg)

![](images/09fbddf5608331e30a044c06aad3b84da7e8d5702e29ea265ff9abb785961866.jpg)  
DropGaussian  
Ours

![](images/32655fa29c1b93e3c186eb3e7421227e7a6892525bfa34b891c51dd697e65a9f.jpg)  
GT

To extract reference features, we employ frozen ResNet-50 [55] and ViT-B/8 vision transformer [56], both pretrained with a DINO [50] objective. We extract feature maps from the first four layers of ResNet-50 and project them into a unified feature space using 1×1 convolutions, followed by upsampling to the size of $\frac { H } { 2 } \times \frac { W } { 2 } \times C _ { r }$ . The output of ViT includes both global and local tokens, which are projected into the same feature space using linear layers. To ensure consistency with the size of the ResNet feature maps, these tokens are replicated spatially. Finally, all feature maps are summed together to generate the reference feature map with a dimension $C _ { r } = 3 2 .$

For the neural Gaussians, the grid size of the anchors is set to 0.03, with each anchor corresponding to $k ~ = ~ 1 0$ neural Gaussians. The modulation network $\mathcal { E } _ { \mathrm { m o d } }$ consists of linear layers followed by a Sigmoid activation function, with initialization settings designed to produce small output values, thereby stabilizing the optimization process. The dimension of the anchor features is set to 32.

We construct a height map using the dense point cloud to constrain the growth of neural Gaussians and suppress artifacts. The grid size for partitioning the XY plane is set to 10. We compute the 95th percentile of the height within each grid, which is then multiplied by a scaling factor to relax the boundary. To avoid disrupting fine structural details, we apply the height constraint only at the lowest resolution of the anchor voxels. To avoid excessive deviation from the training viewpoints, we randomly sample pseudo-view camera poses from the given circular acquisition trajectory.

![](images/df7f8bc618498c2dc3ad80bcac5a435fb23453bc2ca9326af505195efff0c57b.jpg)  
w/o Dense Init.

![](images/5b283db14caaa3d4c28d6cc8c45bc7050df279d646f90f74c4ef2602cfb7d47d.jpg)  
w/o IBR Feature

![](images/b375cf41c921ab6e1e73a3805d6f4ba1739c8394f42a33df682ef790bfe48d58.jpg)  
w/o Height Map

Fig. 5. Visualization of rendering quality when removing different model components.  
TABLE I  
QUANTITATIVE COMPARISON OF RENDERING QUALITY AND EFFICIENCY BETWEEN DIFFERENT METHODS. THE BEST, SECOND-BEST, AND THIRD-BEST ENTRIES ARE MARKED IN , AND RESPECTIVELY.
<table><tr><td rowspan=1 colspan=1>Method</td><td rowspan=1 colspan=5>PSNR↑ SSIM↑ LPIPS↓  AVGE↓  FPS↑</td></tr><tr><td rowspan=1 colspan=1>RegNeRF [20]</td><td rowspan=1 colspan=1>19.83</td><td rowspan=1 colspan=1>0.695</td><td rowspan=1 colspan=1>0.389</td><td rowspan=1 colspan=1>0.131</td><td rowspan=1 colspan=1>0.19</td></tr><tr><td rowspan=1 colspan=1>FreeNeRF [21]</td><td rowspan=1 colspan=1>19.04</td><td rowspan=1 colspan=1>0.524</td><td rowspan=1 colspan=1>0.373</td><td rowspan=1 colspan=1>0.148</td><td rowspan=1 colspan=1>0.19</td></tr><tr><td rowspan=1 colspan=1>MPNeRF [4]</td><td rowspan=1 colspan=1>21.72</td><td rowspan=1 colspan=1>0.800</td><td rowspan=1 colspan=1>0.190</td><td rowspan=1 colspan=1>0.083</td><td rowspan=1 colspan=1>0.18</td></tr><tr><td rowspan=1 colspan=1>TriDF [52]</td><td rowspan=1 colspan=1>24.07</td><td rowspan=1 colspan=1>0.820</td><td rowspan=1 colspan=1>0.213</td><td rowspan=1 colspan=1>0.071</td><td rowspan=1 colspan=1>0.20</td></tr><tr><td rowspan=1 colspan=1>3DGS [16]</td><td rowspan=1 colspan=1>18.29</td><td rowspan=1 colspan=1>0.593</td><td rowspan=1 colspan=1>0.313</td><td rowspan=1 colspan=1>0.144</td><td rowspan=1 colspan=1>280</td></tr><tr><td rowspan=1 colspan=1>FSGS [27]</td><td rowspan=1 colspan=1>21.18</td><td rowspan=1 colspan=1>0.772</td><td rowspan=1 colspan=1>0.230</td><td rowspan=1 colspan=1>0.094</td><td rowspan=1 colspan=1>343</td></tr><tr><td rowspan=1 colspan=1>CoR-GS [30]</td><td rowspan=1 colspan=1>22.39</td><td rowspan=1 colspan=1>0.793</td><td rowspan=1 colspan=1>0.212</td><td rowspan=1 colspan=1>0.082</td><td rowspan=1 colspan=1>268</td></tr><tr><td rowspan=1 colspan=1>DropGaussian [31]</td><td rowspan=1 colspan=1>21.89</td><td rowspan=1 colspan=1>0.793</td><td rowspan=1 colspan=1>0.201</td><td rowspan=1 colspan=1>0.084</td><td rowspan=1 colspan=1>282</td></tr><tr><td rowspan=1 colspan=1>Ours</td><td rowspan=1 colspan=1>30.90</td><td rowspan=1 colspan=1>0.938</td><td rowspan=1 colspan=1>0.076</td><td rowspan=1 colspan=1>0.025</td><td rowspan=1 colspan=1>177</td></tr></table>

2) Training Details: The neural Gaussian anchor attributes $\mathcal { O } _ { k } , ~ l _ { v } .$ , decoder MLP $F _ { \{ \alpha , q , s , c \} }$ , feature projection network ${ \mathcal E } _ { \mathrm { r e f } }$ , and modulation network $\mathcal { E } _ { \mathrm { m o d } }$ are jointly optimized in an end-to-end manner. We use the Adam optimizer, with different initial learning rates assigned to different network modules, and apply a continuous exponential learning rate scheduling strategy. The total number of training iterations is 30,000. The projection module P associated with the ResNet and ViT features is trained jointly with the neural Gaussian model using the AdamW optimizer, with an initial learning rate of $1 \times 1 0 ^ { - \hat { 4 } }$ and its parameters are frozen after 20,000 iterations. Neural Gaussian densification is performed every 100 iterations from iteration 1,500 to 15,000. In addition, pseudo-view sampling is introduced for training once every 50 iterations from iteration 3,000 to 20,000. All experiments are conducted on a single Nvidia GeForce RTX 4090 GPU.

## C. Comparison with Other Methods

The quantitative results on the LEVIR-NVS dataset are reported in Table I. Experimental results demonstrate that our method achieves superior reconstruction quality over all competing approaches, encompassing both advanced remotesensing methods and sparse-view 3DGS-based methods. We also report the per-scene experimental results, as shown in Table II. These results demonstrate that our approach delivers consistently stable, high-quality rendering across scenes of different types and complexities.

![](images/fec33b3aa299d8b68e2a5d47a83476deadf133709411621fd52f223ac57a273d.jpg)  
w/o Pseudo Views

![](images/df504cb551da7a50f8d7f99d917d42a75852565526c483de8904b4f75859e038.jpg)  
Ours

![](images/31f247d100ca90f49f062c747095de69e1273f5326d92d014f9a4f194fdac4d7.jpg)  
Ground Truth

Fig. 4 presents visual comparisons on several representative scenes. Due to the inherent complexity of remote sensing scenes and the extremely sparse input views, existing methods often suffer from incomplete structures, blurred boundaries, and inconsistent details. While TriDF preserves the overall scene structure, it still suffers from blurred textures and insufficient detail recovery in local regions. FSGS can reconstruct coarse geometric structures but shows noticeable artifacts such as ghosting and shape distortion in challenging areas. CoR-GS and DropGaussian alleviate certain artifacts but still exhibit issues including blurred edges and missing details. In contrast, the proposed method consistently produces sharper structures and recovers more textural details across all scenes. This improvement benefits from the unified exploitation of geometric and appearance priors. The geometry-aware initialization with dense depth provides more stable anchors for Gaussian optimization, while cross-view IBR features complement scene representation under limited observations. More importantly, the progressive DIBR-based supervision introduces additional geometric and appearance constraints, enabling the model to recover structures in regions with insufficient observations. As highlighted in the zoomed-in regions, the proposed method maintains more consistent structures and appearance in challenging areas such as image boundaries. Furthermore, it reduces floating artifacts and unreasonable geometric expansion observed in other methods. This demonstrates that the proposed DIBR-based supervision effectively improves multiview consistency, while the height-constrained anchor growth strategy further stabilizes Gaussian growth. Together, they enhance both structural completeness and rendering stability under sparse-view remote sensing scenes.

TABLE II  
QUANTITATIVE COMPARISON PER SCENE ON LEVIR-NVS DATASET.
<table><tr><td>Metrics</td><td>Methods</td><td>Building#1</td><td>Church</td><td>College</td><td>Mountain#1</td><td>Mountain#2</td><td>Observation</td><td>Building#2</td><td>Town#1</td><td>Stadium</td><td>Town#2</td><td>Mountain#3</td><td>Town#3</td><td>Factory Park</td><td>School</td><td>Downtown</td><td>Mean</td></tr><tr><td rowspan="8">PSNR SSIM</td><td>RegNeRF [20]</td><td>13.52</td><td>14.74</td><td>17.63</td><td>20.88</td><td>20.57</td><td>15.24</td><td>14.81 20.84</td><td>22.35</td><td>21.15</td><td>26.50</td><td>21.20</td><td>23.49</td><td>24.74</td><td>20.11</td><td>19.46</td><td>19.83</td></tr><tr><td>FreeNeRF [21]</td><td>15.84</td><td>15.99</td><td>19.74</td><td>19.64</td><td>21.41</td><td>13.76</td><td>20.58</td><td>23.41</td><td>14.66</td><td>20.56</td><td>19.05</td><td>21.03</td><td>23.84</td><td>21.81</td><td>17.46</td><td>19.04</td></tr><tr><td>MPNeRF [4]</td><td>18.81</td><td>17.93</td><td>20.71</td><td>25.50</td><td>24.92</td><td>19.56</td><td>15.85 18.64</td><td>22.08</td><td>21.20</td><td>28.57</td><td>21.41</td><td>22.61</td><td>23.57</td><td>20.71</td><td></td><td></td></tr><tr><td>TriDF [52]</td><td>21.42</td><td>19.56</td><td>22.98</td><td>27.70</td><td>27.05</td><td></td><td></td><td>21.59 25.45</td><td>22.36</td><td>30.70</td><td></td><td></td><td>27.62</td><td>24.02</td><td>19.73</td><td>21.72</td></tr><tr><td>3DGS [16]</td><td>16.27</td><td></td><td>16.27</td><td></td><td></td><td>20.70</td><td>20.74</td><td>24.47</td><td>17.22</td><td></td><td>22.86</td><td>25.28</td><td></td><td>16.49</td><td>22.19 16.04</td><td>24.07</td></tr><tr><td></td><td>17.69</td><td>15.16 17.52</td><td>20.01</td><td>20.16</td><td>21.58 24.39</td><td>16.68</td><td>16.73</td><td>18.54</td><td>18.46</td><td>24.21</td><td>17.93</td><td>20.25</td><td>20.64</td><td></td><td></td><td>18.29</td></tr><tr><td>FSGS [27]</td><td>18.39</td><td>18.13</td><td>20.57</td><td>25.03 26.30</td><td>25.68</td><td>19.76 20.44</td><td>18.36 20.09</td><td>20.74 22.64</td><td>23.42</td><td>19.92 26.46</td><td>21.11</td><td>22.47</td><td>23.93</td><td>18.42</td><td>19.63</td><td>21.18</td></tr><tr><td>CoR-GS [30] DropGaussian [31]</td><td>17.94</td><td>17.67</td><td>20.78</td><td>26.01</td><td>25.53</td><td></td><td></td><td>23.83</td><td>21.58 20.75</td><td>29.41 29.22</td><td>21.26 20.91</td><td>23.72 23.10</td><td>24.37 23.85</td><td>21.32 20.48</td><td>20.49 19.87</td><td>22.39</td></tr><tr><td>Ours</td><td>27.38</td><td>26.41</td><td>29.75</td><td>33.96</td><td>34.50</td><td>20.25 26.83</td><td>19.08 29.30</td><td>21.64 31.22</td><td>23.30 31.07</td><td>29.71</td><td>36.83</td><td>31.04</td><td>34.24 32.97</td><td>29.37</td><td>29.89</td><td>21.90 30.90</td></tr><tr><td rowspan="8"></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td>0.781</td><td>0.897</td><td></td><td></td><td>0.745</td></tr><tr><td>RegNeRF [20]</td><td>0.390 0.441</td><td>0.495 0.580 0.482</td><td>0.680</td><td>0.705</td><td>0.457</td><td>0.410</td><td>0.756</td><td>0.844</td><td>0.836</td><td>0.851</td><td></td><td></td><td>0.884</td><td>0.782</td><td>0.695</td></tr><tr><td>FreeNeRF [21]</td><td>0.730</td><td>0.541</td><td>0.414</td><td>0.511</td><td>0.140</td><td>0.378</td><td>0.627</td><td>0.788</td><td>0.331</td><td>0.294</td><td>0.643</td><td>0.714</td><td>0.810</td><td>0.692 0.584</td><td>0.524</td></tr><tr><td>MPNeRF [4]]</td><td>0.825</td><td>0.720 0.790 0.693 0.759</td><td>0.820 0.796</td><td>0.810 0.839</td><td>0.730</td><td>0.710</td><td>0.810</td><td>0.800</td><td>0.840</td><td>0.890</td><td>0.840</td><td>0.860</td><td>0.850 0.790</td><td>0.760</td><td>0.800</td></tr><tr><td>TriDF [52]</td><td>0.622</td><td>0.385</td><td>0.447</td><td>0.687</td><td>0.660</td><td>0.756</td><td>0.847</td><td>0.882</td><td>0.848</td><td>0.886</td><td>0.822</td><td>0.913 0.918</td><td>0.857</td><td>0.813</td><td>0.820</td></tr><tr><td>3DGS [16]</td><td>0.696</td><td>0.460 0.642 0.682</td><td></td><td></td><td>0.479</td><td>0.521</td><td>0.626</td><td>0.646</td><td>0.617</td><td>0.694</td><td>0.606</td><td>0.857</td><td>0.846</td><td>0.478 0.524</td><td>0.593</td></tr><tr><td>FSGS [27]</td><td>0.707</td><td>0.693</td><td>0.813 0.825</td><td>0.840 0.843</td><td>0.702 0.705</td><td>0.703 0.749</td><td>0.795 0.810</td><td>0.837 0.857</td><td>0.798</td><td>0.864</td><td>0.806</td><td>0.870</td><td>0.863 0.708</td><td>0.732 0.792</td><td>0.772</td></tr><tr><td>CoR-GS [30] DropGaussian [31]</td><td>0.718</td><td>0.665 0.668 0.707</td><td>0.834</td><td>0.852</td><td>0.712</td><td>0.739</td><td>0.814</td><td>0.846</td><td>0.824 0.819</td><td>0.883 0.882</td><td>0.816 0.807</td><td>0.885 0.874</td><td>0.860 0.871</td><td>0.781 0.774</td><td>0.793 0.793</td></tr><tr><td>Ours</td><td></td><td>0.893</td><td>0.914</td><td>0.931</td><td>0.953</td><td>0.886</td><td>0.940</td><td>0.949</td><td>0.949 0.942</td><td>0.955</td><td>0.947</td><td>0.976</td><td>0.968</td><td>0.934</td><td>0.776 0.942</td><td>0.938</td></tr><tr><td rowspan="8"></td><td></td><td>0.935</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>RegNeRF [20]</td><td>0.568 0.445</td><td>0.504 0.482 0.398 0.354</td><td>0.468 0.417</td><td>0.445</td><td>0.509</td><td>0.554</td><td>0.324</td><td>0.268 0.221</td><td>0.261</td><td>0.343</td><td>0.333 0.218</td><td>0.242</td><td>0.364 0.305</td><td>0.379 0.386</td><td>0.389</td></tr><tr><td>FreeNeRF [21]</td><td>0.210</td><td>0.240 0.180</td><td>0.200</td><td>0.401 0.180</td><td>0.543 0.250</td><td>0.439 0.240</td><td>0.318 0.180</td><td>0.200</td><td>0.454</td><td>0.494</td><td>0.299</td><td>0.257 0.233 0.120 0.140</td><td>0.200</td><td>0.190</td><td>0.373</td></tr><tr><td>MPNeRF</td><td>0.188</td><td>0.322 0.264</td><td>0.268</td><td>0.218</td><td>0.339</td><td>0.263</td><td>0.188</td><td>0.138</td><td>0.160 0.170</td><td>0.120 0.190</td><td>0.170 0.223</td><td>0.110</td><td>0.114 0.170</td><td></td><td>0.190</td></tr><tr><td>TriDF [52]</td><td>0.282</td><td>0.393 0.450</td><td>0.433</td><td>0.267</td><td>0.365</td><td>0.359</td><td>0.281</td><td>0.277</td><td>0.296</td><td>0.287</td><td>0.317</td><td>0.118 0.131</td><td>0.397</td><td>0.241 0.363</td><td>0.213</td></tr><tr><td>3DGS [16]</td><td>0.264</td><td>0.288 0.281</td><td></td><td>0.225 0.214</td><td>0.260</td><td>0.272</td><td>0.189</td><td>0.180</td><td>0.195</td><td>0.195</td><td>0.194</td><td>0.147 0.165</td><td>0.322</td><td>0.286</td><td>0.313</td></tr><tr><td>FSGS [27]</td><td>0.247</td><td>0.272 0.295</td><td>0.226</td><td>0.206</td><td>0.264</td><td>0.234</td><td>0.190</td><td>0.161</td><td>0.191</td><td>0.173</td><td>0.188</td><td>0.145 0.169</td><td>0.211</td><td>0.215</td><td>0.230 0.212</td></tr><tr><td>CoR-GS [30] DropGaussian [31]</td><td>0.232</td><td>0.260 0.270</td><td>0.200</td><td>0.193</td></table>

TABLE III  
ABLATION STUDY OF DIFFERENT COMPONENTS. THE AVGE VALUES ARE MULTIPLIED BY 1E2.
<table><tr><td>Setting</td><td>PSNR↑</td><td>SSIM↑</td><td>LPIPS↓</td><td>AVGE↓</td></tr><tr><td>w/o Dense Init.</td><td>29.18</td><td>0.917</td><td>0.097</td><td>3.232</td></tr><tr><td>w/o IBR Feature</td><td>30.56</td><td>0.933</td><td>0.082</td><td>2.652</td></tr><tr><td>w/o Height Map</td><td>30.68</td><td>0.935</td><td>0.080</td><td>2.593</td></tr><tr><td>w/o Pseudo Views</td><td>21.06</td><td>0.782</td><td>0.186</td><td>8.795</td></tr><tr><td>Ours</td><td>30.90</td><td>0.938</td><td>0.076</td><td>2.488</td></tr></table>

## D. Ablation Study

1) Ablation of Model Architecture: To validate the contribution of each component within the overall framework, we conduct an ablation study on the key modules of our model, with the results reported in Table III and Fig. 5. Starting from the full model, we individually remove the dense depth initialization, IBR feature fusion, anchor growth with height map constraint, and pseudo-view supervision modules for comparison. Overall, removing any single component leads to varying degrees of performance degradation, demonstrating that these modules play complementary and synergistic roles within the framework. Specifically, removing the dense depth initialization results in noticeable declines in structural consistency and geometric stability, indicating that high-quality initial point clouds are crucial for stable optimization of neural Gaussian under sparse-view settings. Eliminating the IBR feature fusion module degrades texture fidelity and crossview appearance consistency, suggesting that multi-view feature aggregation effectively compensates for insufficient input observations. The height-constrained anchor growth strategy shows relatively limited influence on quantitative metrics, but it effectively reduces floating artifacts and unreasonable Gaussian expansion, especially in weakly constrained regions. In contrast, removing pseudo-view supervision causes significant performance degradation, particularly in unobserved regions where holes and structural collapse are more likely to occur as shown in Fig. 5. This confirms that pseudo views provide essential multi-view consistency priors. In conclusion, the proposed modules enhance the model from the perspectives of initialization, anchor growth control, appearance enrichment, and multi-view consistency, jointly enabling stable and highquality results in sparse-view remote sensing scenarios.

TABLE IV  
QUANTITATIVE COMPARISON UNDER DIFFERENT MAPPING, RESCALING, AND REFINEMENT STRATEGIES. THE AVGE VALUES ARE MULTIPLIED BY 1E2.
<table><tr><td>Map</td><td>Source</td><td>Refine</td><td>MAE↓</td><td>Abs Rel.↓</td><td>AVGE↓</td></tr><tr><td>Neg.</td><td>SfM</td><td>1</td><td>3.72</td><td>0.0322</td><td>3.853</td></tr><tr><td>Inv.</td><td>SfM</td><td>一</td><td>3.16</td><td>0.0273</td><td>3.522</td></tr><tr><td>Inv.</td><td>Fused</td><td>一</td><td>3.31</td><td>0.0285</td><td>3.485</td></tr><tr><td>Inv.</td><td>SfM</td><td>Reproject</td><td>2.18</td><td>0.0187</td><td>3.406</td></tr><tr><td>Inv.</td><td>SfM</td><td>No ovlp.</td><td>1.91</td><td>0.0165</td><td>3.302</td></tr></table>

2) Ablation of Initialization: We evaluate the impact of different initialization strategies on model performance. We assess the quality of the point cloud by measuring the absolute error (MAE and Abs Rel) of the depth map projected from the dense point cloud onto the input views. All depth results are processed using z-buffering. The results are shown in Table IV and Fig. 6. We compare different mapping strategies, sources of affine transformation, and point cloud refinement strategies. Among them, Neg. and Inv. represent using the negative or reciprocal values of the original disparity for mapping, respectively. SfM and Fused refer to using the original sparse SfM point cloud and the point cloud processed through Patch Match Stereo densification for affine transformation, respectively. Reproject refers to using multi-view maximum depth to alleviate ambiguities, while No ovlp. is our proposed strategy for removing overlapping point clouds. From the experimental results, it is clear that the reciprocal mapping has higher consistency with the original depth distribution. Although Fused point clouds provide more points for scale adjustment, they also introduce more noise. Additionally, it can be observed that monocular depth tends to underestimate the background depth (indicated by the blue areas in the error map), and this issue becomes more pronounced after multiview z-buffer processing, as it defaults to retaining the closer depth. The Reproject strategy, which retains the farthest depth from the multi-view images, can alleviate this issue; however, it leads to significant distortion in the foreground depth, which tend to overestimate the depth. Our proposed local point cloud overlap removal strategy effectively mitigates this problem without damaging the foreground depth. We also computed the AVGE metric and found a significant positive correlation with the accuracy of the initialized point cloud, further confirming the importance of the initialization.

![](images/26fb68a3b699f50587bd6e1caa56688cbe8af5bee7b01fdc2deb31f1404aaf59.jpg)  
Fig. 6. Visualization of depth map and depth error with different initialization strategies.

![](images/2b19431981777014e4ca51c96fd7da25ea7ecea6b7aa210b10e4648b33c69179.jpg)  
Fig. 7. Performance under different size of voxel grid down-sampling.

![](images/43e4ee0438710978757ccb257e7a9290177d00348664c1c5a25a5135e600ab8f.jpg)  
Fig. 8. Performance under different size of anchor voxel.

TABLE V  
PERFORMANCE COMPARISON FOR DIFFERENT SIZES OF VOXEL GRID DOWNSAMPLING.
<table><tr><td>Size</td><td>Pts(k)</td><td>GS(k)</td><td>PSNR↑</td><td>SSIM↑</td><td>LPIPS↓</td><td>T(min)↓</td></tr><tr><td></td><td>574.4</td><td>525.5</td><td>28.43</td><td>0.954</td><td>0.052</td><td>43.9</td></tr><tr><td>0.2</td><td>463.4</td><td>421.1</td><td>28.22</td><td>0.952</td><td>0.052</td><td>38.5</td></tr><tr><td>0.3</td><td>282.8</td><td>263.2</td><td>28.04</td><td>0.950</td><td>0.056</td><td>28.7</td></tr><tr><td>0.5</td><td>130.3</td><td>139.0</td><td>27.72</td><td>0.943</td><td>0.064</td><td>19.9</td></tr><tr><td>1.0</td><td>42.2</td><td>92.7</td><td>27.36</td><td>0.935</td><td>0.074</td><td>17.8</td></tr><tr><td>∞</td><td>5.5</td><td>69.9</td><td>25.59</td><td>0.906</td><td>0.105</td><td>17.3</td></tr></table>

TABLE VI  
PERFORMANCE COMPARISON FOR DIFFERENT SIZES OF ANCHOR VOXEL.
<table><tr><td>Voxel Size</td><td>Pts(k)</td><td>GS(k)</td><td>PSNR↑</td><td>SSIM↑</td><td>LPIPS↓</td></tr><tr><td>0.01</td><td>42.2</td><td>150.4</td><td>28.32</td><td>0.951</td><td>0.057</td></tr><tr><td>0.03</td><td>42.2</td><td>92.7</td><td>27.36</td><td>0.935</td><td>0.074</td></tr><tr><td>0.05</td><td>42.2</td><td>64.6</td><td>26.54</td><td>0.921</td><td>0.089</td></tr><tr><td>0.10</td><td>42.2</td><td>54.3</td><td>25.95</td><td>0.909</td><td>0.100</td></tr><tr><td>0.54</td><td>38.1</td><td>43.7</td><td>25.50</td><td>0.896</td><td>0.115</td></tr></table>

Dense depth priors provide the model with initial structural information; however, a trade-off must be made between efficiency and performance, as an excessive amount of redundant point clouds inevitably slows down optimization. To this end, we apply voxel grid downsampling to the dense initial point cloud. We evaluate the impact of different downsampling rates on model performance, and record both the number of initial points and the final number of Gaussian anchors to assess redundancy in the initialization. The results are shown in Fig. 7 and Table V, which is conducted on scene Building#1. As the number of initial points increases, both training time and overall model performance improve accordingly. Nevertheless, a clear elbow point emerges in the curve, beyond which the performance gains become much smaller relative to the increased computational cost. In Fig. 7, the sizes of the blue and yellow concentric circles represent the numbers of initial points and final Gaussian anchors, respectively. It can be observed that when the initialization contains few points, neural Gaussians must grow extensively to capture scene details. Once the number of initial points exceeds 100k, the final number of Gaussians remains nearly unchanged, indicating that the point number has approached saturation. When the number further exceeds 280k, the final number even decreases, revealing substantial redundancy in the initialization. Overall, downsampling the initial dense point cloud is crucial for improving training efficiency and eliminating redundancy without compromising reconstruction quality.

TABLE VII  
EFFECT OF DIFFERENT ANCHOR FEATURE AND IBR FEATURE CHANNELS.
<table><tr><td> $C _ { f }$ </td><td> $C _ { r }$ </td><td>PSNR↑</td><td>SSIM↑</td><td>LPIPS↓</td></tr><tr><td>32</td><td>32</td><td>30.90</td><td>0.938</td><td>0.076</td></tr><tr><td>32</td><td>64</td><td>30.72</td><td>0.935</td><td>0.080</td></tr><tr><td>32</td><td>128</td><td>30.71</td><td>0.935</td><td>0.080</td></tr><tr><td>32</td><td>256</td><td>30.84</td><td>0.936</td><td>0.078</td></tr><tr><td>16</td><td>64</td><td>30.39</td><td>0.930</td><td>0.088</td></tr><tr><td>32</td><td>64</td><td>30.72</td><td>0.935</td><td>0.080</td></tr><tr><td>64</td><td>64</td><td>30.85</td><td>0.937</td><td>0.077</td></tr><tr><td>64</td><td>128</td><td>30.70</td><td>0.935</td><td>0.079</td></tr></table>

The size of the anchors reflects the level of granularity used for scene modeling. We evaluate the effect of different anchor sizes on scene Building#1, with the results shown in Fig. 8 and Table VI. With the number of initial points fixed, finer grid resolutions lead to higher reconstruction performance, but at the cost of a larger number of anchors and increased computational overhead.

3) Ablation of Network: The anchor feature dimension $C _ { f }$ and the IBR reference feature dimension $C _ { r }$ directly affect both the representational capacity and the computational cost of the model. Therefore, we conduct an ablation study on different combinations of dimensions and the results are reported in Table VII.

When the anchor feature dimension is fixed at 32, increasing the IBR feature dimension from 32 to 256 leads to only minor performance variations. In particular, the best performance is achieved when $C _ { r } ~ = ~ 3 2$ , while further increasing the IBR feature dimension does not yield noticeable improvements. This suggests that in the current task, IBR features primarily serve to supplement cross-view appearance information, which can be sufficiently represented at relatively low dimensionality.

TABLE VIII  
ABLATION STUDY ON DIFFERENT NETWORK SETTINGS.
<table><tr><td>Setting</td><td>PSNR↑</td><td>SSIM↑</td><td>LPIPS↓</td></tr><tr><td>w/o MLP  $\mathcal { E } _ { \mathrm { m o d } }$ </td><td>30.82</td><td>0.937</td><td>0.077</td></tr><tr><td>w/o ft. P</td><td>30.79</td><td>0.936</td><td>0.079</td></tr><tr><td>w/o ResNet Feature</td><td>30.85</td><td>0.937</td><td>0.078</td></tr><tr><td>w/o ViT Feature</td><td>30.81</td><td>0.936</td><td>0.078</td></tr><tr><td>Ours</td><td>30.90</td><td>0.938</td><td>0.076</td></tr></table>

Increasing the number of channels beyond this level introduces redundant information without effectively enhancing the representation capability. We further analyze the influence of the anchor feature dimension. The results show that increasing $C _ { f }$ from 16 to 32 significantly improves model performance, whereas further increasing it to 64 results in only marginal gains. This indicates that, under the complexity of the current scenes, a 32-dimensional feature space is sufficient to effectively encode the key attributes of anchors. Overall, the anchor feature dimension has a more pronounced impact on model performance, while the IBR feature dimension mainly provides auxiliary information. Moreover, we observe that the model tends to achieve better performance when the two feature dimensions are well matched. Considering the tradeoff between performance and computational efficiency, we adopt the configuration $C _ { f } = 3 2$ and $C _ { r } = 3 2$ in the final model, which achieves the best overall performance in our experiments.

We further conduct ablation experiments on the network architecture in Table VIII. Removing the modulation network or freezing the feature projection module consistently degrades performance, confirming the importance of adaptive IBR feature integration. Excluding either ResNet or ViT features also leads to performance drops, indicating that local texture cues and global semantic information are complementary for crossview feature modeling.

## E. Analysis of DIBR-based Pseudo-view Supervision

Fig. 9(a) illustrates the pipeline of the proposed progressive DIBR-based supervision strategy. Given the camera trajectory of the training views, intermediate viewpoints are sampled to construct additional supervision signals. For each target viewpoint, its depth map is obtained from the dense point cloud and used to establish geometric correspondence for depth-based warping. Next, images from multiple source views are projected to the target viewpoint via depth-based warping. The source views are integrated progressively according to their distances to the target viewpoint, since closer views generally provide more reliable correspondences with fewer occlusions. In this way, the visible regions of the target view are gradually filled, while the remaining uncovered regions are completed using a padding strategy, resulting in the final pseudo-view image.

Fig. 9(b) provides an example of the progressive synthesis process. The top row visualizes the coverage of different source views in the target view, where the orange, blue, and green regions represent the projected areas from three source views, and the purple regions denote the padding areas. The bottom row shows the corresponding step-by-step synthesis results. As illustrated, the visible regions contributed by different source views vary significantly in the target view, making it difficult for a single view to fully cover the target image. By progressively integrating projections from multiple views, the observable region can be effectively expanded, gradually establishing a more complete supervision signal. In this example, the target pseudo-view is selected from the test set, and we compute the PSNR between the synthesized pseudo-view and the ground-truth image.

![](images/4f3438d912a86107e4930dba31f8f01a625783698138ca4c29f8adc2adabb7f1.jpg)  
Fig. 9. Illustration and analysis of the proposed pseudo view synthesis. (a) Pipeline of the pseudo view synthesis through progressive DIBR.(b) Example of the progressive synthesis process.

Although the synthesized pseudo-view achieves a relatively low pixel-level fidelity (PSNR = 15.82) due to accumulated depth errors and reprojection inaccuracies, it should be noted that these pseudo views are not expected to provide pixelperfect targets. Instead, they provide approximate geometric and appearance constraints that complement the sparse original observations. This guidance encourages neural Gaussians to expand into unobserved regions. As a result, the final rendered image produced by the model achieves a PSNR of 27.86. Furthermore, when pseudo-view supervision is removed, the model produces noticeable holes in novel view rendering, as shown in Fig. 5, indicating that supervision from the original views alone is insufficient to cover all visible regions under sparse-view settings. These results demonstrate that the proposed strategy effectively improves supervision coverage and provides valuable multi-view consistency constraints for sparse-view reconstruction.

## V. CONCLUSION

This paper addresses the challenge of novel view synthesis in remote sensing scenarios under sparse observations. To this end, we propose DIBR-GS, a neural Gaussian framework that exploits Depth Image-Based Rendering (DIBR) as an additional consistency supervision mechanism between observed and unobserved viewpoints. By integrating reliable geometric initialization, cross-view appearance priors, and progressive DIBR-guided optimization, the proposed framework effectively alleviates geometric ambiguity and insufficient supervision caused by sparse inputs. Extensive experiments demonstrate that DIBR-GS consistently improves reconstruction quality, producing more complete structures, sharper appearance details, and fewer artifacts. These results verify the effectiveness of utilizing DIBR-based supervision for enhancing multi-view consistency in sparse-view remote sensing novel view synthesis. The proposed framework provides a promising solution for remote sensing 3D reconstruction and scene understanding when dense observations are unavailable. However, the proposed method still relies on perscene optimization. Future research will explore more efficient and generalizable novel view synthesis approaches, such as feed-forward methods.

## REFERENCES

[1] Y. Wu, Z. Zou, and Z. Shi, “Remote sensing novel view synthesis with implicit multiplane representations,” IEEE Trans. Geosci. Remote Sens.,

vol. 60, pp. 1–13, 2022.

[2] C. Zhang, Y. Yan, C. Zhao, N. Su, and W. Zhou, “Fvmd-isre: 3-d reconstruction from few-view multidate satellite images based on the implicit surface representation of neural radiance fields,” IEEE Trans. Geosci. Remote Sens., vol. 62, pp. 1–14, 2024.

[3] R. Mar´ı, G. Facciolo, and T. Ehret, “Sat-nerf: Learning multi-view satellite photogrammetry with transient objects and shadow modeling using rpc cameras,” in Proc. IEEE/CVF Conf. Comput. Vis. Pattern Recognit. Workshops, New Orleans, LA, USA, June 2022, pp. 1311– 1321.

[4] Z. Gao, L. Jiao, L. Li, X. Liu, F. Liu, P. Chen, and Y. Guo, “Multiplane prior guided few-shot aerial scene rendering,” in Proc. IEEE/CVF Conf. Comput. Vis. Pattern Recognit., Seattle, WA, USA, 2024, pp. 5009–5019.

[5] Y. Wu, Z. Shi, and Z. Zou, “Progressive gaussian splatting for highfidelity 3d reconstruction of remote sensing scenes,” IEEE Geosci. Remote Sens. Mag., 2026.

[6] B. Lin, Z. Zou, and Z. Shi, “Rsbev-mamba: 3-d bev sequence modeling for multiview remote sensing scene segmentation,” IEEE Trans. Geosci. Remote Sens., vol. 63, pp. 1–13, 2025.

[7] B. Chen, L. Liu, C. Liu, Z. Zou, and Z. Shi, “Spectral-cascaded diffusion model for remote sensing image spectral super-resolution,” IEEE Trans. Geosci. Remote Sens., pp. 1–14, 2024.

[8] C. Liu, J. Zhang, K. Chen, M. Wang, Z. Zou, and Z. Shi, “Remote sensing spatiotemporal vision–language models: A comprehensive survey,” IEEE Geosci. Remote Sens. Mag., vol. 14, no. 1, pp. 383–423, 2026.

[9] W. Fan, X. Liu, X. Wang, L. Cao, T. Li, and Y. Zhang, “Multiscale gaussian splatting scene understanding with k-planes encoding,” IEEE Transactions on Geoscience and Remote Sensing, vol. 64, pp. 5 621 213– 5 621 213, 2026.

[10] Y. Yao, Z. Luo, S. Li, T. Fang, and L. Quan, “Mvsnet: Depth inference for unstructured multi-view stereo,” in Proc. Eur. Conf. Comput. Vis., Munich, Germany, 2018, pp. 767–783.

[11] A. Tewari, O. Fried, J. Thies, V. Sitzmann, S. Lombardi, K. Sunkavalli, R. Martin-Brualla, T. Simon, J. Saragih, M. Nießner et al., “State of the art on neural rendering,” in Comput. Graph. Forum, vol. 39, no. 2. Wiley Online Library, 2020, pp. 701–727.

[12] B. Mildenhall, P. P. Srinivasan, M. Tancik, J. T. Barron, R. Ramamoorthi, and R. Ng, “Nerf: Representing scenes as neural radiance fields for view synthesis,” in Proc. Eur. Conf. Comput. Vis., Glasgow, UK, 2020, pp. 405–421.

[13] Y. Xiangli, L. Xu, X. Pan, N. Zhao, A. Rao, C. Theobalt, B. Dai, and D. Lin, “Bungeenerf: Progressive neural radiance field for extreme multiscale scene rendering,” in Proc. Eur. Conf. Comput. Vis., Tel Aviv, Israel, 2022, pp. 106–122.

[14] T. Muller, A. Evans, C. Schied, and A. Keller, “Instant neural graphics¨ primitives with a multiresolution hash encoding,” ACM Trans. Graph., vol. 41, no. 4, pp. 1–15, 2022.

[15] H. Pan, G. Wu, Z. Hong, S. Liu, H. Xie, Y. Xu, Z. Ye, Y. Xiang, and X. Tong, “Rds-nerf: Residual and depth supervision neural radiance field for multiscene 3-d reconstruction of satellite images,” IEEE Trans. Geosci. Remote Sens., vol. 63, pp. 1–24, 2025.

[16] B. Kerbl, G. Kopanas, T. Leimkuhler, and G. Drettakis, “3d gaussian¨ splatting for real-time radiance field rendering.” ACM Trans. Graph., vol. 42, no. 4, pp. 1–14, 2023.

[17] B. Huang, Z. Yu, A. Chen, A. Geiger, and S. Gao, “2d gaussian splatting for geometrically accurate radiance fields,” in ACM SIGGRAPH Conf. Proc., Denver, CO, USA, 2024, pp. 1–11.

[18] Y. Bao, T. Ding, J. Huo, Y. Liu, Y. Li, W. Li, Y. Gao, and J. Luo, “3d gaussian splatting: Survey, technologies, challenges, and opportunities,” IEEE Trans. Circuits Syst. Video Technol., vol. 35, no. 7, pp. 6832–6852, 2025.

[19] J. Wang, M. Liu, X. Yang, X. Wei, L. Wang, and N. Wang, “Satellitegs: Enhanced 2d gaussian splatting for robust satellite reconstruction,” IEEE Trans. Geosci. Remote Sens., vol. 64, pp. 1–15, 2026.

[20] M. Niemeyer, J. T. Barron, B. Mildenhall, M. S. M. Sajjadi, A. Geiger, and N. Radwan, “Regnerf: Regularizing neural radiance fields for view synthesis from sparse inputs,” in Proc. IEEE/CVF Conf. Comput. Vis. Pattern Recognit., New Orleans, LA, USA, 2022, pp. 5470–5480.

[21] J. Yang, M. Pavone, and Y. Wang, “Freenerf: Improving few-shot neural rendering with free frequency regularization,” in Proc. IEEE/CVF Conf. Comput. Vis. Pattern Recognit., Vancouver, Canada, 2023, pp. 8254– 8263.

[22] N. Somraj and R. Soundararajan, “Vip-nerf: Visibility prior for sparse input neural radiance fields,” in ACM SIGGRAPH Conf. Proc., Los Angeles, CA, USA, 2023, pp. 1–11.

[23] A. Jain, M. Tancik, and P. Abbeel, “Putting nerf on a diet: Semantically consistent few-shot view synthesis,” in Proc. IEEE/CVF Int. Conf. Comput. Vis., Montreal, QC, Canada, 2021, pp. 5885–5894.

[24] H. Guo, C. Liu, H. Zhang, B. Chen, Z. Zou, and Z. Shi, “Taco: Capturing spatio-temporal semantic consistency in remote sensing change detection,” arXiv preprint arXiv:2511.20306, 2025.

[25] K. Deng, A. Liu, J.-Y. Zhu, and D. Ramanan, “Depth-supervised nerf: Fewer views and faster training for free,” in Proc. IEEE/CVF Conf. Comput. Vis. Pattern Recognit., New Orleans, LA, USA, 2022, pp. 12 882–12 891.

[26] B. Roessle, J. T. Barron, B. Mildenhall, P. P. Srinivasan, and M. Nießner, “Dense depth priors for neural radiance fields from sparse input views,” in Proc. IEEE/CVF Conf. Comput. Vis. Pattern Recognit., New Orleans, LA, USA, 2022, pp. 12 892–12 901.

[27] Z. Zhu, Z. Fan, Y. Jiang, and Z. Wang, “Fsgs: Real-time few-shot view synthesis using gaussian splatting,” in Proc. Eur. Conf. Comput. Vis. Milan, Italy: Springer, 2024, pp. 145–163.

[28] J. Li, J. Zhang, X. Bai, J. Zheng, X. Ning, J. Zhou, and L. Gu, “Dngaussian: Optimizing sparse-view 3d gaussian radiance fields with global-local depth normalization,” in Proc. IEEE/CVF Conf. Comput. Vis. Pattern Recognit., Seattle, WA, USA, June 2024, pp. 20 775–20 785.

[29] J. Chung, J. Oh, and K. M. Lee, “Depth-regularized optimization for 3d gaussian splatting in few-shot images,” in Proc. IEEE/CVF Conf. Comput. Vis. Pattern Recognit. Workshops, Seattle, WA, USA, 2024, pp. 811–820.

[30] J. Zhang, J. Li, X. Yu, L. Huang, L. Gu, J. Zheng, and X. Bai, “Cor-gs: sparse-view 3d gaussian splatting via co-regularization,” in Proc. Eur. Conf. Comput. Vis. Milan, Italy: Springer, 2024, pp. 335–352.

[31] H. Park, G. Ryu, and W. Kim, “Dropgaussian: Structural regularization for sparse-view gaussian splatting,” in Proc. IEEE/CVF Conf. Comput. Vis. Pattern Recognit., Nashville, Tennessee, USA, June 2025, pp. 21 600–21 609.

[32] W. Sun, L. Xu, O. C. Au, S. H. Chui, and C. W. Kwok, “An overview of free view-point depth-image-based rendering (dibr),” in APSIPA Annu. Summit Conf., 2010, pp. 1023–1030.

[33] Q. Wang, Z. Wang, K. Genova, P. Srinivasan, H. Zhou, J. T. Barron, R. Martin-Brualla, N. Snavely, and T. Funkhouser, “Ibrnet: Learning multi-view image-based rendering,” in Proc. IEEE/CVF Conf. Comput. Vis. Pattern Recognit., Nashville, TN, USA, 2021, pp. 4688–4697.

[34] J. T. Barron, B. Mildenhall, M. Tancik, P. Hedman, R. Martin-Brualla, and P. P. Srinivasan, “Mip-nerf: A multiscale representation for antialiasing neural radiance fields,” in Proc. IEEE/CVF Int. Conf. Comput. Vis., Montreal, QC, Canada, October 2021, pp. 5855–5864.

[35] L. Liu, J. Gu, K. Zaw Lin, T.-S. Chua, and C. Theobalt, “Neural sparse voxel fields,” Adv. Neural Inf. Process. Syst., vol. 33, pp. 15 651–15 663, 2020.

[36] S. Fridovich-Keil, A. Yu, M. Tancik, Q. Chen, B. Recht, and A. Kanazawa, “Plenoxels: Radiance fields without neural networks,” in Proc. IEEE/CVF Conf. Comput. Vis. Pattern Recognit., New Orleans, LA, USA, June 2022, pp. 5501–5510.

[37] A. Chen, Z. Xu, A. Geiger, J. Yu, and H. Su, “Tensorf: Tensorial radiance fields,” in Proc. Eur. Conf. Comput. Vis. Tel Aviv, Israel: Springer, 2022, pp. 333–350.

[38] Q. Xu, Z. Xu, J. Philip, S. Bi, Z. Shu, K. Sunkavalli, and U. Neumann, “Point-nerf: Point-based neural radiance fields,” in Proc. IEEE/CVF Conf. Comput. Vis. Pattern Recognit., New Orleans, LA, USA, 2022, pp. 5438–5448.

[39] R. Tucker and N. Snavely, “Single-view view synthesis with multiplane images,” in Proc. IEEE/CVF Conf. Comput. Vis. Pattern Recognit., Seattle, WA, USA, 2020, pp. 548–557.

[40] J. Li, Z. Feng, Q. She, H. Ding, C. Wang, and G. H. Lee, “Mine: Towards continuous depth mpi with nerf for novel view synthesis,” in Proc. IEEE/CVF Int. Conf. Comput. Vis., Montreal, QC, Canada, 2021, pp. 12 578–12 588.

[41] W. Hu, Y. Wang, L. Ma, B. Yang, L. Gao, X. Liu, and Y. Ma, “Trimiprf: Tri-mip representation for efficient anti-aliasing neural radiance fields,” in Proc. IEEE/CVF Int. Conf. Comput. Vis., Paris, France, 2023, pp. 19 774–19 783.

[42] T. Lu, M. Yu, L. Xu, Y. Xiangli, L. Wang, D. Lin, and B. Dai, “Scaffold-gs: Structured 3d gaussians for view-adaptive rendering,” in Proc. IEEE/CVF Conf. Comput. Vis. Pattern Recognit., Seattle, WA, USA, 2024, pp. 20 654–20 664.

[43] A. Yu, V. Ye, M. Tancik, and A. Kanazawa, “pixelnerf: Neural radiance fields from one or few images,” in Proc. IEEE/CVF Conf. Comput. Vis. Pattern Recognit., Nashville, TN, USA, 2021, pp. 4576–4585.

[44] A. Chen, Z. Xu, F. Zhao, X. Zhang, F. Xiang, J. Yu, and H. Su, “Mvsnerf: Fast generalizable radiance field reconstruction from multi-view stereo,”

in Proc. IEEE/CVF Int. Conf. Comput. Vis., Montreal, QC, Canada, 2021, pp. 14 104–14 113.

[45] A. Paliwal, W. Ye, J. Xiong, D. Kotovenko, R. Ranjan, V. Chandra, and N. K. Kalantari, “Coherentgs: Sparse novel view synthesis with coherent 3d gaussians,” in Proc. Eur. Conf. Comput. Vis., A. Leonardis, E. Ricci, S. Roth, O. Russakovsky, T. Sattler, and G. Varol, Eds. Cham: Springer Nature Switzerland, 2025, pp. 19–37.

[46] Y. Zheng, Z. Jiang, S. He, Y. Sun, J. Dong, H. Zhang, and Y. Du, “Nexusgs: Sparse view synthesis with epipolar depth priors in 3d gaussian splatting,” in Proc. IEEE/CVF Conf. Comput. Vis. Pattern Recognit., Nashville, Tennessee, USA, 2025, pp. 26 800–26 809.

[47] J. A. Hartigan and P. M. Hartigan, “The dip test of unimodality,” Ann. Stat., pp. 70–84, 1985.

[48] J. L. Bentley, “Multidimensional binary search trees used for associative searching,” Commun. ACM, vol. 18, no. 9, pp. 509–517, 1975.

[49] M. Song, X. Lin, D. Zhang, H. Li, X. Li, B. Du, and L. Qi, “D<sup>2</sup>GS: Depth-and-density guided gaussian splatting for stable and accurate sparse-view reconstruction,” in Int. Conf. Learn. Represent., 2026.

[50] M. Caron, H. Touvron, I. Misra, H. Jegou, J. Mairal, P. Bojanowski, and´ A. Joulin, “Emerging properties in self-supervised vision transformers,” in Proc. IEEE/CVF Int. Conf. Comput. Vis., Montreal, QC, Canada, 2021, pp. 9650–9660.

[51] J. Jung, J. Han, H. An, J. Kang, S. Park, and S. Kim, “Relaxing accurate initialization constraint for 3d gaussian splatting,” arXiv preprint arXiv:2403.09413, 2024.

[52] J. Kang, K. Chen, Z. Zou, and Z. Shi, “Tridf: Triplane-accelerated density fields for few-shot remote sensing novel view synthesis,” IEEE Trans. Geosci. Remote Sens., vol. 63, no. 5656715, 2025.

[53] L. Yang, B. Kang, Z. Huang, Z. Zhao, X. Xu, J. Feng, and H. Zhao, “Depth anything v2,” Adv. Neural Inf. Process. Syst., vol. 37, pp. 21 875– 21 911, 2024.

[54] J. L. Schonberger and J.-M. Frahm, “Structure-from-motion revisited,”¨ in Proc. IEEE/CVF Conf. Comput. Vis. Pattern Recognit., Las Vegas, NV, USA, 2016, pp. 4104–4113.

[55] K. He, X. Zhang, S. Ren, and J. Sun, “Deep residual learning for image recognition,” in Proc. IEEE/CVF Conf. Comput. Vis. Pattern Recognit., 2016, pp. 770–778.

[56] A. Kolesnikov, A. Dosovitskiy, D. Weissenborn, G. Heigold, J. Uszkoreit, L. Beyer, M. Minderer, M. Dehghani, N. Houlsby, S. Gelly, T. Unterthiner, and X. Zhai, “An image is worth 16x16 words: Transformers for image recognition at scale,” in Int. Conf. Learn. Represent., 2021.
# Superquadric Primitive Decomposition of 3D point clouds via Geometric-Aware Inlier Refinement

Alessandro Rinaldi<sup>a,∗</sup>, Edoardo Tedesco<sup>b,∗</sup>, Andrea Ferraris<sup>b</sup>, Filippo Leveni<sup>b</sup>, Daniele Baieri<sup>c</sup>, Filippo Maggioli<sup>d</sup>, Simone Melzi<sup>a</sup> and Luca Magri<sup>b,∗∗</sup>

<sup>a</sup>Department ofInformatics, System and Communication, University ofMilano-Bicocca, Milan, Italy

<sup>b</sup>Department of Electronics, Information and Bioengineering, Politecnico di Milano, Milan, Italy

<sup>c</sup>Department ofComputer Science, University ofBonn, Bonn, Germany

<sup>d</sup>Department ofInformation Science and Technology, Pegaso University, Naples, Italy

## A R T I C L E I N F O

Keywords:   
3D Vision   
Shape representation   
Computer vision

## A BS T RA C T

The decomposition of 3D point clouds into interpretable geometric primitives remains a longstanding challenge in Computer Vision and Computer Graphics. Among the available representations, superquadrics ofer a compact and expressive model capable of capturing a wide range of shapes. However, their estimation is inherently challenging, as it requires solving a non-linear optimization problem and is particularly sensitive to noise, outliers, and overlapping structures.

While robust estimation methods such as RANSAC and its variants achieve strong performance, they rely primarily on spatial proximity and residual-based criteria, often leading to incorrect inlier assignments across adjacent or complex arrangements of primitives.

In this work, we introduce a geometric-aware framework for primitive decomposition that explicitly incorporates local surface properties into the fitting process. Specifically, we propose an inlier refinement step formulated as an energy minimization problem and solved via graph-cut optimization. Our formulation integrates geometric priors, such as normal consistency, enabling more reliable inlier selection beyond purely residual-based criteria. The approach naturally applies to both single-model estimation and multi-model decomposition.

By leveraging geometric information beyond point-wise residuals, our method reduces erroneous inlier propagation and stabilizes parameter estimation. Experiments on synthetic and real datasets show consistent improvements in geometric accuracy, robustness to noise and outliers, and convergence eficiency compared to state-of-the-art RANSAC-based methods.

## 1. Introduction

The task offitting geometric models to visual data to derive structured interpretations of unorganized spatial points is a key challenge in Computer Vision and Graphics. In the context of 3D data, primitive decomposition becomes particularly relevant, as geometric model fitting enables representing complex scenes using simple primitives such as planes, cylinders, or superquadrics, providing a compact and interpretable approximation of real-world 3D structures as depicted in Figure 1. This is fundamental for applications such as scene understanding and robotics, where accurate shape modelling underpins tasks such as object grasping [30], collision avoidance [8], and navigation [31]. More generally, primitive-based representations are attractive because they ofer compact, interpretable descriptions of 3D shape, often requiring only a small number of parameters to represent structures that would otherwise require dense representations such as voxel grids, point clouds or meshes.

However, to be efective, primitive decomposition must balance compactness with geometric accuracy. A useful decomposition should represent the scene with a small number of primitives while still capturing its relevant geometric structure. This is challenging: not only must the method be robust to noise and outliers, which are almost unavoidable in real data, but the simultaneous presence of multiple primitives requires jointly estimating both segmentation and model parameters, significantly increasing the complexity of the problem. This challenge is further exacerbated in scenes where primitives are spatially adjacent and partially overlapping.

One of the most widely adopted approaches for robust model fitting is consensus maximization, as epitomized by the well-known RANSAC [10] paradigm and its numerous variants. RANSAC estimates a model by iteratively sampling subsets of the data and selecting the hypothesis that maximizes the number of inliers according to a residual threshold. This strategy has proven highly efective for single-model fitting, particularly in the presence of noise and outliers.

Several extensions have been proposed to improve the quality of the estimated models. Methods such as LO-RANSAC [7] and GC-RANSAC [4] incorporate local optimization and spatial coherence, refining inlier sets by encouraging neighboring points with low residuals to be jointly selected as inliers, while penalizing assignments that are inconsistent with the local data structure. These approaches improve robustness and convergence, and represent the current standard in many geometric estimation tasks.

![](images/3f296e148253716b663df84bc80ed7d2a20e0dc0d7a3f1555f54057d472c1438.jpg)  
Figure 1: Primitive Decomposition of diferent 3D point clouds (top row) exploiting our Geometric Aware Inlier Refinement (GAIR) to extract superquadrics (bottom row).

In parallel, multi-model fitting methods extend this paradigm to estimate multiple structures. Approaches such as RANSACOV [17] formulate the problem as a model selection or coverage optimization task, where a pool of candidate models is generated, and a subset is selected to maximize the number of explained points.

Despite their efectiveness, these approaches are primarily designed for settings where model fitting can be reliably driven by residual-based criteria and spatial proximity. However, this assumption becomes limiting in 3D primitive decomposition, where the underlying structures are continuous surfaces characterized by local geometric properties such as orientation and smoothness.

In this setting, spatial proximity becomes a weak proxy for surface membership: points that are close in Euclidean space may belong to diferent primitives, especially near regions where surfaces intersect or come into contact. As a result, formulations based solely on residuals and proximity may incorrectly propagate inliers across adjacent structures.

This is illustrated in Figure 2(a), where two pairs of points are shown at comparable spatial distance. In the first pair (highlighted by the blue dashed circle), the points exhibit consistent surface orientations and can plausibly belong to the same primitive. In contrast, the second pair (highlighted by the green circle) has diferent normals, indicating that the points belong to diferent surface regions, despite being spatially close. This example highlights that spatial proximity alone is insuficient to determine surface membership. Instead, proximity and local geometry must be considered jointly: while similar normals do not guarantee that two points lie on the same primitive, nearby points with inconsistent normals should not be assigned to the same structure.

To illustrate this efect, let $h _ { j }$ denote a candidate model hypothesis generated by a consensus maximization procedure, and $\boldsymbol { \mathit { I } } _ { j }$ the corresponding inlier set obtained via residual-based thresholding. Figure 2(b–d) shows how different refinement strategies afect the resulting inlier set: spatial coherence alone may incorrectly expand $\boldsymbol { \mathit { I } } _ { j }$ across geometrically distinct regions, whereas incorporating geometric consistency prevents such assignments and yields a more coherent set of inliers.

To address this limitation, we propose a Geometric-Aware Inlier Refinement strategy (GAIR) that explicitly accounts for the local surface structure of the data. Our key idea is to augment standard residual-based formulations with geometric priors that capture surface-level properties, such as normal consistency and local surface coherence. By incorporating these cues, inlier selection is no longer driven solely by spatial proximity, but also by the coherence of points with respect to the underlying surface.

We formulate this refinement as an energy minimization problem over a labeling of the point cloud, which is eficiently solved via graph-cut optimization. Building on existing energy-based formulations (e.g., GC-RANSAC), we introduce geometric priors tailored to 3D primitive decomposition, enabling inlier selection consistent with the underlying surface structure. The resulting framework can be integrated into standard consensus maximization pipelines, improving the stability of the estimated models and reducing erroneous inlier propagation across adjacent primitives.

In this work, we focus on superquadric primitives, which provide a compact and expressive representation capable of modeling a wide range of shapes with a small number of parameters. Their flexibility makes them particularly appealing for primitive-based scene representations.

However, this expressiveness comes at the cost of increased estimation complexity. Unlike simpler primitives, superquadrics cannot be reliably inferred from minimal samples in closed form, as their parameters must be recovered through a non-linear optimization process. As a result, their estimation is highly sensitive to noise, outliers, and initialization, making robust fitting particularly challenging in realistic settings. This makes superquadric fitting a suitable testbed for evaluating inlier-refinement strategies in 3D.

![](images/87c89aaa2ad63a30767ec9aab50a8638a554503205f7d788346b1a2f682e4904.jpg)

![](images/f9fcf47acd34e37f6a48dbeddc0960dead50f20fdc81072327006905c6b47c2d.jpg)

![](images/e9fd87e5609c87141776164b3500dd258830ff8b3e402ccbf0202d16a0b24247.jpg)  
(c) Proximity based

![](images/10fb2439a6e1d41df3eea65d159b70960a885baab532093ba1a28f4759c1bab1.jpg)  
(d) Geometric-aware  
Figure 2: Geometric ambiguity in 3D primitive decomposition and inlier refinement strategies. (a) Two spatially close points may belong to diferent surface regions, as indicated by their difering normals in the green circle. (b) Initial inlier set $\boldsymbol { \mathit { 1 } } _ { j }$ obtained via residual-based thresholding. (c) Proximity-based refinement propagates inliers across adjacent surfaces due to proximity. (d) Incorporating geometric consistency prevents assignments across incompatible regions, yielding a coherent inlier set.

## 1.1. Contributions.

This work presents a geometric-aware formulation for robust primitive fitting in 3D point clouds. Specifically, we observe that, while consensus maximization methods have been successfully extended with spatial coherence (e.g., GC-RANSAC), their formulation remains primarily driven by point-wise residuals and proximity, which do not explicitly capture the underlying surface geometry in 3D primitive decomposition. In this respect, our contributions are:

• Geometric-Aware Inlier Refinement (GAIR). We propose a novel inlier refinement strategy formulated as an energy minimization problem, extending standard graph-cut formulations by incorporating geometric priors with a particular focus on normal consistency and surface-aware regularization. This enables inlier selection that is coherent with the underlying surface geometry, going beyond purely residual- and proximity-based criteria.

• GAIR-RANSAC for single-model fitting. We integrate the proposed refinement into a consensus maximization framework, resulting in a robust estimator for superquadric fitting. The geometric-aware refinement improves the stability of the inlier set, leading to more accurate parameter estimation and a favorable accuracy–runtime trade-of compared to existing RANSAC-based methods.

• GAIR-RANSACOV for primitive decomposition. We extend the framework to the multi-model setting by combining GAIR-based hypothesis generation with a Maximum Coverage formulation for model selection. This solution enables robust decomposition of point clouds into multiple superquadrics, particularly in challenging scenarios with adjacent or overlapping primitives.

This work extends our previous conference version [9] by introducing the extension to multi-model primitive decomposition and a more comprehensive evaluation. We validate the proposed approach on synthetic, controlled scenarios and real data demonstrating improved robustness and stability compared to existing methods. While the present study focuses on relatively structured settings, the formulation provides a foundation for tackling more complex realworld primitive decomposition tasks.

## 2. Problem formulation

We consider the problem of primitive decomposition in 3D point clouds. A scene is represented by an unordered set of noisy points $D = \{ \mathbf { x } _ { 1 } , \dots , \mathbf { x } _ { N } \}$ , with $\mathbf { x } _ { i } \in \mathbb { R } ^ { 3 }$ . For each point $\mathbf { x } _ { i } ,$ a normal vector ${ \bf n } _ { i }$ is either provided or estimated, yielding a set $\vec { \mathcal { V } } ~ = ~ \{ \mathbf { n } _ { 1 } , \dots , \mathbf { n } _ { N } \}$ , where each $\mathbf { n } _ { i } ~ \in ~ \mathbb { R } ^ { 3 }$ is associated with a point in  and is normalized so that $\| \mathbf { n } _ { i } \| _ { 2 } = 1$ . We assume that all the normals are consistently oriented. The goal is to recover a set of geometric primitives that explain the observed data, represented as a collection of parameters $\{ \theta _ { 1 } , \ldots , \theta _ { \kappa } \}$ , each defining a superquadric, where the number of primitives � may be either specified or estimated as part of the decomposition process.

The inlier threshold �, given a primitive, defines its inliers as those points with residuals lower than �. As customary in consensus-based robust estimation, we assume that � is provided as input as an estimate of the expected measurement uncertainty. While we focus on superquadrics, the formulation can be extended to other primitives (e.g., planes, spheres or collections of surface patches).

Superquadrics. Superquadrics, introduced in [5], are a family of parametric surfaces derived from quadrics that can represent a wide range of shapes using a compact set of parameters. Their flexibility arises from the use of exponent parameters that control the roundness or sharpness of the surface along diferent directions.

A canonical superquadric is defined in its local coordinate system by the implicit function:

$$
F ( \mathbf { x } , \Lambda ) = \left( \left| { \frac { x } { a _ { 1 } } } \right| ^ { \frac { 2 } { \varepsilon _ { 2 } } } + \left| { \frac { y } { a _ { 2 } } } \right| ^ { \frac { 2 } { \varepsilon _ { 2 } } } \right) ^ { \frac { \varepsilon _ { 2 } } { \varepsilon _ { 1 } } } + \left| { \frac { z } { a _ { 3 } } } \right| ^ { \frac { 2 } { \varepsilon _ { 1 } } } = 1 ,\tag{1}
$$

where $\mathbf { x } = ( x , y , z ) ^ { \top } \in \mathbb { R } ^ { 3 }$

The parameter set $\Lambda = \{ a _ { 1 } , a _ { 2 } , a _ { 3 } , \varepsilon _ { 1 } , \varepsilon _ { 2 } \}$ defines the canonical shape, where $a _ { 1 } , a _ { 2 } , a _ { 3 }$ control the scale along the coordinate axes, and $\varepsilon _ { 1 } , \varepsilon _ { 2 }$ control the shape. In particular, $\varepsilon _ { 1 }$ governs the geometry along the �-axis, while $\varepsilon _ { 2 }$ afects the cross-section in the orthogonal plane.

To represent a superquadric in the scene, the canonical model is mapped from its local coordinate system to the world coordinate system via a rigid transformation. This transformation is parameterized by a rotation $R ( \alpha , \beta , \gamma )$ and a translation $\mathbf { t } \in \mathbb { R } ^ { 3 }$ . The full model is therefore defined by $\theta = \{ \Lambda , \alpha , \beta , \gamma , \mathbf { t } \}$ , resulting in a total of 11 parameters (5 shape parameters and 6 pose parameters).

Given a superquadric �, measuring the residual $r ( \mathbf { x } , \theta )$ between a point $\textbf { x } \in { \mathbb { R } } ^ { 3 }$ and the surface is non-trivial, and diferent approximations can be used. Since the implicit function $F ( \cdot , \Lambda )$ is defined in the canonical coordinate system, a point � in the scene is first mapped to canonical coordinates as $\mathbf { x } ^ { \prime } = R ^ { \top } ( \mathbf { x } - \mathbf { t } )$ , where � and � are the rotation and translation associated with �.

We use � as residual distance: the radial distance which projects $\mathbf { x } ^ { \prime }$ onto the surface along the ray from the primitive center, measuring how much the point must be scaled to satisfy $F ( \mathbf { x } ^ { \prime } , \Lambda ) = 1$ . Since $F ( t { \bf x } ^ { \prime } , \Lambda ) = t ^ { 2 / \varepsilon _ { 1 } } F ( { \bf x } ^ { \prime } , \Lambda )$ for $t > 0$ , the point on the surface lying along the ray through $\mathbf { x } ^ { \prime }$ is $t ^ { \star } \mathbf { x } ^ { \prime }$ with $t ^ { \star } = F ( { \bf x } ^ { \prime } , \Lambda ) ^ { - \varepsilon _ { 1 } / 2 }$ . The radial distance residual is then given in closed form by the positive quantity

$$
r ( \mathbf { x } , \theta ) = \| \mathbf { x } ^ { \prime } \| \Big | 1 - F ( \mathbf { x } ^ { \prime } , \Lambda ) ^ { - \varepsilon _ { 1 } / 2 } \Big | ,\tag{2}
$$

where $\begin{array} { r } { \mathbf { x } ^ { \prime } = R ^ { \top } ( \mathbf { x } \mathrm { ~ - ~ } \mathbf { t } ) } \end{array}$ . The signed quantity $\| \mathbf { x } ^ { \prime } \| \big ( \mathrm { 1 ~ - ~ }$ $F ( \mathbf { x } ^ { \prime } , \Lambda ) ^ { - \varepsilon _ { 1 } / 2 } )$ is positive outside the surface and negative inside it.

This formulation is geometrically intuitive and accurate for convex shapes, but becomes less reliable for highly nonconvex configurations $( \varepsilon _ { 1 } \mathrm { o r } \varepsilon _ { 2 } \gg 1 )$ , where the projection may deviate from the true closest point.

## 3. Related Work

We review two closely related research directions. On one side, our work addresses the general problem of robust model estimation, which is traditionally studied within the RANSAC paradigm and its numerous extensions. On the other hand, it tackles the task of geometric primitive fitting, aiming to approximate complex shapes exploiting parametric structures such as planes, cylinders, or superquadrics.

Robust model fitting. Robust model estimation is commonly formulated as a consensus maximization problem, where the goal is to identify a model hypothesis $h _ { j }$ that maximizes the size of its inlier set

$$
{ \cal I } _ { j } = \{ x _ { i } \in { \cal D } \mid r ( x _ { i } , h _ { j } ) < \varepsilon \} ,\tag{3}
$$

defined in terms of a residual function $r ( \cdot )$ and inlier threshold $\varepsilon .$

The RANSAC paradigm [10], originally introduced in the context of model fitting for cartographic data, addresses this problem by iteratively sampling minimal subsets of the data and selecting the hypothesis that yields the largest consensus. Since its introduction, RANSAC has become a cornerstone of robust estimation, particularly in computer vision tasks such as image matching, visual localization, and 3D reconstruction.

Several extensions have been proposed to improve the quality of the estimated models. LO-RANSAC [7] incorporates a local optimization step that iteratively refines the model using the current inlier set. GC-RANSAC [4] further extends this approach by introducing spatial regularization through a graph-cut formulation. In this setting, model fitting can be interpreted as a labelling problem, where each point is assigned to either the inlier or outlier class.

More formally, GC-RANSAC defines an energy function composed of a data fidelity term $E _ { 1 }$ , based on the residual $r ( \mathbf { x } _ { i } , h _ { j } ) .$ , and a smoothness term $E _ { 2 }$ , which promotes spatial coherence by encouraging neighboring points with low residuals to be jointly selected as inliers.

As a result, the initial consensus set $\boldsymbol { \mathit { I } } _ { j }$ is refined into a spatially coherent set, as illustrated in Figure $2 ( \mathrm { c } )$ . This proximity propagation regularizes the estimation and is particularly efective in settings where proximity is a reliable proxy for model membership, such as feature matching and two-view geometry. However, as discussed in Section 1, this assumption becomes less informative in 3D primitive decomposition, where spatial proximity does not necessarily reflect the underlying surface structure.

Multi-model fitting. Multi-model fitting extends consensus maximization to the simultaneous estimation of multiple structures, requiring both model selection and data segmentation. Formally, this corresponds to jointly estimating a set of models $\{ \theta _ { k } \} _ { k = 1 } ^ { \kappa }$ and assigning each point $x _ { i } \in D$ to one of them, or to an outlier class.

Clustering-based approaches, such as T-Linkage [16], group points according to shared model preferences, even considering multiple classes of parametric models [18, 19]. These methods naturally produce a partition of the data , but are typically greedy and often require additional postprocessing to handle the presence of overlapping structures. A related family of methods performs hierarchical segmentation, progressively merging or splitting surface regions to build a multi-resolution decomposition rather than committing to a single partition. Attene et al. [1] recover a hierarchical structure of point-sampled surfaces, while Zhang et al. [32] perform hierarchical mesh segmentation guided by quadric surface fitting. These methods do not require the number of segments to be specified in advance. Because their merging or splitting decisions are typically greedy, early local decisions may produce suboptimal boundaries. Optimization-based methods instead formulate multi-model fitting as a global energy minimization problem. Approaches such as PEARL [13], PROG-X [3], and MULTI-X [2] jointly estimate model assignments by balancing a data fidelity term, typically expressed through residuals $r ( \mathbf { x } _ { i } , h )$ , with regularization terms enforcing spatial coherence. In particular, PEARL constructs a graph over the data points and optimizes the assignment via graph-cut techniques (e.g., ���ℎ�- expansion), similarly to other energy-based formulations. These approaches encourage neighboring points to share the same assignment, efectively promoting spatial consistency in the inferred inlier sets $\tau _ { j }$ . However, the resulting assignments are typically hard and remain primarily driven by residuals and proximity, which can lead to incorrect labeling in regions where multiple primitives intersect or lie in close spatial proximity.

RANSACOV [17] takes a diferent perspective by formulating model selection as a maximum coverage problem, selecting a subset of candidate models that maximizes the number of explained points. While this formulation can better handle overlapping structures, it still relies on residualbased inlier definitions.

Despite their diferences, all these approaches fundamentally rely on residual-based criteria and spatial coherence, which may lead to incorrect assignments in the presence of adjacent or intersecting primitives, where proximity does not reflect the underlying surface structure.

Primitivefitting. Primitive fitting can be seen as a specific instance of model fitting in which the goal is to approximate complex shapes using parametric primitives such as planes, cylinders, or superquadrics. Among the possible primitives, superquadrics have attracted significant attention for their ability to represent a wide range of shapes with a relatively small number of parameters.

Early works introduced superquadrics for modeling objects in range images using least-squares fitting [26], and explored diferent error measures �(�, �) for single superquadric recovery [12]. Notably, Schnabel et al. [25] proposed an eficient RANSAC-based approach for primitive detection that exploits surface normals, using them for inlier validation through both distance and angular thresholds. In particular, a point $\mathbf { x } _ { i }$ is considered an inlier of a superquadric $h _ { j }$ if it satisfies both a residual constraint and an angular consistency constraint:

$$
\begin{array} { r } { T _ { j } = \{ \mathbf { x } _ { i } \in \cal D |  r ( \mathbf { x } _ { i } , h _ { j } ) < \varepsilon , \operatorname { a r c c o s } ( | \mathbf { n } _ { i } ^ { \top } \mathbf { n } _ { h _ { j } } ( \mathbf { x } _ { i } ) | ) < \alpha  \} } \end{array}
$$

where ${ \bf n } _ { h _ { i } } ( { \bf x } _ { i } )$ denotes the unit normal of model $h _ { j } ,$ evaluated at the projection of $\mathbf { x } _ { i }$ onto the model surface, and � is the angular threshold. In contrast, our formulation incorporates normal information directly into the optimization through both unary penalties and pairwise consistency terms. Consequently, geometric information actively influences inlier selection rather than being used solely as a postfitting validation criterion.

Subsequent approaches explored alternative estimation strategies. Solina et al. [14] proposed a recover-and-select paradigm, where models are iteratively refined by progressively adding nearby points and then evaluated through an objective function. Liu et al. [15] introduced a probabilistic framework based on Expectation Maximization (EM), explicitly modeling noise and outliers to improve robustness.

More recent methods have investigated diferent directions. Monnier et al. [21] proposed Diferentiable Blocks World, which optimizes textured superquadric primitives from multi-view images via diferentiable rendering, achieving high reconstruction fidelity at the cost of significant computational complexity. Ramamonjisoa et al. [24] introduced a stochastic search method for fitting primitives to noisy point clouds, handling missing data and noise without requiring training data, though it is limited by the use of predefined primitive types such as cuboids.

Learning-based approaches have also gained momentum, proposing data-driven methods that infer collections of 3D primitives (e.g., superquadrics [23, 22] or cuboids [28]) from clean training data. While promising, these approaches typically require large annotated datasets and may struggle to generalize across diferent domains or primitive families. In contrast, our work follows a purely algorithmic direction and builds upon our previous formulation [9], extending it to the multi-model fitting setting and enabling robust primitive decomposition in complex scenes.

## 4. Method

Our approach is built around a geometric-aware inlier refinement strategy (GAIR), which serves as the core component of the proposed framework. We develop this idea at three complementary levels.

First, in Section 4.1 we introduce GAIR as a standalone inlier refinement module, formulated as an energy minimization problem that integrates geometric priors such as normal consistency and local surface coherence. This component improves the quality of inlier sets associated with individual model hypotheses.

Second, we embed GAIR within a consensus maximization pipeline, resulting in GAIR-RANSAC, a robust estimator for single-model superquadric fitting as detailed in Section 4.2. This integration improves both the stability of model estimation and the convergence of the RANSAC procedure.

Finally, in Section 4.3, we extend the framework to the multi-model setting through GAIR-RANSACOV. In this case, GAIR generates a pool of candidate hypotheses, which are subsequently selected via a global Maximum Coverage formulation to obtain a consistent primitive decomposition. The following sections describe these components in detail.

## 4.1. Geometric-Aware Inlier Refinement

Consensus maximization approaches generate a pool of candidate models $\{ h _ { j } \}$ by sampling subsets $\mathcal { M } _ { i } \subset \mathcal { D }$ and fitting a geometric model to each sample. The final solution consists of selecting the subset  of hypotheses that best explain the data in terms of their associated inlier sets (see Section 3). As illustrated in Figure 3, given a sampled subset $\mathcal { M } _ { j } , \mathtt { a }$ model $h _ { j }$ is estimated and an initial set of inliers $\tau _ { j }$ is obtained as in (3) through residual thresholding. While this step efectively identifies points that are close to the model, it does not explicitly account for the underlying surface structure.

![](images/de4962f7d818787b740454b426c698765fe771cddc94bf849b101819efe6ee56.jpg)  
Figure 3: Our geometric-aware refinement in a nutshell. Given a candidate model $h _ { j }$ and its initial inlier set $\boldsymbol { \mathit { 1 } } _ { j } ,$ GAIR produces a refined set $\widehat { I } _ { j }$ that better adheres to the underlying surface geometry.

The goal of our Geometric-Aware Inlier Refinement (GAIR) is to refine this inlier set. Starting from a candidate model $h _ { j }$ and its initial inliers $\boldsymbol { \mathit { I } } _ { j }$ , we compute a refined set $\widehat { I } _ { j }$ that better adheres to the true geometric structure of the data. In particular, the refinement step removes geometrically inconsistent points and enforces local coherence. As shown in the example, the refined model better adheres to the cap of the mushroom. This leads to more accurate model estimates and provides a favorable accuracy–runtime tradeof within the fitting pipeline, as reported in our experiments (see Figure 6).

The key observation is that inlier sets should be coherent at the surface level: points belonging to the same primitive are expected not only to have low residuals, but also to exhibit consistent local geometry, such as aligned normals. To formalize this idea, we reinterpret the inlier set associated with the �-th model as a binary labeling over the point cloud:

$$
f _ { j } : \mathcal { D }  \{ 0 , 1 \} ,\tag{4}
$$

where $f _ { j } ( { \bf { x } } ) \ = \ { \mathrm { ~ 1 ~ } }$ if � is classified as an inlier, and 0 otherwise.

Given the initial labeling induced by $\tau _ { j }$ , our objective is to compute a refined labeling $\hat { f } _ { j }$ , defining an updated inlier set:

$$
{ \hat { T } } _ { j } = \{ \mathbf { x } \in D \mid { \hat { f } } _ { j } ( \mathbf { x } ) = 1 \} .\tag{5}
$$

Building upon the energy minimization framework of GC-RANSAC, we cast this refinement as a labeling problem over a graph defined on the point cloud , where both data fidelity and geometric consistency are taken into account.

## 4.1.1. Energy Formulation

For clarity, we omit the model index and consider a generic labeling function $f .$ We denote by $f _ { p } \in \{ 0 , 1 \}$ the label assigned to a point �, i.e., $f _ { p } = f ( { \bf p } )$

We define the energy associated with a labeling � over the point cloud graph as:

$$
E ( f ) = \sum _ { { \bf p } \in D } E _ { 1 } ( f _ { p } ) + \sum _ { ( { \bf p } , { \bf q } ) \in \mathcal { E } } E _ { 2 } ( f _ { p } , f _ { q } ) ,\tag{6}
$$

where  denotes the set of edges connecting neighboring points (details on how to construct the graph are provided in Sec.4.1.2).

The energy consists of two terms:

$E _ { 1 }$ is a unary term encoding data fidelity, measuring how well each point agrees with the model in terms of both geometric residual and normal orientation. This corresponds to the standard residual-based criterion used in RANSAC.

$E _ { 2 }$ is a pairwise term promoting coherence between neighboring points.

Unary term (datafidelity). The unary term measures how well each point fits the model:

$$
E _ { 1 } ( f _ { p } ) = \left\{ { \begin{array} { l l } { { \bar { d } } _ { p } + { \displaystyle { \frac { 1 } { 2 } } } ( 1 - { \bf n } _ { p } \cdot { \bf n } _ { h _ { j } } ( { \bf p } ) ) } & { { \mathrm { i f ~ } } f _ { p } = 1 , } \\ { 1 , } & { { \mathrm { i f ~ } } f _ { p } = 0 . } \end{array} } \right.\tag{7}
$$

Here, $\hat { \mathbf { n } } _ { p }$ denotes the normalized observed unit normal associated with point �, while $\hat { \mathbf { n } } _ { h _ { i } } ( \mathbf { p } )$ denotes the unit surface normal predicted by model $h _ { j }$ at � and where

$$
\bar { d } _ { p } = \operatorname* { m i n } \left( \frac { | r ( \mathbf { p } , h _ { j } ) | } { \varepsilon } , 1 \right) .\tag{8}
$$

is the residual of an inlier from model $h _ { j }$ normalized by the inlier threshold � and truncated to 1. This normalization ensures that all residuals lie in [0, 1], which is crucial to balance the unary and pairwise contributions in the energy. In addition, truncation limits the influence of outliers and guarantees bounded costs, a desirable property for stable graph-cut optimization.

Pairwise term (geometric consistency). The pairwise term encourages neighboring points to share the same label when they are geometrically compatible. We first define a normal-based coherency measure between neighboring points � and �:

$$
C ( \mathbf { n } _ { p } , \mathbf { n } _ { q } ) = \frac { 1 } { 2 } ( 1 + \mathbf { n } _ { p } \cdot \mathbf { n } _ { q } ) ,\tag{9}
$$

which lies in [0, 1] and captures the alignment between surface normals.

A basic formulation penalizes label disagreement proportionally to this coherency:

$$
E _ { 2 } ( f _ { p } , f _ { q } ) = { \left\{ \begin{array} { l l } { C ( \mathbf { n } _ { p } , \mathbf { n } _ { q } ) , } & { { \mathrm { i f ~ } } f _ { p } \neq f _ { q } , } \\ { 0 , } & { { \mathrm { i f ~ } } f _ { p } = f _ { q } . } \end{array} \right. }\tag{10}
$$

This encourages neighboring points with aligned normals to share the same label, promoting surface-consistent inlier sets. To improve robustness, we further modulate the pairwise term using the point-to-model residuals. Let $\bar { d } _ { p }$ and $\bar { d } _ { q }$ denote the normalized distances of points � and � from the model. The final pairwise term is defined as:

$$
E _ { 2 } ( f _ { p } , f _ { q } ) = \left\{ \begin{array} { l l } { C ( \mathbf { n } _ { p } , \mathbf { n } _ { q } ) , } & { \mathrm { i f ~ } f _ { p } \neq f _ { q } , } \\ { 0 , } & { \mathrm { i f ~ } f _ { p } = f _ { q } = 1 , } \\ { \left( 1 - \frac { \bar { d } _ { p } + \bar { d } _ { q } } { 2 } \right) C ( \mathbf { n } _ { p } , \mathbf { n } _ { q } ) , } & { \mathrm { i f ~ } f _ { p } = f _ { q } = 0 , } \end{array} \right.\tag{11}
$$

where $\bar { d } _ { p }$ is defined as:

$$
\bar { d } _ { p } = \operatorname* { m i n } \left( \frac { | r ( p , h _ { j } ) | } { \lambda \varepsilon } , 1 \right) ,\tag{12}
$$

where � is an outlier scale factor. The term $\bar { d } _ { q }$ is defined accordingly. In our implementation we set $\lambda = 3 .$

This formulation of the pairwise term introduces two complementary efects: on the one hand neighboring points with aligned normals are encouraged to share the same label. On the other hand, points far from the model have a reduced influence on the labeling, preventing outliers from enforcing incorrect smoothness. The distances $\bar { d } _ { p }$ and $\bar { d } _ { q }$ are normalized and bounded to ensure that the pairwise term remains within [0, 1].

Moreover, the pairwise term (11) satisfies the submodularity condition required by graph-cut optimization:

$$
\left( 1 - \frac { \bar { d } _ { p } + \bar { d } _ { q } } { 2 } \right) C ( \mathbf { n } _ { p } , \mathbf { n } _ { q } ) \leq 2 C ( \mathbf { n } _ { p } , \mathbf { n } _ { q } ) ,\tag{13}
$$

since the residual term is bounded in [0, 1].

In practice, we construct the graph by retaining only edges whose normal coherence satisfies

$$
C ( \mathbf { n } _ { p } , \mathbf { n } _ { q } ) > 0 . 9 .\tag{14}
$$

Edges with $C ( \mathbf { n } _ { p } , \mathbf { n } _ { q } ) \quad \leq \quad 0 . 9$ are discarded by setting their weights to zero. This choice retains connections only between neighboring points whose normals difer by less than $3 7 ^ { \circ }$ . This conservative graph construction restricts the smoothness prior to locally coherent surface regions.

Remark. The pairwise term $E _ { 2 }$ is the only component of the proposed energy that couples the labels of neighboring points. Indeed, if $E _ { 2 } \equiv 0 , \mathrm { E q } .$ . (6) reduces to the purely unary formulation

$$
E ( f ) = \sum _ { { \bf p } \in D } E _ { 1 } ( f _ { p } ) ,\tag{15}
$$

which can be minimized independently for each point. The resulting labeling, however, does not generally coincide with the consensus set of vanilla RANSAC, since the proposed unary term evaluates both the point-to-model residual and the agreement between the observed and model normals. Standard residual-based RANSAC is recovered only as the further special case in which the normal-consistency contribution is removed from $E _ { 1 }$

If the pairwise term is defined using only residual-based quantities, without normal-based geometric consistency, it reduces to the proximity based formulation used to refine the inlier sets in GC-RANSAC. In this case, the pairwise term can be written as:

$$
E _ { 2 } ^ { \mathrm { G C } } ( f _ { p } , f _ { q } ) = \left\{ \begin{array} { l l } { 1 , } & { \mathrm { i f ~ } f _ { p } \neq f _ { q } , } \\ { \frac 1 2 ( \bar { d } _ { p } + \bar { d } _ { q } ) , } & { \mathrm { i f ~ } f _ { p } = f _ { q } = 1 , } \\ { 1 - \frac 1 2 ( \bar { d } _ { p } + \bar { d } _ { q } ) , } & { \mathrm { i f ~ } f _ { p } = f _ { q } = 0 , } \end{array} \right.\tag{16}
$$

Our GAIR refinement extends this framework by incorporating normal-based geometric consistency and residualaware weighting within the pairwise term, enabling surfaceaware inlier refinement.

## 4.1.2. Graph Construction and Optimization

To minimize the energy in Eq. (6), we formulate the problem as a binary labeling task on a graph  and solve it using the graph-cut algorithm.

Graph construction. We construct an undirected graph $\mathcal { G } = ( D , \mathcal { E } )$ from the input point cloud. Each node $p \in \mathcal { D }$ corresponds to a point in , and edges $( p , q ) \in { \mathcal { E } }$ connect neighboring points. The neighborhood structure is defined locally: for each point �, we identify a set of neighboring points using a spatial criterion (e.g., �-nearest neighbors). This results in a sparse graph capturing the local geometric structure of the data. The parameter � controls the efective spatial support of the regularization: for a fixed value of �, a higher sampling density results in a smaller neighborhood radius. Increasing � therefore helps preserve stable local connectivity and reduces sensitivity to non-uniform sampling, while keeping the neighborhood suficiently local to avoid connections across distinct surfaces. In our implementation, the graph is built using a KD-tree-based �- nearest-neighbor search. We use $k \ = \ 6$ in the controlled experiments, whereas for the more densely sampled real scans we use a slightly denser graph with $k = 1 0$ . Each node is associated with unary costs derived from Eq. (7), while each edge is assigned a pairwise cost according to Eq. (11). Normal vectors ${ \bf n } _ { p }$ are either provided or estimated from the local neighborhood. For scanned point clouds, normals are estimated with Open3D using a 90-nearest-neighbor local neighborhood. The estimated normals are then oriented through consistent tangent-plane propagation, and globally flipped when necessary to obtain an outward orientation.

graph-cut optimization. The energy minimization problem is solved via a min-cut/max-flow procedure. The graph is augmented with two terminal nodes (source and sink), representing the inlier and outlier labels, respectively.

Algorithm 1 GAIR   
Input: point cloud ${ \overline { { \cal D } } } ,$ normals , candidate model $h _ { j }$   
initial inlier set $\boldsymbol { \mathit { I } } _ { j }$   
neighborhood graph $\mathcal { G } = ( D , \mathcal { E } )$   
Output: refined inlier set $\widehat { \cal I } _ { j }$   
1: Get the initial labeling $f _ { j }$ from $\boldsymbol { \mathit { 1 } } _ { j }$ using (4)   
2: Compute unary cost $E _ { 1 }$ using $( 7 )$   
3: Compute pairwise costs $E _ { 2 }$ using (11)   
4: $\widehat { f } _ { j } = g r a p h C u t ( D , \mathcal { G } , E _ { 1 } , E _ { 2 } )$   
5: Compute $\widehat { \cal I } _ { j }$ using (5)   
6: return $\widehat { \cal I } _ { j }$

Pairwise terms define the weights of edges between neighboring nodes.

The minimum cut partitions the graph into two disjoint sets:

• nodes connected to the source correspond to inliers $( f ( p ) = 1 )$

• nodes connected to the sink correspond to outliers $( f ( p ) = 0 )$

This yields the optimal labeling $\hat { f }$ that minimizes the energy in Eq. (6). The overall refinement procedure is summarized in Algorithm 1.

While the worst-case complexity of graph-cut is high, in practice it performs eficiently on sparse graphs arising from point cloud neighborhoods. The locality of the graph and the bounded energy terms lead to fast convergence, making the refinement step computationally tractable within a consensus maximization pipeline.

## 4.2. GAIR-RANSAC: Single-Model Superquadric Fitting

We integrate the proposed Geometric-Aware Inlier Refinement (GAIR) into a RANSAC framework for the extraction of a single superquadric from a point cloud . The resulting procedure, summarized in Algorithm 2, follows the classical hypothesize-and-verify paradigm, while introducing several adaptations to cope with the challenges of superquadric fitting.

At a high level, the algorithm iteratively samples, estimates, and refines a pool of � candidate models, selecting the one $\theta ^ { * }$ that best explains the data in terms of its inlier support.

At each iteration �, a subset $\mathcal { M } _ { i } \subset \mathcal { D }$ is sampled (line 14) and used to estimate a model $h _ { j }$ (line 16). While in standard RANSAC minimal samples are suficient for closed-form models, superquadric estimation requires solving a nonlinear optimization problem. As a result, minimal samples are often unstable and lead to poor initializations. To address this issue, we adopt a local geometric sampling strategy that favors surface coherence. At each trial, a seed point is first sampled randomly from . Its normal vector is then used to define a local tangent plane, which induces a geometrically meaningful neighborhood. Specifically, we consider as candidate points those lying within the �-nearest neighbors of the seed and whose distance from the tangent plane is below a tolerance �, thus restricting the pool to points that locally approximate the same surface patch. Finally, this set is further pruned by enforcing normal consistency, discarding points whose normals are not suficiently aligned with that of the seed according to a cosine similarity threshold �.

From this pool, multiple candidate sample sets are generated and evaluated. Each candidate is constructed using Farthest-Point Sampling (FPS), initialized at the seed point, in order to ensure suficient spatial coverage while remaining within a local surface patch. Since superquadric fitting is computationally expensive, we assess the quality of each candidate sample before model estimation. In particular, each sample is assigned a score that balances spatial compactness and geometric coherence. The sample achieving the best score is selected as $\mathcal { M } _ { j }$

Given the selected sample, the model $h _ { j }$ is estimated via non-linear least squares, initialized using PCA to provide a stable starting point. Residuals are computed as the radial distance to the superquadric surface.

Once a candidate model is obtained, an initial inlier set $\tau _ { j }$ is computed via residual thresholding. If the current hypothesis improves over the best-so-far solution (line 18), a local optimization step is triggered. Starting from $( h _ { j } , I _ { j } )$ we iteratively refine the inlier set using GAIR (line 22) and update the model parameters through an inner RANSAC procedure.

The refinement loop continues until convergence, i.e., when no further improvement in the consensus is observed (lines 22–29). The final output is the model $h ^ { * }$ with the highest refined support.

## 4.3. GAIR-RANSACOV: Primitive Decomposition

We now extend the proposed framework to the primitive decomposition, where the goal is to decompose a point cloud into multiple superquadric hypotheses $h _ { 1 } , \ldots , h _ { \kappa } ,$ parameterized respectively by $\theta _ { 1 } , \ldots , \theta _ { \kappa }$ . Unlike the single-model case, the multi-model problem requires disentangling multiple overlapping structures. We address this by decoupling hypothesis generation from model selection: first, we generate a pool of candidate models, and then we select the subset that best explains the data through a global optimization procedure that solves a Maximum Coverage problem [17]: select the set of � superquadrics that explain most of the points in .

Superquadric hypothesis generation strategy. Generating a large pool of superquadric hypotheses by purely random sampling would be computationally impractical. The superquadric fitting is expensive and highly sensitive to initialization, while the subsequent Maximum Coverage optimization becomes more demanding as the number of candidate models increases. For this reason, we adopt a structured strategy that produces a small set of high-quality hypotheses.

Algorithm 2 GAIR-RANSAC   
Input: point cloud , threshold �,   
maximum number of iterations �   
Output: model parameters $\theta ^ { * }$   
1: ${ { T } ^ { * } } = \varnothing , { { h } ^ { * } } = \varnothing$   
2: $\mathcal { G } = i n i t G r a p h ( D ) , \mathcal { V } = i n i t N o r m a l s ( \mathcal { D } )$   
3: for $j = 1  m$ do   
4: /\* Local geometric sampling \*/   
5: $p _ { s e e d } \sim D$   
6: $\mathcal { P } _ { l o c a l } = g e t N e i g h b o r h o o d ( p _ { s e e d } )$   
7: $\mathcal { P } _ { f i l t e r e d } = f i l t e r B y N o r m a l s ( \mathcal { P } _ { l o c a l } , \mathfrak { V } )$   
8: $/ \ast$ Candidate sample generation \*/   
9: $\{ \mathcal { M } _ { i } ^ { k } \} = g e n e r a t e C$ andidates $( \mathcal { P } _ { f i l t e r e d } )$   
10: /\* Sample scoring \*/   
11: for each candidate $\mathcal { M } _ { j } ^ { k }$ do   
12: $s _ { k } = s c o r e S a m p l e ( \mathcal { M } _ { j } ^ { k } )$   
13: end for   
14: $\mathcal { M } _ { j } = \arg \operatorname* { m a x } _ { k } s _ { k }$   
15: /\* Model estimation \*/   
16: $h _ { j } = f u t { \mathrm { S } } u p e r q u a d r i c ( { \mathcal { M } } _ { j } , { \mathrm { P C A } }$ init)   
17: $\bar { I _ { j } } = c o m p u t e C o n s e n s u s ( h _ { j } , \varepsilon )$   
18: if $| T _ { j } | > | T ^ { * } |$ then   
19: ��������� = False   
20: $I = I _ { j } , h = h _ { j }$   
21: while not ��������� do   
22: $\underset { \hat { \mathbf { \phi } } } { \widehat { T } } = \mathbf { G } \mathbf { A } \mathbf { I } \mathbf { R } ( \mathcal { D } , \mathcal { V } , h , T , \mathcal { G } ) _ { j }$   
23: $\widehat { h } , \widehat { c } = i n n e r R A N S A C ( D , \widehat { I } , I , \varepsilon )$   
24: if compareConsensus(, ̂�) then   
25: $\widehat { \cal I } = \widehat { c } , h = \widehat { h }$   
26: else   
27: ��������� = True   
28: end if   
29: end while   
30: $T ^ { * } = T , h ^ { * } = h$   
31: end if   
32: end for   
33: return $h ^ { * }$

We use Sequential GAIR-RANSAC as a structured hypothesis generator (Algorithm 3), rather than as a final decomposition strategy. At each run (lines 2–14), models are extracted sequentially: once a model ℎ is estimated using GAIR-RANSAC (line 5), its inliers are identified and removed from the current point set, and the procedure is repeated on the residual data until no suficiently supported structure remains.

To improve the quality of the extracted hypotheses and remove spurious support, we adopt a more aggressive inlier removal strategy (line 11). After extracting a model, we remove not only its inliers, but also points in a neighborhood defined by a dilated threshold $\beta \cdot \varepsilon .$ , with $\beta ~ > ~ 1$ This suppresses residual support around already-explained structures, and can be interpreted as a mechanism to enforce diversity in the hypothesis pool, preventing multiple hypotheses from explaining the same local support.

Once a sequential run terminates, all points are restored (line 13) and a new run is started from the original point cloud. To increase variability across runs, each run operates on a random subsampling of the input data (line 3).

To control the size and quality of the hypothesis pool, we incorporate a coverage validation step (line 6). After each extraction, a model is accepted only if its surface is suficiently supported by the inlier set. In practice, we sample points uniformly on the estimated superquadric surface and compute the fraction of samples lying within distance � from the data. Models that fail to meet a minimum coverage threshold $\tau _ { \mathrm { c o v } }$ are discarded (line 8). This step reduces the number of redundant hypotheses, which is crucial since the complexity of the Maximum Coverage problem grows with the size of the hypothesis pool.

Repeating the sequential extraction process over multiple runs (lines 2–14) yields a hypothesis pool

$$
\mathcal { H } = \{ h _ { 1 } , \ldots , h _ { M } \} ,
$$

containing models that are generally well supported but still redundant, as the same underlying structure may be extracted multiple times with slightly diferent parameters.

Primitive selection via maximum coverage. Given the hypothesis pool , each model $h _ { j }$ defines a geometricaware consensus set $\widehat { \cal I } _ { j }$ obtained via GAIR. The goal is to select a subset of models that jointly explains the largest number of points. We cast this task as a Maximum Coverage problem [17]. Given a budget of � models, the objective is to select at most � hypotheses whose union of inlier sets maximizes the number of explained points:

$$
\operatorname* { m a x } _ { S \subseteq \mathcal { H } , \ | S | \leq \kappa } \left. \bigcup _ { h _ { j } \in S } \widehat { T } _ { j } \right. .\tag{17}
$$

The selected set  provides the desired primitive decomposition.

## 5. Experiments

We evaluate the proposed geometric-aware inlier refinement (GAIR) by analyzing its behavior in both single-model superquadric fitting and multi-model primitive decomposition scenarios. Specifically, we assess its integration within our GAIR-RANSAC and GAIR-RANSACOV on synthetic and realistic point clouds under controlled noise and outlier conditions. The code is publicly available at https://github .com/Tededo02/3D\_superquadric\_decomposition/tree/gair\_ra nsac.

## 5.1. Datasets

For our evaluation, we consider three complementary types of data: (i) controlled synthetic benchmarks, (ii) challenging synthetic configurations, and (iii) real-world shapes. An overview is reported in Figure 4. All shapes are normalized to a common scale to ensure that evaluation metrics, such as the Chamfer distance and Hausdorf distance, are comparable.

![](images/214158fdfd318057746d33e718a2096e9cb9118d4ac25f7ad58008d690da7842.jpg)  
Figure 4: Overview of the datasets used in our experiments. From left to right: synthetic TangentSuperquadrics point clouds, used to evaluate pure fitting accuracy; SqSoup in both the single-model and multi-model configuration, used to test the algorithm in a setup with few points; a subset of CAD-like shapes from Thingi10K, showcasing the applicability of our method to more complex and realistic geometries, and, finally, a set of four 3d scans from Sketchfab. We also considered real LiDAR data reported in Fig. 11

Algorithm 3 GAIR-RANSACOV   
Input: point cloud , threshold distance $\varepsilon ,$   
number of models $\kappa ,$ number of runs $R ,$   
coverage threshold $\tau _ { \mathrm { { c o v } } } .$ bufer factor $\beta$   
Output: set of model parameters $\{ \theta _ { k } \} _ { k = 1 } ^ { \kappa }$   
1: $\mathcal { H } = \emptyset$   
2: for $r = 1  R$ do   
3: $D _ { c u r r } =$ subsample()   
4: while $D _ { c u r r } \neq$ ∅ do   
5: $h , \widehat { I } = \mathbf { G } \mathbf { A } \mathbf { I } \mathbf { R } \mathbf { - } \mathbf { R } \mathbf { A } \mathbf { N } \mathbf { S } \mathbf { A } \mathbf { C } ( \mathcal { D } _ { c u r r } , \varepsilon )$   
6: $c = c o \nu e r a g e C h e c k ( h , \widehat { I } , \varepsilon )$   
7: $\mathbf { i f } \ c < \tau _ { \mathrm { c o v } }$ then   
8: break   
9: end if   
10: $\mathcal { H } = \mathcal { H } \cup \{ ( h , \widehat { I } ) \}$   
11: $D _ { c u r r } = D _ { c u r r } \setminus d i l a t e ( \widehat { T } , \beta \varepsilon )$   
12: end while   
13: $\mathcal { D } _ { { c u r r } } = \mathcal { D }$   
14: end for   
15: $\{ \theta _ { k } \} _ { k = 1 } ^ { \kappa } = s o l \nu e M a x { C o \nu e r a g e ( \mathcal { H } , \kappa ) }$   
16: return $\operatorname { \ a u } _ { \{ \theta _ { k } \} } \kappa _ { k = 1 }$

SqSoup (synthetic). The SqSoup dataset is a collection of synthetic point clouds specifically designed for superquadric fitting evaluation. We consider two subsets: a SqSoup singlemodel split, containing point clouds sampled from individual superquadrics with varying shape parameters, and a SqSoup multi-model split, comprising composite scenes constructed from multiple interacting superquadrics (e.g., cat-like shapes and a peanut). These multi-model scenes are particularly useful for stress-testing primitive decomposition algorithms, as the constituent primitives exhibit partial overlap and ambiguous spatial configurations.

TangentSuperquadrics. We introduce synthetic scenes where multiple superquadrics are placed close to each other. These configurations are particularly challenging for primitive decomposition, as adjacent primitives exhibit minimal spatial separation while remaining geometrically distinct.

In such cases, methods relying solely on point-to-model distance or spatial proximity tend to incorrectly merge neighboring primitives into a single model. These examples therefore act as a stress-test for the inlier refinement step.

Thingi10K subset (3D shapes). To assess generalization beyond ideal superquadric configurations, we include four shapes from the Thingi10K repository: a mushroom, a bird, a hammer, and an anthropomorphic cartoon character. These models exhibit irregular geometry, non-convex surfaces, and structures that only loosely resemble superquadric primitives, providing a more realistic evaluation scenario.

Sketchfab scans. To evaluate GAIR beyond synthetic and CAD-like benchmarks, we consider four publicly available real-world 3D scans downloaded from Sketchfab: a baptismal font [20], a fire hydrant [6], a film camera [11], and a hammer [27]. The models were acquired using either photogrammetry or dedicated 3D scanning equipment. Since ground-truth primitive decompositions are not available, performance on these data is assessed qualitatively. The downloaded meshes were converted into point clouds by uniformly subsampling 8000 surface points.

LiDAR scans (real data). We additionally acquired three point-cloud scenes using the LiDAR sensor of an iPhone 16 Pro using PointCloud Scanner: a ball placed on top of a box, a sofa with cushions, and a plush toy. The three scenes provide increasing levels of dificulty. The ball-and-box scene represents a simple controlled configuration composed of two clearly separated primitives. The sofa scene contains adjacent and partially overlapping structures, with limited visibility of some cushions. Finally, the plush-toy scan is characterized by a noisy fuzzy surface and weak geometric separation between its constituent parts.

## 5.2. Metrics

We evaluate the performance of the proposed method along three complementary dimensions: inlier-outlier classification accuracy, geometric accuracy, and computational eficiency.

Inlier-outlier classification performance is measured using the Inlier Assignment Error (IAE), which quantifies the fraction of points incorrectly classified as either inliers or outliers. Points mi belonging to the target models are labeled as inliers, whereas the explicitly introduced outlier points are labeled as outliers. Let $y _ { i } \in \{ 0 , 1 \}$ denote the groundtruth binary label of point $\mathbf { x } _ { i } ,$ and let $\hat { y } _ { i }$ denote its predicted label, where 1 indicates an inlier and 0 an outlier. The IAE is defined as:

$$
\mathrm { I A E } = \frac { 1 } { N } \sum _ { i = 1 } ^ { N } \mathbb { I } \big [ \hat { y } _ { i } \neq y _ { i } \big ] ,\tag{18}
$$

where lower values indicate better classification performance. Therefore, the IAE enables a consistent evaluation of inlier–outlier classification across all datasets without requiring each point to be associated with a specific groundtruth primitive.

Geometric fidelity is assessed using the Chamfer Distance (CD), which captures the average discrepancy between the reconstructed and ground-truth surfaces, and the HausdorfDistance (HD), which reflects the worst-case deviation. For the Thingi10K, SqSoup datasets, we report the Normalized Chamfer Distance and the Normalized HausdorfDistance, computed after normalizing the point-cloud bounding boxes to ensure that metric values are comparable across shapes within the same experiment.

Finally, we report execution time and the number of local optimization steps (convergence), providing insight into the computational cost and behavior of the diferent methods.

## 5.3. Baselines and evaluation protocols

We compare the proposed method against three wellestablished variants of the RANSAC paradigm, which can be interpreted as progressively enriching the inlier refinement process.

Vanilla RANSAC [10] represents the baseline formulation, where inliers are selected solely based on a residual threshold, without any refinement. LO-RANSAC [7] introduces a local optimization step that refines the inlier set using residual-based criteria, improving model estimation without incorporating additional structural information. GC-RANSAC [4] further extends this formulation by introducing spatial coherence through a graph-cut optimization, promoting inlier assignments that are locally consistent in Euclidean space.

Within this framework, the proposed method can be interpreted as extending inlier refinement with geometric priors, by incorporating normal consistency in addition to residual and spatial information.

Table 1  
Default parameters used in the experiments.
<table><tr><td>Component</td><td>Parameter</td><td>Value</td></tr><tr><td rowspan="3">RANSAC</td><td>Inlier threshold, synthetic data</td><td>2.5σ</td></tr><tr><td>Inlier threshold, real scans</td><td>0.015 m</td></tr><tr><td>Outer iterations</td><td>20</td></tr><tr><td>Local opt.</td><td>Inner iterations</td><td>25</td></tr><tr><td rowspan="2">Graph</td><td>Neighbors, controlled data</td><td> $k = 6$ </td></tr><tr><td>Neighbors, real scans</td><td> $k = 1 0$ </td></tr><tr><td rowspan="2">Hypothesis generation</td><td>Sample size</td><td>30</td></tr><tr><td>Minimum inlier support Minimum surface coverage</td><td>20</td></tr></table>

For the multi-model case, we adopt a sequential strategy: all methods are embedded into a Sequential-RANSAC framework [29], where models are extracted one at a time and their inliers removed before reapplying the algorithm to the residual data. This procedure allows us to evaluate the robustness of each variant in primitive decomposition tasks.

In the final experiment, we further improve the decomposition quality by applying a RANSACOV selection step [17] as post-processing. After the sequential procedure has terminated, all candidate models collected across the � rounds are passed to RANSACOV, which solves an Integer Linear Program to select the subset of at most � models that jointly maximise the number of explained inliers.

To ensure a fair comparison, we configure all methods under the same conditions: The maximum number of iterations $m$ is fixed to 20 for all algorithms. A larger budget would eventually allow all methods to saturate, thus hiding meaningful performance diferences. The consensus function is the standard count of inliers. The sample-set size $\vert \mathcal { M } _ { j } \vert$ was set to 30 by default. For larger point clouds in the uncapped experiments, however, the sample-set size was increased proportionally to the number of points in the point cloud.

Non-linear model fitting for superquadrics is performed using scipy.optimize.least\_squares solver, using a trustregion reflective method, PCA-based initialization, bounded parameters, and a robust $\mathrm { s o f t } - \ell _ { 1 }$ loss. For algorithms involving an inner RANSAC loop (GC-RANSAC and GAIR-RANSAC), we fix the number of inner iterations to 25. This setup allows us to isolate the contribution of the refinement strategies and proposed energy formulation, while keeping all other factors consistent across the baselines. The default parameter values used in the experimental evaluation are reported in Table 1.

## 5.4. Sensitivity to the inlier threshold

We first analyze the sensitivity of our method to the inlier threshold parameter �, which plays a critical role in consensus maximization frameworks. While all methods rely on this parameter to determine inlier membership, we show that geometric-aware refinement reduces its impact, leading to more stable inlier sets even when the threshold is not perfectly tuned.

Sensitivity to the inlier threshold $\begin{array} { r l r } { \varepsilon } & { { } \ } & { = \ } \end{array} \begin{array} { r l r } { s \sigma } \end{array}$ on TangentSuperquadrics. Chamfer Distance (CD), Hausdorf Distance (HD) averaged over runs. Best values in bold.
<table><tr><td>Scale s</td><td>Method</td><td>CD↓</td><td>HD↓</td></tr><tr><td>1.0</td><td>Vanilla RANSAC GAIR-RANSAC (ours)</td><td> $0 . 6 4 { \pm } 0 . 1 0$   $\pm 0 . 5 0 { \pm } 0 . 0 1 $ </td><td> $2 . 1 5 { \pm } 0 . 4 5$   $1 . 7 3 { \scriptstyle \pm 0 . 2 8 }$ </td></tr><tr><td>1.5</td><td>Vanilla RANSAC GAIR-RANSAC (ours)</td><td> $0 . 7 0 { \scriptstyle \pm 0 . 0 7 }$   $\pm 0 . 4 3 { \pm } 0 . 0 4$ </td><td> $2 . 0 6 { \pm } 0 . 2 2$   $\pm 0 . 9 6 { \pm } 0 . 1 6$ </td></tr><tr><td>2.0</td><td>Vanilla RANSAC GAIR-RANSAC (ours)</td><td> $0 . 6 7 { \scriptstyle \pm 0 . 0 7 }$   $\pm 0 . 4 4 \pm 0 . 0 2$ </td><td> $2 . 0 1 { \pm } 0 . 1 0 $   $\pm . 6 2 \pm 0 . 2 1$ </td></tr><tr><td>2.5</td><td>Vanilla RANSAC GAIR-RANSAC (ours)</td><td> $0 . 6 5 { \pm } 0 . 1 0$   $\pm 0 . 4 1 \pm 0 . 0 1$ </td><td> $2 . 2 2 { \pm } 0 . 1 9$   ${ \bf 1 . 3 0 { \pm } } 0 . 1 8$ </td></tr><tr><td>3.0</td><td>Vanilla RANSAC GAIR-RANSAC (ours)</td><td> $0 . 6 8 { \pm } 0 . 0 3$   $\pm 0 . 4 3 { \pm } 0 . 0 3$ </td><td> $2 . 5 4 { \pm } 0 . 1 4$   $1 . 4 2 { \scriptstyle \pm 0 . 3 6 }$ </td></tr><tr><td>3.5</td><td>Vanilla  $\mathrm { R A N S A C }$   ${ \mathsf { G A l R - R a N S A C \ ( o u r s ) } }$ </td><td> $0 . 7 0 { \pm } 0 . 1 5$   $\pm 0 . 4 2 \pm 0 . 0 1$ </td><td> $2 . 4 3 { \pm } 0 . 6 7$   $1 . 5 8 { \pm } 0 . 0 9$ </td></tr><tr><td>4.0</td><td> $\mathsf { V a n i l l a \ R a N S A C }$   $\mathsf { G A l R \mathrm { - R A N S A C } }$  (ours)</td><td> $0 . 6 1 { \pm } 0 . 0 6$   $\pm 0 . 4 2 \pm 0 . 0 1$ </td><td> $2 . 3 3 { \pm } 0 . 5 5 $   $\pm 1 . 5 6 { \pm } 0 . 0 9$ </td></tr></table>

To this end, we compare GAIR-RANSAC with Vanilla-RANSAC on the TangentSuperquadrics. Each point cloud is corrupted with Gaussian noise of fixed standard deviation $\sigma = 0 . 4$ . Additionally, 10% of the points are replaced with uniformly distributed outliers within the bounding box of the point cloud. We vary the inlier threshold as $\varepsilon = s \cdot \sigma ,$ with $s \in [ 1$ , 4] with step 0.5 and report Chamfer Distance (CD) and Hausdorf Distance (HD), averaged over 5 runs per configuration in Table 2. Two main trends can be observed. First, GAIR-RANSAC consistently outperforms Vanilla RANSAC across all values of �, confirming the benefit of geometricaware refinement independently of the threshold choice. Second, and more importantly, GAIR-RANSAC exhibits significantly lower sensitivity to the threshold parameter: performance remains stable over a wide range of values, with only minor variations in CD and HD. In contrast, Vanilla RANSAC is strongly afected by the choice of �. Small values $( s \leq 2 )$ lead to an underestimation of the inlier set and degraded model quality, while large values $( s ~ \geq ~ 3 . 5 )$ admit points far from the surface, increasing fitting error. This behaviour reflects the fact that, in proximity-based formulations, the threshold alone governs inlier selection.

The improved robustness of GAIR-RANSAC can be attributed to the additional geometric constraints introduced in the refinement step. By enforcing normal consistency, inlier selection is no longer driven solely by point-to-model distance, but also by local surface coherence. As a result, the method is less dependent on the precise tuning of �, making the threshold easier to set in practice.

Single-model fitting: mean IAE across the three single model point clouds from SqSoup and 25 trials per condition. Best result per row in bold.
<table><tr><td>Outlier</td><td>GAIR</td><td>GC</td><td>LO</td><td> $\mathrm { R A N S A C }$ </td></tr><tr><td>0%</td><td>0.0260</td><td>0.0478</td><td>0.0480</td><td>0.0595</td></tr><tr><td>5%</td><td>0.0302</td><td>0.0569</td><td>0.0590</td><td>0.0823</td></tr><tr><td>10%</td><td>0.0226</td><td>0.0493</td><td>0.0522</td><td>0.1113</td></tr><tr><td>15%</td><td>0.0302</td><td>0.0583</td><td>0.0597</td><td>0.1143</td></tr><tr><td>20%</td><td>0.0318</td><td>0.0594</td><td>0.0701</td><td>0.1465</td></tr><tr><td>25%</td><td>0.0361</td><td>0.0700</td><td>0.0754</td><td>0.1425</td></tr><tr><td>30%</td><td>0.0387</td><td>0.0739</td><td>0.0919</td><td>0.1811</td></tr><tr><td>35%</td><td>0.0441</td><td>0.0757</td><td>0.1011</td><td>0.2122</td></tr><tr><td>40%</td><td>0.0370</td><td>0.0725</td><td>0.1091</td><td>0.2295</td></tr></table>

Based on this analysis, we fix $\varepsilon ~ = ~ 2 . 5 \sigma$ for all subsequent controlled synthetic experiments, which provides a good trade-of between geometric accuracy and robustness.

In practice, for acquired point clouds, the noise level � depends on the acquisition process, including the measurement accuracy of the sensing device and the physical units in which the scan is represented. In these cases, the threshold � has a direct geometric interpretation, as it corresponds to the maximum point-to-surface discrepancy that is considered compatible. It can therefore be selected according to the nominal accuracy of the acquisition device or to an externally estimated noise level �. Automatically estimating � from the input data constitutes a separate, sensor-dependent problem, which is outside the scope of the present contribution. Moreover, our sensitivity analysis shows that GAIR remains stable over a relatively broad range of threshold values. For this reason, in the real-data experiments we use a single fixed threshold for each dataset, expressed in the physical units of the corresponding scans as reported in Tab.1.

## 5.5. Single-model fitting

We evaluate the efect of inlier refinement in the singlemodel setting, where the goal is to isolate improvements in fitting accuracy independently of segmentation ambiguities. In this controlled scenario, we consider the single model split of the SqSoup dataset. Each point cloud contains a single superquadric corrupted by increasing levels of uniformly distributed outliers, allowing us to focus purely on the robustness of the fitting process.

Results are reported in Table 3 and in Figure 5. As expected, the performance of all methods degrades as the outlier ratio increases; however, clear trends emerge across refinement strategies. First, Vanilla RANSAC, which relies solely on residual-based inlier selection, consistently achieves the worst performance. Its error rapidly increases with the outlier ratio, exceeding 20% IAE at high contamination levels, confirming the limitations of purely residualbased criteria in the presence of significant noise. Introducing local optimization (LO-RANSAC) improves the results, but only marginally: while it reduces the error at low outlier ratios, its performance still deteriorates significantly as contamination increases, roughly doubling between 0% and 40% outliers. This indicates that residual-based refinement alone is insuficient to ensure robustness. A more substantial improvement is observed with GC-RANSAC, where spatial coherence is enforced through proximity-based regularization. This leads to more stable behavior across outlier levels, highlighting the benefit of incorporating additional priors beyond point-wise residuals.

![](images/4204d580103b5f1f0086028897d27c2cf53c666ed7030cb456362308f338e9d8.jpg)  
Figure 5: Average IAE as a function of the outlier ratio on the single-model subset of SqSoup.

![](images/9721ac9667f663508ba39edacdde920f4058fc966937bf3a532414d0b4a5d0ff.jpg)  
Figure 6: IAE versus execution time for the single-model subset of SqSoup.

Finally, GAIR-RANSAC consistently achieves the best performance across all contamination levels. Even in this simplified setting, where no interaction between primitives is present, the geometric-aware refinement provides a clear advantage, reducing the IAE by approximately a factor of two compared to GC-RANSAC. This confirms that incorporating geometric consistency, in particular normal coherence, leads to more reliable inlier selection and significantly improves robustness to outliers.

In addition to accuracy, we analyze the trade-of between IAE and computational cost in Figure 6, which reports the results of multiple runs for GAIR-RANSAC, GC-RANSAC, and LO-RANSAC. Each point corresponds to a single run, allowing us to visualize both performance and variability. LO-RANSAC exhibits a lower computational cost, but at the price of significantly higher IAE. In contrast, GC-RANSAC and GAIR-RANSAC show comparable execution times, reflecting the additional cost of the graph-cut-based refinement. However, GAIR-RANSAC consistently achieves lower error across runs, clearly dominating GC-RANSAC in terms of the accuracy–eficiency trade-of.

Primitive decomposition on the synthetic scene tangent superquadrics consisting of � = 4 intertwined superquadrics. Best values in bold.
<table><tr><td> $( \sigma , N _ { \mathrm { o u t } } )$ </td><td>Method</td><td>CD↓</td><td>HD↓</td><td>IAE ↓</td></tr><tr><td rowspan="2">(0.1, 0)</td><td>GC-RANSACOV</td><td> $0 . 8 2 { \scriptstyle \pm 1 . 0 7 }$ </td><td> $5 . 0 9 { \pm } 8 . 8 7 $ </td><td> $0 . 0 6 { \scriptstyle \pm 0 . 0 6 }$ </td></tr><tr><td>GAIR-RANSACOV (ours)</td><td> $\pm 0 . 2 9 { \pm } 0 . 0 1 $ </td><td> $\pm 0 . 5 8 { \pm } 0 . 0 3$ </td><td>0.03±0.00</td></tr><tr><td rowspan="2">(0.1, 4000)</td><td>GC-RANSACOV</td><td> $0 . 3 8 { \pm } 0 . 2 0 $ </td><td> $1 . 5 9 { \pm } 1 . 9 1 $ </td><td> $0 . 0 8 { \pm } 0 . 0 9$ </td></tr><tr><td>GAIR-RANSACoV (ours)</td><td>0.27±0.00</td><td> ${ \pm } 0 . 5 9 { \pm } 0 . 1 2$ </td><td> ${ \bf 0 . 0 3 2 0 . 0 0 }$ </td></tr><tr><td rowspan="2">(0.2, 0)</td><td>GC-RANSACOV</td><td>0.29±0.01</td><td>1.51±0.49</td><td> $0 . 0 9 { \pm } 0 . 0 0$ </td></tr><tr><td>GAIR-RANSACOV (ours)</td><td>0.26±0.01</td><td>0.91±0.20</td><td>0.08±0.00</td></tr><tr><td rowspan="2">(0.2, 4000)</td><td>GC-RANSACOV</td><td>0.28±0.01</td><td>1.29±0.39</td><td> $0 . 0 8 { \pm } 0 . 0 0$ </td></tr><tr><td>GAIR-RANSACOV (ours)</td><td>0.26±0.00</td><td>1.10±0.12</td><td> $\pm 0 . 0 8 { \pm } 0 . 0 0$ </td></tr><tr><td rowspan="2">(0.4, 0)</td><td>GC-RANSACOV</td><td>0.70±0.18</td><td>3.22±1.43</td><td>0.14±0.04</td></tr><tr><td>GAIR-RANSACOV (ours)</td><td>0.63±0.00</td><td>2.05±0.12</td><td>0.09±0.00</td></tr><tr><td rowspan="2">(0.4, 4000)</td><td>GC-RANSACOV</td><td>0.59±0.12</td><td>2.97±1.07</td><td> $0 . 1 1 { \pm } 0 . 0 3$ </td></tr><tr><td>GAIR-RANSACOV (ours)</td><td>0.55±0.12</td><td>1.98±0.22</td><td> ${ \bf 0 . 0 9 2 0 . 0 1 }$ </td></tr></table>

## 5.6. Primitive decomposition

We now move to the multi-model setting, where the goal is to decompose a point cloud into multiple primitives. In contrast to the single-model case, this scenario introduces intrinsic ambiguities, as diferent primitives may be spatially adjacent or even intersect, making inlier assignment significantly more challenging. To address this setting, we extend all methods from their single-model RANSAC formulation to a multi-model framework based on RANSACOV, which decouples hypothesis generation from model selection, as described in Section 4.3, while difering in their inlier refinement strategies.

Tangent superquadrics. We first evaluate the two bestperforming methods from the previous experiment, namely GC-RANSACOV and GAIR-RANSACOV, on the Tangent Superquadrics dataset introduced in Section 5.1. This benchmark is specifically designed to stress-test primitive decomposition in the presence of adjacent and touching surfaces.

Using the threshold scale � = 2.5 identified in Section 5.4, we compare the two methods under varying levels of Gaussian noise � and injected outliers $N _ { \mathrm { o u t } }$ . In this setting, spatial proximity becomes an unreliable cue, as points belonging to diferent primitives can lie arbitrarily close in Euclidean space.

Performance is evaluated in terms of Chamfer Distance (CD) and Hausdorf Distance (HD), which quantify how accurately the underlying geometry is recovered, as well as Inlier Assignment Error (IAE). Results reported in Table 4 show that GAIR-RANSAC consistently outperforms GC-RANSAC across all conditions and metrics.

![](images/0f2b04fb8922135ab294f38759c05cda0b94b0ef51ba77c103e86cebceb72879.jpg)  
(a) Cartoon Character (high outliers)

![](images/d543992485fe1a0369fd1455363bc84bef4d5307d0d697237ef68d51733d4d52.jpg)  
(b) Bird

![](images/3b952ba2818e6c536975e2413cbdb9f64994e86448733457bba06a2d4c65175c.jpg)  
Figure 7: Qualitative results of primitive decomposition using GAIR-RanSaCov on shapes from the 3D Shapes dataset. (a) Robustness under severe outlier contamination: despite strong noise, the decomposition remains consistent with the underlying geometry, as normal coherence prevents incorrect inlier assignments. (b) Example on a bird point cloud, illustrating the ability of the method to recover meaningful primitives on complex shapes.

The improvement is systematic: GAIR achieves lower CD and HD, indicating a more faithful geometric reconstruction, and higher segmentation accuracy, reflecting a more reliable separation of adjacent primitives. Moreover, the variance of the results is consistently lower, highlighting the increased stability of the geometric-aware refinement. These findings confirm that incorporating geometric priors, in particular normal consistency, is crucial in scenarios where proximity alone is insuficient to disambiguate surface membership.

Primitive decomposition on SqSoup and 3D Shapes. We then extend the evaluation to all ten point clouds from SqSoup and 3DShapes. For each algorithm, we run 25 trials of sequential RANSAC and collect all candidate models generated across runs. These candidates are then passed to a RANSACOV selection step, which selects the subset of at most � models that jointly maximizes the number of explained inliers. To better analyze the behavior of the diferent methods, we report the trend of the Inlier Assignment Error (IAE) as a function of the outlier ratio separately for the two datasets: SqSoup in Figure 8a and 3D Shapes in Figure 8b. This metric evaluates the consistency of point-to-model assignments without requiring explicit multi-label segmentation, making it applicable also to real-world shapes where ground-truth primitive annotations are not available (e.g., Thingi10K). In both cases, GC-RANSACOV and GAIR-RANSACOV emerge as the bestperforming methods, clearly outperforming Vanilla and LObased formulations. This confirms, also in the multi-model setting, the benefit of moving beyond purely residual-based refinement strategies.

Averaging the results over all ten point clouds, reported in Table 5, confirms that GAIR-RANSACOV achieves the lowest Chamfer Distance at every outlier level, with a consistent margin over all baselines. This trend indicates that geometric-aware refinement is particularly efective in preserving reliable inlier assignments between multiple primitives even under severe contamination.

Table 5  
Chamfer distance (CD) and Inlier Assignment Error (IAE) as a function of the injected outlier fraction, averaged across point clouds.  
Chamfer Distance ↓
<table><tr><td>Outlier Frac.</td><td>GAIR</td><td>GC</td><td>LO</td><td>Vanilla</td></tr><tr><td>0.00</td><td>0.0477</td><td>0.0789</td><td>0.0847</td><td>0.0851</td></tr><tr><td>0.05</td><td>0.0543</td><td>0.0879</td><td>0.0956</td><td>0.1030</td></tr><tr><td>0.10</td><td>0.0592</td><td>0.0908</td><td>0.1065</td><td>0.1133</td></tr><tr><td>0.15</td><td>0.0659</td><td>0.0971</td><td>0.1109</td><td>0.1117</td></tr><tr><td>0.20</td><td>0.0657</td><td>0.1051</td><td>0.1158</td><td>0.1215</td></tr><tr><td>0.25</td><td>0.0687</td><td>0.1031</td><td>0.1224</td><td>0.1275</td></tr><tr><td>0.30</td><td>0.0726</td><td>0.1105</td><td>0.1304</td><td>0.1323</td></tr><tr><td>0.35</td><td>0.0753</td><td>0.1123</td><td>0.1349</td><td>0.1390</td></tr><tr><td>0.40</td><td>0.0771</td><td>0.1100</td><td>0.1470</td><td>0.1464</td></tr></table>

Inlier Assignment Error (IAE) ↓
<table><tr><td>Outlier Frac.</td><td>GAIR</td><td>GC</td><td>LO</td><td>Vanilla</td></tr><tr><td>0.00</td><td>0.0604</td><td>0.0514</td><td>0.0668</td><td>0.0755</td></tr><tr><td>0.05</td><td>0.0599</td><td>0.0635</td><td>0.0914</td><td>0.1002</td></tr><tr><td>0.10</td><td>0.0663</td><td>0.0896</td><td>0.1214</td><td>0.1341</td></tr><tr><td>0.15</td><td>0.0765</td><td>0.1055</td><td>0.1457</td><td>0.1571</td></tr><tr><td>0.20</td><td>0.0804</td><td>0.1464</td><td>0.1717</td><td>0.1819</td></tr><tr><td>0.25</td><td>0.0853</td><td>0.1610</td><td>0.1966</td><td>0.2054</td></tr><tr><td>0.30</td><td>0.0975</td><td>0.1872</td><td>0.2219</td><td>0.2259</td></tr><tr><td>0.35</td><td>0.1081</td><td>0.2124</td><td>0.2488</td><td>0.2540</td></tr><tr><td>0.40</td><td>0.1141</td><td>0.2354</td><td>0.2693</td><td>0.2786</td></tr></table>

Table 6  
Number of local optimization steps and runtime as a function of the injected outlier fraction, averaged across point clouds.  
Number of Local Optimizations
<table><tr><td>Outlier Frac.</td><td>GAIR</td><td>GC</td><td>LO</td><td>Vanilla</td></tr><tr><td>0.00</td><td>19.10</td><td>16.22</td><td>4.34</td><td>0.00</td></tr><tr><td>0.05</td><td>18.00</td><td>16.08</td><td>4.68</td><td>0.00</td></tr><tr><td>0.10</td><td>23.12</td><td>18.55</td><td>5.23</td><td>0.00</td></tr><tr><td>0.15</td><td>24.90</td><td>20.29</td><td>5.31</td><td>0.00</td></tr><tr><td>0.20</td><td>25.49</td><td>21.42</td><td>5.94</td><td>0.00</td></tr><tr><td>0.25</td><td>26.10</td><td>22.60</td><td>5.70</td><td>0.00</td></tr><tr><td>0.30</td><td>27.20</td><td>22.26</td><td>6.30</td><td>0.00</td></tr><tr><td>0.35</td><td>28.95</td><td>23.66</td><td>6.92</td><td>0.00</td></tr><tr><td>0.40</td><td>28.29</td><td>21.88</td><td>7.13</td><td>0.00</td></tr></table>

<table><tr><td>Outlier Frac.</td><td>GAIR</td><td>GC</td><td>LO</td><td>Vanilla</td></tr><tr><td>0.00</td><td>52.33</td><td>58.15</td><td>21.05</td><td>14.64</td></tr><tr><td>0.05</td><td>52.91</td><td>56.95</td><td>26.23</td><td>20.78</td></tr><tr><td>0.10</td><td>63.16</td><td>71.23</td><td>28.39</td><td>22.61</td></tr><tr><td>0.15</td><td>70.36</td><td>78.92</td><td>32.66</td><td>27.45</td></tr><tr><td>0.20</td><td>72.38</td><td>86.08</td><td>36.59</td><td>30.17</td></tr><tr><td>0.25</td><td>70.73</td><td>90.07</td><td>35.95</td><td>31.54</td></tr><tr><td>0.30</td><td>77.51</td><td>97.43</td><td>39.52</td><td>33.73</td></tr><tr><td>0.35</td><td>83.23</td><td>104.96</td><td>44.44</td><td>36.03</td></tr><tr><td>0.40</td><td>82.05</td><td>100.12</td><td>45.25</td><td>38.29</td></tr></table>

A qualitative example of this robustness is illustrated in Figure 7a, where GAIR-RANSACOV is applied to the Cartoon Character shape under high outlier contamination. Despite the severe degradation of the input, the method still recovers a set of superquadrics that provides a coherent approximation of the overall geometry.

The same trend is also reflected by the IAE reported in Table 5b and in Figure 8c. Except for the clean-data setting, where GC-RANSACOV obtains a slightly lower IAE, GAIR-RANSACOV achieves the lowest error at every non-zero outlier level. The gap becomes more pronounced as the contamination level increases, suggesting that geometric awareness becomes increasingly important when residual and proximity cues alone are no longer suficient. In fact, at low outlier injection levels, all methods achieve similar IAE values despite exhibiting substantially diferent Chamfer distances.

A qualitative example is shown in Figure 7b, where we report the primitive decomposition obtained by GAIR-RANSACOV on the Bird shape together with the point-wise Chamfer error with respect to the fitted superquadrics. As expected, the decomposition captures the overall structure of the shape while smoothing out fine-scale details. The largest errors are concentrated around high-curvature distinctive regions, such as the beak, which are less well approximated by a compact superquadric representation. These details could in principle be recovered by increasing the model complexity, e.g., by allowing a larger number of primitives.

Table 7  
Chamfer distance (CD) and Inlier Assignment Error (IAE) as a function of the time budget, averaged fairly across point clouds (outlier fraction fixed at 0.40).  
Chamfer Distance ↓
<table><tr><td>Algorithm</td><td>60s</td><td>90s</td><td>120s</td></tr><tr><td>GAIR-RANSACOV</td><td>0.0987</td><td>0.0942</td><td>0.0921</td></tr><tr><td>GC-RANSACOV</td><td>0.1131</td><td>0.1101</td><td>0.1092</td></tr><tr><td>LO-RANSACOV</td><td>0.1064</td><td>0.1043</td><td>0.1039</td></tr><tr><td>Vanilla RANSACOV</td><td>0.1077</td><td>0.1071</td><td>0.1069</td></tr></table>

<table><tr><td colspan="4">Inlier Assignment Error (IAE)↓</td></tr><tr><td>Algorithm</td><td>60s</td><td>90s</td><td>120s</td></tr><tr><td>GAIR-RANSACOV</td><td>0.1880</td><td>0.1717</td><td>0.1661</td></tr><tr><td>GC-RANSACOV</td><td>0.2169</td><td>0.2092</td><td>0.2053</td></tr><tr><td>LO-RANSACOV</td><td>0.2000</td><td>0.1925</td><td>0.1886</td></tr><tr><td>Vanilla RANSACOV</td><td>0.1959</td><td>0.1899</td><td>0.1869</td></tr></table>

Fixed run-time budget. To compare the methods under the same computational budget, we fix the outlier fraction to 0.40 and allow each method to generate candidate models for 60, 90, or 120 seconds. The numbers of outer and inner iterations are fixed to 20, and the resulting candidates are passed to RANSACOV, which selects the subset maximizing inlier coverage.

As reported in Table 7, GAIR-RANSACOV achieves the lowest Chamfer Distance and Inlier Assignment Error for every tested budget. Its performance improves consistently as more computation becomes available, whereas the competing approaches show smaller gains and tend to plateau. Results are averaged over all ten point clouds of SqSoup and 3D Shapes and 16 independent trials.

Sketchfab point clouds. Additional qualitative comparisons on the publicly available Sketchfab scans are reported in Figure 9. For each example, we show the input point cloud and the decompositions obtained with GC-RANSAC and GAIR-RANSACOV. Both methods employ the same set-cover model-selection stage and difer only in the inlier-refinement strategy used during hypothesis generation. Therefore, the diferences observed in the final decompositions can be directly attributed to the quality of the generated candidate models.

The proximity-based refinement of GC-RANSAC tends to propagate inlier assignments across adjacent but geometrically distinct surface regions. As a result, the resulting hypothesis pool often contains large primitives spanning multiple object parts, while lacking candidates that separately represent the individual components. The subsequent set-cover optimization cannot recover such missing hypotheses and consequently produces under-segmented decompositions. In contrast, the normal-aware refinement of GAIR limits inlier propagation across surface discontinuities, generating candidate primitives that better follow the visible geometric components of the objects. This behavior is particularly evident for the hammer, the hydrant, and the baptismal font, where GAIR-RANSACOV produces a decomposition that is more consistent with the main geometric parts of the shapes, whereas GC-RANSACOV systematically merges distinct components.

![](images/9e4d388b70cf14ca6bd5e6eed4b757ee35a38f746bde6d6f3dc54b54edfc5c52.jpg)

![](images/4099a8154eae8eae0be83af0b9057fabfeff12bc556aa054ed75114808cda27e.jpg)  
(a) Chamfer Distance (CD)

![](images/465ebfb7c93b5839e66a83e8732790afcde22ac8eb19f11ead9cb0861e54c551.jpg)  
(b) Hausdorf Distance (HD)

![](images/a37c78239ff6380a54ee4bcb2795572030a0d1ac44a8583697255a917abc4d64.jpg)  
(c) Inlier Assignment Error (IAE)  
Figure 8: Primitive decomposition results across datasets. (a) and (b) report the average Chamfer Distance (CD) and Hausdorf Distance (HD) on SqSoup and 3D Shapes. (c) shows the Inlier Assignment Error (IAE) across both datasets. GAIR-RanSaCov consistently achieves lower error and better geometric accuracy.

![](images/60995fdc704cd188ddb52d85bfb07b203d8046939d96c6c7f2f233285b6dcbee.jpg)

![](images/400d3421e2b284e2016e07703f46907412f23e9483ac54c0d7908b290e28b873.jpg)

![](images/2d48127f42275f31b786d4d2dc35cc9f9c7df0b4c72d3987ea158c24a088e1d6.jpg)  
(a) Input point cloud  
(b) GC-RANSACOV

![](images/b2f254e272399df7524965bc919617eca8a9aca2e72a329ca14edfeb3e288516.jpg)  
(c) GAIR-RANSACOV

![](images/faab143858abca2338d7e259a1c673c379f0f3b39f2c3413464600e95e39f51d.jpg)  
(d) Input point cloud

![](images/d70a0a4083d33b68f240e149e2fbea54a7a407afc0a25f168eff5ce1dc0ed5fe.jpg)

(e) GC-RANSACOV  
![](images/2d676f0dd75c5d2099c64a9d011eb726777c2d2b0ff3789fe41e459dfbef667a.jpg)  
(f) GAIR-RANSACOV  
Figure 9: Qualitative comparison on publicly available Sketchfab point clouds. Within each group, from left to right, we show the input point cloud, the decomposition obtained with GC-RanSaCov, and the decomposition obtained with GAIR-RanSaCov.

## 6. Ablation studies on the GAIR energy

We perform a controlled ablation study on the Hammer point cloud, using eight independent runs for each configuration. Recall that the refinement minimizes the energy

$$
E ( f ) = \sum _ { \mathbf { p } \in D } E _ { 1 } ( f _ { p } ) + \sum _ { ( \mathbf { p } , \mathbf { q } ) \in \mathcal { E } } E _ { 2 } ( f _ { p } , f _ { q } ) ,\tag{19}
$$

as defined in Eq. (6).

To clarify the contribution of the individual components, we denote by $E _ { 1 } ^ { \mathrm { G A I R } }$ the normal-aware unary term in Eq. (7), by $E _ { 2 } ^ { \mathrm { G A I R } } ( C )$ the residual-aware pairwise term in Eq. (11), and by $E _ { \gamma } ^ { \mathrm { G C } }$ the original GC-RANSAC pairwise term in Eq. (16). We compare the following configurations:

1. GC-RANSAC, using the original GC-RANSAC unary and pairwise terms;

2. GAIR-U, using the GAIR unary term $E _ { 1 } ^ { \mathrm { G A I R } }$ together with the original pairwise term $E _ { 2 } ^ { \mathrm { G C } }$ . This variant isolates the efect of introducing point-to-model normal consistency in the unary cost;

3. GAIR-Uniform, using $\mathsf { \overline { { E } } } _ { 1 } ^ { \mathrm { G A I R } }$ and the GAIR pairwise formulation $E _ { 2 } ^ { \mathrm { G A I R } } ( C )$ , but setting

$$
C ( \mathbf { n } _ { p } , \mathbf { n } _ { q } ) \equiv 1
$$

for every neighboring pair. This removes normal coherence from the pairwise term while retaining its residual-aware structure;

4. GAIR-Hard, using the same GAIR unary and pairwise terms, but replacing the continuous coherence with the binary function

$$
C _ { \mathrm { h a r d } } ( \boldsymbol { \mathbf { n } } _ { p } , \boldsymbol { \mathbf { n } } _ { q } ) = \mathbb { 1 } \left[ C ( \boldsymbol { \mathbf { n } } _ { p } , \boldsymbol { \mathbf { n } } _ { q } ) > 0 . 9 \right] .
$$

Normal information therefore determines whether an edge is active, but all retained edges receive the same weight;

5. GAIR, using the complete formulation with $E _ { 1 } ^ { \mathrm { G A I R } }$ and $E _ { ? } ^ { \mathrm { G A I R } } ( C )$ , including hard edge removal below the coherence threshold and continuous coherence weighting on the retained edges.

![](images/65bc01e9f5a20e8063c61d8fc305e833a0ecca3beb98c5e1e6576e1558e34ddc.jpg)  
(a) Chamfer Distance obtained with diferent energy formulations. Lower values are better.

![](images/bc9d444cccfbbe185ae97aff9926d1cb8e16d5030415b3b1126bdd3f0750fc8a.jpg)  
(b) Impact of the noise on normals on the energy formulations. Lower values are better  
Figure 10: Ablation studies on the energy terms, tested on the Hammer point cloud.

Figure 10a shows that GC-RANSAC and GAIR-U achieve comparable results. Thus, replacing only the unary term with its normal-aware counterpart is not suficient to substantially improve the reconstruction.

GAIR-Uniform achieves a similar average error but reduces the variability across runs. Since this formulation does not use normal coherence in the pairwise term, the improvement can be attributed to the residual-aware structure of $E _ { \gamma } ^ { \mathrm { G A I R } }$ , which provides a more stable regularization than the original GC-RANSAC pairwise term.

Introducing hard normal-based edge selection in GAIR-Hard lowers the average error, confirming that preventing propagation across normal discontinuities is important. However, this formulation also exhibits substantially higher variability. The binary coherence introduces a discontinuous decision at the threshold, and all retained edges receive the same weight, including marginal connections whose coherence is only slightly above 0.9. Small variations in the estimated normals or in the sampled hypotheses can therefore produce markedly diferent label propagations.

The complete GAIR formulation retains the benefit of removing incompatible edges while weighting the remaining connections according to their actual normal coherence. This reduces the influence of uncertain connections and yields both the lowest average error and the most stable results.

Sensitivity to normal perturbations. Since GAIR relies on surface normals, we additionally evaluate its sensitivity to normal-estimation errors. Each input normal is independently perturbed with zero-mean Gaussian noise and renormalized. The perturbation magnitude is selected to produce diferent mean angular deviations, including approximately $5 ^ { \circ }$ and $1 0 ^ { \circ }$ . As shown in Figure 10b, the advantage of the normal-aware formulations decreases as the angular error increases. In particular, the variants using normal coherence in the pairwise term are the most afected, since corrupted normals alter both the graph connectivity and the edge weights. When the normals become severely inaccurate, formulations that do not rely on them are preferable, as expected.

## 7. Primitive decomposition of real LiDAR scans

We evaluate the algorithms on three real point-cloud scenes exhibiting realistic acquisition artifacts, including non-uniform sampling, missing regions, and imperfect surface coverage. In particular, object surfaces in contact with the floor are generally not visible. Surface normals are estimated directly from the noisy acquired geometry.

Since all scans are represented in metric scale, we fix the inlier threshold to $\varepsilon = 0 . 0 1 5$ m and keep it unchanged across all scenes. For particularly dense scans, we apply uniform subsampling before decomposition. In particular, the ball and box and the sofa scans are reduced to 100k points, preserving their main geometric structure while substantially reducing the computational cost. The Plush toy dataset is less dense and has 56k points.

Ball and box. The first scene consists of a ball placed on top of a box. Although the acquisition contains realistic occlusions and the lower face of the box is not observed, the underlying geometry is simple and consists of only two clearly distinguishable primitives. Consequently, both GC-RANSAC and GAIR-RANSAC recover the expected decomposition, showing that both methods behave correctly in a geometrically simple real-world setting.

Sofa. The second scene contains a sofa with several cushions. This configuration is substantially more challenging, since it contains many adjacent primitive-like components, several contact regions, and cushions whose surfaces are only partially visible. In this setting, proximity alone is not suficient to distinguish neighboring parts: the refinement of GC-RANSAC tends to propagate inliers across contact regions and fails to recover some of the more sparsely sampled lower components. In contrast, the normal-aware refinement of GAIR-RANSAC better preserves the boundaries between

Ballandbox <sub>o</sub>f<sup>a</sup>

![](images/254d64b175d2b7cc5b4c5cd8815e14e4830a43e9c5b51004421ddfb6f138818e.jpg)  
(a) Input point cloud.

![](images/ff68288415cd95ce796d588882c2c644ddd640a84ec58937706b305ce7a8a82e.jpg)  
(b) GC-RANSAC decomposition.

![](images/3f26a3a7864a3700108395cfd3ce77d65aeb2c5de23c8114b16bc724e9a46b22.jpg)  
(c) GAIR-RANSAC decomposition.

Figure 11: Qualitative comparison on 3D real scanned point clouds captured with an iPhone 16 Pro equipped with LiDAR. Within each group, from left to right, we show the input point cloud, the decomposition obtained with GC-RanSaC, and the decomposition obtained with GAIR-RanSaC.

adjacent cushions and produces a more complete decomposition of the scene.

Plush toy. The plush toy represents the most challenging example. Unlike the sofa components, its body parts are only approximately represented by superquadrics. Moreover, its fuzzy surface, weakly separated components, and multiple contact regions result in noisy normals and poorly defined geometric boundaries. GC-RANSAC produces a strongly under-segmented solution, using a single large superquadric to approximate both the head and torso, while the remaining primitives provide an irregular approximation of the legs, arms, and muzzle. GAIR-RANSAC instead obtains a decomposition that is more consistent with the visible object structure, separately representing the head, torso, limbs, and muzzle. Some fine details, such as one foot and part of one limb, are nevertheless not recovered.

Overall, these experiments show that GAIR transfers efectively to noisy and incomplete LiDAR acquisitions and improves the decomposition of objects composed of several adjacent parts. Nevertheless, fine part-level decomposition remains dificult when the input scan provides insuficient surface coverage or lacks clear geometric separation between components.

## 8. Future Work and Conclusions

Several promising directions remain open for future investigation: Additional geometric priors. The energy formulation can be naturally extended to incorporate other geometric constraints. An exciting direction would be to integrate a term based on Chamfer distance, which could further regularize the solution by promoting the selection of compact superquadrics that provide better coverage of the point data while discouraging excessive surface extensions into empty regions.

Integration with global multi-model fitting. An important direction is to explore how the proposed inlier refinement can be embedded within more sophisticated multimodel fitting frameworks. A promising approach would reformulate the problem as a global multi-label optimization, where each label corresponds to a diferent model (plus one for outliers). This would reduce the dependence on the order of model extraction and potentially lead to more robust decompositions in complex scenes.

Automatic model complexity selection. Finally, integrating a model complexity term directly into the RANSAC loop could automatically determine the number of primitives �, moving beyond the current requirement of specifying it in advance. This would bring the method closer to a fully automatic primitive decomposition pipeline.

This work presents an algorithmic contribution to the problem of primitive decomposition in 3D point clouds. At its core lies a novel energy formulation that extends the GC-RANSAC framework by incorporating geometric priors specifically tailored for primitive fitting. While we focus primarily on normal consistency in this work, the proposed formulation is general and provides a sound foundation for integrating additional geometric constraints. We demonstrate how this formulation can be efectively integrated into both single-model and multi-model estimation pipelines, leading to the GAIR-RANSAC and GAIR-RANSACOV frameworks. The former improves the stability and accuracy of superquadric fitting in the presence of noise and outliers, while the latter enables robust primitive decomposition in challenging scenarios with adjacent or overlapping structures. Although graph-cut optimization is computationally more expensive per iteration than the refinement step in

LO-RANSAC, our experiments reveal a favourable tradeof: GAIR remains faster than GC-RANSAC in our experiments while consistently providing more accurate decompositions. This results in both better accuracy and more consistent execution times overall.

## Acknowledgments

This work is supported by GEOPRIDE under ID 202224 5ZYB and CUP D53D23008370001 (PRIN 2022 M4.C2.1.1 Investment).

## References

[1] Attene, M., Patanè, G., 2010. Hierarchical structure recovery of point-sampled surfaces. Computer Graphics Forum 29, 1905–1920. doi:https://doi.org/10.1111/j.1467-8659.2010.01658.x.

[2] Barath, D., Matas, J., 2018. Multi-class model fitting by energy minimization and mode-seeking, in: Computer Vision – ECCV 2018: 15th European Conference, Munich, Germany, September 8-14, 2018, Proceedings, Part XVI, Springer-Verlag, Berlin, Heidelberg. p. 229–245. URL: https://doi.org/10.1007/978-3-030-01270-0\_14, doi:10.1007/ 978-3-030-01270-0\_14.

[3] Baráth, D., Matas, J., 2019. Progressive-x: Eficient, anytime, multimodel fitting algorithm. 2019 IEEE/CVF International Conference on Computer Vision (ICCV) , 3779–3787URL: https://api.semant icscholar.org/CorpusID:174802993.

[4] Barath, D., Matas, J., 2022. Graph-cut ransac: Local optimization on spatially coherent structures. IEEE Transactions on Pattern Analysis and Machine Intelligence 44, 4961–4974. doi:10.1109/TPAMI.2021.3 071812.

[5] Barr, 1981. Superquadrics and angle-preserving transformations. IEEE Computer Graphics and Applications 1, 11–23. doi:10.1109/ MCG.1981.1673799.

[6] Berger, B., 2021. German fire hydrant 3d scan. Sketchfab 3D model. URL: https://sketchfab.com/3d- models/german- fire- h ydrant- 3d- scan- 98772ce2bbde4ef48b13a90b94aee687. model ID: 98772ce2bbde4ef48b13a90b94aee687; CC BY; accessed July 2026.

[7] Chum, O., Matas, J., Kittler, J., 2003. Locally optimized ransac, in: Michaelis, B., Krell, G. (Eds.), Pattern Recognition, Springer Berlin Heidelberg, Berlin, Heidelberg. pp. 236–243.

[8] Dhal, K., Kashyap, A., Chakravarthy, A., 2025. Collision avoidance of moving 3-d objects in dynamic environments. IEEE Transactions on Automation Science and Engineering 22, 17914–17930. doi:10.1 109/TASE.2025.3585450.

[9] Ferraris, A., Leveni, F., Baieri, D., Maggioli, F., Melzi, S., Magri, L., et al., 2025. Geometric aware local optimization for robust primitive fitting, in: ITALIAN CHAPTER CONFERENCE, Eurographics Association. pp. 1–11.

[10] Fischler, M.A., Bolles, R.C., 1981. Random sample consensus: a paradigm for model fitting with applications to image analysis and automated cartography. Commun. ACM 24, 381–395. URL: https: //doi.org/10.1145/358669.358692, doi:10.1145/358669.358692.

[11] GoMeasure3D, 2021. Kiev 35mm slr camera – artec spider 3d scan. Sketchfab 3D model. URL: https://sketchfab.com/3d-models/kiev -35mm-slr-camera-artec-spider-3d-scan-fd35db24d4c3403c8abd483 30e6a894f. model ID: fd35db24d4c3403c8abd48330e6a894f; CC BY; accessed July 2026.

[12] Gross, A., Boult, T., 1988. Error of fit measures for recovering parametric solids, in: [1988 Proceedings] Second International Conference on Computer Vision, pp. 690–694. doi:10.1109/CCV.1988.590 052.

[13] Isack, H., Boykov, Y., 2012. Energy-based geometric multi-model fitting. International Journal of Computer Vision 97, 123–147. doi:10 .1007/s11263-011-0474-7.

[14] Leonardis, A., Solina, F., Macerl, A., 1994. A direct recovery of superquadric models in range images using recover-and-select

paradigm, in: Eklundh, J.O. (Ed.), Computer Vision — ECCV ’94, Springer Berlin Heidelberg, Berlin, Heidelberg. pp. 309–318.

[15] Liu, W., Wu, Y., Ruan, S., Chirikjian, G.S., 2021. Robust and accurate superquadric recovery: a probabilistic approach. CoRR abs/2111.14517. URL: https://arxiv.org/abs/2111.14517, arXiv:2111.14517.

[16] Magri, L., Fusiello, A., 2014. T-linkage: A continuous relaxation of j-linkage for multi-model fitting, in: Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR).

[17] Magri, L., Fusiello, A., 2016. Multiple models fitting as a set coverage problem, in: IEEE Conference on Computer Vision and Pattern Recognition (CVPR), pp. 3318–3326.

[18] Magri, L., Fusiello, A., 2019. Fitting multiple heterogeneous models by multi-class cascaded t-linkage, in: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 7460– 7468.

[19] Magri, L., Leveni, F., Boracchi, G., 2021. Multilink: Multi-class structure recovery via agglomerative clustering and model selection, in: Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pp. 1853–1862.

[20] Marchal, G., 2018. Baptismal font, saint-clément d’achêne church. Sketchfab 3D model. URL: https://sketchfab.com/3d-models/bapt ismal-font-saint-clement-dachene-church-7b254c1529084006a2480 . model ID: 7b254c1529084006a248092b284e796d; CC BY-NC-SA; accessed July 2026.

[21] Monnier, T., Austin, J., Kanazawa, A., Efros, A., Aubry, M., 2023. Diferentiable blocks world: Qualitative 3d decomposition by rendering primitives. doi:10.48550/arXiv.2307.05473.

[22] Paschalidou, D., van Gool, L., Geiger, A., 2020. Learning unsupervised hierarchical part decomposition of 3d objects from a single rgb image, in: Proceedings IEEE Conf. on Computer Vision and Pattern Recognition (CVPR).

[23] Paschalidou, D., Ulusoy, A.O., Geiger, A., 2019. Superquadrics revisited: Learning 3d shape parsing beyond cuboids. CoRR abs/1904.09970. URL: h t tp : / / a r x i v . o r g / a b s / 1 9 0 4 . 0 9 9 70, arXiv:1904.09970.

[24] Ramamonjisoa, M., Stekovic, S., Lepetit, V., 2022. Monteboxfinder: Detecting and filtering primitives to fit a noisy point cloud. doi:10.4 8550/arXiv.2207.14268.

[25] Schnabel, R., Wahl, R., Klein, R., 2007. Eficient RANSAC for Point-Cloud Shape Detection. Computer Graphics Forum 26, 214–226.

[26] Solina, F., Bajcsy, R., 1990. Recovery of parametric models from range images: the case for superquadrics with global deformations. IEEE Transactions on Pattern Analysis and Machine Intelligence 12, 131–147. doi:10.1109/34.44401.

[27] Spognetta, A., . Hammer02: Raw scan. Sketchfab 3D model. URL: ht tps://sketchfab.com/3d-models/hammer02-rawscan-79bf24b80c024455b 16369d6c137052b. model ID: 79bf24b80c024455b16369d6c137052b; photogrammetric scan; CC BY; accessed July 2026.

[28] Tulsiani, S., Su, H., Guibas, L.J., Efros, A.A., Malik, J., 2017. Learning shape abstractions by assembling volumetric primitives, in: 2017 IEEE Conference on Computer Vision and Pattern Recognition (CVPR), pp. 1466–1474. doi:10.1109/CVPR.2017.160.

[29] Vincent, E., Laganiere, R., 2001. Detecting planar homographies in an image pair, pp. 182 – 187. doi:10.1109/ISPA.2001.938625.

[30] Wu, Y., Li, W., Liu, Z., Liu, W., Chirikjian, G.S., 2025. Autonomous learning-free grasping and robot-to-robot handover of unknown objects. Autonomous Robots 49, 1–16.

[31] Zhang, C., Yang, Z., Zhuo, H., Liao, L., Yang, X., Zhu, T., Li, G., 2023. A lightweight and drift-free fusion strategy for drone autonomous and safe navigation. Drones URL: https://api.sema nticscholar.org/CorpusID:255697797.

[32] Zhang, H., Li, C., Gao, L., Wang, G., 2015. Hierarchical mesh segmentation based on quadric surface fitting, in: 2015 14th International Conference on Computer-Aided Design and Computer Graphics (CAD/Graphics), pp. 33–40. doi:10.1109/CADGRAPHICS.2015.26.
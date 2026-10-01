# Unveiling the Value of Motion for Cinematic Camera Trajectories

Ziqi Zhou<sup>1</sup> <sup>1</sup>University of Edinburgh Ziqi.Zhou@ed.ac.uk

Yujian Yuan<sup>2</sup>

Laura Sevilla-Lara<sup>1</sup>

<sup>2</sup>The Hong Kong University of Science and Technology yyuanbn@connect.ust.hk l.sevilla@ed.ac.uk

## Abstract

Cinematic camera motion is a fundamental storytelling tool, defined not only by where the camera is positioned in the scene, but also by how it moves in terms of direction and speed. Recent work on camera trajectory generation and alignment to text relies on pose-centric representations. While in principle a network could derive direction of movement and speed, we find that in practice this might not happen. In fact, in this paper we discover that decomposing the camera trajectory representation from the traditional per-frame poses to direction and speed has surprising benefits across multiple tasks, including trajectory-to-text alignment as well as text-to-trajectory generation. To accurately evaluate the former, we introduce a simple and reliable protocol that overcomes the limitations of prior evaluation baselines. For the latter, building on this representational insight, we propose a novel generative model for camera trajectories, CINEGEN, that achieves superior performance across a variety of metrics. We also propose a novel dataset, CINESCRIPT, containing movie clips that are enriched with scene descriptions as well as higher-level metadata. This novel data allows us to test models’ ability to capture high-level cinematographic information. We show that, despite its simplicity, representing camera trajectories through direction and speed not only helps numerically to achieve better alignment and generation, but also inherently encodes complex directorial intent. Feel free to visit our project page for more information.

## 1 Introduction

Camera motion plays a vital role in cinematic storytelling, shaping visual pacing, audience attention, and directorial style. Driven by applications in virtual production [1, 2] and controllable video generation [3–7], recent text-conditioned camera trajectory generators have made significant progress in synthesizing camera paths from natural-language descriptions [8–11]. However, despite these advances, a fundamental challenge remains: how can we represent camera trajectories in a way that naturally aligns with human language?

Most existing methods parameterize camera trajectories as sequences of absolute per-frame poses [8– 11]. While geometrically complete, this pose-centric approach fundamentally clashes with the nature of cinematographic language. Human descriptions emphasize how a camera moves, such as “gradually dollies in and pans right”, rather than where it is positioned in a global 3D space. By forcing semantic motion instructions into an absolute coordinate system, current models unnecessarily entangle direction, speed, and spatial location into a single opaque vector. This representational bottleneck severely complicates both text-to-trajectory generation and cross-modal alignment.

We argue that trajectory representation is not a minor implementation detail, but the central key to unlocking controllable cinematic generation. To overcome the geometric entanglement, we propose a remarkably simple yet highly effective shift to a motion-centric perspective. We introduce DIRSPEED, a parameterization that converts absolute poses into frame-to-frame velocities, explicitly decoupling them into normalized direction and log-speed. By mathematically isolating the axis of movement from its magnitude, this decomposition naturally mirrors the human vocabulary of cinematography. By natively aligning the physical action space with human descriptions, DIRSPEED frees the network from implicitly disentangling global coordinates, transforming a complex reasoning problem into a direct geometric mapping.

Moving beyond theoretical representation, evaluating a model’s grasp of real-world cinematography demands a foundation of deeply contextualized data. We therefore construct CINESCRIPT, a comprehensive real-movie benchmark comprising approximately 28K clips and 10M frames drawn from diverse public sources [12–16]. Going beyond prior datasets that rely solely on local motion captions [8–10], CINESCRIPT provides screenplay-style loglines, which act as concise textual summaries of scene context and narrative action, as well as real movie-level global attributes (e.g., era, genre, director) retrieved by linking identifiable clips to public knowledge bases. This rich annotation enables the study of scene-aware generation and opens up the exploration of higher-level cinematic attributes—an indispensable yet historically overlooked dimension of camera motion.

While CINESCRIPT unlocks the exploration of high-level cinematic concepts, a critical bottleneck remains in evaluating the foundational alignment between camera trajectories and local motion descriptions. Reliably quantifying this core motion–text alignment calls for a dedicated discriminative protocol. However, prior baselines (e.g., CLaTr [9–11, 17]) often entangle text alignment with heavy trajectory-reconstruction objectives, which can obscure true instance-level correspondence. By stripping away this generative overhead and establishing a lightweight, purely contrastive setup, we explicitly unveil the value of the motion-centric approach, demonstrating that DIRSPEED dramatically improves text–trajectory retrieval over pose-based alternatives regardless of the underlying evaluator. Building on this validated representation, we propose CINEGEN, a novel generative model for camera trajectories. Existing standard diffusion models [8, 9] lack a mechanism for progressive motion commitment, while strictly causal autoregressive models [10] rely on a unidirectional generation order that may limit the flexibility of global cinematic planning. To bridge this gap, CINEGEN employs a text-conditioned masked autoregressive (MAR) architecture [18–20]. This formulation provides the best of both worlds, naturally balancing progressive step-by-step commitment with the bidirectional context necessary to satisfy global cinematic constraints.

Our experiments unveil the hidden fundamental value of motion-centric representations in this task. Under DIRSPEED, text alignment improves noticeably compared to pose-based alternatives, and our contrastive evaluation protocol establishes a strictly more reliable evaluation space than CLaTr. Building on these insights, CINEGEN achieves superior performance in both trajectory quality and text alignment. Furthermore, CINESCRIPT enables us to step beyond local motion descriptions and explore the previously overlooked correlation between camera motion and high-level movie attributes. Our probes confirm that the generated trajectories successfully preserve broader cinematic patterns, such as era, genre, and directorial style, highlighting exciting new avenues for future exploration. Ultimately, these integrated components establish a thoroughly language-aligned framework for understanding and generating cinematic camera motion.

## 2 Related Work

Camera trajectory generation. Recent research has shifted from rule-based virtual cinematography to data-driven generative models. CCD [8] pioneered text-conditioned trajectory diffusion, which E.T. [9] later extended to real-movie data alongside the CLaTr evaluation metric. GenDoP [10] further advanced this domain using causal autoregressive models trained on real-world datasets. Concurrently, the scope of trajectory generation has expanded significantly: Director3D [21] jointly generates trajectories and 3D scenes, PulpMotion [11] enforces actor-camera coherence, and ShotVerse [22] tackles multi-shot cinematic planning. In adjacent embodied domains, NWM [23] and DVGFormer [24] explore navigation and drone control. Additionally, studies like Seeing without Pixels [25] highlight that trajectories inherently possess semantic meaning alignable with language. While these diverse efforts primarily focus on novel generative architectures or expanded task definitions, they largely inherit pose-centric parameterizations. Our work departs from this norm by identifying the underlying trajectory representation as a critical bottleneck, demonstrating that a motion-centric reformulation can fundamentally bridge the semantic gap between physical geometry and natural language.

![](images/56e6224934260c5c8ebaaa0cba756c5d04e7141495bd82a576a7d4b6677f9e92.jpg)  
Figure 1: Overview of CINESCRIPT construction and trajectory representation. Movie clips are re-processed into camera trajectories, motion captions, screenplay-style loglines, and linked movie attributes. The right panel illustrates two kinds of representations mentioned in our paper.

Masked autoregressive models. Masked autoregressive (MAR) models combine the bidirectional context of masked modeling with the progressive commitment of autoregressive decoding. Initially popularized for discrete visual tokens through works like MaskGIT [18], Muse [26], and MAGVIT [27], the paradigm has recently shifted toward continuous-token generation. Methods such as GIVT [28], MAR [19], and Fluid [20] bypass vector quantization entirely, directly modeling real valued sequences with diffusion-based prediction heads. This continuous approach has demonstrated strong scalability for complex spatiotemporal and planning tasks [29–31]. Our approach extends this continuous MAR paradigm to the domain of camera trajectory generation. However, unlike image and video models that must rely on lossy autoencoders to compress high-dimensional visual vocabularies, cinematic camera trajectories are natively compact. We exploit this unique property to apply masked autoregression directly in the continuous feature space, enabling a generation process that naturally balances progressive motion commitment with bidirectional cinematic context.

## 3 DIRSPEED: A Motion-Centric Trajectory Representation

As established in Sec. 1, the choice of trajectory representation is not a minor implementation detail; it fundamentally dictates how well a model can align visual motion with natural language. Most prior work parameterizes camera trajectories using absolute per-frame poses. We compare this standard pose representation, denoted by POSE9D, with our proposed motion-centric direction-speed representation, denoted by DIRSPEED. Fig. 1 provides a visual demonstration of the two approaches.

Original pose parameterization (POSE9D). For each frame t, let the camera pose be given by a rotation matrix $\mathbf { \bar { \boldsymbol { R } } } _ { t } \in S O ( 3 )$ and a translation vector $\mathbf { t } _ { t } \in \mathbb { R } ^ { 3 }$ . Following prior work, we represent the rotation by its continuous 6D form [32], denoted by $\phi ( R _ { t } ) \in \mathbb { R } ^ { 6 }$ , and concatenate it with translation as $\mathbf { x } _ { t } ^ { \mathrm { p o s e } } \doteq [ \phi ( R _ { t } ) , \mathbf { t } _ { t } ] \in \mathbb { R } ^ { 9 }$ . While this formulation is geometrically complete, it forces models to implicitly deduce relative motion from absolute coordinates, leading to the geometric entanglement that makes text–trajectory alignment difficult.

Direction-speed parameterization (DIRSPEED). To bridge the gap between physical geometry and natural language, we introduce DIRSPEED, a representation designed to be both semantically aligned and numerically robust. We begin with the intuition that human motion captions describe frame-to-frame camera behavior—specifically, the direction and speed of movement—rather than absolute spatial coordinates. Driven by this semantic alignment, we first compute the translational and rotational velocities from the pose sequence $\{ ( R _ { t } , \mathbf { t } _ { t } ) \} _ { t = 1 } ^ { T }$ for $t = 2 , \ldots , \bar { T }$

$$
\begin{array} { r } { \Delta \mathbf { t } _ { t } = \mathbf { t } _ { t } - \mathbf { t } _ { t - 1 } , \qquad \omega _ { t } = \log _ { S O ( 3 ) } ( R _ { t - 1 } ^ { \top } R _ { t } ) , } \end{array}
$$

where $\log _ { S O ( 3 ) }$ maps a relative rotation matrix to its axis-angle vector. To align sequence lengths, we set $\Delta \mathbf { t } _ { 1 } = \mathbf { 0 }$ and $\boldsymbol { \omega } _ { 1 } = \mathbf { 0 }$ as zero placeholders.

Crucially, raw velocity vectors still entangle the direction of movement with its overall speed. To isolate the exact motion factors described by language, we explicitly decompose each velocity vector into a normalized unit direction and a scalar speed:

$$
\mathbf { d } _ { t } ^ { \mathrm { t r } } = \frac { \Delta \mathbf { t } _ { t } } { \| \Delta \mathbf { t } _ { t } \| + \varepsilon } , \qquad s _ { t } ^ { \mathrm { t r } } = \log ( \| \Delta \mathbf { t } _ { t } \| + \varepsilon ) ,
$$

$$
{ \bf d } _ { t } ^ { \mathrm { r o t } } = \frac { \omega _ { t } } { \| \omega _ { t } \| + \varepsilon } , \qquad s _ { t } ^ { \mathrm { r o t } } = \log ( \| \omega _ { t } \| + \varepsilon ) .
$$

Here, ${ \bf d } _ { t } ^ { \mathrm { t r } } , { \bf d } _ { t } ^ { \mathrm { r o t } } \in \mathbb { R } ^ { 3 }$ encode the pure directions of translation and rotation, with normalization explicitly isolating geometric orientation from scale. For the speed components $s _ { t } ^ { \mathrm { t r } }$ and $s _ { t } ^ { \mathrm { r o t } }$ , we apply a logarithmic transformation to compress the massive dynamic range of real cinematic movements into a stable distribution. The final per-step feature vector concatenates these components as $\mathbf { x } _ { t } ^ { \mathrm { { d s } } } =$ $[ \mathbf { d } _ { t } ^ { \mathrm { t r } } , \ \mathbf { d } _ { t } ^ { \mathrm { r o t } } , \ s _ { t } ^ { \mathrm { t r } } , \ s _ { t } ^ { \mathrm { r o t } } ] \ \in \ \mathbb { R } ^ { 8 }$ . Ultimately, this explicit decoupling of bounded directions and logcompressed speeds provides a natively language-aligned and numerically stable foundation that significantly eases neural network optimization, which we empirically validate in Sec. 7.

## 4 The CINESCRIPT Dataset

Existing datasets [8–11] for text-conditioned camera trajectory generation mainly provide motionlevel supervision, pairing trajectories solely with direct movement descriptions. To ground our study in actual filmmaking and support scene-aware exploration, we construct CINESCRIPT, which extends this setting with screenplay-style loglines and linked real-movie attributes.

Fig. 1 illustrates the construction of CINESCRIPT. Starting from heterogeneous public movie-clip datasets, we first apply quality filtering and spatial pre-processing to ensure visual consistency (detailed in the following filtering step). We then extract per-frame camera trajectories using VIPE [33], generate motion captions via motion tagging and large language models (LLMs), produce screenplaystyle loglines using vision-language models (VLMs), and link identifiable clips to real movie metadata via public knowledge bases (e.g., Wikipedia and IMDb). This pipeline ensures that each retained clip is associated with synchronized visual content, camera motion, motion-level text, scene-level context, and, when available, higher-level movie attributes.

Clip collection and filtering. To ensure a diverse distribution of cinematic styles, shot patterns, and scene content, we construct CINESCRIPT from movie clips drawn from five public datasets: ShotBench [12], CineTechBench [13], MovieShots [14], CMD [15], and the film-derived subset of VADB [16]. Because these sources are heterogeneous in scale and provenance, we re-process all clips through a unified pipeline rather than using any source annotations as-is. During this stage, we apply rigorous quality-control filters to remove clips with advertisement or UI artifacts, discard low-quality videos, and exclude abnormal aspect ratios. We also crop black borders prior to pose extraction to cleanly isolate the core content regions.

Camera trajectories. For each retained clip, we extract per-frame camera poses using VIPE [33]. The extracted trajectories are subsequently cleaned, smoothed, and converted into fixed-length sequences by truncating clips above the 90th length percentile and padding shorter clips with masks.

Motion captions. We generate camera-motion captions following the motion-tagging and captiongeneration procedure of E.T. [9]. Specifically, each trajectory is segmented into temporally coherent motion primitives using velocity-based thresholding. The resulting structured tags are converted into natural-language descriptions by an LLM (Mistral-7B [34]), such as “a steady dolly-in with a pan right.” This annotation captures exactly what the camera does, serving as the primary supervision for motion-level controllability.

Loglines. Crucially, each clip is also paired with a screenplay-style logline describing the visual scene. Generated using Qwen3-VL-32B-Instruct [35] via a structured prompt and manually reviewed for consistency, the final loglines follow the format [INT./EXT.] [Location] – [Time] – [Action] (see Fig. 1 for an example). Unlike motion captions, loglines summarize spatial and narrative context without describing the camera movement itself. Because coarse motion captions alone cannot capture the full physical nuance of a camera path, this complementary text allows us to explore how the exact execution of a given motion naturally adapts to different scene contexts, revealing stylistic subtleties that motion text inherently misses. Details for logline generation are provided in Appendix A.

Movie attribute linking. Finally, for clips whose filenames preserve identifiable information (e.g., movie title or IMDb ID), we link them to real movie metadata through public databases (Wikidata, Wikipedia, and IMDb). We retrieve attributes such as release year, genre, and director. This metadata allows us to go beyond local motion evaluation and explore the often-overlooked correlation between camera behavior and macro-level cinematic style.

Dataset Statistics. CINESCRIPT comprises approximately 28K clips and 10M frames, with an average duration of 12.0 seconds per clip. Compared to prior datasets [8–11], CINESCRIPT is the first one to jointly contain motion captions, scene loglines, and real movie attributes (in a subset of the clips). This resulting metadata-linked subset contains 3,163 clips from roughly 1,400 unique films.

![](images/47aa7303104b0b65b8dd510d33ac05fef53f706e31478162cf7bdcbf3a2f0f4f.jpg)  
(a) Motion primitive distribution

![](images/dffe44907748a8067a11eb1d0802a053c0c4866dbe04637d04e9ecc43c265b65.jpg)  
(b) Dolly-in with different movie attributes  
Figure 2: Data statistics and examples in CINESCRIPT. Left: distribution of extracted translation (yellow) and rotation (green) primitives. Right: examples of clips with a similar coarse motion pattern (dolly-in) but different movie attributes, illustrating that camera style can vary substantially even under the same high-level motion.

Figure 2(a) summarizes the distribution of the extracted motion primitives. While dominated by static and simple single-axis motions, which accurately reflects the natural imbalance of real cinematic camera behavior, the dataset still contains a substantial volume of multi-axis and composite motions to provide necessary diversity. Furthermore, Fig. 2(b) illustrates that clips sharing the same coarse motion pattern (e.g., a dolly-in) can exhibit markedly different trajectory styles depending on their movie attributes. This validates our motivation for incorporating metadata to explore cinematic structures beyond basic motion alignment. More detailed statistics are provided in Appendix B.

## 5 A Closer Look at Evaluation

While decomposing camera motion into direction and speed is mathematically simple, this representational shift yields profound benefits for both trajectory alignment and generation. To accurately quantify these advantages, however, it is essential to first establish a reliable evaluation framework. Prior work commonly relies on CLaTr [9] for evaluation, which attempts to perform both text alignment and trajectory reconstruction simultaneously. This dual objective often obscures true alignment quality. To establish a clearer standard, we introduce a lightweight, purely contrastive evaluation protocol. Furthermore, leveraging CINESCRIPT, we complement standard text alignment with attribute-based probes to measure how well our representation captures higher-level cinematic information. The right panel of Fig. 4 summarizes this evaluation framework.

## 5.1 Alignment Evaluation

Unless otherwise specified, we evaluate alignment using motion captions, which provide the most direct text–trajectory correspondence. To project these modalities into a shared space, we employ a frozen CLIP ViT-L/14 [36] model as our text encoder $E _ { y }$ and a lightweight Transformer as our trajectory encoder $E _ { x }$ . For a batch of B matched pairs, let $\widehat { \mathbf { h } } _ { i } ^ { y } = E _ { y } ( y _ { i } )$ and $\widehat { \mathbf { h } } _ { i } ^ { x } = E _ { x } ( x _ { i } )$ denote the $L _ { 2 } .$ -normalized embeddings for text and trajectory, respectively. Both CLaTr and our proposed protocol utilize a symmetric InfoNCE contrastive loss to align these representations:

$$
\mathcal { L } _ { \mathrm { N C E } } = \frac { 1 } { 2 } ( \mathcal { L } _ { y  x } + \mathcal { L } _ { x  y } ) , \quad \mathrm { w h e r e } \quad \mathcal { L } _ { y  x } = - \frac { 1 } { B } \sum _ { i = 1 } ^ { B } \log \frac { \exp ( \langle \widehat { \mathbf { h } } _ { i } ^ { y } , \widehat { \mathbf { h } } _ { i } ^ { x } \rangle / \tau ) } { \sum _ { j \in \mathcal { N } ( i ) } \exp ( \langle \widehat { \mathbf { h } } _ { i } ^ { y } , \widehat { \mathbf { h } } _ { j } ^ { x } \rangle / \tau ) } .
$$

Here, τ is a fixed temperature and $\mathcal { N } ( i )$ contains the positive pair alongside non-duplicate negatives. However, CLaTr [9] functions as a retrieval-VAE, coupling this contrastive loss with a heavy trajectory reconstruction objective. While this multi-objective design is useful for general representation learning, it compromises the precise discriminative margins required to evaluate strict instancelevel alignment. To establish a more rigorous standard, we introduce a deterministic, encoder-only evaluation protocol. By stripping away the generative overhead and training solely with the contrastive objective $( \bar { \mathcal { L } } _ { \mathrm { N C E } } )$ , this lightweight setup strictly isolates pure alignment quality.

Table 1: Alignment evaluator validation on motion captions. We compare CLaTr and our purely contrastive protocol under both trajectory representations.
<table><tr><td>Model</td><td>Rep.</td><td>R@1↑</td><td>R@5↑</td><td>R@10↑</td><td>MedR↓</td><td>AlignScore ↑</td><td>#Params</td></tr><tr><td>Ours</td><td>DIRSPEED POSE9D</td><td>25.2 17.8</td><td>40.7 26.5</td><td>49.0 31.4</td><td>11 51</td><td>66.3 46.6</td><td>3.6M 3.6M</td></tr><tr><td></td><td>DIRSPEED</td><td>19.7</td><td>30.5</td><td>39.2</td><td>23</td><td>70.6</td><td>30M</td></tr><tr><td>CLaTr [9]</td><td>POSE9D</td><td>6.9</td><td>12.0</td><td>14.7</td><td></td><td></td><td></td></tr><tr><td></td><td></td><td></td><td></td><td></td><td>361</td><td>37.4</td><td>30M</td></tr></table>

![](images/7911672b8505aa87c44c093b07a1d6f90dbb5b7a396e306d1970ffb95d1e934b.jpg)  
(a) CLaTr-DIRSPEED

![](images/eb556245ac246f4a3886f54a8ccd6d3cbeb85e23878220a7be13dc226d3250a0.jpg)  
(b) CLaTr-POSE9D

![](images/930b6c21202c4533dcd25af1335eee3ab643fc4d3ac88d2304b3be7768ddb08b.jpg)  
(c) Ours-DIRSPEED

![](images/e16cd6ba28e6bd9dfdf480d0f68704d3daa1d2f77a5d44680c60cd6887d25d99.jpg)  
(d) Ours-POSE9D  
Figure 3: Joint embedding spaces of the four alignment evaluators on the validation set, projected jointly with t-SNE. Trajectory embeddings (•) and matched text embeddings (⋆) are coloured by the same K-means cluster (computed on trajectory embeddings, k = 8); thin lines connect 200 random matched (•, ⋆) pairs.

Crucially, this clean evaluation space explicitly unveils the value of our motion-centric approach. As shown in Table 1, our purely contrastive protocol significantly improves retrieval performance over CLaTr, with DIRSPEED yielding a consistent and surprising boost regardless of the underlying evaluator (e.g., R@1 rising from 17.8 to 25.2). These gains are visually corroborated by the joint embedding spaces in Fig. 3. In the DIRSPEED variants, trajectory points (•) and matched text stars (⋆) of the same cluster color are more integrated into cohesive cross-modal groups with minimal connecting lines. In contrast, the POSE9D variants exhibit disjoint color regions and numerous long lines, indicating that matched pairs remain far apart and poorly aligned. These results confirm that, despite its structural simplicity, mapping motion to direction and speed resolves geometric entanglement far more effectively than absolute poses.

Metrics. To support both the immediate validation of our alignment space (Table 1) and the subsequent evaluation of our generative model, we establish two sets of functional metrics. First, for text–trajectory alignment, we compute the Alignment Score (AlignScore), defined concisely as the mean clipped cosine similarity $\begin{array} { r } { \frac { 1 0 0 } { N _ { \cdot } } \sum _ { i } \operatorname* { m a x } ( 0 , \langle \widehat { \mathbf { h } } _ { i } ^ { y } , \widehat { \mathbf { h } } _ { i } ^ { x } \rangle ) } \end{array}$ , alongside retrieval recall (R@1, 5, 10) and Median Rank (MedR) in the trajectory-to-text direction. Second, for evaluating the physical realism and diversity of final generated outputs, we assess trajectory quality using motion-primitive F1 (following the segmentation protocol of [9, 10]), Fréchet Camera Distance (FCD) and Manifold Coverage [37] computed in our new contrastive embedding space. Further implementation details are provided in Appendix C.

## 5.2 Attribute-based Evaluation

Standard text alignment confirms adherence to local motion descriptions. However, as suggested by our earlier dataset statistics (Fig. 2(b)), the physical execution of a camera path encodes rich stylistic nuances that coarse motion captions simply cannot capture. To formally test whether a representation preserves these macro-level signatures, we train classifiers solely on real trajectories using the movie metadata in CINESCRIPT.

Specifically, we construct three classification probes targeting Era, Genre, and Director. To mitigate the heavy long-tail distribution inherent in movie metadata, we group related genres into three broad buckets (Drama/Romance, Comedy, and Action/Thriller/Sci-Fi) and focus the directorial probe on three iconic stylists: Christopher Nolan, Wes Anderson, and Steven Spielberg. To ensure a rigorous evaluation, we restrict each task to these target categories, preventing classifiers from achieving artificially high performance by simply predicting a dominant background class. The Era probe follows a similar balanced design, formulated as a single-label task across three historical bins (Film, Early digital, and Digital mature). More details are previded in Appendix D.

Table 2 presents the performance of these attribute probes on real validation trajectories. While the absolute F1-scores suggest that these high-level attributes are not trivially mapped from motion alone, the results confirm that camera trajectories do carry a measurable stylistic signal. Directorial style, in particular, emerges as the most distinctive attribute among the three. Notably, DIRSPEED consistently provides a more reliable signal than POSE9D across all tasks. This performance gap suggests that our decomposition helps expose subtle cinematic signatures that are otherwise difficult to capture through absolute coordinates. We leverage these real-trained probes to monitor attribute preservation in our generative experiments in Sec. 7.

Table 2: Attribute prediction on real trajectories. All entries report macro-F1 (×100).
<table><tr><td>Attribute</td><td>Task</td><td>#Class</td><td>DIRSPEED</td><td>POSE9D</td><td>∆</td></tr><tr><td>Era</td><td>single-label</td><td>3</td><td>49.8</td><td>47.3</td><td>+2.6</td></tr><tr><td>Genre</td><td>multi-label</td><td>3</td><td>62.1</td><td>60.7</td><td>+1.4</td></tr><tr><td>Director</td><td>single-label</td><td>3</td><td>68.2</td><td>51.8</td><td>+16.4</td></tr></table>

## 6 Generation

With the representation and evaluation protocol established, we now model the conditional distribution of continuous camera trajectories. Unlike image-generation tasks where Masked Auto-Regressive (MAR) [19] models typically operate on compressed latent tokens, our trajectory features are natively low-dimensional. We therefore design CINEGEN to generate directly in the trajectory feature space, bypassing the need for a lossy autoencoding bottleneck. As illustrated in Fig. 4 (left), the model consists of two core components: a Transformer-based [38] MAR sequencer that captures dependencies among partially visible trajectory tokens, and a conditional diffusion denoiser that samples the continuous values for missing positions.

Masked sequence modeling. Let ${ \bf x } = ( { \bf x } _ { 1 } , \dots , { \bf x } _ { T } ) \in \mathbb { R } ^ { T \times D }$ represent a trajectory sequence. During training, we randomly sample a set of masked positions $\mathcal { M } \subseteq \{ 1 , \ldots , T \}$ and replace them with a learned mask embedding m $\mathbf { \Psi } \in \mathbb { R } ^ { D }$

$$
\begin{array} { r } { \mathbf { x } _ { i } ^ { \mathrm { m a s k e d } } = \left\{ \begin{array} { l l } { \mathbf { m } , } & { i \in \mathcal { M } , } \\ { \mathbf { x } _ { i } , } & { i \notin \mathcal { M } . } \end{array} \right. } \end{array}
$$

The resulting sequence is processed by a Transformer-based sequencer Seq, which employs selfattention to compute context vectors $\dot { \bf H } ^ { \mathrm { s e q } }$ from the visible tokens and conditioning signals:

$$
{ \bf H } ^ { \mathrm { s e q } } = \mathrm { S e q } ( { \bf x } ^ { \mathrm { m a s k e d } } , { \bf c } ) , \quad { \bf H } ^ { \mathrm { s e q } } \in \mathbb { R } ^ { T \times h } .
$$

Text and first-pose conditioning. The conditioning vector c integrates three distinct signals: the motion caption, the logline, and the initial camera pose. We encode the two text components using a frozen CLIP ViT-B/32 [36] to obtain $\mathbf { e } _ { \mathrm { m o t i o n } }$ and e<sub>logline</sub>.

The first-pose feature $\mathbf { x } _ { \mathrm { f p } }$ ensures the generated motion remains geometrically grounded to the starting viewpoint. For the POSE9D representation, we use $\mathbf { x } _ { 1 } ^ { \mathrm { p o s e } }$ as defined in Sec. 3. For DIRSPEED, we construct an 8D first-pose feature by applying the same direction-speed decomposition to the initial camera state $( R _ { 1 } , \mathbf { t } _ { 1 } )$ . These components are fused via a small MLP, $f _ { \mathrm { c o n d } }$ , to produce the final conditioning vector: $\mathbf { c } = f _ { \mathrm { c o n d } } ( [ \mathbf { e } _ { \mathrm { m o t i o n } } , \mathbf { e } _ { \mathrm { l o g l i n e } } , \mathbf { x } _ { \mathrm { f p } } ] )$ . This signal is injected into both the sequencer and the denoiser through adaptive layer normalization (AdaLN).

Diffusion denoiser for continuous tokens. For each masked position $j \in { \mathcal { M } }$ , we model the continuous value of $\mathbf { x } _ { j }$ using a conditional diffusion process. Following the DDPM [39] forward process, we sample a diffusion step τ and Gaussian noise ϵ to form a noised token $\mathbf { x } _ { j , \tau }$ . An

![](images/f538a47f5e043ecd4cfbbc84849efbe3c9a9a6a6bdfc1aa00575b55f4c4bbb37.jpg)  
Figure 4: Overview of the proposed generation and evaluation framework. Left: CINEGEN integrates motion captions, loglines, and initial pose constraints to synthesize camera trajectories. The model progressively recovers masked trajectory tokens using a Transformer-based sequencer and a conditional denoiser through a variance-guided unmasking process. Right: Our evaluation protocol employs a purely contrastive InfoNCE objective to align motion captions and trajectories within a normalized joint embedding space, providing a rigorous measure of cross-modal consistency.

MLP-based denoiser $\epsilon _ { \theta }$ then predicts the noise conditioned on the sequencer’s output:

$$
\begin{array} { r } { \widehat { \mathbf { \epsilon } } = \epsilon _ { \theta } ( \mathbf { x } _ { j , \tau } , \tau , \mathbf { H } _ { j } ^ { \mathrm { s e q } } ) . } \end{array}
$$

The entire system is trained jointly to minimize the denoising error:

$$
\begin{array} { r } { \mathcal { L } _ { \mathrm { M A R } } = \mathbb { E } _ { j \in \mathcal { M } , \tau , \epsilon } \left[ \| \widehat { \epsilon } - \epsilon \| _ { 2 } ^ { 2 } \right] . } \end{array}
$$

Variance-guided unmasking order. At inference, CINEGEN must determine the order in which masked positions are revealed. Rather than a fixed linear order, we employ a variance-guided policy. At each autoregressive step s, we compute the variance across the h channels of the context vector $\mathbf { H } _ { j } ^ { \mathrm { s e q } }$ for each remaining masked position $j \in \mathcal { M } _ { s }$ . Let $H _ { j , k } ^ { \mathrm { s e q } }$ denote the k-th component of this vector; we score each position by:

$$
\rho _ { j } = \frac { 1 } { h } \sum _ { k = 1 } ^ { h } \left( H _ { j , k } ^ { \mathrm { s e q } } - \bar { H } _ { j } ^ { \mathrm { s e q } } \right) ^ { 2 } , \quad \mathrm { w h e r e } \quad \bar { H } _ { j } ^ { \mathrm { s e q } } = \frac { 1 } { h } \sum _ { k ^ { \prime } = 1 } ^ { h } H _ { j , k ^ { \prime } } ^ { \mathrm { s e q } } .
$$

We then reveal the $n _ { s }$ positions with the smallest variance, $\begin{array} { r } { \mathcal { P } _ { s } = \arg \operatorname* { m i n } _ { \mathcal { P } \subseteq \mathcal { M } _ { s } , | \mathcal { P } | = n _ { s } } \sum _ { j \in \mathcal { P } } \rho _ { j } \mathrm { , } } \end{array}$ based on the intuition that lower variance indicates higher contextual confidence. This process repeats until the full trajectory is synthesized and converted back to per-frame poses.

## 7 Experiments

We evaluate CINEGEN against prior baselines on CINESCRIPT to quantify how the shift from traditional poses to our motion-centric (DIRSPEED) representation improves generation quality. Unless otherwise stated, all metrics employ our contrastive evaluation protocol (Sec. 5) using the DIRSPEED encoder.

Setup and Baselines. We compare against three representative text-to-trajectory methods: CCD [8], E.T. [9], and GENDOP [10]. For each baseline, we evaluate both the original released pretrained checkpoints (denoted by †) and versions retrained on CINESCRIPT to ensure a fair comparison. To avoid data leakage in our attribute-based evaluation, we train our movie-attribute classifiers on a separate movie-level split, ensuring that no clips from the same film are shared between classifier training and generation validation. All methods are evaluated using the metrics defined in Sec. 5.

Comparison with Prior Methods. As Table 3 shows, CINEGEN consistently outperforms all baselines across trajectory quality, text alignment, and cinematic attribute preservation. Compared to the strongest retrained baseline (GENDOP), our model achieves a substantial leap in retrieval R@1 (from 0.93 to 3.35) while reducing FCD by nearly 70%. Beyond local motion fidelity, CINEGEN achieves the highest macro-F1 across all attribute probes, successfully capturing the macro-level cinematic structures identified in our dataset (Fig. 2). Supported by qualitative examples of smoother, more coherent compound paths (Fig. 5), these comprehensive gains demonstrate that generating directly in the DIRSPEED space yields trajectories that are more realistic and precisely grounded.

Table 3: Comparison with prior text-to-trajectory methods on CINESCRIPT. ∗ denotes evaluation using released pretrained checkpoints, while other baseline rows are trained on CINESCRIPT. The light green row highlights our model. Bold indicates the best result.
<table><tr><td></td><td colspan="3">Trajectory Quality</td><td colspan="3">Text-Trajectory Alignment</td><td colspan="3">Movie Attributes</td></tr><tr><td>Method / Setting</td><td>F1↑</td><td>FCD↓</td><td>Coverage ↑</td><td>AlignScore ↑</td><td>R@1↑</td><td>MedR↓</td><td>Era↑</td><td>Genre ↑</td><td>Director ↑</td></tr><tr><td>CCD* [8]</td><td>0.117</td><td>52.45</td><td>0.298</td><td>4.90</td><td>0.08</td><td>1032.5</td><td>44.3</td><td>46.3</td><td>12.1</td></tr><tr><td>CCD [8]</td><td>0.128</td><td>49.19</td><td>0.270</td><td>18.82</td><td>0.43</td><td>397.0</td><td>47.2</td><td>56.3</td><td>32.7</td></tr><tr><td>E.T.* [9]</td><td>0.015</td><td>135.74</td><td>0.034</td><td>0.00</td><td>0.08</td><td>960.0</td><td>40.7</td><td>56.7</td><td>25.0</td></tr><tr><td>E.T. [9]</td><td>0.000</td><td>179.88</td><td>0.014</td><td>0.36</td><td>0.16</td><td>745.0</td><td>40.7</td><td>45.8</td><td>33.6</td></tr><tr><td>GENDoP* [10]</td><td>0.175</td><td>102.97</td><td>0.085</td><td>7.63</td><td>0.23</td><td>715.5</td><td>33.5</td><td>45.6</td><td>18.1</td></tr><tr><td>GENDoP [10]</td><td>0.234</td><td>22.10</td><td>0.586</td><td>33.11</td><td>0.93</td><td>195.0</td><td>33.2</td><td>52.5</td><td>29.9</td></tr><tr><td>CINEGEN (ours)</td><td>0.437</td><td>6.77</td><td>0.783</td><td>57.79</td><td>3.35</td><td>52.0</td><td>47.8</td><td>60.7</td><td>48.1</td></tr></table>

Crucially, the advantages of our motion-centric approach extend beyond our specific architecture. As detailed in Appendix E, retrofitting prior generative models with DIRSPEED broadly improves their trajectory quality and alignment, demonstrating its utility as a strong inductive bias. However, representation alone is insufficient; state-of-the-art performance relies on the synergy between this parameterization and our CINEGEN framework. Ablations (Appendix H) corroborate this: while reverting to absolute poses triggers the most severe degradation across all metrics, omitting the first-pose anchor or replacing variance-guided unmasking with a random schedule also leads to a substantial drop in overall trajectory quality. Furthermore, removing the auxiliary logline noticeably weakens both trajectory realism and movie-attribute preservation. While DIRSPEED intrinsically ensures precise geometric alignment, Appendix J qualitatively demonstrates that the logline provides the essential narrative anchor required for stylistically nuanced cinematic generation.

![](images/048a62eb67bb11576fef929dbac333c74bb78f5a743a73d1675ea77018468636.jpg)  
Figure 5: Qualitative comparison with prior generation methods. Starred columns denote released pretrained checkpoints, corresponding to ∗ in Table 3. Compared with prior methods, CINEGEN produces smoother and more coherent camera paths that better reflect compound motion instructions.

## 8 Conclusion

In this paper, we demonstrate that cinematic camera motion is fundamentally defined not just by where the camera is positioned, but by how it moves in terms of direction and speed. By decomposing the camera trajectory into these components (DIRSPEED), we unlock surprising benefits for both trajectory alignment and generation. Beyond its numerical advantages, this motion-centric approach inherently allows models to capture the higher-level cinematographic information that traditional absolute poses obscure. To support and rigorously evaluate and validate this insight, we introduced CINESCRIPT for scene-level and attribute-based analysis, a purely contrastive alignment protocol, and CINEGEN, a tailored masked autoregressive generator. Together, these contributions overcome the limitations of prior pose-centric baselines, establishing a new standard for synthesizing realistic and precisely text-aligned camera paths. Future work will leverage this motion-centric foundation to explore richer multimodal conditioning and dedicated architectures for controllable cinematic styling.

## References

[1] Christophe Lino and Marc Christie. Intuitive and efficient camera control with the toric space. In ACM SIGGRAPH 2015 Conferences, pages 1–7, 2015.

[2] Quentin Galvane, Christophe Lino, Marc Christie, Julien Fleureau, Fabien Servant, François Nicolas, and Philippe Tariel. Directing cinematographic drones. ACM Transactions on Graphics (TOG), 37(3):1–18, 2018.

[3] Jianhong Bai, Menghan Xia, Xiao Fu, Xintao Wang, Lianrui Mu, Jinwen Cao, Zuozhu Liu, Haoji Hu, Xiang Bai, Pengfei Wan, and Di Zhang. Recammaster: Camera-controlled generative rendering from a single video. In Proceedings ofthe IEEE/CVF International Conference on Computer Vision (ICCV), 2025.

[4] Mark YU, Wenbo Hu, Jinbo Xing, and Ying Shan. Trajectorycrafter: Redirecting camera trajectory for monocular videos via diffusion models. In Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV), 2025.

[5] Zhouxia Wang, Ziyang Yuan, Xintao Wang, Tianshui Chen, Menghan Xia, Ping Luo, and Ying Shan. Motionctrl: A unified and flexible motion controller for video generation. In ACM SIGGRAPH 2024 Conference Papers, 2024.

[6] Hao He, Yinghao Xu, Yuwei Guo, Gordon Wetzstein, Bo Dai, Hongsheng Li, and Ceyuan Yang. Cameractrl: Enabling camera control for text-to-video generation. In European Conference on Computer Vision (ECCV), 2024.

[7] Yuwei Guo, Ceyuan Yang, Anyi Rao, Yaohui Wang, Yu Qiao, Dahua Lin, and Bo Dai. Animatediff: Animate your personalized text-to-image diffusion models without specific tuning. In The Twelfth International Conference on Learning Representations (ICLR), 2024.

[8] Hongda Jiang, Xi Wang, Marc Christie, Libin Liu, and Baoquan Chen. Cinematographic camera diffusion model. Computer Graphics Forum (Eurographics), 43(2), 2024.

[9] Robin Courant, Nicolas Dufour, Xi Wang, Marc Christie, and Vicky Kalogeiton. E.t. the exceptional trajectories: Text-to-camera-trajectory generation with character awareness. In Proceedings ofthe IEEE/CVF European Conference on Computer Vision (ECCV), 2024.

[10] Mengchen Zhang, Tong Wu, Jing Tan, Ziwei Liu, Gordon Wetzstein, and Dahua Lin. Gendop: Auto-regressive camera trajectory generation as a director of photography. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2025.

[11] Robin Courant, Xi Wang, David Loiseaux, Marc Christie, and Vicky Kalogeiton. Pulp motion: Framing-aware multimodal camera and human motion generation. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2025.

[12] Hongbo Liu, Jingwen He, Yi Jin, Dian Zheng, Yuhao Dong, Fan Zhang, Ziqi Huang, Yinan He, Yangguang Li, Weichao Chen, Yu Qiao, Wanli Ouyang, Shengjie Zhao, and Ziwei Liu. Shotbench: Expert-level cinematic understanding in vision-language models. In Advances in Neural Information Processing Systems (NeurIPS), 2025.

[13] Xinran Wang, Songyu Xu, Xiangxuan Shan, Yuxuan Zhang, Muxi Diao, Xueyan Duan, Yanhua Huang, Kongming Liang, and Zhanyu Ma. Cinetechbench: A benchmark for cinematographic technique understanding and generation. In Advances in Neural Information Processing Systems (NeurIPS), 2025.

[14] Anyi Rao, Jiaze Wang, Linning Xu, Xuekun Jiang, Qingqiu Huang, Bolei Zhou, and Dahua Lin. A unified framework for shot type classification based on subject centric lens. In Proceedings ofthe IEEE/CVF European Conference on Computer Vision (ECCV), 2020.

[15] Max Bain, Arsha Nagrani, Andrew Brown, and Andrew Zisserman. Condensed movies: Story based retrieval with contextual embeddings. In Proceedings of the Asian Conference on Computer Vision (ACCV), 2020.

[16] Qianqian Qiao, DanDan Zheng, Yihang Bo, Bao Peng, Heng Huang, Longteng Jiang, Huaye Wang, Jingdong Chen, Jun Zhou, and Xin Jin. Vadb: A large-scale video aesthetic database with professional and multi-dimensional annotations. In Advances in Neural Information Processing Systems (NeurIPS), 2025.

[17] Mathis Petrovich, Michael J. Black, and Gül Varol. TMR: Text-to-motion retrieval using contrastive 3D human motion synthesis. In Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV), 2023.

[18] Huiwen Chang, Han Zhang, Lu Jiang, Ce Liu, and William T. Freeman. MaskGIT: Masked generative image transformer. In Proceedings ofthe IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2022.

[19] Tianhong Li, Yonglong Tian, He Li, Mingyang Deng, and Kaiming He. Autoregressive image generation without vector quantization. In Advances in Neural Information Processing Systems (NeurIPS), 2024.

[20] Lijie Fan, Tianhong Li, Siyang Qin, Yuanzhen Li, Chen Sun, Michael Rubinstein, Deqing Sun, Kaiming He, and Yonglong Tian. Fluid: Scaling autoregressive text-to-image generative models with continuous tokens. In International Conference on Learning Representations, 2025.

[21] Xinyang Li, Zhangyu Lai, Linning Xu, Yansong Qu, Liujuan Cao, Shengchuan Zhang, Bo Dai, and Rongrong Ji. Director3d: Real-world camera trajectory and 3d scene generation from text. In Proceedings of the IEEE/CVF European Conference on Computer Vision (ECCV), 2024.

[22] Songlin Yang, Zhe Wang, Xuyi Yang, Songchun Zhang, Xianghao Kong, Taiyi Wu, Xiaotong Zhao, Ran Mirror Zhang, Alan Zhao, and Anyi Rao. Shotverse: Advancing cinematic camera control for text-driven multi-shot video creation. In Proceedings ofthe IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2026.

[23] Amir Bar, Gaoyue Zhou, Danny Tran, Trevor Darrell, and Yann LeCun. Navigation world models. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2025.

[24] Yunzhong Hou, Liang Zheng, and Philip Torr. Learning camera movement control from realworld drone videos. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2025.

[25] Zihui Xue, Kristen Grauman, Dima Damen, Andrew Zisserman, and Tengda Han. Seeing without pixels: Perception from camera trajectories. In Proceedings ofthe IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2026.

[26] Huiwen Chang, Han Zhang, Jarred Barber, AJ Maschinot, Jose Lezama, Lu Jiang, Ming-Hsuan Yang, Kevin Murphy, William T. Freeman, Michael Rubinstein, Yuanzhen Li, and Dilip Krishnan. Muse: Text-to-image generation via masked generative transformers. In Proceedings ofthe 40th International Conference on Machine Learning, 2023.

[27] Lijun Yu, Yong Cheng, Kihyuk Sohn, Jose Lezama, Han Zhang, Huiwen Chang, Alexander G. Hauptmann, Ming-Hsuan Yang, Yuan Hao, Irfan Essa, and Lu Jiang. MAGVIT: Masked generative video transformer. In Proceedings ofthe IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2023.

[28] Michael Tschannen, Cian Eastwood, and Fabian Mentzer. GIVT: Generative infinite-vocabulary transformers. In European Conference on Computer Vision, 2024.

[29] Ting Yao, Yehao Li, Yingwei Pan, Zhaofan Qiu, and Tao Mei. Denoising token prediction in masked autoregressive models. In Proceedings of the IEEE/CVF International Conference on Computer Vision, 2025.

[30] Haozhe Liu, Shikun Liu, Zijian Zhou, Mengmeng Xu, Yanping Xie, Xiao Han, Juan C. Pérez, Ding Liu, Kumara Kahatapitiya, Menglin Jia, Jui-Chieh Wu, Sen He, Tao Xiang, Jürgen Schmidhuber, and Juan-Manuel Pérez-Rúa. MarDini: Masked autoregressive diffusion for video generation at scale. arXiv preprint arXiv:2410.20280, 2024.

[31] Deyu Zhou, Quan Sun, Yuang Peng, Kun Yan, Runpei Dong, Duomin Wang, Zheng Ge, Nan Duan, Xiangyu Zhang, Lionel M. Ni, and Heung-Yeung Shum. Taming teacher forcing for masked autoregressive video generation. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2025.

[32] Yi Zhou, Connelly Barnes, Jingwan Lu, Jimei Yang, and Hao Li. On the continuity of rotation representations in neural networks. In Proceedings ofthe IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2019.

[33] Jiahui Huang, Qunjie Zhou, Hesam Rabeti, Aleksandr Korovko, Huan Ling, Xuanchi Ren, Tianchang Shen, Jun Gao, Dmitry Slepichev, Chen-Hsuan Lin, Jiawei Ren, Kevin Xie, Joydeep Biswas, Laura Leal-Taixe, and Sanja Fidler. Vipe: Video pose engine for 3d geometric perception. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2025.

[34] Albert Q. Jiang, Alexandre Sablayrolles, Arthur Mensch, Chris Bamford, Devendra Singh Chaplot, Diego de las Casas, Florian Bressand, Gianna Lengyel, Guillaume Lample, Lucile Saulnier, Lélio Renard Lavaud, Marie-Anne Lachaux, Pierre Stock, Teven Le Scao, Thibaut Lavril, Thomas Wang, Timothée Lacroix, and William El Sayed. Mistral 7b, 2023. URL https://arxiv.org/abs/2310.06825.

[35] Qwen Team. Qwen3-vl technical report, 2025.

[36] Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, et al. Learning transferable visual models from natural language supervision. In International Conference on Machine Learning, pages 8748–8763, 2021.

[37] Muhammad Ferjad Naeem, Seong Joon Oh, Youngjung Uh, Yunjey Choi, and Jaejun Yoo. Reliable fidelity and diversity metrics for generative models. In Proceedings of the International Conference on Machine Learning (ICML), 2020.

[38] Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N. Gomez, Łukasz Kaiser, and Illia Polosukhin. Attention is all you need. In Advances in Neural Information Processing Systems 30: Annual Conference on Neural Information Processing Systems 2017, pages 6000–6010, 2017.

[39] Jonathan Ho, Ajay Jain, and Pieter Abbeel. Denoising diffusion probabilistic models. In Advances in Neural Information Processing Systems (NeurIPS), 2020.

## Appendix

This appendix provides supplementary details, extended experimental results, and ablation studies to support the core findings in the main text. We begin by detailing the logline generation pipeline and additional dataset statistics. We then formalize the evaluation metrics and the attribute-based evaluation protocol. Finally, we provide an extended experiments and an empirical analysis of text choice for alignment evaluation.

## A Logline Generation Details

Motivation. Motion captions describe local camera behavior, such as panning, dollying, or tracking. While they provide direct supervision for motion-level controllability, they ignore the visual and narrative context in which the motion occurs. To address this, we associate each clip with a screenplaystyle logline that summarizes the scene content. The goal of the logline is not to prescribe a strict, unique camera trajectory, but rather to provide scene-level context for studying text-to-trajectory generation.

Generation procedure. We generate loglines using Qwen3-VL-32B-Instruct [35]. The visionlanguage model takes the video clip as input and is prompted to summarize the visible scene in a concise, screenplay-style format. Crucially, the prompt instructs the model to focus on the setting, time of day, main subject, and visible action, while strictly avoiding any camera-motion descriptions.

Prompt used for annotation. The following prompt illustrates the exact instruction used in our pipeline:

Role: You are a professional Assistant Director and Script Supervisor.

Task: Analyze the provided video clip and write a concise screenplay-style logline describing the scene.

Goal: Produce a short scene description that captures the setting, time of day, main subject, and visible action. The description should summarize what is happening in the video, not how the camera moves.

Constraints:

1. Do not mention camera motion, camera direction, zoom, pan, tilt, dolly, or tracking.

2. Use present tense.

3. Keep the description concise and visually grounded.

4. Avoid hallucinating details that are not visible or strongly implied by the video.

5. Use standard screenplay-style format.

Output format:

[INT./EXT.] [Specific Location] - [Time of Day] - [Brief

Action/Subject Description]

Examples:

INT. HOSPITAL CORRIDOR - NIGHT - A nurse runs toward the emergency room.

EXT. MOUNTAIN RIDGE - DAY - Clouds roll over the jagged peaks.

Output only the logline.

Output format. The final logline follows a standard screenplay-inspired format, e.g.:

EXT. STONE BRIDGE OVER RIVER – DAY – A lone hiker in red crosses a stone bridge under a clear sky.

The descriptions are consistently kept in the present tense.

Quality control. We manually review and correct generated loglines to ensure consistency and quality. Specifically, we fix malformed screenplay formats, remove hallucinated details that conflict with the visual evidence, and condense overly verbose descriptions. This manual review ensures that the loglines reliably serve as conditioning inputs for studying the relationship between scene context and camera motion.

Relation to motion captions. Motion captions and loglines provide complementary supervision. A motion caption dictates the camera action directly (e.g., “the camera slowly dollies in”). A logline instead establishes the narrative context (e.g., “INT. DIMLY LIT ROOM – NIGHT – A detective studies a wall ofphotographs”). Consequently, motion captions provide a relatively strict text– trajectory correspondence, whereas loglines are inherently underdetermined—many plausible camera trajectories might suit the same scene. For this reason, we rely on motion captions as the primary text for alignment evaluation, while utilizing loglines as an auxiliary conditioning signal for generation.

## B Additional Dataset Statistics

Movie attribute distribution. A subset of CINESCRIPT can be explicitly linked to real movie metadata, enabling the attribute-based analyses presented in the main paper. Figure 6 visualizes the distribution of this linked subset across era, genre, and director labels. The era labels are historically imbalanced, with a heavy concentration in the “Digital mature” period. Genre labels exhibit more diversity but remain skewed toward several dominant categories. Director labels show a severe long-tail distribution: even the most frequently credited directors account for only a small fraction of the total dataset, with thousands of others appearing only rarely. To mitigate this heavy long-tail effect during evaluation, we restrict our attribute-based probes to frequent-class subsets, as detailed in Sec. 5.2.

![](images/61639558f6ac6877e57c1404cf9f1f4410eab426e8d9c245450bee58a722d71a.jpg)

![](images/c1050968e50551c151a54e23ce60fba6aa5624df9f8259b40087ab833119d8aa.jpg)

![](images/4a1b6a52c197b66319919468aad0dcd6363961756faa9d8b88b42669962ff5e3.jpg)  
Figure 6: Distribution of linked movie attributes in CINESCRIPT. Left: era distribution. Middle: genre distribution. Right: director distribution, highlighting only the top 10 named directors alongside the grouped long tail. This inherent imbalance motivates our use of targeted, frequent-class subsets for rigorous attribute evaluation.

Comparison with prior datasets. Table 4 compares CINESCRIPT against representative datasets for text-to-camera-trajectory generation. While comparable in raw scale to recent movie-based datasets, CINESCRIPT is uniquely distinguished by its annotation richness. By jointly providing motion captions, loglines, and a metadata-linked subset with real movie attributes, it uniquely supports both motion- and scene-conditioned generation, alongside high-level cinematographic evaluation.

Table 4: Comparison of datasets for text-to-camera-trajectory generation. CINESCRIPT is distinguished by the joint availability of motion captions, loglines, and linked real movie attributes.
<table><tr><td></td><td colspan="3">Scale</td><td colspan="3">Annotations</td><td></td></tr><tr><td>Dataset</td><td>#Samples</td><td>#Frames</td><td>Avg(s)</td><td>motion</td><td>logline</td><td>movie attributes</td><td>Source</td></tr><tr><td>CCD [8]</td><td>25K</td><td>4.5M</td><td>7.2</td><td>√</td><td>x</td><td>x</td><td>Synthetic</td></tr><tr><td>E.T. [9]</td><td>115K</td><td>11M</td><td>3.8</td><td>√</td><td>x</td><td>x</td><td>Movie</td></tr><tr><td>DataDoP [10]</td><td>29K</td><td>11M</td><td>14.4</td><td>√</td><td>x</td><td>x</td><td>Movie</td></tr><tr><td>CINESCRIPT (ours)</td><td>28K</td><td>10M</td><td>12.0</td><td>√</td><td>√</td><td>√(3.2K)</td><td>Movie</td></tr></table>

## C Alignment Evaluation Metrics

This section formalizes the text–trajectory alignment and trajectory-quality metrics utilized in our evaluation framework.

## C.1 Text–Trajectory Alignment Metrics

Unless otherwise specified, alignment is evaluated using the motion caption paired with each trajectory. For a validation set of N matched pairs $\{ ( \mathbf { x } _ { i } , \mathbf { y } _ { i } ) \} _ { i = 1 } ^ { N }$ , we extract normalized trajectory and text embeddings:

$$
\widehat { \mathbf { h } } _ { i } ^ { x } = \frac { E _ { x } ( \mathbf { x } _ { i } ) } { \| E _ { x } ( \mathbf { x } _ { i } ) \| _ { 2 } } , \qquad \widehat { \mathbf { h } } _ { i } ^ { y } = \frac { E _ { y } ( \mathbf { y } _ { i } ) } { \| E _ { y } ( \mathbf { y } _ { i } ) \| _ { 2 } } ,
$$

where $E _ { x }$ and $E _ { y }$ are the trajectory and text encoders, respectively. The resulting cosine-similarity matrix is:

$$
S _ { i j } = \left. \widehat { \mathbf { h } } _ { i } ^ { x } , \widehat { \mathbf { h } } _ { j } ^ { y } \right. , \qquad S \in \mathbb { R } ^ { N \times N } ,
$$

where the diagonal entry $S _ { i i }$ represents the similarity of the matched pair.

Alignment score. The alignment score (AlignScore) computes the average clipped cosine similarity of all matched pairs:

$$
{ \mathrm { A l i g n S c o r e } } = { \frac { 1 0 0 } { N } } \sum _ { i = 1 } ^ { N } \operatorname* { m a x } \left( 0 , S _ { i i } \right) .
$$

While this captures absolute matched-pair similarity, it does not evaluate discrimination against negative distractors. Because separately trained evaluators define different embedding spaces, their absolute AlignScore values are not directly comparable; we use retrieval metrics for cross-evaluator comparisons.

Retrieval recall. For trajectory-to-text retrieval, each trajectory embedding $\widehat { \mathbf { h } } _ { i } ^ { x }$ acts as a query, and all text embeddings $\{ \widehat { \mathbf { h } } _ { j } ^ { y } \} _ { j = 1 } ^ { N }$ are ranked by descending similarity $S _ { i j }$ . Let rank<sub>i</sub> be the 1-indexed rank of the correct text $\mathbf { \bar { y } } _ { i } .$ . Ignoring ties, this is defined as:

$$
\mathrm { r a n k } _ { i } = 1 + \sum _ { j \neq i } { \bf 1 } \left[ S _ { i j } > S _ { i i } \right] ,
$$

where $\mathbf { 1 } [ \cdot ]$ is the indicator function. The Recall@K metric is then:

$$
\mathsf { R @ { K } } = \frac { 1 0 0 } { N } \sum _ { i = 1 } ^ { N } \mathbf { 1 } \left[ \mathrm { r a n k } _ { i } \le K \right] , \qquad K \in \{ 1 , 5 , 1 0 \} .
$$

In practice, ties are resolved by assigning the average rank among tied candidates.

Median rank. The Median Rank (MedR) summarizes the overall retrieval distribution:

$$
\mathrm { M e d R } = \mathrm { m e d i a n } \left( \mathrm { r a n k } _ { 1 } , \dots , \mathrm { r a n k } _ { N } \right) .
$$

A lower MedR indicates that, on average, the correct match is ranked closer to the top. The text-totrajectory direction is computed symmetrically using $S ^ { \top }$ . As noted in the main text, trajectory-to-text retrieval serves as our primary metric, since it directly answers whether the text conditioning is recoverable from the generated physical motion.

## C.2 Trajectory-Quality Metrics

For physical quality evaluation, each camera trajectory is treated as a sequence of camera-to-world matrices $\hat { P _ { 1 : T } } = ( \hat { P _ { 1 } } , \ldots , P _ { T } )$ , where $P _ { t } \in S E ( 3 )$

Motion-primitive F1. This metric evaluates instance-level structural fidelity. For each consecutive frame pair, we extract the relative transformation:

$$
\Delta _ { t } = P _ { t } ^ { - 1 } P _ { t + 1 } , \qquad t = 1 , \ldots , T - 1 .
$$

From $\Delta _ { t } .$ , we isolate the translational velocity $\mathbf { v } _ { t } \in \mathbb { R } ^ { 3 }$ and angular velocity $\boldsymbol { \omega } _ { t } \in \mathbb { R } ^ { 3 }$ . We quantize the translation into 27 sign-based bins $( \{ - 1 , \bar { 0 } , + 1 \} ^ { 3 } )$ and the rotation into 7 bins (stationary plus positive/negative rotation around each dominant axis). This yields a 189-class label $\ell _ { t } \in \{ 0 , . . . , \bar { 1 8 } 8 \}$ for each frame transition. Following prior protocols, the label sequence is smoothed via mode filtering.

Let Prec and Rec represent the precision and recall for class c, and $n _ { c }$ the number of reference frames belonging to c. The weighted multi-class F1 is:

$$
\mathrm { F } 1 _ { c } = \frac { 2 \mathrm { P r e c } _ { c } \mathrm { R e c } _ { c } } { \mathrm { P r e c } _ { c } + \mathrm { R e c } _ { c } } , \qquad \mathrm { F } 1 = \frac { \sum _ { c } n _ { c } \mathrm { F } 1 _ { c } } { \sum _ { c } n _ { c } } .
$$

Fréchet Camera Distance. FCD measures the distributional gap between real and generated trajectories within the learned feature space of our contrastive trajectory encoder. Let ϕ(·) denote the unnormalized embedding. For sets of real $( X _ { r } )$ and generated $( X _ { g } )$ trajectories, we compute embeddings $\mathbf { u } _ { i } ^ { r } = \phi ( \mathbf { x } _ { i } ^ { r } )$ and $\mathbf { u } _ { j } ^ { g } = \phi ( \mathbf { x } _ { j } ^ { g } )$ . Fitting Gaussian statistics to these sets yields moments $\left( \pmb { \mu } _ { r } , \pmb { \Sigma } _ { r } \right)$ and $( \mu _ { g } , \Sigma _ { g } ) \colon$

$$
{ \pmb \mu } = \frac { 1 } { N } \sum _ { i } { \bf u } _ { i } , \qquad { \pmb \Sigma } = \frac { 1 } { N - 1 } \sum _ { i } ( { \bf u } _ { i } - { \pmb \mu } ) ( { \bf u } _ { i } - { \pmb \mu } ) ^ { \top } .
$$

The distance is then defined as:

$$
\begin{array} { r } { \mathrm { F C D } = \| \pmb { \mu _ { r } } - \pmb { \mu _ { g } } \| _ { 2 } ^ { 2 } + \operatorname { T r } ( \pmb { \Sigma _ { r } } ) + \operatorname { T r } ( \pmb { \Sigma _ { g } } ) - 2 \operatorname { T r } \left( ( \pmb { \Sigma _ { r } } \pmb { \Sigma _ { g } } ) ^ { 1 / 2 } \right) . } \end{array}
$$

Manifold Coverage. Coverage assesses the proportion of the real trajectory manifold successfully reached by generated samples, again utilizing the contrastive trajectory encoder. We compute $L _ { 2 } .$ -normalized embeddings:

$$
\bar { \mathbf { u } } _ { i } ^ { r } = \frac { \phi ( \mathbf { x } _ { i } ^ { r } ) } { \| \phi ( \mathbf { x } _ { i } ^ { r } ) \| _ { 2 } } , \qquad \bar { \mathbf { u } } _ { j } ^ { g } = \frac { \phi ( \mathbf { x } _ { j } ^ { g } ) } { \| \phi ( \mathbf { x } _ { j } ^ { g } ) \| _ { 2 } } .
$$

For each real embedding $\bar {  { \mathbf { u } } } _ { i } ^ { r }$ , we calculate the distance to its k-th nearest real neighbor $\left( k = 3 \right)$

$$
{ \rho } _ { i } = \left\| \bar { \mathbf { u } } _ { i } ^ { r } - \bar { \mathbf { u } } _ { i , ( k ) } ^ { r } \right\| _ { 2 } .
$$

A real sample is considered “covered” if at least one generated sample falls within this local radius:

$$
\mathrm { c o v } _ { i } = \mathbf { 1 } \left[ \operatorname* { m i n } _ { j } \left\| \bar { \mathbf { u } } _ { i } ^ { r } - \bar { \mathbf { u } } _ { j } ^ { g } \right\| _ { 2 } < \rho _ { i } \right] .
$$

The final Coverage score averages this indicator across the real set:

$$
\mathrm { C o v } = \frac { 1 } { N _ { r } } \sum _ { i = 1 } ^ { N _ { r } } \mathrm { c o v } _ { i } .
$$

## D Details of Attribute-based Evaluation

Task formulation. We utilize movie metadata linked from public knowledge bases to construct the three attribute probes (Era, Genre, and Director) summarized in Table 5. Era and Director are formulated as single-label classification tasks. Given that modern films frequently span multiple stylistic categories, Genre is treated as a multi-label classification task.

Table 5: Attribute task definitions. Clips outside the target classes are completely dropped rather than assigned to a generic “Other” class. This strictly prevents the classifiers from cheating the metric by defaulting to a trivial majority-class background prediction.
<table><tr><td>Attribute</td><td>Classes</td><td>Type</td></tr><tr><td>Era</td><td>Film era (&lt; 2005); Early digital (2005–2012); Digital mature (≥ 2013)</td><td>single-label</td></tr><tr><td>Genre</td><td>Drama/Romance; Comedy; Action/Thriller/Sci-Fi</td><td>multi-label</td></tr><tr><td>Director</td><td>Christopher Nolan; Wes Anderson; Steven Spielberg</td><td>single-label</td></tr></table>

Genre coarsening. To handle the extreme granularity and noise in Wikidata genre labels, we map them into coarse groups and merge related themes into three robust buckets. Drama and Romance are merged into one bucket, while action-oriented genres (Action, Thriller/Horror, Sci-Fi/Fantasy, Adventure) are collapsed into another. This prevents arbitrary decision boundaries among overlapping themes. Clips containing exclusively documentary or biographical labels are discarded for this specific probe.

Classifier architecture. The attribute classifier consists of a small Transformer encoder processing the trajectory features, fused with a trajectory-statistics vector, a first-pose summary, and a frozen

CLIP feature extracted from the first frame’s depth map. Separate probes are trained from scratch for DIRSPEED and POSE9D to ensure fair comparison.

Evaluation on generated trajectories. These probes measure preservation of latent attribute correlations rather than explicit controllable style, since the generators are not conditioned on era, genre, or director labels. The classifiers are trained on a separate movie-level split, so clips from the same film do not appear in both classifier training and validation. For generated trajectories, every compared method is evaluated with the same frozen classifier and identical first-pose/depth auxiliary inputs; only the generated trajectory differs across methods. We evaluate all generated validation clips whose corresponding source clips possess defined labels for a given task. Generated trajectories are never used to train the classifiers.

## E Extended Baseline Comparison

Table 6 expands the main comparison with diagnostic variants, denoted by <sup>†</sup>, that replace each baseline’s native trajectory output with continuous DIRSPEED prediction while retaining the rest of its generative framework. These variants isolate how well the representation transfers across architectures. As in the main paper, <sup>∗</sup> denotes released pretrained checkpoints, while unmarked rows denote versions retrained on CINESCRIPT using each method’s native formulation. The retrained E.T. row uses its complete released training and sampling pipeline, including EDM preconditioning, EMA, and valid-length masking.

The diagnostic variants show that DIRSPEED generally benefits motion F1 and language alignment, but the gains are not uniform across all distributional metrics. In particular, GENDOP was originally designed to predict categorical distributions over discrete pose tokens; replacing this output space with continuous motion changes the demands placed on the same causal autoregressive framework and can affect FCD and Coverage. This motivates pairing the continuous representation with a generator designed for it, rather than interpreting representation changes independently of architecture.

Table 6: Extended comparison against prior generation methods on CINESCRIPT. <sup>∗</sup> denotes released pretrained checkpoints; unmarked rows denote baselines retrained on CINESCRIPT using their native formulations; and <sup>†</sup> denotes diagnostic variants retrained with continuous DIRSPEED prediction and motion-caption conditioning. Movie-attribute metrics report macro-F1 (×100).
<table><tr><td rowspan="2">Method</td><td colspan="3">Trajectory Quality</td><td colspan="3">Text-Trajectory Alignment</td><td colspan="3">Movie Attributes</td></tr><tr><td>F1↑</td><td>FCD↓</td><td>Coverage ↑</td><td>AlignScore ↑</td><td>R@1↑</td><td>MedR↓</td><td>Era↑</td><td>Genre ↑</td><td>Director ↑</td></tr><tr><td>CCD*[8]</td><td>0.117</td><td>52.45</td><td>0.298</td><td>4.90</td><td>0.08</td><td>1032.5</td><td>44.3</td><td>46.3</td><td>12.1</td></tr><tr><td>CCD [8]</td><td>0.128</td><td>49.19</td><td>0.270</td><td>18.82</td><td>0.43</td><td>397.0</td><td>47.2</td><td>56.3</td><td>32.7</td></tr><tr><td>CCD† [8]</td><td>0.168</td><td>25.96</td><td>0.576</td><td>11.71</td><td>0.35</td><td>690.0</td><td>42.5</td><td>48.5</td><td>39.8</td></tr><tr><td>E.T.* [9]</td><td>0.015</td><td>135.74</td><td>0.034</td><td>0.00</td><td>0.08</td><td>960.0</td><td>40.7</td><td>56.7</td><td>25.0</td></tr><tr><td>E.T. [9]</td><td>0.194</td><td>16.39</td><td>0.596</td><td>29.78</td><td>0.81</td><td>251.0</td><td>42.0</td><td>57.5</td><td>28.1</td></tr><tr><td>E.T.† [9]</td><td>0.338</td><td>48.32</td><td>0.277</td><td>25.22</td><td>0.19</td><td>389.5</td><td>40.0</td><td>50.2</td><td>37.2</td></tr><tr><td>GENDoP* [10]</td><td>0.175</td><td>102.97</td><td>0.085</td><td>7.63</td><td>0.23</td><td>715.5</td><td>33.5</td><td>45.6</td><td>18.1</td></tr><tr><td>GENDoP [10]</td><td>0.234</td><td>22.10</td><td>0.586</td><td>33.11</td><td>0.93</td><td>195.0</td><td>33.2</td><td>52.5</td><td>29.9</td></tr><tr><td>GENDoP† [10]</td><td>0.293</td><td>55.30</td><td>0.260</td><td>40.62</td><td>0.93</td><td>124.0</td><td>40.5</td><td>48.4</td><td>30.6</td></tr><tr><td>CINEGEN (Ours)</td><td>0.437</td><td>6.77</td><td>0.783</td><td>57.79</td><td>3.35</td><td>52.0</td><td>47.8</td><td>60.7</td><td>48.1</td></tr></table>

Conditioning-matched comparison The native baselines use motion-caption conditioning, whereas the full CINEGEN model additionally uses the scene logline and first-pose anchor. Adding these modules to the baselines would substantially alter their original architectures, so we retain their released conditioning interfaces and instead remove both auxiliary conditions from CINEGEN for a fully matched caption-only comparison.

The full and caption-only CINEGEN variants achieve comparable trajectory quality, while the caption-only model scores higher on caption-based alignment because it focuses exclusively on the signal used by those metrics. Under the matched caption-only setting, CINEGEN still outperform GENDOP across all reported metrics, showing that richer conditioning does not explain the main performance gap.

Table 7: Conditioning-matched comparison. The caption-only CINEGEN variant removes both logline and first-pose conditioning.
<table><tr><td>Method</td><td>F1↑</td><td>FCD↓</td><td>Coverage ↑</td><td>AlignScore ↑</td><td>R@1↑</td><td>MedR↓</td><td>Era↑</td><td>Genre ↑</td><td>Director↑</td></tr><tr><td>CINEGEN</td><td>0.437</td><td>6.77</td><td>0.783</td><td>57.79</td><td>3.35</td><td>52</td><td>47.8</td><td>60.7</td><td>48.1</td></tr><tr><td>CINEGEN caption-only</td><td>0.419</td><td>6.76</td><td>0.749</td><td>65.83</td><td>5.82</td><td>42</td><td>45.6</td><td>60.6</td><td>41.6</td></tr><tr><td>GENDoP [10]</td><td>0.234</td><td>22.10</td><td>0.586</td><td>33.11</td><td>0.93</td><td>195</td><td>33.2</td><td>52.5</td><td>29.9</td></tr></table>

![](images/5d6612884e50894f62b1a3b79a4c731c256bb049ac5dfed5e2ac20f4e0968585.jpg)  
Multiple selections per clip are allowed, so bars do not sum to 100%. 24 participants, 30 clips, 720 judgements

Figure 7: Blinded human evaluation across 30 clips and 24 participants. Multiple selections are allowed, so the selection-rate bars do not sum to 100%.

## F Human Evaluation

We conduct a blinded multi-selection study on 30 randomly sampled CameraBench test clips. For each clip, the source scene and the camera-conditioned renderer are fixed; only the generated trajectory changes. We generate one trajectory from each of 7 methods, re-anchor it to the source pose, and rescale its translation extent to match the reference. Participants view the reference and seven anonymized re-renderings and select every version whose camera motion is smooth, natural, and follows the reference direction. Multiple selections are allowed, and the same rendering settings and random seed are used for all methods.

24 participants completed all clips, yielding 720 method-level judgements. One participant was an author; removing that response changes the selection rate by only 0.2 percentage points and leaves the conclusions unchanged. We report the selection rate (the fraction of judgements in which a method is selected) and the stricter sole-selection rate (the fraction in which it is the only selected method). Since several methods can be selected for one clip, selection rates do not sum to 100%.

Figure 7 summarizes the results. CINEGEN is selected in 70.1% of judgements and is the sole selection in 30.1%, compared with 28.9% and 6.2% for the strongest baseline, the retrained GENDOP. Every participant selected CINEGEN more often than that baseline individually (two-sided sign test, $p < \bar { 1 0 ^ { - 6 } } )$ . The released GenDoP and CCD models are selected in less than 1% of judgements, while their re-implemented counterparts recover part of the gap, supporting the importance of the trajectory representation in this comparison.

## G Evaluator Robustness and Retrieval Context

Independent frozen-CLaTr evaluation. To ensure that the main generation results do not depend on our DIRSPEED-trained evaluation space, we re-evaluate all methods using the frozen official CLaTr encoder released with E.T.. Table 8 shows that CINEGEN remains best on all metrics under this independent embedding space.

Table 8: Independent evaluation using the frozen official CLaTr encoder. <sup>∗</sup> denotes released pretrained checkpoints, while unmarked baselines are retrained on CINESCRIPT using their native formulations.
<table><tr><td>Method</td><td>FCD↓</td><td>Coverage ↑</td><td>R@1↑</td><td>MedR↓</td></tr><tr><td>CCD* [8]</td><td>195.23</td><td>0.293</td><td>0.39</td><td>1158.5</td></tr><tr><td>CCD [8]</td><td>567.67</td><td>0.253</td><td>0.43</td><td>965.5</td></tr><tr><td>E.T.* [9]</td><td>1311.94</td><td>0.006</td><td>0.04</td><td>1175.0</td></tr><tr><td>E.T. [9]</td><td>111.71</td><td>0.593</td><td>0.39</td><td>558.0</td></tr><tr><td>GENDoP* [10]</td><td>339.73</td><td>0.188</td><td>0.12</td><td>810.0</td></tr><tr><td>GENDoP [10]</td><td>59.85</td><td>0.656</td><td>0.27</td><td>467.5</td></tr><tr><td>CINEGEN (Ours)</td><td>30.08</td><td>0.748</td><td>1.36</td><td>320.5</td></tr></table>

AlignScore interpretation. Absolute cosine values from separately trained evaluators are not directly comparable. On the same validation set, CLaTr-DIRSPEED gives a larger matched cosine but also higher and substantially more dispersed mismatched similarities:

Table 9: Matched and mismatched cosine statistics for the two DIRSPEED evaluators.
<table><tr><td>Evaluator</td><td>Matched cosine ↑</td><td>Mismatched cosine</td><td>R@1↑</td><td>MedR↓</td></tr><tr><td>CLaTr-DIRSPEED</td><td>70.6</td><td> $3 . 2 \pm 3 1 . 4$ </td><td>19.7</td><td>23</td></tr><tr><td>Ours-DIRSPEED</td><td>66.3</td><td> $- 0 . 5 \pm 2 2 . 8$ </td><td>25.2</td><td>11</td></tr></table>

Accordingly, we use retrieval metrics for cross-evaluator comparison and AlignScore only within a fixed evaluator.

Absolute retrieval context. The validation retrieval pool contains approximately 2.5K candidates, so random R@1 is about 0.04%. In addition, retrieval counts only the designated text–trajectory pair as correct even though multiple trajectories can plausibly satisfy the same motion instruction. CINEGEN reaches R@1 of 3.35%, about 84× random, and its R@5/R@10 are 12.67%/19.79%, compared with 4.03%/7.33% for the strongest retrained caption-only baseline, GENDOP. We therefore interpret retrieval as a discriminative alignment diagnostic rather than a claim of near-perfect instance-level matching.

## H Ablation Studies and Extended Analysis

Design choices of CINEGEN. To identify the specific drivers behind CINEGEN’s performance, we systematically ablate its key architectural and conditioning choices. As reported in Table 10, we evaluate the necessity of our trajectory representation, multi-modal conditioning signals, and variance-guided unmasking policy.

Consistent with our core premise, replacing the DIRSPEED representation with standard poses (POSE9D rep.) triggers the most severe degradation across all quality, alignment, and cinematic metrics. Removing either the initial geometric anchor $( w / o \ \mathbf { x } _ { \mathrm { f p } } )$ or the scene-level context (w/o $\mathbf { e } _ { \mathrm { l o g l i n e } } )$ similarly harms performance, proving that text descriptions alone are insufficient to constrain highly plausible physical paths. Finally, while random unmasking (w/o variance) slightly elevates manifold coverage, it sacrifices crucial stability, particularly harming alignment and directorial style preservation.

Detailed DIRSPEED component ablation. To precisely isolate why DIRSPEED yields such powerful alignment signals, we ablate its mathematical components directly within our contrastive evaluation architecture (training a new probe from scratch for each variant). In this analysis, all feature vectors refer to per-step quantities, and we omit the time index t for readability. We specifically compare direction-only features $[ \mathbf { d } ^ { \mathrm { t r } } , \mathbf { d } ^ { \mathrm { r o t } } ]$ , speed-only features $[ s ^ { \mathrm { t r } } , s ^ { \mathrm { r o t } } ]$ , raw velocity $[ \Delta \mathbf { t } , \omega ]$ , and the full DIRSPEED representation.

As Table 11 demonstrates, normalized direction provides the vast majority of the alignment signal; relying on speed alone is insufficient for effective retrieval. However, combining both elements produces the highest R@K and MedR, verifying their fundamental complementarity. Importantly, the raw velocity baseline performs poorly. This confirms that simply taking temporal differences is

Table 10: Ablation study of CINEGEN on CINESCRIPT. Underlined entries indicate specific metrics where an ablated variant marginally exceeds the full model.
<table><tr><td rowspan="2">Setting</td><td colspan="3">Trajectory Quality</td><td colspan="3">Text-Trajectory Alignment</td><td colspan="3">Movie Attributes</td></tr><tr><td>F1↑</td><td>FCD↓</td><td>Coverage ↑</td><td>AlignScore ↑</td><td>R@1↑</td><td>MedR↓</td><td>Era↑</td><td>Genre ↑</td><td>Director ↑</td></tr><tr><td>CINEGEN (ours)</td><td>0.437</td><td>6.77</td><td>0.783</td><td>57.79</td><td>3.35</td><td>52.0</td><td>47.8</td><td>60.7</td><td>48.1</td></tr><tr><td>POSE9D rep.</td><td>0.189</td><td>20.99</td><td>0.638</td><td>41.11</td><td>1.94</td><td>110.5</td><td>42.5</td><td>58.2</td><td>29.8</td></tr><tr><td>w/o Xfp</td><td>0.425</td><td>15.13</td><td>0.722</td><td>53.07</td><td>3.84</td><td>54.0</td><td>40.2</td><td>61.2</td><td>35.2</td></tr><tr><td>w/o elogline</td><td>0.402</td><td>9.75</td><td>0.769</td><td>51.16</td><td>2.44</td><td>72.0</td><td>47.7</td><td>56.1</td><td>45.1</td></tr><tr><td>w/o variance</td><td>0.401</td><td>8.84</td><td>0.784</td><td>53.60</td><td>2.79</td><td>58.0</td><td>42.1</td><td>61.2</td><td>43.2</td></tr></table>

not the silver bullet—the specific structural decomposition into normalized direction and log-speed is what unlocks the representation’s strength.

Table 11: Component ablation of DIRSPEED for text–trajectory alignment. All variants utilize the identical contrastive architecture trained from scratch. DIRSPEED dominates retrieval metrics, confirming the vital synergy between normalized direction and log-speed.
<table><tr><td>Feature</td><td>R@1↑</td><td>R@5↑</td><td>R@10↑</td><td>MedR↓</td><td>AlignScore ↑</td></tr><tr><td>direction-only</td><td>19.7</td><td>33.0</td><td>40.1</td><td>22</td><td>61.2</td></tr><tr><td>speed-only</td><td>5.7</td><td>9.1</td><td>12.3</td><td>290</td><td>46.8</td></tr><tr><td>velocity</td><td>3.8</td><td>8.0</td><td>9.3</td><td>377</td><td>48.4</td></tr><tr><td>DIRSPEED (ours)</td><td>25.2</td><td>40.7</td><td>49.0</td><td>11</td><td>66.3</td></tr></table>

Table 12: Effect of text conditioning on alignment retrieval across 2,578 validation clips. We train and evaluate our contrastive protocol using varying text inputs. Motion captions provide the unambiguous signal necessary for geometric alignment, whereas narrative loglines fail to act as reliable singular targets. Note: Random motion-to-text R@1 is approximately 0.04%.
<table><tr><td>Rep.</td><td>Text</td><td>AlignScore ↑</td><td>R@1↑</td><td>R@5↑</td><td>R@10↑</td><td>MedR↓</td></tr><tr><td rowspan="3">DIRSPEED</td><td rowspan="3">motion motion + logline logline</td><td>66.3</td><td>25.2</td><td>40.7</td><td>49.0</td><td>11</td></tr><tr><td>50.6</td><td>4.6</td><td>14.6</td><td>23.7</td><td>41</td></tr><tr><td>31.1</td><td>1.0</td><td>2.9</td><td>4.8</td><td>227</td></tr><tr><td rowspan="3">POSE9D</td><td>motion</td><td>38.7</td><td>8.7</td><td>15.9</td><td>21.5</td><td>66</td></tr><tr><td>motion + logline</td><td>33.3</td><td>2.3</td><td>7.7</td><td>12.3</td><td>102</td></tr><tr><td>logline</td><td>26.2</td><td>0.7</td><td>1.9</td><td>3.0</td><td>428</td></tr></table>

## H.1 Representation Robustness and Matched Controls

Coordinate and scale controls. Our default POSE9D representation centers translation at the first frame but retains world-frame rotation, whereas DIRSPEED uses world-frame translation increments and camera-local rotation increments. To test whether the advantage comes merely from removing coordinate gauge or trajectory scale, we additionally canonicalize POSE9D to the first camera frame and then apply the trajectory-scale normalization used by GENDOP, dividing translations by max<sub>t</sub>∥t<sub>t</sub> − t<sub>1</sub>∥<sub>2</sub>.

Canonicalization provides only a limited gain and scale normalization a moderate additional improvement, while a substantial gap to DIRSPEED remains. Thus, the representation advantage is not explained solely by coordinate or scale conventions.

Speed parameterization. We next keep the normalized direction features fixed and vary only the speed transformation. All variants use the same split, evaluator architecture, and training protocol.

The relatively narrow range across these variants confirms that normalized direction carries most of the alignment signal. The default logarithmic parameterization gives the best overall result and compresses the long-tailed distribution of camera-motion magnitudes.

Table 13: Matched coordinate and scale controls for POSE9D.
<table><tr><td>Representation</td><td>AlignScore↑</td><td>R@1↑</td><td>MedR↓</td></tr><tr><td>First-frame-canonical PosE9D</td><td>46.4</td><td>15.2</td><td>51</td></tr><tr><td>+ Scale normalization PoSE9D</td><td>49.0</td><td>18.9</td><td>40</td></tr><tr><td>DIRSPEED</td><td>66.3</td><td>25.2</td><td>11</td></tr></table>

Table 14: Ablation of the speed parameterization with direction features held fixed.
<table><tr><td>Magnitude parameterization</td><td>AlignScore ↑</td><td>R@1↑</td><td>MedR↓</td></tr><tr><td>Linear speed</td><td>63.6</td><td>21.5</td><td>14</td></tr><tr><td>Global z-score</td><td>64.7</td><td>24.4</td><td>11</td></tr><tr><td>log(1 + speed)</td><td>64.5</td><td>22.4</td><td>14</td></tr><tr><td>log(speed + ε)</td><td>66.3</td><td>25.2</td><td>11</td></tr></table>

Reconstruction and long-range fidelity. Given the initial camera-to-world pose $( R _ { 1 } , \mathbf { t } _ { 1 } )$ , a DIRSPEED sequence is converted back to poses by normalizing the predicted directions and recovering the increments

$$
\Delta { \bf t } _ { t } = \bar { { \bf d } } _ { t } ^ { \mathrm { t r } } \big ( \mathrm { e x p } ( s _ { t } ^ { \mathrm { t r } } ) - \varepsilon \big ) , \qquad \Delta R _ { t } = \mathrm { E x p } \big ( \bar { { \bf d } } _ { t } ^ { \mathrm { r o t } } \big ( \mathrm { e x p } ( s _ { t } ^ { \mathrm { r o t } } ) - \varepsilon \big ) \big ) ,
$$

where $\bar { \bf d }$ denotes the normalized predicted direction and Exp $= \exp _ { S O ( 3 ) }$ . We then recursively apply

$$
\mathbf { t } _ { t } = \mathbf { t } _ { t - 1 } + \Delta \mathbf { t } _ { t } , \qquad R _ { t } = R _ { t - 1 } \Delta R _ { t } .
$$

A pose → DIRSPEED→ pose round trip on validation trajectories longer than 200 frames yields a median endpoint error of $\mathrm { { \bar { 3 } . 8 4 \times 1 0 ^ { - 9 } } }$ in translation and 0.130<sup>◦</sup> in rotation. We also split generated outputs into short (59–95 frames) and long (≥ 213 frames) sequences:

Table 15: Generation quality on short and long trajectories.
<table><tr><td>Representation</td><td>Short F1 ↑</td><td>Long F1 ↑</td><td>Short FCD↓</td><td>Long FCD↓</td><td>Short Cov. ↑</td><td>Long Cov. ↑</td></tr><tr><td>POSE9D</td><td>0.240</td><td>0.187</td><td>10.14</td><td>33.08</td><td>0.794</td><td>0.635</td></tr><tr><td>DIRSPEED</td><td>0.411</td><td>0.419</td><td>10.72</td><td>17.72</td><td>0.820</td><td>0.786</td></tr></table>

DIRSPEED remains stable as the sequence horizon grows, while POSE9D degrades substantially on the long subset.

Joint pose–motion representation. Absolute pose can still provide useful complementary information for tasks where global spatial context matters. Concatenating a canonicalized POSE9D stream with DIRSPEED gives a small additional gain, while DIRSPEED alone captures most of the improvement with less than half the input dimensionality.

Data and model scaling. Finally, we compare how the two representations use additional data and model capacity. DIRSPEED benefits more consistently from larger training sets; with only 50% of the data it already exceeds POSE9D trained on the full set. Increasing encoder size beyond the base model yields little further improvement for either representation.

## I Experimental Settings

This section provides the comprehensive hyperparameters and architectural details for both our contrastive alignment evaluator and the CINEGEN generative model. All experiments are conducted on a single NVIDIA A100 GPU. Due to the differences in architectural complexity, the training costs vary significantly: our lightweight contrastive evaluator converges rapidly, typically within 30 epochs (requiring only ∼10 minutes of wall-clock time), whereas training the CINEGEN generative model typically requires 150 to 200 epochs, totaling approximately 12 hours.

Table 16: Joint use of pose and motion features.
<table><tr><td>Representation</td><td>Input dim.</td><td>AlignScore ↑</td><td>R@1↑</td><td>MedR↓</td></tr><tr><td>POSE9D</td><td>9</td><td>46.6</td><td>17.8</td><td>51</td></tr><tr><td>DIRSPEED</td><td>8</td><td>66.3</td><td>25.2</td><td>11</td></tr><tr><td>POSE9D+DIRSPEED</td><td>17</td><td>66.9</td><td>27.0</td><td>9</td></tr></table>

Table 17: Data scaling for POSE9D and DIRSPEED.
<table><tr><td>Training data</td><td>POSE9D R@1↑</td><td>DIRSPEED R@1↑</td></tr><tr><td>10%</td><td>11.3</td><td>15.1</td></tr><tr><td>25%</td><td>11.0</td><td>15.9</td></tr><tr><td>50%</td><td>12.2</td><td>20.2</td></tr><tr><td>100%</td><td>17.8</td><td>25.2</td></tr></table>

## I.1 Contrastive Alignment Evaluator Configurations

The contrastive evaluator is trained using a symmetric InfoNCE loss with a fixed temperature of 0.1. To prevent penalizing semantically identical captions, we apply false-negative filtering by masking off-diagonal pairs whose text-to-text cosine similarity exceeds 0.99. Unlike prior evaluators (e.g., CLaTr), our protocol strictly avoids any trajectory reconstruction, KL divergence, or decoder overhead. Detailed hyperparameters are listed in Table 19.

## I.2 CINEGEN Generative Model Configurations

CINEGEN is trained to predict denoised trajectory tokens conditioned on multimodal text inputs and initial geometric anchors. The text conditioning uses a combined approach, encoding the motion caption and logline separately. The model comprises approximately 28.4M trainable parameters (excluding the frozen text encoder and EMA copies). Detailed hyperparameters are summarized in Table 20.

## J The Complementary Roles of Motion Captions and Loglines

Throughout the main text, we consistently train and evaluate alignment using motion captions. This section provides the empirical justification for isolating this specific textual signal, while also examining the role of loglines during generation.

Alignment requires strict geometric correspondence. Given that motion captions explicitly describe mechanical camera behavior, they form a tightly coupled correspondence with the physical trajectory. Loglines, in contrast, describe contextual scene elements. Table 12 compares our contrastive protocol trained under three distinct text inputs: isolated motion captions, concatenated motion and loglines, and isolated loglines.

The empirical results perfectly align with our intuition. Motion captions yield by far the strongest text– trajectory retrieval across both DIRSPEED and POSE9D architectures. Diluting the motion caption with loglines degrades retrieval performance, and relying on loglines alone leads to a near-collapse in recall capability. This demonstrates that contextual scene descriptions are too underdetermined to serve as a direct geometric alignment target.

Table 18: Trajectory-encoder scaling.
<table><tr><td>Model</td><td>Trainable params</td><td>POSE9D R@1↑</td><td>DIRSPEED R@1↑</td></tr><tr><td>Small</td><td>0.53M</td><td>16.3</td><td>24.2</td></tr><tr><td>Base</td><td>3.49M</td><td>17.8</td><td>25.2</td></tr><tr><td>Large</td><td>26.14M</td><td>17.1</td><td>25.0</td></tr></table>

Table 19: Architecture and training hyperparameters for the Contrastive Alignment Evaluator.
<table><tr><td>Hyperparameter</td><td>Value</td></tr><tr><td colspan="2">Trajectory Encoder (Trainable, ~3.6M params) 8 (DIRSPEED) or 9 (POSE9D)</td></tr><tr><td>Input dimension Linear projection Positional encoding Transformer layers</td><td>256 Sinusoidal (max length 5000)</td></tr><tr><td>Transformer dimensions</td><td>4  $d _ { \mathrm { m o d e l } } = 2 5 6 , 4$  heads,  $d _ { \mathrm { f f } } = 1 0 2 4$ </td></tr><tr><td>Transformer configuration</td><td>Pre-LayerNorm, Dropout 0.1, GELU</td></tr><tr><td>Output pooling</td><td>Learnable [CLS] token prepended</td></tr><tr><td>Output head</td><td>LayerNorm → Linear → GELU → Linear (256 dims)</td></tr><tr><td>Text Encoder (Frozen Backbone, ∼197K trainable params)</td><td></td></tr><tr><td>Backbone</td><td>CLIP ViT-L/14 (frozen)</td></tr><tr><td>Tokens</td><td>Max 77, using pooled [EOS] embedding (768 dims)</td></tr><tr><td>Projection head</td><td></td></tr><tr><td></td><td>Linear (768 → 256)</td></tr><tr><td>Training and Optimization</td><td></td></tr><tr><td>Optimizer</td><td>AdamW  $( \beta _ { 1 } = 0 . 9 , \beta _ { 2 } = 0 . 9 9 9 ,$  weight decay  $= 1 0 ^ { - 4 } )$ </td></tr><tr><td>Learning rate</td><td> $2 \times 1 0 ^ { - 4 }$  with Cosine Annealing</td></tr><tr><td>Batch size</td><td>64</td></tr><tr><td>Gradient clipping</td><td>Max norm 1.0</td></tr><tr><td>Epochs</td><td>150 (Early stopping patience = 10)</td></tr></table>

Table 20: Architecture, sampling, and training hyperparameters for CINEGEN.
<table><tr><td>Hyperparameter</td><td>Value</td></tr><tr><td colspan="2">Conditioning and Architecture</td></tr><tr><td>Text encoder Logline embedding dim</td><td>CLIP ViT-B/32 (frozen, 512 dims) 64</td></tr><tr><td>Sequencer Sequencer dimensions</td><td>TransformerAdaLN (1 block)</td></tr><tr><td>Conditioning fusion Diffuser</td><td> $d _ { \mathrm { m o d e l } } = 5 1 2 , \ S$  heads,  $d _ { \mathrm { f f } } = 4 0 9 6 .$  Dropout 0.2 Text (512) + First Pose (64) + Logline (64) = 640 dims → 512 dims SimpleMLPAdaLN (3 ResBlocks, dmodel = 1024)</td></tr><tr><td>Masking and Sampling Masking ratio range</td><td>[0.5, 1.0] uniformly sampled during training</td></tr><tr><td>Unmasking order</td><td>Variance-guided</td></tr><tr><td>Autoregressive (AR) steps DDPM scheduling</td><td>18</td></tr><tr><td>DDPM sampling steps</td><td>Cosine  $( \mathrm { s } = 0 . 0 0 8 ) , T = 1 0 0$ </td></tr><tr><td>Classifier-Free Guidance</td><td>50</td></tr><tr><td>Training and Optimization</td><td>Drop prob = 0.1 (train), CFG scale = 3.5 (inference)</td></tr><tr><td>Optimizer</td><td></td></tr><tr><td>Learning rate</td><td>AdamW  $( \beta _ { 1 } = 0 . 9 , \beta _ { 2 } = 0 . 9 9 ,$  weight decay  $= 1 0 ^ { - 5 } )$ </td></tr><tr><td></td><td> $3 \times 1 0 ^ { - 4 }$  (Linear warmup for 2000 steps, then constant)</td></tr><tr><td>Batch size</td><td>128</td></tr><tr><td>Batch multiplier</td><td></td></tr><tr><td>Gradient clipping</td><td>5 (noise levels per masked token)</td></tr><tr><td>Epochs</td><td>Max norm 1.0</td></tr></table>

Generation benefits from narrative context. However, this lack of strict one-to-one geometric correlation does not render loglines useless. As established by our generation metrics (Table 10), using loglines as an auxiliary input affects trajectory realism and the preservation of movie-attribute correlations beyond what is captured by the motion caption alone.

To qualitatively illustrate this effect, Figure 8 visualizes trajectories generated by CINEGEN conditioned on a fixed motion caption but paired with different scene loglines. The top row shows variations under a “truck left” instruction, while the bottom row uses “pedestal up.” Although the base instruction remains fixed, the logline can modulate the trajectory’s scale, pace, and subtle dynamics. These examples illustrate scene-conditioned variation; they do not constitute explicit era-, genre-, or director-style control.

Robustness to conflicting logline cues. The annotation prompt excludes camera-motion language to keep the two text conditions semantically distinct, but inference does not require a sanitized logline. To test robustness, we select 128 validation samples with explicit directional motion captions and inject the opposite direction into the corresponding logline while holding the motion caption, first pose, and initial sampling noise fixed. Similarity to the original motion caption does not decrease (0.574 with the clean logline versus 0.592 with the contradictory logline). Moreover, in 90/128 cases (70.3%), the generated trajectory remains closer to the dedicated motion-caption direction than to the conflicting logline direction. This stress test indicates that the motion caption remains the dominan control signal even when the auxiliary scene description contains an explicit conflict.

![](images/bca0976f9b59f96a616ff26d5b1f6889dc7ff7cd599113244b8481424ca45c2f.jpg)  
Figure 8: Impact of logline conditioning on trajectory generation. We generate camera paths using the same motion caption (Top row: “truck left”, Bottom row: “pedestal up”) but varying loglines. While the fundamental camera movement firmly aligns with the motion caption, the logline modulates the trajectory’s scale, pace, and subtle dynamics to suit the specific narrative context.

## K Limitations and Broader Impacts

Limitations. While CINEGEN establishes a new paradigm for motion-centric camera trajectory generation, it has several limitations that provide promising avenues for future work. First, our exploration of cinematic attributes is constrained by data availability. The subset of CINESCRIPT with explicitly linked real-movie metadata (e.g., era, genre, director) is relatively small (∼3.2K clips) and severely long-tailed, making reliable supervised style control difficult. We therefore use these attributes to evaluate preservation of latent cinematic correlations rather than to train an era-, genre-, or director-controlled generator. Scaling and balancing the metadata-linked subset is an important direction for explicit controllable cinematic styling. Second, like all data-driven motion models, our framework relies on the accuracy of the underlying Structure-from-Motion (SfM) or SLAM pipeline [33] used to extract camera poses from raw video. Our filtering, smoothing, and caption-tag stabilization remove severe failures and isolated jitter before training, but some residual estimation noise can remain. Finally, while our VLM-generated loglines provide valuable scene context, they are inherently synthetic. Potential hallucinations or missed subtle narrative cues from the VLM could introduce noise into the text-conditioning space.

Broader Impacts. Our research holds significant positive potential for virtual production, 3D animation, and independent filmmaking. By allowing creators to synthesize complex, realistic camera movements using natural language and scene descriptions, CINEGEN can democratize pre-visualization and reduce the steep learning curve associated with professional 3D camera rigging. Conversely, as with any generative video technology, improved camera trajectory synthesis could theoretically be dual-used to enhance the realism of synthetic media or deepfakes, making them harder to distinguish from real footage. However, we note that our model specifically outputs abstract physical parameters (camera trajectories) rather than rendering raw visual pixels. To generate misleading video content, our framework would need to be coupled with a high-fidelity rendering engine or video generation model. We advocate for the continued development of robust watermarking and synthetic-media detection protocols to mitigate these downstream risks.
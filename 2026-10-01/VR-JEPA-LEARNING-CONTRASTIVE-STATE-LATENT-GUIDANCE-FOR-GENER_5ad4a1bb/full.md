# VR-JEPA: LEARNING CONTRASTIVE-STATE LATENT GUIDANCE FOR GENERATION-BASED VIDEO REASON-ING

Zehua Ma<sup>1\*</sup>, Kun Xiang<sup>1\*</sup>, Yunshuang Nie<sup>3</sup> Quanlin Chen<sup>1,2</sup>, Haoyuan Li<sup>1</sup>, Xiuwei Chen<sup>1</sup> Jiang Ji<sup>4</sup>, Haijun Wu<sup>4</sup>, Zhenyu Xie<sup>5</sup> Michael Kampffmeyer<sup>6</sup>, Hanhui Li<sup>1†</sup>, Xiaodan Liang<sup>1†</sup>

<sup>1</sup>Shenzhen Campus of Sun Yat-sen University

<sup>2</sup>Shenzhen Loop Area Institute <sup>3</sup>Tsinghua Shenzhen International Graduate School <sup>4</sup>Tencent

<sup>5</sup>Mohamed bin Zayed University of Artificial Intelligence <sup>6</sup>UiT The Arctic University of Norway

mazh58@mail2.sysu.edu.cn

![](images/142480381809de5f53a8f04323c3e329325442e1a64911fbee4833df8b67222f.jpg)  
Figure 1: Generation-based visual reasoning with VR-JEPA. Top: Comparison of different video reasoning paradigms: (1) direct video generation; (2) video generation guided by static DINOv3 features of the initial state; and (3) VR-JEPA, which adapts V-JEPA’s spatiotemporal priors to diverse reasoning skills and guides video generation through predicted latent trajectories. Bottom: Video reasoning results of VR-JEPA and VBVR-Wan2.2, showing that VR-JEPA generates trajectories that satisfy task requirements. Red dashed boxes highlight reasoning errors.

## ABSTRACT

Reasoning through video generation offers a promising path toward visual intelligence by modeling latent visual states and their dynamics. However, current video generation models often lack explicit guidance on how these states should

evolve, leaving generated trajectories prone to physical and structural inconsistencies that undermine reasoning reliability. While the Video Joint-Embedding Predictive Architecture (V-JEPA) provides rich spatiotemporal priors learned through latent prediction, these general priors do not naturally adapt to the logical reasoning capabilities required for complex visual tasks. To bridge this gap, we propose VR-JEPA, a framework that aligns the V-JEPA predictor with task-specific reasoning logic through localized contrastive-state learning and uses its predicted latent trajectories to guide video generation for visual reasoning. Specifically, (i) we pair successful trajectories with generated alternatives under the same input conditions and use discrepancies in their V-JEPA representations to identify informative states and tokens for localized contrastive supervision. (ii) We further equip the V-JEPA predictor with skill-specific experts trained on anchor-task data, allowing the model to adaptively specialize its shared spatiotemporal priors across diverse cognitive domains. Together with skill-specific experts, this contrastive supervision enables VR-JEPA to predict latent trajectories that provide task-specific logical guidance for video generation. Comprehensive experiments on the largescale VBVR-Pro-Bench dataset demonstrate that VR-JEPA achieves an 11.33% relative improvement over the cutting-edge generation-based reasoning baseline, significantly mitigating physical artifacts and enhancing logical consistency.

## 1 INTRODUCTION

Anticipating how the world will change is central to planning and problem solving, motivating world models that support prediction and decision making (Ha & Schmidhuber, 2018; Micheli et al., 2023; Du et al., 2024; Xiang et al., 2025). Video generation provides a setting for examining predicted scene evolution: it renders intermediate state transitions and final outcomes as a sequence. Diffusion models have made substantial progress in generating realistic and temporally coherent videos (Ho et al., 2022; Blattmann et al., 2023). However, their denoising objective does not by itself ensure that generated transitions satisfy a task’s requirements. A video may look convincing while moving the wrong object or depicting an incorrect object relation. This gap motivates guidance that links generated state changes to task requirements (Liu et al., 2026; Cheng et al., 2026).

The Video Joint-Embedding Predictive Architecture (V-JEPA) offers pretrained spatiotemporal representations that can support such guidance. By predicting visual features rather than pixels, it captures scene content and motion while reducing emphasis on appearance details (Bardes et al., 2024). Yet representing how a scene evolves is not the same as predicting how it should evolve to satisfy an instruction. A predictor may track an object’s motion while missing the spatial relation that motion must establish. Furthermore, different tasks also require distinct reasoning skills, such as understanding physical interactions, comparing quantities, and inferring abstract rules. A predictor built on V-JEPA features must therefore focus on task-relevant transitions and adapt to diverse reasoning requirements. Its predicted trajectories should guide video generation toward sequences that satisfy these requirements.

To address these challenges, we introduce VR-JEPA, a framework that predicts task-relevant trajectories in a latent space derived from V-JEPA features and uses them to guide video generation (Figure 1). Given an initial scene and an instruction, a learned predictor autoregressively produces a sequence of compact latent states describing the desired scene evolution. These states condition a video diffusion model, separating the prediction of task-relevant states from visual synthesis.

Specifically, to make latent prediction more sensitive to task-relevant changes, VR-JEPA trains the predictor with localized contrastive-state learning. A solution may depend on only a few object movements or changes in inter-object relations, while many other visual details are incidental. We pair successful reference trajectories with task-matched generated candidates and use discrepancies in their V-JEPA features as a heuristic for weighting contrastive supervision at the state and token levels. This encourages predicted rollouts to follow the reference evolution rather than merely capture generic scene motion. Complementing this focused supervision, VR-JEPA incorporates a skill-routed mixture of experts (SR-MoE) to model heterogeneous state transitions. Lightweight residual experts are first trained on representative anchor tasks; a sparse, state-conditioned router then learns to combine them as the rollout evolves, using the predictor’s training objectives without expert-assignment labels.

![](images/e7ba374e8387fd58f6877fbc0076cbb088bb4d6781f9fab98a1e0f3c36c5c039.jpg)  
Figure 2: Overview of VR-JEPA. VR-JEPA comprises a Latent Reasoner and a Video Generator. The Latent Reasoner predicts a latent rollout from the initial scene and task instruction, with SR-MoE adaptively selecting expert combinations for different reasoning tasks. Its seven experts cover VIS (visual discrimination), NUM (numerical reasoning), IDN (object identity and persistence), SPA (spatial reasoning and planning), DYN (physical dynamics), CMP (compositional reasoning), and ABS (abstract rule inference). During predictor training, localized contrastive-state learning uses dense V-JEPA discrepancies between successful and generated videos to focus contrastive su pervision on tokens corresponding to informative states and regions. The Video Generator conditions Wan2.2 DiT on the rollout to synthesize a reasoning video.

On VBVR-Pro-Bench, VR-JEPA improves overall task success from 50.3% for VBVR-Wan2.2 to 56.0%. Component ablations support the contributions of localized contrastive-state learning and skill routed mixture of experts. Our contributions are threefold:

• We introduce VR-JEPA, a framework that predicts task-relevant latent trajectories from V-JEPA features and uses them to guide video generation, separating state evolution prediction from visual synthesis.

• We develop localized contrastive-state learning (LCL), which uses feature discrepancies between successful reference trajectories and task-matched generated candidates to focus contrastive supervision on informative states and tokens.

• We develop a skill-routed mixture of experts (SR-MoE) for latent prediction. Anchortask-trained residual experts are combined by a sparse, state-conditioned router to adapt to heterogeneous state transitions without expert-assignment labels.

## 2 RELATED WORK

## 2.1 JOINT-EMBEDDING PREDICTIVE ARCHITECTURES (JEPA)

Joint-Embedding Predictive Architecture (JEPA) learns representations by predicting target embeddings from contextual observations. I-JEPA introduces this approach for images, and V-JEPA extends it to spatiotemporal representations (Assran et al., 2023; Bardes et al., 2024). V-JEPA 2 demonstrates latent prediction for video understanding and action-conditioned planning (Assran et al., 2025). Recent work extends JEPA to reasoning and generation. JEPA-Reasoner separates latent reasoning from text generation, while ThinkJEPA incorporates vision-language reasoning features into latent world modeling (Liu & Chen, 2025; Zhang et al., 2026). For video generation, WMReward uses V-JEPA prediction consistency for candidate selection and sampling guidance, while Off-Manifold Refinement refines generation trajectories with prediction-energy gradients (Yuan et al., 2026; Nguyen-Truong et al., 2026). PhysVideoGenerator predicts V-JEPA features from noisy diffusion latents and injects them into the generator (Satish et al., 2026). These generation approaches primarily target physical plausibility. VR-JEPA focuses on visual problem solving, adapting a JEPA predictor through localized contrastive-state learning and skill specialization to produce latent trajectories that condition video generation.

## 2.2 VISUAL REASONING THROUGH GENERATION

Visual generation provides a medium for expressing intermediate states during problem solving. Wiedemer et al. (2025) demonstrate emergent zero-shot perception, manipulation, and reasoning capabilities in video models. Thinking with Video further examines generation-based reasoning through VideoThinkBench, covering both vision-centric and text-centric tasks (Tong et al., 2026). Moving toward systematic training and evaluation, VBVR introduces large-scale reasoning data, verifiable scoring, and scaling studies with VBVR-Wan2.2 (Wang et al., 2026). VBVR-Pro extends this direction to native visual reasoning across generation modalities and evaluates transfer to external benchmarks (Xu et al., 2026). Beyond establishing these capabilities, recent approaches im prove video reasoning through explicit feedback. Wan-R1 studies verifiable reinforcement learning for spatial reasoning and planning, while VLMs are Good Teachers uses a vision-language model to formulate rewards for test-time optimization of a video generator (Liu et al., 2026; Cheng et al., 2026). These approaches motivate supervision that evaluates task correctness beyond visual plausibility. Our work takes a complementary route by learning reasoning guidance within a latent predictor. Rather than applying correctness feedback only to the generator, VR-JEPA uses successful and contrastive trajectories to train intermediate state predictions, which subsequently guide the generation of visual solutions.

## 3 METHOD

## 3.1 FRAMEWORK OVERVIEW

As shown in Figure 2, VR-JEPA comprises a Latent Reasoner and a Video Generator. A frozen V-JEPA encoder and a learnable pooler encode the initial scene $x _ { 0 }$ as a latent state $Z _ { 0 }$ . Given $Z _ { 0 }$ the instruction $c ,$ and up to H recent states, the predictor $P _ { \theta }$ produces a latent rollout that conditions the video generator $G _ { \phi } \colon$

$$
\widehat { Z } _ { t + 1 } = P _ { \theta } \Big ( Z _ { 0 } , \widehat { Z } _ { \operatorname* { m a x } ( 0 , t - H + 1 ) ; t } , c \Big ) , \qquad \widehat { \mathcal { Z } } = ( Z _ { 0 } , \widehat { Z } _ { 1 } , \ldots , \widehat { Z } _ { T } ) , \qquad \widehat { V } = G _ { \phi } ( x _ { 0 } , c , \widehat { \mathcal { Z } } ) .\tag{1}
$$

Here, $\widehat { Z } _ { 0 } = Z _ { 0 } , T$ is the prediction horizon, $t = 0 , \ldots , T - 1$ , and H is the maximum number of recent predicted states supplied to the predictor.

The rollout conditions the generator through two pathways. A semantic memory adapter applies temporal self-attention to rollout states and projects them into keys and values, which the DiT video tokens query through gated cross-attention. A rollout modulation adapter adds rollout-dependent residual offsets to the shift, scale, and residual-gate parameters of the DiT self-attention and MLP branches, following the adaptive modulation used in DiT (Peebles & Xie, 2023).

## 3.2 LOCALIZED CONTRASTIVE-STATE LEARNING

V-JEPA’s general spatiotemporal priors may not emphasize the state transitions and regions critical to visual reasoning. To address this limitation, VR-JEPA introduces localized contrastive-state learning (LCL), which compares successful videos with task-matched generated candidates in dense V-JEPA feature space, identifies informative states and regions, and strengthens contrastive supervision on the corresponding latent tokens.

V-JEPA latent rollout prediction. For each successful training video, frozen V-JEPA 2.1 encoder E (Mur-Labadia et al., 2026) extracts features $F _ { t } ^ { + }$ from local clip $v _ { t } ^ { + }$ . The learnable pooler $Q _ { \psi }$ maps them to M tokens of dimension d:

$$
\begin{array} { r } { F _ { t } ^ { + } = E ( v _ { t } ^ { + } ) , \qquad Z _ { t } ^ { + } = Q _ { \psi } ( F _ { t } ^ { + } ) \in \mathbb R ^ { M \times d } . } \end{array}\tag{2}
$$

We train the pooler and predictor with one- and two-step prediction (feeding back the first prediction) and VISReg regularization (Assran et al., 2025; Wu et al., 2026):

$$
\begin{array} { r l } & { \mathcal { L } _ { j } = \displaystyle \frac { 1 } { | \mathcal { T } _ { j } | M d } \sum _ { t \in \mathcal { T } _ { j } } \left\| \nu ( \widehat { Z } _ { t + j \mid t } ) - \mathrm { s g } \big [ \nu ( Z _ { t + j } ^ { + } ) \big ] \right\| _ { F } ^ { 2 } , \quad j \in \{ 1 , 2 \} , } \\ & { \qquad \mathcal { L } _ { \mathrm { b a s e } } = \mathcal { L } _ { 1 } + \lambda _ { 2 } \mathcal { L } _ { 2 } + \lambda _ { \mathrm { r e g } } \mathcal { R } _ { \mathrm { V I S R e g } } . } \end{array}\tag{3}
$$

Here, $\widehat { Z } _ { t + j | t }$ is the $j \cdot$ -step prediction from ground-truth history at $t ; \tau _ { j }$ contains valid starting times. The operator ν applies token-wise $\ell _ { 2 }$ normalization, and sg stops gradients through the target. The coefficients $\lambda _ { 2 }$ and $\lambda _ { \mathrm { r e g } }$ weight the two-step and regularization losses.

Trajectory differences localization. VR-JEPA treats each training video $V ^ { + }$ as a successful reasoning sample and generates a candidate video $\hat { V } ^ { - }$ using VBVR-Wan2.2 (Wang et al., 2026) under the same input conditions. Since the generated candidate may satisfy the reasoning requirements, it may not serve as a valid negative sample. VR-JEPA uses discrepancies between their dense V-JEPA features as a heuristic to select pairs and locate states and regions that merit stronger supervision. Figure 3 shows the selection and localization.

![](images/bf050a1661751b6814007a83676fa522728652376035feee75a2f29dad06ff25.jpg)

To measure differences between each training video and its candidate, the frozen V-JEPA encoder processes clips at the same temporal index t, yielding $F _ { t } ^ { \pm } ~ = ~ E ( v _ { t } ^ { \pm } )$ . VR-JEPA computes cosine discrepancies at corresponding feature-grid positions n. To reduce the influence of scene-wide appearance differences, it subtracts the q-quantile of discrepancies within each state and retains the positive residuals:

Figure 3: State selection for localized contrastive-state learning. Successful reference trajectories and taskmatched generated candidates are compared in dense V-JEPA feature space. After state-specific quantile subtraction, large residual discrepancies select states for contrastive supervision (green checks), while small discrepancies exclude them (red crosses).

$$
D _ { t , n } = 1 - \frac { \langle F _ { t , n } ^ { + } , F _ { t , n } ^ { - } \rangle } { \| F _ { t , n } ^ { + } \| _ { 2 } \| F _ { t , n } ^ { - } \| _ { 2 } } , \qquad S _ { t , n } = \big [ D _ { t , n } - \mathrm { Q u a n t i l e } _ { q } ( D _ { t , : } ) \big ] _ { + } .\tag{4}
$$

Here, q is the quantile-level hyperparameter, and $[ x ] _ { + } = \operatorname* { m a x } ( x , 0 )$ . Averaging the residuals over the N dense positions yields the state score $s _ { t } \stackrel { \cdot } { = } \stackrel { \cdot } { N } ^ { - 1 } \sum _ { n } S _ { t , n } ^ { \cdot }$ VR-JEPA maps this score to a state-level weight $a _ { t } = \mathrm { c l i p } ( ( s _ { t } - \tau _ { \mathrm { m i n } } ) / ( \tau _ { \mathrm { f u l l } } - \tau _ { \mathrm { m i n } } ) , 0 , \mathrm { i } )$ , which scales the state’s contribution to the contrastive-state loss. The thresholds $\tau _ { \mathrm { m i n } }$ and $\tau _ { \mathrm { f u l l } }$ specify the scores at which this weight is zero and one, respectively.

For each state with $a _ { t } > 0$ , VR-JEPA uses the pooler’s attention maps to transfer localized discrepancies to latent tokens. Let $A _ { t , h , m , n } ^ { + }$ denote attention from token m to dense position n in head h for the successful video. The normalized discrepancy map and token weights are computed as

$$
\widetilde { S } _ { t , n } = \frac { S _ { t , n } } { \sum _ { n ^ { \prime } } S _ { t , n ^ { \prime } } + \epsilon } , \qquad b _ { t , m } = \frac { 1 } { N _ { h } } \sum _ { h } \left[ \sum _ { n } A _ { t , h , m , n } ^ { + } \widetilde { S } _ { t , n } - \frac { 1 } { N } \right] _ { + } ,\tag{5}
$$

where $N _ { h }$ is the number of attention heads, and $\epsilon > 0$ ensures numerical stability. The term $1 / N$ provides a uniform-attention baseline over the $N$ dense positions. A token receives positive weight only if its attention to the localized discrepancies exceeds this baseline in at least one head; states with no positively weighted tokens are excluded.

Contrastive-state supervision. Using the state and token weights, VR-JEPA encourages each one-step prediction to be closer to the successful target than to its paired generated candidate. Let Ω denote the selected state–token positions with $a _ { t } > 0$ and $b _ { t , m } > 0$ . The weighted margin loss and overall training objective are

$$
\mathcal { L } _ { \mathrm { L C L } } = \frac { 1 } { | \Omega | } \sum _ { ( t , m ) \in \Omega } a _ { t } b _ { t , m } \left[ \mu + d ( \widehat { Z } _ { t , m } , Z _ { t , m } ^ { + } ) - d ( \widehat { Z } _ { t , m } , Z _ { t , m } ^ { - } ) \right] _ { + } ,\tag{6}
$$

Here, $\widehat { Z } _ { t , m }$ denotes token m of the one-step prediction $\widehat { Z } _ { t \mid t - 1 }$ , while $Z _ { t , m } ^ { + }$ and $Z _ { t , m } ^ { - }$ are the corresponding pooled tokens from the successful and candidate videos. The function d is cosine distance, $\mu$ is the margin, and $\lambda _ { \mathrm { L C L } }$ controls the contribution of contrastive-state supervision. Gradients are stopped through the weights and both target representations, and $\mathcal { L } _ { \mathrm { { L C L } } } = 0$ when Ω is empty.

## 3.3 SKILL-ROUTED MIXTURE OF EXPERTS

A compact V-JEPA predictor may lack the capacity to model the diverse reasoning patterns required across tasks. To address this limitation, VR-JEPA equips the predictor with a lightweight skill-routed mixture of experts, which adaptively selects different expert combinations for different tasks.

Learning skill-specific experts. VR-JEPA first trains experts on representative anchor tasks to encourage specialization and provide a skill-oriented basis for subsequent composition. Starting from the trained dense predictor, SR-MoE adds lightweight bottleneck adapters to the final two blocks of the six-layer predictor while retaining the shared feed-forward networks:

$$
Y _ { t , m } ^ { ( \ell ) } = F _ { \mathrm { s h a r e d } } ^ { ( \ell ) } \Big ( H _ { t , m } ^ { ( \ell ) } \Big ) + \alpha \sum _ { k = 1 } ^ { K } \pi _ { t , k } ^ { ( \ell ) } A _ { k } ^ { ( \ell ) } \Big ( H _ { t , m } ^ { ( \ell ) } \Big ) .\tag{7}
$$

Here, $H _ { t , m } ^ { ( \ell ) }$ is the input token, $Y _ { t , m } ^ { ( \ell ) }$ is its expert-augmented output, $F _ { \mathrm { s h a r e d } } ^ { ( \ell ) }$ is the shared feedforward network, and $A _ { k } ^ { ( \ell ) }$ is the k-th expert adapter. The expert residual coefficient is fixed to $\alpha = 1$ , so the weighted expert output is added directly to the shared output without an additional learned or scheduled scale; $\pi _ { t , k } ^ { ( \ell ) }$ is the routing weight. The model uses $K = 7$ skill-specific experts. Please refer to Figure 2 for their abbreviations and corresponding skills.

For each anchor task, VR-JEPA directly activates the expert associated with its skill and updates only that expert’s adapters, leaving the shared predictor fixed. This gives each expert an initial skill-specific capability before the router learns to combine them across tasks.

Learning adaptive routing. Different categories of reasoning tasks require different skills or combinations of skills. VR-JEPA therefore learns a router to select and combine the pretrained experts according to each task’s reasoning requirements. The router conditions on the current state, the preceding state, their difference, and the task context:

$$
r _ { t } ^ { ( \ell ) } = \left[ \bar { H } _ { t } ^ { ( \ell ) } ; \bar { H } _ { t - 1 } ^ { ( \ell ) } ; \bar { H } _ { t } ^ { ( \ell ) } - \bar { H } _ { t - 1 } ^ { ( \ell ) } ; g ^ { ( \ell ) } \right] , \qquad \pi _ { t } ^ { ( \ell ) } = \mathrm { E n t m a x } _ { 1 . 5 } \Big ( f _ { \mathrm { r o u t e } } ^ { ( \ell ) } ( r _ { t } ^ { ( \ell ) } ) \Big ) .\tag{8}
$$

Here, $\bar { H } _ { t } ^ { ( \ell ) }$ averages the hidden tokens within a state, $g ^ { ( \ell ) }$ summarizes the conditioning context, and $f _ { \mathrm { r o u t e } } ^ { ( \ell ) }$ maps the concatenated features $r _ { t } ^ { ( \ell ) }$ to expert scores. Entmax $_ { 1 . 5 }$ (Peters et al., 2019) converts these scores into normalized sparse weights, enabling the router to select and combine experts for different reasoning tasks.

The router is trained through the prediction and contrastive-state objectives, without expertassignment annotations for the full training set or additional load-balancing losses. Training first optimizes the router with fixed experts, then jointly adapts the router and experts, and finally updates selected shared components. This progression learns how to combine the initialized experts before refining their capabilities and coordination beyond the anchor tasks. The resulting adaptive composition supports latent rollout prediction for tasks requiring multiple reasoning skills.

## 4 EXPERIMENTS

## 4.1 EXPERIMENTAL SETUP

Benchmark and Evaluation. We evaluate VR-JEPA on VBVR-Pro-Bench (Xu et al., 2026), which comprises 100 visual reasoning tasks spanning abstraction, knowledge, perception, spatial reasoning, and transformation. We evaluate five instances per task, totaling 500 instances. Given an initial image and an instruction, models generate a video depicting the solution. The benchmark includes 50 in-domain (ID) tasks from task families represented in its training set and 50 held-out out-of-domain (OOD) tasks. Task-specific, rule-based evaluators assess generated outputs against the required outcomes and constraints. We report overall, ID/OOD, and category-level scores; absolute gains are expressed in percentage points.

The publicly released baseline outputs for VBVR-Pro-Bench differ in frame count and spatial resolution. We rerun VBVR-Wan2.2 on VBVR-Pro-Bench. Before evaluation, we temporally sample each generated video to match the frame count of its corresponding ground-truth video and resize the frames to 512 × 512 pixels. We apply this preprocessing to outputs from all compared methods and recompute their scores using the same evaluation pipeline, ensuring a consistent comparison.

Baselines. We compare VR-JEPA with six open-source video models: Wan2.2-I2V-A14B and Wan2.2-TI2V-5B (Wan Team, 2025), Wan2.1-I2V-14B-720P (Team Wan et al., 2025), LTX-2.3- I2AV (HaCohen et al., 2026; Lightricks, 2026), CogVideoX1.5-5B-I2V (Yang et al., 2024; Z.ai, 2024), and HunyuanVideo-I2V (Kong et al., 2024; Tencent, 2025). We also include three proprietary video models: Seedance 2.0 (Team Seedance et al., 2026), Kling VIDEO 3.0 (Kling AI, 2026), and Veo 3.1 (Gallegos & Iljic, 2025). Among video reasoning models, our primary baseline is VBVR-Wan2.2 (Wang et al., 2026), which adapts Wan2.2-I2V-A14B to the VBVR dataset without modifying the model architecture.

Implementation Details. VR-JEPA is trained in two stages. First, the Latent Reasoner is trained with the V-JEPA 2.1 encoder (Mur-Labadia et al., 2026) frozen. The learnable pooler and predictor are optimized using the prediction and localized contrastive-state objectives described in Section 3.2, with expert training and routing detailed in Section 3.3. Second, the Latent Reasoner is frozen, and the Wan2.2-14B generator, initialized from VBVR-Wan2.2, is adapted by optimizing only the LoRA parameters (Hu et al., 2022) and rollout-conditioning modules with a flow-matching loss (Lipman et al., 2023). Additional implementation details are provided in Appendix A.

## 4.2 MAIN RESULTS

Quantitative Comparisons. Table 1 reports quantitative comparisons on VBVR-Pro-Bench. VR-JEPA achieves the highest overall score of 56.0% among the evaluated models, outperforming VBVR-Wan2.2 by 5.7 percentage points and Seedance 2.0, the highest-scoring proprietary model overall, by 7.5 points. Compared with VBVR-Wan2.2, it improves the ID average from 55.4% to 65.9% and the OOD average from 45.1% to 46.1%. In particular, ID abstraction and perception improve by 22.5 and 10.9 points, reaching 56.3% and 61.8%, respectively. ID knowledge and spatial reasoning also improve by 9.0 and 3.2 points, while transformation performance remains nearly unchanged at 91.4%. These gains over VBVR-Wan2.2 support the effectiveness of the proposed latent reasoning framework, which combines supervision on informative states and regions with adaptive skill-expert composition. Section 4.3 examines the contributions of localized contrastive-state learning and SR-MoE individually and jointly.

Qualitative Comparisons. Figure 4 compares VR-JEPA with VBVR-Wan2.2, LTX-2.3-I2AV, Kling VIDEO 3.0, and Seedance 2.0 on four visual reasoning examples. In the ball-bouncing task, VR-JEPA produces a trajectory closely matching the ground truth, whereas VBVR-Wan2.2 follows an incorrect initial direction and the other models produce curved or irregular paths. In the shapesorting task, VR-JEPA preserves the objects and groups them by type in ascending size order. By contrast, VBVR-Wan2.2 interleaves the two shape groups, while LTX-2.3-I2AV alters object shapes and colors. In the transformation task, VR-JEPA correctly applies the demonstrated color change followed by size reduction to the bottom-row ellipses. VBVR-Wan2.2 leaves the answer positions unfilled, while Kling VIDEO 3.0 and Seedance 2.0 change the final ellipse into a circle. In the sequence-completion task, VR-JEPA circles the same option as the ground truth, whereas VBVR-Wan2.2 circles the red distractor and Seedance 2.0 modifies the sequence instead of marking an answer option. Together, these examples highlight VR-JEPA’s advantage over the compared models in following task rules while preserving relevant object attributes, yielding reasoning outputs more consistent with the intended solutions.

Table 1: Main results on VBVR-Pro-Bench (Xu et al., 2026). We report overall, in-domain (ID), and out-of-domain (OOD) scores, followed by results for abstraction (Abst.), knowledge (Know.), perception (Perc.), spatial reasoning (Spat.), and transformation (Trans.). Avg. denotes the mean of task scores weighted by the number of samples per task. All scores are reported as percentages. Higher is better. Bold and underlined values indicate the best and second-best results.
<table><tr><td></td><td></td><td colspan="6">In-Domain (ID) (%)</td><td colspan="6">Out-of-Domain (OOD) (%)</td></tr><tr><td>Models</td><td>| Overall (%)|</td><td>Avg.</td><td></td><td></td><td>Abst. Know. Perc. Spat. Trans.</td><td></td><td></td><td>Avg.</td><td></td><td>Abst. Know. Perc. Spat. Trans.</td><td></td><td></td><td></td></tr><tr><td colspan="10">Open-source Video Models</td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Wan2.2-I2V-A14B</td><td>18.2</td><td>15.7</td><td>12.3</td><td>15.2</td><td>15.0</td><td>17.0</td><td>23.3</td><td>20.7</td><td>29.2</td><td>17.9</td><td>13.8</td><td>19.5</td><td>31.8</td></tr><tr><td>LTX-2.3-I2AV</td><td>11.2</td><td>10.6</td><td>9.3</td><td>12.3</td><td>9.7</td><td>13.6</td><td>5.1</td><td>11.9</td><td>21.7</td><td>19.8</td><td>7.0</td><td>9.1</td><td>5.9</td></tr><tr><td>Wan2.1-I2V-14B-720P</td><td>10.0</td><td>10.5</td><td>8.7</td><td>14.6</td><td>12.4</td><td>9.9</td><td>4.0</td><td>9.5</td><td>13.9</td><td>8.7</td><td>7.6</td><td>12.3</td><td>5.2</td></tr><tr><td>Wan2.2-TI2V-5B</td><td>9.4</td><td>6.6</td><td>5.0</td><td>7.4</td><td>7.7</td><td>9.2</td><td>2.5</td><td>12.2</td><td>22.2</td><td>9.2</td><td>9.1</td><td>6.3</td><td>11.6</td></tr><tr><td>CogVideoX1.5-5B-I2V</td><td>8.6</td><td>9.7</td><td>9.2</td><td>14.2</td><td>7.8</td><td>7.9</td><td>5.5</td><td>7.5</td><td>16.5</td><td>5.6</td><td>5.0</td><td>3.9</td><td>2.8</td></tr><tr><td>HunyuanVideo-I2V</td><td>5.6</td><td>5.5</td><td>4.0</td><td>7.2</td><td>1.0</td><td>9.1</td><td>3.8</td><td>5.7</td><td>11.6</td><td>1.5</td><td>2.7</td><td>7.4</td><td>6.4</td></tr><tr><td colspan="10">Proprietary Video Models</td><td colspan="7"></td></tr><tr><td>Seedance 2.0</td><td>48.5</td><td>43.9</td><td>41.9</td><td>40.6</td><td>47.5</td><td>49.7</td><td>41.8</td><td>53.0</td><td>43.7</td><td>67.9</td><td>52.4</td><td>53.3</td><td></td><td>61.3</td></tr><tr><td>Kling VIDEO 3.0</td><td>38.7</td><td>36.1</td><td>28.7</td><td>34.6</td><td>48.2</td><td>40.3</td><td>34.6</td><td>41.3</td><td>34.6</td><td>75.5</td><td></td><td>40.1 24.1</td><td></td><td>47.7</td></tr><tr><td>Veo 3.1</td><td>14.9</td><td>16.3</td><td>15.1</td><td>20.5</td><td>19.6</td><td>13.6</td><td>9.7</td><td>13.4</td><td>21.2</td><td></td><td>14.8</td><td>10.1</td><td>12.9</td><td>9.1</td></tr><tr><td colspan="10">Video Reasoning Models</td><td colspan="7"></td></tr><tr><td>VBVR-Wan2.2</td><td>50.3</td><td>55.4</td><td>33.8</td><td>54.8</td><td>50.9</td><td>65.9</td><td>91.2</td><td>45.1</td><td>35.8</td><td></td><td>42.7</td><td>34.0</td><td>72.2</td><td>77.7</td></tr><tr><td>VR-JEPA</td><td>56.0</td><td>65.9</td><td>56.3</td><td>63.8</td><td>61.8</td><td>69.1</td><td>91.4</td><td>46.1</td><td>41.5</td><td>42.1</td><td></td><td>35.2</td><td>65.2</td><td>77.5</td></tr></table>

![](images/ba26b3a60551bc848b4fda2cc90fa363d4ed68027b177b61e9751dedb74080f1.jpg)  
Figure 4: Qualitative comparison on VBVR-Pro-Bench. Given the initial state and instruction, we compare the reasoning results of VR-JEPA with those of a video reasoning model (VBVR-Wan2.2), an open-source video model (LTX-2.3-I2AV), and two proprietary video models (Kling VIDEO 3.0 and Seedance 2.0). Green and red dashed boxes highlight correct results and reasoning errors, respectively.

![](images/578f1b0cd927de5fc87ef7a14ac0825ec4223ae3b16ddb458fb7df6f0782888a.jpg)  
Figure 5: Qualitative ablation of localized contrastive-state learning (LCL) and skill-routed mixture of experts (SR-MoE). Top and middle: adding LCL and SR-MoE to the dense baseline, respectively. Bottom: adding SR-MoE to the LCL-equipped predictor to obtain the full model. Green and red dashed boxes highlight correct results and reasoning errors, respectively.

Table 2: Ablation of localized contrastive-state learning (LCL) and skill-routed mixture of experts (SR-MoE). Scores are reported as percentages; higher is better.
<table><tr><td rowspan="2">Variant</td><td colspan="2">Components</td><td rowspan="2">Overall (%)</td><td rowspan="2"></td><td colspan="5">In-Domain (ID) (%)</td><td colspan="6">Out-of-Domain (OOD) (%)</td></tr><tr><td>LCL SR-MoE</td><td></td><td>Avg. Abst.</td><td>Know.</td><td>Perc.</td><td>Spat.</td><td></td><td>Trans. Avg.</td><td>Abst.</td><td>Know.</td><td>Perc.</td><td>Spat.</td><td>Trans.</td></tr><tr><td>Dense baseline</td><td>X</td><td>X</td><td>53.9</td><td>63.4</td><td>53.8</td><td>58.9</td><td>58.9</td><td>68.8</td><td>91.4</td><td>44.4</td><td>43.3</td><td>44.5</td><td>31.9</td><td>59.3</td><td>75.3</td></tr><tr><td>+ LCL</td><td>√</td><td>X</td><td>55.1</td><td>64.9</td><td>52.6</td><td>61.2</td><td>61.3</td><td>71.6</td><td>93.0</td><td>45.3</td><td>44.1</td><td>38.5</td><td>33.1</td><td>65.4</td><td>75.8</td></tr><tr><td>+ SR-MoE</td><td>X</td><td>√</td><td>55.5</td><td>65.2</td><td>54.0</td><td>61.4</td><td>60.7</td><td>69.8</td><td>95.6</td><td>45.9</td><td>42.8</td><td>42.3</td><td>34.3</td><td>64.2</td><td>77.3</td></tr><tr><td>Full model</td><td>√</td><td>√</td><td>56.0</td><td>65.9</td><td>56.3</td><td>63.8</td><td>61.8</td><td>69.1</td><td>91.4</td><td>46.1</td><td>41.5</td><td>42.1</td><td>35.2</td><td>65.2</td><td>77.5</td></tr></table>

Table 3: Ablation of visual representations. Scores are reported as percentages; higher is better.
<table><tr><td>Visual representation</td><td>Representation form</td><td>ID (%) ↑</td><td>OOD (%) ↑</td></tr><tr><td>DINOv3</td><td>Initial-state features</td><td>62.8</td><td>45.3</td></tr><tr><td>DINOv3</td><td>Predicted latent rollout</td><td>65.3</td><td>43.8</td></tr><tr><td>V-JEPA 2.1 (VR-JEPA)</td><td>Predicted latent rollout</td><td>65.9</td><td>46.1</td></tr></table>

## 4.3 ABLATION STUDIES

Effectiveness of LCL and SR-MoE. The dense baseline guides Wan2.2 video generation with latent rollouts predicted by a trained V-JEPA predictor. The three variants add localized contrastivestate learning (LCL), SR-MoE, or both to this baseline. Table 2 shows that LCL and SR-MoE individually raise the overall score from 53.9% to 55.1% and 55.5%, respectively. Combining both achieves the best overall, ID, and OOD averages among the ablated variants: 56.0%, 65.9%, and 46.1%. Figure 5 illustrates these benefits: LCL corrects the mirrored checkerboard completion, while SR-MoE selects the instructed pentagon instead of a distractor. Combining both also enables the model to identify the second-largest circle, which the LCL-only variant fails to select. Together, these results support the complementary benefits of supervision focused on informative states and regions and skill-specific expert routing in improving VR-JEPA’s video reasoning.

Visual Representation Space. Table 3 compares DINOv3 (Simeoni et al., 2025) and V-JEPA´ 2.1 (Mur-Labadia et al., 2026) as representation spaces for latent reasoning under the same framework, with initial-state DINOv3 features included as a reference. The selected DINOv3 and V-JEPA 2.1 encoders have comparable parameter counts. Using V-JEPA 2.1 for latent rollout prediction achieves video reasoning scores of 65.9% on ID tasks and 46.1% on OOD tasks, outperforming the DINOv3 rollout variant by 0.6 and 2.3 percentage points, respectively. These results favor V-JEPA 2.1 for video reasoning, consistent with its ability to capture spatiotemporal dynamics beyond the static visual semantics primarily encoded by DINOv3. Corresponding qualitative comparisons are provided in Figure 9 in Appendix C.3.

## 5 CONCLUSION

We presented VR-JEPA, a framework that predicts latent rollouts in V-JEPA representation space to guide video generation for visual reasoning. VR-JEPA uses localized contrastive-state learning to focus contrastive supervision on informative states and regions identified from successful reference videos and task-matched generated candidates. It further equips the predictor with a skill-routed mixture of experts that adaptively combines skill-specific experts for different reasoning tasks. On VBVR-Pro-Bench, VR-JEPA improves the overall score from 50.3% to 56.0% over VBVR-Wan2.2, an absolute gain of 5.7 percentage points. Component ablations support the complementary contributions of localized supervision and skill-expert composition. Together, these results highlight the potential of reasoning-guided latent prediction for reasoning through video generation.

## 6 AI USE STATEMENT

AI tools were used to assist with literature searches. We reviewed and selected the relevant papers and independently organized and wrote the related work section. We take full responsibility for the content and references of this paper.

## REFERENCES

Mahmoud Assran, Quentin Duval, Ishan Misra, Piotr Bojanowski, Pascal Vincent, Michael Rabbat, Yann LeCun, and Nicolas Ballas. Self-supervised learning from images with a joint-embedding predictive architecture. arXiv preprint arXiv:2301.08243, 2023. URL https://arxiv.org/ abs/2301.08243.

Mido Assran, Adrien Bardes, David Fan, Quentin Garrido, et al. V-JEPA 2: Self-supervised video models enable understanding, prediction and planning. arXiv preprint arXiv:2506.09985, 2025. URL https://arxiv.org/abs/2506.09985.

Adrien Bardes, Quentin Garrido, Jean Ponce, Xinlei Chen, Michael Rabbat, Yann LeCun, Mido Assran, and Nicolas Ballas. Revisiting feature prediction for learning visual representations from video. Transactions on Machine Learning Research, 2024. ISSN 2835-8856. URL https: //openreview.net/forum?id=QaCCuDfBk2.

Andreas Blattmann, Robin Rombach, Huan Ling, Tim Dockhorn, Seung Wook Kim, Sanja Fidler, and Karsten Kreis. Align your latents: High-resolution video synthesis with latent diffusion models. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 22563–22575, 2023. doi: 10.1109/CVPR52729.2023.02161.

Junhao Cheng, Liang Hou, Tianxiong Zhong, Xin Tao, Pengfei Wan, Kun Gai, and Jing Liao. VLMs are good teachers for video reasoning via adaptive test-time optimization. arXiv preprint arXiv:2606.02564, 2026. URL https://arxiv.org/abs/2606.02564.

Yilun Du, Sherry Yang, Pete Florence, Fei Xia, Ayzaan Wahid, Brian Ichter, Pierre Sermanet, Tianhe Yu, Pieter Abbeel, Joshua B. Tenenbaum, Leslie Pack Kaelbling, Andy Zeng, and Jonathan Tompson. Video language planning. In The Twelfth International Conference on Learning Representations, 2024. URL https://openreview.net/forum?id=9pKtcJcMP3.

Jess Gallegos and Thomas Iljic. Introducing veo 3.1 and advanced capabilities in flow. Google blog post, 2025. URL https://blog.google/innovation-and-ai/products/ veo-updates-flow/.

David Ha and Jurgen Schmidhuber. World models.¨ arXiv preprint arXiv:1803.10122, 2018. URL https://arxiv.org/abs/1803.10122.

Yoav HaCohen, Benny Brazowski, Nisan Chiprut, Yaki Bitterman, Andrew Kvochko, Avishai Berkowitz, Daniel Shalem, Daphna Lifschitz, Dudu Moshe, Eitan Porat, Eitan Richardson, Guy Shiran, Itay Chachy, Jonathan Chetboun, Michael Finkelson, Michael Kupchick, Nir Zabari, Nitzan Guetta, Noa Kotler, Ofir Bibi, Ori Gordon, Poriya Panet, Roi Benita, Shahar Armon, Victor Kulikov, Yaron Inger, Yonatan Shiftan, Zeev Melumian, and Zeev Farbman. LTX-2: Efficient joint audio-visual foundation model. arXiv preprint arXiv:2601.03233, 2026. URL https://arxiv.org/abs/2601.03233.

Jonathan Ho, Tim Salimans, Alexey Gritsenko, William Chan, Mohammad Norouzi, and David J. Fleet. Video diffusion models. arXiv preprint arXiv:2204.03458, 2022. URL https: //arxiv.org/abs/2204.03458.

Edward J. Hu, Yelong Shen, Phillip Wallis, Zeyuan Allen-Zhu, Yuanzhi Li, Shean Wang, Lu Wang, and Weizhu Chen. LoRA: Low-rank adaptation of large language models. In International Conference on Learning Representations, 2022. URL https://openreview.net/forum? id=nZeVKeeFYf9.

Kling AI. Kling video 3.0 model user guide. Official model guide, 2026. URL https://kling. ai/quickstart/klingai-video-3-model-user-guide.

Weijie Kong, Qi Tian, Zijian Zhang, Rox Min, et al. HunyuanVideo: A systematic framework for large video generative models. arXiv preprint arXiv:2412.03603, 2024. URL https:// arxiv.org/abs/2412.03603.

Lightricks. Ltx-2.3. Official model card, 2026. URL https://huggingface.co/ Lightricks/LTX-2.3.

Yaron Lipman, Ricky T. Q. Chen, Heli Ben-Hamu, Maximilian Nickel, and Matt Le. Flow matching for generative modeling. In International Conference on Learning Representations, 2023. URL https://openreview.net/forum?id=PqvMRDCJT9t.

Bingyang Kelvin Liu and Ziyu Patrick Chen. JEPA-Reasoner: Decoupling latent reasoning from token generation. arXiv preprint arXiv:2512.19171, 2025. URL https://arxiv.org/abs/ 2512.19171.

Ming Liu, Yunbei Zhang, Shilong Liu, Liwen Wang, and Wensheng Zhang. Wan-R1: Verifiablereinforcement learning for video reasoning. arXiv preprint arXiv:2603.27866, 2026. URL https://arxiv.org/abs/2603.27866.

Vincent Micheli, Eloi Alonso, and Franc¸ois Fleuret. Transformers are sample-efficient world models. In International Conference on Learning Representations, 2023. URL https: //openreview.net/forum?id=vhFu1Acb0xb.

Lorenzo Mur-Labadia, Matthew Muckley, Amir Bar, Mido Assran, Koustuv Sinha, Mike Rabbat, Yann LeCun, Nicolas Ballas, and Adrien Bardes. V-JEPA 2.1: Unlocking dense features in video self-supervised learning. arXiv preprint arXiv:2603.14482, 2026. URL https://arxiv. org/abs/2603.14482.

Hai Nguyen-Truong, Tuan-Anh Vu, and Dang Huynh. Off-manifold refinement: Guiding video generators with a frozen world model. arXiv preprint arXiv:2608.29904, 2026. URL https: //arxiv.org/abs/2608.29904.

William Peebles and Saining Xie. Scalable diffusion models with transformers. In Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV), pp. 4195–4205, 2023. URL https://openaccess.thecvf.com/content/ICCV2023/html/Peebles\_ Scalable\_Diffusion\_Models\_with\_Transformers\_ICCV\_2023\_paper.html.

Ben Peters, Vlad Niculae, and Andre F. T. Martins. Sparse sequence-to-sequence models. In´ Proceedings of the 57th Annual Meeting of the Association for Computational Linguistics, pp. 1504–1519, Florence, Italy, 2019. Association for Computational Linguistics. doi: 10.18653/v1/ P19-1146. URL https://aclanthology.org/P19-1146/.

Siddarth Nilol Kundur Satish, Devesh Jaiswal, Hongyu Chen, and Abhishek Bakshi. PhysVideo-Generator: Towards physically aware video generation via latent physics guidance. arXiv preprint arXiv:2601.03665, 2026. URL https://arxiv.org/abs/2601.03665.

Oriane Simeoni, Huy V. Vo, Maximilian Seitzer, Federico Baldassarre, Maxime Oquab, Cijo Jose,´ Vasil Khalidov, Marc Szafraniec, Seungeun Yi, Michael Ramamonjisoa, Francisco Massa, Daniel¨ Haziza, Luca Wehrstedt, Jianyuan Wang, Timothee Darcet, Th´ eo Moutakanni, Leonel Sentana,´ Claire Roberts, Andrea Vedaldi, Jamie Tolan, John Brandt, Camille Couprie, Julien Mairal, Herve´ Jegou, Patrick Labatut, and Piotr Bojanowski. DINOv3. ´ arXiv preprint arXiv:2508.10104, 2025. URL https://arxiv.org/abs/2508.10104.

Team Seedance et al. Seedance 2.0: Advancing video generation for world complexity. arXiv preprint arXiv:2604.14148, 2026. URL https://arxiv.org/abs/2604.14148.

Team Wan et al. Wan: Open and advanced large-scale video generative models. arXiv preprint arXiv:2503.20314, 2025. URL https://arxiv.org/abs/2503.20314.

Tencent. Hunyuanvideo-i2v. Official model card, 2025. URL https://huggingface.co/ tencent/HunyuanVideo-I2V.

Jingqi Tong, Yurong Mou, Hangcheng Li, Mingzhe Li, Yongzhuo Yang, Ming Zhang, Qiguang Chen, Tianyi Liang, Xiaomeng Hu, Yining Zheng, Xinchi Chen, Jun Zhao, Xuanjing Huang, and Xipeng Qiu. Thinking with video: Video generation as a promising multimodal reasoning paradigm. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pp. 41121–41129, 2026. URL https://openaccess.thecvf.com/ content/CVPR2026/html/Tong\_Thinking\_with\_Video\_Video\_Generation\_ as\_a\_Promising\_Multimodal\_Reasoning\_CVPR\_2026\_paper.html.

Wan Team. Wan2.2. Official model repository, 2025. URL https://github.com/ Wan-Video/Wan2.2.

Maijunxian Wang, Ruisi Wang, Juyi Lin, Ran Ji, et al. A very big video reasoning suite. arXiv preprint arXiv:2602.20159, 2026. URL https://arxiv.org/abs/2602.20159.

Thaddaus Wiedemer, Yuxuan Li, Paul Vicol, Shixiang Shane Gu, Nick Matarese, Kevin Swersky,¨ Been Kim, Priyank Jaini, and Robert Geirhos. Video models are zero-shot learners and reasoners. arXiv preprint arXiv:2509.20328, 2025. URL https://arxiv.org/abs/2509.20328.

Haiyu Wu, Randall Balestriero, and Morgan Levine. VISReg: Variance-invariance-sketching regularization for JEPA training. arXiv preprint arXiv:2606.02572, 2026. URL https://arxiv. org/abs/2606.02572.

Kun Xiang, Terry Jingchen Zhang, Yinya Huang, Jixi He, Zirong Liu, Yueling Tang, Ruizhe Zhou, Lijing Luo, Youpeng Wen, Xiuwei Chen, Bingqian Lin, Jianhua Han, Hang Xu, Hanhui Li, Bin Dong, and Xiaodan Liang. Aligning perception, reasoning, modeling and interaction: A survey on physical AI. arXiv preprint arXiv:2510.04978, 2025. URL https://arxiv.org/abs/ 2510.04978.

Junxiang Xu, Ruisi Wang, Fanyi Pu, Maijunxian Wang, et al. VBVR-Pro: A scalable and verifiable suite for native visual reasoning. arXiv preprint arXiv:2608.26105, 2026. URL https:// arxiv.org/abs/2608.26105.

Zhuoyi Yang, Jiayan Teng, Wendi Zheng, Ming Ding, Shiyu Huang, Jiazheng Xu, Yuanming Yang, Wenyi Hong, Xiaohan Zhang, Guanyu Feng, Da Yin, Xiaotao Gu, Yuxuan Zhang, Weihan Wang, Yean Cheng, Ting Liu, Bin Xu, Yuxiao Dong, and Jie Tang. Cogvideox: Textto-video diffusion models with an expert transformer. ArXiv, abs/2408.06072, 2024. URL https://api.semanticscholar.org/CorpusID:271855655.

Jianhao Yuan, Xiaofeng Zhang, Felix Friedrich, Nicolas Beltran-Velez, Melissa Hall, Reyhane Askari-Hemmat, Xiaochuang Han, Nicolas Ballas, Michal Drozdzal, and Adriana Romero-Soriano. Inference-time physics alignment of video generative models with latent world models. arXiv preprint arXiv:2601.10553, 2026. URL https://arxiv.org/abs/2601.10553.

Z.ai. Cogvideox1.5-5b-i2v. Official model card, 2024. URL https://huggingface.co/ zai-org/CogVideoX1.5-5B-I2V.

Haichao Zhang, Yijiang Li, Shwai He, Tushar Nagarajan, Mingfei Chen, Jianglin Lu, Ang Li, and Yun Fu. ThinkJEPA: Empowering latent world models with large vision-language reasoning model. arXiv preprint arXiv:2603.22281, 2026. URL https://arxiv.org/abs/2603. 22281.

## APPENDIX CONTENTS

## • A Implementation Details

– A.1 Training Data

– A.2 Architecture and Training Procedure

– A.3 Eight-GPU Reproduction Configuration

– A.4 Parameter Overhead

## • B Additional Ablation Studies

– B.1 Training Strategies

– B.2 Expert Routing Strategies

– B.3 Interventions on the Predicted Rollout

## • C Further Visualization

– C.1 Comparisons with Open-source Models

– C.2 Comparisons with Proprietary Video Models

– C.3 Visual Representation Space Ablation

– C.4 Comparison of latent trajectory guidance

– C.5 Task-Dependent Expert Routing

– C.6 Expert Removal

## A IMPLEMENTATION DETAILS

## A.1 TRAINING DATA

Our training corpus comprises 100 tasks with 1,000 samples per task. Each sample contains an initial state, a task instruction, and a successful reference trajectory. These 100,000 reference samples support latent reasoner training in Stage 1 and video generator adaptation in Stage 2. For localized contrastive-state learning, we additionally generate one negative candidate per sample using VBVR-Wan2.2 (Wang et al., 2026) under the same initial state and instruction, yielding 100,000 positive– negative trajectory pairs. Since generated candidates may also satisfy the task, our localization mechanism identifies informative differences within these pairs for supervision. Of the 100 training tasks, 50 are included in the ID evaluation using held-out instances. The 50 OOD evaluation tasks are absent from the training corpus.

For expert pretraining, we first define the reasoning skill associated with each of the seven experts and manually select seven representative anchor tasks for that skill from the same corpus. The task subsets are disjoint across experts, yielding 49 anchor tasks and 49,000 reference samples in total. Each expert is pretrained on its corresponding subset to establish an initial skill specialization. Subsequent router training and joint refinement cover all 100 tasks without additional task-to-expert annotations.

## A.2 ARCHITECTURE AND TRAINING PROCEDURE

Model initialization. We initialize the visual encoder from the V-JEPA 2.1 ViT-L (Mur-Labadia et al., 2026) checkpoint vjepa2 1 vitl dist vitG 384.pt, loading its ema encoder weights. The encoder remains frozen throughout both training stages. The state-token pooler and six-layer task-relevant latent predictor are trained for our framework rather than initialized from the pretrained V-JEPA predictor. For Stage 2, we initialize the video generator from VBVR-Wan2.2 and adapt its high-noise branch, while retaining the low-noise branch.

Expert architecture and routing. Seven bottleneck-adapter experts are inserted into the final two predictor blocks. Their designated skills are visual discrimination (VIS), numerical reasoning (NUM), object identity and persistence (IDN), spatial reasoning and planning (SPA), physical dynamics (DYN), compositional reasoning (CMP), and abstract rule inference (ABS). Routing uses Entmax (Peters et al., 2019) and the objectives in Section 3.3.

Stage 1: Optimization settings. We use a history window of $H = 4$ states and represent each state with $M = 8$ tokens of dimension $d = 2 5 6$ . The pooler has $N _ { h } = 8$ attention heads. Each rollout contains ten states including the initial state, corresponding to a prediction horizon of $T = 9$ in Equation 1. Early stopping is disabled. The two-step prediction and VISReg weights in Equation 3 are $\lambda _ { 2 } = 2 . 0$ and $\lambda _ { \mathrm { r e g } } = \bar { 1 0 } ^ { - 4 }$ , respectively. The expert residual coefficient in Equation 7 is fixed to $\alpha = 1$ ; the implementation adds the routed expert output directly to the shared feed-forward output, without learning or scheduling a separate residual scale.

Latent reasoner training consists of base training, expert pretraining, router initialization, and two successive joint refinement phases. All phases use AdamW with $\beta \overset {  } { = } ( 0 . 9 , 0 . 9 5 ) , \epsilon = 1 0 ^ { - 8 }$ , and an effective global batch size of 512. Learning rates follow a linear warmup followed by cosine decay. The V-JEPA encoder remains frozen in every phase.

LCL hyperparameters. For localized contrastive-state learning in Section 3.2, we use the quantile level $q = 0 . 7 5$ in Equation 4 and state-weight thresholds $\tau _ { \operatorname* { m i n } } = 0 . 1$ and $\tau _ { \mathrm { f u l l } } = 0 . 2$ . The numerical stability term for discrepancy normalization in Equation 5 is $\epsilon = 1 0 ^ { - 6 }$ . The margin and loss weight in Equation 6 are $\mu = 0 . 0 5$ and $\lambda _ { \mathrm { L C L } } = 0 . 0 5$ , respectively.

Base latent reasoner training. We train the predictor with the one- and two-step objectives, VIS-Reg regularization (Wu et al., 2026), and localized contrastive-state learning (LCL) defined in Section 3.2. We first train the shared latent reasoner for 20 epochs (3,720 optimizer updates). The learning rates are $1 0 ^ { - 5 }$ for the predictor and other trainable modules, $3 \times \mathrm { i 0 ^ { - 5 } }$ for the state-token pooler, and $3 \times 1 0 ^ { - 6 }$ for the text-conditioning projection. We use weight decay of $1 0 ^ { - 3 }$ and 200 warmup updates.

Anchor-task expert pretraining. Starting from the trained shared reasoner, we specialize the seven experts on their respective anchor-task subsets. For each subset, only the corresponding expert adapters are activated and optimized; the shared predictor, pooler, and other modules remain frozen. Expert pretraining runs for 20 epochs with a learning rate of $1 0 ^ { - 5 }$ , weight decay of $1 0 ^ { - 3 }$ , and 200 warmup updates. This phase establishes skill-specific specializations before learning how to route heterogeneous tasks.

Prediction-driven router initialization. We then freeze the expert adapters and shared modules and optimize only the router on the multi-task training split. Routing is learned through the latent prediction objective, without task-to-expert labels or an auxiliary load-balancing loss. We use a learning rate of $1 0 ^ { - 4 }$ , zero weight decay, and 200 warmup updates. The cosine schedule is configured for 20 epochs; the checkpoint used to initialize joint refinement is selected after four epochs (744 optimizer updates).

Joint refinement. We first jointly optimize the router and expert adapters for two epochs (372 optimizer updates), using learning rates of $2 \times 1 0 ^ { - 5 }$ and $1 0 ^ { - 5 }$ , respectively, while keeping the shared modules frozen.

We subsequently unfreeze the predictor’s attention layers, shared feed-forward layers, and statetoken pooler for three additional epochs (558 optimizer updates). The learning rates are $1 0 ^ { - 5 }$ for the router, $5 \times 1 0 ^ { - 6 }$ for expert adapters, $5 \times 1 0 ^ { - 7 }$ for both attention and shared feed-forward layers, and $2 \times 1 0 ^ { - 7 }$ for the pooler. Both refinement phases use zero weight decay and 25 warmup updates. This progressive refinement first coordinates expert selection and specialization, then adapts the shared latent representation to support them.

Stage 2: Video generator adaptation. We freeze the trained latent reasoner and optimize the video generator using the flow-matching objective (Lipman et al., 2023) on reference videos, conditioned on the initial state, instruction, and predicted latent trajectory. Only the LoRA parameters (Hu et al., 2022) and trajectory-conditioning modules in the adapted Wan2.2 generator branch (Wan Team, 2025) are optimized. The trajectory-conditioning interface (TCI) comprises the semantic memory adapter and rollout modulation adapter described in Section 3.1. The former supplies rollout features through gated cross-attention; the latter produces offsets for the DiT modulation parameters.

We use AdamW with $\beta = ( 0 . 9 , 0 . 9 9 9 ) , \epsilon = 1 0 ^ { - 8 }$ , and weight decay of 0.01. The learning rates are $1 0 ^ { - 5 }$ for LoRA and $1 0 ^ { - 4 }$ for the trajectory-conditioning modules, with 50 warmup updates followed by constant learning rates. Training runs for 2,000 optimizer updates with an effective global batch size of 64. We use rank-32 LoRA on the attention projections and feed-forward linear layers, and clip the gradient norm to 1.0.

## A.3 EIGHT-GPU REPRODUCTION CONFIGURATION

For reproduction on eight H100 GPUs, the Stage 1 effective batch size of 512 can be obtained using a per-GPU batch size of 32 with two gradient accumulation steps for base training, or a per-GPU batch size of eight with eight accumulation steps for expert pretraining. Router initialization and both joint refinement phases use a per-GPU batch size of one with 64 accumulation steps. Stage 2 uses a per-GPU batch size of one with eight accumulation steps, yielding an effective batch size of 64.

## A.4 PARAMETER OVERHEAD

VR-JEPA is initialized from VBVR-Wan2.2 and inherits its 306.708M LoRA parameters across the two generator branches (153.354M per branch). The underlying Wan2.2 DiT contains 28.578B parameters, with approximately 14.289B per branch. Our latent predictor contains 38.093M parameters, including the SR-MoE experts and router. The trajectory-conditioning interface adds 17.181M parameters: 11.807M for the semantic memory adapter and 5.374M for the rollout modulation adapter. Relative to VBVR-Wan2.2, the predictor and conditioning modules therefore add 55.274M parameters, equivalent to approximately 0.19% of the underlying 28.578B-parameter DiT. Relative to the original Wan2.2 DiT, the inherited LoRA parameters and these new modules together account for 361.982M additional parameters, approximately 1.27% of the DiT parameter count.

As shown in Table 1, VR-JEPA improves the overall score from 50.3% to 56.0% over VBVR-Wan2.2 with only 55.274M additional predictor and conditioning parameters beyond that baseline; the inherited LoRA parameters are already present in VBVR-Wan2.2. This small overhead supports a design that emphasizes informative contrastive supervision through LCL and adaptive skill-expert composition through SR-MoE, rather than substantial model expansion.

## B ADDITIONAL ABLATION STUDIES

## B.1 TRAINING STRATEGIES

Table 4: Ablation of the training strategy. We compare tuning the trajectory-conditioning interface (TCI), LoRA parameters, or both, with VBVR-Wan2.2 as the baseline. ID and OOD tasksuccess scores are reported on VBVR-Pro-Bench as percentages; higher is better.
<table><tr><td rowspan="2">Training strategy</td><td colspan="2">Trainable modules</td><td colspan="2">Task success (%)</td></tr><tr><td>TCI</td><td>LoRA</td><td>ID↑</td><td>OOD↑</td></tr><tr><td>VBVR-Wan2.2</td><td>X</td><td>X</td><td>55.4</td><td>45.1</td></tr><tr><td>LoRA-only tuning</td><td>X</td><td>√</td><td>57.4</td><td>45.5</td></tr><tr><td>TCI-only tuning</td><td>√</td><td>X</td><td>63.1</td><td>43.7</td></tr><tr><td>Joint tuning</td><td>√</td><td>√</td><td>65.9</td><td>46.1</td></tr></table>

Table 4 compares LoRA-only, TCI-only, and joint tuning. Joint tuning achieves 65.9%/46.1%, exceeding TCI-only tuning by 2.8 percentage points on ID tasks and 2.4 points on OOD tasks. It also exceeds LoRA-only tuning by 8.5 and 0.6 points, respectively. Training the conditioning interface alone is therefore sufficient to improve ID performance in this comparison, whereas the strongest scores on both splits are obtained when the interface and renderer LoRA parameters are adapted together.

Table 5: Ablation of the SR-MoE routing strategy. We compare learned routing with uniform routing, random routing, and a shared-FFN-only variant to assess the contribution of adaptive expert composition. Shared FFN only disables the expert branches of the trained full model at inference, retaining the shared FFNs. ID and OOD task-success scores are reported on VBVR-Pro-Bench as percentages; higher is better.
<table><tr><td rowspan="2">Routing strategy</td><td colspan="2">Task success(%)</td></tr><tr><td>ID↑</td><td>OOD↑</td></tr><tr><td>Shared FFN only</td><td>64.2</td><td>44.7</td></tr><tr><td>Uniform routing</td><td>63.9</td><td>44.7</td></tr><tr><td>Random routing</td><td>64.8</td><td>44.8</td></tr><tr><td>Learned routing</td><td>65.9</td><td>46.1</td></tr></table>

## B.2 EXPERT ROUTING STRATEGIES

Table 5 compares learned routing with uniform and random expert combinations, together with a shared-FFN-only reference. The shared-FFN-only variant disables the expert branches of the trained full model at inference; the + LCL variant in Table 2 is trained without SR-MoE. Learned routing achieves the highest ID and OOD scores of 65.9% and 46.1%, exceeding uniform routing by 2.0 and 1.4 percentage points and random routing by 1.1 and 1.3 points, respectively. Uniform routing does not improve upon the shared-FFN-only reference, while random routing yields only small increases. These results suggest that the benefit of the expert set depends on how its outputs are combined and support learning the routing weights from the prediction context. Figure 6 provides qualitative comparisons of the same expert routing strategies.

![](images/038ef7825124c9b89f47f4279745dd1c49a837ed20247ae47d00e9857278a484.jpg)  
Figure 6: Qualitative comparison of SR-MoE routing strategies. We compare video reasoning results with learned routing, shared FFN only, uniform routing, and random routing. The shared FFN-only variant disables expert branches at inference. Disrupting learned expert coordination degrades reasoning performance in the illustrated examples. Green and red dashed boxes highlight correct and incorrect results, respectively.

## B.3 PREDICTED TRAJECTORY

Table 6 examines whether the generator uses the content of the predicted rollout by changing its trajectory input while keeping the observation, instruction, and sampling seed fixed. Removing trajectory conditioning reduces ID performance from 65.9% to 51.5%, a drop of 14.4 percentage points. Replacing the predicted rollout with one from another sample yields 58.5%, a drop of 7.4 points despite retaining the conditioning pathway. This comparison indicates that the ID benefit depends in part on the correspondence between the rollout and the current sample, beyond the mere presence of an additional conditioning signal. The effect is smaller on OOD tasks: removing conditioning lowers the score from 46.1% to 44.9%, whereas cross-sample substitution

Table 6: Ablation of latent trajectory guidance. We test whether task success depends on the sample-specific temporal content of the rollout, rather than merely the presence of an additional conditioning signal.
<table><tr><td>Rollout condition Intervention</td><td></td><td colspan="2">Task success (%)</td></tr><tr><td></td><td></td><td>ID↑</td><td>OOD↑</td></tr><tr><td>Predicted rollout</td><td>Use the model-predicted trajectory</td><td>65.9</td><td>46.1</td></tr><tr><td>No conditioning</td><td>Remove all trajectory conditioning</td><td>51.5</td><td>44.9</td></tr><tr><td>Repeated initial state</td><td>Repeat the initial state across all rollout steps</td><td>62.5</td><td>43.3</td></tr><tr><td>Cross-sample rollout</td><td>Use the predicted trajectory from another sample</td><td>58.5</td><td>45.9</td></tr></table>

yields 45.9%, close to the original score. Thus, sample-matched rollout content contributes more clearly to ID performance, consistent with the smaller OOD gains in the main evaluation.

## C FURTHER VISUALIZATION

## C.1 COMPARISONS WITH OPEN-SOURCE MODELS

Figure 7 compares VR-JEPA with VBVR-Wan2.2 and LTX-2.3-I2AV on additional reasoning tasks. In the shape-sorting example, VR-JEPA groups objects by type and orders them by size, while VBVR-Wan2.2 interleaves the groups and LTX-2.3-I2AV changes object attributes. In the selectiveoutlining example, VR-JEPA marks the target circle, whereas the baselines additionally select an outside circle or outline the enclosing region, illustrating errors in applying spatial selection constraints.

![](images/7c466a154f519e466e33ece255bd0bb01ec6441d4f24a73b9b2e7299f60c4a84.jpg)  
Figure 7: Additional qualitative comparisons with open-source models on VBVR-Pro-Bench. Given the initial state and instruction, we compare the video reasoning results of VR-JEPA with VBVR-Wan2.2 and LTX-2.3-I2AV. Green and red dashed boxes highlight correct results and reasoning errors, respectively.

## C.2 COMPARISONS WITH PROPRIETARY VIDEO MODELS

Figure 8 compares VR-JEPA with Kling VIDEO 3.0 and Seedance 2.0. In the domino example, VR-JEPA completes the red branch while stopping the blue branch at the marked gap, whereas both baselines propagate beyond it. In the animal-matching example, VR-JEPA places each face in its corresponding outline, while both baselines mismatch the identities and destinations. These examples illustrate better adherence to interaction rules and object correspondence.

![](images/9b92b19d6fbc397d36ee019ee389eedec45c4b674b58a93275934f157d4cd7b6.jpg)  
Figure 8: Additional qualitative comparisons with proprietary video models on VBVR-Pro-Bench. Given the initial state and instruction, we compare the video reasoning results of VR-JEPA with Kling VIDEO 3.0 and Seedance 2.0. Green and red dashed boxes highlight correct results and reasoning errors, respectively.

## C.3 VISUAL REPRESENTATION SPACE ABLATION

Figure 9 complements Table 3 by comparing video reasoning results under the three visual representation configurations.  
![](images/7aa9e7a0ceb27caba801e198bdb066441f994c718cf32a3386d76785a46024fd.jpg)  
Figure 9: Qualitative comparison of visual representations on VBVR-Pro-Bench. Given the initial state and instruction, we compare video reasoning results guided by, from left to right: (1) predicted latent rollouts in V-JEPA 2.1 space (VR-JEPA), (2) initial-state DINOv3 features, and (3) predicted latent rollouts in DINOv3 space. Green and red dashed boxes highlight correct results and reasoning errors, respectively.

![](images/20d54e6f8a76611405ad1ddfe22e53f609da3c438baa0e4bd65c48a1b9e43004.jpg)

Figure 10: Qualitative ablation of latent trajectory guidance. We compare video reasoning results using the predicted rollout with three interventions: removing trajectory conditioning, repeating the initial state across rollout steps, and substituting a rollout from another sample. Red dashed boxes highlight reasoning errors.

## C.4 COMPARISON OF LATENT TRAJECTORY GUIDANCE

Figure 10 compares generations conditioned on the predicted rollout with three interventions: removing trajectory conditioning, repeating the initial state, and substituting another sample’s rollout. In the upper example, predicted-rollout conditioning more closely follows the reference sequence of shape and attribute changes, whereas the interventions introduce mismatches in intermediate or final states. In the maze example, the predicted-rollout result more closely follows the reference path, while the intervened results deviate at the locations marked in red. Removing conditioning tests the presence of trajectory guidance; repeating the initial state tests whether its temporal evolution matters; and substituting another sample’s rollout tests the importance of instance-specific content. These examples illustrate the different failure modes, while the quantitative intervention results are needed to assess how consistently they occur.

## C.5 TASK-DEPENDENT EXPERT ROUTING

Figure 11 visualizes the mean routing weights for ten tasks, averaged across layer4 and layer5, the two SR-MoE blocks of the latent predictor. The task-dependent distributions are broadly consistent with the experts’ designated skills. Maze navigation and LEGO assembly assign nearly all routing weight to SPA (0.993) and CMP (0.998), respectively, while gravity and bouncing favors DYN (0.688). Unique-shape identification assigns its largest weight to VIS, object-to-target matching to IDN, and next-color prediction and sequence completion to ABS.

<table><tr><td>Unique-shape identification</td><td>0.634</td><td>0.024</td><td>0.326</td><td>0.000</td><td>0.002</td><td>0.000</td><td>0.014</td><td rowspan="5">1.00 Meg Wwng ght 0.75</td></tr><tr><td>Object-to-target matching</td><td>0.418</td><td>0.085</td><td>0.487</td><td>0.000</td><td>0.000</td><td>0.010</td><td>0.000</td></tr><tr><td>Next-color prediction</td><td>0.124</td><td>0.020</td><td>0.368</td><td>0.000</td><td>0.000</td><td>0.000</td><td>0.488</td></tr><tr><td>Shortest directed-graph path</td><td>0.278</td><td>0.247</td><td>0.006</td><td>0.408</td><td>0.055</td><td>0.006</td><td>&lt;0.001</td></tr><tr><td>Maze navigation</td><td>0.006</td><td>0.000</td><td>0.000</td><td>0.993</td><td>0.000</td><td>0.000</td><td>0.001</td></tr><tr><td>Sliding puzzle</td><td>0.000</td><td>0.005</td><td>0.000</td><td>0.634</td><td>0.000</td><td>0.279</td><td>0.081</td></tr><tr><td>Gravity and bouncing</td><td>0.000</td><td>0.310</td><td>0.002</td><td>0.000</td><td>0.688</td><td>0.000</td><td>0.000</td></tr><tr><td>LEGO assembly</td><td>0.000</td><td>0.000</td><td>0.000</td><td>0.000</td><td>0.002</td><td>0.998</td><td>0.000</td></tr><tr><td>Symmetry completion</td><td>0.026</td><td>0.000</td><td>&lt;0.001</td><td>0.000</td><td>0.000</td><td>0.814</td><td>0.159</td></tr><tr><td>Sequence completion</td><td>0.039</td><td>0.094</td><td>0.155</td><td>0.000</td><td>&lt;0.001</td><td>0.000</td><td>0.712</td></tr><tr><td></td><td>VIS</td><td>NUM</td><td>IDN</td><td>SPA</td><td>DYN</td><td>CMP</td><td>ABS</td></tr></table>

Figure 11: Task-dependent expert routing. Mean routing weights for ten tasks, averaged across the two SR-MoE predictor layers (layer4 and layer5). Rows denote tasks and columns denote the seven skill-specific experts; darker cells indicate larger weights. The distributions show both concentrated expert selection and allocations spread across multiple experts.

Other tasks distribute substantial average weight across several experts. Sliding puzzle allocates weight primarily to SPA (0.634) and CMP (0.279), while shortest directed-graph path draws on SPA (0.408), VIS (0.278), and NUM (0.247). These patterns are consistent with task-dependent expert composition rather than a fixed assignment of every task to a single expert. The averages summarize expert utilization rather than establish each expert’s causal contribution; the removal analysis in Section C.6 provides complementary evidence by examining the effects of suppressing a selected expert.

## C.6 SKILL-SPECIFIC EXPERT REMOVAL

Figure 12 compares learned routing with removal of the highest-weight expert in seven selected examples, ordered by the designated skills VIS, NUM, IDN, SPA, DYN, CMP, and ABS. The interventions produce several distinct errors. Border completion leaves a shape unoutlined; ordinal selection marks the largest circle instead of the second-largest; and shape sorting duplicates a small red square. The maze output includes an invalid path, and the domino output propagates beyond the marked gap. The final two examples select the wrong size in a shape-color sequence and the wrong color in a repeating cycle.

These errors are consistent with the reasoning operations associated with the removed experts and provide qualitative evidence of differentiated contributions. Because the experiment removes the highest-weight expert in selected examples, it does not establish an exclusive mapping between each expert and one skill or separate skill specificity from general sensitivity to removing a strongly weighted expert.

![](images/e05ed1483c9f821d398a2e2e083286f45adbea15326fe948ff6dfb4601a6dfb7.jpg)  
Figure 12: Probing expert specialization through skill removal. We remove the highest-weight expert for each of seven examples and examine whether the resulting errors reflect the loss of its associated reasoning skill. From top to bottom, the removed experts correspond to visual discrimination, numerical reasoning, object identity and persistence, spatial reasoning and planning, physical dynamics, compositional reasoning, and abstract rule inference. Green and red dashed boxes highlight correct results and reasoning errors, respectively.
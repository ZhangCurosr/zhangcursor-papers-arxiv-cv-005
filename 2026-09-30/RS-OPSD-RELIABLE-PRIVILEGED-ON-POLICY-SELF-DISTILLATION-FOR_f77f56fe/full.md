![](images/c7f1630cc8c2347c9b4d0e844925e46de834fc94a0f9357ea96fa384f8918523.jpg)

# RS-OPSD: RELIABLE PRIVILEGED ON-POLICY SELF-DISTILLATION FOR ULTRA-HIGH-RESOLUTION REMOTE SENSING VQA

Chengjie Jiang<sup>1,∗†</sup> Yunqi Zhou<sup>2,∗</sup> Jiafeng Yan<sup>3</sup> Sihang Zhao<sup>4,5</sup> Chun Yuan<sup>1,‡</sup> Jing Li<sup>4,5,†‡</sup>

<sup>1</sup>Tsinghua University <sup>2</sup>Zhejiang University <sup>3</sup>Central University of Finance and Economics <sup>4</sup>East China Normal University <sup>5</sup>Key Laboratory of Geographic Information Science

![](images/a90983b3a257c791bdb6aa0b00f343056be47e8b8549ff3842271750778b4c51.jpg)

Figure 1: Average scores on Ultra-High-Resolution remote sensing VQA benchmarks, including XLRS-Bench, MME-RealWorld-RS, and LRS-VQA.

## ABSTRACT

Ultra-high-resolution (UHR) remote sensing visual question answering (VQA) requires models to resolve small visual evidence within extremely large images. Existing approaches typically rely on token pruning, visual search, or toolaugmented reasoning at inference time. We instead investigate whether the benefit of zoom-in visual privilege can be internalized into the model. We introduce RS-OPSD, a reliable privileged on-policy self-distillation (OPSD) framework for UHR remote sensing VQA. To provide high-quality privileged information with explicit question-relevant evidence, we construct GeoEvidence-6K, containing 6,750 VQA samples across seven task categories with evidence-region annotations, and develop Human Feedback-Guided Skill Refinement (HF-SR) for scalable annotation. To address context loss from tight crops and conflicting signals from imperfect teachers, RS-OPSD introduces Context-Preserving Visual Privilege (CPVP) and Correctness-Aligned Distillation (CAD). Without any additional visual search or tool calls at inference time, RS-OPSD achieves state-of-the-art (SOTA) performance on XLRS-Bench, MME-RealWorld-RS, and LRS-VQA, outperforming pervious SOTA models of comparable scale by an average of 4.0 percentage points. Moreover, our 2B variant, RS-OPD-Lite, surpasses most 8B-scale models while achieving the fastest measured inference speed. Our Code, GeoEvidence-6K, and the model weights for RS-OPSD and RS-OPD-Lite are publicly available.

## 1 INTRODUCTION

UHR remote sensing VQA requires models to reason over images containing tens of millions of pixels, where the evidence is often confined to only a tiny fraction of the scene. Although modern vision-language foundation models (Bai et al., 2025b;a; Li et al., 2024; Zhu et al., 2025) exhibit strong semantic understanding, their effective perceptual capacity is easily overwhelmed by such extremely large visual inputs. Recent studies such as ZoomSearch (Zhou et al., 2025) and Vision-OPD (Yuan et al., 2026) suggest that many fine-grained failures are not simply caused by an inability to recognize the target itself, but by a more fundamental where-to-look bottleneck—the model often fails to effectively attend to and exploit question-relevant visual evidence in extremely large images.

![](images/254f5f9877c87553c4caf799597650f69a98ec60d8213056394bbe1d1a1f114e.jpg)  
Figure 2: Comparison of representative paradigms for UHR remote sensing VQA.

As illustrated in Figure 2, existing approaches mainly address this bottleneck from three directions. First, token-pruning methods exploit the high redundancy of UHR imagery by retaining only a subset of visual tokens (Luo et al., 2025; Dang et al., 2026). However, hard pruning is typically difficult to reverse: once evidence token is discarded, it is no longer available to subsequent reasoning. Second, visual search methods search over image patches to identify relevant regions before answering (Zhou et al., 2025; Ma et al., 2026). While effective, such search introduces additional inference-time computation. Third, tool-augmented reasoning methods allow the model to actively invoke zoom-in tools during inference, often through multiple rounds of refinement (Liu et al., 2026; Wang et al., 2026). These methods demonstrate the benefit of exposing the model to targeted high-resolution views, but require additional tool calls and visual search. This observation motivates a different question: Can the privilege brought by visual zooming be internalized into the model during training, such that it can exploit the samefine-grained visual prior without additional search or tool use?

A natural way to realize this idea is on-policy self-distillation (OPSD) (Zhao et al., 2026). Instead of introducing additional visual operations at inference time, OPSD equips a teacher with privileged visual information during training and transfers this advantage to a student operating under the standard input. In UHR remote sensing VQA, zoomed-in evidence can naturally serve as such visual privilege: during training, the teacher is exposed to fine-grained question-relevant views, while the student learns to better exploit the corresponding visual evidence from the standard full-image input.

Realizing this training paradigm requires high-quality zoom-in evidence as visual privilege. We therefore construct GeoEvidence-6K, a UHR remote sensing VQA dataset containing 6,750 samples across seven task categories, with explicit annotations for question-relevant visual evidence. Since accurately annotating such regions requires careful inspection of extremely large images, we further develop Human Feedback-Guided Skill Refinement (HF-SR), which progressively incorporates human feedback into reusable annotation skills to improve subsequent data construction.

We then conduct a pilot study using full-image inputs for the student and evidence-crop privileges for a teacher initialized from the same model. However, this straightforward formulation yields only marginal gains. We identify two major bottlenecks: first, the crop-only privilege provides only a modest advantage over the student; second, the privileged teacher itself remains imperfect, which can introduce substantial conflicting or even harmful supervision. To address these issues, we introduce two complementary designs in RS-OPSD. Context-Preserving Visual Privilege (CPVP) equips the teacher with global, contextual, and fine-grained evidence views, strengthening the privileged signal while preserving surrounding context. Correctness-Aligned Distillation (CAD) filters teacher supervision through sample-level reliability gating and retains only token-level update directions aligned with the correctness of the student response.

Using Qwen3-VL-8B (Bai et al., 2025a) as the base model, RS-OPSD achieves SOTA performance on all three UHR remote sensing VQA benchmarks, outperforming the strongest competing methods by 1.6 percentage points on XLRS-Bench (Wang et al., 2025b), 3.9 points on MME-RealWorld RS (Zhang et al., 2025), and 2.0 points on LRS-VQA (Luo et al., 2025). We further instantiate the same framework with a Qwen3-VL-2B student and a privileged Qwen3-VL-8B teacher, yielding RS-OPD-Lite. Despite retaining only the 2B student at inference time, RS-OPD-Lite surpasses most 8B-scale models. Moreover, since neither variant requires additional visual search or tool calls, they both achieve low inference latency, with RS-OPD-Lite achieving the lowest inference latency, 12.8% lower than the second-fastest method. Our main contributions are summarized as follows:

• We introduce RS-OPSD, which formulates UHR remote sensing VQA as an on-policy selfdistillation problem and internalizes the priviledge of zoom-in evidence into the model.

• To support this paradigm, we construct GeoEvidence-6K, a high-quality UHR remote sensing VQA dataset with explicit evidence-region annotations. We further propose Human Feedback-Guided Skill Refinement, which iteratively converts accumulated human feedback into reusable annotation skills for subsequent data construction.

• We develop two complementary components for reliable privileged distillation: Context-Preserving Visual Privilege, which provides the teacher with global, contextual, and fine-grained visual evidence, and Correctness-Aligned Distillation, which suppresses unreliable teacher supervision through sample-level reliability gating and correctness-aligned token updates.

• Extensive experiments show that RS-OPSD achieves SOTA performance while maintaining efficient, search-free and tool-free inference. Moreover, RS-OPD-Lite retains competitive accuracy and achieves the fastest inference speed among all compared methods.

## 2 RELATED WORK

As shown in Figure 2, existing approaches to UHR remote sensing VQA mainly follow three paradigms: token pruning, visual search, and tool-augmented reasoning.

Token-pruning methods exploit the redundancy in UHR remote sensing imagery by retaining question-relevant visual tokens while discarding less informative regions. GeoLLaVA-8K (Wang et al., 2025a) combines Background Token Pruning with Anchored Token Selection to suppress redundant background tokens while preserving object-centric visual tokens. UHR-BAT (Dang et al., 2026) further introduces query-guided multi-scale token selection and region-faithful preserve-andmerge strategies to allocate a fixed token budget toward question-relevant regions.

Visual search methods explicitly identify informative regions and construct compact evidence from the original UHR image before reasoning, avoiding dense processing of the entire scene. Zoom-Search (Zhou et al., 2025) performs hierarchical zoom search to locate question-relevant patches and reassembles the selected evidence with spatial layout preserved. WeaveEarth (Ma et al., 2026) constructs a compact Minimal Support Evidence Set through global-aware evidence selection and integrates local evidence, spatial metadata, and relative topology for subsequent reasoning.

Tool-augmented reasoning methods equip VLMs with explicit visual tools, enabling active acquisition of high-resolution evidence during inference. ZoomEarth (Liu et al., 2026) introduces an active perception paradigm that learns adaptive cropping and zooming through supervised finetuning and reinforcement learning, allowing the model to revisit informative regions during reasoning. Building on this direction, GeoEyes (Wang et al., 2026) further addresses the tendency of zoom-enabled models to follow homogeneous tool-use patterns, learning more adaptive on-demand zooming and stopping behaviors through staged supervised and reinforcement learning.

## 3 GEOEVIDENCE-6K

To provide the teacher with zoom-in visual privileges, we require high-quality UHR remote sensing data pairing each question with a target region. We therefore construct GeoEvidence-6K, a manually verified UHR remote sensing VQA dataset with explicit target-region annotations. Since annotating these regions is labor-intensive and prone to repetitive errors, we introduce Human Feedback-Guided Skill Refinement (HF-SR), a human-in-the-loop annotation framework that consolidates human corrections into reusable annotation skills. Human feedback thus not only cor rects individual samples but also continuously improves subsequent annotation.

## 3.1 DATASET OVERVIEW

GeoEvidence-6K is constructed from 1,050 UHR remote sensing images collected from five public data sources, as summarized in Table 1. The images average $1 0 , 7 1 4 \times 1 0 , 7 1 4$ pixels, with ground sampling distances (GSDs) ranging from 0.08 m to 0.50 m, and span diverse urban, agricultural, natural, coastal, and mountainous scenes. The annotated target regions average only $1 , 0 9 5 \times 1 , 0 7 4$ pixels, roughly 1% of the full-image area. This large scale disparity highlights the difficulty of locating and resolving question-relevant visual evidence in UHR scenes.

Table 1: Statistics of the image sources used in GeoEvidence-6K.
<table><tr><td>Source</td><td>Nums.</td><td>Image Size</td><td>Target Size</td><td>GSD</td><td>Country</td></tr><tr><td>MiniFrance (Castillo-Navarro et al., 2022)</td><td>110</td><td>10K</td><td>509×517</td><td>0.50 m</td><td>France</td></tr><tr><td>HRSCD (Daudt et al., 2019)</td><td>40</td><td>10K</td><td>516×517</td><td>0.50 m</td><td>France</td></tr><tr><td>SWISSIMAGE (Federal Office of Topography swisstopo, 2024)</td><td>300</td><td>10K</td><td> $8 3 7 \times 8 1 7$ </td><td>0.10 m</td><td>Switzerland</td></tr><tr><td>GeoNRW (Geobasis NRW, 2025)</td><td>300</td><td>10K</td><td> $9 9 7 \times 9 8 7$ </td><td>0.10 m</td><td>Germany</td></tr><tr><td>Beeldmateriaal (Beeldmateriaal Nederland, 2025)</td><td>300</td><td>12K</td><td>1,741×1,696</td><td>0.08m</td><td>Netherlands</td></tr></table>

The dataset covers seven multiple-choice task categories. Five are local-evidence tasks, including counting, position, color, category, and shape, while land-use and route-planning are global-context tasks. For each source image, we select diverse target regions and annotate questions of different task types, resulting in a total of 6,750 training samples. Representative annotations are shown in Figure 3.

## 3.2 HF-SR: HUMAN FEEDBACK-GUIDED SKILL REFINEMENT

Figure 3 illustrates our HF-SR pipeline. Let A denote a fixed annotation agent and $S _ { k }$ the annotation skill at refinement stage k. Given a UHR remote sensing image $x _ { i }$ , the agent generates ${ \hat { y } } _ { i } = { \mathcal { A } } ( x _ { i } ; S _ { k } )$ , which is reviewed by humans to obtain feedback $f _ { i } = { \mathcal { H } } ( x _ { i } , { \hat { y } } _ { i } )$ . Unlike conventional human verification, HF-SR reuses accumulated feedback to update the annotation skill. Specifically, feedback collected between two checkpoints is ${ \mathcal { F } } _ { k } = \{ f _ { i } \ | \ c _ { k } < i \leq c _ { k + 1 } \}$ , and the skill is refined as $S _ { k + 1 } = \mathcal { R } ( S _ { k } , \mathcal { F } _ { k } )$ , where R denotes an agent-driven refinement process that summarizes recurring errors and human corrections into reusable procedural rules.

The agent follows an overview-to-detail procedure: it first inspects a downsampled full-image overview and then examines candidate regions at native resolution. For local-evidence tasks, it selects spatially diverse target regions and generates the corresponding bounding boxes, questions, and reference answers. Human reviewers jointly inspect the full image, target regions, and structured annotations, correcting issues such as ambiguous references, incomplete target regions, invalid crop geometry, or inconsistent question–answer design.

To account for the evolving maturity of the annotation skill, we schedule refinement checkpoints using cumulative Fibonacci intervals, with $\begin{array} { r } { c _ { k } = \sum _ { i = 1 } ^ { k } F _ { j } } \end{array}$ and $F _ { 1 } = F _ { 2 } = 1$ . Frequent updates are used during the cold-start stage, when new failure patterns emerge rapidly, while progressively longer intervals are adopted as the skill stabilizes to amortize refinement cost.

![](images/92d7ba2c90f64665dbae24162c7e4049c5fedf443bbf4fac2612890b131aa57e.jpg)  
Figure 3: Overview of the HF-SR pipeline, together with representative cases of GeoEvidence-6K.

## 4 PRELIMINARY

On-Policy Distillation. On-policy distillation (OPD) transfers knowledge from a teacher policy π<sub>T</sub> to a student policy $\pi _ { \theta }$ on trajectories sampled from the student itself. Given an input $x ,$ the student first generates $\mathbf { y } = ( y _ { 1 } , \dots , y _ { T } ) \sim \pi _ { \theta } ( \cdot \mid x )$ . At each student-visited prefix $\mathbf { y } _ { < t }$ , the student and teacher produce next-token distributions:

$$
p _ { t } ( \cdot ) = \pi _ { \theta } ( \cdot  { | } x ,  { \mathbf { y } } _ { < t } ) , \qquad q _ { t } ( \cdot ) = \pi _ { T } ( \cdot  { | } x ,  { \mathbf { y } } _ { < t } ) .\tag{1}
$$

OPD then minimizes the reverse KL divergence along the student trajectory:

$$
\mathcal { L } _ { \mathrm { O P D } } = \mathbb { E } _ { \mathbf { y } \sim \pi _ { \theta } ( \cdot | x ) } \left[ \sum _ { t = 1 } ^ { T } D _ { \mathrm { K L } } \left( p _ { t } \| q _ { t } \right) \right] .\tag{2}
$$

Unlike off-policy distillation, the teacher provides supervision on states actually visited by the current student policy, reducing train–inference distribution mismatch.

On-Policy Self-Distillation. On-policy self-distillation (OPSD) removes the need for a separate stronger teacher by assigning different contextual roles to the same underlying model (Zhao et al., 2026). The student observes the standard input $x ,$ whereas the teacher is additionally conditioned on privileged information z that is available only during training:

$$
p _ { t } ^ { S } = \pi _ { \theta } ( \cdot  { | } x ,  { \mathbf { y } } _ { < t } ) , \qquad p _ { t } ^ { T } = \pi _ { \theta _ { T } } ( \cdot  { | } x , z ,  { \mathbf { y } } _ { < t } ) .\tag{3}
$$

The privileged teacher evaluates the same student-generated trajectory, and the student is optimized toward its distribution:

$$
\mathcal { L } _ { \mathrm { O P S D } } = \mathbb { E } _ { { \mathbf { y } } \sim \pi _ { \theta } ( \cdot \vert x ) } \left[ \sum _ { t = 1 } ^ { T } D _ { \mathrm { K L } } \left( p _ { t } ^ { S } \parallel \mathrm { s g } \big [ p _ { t } ^ { T } \big ] \right) \right] ,\tag{4}
$$

where $\mathrm { s g } [ \cdot ]$ denotes stop-gradient. In this way, OPSD transfers capabilities exposed by privileged training information into the student while retaining the standard input at inference time.

## 5 RS-OPSD

## 5.1 MOTIVATION

We first conduct a pilot study to examine whether zoom-in evidence can be directly exploited through OPSD for UHR remote sensing VQA. As shown in Table 3, LoRA fine-tuning on GeoEvidence-6K improves the baseline from 40.8 to 45.7 on average, whereas directly applying OPSD with the evidence crop as the teacher privilege achieves only 45.2. Although OPSD still improves over the base model, it fails to outperform standard LoRA fine-tuning, suggesting that crop-based privileged distillation is not yet sufficiently effective in this setting.

To understand this limitation, we further evaluate the teacher–student capability gap on GeoEvidence-6K under direct-answer evaluation. Using the same Qwen3-VL-8B model, the full image input achieves 63.85% accuracy, while the evidence crop reaches 69.91%. This diagnostic reveals two bottlenecks: first, the crop privilege provides only a 6.06-point advantage, resulting in a relatively weak privileged signal; second, the privileged model itself remains imperfect, with nearly 30% of the training samples answered incorrectly, which can introduce conflicting supervision. These observations motivate two complementary designs in RS-OPSD: Context-Preserving Visual Privilege (CPVP) enriches the teacher with additional contextual and global information to strengthen its privileged advantage, while Correctness-Aligned Distillation (CAD) filters unreliable teacher signals and retains only update directions aligned with verified answer correctness.

![](images/7024973aa73dbeb627d401401aa5f021848427c0fdb9607c722b8e8e1d0c13bd.jpg)  
Figure 4: Overview of RS-OPSD, consisting of CPVP and CAD.

## 5.2 CPVP: WHAT SHOULD THE TEACHER SEE?

Building on the OPSD formulation in Section 4, each training sample contains a UHR remote sensing image $I _ { i } ,$ a question $q _ { i } ,$ , a ground-truth answer $a _ { i } ^ { * }$ , and an annotated evidence region $B _ { i }$ . In our setting, the standard input x corresponds to the global evidence $I _ { G }$ , while the privileged information z is constructed from the annotated evidence region and provided only to the teacher.

A tight evidence crop improves target visibility but may discard surrounding context. Conversely, the global image preserves scene-level information but weakens fine-grained targets. As shown in Figure 4, we therefore propose CPVP, which provides the teacher with three complementary views: Global Evidence $I _ { G }$ , Contextual Evidence $I _ { C } ,$ , and Fine-grained Evidence $I _ { F }$ . Specifically, $I _ { F }$ is the tight crop corresponding to $B _ { i } ,$ while $I _ { C }$ is obtained by keeping the same crop center and doubling its width and height to include surrounding context. The resulting privileged information is $z _ { i } = \{ I _ { C , i } , I _ { F , i } \}$ . The student observes the standard input $\left( I _ { G , i } , q _ { i } \right)$ , while the teacher is additionally conditioned on the privileged information $z _ { i } = \{ I _ { C , i } , I _ { F , i } \}$ CPVP thus strengthens the teacher privilege with fine-grained evidence and surrounding context while preserving the global input.

## 5.3 CAD: WHEN SHOULD THE STUDENT TRUST IT?

Although CPVP provides the teacher with richer visual evidence, privileged information does not guarantee reliable supervision. We therefore propose CAD, which filters privileged supervision at both the sample and token levels. Following the on-policy formulation in Section 4, the privileged teacher evaluates the same student-generated trajectory. For each sampled token $y _ { i , t }$ , we define the teacher–student preference as

$$
r _ { i , t } = \mathrm { s g } \left[ \log p _ { i , t } ^ { T } - \log p _ { i , t } ^ { S } \right] ,\tag{5}
$$

where $p _ { i , t } ^ { S }$ and $p _ { i , t } ^ { T }$ denote the probabilities assigned to the sampled token $y _ { i , t }$ under the student and privileged-teacher distributions defined in Section 4, respectively.

Teacher reliability gate. Before using these token-level preferences, CAD first evaluates whether the privileged teacher provides reliable supervision for the current sample. Given the GT response $a _ { i } ^ { * }$ , we perform a teacher-forced prefix probe and define

$$
g _ { i } = \prod _ { k } \mathbf { 1 } \left[ \tau _ { i , k } ^ { * } = \arg \operatorname* { m a x } _ { u \in \mathcal { C } _ { i , k } } \pi _ { \theta _ { T } } \left( u \mid I _ { G , i } , z _ { i } , q _ { i } , \pmb { \tau } _ { i , < k } ^ { * } \right) \right] ,\tag{6}
$$

where $\tau _ { i } ^ { * }$ denotes the tokenized GT response and $\mathcal { C } _ { i , k }$ denotes the valid continuations at position k.   
Thus, $g _ { i } = 1$ only when the teacher remains aligned with the GT response throughout the probe.

Correctness-aligned token filtering. We further use the correctness of the student response to determine which teacher preferences should be retained:

$$
R _ { i } = \left\{ { \begin{array} { l l } { + 1 , } & { { \hat { a } } _ { i } = a _ { i } ^ { * } , } \\ { - 1 , } & { { \hat { a } } _ { i } \neq a _ { i } ^ { * } . } \end{array} } \right.\tag{7}
$$

The resulting CAD signal is

$$
A _ { i , t } = g _ { i } R _ { i } \left[ R _ { i } r _ { i , t } \right] _ { + } .\tag{8}
$$

When the student response is correct, CAD retains teacher preferences that reinforce the sampled tokens; when the response is incorrect, it retains preferences that suppress them. Teacher signals that are unreliable or inconsistent with verified answer correctness are discarded.

The CAD objective is

$$
\mathcal { L } _ { \mathrm { C A D } } = - \frac { \sum _ { i , t } m _ { i , t } ^ { \mathrm { a n s } } \ \mathrm { s g } [ A _ { i , t } ] \log p _ { i , t } ^ { S } } { \sum _ { i , t } m _ { i , t } ^ { \mathrm { a n s } } } ,\tag{9}
$$

where $m _ { i , t } ^ { \mathrm { a n s } }$ is a binary mask selecting valid answer tokens.

Finally, we regularize the student toward a frozen reference model that receives the same standard input $( I _ { G , i } , q _ { i } )$ as the student. The overall training objective is

$$
\begin{array} { r } { \mathcal { L } = \mathcal { L } _ { \mathrm { C A D } } + \lambda _ { \mathrm { K L } } \mathcal { L } _ { \mathrm { K L } } , } \end{array}\tag{10}
$$

where $\lambda _ { \mathrm { K L } }$ controls the strength of reference regularization. We further apply gradient-norm clipping with threshold $c _ { \mathrm { g r a d } }$ to stabilize optimization.

## 6 EXPERIMENTS

## 6.1 EXPERIMENTAL SETTINGS

Model Training. RS-OPSD is initialized from Qwen3-VL-8B-Instruct and trained exclusively on GeoEvidence-6K. We set the KL regularization coefficient $\lambda _ { \mathrm { K L } } = 1 \times 1 0 ^ { - 3 }$ , the gradient clipping threshold $c _ { \mathrm { g r a d } } = 5$ , and the global batch size to 96. Training lasts for 150 steps. The teacher is initialized from the student and updated via exponential moving average (EMA) throughout training. For RS-OPD-Lite, the student is initialized from Qwen3-VL-2B-Instruct, while Qwen3-VL-8B-Instruct serves as the privileged teacher and remains frozen during training. All other settings follow RS-OPSD, except that RS-OPD-Lite is trained for 120 steps.

Benchmarks. We evaluate our models on three UHR remote sensing VQA benchmarks: XLRS-Bench (Wang et al., 2025b), MME-RealWorld-RS (Zhang et al., 2025), and LRS-VQA (Luo et al., 2025). XLRS-Bench has an average image resolution of 8,500 × 8,500 pixels and evaluates fine-grained perception and reasoning tasks. MME-RealWorld-RS has an average resolution of 5,602 × 4,445 pixels and contains 3,738 single-choice questions covering color, counting, and position. LRS-VQA contains 1,657 images with an average resolution of 7,099×6,329 pixels and 7,333 question–answer pairs across eight categories. We follow the official evaluation protocols of each benchmark and report the corresponding task-averaged scores.

Baselines. We compare against 14 representative baselines spanning four categories. (1) Closedsource VLMs: GPT-4o, Claude 3.7 Sonnet, and Gemini 2.5 Pro. (2) Open-source VLMs: LLaVA-OV-7B, Qwen2.5-VL-7B, Qwen3-VL-8B, and InternVL3-8B. (3) Remote sensing VLMs: GeoChat and VHM. (4) Specialized methods for UHR remote sensing VQA, including token-pruning approaches such as GeoLLaVA-8K and UHR-BAT, visual-search approaches such as ZoomSearch and WeaveEarth, and tool-augmented reasoning approaches represented by ZoomEarth.

Table 2: Quantitative comparison results. Green, yellow, and blue denote the first-, second-, and third-best results. Speed is the average inference time across the three benchmarks (s/sample).
<table><tr><td>Method</td><td>Pub.</td><td>Backbone</td><td>Param.</td><td>XLRS</td><td>MME</td><td>LRS</td><td>Avg.</td><td>Speed</td></tr><tr><td colspan="9">Closed-Source Vision Language Models</td></tr><tr><td>GPT-4o (Hurst et al., 2024)</td><td>一</td><td>一</td><td>一</td><td>32.4</td><td>28.2</td><td>27.1</td><td>29.2</td><td></td></tr><tr><td>Claude 3.7 Sonnet (Cla)</td><td>一</td><td>一</td><td>一</td><td>40.5</td><td>45.7</td><td>28.4</td><td>38.2</td><td></td></tr><tr><td>Gemini 2.5 Pro (Comanici et al., 2025)</td><td>1</td><td></td><td></td><td>44.1</td><td>51.7</td><td>30.9</td><td>42.2</td><td></td></tr><tr><td colspan="9">Open-Source Vision Language Models</td></tr><tr><td>LLaVA-OV-7B (Li et al., 2024)</td><td>TMLR&#x27;25</td><td>LLaVA-OV</td><td>7B</td><td>43.4</td><td>53.3</td><td>27.3</td><td>41.3</td><td>2.38</td></tr><tr><td>Qwen2.5-VL-7B (Bai et al., 2025b)</td><td>arXiv&#x27;25</td><td>Qwen2.5-VL</td><td>7B</td><td>47.4</td><td>41.0</td><td>29.4</td><td>39.3</td><td>1.87</td></tr><tr><td>Qwen3-VL-8B (Bai et al., 2025a)</td><td>arXiv&#x27;25</td><td>Qwen3-VL</td><td>8B</td><td>50.5</td><td>41.9</td><td>30.1</td><td>40.8</td><td>1.75</td></tr><tr><td>InternVL3-8B (Zhu et al., 2025)</td><td>arXiv’25</td><td>InternVL3</td><td>8B</td><td>45.6</td><td>55.0</td><td>29.2</td><td>43.3</td><td>1.25</td></tr><tr><td colspan="9">Remote Sensing Vision Language Models</td></tr><tr><td>GeoChat (Kuckreja et al., 2024)</td><td>CVPR’24</td><td>LLaVA-v1.5</td><td>7B</td><td>22.9</td><td>21.3</td><td>19.5</td><td>21.2</td><td>1.33</td></tr><tr><td>VHM (Pang et al., 2025)</td><td>AAAI&#x27;25</td><td>Vicuna</td><td>7B</td><td>32.8</td><td>24.1</td><td>26.7</td><td>27.9</td><td>2.64</td></tr><tr><td colspan="9">Specialized Methods for UHR Remote Sensing VQA</td></tr><tr><td>GeoLLaVA-8K (Wang et al., 2025a) UHR-BAT (Dang et al., 2026)</td><td>NIPS&#x27;25</td><td>LongVA</td><td>7B</td><td>51.5</td><td>28.4</td><td>25.8</td><td>35.2</td><td>1.30</td></tr><tr><td>ZoomSearch (Zhou et al., 2025)</td><td>ICML&#x27;26</td><td>LongVA</td><td>7B</td><td>48.6</td><td>33.3</td><td>20.0</td><td>34.0</td><td>7.40</td></tr><tr><td>WeaveEarth (Ma et al., 2026)</td><td>arXiv&#x27;25 MM&#x27;26</td><td>LLaVA-OV Qwen3-VL</td><td>7B 8B</td><td>48.0 49.5</td><td>57.6 47.2</td><td>30.2 31.3</td><td>45.3 42.7</td><td>43.5 2.94</td></tr><tr><td>ZoomEarth (Liu et al., 2026)</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td></td><td>CVPR&#x27;26</td><td>Qwen2.5-VL</td><td>3B</td><td>39.2</td><td>38.7</td><td>21.6</td><td>33.2</td><td>14.2</td></tr><tr><td colspan="9">Ours</td></tr><tr><td>RS-OPD-Lite</td><td>一</td><td>Qwen3-VL</td><td>2B</td><td>45.8</td><td>56.2</td><td>30.5</td><td>44.2</td><td>1.09</td></tr><tr><td>RS-OPSD</td><td></td><td>Qwen3-VL</td><td>8B</td><td>53.1</td><td>61.5</td><td>33.3</td><td>49.3</td><td>1.58</td></tr></table>

## 6.2 COMPARISON WITH STATE-OF-THE-ART METHODS

Overall performance. As shown in Table 2, RS-OPSD achieves the best performance on all three benchmarks, reaching 53.1 on XLRS-Bench, 61.5 on MME-RealWorld-RS, and 33.3 on LRS-VQA, with an average score of 49.3, exceeding the strongest competing method by 4.0 points. Detailed category-wise results and analyses are provided in Appendix D.

Accuracy–efficiency trade-off. RS-OPSD runs at 1.58 s/sample, even faster than Qwen3-VL-8B baseline, while avoiding the substantial overhead of visual search and tool calls. More notably, RS-OPD-Lite achieves an average score of 44.2 with only 2B parameters, outperforming most 8B-scale models. It is also the fastest method at 1.09 s/sample, with 12.8% lower latency than the secondfastest method. These results demonstrate that internalizing privileged visual information through OPSD can improve UHR remote sensing VQA while preserving both accuracy and efficiency.

(c) Task-Wise Student Accuracy Dynamics  
(b) Token-Level Reward Gating Dynamics  
Table 3: Ablation study of the core designs in RS-OPSD.
<table><tr><td colspan="2">Core Design</td><td rowspan="2"></td><td rowspan="2"></td><td colspan="3">Benchmarks</td></tr><tr><td>OPSD CPVP</td><td>CAD</td><td>XLRS MME-RW-RS LRS-VQA Avg.</td><td></td><td></td></tr><tr><td colspan="3">Qwen3-VL-8B-Instruct</td><td>50.5</td><td>41.9</td><td>30.1</td><td>40.8</td></tr><tr><td>x X</td><td>X</td><td>LoRA Finetuning on GeoEvidence-6K</td><td>49.9 50.5</td><td>55.2</td><td>32.1</td><td>45.7</td></tr><tr><td>√</td><td></td><td>Naïve OPSD in Subsec. 5.1</td><td></td><td>53.8</td><td>31.3 31.2</td><td>45.2</td></tr><tr><td>√</td><td>√</td><td></td><td> $\mathrm { O P S D + C P V P }$ </td><td>51.3</td><td>59.0</td><td>47.2</td></tr><tr><td>√</td><td>X</td><td></td><td>OPSD + CAD</td><td>51.2</td><td>58.3</td><td>47.0</td></tr><tr><td></td><td>√</td><td>√ √</td><td>RS-OPSD</td><td>53.1</td><td>61.5</td><td>49.3</td></tr></table>

Table 4: Hyperparameter studies of KL regularization and gradient clipping.
<table><tr><td colspan="2">KL Coefficient</td><td colspan="4">Benchmarks</td><td colspan="2">Clipping Threshold</td><td colspan="4">Benchmarks</td></tr><tr><td> $c _ { \mathrm { g r a d } }$ </td><td> $\lambda _ { \mathrm { K L } }$ </td><td>XLRS</td><td>MME-RW-RS</td><td>LRS-VQA</td><td>Avg.</td><td> $c _ { \mathrm { g r a d } }$ </td><td> $\lambda _ { \mathrm { K L } }$ </td><td>XLRS</td><td>MME-RW-RS</td><td>LRS-VQA</td><td>Avg.</td></tr><tr><td>1</td><td>0</td><td>51.3</td><td>59.0</td><td>31.2</td><td>47.2</td><td>5</td><td> $1 \times 1 0 ^ { - 3 }$ </td><td>53.1</td><td>61.5</td><td>33.3</td><td>49.3</td></tr><tr><td>1</td><td> $1 \times 1 0 ^ { - 2 }$ </td><td>51.2</td><td>62.0</td><td>32.8</td><td>48.7</td><td>20</td><td> $1 \times 1 0 ^ { - 3 }$ </td><td>51.0</td><td>60.0</td><td>34.0</td><td>48.3</td></tr><tr><td>1</td><td> $1 \times 1 0 ^ { - 3 }$ </td><td>53.2</td><td>60.8</td><td>33.1</td><td>49.0</td><td>100</td><td> $1 \times 1 0 ^ { - 3 }$ </td><td>51.5</td><td>59.9</td><td>33.4</td><td>48.3</td></tr></table>

## 6.3 ABLATION STUDIES AND ANALYSIS

Core design analysis. Table 3 evaluates the contribution of the core designs in RS-OPSD. LoRA (Hu et al., 2021) fine-tuning on GeoEvidence-6K improves the average score of Qwen3- VL-8B from 40.8 to 45.7, indicating the high quality of the GeoEvidence-6K. In contrast, directly applying OPSD with crop-only privilege in our pilot setting slightly reduces the average score to 45.2, confirming that na¨ıve privileged distillation is limited by the insufficient teacher–student gap and imperfect teacher supervision. Introducing CPVP raises the average score to 47.2, while CAD independently improves it to 47.0, showing that richer visual privilege and correctness-aligned supervision both contribute positively to mitigating these two bottlenecks. Combining CPVP and CAD further yields the best performance of 49.3, reaching the state-of-the-art results reported in Table 2.

Hyperparameter analysis. Table 4 examines the effects of the KL coefficient λ and gradient clipping threshold $c _ { \mathrm { g r a d } }$ . KL regularization consistently improves over the setting with $\lambda _ { \mathrm { K L } } = 0 .$ with $\lambda _ { \mathrm { K L } } = 1 0 ^ { - 3 }$ achieving the best average score of 49.0. Increasing $\lambda _ { \mathrm { K L } }$ to $1 0 ^ { - 2 }$ slightly degrades performance, suggesting that overly strong regularization may constrain adaptation to the privileged teacher. With $\lambda _ { \mathrm { K L } } = 1 \mathbf { \bar { 0 } } ^ { - 3 }$ fixed, $c _ { \mathrm { g r a d } } = 5$ performs best at 49.3, while $c _ { \mathrm { g r a d } } = 2 0$ and 100 both decrease the average to 48.3. We therefore use $\lambda _ { \mathrm { K L } } = 1 0 ^ { - 3 }$ and $c _ { \mathrm { g r a d } } = 5$ in all main experiments.

![](images/2b1803d7daef6034062cb2d2ed2407d69d06461ac4276519d0330480dc71ea33.jpg)

![](images/b612c4542bbabce01080cfa9f0eab94742f0714460274b079f273743d8db7871.jpg)

![](images/bf7d71f20eca5232862593ea9c1ed20c5d23ac7e9197a0f29b68996736cc7a2c.jpg)  
Figure 5: Training dynamics of RS-OPSD across three epochs.

Training dynamics analysis. Figure 5 further examines the training dynamics of RS-OPSD over three epochs. Figure 5(a) tracks the accuracy of the student and EMA teacher. Both improve steadily throughout training, while the teacher consistently remains stronger and the teacher–student gap gradually narrows, indicating progressive transfer of privileged knowledge to the student. Figure 5(b) shows that suppressive token activation steadily decreases, while reinforced and overall active tokens drop sharply in the third epoch, suggesting that fewer teacher signals remain informa-

tive as the student approaches the teacher. Finally, Figure 5(c) reports task-wise student accuracy.   
Nearly all task categories exhibit consistent improvements across training.

## 7 CONCLUSION

In this work, we present RS-OPSD, a reliable privileged OPSD framework for UHR remote sensing VQA. Instead of relying on additional visual search or tool use at inference time, RS-OPSD internalizes zoom-in visual privilege into the model during training. To support this paradigm, we construct GeoEvidence-6K with explicit evidence-region annotations and introduce HF-SR for human-feedback-guided data construction. We further develop CPVP to provide richer contextual and fine-grained visual privilege, and CAD to suppress unreliable teacher guidance. Extensive experiments demonstrate that RS-OPSD achieves SOTA performance while maintaining efficient inference, and that RS-OPD-Lite further offers a favorable accuracy–efficiency trade-off with a sub stantially smaller student model. We further discuss the limitations of the current framework and potential directions for future work in Appendix E.

## REFERENCES

Claude 3.7 sonnet system card. URL https://api.semanticscholar.org/CorpusID: 276612236.

Shuai Bai, Yuxuan Cai, Ruizhe Chen, Keqin Chen, Xionghui Chen, Zesen Cheng, Lianghao Deng, Wei Ding, Chang Gao, Chunjiang Ge, et al. Qwen3-vl technical report. arXiv preprint arXiv:2511.21631, 2025a.

Shuai Bai, Keqin Chen, Xuejing Liu, Jialin Wang, Wenbin Ge, Sibo Song, Kai Dang, Peng Wang, Shijie Wang, Jun Tang, Humen Zhong, Yuanzhi Zhu, Mingkun Yang, Zhaohai Li, Jianqiang Wan, Pengfei Wang, Wei Ding, Zheren Fu, Yiheng Xu, Jiabo Ye, Xi Zhang, Tianbao Xie, Zesen Cheng, Hang Zhang, Zhibo Yang, Haiyang Xu, and Junyang Lin. Qwen2.5-vl technical report. arXiv preprint arXiv:2502.13923, 2025b.

Beeldmateriaal Nederland. High-resolution aerial imagery of the netherlands. https://www. beeldmateriaal.nl/dataroom, 2025. Accessed September 5, 2026.

Javiera Castillo-Navarro, Bertrand Le Saux, Alexandre Boulch, Nicolas Audebert, and Sebastien´ Lefevre. Semi-supervised semantic segmentation in earth observation: The MiniFrance suite,\` dataset analysis and multi-task network study. Machine Learning, 111(9):3125–3160, 2022. doi: 10.1007/s10994-020-05943-y.

Gheorghe Comanici, Eric Bieber, Mike Schaekermann, Ice Pasupat, Noveen Sachdeva, Inderjit Dhillon, Marcel Blistein, Ori Ram, Dan Zhang, Evan Rosen, et al. Gemini 2.5: Pushing the frontier with advanced reasoning, multimodality, long context, and next generation agentic capabilities. arXiv preprint arXiv:2507.06261, 2025.

Yunkai Dang, Minxin Dai, Yuekun Yang, Zhangnan Li, Wenbin Li, Feng Miao, and Yang Gao. Uhrbat: Budget-aware token compression vision-language model for ultra-high-resolution remote sensing. In Forty-third International Conference on Machine Learning, 2026.

Rodrigo Caye Daudt, Bertrand Le Saux, Alexandre Boulch, and Yann Gousseau. Multitask learning for large-scale semantic change detection. Computer Vision and Image Understanding, 187: 102783, 2019. doi: 10.1016/j.cviu.2019.07.003.

Federal Office of Topography swisstopo. SWISSIMAGE 10 cm: The digital color orthophotomosaic of switzerland. https://www.swisstopo.admin.ch/en/ orthoimage-swissimage-10, 2024. Accessed September 5, 2026.

Geobasis NRW. Digital orthophotos of north rhine-westphalia (DOP). https://registry. gdi-de.org/id/de.nw/DOP, 2025. Accessed September 5, 2026.

Edward J Hu, Yelong Shen, Phillip Wallis, Zeyuan Allen-Zhu, Yuanzhi Li, Shean Wang, Lu Wang, and Weizhu Chen. Lora: Low-rank adaptation of large language models. arXiv preprint arXiv:2106.09685, 2021.

Aaron Hurst, Adam Lerer, Adam P Goucher, Adam Perelman, Aditya Ramesh, Aidan Clark, AJ Ostrow, Akila Welihinda, Alan Hayes, Alec Radford, et al. Gpt-4o system card. arXiv preprint arXiv:2410.21276, 2024.

Kartik Kuckreja, Muhammad Sohail Danish, Muzammal Naseer, Abhijit Das, Salman Khan, and Fahad Shahbaz Khan. Geochat: Grounded large vision-language model for remote sensing. In Proceedings ofthe IEEE/CVF conference on computer vision andpattern recognition, pp. 27831– 27840, 2024.

Woosuk Kwon, Zhuohan Li, Siyuan Zhuang, Ying Sheng, Lianmin Zheng, Cody Hao Yu, Joseph E. Gonzalez, Hao Zhang, and Ion Stoica. Efficient memory management for large language model serving with pagedattention. In Proceedings of the ACM SIGOPS 29th Symposium on Operating Systems Principles, 2023.

Bo Li, Yuanhan Zhang, Dong Guo, Renrui Zhang, Feng Li, Hao Zhang, Kaichen Zhang, Peiyuan Zhang, Yanwei Li, Ziwei Liu, et al. Llava-onevision: Easy visual task transfer. arXiv preprint arXiv:2408.03326, 2024.

Ruixun Liu, Bowen Fu, Jiayi Song, Kaiyu Li, Wanchen Li, Lanxuan Xue, Hui Qiao, Weizhan Zhang, Deyu Meng, and Xiangyong Cao. Zoomearth: Active perception for ultra-high-resolution geospatial vision-language tasks. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2026.

Junwei Luo, Yingying Zhang, Xue Yang, Kang Wu, Qi Zhu, Lei Liang, Jingdong Chen, and Yansheng Li. When large vision-language model meets large remote sensing imagery: Coarse-to-fine text-guided token pruning. In Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV), pp. 9206–9217, October 2025.

Xianzhi Ma, Shujun Wang, Xiaohan Li, Hao Liu, Changhua Pei, et al. Weaveearth: Structured evidence construction and reasoning for training-free uhr remote sensing understanding. arXiv preprint arXiv:2607.10120, 2026.

Chao Pang, Xingxing Weng, Jiang Wu, Jiayu Li, Yi Liu, Jiaxing Sun, Weijia Li, Shuai Wang, Litong Feng, Gui-Song Xia, et al. Vhm: Versatile and honest vision language model for remote sensing image analysis. In Proceedings of the AAAI Conference on Artificial Intelligence, volume 39, pp. 6381–6388, 2025.

Guangming Sheng, Chi Zhang, Zilingfeng Ye, Xibin Wu, Wang Zhang, Ru Zhang, Yanghua Peng, Haibin Lin, and Chuan Wu. Hybridflow: A flexible and efficient rlhf framework. arXiv preprint arXiv: 2409.19256, 2024.

Fengxiang Wang, Mingshuo Chen, Yueying Li, Di Wang, Haotian Wang, Zonghao Guo, Zefan Wang, Boqi Shan, Long Lan, Yulin Wang, Hongzhen Wang, Wenjing Yang, Bo Du, and Jing Zhang. Geollava-8k: Scaling remote-sensing multimodal large language models to 8k resolution. In Advances in Neural Information Processing Systems(NeurIPS), 2025a.

Fengxiang Wang, Hongzhen Wang, Zonghao Guo, Di Wang, Yulin Wang, Mingshuo Chen, Qiang Ma, Long Lan, Wenjing Yang, Jing Zhang, et al. Xlrs-bench: Could your multimodal llms understand extremely large ultra-high-resolution remote sensing imagery? In Proceedings of the Computer Vision and Pattern Recognition Conference, pp. 14325–14336, 2025b.

Fengxiang Wang, Mingshuo Chen, Yueying Li, Yajie Yang, Yifan Zhang, Long Lan, Xue Yang, Hongda Sun, Yulin Wang, Di Wang, et al. Geoeyes: On-demand visual focusing for evidence-grounded understanding of ultra-high-resolution remote sensing imagery. arXiv preprint arXiv:2602.14201, 2026.

Qianhao Yuan, Jie Lou, Xing Yu, Hongyu Lin, Le Sun, Xianpei Han, and Yaojie Lu. Vision-opd: Learning to see fine details for multimodal llms via on-policy self-distillation. arXiv preprint arXiv:2605.18740, 2026.

Yi-Fan Zhang, Huanyu Zhang, Haochen Tian, Chaoyou Fu, Shuangqing Zhang, Junfei Wu, Feng Li, Kun Wang, Qingsong Wen, Zhang Zhang, et al. Mme-realworld: Could your multimodal llm challenge high-resolution real-world scenarios that are difficult for humans? In Proceedings of the International Conference on Learning Representations (ICLR), 2025.

Siyan Zhao, Zhihui Xie, Mengchen Liu, Jing Huang, Guan Pang, Feiyu Chen, and Aditya Grover. Self-distilled reasoner: On-policy self-distillation for large language models. arXiv preprint arXiv:2601.18734, 2026.

Yunqi Zhou, Chengjie Jiang, Chun Yuan, and Jing Li. Look where it matters: Training-free ultra-hr remote sensing vqa via adaptive zoom search. arXiv preprint arXiv:2511.20460, 2025.

Jinguo Zhu, Weiyun Wang, Zhe Chen, Zhaoyang Liu, Shenglong Ye, Lixin Gu, Hao Tian, Yuchen Duan, Weijie Su, Jie Shao, et al. Internvl3: Exploring advanced training and test-time recipes for open-source multimodal models. arXiv preprint arXiv:2504.10479, 2025.

## A ALGORITHM OF RS-OPSD

Algorithm 1 summarizes the training procedure of RS-OPSD. For each sample, the student generates an on-policy response from the Global Evidence, while the privileged teacher evaluates the same trajectory using CPVP. CAD then filters privileged supervision according to teacher reliability and student-answer correctness. The student is optimized with the CAD objective and reference-model regularization, while the teacher is updated by EMA.

Algorithm 1 Training procedure of RS-OPSD   
Require: Training set $\mathcal { D } = \{ ( I _ { G } , I _ { C } , I _ { F } , q , a ^ { * } ) \} ;$ student $\pi _ { \boldsymbol { \theta } } ;$ teacher $\pi _ { \boldsymbol { \theta } _ { T } } ;$ frozen reference $\pi _ { \mathrm { r e f } } ;$   
KL coefficient $\lambda _ { \mathrm { K L } } ;$ gradient clipping threshold $c _ { \mathrm { g r a d } } ;$ EMA coefficient $\beta$   
1: Initialize $\theta _ { T }  \theta$   
2: Freeze $\pi _ { \mathrm { r e f } }$   
3: for each training batch $B \subset D$ do   
4: for each $( I _ { G } , I _ { C } , I _ { F } , q , a ^ { * } ) \in \mathcal { B }$ do   
5: Construct privileged information $z \gets \{ I _ { C } , I _ { F } \}$   
6: Sample one student trajectory $\mathbf { y } \sim \pi _ { \boldsymbol { \theta } } ( \cdot \mid I _ { G } , q )$   
7: Construct the valid-answer mask $m ^ { \mathrm { a n s } }$ from y   
8: Compute student token probabilities $p _ { t } ^ { S } \gets \bar { \pi _ { \theta } } ( y _ { t } \mid I _ { G } , q , \mathbf { y } _ { < t } )$   
9: Compute privileged-teacher probabilities $p _ { t } ^ { T } \gets \pi _ { \theta _ { T } } ( y _ { t } \mid I _ { G } , z , q , \mathbf { y } _ { < t } )$   
10: Run the teacher-forced GT-prefix reliability probe: $\mathbf { \bar { \rho } } g \gets \mathrm { P r o b e } _ { T } ( a ^ { * } ) \in \{ 0 , 1 \}$   
11: Verify the student response: $R \gets + 1 \mathrm { i f } \hat { a } = a ^ { * }$ , otherwise $R \gets - 1$   
12: for each valid answer token $y _ { t }$ do   
13: $r _ { t } \gets \mathrm { s g } [ \log p _ { t } ^ { T } - \log p _ { t } ^ { S } ]$   
14: $A _ { t } \gets g R [ R r _ { t } ] _ { + }$   
15: end for   
16: end for   
17: Compute CAD loss   
${ \mathcal { L } } _ { \mathrm { C A D } } = - { \frac { \sum _ { i , t } m _ { i , t } ^ { \mathrm { a n s } } \operatorname { s g } [ A _ { i , t } ] \log p _ { i , t } ^ { S } } { \sum _ { i , t } m _ { i , t } ^ { \mathrm { a n s } } } }$   
18: Compute reference-model regularization ${ \mathcal { L } } _ { \mathrm { K L } }$   
19: $\mathcal { L }  \mathcal { L } _ { \mathrm { C A D } } + \lambda _ { \mathrm { K L } } \mathcal { L } _ { \mathrm { K L } }$   
20: Backpropagate $\nabla _ { \boldsymbol { \theta } } \mathcal { L }$   
21: Clip the student gradient norm with threshold $c _ { \mathrm { g r a d } }$   
22: Update student parameters θ   
23: Update teacher by EMA: $\theta _ { T }  \beta \theta _ { T } + ( 1 - \beta ) \theta$   
24: end for

For RS-OPD-Lite, the same training procedure is used with a Qwen3-VL-2B student and a Qwen3- VL-8B privileged teacher. The teacher remains frozen throughout training.

## B DATASET QUALITY AND INTEGRITY

We verified that GeoEvidence-6K has no exact image-level overlap with XLRS-Bench (Wang et al., 2025b), MME-RealWorld-RS (Zhang et al., 2025), or LRS-VQA (Luo et al., 2025), ruling out direct train–test leakage from reused evaluation images. The annotation process involved ten domain experts with expertise in remote sensing image interpretation. All generated samples were manually reviewed, and the complete dataset underwent a final cross-validation stage across the expert pool to resolve remaining ambiguities and annotation errors. We measure annotation reliability using the Cross-Validation Acceptance Rate (CVAR), defined as the proportion of samples accepted during the final cross-validation without requiring further correction. GeoEvidence-6K achieves an overall CVAR of 97.8%, indicating a high level of consistency in the final annotations.

![](images/cb1e5eb51d7b0d03e88209a4957036e872e9fed4552e42d5b1525bce69d47396.jpg)  
Figure 6: Dataset statistics of GeoEvidence-6K. Left: word cloud of the question texts. Middle: token-length distribution of the questions, with the mean, median, and 95th percentile marked by dashed lines. Right: distribution of task categories.

Figure 6 further summarizes several statistics of GeoEvidence-6K, including the question vocabulary, token-length distribution, and task distribution.

As shown in Figure 6(left), the most frequent words include terms such as roof, parcel, road, color, shape, center, and relative-location expressions such as upper-right, beside, and near. This indicates that GeoEvidence-6K covers diverse UHR remote sensing VQA patterns, including attribute recognition, object counting, spatial localization, and fine-grained geometric reasoning.

Figure 6(middle) shows that the token-length distribution is concentrated in a moderate range, with a mean of 56.87, a median of 54, and a 95th percentile of 84. Most samples therefore remain relatively compact, while a smaller fraction of longer questions reflects the presence of more compositional and spatially specific instructions.

Figure 6(right) shows that the task categories are overall well balanced. Five local-evidence tasks— counting, position, category, color, and shape—each account for about 15.6% of the dataset, while the two global-context tasks—route planning and land use—each account for about 11.1%. This composition provides balanced supervision across both local fine-grained perception and global scene-level understanding.

## C IMPLEMENTATION DETAILS

Training framework. We implement RS-OPSD using the verl (Sheng et al., 2024) training framework. For each training sample, the student performs a single on-policy rollout (n = 1), which is then reused for privileged-teacher evaluation, CAD computation, and reference-model regularization.

Visual preprocessing. During training, all visual inputs are resized while preserving the original aspect ratio, with the longest image side capped at 2,048 pixels. The same preprocessing strategy is applied to the Global Evidence, Contextual Evidence, and Fine-grained Evidence. For training samples with localized evidence annotations, the student Global Evidence contains a rendered red bounding box indicating the annotated evidence region, while the privileged teacher additionally receives the corresponding contextual and fine-grained views. At inference time, we do not introduce any additional bounding boxes, crop coordinates, or localization cues beyond those originally provided by the benchmark. Specifically, MME-RealWorld-RS and LRS-VQA are evaluated using their original unmarked images, whereas a subset of XLRS-Bench samples natively contains red-circle annotations in the released benchmark images.

![](images/54d4fac5eb6fb3f22c21f76f21d53bac1236e21e819463ceebd1980aa711544f.jpg)  
Figure 7: Prompt templates for the student and privileged teacher in RS-OPSD.

Training prompts. To ensure reproducibility, Figure 7 summarizes the prompt templates used during RS-OPSD training. The student receives a single full-image input together with the original question, options, and bounding-box instruction when applicable. In contrast, the privileged teacher receives three complementary views—Global Evidence, Contextual Evidence, and Finegrained Evidence—with an explicit instruction describing the role of each view. The question and answer options are otherwise kept consistent between the student and teacher prompts. All prompts are wrapped with the Qwen chat template before being passed to the model.

Inference efficiency. Inference latency is measured on a single NVIDIA H100 GPU with a batch size of 1. We report the average per-sample latency under the same evaluation protocol used for the three UHR remote sensing VQA benchmarks. Since RS-OPSD introduces no architectural modifications to the underlying vision-language model and requires neither additional visual search nor external tool calls at inference time, the trained model remains directly compatible with standard inference acceleration frameworks such as vLLM (Kwon et al., 2023), without requiring specialized serving infrastructure.

## D DETAILED QUANTITATIVE COMPARISON RESULTS

Tables 5 and 6 provide fine-grained quantitative comparison results across the three UHR remote sensing VQA benchmarks.

XLRS-Bench (Wang et al., 2025b). XLRS-Bench evaluates eight perception categories—Overall Counting (OC), Regional Counting (RC), Overall Land Use Classification (OLUC), Regional Land Use Classification (RLUC), Object Classification (OCC), Object Color (OCL), Object Motion State (OMS), and Object Spatial Relationship (OSR)—and five reasoning categories, including Anomaly Detection (AD), Environmental Conditional Reasoning (ECR), Route Planning (RP), Regional Counting with Change Detection (RCCD), and Counting with Complex Reasoning (CCR). As shown in Table 5, RS-OPSD achieves the best overall score of 53.1, exceeding the previous best GeoLLaVA-8K (Wang et al., 2025a) by 1.6 points. It ranks first on RC, RLUC, OCC, ECR, and

Table 5: Detailed quantitative results on XLRS-Bench across perception and reasoning sub-tasks. Green, yellow, and blue denote the first-, second-, and third-best results, respectively.
<table><tr><td rowspan="2">Method Sub-tasks</td><td colspan="8">Perception</td><td rowspan="2">Reasoning</td><td colspan="4"></td><td rowspan="2">Avg.</td></tr><tr><td>OC</td><td></td><td>RC OLUC</td><td>RLUC</td><td>OCC</td><td>OCL</td><td>OMS</td><td>OSR</td><td>AD ECR</td><td>RP</td><td>RCCD</td><td>CCR</td></tr><tr><td></td><td colspan="14">Closed-Source Vision Language Models</td></tr><tr><td>GPT-40</td><td>25.0</td><td>32.0</td><td>15.0</td><td>66.0</td><td>9.5</td><td>11.3</td><td>11.7</td><td>24.6</td><td>73.0</td><td>73.0</td><td>35.0</td><td>20.0</td><td>25.0</td><td>32.4</td></tr><tr><td>Claude 3.7 Sonnet</td><td>27.6</td><td>22.7</td><td>17.4</td><td>68.4</td><td>30.5</td><td>29.9</td><td>63.6</td><td>27.6</td><td>64.8</td><td>78.4</td><td>34.5</td><td>27.8</td><td>32.6</td><td>40.5</td></tr><tr><td>Gemini 2.5 Pro</td><td>50.0</td><td>42.0</td><td>12.0</td><td>65.5</td><td>37.5</td><td>38.1</td><td>60.0</td><td>30.2</td><td>68.0</td><td>72.0</td><td>36.0</td><td>26.7</td><td>35.0</td><td>44.1</td></tr><tr><td colspan="14">Open-Source Vision Language Models</td></tr><tr><td>LLaVA-OV-7B</td><td>25.0</td><td>38.0</td><td>8.0</td><td>69.5</td><td>35.9</td><td>35.3</td><td>65.0</td><td>25.2</td><td>76.0</td><td>83.0</td><td>24.0</td><td>43.3</td><td>36.0</td><td>43.4</td></tr><tr><td>Qwen2.5-VL-7B</td><td>33.3</td><td>40.0</td><td>31.0</td><td>77.0</td><td>40.6</td><td>40.5</td><td>66.7</td><td>36.2</td><td>68.0</td><td>72.0</td><td>27.0</td><td>38.3</td><td>45.0</td><td>47.4</td></tr><tr><td>Qwen3-VL-8B</td><td>23.3</td><td>45.0</td><td>20.0</td><td>82.0</td><td>45.9</td><td>44.4</td><td>66.7</td><td>30.6</td><td>74.0</td><td>79.0</td><td>42.0</td><td>48.3</td><td>55.0</td><td>50.5</td></tr><tr><td>InternVL3-8B</td><td>40.0</td><td>39.0</td><td>10.0</td><td>71.5</td><td>44.5</td><td>30.8</td><td>65.0</td><td>25.2</td><td>77.0</td><td>82.0</td><td>36.0</td><td>21.7</td><td>50.0</td><td>45.6</td></tr><tr><td colspan="14">Remote Sensing Vision Language Models</td><td></td></tr><tr><td>GeoChat</td><td>16.7</td><td>29.0</td><td>2.0</td><td>23.0</td><td>21.1</td><td>16.8</td><td>35.0</td><td>24.2</td><td>33.0</td><td>43.0</td><td>10.0</td><td></td><td></td><td>22.9</td></tr><tr><td>VHM</td><td>18.3</td><td>33.0</td><td>5.0</td><td>38.5</td><td>27.5</td><td>25.4</td><td>35.0</td><td>33.0</td><td>53.0</td><td>58.0</td><td>39.0</td><td>26.7</td><td>21.0 34.0</td><td>32.8</td></tr><tr><td colspan="14">Specialized Methods for UHR Remote Sensing VQA</td><td></td></tr><tr><td>GeoLLaVA-8K</td><td></td><td>26.7 38.0</td><td>49.0</td><td>69.0</td><td>41.6</td><td>31.6</td><td>65.0</td><td>35.0</td><td>67.0</td><td>78.0</td><td>66.0</td><td>50.0</td><td>52.0</td><td>51.5</td></tr><tr><td>UHR-BAT</td><td>21.7</td><td>33.0</td><td>50.0</td><td>55.5</td><td>43.5</td><td>33.8</td><td>65.0</td><td>44.8</td><td>62.0</td><td>71.0</td><td>54.0</td><td></td><td></td><td>48.6</td></tr><tr><td>ZoomSearch</td><td>46.7</td><td>42.0</td><td>12.0</td><td>70.5</td><td>44.5</td><td>35.8</td><td>75.0</td><td>32.2</td><td>74.0</td><td></td><td></td><td>46.7</td><td>51.0</td><td></td></tr><tr><td>WeaveEarth</td><td>20.0</td><td>47.0</td><td>24.0</td><td>79.0</td><td>43.1</td><td>38.0</td><td>66.7</td><td>26.8</td><td>69.0</td><td>82.0</td><td>29.0</td><td>43.3 48.3</td><td>37.0</td><td>48.0</td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td>81.0</td><td>46.0</td><td></td><td>55.0</td><td>49.5</td></tr><tr><td>ZoomEarth</td><td>18.3</td><td>35.0</td><td>30.0</td><td>61.5</td><td>39.2</td><td>32.4 Ours</td><td>31.7</td><td>20.8</td><td>70.0</td><td>75.0</td><td>38.0</td><td>28.3</td><td>30.0</td><td>39.2</td></tr><tr><td colspan="14"></td><td></td><td></td><td></td></tr><tr><td>RS-OPD-Lite RS-OPSD</td><td>25.0</td><td>45.0</td><td>30.0</td><td>72.0</td><td>31.1</td><td>36.8</td><td>63.3</td><td>28.0</td><td>74.0</td><td>83.0</td><td>20.0</td><td>36.7</td><td>50.0</td><td>45.8</td></tr><tr><td></td><td>28.3</td><td>55.0</td><td>33.0</td><td>85.0</td><td>46.8</td><td>41.9</td><td>71.7</td><td>29.2</td><td>74.0</td><td>84.0</td><td>45.0</td><td>51.7</td><td>45.0</td><td>53.1</td></tr></table>

RCCD, while obtaining the second-best results on OCL and OMS. The improvements span both perception and reasoning tasks, indicating that the gains are not concentrated in a single capability.

MME-RealWorld-RS (Zhang et al., 2025) and LRS-VQA (Luo et al., 2025). As shown in Table 6, RS-OPSD achieves 61.5 on MME-RealWorld-RS, outperforming the previous best Zoom-Search (Zhou et al., 2025) by 3.9 points. In particular, it achieves the best results on Position (76.8) and Color (70.4), while remaining competitive on Count. On LRS-VQA, RS-OPSD reaches the best average score of 33.3, improving over WeaveEarth (Ma et al., 2026) by 2.0 points. Although it does not dominate every individual subset, it achieves consistently strong results across FAIR, Bridge, and STAR, leading to the highest overall average.

Overall accuracy and efficiency. Averaged across the three benchmarks, RS-OPSD reaches 49.3, a 4.0-point improvement over the strongest competing method. Meanwhile, RS-OPD-Lite achieves an average accuracy of 44.2 with only a 2B student, surpassing all evaluated 8B-scale baselines. It also achieves the lowest average inference time at 1.09 s/sample, compared with 1.25 s/sample for the second-fastest method. The full RS-OPSD remains efficient at 1.58 s/sample, even faster than its Qwen3-VL-8B (Bai et al., 2025a) base model at 1.75 s/sample. These results further show that internalizing privileged visual information through OPSD provides a favorable balance between accuracy and inference efficiency.

## E LIMITATIONS AND FUTURE WORK

The current formulation of RS-OPSD relies on explicit supervision at two stages. First, targetregion annotations are required to construct the privileged visual inputs used by CPVP. Second, ground-truth answers are used in CAD to assess teacher reliability and determine the correctness of student rollouts. Although these annotations enable reliable privileged distillation, acquiring them for UHR remote sensing imagery introduces substantial annotation cost and limits the scalability of the framework.

Table 6: Detailed quantitative results on MME-RealWorld-RS, LRS-VQA, and XLRS-Bench. Average inference speed is measured in s/sample.
<table><tr><td rowspan="2">Method Sub-set</td><td colspan="4">MME-RealWorld-RS</td><td colspan="3">LRS-VQA</td><td colspan="2">XLRS-Bench</td><td colspan="2">Total</td></tr><tr><td>Position</td><td>Color</td><td>Count</td><td>Avg.</td><td>FAIR</td><td>Bridge</td><td>STAR</td><td>Avg.</td><td>Avg.</td><td>Accuracy</td><td>Speed</td></tr><tr><td colspan="10">Closed-Source Vision Language Models</td></tr><tr><td>GPT-40</td><td>36.4</td><td>32.4</td><td>15.9</td><td>28.2</td><td>22.2</td><td>31.8</td><td>27.4 27.1</td><td></td><td>32.4</td><td>29.2</td><td></td></tr><tr><td>Claude 3.7 Sonnet</td><td>56.6</td><td>52.1</td><td>28.5</td><td>45.7</td><td>25.7</td><td>29.5</td><td>30.1</td><td>28.4</td><td>40.5</td><td>38.2</td><td>一</td></tr><tr><td>Gemini 2.5 Pro</td><td>60.6</td><td>60.3</td><td>34.3</td><td>51.7</td><td>28.9</td><td>31.2</td><td>32.7</td><td>30.9</td><td>44.1</td><td>42.2</td><td>一</td></tr><tr><td colspan="10">Open-Source Vision Language Models</td></tr><tr><td>LLaVA-OV-7B</td><td>64.8</td><td>61.4</td><td>33.8</td><td>53.3</td><td>20.6</td><td>35.1</td><td>26.1</td><td>27.3</td><td>43.4</td><td>41.3</td><td>2.38</td></tr><tr><td>Qwen2.5-VL-7B</td><td>55.9</td><td>48.5</td><td>18.6</td><td>41.0</td><td>23.9</td><td>34.2</td><td>30.2</td><td>29.4</td><td>47.4</td><td>39.3</td><td>1.87</td></tr><tr><td>Qwen3-VL-8B</td><td>56.4</td><td>48.1</td><td>21.3</td><td>41.9</td><td>24.6</td><td>38.2</td><td>27.5</td><td>30.1</td><td>50.5</td><td>40.8</td><td>1.75</td></tr><tr><td>InternVL3-8B</td><td>71.0</td><td>62.6</td><td>31.4</td><td>55.0</td><td>24.8</td><td>33.4</td><td>29.3</td><td>29.2</td><td>45.6</td><td>43.3</td><td>1.25</td></tr><tr><td colspan="10">Remote Sensing Vision Language Models</td></tr><tr><td>GeoChat</td><td>25.1</td><td>23.1</td><td>15.7</td><td>21.3</td><td>20.2</td><td>24.5</td><td>13.8</td><td>19.5</td><td>22.9</td><td>21.2</td><td>1.33</td></tr><tr><td>VHM</td><td>35.2</td><td>20.3</td><td>16.8</td><td>24.1</td><td>24.3</td><td>27.5</td><td>28.3</td><td>26.7</td><td>32.8</td><td>27.9</td><td>2.64</td></tr><tr><td colspan="10">Specialized Methods for UHR Remote Sensing VQA</td></tr><tr><td>GeoLLaVA-8K</td><td>34.9</td><td>27.9</td><td>22.3</td><td>28.4</td><td>21.8</td><td>29.2</td><td>26.4</td><td>25.8</td><td>51.5</td><td>35.2</td><td>1.30</td></tr><tr><td>UHR-BAT</td><td>44.0</td><td>42.0</td><td>14.0</td><td>33.3</td><td>16.6</td><td>23.5</td><td>20.0</td><td>20.0</td><td>48.6</td><td>34.0</td><td>7.40</td></tr><tr><td>ZoomSearch</td><td>67.6</td><td>66.1</td><td>39.2</td><td>57.6</td><td>25.9</td><td>31.2</td><td>33.5</td><td>30.2</td><td>48.0</td><td>45.3</td><td>43.5</td></tr><tr><td>WeaveEarth</td><td>51.4</td><td>27.7</td><td>62.6</td><td>47.2</td><td>31.0</td><td>26.1</td><td>36.7</td><td>31.3</td><td>49.5</td><td>42.7</td><td>2.94</td></tr><tr><td>ZoomEarth</td><td>46.6</td><td>41.9</td><td>27.6</td><td>38.7</td><td>20.0</td><td>23.2</td><td>21.6</td><td>21.6</td><td>39.2</td><td>33.2</td><td>14.2</td></tr><tr><td colspan="10">Ours</td></tr><tr><td>RS-OPD-Lite</td><td>74.2</td><td>61.9</td><td>32.5</td><td>56.2</td><td>27.9</td><td>29.6</td><td>33.9</td><td>30.5</td><td>45.8</td><td>44.2</td><td>1.09</td></tr><tr><td>RS-OPSD</td><td>76.8</td><td>70.4</td><td>37.4</td><td>61.5</td><td>29.4</td><td>34.9</td><td>35.6</td><td>33.3</td><td>53.1</td><td>49.3</td><td>1.58</td></tr></table>

A natural extension is to explore more self-supervised forms of OPSD that reduce or remove these dependencies. Future work may investigate automatic discovery of question-relevant visual privilege without target-region annotations, as well as teacher reliability estimation without ground-truth answer constraints. More broadly, combining model-intrinsic evidence discovery with self-verification could enable privileged self-distillation to scale to larger unlabeled UHR datasets and broader high resolution vision-language tasks.

Our RS-OPD-Lite experiments further suggest that visual privilege remains beneficial beyond the self-distillation setting, where a larger privileged teacher transfers knowledge to a smaller student. This observation opens another direction for privilege-enhanced model distillation, where additional training-time information can improve knowledge transfer across model scales while retaining a lightweight student at inference time. Exploring such privileged distillation may provide a promising route toward more accurate and efficient vision-language models.
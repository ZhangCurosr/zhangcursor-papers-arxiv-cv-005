# RefineDrive: Reliable Failure-Guided Learning for Vision-Language-Action Driving

Zhe Sun<sup>1</sup>, Ziyi Luo<sup>1</sup>, Yehao Lu<sup>1</sup>, Lei Zhou<sup>2</sup>, and Xi Li<sup>1\*</sup>

<sup>1</sup>College of Computer Science and Technology, Zhejiang University, Hangzhou, China <sup>2</sup>Yinwang Intelligent Technology Co., Ltd.

## Abstract

Vision-Language-Action (VLA) models for autonomous driving rely heavily on successful expert demonstrations, leaving model-specific failures underexploited. Learning from these failures is hindered by unreliable diagnoses, poorly matched correction targets, and coarse rewards. We propose RefineDrive, a failure-guided post-training framework that learns from self-generated failures through targeted supervision and safety-aware reinforcement learning. Reliable Diagnosis derives structured, verifiable feedback on collisions and drivable-area violations directly from simulator states. Minimum-Correction Target Retrieval searches a clustered human trajectory bank for nearby corrections that satisfy hard-safety constraints in the current scene, prioritizing preservation of the failed prediction’s motion pattern. Conditioned on the driving context and failed trajectory, Correction SFT learns to generate the diagnosis followed by the retrieved correction as a training-only auxiliary task. We then apply GRPO with a Safety-Layered Reward that strictly prioritizes hard-safe trajectories, retains continuous safety feedback for both unsafe and hard-safe trajectories, and rewards driving progress only after hard safety is satisfied. At inference, the policy directly predicts trajectories from the driving context without an explicit diagnosis or repair stage. On NAVSIM v1, RefineDrive improves the 4B base SFT policy from 87.7 to 91.7 PDMS. Using the same checkpoint without additional training, RefineDrive achieves 89.4 EPDMS on the original NAVTEST scenes evaluated with NAVSIM v2 extended metrics. Controlled ab lations support the benefits of structured diagnosis supervision, retrieved corrections, and safety-layered optimization for direct planning.

## 1 Introduction

Vision-Language-Action (VLA) models have shown strong potential for end-to-end autonomous driving, benefiting from large-scale vision-language pretraining and expert demonstrations. However, most approaches still rely primarily on imitation learning from successful human behavior, while failures exposed by the model’s own rollouts—such as collisions and departures from the drivable area—remain underexploited. For a policy with strong basic driving capabilities, further improvement may depend less on imitating more expert behavior and more on identifying and correcting the remaining weaknesses revealed by its own failures.

Recent studies have begun to explore learning from failures, but three key limitations remain, as illus trated in Fig. 1. First, failure diagnosis can be unreliable. Methods that rely on large VLMs or teacher models to explain failures may produce open-ended diagnoses that are difficult to verify, with inaccurate event localization, incorrect causal attribution, or inconsistency with the simulated outcome. Second, the correction target may be poorly matched to the failure. Since a failed trajectory is often only partially incorrect, directly replacing it with human ground truth can introduce unnecessary behavioral changes and lead to over-correction. Third, coarse reward signals provide limited optimization guidance. An aggregated driving score may insufficiently distinguish the extent and timing of safety violations among unsafe trajectories or safety margins among safe ones, limiting fine-grained safety optimization.

![](images/baf9856c5f0ae18f723f872581e40ec6bb4ee87122c6e0ba2502a859f2e9884a.jpg)  
Figure 1: Motivation and overview of RefineDrive. Existing failure-learning methods can suffer from unreliable VLM-based diagnoses, over-corrective human ground-truth supervision, and coarse aggregated rewards. We address these limitations with Reliable Diagnosis, which grounds structured feedback in NAVSIM PDM simulation; Minimum-Correction Target Retrieval, which searches a clustered human trajectory bank for safe corrections close to failed predictions; and Safety-Layered Reward, which provides hierarchical signals over hard safety, continuous safety, and driving progress.

To address these limitations, we propose RefineDrive, a failure-guided post-training approach with three components. Reliable Diagnosis derives structured, simulator-verifiable feedback for collisions and drivable-area violations. Minimum-Correction Target Retrieval searches a clustered human trajectory bank for a nearby correction that satisfies hard-safety constraints in the current scene, prioritizing the motion pattern of the failed prediction. During training, Correction SFT conditions on the driving context and a failed trajectory to learn structured diagnosis and correction. At inference, RefineDrive directly predicts the final trajectory from the driving context, without an explicit diagnosis or repair stage.

Finally, Safety-Layered GRPO combines hard safety, continuous safety, and driving progress. It strictly prioritizes hard-safe trajectories, retains continuous safety feedback for both unsafe and hard-safe trajectories, and rewards progress only after hard safety is satisfied.

RefineDrive uses self-generated rollouts to expose residual weaknesses, derives simulator-verifiable diagnoses, retrieves minimally modified safe corrective targets, and further refines the policy with safetylayered reinforcement learning. Rather than repeatedly learning the full distribution of expert behavior, RefineDrive focuses post-training on the specific deficiencies currently exhibited by the policy.

Our main contributions are summarized as follows:

• We introduce RefineDrive, which explicitly exploits failures from the model’s own rollouts to address residual weaknesses in autonomous driving VLAs.

• We propose Reliable Diagnosis and Minimum-Correction Target Retrieval. Simulator-derived structured diagnoses provide reproducible and verifiable failure feedback, while cluster-based retrieval selects safe corrective trajectories close to failed predictions, reducing unnecessary behavioral changes and over-correction.

• We develop a Safety-Layered Reward that prioritizes hard safety, provides continuous safety feedback for both unsafe and hard-safe trajectories, and rewards driving progress only after hard safety is satisfied. Experiments on NAVSIM demonstrate the effectiveness of RefineDrive.

## 2 Related Work

## 2.1 Vision-Language-Action Models for Autonomous Driving

Vision-language models (VLMs) have been applied to autonomous driving for scene understanding, reasoning, and decision making [1, 2, 3, 4, 5]. DriveVLM [6] emphasizes language-space reasoning for hierarchical planning, whereas VLA models more tightly integrate multimodal reasoning and action generation. OpenDriveVLA [7] aligns instance-aware visual and language representations for autoregressive actions, while AutoVLA [8] combines semantic reasoning, discretized action tokens, and reinforcement fine-tuning. ReCogDrive [9] integrates a cognitive VLM with a diffusion planner, and Qwen-Drive [10] explores unified representations for perception, understanding, and planning.

ReflectDrive [11] introduces discrete-diffusion planning with inference-time safety reflection: a safety scorer identifies unsafe waypoints, local search finds safe anchors, and inpainting regenerates the surrounding trajectory. Although both methods repair unsafe predictions, ReflectDrive performs waypoint-level inference editing, whereas we retrieve complete, scene-validated trajectories and pair them with structured diagnoses as offline Correction SFT supervision.

## 2.2 Learning from Failures

Learning from failures complements imitation from successful demonstrations. ChauffeurNet [12] perturbs expert trajectories to synthesize unsafe situations. Recent VLA methods exploit failures more directly: ELF-VLA [13] uses teacher-generated diagnoses for trajectory refinement, SafeAlign-VLA [14] constructs failure descriptions and counterfactual positives for supervised and reinforcement learning, and FIRE-VLA [15] employs failure-triggered self-distillation with privileged future information. Rather than relying on openended teacher explanations, we derive structured diagnoses directly from simulator states, grounding violation times, involved objects, and boundary-exit directions in verifiable geometric and kinematic evidence.

The concurrent R<sup>2</sup>LPL framework [16] retrieves feasible trajectory anchors from recoverable closedloop states and updates an anchor-scoring planner through replay-based lifelong learning. Its target scores combine rule-based planning quality with route and expert consistency using state-dependent weights. Both methods construct corrective supervision from policy failures, but differ in the correction setting and targetselection criterion. We formulate our correction stage as trajectory-conditioned behavior repair as a trainingonly auxiliary task: the failed prediction serves as both the reference for retrieving a nearby, scene-verified safe trajectory from a clustered human trajectory bank and an explicit conditioning input for Correction SFT.

Rather than supervising anchor scores, we train the VLA to autoregressively generate a simulator-grounded diagnosis followed by the corrected trajectory. Safety-Layered GRPO subsequently refines the policy.

![](images/39c0ccc818a92730a0e9a01a15b1cf5f14e7d075681ad9eb9b81dbc92314e5ef.jpg)  
Figure 2: Training pipeline of RefineDrive. Offline rollouts and simulator-based safety checks expose model-specific failures. For each failure, we construct a structured diagnosis and retrieve a nearby safe correction from a human trajectory bank. Conditioned on the driving context and failed trajectory, Correction SFT learns to generate the diagnosis followed by the correction. Finally, GRPO refines the policy with the Safety-Layered Reward, providing fine-grained feedback beyond aggregated driving scores. Diagnosis and correction are training-only tasks; inference directly maps the driving context to the final trajectory.

## 3 Method

## 3.1 Overview

We propose RefineDrive that enables an already capable driving VLA to improve from safety-critical failures exposed by its own behavior. As illustrated in Fig. 2, our framework consists of four stages: (1) Failure Discovery, (2) Structured Failure Correction, (3) Correction SFT, and (4) Safety-Layered Reinforcement Learning.

The resulting framework progressively improves the policy by focusing learning on weaknesses revealed by its own rollouts rather than repeatedly imitating already-mastered expert behavior.

## 3.2 Failure Discovery

We start from a supervised fine-tuned driving policy and perform offline rollouts on the training scenarios. Each predicted trajectory is re-executed using the NAVSIM PDM Simulator to obtain reproducible safety

outcomes.

We focus on two directly observable safety-critical failures: at-fault collisions and departures from the drivable area. Accordingly, we construct the model-specific failure set as

$$
\begin{array} { r } { \mathcal { D } _ { \mathrm { f a i l } } = \left\{ ( q , \tau ^ { - } ) \ : \middle | \ : \mathrm { N C } ( \tau ^ { - } ) = 0 \ : \vee \ : \mathrm { D A C } ( \tau ^ { - } ) = 0 \right\} , } \end{array}\tag{1}
$$

where $q$ denotes the driving context, $\tau ^ { - }$ is the trajectory predicted by the current policy, and NC and DAC denote No At-Fault Collisions and Drivable Area Compliance, respectively.

Unlike artificially synthesized negative samples, the trajectories in $\mathcal { D } _ { \mathrm { f a i l } }$ expose failure modes actually produced by the current policy. They therefore provide targeted cases for subsequent diagnosis and correction.

## 3.3 Structured Failure Correction

For each discovered failure, we construct correction supervision that answers two complementary questions: why does the current prediction fail? and what is the smallest behavioral change that makes it safe? We address them through verifiable failure diagnosis and counterfactual minimal correction.

Verifiable Failure Diagnosis. Rather than relying on a teacher VLM to produce open-ended explanations, we derive failure feedback directly from simulator states.

For an at-fault collision, we identify the first ego at-fault collision and extract its occurrence time, object category, and relative position with respect to the ego vehicle. The corresponding traffic participant is further projected onto the current front-view image to obtain its 2D bounding box. The resulting feedback explicitly indicates when the collision occurs, which object is involved, and where the object appears in the current visual observation.

For a drivable-area violation, we identify the first time at which the predicted trajectory leaves the valid driving region and determine whether the ego vehicle exits through the left or right boundary.

When the collision object cannot be reliably localized in the current visual observation, we discard the corresponding sample to avoid introducing textual supervision that cannot be grounded in the model input.

The resulting diagnosis is therefore derived from reproducible geometric and kinematic verification rather than subjective model interpretation.

Counterfactual Minimal Correction. For an already capable driving policy, a failed trajectory’s maneuver intention and most of its future motion may remain reasonable, with only a limited deviation causing the safety violation. Direct replacement with the human trajectory may introduce unnecessary behavioral changes; we instead search for a nearby feasible correction from a real human trajectory bank

We construct the bank from the training set and group trajectories by motion pattern using K-Means. Given a failed trajectory $\tau ^ { - }$ , we identify its assigned cluster in the same feature space and search it first to preserve the original motion pattern. If no feasible correction is found, we search the remaining clusters in ascending order of the distance from their centers to the feature representation of $\tau ^ { - }$

Importantly, human trajectories are not regarded as valid correction targets by default. Since trajectories in the bank may originate from different scenes, every candidate is re-evaluated in the current failure scene. We define hard-safety feasibility as

$$
F ( \tau ) = \mathbb { I } [ \mathrm { N C } ( \tau ) = 1 \land \mathrm { D A C } ( \tau ) = 1 \land \mathrm { T T C } ( \tau ) = 1 ] ,\tag{2}
$$

where NC, DAC, and TTC denote no-at-fault collision, drivable-area compliance, and time-to-collision safety, respectively.

To measure deviation from the original prediction, we use the same trajectory representation as K-Means clustering. We unwrap each trajectory’s heading sequence and express it relative to the first waypoint’s heading. Longitudinal positions, lateral positions, and relative headings are normalized using trajectorybank statistics and flattened across all future waypoints into $\phi ( \tau )$ . The trajectory distance is

$$
d ( \tau _ { i } , \tau _ { j } ) = \lVert \phi ( \tau _ { i } ) - \phi ( \tau _ { j } ) \rVert _ { 2 } .\tag{3}
$$

Let $\boldsymbol { B _ { k } }$ denote the set of human trajectories belonging to cluster k. For each searched cluster, we define its feasible candidate set as

$$
S _ { k } = \{ \tau \in { \mathcal { B } } _ { k } \mid F ( \tau ) = 1 \} .\tag{4}
$$

If $\boldsymbol { S _ { k } }$ is non-empty, the closest feasible trajectory within that cluster is selected as

$$
\tau _ { k } ^ { * } = \arg \operatorname* { m i n } _ { \tau \in S _ { k } } d ( \tau , \tau ^ { - } ) .\tag{5}
$$

The final correction target $\tau ^ { * }$ is given by $\tau _ { k } ^ { * }$ from the first searched cluster that contains at least one feasible candidate. In practice, candidates within each cluster are evaluated in ascending order of $d ( \tau , \tau ^ { - } )$ , so the search terminates as soon as the first safe candidate is found.

If no feasible trajectory is found after all available candidates have been considered, or if the predefined search budget is reached before finding one, no corrected trajectory is assigned to the corresponding failure case.

## 3.4 Correction SFT

We use Correction SFT as a training-only auxiliary task that exposes the policy to self-generated failures, their simulator-grounded diagnoses, and corresponding safe corrections.

For each training sample, the input contains the driving context $q$ and the failed trajectory $\tau ^ { - }$ . The target response $y ^ { * }$ consists of the structured diagnosis followed by the retrieved corrective trajectory $\tau ^ { * }$ . We optimize the policy using the standard autoregressive objective:

$$
\mathcal { L } _ { \mathrm { c o r r } } = - \sum _ { t = 1 } ^ { | y ^ { * } | } \log \pi _ { \theta } \left( y _ { t } ^ { * } \mid q , \tau ^ { - } , y _ { < t } ^ { * } \right) .\tag{6}
$$

The purpose of this task is to improve direct planning through failure-specific supervision, rather than to define an inference-time repair procedure. At inference, the policy directly generates the final trajectory conditioned only on the driving context q. It is not given a failed trajectory and does not perform an explicit diagnosis-and-correction pass.

After Correction SFT, we further refine the policy with Safety-Layered GRPO.

## 3.5 Safety-Layered Reward

We refine the corrected policy with GRPO using a reward that combines hard safety, continuous safety assessment, and driving progress. Hard-safety feasibility is determined by $F ( \tau ) \in \{ 0 , 1 \}$ in Eq. 2, which requires NC, DAC, and TTC to be satisfied simultaneously.

Continuous Safety Score. Binary feasibility alone cannot distinguish failure severity among unsafe trajectories or safety margins among feasible ones. We therefore construct a continuous safety score from simulator-derived margins and violation times. For each safety factor $k \in \{ \mathrm { c l r , t t c , r o a d } \}$ , corresponding to

obstacle clearance, TTC safety, and drivable-area compliance, let $m _ { k } ( \tau )$ denote its minimum signed margin and $t _ { k } ( \tau )$ its first violation time. We normalize them as

$$
\bar { m } _ { k } ( \tau ) = \mathrm { c l i p } \left( \frac { m _ { k } ( \tau ) - l _ { k } } { u _ { k } - l _ { k } } , 0 , 1 \right) , \qquad \bar { t } _ { k } ( \tau ) = \mathrm { c l i p } \left( \frac { t _ { k } ( \tau ) } { T } , 0 , 1 \right) ,\tag{7}
$$

where $[ l _ { k } , u _ { k } ]$ is the factor-specific normalization interval and T is the simulation horizon. We set $t _ { k } ( \tau ) = T$ when the corresponding violation does not occur. The continuous safety score is

$$
S ( \tau ) = \sum _ { k } w _ { k } \left[ ( 1 - \rho _ { k } ) \bar { m } _ { k } ( \tau ) + \rho _ { k } \bar { t } _ { k } ( \tau ) \right] , \quad \quad \sum _ { k } w _ { k } = 1 ,\tag{8}
$$

where $w _ { k } \geq 0$ and $\rho _ { k } \in [ 0 , 1 ]$ , ensuring $S ( \tau ) \in [ 0 , 1 ]$ . Higher scores favor larger safety margins and later violations. Thus, S provides graded feedback on failure severity for unsafe trajectories while continuing to distinguish safety margins among hard-safe trajectories.

Safety-Layered Composition. We reward driving progress only after hard safety is satisfied. The progress score is defined using Ego Progress (EP):

$$
P ( \tau ) = \mathrm { E P } ( \tau )\tag{9}
$$

The final reward is

$$
R ( \tau ) = \lambda _ { F } F ( \tau ) + \lambda _ { S } \big [ 1 - \eta F ( \tau ) \big ] S ( \tau ) + \lambda _ { P } F ( \tau ) P ( \tau ) ,\tag{10}
$$

where $\eta \in [ 0 , 1 ]$ controls the attenuation of the continuous safety reward once hard safety is satisfied. Unsafe trajectories receive only $\lambda _ { S } S ( \tau )$ . Hard-safe trajectories receive the feasibility bonus, retain a reduced safetyscore weight of $\lambda _ { S } ( 1 - \eta )$ , and additionally receive the progress reward.

We set $( \lambda _ { F } , \lambda _ { S } , \lambda _ { P } ) = ( 2 . 0 , 1 . 0 , 0 . 2 5 )$ and $\eta = 0 . 5$ , retaining half of the continuous safety weight for hard-safe trajectories. Consequently, unsafe trajectories have $R ( \tau ) \in [ 0 , 1 ]$ , whereas hard-safe trajectories have $R ( \tau ) \in [ 2 , 2 . 7 5 ]$ ]. This guarantees a strict reward preference for hard-safe trajectories while preserving fine-grained safety feedback within both groups.

## 4 Experiments

## 4.1 Experimental Setup

Dataset and Evaluation Protocol. We conduct experiments on NAVSIM [17], a planning-oriented autonomous driving benchmark built on OpenScene/nuPlan. Following the official NAVSIM v1 protocol, we use NAVTRAIN for supervised training, failure mining, and reinforcement learning, and reserve NAVTEST exclusively for evaluation. All corrective trajectories are constructed from the training split only. We additionally evaluate the same checkpoint used for NAVSIM v1 on the original NAVTEST scenes using the extended metrics of NAVSIM $\mathbf { v } 2$ [18], without additional training. This is a single-stage, original-scene evaluation and does not include synthetic second-stage observations or two-stage pseudo-simulation aggregation.

Metrics. We report Predictive Driver Model Score (PDMS) for overall benchmark comparison. To evaluate our safety-oriented objective, we analyze No At-Fault Collision (NC), Drivable Area Compliance (DAC), and Time-to-Collision (TTC) as the key safety outcomes, alongside Ego Progress (EP) and Comfort (C). We interpret safety and progress jointly: similar aggregate PDMS values need not imply similar safety performance. All metrics are higher-is-better.

Table 1: Comparison with state-of-the-art methods on NAVSIM v1. All metrics are higher-is-better.
<table><tr><td>Method</td><td>Params.</td><td>NC</td><td>DAC</td><td>TTC</td><td>C</td><td>EP</td><td>PDMS</td></tr><tr><td>End-to-End Methods</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>UniAD [19]</td><td></td><td>97.8</td><td>91.9</td><td>92.9</td><td>100</td><td>78.8</td><td>83.4</td></tr><tr><td>TransFuser [20]</td><td></td><td>97.7</td><td>92.8</td><td>92.8</td><td>100</td><td>79.2</td><td>84.0</td></tr><tr><td>DiffusionDrive [21]</td><td></td><td>98.2</td><td>96.2</td><td>94.7</td><td>100</td><td>82.2</td><td>88.1</td></tr><tr><td>Driving VLA Methods</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Base SFT (Qwen3.5-4B) [22]</td><td>4B</td><td>98.6</td><td>95.8</td><td>95.4</td><td>100</td><td>81.1</td><td>87.7</td></tr><tr><td>AutoVLA [8]</td><td>3B</td><td>98.4</td><td>95.6</td><td>98.0</td><td>99.9</td><td>81.9</td><td>89.1</td></tr><tr><td>SafeAlign-VLA [14]</td><td>7B</td><td>98.6</td><td>97.2</td><td>98.1</td><td>100</td><td>81.7</td><td>89.1</td></tr><tr><td>DriveVLA-W0 [23]</td><td>8B</td><td>98.7</td><td>99.1</td><td>95.3</td><td>99.3</td><td>83.3</td><td>90.2</td></tr><tr><td>ReCogDrive [9]</td><td>2B</td><td>97.9</td><td>97.3</td><td>94.9</td><td>100</td><td>87.3</td><td>90.8</td></tr><tr><td>ELF-VLA [13]</td><td>8B</td><td>98.9</td><td>98.1</td><td>96.0</td><td>100</td><td>85.3</td><td>91.0</td></tr><tr><td>DriveTeach-VLA [24]</td><td>3B</td><td>98.5</td><td>96.9</td><td>97.9</td><td>98.2</td><td>88.5</td><td>90.4</td></tr><tr><td>DriveMA [25]</td><td>4B</td><td>98.7</td><td>97.8</td><td>95.5</td><td>99.9</td><td>86.7</td><td>91.2</td></tr><tr><td>Qwen-Drive-1.0-RL [10]</td><td>4B</td><td>98.6</td><td>98.2</td><td>95.9</td><td>100</td><td>84.8</td><td>90.7</td></tr><tr><td>ReflectDrive [11]</td><td>8B</td><td>97.7</td><td>99.3</td><td>93.5</td><td>100</td><td>86.9</td><td>91.1</td></tr><tr><td>DriveFine [26]</td><td>8B</td><td>98.6</td><td>97.9</td><td>95.2</td><td>99.9</td><td>85.5</td><td>90.7</td></tr><tr><td>RefineDrive</td><td>4B</td><td>98.7</td><td>98.2</td><td>95.9</td><td>100</td><td>87.2</td><td>91.7</td></tr></table>

Model and Training Details. We instantiate RefineDrive with Qwen3.5-4B and first perform trajectory SFT on NAVTRAIN. The resulting policy is rolled out to mine safety-critical failures, from which we construct the diagnosis-and-correction dataset. We then perform Correction SFT followed by Safety-Layered GRPO. All ablations start from the same base SFT checkpoint. During evaluation, every checkpoint, including those evaluated immediately after Correction SFT, directly predicts eight (x, y, heading) waypoints at 0.5 s intervals over a 4 s horizon from the current front-view image, ego state, and historical trajectory. No failed trajectory is supplied, and no additional diagnosis or trajectory-repair pass is performed.

## 4.2 Main Results

Results on NAVSIM v1. Table 1 compares our method with representative end-to-end and VLA planners on NAVSIM v1. Our 4B model achieves 91.7 PDMS, together with 98.7 NC, 98.2 DAC, and 95.9 TTC, demonstrating strong overall and safety performance without relying on a substantially larger foundation model.

Extended-Metric Evaluation on NAVTEST. Using the extended metrics of NAVSIM v2 on the original NAVTEST scenes, RefineDrive achieves 89.4 EPDMS with the same checkpoint used for NAVSIM v1 and no additional training. This complements the v1 results with a broader assessment of planning quality on the same test split.

Table 2: Comparison on the original NAVTEST scenes using NAVSIM v2 extended metrics. Our result uses the same checkpoint as the NAVSIM v1 evaluation, without additional training. All metrics are higheris-better. \* indicates the exact NAVSIM v2 evaluator revision is not specified in the paper. † indicates results evaluated on the bug-fixed version of NAVSIM.
<table><tr><td>Method</td><td>NC</td><td>DAC</td><td>DDC</td><td>TLC</td><td>EP</td><td>TTC</td><td>LK</td><td>HC</td><td>EC</td><td>EPDMS</td></tr><tr><td>ReCogDrive [9]</td><td>98.3</td><td>95.2</td><td>99.5</td><td>99.8</td><td>87.1</td><td>97.5</td><td>96.6</td><td>98.3</td><td>86.5</td><td>83.6</td></tr><tr><td>DiffusionDriveV2* [27]</td><td>97.7</td><td>96.6</td><td>99.2</td><td>99.8</td><td>88.9</td><td>97.2</td><td>96.0</td><td>97.8</td><td>91.0</td><td>85.5</td></tr><tr><td>DriveVLA-W0* [23]</td><td>98.5</td><td>99.1</td><td>98.0</td><td>99.7</td><td>86.4</td><td>98.1</td><td>93.2</td><td>97.9</td><td>58.9</td><td>86.1</td></tr><tr><td>DriveSuprim (ViT-L) [28]</td><td>98.4</td><td>98.6</td><td>99.6</td><td>99.8</td><td>90.5</td><td>97.8</td><td>97.0</td><td>98.3</td><td>78.6</td><td>87.1</td></tr><tr><td>ExploreVLA* [29]</td><td>98.8</td><td>96.2</td><td>99.6</td><td>99.8</td><td>87.1</td><td>98.2</td><td>97.8</td><td>98.3</td><td>86.8</td><td>88.8</td></tr><tr><td>DriveFine† [26]</td><td>98.7</td><td>97.3</td><td>99.5</td><td>99.8</td><td>88.7</td><td>97.8</td><td>97.7</td><td>98.4</td><td>83.8</td><td>89.7</td></tr><tr><td>RefineDrive†</td><td>98.7</td><td>98.2</td><td>98.1</td><td>99.7</td><td>90.5</td><td>98.2</td><td>94.4</td><td>97.8</td><td>82.7</td><td>89.4</td></tr></table>

Table 3: Ablation of failure-aware Correction SFT. Human GT and Minimal use the same failure cases and identical structured diagnosis supervision. Minimal w/o Diagnosis retains the same corrective trajectory targets as Minimal but removes diagnosis supervision.
<table><tr><td>Correction Strategy</td><td>NC</td><td>DAC</td><td>EP</td><td>TTC</td><td>C</td><td>PDMS</td></tr><tr><td>Base SFT</td><td>98.6</td><td>95.8</td><td>81.1</td><td>95.4</td><td>100</td><td>87.7</td></tr><tr><td>Human GT</td><td>98.6</td><td>96.1</td><td>81.6</td><td>95.7</td><td>100</td><td>88.2</td></tr><tr><td>Minimal w/o Diagnosis</td><td>98.5</td><td>96.2</td><td>81.7</td><td>95.7</td><td>100</td><td>88.3</td></tr><tr><td>Minimal</td><td>98.7</td><td>96.4</td><td>82.0</td><td>95.9</td><td>100</td><td>88.6</td></tr></table>

## 4.3 Ablation Studies

We conduct controlled ablations on NAVSIM v1 to study: (1) learning from self-generated failures; (2) minimal correction versus Human GT; (3) the benefit of structured failure-diagnosis supervision; and (4) Safety-Layered Reward versus direct PDMS optimization. All variants share the same base SFT model and evaluation protocol, and Human-GT and minimal-correction variants use the same failure cases.

Failure-Aware Correction. Using self-generated failures with Human GT supervision improves NAVTEST PDMS from 87.7 to 88.2. With the failure cases and structured diagnosis supervision held fixed, replacing Human GT with retrieved minimal corrections further increases PDMS to 88.6, with gains in NC, DAC, EP, and TTC. Across the 32,650 paired Correction-SFT records, however, the retrieved targets have a slightly lower mean PDMS than the corresponding Human GT trajectories (95.4 vs. 95.8). Thus, the downstream improvement cannot be explained by higher average target PDMS. Furthermore, removing diagnosis supervision while keeping the corrective trajectory targets unchanged reduces PDMS from 88.6 to 88.3. This suggests that structured failure-diagnosis supervision benefits the resulting planning policy beyond corrective trajectory supervision alone.

Safety-Layered Reinforcement Learning. Table 4 isolates the RL objective from the same minimalcorrection SFT checkpoint. Direct PDMS optimization increases EP from 82.0 to 87.2, but lowers NC from 98.7 to 98.2 and TTC from 95.9 to 94.9. Our Safety-Layered Reward reaches the same reported mean EP (87.2), while retaining the pre-RL NC and TTC scores and increasing DAC to 98.2. Relative to PDMS Reward, it improves NC, DAC, and TTC by 0.5, 0.4, and 1.0 percentage points, respectively, and PDMS from 91.1 to 91.7. This comparison supports improved safety without a reduction in reported mean progress.

Table 4: Ablation of the RL objective from the same minimal-correction SFT checkpoint.
<table><tr><td>RL Objective</td><td>NC</td><td>DAC</td><td>EP</td><td>TTC</td><td>C</td><td>PDMS</td></tr><tr><td>No RL</td><td>98.7</td><td>96.4</td><td>82.0</td><td>95.9</td><td>100</td><td>88.6</td></tr><tr><td>PDMS Reward</td><td>98.2</td><td>97.8</td><td>87.2</td><td>94.9</td><td>100</td><td>91.1</td></tr><tr><td>Safety-Layered Reward</td><td>98.7</td><td>98.2</td><td>87.2</td><td>95.9</td><td>100</td><td>91.7</td></tr></table>

Table 5: Effect of Correction SFT under the same Safety-Layered GRPO objective. Full Correction SFT achieves the highest NC, DAC, and TTC among the compared variants, with lower EP.
<table><tr><td>Training Variant</td><td>NC</td><td>DAC</td><td>EP</td><td>TTC</td><td>C</td><td>PDMS</td></tr><tr><td>w/o Correction SFT</td><td>97.8</td><td>97.7</td><td>88.8</td><td>94.1</td><td>100</td><td>91.2</td></tr><tr><td>Minimal-Correction SFT w/o Diagnosis</td><td>98.2</td><td>97.8</td><td>87.5</td><td>95.3</td><td>99.9</td><td>91.3</td></tr><tr><td>Human-GT Correction SFT</td><td>98.1</td><td>97.7</td><td>89</td><td>94.7</td><td>99.9</td><td>91.6</td></tr><tr><td>Full Correction-SFT</td><td>98.7</td><td>98.2</td><td>87.2</td><td>95.9</td><td>100</td><td>91.7</td></tr></table>

![](images/7bc68b24c449d02988a0010d24cfd91cb59686566f39dcd532b4ed9007f995bc.jpg)  
Figure 3: Qualitative comparison of direct trajectory predictions on NAVTEST: (a) off-road cases and (b) collision cases. Red, blue, and green denote the base-policy prediction, RefineDrive prediction, and human trajectory, respectively. Both policies predict independently from the same driving context; RefineDrive is not conditioned on the base-policy prediction.

Effect on the Final Policy. Table 5 evaluates Correction SFT variants followed by the same Safety-Layered GRPO. Full Correction SFT attains the highest NC, DAC, and TTC among all compared variants. Relative to omitting Correction SFT, the gains are 0.9, 0.5, and 1.8 percentage points, respectively, with EP decreasing from 88.8 to 87.2. Relative to Human-GT Correction SFT, it increases the three safety metrics by $0 . 6 , 0 . 5 ,$ , and 1.2 points, while EP decreases from 89.0 to 87.2. Thus, their similar PDMS values (91.7 vs. 91.6) accompany different safety–progress profiles and do not imply similar safety performance. Removing diagnosis while retaining the same corrective targets lowers NC, DAC, and TTC by 0.5, 0.4, and 0.6 points, with EP increasing from 87.2 to 87.5. Together, these comparisons support a safety-oriented contribution from diagnosis-and-correction supervision after RL, with an explicit progress trade-off.

## 4.4 Qualitative Analysis

Fig. 3 compares the base SFT policy and RefineDrive on the same NAVTEST scenes. Both policies directly predict trajectories from the driving context. In these examples, RefineDrive avoids the off-road and collision behaviors exhibited by the base policy.

## 5 Conclusion

We present RefineDrive, a failure-guided post-training framework integrating Reliable Diagnosis, Minimum-Correction Target Retrieval, and Safety-Layered GRPO, with no inference-time diagnosis or repair. Controlled ablations show that diagnosis-and-correction supervision improves NC, DAC, and TTC after RL at some cost to progress, while Safety-Layered Reward improves all three at the same mean EP as direct PDMS optimization. RefineDrive achieves 91.7 PDMS on NAVSIM v1 and, with the same checkpoint and no further training, 89.4 EPDMS on the original NAVTEST scenes under NAVSIM v2 extended metrics.

## References

[1] Zhenhua Xu, Yujia Zhang, Enze Xie, Zhen Zhao, Yong Guo, Kwan-Yee K Wong, Zhenguo Li, and Hengshuang Zhao. Drivegpt4: Interpretable end-to-end autonomous driving via large language model. IEEE Robotics and Automation Letters, 9(10):8186–8193, 2024.

[2] Jiageng Mao, Yuxi Qian, Junjie Ye, Hang Zhao, and Yue Wang. Gpt-driver: Learning to drive with gpt. arXiv preprint arXiv:2310.01415, 2023.

[3] Hao Shao, Yuxuan Hu, Letian Wang, Guanglu Song, Steven L Waslander, Yu Liu, and Hongsheng Li. Lmdrive: Closed-loop end-to-end driving with large language models. In 2024 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 15120–15130. IEEE, 2024.

[4] Chonghao Sima, Katrin Renz, Kashyap Chitta, Li Chen, Hanxue Zhang, Chengen Xie, Jens Beißwenger, Ping Luo, Andreas Geiger, and Hongyang Li. Drivelm: Driving with graph visual question answering. In European conference on computer vision, pages 256–274. Springer, 2024.

[5] Ming Nie, Renyuan Peng, Chunwei Wang, Xinyue Cai, Jianhua Han, Hang Xu, and Li Zhang. Reason2drive: Towards interpretable and chain-based reasoning for autonomous driving. In European Conference on Computer Vision, pages 292–308. Springer, 2024.

[6] Xiaoyu Tian, Junru Gu, Bailin Li, Yicheng Liu, Yang Wang, Zhiyong Zhao, Kun Zhan, Peng Jia, Xianpeng Lang, and Hang Zhao. Drivevlm: The convergence of autonomous driving and large visionlanguage models. arXiv preprint arXiv:2402.12289, 2024.

[7] Xingcheng Zhou, Xuyuan Han, Feng Yang, Yunpu Ma, Volker Tresp, and Alois Knoll. Opendrivevla: Towards end-to-end autonomous driving with large vision language action model. In Proceedings of the AAAI Conference on Artificial Intelligence, volume 40, pages 13782–13790, 2026.

[8] Zewei Zhou, Tianhui Cai, Seth Zhao, Yun Zhang, Zhiyu Huang, Bolei Zhou, and Jiaqi Ma. Autovla: A vision-language-action model for end-to-end autonomous driving with adaptive reasoning and rein forcement fine-tuning. Advances in Neural Information Processing Systems, 38:27920–27956, 2026.

[9] Kaixin Xiong, Xiangyu Guo, Fang Li, Sixu Yan, Gangwei Xu, Lijun Zhou, Long Chen, Haiyang Sun, Bing Wang, Kun Ma, et al. Recogdrive: A reinforced cognitive framework for end-to-end autonomous driving. In International Conference on Learning Representations, volume 2026, pages 157518–157556, 2026.

[10] Xin Zhou, Zongchuang Zhao, Zhibo Yang, Mingsheng Li, Humen Zhong, Shuai Bai, Du Chu, Ruizhe Chen, Zhaohai Li, Jun Tang, et al. Qwen-drive-1.0: An initial step towards a vision-language foundation model for autonomous driving. arXiv preprint arXiv:2609.00111, 2026.

[11] Pengxiang Li, Yinan Zheng, Yue Wang, Huimin Wang, Hang Zhao, Jingjing Liu, Xianyuan Zhan, Kun Zhan, and Xianpeng Lang. Discrete diffusion for reflective vision-language-action models in autonomous driving. In International Conference on Learning Representations, volume 2026, pages 14647–14666, 2026.

[12] Mayank Bansal, Alex Krizhevsky, and Abhijit Ogale. Chauffeurnet: Learning to drive by imitating the best and synthesizing the worst. arXiv preprint arXiv:1812.03079, 2018.

[13] Yuechen Luo, Fang Li, Qimao Chen, Shaoqing Xu, Jiaxin Liu, Ziying Song, Zhi-xin Yang, and Fuxi Wen. Unleashing vla potentials in autonomous driving via explicit learning from failures. In Proceedings ofthe IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 24833–24842, 2026.

[14] Kefei Tian, Yuansheng Lian, Kai Yang, Xiangdong Chen, and Shen Li. Safealign-vla: A negative-enhanced safe alignment framework for risk-aware autonomous driving. arXiv preprint arXiv:2605.19524, 2026.

[15] Hao Dou. Fire-vla: Failure-informed self-evolution for vision-language-action models in autonomous driving. arXiv preprint arXiv:2608.13395, 2026.

[16] Cheng Gong, Haoyang Wang, Chao Lu, Zirui Li, and Jianwei Gong. Learning from mistakes: Rolloutretrieval lifelong policy learning for autonomous driving. arXiv preprint arXiv:2606.30537, 2026.

[17] Daniel Dauner, Marcel Hallgarten, Tianyu Li, Xinshuo Weng, Zhiyu Huang, Zetong Yang, Hongyang Li, Igor Gilitschenski, Boris Ivanovic, Marco Pavone, et al. Navsim: Data-driven non-reactive autonomous vehicle simulation and benchmarking. Advances in Neural Information Processing Systems, 37:28706–28719, 2024.

[18] Wei Cao, Marcel Hallgarten, Tianyu Li, Daniel Dauner, Xunjiang Gu, Caojun Wang, Yakov Miron, Marco Aiello, Hongyang Li, Igor Gilitschenski, Boris Ivanovic, Marco Pavone, Andreas Geiger, and Kashyap Chitta. Pseudo-simulation for autonomous driving. In Conference on Robot Learning (CoRL), 2025.

[19] Yihan Hu, Jiazhi Yang, Li Chen, Keyu Li, Chonghao Sima, Xizhou Zhu, Siqi Chai, Senyao Du, Tianwei Lin, Wenhai Wang, et al. Planning-oriented autonomous driving. In 2023 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 17853–17862. IEEE, 2023.

[20] Kashyap Chitta, Aditya Prakash, Bernhard Jaeger, Zehao Yu, Katrin Renz, and Andreas Geiger. Transfuser: Imitation with transformer-based sensor fusion for autonomous driving. IEEE transactions on pattern analysis and machine intelligence, 45(11):12878–12895, 2022.

[21] Bencheng Liao, Shaoyu Chen, Haoran Yin, Bo Jiang, Cheng Wang, Sixu Yan, Xinbang Zhang, Xiangyu Li, Ying Zhang, Qian Zhang, et al. Diffusiondrive: Truncated diffusion model for end-to-end autonomous driving. In 2025 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 12037–12047. IEEE, 2025.

[22] Qwen Team. Qwen3.5: Towards native multimodal agents, February 2026.

[23] Yingyan Li, Shuyao Shang, Weisong Liu, Bing Zhan, Haochen Wang, Yuqi Wang, Yuntao Chen, Xiaoman Wang, Yasong An, Chufeng Tang, et al. Drivevla-w0: World models amplify data scaling law in autonomous driving. In International Conference on Learning Representations, volume 2026, pages 7890–7911, 2026.

[24] Yuguang Yang, Canyu Chen, Zhewen Tan, Yizhi Wang, Zichao Feng, Chunyang Liu, Kehua Sheng, Juan Zhang, Linlin Yang, Baochang Zhang, et al. Teaching vision-language-action models what to see and where to look. In European Conference on Computer Vision, pages 317–335. Springer, 2026.

[25] Weicheng Zheng, Yixin Huang, Qiao Sun, Derun Li, et al. Drivema: Rethinking language interfaces in driving vlas with one-step meta-actions. arXiv preprint arXiv:2605.21273, 2026.

[26] Chenxu Dang, Sining Ang, Yongkang Li, Haochen Tian, Jie Wang, Guang Li, Hangjun Ye, Jie Ma, Long Chen, and Yan Wang. Drivefine: Refining-augmented masked diffusion vla for precise and robust driving. arXiv preprint arXiv:2602.14577, 2026.

[27] Jialv Zou, Shaoyu Chen, Bencheng Liao, Zhiyu Zheng, Yuehao Song, Lefei Zhang, Qian Zhang, Wenyu Liu, and Xinggang Wang. Diffusiondrivev2: Reinforcement learning-constrained truncated diffusion modeling in end-to-end autonomous driving. arXiv preprint arXiv:2512.07745, 2025.

[28] Wenhao Yao, Zhenxin Li, Shiyi Lan, Zi Wang, Xinglong Sun, Jose M Alvarez, and Zuxuan Wu. Drivesuprim: Towards precise trajectory selection for end-to-end planning. In Proceedings of the AAAI Conference on Artificial Intelligence, volume 40, pages 11910–11918, 2026.

[29] Zihao Sheng, Xin Ye, Jingru Luo, Sikai Chen, and Liu Ren. Explorevla: Dense world modeling and exploration for end-to-end autonomous driving. arXiv preprint arXiv:2604.02714, 2026.
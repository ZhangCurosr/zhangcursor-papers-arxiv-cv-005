# SPATIALSKILL: SELF-EVOLVING SKILLS FOR CROSS-VIEW SPATIAL REASONING

Ruifan Zuo<sup>1</sup> Guocheng Hu<sup>1</sup> Wanshui Gan<sup>2</sup> Junyi Wang<sup>1</sup> Xiang Lei<sup>3</sup> Tian Gan<sup>1†</sup> <sup>1</sup>Shandong University <sup>2</sup>Shanghai AI Laboratory <sup>3</sup>Zhiyang Innovation Co., Ltd.   
<sup>†</sup> Corresponding Author   
gantian@sdu.edu.cn

## ABSTRACT

Cross-view spatial reasoning requires a model to align different viewpoints into a coherent spatial representation, yet this ability remains challenging for visionlanguage models despite being natural to humans. Existing methods typically improve spatial reasoning by updating model weights, which keeps the acquired knowledge implicit and tied to a specific backbone. We propose SpatialSkill, a weight-update-free framework that enables a frozen vision-language model to accumulate explicit natural-language reasoning skills from offline trajectories. Unlike symbolic tasks, perceptual skills cannot be reliably verified simply by executing them: a plausible spatial rule may lack visual support or require transformations that the frozen model cannot perform. SpatialSkill therefore admits candidate skills only after visual-grounding and executability checks, constrains manual evolution to prevent harmful regressions, and routes skills by spatial-reasoning category to reduce negative transfer. On CityCube, across four frozen executors, SpatialSkill yields consistent gains, and a 9B executor equipped with SpatialSkill surpasses the strongest closed-source reference in our evaluation. The skills are stored in a versioned natural-language manual, making the reasoning strategies explicit and auditable without modifying model parameters. Code at https://github.com/vindahi/SpatialSkill.

## 1 INTRODUCTION

Spatial intelligence (Hong et al., 2026; Jia et al., 2026; Majumdar et al., 2024) is central to embodied cognition. Missions such as UAV inspection and disaster response require relating objects, directions, and structures across aerial and ground-level observations. We refer to this ability as cross-view spatial reasoning (Zhou et al., 2026; Peng et al., 2026; Xiang et al., 2026; Du et al., 2024; Qiao et al., 2026). Solving these cross-view problems often requires mentally transforming and aligning viewpoints rather than relying only on visual similarity or linguistic priors, yet current vision-language models remain weak at these operations (Huang et al., 2026; Zhang et al., 2026b; 2025; Xu et al., 2026b; Lee et al., 2025). Existing approaches (Ma et al., 2025; Hong et al., 2023; Chen et al., 2024) typically improve spatial reasoning through parameter or representation updates, which encode the acquired knowledge implicitly in a particular model and make the resulting reasoning strategies difficult to inspect or revise independently of the model. We therefore ask: can a frozen VLM improve its cross-view spatial reasoning without parameter updates, simply by accumulating explicit natural-language skills? Fig. 1 illustrates that even a simple visually grounded rule can correct an error made by a frozen executor. The key challenge is therefore not whether explicit skills can help, but how to acquire, maintain, and deploy them reliably.

This raises three key challenges. ➊ Reliable skill admission. Unlike symbolic skills that can be validated through direct execution, a natural-language spatial rule lacks a direct rule-level verifier and may sound plausible even when unsupported by visual evidence (Han et al., 2026; Tong et al., 2026; Yang et al., 2025b). Moreover, a geometrically valid rule may still require reasoning operations beyond what the frozen executor can reliably perform. Thus, candidate skills must be checked for both visual grounding and executor executability before entering the skill library. ➋ Stable skill evolution. Even when individual candidate skills are plausible, repeated revisions may accumulate noise, interact unexpectedly, or remove useful rules during compression, gradually degrading the skill library over time (Yashwante & Yu, 2026; Sun et al., 2026). Reliable skill evolution therefore requires controlling growth and preventing harmful updates from being propagated. ➌ Category-dependent skill effectiveness. Different spatial-reasoning categories involve different reasoning patterns and operations, so the same manual or inference strategy may not benefit every category equally (Wang et al., 2026; Zhou et al., 2025; Wang et al., 2025b; Khalid et al., 2025). Uniformly injecting skills can therefore introduce negative transfer, motivating category-conditioned deployment.

![](images/a029c8458b18d5a9c552bf77955c08714eff9c5a08b453bed9714a11d66d9bf3.jpg)  
Figure 1: A perspective-taking (PT) instance and a spatial-relation (SR) instance, each pairing aerial and ground views. Frozen executor fails on both without manuals, while a visually grounded rule from the evolved SpatialSkill manual enables it to correct its prediction without any weight update.

Motivated by these challenges, we introduce Self-Evolving Skillsfor Cross-View Spatial Reasoning (SpatialSkill), a framework that improves a frozen VLM by accumulating explicit natural-language skills without updating its parameters. SpatialSkill consists of three core components. First, a grounded admission gate admits candidate skills only after checks for visual support and compatibility with the frozen executor’s capabilities. Second, stability controls regulate skill accumulation by limiting uncontrolled growth, protecting stable skills from blind removal, and rejecting updates that cause excessive performance degradation. Third, category-conditioned routing selects how the final manual is used for each spatial reasoning category, reducing negative transfer from uniform skill deployment. Together, these components produce a versioned and auditable skill manual that can be selectively deployed while keeping the executor unchanged.

Our contributions are threefold. (i) To our knowledge, we are the first to study self-evolving explicit natural-language skills for frozen VLMs in real-world cross-view spatial reasoning, identifying three requirements for reliability: grounded admission, stable accumulation, and category-conditioned deployment. (ii) We propose SpatialSkill, a framework that addresses these requirements through grounded admission, stability-constrained evolution, and category-conditioned routing, storing acquired reasoning strategies in an explicit, versioned, and auditable natural-language manual without parameter updates. (iii) On CityCube across four frozen executors, SpatialSkill consistently yields accuracy gains of 2.65–4.92 percentage points. With SpatialSkill, the 9B executor reaches 57.58%, outperforming the strongest closed-source baseline evaluated in our experiments.

## 2 RELATED WORK

Spatial Reasoning in Vision-Language Models. Spatial reasoning is a central diagnostic axis for vision-language models. Recent benchmarks examine complementary capabilities, including egocentric video (Yang et al., 2025a), multi-viewpoint localization (Li et al., 2025), multi-image intelligence (Yang et al., 2026), cognitive-psychology axes (Jia et al., 2026), unified cross-task evaluation (Huang et al., 2026), and fine-grained failure analysis (Stogiannidis et al., 2025; Liu et al., 2025). Across these settings, models remain challenged when reasoning requires viewpoint transformation and cross-view alignment rather than language priors or appearance matching. Recent methods address these limitations through several forms of model adaptation. Reinforcement learning improves grounded multi-step 3D reasoning (Batra et al., 2025), activates latent abilities (Zhan et al., 2025), and extends to multi-view transformations (Li et al., 2026c). Supervised approaches inject depth signals (Cai et al., 2025) or align implicit representations (Li et al., 2026a), while robotscale pretraining embeds spatial structure (Qu et al., 2025). Together, these studies establish the importance of explicit spatial structure and learned adaptation for cross-view reasoning, with most improvements encoded in model weights or representations.

Self-Evolving Agents and Skill Accumulation. Self-improvement from practice provides a complementary direction for turning experience into reusable knowledge (Gao et al., 2025). Existing approaches develop skill libraries that compile procedures into APIs (Zhao et al., 2025; Zheng et al., 2025; Prabhu et al., 2026; Tian et al., 2026), recursive augmentation with RL (Xia et al., 2026; Liu et al., 2026a), resource-to-library conversion (Shen et al., 2026; Ju et al., 2026), structured experience notes (Xu et al., 2025), and OS-style hierarchies (Kang et al., 2025). Many of these systems evolve executable artifacts or model parameters, allowing environments or training loops to provide direct evaluation signals. A close text-playbook precedent is ACE (Zhang et al., 2026a), which accumulates reusable lessons under text-task scoring. These two research directions leave a less explored intersection: evolving explicit natural-language skills for frozen VLMs in perceptual cross-view reasoning. A skill may lack visual support, exceed the executor’s capabilities, or interact negatively with previously accumulated guidance. This setting therefore raises questions of grounded admission, stable accumulation, and selective deployment, which motivate SpatialSkill.

## 3 METHODOLOGY

## 3.1 PROBLEM DEFINITION AND FRAMEWORK OVERVIEW

We study cross-view spatial reasoning in a multiple-choice setting. Each example is defined as $\boldsymbol { x } = ( I _ { 1 : k } , q , \mathcal { O } )$ , where $I _ { 1 : k }$ represents k distinct views of a scene, q is a question, and O is the candidate answer set. A frozen executor predicts an answer $\hat { y } \in \mathcal { O }$ . Each example belongs to one of five spatial reasoning categories: mental reconstruction (MR), perspective taking (PT), spatial relation reasoning (SR), world knowledge (WK), and comprehensive reasoning (CR), which are used for both category-wise evaluation and routing. During skill evolution, training examples additionally provide reference reasoning trajectories used as attribution references during reflection. SpatialSkill distills reusable reasoning patterns from these trajectories into natural-language skills, which are organized into a structured manual to guide the unchanged executor at inference.

![](images/2524cb7fe3412f162e09a7c1f87c3d7efbfe074214c2d60625afabb9c9090445.jpg)  
Figure 2: Overview of Self-Evolving Skills for Cross-View Spatial Reasoning (SpatialSkill)

SpatialSkill converts offline reasoning trajectories from a frozen VLM into a reusable naturallanguage skill manual and selectively deploys that manual at inference. Fig. 2 summarizes this process in three stages: the frozen VLM executor first produces cross-view reasoning trajectories; a skill-evolution module reflects on these trajectories and updates the manual under grounding and stability constraints; finally, a category router determines how the manual is used for each reasoning category.

## 3.2 SKILL REFLECTION AND GROUNDED ADMISSION

Reasoning trajectories capture useful experience from the frozen executor, but remain instancespecific. SpatialSkill therefore uses reflection to abstract reusable revisions to a natural-language skill manual. Because reflected candidates may still lack visual support or require operations beyond the executor’s capabilities, they are screened for visual grounding and executability before entering the manual. We maintain a skill manual M, initialized empty and organized into four sections: planning, cross-view alignment, spatial reasoning, and failure avoidance, together with execution notes. To evolve the manual incrementally, we process the evolution stream in windows. For window $w ,$ let $T _ { w }$ denote the collected trajectories and $M _ { w - 1 }$ the current manual. A fixed reflection model Φ proposes candidate revisions:

$$
\Delta _ { w } \ = \ \Phi \big ( T _ { w } , M _ { w - 1 } \big ) ,\tag{1}
$$

where $\Delta _ { w }$ denotes the proposed revisions. Each revision performs one of four operations: adding a skill, rewriting a skill, correcting a skill, or adding an execution note. Each candidate is further associated with skill-level or execution-level attribution estimated against the reference trajectories. Execution-level revisions only update execution notes, preventing episodic noise from contaminating reusable skill rules.

Because reflected candidates are not necessarily reliable, each candidate rule r is screened by an admission gate, implemented by the fixed reflection model, that checks visual grounding and executability:

$$
\mathcal { G } ( \boldsymbol { r } ) \ : = \ : \mathcal { V } ( \boldsymbol { r } ) \ : \wedge \ : \mathcal { E } ( \boldsymbol { r } ) ,\tag{2}
$$

where $\mathcal { V } ( r ) = 1$ requires the rule to be supported by visual evidence, while $\mathcal { E } ( r ) = 1$ requires the prescribed operation to remain within the frozen executor’s capabilities. Only candidates satisfying $\mathcal { G } ( r ) = 1$ are admitted into the manual. Rejected candidates are logged with their triggering evidence, keeping the evolution process auditable.

The admitted revisions are then applied to the current manual to form a provisional manual:

$$
\widetilde { M } _ { w } \ = \ U \Big ( M _ { w - 1 } , \big \{ \rho \in \Delta _ { w } : \mathcal { G } ( r _ { \rho } ) = 1 \big \} \Big ) ,\tag{3}
$$

where U applies the admitted revisions and $r _ { \rho }$ denotes the rule introduced or modified by revision $\rho .$ Although each incorporated revision is individually admissible, multiple revisions may still interact negatively after accumulation. The provisional manual is therefore further evaluated by the stability controls described next before becoming the delivered manual $M _ { w }$

## 3.3 STABILITY-CONSTRAINED SKILL EVOLUTION

The admission gate ensures that individual candidate rules are visually grounded and executable, but admissible revisions do not necessarily yield a better manual after accumulation. Multiple revisions may interact unexpectedly or gradually introduce noise into the skill library. We therefore evaluate each provisional manual on a held-out probe set $P$ before delivering an update. The probe set is stratified by reasoning category and remains disjoint from the evolution stream. It is never used for reflection or revision. The probe score of a manual is defined as:

$$
A ( M ) = \frac { 1 0 0 } { | P | } \sum _ { x \in P } \mathbf { 1 } \{ \hat { y } _ { M } ( x ) = y ( x ) \} ,\tag{4}
$$

where ${ \hat { y } } _ { M } ( x )$ denotes the executor prediction under manual $M , y ( x )$ is the answer option, and $\mathbf { 1 } \{ \cdot \}$ is the indicator function.

Manual-level evaluation alone, however, does not prevent instability during repeated evolution. As rules accumulate, the manual may reach its capacity and require compression. Blind compression

can remove useful rules together with noisy ones, creating new failures and further revisions. We characterize this pathological grow-and-compress regime as:

$$
\operatorname * { l i m } _ { W  \infty } \frac { 1 } { W } \sum _ { w = 1 } ^ { W } { \bf 1 } \{ R _ { w } \} \ = \ 1 \qquad \mathrm { a n d } \qquad \mathbb { E } \big [ A ( M _ { W } ) \big ] \ < \ A ( M _ { 0 } ) ,\tag{5}
$$

where $R _ { w }$ denotes the event that compression is triggered in window w. To control this failure pattern, SpatialSkill uses three complementary controls: a Growth Budget, Seniority Protection, and a Probe-Score Ratchet. They respectively limit manual expansion, protect long-standing rules during compression, and determine whether the resulting update is delivered.

First, excessive growth increases the need for later compression. We impose a net-growth budget:

$$
| M _ { w } | - | M _ { w - 1 } | \le \beta ,\tag{6}
$$

where $| M |$ is the number of rules and $\beta$ is the maximum net number of rules added in one window.   
This constraint limits excessive rule accumulation before compression.

Second, compression should not discard long-standing rules indiscriminately. We therefore protect stable rules from blind removal. A rule that survives multiple windows remains in the next manual unless an evidence-targeted revision explicitly modifies it:

$$
r \in M _ { w - 1 } , \mathrm { a g e } ( r ) \geq \tau , r \notin \mathcal { T } _ { w } \Longrightarrow r \in M _ { w } ,\tag{7}
$$

where $\mathrm { a g e } ( r )$ is the number of consecutive windows survived by rule $r , \tau$ is the protection window, and $\mathcal { T } _ { w }$ is the set of rules explicitly targeted by evidence-based revisions in window w. Thus, long-standing rules are protected from blind compression, while evidence-targeted revisions remain possible.

Finally, controlling growth and compression does not guarantee that the resulting manual improves the executor as a whole. We therefore use a probe-score ratchet to determine whether a provisional update is delivered. Let $b _ { 0 } = A ( M _ { 0 } )$ denote the initial probe score:

$$
M _ { w } = \left\{ \begin{array} { l l } { \widetilde { M } _ { w } , } & { A ( \widetilde { M } _ { w } ) \geq b _ { w - 1 } - \varepsilon , } \\ { M _ { w - 1 } , } & { \mathrm { o t h e r w i s e } , } \end{array} \right. \quad b _ { w } = \operatorname* { m a x } \big \{ b _ { w - 1 } , ~ A ( M _ { w } ) \big \} ,\tag{8}
$$

where $\widetilde { M } _ { w }$ denote the constrained provisional update passed to the ratchet, $b _ { w }$ records the best accepted probe score, and ε absorbs evaluation noise. Updates that exceed this tolerance are rejected, while the historical best score is preserved. Together, these constraints yield the following toleranceband invariant:

$$
A ( M _ { w } ) \geq A _ { w } ^ { \star } - \varepsilon , \qquad A _ { w } ^ { \star } = \operatorname* { m a x } _ { t \leq w } A ( M _ { t } ) .\tag{9}
$$

where $A _ { w } ^ { \star }$ denotes the best probe score achieved by any accepted manual up to window w. Therefore, evolution may plateau or fluctuate within the tolerance range, but large probe-score regressions are not delivered.

## 3.4 CATEGORY-CONDITIONED SKILL DEPLOYMENT

The final skill manual is not uniformly beneficial across reasoning categories. A strategy that helps one category may provide limited benefit for another. We therefore select the inference strategy according to the reasoning category instead of applying the manual globally.

Given the precomputed reasoning category $c \in { \mathcal { C } } ,$ the router selects a strategy $s = ( i , j )$ from ${ \cal { S } } = \{ 0 , 1 \} ^ { 2 }$ , where $i = 0$ denotes direct answering, $i \ : = \ : 1$ denotes chain-of-thought reasoning, $j = 1$ denotes providing the final manual $M _ { W }$ , and $j = 0$ denotes inference without the manual. The routing policy is a mapping $\pi : { \mathcal { C } } \to { \mathcal { S } }$ . If the performance of each strategy were known exactly, the optimal routing policy would maximize expected accuracy:

$$
\pi ^ { \star } = \arg \operatorname* { m a x } _ { \pi } \sum _ { c \in { \mathcal { C } } } p ( c ) \operatorname { A c c } \bigl ( c , \pi ( c ) \bigr ) ,\tag{10}
$$

where $p ( c )$ is the category prior and $\operatorname { A c c } ( c , s )$ denotes the accuracy of strategy s on category $c .$ In practice, these quantities are estimated from a finite probe set. To reduce sensitivity to sampling noise, we switch strategies when improvement satisfies a conservative confidence-bound criterion.

Let $\hat { a } _ { c , s }$ denote the empirical accuracy of strategy s on category c. Using a Wilson confidence interval $[ \ell _ { s } ( c ) , u _ { s } ( c ) ]$ , the router selects a candidate strategy only when its lower bound exceeds the default strategy’s upper bound by a margin δ:

$$
\pi ( c ) = \arg \operatorname* { m a x } _ { s \in \mathcal { F } ( c ) } \hat { a } _ { c , s } , \qquad \mathcal { F } ( c ) = \Big \{ s \in \mathcal { S } : \ell _ { s } ( c ) - u _ { s _ { 0 } ( c ) } ( c ) \geq \delta \Big \} ,\tag{11}
$$

where $s _ { 0 } ( c )$ is the stronger of the two non-manual strategies on the probe set. If $\mathcal { F } ( c )$ is empty, the router retains the default strategy. This conservative rule avoids unnecessary switching caused by probe variance.

## 4 EXPERIMENTS

## 4.1 EXPERIMENT SETUP

We evaluate on CityCube (Xu et al., 2026a) with its official 4,494/528 train/validation split. Validation is used only for final reporting. The 200-example training-split probe is disjoint from evolution and supplies acceptance and routing decisions. At test time, routing receives the pre-annotated CvSI category, we report category-wise and overall micro accuracy. We compare each frozen executor with its matched no-skill version: Qwen3.5-4B, Qwen3.5-9B, Gemma-3-4B, and GPT-5.6-luna.

## 4.2 IMPLEMENT DETAILS

The treated executors are Qwen3.5-4B, Qwen3.5-9B, Gemma-3-4B, and GPT-5.6-luna. The opensource reference models in the main table are Qwen3-VL-4B/8B/32B (Team, 2025b), Qwen3.5- 2B/27B/35B-A3B (Team, 2026b), InternVL3-8B/14B (Team, 2025a), Gemma-4-31B/26B-A4B, Gemma-3-12B (Team, 2026a), and Skywork-VL-Reward-7B (Wang et al., 2025a). We also evaluate the open-source spatial models Spatial-SSRL-7B (Liu et al., 2026b), SpaceR-SFT-7B (Ouyang et al., 2025), Spatial-MLLM-subset-sft (Li et al., 2026b), and ViLaSR (Wu et al., 2025). The crossmodel transfer analysis additionally uses Qwen3.5-2B as a target executor. The proprietary reference baselines are GPT-5.4, GPT-5.6-sol, Gemini-3.5-Flash, Gemini-3.7-Flash, Claude-Opus-5, and Claude-Sonnet-5; GPT-5.6-luna is also evaluated as a treated executor.

The executor and reflection model are frozen throughout. The reflection model is Qwen3.5-9B and is served locally for candidate generation, admission checks, and manual revision. Open-source executors are served with vLLM on a machine with four NVIDIA RTX 5090 GPUs. Smaller models use independent single-GPU replicas; 12B–14B models use tensor parallelism of two; and models of 26B or larger use a four-GPU tensor-parallel server. GPU memory utilization is set to 0.95, each request accepts up to ten images, and each server processes one concurrent multimodal sequence. Local runs use greedy decoding, while API baselines use their fixed native serving configuration and the same answer-extraction rule.

## 4.3 MAIN RESULTS

Table 1 reports the main results on CityCube (Xu et al., 2026a). We focus on paired comparisons between each frozen executor and its SpatialSkill-enhanced counterpart, while also reporting stronger open-source, spatially specialized, and closed-source models as reference points.

SpatialSkill consistently improves frozen executors. Across all four paired executors, Spatial Skill improves overall accuracy by 2.65–4.92 percentage points, with a mean gain of 3.41 points. Qwen3.5-4B improves from 47.16% to 52.08%, Qwen3.5-9B from 54.55% to 57.58%, Gemma-3-4B from 49.24% to 51.89%, and GPT-5.6-luna from 53.79% to 56.82%. Because the executor weights remain frozen throughout, these gains are achieved through explicit skill evolution and prompt-level deployment rather than parameter updates. The resulting knowledge remains stored in a natural-language manual, making the acquired reasoning strategies explicit, inspectable, and retractable.

SpatialSkill narrows the gap to stronger reference models. Qwen3.5-9B equipped with SpatialSkill reaches 57.58%, only 0.56 percentage points below the larger Qwen3.5-27B baseline at 58.14%. It also exceeds the best-performing closed-source reference models in our evaluation,

Table 1: Main results on the CityCube dataset. Accuracy is reported in percent. Option-wise Acc. Range (pp) is the maximum difference in conditional accuracy across predicted answer options.
<table><tr><td rowspan="2">Method</td><td colspan="7">Accuracy (%) ↑</td><td rowspan="2">Option-wise Acc. Range (pp) ↓</td></tr><tr><td>Overall</td><td>∆</td><td>CR</td><td>MR</td><td>PT</td><td>SR</td><td>WK</td></tr><tr><td>Human</td><td>88.3</td><td></td><td>93.1</td><td>92.4</td><td>87.4</td><td>90.2</td><td>78.6</td><td></td></tr><tr><td>Paired executors</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Qwen3.5-4B + SpatialSkill</td><td>52.08</td><td>+4.92</td><td>43.24</td><td>51.22</td><td>41.77</td><td>56.10</td><td>64.06</td><td>23.52</td></tr><tr><td>Qwen3.5-4B</td><td>47.16</td><td></td><td>51.35</td><td>53.66</td><td>39.87</td><td>45.53</td><td>52.34</td><td>27.28</td></tr><tr><td>Qwen3.5-9B + SpatialSkill Qwen3.5-9B</td><td>57.58</td><td>+3.03</td><td>59.46</td><td>67.07</td><td>51.90</td><td>52.85</td><td>62.50</td><td>23.10 39.21</td></tr><tr><td>Gemma-3-4B + SpatialSkill</td><td>54.55</td><td></td><td>56.76</td><td>64.63</td><td>47.47</td><td>47.97</td><td>62.50</td><td>47.94</td></tr><tr><td>Gemma-3-4B</td><td>51.89</td><td>+2.65</td><td>51.35 51.35</td><td>60.98</td><td>41.77</td><td>52.03</td><td>58.59</td><td></td></tr><tr><td>GPT-5.6-luna + SpatialSkill</td><td>49.24 56.82</td><td></td><td>62.16</td><td>59.76</td><td>37.97</td><td>52.03</td><td>53.12</td><td>72.12</td></tr><tr><td>GPT-5.6-luna</td><td>53.79</td><td>+3.03</td><td></td><td>62.20</td><td>45.57</td><td>56.91</td><td>65.62</td><td>19.37</td></tr><tr><td></td><td></td><td></td><td>48.65</td><td>57.32</td><td>42.41</td><td>59.35</td><td>61.72</td><td>30.43</td></tr><tr><td>Open-source reference models</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Qwen3-VL-4B</td><td>53.03</td><td></td><td>56.76</td><td>59.76</td><td>41.14</td><td>56.91</td><td>58.59</td><td>24.93</td></tr><tr><td>Qwen3-VL-8B</td><td>50.95</td><td></td><td>45.95</td><td>53.66</td><td>39.24</td><td>51.22</td><td>64.84</td><td>24.96</td></tr><tr><td>Qwen3-VL-32B</td><td>53.79</td><td></td><td>51.35</td><td>59.76</td><td>41.77</td><td>54.47</td><td>64.84</td><td>18.04</td></tr><tr><td>Qwen3.5-2B</td><td>46.59</td><td></td><td>48.65</td><td>53.66</td><td>33.54</td><td>52.85</td><td>51.56</td><td>49.77</td></tr><tr><td>Qwen3.5-27B</td><td>58.14</td><td></td><td>51.35</td><td>64.63</td><td>45.57</td><td>56.10</td><td>73.44</td><td>17.36</td></tr><tr><td>Qwen3.5-35B-A3B</td><td>53.22</td><td></td><td>56.76</td><td>54.88</td><td>41.77</td><td>55.28</td><td>63.28</td><td>31.00</td></tr><tr><td>InternVL3-8B</td><td>49.43</td><td></td><td>48.65</td><td>47.56</td><td>43.67</td><td>49.59</td><td>57.81</td><td>20.14</td></tr><tr><td>InternVL3-14B</td><td>51.70</td><td></td><td>56.76</td><td>51.22</td><td>44.30</td><td>52.03</td><td>59.38</td><td>51.97</td></tr><tr><td>Gemma-4-31B</td><td>54.55</td><td></td><td>40.54</td><td>53.66</td><td>41.77</td><td>56.10</td><td>73.44</td><td>26.52</td></tr><tr><td>Gemma-4-26B-A4B</td><td>52.27</td><td></td><td>40.54</td><td>56.10</td><td>41.14</td><td>56.91</td><td>62.50</td><td>17.78</td></tr><tr><td>Gemma-3-12B</td><td>50.95</td><td></td><td>56.76</td><td>54.88</td><td>39.87</td><td>47.15</td><td>64.06</td><td>39.70</td></tr><tr><td>Skywork-VL-Reward-7B</td><td>45.83</td><td></td><td>37.84</td><td>45.12</td><td>34.81</td><td>51.22</td><td>57.03</td><td>28.53</td></tr><tr><td>Open-source Spatial models</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Spatial-SSRL-7B</td><td>46.97</td><td></td><td>45.95</td><td>46.34</td><td>34.81</td><td>50.41</td><td>59.38</td><td>13.76</td></tr><tr><td>SpaceR-SFT-7B</td><td>44.13</td><td></td><td>54.05</td><td>47.56</td><td>36.71</td><td>33.33</td><td>58.59</td><td>41.07</td></tr><tr><td>Spatial-MLLM-subset-sft</td><td>39.77</td><td></td><td>37.84</td><td>36.59</td><td>36.71</td><td>39.02</td><td>46.88</td><td>73.94</td></tr><tr><td>ViLaSR</td><td>49.81</td><td></td><td>51.35</td><td>56.1</td><td>36.08</td><td>47.97</td><td>64.06</td><td>26.79</td></tr><tr><td>Closed-source reference models</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>GPT-5.4</td><td>50.38</td><td></td><td>37.84</td><td>57.32</td><td>41.14</td><td>50.41</td><td>60.94</td><td>31.00</td></tr><tr><td>Gemini-3.5-Flash</td><td>48.11</td><td></td><td>40.54</td><td>53.66</td><td>39.87</td><td>50.41</td><td>54.69</td><td>16.95</td></tr><tr><td>GPT-5.6-sol</td><td>56.06</td><td></td><td>45.95</td><td>67.07</td><td>44.30</td><td>54.47</td><td>67.97</td><td>24.36</td></tr><tr><td>Qwen3.7-plus</td><td>56.06</td><td></td><td>48.65</td><td>62.20</td><td>46.20</td><td>56.91</td><td>65.62</td><td>12.96</td></tr><tr><td>Gemini-3.7-Flash</td><td>52.27</td><td></td><td>45.95</td><td>57.32</td><td>43.67</td><td>53.66</td><td>60.16</td><td>14.28</td></tr><tr><td>Claude-Sonnet-5</td><td>52.65</td><td></td><td>45.95</td><td>50.00</td><td>49.37</td><td>48.78</td><td>64.06</td><td>24.05</td></tr><tr><td>Claude-Opus-5</td><td>52.08</td><td></td><td>35.14</td><td>60.98</td><td>44.30</td><td>48.78</td><td>64.06</td><td>20.67</td></tr></table>

GPT-5.6-sol and Qwen3.7-plus, both at 56.06%, by 1.52 points. Although Qwen3.5-27B remains the best-performing non-human model overall, these results show that explicit skill evolution can recover part of the performance gap to stronger models without modifying the executor backbone.

The gains are category- and executor-dependent. For Qwen3.5-4B, SpatialSkill improves SR and WK by 10.57 and 11.72 percentage points, respectively, but decreases performance on CR and MR. GPT-5.6-luna gains 13.51 points on CR but slightly declines on SR, while Qwen3.5-9B improves or matches its baseline across all five categories and Gemma-3-4B benefits mainly on MR, PT, and WK. This heterogeneity shows that skill effectiveness is not uniform across reasoning categories or executors, supporting category-conditioned routing rather than a single global deployment strategy.

A substantial gap remains between current models and human performance. For context, City Cube reports an overall human accuracy of 88.3%, whereas the best-performing non-human model in our evaluation, Qwen3.5-27B, reaches 58.14%, leaving a gap of 30.2 percentage points. The gap is particularly pronounced for perspective taking (PT), where the best non-human result is 51.9% compared with 87.4% for humans, but is much smaller for world knowledge (WK), at 73.44% versus 78.6%. This pattern suggests that current models remain particularly weak on perspectivedependent operations involving viewpoint transformation and cross-view alignment. Moreover, all four evaluated open-source spatial models remain below 50% overall accuracy, indicating that existing spatially specialized models still leave substantial room for improvement on robust cross-view reasoning.

The gains are not accompanied by larger option-dependent disparity. As a descriptive diagnostic, we report the option-wise accuracy range, defined as the maximum difference in conditional accuracy across predicted answer options. Across all four paired executors, SpatialSkill reduces this range: from 27.28 to 23.52 for Qwen3.5-4B, 39.21 to 23.10 for Qwen3.5-9B, 72.12 to 47.94 for Gemma-3-4B, and 30.43 to 19.37 for GPT-5.6-luna. Thus, the overall accuracy gains are not accompanied by greater option-dependent disparity.

## 4.4 ABLATION EXPERIMENTS

Table 2 reports results under the same frozen executor. All ablation settings use the same frozen executor, decoding parameters, validation split, and answer-extraction procedure as the complete system. Evolution-based variants restart from the empty manual and produce a terminal artifact that is fixed before validation.

The Direct baseline answers without a manual or chain-of-thought prompting. The CoT baseline uses chain-of-thought prompting without a manual. SpatialSkill uses the final manual with the category-conditioned policy. The component variants remove one named mechanism at a time: w/o attribution-aware reflection removes the separation between reusable skill-level feedback and executionlevel notes; w/o visual-evidence criterion removes V while retaining $\mathcal { E } ;$ w/o executability criterion removes $\mathcal { E }$ while retaining $\nu ;$ w/o joint admission gate removes both predicates from ${ \dot { \mathcal { G } } } ( r ) ;$ ; w/o growth budget removes the per-window net-growth constraint $\beta ;$ w/o seniority protection removes the protection threshold $\tau ;$ and w/o probe-score ratchet removes the probe-based acceptance rule and is referred to as Naive in the stability analysis.

Table 2: Overall ablations validation accuracy (%) on Qwen3.5-4B and Gemma-3-4B.
<table><tr><td>Condition</td><td>Qwen Gemma</td></tr><tr><td>Direct baseline 47.16 CoT baseline 49.62</td><td>49.24 48.67</td></tr><tr><td>w/o attribution-aware reflection w/o visual-evidence criterion w/o executability criterion w/o joint admission gate w/o growth budget</td><td>50.00 47.35 50.76 50.74 51.14 51.32 50.19 50.95 51.57 51.31</td></tr><tr><td>w/o seniority protection w/o probe-score ratchet</td><td>51.68 51.44 51.14</td></tr><tr><td>SPATIALSKILL 52.08</td><td>50.57 51.89</td></tr></table>

Learned skills improve over no-skill baselines. The direct baseline reaches 47.16% on Qwen3.5- 4B and 49.24% on Gemma-3-4B, while the CoT baseline reaches 49.62% and 48.67% without a manual. SpatialSkill reaches 52.08% and 51.89%, yielding gains of 4.92 and 2.65 percentage points over the corresponding Direct baselines. The gap between CoT and SpatialSkill shows that the improvement cannot be explained by CoT prompting alone.

Attribution-aware reflection is important for reliable skill induction. Removing attributionaware reflection reduces accuracy to 50.00% on Qwen3.5-4B and 47.35% on Gemma-3-4B, a loss of 2.08 and 4.54 points from the complete system. The larger degradation on Gemma suggests that separating reusable and execution-specific feedback is particularly important for this executor. The result suggests that separating reusable skills from execution-specific notes is important for stable reflection.

Grounded admission benefits from both evidence and executability checks. Removing the visual-evidence criterion gives 50.76% and 50.74%, whereas removing the executability criterion gives 51.14% and 51.32% on Qwen3.5-4B and Gemma-3-4B, respectively. Removing the joint admission gate gives 50.19% and 50.95%. All three variants underperform the complete system on both executors. On Qwen3.5-4B, removing both criteria is more harmful than removing either criterion alone; on Gemma-3-4B, the effects are not strictly additive. These results support the util ity of grounded admission while also showing that the relative contribution of the two criteria is executor-dependent.

Stability controls provide complementary protection. Removing growth budget reduces accuracy to 51.57% and 51.31%, while removing seniority protection gives 51.68% and 51.44%. Removing probe-score ratchet produces larger drops of 0.94 points on Qwen3.5-4B and 1.32 points on Gemma-3-4B, reaching 51.14% and 50.57%. This pattern suggests that the ratchet serves as the primary safeguard against harmful revisions, whereas budget and seniority constraints regulate the trajectory prior to acceptance. Their effects are therefore complementary rather than interchangeable.

![](images/bdcfdbaccc61baaf0cea25361b9dceb6670022a122beeffd9ef336bf0a4c75d9.jpg)  
(a) Refactor-event windows.

![](images/2fd866aacfcb570446309b95d705159a008f584adfe8d851dcd0c7b7a3d00c47.jpg)  
(b) Qwen3.5-4B manual size.

![](images/f12c67fed0f76e6a10b59e37671a356ebfce879ecf73249324fbfb6654fc84ea.jpg)  
(c) Gemma-3-4B manual size.  
Figure 3: Stability diagnostics over 43 evolution windows for SpatialSkill and Naive.

## 4.5 STABILITY OF SKILL EVOLUTION

Perceptual spatial reasoning has no symbolic execution oracle, so we measure stability over 43-window trajectories from the same frozen executor and training stream. A refactor event counts when the manual is explicitly refactored or shrinks between adjacent windows. Naive denotes the ablation variant following the same skillevolution loop but removing the probescore ratchet, lacking the safeguard against harmful revision deployment.

The ratchet blocks harmful revisions and reduces unstable refactors. SpatialSkill reduces refactor-event windows from 35 to 6 on Qwen3.5-4B and

Table 3: Routing summary and paired error turnover on the held-out validation split.
<table><tr><td colspan="4">(a) Routing summary</td></tr><tr><td>Executor</td><td>Main Probe replay No-margin</td><td></td><td>Oracle</td></tr><tr><td>Qwen3.5-4B</td><td>52.08 53.03</td><td>51.33</td><td>54.17</td></tr><tr><td>Gemma-3-4B 51.89</td><td>50.95</td><td>51.33</td><td>53.03</td></tr><tr><td colspan="4">(b) Paired error turnover</td></tr><tr><td>Executor</td><td>Base Main Fixed</td><td>Broken</td><td>Net Persistent</td></tr><tr><td>Qwen3.5-4B</td><td>47.1652.08 87</td><td>61 +26</td><td>192</td></tr><tr><td>Gemma-3-4B</td><td>49.24 51.89 70</td><td>56+14</td><td>198</td></tr></table>

from 24 to 8 on Gemma-3-4B in Fig. 3a. The ratchet rejects 14 and 21 revisions, all of which score lower than the accepted reference, with worst drops of 9.0 and 5.5 points. In the evaluated trajectories, no accepted revision decreases the probe score. Manual-size contractions occur in 17 and 16 Naive windows, compared with 6 and 8 SpatialSkill windows shown in Figs. 3b–3c.

Stability translates into better held-out validation. SpatialSkill reaches 52.08% versus 51.14% for Naive on Qwen3.5-4B, and 51.89% versus 50.57% on Gemma-3-4B. Thus, the safeguards do not merely suppress change. The joint admission gate screens candidate rules, while the retained growth and protection controls constrain the trajectory. The ratchet prevents deploying revisions whose probe score falls beyond the tolerance band. Safe compression remains allowed, but harmful revisions are not deployed.

## 4.6 SKILL-AWARE ROUTING ANALYSIS

Routing balances empirical performance and conservative deployment. The analysis-only probe replay reaches 53.03% for Qwen3.5-4B and 50.95% for Gemma-3-4B, compared with the Main results of 52.08% and 51.89%. The No-margin diagnostic uses the same probe-based routing rule with the switching margin set to δ = 0, reaching 51.33% on both executors. The non-deployable validation oracle reaches 54.17% and 53.03% in Table 3 (a). Probe replay applies a fixed finite-probe route to the bank of validation predictions. It diagnoses sensitivity to route estimation. The lower No-margin accuracy suggests that conservative switching can reduce over-selection under finiteprobe uncertainty. Moreover, routed accuracy varies widely across 1,000 resampled 40-example probes in Fig. 4a, which further motivates the default-and-margin rule: under insufficient evidence, the router retains the default instead of selecting a noise-inflated winner.

Optimal delivery is family- and executor-dependent. Qwen3.5-4B favors manual+direct on MR, SR, and WK, and manual+CoT on PT. Gemma-3-4B favors manual+direct on PT and SR, manual+CoT on WK, and the base configuration on CR, as shown in Fig. 4b. The hatched probe-selected cells need not equal the validation-best cells, because validation labels never enter policy construction. This variation supports the multi-referential heterogeneity premise: delivery utility depends on cognitive family and executor.

![](images/ca9e8da89efdf3a3cc296a6faa1865afeb819f65b2ae206ff1c1a9216a08c431.jpg)  
(a) Sensitivity under resampled probes.

![](images/25c83958102dd97b71c719a87eef2bddf9ef89be47eacde94588dc8df854473b.jpg)  
(b) Validation cells with probe selections hatched.

Figure 4: Probe-derived routing diagnostics: full-probe route and validation oracle.  
![](images/097aa40003654248bf56a37f4f32a259a60543e86b119225ba92dacc573a5641.jpg)  
(a) Error turnover by CvSI category.

![](images/8ebfa1038a89c19a49892b881bcbb08efabfa94562c34b9af63962e53cf0b4b4.jpg)  
(b) Output-token expansion by transition.  
Figure 5: Paired error-turnover diagnostics for the frozen predictions.

## 4.7 PAIRED ERROR ANALYSIS

Gains come from targeted repairs rather than uniform correction. We froze base and SpatialSkill predictions on the same validation examples. Qwen3.5-4B repairs 87 baseline errors and introduces 61 new errors, for a net gain of 26. Gemma-3-4B repairs 70 and breaks 56, for a net gain of 14 shown in Table 3 (b). Repairs concentrate in WK and SR for Qwen and in WK and PT for Gemma, whereas 192 and 198 baseline errors persist in Fig. 5a. Thus, the evolved manual improves aggregate accuracy through targeted repairs rather than uniform correction of the baseline error set.

Longer outputs do not indicate correct answers. Conditioned on transition type, the mean SpatialSkill-to-base output-token ratio is largest for newly broken examples (4.74× for Qwen and 8.33× for Gemma), and remains elevated for fixed examples, as shown in Fig. 5b. Length expansion is therefore only a descriptive correlate of the applied reasoning process rather than a reliable indicator of correctness.

## 4.8 CROSS-MODEL TRANSFER WITH SOURCE-ROUTED DELIVERY

We evaluate whether a manual and its category-level delivery policy can transfer to a different executor without target-specific adaptation. For each pair, the target executor, validation split, decoding configuration, and metric remain fixed; the source manual and source-derived routing policy are applied directly to the target. Each pair uses a pre-specified terminal source artifact, and no transfer round is selected using held-out validation performance.

All eight completed cross-model transfers improve over their target bases, with gains ranging from 0.57 to 6.82 percentage points and a mean gain of 2.72 points. Gemma-source gives the largest gain on Qwen3.5-4B, improving from 47.16% to 53.98%, while Qwen-source is strongest on Qwen3.5- 2B, improving from 46.59% to 50.19%. GPT-5.6-luna reaches 55.87% with Qwen-source and 56.82% with Gemma-source. Gemini-3.7-Flash reaches 54.73% and 54.36% with the two source manuals, respectively. The results indicate that source-routed skills can transfer across executor families, but neither source manual is uniformly optimal.

![](images/e42e96f6f67fb08fc3021402ee2c2f1af0f104aca10cc7fb12873cac57a5348e.jpg)  
Figure 6: Own-base and cross-model source-routed validation accuracy. Each transferred manual and its source-derived routing policy is applied to the target executor without target-specific adaptation; N/A denotes an incomplete pairing.

## 4.9 SKILL MANUAL TRANSFER EXPERIMENT

Our external transfer set originates from MMSI-Bench (Yang et al., 2026) as the second dataset. After data cleaning, we use the 1,000 instances for transfer evaluation. We evaluate the frozen executor-specific manual and routing policy without adapting the model or manual.

Table 4: Own-base and frozen manual-routed MMSI-Bench accuracy and option-wise accuracy range. Each executor’s frozen manual and source-derived routing policy is applied to the 1,000- example transfer set .
<table><tr><td>Method</td><td>Overall Acc. (%) ↑ Option-wise Acc. Range (pp) ↓</td></tr><tr><td>Qwen3.5-4B Base 29.30</td><td>10.45</td></tr><tr><td>Qwen3.5-4B Manual + route</td><td>31.30 8.01</td></tr><tr><td>GPT-5.6-luna Base</td><td>34.30 10.62</td></tr><tr><td>GPT-5.6-luna Manual + route</td><td>38.40 3.95</td></tr></table>

This experiment tests transfer across datasets and scene domains rather than transfer between executors. We retain overall accuracy and the option-wise accuracy range used in Table 4.

Both frozen manual policy pairs improve over their bases. Qwen3.5-4B improves by 2.00 points, from 29.30% to 31.30%, while its option-wise accuracy range decreases from 10.45 to 8.01 points. GPT-5.6-luna improves by 4.10 points, from 34.30% to 38.40%, while its option-wise accuracy range decreases from 10.62 to 3.95 points. These single-run external results support the transfer of the frozen delivery artifacts. They further suggest that the learned manual has some transferability to similar spatial-reasoning tasks.

Table 5: Reflection-model sensitivity with Qwen3.5-4B as the frozen executor. Only the reflection model is changed; all other evolution and inference settings are fixed.
<table><tr><td>Reflection Model</td><td>Executor</td><td>Accuracy (%) ↑</td><td>∆ (pp)↑</td></tr><tr><td>None</td><td>Qwen3.5-4B</td><td>47.16</td><td>一</td></tr><tr><td>Qwen3-4B</td><td>Qwen3.5-4B</td><td>50.57</td><td>+3.41</td></tr><tr><td>Qwen3-8B</td><td>Qwen3.5-4B</td><td>49.05</td><td>+1.89</td></tr><tr><td>Qwen3.5-9B</td><td>Qwen3.5-4B</td><td>52.08</td><td>+4.92</td></tr></table>

## 4.10 PAIRED ERROR ANALYSIS

For Qwen3.5-4B and Gemma-3-4B, base and SpatialSkill predictions are paired on the same validation examples. We classify each example as repaired, broken, fixed, or persistent according to the transition from the base prediction to the SpatialSkill prediction. Qwen3.5-4B repairs 87 baseline errors and introduces 61 new errors, for a net gain of 26. Gemma-3-4B repairs 70 baseline errors and introduces 56 new errors, for a net gain of 14. The remaining baseline errors persist in the majority of cases, showing that the manual produces targeted repairs rather than uniform correction.

We also compare generated-token ratios across transition types. The largest SpatialSkill-to-base expansion occurs on newly broken examples, with ratios of 4.74 for Qwen3.5-4B and 8.33 for Gemma-3-4B. Output length is therefore a descriptive correlate of skill-conditioned reasoning, not evidence of correctness or a causal mechanism.

## 4.11 GENERALITY ACROSS REFLECTION MODELS

The main experiments use Qwen3.5-9B as the fixed reflection model for candidate skill generation, admission checking, and manual revision. To examine whether SpatialSkill depends on this particular reflector, we keep the executor fixed as Qwen3.5-4B and repeat the complete skill-evolution pipeline with two alternative reflection models: Qwen3-4B and Qwen3-8B. All other settings, including the evolution stream, probe split, window size, admission criteria, stability controls, routing procedure, and executor decoding configuration, are kept unchanged. Thus, the reflection model is the only experimental variable.

Table 5 reports the results. With the default Qwen3.5-9B reflector, SpatialSkill improves the frozen Qwen3.5-4B executor from 47.16% to 52.08%. Replacing the reflector with Qwen3-4B and Qwen3- 8B yields accuracies of 50.57% and 49.05%, corresponding to gains of +3.41 and +1.89 percentage points over the same frozen executor baseline. The consistent gains across these reflectors suggest that SpatialSkill is not tied to the default reflection model, although performance remains sensitive to reflector choice.

## 5 CONCLUSION

In this paper, we study the self-evolution of spatial skills in frozen vision-language models and propose SpatialSkill, a framework that accumulates explicit natural-language skills without parameter updates. Our analysis highlights three requirements for reliable perceptual skill evolution: grounding candidate skills, stabilizing repeated accumulation, and selectively deploying skills across reasoning categories. Experiments on CityCube show that explicit skill evolution can improve frozen executors while reducing instability across repeated evolution. We further find that the benefit of skill usage varies across spatial-reasoning categories, motivating category-aware skill routing rather than a uniform manual application strategy. The limitation is that current routing relies on precomputed category annotations, and natural-language manuals can improve reasoning strategies but do not modify the executor’s underlying visual capabilities. Future work will explore the use of explicit manuals as fine-tuning priors to combine explicit and implicit knowledge updates.

## REFERENCES

Hunar Batra, Haoqin Tu, Hardy Chen, Yuanze Lin, Cihang Xie, and Ronald Clark. Spatialthinker: Reinforcing 3d reasoning in multimodal llms via spatial rewards. arXiv, pp. 1–29, 2025.

Wenxiao Cai, Iaroslav Ponomarenko, Jianhao Yuan, Xiaoqi Li, Wankou Yang, Hao Dong, and Bo Zhao. Spatialbot: Precise spatial understanding with vision language models. In Proceedings ofthe IEEE International Conference on Robotics and Automation, pp. 9490–9498, 2025.

Boyuan Chen, Zhuo Xu, Sean Kirmani, Brain Ichter, Dorsa Sadigh, Leonidas Guibas, and Fei Xia. Spatialvlm: Endowing vision-language models with spatial reasoning capabilities. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 14455–14465, 2024.

Haolin Du, Jingfei He, and Yuanqing Zhao. Ccr: A counterfactual causal reasoning-based method for cross-view geo-localization. IEEE Transactions on Circuits and Systems for Video Technology, 34(11):11630–11643, 2024.

Huan-ang Gao, Jiayi Geng, Wenyue Hua, Mengkang Hu, Xinzhe Juan, Hongzhang Liu, Shilong Liu, Jiahao Qiu, Xuan Qi, Qihan Ren, Yiran Wu, Hongru Wang, Han Xiao, Yuhang Zhou, Shaokun Zhang, Jiayi Zhang, Jinyu Xiang, Yixiong Fang, Qiwen Zhao, Dongrui Liu, Cheng Qian, Zhenhailong Wang, Minda Hu, Huazheng Wang, Qingyun Wu, Heng Ji, and Mengdi Wang. A survey of self-evolving agents: What, when, how, and where to evolve on the path to artificial super intelligence. arXiv, pp. 1–77, 2025.

Haoxuan Han, Zeyu Zhang, Yefei He, Bohan Zhuang, et al. Less detail, better answers: Degradationdriven prompting for vqa. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 2903–2912, 2026.

Yining Hong, Haoyu Zhen, Peihao Chen, Shuhong Zheng, Yilun Du, Zhenfang Chen, and Chuang Gan. 3d-llm: Injecting the 3d world into large language models. In Proceedings of the Advances in Neural Information Processing Systems, volume 36, pp. 20482–20494, 2023.

Yining Hong, Jiageng Liu, Han Yin, Manling Li, Leonidas J. Guibas, Li Fei-Fei, Jiajun Wu, and Yejin Choi. Esi-bench: Towards embodied spatial intelligence that closes the perception-action loop. arXiv, pp. 1–38, 2026.

Xinmiao Huang, Qisong He, Zhenglin Huang, Boxuan Wang, Zhuoyun Li, Guangliang Cheng, Yi Dong, and Xiaowei Huang. Spatial-dise: A unified benchmark for evaluating spatial reasoning in vision-language models. In Proceedings of the International Conference on Learning Representations, volume 2026, pp. 135833–135865, 2026.

Mengdi Jia, Zekun Qi, Shaochen Zhang, Wenyao Zhang, XinQiang Yu, Jiawei He, He Wang, and Li Yi. Omnispatial: Towards comprehensive spatial reasoning benchmark for vision language models. In International Conference on Learning Representations, volume 2026, pp. 35634– 35670, 2026.

Ruofei Ju, Xinrui Wang, Xin Ding, Yifan Yang, Hao Wu, Shiqi Jiang, Qianxi Zhang, Hao Wen, Xiangyu Li, Weijun Wang, Kun Li, Yunxin Liu, Haipeng Dai, Wei Wang, and Ting Cao. Embodiskill: Skill-aware reflection for self-evolving embodied agents. arXiv, pp. 1–15, 2026.

Jiazheng Kang, Mingming Ji, Zhe Zhao, and Ting Bai. Memory OS of AI agent. In Christos Christodoulopoulos, Tanmoy Chakraborty, Carolyn Rose, and Violet Peng (eds.), Proceedings of the Conference on Empirical Methods in Natural Language Processing, pp. 25961–25970, 2025.

Wajahat Khalid, Bin Liu, Xulin Li, Muhammad Waqas, and Muhammad Sher Afgan. Bridging the sky and ground: Towards view-invariant feature learning for aerial-ground person reidentification. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pp. 9749–9758, 2025.

Phillip Y Lee, Jihyeon Je, Chanho Park, Mikaela Angelina Uy, Leonidas Guibas, and Minhyuk Sung. Perspective-aware reasoning in vision-language models via mental imagery simulation. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pp. 9241–9251, 2025.

Dingming Li, Hongxing Li, Zixuan Wang, Yuchen Yan, Hang Zhang, Siqi Chen, Guiyang Hou, Shengpei Jiang, Wenqiao Zhang, Yongliang Shen, Weiming Lu, and Yueting Zhuang. Viewspatial-bench: Evaluating multi-perspective spatial localization in vision-language models. arXiv, pp. 1–17, 2025.

Fuhao Li, Wenxuan Song, Han Zhao, Jingbo Wang, Pengxiang Ding, Donglin Wang, Long Zeng, and Haoang Li. Spatial forcing: Implicit spatial representation alignment for vision-languageaction model. In Proceedings of the International Conference on Learning Representations, volume 2026, pp. 132324–132345, 2026a.

Haoyuan Li, Qihang Cao, Tao Tang, Kun Xiang, Zihan Guo, Jianhua Han, JiaWang Bian, Hang Xu, and Xiaodan Liang. Thinking with geometry: Active geometry integration for spatial reasoning. arXiv, pp. 1–19, 2026b.

Zongzhao Li, Zongyang Ma, Mingze Li, Songyou Li, Yu Rong, Tingyang Xu, Ziqi Zhang, Deli Zhao, and Wenbing Huang. Star-r1: Multi-view spatial transformation reasoning by reinforcing multimodal llms. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 12041–12051, 2026c.

Junming Liu, Yuqi Li, Yifei Sun, Maonan Wang, Piotr Koniusz, Yirong Chen, and Ding Wang. Selfevolving spatial reasoning in vision language models via geometric logic consistency. arXiv, pp. 1–23, 2026a.

Weichen Liu, Qiyao Xue, Haoming Wang, Xiangyu Yin, Boyuan Yang, and Wei Gao. Spatial reasoning in multimodal large language models: A survey of tasks, benchmarks and methods. arXiv, pp. 1–34, 2025.

Yuhong Liu, Beichen Zhang, Yuhang Zang, Yuhang Cao, Long Xing, Xiaoyi Dong, Haodong Duan, Dahua Lin, and Jiaqi Wang. Spatial-ssrl: Enhancing spatial understanding via self-supervised reinforcement learning. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 9570–9581, 2026b.

Wufei Ma, Luoxin Ye, Celso M de Melo, Alan Yuille, and Jieneng Chen. Spatialllm: A compound 3d-informed design towards spatially-intelligent large multimodal models. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 17249–17260, 2025.

Arjun Majumdar, Anurag Ajay, Xiaohan Zhang, Pranav Putta, Sriram Yenamandra, Mikael Henaff, Sneha Silwal, Paul McVay, Oleksandr Maksymets, Sergio Arnaud, Karmesh Yadav, Qiyang Li, Ben Newman, Mohit Sharma, Vincent-Pierre Berges, Shiqi Zhang, Pulkit Agrawal, Yonatan Bisk, Dhruv Batra, Mrinal Kalakrishnan, Franziska Meier, Chris Paxton, Alexander Sax, and Aravind Rajeswaran. Openeqa: Embodied question answering in the era of foundation models. In Proceedings ofthe IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 16488– 16498, 2024.

Kun Ouyang, Yuanxin Liu, Haoning Wu, Yi Liu, Hao Zhou, Jie Zhou, Fandong Meng, and Xu Sun. Spacer: Reinforcing mllms in video spatial reasoning. arXiv, pp. 1–16, 2025.

Kunyu Peng, Zhikun Zhou, Kailun Yang, Di Wen, Ruiping Liu, Yufan Chen, Junwei Zheng, Hao Shi, Yi Zhou, M. Saquib Sarfraz, Danda Pani Paudel, and Luc Van Gool. Seeing together: Multirobot cooperative egocentric spatial reasoning with multimodal large language models. arXiv, pp. 1–20, 2026.

Viraj Prabhu, Yutong Dai, Matthew Fernandez, Krithika Ramakrishnan, Jing Gu, Yanqi Luo, silvio savarese, Caiming Xiong, Junnan Li, Zeyuan Chen, and Ran Xu. Walt: Web agents that learn tools. In Proceedings ofthe International Conference on Learning Representations, volume 2026, pp. 55589–55607, 2026.

Yuming Qiao, Liang Luo, Dan Meng, Yifan Yang, Qingyuan Wang, Juntuo Wang, Yuwei Zhang, Ru Zhen, Yanhao Zhang, Haonan Lu, et al. Aligning cross-view visual geometries in lvlms through human-like reasoning learning. In Proceedings of the AAAI Conference on Artificial Intelligence, volume 40, pp. 19345–19353, 2026.

Delin Qu, Haoming Song, Qizhi Chen, Yuanqi Yao, Xinyi Ye, Yan Ding, Zhigang Wang, JiaYuan Gu, Bin Zhao, Dong Wang, and Xuelong Li. Spatialvla: Exploring spatial representations for visual-language-action model. arXiv, pp. 1–20, 2025.

Shuaike Shen, Wenduo Cheng, Mingqian Ma, Alistair Turcan, Martin Jinye Zhang, and Jian Ma. Skillfoundry: Building self-evolving agent skill libraries from heterogeneous scientific resources. arXiv, pp. 1–19, 2026.

Ilias Stogiannidis, Steven McDonagh, and Sotirios A. Tsaftaris. Mind the gap: Benchmarking spatial reasoning in vision-language models. arXiv, pp. 1–14, 2025.

Haotong Sun, Jianye Xie, Bocheng Xu, and Yinghui Jiang. Know thyself, know thy user: Intrinsic dual-perspective reasoning for role-playing llms. In Proceedings of the International Conference on Machine Learning, pp. 1–33, 2026.

Gemma Team. Gemma 4 technical report. arXiv, pp. 1–17, 2026a.

InternVL Team. Internvl3: Exploring advanced training and test-time recipes for open-source multimodal models. arXiv, pp. 1–27, 2025a.

Qwen Team. Qwen3-vl technical report. arXiv, pp. 1–42, 2025b.

Qwen Team. Qwen3.5: Towards native multimodal agents, 2026b. URL https://qwen.ai/ blog?id=qwen3.5.

Shi-Yu Tian, Zhuo-Xia Wang, Xuan-Yi Zhu, Zhi Zhou, Xinwei Yang, Kun-Yang Yu, Ming Yang, Yang Chen, and Yu-Feng Li. Self-evolving neuro-symbolic skills for tool-augmented spatial reasoning. arXiv, pp. 1–25, 2026.

Jingqi Tong, Yurong Mou, Hangcheng Li, Mingzhe Li, Yongzhuo Yang, Ming Zhang, Qiguang Chen, Tianyi Liang, Xiaomeng Hu, Yining Zheng, et al. Thinking with video: Video generation as a promising multimodal reasoning paradigm. In Proceedings ofthe IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 41121–41129, 2026.

Hengyi Wang, Ruiqiang Zhang, Chang Liu, Guanjie Wang, Zehua Ma, Han Fang, and Weiming Zhang. Allocentric perceiver: Disentangling allocentric reasoning from egocentric visual priors via frame instantiation. arXiv, pp. 1–22, 2026.

Xiaokun Wang, Peiyu Wang, Jiangbo Pei, Wei Shen, Yi Peng, Yunzhuo Hao, Weijie Qiu, Ai Jian, Tianyidan Xie, Xuchen Song, Yang Liu, and Yahui Zhou. Skywork-vl reward: An effective reward model for multimodal understanding and reasoning. arXiv, pp. 1–13, 2025a.

Zhihu Wang, Shiwan Zhao, Yu Wang, Heyuan Huang, Sitao Xie, Yubo Zhang, Jiaxin Shi, Zhixing Wang, Hongyan Li, and Junchi Yan. Re-task: Revisiting llm tasks from capability, skill, and knowledge perspectives. In Proceedings of the Findings of the Association for Computational Linguistics, pp. 4925–4936, 2025b.

Junfei Wu, Jian Guan, Kaituo Feng, Qiang Liu, Shu Wu, Liang Wang, Wei Wu, and Tieniu Tan. Reinforcing spatial reasoning in vision-language models with interwoven thinking and visual drawing. In Proceedings of the Advances in Neural Information Processing Systems, volume 38, pp. 143297–143330, 2025.

Peng Xia, Jianwen Chen, Hanyang Wang, Jiaqi Liu, Kaide Zeng, Yu Wang, Siwei Han, Yiyang Zhou, Xujiang Zhao, Haifeng Chen, Zeyu Zheng, Cihang Xie, and Huaxiu Yao. Skillrl: Evolving agents via recursive skill-augmented reinforcement learning. In Proceedings of the International Conference on Learning Representations Workshop on Memory for LLM-Based Agentic Systems, pp. 1–20, 2026.

Kun Xiang, Terry Jingchen Zhang, Yinya Huang, Jixi He, Zirong Liu, Yueling Tang, Ruizhe Zhou, Lijing Luo, Youpeng Wen, Xiuwei Chen, Bingqian Lin, Jianhua Han, Hang Xu, Hanhui Li, Bin Dong, and Xiaodan Liang. Aligning perception, reasoning, modeling and interaction: A survey on physical ai. IEEE Transactions on Pattern Analysis and Machine Intelligence, 48(10):12938– 12957, 2026.

Haotian Xu, Yue Hu, Zhengqiu Zhu, Chen Gao, Ziyou Wang, Junreng Rao, Wenhao Lu, Weishi Li, Quanjun Yin, and Yong Li. Citycube: Benchmarking cross-view spatial reasoning on visionlanguage models in urban environments. In Proceedings ofthe Annual Meeting ofthe Association for Computational Linguistics, pp. 9190–9215, 2026a.

Weiye Xu, Jiahao Wang, Weiyun Wang, Zhe Chen, Wengang Zhou, Aijun Yang, Lewei Lu, Houqiang Li, Xiaohua Wang, Xizhou Zhu, et al. Visulogic: A benchmark for evaluating visual reasoning in multi-modal large language models. In Proceedings ofthe International Conference on Learning Representations, volume 2026, pp. 25966–26003, 2026b.

Wujiang Xu, Zujie Liang, Kai Mei, Hang Gao, Juntao Tan, and Yongfeng Zhang. A-mem: Agentic memory for LLM agents. In Proceedings of the Advances in Neural Information Processing Systems, volume 38, pp. 17577–17604, 2025.

Jihan Yang, Shusheng Yang, Anjali W. Gupta, Rilyn Han, Li Fei-Fei, and Saining Xie. Thinking in space: How multimodal large language models see, remember, and recall spaces. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 10632–10643, 2025a.

Sihan Yang, Runsen Xu, Yiman Xie, Sizhe Yang, Mo Li, Jingli Lin, Chenming Zhu, Xiaochen Chen, Haodong Duan, Xiangyu Yue, Dahua Lin, Tai Wang, and Jiangmiao Pang. Mmsi-bench: A benchmark for multi-image spatial intelligence. In Proceedings of the International Conference on Learning Representations, volume 2026, pp. 157051–157088, 2026.

Yi Yang, Xiaoxuan He, Hongkun Pan, Xiyan Jiang, Yan Deng, Xingtao Yang, Haoyu Lu, Dacheng Yin, Fengyun Rao, Minfeng Zhu, et al. R1-onevision: Advancing generalized multimodal reasoning through cross-modal formalization. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pp. 2376–2385, 2025b.

Pratham Yashwante and Rose Yu. Time series, vision, and language: Exploring the limits of alignment in contrastive representation spaces. arXiv, pp. 1–32, 2026.

Xiaoyu Zhan, Wenxuan Huang, Hao Sun, Xinyu Fu, Changfeng Ma, Shaosheng Cao, Bohan Jia, Shaohui Lin, Zhenfei Yin, Lei Bai, Wanli Ouyang, Yuanqi Li, Jie Guo, and Yanwen Guo. Actial: Activate spatial reasoning ability of multimodal large language models. In Proceedings of the Advances in Neural Information Processing Systems, 2025.

Qizheng Zhang, Changran Hu, Shubhangi Upasani, Boyuan Ma, Fenglu Hong, Vamsidhar Kamanuru, Jay Rainton, Chen Wu, Mengmeng Ji, Hanchen Li, Urmish Thakker, James Zou, and Kunle Olukotun. Agentic context engineering: Evolving contexts for self-improving language models. In Proceedings of the International Conference on Learning Representations, volume 2026, pp. 86069–86100, 2026a.

Wenyu Zhang, Wei En Ng, Lixin Ma, Yuwen Wang, Junqi Zhao, Allison Koenecke, Boyang Li, and Lu Wang. Sphere: Unveiling spatial blind spots in vision-language models through hierarchical evaluation. In Proceedings of the Annual Meeting of the Association for Computational Linguistics, pp. 11591–11609, 2025.

Yuyou Zhang, Radu Corcodel, Chiori Hori, Anoop Cherian, and Ding Zhao. Spinbench: Perspective and rotation as a lens on spatial reasoning in vlms. In Proceedings ofthe International Conference on Learning Representations, volume 2026, pp. 70072–70141, 2026b.

Ruosen Zhao, Zhikang Zhang, Jialei Xu, Jiahao Chang, Dong Chen, Lingyun Li, Weijian Sun, and Zizhuang Wei. Spacemind: Camera-guided modality fusion for spatial reasoning in visionlanguage models. arXiv, pp. 1–12, 2025.

Boyuan Zheng, Michael Y. Fatemi, Xiaolong Jin, Zora Zhiruo Wang, Apurva Gandhi, Yueqi Song, Yu Gu, Jayanth Srinivasa, Gaowen Liu, Graham Neubig, and Yu Su. Skillweaver: Web agents can self-improve by discovering and honing skills. arXiv, pp. 1–40, 2025.

Heng Zhou, Li Kang, Yiran Qin, Xiufeng Song, Ao Yu, Zilu Zhang, Haoming Song, Kaixin Xu, Yuchen Fan, Dongzhan Zhou, Xiaohong Liu, Ruimao Zhang, Philip Torr, Lei Bai, and Zhenfei Yin. Ego to world: Collaborative spatial reasoning in embodied systems via reinforcement learning. arXiv, pp. 1–28, 2026.

Zhongyi Zhou, Yichen Zhu, Minjie Zhu, Junjie Wen, Ning Liu, Zhiyuan Xu, Weibin Meng, Yaxin Peng, Chaomin Shen, Feifei Feng, et al. Chatvla: Unified multimodal understanding and robot control with vision-language-action model. In Proceedings ofthe Conference on Empirical Methods in Natural Language Processing, pp. 5377–5395, 2025.
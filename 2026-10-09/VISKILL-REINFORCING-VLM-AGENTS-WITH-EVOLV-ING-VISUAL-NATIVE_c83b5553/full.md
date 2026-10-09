# VISKILL: REINFORCING VLM AGENTS WITH EVOLV-ING VISUAL-NATIVE SKILLS

Hongxing Li<sup>1</sup>, Dingming Li<sup>1</sup>, Yixin Li<sup>1</sup>, Yong Du<sup>1</sup>, Wenqi Zhang<sup>1</sup>, Weiming Lu<sup>1</sup>, Jun Xiao<sup>1</sup>, Yueting Zhuang<sup>1</sup>, Yongliang Shen<sup>1†</sup>

<sup>1</sup>Zhejiang University

{hongxing.li, syl}@zju.edu.cn

## ABSTRACT

Skill-augmented agents improve sample efficiency by distilling successful trajectories into reusable strategies. Yet most existing approaches remain text-centric, linearizing spatial layouts and action–state correspondences into language that loses critical geometric structure. Recent efforts have begun incorporating visual evidence, but construct and update skills separately from policy optimization, leaving their mutual improvement underexplored. We propose ViSkill, a visualnative skill learning framework that encodes successful interactions as composite visual skill cards directly accessible to VLM agents. Retrieved skills guide both inference and reward shaping, while successful trajectories are distilled back into the library, forming a closed feedback loop in which skill accumulation and policy improvement reinforce each other. An optional cold-start mechanism further accelerates early-stage learning. Evaluated on Sokoban, FrozenLake, and PrimitiveSkill, ViSkill achieves an overall success rate of 0.89, rising to 0.91 with cold-start initialization, outperforming all evaluated proprietary and open-source baselines while converging faster than standard PPO. Our code is available at https://github.com/ZJU-REAL/ViSkill.

## 1 INTRODUCTION

Reinforcement learning (RL) (Schulman et al., 2017; Guo et al., 2025; Ahmadian et al., 2024) has become a central paradigm for training Vision-Language Model (VLM) agents that interact with complex, multi-turn environments (Zhai et al., 2024; Wang et al., 2026a). Under standard RL, successful strategies are absorbed only implicitly into policy parameters, requiring the agent to rediscover similar spatial solutions across episodes (Feng et al., 2026; Wang et al., 2026c; Dong et al., 2026). Yet successful trajectories already encode reusable procedural knowledge, such as how to navigate around obstacles or how to coordinate multi-step object manipulations, that could directly benefit future tasks. This motivates a natural complement to parameterbased learning: explicitly retaining and reusing successful experience across episodes.

Skill-augmented agents realize this by distilling successful trajectories into reusable strategies and retrieving them for future tasks (Wang et al., 2023; Zhao et al., 2024; Xia et al., 2026; Shi et al., 2026; Zhang et al., 2026b), and have shown con-

![](images/1bb41561dc1c2bd22bff2490b4bbc6253dde95d6aff8c3b77569f6d60bc3526c.jpg)

![](images/fac4d13a2964620d37de0b5b180fd84ba36f1bb57744b1d65ce4a360bffc382b.jpg)

![](images/4425b5da60995d11d975b714b5b09e29f08f071362a42a26dbb79f6ed76acb10.jpg)  
Figure 1: Text-centric skills linearize spatial experience into language, causing lossy abstraction. ViSkill instead represents reusable experience as visual skill cards that preserve spatial structure natively.

sistent gains in both sample efficiency and task performance. However, two fundamental gaps limit their effectiveness in visually grounded environments, where, unlike text-based settings, the agent must interpret spatial layouts, track object relationships, and reason about how actions induce visual state transitions (Yang et al., 2025; Zhai et al., 2024). The first is a representation gap: existing skills are predominantly text-centric (Ni et al., 2026; Yang et al., 2026; Lin et al., 2026), lineariz ing spatial layouts, relative positions, and action–state correspondences into language, which risks losing geometric information critical for strategy execution, as shown in Figure 1. The second is a learning gap: although recent work has begun incorporating visual evidence into skill construction and utilization (Zhang et al., 2026a; Jiang et al., 2026; Liu et al., 2026; Wang et al., 2026b), these approaches typically decouple skill construction from policy optimization, leaving the bidirectional interplay between visual skill acquisition and policy learning largely underexplored.

We introduce ViSkill, a visual-native skill learning framework that couples reusable visual experience with reinforcement optimization for VLM agents. Each skill is stored as a composite visual skill card that renders annotated trajectory frames alongside a policy-distilled strategy description into a unified image, directly preserving the spatial and procedural context of successful experience. Given the current task state, ViSkill retrieves skills by matching geometric configurations and injects the full skill card into the VLM’s visual context. The retrieved skill further provides a skill-guided reward that assigns partial credit to failed trajectories that show skill-relevant progress. After each episode, successful trajectories that pass quality and novelty gates are distilled back into the library as new visual skills. To bootstrap this loop, a cold-start mechanism optionally pre-populates the library with a small set of diverse, solver-derived seed skills. Together, these mechanisms form a closed feedback loop: retrieved skills improve the policy through in-context demonstration and reward shaping, while the improved policy contributes higher-quality skills back to the library.

We evaluate ViSkill on Sokoban, FrozenLake, and PrimitiveSkill, spanning discrete 2D spatial planning and coordinate-grounded 3D manipulation. ViSkill outperforms all evaluated proprietary and open-source baselines with an overall success rate of 0.89, rising to 0.91 with cold-start initialization. Beyond final performance, ViSkill converges faster and achieves higher success rates throughout training, reflecting the positive feedback loop between skill accumulation and policy improvement. Ablations further confirm that visual skill representations consistently outperform text-centric alternatives, and that skills distilled under joint policy optimization transfer more effectively than those accumulated without it.

Overall, our contributions are threefold:

• We propose ViSkill, a visual-native skill learning framework that represents reusable procedural experience as composite visual skill cards, preserving spatial and procedural context in a form directly accessible to VLM agents.

• We develop geometry-aware skill retrieval and a skill-guided reinforcement objective that form a closed feedback loop, where retrieved skills guide policy optimization and successful trajectories are distilled back into the library.

• We empirically demonstrate across 2D spatial planning and 3D manipulation that ViSkill outperforms proprietary and open-source baselines, converging faster and achieving higher final success rates than standard PPO.

## 2 RELATED WORK

## 2.1 REINFORCEMENT LEARNING FOR VLM AGENTS

Recent work has extended reinforcement learning from language-model alignment to visionlanguage reasoning, using verifiable rewards to improve visual perception and reasoning (Liu et al., 2025a; Shen et al., 2025; Li et al., 2026a; Huang et al., 2026a; Li et al., 2026b). Beyond single-turn prediction, several studies formulate VLMs as sequential decision-making agents and optimize them through interaction with visual environments (Zhai et al., 2024; Liu et al., 2025b; Zheng et al., 2026; Wang et al., 2026a), addressing challenges such as long-horizon credit assignment and sparse reward in visually grounded tasks (Wang et al., 2025; Feng et al., 2026; Wang et al., 2026c). However, interaction experience is typically consumed as transient rollout data and absorbed implicitly into policy parameters. Rather than treating experience solely as a training signal, ViSkill externalizes successful trajectories into an explicit library of retrievable visual-native skills, allowing accumulated experience to guide future inference and provide dense rewards that facilitate policy optimization.

![](images/ac9b033771f95db800bc86a9a86994139dd18a9dfc172042383e43b3df6b29aa.jpg)  
Figure 2: Overview of ViSkill. The agent retrieves a visual skill $z _ { k }$ ⋆ from the skill library to guide interaction, collects trajectory τ, and distills eligible trajectories into new skills. The skill-guided reward $R ( \tau )$ and PPO jointly optimize the policy.

## 2.2 SKILL-AUGMENTED AGENTS

Skills provide a modular mechanism for transferring procedural knowledge across episodes. Most existing approaches distill interaction trajectories into natural-language instructions or structured workflows organized into persistent libraries for retrieval and reuse (Shinn et al., 2023; Zhao et al., 2024; Wang et al., 2023), with recent work further incorporating such libraries into reinforcement learning (Xia et al., 2026; Shi et al., 2026; Lu et al., 2026). However, these approaches remain text-centric, linearizing spatial layouts and action–state correspondences that are central to visually grounded decision making. Recent methods have begun incorporating visual information into skill construction and memory (Jiang et al., 2026; Zhang et al., 2026a; Huang et al., 2026b; Liu et al., 2026). AtlasVA (Wang et al., 2026b) further evolves spatial atlases from interaction statistics for dense reward shaping. Within this broader direction, ViSkill represents reusable procedures as composite visual skill cards and couples their online distillation with policy optimization.

## 3 METHODOLOGY

## 3.1 SKILL-INTEGRATED AGENT LOOP

Figure 2 provides an overview of ViSkill. For each task $x \sim \mathcal { D }$ , ViSkill proceeds through three stages: skill retrieval, skill-guided interaction, and online skill distillation. A complete interaction trajectory is represented as ${ \boldsymbol { \tau } } = ( x , z , \left( o _ { t } , a _ { t } \right) _ { t = 1 } ^ { T } )$ , where z denotes the retrieved visual skill and $( o _ { t } , a _ { t } ) _ { t = 1 } ^ { T }$ denotes the sequence of observations and actions. During interaction, the policy $\pi _ { \theta }$ generates actions conditioned on the task, interaction history, and the retrieved skill. Upon termination, the environment provides a binary success indicator $y ( \tau ) \in \{ 0 , 1 \}$ , which serves as the outcome reward for policy optimization, updates the utility of the retrieved skill, and determines whether the trajectory is eligible for distillation into a new skill.

## 3.2 VISUAL-NATIVE SKILL LIBRARY

Visual-Native Skill Representation. Textual skill descriptions often lose the spatial and procedural structure needed for visually grounded decision making. To preserve this information in a form directly accessible to VLMs, we maintain a visual skill library $B = \left\{ z _ { k } \right\} _ { k = 1 } ^ { | B | }$ , where each skill $z _ { k }$ is constructed from a successful interaction trajectory $\hat { \tau } _ { k }$ through a structured rendering process. We retain the original visual observations, augment them with trajectory annotations to clarify spatial and procedural progression, and organize them into a unified visual sequence. A strategy description $d _ { k } ,$ , distilled by the policy from $\hat { \tau } _ { k } ,$ is rendered as text within the same image to capture the high-level intent of the skill. The annotated trajectory and rendered strategy are then composed into a single composite image $C _ { k }$ , which is directly provided to the VLM as visual context. Formally:

$$
z _ { k } = \bigl ( C _ { k } , g _ { k } , u _ { k } , n _ { k } \bigr ) , \qquad C _ { k } = \mathrm { R e n d e r } \bigl ( d _ { k } , \mathrm { A n n o t a t e } \bigl ( \hat { \tau } _ { k } \bigr ) \bigr ) ,\tag{1}
$$

where $C _ { k }$ denotes the composite skill card, $g _ { k }$ is the geometric descriptor used for skill retrieval, and $u _ { k }$ and $n _ { k }$ are its utility and usage count, respectively.

Skill Retrieval and Utilization. Text-centric skill libraries often retrieve skills through languagelevel matching between the current instruction and textual skill descriptions (Shi et al., 2026; Xia et al., 2026; Zhang et al., 2025). In visually grounded tasks, however, similar instructions may correspond to different spatial layouts that require distinct strategies. ViSkill therefore determines skill relevance through geometric similarity rather than language matching, while incorporating historical utility to account for skill reliability. At the beginning of each episode, an environment-specific feature extractor maps the initial task configuration to a geometric descriptor $^ { g , }$ capturing spatial relations such as relative positions, distances, and obstacles. For each skill $z _ { k } ,$ we compute a composite score:

$$
q _ { k } = \alpha s _ { k } + ( 1 - \alpha ) u _ { k } ,\tag{2}
$$

where $s _ { k } = \sin ( g , g _ { k } )$ is the geometric similarity and $u _ { k }$ is the historical utility. We first identify the highest-scoring skill as $k ^ { \star } = \arg \operatorname* { m a x } _ { k } q _ { k }$ . The corresponding skill $z _ { k ^ { \star } }$ is retrieved only $\mathrm { i f } q _ { k ^ { \star } } \geq \delta _ { r } ;$ otherwise, the policy proceeds without skill guidance. The retrieved composite card $C _ { k } ,$ is directly appended to the VLM’s visual context. After termination, the utility and usage count are updated as follows:

$$
u _ { k ^ { \star } } \gets ( 1 - \beta ) u _ { k ^ { \star } } + \beta y ( \tau ) , \qquad n _ { k ^ { \star } } \gets n _ { k ^ { \star } } + 1 ,\tag{3}
$$

where $\beta$ is the EMA coefficient and $y ( \tau )$ is the binary success indicator.

Skill Distillation and Library Update. Indiscriminately storing every successful trajectory risks introducing suboptimal and redundant skills, degrading both the quality and diversity of the library. After each episode, a successful trajectory is therefore considered for distillation only if it passes both quality and novelty gates:

$$
L ( \tau ) \leq L _ { \mathrm { b u d } } ( x ) , \qquad q _ { \mathrm { m a x } } < \delta _ { d } ,\tag{4}
$$

where $L _ { \mathrm { b u d } } ( x )$ is the trajectory-length budget for task $x ,$ and $q _ { \operatorname* { m a x } } \ = \ \operatorname* { m a x } _ { z _ { k } \in B } q _ { k }$ is the bestmatch score used as a proxy for library coverage. The quality gate ensures that only high-quality trajectories are distilled into the library; we instantiate ${ \cal L } _ { \mathrm { b u d } } ( x ) = { \cal L } ^ { \star } ( x )$ using a task-specific solver. The novelty gate admits a trajectory only when no existing skill sufficiently covers the current configuration; for an empty library we define $q _ { \operatorname* { m a x } } = 0$

For each eligible trajectory, the policy produces a reusable high-level strategy description by summarizing its own successful interaction, which is then rendered together with the annotated trajectory to construct a new skill $z _ { \mathrm { n e w } }$ . The new skill is added to the library with an initial utility and usage count. When the library reaches its capacity, the skill with the lowest utility–frequency score is evicted:

$$
k _ { \mathrm { e v i c t } } = \arg \operatorname* { m i n } _ { k } u _ { k } \log ( n _ { k } + 1 ) .\tag{5}
$$

## 3.3 SKILL-GUIDED REINFORCEMENT OPTIMIZATION

Cold-Start Initialization. When starting from an empty library, early-stage exploration lacks skill guidance, delaying the positive feedback loop between skill distillation and policy learning. To address this cold-start problem, ViSkill optionally pre-populates the library with a diverse set of seed skills before training begins. Candidate trajectories are generated using offline task-specific solvers with strategy descriptions distilled by an external model, and selected through farthest-point sampling in the geometric feature space to maximize initial coverage. These seed skills serve only as an initialization: once training begins, all subsequent skills are distilled entirely by the policy from its own successful trajectories. The cold-start library size is set to $N _ { \mathrm { c o l d } } = \lfloor \gamma N _ { \mathrm { c o n v } } \rfloor$ , where $\gamma \in [ 0 , 1 ]$ is the cold-start ratio and $N _ { \mathrm { c o n v } }$ is the converged library size from a preliminary run without coldstart initialization; setting $\gamma = 0$ disables external-model initialization, providing a flexible trade-off between early-stage sample efficiency and independence from external-model supervision.

Algorithm 1 Training Procedure of ViSkill   
Require: Policy $\pi _ { \theta } .$ skill library B, task distribution D, library capacities $N _ { \mathrm { c o l d } } , N _ { \mathrm { m a x } }$ , thresholds $\delta _ { r } , \delta _ { d }$   
1: if cold-start initialization is enabled then   
2: B ← COLDSTART $( N _ { \mathrm { c o l d } } )$ ▷ Cold-start initialization   
3: end if   
4: for batch of M tasks $\{ x _ { i } \} _ { i = 1 } ^ { M } \sim \mathcal { D }$ do   
5: for sample $i = 1 , \ldots , M$ do ▷ Skill-integrated agent loop   
6: g<sub>i</sub> ← FEATUREEXTRACTOR(x<sub>i</sub>); (z<sub>i</sub>, q<sub>max</sub>) ← RETRIEVE(g<sub>i</sub>, B, δ<sub>r</sub>)   
7: $\tau _ { i } \sim \pi _ { \theta } ( \cdot \mid x _ { i } , z _ { i } )$ ▷ Skill retrieval and utilization   
8: $R ( \tau _ { i } ) \longleftarrow \dot { \mathrm { S K I L L G U I D E D R E W A R D } } ( \tau _ { i } , z _ { i } )$ ▷ Skill-guided reward   
9: $\mathcal { B } \gets \mathrm { U P D A T E U T I L I T Y } \big ( \mathcal { B } , z _ { i } , y ( \tau _ { i } \big ) \big )$   
10: $\mathbf { i f } y ( \tau _ { i } ) = 1$ and PASSGATES(τ<sub>i</sub>, x<sub>i</sub>, q<sub>max</sub>, $\delta _ { d } )$ then   
11: z<sub>new,i</sub> ← DISTILLVISUALSKILL(π<sub>θ</sub>, τ<sub>i</sub>, g<sub>i</sub>)   
12: ${ \mathcal { B } } \gets \mathrm { U P D A T E L I B R A R Y } \left( { \mathcal { B } } , z _ { \mathrm { n e w } , i } , N _ { \mathrm { m a x } } \right)$ ▷ Skill distillation and library update   
13: end if   
14: end for   
15: θ ← PPOUPDATE $: ( \theta , \{ \tau _ { i } , R ( \tau _ { i } ) \} _ { i = 1 } ^ { M } )$ ▷ Policy optimization   
16: end for

Skill-Guided Reward. Task-success rewards provide little learning signal for failed trajectories, even when the agent partially follows a useful retrieved skill. We therefore introduce an auxiliary reward based on the agreement between the agent trajectory and the retrieved skill. Let ℓ denote the furthest point along the reference trajectory that the agent trajectory visits, and let $L ( \tau )$ and $L _ { z }$ denote the lengths of the executed and reference trajectories. The trajectory agreement is defined as:

$$
r _ { \mathrm { s k i l l } } ( \tau , z ) = \operatorname* { m a x } \left( 0 , \frac { \ell - ( L ( \tau ) - \ell ) } { L _ { z } } \right) .\tag{6}
$$

The final reward combines the environment reward $r _ { \mathrm { e n v } } ( \tau )$ , which includes formatting and taskcompletion signals, with the skill-guidance term:

$$
R ( \tau ) = \left\{ \begin{array} { l l } { r _ { \mathrm { e n v } } ( \tau ) + \lambda r _ { \mathrm { s k i l l } } ( \tau , z ) , } & { y ( \tau ) = 0 \mathrm { a n d } z \mathrm { i s ~ r e t r i e v e d } , } \\ { r _ { \mathrm { e n v } } ( \tau ) , } & { \mathrm { o t h e r w i s e } , } \end{array} \right.\tag{7}
$$

where λ controls the strength of skill guidance. Let $r _ { \mathrm { s u c c } } > 0$ denote the terminal bonus for task completion. Since $r _ { \mathrm { s k i l l } } \in [ 0 , 1 ]$ and $\lambda <  { r _ { \mathrm { s u c c } } }$ , the skill-guidance reward is always smaller than the completion bonus, providing bounded partial credit for skill-aligned failed trajectories without distorting the global reward ordering.

Policy Optimization. Given the resulting trajectory rewards, we optimize the VLM policy π<sub>θ</sub> using Proximal Policy Optimization (PPO) (Schulman et al., 2017). Denote by τ¯ the tokenized representation of trajectory τ with N tokens, and let $\rho _ { i } ( \theta ) = \pi _ { \theta } ( \bar { \tau } _ { i } \mid \bar { \tau } _ { < i } ) / \pi _ { \mathrm { o l d } } ( \bar { \tau } _ { i } \mid \bar { \tau } _ { < i } )$ be the probability ratio between the current and rollout policies. The clipped PPO objective is:

$$
J _ { \mathrm { P P O } } ( \theta ) = \frac { 1 } { \sum _ { i = 1 } ^ { N } m _ { i } } \sum _ { i = 1 } ^ { N } m _ { i } \operatorname* { m i n } ( \rho _ { i } ( \theta ) A _ { i } , \exp ( \rho _ { i } ( \theta ) , 1 - \epsilon , 1 + \epsilon ) A _ { i } ) ,\tag{8}
$$

where $m _ { i }$ masks policy-generated tokens from observation tokens, $A _ { i }$ is the advantage estimated using generalized advantage estimation, and ϵ is the clipping coefficient. Optimizing this objective drives policy improvement while the skill library grows concurrently through distillation, enabling the two components to co-evolve as interaction experience accumulates.

The complete training procedure is summarized in Algorithm 1.

## 4 EXPERIMENTS

## 4.1 EXPERIMENTAL SETUP

Environments. We evaluate ViSkill on three visually grounded interactive environments spanning 2D spatial reasoning and 3D embodied manipulation, adopted from VAGEN (Wang et al., 2026a). Sokoban requires the agent to push all boxes onto target locations; FrozenLake requires it to navigate to a goal while avoiding holes. Both environments provide grid-based visual observations and discrete directional actions. PrimitiveSkill evaluates coordinate-grounded manipulation in third-person 3D scenes, comprising the Place, Stack, Drawer, and Align tasks from VAGEN and the Swap task introduced by AtlasVA (Wang et al., 2026b). We report success rate across all environments.

<table><tr><td rowspan="2">Method</td><td rowspan="2">Sokoban</td><td rowspan="2">FrozenLake</td><td colspan="6">PrimitiveSkill</td><td rowspan="2">Overall</td></tr><tr><td>Place</td><td>Stack</td><td>Drawer</td><td>Align</td><td>Swap</td><td>Average</td></tr><tr><td>Proprietary Models</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>GPT-5</td><td>0.70</td><td>0.77</td><td>1.00</td><td>0.63</td><td>0.00</td><td>1.00</td><td>0.55</td><td>0.64</td><td>0.70</td></tr><tr><td>03</td><td>0.60</td><td>0.78</td><td>1.00</td><td>0.63</td><td>0.00</td><td>1.00</td><td>0.82</td><td>0.69</td><td>0.69</td></tr><tr><td>04-mini</td><td>0.44</td><td>0.82</td><td>1.00</td><td>0.50</td><td>0.00</td><td>0.75</td><td>0.33</td><td>0.52</td><td>0.59</td></tr><tr><td>GPT-40</td><td>0.43</td><td>0.54</td><td>0.50</td><td>0.63</td><td>0.00</td><td>0.88</td><td>0.94</td><td>0.59</td><td>0.52</td></tr><tr><td>Gemini 2.5 Pro</td><td>0.58</td><td>0.78</td><td>0.63</td><td>0.63</td><td>0.00</td><td>0.75</td><td>0.00</td><td>0.40</td><td>0.59</td></tr><tr><td>Gemini 2.0</td><td>0.28</td><td>0.61</td><td>0.75</td><td>0.13</td><td>0.00</td><td>0.25</td><td>0.32</td><td>0.29</td><td>0.39</td></tr><tr><td>Claude 4.5 Sonnet</td><td>0.31</td><td>0.80</td><td>0.63</td><td>0.50</td><td>0.00</td><td>1.00</td><td>1.00</td><td>0.63</td><td>0.58</td></tr><tr><td>Claude 3.7 Sonnet</td><td>0.25</td><td>0.69</td><td>0.63</td><td>0.13</td><td>0.00</td><td>1.00</td><td>0.90</td><td>0.53</td><td>0.49</td></tr><tr><td>Open-Source Models</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Qwen2.5-VL-72B</td><td>0.18</td><td>0.44</td><td>1.00</td><td>0.50</td><td>0.00</td><td>1.00</td><td>0.33</td><td>0.57</td><td>0.40</td></tr><tr><td>Qwen2.5-VL-7B</td><td>0.13</td><td>0.14</td><td>0.00</td><td>0.00</td><td>0.00</td><td>0.75</td><td>0.03</td><td>0.16</td><td>0.14</td></tr><tr><td>Qwen2.5-VL-3B</td><td>0.14</td><td>0.14</td><td>0.00</td><td>0.00</td><td>0.00</td><td>0.00</td><td>0.00</td><td>0.00</td><td>0.09</td></tr><tr><td>VLM-R1-3B</td><td>0.13</td><td>0.13</td><td>0.00</td><td>0.00</td><td>0.00</td><td>0.00</td><td>0.00</td><td>0.00</td><td>0.09</td></tr><tr><td>VAGEN</td><td>0.79</td><td>0.74</td><td>1.00</td><td>0.88</td><td>1.00</td><td>1.00</td><td>0.50</td><td>0.88</td><td>0.80</td></tr><tr><td>Atlas-VA</td><td>0.79</td><td>0.83</td><td>1.00</td><td>1.00</td><td>1.00</td><td>1.00</td><td>1.00</td><td>1.00</td><td>0.87</td></tr><tr><td>ViSkill (Ours)</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>ViSkill</td><td>0.82</td><td>0.84</td><td>1.00</td><td>1.00</td><td>1.00</td><td>1.00</td><td>1.00</td><td>1.00</td><td>0.89</td></tr><tr><td>ViSkill + Cold-Start</td><td>0.88</td><td>0.85</td><td>1.00</td><td>1.00</td><td>1.00</td><td>1.00</td><td>1.00</td><td>1.00</td><td>0.91</td></tr></table>

Table 1: Success rates on Sokoban, FrozenLake, and PrimitiveSkill. Bold and underlined entries indicate the best and second-best results, respectively.

Baselines. We compare ViSkill against representative proprietary and open-source VLMs. Proprietary baselines include frontier models from OpenAI, Google, and Anthropic (OpenAI, 2025a;b; Hurst et al., 2024; Google, 2025; 2024; Anthropic, 2025b;a). For open-source models, we evaluate the Qwen2.5-VL family at 3B, 7B, and 72B scales (Bai et al., 2025), along with three reinforcementlearning-enhanced 3B baselines. VLM-R1 applies R1-style reinforcement learning with rule-based rewards to enhance general visual reasoning (Shen et al., 2025). VAGEN trains multi-turn VLM agents by explicitly reinforcing world-model reasoning (Wang et al., 2026a). AtlasVA (Wang et al., 2026b) is a memory-augmented RL baseline that evolves danger and affinity heatmaps from interaction statistics and uses them for dense spatial reward shaping. For a controlled comparison, VAGEN, AtlasVA, and ViSkill share the same Qwen2.5-VL-3B backbone and environment seeds. ViSkill follows the training horizon of VAGEN, which is shorter than that of AtlasVA.

Implementation Details. We use Qwen2.5-VL-3B-Instruct as the policy backbone, following the PPO configuration of VAGEN (Wang et al., 2026a) with a rollout batch size of 128, a PPO mini-batch size of 32, and actor and critic learning rates of $1 \times 1 0 ^ { - 6 }$ and $1 \times 1 0 ^ { - 5 }$ . Sokoban and FrozenLake are trained for 300 steps on 4× H100 GPUs; PrimitiveSkill is trained jointly for 60 steps on 2× H100 GPUs with an additional H100 GPU dedicated to the ManiSkill (Mu et al., 2021) environment server. We set the similarity weight $\alpha = 0 . 9$ , EMA coefficient $\beta = 0 . 1$ , skill-guidance weight $\lambda = 0 . 8$ , novelty threshold $\delta _ { d } = 0 . 9$ , retrieval threshold $\delta _ { r } = 0 . 8 .$ , and cold-start ratio $\gamma = 0 . 2$ , with GPT-5.4-mini (OpenAI, 2026) serving as the external model for cold-start strategy distillation.

## 4.2 MAIN RESULTS

ViSkill outperforms all proprietary and open-source baselines in overall success rate, achieving 0.89 (Table 1). On the 2D environments, it obtains 0.82 on Sokoban and 0.84 on FrozenLake, surpassing both VAGEN (0.79, 0.74), which relies on world-model reasoning, and AtlasVA (0.79, 0.83), which employs spatially grounded reward shaping. On PrimitiveSkill, ViSkill achieves a perfect average success rate of 1.00, matching AtlasVA and outperforming VAGEN by 12 percentage points. These results demonstrate that explicitly retaining and reusing successful visual experience provides an effective learning signal while also offering actionable guidance at both training and inference time, enabling the policy to ground its decisions in previously successful strategies across both discrete 2D planning and coordinate-grounded 3D manipulation.

![](images/b3c2b46a91e6acd48b73a6e377552a17c36ff318563fd2b4438ea251628ef71e.jpg)

![](images/d1a3bbf0b81b940e9087c81007fb09af22c0bc637775440e20b1175d0672e337.jpg)

![](images/d4c8ea2550c90e4bcdaa8e1524e65dd4bd2df5e5f798d97ebd220d6ba1810553.jpg)

Figure 3: Training dynamics of PPO, ViSkill, and ViSkill + Cold-Start across Sokoban, FrozenLake, and PrimitiveSkill. ViSkill converges faster and achieves higher success rates than PPO across all environments, with cold-start initialization providing additional gains in early-stage learning.
<table><tr><td>Method</td><td>Sokoban</td><td>FrozenLake</td></tr><tr><td>ViSkill</td><td>0.82</td><td>0.84</td></tr><tr><td>Text-Skill</td><td> $0 . 6 3 _ { - 0 . I 9 }$ </td><td> $0 . 5 9 _ { - 0 . 2 5 }$ </td></tr><tr><td>OCR-Skill</td><td> $0 . 6 8 _ { - 0 . I 4 }$ </td><td> $0 . 6 2 _ { - 0 . 2 2 }$ </td></tr></table>

<table><tr><td>Method</td><td></td><td>Sokoban FrozenLake</td></tr><tr><td>ViSkill w/o Skill-Guided Reward</td><td> $0 . 7 7 _ { - 0 . 0 5 }$ </td><td> $0 . 8 0 _ { - 0 . 0 4 }$ </td></tr><tr><td>ViSkill w/o Strategy</td><td> $0 . 7 2 _ { - 0 . I O }$ </td><td> $0 . 8 1 _ { - 0 . 0 3 }$ </td></tr><tr><td>ViSkill w/o Optimization</td><td> $0 . 2 9 _ { - 0 . 5 3 }$ </td><td> $0 . 3 3 _ { - 0 . 5 I }$ </td></tr></table>

Table 2: Comparison of skill representations on Sokoban and FrozenLake.  
Table 3: Ablation of ViSkill components on Sokoban and FrozenLake.

Cold-start initialization further raises the overall success rate from 0.89 to 0.91, exceeding AtlasVA (0.87) by 4 percentage points. The gains are concentrated in the 2D environments: performance improves from 0.82 to 0.88 on Sokoban and from 0.84 to 0.85 on FrozenLake, while PrimitiveSkill remains saturated at 1.00. The larger improvement on Sokoban suggests that diverse seed skills are particularly beneficial for combinatorial configurations, reducing unguided early-stage exploration and providing broader initial coverage for subsequent policy learning.

## 4.3 TRAINING DYNAMICS

Figure 3 presents the training dynamics of PPO, ViSkill, and ViSkill + Cold-Start across all three environments. ViSkill consistently converges faster and reaches higher success rates than PPO, demonstrating the benefit of retaining and reusing successful visual experience as an explicit training signal. Cold-start initialization further accelerates early-stage learning: on Sokoban, ViSkill + Cold-Start achieves substantially higher success rates from the first few steps and maintains its lead throughout training, suggesting that diverse seed skills provide broader initial coverage for combinatorial configurations. On PrimitiveSkill, both ViSkill variants converge to 1.00 significantly faster than PPO, with cold-start initialization providing a notable advantage in the earliest training steps.

## 4.4 ABLATION STUDY

Skill Representation. Since PrimitiveSkill is saturated by the full model, we conduct controlled ablations on Sokoban and FrozenLake. Table 2 compares three skill representation strategies. Text-Skill and OCR-Skill, which encode successful strategies as raw text and OCR-rendered images respectively, both underperform ViSkill by a substantial margin across Sokoban and FrozenLake . This suggests that preserving the original visual trajectory as the skill representation is essential, as it retains the spatial and structural context of successful strategies that text-based representations inevitably lose, even when rendered back into image form.

ViSkill Component. Table 3 ablates the three core components of ViSkill. Removing Skill Guided Reward leads to moderate performance drops (−0.05 on Sokoban, −0.04 on FrozenLake), indicating that the skill-derived reward signal provides a useful but not dominant training contribution. Removing Strategy causes a larger drop on Sokoban (−0.10), suggesting that high-level strategic guidance is particularly beneficial for combinatorial planning tasks. Removing Optimiza tion yields largest degradation (−0.53 on Sokoban, −0.51 on FrozenLake), where skill retrieval and distillation are retained but model training is disabled. This highlights the importance of the co-evolution between policy optimization and skill learning: training enables the model to better leverage retrieved skills, while an improving policy in turn contributes higher-quality skills to the library, creating a mutually reinforcing cycle that is essential for effective learning.

![](images/2314130ebeee45eb8803a5c41cafbe9e71e971364d09b2aa3be1e5b5a7c25f25.jpg)

Figure 4: Impact of the retrieval threshold $\delta _ { r }$ and distillation diversity threshold $\delta _ { d }$ on success rate in Sokoban and FrozenLake.  
![](images/99a86f8edcb7dd90c82b3c7014f4d9ef2d744cfbfd2f089a9cc611146f465cd4.jpg)  
Figure 5: Skill-library dynamics under different distillation diversity thresholds $\delta _ { d }$ , measured by library size and average utility.

Skill-Guided Reward Weight. Table 4 studies the sensitivity to the skill-guided reward weight λ. Performance peaks at $\lambda = 0 . 8$ on both environments, achieving 0.82 on Sokoban and 0.84 on FrozenLake. Setting $\lambda = 0 . 0$ removes the skill-guided signal entirely and leads to a drop of 0.05 on Sokoban, while setting $\lambda = 1 . 0$ over-emphasizes progress toward retrieved reference states and similarly degrades performance. An intermediate value of $\lambda = 0 .$ 8 strikes the best balance, effectively complementing the environment reward with retrieved skill guidance without suppressing policy exploration.

<table><tr><td>λ</td><td>0.0</td><td>0.6</td><td>0.8</td><td>1.0</td></tr><tr><td>Sokoban</td><td>0.77</td><td>0.71</td><td>0.82</td><td>0.77</td></tr><tr><td>FrozenLake</td><td>0.80</td><td>0.79</td><td>0.84</td><td>0.81</td></tr></table>

Table 4: Impact of λ on success rate.

Retrieval Threshold. Figure 4 (a) shows the effect of the retrieval threshold $\delta _ { r }$ on success rate. Performance peaks at $\delta _ { r } = 0 . 8 0$ on both Sokoban (0.82) and FrozenLake (0.84). A lower threshold $( \delta _ { r } ~ = ~ 0 . 7 5 )$ admits skills whose geometric states are poorly matched to the current observation, introducing misaligned guidance that degrades performance. A higher threshold $( \delta _ { r } ~ = ~ 0 . 8 5 )$ is overly restrictive, reducing the frequency of successful retrievals and limiting the influence of skill guidance during training. These results confirm that $\delta _ { r } = 0 . 8 0$ strikes the right balance between retrieval relevance and coverage.

Distillation Diversity Threshold. Figure 4 (b) and Figure 5 examine the effect of the novelty threshold $\delta _ { d } ,$ which gates library admission based on the similarity between a new trajectory and existing skills. A low threshold $( \delta _ { d } = 0 . 8 5 )$ is overly permissive: the library grows rapidly throughout training, but the influx of redundant skills causes average utility to decline steadily, ultimately hurting success rate. A high threshold $( \delta _ { d } = 0 . 9 5 )$ enforces strict diversity, keeping utility high but severely limiting library growth and reducing the range of scenarios the library can cover. The intermediate setting $\delta _ { d } = 0 . 9 0$ maintains a moderate library size with consistently high utility, achieving the best success rate on both environments.

## 4.5 FURTHER ANALYSIS

Cold-Start Budget. Figure 6 investigates how the size of the cold-start library affects final performance. The cold-start ratio γ scales the library size relative to $N _ { \mathrm { c o n v } }$ , the converged library size from a run without cold-start initialization. As γ increases, performance improves consistently on both environments, indicating that a richer initial skill library accelerates and strengthens the feedback loop between skill distillation and policy learning. On Sokoban, performance saturates at $\gamma = 0 . 6$ with a success rate of 1.00, while FrozenLake continues to improve, reaching 0.89 at $\gamma = 1 . 0$ . These results suggest that investing in a higher-quality cold-start library yields reliable gains, particularly in environments with more diverse task configurations.

![](images/198ed8a233428d21b67b8e7893e3523f2aa9b19c05520acfe9a3bd5d381a87e4.jpg)

![](images/875b3975647c9b6d33641aceff3f2c5264c7a2dabf237329627b878340ef9a55.jpg)  
Figure 6: Impact of the cold-start ratio γ on success rate in Sokoban and FrozenLake.

![](images/03d2aa04d442e7aa5f249264bbda40365e4def2ffb01024f5ee8aff8613a4ec8.jpg)  
Figure 7: Success rate and token efficiency of visual skills at different card resolutions on Sokoban and FrozenLake, compared against the Text-Skill and OCR-Skill baselines.

Visual Skill Efficiency. Figure 7 examines the trade-off between skill card resolution and token efficiency. The scaling factor is applied to both the height and width of the skill card image, producing lower-resolution representations that consume fewer tokens. Across both environments, success rate increases monotonically with scale, with ViSkill (1.0) achieving the highest performance. No tably, even the most compact variant, ViSkill (0.2), surpasses both Text-Skill and OCR-Skill on Sokoban while consuming the fewest tokens, and on FrozenLake, ViSkill (0.2) and ViSkill (0.4) similarly outperform both baselines with substantially lower token consumption. This demonstrates that visual skill representations remain effective even at significantly reduced resolutions, offering a favorable efficiency-performance trade-off over text-based alternatives.

Evolved Skill Library Transfer. Figure 8 examines whether the skill library accumulated during training can transfer to the unoptimized base model. We compare two libraries: one distilled with joint policy optimization and one without. Augmenting the base model with either library yields consistent gains over the no-skill baseline, confirming that evolved skills are reusable beyond the trained policy. Skills accumulated with joint policy optimization further improve success rate to 0.41 on Sokoban and 0.39 on FrozenLake, outperforming those distilled without optimization. This demonstrates that the co-evolution of policy and skill library produces higher-quality skills: an improving policy generates more successful and diverse trajectories and strategies, which in turn enrich the library with more effective and transferable skills.

![](images/7d603870669e8a4829dec8e0315a13932522d3989266186d5c758f5b8152af51.jpg)

![](images/2381c9a58278e7637e86eea414ca796591b7570f8abc3858f82b7503ec1e0fba.jpg)  
Figure 8: Transfer performance of evolved skill libraries on the unoptimized base model, with and without joint policy optimization.

## 5 CONCLUSION

We present ViSkill, a visual-native skill learning framework that represents reusable procedural experience as composite visual skill cards and couples skill retrieval with reinforcement optimization through a closed feedback loop. By preserving spatial and geometric structure in a form directly accessible to VLM agents, ViSkill addresses the representation gap of text-centric skill approaches, while skill-guided rewards and online distillation enable policy improvement and skill accumulation to co-evolve. Experiments on Sokoban, FrozenLake, and PrimitiveSkill demonstrate consistent gains over proprietary and open-source baselines, and ablations confirm that visual representations, strategic guidance, and joint optimization each contribute to the positive feedback loop.

Limitations. ViSkill is currently evaluated on environments supported by the VAGEN framework; whether visual skill cards provide consistent benefits in more complex or open-ended settings remains to be validated. The skill-guided reward further assumes that useful partial progress reaches states represented by the retrieved skill, which may provide limited guidance when valid solutions traverse disjoint state paths.

## AI USE STATEMENT

We used generative AI tools to assist with implementation, figure plotting, and manuscript drafting and editing. All AI-assisted content was reviewed and verified by the authors, who take full responsibility for the final work.

## ETHICS STATEMENT

This work is conducted exclusively in simulated environments and involves no human participants, personal data, or physical systems. Real-world deployment would require task-specific safety constraints and human oversight. Trajectory-quality and novelty gates mitigate the accumulation of low-quality behaviors, though they do not replace safety validation in open-ended applications.

## REPRODUCIBILITY STATEMENT

We have made substantial efforts to ensure reproducibility. Appendix A details the geometric descriptors, similarity functions, trajectory-length budgets, skill rendering, library management, and reward and PPO hyperparameters. Appendix B reports environment configurations, training and evaluation seed ranges, and training hyperparameters. Appendix C provides the complete prompt templates for strategy distillation and environment interaction.

## REFERENCES

Arash Ahmadian, Chris Cremer, Matthias Galle, Marzieh Fadaee, Julia Kreutzer, Olivier Pietquin,´ Ahmet Ust<sup>¨</sup> un, and Sara Hooker. Back to basics: Revisiting reinforce-style optimization for learn-¨ ing from human feedback in llms. In Proceedings ofthe 62nd Annual Meeting ofthe Association for Computational Linguistics (Volume 1: Long Papers), pp. 12248–12267, 2024.

Anthropic. Claude 3.7 Sonnet and Claude Code, 2025a. URL https://www.anthropic. com/news/claude-3-7-sonnet.

Anthropic. Introducing Claude Sonnet 4.5, 2025b. URL https://www.anthropic.com/ news/claude-sonnet-4-5.

Shuai Bai, Keqin Chen, Xuejing Liu, Jialin Wang, Wenbin Ge, Sibo Song, Kai Dang, Peng Wang, Shijie Wang, Jun Tang, Humen Zhong, Yuanzhi Zhu, Mingkun Yang, Zhaohai Li, Jianqiang Wan, Pengfei Wang, Wei Ding, Zheren Fu, Yiheng Xu, Jiabo Ye, Xi Zhang, Tianbao Xie, Zesen Cheng, Hang Zhang, Zhibo Yang, Haiyang Xu, and Junyang Lin. Qwen2.5-vl technical report, 2025. URL https://arxiv.org/abs/2502.13923.

Guanting Dong, Hangyu Mao, Kai Ma, Licheng Bao, Yifei Chen, Zhongyuan Wang, Zhongxia Chen, Jiazhen Du, Huiyang Wang, Fuzheng Zhang, et al. Agentic reinforced policy optimization. In International Conference on Learning Representations, volume 2026, pp. 16981–17017, 2026.

Lang Feng, Zhenghai Xue, Tingcong Liu, and Bo An. Group-in-group policy optimization for llm agent training. Advances in Neural Information Processing Systems, 38:46375–46408, 2026.

Google. Introducing gemini 2.0: Our new AI model for the agentic era, 2024. URL https://blog.google/innovation-and-ai/models-and-research/ google-deepmind/google-gemini-ai-update-december-2024/.

Google. Gemini 2.5: Our most intelligent AI model, 2025. URL https://blog. google/innovation-and-ai/models-and-research/google-deepmind/ gemini-model-thinking-updates-march-2025/.

Daya Guo, Dejian Yang, Haowei Zhang, Junxiao Song, Peiyi Wang, Qihao Zhu, Runxin Xu, Ruoyu Zhang, Shirong Ma, Xiao Bi, et al. Deepseek-r1: Incentivizing reasoning capability in llms via reinforcement learning. arXiv preprint arXiv:2501.12948, 2025.

Wenxuan Huang, Bohan Jia, Shaosheng Cao, Zheyu Ye, Zhe Xu, Yao Hu, Shaohui Lin, et al. Visionr1: Incentivizing reasoning capability in multimodal large language models. In International Conference on Learning Representations, volume 2026, pp. 63794–63812, 2026a.

Zihan Huang, Junda Wu, Tong Yu, Qianqi Yan, Rohan Surana, Uttaran Bhattacharya, Lina Yao, Xin Eric Wang, and Julian McAuley. Skill-cmib: Multimodal agent skill for consistent action via conditional multimodal information bottleneck. arXiv preprint arXiv:2605.08526, 2026b.

Aaron Hurst, Adam Lerer, Adam P Goucher, Adam Perelman, Aditya Ramesh, Aidan Clark, AJ Ostrow, Akila Welihinda, Alan Hayes, Alec Radford, et al. Gpt-4o system card. arXiv preprint arXiv:2410.21276, 2024.

Guanyu Jiang, Zhaochen Su, Xiaoye Qu, and Yi R Fung. Xskill: Continual learning from experience and skills in multimodal agents. arXiv preprint arXiv:2603.12056, 2026.

Hongxing Li, Xiufeng Huang, Dingming Li, Wenjing Jiang, Zixuan Wang, Haolei Xu, Hanrong Zhang, Haiwen Hong, Longtao Huang, Hui Xue, et al. Perceive-to-reason: Decoupling perception and reasoning for fine-grained visual reasoning. arXiv preprint arXiv:2607.01191, 2026a.

Hongxing Li, Dingming Li, Zixuan Wang, Yuchen Yan, Hang Wu, Wenqi Zhang, Yongliang Shen, Weiming Lu, Jun Xiao, and Yueting Zhuang. Spatialladder: Progressive training for spatial reasoning in vision-language models. In International Conference on Learning Representations, volume 2026, pp. 76566–76592, 2026b.

Huawei Lin, Peng Li, Jie Song, Fuxin Jiang, and Tieying Zhang. Muse-autoskill: Self-evolving agents via skill creation, memory, management, and evaluation. arXiv preprint arXiv:2605.27366, 2026.

Zhou Liu, Ligang Huang, Zeli Su, Zewei Pan, Zhaoyang Han, Xing Chen, Yuanfeng Song, and Wentao Zhang. Skilllens: Visual skill cards for retrieval-augmented gui action prediction and on-policy distillation. arXiv preprint arXiv:2608.10775, 2026.

Ziyu Liu, Zeyi Sun, Yuhang Zang, Xiaoyi Dong, Yuhang Cao, Haodong Duan, Dahua Lin, and Jiaqi Wang. Visual-rft: Visual reinforcement fine-tuning. In 2025 IEEE/CVF International Conference on Computer Vision (ICCV), pp. 2034–2044. IEEE, 2025a.

Ziyu Liu, Yuhang Zang, Yushan Zou, Zijian Liang, Xiaoyi Dong, Yuhang Cao, Haodong Duan, Dahua Lin, and Jiaqi Wang. Visual agentic reinforcement fine-tuning. arXiv preprint arXiv:2505.14246, 2025b.

Zhengxi Lu, Zhiyuan Yao, Jinyang Wu, Chengcheng Han, Qi Gu, Xunliang Cai, Weiming Lu, Jun Xiao, Yueting Zhuang, and Yongliang Shen. Skill0: In-context agentic reinforcement learning for skill internalization. arXiv preprint arXiv:2604.02268, 2026.

Tongzhou Mu, Zhan Ling, Fanbo Xiang, Derek Yang, Xuanlin Li, Stone Tao, Zhiao Huang, Zhiwei Jia, and Hao Su. Maniskill: Generalizable manipulation skill benchmark with large-scale demonstrations. arXiv preprint arXiv:2107.14483, 2021.

Jingwei Ni, Yihao Liu, Xinpeng Liu, Yutao Sun, Mengyu Zhou, Pengyu Cheng, Dexin Wang, Erchao Zhao, Xiaoxi Jiang, and Guanjun Jiang. Trace2skill: Distill trajectory-local lessons into transferable agent skills. arXiv preprint arXiv:2603.25158, 2026.

OpenAI. Introducing GPT-5, 2025a. URL https://openai.com/index/ introducing-gpt-5/.

OpenAI. Introducing OpenAI o3 and o4-mini, 2025b. URL https://openai.com/index/ introducing-o3-and-o4-mini/.

OpenAI. Introducing GPT-5.4 mini and nano, 2026. URL https://openai.com/index/ introducing-gpt-5-4-mini-and-nano/.

Maxime Oquab, Timothee Darcet, Th´ eo Moutakanni, Huy Vo, Marc Szafraniec, Vasil Khalidov,´ Pierre Fernandez, Daniel Haziza, Francisco Massa, Alaaeldin El-Nouby, et al. Dinov2: Learning robust visual features without supervision. arXiv preprint arXiv:2304.07193, 2023.

John Schulman, Filip Wolski, Prafulla Dhariwal, Alec Radford, and Oleg Klimov. Proximal policy optimization algorithms. arXiv preprint arXiv:1707.06347, 2017.

Haozhan Shen, Peng Liu, Jingcheng Li, Chunxin Fang, Yibo Ma, Jiajia Liao, Qiaoli Shen, Zilun Zhang, Kangjia Zhao, Qianqian Zhang, et al. Vlm-r1: A stable and generalizable r1-style large vision-language model. arXiv preprint arXiv:2504.07615, 2025.

Guangming Sheng, Chi Zhang, Zilingfeng Ye, Xibin Wu, Wang Zhang, Ru Zhang, Yanghua Peng, Haibin Lin, and Chuan Wu. Hybridflow: A flexible and efficient rlhf framework. In Proceedings of the Twentieth European Conference on Computer Systems, pp. 1279–1297, 2025.

Yaorui Shi, Yuxin Chen, Zhengxi Lu, Yuchun Miao, Shugui Liu, Qi Gu, Xunliang Cai, Xiang Wang, and An Zhang. Skill1: Unified evolution of skill-augmented agents via reinforcement learning. arXiv preprint arXiv:2605.06130, 2026.

Noah Shinn, Federico Cassano, Ashwin Gopinath, Karthik Narasimhan, and Shunyu Yao. Reflexion: Language agents with verbal reinforcement learning. Advances in neural information processing systems, 36:8634–8652, 2023.

Guanzhi Wang, Yuqi Xie, Yunfan Jiang, Ajay Mandlekar, Chaowei Xiao, Yuke Zhu, Linxi Fan, and Anima Anandkumar. Voyager: An open-ended embodied agent with large language models. arXiv preprint arXiv:2305.16291, 2023.

Kangrui Wang, Pingyue Zhang, Zihan Wang, Yaning Gao, Linjie Li, Qineng Wang, Hanyang Chen, Yiping Lu, Zhengyuan Yang, Lijuan Wang, et al. Vagen: Reinforcing world model reasoning for multi-turn vlm agents. Advances in Neural Information Processing Systems, 38:172871–172933, 2026a.

Pan Wang, Yihao Hu, Xiujin Liu, Jingchu Yang, Hang Wang, and Zhihao Wen. Atlasva: Selfevolving visual skill memory for teacher-free vlm agents. arXiv preprint arXiv:2605.17933, 2026b.

Zihan Wang, Kangrui Wang, Qineng Wang, Pingyue Zhang, Linjie Li, Zhengyuan Yang, Xing Jin, Kefan Yu, Minh Nhat Nguyen, Licheng Liu, et al. Ragen: Understanding self-evolution in llm agents via multi-turn reinforcement learning. arXiv preprint arXiv:2504.20073, 2025.

Zixuan Wang, Yuchen Yan, Hongxing Li, Teng Pan, Dingming Li, Ruiqing Zhang, Weiming Lu, Jun Xiao, Yueting Zhuang, and Yongliang Shen. Milestone-guided policy learning for long-horizon language agents. arXiv preprint arXiv:2605.06078, 2026c.

Peng Xia, Jianwen Chen, Hanyang Wang, Jiaqi Liu, Kaide Zeng, Yu Wang, Siwei Han, Yiyang Zhou, Xujiang Zhao, Haifeng Chen, et al. Skillrl: Evolving agents via recursive skill-augmented reinforcement learning. arXiv preprint arXiv:2602.08234, 2026.

Rui Yang, Hanyang Chen, Junyu Zhang, Mark Zhao, Cheng Qian, Kangrui Wang, Qineng Wang, Teja Venkat Koripella, Marziyeh Movahedi, Manling Li, et al. Embodiedbench: Comprehensive benchmarking multi-modal large language models for vision-driven embodied agents. arXiv preprint arXiv:2502.09560, 2025.

Yifan Yang, Ziyang Gong, Weiquan Huang, Qihao Yang, Ziwei Zhou, Zisu Huang, Yan Li, Xuemei Gao, Qi Dai, Bei Liu, et al. Skillopt: Executive strategy for self-evolving agent skills. arXiv preprint arXiv:2605.23904, 2026.

Yuexiang Zhai, Hao Bai, Zipeng Lin, Jiayi Pan, Shengbang Tong, Yifei Zhou, Alane Suhr, Saining Xie, Yann LeCun, Yi Ma, et al. Fine-tuning large vision-language models as decisionmaking agents via reinforcement learning. Advances in neural information processing systems, 37:110935–110971, 2024.

Kangning Zhang, Shuai Shao, Qingyao Li, Jianghao Lin, Lingyue Fu, Shijian Wang, Wenxiang Jiao, Yuan Lu, Weiwen Liu, Weinan Zhang, et al. Mmskills: Towards multimodal skills for general visual agents. arXiv preprint arXiv:2605.13527, 2026a.

Xiaoying Zhang, Zichen Liu, Yipeng Zhang, Xia Hu, and Wenqi Shao. Retroagent: From solving to evolving via retrospective dual intrinsic feedback. arXiv preprint arXiv:2603.08561, 2026b.

Yanzhao Zhang, Mingxin Li, Dingkun Long, Xin Zhang, Huan Lin, Baosong Yang, Pengjun Xie, An Yang, Dayiheng Liu, Junyang Lin, et al. Qwen3 embedding: Advancing text embedding and reranking through foundation models. arXiv preprint arXiv:2506.05176, 2025.

Andrew Zhao, Daniel Huang, Quentin Xu, Matthieu Lin, Yong-Jin Liu, and Gao Huang. Expel: Llm agents are experiential learners. In Proceedings of the AAAI Conference on Artificial Intelligence, volume 38, pp. 19632–19642, 2024.

Ziwei Zheng, Minghao Yang, Jack Hong, Chenxiao Zhao, Guohai Xu, Le Yang, and Chao Shen. Deepeyes: Incentivizing” thinking with images” via reinforcement learning. In International Conference on Learning Representations, volume 2026, pp. 126775–126798, 2026.

![](images/8d283a324db4f4f4695d1c88a323fa66ad0bd784e1573c3c1bff00e7efb49de4.jpg)  
Figure 9: Geometry-based skill retrieval on three held-out Sokoban configurations. Each row shows a query state alongside its top-five retrieved skills, ordered by composite score, with the selected skill highlighted in green.

## A METHODOLOGY DETAILS

Geometric Features and Similarity. Language-level task descriptions alone do not reliably identify a transferable skill in spatial environments: identical goals may require different action sequences under different layouts. We therefore retrieve skills using geometric descriptors derived from the initial configuration, capturing directional relations, distances, and local obstacle structure between task-relevant entities.

For discrete environments (Sokoban and FrozenLake), we partition the descriptor into semantically meaningful blocks and compute similarity as a weighted average of their exact-match rates:

$$
\sin ( g , g _ { k } ) = \sum _ { c \in \mathcal { C } } w _ { c } \left( \frac { 1 } { | g ^ { ( c ) } | } \sum _ { r = 1 } ^ { | g ^ { ( c ) } | } \mathbb { I } \left[ g _ { r } ^ { ( c ) } = g _ { k , r } ^ { ( c ) } \right] \right) ,\tag{9}
$$

where C denotes the set of descriptor blocks and $\textstyle \sum _ { c \in { \mathcal { C } } } w _ { c } = 1$ . For Sokoban, descriptors encode the player–box and box–target spatial relations together with local wall indicators; for FrozenLake, they encode the start–goal relation and hole indicators along directions connecting the start and goal.

Figure 9 visualizes retrieval behavior on three held-out Sokoban configurations. In each case, the top-ranked skill exhibits the closest spatial arrangement, while lower-ranked candidates share progressively fewer task-relevant geometric relations.

For PrimitiveSkill, descriptors contain the xy coordinates of task-relevant objects and a task identifier; skills are never retrieved across distinct manipulation tasks. Within the same task, we use a continuous similarity:

$$
\sin ( g , g _ { k } ) = \operatorname* { m a x } \_ { } \left( 0 , 1 - \frac { \operatorname* { m a x } _ { j } | | p _ { j } - p _ { k , j } | | _ { 2 } } { d _ { \operatorname* { m a x } } } \right) ,\tag{10}
$$

where $p _ { j }$ and $p _ { k , j }$ are the xy coordinates of corresponding objects in the current and retrieved configurations, and $d _ { \mathrm { m a x } }$ is a normalization constant.

![](images/b9fe4789a1ea1be49423a2e73b322e5cb1bcf2bf9db251065b4b710c4047d622.jpg)  
Figure 10: Representative visual skill cards from Sokoban, FrozenLake, and PrimitiveSkill. Each card combines a rendered strategy description with annotated trajectory frames and action labels into a single composite image.

Trajectory-Length Budgets and Solvers. The trajectory-length budget prevents successful but unnecessarily long trajectories from entering the skill library. In general, the quality gate uses a task-dependent budget $L _ { \mathrm { b u d } } ( x )$ ; in our experiments we instantiate it with the shortest available solution length, $L _ { \mathrm { b u d } } ( x ) = L ^ { \star } ( x )$ , so a trajectory is admitted only if its length does not exceed the corresponding reference.

For Sokoban and FrozenLake, we obtain $L ^ { \star } ( x )$ via breadth-first search over the grid state space, treating holes as impassable cells for FrozenLake. For PrimitiveSkill, each manipulation task has a fixed reference length given by its scripted oracle plan: 4 actions for Place, Stack, and Align, 3 for Drawer, and 6 for Swap. The solvers are used solely to provide trajectory-quality budgets and optional cold-start demonstrations; they are not provided to the policy or used during skill retrieval.

Skill Rendering. For grid environments, the renderer arranges trajectory frames in temporal order, augmenting them with aligned action labels and cumulative path overlays to clarify spatial progression. For PrimitiveSkill, the original third-person frames are preserved to retain object configurations and manipulation context. Figure 10 shows representative visual skill cards across the three environments.

Skill Library Management. Each library entry stores the composite card, geometric descriptor, reference action trajectory, strategy description, utility, and usage count. The reference trajectory is retained for computing the skill-guidance reward; the composite card is appended to the VLM’s visual context at decision time. Newly distilled skills are initialized with utility 0.5 and usage count 0; cold-start skills use the same initialization and are subsequently managed identically to online distilled skills.

Only the skill retrieved for an episode receives a utility and usage-count update after termination; all non-retrieved skills retain their current statistics. When the library reaches capacity, we apply the utility–frequency eviction rule before admitting the new skill, so that infrequently used or ineffective skills are gradually displaced by more recent acquisitions.

Reward and Optimization. The environment reward is accumulated over interaction turns before the skill-guidance term is added. In Sokoban, each turn with a non-empty valid action sequence receives a formatting reward of 0.1, and task completion receives a terminal reward of 1.0. Frozen-Lake uses the same terminal reward with a formatting reward of 0.02. In PrimitiveSkill, each valid turn receives a formatting reward of 0.1, each newly completed manipulation stage (e.g., grasping, placing, or aligning an object) provides an incremental reward of 2.0, and full task completion adds a terminal reward of 1.0.

We instantiate ℓ using task-specific state progress rather than exact action alignment. In Sokoban and FrozenLake, progress is measured by relative position with respect to the initial state: the player– box configuration for Sokoban and the agent position for FrozenLake. In PrimitiveSkill, progress is determined by matching completed actions against the reference trajectory. In each case, ℓ is the furthest matching progress point along the retrieved skill trajectory.

For PPO optimization, the complete multi-turn interaction is tokenized into τ¯. Observation tokens, including image tokens from the environment and retrieved skill card, are excluded from both actor and critic losses; only policy-generated response tokens have $m _ { i } = 1$ . We estimate advantages and returns via generalized advantage estimation with both its discount and smoothing parameters set to 1.0. The actor is optimized with a clipped objective and $\epsilon = 0 . 2$ . The critic $V _ { \phi }$ minimizes the clipped value loss:

$$
\begin{array} { c l r } { \displaystyle \mathcal { L } _ { \mathrm { V } } ( \phi ) = \frac { 1 } { 2 \sum _ { i = 1 } ^ { N } m _ { i } } \sum _ { i = 1 } ^ { N } m _ { i } \mathrm { m a x } \Bigg ( \left( V _ { \phi } ( \bar { \tau } _ { \le i } ) - \widehat { G } _ { i } \right) ^ { 2 } , } & \\ { \displaystyle \left( \mathrm { c l i p } \Big ( V _ { \phi } ( \bar { \tau } _ { \le i } ) , V _ { \mathrm { o l d } , i } - \epsilon _ { \mathrm { V } } , V _ { \mathrm { o l d } , i } + \epsilon _ { \mathrm { V } } \Big ) - \widehat { G } _ { i } \right) ^ { 2 } \Bigg ) , } \end{array}\tag{11}
$$

where $\widehat { G } _ { i }$ is the GAE return, $V _ { \mathrm { o l d } , i }$ is the rollout value prediction, and $\epsilon _ { \mathrm { V } } = 0 . 5$ . We use one PPO epoch per rollout batch with no entropy or KL auxiliary loss.

## B TRAINING AND EVALUATION DETAILS

## B.1 ENVIRONMENT AND TASK CONFIGURATIONS

All tasks provide the policy with RGB observations together with a textual task instruction and action-format specification; no symbolic state is exposed to the policy. Each episode is generated deterministically from its random seed, and success is evaluated by the task-specific environment predicate. Training and evaluation use disjoint seed ranges across all environments, ensuring that evaluation layouts and object configurations are never seen during training.

Sokoban. At each turn, the policy outputs up to three actions from $\{ \mathrm { u p , d o w n , l e f t , r i g h t } \}$ An episode lasts at most five turns and succeeds when the box reaches its target.

FrozenLake. We use randomly generated maps with frozen-cell probability 0.8. Transitions are deterministic, with slippery dynamics disabled. The policy selects from {up, down, left, right} and may output up to five actions per turn over at most five turns. An episode terminates when the agent reaches the goal or falls into a hole.

PrimitiveSkill. We use third-person RGB observations from ManiSkill (Mu et al., 2021) across five tasks: Place, Stack, Drawer, Align, and Swap. The policy controls a Franka Panda arm via parameterized actions from $\left\{ \mathtt { p i c k } ( x , y , z ) , \mathtt { p l a c e } ( x , y , z ) , \mathtt { p u s h } ( x _ { 1 } , y _ { 1 } , z _ { 1 } , x _ { 2 } , y _ { 2 } , z _ { 2 } ) \right\}$ with continuous xyz coordinates. The policy may issue up to two actions per turn, with a maximum of three turns for Place, Stack, Drawer, and Align, and four for Swap.

## B.2 TRAINING HYPERPARAMETERS

Our implementation builds on VAGEN (Wang et al., 2026a), which is developed on top of the veRL framework (Sheng et al., 2025). We share a common optimization and rollout configuration across all three environments, adapting only the training horizon, computational resources, and sequencelength limits to each task. Tables 5 and 6 summarize the shared and environment-specific hyperparameters, respectively.

<table><tr><td>Hyperparameter Value</td></tr><tr><td>Optimization</td></tr><tr><td>Policy backbone Qwen2.5-VL-3B-Instruct</td></tr><tr><td>Algorithm PPO</td></tr><tr><td>Advantage estimator GAE</td></tr><tr><td>PPO mini-batch size 32</td></tr><tr><td>Actor learning rate  $1 \times 1 0 ^ { - 6 }$ </td></tr><tr><td>Critic learning rate  $1 \times 1 0 ^ { - 5 }$ </td></tr><tr><td>Weight decay 0.01</td></tr><tr><td>GAE discount parameter 1.0</td></tr><tr><td>GAE smoothing parameter 1.0 1</td></tr><tr><td>PPO epochs per rollout batch</td></tr><tr><td>Actor clipping coefficient 0.2</td></tr><tr><td>Critic clipping coefficient 0.5</td></tr><tr><td>Entropy coefficient 0.0</td></tr><tr><td>KL coefficient 0.0</td></tr><tr><td>Rollout</td></tr><tr><td>Rollout batch size 128 1</td></tr><tr><td>Rollouts per task 0.6</td></tr><tr><td>GPU memory utilization</td></tr><tr><td>Skill System</td></tr><tr><td>Initial skill utility 0.5</td></tr><tr><td>Similarity weight α 0.9</td></tr><tr><td>EMA coefficientβ</td></tr><tr><td>0.1 Retrieval threshold</td></tr><tr><td> $\delta _ { r }$  0.8 Novelty threshold</td></tr><tr><td> $\delta _ { d }$  0.9</td></tr><tr><td>Skill-guidance weight λ 0.8</td></tr></table>

Table 5: Shared training hyperparameters across all environments.

<table><tr><td>Hyperparameter</td><td>Sokoban</td><td>FrozenLake</td><td>PrimitiveSkill</td></tr><tr><td>Total training steps</td><td>300</td><td>300</td><td>60</td></tr><tr><td>Training GPUs</td><td>4</td><td>2</td><td>2</td></tr><tr><td>Environment GPUs</td><td></td><td></td><td>1</td></tr><tr><td>Maximum prompt length</td><td>5,000</td><td>5,000</td><td>3,000</td></tr><tr><td>Maximum response length</td><td>4,000</td><td>4,000</td><td>10,000</td></tr></table>

Table 6: Environment-specific training hyperparameters.

## C PROMPT TEMPLATES

We provide the full prompt templates used for environment interaction and strategy distillation.   
Braced fields denote runtime-supplied values.

## C.1 STRATEGY DISTILLATION PROMPT

This prompt generates the strategy description for a newly accepted skill, conditioned on the task description, initial observation, rendered trajectory, and environment-specific examples. During online self-distillation, we append the <strategy> prefix to elicit an extractable continuation, with generation capped at 256 tokens. Cold-start distillation uses the same prompt body without the forced prefix and allows at most 512 tokens from the external model.

<table><tr><td>Strategy Distillation Prompt Template</td><td></td></tr><tr><td>[Skill Distillation]</td><td></td></tr><tr><td>Task: {task_description}</td><td></td></tr><tr><td>You successfully completed this task. Based on the initial state and your trajectory, distill a reusable strategy that can guide solving similar tasks.</td><td></td></tr></table>

The strategy should describe WHEN to apply it (what the state looks like) and WHAT to do (concrete   
actions or approach).   
Your response should be in the format of:   
<strategy>...</strategy>   
{examples}   
Initial state:   
<image>   
Successful trajectory:   
<image>   
Now distill your strategy:<strategy>

## C.2 ENVIRONMENT INTERACTION PROMPTS

Sokoban Prompts. The following templates specify the Sokoban system prompt and initial instruction prompt. The retrieved composite skill card is inserted between the system prompt and the initial observation.

Sokoban System Prompt Template   
You are a Sokoban solver.   
Sokoban Quick Guide   
Goal: Push all boxes onto targets.   
Symbols (If image is provided there are no symbols):   
<sub>#</sub> <sub>Wall</sub> <sub>—</sub> <sub>Floor</sub> <sub>—</sub> <sub>O</sub> <sub>Target</sub> <sub>—</sub> <sub>X</sub> <sub>Box</sub> <sub>—</sub> <sub>P</sub> <sub>You</sub> <sub>—</sub> √ <sub>Box</sub> <sub>on</sub> <sub>Target</sub> <sub>—</sub> <sub>S</sub> <sub>You</sub> <sub>on</sub> <sub>Target</sub>   
Rules:   
1. Push boxes (can’t pull).   
2. Avoid walls.   
Actions you can take: Left, Down, Right, Up.   
You can take up to 3 action(s) at a time, separated by ,.   
You should first give your reasoning, and then your answer.   
Your response should be in the format of:   
<think>...</think><answer>...</answer>   
Example 1:   
<think>The box is one step below me, and the target is two steps below me. I should go down to reach   
the box and then push it down to the target.</think>   
<answer>Down</answer>   
Example 2:   
<think>The box is to the right of me, and the target is further to the right. I need to move right to get   
behind the box and push it toward the target.</think>   
<answer>Right</answer>   
Example 3:   
<think>The box is above me, and the target is above the box. I should move up to reach the box and   
then push it upward to the target.</think>   
<answer>Up</answer>

## Sokoban Instruction Prompt Template

[SYSTEM]   
{Sokoban system prompt above}   
[Skill Reference]:   
The following shows a successful solution for a similar task. Use it as guidance for your approach.   
<image>   
[USER]   
[Initial Observation]:   
<image>   
Decide your next action(s).

FrozenLake Prompts. The following templates specify the FrozenLake system prompt and initial instruction prompt. The retrieved composite skill card is inserted between the system prompt and the initial observation.

## FrozenLake System Prompt Template

You are a FrozenLake solver.   
FrozenLake Quick Guide   
Goal: Reach the goal (G).   
Symbols (If image is provided there are no symbols):   
Frozen — O Hole — G Goal — P Player — X Player fell into hole — V Player on goal   
Rules:   
1. Avoid falling into holes.   
2. Frozen tiles are slippery, you may move perpendicular to your intended direction.   
Actions you can take: Left, Down, Right, Up.   
You can take up to 5 action(s) at a time, separated by ,.   
You should first give your reasoning, and then your answer.   
Your response should be in the format of:   
<think>...</think><answer>...</answer>   
Example 1:   
<think>The goal is below me. I should go down to reach it while avoiding the hole on my   
left.</think>   
<answer>Down</answer>   
Example 2:   
<think>The goal is to my right and there’s a hole directly below me. I should go right first to avoid the   
hole.</think>   
<answer>Right</answer>   
Example 3:   
<think>I can see the goal is up and to the left. I should move up first to get closer.</think>   
<answer>Up</answer>

## FrozenLake Instruction Prompt Template

[SYSTEM]   
{FrozenLake system prompt above}   
[Skill Reference]:   
The following shows a successful solution for a similar task. Use it as guidance for your approach.   
<image>   
[USER]   
[Initial Observation]:   
<image>   
Decide your next action(s).

PrimitiveSkill Prompts. The following templates specify the PrimitiveSkill system prompt and initial instruction prompt. The retrieved composite skill card is inserted between the system prompt and the initial observation.

## PrimitiveSkill System Prompt Template

You are an AI assistant controlling a Franka Emika robot arm. Your goal is to understand human instructions and translate them into a sequence of executable actions for the robot, based on visual input and the instruction.

You can command the robot using the following actions:

1. pick(x, y, z) # To grasp an object located at position(x,y,z) in the robot’s workspace.

2. place(x, y, z) # To place the object currently held by the robot’s gripper at the target position (x,y,z).

3. push(x1, y1, z1, x2, y2, z2) # To push an object from position (x1,y1,z1) to (x2,y2,z2).

1. The coordinates (x, y, z) are in millimeters and are all integers.

2. Please ensure that the coordinates are within the workspace limits.

<table><tr><td>3. The position is the center of the object, when you place, please consider the volume of the object. It&#x27;s always fine to set z much higher when placing an item. 4. We will provide the object positions to you, but you need to match them to the object in the image by yourself. You’re facing toward the negative x-axis, and the negative y-axis is to your left, the positive y-axis is to your right, and the positive z-axis is up. Examples: round1:</td></tr><tr><td>image1 Human Instruction: Put red cube on green cube and yellow cube on left target Object positions:</td></tr><tr><td>[(62,-55,20),(75,33,20),(-44,100,20),(100,-43,0),(100,43,0)] Reasoning: I can see from the picture that the red cube is on my left and green cube is on my right and</td></tr><tr><td>near me. Since I&#x27;m looking toward the negative x axis, and negative y-axis is to my left, (62,-55,20) would be</td></tr><tr><td>the position of the red cube, (75,33,20) would be the position of the green cube and (-44,100,20) is the position of the yellow cube.</td></tr><tr><td>Also the (100,-43,0) would be the position of the left target, and (100,43,0) would be the position of the right target. I need to pick up red cube first and place it on the green cube, when placing, I should set z much higher. Answer: pick(62,-55,20)—place(75,33,50)</td></tr><tr><td>round2: image2 Human Instruction: Put red cube on green cube and yellow cube on left target</td></tr><tr><td>Object positions: [(75,33,50),(75,33,20),(-44,100,20),(100,-43,0),(100,43,0)]</td></tr><tr><td>Reasoning: Now the red cube is on the green cube, so I need to pick up the yellow cube and place it on the left target.</td></tr><tr><td>Answer: pick(-44,100,20)—place(100,-43,50) You can take up to 2 action(s) at a time, separated by ’—’.</td></tr><tr><td>You should first give your thought process, and then your answer. Your response should be in the format of: &lt;think&gt;...&lt;/think&gt;&lt;answer&gt;...&lt;/answer&gt;</td></tr></table>

## PrimitiveSkill Instruction Prompt Template

<table><tr><td>[SYSTEM]</td></tr><tr><td>{PrimitiveSkill system prompt above}</td></tr><tr><td>[Skill Reference]:</td></tr><tr><td>The following shows a successful solution for a similar task. Use it as guidance for your approach.</td></tr><tr><td>&lt;image&gt;</td></tr><tr><td>[USER]</td></tr><tr><td>[Initial Observation]:</td></tr><tr><td>&lt;image&gt;</td></tr><tr><td>Human Instruction: {instruction}</td></tr><tr><td>x_workspace_limit: {x_workspace}</td></tr><tr><td>y_workspace_limit: {y_workspace}</td></tr><tr><td>z_workspace_limit: {z_workspace}</td></tr><tr><td>Object positions:</td></tr><tr><td>{object_positions}</td></tr><tr><td>Other information:</td></tr><tr><td>{other_information}</td></tr><tr><td>Decide your next action(s).</td></tr><tr><td>You can take up to 2 action(s) at a time, separated by ’—’.</td></tr><tr><td>You should first give your thought process, and then your answer.</td></tr><tr><td>Your response should be in the format of:</td></tr><tr><td>&lt;think&gt;...&lt;/think&gt;&lt;answer&gt;..&lt;/answer&gt;</td></tr></table>

![](images/e801817183a450d8f61f4732c403b08333391e08c432fb42c464fda8d9c8f3bf.jpg)

<think>I need to pick the red cube at (62,-55,20) and place it on the green cube at (75,33,20). When   
placing, I should set z much higher to stack properly.</think>   
<answer>pick(62,-55,20)—place(75,33,50)</answer>

![](images/46c3c9da77554bd809a686bb0b1e95e5a3faf35f569adb18bba0ceecadbaee4c.jpg)  
Figure 11: Task-wise success-rate dynamics across the five PrimitiveSkill manipulation tasks. Both ViSkill variants reach full success faster than PPO, with cold-start initialization further accelerating early-stage learning.  
Figure 12: Skill retrieval and distillation dynamics on Sokoban and FrozenLake. Retrieval fraction (left) and newly admitted skills per step (right). Retrieval becomes nearly ubiquitous as the library grows, while admissions diminish once common configurations are covered.

## D ADDITIONAL ANALYSIS

## D.1 ADDITIONAL TRAINING DYNAMICS

PrimitiveSkill Task-Wise Success-Rate Dynamics. Figure 11 disaggregates the average PrimitiveSkill result across its five manipulation tasks. ViSkill consistently reaches full success earlier than PPO on all five tasks. PPO eventually solves Place but plateaus below full success on the remainder, whereas both ViSkill variants achieve a success rate of 1.00. Cold-Start further accelerates early-stage learning across tasks. These results confirm that visual skills provide reusable guidance across diverse coordinate-grounded manipulation objectives, rather than improving only the aggregate metric.

![](images/8aa68570b7864b38fba11f676d593bc12425a562c0fe4353006a818399b1e281.jpg)  
Instruction：Push all boxes onto targets. Push boxes (can’t pull). Avoid walls. Actions you can take: Left, Down, Right, Up. …

![](images/2717ecff5b76cf80ee01e783c80f3c7138fdbafd9ad7316dc8ccd51c2bc40398.jpg)

![](images/ef59cb4ef49ba7c25fe2296318a42776e38c460d270e1f5835adb630f3cce420.jpg)

Instruction：Reach the goal (G). Avoid falling into holes. Frozen tiles are slippery, you may move perpendicular to your intended direction. Actions you can take: Left, Down, Right, Up. …  
![](images/217912f28530fa41a91663105c727b02af3ade5bae113831c9002ff85bff42fe.jpg)  
Figure 13: Qualitative comparison with PPO on Sokoban and FrozenLake. For each configuration, PPO reaches a failure state while ViSkill retrieves a relevant visual skill card and completes the task.

Skill Retrieval and Distillation Dynamics. Figure 12 illustrates the closed-loop evolution of the skill library. In both environments, the retrieval fraction rises rapidly and stabilizes near one as the library gains coverage, reflecting that existing skills increasingly satisfy the geometric retrieval criterion. Skill admissions, by contrast, concentrate in the early stage and taper off thereafter. This separation confirms that the novelty gate prevents redundant expansion: once common configurations are covered, the agent reuses existing skills while admitting only genuinely novel trajectories.

## D.2 CASE STUDY

Figure 13 compares PPO and ViSkill on matched Sokoban and FrozenLake configurations. In both cases, the retrieved visual skill provides a spatially grounded reference that guides the policy toward a productive direction and encourages successful multi-turn behavior. PPO reaches failure states under the same configurations.

In the FrozenLake example, the trajectory stored in the retrieved skill does not exactly match the one executed by ViSkill. The policy takes a different valid route while remaining consistent with the spatial structure and high-level intent of the skill card, indicating that ViSkill uses the card as procedural guidance rather than replaying reference actions.

## D.3 QUALITY THRESHOLD ABLATION

Table 7 examines whether ViSkill depends critically on solver-derived quality thresholds by substituting them with a fixed trajectory-length budget. Static-10 achieves success rates of 0.84 on Sokoban and 0.78 on FrozenLake, against 0.82 and 0.84 for solver-based filtering. The fixed threshold slightly outperforms the solver on Sokoban while incurring only a modest decrease on FrozenLake, indicating that solver access is not a prerequisite for effective quality control. A simple fixed budget therefore provides a practical alternative when task-specific solvers are unavailable.

<table><tr><td>Threshold</td><td>Sokoban FrozenLake</td><td></td></tr><tr><td>Solver-Based</td><td>0.82</td><td>0.84</td></tr><tr><td>Static-10</td><td>0.84</td><td>0.78</td></tr></table>

Table 7: Success rates under different quality thresholds.

## D.4 FEATURE EXTRACTOR ABLATION

Table 8 investigates the role of hand-crafted geometric descriptors by replacing them with DI-NOv2 (Oquab et al., 2023) visual features for skill retrieval. Despite receiving no task-specific supervision, DINOv2 achieves success rates of 0.80 on Sokoban and 0.83 on FrozenLake, falling short of the rule-based extractor by only 0.02 and

<table><tr><td>Extractor</td><td colspan="2">Sokoban FrozenLake</td></tr><tr><td>Rule-Based</td><td>0.82</td><td>0.84</td></tr><tr><td>DINOv2</td><td>0.80</td><td>0.83</td></tr></table>

Table 8: Success rates with different feature extractors.

0.01 in absolute terms. This narrow gap demonstrates that the spatial structure needed for reliable retrieval is largely recoverable from general-purpose visual representations, and that ViSkill is not fundamentally tied to hand-crafted descriptors. Replacing them with learned encoders thus emerges not merely as a future direction but as a practically viable alternative, broadening the applicability of ViSkill to environments where geometric features are difficult to specify manually.
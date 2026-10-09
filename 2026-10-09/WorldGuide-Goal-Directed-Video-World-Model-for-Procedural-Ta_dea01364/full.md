# WorldGuide: Goal-Directed Video World Model for Procedural Task Execution

Ankan Deria, Komal Kumar, Hisham Cholakkal, Fahad Shahbaz Khan, Salman Khan

Mohamed bin Zayed University of Artificial Intelligence

## Abstract

Video generators and video-based world models can synthesize plausible visual trajectories, but long-horizon procedural tasks require generation to adapt to what has actually been produced. A model must determine the next action from its generated state, execute that action, and recognize when the task is complete. Open-loop generation cannot adapt to execution outcomes, while existing closed-loop systems often rely on pretrained executors or indirect verification. This leaves a gap between deciding an action and successfully realizing it. We formulate procedural video generation as closed-loop task execution in visual world space and introduce WorldGuide. Given only an initial image and a task goal, WorldGuide predicts an atomic action, generates its corresponding video clip, and uses the generated result to select the next action or terminate. The Planner and Executor are trained on the same step-level procedural demonstrations: the Planner learns to predict the next atomic action or task completion from visual progress, while the Executor is directly trained to realize the predicted actions. Hierarchical visual memory maintains state across long-horizon execution with bounded history token cost. Due to the lack of step-level action-video supervision for joint planner-executor training, we introduce WorldGuide Bench: approximately 59K step-annotated videos across 245 tasks and 27 procedural categories. WorldGuide achieves a 33.33% Task Success on WorldGuide-Bench, compared with 29.90% for the strong recent video model MiniMax-H3, even though MiniMax-H3 receives reference action plans, and achieves 47.69% on VideoCraft-Bench compared with 32.73% for MiniMax-H3 under goal-only conditioning. These results demonstrate the importance of coupling planning with learned execution for goal-directed procedural video generation.

 Project Page: mbzuai-oryx.github.io/WorldGuide <sup>§</sup> GitHub: mbzuai-oryx/WorldGuide

## 1 Introduction

Video-based world models predict how visual states evolve under actions, goals, or other control signals. Direct prompt-to-video models [25, 10, 41, 17, 38, 45] and autoregressive approaches [46, 5] can synthesize plausible visual trajectories, but visual plausibility does not ensure task completion. Open-loop generation fixes the prompt or action sequence before rollout, without observing progress, revising subsequent actions, or determining when the goal is reached. As a result, an incorrect or incomplete step cannot be corrected and generation may continue beyond task completion (Fig. 1a). This becomes particularly problematic for long-horizon procedures such as origami, assembly, and cooking, where errors in one step can affect everything that follows.

Closed-loop approaches introduce feedback, but a gap remains between planning an action and reliably executing it. ORCA [14], CollabVR [15], and NovaPlan [7] pair a pretrained vision-language planner with a frozen video executor. A correct plan can therefore still fail when the executor cannot realize the requested atomic action (Fig. 1b). SPIRAL [44] uses a separately trained critic to evaluate generated outcomes and provides its judgment, rather than directly supervising atomic action execution. Bernini [34] plans in latent semantic space before rendering, but does not predict successive procedural actions from generated progress or determine task completion. Goal-directed generation therefore requires a planner grounded in the evolving visual state and an executor trained to realize the actions it predicts. This motivates our formulation of closed-loop task execution in visual world space.

![](images/cdfd37047b2218367465ce7f6cba608aa885042b5d13734478c6f5a6876d2286.jpg)  
Figure 1: From open-loop synthesis to learned procedural execution. (a) Open-loop generation follows a fixed prompt with no intermediate decisions, so it can drift into a wrong step and keep generating after the task is done. (b) A planner can inspect outcomes and replan, but with a frozen pretrained executor, a correct instruction can still fail to execute. (c) WorldGuide trains the ContextPlanner and Executor on the same demonstrations, the Executor conditioned on the planner’s action embeddings. Thus, generated outcomes guide the next action and the decision to stop.

This gap raises five challenges that a closed-loop procedural video model must address. First, if the executor is frozen or only indirectly supervised through a reward model, it cannot be trusted to correctly realize an arbitrary planner-proposed action; the executor itself needs to be trained on the atomic actions it is asked to perform. Second, a planner conditioned only on its previously requested actions cannot determine whether those actions were successfully executed, partially completed, or resulted in visual drift. Therefore, planning must be grounded in the generated visual state, not action history alone. Third, task completion must be predicted from visual progress rather than determined by a fixed generation length or external retry budget. Fourth, because every generated clip affects subsequent decisions, long-horizon execution requires persistent visual context without unbounded memory growth. Fifth, existing instructional-video datasets do not provide demonstrations that pair atomic actions with their visual consequences and explicit completion signals. Therefore, closed-loop procedural training requires dedicated step-level action–video supervision.

To address these challenges, we introduce WorldGuide, a video world model for closed-loop procedural task execution (Fig. 1c). First, we construct WorldGuide Bench (§ 3.5), comprising approximately 59K procedural videos across 245 tasks and 27 categories, segmented into temporally grounded atomic action clips with explicit completion signals. Second, we train the ContextPlanner (§ 3.2) on this data to predict the next atomic action or a completion token from the generated visual state and action history (§ 3.1). Third, we train the Executor (§ 3.3) on the same demonstrations, conditioned on the frozen ContextPlanner’s action embeddings, so it learns to render the same atomic actions the planner is trained to propose. Fourth, to support long-horizon state continuity, the

Executor maintains a hierarchical, compressed visual memory (§ 3.4), representing older history at progressively coarser spatial resolution to bound the token cost as the rollout grows. Our experiments validate these design choices. Closed-loop execution improves Task Success by 21.62% over the open-loop variant (§ 4.2); visual feedback improves Task Success by 18.61% (§ 4.3) and Plan Success by 5.34% while reducing Repeat/Skip by 4.46% (§ 4.4); while memory provides gains of 5.77% on WorldGuide Bench and 17.91% on Video-CraftBench (§ 4.5–§ 4.6). Our proposed approach achieves the highest Aesthetic Quality and Overall Consistency on WorldGuide Bench (§ 4.7).

## 2 Related Work

## 2.1 Video Generation and World Models

Modern video generators [45, 38, 41, 10] synthesize strong visual trajectories, but operate open-loop, without selecting the next semantic action from generated progress or recognizing task completion. Long-context methods such as FramePack [48] and interactive world generators [22, 21] improve temporal continuity via compressed visual history, but memory alone does not determine how to advance a procedure. World models extend visual prediction toward decision making: VideoWorld 2 [28] learns policies through autoregressive latent dynamics, hierarchical world models [49] plan over latent subgoals, vision-language-action models [54, 16] predict robot controls, and Video Language Planning [6] combines policies, value functions, and video dynamics through tree search. None of these approaches select actions from visual progress, execute them through directly supervised video generation, and decide completion within a single closed loop. WorldGuide addresses this by coupling visual memory with progress-conditioned action selection, atomic-action execution, and learned termination, all grounded in generated visual states.

## 2.2 Planner–Executor Video Generation

Planner–executor systems move video generation toward closed-loop interaction, but couple planning and execution differently. BERNINI [34] plans latent visual semantics before rendering, without recurrent procedural decisions from generated progress. TempAct [39] jointly trains planning and execution but follows prescribed temporal spans without visual-state replanning. PhysAgent [19] refines physical programs via scene reconstruction and physics simulation before synthesis. ORCA [14] and CollabVR [15] add visual feedback through verification and retries, but keep pretrained executors untrained with their planners. SPIRAL [44] trains planning and generation jointly, but still relies on a separate critic and reinforcement learning for execution refinement. None learn the full procedural loop: selecting an action from generated progress, executing it under direct supervision, and deciding completion. WorldGuide instead learns next-action selection, atomic-action execution, and completion prediction from the same demonstrations, coupling planning with learned execution in a recurrent closed loop.

## 3 Method

## 3.1 Overview and Problem Formulation

Given an initial image $x _ { 0 }$ and task goal g, WorldGuide performs goal-directed video world modeling through a recurrent planner–executor loop (Fig. 2). At step t, the ContextPlanner $\pi _ { \theta }$ predicts $a _ { t } \in \mathcal A \cup$ {DONE} from the goal $^ { g , }$ the current visual state $x _ { t } .$ , and the clip–action history $h _ { t } .$ , where A is the atomic-action space and DONE is the completion token <|Task Completed|>. If $a _ { t } = \mathtt { D 0 N E }$ , the rollout terminates; otherwise, the Executor $f _ { \phi }$ generates the next visual state $x _ { t + 1 }$ , a short video clip, conditioned on $a _ { t } , x _ { t } .$ , and visual memory $M _ { t } .$

$$
\begin{array} { r l r } & { } & { a _ { t } \sim \pi _ { \theta } ( \cdot \mid g , x _ { t } , h _ { t } ) , \qquad x _ { t + 1 } = f _ { \phi } ( a _ { t } , x _ { t } , M _ { t } ) , } \\ & { } & { h _ { t + 1 } = h _ { t } + a _ { t } , \qquad M _ { t + 1 } = \mathrm { u p d a t e } ( M _ { t } , x _ { t + 1 } ) \quad } \end{array}\tag{1}
$$

with $h _ { 0 }$ and $M _ { 0 }$ empty. For $t \geq 1 , x _ { t }$ is the clip generated for $a _ { t - 1 }$ , and the Executor uses its last frame as the reference image. The history $h _ { t }$ holds the latest $K { = } 3$ actions, each paired with the clip it generated: $h _ { t } + a _ { t }$ appends $a _ { t }$ together with its clip $x _ { t + 1 }$ and drops pairs older than K steps. update adds $x _ { t + 1 }$ to the Executor’s memory. The rollout ends when the planner predicts DONE or the step budget is reached, and the generated clips $x _ { 1 } , x _ { 2 } , \dotsc .$ . are concatenated into the final video (Appendix D.3).

![](images/b57563b8dd8741e5d481a919d6202002276b5d47d90616c65618c9ff0fb83e57.jpg)  
Figure 2: Overview of WorldGuide. Given the initial image $x _ { 0 } ,$ task goal $^ { g , }$ and the latest $K { = } 3$ clip–action pairs $h _ { t } ,$ the ContextPlanner $\pi _ { \theta }$ predicts the next atomic action $a _ { t }$ or DONE. The Executor $f _ { \phi }$ renders $a _ { t }$ into the next clip $x _ { t + 1 }$ from the current visual state $x _ { t }$ and hierarchical memory $M _ { t } .$ which progressively compresses older latent frames. Each new clip $x _ { t + 1 }$ becomes the next visual state, is paired with $a _ { t }$ in the history $h _ { t + 1 }$ , and updates the memory $M _ { t + 1 }$ , and the loop continues until DONE is predicted or the step budget is reached. Notation follows Eq. 1.

The ContextPlanner and Executor are initialized from Qwen2.5-VL-7B and HunyuanVideo-1.5 respectively. We train the ContextPlanner first, freeze it, and use it to embed action captions for Executor fine-tuning (Appendix A); at inference, action prediction and termination depend on the Executor’s generated results. The Executor also adapts an existing hierarchical memory design (§ 3.4) for visual continuity.

## 3.2 ContextPlanner

The planner predicts the next atomic action $a _ { t }$ from the task goal $^ { g , }$ the current visual state $x _ { t } .$ , and the history $h _ { t } = \big ( ( a _ { i } , x _ { i + 1 } ) \big ) _ { i = \operatorname* { m a x } ( 0 , t - K ) } ^ { t - 1 } ;$ which pairs each recent action with the clip it generated; for $t > 0 , x _ { t }$ is the newest clip in $h _ { t }$ . Thus, $K = 1$ supplies one clip and its action, and $K = 3$ supplies the last three clips and actions. We use $K = 3$ . This history is distinct from the Executor’s compressed latent memory $M _ { t } .$ , which the planner does not consume. The output is a single natural-language instruction or a completion token; the visual history provides evidence of recent progress and helps avoid repeated steps. We train it by supervised fine-tuning of Qwen2.5-VL-7B on demonstration histories and their next-action or completion targets, using token-level cross-entropy masked to response tokens. No architectural modification or auxiliary objective is used.

## 3.3 Executor

The Executor is a HunyuanVideo-1.5 [41] diffusion transformer conditioned on action embeddings from the frozen ContextPlanner and on visual context. During training, it receives the ground-truth action caption paired with the target clip; at inference, it receives the ContextPlanner’s predicted action through the same embedding pathway.

Each training sample provides a target video latent $z \in \mathbb { R } ^ { C \times L \times H \times W }$ , where $L$ is the clip’s latent-frame length, with text and vision conditioning features $c _ { \mathrm { t e x t } } , c _ { \mathrm { v i s i o n } }$ . The text condition $c _ { \mathrm { t e x t } }$ includes the ground-truth action embeddings from the frozen ContextPlanner. Under a flow-matching parameterization, the noisy latent at noise level σ is $\tilde { z } = ( 1 - \sigma ) z + \sigma \epsilon \mathrm { w i t h } \epsilon \sim \mathcal { N } ( 0 , I )$ , and the model predicts the residual velocity $\hat { y } _ { \phi } = f _ { \phi } ( \widetilde { z } , \sigma , c )$ toward the

target (ϵ − z):

$$
\mathcal { L } _ { \mathrm { e x e c } } = \mathbb { E } _ { z , \epsilon , \sigma } \left[ \frac { \| m \odot ( \hat { y } _ { \phi } - ( \epsilon - z ) ) \| _ { 2 } ^ { 2 } } { \operatorname* { m a x } ( \sum m , 1 ) } \right] ,\tag{2}
$$

where m restricts supervision to valid regions. This loss updates only Executor parameters ϕ; ContextPlanner parameters $\theta$ remain frozen. Reference-image conditioning enters as a latent tensor $z _ { \mathrm { c o n d } } \in \mathbb { R } ^ { C \times L \times H \times W }$ stacked channel-wise with $\tilde { z } ,$ accompanied by a validity mask $m _ { \mathrm { c o n d } } ,$ so that $c = ( z _ { \mathrm { c o n d } } , m _ { \mathrm { c o n d } } , c _ { \mathrm { t e x t } } , c _ { \mathrm { v i s i o n } } )$ . Without memory, only the reference frame is populated and the remaining slots are masked out. With memory, history latents also populate the conditioning tensor and are embedded into context tokens (Sec. 3.4; Appendix C.2).

## 3.4 Hierarchical Memory Compression

We adopt YUME’s spatial historycompression schedule, following Yume-1.5 [21] and FramePack [48]. Recent latent frames receive finer spatial embeddings than older ones, bounding the Executor’s history-token count.

Memory bank. Let $T \le 1 3 6 5$ denote the capacity for non-anchor history latent frames. The compressor additionally handles the oldest history latent frame as an anchor, giving a total history length $F \leq T + 1 \leq 1 3 6 6 $

<table><tr><td>Oldest</td><td colspan="4">Teporar memory merarcy Distant Far</td><td colspan="2">Newest</td></tr><tr><td rowspan="2">Anchor F</td><td rowspan="2">past</td><td>past (85, 341]</td><td>Mid past</td><td>Recent past</td><td>Near present</td><td>Present</td></tr><tr><td>(341, F)</td><td>(21,85]</td><td>(5,21]</td><td>(3,5]</td><td>[1, 3]</td></tr><tr><td></td><td>1</td><td>-</td><td></td><td></td><td></td><td></td></tr><tr><td>r = 2</td><td></td><td>r = 16</td><td>r = 8</td><td>r = 4</td><td>r = 2</td><td>r = 1</td></tr><tr><td>cost 0.25</td><td>r = 64 cost ≤ 0.25</td><td>cost 1</td><td>cost 1</td><td>cost 1</td><td>cost 0.5</td><td>cost 3</td></tr></table>

Temporal memory hierarchy  
Figure 3: Longest-history branch $( 3 4 3 \leq F \leq 1 3 6 6 )$ . Age 1 is newest; age F is the anchor. Rates r denote spatial reduction relative to base patch embedding, and costs are nominal latent-frame equivalents before spatial padding, with the separately embedded reference adding cost 1 at $r = 1$ Shorter branches appear in Appendix C.3.

$$
M _ { t } = \Pi _ { T } { \Big ( } { \Big [ } z ^ { ( 1 ) } \parallel \cdots \parallel z ^ { ( t ) } { \Big ] } { \Big ) }\tag{3}
$$

Here, $\boldsymbol { z } ^ { ( i ) } \in \mathbb { R } ^ { C \times F \times H \times W }$ is the VAE latent of the generated clip $x _ { i } , \Pi _ { T }$ forms the bounded history tensor from these preceding clips, and $H , W$ are latent spatial dimensions. F counts history latent slots supplied to compression, including the anchor and temporal padding; the current reference is separate. These are latent-frame counts after VAE encoding, not video frames. The schedule capacity is an upper limit, not the length used at every step; Appendix C.1 specifies how the conditioning path limits $F .$

Multi-scale patch embedding. With DiT spatial patch size $p = 2 , E _ { r }$ is a 3-D convolution with kernel and stride $( 1 , p r , p r )$ for $r \in \{ 1 , 2 , 4 , 8 , 1 6 \}$ . It maps C latent channels to d features. We initialize $E _ { 1 }$ from the pretrained patch embedding and larger kernels by trilinear upsampling. The coarsest embedder is $E _ { 6 4 } = E _ { 1 6 } \circ \Phi$ , where Φ is a channel-preserving convolution with kernel and stride $( 1 , 4 , 4 )$ . Its effective spatial stride is $1 2 8 = 6 4 p$ . All embedders preserve temporal length.

Partition and token budget. For nonempty history, the selected branch splits it into segments $S _ { j }$ with spatial rates $r _ { j }$ , the anchor at rate $r _ { a } ;$ Fig. 3 shows the longest branch. Ignoring spatial padding,

$$
\begin{array} { r l } { \boldsymbol { h } _ { j } = \mathrm { { f l a t t e n } } \big ( E _ { r _ { j } } ( M _ { t } [ : , S _ { j } ] ) \big ) \in \mathbb { R } ^ { n _ { j } \times d } , n _ { j } } & { = | S _ { j } | \frac { H W } { p ^ { 2 } r _ { j } ^ { 2 } } . } \end{array}\tag{4}
$$

The anchor, chronological history segments, and current reference form

$$
{ \mathcal { M } } = [ h _ { a } ; h _ { 1 } ; \cdots ; h _ { J } ; \mathrm { f a t t e n } ( E _ { 1 } ( z _ { \mathrm { r e f } } ) ) ] ,\tag{5}
$$

Table 1: Procedural video datasets and their supervision for planning and generation. Sizes retain their original units: videos (vid.), segments (seg.), clips. Tasks/Dom.: tasks/domains; Step-Clip: temporally localized step annotations; Atomic Act.: single-action granularity; Coverage Target: annotation protocol targets all visible task-relevant steps, excluding background intervals, rather than a verified coverage rate; Task Goal: explicit task-level goal; Completion: explicit task completion supervision. Gen., Eval., and Recog. denote generation, evaluation, and recognition.
<table><tr><td>Dataset</td><td>Size</td><td>Tasks/Dom.</td><td>Step-Clip</td><td>Atomic Act.</td><td>Coverage Target</td><td>Task Goal</td><td>Completion</td><td>Type</td><td>Public</td></tr><tr><td>EgoPlan-IT</td><td>50K QA</td><td>-/1</td><td>x</td><td>√</td><td>x</td><td>√</td><td>x</td><td>Planning QA</td><td></td></tr><tr><td>Ego4D Goal-Step</td><td>48K seg.</td><td>86/1</td><td>√</td><td>x</td><td>x</td><td></td><td>x</td><td>Localization</td><td></td></tr><tr><td>COIN</td><td>11.8K vid.</td><td>180/12</td><td>Fixed-vocab</td><td>Partial</td><td>x</td><td></td><td>x</td><td>Localization</td><td></td></tr><tr><td>CrossTask (primary)</td><td>2.75K vid.</td><td>18/4</td><td>V</td><td>Partial</td><td>x</td><td></td><td>x</td><td>Weak localization</td><td></td></tr><tr><td>YouCook2</td><td>2K vid.</td><td>89/1</td><td></td><td>x</td><td>√</td><td></td><td>x</td><td>Segmentation</td><td></td></tr><tr><td>HT-Step</td><td>19.7K vid.</td><td>433/1</td><td></td><td>Partial</td><td>x</td><td></td><td>x</td><td>Grounding</td><td></td></tr><tr><td>Assembly101</td><td>4.3K vid.</td><td>101/1</td><td></td><td>√</td><td></td><td>x</td><td>x</td><td>Recognition</td><td></td></tr><tr><td>CaptainCook4D</td><td>384 vid.</td><td>24/1</td><td>√</td><td>Partial</td><td>√</td><td></td><td>x</td><td>Error/localization</td><td></td></tr><tr><td>SemComp-Data</td><td>1.27K</td><td>21/6</td><td>x</td><td>x</td><td>x</td><td></td><td>x</td><td>Eval-only</td><td>Partial</td></tr><tr><td>ActVideoGen</td><td>118K clips</td><td>-/Multi</td><td>1</td><td>Partial</td><td>Partial</td><td></td><td>x</td><td>Gen. training</td><td>x</td></tr><tr><td>EgoForge / X-Ego</td><td>15K clips</td><td>-/Multi</td><td>x</td><td>√</td><td>X</td><td></td><td>x</td><td>Gen. training + eval</td><td>x</td></tr><tr><td>Ego-Exo4D Keystep</td><td>27.6K seg.</td><td>17/3</td><td>√</td><td>Partial</td><td>X</td><td></td><td>x</td><td>Recognition</td><td>J</td></tr><tr><td>WorldGuide (ours)</td><td>59K vid.</td><td>245/27</td><td>√</td><td>√</td><td>√</td><td>J</td><td>√</td><td>Gen. training + eval</td><td>Planned</td></tr></table>

where $h _ { a }$ embeds the oldest history latent frame. Across all supported branches,

$$
\frac { p ^ { 2 } } { H W } | \mathcal { M } | = \frac { 1 } { r _ { a } ^ { 2 } } + \sum _ { j } \frac { | S _ { j } | } { r _ { j } ^ { 2 } } + 1 \le 8 < 9 , \qquad T \le 1 3 6 5 .\tag{6}
$$

One full-resolution latent-frame equivalent is $H W / p ^ { 2 }$ tokens. This bound measures token cost, including the anchor and reference, not a reduction of history to eight temporal latent frames; spatial padding changes the actual count (Appendix C.3).

## 3.5 Dataset for WorldGuide Training

Closed-loop procedural execution requires supervision for three connected decisions: which action follows observed progress, how that action changes the visual state, and when the task is complete. Existing instructional datasets provide narration, action labels, or localized step descriptions [23, 52, 33, 53, 4], but need additional processing to support this training formulation. We construct WorldGuide Bench to combine ordered atomicaction clips, task goals, and explicit completion signals within each demonstration (Table 1). This shared supervision supports both ContextPlanner decisions and Executor training.

The dataset contains 58,679 videos spanning 245 tasks across 27 procedural categories (Fig. 4). We use Gemini 2.5 Flash [9] to filter web instructional videos

![](images/987522a9d56bd4735d644e20f45d78579a4ddcd41f8aeb283397980716c1e504.jpg)  
Figure 4: Task taxonomy of WorldGuide Bench. The inner ring shows procedural categories and their dataset shares; the outer ring lists representative tasks within each category.

and annotate temporally grounded atomic actions and completion. We used total 980 videos in test dataset. Construction details, annotation examples, and training-sample preparation are provided in Appendix B, and a human audit of the annotations in Appendix B.6.

Table 2: WorldGuide Bench evaluation. Unstarred baselines use reference actions; ∗ denotes methods with their own planning-execution loop. WorldGuide predicts actions from the task goal and generated history. Aesth.: Aesthetic Quality; Imag.: Imaging Quality; Dynamic: Dynamic Degree; Smooth.: Motion Smoothness; Consist.: Overall Consistency; Align.: Alignment; Plan Acc.: Plan Accuracy; Task Succ.: Task Success Rate. Higher is better except Repeat/Skip (Appendix E).
<table><tr><td rowspan="2">Model</td><td colspan="5">Video Quality Check</td><td colspan="5">Video Planning Check</td></tr><tr><td>Aesth.</td><td>Imag.</td><td>Dynamic</td><td>Smooth.</td><td>Consist.</td><td>Align.</td><td>Plan Acc.</td><td>Order</td><td>Repeat/Skip ↓</td><td>Task Succ.</td></tr><tr><td>Cosmos-Predict2.5-2B</td><td>38.77</td><td>62.11</td><td>82.52</td><td>96.32</td><td>17.46</td><td>63.75</td><td>38.05</td><td>84.08</td><td>3.98</td><td>11.94</td></tr><tr><td>Open-Sora 2.0-11B</td><td>33.42</td><td>44.91</td><td>60.19</td><td>95.06</td><td>17.42</td><td>62.43</td><td>31.27</td><td>79.56</td><td>6.53</td><td>8.54</td></tr><tr><td>HunyuanVideo-1.5-8.3B</td><td>42.43</td><td>65.57</td><td>88.84</td><td>96.00</td><td>18.91</td><td>62.28</td><td>43.53</td><td>80.00</td><td>3.00</td><td>12.00</td></tr><tr><td>Wan2.2-14B</td><td>41.46</td><td>58.87</td><td>16.02</td><td>93.67</td><td>17.52</td><td>67.33</td><td>4.37</td><td>21.65</td><td>2.06</td><td>2.58</td></tr><tr><td>CogVideoX-5B</td><td>38.76</td><td>54.99</td><td>64.08</td><td>96.58</td><td>18.09</td><td>68.30</td><td>34.33</td><td>75.74</td><td>3.96</td><td>9.90</td></tr><tr><td>Helios-14B</td><td>39.32</td><td>56.06</td><td>72.33</td><td>95.90</td><td>18.60</td><td>63.99</td><td>25.71</td><td>76.47</td><td>1.51</td><td>8.04</td></tr><tr><td>UniVideo-13B</td><td>45.75</td><td>61.92</td><td>48.00</td><td>97.09</td><td>18.05</td><td>65.63</td><td>10.77</td><td>52.00</td><td>4.00</td><td>4.00</td></tr><tr><td>FlashMotion-34B</td><td>41.69</td><td>67.26</td><td>90.29</td><td>95.46</td><td>17.93</td><td>62.92</td><td>24.10</td><td>77.39</td><td>2.51</td><td>5.53</td></tr><tr><td>MAGI-1-24B</td><td>39.14</td><td>63.08</td><td>2.30</td><td>98.31</td><td>17.00</td><td>69.33</td><td>17.70</td><td>61.63</td><td>2.33</td><td>3.49</td></tr><tr><td>RoboMaster-5B</td><td>32.60</td><td>17.68</td><td>4.37</td><td>97.09</td><td>6.66</td><td>37.13</td><td>4.29</td><td>5.50</td><td>0.50</td><td>4.00</td></tr><tr><td>SpMem-5B</td><td>35.79</td><td>47.69</td><td>96.73</td><td>98.94</td><td>20.72</td><td>60.92</td><td>33.74</td><td>78.95</td><td>2.94</td><td>10.53</td></tr><tr><td>Astra-1.3B</td><td>32.00</td><td>43.43</td><td>97.67</td><td>95.73</td><td>15.03</td><td>65.40</td><td>27.23</td><td>60.71</td><td>0.64</td><td>11.90</td></tr><tr><td>Yume-1.5-5B</td><td>40.61</td><td>65.72</td><td>87.86</td><td>96.90</td><td>17.39</td><td>60.79</td><td>19.72</td><td>74.26</td><td>2.39</td><td>2.97</td></tr><tr><td>SkyReels-V3-14B</td><td>42.64</td><td>58.78</td><td>14.56</td><td>99.66</td><td>17.03</td><td>72.54</td><td>11.23</td><td>48.52</td><td>1.01</td><td>3.90</td></tr><tr><td>MiniMax-H3-33B1</td><td>41.61</td><td>56.60</td><td>80.58</td><td>97.99</td><td>19.84</td><td>70.04</td><td>57.99</td><td>86.79</td><td>4.06</td><td>29.90</td></tr><tr><td>LTX-2.5-22B</td><td>43.85</td><td>57.92</td><td>85.92</td><td>99.29</td><td>15.63</td><td>70.10</td><td>16.14</td><td>49.76</td><td>1.46</td><td>7.32</td></tr><tr><td>HY-WorldPlay-8.3B</td><td>39.25</td><td>64.56</td><td>85.95</td><td>96.30</td><td>15.28</td><td>55.47</td><td>15.67</td><td>73.46</td><td>3.07</td><td>1.26</td></tr><tr><td>PhysAgent*</td><td>42.76</td><td>65.09</td><td>52.68</td><td>98.39</td><td>16.72</td><td>70.61</td><td>18.20</td><td>56.56</td><td>1.98</td><td>8.42</td></tr><tr><td>TempAct-1.3B*</td><td>44.68</td><td>71.06</td><td>100.00</td><td>97.81</td><td>15.23</td><td>65.09</td><td>9.11</td><td>29.68</td><td>0.50</td><td>3.98</td></tr><tr><td>Bernini-14B*</td><td>45.12</td><td>63.13</td><td>93.20</td><td>98.23</td><td>11.91</td><td>53.48</td><td>5.30</td><td>22.06</td><td>0.00</td><td>1.96</td></tr><tr><td>WorldGuide-8.3B (ours)</td><td>49.36</td><td>64.77</td><td>98.06</td><td>98.96</td><td>31.13</td><td>73.85</td><td>62.85</td><td>92.42</td><td>6.06</td><td>33.33</td></tr></table>

## 4 Experiments

Open-loop video generation follows instructions and a duration chosen before rollout, without checking intermediate task progress or deciding when the goal has been reached. Planner–executor systems can introduce feedback, but some use pretrained components without procedural training, while others plan the action sequence before rendering. WorldGuide trains both the ContextPlanner and Executor on procedural demonstrations to select and carry out atomic actions toward a long-horizon goal. Generated outcomes then guide the next action and the decision to stop.

This formulation raises five empirical questions. First, does closed-loop execution improve procedural task completion? Second, does generated visual feedback improve replanning? Third, can WorldGuide recognize completion and stop? Fourth, how do planning and execution failures interact? Fifth, does improving procedural correctness come at the cost of visual quality?

## 4.1 Experimental Setup

We fine-tune Qwen2.5-VL-7B-Instruct as the ContextPlanner and HunyuanVideo-1.5 as the Executor, on WorldGuide Bench. The planner uses K=3 recent clip-action pairs. We evaluate 980 test samples from WorldGuide Bench and all 294 Video-CraftBench samples [28]. Training and inference settings are given in Appendix D. On WorldGuide Bench, starred baselines (Bernini, PhysAgent, and TempAct) use their own planning-execution loops. WorldGuide receives only the initial image and task goal. On Video-CraftBench, all models receive the initial image and goal without reference actions. Appendix E.1 details the protocols and ablations.

Table 3: Video-CraftBench: goal-conditioned generation. All models receive the initial image and task goal without reference actions. ∗ denotes methods with their own planning–execution loops. Higher is better except Repeat/Skip.
<table><tr><td rowspan="2">Model</td><td colspan="5">Video Quality Check</td><td colspan="5">Video Planning Check</td></tr><tr><td>Aesth.</td><td>Imag.</td><td>Dynamic</td><td>Smooth.</td><td>Consist.</td><td>Align.</td><td>Plan Acc.</td><td>Order</td><td>Repeat/Skip ↓</td><td>Task Succ.</td></tr><tr><td>Cosmos-Predict2.5-2B</td><td>35.83</td><td>63.66</td><td>95.92</td><td>98.61</td><td>19.49</td><td>63.54</td><td>5.66</td><td>32.42</td><td>5.46</td><td>2.05</td></tr><tr><td>Open-Sora 2.0</td><td>33.86</td><td>30.26</td><td>47.62</td><td>99.04</td><td>21.61</td><td>52.97</td><td>22.03</td><td>82.96</td><td>2.05</td><td>14.32</td></tr><tr><td>HunyuanVideo-1.5</td><td>45.30</td><td>67.62</td><td>99.66</td><td>98.91</td><td>21.91</td><td>63.94</td><td>14.66</td><td>64.97</td><td>3.06</td><td>24.15</td></tr><tr><td>Wan2.2</td><td>46.05</td><td>72.47</td><td>100.00</td><td>98.31</td><td>22.28</td><td>71.98</td><td>2.98</td><td>18.22</td><td>4.11</td><td>2.40</td></tr><tr><td>CogVideoX</td><td>40.60</td><td>63.49</td><td>89.46</td><td>98.61</td><td>21.52</td><td>75.13</td><td>15.04</td><td>69.73</td><td>8.16</td><td>2.38</td></tr><tr><td>Helios</td><td>36.75</td><td>49.98</td><td>92.18</td><td>99.20</td><td>19.13</td><td>54.76</td><td>13.83</td><td>80.56</td><td>1.36</td><td>6.46</td></tr><tr><td>UniVideo</td><td>45.05</td><td>51.71</td><td>62.46</td><td>97.52</td><td>20.99</td><td>60.41</td><td>9.77</td><td>50.55</td><td>4.03</td><td>12.09</td></tr><tr><td>FlashMotion</td><td>43.78</td><td>67.29</td><td>99.32</td><td>98.83</td><td>20.73</td><td>59.19</td><td>7.32</td><td>38.89</td><td>1.70</td><td>3.74</td></tr><tr><td>MAGI-1</td><td>36.72</td><td>75.40</td><td>0.00</td><td>99.61</td><td>16.40</td><td>71.67</td><td>6.62</td><td>75.00</td><td>8.50</td><td>0.68</td></tr><tr><td>RoboMaster</td><td>35.98</td><td>71.30</td><td>100.00</td><td>99.26</td><td>18.59</td><td>65.75</td><td>3.36</td><td>15.58</td><td>1.37</td><td>0.68</td></tr><tr><td>SpMem</td><td>36.17</td><td>62.39</td><td>94.90</td><td>99.18</td><td>20.11</td><td>65.49</td><td>11.72</td><td>55.11</td><td>6.46</td><td>3.74</td></tr><tr><td>Astra</td><td>27.80</td><td>38.23</td><td>95.58</td><td>94.29</td><td>7.94</td><td>55.56</td><td>3.11</td><td>47.96</td><td>6.80</td><td>0.00</td></tr><tr><td>Yume-1.5</td><td>39.99</td><td>62.40</td><td>99.66</td><td>99.26</td><td>23.13</td><td>67.17</td><td>5.36</td><td>29.58</td><td>3.46</td><td>5.88</td></tr><tr><td>SkyReels-V3</td><td>42.01</td><td>72.14</td><td>41.84</td><td>99.59</td><td>19.74</td><td>64.50</td><td>8.29</td><td>58.80</td><td>1.74</td><td>1.04</td></tr><tr><td>MiniMax-H31</td><td>40.65</td><td>69.65</td><td>100.00</td><td>99.14</td><td>23.18</td><td>68.84</td><td>28.64</td><td>88.55</td><td>2.16</td><td>32.73</td></tr><tr><td>LTX-2.5</td><td>41.60</td><td>68.30</td><td>85.71</td><td>99.46</td><td>21.37</td><td>66.22</td><td>6.59</td><td>46.08</td><td>11.68</td><td>3.44</td></tr><tr><td>HY-WorldPlay</td><td>35.03</td><td>63.84</td><td>62.24</td><td>99.03</td><td>13.66</td><td>66.17</td><td>2.77</td><td>38.74</td><td>1.37</td><td>0.00</td></tr><tr><td>PhysAgent*</td><td>39.08</td><td>67.69</td><td>88.10</td><td>97.69</td><td>18.86</td><td>68.61</td><td>16.62</td><td>83.13</td><td>1.40</td><td>1.05</td></tr><tr><td>TempAct*</td><td>42.43</td><td>72.22</td><td>100.00</td><td>98.32</td><td>23.54</td><td>64.71</td><td>7.57</td><td>51.26</td><td>0.34</td><td>24.47</td></tr><tr><td>Bernini*</td><td>41.14</td><td>61.84</td><td>98.26</td><td>98.15</td><td>16.47</td><td>58.36</td><td>5.56</td><td>29.51</td><td>1.77</td><td>7.77</td></tr><tr><td>WorldGuide (ours)</td><td>40.19</td><td>64.63</td><td>99.65</td><td>98.86</td><td>22.09</td><td>71.80</td><td>57.42</td><td>94.45</td><td>17.08</td><td>47.69</td></tr></table>

Metrics. We assess visual quality and video-text consistency using VBench [50], and measure similarity between generated and reference videos using Paired Alignment. We use Gemini to assess task execution in generated videos. Task Success Rate measures final goal achievement, and Plan Accuracy measures completion of required sub-goals. Order Score evaluates prerequisite order, while Repeat/Skip penalizes unnecessary repetition. Finally, GPT-5.2 evaluates the predicted action sequences. Its Plan Success score indicates whether a sequence would achieve the goal if executed correctly. Full metric definitions and evaluation prompts are provided in Appendix E.

## 4.2 Does Closed-Loop Execution Improve Procedural Task Completion?

Open-loop generation commits to an action sequence and rollout length before execution, preventing subsequent decisions from responding to incomplete or incorrect actions. WorldGuide instead predicts each action-consequences and termination from generated progress. Compared with its open-loop variant, this formulation raises Task Success from 11.71% to 33.33% and Plan Accuracy from 22.87 to 62.85 (Table 4). WorldGuide also outperforms MiniMax-H3 on WorldGuide Bench despite its access to reference actions (29.90% Task Success; Table 2), and on Video-CraftBench under goal-only conditioning (47.69% versus 32.73%; Table 3). A human study confirms this ranking (Appendix F).

However, closing the loop alone does not guarantee task completion: the planner must select appropriate atomic actions, and the executor must realize their consequences. As shown in Table 2 and 3, WorldGuide consistently out performs other closed-loop planner–executor methods, including PhysAgent, TempAct, and Bernini, demonstrating that procedural training and grounding each decision in generated outcomes are critical for reliable completion.

## 4.3 Does Generated Visual Feedback Improve Replanning?

A closed-loop planner is only useful if it observes what was actually executed. To isolate this effect, we remove generated clips from the planner input, keeping only the task goal and previous actions; the Executor is unchanged. Restoring visual feedback raises Task Success from 14.72% to 33.33% and Plan Accuracy from 29.29 to 62.85 (Table 4). This shows action history alone is insufficient: knowing what the planner requested does not reveal whether the action was completed, partially realized, or drifted visually. Observing the generated outcome lets WorldGuide replan from the state that exists, so later decisions can compensate for execution errors and stay aligned with task progress.

Table 4: WorldGuide ablations. Removing step-wise replanning (open-loop), planner visual feedback (w/o visual), or Executor memory (w/o Mem) lowers Task Success; oracle ContextPlanner substitutes reference actions to isolate planning errors. Protocols follow Tables 2 and 3 (Appendix E.1).
<table><tr><td rowspan="2">Variant</td><td colspan="5">Video Quality Check</td><td colspan="5">Video Planning Check</td></tr><tr><td>Aesth.</td><td>Imag.</td><td>Dynamic</td><td>Smooth.</td><td>Consist.</td><td>Align.</td><td>Plan Acc.</td><td>Order</td><td>Repeat/Skip ↓</td><td>Task Succ.</td></tr><tr><td>WorldGuide Bench</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>WorldGuide (open-loop)</td><td>36.76</td><td>59.38</td><td>73.30</td><td>98.42</td><td>18.86</td><td>67.40</td><td>22.87</td><td>53.81</td><td>0.49</td><td>11.71</td></tr><tr><td>WorldGuide (w/o visual)</td><td>36.77</td><td>58.74</td><td>83.92</td><td>98.25</td><td>18.67</td><td>68.11</td><td>29.29</td><td>68.70</td><td>0.08</td><td>14.72</td></tr><tr><td>WorldGuide (w/o Mem)</td><td>38.70</td><td>55.35</td><td>88.35</td><td>98.14</td><td>19.70</td><td>64.74</td><td>55.36</td><td>91.12</td><td>4.39</td><td>27.56</td></tr><tr><td>WorldGuide (oracle ContextPlanner)</td><td>52.44</td><td>66.72</td><td>97.68</td><td>99.49</td><td>33.37</td><td>77.14</td><td>65.42</td><td>94.28</td><td>2.04</td><td>38.82</td></tr><tr><td>WorldGuide (ours)</td><td>49.36</td><td>64.77</td><td>98.06</td><td>98.96</td><td>31.13</td><td>73.85</td><td>62.85</td><td>92.42</td><td>6.06</td><td>33.33</td></tr><tr><td>Video-CraftBench</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>WorldGuide (w/o Mem)</td><td>38.82</td><td>63.81</td><td>99.63</td><td>98.64</td><td>20.21</td><td>73.77</td><td>39.62</td><td>92.51</td><td>4.10</td><td>29.78</td></tr><tr><td>WorldGuide (ours)</td><td>40.19</td><td>64.63</td><td>99.65</td><td>98.86</td><td>22.09</td><td>71.80</td><td>57.42</td><td>94.45</td><td>17.08</td><td>47.69</td></tr></table>

Table 5: ContextPlanner evaluation on WorldGuide Bench. Comparison of ground-truth-history planning against autonomous rollout with generated history; visual feedback improves plan quality during autonomous execution. FT: supervised fine-tuning; K: history length.
<table><tr><td>Setting</td><td>FT</td><td>K</td><td>Plan Acc. ↑</td><td>Order ↑</td><td>Repeat/Skip↓</td><td>Success ↑</td></tr><tr><td>Qwen2.5-VL baseline</td><td>米</td><td>3</td><td>1.04</td><td>0.00</td><td>8.82</td><td>0.00</td></tr><tr><td>ContextPlanner</td><td>S</td><td>1</td><td>56.48</td><td>42.35</td><td>21.16</td><td>39.17</td></tr><tr><td>ContextPlanner</td><td>S</td><td>2</td><td>75.83</td><td>71.28</td><td>15.56</td><td>44.66</td></tr><tr><td>ContextPlanner</td><td>S</td><td>3</td><td>79.66</td><td>72.56</td><td>14.71</td><td>47.04</td></tr><tr><td>WorldGuide (w/o visual)</td><td>S</td><td>3</td><td>63.84</td><td>77.46</td><td>20.19</td><td>86.41</td></tr><tr><td>WorldGuide planner-executor</td><td>S</td><td>3</td><td>64.03</td><td>79.45</td><td>15.73</td><td>91.75</td></tr></table>

## 4.4 Can WorldGuide Recognize Completion and Stop?

Completing a task requires knowing not only what to do next, but when to stop. The same ContextPlanner handles both, predicting either the next atomic action or a completion token from the rollout history. In autonomous execution, using generated visual feedback improves Plan Success from 86.41% to 91.75% and reduces Repeat/Skip from 20.19 to 15.73 (Table 5). This shows that visual observations help the planner distinguish genuine task progress from merely knowing which actions were previously requested, leading to more complete and less redundant procedures.

With ground-truth histories, the planner reaches 79.66 Plan Accuracy at K = 3, compared with 64.03 in autonomous rollout, where every decision depends on its generated outcomes. Despite this harder setting, Plan Success still reaches 91.75%, showing WorldGuide can construct and terminate valid procedures from generated progress. Plan Success measures whether the predicted plan would achieve the goal if executed correctly. Visual completion is measured separately by the video-judged Task Success (Tables 2–4).

## 4.5 How Do Planning and Execution Failures Interact?

Planning and execution errors are tightly coupled in closed-loop generation. Replacing the learned planner with oracle reference actions improves Task Success by +5.49 % (Table 4), using the same trained Executor and memory, showing stronger planning can push task completion further.

Execution quality also affects what the planner sees next. With the ContextPlanner and Executor checkpoints unchanged, enabling memory improves Task Success by +5.77% on WorldGuide Bench and +17.91% on Video

![](images/3a52afc232c4a0e2c168bd87e92040f14f5016b83d529f9520a1c54a678f3e29.jpg)  
Task : step by step fried rice tutorial

Figure 5: Qualitative comparison for “step-by-step fried rice tutorial”. MiniMax-H3, LTX-2.5, HunyuanVideo, and Cosmos-Predict2.5 receive reference action sequences; Bernini and TempAct use their own planning–execution loops. WorldGuide predicts actions from the task goal and generated progress. Frames are uniformly sampled from each trajectory.

CraftBench, and video-judged Plan Accuracy by +7.49% (Table 4). Since each generated state feeds the next planning step, execution errors can corrupt later evidence, underscoring how tightly planning and execution depend on each other in closed-loop task completion.

## 4.6 Does Procedural Improvement Compromise Visual Quality?

WorldGuide primarily targets reliable procedural completion, so its gains do not uniformly extend to every lowlevel video-quality metric. On WorldGuide Bench, it achieves the highest Aesthetic Quality (49.36) and Overall Consistency (31.13), while baselines like TempAct and SkyReels score higher on individual metrics such as Imaging Quality, Dynamic Degree, or Motion Smoothness (Table 2), consistent with differences in backbone scale, pretraining data, and optimization for visual fidelity.

More importantly, WorldGuide substantially improves Task Success, reaching 33.33% on WorldGuide Bench and 47.69% on Video-CraftBench. On Video-CraftBench, this comes with modest drops in some visual-quality metrics relative to MiniMax-H3, including Imaging Quality (64.63 vs. 69.65) and Overall Consistency (22.09 vs. 23.18). WorldGuide thus prioritizes task execution over visual fidelity in challenging cases, and improving this balance remains an important direction.

## 4.7 Qualitative Results

In Fig. 5, WorldGuide adds aromatics, vegetables, and rice before seasoning and mixing. Its text actions identify which ingredients to add and in what order. MiniMax-H3 shows seasoned rice early, then adds white rice again and changes cookware. TempAct and LTX-2.5 largely remain at ingredient handling, while Bernini loses scene consistency. These examples show why plausible individual actions are insufficient for a coherent procedure. Additional tasks appear in Appendix G.1.

## 5 Discussion & Limitations

WorldGuide shows that closed-loop procedural video generation substantially improves task completion, though the remaining gap points to room for more robust long-horizon execution. Failures typically arise from a premature or redundant planner action, or an Executor clip that does not fully realize the intended state transition; since each generated state feeds the next decision, such errors can propagate. The gains from visual feedback and memory show that better state tracking directly improves both planning and execution.

A promising direction is to strengthen this coupling further with richer state-validity signals and explicit recovery when a step is incomplete or inconsistent. The Executor already conditions on the ContextPlanner’s action embeddings; joint optimization of both modules is a natural next step. Bounded visual memory compresses older states to control inference cost, and more selective retention of task-relevant information could improve long-horizon consistency while keeping this efficiency. These directions extend the current closed-loop formulation rather than requiring a different one.

## 6 Conclusion

We introduced WorldGuide, a video world model that turns procedural video generation into closed-loop task execution. The central idea is that a procedure should unfold through decisions grounded in the generated visual state: what to do next and when to stop depend on what has actually been accomplished. WorldGuide implements this principle by repeatedly selecting a semantic action, rendering its consequence, and using the resulting visual state to guide subsequent execution. Compressed visual memory supports state continuity across successive steps of long-horizon, goal-directed execution. We also introduced WorldGuide Bench, comprising approximately 59K annotated videos across 245 tasks and 27 procedural categories. Among the evaluated video generators and video world models, WorldGuide achieves the highest Task Success on both WorldGuide Bench and Video-CraftBench. Despite remaining execution and stopping errors, these results support a shift from open-loop future synthesis to closed-loop task execution. Our work takes a step towards how video world models can use their generated consequences to decide what to do next and when to stop, guiding step-by-step execution toward the final goal.

## References

[1] Triantafyllos Afouras, Efrosyni Mavroudi, Tushar Nagarajan, Huiyu Wang, and Lorenzo Torresani. Ht-step: Aligning instructional articles with how-to videos. Advances in Neural Information Processing Systems, 36: 50310–50326, 2023.

[2] Arslan Ali, Junjie Bai, Maciej Bala, Yogesh Balaji, Aaron Blakeman, Tifany Cai, Jiaxin Cao, Tianshi Cao, Elizabeth Cha, Yu-Wei Chao, et al. World simulation with video foundation models for physical ai. arXiv preprint arXiv:2511.00062, 2025.

[3] Shuai Bai, Keqin Chen, Xuejing Liu, Jialin Wang, Wenbin Ge, Sibo Song, Kai Dang, Peng Wang, Shijie Wang, Jun Tang, Humen Zhong, Yuanzhi Zhu, Mingkun Yang, Zhaohai Li, Jianqiang Wan, Pengfei Wang, Wei Ding, Zheren Fu, Yiheng Xu, Jiabo Ye, Xi Zhang, Tianbao Xie, Zesen Cheng, Hang Zhang, Zhibo Yang, Haiyang Xu, and Junyang Lin. Qwen2.5-vl technical report, 2025. URL https://arxiv.org/abs/2502.13923.

[4] Dima Damen, Hazel Doughty, Giovanni Maria Farinella, Sanja Fidler, Antonino Furnari, Evangelos Kazakos, Davide Moltisanti, Jonathan Munro, Toby Perrett, Will Price, et al. The epic-kitchens dataset: Collection, challenges and baselines. IEEE Transactions on Pattern Analysis and Machine Intelligence, 43(11):4125–4141, 2020.

[5] Haoge Deng, Ting Pan, Haiwen Diao, Zhengxiong Luo, Yufeng Cui, Huchuan Lu, Shiguang Shan, Yonggang Qi, and Xinlong Wang. Autoregressive video generation without vector quantization. arXiv preprint arXiv:2412.14169, 2024.

[6] Yilun Du, Sherry Yang, Pete Florence, Fei Xia, Ayzaan Wahid, Pierre Sermanet, Tianhe Yu, Pieter Abbeel, Joshua B Tenenbaum, Leslie Kaelbling, et al. Video language planning. In International Conference on Learning Representations, volume 2024, pages 31138–31155, 2024.

[7] Jiahui Fu, Junyu Nan, Lingfeng Sun, Hongyu Li, Jianing Qian, Jennifer L Barry, Kris Kitani, and George Konidaris. Novaplan: Zero-shot long-horizon manipulation via closed-loop video language planning. arXiv preprint arXiv:2602.20119, 2026.

[8] Xiao Fu, Xintao Wang, Xian Liu, Jianhong Bai, Runsen Xu, Pengfei Wan, Di Zhang, and Dahua Lin. Learning video generation for robotic manipulation with collaborative trajectory control. In International Conference on Learning Representations, volume 2026, pages 144128–144142, 2026.

[9] Gemini Team, Google. Gemini 2.5: Pushing the frontier with advanced reasoning, multimodality, long context, and next generation agentic capabilities. arXiv preprint arXiv:2507.06261, 2025.

[10] Google DeepMind. Veo. https://deepmind.google/models/veo/, 2025. Accessed May 2, 2026.

[11] Google DeepMind. Gemini 3.5 flash. Model Card, 2026. URL https://deepmind.google/models/mod el-cards/gemini-3-5-flash/. Published May 19, 2026.

[12] Kristen Grauman, Andrew Westbury, Lorenzo Torresani, Kris Kitani, Jitendra Malik, Triantafyllos Afouras, Kumar Ashutosh, Vijay Baiyya, Siddhant Bansal, Bikram Boote, et al. Ego-exo4d: Understanding skilled human activity from first-and third-person perspectives. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 19383–19400, 2024.

[13] Yoav HaCohen, Benny Brazowski, Nisan Chiprut, Yaki Bitterman, Andrew Kvochko, Avishai Berkowitz, Daniel Shalem, Daphna Lifschitz, Dudu Moshe, Eitan Porat, et al. Ltx-2: Eficient joint audio-visual foundation model. arXiv preprint arXiv:2601.03233, 2026.

[14] Xuanhua He, Tianyu Yang, Ke Cao, Ruiqi Wu, Cheng Meng, Yong Zhang, Zhuoliang Kang, Xiaoming Wei, and Qifeng Chen. Active intelligence in video avatars via closed-loop world modeling. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 27239–27248, 2026.

[15] Joowon Kim, Seungho Shin, Joonhyung Park, and Eunho Yang. Collabvr: Collaborative video reasoning with vision-language and video generation models. arXiv preprint arXiv:2605.08735, 2026.

[16] Moo Jin Kim, Karl Pertsch, Siddharth Karamcheti, Ted Xiao, Ashwin Balakrishna, Suraj Nair, Rafael Rafailov, Ethan Foster, Grace Lam, Pannag Sanketi, et al. Openvla: An open-source vision-language-action model. arXiv preprint arXiv:2406.09246, 2024.

[17] Weijie Kong, Qi Tian, Zijian Zhang, Rox Min, Zuozhuo Dai, Jin Zhou, Jiangfeng Xiong, Xin Li, Bo Wu, Jianwei Zhang, et al. Hunyuanvideo: A systematic framework for large video generative models. arXiv preprint arXiv:2412.03603, 2024.

[18] Debang Li, Zhengcong Fei, Tuanhui Li, Yikun Dou, Zheng Chen, Jiangping Yang, Mingyuan Fan, Jingtao Xu, Jiahua Wang, Baoxuan Gu, et al. Skyreels-v3 technique report. arXiv preprint arXiv:2601.17323, 2026.

[19] Qirui Li, Jinkun Hao, Yibo Li, Ran Yi, Paul L Rosin, and Yu-Kun Lai. Physagent: Reflective agentic physics control for physically plausible video generation. arXiv preprint arXiv:2607.16355, 2026.

[20] Quanhao Li, Zhen Xing, Rui Wang, Haidong Cao, Qi Dai, Daoguo Dong, and Zuxuan Wu. Flashmotion: Few-step controllable video generation with trajectory guidance. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 8986–8996, 2026.

[21] Xiaofeng Mao, Zhen Li, Chuanhao Li, Xiaojie Xu, Kaining Ying, Tong He, Jiangmiao Pang, Yu Qiao, and Kaipeng Zhang. Yume-1.5: A text-controlled interactive world generation model. arXiv preprint arXiv:2512.22096, 2025.

[22] Xiaofeng Mao, Shaoheng Lin, Zhen Li, Chuanhao Li, Wenshuo Peng, Tong He, Jiangmiao Pang, Mingmin Chi, Yu Qiao, and Kaipeng Zhang. Yume: An interactive world generation model. arXiv preprint arXiv:2507.17744, 2025.

[23] Antoine Miech, Dimitri Zhukov, Jean-Baptiste Alayrac, Makarand Tapaswi, Ivan Laptev, and Josef Sivic. Howto100m: Learning a text-video embedding by watching hundred million narrated video clips. In 2019 IEEE/CVF International Conference on Computer Vision (ICCV), pages 2630–2640. IEEE, 2019.

[24] OpenAI. Gpt-5.2 system card. OpenAI, December 2025. URL https://openai.com/index/gpt-5-sys tem-card-update-gpt-5-2/. Published December 11, 2025.

[25] OpenAI. Sora 2: Advancing video generation models. https://openai.com/index/sora-2/, 2025. OpenAI research release, published September 30, 2025. Accessed November 11, 2025.

[26] Rohith Peddi, Shivvrat Arya, Bharath Challa, Likhitha Pallapothula, Akshay Vyas, Bhavya Gouripeddi, Qifan Zhang, Jikai Wang, Vasundhara Komaragiri, Eric Ragan, et al. Captaincook4d: A dataset for understanding errors in procedural activities. Advances in Neural Information Processing Systems, 37:135626–135679, 2024.

[27] Lu Qiu, Yi Chen, Yuying Ge, Yixiao Ge, Ying Shan, and Xihui Liu. Egoplan-bench2: A benchmark for multimodal large language model planning in real-world scenarios. International Journal of Computer Vision, 134(5):222, 2026.

[28] Zhongwei Ren, Yunchao Wei, Xiao Yu, Guixun Luo, Yao Zhao, Bingyi Kang, Jiashi Feng, and Xiaojie Jin. Videoworld 2: Learning transferable knowledge from real-world videos. arXiv preprint arXiv:2602.10102, 2026.

[29] Fadime Sener, Dibyadip Chatterjee, Daniel Shelepov, Kun He, Dipika Singhania, Robert Wang, and Angela Yao. Assembly101: A large-scale multi-view video dataset for understanding procedural activities. In 2022 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 21064–21074. IEEE, 2022.

[30] Yifan Shen, Jiateng Liu, Xinzhuo Li, Yuanzhe Liu, Bingxuan Li, Houze Yang, Wenqi Jia, Yijiang Li, Tianjiao Yu, James Matthew Rehg, et al. Egoforge: Goal-directed egocentric world simulator. arXiv preprint arXiv:2603.20169, 2026.

[31] Yale Song, Eugene Byrne, Tushar Nagarajan, Huiyu Wang, Miguel Martin, and Lorenzo Torresani. Ego4d goalstep: Toward hierarchical understanding of procedural activities. Advances in neural information processing systems, 36:38863–38886, 2023.

[32] Wenqiang Sun, Haiyu Zhang, Haoyuan Wang, Junta Wu, Zehan Wang, Zhenwei Wang, Yunhong Wang, Jun Zhang, Tengfei Wang, and Chunchao Guo. Worldplay: Towards long-term geometric consistency for real-time interactive world modeling. arXiv preprint arXiv:2512.14614, 2025.

[33] Yansong Tang, Dajun Ding, Yongming Rao, Yu Zheng, Danyang Zhang, Lili Zhao, Jiwen Lu, and Jie Zhou. Coin: A large-scale dataset for comprehensive instructional video analysis. In 2019 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 1207–1216. IEEE, 2019.

[34] Bernini Team, Chenchen Liu, Junyi Chen, Lei Li, Lu Chi, Mingzhen Sun, Zhuoying Li, Yi Fu, Ruoyu Guo, Yiheng Wu, et al. Bernini: Latent semantic planning for video difusion. arXiv preprint arXiv:2605.22344, 2026.

[35] Hansi Teng, Hongyu Jia, Lei Sun, Lingzhi Li, Maolin Li, Mingqiu Tang, Shuai Han, Tianning Zhang, WQ Zhang, Weifeng Luo, et al. Magi-1: Autoregressive video generation at scale. arXiv preprint arXiv:2505.13211, 2025.

[36] Xiaoyu Tian, Haotian Wang, Shuaiting Chen, Hao Zhou, Kaichi Yu, Yudian Zhang, Jade Ouyang, Junxi Yin, Jiong Chen, Baoyan Guo, et al. Astra: Automated synthesis of agentic trajectories and reinforcement arenas. arXiv preprint arXiv:2601.21558, 2026.

[37] Keyu Tu, Zhuowei Chen, Mengqi Huang, Yuxin Wang, Jiahao Zhu, Zhendong Mao, and Yongdong Zhang. Semcomp-bench: Benchmarking semantic task completion in video generation. arXiv preprint arXiv:2608.17426, 2026.

[38] Team Wan, Ang Wang, Baole Ai, Bin Wen, Chaojie Mao, Chen-Wei Xie, Di Chen, Feiwu Yu, Haiming Zhao, Jianxiao Yang, et al. Wan: Open and advanced large-scale video generative models. arXiv preprint arXiv:2503.20314, 2025.

[39] Jing Wang, Xiangxin Zhou, Jiajun Liang, Kaiqi Liu, Wanyuan Pang, Zhenyu Xie, Tianyu Pang, and Xiaodan Liang. Tempact: Advancing temporal plausibility in autoregressive video generation via planner-executor rl. arXiv preprint arXiv:2606.28016, 2026.

[40] Cong Wei, Quande Liu, Zixuan Ye, Qiulin Wang, Xintao Wang, Pengfei Wan, Kun Gai, and Wenhu Chen. Univideo: Unified understanding, generation, and editing for videos. In International Conference on Learning Representations, volume 2026, pages 113905–113933, 2026.

[41] Bing Wu, Chang Zou, Changlin Li, Duojun Huang, Fang Yang, Hao Tan, Jack Peng, Jianbing Wu, Jiangfeng Xiong, Jie Jiang, et al. Hunyuanvideo 1.5 technical report. arXiv preprint arXiv:2511.18870, 2025.

[42] Tong Wu, Shuai Yang, Ryan Po, Yinghao Xu, Ziwei Liu, Dahua Lin, and Gordon Wetzstein. Video world models with long-term spatial memory. Advances in Neural Information Processing Systems, 38:49371–49393, 2026.

[43] Linting Xue, Aditya Barua, Noah Constant, Rami Al-Rfou, Sharan Narang, Mihir Kale, Adam Roberts, and Colin Rafel. Byt5: Towards a token-free future with pre-trained byte-to-byte models, 2022. URL https: //arxiv.org/abs/2105.13626.

[44] Yu Yang, Yue Liao, Jianbiao Mei, Baisen Wang, Xuemeng Yang, Licheng Wen, Jiangning Zhang, Xiangtai Li, Liang Lv, Hanlin Chen, et al. Spiral: Self-evolving action-conditioned video generation via reflective planning agents. arXiv preprint arXiv:2603.08403, 2026.

[45] Zhuoyi Yang, Jiayan Teng, Wendi Zheng, Ming Ding, Shiyu Huang, Jiazheng Xu, Yuanming Yang, Wenyi Hong, Xiaohan Zhang, Guanyu Feng, et al. Cogvideox: Text-to-video difusion models with an expert transformer. In International Conference on Learning Representations, volume 2025, pages 83048–83077, 2025.

[46] Hangjie Yuan, Weihua Chen, Jun Cen, Hu Yu, Jingyun Liang, Shuning Chang, Zhihui Lin, Tao Feng, Pengwei Liu, Jiazheng Xing, et al. Lumos-1: On autoregressive video generation from a unified model perspective. arXiv e-prints, pages arXiv–2507, 2025.

[47] Shenghai Yuan, Yuanyang Yin, Zongjian Li, Xinwei Huang, Xiao Yang, and Li Yuan. Helios: Real real-time long video generation model. arXiv preprint arXiv:2603.04379, 2026.

[48] Lvmin Zhang, Shengqu Cai, Muyang Li, Gordon Wetzstein, and Maneesh Agrawala. Frame context packing and drift prevention in next-frame-prediction video difusion models. Advances in Neural Information Processing Systems, 38:30546–30566, 2026.

[49] Wancong Zhang, Basile Terver, Artem Zholus, Soham Chitnis, Harsh Sutaria, Mido Assran, Randall Balestriero, Amir Bar, Adrien Bardes, Yann LeCun, et al. Hierarchical planning with latent world models. arXiv preprint arXiv:2604.03208, 2026.

[50] Dian Zheng, Ziqi Huang, Hongbo Liu, Kai Zou, Yinan He, Fan Zhang, Lulu Gu, Yuanhan Zhang, Jingwen He, Wei-Shi Zheng, et al. Vbench-2.0: Advancing video generation benchmark suite for intrinsic faithfulness. arXiv preprint arXiv:2503.21755, 2025.

[51] Zangwei Zheng, Xiangyu Peng, Yuxuan Lou, Chenhui Shen, Tom Young, Xinying Guo, Binluo Wang, Hang Xu, Hongxin Liu, Mingyan Jiang, et al. Open-sora 2.0: Training a commercial-level video generation model in \$200k. arXiv preprint arXiv:2503.09642, 2025.

[52] Luowei Zhou, Chenliang Xu, and Jason Corso. Towards automatic learning of procedures from web instructional videos. In Proceedings of the AAAI conference on artificial intelligence, volume 32, 2018.

[53] Dimitri Zhukov, Jean-Baptiste Alayrac, Ramazan Gokberk Cinbis, David Fouhey, Ivan Laptev, and Josef Sivic. Cross-task weakly supervised learning from instructional videos. In 2019 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 3532–3540. IEEE, 2019.

[54] Brianna Zitkovich, Tianhe Yu, Sichun Xu, Peng Xu, Ted Xiao, Fei Xia, Jialin Wu, Paul Wohlhart, Stefan Welker, Ayzaan Wahid, et al. Rt-2: Vision-language-action models transfer web knowledge to robotic control. In Conference on Robot Learning, pages 2165–2183. PMLR, 2023.

A. Design Rationale . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 16   
B. WorldGuide Bench . . .   
B.1. Collection and Filtering . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 17   
B.2. Temporal Action Annotation . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 17   
B.3. Training Samples . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 17   
B.4. Statistics and Evaluation Split . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 17   
B.5. Annotation Example . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 17   
B.6. Human Audit of Annotations . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 19   
C. Hierarchical Memory . . . .   
C.1. History and Reference Conditioning . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 21   
C.2. Multi-Scale Embedding and Injection . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 21   
C.3. Compression Schedule and Token Budget . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 22   
D. Training and Inference . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 22   
D.2. Executor Training . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 22   
D.3. Rollout and Termination . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 23   
E.1. Benchmarks, Baselines, and Ablations . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 23   
E.4. Paired Alignment . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 27   
E.5. VBench Video-Quality Metrics . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 28   
F. Human Evaluation of Generated Videos . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 28   
G. Additional Results . . . . . . . . . . . . . . . . . . . . . . . . . . . . .   
G.1. WorldGuide Bench Qualitative Comparisons . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 28   
G.2. Video-CraftBench Results . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .   
H. Limitations and Failure Analysis. . . . . . . 29

Interactive qualitative results: Our project page provides complete rollouts of WorldGuide and the baselines side by side, together with additional WorldGuide Bench videos and their step-level annotations: https://mbzu ai-oryx.github.io/WorldGuide. These pages are supplementary qualitative material; all quantitative results and conclusions are reported in the paper.

## A Design Rationale

This section explains three design choices of WorldGuide: the choice of backbones, the sequential training of the two modules, and the language-level interface between planning and execution.

One model for planning and action encoding. HunyuanVideo-1.5 [41] encodes its text prompts with Qwen2.5-VL-7B-Instruct [3]. We therefore initialize the ContextPlanner from this same model, so that a single network both predicts the next action and encodes it for the Executor. This choice has three advantages. First, the Executor receives action embeddings from the model it was pretrained with, so fine-tuning adapts an existing interface rather than learning a new one. Second, WorldGuide adds no separate planning network: the ContextPlanner takes the place ofthe Executor’s text encoder, keeping the complete system at the size ofthe original HunyuanVideo-1.5 pipeline. Third, the action predicted by the planner at inference reaches the Executor through exactly the embedding pathway used during training.

Choice of Executor. HunyuanVideo-1.5 has 8.3B parameters, fewer than many of the evaluated generators, yet it is among the strongest at procedural execution. Given reference actions on WorldGuide Bench, it achieves the highest Plan Accuracy (43.53) and Task Success (12.00%) of all baselines except the 33B MiniMax-H3, ahead of larger models such as Wan2.2-14B, LTX-2.5-22B, and FlashMotion-34B (Table 2). Building on it keeps WorldGuide compact while starting from a strong executor.

Sequential training. Both modules are trained on the same procedural demonstrations, but sequentially rather than end-to-end. The ContextPlanner is first fine-tuned for next-action and completion prediction; it is then frozen while the Executor is fine-tuned on action captions embedded by it (Appendix D). Freezing is deliberate. If the Executor’s flow-matching loss also updated the ContextPlanner, its parameters would be optimized to produce embeddings that ease video generation, with no signal that preserves its language outputs. Its predicted actions could then drift away from faithful, readable instructions. Freezing the ContextPlanner keeps the interface between planning and execution in natural language.

Why a language-level interface matters. WorldGuide represents each predicted action as a natural-language instruction and uses it to generate the corresponding video clip. This language interface has three main benefits:

• Interpretable guidance. Each generated clip is paired with a clear instruction describing the intended action. This makes the generated procedure easier to understand and follow.

• Diagnosable failures. Planning and execution errors can be analyzed separately. Since actions are represented as text, we can replace predicted actions with reference actions while keeping the Executor unchanged. We use this setting for the oracle ContextPlanner experiment in Table 4.

• Independent evaluation. The predicted action sequence can be evaluated directly as text, independently of the quality of the generated videos (Table 5).

In this work, we keep the language interface fixed during Executor training. Jointly optimizing the Planner and Executor while preserving clear language outputs is an interesting direction for future work (Sec. 5).

## B WorldGuide Bench

WorldGuide Bench supplies the three signals on which the closed loop is trained: which atomic action comes next, how that action changes the visual state, and when the task is complete. This section describes how videos are collected (Appendix B.1), annotated (Appendix B.2), converted into training samples (Appendix B.3), and split (Appendix B.4), followed by an annotation example (Appendix B.5) and a human audit (Appendix B.6).

Table 1 compares WorldGuide Bench with existing procedural video datasets. These target planning question answering (EgoPlan-IT [27]), step localization and grounding (Ego4D Goal-Step [31], COIN [33], CrossTask [53], HT-Step [1]), segmentation, recognition, and error detection (YouCook2 [52], Assembly101 [29], CaptainCook4D [26], Ego-Exo4D Keystep [12]), or video generation and its evaluation (SemComp-Data [37], ActVideoGen [44], Ego Forge [30]). None of them provides explicit task-completion supervision, which WorldGuide Bench adds to step-level atomic-action clips and task goals.

## B.1 Collection and Filtering

We collect public instructional videos of at most seven minutes from YouTube, covering 245 tasks in 27 procedural categories such as origami, cooking, construction, assembly and repair, sorting, and knotting. Each task is retrieved with task-specific queries expanded with short-form and multilingual variants in 15 languages, so every task has many demonstrations with different objects, scenes, and demonstrators. Files that fail integrity or frame-decoding checks are discarded.

Gemini 2.5 Flash [9] rates each video with a Video Planning Score from 0 to 5 for topical relevance and demonstration completeness (Fig. 6). We keep only videos scored 5, retaining 58,679 of 133,360 retrieved records (44.0%; Table 7). Annotation quality is verified by a human audit (Appendix B.6).

All source videos are publicly available and used for non-commercial research. We will release source identifiers, timestamps, and step annotations rather than the videos themselves.

## B.2 Temporal Action Annotation

Gemini converts each video into an ordered sequence of atomic physical actions. Each action is assigned a start and end timestamp, with a target duration of 1–5 seconds (Fig. 6). Actions longer than 5 seconds are split into multiple consecutive steps. The final step is marked as task completion. For videos longer than 5 minutes, we run annotation at 1.5× playback speed for efficiency and rescale the predicted timestamps back to the original video timeline. As a result, annotated steps can span 1.5–7.5 seconds in the original video. All clips are extracted from the original-speed videos, and empty or invalid clips are removed.

## B.3 Training Samples

Each annotated step becomes one training record (Table 6): the task context, the action caption, the target clip cut at the step’s timestamps, the step index and normalized progress, the preceding clips and captions, and a completion flag set on the final step. A one-second preview from the start of the source video provides the initial visual state. From these records, the ContextPlanner learns to predict the next caption from the goal and up to K=3 preceding clip–caption pairs, and the Executor learns to generate each clip from its caption and preceding visual context. To supervise stopping, preprocessing appends one additional record after the final step, whose target is <|Task Completed|> and whose history is the last K=3 clip–caption pairs, ending with the final step. This record has no target clip, so it trains only the ContextPlanner and matches inference, where the planner emits <|Task Completed|> after observing the last generated clip. Both modules are thus supervised by the same demonstrations, each with its own objective (Appendix A).

## B.4 Statistics and Evaluation Split

Table 7 reports the number of videos after each filtering stage. We hold out four videos per task, giving 980 test videos and 57,699 training videos; all clips of a held-out video are excluded from the training of both modules. The split therefore evaluates unseen demonstrations of known tasks.

## B.5 Annotation Example

Table 8 shows a complete annotation example from an origami video. Each video in WorldGuide Bench follows the same format: a task description, a short description of the work subject, and an ordered sequence of atomic actions with their start and end times. During preprocessing, we split the video into one clip for each annotated step.

Gemini Dataset-Captioning Prompt   
You are an expert temporal video captioning model.   
Task: Watch the video and convert it into a timeline of actionable steps.   
Temporal alignment between each step and its timestamp is very important.   
Main task/topic: {query}.   
Description: {description}.   
Video Planning Score:   
- 0: Video does not match the topic.   
- 1-2: Poor quality, missing steps, or hard to follow.   
- 3-4: Good quality, most steps visible.   
- 5: Excellent, complete step-by-step tutorial.   
Timeline Rules:   
- Output only the steps necessary to complete the task. Please align the   
tasks with the particular timeline.   
- Each step must be between 1 and 5 seconds. (Adjusted for long-video   
stability).   
- If an action takes longer than 5s, split it into consecutive steps.   
- Use precise verbs: Fold, Aligns, Presses, Flips, Rotates.   
- No descriptions of background, hands, or music.   
Important:   
- If Score = 0, output "Video Planning Score: 0" and stop.   
- If Score > 0, output the Score followed by the timeline.   
Strict output format:   
Video Planning Score: <0-5>   
Work subject: <one-line title describing the specific {task} of {category}   
shown in the video>   
Time: <start>-<end> sec, Step <N>: <one clear physical action>  
Figure 6: Per-video prompt supplied to Gemini 2.5 Flash for dataset captioning. The video, together with the retrieval query, task, category, and description, is passed alongside this prompt, prefixed by a fixed system instruction: “You are a professional technical writer. Your task is to provide a non-repetitive, high-accuracy timeline of physical actions from video.” Only videos assigned a Video Planning Score of 5 are retained.

Table 6: WorldGuide Bench sample format. Each record contains an annotated action, its corresponding video clip, and the preceding procedural context. The completion flag identifies the final step.
<table><tr><td>Information</td><td>Contents</td></tr><tr><td>Task context</td><td>Category, task label, and task goal.</td></tr><tr><td>Action annotation</td><td>Step index, action caption, and start/end timestamps in the original video.</td></tr><tr><td>Target clip</td><td>Video segment corresponding to the annotated action.</td></tr><tr><td>Preceding history</td><td>Previous clips and their action captions. The ContextPlanner uses up to K = 3 previous steps, while the Executor uses visual memory.</td></tr><tr><td>Progress</td><td>Step index normalized by the total number of annotated steps.</td></tr><tr><td>Completion</td><td>Flag marking the final step. After it, one ContextPlanner-only record with target &lt; | Tas k Completed | &gt; and no target clip is appended.</td></tr></table>

We then organize the clips, action captions, step order, and completion flag into the step-level records described in Appendix B.3. The example in Table 8 contains 21 aligned clip–caption pairs. Step 21 carries the completion flag, so the ContextPlanner’s target after it is <|Task Completed|>. More annotation examples are available on the project page.

This example shows three important properties of our annotations. First, each caption describes a single physical action. This includes fine-grained intermediate actions, such as folding and then unfolding the paper in Steps 1–4. Second, the same action can appear at different stages of a task. For example, “Unfolds the paper” appears at Steps 2, 4, and 7, while “Flips the paper over” appears at Steps 5, 8, 16, and 19. Similarly, “Presses the wing fold” appears at Steps 18 and 21. Therefore, the action text alone is not enough to determine the current task progress. The ContextPlanner also observes the generated clips to understand the current visual state (Sec. 4.3). Third, the <|Task Completed|> target after the final step provides supervision for learning when to stop.

For each step, the Executor learns to generate the corresponding video clip from the action caption and previous visual context. The ContextPlanner learns to predict

Table 7: WorldGuide Bench construction statistics. Counts are reported after each filtering stage. Only videos receiving a Gemini Video Planning Score of 5/5 are retained. The video-disjoint test split holds out four videos per task.
<table><tr><td>Quantity Count</td></tr><tr><td>Top-level categories 27 Task topics 245</td></tr><tr><td>Merged raw video metadata records 133,360 Validated (decodable) video records 132,911</td></tr><tr><td>Videos with a non-null Gemini annotation 119,798 Videos with planning score = 5 (retained) 58,679</td></tr><tr><td>Training split videos 57,699</td></tr><tr><td>Test split videos (held out) 980</td></tr><tr><td>videos per task 4 tasks / categories 245 / 27</td></tr></table>

the next action from up to K=3 previous clip-action pairs. For example, Steps 15–17 are used as context when predicting Step 18, and Steps 19–21 are used as context when predicting <|Task Completed|>.

## B.6 Human Audit of Annotations

Since WorldGuide Bench is automatically annotated, we conduct a human audit to evaluate the annotation quality. Three independent annotators review 245 videos, with one video sampled from each task. These videos contain a total of 3,168 annotated action steps. For each video, the annotators watch the original video together with its task label and ordered action captions. Each annotator evaluates the annotations independently.

Evaluation criteria. We evaluate the annotations using six criteria. Each criterion is answered based only on the visual evidence in the video.

• Caption correctness (per step): Does the caption correctly describe the visible action, the involved objects, and the result of the action?

• Action atomicity (per step): Does the caption describe one clear action rather than multiple actions?

• Task-label correctness (per video): Does the task label correctly describe the task performed in the video?

• Essential-action coverage (per video): Do the captions cover all actions needed to complete the task?

• Temporal order (per video): Are the captions ordered according to when the actions occur in the video?

• Demonstration completeness (per video): Does the final visible state show that the task has been completed?

Scoring. For each item, annotators choose Correct (1), Incorrect (0), or Cannot determine (U). The U option is used when the video does not provide enough visual evidence to make a decision, and these ratings are excluded from the score. The score is the percentage of rated items marked as Correct. For step-level criteria, scores are first averaged within each video and then across videos, so that every task contributes equally. We also report a majority-vote score. An item receives a final Correct or Incorrect label when at least two of the three annotators agree.

Table 8: Complete annotation of one WorldGuide Bench video. Task: origami, airplane\_dart. Work subject: folding a paper airplane (dart style). Each row is one clip; Time gives its start and end in the source video (mm:ss).
<table><tr><td>Step</td><td>Time</td><td>Action caption</td></tr><tr><td>1</td><td>00:04-00:05</td><td>Folds the bottom right corner of the paper to the top left.</td></tr><tr><td>2</td><td>00:05-00:06</td><td>Unfolds the paper.</td></tr><tr><td>3</td><td>00:06-00:07</td><td>Folds the bottom left corner of the paper to the top right.</td></tr><tr><td>4</td><td>00:07-00:08</td><td>Unfolds the paper.</td></tr><tr><td>5</td><td>00:08-00:09</td><td>Flips the paper over.</td></tr><tr><td>6</td><td>00:09-00:10</td><td>Folds the bottom edge of the paper upwards.</td></tr><tr><td>7</td><td>00:10-00:11</td><td>Unfolds the paper.</td></tr><tr><td>8</td><td>00:11-00:12</td><td>Flips the paper over.</td></tr><tr><td>9</td><td>00:12-00:13</td><td>Collapses the side edges inwards to form a triangle at the bottom.</td></tr><tr><td>10</td><td>00:13-00:14</td><td>Presses the folded triangle flat.</td></tr><tr><td>11</td><td>00:14-00:15</td><td>Folds the left corner of the triangle towards the center.</td></tr><tr><td>12</td><td>00:15-00:16</td><td>Folds the right corner of the triangle towards the center.</td></tr><tr><td>13</td><td>00:16-00:17</td><td>Folds the bottom tip of the triangle upwards.</td></tr><tr><td>14</td><td>00:17-00:18</td><td>Presses the fold firmly.</td></tr><tr><td>15</td><td>00:18-00:19</td><td>Folds the entire paper in half lengthwise.</td></tr><tr><td>16</td><td>00:19-00:20</td><td>Flips the paper over.</td></tr><tr><td>17</td><td>00:20-00:21</td><td>Folds down one wing, aligning it with the bottom edge.</td></tr><tr><td>18</td><td>00:21-00:22</td><td>Presses the wing fold.</td></tr><tr><td>19</td><td>00:22-00:23</td><td>Flips the paper over.</td></tr><tr><td>20</td><td>00:23-00:24</td><td>Folds down the second wing, aligning it with the bottom edge.</td></tr><tr><td>21</td><td>00:24-00:25</td><td>Presses the wing fold.</td></tr><tr><td>一</td><td>一</td><td>&lt; | Task Completed |&gt; (ContextPlanner target after Step 21; no clip)</td></tr></table>

Results. Table 9 summarizes the human audit results. The majority-vote scores range from 81.6% to 84.5% across all six criteria. Caption correctness reaches 83.0%, task-label correctness reaches 83.7%, and essential-action coverage reaches 84.5%. The scores from the three annotators differ by at most 1.3 percentage points for each criterion, showing similar judgments across annotators. Overall, the audit indicates that the automatic

Table 9: Human audit of WorldGuide Bench annotations (% judged correct) on 245 task-balanced videos with 3,168 action steps. Majority aggregates the three annotators by item-level vote.
<table><tr><td>Criterion</td><td>Level</td><td>Annot. 1</td><td>Annot. 2</td><td>Annot. 3</td><td>Majority</td></tr><tr><td>Caption correctness</td><td>Step</td><td>82.4</td><td>82.6</td><td>81.9</td><td>83.0</td></tr><tr><td>Action atomicity</td><td>Step</td><td>82.2</td><td>81.5</td><td>82.3</td><td>82.9</td></tr><tr><td>Task-label correctness</td><td>Video</td><td>82.9</td><td>82.4</td><td>83.2</td><td>83.7</td></tr><tr><td>Essential-action coverage</td><td>Video</td><td>83.2</td><td>84.4</td><td>83.9</td><td>84.5</td></tr><tr><td>Temporal order</td><td>Video</td><td>82.1</td><td>81.0</td><td>81.6</td><td>81.6</td></tr><tr><td>Demonstration completeness</td><td>Video</td><td>81.4</td><td>81.7</td><td>82.6</td><td>82.9</td></tr></table>

annotation pipeline provides reliable supervision for action captions, temporal order, and task completion.

## C Hierarchical Memory

The Executor uses the spatial history compression strategy from YUME [21, 48]. Recent latent frames are kept at higher spatial resolution, while older frames are progressively compressed to lower resolutions. This keeps the number of memory tokens bounded as the generated video becomes longer.

## C.1 History and Reference Conditioning

Terminology. The history contains $F \leq T + 1 \leq 1 3 6 6$ latent frames. It includes up to $T \le 1 3 6 5$ recent frames and the oldest frame, which we call the anchor. We also use a separate reference slot containing the latest generated frame, i.e., the last frame of the current visual state $x _ { t }$ in Eq. 1 (the initial image $x _ { 0 }$ at the first step). The reference may contain the same frame as the newest history entry, but it uses separate token positions.

We use L to denote the latent length of the clip being generated. All frame counts in this section refer to latent frames after VAE encoding. For N continuously encoded video frames, the VAE produces $\lfloor ( N - 1 ) / 4 \rfloor + 1$ latent frames.

Supplied history. To construct the history, we concatenate the latent frames from previous clips and keep the most recent frames that fit within the available history length. If the history is shorter than the required length, we left-pad it; these padded positions add tokens but contain no visual information. Since one conditioning slot is reserved for the reference frame, the history length satisfies

$$
F \leq \operatorname* { m i n } ( L - 1 , T + 1 ) .
$$

The history is updated after each generated clip. If there is no previous history, the memory branch is skipped.

Conditioning input. The input to the DiT is

$$
x _ { \mathrm { D i T } } = [ \tilde { z } ; z _ { \mathrm { c o n d } } ; m _ { \mathrm { c o n d } } ] \in \mathbb { R } ^ { ( 2 C + 1 ) \times L \times H \times W } .\tag{7}
$$

Here, z˜ is the noisy video latent, $z _ { \mathrm { c o n d } }$ contains the visual conditioning, and $m _ { \mathrm { c o n d } }$ indicates which conditioning positions are valid. Without memory, only the reference slot in $z _ { \mathrm { c o n d } }$ is filled. With memory, $z _ { \mathrm { c o n d } }$ contains the history followed by the reference frame. The same history latents are also spatially compressed into memory tokens, as described in Appendix C.2.

## C.2 Multi-Scale Embedding and Injection

Multi-scale embedders. We use different spatial resolutions to represent the visual history. With patch size $p = 2 .$ , we define embedders $E _ { r }$ for compression rates $r \in \{ 1 , 2 , 4 , 8 , 1 6 \}$ . Each $E _ { r }$ is a 3-D convolution with kernel size and stride $( 1 , p r , p r )$ , which maps C input channels to d-dimensional features. $E _ { 1 }$ uses the pretrained patchembedding weights. For larger compression rates, we initialize the convolution kernels by trilinearly interpolating the pretrained weights.

For the coarsest rate, $r = 6 4$ , we first apply a channel-preserving 3-D convolution with kernel size and stride (1, 4, 4), followed by $E _ { 1 6 }$ . This gives an effective spatial stride of 128. The temporal kernel size and stride are always one. Therefore, the compression only reduces the number of spatial tokens while preserving all temporal latent frames.

Positional encoding and memory injection. Each compressed history segment receives 3-D rotary positional embeddings based on its spatial grid. Temporal positions continue across consecutive history segments, preserving their temporal order. These positions correspond to latent-frame positions rather than timestamps in the original video.

The resulting memory tokens $\mathcal { M }$ are added to the Executor’s context sequence together with the visual and text conditioning:

$$
\tau = [ \mathcal { M } ; c _ { \mathrm { v i s i o n } } ; c _ { \mathrm { t e x t } } ] .\tag{8}
$$

This adds visual-history information as conditioning tokens without increasing the number of video tokens being denoised or changing the training objective in Eq. 2. When no history is available, the memory branch is skipped

and the Executor uses only the reference-image conditioning.

## C.3 Compression Schedule and Token Budget

The schedule is selected by the history length F. Table 10 gives the longest schedule $( 3 4 3 \leq F \leq 1 3 6 6 )$ , with ages running from the newest frame (1) to the anchor (F).

Shorter histories. For $7 \leq F \leq 3 4 2$ , the newest three frames use rate one, the next two use rate two, and successively older tiers use rates four, eight, and sixteen as needed. The oldest anchor uses rate one for $F \le 8 6$ and rate two thereafter. For $F \leq 6 .$ both endpoints use rate one and the intervening frames use rate two. For $F \leq 2 .$ , the empty intermediate slice reuses the newest latent, as in YUME. Using nominal area ratios before spatial padding, the bounds including anchor and reference are

$$
\frac { p ^ { 2 } } { H W } | \mathcal { M } | _ { \mathrm { n o m i n a l } } \leq \left\{ \begin{array} { l l } { 4 , } & { 1 \leq F \leq 6 , } \\ { 6 . 5 , } & { 7 \leq F \leq 2 2 , } \\ { 7 . 5 , } & { 2 3 \leq F \leq 8 6 , } \\ { 7 . 7 5 , } & { 8 7 \leq F \leq 3 4 2 , } \\ { 8 , } & { 3 4 3 \leq F \leq 1 3 6 6 . } \end{array} \right.\tag{9}
$$

For every history length, the history, anchor, and reference therefore cost at most eight full-resolution latent-frame equivalents before spatial padding. This bounds the number of context tokens, not the number of retained temporal frames.

Spatial padding. With padding, a rate-r embedder produces $\lceil H / ( p r ) \rceil \lceil W / ( p r ) \rceil$ tokens per latent frame, the coarsest embedder $\lceil H / 1 2 8 \rceil \lceil W / 1 2 8 \rceil$ and the reference $\lceil H / p \rceil \lceil W / p \rceil$ These integer counts can exceed the nominal bound when H and W are not divisible by the strides, but for a fixed history capacity and resolution the token count remains bounded independently of rollout length.

## D Training and Inference

The two modules are trained sequentially on the WorldGuide Bench training split (Appendix B.4): first the ContextPlanner, then the Executor with the ContextPlanner frozen (Appendix A).

Table 10: Nominal token cost for $3 4 3 \leq F \leq 1 3 6 6$ history latent frames. One full-resolution latent-frame equivalent is $H W / p ^ { 2 }$ tokens; the reference is counted separately.
<table><tr><td>Segment</td><td>Latent frames</td><td>Rate</td><td>Cost upper bound</td></tr><tr><td>Present</td><td>3</td><td>1</td><td>3.00</td></tr><tr><td>Near present</td><td>2</td><td>2</td><td>0.50</td></tr><tr><td>Recent past</td><td>16</td><td>4</td><td>1.00</td></tr><tr><td>Mid past</td><td>64</td><td>8</td><td>1.00</td></tr><tr><td>Far past</td><td>256</td><td>16</td><td>1.00</td></tr><tr><td>Distant past</td><td> $F - 3 4 2 \leq 1 0 2 4$ </td><td>64</td><td>0.25</td></tr><tr><td>Oldest anchor</td><td>1</td><td>2</td><td>0.25</td></tr><tr><td>History subtotal</td><td>F</td><td>一</td><td>7.00</td></tr><tr><td>Reference</td><td>1</td><td>1</td><td>1.00</td></tr><tr><td>Total</td><td> $F + 1$ </td><td>一</td><td>8.00</td></tr></table>

## D.1 ContextPlanner Training

We fine-tune all parameters of Qwen2.5-VL-7B-

Instruct [3] to predict the next action or the completion token from the task goal, the current visual state, and up to K=3 preceding clip–action pairs. The cross-entropy loss is applied only to response tokens (Table 11).

## D.2 Executor Training

We fine-tune HunyuanVideo-1.5 [41] for action-conditioned image-to-video generation with the masked flowmatching objective of Eq. 2, using precomputed VAE latents. Each action caption is embedded by the frozen ContextPlanner, and the multi-scale memory embedders are trained together with the Executor (Table 12). The memory ablation evaluates this final checkpoint with and without memory (Appendix E.1).

Table 11: ContextPlanner training configuration. Full-parameter fine-tuning of Qwen2.5-VL-7B-Instruct for next-step action prediction from goal, visual state, and a history of K=3 clip–action pairs.
<table><tr><td>Component</td><td>Configuration</td></tr><tr><td>Base model</td><td>Qwen2.5-VL-7B-Instruct</td></tr><tr><td>Training objective</td><td>Autoregressive next-step prediction</td></tr><tr><td>Trainable parameters</td><td>Full-parameter fine-tuning</td></tr><tr><td>Precision</td><td>BF16</td></tr><tr><td>Distributed training</td><td>DeepSpeed ZeRO-3</td></tr><tr><td>Hardware</td><td>8 nodes × 8 AMD Instinct MI210 (64 GB)</td></tr><tr><td>Training time</td><td>7 days</td></tr><tr><td>Per-GPU batch size</td><td>2</td></tr><tr><td>Gradient accumulation</td><td>8</td></tr><tr><td>Effective global batch size</td><td>1024</td></tr><tr><td>Optimizer</td><td>AdamW</td></tr><tr><td>Learning rate</td><td> $2 \times 1 0 ^ { - 5 }$ </td></tr><tr><td>Weight decay</td><td>0.01</td></tr><tr><td>LR schedule</td><td>Cosine</td></tr><tr><td>Warmup steps</td><td>100 2</td></tr><tr><td>Epochs</td><td>4096</td></tr><tr><td>Maximum sequence length</td><td></td></tr><tr><td>History length K</td><td>3 clip-action pairs</td></tr></table>

## D.3 Rollout and Termination

At the first step, the ContextPlanner receives the task goal g and the initial image x ; afterwards it receives the latest generated clip $x _ { t }$ and the history $h _ { t }$ of the latest $K { = } 3$ generated clip–action pairs (Eq. 1). When it emits <|Task Completed|>, the rollout stops immediately and no further clip is generated. Otherwise, the Executor renders the predicted action $a _ { t }$ as the next clip $x _ { t + 1 }$ , and its memory is refreshed with this clip (Appendix C.1). Rollouts are capped at 80 steps, above the longest reference procedure in WorldGuide Bench (62 steps). The planner is never given the reference step count, and a rollout that reaches the cap is scored as generated rather than marked successful. The generated clips are concatenated into the final video. Because each clip is generated separately and the memory token count is bounded (Appendix C.3), the per-step cost does not grow with rollout length.

Sampling uses 10 denoising steps, guidance scale 7.5, and flow shift 5.0. Training clips are sampled at 24 FPS, and generated videos are played back at 16 FPS.

## E Evaluation Protocol and Metrics

## E.1 Benchmarks, Baselines, and Ablations

Baselines. We compare WorldGuide with recent video generation models [41, 38, 47, 40, 20, 51, 45, 18], video world models [2, 32, 21, 8, 42, 36, 35, 13], and three planner–executor methods: Bernini [34], PhysAgent [19], and TempAct [39]. These planner–executor methods use their own planning–execution loops and are marked with ∗ in the result tables.

WorldGuide Bench. We evaluate one rollout for each of the 980 held-out videos (Table 2). Unstarred baselines are given the complete ground-truth action plan. They generate the procedure step by step, following the reference actions in order. For each step, the model receives one action caption and continues generation from the last generated frame. Each step is generated for approximately 2 seconds (48 frames at 24 FPS or 32 frames at 16 FPS, depending on the lengths supported by each model). Therefore, these baselines know the full action sequence before

Table 12: Executor training configuration. HunyuanVideo-1.5 DiT fine-tuned for language-conditioned image-to-video generation on step-level clips under the masked flow-matching objective (Eq. 2).
<table><tr><td>Component</td><td>Configuration</td></tr><tr><td>Base model</td><td>HunyuanVideo-1.5 DiT</td></tr><tr><td>Training objective</td><td>Masked MSE flow matching</td></tr><tr><td>Input mode</td><td>Image-to-video</td></tr><tr><td>Target resolution</td><td>480 × 832</td></tr><tr><td>Target FPS</td><td>24</td></tr><tr><td>Text conditioning</td><td>Action-caption embeddings from the frozen ContextPlanner, and ByT5 [43] features</td></tr><tr><td>Visual conditioning</td><td>Image-conditioned latent branch and vision states</td></tr><tr><td>Hardware</td><td>8 nodes × 8 AMD Instinct MI210 (64 GB)</td></tr><tr><td>Training time</td><td>42 days</td></tr><tr><td>Learning rate</td><td>1 × 10−5</td></tr><tr><td>Weight decay</td><td>1 × 10−4</td></tr><tr><td>Gradient clipping</td><td>1.0</td></tr><tr><td>Precision</td><td>Mixed BF16; DiT backbone in FP32</td></tr><tr><td>Diffusion scheduler</td><td>FlowMatchEulerDiscreteScheduler</td></tr><tr><td>Timestep sampling</td><td>Logit-normal</td></tr><tr><td>Logit-normal parameters</td><td>mean 0.0, std. 1.0</td></tr><tr><td>CFG dropout rate</td><td>0.1 FSDP</td></tr><tr><td>Distributed training</td><td></td></tr><tr><td>Sequence parallelism Per-GPU batch size</td><td>Disabled (size 1)</td></tr><tr><td>Gradient accumulation</td><td>1 2</td></tr><tr><td></td><td></td></tr><tr><td>Effective global batch size</td><td>128 video segments</td></tr></table>

generation begins.

In contrast, the starred planner–executor baselines and WorldGuide receive only the initial image and task goal. They must determine how to complete the task using their own planning process. No model receives future ground-truth video frames.

Video-CraftBench. We evaluate all 294 samples from Video-CraftBench\_v0.1 [28], which contains block assembly and paper folding tasks.<sup>2</sup> In this benchmark, all models receive only the initial image and task goal, without a reference action plan (Table 3). Since no reference plan is provided, the judge evaluates whether the generated video completes the sub-goals needed to achieve the given task. Different valid procedures and step granularities are accepted.

Ablation variants. Table 4 evaluates four variants using the same evaluation protocol.

• Open-loop: The model predicts the full action sequence once from the initial image and task goal, then executes all actions without replanning. The action sequence and video length are therefore fixed before generation starts.

• W/o visual: The model still predicts actions step by step, but generated video clips are removed from the ContextPlanner input. The Executor remains unchanged.

• W/o Mem: We use the same final Executor checkpoint but disable its memory branch. Each new clip is conditioned only on the reference image.

Table 13: Metric meanings on WorldGuide Bench. Video scores assess rendered execution; text scores assess the predicted procedure. Video-CraftBench uses goal-based video judging without a reference plan.
<table><tr><td></td><td>Video judge</td><td>Text judge</td></tr><tr><td>Input</td><td>Rollout, goal, reference plan</td><td>Predicted plan, goal, reference plan</td></tr><tr><td>Plan Accuracy</td><td>Visibly achieved sub-goals</td><td>Semantic correctness and coverage</td></tr><tr><td>Order Score</td><td>Executed prerequisite order</td><td>Planned prerequisite order</td></tr><tr><td>Repeat/Skip</td><td>Unnecessary executions; omissions reduce Plan Redundant actions and missing sub-goals Accuracy</td><td></td></tr><tr><td>Success</td><td>Final visible task completion</td><td>Plan would succeed if executed correctly</td></tr></table>

• Oracle ContextPlanner: We replace the predicted actions with the ground-truth actions and provide them to the same Executor and memory in their reference order. Generation stops after the final reference action. The model does not receive any future ground-truth video frames.

Judges. We use Gemini 3.5 Flash [11] to evaluate the generated videos in Tables 2, 3, and 4. We use GPT-5.2 [24] to evaluate the predicted text plans in Table 5. Although the two judges share some metric names, they evaluate different aspects of the output, as summarized in Table 13. Their scores should therefore not be compared directly. Judge responses that cannot be parsed are excluded, and each score is averaged over the remaining valid judgments. We further validate the video evaluation with the independent human study in Appendix F, where WorldGuide is ranked first, followed by MiniMax-H3.

## E.2 Video-Judge Metrics

The video judge (gemini-3.5-flash, temperature 0) receives the complete generated video, the task description, and, on WorldGuide Bench, the ordered reference plan with the completion token removed; it never sees the ground truth video (Fig. 7). From its structured JSON output, we compute four metrics for video i with $N _ { i }$ reference steps. Equivalent actions, different step granularities, and valid reorderings of independent steps receive credit.

Plan Accuracy (↑). With $c _ { i j } = 1$ if the sub-goal of step $j$ is visibly achieved and 0 otherwise,

$$
\mathrm { P l a n A c c } _ { i } = \frac { 1 0 0 } { N _ { i } } \sum _ { j = 1 } ^ { N _ { i } } c _ { i j } .\tag{10}
$$

Missing, partial, or ambiguous effects receive no credit.

Order Score (↑). Of the $N _ { i } ^ { \mathrm { e x e c u t e d } }$ observed steps, $N _ { i } ^ { \mathrm { o r d e r e d } }$ respect their prerequisites:

$$
\mathrm { O r d e r } _ { i } = 1 0 0 \frac { N _ { i } ^ { \mathrm { o r d e r e d } } } { N _ { i } ^ { \mathrm { e x e c u t e d } } } ,\tag{11}
$$

and ${ \mathrm { O r d e r } } _ { i } = 0$ if no step is executed. Because a short but correctly ordered execution can score highly, Order is read together with Plan Accuracy.

Repeat/Skip (↓). With $R _ { i }$ repeated executions that add no task progress,

$$
{ \mathrm { R e p e a t } } _ { i } = 1 0 0 { \frac { R _ { i } } { N _ { i } } } .\tag{12}
$$

For the video judge, omitted steps are penalized by Plan Accuracy rather than by this term.

![](images/02d9f83876ed7dd00e65b645cabc865a3c86ff3e86d2188cc0985841704d94da.jpg)  
Figure 7: Prompt structure for procedural video evaluation with the Gemini 3.5 Flash judge (temperature 0.0), used to produce the video-judge columns of Table 2. The judge receives the generated rollout, task description, and ground-truth plan with the <|Task Completed|> token stripped, and scores Plan Accuracy, Order Score, Repeat/Skip, and Task Success from visible execution alone; the ground-truth video is never shown.

Task Success Rate (↑). Success<sub>i</sub> = 1 only if the final visible state achieves the complete task. Over $N _ { \mathrm { e v a l } }$ valid judgments,

$$
\mathrm { T S R } = \frac { 1 0 0 } { N _ { \mathrm { e v a l } } } \sum _ { i = 1 } ^ { N _ { \mathrm { e v a l } } } \mathrm { S u c c e s s } _ { i } .\tag{13}
$$

Plan Accuracy, Order, and Repeat/Skip are averaged over videos in the same way.

Skipped judge responses. For a few videos, the Gemini judge returns a response that cannot be parsed into the required JSON format. We skip these videos, so Task Success is computed over the remaining $N _ { \mathrm { e v a l } }$ videos rather than all 980. For example, on WorldGuide Bench: WorldGuide 326/978 = 33.33% (2 skipped due to format issue in Gemini judge limitation), MiniMax-H3 293/980 = 29.90% (0 skipped), HunyuanVideo-1.5 117/975 = 12.00% (5 skipped), Cosmos-Predict2.5 117/980 = 11.94% (0 skipped), Bernini 19/971 = 1.96% (9 skipped), and HY-WorldPlay 12/956 = 1.26% (24 skipped, the maximum across models).

## E.3 Planner Evaluation

Table 5 evaluates the action sequences predicted by the planner as text. The upper block evaluates Qwen2.5-VL and the ContextPlanner with different history lengths $( K \in \{ 1 , 2 , 3 \} )$ . At each step, the planner predicts the next action from the task goal and the ground-truth clip–action history. The lower block evaluates the full WorldGuide loop, where the model runs autonomously using its own generated clips and predicted actions. The w/o visual variant follows the same loop but removes generated clips from the planner input.

We use GPT-5.2 [24] to evaluate the predicted action sequences (Fig. 8). The judge receives the task description, predicted action sequence, and reference plan, but does not see the video. The reference plan defines the required sub-goals and their dependencies, but it is not treated as the only correct solution. The judge accepts equivalent actions, different step granularities, and valid changes in action order. We report four metrics. Plan Accuracy measures whether the predicted actions are semantically correct and cover the required sub-goals. Order Score measures whether prerequisite actions appear in the correct order. Repeat/Skip measures redundant actions and missing sub-goals. Plan Success measures whether the full predicted plan would achieve the task goal if all actions were executed correctly. Plan Success evaluates the action sequence itself, not whether the generated video stops at the correct visual state. Visual stopping is evaluated separately by the video judge. Stopping before the task is complete leads to failure in Task Success, while continuing after completion typically introduces unnecessary repeated actions.

![](images/a58d71b5305a603691e1a95f4a1a35eca3eb09f585943b0c6e738900f8f4c10c.jpg)  
Figure 8: Prompt structure for text-only planner evaluation with a GPT-5.2 judge, used to produce Table 5. The judge receives the task description, predicted plan, and ground-truth plan, with no video, and scores the procedural validity of the predicted text. The ground truth serves as a reference rather than a unique target, so semantically equivalent actions, orderings, and granularities are accepted.

## E.4 Paired Alignment

Alignment compares each generated video with its paired ground-truth video using deterministic appearance and motion descriptors, independently of the LLM judges. Both videos are sampled at B = 16 uniformly spaced relative positions.

Appearance similarity (AS<sub>i</sub>) is the mean cosine similarity between $\ell _ { 2 } .$ -normalized 16-bin RGB histograms at corresponding positions.

Table 14: The five VBench dimensions reported in Tables 2 and 3, with the predictor used by the official implementation. All are perceptual or temporal quality measures; none assesses procedural correctness.
<table><tr><td>Dimension</td><td>Predictor</td><td>Measures</td></tr><tr><td>Aesthetic Quality (AQ)</td><td>LAION CLIP aesthetic</td><td>Mean frame-level aesthetic score</td></tr><tr><td>Imaging Quality (IQ)</td><td>MUSIQ</td><td>Technical image quality; penalizes blur, noise, and distortion</td></tr><tr><td>Dynamic Degree (DD)</td><td>RAFT optical flow</td><td>Percentage of videos classified as containing suf- ficient motion</td></tr><tr><td>Motion Smoothness (MS)</td><td>Frame interpolation</td><td>Agreement between observed and interpolated intermediate frames</td></tr><tr><td>Overall Consistency (OC)</td><td>ViCLIP</td><td>Video-text semantic consistency with the task de- scription</td></tr></table>

State similarity (SS<sub>i</sub>) compares the start, middle, and end frames. Each descriptor concatenates 8-bin perchannel RGB histograms with a 16-bin gradient-magnitude histogram of a 32×32 grayscale frame, and the cosine similarities are weighted toward the final state:

$$
\mathrm { S S } _ { i } = 0 . 2 0 s _ { i } ^ { \mathrm { s t a r t } } + 0 . 3 0 s _ { i } ^ { \mathrm { m i d d l e } } + 0 . 5 0 s _ { i } ^ { \mathrm { f i n a l } } .\tag{14}
$$

Temporal alignment (TA<sub>i</sub>) compares motion-intensity curves, the normalized mean absolute grayscale difference between adjacent samples. If both curves vary, $\mathrm { T A } _ { i } = ( \rho _ { i } + 1 ) / 2$ with Pearson correlation $\rho _ { i } ;$ if both are constant, $\mathrm { T A } _ { i } = 1$ ; if only one is constant, $\mathrm { T A } _ { i } = \operatorname* { m a x } ( 0 , 1 - 2 0 | \mu _ { i } ^ { \mathrm { g e n } } - \mu _ { i } ^ { \mathrm { g t } } | )$ with mean intensities $\mu _ { i }$ . The result is clipped to [0, 1].

The final score is

$$
\mathrm { A l i g n m e n t } _ { i } = 1 0 0 \left( 0 . 4 0 \mathrm { A S } _ { i } + 0 . 3 0 \mathrm { S S } _ { i } + 0 . 3 0 \mathrm { T A } _ { i } \right) \in [ 0 , 1 0 0 ] .\tag{15}
$$

Because it compares low-level appearance and motion, Alignment complements rather than replaces the task metrics.

## E.5 VBench Video-Quality Metrics

We use the official VBench implementation [50] for the five dimensions in Table 14, reported on a 0–100 scale (higher is better). They measure perceptual and temporal quality, not procedural correctness.

## F Human Evaluation of Generated Videos

Setup. We conduct a human evaluation with four evaluators, nine models, and 18 task cases, resulting in 162 evaluated videos and 648 individual ratings. For each video, the evaluator first reads the task description and then watches the complete generated video. The evaluator assigns a task-completion score from 1 to 5.

Evaluators are asked to consider three aspects when assigning the score: (1) whether the necessary actions are visibly performed in a sensible order, (2) whether the essential steps are completed, and (3) whether the final task goal is visibly achieved. A higher score indicates more successful task completion.

Results. For each model, we average its 72 human ratings and report the results in Table 15. WorldGuide achieves the highest average score of 2.750, followed by MiniMax-H3 with 2.542 and LTX-2.5 with 2.472. This human evaluation provides an independent validation of the task-completion results obtained with the automatic video judge.

## G Additional Results

## G.1 WorldGuide Bench Qualitative Comparisons

Figs. 9–11 compare WorldGuide (Ours) with 12 baselines, including Bernini, PhysAgent, and TempAct. The latter three use their own planning–execution loops; the other baselines receive reference action sequences. WorldGuide predicts actions from the goal and generated history. Frames in each row follow temporal order and illustrate action progression and visual continuity; complete rollouts are used for task scoring. Complete generated videos for all compared methods are available on the project page.

The comparisons show distinct limitations. In block construction, TempAct changes the supplied pieces and Bernini introduces floating blocks (Fig. 9); in burrito assembly, TempAct folds a tortilla without visible fillings and PhysAgent shifts to stacking round objects (Fig. 10). WorldGuide shows successive assembly and preparation steps (Figs. 9–11); its late visual degradation in folding is shown in Appendix G.2.

Table 15: Human evaluation of procedural video generation. Average task-completion scores on a 1–5 scale. Higher scores indicate better task execution. Model labels follow Table 2.
<table><tr><td rowspan=1 colspan=1>Model               Avg. score / 5</td></tr><tr><td rowspan=1 colspan=1>Bernini*                          1.583Cosmos-Predict2.5              1.000</td></tr><tr><td rowspan=1 colspan=1>HunyuanVideo-1.5              1.000</td></tr><tr><td rowspan=1 colspan=1>LTX-2.5                         2.472</td></tr><tr><td rowspan=1 colspan=1>MiniMax-H3                    2.542</td></tr><tr><td rowspan=1 colspan=1>PhysAgent*                     1.000</td></tr><tr><td rowspan=1 colspan=1>SkyReels                         1.917</td></tr><tr><td rowspan=1 colspan=1>TempAct*                       1.444</td></tr><tr><td rowspan=1 colspan=1>WorldGuide                   2.750</td></tr></table>

## G.2 Video-CraftBench Results

Table 3 reports the goal-conditioned results under the protocol of Appendix E.1, and Table 4 the corresponding memory ablation.

Figs. 12–14 compare the same 13 models on assembly and folding, with all models receiving an initial image and the task goal, without reference action captions. In the block-horse example, WorldGuide adds a yellow bar and orange cylinder to the starting arch, while MiniMax-H3 introduces shaped toy pieces and LTX-2.5 substitutes different blocks (Fig. 12).

Folding remains challenging. TempAct changes the blue sheet into a white model aircraft in the airplane example (Fig. 13) and substitutes a white boat-shaped object in the boat example (Fig. 14). WorldGuide shows successive folds, but its late paper appearance and scene texture deteriorate in Figs. 13 and 14.

## H Limitations and Failure Analysis

Errors can compound across steps. The ContextPlanner may repeat an action, skip a prerequisite, or stop too early; the Executor may distort geometry or appearance even for a correct instruction. Because each generated clip becomes the next planning observation, replanning can respond to such errors but cannot undo an incorrect transition. The folding examples in Figs. 13 and 14 show correct action progression alongside late visual degradation.

The two modules are trained sequentially, without a joint task-completion objective (Appendix A). The bounded memory compresses older frames, which can discard details needed later, and enabling memory also increases repetition (Table 4). Outcome verification, explicit recovery from failed steps, and joint training are natural next steps.

![](images/4f8124836dce3bd9d95ccd19c629f210bfb3e1bbe33a7c68223232ec6bfdbb7e.jpg)  
Task : Building a tower using colorful wooden blocks  
Figure 9: Building a tower from colorful wooden blocks. WorldGuide progressively adds upright pieces and a top to the arch base. TempAct substitutes different blocks, Bernini shows floating pieces and scene changes, and PhysAgent retains several small stacks.

![](images/0019460f632cda00b9a58aaabbe1fe9c29e31b71d6e6d9b4d02bba6bc896b9e1.jpg)  
Task : Burrito wrap folding technique tutorial

Figure 10: Burrito assembly and folding. WorldGuide adds fillings before folding the tortilla. TempAct folds a tortilla without visible fillings, PhysAgent shifts to stacking round objects, and Bernini replaces the split-screen demonstration with a different scene.

![](images/29ced0237dde6c92c33d5f5cb4ce6b74a0538695fd35074a4e3974f4a15e26f7.jpg)  
Task : dessert cup in order layer step by step demonstration

Figure 11: Layered dessert preparation. WorldGuide shows mixing, cream addition, and toppings. HunyuanVideo-1.5 and Cosmos-Predict2.5 also show preparation stages, while Helios develops severe distortion and SpMem changes the vessel and contents.

![](images/96883d7fe8fe038227081f1c38a9f2c6d9fd95099690e7b1b17dd6a67efc672f.jpg)  
Task : How to arrange the blocks to build horse tutorial step by step  
Figure 12: Video-CraftBench: a second block-horse example. WorldGuide adds a yellow bar and orange cylinder to the starting arch. MiniMax-H3 introduces shaped toy pieces, while LTX-2.5 substitutes different blocks.

![](images/995c80386fec76166dff89ddd6147d7ec0296464eaff32374961142fed7e38f2.jpg)  
Task : How to make a paper airplane tutorial step by step  
Figure 13: Video-CraftBench: folding a blue paper airplane. WorldGuide progresses through several folds, with visible late texture distortion. PhysAgent repeatedly handles a broad fold, while TempAct changes the blue sheet into a white model aircraft.

![](images/6287d2537f25e7c9bdfd81da38516d306347ba66b1fe1fe093d124ffbbf05ece.jpg)  
Task : How to make a paper boat tutorial step by step

Figure 14: Video-CraftBench: folding an orange paper boat. WorldGuide shows successive folds with late changes in paper color and scene texture. TempAct substitutes a white boat-shaped object, while PhysAgent remains near a rectangular fold.
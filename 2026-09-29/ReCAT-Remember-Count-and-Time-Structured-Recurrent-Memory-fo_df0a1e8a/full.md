# ReCAT: Remember, Count, and Time: Structured Recurrent Memory for Robot Manipulation

Pankhuri Vanjani<sup>1∗</sup>, Mostafa Hatab<sup>1∗</sup>, Can Mizrakli<sup>1∗</sup>, Vaisakh Shaj<sup>2</sup>, Zhuoyue Li<sup>1</sup>, Moritz Reuss<sup>3</sup>, Rudolf Lioutikov<sup>1,4</sup>

<sup>1</sup>Intuitive Robots Lab, Karlsruhe Institute of Technology (KIT), Germany <sup>2</sup>University of Edinburgh, UK <sup>3</sup>NVIDIA <sup>4</sup>Robotics Institute Germany (RIG)

Abstract— Memory-dependent manipulation requires robots to make decisions using information that is no longer available to their current sensors, such as recalling an earlier visual cue, tracking task progress, counting repeated events, or estimating elapsed time. We present ReCAT, a language-conditioned policy with structured recurrent memory. An instruction-conditioned encoder forms features from the current observation. A recurrent memory integrates the observation stream through Mamba-2 layers and one causal attention layer. A flow-matching Transformer decoder reads the current and the historical representation through separate cross-attention in every block. ReCAT reaches 95.3% average success on LIBERO and 62.4% on RMBench, with the best or tied-best result on six of nine tasks. On three real-robot tasks probing spatial recall, event counting, and interval timing, the best ReCAT variant reaches 66.7% average success, against 8.3% for the strongest short-history baseline. Controlled comparisons within ReCAT show that the observation encoder and every-block memory conditioning are needed for this performance. They also show that update rules developed for efficient sequence modeling behave differently as robot memory: additive updates have the highest observed success on counting and timing, and delta-rule updates on spatial recall. Project website is at https://intuitiverobots.github.io/ReCAT

## I. INTRODUCTION

Robot manipulation often requires information that is no longer available in the current observation. An object’s original location may have left the camera view, the number of completed repetitions may not be apparent, or the next action may depend on how much time has elapsed. In the PLANT task (Fig. 1), for example, the shovel-over-pot view looks alike after one scoop and after two, yet the required actions differ: scoop again, or put down the shovel and proceed to planting. Such tasks are non-Markovian with respect to the current observation, and the policy must combine current perceptual information with relevant interaction.

Policies obtain this history in different ways. Some use a window of recent observations. Some store selected observations in a memory bank [1], [2]. Others summarize the whole observation stream in a compact recurrent state [3]– [6]. Recent and concurrent work shows that such full-history recurrence helps in manipulation [7]–[12]. Yet, precise manipulation also requires the current spatial and proprioceptive observation. So, a policy needs both a history representation and direct access to the present.

![](images/93635d04b3de23ceded7eea0299ae7fc82b6fb7cc0e7e6b383a37c55e7bcd768.jpg)

Task: "Add two scoops of soil in the pot then plant the flower." Earlier in the episode Correct next action  
![](images/a5c3716aabd08e2bdf14598b4dc15482f4dff07f6afad0520722367a05a3cd78.jpg)  
Fig. 1: Visually similar observations can require different actions depending on task history. ReCAT retains the completed scoop count in a compact recurrent memory, allowing the robot to scoop again after one scoop and put down the shovel after two.

We introduce ReCAT, a structured memory policy for language-conditioned manipulation. An instructionconditioned encoder turns each observation into multimodal features. These features take two paths. One path feeds the action decoder directly. The other path updates a temporal memory. This memory is a stack of recurrent layers with one causal attention layer in the middle. A flow-matching Transformer decoder reads the current and the historical representation through separate cross-attention in every block. ReCAT therefore keeps a compact history representation and still sees the current observation directly when it generates actions.

We evaluate ReCAT on LIBERO, RMBench, and three real-robot tasks. The real-robot tasks probe spatial recall, event counting, and interval timing. ReCAT reaches 95.3% average success on LIBERO and 62.4% on RMBench. On the real-robot tasks it reaches 66.7%, compared with 8.3% for the strongest short-history baseline. Stage-wise evaluation shows where a policy fails: during manipulation, or when it must use history. Controlled comparisons within ReCAT then show how observation encoding, recurrent update rule, memory capacity, and decoder conditioning affect performance. Mamba variants reach the highest success on counting and timing tasks.

## Our contributions are as follows:

• We present ReCAT, a structured policy whose temporal memory combines recurrent layers with a single causal attention layer. It conditions flow-matching action generation on separate current-observation and history pathways.

• ReCAT reaches 95.3% on LIBERO, 62.4% on RMBench with the best or tied-best result on six of nine tasks, and 66.7% across three real-robot memory tasks. It has the lowest measured inference latency among the evaluated memory-based policies.

• Controlled architectural and stage-wise comparisons show how observation encoding, recurrent update rule, memory capacity, and decoder conditioning affect immediate manipulation and later history-dependent decisions.

## II. RELATED WORK

## A. Memory in Visuomotor Imitation Learning

Robot policies differ in how they represent the past. Many keep past observations and attend to them. Fixed-window methods condition on recent visual tokens [13], [14]. Memorybank methods store observations over longer horizons and retrieve them when needed [1], [15], [16].These methods give direct access to earlier frames, as a result their attention cost grows with the number of retained tokens unless they select or compress. The second way carries a compact learned state through time. ReMem-VLA and µVLA propagate recurrent queries or latent memory tokens [2], [12]. Gated Memory Policy adds a learned retrieval gate [17].

These policies show that a compact state can carry task history. Each of them, however, pairs one memory mechanism with one way of feeding it to the policy. This leaves open a question: how does the recurrent update itself shape what is remembered, and how should that memory be combined with the current observation for action generation? ReCAT is built around both parts of it. Its memory is a stack of recurrent layers with one causal attention layer, and the update rule inside that stack is interchangeable. It’s flow-matching decoder attends to the current observation and to the memory through separate cross-attention in every block. Section IV evaluates this design and compares update rules, capacities within it.

## B. Recurrent Memory Architectures for Robot Manipulation

State-space models (SSMs) and related recurrent architectures have been used both as action-generation backbones and as history encoders in robot learning. MaIL uses a

Mamba encoder-decoder for behavior cloning, Mamba Policy combines Mamba and attention within a 3D diffusion policy, and DiSPo applies an SSM across discretized action scales [18]–[20]. AnoleVLA uses causal Mamba recurrence to fuse proprioceptive, visual, and language features for action chunk prediction [21]. For temporal conditioning, the Mamba motion encoder extracts temporal features, MTIL processes full trajectories with Mamba-2, RoboSSM processes long in-context demonstrations, and Embodied-SlotSSM maintains object-centric memory [7]–[9]. Concurrent work conditions on the full observation history as well. DSSP pairs a Mamba history representation with a dynamics-aware objective and hierarchical prefix conditioning for an SSM diffusion denoiser [10]. Chronos combines a selective historical state with a physics-informed action prior [11]. RoboTTT adapts fast weights at test time with Gated DeltaNet recurrence [22]. These policies each fix one recurrent backbone and one route from memory to action. ReCAT differs on both counts. Its temporal memory mixes recurrent layers with causal attention, and its action decoder receives the memory and the current observation through separate cross-attention rather than as a prefix or a prior.

These approaches draw on several families of linear recurrence. All keep a fixed-size state at constant per-step cost [23]–[25]. Additive memories, such as linear attention and Mamba-2, accumulate key-value writes under a learned decay [23], [25], [26]. Delta-rule models erase and rewrite information along matching key directions [27], [28]. A third family adds richer state dynamics through negative eigenvalues, reflections, or complex-valued rotations [5], [29]. All of them can be written as $S _ { t } ~ = ~ A _ { t } S _ { t - 1 } + b _ { t }$ . The structure of $A _ { t }$ sets the memory semantics: accumulation, replacement, or rotation. Pure recurrence has a known limit, though. A fixed-size state struggles with exact recall of earlier tokens [30]. Efficient long-context language models therefore mix a few attention layers into the recurrent stack, and this hybrid design is now common [25], [31]. Mamba Policy brings the same idea to a robot policy backbone [19]. ReCAT uses this hybrid design in two places. Its observation encoder mixes SSM and attention layers over the tokens of the current frame. Its temporal memory mixes them across time, with one causal attention layer among the recurrent layers, and the recurrent update rule within that stack is interchangeable. The two parts are meant to complement each other: the recurrent layers track task progress at constant cost, while the attention layer can look back at a specific earlier observation.

## III. METHODOLOGY

## A. Problem Formulation

We consider language-conditioned imitation learning for partially observed robot manipulation. The training set consists of demonstrations $\mathcal { D } = ( \dot { \ell } ^ { ( i ) } , ( s _ { t } ^ { ( i ) } , a _ { t } ^ { ( i ) } ) t = 1 ^ { \check { T _ { i } } } ) i = 1 ^ { N }$ 2 where $\ell ^ { ( i ) }$ is the task instruction, $s _ { t } ^ { ( i ) } = \bar { ( v _ { t } ^ { ( i ) } , p _ { t } ^ { ( i ) } ) }$ combines the camera observations and proprioception, and $a _ { t } ^ { ( i ) } \in \mathbb { R } ^ { d _ { a } }$ is the demonstrated action. We denote the available observation history by $\mathcal { H } _ { t } = ( s _ { 1 : t } , \ell )$ and an action chunk of length K by $A _ { t } = ( a _ { t } , \dotsc , a _ { t + K - 1 } )$ .

![](images/f48879f05892491fd9f095e56123e1507b7e49881ea9eda3b8d34aefb1a15821.jpg)  
Fig. 2: Overview of ReCAT. Frozen DynaFLIP and T5 encoders (LoRA-adapted) feed a language-conditioned visual compressor and a six-layer Mamba–attention frame encoder, which produces one feature vector $z _ { t }$ per step. The same $z _ { t }$ feeds the action decoder directly and is the input to the recurrent memory, which returns the readout $m _ { t }$ . The flow-matching decoder attends to $z _ { t }$ and $m _ { t }$ through separate cross-attention in every block. The decoder backbone is shared across all experiments. The ablations vary the frame-encoder mixer, the recurrent update rule, memory depth and width, and how $m _ { t }$ enters the decoder.

In memory-dependent tasks, the current observation $s _ { t }$ may be insufficient to determine the task state, so visually similar observations can require different actions depending on prior events. The policy is therefore non-Markovian with respect to $s _ { t }$ alone and is modeled as $\pi _ { \theta } ( A _ { t } \mid \mathcal { H } _ { t } )$ . Rather than storing the growing history $\mathcal { H } _ { t }$ , ReCAT updates a compact recurrent representation as new observations arrive.

## B. ReCAT

ReCAT has three parts (Fig. 2). An instruction-conditioned observation encoder turns the current images, instruction, and proprioception into one feature vector $z _ { t } .$ . A temporal memory takes $z _ { t }$ as its input at every step, updates its retained state $h _ { t } ,$ and returns a history-conditioned readout $m _ { t }$ . A flowmatching action decoder predicts the next action chunk from $z _ { t }$ and $m _ { t } .$ The same $z _ { t }$ therefore feeds the decoder directly and drives the memory.

a) Current-Observation Encoding: DynaFLIP [32] encodes each camera image into a $1 6 \times 1 6$ patch grid $P _ { t }$ Its T5 text encoder produces instruction tokens L. Both backbones stay frozen and are adapted with LoRA. Following Compressor-VLA [33], we compress each patch grid with language-conditioned global queries and local window queries. The global queries keep task-relevant context. The local queries keep fine spatial detail.

$$
Z _ { t } = { \mathrm { C r o s s } } { \mathrm { A t t n } } \big ( \gamma ( { \bar { \ell } } ) \odot Q + \beta ( { \bar { \ell } } ) , P _ { t } \big ) ,\tag{1}
$$

Here $\bar { \ell }$ summarizes the instruction, $Q$ are the learned global queries, and $\gamma , \beta$ are FiLM parameters. The local window queries work the same way over their windows. Global and local outputs are concatenated into 80 tokens per camera. A six-layer encoder then mixes these tokens with the instruction tokens and a proprioception token within the frame. Five of its layers are Mamba-2 and the fourth is bidirectional attention. This encoder carries no state across timesteps. Its last position is read out as the frame feature $z _ { t } \in \mathbb { R } ^ { 7 6 8 }$

b) Recurrent Memory: The recurrent memory summarizes the whole observation history. It is a stack of six blocks of width 768. Five blocks are Mamba-2 layers and have no feed-forward network (FFN). The fourth block is a causal attention layer and has one FFN. At every step the memory takes $z _ { t }$ as input, updates its retained state, and returns a readout,

$$
( q _ { t } , m _ { t } ) = R _ { \phi } ( z _ { t } , q _ { t - 1 } ) ,\tag{2}
$$

where $q _ { t } = ( h _ { t } , C _ { t } )$ . Here $h _ { t }$ denotes the persistent recurrent states of the five Mamba-2 layers, and $C _ { t }$ is the key-value cache of the attention layer. The size of $h _ { t }$ is fixed by the architecture. The cache $C _ { t }$ holds one entry per step and therefore grows linearly with episode length. Each entry is one 768-dimensional vector, so the memory never stores images or patch tokens. All of $q _ { t }$ is reset at the start of each episode. The readout $m _ { t }$ is the normalized output of the last block, not the state itself.

Two properties of the memory are set by configuration. The first is capacity: the number of blocks and the state width $d _ { \mathrm { s t a t e } }$ of each Mamba-2 layer. The second is the update rule of the recurrent layers. Mamba-2 is the default. For comparison we replace it with Mamba-3 or Gated DeltaNet-2 and keep the observation encoder, the attention layer, the action decoder, and the training recipe unchanged. We match parameter counts across update rules. Table I summarizes what each rule does to information already in the state. During training, the recurrence runs over the whole trajectory as a parallel scan. At inference, it runs one step at a time.

c) Observation and memory conditioned Action Gener ation: Following flow-matching based policies [34]–[36], a Transformer decoder predicts an action chunk $A _ { t } \in \mathbb { R } ^ { K \times d _ { a } }$ with $K = 1 6$ . For Gaussian noise $X _ { 0 }$ and $\tau \sim \mathcal { U } [ 0 , 1 ]$ , we set $X _ { \tau } = ( 1 - \tau ) X _ { 0 } + \tau A _ { t }$ and minimize

TABLE I: Update rules of the recurrent layers compared in Sec. IV. Each step writes a cue-content pair $( k _ { t } , v _ { t } )$ derived from $z _ { t }$ into the state $h _ { t }$ . The rules differ in what the write does to information already stored. The descriptions give the intended semantics of each rule. Whether a trained policy uses them this way is an empirical question.
<table><tr><td>Model</td><td>State update</td><td>Update semantics</td></tr><tr><td>Mamba-2</td><td> $\mathrm { d i a g } ( a _ { t } ) h _ { t - 1 } + v _ { t } k _ { t } ^ { \top }$ </td><td>Accumulate: storing a similar cue again adds another write to its trace, so repeated occurrences of an event build up in the state, the property counting requires.</td></tr><tr><td>Mamba-3</td><td> $\operatorname* { d i a g } _ { v _ { t } k _ { t } ^ { \top } } ( a _ { t } e ^ { i \theta _ { t } } ) h _ { t - 1 }$ </td><td>+ Accumulate + rotate: as above, with phase evolution that provides an additional signal for temporal progression or motion.</td></tr><tr><td>GDN-2</td><td> $\begin{array} { r l } & { \alpha _ { t } ( I - \beta _ { t } k _ { t } k _ { t } ^ { \top } ) h _ { t - 1 } + } \\ & { \beta _ { t } v _ { t } k _ { t } ^ { \top } } \end{array}$ </td><td>Replace: storing a similar cue overwrites its previous content with the latest value, while other stored cues remain undisturbed. Repeated identical events therefore leave a single trace.</td></tr></table>

$$
\mathcal { L } _ { \mathrm { F M } } = \mathbb { E } \Big [ \| v _ { \theta } ( X _ { \tau } , \tau ; z _ { t } , m _ { t } ) - ( A _ { t } - X _ { 0 } ) \| _ { 2 } ^ { 2 } \Big ] .\tag{3}
$$

The decoder has four blocks. Each block applies selfattention over the action tokens, then cross-attention to $z _ { t } ,$ then cross-attention to $m _ { t } ,$ then a feed-forward network. The two cross-attention operations have separate parameters, so the current observation and the memory enter every block through separate paths. Flow time τ conditions every block through zero-initialized adaptive layer normalization.

d) Implementation and training: Training processes each demonstration as one causal scan over the full episode. The vision and text backbones stay frozen and are adapted with LoRA of rank 16 and 8, respectively. Real-robot policies train for 500 epochs. At inference, the decoder integrates the velocity field with $N = 4$ Euler steps. The robot executes the first $k = 4$ actions of each chunk and then replans. The frame encoder runs and the memory advances at every control step, including during chunk execution. The memory’s recurrent states and attention cache are reset between episodes.

## IV. EVALUATION

We evaluate ReCAT on two simulation benchmarks and three real-robot tasks. The evaluation asks three questions:

• (RQ1) How effective is ReCAT on standard and on memory-dependent manipulation?

• (RQ2) How do its observation encoder and its memory conditioning affect performance?

• (RQ3) How do the recurrent update rule and the memory capacity affect different memory demands?

We report stage-wise behavior within these sections and computational cost separately.

## A. Experimental setup

a) Simulation benchmarks: We evaluate ReCAT on LIBERO [37], including Spatial, Object, Goal, and LIBERO-10, using two cameras and 50 epochs. Each 10-task suite is evaluated over 500 episodes (50 per task). Since LIBERO is largely Markovian, it tests whether full-history recurrence preserves performance when the current observation is sufficient. We use LIBERO-Plus [38] to evaluate robustness to controlled visual and physical perturbations. RMBench [39] contains nine long-horizon bimanual tasks: five M(1) and four $M ( n )$ in which actions depend on past events no longer observable. Following its protocol, we train one policy per task on 50 demonstrations and evaluate 100 rollouts with unseen scenes.

![](images/8d875dd39e0ad9cf9df70d974d280012d5f6234bbb877b13d565b29b407cb5c2.jpg)  
the start position (left) has left the view by the re-grasp (right): the return target must be recalled  
Fig. 3: Real-robot tasks, one per row, each isolating one memory demand: PLANT counts scoops across five stages, POT TIMER times an interval while the scene is static, and SPONGE recalls an unmarked start position that has left the camera view. Outlined frames mark the decision point where the correct action depends on an earlier event, not on the current observation.

b) Real robot setup: We further evaluate the policies on three real-robot tasks that probe distinct memory demands under real-world execution variability. Unlike simulation, realworld manipulation introduces visual clutter, contact variation, friction, and trajectory variability, which can interfere with retaining task-relevant history. We therefore consider three tasks requiring spatial recall, discrete event counting, and interval timing, each containing states where the correct action cannot be determined from the current observation alone. The three tasks are shown in Fig. 3. We use a single-arm Franka Emika Panda in the DROID setup [40]. The robot receives 224 × 224 RGB images from a fixed right-side camera and a wrist-mounted camera, together with joint-space proprioception. Demonstrations are collected through teleoperation at 15 Hz. The policy predicts an eightdimensional action comprising seven joint positions and a continuous gripper width.

The SPONGE task evaluates spatial recall under clutter and broad spatial variation. The sponge is initialized across the robot’s reachable workspace without any positional markers indicating its start location. The robot places it on a plate, releases it, and must then return it to its original position, which is no longer directly observable once the sponge has moved. This requires the remembered location to be maintained through the intervening manipulation and surrounding clutter. 48 demonstrations, ∼480 steps per episode.

The PLANT task evaluates discrete event counting within a long-horizon, multi-stage manipulation sequence. The robot must place exactly two scoops of soil into a pot, put down the shovel, and pick up and insert a flower into the pot. Since the observations during the two scooping cycles are visually similar, the policy must track the number of completed scoops to determine whether to repeat the scooping action or proceed to the remaining stages. 45 demonstrations, ∼1,200 steps per episode, with varying scoop durations (125.4 ± 24.3 frames).

The POT TIMER task evaluates interval timing. The robot places a pot on a stove, waits for approximately 30 seconds, and then picks up a pepper and places it in the pot. The visual scene remains nearly unchanged during the waiting period and provides no direct indication of elapsed time, requiring the policy to maintain temporal state internally. 45 demonstrations, ∼980 steps per episode.

c) Protocol: We train one policy per task and configuration with a single seed for 500 epochs and evaluate the final checkpoint. Each policy runs 20 rollouts per task with randomized initial object positions. We report final success and the stage each rollout reached. In SPONGE, a return counts as success when the sponge lands within 2-4 cm of its original centre. In POT TIMER, a wait counts as correct when it lasts 29-31 s. With 20 rollouts, one rollout is five percentage points, and differences of that size are within rollout noise.

d) Metrics: We report success rate in all evaluations. For the real robot we also report policy inference latency and its reciprocal rate (Table VII), and trajectory smoothness as the average spectral arc length (SPARC) [41] of the joint velocity profiles across all joints. Higher SPARC is smoother.

## B. Baselines

For LIBERO and RMBench we report results from prior work. For the real-robot tasks we retrained every baseline on our demonstrations with the same cameras, control rate, action space, and rollout protocol. The short-history variants are our own extensions, since neither base method ships one. The baselines cover three ways a policy can access the past. All baselines run on the robot’s inference GPU 4060Ti. This excludes memory-augmented VLAs built on multibillion-parameter backbones [1]. An official implementation of DSSP [10] was not available at the time of our experiments, so we discuss it as concurrent work and restrict comparisons to the methods above and to reported benchmark results. The baselines cover three ways a policy can access the past. (i) No memory: DP is a standard Diffusion Policy conditioned on the current right- and wrist-view images and proprioception, and X-VLA [42] is a vision-language-action policy with no persistent temporal memory. (ii) Short-term memory: DP-H and X-VLA-H add the last eight observations (stride 10), encoded by a shared trainable ResNet-18 and by a frozen Florence-2 encoder compressing them into eight history tokens through a two-layer Transformer, respectively. (iii) Explicit attention-based memory: Gated Memory Policy (GMP) [17] retrieves visual and action information from a finite stored history using cross-attention and a calibrated binary memory gate. Diffusion Policy is the standard imitation-learning policy at a size comparable to ours (268M parameters). X-VLA is a 0.9B-parameter pretrained VLA and one of the strongest model that fits the robot’s GPU. GMP is the closest published memory policy at comparable size. All baselines were selected to be deployable on the robot’s inference GPU, which excludes recent memory-augmented VLAs built on multi-billion-parameter backbones [1].

## C. Results and Discussion

Unless stated otherwise, ReCAT denotes the Mamba-2 instantiation.

RQ1: ReCAT is competitive on standard manipulation and strong on memory-dependent manipulation. On LIBERO, ReCAT reaches 95.3% average success across the four suites (Table II). Diffusion Policy reaches 72.4%. X-VLA, which uses large-scale pretraining, reaches 98.1%. On LIBERO-Plus, ReCAT reaches 64.2% and X-VLA 71.4%. ReCAT therefore remains competitive on tasks that the current observation alone can solve.

On RMBench, ReCAT reaches 62.4% average success over nine tasks (Table III). It has the best or tied-best score on six of them, including three of the four M(n) tasks. EventVLA has a higher overall average of 67.9%. It builds on a 4Bparameter backbone, about ten times larger than ReCAT, with an additional large-scale pretraining phase. Performance varies across tasks. ReCAT reaches 100% on Put Back Block and Swap Blocks but 4% on Observe and Pick Up and 6% on Press Button. Chronos [11] is not in the table because we could not establish a matched protocol as it is not evaluated on all tasks.

The real-robot tasks show the largest gap (Table IV). Policies without persistent memory succeed in 0-8.3% of rollouts. X-VLA-H, the strongest baseline, reaches 8.3% (95% Wilson CI [3.6%, 18.1%]). ReCAT variants reach 50.0-66.7%, and the best variant reaches 66.7% (95% CI [54.1%, 77.3%]). The two intervals do not overlap. All three ReCAT variants wait the correct interval in every POT TIMER rollout, and the Mamba variants complete 85-90% of PLANT rollouts. The stage-wise columns of Table IV show where the baselines fail, which we examine next.

Stage-wise view. The three tasks share a structure: observation-driven stages lead to one decision that the current observation cannot resolve. Table IV breaks each rollout down by stage. Baselines complete the early stages and fail at that decision. X-VLA-H places the sponge on the plate in 15 of 20 rollouts but returns it to the right place in only 2. In PLANT, it scoops at least once in 19 of 20 rollouts but completes the task in 2, and in 17 of them it keeps scooping past two. In POT TIMER, it places the pot in every rollout but waits the correct interval in 1. ReCAT completes the decision in most rollouts, and the failures that remain, occur before the temporal memory decision stage.

TABLE II: Success rates (%) on LIBERO.
<table><tr><td>Method</td><td>Spatial</td><td>Object</td><td>Goal</td><td>LIBERO-10</td><td>Avg.</td><td>LIBERO-Plus</td></tr><tr><td>Diffusion policy</td><td>78.3</td><td>92.5</td><td>68.3</td><td>50.5</td><td>72.4</td><td></td></tr><tr><td>X-VLA [42]</td><td>98.2</td><td>98.6</td><td>97.8</td><td>97.6</td><td>98.1</td><td>71.4</td></tr><tr><td>Ours</td><td>96.4</td><td>97.2</td><td>97.2</td><td>90.4</td><td>95.3</td><td>64.2</td></tr></table>

TABLE III: Success rates (%) on RMBench. \* Indicates models utilizing large scale pretraining.
<table><tr><td>Task</td><td>TMC</td><td>DP (90M)</td><td>ACT (80M)</td><td>*pi0.5 (3B)</td><td>*X-VLA (0.9B)</td><td>MEM-0 (10B)</td><td>EventVLA (4B)</td><td>*MemoryVLA (7.3B)</td><td>*MemER (10B)</td><td>Ours (374M)</td></tr><tr><td>Observe and Pick Up</td><td>M(1)</td><td>1</td><td>1</td><td>9</td><td>9</td><td>4</td><td>21</td><td>2</td><td>7</td><td>4</td></tr><tr><td>Rearrange Blocks</td><td>M(1)</td><td>0</td><td>29</td><td>13</td><td>13</td><td>89</td><td>96</td><td>53</td><td>17</td><td>96</td></tr><tr><td>Put Back Block</td><td>M(1)</td><td>0</td><td>0</td><td>11</td><td>18</td><td>90</td><td>95</td><td>81</td><td>0</td><td>100</td></tr><tr><td>Swap Blocks</td><td>M(1)</td><td>11</td><td>2</td><td>24</td><td>16</td><td>67</td><td>96</td><td>76</td><td>14</td><td>100</td></tr><tr><td>Swap T</td><td>M(1)</td><td>20</td><td>2</td><td>15</td><td>3</td><td>14</td><td>87</td><td>9</td><td>7</td><td>64</td></tr><tr><td>Average</td><td>M(1)</td><td>6.4</td><td>6.8</td><td>14.4</td><td>11.8</td><td>52.8</td><td>79.0</td><td>44.2</td><td>9.0</td><td>72.8</td></tr><tr><td>Battery Try</td><td>M(n)</td><td>10</td><td>19</td><td>16</td><td>26</td><td>28</td><td>35</td><td>33</td><td>27</td><td>43</td></tr><tr><td>Blocks Ranking Try</td><td>M(n)</td><td>10</td><td>0</td><td>6</td><td>1</td><td>18</td><td>81</td><td>53</td><td>12</td><td>99</td></tr><tr><td>Cover Blocks</td><td>M(n)</td><td>0</td><td>0</td><td>0</td><td>2</td><td>68</td><td>97</td><td>69</td><td>40</td><td>50</td></tr><tr><td>Press Button</td><td>M(n)</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>3</td><td>0</td><td>0</td><td>6</td></tr><tr><td>Average</td><td>M(n)</td><td>5.0</td><td>4.8</td><td>5.5</td><td>7.3</td><td>28.5</td><td>54.0</td><td>38.8</td><td>19.8</td><td>49.5</td></tr><tr><td>Overall average</td><td></td><td>5.8</td><td>5.9</td><td>10.4</td><td>9.8</td><td>42.0</td><td>67.9</td><td>41.8</td><td>13.8</td><td>62.4</td></tr></table>

## RQ2: Both the observation encoder and the memory conditioning matter, and the largest effects appear on PLANT.

Observation encoder. We fix the memory to Mamba-2 and vary the frame encoder (Table VI). The hybrid Mambaattention encoder reaches 66.7% average success. A full-Transformer encoder reaches 25.0%. It drops from 85% to 0% on PLANT, from 95% to 65% on POT TIMER, and from 20% to 10% on SPONGE. A pure-SSM configuration removes the attention layer from both the frame encoder and the memory. It reaches 25% on PLANT, the only task on which we evaluated it. Removing the memory module gives 0% on all three tasks. Removing the multimodal encoder also gives 0%. The encoder supplies both the decoder input and the memory input, so these differences are end-to-end effects of representation quality. Counting is the most sensitive demand: repeated scoops must produce consistent representations before the memory can accumulate them. The no-memory result shows that the temporal module as a whole is necessary. It does not separate the contributions of its recurrent layers and its attention layer.

Memory conditioning. We fix the encoder and memory and vary how m enters the four-block decoder (Table V). Memory cross-attention in every block reaches 66.7% average success. Cross-attention in the final two blocks only (late fusion) reaches 38.3%. Modulation through AdaLN, a learned scale, or a content gate reaches 31.7%, 28.3%, and 8.3%. Memory dropout changes training rather than the readout and reaches 20.0%. The largest gap is on PLANT, where every-block cross-attention reaches 85% and no alternative exceeds 10%. Several alternatives fail before the count matters. Late fusion never picks up the shovel. Dropout and Scale miss it in 17 and 15 of 20 rollouts. The Gate and AdaLN variants stop after a single scoop in 7 of 20 rollouts each. Weak conditioning therefore harms both manipulation and the count readout. In these evaluations, POT TIMER is less sensitive. Every variant that places the pot also waits the correct interval. Late fusion and AdaLN reach 90%, against 95% for every-block crossattention. The Gate variant places the pot in only 10 of 20 rollouts, and in 6 of the other 10 it reaches for the pepper first. SPONGE is the exception. Late fusion reaches the plate in 14 of 20 rollouts and returns the sponge correctly in 5 of them, for the highest observed success at 25%. Every-block cross-attention reaches the plate in 4 of 20 and returns all 4, for 20%. That difference is one rollout. No evaluated variant reaches high SPONGE success. Grasping and spatial recall must both be learned from 48 demonstrations with broad position variation, and conditioning choices do not close that gap.

## RQ3: The update rule and the memory capacity have task-dependent effects, and the additive rules have the highest observed success on counting.

Update rule. We keep the encoder, the attention layer, and the decoder fixed and replace the recurrent update (Table IV). On PLANT, Mamba-2 and Mamba-3 reach 85% and 90%. Gated DeltaNet-2 reaches 20%. It records two scoops in 14 of 20 rollouts but completes the task in 4, so most of its failures come after the count is reached. On POT TIMER, all three rules wait the correct interval in every rollout and reach 90-100% success. On SPONGE, Gated DeltaNet-2 has the highest observed success at 40%. Mamba-2 reaches 20% and Mamba-3 0%. The stage-wise columns qualify this gap. Mamba-2 and Gated DeltaNet-2 return the sponge correctly in every rollout that reaches the plate, 4 of 4 and 8 of 8. Mamba-3 never reaches the plate. The end-to-end difference on SPONGE therefore arises before the recall step, in grasping and placement. The update rules motivate a hypothesis for the PLANT pattern. Additive writes let evidence of repeated scoops accumulate, while the delta rule replaces the content stored under a matching cue and may represent multiplicity less directly (Table I). We do not measure the internal count, so this remains a hypothesis. Mamba-2 has the highest average success and is the default in all other experiments.

Memory capacity. We fix the rule to Mamba-2 and vary depth and state width, with PLANT as the most sensitive task. Reducing the memory from six blocks to two, which keeps the attention layer and leaves one Mamba-2 layer, lowers PLANT success from 85% to 65%. For state width we evaluate $d _ { \mathrm { s t a t e } } \in \{ 3 2 , 6 4 , 1 2 8 , 2 5 6 \}$ with everything else fixed. PLANT success is 15%, 20%, 85%, and 0%. At widths 32 and 64, 35-40% of rollouts reach two scoops but do not complete the task due to precision failures. At width 256 the policy fails before reaching two scoops. Width 128 has the highest observed success in this sweep. The non-monotonic result shows sensitivity to the configuration under a single training run. It does not establish that larger states hold less. We use six blocks and width 128 in all other experiments.

Computational efficiency. Table VII compares model size, inference cost, and trajectory smoothness under the same hardware and inference settings. ReCAT has 374M parameters, of which 142M are trainable. Policy inference takes 59 ms per step, against 110 ms for DP-H, 93 ms for GMP, and 288 ms for X-VLA-H. The reciprocal rate is 16.9 Hz. ReCAT also has the highest SPARC, that is the smoothest end-effector trajectories, among the compared policies.

TABLE IV: Real-robot results, step by step. Each task moves from observation-driven stages to a later decision that depends on retained history (outlined in Fig. 3). Breaking results down this way, instead of reporting only final success, shows exactly where a policy fails. For PLANT, the three scoop columns are mutually exclusive: a rollout is counted under the number of scoops it performed before it either moved on to the flower or stayed in the scooping loop until timeout. For POT TIMER, “No wait” and “Wrong wait” are rollouts that left the waiting state early or outside the 29-31 s window. For SPONGE, “Returned to wrong position” and “Returned to same position” partition the rollouts that placed the sponge, except rollouts that stalled at the plate. Success is the final column of each task. All values are percentages of 20 rollouts with randomised start positions. Avg. is the mean over the success rate of three tasks.
<table><tr><td></td><td colspan="4">Plant</td><td colspan="6">Pot timer</td><td colspan="3">Sponge</td><td></td></tr><tr><td>Method</td><td>1 scoop</td><td>2 scoops</td><td>3 or more</td><td>Flower planted Success</td><td>Pot on stove</td><td>No wait</td><td>Wrong wait</td><td>≈30s wait</td><td>Pepper in the pot</td><td>Success</td><td>Pick sponge, put on plate</td><td>Returned to wrong position</td><td>Returned to same position Success</td><td>Avg.</td></tr><tr><td>Baselines</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Diffusion policy [43]</td><td></td><td></td><td>一</td><td>0</td><td></td><td></td><td></td><td></td><td></td><td>0</td><td>一</td><td></td><td>0 0</td><td>0.0 0.0</td></tr><tr><td>Diffusion policy [43] + history</td><td>15</td><td>10</td><td>0</td><td>0</td><td>100</td><td>25</td><td>0</td><td>0</td><td>25</td><td>0</td><td>15</td><td>-</td><td></td><td>0.0</td></tr><tr><td>X-VLA [42] X-VLA [42] + history</td><td>– 0</td><td>一 10</td><td>- 85</td><td>0 10</td><td>一 100</td><td>一 0</td><td>I 35</td><td>- 5</td><td>一 40</td><td>0</td><td>一 75</td><td>一 65</td><td>0 10</td><td>8.3</td></tr><tr><td>Gated Memory Policy (GMP) [17]</td><td>65</td><td>20</td><td>10</td><td>15</td><td>95</td><td>20</td><td>10</td><td>5</td><td>35</td><td>5.5</td><td>50</td><td>50</td><td>0</td><td>6.7</td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Ours</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td>20</td><td>66.7</td></tr><tr><td>ReCAT-Mamba-2</td><td>0</td><td>100 95</td><td>0</td><td>85</td><td>100</td><td>0</td><td>0</td><td>100</td><td>95</td><td>95</td><td>20</td><td>0 0</td><td>0</td><td></td></tr><tr><td>ReCAT-Mamba-3</td><td>0 0</td><td>70</td><td>0 0</td><td>90 20</td><td>100 100</td><td>0</td><td>0</td><td>100 100</td><td>100</td><td>100 90</td><td>0 40</td><td>0</td><td>40</td><td>63.3 50.0</td></tr><tr><td>ReCAT-Gated DeltaNet 2</td><td></td><td></td><td></td><td></td><td></td><td>0</td><td>0</td><td></td><td>90</td><td></td><td></td><td></td><td></td><td></td></tr></table>

TABLE V: Step-wise real-robot success across six memory-to-action conditioning mechanisms. Columns as in Table IV.
<table><tr><td></td><td colspan="4">Plant</td><td></td><td colspan="5">Pot timer</td><td colspan="3">Sponge</td><td></td></tr><tr><td>Integration</td><td>1 scoop</td><td>2 scoops</td><td>3 or more</td><td>Flower planted Success</td><td>Pot on stove</td><td>No wait</td><td>Wrong wait</td><td>≈30 s wait</td><td>Pepper in the pot</td><td>Success</td><td>Pick sponge, put on plate</td><td>Returned to wrong position</td><td>Returned to same position</td><td>Avg.</td></tr><tr><td>ReCAT, cross-attention</td><td>0</td><td>100</td><td>0</td><td>85</td><td>100</td><td>0</td><td>0</td><td>100</td><td>95</td><td>95</td><td>20</td><td>0</td><td>20</td><td>66.7</td></tr><tr><td>Integration variants</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Late fusion</td><td>0</td><td>0</td><td>0</td><td>0</td><td>90</td><td>0</td><td>0</td><td>90</td><td>90</td><td>90</td><td>70†</td><td>40</td><td>25</td><td>38.3</td></tr><tr><td>AdaLN</td><td>35</td><td>15</td><td>0</td><td>5</td><td>100</td><td>0</td><td>0</td><td>100</td><td>90</td><td>90</td><td>25</td><td>5</td><td>0</td><td>31.7</td></tr><tr><td>Scale</td><td>0</td><td>25</td><td>0</td><td>10</td><td>80</td><td>0</td><td>0</td><td>80</td><td>75</td><td>75</td><td>40</td><td>40</td><td>0</td><td>28.3</td></tr><tr><td>Dropout</td><td>5</td><td>10</td><td>0</td><td>0</td><td>90</td><td>0</td><td>0</td><td>90</td><td>60</td><td>60</td><td>20</td><td>10</td><td>0</td><td>20.0</td></tr><tr><td>Gate</td><td>35</td><td>5</td><td>0</td><td>5</td><td>50</td><td>0</td><td>0</td><td>50</td><td>20</td><td>20</td><td>20</td><td>0</td><td>0</td><td>8.3</td></tr></table>

TABLE VI: Real-robot success rates for observation-encoder and component ablations.
<table><tr><td>Setting</td><td>Sponge</td><td>Plant</td><td>Pot timer</td><td>Avg.</td></tr><tr><td>Ours, hybrid encoder</td><td>20</td><td>85</td><td>95</td><td>66.7</td></tr><tr><td>Ours, full-transformer encoder</td><td>10</td><td>0</td><td>65</td><td>25.0</td></tr><tr><td>Ours, pure-SSM encoder</td><td></td><td>25</td><td></td><td></td></tr><tr><td>Ours, w/o multimodal observation encoder tokens</td><td>0</td><td>0</td><td>0</td><td>0.0</td></tr><tr><td>Ours, without memory module</td><td>0</td><td>0</td><td>0</td><td>0.0</td></tr></table>

TABLE VII: Computational cost on the same hardware and inference setting at. Latency is per policy inference; the rate is its reciprocal. SPARC: higher is smoother.
<table><tr><td>Metric</td><td>ReCAT</td><td>DP-H</td><td>X-VLA-H</td><td>GMP</td></tr><tr><td>Total / trainable parameters</td><td>374M / 142M</td><td>267.9M / 267.9M</td><td>881.9M / 881.9M</td><td>282.2M / 188.7M</td></tr><tr><td>Peak inference VRAM</td><td>1.44GB</td><td>0.65GB</td><td>2.55GB</td><td>1.09GB</td></tr><tr><td>Inference latency/frequency</td><td>0.059s/16.9hz</td><td>0.11s/9hz</td><td>0.288s/3.47hz</td><td>0.093s/10.75hz</td></tr><tr><td>Smoothness metric (SPARC [41])</td><td>-5.16</td><td>-6.19</td><td>-6.76</td><td>-13.06</td></tr></table>

## V. CONCLUSIONS

We presented ReCAT, a structured recurrent policy for memory-dependent manipulation. It’s memory integrates the full observation history through Mamba-2 layers and one causal attention layer. It’s flow-matching decoder reads the current observation and the memory through separate crossattention in every block. This design reaches 95.3% on

LIBERO, 62.4% on RMBench, and 66.7% across three realrobot tasks that require spatial recall, counting, and timing. It also has the lowest measured inference latency among the compared memory-based policies, at 59 ms per step. The analysis explains where this performance comes from. Stage-wise results show that short-history policies execute the immediate manipulation but fail at the decision that depends on earlier information. Within the evaluated configurations, the hybrid observation encoder and memory cross-attention in every decoder block gave the highest average success. The recurrent update rule has task-dependent effects: the additive rules have the highest observed success on counting and timing, and the delta rule on spatial recall. Effective memorydependent control therefore, depends on what is written to memory, how it is updated, and how the action generator reads it alongside the current observation.

## VI. LIMITATIONS AND FUTURE WORK

Our real-robot evaluation covers one task per memory demand, each evaluated over 20 rollouts. Broader studies across multiple tasks, embodiments, are needed to assess generalization, including multiple counts and durations distinguished by language instructions. The memory ablation removes the recurrent layers and the attention layer together, so their individual contributions remain open. The SPONGE results show a remaining trade-off between memory use and fine-grained spatial control. Future work should examine these factors and improve success on spatial-return manipulation under limited demonstrations with larger task diversity.

## ACKNOWLEDGMENT

We used Claude and ChatGPT to polish the text of this paper. The authors reviewed every generated item, and all technical claims, experiments, and results are our own. The research presented in this paper was funded by the Deutsche Forschungsgemeinschaft (DFG, German Research Foundation) – 448648559. The authors gratefully acknowledge the computing time provided on the high-performance computer HoreKa by the National High-Performance Computing Center at KIT and by Gauss Centre for Supercomputing e.V. (www.gauss-centre.eu) for GCS Supercomputer JUPITER at Julich Supercomputing Centre (JSC).¨

## REFERENCES

[1] H. Shi, B. Xie, Y. Liu, L. Sun, F. Liu, T. Wang, E. Zhou, H. Fan, X. Zhang, and G. Huang, “Memoryvla: Perceptual-cognitive memory in vision-language-action models for robotic manipulation,” in International Conference on Learning Representations, vol. 2026, 2026, pp. 18 567–18 602.

[2] H. Li, F. Shen, D. Chen, L. Yang, X. Wang, J. Shi, Z. Bing, Z. Liu, and A. Knoll, “Remem-vla: Empowering vision-languageaction model with memory via dual-level recurrent queries,” arXiv preprint arXiv:2603.12942, 2026.

[3] A. Gu, K. Goel, and C. Re, “Efficiently modeling long sequences with structured state spaces,” in International Conference on Learning Representations, 2021.

[4] S. Yang, J. Kautz, and A. Hatamizadeh, “Gated delta networks: Improving Mamba2 with delta rule,” arXiv preprint arXiv:2412.06464, 2024.

[5] A. S. Lahoti, K. Li, B. Chen, C. Wang, A. Bick, Z. Kolter, T. Dao, and A. Gu, “Mamba-3: Improved sequence modeling using state space principles,” in International Conference on Learning Representations, vol. 2026, 2026, pp. 86 173–86 202.

[6] P. Vanjani, Z. Li, J. Suliga, M. Reuss, G. Geraci, X. Jiang, and R. Lioutikov, “Dam-vla: Decoupled asynchronous multimodal vision language action model,” arXiv preprint arXiv:2606.12105, 2026.

[7] Y. Zhou, Y. Lin, F. Peng, J. Chen, K. Huang, H. Yang, and Z. Yin, “Mtil: Encoding full history with mamba for temporal imitation learning,” IEEE Robotics and Automation Letters, 2025.

[8] Y. Yoo, J. Hu, Y. Zhu, B. Liu, Q. Liu, R. Mart´ın-Mart´ın, and P. Stone, “Robossm: Scalable in-context imitation learning via state-space models,” arXiv preprint arXiv:2509.19658, 2025.

[9] T. Tsuji, “Mamba as a motion encoder for robotic imitation learning,” IEEE Access, vol. 13, pp. 69 941–69 949, 2025.

[10] Z. Guan, J. Hu, H. Fang, Y. Jiang, Y. Huang, S. Li, X. Li, and Y. Ban, “Dssp: Diffusion state space policy with full-history encoding,” arXiv preprint arXiv:2605.14598, 2026.

[11] Y. Zhou, Y. Wang, N. Wang, S. Xing, S. Tu, X. Li, J. Zhang, N. Jiang, Y. Lin, H. Yang, et al., “Chronos: A physics-informed full-history framework for non-markovian long-horizon manipulation,” arXiv preprint arXiv:2606.30318, 2026.

[12] E. Cherepanov, N. Kachaev, D. Zelezetsky, A. Bulatov, A. Pshenitsyn, Y. Kuratov, A. Skrynnik, A. I. Panov, and A. K. Kovalev, “µvla: On recurrent memory for partially observable manipulation in vla models,” arXiv preprint arXiv:2606.12497, 2026.

[13] M. S. Mark, J. Liang, M. Attarian, C. Fu, D. Dwibedi, D. Shah, and A. Kumar, “Bpp: Long-context robot imitation learning by focusing on key history frames,” arXiv preprint arXiv:2602.15010, 2026.

[14] C. Yin, W. Xu, J. Yang, S. Tam, H. Liu, Y. Yao, X. Zeng, J. Cui, Y. Wang, Z. Yin, et al., “Simplememvla: A simple but effective nativevideo memory for vision-language-action models,” arXiv preprint arXiv:2609.05533, 2026.

[15] M. Torne, K. Pertsch, H. Walke, K. Vedder, S. Nair, B. Ichter, A. Z. Ren, H. Wang, J. Tang, K. Stachowicz, et al., “Mem: Multi-scale embodied memory for vision language action models,” arXiv preprint arXiv:2603.03596, 2026.

[16] R. Shah, Y. Li, F. Bello, Y. Zhu, and R. Mart´ın-Mart´ın, “Memory retrieval in visuomotor policies for long-horizon robot control,” arXiv preprint arXiv:2606.25136, 2026.

[17] Y. Gao, J. Liu, S. Li, and S. Song, “Gated memory policy,” arXiv preprint arXiv:2604.18933, 2026.

[18] X. Jia, Q. Wang, A. Donat, B. Xing, G. Li, H. Zhou, O. Celik, D. Blessing, R. Lioutikov, and G. Neumann, “Mail: Improving imitation learning with selective state space models,” in 8th Annual Conference on Robot Learning, 2024.

[19] J. Cao, Q. Zhang, J. Sun, J. Wang, H. Cheng, Y. Li, J. Ma, K. Wu, Z. Xu, Y. Shao, et al., “Mamba policy: Towards efficient 3d diffusion policy with hybrid selective state models,” in 2025 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS). IEEE, 2025, pp. 11 359–11 366.

[20] N. Oh, J. Jang, M. Jung, and D. Park, “Dispo: diffusion-ssm based policy learning for coarse-to-fine action discretization,” arXiv preprint arXiv:2409.14719, 2024.

[21] Y. Takagi, M. Kambara, D. Yashima, K. Seno, K. Tokura, and K. Sugiura, “Anolevla: Lightweight vision-language-action model with deep state space models for mobile manipulation,” arXiv preprint arXiv:2603.15046, 2026.

[22] Y. Jiang, Y. Chebotar, R. Zheng, F. Hu, Y. Ge, J. Wu, T. Dai, S. Reed, L. Fei-Fei, Y. Zhu, et al., “Robottt: Context scaling for robot policies,” arXiv preprint arXiv:2607.15275, 2026.

[23] A. Katharopoulos, A. Vyas, N. Pappas, and F. Fleuret, “Transformers are RNNs: Fast autoregressive transformers with linear attention,” in International Conference on Machine Learning. PMLR, 2020, pp. 5156–5165.

[24] A. Gu and T. Dao, “Mamba: Linear-time sequence modeling with selective state spaces,” arXiv preprint arXiv:2312.00752, 2023.

[25] T. Dao and A. Gu, “Transformers are ssms: Generalized models and efficient algorithms through structured state space duality,” arXiv preprint arXiv:2405.21060, 2024.

[26] I. Schlag, K. Irie, and J. Schmidhuber, “Linear transformers are secretly fast weight programmers,” in International Conference on Machine Learning. PMLR, 2021, pp. 9355–9366.

[27] S. Yang, B. Wang, Y. Zhang, Y. Shen, and Y. Kim, “Parallelizing linear transformers with the delta rule over sequence length,” Advances in neural information processing systems, vol. 37, pp. 115 491–115 522, 2024.

[28] S. Yang, J. Kautz, and A. Hatamizadeh, “Gated delta networks: Improving mamba2 with delta rule,” in International Conference on Learning Representations, vol. 2025, 2025, pp. 29 687–29 707.

[29] R. Grazzi, J. Siems, A. Zela, J. Franke, F. Hutter, et al., “Unlocking state-tracking in linear rnns through negative eigenvalues,” in International Conference on Learning Representations, vol. 2025, 2025, pp. 36 565–36 597.

[30] S. Arora, S. Eyuboglu, A. Timalsina, I. Johnson, M. Poli, J. Zou, A. Rudra, and C. Re, “Zoology: Measuring and improving recall in´ efficient language models,” arXiv preprint arXiv:2312.04927, 2023.

[31] K. Team, Y. Zhang, Z. Lin, X. Yao, J. Hu, F. Meng, C. Liu, X. Men, S. Yang, Z. Li, et al., “Kimi linear: An expressive, efficient attention architecture,” arXiv preprint arXiv:2510.26692, 2025.

[32] J. Lee, S. Lee, J. Shin, H. Jung, S. Kim, D. Cho, H. J. Kim, J.-B. Huang, and F. Huang, “Dynaflip: Rethinking robotics perception via tri-modaldynamics guided representation,” arXiv preprint arXiv:2605.30350, 2026.

[33] J. Gao, F. Ye, J. Zhang, and W. Qian, “Compressor-vla: Instructionguided visual token compression for efficient robotic manipulation,” arXiv preprint arXiv:2511.18950, 2025.

[34] M. Reuss, H. Zhou, M. Ruhle,¨ Omer Erdin<sup>¨</sup> c¸ Yagmurlu, F. Otto,˘ and R. Lioutikov, “Flower: Democratizing generalist robot policies with efficient vision-language-action flow policies,” 2025. [Online]. Available: https://arxiv.org/abs/2509.04996

[35] P. Vanjani, P. Mattes, X. Jia, V. Dave, and R. Lioutikov, “Disdp: Robust imitation learning via disentangled diffusion policies,” in Reinforcement Learning Conference, 2025.

[36] M. Reuss, M. Li, X. Jia, and R. Lioutikov, “Goal-conditioned imitation learning using score-based diffusion policies,” arXiv preprint arXiv:2304.02532, 2023.

[37] B. Liu, Y. Zhu, C. Gao, Y. Feng, Q. Liu, Y. Zhu, and P. Stone, “Libero: Benchmarking knowledge transfer for lifelong robot learning,” Advances in Neural Information Processing Systems, vol. 36, pp. 44 776– 44 791, 2023.

[38] S. Fei, S. Wang, J. Shi, Z. Dai, J. Cai, P. Qian, L. Ji, X. He, S. Zhang, Z. Fei, et al., “Libero-plus: In-depth robustness analysis of visionlanguage-action models,” arXiv preprint arXiv:2510.13626, 2025.

[39] T. Chen, Y. Wang, M. Li, Y. Qin, H. Shi, Z. Li, Y. Hu, Y. Zhang, K. Wang, Y. Chen, et al., “Rmbench: Memory-dependent robotic manipulation benchmark with insights into policy design,” arXiv preprint arXiv:2603.01229, 2026.

[40] A. Khazatsky, K. Pertsch, S. Nair, A. Balakrishna, S. Dasari, S. Karamcheti, S. Nasiriany, M. K. Srirama, L. Y. Chen, K. Ellis, et al., “Droid: A large-scale in-the-wild robot manipulation dataset,” arXiv preprint arXiv:2403.12945, 2024.

[41] A. J. Scholp, M. R. Hoffman, S. P. Rosen, S. M. Abdelhalim, C. A. Jones, J. J. Jiang, and T. M. McCulloch, “Spectral arc length as a method to quantify pharyngeal high-resolution manometric curve smoothness,” Neurogastroenterology & Motility, vol. 33, no. 10, p. e14122, 2021.

[42] J. Zheng, J. Li, Z. Wang, D. Liu, X. Kang, Y. Feng, Y. Zheng, J. Zou, Y. Chen, J. Zeng, et al., “X-vla: Soft-prompted transformer as scalable cross-embodiment vision-language-action model,” in International Conference on Learning Representations, vol. 2026, 2026, pp. 60 580– 60 606.

[43] C. Chi, S. Feng, Y. Du, Z. Xu, E. Cousineau, B. Burchfiel, and S. Song, “Diffusion policy: Visuomotor policy learning via action diffusion,” in Robotics: Science and Systems (RSS), 2023.
# URUQI: LEARNING SPATIAL COGNITION FROM VISUAL EXPERIENCE

Shichao Li<sup>1</sup> Meiqi Wang<sup>2</sup> Fei Su<sup>1</sup> Zhicheng Zhao<sup>1</sup>

<sup>1</sup>Beijing University of Posts and Telecommunications <sup>2</sup>Tsinghua University lishichao@bupt.edu.cn zhaozc@bupt.edu.cn

## ABSTRACT

Spatial intelligence requires maintaining a coherent understanding of the world as the embodied agent moves. Like humans, the agent must use its own motion to interpret changes across observations and update object locations and spatial relations accordingly. Despite spatial post-training having substantially broadened the spatial intelligence of vision-language models (VLMs), they still struggle with two atomic spatial capabilities: tracking self-motion and mapping the surrounding world during motion. To address this gap, we provide dense multi-turn supervision over interleaved atomic capabilities within each training episode, mimicking the visual experience of a continuously moving agent that reasons as it observes. To scale this up, we synthesize 11,738 motif-driven camera trajectories over a broad range of 3D scenes, supporting self-motion tracking, persistent object mapping, and rich spatial operations within each visual experience. By training models to reason over these atomic questions, our URUQI<sub>Syn</sub>-8B improves accuracy from 15.84% to 47.73% on our URUQI benchmark comprising 52k questions across 2.7k episodes. URUQI-SI-Mix-8B further reaches 50.41%, comparable to the 50.08% achieved by GPT-6 Astra. Trained solely on our synthesized data, $\mathrm { U R U Q I _ { S y n } } – 8 \mathbf { B }$ achieves an average relative accuracy improvement of 17.13% over its InternVL3-8B backbone across three external spatial benchmarks. These results highlight continuous visual experience as a scalable source of supervision for developing spatial cognition in VLMs.

## 1 INTRODUCTION

Spatial reasoning is becoming a core capability of general-purpose vision-language models (VLMs) that aim to understand the physical world. A VLM with native spatial intelligence is expected to ground open-vocabulary objects and integrate metric, relational, and reference-frame spatial information directly into open-ended visual-language reasoning. Recent work has shown that such capabilities can be substantially improved with large-scale spatial instruction tuning (Chen et al., 2024; Cheng et al., 2024; Cai et al., 2026).

However, it remains unclear whether success on diverse spatial questions reflects a coherent understanding of the scene across changing viewpoints. Figure 1 illustrates this issue with two examples from MMSI-Bench (Yang et al., 2026c) through self-motion tracking (M1) and object mapping (M2). Qwen3.8-27B (Qwen Team, 2026b) and SenseNova-SI-1.5 (Cai et al., 2026) make inaccurate motion and object-location predictions, while their estimates of the same objects are inconsistent across viewpoints. GPT-5.5 (OpenAI, 2026a) also predicts inaccurate object locations, but its estimates remain consistent across views. We swap the query order of M1 and M2 and find that its object predictions change accordingly. This dependency on query order suggests that GPT-5.5 computes object locations from earlier location estimates and its predicted camera motion. These results highlight a mismatch between task accuracy and coherent spatial understanding across observations.

To address this gap, prior work has explored supervising intermediate reasoning steps through chainof-thought (Liu et al., 2025; Cai et al., 2026; Li et al., 2026a), cognitive maps (Wang et al., 2025; Yang et al., 2025), structured representations (Hua et al., 2026), and executable reasoning programs (Marsili et al., 2025). However, linguistic reasoning traces provide potentially ambiguous supervision for geometric state transitions (Kancheti et al., 2026), while explicit maps and programs rely on predefined representations and task-specific operations. This leaves a more general question: What intermediate supervision can capture how spatial information evolves across observations and generalize to unseen spatial tasks?

![](images/e0a77b20edee420d903425119cb12c6d818b01bc81733bede38e8764524f5ae7.jpg)  
Figure 1: Visualization of two MMSI-Bench (Yang et al., 2026c) examples by querying M1 and M2 before asking the final question. The query order is shown below each plot. The coordinate frame is defined at $C _ { 1 }$ . The black and blue arrows mark the viewing directions at $C _ { 1 }$ and $C _ { 2 } .$ , respectively, while the blue dashed line denotes the displacement from $C _ { 1 }$ to $C _ { 2 }$ predicted by M1. For M2, filled markers show object locations predicted from Image 1, while hollow markers show Image 2 predictions transformed into the $C _ { 1 }$ frame using the predicted self-motion. The overlap between filled and hollow markers reflects the cross-view consistency of object-location predictions.

We answer this question from a basic geometric property of visual experience: consecutive spatial observations are connected by the self-motion of the observer (Wang & Spelke, 2000; Wolbers et al., 2008). Unlike generic multi-image understanding, which mainly integrates semantic information across images (Jiang et al., 2024), spatial reasoning must also recover the geometric relationship between observations. Our key insight is that observer self-motion is more than a prediction target because it determines how spatial information should change from one observation to the next.

This suggests that motion estimation should be explicitly coupled with spatial state updates during learning. The model must first recover how the observer moves, use self-motion to update spatial relations established from previous observations, and then use the resulting state to answer the required query. Current spatial post-training often supervises these capabilities through separate task-specific questions, leaving the transition from observer motion to spatial state update largely implicit.

Learning this transition requires visual observations to be aligned with observer motion and the resulting changes in spatial relations. We therefore introduce URUQI, a scalable spatial compiler for synthesizing continuous visual experiences and compiling them into dense spatial supervision. URUQI generates motif-driven camera trajectories across diverse 3D scenes, where each motif controls how observer motion, visibility, and spatial evidence unfold over time. Using the underlying camera poses, scene geometry, and object identities, each trajectory is then compiled into multiturn supervision over three linked capabilities: self-motion tracking, which captures changes in viewpoint; persistent object mapping, which tracks object locations as the observer moves; and operations over state, which supports downstream reasoning over the acquired spatial information. Questions derived from the same trajectory are further organized into shared episodes, so that motion, state updates, and spatial operations are learned as connected parts of the same experience rather than as isolated QA examples.

To the best of our knowledge, URUQI is the first to jointly supervise self-motion, persistent object mapping, and downstream spatial reasoning within continuous visual experiences. Scaling this compilation across diverse trajectories, we construct URUQI-600k, a large-scale dataset with dense spatial supervision. $\mathrm { U R U Q I _ { S y n } } – 8 \mathbf { B }$ trained on this corpus achieves substantial gains in episodic spatial reasoning on URUQI benchmark in Section 4.2 and transfers effectively to external spatial benchmark without using any training data from these target benchmarks (Section 4.3). Moreover, combining our supervision with only 100K Sensenova-SI-8M examples further improves SenseNova-SI-1.5 (Cai et al., 2026), showing that our supervision complements existing spatial posttraining. Beyond benchmark-level gains, dense trajectory evaluation in Section 4.4 further shows that $\mathrm { U \bar { R } U Q I _ { S y n } \mathrm { - } 8 B }$ improves self-motion estimation and maintenance of previously observed object locations as the observer moves.

## 2 RELATED WORK

Spatial Intelligence in Vision-Language Models. Recent work has improved spatial reasoning in VLMs by scaling spatially grounded instruction data for metric and relational reasoning (Chen et al., 2024; Cai et al., 2026), or by introducing stronger geometric priors such as depth and reconstructed 3D features (Cheng et al., 2024; Wu et al., 2026; Fan et al., 2026). Another line of work delegates geometric operations to specialized tools, with the VLM serving primarily as a semantic reasoner and planner (Dai et al., 2026; Chen et al., 2026a). While these approaches demonstrate the value of richer supervision, geometric representations, and external computation, the resulting progress motivates a fundamental question: can spatial intelligence be learned as a native capability of VLMs, enabling them to build and update spatial understanding from visual experience? We study this question by using privileged geometry to supervise how spatial information evolves during training.

Intermediate Supervision for Spatial Reasoning Recent work has complemented final-answer supervision by explicitly supervising intermediate spatial reasoning processes. Spatial CoT supervises linguistic reasoning traces (Li et al., 2026b). Cognitive-map methods construct explicit spatial scaffolds for subsequent reasoning or verification (Wang et al., 2025; Deng et al., 2026). Representation-based approaches instead supervise intermediate visual or geometric states, such as canonical views or latent 3D representations (Zhan et al., 2026; Chen et al., 2026b). These approaches demonstrate the benefit of exposing intermediate structure during spatial reasoning. However, their intermediate targets are typically defined by a particular reasoning format or spatial representation. URUQI instead derives supervision directly from scene geometry, capturing observer motion and spatial information updates as shared components of spatial reasoning.

Spatial Reasoning from Egocentric Visual Experience Recent work has extended spatial reasoning from images to egocentric videos, studying how models integrate spatial evidence across changing viewpoints and long temporal contexts (Yang et al., 2025; Ravi et al., 2025; Yang et al., 2026b). A growing line of work further maintains persistent spatial information through objectcentric memories, learned spatial memories, or explicit 3D scene representations (Fan et al., 2025; Liu et al., 2026). From a complementary cognitive perspective, self-motion supports continuous spatial updating without requiring complete scene reconstruction (Wang & Spelke, 2000; Wolbers et al., 2008). Motivated by human spatial cognition, URUQI instead learns to maintain spatial understanding across observer motion, supporting persistent object mapping and subsequent spatial reasoning without complete scene reconstruction.

## 3 METHODOLOGY

In this section, we first formulate spatial supervision over visual experience in Section 3.1. We then introduce motif-driven trajectory rendering for generating visual experiences in Section 3.2. Finally, Section 3.3 describes how privileged scene geometry is used to derive spatial supervision from visual experiences and organize resulting targets into training episodes with shared context.

![](images/a0bed5bf2748bcf0ca99cd578023c3d67a79723bf6fd5de028a84e822f00f33a.jpg)  
Figure 2: Overview of URUQI. Top: Visual experiences are generated by instantiating reusable motifs in 3D scenes, searching for feasible camera trajectories, and rendering RGB observations together with privileged depth and instance identities. Bottom: Each verified trajectory is compiled into supervision over three stages: self-motion tracking (M1), persistent object mapping (M2), and operations over spatial state (M3). The example illustrates spatial information evolving along a shared trajectory, with solid and dashed markers denoting visible and previously observed objects.

## 3.1 PROBLEM FORMULATION

We consider a static physical world W observed along a moving camera trajectory $\tau =$ $( T _ { 1 } , \dots , T _ { L } )$ , where $T _ { t } \in \mathrm { S E } ( 3 )$ denotes the camera pose at step t. The trajectory yields an RGB sequence $I _ { 1 : L }$ , which constitutes the observer’s visual experience. The VLM observes only $I _ { 1 : L } .$ while the world geometry W and camera poses $T _ { 1 : L }$ are privileged information used only to con struct supervision.

For two consecutive observations, the relative observer motion is given by:

$$
\Delta T _ { t } = T _ { t } ^ { - 1 } T _ { t + 1 } \in \mathrm { S E } ( 3 ) .\tag{1}
$$

This motion changes the observer’s reference frame and therefore constrains how previously acquired spatial information should be interpreted at the next viewpoint. Let $S _ { t }$ denote the supervisionlevel spatial state supported by observations up to step t. Upon receiving $I _ { t + 1 }$ , the state must be updated consistently with both the relative motion and the new visual evidence:

$$
\begin{array} { r } { S _ { t } \xrightarrow { \Delta T _ { t } , I _ { t + 1 } } S _ { t + 1 } . } \end{array}\tag{2}
$$

Here, $S _ { t }$ is an abstraction for supervision and does not assume that the VLM maintains an explicit map or a particular internal state representation. $\mathbf { A }$ spatial query q issued at step $t _ { q }$ specifies how the available spatial information should be used:

$$
y _ { q } = g _ { q } ( S _ { t _ { q } } ) ,\tag{3}
$$

where $g _ { q }$ denotes the query-dependent spatial operation and $y _ { q }$ is its target answer. This factorization exposes three supervision targets: observer motion $\Delta T _ { t }$ , spatial-state transitions $S _ { t }  S _ { t + 1 }$ , and query-dependent operations over the resulting state.

## 3.2 MOTIF-DRIVEN VISUAL EXPERIENCE GENERATION

The same physical world can present distinct visual experiences depending on how it is observed. Changes in visual experience reveal different geometric evidence and place distinct demands on spatial reasoning. For example, rotating in place changes the observer’s orientation while keeping its position fixed, isolating changes in the reference frame. Exploring and revisiting a scene requires spatial information to persist across viewpoints and temporary loss of visibility. We aim to design a broad set of visual experiences to evaluate spatial reasoning under different conditions.

As shown in the top of Figure 2, we control the generation of visual experience through spatial motifs: reusable patterns of observer motion and visibility that determine how spatial evidence unfolds along a trajectory. Given a 3D scene, we instantiate a motif by selecting the relevant objects and searching for a feasible trajectory under the scene geometry. Each motif then defines how the observer should move and how spatial information should be revealed, such as rotating at a fixed location, moving until a previously seen object becomes occluded, or walking through landmarks.

The motif is instantiated through constrained trajectory search over the scene geometry, jointly determining camera positions, motion paths, and viewing directions. Detailed motif definitions and the corresponding search procedures are provided in Appendix B.1. Each candidate trajectory is then executed in OmniGibson (Li et al., 2022) to produce the RGB sequence together with privileged geometric signals, including camera poses, depth, and instance identities. We then verify that these observations and signals satisfy the motion, visibility, and object-identity requirements of the motif.

Using this procedure, we construct 11,738 visual trajectories, covering a broad range of ways in which spatial information can emerge, disappear, and reconnect over time. Detailed trajectory statistics are provided in Appendix B.3.

## 3.3 COMPILING SPATIAL SUPERVISION INTO EPISODES

As shown in the bottom of Figure 2, for each visual trajectory, we use its privileged camera poses, object identities, and visibility annotations to instantiate the spatial supervision defined in Section 3.1. We first derive supervision targets from geometric quantities and then express them as natural-language QA pairs. These targets are subsequently placed along the trajectory according to the visual evidence available at each point to form a temporally grounded training episode.

Geometric supervision. The trajectory geometry provides three complementary forms of supervision. For self-motion tracking (M1), relative camera poses determine the translation and heading change between viewpoints, including both consecutive and longer-range motion along the trajectory. For persistent object mapping (M2), object identities associate entities across observations, while object geometry and camera poses determine their locations and relations in the reference frame required by each query. These targets cover both currently visible objects and previously observed objects that have moved outside the current field of view. For operations over state (M3), the query-dependent operator $g _ { q }$ acts on the spatial information represented by $S _ { t }$ to produce targets such as reference-frame transformations, hypothetical motion updates, metric or directional comparisons, and compositions of information acquired across observations.

Temporal grounding. We align each supervision target with the point in the trajectory at which its supporting visual evidence becomes available. At step t, a target may involve only entities and spatial information established by observations $I _ { 1 : t }$ . Once an entity has been observed, its spatial information remains available for subsequent targets as the observer moves, including after the entity becomes occluded or leaves the current view. Entities not yet encountered are introduced only afte their first supporting observation. Accordingly, self-motion targets follow the observations defining the corresponding relative pose, object-mapping targets follow the observations establishing the relevant entities and spatial information, and operations targets are introduced once their required spatial information and reference frame are available.

Episode construction. Let $t _ { 1 } < \cdots < t _ { K }$ denote the points at which the K supervision queries are inserted. We serialize the trajectory and its associated question–answer pairs into a single episode:

$$
\mathcal { E } = \left( I _ { 1 : t _ { 1 } } , q _ { 1 } , y _ { 1 } ; \ I _ { t _ { 1 } + 1 : t _ { 2 } } , q _ { 2 } , y _ { 2 } ; \ . . . ; \ I _ { t _ { K - 1 } + 1 : t _ { K } } , q _ { K } , y _ { K } \right) .
$$

Observations retain their temporal order, while the context accumulates as the episode progresses. Thus, when answering $q _ { k } .$ , the model has access to preceding visual observations and earlier QA exchanges. M1, M2, and M3 targets are interleaved along the visual trajectory, placing different forms of spatial supervision within the same evolving context.

Table 1: Evaluation on URUQI benchmark. Each column reports task-specific accuracy (%). Subclasses are grouped by the three supervision stages. Overall is computed as the mean of the three stage scores, with subclasses weighted equally within each stage. Bold and underline denote the best and second-best scores, respectively.
<table><tr><td rowspan="2">Model</td><td colspan="3">M1: self-motion</td><td colspan="4">M2: object mapping</td><td colspan="4">M3: state operations</td><td rowspan="2">Overall ↑</td></tr><tr><td>Mot.</td><td>Loc.</td><td>Yaw</td><td>ID</td><td>Pos.</td><td>Hist.</td><td>Inv.</td><td>Ref.</td><td>Upd.</td><td>Rel.</td><td>Evt.</td></tr><tr><td>Proprietary models</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Gemini 3.1 Pro</td><td>13.64</td><td>17.96</td><td>43.58</td><td>29.41</td><td></td><td>11.8425.15</td><td>48.65</td><td>25.40</td><td>34.82</td><td>27.37</td><td>25.56</td><td>27.37</td></tr><tr><td>Qwen3.8-Max</td><td>9.77</td><td>16.45</td><td>39.57</td><td>32.10</td><td>12.85</td><td>24.08</td><td>47.27</td><td>14.75</td><td>20.01</td><td>24.58</td><td>19.60</td><td>23.58</td></tr><tr><td>GPT-5.5</td><td>11.86</td><td>21.16</td><td>44.53</td><td>31.22</td><td>11.36</td><td>20.18</td><td>44.20</td><td>22.43</td><td>19.04</td><td>21.30</td><td>20.90</td><td>24.50</td></tr><tr><td>GPT-6 Astra</td><td>25.31</td><td>41.30</td><td>89.70</td><td>52.53</td><td>29.72</td><td>54.91</td><td>61.38</td><td>47.13</td><td>63.25</td><td>45.89</td><td>37.68</td><td>50.08</td></tr><tr><td colspan="11">General-purpose open-source models</td></tr><tr><td>InternVL3-8B</td><td>5.47</td><td></td><td>22.4029.18</td><td>8.08</td><td>9.71</td><td>7.03</td><td>26.66</td><td>12.61</td><td>11.18</td><td>21.10</td><td>17.62</td><td>15.84</td></tr><tr><td>Qwen3-VL-8B</td><td>7.49</td><td>17.60</td><td>30.17</td><td>13.02</td><td>11.01</td><td>10.92</td><td>35.89</td><td>16.17</td><td>11.55</td><td>23.75</td><td>27.32</td><td>18.61</td></tr><tr><td>Qwen3.8-27B</td><td>7.80</td><td>13.48</td><td>33.75</td><td>19.83</td><td>9.37</td><td>15.01</td><td>37.76</td><td>9.15</td><td>14.90</td><td>16.53</td><td>7.30</td><td>16.94</td></tr><tr><td colspan="11">Spatially specialized open-source models</td></tr><tr><td>SenseNova-SI-1.5</td><td>2.09</td><td></td><td>12.9935.34</td><td>6.81</td><td>10.82</td><td>10.87</td><td>24.14</td><td>22.92</td><td>11.25</td><td>24.71</td><td>25.47</td><td>17.02</td></tr><tr><td>Spatial-MLLM</td><td>0.00</td><td>1.50</td><td>6.85</td><td>0.09</td><td>2.93</td><td>1.51</td><td>3.93</td><td>5.99</td><td>4.94</td><td>11.13</td><td>18.25</td><td>4.99</td></tr><tr><td>SpaceR-7B</td><td>0.18</td><td>6.95</td><td>30.47</td><td>3.36</td><td>10.23</td><td>6.56</td><td>21.38</td><td>12.67</td><td>10.26</td><td>17.86</td><td>31.18</td><td>13.64</td></tr><tr><td>VST-7B-SFT</td><td>0.00</td><td>5.19</td><td>28.02</td><td>2.46</td><td>9.79</td><td>10.14</td><td>13.40</td><td>16.35</td><td>10.48</td><td>17.86</td><td>31.14</td><td>12.99</td></tr><tr><td>MindCube-RL</td><td>0.00</td><td>4.93</td><td>29.44</td><td>2.00</td><td>8.17</td><td>5.17</td><td>10.80</td><td>10.75</td><td>10.19</td><td>9.47</td><td>22.03</td><td>10.37</td></tr><tr><td>URUQI family</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td> $\mathrm { U R U Q I } _ { \mathrm { S y n } ^ { - } } 8 \mathrm { B }$ </td><td>33.48</td><td>54.91</td><td>71.08</td><td>24.07</td><td>28.03</td><td></td><td>37.68 52.82</td><td></td><td>59.49</td><td>44.08 41.07</td><td>72.93</td><td>47.73</td></tr><tr><td>URUQI-Mix-8B</td><td>33.05</td><td>54.88</td><td>69.27</td><td>25.49</td><td>27.85</td><td>39.05</td><td>53.19</td><td>60.19</td><td>43.56</td><td>44.60</td><td>76.63</td><td>48.35</td></tr><tr><td>URUQI-SI-Mix-8B</td><td>30.10</td><td>56.45</td><td>71.68</td><td>25.84</td><td>30.27</td><td>42.13</td><td>51.38</td><td>66.31</td><td>47.25</td><td>46.47</td><td>84.31</td><td>50.41</td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr></table>

Training objective. We train URUQI models using autoregressive next-token prediction on answer tokens. Let i index the N QA occurrences across all training episodes. For QA i, the context $H _ { i }$ contains the available images, the current question, and any preceding QA exchanges in the same episode. Its target answer is the token sequence $y _ { i } = ( y _ { i , 1 } , \dotsc , y _ { i , n _ { i } } )$ , where $n _ { i }$ is the number of answer tokens. The training objective is:

$$
\mathcal { L } ( \theta ) = - \frac { 1 } { N } \sum _ { i = 1 } ^ { N } \frac { 1 } { n _ { i } } \sum _ { j = 1 } ^ { n _ { i } } \log p _ { \theta } \big ( y _ { i , j } \mid H _ { i } , y _ { i , < j } \big ) ,\tag{4}
$$

where $y _ { i , < j }$ denotes the target answer tokens preceding position $j .$

## 4 EXPERIMENTS

In this section, we conduct experiments to address the following research questions:

• RQ1: Does post-training on continuous visual experience improve the underlying spatial capabilities targeted by our supervision?

• RQ2: Do these gains transfer to external spatial benchmarks and real-world visual observations?

• RQ3: Can VLMs maintain egocentric spatial understanding as the observer moves?

## 4.1 EXPERIMENTAL SETUP

Training data and evaluation data. The URUQI training corpus contains 611,348 QA instances in 455,863 episodes: 211,552 for self-motion tracking, 136,398 for persistent object mapping, and 263,398 for spatial-state operations. The benchmark contains 52,920 QA instances in 2,692 episodes from eight held-out scenes (Appendix B.3).

Table 2: External spatial evaluation. Num and MC denote numerical and multiple-choice tasks in VSI-Bench; Pos and MSR denote positional and multi-step spatial reasoning in MMSI-Bench; Rot, Amg, and Ard denote rotation, among, and around in MindCube-tiny. Avg reports each benchmark’s overall score. Bold and underline indicate the best and second-best scores among open-source models, respectively. Superscripts indicate externally reported results: <sup>a</sup>SpatialAxiom (Lou et al., 2026); <sup>b</sup>Wang (Wang, 2026); <sup>c</sup>PhysBrain 1.5 (Team et al., 2026) .
<table><tr><td rowspan="2">Model</td><td colspan="3">VSI-Bench</td><td colspan="3">MMSI-Bench</td><td colspan="4">MindCube-tiny</td></tr><tr><td>Num</td><td>MC</td><td>Avg</td><td>Pos.</td><td>MSR</td><td>Avg</td><td>Rot</td><td>Amg</td><td>Ard</td><td>Avg</td></tr><tr><td>Proprietary models (reference)</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Gemini 3.1 Pro</td><td>38.48</td><td>61.36</td><td>49.92</td><td></td><td></td><td>49.50ª</td><td>90.50</td><td>71.33</td><td>83.20</td><td>77.81</td></tr><tr><td>Qwen3.7-Plus</td><td>61.56</td><td>68.51</td><td>65.04</td><td></td><td></td><td>45.00a</td><td>92.50</td><td>59.83</td><td>80.40</td><td>70.95</td></tr><tr><td>GPT-5.5</td><td></td><td></td><td>60.40ª</td><td></td><td></td><td>42.20ª</td><td></td><td></td><td></td><td>65.50ª</td></tr><tr><td>GPT-6 Astra</td><td>64.57</td><td>81.53</td><td>73.05b</td><td></td><td></td><td>57.90c</td><td></td><td></td><td></td><td>78.80c</td></tr><tr><td>General-purpose open-source models</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>InternVL3-8B</td><td>48.02</td><td>36.26</td><td>42.14</td><td>30.84</td><td>21.72</td><td>27.80</td><td>33.00</td><td>34.33</td><td>51.60</td><td>38.19</td></tr><tr><td>Qwen3-VL-8B</td><td>63.19</td><td>53.32</td><td>58.25</td><td>30.84</td><td>28.28</td><td>29.40</td><td>29.50</td><td>28.33</td><td>32.00</td><td>29.43</td></tr><tr><td>Qwen3.8-27B</td><td>48.40</td><td>53.10</td><td>50.75</td><td>40.80</td><td>37.88</td><td>42.70</td><td>92.50</td><td>67.33</td><td>83.60</td><td>76.00</td></tr><tr><td colspan="9">Spatially specialized open-source models</td><td></td></tr><tr><td>SenseNova-SI-1.5</td><td>65.85</td><td>68.57</td><td>67.21</td><td>45.40</td><td>28.28</td><td>39.10</td><td>90.50</td><td>93.83</td><td>88.80</td><td>92.00</td></tr><tr><td>Spatial-MLLM</td><td>50.96</td><td>41.49</td><td>46.33</td><td>27.16</td><td>29.80</td><td>26.10</td><td>39.00</td><td>30.51</td><td>36.00</td><td>33.46</td></tr><tr><td>SpaceR-7B</td><td>49.85</td><td>41.55</td><td>45.60</td><td>35.29</td><td>23.23</td><td>27.80</td><td>34.50</td><td>31.02</td><td>33.60</td><td>32.31</td></tr><tr><td>VST-7B-SFT</td><td>60.72</td><td>50.28</td><td>55.50</td><td>35.35</td><td>18.18</td><td>32.50</td><td>37.00</td><td>35.93</td><td>50.80</td><td>39.71</td></tr><tr><td>VST-7B-RL</td><td>61.21</td><td>51.10</td><td>56.15</td><td>36.13</td><td>19.19</td><td>32.50</td><td>37.00</td><td>37.80</td><td>45.60</td><td>39.52</td></tr><tr><td>MindCube-3B-SFT</td><td>15.87</td><td>18.62</td><td>17.24</td><td>2.15</td><td>2.02</td><td>1.70</td><td>34.00</td><td>51.02</td><td>67.60</td><td>51.73</td></tr><tr><td>MindCube-RL</td><td>25.85</td><td>37.09</td><td>31.47</td><td>27.39</td><td>27.78</td><td>27.70</td><td>30.50</td><td>51.17</td><td>61.60</td><td>49.71</td></tr><tr><td>Cambrian-S-7B</td><td>63.52</td><td>62.34</td><td>62.92</td><td>29.56</td><td>24.24</td><td>27.10</td><td>33.00</td><td>38.98</td><td>39.20</td><td>37.88</td></tr><tr><td colspan="9">URUQI family</td><td></td><td></td></tr><tr><td>URUQISyn-8B</td><td>39.28</td><td>55.23</td><td>47.26</td><td>34.67</td><td>22.22</td><td>33.10</td><td>44.00</td><td>50.33</td><td>36.80</td><td>45.90</td></tr><tr><td>URUQI-Mix-8B</td><td>55.38</td><td>59.73</td><td>57.56</td><td>45.02</td><td>21.72</td><td>39.80</td><td>49.00</td><td>57.83</td><td>57.20</td><td>56.00</td></tr><tr><td>URUQI-SI-Mix-8B</td><td>66.39</td><td>69.12</td><td>67.76</td><td>50.19</td><td>28.79</td><td>41.80</td><td>90.00</td><td>93.83</td><td>91.60</td><td>92.57</td></tr></table>

Training setting. We post-train all models in the URUQI family for one epoch on 16 NVIDIA H800 GPUs using AdamW with a learning rate of 10<sup>−5</sup> and an effective batch size of 192 training episodes. We consider three URUQI-8B configurations.

• URUQI<sub>Syn</sub>-8B: initialized from InternVL3-8B (Zhu et al., 2025) and post-trained only on our synthesized visual-experience supervision.

• URUQI-Mix-8B: initialized from InternVL3-8B and trained on a mixture of our visualexperience supervision and 100k examples randomly selected from SenseNova-SI-8M (Cai et al., 2026).

• URUQI-SI-Mix-8B: initialized from SenseNova-SI-1.5-InternVL3-8B (Cai et al., 2026) and further trained with the same mixed-data recipe as URUQI-MIX-8B.

Evaluation setting We evaluate spatial capabilities on URUQI benchmark in a multi-turn setting in Table 1. Observations and questions are presented in temporal order, and the model retains preceding observations and its own answers within each episode. We further evaluate models on VSI-Bench (Yang et al., 2025), MMSI-Bench (Yang et al., 2026c), and MindCube-tiny (Wang et al., 2025) using their respective evaluation protocols in Table 2. We compare URUQI model family against proprietary VLMs including Gemini 3.1 Pro (Google DeepMind, 2026), Qwen3.7- Plus (Qwen Team, 2026a), Qwen3.8-Max (Qwen Team, 2026c), GPT-5.5 (OpenAI, 2026a), and GPT-6 Astra (OpenAI, 2026b), general-purpose open-source VLMs including InternVL3-8B (Zhu et al., 2025), Qwen3-VL-8B (Bai et al., 2025), and Qwen3.8-27B (Qwen Team, 2026b); and spatially specialized models including SenseNova-SI-1.5 (Cai et al., 2026), Spatial-MLLM (Wu et al., 2026), SpaceR-7B (Ouyang et al., 2025), VST-7B-SFT and VST-7B-RL (Yang et al., 2026a), MindCube-3B-SFT and MindCube-RL (Wang et al., 2025), and Cambrian-S-7B (Yang et al., 2026b).

Table 3: Dense evaluation of self-motion estimation and egocentric object localization. Acc@0.5m measures the percentage of object predictions within 0.5 m of the ground-truth horizontal position. Bold and underline denote the best and second-best results per column, respectively.
<table><tr><td rowspan="2">Model</td><td rowspan="2">QA history</td><td colspan="2">Self-motion</td><td rowspan="2"></td><td colspan="4">Object mapping</td><td rowspan="2">Mean error (m) ↓</td></tr><tr><td>Translation error (m) ↓</td><td>Heading error (°) ↓</td><td></td><td>Acc@0.5m (%) ↑</td><td></td></tr><tr><td rowspan="3">InternVL3-8B</td><td></td><td></td><td></td><td>All</td><td>Initial</td><td>Visible</td><td>Absent</td><td></td><td></td></tr><tr><td>None Model</td><td>0.397 0.341</td><td>93.57 16.57</td><td>1.00 0.67</td><td>0.00 0.00</td><td>2.42 1.45</td><td></td><td>0.00</td><td>3.004 3.040</td></tr><tr><td>GT motion</td><td>0.189</td><td>12.72</td><td>0.78</td><td>0.00</td><td>1.82</td><td></td><td>0.13 0.09</td><td>3.052</td></tr><tr><td rowspan="3">Qwen3.8-27B</td><td>None</td><td></td><td>9.43</td><td></td><td></td><td></td><td>12.30</td><td></td><td>4.725</td></tr><tr><td>Model</td><td>0.297 0.356</td><td>7.22</td><td>6.44 9.41</td><td>16.02 22.08</td><td></td><td>17.99</td><td>1.29 1.88</td><td>3.594</td></tr><tr><td>GT motion</td><td>0.127</td><td>4.84</td><td>8.75</td><td>23.81</td><td></td><td>16.41</td><td>1.79</td><td>3.263</td></tr><tr><td rowspan="3"> $\mathrm { U R U Q I _ { S y n } } – 8 \mathbf { B }$ </td><td>None</td><td>0.154</td><td>2.36</td><td>49.56</td><td>75.32</td><td></td><td>66.14</td><td>35.13</td><td>0.743</td></tr><tr><td>Model</td><td>0.090</td><td>0.97</td><td>39.62</td><td>74.89</td><td></td><td>47.24</td><td>30.94</td><td>1.027</td></tr><tr><td>GT motion</td><td>0.031</td><td>0.56</td><td>47.04</td><td>73.16</td><td></td><td>55.97</td><td>38.30</td><td>0.832</td></tr></table>

## 4.2 EVALUATING SPATIAL CAPABILITIES IN CONTINUOUS VISUAL EXPERIENCE (RQ1)

Metrics. Table 1 reports three groups of task-specific accuracies. M1 measures inter-view translation and rotation (Mot.), relative observer localization (Loc.), and heading changes (Yaw). M2 measures cross-view correspondence and referent resolution (ID), object positions, distances, and directions (Pos.), historical spatial states and visibility (Hist.), and distinct-object counts across views (Inv.). M3 measures reference-frame transformations (Ref.), hypothetical and composed motion up dates (Upd.), metric relations and comparisons (Rel.), and counts of observed turns (Evt.). Overall averages the three stage scores, each computed as the mean of its subclasses.

Dense experience supervision improves the targeted capabilities. Table 1 evaluates self-motion estimation (M1), object mapping (M2), and operations over spatial state (M3). Post-training InternVL3-8B solely on our synthetic supervision raises its Overall score from 15.84% to 47.73%, with improvements in all 11 subclasses.

Supervision from experience complements existing spatial post-training. URUQI-Mix-8B reaches 48.35% Overall, compared with 47.73% for $\mathrm { U \bar { R } U \bar { Q } I _ { S y n } \bar { - } \tilde { 8 } B }$ . Starting from SenseNova-SI-1.5, URUQI-SI-Mix-8B increases Overall from 17.02% to 50.41%, improving all 11 subclasses. It also obtains the highest Overall score among the evaluated models, closely followed by GPT-6 Astra at 50.08%. GPT-6 Astra retains a higher M2 average (49.64% vs 37.41%).

## 4.3 EVALUATING TRANSFER TO EXTERNAL SPATIAL BENCHMARKS (RQ2)

Benchmarks and metrics. We evaluate transfer on three external spatial benchmarks: VSI-Bench (Yang et al., 2025), MMSI-Bench (Yang et al., 2026c), and MindCube-tiny (Wang et al., 2025). Table 2 reports their overall scores together with task-group results.

Synthetic supervision transfers to external spatial tasks. Compared with InternVL3-8B, $\mathrm { U R U Q I _ { S y n } } – 8 \mathbf { B }$ improves the overall scores from 42.14% to 47.26% on VSI-Bench, from 27.80% to 33.10% on MMSI-Bench, and from 38.19% to 45.90% on MindCube-tiny. These gains show that the benefits of our synthetic supervision extend beyond the tasks in URUQI benchmark.

Combining synthetic and existing spatial data improves transfer. With the addition of 100K examples from SenseNova-SI-8M (1.25% of the corpus), URUQI-Mix-8B reaches 57.56%, 39.80%, and 56.00% on VSI-Bench, MMSI-Bench, and MindCube-tiny, respectively, improving over the synthetic-only variant on all three benchmarks. VSI-Bench Num also increases to 55.38%, exceeding the InternVL3-8B baseline. Starting from SenseNova-SI-1.5, URUQI-SI-Mix-8B achieves the highest overall scores on VSI-Bench and MindCube-tiny among the open-source models in Table 2, while ranking second on MMSI-Bench, behind only Qwen3.8-27B.

Table 4: Localization from visual estimates and camera poses. Acc@0.5m (%) is averaged over 2,240 absent-object queries. (a) Direct prediction uses each current VLM answer; pose-based propagation stores the first accepted object estimate and computes subsequent locations from camera poses. Both use No QA history, with GT poses supplied only to the geometry module. (b) Posebased propagation uses Model-history object predictions; Predicted denotes poses integrated from the model’s motion estimates. Bold marks the best result within each model and panel. Complete metrics are in Table 7.  
(a) Localization method  
No QA history; GT poses  
(b) Camera-pose source
<table><tr><td>Model</td><td>Method</td><td>Absent Acc@0.5m ↑</td></tr><tr><td rowspan="2">Qwen3.8-27B</td><td>Direct prediction</td><td>1.29</td></tr><tr><td>Pose-based propagation</td><td>16.88</td></tr><tr><td rowspan="2"> $\mathrm { U R U Q I _ { S y n } } – 8 \mathbf { B }$ </td><td>Direct prediction</td><td>35.13</td></tr><tr><td>Pose-based propagation</td><td>71.61</td></tr></table>

Model history; pose-based propagation
<table><tr><td>Model</td><td>Pose</td><td>Absent Acc@0.5m ↑</td></tr><tr><td rowspan="2">Qwen3.8-27B</td><td>GT</td><td>18.48</td></tr><tr><td>Predicted</td><td>3.97</td></tr><tr><td rowspan="2"> $\mathrm { U R U Q I _ { S y n } } – 8 \mathbf { B }$ </td><td>GT</td><td>72.86</td></tr><tr><td>Predicted</td><td>34.02</td></tr></table>

## 4.4 DENSE EVALUATION OF EGOCENTRIC SPATIAL TRACKING (RQ3)

We evaluate self-motion estimation and egocentric object localization on 100 trajectories in URUQI benchmark, comprising 2,312 frames and 231 registered objects.

To examine whether previous answers help subsequent localization, we compare three history conditions, all of which provide the full observed image history. No QA history excludes previous questions and answers; Model history includes the model’s earlier motion and location answers; GT motion history replaces earlier motion answers with ground truth.

Table 3 reports motion errors and localization metrics, including Acc@0.5m, the percentage of predictions within 0.5 m of the true horizontal position. Construction rules, visibility stages, and scoring details are in Appendix D.

Spatial tracking from visual history. Under No QA history, $\mathrm { U R U Q I _ { S y n } } – 8 \mathbf { B }$ improves both motion estimation and object localization over InternVL3-8B and Qwen3.8-27B. Its overall localization accuracy reaches 49.56%, compared with 1.00% and 6.44%, respectively. For absent objects, it achieves 35.13%, versus 0.00% and 1.29%, showing improved localization out of view.

Effect of answer history. For $\mathrm { U R U Q I _ { S y n } }$ -8B, Model history improves motion estimation but reduces overall object-localization accuracy from 49.56% to 39.62%. GT motion history partially recovers overall object-localization accuracy and yields the highest absent-object accuracy among the three conditions (38.30%). The improvement over Model history suggests that accurate motion history provides useful context for absent-object localization.

Pose-assisted localization. We test whether an external geometry module can use camera poses to propagate each object’s first accepted location estimate. Under No QA history, supplying GT poses only to this module improves $\mathrm { U R U Q I _ { S y n } } – 8 \mathbf { B } ^ { \prime } \mathbf { s }$ absent-object accuracy from 35.13% with direct prediction to 71.61% with propagation (Table 4a).

In a separate comparison, we fix the Model history object predictions and propagation method, replacing GT poses with poses obtained by integrating model-predicted self-motion. Accuracy decreases from 72.86% to 34.02%, compared with 3.97% for Qwen3.8-27B using predicted poses (Table 4b). Thus, the initial object estimates support pose-assisted localization, but its accuracy remains sensitive to errors in the estimated camera trajectory.

## 5 CONCLUSION

We study how spatial cognition in VLMs can be learned from continuous visual experience. URUQI couples consecutive observations through observer self-motion and jointly supervises motion tracking, persistent object mapping, and spatial operations. Our 8B model achieves the best overall performance among evaluated open-source models, reaching 50.41% on URUQI benchmark and leading open-source results on two of three external benchmarks. It matches GPT-6 Astra on URUQI benchmark and surpasses several proprietary models on external benchmarks, highlighting continuous visual experience as an effective, scalable supervision paradigm for spatially capable VLMs.

## AI USE STATEMENT

Generative AI tools were used to assist with implementing the trajectory generation and spatialsupervision compilation pipeline, as well as language polishing, manuscript organization and the presentation of experimental results. All codes, references, experimental results, and content were reviewed and verified by the authors, who take responsibility for the final text, methods, and claims.

## ETHICS STATEMENT

Our spatial supervision is generated from simulated scenes and does not involve human participants or personally identifiable information. All experiments are conducted in simulated or benchmark environments, and any released data, models, and derived assets will follow applicable licensing and usage requirements. The resulting models are developed for research purposes and are not evaluated for safety-critical deployment.

## REPRODUCIBILITY STATEMENT

We provide details of data construction, model training, evaluation protocols, and implementation settings in the main text and appendix. The appendix further documents the procedures used for trajectory generation, spatial supervision, and evaluation. Where permitted, we will release the configurations, checkpoints, and evaluation pipeline.

## REFERENCES

Shuai Bai, Yuxuan Cai, Ruizhe Chen, Keqin Chen, Xionghui Chen, Zesen Cheng, Lianghao Deng, Wei Ding, Chang Gao, Chunjiang Ge, Wenbin Ge, Zhifang Guo, Qidong Huang, Jie Huang, Fei Huang, Binyuan Hui, Shutong Jiang, Zhaohai Li, Mingsheng Li, Mei Li, Kaixin Li, Zicheng Lin, Junyang Lin, Xuejing Liu, Jiawei Liu, Chenglong Liu, Yang Liu, Dayiheng Liu, Shixuan Liu, Dunjie Lu, Ruilin Luo, Chenxu Lv, Rui Men, Lingchen Meng, Xuancheng Ren, Xingzhang Ren, Sibo Song, Yuchong Sun, Jun Tang, Jianhong Tu, Jianqiang Wan, Peng Wang, Pengfei Wang, Qiuyue Wang, Yuxuan Wang, Tianbao Xie, Yiheng Xu, Haiyang Xu, Jin Xu, Zhibo Yang, Mingkun Yang, Jianxin Yang, An Yang, Bowen Yu, Fei Zhang, Hang Zhang, Xi Zhang, Bo Zheng, Humen Zhong, Jingren Zhou, Fan Zhou, Jing Zhou, Yuanzhi Zhu, and Ke Zhu. Qwen3-vl technical report, 2025. URL https://arxiv.org/abs/2511.21631.

Zhongang Cai, Ruisi Wang, Chenyang Gu, Fanyi Pu, Junxiang Xu, Yubo Wang, Wanqi Yin, Zhitao Yang, Chen Wei, Tongxi Zhou, et al. Scaling spatial intelligence with multimodal foundation models. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 7879–7890, 2026.

Boyuan Chen, Zhuo Xu, Sean Kirmani, Brain Ichter, Dorsa Sadigh, Leonidas Guibas, and Fei Xia. Spatialvlm: Endowing vision-language models with spatial reasoning capabilities. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pp. 14455–14465, 2024.

Zeren Chen, Xiaoya Lu, Zhijie Zheng, Pengrui Li, Lehan He, Yijin Zhou, Jing Shao, Bohan Zhuang, and Lu Sheng. Geometrically-constrained agent for spatial reasoning. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 38689–38699, 2026a.

Zhangquan Chen, Manyuan Zhang, Xinlei Yu, Xufang Luo, Mingze Sun, Zihao Pan, Xiang An, Yan Feng, Peng Pei, Xunliang Cai, et al. Think with 3d: Geometric imagination grounded spatial reasoning from limited views. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 2613–2624, 2026b.

An-Chieh Cheng, Hongxu Yin, Yang Fu, Qiushan Guo, Ruihan Yang, Jan Kautz, Xiaolong Wang, and Sifei Liu. Spatialrgpt: Grounded spatial reasoning in vision-language models. Advances in Neural Information Processing Systems, 37:135062–135093, 2024.

Yalun Dai, Hao Li, Shulin Tian, Runmao Yao, Yuhao Dong, Fangzhou Hong, Zhaoxi Chen, Fangfu Liu, Baoliang Tian, Dingwen Zhang, et al. S-agent: Spatial tool-use elicits reasoning for spatial intelligence. arXiv preprint arXiv:2606.20515, 2026.

Wei Deng, Xianlin Zhang, and Mengshi Qi. Active exploring like a pigeon: Reinforcing spatial reasoning via agentic vision-language models. arXiv preprint arXiv:2606.02459, 2026.

Yue Fan, Xiaojian Ma, Rongpeng Su, Jun Guo, Rujie Wu, Xi Chen, and Qing Li. Embodied videoagent: Persistent memory from egocentric videos and embodied sensors enables dynamic scene understanding. In 2025 IEEE/CVF International Conference on Computer Vision (ICCV), pp. 6342–6352. IEEE, 2025.

Zhiwen Fan, Jian Zhang, Renjie Li, Junge Zhang, Runjin Chen, Hezhen Hu, Kevin Wang, Peihao Wang, Huaizhi Qu, Shijie Zhou, et al. Vlm-3r: Vision-language models augmented with instruction-aligned 3d reconstruction. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 31054–31065, 2026.

Google DeepMind. Gemini 3.1 pro. Model Card, February 2026. URL https://deepmind. google/models/model-cards/gemini-3-1-pro/. Published February 19, 2026.

Jiacheng Hua, Yishu Yin, Yuhang Wu, Tai Wang, Yifei Huang, and Miao Liu. Unleashing spatial reasoning in multimodal large language models via textual representation guided reasoning. In Proceedings of the 64th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pp. 13616–13637, 2026.

Dongfu Jiang, Xuan He, Huaye Zeng, Cong Wei, Max Ku, Qian Liu, and Wenhu Chen. Mantis: Interleaved multi-image instruction tuning. arXiv preprint arXiv:2405.01483, 2024.

Sai Srinivas Kancheti, Aditya Sanjiv Kanade, Vineeth N Balasubramanian, and Tanuja Ganu. Chainof-thought degrades visual spatial reasoning capabilities of multimodal llms. In Proceedings of the 64th Annual Meeting ofthe Associationfor Computational Linguistics (Volume 2: Short Papers), pp. 862–876, 2026.

Chengshu Li, Ruohan Zhang, Josiah Wong, Cem Gokmen, Sanjana Srivastava, Roberto Mart´ın-Mart´ın, Chen Wang, Gabrael Levine, Michael Lingelbach, Jiankai Sun, Mona Anvari, Minjune Hwang, Manasi Sharma, Arman Aydin, Dhruva Bansal, Samuel Hunter, Kyu-Young Kim, Alan Lou, Caleb R Matthews, Ivan Villa-Renteria, Jerry Huayang Tang, Claire Tang, Fei Xia, Silvio Savarese, Hyowon Gweon, Karen Liu, Jiajun Wu, and Li Fei-Fei. BEHAVIOR-1k: A benchmark for embodied AI with 1,000 everyday activities and realistic simulation. In 6th Annual Conference on Robot Learning, 2022. URL https://openreview.net/forum?id=\_8DoIe8G3t.

Hongxing Li, Dingming Li, Zixuan Wang, Yuchen Yan, Hang Wu, Wenqi Zhang, Yongliang Shen, Weiming Lu, Jun Xiao, and Yueting Zhuang. Spatialladder: Progressive training for spatial reasoning in vision-language models. In International Conference on Learning Representations, volume 2026, pp. 76566–76592, 2026a.

Zongzhao Li, Zongyang Ma, Mingze Li, Songyou Li, Yu Rong, Tingyang Xu, Ziqi Zhang, Deli Zhao, and Wenbing Huang. Star-r1: Multi-view spatial transformation reasoning by reinforcing multimodal llms. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 12041–12051, 2026b.

Fangfu Liu, Diankun Wu, Jiawei Chi, Yimo Cai, Yi-Hsin Hung, Xumin Yu, Hao Li, Han Hu, Yongming Rao, and Yueqi Duan. Spatial-ttt: Streaming visual-based spatial intelligence with test-time training. In European Conference on Computer Vision, pp. 339–357. Springer, 2026.

Yuecheng Liu, Dafeng Chi, Shiguang Wu, Zhanguang Zhang, Yaochen Hu, Lingfeng Zhang, Yingxue Zhang, Shuang Wu, Tongtong Cao, Guowei Huang, Helong Huang, Guangjian Tian, Weichao Qiu, Quan, Jianye Hao, and Yuzheng Zhuang. Spatialcot: Advancing spatial reasoning through coordinate alignment and chain-of-thought for embodied task planning. In arxiv 2025, 2025.

Yujing Lou, Pingyi Chen, Shen Cao, Jiaqi Gu, Jinhui Guo, Jintao Tong, Yunzhuo Hao, Yao Liu, Yue Wu, Lubin Fan, and Jieping Ye. Spatialaxiom: An open spatial intelligence model for general spatial reasoning, July 2026. URL https://d2i-ai.github.io/SpatialAxiom.

Damiano Marsili, Rohun Agrawal, Yisong Yue, and Georgia Gkioxari. Visual agentic ai for spatial reasoning with a dynamic api. In 2025 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pp. 19446–19455. IEEE, 2025.

OpenAI. GPT-5.5 System Card. https://openai.com/index/ gpt-5-5-system-card/, 2026a.

OpenAI. GPT-6 Astra system card. System Card, September 2026b. URL https:// deploymentsafety.openai.com/gpt-6-astra. Published September 3, 2026.

Kun Ouyang, Yuanxin Liu, Haoning Wu, Yi Liu, Hao Zhou, Jie Zhou, Fandong Meng, and Xu Sun. Spacer: Reinforcing mllms in video spatial reasoning. arXiv preprint arXiv:2504.01805, 2025.

Qwen Team. Qwen3.7-plus. Model Card, 2026a. URL https://www.qwencloud.com/ models/qwen3.7-plus.

Qwen Team. Qwen3.8-27B. Hugging Face Model Card, August 2026b. URL https:// huggingface.co/Qwen/Qwen3.8-27B. Released August 14, 2026.

Qwen Team. Qwen3.8-Max: A new bar for coding and cowork, August 2026c. URL https: //qwen.ai/blog?id=qwen3.8.

Sahithya Ravi, Gabriel Herbert Sarch, Vibhav Vineet, Andrew D Wilson, and Balasaravanan Thoravi Kumaravel. Out of sight, not out of context? egocentric spatial reasoning in vlms across disjoint frames. In Proceedings of the 2025 Conference on Empirical Methods in Natural Language Processing, pp. 16146–16161, 2025.

DeepCybo Team, Yu Bin, Haipeng Cao, Zheng Chang, Kai Chen, Youning Chen, Kailin Deng, Yichao Du, Xiaotong Fu, Haoyang Ge, Yunlong Guo, Chenliu Hao, Jiyan He, Xuguo He, Yakun Hou, Kai Hu, Cong Huang, Tuopusen Huang, Yu Huang, Hong Li, Peize Li, Shijie Lian, Xiaopeng Lin, Yun Lin, Haibao Liu, Haochen Liu, Qiuzhi Liu, Shengcai Liu, Zhiqiang Liu, Tao Luo, Peng Ren, Shuo Ren, Chaoyi Ruan, Zhaolong Shen, Yukun Shi, Qiyuan Su, Yuxuan Tian, Yining Wang, Changti Wu, Hao Wu, Xueyin Xu, Ruoqi Yang, Zhaoyang Yang, Hang Yuan, Zhaoyang Zeng, Hanwen Zhang, Ruimeng Zhang, Yao Zhang, Yibo Zhang, Yuxiang Zhang, Zhirui Zhang, Ziyi Zhang, Zubin Zheng, and Zishen Zhuang. Physbrain 1.5: From vision-language models to physical foundation models, 2026. URL https://arxiv.org/abs/2609.14973.

Qineng Wang, Baiqiao Yin, Pingyue Zhang, Jianshu Zhang, Kangrui Wang, Zihan Wang, Jieyu Zhang, Keshigeyan Chandrasegaran, Han Liu, Ranjay Krishna, Saining Xie, Jiajun Wu, Li Fei-Fei, and Manling Li. Mindcube: Spatial mental modeling from limited views, 2025. URL https://arxiv.org/abs/2506.21458.

Ranxiao Frances Wang and Elizabeth S Spelke. Updating egocentric representations in human navigation. Cognition, 77(3):215–250, 2000.

Yipeng Wang. How well does astra understand real-world space? https://www.yipeng. dev/blog/astra-spatial-intelligence, September 2026. Independent evaluation. Published September 22, 2026. Accessed September 26, 2026.

Thomas Wolbers, Mary Hegarty, Christian Buchel, and Jack M Loomis. Spatial updating: how the ¨ brain keeps track of changing object locations during observer motion. Nature neuroscience, 11 (10):1223–1230, 2008.

Diankun Wu, Fangfu Liu, Yi-Hsin Hung, and Yueqi Duan. Spatial-mllm: Boosting mllm capabili ties in visual-based spatial intelligence. Advances in neural information processing systems, 38: 13569–13597, 2026.

Jihan Yang, Shusheng Yang, Anjali W Gupta, Rilyn Han, Li Fei-Fei, and Saining Xie. Thinking in space: How multimodal large language models see, remember, and recall spaces. In 2025 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pp. 10632–10643. IEEE, 2025.

Rui Yang, Ziyu Zhu, Yanwei Li, Jingjia Huang, Shen Yan, Siyuan Zhou, Zhe Liu, Xiangtai Li, Shuangye Li, Wenqian Wang, et al. Visual spatial tuning. In European Conference on Computer Vision, pp. 192–211. Springer, 2026a.

Shusheng Yang, Jihan Yang, Pinzhi Huang, Ellis Brown, Zihao Yang, Yue Yu, Shengbang Tong, Zi han Zheng, Yifan Xu, Muhan Wang, et al. Cambrian-s: Towards spatial supersensing in video. In International Conference on Learning Representations, volume 2026, pp. 78185–78225, 2026b.

Sihan Yang, Runsen Xu, Yiman Xie, Sizhe Yang, Mo Li, Jingli Lin, Chenming Zhu, Xiaochen Chen, Haodong Duan, Xiangyu Yue, et al. Mmsi-bench: A benchmark for multi-image spatial intelligence. In International Conference on Learning Representations, volume 2026, pp. 157051–157088, 2026c.

Shaoxiong Zhan, Yanlin Lai, Zheng Liu, Hai Lin, Shen Li, Xiaodong Cai, Zijian Lin, Wen Huang, and Hai-Tao Zheng. 3viewsense: Spatial and mental perspective reasoning from orthographic views in vision-language models. arXiv preprint arXiv:2603.07751, 2026.

Jinguo Zhu, Weiyun Wang, Zhe Chen, Zhaoyang Liu, Shenglong Ye, Lixin Gu, Hao Tian, Yuchen Duan, Weijie Su, Jie Shao, et al. Internvl3: Exploring advanced training and test-time recipes for open-source multimodal models. arXiv preprint arXiv:2504.10479, 2025.

t = 8 | Visible

0.5 m tolerance

(b) Egocentric object localization t = 5 | Initial t = 12 | Absent

## A QUALITATIVE EXAMPLES OF EGOCENTRIC SPATIAL TRACKING

We compare $\mathrm { U R U Q I _ { S y n } } – 8 \mathbf { B }$ , Qwen3.8-27B, and InternVL3-8B on five selected trajectories. The examples illustrate localization during visibility, absence, and reappearance, followed by matched comparisons of No QA history and Model history. Numbers report horizontal localization errors; dashed circles mark the 0.5 m threshold. GT scene geometry and poses are used only for visualiza tion. Only selected frames are shown and each prediction uses the full image history available at that step.

(a) Scene geometry and observer motion

![](images/bc194b99c34115f06d6887cbb8611081650ce7ee665400ff9fd1dc85e693573b.jpg)  
Right of initial camera (m)  
(b) Egocentric object localization

![](images/2608c625d95dfb77b5daa4ba645b4e0f250a83947c0a7a451767ee754a6447f8.jpg)  
URUQI<sub>Syn</sub>-8B: 0.22 mQwen3.8-27B: 2.25 mInternVL3-8B: 2.36 m

![](images/d0ef2ccee3b9bcd4d8c8690f8170a3930b1bdd4641816784cec4a42a99a14129.jpg)  
URUQI<sub>Syn</sub>-8B: 0.03 mQwen3.8-27B: 4.91 mInternVL3-8B: 1.30 m  
t = 10 | Visible

![](images/65b98c5d1ae9b6258a4a314acd7769bd3fa84d8576221580bf9049b030143f8e.jpg)  
URUQI<sub>Syn</sub>-8B: 0.03 mQwen3.8-27B: 4.04 mInternVL3-8B: 0.79 m

$$
\mathsf { U R U Q l } _ { \mathsf { S y n } ^ { - 8 \mathsf { B } } }
$$

![](images/ca73f924545891cb886fc28c393272d9e42f2f54ecd89487f63e48ad6baaa381.jpg)

![](images/5c0144f871e07e1b03f88b1f28c154bc4c7cf56398f2ef140cc3692bacac3d7e.jpg)

![](images/f6ff36b2245ebf9fc16817f1454794f1caef8d4916d0e89e17bd8a70a50c315f.jpg)  
Figure 3: Localization while the target remains visible. Under No QA history, $\mathrm { U R U Q I _ { S y n } } – 8 \mathbf { B }$ localizes the TV across changing viewpoints, with errors of 0.22, 0.03, and 0.03 m.  
(a) Scene geometry and observer motion

![](images/f71fb2a86af4a5347a53d2fec4554218e43b70e154539947f69353db0e4a4d0d.jpg)

![](images/21c27b99cbce20bb65db9f43f343389a30643e9889dbc32d34880210868377f1.jpg)  
URUQI<sub>Syn</sub>-8B: 0.06 mQwen3.8-27B: 3.29 mInternVL3-8B: 2.16 m

![](images/1b678f2a3d8d243cc147e3fecc2e5c992da9dd4b22ee558852ca08a11b2a8a66.jpg)  
URUQI<sub>Syn</sub>-8B: 0.08 mQwen3.8-27B: 1.62 mInternVL3-8B: 1.59 m  
t = 21 | Absent

![](images/3ae4bb2e1258e35c1053b8a75c14cb25e790554a9f105ccb332d7c3a4d2da784.jpg)  
URUQI<sub>Syn</sub>-8B: 0.05 mQwen3.8-27B: 2.08 mInternVL3-8B: 2.22 m

![](images/37a7604f76c9d5997f3681a44ca7d4500a47c0a29632948af8bfd12819904d4e.jpg)

$$
\mathsf { U R U Q l } _ { \mathsf { S y n } ^ { - 8 \mathsf { B } } }
$$

![](images/6f2d3e92b22b91e8c386308a66f02d3a3930950773d3f2c47b5d02d22abff683.jpg)

![](images/6cfd5b244309e62b6c64f5435897d7121cc2df7a5e4a2e3beb61bb32d3f7dd2d.jpg)  
Figure 4: Localization after the target leaves view. Under No QA history, the picture is absent at the two later observations; $\mathrm { U R U Q I _ { S y n } } – 8 \mathbf { B }$ maintains low errors of 0.08 and 0.05 m in this selected trajectory.

(a) Scene geometry and observer motion  
![](images/5c90258d58f76f6bb7ec28e0395d0cdd6e64e666f61d4d02b557f30a8cf3b37f.jpg)

t = 2 | Initial  
(b) Egocentric object localization No QA history  
![](images/96d22118f421a4c895219e3b1e32246dd74d921bbb1f4b84cc268a4b06ada6b6.jpg)  
URUQI<sub>Syn</sub>-8B: 0.05 mQwen3.8-27B: 1.96 mInternVL3-8B: 0.63 m

t = 4 | Absent

$$
\mathsf { U R U Q l } _ { \mathsf { S y n } ^ { - 8 \mathsf { B } } }
$$

t = 16 | Absent  
![](images/2ed499c2ea2c64c3c09cf0be091221563e9d51249bdcf8141dcb666e6261672f.jpg)

![](images/06b38667a8cd63278996836d3c5ee10be2ce4e622559ee181622c15587e1c444.jpg)  
URUQI<sub>Syn</sub>-8B: 0.06 mQwen3.8-27B: 11.00 mInternVL3-8B: 1.32 m  
URUQI<sub>Syn</sub>-8B: 0.53 mQwen3.8-27B: 9.75 mInternVL3-8B: 2.27 m

![](images/b8c16a9838e8de859a16c9d7b1b4c061b9bf60e40de83babfb53bc1b4e2e8323.jpg)  
Qwen3.8-27B

![](images/5c106c9d46f3e0762c2c758c4c0a35ebd2b6a040fa5b853f6197dcb968283f97.jpg)

![](images/735a0c38ef085f3907e698a56bd43fb283a7676352fdeeaf3e077594042fd369.jpg)

(a) Scene geometry and observer motion  
(b) Egocentric object localization Model history  
![](images/c735eeb9dc66d8ea4ae585ab39ee298c48850e098247e7f1fa78524768dbe507.jpg)

t = 2 | Initial  
![](images/d5df02d355b4f643f8f6d637b18551e02c49574ce7999fef67f1f158f71f0f35.jpg)

![](images/786143282504a995c327b86382fa8258d717a47711962ff9d7c60043bc765296.jpg)  
t = 16 | Absent  
URUQI<sub>Syn</sub>-8B: 0.05 mQwen3.8-27B: 0.97 mInternVL3-8B: 1.75 m

![](images/710dad030cfc8f42599bfea27effffb1b24aa7c163fc9338f4b0cde2e045c99d.jpg)

![](images/0f1e1316568ad4da251aea07a1e96a27f5feed342bb3199e0287afc87b9565e5.jpg)  
URUQI<sub>Syn</sub>-8B: 0.05 mQwen3.8-27B: 2.65 mInternVL3-8B: 1.84 m

![](images/b9d5e340f3d5d2cb671ad8e4c89c2a37f406a12495a5045db06bf9af0e52f850.jpg)  
URUQI<sub>Syn</sub>-8B: 1.48 m Qwen3.8-27B: 8.00 m InternVL3-8B: 2.60 m

![](images/90800892fd05157d3c2fb8afb5c6be7f5864130bfff6f449072995a7926b9d80.jpg)

$$
\mathsf { U R U Q l } _ { \mathsf { S y n } ^ { - 8 \mathsf { B } } }
$$

Figure 5: Localization error during object absence. The room divider is absent at $t = 4$ and $t = 1 6 . \ \mathrm { A t } \ t = 1 6 , \mathrm { U R U Q I _ { S y n } - 8 B }$ has an error of 0.53 m under No QA history and 1.48 m under Model history. Top: No QA history; bottom: Model history. Both conditions use the same object and observation steps, with identical coordinate limits across all six localization plots.

Qwen3.8-27B  
t = 2 | Initial  
t = 2 | Initial  
(a) Scene geometry and observer motion  
![](images/de55b86aec0e4b9015572b5fbc8b6d9abc3a66ad99c22be0c6e30d8a1aeaa89e.jpg)

(b) Egocentric object localization No QA history  
![](images/75981b768584da801a873bb1c77f176f2fcf01b3b4711d9b185e41aa8bf0ec15.jpg)  
t = 27 | Absent  
URUQI<sub>Syn</sub>-8B: 0.25 mQwen3.8-27B: 0.78 mInternVL3-8B: 2.79 m

$$
\mathsf { U R U Q l } _ { \mathsf { S y n } ^ { - 8 \mathsf { B } } }
$$

t = 32 | Reappeared  
![](images/c2d183de20243782e2b1489236c26ca61fa8db9d8c163eba29c111d6ff56f5d9.jpg)

![](images/3094eb535d9a054cf2c892b01b36c8e811ee08f4bda494cdd60c4a6f0345c8fb.jpg)  
URUQI<sub>Syn</sub>-8B: 0.24 mQwen3.8-27B: 2.13 mInternVL3-8B: 3.34 m

![](images/c88df2bcf63307ae6c95d34594be14ddb9512b9c89fa198ee2041d5036d8d423.jpg)

![](images/49168c61dea4d4d9158fa8129a7fa9b59220df53d9ae9ea0e99fc720bd01b537.jpg)

![](images/39ed19d343215aecc97eb51e9b4dfe5633e6f1fabd7bcbda6590a17afa405a4b.jpg)

(a) Scene geometry and observer motion  
(b) Egocentric object localization Model history  
![](images/9282111b73d72c3923622b213be35b6f8a7b29bbd93a18720f55ca58d9f49078.jpg)

![](images/6cb14f0fb2c2d24dcc3a40401f0ad3cfc2d573a57ba2ece2458dbbbf50dc0c75.jpg)  
t = 27 | Absent  
URUQI<sub>Syn</sub>-8B: 0.07 mQwen3.8-27B: 1.19 mInternVL3-8B: 2.90 m

![](images/3195a90388ae645a795ed48eb6753b52a79c0b4711b12ea13a00ba2b6d66a390.jpg)

t = 32 | Reappeared  
![](images/39b42185a09c0f3e097deb71a1611188757e5def116d27221d994d00ae6de24c.jpg)  
URUQI<sub>Syn</sub>-8B: 0.70 m Qwen3.8-27B: 2.03 m InternVL3-8B: 3.37 m

$$
\mathsf { U R U Q l } _ { \mathsf { S y n } ^ { - 8 \mathsf { B } } }
$$

![](images/fdfd2aeca675aba1e5d8f6700a3bf8fb13a58d056959599e8388ddac70f8ce1e.jpg)

![](images/7f597b4cb72b0ff2e793c95b650977d0b74d6c9945fc7d577ce033eec8153d75.jpg)

![](images/ae00d9d13ad4571e48835ae98515704eb26a6662ddcfa848737e285e38d273f7.jpg)

Figure 6: Localization after object reappearance. The painting is absent at $t = 2 7$ and reappears at $t = 3 2 .$ For $\mathrm { U R U Q I _ { S y n } } – 8 \mathbf { B } .$ , error decreases from 0.52 to 0.24 m under No QA history and from $0 . 8 0 \mathrm { ~ t o ~ } 0 . 7 0$ m under Model history. Top: No QA history; bottom: Model history. Both conditions use the same object and observation steps, with identical coordinate limits across all six localization plots.

(a) Scene geometry and observer motion

![](images/66b26ed3879b82ce2084a60dfd4ce0a81b811d60b589dab9ee1e164c74f26c9a.jpg)

(b) Egocentric object localization No QA history  
![](images/52db09ac60928daee6cdeb3357452d72fc289f6a5a5d7bee1bcaa373e8e0d839.jpg)  
URUQI<sub>Syn</sub>-8B: 0.19 m Qwen3.8-27B: 0.57 m InternVL3-8B: 2.20 m

$$
\mathsf { U R U Q l } _ { \mathsf { S y n } ^ { - 8 \mathsf { B } } }
$$

t = 26 | Absent

![](images/e812b65a298645ac3cc8ec12e4be61d015e19050f483528fe5c2afdeb1aaf225.jpg)  
URUQI<sub>Syn</sub>-8B: 0.53 mQwen3.8-27B: 4.67 mInternVL3-8B: 1.63 m

![](images/abd31908d674e0bb1268be4566c1e6ef1d86859a6c5efd528e9b14a683d4203b.jpg)

![](images/a536106a6987a597c20618b3d2c0572ccdac5a1763b573e7eeff1f75bb4e489d.jpg)  
URUQI<sub>Syn</sub>-8B: 0.07 mQwen3.8-27B: 9.14 mInternVL3-8B: 1.82 m

![](images/66a7855263fd5f863e7ff1cecabe0bf7c2e86dedea9dd906b32c397d509e0bde.jpg)

![](images/8f1834391ad76af9bce08d3c5c63f199229010a4b2f22988eb901e309af11ca1.jpg)

(a) Scene geometry and observer motion  
(b) Egocentric object localization Model history  
![](images/7f214957f232bcd08daa1d2b9525acba4777ee09728271cc38071843b19a87c2.jpg)

![](images/b42b04d1914405c13063f8d93d13d709157ac65fdb254f6d01bc2476c15cf079.jpg)

t = 26 | Absent  
![](images/90cf2b05158984201c4498b60a48ace6484612bf9be4b1e8c830ab5707375258.jpg)  
URUQI<sub>Syn</sub>-8B: 0.19 mQwen3.8-27B: 0.94 mInternVL3-8B: 2.31 m

![](images/5baaa0a9e47b53b6197a46989f3afccbc29b97a66595e45317bc60c5c2e6f80b.jpg)

$$
\mathsf { U R U Q l } _ { \mathsf { S y n } ^ { - 8 \mathsf { B } } }
$$

![](images/74404a2f2f26f67201a3c570ebe1454451e5a3a2416b518745a38a69da17accc.jpg)  
URUQI<sub>Syn</sub>-8B: 0.41 mQwen3.8-27B: 2.53 mInternVL3-8B: 1.58 m  
URUQI<sub>Syn</sub>-8B: 0.57 mQwen3.8-27B: 3.10 mInternVL3-8B: 1.77 m

![](images/1f4616d3d61bb76fed5f7644fff2bb13e1abc67b08c823bd1c63fd74d05c41ac.jpg)

![](images/a728b074aa482c9eac2c41e97f59e24ce3383b0d1ca590cc4da3f159a6642857.jpg)  
Figure 7: Variation in the effect of answer history. For the absent cabinet, $\mathrm { U R U Q I _ { S y n } } – 8 \mathbf { B }$ has lower error under Model history at $t = 1 5 ~ ( 0 . 4 1$ versus 0.53 m), but higher error at t = 26 (0.57 versus 0.07 m). Top: No QA history; bottom: Model history. Both conditions use the same object and observation steps, with identical coordinate limits across all six localization plots.

## B SPATIAL MOTIFS AND TRAJECTORY ACQUISITION

## B.1 MOTIF INSTANTIATION

Motif definitions. We instantiate each motif by selecting scene objects and searching for a camera trajectory that satisfies its motion and visibility constraints. (1) Rotate in place bounds displacement from the initial camera position while changing heading. (2) Straight pass translates past the object with a bound on cumulative rotation. (3) Walk and turn combines translation and heading changes after an initial observation of the target. (4) Multi-turn walk separates the motion into translational and turning segments, with constraints on both the number of turns and accumulated rotation. (5) Occlusion traversal moves the observer toward or through an occluding region, with constraints that distinguish physical occlusion from disappearance outside the field of view. (6) Reference survey binds an origin object, a facing anchor, and a target, then scans their bearings from a fixed station. Each object must be observed, while no single frame provides a clear view of all three. (7) Landmark chain connects two endpoint objects through intermediate anchors. Each adjacent pair must have clear co-visible observations, while the endpoints remain non-co-visible throughout the sequence.

Trajectory search. To search for these trajectories, we project scene geometry onto the ground plane to construct an occupancy map and retain object centers, extents, categories, and instance identities for role binding. Camera stations are sampled in free space. Unobstructed station pairs are connected directly; other pairs use grid-based A\* search. Routes are simplified and interpolated into camera poses with bounded inter-frame translation and rotation. Candidate trajectories are screened for camera clearance, path validity, and the constraints of the selected motif.

Acquisition settings. The default trajectory-planning configuration uses a camera height of 1.5 m and a horizontal field of view of 90<sup>◦</sup> for geometric planning. Consecutive planned observations are constrained to at most 1.2 m of translation and 40<sup>◦</sup> of rotation. Camera clearance is checked with a 0.30 m horizontal radius over heights of 0.10–1.70 m. The planned poses are exported as an ordered camera schedule, with one observation associated with each scheduled viewpoint. Sequence length is therefore measured in observation steps.

## B.2 TRAJECTORY RENDERING AND VALIDATION

Candidate trajectories are rendered in OmniGibson (Li et al., 2022), recording RGB, camera pose, metric depth, and instance and semantic identity channels at each observation. Geometric visibility estimates guide trajectory search; rendered instance masks establish object support in the resulting images. The evidence checks evaluate the required visibility transitions and co-visibility relations. For occlusion-traversal candidates, depth and instance evidence are used to check whether disappearance is caused by a foreground blocker. Object bindings, poses, and supporting observations are retained for compilation.

## B.3 TRAJECTORY STATISTICS

The source pool for URUQI contains 11,738 distinct trajectories across 50 BEHAVIOR scenes (Li et al., 2022), with 212,209 rendered observations in total. Trajectories contain 6–52 observations, with a mean of 18.08 and a median of 15. Each source trajectory yields multiple QA episodes by varying the selected observations, target objects, and queries.

Data splits. We partition the source trajectories into 8,818 training, 2,696 test, and 224 validation trajectories. The 12 test scenes are disjoint from the 38 training scenes. The final benchmark uses eight of the 12 held-out test scenes.

## C SPATIAL SUPERVISION COMPILATION

## C.1 COORDINATE CONVENTIONS AND MOTION TARGETS

Reference frames. Positions are expressed in meters and reported angles in degrees. The world frame is right-handed with $+ Z$ upward. The pose $T _ { t }$ maps observer-frame coordinates to world coordinates; therefore, $T _ { i } ^ { - 1 } T _ { j }$ maps coordinates from frame $j$ to frame i.

For planar motion targets and horizontal object-location targets, we use forward–left coordinates: forward and left are positive, while backward and right are negative. At zero yaw, the observer faces world $+ Y$ , with world $+ X$ to its right. For yaw $\psi _ { t }$ , the forward and left basis vectors form the matrix

$$
\begin{array} { r l } { B _ { t } = \left[ { \begin{array} { c c } { - \sin \psi _ { t } } & { - \cos \psi _ { t } } \\ { \cos \psi _ { t } } & { - \sin \psi _ { t } } \end{array} } \right] , } \end{array}\tag{5}
$$

which maps observer-relative horizontal coordinates to world-plane coordinates.

Self-motion targets. Let $o _ { t } \in \mathbb { R } ^ { 2 }$ denote the observer’s horizontal position. For observations $i < j$ , the displacement and heading targets are

$$
d _ { i j } = B _ { i } ^ { \mathsf { T } } ( o _ { j } - o _ { i } ) , \qquad \delta \psi _ { i j } = \mathrm { w r a p } ( \psi _ { j } - \psi _ { i } ) ,\tag{6}
$$

where wrap returns the signed principal angle. The two components of $d _ { i j }$ give displacement along the forward and left axes of the earlier observation. Positive heading change denotes a left turn. These targets describe the net change between the named viewpoints. Traveled distance and accumulated rotation are computed separately by summing translation magnitudes and absolute heading changes over the intervening observed steps.

Object coordinates under observer motion. For a static object with world bounding-box center c, let $c _ { x y }$ denote its horizontal projection. Its horizontal location at step t is

$$
z _ { t } = B _ { t } ^ { \mathsf { T } } ( c _ { x y } - o _ { t } ) .\tag{7}
$$

The same location expressed at a later viewpoint satisfies

$$
z _ { j } = B _ { j } ^ { \top } B _ { i } ( z _ { i } - d _ { i j } ) .\tag{8}
$$

This relation expresses the same static object in the observer frames before and after motion. Objectlocation targets are computed directly from scene geometry.

## C.2 OBJECT IDENTITY AND PERSISTENT LOCALIZATION

Establishing object identity. Persistent object-localization queries refer to objects clearly observed in the available image history. Each question identifies its target by an unambiguous category name or by a marker in an earlier image. Scene instance identities associate these references across viewpoints, while rendered visibility evidence determines whether the object is currently visible or supported only by earlier observations. This ensures that queries about objects outside the current view refer to entities already encountered in the episode.

Localization across viewpoints. We construct sequences of location queries for the same object as the observer moves, including periods when the object is no longer visible and, when available, its subsequent reappearance. Each target is computed from the object’s fixed world center and the camera pose at the queried viewpoint, following Eq. 8. These sequences supervise how an established object’s observer-relative location changes with self-motion, even when no new visua observation of that object is available.

## C.3 OPERATIONS OVER SPATIAL STATE

Operation families. M3 supervision applies spatial operations to information established along a trajectory. We organize these operations into six broad families, described in Table 5.

Table 5: Examples of spatial operations used in M3 supervision.
<table><tr><td>Operation family</td><td>Operation</td><td>Examples</td></tr><tr><td>Reference-frame reasoning</td><td>Express relations in a specified frame</td><td>Camera-relative direction; object-centered reference</td></tr><tr><td>Hypothetical motion</td><td>Update relations after imagined motion</td><td>Rotation; translation; turn-move sequences</td></tr><tr><td>Inverse spatial inference</td><td>Solve for an unknown from spatial constraints</td><td>Target identification from directional constraints</td></tr><tr><td>Measurement and comparison</td><td>Measure, compare, or rank geometry</td><td>Distance; size; height; area</td></tr><tr><td>Temporal and set operations</td><td>Query events and combine observations</td><td>Appearance order; reappearance; counts; set overlap</td></tr><tr><td>Route reasoning</td><td>Evaluate paths between locations</td><td>Route description; traversable-length comparison</td></tr></table>

## D DENSE TRAJECTORY EVALUATION DETAILS

## D.1 EVALUATION PROTOCOL

Object registration. An object is registered after two consecutive observations satisfying the following conditions: its center is 0.5–5 m from the camera horizontally, its instance mask contains at least 4,000 pixels and lies at least 16 pixels from the image boundary, the mask covers at least 60% of the projected bounding-box area, and at most 5% of the projected box lies outside the image. Pixel thresholds are normalized to a 1024 × 1024 image. Mask coverage serves as a visibility proxy.

We register at most three objects per trajectory in chronological order. Registration uses only current and past observations. A numbered marker identifies each object in its registration image, which subsequent questions reference. Once registered, an object remains eligible for queries even when it becomes invisible or leaves the registration distance range.

Queries and visibility stages. At each frame after the first, we query motion from the preceding frame, followed by the current observer-relative locations of all registered objects. Location targets are the horizontal camera-frame coordinates of the scene object bounding-box centers. Of the 100 trajectories, 90 contain registered objects and 87 include absent-object queries; the other ten contribute only self-motion queries.

Table 6 defines the five mutually exclusive stages used to group localization queries. Overall localization scores include all five stages.

Table 6: Localization query stages in the dense evaluation. Pixel counts are normalized to 1024 × 1024 resolution; stages other than Initial apply after registration.
<table><tr><td>Stage</td><td>Definition</td><td>Queries</td></tr><tr><td>Initial</td><td>Query at the object&#x27;s registration frame</td><td>231</td></tr><tr><td>Absent</td><td>Zero visible instance pixels</td><td>2,240</td></tr><tr><td>Reappeared</td><td>First observation with at least 256 instance pixels after an absence</td><td>79</td></tr><tr><td>Visible</td><td>Other observations with at least 256 instance pixels after registration</td><td>1,651</td></tr><tr><td>Weak visibility</td><td>Positive instance-pixel count below 256</td><td>6</td></tr><tr><td>Total</td><td></td><td>4,207</td></tr></table>

## D.2 BASELINE PROMPTS

For InternVL3-8B and Qwen3.8-27B, we append answer-format instructions to each motion and object-localization question.

## Self-motion estimation.

User. For this question, compare Image 1 with the current image.

How do your position and heading change from Image 1 to the current image?

Answer briefly with estimated numeric distances in meters and the heading change in degrees, relative to the earlier camera. Use forward/backward, left/right, and turn left/right. Format example only: Move 0.4 m forward and 0.2 m right. Turn left 15 degrees. For no change, answer: Stay still. Give your best estimate; output only the short answer, no explanation.

InternVL3-8B: Move 0.3 m forward. Turn right 10 degrees.

Qwen3.8-27B: Stay still.

## Object localization.

User. Using observations through Image 2, where is the object introduced as Marker 1 in Image 2 relative to your position and heading in Image 2? Give forward/backward and left/right distances in meters.

Answer briefly with two estimated distances in meters in the current camera frame, using forward/backward and left/right. Format example only: 2.0 m forward, 0.5 m left. Give your best estimate; output only the short answer, no explanation.

InternVL3-8B: 1.5 m forward, 0.5 m right.

Qwen3.8-27B: 3.0 m forward, 1.5 m right

## D.3 METRICS AND GEOMETRIC PROPAGATION

Scoring. Object locations use the horizontal observer frame defined in Appendix C.1. For a query set $Q .$ , let $V \subseteq Q$ contain the queries with valid location estimates. Position error and correctness are

$$
e _ { i } = \| \hat { z } _ { i } - z _ { i } \| _ { 2 } , \qquad c _ { i } = \left\{ \begin{array} { l l } { { \bf 1 } [ e _ { i } \leq 0 . 5 \mathrm { m } ] , } & { i \in V , } \\ { 0 , } & { i \notin V . } \end{array} \right.\tag{9}
$$

Missing or unparseable predictions, including uninitialized propagated states, count as incorrect. Accuracy and mean position error are

$$
\mathrm { A c c } _ { \mathrm { q u e r y } } = \frac { 1 0 0 } { \left| Q \right| } \sum _ { i \in Q } c _ { i } , \qquad \overline { { e } } = \frac { 1 } { \left| V \right| } \sum _ { i \in V } e _ { i } .\tag{10}
$$

All localization results weight queries equally. Accuracy includes every query, while mean error is computed over valid estimates. The geometric conditions in Table 7 provide estimates for all queries.

For self-motion, translation error is the Euclidean difference between predicted and ground-truth horizontal displacements in the earlier camera frame. For motion query i, the predicted and groundtruth heading changes are $\widehat { \delta \psi } _ { i }$ and $\delta \psi _ { i }$ ; their error is $\vert \mathrm { w r a p } _ { [ - 1 8 0 ^ { \circ } , 1 8 0 ^ { \circ } ) } ( \widehat { \delta \psi } _ { i } - \delta \psi _ { i } )$ . Motion errors are averaged over valid outputs.

Geometric propagation. An object is initialized at the first step $t _ { 0 }$ with a positive model visibility response and a valid location estimate $\hat { z } _ { t _ { 0 } }$ . Using the horizontal basis $B _ { t }$ and observer position $o _ { t }$ from Appendix C.1, we store a fixed world-plane point and compute its subsequent observer-relative locations as

$$
\begin{array} { r } { \hat { c } _ { x y } = B _ { t _ { 0 } } \hat { z } _ { t _ { 0 } } + o _ { t _ { 0 } } , \qquad \hat { z } _ { t } ^ { \mathrm { p r o p } } = B _ { t } ^ { \top } ( \hat { c } _ { x y } - o _ { t } ) . } \end{array}\tag{11}
$$

Later VLM location estimates do not update the stored point. Initialization uses model predictions without ground-truth instance masks; direct prediction requires no visibility gate. The posesource comparison holds object-location and auxiliary visibility predictions fixed and uses the same propagation rule, varying only whether poses are ground truth or integrated from predicted motion. Ground-truth poses are supplied only to the geometry module.

## D.4 COMPLETE RESULTS

The following table supplements Table 4 with overall accuracy and mean position errors under the same evaluation conditions.

Table 7: Complete pose-assisted object-localization results. (a) Direct prediction and pose-based propagation use the same object predictions under No QA history; GT poses are used for propagation. (b) With Model-history object predictions and pose-based propagation fixed, we vary the source of camera poses. GT poses are supplied only to the geometry module, not to the VLM. Acc@0.5m uses a 0.5 m horizontal-error threshold in the current camera frame; all metrics weight queries equally. All and Absent contain 4,207 and 2,240 queries, respectively. Bold indicates the best result within each model and panel.

(a) Direct prediction versus pose-based propagation  
Fixed: No QA history; GT poses for propagation.
<table><tr><td rowspan="2">Model</td><td rowspan="2">Localization method</td><td colspan="2">Acc@0.5m (%) ↑</td><td rowspan="2">Mean error (m) ↓ All</td></tr><tr><td>All</td><td>Absent</td></tr><tr><td rowspan="2">Qwen3.8-27B</td><td>Direct prediction</td><td>6.44</td><td>1.29</td><td>4.725</td></tr><tr><td>Pose-based propagation</td><td>16.81</td><td>16.88</td><td>1.322</td></tr><tr><td rowspan="2">URUQIsy-8B</td><td>Direct prediction</td><td>49.56</td><td>35.13</td><td>0.743</td></tr><tr><td>Pose-based propagation</td><td>69.65</td><td>71.61</td><td>0.412</td></tr></table>

(b) Effect of camera-pose source

Fixed: Model history; pose-based propagation. Evaluated on absent objects.
<table><tr><td>Model</td><td>Camera-pose source</td><td>Acc@0.5m (%) ↑ Absent</td><td>Mean error (m) ↓ Absent</td></tr><tr><td rowspan="2">Qwen3.8-27B</td><td>GT poses</td><td>18.48</td><td>0.961</td></tr><tr><td>Integrated predicted motion</td><td>3.97</td><td>3.737</td></tr><tr><td rowspan="2"> $\mathrm { U R U Q I { s y n } } { - } 8 \mathrm { B }$ </td><td>GT poses</td><td>72.86</td><td>0.397</td></tr><tr><td>Integrated predicted motion</td><td>34.02</td><td>1.084</td></tr></table>

## D.5 AUXILIARY VISIBILITY AND OBJECT CORRESPONDENCE

Diagnostic protocol. For each localization query, we separately ask whether the registered object is currently visible and, if so, request a pixel on that object. The model receives all images up to the current frame, including the marked registration image, without QA history or privileged geometry. The same predictions are reused across the model’s history and pose conditions. A valid positive response includes an in-bounds pixel coordinate; invalid or uncertain responses cannot initialize propagation. Instance masks are used only to score pixel correspondence, not to accept an initialization. This auxiliary query does not affect the native localization scores in Table 3.

Metrics. Ground-truth visibility requires at least one target-instance pixel, giving 1,967 visible and 2,240 absent queries. Precision and recall measure valid positive visibility responses against these labels. False-visible rate is the fraction of absent queries receiving a valid positive response. Pixel Hit is the fraction of all ground-truth-visible queries with a valid positive response and a predicted pixel on the correct instance; missed detections and invalid responses receive no credit.

Table 8: Auxiliary visibility and object correspondence on 4,207 queries. All values are percentages; bold marks the best value in each column.
<table><tr><td></td><td>Visibility</td><td>Visibility recall ↑</td><td>Pixel Hit ↑</td><td>False-visible</td></tr><tr><td>Model</td><td>precision ↑</td><td>88.87</td><td>7.52</td><td>rate ↓</td></tr><tr><td>InternVL3-8B Qwen3.8-27B</td><td>54.13 93.51</td><td>78.39</td><td>50.64</td><td>66.12 4.78</td></tr><tr><td> $\mathrm { U R U Q I _ { S y n } } – 8 \mathbf { B }$ </td><td>68.31</td><td>99.49</td><td>71.68</td><td>40.54</td></tr><tr><td></td><td></td><td></td><td></td><td></td></tr></table>

Results. Table 8 shows that $\mathrm { U R U Q I _ { S y n } }$ -8B achieves the highest Pixel Hit, but has a higher falsevisible rate than Qwen3.8-27B (40.54% versus 4.78%). These metrics are computed over their respective query sets, not only at initialization. As described in Appendix D.3, visibility responses after initialization do not change the stored object location.
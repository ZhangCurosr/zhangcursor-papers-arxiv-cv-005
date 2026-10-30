# VideoMemBench: How Well Do MLLMs Remember in Video Conversations?

Zhengdong Hu<sup>1,∗</sup>, Chao Wang<sup>1,∗</sup>, Kaituo Feng<sup>2</sup>, Huanjin Yao<sup>3</sup>, Mingfei Lu<sup>1</sup>, Hehe Fan<sup>4</sup>, Yi Yang<sup>4,†</sup>

<sup>1</sup> University of Technology Sydney, Sydney, Australia

<sup>2</sup> The Chinese University of Hong Kong, Hong Kong, China

<sup>3</sup> The Hong Kong University of Science and Technology, Hong Kong, China

<sup>4</sup> Zhejiang University, Hangzhou, China

Multimodal large language models (MLLMs) can now process hour-long videos, making extended multi-turn dialogue over a single video a natural mode of use. Sustaining such a dialogue requires more than answering each question in isolation: the model must retrieve and reuse information established earlier in the conversation, both what it has seen and what it has said, an ability we term in-context video memory. Existing video QA benchmarks leave this ability unmeasured: their questions are mutually independent, never reference one another, and each is answerable from the video alone, so nothing ever requires carrying information across turns. We introduce VideoMemBench, in which each sample pairs a single video with a long multi-turn dialogue, whose turns are connected by typed cross-turn dependencies: an Episodic Dependency refers back to a visual episode described in an earlier turn, probing memory of what the model has seen, while an Answer Dependency builds on the model’s own earlier answer, probing memory of what the model has said. Every dependency is verified considering two properties: 1) necessity, meaning the question cannot be answered without its earlier turns, and 2) non-leakage, meaning the question does not restate the information from its earlier turns. To enforce these properties, we design a construction pipeline with three LLM roles: a planner, a generator, and a judger, with each dialogue reviewed by human experts. VideoMemBench comprises 326 videos with 2,476 turns, of which 1,387 depend on earlier turns, including multi-parent dependencies and long-range call-back. Among over 20 evaluated open- and closed-source MLLMs, Claude Opus 5 performs the best with an overall score of only 45.7%, highlighting the challenge of in-context video memory. Our dataset and evaluation are released.

Project: https://videomembench.github.io Video: https://huggingface.co/datasets/VideoMemBench/data GitHub: https://github.com/Oliiveralien/VideoMemBench

## 1 Introduction

Frontier multimodal large language models (MLLMs) (Comanici et al., 2025; Singh et al., 2025; Team et al., 2026; Yang et al., 2025a; Anthropic, 2026a) can now process hour-long videos, with million-token contexts that accommodate an entire film, a full match, or a day-long recording. Building on this capability, extended multi-turn dialogue over a single long video is emerging as a natural mode of use: the model accompanies a user through the video with a stream of follow-up questions about its characters, events, and details. Sustaining such a conversation raises a key question: when consecutive questions concern the same video, can the model correctly retrieve and reuse what the dialogue has already established? We refer to this ability as in-context video memory.

Recently, in-context memory has been systematically evaluated for large language models (LLMs) in text-only settings (Maharana et al., 2024; Hsieh et al., 2024; Kuratov et al., 2024; Tan et al., 2025; Wu et al., 2025; Huet et al., 2025). However, extending this evaluation to video poses a key challenge: distinguishing questions that genuinely depend on prior dialogue context from those answerable using only the video and the current question. Existing video understanding benchmarks (Fu et al., 2025; Li et al., 2024; Mangalam et al., 2023;

![](images/d5c7d189a68a4922cccb8623283a9bf61a4fba5142c7f208d1a452a9baa9c6ca.jpg)  
Figure 1 Example of VideoMemBench with cross-turn dependencies: Independent turns (e.g., T<sub>1</sub>, T<sub>8</sub>) establish visually grounded facts; dependent turns refer back to earlier turns through expressions such as “the scene from T ” or “the target from $T _ { 1 } { } ^ { \dag } ,$ without restating earlier content. An Episodic Dependency (e.g., T<sub>2</sub>) requires recalling what the model has seen: the model must recover which scene $T _ { 1 }$ described before answering the question. An Answer Dependency (e.g., T<sub>3</sub>, T<sub>4</sub>) requires recalling what the model has said: each builds on the answer produced at an earlier turn.

Wu et al., 2024; Wang et al., 2025; Fu et al., 2026) are structurally unable to evaluate it. Typically, they decompose the QA evaluation into mutually independent questions: no question references the content or answer of another, so nothing ever needs to be carried across turns, and every question is answerable from the video alone. In other words, their dialogue histories are never load-bearing.

To this end, we introduce VideoMemBench. Each sample pairs a single video with a long multi-turn dialogue whose turns are connected by explicit, typed cross-turn dependencies, so that a sample is not a concatenation of separate questions but a memory chain running through the entire conversation. The key design is that answering a dependent turn should require the dialogue and the video jointly: the dialogue supplies what an earlier turn established, and the video supplies the visual facts to ground the current one, so that neither source alone sufices. Specifically, we design two dependency types, enforcing two forms of memory: Episodic Dependency, for recalling previously discussed visual episodes, and Answer Dependency, for recalling the model’s own earlier responses.

(1) Episodic Dependency arises when a question depends on the visual episode that an earlier turn described: only by recalling that referent from the dialogue can the model determine what is being asked. As shown in Figure 1, $T _ { 1 }$ asks about the suspect mugshot pinned to the briefing-room corkboard, and $T _ { 2 }$ then asks whether the hippo oficers are present in “the scene from $T _ { 1 } { } ^ { \dag }$ and what their names are, without restating where that scene is. Unless the model retains that $T _ { 1 }$ was about the briefing room, it cannot tell which part of the video “the scene from $T _ { 1 } { } ^ { \dag }$ refers to, let alone check who is in it. Episodic dependencies thus hinge on memory of what the model has seen.

(2) Answer Dependency arises when a question builds on the answer the model itself gave at an earlier turn: the model must continue from its own earlier response. As shown in Figure 1, $T _ { 1 }$ establishes that the suspect is an anteater, $T _ { 3 }$ then asks whether the animal that fox and rabbit meet at the shipping-container yard is “the target from $T _ { 1 } { } ^ { \dag }$ , which cannot be decided without the answer given at $T _ { 1 } , \ T _ { 4 }$ in turn asks what happens next to “the animal met in $T _ { 3 } { } ^ { , }$ , building on the answer to $T _ { 3 } .$ . Answering thus demands two abilities at once: retrieval, locating the earlier answer within the accumulated dialogue, and consistency, keeping the current answer faithful to it rather than silently re-judging the video. Answer dependencies thus hinge on memory of what the model has said.

Table 1 Comparison with existing benchmarks. Max Dur. is the maximum video duration in the benchmark. Multi-turn indicates multi-turn dialogue evaluation. Max Turns is the largest number of QA turns in one dialogue. Dep. Graph denotes structured dependency graphs for individual turns; Typed distinguishes answer and episodic dependencies; M-Parent indicates multiple parents.
<table><tr><td rowspan="2">Benchmark</td><td rowspan="2"></td><td rowspan="2">Input #Videos #Dialogs</td><td rowspan="2"></td><td rowspan="2"> $\# \mathrm { Q A s }$ </td><td rowspan="2">Max Dur. Max Turns Multi-turn</td><td rowspan="2"></td><td rowspan="2"></td><td rowspan="2"> $\begin{array} { c } { { \mathrm { D e p . } } } \\ { { \mathrm { G r a p h } } } \end{array}$ </td><td rowspan="2"></td><td rowspan="2">Typed M-Parent</td></tr><tr><td></td></tr><tr><td>MVBench (Li et al., 2024)</td><td>Video</td><td>3,641</td><td></td><td>4,000</td><td>3min</td><td>1</td><td>x</td><td>x</td><td>x</td><td>x</td></tr><tr><td>Video-MME (Fu et al., 2025)</td><td>Video</td><td>900</td><td></td><td>2,700</td><td>1h</td><td>1</td><td>x</td><td>x</td><td>x</td><td>x</td></tr><tr><td>LVBench (Wang et al., 2025)</td><td>Video</td><td>103</td><td></td><td>1,549</td><td>2h</td><td>1</td><td>X</td><td>x</td><td>x</td><td>X</td></tr><tr><td>MMDU (Liu et  ${ \mathrm { a l . , } }$  2024b)</td><td>Images</td><td></td><td>110</td><td>1,645</td><td></td><td>27</td><td>√</td><td>x</td><td>x</td><td>x</td></tr><tr><td>MT-Video-Bench (Pan et al., 2026)</td><td>Video</td><td>135</td><td>1,000</td><td>5,887</td><td>20 min</td><td>8</td><td>√</td><td>X</td><td>x</td><td>x</td></tr><tr><td>VideoMemBench (ours)</td><td>Video</td><td>326</td><td>326</td><td>2,476</td><td>23.47 h</td><td>21</td><td>√</td><td>√</td><td>√</td><td>√</td></tr></table>

Furthermore, as shown in the closing $T _ { 9 }$ of Figure 1, VideoMemBench stresses memory along two additional dimensions: a turn may depend on several earlier turns at once, requiring parallel recall of multiple episodes (T<sub>1</sub>) or answers $( T _ { 8 } ) ;$ ; and a dependency may span eight turns $( T _ { 1 }  T _ { 9 } )$ , keeping content from the beginning of the dialogue live throughout.

These structurally diverse dependencies ensure that answering questions from our VideoMemBench requires the model to exercise in-context video memory throughout the dialogue. We further verify every dependent turn for necessity and non-leakage: the question must be unanswerable without its parent turns, and must not restate the information they established before. In total, VideoMemBench comprises 326 videos, each paired with a human-reviewed dialogue, totaling 2,476 turns, of which 1,387 depend on earlier turns. Videos are drawn from five existing benchmarks and the web, covering 31 semantic domains across four categories (lifestyle & entertainment, knowledge & education, art & literature, sports & competition), and last up to 23.47 hours each. The comparison in Table 1 shows that single-turn benchmarks evaluate questions independently, whereas multi-turn benchmarks extend evaluation to dialogues. VideoMemBench further provides explicit dependency graphs with typed edges and multiple parents, enabling structured evaluation of in-context video memory.

Evaluating over 20 representative open- and closed-source MLLMs on VideoMemBench, we find that current models clearly struggle with cross-turn dependencies: the best model we tested (Claude Opus 5) achieves an overall score of only 45.7%, with 35.7% on answer-dependent turns and 49.1% on episodic-dependent turns, both below 53.9% on independent turns. These results indicate that in-context video memory remains an open problem for current MLLMs.

Our main contributions are summarized as follows:

• We study in-context video memory, the ability of a model to retrieve and reuse what earlier turns have established in a consecutive video QA dialogue, which existing benchmarks leave unmeasured.

• We build VideoMemBench, in which each sample is a long multi-turn dialogue grounded in a single video, with turns connected by typed cross-turn Episodic and Answer dependencies. VideoMemBench comprises 326 video dialogues, totaling 2,476 turns with 1,387 dependent, verified for necessity and non-leakage, and reviewed by human experts.

• We conduct extensive evaluations of over 20 open- and closed-source MLLMs on VideoMemBench. The best model achieves only 45.7% overall, with performance declining as dependency complexity increases, indicating that in-context video memory remains challenging for current MLLMs.

## 2 Related Works

Video Understanding Benchmarks. Existing video understanding benchmarks usually focus on temporal perception and reasoning. MVBench (Li et al., 2024) and TempCompass (Liu et al., 2024a) evaluate temporal understanding, while EgoSchema (Mangalam et al., 2023) targets long-form video QA. Broadening the scope, Video-MME (Fu et al., 2025) covers diverse video durations and tasks, and Video-MMMU (Hu et al., 2026) evaluates knowledge acquisition from multidisciplinary videos. For longer contexts, LongVideoBench (Wu et al., 2024) and LVBench (Wang et al., 2025) focus on long-video understanding, while EgoLife (Yang et al., 2025d) examines egocentric experiences spanning hours and days. Beyond temporal understanding, VSI-Bench (Yang et al., 2025c) evaluates video-based spatial reasoning. Despite this growing coverage and temporal range, these benchmarks evaluate questions independently, without requiring information established in the earlier dialogue.

Memory and Multi-Turn Evaluation. Existing benchmarks evaluate conversational memory and multi-turn interaction from several perspectives. LoCoMo (Maharana et al., 2024) and LongMemEval (Wu et al., 2025) assess long-term conversational memory, while MemBench (Tan et al., 2025) distinguishes factual and reflective memory. MT-Bench (Zheng et al., 2023) evaluates two-turn response quality. In multimodal settings, MMDU (Liu et al., 2024b) evaluates multi-turn, multi-image conversations, and MMRC (Xue et al., 2025) includes multi-turn image reasoning and memory recall, but neither targets dialogue memory grounded in a single video. VideoMemBench addresses this gap through explicit episodic and answer dependencies that link later questions to previously discussed video scenes and generated answers, with support for multiple parents and long-range connections.

## 3 VideoMemBench

Each VideoMemBench sample pairs a video with a multi-turn dialogue, annotated with reference answers and a typed dependency graph. Each edge links a dependent turn to a parent turn and is labeled as Episodic or Answer. Below, we describe the requirements for valid dependencies and visually grounded answers, followed by the construction pipeline and dataset statistics.

## 3.1 Design Requirements

Constructing valid dependencies requires going beyond ‘simple references to earlier turns. The key challenge lies in ensuring that cross-turn dependencies are genuinely necessary, do not leak prior information, and remain grounded in the video. This leads to three requirements:

Req-1: Dependency Necessity. A dependent question must become unanswerable when its parent turns are removed, even with access to the video and the remaining dialogue. Considering a cooking video, an implicit reference alone does not ensure this: asking how much of “that ingredient” is used requires no dialogue history if the video contains only one ingredient. Such a question avoids revealing the ingredient’s identity but fails the necessity requirement.

Req-2: Dependency Non-leakage. A dependent question must not restate the episode descriptions or answers from its parents, even partially. For example, if an earlier answer identifies “the flour in the red bowl”, asking “How much of the flour identified earlier is used?” partially reveals that answer. If both a red and a blue bowl contain flour, the parent remains necessary to identify the target bowl. Thus, a turn may satisfy necessity while violating non-leakage.

Req-3: Visual grounding. Every turn, whether dependent or not, must be anchored in a verifiable visual fact in the video. For dependent turns, this evidence must support the answer under the interpretation established by the parent turns.

## 3.2 Construction Pipeline

Guided by these requirements, we construct VideoMemBench through a pipeline combining three LLM roles with human expert review, as shown in Figure 2. A planner lays out the dependency structure of a dialogue before any question is written, providing the structural precondition for necessity (Req-1). A generator realizes the plan turn by turn, referring to earlier turns only through referring expressions (Req-2) and anchoring every answer in frame evidence (Req-3). A judger then verifies each generated turn against the three requirements and checks completed dialogues for consistency with the initial structural plan. Human experts finally review each retained dialogue.

![](images/cf50a4761775c2968b5fc8f6c96cd5f293bc275a6d447151338d9a69f4aa353b.jpg)  
Figure 2 The VideoMemBench construction pipeline. Starting from a curated video pool, a planner specifies turn intents and a typed dependency graph. A generator constructs the dialogue turn by turn, while a judger checks necessity, non-leakage, and visual grounding, returning failed turns for regeneration. Completed dialogues are checked against the plan and reviewed by human experts.

• Video Collection. We collect videos from multiple sources: Video-MME (Fu et al., 2025), VSI-Bench (Yang et al., 2025c), Video-MMMU (Hu et al., 2026), LongVideoBench (Wu et al., 2024) and the web. Web videos are sourced from YouTube across sports, talks and lectures, tutorials, film recaps, and cooking. We also include EgoLife (Yang et al., 2025d) recordings, allowing recurring people and activities to support dependencies across distant moments. This yields an initial pool of about 3.2k videos. We then use a pre-trained LLM to annotate video domains and scene composition, guiding the selection of videos across diferent semantic domains with recurring entities and events suitable for cross-turn dependencies. Further collection details are provided in Appendix A.

• Dependency Planning. To make dependencies structural rather than incidental (Req-1), the dialogue is designed before any question is generated. The planner samples keyframes across the video and produces a global dialogue plan with four components: (i) a video synopsis summarizing characters, events, and settings; (ii) dialogue guidelines specifying target content, capabilities to assess, and the number of turns; (iii) a dependency graph assigning parent turns and edge types to each dependent turn; and (iv) per-turn intents specifying the information to establish, the relevant video segment, and the information required from parent turns. The plan guides subsequent generation under two constraints: dependencies must arise from recurring video content, and the planned turns must collectively cover the full video. More planning details are provided in Appendix B.

![](images/44e2039f228c09377440b3d0a6331f3412f5a3e0aa67c3770ee0d6b8d5a565b8.jpg)

![](images/d0b5c8441254e98f69ebfdd63a402dda29923cbdfb9e3233e29d14b239047317.jpg)  
Figure 3 Dataset statistics of VideoMemBench. (a) Category distribution across four major domains and their finegrained subcategories. (b) Distribution of videos by duration. (c) Distribution of the number of dialogue turns per video. (d) Distribution of question dependencies, including independent questions, episodic dependencies, and answer dependencies. (e) Text-length statistics for questions, answers, and answer choices, reported as mean ± standard deviation.

• Turn-wise Generation. Following the plan, the generator constructs the dialogue turn by turn, densely sampling frames from each turn’s target segment. For dependent turns, the segments are associated with their parents. It first identifies a verifiable visual fact as the reference answer, then formulates a question according to the planned intent and dependency types. Dependent questions use referring expressions without restating the information supplied by their parents. The supporting temporal interval is recorded as the answer’s evidence window.

• Turn-level Check. For dependent turns, the judger checks necessity by removing parent turns and rejecting questions that remain answerable from the video frames and remaining dialogue. It checks non-leakage by comparing each question with its parents for restated episode descriptions or disclosed answers. For every turn, it verifies that frames from the annotated evidence window support the reference answer and rejects questions that semantically duplicate earlier ones. Rejected turns are regenerated using the judger’s feedback; two consecutive failures trigger replanning of the afected portion of the dialogue. Our prompt templates for all stages are given in Appendix C.

• Expert Review. Eight human experts review all dialogues that pass the automated checks against the corresponding videos. Turns with incorrect answers, invalid dependencies, or information leakage are revised or removed, ensuring that every retained dialogue is human-reviewed. Details of the dataset review and the separate human baseline evaluation are provided in Appendix D.

## 3.3 Dataset Statistics

Figure 3 summarizes the overall statistics of VideoMemBench. VideoMemBench contains 326 dialogues grounded in 326 videos, totaling 2,476 QA turns. Of these, 1,089 (43.98%) are independent, while 1,387 (56.02%) are dependent, with 1,794 typed dependencies in total: 1,200 Episodic and 594 Answer dependencies.

Among dependent turns, 348 have multiple parents, including 289 double-parent and 59 triple-parent turns. Dialogues span 4-21 turns, averaging 7.6 turns. Among dependent turns, dependency distance ranges from one to nine turns $( e . g . , T _ { 1 }  T _ { 5 }$ has distance 4), with up to three parents $( e . g . , T _ { 1 } + T _ { 4 }  T _ { 6 }$ has 2 parents) and a maximum chain depth of seven $( e . g . , T _ { 1 }  T _ { 3 }  T _ { 4 }  T _ { 7 }$ has depth 3). Further details are provided in the appendix.

## 4 Experiments

## 4.1 Experimental Setup

Models. We evaluate 25 MLLMs, including 8 closed-source models: Claude Opus 5 (Anthropic, 2026a), Claude Sonnet 5 (Anthropic, 2026b), GPT-5.6-sol (OpenAI, 2026d), GPT-5.5 (OpenAI, 2026c), GPT-5.4 (OpenAI, 2026a), Gemini 3.8 Flash (Google DeepMind, 2026b), Gemini 3.5 Flash (Google DeepMind, 2026a), and Gemini 3.1 Pro Preview (Google, 2026), as well as 17 open-weight models: Kimi K3 (Team et al., 2026), Qwen3.5 (0.8B, 4B, 9B, 27B, 35B-A3B, 122B-A10B, 397B-A17B) (Team, 2026), Qwen3-VL (4B, 8B, 30B-A3B, 32B, 235B-A22B) (Bai et al., 2025a), Qwen2.5-VL (3B, 7B, 32B) (Bai et al., 2025b), and Keye-VL-1.5-8B (Yang et al., 2025b).

Evaluation protocol. All models receive the same 128 frames sampled across the full video, with timestamps indicating their temporal positions. The sampled frame sequence is fixed across models. The video frames are provided at the beginning of the conversation and remain accessible throughout the dialogue. Questions are presented strictly in dialogue order, with the model’s own previous responses retained in the conversation history. Specifically, the input at turn t is

$$
V \parallel \left( q _ { 1 } , \hat { a } _ { 1 } \right) \parallel \cdots \parallel \left( q _ { t - 1 } , \hat { a } _ { t - 1 } \right) \parallel q _ { t } ,\tag{1}
$$

where V denotes the timestamped frame sequence, q<sub>i</sub> the question and its answer options where applicable, $\hat { a } _ { i }$ is the model response at turn $i ,$ and ∥ denotes concatenation.

## 4.2 Evaluation Metrics

We use GPT-5.4 mini (OpenAI, 2026b) as the judge for open-ended responses and extract the predicted option for multiple-choice questions. Each response is graded as correct or incorrect. We report three metrics with distinct evaluation subsets and scoring rules, plus an aggregate score.

Turn Accuracy (TurnAcc). TurnAcc measures accuracy over all turns. Each response is scored independently of whether the model answered its parent turns correctly.

Episodic Dependency Accuracy (EpiDepAcc). EpiDepAcc measures accuracy over turns with at least one Episodic dependency. Each response is scored independently of its parents’ answer correctness, since an Episodic dependency refers to the visual context described by a parent question rather than the parent’s answer. This metric measures recall of what the model has seen.

Answer Dependency Accuracy (AnsDepAcc). AnsDepAcc evaluates turns with at least one Answer dependency using chain crediting: a turn receives credit only if its response and all responses along its Answer-dependency ancestry are correct. Formally,

$$
{ \mathrm { A n s D e p A c c } } = { \frac { 1 } { | { \mathcal { T } } _ { \mathrm { a n s } } | } } \sum _ { t \in { \mathcal { T } } _ { \mathrm { a n s } } } { \mathrm { c o r r } } ( t ) \prod _ { a \in { \mathrm { A n c } } _ { \mathrm { a n s } } ( t ) } { \mathrm { c o r r } } ( a ) ,\tag{2}
$$

where $\mathcal { T } _ { \mathrm { a n s } }$ contains turns with at least one Answer dependency, corr $\cdot ( t ) \in \{ 0 , 1 \}$ indicates whether the response at turn t is correct, and $\operatorname { A n c } _ { \mathrm { a n s } } ( t )$ contains all ancestors reachable by recursively following only Answer edges toward earlier turns. Episodic edges are excluded from this traversal. By accounting for both current-turn errors and upstream failures across answer-dependent chains, this metric evaluates recall of what the model has said.

Multi-dependent turns with both dependency types contribute to both EpiDepAcc and AnsDepAcc under their respective scoring rules.

Table 2 Main results on VideoMemBench. Models are evaluated under the self-rollout protocol. TurnAcc: accuracy over all turns, scored individually. EpiDepAcc: accuracy on turns with at least one Episodic dependency. AnsDepAcc: accuracy on turns with at least one Answer dependency under chain crediting. Overall: the arithmetic mean of the three metrics. The best and second-best results in each metric within each model group are bold and underlined, respectively.
<table><tr><td>Model</td><td>Frames</td><td>TurnAcc (%)</td><td>EpiDepAcc (%)</td><td>AnsDepAcc (%)</td><td>Overall (%)</td></tr><tr><td>Human</td><td></td><td>88.1</td><td>86.0</td><td>84.2</td><td>86.1</td></tr><tr><td>Closed-source models</td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Claude Opus 5</td><td>128</td><td>52.3</td><td>49.1</td><td>35.7</td><td>45.7</td></tr><tr><td>Gemini 3.8 Flash</td><td>128</td><td>50.4</td><td>46.6</td><td>30.3</td><td>42.4</td></tr><tr><td>GPT-5.6-sol</td><td>128</td><td>50.0</td><td>47.8</td><td>27.1</td><td>41.7</td></tr><tr><td>Gemini 3.5 Flash</td><td>128</td><td>49.6</td><td>45.5</td><td>29.7</td><td>41.6</td></tr><tr><td>GPT-5.5</td><td>128</td><td>48.0</td><td>46.5</td><td>27.0</td><td>40.5</td></tr><tr><td>Claude Sonnet 5</td><td>128</td><td>47.8</td><td>45.3</td><td>27.3</td><td>40.2</td></tr><tr><td>Gemini 3.1 Pro Preview</td><td>128</td><td>45.7</td><td>44.8</td><td>27.0</td><td>39.1</td></tr><tr><td>GPT-5.4</td><td>128</td><td>44.4</td><td>42.6</td><td>21.9</td><td>36.3</td></tr><tr><td>Open-weight models</td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Kimi K3</td><td>128</td><td>50.5</td><td>49.0</td><td>27.1</td><td>42.2</td></tr><tr><td>Qwen3-VL-235B-A22B</td><td>128</td><td>43.1</td><td>41.6</td><td>18.6</td><td>34.4</td></tr><tr><td>Qwen3.5-27B</td><td>128</td><td>42.2</td><td>39.8</td><td>20.9</td><td>34.3</td></tr><tr><td>Qwen3.5-397B-A17B</td><td>128</td><td>42.3</td><td>40.1</td><td>20.3</td><td>34.2</td></tr><tr><td>Qwen3.5-122B-A10B</td><td>128</td><td>42.4</td><td>39.5</td><td>19.9</td><td>33.9</td></tr><tr><td>Qwen3.5-35B-A3B</td><td>128</td><td>38.2</td><td>36.3</td><td>18.0</td><td>30.8</td></tr><tr><td>Qwen3-VL-32B</td><td>128</td><td>39.3</td><td>35.9</td><td>17.2</td><td>30.8</td></tr><tr><td>Qwen3-VL-30B-A3B</td><td>128</td><td>36.4</td><td>35.1</td><td>15.8</td><td>29.1</td></tr><tr><td>Qwen3.5-9B</td><td>128</td><td>34.6</td><td>33.4</td><td>12.9</td><td>27.0</td></tr><tr><td>Qwen3-VL-8B</td><td>128</td><td>34.8</td><td>32.1</td><td>12.3</td><td>26.4</td></tr><tr><td>Qwen3.5-4B</td><td>128</td><td>33.8</td><td>32.6</td><td>12.5</td><td>26.3</td></tr><tr><td>Qwen3-VL-4B</td><td>128</td><td>32.4</td><td>29.4</td><td>12.9</td><td>24.9</td></tr><tr><td>Qwen2.5-VL-32B</td><td>128</td><td>31.7</td><td>32.6</td><td>9.6</td><td>24.6</td></tr><tr><td>Qwen2.5-VL-7B</td><td>128</td><td>26.2</td><td>24.9</td><td>8.4</td><td>19.8</td></tr><tr><td>Keye-VL-1.5-8B</td><td>128</td><td>25.4</td><td>24.5</td><td>8.2</td><td>19.4</td></tr><tr><td>Qwen2.5-VL-3B</td><td>128</td><td>23.5</td><td>24.4</td><td>5.7</td><td>17.9</td></tr><tr><td>Qwen3.5-0.8B</td><td>128</td><td>21.6</td><td>21.3</td><td>5.7</td><td>16.2</td></tr></table>

Overall. Overall is the arithmetic mean of TurnAcc, EpiDepAcc, and AnsDepAcc, which is used to simply rank models. It aggregates complementary, overlapping evaluation subsets rather than representing accuracy on a separate set of questions.

## 4.3 Benchmark Results

Table 2 reports results for 25 models, including 8 closed-source and 17 open-weight MLLMs. Claude Opus 5 achieves the highest overall score of 45.7%, compared with 86.1% for human annotators. This gap suggests substantial room for MLLMs to improve in-context video memory.

Closed-source Models. Claude Opus 5 leads the closed-source models across all three metrics, achieving 52.3% on TurnAcc, 49.1% on EpiDepAcc, and 35.7% on AnsDepAcc. Gemini 3.8 Flash and GPT-5.6-sol follow with overall scores of 42.4% and 41.7%, respectively, while GPT-5.4 scores lowest at 36.3%. Across these closedsource models, AnsDepAcc is 16.6–22.9 percentage points below TurnAcc, compared with only 3.9 percentage points for human annotators. Since AnsDepAcc requires correctness throughout the Answer-dependency ancestry, this gap reflects both current-turn errors and accumulated upstream failures. These results highlight the dificulty of maintaining correct answers across dependent turns in extended video conversations, even for frontier models.

Open-weight Models. Kimi K3 leads the open-weight models with an overall score of 42.2%, ranking third overall and second on both TurnAcc and EpiDepAcc across all evaluated models. It achieves 50.5% on TurnAcc, 49.0% on EpiDepAcc, and 27.1% on AnsDepAcc. All other open-weight models score below GPT-5.4, the lowest-performing closed-source model (36.3% overall). The runner-up, Qwen3-VL-235B-A22B, scores

(c) Turn position: independent vs. dependent dependent Extension to Independent  
(a) Dependency distance  
![](images/96dae6fe899ad39eda43edca3e11e5c30db267d3b4ba52b4e24d7d52c0737816.jpg)

(b) Dependencies count  
![](images/9760b88a68b7945464e9d59a54c183535985dba240a11e7609aa6e84f9c74321.jpg)  
(d) Answer dependent analysis

![](images/227a4c2414d67fefe802ec0b2e07e54e940c8ed1c4f06772db77cd707a4c127f.jpg)

![](images/9997d65a7d7277b230f7131f3f38b7336e3ad97f55a13f7cdb51034bf5d56d7a.jpg)

![](images/99ce254f868ff491aab185158878af29f3b6fdace055671e55090c6f40abddf1.jpg)  
Figure 4 Analysis of factors affecting in-context video memory. (a) Accuracy on dependent turns by dependency distance, measured from the current turn to its earliest parent; the dashed line shows the mean across all evaluated models. (b) Accuracy on dependent turns by the number of parents. (c) Independent- and dependent-turn accuracy by dialogue position for two models: the light segment shows dependent-turn accuracy, the full bar height shows independent-turn accuracy, and the dark segment represents their diference. (d) Accuracy on answer-dependent turns under self-rollout and oracle histories. Oracle histories replace previous model responses with ground-truth answers.

Table 3 Independent versus dependent turns. Indep.: accuracy over the independent turns. Dep.: mean of EpiDepAcc and AnsDepAcc. ∆ = Indep. − Dep., measured in percentage points (pp).
<table><tr><td>Model</td><td>Indep. (%)</td><td>EpiDepAcc (%)</td><td>AnsDepAcc (%)</td><td>Dep. (%)</td><td>∆(pp)</td></tr><tr><td>Kimi K3</td><td>51.4</td><td>49.0</td><td>27.1</td><td>38.1</td><td>13.4</td></tr><tr><td>GPT-5.6-sol</td><td>50.6</td><td>47.8</td><td>27.1</td><td>37.5</td><td>13.1</td></tr><tr><td>Claude Sonnet 5</td><td>50.0</td><td>45.3</td><td>27.3</td><td>36.3</td><td>13.7</td></tr><tr><td>Qwen3.5-27B</td><td>46.0</td><td>39.8</td><td>20.9</td><td>30.3</td><td>15.7</td></tr><tr><td>Qwen3-VL-235B-A22B</td><td>47.4</td><td>41.6</td><td>18.6</td><td>30.1</td><td>17.3</td></tr></table>

34.4% overall, 7.8 percentage points below Kimi K3 and narrowly ahead of Qwen3.5-27B (34.3%). Within the evaluated dense Qwen3.5 models, the overall score increases with model size, from 16.2% at 0.8B to 34.3% at 27B. At comparable scales, Qwen3-VL-32B outperforms Qwen2.5-VL-32B by 6.2 percentage points. These results show substantial variation among open-weight models: Kimi K3 is competitive with frontier closed-source models in our in-context video memory setting, while the other evaluated models remain behind.

## 4.4 Analysis

Dependency Distance. Figure 4(a) groups dependent turns by the distance to their earliest parent. Mean accuracy across all evaluated models decreases from 30.2% at distance 1 to 23.1% at distances of at least five turns, with a larger decline in the longest-distance group. The model-specific trends difer: Gemini 3.8 Flash declines relatively steadily from 41.1% to 30.2%, whereas Qwen3.5-35B-A3B drops more sharply beyond distance 3, falling from 27.8% to 19.2%. These trends suggest that longer dependencies pose greater challenges for in-context video memory, although models difer in their sensitivity to dependency distance.

Number of Parents. Figure 4(b) shows a consistent decline in accuracy as the number of parent turns increases across all four models. Kimi K3 remains the strongest at every parent count, but its accuracy falls from 40.8% with one parent to 29.8% with three. GPT-5.4 and Qwen3.5-27B show the same trend, falling from 34.9% to 23.7% and from 30.1% to 17.3%, respectively. These results suggest that integrating information from multiple earlier turns poses an additional challenge for in-context video memory, even for the strongest model in this comparison.

Turn Position and Dependency. Table 3 shows that the mean of EpiDepAcc and AnsDepAcc is 13.1–17.3 percentage points below independent-turn accuracy across the five reported models, although this gap also reflects the stricter chain-crediting rule used by AnsDepAcc. Figure 4(c) further compares independent and dependent turns at diferent dialogue positions. For Gemini 3.8 Flash, dependent-turn accuracy decreases from 46.1% at turn 2 to 35.2% at turns 5 and beyond, while independent-turn accuracy declines more modestly, from 53.8% to 49.2%. Qwen3.5-27B exhibits a similar pattern: dependent-turn accuracy falls from 36.2% to 28.1%, compared with a decline from 46.1% to 40.3% on independent turns. Although independent-turn accuracy fluctuates at intermediate positions, dependent-turn accuracy declines consistently for both models. These patterns suggest that later dialogue positions are associated with greater dificulty, particularly when answering requires information from earlier turns.

Effect of Answer History. Figure 4(d) compares accuracy on answer-dependent turns under self-rollout and oracle histories, with independent-turn accuracy as a reference. Oracle histories replace previous model responses with reference answers. All three models benefit substantially: Qwen3-VL-8B improves from 12.3% to 32.4%, Qwen3-VL-32B from 17.2% to 37.3%, and Gemini 3.5 Flash from 29.7% to 47.2%. These gains indicate that errors in earlier responses are an important source of dificulty for subsequent answer-dependent questions. Nevertheless, oracle accuracy remains below independent-turn accuracy, suggesting that correct history alone is insuficient for reliable in-context video memory.

We further provide frame-count ablations with 16, 32, and 64 sampled frames in Appendix E.1 and additional dialogue examples in Appendix E.2.

## 5 Conclusion

We introduce VideoMemBench to evaluate video memory through explicit Episodic and Answer dependencies, distinguishing what a model has seen from what it has said, with automated verifications and human reviews. Evaluation of over 20 MLLMs shows that the best model achieves only 45.7% overall, with lower accuracy on longer-range and multi-parent dependencies. These findings highlight the dificulty of maintaining and using dialogue context in video conversations. We hope VideoMemBench supports the development of video MLLMs that reliably recall and build on earlier observations and responses.

## References

Anthropic. Introducing Claude Opus 5, 2026a. URL https://www.anthropic.com/news/claude-opus-5.

Anthropic. Introducing Claude Sonnet 5. https://www.anthropic.com/news/claude-sonnet-5, June 2026b.

Shuai Bai, Yuxuan Cai, Ruizhe Chen, Keqin Chen, Xionghui Chen, Zesen Cheng, Lianghao Deng, Wei Ding, Chang Gao, Chunjiang Ge, et al. Qwen3-VL technical report. arXiv preprint arXiv:2511.21631, 2025a.

Shuai Bai, Keqin Chen, Xuejing Liu, Jialin Wang, Wenbin Ge, Sibo Song, Kai Dang, Peng Wang, Shijie Wang, Jun Tang, et al. Qwen2.5-VL technical report. arXiv preprint arXiv:2502.13923, 2025b.

Gheorghe Comanici, Eric Bieber, Mike Schaekermann, Ice Pasupat, Noveen Sachdeva, Inderjit Dhillon, Marcel Blistein, Ori Ram, Dan Zhang, Evan Rosen, et al. Gemini 2.5: Pushing the frontier with advanced reasoning, multimodality, long context, and next generation agentic capabilities. arXiv preprint arXiv:2507.06261, 2025.

Chaoyou Fu, Yuhan Dai, Yongdong Luo, Lei Li, Shuhuai Ren, Renrui Zhang, Zihan Wang, Chenyu Zhou, Yunhang Shen, Mengdan Zhang, et al. Video-MME: The first-ever comprehensive evaluation benchmark of multi-modal llms in video analysis. In 2025 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pp. 24108–24118. IEEE, 2025.

Chaoyou Fu, Haozhi Yuan, Yuhao Dong, Yi-Fan Zhang, Yunhang Shen, Xiaoxing Hu, Xueying Li, Jinsen Su, Chengwu Long, Xiaoyao Xie, et al. Video-mme-v2: Towards the next stage in benchmarks for comprehensive video understanding. arXiv preprint arXiv:2604.05015, 2026.

Google. Gemini 3.1 Pro: A smarter model for your most complex tasks. https://blog.google/innovation-and-ai/ models-and-research/gemini-models/gemini-3-1-pro/, February 2026.

Google DeepMind. Gemini 3.5: Frontier intelligence with action. https://blog.google/innovation-and-ai/ models-and-research/gemini-models/gemini-3-5/, May 2026a.

Google DeepMind. Introducing Gemini 3.8 Flash and 3.8 Flash Cyber. https://blog.google/innovation-and-ai/ models-and-research/gemini-models/3-8-flash-and-3-8-flash-cyber/, September 2026b.

Cheng-Ping Hsieh, Simeng Sun, Samuel Kriman, Shantanu Acharya, Dima Rekesh, Fei Jia, Yang Zhang, and Boris Ginsburg. Ruler: What’s the real context size of your long-context language models? arXiv preprint arXiv:2404.06654, 2024.

Kairui Hu, Penghao Wu, Fanyi Pu, Wang Xiao, Xiang Yue, Bo Li, Yuanhan Zhang, and Ziwei Liu. Video-MMMU: Evaluating knowledge acquisition from multidisciplinary professional videos. In Proceedings of the 64th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pp. 27798–27828, 2026.

Alexis Huet, Zied Houidi, and Dario Rossi. Episodic memories generation and evaluation benchmark for large language models. In International Conference on Learning Representations, volume 2025, pp. 51409–51457, 2025.

Yuri Kuratov, Aydar Bulatov, Petr Anokhin, Ivan Rodkin, Dmitry Sorokin, Artyom Sorokin, and Mikhail Burtsev. Babilong: Testing the limits of llms with long context reasoning-in-a-haystack. Advances in Neural Information Processing Systems, 37:106519–106554, 2024.

Kunchang Li, Yali Wang, Yinan He, Yizhuo Li, Yi Wang, Yi Liu, Zun Wang, Jilan Xu, Guo Chen, Ping Luo, et al. MvBench: A comprehensive multi-modal video understanding benchmark. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pp. 22195–22206, 2024.

Yuanxin Liu, Shicheng Li, Yi Liu, Yuxiang Wang, Shuhuai Ren, Lei Li, Sishuo Chen, Xu Sun, and Lu Hou. TempCompass: Do video llms really understand videos? In Findings of the Association for Computational Linguistics: ACL 2024, pp. 8731–8772, 2024a.

Ziyu Liu, Tao Chu, Yuhang Zang, Xilin Wei, Xiaoyi Dong, Pan Zhang, Zijian Liang, Yuanjun Xiong, Yu Qiao, Dahua Lin, et al. MMDU: A multi-turn multi-image dialog understanding benchmark and instruction-tuning dataset for lvlms. Advances in Neural Information Processing Systems, 37:8698–8733, 2024b.

Adyasha Maharana, Dong-Ho Lee, Sergey Tulyakov, Mohit Bansal, Francesco Barbieri, and Yuwei Fang. Evaluating very long-term conversational memory of llm agents. In Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pp. 13851–13870, 2024.

Karttikeya Mangalam, Raiymbek Akshulakov, and Jitendra Malik. EgoSchema: A diagnostic benchmark for very long-form video language understanding. Advances in Neural Information Processing Systems, 36:46212–46244, 2023.

OpenAI. Introducing GPT-5.4. https://openai.com/index/introducing-gpt-5-4/, March 2026a.

OpenAI. Introducing GPT-5.4 mini and nano. https://openai.com/index/introducing-gpt-5-4-mini-and-nano/, March 2026b.

OpenAI. Introducing GPT-5.5. https://openai.com/index/introducing-gpt-5-5/, April 2026c.

OpenAI. GPT-5.6: Frontier intelligence that scales with your ambition. https://openai.com/index/gpt-5-6/, July 2026d.

Yaning Pan, Qianqian Xie, Guohui Zhang, Zekun Moore Wang, Yongqian Wen, Yuanxing Zhang, Haoxuan Hu, Zhiyu Pan, Yibing Huang, Zhidong Gan, et al. MT-Video-Bench: A holistic video understanding benchmark for evaluating multimodal llms in multi-turn dialogues. In Findings of the Association for Computational Linguistics: ACL 2026, pp. 8105–8126, 2026.

Aaditya Singh, Adam Fry, Adam Perelman, Adam Tart, Adi Ganesh, Ahmed El-Kishky, Aidan McLaughlin, Aiden Low, AJ Ostrow, Akhila Ananthram, et al. Openai GPT-5 system card. arXiv preprint arXiv:2601.03267, 2025.

Haoran Tan, Zeyu Zhang, Chen Ma, Xu Chen, Quanyu Dai, and Zhenhua Dong. MemBench: Towards more comprehensive evaluation on the memory of llm-based agents. In Findings of the Association for Computational Linguistics: ACL 2025, pp. 19336–19352, 2025.

Kimi Team, Tongtong Bai, Yifan Bai, Yiping Bao, Jianfeng Cai, Xinyuan Cai, Peizhou Cao, Yuxuan Cao, Ziwei Chai, Y Charles, et al. Kimi K3: Open frontier intelligence. arXiv preprint arXiv:2607.24653, 2026.

Qwen Team. Qwen3.5: Towards native multimodal agents, February 2026. URL https://qwen.ai/blog?id=qwen3.5.

Weihan Wang, Zehai He, Wenyi Hong, Yean Cheng, Xiaohan Zhang, Ji Qi, Ming Ding, Xiaotao Gu, Shiyu Huang, Bin Xu, et al. LVBench: An extreme long video understanding benchmark. In 2025 IEEE/CVF International Conference on Computer Vision (ICCV), pp. 22958–22967, 2025.

Di Wu, Hongwei Wang, Wenhao Yu, Yuwei Zhang, Kai-Wei Chang, and Dong Yu. LongMemEval: Benchmarking chat assistants on long-term interactive memory. In International Conference on Learning Representations, volume 2025, pp. 86809–86836, 2025.

Haoning Wu, Dongxu Li, Bei Chen, and Junnan Li. LongVideoBench: A benchmark for long-context interleaved video-language understanding. Advances in Neural Information Processing Systems, 37:28828–28857, 2024.

Haochen Xue, Feilong Tang, Ming Hu, Yexin Liu, Qidong Huang, Yulong Li, Chengzhi Liu, Zhongxing Xu, Chong Zhang, Chun-Mei Feng, et al. MMRC: A large-scale benchmark for understanding multimodal large language model in real-world conversation. In Proceedings of the 63rd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pp. 22477–22503, 2025.

An Yang, Anfeng Li, Baosong Yang, Beichen Zhang, Binyuan Hui, Bo Zheng, Bowen Yu, Chang Gao, Chengen Huang, Chenxu Lv, et al. Qwen3 technical report. arXiv preprint arXiv:2505.09388, 2025a.

Biao Yang, Bin Wen, Boyang Ding, Changyi Liu, Chenglong Chu, Chengru Song, Chongling Rao, Chuan Yi, Da Li, Dunju Zang, et al. Kwai Keye-VL 1.5 technical report. arXiv preprint arXiv:2509.01563, 2025b.

Jihan Yang, Shusheng Yang, Anjali W Gupta, Rilyn Han, Li Fei-Fei, and Saining Xie. Thinking in space: How multimodal large language models see, remember, and recall spaces. In 2025 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pp. 10632–10643. IEEE, 2025c.

Jingkang Yang, Shuai Liu, Hongming Guo, Yuhao Dong, Xiamengwei Zhang, Sicheng Zhang, Pengyun Wang, Zitang Zhou, Binzhu Xie, Ziyue Wang, et al. EgoLife: Towards egocentric life assistant. In 2025 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pp. 28885–28900. IEEE, 2025d.

Lianmin Zheng, Wei-Lin Chiang, Ying Sheng, Siyuan Zhuang, Zhanghao Wu, Yonghao Zhuang, Zi Lin, Zhuohan Li, Dacheng Li, Eric Xing, et al. Judging llm-as-a-judge with mt-bench and chatbot arena. Advances in neural information processing systems, 36:46595–46623, 2023.

## A Video Collection Details

Video sources. We assemble an initial pool of approximately 3.2k videos from five existing videounderstanding benchmarks and the web. For web collection, we search YouTube using queries spanning sports (“full match” and “match replay”), talks and lectures (“keynote” and “course lecture”), tutorials (“full lesson”), film recaps (“movie recap” and “film summary”), and cooking (“cooking lesson” and “chef pov”).

Domain taxonomy. The videos are organized into four broad categories, each comprising finer-grained domains:

• Lifestyle & Entertainment (130 videos). This category covers daily activities and entertainment, including Daily Life & Narrative, Cooking & Baking, Digital & Auto, Travel & Exploration, Variety & Comedy, Pets & Animals, Gaming & Play, and Other Lifestyle.

• Knowledge & Education (80 videos). This category spans scientific, technical, and social topics: Computer Science & AI, Physics & Math, Tech Product Launch, Nature & Geography, Politics & Global News, Economics & Law, Humanities & History, Mechanics Structures, and Chemistry & Materials, together with Other Knowledge.

• Art & Literature (66 videos). This category includes Painting & Handicraft, Film & Commentary, Performing Arts, Fashion & Beauty, Animation & Comics, Architecture & Design, and Other Art.

• Sports & Competition (50 videos). This category covers Basketball, Soccer, Combat Sports, Gym & Fitness, Outdoor & Extreme, and Other Sports.

## B Dependency Planning Details

Before question generation, the planner examines keyframes sampled across the video and produces a complete dialogue specification. The plan contains four components:

Video synopsis. The synopsis summarizes the main characters, events, and settings, providing the generator with global context for interpreting individual video segments.

Dialogue guidelines. The planner identifies which parts of the video to cover, which capabilities to assess, and how many turns to construct. Target capabilities include entity recall, event continuation, and cross-scene character tracking.

Dependency graph. The planner constructs a directed acyclic graph whose nodes represent dialogue turns. Each dependent turn is assigned one or more earlier parent turns, with each incoming edge labeled as Episodic or Answer. Specifying the graph before generation allows dependencies to be coordinated across the dialogue.

Per-turn intents. For each turn, the planner specifies what information it should establish and which video segment supports it. For dependent turns, the intent additionally identifies the information required from each parent, giving the generator an explicit basis for constructing the question.

The plan must satisfy two constraints. First, dependencies must be motivated by recurring entities, events, or details in the video, rather than imposed between unrelated topics. Second, the selected segments must span the video rather than concentrate on a local clip. The complete plan is passed to the generator as the specification for turn-wise generation.

## C Prompt Templates

We report the prompt templates used to construct and evaluate VideoMemBench, following the pipeline in Figure 2. During initial video curation, a pre-trained LLM first assigns each collected video a semantic domain and scene tags. In Stage 1, a Planner LLM watches each curated video and produces a dialogue-construction plan: a video introduction, a dialogue suggestion, the number of turns, a per-turn intent with its evidence window, and a typed dependency graph. In Stages 2 and 3, a Generator LLM writes the dialogue turn by turn according to the plan, and a Judge LLM applies three turn-level checks (dependency necessity, non-leakage, and visual grounding); turns that fail are returned to the Generator LLM with the judge’s feedback for regeneration. Each completed dialogue is then checked against the planner’s dependency graph (plan fidelity) and reviewed by human experts. We preserve runtime placeholders such as {video\_frames}, {plan}, and {parent\_turns}. In all prompts, an answer dependency means the child turn needs the parent turn’s correct answer, and an episodic dependency means the child turn only needs to relocate the same scene or entity as the parent turn.

## Video curation.

## Labeler LLM (Gemini 3 Flash): Domain and Scene Tagging

```jsonl
You are tagging videos for a multi−turn video memory benchmark.
You will see frames sampled uniformly across one video, each preceded by a <MM:SS> time tag.
Assign exactly ONE semantic domain from this list:
{domain_taxonomy}
Then describe the scene structure of the video.
Rules:
Choose the domain from what is shown on screen, not from the video title or file name.
A scene is a continuous stretch with the same location and the same main participants.
List every distinct scene with its time window.
Mark a video as unsuitable if it has fewer than 3 visually distinct scenes, if most frames are
static text or slides, or if the content cannot be verified from the frames alone.
Do not guess what happens between sampled frames.
Output ONLY JSON:
{
"domain": string, // one entry from the domain list
"domain_confidence": number, // 0..1
"scenes": [
{"scene_id": "S1", "t_start": number, "t_end": number,
"location": string, "main_entities": [string, ...], "summary": string}
],
"num_distinct_scenes": int,
"suitable": true|false,
"unsuitable_reason": string|null
}
```

## Stage 1: planning.

You are planning a multi−turn question−answering dialogue about one video for a benchmark   
that tests in−context video memory: later questions should only be answerable by remembering   
what earlier turns established.   
You will see frames sampled across the whole video, each preceded by a <MM:SS> time tag, and   
the video's domain and scene list from the labeler.   
Do NOT write questions or answers. Produce a PLAN that another model will follow.   
Step 1: Video introduction   
Write 2−4 sentences describing what happens in the video, in order, naming the main entities   
and scene changes. Only describe what is visible in the frames.   
Step 2: Dialogue suggestion   
Decide how many turns the dialogue should have (between {min\_turns} and {max\_turns}) and how   
it should move through the video. Pick at least three scene clusters that are far apart in time   
and describe the route: what is established first, what is revisited next, what is tracked last.

```jsonl
List the memory abilities the dialogue will exercise, chosen from: entity recall, event
continuation, clue−based reasoning, cross−scene tracking, spatial localization.
Step 3: Intent for every turn
For each turn, give:
turn_id (T1, T2, ...)
question_type: multi_choice_question, true_or_false_question, or open_structured
dependency: independent, or a list of parent turns, each typed as "answer" or "episodic"
evidence window [t_start, t_end] in seconds where the answer is visible
intent: one sentence saying what the turn should establish or test
Step 4: Dependency structure
Design the dependency graph with these rules:
"answer": the child cannot be answered without the parent's CORRECT answer.
Test: if the parent were answered wrongly, would the child's correct answer change? Only then use
"answer".
The parent of an "answer" edge may use any supported question type, provided that its answer
supplies information required by the child.
"episodic": the child only needs to return to the same scene, moment, or entity as the parent.
The parent's answer being wrong does not change the child's answer.
Parents must come earlier than the child. The graph must have no cycles.
More than half of the turns must have at least one parent.
At least one turn must have two or three parents.
At least one edge must span two or more turns (not only adjacent turns).
T1 must be independent and may use any supported question type.
Evidence windows must cover the beginning, middle, and end of the video.
Every edge must be genuine: the child's target must be hard to describe without the parent.
Do not connect turns that are about unrelated things.
Input:
domain: {domain}
scenes: {scenes}
frames: {video_frames}
Output ONLY JSON:
{
"video_intro": string,
"dialogue_suggestion": {
"num_turns": int,
"scene_route": string,
"abilities": [string, ...]
},
"turns": [
{"turn_id": "T1", "question_type": string,
"dependency": "independent" | [{"parent": "T#", "type": "answer"|"episodic"}, ...],
"evidence": {"t_start": number, "t_end": number},
"intent": string}
],
"dependency_graph": [{"from": "T#", "to": "T#", "type": "answer"|"episodic"}, ...]
}
```

## Stage 2: generation.

Generate LLM (Gemini 3.5 Flash): Turn Generation   
You are writing one turn of a multi−turn video QA dialogue, following a plan.   
You will see frames from the video with <MM:SS> time tags, the plan, and the turns already written.   
Write turn {turn\_id} so that it matches its planned intent, question type, dependency, and   
evidence window exactly.   
Build the answer first   
1. Look at the frames inside the planned evidence window and pick ONE specific fact you can   
clearly see (an identity, position, count, color, action, text, or change).   
2. That fact is the answer. If you cannot see a clear fact that serves the intent, say so in   
"plan\_issue" instead of guessing.   
3. Write a short question (one sentence, at most two) that asks for that fact.   
4. For multiple choice, write 4 options (A−D) of the same kind, each under 15 words.

Every wrong option should be plausible for this video and wrong for exactly one reason.   
5. Write answer\_reasoning: 1−2 sentences naming what in the frames makes the answer correct.   
Dependencies   
Refer to each parent turn inside the sentence by its turn id, for example   
"the player identified in T1" or "the shelf asked about in T3".   
For an "answer" parent, the question must be impossible to answer without knowing which   
option was correct in the parent. Refer to that result abstractly; never state it.   
For an "episodic" parent, the question returns to the same scene or entity. Give just enough   
visual anchors (appearance, location, nearby landmark) to find it again, but do not describe   
the parent's scene so fully that the parent becomes unnecessary.   
Never repeat a parent's correct option, and never use words that make it obvious.   
The answering model will see earlier questions but not earlier correct answers.   
Do not add dependencies that are not in the plan, and do not drop planned ones.   
Wording   
− No exact timestamps in the question; use scene descriptions ("when the scoreboard first   
appears").   
One question per turn. Natural language, no placeholders, no labels like "object A".   
true\_or\_false\_question: options are exactly A = "Yes", B = "No".   
open\_structured: no options; give a short reference answer and 1−3 acceptable paraphrases.   
Input:   
plan: {plan}   
turns written so far: {previous\_turns}   
turn to write: {turn\_plan}   
frames: {video\_frames}   
Output ONLY JSON:   
{   
"turn\_id": "T#",   
"type": "multi\_choice\_question"|"true\_or\_false\_question"|"open\_structured",   
"dependency": {"depends\_on": ["T#", ...], "dep\_type": {"T#": "answer"|"episodic"}} | null,   
"question": string,   
"options": {"A": string, "B": string, "C": string, "D": string} | null,   
"answer": string,   
"acceptable\_answers": [string, ...] | null,   
"answer\_reasoning": string,   
"evidence": [{"t\_start": number, "t\_end": number}],   
"plan\_issue": string|null   
}

## Generator LLM (Gemini 3.5 Flash): Regeneration of a Failed Turn

A turn you wrote was rejected by a quality check. Rewrite it so it passes, keeping everything   
the plan requires.   
You will see the plan entry for this turn, the turns written so far, the rejected turn, the   
name of the failed check, and the judge's reason.   
How to fix each failed check   
dependency\_necessity: the turn could be answered without its parent(s). Remove the details   
that let the target be found on its own, and refer to the parent turn instead, so that the   
parent is genuinely required.   
non\_leakage: the question re−described the parent's episode or revealed the parent's answer.   
Replace those words with a neutral reference to the parent turn by its turn id.   
visual\_grounding: the frames in the evidence window do not support the answer. Pick a fact   
that IS visible in that window and rebuild the question and answer around it.   
Keep fixed: turn\_id, question type, planned dependency and its types, evidence window.   
If the turn cannot be fixed without changing the plan, return the turn unchanged and explain   
in "plan\_issue".   
Input:   
plan entry: {turn\_plan}   
turns written so far: {previous\_turns}   
rejected turn: {turn}   
failed check: {check\_name}

judge reason: {judge\_reason}   
frames: {video\_frames}   
Output ONLY JSON with the same schema as Turn Generation.

## Stage 3: turn-level checks.

Judge LLM (Claude Opus 4.8): Dependency Necessity Probe   
You are answering one question about a video. Some earlier turns of the conversation have been   
removed on purpose. Answer only if the frames and the remaining conversation are enough.   
You will see frames from the video with <MM:SS> time tags, the conversation so far with some   
turns removed, and one question.   
Rules:   
If the question refers to an earlier turn that is not shown, you do not know what that turn   
established. Do not guess what it said.   
If you can still identify the answer with confidence from the frames alone, give it.   
If the answer depends on information you do not have, output "ABSTAIN".   
Input:   
frames: {video\_frames}   
conversation (parent turns removed): {history\_without\_parents}   
question: {question}   
options: {options}   
Output ONLY JSON:   
{"answer": string, "confidence": number, "why": string}   
// answer: an option letter, a short answer for open questions, or "ABSTAIN"   
Decision rule (applied by the pipeline, not shown to the probe): the turn is REJECTED if the probe   
answers correctly with confidence >= 0.5, because then the memory chain was decorative.

## Judge LLM (Claude Opus 4.8): Non-Leakage Check

You are checking whether a later question gives away its parent turns.   
You will see the later question next to each of its parent turns (question, options, and correct   
answer). You do not see the video.   
A question LEAKS if either condition holds:   
1. Answer leak: it states, paraphrases, or reveals any part of a   
parent's correct answer, even if the parent is still needed to   
recover the remaining information.   
2. Episode re−description: it repeats identifying details of the   
scene, moment, or entity established by an episodic parent,   
rather than referring to it without restating those details.   
Generic references such as "the player identified in T1" or   
"the moment discussed in T3" are allowed.   
For example, if a parent answer is "the flour in the red bowl",   
"How much of the flour identified earlier is used?" leaks the   
ingredient, even if the parent is still needed to identify the bowl.   
Leakage does not require making the parent unnecessary.   
Input:   
later question: {question}   
dependency types: {dep\_type}   
parent turns: {parent\_turns}   
Output ONLY JSON:   
{   
"leak": true|false,   
"answer\_leak\_parents": ["T#", ...],

```python
"redescribed_parents": ["T#", ...],
"offending_text": [string, ...],
"reason": string
}
```

## Judge LLM (Claude Opus 4.8): Visual Grounding Check

You are checking whether a question and its reference answer are supported by the video.   
You will see ONLY the frames from the turn's annotated evidence window, each with a <MM:SS> tag,   
and the question−answer pair.   
Steps:   
1. Before reading the reference answer, answer the question yourself from these frames.   
2. Compare your answer with the reference answer.   
3. Check that the question itself contains no facts that contradict the frames (wrong numbers,   
text, colors, or objects).   
4. For multiple choice, check that exactly one option is correct in these frames.   
The turn is SUPPORTED only if the frames clearly show the fact the reference answer states and   
no other option is also correct. If the key object or action is not visible in these frames,   
it is NOT supported, even if you believe it happens elsewhere in the video.   
Input:   
frames (evidence window): {evidence\_frames}   
question: {question}   
options: {options}   
reference answer: {answer}   
Output ONLY JSON:   
{   
"supported": true|false,   
"my\_answer": string,   
"support\_ts": [string, ...], // <MM:SS> tags of frames showing the fact   
"question\_factual\_errors": [string, ...],   
"multiple\_correct\_options": true|false,   
"reason": string   
}

## Dialogue-level check.

Judge LLM (Claude Opus 4.8): Plan Fidelity Check   
You are checking whether a finished dialogue was built the way it was planned.   
You will see the planner's per−turn intents and dependency graph, and the completed dialogue.   
You do not see the video.   
For every turn, read the question and decide which earlier turns it actually needs, and for each   
one whether it needs the parent's correct ANSWER or only the same scene or entity (EPISODIC).   
Base this on the question text, not on the dependency fields the generator wrote.   
Then compare this realized graph with the planned graph:   
missing edge: planned but the question does not actually need that parent   
extra edge: the question needs a parent that the plan did not include   
retyped edge: planned as "answer" but only episodic is needed, or the reverse   
Also flag any turn whose question does not serve its planned intent.   
The dialogue is REJECTED if any edge is missing, extra, or retyped, or if any turn does not   
serve its intent.   
Input:   
planned turns: {plan\_turns}   
planned dependency graph: {dependency\_graph}   
completed dialogue: {dialogue}

Table 4 Frame-count ablation on VideoMemBench under self-rollout. Each model is evaluated with 16, 32, and 64 uniformly sampled frames per video, using the same metrics as the 128-frame evaluation in Table 2. Models are ranked by the overall score within each frame budget, with the best result for each metric shown in bold.
<table><tr><td>Model</td><td>Frames</td><td>TurnAcc (%)</td><td>EpiDepAcc (%)</td><td>AnsDepAcc (%)</td><td>Overall (%)</td></tr><tr><td>Qwen3-VL-235B-A22B</td><td>16</td><td>33.7</td><td>32.3</td><td>11.7</td><td>25.9</td></tr><tr><td>Qwen3.5-122B-A10B</td><td>16</td><td>33.2</td><td>32.4</td><td>11.9</td><td>25.8</td></tr><tr><td>Qwen3.5-27B</td><td>16</td><td>33.9</td><td>33.7</td><td>8.2</td><td>25.3</td></tr><tr><td>Qwen3-VL-32B</td><td>16</td><td>30.6</td><td>28.9</td><td>10.5</td><td>23.3</td></tr><tr><td>Qwen3-VL-30B-A3B</td><td>16</td><td>29.6</td><td>30.6</td><td>7.6</td><td>22.6</td></tr><tr><td>Qwen3.5-35B-A3B</td><td>16</td><td>30.7</td><td>28.1</td><td>9.0</td><td>22.6</td></tr><tr><td>Qwen3-VL-8B</td><td>16</td><td>27.5</td><td>27.7</td><td>6.3</td><td>20.5</td></tr><tr><td>Qwen3.5-9B</td><td>16</td><td>28.0</td><td>26.5</td><td>6.8</td><td>20.5</td></tr><tr><td>Qwen3-VL-4B</td><td>16</td><td>26.7</td><td>26.5</td><td>8.0</td><td>20.4</td></tr><tr><td>Keye-VL-1.5-8B</td><td>16</td><td>22.0</td><td>22.5</td><td>4.7</td><td>16.4</td></tr><tr><td>Qwen2.5-VL-7B</td><td>16</td><td>21.4</td><td>21.3</td><td>3.7</td><td>15.5</td></tr><tr><td>Qwen2.5-VL-3B Qwen3.5-0.8B</td><td>16</td><td>20.2</td><td>20.1</td><td>3.7</td><td>14.7</td></tr><tr><td></td><td>16</td><td>17.8</td><td>18.0</td><td>3.9</td><td>13.2</td></tr><tr><td>Qwen3-VL-235B-A22B</td><td>32</td><td>36.3</td><td>37.8</td><td>14.5</td><td>29.5</td></tr><tr><td>Qwen3.5-122B-A10B</td><td>32</td><td>35.0</td><td>35.2</td><td>10.7</td><td>27.0</td></tr><tr><td>Qwen3.5-35B-A3B</td><td>32</td><td>34.5</td><td>34.1</td><td>12.3</td><td>27.0</td></tr><tr><td>Qwen3.5-27B Qwen3-VL-32B</td><td>32</td><td>34.9</td><td>34.2</td><td>11.7</td><td>26.9</td></tr><tr><td>Qwen3-VL-30B-A3B</td><td>32</td><td>34.1</td><td>32.6</td><td>13.1</td><td>26.6</td></tr><tr><td>Qwen3.5-9B</td><td>32</td><td>32.1</td><td>32.5</td><td>8.0</td><td>24.2</td></tr><tr><td>Qwen3-VL-8B</td><td>32</td><td>29.8</td><td>30.4</td><td>8.6</td><td>22.9</td></tr><tr><td>Qwen3-VL-4B</td><td>32</td><td>29.4</td><td>29.4</td><td>9.2</td><td>22.7</td></tr><tr><td>Qwen2.5-VL-7B</td><td>32</td><td>29.0</td><td>29.0</td><td>8.8</td><td>22.3</td></tr><tr><td>Keye-VL-1.5-8B</td><td>32 32</td><td>23.2 23.3</td><td>23.7</td><td>4.9</td><td>17.3</td></tr><tr><td>Qwen2.5-VL-3B</td><td>32</td><td>22.6</td><td>23.1 23.4</td><td>5.1 4.7</td><td>17.2</td></tr><tr><td>Qwen3.5-0.8B</td><td>32</td><td>20.1</td><td>21.8</td><td>4.7</td><td>16.9 15.5</td></tr><tr><td>Qwen3-VL-235B-A22B</td><td>64</td><td></td><td></td><td></td><td></td></tr><tr><td>Qwen3.5-122B-A10B</td><td>64</td><td>40.3 37.8</td><td>39.3</td><td>15.6 15.2</td><td>31.7 29.8</td></tr><tr><td>Qwen3.5-27B</td><td>64</td><td>37.9</td><td>36.5 36.7</td><td>14.8</td><td>29.8</td></tr><tr><td>Qwen3-VL-32B</td><td>64</td><td>37.8</td><td>36.1</td><td>14.5</td><td>29.5</td></tr><tr><td>Qwen3.5-35B-A3B</td><td>64</td><td>37.0</td><td>36.7</td><td>13.5</td><td>29.1</td></tr><tr><td>Qwen3-VL-30B-A3B</td><td>64</td><td>33.6</td><td>32.9</td><td>11.5</td><td>26.0</td></tr><tr><td>Qwen3-VL-8B</td><td>64</td><td>32.7</td><td>32.0</td><td>11.9</td><td>25.5</td></tr><tr><td>Qwen3.5-9B</td><td>64</td><td>33.4</td><td>33.2</td><td>9.6</td><td>25.4</td></tr><tr><td>Qwen3-VL-4B</td><td>64</td><td>31.3</td><td>30.3</td><td>12.1</td><td>24.5</td></tr><tr><td>Keye-VL-1.5-8B</td><td>64</td><td>24.5</td><td>24.9</td><td>6.3</td><td>18.5</td></tr><tr><td>Qwen2.5-VL-7B</td><td>64</td><td>24.1</td><td>22.7</td><td>6.4</td><td>17.7</td></tr><tr><td>Qwen2.5-VL-3B</td><td>64</td><td>22.5</td><td>22.5</td><td>5.1</td><td>16.7</td></tr><tr><td>Qwen3.5-0.8B</td><td>64</td><td>21.0</td><td>21.1</td><td>4.7</td><td>15.6</td></tr></table>

## D Human Review Details

## D.1 Dataset Review

We conduct a human review of all constructed multi-turn dialogues and dependency annotations before finalizing VideoMemBench. Eight experts participate in this review. As shown in Figure 5, our web interface presents the video alongside the dialogue in turn order. Each turn displays its question, reference answer, supporting timestamps, question type, and dependency links to earlier turns.

Reviewers verify five aspects: (1) answer correctness, whether the reference answer is supported by the video; (2) temporal grounding, whether the annotated evidence intervals contain the required visual evidence; (3) dependency validity, whether the annotated parent turns and dependency types match the question’s intended references; (4) necessity, whether the question becomes unanswerable without its parent turns, even with access to the video and the remaining dialogue; and (5) non-leakage, whether the question restates information that should instead be recalled from its parents. For turns with multiple parents, reviewers additionally assess whether each parent contributes indispensable information.

## D.2 Human Baseline Evaluation

The human baseline in Table 2 is evaluated separately from dataset review, using the same web interface configured for evaluation. In this mode, annotators see only the inputs available to models, with reference answers, evidence annotations, and dependency labels hidden. Twenty annotators evaluate all 326 dialogues, with each dialogue assigned to two annotators who answer independently. This yields 652 complete dialogue evaluations: 12 annotators each evaluate 33 dialogues, and the remaining eight each evaluate 32. Assignments are balanced across video categories, dialogue lengths, and dependency complexity. Where possible, annotators are assigned dialogues they did not previously review.

Evaluation inputs. Annotators receive the same 128 timestamped frames as the models, without access to the full video, reference answers, annotated evidence intervals, or dependency labels. Questions and answer options, where applicable, are presented sequentially. Each annotator can access the preceding questions and their own previous responses, but not the other annotator’s answers. Dialogues are evaluated in full rather than divided among annotators, preserving a separate response history for each evaluation.

Scoring and aggregation. Human responses are graded using the same correctness criteria as model responses. TurnAcc, EpiDepAcc, and AnsDepAcc follow the definitions in subsection 4.2. For AnsDepAcc, chain credit is computed separately within each annotator’s dialogue: a turn receives credit only if the current response and all of its Answer-dependency ancestors are correct. Scores are then aggregated across both evaluations of every dialogue, with equal weight given to each evaluation. Responses from diferent annotators are not combined through majority voting. Overall is the arithmetic mean of the three aggregated metrics.

## E Additional Results and Examples

## E.1 Effect of Frame Count

We evaluate 13 open-weight models with 16, 32, and 64 uniformly sampled frames per video under the same self-rollout protocol and scoring rules as the main experiments. Table 4 reports the results, complementing the 128-frame evaluation in Table 2.

Performance across frame budgets. Increasing the number of frames generally improves performance. All 13 models achieve higher Overall scores with 64 frames than with 16 frames. Qwen3-VL-235B-A22B ranks first at all three frame budgets, improving from 25.9% to 29.5% and 31.7% with 16, 32, and 64 frames, respectively. Its score rises further to 34.4% with 128 frames. Qwen3.5-27B similarly improves from 25.3% at 16 frames to 29.8% at 64 frames and 34.3% at 128 frames. These results suggest that denser temporal sampling provides useful visual evidence for answering questions throughout the dialogue.

Variation across models. The gains from additional frames vary considerably. From 16 to 64 frames, Qwen3- VL-32B improves by 6.2 percentage points overall, compared with 2.1 points for Keye-VL-1.5-8B. Increasing the frame budget from 32 to 64 improves Overall for 12 of the 13 models, with Qwen2.5-VL-3B showing a slight decrease from 16.9% to 16.7%. Qwen3.5-0.8B gains only 0.1 percentage points over the same interval. Thus, additional visual input generally benefits performance, but does not yield uniform gains across models.

Answer-dependent performance. AnsDepAcc remains below both TurnAcc and EpiDepAcc for every model at every frame budget in this ablation. For Qwen3-VL-235B-A22B, increasing the frame count from 16 to 64 raises AnsDepAcc from 11.7% to 15.6%, while Qwen3.5-27B improves from 8.2% to 14.8%. With 128 frames, their AnsDepAcc scores increase further to 18.6% and 20.9%, respectively. These gains show that additional visual evidence can improve correctness across answer-dependent chains, although the metric remains sensitive to both current-turn errors and upstream failures.

## E.2 Additional Dataset Examples

Figure 6 and Figure 7 present more sports video dialogues in which later questions revisit previously discussed players and teams, illustrating the combined challenges of visual recognition, cross-turn reference resolution, and reasoning.

![](images/3535c6e3a3a9d86f520b7f4de17181286f053fb0a5a5d5ee6cbbf3500474700e.jpg)  
Figure 5 The expert review interface. For each dialogue, the expert replays the video in the top panel, where the evidence window of the current turn is marked on the timeline, and reads the dialogue turn by turn, where the parent turns and the type of each dependency are highlighted. The expert answers the current question before the reference answer is revealed, then marks the turn as correct, answer wrong, dependency not required, wording leaks a parent, or duplicate, and edits the question, the options, or the answer in place. A dialogue is retained only after every turn has been marked correct or revised, and the interface records each decision together with the time spent on the turn. Referent here means episodic dependency. Please zoom in to see the details.

![](images/04d5cacb553bc0e2888674c31a97c40acc06e0da2e4eeaa5801ab5cb2859ec47.jpg)

<Question 1>: At kickoff, what brand is on the ponytailed player's jersey?   
<Ground truth 1>: Nike   
Qwen3-VL-2B: Next (wrong)   
Qwen3.5-122B: Nike (correct)   
GPT-5.6: Nike (correct)   
Gemini-3.1-Pro: Nike (correct)   
Claude-Sonnet-5: Nike (correct)   
<Question 2>: Is the player from T1 left or right of the center circle in the first attack?   
<Ground truth 2>: Left   
Qwen3-VL-2B: right (wrong)   
Owen3.5-122B: right (wrong)   
GPT-5.6: Right (wrong)   
Gemini-3.1-Pro: right (wrong)   
Claude-Sonnet-5: Right (wrong)   
<Question 3>: How does the player from T1 move relative to the center circle after T2?   
<Ground truth 3>: Moves away from the center circle   
Qwen3-VL-2B: left (wrong)   
Qwen3.5-122B: away from (correct)   
GPT-5.6: She moves away from the center circle into the attacking half. (correct   
Gemini-3.1-Pro: right (wrong)   
Claude-Sonnet-5: The Birmingham (T1) player dribbles the ball forward, moving away from the center circle and further to the right, advancing toward   
Nottingham Forest's half. (correct) <Question 4>: What does the ponytailed player wearing number 16 do near the penalty arc?   
<Ground truth 4>: Wipes the ball and throws it out.   
Qwen3-VL-2B: runs (wrong)   
Qwen3.5-122B: walks (wrong)   
GPT-5.6: She shoots the ball toward goal. (wrong)   
Gemini-3.1-Pro: runs and tackles down an opponent player (wrong)   
Claude-Sonnet-5: The ponytailed player wearing number 16 dribbles the ball into the penalty arc area and takes a shot on goal. (wrong)   
<Question 5>: Who scored Birmingham City's second goal?   
<Ground truth 5>: LEIDHAMMAR   
Qwen3-VL-2B: Lindsey von Flotow (wrong)   
Owen3.5-122B: Leidhammar (correct   
GPT-5.6: Ledrihamma (wrong)   
Gemini-3.1-Pro: Leidhammar (correct)   
Claude-Sonnet-5: Leidhammar (correct)   
<Question 6>: What color is the opposing goalkeeper's jersey when the player from T5 scores?   
<Ground truth 6>: Yellow   
Owen3-VL-2B: vellow (correct)   
Owen3.5-122B: vellow (correct   
GPT-5.6: Yellow (correct)   
Gemini-3.1-Pro: yellow (correct   
Claude-Sonnet-5: Yellow (correct)

<Question 7>: Per the match summary graphic, who won? <Ground truth 7>: Birmingham City Qwen3-VL-2B: Birmingham City (correct) Qwen3.5-122B: Birmingham City (correct) GPT-5.6: Birmingham City (correct) Gemini-3.1-Pro: Birmingham City (correct) Claude-Sonnet-5: Birmingham City won 5-1 (correct)

Figure 6 A soccer dialogue example from VideoMemBench. Seven dialogue turns cover player identification, spatial tracking, and match events, with later questions referring to previously discussed players. Reference answers are shown in red, and model responses marked correct at the turn level are shown in green.

![](images/e248ff6f195ebcb6434c5d44b4696c34fa7d90c63ea644dc4ed882afbacb616d.jpg)

![](images/7625f0f8502c852bee58e1316bcd963250696c1880c087171a0687bfe8d54c14.jpg)  
Figure 7 A basketball dialogue example from VideoMemBench. Selected video frames are shown above, followed by eight dialogue turns with reference answers and responses from five models. Reference answers are shown in red, and responses marked correct are shown in green.
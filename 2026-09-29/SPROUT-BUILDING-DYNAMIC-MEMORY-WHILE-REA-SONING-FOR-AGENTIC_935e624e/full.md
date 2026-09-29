# SPROUT: BUILDING DYNAMIC MEMORY WHILE REA-SONING FOR AGENTIC VIDEO UNDERSTANDING

Wei Chen<sup>1</sup>, Xuanyu Zheng<sup>2</sup>, Yancheng Long<sup>2</sup>, Haoyang Xu<sup>2</sup>, Kaiyu Jiang<sup>2</sup>, Bin Wen<sup>2</sup>, Tingting Gao<sup>2</sup>, Han Li<sup>2</sup>, Long Chen<sup>1,†</sup>

<sup>1</sup>HKUST <sup>2</sup>Kling AI

wchendb@connect.ust.hk, longchen@ust.hk

https://github.com/HKUST-LongGroup/Sprout

## ABSTRACT

Long video understanding relies on video memory to overcome the context limits of multimodal large language models. Existing methods follow a build-thenreasoning pipeline: memory is built offline for the entire video, then reasoned over as a static source. In practice a long video is shared by several questions, and this pipeline is costly at both ends: with few questions, building memory for the whole video costs far more than answering them; with many questions, the memory is never updated, so what is learned while answering questions is lost to the next question. To alleviate these, we introduce Sprout, an agentic framework that builds memory while reasoning: a temporal tree that sprouts detailed nodes as questions are answered. The agent watches the video segment by segment at a low frame rate, stopping when the current question can be answered, remembers each segment as a coarse node of the tree, and revisits key intervals at a higher frame rate to refine the tree with the recovered details. Once a segment is recorded as text, its video input is removed from the context history, while the original video remains reachable through the video tools. The memory tree and prior question– answer records persist across questions, so the memory is online and dynamic: built from the first question onward and updated by every question thereafter. We find that replacing accumulated video inputs with textual memory substantially reduces context usage while maintaining accuracy, with slight improvements in some settings. Across benchmarks on three models, Sprout achieves competitive or improved accuracy relative to representative offline memory methods, with no upfront construction stage and lower context cost per question.

## 1 INTRODUCTION

Advances in multimodal large language models (MLLMs) (Qwen Team, 2026; Google DeepMind, 2026; Keye Team, 2026) have extended video understanding from short clips to long recordings spanning hours, such as lectures, films, and days of egocentric activity (Fu et al., 2025; Wang et al., 2025a; Yang et al., 2025). Answering questions about such videos requires locating brief moments scattered across time and connecting the evidence among them. As duration grows, however, feeding the video directly into the model becomes infeasible: the number of visual tokens scales linearly with length and quickly exceeds what a context window can hold. Sampling more sparsely fits the budget but discards the brief moments that questions often hinge on, while sampling more densely preserves them at a cost the context cannot absorb. Long video understanding therefore relies on video memory: a representation of the video maintained outside the context, from which questionrelevant content is retrieved when needed.

Existing methods (QwenLM, 2026; Choi et al., 2026) construct video memory through a buildthen-reasoning pipeline (Figure 1a): they first build memory offline, then reason over the resulting static memory with an agentic framework. In the construction stage, the model processes the entire video without knowing future questions and stores it as persistent memory; In the reasoning stage, it retrieves relevant evidence from that memory to answer a given question. Prior work within this pipeline mainly differs in how the offline memory is organized. Hierarchical memories (Yang et al., 2025; Tian et al., 2025) recursively summarize short segments into multi-scale representations, enabling top-down retrieval from coarse summaries to fine segments. Graph-based memories (Long et al., 2026a; Yeo et al., 2026; Chen et al., 2026; QwenLM, 2026) model relations among events, entities, and time, answering questions through navigation and traversal over the resulting structure. Flat episodic memories (Choi et al., 2026) keep construction minimal by indexing clip captions with multiple retrieval keys and deferring semantic composition to inference time. Despite these differences, all share two properties. First, memory is built offline over the entire video before answering, requiring costly conversion of fine-grained details into memory. Second, memory is static: it is only read during inference and not updated, so the model’s observations and conclusions cannot persist across questions. In practice, a long video may be used to answer anywhere from a single question to a dozen or more. This leaves offline, static memory limited at both extremes: 1) upfront construction is wasteful when few questions are asked, since it must record every segment regardless of what is eventually asked; 2) a non-updatable memory prevents useful information from accumulating when many questions are asked, forcing each question to recover the same evidence from scratch.

![](images/05c0d21b7df1e4f5fc52834e6ee9af5da38d416c9664f72890d3446fe6cdb3cb.jpg)  
(a) Build-then-Reasoning

![](images/64148979387160cb3eb24d9d8365ae1db5554b1ac053f305df1ddd1f29dba934.jpg)  
(b) Per-question Token Cost

![](images/70f21ca8d2b2d435be2fcd1d50b65e12b14c0bd273eac6984cef20f61af030e7.jpg)  
(c) Build-while-Reasoning (Ours)  
Figure 1: (a) Offline memory is first built for the whole video before the first question; Q1–Q6 then reason over the same memory without updating it. Nodes used by a question take that question’s color, and the rest are never touched. (b) Per-question tokens with Gemini 3.8 Flash as the model. Direct feeds the video and question directly to the model. Offline memory is Qwen-MM-Plugins: its construction cost, amortized over the benchmark’s questions. Our method has no construction stage. (c) Sprout builds memory while reasoning. The agent watches the video segment by segment at a low frame rate and stops once the question has enough evidence, so the tail of the video is never watched and costs nothing. Each watched segment is remembered as a coarse textual node written into memory; revisits at a higher frame rate add child nodes beneath the intervals a question needs.

To address these limitations, we propose a new framework whose memory is built while reasoning rather than before it, following two principles. First, the memory is built online from what the model has watched: rather than recording every segment in advance, it grows as the model watches for each question, so no cost is paid for video that no question needs. Second, the memory is refined from coarse to fine: it starts as brief summaries of what happens in each segment, and whenever the model looks closer at a segment to answer a question, it adds the details it finds.

Based on the build-while-reasoning framework, we introduce Sprout, an agentic framework whose memory is a temporal tree that sprouts new nodes as questions are answered (Figure 1c), built through three actions: watch, remember, and revisit. Watch. Given a question, the agent streams the video segment by segment at a low frame rate, starting from the beginning of videos or where the memory left off, and stops once it judges that the segments the question needs have been found. Remember. As each segment is watched, the agent writes it into memory as a coarse node with a time interval and an event summary, forming the first level of a temporal tree. Once a segment is recorded, its video input is removed from the context history, allowing the agent to reason without that video chunk and greatly reducing context pressure. Revisit. When a retrieved node identifies a relevant video interval but lacks the needed detail, the agent revisits it at a higher frame rate, saves the recovered details as child nodes, and may search memory for related earlier information. The revisited frames are released from the context history in the same way.

Together, the three actions follow the two principles above. Watching and remembering build the memory online from the video the model has actually watched, and keep text rather than video in the context. Revisiting refines coarse nodes into finer ones while the model reasons. As a result, our memory is online and dynamic: it is built online after the first question arrives, even when the full video is available in advance, and it keeps growing as each question adds nodes, refines existing ones, and leaves its answer behind.

We evaluate Sprout with three models (GPT-5 (OpenAI, 2025), Qwen3.8-Max, and Gemini 3.8 Flash) on LVBench (Wang et al., 2025a), Video-MME Long (Fu et al., 2025), and Video-Holmes (Cheng et al., 2026). Its accuracy is competitive with or better than representative offline memory methods, while it pays no construction cost before the first question. It also uses fewer tokens (Figure 1b): with Gemini 3.8 Flash, it spends 71.7K and 99.3K tokens per question on LVBench and Video-MME Long, 74% and 55% fewer than direct inference.

In summary, our contributions are threefold:

• We identify two limitations of the build-then-reasoning pipeline: offline construction is wasteful for few-question videos, and static memory cannot accumulate insights from earlier reasoning.

• We propose Sprout, an agentic framework that builds memory while reasoning through three actions: it watches the video at a low frame rate, remembers each segment as a coarse node of a temporal tree, and revisits key intervals at a higher frame rate to add finer nodes.

• We further find that removing video frames from the context once they are recorded as text keeps the context small without reducing accuracy. Experiments with three models on benchmarks show that Sprout is more effective than an offline memory method.

## 2 RELATED WORK

Agentic Multimodal Reasoning. Rather than answering from what is already in the context, agentic multimodal methods call external tools during reasoning to gather the evidence they need. MM-REACT routes a language model to vision experts through a textual interface (Yang et al., 2023), while ViperGPT composes visual and language modules into executable programs (Sur´ıs et al., 2023). For video, VideoAgent iteratively gathers relevant visual information under the control of an LLM (Wang et al., 2024). DrVideo converts video into a coarse textual document and augments selected entries with additional visual information as its agent searches for missing evidence (Ma et al., 2025). More recently, VideoSeek provides overview, skim, and focus tools, allowing an agent to adapt its inspection granularity while reasoning over accumulated conversation history (Lin et al., 2026a). VideoSearcher instead trains the tool-invocation policy itself with reinforcement learning, unifying temporal localization, spatial focusing, and external multimodal search for open-world video questions (Gao et al., 2026). These methods show that tool use during reasoning can improve multimodal performance. More broadly, reliable multimodal reasoning rests on grounding outputs in explicit visual evidence: models comprehend relations among objects (Li et al., 2026), keep interleaved image–text content coherent (Chen et al., 2025a), suppress hallucinations unsupported by the image (Chen et al., 2025b), and ground reward judgments in fine-grained, region-level visual evidence (Long et al., 2026b; Yang et al., 2026b;a). Sprout connects this tool use to persistent memory management: each inspection updates a temporal node, its video inputs can then leave the context history, and the recorded information remains available to subsequent questions.

Efficient Long Video Understanding. Long video understanding requires balancing temporal coverage, visual detail, and input cost, as reflected in Video-MME, LVBench, and EgoLife (Fu et al., 2025; Wang et al., 2025a; Yang et al., 2025). One direction expands the amount of visual information a model can process: LongVILA combines context extension, long-video instruction tuning, and sequence parallelism to support longer video inputs (Chen et al., 2025c), and Keye-VL-2.0 adapts sparse attention to a multimodal backbone to process hour-level videos within a 256K-token context (Keye Team, 2026). Another direction reduces the representation size: LongVU adaptively compresses spatial and temporal visual tokens to fit long videos within a limited context (Shen et al., 2025a). At inference time, VideoTree allocates representation detail according to query relevance through adaptive expansion of a video tree (Wang et al., 2025b). Together, these approaches improve the trade-off between coverage and detail through model capacity, compression, or evidence selection. Our focus is how observation density and retained context evolve over an answering session. Sprout first builds coarse temporal nodes from low-frame-rate observations, then revisits relevant intervals at higher frame rates and writes the recovered details back to those nodes. Textual memory carries these observations into later reasoning steps, allowing the corresponding visual inputs to be released from the context history while the source video remains accessible.

Video Memory and Retrieval. Long-video methods keep a memory outside the context and answer by retrieving from it. This memory follows a build-then-reasoning pipeline: it is built before any question, in one pass or incrementally as the video arrives, and it stays static while questions are answered. Streaming methods such as MA-LMM and M3-Agent are the incremental case: their memory grows as frames arrive, but it is still formed without questions and only read at answer time (He et al., 2024; Long et al., 2026a). Within this pipeline, methods differ in how the memory is organized. Hierarchical memories summarize segments at multiple temporal scales and answer by top-down retrieval (Yang et al., 2025; Tian et al., 2025). Graph-based memories link events, entities, and time, then answer by search or traversal: WorldMM, MemDreamer, and Qwen-MM-Plugins retrieve over multi-scale or hierarchical graphs (Yeo et al., 2026; Chen et al., 2026; QwenLM, 2026); Vgent and EGAgent add verification or entity search on the graph (Shen et al., 2025b; Rege et al., 2026); HippoMM consolidates short-term observations into longer-term memory (Lin et al., 2026b). MERIT keeps the memory flat, indexing clip captions under multiple keys and assembling evidence at inference time (Choi et al., 2026). Caption-based methods such as LLoVi, Video-RAG, and iRAG let the question decide which text is fetched or refined (Zhang et al., 2024; Luo et al., 2025; Arefeen et al., 2024). Sprout builds the memory while reasoning, adding and refining nodes only where questions lead and keeping those nodes and earlier answers for later questions.

## 3 METHOD

In this section, we present Sprout, an agentic framework that builds video memory while reasoning. Unlike build-then-reasoning methods, which construct memory over the entire video before any question arrives, Sprout starts from an empty memory and builds it only while answering questions. Given a video V of duration T and questions $q _ { 1 } , \ldots , q _ { K }$ about it, the model answers each question over multiple rounds, calling tools (Appendix A) that perform three actions: watch attaches the next unwatched chunk of the video at a low frame rate, remember writes it into memory as text, and revisit watches a recorded interval again at a higher frame rate to add details. The memory is shared by all questions about the video, so later questions can reuse what earlier ones have built. Figure 2 illustrates this process on three questions.

## 3.1 DYNAMIC TEMPORAL MEMORY

The memory has three parts: a temporal tree of observations, a watch record of how far the video has been watched, and a question history of earlier questions and their answers.

Temporal tree. The tree is organized by time. Its root spans the whole video [0, T]. Every other node covers an interval $[ s _ { i } , e _ { i } ] \subseteq [ 0 , T ]$ and stores an event summary, observed facts, and key entities for that interval; the children of a node split its interval into shorter pieces in temporal order. Because each node keeps its interval, the video behind any node can be watched again. The tree grows with the three actions: watching and remembering add coarse nodes under the root in temporal order, and revisiting adds children under a node where a question needs more detail. At any moment, the tree is therefore deep only where questions have looked closely, and it ends where the unwatched video begins. Nodes store only what was observed in the video; questions and answers are kept in the question history, so that the nodes remain useful to unrelated later questions.

![](images/77a319d0b3233910ddb987723b055eb4538bf1c14ed761423c89b13fac520a3a.jpg)  
Figure 2: Pipeline of Sprout. Three questions about one video are answered in order, building one memory. Each row shows the rounds of one question: tool calls (circles) and their results (boxes). Video frames stay in the context for one round only; text results are kept. Q1 watches the first two chunks, remembers them as coarse nodes, and revisits $n _ { 2 }$ at a higher frame rate to add finer child nodes. Q2 reuses the memory: search returns a node and Q1’s answer, and a short revisit adds one more detail, without watching new video. Q3 finds that the memory does not cover what it asks, so it watches two more chunks. The last two chunks are never watched. The memory tree (bottom) is drawn on the same time axis as the video; node colors show which question added each node.

Access. The model does not see the tree directly. Its context holds the current question, the watch record, and a compact list of the question history; the content of the tree is reached only through search (Section 3.4). All writes go through tools that check the new nodes against the current tree, so that at every level the tree stays in chronological order within the watched video.

## 3.2 WATCH: COARSE TEMPORAL EXPLORATION

The agent watches new video with read next, either when the memory is empty, as at the first question, or when the memory does not yet cover what the question asks about. The tool takes no arguments: the runtime selects the next unwatched chunk from the watch record and attaches its frames at a low rate $f _ { \mathrm { c } } ,$ together with the speaker-labelled transcript of the chunk when audio is available. At this rate, the agent learns what happens and when at a small visual cost.

How far to watch is decided by the current question. After each chunk is remembered (Section 3.3), the agent has three options: watch the next chunk, revisit an interval that looks relevant, or answer. Watching can therefore stop before the end of the video, and dense revisits can already happen during the first question. Everything watched stays in memory for later questions.

## 3.3 REMEMBER: RECORDING OBSERVATIONS AND RELEASING FRAMES

Each watched chunk is written into memory right away. After read next, the chunk is pending: the agent must store it with write chunk memory in the next round before it can watch another chunk. The call passes a list of segments, each with an interval, a summary, and a set of facts. The runtime appends them as coarse nodes under the root and advances the watch record to the end of the chunk. The agent is instructed to write about 4–8 segments per chunk with 5–10 objective facts each, and to record only what is in the video, never the question, its options, or its reasoning.

Releasing frames. Once the model has responded to a chunk, its frames are removed from the context history before the next model call; the textual tool results and the model’s responses are kept. Frames therefore never pile up in the context: each chunk costs one call with frames and one text-only call to store it. Frames from revisits are removed in the same way.

To see the saving, consider t observations with $v _ { 1 } , \ldots , v _ { t }$ visual tokens and a textual context of length $B _ { t }$ . Keeping all frames and releasing them after use give context lengths

$$
L _ { t } ^ { \mathrm { r e t a i n } } = B _ { t } + \sum _ { j = 1 } ^ { t } v _ { j } , \qquad L _ { t } ^ { \mathrm { r e l e a s e } } = B _ { t } + v _ { t } .\tag{1}
$$

Each observation is still paid for once, but only the latest one stays in the context.

## 3.4 REVISIT: RETRIEVAL, DENSE INSPECTION, AND WRITE-BACK

Revisiting lets the agent return to video it has already watched. It has three steps: retrieve relevant nodes from the memory, watch their intervals at a higher frame rate, and write new details back.

Retrieval. Since the model does not see the tree, it searches the memory with search memory, which returns text only. A query is a set of literal, case-insensitive alternatives (e.g., "vest | waistcoat"), a time window, or both; a window alone returns everything stored in that span (Appendix B compares this literal matching with embedding retrieval). The result lists the matching facts with their timestamps under their nodes, followed by entries of the question history whose question mentions the query, so earlier answers come back together with the observations behind them. The agent is instructed to search before answering detail questions, and the number of searches per question is capped.

Dense inspection. When a retrieved node points to the right event but lacks the detail the question needs, the agent watches that part of the video again. view node attaches the whole interval of a node, and view time attaches an explicit interval inside the watched video, which the runtime clamps to the node that contains it. Both tools take a focus describing what to look for and a frame rate $f _ { \mathrm { r } } > f _ { \mathrm { c } }$ chosen by the agent, so dense frames are spent only on the chosen interval. This can reveal brief events or intermediate actions missed at the low rate.

Write-back. Each inspection tool is paired with a write-back tool that accepts only the interval just viewed. refine node splits the node just viewed with view node into contiguous children in temporal order that cover its interval. This is allowed only for a leaf, so an internal node can serve as evidence but its subtree is never rewritten. add observation stores a summary and facts for the interval just viewed with view time as a child of the node. In both cases, the parent keeps its description, so coarse context and new detail are kept side by side, and the revisited frames are removed as in Section 3.3. If the evidence is still not enough, the agent can search again, revisit other nodes, or watch more video. A question ends when the model calls answer or reaches the maximum number of rounds, in which case it must answer with the evidence it has.

## 3.5 ACCUMULATION ACROSS QUESTIONS

After a question is answered, the tree and the watch record are kept for the next question. The answer call also appends the question, its options, and the predicted answer to the question history; later questions see these entries in compact form and can retrieve them with search memory. A new question therefore starts from everything earlier questions have built: it can reuse existing nodes, refine them, or extend the tree into unwatched video. Since all watching and writing happens while questions are answered, the memory covers only what the questions have needed, and each question pays only for the video and detail it adds.

## 4 EXPERIMENTS

## 4.1 EXPERIMENTAL SETUP

Benchmarks and metric. We evaluate on three complementary benchmarks: LVBench (Wang et al., 2025a) contains 1,549 questions over 103 hour-scale videos; the Long subset of Video-

Table 1: Accuracy (%) on three video benchmarks. Rows are grouped into direct inference, agentic and memory methods, and Sprout; Video-MME(L) is the Long subset of Video-MME.
<table><tr><td>Method</td><td>Model</td><td>LVBench</td><td>Video-MME(L)</td><td>Video-Holmes</td></tr><tr><td>Direct</td><td>GPT-5</td><td>60.4</td><td>74.3</td><td>44.1</td></tr><tr><td>Direct</td><td>Qwen3.8-Max</td><td>81.8</td><td>80.4</td><td>67.3</td></tr><tr><td>Direct</td><td>Gemini 3.8 Flash</td><td>87.1</td><td>90.1</td><td>72.0</td></tr><tr><td>WorldMM (Yeo et al., 2026)</td><td>GPT-5</td><td>61.9</td><td>76.6</td><td>一</td></tr><tr><td>MERIT (Choi et al., 2026)</td><td>GPT-5</td><td>71.8</td><td>77.7</td><td></td></tr><tr><td>VideoSeek (Lin et al., 2026a)</td><td>GPT-5</td><td>68.4</td><td>81.2</td><td>47.3</td></tr><tr><td>Qwen-MM-Plugins</td><td>GPT-5</td><td>73.3</td><td>79.2</td><td>48.4</td></tr><tr><td>Qwen-MM-Plugins</td><td>Qwen3.8-Max</td><td>85.6</td><td>85.8</td><td>68.5</td></tr><tr><td>Qwen-MM-Plugins</td><td>Gemini 3.8 Flash</td><td>85.5</td><td>89.4</td><td>69.2</td></tr><tr><td>Sprout (ours)</td><td>GPT-5</td><td>73.4</td><td>80.4</td><td>56.7</td></tr><tr><td>Sprout (ours)</td><td>Qwen3.8-Max</td><td>86.9</td><td>86.0</td><td>71.6</td></tr><tr><td>Sprout (ours)</td><td>Gemini 3.8 Flash</td><td>88.2</td><td>91.2</td><td>76.8</td></tr></table>

Table 2: Accuracy (%) on EgoLifeQA. Ent., Event, Habit, Rel., and Task denote the EntityLog, EventRecall, HabitInsight, RelationMap, and TaskMaster categories; Overall is the accuracy over all questions, not the mean of the categories.
<table><tr><td>Method</td><td>Model</td><td>Ent.</td><td>Event</td><td>Habit</td><td>Rel.</td><td>Task</td><td>Overall</td></tr><tr><td>Direct</td><td>Gemini 2.5 Pro</td><td>43.2</td><td>40.5</td><td>41.0</td><td>55.2</td><td>52.4</td><td>46.4</td></tr><tr><td>Ego-R1 (Tian et al., 2025)</td><td>3B</td><td>51.2</td><td>53.2</td><td>63.9</td><td>50.4</td><td>50.8</td><td>53.0</td></tr><tr><td>M3-Agent (Long et al., 2026a)</td><td>7B</td><td>44.4</td><td>54.8</td><td>62.3</td><td>56.8</td><td>54.0</td><td>53.5</td></tr><tr><td>WorldMM (Yeo et al., 2026)</td><td>Qwen3-VL-8B</td><td>49.6</td><td>56.4</td><td>63.9</td><td>58.4</td><td>58.7</td><td>56.4</td></tr><tr><td>MERIT (Choi et al., 2026)</td><td>Ġemini 2.5 Pro</td><td>60.8</td><td>61.1</td><td>65.6</td><td>65.6</td><td>73.0</td><td>64.2</td></tr><tr><td>Qwen-MM-Plugins</td><td>Qwen3.8-Max</td><td>72.0</td><td>77.0</td><td>80.3</td><td>72.8</td><td>82.5</td><td>75.8</td></tr><tr><td>Qwen-MM-Plugins</td><td>Gemini 3.8 Flash</td><td>76.0</td><td>77.0</td><td>77.0</td><td>75.2</td><td>87.3</td><td>77.6</td></tr><tr><td>Sprout (ours)</td><td>Qwen3.8-Max</td><td>79.2</td><td>77.0</td><td>80.3</td><td>84.0</td><td>84.1</td><td>80.6</td></tr><tr><td>Sprout (ours)</td><td>Gemini 3.8 Flash</td><td>76.0</td><td>75.4</td><td>85.2</td><td>86.4</td><td>81.0</td><td>80.2</td></tr></table>

MME (Fu et al., 2025) contains 900 questions over 300 videos lasting 30–60 minutes; and Video-Holmes (Cheng et al., 2026) contains 1,837 questions over 270 short suspense videos, emphasizing reasoning across scattered visual clues. We also evaluate on EgoLifeQA (Yang et al., 2025), using the 500 questions of one participant (Jake) over a 44.3-hour recording spanning seven days. With hundreds of questions about the same recording, it tests how well the memory is reused. Table 1 reports the first three benchmarks and Table 2 reports EgoLifeQA.

Implementation. We run Sprout with three models: GPT-5, Qwen3.8-Max, and Gemini 3.8 Flash. The memory starts empty for each video and is kept across its questions. Questions arrive one at a time; the agent sees its own earlier answers, but not the ground truth or future questions. On LVBench and Video-MME Long, read next attaches 10-minute chunks at $f _ { \mathrm { c } } = 0 . 1$ fps; GPT-5 uses 5-minute chunks at the same rate to fit the image limit of its endpoint. On Video-Holmes, chunks are 30 seconds long at $f _ { \mathrm { c } } ~ = ~ 1$ fps. Revisits choose $f _ { \mathrm { r } } \in \lbrace 1 , 2 , 4 , 8 \rbrace$ fps and attach at most 768, 48, and 2,048 frames per view for Qwen3.8-Max, GPT-5, and Gemini, respectively. For comparison, Direct gives Qwen3.8-Max up to 768 sampled frames on Video-MME Long and 1 fps on Video-Holmes, and gives Gemini up to 3,072 frames. Each question may use at most 32 rounds. An ASR transcript is attached to each chunk on Video-MME Long, but not on LVBench or Video Holmes. As described in Section 3.1, the model does not see the tree: its context shows only the watch record and the question history, and the tree is reached through search memory. We call this setting the progress exposure. For Qwen-MM-Plugins, the memory is built once by Qwen3.7- Plus and shared by all three models.

![](images/dc0f683d80c9bf3de9b29227042ea35255a41e3fca670046458a44ab9f41c672.jpg)

![](images/bb91275b30763da9d4e4b1904fca16f062e98fc773dd0dd2a627c153583a84bb.jpg)  
(c) Context window of each method  
Figure 3: Effect of releasing frames with Gemini 3.8 Flash. (a) The context of one LVBench question in each round, followed by the model output of that round; the bar chart gives the measured context size of each round. Without releasing frames, every frame stays, so the context only grows and its last round reaches 47.8K tokens. Sprout releases frames once the model has written them as text, so its context rises and falls: the rounds that watch a chunk (r1, r3) have the most tokens, and even the peak (24.3K) is about half of the variant’s last round. (b) Context window and accuracy of the two variants. (c) Context window of Direct, Qwen-MM-Plugins, and Sprout. The context window is the peak context size over all rounds of a question (the single request for Direct), in thousands of tokens (K); we report its mean and maximum over all questions.

## 4.2 MAIN RESULTS

Long videos. Table 1 reports results on LVBench and Video-MME Long. Memory built offline does not always help: with Gemini 3.8 Flash, Qwen-MM-Plugins even falls below Direct. In contrast, Sprout matches or exceeds both with every model, although it builds no memory before the first question. With Gemini 3.8 Flash, it achieves new state-of-the-art results on both benchmarks. The gain over Direct is largest with GPT-5 (13.0 points on LVBench) and small with Gemini 3.8 Flash, so the memory helps most when the model cannot use a long context well.

Short videos. The advantage is larger on Video-Holmes, where Sprout achieves the best results with all three models. Offline memory again falls below Direct with Gemini 3.8 Flash, while Sprout exceeds Qwen-MM-Plugins by up to 8.3 points. These results show that Sprout is a general method that applies to both long and short videos.

Day-long video. Table 2 reports results on EgoLifeQA, whose videos are much longer than those above. With both Qwen3.8-Max and Gemini 3.8 Flash, Sprout outperforms Qwen-MM-Plugins overall, and its Qwen3.8-Max run achieves the best result in the table. This shows that Sprout also performs well on ultra-long videos.

## 4.3 RELEASING FRAMES AND CONTEXT WINDOW

As described in Section 3.3, Sprout releases frames from the context once they have been recorded as text. We now examine how this affects accuracy and the context window, with Gemini 3.8 Flash as the model. Since Sprout answers a question over multiple rounds, its context size changes from round to round (Figure 3a): a round that carries frames uses more tokens than the rounds after it, so the last round is not necessarily the one with the most tokens. We therefore take the peak context size over all rounds of a question as its context window, and report the mean and maximum of this value over all questions.

Releasing frames. Figure 3b shows that releasing frames does not hurt accuracy. We compare Sprout with a variant w/o releasing frames, which keeps every frame in the context until the question ends. Keeping frames enlarges the context window by up to 4.3×. Yet accuracy does not improve. The variant revisits less often because old frames remain visible, but what these frames provide is already recorded in memory or can be recovered by a revisit.

Table 3: Token cost per question with Gemini 3.8 Flash, in thousands of tokens (K) on LVBench / Video-MME(L), the Long subset of Video-MME. Construction builds the memory offline, amortized over the 1,549 and 900 questions of the two benchmarks; Answering sums input and output over all rounds of a question; Total is their sum, with the lowest in bold.
<table><tr><td>Method</td><td>Construction</td><td>Answering</td><td>Total</td></tr><tr><td>Direct</td><td></td><td>272.5/222.2</td><td>272.5/222.2</td></tr><tr><td>Qwen-MM-Plugins</td><td>75.2/210.0</td><td>95.3/83.7</td><td>170.5/293.7</td></tr><tr><td>Sprout (ours)</td><td></td><td>71.7/99.3</td><td>71.7/99.3</td></tr></table>

Table 4: Effect of the question history with Gemini 3.8 Flash; accuracy (%), with Video-MME(L) the Long subset of Video-MME. Direct answers each question independently. Causal answers the questions of a video in order: the whole video is attached to the first question, and each later question sees all earlier questions, the model’s reasoning, and its answers as text.
<table><tr><td>Method</td><td>LVBench</td><td>Video-MME(L)</td><td>Video-Holmes</td></tr><tr><td>Direct</td><td>87.1</td><td>90.1</td><td>72.0</td></tr><tr><td>Causal</td><td>87.0(-0.1)</td><td>90.9(+0.8)</td><td>75.9 (+3.9)</td></tr></table>

Context window. Releasing frames also keeps the context small throughout a question, as shown in Figure 3c. Direct places the whole sampled video in a single request of more than 200K tokens. Qwen-MM-Plugins keeps all retrieved text and frames across rounds, so its context grows as a question proceeds, and its largest question approaches the size of Direct. In contrast, Sprout watches each video segment only once and then removes it from the context history. Its mean window is similar to that of Qwen-MM-Plugins, but its largest window is several times smaller and stays below 55K tokens on both benchmarks. Therefore, recording each observation as text and releasing its frames keeps the context small without losing accuracy.

## 4.4 COST ANALYSIS

We next measure the total tokens a question uses over all its rounds, with Gemini 3.8 Flash (Table 3); for Qwen-MM-Plugins, we add its offline construction cost, spread over the questions of each benchmark. Sprout uses 74% and 55% fewer tokens than Direct on LVBench and Video-MME Long, since it watches each segment only once at a low frame rate and revisits only the intervals a question needs. Qwen-MM-Plugins answers with about as many tokens as Sprout, but it pays for construction no matter how many questions a video receives: on Video-MME Long, with only 3 questions per video, construction alone costs 210K tokens per question and pushes the total above Direct. In contrast, Sprout builds memory only as questions arrive, so its total cost is the lowest on both benchmarks.

## 4.5 EFFECT OF THE QUESTION HISTORY

Sprout keeps the questions of a video and its own answers in the question history, which later questions can see. To test whether this history helps on its own, we separate it from the memory tree (Table 4): Direct answers each question independently, while Causal answers the questions of a video in order and shows each later question the earlier questions, reasoning, and answers as text. Although these earlier answers may be wrong, Causal does not hurt accuracy. On Video-Holmes, where questions about one video build on the same clues, accuracy improves by 3.9 points; on LVBench and Video-MME Long, where questions are largely unrelated, the differences stay within 1 point. Earlier answers thus help when questions share content and do no harm otherwise, so keeping the question history is worth its small text cost.

## 5 CONCLUSION

We presented Sprout, an agentic framework that builds video memory while reasoning instead of before it. The agent watches the video in coarse chunks, remembers each chunk as a node of a temporal tree, and revisits the intervals a question needs at a higher frame rate, writing the recovered details back into the tree. Frames are released once they are recorded as text, and the tree and the question–answer records persist across questions, so the memory is built from the first question onward and refined by every question thereafter. Across three benchmarks and three models, Sprout matches or exceeds offline memory methods without an upfront construction stage, keeps the context window bounded by a single observation, and spends fewer tokens per question on long videos. Our experiments also show that releasing frames after use costs no accuracy and that earlier answers help later questions when they share content. Future work includes richer retrieval over the tree, such as multi-key indexing, and mechanisms for merging or forgetting nodes, which the current memory, like existing offline memories, does not have.

## REFERENCES

Md Adnan Arefeen, Biplob Debnath, Md Yusuf Sarwar Uddin, and Srimat Chakradhar. iRAG: Advancing RAG for videos with an incremental approach. In CIKM, pp. 4341–4348, 2024.

Cong Chen, Guo Gan, Kaixiang Ji, ZhaoYang Zhang, Zhen Yang, Guangming Yao, Hao Chen, Jingdong Chen, Yi Yuan, and Chunhua Shen. MemDreamer: Decoupling perception and reasoning for long video understanding via hierarchical graph memory and agentic retrieval mechanism. arXiv preprint arXiv:2606.07512, 2026.

Wei Chen, Lin Li, Yongqi Yang, Bin Wen, Fan Yang, Tingting Gao, Yu Wu, and Long Chen. CoMM: A coherent interleaved image-text dataset for multimodal understanding and generation. In CVPR, pp. 8073–8082, 2025a.

Wei Chen, Xin Yan, Bin Wen, Fan Yang, Tingting Gao, Di Zhang, and Long Chen. Decoupling contrastive decoding: Robust hallucination mitigation in multimodal large language models. In NeurIPS, 2025b.

Yukang Chen, Fuzhao Xue, Dacheng Li, Qinghao Hu, Ligeng Zhu, Xiuyu Li, Yunhao Fang, Haotian Tang, Shang Yang, Zhijian Liu, Ethan He, Hongxu Yin, Pavlo Molchanov, Jan Kautz, Linxi Fan, Yuke Zhu, Yao Lu, and Song Han. LongVILA: Scaling long-context visual language models for long videos. In ICLR, 2025c.

Junhao Cheng, Yuying Ge, Teng Wang, Yixiao Ge, Jing Liao, and Ying Shan. Video-Holmes: Can MLLM think like Holmes for complex video reasoning? In ECCV, 2026.

Yeeun Choi, Youngbeom Yoo, Joon-Young Lee, Hyolim Kang, and Seon Joo Kim. Keep it simple: Multi-key episodic memory retrieval for ultra-long video understanding. In ECCV, 2026.

Chaoyou Fu, Yuhan Dai, Yongdong Luo, Lei Li, Shuhuai Ren, Renrui Zhang, Zihan Wang, Chenyu Zhou, Yunhang Shen, Mengdan Zhang, Peixian Chen, Yanwei Li, Shaohui Lin, Sirui Zhao, Ke Li, Tong Xu, Xiawu Zheng, Enhong Chen, Caifeng Shan, Ran He, and Xing Sun. Video-MME: The first-ever comprehensive evaluation benchmark of multi-modal LLMs in video analysis. In CVPR, pp. 24108–24118, 2025.

Zhenkun Gao, Yicheng Bao, Jinlong Peng, Xueheng Li, Theo Huang, Bangwei Liu, Kunquan Li, Zhenye Gan, Tao Hu, Chengjun Xie, Mingqian Yang, Xuanhua He, Zhizhong Zhang, Xin Tan, Chengjie Wang, and Yuan Xie. VideoSearcher: Empowering video deep research with multi-tool agentic reasoning via reinforcement learning. arXiv preprint arXiv:2607.02927, 2026.

Google DeepMind. Gemini 3.8 Flash. Model card, 2026. URL https://deepmind.google/ models/model-cards/gemini-3-8-flash/.

Bo He, Hengduo Li, Young Kyun Jang, Menglin Jia, Xuefei Cao, Ashish Shah, Abhinav Shrivastava, and Ser-Nam Lim. MA-LMM: Memory-augmented large multimodal model for long-term video understanding. In CVPR, pp. 13504–13514, 2024.

Keye Team. Kwai Keye-VL-2.0 technical report. arXiv preprint arXiv:2606.10651, 2026.

Lin Li, Wei Chen, Jiahui Li, Kwang-Ting Cheng, and Long Chen. Relation-R1: Progressively cognitive chain-of-thought guided reinforcement learning for unified relation comprehension. In AAAI, 2026.

Jingyang Lin, Jialian Wu, Jiang Liu, Ximeng Sun, Ze Wang, Xiaodong Yu, Jiebo Luo, Zicheng Liu, and Emad Barsoum. VideoSeek: Long-horizon video agent with tool-guided seeking. In CVPR, 2026a.

Yueqian Lin, Jingyang Zhang, Qinsi Wang, Hancheng Ye, Yuzhe Fu, Yudong Liu, Hai Helen Li, and Yiran Chen. HippoMM: Hippocampal-inspired multimodal memory for long audiovisual event understanding. In CVPR Findings, pp. 5968–5977, 2026b.

Lin Long, Yichen He, Wentao Ye, Yiyuan Pan, Yuan Lin, Hang Li, Junbo Zhao, and Wei Li. Seeing, listening, remembering, and reasoning: A multimodal agent with long-term memory. In ICLR, 2026a.

Yancheng Long, Yankai Yang, Hongyang Wei, Wei Chen, Tianke Zhang, Haonan Fan, Changyi Liu, Kaiyu Jiang, Jiankang Chen, Kaiyu Tang, Bin Wen, Fan Yang, Tingting Gao, Han Li, and Shuo Yang. SpatialReward: Bridging the perception gap in online RL for image editing via explicit spatial reasoning. In ICML, 2026b.

Yongdong Luo, Xiawu Zheng, Xiao Yang, Guilin Li, Haojia Lin, Jinfa Huang, Jiayi Ji, Fei Chao, Jiebo Luo, and Rongrong Ji. Video-RAG: Visually-aligned retrieval-augmented long video comprehension. In NeurIPS, 2025.

Ziyu Ma, Chenhui Gou, Hengcan Shi, Bin Sun, Shutao Li, Hamid Rezatofighi, and Jianfei Cai. DrVideo: Document retrieval based long video understanding. In CVPR, pp. 18936–18946, 2025.

OpenAI. GPT-5 system card. System card, 2025. URL https://openai.com/index/ gpt-5-system-card/.

Qwen Team. Qwen3.8. Official blog, 2026.

QwenLM. Qwen-MM-Plugins: Video memory. Software and documentation, 2026. Accessed September 17, 2026.

Aniket Rege, Arka Sadhu, Yuliang Li, Kejie Li, Ramya Korlakai Vinayak, Yuning Chai, Yong Jae Lee, and Hyo Jin Kim. Agentic very long video understanding. In ACL, 2026.

Xiaoqian Shen, Yunyang Xiong, Changsheng Zhao, Lemeng Wu, Jun Chen, Chenchen Zhu, Zechun Liu, Fanyi Xiao, Balakrishnan Varadarajan, Florian Bordes, Zhuang Liu, Hu Xu, Hyunwoo J. Kim, Bilge Soran, Raghuraman Krishnamoorthi, Mohamed Elhoseiny, and Vikas Chandra. LongVU: Spatiotemporal adaptive compression for long video-language understanding. In ICML, volume 267, pp. 54582–54599. PMLR, 2025a.

Xiaoqian Shen, Wenxuan Zhang, Jun Chen, and Mohamed Elhoseiny. Vgent: Graph-based retrievalreasoning-augmented generation for long video understanding. In NeurIPS, 2025b.

D´ıdac Sur´ıs, Sachit Menon, and Carl Vondrick. ViperGPT: Visual inference via Python execution for reasoning. In ICCV, pp. 11888–11898, 2023.

Shulin Tian, Ruiqi Wang, Hongming Guo, Penghao Wu, Yuhao Dong, Xiuying Wang, Jingkang Yang, Hao Zhang, Hongyuan Zhu, and Ziwei Liu. Ego-R1: Chain-of-tool-thought for ultra-long egocentric video reasoning. arXiv preprint arXiv:2506.13654, 2025.

Weihan Wang, Zehai He, Wenyi Hong, Yean Cheng, Xiaohan Zhang, Ji Qi, Ming Ding, Xiaotao Gu, Shiyu Huang, Bin Xu, Yuxiao Dong, and Jie Tang. LVBench: An extreme long video understanding benchmark. In ICCV, pp. 22958–22967, 2025a.

Xiaohan Wang, Yuhui Zhang, Orr Zohar, and Serena Yeung-Levy. VideoAgent: Long-form video understanding with large language model as agent. In ECCV, 2024.

Ziyang Wang, Shoubin Yu, Elias Stengel-Eskin, Jaehong Yoon, Feng Cheng, Gedas Bertasius, and Mohit Bansal. VideoTree: Adaptive tree-based video representation for LLM reasoning on long videos. In CVPR, pp. 3272–3283, 2025b.

Jingkang Yang, Shuai Liu, Hongming Guo, Yuhao Dong, Xiamengwei Zhang, Sicheng Zhang, Pengyun Wang, Zitang Zhou, Binzhu Xie, Ziyue Wang, Bei Ouyang, Zhengyu Lin, Marco Cominelli, Zhongang Cai, Bo Li, Yuanhan Zhang, Peiyuan Zhang, Fangzhou Hong, Joerg Widmer, Francesco Gringoli, Lei Yang, and Ziwei Liu. EgoLife: Towards egocentric life assistant. In CVPR, pp. 28885–28900, 2025.

Yankai Yang, Yancheng Long, Wei Chen, Xingyu Lu, Hongyang Wei, Bin Wen, Fan Yang, Tingting Gao, Han Li, and Shuo Yang. SpatialFlow-GRPO: Where spatial credit drives image editing. arXiv preprint arXiv:2606.26872, 2026a.

Yankai Yang, Yancheng Long, Hongyang Wei, Wei Chen, Kaiyu Jiang, Changyi Liu, Jiankang Chen, Tianke Zhang, Haonan Fan, Kaiyu Tang, Bin Wen, Fan Yang, Tingting Gao, Han Li, and Shuo Yang. Joint reward modeling: Internalizing chain-of-thought for efficient visual reward models. arXiv preprint arXiv:2602.07533, 2026b.

Zhengyuan Yang, Linjie Li, Jianfeng Wang, Kevin Lin, Ehsan Azarnasab, Faisal Ahmed, Zicheng Liu, Ce Liu, Michael Zeng, and Lijuan Wang. MM-REACT: Prompting ChatGPT for multimodal reasoning and action. arXiv preprint arXiv:2303.11381, 2023.

Woongyeong Yeo, Kangsan Kim, Jaehong Yoon, and Sung Ju Hwang. WorldMM: Dynamic multimodal memory agent for long video reasoning. In CVPR, pp. 25599–25609, 2026.

Ce Zhang, Taowen Lu, Md Mohaiminul Islam, Ziyang Wang, Shoubin Yu, Mohit Bansal, and Gedas Bertasius. A simple LLM framework for long-range video question-answering. In EMNLP, pp. 21715–21737, 2024.

## APPENDIX

## A TOOL INTERFACE

Table 5 lists the tools that implement the three actions of Sprout (Section 3).

Table 5: Tool interface of Sprout. Frames attached by a tool stay in the context for the current round only; each write-back tool accepts only the chunk or interval watched in the preceding round.
<table><tr><td>Action</td><td>Tool (arguments)</td><td>Role</td></tr><tr><td>Watch</td><td>read_next ()</td><td>Attach the next unwatched chunk at the low frame rate  $f _ { \mathrm { c } } .$ </td></tr><tr><td>Remember</td><td>write_chunk_memory (segments)</td><td>Store the pending chunk as contiguous coarse memory nodes.</td></tr><tr><td rowspan="5">Revisit</td><td>search_memory (pattern, window)</td><td>Retrieve facts and prior questions by literal alternatives and/or a time window.</td></tr><tr><td>view_node (node_id, focus, fps)</td><td>Re-attach the whole interval of an existing node at the chosen frame rate  $f _ { \mathrm r }$ </td></tr><tr><td>view_time (start, end, focus, fps)</td><td>Attach a short interval inside the watched video at the chosen frame rate  $f _ { \mathrm r }$ </td></tr><tr><td>refine_node (node_id, segments)</td><td>Subdivide the leaf just viewed into contiguous children.</td></tr><tr><td>add_observation (summary, facts)</td><td>Attach what was just viewed with view_time as a timestamped child of its node.</td></tr></table>

## B LITERAL MATCHING VERSUS EMBEDDING RETRIEVAL

Sprout and Qwen-MM-Plugins both let the agent search a textual memory, but they match queries to memory in different ways.

How each method searches. In Sprout, search memory uses literal matching. A query is a list of alternatives separated by “|”, and a fact matches if it contains any alternative as a case-insensitive substring; an optional time window restricts the search to one span. The tool returns every matching fact in temporal order under its node, up to 40 facts, followed by up to five earlier questions that mention the query. Qwen-MM-Plugins uses embedding retrieval. When its memory is built, every node of the graph is encoded by a text embedding model (qwen3-vl-embedding). At query time, the query is encoded by the same model, nodes are ranked by cosine similarity and by BM25, the two rankings are fused, and the top 10 nodes are returned.

Why literal matching is enough. Current agent models are already strong at searching: they can choose keywords, list synonyms in one query, as in "vest | waistcoat | tank top", and rewrite the query when a search returns nothing. With such an agent, simple literal matching finds the facts it needs, so Sprout does not rely on an extra text embedding model. This saves both time and tokens. In Qwen-MM-Plugins, embedding the nodes alone costs 7.5M, 7.9M, and 1.1M tokens of construction on LVBench, Video-MME Long, and Video-Holmes, and every search waits for one more call to the embedding model. Literal matching instead runs directly on the current memory tree, with no embedding cost and no index to rebuild when new nodes are written.

Comparison. To compare the two directly, we replace literal matching in search memory with embedding retrieval, using the same text embedding model as Qwen-MM-Plugins, and keep everything else unchanged. Table 6 reports the results with Qwen3.8-Max. The two perform within one point of each other: literal matching is 0.8 and 0.4 points higher on LVBench and Video-MME Long, and embedding retrieval is 0.5 points higher on Video-Holmes. Literal matching therefore gives similar accuracy without the cost of an embedding model.

![](images/a98c9de1d6c2ae833d2b7c7dad79aac27bd5ef3b102e95ee2b453785e0160a51.jpg)

Table 6: Literal matching versus embedding retrieval in search memory, with Qwen3.8-Max. Accuracy (%); Video-MME(L) is the Long subset of Video-MME. The embedding variant uses the same text embedding model as Qwen-MM-Plugins (qwen3-vl-embedding); everything else is unchanged.
<table><tr><td>Search</td><td>LVBench</td><td>Video-MME(L)</td><td>Video-Holmes</td></tr><tr><td>Embedding retrieval</td><td>86.1</td><td>85.6</td><td>72.1</td></tr><tr><td>Literal matching (ours)</td><td>86.9</td><td>86.0</td><td>71.6</td></tr></table>

## C PROMPTS

Sprout uses one system prompt and one user turn per question; everything about individual tools (arguments, usage, return values) is sent as the tool schemas of Table 5. Prompt 1 is the system prompt, and Prompt 2 is the user turn that opens each question, with placeholders in angle brackets. The timestamp sentence follows the video length: videos of at most 60 minutes use MM:SS clock strings instead of HH:MM:SS. For GPT-5, the system prompt ends with one more sentence, “Call at most one tool per turn; wait for its result before choosing the next call.”, because its endpoint ignores the request option that disables parallel tool calls.

## Prompt 1: System prompt

![](images/36fea8b28b20c541035abb9d56d9cfa557a9a3e67c9ef547d7e4a28e14861f9f.jpg)  
Figure 4: The memory tree that Sprout builds on one LVBench video (Gemini 3.8 Flash, the video of Figures 5 and 6), after all 17 of its questions, all answered correctly. Each question has its own color. Q1 watches the first 10-minute chunk and records it as n1–n4; Q2–Q13 watch nothing further, Q14 watches the next two chunks, and Q17 the rest. Each child below a coarse node is written by a revisit and carries the color of the question that wrote it; Q2, Q3, Q4, Q6, Q7, and Q9 answer without writing anything. The child of n2 written by Q5 is the one shown in Figure 6.

![](images/a089bd160d5f43bee53735d015e56ca5e89b368bea183b5ba3d04d3bc8be2929.jpg)

## D QUALITATIVE EXAMPLES

Figure 4 shows the memory tree that Sprout builds on one LVBench video over all 17 of its questions, with Gemini 3.8 Flash and the prompts of Appendix C; all 17 are answered correctly. The video is watched only as far as the questions require, and every node below the coarse nodes is written by a revisit that a question triggers. Figures 5–9 follow nine of these questions: the first five, and the last four, where the questions move past the watched prefix. The frames are taken from the video at the timestamps shown. The first question watches the first chunk and records it; every later question starts from the memory left by the questions before it. Depending on what the memory already holds, a question is answered by one search (Q2), revisits the video and uses the frames as evidence (Q3, Q4, and Q17), revisits and writes a corrected or missing fact back into the tree (Q5, Q14, Q15, and Q16), or first watches further when the question lies past the watched prefix (Q14 and Q17).

## LVBench Z86xysw5Ncc (38:27), questions 1–3 of the video

## Q1 What category does this video belong to?

3 rounds 11.7K tokens

A. Olympic games ✓ B. The BBC News C. The NBA D. The country life vlog

![](images/dba2f87a6d42084af4c2068202b0d1d2fbe899a5d763e05c1ec1297ff6fd938a.jpg)

## ▷ write chunk memory(4 segments)

node 1 00:00–01:40 Introduction and start list for Men’s 100m Round 1 Heat 1 at London 2012 Olympic Games.   
node 2 01:40–05:20 Men’s 100m Round 1 Heat 1 race won by Tyson Gay.   
node 3 05:20–08:40 Men’s 100m Round 1 Heat 2 race won by Justin Gatlin.   
node 4 08:40–10:00 Women’s pole vault competition at London 2012 Olympic stadium.

Answer

▷ answer("A")

## Q2 How many meters is the race about men in the video?

2 rounds 5.0K tokens

A. 1500 B. 3000 C. 100 ✓ D. 50

Revisit

▷ search memory("race|meter|1500|3000|100|50|track|run")

5 facts in 4 nodes, led by node 1: “Introduction and start list for Men’s 100m Round   
1 Heat 1 . . .

Answer

▷ answer("C")

answered from memory alone, no frames

## Q3 How many athletes in Men’s 100M Start-List-Round 1 Heat 1?

4 rounds 12.7K tokens

A. 9 B. 7 C. 8 ✓ D. 10

Revisit

▷ search memory("Heat 1|Start-List|Round 1|Heat")

8 facts in 3 nodes, including @01:40 “Start list for Men’s 100m Round 1 Heat 1 is displayed on screen.” The count is not recorded.

Revisit

▷ view time(01:30, 02:00, fps=1) The end is clamped to 01:40, the end of node 1.

Revisit

▷ view time(01:40, 01:45, fps=1)

5 frames of node 2.

![](images/79b86ea9c7650e9d436cd5b585afe0bdfee6d52a5fa5ed5ffd79315aa57c60de.jpg)

![](images/e2235d3cfa261d9202ceb0d6ec06f44a1f1b2815e846a95796f310c1f70fc180.jpg)  
01:41  
frames used as evidence only, nothing written

▷ answer("C")

Figure 5: The first three questions on one LVBench video (Sprout with Gemini 3.8 Flash; Figure 6 continues with questions 4 and 5). Each round shows the tool call ( ▷ ) and, in the shaded block, its result. Q1 watches the first chunk and records it as four nodes; no later question in this figure watches further. Q2 is answered by one search. Q3 finds where the start list is shown but not its length, so it revisits that interval at 1 fps, first clamped at the boundary of node 1, and counts the athletes on the frames. Tool calls keep only their key arguments and tool results are abridged; memory text is verbatim from the request log, and token counts are total input tokens per question.

![](images/aac40c7909de8f98dc308f8582e9b6b5f77046153a1fbe55406cebad55538c19.jpg)  
Figure 6: Questions 4 and 5 on the video of Figure 5. Q4 finds that the bib number is not in memory, by pattern and then by time window, and revisits the whole node 2 at 1 fps to read it. Q5 revisits the start of the race, finds the gun at 03:56 rather than the 03:50 recorded at 0.1 fps, and writes the corrected time back as the child node 2 1.

![](images/ffbc5d4a2d2294fb44cc7bf74368724060b94bc4722205703971c953b251e9d5.jpg)  
Figure 7: Question 14 on the video of Figure 5. Q2–Q13 are all answered within the first chunk. Q14 asks about a moment after 10:00, so it watches and records the next two chunks, then revisits the start of Heat 4 at 1 fps to tell 20:28 from 20:38 and writes what it sees as node 9 1.

![](images/43c8a789e3876c135c62ab3c73fba873bce607933c59a437075fef0d013d850b.jpg)  
Figure 8: Questions 15 and 16 on the video of Figure 5. Q15 finds a winning time of 10.08s recorded at 0.1 fps, which is not among the options; the revisit reads 10.09s from the result graphic and writes it back as node 9 2. Q16 finds the winner already in memory and revisits the finish to confirm it before answering.

![](images/e8205ed7b325372f26453e9c3a5ecafbf23efadb5ec932d1aa03e44f8ce18ea4.jpg)  
Figure 9: Question 17 on the video of Figure 5. Q17 asks about Heat 6, whose race lies past the watched prefix, so it watches and records the last chunk; the new memory already names Su Bingtian third, and four revisits of the result graphics confirm it without writing anything.
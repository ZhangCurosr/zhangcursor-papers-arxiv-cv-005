# Towards Unified Evaluation of Prompt Enhancers for Video Generation

Yawen Shao1,2\*, Yubo Zhu2,3\*, Ziyun Dai2,4\*, Zixun Fang1,2\*, Kai Zhu²†‡ Zeyinzi Jiang2, Yufeng Ai2, Siyang Sun2, Haolan Xue2, Yu Shang2, Yuxiang Bao2 Zoubin Bi2, Jingming Luo2, Jie Xiao2, Chaojie Mao2, Zhehan Kan⁵, Hongchen Luo6 Yu Liu2, Sheng Zhong3, Wei Tong³‡, Xueyang Fu¹, Yang Cao¹, Wei Zhai¹, Zheng-Jun Zha1

1University of Science and Technology of China 2Wan Team, Alibaba Group 3Nanjing University 4Fudan University 5Tsinghua University 6Northeastern University

\*Equal contribution, †Project leader, Corresponding author

## Abstract

Modern video generators can realize increasingly complex visual narratives, positioning the prompt enhancer (PE) as a critical bridge from concise user instructions and multimodal references to structured cinematic plans. However, existing PE evaluation relies on rendered videos, imposing substantial computational and human costs, slowing PE training and iteration, and conflating PE quality with downstream generator behavior. To address this gap, we introduce PEBench, the first unified benchmark for direct PE evaluation across text-tovideo, image-to-video, and reference-to-video prompt enhancement. It comprises 1,100 expert-verified cases and 1,005 visual assets, spanning 35 fine-grained tasks with diverse temporal, cinematic, audiovisual, and multi-reference requirements. In addition, we develop PEBench evaluation, an evidence-grounded framework that combines modality-aware fact extraction with rubric-based assessment across 24 criteria. Our systematic evaluation of representative open- and closed-source PE methods reveals an emerging shift from finegrained descriptive expansion toward intent-preserving cinematic planning, while the caption-reconstruction and forward-refinement methods show complementary strengths in cinematic coverage and semantic fidelity or internal coherence, respectively. Human validation shows that PEBench scores align closely with expert judgments of enhanced prompts and downstream videos from Wan3.0 and MiniMax-H3, indicating that promptlevel evaluation reliably reflects downstream utility.

Project page: https://github.com/yawen-shao/PEBench

## 1 Introduction

Rapid advances in large-scale video generation models [1-3] have transformed content creation by enabling diverse video synthesis from natural-language instructions and multimodal references. Despite their remarkable capabilities, the quality and fidelity of generated videos remain strongly influenced by input conditions that adequately express user intent and generation requirements [4, 5]. Concise or ambiguous user instructions can lead generators to misinterpret the bindings between subjects and actions, the intended event order, or the roles of visual references, revealing a mismatch between user intent and model interpretation. A prompt enhancer (PE) addresses this mismatch by converting user instructions and conditioning inputs into an enhanced prompt that guides downstream generation, making PE quality an important factor in high-quality video synthesis [5].

However, PE development remains constrained by an indirect evaluation loop: validating each update requires downstream video synthesis followed by automatic scoring or blind human comparisons [5–10]. These computational and human costs delay feedback for training and refinement. Beyond this inefficiency, rendered videos reflect the joint influence of PE quality and generator capability, making observed errors difficult to attribute reliably to either component. These limitations motivate direct evaluation of enhanced prompts to assess PE quality independently of generator behavior and provide rapid, PE-specific feedback throughout iterative refinement.

![](images/284dff6fe66f2c5be5883b816b4631c4a6fa3257ea9df01513dbf7c3de4ee419.jpg)  
Figure 1 Overview of PEBench. (a) PEBench unifies T2V-PE, I2V-PE, and R2V-PE for direct prompt-level evaluation without video generation. (b) The dataset contains 1,100 expert-verified cases and 1,005 visual assets, covering diverse tasks, content, and conditioning complexity through a four-stage construction pipeline. (c) The evidence-grounded framework combines modality-aware evidence construction with rubric-based evaluation across 24 fine-grained criteria organized into Semantic Fidelity. Internal Consistency, and Directorial Creation.

In this paper, we introduce PEBench, the first unified benchmark for direct evaluation of video prompt enhancers. As illustrated in Figure 1(a), PEBench spans T2V-PE, I2V-PE, and R2V-PE, covering text-only, first-frame, and multireference conditioning. Figure 1(b) summarizes 1,100 expert-verified cases and 1,005 visual assets, covering 35 finegrained task types with broad variation in content, language, and conditioning complexity. To ensure diversity and quality, we employ a staged dataset construction pipeline combining broad scenario coverage with rigorous verification. Beyond dataset construction, the evidence-grounded framework in Figure 1(c) evaluates each enhanced prompt against the original instruction and modality-aware facts extracted from visual inputs where applicable. Its 24 fine-grained criteria form three complementary dimensions: Semantic Fidelity verifies the preservation of source requirements, Internal Consistency examines narrative, spatiotemporal, and audiovisual coherence, and Directorial Creation assesses the quality of creative guidance for realizing the intended video. A unified rubric-based evaluator reports criterion pass rates and dimension scores, yielding an interpretable profile of PE quality.

Using PEBench, we systematically evaluate representative open- and closed-source PE systems across T2V-PE, I2V-PE, and R2V-PE, revealing four key findings, most notably an emerging shift toward cinematic planning among industrydeveloped PE and observed links between specific training strategies and capability profiles. (i) Prompt enhancement is evolving from fine-grained descriptive expansion toward intent-preserving cinematic planning as video generators become increasingly capable. (ii) Caption-reconstruction methods achieve broader cinematic coverage, whereas forwardrefinement methods favor semantic preservation or internal coherence. (iii) Increasing temporal and reference complexity generally degrades PE performance. (iv) PEBench evaluation scores achieve Spearman correlations above 0.8 with expert rankings of enhanced prompts and downstream videos produced by Wan3.0 and MiniMax-H3, indicating that prompt-level evaluation reliably reflects downstream utility across generators.

## Our contributions are summarized as follows:

• We introduce PEBench, the first unified benchmark for direct evaluation of video prompt enhancers across T2V-PE, I2V-PE, and R2V-PE, comprising 1,100 expert-verified cases and 1,005 visual conditioning assets spanning 35 fine-grained task types.

Table 1 Comparison with existing video-generation benchmarks. Cine.: cinematic and camera control; Scripted: temporal or shot-level prompts; Audio Req.: explicit audio requirements in prompts; Multi-lang.: three or more languages; Video Ref.: conditioning on reference videos; Direct PE: direct prompt-level PE evaluation decoupled from downstream video generators.
<table><tr><td>Benchmark</td><td>T2V</td><td>I2V</td><td>R2V</td><td>Cine.</td><td>Scripted</td><td>Multi- shot</td><td>Audio Req.</td><td>Multi- lang.</td><td>Video Ref.</td><td>Direct PE</td><td># Metrics</td><td># Prompts</td><td># Ref. Assets</td></tr><tr><td>VBench [18]</td><td></td><td>x</td><td>x</td><td>x</td><td>X</td><td>x</td><td>x</td><td>X</td><td>X</td><td>x</td><td>16</td><td>~1,600</td><td></td></tr><tr><td>EvalCrafter [19]</td><td></td><td>X</td><td>X</td><td>x</td><td>X</td><td>x</td><td>x</td><td>x</td><td>X</td><td>x</td><td>17</td><td>700</td><td></td></tr><tr><td>Video-Bench [20]</td><td></td><td>X</td><td>X</td><td>x</td><td>x</td><td>x</td><td>x</td><td>X</td><td>X</td><td>x</td><td>9</td><td>419</td><td></td></tr><tr><td>OpenS2V-Nexus [21]</td><td></td><td>X</td><td></td><td>X</td><td>x</td><td>x</td><td>x</td><td>X</td><td>X</td><td>x</td><td>6</td><td>180</td><td>180</td></tr><tr><td>UniVBench [22]</td><td></td><td>x</td><td></td><td></td><td></td><td>v</td><td>x</td><td>X</td><td>X</td><td>x</td><td>21</td><td>200</td><td>864</td></tr><tr><td>MSAVBench [23]</td><td></td><td>X</td><td></td><td></td><td></td><td>V</td><td></td><td></td><td>X</td><td>X</td><td>20</td><td>286</td><td>165</td></tr><tr><td>StreamAV-Bench [24]</td><td></td><td>x</td><td>x</td><td>X</td><td></td><td>x</td><td>V</td><td>X</td><td>X</td><td>x</td><td>32</td><td>320</td><td></td></tr><tr><td>PEBench (Ours)</td><td></td><td></td><td></td><td></td><td></td><td>J</td><td></td><td>V</td><td></td><td>V</td><td>24</td><td>1,100</td><td>1,005</td></tr></table>

• We propose PEBench Evaluation, an evidence-grounded framework that combines modality-aware fact extraction for visually conditioned tasks with rubric-based assessment.

• We systematically evaluate representative PE, revealing a shift toward cinematic planning and capability differences associated with training strategies, while confirming human alignment and downstream utility across generators.

## 2 Related work

Prompt Enhancement for Video Generation. As video generators increasingly support temporally structured and multimodally conditioned content [1, 2, 11], prompt enhancers (PE) have become important for translating concise user intent into detailed instructions for downstream video generation. Existing approaches [12–14] broadly follow training-based or inference-time paradigms. Training-based methods learn prompt rewriting from constructed supervision or preference signals. VPO [6] and Prompt-A-Video [9] optimize LLM-based rewriters using supervised and preference objectives. WanPE [5] instead reconstructs detailed video captions from compressed prompts using videogrounded inverse pairs, while LingBot-Video-PE [15] maps expanded prompts into model-specific structured captions. In contrast, inference-time methods adopt diverse strategies: RAPO [8] retrieves relevant examples, while SCMaPR [7] performs agentic verification and correction. LTX-2.5-PE [16] directly rewrites prompts using its integrated Gemma 4 text encoder [17], whereas H3-Context-IR [3] employs a multi-stage multimodal workflow.

Benchmarks for Video Generation and Prompt Enhancement. Video-generation benchmarks have evolved from assessing T2V quality and prompt alignment [18–20] to covering reference-conditioned, multi-shot, audio-video, and streaming generation [21-24]. However, these benchmarks are designed to assess rendered videos rather than enhanced prompts. Applying them to PE evaluation requires an additional generation stage, increasing evaluation cost and latency while entangling PE performance with downstream generator behavior. As summarized in Table 1, PEBench instead evaluates enhanced prompts directly across T2V-PE, I2V-PE, and R2V-PE using standardized rubrics covering 24 criteria, enabling fine-grained assessment independent of downstream video generation while offering broad coverage of prompt structures and conditioning modalities at scale.

## 3 PEBench

## 3.1 Task Definition

A video prompt enhancer takes a user prompt, optional reference assets R, and a target duration as input and produces an enhanced prompt for a downstream video generator, as illustrated in Figure 1(a). PEBench evaluates this output directly without requiring video generation, thereby separating PE quality from downstream generator behavior. Specifically, it assesses whether the enhanced prompt preserves user intent and conditioning constraints, remains internally consistent, and provides sufficient detail to specify the intended video.

To provide comprehensive coverage, PEBench introduces a unified benchmark for evaluating prompt enhancers across three distinct task settings:

![](images/a311930be8b1e54750969094a471a21b1ce28b13781f1fc0619f292d2759086e.jpg)  
Figure 2 PEBench data construction pipeline. The four-stage pipeline comprises taxonomy design and composition, prompt synthesis and diversification, reference asset acquisition from public datasets and generative models, and automated verification followed by expert curation.

• Text-to-Video Prompt Enhancement (T2V-PE). Given a user prompt and a target duration, with no reference assets (R = ∅), the PE should produce a detailed prompt that faithfully expands the requested video content.

• Image-to-Video Prompt Enhancement (I2V-PE). Given a user prompt, a first-frame image $( \mathcal { R } = \{ I _ { 0 } \} )$ and a target duration, the PE should preserve the visible initial state and describe a coherent, instruction-aligned temporal continuation

• Reference-to-Video Prompt Enhancement (R2V-PE). Given a user prompt, a set of reference images or videos $( \mathcal { R } = \{ a _ { i } \} _ { i = 1 } ^ { N } )$ , and a target duration, the PE should incorporate each asset according to its specified role, such as subject, scene, style, motion, or cinematography, into a single coherent generation prompt.

## 3.2 Data Construction

To construct a diverse and reliable benchmark, we develop a four-stage pipeline that combines automated synthesis and verification with expert-driven design and curation, as illustrated in Figure 2.

Stage 1: Taxonomy design and composition. Domain experts define nine video genres and refine them into 35 fine-grained task types to establish broad and structured task coverage. They further specify five subject categories, eight scene domains, seven visual styles, and eight audio domains to guide scenario composition. Using the genre and these four additional dimensions, each scenario is represented as a (genre, subject, scene, style, audio) quintuple.

Stage 2: Prompt synthesis and diversification. We sample a scenario quintuple and associate it with a generation mode (T2V, I2V, or R2V), a fine-grained task type, and a target duration. Gemini 3.1 Pro [25] then synthesizes a user prompt from these specifications, jointly considering intent and modality, content composition, and duration planning. To reflect real-world prompting behavior, we vary instruction specificity from concise to detailed and temporal structure between single- and multi-shot formats, producing inputs ranging from short concepts to structured scripts.

Stage 3: Reference asset acquisition. We construct a diverse reference pool through two complementary routes. Public images and videos from existing datasets [21, 26–28] undergo modality-specific curation for quality, safety, and reference suitability. Model-generated references produced with Wan-Image [29], Seedream [30], and GPT Image 2 [31] follow image-guided and text-guided pipelines to create identity-consistent variants and scenario-aligned assets. The resulting pool supports single-image, multi-image, video-only, and hybrid image-video conditioning.

Stage 4: Verification and curation. Each candidate undergoes a two-stage quality-control process combining scalable model-based screening with expert review. Gemini 3.1 Pro first derives case-specific checks from the instruction and target duration to detect infeasible or inconsistent specifications. For reference-conditioned cases, it jointly inspects the instruction and candidate assets to verify role correspondence, script-asset alignment, reference coverage, and cross-asset consistency. Ten domain experts then refine the verified cases for clarity and difficulty, removing ambiguous, template-like, infeasible, or redundant content while preserving dataset balance and diversity.

(b) Subjects, Scenes, Styles Diversity  
(a) Video Domains  
![](images/35d14028a638cb438679b25395ab86856e80673bdec101c93fe6f9a1255d7fde.jpg)

(c) Diverse Language, Audio, Length  
![](images/4d6d5a58e18f0199a4f5462da6d4dc6557eddf24a86da362bc07e105d41d84a7.jpg)  
(d) Cinematic Language Distribution

![](images/62e3613466d10b5ad3bb6fb358dae6e221f83146df5bc50b34739b15623896b5.jpg)  
(e) Reference Asset Diversity

![](images/b8b15e089914c0de53ce88407bb0c01b2e0c3e3afd4947b6b2425028cec562ce.jpg)

![](images/2e5c93ed9e07ba4f52e769514eb2977f5743f6344555066ad9f2854bf0eacffe.jpg)  
Figure 3 Detailed Data Statistics and Diversity of PEBench. The benchmark covers a broad taxonomy of video-generation domains and fine-grained tasks (a), diverse subjects, scenes, and visual styles (b), multilingual prompts, audio conditions, and prompt lengths (c), rich cinematic requirements (d), and diverse reference assets (e). More statistics are in Appendix A.3.

## 3.3 Data Statistics

Scale and Diversity. PEBench comprises 1,100 prompts across three PE tasks: T2V-PE (500), I2V-PE (300), and R2V-PE (300). As shown in Figure 3(a,b), its taxonomy spans 9 video genres and 35 fine-grained task types, while its scenarios cover 5 subject categories (e.g., human and fantasy beings), 8 scene domains (e.g., domestic and nature), and 7 visual styles (e.g., photorealistic and 3D CGI), providing broad task, semantic, environmental, and aesthetic diversity.

Linguistic and Audio Diversity. In Figure 3(c), PEBench spans 12 languages (e.g., English and Chinese) and 8 audio domains (e.g., speech and vocalization, and music). Prompt lengths range from 6 to 4,120 tokens, encompassing concise instructions and long, temporally structured scripts.

Temporal and Cinematic Complexity. PEBench captures temporal and cinematic complexity through structured prompts containing 1–13 shots and rich cinematic language. As shown in Figure 3(d), it covers 3 shot scales (e.g., near and long shots), 5 camera angles (e.g., eye-level and low angles), 4 lens types (e.g., wide-angle and telephoto/zoom), and 3 camera-movement classes (e.g., pure translation and compound movement). Beyond these dimensions, the prompts also specify diverse composition patterns, transitions, lighting, and color schemes

Reference Asset Diversity. As shown in Figure 3(e), PEBench contains 1,005 visual conditioning assets, with 944 images and 61 videos. I2V-PE uses 300 first-frame images, while R2V-PE includes 644 images and 61 videos in image-only, video-only, and hybrid configurations, with up to 7 images or 2 videos per case. Beyond modality diversity, R2V-PE covers multi-reference structures (e.g., multi-view and multiple subjects) across realistic (54.6%), non-realistic (24.8%), and mixed or unspecified (20.6%) styles. Four representative cases illustrate multi-view characters, stylized frame sequences, photorealistic character-object-scene compositions, and multi-scene images paired with rhythmguiding videos, supporting compositional, sequential, and cross-modal evaluation.

## 3.4 Evaluation Framework

Evaluating prompt enhancers through generated videos conflates prompt quality with downstream generator behavior while incurring substantial computational cost and feedback latency, slowing training and iteration. PEBench thus directly evaluates enhanced prompts along three dimensions: Semantic Fidelity (SF), Internal Consistency (IC), and Directorial Creation (DC). These dimensions respectively assess fidelity to the inputs, internal coherence, and creative directorial planning for realizing the intended video. Figure 1(c) summarizes the 24 fine-grained criteria.

Semantic Fidelity. SF assesses whether the enhanced prompt preserves the original instruction and relevant information from visual references. Genre and Style Fidelity (GSF) measures retention of genre, medium, and visual style. Subject and Attribute Preservation (SAP) examines subject identity, appearance, and attribute bindings, while Action and Narrative Fidelity (ANF) covers actions, event order, and narrative intent. Dialogue and Lyrics Integrity (DLI) verifies verbal content, language, and speaker attribution, while Audio Requirement Preservation (ARP) covers speech, music, and sound effects. Camera Constraint Preservation (CCP), Lighting and Color Fidelity (LCF), Scene and Spatial Preservation (SSP), and On-Screen Text Integrity (OTI) respectively assess camera constraints, visual conditions, spatial relations, and visible text. In I2V-PE, the first frame anchors the initial state. In R2V-PE, each reference's instruction-specified role determines the asset facts evaluated.

Internal Consistency. Whereas SF measures fidelity to the inputs, IC evaluates the logical and spatiotemporal coherence of the enhanced prompt. Subject and Attribute Continuity (SAC) assesses the stability of subject identity and attributes. Action and Narrative Logic (ANL) evaluates event progression, causal relations, and subject interactions, while Temporal and State Continuity (TSC) examines event order and state transitions. Spatial Consistency (SPC) assesses locations, relative positions, and movement paths. Physical Logic (PL) checks whether events follow realworld physics or established fictional logic. Audiovisual Synchronization (AVS) evaluates the alignment between visual events and associated dialogue, music, and sound effects.

Directorial Creation. DC assesses whether the enhanced prompt provides relevant, explicit, and creative direction to specify the intended video across temporal, cinematic, narrative, and audiovisual aspects. Storyboard and Timing Completeness (STC) evaluates shot organization and duration allocation. Camera Design Specificity (CDS) examines framing, viewpoint, focus, and camera movement, while Editing and Transition Design (ETD) evaluates shot connections and editing logic. Lighting and Color Specification (LCS) assesses illumination and color planning. Narrative Development and Engagement (NDE) measures content progression and variation, while Core-Intent Focus (CIF) checks whether added content supports the original request. Sound Design Integration (SDI), Subject and Performance Specificity (SPS), and Visual Style Specification (VSS) respectively evaluate audio design, subject presentation, and visual-style articulation.

Evidence-Grounded Fine-Grained Evaluation. We formulate evaluation as a two-stage process combining modalityaware evidence construction with unified criterion-level assessment. First, we construct this evidence using modalityspecific extraction checklists. GPT-5.4 [32] extracts static visual facts from image-only inputs, while Qwen3.5-Omni-Plus [33] jointly analyzes all assets when video references are present, capturing static, temporal, audiovisual, and cross-asset evidence. Each fact is grounded in its source asset. The resulting ImageFacts for I2V-PE and AssetFacts for R2V-PE are constructed once and shared across methods. Second, GPT-5.4 serves as the unified evaluator across all three PE tasks. Given the original instruction, enhanced prompt, and applicable visual evidence, it assesses all 24 criteria using fixed criterion-specific rubrics and structured checklists. Shared evidence and fixed rubrics across methods enable consistent, interpretable criterion-level comparisons within each task setting. See Appendix B for more details.

## 4 Experiments

## 4.1 Experimental Setup

We evaluate ten PE systems: H3-Context-IR [3], WanPE-397B and WanPE-35B [5], LTX-2.5-PE [16], LingBot-Video-PE [15], SCMaPR [7], VPO [6], Prompt-A-Video-Cog and Prompt-A-Video-OS [9], and RAPO [8]. All systems are evaluated on T2V-PE, with multimodal-capable systems also evaluated on I2V-PE and R2V-PE.

## 4.2 Main Results

Table 2 reports PE performance across all three tasks. On T2V-PE, different methods lead in three dimensions, while WanPE-397B presents a balanced profile by ranking first in Directorial Creation and second in Semantic Fidelity. On the visually conditioned tasks, H3-Context-IR ranks first in all three dimensions on both I2V-PE and R2V-PE. No single method therefore dominates across all tasks and dimensions, revealing distinct capability profiles among current PE.

Finding 1: Modern prompt enhancement is evolving from fine-grained descriptive expansion toward intent-preserving cinematic planning, enabled by increasingly capable video generators.

Modern video generators can follow richer, more structured instructions for temporal progression, camera design narrative development, and audiovisual coordination within a coherent generation plan, expanding the role of PE beyond local detail enrichment. This shift is evident in large-scale, industry-developed PE: WanPE establishes a clear lead in Directorial Creation on T2V-PE, while H3-Context-IR leads by substantial margins on I2V-PE and R2V-PE. Appendix B.3 contrasts descriptive expansion with structured cinematic planning across PE systems. Beyond this emerging industrial trend, Internal Consistency remains uniformly high, varying by less than 9 points within each task, whereas Semantic Fidelity and Directorial Creation each span more than 70 points on T2V-PE. These results indicate that the central challenge lies not merely in expanding prompts or maintaining internal coherence, but in cinematic planning that preserves user intent.

Table 2 Main results on PEBench. Fine-grained metrics report criterion-level pass rates, defined as the percentage of cases without corresponding issues. The Overall scores aggregate criterion-level issues according to their number and severity in each case within each dimension. The best and second-best results for each task are highlighted in bold and underlined, respectively.
<table><tr><td rowspan="2">Method</td><td colspan="8">Semantic Fidelity</td><td colspan="8">Internal Consistency</td></tr><tr><td colspan="10">|GSF ↑ SAP ↑ ANF ↑ DLI ↑ ARP ↑ CCP ↑ LCF ↑ SSP ↑ OTI ↑ Overall ↑|SAC ↑ ANL ↑ TSC ↑ SPC ↑ PL ↑ AVS ↑ Overall ↑</td><td colspan="8"></td></tr><tr><td colspan="10"></td><td colspan="10"></td></tr><tr><td>WanPE-397B</td><td>89.20</td><td>70.00</td><td>80.80</td><td>94.20</td><td>62.00</td><td>57.40</td><td>80.40</td><td>80.60</td><td>94.80</td><td></td><td>74.89</td><td>91.20</td><td>87.40</td><td>94.60</td><td>82.40</td><td>92.80</td><td>94.40</td><td>91.51</td></tr><tr><td>WanPE-35B</td><td>86.40</td><td>71.80</td><td>82.60</td><td>88.00</td><td>59.60</td><td>54.40</td><td>73.20</td><td>81.20</td><td>97.00</td><td>72.83</td><td></td><td>91.60</td><td>89.20</td><td>92.40</td><td>81.60</td><td>91.60</td><td>90.20</td><td>91.40</td></tr><tr><td>H3-Context-IR</td><td>81.40</td><td>66.40</td><td>70.40</td><td>93.00</td><td>58.40</td><td>49.20</td><td>76.80</td><td>81.20</td><td>95.40</td><td>67.34</td><td></td><td>97.20</td><td>95.40</td><td>96.80</td><td>98.00</td><td>91.80</td><td>90.80</td><td>94.50</td></tr><tr><td>SCMaPR</td><td>94.60</td><td>87.60</td><td>85.60</td><td>80.20</td><td>82.00</td><td>87.20</td><td>90.60</td><td>92.60</td><td>95.00</td><td>87.92</td><td>94.60</td><td></td><td>92.60</td><td>96.80</td><td>91.40</td><td>93.60</td><td>95.60</td><td>95.06</td></tr><tr><td>LTX-2.5-PE</td><td>83.00</td><td>78.00</td><td>76.20</td><td>80.40</td><td>75.20</td><td>43.00</td><td>72.20</td><td>82.20</td><td>91.20</td><td>71.21</td><td></td><td>88.60</td><td>80.60</td><td>85.60</td><td>77.60</td><td>89.80</td><td>91.00</td><td>87.52</td></tr><tr><td>LingBot-Video-PE</td><td>67.80</td><td>28.00</td><td>30.40</td><td>72.00</td><td>54.00</td><td>16.80</td><td>28.00</td><td>45.40</td><td>84.60</td><td>41.61</td><td></td><td>91.20</td><td>91.00</td><td>93.80</td><td>66.60</td><td>92.40</td><td>93.20</td><td>90.96</td></tr><tr><td>VPO</td><td>74.00</td><td>65.00</td><td>45.60</td><td>76.20</td><td>63.40</td><td>45.00</td><td>66.00</td><td>66.40</td><td>89.00</td><td>62.80</td><td></td><td>92.60</td><td>90.00</td><td>97.20</td><td>91.40</td><td>93.00</td><td>94.00</td><td>95.07</td></tr><tr><td>Prompt-A-Video-Cog</td><td>84.60</td><td>47.60</td><td>35.40</td><td>78.60</td><td>69.20</td><td>29.00</td><td>58.60</td><td>44.40</td><td>91.20</td><td>56.48</td><td>82.40</td><td></td><td>81.80</td><td>93.40</td><td>86.20</td><td>89.40</td><td>89.80</td><td>91.09</td></tr><tr><td>Prompt-A-Video-OS</td><td>81.20</td><td>46.60</td><td>30.40</td><td>76.80</td><td>67.20</td><td>33.20</td><td>56.20</td><td>48.00</td><td>89.80</td><td>54.83</td><td>86.00</td><td></td><td>85.80</td><td>96.40</td><td>90.00</td><td>93.40</td><td>93.00</td><td>93.87</td></tr><tr><td>RAPO</td><td>37.20</td><td>6.40</td><td>1.00</td><td>72.60</td><td>52.40</td><td>24.20</td><td>25.40</td><td>4.80</td><td>79.00</td><td>15.47</td><td>80.80</td><td>64.00</td><td></td><td>88.20</td><td>76.80</td><td>93.80</td><td>93.80</td><td>88.30</td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td>I2V-PE</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>H3-Context-IR</td><td>64.67</td><td>56.67</td><td>67.00</td><td>89.00</td><td>47.33</td><td>52.00</td><td>87.67</td><td>76.33</td><td>81.00</td><td>58.59</td><td>97.67</td><td></td><td>96.33</td><td>97.33</td><td>99.33</td><td></td><td>94.33</td><td></td></tr><tr><td>LTX-2.5-PE</td><td>84.67</td><td>56.33</td><td>71.00</td><td>81.67</td><td>65.00</td><td>24.00</td><td>78.00</td><td>72.00</td><td>71.67</td><td>56.77</td><td>85.67</td><td></td><td>80.00</td><td>88.00</td><td>76.33</td><td>89.00 86.33</td><td>90.33</td><td>95.05 87.36</td></tr><tr><td>LingBot-Video-PE</td><td>72.67</td><td>54.33</td><td>23.33</td><td>77.00</td><td>57.33</td><td>10.67</td><td>42.33</td><td>73.00</td><td>70.67</td><td>45.70</td><td>95.67</td><td>96.33</td><td></td><td>90.00</td><td>66.67</td><td>89.67 95.00</td><td></td><td>91.14</td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>H3-Context-IR</td><td></td><td></td><td></td><td></td><td>56.67</td><td>51.00</td><td>76.33</td><td>R2V-PE</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>LTX-2.5-PE</td><td>73.33 61.67</td><td>54.00 43.67</td><td>65.00</td><td>83.33 70.00</td><td>71.00</td><td>27.67</td><td></td><td>73.33 58.00</td><td>82.00 75.67</td><td>58.11</td><td></td><td>87.33</td><td>90.00</td><td>98.67</td><td>95.33</td><td>94.67</td><td>91.67</td><td>94.90</td></tr><tr><td></td><td></td><td>62.67</td><td></td><td></td><td></td><td></td><td>64.00</td><td></td><td></td><td>49.03</td><td></td><td>76.33</td><td>79.33</td><td>88.67</td><td>74.67</td><td>93.67</td><td>82.67</td><td>86.66</td></tr><tr><td>Method</td><td></td><td colspan="10"></td><td colspan="8"></td></tr><tr><td></td><td></td><td></td><td></td><td>CDS ↑</td><td></td><td>ETD↑</td><td>LCS ↑</td><td></td><td>NDE↑</td><td>Directorial Creation</td><td>CIF ↑</td><td></td><td></td><td></td><td>VSS ↑</td><td></td><td></td><td></td></tr><tr><td></td><td colspan="10">STC ↑</td><td colspan="8">SDI↑ SPS ↑</td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td>T2V-PE</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td>Overall ↑</td><td></td></tr><tr><td>WanPE-397B WanPE-35B</td><td></td><td>66.60 59.80</td><td></td><td>81.00 80.20</td><td></td><td>63.20 63.60</td><td>80.20 75.60</td><td></td><td>68.60 66.60</td><td></td><td>91.20 94.20</td><td>77.60 83.00</td><td></td><td>81.60 80.40</td><td></td><td>82.20 78.60</td><td></td><td>78.97 78.15</td></tr></table>

Finding 2: PE trained to reconstruct real-video captions exhibits broader coverage of cinematic language, while forward-refinement methods excel at semantic preservation or internal coherence.

The dimension-level results on T2V-PE reveal a clear correspondence between enhancement paradigms and capability profiles. WanPE-397B and WanPE-35B are trained on pairs constructed by annotating real videos with a video captioner and reconstructing compatible user requests from the resulting captions, while retaining the full captions as targets. This supervision exposes them to real-world temporal organization, cinematic language, and audiovisual coordination, consistent with their leading Directorial Creation scores of 78.97 and 78.15 and their strong performance in storyboard and timing, editing and transitions, and sound design. In contrast, SCMaPR combines policy-conditioned rewriting, atom-level semantic verification, and conditional revision, whereas VPO trains a prompt optimizer through principle-based SFT and multi-feedback preference optimization. These mechanisms help preserve user requirements and constrain unsupported or contradictory elaboration, consistent with their respective leadership in Semantic Fidelity (87.92) and Internal Consistency (95.07), but their lower Directorial Creation scores (49.18 and 33.07) indicate narrower cinematic coverage. These complementary profiles suggest that effective PE should combine caption-derived planning knowledge with explicit semantic and logical constraints.

Table 3 Temporal-complexity results.
<table><tr><td>Structure</td><td>Count</td><td>SF</td><td>IC</td><td>DC</td></tr><tr><td>Event stages</td><td>1</td><td>69.60</td><td>96.66</td><td>78.43</td></tr><tr><td></td><td>2</td><td>68.79</td><td>94.02</td><td>76.70</td></tr><tr><td></td><td>3+</td><td>65.93</td><td>92.74</td><td>76.45</td></tr><tr><td>Requested shots</td><td>1</td><td>72.48</td><td>95.81</td><td>79.03</td></tr><tr><td></td><td>2</td><td>69.90</td><td>94.52</td><td>75.62</td></tr><tr><td></td><td>3+</td><td>62.51</td><td>88.20</td><td>73.51</td></tr></table>

Table 4 Reference-complexity results.
<table><tr><td>Reference</td><td>Group</td><td>SF</td><td>IC</td><td>DC</td></tr><tr><td rowspan="3">Images</td><td>1</td><td>62.01</td><td>96.70</td><td>77.09</td></tr><tr><td>2</td><td>60.37</td><td>95.57</td><td>76.56</td></tr><tr><td>3+</td><td>54.59</td><td>92.31</td><td>74.38</td></tr><tr><td rowspan="3">Modality</td><td>Image-only</td><td>58.64</td><td>94.66</td><td>75.88</td></tr><tr><td>Video-only</td><td>67.79</td><td>94.46</td><td>74.75</td></tr><tr><td>Hybrid</td><td>45.62</td><td>97.09</td><td>55.69</td></tr></table>

## 4.3 Analysis

Building on the preceding findings, we analyze the necessity of prompt enhancement, the relationship between output length and adequacy, and the impact of task complexity.

Original prompts reveal a substantial adequacy gap. We evaluate the original prompts as a no-enhancement baseline. Across all tasks, they achieve high Semantic Fidelity but low Directorial Creation. On T2V-PE, they attain 100.00 in Semantic Fidelity but only 33.56 in Directorial Creation, indicating that preserving user intent alone is insufficient to fully specify the intended video. This observation raises the question of whether greater prompt length necessarily yields higher Directorial Creation. A length-stratified analysis shows no consistent benefit: across tasks, the longest output quartile trails the shortest by 5.19-10.09 points (see Appendix C.2 for details). Effective PE should therefore add structured, task-relevant specifications for temporal planning, cinematography, narrative development, and audiovisual design rather than merely lengthening prompts.

Finding 3: Temporal and reference complexity remain persistent challenges for prompt enhancers.

To examine these challenges, we use H3-Context-IR as a fixed enhancer and analyze two forms of task complexity: temporal complexity on T2V-PE and reference complexity on R2V-PE.

Temporal-structure complexity. We analyze T2V-PE to isolate temporal complexity from visual-reference grounding. Specifically, we characterize temporal structure by the numbers of ordered event stages and explicitly requested shots, grouping both into 1, 2, and 3+. An event stage denotes a distinct action, state change, or outcome, while the shot analysis includes only prompts with explicit shot structures. Under these groupings, Table 3 shows that cases with three or more event stages trail those with one stage by 3.67, 3.92, and 1.98 points in SF, IC, and DC, respectively The corresponding gaps for requested shots are larger at 9.97, 7.61, and 5.52 points. Together, these results show that both event progression and multi-shot organization pose temporal challenges, with the latter placing greater demands on cross-segment coordination.

Reference complexity. On R2V-PE, we characterize reference complexity along two axes: quantity and modality. For quantity, we group image-only cases by 1, 2, and 3+ images. For modality, we compare image-only, video-only, and hybrid inputs. As shown in Table 4, cases with three or more images trail single-image cases by 7.42, 4.39, and 2.71 points in SF, IC, and DC, respectively. The modality comparison reveals a larger gap. Hybrid inputs score only 45.62 in SF and 55.69 in DC, substantially below both single-modality settings, despite achieving the highest IC of 97.09. This divergence reveals a reference-grounding bottleneck. Enhanced prompts can remain internally coherent while failing to faithfully and adequately integrate heterogeneous references.

Qualitative failure analysis. As shown in Figure 4, evaluated PE exhibit representative failures across all three dimensions. The corresponding videos generated by Wan3.0 visibly reflect these errors, illustrating the interpretability and downstream relevance of our prompt-level evaluation.

![](images/cbf154894d9178e870dd9635fbff35f1d02268a0c1b680547aa367cba421914b.jpg)  
Figure 4 Qualitative failure cases of evaluated prompt enhancers, visualized with Wan3.0 video model. Examples include (a) visual-style deviation, (b) incorrect subject-action binding, (c) language and speaker attribution errors, (d) narrative truncation caused by inadequate temporal planning, and (e) incorrect grounding across multiple references.

Table 5 Spearman correlations for human alignment and downstream utility across three tasks.
<table><tr><td></td><td colspan="3">T2V-PE</td><td colspan="3">I2V-PE</td><td colspan="3">R2V-PE</td></tr><tr><td>Evaluation</td><td>SF</td><td>IC</td><td>DC</td><td></td><td>IC</td><td>DC</td><td>SF</td><td>IC</td><td>DC</td></tr><tr><td colspan="10">Enhanced-prompt alignment</td></tr><tr><td>Direct VLM Scoring</td><td>0.618</td><td>0.464</td><td>0.527</td><td>1.000</td><td>0.600</td><td>0.400</td><td>1.000</td><td>0.500</td><td>0.500</td></tr><tr><td>PEBench Evaluation</td><td>0.955</td><td>0.818</td><td>0.909</td><td>1.000</td><td>1.000</td><td>1.000</td><td>1.000</td><td>1.000</td><td>1.000</td></tr><tr><td colspan="10">Downstream-video alignment</td></tr><tr><td>PEBench Evaluation (Wan3.0)</td><td>0.845</td><td>0.827</td><td>0.945</td><td>1.000</td><td>1.000</td><td>1.000</td><td>1.000</td><td>1.000</td><td>1.000</td></tr><tr><td>PEBench Evaluation (MiniMax-H3)</td><td>0.809</td><td>0.864</td><td>0.936</td><td>1.000</td><td>1.000</td><td>1.000</td><td>1.000</td><td>1.000</td><td>1.000</td></tr></table>

## 4.4 Human Alignment

Finding 4: PEBench evaluation scores reliably reflect expert judgments of both enhanced prompts and downstream videos across generators.

To validate the reliability of our evaluation framework, we measure Spearman correlations $( \rho _ { s } )$ with expert rankings at the enhanced-prompt and downstream-video levels. The resulting correlations are reported in Table 5. At the enhanced-prompt level, our framework aligns more closely with expert rankings than direct VLM scoring, with dimension-averaged gains of 0.333-0.358 across tasks. This improvement supports the effectiveness of rubric-based decomposition. At the downstream-video level, we compare the same evaluation scores with expert rankings of downstream videos generated by Wan3.0 and MiniMax-H3. The resulting correlations of 0.809-1.000 indicate that direct prompt evaluation reliably reflects downstream utility across generators. The study covers all PE systems in Table 2, with the original prompts as a baseline. Annotation details are provided in Appendix D.

## 5 Conclusion

We present PEBench, the first unified benchmark for direct evaluation of prompt enhancers across T2V-PE, I2V-PE. and R2V-PE. It comprises 1,100 expert-verified cases and 1,005 visual assets spanning 35 fine-grained tasks encompassing diverse temporal, cinematic, audiovisual, and multi-reference requirements. Its evidence-grounded framework combines modality-aware evidence with 24 criterion-specific rubrics, enabling consistent, generator-independent assessment without costly video rendering. Our evaluation reveals an emerging shift from descriptive expansion toward intent-preserving cinematic planning. At the method level, caption-reconstruction and forward-refinement methods exhibit complementary strengths, suggesting that future PE systems should combine cinematic knowledge with semantic and logical safeguards. Meanwhile, performance degradation under temporal and multi-reference complexity highlights persistent challenges in temporal planning and cross-asset grounding. Strong alignment with expert judgments at both the prompt and downstream-video levels confirms the practical relevance of direct PE evaluation. By unifying task coverage, evidence-grounded assessment, and criterion-level analysis, PEBench provides a scalable foundation for evaluating, comparing, and advancing video prompt enhancers.

## References

[1]Wan Team. Wan AI: Leading ai video generation model. https : //wan . video, 2026.

[2]ByteDance Seed. Seedance 2.5. https://seed.bytedance.com/en/seedance2\_5, 2026.

[3] MiniMax Team. Minimax-h3. https://www.minimax.io/blog/minimax-h3,2026.

[4] Linqing Wang, Ximing Xing, Yiji Cheng, Zhiyuan Zhao, Donghao Li, Tiankai Hang, Jiale Tao, Qixun Wang, Ruihuang Li, Comi Chen, Xin Li, Mingrui Wu, Xinchi Deng, Shuyang Gu, Chunyu Wang, and Qinglin Lu. Promptenhancer: A simple approach to enhance text-to-image models via chain-of-thought prompt rewriting, 2025. URL https : //arxiv.org/ abs/2509.04545.

[5] Yubo Zhu, Yawen Shao, Ziyun Dai, Zixun Fang, Kai Zhu, Siyang Sun, Haolan Xue, Chuxin Wang, Tingyu Weng, Jingming Luo, Chen Shi, Lianghua Huang, Yufeng Ai, Yuzheng Wang, Wenyuan Zhang, Yu Shang, Yuxiang Bao, Zoubin Bi, Jie Xiao, Jinbo Xing, Jiaxing Zhao, Chongyang Zhong, Hengjian Chen, Chenwei Xie, Akide Liu, Zhehan Kan, Yu Liu, Wei Zhai, Sheng Zhong, and Wei Tong. Wanpe: Towards cinematic prompt enhancement for modern text-to-video generation, 2026. URLhttps://arxiv.org/abs/2609.30221.

[6] Jiale Cheng, Ruiliang Lyu, Xiaotao Gu, Xiao Liu, Jiazheng Xu, Yida Lu, Jiayan Teng, Zhuoyi Yang, Yuxiao Dong, Jie Tang, Hongning Wang, and Minlie Huang. Vpo: Aligning text-to-video generation models with prompt optimization, 2025. URL https://arxiv.org/abs/2503.20491.

[7] Chengyi Yang, Pengzhen Li, Jiayin Qi, Aimin Zhou, Ji Wu, and Ji Liu. Scmapr: Self-correcting multi-agent prompt refinement for complex-scenario text-to-videogeneration, 2026. URL https ://arxiv.org/abs/2604.05489.

[8] Bingjie Gao, Xinyu Gao, Xiaoxue Wu, Yujie Zhou, Yu Qiao, Li Niu, Xinyuan Chen, and Yaohui Wang. The devil is in the prompts: Retrieval-augmented prompt optimization for text-to-video generation. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 3173–3183, June 2025.

[9] Yatai Ji, Jiacheng Zhang, Jie Wu, Shilong Zhang, Shoufa Chen, Chongjian GE, Peize Sun, Weifeng Chen, Wenqi Shao, Xuefeng Xiao, Weilin Huang, and Ping Luo. Prompt-a-video: Prompt your video diffusion model via preference-aligned llm, 2024.URLhttps://arxiv.org/abs/2412.15156.

[10] Bingjie Gao, Qianli Ma, Xiaoxue Wu, Shuai Yang, Guanzhou Lan, Haonan Zhao, Jiaxuan Chen, Qingyang Liu, Yu Qiao, Xinyuan Chen, Yaohui Wang, and Li Niu. Rapo++: Cross-stage prompt optimization for text-to-video generation via data alignment and test-time scaling. arXiv preprint arXiv:2510.20206, 2025.

[11]Kling AI. Kling video 3.0: Professional cinematic ai video production, 2026. URL https://kling. ai/feature/ kling-video-3.

[12] Jiapeng Wang, Chengyu Wang, Jun Huang, and Lianwen Jin. Hallucination-aware prompt optimization for text-to-video synthesis. In James Kwok, editor, Proceedings of the Thirty-Fourth International Joint Conference on Artificial Intelligence, IJCAI-25, pages 10198–10206. International Joint Conferences on Artificial Intelligence Organization, 8 2025. doi: 10.24963/ ijcai.2025/1133.URL https://doi.org/10.24963/ijcai.2025/1133. AI, Arts & Creativity.

[13] Yizhuo Jia, Jingyun Hua, and Yuanxing Zhang. Cape-t2v: Captioner-anchored prompt enhancement toward two-sided conditioning alignment in text-to-video generation, 2026. URL https : //arxiv.org/abs/2608.03046.

[14] Do Xuan Long, Xingchen Wan, Hootan Nakhost, Chen-Yu Lee, Tomas Pfister, and Sercan Ö. Arık. Vista: A test-time selfimproving video generation agent, 2025. URL https://arxiv.org/abs/2510.15831.

[15] Shuailei Ma, Jiaqi Liao, Xinyang Wang, Jingjing Wang, Chaoran Feng, Zijing Hu, Chong Bao, Zichen Xi, Yuqi Gan, Weisen Wang, Yanhong Zeng, Qin Zhao, Zifan Shi, Wei Wu, Hao Ouyang, Qiuyu Wang, Shangzhan Zhang, Jiahao Shao, Yipengjing Sun, Liangxiao Hu, Lunke Pan, Nan Xue, Kecheng Zheng, Yinghao Xu, Xing Zhu, Yujun Shen, and Ka Leong Cheng. Scaling mixture-of-experts video pretraining for embodied intelligence, 2026. URL https : //arxiv. org/abs/2607.07675.

[16] Lightricks. Ltx-2.5. https://1tx.io/model/1tx-2-5,2026.

[17] Gemma Team, Sherif El Abd, Vaibhav Aggarwal, Robin Algayres, Alek Andreev, Olivier Bachem, Ian Ballantyne, Cormac Brick, Victor Cărbune, Michelle Casbon, Mayank Chaturvedi, Aditya Chawla, Victor Cotruta, Alice Coucke, Phil Culliton, Robert Dadashi, Lucas Dixon, Mohamed Elhawaty, Utku Evci, Clément Farabet, Johan Ferret, Filippo Galgani, Sertan Girgin, Jean-Bastien Grill, Maarten Grootendorst, Jiaxian Guo, Cassidy Hardin, Yanzhang He, Steven M. Hernandez, Omri Homburger, Léonard Hussenot, Juyeong Ji, Armand Joulin, Aishwarya Kamath, Parnian Kassraie, Olivier Lacombe, Preethi Lahoti, Gaël Liu, Gus Martins, Luciano Martins, Tatiana Matejovicova, Ramona Merhej, Nikola Momchev, Sneha Mondal, Ryan Mullins, Sindhu Raghuram Panyam, Shreya Pathak, Sarah Perrin, André Susano Pinto, Etienne Pot, Angéline Pouget, Alexandre Ramé, Sabela Ramos, Douglas Reid, David Rim, Morgane Rivière, Karsten Roth, Louis Rouillard, Omar Sanseviero, Pier Giuseppe Sessa, Shane Settle, Danila Sinopalnikov, Sara Smoot, Piotr Stanczyk, Andreas Steiner, Lawrence Stewart, Ilya

Tolstikhin, Michael Tschannen, Anton Tsitsulin, Nino Vieillard, Renjie Wu, Pingmei Xu, Haichuan Yang, Edouard Yvinec, Biao Zhang, Li Zhang, Joe Zou, Nicolas Aagnes, Abdelrahman Abdelhamed, Jakub Adamek, Shivani Agrawal, Shubham Agrawal, Ibrahim Alabdulmohsin, Jean Baptiste Alayrac, Uri Alon, Chandramouli Amarnath, Ankesh Anand, Chrysovalantis Anastasiou, Setareh Ariafar, François-Xavier Aubet, Kyriakos Axiotis, Federico Barbero, Joelle Barral, Alexei Bendebury, Urs Bergmann, Stanley Bileschi, Kat Black, Mathieu Blondel, Sebastian Borgeaud, Arthur Bražinskas, Ryan Burnell, Robert Busa-Fekete, Mu Cai, Daniele Calandriello, Glenn Cameron, Charlotte Caucheteux, Rahma Chaabouni, Garima Chadha, Jetha Chan. Blake Jianhang Chen, Jesse Chen, Lin Chen, Xu Chen, Derek Cheng, Tzu hsiang Chien, Nikolai Chinaev, Yi Chou, Zhaohui Chu, Benjamin Coleman, Pooja Consul, Sam Conway-Rahman, Scott Crowell, Dylan Cutler, Vivek Dani, Samira Daruki. Anil Das, Daniel Deutsch, Nishanth Dikkala, Li Ding, Qiuhan Ding, Shenil Dodhia, Konstantin Donhauser, Tulsee Doshi, Anca Dragan, Alex Druinsky, Sahil Dua, Zoltan Egyed, Danielle Eisenbud, Daniel Eppens, Cindy Fan, Bahare Fatemi, Yassir Fathullah, Vlad Feinberg, Milen Ferev, Sebastian Flennerhag, Takumi Fujimoto, João Gabriel Oliveira, Isaac Galatzer-Levy, João Gante, Simon Geisler, Soham Ghosal, Antonious M. Girgis, Tamara von Glehn, Alec Go, Alhaad Gokhale, Alex Grills. Yiming Gu, Mayank Gupta, Pramod Gupta, Guru Guruganesh, Raia Hadsell, Hamza Harkous, Jitendra Harlalka, Demis Hassabis, Anja Hauth, Joe Heyward, Arian Hosseini, Chih-Yang Hsia, I-Hung Hsu, Xiaopeng Huang, Yangsibo Huang, Kevin Hui, Adrian Hutter, Te I, Fotis Iliopoulos, Advait Jain, Ganesh Jawahar, Ziwei Ji, Qilin Jin, Melvin Johnson, Kandarp Joshi, Arun Kandoor, Wang-Cheng Kang, Koray Kavukcuoglu, Mehran Kazemi, Kathleen Kenealy, Amr Khalifa, Phoebe Kirk, Ivan Korotkov, Suraj Kothawade, Vitaly Kovalev, Neel Kovelamudi, Adam Kraft, Ravin Kumar, Vivek Kumar, Harish Kuppam, Justin Lannin, Chen-Yu Lee, Seungji Lee, Dmitry Lepikhin, Alon Levkovitch, Dongdong Li, Qiujia Li, Valentin Liévin, Ethan Lin, Ziqian Lin, Casper Liu, Tianlin Liu, Tianqi Liu, Xin Liu, Ivan Lobov, Mayank Lunayach, Min Ma, Gagan Madan, Andrii Maksai, Eric Malmi, Michal Matuszak, Daniel McDuff, Gaurav Menghani, Maciej Mikuła, Daniil Mirylenka, Karolis Misiunas. Vedant Misra, Andreea Mitran, Kareem Mohamed, Maksim Mukha, Eric Noland, James O'Donnell, Brendan O'Donoghue, Kate Olszewska, Bernett Orlando, Wanqiong Pan, Rina Panigrahy, Unnati Parekh, Nicolas Perez-Nieves, Chunjong Park, Eric Paskie, Liqian Peng, Bryce Petrini, Slav Petrov, Jonas Pfeiffer, Bilal Piot, Martyna Plomecka, Siim Poder, Octavio Ponce, Arijit Pramanik, David Racz, Anish Rajan, Michelle Ramanovich, Anand Rao, Marvin Ritter, Vitor Rodrigues, Evan Rosen, Mikołaj Rybiński, Noveen Sachdeva, Michaël E. Sander, Rohit Sathyanarayana, Sagar Savla, Samuel Schmidgall, Tal Schuster, George Scrivener, Benoit Seguin, Andrew Sellergren, Aliaksei Severyn, Izhak Shafran, Dhruv Shah, Bobak Shahriari. Yuan Shangguan, Ashish Shenoy, Pradeep Shenoy, Rakesh Shivanna, Pauline Sho, Lucas Spangher, Wojciech Stokowiec, Tim Strother, Yao Su, Yinghao Sun, Mukund Sundararajan, Andrea Tacchetti, Mor Hazan Taege, Pouya Tafti, Jean Tarbouriech Chetan Tekur, Shantanu Thakoor, Rahul Thapa, Madeleine Traverse, Lenart Treven, Tao Tu, Chien Te Tung, Cağlar Unlü. Petar Veličković, Malini Pooni Venkat, Sagar Gubbi Venkatesh, Vidya Venkiteswaran, Francesco Visin, Alex Vitvitskyi, Kiran Vodrahalli, Weiyi Wang, Xin Wang, Tris Warkentin, Jan Wassenberg, John Wieting, Cindy Wu, Lechao Xiao, Hao Xu, Yuhui Xu, Fuzhao Xue, Arun Yadav, Jun Yan, Antoine Yang, Lin Yang, Ming-Hsuan Yang, Ziyu Ying, Jae Hyeon Yoo, Morteza Zadimoghaddam, Sajjad Zafar, Fred Zhang, Jiageng Zhang, Jianyi Zhang, Xiaofan Zhang, Chao Zhao, David Zhou, and Chen Zou. Gemma 4 technical report, 2026. URL https://arxiv.org/abs/2607.02770.

[18] Ziqi Huang, Yinan He, Jiashuo Yu, Fan Zhang, Chenyang Si, Yuming Jiang, Yuanhan Zhang, Tianxing Wu, Qingyang Jin, Nattapol Chanpaisit, Yaohui Wang, Xinyuan Chen, Limin Wang, Dahua Lin, Yu Qiao, and Ziwei Liu. Vbench: Comprehensive benchmark suite for video generative models, 2023. URL https ://arxiv. org/abs/2311.17982.

[19] Yaofang Liu, Xiaodong Cun, Xuebo Liu, Xintao Wang, Yong Zhang, Haoxin Chen, Yang Liu, Tieyong Zeng, Raymond Chan, and Ying Shan. Evalcrafter: Benchmarking and evaluating large video generation models, 2024. URL https : //arxiv. org/abs/2310.11440.

[20] Hui Han, Siyuan Li, Jiaqi Chen, Yiwen Yuan, Yuling Wu, Yufan Deng, Chak Tou Leong, Hanwen Du, Junchen Fu, Youhua Li, Jie Zhang, Chi Zhang, Li-jia Li, and Yongxin Ni. Video-Bench: Human-Aligned Video Generation Benchmark . In 2025 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 18858–18868, Los Alamitos, CA, USA, June 2025. IEEE Computer Society. doi: 10.1109/CVPR52734.2025.01757. URL https : // doi. ieeecomputersociety.org/10.1109/CVPR52734.2025.01757.

[21] Shenghai Yuan, Xianyi He, Yufan Deng, Yang Ye, Jinfa Huang, Bin Lin, Jiebo Luo, and Li Yuan. Opens2v-nexus: A detailed benchmark and million-scale dataset for subject-to-video generation, 2025. URL https ://arxiv. org/abs/ 2505.20292.

[22] Jianhui Wei, Xiaotian Zhang, Yichen Li, Yuan Wang, Yan Zhang, Ziyi Chen, Zhihang Tang, Wei Xu, and Zuozhu Liu. Univbench: Towards unified evaluation for video foundation models, 2026. URL https : //arxiv. org/abs/2602.21835.

[23] Yujie Wei, Yujin Han, Zhekai Chen, Yongming Li, Kaixun Jiang, Zhihang Liu, Quanhao Li, Zhiwu Qing, Xiang Wang, Zhen Xing, Ruihang Chu, Lingyi Hong, Yefei He, Junjie Zhou, Junqiu Yu, Yang Shi, Difan Zou, Kai Zhu, Shiwei Zhang, Yingya Zhang, Yu Liu, Xihui Liu, and Hongming Shan. Msavbench: Towards comprehensive and reliable evaluation of multi-shot audio-video generation, 2026.URL https://arxiv.org/abs/2605.20183.

[24] Kaiqi Liu, Haoxuan Zeng, Jingqi Liu, Jiacong Fang, Ziqi Cai, Yunyao Mao, Henglin Liu, Yu Sheng, Shuchen Weng, and Boxin Shi. Streamav-bench: A comprehensive benchmark for streaming audio-video generation, 2026. URL https : //arxiv. org/abs/2608.26336.

[25] Google. Gemini 3.1 pro: A smarter model for your most complex tasks. https://blog.google/ innovation-and-ai/models-and-research/gemini-models/gemini-3-1-pro/,2026.

[26] Yexin Liu, Manyuan Zhang, Yueze Wang, Hongyu Li, Dian Zheng, Weiming Zhang, Changsheng Lu, Xunliang Cai, Yan Feng, Peng Pei, and Harry Yang. Opensubject: Leveraging video-derived identity and diversity priors for subject-driven image generation and manipulation, 2025. URL https://arxiv.org/abs/2512.08294.

[27] Kepan Nan, Rui Xie, Penghao Zhou, Tiehan Fan, Zhenheng Yang, Zhijie Chen, Xiang Li, Jian Yang, and Ying Tai. Openvid-1m: A large-scale high-quality dataset for text-to-video generation, 2025. URL https://arxiv.org/abs/2407. 02371.

[28] Sen Liang, Cong Wang, Zhentao Yu, Fengbin Guan, Zhengguang Zhou, Teng Hu, Youliang Zhang, Yuan Zhou, Xin Li, Qinglin Lu, and Zhibo Chen. Goku: A million-scale universal dataset and benchmark for instruction-based video editing, 2026. URL https://arxiv.org/abs/2606.30599.

[29] Chaojie Mao, Chen-Wei Xie, Chongyang Zhong, Haoyou Deng, Jiaxing Zhao, Jie Xiao, Jinbo Xing, Jingfeng Zhang, Jingren Zhou, Jingyi Zhang, Jun Dan, Kai Zhu, Kang Zhao, Keyu Yan, Minghui Chen, Pandeng Li, Shuangle Chen, Tong Shen, Yu Liu, Yue Jiang, Yulin Pan, Yuxiang Tuo, Zeyinzi Jiang, Zhen Han, Ang Wang, Bang Zhang, Baole Ai, Bin Wen, Boang Feng, Feiwu Yu, Gang Wang, Haiming Zhao, He Kang, Jianjing Xiang, Jianyuan Zeng, Jinkai Wang, Junjie Zhou, Ke Sun, Linqian Wu, Pei Gong, Pingyu Wu, Ruiwen Wu, Tongtong Su, Wenmeng Zhou, Wenting Shen, Wenyuan Yu, Xianjun Xu, Xiaoming Huang, Xiejie Shen, Xin Xu, Yan Kou, Yangyu Lv, Yifan Zhai, Yitong Huang, Yun Zheng, Yuntao Hong, Zhe Zhang, and Zhicheng Zhang. Wan-image: Pushing the boundaries of generative visual intelligence, 2026. URL ht tps : //arxiv.org/abs/2604.19858.

[30] Team Seedream, Yunpeng Chen, Yu Gao, Lixue Gong, Meng Guo, Qiushan Guo, Zhiyao Guo, Xiaoxia Hou, Weilin Huang, Yixuan Huang, et al. Seedream 4.0: Toward next-generation multimodal image generation. arXiv preprint arXiv:2509.20427, 2025.

[31] OpenAI. Gpt-image-2. https://openai.com/index/introducing-chatgpt-images-2-0/,2026.

[32]OpenAI. Gpt-5.4. https://openai.com/index/introducing-gpt-5-4/,2026.

[33]Qwen Team. Qwen3.5-omni technical report, 2026. URL https://arxiv.org/abs/2604.15804.

## Appendix Overview

This appendix supplements the main paper with data, evaluation, experimental, and human annotation details, followed by ethical considerations and limitations. It is organized as follows.

• Appendix A: More Data Details. Data design, construction, and detailed analysis.

• Appendix B: More Evaluation Details. Evaluation dimensions and criteria, structured output formats, and representative case.

• Appendix C: Additional Experimental Details. Implementation details and length-stratified analysis.

• Appendix D: Human Expert Annotation. Pairwise annotation protocol, quality control, and annotation interfaces.

• Appendix E: Ethics, Privacy, and Licensing. Data screening, asset licensing, and release policy.

• Appendix F: Limitations. Benchmark scale and future expansion.

## A More Data Details

## A.1 Data Design Details

Prompt enhancement settings. The benchmark unifies three settings with progressively richer conditioning. T2V-PE contains 500 text-only cases. I2V-PE contains 300 cases, each paired with one first-frame image. R2V-PE contains 300 cases conditioned on one or more images or videos whose roles may include subject identity, object, scene, style, motion, cinematography, transition pattern, or audiovisual rhythm. Together, these settings cover prompt enhancement from text, temporal continuation from a first frame, and integration of multiple references.

Tasks. The benchmark defines a two-level task taxonomy with 9 high-level capability categories and 35 fine-grained task types. Each task type belongs to one parent category. Table 6 lists the taxonomy and the number of task types in each category.

Table 6 Hierarchical task taxonomy of PEBench. The nine high-level categories contain 35 fine-grained task types.
<table><tr><td>High-level category</td><td></td><td># Fine-grained task types</td></tr><tr><td>Action</td><td></td><td>4 Daily Activity, Combat, Sports, Dance</td></tr><tr><td>Animation</td><td></td><td>2 2D Animation, 3D Animation</td></tr><tr><td>Visual Effects</td><td></td><td>5 VFX Character, VFX Object, Transformation, Light/Weather Effects, Sci-Fi Creature</td></tr><tr><td>General</td><td></td><td>5 Physical Laws, On-Screen Text, Portrait, Nature, Intellectual Property (IP)</td></tr><tr><td>Narrative</td><td>3</td><td>Modern Story, Historical Story, Fantasy Story</td></tr><tr><td>Cinematography</td><td>3</td><td>Camera Motion, Time-Lapse, Lighting and Color</td></tr><tr><td>Speech</td><td></td><td>5 Dialogue, Monologue, Narration, Explanation, Emotional Vocalization</td></tr><tr><td>Music</td><td>6</td><td>Singing, Song and Dance, Music Video, Opera, Background Music, Instrumental Performance</td></tr><tr><td>Sound Effects</td><td></td><td>2 Animal Sound, Environmental Sound</td></tr></table>

Scenario composition dimensions. Each scenario is composed along four additional axes. The five subject categories are Human, Animal, Object, Fantasy Being, and No Salient Subject. The eight scene domains are Domestic, Urban, Workplace, Performance, Sports, Transportation, Nature, and Otherworldly. The seven visual style classes are Photorealistic, Anime, Graphic Illustration, 3D CGI, Painting, Digital Stylized, and Mixed/Unspecified. The eight explicit audio domains are Speech and Vocalization, Singing, Music, Human Activity Sounds, Foley and Object Sounds, Natural and Animal Sounds, Environmental Ambience, and Mechanical and Vehicle Sounds

Linguistic, temporal, and cinematic dimensions. The multilingual portion spans English, Chinese, Japanese, Korean, Spanish, French, Arabic, Russian, Bengali, German, Portuguese, and Hindi. Prompts vary from concise concepts to detailed scripts and may describe a single continuous shot or as many as 13 explicitly structured shots. Cinematic coverage includes near, medium, and long shot scale families. It also covers level, high, low, overhead or aerial, and subjective or special viewpoints. Lens families include wide, standard, telephoto, and specialty lenses. Camera motion includes translational, rotational, and compound movement. Composition, focus, transition, lighting, palette, and color grading provide additional controls that may overlap.

## A.2 Data Construction Details

Figure 2 summarizes the four-stage construction pipeline. This section describes the design choices and quality checks used at each stage.

## Stage 1: Taxonomy Design and Composition

Domain experts define the nine capability categories and 35 tasks in Table 6. The taxonomy covers visual, narrative, cinematic, speech, music, and sound capabilities. It therefore tests both frame content and the development of events, camera behavior, and audio over time. The experts also define the subject, scene, style, and audio vocabularies described in Section A.1. Each candidate scenario is represented as a (category, subject, scene, style, audio) quintuple. It is then paired with a task, a prompt enhancement setting, and a target duration. Incompatible combinations are discarded rather than forced. Coverage is monitored across the dataset to maintain diversity without producing incoherent or infeasible requests.

## Stage 2: Prompt Synthesis and Diversification

Gemini 3.1 Pro synthesizes a candidate user request from the sampled specification. It considers the task and conditioning modality, the scenario composition, and the content that can reasonably unfold within the target duration. Instruction specificity ranges from short concepts to descriptive requests and detailed scripts. Temporal structure also varies. Some requests leave the shot structure implicit, while others specify ordered events, timestamps, or multiple shots. This process varies how much information and temporal planning are provided to the prompt enhancer.

## Stage 3: Reference Asset Acquisition

Cases with reference inputs use two acquisition routes. Public data selection draws candidate images and videos from existing datasets. Images are screened for visual quality, content safety, resolution, and redundancy. Videos are screened for visual quality and motion prominence, segmented into semantically complete clips, and represented by keyframes when a static reference is needed.

Generation with models expands the public pool using Wan-Image [29], Seedream [30], and GPT Image 2 [31]. In the image guided route, a primary image is used to create variations in viewpoint, pose, or expression while preserving identity. In the text guided route, Gemini 3.1 Pro converts the sampled scenario or a relevant shot description into a concise image generation instruction. Together, the two routes support single image, multiple image, video only, and mixed image and video conditioning. Each retained reference is assigned an explicit role such as subject, scene, style, motion, or cinematography

## Stage 4: Automated Verification and Expert Curation

Quality control combines automated screening with expert review. Gemini 3.1 Pro derives a checklist from each request and its target duration. It checks task alignment, logical coherence, and duration feasibility. For cases with reference inputs, it jointly inspects the instruction and candidate assets to verify their assigned roles, alignment with the script, reference coverage, and consistency across assets.

Ten domain experts review the verified pool. They refine awkward language, remove ambiguous or infeasible instructions, reject uninformative or mismatched references, and eliminate repetitive or redundant cases. They also monitor task balance, difficulty, and novelty across cases. The final benchmark contains 1,100 cases verified by experts and 1,005 visual assets.

## A.3 Detailed Data Analysis

Task coverage. PEBench covers 9 high level capability categories: Action (25.3%), Visual Effects (16.0%), Narrative (13.4%), Animation (10.0%), Speech (9.7%), General (9.1%), Music (7.6%), Cinematography (6.8%), and Sound Effects (2.1%).

Subject, scene, and style diversity. Subjects span 5 categories: Human (50.4%), Object (16.6%), Fantasy Being (14.4%), Animal (10.1%), and No Salient Subject (8.5%). Scenes span 8 categories: Otherworldly (19.8%), Domestic (14.5%), Urban (14.1%), Workplace (14.1%), Nature (13.2%), Performance (11.9%), Sports (7.5%), and Transportation (5.0%). T2V-PE visual styles span 7 categories: Photorealistic (61.2%), Digital Stylized (10.4%), 3D CGI (8.2%), Mixed/Unspecified (8.2%), Graphic Illustration (5.0%), Painting (3.8%), and Anime (3.2%).

Audio and 1anguage diversity. Prompts cover 12 languages: English (50.0%), Chinese (40.0%), Japanese (1.0%), Korean (1.0%), Spanish (1.0%), French (1.0%), Arabic (1.0%), Russian (1.0%), Bengali (1.0%), German (1.0%), Portuguese (1.0%), and Hindi (1.0%). Audio content spans 8 categories: Speech and Vocalization (38.3%), Music (19.9%), Foley and Object Sounds (14.6%), Singing (12.6%), Natural and Animal Sounds (7.7%), Mechanical and Vehicle Sounds (3.1%), Human Activity Sounds (2.7%), and Environmental Ambience (1.1%).

Prompt length. Prompt length spans 7 intervals: at most 50 tokens (14.1%), 51 to 100 tokens (13.8%), 101 to 200 tokens (17.7%), 201 to 400 tokens (20.1%), 401 to 800 tokens (16.4%), 801 to 1,600 tokens (15.9%), and more than 1,600 tokens (2.0%). Table 7 reports detailed token length statistics for each task. Target durations range from 2 to 30 seconds. There are 147 cases from 2 to 5 seconds, 321 from 6 to 10 seconds, 614 from 11 to 15 seconds, 11 from 16 to 20 seconds, and 7 from 21 to 30 seconds.

Table 7 Input prompt token length statistics. Quartiles are computed independently within each setting.
<table><tr><td>Setting</td><td>Min.</td><td>Q1</td><td>Median</td><td>Mean</td><td>Q3</td><td>Max.</td></tr><tr><td>T2V-PE</td><td>6</td><td>85.3</td><td>237.5</td><td>425.8</td><td>597.3</td><td>3,431</td></tr><tr><td>I2V-PE</td><td>8</td><td>91.8</td><td>221.5</td><td>448.9</td><td>642.8</td><td>4,120</td></tr><tr><td>R2V-PE</td><td>13</td><td>89.8</td><td>210.5</td><td>363.8</td><td>549.0</td><td>2,412</td></tr><tr><td>All</td><td>6</td><td>88.8</td><td>224.0</td><td>415.2</td><td>596.8</td><td>4,120</td></tr></table>

Cinematic language. PEBench reports 3 major shot scale families: near (49.5%), long (30.7%), and medium (19.8%). It covers 5 major camera angles: eye level (38.0%), high (23.2%), 1ow (20.2%), subjective (13.7%), and others (4.9%). It also covers 4 lens categories: wide angle (38.1%), telephoto or zoom (27.6%), specialty (25.4%), and standard (9.0%). Camera movement spans 3 types: pure translation (50.7%), compound (44.2%), and pure rotation (5.0%).

Reference diversity and complexity. R2V-PE reference configurations comprise 3 types: image only (80.0%), video only (9.3%), and mixed image and video (10.7%). Cases with multiple images span 6 reference counts: two (39.2%), three (21.7%), four (19.3%), five (16.9%), six (1.8%), and seven (1.2%). Multi-image reference structures comprise 5 types: multi view (5.4%), multiple subjects (15.1%), subject with object or scene (56.0%), sequential frames (21.7%), and others (1.8%). Reference asset styles comprise 3 groups: realistic (54.6%), nonrealistic (24.8%), and mixed or unspecified (20.6%). In absolute terms, I2V-PE contributes 300 first frame images. R2V-PE contributes 644 images and 61 videos. Its image conditioned cases include 106 with one image, 65 with two, 36 with three, 32 with four, 28 with five, 3 with six, and 2 with seven. Its video conditioned cases include 59 with one video and 1 with two videos.

## B More Evaluation Details

## B.1 Evaluation Dimensions and Criteria

PEBench evaluates enhanced prompts using 24 criteria grouped into Semantic Fidelity (SF), Internal Consistency (IC), and Directorial Creation (DC). For each applicable criterion, the evaluator determines whether the enhanced prompt contains no issue, a minor issue, or a major issue, and cites evidence for that decision. A criterion is counted as passed only when no issue is identified. Criterion-level pass rate is therefore the percentage of applicable cases that pass the criterion, while each dimension-level score aggregates the number and severity of its constituent issues. Criteria tied to optional source content, such as dialogue or on-screen text, are applied only when that content is present or explicitly requested.

## B.1.1 Semantic Fidelity Metrics

Semantic Fidelity evaluates whether the enhanced prompt preserves the user instruction and instruction-relevant visual evidence. Added details must not omit, weaken, substitute, or contradict source requirements.

(1) Genre and Style Fidelity (GSF). Measures preservation of the requested genre, medium, aesthetic, and visual style. Replacing the medium, weakening a defining genre convention, or adding an incompatible reference style constitutes an issue

(2) Subject and Attribute Preservation (SAP). Measures preservation of subject identity, count, appearance, attributes, objects, and their bindings. I2V uses the first frame as the initial-state anchor. R2V follows each reference's assigned role.

(3) Action and Narrative Fidelity (ANF). Measures preservation of required actions, interactions, event order, outcomes, and narrative intent. Omissions, substitutions, reversals, or added events that alter the requested event chain are treated as issues.

(4) Dialogue and Lyrics Integrity (DLI). Measures fidelity to dialogue, narration, lyrics, language, delivery, and speaker attribution. Exact quotations must not be paraphrased, deleted, translated, or converted into another form unless explicitly permitted.

(5) Audio Requirement Preservation (ARP). Measures preservation of speech, vocalization, music, sound effects, ambience, and silence constraints. Missing, substituted, or added sounds that violate negative requirements such as “no music” are penalized.

(6) Camera Constraint Preservation (CCP). Measures preservation of shot count, scale, angle, lens, focus, movement, and single-take or static-camera constraints. Added camera design is allowed only where it does not override explicit cinematic intent.

(7) Lighting and Color Fidelity (LCF). Measures preservation of time of day, illumination, light quality, brightness, palette, saturation, contrast, and grading. Compatible elaboration is allowed, but it must not change a defining lighting or color condition.

(8) Scene and Spatial Preservation (SSP). Measures preservation of setting, background elements, layout, relative positions, and movement directions. Changing the location, removing required objects, or reversing spatial relations is a violation.

(9) On-Screen Text Integrity (OTI). Measures fidelity to visible titles, signs, subtitles, and interface text, including exact wording, spelling, language, placement, and timing. Visible text cannot be replaced by semantically similar spoken content.

## B.1.2 Internal Consistency Metrics

Internal Consistency evaluates whether the complete enhanced prompt is logically, spatially, temporally, physically, and audiovisually coherent.

(10) Subject and Attribute Continuity (SAC). Measures stability of subject identity, count, appearance, clothing, props, and persistent attributes across shots. Unexplained renaming, duplication, disappearance, or attribute drift constitutes an issue.

(11) Action and Narrative Logic (ANL). Measures whether actions form a coherent causal or procedural progression. The evaluator checks prerequisites, bindings between subjects and actions, reactions, resolution, and incompatible or dangling events.

(12) Temporal and State Continuity (TSC). Measures chronology, duration allocation, event order, and state transitions. Overlapping or infeasible timestamps, unexplained resets, and later shots showing states that should no longer exist are penalized.

(13) Spatial Consistency (SPC). Measures consistency of location, screen direction, relative position, orientation, trajectory, and entrances or exits. Unexplained position swaps or movement paths that cannot connect adjacent states constitute issues.

(14) Physical Logic (PL). Measures compliance with real-world physics or established fictional rules, including gravity, contact, object permanence, bodily feasibility, and causal relations. Deliberate fantasy is judged by its own world logic.

(15) Audiovisual Synchronization (AVS). Measures temporal and semantic alignment between visible events and dialogue, lyrics, music, effects, or ambience. Wrong speakers, premature effects, and misaligned lip-sync or beat-sync actions are penalized.

## B.1.3 Directorial Creation Metrics

Even casually captured videos involve temporal progression, framing, camera behavior, lighting, performance, visual style, and sound. A prompt may preserve the source and remain logically coherent while leaving these directorial elements underspecified. To assess whether a prompt enhancer develops them into an actionable production plan, we introduce Directorial Creation, covering storyboarding, cinematography, editing, lighting, narrative, sound, performance, and visual style.

(16) Storyboard and Timing Completeness (STC). Measures whether the content is organized into a feasible temporal plan with adequate event or shot boundaries, ordering, and durations.

(17) Camera Design Specificity (CDS). Measures actionable specs of framing, scale, angle, viewpoint, lens, focus, and movement.

(18) Editing and Transition Design (ETD). Measures whether cuts and transitions support temporal, spatial, narrative, or rhythmic progression. Disconnected shots or repeated transitions without clear continuity or motivation are considered inadequate.

(19) Lighting and Color Specification (LCS). Measures actionable specification of light source, direction, quality, contrast, palette, saturation, and grading.

(20) Narrative Development and Engagement (NDE). Measures sufficient progression, variation, escalation, or resolution for the requested content form. Static restatement and unmotivated repetition are inadequate.

(21) Core-Intent Focus (CIF). Measures whether additions support the central request. Repetitive padding, template language, unnecessary subplots, and decorative details that distract from the intended result reduce adequacy,

(22) Sound Design Integration (SDI). Measures whether speech, music, effects, and ambience are integrated through source, character, intensity, progression, and visual timing. It rewards content-appropriate design rather than requiring every audio type.

(23) Subject and Performance Specificity (SPS). Measures generation-relevant subject and performance detail, including appearance, pose, expression, gesture, movement, interaction, gaze, and vocal delivery.

(24) Visual Style Specification (VSS). Measures actionable specification of medium, texture, rendering language, production design, and visual treatment.

## B.2 Structured Output Formats

## B.2.1 Asset-Fact Extraction Output

We present R2V as the representative example because it contains the most complete multimodal evidence structure.   
Qwen3.5-Omni-Plus [33] returns the following fixed AssetFacts structure, with each fact traceable to its source asset.

## R2V AssetFacts Output Format

"asset\_inventory": {   
"total\_count": 0,   
"image\_count": 0,   
"video\_count": 0,   
"items":[   
{   
"asset\_id": "image\_1|video\_1",   
"asset\_type": "imagelvideo",   
"input\_index": 1,   
"modality\_index": 1

},   
"assets": [   
"asset\_id": "image\_1|video\_1",   
"asset\_type": "imagelvideo",   
"duration\_seconds": "xxx",   
"media\_and\_style": {   
"media\_type": "live\_action|2d\_animation|3d\_cglstop\_motion|illustration|game\_render|mixed",   
"realism": "realisticlstylized\_realisticlnon\_realistic",   
"primary\_style": "xxx [dominant visual style]",   
"secondary\_styles": ["xxx [additional visual style]"],   
"visible\_style\_evidence": ["xxx [visible evidence for the style]"],   
"certainty": "certain|uncertain",   
"uncertainty\_reason": "xxx [reason when uncertain]"   
},   
"overall\_content\_summary": "xxx [brief summary of the asset]",   
"subjects": [   
"id": "image\_1\_subject\_1|video\_1\_subject\_1",   
"role": "primary|secondary|background\_group",   
"category": "xxx [person, animal, object, or group]",   
"identity": "xxx [visible identity description]",   
"count\_exact": 1,   
"count\_estimate": "xxx",   
"appearance": ["xxx [visible appearance detail]"],   
"pose\_and\_current\_state": ["xxx [visible pose or current state]"],   
"expression": "xxx [visible expression if identifiable]",   
"held\_or\_worn\_items": ["xxx [held or worn item]"],   
"frame\_position": "xxx [location within the frame]",   
"occlusion": "nonelpartial|heavy",   
"certainty": "certain|uncertain",   
"uncertainty\_reason": "xxx [reason when uncertain]"   
],   
"scene\_and\_spatial\_relations": {   
"scene\_type": "xxx [visible scene description]",   
"indoor\_outdoor": "indoor|outdoor|mixed",   
"foreground\_elements": ["xxx [foreground element]"],   
"midground\_elements": ["xxx [midground element]"],   
"background\_elements": ["xxx [background element]"],   
"spatial\_relations": [   
"subject": "xxx [subject ID or entity]",   
"relation": "xxx [visible spatial relation]",   
"object": "xxx [related object or entity]",   
"certainty": "certain|uncertain"   
}   
],   
"cropped\_or\_offscreen\_information": ["xxx [cropped or offscreen cue]"]   
},   
"camera\_and\_composition": {   
"shot\_scale": "extreme\_close\_uplclose\_up|medium\_close\_uplmedium\_shot|medium\_full\_shot|full\_shot|   
long\_shot|extreme\_long\_shot|varied",   
"camera\_angle": "xxx [camera angle]",   
"camera\_height": "xxx [camera height]",   
"view\_characteristics": "wide\_angle\_feel|standard\_view|telephoto\_compression|fisheye|varied",   
"composition": ["xxx [composition characteristic]"],   
"visual\_focus": "xxx [primary visual focus]",   
"depth\_of\_field": "shallow|moderateldeeplvaried",   
"camera\_motion": ["xxx [observed camera motion]"]   
"certainty": "certain|uncertain",   
"uncertainty\_reason": "xxx [reason when uncertain]"   
11   
"lighting\_and\_color": {   
"light\_sources": ["xxx [visible light source]"],   
"light\_direction\_and\_effect": ["xxx [light direction and visible effect]"],   
"time\_of\_day\_cue": "xxx [visible time-of-day cue if present]",   
"color\_temperature": "warm|neutral|cool|mixed",   
"dominant\_colors": ["xxx [dominant color]"],   
"accent\_colors": ["xxx [accent color]"],   
"certainty": "certainluncertain",   
"uncertainty\_reason": "xxx [reason when uncertain]"   
},   
"onscreen\_text": {   
"present": false,   
"items": [   
"text": "xxx [visible text transcription]",   
"type": "subtitleltitlellogolsignlinterfacelother",   
"language": "xxx [text language]",

"frame\_position": "xxx [text location in the frame]",   
"time\_range": {   
"start\_seconds": "xxx",   
"end\_seconds": "xxx"   
},   
"certainty": "certain|uncertain",   
"uncertain\_fragments": ["xxx [uncertain text fragment]"]   
]   
"segments": [   
"segment\_id": "video\_1\_segment\_1",   
"shot\_index": 1,   
"is\_new\_shot": true,   
"start\_seconds": "xxx",   
"end\_seconds": "xxx",   
"transition\_in": "xxx [incoming transition if present]",   
"scene": "xxx [scene during the segment]",   
"subjects\_present": ["xxx [visible subject ID]"],   
"visible\_actions\_and\_state\_changes": ["xxx [visible action or state change]"],   
"subject\_interactions": ["xxx [visible subject interaction]"],   
"object\_events": ["xxx [visible object event]"],   
"camera": {   
"shot\_scale": "xxx [segment shot scale]",   
"angle": "xxx [segment camera angle]",   
"camera\_height": "xxx [segment camera height]",   
"view\_characteristics": "xxx [segment lens characteristic]",   
"composition": ["xxx [segment composition]"],   
"focus": "xxx [segment visual focus]",   
"depth\_of\_field": "xxx [segment depth of field]",   
"movement": {   
"type": "staticlpush\_in|pull\_out|pan|tilt|track|crane|orbit|zoom|handheld|compound",   
"direction": "xxx [movement direction]",   
"speed": "xxx [movement speed]",   
"amplitude": "xxx [movement amplitude]",   
"stability": "xxx [movement stability]",   
"tracking\_subject": "xxx [tracked subject ID if applicable]",   
"start\_frame": "xxx [movement start state]",   
"end\_frame": "xxx [movement end state]"   
},   
"shot\_change\_from\_previous": {   
"shot\_scale\_change": "xxx [change in shot scale]",   
"angle\_change": "xxx [change in camera angle]",   
"camera\_position\_change": "xxx [change in camera position]",   
"composition\_change": "xxx [change in composition]",   
"focus\_change": "xxx [change in visual focus]",   
"movement\_change": "xxx [change in camera movement]",   
"scene\_change": "xxx [change in scene]",   
"subject\_spatial\_change": "xxx [change in subject position]",   
"continuity\_basis": ["xxx [evidence supporting continuity]"]   
},   
"visual\_effects": ["xxx [visible effect]"],   
"onscreen\_text\_refs": ["xxx [related text item ID]"],   
"audio\_event\_refs": ["xxx [related audio event ID]"],   
"certainty": "certain|uncertain",   
"uncertainty\_reason": "xxx [reason when uncertain]"   
1   
"subject\_tracks": [   
"subject\_id": "video\_1\_subject\_1",   
"first\_segment": "video\_1\_segment\_1",   
"last\_segment": "video\_1\_segment\_1",   
"stable\_identity\_anchors": ["xxx [stable identity cue]"],   
"actions\_in\_order": ["xxx [ordered subject action]"],   
"state\_changes\_in\_order": ["xxx [ordered subject state change]"],   
"certainty": "certain|uncertain",   
"uncertainty\_reason": "xxx [reason when uncertain]"   
1   
"audio": {   
"present": false,   
"speech": [   
"id": "video\_1\_audio\_1",   
"time\_range": {   
"start\_seconds": "xxx",   
"end\_seconds": "xxx"   
},   
"speaker\_subject\_id": "xxx [speaker subject ID if identifiable]",

```csv
"source_type": "visible_speaker|offscreen_speaker|voiceover|unknown",
"language": "xxx [spoken language]",
"transcript": "xxx [verbatim speech transcript]",
"vocal_characteristics": ["xxx [audible vocal characteristic]"],
"certainty": "certain|uncertain",
"uncertainty_reason": "xxx [reason when uncertain]"
}
1,
"music": [
{
"id": "video_1_music_1",
"time_range": {
"start_seconds": "xxx",
"end_seconds": "xxx"
},
"description": ["xxx [audible music characteristic]"],
"certainty": "certain|uncertain",
"uncertainty_reason": "xxx [reason when uncertain]"
}
],
"ambient_sounds": ["xxx [audible ambient sound]"],
"sound_effects": ["xxx [audible sound effect]"],
"audio_visual_relations": ["xxx [audio-visual synchronization relation]"]
},
"uncertainties": ["xxx [asset-level uncertainty]"]
}
1,
"cross_asset_relations": [
1
"source": {
"asset_id": "image_1",
"entity_id": "image_1_subject_1"
},
"target": {
"asset_id": "video_1",
"entity_id": "video_1_subject_1"
},
"relation": "same_entitylpossibly_same_entitylsame_scenelpossibly_same_scenelsame_object|
possibly_same_object|style_similaritylother",
"evidence": ["xxx [evidence for the cross-asset relation]"],
"certainty": "certainluncertain",
"uncertainty_reason": "xxx [reason when uncertain]"
}
1,
"asset_anchor_summary": ["xxx [instruction-relevant asset anchor]"],
"uncertainties": ["xxx [global uncertainty]"]
一
```

## B.2.2 Three-Dimension Evaluation Outputs

The evaluator uses separate fixed outputs for Semantic Fidelity, Internal Consistency, and Directorial Creation.

## Semantic Fidelity Output Format

```jsonl
"overall_verdict": "good|acceptable|bad",
"semantic_fidelity_score": "xxx [score from 0 to 100]",
"primary_issue": "genre_stylelsubject|action_narrative|dialogue|sound|camera|lighting_color|scenel
onscreen_text|none",
"summary": "",
"dimensions": {
"genre_style": {"status": "consistent|lossylaltered|not_present", "severity": "none|minor|major",
evidence before": "", "evidence after": "", "note": ""},
"subject": {"status": "consistent|lossylaltered|not_present", "severity": "none|minor|major",
evidence_before": "", "evidence_after": "", "note": ""},
"action_narrative": {"status": "consistent|lossylaltered|not_present", "severity": "none|minor|major",
evidence_before": "", "evidence_after": "", "note": ""},
"dialogue": {"status": "consistent|lossylaltered|not_present", "severity": "none|minor|major",
evidence_before": "", "evidence_after": "", "note": ""},
"sound": {"status": "consistent|lossylaltered|not_present", "severity": "none|minor|major",
evidence_before": "", "evidence_after": "", "note": ""},
"camera": {"status": "consistent|lossylaltered|not_present", "severity": "none|minor|major",
evidence_before": "", "evidence_after": "", "note": ""},
"lighting_color": {"status":"consistent|lossylaltered|not_present","severity": "none|minor|major",
evidence_before": "", "evidence_after": "", "note": ""},
"scene": {"status": "consistent|lossylaltered|not_present", "severity": "none|minor|major",
evidence_before": "", "evidence_after": "", "note": ""},
```

"onscreen\_text": {"status": "consistent|lossylaltered|not\_present", "severity": "none|minor|major", "   
evidence\_before": "", "evidence\_after": "", "note": ""}   
}   
1

## Internal Consistency Output Format

1   
"overall\_verdict": "good|acceptable|bad",   
"internal\_consistency\_score": "xxx [score from 0 to 100]",   
"primary\_issue": "subject\_continuitylaction\_narrative\_logic|temporal\_state\_continuitylspatial\_consistencyl   
physical\_plausibilitylaudio\_visual\_sync|none",   
"summary": "",   
"dimensions": {   
"subject\_continuity": {"status": "consistent|ambiguouslinconsistent|not\_present", "severity":"   
none|minor|major", "evidence": "", "note": ""},   
"action\_narrative\_logic": {"status":"consistent|ambiquous|inconsistent|not\_present", "severity":"   
none|minor|major", "evidence": "", "note": ""},   
"temporal\_state\_continuity": {"status": "consistent|ambiguouslinconsistent|not\_present", "severity": "   
none|minor|major", "evidence": "", "note": ""},   
"spatial\_consistency": {"status": "consistentlambiguouslinconsistent|not\_present", "severity": "   
none|minor|major", ""evidence": "", "note": ""},   
"physical\_plausibility": {"status": "consistent|ambiguous|inconsistent|not\_present","severity":"   
none|minor|major", "evidence": "", "note": ""},   
"audio\_visual\_sync": {"status": "consistent|ambiguouslinconsistent|not\_present", "severity":"   
none|minor|major", "evidence": "", "note": ""}   
}   
一

## Directorial Creation Output Format

```csv
"overall_verdict":"good|acceptable|bad",
"directorial_creation_score": "xxx [score from 0 to 100]",
"major_criterion_count": 0,
"minor_criterion_count": 0,
"primary_strength": "storyboard_timinglcamera_design|editing_transition_design|lighting_color|
story_richness_interest|core_prompt_centralitylsound_design|subject_performance_specificityl
visual_style|none",
"primary_issue": "storyboard_timinglcamera_design|editing_transition_design|lighting_colorl
story_richness_interest|core_prompt_centralitylsound_design|subject_performance_specificityl
visual_style|none",
"summary": ""
"dimensions": {
"storyboard_timing": {
"status": "completeldeficient",
"severity": "none|minor|major",
"missing_elements": [],
"evidence":""
"note":""
},
"camera_design": {
"status": "completeldeficient",
"severity":"none|minor|major",
"missing_elements": [],
"evidence": ""
"note":""
},
"editing_transition_design": {
"status":"completel|deficient",
"severity": "none|minor|major",
"missing_elements": [],
"evidence": "",
"note": ""
},
"lighting_color": {
"status":"completel|deficient",
"severity": "nonel|minor|major",
"missing_elements": [],
"evidence":
"note":""
},
"story_richness_interest": {
"status":"completel|deficient",
"severity": "none|minor|major",
"missing_elements": [],
```

```csv
"evidence": "",
"note": ""
},
"core_prompt_centrality": {
"status": "completeldeficient"
"severity": "none|minor|major",
"missing_elements": [],
"evidence": "n
"note": ""
},
"sound_design": {
"status": "completeldeficient",
"severity": "none|minor|major",
"missing_elements": [],
"evidence": "",
"note": ""
}1
"subject_performance_specificity": {
"status": "completeldeficient",
"severity": "none|minor|major",
"missing_elements": [],
"evidence": "",
"note": ""
},
"visual_style": {
"status": "completeldeficient",
"severity": "none|minor|major",
"missing_elements": [],
"evidence": "",
"note": ""
一
}
一
```

## B.3 Representative Case

We provide one representative case together with the outputs from all evaluated models for direct comparison.

## Input Prompt

A sweet 9-year-old boy stands alone in a quiet neighborhood garden, looking slightly downcast. Moments later, his big brother walks up from behind with a kind smile and softly wraps him in a comforting embrace. The child's expression instantly brightens with joy. Warm afternoon light, vibrant animated style, fluid motion, touching and uplifting atmosphere, 3D family-film-style characters.

Duration: 15 seconds.

## Model outputs.

## WanPE-397B

The video consists of 4 shots. This 3D Animation, a work in the style of a Disney-Esque 3D Animation, is a short scene depicting a touching moment between two brothers. The overall art style is Cartoon/Animated, featuring characters with Exaggerated Semi-Realistic Proportions, rendered with Stylized PBR materials and illuminated by Stylized Realistic Lighting to create a warm, vibrant, and family-friendly atmosphere. The story shows a 9-year-old boy who is comforted by his older brother in a neighborhood garden. The scene is set in a quiet, sunlit neighborhood garden. A low, white picket fence lines the back of the garden, with lush green hedges and blooming flowers visible beyond it. A small, leafy tree stands on the left side of the scene. The ground is covered in soft, well-kept grass. The scene is bathed in the warm, golden light of a late afternoon. The video scene is set at late afternoon, the lighting overall led by soft Sunny Light, the image's light-dark contrast is medium-low, presenting a bright mid-tone characteristic. Color saturation is high, overlaid with a medium-intensity Pale Warm Yellow filter, the whole leaning toward a warm tone tendency, together creating a warm and uplifting emotional atmosphere. The narrative is presented from a third-person objective perspective using standard cinematic animation techniques with a measured editing pace, while the audio landscape combines a gentle, evolving piano score and subtle ambient sounds to create a touching and uplifting atmosphere.

Shot 1 [00.00-03.50s] The scene opens with a static, eye-level medium shot with a deep depth of field, employing a center composition. In the middle of the frame, a 9-year-old boy with a sweet face and large eyes stands alone on the soft, green grass. He wears a simple blue t-shirt and khaki shorts. The key light source is the soft, warm-yellow Sunny Light, forming a front side light effect on the boy and presenting a medium lighting ratio. The background features the white picket fence, lush green hedges, and blooming flowers, all softly lit by the afternoon sun. The boy performs an action of standing downcast. With his shoulders slightly slumped, he looks down at the ground, his expression downcast and lonely. His body remains still as a gentle, melancholic piano melody begins, accompanied by the faint, ambient sound of birds chirping and a soft breeze rustling leaves.

Shot 2 [03.50-06.50s] After a hard cut, the camera shifts to a medium long shot from a low angle, with a deep depth of field. The camera then performs a slow, stable dolly-in movement, moving forward to focus on the approaching character. The 9-year-old boy remains in the midground, his back partially to the camera. From the background, his big brother, a teenager with a kind, reassuring smile, wearing a gray hoodie and jeans, walks into the frame. The warm, golden sunlight creates a soft rim light on his form. The brother performs an action of walking towards the boy. He moves with a calm, steady gait, his soft footsteps audible on the grass. As he approaches, the piano melody continues, its tone becoming warmer and more hopeful.

Shot 3 [06.50-10.50s] A hard cut transitions to a static, over-the-shoulder medium close-up, with a shallow depth of field that focuses on the interaction. The 9-year-old boy's back is in the foreground, while the big brother is in the midground. The brother performs an action of embracing the boy. With a gentle, fluid motion, he wraps his arms around the boy from behind, the movement accompanied by the soft rustle of clothing. He rests his chin on the boy's shoulder, his expression full of warmth. The piano melody swells gently, becoming more prominent and emotional, as the ambient sounds of birds and breeze continue.

Shot 4 [10.50-15.00s] A final hard cut shifts to a close-up of the boy's face, captured at eye level with a shallow depth of field. The camera then performs a slow, stable zoom-in, tightening the frame on his expression. The boy is in the center of the frame, held in his brother's embrace. The boy's expression changes from downcast to joyful. Initially, his face is still, but then his eyes widen slightly, and a bright, genuine smile spreads across his face, his expression instantly brightening with joy. The camera slowly pushes in, emphasizing this emotional shift. The piano melody reaches a warm, uplifting crescendo, holding the note before beginning a slow fade, as the ambient sounds of the garden continue.

## WanPE-35B

The video consists of 3 shots. This 3D family-film-style scene, in a warm, vibrant, and animated style, depicts a touching moment where a big brother comforts his downcast younger brother in a garden. The overall visual style is defined by a 3D Animation genre with a 3D family-film-style, featuring characters with a 3D family-film-style, rendered with a vibrant, polished, and cinematic quality. The scene is set in a quiet, well-kept neighborhood garden during the afternoon. The narrative is presented from a third-person objective perspective, using a standard cinematic animation style with a slow, deliberate editing pace that emphasizes the emotional connection. The audio landscape is composed of a gentle, warm piano score, ambient garden sounds, and a single line of soft, comforting dialogue, all contributing to a warm, uplifting, and touching atmosphere.

Shot 1 [00.00-04.00s] The scene opens with a medium shot at an eye-level angle, employing a center composition. The scene is set in a quiet, well-kept neighborhood garden during the afternoon, with a low brick wall and a wooden fence in the background, and a lush green lawn in the foreground. The video scene is set at day, the lighting overall led by soft Sunny Light, the image's light-dark contrast is medium-low, presenting a moderate mid-tone characteristic. Color saturation is high, overlaid with a medium intensity Pale Warm Yellow filter, the whole leaning toward a warm tone tendency, together creating a warm and uplifting emotional atmosphere. A 9-year-old boy, with a clean, innocent face, stands alone in the center of the frame, his body facing forward. He wears a simple t-shirt and shorts. The boy is looking slightly downcast. His shoulders are slightly slumped, and his expression is somber as he looks down at the ground, his posture conveying a sense of sadness. The camera performs a very slow, small-amplitude dolly-in, gradually tightening the frame on the boy, which enhances the feeling of his isolation. The key light source is the soft, warm-yellow afternoon sunlight, forming a front side light effect on the boy, presenting a medium lighting ratio. The background is softly lit by the same ambient daylight. The audio landscape is established with a gentle, warm piano melody, accompanied by the soft, ambient sound of a gentle breeze and distant birds chirping, creating a quiet, contemplative mood.

Shot 2 [04.00-09.00s] A hard cut transitions to a static, eye-level medium shot, maintaining the warm, sunny lighting. The 9-year-old boy remains in the center of the frame, still looking downcast. A 15-year-old big brother, with a kind, caring face, walks into the frame from the right, moving behind the boy. He wears a casual t-shirt and jeans. The big brother approaches with a gentle, kind smile, his movements fluid and deliberate. As he reaches the boy, he gently wraps his arms around the boy's shoulders, pulling him into a comforting embrace. The action is smooth and continuous, with the brother's arms moving in a gentle arc. The key light source is the soft, warm-yellow afternoon sunlight, forming a front side light effect on both boys, presenting a medium lighting ratio. The background is softly lit by the same ambient daylight. The piano melody continues, its warmth increasing slightly, and the sound of the brother's soft footsteps on the grass is heard, followed by the gentle rustle of clothing as he embraces the boy.

Shot 3 [09.00-15.00s] A hard cut shifts to a close-up of the 9-year-old boy's face, captured at an eye-level angle. The boy is being embraced by his brother, who is just out of frame. The boy's expression instantly brightens with joy. His eyes, previously downcast, lift to look up towards his brother, and a wide, genuine smile spreads across his face, his eyes crinkling with happiness. The camera performs a slow, small-amplitude dolly-in, emphasizing the boy's joyful reaction. The key light source is the soft, warm-yellow afternoon sunlight, forming a front side light effect on the boy, presenting a medium lighting ratio. The background is softly lit by the same ambient daylight. The piano melody swells to a warm, uplifting crescendo, and the ambient sound of a gentle breeze is present. The big brother speaks, his voice soft and comforting, "It's okay." The sound of a soft, contented sigh from the boy is heard, completing the emotional arc of the scene.

## H3-Context-IR

integrated\_multimodal\_description: [Shot 1] 3D-animated, a medium shot with a slow push in on a 9-year-old boy standing alone in a stylized, vibrant neighborhood garden. The young boy features oversized, expressive hazel eyes, slightly messy chestnut-brown hair, and is dressed in a yellow-and-white striped cotton t-shirt tucked into blue denim overalls with bright red sneakers. He stands quietly on a winding cobblestone path, surrounded by oversized blooming blue hydrangeas and a pristine white picket fence in the background. Warm, golden afternoon sunlight filters through a dense canopy of green leaves above, casting distinct, dappled light rays across his downcast face. His shoulders slump forward as his gaze remains fixed on the ground, his right red sneaker lightly tapping against a small grey pebble. The camera pushes in steadily, emphasizing the boy's drooping posture and melancholic expression. [ Shot 2] At 00:07.500, the camera cuts to a medium wide arc shot, circling the pair slowly as the boy's older brother steps into the frame from the background. The brother, a tall teenager with neatly parted brown hair, wears a thick forest-green hoodie and khaki trousers. He approaches the young boy from behind with a gentle, wide smile on his face. Upon reaching the child, the brother drops to one knee and softly wraps both arms around the boy's shoulders in a tight, comforting embrace. As the camera completes its arc to fully reveal the young boy's face, his expression instantly transforms. His large eyes widen, his eyebrows raise, and a massive, toothy smile breaks across his face. The golden afternoon light catches the edges of the brother's green hoodie and the boy's brown hair, creating a bright, glowing rim light around them as they hold the embrace.

overall\_soundscape: A continuous ambient track of a gentle breeze and light bird chirps fills the background. In the foreground, a distinct, repetitive scrape and tap is clearly heard as the boy's sneaker hits the cobblestone and the pebble. This is followed by the audible, rhythmic thud of the older brother's footsteps approaching on the stone path, which transitions into a pronounced, loud rustle of heavy cotton fabric as the brother wraps his arms around the boy.

non\_diegetic\_music: Solo acoustic quitar accompanied by a gentle glockenspiel, slow tempo. The instrumentation begins with sparse, sustained notes, gradually building in volume and shifting to a bright, major-key chord progression in the second half, with no percussion.

## LTX-2.5-PE

A medium shot frames a sweet 9-years-old boy with light brown hair and a pale blue t-shirt standing alone in a quiet, lush neighborhood garden filled with vibrant green foliage and colorful flowers, his gaze directed slightly downward, captured from a slightly low-angle perspective under warm afternoon light that casts soft shadows, the scene rendered in a vibrant animated style with fluid motion and rich, saturated film-grade colors, accompanied by a soft, gentle piano melody, a moment later, a tall, smiling big brother with dark hair and a casual light-colored shirt walks up from behind the boy, the camera smoothly tracks backward as the brother approaches, and he softly wraps his arms around the younger boy in a comforting embrace, the camera remains static as the embrace occurs, the boy's downcast expression instantly shifts to a bright, joyful smile, the warm cinematic lighting enhances the texture of the leaves and the soft glow on their faces, creating a touching and uplifting atmosphere with 3D family film-style characters and a subtle swell in the background music.

## LingBot-Video-PE

{"comprehensive\_description": {"scene\_content\_description": "In a lush, sun-drenched garden, a young boy with short brown hair and a white t-shirt stands alone, looking down with a somber expression. The background is filled with vibrant green foliage and a large, leafy tree. As the scene progresses, an older boy with curly brown hair, wearing a light blue t-shirt, enters from the right and approaches the younger boy. He gently places his hands on the younger boy's shoulders and then wraps his arms around him in a warm, comforting hug. The younger boy's expression shifts from sadness to a bright, joyful smile as he leans into the embrace. The lighting is warm and golden, suggesting late afternoon, creating a tender and emotional atmosphere.",

"camera\_movement\_description": "The camera remains essentially stationary throughout the video, maintaining a medium shot that focuses on the two boys. There is a very slight, natural handheld tremor, but no intentional panning, tilting, or zooming occurs."},

"camera\_info": {"color": "Warm", "frame\_size": "Medium", "shot\_type\_angle": "Low angle", "lens\_size": " Medium", "composition": "Center", "lighting": "Hard light", "lighting\_type": "Daylight"}, "world\_knowledge": [],

"prominent\_elements": [{"name": "young boy", "description": "A young boy with short brown hair and a white tshirt, appearing sad at the start of the video.", "actions": [{"timestamp": "[0.0s - 4.5s]", "action": "stands still, looking down with a sad expression"}, {"timestamp": "[4.5s - 10.5s]", "action": "remains still as the older boy approaches and hugs him"}, {"timestamp": "[10.5s - 13.0s]", "action": "smiles broadly and leans into the hug"}, {"timestamp": "[13.0s - 15.0s]", "action": "stands still, smiling while being hugged"}], "location": "center of the frame", "relative\_size": "medium", "shape\_and\_color": "humanoid shape, wearing a white t-shirt", "texture": "soft skin, cotton t-shirt", "appearance\_details ": "short brown hair, white t-shirt with a small graphic on the chest", "relationship": "being comforted by the older boy", "orientation": "facing forward, slightly tilted down", "pose": "standing upright, then leaning into a hug", "expression": "changes from sad and downcast to happy and smiling", "clothing": "white t-shirt", "gender": "male", "skin\_tone\_and\_texture": "fair, smooth skin"},

{"name": "older boy", "description": "An older boy with curly brown hair and a light blue t-shirt, acting as a comforting figure.", "actions": [{"timestamp": "[4.5s - 7.0s]", "action": "enters from the right and walks toward the younger boy"}, {"timestamp": "[7.0s - 10.5s]", "action": "places hands on the younger boy's shoulders and hugs him"}, {"timestamp": "[10.5s - 15.0s]", "action": "holds the younger boy in a hug, smiling"}], "location": "enters from the right, moves to the center", "relative\_size": "medium", "shape\_and\_color": "humanoid shape, wearing a light blue t-shirt", "texture": "curly hair, cotton t-

shirt", "appearance\_details": "curly brown hair, light blue t-shirt", "relationship": "comforting the younger boy", "orientation": "facing left toward the younger boy", "pose": "walking, then standing and hugging", "expression": "kind and smiling", "clothing": "light blue t-shirt", "gender": "male", " skin\_tone\_and\_texture": "fair, smooth skin"}]}

## SCMaPR

A sweet 9-year-old boy stands alone in a quiet neighborhood garden, looking slightly downcast with a slightly downcast expression under warm afternoon light. His older brother walks up gently from behind and softly wraps his arms around the younger boy in a comforting embrace. The younger boy's face immediately brightens with joy in direct response to this tender physical contact. The scene is rendered in a vibrant animated style with fluid motion and 3D family-film-style characters to create a touching and uplifting atmosphere.

## VPO

A sweet 9-year-old boy with tousled hair and a plain T-shirt stands alone in a quiet neighborhood garden, surrounded by blooming flowers and a wooden bench under the warm sunlight. His posture is slightly slouched, and his gaze is cast downward, suggesting a moment of quiet melancholy. The scene captures the soft rustle of leaves and the gentle sway of nearby trees under the golden afternoon light, adding to the serene yet slightly somber mood. Moments later, his big brother, slightly taller and wearing a casual hoodie, walks up from behind with a warm and kind smile. The boy's expression softens as his older brother approaches, and with a heartfelt gesture, he gently wraps his arms around him in a comforting embrace. The younger boy's face lights up with pure joy, radiating warmth and the unspoken bond of sibling love. The animated style features soft, fluid motion and vibrant colors, enhancing the heartfelt and uplifting atmosphere of the moment. The background includes subtle details like a colorful butterfly fluttering near the flowers and distant chirping birds, reinforcing the wholesome, family-film-style aesthetic.

## Prompt-A-Video-Cog

In a serene 3D neighborhood garden, a cherubic 9-year-old boy with a softly tousled hairdonned a lightweight jacket, standing alone under the embracing branches of a oak tree. His eyes, cast downward, whisper stories of a gentle sorrow. As his stalwart big brother, with a warm smile, wraps him in a comforting embrace, the younger boy's expression transforms, brightening with joy. The brothers enclave is bathed in the warm, honeyed light of a late afternoon sun. The animation is vibrant, with fluid motions, captivating in a style reminiscent of timeless family films. The scene is a powerful testament to the embracing love of a big brother, as the animation exudes a touching, uplifting atmosphere that lasts for 15 captivating seconds.

## Prompt-A-Video-OS

In a vibrant neighborhood garden bathed in warm afternoon light, a 9-year-old boy with a downcast expression sits alone. His big brother, with a kind face, walks into the frame and envelops him in a comforting hug. The boy's demeanor instantly brightens with joy. The video is rendered in a 3D animated style, characteristic of family films, and exudes a touching and uplifting atmosphere. (15 seconds)

## RAPO

a young boy wearing a red shirt and sitting on the ground. He is then seen being hugged by his big brother. The scene is set in a warm and sunny afternoon. The animation is colorful and vibrant. The boy's expression changes from sadness to joy as he is being hugged by his big brother.

## C Additional Experimental Details

## C.1 Implementation Details

We evaluate the ten prompt-enhancement systems listed in Table 2, together with the original prompt as a no-enhancement baseline. All systems are evaluated on T2V-PE, while systems supporting visual inputs are additionally evaluated on I2V-PE and R2V-PE. Proprietary systems are accessed through their official interfaces, and released models are run using their recommended inference settings.

For each case, modality-aware evidence is extracted once and cached for all methods. GPT-5.4 [32] extracts facts for image-only inputs, and Qwen3.5-Omni-Plus processes all assets for cases containing video references. GPT-5.4 then evaluates every enhanced prompt using the same criterion-specific rubrics. We report criterion pass rates and aggregate Semantic Fidelity, Internal Consistency, and Directorial Creation scores. The additional experiments analyze temporal complexity, reference complexity, and output length. Human studies further test prompt-level alignment and downstream utility using videos generated by Wan3.0 and MiniMax-H3. For downstream video generation, we access Wan3.0 through its online API with prompt enhancement disabled, while MiniMax-H3 is run locally on eight NVIDIA A100 GPUs.

## C.2 Length-Stratified Analysis

Prompt enhancement often transforms concise inputs into substantially longer descriptions. To test whether output length alone yields better descriptive adequacy, we analyze H3-Context-IR separately on each task. Valid outputs are sorted by token length and divided into four equal-sized quartiles, from the shortest (L1) to the longest (L4).

Table 8 Length-stratified results for H3-Context-IR. Quartiles are defined independently within each task. DC denotes Directorial Creation.
<table><tr><td>Task</td><td>Metric</td><td>L1</td><td>L2</td><td>L3</td><td>L4</td></tr><tr><td>T2V-PE</td><td>Avg. tokens DC</td><td>394.7 79.32</td><td>490.9 78.60</td><td>584.9 77.35</td><td>861.1 74.13</td></tr><tr><td>I2V-PE</td><td>Avg. tokens</td><td>819.1</td><td>945.7</td><td>1066.6</td><td>1382.0</td></tr><tr><td></td><td>DC</td><td>75.00</td><td>75.20</td><td>75.99</td><td>67.20</td></tr><tr><td>R2V-PE</td><td>Avg. tokens</td><td>809.9</td><td>1015.7</td><td>1164.4</td><td>1479.3</td></tr><tr><td></td><td>DC</td><td>79.37</td><td>73.75</td><td>72.07</td><td>69.28</td></tr></table>

Across all three tasks, the longest quartile underperforms the shortest by 5.19 to 10.09 DC points. Thus, longer outputs do not consistently provide better directorial guidance. This result is consistent with the evaluation design: DC rewards concrete, relevant, and executable specifications for temporal planning, cinematography, narrative progression, subject performance, and sound, rather than verbosity or repeated template language.

## D Human Expert Annotation

## D.1 Pairwise Annotation Protocol

The human study evaluates alignment at two levels: (i) the quality of enhanced prompts and (ii) the quality of downstream videos generated from those prompts. We recruit 60 experts and provide task-specific training before annotation. Each expert evaluates approximately 50 T2V-PE pairs, 20 I2V-PE pairs, and 10 R2V-PE pairs. The comparison pool contains 11 systems for T2V-PE, 4 for I2V-PE, and 3 for R2V-PE, including the original prompt as the no-enhancement baseline in every task.

## Pairwise judgment procedure.

For enhanced-prompt evaluation, annotators inspect the original instruction, target duration, reference assets when applicable, and an anonymized pair of enhanced prompts. Annotators make a separate choice for Semantic Fidelity, Internal Consistency, and Directorial Creation. For each dimension, they select the better candidate, both good, or both bad. For downstream-video evaluation, annotators inspect the source request and references together with an anonymized pair of generated videos and make the same dimension-specific judgment. Candidate identities are hidden, and the left and right positions are randomized independently for every comparison.

We aggregate the pairwise choices with a Bradley-Terry model to obtain a system ranking for each task, evaluation level, and dimension. We then compute the Spearman correlation between this expert-derived ranking and the corresponding automatic dimension scores. The resulting coefficients are reported in Table 5.

## Quality control and agreement.

All experts complete training before formal annotation. We test each annotator's selections for a significant left or right position preference. Annotators exhibiting significant position bias are excluded, together with all of their judgments. before fitting the Bradley-Terry models and computing the correlations.

## D.2 Annotation Interface

The interface presents the original user prompt and target duration at the top, followed by reference assets for I2V-PE and R2V-PE. Prompt-level evaluation displays two anonymized enhanced prompts, while video-level evaluation replaces them with two generated videos. The three evaluation dimensions are shown as independent rating rows.

Interface behavior. Candidate identities are replaced by A and B, and their left and right positions are randomized independently for each case. Video-level evaluation supports synchronized playback, individual playback, and replay. The interface requires one response for each dimension before submission and allows annotators to revise their selections before completing the case. Figures 5–7 show the annotation interfaces for the three PE settings.

## E Ethics, Privacy, and Licensing

PEBench prompts are generated from an expert-designed taxonomy and undergo manual verification. During curation, we remove personal information, identifiable private individuals, sensitive political content, and material that may raise privacy or cultural concerns. Reference assets originate from public benchmarks or model-assisted generation. We retain source and license metadata and release only assets permitted for redistribution, providing source identifiers for restricted material. Videos generated for downstream validation are used only for evaluation and are not included in the benchmark release. We release the verified prompts, legally shareable reference assets, and evaluation framework to support reproducible research.

## F Limitations

One limitation is the scale of the benchmark. Although 1,100 expert-verified cases provide broad coverage across three PE settings, 35 task types, and diverse conditioning patterns, they cannot represent the full range of rapidly evolving video-generation requests. Future work will expand PEBench with additional domains, languages, reference configurations, and emerging control formats, enabling more comprehensive evaluation and supporting the development of increasingly general prompt enhancers.

A second limitation is the comparatively limited model coverage for I2V-PE and R2V-PE. Unlike T2V-PE, few existing methods support multimodal prompt enhancement, and only a small number of proprietary systems, such as H3-Context-IR, provide accessible PE interfaces. Consequently, our I2V-PE and R2V-PE evaluations include fewer baselines. We will continue updating PEBench as more multimodal PE methods become available, enabling broader comparisons across open and proprietary systems.

![](images/be3e2267daa42606f029be157d02f56d057e2250a11061f288169cef13b8024a.jpg)  
T2V-PE Annotation Interface  
(b) Video-level evaluation  
Figure 5 T2V-PE annotation interface. Anonymous prompts and generated videos are compared against the text instruction and target duration.

I2V-PE Annotation Interface  
![](images/6c3a2a0769719da445615f2daa1ca1842bce085f7b12c303a859f264d2f7cb97.jpg)  
(b) Video-level evaluation  
Figure 6 I2V-PE annotation interface. The initial-frame image is displayed with the instruction for prompt-level and video-level comparison.

R2V-PE Annotation Interface  
![](images/7213607ac96026fdfb596c1549c9f8cd5d6ea3425bd11b4a036b19d494aa9853.jpg)  
(a) Prompt-level evaluation

![](images/4573030ee1ceddf03e2436b6c804c0474de28f235d49e7efa376367c0e855bf7.jpg)  
(b) Video-level evaluation  
Figure 7 R2V-PE annotation interface. All reference images and videos are displayed to support cross-asset grounding judgments.
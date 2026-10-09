# Transforming Image Editors into Video Editors

Feng Wang<sup>1,2∗</sup>, Zijie Li<sup>1</sup>, Ceyuan Yang<sup>1</sup>, Alan Yuille<sup>2</sup>, Peng Wang<sup>1</sup> <sup>1</sup>ByteDance Seed <sup>2</sup>Johns Hopkins University

## Abstract

Recent image editing systems have achieved impressive semantic understanding, visual fidelity, and instruction-following ability, while video editing remains substantially more difficult and costly. In this paper, we present a simple alternative to end-to-end video editing: instead of training a monolithic video editor, we transform a strong image editor into a video editor through anchor-based generation. Our key insight is that video editing can be decomposed into two subproblems: editing a sparse set of keyframes and propagating those edits across time. Based on this observation, we propose Anchor-based Video Editing (AVE), a two-stage framework in which a powerful image editor first performs composed editing on selected keyframes, and a motion-guided image-to-video diffusion model then generates the final video by treating the edited keyframes as fixed anchors. This design directly inherits the strengths of modern image editors while avoiding expensive end-to-end video editing training. Experiments on IVEBench and VIE-Bench show that AVE achieves strong performance in instruction following, temporal consistency, and content fidelity. Further ablations reveal that final video editing quality is strongly correlated with the quality of the image editor, suggesting that future progress in video editing may come from stronger image editing foundations and lightweight transfer to video. Code is available at https://github.com/wangf3014/AVE.

## 1 Introduction

Recent advances in image editing have pushed the field to a remarkable level of maturity. Modern image editing systems now exhibit strong semantic understanding, high visual fidelity, and precise instruction-following ability, often reaching a quality level suitable for real-world applications (OpenAI, 2025; Google, 2025; Deng et al., 2025; Qwen Team, 2025; Brooks et al., 2023; Black Forest Labs, 2025; Team Seedream et al., 2025; Liu et al., 2025; Xiao et al., 2025). In contrast, video editing still lags noticeably behind. Compared with image editing, video editing models must handle much longer spatiotemporal contexts, which makes them significantly more expensive to train. Meanwhile, collecting high-quality video editing data is also considerably harder, due to the need for temporal consistency, motion diversity, and reliable instruction alignment across frames. As a result, while image editing systems have rapidly advanced to a high level of quality and controllability, mainstream video editing models still remain clearly behind in visual quality, semantic consistency, and instruction-following performance (Kling Team et al., 2025; Jiang et al., 2025; Bai et al., 2026; Cheng et al., 2024; Wu et al., 2025; Lin et al., 2026; Rao et al., 2026; Zhang et al., 2026a; Runway, 2025; Pika, 2026; Zi et al., 2025; Li et al., 2025; Tan et al., 2025; DecartAI Team, 2025; Wei et al., 2026; Mou et al., 2025; Liao et al., 2025; He et al., 2026; Chen et al., 2026a; Zhang et al., 2026b).

To narrow this gap, this paper studies how to transfer the editing capability of image systems to video models at minimal cost. Through a series of studies, we surprisingly find that this transfer is much simpler than it appears to be. In particular, although video editing is a 3D video-to-video task, its primary challenge remains the underlying 2D image-to-image editing problem. By contrast, handling temporal information, such as preserving or modifying motion, can be achieved with relatively little additional training cost. Motivated by this observation, we introduce Anchor-based Video Editing (AVE), a new instruction-guided video editing paradigm which explicitly decomposes the task into two stages: Composed Image Editing and Keyframe Interpolation.

<table><tr><td>Paradigm</td><td>Consistency</td><td>Frame Quality</td><td>Instruction Following</td><td>Training Cost</td></tr><tr><td>End-to-end video editing</td><td>Good</td><td>Good</td><td>Depends on training 商</td><td>Expensive B</td></tr><tr><td>Per-frame editing</td><td>Bad X</td><td>Excellent</td><td>Bad X</td><td>Cheap </td></tr><tr><td>Two-stage editing of AVE (ours)</td><td>Good√</td><td>Excellent🌟</td><td>Great and robust🌟</td><td>Normal </td></tr></table>

Table 1: Intuitive comparison of three video editing paradigms. Our model attains favorable video consistency, frame-wise quality and instruction-following abilities with a normal cost.

Specifically, given an input video, we first extract a sparse set of keyframes and jointly edit them with a powerful image editing model. By performing composed image editing over multiple keyframes, we preserve scene-level and semantic consistency across the edited frames while fully benefiting from the strong instruction-following and visual generation capabilities of modern image editors. We then use an image-to-video diffusion model (e.g., Wan2.2-I2V (Wan Team et al., 2025)) to generate the target video, where the edited keyframes serve as fixed anchors and the generation is constrained by the motion representation of the input video. In this way, the model is required to preserve the edited content at key moments while propagating the edits smoothly across intermediate frames. This decomposition provides a simple and scalable path for transforming image editors into video editors, allowing video editing systems to directly inherit the rapid progress of image editing models without requiring expensive end-to-end video editing training.

Compared with the common end-to-end video editing frameworks (e.g., VACE (Jiang et al., 2025)) and the most naive per-frame editing strategy, the main advantages of AVE can be summarized as follows. First, the image editing model in Stage-1 is fully decoupled from Stage-2, which allows AVE to directly leverage any mature image editor to achieve the best possible frame-wise editing quality. As a result, AVE can seamlessly inherit the rapid progress of image editing systems, including stronger semantic understanding, better visual fidelity, and more accurate instruction following, without any modification to the image editor itself. Second, unlike naive per-frame editing, which often suffers from severe flickering and temporal inconsistency, AVE performs interpolation between edited keyframes using a video diffusion model, enabling smooth propagation of edits across intermediate frames while maintaining strong temporal consistency. Third, AVE is highly cost-efficient to train. In contrast to end-to-end video editing frameworks that require substantial training resources, AVE only needs to train a lightweight motion encoder, while keeping all other components fixed and initialized from pretrained models. In our study, we find that a small motion encoder and a relatively short context length are already sufficient to capture the motion information needed for high-quality video editing. The comparison is summarized in Table 1.

Extensive experiments validate the effectiveness of AVE from multiple perspectives. On IVEBench (Chen et al., 2026b), AVE achieves the best performance on both the short and long subsets across all reported metrics, showing that the proposed decomposition leads to a strong and well-balanced video editing capability rather than improving only a single aspect of the task. In particular, AVE is especially strong in motion-related metrics, which supports our design choice of using a motion-guided video interpolation stage to propagate edits smoothly while preserving the original temporal dynamics. AVE also obtains clear gains in semantic consistency and content fidelity, suggesting that high-quality edited anchors provide a reliable foundation for long-range video generation. On VIE-Bench (Mou et al., 2025), AVE generalizes well across diverse editing types, including add, swap, remove, and style transfer, and remains competitive even against strong closed-source systems. In addition, our ablations reveal two important findings: first, the final video editing quality is strongly correlated with the quality of the Stage-1 image editor, confirming that AVE effectively transfers image editing ability into the video domain; second, AVE is robust to different keyframe selection strategies, with a moderate keyframe budget already sufficient to achieve strong performance. These results consistently support our central claim that, once high-quality edited anchors are available, the remaining image-to-video transfer is much simpler than end-to-end video editing would suggest. We hope this study provides useful insights for future video editing systems.

## 2 Related Work

Instruction-based image editing. Instruction-based image editing has advanced rapidly, evolving from early diffusion-based editors such as InstructPix2Pix (Brooks et al., 2023) and InstructEdit (Wang et al., 2023a) to more capable multimodal systems such as SmartEdit (Huang et al., 2024),

OmniGen (Xiao et al., 2025), Step1X-Edit (Liu et al., 2025), and BAGEL (Deng et al., 2025). Recent open and commercial image systems, including Qwen-Image-Edit (Qwen Team, 2025), FLUX.1 Kontext (Black Forest Labs, 2025), and Nano Banana 2 (Fortin and Alessio, 2026), further improve semantic understanding, visual fidelity, and instruction following. These advances suggest that high-quality editing is increasingly solved in the image domain. Our work is complementary to this line of research: instead of proposing another image editor, we study how to transfer the editing capability of mature image systems into video models with minimal additional cost.

Instruction-guided video editing. Text-guided video editing is substantially more challenging because it must preserve temporal consistency while executing semantic edits. Early methods mainly adapt pretrained image diffusion models through inversion, one-shot tuning, or feature propagation, such as Tune-A-Video (Wu et al., 2023), FateZero (Qi et al., 2023), TokenFlow (Geyer et al., 2024), and InstructVideo (Yuan et al., 2024). More recent feed-forward approaches rely on synthetic datasets and end-to-end training, including InsV2V (Cheng et al., 2024), InsViE (Wu et al., 2025), Ditto (Ba et al., 2026), VACE (Jiang et al., 2025), Kiwi-Edit (Lin et al., 2026), InsEdit (Rao et al., 2026), and SAMA (Zhang et al., 2026a). Concurrent efforts also explore unified or in-context formulations and larger-scale supervision, such as ICVE (Liao et al., 2025), ReCo (Zhang et al., 2026b), and OpenVE-3M (He et al., 2026). While these methods have significantly improved the field, they typically require substantial video-specific training data and computation, and their editing quality still trails the best image editors. In contrast, our method does not treat video editing as a monolithic end-to-end problem. Instead, it explicitly decomposes the task into composed keyframe editing and temporal interpolation, so that the hardest part of the problem can be delegated to a strong image editor.

Image-to-video generation and motion modeling. A related line of work studies image-to-video generation and controllable motion synthesis. Representative approaches include AnimateDiff (Guo et al., 2024), VideoCrafter2 (Chen et al., 2024), VideoComposer (Wang et al., 2023b), and methods that explicitly decouple content and motion (Shen et al., 2024). These models show that motion can often be modeled separately from appearance, and that pretrained image generators can be extended to videos with relatively lightweight temporal modules. AVE builds on this insight but applies it to editing rather than pure generation. Instead of asking a video model to jointly solve semantic editing and temporal reasoning, we use the video model mainly to interpolate between edited keyframe anchors under the motion cues of the source video. This design allows AVE to inherit the rapid progress of image editing models while keeping video-specific training lightweight.

## 3 Method

## 3.1 Overview

As illustrated in Figure 1, given an input video $V ~ = ~ \{ x _ { t } \} _ { t = 1 } ^ { T }$ and a text editing instruction y, our goal is to generate an edited video $\hat { V } = \{ \hat { x } _ { t } \} _ { t = 1 } ^ { T }$ that faithfully follows the instruction while preserving the source motion and temporal coherence. Our key observation is that video editing can be decomposed into two largely independent subproblems: (i) high-quality semantic editing on a sparse set of keyframes, and (ii) temporally coherent interpolation between the edited keyframes. Let $\mathcal { A } = \{ a _ { 1 } , \ldots , a _ { K } \}$ denote the indices of K selected keyframes from the source video. AVE first edits the keyframe set $\mathbf { x } _ { \mathcal { A } } = \{ x _ { a } \} _ { a \in \mathcal { A } }$ with an image editing model $\hat { \mathbf { x } } _ { \mathcal { A } } = \mathcal { E } ( \mathbf { x } _ { \mathcal { A } } , y )$ , and then uses a video diffusion model to generate the final video conditioned on the edited keyframes as fixed anchors and a lightweight motion representation extracted from the source video $\hat { V } = \mathcal { G } ( \hat { \mathbf { x } } _ { A } , \mathbf { m } ( V ) )$ A key design of AVE is that Stage-1 and Stage-2 are fully decoupled: the only interface between them is the edited keyframes. This allows us to improve the frame-wise editing quality by simply replacing the image editor, without retraining the video generator.

## 3.2 Stage-1: Composed Image Editing

Training a multi-image editor from video data. For open-source instantiations, we train a composed image editor by adapting BAGEL (Deng et al., 2025) with fine-tuning. Starting from a private video editing dataset of triplets $( V , y , V ^ { \star } )$ , where $V ^ { \star }$ is the edited target video, we construct multi-image editing training tuples by sampling aligned keyframes from the source and target videos $( \mathbf { x } _ { \mathbf { \mathcal { A } } } , y , \mathbf { x } _ { \mathbf { \mathcal { A } } } ^ { \star } )$

![](images/4007ece18d14c34d5b84942e48dc9d68213262b1b1139261ed9fabcacc9a24c6.jpg)  
Figure 1: Overview of our two-stage pipeline. In Stage-1, we edit a sparse set of keyframes using either open-source or API-based image editors. In Stage-2, an image-to-video diffusion model interpolates between the edited anchors while preserving the source motion via a lightweight motion encoder and anchor clamping.

where $\mathbf { x } _ { A } ^ { \star } = \{ x _ { a } ^ { \star } \} _ { a \in \mathcal { A } }$ denotes the target keyframes corresponding to the same timestamps. In total, we build approximately 1.5M multi-image editing examples from private video editing data.

Let $\mathcal { V } _ { \mathrm { i m g } }$ be the image VAE used by the editor, and let $\mathbf { z } _ { \mathcal { A } } = \mathcal { V } _ { \mathrm { i m g } } ( \mathbf { x } _ { \mathcal { A } } ) , \mathbf { z } _ { \mathcal { A } } ^ { \star } = \mathcal { V } _ { \mathrm { i m g } } ( \mathbf { x } _ { \mathcal { A } } ^ { \star } )$ be the source and target latent codes. We fine-tune BAGEL with the standard diffusion denoising objective:

$$
\mathcal { L } _ { \mathrm { i m g } } = \mathbb { E } _ { \mathbf { z } _ { A } ^ { \star } , \epsilon , \tau } \left[ \left\| \epsilon - \epsilon _ { \phi } ( \alpha _ { \tau } \mathbf { z } _ { A } ^ { \star } + \sigma _ { \tau } \epsilon , \mathbf { z } _ { A } , y , \tau ) \right\| _ { 2 } ^ { 2 } \right] ,\tag{1}
$$

where $\epsilon _ { \phi }$ is the denoiser of the fine-tuned BAGEL, and $( \alpha _ { \tau } , \sigma _ { \tau } )$ are the diffusion coefficients at timestep τ. By training on multiple aligned frames jointly, the editor learns to perform consistent edits across keyframes, which is critical for downstream video generation.

Closed-source alternative. While the above open-source instantiation is useful for controlled experiments, our framework is not tied to any particular image editor. In practice, unless otherwise specified, we instantiate Stage-1 with the latest Nano Banana API (Fortin and Alessio, 2026) to maximize editing quality. For strong closed-source models, we find that the most effective strategy is to edit all selected keyframes within a single multi-turn conversation, which naturally preserves subject identity, style, and scene-level consistency across frames. In the ablation study, we compare different API-based editors as well as open-source image editors within the same Stage-2 pipeline.

## 3.3 Stage-2: Keyframe Interpolation

Stage-2 takes the edited keyframes from Stage-1 and generates a temporally coherent target video using a pretrained image-to-video model. We use Wan2.2-I2V-A14B (Wan Team et al., 2025) as the backbone and keep all its pretrained components frozen, except for a lightweight motion encoder and its associated control branch.

Lightweight motion encoder. Let $\gamma _ { \mathrm { 3 D } } ^ { \mathrm { e n c } }$ and $\gamma _ { \mathrm { 3 D } } ^ { \mathrm { d e c } }$ denote the 3D VAE encoder and decoder of Wan2.2-I2V-A14B. The source video is encoded into latent space as $Z = \mathcal { V } _ { \mathrm { 3 D } } ^ { \mathrm { e n c } } ( V ) \in \mathbb { R } ^ { T \times H \times W \times C }$ We project latent tokens into the motion embedding space: $U _ { t , h , w } ^ { ( 0 ) } = W _ { \mathrm { i n } } Z _ { t , h , w } + e _ { t } ^ { \mathrm { t e m p } } + e _ { h , w } ^ { \mathrm { s p a t i a l } } ,$ where $W _ { \mathrm { i n } } \in \mathbb { R } ^ { C \times d }$ is a learnable linear projection, and $e _ { t } ^ { \mathsf { t e m p } }$ and $e _ { h , w } ^ { \mathrm { s p a t i a l } }$ denote temporal and spatial positional embeddings, respectively. The resulting tensor $U ^ { ( 0 ) } \in \mathbb { R } ^ { T \times H \times W \times d }$ is then processed by a 6-layer factorized spatiotemporal Transformer. Each layer consists of a spatial self-attention block followed by a temporal self-attention block and an MLP. We inject the final motion representation into the frozen Wan2.2 DiT through a ControlNet-style residual branch (Zhang et al., 2023), allowing the generator to preserve source motion while leaving appearance and editing semantics to be determined by the edited anchors. The overall framework of the motion encoder is shown in Figure 2. More details about the architecture and training approach can be found in Appendix.

![](images/1428edf3348bd9491377f5b1fedd4d9792a7be9fc15c26aa3e81d94c9800fa56.jpg)  
Figure 2: Architecture of the lightweight motion encoder used in Stage-2. The encoder projects video latents into a token sequence and applies factorized spatial-then-temporal self-attention to extract motion features that condition the frozen image-to-video backbone.

Anchor-preserving diffusion. Let $D _ { \psi }$ denote the pretrained Wan2.2 DiT. During generation, the edited keyframes are treated as fixed anchors: their latent features are clamped throughout the entire reverse diffusion process and are never denoised. Formally, we define a binary anchor mask

$$
M _ { A } \in \{ 0 , 1 \} ^ { T \times H \times W \times C } ,\tag{2}
$$

where $M _ { \mathcal { A } } ( t ) = 1$ if frame $t \in { \mathcal { A } }$ and 0 otherwise. We encode the anchor frames by Wan2.2’s 3D VAE encoder:

$$
\hat { Z } _ { A } = \mathcal { V } _ { \mathrm { 3 D } } ^ { \mathrm { e n c } } ( \hat { \mathbf { x } } _ { A } ) .\tag{3}
$$

At the initial diffusion step S, the latent video is initialized as

$$
X _ { S } = M _ { \cal A } \odot \hat { Z } _ { \cal A } + ( 1 - M _ { \cal A } ) \odot \epsilon , \qquad \epsilon \sim { \mathcal { N } } ( 0 , I ) ,\tag{4}
$$

that is, anchor frames are directly set to the edited latent anchors, while all non-anchor frames start from random noise. At each reverse diffusion step s, the frozen DiT predicts the noise residual conditioned on the current latent video and the motion control:

$$
\hat { \epsilon } _ { s } = D _ { \psi } ( X _ { s } , h _ { \mathrm { m o t i o n } } ) .\tag{5}
$$

A standard diffusion update produces an intermediate estimate

$$
\tilde { X } _ { s - 1 } = \operatorname { S t e p } ( X _ { s } , \hat { \epsilon } _ { s } ) ,\tag{6}
$$

after which the anchor frames are re-imposed:

$$
X _ { s - 1 } = M _ { \cal A } \odot \hat { Z } _ { \cal A } + ( 1 - M _ { \cal A } ) \odot \tilde { X } _ { s - 1 } .\tag{7}
$$

Therefore, anchor features remain unchanged throughout denoising, while the model fills in all intermediate frames conditioned on both the fixed visual anchors and the source motion representation. After the final denoising step, the edited video is decoded back to pixel space by the 3D VAE decoder:

$$
\hat { V } = \mathcal { V } _ { \mathrm { 3 D } } ^ { \mathrm { d e c } } ( X _ { 0 } ) .\tag{8}
$$

## 4 Experiments

## 4.1 Experimental Setup

We use Wan2.2-I2V-A14B (Wan Team et al., 2025) as the video generation backbone. Unless otherwise specified, all pretrained components in Wan, including the 3D VAE encoder/decoder and the DiT generator, are kept frozen during training. The total number of learnable parameters is 710M, which is about 5% of the diffusion model’s parameter count. In our experiments, we find that only lightweight training on the motion-related components is sufficient to achieve strong motion preservation and frame-to-video generation performance. This is mainly because we already effectively leverage the motion modeling and understanding capability of Wan2.2’s pretrained 3D

<table><tr><td>Model</td><td>Total</td><td>Quality</td><td>Ins. Comp.</td><td>Fidelity</td><td>Motion</td><td>Sem. Consist.</td><td>Content</td></tr><tr><td>Ditto (Bai et al., 2026)</td><td>66.75</td><td>78.12</td><td>49.08</td><td>73.04</td><td>78.99</td><td>24.92</td><td>3.64</td></tr><tr><td>InsV2V (Cheng et al., 2024)</td><td>66.68</td><td>79.58</td><td>38.61</td><td>81.85</td><td>85.55</td><td>24.11</td><td>4.05</td></tr><tr><td>Lucy-Edit-Dev (DecartAI Team, 2025)</td><td>63.53</td><td>82.09</td><td>33.94</td><td>74.55</td><td>67.79</td><td>23.83</td><td>3.83</td></tr><tr><td>VACE (Jiang et al., 2025)</td><td>62.61</td><td>79.83</td><td>25.42</td><td>82.58</td><td>88.59</td><td>23.80</td><td>4.03</td></tr><tr><td>ICVE (Liao et al., 2025)</td><td>60.33</td><td>71.25</td><td>45.35</td><td>64.37</td><td>45.72</td><td>22.86</td><td>3.55</td></tr><tr><td>Omni-Video (Tan et al., 2025)</td><td>58.65</td><td>78.20</td><td>43.64</td><td>54.11</td><td>50.72</td><td>21.98</td><td>2.85</td></tr><tr><td>AnyV2V (Ku et al., 2024)</td><td>57.67</td><td>72.71</td><td>41.72</td><td>58.59</td><td>81.59</td><td>21.59</td><td>2.75</td></tr><tr><td>StableV2V (Liu et al., 2026)</td><td>50.89</td><td>69.17</td><td>42.67</td><td>40.84</td><td>75.13</td><td>19.78</td><td>1.79</td></tr><tr><td>AVE (ours)</td><td>70.58</td><td>82.10</td><td>49.65</td><td>80.01</td><td>89.26</td><td>26.12</td><td>4.10</td></tr></table>

Table 2: IVEBench short subset results. Here we report the overall performance of total score, video quality, instruction compliance, video fidelity, and detailed metrics of motion smoothness, semantic consistency, and content fidelity and leave the remaining scores of IVEBench in Appendix. Our AVE model consistently obtains the best results among the evaluation metrics.

<table><tr><td>Model</td><td>Total</td><td>Quality</td><td>Ins. Comp.</td><td>Fidelity</td><td>Motion</td><td>Sem. Consist.</td><td>Content</td></tr><tr><td>Ditto (Bai et al., 2026)</td><td>65.93</td><td>77.95</td><td>47.85</td><td>71.98</td><td>72.82</td><td>23.42</td><td>3.69</td></tr><tr><td>InsV2V (Cheng et al., 2024)</td><td>65.72</td><td>80.24</td><td>37.41</td><td>79.50</td><td>67.88</td><td>24.06</td><td>4.13</td></tr><tr><td>Lucy-Edit-Dev (DecartAI Team, 2025)</td><td>64.89</td><td>82.10</td><td>31.51</td><td>81.05</td><td>73.46</td><td>23.92</td><td>4.13</td></tr><tr><td>VACE (Jiang et al., 2025)</td><td>61.61</td><td>80.12</td><td>26.73</td><td>77.98</td><td>88.40</td><td>23.56</td><td>3.74</td></tr><tr><td>ICVE (Liao et al., 2025)</td><td>58.77</td><td>71.96</td><td>40.38</td><td>63.98</td><td>48.40</td><td>22.60</td><td>3.48</td></tr><tr><td>Omni-Video (Tan et al., 2025)</td><td>57.09</td><td>77.84</td><td>42.44</td><td>51.00</td><td>55.11</td><td>21.90</td><td>2.59</td></tr><tr><td>AnyV2V (Ku et al., 2024)</td><td>55.05</td><td>72.40</td><td>35.55</td><td>57.20</td><td>82.47</td><td>21.55</td><td>2.65</td></tr><tr><td>StableV2V (Liu et al., 2026)</td><td>50.93</td><td>69.37</td><td>42.09</td><td>41.33</td><td>77.33</td><td>20.41</td><td>1.79</td></tr><tr><td>AVE (ours)</td><td>68.75</td><td>82.15</td><td>48.31</td><td>75.79</td><td>88.95</td><td>24.68</td><td>4.16</td></tr></table>

Table 3: IVEBench long subset results. Similar to the short subset, our model attains the best results across the reported metrics. AVE has good robustness to long-video editing.

VAE and diffusion transformer. In practice, we train the motion encoder and ControlNet on only 500K video editing samples, and do not observe further improvements when scaling up either the data size or the training duration. We train our model on 64 A100 GPUs. We optimize the learnable parameters using AdamW with a base learning rate of $1 \times 1 0 ^ { - 4 }$ and weight decay 0.05. The optimizer uses $\beta _ { 1 } = 0 . 9$ and $\beta _ { 2 } = 0 . 9 9 9$ . We apply a cosine learning rate schedule with linear warmup at the beginning of training. All experiments are conducted in mixed precision. We evaluate our method on two public benchmarks, IVEBench (Chen et al., 2026b) and VIEBench (Mou et al., 2025), which assess instruction-guided video editing from different perspectives. Unless otherwise specified, we follow the standard evaluation protocols of each benchmark and report their official metrics.

## 4.2 Main Results

Tables 2 and 3 report the main results on the short and long subsets of IVEBench, respectively. AVE achieves the best performance on most reported metrics on both subsets, showing that our two-stage design leads to a strong and well-balanced video editing capability rather than improving only a single aspect of the task. On the short subset, AVE improves the total score from the previous best 66.75 to 70.58, while also establishing new best results in video quality (82.10), instruction compliance (49.65), and semantic consistency (26.12). The same trend holds on the long subset, where AVE reaches a total score of 68.75, clearly outperforming the previous best result. These results suggest that AVE does not merely optimize perceptual quality or instruction following; instead, it achieves a favorable balance among edit faithfulness, temporal coherence, and source-video preservation.

A key strength of AVE lies in motion-related quality. On IVEBench, AVE obtains the best motion score on both the short and long subsets, reaching 89.26 and 88.95, respectively. We attribute this advantage to the design of Stage-2. Instead of asking the video model to jointly solve semantic editing and temporal reasoning from scratch, AVE treats video generation as an anchor-constrained interpolation problem. The edited keyframes explicitly define the desired content at several time steps, while the Wan-based video generator fills in the intermediate frames under the guidance of a motion representation extracted from the source video. This division of labor makes the temporal generation process much more stable, and naturally reduces flickering, drift, and motion inconsistency that commonly appear in naive per-frame editing and end-to-end video editing systems. In other words, AVE preserves motion well precisely because the video model is used for what it is best at: temporally coherent interpolation rather than full semantic editing.

<table><tr><td>Method</td><td>Add</td><td>Swap / Change</td><td>Remove</td><td>Style / Tone</td><td>Ins. Follow</td><td>Pres.</td><td>Quality</td></tr><tr><td>Kling1.6 (Kling Team et al., 2025)</td><td>6.602</td><td>8.800</td><td>8.253</td><td></td><td>7.813</td><td>8.697</td><td>7.143</td></tr><tr><td>Kling-Omni (Kling Team et al., 2025)</td><td>9.181</td><td>9.194</td><td>9.133</td><td>9.341</td><td>9.518</td><td>9.346</td><td>8.751</td></tr><tr><td>Runway (Runway, 2025)</td><td>8.447</td><td>9.161</td><td>8.504</td><td>9.133</td><td>9.109</td><td>8.972</td><td>8.354</td></tr><tr><td>Pika (Pika, 2026)</td><td></td><td>7.408</td><td></td><td></td><td>7.542</td><td>7.847</td><td>6.837</td></tr><tr><td>MiniMax (Zi et al., 2025)</td><td>=</td><td></td><td>6.839</td><td>1</td><td>6.963</td><td>7.518</td><td>6.037</td></tr><tr><td>DiffuEraser (Li et al., 2025)</td><td></td><td></td><td>6.243</td><td></td><td>6.346</td><td>6.807</td><td>5.576</td></tr><tr><td>InsV2V (Cheng et al., 2024)</td><td>4.281</td><td>5.567</td><td>2.098</td><td>7.452</td><td>4.475</td><td>6.043</td><td>4.033</td></tr><tr><td>VACE (Jiang et al., 2025)</td><td>4.854</td><td>6.640</td><td>2.682</td><td></td><td>3.974</td><td>6.042</td><td>4.162</td></tr><tr><td>Omni-Video (Tan et al., 2025)</td><td>6.242</td><td>4.748</td><td>5.593</td><td>5.366</td><td>5.481</td><td>5.404</td><td>5.429</td></tr><tr><td>UniVideo (Wei et al., 2026)</td><td>8.656</td><td>8.683</td><td>8.233</td><td>8.711</td><td>8.708</td><td>8.963</td><td>8.042</td></tr><tr><td>InstructX (Mou et al., 2025)</td><td>8.349</td><td>9.072</td><td>8.322</td><td>9.196</td><td>9.059</td><td>8.905</td><td>8.241</td></tr><tr><td>SAMA (Zhang et al., 2026a)</td><td>8.711</td><td>9.340</td><td>9.144</td><td>9.259</td><td>9.344</td><td>9.370</td><td>8.626</td></tr><tr><td>AVE (ours)</td><td>9.054</td><td>9.448</td><td>9.352</td><td>9.203</td><td>9.539</td><td>9.425</td><td>8.630</td></tr></table>

Table 4: Comparison results on VIE-Bench. We report the average score of each editing category (add, swap, remove, and style) and each evaluation metric (instruction following, motion preservation, and video quality). Results of closed-source commercial models are marked in gray.

Table 4 further shows that AVE generalizes well to VIE-Bench, which evaluates diverse editing types including add, swap/change, remove, and style/tone transfer. AVE is particularly strong on swap/change and remove, where it achieves the best average scores of 9.448 and 9.352, respectively. We believe these categories benefit the most from anchor-based generation, because the edited keyframes explicitly specify which object or region should be replaced or removed, and the video model only needs to propagate the edited result consistently through time. AVE also achieves the best averaged instruction-following score (9.539) and preservation score (9.425) across categories, indicating that it not only executes the edit faithfully, but also preserves the remaining content better than prior methods. This is exactly the behavior we would expect from our decomposition: Stage-1 handles the difficult semantic edit with a strong image editor, while Stage-2 focuses on preserving motion and completing the video around those edited anchors.

It is also worth noting that AVE remains competitive even against strong closed-source commercial systems. Although the best closed-source models still have a slight edge on a few appearance-centric categories such as add and style/tone, AVE already surpasses them on several important dimensions, including swap/change, remove, and the averaged instruction-following and preservation metrics. At the same time, AVE achieves a competitive overall quality score of 8.630, which is on par with or slightly better than the strongest open-source baselines. Taken together, these results show that AVE is not simply a lightweight approximation to end-to-end video editing. Rather, it provides a more effective formulation of the problem, enabling a video editor to directly inherit the strengths of modern image editing systems while retaining strong temporal coherence through a lightweight video generation stage.

Qualitative results. We provide qualitative examples in Fig. 3 to further illustrate the versatility of AVE across a wide range of editing scenarios. Our method performs well on diverse instruction types, including background removal, viewpoint change, object relocation, style transfer, local attribute editing, and explicit motion modification. A notable advantage of AVE is that, when the edit mainly changes appearance or scene content, the original motion and temporal structure are preserved very well. For example, in the first example, where the mountains are removed from the background, the rider’s pose and motion remain nearly unchanged across frames, while the new background stays consistent and coherent over time. In the zoom-out example, AVE expands the room around the father and daughter while maintaining stable identity, posture, and scene layout. The beach example further shows that AVE can handle object relocation reliably: the woman is moved to the right side of the frame, while the coastline and ocean remain temporally coherent.

The second row demonstrates that AVE is also effective on more challenging edits. In the watercolor example, the style is transformed consistently across frames while preserving the wave structure and overall scene composition, showing that AVE can handle global appearance changes rather than only local edits. In the tattoo removal example, the model performs a subtle local modification while keeping the surrounding body shape and background stable, indicating strong fine-grained control. Most importantly, the final example shows that AVE can support substantial motion changes: the couple is edited to start spinning while embracing, yet the generated motion remains smooth and stable instead of becoming abrupt or jittery. Together, these examples highlight the core strength of AVE: strong semantic edits are established at the keyframe level, and the motion-guided interpolation stage then propagates them through time in a controlled and coherent manner.

![](images/c4bfb57b6cb5012923cb88a0b3dab85f936cc35c14e1c303066c1d60563496bd.jpg)  
Figure 3: Instruction-based video editing results produced by AVE. Guided by keyframe-level instructions and image-to-video extension, our method delivers high-fidelity visual details, strong temporal coherence, and robust performance across diverse editing tasks.

## 4.3 Ablation Study

Unless otherwise specified, all ablations are conducted on the long subset of IVEBench. Each experiment is done by three runs and the standard deviation is less than 0.1%. We focus on two questions: (1) how much the final video editing performance depends on the image editor used in Stage-1, and (2) how sensitive AVE is to the keyframe selection strategy.

Effect of the Stage-1 image editor. Table 5 studies the impact of replacing the image editor in Stage-1 while keeping Stage-2 unchanged. By default, AVE uses Nano Banana 2, which yields the best overall results. Replacing it with other strong closed-source editors such as GPT-5.1 or Seedream 4.0 only causes a mild drop, while switching to our finetuned BAGEL leads to a further decrease. Using the original untuned BAGEL causes a much larger degradation. This trend provides a particularly important insight: the editing ability of AVE is strongly correlated with the quality of the image editor. Notably, the largest drops appear in instruction compliance, semantic consistency, and content fidelity, while motion remains relatively stable. This is consistent with our design: motion is mainly handled by Stage-2, which is fixed across all variants, whereas the semantic correctness of the edit is largely determined by the anchor frames produced by Stage-1. In other words, better image editing almost directly translates into better video editing. This strongly supports our main claim that the more difficult part of video editing is still the underlying image editing problem, while the image-to-video transfer itself is comparatively simple once high-quality anchors are available.

It is also worth noting that the tuned BAGEL model significantly outperforms the original untuned BAGEL. This suggests that multi-image consistent editing is important even before the video generation stage, and validates our choice of adapting open-source image editors with multi-frame video-derived supervision when closed-source APIs are unavailable.

<table><tr><td>Stage-1 Image Editor</td><td>Total</td><td>Quality</td><td>Ins. Comp.</td><td>Fidelity</td><td>Motion</td><td>Sem. Consist.</td><td>Content</td></tr><tr><td>Nano Banana 2 (default)</td><td>68.75</td><td>82.15</td><td>48.31</td><td>75.79</td><td>88.95</td><td>24.68</td><td>4.16</td></tr><tr><td>GPT-5.1</td><td>68.10</td><td>81.43</td><td>47.11</td><td>75.76</td><td>88.74</td><td>24.10</td><td>4.02</td></tr><tr><td>Seedream 4.0</td><td>67.59</td><td>80.36</td><td>46.67</td><td>75.74</td><td>88.61</td><td>23.83</td><td>3.85</td></tr><tr><td>BAGEL (finetuned)</td><td>66.07</td><td>79.38</td><td>43.06</td><td>75.78</td><td>88.10</td><td>21.96</td><td>3.83</td></tr><tr><td>BAGEL (base)</td><td>58.54</td><td>68.12</td><td>37.84</td><td>69.66</td><td>77.21</td><td>20.34</td><td>3.42</td></tr></table>

Table 5: Ablation on the Stage-1 image editor on IVEBench-long. We replace only the composed image editor in Stage-1 and keep the entire Stage-2 pipeline fixed. The results show a strong correlation between image editing quality and final video editing performance.

<table><tr><td>Strategy</td><td>#Keyframes</td><td>Total</td><td>Quality</td><td>Ins. Comp.</td><td>Fidelity</td><td>Motion</td></tr><tr><td>Uniform</td><td>2</td><td>66.85</td><td>79.58</td><td>46.02</td><td>74.96</td><td>87.42</td></tr><tr><td>Uniform</td><td>[T/48]</td><td>67.73</td><td>80.84</td><td>47.12</td><td>75.24</td><td>88.21</td></tr><tr><td>Uniform</td><td>[T/24]</td><td>68.21</td><td>81.69</td><td>47.86</td><td>75.09</td><td>88.71</td></tr><tr><td>Uniform</td><td>[T/12]</td><td>67.94</td><td>81.27</td><td>47.40</td><td>75.15</td><td>88.38</td></tr><tr><td>Scene-first + max-gap (ours)</td><td>2</td><td>67.28</td><td>79.93</td><td>46.48</td><td>75.44</td><td>87.67</td></tr><tr><td>Scene-first + max-gap (ours)</td><td>[T/48]</td><td>68.09</td><td>81.16</td><td>47.56</td><td>75.55</td><td>88.46</td></tr><tr><td>Scene-first + max-gap (ours)</td><td>[T/24]</td><td>68.75</td><td>82.15</td><td>48.31</td><td>75.79</td><td>88.95</td></tr><tr><td>Scene-first + max-gap (ours)</td><td>[T/12]</td><td>68.34</td><td>81.61</td><td>47.83</td><td>75.59</td><td>88.54</td></tr></table>

Table 6: Ablation on keyframe selection strategies and keyframe budgets on IVEBench-long. Our default strategy combines scene-aware anchors from PySceneDetect with largest-gap completion. AVE is robust to the exact keyframe selection rule, while a moderate keyframe budget around ⌊T /24⌋ gives the best trade-off.

Effect of keyframe selection. Table 6 compares different keyframe selection strategies and budgets. Our default strategy first extracts scene-aware anchors using PySceneDetect, keeps the first frame of each scene together with the final frame of the video, and then fills the remaining budget using a largest-gap rule. We compare this strategy against simple uniform sampling under different keyframe budgets. Two observations are clear. First, the gap between the two strategies is relatively small across all settings, indicating that AVE is fairly robust to the exact keyframe selection rule. This is encouraging, as it suggests that once a sparse set of reasonable anchors is provided, the Stage-2 interpolation model can reliably propagate the edits to the full video. Still, our scene-aware strategy is consistently better than uniform sampling, especially when the budget is limited, likely because it is more likely to place anchors near semantic transitions and scene changes.

Second, the best performance is achieved when the number of keyframes is set to approximately ⌊T/24⌋. Using too few keyframes leads to weaker instruction compliance and lower fidelity because the anchors become too sparse to fully specify the desired edit over long sequences. On the other hand, using too many keyframes also hurts performance slightly. We hypothesize that this is because the Wan2.2 3D VAE already performs a 4× temporal downsampling, so overly dense anchors reduce the effective interpolation space and make temporal completion less stable. As a result, AVE performs best when the anchors are sparse enough to allow interpolation, but dense enough to stabilize the desired edit.

## 5 Conclusion

In this paper, we propose Anchor-based Video Editing (AVE), a simple two-stage framework that transforms image editors into video editors. Our key insight is that the main difficulty of video editing still lies in image-level semantic editing, while temporal propagation can be handled much more efficiently once a sparse set of high-quality edited keyframes is available. Based on this observation, AVE first performs composed image editing on selected keyframes, and then uses a motion-guided image-to-video diffusion model to interpolate the full edited video. This design directly inherits the strengths of modern image editing systems while avoiding expensive end-to-end video editing training. Experiments on IVEBench and VIE-Bench show that AVE achieves strong performance across instruction following, semantic consistency, content fidelity, and motion preservation, and our ablations further reveal that final video editing quality is strongly correlated with the quality of the image editor. We hope this work offers useful insights into building future video editing systems in a more modular, efficient, and scalable way.

## References

Qingyan Bai, Qiuyu Wang, Hao Ouyang, Yue Yu, Hanlin Wang, Wen Wang, Ka Leong Cheng, Shuailei Ma, Yanhong Zeng, Zichen Liu, Yinghao Xu, Yujun Shen, and Qifeng Chen. Scaling instruction-based video editing with a high-quality synthetic dataset. In CVPR, 2026. 1, 3, 6

Black Forest Labs. Introducing FLUX.1 Kontext and the BFL playground. https://bfl.ai/blog/ flux-1-kontext, 2025. 1, 3

Tim Brooks, Aleksander Holynski, and Alexei A. Efros. InstructPix2Pix: Learning to follow image editing instructions. In CVPR, 2023. 1, 2

Haoxin Chen, Yong Zhang, Xiaodong Cun, Menghan Xia, Xintao Wang, Chao Weng, and Ying Shan. VideoCrafter2: Overcoming data limitations for high-quality video diffusion models. In CVPR, 2024. 3

Junyi Chen, Tong He, Zhoujie Fu, Pengfei Wan, Kun Gai, and Weicai Ye. VINO: A unified visual generator with interleaved OmniModal context. arXiv preprint arXiv:2601.02358, 2026a. 1

Yinan Chen, Jiangning Zhang, Teng Hu, Yuxiang Zeng, Zhucun Xue, Qingdong He, Chengjie Wang, Yong Liu, Xiaobin Hu, and Shuicheng Yan. IVEBench: Modern benchmark suite for instruction-guided video editing assessment. In ICLR, 2026b. 2, 6

Jiaxin Cheng, Tianjun Xiao, and Tong He. Consistent video-to-video transfer using synthetic dataset. In ICLR, 2024. 1, 3, 6, 7

DecartAI Team. Lucy Edit: Open-weight text-guided video editing. https://huggingface.co/decart-ai/ Lucy-Edit-Dev, 2025. 1, 6

Chaorui Deng, Deyao Zhu, Kunchang Li, Chenhui Gou, Feng Li, Zeyu Wang, Shu Zhong, Weihao Yu, Xiaonan Nie, Ziang Song, Guang Shi, and Haoqi Fan. Emerging properties in unified multimodal pretraining. arXiv preprint arXiv:2505.14683, 2025. 1, 3

Alisa Fortin and Bea Alessio. Build with Nano Banana 2, our best image generation and editing model. https://blog.google/innovation-and-ai/technology/developers-tools/ build-with-nano-banana-2/, 2026. 3, 4

Michal Geyer, Omer Bar-Tal, Shai Bagon, and Tali Dekel. TokenFlow: Consistent diffusion features for consistent video editing. In ICLR, 2024. 3

Google. Gemini 3. https://gemini.google.com/, 2025.

Yuwei Guo, Ceyuan Yang, Anyi Rao, Zhengyang Liang, Yaohui Wang, Yu Qiao, Maneesh Agrawala, Dahua Lin, and Bo Dai. AnimateDiff: Animate your personalized text-to-image diffusion models without specific tuning. In ICLR, 2024. 3

Haoyang He, Jie Wang, Jiangning Zhang, Zhucun Xue, Xingyuan Bu, Qiangpeng Yang, Shilei Wen, and Lei Xie. OpenVE-3M: A large-scale high-quality dataset for instruction-guided video editing. In ECCV, 2026. 1, 3

Yuzhou Huang, Liangbin Xie, Xintao Wang, Ziyang Yuan, Xiaodong Cun, Yixiao Ge, Jiantao Zhou, Chao Dong, Rui Huang, Ruimao Zhang, and Ying Shan. SmartEdit: Exploring complex instruction-based image editing with multimodal large language models. In CVPR, 2024. 2

Zeyinzi Jiang, Zhen Han, Chaojie Mao, Jingfeng Zhang, Yulin Pan, and Yu Liu. VACE: All-in-one video creation and editing. In ICCV, 2025. 1, 2, 3, 6, 7

Kling Team, Jialu Chen, Yuanzheng Ci, Xiangyu Du, Zipeng Feng, Kun Gai, Sainan Guo, Feng Han, Jingbin He, Kang He, et al. Kling-Omni technical report. arXiv preprint arXiv:2512.16776, 2025. 1, 7

Max Ku, Cong Wei, Weiming Ren, Harry Yang, and Wenhu Chen. AnyV2V: A tuning-free framework for any video-to-video editing tasks. Transactions on Machine Learning Research, 2024. 6

Xiaowen Li, Haolan Xue, Peiran Ren, and Liefeng Bo. DiffuEraser: A diffusion model for video inpainting. arXiv preprint arXiv:2501.10018, 2025. 1, 7

Xinyao Liao, Xianfang Zeng, Ziye Song, Zhoujie Fu, Gang Yu, and Guosheng Lin. In-context learning with unpaired clips for instruction-based video editing. arXiv preprint arXiv:2510.14648, 2025. 1, 3, 6

Yiqi Lin, Guoqiang Liang, Ziyun Zeng, Zechen Bai, Yanzhe Chen, and Mike Zheng Shou. Kiwi-Edit: Versatile video editing via instruction and reference guidance. arXiv preprint arXiv:2603.02175, 2026. 1, 3

Chang Liu, Rui Li, Kaidong Zhang, Yunwei Lan, and Dong Liu. StableV2V: Stabilizing shape consistency in video-to-video editing. IEEE Transactions on Circuits and Systemsfor Video Technology, 36(4):5467–5482, 2026. 6

Shiyu Liu, Yucheng Han, Peng Xing, Fukun Yin, Rui Wang, Wei Cheng, Jiaqi Liao, Yingming Wang, Honghao Fu, Chunrui Han, Guopeng Li, Yuang Peng, Quan Sun, Jingwei Wu, Yan Cai, Zheng Ge, Ranchen Ming, Lei Xia, Xianfang Zeng, Yibo Zhu, Binxing Jiao, Xiangyu Zhang, Gang Yu, and Daxin Jiang. Step1X-Edit: A practical framework for general image editing. arXiv preprint arXiv:2504.17761, 2025. 1, 3

Chong Mou, Qichao Sun, Yanze Wu, Pengze Zhang, Xinghui Li, Fulong Ye, Songtao Zhao, and Qian He. InstructX: Towards unified visual editing with MLLM guidance. arXiv preprint arXiv:2510.08485, 2025. 1, 2, 6, 7

OpenAI. Introducing GPT-5. https://openai.com/index/introducing-gpt-5/, 2025. 1

Pika. Pika. https://pika.art/, 2026. 1, 7

Chenyang Qi, Xiaodong Cun, Yong Zhang, Chenyang Lei, Xintao Wang, Ying Shan, and Qifeng Chen. FateZero: Fusing attentions for zero-shot text-based video editing. In ICCV, 2023. 3

Qwen Team. Qwen-Image-Edit: Image editing with higher quality and efficiency. https://qwen.ai/blog? id=qwen-image-edit, 2025. 1, 3

Zhefan Rao, Bin Zou, Haoxuan Che, Xuanhua He, Chong Hou Choi, Yanheng Li, Rui Liu, and Qifeng Chen. InsEdit: Towards instruction-based visual editing via data-efficient video diffusion models adaptation. arXiv preprint arXiv:2604.08646, 2026. 1, 3

Runway. Introducing Runway Gen-4. https://runwayml.com/research/introducing-runway-gen-4, 2025. 1, 7

Cuifeng Shen, Yulu Gan, Chen Chen, Xiongwei Zhu, Lele Cheng, Tingting Gao, and Jinzhi Wang. Decouple content and motion for conditional image-to-video generation. In AAAI, 2024. 3

Zhiyu Tan, Hao Yang, Luozheng Qin, Jia Gong, Mengping Yang, and Hao Li. Omni-Video: Democratizing unified video understanding and generation. arXiv preprint arXiv:2507.06119, 2025. 1, 6, 7

Team Seedream, Yunpeng Chen, Yu Gao, Lixue Gong, et al. Seedream 4.0: Toward next-generation multimodal image generation. arXiv preprint arXiv:2509.20427, 2025. 1

Wan Team, Ang Wang, Baole Ai, Bin Wen, Chaojie Mao, Chen-Wei Xie, Di Chen, Feiwu Yu, Haiming Zhao, Jianxiao Yang, et al. Wan: Open and advanced large-scale video generative models. arXiv preprint arXiv:2503.20314, 2025. 2, 4, 5

Qian Wang, Biao Zhang, Michael Birsak, and Peter Wonka. InstructEdit: Improving automatic masks for diffusion-based image editing with user instructions. arXiv preprint arXiv:2305.18047, 2023a. 2

Xiang Wang, Hangjie Yuan, Shiwei Zhang, Dayou Chen, Jiuniu Wang, Yingya Zhang, Yujun Shen, Deli Zhao, and Jingren Zhou. VideoComposer: Compositional video synthesis with motion controllability. In NeurIPS, 2023b. 3

Cong Wei, Quande Liu, Zixuan Ye, Qiulin Wang, Xintao Wang, Pengfei Wan, Kun Gai, and Wenhu Chen. UniVideo: Unified understanding, generation, and editing for videos. In ICLR, 2026. 1, 7

Jay Zhangjie Wu, Yixiao Ge, Xintao Wang, Weixian Lei, Yuchao Gu, Yufei Shi, Wynne Hsu, Ying Shan, Xiaohu Qie, and Mike Zheng Shou. Tune-A-Video: One-shot tuning of image diffusion models for text-to-video generation. In ICCV, 2023. 3

Yuhui Wu, Liyi Chen, Ruibin Li, Shihao Wang, Chenxi Xie, and Lei Zhang. InsViE-1M: Effective instructionbased video editing with elaborate dataset construction. In ICCV, 2025. 1, 3

Shitao Xiao, Yueze Wang, Junjie Zhou, Huaying Yuan, Xingrun Xing, Ruiran Yan, Chaofan Li, Shuting Wang, Tiejun Huang, and Zheng Liu. OmniGen: Unified image generation. In CVPR, 2025. 1, 3

Hangjie Yuan, Shiwei Zhang, Xiang Wang, Yujie Wei, Tao Feng, Yining Pan, Yingya Zhang, Ziwei Liu, Samuel Albanie, and Dong Ni. InstructVideo: Instructing video diffusion models with human feedback. In CVPR, 2024. 3

Lvmin Zhang, Anyi Rao, and Maneesh Agrawala. Adding conditional control to text-to-image diffusion models. In ICCV, 2023. 4, 13

Xinyao Zhang, Wenkai Dong, Yuxin Song, Bo Fang, Qi Zhang, Jing Wang, Fan Chen, Hui Zhang, Haocheng Feng, Yu Lu, et al. SAMA: Factorized semantic anchoring and motion alignment for instruction-guided video editing. arXiv preprint arXiv:2603.19228, 2026a. 1, 3, 7

Zhongwei Zhang, Fuchen Long, Wei Li, Zhaofan Qiu, Wu Liu, Ting Yao, and Tao Mei. In-context generation with regional constraints for instructional video editing. In ICML, 2026b. 1, 3

Bojia Zi, Weixuan Peng, Xianbiao Qi, Jianan Wang, Shihao Zhao, Rong Xiao, and Kam-Fai Wong. MiniMax-Remover: Taming bad noise helps video object removal. In NeurIPS, 2025. 1, 7

## Appendix

## Formal Details of The Motion Encoder

Let $U ^ { ( \ell - 1 ) }$ denote the input of the ℓ-th layer. We first perform spatial attention independently within each frame:

$$
\tilde { U } _ { t , : , : } ^ { ( \ell ) } = U _ { t , : , : } ^ { ( \ell - 1 ) } + \mathrm { A t t n } _ { \mathrm { s p a t i a l } } \Big ( \mathrm { L N } ( U _ { t , : , : } ^ { ( \ell - 1 ) } ) \Big ) ,\tag{9}
$$

where $\mathrm { A t t n } _ { \mathrm { s p a t i a l } }$ attends over all spatial locations $( h , w )$ at fixed time t. We then perform temporal attention independently at each spatial location:

$$
\bar { U } _ { : , h , w } ^ { ( \ell ) } = \tilde { U } _ { : , h , w } ^ { ( \ell ) } + \mathrm { A t t n } _ { \mathrm { t e m p o r a l } } \left( \mathrm { L N } ( \tilde { U } _ { : , h , w } ^ { ( \ell ) } ) \right) ,\tag{10}
$$

where $\mathrm { A t t n } _ { \mathrm { t e m p o r a l } }$ attends over all timesteps at fixed spatial position $( h , w )$ . Finally, the output of the layer is obtained with a feed-forward network:

$$
U ^ { ( \ell ) } = \bar { U } ^ { ( \ell ) } + \mathrm { M L P } \left( \mathrm { L N } ( \bar { U } ^ { ( \ell ) } ) \right) .\tag{11}
$$

After $L _ { m } = 6$ layers, the final motion representation is

$$
h _ { \mathrm { m o t } } = U ^ { ( L _ { m } ) } .\tag{12}
$$

We inject $h _ { \mathrm { m o t } }$ into the frozen Wan2.2 DiT through a ControlNet-style residual branch (Zhang et al., 2023), allowing the generator to preserve source motion while leaving appearance and editing semantics to be determined by the edited anchors. The overall framework of the motion encoder is shown in Figure 2.

## Training Stage-2

To train the lightweight motion branch, we keep the Wan 3D VAE and DiT frozen and optimize only $\mathcal { M } _ { \theta }$ and the ControlNet-style residual branch. Given a training target video $V ^ { \star }$ with latent code

$$
Z ^ { \star } = { \mathcal { V } } _ { \mathrm { 3 D } } ^ { \mathrm { e n c } } ( V ^ { \star } ) ,\tag{13}
$$

we sample a diffusion timestep τ and noise ϵ, and obtain a noisy latent video

$$
X _ { \tau } ^ { \star } = \alpha _ { \tau } Z ^ { \star } + \sigma _ { \tau } \epsilon .\tag{14}
$$

To mimic inference, the selected anchor frames are replaced by their clean latent values:

$$
\tilde { X } _ { \tau } ^ { \star } = M _ { A } \odot Z _ { A } ^ { \star } + ( 1 - M _ { A } ) \odot X _ { \tau } ^ { \star } .\tag{15}
$$

The model predicts the noise residual

$$
\hat { \epsilon } = D _ { \psi } ( \tilde { X } _ { \tau } ^ { \star } , h _ { \mathrm { m o t } } ) ,\tag{16}
$$

and we compute the training loss only on non-anchor frames:

$$
\mathcal { L } _ { \mathrm { v i d } } = \| ( 1 - M _ { A } ) \odot ( \hat { \epsilon } - \epsilon ) \| _ { 2 } ^ { 2 } .\tag{17}
$$

This objective explicitly teaches the model to interpolate and propagate content between fixed anchors, rather than re-synthesizing the anchors themselves.

## Broader Impacts

This work focuses on foundational research in video generation. We did not identify any specific direct societal impacts beyond the general considerations that accompany generative media.
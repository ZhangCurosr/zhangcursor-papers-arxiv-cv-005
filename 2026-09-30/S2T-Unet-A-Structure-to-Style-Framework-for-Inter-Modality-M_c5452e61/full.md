# S2T-Unet: A Structure-to-Style Framework for Inter-Modality MRI Translation

Yichao Liu<sup>1</sup>

Heidelberg University, IWR, Germany

Abstract. Inter-modality MRI translation aims to synthesize missing MRI modalities from available acquisitions, reducing the need for additional scanning while preserving clinically relevant anatomical information. However, existing image translation methods often learn intensity mappings without explicitly separating modality-invariant structural information from modality-specific appearance, which may lead to structural information loss or unrealistic image details. In this work, we propose S2T-Unet, a structure-to-style framework that explicitly models these two aspects. Specifically, vector quantization is introduced at the lower-level bottleneck to encode modality-invariant structural information using a learned discrete codebook. At higher levels, a modality transformation module uses decoder features to condition and transform encoder representations toward the target modality, thereby recovering modality-specific intensity and contrast information. Experiments on the IXI multi-contrast MRI dataset across four translation tasks demonstrate that S2T-Unet is comparable or outperform with state-of-art method.

Keywords: MRI · Image translation · Vector quantization.

## 1 Introduction

Magnetic Resonance Imaging (MRI) is a cornerstone of modern clinical diagnosis, ofering excellent soft-tissue contrast without ionizing radiation [5]. A comprehensive MRI protocol typically acquires multiple complementary sequences, such as T1-weighted, T2-weighted, and PD, each capturing distinct tissue characteristics that, taken together, support accurate lesion characterization, segmentation, and longitudinal follow-up [10, 1]. In practice, however, obtaining a complete set of sequences is often infeasible: long acquisition times increase patient discomfort and susceptibility to motion artifacts, contrast-enhanced scans raise safety and cost concerns, and retrospective or multi-site datasets frequently sufer from missing or corrupted modalities. These limitations have motivated growing interest in MRI image translation, which aims to synthesize a target modality from one or more available source modalities, thereby completing missing information without additional scanning.

Deep generative models have been central to progress in this area. Early efforts were dominated by generative adversarial networks (GANs), whose adversarial training enabled sharp, visually realistic synthesis and established strong baselines for both paired and unpaired translation. As the representative method for paired data, conditional models such as pix2pix [6] use a source-conditioned generator to learn a direct mapping from source to target images. For unpaired settings, CycleGAN [14] introduces a cycle-consistency constraint to align the two modalities without pixel-wise correspondence. These methods established strong baselines and remain widely used points of comparison for medical image synthesis.

However, GAN-based approaches are constrained by several well-known limitations. The adversarial minmax objective is dificult to optimize, resulting in unstable training and mode collapse [4]; more critically in a clinical setting, GANs are prone to hallucinating anatomically plausible yet non-existent structures that are not supported by the source image [13]. Such fabricated details pose a serious risk when synthesized images inform downstream diagnosis or treatment planning.

These limitations motivate a reconsideration not only of the training objective but also of the backbone architecture on which such generators are built. Convolutional networks operate over a fundamentally local receptive field and therefore struggle to enforce consistency across spatially distant regions of an image. The Transformer architecture ofers a complementary inductive bias: through the self-attention mechanism, every spatial location interacts directly with every other, enabling the model to capture long-range dependencies and global anatomical context [12, 3]. Motivated by this property, a growing body of work has adapted Transformers to medical image translation and reconstruction [2], where preserving globally coherent anatomy across the entire field of view is essential.

However, existing image translation methods primarily focus on learning the mapping between image intensities, while the underlying structural and semantic information of the image is often not explicitly modeled. This limitation is particularly important in medical image translation. For example, in inter-modality MRI translation, diferent MRI modalities exhibit substantially diferent intensity distributions, while sharing largely consistent anatomical structures. Therefore, successful translation requires not only preserving the modality-invariant structural and semantic information, but also efectively modeling and transferring the modality-specific appearance, such as intensity and contrast. Specifically, in hierarchical image reconstruction models, such as Unet [9], deeper layers tend to capture more abstract anatomical information, whereas upper layers progressively conduct style or intensity transformation. Recent studies have explored vector quantization with learned codebooks to represent images using discrete latent representations, which have shown promising results in capturing meaningful semantic information for image generation [11, 15], while conditional feature modulation provides an efective mechanism for controlling image-specific style and appearance [8, 7]. These complementary advances motivate us to explicitly model the two aspects of image translation at diferent hierarchical levels: a discrete bottleneck for preserving modality-invariant structural content, and higher-level representations for capturing and transferring modality-specific appearance.

In this paper, we propose a novel framework, S2T-Unet, for inter-modality MRI translation by rethinking how diferent levels of the Unet representation can be utilized. Specifically, we introduce vector quantization at the bottleneck of the encoder to preserve modality-invariant structural and semantic information, while using the decoded representation as a condition to guide modality-specific style transformation toward the target MRI modality. Our framework builds upon the existing Unet architecture and requires no additional hyperparameter tuning beyond that of the original vector quantization module. Extensive experiments on a publicly available multi-contrast MRI dataset demonstrate that our approach consistently outperforms state-of-the-art methods.

## 2 Methods

![](images/0b9e78a270206be4d76ed6dba004ea65873ebd584da93f577c2f55a353828d39.jpg)  
Fig. 1: Overview of S2T-Unet. It comprises three components: a Unet encoderdecoder, style transformation modules, and vector quantization modules. Dashed arrows denote features extracted before batch normalization and ReLU function, which are fed into the style transformation module. IN denotes that the normalization in the Conv block is instance normalization.

The S2T-Unet architecture of our method is shown in Fig. 1. It consists of style transformation modules, vector quantization modules, and a Unet encoderdecoder structure. We assume in the MRI image translation tasks, the Unet

decoder is responsible for the image translation or style translation, while encoder is responsible for information extraction, such as semantic information extraction.

## 2.1 Vector Quantization in the Low-Level Unet

The key idea behind vector quantization is to represent an image using a finite set of discrete codes drawn from a learned codebook, a dictionary in which each entry captures a recurring structural pattern of the image. Instead of encoding an image as continuous, unconstrained features, a vector quantization model maps each local feature vector to its nearest entry in the codebook, yielding a compact discrete representation. Because the codebook is shared and its entries are repeatedly reused across images, this representation is encouraged to capture the essential anatomical structure rather than modality-specific appearance or noise.

This property is particularly well suited to intra-modality MRI translation. When a T1-weighted image is translated to T2, the underlying anatomy must remain unchanged. Since the skip connection of the lower-level U-Net carries the structural content shared between the source and target contrasts, we quantize it (Fig. 1, bottom right) so that these structural features are passed to the decoder in a discrete form and reused during synthesis. By anchoring synthesis to a fixed vocabulary of learned structures, vector quantization constrains the model from fabricating anatomically implausible details, directly mitigating the hallucination problem of GAN-based approaches.

Formally, given an encoder feature map $z _ { e } ( x ) \in \mathcal { R } ^ { h \times w \times C }$ , where C is channel numbers, and a codebook $E = \{ e _ { k } \} _ { k = 1 } ^ { K }$ , where k is the code numbers. Vector quantization replaces each spatial feature of MRI by its nearest entry, $z _ { q } ( x ) _ { i , j } =$ $e _ { k }$ with k = arg min $\boldsymbol { \mathbf { \ell } } _ { m } | | z _ { e } ( \boldsymbol { x } ) _ { i , j } - e _ { m } | | _ { 2 }$ . Then the quantized latent space will feed into decoder and be concatenated with decoder features.

## 2.2 Modality Transformation

While vector quantization preserves the anatomical structure of the MRI image, its discrete nature inevitably discards continuous appearance information such as texture and contrast, which is precisely the modality-specific content that a translation model must alter. Moreover, the Unet decoder reconstructs the transformed image through direct feature concatenation, which may oversimplify the interaction between transformed representations and low-level source features, potentially limiting the exploitation of causal information. To recover this information while performing the translation, we draw on the idea of style transformation: the decoder features, which encode the target-modality appearance, serve as a conditioning signal, while the quantized skip-connection features from the encoder are treated as the content whose modality is to be transformed. Concretely, we design a Modality Transformation module at upper layers based on Feature-wise Linear Modulation (FiLM) [8], which modulates the encoder features conditioned on the decoder features, thereby transferring the modality and compensating for the information loss introduced by vector quantization at lower level layers.

In practice, the feature $\mathbf { x } ^ { i }$ denotes activation of i-th layer in decoder, and the height and width are $H ^ { i }$ and $W ^ { i }$ , respectively. The feature from encoder is denoted by $\mathbf { x } _ { e } ,$ , and the height and width are $2 H ^ { i }$ and $2 W ^ { i }$ , respectively. The upper-right part of Fig. 1 illustrates the design. It can be formulated as below:

$$
\begin{array} { r } { \mathbf { x } _ { o u t } ^ { i } = ( 1 + \gamma ^ { i } ) \mathbf { x } _ { e } ^ { i } + \beta ^ { i } } \\ { \gamma ^ { i } = f _ { \gamma } ( R e L U ( f _ { c } ( f _ { r } ( \mathbf { x } ^ { i } ) ) ) ) } \\ { \beta ^ { i } = f _ { \beta } ( R e L U ( f _ { c } ( f _ { r } ( \mathbf { x } ^ { i } ) ) ) ) } \end{array}\tag{1}
$$

where, $\gamma ^ { i }$ denotes scaling feature and $\beta$ denotes bias feature. $f _ { r }$ denotes the reshape function. $f _ { c } , f _ { \gamma }$ i and $f _ { \beta ^ { i } }$ are $3 \times 3$ conv layer.

## 2.3 Objective Function

To train the network, we use L1 norm to calculate the distance between the output and target images, $f ( x )$ and $y ,$ respectively. The function is shown as:

$$
L _ { 1 } = E _ { x , y } | | y - f ( x ) | | _ { 1 }\tag{2}
$$

To update the codebook in vector quantization module, we have applied vector quantization loss and commitment loss from this study [11]. Thus, the whole loss function is:

$$
L = L _ { 1 } + | | s g [ z _ { e } ( x ) ] - e _ { m } | | _ { 2 } ^ { 2 } + \alpha | | z _ { e } ( x ) - s g [ e _ { m } ] | | _ { 2 } ^ { 2 }\tag{3}
$$

where, $s g$ is a stopgradient operator. α is a coeficient to balance vector quantization loss, second term and commitment loss, third term. We use $\alpha = 0 . 2 5$ here.

## 3 Experimental Results

We have evaluated our proposed network S2T-Unet on the IXI dataset <sup>1</sup>. The IXI dataset has 581 subjects. Each one includes T1-, T2- and PD-weighted MRI image. The voxel size of the MRI images is $( 0 . 9 4 \times 0 . 9 4 \times 1 . 2 ~ \mathrm { m m ^ { 3 } } )$ . We used 115 subjects for training and testing separately. The images are preprocessed by ants Python package, and HD-BET is used for skull stripping. We have conducted 4 one-to-one image translation experiments: T1 → T2, T2 → T1, T1 → PD, PD $ \mathrm { T 1 }$ . We use a batch size 32, learning rate of $3 e - 4$ for training. The models are trained with 20 epochs. Three metrics are used for the evaluation: Peak Signalto-Noise Ratio (PSNR), Structural Similarity Index Measure (SSIM), and Root Mean Squared Error (RMSE).

We compare our method with popular CNN-based and transformer-based methods: Unet [9], CycleGAN [14], Pix2Pix [6], and ResViT [2]. The quantitative performance is shown in Table 1. Our method achieves competitive or superior performance across all four translation tasks compared with both CNN-based and Transformer-based methods. In particular, it consistently outperforms the Transformer-based ResViT in terms of PSNR and RMSE, while achieving comparable or slightly improved SSIM scores. Notably, these results are obtained using a purely CNN-based architecture built upon the Unet framework, without introducing adversarial loss. Compared with Unet, our base model, the proposed approach substantially improves the translation performance across all four tasks. This improvement suggests that the encoder can efectively preserve and pass semantic and structural information to the decoder, thereby reducing structural information loss during reconstruction. Moreover, the proposed approach consistently achieves higher PSNR than the Transformer-based ResViT across all four translation tasks. This performance advantage suggests that the proposed modality transformation module efectively leverages decoder features to modulate encoder features, facilitating the synthesis of target-modality intensity and contrast.

Table 1: Quantitative performance comparison between CNN and Transformerbased architectures for one-to-one image-to-image translation using the IXI dataset.
<table><tr><td rowspan="2">Model</td><td>T1→T2</td><td>T2→T1</td><td>T1→PD</td><td>PD→T1</td></tr><tr><td>PSNR SSIM RMSE</td><td>PSNR SSIM RMSE</td><td>PSNR SSIM RMSE</td><td>PSNR SSIM RMSE</td></tr><tr><td>UNet</td><td>24.95 0.885 0.0576</td><td>21.81 0.862 0.0846</td><td>22.63 0.815 0.0752</td><td>21.63 0.813 0.085</td></tr><tr><td>Pix2Pix</td><td>26.500.899 0.0477</td><td>23.38 0.875 0.0693</td><td>25.02 0.891 0.0572</td><td>23.30 0.874 0.07</td></tr><tr><td>CycleGAN</td><td>25.440.8890.0522</td><td>22.69 0.869 0.075</td><td>23.45 0.858 0.066</td><td>22.51 0.847 0.076</td></tr><tr><td>ResViT</td><td>27.370.917 0.0432</td><td>23.72 0.887 0.0681</td><td>25.11 0.899 0.0569</td><td>23.28 0.886 0.0719</td></tr><tr><td>Ours</td><td>27.480.9160.0428</td><td>24.10 0.8930.0657</td><td>25.340.900 0.0554</td><td>23.78 0.892 0.0677</td></tr></table>

We visualize our synthesized MRI modality image in Fig. 2. Overall, the proposed method produces synthesized images with appearance characteristics that are visually closer to the target modality while maintaining the anatomical structures of the source image. Notably, the diference maps reveal a clear contrast in error structure between our method and the competing baselines. The residuals produced by our method are spatially coherent and concentrated at genuine anatomical boundaries, whereas the other methods exhibit noisy, disorganized diference maps in which positive and negative errors (red and blue) are densely intermixed throughout the image. We attribute this to the efect of adversarial training: adversarial losses encourage the generator to hallucinate realistic-looking high-frequency texture, but such texture is inherently stochastic and rarely aligns pixel-wise with the ground truth, resulting in scattered, high-frequency error patterns. In contrast, our deterministic formulation, trained without any adversarial objective, avoids fabricating such texture and instead yields errors that are spatially consistent and largely confined to regions of genuine structural mismatch.

![](images/73d7e8028f3c4a4efc808e9f6f3a70bd3766525c248067df36e24d27f7cce5e9.jpg)

![](images/feecc6019292e4741f515f8cb4e24036e1a0fb3c5ebb29d8245a441511a51df8.jpg)  
Fig. 2: Qualitative comparison of MRI cross-modality translation in four directions: (a) T1 → T2, (b) T2 → T1, (c) T1 → PD, and (d) PD → T1. In each panel, the input (source modality) and ground truth (GT, target modality) are shown on the left, and the synthesized results of Pix2Pix, ResViT, and our method are shown on the right. The bottom row of each panel shows the corresponding diference maps between the synthesized image and the GT, where red and blue indicate positive and negative errors, respectively.

<table><tr><td rowspan=2 colspan=1></td><td rowspan=1 colspan=1>without VQ</td><td rowspan=1 colspan=1>without MT</td><td rowspan=1 colspan=1>with both</td></tr><tr><td rowspan=1 colspan=1>PSNR SSIM RMSE</td><td rowspan=1 colspan=1>PSNR SSIM RMSE</td><td rowspan=1 colspan=1>PSNR SSIM RMSE</td></tr><tr><td rowspan=1 colspan=1>S2T-Unet</td><td rowspan=1 colspan=1>26.100.900 0.0510</td><td rowspan=1 colspan=1>25.200.888 0.0559</td><td rowspan=1 colspan=1>27.48 0.916 0.0428</td></tr></table>

Table 2: Ablation study for vector quantization (VQ) and modality transformation (MT).

Table 2 presents the ablation study of the vector quantization and modality transformation modules. Removing either vector quantization or modality transformation leads to a noticeable degradation in performance, confirming that both components contribute to the overall efectiveness of S2T-Unet. In particular, removing modality transformation results in a larger performance drop, with the PSNR decreasing from 27.48 dB to 25.20 dB, compared with a decrease to 26.10 dB when vector quantization is removed. This suggests that the modality transformation module plays a more direct role in the reconstruction process, as it explicitly modulates the encoder features using decoder features to adapt the representation to the target modality. In contrast, vector quantization is mainly introduced to preserve modality-invariant structural and semantic information at the bottleneck, and its contribution may therefore be less directly reflected in the final reconstruction. Nevertheless, the removal of vector quantization still causes a substantial performance degradation, indicating that structural information preserved by vector quantization is important for achieving the best translation performance when combined with modality transformation.

## 4 Conclusion

We proposed S2T-Unet, an MRI inter-modality image translation framework. The model learns semantic information, i.e., the modality-invariant feature, from a codebook, thereby avoiding changes to the underlying image structure. Vector quantization at the lower-level bottleneck anchors synthesis to a discrete vocabulary of structural codes, preserving modality-invariant anatomy and mitigating the hallucination of anatomically implausible details, while a modality transformation module at the upper levels uses decoder features to condition and transfer the target-modality appearance onto encoder representations. Experiments on the IXI dataset across four translation tasks show that our purely convolutional, deterministic formulation, requiring neither adversarial training nor Transformer-based self-attention, matches or surpasses popular GAN-based and Transformer-based baselines, indicating that explicit structure-appearance separation ofers an efective and stable alternative to adversarial or attentionbased designs. Future work includes extending the framework to 3D volumetric translation and evaluating generalization on additional multi-contrast and multimodal datasets.

Acknowledgments. We are grateful for access to the University of Heidelbergs IWR HPC service for running simulations used in this study.

## References

1. Alex, V., Vaidhya, K., Thirunavukkarasu, S., Kesavadas, C., Krishnamurthi, G.: Semisupervised learning using denoising autoencoders for brain lesion detection and segmentation. Journal of Medical Imaging 4(4), 041311–041311 (2017)

2. Dalmaz, O., Yurt, M., Çukur, T.: Resvit: Residual vision transformers for multimodal medical image synthesis. IEEE Transactions on Medical Imaging 41(10), 2598–2614 (2022)

3. Dosovitskiy, A., Beyer, L., Kolesnikov, A., Weissenborn, D., Zhai, X., Unterthiner, T., Dehghani, M., Minderer, M., Heigold, G., Gelly, S., et al.: An image is worth 16x16 words: Transformers for image recognition at scale. arXiv preprint arXiv:2010.11929 (2020)

4. Goodfellow, I., Pouget-Abadie, J., Mirza, M., Xu, B., Warde-Farley, D., Ozair, S., Courville, A., Bengio, Y.: Generative adversarial networks. Communications of the ACM 63(11), 139–144 (2020)

5. Hussain, S., Mubeen, I., Ullah, N., Shah, S.S.U.D., Khan, B.A., Zahoor, M., Ullah, R., Khan, F.A., Sultan, M.A.: Modern diagnostic imaging technique applications and risk factors in the medical field: a review. BioMed research international 2022(1), 5164970 (2022)

6. Isola, P., Zhu, J.Y., Zhou, T., Efros, A.A.: Image-to-image translation with conditional adversarial networks. In: 2017 IEEE conference on computer vision and pattern recognition (CVPR). pp. 5967–5976. Ieee (2017)

7. Park, T., Liu, M.Y., Wang, T.C., Zhu, J.Y.: Semantic image synthesis with spatially-adaptive normalization. In: 2019 IEEE/CVF conference on computer vision and pattern recognition (CVPR). pp. 2332–2341. IEEE (2019)

8. Perez, E., Strub, F., De Vries, H., Dumoulin, V., Courville, A.: Film: Visual reasoning with a general conditioning layer. In: Proceedings of the AAAI conference on artificial intelligence. vol. 32 (2018)

9. Ronneberger, O., Fischer, P., Brox, T.: U-net: Convolutional networks for biomedical image segmentation. In: International Conference on Medical image computing and computer-assisted intervention. pp. 234–241. Springer (2015)

10. Tae, W.S., Ham, B.J., Pyun, S.B., Kim, B.J.: Current clinical applications of structural mri in neurological disorders. Journal of clinical neurology (Seoul, Korea) 21(4), 277 (2025)

11. Van Den Oord, A., Vinyals, O., et al.: Neural discrete representation learning. Advances in neural information processing systems 30 (2017)

12. Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A.N., Kaiser, Ł., Polosukhin, I.: Attention is all you need. Advances in neural information processing systems 30 (2017)

13. Wang, S., Zhou, X., Li, C., Wang, S., Li, Y., Tan, T., Zheng, H.: Generative artificial intelligence in medical imaging: foundations, progress, and clinical translation. Research 8, 1029 (2025)

14. Zhu, J.Y., Park, T., Isola, P., Efros, A.A.: Unpaired image-to-image translation using cycle-consistent adversarial networks. In: 2017 IEEE international conference on computer vision (ICCV). pp. 2242–2251. Ieee (2017)

15. Zhu, Y., Li, B., Xin, Y., Xia, Z., Xu, L.: Addressing representation collapse in vector quantized models with one linear layer. In: 2025 IEEE/CVF International Conference on Computer Vision (ICCV). pp. 22968–22977. IEEE (2025)
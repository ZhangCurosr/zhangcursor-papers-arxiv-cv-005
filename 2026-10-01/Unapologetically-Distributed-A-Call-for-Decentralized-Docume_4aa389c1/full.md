# Unapologetically Distributed: A Call for Decentralized Document Analysis

Adrià Molina<sup>1,2</sup> amolina@cvc.uab.cat Oriol Ramos Terrades<sup>1,2</sup> <sup>6</sup>oriolrt@cvc.uab.cat 0Josep Lladós<sup>1,2</sup> 2josep@cvc.uab.cat

<sup>1</sup> Centre de Visió per Computador Universitat Autònoma de Barcelona Bellaterra, Catalonia

<sup>2</sup> Computer Science Department Universitat Autònoma de Barcelona, Bellaterra, Catalonia

## Abstract

Privacy has become an increasingly important concern in the Document Analysis community, to the extent that in many environments such as archives, governmental institutions, and local businesses, the adoption of automation is restricted by legal and policy constraints. While federated learning has often been regarded as a “necessary evil”, implying an unavoidable performance trade-off in exchange for decentralization and privacy, many prior works overlook its potential to improve robustness to out-of-distribution data. In this paper, we present Unapologetically Distributed, the first comprehensive study evaluating distributed learning in Document Analysis along three key axes simultaneously: the tasks addressed, the architectures employed, and the fine-tuning strategies applied. Specifically, we demonstrate how various distributed training approaches enhance generalization capabilities across diverse tasks such as Table Recognition, handwriting recognition, and Word Spotting, particularly during transfer learning stages. Our results provide strong evidence that decentralization is not merely a constraint, but a valuable opportunity to improve model robustness and adaptability in real-world Document Analysis scenarios.

## Introduction

iIn many Document Analysis scenarios, the assumptions underlying centralized machine X<sub>learning pipelines are difficult to satisfy. Training modern deep models typically relies on</sub> <sup>r</sup>large-scale data aggregation and expensive computational infrastructures, a strategy that has raised growing concerns regarding data privacy and information leakage [6]. These issues are particularly critical for document data, which frequently contains sensitive, personal, legal, or historical information. As a result, centralization is often impractical or undesirable, despite the availability of valuable training material. This limitation is especially evident in contexts involving privately owned administrative records. Additionally, cultural heritage institutions have long emphasized the importance of preserving documents close to their place of origin, as this practice safeguards their contextual integrity [5], authenticity, and cultural significance [19], while also aligning with distributed preservation strategies [30].

![](images/594398e29f2d46fad84dbd8273e3046baa694ddc9f7c6d6b8a3301b8121b4d9c.jpg)  
(a) Example of a distributed learning set-up

![](images/16dbfdd7b35f11eb9867d1c3ae81a512a89db0ae4be7413bd8907acbe3ca7865.jpg)  
(b) Word spotting results comparison.  
Figure 1: We show that distributed regimes (orange area, proposed) consistently outperform centralized pretraining (blue area) across a wide range of architectures and downstream tasks. We perform an extensive evaluation on 27 well-established Word Spotting and recognition datasets.

Similarly, small and medium-sized enterprises routinely handle confidential documents but lack the resources or legal capacity to externalize data labeling or participate in centralized training schemes. In all cases, the inability to pool data significantly limits the applicability of conventional learning approaches.

Distributed learning provides a natural alternative by enabling collaborative model training without direct data sharing, more precisely in our case through post-hoc model merging, where independently trained models are aggregated at the parameter level. However, their adoption in Document Analysis has been hindered by concerns regarding robustness under non-identically distributed data, a setting in which performance degradation has been widely reported [24]. At the same time, insights from personalized and meta-learning suggest that distributed training can remain effective when local heterogeneity is explicitly accounted for [2]. This work builds on these observations and investigates distributed learning as a practical and effective solution for Document Analysis in privacy-constrained and low-resource environments.

Our research hypothesis is threefold: (i) following model merging principles, features learned by distributed models should be more generalizable and thus serve as better teachers in knowledge distillation; (ii) these features should be sufficiently general to enable multiscript learning, even when data distributions differ significantly from the original training data; and (iii) the resulting models should provide superior initializations for subsequent in-distribution finetuning stages.

The contribution of this work is to present a large-scale empirical and analytical study of decentralized training in Document Analysis, examining when and why simple model merging strategies yield consistent benefits (see Figure 1). Our analysis is structured along three complementary axes: architectural families, task types, and data regimes. We evaluate distributed pre-training across a diverse set of representative tasks which, when considered jointly, cover a broad and practically relevant spectrum of use cases in Document Analysis. By controlling computational budgets, we isolate the effect of decentralized training and characterize the conditions under which it provides measurable advantages. Our objective is not to advocate decentralization as a universal solution, but to characterize the regimes in which it is most effective. In particular, we identify the conditions under which decentralized learning delivers competitive or superior performance; most notably in low-resource, distribution-shifted, and operationally constrained scenarios that frequently arise in realworld Document Analysis applications. To this end, we conduct an extensive evaluation comprising:

• Cross-modal knowledge distillation, where features learned in a distributed manner serve as teachers for Word-Spotting students, allowing models to be trained from scratch.

• Multi-script learning for new and unseen alphabets in Handwritten Text Recognition tasks, implemented via layer-wise finetuning in a personalized manner.

• End-to-end, finetuning and zero-shot evaluations of Table Recognition models initialized through distributed training of Graph Neural Networks.

To the best of our knowledge, this article represents the first comprehensive evaluation of distributed learning in Document Analysis that simultaneously considers multiple dimensions: the specific tasks being addressed, the underlying neural architectures employed, and the strategies used for fine-tuning. By systematically examining these aspects, we provide a holistic understanding of how distributedly learned models perform across diverse Document Analysis scenarios and highlight the conditions under which distributed learning provides tangible benefits over centralized approaches.

## 2 Related Work

Existing studies on distributed <sup>1</sup> document understanding predominantly frame distributed learning as a proxy for centralized performance, rather than as a first-class paradigm capable of reshaping how sensitive tasks on structured documents are learned through multiple domains.

In [38], a representative example of distributed OCR, the authors note that “Expectantly, our FedOCR achieves comparable results, which are very close to the results ofthe centralized training manner,” a formulation that frames federated learning as an approximation of centralized training rather than a standalone paradigm. In their seminal contributions such as [12], where distributed learning is applied to Graph Neural Network architectures, the conclusions are drawn in the same direction. In recent years, some works have started to point out that distributedly learned systems may hold significant potential for subsequent fine-tuning strategies. This is the case in [29], a pilot study on federated Document Visual Question Answering, where the authors report the outperformance of the centralized regime (C=1, K=1 in their setting, where C denotes the client sampling probability and K the number of participating clients) while also observing that their federated approach achieves comparable results with smaller models in later fine-tuning stages. Nevertheless, the broader implications and global potential of federated learning are not explicitly emphasized. To the best of our knowledge, no prior study has addressed the task of federated learning for Table Recognition, a notable gap given the sensitivity of such data in modern industrial settings.

In summary, this work is contextualized as the first to holistically demonstrate the potential of distributed learning in Document Analysis across multiple tasks and architectures, while also being the first to perform distributed learning for Table Recognition and knowledge distillation from federatedly learned Document Analysis Systems. In this paper, readers may find similarities to Model-Agnostic Meta-Learning [17], Model Soups [36], Model Editing with Task Arithmetic [15], and Federated Learning through FedAvg [25], depending on their background. We argue that all of the methods above can be treated similarly with regard to their capacity for distributed learning, in the sense that models trained across different computing units can be aggregated without sharing data.

In contrast to FedAvg, we do not conduct multiple federated rounds. In contrast to MAML, our dataset splits are domain-specific, emulating different institutions. Perhaps the closest related approaches are model editing and model soups; however, neither focuses on the ability of distributedly learned models to incorporate personalized, out-of-domain data. Furthermore, to the best of our knowledge, no prior model soup work has conducted such an extensive evaluation across different architectures, datasets, and application domains.

We note again that all of the aforementioned algorithms share the common principle described in Eq. 3, but implemented within different optimization loops and with emphasis on different aspects of the learning outcomes.

## 3 Methodology

In this study, we aim to evaluate distributed learning holistically across Document Analysis tasks. Specifically, we conduct experiments under three training strategies: knowledge distillation, layer-wise fine-tuning, and end-to-end fine-tuning. This design allows us to assess whether distributedly learned models exhibit higher generality than their centralized counterparts across different Document Analysis scenarios. To support this claim, it is first necessary to clearly define the three methodological settings considered in this work.

Given the broad audience targeted by this study, we begin by introducing the fundamentals of our model merging strategy. Following the notation introduced in [15], we present the principles of model merging in Section 3.1. Readers already familiar with the FedAvg algorithm and the concept of task vectors may directly refer to Section 3.2, where we detail the specific methodological choices of this study, namely the three fine-tuning strategies evaluated.

## 3.1 Model Merging Strategy

We consider a model as a vector of weights, $\pmb { \theta } = \left( w _ { 1 } , w _ { 2 } , . . . , w _ { m } \right)$ such that $\theta \in \mathbb { R } ^ { m }$ . Given a model with initial parameters $\theta ^ { 0 }$ , we train it on a dataset D composed of multiple small sub-sets $d _ { i }$ :

$$
D = \{ d _ { 1 } , d _ { 2 } , \ldots , d _ { n } \} ,\tag{1}
$$

which converges to a model $\theta ^ { D }$ . This is what we denote as a centralized model, i.e., a set of parameters that have been trained on the dataset $D$ as a whole.

Analogously, a sub-domain $d _ { i } \in D$ optimizes a model that converges to $\theta ^ { d _ { i } }$ . By optimizing on each sub-dataset independently, we obtain different models:

$$
\{ \theta ^ { d _ { 1 } } , \theta ^ { d _ { 2 } } , \ldots , \theta ^ { d _ { n } } \} .\tag{2}
$$

The parameters of these models (which share a common architecture) can then be averaged as follows:

$$
\overline { { \theta _ { D } } } = \frac { 1 } { n } \sum _ { i = 1 } ^ { n } \theta ^ { d _ { i } } .\tag{3}
$$

This process is often referred to as task arithmetic [15], and it also constitutes the core procedure of the FedAvg algorithm [25]. We refer to $\overline { { \theta _ { D } } }$ as a distributed model, as it enables each dataset to remain on different computing units while only sharing model weights, but not the underlying data.

## 3.2 Finetuning Strategy

In this section, three methodological approaches are proposed to verify the hypothesis that distributedly trained models exhibit superior learning capabilities during the fine-tuning stage. Specifically, we investigate: (i) knowledge distillation from a frozen encoder in a Word Spotting task; (ii) layer-wise fine-tuning strategies for handwritten text recognition (HTR) and optical character recognition (OCR) problems when adapting models to new alphabets; and (iii) graph-based Table Recognition using end-to-end fine-tuned models. Through these approaches, we aim to present a broad and transversal methodology that encompasses a wide range of use cases, architectures, and training strategies, thereby providing a robust and empirical validation of the proposed hypothesis.

Knowledge Distillation in Word Spotting Knowledge distillation is the training strategy consisting in transfering information from a base network (teacher), which is usually trained with high computational and data resources, to a smaller (student) network. For this task, we aim to leverage the information of a frozen vision encoder into a small text encoder to perform a Query-by-String evaluation of a Word Spotting system. According to our hypothesis, distributed models should behave better as teachers due to its higher generalization capability.

As seen in Figure 2, different off-the-shelf OCR models from [27] are sampled as Vision Encoders dedicated to both scene and handwritten text recognition. After removing the language layer, we perform model merging (see Equation 3) to obtain a distributedly learnt feature extractor (i.e. an extractor for which the different training datasets are not shared). Then, given a target dataset (the one we want to perform distillation on), visual embeddings are computed. For the textual embeddings, we make use of standard bidirectional Recurrent Neural Networks (LSTM [14] and GRU [7]) and a transformer encoder architecture [3]. In all the cases, the text encoder is computed by adding a special <RET> token in the last position. The distance between textual and vision embeddings are minimized through a Triplet Margin Loss [4], transferring in consequence the spatial distribution of the visual embeddings into the text encoder.

Multi-Script Learning in Text Recognition In a personalized learning framework, the goal is for a federated model to effectively incorporate new data, typically through layerwise fine-tuning. In this section, we adopt a standard Vision Transformer (ViT) architecture as our reference model (see [3]), pre-trained on the HierText dataset [22]. This model serves as our baseline.

![](images/c4a93f43840efac5308600ebea4735aa0dbf4ba173d463777df5e76c844f078a.jpg)  
Figure 2: Knowledge distillation scheme: multiple pretrained OCRs are combined into a frozen teacher, which guides an RNN to map text to images in unseen alphabets or domains.

![](images/c6e802cb826a44a9a55d388c5507b402bc9a0582124d2e6372286477eade6756.jpg)  
Figure 3: Distributed models trained on Latin alphabets are evaluated on unseen scripts by averaging backbones and fine-tuning only the language head.

Starting from this baseline, we conduct pre-training using a collection of datasets described in Section 4.1. Two training paradigms are considered: (i) a centralized setting, where all datasets are combined and used to train a single model, and (ii) a distributed setting, where separate models are trained on individual datasets and later merged through model merging.

Since all pre-training datasets predominantly contain Latin characters (along with arabic numerals), we frame the recognition of an entirely new alphabet as a personalized learning task. Specifically, we perform personalized fine-tuning on ciphered and multilingual datasets (see Section 4.1), using limited data and a small number of epochs, and restricting updates to the later stages of the network. As shown in Figure 3, to accommodate the new alphabet, the OCR language modeling head is replaced with a single layer whose output dimension matches the size of the new character dictionary.

Under our hypothesis, the distributed pre-training approach should enable the model to incorporate personalized data more effectively, resulting in improved recognition performance compared to the centralized model, when trained under identical hyperparameter settings and number of epochs.

Additionally, we report results for experiments that do not rely on HierText pre-training.

In this alternative setup, models are initialized randomly (using the same seed for both centralized and distributed approaches), trained independently on each dataset, and subsequently merged via model merging in the distributed case.

Fine-tuning in Graph-Based Table Recognition We also include experiments regarding End-to-End training of Graph Neural Networks in the context of Table Recognition; where, as seen in [31], we consider text regions in a document as nodes consisting of their $\left( x _ { 1 } , y _ { 1 } , x _ { 2 } , y _ { 2 } \right)$ coordinates and a vector class consisting of a histogram of the alphanumeric characters of the OCR text as node features. Then, a visibility graph is constructed with edges connecting adjacent nodes. The objective of the message passing algorithm is to classify each node as table, non-table or header categories only from its geometry, connectivity and the histogram of characters. For further details on the graph construction we refer readers to the cited original implementation.

As seen in Figure 4, we train N graph-based Table Recognition models independently (hence, distributedly) and then merge their message passing layers (i.e. the weights of a Graph Neural Network). We, then, fine-tune the model on an unseen dataset (N + 1) and test the performance in this fine-tuning dataset. Namely, each i-th node is updated on a k-th step with weights $\theta ^ { K }$ as follows:

$$
h _ { i } ^ { ( k + 1 ) } = \sigma \left( \sum _ { j \in \mathcal { N } ( i ) } \theta ^ { ( k ) } h _ { j } ^ { ( k ) } \right)\tag{4}
$$

Where $h _ { i } ^ { k }$ denotes the node i features during the message passing step k, σ represents a non-linearity and $j \in \mathcal N$ denotes the set of nodes connected to i. Consequently, for a distributed message passing model the following expression is used:

$$
h _ { i } ^ { ( k + 1 ) } = \sigma \left( \sum _ { j \in \mathcal { N } ( i ) } \frac { 1 } { N } \sum _ { d _ { n } \in D } \theta ^ { ( k , d _ { n } ) } h _ { j } ^ { ( k ) } \right)\tag{5}
$$

Where the message is computed by using the k-th layer on every GNN trained on a different distributed dataset $d _ { n }$ and then averaged to obtain a distributedly learned representation. It is important to note that the node representation and the edge construction requires to be the same in both source and target tasks. This works for systems on which node features are agnostic to the training data.

## 4 Experimental Setup

The employed pre-training and fine-tuning datasets, as well as the evaluation setup, are defined in this section. When referring to a fine-tuning dataset, we denote the process in which a model, after being pre-trained using either a centralized or distributed strategy, is fine-tuned on that dataset individually. The effectiveness of the fine-tuning stage is then assessed using the corresponding test partition of the same dataset.

## 4.1 Datasets

In this Section, we list the different 27 employed datasets and the rationale behind the choice of each of them. For a comprehensive summary of the datasets and tasks used for this study reader may refer to Table 1.

![](images/b8bee9c4e5685caada8ece4601461a9b99275e6ae3152c6e1b8bbed4f345a3f2.jpg)  
Figure 4: Distributed learning of message passing layers. Standard models (red/blue) are combined via distributed message passing (green), aggregating edge parameters (θ) while keeping node features $( h _ { j } )$ fixed.

Knowledge Distillation in Word Spotting For the Word Spotting task, we first train an encoder which, as introduced in the methodology section, serves as a frozen teacher within a knowledge distillation framework. This encoder is trained using what we define as in-domain HTR and OCR datasets, which share a common Latin alphabet and consist of well-cropped, centered text samples with limited font variability.

Specifically, the handwritten text datasets include IAM [23], Esposalles [32], George Washington and Parzival [9], while the scene text datasets comprise CoCoText [33] and the Latin split of MLT19 [28]. Due to their limited contribution to the overall training corpus (stemming from their reduced size and narrow scope) the SVT [34] and IIIT5K [26] datasets are excluded from the training process and instead used as control variables to evaluate performance on unseen yet still in-domain datasets.

Subsequently, knowledge from the trained vision encoder is distilled into a text encoder for the Query-by-String Word Spotting task. We define the Historical Maps [35], SROIE [18], and FUNSD [16] datasets as out-of-domain due to the presence of printed artifacts acting as background distractors in word images. Similarly, the WordArt dataset [37] and TotalText [8] are considered out-of-domain because of their inclusion of artistic and highly variable fonts. The AMR dataset [21] is also employed as a fine-tuning dataset, motivated by the presence of glares and visual artifacts that partially occlude textual information. In addition to the aforementioned out-of-domain datasets containing primarily Latin characters, we further incorporate the Arabic, Chinese, Japanese, Korean, Hindi, and Bangla partitions of MLT19 [28] as both out-of-domain and out-of-vocabulary fine-tuning datasets, representing the most challenging scenario for knowledge distillation. Finally, the Copiale [20], Borg [1], and Vatican cipher datasets [13] are included under the same rationale, as they introduce previously unseen symbol systems and vocabularies. We report the performance of this fine-tuning strategy across all previously described datasets, including in-domain, out-ofdomain Latin, and out-of-domain out-of-vocabulary scenarios. For each dataset, a different model and independent is fine-tuned for ten epochs and evaluated on the corresponding test partition.

Multi-Script Learning in Text Recognition In the case of multi-script personalized Text Recognition, we are only interested in the capacity of the model to integrate highly personalized data, hence, data that significantly differs from the original training distribution. We make use of the same latin alphabets datasets as in the previous section. Namely, we train a latin OCR model using IAM [23], Esposalles [32], George Washington [9], Parzival, CoCoText [33] and the Latin split of MLT19 [28].

<table><tr><td>Dataset Name</td><td>Tasks</td><td>Finetuning</td><td>Pretraining</td><td>Alphabet</td></tr><tr><td>IAM [3]</td><td>WS</td><td>KD</td><td>√</td><td>Latin</td></tr><tr><td>Esposalles [[]]</td><td>WS</td><td>KD</td><td>√</td><td>Latin</td></tr><tr><td>George Washington [0]</td><td>WS</td><td>KD</td><td>√</td><td>Latin</td></tr><tr><td>Parzival [0]</td><td>WS</td><td>KD</td><td>√</td><td>Latin</td></tr><tr><td>CoCoText [3]</td><td>WS</td><td>KD</td><td>√</td><td>Latin</td></tr><tr><td>MLT19 (Latin) [28]</td><td>WS</td><td>KD</td><td>√</td><td>Latin</td></tr><tr><td>SVT [B4]</td><td>WS</td><td>KD</td><td>x</td><td>Latin</td></tr><tr><td>IIIT5K [06]</td><td>WS</td><td>KD</td><td>x</td><td>Latin</td></tr><tr><td>MLT19 (Arabic) [28]</td><td>WS, OCR</td><td>KD /PL</td><td>x</td><td>Arabic</td></tr><tr><td>MLT19 (Chinese) [28]</td><td>WS, OCR</td><td>KD /PL</td><td>x</td><td>Chinese</td></tr><tr><td>MLT19 (Japanese) [28]</td><td>WS, OCR</td><td>KD /PL</td><td>x</td><td>Japanese</td></tr><tr><td>MLT19 (Korean) [8]</td><td>WS, OCR</td><td>KD /PL</td><td>x</td><td>Korean</td></tr><tr><td>MLT19 (Hindi) [28]</td><td>WS, OCR</td><td>KD /PL</td><td>x</td><td>Devanagari</td></tr><tr><td>MLT19 (Bangla) [28]</td><td>WS, OCR</td><td>KD /PL</td><td>x</td><td>Bangla</td></tr><tr><td>Copiale [[0]</td><td>WS, HTR</td><td>KD /PL</td><td>x</td><td>Ciphered</td></tr><tr><td>Borg [0]</td><td>WS, HTR</td><td>KD /PL</td><td>x</td><td>Ciphered</td></tr><tr><td>Vatican [[3]</td><td>WS, HTR</td><td>KD /PL</td><td>x</td><td>Mixed</td></tr><tr><td>Historical Maps []</td><td>WS</td><td>KD</td><td>x</td><td>Latin</td></tr><tr><td>SROIE []</td><td>WS</td><td>KD</td><td>x</td><td>Latin</td></tr><tr><td>FUNSD []</td><td>WS</td><td>KD</td><td>x</td><td>Latin</td></tr><tr><td>AMR []</td><td>WS</td><td>KD</td><td>x</td><td>Digits</td></tr><tr><td>TotalText [8]</td><td>WS</td><td>KD</td><td>x</td><td>Latin</td></tr><tr><td>WordArt [B]]</td><td>WS</td><td>KD</td><td>x</td><td>Latin</td></tr><tr><td>ICDAR2019 []</td><td>TR</td><td>E2E</td><td>√</td><td>Latin</td></tr><tr><td>RVL-CDIP []</td><td>TR</td><td>E2E</td><td>√</td><td>Latin</td></tr><tr><td>con-anonym []</td><td>TR</td><td>E2E</td><td>√</td><td>Latin</td></tr><tr><td>M96</td><td>TR</td><td>E2E</td><td>x</td><td>Latin</td></tr></table>

Table 1: Overview of the 27 evaluation datasets. For each dataset, we indicate whether it is in-pretraining or out-of-domain.

Because of Latin Out-of-Distribution data constitutes nothing but an extension of the original training data, we only consider non-latin alphabets as suitable for a personalized learning evaluation. We include fine-tuning on the Arabic, Chinese, Japanese, Korean, Hindi, and Bangla partitions of MLT19 [28]and the Copiale [20], Borg [1], and Vatican cipher datasets [13]. We include both the performance for personalized OCR systems for every language and personalized approaches to multi-linguality (i.e. learning all new alphabets simultaneously).

Fine-tuning in Graph-Based Table Recognition Because of the sparsity in categories of Table Recognition benchmarks, we have taken the decision to simplify every dataset to contain only Table / Not-Table labels, hence posing a binary segmentation problem. We have conducted experiments with the well-known dataset from the ICDAR 2019 Competition [10], RLV-CDIP [11] and Con-Anonym [31]. Additionally, and perhaps more interestingly, we present an evaluation on a privately owned dataset $( \mathbf { M } 9 6 ^ { 2 } )$ consisting of only 50 pagelevel annotations. Hence yielding insights on the capacity of such models to integrate new distributions of data in a low-resource set-up.

![](images/4ea40da5508bd14cd635bd2c5398d9443f3fb8fdbde76bcca3d1f795de92212f.jpg)  
Figure 5: Mean Average Precision (mAP) for Query-by-String Word Spotting across 22 datasets using GRU, LSTM, and Transformer. Distributed (squares) excels in out-of-domain tasks, especially with shared alphabets.

## 4.2 Metrics and evaluation

The evaluation protocol on the presented experiments is constructed as follows: First, for every task, a model is trained with every data available as training dataset; this is a centralized model containing all the training datasets information, noted as $\theta ^ { D }$ . For every individual dataset, we also train its corresponding individual model with the same number of steps as the centralized model $\theta ^ { d _ { n } }$ . These models are aggregated following Equation 3, which yields the distributed version of $\theta ^ { D } , \overline { { \theta _ { D } } }$

Note that training N data points $( N = n _ { 1 } + n _ { 2 } )$ during T epochs / steps causes $N \times T$ forward/backward steps. In the case of decomposing the dataset, the total cost is $n _ { 1 } \times T +$ $n _ { 2 } \times T$ forward / backward steps. Note that product is distributive, therefore this expression is converted to $T \times \left( n _ { 1 } + n _ { 2 } \right)$ , which knowing that $n _ { 1 }$ and $n _ { 2 }$ are parts of N $( N = n _ { 1 } + n _ { 2 } )$ , the total cost and the data seen by each model is exactly equivalent.

Second, we consider a fine-tuning dataset, for which we train (using its train partition) during 25 epochs every model (distributed or centralized). This specialized model is tested on its fine-tuning task using the corresponding metric for each one of the showcased applications; this is:

Word Spotting We follow a Query-by-String (QbS) evaluation protocol using the mean Average Precision (mAP) metric. In this approach, given a set of test queries (text strings) and a gallery of test images (word crops), the model retrieves images by minimizing the cosine distance between the query text embedding and the image embeddings.

<table><tr><td colspan="2">mAP</td></tr><tr><td>Mean mAP (C / D)</td><td>.313 / .363</td></tr><tr><td>Recall</td><td></td></tr><tr><td>R@1 (C/D)</td><td>.256/.300</td></tr><tr><td>R@5 (C/D)</td><td>.388/.451</td></tr><tr><td>R@10(C/D)</td><td>.447/.516</td></tr></table>

<table><tr><td>Stat</td><td>ID</td><td>OOD-L</td><td>O0D-U</td></tr><tr><td>Abs. gain</td><td>.005</td><td>.119</td><td>.034</td></tr><tr><td>Gain (%)</td><td>11.58%</td><td>36.68%</td><td>104.02%</td></tr><tr><td>Std gain</td><td>.109</td><td>.056</td><td>.063</td></tr><tr><td>Min gain</td><td>-.228</td><td>.050</td><td>-.181</td></tr><tr><td>Max gain</td><td>.146</td><td>.222</td><td>.172</td></tr></table>

<table><tr><td>Encoder</td><td>Regime</td><td>mAP (± std)</td></tr><tr><td rowspan="2">GRU</td><td>C</td><td> $. 3 5 6 \pm . 2 9 4$ </td></tr><tr><td>D</td><td> $\mathbf { . 3 9 9 } \pm . 2 9 9$ </td></tr><tr><td rowspan="2">LSTM</td><td>C</td><td> $. 3 5 8 \pm . 2 9 3$ </td></tr><tr><td>D</td><td> $. 4 1 2 \pm . 3 0 1$ </td></tr><tr><td rowspan="2">Transformer</td><td>C</td><td> $. 1 8 4 \pm . 2 0 5$ </td></tr><tr><td>D</td><td> $. 2 4 6 \pm . 2 3 8$ </td></tr></table>

Table 2: Centralized (C) vs. distributed (D) pre-training: average recall across word spotting datasets, including ID, OOD-L, and OOD-U scenarios.

The mean Average Precision (mAP) is computed as the average of the Average Precision (AP), defined as:

$$
\mathrm { A P } ( q ) = { \frac { 1 } { N _ { q } } } \sum _ { k = 1 } ^ { n } P ( k ) \cdot \mathrm { r e l } ( k ) ,\tag{6}
$$

where $N _ { q }$ is the number of relevant (ground-truth) images for that query and n is the total number of retrieved images. Here, $P ( k )$ represents the precision at rank k, defined as the number of relevant images in the top k results divided by k, and rel(k) is a binary relevance indicator that equals 1 if the image at rank k is relevant to the query and 0 otherwise.

Text Recognition We evaluate character recognition performance using word-level accuracy. We conduct experiments under two training paradigms: (1) language-specific models trained independently on each language, and (2) a single multi-lingual model trained jointly on all languages to assess cross-lingual transfer capabilities.

Word-level Accuracy measures the percentage of words that are transcribed perfectly without any character errors:

$$
{ \mathrm { A c c u r a c y } } = { \frac { \# \operatorname { c o r r e c t l y } \operatorname { t r a n s c r i b e d } \operatorname { w o r d s } } { \# \operatorname { t o t a l } \operatorname { w o r d s } } } \times 1 0 0 \%\tag{7}
$$

Table Recognition We evaluate Table Recognition performance using a graph-based approach with two complementary metrics: node accuracy and edge accuracy. The model represents document structure as a graph where nodes correspond to semantic regions and edges encode spatial relationships between them. In one hand, node accuracy reflects how well the model identifies the type of each region. In our case, nodes represent regions labeled as either table or non-table. A node is considered correct if the predicted label matches the ground truth. On the other, edge accuracy measures how well the model captures spatial relationships between regions. This includes relationships within the same region, between regions of the same type, and between regions of different types. An edge is correct if both its presence and relationship type match the ground truth.

## 5 Results

This section presents results for the three tasks introduced earlier. We analyze when distributed learning setups improve fine-tuning with knowledge distillation for Word Spotting, personalized text recognition, and E2E table detection.

## 5.1 Word Spotting

The main results of knowledge distillation for Query-by-String Word Spotting are shown in Figure 5. For latin datasets included in pre-training, results are mixed: fine-tuning from distributed setups occasionally degrades performance, yet six out of eight datasets benefit from distributed pre-training in at least one architecture, yielding improvements in 75% of cases. Out-of-domain fine-tuning consistently favors distributed pre-trained models. Datasets not seen during centralized or distributed pre-training show clear gains, including for unseen alphabets. Performance improvements are larger when the fine-tuning alphabet matches that of pre-training, highlighting both the generality and limitations of the learned features. Overall, Table 2 reports an average improvement of 5% in mean average precision. In-domain gains can be negative, but out-of-domain datasets show substantially larger improvements across both Latin and unseen alphabets. Transformer-based text encoders achieve the lowest absolute performance, likely due to higher data requirements, yet relative gains from distributed pre-training are comparable to LSTM encoders, with an average increase of 6% in mean average precision.

## 5.2 Optical Text Recognition

Table 3 shows results for three training setups: (1) fine-tuning from a baseline trained only on HierText, (2) centralized pre-training on HierText and a multilingual dataset before perlanguage fine-tuning, and (3) the fully distributed version. We also include distributed learning from random initialization without auxiliary pre-training. We observe a consistent advantage of our personalized learning approach with respect the centralized and baseline approaches, with an average improvement of ×1.62 in the best-case scenario. In some cases, such as the Borg cipher, distributed learning is essential for the weights to accommodate the inclusion of such a different domain. Needless to say, as it was specified on previous sections, both centralized and distributed pretraining have performed the same amount of forward-backward steps on the same exact data; hence, we can only attribute the incapacity of the centralized approach to incorporate certain languages to the training regime itself. Additionally, we include results for multilingual training (Table 4). In contrast to the previous experiments, where languages were fine-tuned independently, these settings consider an optical character recognition system trained to learn all alphabets simultaneously. Under this formulation, we observe that the performance gap between centralized and distributed training regimes becomes narrower. Notably, accuracy for the Borg cipher is fully recovered. This suggests that distributed models adapt more effectively in low-resource scenarios, whereas centralized approaches are sufficient when abundant data is available in the personalization step. Nevertheless, even in these settings, the distributed approach outperforms the centralized model.

## 5.3 Table Detection

The results for graph-based recognition, reported in Table 5, correspond to a leave-one-out evaluation at the dataset level. In this setup, models are trained on all datasets except the one indicated by the corresponding column. For instance, zero-shot results in the RLV column are obtained by training on M96, ICDAR, and Con-Anonym while excluding RLV from pre-training. Conversely, end-to-end results in the ICDAR column correspond to models pre-trained on M96, RLV, and Con-Anonym and subsequently fine-tuned on ICDAR. Zeroshot results, where no fine-tuning is performed on the target dataset, consistently favor the distributed approach in both node and edge accuracy. With partial fine-tuning of the upper layers, performance becomes mixed, while end-to-end fine-tuning generally benefits the centralized model, with the exception of M96. These results support our hypothesis: distributed models are most effective in low-resource regimes, either when no adaptation is possible (zero-shot) or when only limited data is available (e.g., M96). As data availability and adaptation capacity increase, performance progressively shifts toward centralized training, with partial fine-tuning representing an intermediate regime.

<table><tr><td></td><td>Vatican</td><td>Borg</td><td>Copiale</td><td>Arabic</td><td>Chinese</td><td>Japanese</td><td>Korean</td><td>Bangla</td><td>Hindi</td><td>×∆</td></tr><tr><td>From Baseline</td><td>.549</td><td>.382</td><td>.825</td><td>.131</td><td>.020</td><td>.116</td><td>.282</td><td>.260</td><td>.470</td><td></td></tr><tr><td>Centr. (base)</td><td>.465</td><td>.000</td><td>.794</td><td>.175</td><td>.0103</td><td>.0951</td><td>.214</td><td>.115</td><td>.384</td><td>0.69</td></tr><tr><td>Dist. (base)</td><td>.591</td><td>.505</td><td>.838</td><td>.422</td><td>.073</td><td>.197</td><td>.374</td><td>.462</td><td>.514</td><td>1.62</td></tr><tr><td>Dist. (random)</td><td>.480</td><td>.272</td><td>.930</td><td>.410</td><td>.010</td><td>.114</td><td>.271</td><td>.266</td><td>.282</td><td>1.04</td></tr></table>

Table 3: Transfer learning accuracy across languages. Distributed fine-tuning from a shared pretrained model (Z) outperforms centralized fine-tuning and task arithmetic from scratch, especially on low-resource alphabets.

Table 4: Multi-lingual and multi-cipher training results.
<table><tr><td></td><td colspan="6">Multi-Lingual</td><td colspan="3">Multi-Cipher</td></tr><tr><td>Accuracy ↑</td><td>Arabic</td><td>Bangla</td><td>Chinese</td><td>Hindi</td><td>Japanese</td><td>Korean</td><td>Borg</td><td>Copiale</td><td>Vatican</td></tr><tr><td>Distributed</td><td>.472</td><td>.469</td><td>.127</td><td>.539</td><td>.252</td><td>.435</td><td>.573</td><td>.840</td><td>.566</td></tr><tr><td>Centralized</td><td>.402</td><td>.385</td><td>.076</td><td>.468</td><td>.183</td><td>.324</td><td>.573</td><td>.819</td><td>.524</td></tr></table>

## 6 Conclusions

In this study, we conducted an extensive set of comparative experiments between centralized and distributed learning paradigms. Our analysis goes beyond the commonly cited privacy and security benefits of distributed learning, and instead focuses on identifying the conditions under which it also provides performance advantages. This naturally raises the question: under which circumstances does distributed learning yield the greatest gains? To achieve sound conclusions in the study, we have designed a comprehensive set of experiments corresponding to a variety of settings in terms of tasks, learning scenarios and datasets. In the Word

<table><tr><td></td><td colspan="2">M96</td><td colspan="2">RLV</td><td colspan="2">ICDAR</td><td colspan="2">CON-Anonym</td></tr><tr><td></td><td>Node Acc</td><td>Edge Accuracy</td><td>Node Acc</td><td>Edge Accuracy</td><td>Node Acc</td><td>Edge Accuracy</td><td>Node Acc</td><td>Edge Accuracy</td></tr><tr><td>Centr. (zero-shot)</td><td>87.50%</td><td>85.70%</td><td>76.70%</td><td>72.90%</td><td>68.10%</td><td>78.00%</td><td>75.50%</td><td>70.10%</td></tr><tr><td>Dist. (zero-shot)</td><td>88.00%</td><td>87.00%</td><td>78.10%</td><td>78.00%</td><td>79.00%</td><td>84.70%</td><td>73.00%</td><td>77.20%</td></tr><tr><td>Centr. (partial)</td><td>87.60%</td><td>85.20%</td><td>76.90%</td><td>69.40%</td><td>82.60%</td><td>81.70%</td><td>89.30%</td><td>73.80%</td></tr><tr><td>Dist. (partial)</td><td>88.00%</td><td>86.50%</td><td>79.70%</td><td>72.30%</td><td>83.40%</td><td>78.70%</td><td>89.10%</td><td>75.40%</td></tr><tr><td>Centr. (E2E)</td><td>86.10%</td><td>69.90%</td><td>85.30%</td><td>81.50%</td><td>92.00%</td><td>95.50%</td><td>97.80%</td><td>90.60%</td></tr><tr><td>Centr. ([])</td><td></td><td></td><td>67.06%</td><td>83.02%</td><td>92.20%</td><td>91.60%</td><td>84.74%</td><td>89.42%</td></tr><tr><td>Dist. (E2E)</td><td>88.90%</td><td>78.30%</td><td>86.10%</td><td>79.80%</td><td>90.60%</td><td>94.60%</td><td>97.70%</td><td>89.80%</td></tr></table>

Table 5: Table Recognition results: Node (cell) and Edge (link) accuracy. Models are trained on three datasets and fine-tuned on the fourth.

Table 6: Summary of the conclusions and general performance trends.
<table><tr><td>Regime</td><td>Low-Data</td><td>ZS</td><td>OOD</td><td>Param.-Efficient</td><td>E2E</td><td>ID</td></tr><tr><td>Distributed</td><td>Strong</td><td>Strong</td><td>Strong</td><td>Strong</td><td>Moderate</td><td>Low</td></tr><tr><td>Centralized</td><td>Moderate</td><td>Low</td><td>Moderate</td><td>Moderate</td><td>Strong</td><td>Strong</td></tr></table>

Spotting experiments, where only the text encoder is trainable and the vision encoder remains frozen, results (Figure 5) are mixed when fine-tuning is performed in-domain. However, in out-of-domain settings, distributed approaches consistently dominate. This effect is particularly pronounced for GRU- and LSTM-based text encoders (Table 2) which, in contrast to Transformer-based architectures, achieve strong performance with substantially lower data requirements. For character recognition, distributed learning exhibits its largest advantage when languages are learned independently (Table 3). When fine-tuning data is abundant, as in the multilingual setting (Table 4), the performance gap narrows. In graph-based Table Recognition, contrary to our initial expectations, end-to-end fine-tuning proves to be the least favorable scenario for distributed models (Table 5).

What overarching pattern emerges from these findings? From an architectural perspective, distributed learning performs best in scenarios with a limited number of learnable parameters. In Table Recognition, performance is strongest in the zero-shot setting, becomes mixed when only part of the model is fine-tuned, and degrades under full end-to-end finetuning. Similarly, in Word Spotting, lightweight architectures such as GRUs and LSTMs benefit more from distributed learning than heavier Transformer models. These results suggest that practitioners employing parameter-efficient architectures or operating under constraints that prevent fine-tuning can particularly benefit from distributed pre-training. From a data-centric perspective, distributed learning is most effective in low-resource regimes. In Table Recognition, the largest gains are observed on the smallest dataset, M96. A similar trend appears in character recognition, where personalization to new alphabets yields a larger advantage when languages are trained independently than in joint multilingual training, where data availability is higher and the performance gap diminishes. Overall, these findings (summarized in Table 6) indicate that distributed pre-training is especially advantageous for practitioners working with limited data or in low-resource scenarios.

In conclusion, the question raised by the title becomes clear: decentralization in Document Analysis is not a universal prescription, but a targeted call. It is addressed to practitioners operating under low-resource conditions, limited adaptability, or stringent deployment constraints, for whom distributed learning consistently delivers competitive and often superior performance. Domains such as historical Document Analysis stand out as particularly well aligned with this paradigm. In these settings, annotations are rarely abundant, and data distributions often diverge significantly from those of modern, large-scale datasets. Decentralized learning offers a promising pathway in such contexts, further reinforced by the privacy and access constraints commonly imposed by cultural heritage institutions and archives. A similar alignment can be found in industrial environments involving privately owned administrative records, where automation is hindered by the sensitivity of the data and the infeasibility of large-scale annotation. In this regard, small and medium-sized enterprises, which frequently handle confidential documents without the resources or legal capacity to externalize a labeling procedure, represent a natural and highly relevant application domain for distributed Document Analysis.

## Acknowledgments

This work has been partially supported by the Spanish project PID2024-157778OB-I00, Ministerio de Ciencia e Innovación, the Departament de Cultura of the Generalitat de Catalunya, and the CERCA Program. Adrià Molina is funded with the PRE2022-101575 grant provided by MCIN / AEI / 10.13039 / 501100011033 and by ERDF/EU. We extend our gratitude to Jialuo Chen for his work on adapting [31] to our framework.

## References

[1] Nada Aldarrab, Kevin Knight, and Beáta Megyesi. The Borg.lat.898 Cipher.

[2] Manoj Ghuhan Arivazhagan, Vinay Aggarwal, Aaditya Kumar Singh, and Sunav Choudhary. Federated learning with personalization layers. 2019.

[3] Rowel Atienza. Vision transformer for fast and efficient scene text recognition. In ICDAR, pages 319–334. Springer, 2021.

[4] Vassileios Balntas, Edgar Riba, Daniel Ponsa, and Krystian Mikolajczyk. Learning local feature descriptors with triplets and shallow convolutional neural networks. In Bmvc.

[5] Brien Brothman. The past that archives keep: memory, history, and the preservation of archival records. Archivaria, pages 48–80, 2001.

[6] Nicolas Carlini, Jamie Hayes, Milad Nasr, Matthew Jagielski, Vikash Sehwag, Florian Tramer, Borja Balle, Daphne Ippolito, and Eric Wallace. Extracting training data from diffusion models. In 32nd USENIX Security, 2023.

[7] Junyoung Chung, Caglar Gulcehre, KyungHyun Cho, and Yoshua Bengio. Empirical evaluation of gated recurrent neural networks on sequence modeling. arXiv preprint arXiv:1412.3555, 2014.

[8] Chee-Kheng Ch’ng, Chee Seng Chan, and Cheng-Lin Liu. Total-text: toward orientation robustness in scene text detection. International Journal on Document Analysis and Recognition (IJDAR), 23(1):31–52, 2020.

[9] Andreas Fischer, Andreas Keller, Volkmar Frinken, and Horst Bunke. Lexicon-free handwritten word spotting using character hmms. Pattern recognition letters, 33(7):934–942, 2012.

[10] Liangcai Gao, Yilun Huang, Hervé Déjean, Jean-Luc Meunier, Qinqin Yan, Yu Fang, Florian Kleber, and Eva Lang. Icdar 2019 competition on table detection and recognition (ctdar). In 2019 ICDAR. IEEE, 2019.

[11] Adam W Harley, Alex Ufkes, and Konstantinos G Derpanis. Evaluation of deep convolutional nets for document image classification and retrieval. In 2015 13th ICDAR, 2015.

[12] Chaoyang He, Keshav Balasubramanian, Emir Ceyani, Carl Yang, Han Xie, Lichao Sun, Lifang He, Liangwei Yang, Philip S Yu, Yu Rong, et al. Fedgraphnn: A federated learning system and benchmark for graph neural networks. arXiv preprint arXiv:2104.07145, 2021.

[13] Mihály Héder and Beáta Megyesi. The decode database of historical ciphers and keys: Version 2. In International Conference on Historical Cryptology, pages 111–114, 2022.

[14] Sepp Hochreiter and Jürgen Schmidhuber. Long short-term memory. Neural computation, 1997.

[15] Gabriel Ilharco, Marco Tulio Ribeiro, Mitchell Wortsman, Ludwig Schmidt, Hannaneh Hajishirzi, and Ali Farhadi. Editing models with task arithmetic. In The Eleventh International Conference on Learning Representations, 2023.

[16] Guillaume Jaume, Hazim Kemal Ekenel, and Jean-Philippe Thiran. Funsd: A dataset for form understanding in noisy scanned documents. In 2019 ICDAR Workshops (ICDARW). IEEE.

[17] Yihan Jiang, Jakub Konecnˇ y, Keith Rush, and Sreeram Kannan. Improving federated learning \` personalization via model agnostic meta learning. arXiv preprint arXiv:1909.12488, 2019.

[18] Dimosthenis Karatzas, Faisal Shafait, Seiichi Uchida, Masakazu Iwamura, Lluis Gomez i Bigorda, Sergi Robles Mestre, Joan Mas, David Fernandez Mota, Jon Almazan Almazan, and Lluis Pere De Las Heras. Icdar 2013 robust reading competition. In 2013 12th ICDAR, pages 1484–1493. IEEE, 2013.

[19] Eric Ketelaar. Sharing, collected memories in communities of records. Archives and manuscripts, 2005.

[20] Kevin Knight, Beáta Megyesi, and Christiane Schaefer. The Copiale Cipher. In Proceedings of the 4th Workshop on Building and Using Comparable Corpora: Comparable Corpora and the Web, pages 2–9, Portland, Oregon, June 2011. Association for Computational Linguistics.

[21] R. Laroca and et.al. Convolutional neural networks for automatic meter reading. Journal of Electronic Imaging, 2019.

[22] Shangbang Long, Siyang Qin, Dmitry Panteleev, Alessandro Bissacco, Yasuhisa Fujii, and Michalis Raptis. Towards end-to-end unified scene text detection and layout analysis. In Proceedings ofthe IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2022.

[23] U-V Marti and Horst Bunke. The iam-database: an english sentence database for offline handwriting recognition. International Journal on Document Analysis and Recognition, 5:39–46, 2002.

[24] Koji Matsuda, Yuya Sasaki, Chuan Xiao, and Makoto Onizuka. An empirical study of personalized federated learning. arXiv preprint arXiv:2206.13190, 2022.

[25] H. Brendan McMahan, Eider Moore, Daniel Ramage, Seth Hampson, and Blaise Agüera y Arcas. Communication-efficient learning of deep networks from decentralized data, 2023.

[26] Anand Mishra, Karteek Alahari, and CV Jawahar. Scene text recognition using higher order language priors. In BMVC-British machine vision conference. BMVA, 2012.

[27] Adrià Molina Rodríguez, Oriol Ramos Terrades, and Josep Lladós. The ocr quest for generalization: Learning to recognize low-resource alphabets with model editing. arXiv preprint arXiv:2506.06761, 2025.

[28] Nibal Nayef, Yash Patel, Michal Busta, Pinaki Nath Chowdhury, Dimosthenis Karatzas, Wafa Khlif, Jiri Matas, Umapada Pal, Jean-Christophe Burie, Cheng-lin Liu, et al. Icdar2019 robust reading challenge on multi-lingual scene text detection and recognition—rrc-mlt-2019, 2019.

[29] Khanh Nguyen and Dimosthenis Karatzas. Federated document visual question answering: A pilot study. In ICDAR, 2024.

[30] Martin T Olliff and Elizabeth Dill. The distributed archives model: a strategy for sharing authority with partners to document communities. Archival Issues, 41(1), 2021.

[31] Pau Riba, Lutz Goldmann, Oriol Ramos Terrades, Diede Rusticus, Alicia Fornés, and Josep Lladós. Table detection in business document images by message passing networks. Pattern Recognition, 2022.

[32] Verónica Romero, Alicia Fornés, Nicolás Serrano, Joan Andreu Sánchez, Alejandro H Toselli, Volkmar Frinken, Enrique Vidal, and Josep Lladós. The esposalles database: An ancient marriage license corpus for off-line handwriting recognition. Pattern Recognition, 46(6):1658–1669, 2013.

[33] Andreas Veit, Tomas Matera, Lukas Neumann, Jiri Matas, and Serge Belongie. Coco-text: Dataset and benchmark for text detection and recognition in natural images, 2016.

[34] Kai Wang, Boris Babenko, and Serge Belongie. End-to-end scene text recognition. In 2011 International conference on computer vision, pages 1457–1464. IEEE, 2011.

[35] Jerod Weinman, Ziwen Chen, Ben Gafford, Nathan Gifford, Abyaya Lamsal, and Liam Niehus-Staab. Deep neural networks for text detection and recognition in historical maps. In 2019 ICDAR (ICDAR), pages 902–909. IEEE, 2019.

[36] Mitchell Wortsman, Gabriel Ilharco, Samir Ya Gadre, Rebecca Roelofs, Raphael Gontijo-Lopes, Ari S Morcos, Hongseok Namkoong, Ali Farhadi, Yair Carmon, Simon Kornblith, et al. Model soups: averaging weights of multiple fine-tuned models improves accuracy without increasing inference time. In International conference on machine learning, pages 23965–23998. PMLR, 2022.

[37] Xudong Xie, Linger Deng, Zhifei Zhang, Zhaowen Wang, and Yuliang Liu. Icdar 2024 competition on artistic text recognition. In ICDAR, 2024.

[38] Wenqing et.al Zhang. Communication-efficient federated learning for scene text recognition. 2020.

## Supplementary Material

This text correponds to the supplementary material for the BMVC2026 paper Unapologetically Distributed: A Call for Decentralized Document Analysis . The notes found in the following sections answer some of the questions arised during the review stage, and may be useful for a part of the audience.

## Single-round merging as distributed learning

We choose single-round aggregation because it is the unique operation shared by FedAvg, task arithmetic, model soups and 1-epoch meta-learning (Eq. 3): it lets readers from the federated, model-merging and meta-learning communities read the paper jointly, which is our stated goal. Advanced federated optimizers (FedProx, SCAFFOLD) are corrections to client drift across rounds; at one round they collapse to FedAvg, so under our protocol they introduce no distinct comparison. Regarding multi-round FedAvg, our internal ablation on Table Recognition (Table 7) shows rounds 2–5 shift node/edge accuracy by at most 1–2 points, with no monotonic trend and no change to which regime wins.

## Statistical significance

Table 2 aggregates 22 datasets, so the reported gains act as paired comparisons across many tasks rather than a single run. A one-sided Wilcoxon signed-rank test over all paired centralized/distributed Word Spotting runs confirms significance:

$$
p = 8 . 8 \times 1 0 ^ { - 6 } ( n { = } 7 0 \mathrm { p a i r s } ; \mathrm { d i s t . } \mathrm { w i n s } 5 7 ) .
$$

Table 7 additionally shows small spreads across five independent aggregation repetitions.

## ×1.62 and claim strength

All rows of Table 3, including Centr.(base), share the same HierText pretraining stage; only Dist.(random) starts from scratch, and it still reaches ×1.04, which we consider evidence of robustness rather than a hidden weakness. We will label ×1.62 explicitly as best-case in the text. The negative in-domain gains (Table 2) are deliberate findings: the paper does not claim universal superiority, and Table 6 exists precisely to delimit where distributed learning helps (low-resource, zero-shot, OOD, parameter-efficient) and where it does not (in-domain E2E). We will soften any remaining absolute wording.

## Why does merging generalize better?

As an intuition, parameter averaging acts as an implicit smoothing operator over the loss landscape. Each independently trained model $\theta ^ { d _ { i } }$ converges to a minimum shaped in part by dataset-specific noise, directions that fit peculiarities of $d _ { i }$ but carry no transferable signal. Since these idiosyncratic directions are largely uncorrelated across the n independent runs, averaging them cancels them out, while directions that are consistently reinforced across datasets, corresponding to genuinely shared, task-relevant structure, survive and dominate the resulting $\overline { { \theta } } _ { D }$ The merged model effectively behaves as a coarse ensemble collapsed into a single set of weights, inheriting the flatter, wider regions of the loss surface that independent minima tend to share, which is also the usual explanation for why flat minima generalize better. This is consistent with the pattern observed throughout the paper: distributed pre-training helps most when the downstream task or architecture leaves little room to re-specialize away from this shared component (low-resource, zero-shot, parameter-efficient regimes), and helps least when full end-to-end fine-tuning can freely override it. A full theoretical treatment would require an article of its own; we want to note the experimental basis is comprehensive, in consequence, we position this paper as the empirical foundation such theory needs.

<table><tr><td></td><td colspan="2">M96</td><td colspan="2">RLV</td><td colspan="2">ICDAR</td><td colspan="2">CON</td></tr><tr><td></td><td>Node</td><td>Edge</td><td>Node</td><td>Edge</td><td>Node</td><td>Edge</td><td>Node</td><td>Edge</td></tr><tr><td>ZS - r2</td><td>88.6</td><td>87.3</td><td>76.9</td><td>79.6</td><td>80.6</td><td>76.9</td><td>72.4</td><td>75.7</td></tr><tr><td>ZS - r3</td><td>88.1</td><td>86.7</td><td>75.0</td><td>78.6</td><td>83.5</td><td>82.2</td><td>74.5</td><td>76.7</td></tr><tr><td>ZS - r4</td><td>88.2</td><td>86.8</td><td>75.2</td><td>79.0</td><td>82.2</td><td>78.7</td><td>72.5</td><td>79.4</td></tr><tr><td>ZS - r5</td><td>87.6</td><td>86.7</td><td>74.3</td><td>78.2</td><td>82.3</td><td>81.8</td><td>68.7</td><td>78.9</td></tr><tr><td>Part – r2</td><td>88.6</td><td>87.0</td><td>80.9</td><td>72.3</td><td>83.5</td><td>81.4</td><td>88.5</td><td>76.4</td></tr><tr><td>Part – r3</td><td>88.2</td><td>86.2</td><td>78.7</td><td>71.8</td><td>83.8</td><td>81.4</td><td>89.0</td><td>76.6</td></tr><tr><td>Part – r4</td><td>88.2</td><td>86.4</td><td>80.0</td><td>72.5</td><td>82.5</td><td>79.6</td><td>88.6</td><td>74.8</td></tr><tr><td>Part – r5</td><td>87.8</td><td>86.3</td><td>78.6</td><td>71.2</td><td>82.5</td><td>79.0</td><td>88.9</td><td>75.7</td></tr><tr><td>E2E − r2</td><td>88.7</td><td>77.0</td><td>86.6</td><td>82.2</td><td>91.0</td><td>94.1</td><td>97.6</td><td>91.9</td></tr><tr><td>E2E - r3</td><td>89.0</td><td>73.5</td><td>86.0</td><td>85.9</td><td>91.7</td><td>95.0</td><td>97.7</td><td>92.1</td></tr><tr><td>E2E - r4</td><td>87.6</td><td>80.0</td><td>86.5</td><td>84.0</td><td>90.3</td><td>92.6</td><td>97.7</td><td>89.8</td></tr><tr><td>E2E - r5</td><td>88.4</td><td>80.9</td><td>86.0</td><td>85.7</td><td>91.9</td><td>95.3</td><td>97.7</td><td>91.2</td></tr></table>

Table 7: Multi-round FedAvg on Table Recognition, rounds 2–5 (round 1 and centralized in submitted Tab. 5). No consistent gain beyond round 1.
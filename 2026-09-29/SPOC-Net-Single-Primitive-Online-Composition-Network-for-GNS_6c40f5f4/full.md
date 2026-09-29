# SPOC-Net: Single-Primitive Online Composition Network for GNSS Jamming Set Recognition

Zhihan Zeng, Graduate Student Member, IEEE, Kaihe Wang, Graduate Student Member, IEEE, Jose A. L ´ opez-Salcedo, ´ Senior Member, IEEE, Gonzalo Seco-Granados, Fellow, IEEE, Zhongpei Zhang

Abstract—Reliable positioning, navigation, and timing support intelligent transportation, autonomous systems, and space-airground integrated networks. However, global navigation satellite system (GNSS) jamming recognizers that treat each mixture as a separate class are difficult to extend to new combinations. Therefore, this paper proposes SPOC-Net, which decomposes the recognition problem into identifying a set of basic jamming components. Multi-resolution time-frequency features and learned component queries provide evidence for each component type. A high-resolution branch estimates the number of active types, and a structured decoder combines this estimate with component evidence to select a valid set. For training, measured single-component records are the only physical samples used in gradient optimization. Their associated clean in-phase and quadrature (IQ) sequences are combined on demand during training to produce labeled mixtures with different relative powers and jamming-to-noise ratios. Separate measured mixtures from ten training-listed compositions support model selection and decoder calibration; six other compositions are reserved for final testing. Evaluation on 14,220 independently generated, conductively combined, and recorded radio frequency mixtures yields 80.69% exact-set accuracy and a 92.84% micro-averaged F1 score. On combinations excluded from model development, SPOC-Net achieves 80.89% exact-set accuracy, exceeding the strongest comparison method by 18.77 percentage points under the reported protocols.

Index Terms—GNSS jamming recognition, online IQ composition, compositional generalization, component-set recognition, physical measurement.

## I. INTRODUCTION

tivity, integrated sensing and communication, high-accuracy positioning, and stronger security and resilience [1]. Spaceair-ground integrated networks (SAGINs) connect satellites, aerial platforms, and terrestrial systems to extend services over land, sea, air, and remote regions [2]. These networks depend on reliable positioning, navigation, and timing for synchronization, mobility control, autonomous operation, and emergency services. To support these functions, many civil and industrial systems rely on the global navigation satellite system (GNSS).

A GNSS receiver observes satellite signals after severe propagation loss, and the useful signal is normally close to or below the receiver noise floor before correlation. A nearby jammer can therefore dominate the receiver front end and interrupt acquisition, tracking, or navigation with limited transmitted power [3], [4]. Reliable jamming recognition is needed before a receiver or monitoring system can select an appropriate suppression or monitoring method.

In this context, a receiver must contend with several possible jamming signals, including single-tone, multitone, chirp, pulsed, and partial-band noise interference. Each individual jamming type is termed a primitive, and a composition is a set of primitive types that are simultaneously active. With C primitive types, there are up to $2 ^ { C } - 1$ nonempty compositions. Even when mixed-jamming recognition is restricted to two or three active types, the number of combinations requiring labeled training data grows rapidly. In a controlled dataset-collection campaign, acquiring each mixture requires coordinated signal generation, controlled component powers, repeated receiver captures, and verified labels. These requirements concern the construction of a reproducible labeled dataset; they do not imply that real-world jammers are synchronized or have controlled power relations. The resulting measurement effort makes comprehensive mixture collection difficult.

Most learning-based GNSS jamming methods use a closed class list and assign one softmax label to each single or mixed type [9], [20]–[22], [24], [26]. Such methods can perform well for a fixed taxonomy, but accommodating an additional combination generally requires extending the class list and retraining the classifier. A component-wise output instead identifies the active primitive types and can represent different combinations without defining a separate output neuron for each mixture. Multi-label compound recognition and generalization to combinations excluded from training have already been studied in radar jamming [16]–[19], [40], [41].

This paper proposes the Single-Primitive Online Composition Network for GNSS Jamming Set Recognition (SPOC-Net), a recognition method that learns to identify mixed jamming while using only measured single-component records for gradient optimization. A record containing one primitive type is called a singleton. The associated clean in-phase and quadrature (IQ) sequences are combined before the short-time Fourier transform (STFT) to create additional labeled training examples. Here, online composition means generating a fresh mixture from stored singleton IQ sequences when the training loader requests it, rather than constructing a fixed mixed dataset in advance; it does not mean real-time radio frequency (RF) acquisition or signal combination during inference. The resulting auxiliary examples supplement measured singletons, and their known component labels provide the training targets, or supervision. Relative component powers and aggregate jamming-to-noise ratio (JNR) are sampled by this trainingdata generator.

SPOC-Net receives an STFT image, not a JNR value or acquisition metadata: the sampled JNR controls auxiliarysignal generation, whereas the recorded JNR labels organize the experimental results. Ten compositions are permitted in auxiliary training and are termed training-listed. Separate measured mixtures of these same ten compositions are used for model selection and decoder calibration. Six other combinations of the known primitives are held out: their complete component sets are excluded from auxiliary training and all model-development decisions, and their measured records are used only for final testing. This distinction tests generalization to new combinations of known types, not recognition of an unknown jamming type. The evaluated task contains twocomponent and three-component mixtures; singleton records additionally support an auxiliary training task that distinguishes one component from a mixture.

The contributions of this paper are summarized below.

1) An online clean-IQ composer supplies labeled mixedjamming examples without requiring measured mixtures for gradient updates. It varies component powers, aggregate JNR, and background noise before the STFT, and forms related two-component and three-component examples to train component-count estimation.

2) A new recognition method, referred to as SPOC-Net, combines a coordinate-aware multi-resolution encoder, learned primitive queries, a detached high-resolution cardinality branch, and structured candidate-set scoring. Cardinality denotes the number of active primitive types. The model shares component evidence across compositions and jointly selects component identities and a valid set size.

3) A conducted RF evaluation platform and an explicitly separated development–test protocol assess transfer from singleton-based training to measured mixtures. The platform combines a live GNSS signal with up to three generated jammer signals. Entire held-out compositions, not merely individual recordings, are excluded from gradient training, checkpoint selection, hyperparameter tuning, and decoder calibration.

On 14,220 measured mixed records, SPOC-Net achieves 80.69% exact-set accuracy and 92.84% micro-averaged F1 (micro-F1). Its held-out exact-set accuracy is 80.89%, compared with 62.12% for the strongest comparison method. These results assess the complete training and inference procedures described below; they are not an architecture-only comparison.

The remainder of this paper is organized as follows. Section II reviews related work. Section III defines the receivedsignal model, primitive waveforms, recognition task, and datause protocol. Section IV presents the complete SPOC-Net method, from signal representation and online composition to network architecture, learning, and decoding. Section V describes the experimental setup and comparison protocols. Section VI analyzes the results, and Section VII concludes the paper.

## II. RELATED WORK

## A. Conventional GNSS Interference Detection and Mitigation

GNSS interference protection usually includes detection and mitigation. Before correlation, useful satellite signals are close to or below the noise floor, so interference detection often relies on changes in received power, signal statistics, or patterns in the time-frequency plane. Statistical methods can detect both stationary and nonstationary interference [5], [6], while power and distortion monitoring can provide receiverlevel evidence of abnormal signals [4]. These approaches are interpretable and usually do not require a large labeled dataset, but they mainly determine whether interference exists and do not always identify every active component in a compound record.

Mitigation methods are commonly matched to signal structure. Adaptive notch filters are widely used for narrowband interference [7], [33], [34], [46]–[49], while subband gain control provides frequency-selective suppression [35]. Chirp interference can be processed through fractional Fourier and Zak transform methods [8], [13], [14]. Nonnegative matrix factorization can separate structured components in a timefrequency representation [11], [12], and antenna-array or polarization methods can exploit spatial and polarization information [23]. Since each method depends on a particular signal feature, reliable component recognition is needed before an appropriate suppression method can be selected.

## B. Learning-Based GNSS Interference Recognition

Machine learning has been widely studied for GNSS interference recognition. Spectra, spectrograms, and spectrumwaterfall images are common inputs because they expose both frequency positions and temporal variations. Support vector machines and convolutional neural networks have been used to classify interference types from these representations [30], [36], [39]. Fingerprint-spectrum networks and temporal-spatial feature aggregation further improve the use of spectral and local pattern information [9], [24].

Recent studies also address limited training data, low computational cost, and difficult JNR conditions. Low-resource classifiers and lightweight networks have been developed for practical GNSS receivers [15], [20], [21], [25]. Graphconvolution models provide another way to describe relations among signal features [27]. Interpretable models, recurrent networks, and few-shot learning have also been studied to improve decision transparency, recognition speed, and adaptation ability [29], [31], [38]. Most existing methods still use closed-set classification, in which each observation receives one class or every predefined compound type is treated as an independent class. This setting is effective when training covers all test classes, but it does not directly identify the active jamming types. As the primitive vocabulary and component count increase, the number of compound classes and the demand for measured training records also increase.

## C. Compound Interference Recognition and Physical Data

Compound GNSS interference recognition has received increasing attention. Object-detection methods have been used to locate several interference patterns in one time-frequency image [37], and dedicated convolutional networks have been designed for compound classification [26]. Time-frequency features have also been combined with power-spectrum features to improve recognition of overlapping signals [22]. Highresolution interference sensing has been coupled with neural networks for multiple-interference processing [28]. These studies confirm that mixed observations contain useful component information, although their training protocols commonly depend on a fixed compound list or measured compound records.

Physical data collection creates an additional constraint. A controlled single-jammer capture is relatively straightforward to label, whereas collecting a reproducible mixture requires multiple sources, specified power relations, repeated captures, and checks of receiver behavior. The RF hardware can introduce channel-dependent responses, combiner loss, gain changes, and nonlinear effects that numerical signal addition does not reproduce completely. Measured GNSS interference, low-resource recognition, and few-shot adaptation have been studied in recent work [25], [30]–[32].

## D. Compositional Generalization and Structured Set Prediction

Multi-label learning assigns one binary output to each primitive and directly identifies the active jamming types in a mixed record [42]. It also allows one primitive representation to be reused across several compositions. Meng et al. used complexvalued multi-label learning for simulated radar compound jamming [40]. Xiao et al. studied simulated radar compositions excluded from training and combined multi-label classification with image reconstruction and extreme-value modeling for unknown-primitive detection [41]. These studies establish the value of component-wise outputs for compound signals.

Query-based decoders assign a learned query to each label and collect label-related evidence from a shared feature map. Attention provides the basic operation [62], while Query2Label and ML-Decoder show how label queries support multi-label (ML) classification [45], [50]. The asymmetric loss (ASL) reduces the influence of easy negative labels [51]. Independent binary decisions, however, do not constrain the number of selected components or enforce an admissible combination.

The present work brings these ideas together for GNSS jamming recognition when measured mixtures are excluded from gradient optimization. It combines online complex-IQ composition with separate primitive-evidence and cardinality branches and a decoder that scores complete valid sets. Independently acquired conducted RF mixtures assess whether the resulting recognizer transfers to combinations omitted from training and model development. This setting concerns new combinations of known primitives; unknown-primitive rejection is outside its scope.

![](images/73133c9169c1c2a1003c7dc96e7380f36ad1bc055476eb170cc71cbf8677f1cd.jpg)  
Fig. 1. Conceptual GNSS jamming scenario motivating the received-signal model.

## III. SYSTEM MODEL

## A. Received-Signal Formulation

Fig. 1 depicts a conceptual reception scenario in which GNSS and jamming signals arrive at a receiver. Its analog front end observes their superposition together with noise. After downconversion and sampling at frequency $F _ { s } ,$ one signal snapshot, termed a record, contains N complex baseband samples. The model applies to reception through an antenna as well as conducted injection; the RF combiner used for the experiments is a particular implementation described in Section V. The sampled signal and its aggregate JNR are

$$
x [ n ] = s _ { \mathrm { G N S S } } [ n ] + \sum _ { k \in { \cal K } } \sqrt { P _ { k } } j _ { k } [ n ] + w [ n ] , \qquad 0 \leq n < N ,\tag{1a}
$$

$$
\begin{array} { r } { \mathrm { J N R } = 1 0 \log _ { 1 0 } \frac { \frac { 1 } { N } \sum _ { n = 0 } ^ { N - 1 } \left| \sum _ { k \in \mathcal { K } } \sqrt { P _ { k } } j _ { k } [ n ] \right| ^ { 2 } } { \sigma _ { w } ^ { 2 } } . } \end{array}\tag{1b}
$$

Here, $s _ { \mathrm { G N S S } } [ n ]$ is the received GNSS signal, K is the active set of jamming types, $j _ { k } [ n ]$ is the unit-power waveform of type k, and $P _ { k }$ is its received component power. The term w[n] represents thermal and receiver noise with average power $\sigma _ { w } ^ { 2 }$ . These definitions do not depend on whether the signals propagate over the air or pass through a conducted path. At the precorrelation stage, the spread-spectrum GNSS signal is typically below the noise floor, so the classifier primarily observes jamming time-frequency patterns in a noise-like background [3], [6]. For the reported implementation, $F _ { s } = 2 0$ MHz and each record spans 1 ms, giving $N = 2 0 { , } 0 0 0$ . The evaluation covers aggregate JNR values from −20 to 15 dB in 5-dB increments.

## B. Primitive Waveforms

The primitive vocabulary contains five jamming waveform types. Each realized waveform is normalized to unit average power over the record before relative component weighting. Single-tone jamming (STJ) forms one persistent narrowband ridge and is a canonical continuous-wave threat [33], [34]. Multitone jamming (MTJ) occupies several narrow frequency locations and can defeat a single-notch response [36], [39].

Linear frequency-modulated jamming (LFMJ) produces a chirp trajectory that motivates specialized time-frequency and fractional-domain processing [8], [13]. Pulsed-tone jamming (PTJ) gates a narrowband carrier and creates intermittent structures, while partial-band noise jamming (PBNJ) injects shaped noise into only part of the receiver bandwidth [12], [35]. Their complex envelopes, before this per-record normalization, are

$$
j _ { \mathrm { S T J } } [ n ] = \exp \{ \mathrm { j } \left( 2 \pi f _ { c } t _ { n } + \phi \right) \} ,\tag{2a}
$$

$$
j _ { \mathrm { M T J } } [ n ] = \frac { 1 } { \sqrt { M } } \sum _ { m = 1 } ^ { M } \exp \{ \mathrm { j } \left( 2 \pi f _ { m } t _ { n } + \phi _ { m } \right) \} ,\tag{2b}
$$

$$
j _ { \mathrm { L F M J } } [ n ] = \exp \left\{ \mathrm { j } 2 \pi \left( f _ { 0 } t _ { n } + \frac { \mu } { 2 } t _ { n } ^ { 2 } \right) \right\} ,\tag{2c}
$$

$$
j _ { \mathrm { P T J } } [ n ] = b [ n ] \exp \{ \mathrm { j } \left( 2 \pi f _ { p } t _ { n } + \phi _ { p } \right) \} ,\tag{2d}
$$

$$
j _ { \mathrm { P B N J } } [ n ] = \left( \nu * h _ { \mathrm { L P } } \right) [ n ] \exp ( \mathrm { j } 2 \pi f _ { b } t _ { n } ) .\tag{2e}
$$

The discrete time is $t _ { n } = n / F _ { s }$ . The parameters $f _ { c }$ and $\phi$ denote the STJ carrier frequency and phase, while $M , f _ { m }$ , and $\phi _ { m }$ describe the MTJ tones. The LFMJ parameters $f _ { 0 }$ and $\mu$ are the starting frequency and chirp rate. The PTJ pulse mask $b [ n ] \in \{ 0 , 1 \}$ is controlled by the repetition interval, pulse width, and optional timing jitter. For PBNJ, $\nu [ n ]$ is circular complex Gaussian noise, $h _ { \mathrm { L P } } [ n ]$ is a low-pass shaping filter, and $f _ { b }$ is the translated center frequency.

## C. Component-Set Representation and Mixed-Recognition Task

For a record with active set $\kappa ,$ the target is a binary membership vector ${ \pmb y } \in \{ 0 , 1 \} ^ { C }$ , where $y _ { k } = \mathbb { I } ( k \in \mathcal { K } )$ . This is a multi-hot vector: an entry equals one when its primitive is present and zero otherwise, and several entries may equal one simultaneously. The cardinality is $\textstyle K = \sum _ { k = 1 } ^ { C } y _ { k }$ . Here, $C = 5$ is the number of available types, whereas K is the number active in one record. Thus, the output identifies jamming types, not individual transmitters or separated waveforms. The measured singleton bank contains five classes, and the reported recognition task contains two-component and three-component mixtures. Simultaneous STJ and MTJ activation is excluded because their narrowband patterns can be ambiguous in the selected STFT representation, as in the compound-jamming setting of [26]. This is a task restriction, not a claim that the two signals cannot coexist physically.

Let $\mathcal { A } _ { 1 }$ denote the singleton masks and $\mathcal { A } _ { \mathrm { m i x } }$ denote the valid mixed masks with $K \in \{ 2 , 3 \}$ . Under the adopted restriction, $| \mathcal { A } _ { 1 } | = 5$ and $\left. \mathcal { A } _ { \mathrm { m i x } } \right. = 1 6$ . Singleton masks support primitive-evidence learning and an auxiliary singleton-versusmixture task during training. Final inference is restricted to $\mathcal { A } _ { \mathrm { m i x } }$ , so the model never returns a singleton in the reported mixed-interference test.

## D. Training-Listed and Held-Out Composition Split

Let $\mathcal { A } _ { \mathrm { l i s t } } \subset \mathcal { A } _ { \mathrm { m i x } }$ contain the compositions admitted to auxiliary training, and let $A _ { \mathrm { h o l d } } = \mathcal { A } _ { \mathrm { m i x } } \ : \backslash \ : A _ { \mathrm { l i s t } }$ . The reported experiment uses ten training-listed and six held-out compositions, detailed in Table IV. Every primitive in the held-out partition appears in the singleton training bank, but each complete held-out set is excluded from auxiliary training and model development. The 10/6 partition is an experimental choice, not a requirement of the network architecture. Consequently, the results demonstrate transfer on this particular partition of known jamming types; they do not establish invariance to the choice or number of held-out compositions. Unknown primitive types remain outside the task.

## E. Singleton-to-Mixture Physical-Data Protocol

The singleton-to-mixture data-use protocol separates three stages. First, gradient optimization uses measured singleton records together with auxiliary mixtures generated from their associated clean IQ; the auxiliary masks must belong to $\mathcal { A } _ { \mathrm { l i s t } }$ . Second, a separate measured development set from $\mathcal { A } _ { \mathrm { l i s t } }$ supports checkpoint selection and decoder calibration. Third, after all model and decoder settings are fixed, independently acquired mixed test records from both $\mathcal { A } _ { \mathrm { l i s t } }$ and $A _ { \mathrm { h o l d } }$ are evaluated. Held-out compositions are excluded from early stopping, hyperparameter and threshold selection, and calibration as well as gradient training. Table I shows which sample source can enter each stage.

Thus, “singleton-only physical training” refers specifically to the measured records used for gradient updates. It does not mean that optimization uses only singleton labels, or that no measured mixtures are available for model development. Each optimization batch includes measured singletons and generated mixed examples, whereas the final recognition task is restricted to $K \in \{ 2 , 3 \}$

## IV. SPOC-NET METHODOLOGY

SPOC-Net comprises a training-time signal composer and an image-based recognition network. This section presents the complete processing chain in that order: STFT representation, online generation of labeled mixtures, shared feature extraction, primitive and cardinality estimation, structured decoding, and the learning objective. Fig. 3 summarizes the recognition architecture. During inference, only STFT preprocessing and the trained recognition network are used; the IQ composer is not run.

## A. Time-Frequency Representation and Jamming Patterns

SPOC-Net uses a grayscale STFT image. For a complex record $x [ n ]$ , the transform and compressed logarithmic magnitude are

$$
Z [ m , \ell ] = \sum _ { n } x [ n ] g [ n - m R ] \exp \left( - \mathrm { j } \frac { 2 \pi \ell n } { N _ { f } } \right) ,\tag{3a}
$$

$$
D [ m , \ell ] = 2 0 \log _ { 1 0 } \left( \frac { | Z [ m , \ell ] | ^ { \gamma } } { \operatorname* { m a x } _ { m , \ell } | Z [ m , \ell ] | ^ { \gamma } } + \epsilon \right) .\tag{3b}
$$

Here, $g [ n ]$ is a periodic Hann window, R is the hop length, $N _ { f }$ is the fast Fourier transform (FFT) size, γ controls contrast, and ϵ prevents a logarithmic singularity. At the sampling frequency $F _ { s } ~ = ~ 2 0 $ MHz specified in Section III, the 128- sample window spans $6 . 4 \mu \mathrm { s }$ , the 118-sample overlap spans $5 . 9 \mu \mathrm { s } ,$ and the 10-sample hop spans $0 . 5 \mu \mathrm { s }$ . The transform uses $N _ { f } = 4 0 9 6$ . The Hann window and high overlap provide smooth time-frequency descriptions [60], [61]; zero-padding to the FFT size samples the spectrum on a finer frequency grid but does not increase the resolving power of the 128- sample window. With $\gamma = 0 . 9$ , the map is clipped to [−35, 0] dB, mapped to [0, 1], flipped vertically to preserve frequency orientation, resized bilinearly to $2 2 4 \times 2 2 4$ , and stored as a single-channel image. The stored image is subsequently normalized to [−1, 1] for network input.

![](images/8961d9544963610c9620773f8acb808ddb6e084e2a35908c7ffee4280a9afe8e.jpg)  
Fig. 2. Representative time-frequency representations of the five primitive jamming signals and their two-component and three-component compositions

TABLE I  
ROLES OF PHYSICAL, AUXILIARY, DEVELOPMENT, AND TEST SAMPLES
<table><tr><td>Sample source</td><td>Gradient optimization</td><td>Model selection</td><td>Decoder calibration</td><td>Final test</td></tr><tr><td>Measured singleton training records</td><td>Yes</td><td>No</td><td>No</td><td>No</td></tr><tr><td>Online mixed auxiliaries from  $\mathcal { A } _ { \mathrm { l i s t } }$ </td><td>Yes</td><td>No</td><td>No</td><td>No</td></tr><tr><td>Measured development records from</td><td>No</td><td>Yes</td><td>Yes</td><td>No</td></tr><tr><td> $\mathcal { A } _ { \mathrm { l i s t } }$  Measured final-test records from  $\mathcal { A } _ { \mathrm { l i s t } }$ </td><td>No</td><td>No</td><td>No</td><td>Yes</td></tr><tr><td>Measured final-test records from  $A _ { \mathrm { h o l d } }$ </td><td>No</td><td>No</td><td>No</td><td>Yes</td></tr></table>

Fig. 2 shows examples of the generated primitive signals and interference combinations whose STFT images are supplied to the recognizer. STJ produces a persistent horizontal line, MTJ produces several parallel lines, and LFMJ produces sloped or repeated chirp tracks. PTJ gives intermittent narrowband structures, whereas PBNJ appears as a band of stochastic texture. These are descriptions of the input signal patterns, not additional class definitions. In a mixture, unequal powers and spectral overlap can make one pattern much less visible than the others, motivating separate estimation of component identity and component count.

## B. Online Clean-IQ Composition

Let $s _ { i } [ n ]$ denote the clean jammer IQ sequence associated with a measured singleton of type i in the training bank. The composer uses these stored clean sequences rather than adding noisy receiver records. For a sampled two-component or threecomponent set in $\mathcal { A } _ { \mathrm { l i s t } }$ , each sequence is centered and rootmean-square (RMS) normalized. Relative powers are drawn in decibels and centered across the selected components, yielding values $\rho _ { i }$ and amplitude factors $a _ { i } = 1 0 ^ { \rho _ { i } / 2 0 }$ . A fresh circular complex Gaussian background v[n] is also centered and normalized. Denoting sample means by overbars, the resulting auxiliary record is

$$
\widetilde { s } _ { i } [ n ] = \frac { s _ { i } [ n ] - \overline { { s } } _ { i } } { \operatorname { r m s } ( s _ { i } - \overline { { s } } _ { i } ) } ,\tag{4a}
$$

$$
u [ n ] = \sum _ { i \in \mathcal { K } } a _ { i } \widetilde { s } _ { i } [ n ] ,\tag{4b}
$$

$$
x _ { \mathrm { a u x } } [ n ] = { \sqrt { 1 0 ^ { \Gamma / 1 0 } } } { \frac { u [ n ] } { \operatorname { r m s } ( u ) } } + { \frac { v [ n ] - { \overline { { v } } } } { \operatorname { r m s } ( v - { \overline { { v } } } ) } } .\tag{4c}
$$

The target aggregate JNR is Γ. Relative component powers are sampled from [−6, 6] dB before centering, and Γ is drawn from $\{ - 2 0 , - 1 5 , - 1 0 , - 5 , 0 , 5 , 1 0 , 1 5 \}$ dB. Normalizing u[n] sets the aggregate jammer power independently of the selected component count, while the normalized background has unit power. Each newly requested auxiliary example therefore has a known component mask and a controlled aggregate JNR. Composition precedes the STFT so that complex phases and overlapping spectra are combined before magnitude compression, rather than approximated by adding spectrogram images. The generated record then follows the same preprocessing as a measured record. This is a linear-superposition training model with a noise background, not a reproduction of every RF frontend effect.

The composer also forms paired examples for cardinality learning: a two-component record and a three-component extension share the source sequences of their common primitives, the target aggregate JNR, and the background realization.

![](images/e9aa7d00ad61f068a02590e7b20ccff90f4b4c0074bea332754d6bc7fe064589.jpg)  
Fig. 3. Architecture of SPOC-Net. A shared multi-resolution spectro-temporal encoder feeds a primitive-evidence branch and a detached high-resolution cardinality branch. The singleton-versus-mixture output provides auxiliary training supervision. Final inference is restricted to the 16 valid two-component and three-component masks and uses the conditional two-versus-three cardinality probability.

Their component masks differ by one added primitive, with both masks restricted to $\mathcal { A } _ { \mathrm { l i s t } }$ . Because the aggregate jammer signal is renormalized in each example, the received amplitudes of the shared components need not remain identical. The pair therefore supervises the effect of an additional component under fixed aggregate JNR, not an unconstrained increase in total jammer power.

C. Overall Architecture and Shared Spectro-Temporal Encoder

Fig. 3 details the network that processes measured and composed examples through the same image pipeline. Let $X \in [ - 1 , 1 ] ^ { 1 \times H \times W }$ be the normalized grayscale STFT image. The network produces primitive logits $z \in \mathbb { R } ^ { C }$ , an auxiliary singleton-versus-mixture logit, and a conditional two-versusthree cardinality logit. A logit is an unnormalized score that is converted to a probability by a sigmoid function. The training candidate set includes singletons and training-listed mixtures; the final decoder instead considers all protocol-valid mixed sets. These candidate sets and the final decision are

$$
\mathcal { A } _ { \mathrm { t r } } = \mathcal { A } _ { 1 } \cup \mathcal { A } _ { \mathrm { l i s t } } ,\tag{5a}
$$

$$
\mathcal { A } _ { \mathrm { { e v a l } } } = \mathcal { A } _ { \mathrm { { m i x } } } ,\tag{5b}
$$

$$
{ \widehat { K } } = \arg \operatorname* { m a x } _ { A \in A _ { \mathrm { e v a l } } } S _ { \mathrm { m i x } } ( A \mid X ) .\tag{5c}
$$

The training universe contains the five singleton masks and the 10 masks admitted to online composition. The evaluation universe contains only the 16 valid two-component and threecomponent masks, so no singleton candidate is permitted in the reported test. Held-out compositions use the same five primitive logits and require no new output neuron.

The shared encoder augments the STFT with fixed frequency and time coordinates. For pixel indices $0 \leq i < H$ and $0 \leq j < W$ , the coordinate maps and augmented input are

$$
C _ { f } ( i , j ) = \frac { 2 i } { H - 1 } - 1 ,\tag{6a}
$$

$$
C _ { t } ( i , j ) = \frac { 2 j } { W - 1 } - 1 ,\tag{6b}
$$

$$
\overline { { { X } } } = \operatorname { C o n c a t } ( X , C _ { f } , C _ { t } ) .\tag{6c}
$$

These maps retain absolute location information, which helps distinguish local structures with similar gradients but different positions or extents. A parallel stem observes the augmented input with several receptive fields. Let R be the kernel set, let $\Phi _ { l } ( \cdot )$ be encoder stage l, and let $D _ { l } ( \cdot )$ be stride-two downsampling. The encoder stages are

$$
F _ { 1 } = \Phi _ { 1 } \biggl ( \mathrm { C o n c a t } \left\{ \mathrm { C o n v } _ { r \times r } ^ { s = 4 } ( \overline { { X } } ) \right\} \biggr ) ,\tag{7a}
$$

$$
F _ { l } = \Phi _ { l } ( D _ { l } ( F _ { l - 1 } ) ) , \qquad l \in \{ 2 , 3 , 4 \} .\tag{7b}
$$

The reported model uses $\begin{array} { l l l } { \mathcal { R } } & { = } & { \{ 3 , 5 , 1 1 \} } \end{array}$ , stage depths (2, 2, 6, 2), and channel dimensions (64, 128, 256, 384). The output scales are $1 / 4 , 1 / 8 , 1 / 1 6$ , and 1/32. The parallel stem preserves narrow ridges, short pulse fragments, and broad noise regions before deeper feature extraction.

Each encoder block uses a $7 \times 7$ depthwise convolution, channel normalization, pointwise expansion, a Gaussian error linear unit, global response normalization, pointwise projection, channel-spatial reweighting, and stochastic depth [52], [53], [57]. Features from all stages are projected to a common dimension d and reduced to the spatial size of $F _ { 4 }$ through adaptive average pooling (AAP). Their fusion and one anisotropic context unit are defined by

$$
U = \mathcal { C } ^ { ( 2 ) } \left( \mathcal { N } \Bigg [ P _ { 4 } ( F _ { 4 } ) + \sum _ { l = 1 } ^ { 3 } \mathrm { A A P } _ { H _ { 4 } \times W _ { 4 } } ( P _ { l } ( F _ { l } ) ) \Bigg ] \right) ,\tag{8a}
$$

$$
\mathcal { C } ( V ) = V + P _ { c } ( D _ { 1 \times \kappa } ( \mathcal { N } ( V ) ) + D _ { \kappa \times 1 } ( \mathcal { N } ( V ) ) ) .\tag{8b}
$$

Here, $P _ { l } ( \cdot )$ is a 1×1 projection, $\mathcal { N } ( \cdot )$ is channel normalization, and $\mathscr { C } ^ { ( 2 ) } ( \cdot )$ applies two context units. The implementation uses $\kappa ~ = ~ 9$ . The anisotropic depthwise kernels collect evidence mainly along time and frequency, which supports tones, pulse transitions, and chirp tracks with fewer parameters than a dense large two-dimensional kernel.

## D. Primitive-Evidence Modeling

The primitive branch uses multi-level tokens together with a global average pooling (GAP) descriptor. Let $R _ { l } = P _ { l } ( F _ { l } )$ for $l \in \{ 1 , 2 , 3 \}$ and let $R _ { 4 } = U$ . The token sequence and global descriptor are

$$
T _ { l } = \operatorname { v e c } ( \operatorname { A A P } _ { r _ { l } \times r _ { l } } ( R _ { l } ) ) + \mathbf { 1 } e _ { l } ^ { \mathsf { T } } ,\tag{9a}
$$

$$
T = { \mathcal { N } } _ { T } ( \operatorname { C o n c a t } ( T _ { 1 } , T _ { 2 } , T _ { 3 } , T _ { 4 } ) ) ,\tag{9b}
$$

$$
\begin{array} { r } { \pmb { g } = \mathrm { G A P } ( U ) . } \end{array}\tag{9c}
$$

The learned vector $e _ { l }$ identifies the feature level, $\mathcal { N } _ { T }$ is token normalization, and GAP produces g. The token grids are $1 4 \times 1 4$ $1 4 \times 1 4$ $7 \times 7 .$ , and $7 \times 7 .$ . Shallow levels preserve local details, while deep levels provide compact semantic information.

One learnable query is assigned to each primitive type. A query-decoder layer applies cross-attention (CA), selfattention (SA), and a feed-forward network (FFN) through

$$
\overline { { Q } } ^ { ( m ) } = Q ^ { ( m - 1 ) } + \mathrm { C A } \left( Q ^ { ( m - 1 ) } , T , T \right) ,\tag{10a}
$$

$$
\widetilde { Q } ^ { ( m ) } = \overline { { Q } } ^ { ( m ) } + \mathrm { S A } \left( \overline { { Q } } ^ { ( m ) } \right) ,\tag{10b}
$$

$$
Q ^ { ( m ) } = \widetilde { Q } ^ { ( m ) } + \mathrm { F F N } \Big ( \widetilde { Q } ^ { ( m ) } \Big ) .\tag{10c}
$$

CA lets each query collect primitive-specific evidence from all feature levels, while SA models relations and competition among primitive types [45], [50], [62]. The implementation uses two decoder layers.

Let $\pmb q _ { k }$ be the final query for primitive k. Query-conditioned evidence, global evidence, and their fusion are

$$
z _ { k } ^ { ( q ) } = h _ { q } ( [ { \pmb q } _ { k } ; { \pmb g } ] ) ,\tag{11a}
$$

$$
{ \pmb z } ^ { ( g ) } = h _ { g } ( { \pmb g } ) ,
$$

$$
z = z ^ { ( q ) } + \alpha _ { g } z ^ { ( g ) } .\tag{11b}
$$

(11c)

The query term emphasizes class-specific local shape, while the global term stabilizes decisions for primitives with strong full-image signatures. Table II gives the value of $\alpha _ { g }$

## E. Hierarchical Cardinality Training and Mixed-Set Decoding

Primitive identity and set size require related but different evidence. Downsampling compresses the spatial grid of an STFT feature map. A weak tone ridge or short pulse fragment may then be averaged with surrounding background, so a feature indicating a third jamming type can become less distinct even when the two dominant types remain recognizable. This motivates estimating cardinality from a higher-resolution image path and the shallow feature $F _ { 1 }$ , rather than relying only on the deepest pooled descriptor. The image path, detached shallow path, and fused evidence are

$$
E _ { x } = \Psi _ { x } ( X ) ,
$$

$$
E _ { s } = P _ { s } ( \mathrm { s g } ( F _ { 1 } ) ) ,\tag{12a}
$$

(12b)

$$
E = \Psi _ { f } ( \operatorname { C o n c a t } ( E _ { x } , \operatorname { R e s i z e } ( E _ { s } ) ) ) .\tag{12c}
$$

The stop-gradient operator sg(·) passes feature values forward but blocks gradients through this connection, preventing this cardinality path from updating the shared encoder. The image path contains a stride-two $5 \times 5$ convolution, two anisotropic context units, and a stride-two $3 \times 3$ convolution. The image path and cardinality predictor remain trainable.

High-resolution evidence is summarized through generalized-mean (GeM), max, and attention pooling. For channel c and spatial feature vector $e _ { j }$ , let $N _ { s }$ be the number of spatial locations in E and $p \ > \ 0$ the GeM exponent. The pooled descriptors are

$$
g _ { \mathrm { g e m } , c } = \left( \frac { 1 } { N _ { s } } \sum _ { j = 1 } ^ { N _ { s } } | E _ { c , j } | ^ { p } \right) ^ { 1 / p } ,\tag{13a}
$$

$$
{ \pmb g } _ { \mathrm { m a x } } = \operatorname* { m a x } _ { j } { \pmb e } _ { j } ,\tag{13b}
$$

$$
a _ { j } = \frac { \exp ( \pmb { w } ^ { \top } \pmb { e } _ { j } ) } { \sum _ { n } \exp ( \pmb { w } ^ { \top } \pmb { e } _ { n } ) } ,\tag{13c}
$$

$$
g _ { \mathrm { a t t } } = \sum _ { j = 1 } ^ { N _ { s } } a _ { j } \mathbf { e } _ { j } ,\tag{13d}
$$

$$
e _ { \mathrm { c a r d } } = [ { g } _ { \mathrm { g e m } } ; { g } _ { \mathrm { m a x } } ; { g } _ { \mathrm { a t t } } ] .\tag{13e}
$$

GeM pooling captures distributed energy, max pooling retains isolated peaks, and attention pooling focuses on informative regions [54]. Their combination helps when a weak third component occupies only a small part of the STFT.

The pooled evidence is combined with the global descriptor and class-order-invariant score summaries from the primitive branch as

$$
\begin{array} { c } { { r _ { \mathrm { c a r d } } = [ e _ { \mathrm { c a r d } } ; \mathrm { s g } ( g ) ; \mathrm { s o r t } ( \mathrm { s g } ( z ) ) ; } } \\ { { { \mathrm { s o r t } } ( \mathrm { s i g m o i d } ( \mathrm { s g } ( z ) ) ) ] . } } \end{array}\tag{14}
$$

The sorted logit and probability profiles describe how many primitives have strong support without tying the set-size decision to a fixed class identity. A shared predictor maps r<sub>card</sub> to an auxiliary logit $u _ { \mathrm { m i x } }$ for $K \geq 2$ and a conditional logit $u _ { \mathrm { 3 | m i x } }$ for $K = 3$ given a mixed observation. With cardinality temperature $T _ { c }$ , the predicted cardinality probabilities are

$$
\pi _ { \mathrm { m i x } } = \mathrm { s i g m o i d } ( u _ { \mathrm { m i x } } / T _ { c } ) ,\tag{15a}
$$

$$
\pi _ { 3 | \mathrm { m i x } } = \mathrm { s i g m o i d } ( u _ { 3 | \mathrm { m i x } } / T _ { c } ) ,\tag{15b}
$$

$$
p _ { 1 } = 1 - \pi _ { \operatorname* { m i x } } ,\tag{15c}
$$

$$
p _ { 2 } = \pi _ { \mathrm { m i x } } \left( 1 - \pi _ { 3 | \mathrm { m i x } } \right) ,\tag{15d}
$$

$$
p _ { 3 } = \pi _ { \mathrm { m i x } } \pi _ { 3 | \mathrm { m i x } } ,\tag{15e}
$$

$$
\overline { { p } } _ { 2 } = 1 - \pi _ { 3 | \operatorname* { m i x } } ,\tag{15f}
$$

$$
\overline { { p } } _ { 3 } = \pi _ { 3 | \mathrm { m i x } } .\tag{15g}
$$

For each input image, $p _ { 1 } , p _ { 2 }$ , and $p _ { 3 }$ are nonnegative predicted probabilities for one, two, and three active types, respectively, and sum to one. These input-dependent probabilities act as cardinality priors when candidate sets are scored. Training uses all three because the training dataset includes singleton and mixed examples. The mixed-only test task uses the conditional probabilities $\overline { { p } } _ { 2 }$ and ${ \overline { { p } } } _ { 3 } .$ which sum to one over the two permitted test cardinalities. Thus, the auxiliary singleton-versusmixture output is not used to rank final mixed candidates.

Thresholding each primitive probability independently does not enforce the task constraints. For example, it can select both STJ and MTJ, select fewer than two or more than three types, or omit a weak third type whose score lies just below the threshold. SPOC-Net instead scores complete admissible candidates using both component evidence and the predicted set size. Let $\pmb { a } \in \{ 0 , 1 \} ^ { C }$ be the membership mask of candidate A. The primitive evidence, training score, mixed-test score, and their ranking equivalence for $A \in \mathcal { A } _ { \operatorname* { m i x } }$ are

$$
\begin{array} { l } { { \displaystyle { \mathcal B } ( A \mid X ) = \sum _ { k = 1 } ^ { C } a _ { k } \log \mathrm { s i g m o i d } \Biggl ( \frac { z _ { k } } { T _ { p } } \Biggr ) } } \\ { { \displaystyle \qquad + \lambda _ { n } \sum _ { k = 1 } ^ { C } ( 1 - a _ { k } ) \log \mathrm { s i g m o i d } \Biggl ( - \frac { z _ { k } } { T _ { p } } \Biggr ) } , } \end{array}\tag{16a}
$$

$$
S _ { \mathrm { t r } } ( A \mid X ) = { \mathcal { B } } ( A \mid X ) + \lambda _ { c } \log p _ { | A | } + { \frac { \beta _ { 3 } } { T _ { c } } } \mathbb { I } ( | A | = 3 ) ,\tag{16b}
$$

$$
S _ { \operatorname* { m i x } } ( A \mid X ) = { \mathcal { B } } ( A \mid X ) + \lambda _ { c } \log { \overline { { p } } } _ { | A | } + { \frac { \beta _ { 3 } } { T _ { c } } } \mathbb { I } ( | A | = 3 ) ,\tag{16c}
$$

$$
S _ { \mathrm { t r } } ( A \mid X ) = S _ { \mathrm { m i x } } ( A \mid X ) + \lambda _ { c } \log \pi _ { \mathrm { m i x } } .\tag{16d}
$$

Here, $T _ { p }$ is the primitive temperature, $\lambda _ { n }$ weights evidence for absent components, $\lambda _ { c }$ weights the cardinality probability, and $\beta _ { 3 }$ adjusts the preference for three-component candidates. The component terms define a weighted Bernoulli log-score, reducing to the standard Bernoulli log-likelihood when $\lambda _ { n } =$ 1. For every mixed candidate, $p _ { | A | } = \pi _ { \operatorname* { m i x } } \overline { { p } } _ { | A | }$ , so the term $\lambda _ { c } \log \pi _ { \operatorname* { m i x } }$ is common to all mixed candidates and cancels in their ranking.

Decoder calibration uses the measured training-listed development records defined in Table I. The candidate masks themselves are fixed by the recognition task and have no learned parameters. Let $\theta = ( T _ { c } , \lambda _ { c } , \beta _ { 3 } )$ and let Θ be the calibration grid. To avoid improving three-component recognition at an unacceptable cost to two-component recognition, calibration imposes a two-component exact-set accuracy floor $A _ { 2 } ^ { \mathrm { { f l o o r } } }$ with tolerance ϵ. The feasible set and selection rule are

$$
\Theta _ { \mathrm { f e a s } } = \left\{ \theta \in \Theta : \operatorname { A c c } _ { 2 } ^ { \mathrm { d e v } } ( \theta ) \geq A _ { 2 } ^ { \mathrm { f l o o r } } - \epsilon \right\} ,\tag{17a}
$$

$$
\theta ^ { \star } = \operatorname* { l e x m a x } _ { \theta \in \Theta _ { \mathrm { f e a s } } } \left[ \operatorname { A c c } _ { 3 } ^ { \mathrm { d e v } } ( \theta ) , \operatorname { B a l A c c } _ { 2 , 3 } ^ { \mathrm { d e v } } ( \theta ) , \operatorname { A c c } _ { \mathrm { a l l } } ^ { \mathrm { d e v } } ( \theta ) \right] .\tag{17b}
$$

Here, $\operatorname { A c c } _ { r } ^ { \mathrm { { d e v } } }$ denotes exact-set accuracy for cardinality $r ,$ Bal $\mathrm { A c c } _ { 2 . 3 } ^ { \mathrm { d e v } }$ averages the two cardinality-specific accuracies, and $\mathrm { A c c } _ { \mathrm { a l l } } ^ { \mathrm { d e v } }$ pools the development records. The lexicographic rule first maximizes three-component accuracy, then uses balanced accuracy and overall accuracy to break ties. The checkpoint and selected decoder settings are frozen before final testing.

## F. Learning Objective and Implementation Details

The learning objective covers primitive membership, validset discrimination, auxiliary singleton-versus-mixture separation, conditional two-versus-three recognition, and the response to adding a third component. Let $p _ { i k } = \mathrm { s i g m o i d } ( z _ { i k } )$ for sample i and primitive k. The shifted negative probability, one-label ASL term, and cardinality-balanced classification loss are

$$
\overline { { { p } } } _ { i k } ^ { - } = \operatorname* { m i n } ( 1 , 1 - p _ { i k } + m _ { a } ) ,\tag{18a}
$$

$$
\ell _ { i k } ^ { \mathrm { a s y } } = - y _ { i k } ( 1 - p _ { i k } ) ^ { \gamma _ { + } } \log p _ { i k }
$$

$$
- \left( 1 - y _ { i k } \right) \left( 1 - \overline { { p } } _ { i k } ^ { - } \right) ^ { \gamma _ { - } } \log \overline { { p } } _ { i k } ^ { - } ,\tag{18b}
$$

$$
\mathcal { L } _ { \mathrm { c l s } } = \frac { 1 } { | \mathcal { R } _ { B } | } \sum _ { r \in \mathcal { R } _ { B } } \frac { \sum _ { i \in \mathcal { B } _ { r } } \sum _ { k = 1 } ^ { C } \ell _ { i k } ^ { \mathrm { a s y } } } { C | \mathcal { B } _ { r } | } .\tag{18c}
$$

Let B denote the batch size. Here, $B _ { r } = \{ i : | \mathcal { K } _ { i } | = r \}$ and $\mathcal { R } _ { B } = \left\{ r : | B _ { r } | > 0 \right\}$ . ASL reduces the influence of easy negative labels while retaining hard negative evidence [51], and the outer average limits imbalance among set sizes.

The candidate-set loss uses $S _ { \mathrm { t r } }$ and the training universe $\mathcal { A } _ { \mathrm { t r } }$ . For ground-truth set $A _ { i }$ , it is

$$
\mathcal { L } _ { \mathrm { s e t } } = - \frac { 1 } { B } \sum _ { i = 1 } ^ { B } \log \frac { \exp { S _ { \mathrm { t r } } ( A _ { i } \mid X _ { i } ) } } { \sum _ { A \in \mathcal { A } _ { \mathrm { t r } } } \exp { S _ { \mathrm { t r } } ( A \mid X _ { i } ) } } .\tag{19}
$$

Measured singleton samples and online-composed mixed sam ples compete against the five singleton masks and the 10 training-listed mixed masks. Held-out compositions are absent from this loss.

Let $\ell _ { \mathrm { b c e } } ( u , t )$ denote binary cross-entropy (BCE) with logits, and let $B _ { \mathrm { m i x } } = \{ i : | K _ { i } | \geq 2 \}$ . The auxiliary, conditional, and combined cardinality losses are

$$
\mathcal { L } _ { \mathrm { m i x } } = \frac { 1 } { B } \sum _ { i = 1 } ^ { B } \ell _ { \mathrm { b c e } } \big ( u _ { \mathrm { m i x } , i } , \mathbb { I } ( | \mathcal { K } _ { i } | \geq 2 ) \big ) ,\tag{20a}
$$

$$
\mathcal { L } _ { \mathrm { 3 | m i x } } = \frac { 1 } { | \mathcal { B } _ { \mathrm { m i x } } | } \sum _ { i \in \mathcal { B } _ { \mathrm { m i x } } } w _ { | \mathcal { K } _ { i } | } \ell _ { \mathrm { b c e } } \big ( u _ { 3 | \mathrm { m i x } , i } , \mathbb { I } ( | \mathcal { K } _ { i } | = 3 ) \big ) ,\tag{20b}
$$

$$
\mathcal { L } _ { \mathrm { c a r d } } = \frac { 1 } { 2 } \left( \mathcal { L } _ { \mathrm { m i x } } + \mathcal { L } _ { 3 | \mathrm { m i x } } \right) .\tag{20c}
$$

The auxiliary loss separates singleton and mixed training samples, while the conditional loss directly controls the twoversus-three decision used during final inference. The decoder uses only $\pi _ { 3 | \mathrm { m i x } }$ through $\overline { { p } } _ { 2 }$ and ${ \overline { { p } } } _ { 3 }$

TABLE II  
ARCHITECTURE AND DECODER CONFIGURATION
<table><tr><td>Setting</td><td>Value</td></tr><tr><td>Network input</td><td>224 × 224 grayscale STFT only</td></tr><tr><td>Coordinate augmentation</td><td>Time and frequency maps</td></tr><tr><td>Stem kernels and stride</td><td>3, 5, 11 and 4</td></tr><tr><td>Stage depths</td><td>(2, 2, 6, 2)</td></tr><tr><td>Stage channels</td><td>(64, 128, 256, 384)</td></tr><tr><td>Decoder dimension d</td><td>384</td></tr><tr><td>Token grids</td><td>14, 14, 7, 7</td></tr><tr><td>Query layers and heads</td><td>2 and 6</td></tr><tr><td>Primitive context units</td><td>2</td></tr><tr><td>Cardinality context units</td><td>2 before fusion and 1 after fusion</td></tr><tr><td>Global-logit coefficient  $\alpha _ { g }$ </td><td>0.35</td></tr><tr><td>Cardinality evidence channels</td><td>64</td></tr><tr><td>Cardinality hidden dimension</td><td>256</td></tr><tr><td>JNR or metadata side input</td><td>None</td></tr><tr><td>Auxiliary cardinality output</td><td>Singleton versus mixture</td></tr><tr><td>Final cardinality output</td><td>Two versus three components</td></tr><tr><td>Final candidate universe</td><td>16 mixed masks</td></tr><tr><td>Dropout and stochastic depth</td><td>0.15 and 0.10</td></tr><tr><td>Trainable parameters</td><td>14.05 million</td></tr></table>

For the paired examples defined in Section IV-B, let $u _ { 3 | \mathrm { m i x } , p } ^ { ( 2 ) }$ and $u _ { 3 | \mathrm { m i x } , p } ^ { ( 3 ) }$ be the conditional three-component logits of the two-component and three-component records, respectively. With $N _ { p }$ pairs and margin $m _ { p }$ , the monotonic pair loss and complete objective are

$$
\mathcal { L } _ { \mathrm { p a i r } } = \frac { 1 } { N _ { p } } \sum _ { p = 1 } ^ { N _ { p } } \mathrm { m a x } \Big ( 0 , m _ { p } - u _ { 3 | \mathrm { m i x } , p } ^ { ( 3 ) } + u _ { 3 | \mathrm { m i x } , p } ^ { ( 2 ) } \Big ) ,\tag{21a}
$$

$$
\begin{array} { r } { \mathcal { L } = \lambda _ { \mathrm { c l s } } \mathcal { L } _ { \mathrm { c l s } } + \lambda _ { \mathrm { s e t } } \mathcal { L } _ { \mathrm { s e t } } + \lambda _ { \mathrm { c a r d } } \mathcal { L } _ { \mathrm { c a r d } } + \lambda _ { \mathrm { p a i r } } \mathcal { L } _ { \mathrm { p a i r } } . } \end{array}\tag{21b}
$$

The paired term encourages stronger three-component evidence for the extension than for its two-component counterpart. Each batch is approximately balanced among one-, two-, and three-component examples, with measured singletons and online-composed mixtures used according to Table I.

AdamW is used for optimization [58]. Dropout and stochastic depth limit overfitting [63], and the learning rate follows cosine annealing with warm restarts [59]. Mixed-precision training, gradient clipping, exponential moving average (EMA) weights, and light time-frequency perturbations are also used. The default model has about 14.05 million trainable parameters. Tables II and III list the architecture and training settings.

## V. EXPERIMENTAL SETUP

## A. Computing Environment and Conducted RF Acquisition Platform

Training and inference are carried out on a workstation with an Intel Core i9-12900KF processor, 64 GB of system memory, and an NVIDIA GeForce RTX 5060 Ti with 16 GB of memory. The implementation uses Python 3.11 and PyTorch 2.x with NVIDIA Compute Unified Device Architecture (CUDA) and the CUDA Deep Neural Network library (cuDNN) for acceleration.

The dataset is collected with the conducted RF acquisition platform shown in Fig. 4. An ANTF004-SMAJ GNSS antenna receives live satellite signals and feeds one input of a fourinput RF combiner through a coaxial cable. One Universal

Software Radio Peripheral (USRP) X410 is configured with up to three independently controlled transmit channels, denoted TX1–TX3, to generate the selected jamming signals. Each transmit channel is connected conductively to a separate combiner input. The combiner output is delivered by coaxial cable to the receive channel of a second USRP X410, which records the complex IQ sequence for subsequent signal processing. Each acquired record therefore contains the live GNSS signal, receiver noise, and the selected one-, two-, or three-component jamming set.

Each processed signal snapshot is one 1-ms record sampled at $F _ { s } = 2 0 $ MHz, containing $N = 2 0 { , } 0 0 0$ complex samples as defined in Section III. The component set, cardinality, JNR label, and IQ sequence are stored for every record. The jammer transmit channels are activated for the target set, and their output levels are controlled independently to achieve the specified component-power relations and aggregate JNR. Singleton acquisition provides the measured receiver records and associated clean jammer IQ used by the composer. Measured mixtures are produced by simultaneous conducted RF injection and recorded independently at the X410 acquisition receiver. Their uses in development, comparison-model training, and testing are distinguished below. This use of physical measurements is related to prior work on measured GNSS interference and practical recognition [22], [25], [31], [32].

## B. Evaluation Protocol and Composition Partitions

The final test set contains $N _ { \mathrm { r e c } } ~ = ~ 1 4 { , } 2 2 0$ independently acquired 1-ms conducted mixed records, covering the 16 valid compositions and JNR values from −20 to 15 dB in 5-dB steps. Here and in the result tables, $N _ { \mathrm { r e c } }$ counts records in the indicated subset; N remains the number of IQ samples per record. The training-listed test subset contains 8,887 records from the ten compositions in $\mathcal { A } _ { \mathrm { l i s t } }$ , and the held-out test subset contains 5,333 records from the six compositions in $A _ { \mathrm { h o l d } }$

The measured development set contains only $\boldsymbol { \mathcal { A } } _ { \mathrm { l i s t } }$ compositions and is disjoint from both final-test subsets. It supports checkpoint selection, early stopping when used, hyperparameter selection, and decoder calibration. No waveform, image, label, or aggregate result from $A _ { \mathrm { h o l d } }$ is used for these decisions. After development, each final-test partition is evaluated with the same fixed network and decoder, using the view-averaging settings reported for that model. The same composition-exclusion rule applies to the comparison models; their training and output protocols are specified in Section V-C.

## C. Comparison Models and Component-Set Output Protocol

We compare SPOC-Net with six reference architectures for mixed-jamming component-set recognition: AlexNet [56], ResNet18 [55], Feature-ResNet18, ACSNet [26], MSFF-KAN, and the Temporal-Spatial Feature Aggregation Network (TS-FANet) [21], [24]. Feature-ResNet18 is a ResNet18-based reference augmented with signal statistics, as specified below. The reference architectures are adapted to a common fivelabel primitive-set output and are denoted by the suffix “-ML.” This makes their output space comparable with SPOC-Net and allows evaluation on new combinations of known labels, unlike a fixed-list compound-class softmax head.

TABLE III  
TRAINING, AUXILIARY COMPOSITION, LOSS, PROTOCOL, AND CALIBRATION SETTINGS
<table><tr><td>Category</td><td>Setting</td><td>Category</td><td>Setting</td></tr><tr><td>Optimization</td><td>120 epochs; batch size 32</td><td>Optimizer</td><td>AdamW</td></tr><tr><td>Learning rate</td><td> $2 . 5 \times \mathrm { \bar { 1 0 ^ { - 4 } } ~ t o ~ } 1 0 ^ { - 6 }$ </td><td>Weight decay</td><td> $7 \times 1 0 ^ { - 4 }$ </td></tr><tr><td>Gradient handling</td><td>Norm clipping at 1.0</td><td>EMA decay</td><td>0.999</td></tr><tr><td>Physical gradient data</td><td>Measured singleton records only</td><td>Auxiliary source</td><td>Clean-IQ singleton bank only</td></tr><tr><td>Training mixed masks</td><td> $\mathcal { A } _ { \mathrm { l i s t } }$  only</td><td>Training candidate uni- verse</td><td> $\mathcal { A } _ { 1 } \cup \mathcal { A } _ { \mathrm { l i s t } }$ </td></tr><tr><td>Development composi- tions</td><td> $\mathcal { A } _ { \mathrm { l i s t } }$  only</td><td>Held-out development None records</td><td></td></tr><tr><td>Final candidate cardi- nalities</td><td> $K \in \{ 2 , 3 \}$ </td><td>Singleton-versus- mixture output</td><td>Auxiliary training only</td></tr><tr><td>Batch cardinalities</td><td>Approximately balanced one, two, and three com- ponents</td><td>Paired auxiliaries</td><td>4 pairs per batch</td></tr><tr><td>Relative component power</td><td>[-6, 6] dB</td><td>Auxiliary JNR grid</td><td>-20:5:15 dB</td></tr><tr><td>Low-JNR sampling</td><td>1.3× for JNR ≤ −10 dB</td><td>Very-low-JNR</td><td>Additional 1.8× for JNR ≤ −20 dB</td></tr><tr><td>Loss weights</td><td>1.00, 0.25, 0.50, 0.15</td><td>sampling  $\mathrm { A S L } \left( \gamma _ { + } , \gamma _ { - } \right)$ </td><td>(0,4)</td></tr><tr><td>ASL negative shift ma</td><td>0.05</td><td>Pair margin mp</td><td>0.50</td></tr><tr><td>Conditional costs</td><td>(1.50,1.25)</td><td>View averaging</td><td>2 for calibration and selection; 4 for test</td></tr><tr><td>(w2, w3) Calibration records</td><td>Measured  $\mathcal { A } _ { \mathrm { l i s t } }$  development set</td><td>Calibration score</td><td> $ { S } _ { \mathrm { m i x } }$ </td></tr><tr><td>Calibration  $T _ { c }$ </td><td>{0.70, 0.85, 1.00, 1.20, 1.40}</td><td>Calibration  $\lambda _ { c }$ </td><td></td></tr><tr><td>Calibration  $\beta _ { 3 }$ </td><td> $- 1 . 0 0 { : 0 } . 2 5 { : 1 } . 2 5$ </td><td>Two-component tolerance €</td><td>{0.50, 0.75, 1.00, 1.25, 1.50, 2.00} floor 0.005</td></tr></table>

![](images/897872bcacaf55aa0077bd3d82b5a99a8cd5f6b08cc96f25c8a38f9b30957587.jpg)  
(a)

![](images/36e4c4f2a094a2b196b7bfe5db280e1c3ebac3b3bc6c909bfd85abcc312570e5.jpg)  
(b)  
Fig. 4. Conducted RF acquisition platform. Panel (a) shows the hardware implementation with two USRP X410 units, a GNSS antenna, and a four-input RF combiner. Panel (b) shows the signal flow: the live GNSS antenna signal and up to three jammer transmit channels are combined in the RF domain, recorded by the USRP receive channel, and passed to the IQ-processing pipeline.

The adapted final layer contains one logit for each primitive in the order STJ, MTJ, LFMJ, PTJ, and PBNJ. The five logits are trained with ASL and converted to sigmoid probabilities. A composition uses a multi-hot primitive mask, so STJ+PTJ is represented by [1, 0, 0, 1, 0]. This shared output form allows every architecture to return a held-out composition from known primitive evidence. A strict 10-class softmax head cannot predict an unlisted composition and is not used for the component-set comparisons reported here.

Index the six adapted reference architectures above by $b \in \{ 1 , \ldots , 6 \}$ . For architecture $b ,$ let $z _ { b , k }$ be the logit for primitive k, $T _ { b }$ its calibration temperature, $\delta _ { b , k }$ a componentspecific bias, and a the membership mask of candidate A. Each reference model uses the following calibrated probability, candidate score, and final decision:

(22a)

$$
\begin{array} { l } { \displaystyle p _ { b , k } = \mathrm { s i g m o i d } \left( \frac { z _ { b , k } + \delta _ { b , k } } { T _ { b } } \right) , } \\ { \displaystyle S _ { b } ( \boldsymbol { A } ) = \sum _ { k = 1 } ^ { 5 } a _ { k } \log p _ { b , k } } \\ { \displaystyle \qquad + \lambda _ { b } \sum _ { k = 1 } ^ { 5 } ( 1 - a _ { k } ) \log ( 1 - p _ { b , k } ) } \\ { \displaystyle \qquad + \beta _ { b } \mathbb { I } ( | \boldsymbol { A } | = 3 ) , } \\ { \displaystyle \qquad \widehat { K } _ { b } = \arg _ { \boldsymbol { A } \in \mathcal { A } _ { m + n } } S _ { b } ( \boldsymbol { A } ) . } \end{array}\tag{22b}
$$

(22c)

The fixed set $\mathcal { A } _ { \mathrm { m i x } }$ contains all 16 protocol-valid twocomponent and three-component masks, including the six held-out masks. These masks follow the task definition and contain no learned information. The parameters $T _ { b } , \lambda _ { b } , \beta _ { b } ,$ and $\delta _ { b , k }$ are selected only on training-listed development data.

TABLE IV  
COMPOSITION-LEVEL FINAL-TEST PARTITIONS
<table><tr><td>Partition</td><td>Classes</td><td> $N _ { \mathrm { r e c } }$ </td><td>Component sets</td></tr><tr><td>Training-listed final test</td><td>10</td><td>8,887</td><td>STJ+LFMJ, STJ+PBNJ, MTJ+LFMJ, MTJ+PTJ, LFMJ+PTJ, PTJ+PBNJ, STJ+LFMJ+PBNJ, STJ+PTJ+PBNJ, MTJ+LFMJ+PTJ.MTJ+LFMJ+PBNJ</td></tr><tr><td>Held-out final test</td><td>6</td><td>5,333</td><td>STJ+PTJ, MTJ+PBNJ, LFMJ+PBNJ, STJ+LFMJ+PTJ, MTJ+PTJ+PBNJ, LFMJ+PTJ+PBNJ</td></tr><tr><td>Full mixed final test</td><td>16</td><td>14,220</td><td>Union of the training-listed and held-out final-test partitions</td></tr></table>

The comparison decoder has no label-query module, detached cardinality branch, paired loss, or set-level training loss.

The reference models receive measured singleton records and measured mixtures from the ten training-listed compositions for gradient training, with no online IQ or image composition. SPOC-Net instead uses measured singletons and online-composed mixtures, as in Table I. Both protocols exclude $\scriptstyle A _ { \mathrm { h o l d } }$ compositions from model development. Thus, the reference models provide a measured-mixture-supervised benchmark, rather than a matched-data ablation of the proposed architecture. Their training-listed results assess recognition with direct mixture supervision, whereas their held-out results assess transfer beyond those combinations.

All comparison models use 224×224 grayscale STFT inputs and random initialization. ImageNet pretraining is not used. They are trained with AdamW for 120 epochs, batch size 32, a common balanced sampler, and the same baseline augmentation policy. The augmentation includes time and frequency shifts, time and frequency flips, intensity and bias changes, light image noise, random erasing, and stripe masking. EMA weights with decay 0.999 are used for evaluation. Each reference model uses one test view per record, whereas SPOC-Net uses four-view averaging (Table III). This difference is retained in the reported results; the comparison therefore reflects the complete inference procedures as well as the different training data. Decoder calibration uses only training-listed development data and never uses a held-out waveform or label.

AlexNet-ML and ResNet18-ML use their canonical convolutional backbones with a five-output final layer. ACSNet-ML keeps its asymmetric convolution blocks and replaces the original closed-set classifier with the five-output head. MSFF-KAN-ML and TSFANet-ML are our implementations of the published architectures, adapted to five-label prediction. Feature-ResNet18-ML combines the 512-dimensional ResNet18 image descriptor with 14 signal statistics. The statistics contain 11 time-domain values and three frequencydomain values. For the reported experiments, the 14 signal statistics were computed from the recorded IQ sequences. The 14 values are mapped to 128 dimensions, concatenated with the image descriptor, and passed through a 512-dimensional fusion layer before the five-output head.

The reference-model names in the figure legends omit the “-ML” suffix but refer throughout to the five-label versions specified here. None of those plotted component-set results uses a strict 10-class softmax classifier.

## D. Evaluation Metrics

For an evaluation subset containing $N _ { \mathrm { r e c } }$ records, let $\widehat { \mathbf { y } } _ { i }$ and y be the predicted and true multi-hot membership vectors.

Exact-set accuracy, Hamming loss, and cardinality accuracy are

$$
\mathrm { E x a c t } = \frac { 1 } { N _ { \mathrm { r e c } } } \sum _ { i = 1 } ^ { N _ { \mathrm { r e c } } } \mathbb { I } ( \widehat { \mathbf { y } } _ { i } = \mathbf { y } _ { i } ) ,\tag{23a}
$$

$$
\mathrm { H a m m i n g } = \frac { 1 } { 5 N _ { \mathrm { r e c } } } \sum _ { i = 1 } ^ { N _ { \mathrm { r e c } } } \sum _ { k = 1 } ^ { 5 } \mathbb { I } ( \widehat { y } _ { i k } \neq y _ { i k } ) ,
$$

$$
\mathrm { C a r d A c c } = \frac { 1 } { N _ { \mathrm { r e c } } } \sum _ { i = 1 } ^ { N _ { \mathrm { r e c } } } \mathbb { I } ( | \widehat { \mathcal { K } } _ { i } | = | \mathcal { K } _ { i } | ) .\tag{23b}
$$

(23c)

Micro-precision, micro-recall, and micro-F1 pool true positives, false positives, and false negatives over all records and primitive labels. Exact-set accuracy is the primary metric because it requires both correct primitive identities and correct set size.

## VI. RESULTS AND ANALYSIS

The results address three questions: how reliably SPOC-Net recovers complete component sets across JNR, whether this ability transfers to compositions excluded from model development, and which errors limit that transfer. We first examine aggregate and JNR-dependent performance, then compare the training-listed and held-out partitions, and finally relate composition-level errors to primitive recognition and cardinality estimation. All numerical results below use the test records and model protocols of Section V.

## A. Overall and JNR-Dependent Performance

The aggregate results in Table VI distinguish recovering most components from recovering the complete set. Exactset accuracy is 80.69%, whereas micro-F1 is 92.84%: many unsuccessful set decisions therefore still contain correct primitive labels. Micro-precision exceeds micro-recall, indicating a greater tendency to omit active components than to introduce false ones. This matters for interference monitoring because identifying only the dominant jammer can leave an additional type unaccounted for. The difference between twocomponent and three-component exact-set accuracy, 88.35% versus 70.83%, locates the main difficulty in the more crowded mixtures rather than in component-set recognition uniformly.

The JNR dependence in Table VII shows where this limitation is most severe. At nonnegative JNR, exact-set accuracy is at least 99.49%; reducing JNR therefore exposes a detectability problem that is largely hidden by the high-JNR results. At −15 dB, two-component accuracy remains 82.65%, but three-component accuracy falls to 11.60%. At −5 dB, the corresponding values have recovered to 99.40% and 91.93%. Because JNR describes aggregate jammer power, it does not guarantee that every constituent is equally visible, especially when relative component powers are unequal. The pattern is consistent with a weak additional component being obscured by noise or overlap. These aggregate results alone cannot isolate the respective effects of STFT clipping, image resizing, and learned feature extraction.

TABLE V  
TRAINING AND INFERENCE PROTOCOL FOR THE COMPONENT-SET COMPARISON MODELS
<table><tr><td>Category</td><td>Setting</td><td>Category</td><td>Setting</td></tr><tr><td>Prediction task</td><td>Five-label primitive-set prediction</td><td>Output and loss</td><td>Five logits, sigmoid, and ASL</td></tr><tr><td>Gradient data Held-out development</td><td>Measured singletons and measured  $\mathcal { A } _ { \mathrm { l i s t } }$  mixtures</td><td>Online composition Final candidate uni-</td><td>None All 16 valid two-component and three-component</td></tr><tr><td>data</td><td>None</td><td>verse</td><td>masks</td></tr><tr><td>Primary image input</td><td>224 × 224 grayscale STFT</td><td>JNR side information</td><td>None for every model</td></tr><tr><td>Initialization Augmentation</td><td>Random, without ImageNet pretraining Shift, flip, jitter, noise, erase, and stripe mask</td><td>Optimization</td><td>AdamW, 120 epochs, batch size 32</td></tr><tr><td></td><td></td><td>Sampling</td><td>Composition-balanced with higher three-component and low-JNR weight</td></tr><tr><td>EMA</td><td>Decay 0.999</td><td>Test-time averaging</td><td>One view</td></tr><tr><td>Calibration</td><td>Training-listed development data only</td><td>Calibrated terms</td><td>Temperature, negative weight, three-component bias, and component biases</td></tr><tr><td></td><td>Feature-ResNet18-MLResNet18 plus 14 signal statistics</td><td>Strict softmax heads</td><td>Not used in the reported component-set compar- isons</td></tr></table>

TABLE VI  
OVERALL SPOC-NET RESULTS ON THE FULL MEASURED MIXED SET
<table><tr><td>Test data</td><td> $N _ { \mathrm { r e c } }$ </td><td> $( \% )$ </td><td>(%)</td><td>(%)</td><td> $( \% )$ </td><td>Exact-set Micro-precision Micro-recall Micro-F1 Cardinality accuracy Hamming loss (%)</td><td>(%)</td></tr><tr><td>Full mixed set 14,220</td><td></td><td>80.69</td><td>95.31</td><td>90.49</td><td>92.84</td><td>87.08</td><td>6.81</td></tr></table>

## B. Generalization Across Composition Partitions

Table VIII separates two distinct questions: recognition of compositions represented during development and recognition of combinations never used for development. These are complementary views of the same final test set, not results from separately fitted models. The distinction is important because strong accuracy on the training-listed partition does not by itself demonstrate compositional generalization.

On training-listed compositions, SPOC-Net is competitive but not the best-performing method: its 80.57% exact-set accuracy is 1.43 percentage points below MSFF-KAN-ML. Direct measured-mixture supervision remains useful in this setting. The ranking changes on held-out combinations, where SPOC-Net achieves 80.89%, compared with 62.12% for ResNet18- ML, the strongest reference. Thus, the principal benefit is not a uniform improvement on familiar mixtures, but retention of recognition accuracy when known primitive types appear in an untrained combination.

The similar training-listed and held-out averages for SPOC-Net should not be interpreted as evidence that every composition is equally difficult: the two partitions contain different component sets. Rather, together with the class-level results below, they support transfer on the specified split. Its leading full-set accuracy is driven by this held-out performance. Because the reference methods use measured mixed training data and one test view while SPOC-Net uses generated mixtures and four test views, the 18.77-point held-out margin measures the difference between complete reported procedures. It does not isolate the contribution of the composer, an individual network branch, or test-time averaging.

Fig. 5 examines only the 8,887 training-listed test records, whereas Fig. 6 includes all 14,220 records. Their rankings are therefore not contradictory: the JNR curves emphasize familiar compositions, while the confusion matrices also expose errors on held-out combinations. The model settings remain fixed across these views.

The low-JNR three-component comparison in Table IX identifies a clear cost of the proposed training recipe. At −15 dB, SPOC-Net attains 10.9% accuracy, compared with 40.4% and 43.3% for MSFF-KAN-ML and TSFANet-ML. The gap narrows as JNR increases. This suggests that the advantage on new compositions does not remove the need for better weak-component recognition on familiar ones. Exposure to measured mixtures is one plausible contributor to the references’ advantage in this region, but the present comparison does not separate that factor from architectural and inference differences.

## C. Composition-Level and Primitive-Level Results

The row-normalized confusion matrices in Fig. 6 show whether the partition averages conceal composition-specific failures. The reference models exhibit pronounced errors on several held-out rows, whereas SPOC-Net retains a stronger diagonal there. The numerical discussion below and Table X complement the matrices by identifying the relevant compositions explicitly.

Table X also shows that held-out status alone does not determine difficulty. For example, SPOC-Net recognizes MTJ+PBNJ with 98.76% accuracy, while the three held-out three-component sets range from 67.91% to 76.49%. STJ+PTJ provides a particularly clear contrast with the references: SPOC-Net reaches 87.7%, whereas the next-best result in Fig. 6 is 19.8% for ResNet18-ML. This example demonstrates successful recombination of known labels, but the lower threecomponent accuracies prevent a claim of uniformly reliable transfer. The 94.13% micro-F1 for MTJ+PTJ+PBNJ alongside its 67.91% exact-set accuracy further illustrates how recovering most labels can conceal frequent incomplete sets.

TABLE VII  
SPOC-NET PERFORMANCE VERSUS JNR ON THE FULL MIXED SET
<table><tr><td>JNR (dB)</td><td> $N _ { \mathrm { r e c } }$ </td><td>Exact-set (%)</td><td>Micro-F1 (%)</td><td>Two-component exact-set (%)</td><td>Three-component exact-set (%)</td></tr><tr><td>-20</td><td>1,778</td><td>16.59</td><td>58.85</td><td>29.28</td><td>0.13</td></tr><tr><td>-15</td><td>1,773</td><td>51.55</td><td>85.56</td><td>82.65</td><td>11.60</td></tr><tr><td>-10</td><td>1,777</td><td>81.99</td><td>95.84</td><td>96.10</td><td>63.79</td></tr><tr><td>-5</td><td>1,776</td><td>96.11</td><td>99.18</td><td>99.40</td><td>91.93</td></tr><tr><td>0</td><td>1,778</td><td>99.49</td><td>99.90</td><td>99.70</td><td>99.23</td></tr><tr><td>5</td><td>1,777</td><td>99.72</td><td>99.94</td><td>99.90</td><td>99.49</td></tr><tr><td>10</td><td>1,785</td><td>99.94</td><td>99.98</td><td>100.00</td><td>99.87</td></tr><tr><td>15</td><td>1,776</td><td>100.00</td><td>100.00</td><td>100.00</td><td>100.00</td></tr></table>

TABLE VIII  
EXACT-SET ACCURACY OF FIVE-LABEL COMPONENT-SET MODELS ACROSS COMPOSITION PARTITIONS
<table><tr><td>Model</td><td>Training-listed 10 classes (%)</td><td>Held-out 6 classes (%)</td><td>Full 16 classes (%)</td></tr><tr><td>AlexNet-ML</td><td>75.48</td><td>34.97</td><td>60.29</td></tr><tr><td>ResNet18-ML</td><td>79.00</td><td>62.12</td><td>72.67</td></tr><tr><td>Feature-ResNet18-ML</td><td>80.45</td><td>59.48</td><td>72.59</td></tr><tr><td>ACSNet-ML</td><td>78.87</td><td>43.95</td><td>65.77</td></tr><tr><td>MSFF-KAN-ML</td><td>82.00</td><td>50.85</td><td>70.32</td></tr><tr><td>TSFANet-ML</td><td>81.69</td><td>41.83</td><td>66.74</td></tr><tr><td>SPOC-Net</td><td>80.57</td><td>80.89</td><td>80.69</td></tr></table>

Training-listed-composition accuracy vs. JNR

![](images/c5e2fb35b86f2c9f213076c52ab4b41fd3aeadf9995dd4ce8f1f1efa7d61f366.jpg)  
Fig. 5. Exact-set accuracy versus JNR on the 10 training-listed mixed compositions. The panels report overall, two-component, and three-component results for 8,887 measured records. The figure legend omits the “-ML” suffix for comparison models.

TABLE IX  
THREE-COMPONENT ACCURACY ON TRAINING-LISTED COMPOSITIONS AT LOW JNR
<table><tr><td>JNR (dB) SPOC-Net (%)</td><td>MSFF-KAN-ML (%)</td><td>TSFANet-ML (%)</td></tr><tr><td>-20</td><td>0.2</td><td>2.9</td></tr><tr><td>-15</td><td>10.9</td><td>20.0 40.4 43.3</td></tr><tr><td>-10</td><td>61.0</td><td>80.4 76.6</td></tr><tr><td>-5</td><td>92.1</td><td>96.4 93.9</td></tr></table>

The primitive-level results in Table XI distinguish missed labels from spurious ones. STJ, LFMJ, and PTJ have precision above 98.9%, but their lower recall indicates conservative predictions. LFMJ has the lowest recall, consistent with the difficulty of retaining a weak chirp in an overlapping mixture. MTJ exhibits the opposite imbalance: its recall exceeds its precision, indicating more false activations relative to missed detections. Confusion between narrowband patterns and structured regions of other signals is a possible explanation, but these aggregate primitive scores do not establish which competing type causes each error.

TABLE X  
SPOC-NET RESULTS ON THE SIX HELD-OUT COMPOSITIONS
<table><tr><td>Composition</td><td> $N _ { \mathrm { r e c } }$ </td><td>Exact-set (%)</td><td>Micro-F1 (%)</td></tr><tr><td>STJ + PTJ</td><td>889</td><td>87.74</td><td>93.51</td></tr><tr><td>MTJ + PBNJ</td><td>890</td><td>98.76</td><td>99.47</td></tr><tr><td>LFMJ + PBNJ</td><td>889</td><td>84.81</td><td>92.77</td></tr><tr><td>STJ + LFMJ + PTJ</td><td>889</td><td>76.49</td><td>89.84</td></tr><tr><td>MTJ + PTJ + PBNJ</td><td>888</td><td>67.91</td><td>94.13</td></tr><tr><td>LFMJ + PTJ + PBNJ</td><td>888</td><td>69.59</td><td>90.24</td></tr></table>

TABLE XI  
PRIMITIVE-WISE PERFORMANCE ON THE FULL MIXED SET
<table><tr><td>Primitive</td><td>Precision (%)</td><td>Recall (%)</td><td>F1 (%)</td></tr><tr><td>STJ</td><td>98.99</td><td>90.29</td><td>94.44</td></tr><tr><td>MTJ</td><td>85.34</td><td>95.72</td><td>90.23</td></tr><tr><td>LFMJ</td><td>99.59</td><td>87.62</td><td>93.22</td></tr><tr><td>PTJ</td><td>99.86</td><td>90.50</td><td>94.95</td></tr><tr><td>PBNJ</td><td>92.57</td><td>90.00</td><td>91.26</td></tr></table>

![](images/a79b1017d67b2902cce76db274fdbc5abe33835c1aad905c070ad0b2cd8189df.jpg)  
Fig. 6. Row-normalized composition confusion matrices on the full measured mixed set. The upper row contains AlexNet-ML, ResNet18-ML, and Feature ResNet18-ML. The middle row contains ACSNet-ML, MSFF-KAN-ML, and TSFANet-ML. The lower panel contains SPOC-Net. All 16 two-component and three-component classes and all tested JNR values are included. The artwork omits the “-ML” suffix.

TABLE XII  
ROW-NORMALIZED CARDINALITY CONFUSION
<table><tr><td>True cardinality</td><td>Predicted 2 (%)</td><td>Predicted 3 (%)</td></tr><tr><td>2</td><td>99.48</td><td>0.52</td></tr><tr><td>3</td><td>28.86</td><td>71.14</td></tr></table>

## D. Cardinality Errors and Practical Limitations

Table XII identifies underestimation of component count as the dominant cardinality error: 28.86% of true threecomponent records are assigned two components, whereas only 0.52% of two-component records are assigned three. Moreover, three-component cardinality accuracy (71.14%) is close to exact-set accuracy (70.83%). On this subset, selecting the correct number of types therefore almost always coincides with selecting the correct identities. The principal remaining challenge is recognizing that another component is present, rather than frequently choosing the wrong three-type combination.

The most frequent set errors in Table XIII reinforce this interpretation. Every listed prediction is a two-component subset of a three-component target, with the remaining components identified correctly. In these cases, the missing label is PBNJ or LFMJ. Together with the low-JNR and recall results, these omissions are consistent with insufficient evidence for an additional type. The error counts describe the observed failures; they do not by themselves establish the received power of each omitted component.

Taken together, the results support singleton-based IQ composition as a useful route to recognizing untrained combinations under the reported acquisition conditions. The method retains competitive accuracy on training-listed mixtures and transfers more effectively than the reference procedures on the chosen held-out split. The evidence supports this systemlevel conclusion, not a separate causal claim for each network module.

TABLE XIII  
MOST FREQUENT SET ERRORS
<table><tr><td>True set</td><td>Predicted set</td><td>Count</td></tr><tr><td>{MTJ, LFMJ, PBNJ}</td><td>{MTJ, PBNJ}</td><td>130</td></tr><tr><td>{MTJ, PTJ, PBNJ}</td><td>{MTJ, PTJ}</td><td>128</td></tr><tr><td>{STJ, PTJ, PBNJ}</td><td>{STJ, PTJ}</td><td>119</td></tr><tr><td>{MTJ, LFMJ, PBNJ}</td><td>{MTJ, LFMJ}</td><td>112</td></tr><tr><td>{STJ, LFMJ, PBNJ}</td><td>{STJ, LFMJ}</td><td>112</td></tr></table>

The main limitation is low-JNR recognition of a third component. Fixed-range STFT clipping and subsequent feature compression can suppress weak patterns; the high-resolution branch and paired objective are designed to address this issue, but their individual effects are not isolated by the reported results. The conducted platform provides repeatable RF superposition but does not reproduce outdoor source-dependent propagation, antenna-pattern changes, or multipath. Relative to the linear composer, the physical path can introduce transmitter and receiver responses, combiner effects, gain variation, and front-end impairments. Their individual contributions are not quantified here, so the results establish performance on this hardware setup rather than general robustness to every RF impairment.

The present evaluation also uses one fixed composition partition and does not report uncertainty across repeated training runs. Generalization to other partitions therefore remains to be assessed. The decoder is limited to the five defined primitives and valid two-component and three-component sets; it is not a complete interference monitor with no-interference, singleton, and unknown-type decisions. Extending those decisions, evaluating additional composition splits, and quantifying runto-run variability are directions for future work.

## VII. CONCLUSION

This paper presented SPOC-Net for identifying the component types of mixed GNSS jamming when measured singlecomponent records are the only physical observations used for gradient optimization. Its online IQ composer generates labeled mixtures during training, while component queries, a high-resolution cardinality branch, and structured decoding combine identity and set-size evidence at inference. Measured mixtures of training-listed compositions remain available for model selection and calibration, and six other compositions are excluded from every development decision. On 14,220 independently acquired conducted mixed records, the method achieves 80.69% exact-set accuracy and 92.84% micro-F1. Its held-out accuracy of 80.89% exceeds the strongest reference by 18.77 percentage points under the reported training and inference protocols. The results support recognition of new combinations of known jamming types without measuredmixture gradient training. Weak-component omissions at low JNR, the fixed composition split, and the conducted acquisition setting delimit that conclusion.

## REFERENCES

[1] International Telecommunication Union, “Framework and Overall Objectives of the Future Development of IMT for 2030 and Beyond,” Recommendation ITU-R M.2160-0, Nov. 2023.

[2] H. Cui, J. Zhang, Y. Geng, Z. Xiao, T. Sun, N. Zhang, J. Liu, Q. Wu, and X. Cao, “Space-Air-Ground Integrated Network (SAGIN) for 6G: Requirements, Architecture and Challenges,” China Commun., vol. 19, no. 2, pp. 90–108, Feb. 2022.

[3] G. X. Gao, M. Sgammini, M. Lu, and N. Kubo, “Protecting GNSS Receivers From Jamming and Interference,” Proc. IEEE, vol. 104, no. 6, pp. 1327–1338, Jun. 2016.

[4] K. D. Wesson, J. N. Gross, T. E. Humphreys, and B. L. Evans, “GNSS Signal Authentication Via Power and Distortion Monitoring,” IEEE Trans. Aerosp. Electron. Syst., vol. 54, no. 2, pp. 739–754, Apr. 2018.

[5] P. Wang, E. Cetin, A. G. Dempster, Y. Wang, and S. Wu, “Time-Frequency and Statistical Inference Based Interference Detection Technique for GNSS Receivers,” IEEE Trans. Aerosp. Electron. Syst., vol. 53, no. 6, pp. 2865–2876, Dec. 2017.

[6] P. Wang, E. Cetin, A. G. Dempster, Y. Wang, and S. Wu, “GNSS Interference Detection Using Statistical Analysis in the Time-Frequency Domain,” IEEE Trans. Aerosp. Electron. Syst., vol. 54, no. 1, pp. 416– 428, Feb. 2018.

[7] W. Qin, M. T. Gamba, E. Falletti, and F. Dovis, “An Assessment of Impact of Adaptive Notch Filters for Interference Removal on the Signal Processing Stages of a GNSS Receiver,” IEEE Trans. Aerosp. Electron. Syst., vol. 56, no. 5, pp. 4067–4082, Oct. 2020.

[8] W. Qin and F. Dovis, “Situational Awareness of Chirp Jamming Threats to GNSS Based on Supervised Machine Learning,” IEEE Trans. Aerosp. Electron. Syst., vol. 58, no. 3, pp. 1707–1720, Jun. 2022.

[9] X. Chen, D. He, X. Yan, W. Yu, and T.-K. Truong, “GNSS Interference Type Recognition With Fingerprint Spectrum DNN Method,” IEEE Trans. Aerosp. Electron. Syst., vol. 58, no. 5, pp. 4745–4760, Oct. 2022.

[10] A. Siemuri, K. Selvan, H. Kuusniemi, P. Valisuo, and M. S. Elmusrati,¨ “A Systematic Review of Machine Learning Techniques for GNSS Use Cases,” IEEE Trans. Aerosp. Electron. Syst., vol. 58, no. 6, pp. 5043– 5077, Dec. 2022.

[11] F. B. da Silva, E. Cetin, and W. A. Martins, “Radio Frequency Interference Detection Using Nonnegative Matrix Factorization,” IEEE Trans. Aerosp. Electron. Syst., vol. 58, no. 2, pp. 868–878, Apr. 2022.

[12] F. B. da Silva, E. Cetin, and W. A. Martins, “Radio Frequency Interference Mitigation Via Nonnegative Matrix Factorization for GNSS,” IEEE Trans. Aerosp. Electron. Syst., vol. 59, no. 4, pp. 3493–3504, Aug. 2023.

[13] K. Sun, M. Elhajj, and W. Y. Ochieng, “A GNSS Anti-Interference Method Based on Fractional Fourier Transform,” IEEE Trans. Aerosp Electron. Syst., vol. 60, no. 5, pp. 5636–5650, Oct. 2024.

[14] Y. Luo, H. Luo, P. Hu, B. Han, J. Tian, M. Zeng, and H. Jiang, “Zak-Transform-Based Adaptive Interference Extraction Method for GNSS Interference Mitigation,” IEEE Trans. Aerosp. Electron. Syst., vol. 60, no. 4, pp. 4784–4793, Aug. 2024.

[15] J. R. van der Merwe, D. Contreras Franco, T. Feigl, and A. Rugamer,¨ “Optimal Machine Learning and Signal Processing Synergies for Low-Resource GNSS Interference Classification,” IEEE Trans. Aerosp. Electron. Syst., vol. 60, no. 3, pp. 2705–2721, Jun. 2024.

[16] Z. Zeng, K. Wang, Z. Zhang, and Y. Xiu, “GAC-KAN: An Ultra-Lightweight GNSS Interference Classifier for GenAI-Powered Consumer Edge Devices,” arXiv preprint arXiv:2602.11186, 2026.

[17] Z. Zeng, Y. Zhao, K. Wang, D. Niyato, Y. Xiu, L. Chen, Z. Zhang, and N. Wei, “PhyG-MoE: A Physics-Guided Mixture-of-Experts Framework for Energy-Efficient GNSS Interference Recognition,” arXiv preprint arXiv:2601.12798, 2026.

[18] Z. Zeng, Y. Zhao, K. Wang, D. Niyato, H. Shu, J. Zhao, Y. Huang, Y. Xiu, Z. Zhang, and N. Wei, “SKANet: A Cognitive Dual-Stream Framework with Adaptive Modality Fusion for Robust Compound GNSS Interference Classification,” arXiv preprint arXiv:2601.12791, 2026.

[19] Z. Zeng, H. Shu, K. Wang, L. Chen, A. Hussain, Y. Huang, J. Zhao, Y. Xiu, and Z. Zhang, “JSR-GFNet: Jamming-to-Signal Ratio-Aware Dynamic Gating for Interference Classification in future Cognitive Global Navigation Satellite Systems,” arXiv preprint arXiv:2602.00042, 2026.

[20] I. E. Mehr and F. Dovis, “A Deep Neural Network Approach for Classification of GNSS Interference and Jamming,” IEEE Trans. Aerosp. Electron. Syst., vol. 61, no. 2, pp. 1660–1676, Apr. 2025.

[21] Q. Jia, L. Zhang, and R. Wu, “Lightweight GNSS Interference Classifier and Its Solution Under Few-Shot Conditions,” IEEE Trans. Aerosp. Electron. Syst., vol. 61, no. 6, pp. 17426–17444, Dec. 2025.

[22] Z. Xiao, X. Jiang, T. Li, T. Li, Y. Zhang, and S. Zhou, “Compound Interference Recognition for GNSS Via Integrating Time-Frequency With Power Spectrum Features,” IEEE Trans. Aerosp. Electron. Syst., vol. 61, no. 5, pp. 13010–13021, Oct. 2025.

[23] X. Li, K. Zhang, J. Wang, Z. Lu, F. Chen, F. Wang, and P. Liu, “A Multipolarization Interference Suppression Strategy of GNSS Antenna Array Based on Polarization Equalization,” IEEE Trans. Aerosp. Electron. Syst., vol. 61, no. 5, pp. 12412–12424, Oct. 2025.

[24] W. Zhong, H. Xiong, Y. Hua, D. H. Shah, Z. Liao, and Y. Xu, “TSFANet: Temporal-Spatial Feature Aggregation Network for GNSS Jamming Recognition,” IEEE Trans. Instrum. Meas., vol. 73, pp. 1–13, 2024.

[25] Q. Jia, L. Zhang, and R. Wu, “Low-Power Interference Identification Based on Convolutional Neural Networks,” IEEE Trans. Instrum. Meas., vol. 74, pp. 1–17, 2025.

[26] M. Jiang, Z. Ye, Y. Xiao, Y. Gao, M. Xiao, and D. Niyato, “ACSNet: A Deep Neural Network for Compound GNSS Jamming Signal Classification,” IEEE Trans. Cogn. Commun. Netw., vol. 12, pp. 1601–1615, 2026.

[27] Z. Li, L. Zheng, Q. Zhang, H. Wang, Z. Du, and J. Liu, “GNSS Jamming Attacks Recognition Based on Dual GCN With Adaptive Weight Learning,” IEEE Sensors J., vol. 25, no. 13, pp. 26152–26168, Jul. 2025.

[28] J. Song, Z. Lu, X. Zhao, W. Xiao, X. Tang, and G. Sun, “GNSS Multiple Interference Mitigation With TF-Unet Method Based on High-Resolution Interference Sensing,” IEEE Sensors J., vol. 26, no. 1, pp. 1358–1369, Jan. 2026.

[29] S. Jeeru, L. Jiao, P.-A. Andersen, and O.-C. Granmo, “Interpretable Rule-Based Architecture for GNSS Jamming Signal Classification,” IEEE Sensors J., vol. 25, no. 10, pp. 17942–17959, May 2025.

[30] I. E. Mehr and F. Dovis, “Detection and Classification of GNSS Jammers Using Convolutional Neural Networks,” in Proc. Int. Conf. Localization GNSS (ICL-GNSS), 2022, pp. 1–6.

[31] F. Ott, L. Heublein, N. L. Raichur, T. Feigl, J. Hansen, A. Rugamer, and¨ C. Mutschler, “Few-Shot Learning With Uncertainty-Based Quadruplet Selection for Interference Classification in GNSS Data,” in Proc. Int. Conf. Localization GNSS (ICL-GNSS), 2024, pp. 1–7.

[32] M. Spanghero, F. Geib, R. Panier, and P. Papadimitratos, “GNSS Jammer Localization and Identification With Airborne Commercial GNSS

Receivers,” IEEE Trans. Inf. Forensics Security, vol. 20, pp. 3550–3565, 2025.

[33] D. Borio, “A Multi-State Notch Filter for GNSS Jamming Mitigation,” in Proc. Int. Conf. Localization GNSS (ICL-GNSS), 2014, pp. 1–6.

[34] M. T. Gamba and E. Falletti, “Performance Comparison of FLL Adaptive Notch Filters to Counter GNSS Jamming,” in Proc. Int. Conf. Localization GNSS (ICL-GNSS), 2019, pp. 1–6.

[35] F. Garzia, J. R. van der Merwe, A. Rugamer, S. Urquijo, S. Taschke, and¨ W. Felber, “Sub-Band AGC-Based Interference Mitigation,” in Proc. Int. Conf. Localization GNSS (ICL-GNSS), 2021, pp. 1–6.

[36] Y. Cai, K. Shi, F. Song, Y. Xu, X. Wang, and H. Luan, “Jamming Pattern Recognition Using Spectrum Waterfall: A Deep Learning Method,” in Proc. IEEE 5th Int. Conf. Comput. Commun. (ICCC), 2019, pp. 2113– 2117.

[37] C. Liu, B. Ren, D. Fu, Y. Xie, and F. Chen, “A GNSS Composite Interference Recognition Method Based on YOLOv5,” in Proc. IEEE 6th Int. Conf. Civil Aviation Safety Inf. Technol. (ICCASIT), 2024, pp. 1157–1162.

[38] I. E. Mehr, G. Caputo, D. Salza, M. Fantino, and F. Dovis, “Towards a Faster GNSS Interference Classification: A GRU-Based Approach Using Spectrograms,” in Proc. IEEE/ION Position, Location Navig. Symp. (PLANS), 2025, pp. 372–380.

[39] R. M. Ferre, A. de la Fuente, and E. S. Lohan, “Jammer Classification in GNSS Bands Via Machine Learning Algorithms,” Sensors, vol. 19, no. 22, p. 4841, 2019.

[40] Y. Meng, L. Yu, and Y. Wei, “Multi-Label Radar Compound Jamming Signal Recognition Using Complex-Valued CNN With Jamming Class Representation Fusion,” Remote Sens., vol. 15, no. 21, p. 5180, 2023.

[41] Y. Xiao, R. Zhang, X. Yu, and Y. Jiang, “Open-Set Recognition of Compound Jamming Signal Based on Multi-Task Multi-Label Learning,” IET Radar Sonar Navig., vol. 18, no. 8, pp. 1235–1246, 2024.

[42] M.-L. Zhang and Z.-H. Zhou, “A Review on Multi-Label Learning Algorithms,” IEEE Trans. Knowl. Data Eng., vol. 26, no. 8, pp. 1819– 1837, Aug. 2014.

[43] W. J. Scheirer, A. de Rezende Rocha, A. Sapkota, and T. E. Boult, “Toward Open Set Recognition,” IEEE Trans. Pattern Anal. Mach. Intell., vol. 35, no. 7, pp. 1757–1772, Jul. 2013.

[44] C. Geng, S.-J. Huang, and S. Chen, “Recent Advances in Open Set Recognition: A Survey,” IEEE Trans. Pattern Anal. Mach. Intell., vol. 43, no. 10, pp. 3614–3631, Oct. 2021.

[45] S. Liu, L. Zhang, X. Yang, H. Su, and J. Zhu, “Query2Label: A Simple Transformer Way to Multi-Label Classification,” arXiv preprint arXiv:2107.10834, 2021.

[46] Z. Zeng, N. Wei, Y. Xiu, G. Gui, J. Wu, K. Wang, and Z. Zhang, “A Physics-Guided Mixture of Experts Framework for Energy-Efficient Electromagnetic Interference Recognition in Global Navigation Satellite Systems,” in Proc. IEEE/CIC Int. Conf. Commun. China (ICCC), 2026, pp. 760–765.

[47] Z. Zeng, A. Hussain, Y. Xiu, P. L. Yeoh, L. Chen, Z. Zhang, and G. Gui, “Geometry-Aware Cross-Height Channel Knowledge Map Prediction for UAV-Assisted Communications With Uncertainty-Guided 3D Sensing,” arXiv preprint arXiv:2607.00887, 2026.

[48] Z. Zeng, K. Wang, Z. Zhang, and C. Huang, “QuaMoE-DRF: Proactive Beam and Rate Adaptation via Multimodal Dynamic Radio Map Forecasting in ISAC Networks,” arXiv preprint arXiv:2607.00974, 2026.

[49] Z. Zeng, N. Wei, K. Wang, P. L. Yeoh, F. Xu, Y. Xiu, and Z. Zhang, “Sparse Gain Radio Map Reconstruction With Geometry Priors and Uncertainty-Guided Measurement Selection,” arXiv preprint arXiv:2604.05788, 2026.

[50] T. Ridnik, G. Sharir, A. Ben-Cohen, E. Ben-Baruch, and A. Noy, “ML-Decoder: Scalable and Versatile Classification Head,” in Proc. IEEE/CVF Winter Conf. Appl. Comput. Vis. (WACV), 2023, pp. 32–41.

[51] T. Ridnik, E. Ben-Baruch, N. Zamir, A. Noy, I. Friedman, M. Protter, and L. Zelnik-Manor, “Asymmetric Loss for Multi-Label Classification,” in Proc. IEEE/CVF Int. Conf. Comput. Vis. (ICCV), 2021, pp. 82–91.

[52] S. Woo, S. Debnath, R. Hu, X. Chen, Z. Liu, I. S. Kweon, and S. Xie, “ConvNeXt V2: Co-Designing and Scaling ConvNets With Masked Autoencoders,” in Proc. IEEE/CVF Conf. Comput. Vis. Pattern Recognit. (CVPR), 2023, pp. 16133–16142.

[53] S. Woo, J. Park, J.-Y. Lee, and I. S. Kweon, “CBAM: Convolutional Block Attention Module,” in Proc. Eur. Conf. Comput. Vis. (ECCV), 2018, pp. 3–19.

[54] F. Radenovic, G. Tolias, and O. Chum, “Fine-Tuning CNN Image´ Retrieval With No Human Annotation,” IEEE Trans. Pattern Anal. Mach. Intell., vol. 41, no. 7, pp. 1655–1668, Jul. 2019.

[55] K. He, X. Zhang, S. Ren, and J. Sun, “Deep Residual Learning for Image Recognition,” in Proc. IEEE Conf. Comput. Vis. Pattern Recognit. (CVPR), 2016, pp. 770–778.

[56] A. Krizhevsky, I. Sutskever, and G. E. Hinton, “ImageNet Classification With Deep Convolutional Neural Networks,” in Adv. Neural Inf. Process. Syst., vol. 25, 2012, pp. 1097–1105.

[57] Z. Liu, H. Mao, C.-Y. Wu, C. Feichtenhofer, T. Darrell, and S. Xie, “A ConvNet for the 2020s,” in Proc. IEEE/CVF Conf. Comput. Vis. Pattern Recognit. (CVPR), 2022, pp. 11976–11986.

[58] I. Loshchilov and F. Hutter, “Decoupled Weight Decay Regularization,” in Proc. Int. Conf. Learn. Represent. (ICLR), 2019.

[59] I. Loshchilov and F. Hutter, “SGDR: Stochastic Gradient Descent With Warm Restarts,” in Proc. Int. Conf. Learn. Represent. (ICLR), 2017.

[60] P. D. Welch, “The Use of Fast Fourier Transform for the Estimation of Power Spectra: A Method Based on Time Averaging Over Short, Modified Periodograms,” IEEE Trans. Audio Electroacoust., vol. 15, no. 2, pp. 70–73, Jun. 1967.

[61] L. Cohen, “Time-Frequency Distributions—A Review,” Proc. IEEE, vol. 77, no. 7, pp. 941–981, Jul. 1989.

[62] A. Vaswani, N. Shazeer, N. Parmar, J. Uszkoreit, L. Jones, A. N. Gomez, L. Kaiser, and I. Polosukhin, “Attention Is All You Need,” in Adv. Neural Inf. Process. Syst., vol. 30, 2017, pp. 5998–6008.

[63] N. Srivastava, G. Hinton, A. Krizhevsky, I. Sutskever, and R. Salakhutdinov, “Dropout: A Simple Way to Prevent Neural Networks From Overfitting,” J. Mach. Learn. Res., vol. 15, no. 1, pp. 1929–1958, 2014.
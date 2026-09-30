# SEEING IS NOT ADDRESSING: AUDITING LINGUISTIC ACCESS TO FROZEN VISUAL GEOMETRY

Woosang Jeon<sup>1,∗</sup> Jiwon Yang<sup>1,∗</sup> Soo Chung<sup>1</sup> Taehyeong Kim<sup>1,†</sup>

<sup>1</sup>Seoul National University

{jwoosang1,jwyang0424,soochung,taehyeong.kim}@snu.ac.kr

## ABSTRACT

Visual distinctions are often finer than those reflected in linguistic conceptualization. Vision–language models exhibit a similar asymmetry: a distinction can remain discriminable in frozen image geometry while being weakly addressable through the native text interface. We study this gap by separating visual discriminability from linguistic addressability in text-to-image retrieval. Using FactorAtlas, a fully crossed testbed of 23,040 images spanning shape, hue, pattern, and nuisance variation, we compare both readouts on held-out images of the same distinctions. We then derive image-side contrasts that separate each value from its alternatives for matched visual grounding, and test whether this reduces the native-text access gap across factors and models. Direction-specific and visual-absence controls tie these gains to the relevant visual contrast; the gains persist after global alignment and extend to compositional retrieval and natural images. Together, these results show that visual discriminability and linguistic addressability need not coincide, and that matched visual grounding can probe and reduce the resulting access gap.

## 1 INTRODUCTION

Visual representations often preserve finer distinctions than are reflected in linguistic conceptualization (Liao et al., 2024). Such distinctions become addressable through language when relevant words are linked to their visual referents, and these associations are learned through repeated cooccurrence (Yu & Smith, 2007; Smith & Yu, 2008). Contrastively trained vision–language models have a similar learning structure, acquiring cross-modal correspondences from large-scale paired image–text data (Radford et al., 2021; Zhai et al., 2023). The paired text, however, need not describe every visual distinction present in the image. Accordingly, in the resulting shared embedding space, some of these distinctions can remain discriminable in the image representation while being only weakly addressable through the model’s text interface.

In text-to-image retrieval, this asymmetry motivates separating two aspects of performance: whether image geometry supports the relevant visual distinction, and how well the native text query can access it. Related work shows that representations within each modality retain structured information (Koishigarina et al., 2026), with compositional and ordinal structure in image embeddings (Berasi et al., 2025; Sonthalia et al., 2026). Meanwhile, cross-modal alignment and mismatch have been studied at broader representational and compositional scales (Liang et al., 2022; Kamath et al., 2023; Koishigarina et al., 2026). We instead take each named visual distinction as the unit of analysis, asking how the image-side structure underlying its discriminability relates to native linguistic access.

We operationalize this question using FactorAtlas, a fully crossed testbed of 23,040 images spanning 8 shapes, 12 hues, 10 patterns, and 24 nuisance realizations. For each named factor value, we compare two readouts under held-out conditions: visual discriminability—how reliably each value can be distinguished from its alternatives in frozen image geometry—and linguistic addressability—how well the corresponding text query retrieves images exhibiting that value. We then derive a targetversus-rest visual contrast for each value and use its direction to ground the native text query. We call this matched visual grounding and test whether it improves distinction-specific linguistic access.

Across our experiments, visual discriminability and linguistic addressability diverge, while matched visual grounding yields held-out access gains across factors and models. These gains are directionspecific and weaken as the underlying visual contrast is attenuated. Broad alignment reduces part of the mismatch but leaves residual value-specific access gaps that matched grounding further reduces.

These access gains, in turn, support exact compositional retrieval under held-out conditions, while the same value-specific intervention can be applied selectively and generalizes to natural images. Together, these findings show that linguistic access can lag behind visual structure already present in frozen representations, suggesting a way to expose and use under-accessed visual distinctions.

## 2 FROM VISUAL DISTINCTIONS TO LINGUISTIC ACCESS

## 2.1 VISUAL DISCRIMINABILITY AND LINGUISTIC ADDRESSABILITY

Our analysis separates two questions for the same named visual property: whether its distinction remains discriminable in frozen image geometry and how well the native text query can access it. We term these visual discriminability and linguistic addressability, measured from image-derived evidence and the native text query, respectively. When they diverge, we use the held-out improvement obtained by grounding the native text query with the corresponding image-side structure, which we call matched access gain, as an interventional readout of how much the access gap can be reduced.

![](images/1bfbf3d0c66d98f02672a02d6b6ba537ebc560f2ed28b02ffb566df087619dee.jpg)  
Shared Embedding Space

![](images/1b4e93b2edb691b2bb11f913ae66e1850a58f4c87e23018a20c7d36093fface3.jpg)  
Value-speci��c Visual Contrast

![](images/edba39110a61b18f6573e62125a0a36fb6fbb88d1b81334cfcb86a9edfbddf66.jpg)  
Matched Visual Grounding  
Figure 1: From visual discriminability to linguistic access. A visual distinction may remain discriminable in frozen image geometry yet weakly accessed by its native text query; matched visual grounding connects the two through a value-specific visual contrast.

## 2.2 MATCHED VISUAL GROUNDING VIA VALUE-SPECIFIC VISUAL CONTRASTS

We instantiate this concept by representing each visual value v with an image-side direction that distinguishes it from the other values in the same factor vocabulary. We use this direction to ground the corresponding native text query, a procedure we call matched visual grounding (Fig. 1).

Let $I ( x ) \in \mathbb { R } ^ { d }$ denote the frozen image embedding. For a factor with value set V and value $v \in \mathcal { V } .$ let $C _ { v }$ denote a visual support set for v. We define the target and average-rest image prototypes as

$$
\mu _ { v } ^ { I } = \mathrm { n o r m } \left( \frac { 1 } { | C _ { v } | } \sum _ { x \in C _ { v } } \mathrm { n o r m } ( I ( x ) ) \right) , \qquad \bar { \mu } _ { - v } ^ { I } = \frac { 1 } { | \mathcal { V } | - 1 } \sum _ { u \neq v } \mu _ { u } ^ { I } .
$$

We leave $\bar { \mu } _ { - v } ^ { I }$ unnormalized so that its score under a unit query equals the average score of the remaining value prototypes. Their difference defines the matched visual contrast and its unit direction:

$$
\Delta _ { v } ^ { I } = \mu _ { v } ^ { I } - \bar { \mu } _ { - v } ^ { I } , \qquad d _ { v } ^ { I } = \frac { \Delta _ { v } ^ { I } } { \lVert \Delta _ { v } ^ { I } \rVert } .
$$

Given the native text query $q _ { v }$ , we ground it along this direction:

$$
q _ { v } ^ { \prime } ( \alpha ) = \mathrm { n o r m } \bigl ( q _ { v } + \alpha d _ { v } ^ { I } \bigr ) , \qquad \alpha \geq 0 .\tag{1}
$$

The parameter α controls the grounding strength. Comparing retrieval with $q _ { v } ^ { \prime } ( \alpha )$ against the native query $q _ { v }$ yields the matched access gain introduced above.

## 2.3 GEOMETRIC GUARANTEE

We derive a guarantee in terms of the target-versus-average-rest similarity margin.

$$
M _ { v } ( q ) = q ^ { \top } \Delta _ { v } ^ { I } , \qquad c = q _ { v } ^ { \top } d _ { v } ^ { I } .
$$

## Proposition 1. Matched similarity-margin guarantee

For every $\alpha \geq 0$ for which $q _ { v } ^ { \prime } ( \alpha )$ is defined,

$$
M _ { v } ( q _ { v } ^ { \prime } ( \alpha ) ) \geq M _ { v } ( q _ { v } ) .
$$

The grounded similarity margin has the closed form

$$
M _ { v } ( q _ { v } ^ { \prime } ( \alpha ) ) = \| \Delta _ { v } ^ { I } \| \frac { c + \alpha } { \sqrt { 1 + \alpha ^ { 2 } + 2 \alpha c } } .\tag{2}
$$

For $| c | < 1 , M _ { v } ( q _ { v } ^ { \prime } ( \alpha ) )$ increases strictly with $\alpha .$ . When the initial margin is negative $( - 1 < c < 0 )$ it crosses zero at

$$
\alpha ^ { \star } = - c = - q _ { v } ^ { \top } d _ { v } ^ { I } .
$$

The guarantee is local to the matched target-versus-average-rest margin; whether it translates into held-out retrieval gains and broader retrieval behavior is evaluated empirically below. The full proof, including endpoint cases, is given in Appendix A.

## 3 DIAGNOSING AND REDUCING DISTINCTION-LEVEL ACCESS GAPS

Balanced nuisance variation (12 contexts, 2 renderer seeds)  
![](images/967ae87665da31720b603fc58f10eee22ca34071fd90da58ac944ab03b0e18b2.jpg)

![](images/4b588b7bd9d6707489606dd0c1672f3b43f635d38d74f05d0e315a756a837bb6.jpg)  
Figure 2: FactorAtlas and its primary held-out protocol. The testbed factorially combines 8 shapes, 12 hues, and 10 surface patterns under balanced nuisance variation. Disjoint nuisance contexts are used for calibration, validation, and held-out testing, while the semantic factors remain fully crossed and balanced.

## 3.1 FACTORATLAS AND EVALUATION PROTOCOL

FactorAtlas is a procedurally rendered visual testbed designed to compare visual discriminability and linguistic addressability under controlled variation. It contains 23,040 images spanning 8 shapes, 12 hues, 10 surface patterns, and 24 nuisance realizations. Each realization is jointly determined by a context and renderer seed, varying object pose, viewpoint, illumination, background, and material appearance while preserving the underlying semantic factors. This lets the same semantic distinctions recur across varied visual conditions.

For each semantic factor, we evaluate all of its values jointly. When one factor is the target, the other factors are fully crossed and balanced, so every target value appears equally often with every combination of the remaining factors. This rules out simple cross-factor frequency shortcuts, preventing the readout from relying on correlations between the target value and the other factors.

Our primary protocol holds out nuisance contexts, using four for calibration, two for validation, and six for testing, with both renderer seeds of each context assigned to the same split. For each factor value, the native text query is formed by averaging and renormalizing the embeddings of three fixed prompt templates specific to its factor. Calibration defines the visual support sets in Section 2.2, while validation selects the factor-level grounding strength α. All choices are fixed before evaluation on held-out contexts, so the readouts are tested under nuisance conditions not used to construct or tune them. Full construction and evaluation details are provided in Appendix B.

## 3.2 DISCRIMINABILITY, ADDRESSABILITY, AND MATCHED ACCESS

Discriminability versus addressability. We first compare visual discriminability and linguistic addressability on the same held-out gallery. For each factor value, the former is measured with an image-derived query built from calibration images, whereas the latter uses the corresponding native text query. Thus, the two readouts differ only in how the query is obtained.

![](images/19c37d94baeac27f516a82adf8bf69d6fc597a973a0448ad1e694899cfa54f3a.jpg)

As shown in Fig. 3, image-based queries identify the correct value more reliably than native text queries across shape, hue, and pattern. This reveals the distinctionlevel asymmetry motivating our audit, as the relevant visual distinctions remain readily discriminable in image geometry even when their linguistic handles provide we

Figure 3: Factor-value top-1 readout gap. Visual discriminability exceeds native linguistic addressability across all factors.  
![](images/656769abae262e6ecb46e5b6460d446efd32587e238d0d224865b7c4857d82da.jpg)

![](images/85c903445b45ec0166b0f7ed10a1fe41d52ad9eea67ff8bd711de307aad568a7.jpg)

![](images/2001722bf6204081e173e0acf78620667e2ae94920e8a4554096fcec4e0b6e77.jpg)  
Figure 4: Matched access gain across individual factor values. Native text retrieval and matched visual grounding are compared for each value of shape, hue, and pattern on held-out images.

Matched access gain across factors. We next ask whether grounding the text query with valuespecific image-side information can reduce this access gap. Using the matched visual grounding procedure in Section 2.2, we estimate each visual direction from calibration images and select the factor-level strength α on validation before evaluating the held-out gallery.

Figure 4 shows the result for every value of shape, hue, and pattern. Matched grounding improves average precision for 29 of the 30 values across all three factor vocabularies. Although the gains vary in magnitude across values, their broadly positive pattern shows that matched image-side information can substantially improve linguistic access on held-out images.

![](images/8185e8b33b9af51e1b36c85e153828972d9c5ab41b119eb42dcf0481cce7b840.jpg)  
Figure 5: Visual contrast matters. Matched contrast outperforms prototypeonly and mismatched-direction controls.

Specificity of the visual contrast. The observed access gain raises a natural question: does it depend on the matched target-versus-rest visual contrast, or is moving the text query toward the target visual prototype sufficient? We test the latter with prototype-only grounding,

$$
q _ { v } ^ { + } ( \lambda ) = \mathrm { n o r m } \left( ( 1 - \lambda ) q _ { v } + \lambda \mu _ { v } ^ { I } \right) ,
$$

with λ selected on validation and fixed before test.

On SigLIP2 Base, prototype-only grounding yields modest gains over the native query but remains consis tently below matched grounding (Fig. 5). Moreover, assigning each target a nonmatching visual contrast sharply degrades retrieval, even when averaged over all nonmatching assignments. Together, these controls isolate what drives the access gain. Rather than reflecting generic attraction toward target images, the improvement depends specifically on the matched visual direction that separates the target from its alternatives. This interpretation is also consistent with the geometric guarantee, which applies to movement along the matched target-versus-rest visual contrast.

Prompt and lexical controls. The native query already averages three fixed prompt embeddings, reducing its dependence on any single wording. We nevertheless test whether the observed access gain depends on this ensemble or on particular prompt formulations. In the pattern setting, native mAP ranges from 0.501 to 0.629 across five fixed templates, whereas matched grounding improves performance for every template and remains between 0.872 and 0.881 (Table 1). Thus, the access gain is not tied to the three-template ensemble or to any one tested formula

Table 1: Prompt robustness in the pattern setting. Matched grounding improves across all tested query forms.
<table><tr><td colspan="4">Query form</td><td>Native Matched</td></tr><tr><td>{v}</td><td></td><td></td><td>.629</td><td>.876</td></tr><tr><td></td><td>a {v} surface</td><td></td><td>.549</td><td>.874</td></tr><tr><td></td><td></td><td>an object with a {v} pattern</td><td>.577</td><td>.881</td></tr><tr><td></td><td></td><td>a surface that is {v}</td><td>.528</td><td>.874</td></tr><tr><td></td><td></td><td>a textured {v} surface</td><td>.501</td><td>.872</td></tr></table>

We also ask whether lexical substitution alone can close the access gap. For each pattern, we form an equal-weight ensemble of the canonical term and two predetermined lexical alternatives. This lexical ensemble reaches 0.512 mAP, compared with 0.635 for the original native ensemble, and therefore does not reproduce the improvement from matched grounding. Together, these controls indicate that the observed access gains are not explained by prompt formulation or lexical alternatives. The prompt-robustness result also holds for hue and shape, with full results reported in Appendix C.3.

![](images/5d29a12fc35e5ea47df69c80b08f4deab256a2ec17f3f7e1434a5e5ea4d02289.jpg)  
Figure 6: Matched access gain under attenuated visual evidence. Matched access gain remains substantial as the pattern signal weakens and disappears when the visual distinction is removed.

Dependence on visual evidence. We further extend the analysis by varying the strength of the underlying visual distinction for the pattern factor, where the visual signal can be directly controlled. Each patterned image is progressively blended with its nuisance-matched plain counterpart, producing seven signal strengths from $\gamma = 1 . 0 \mathrm { t o } \gamma = 0 . 0$ . At each strength, visual directions are re-estimated from calibration images and the grounding strength α is selected on validation.

As shown in Fig. 6, as the visual signal weakens, the available image-side contrast also decreases, and retrieval performance under matched grounding declines accordingly. Even at $\gamma = 0 . 1$ , however, retrieval with matchedgrounded queries reaches 0.659 mAP, compared with 0.416 for native-text retrieval, while the estimated visual directions retain a mean cosine of 0.825 with their full-signal counterparts. $\mathbf { A } \mathbf { t } \gamma = 0$ , all pattern variants collapse to nuisance-matched plain images, and grounding provides no gain over chance. These results indicate that matched access gains depend on the presence of the distinction-specific visual contrast itself. Full results are provided in Appendix C.4.

Table 2: Matched access gain across factors, backbones, and held-out protocols. Each cell reports native → matched full-gallery mAP and factor-value top-1 accuracy. Per-backbone rows use the context holdout; the lower block reports seven-backbone macro results across additional protocols.
<table><tr><td></td><td colspan="2">Shape</td><td colspan="2">Hue</td><td colspan="2"></td><td colspan="2">Pattern</td></tr><tr><td></td><td>mAP</td><td>Top-1</td><td></td><td>mAP</td><td>Top-1</td><td>mAP</td><td></td><td>Top-1</td></tr><tr><td colspan="2">Context holdout, per backbone</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>CLIP family</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>CLIP ViT-B/16</td><td>.519 → .723</td><td>.543 → .690</td><td></td><td>.649 → .809</td><td>.583 → .836</td><td>.361 → .830</td><td></td><td>.425 → .844</td></tr><tr><td>EVA02-B/16</td><td>.563 → .804</td><td>.645 → .767</td><td></td><td>.586 → .796</td><td>.656 → .841</td><td>.515 → .843</td><td></td><td>.544 → .884</td></tr><tr><td>FG-CLIP Base</td><td>.599 → .740</td><td>.585 → .706</td><td></td><td>.678 → .892</td><td>.720 → .901</td><td>.555 → .848</td><td></td><td>.496 → .882</td></tr><tr><td>SigLIP family</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>SigLIP SO400M</td><td>.730 → .850</td><td>.741 → .835</td><td></td><td>.667 → .825</td><td>.627 → .856</td><td>.663 → .891</td><td></td><td>.718 → .911</td></tr><tr><td>SigLIP2 Base</td><td>.702 → .813</td><td>.692 → .789</td><td></td><td>.678 → .854</td><td>.603 → .882</td><td>.635 → .878</td><td></td><td>.676 → .911</td></tr><tr><td>SigLIP2 Large</td><td>.744 → .837</td><td>.770 → .830</td><td></td><td>.590 → .736</td><td>.573 → .756</td><td>.620 → .875</td><td></td><td>.653 → .927</td></tr><tr><td>SigLIP2 SO400M</td><td>.764 → .851</td><td>.774 → .838</td><td></td><td>.488 → .592</td><td>.432 → .620</td><td>.621 → .867</td><td></td><td>.676 → .914</td></tr><tr><td colspan="2">Seven-backbone macro, per protocol</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Context holdout</td><td>.660 → .803</td><td>.678 → .779</td><td></td><td>.619 → .786</td><td>.599 → .813</td><td>.567 → .862</td><td></td><td>.598 → .896</td></tr><tr><td>Unseen combinations</td><td>.667 → .814</td><td>.690 → .798</td><td></td><td>.618 → .801</td><td>.592 → .839</td><td>.594 → .876</td><td></td><td>.619 → .906</td></tr><tr><td>24 nuisance partitions</td><td>.663 → .809</td><td>.687 → .791</td><td></td><td>.621 → .788</td><td>.596 → .818</td><td>.570 → .862</td><td></td><td>.603 → .898</td></tr></table>

## 3.3 ROBUSTNESS ACROSS MODELS AND HELD-OUT STRUCTURE

Across backbones and factors. We broaden the context-holdout audit to cover shape, hue, and pattern across seven frozen vision–language backbones. Matched visual grounding improves both mAP and factor-value top-1 for all 21 backbone–factor combinations. Though the gains vary in magnitude across models and factors, the improvement is consistent in every case.

Unseen semantic combinations. We also test a different kind of holdout based on semantic combinations. Here, individual factor values remain present in calibration, validation, and test, but some combinations of the two non-target factors appear in only one split. For example, when pattern is the target, evaluation uses hue–shape combinations that were not seen when the pattern directions were built or tuned. By design, all nuisance contexts and renderer seeds appear in every split, so the evaluation isolates unseen semantic combinations rather than unseen nuisance conditions. Matched visual grounding improves both metrics for all 21 backbone–factor combinations under this protocol. The access gains therefore do not depend on memorizing particular semantic combinations and generalize to new combinations of otherwise familiar factor values.

Robustness to nuisance partitions. The context holdout used in our main analysis shows broad gains across backbone–factor combinations, but a single nuisance split could still be unusually favorable. We therefore repeat the audit across 24 balanced partitions of the joint context–renderer realizations, assigning eight groups to calibration, four to validation, and twelve to test. Visual directions are re-estimated separately for every partition. Matched visual grounding improves both mAP and factor-value top-1 across all backbone–factor–partition configurations, with little variation across partitions. This consistency indicates that the observed access gains are robust to how the available nuisance realizations are divided among calibration, validation, and test.

Taken together, these results show that matched access gains are not confined to a particular backbone– factor setting or held-out protocol. Complete backbone-, protocol-, and value-resolved results are reported in Appendix D.

## 3.4 GLOBAL ALIGNMENT AND RESIDUAL ACCESS GAPS

We close this analysis by asking whether the access gaps identified for individual visual distinctions persist after a broader correction to the text–image representation geometry. Such broad approaches are common in prior work on fine-grained cross-modal alignment, where correspondence is improved through shared transformations applied across many distinctions rather than distinction-specific corrections. We test this using full-rank linear maps on the text representation, similar in spirit to prior work on linear cross-modal alignment (Koishigarina et al., 2026; Moayeri et al., 2023).

![](images/14252df1fc892162b223a52ec97e8e93e008e4a8c9d9dae16fa9fcba5c473e59.jpg)

![](images/db2172b3d0ad7fa12cd436a661a1c6e6cdd9f53e09bb7d4718e21cf919242e34.jpg)  
Factor global + Matched grounding Matched alone  
Figure 7: Global alignment leaves residual access gaps. (Left) Factor-level macro-mAP under the context holdout. (Right) Factor-global alignment before and after matched grounding across seven backbones, with native-space matched grounding shown for reference.

For a native query q, the aligned query is

$$
q ^ { G } = \operatorname { n o r m } ( A q ) ,\tag{3}
$$

where $A \in \mathbb { R } ^ { d \times d }$ is a bias-free, identity-initialized map and the pretrained encoders remain fixed. We learn either a separate map $A _ { f }$ for each factor or a single map $A _ { \mathrm { s h a r e d } }$ across all three factors. Both map types are trained on calibration image–caption pairs with factor-replacement hard negatives and selected on validation before test evaluation. We use these maps as expressive baselines for broad cross-modal alignment.

Global alignment improves overall retrieval but leaves substantial residual access gaps across factors. Under the context holdout, seven-backbone macro-mAP rises from 0.616 natively to 0.682 with shared-global alignment and 0.716 with factor-global alignment. Both maps improve pattern and shape mAP but reduce hue mAP, whereas matched visual grounding achieves the highest mAP on all three factors (Fig. 7, left).

We next test whether the value-specific visual structure remains useful even after broad alignment. To examine this, we fix each global map and use matched visual directions estimated from calibration images to ground the aligned queries, with the grounding strength reselected on validation. Factorglobal alignment provides the stronger broad correction, yet matched visual grounding further raises macro-mAP from 0.716 to 0.806 and yields additional access gains across all seven backbones (Fig. 7, right). The same overall trend holds across additional global alignment baselines and broader held-out evaluations (Appendix E). Thus, even after a broad correction to text–image alignment, matched grounding that targets the value-specific visual contrast can further reduce the remaining access gaps.

## 4 CONSEQUENCES AND GENERALIZATION OF IMPROVED ACCESS

Having established gaps between visual discriminability and linguistic addressability, together with consistent gains from matched visual grounding, we next turn to the broader consequences of improved access and its generalization beyond the controlled setting.

## 4.1 FROM FACTOR ACCESS TO COMPOSITIONAL RETRIEVAL

Matched grounding is local to the visual contrast associated with each factor value, so improved access to an individual factor value does not by itself establish that the resulting queries remain compatible when several distinctions must be satisfied jointly. We therefore evaluate exact composition, where a query specifies one pattern, one hue, and one shape and the correct image must match all three values simultaneously. The exact target must outrank near-miss images that differ in one or more factors.

Following prior work showing the utility of factorized inference for compositional retrieval (Alshehri et al., 2026), we score each requested factor value within its own vocabulary before combining the resulting factor-level evidence. Native and matched queries use the same composition rule, so the comparison isolates the effect of improved factor-level access rather than differences in the composition operator. Specifically, for each factor $f , p _ { f } ( v \mid x )$ denotes the softmax-normalized similarity of image x to value v within that factor vocabulary, using a shared temperature τ.

![](images/d44f27088e80929006816ceb8a98ee4835ee1f79c7b7db53a814c72daa919990.jpg)  
Figure 8: Factor-level access gains support exact compositional retrieval. Native and matched queries are compared under the context holdout and three unseen-combination protocols. Matched visual grounding improves exact R@1, R@5, and mAP across all held-out conditions. H×S, P×S, and P×H denote held-out hue–shape, pattern–shape, and pattern–hue combinations.

For each factor $f ,$ we define

$$
p _ { f } ( v \mid x ) = \frac { \exp ( \langle q _ { f , v } , x \rangle / \tau ) } { \sum _ { u \in V _ { f } } \exp ( \langle q _ { f , u } , x \rangle / \tau ) } , \qquad S ( y , x ) = \sum _ { f \in \{ \mathrm { s h a p e , h u c , p a t t e r n } \} } w _ { f } \log p _ { f } ( y _ { f } \mid x ) .
$$

Each factor-specific grounding strength $\alpha _ { f }$ is chosen on factor-level validation and then fixed. The shared temperature $\tau$ and factor weights $w _ { f }$ are chosen on compositional validation to balance how factor-level evidence is combined before test evaluation.

In this setting, matched visual grounding substantially improves exact retrieval across all held-out conditions and metrics (Fig. 8). Under the context holdout, R@1 rises from 0.357 to 0.711, R@5 from 0.714 to 0.962, and mAP from 0.517 to 0.819. The gains also persist under unseen-combination protocols, where individual factor values remain familiar but their test-time combinations are new. The factor-level access gains therefore carry over to exact compositional retrieval, enabling new multi-factor combinations to be retrieved jointly rather than only improving isolated factor readouts.

## 4.2 VALIDATION-GUIDED SELECTIVE GROUNDING

Because access gains vary across individual factor values, the valuelevel audit can also guide where grounding should be applied. Some native queries are already effective, while others show clear validation gains from matched grounding. We therefore retain grounding only for values whose validation AP gain remains positive under bootstrap resampling of nuisance groups; otherwise, we keep the native query. A single grounding strength is selected per factor on validation, and all decisions are fixed before test evaluation.

On SigLIP2 Base, across the context-holdout and unseen hue–shapecombination protocols, grounding every value raises mean test mAP from 0.6797 to 0.8220, while the validation-guided policy reaches

Table 3: Validation-guided selective grounding. Test mAP across the context and unseen-combination protocols.
<table><tr><td>Policy</td><td>Test mAP</td></tr><tr><td>Native</td><td>.6797</td></tr><tr><td>Ground every value</td><td>.8220</td></tr><tr><td>Validation-guided</td><td>.8245</td></tr></table>

0.8245. Treating each value-specific visual distinction separately enables more targeted grounding, applying it only where validation supports a gain while yielding a small additional improvement.

## 4.3 MATCHED ACCESS GAINS BEYOND FACTORATLAS

Table 4: Natural-image transfer settings. We evaluate texture, garment pattern and length, and material distinctions using dataset-specific splits that separate evaluation examples from those used to estimate and tune the grounding.
<table><tr><td>Dataset</td><td>Target distinction</td><td>Held-out setting</td></tr><tr><td>DTD (Cimpoi et al., 2014)</td><td>Texture</td><td>Official partitions</td></tr><tr><td>Fashionpedia (Jia et al., 2020)</td><td>Pattern / garment length</td><td>Category-held-out / image-disjoint</td></tr><tr><td>COCO-Facet (Li et al., 2026)</td><td>Material</td><td>Image-disjoint</td></tr><tr><td>UT-Zappos (Yu &amp; Grauman, 2014)</td><td>Material</td><td>Product-disjoint</td></tr></table>

![](images/6cc07fd8efd0d09152abe2ef6628d0025b842d0ec43d63b0ed6f8bc0268c9d3e.jpg)  
Figure 9: Matched Access Gain beyond FactorAtlas. Seven-backbone macro mAP improves across natural texture, garment, and material distinctions.

We finally evaluate matched visual grounding beyond the controlled FactorAtlas setting using natural images. To do so, we draw on datasets that support fine-grained vocabularies for visual factors such as texture, garment pattern and length, and material, and construct controlled text-to-image retrieval settings with dataset-specific splits (Table 4). Despite substantial within-dataset variation in how target values are visually realized, matched grounding yields consistent access gains across the evaluated settings, with mAP improving from 0.262 to 0.369 for Fashionpedia garment length and from 0.624 to 0.860 for COCO-Facet material when averaged across seven backbones (Fig. 9). These results show that the diagnosis and intervention developed in the controlled setting also carry over to natural images.

We further examine whether visual contrasts derived in FactorAtlas can themselves transfer to natural images. For the four pattern values shared with Fashionpedia, directly applying the corresponding contrasts and grounding strengths raises mAP from 0.583 to 0.719 across seven backbones, without target-domain fitting or validation. This suggests that visual structure derived for a well-defined distinction in a controlled setting can remain useful when the same distinction recurs in natura images. Full results and protocols are reported in Appendix H.

## 5 RELATION TO PRIOR WORK

Fine-grained and compositional vision–language retrieval. Vision–language models such as CLIP and SigLIP enable retrieval and zero-shot recognition through cross-modal similarity (Radford et al., 2021; Zhai et al., 2023). Yet benchmarks such as Winoground, VL-CheckList, SugarCrepe, and COLA reveal limitations of native vision–language matching on fine-grained attributes, relations, binding, and composition (Thrush et al., 2022; Zhao et al., 2022; Hsieh et al., 2023; Ray et al., 2023). Recent work has further examined these challenges in multi-condition retrieval, attribute binding, and factorized inference (Chow et al., 2026; Lu et al., 2026; Zhang et al., 2025; Alshehri et al., 2026). Complementing these lines of work, we ask whether weak retrieval for a named visual distinction reflects limited image-side discriminability or limited access through the native text query.

Representation, hidden structure, and linguistic access. Shared VLM embedding spaces exhibit meaningful semantic structure across image and text representations (Papadimitriou et al., 2025). On the image side in particular, frozen embeddings preserve compositional organization and ordinal directions identifiable from visual examples (Berasi et al., 2025; Sonthalia et al., 2026). Yet informa tion preserved within a modality may remain weakly accessible through native image–text similarity when image and text representations are not well aligned (Koishigarina et al., 2026). Building on this gap, we take each named visual distinction as the unit of analysis, enabling a finer-grained audit of the relation between image-side structure and linguistic access.

Cross-modal alignment and distinction-specific access. Prior work studies broad cross-modal mis match (Liang et al., 2022) and approaches for improving image–text correspondence through learned mappings, structure-preserving alignment, or stronger modality-specific representations (Moayeri et al., 2023; Eslami & De Melo, 2024; Gröger et al., 2026; Gong et al., 2025; Huang et al., 2026). Among these approaches, LABCLIP improves attribute–object correspondence through a learned transformation shared across text embeddings (Koishigarina et al., 2026). Support-based adaptation provides another related direction, using labeled visual examples to improve downstream performance with frozen vision–language models (Zhang et al., 2021). In our approach, we focus on distinction-specific access, deriving a matched target-versus-rest contrast for each visual value and using it to adjust the native text query on held-out data. This value-specific intervention tests whether the corresponding image-side structure can reduce the access gap, including after global alignment.

## 6 CONCLUSION

A visual distinction can remain discriminable in frozen image geometry while being only weakly addressed by its native text query, so retrieval failure need not imply that the corresponding distinction is absent from the visual representation. Matched visual grounding provides a held-out intervention: when the relevant image-side contrast is present, using that contrast can improve linguistic access and reduce the observed access gap.

This interpretation is supported by direction-specific and visual-evidence controls, robustness across models and held-out conditions, and residual gains after global alignment. Improved access to individual factor values supports exact compositional retrieval and validation-guided selective ground ing, while the same matched-grounding logic extends to natural-image distinctions. More broadly, strong aggregate cross-modal alignment does not by itself ensure reliable linguistic access to every distinction that remains discriminable in image geometry.

These conclusions are conditioned on visual discriminability. Access gains vary across settings, the geometric guarantee is local, and the global maps cover only one family of broader alignment corrections. Within this scope, what frozen image geometry can reliably distinguish and what the native language interface can reliably address should be treated as separate empirical questions. When a distinction remains visually discriminable but linguistically under-addressed, matched visual grounding provides a held-out intervention that tests this diagnosis and can improve linguistic access. Extending this perspective beyond predefined distinctions and labeled visual support could enable vision–language systems to more flexibly expose and use visual structure that their representations already support, even when that structure is not well captured by the native language interface.

## REFERENCES

Sultan Alshehri, Zhantao Yang, Han Zhang, and Marios Savvides. Similarity is not logic: Factored inference for dual-encoder vision-language models. arXiv preprint arXiv:2607.23052, 2026.

Davide Berasi, Matteo Farina, Massimiliano Mancini, Elisa Ricci, and Nicola Strisciuglio. Not only text: Exploring compositionality of visual representations in vision-language models. In 2025 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pp. 24917–24927. IEEE, 2025.

Holger Caesar, Jasper Uijlings, and Vittorio Ferrari. Coco-stuff: Thing and stuff classes in context. In 2018 IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 1209–1218. IEEE, 2018.

Wei Chow, Yuan Gao, Linfeng Li, Xian Wang, Qi Xu, Hang Song, Lingdong Kong, Ran Zhou, Yi Zeng, Yidong Cai, et al. Merit: Multilingual semantic retrieval with interleaved multi-condition query. Advances in Neural Information Processing Systems, 38:74806–74867, 2026.

Mircea Cimpoi, Subhransu Maji, Iasonas Kokkinos, Sammy Mohamed, and Andrea Vedaldi. Describing textures in the wild. In Proceedings of the IEEE conference on computer vision and pattern recognition, pp. 3606–3613, 2014.

Sedigheh Eslami and Gerard De Melo. Mitigate the gap: Investigating approaches for improving cross-modal alignment in clip. arXiv preprint arXiv:2406.17639, 2024.

Federico Girella, Davide Talon, Ziyue Liu, Zanxi Ruan, Yiming Wang, and Marco Cristani. Lots of fashion! multi-conditioning for image generation via sketch-text pairing. In 2025 IEEE/CVF International Conference on Computer Vision (ICCV), pp. 19711–19720. IEEE, 2025.

Shizhan Gong, Yankai Jiang, Qi Dou, and Farzan Farnia. Kernel-based unsupervised embedding alignment for enhanced visual representation in vision-language models. arXiv preprint arXiv:2506.02557, 2025.

Fabian Gröger, Shuo Wen, Huyen Le, and Maria Brbic. With limited data for multimodal alignment, let the structure guide you. Advances in Neural Information Processing Systems, 38:151747– 151776, 2026.

Cheng-Yu Hsieh, Jieyu Zhang, Zixian Ma, Aniruddha Kembhavi, and Ranjay Krishna. Sugarcrepe: Fixing hackable benchmarks for vision-language compositionality. Advances in neural information processing systems, 36:31096–31116, 2023.

Weiquan Huang, Aoqi Wu, Yifan Yang, Xufang Luo, Yuqing Yang, Usman Naseem, Chunyu Wang, Qi Dai, Xiyang Dai, Dongdong Chen, et al. Llm2clip: Powerful language model unlocks richer cross-modality representation. In Proceedings of the AAAI Conference on Artificial Intelligence, volume 40, pp. 5131–5139, 2026.

Menglin Jia, Mengyun Shi, Mikhail Sirotenko, Yin Cui, Claire Cardie, Bharath Hariharan, Hartwig Adam, and Serge Belongie. Fashionpedia: Ontology, segmentation, and an attribute localization dataset. In European conference on computer vision, pp. 316–332. Springer, 2020.

Amita Kamath, Jack Hessel, and Kai-Wei Chang. Text encoders bottleneck compositionality in contrastive vision-language models. In Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing, pp. 4933–4944, 2023.

Darina Koishigarina, Arnas Uselis, and Seong Joon Oh. Clip behaves like a bag-of-words model cross-modally but not uni-modally. In International Conference on Learning Representations, volume 2026, pp. 96106–96127, 2026.

Siting Li, Xiang Gao, and Simon Du. Highlighting what matters: Promptable embeddings for attributefocused image retrieval. Advances in Neural Information Processing Systems, 38:115077–115110, 2026.

Victor Weixin Liang, Yuhui Zhang, Yongchan Kwon, Serena Yeung, and James Y Zou. Mind the gap: Understanding the modality gap in multi-modal contrastive representation learning. Advances in neural information processing systems, 35:17612–17625, 2022.

Chenxi Liao, Masataka Sawayama, and Bei Xiao. Probing the link between vision and language in material perception using psychophysics and unsupervised learning. PLOS Computational Biology, 20(10):e1012481, 2024.

Xuan Lu, Kangle Li, Haohang Huang, Rui Meng, Wenjun Zeng, and Xiaoyu Shen. Beyond global similarity: Towards fine-grained, multi-condition multimodal retrieval. arXiv preprint arXiv:2603.01082, 2026.

Mazda Moayeri, Keivan Rezaei, Maziar Sanjabi, and Soheil Feizi. Text-to-concept (and back) via cross-model alignment. In International Conference on Machine Learning, pp. 25037–25060. PMLR, 2023.

Isabel Papadimitriou, Huangyuan Su, Thomas Fel, Sham Kakade, and Stephanie Gil. Interpreting the linear structure of vision-language model embedding spaces. arXiv preprint arXiv:2504.11695, 2025.

Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, et al. Learning transferable visual models from natural language supervision. In International conference on machine learning, pp. 8748–8763. PmLR, 2021.

Arijit Ray, Filip Radenovic, Abhimanyu Dubey, Bryan Plummer, Ranjay Krishna, and Kate Saenko. Cola: A benchmark for compositional text-to-image retrieval. Advances in Neural Information Processing Systems, 36:46433–46445, 2023.

Linda Smith and Chen Yu. Infants rapidly learn word-referent mappings via cross-situational statistics. Cognition, 106(3):1558–1568, 2008.

Ankit Sonthalia, Arnas Uselis, and Seong Joon Oh. On the rankability of visual embeddings. Advances in Neural Information Processing Systems, 38:66169–66203, 2026.

Quan Sun, Yuxin Fang, Ledell Wu, Xinlong Wang, and Yue Cao. Eva-clip: Improved training techniques for clip at scale. arXiv preprint arXiv:2303.15389, 2023.

Tristan Thrush, Ryan Jiang, Max Bartolo, Amanpreet Singh, Adina Williams, Douwe Kiela, and Candace Ross. Winoground: Probing vision and language models for visio-linguistic compositionality. In 2022 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pp. 5228–5238. IEEE, 2022.

Michael Tschannen, Alexey Gritsenko, Xiao Wang, Muhammad Ferjad Naeem, Ibrahim Alabdulmohsin, Nikhil Parthasarathy, Talfan Evans, Lucas Beyer, Ye Xia, Basil Mustafa, et al. Siglip 2: Multilingual vision-language encoders with improved semantic understanding, localization, and dense features. arXiv preprint arXiv:2502.14786, 2025.

Chunyu Xie, Bin Wang, Fanjing Kong, Jincheng Li, Dawei Liang, Gengshen Zhang, Dawei Leng, and Yuhui Yin. Fg-clip: Fine-grained visual and textual alignment. arXiv preprint arXiv:2505.05071, 2025.

A. Yu and K. Grauman. Fine-Grained Visual Comparisons with Local Learning. In Computer Vision and Pattern Recognition (CVPR), June 2014.

Chen Yu and Linda B Smith. Rapid word learning under uncertainty via cross-situational statistics. Psychological science, 18(5):414–420, 2007.

Xiaohua Zhai, Basil Mustafa, Alexander Kolesnikov, and Lucas Beyer. Sigmoid loss for language image pre-training. In 2023 IEEE/CVF International Conference on Computer Vision (ICCV), pp. 11941–11952. IEEE, 2023.

Qi Zhang, Yuxu Chen, Lei Deng, and Lili Shen. Abe-clip: Training-free attribute binding enhancement for compositional image-text matching. arXiv preprint arXiv:2512.17178, 2025.

Renrui Zhang, Rongyao Fang, Wei Zhang, Peng Gao, Kunchang Li, Jifeng Dai, Yu Qiao, and Hongsheng Li. Tip-adapter: Training-free clip-adapter for better vision-language modeling. arXiv preprint arXiv:2111.03930, 2021.

Tiancheng Zhao, Tianqi Zhang, Mingwei Zhu, Haozhan Shen, Kyusong Lee, Xiaopeng Lu, and Jianwei Yin. Vl-checklist: Evaluating pre-trained vision-language models with objects, attributes and relations. arXiv preprint arXiv:2207.00221, 2022.

## A FULL PROOF OF THE GEOMETRIC GUARANTEE

Let $\| q \| = \| d \| = 1 , \Delta = \| \Delta \| d .$ , and define

$$
q _ { \alpha } = \frac { q + \alpha d } { \| q + \alpha d \| }
$$

whenever $q +$ αd $\neq 0 .$ . For any unit query r, define the matched score gap

$$
M ( r ) = r ^ { \top } \Delta .
$$

Proposition. For every $\alpha \geq 0$ for which $q _ { \alpha }$ is defined,

$$
M ( q _ { \alpha } ) \geq M ( q ) .
$$

Proof. Let $c = q ^ { \top } d \in [ - 1 , 1 ]$ . Since

$$
\| q + \alpha d \| ^ { 2 } = 1 + 2 \alpha c + \alpha ^ { 2 } ,
$$

we obtain

$$
M ( q _ { \alpha } ) = \| \Delta \| \frac { c + \alpha } { \sqrt { 1 + 2 \alpha c + \alpha ^ { 2 } } } .
$$

For $| c | < 1$

$$
\frac { d } { d \alpha } M ( q _ { \alpha } ) = \lVert \Delta \rVert \frac { 1 - c ^ { 2 } } { ( 1 + 2 \alpha c + \alpha ^ { 2 } ) ^ { 3 / 2 } } > 0 ,
$$

so the matched score gap increases strictly with α.

For $c = 1$ , we have $q = d ,$ hence $q _ { \alpha } = d$ for every $\alpha \geq 0$ , and therefore

$$
\begin{array} { r } { M ( q _ { \alpha } ) = M ( q ) = \| \Delta \| . } \end{array}
$$

For $c = - 1 , q = - d$ and

$$
q _ { \alpha } = \left\{ { \begin{array} { l l } { - d , } & { 0 \leq \alpha < 1 , } \\ { { \mathrm { u n d e f i n e d } } , } & { \alpha = 1 , } \\ { d , } & { \alpha > 1 , } \end{array} } \right.
$$

so

$$
M ( q _ { \alpha } ) = \left\{ { \begin{array} { l l } { - \| \Delta \| , } & { 0 \leq \alpha < 1 , } \\ { + \| \Delta \| , } & { \alpha > 1 . } \end{array} } \right.
$$

Thus $M ( q _ { \alpha } ) \geq M ( q )$ for every defined $\alpha \geq 0$

Negative initial gap. $\mathrm { I f } - 1 < c < 0 .$ , then $M ( q ) < 0$ , and the grounded score gap crosses zero at

$$
\alpha ^ { \star } = - c = - q ^ { \top } d .
$$

Hence

$$
M ( q _ { \alpha ^ { \star } } ) = 0 , \qquad M ( q _ { \alpha } ) > 0 \quad \mathrm { f o r } \alpha > \alpha ^ { \star } .
$$

The singular endpoint $c = - 1$ is excluded from this continuous crossing statement.

Scope. The guarantee concerns only the matched calibration target-versus-average-rest gap. It does not imply monotonic improvement in held-out retrieval, preservation of unrelated factors, or successful composition; these are evaluated empirically.

## B FACTORATLAS CONSTRUCTION AND EVALUATION PROTOCOL

FactorAtlas is the complete product of 10 patterns, 12 hues, 8 shapes, 12 nuisance contexts, and two renderer replicas, totaling 23,040 images. Contexts 0–3 are calibration (7,680 images), 4–5 validation (3,840), and 6–11 test (11,520); both replicas of a context remain in the same split. Contexts vary pose, viewpoint, illumination, background, roughness, and material appearance. Images are rendered at 224 × 224 using Cycles. Camera elevation is sampled in [0.14, 0.98] radians, camera distance in [3.65, 6.35], lens in [42, 66], object pitch and roll in [−0.72, 0.72], yaw in [0, 2π], world strength in [0.08, 0.72], and material roughness in [0.25, 0.90]. Each scene uses three randomized area lights. Calibration constructs prototypes and visual contrasts; validation selects all hyperparameters; test is evaluated after those choices are fixed.

The native query is the normalized mean of three prompts: {v}, a {v} surface, and an object with a {v} pattern for pattern; {v}, a {v} object, and a {v} surface for hue; and {v}, a {v}, and an object shaped like a {v} for shape. α ∈ $\{ 0 , . 0 5 , \ldots , 8 \}$ is selected by validation macro-mAP, breaking ties by factor-value top-1 and then smaller α. Macro-mAP averages per-value full-gallery AP; factor-value top-1 is an image-wise K-way readout and is distinct from retrieval R@1.

All encoders are frozen. We use official preprocessing for CLIP ViT-B/16 (Radford et al., 2021), EVA02-B/16 (Sun et al., 2023), FG-CLIP Base (Xie et al., 2025), SigLIP SO400M (Zhai et al., 2023), SigLIP2 Base, SigLIP2 Large, and SigLIP2 SO400M (Tschannen et al., 2025). Exact checkpoints are respectively openai/clip-vit-base-patch16, OpenCLIP EVA02-B-16 with merged2b\_s8b\_b131k, qihoo360/fg-clip-base, google/siglip-so400m-patch14-384, google/siglip2-base-patch16-224, google/siglip2-large-patch16-256, and google/siglip2-so400m-patch16-384.

## C FULL DIAGNOSTIC CONTROLS

## C.1 INTERVENTION-FREE SEPARATION

We begin with the comparison that does not modify the text query. For each factor value, the image-prototype query is the normalized centroid estimated only from labeled calibration images, whereas the native query is produced by the frozen text encoder from the prompt ensemble specified in Appendix B. Both queries retrieve the same held-out test gallery. The comparison therefore asks whether a distinction remains discriminable in calibration-derived image geometry even when the frozen text interface addresses it less reliably; it is not a comparison between two trained classifiers.

Table 5: Intervention-free factor-value top-1.
<table><tr><td></td><td>Pattern</td><td>Hue</td><td>Shape</td></tr><tr><td>SigLIP2 Base</td><td></td><td></td><td></td></tr><tr><td>Image prototype</td><td>.919</td><td>.909</td><td>.825</td></tr><tr><td>Native text</td><td>.676</td><td>.603</td><td>.692</td></tr><tr><td colspan="4">Seven-backbone macro</td></tr><tr><td>Image prototype</td><td>.907</td><td>.835</td><td>.815</td></tr><tr><td>Native text</td><td>.598</td><td>.599</td><td>.678</td></tr></table>

Table 5 reports the image-wise factor-value top-1 readout. The image prototype exceeds the native text query for pattern, hue, and shape on SigLIP2 Base, and the same ordering remains after averaging the seven backbones. The gap is largest for pattern in the seven-backbone macro (.907 versus .598), but is also present for hue and shape. This is the intervention-free evidence that visual discriminability can exceed native linguistic addressability; matched grounding has not yet been applied in this table.

## C.2 SPECIFICITY OF THE VISUAL DISTINCTION

The main results raise a stronger question: is the target-versus-rest visual distinction itself important, or is it sufficient to pull the text query toward images of the correct value? We test the latter possibility

with target-prototype attraction. For each value v, we interpolate the native text query toward its calibration-derived target image centroid:

$$
q _ { v } ^ { + } ( \lambda ) = \mathrm { n o r m } \left( ( 1 - \lambda ) q _ { v } + \lambda \mu _ { v } ^ { I } \right) .
$$

This baseline uses the correct target-image support but does not contrast the target value against the remaining values in its factor vocabulary. By comparison, target-versus-rest grounding uses the direction

$$
d _ { v } ^ { I } = \mathrm { n o r m } \left( \mu _ { v } ^ { I } - \bar { \mu } _ { - v } ^ { I } \right) ,
$$

which explicitly represents the visual distinction between the target value and its alternatives.

For each backbone and factor, a single λ shared by all values is selected from $\{ 0 , 0 . 0 1 2 5 , \ldots , 1 \}$ using validation mAP and then fixed before test evaluation. The grounding strength α is selected independently on the same validation split. Both methods use the same calibration support, native queries, held-out gallery, and evaluation metric.

Table 6: Target attraction versus target-versus-rest visual contrast. Each factor cell reports native → target-prototype attraction → target-versus-rest grounding mAP under the primary context holdout. The final column reports the validation-selected attraction strengths for pattern, hue, and shape.
<table><tr><td>Backbone</td><td>Pattern</td><td></td><td>Hue</td><td></td><td></td><td>Shape</td><td>λP/H/S</td></tr><tr><td>CLIP ViT-B/16</td><td>.361 → .568 → .830</td><td></td><td>.649 → .656 → .809</td><td></td><td></td><td>.519 → .584 → .723</td><td>.713/.163/.500</td></tr><tr><td>EVA02-B/16</td><td>.515 → .643 → .843</td><td></td><td>.586 → .595 → .796</td><td></td><td></td><td>.563 → .649 → .804</td><td>.525/.150/.488</td></tr><tr><td>FG-CLIP Base</td><td>.555 → .689 → .848</td><td></td><td>.678 → .708 → .892</td><td></td><td></td><td>.599 → .653 → .740</td><td>.700/.363/.500</td></tr><tr><td>SigLIP SO400M</td><td>.663 → .745 → .891</td><td></td><td>.667 → .683 → .825</td><td></td><td></td><td>.730 → .765 → .850</td><td>.613/.175/.538</td></tr><tr><td>SigLIP2 Base</td><td>.635 → .716 → .878</td><td></td><td>.678 → .688 → .854</td><td></td><td></td><td>.702 → .726 → .813</td><td>.425/.100/.400</td></tr><tr><td>SigLIP2 Large</td><td>.620 → .732 → .875</td><td></td><td>.590 → .597 → .736</td><td></td><td></td><td>.744 → .756 → .837</td><td>.450/.113/.213</td></tr><tr><td>SigLIP2 SO400M</td><td>.621 → .726 → .867</td><td></td><td>.488 → .497 → .592</td><td></td><td></td><td>.764 → .772 → .851</td><td>.463/.100/.138</td></tr><tr><td>Seven-backbone macro</td><td>.567 → .689 → .862</td><td></td><td>.619 → .632 → .786</td><td></td><td></td><td>.660 → .701 → .803</td><td></td></tr></table>

Target-prototype attraction improves over native retrieval in all 21 backbone–factor cases, showing that target attraction alone can yield gains. However, target-versus-rest grounding performs better than prototype attraction in all 21 cases. Across factors and backbones, macro-mAP increases from .616 under native retrieval to .674 with prototype attraction and .817 with target-versus-rest grounding. Thus, target attraction alone does not explain the observed access gain; explicitly isolating the visual distinction between each value and its alternatives yields substantially larger held-out gains. This held-out result is separate from, but consistent with, the local geometric guarantee for movement along the matched target-versus-average-rest calibration gap.

## C.3 PROMPT AND LEXICAL CONTROLS

We test whether matched access gain can be explained by the wording of the native text query. The detailed tables below use the frozen SigLIP2 Base backbone and the primary context holdout. Visual directions are estimated from calibration contexts 0–3, grounding strength is selected on validation contexts 4–5, and all results are reported on test contexts 6–11.

## C.3.1 FIXED PROMPT TEMPLATES

For each factor, we define five grammatical templates that express the same factor value in different ways. Every template is applied uniformly to the complete factor vocabulary; no template is selected separately for individual values. Table 7 lists the complete template bank.

For each factor–template combination, we compare the native query with its matched-grounded counterpart. A single factor-level α is selected on validation separately for each template and fixed before test evaluation. Table 8 reports all 15 test endpoints. Matched grounding improves over the corresponding native query in every case.

The validation-selected strengths for T1–T5 are [.95, 1.00, .90, 1.05, 1.40] for pattern, [.80, .80, .85, .65, 1.05] for hue, and [2.40, 2.70, 3.00, 2.95, 3.55] for shape. The result is therefore not tied to any of the tested query forms. Across all seven audited backbones, matched grounding improves all $7 \times 3 \times 5 = 1 0 5$ backbone–factor–template endpoints under the same per-template evaluation.

Table 7: Fixed factor-specific prompt templates. The placeholder {v} is replaced by every value in the corresponding factor vocabulary.
<table><tr><td>ID</td><td>Pattern</td><td></td><td></td><td></td><td></td><td>Hue</td><td></td><td></td><td>Shape</td><td></td></tr><tr><td>T1</td><td>{v}</td><td></td><td></td><td></td><td></td><td>{v}</td><td></td><td></td><td>{v}</td><td></td></tr><tr><td>T2</td><td></td><td>a {v} surface</td><td></td><td></td><td></td><td></td><td> $\begin{array} { r l r } { \textsf { a } } & { { } \{ \mathrm { ~ v ~ } \} } & { \mathrm { o b j e c t } } \end{array}$ </td><td></td><td></td><td> $a \quad \{ \tau \}$ </td></tr><tr><td>T3</td><td></td><td>an object with a {v} pattern</td><td></td><td></td><td></td><td></td><td></td><td>an object that is {v}</td><td></td><td>an object shaped like a {v}</td></tr><tr><td>T4</td><td></td><td>a surface that is {v}</td><td></td><td></td><td></td><td></td><td></td><td>a {v} colored object</td><td></td><td>a geometric {v}</td></tr><tr><td>T5</td><td></td><td>a textured {v} surface</td><td></td><td></td><td></td><td></td><td>a {v} surface</td><td></td><td></td><td>a three dimensional {v} object</td></tr></table>

Table 8: Prompt robustness on SigLIP2 Base. Each entry reports native → matched test mAP.
<table><tr><td>Template</td><td>Pattern</td><td>Hue</td><td>Shape</td></tr><tr><td>T1</td><td> $. 6 2 9 \to . 8 7 6$ </td><td> $. 6 5 7  . 8 4 6$ </td><td> $. 7 0 4  . 8 1 3$ </td></tr><tr><td>T2</td><td> $. 5 4 9 \to . 8 7 4$ </td><td> $. 6 3 6 \to . 8 5 3$ </td><td> $. 6 9 6 \to . 8 1 3$ </td></tr><tr><td>T3</td><td> $. 5 7 7 \to . 8 8 1$ </td><td> $. 6 4 6 \to . 8 5 3$ </td><td> $. 6 8 5  . 8 1 3$ </td></tr><tr><td>T4</td><td> $. 5 2 8 \to . 8 7 4$ </td><td> $. 6 5 6 \to . 8 6 2$ </td><td> $. 6 4 2  . 8 1 3$ </td></tr><tr><td>T5</td><td> $. 5 0 1  . 8 7 2$ </td><td> $. 5 8 5  . 8 4 6$ </td><td> $. 6 5 9  . 8 1 3$ </td></tr><tr><td>Improved</td><td>5/5</td><td>5/5</td><td>5/5</td></tr></table>

## C.3.2 PATTERN LEXICAL SUBSTITUTION

We separately test whether lexical substitution alone can close the observed pattern access gap. For each pattern, we construct an equal-weight ensemble from three predetermined lexical forms using the template a TERM surface. No term is selected using validation or test performance. This experiment does not apply matched grounding to the lexical variants; it tests whether replacing the words alone can reproduce the observed access gain.

Table 9: Pattern lexical-substitution control. The canonical native ensemble is compared with an equal-weight ensemble of three fixed lexical forms.
<table><tr><td>Value</td><td>Additional lexical forms</td><td>Canonical AP</td><td>Lexical AP</td></tr><tr><td rowspan="5">Plain Striped Wavy Checkered Dotted</td><td>solid, unpatterned</td><td>.287</td><td>.109</td></tr><tr><td>banded, lined</td><td>.375</td><td>.367</td></tr><tr><td>undulating, wave patterned</td><td>.666</td><td>.558</td></tr><tr><td>checkerboard, chequered polka dotted, dot patterned</td><td>.999 .711</td><td>.976</td></tr><tr><td>fissured, crackle patterned</td><td>.936</td><td>.490 .744</td></tr><tr><td>Cracked Mottled</td><td>blotchy, variegated</td><td>.669</td><td>.345</td></tr><tr><td>Speckled</td><td>flecked, peppered</td><td>.071</td><td>.084</td></tr><tr><td>Marbled</td><td>veined, marble patterned</td><td>.815</td><td>.687</td></tr><tr><td>Brick</td><td>brickwork, brick patterned</td><td>.815</td><td>.758</td></tr><tr><td colspan="2">Macro mAP Label top-1</td><td>.635</td><td>.512</td></tr></table>

Lexical substitution decreases aggregate mAP from 0.635 to 0.512 and decreases AP for nine of the ten pattern values. Together, the two controls show that matched access gain persists across the tested factor-specific prompt formulations, while lexical substitution alone does not reproduce it.

## C.4 VISUAL-EVIDENCE DEPENDENCE

The preceding controls establish direction specificity, but labeled calibration support could still appear to produce gains independently of a repeatable visual contrast. To test dependence on that signal, each patterned image is blended with a nuisance-matched plain rendering of the same scene. Thus pose, viewpoint, illumination, shape, hue, and background are held fixed while the pattern evidence is progressively attenuated. A strength of $\gamma = 1$ is the original patterned image, and $\gamma = 0$ is the matched plain image. Visual directions are reconstructed from calibration and α is reselected on validation independently at every strength, so intermediate points do not reuse the full-signal direction.

Table 10: Pattern attenuation on SigLIP2 Base. Performance is mAP; cosine is measured against the full-signal directions.
<table><tr><td>γ</td><td>Image query</td><td>Native</td><td>Matched</td><td>Cosine</td></tr><tr><td>1.00</td><td>.649</td><td>.635</td><td>.878</td><td>1.000</td></tr><tr><td>.70</td><td>.609</td><td>.611</td><td>.863</td><td>.995</td></tr><tr><td>.50</td><td>.567</td><td>.590</td><td>.845</td><td>.981</td></tr><tr><td>.35</td><td>.518</td><td>.566</td><td>.824</td><td>.962</td></tr><tr><td>.20</td><td>.423</td><td>.519</td><td>.774</td><td>.920</td></tr><tr><td>.10</td><td>.269</td><td>.416</td><td>.659</td><td>.825</td></tr><tr><td>.00</td><td>.100</td><td>.100</td><td>.100</td><td>-.001</td></tr></table>

As the signal weakens, image-side discriminability and the absolute level of retrieval achieved by matched grounding decline rather than failing abruptly. $\mathrm { { A t } } \gamma = . 1 0$ , the image-side direction is weaker but remains aligned with its full-signal counterpart (mean cosine .825), and matched mAP remains .659 compared with native .416. $\mathrm { A t } \gamma = 0$ , images with different pattern labels are identical within each nuisance-matched group, all three readouts reach the ten-way chance value of .100, and the matched access gain disappears. The three readouts use different queries and therefore are not upper bounds on one another; in particular, matched mAP may exceed image-prototype mAP. The relevant result is their shared dependence on a repeatable visual contrast: weak but stable evidence can support matched access gains, whereas labels alone cannot do so after that contrast is removed.

## C.5 FINITE GROUNDING AND THE PURE VISUAL ENDPOINT

Our main analysis treats matched visual grounding as an intervention on the native text query. The native query $q _ { v }$ is moved along the value-specific visual contrast $d _ { v } ^ { I } .$ , with a factor-level grounding strength selected on validation. A possible limiting interpretation is that the resulting gains arise simply because the grounded query approaches an image-derived visual readout, effectively discarding the native text query.

We therefore characterize the relation between finite matched grounding and this limiting endpoint. This is not a separate adaptation method or an additional leaderboard comparison. Rather, it is an endpoint analysis of the same intervention family defined in Eq. (1).

The intervention path. Recall that matched grounding is defined as

$$
q _ { v } ^ { \prime } ( \alpha ) = \mathrm { n o r m } \bigl ( q _ { v } + \alpha d _ { v } ^ { I } \bigr ) , \qquad \alpha \geq 0 .\tag{4}
$$

The two endpoints are

$$
q _ { v } ^ { \prime } ( 0 ) = q _ { v }\tag{5}
$$

and

$$
\operatorname* { l i m } _ { \alpha \to \infty } q _ { v } ^ { \prime } ( \alpha ) = d _ { v } ^ { I } .\tag{6}
$$

Thus, the native text query, the validation-selected finite grounded query, and the pure visual direction lie on the same intervention path:

$$
\underbrace { q _ { v } } _ { \alpha = 0 \colon \mathrm { ~ n a t i v e ~ t e x t } } \quad \longrightarrow \quad \underbrace { q _ { v } ^ { \prime } ( \widehat { \alpha } ) } _ { \mathrm { ~ v a l i d a t i o n - s e l e c t e d ~ f i n i t e ~ g r o u n d i n g ~ } } \quad \longrightarrow \quad \underbrace { d _ { v } ^ { I } } _ { \alpha \to \infty \colon \mathrm { p u r e ~ v i s u a l ~ e n d p o i n t } } .\tag{7}
$$

The pure visual direction $d _ { v } ^ { I }$ is therefore the limiting endpoint of the same query family rather than an unrelated baseline.

Geometric location of a finite grounded query. The grounding strength also has a simple geometric interpretation. Let q and d be unit vectors with $q \neq - d$ , and let $\alpha \geq 0$ . Define

$$
q _ { \alpha } = \frac { q + \alpha d } { \| q + \alpha d \| }\tag{8}
$$

and let

$$
c = q ^ { \top } d .\tag{9}
$$

Then

$$
q _ { \alpha } ^ { \top } q = \frac { 1 + \alpha c } { \sqrt { 1 + \alpha ^ { 2 } + 2 \alpha c } } ,\tag{10}
$$

while

$$
q _ { \alpha } ^ { \top } d = \frac { c + \alpha } { \sqrt { 1 + \alpha ^ { 2 } + 2 \alpha c } } .\tag{11}
$$

Subtracting gives

$$
q _ { \alpha } ^ { \top } q - q _ { \alpha } ^ { \top } d = \frac { ( 1 - \alpha ) ( 1 - c ) } { \sqrt { 1 + \alpha ^ { 2 } + 2 \alpha c } } .\tag{12}
$$

Proposition. Let $q$ and d be distinct, non-antipodal unit vectors and let

$$
q _ { \alpha } = \operatorname { n o r m } ( q + \alpha d ) .\tag{13}
$$

Then $q _ { \alpha }$ is closer in cosine similarity to $q$ than to $d$ whenever $0 \leq \alpha < 1$ . It is equally close to the two directions at $\alpha = 1$ . It is closer in cosine similarity to d than to q whenever $\alpha > 1$

Proof. Because $q \neq d ,$ we have $c = q ^ { \top } d < 1$ . Therefore, $1 - c > 0$ . The denominator in Eq. 12 is positive whenever $q _ { \alpha }$ is defined. Hence, the sign of $q _ { \alpha } ^ { \top } q - q _ { \alpha } ^ { \top }$ d is determined entirely by $1 - \alpha$ . The three cases follow immediately. □

This proposition describes only the geometric position of the grounded query along the intervention path. It does not assign a fraction of retrieval performance to the text or visual component. Accordingly, α should not be interpreted as an attribution weight.

Empirical behavior under limited calibration support. We compare finite grounding with the corresponding pure visual endpoint as the amount of calibration support used to estimate the visual contrast varies. For each factor value, we subsample $n \in \{ 1 , 2 , 4 , 8 , \bar { 1 6 } , 3 2 , 6 4 , 1 2 8 , 2 5 6 \}$ calibration images. Each budget uses 12 balanced resamples; visual directions are re-estimated and α is independently selected on the unchanged validation split for every resample before test evaluation.

As shown in Fig. 10, finite grounding and direction-only retrieval differ substantially at small support, particularly for hue and pattern, and the gap generally narrows as support increases. From one to 256 images per value, direction-only mAP increases from .524 to .811 for shape, .187 to .813 for hue, and .372 to .867 for pattern; the corresponding finite-grounding trajectories are .718 to .812, .679 to .841, and .672 to .876.

The selected α also changes with support. At one image per value, mean α is .10, .03, and .06 for shape, hue, and pattern; at 256 images per value, it reaches 2.78, .53, and .74. Thus, validation selects only a small movement from the native query at very limited support, with factor-dependent movement along the intervention path as additional calibration evidence becomes available.

For completeness, the resampled sweep stops at 256 images per value so that every plotted point follows the same 12-resample protocol. Using the complete calibration support, finite/direction-only mAP is .813/.812 for shape, .854/.833 for hue, and .878/.870 for pattern, with selected α values 2.90, .65, and .70, respectively.

Scope. This analysis characterizes the intervention path rather than attributing retrieval performance to separate linguistic and visual components. Because both the estimated visual direction and the validation-selected α vary with calibration support, α should not be interpreted as a mixture weight or as the fraction of performance contributed by either component. The comparison instead shows that finite grounding and the pure visual endpoint need not behave interchangeably across support regimes.

![](images/cbe1f2c166e0780b1859d18861eab5a6a793fa949a1e8470144b38cbe2029423.jpg)  
Figure 10: Finite grounding and the pure visual endpoint under limited calibration support. Top: test mAP for direction-only retrieval and validation-selected finite grounding. Bottom: corresponding selected grounding strength α. Points and shading show mean ±1 standard deviation over 12 balanced calibration resamples; the dashed line denotes native-text retrieval.

## D BACKBONE-, PROTOCOL-, AND VALUE-RESOLVED READOUTS

This appendix expands the breadth results summarized in Table 2. Factor-level mAP is the macro average of one-vs-rest AP over the complete factor vocabulary. Factor-value top-1 is image-wise classification accuracy obtained by assigning each image to its highest-scoring value query. It is distinct from the exact-composition retrieval R@1 reported later.

## D.1 UNSEEN FACTOR COMBINATIONS

The unseen-combination evaluation withholds combinations of the two non-target factors while keeping every individual value represented in calibration. Pattern is evaluated on unseen hue–shape cells, hue on unseen pattern–shape cells, and shape on unseen pattern–hue cells. Visual directions and grounding strengths are re-estimated independently for each backbone and held-out protocol.

Table 11: Per-backbone matched access gains on unseen factor combinations. Each entry reports native → matched.
<table><tr><td></td><td colspan="2">Pattern</td><td colspan="4"></td><td colspan="4">Shape</td></tr><tr><td>Backbone</td><td>mAP</td><td>Top-1</td><td></td><td>mAP</td><td></td><td>Top-1</td><td></td><td>mAP</td><td></td><td>Top-1</td></tr><tr><td>CLIP ViT-B/16</td><td>.371 → .838</td><td>.434 → .856</td><td></td><td>.650 → .820</td><td></td><td>.579 → .856</td><td>.520 → .729</td><td></td><td></td><td>.549 → .707</td></tr><tr><td>EVA02-B/16</td><td>.533 → .856</td><td>.562 → .888</td><td></td><td>.578 → .816</td><td></td><td>.645 → .841</td><td>.579 → .817</td><td></td><td></td><td>.660 → .787</td></tr><tr><td>FG-CLIP Base</td><td>.587 → .864</td><td>.529 → .910</td><td></td><td>.698 → .906</td><td></td><td>.710 → .950</td><td>.606 → .746</td><td></td><td></td><td>.602 → .736</td></tr><tr><td>SigLIP SO400M</td><td>.691 → .908</td><td>.739 → .921</td><td></td><td>.662 → .838</td><td></td><td>.613 → .908</td><td>.735 → .862</td><td></td><td></td><td>.756 → .847</td></tr><tr><td>SigLIP2 Base</td><td>.673 → .895</td><td>.709 → .924</td><td></td><td>.672 → .872</td><td></td><td>.601 → .910</td><td></td><td>.710 → .832</td><td></td><td>.695 → .810</td></tr><tr><td>SigLIP2 Large</td><td>.651 → .888</td><td>.669 → .924</td><td></td><td>.576 → .745</td><td></td><td>.570 → .766</td><td></td><td>.753 → .853</td><td></td><td>.788 → .846</td></tr><tr><td>SigLIP2 SO400M</td><td>.649 → .884</td><td>.689 → .922</td><td></td><td>.491 → .613</td><td></td><td>.426 → .641</td><td></td><td>.769 → .861</td><td>.782 → .848</td><td></td></tr><tr><td>Macro</td><td>.594 → .876</td><td>.619 → .906</td><td></td><td>.618 → .801</td><td></td><td>.592 → .839</td><td>.667 → .814</td><td></td><td>.690 → .798</td><td></td></tr></table>

Matched grounding improves both metrics for all 21 target-aligned backbone–factor evaluations. This result tests generalization to unseen combinations of observed semantic values rather than transfer to unseen nuisance contexts.

## D.2 BALANCED NUISANCE PARTITIONS

We construct 24 deterministic balanced partitions of the joint context–renderer realizations. Each partition assigns eight groups to calibration, four to validation, and twelve to testing. Visual directions are estimated from calibration data and the factor-level grounding strength is selected on validation independently within every fold.

Table 12: Matched access gains across 24 balanced nuisance partitions. Values are mean ± standard deviation across folds after averaging the seven backbones within each fold.
<table><tr><td>Factor</td><td>mAP: native → matched</td><td>Top-1: native → matched</td><td>Positive endpoints</td></tr><tr><td>Pattern</td><td>.5701±.0014 → .8618±.0005</td><td>.6031±.0019 → .8977±.0008</td><td>168/168</td></tr><tr><td>Hue</td><td>.6207±.0026 → .7877±.0032</td><td>.5961±.0019 → .8182±.0040</td><td>168/168</td></tr><tr><td>Shape</td><td>.6630±.0026 → .8093±.0030</td><td>.6871±.0031 → .7905±.0049</td><td>168/168</td></tr></table>

Each factor contributes 7 backbones × 24 folds = 168 endpoints. Matched grounding improves both mAP and top-1 at every endpoint. The small fold-level variation indicates robustness to split choice within the FactorAtlas nuisance generator; it should not be interpreted as transfer to an unrelated renderer distribution.

## D.3 VALUE-RESOLVED ACCESS GAINS

Aggregate factor scores can hide whether access gains are shared broadly across the vocabulary or dominated by a few values. We therefore recompute AP separately for every one of the 10 pattern, 12 hue, and 8 shape values, and then average each value over the seven backbones. Native and matched entries use the same primary context holdout; only the query changes.

Factor-level improvement need not be uniform over individual values. Table 13 reports per-value AP averaged across the seven backbones under the primary context holdout.

Table 13: Value-resolved AP under the primary context holdout. Each entry reports the sevenbackbone macro native → matched AP and its change. Access gains are broad but value-dependent; teal is the only value with a negative macro change in this protocol.
<table><tr><td colspan="3">Pattern</td><td colspan="3">Hue</td><td colspan="3">Shape</td></tr><tr><td>Value</td><td>Native → Matched</td><td>∆</td><td>Value</td><td>Native → Matched</td><td>∆</td><td>Value</td><td>Native → Matched</td><td>∆</td></tr><tr><td>plain</td><td>.227 → .770</td><td>+.542</td><td>red</td><td>.825 → .898</td><td>+.072</td><td>cube</td><td>.551 → .823</td><td>+.271</td></tr><tr><td>striped</td><td>.372 → .854</td><td>+.482</td><td>orange</td><td>.789 → .923</td><td>+.133</td><td>sphere</td><td>.895 → .978</td><td>+.083</td></tr><tr><td>wavy</td><td>.592 → .763</td><td>+.171</td><td>yellow</td><td>.767 → .886</td><td>+.119</td><td>cylinder</td><td>.832 → .908</td><td>+.076</td></tr><tr><td>checkered</td><td>.975 → .998</td><td>+.023</td><td>lime</td><td>.669 → .878</td><td>+.209</td><td>cone</td><td>.753 → .915</td><td>+.161</td></tr><tr><td>dotted</td><td>.643 → .884</td><td>+.241</td><td>green</td><td>.623 → .842</td><td>+.218</td><td>torus</td><td>.885 → .999</td><td>+.114</td></tr><tr><td>cracked</td><td>.865 → .996</td><td>+.131</td><td>teal</td><td>.606 → .594</td><td>-.011</td><td>pyramid</td><td>.514 → .649</td><td>+.135</td></tr><tr><td>mottled</td><td>.517 → .920</td><td>+.403</td><td>cyan</td><td>.445 → .619</td><td>+.174</td><td>triangular prism</td><td>.261 → .466</td><td>+.205</td></tr><tr><td>speckled</td><td>.077 → .547</td><td>+.469</td><td>blue</td><td>.630 → .897</td><td>+.267</td><td>octahedron</td><td>.589 → .683</td><td>+.094</td></tr><tr><td>marbled</td><td>.726 → .931</td><td>+.205</td><td>indigo</td><td>.379 → .758</td><td>+.379</td><td></td><td></td><td></td></tr><tr><td>brick</td><td>.677 → .953</td><td>+.276</td><td>purple</td><td>.577 → .729</td><td>+.152</td><td></td><td></td><td></td></tr><tr><td></td><td></td><td></td><td>magenta</td><td>.695 → .852</td><td>+.157</td><td></td><td></td><td></td></tr><tr><td></td><td></td><td></td><td>pink</td><td>.427 → .558</td><td>+.131</td><td></td><td></td><td></td></tr></table>

The table shows that access gains are distributed across the vocabularies rather than arising from a single value. Large gains occur for values such as plain, striped, speckled, indigo, and cube, while values already near ceiling, such as checkered and torus, leave less room for improvement. Teal is the sole negative seven-backbone macro endpoint in this protocol. We therefore interpret the result as broad factor-level access gains, not as a claim that every individual value must improve.

## E GLOBAL ALIGNMENT AND ALTERNATIVE ADAPTATION

## E.1 FULL-RANK LEARNED ALIGNMENT

The first global comparison asks whether the observed access gap can be removed by learning one expressive transformation over a whole vocabulary, rather than by grounding values individually. This baseline receives direct factor supervision and is retrained for every held-out protocol, making it a target-aware broad correction rather than a zero-shot transfer map.

The global baseline applies $q ^ { G } = \operatorname { n o r m } ( A q )$ with a bias-free full-rank map initialized to identity. Encoders remain frozen. We train one shared map or one map per factor from calibration image– caption pairs, using cyclic factor replacements as hard negatives. Training alternates a {pattern} {hue} {shape} object and a {hue} {shape} with a {pattern} surface. We use AdamW for 30 epochs, batch size 256, weight decay $1 0 ^ { - 5 }$ , learning rates $\{ 1 0 ^ { - 4 } , 3 \times 1 0 ^ { - 4 } , 1 0 ^ { - 3 } \}$ and seed 20260820. Checkpoints are chosen by validation macro-mAP and top-1. Each held-out protocol is trained separately. Before training, caption-cache rows are asserted against image identifiers and metadata hashes to verify ordering.

Table 14: Primary-context global comparison. Seven-backbone macro-mAP.
<table><tr><td>Factor</td><td>Native</td><td>Shared</td><td>Factor</td><td>Matched</td></tr><tr><td>Pattern</td><td>.567</td><td>.779</td><td>.806</td><td>.862</td></tr><tr><td>Hue</td><td>.619</td><td>.528</td><td>.580</td><td>.786</td></tr><tr><td>Shape</td><td>.660</td><td>.739</td><td>.762</td><td>.803</td></tr><tr><td>Macro</td><td>.616</td><td>.682</td><td>.716</td><td>.817</td></tr></table>

Table 14 compares four query representations on the same primary-context test galleries. Shared alignment uses one map for all three factors, factor alignment uses a separately trained map for each factor, and matched grounding uses the value-specific image contrast without learning a full-rank map. The global maps substantially improve pattern and shape. Their effect is not uniform, however, and both global maps reduce hue mAP relative to native retrieval in this split. Matched grounding has the highest macro result and the highest result for each factor. The point is not that global alignment fails, but that a broad correction does not uniformly eliminate value-specific access gaps.

## E.2 ADDITIONAL WHOLE-VOCABULARY LINEAR MAPS

We also compare closed-form maps trained from whole-vocabulary centroid correspondences. A factor-specific map uses the K native text queries and K calibration image centroids of one factor; a shared map uses all 30 pattern, hue, and shape correspondences. Validation selects among ridge $\lambda \in \{ 1 0 ^ { - 4 } , \dot { 1 } 0 ^ { - 3 } , 1 0 ^ { - 2 } , \dot { 1 } 0 ^ { - 1 } , 1 , 1 0 , 1 0 0 \}$ , truncated low-rank ridge with ranks $\{ 1 , 2 , 4 , K - \bar { 1 } \}$ (or 29 for the shared map), and row-span orthogonal Procrustes. Each candidate is blended with the identity using weights $\{ 0 , . 1 2 5 , . . . , 1 \}$ . Selection uses validation macro-mAP, then top-1, then the smaller map blend.

Table 15: Additional whole-vocabulary maps on SigLIP2 Base. Values are primary-context test mAP. Validation selects Procrustes for all map scopes in this split.
<table><tr><td>Method</td><td>Pattern</td><td>Hue</td><td>Shape</td></tr><tr><td>Native</td><td>.635</td><td>.678</td><td>.702</td></tr><tr><td>Factor-specific map</td><td>.772</td><td>.727</td><td>.789</td></tr><tr><td>Shared map</td><td>.775</td><td>.712</td><td>.768</td></tr><tr><td>Matched grounding</td><td>.878</td><td>.854</td><td>.813</td></tr></table>

These closed-form baselines test whether the conclusion depends on the optimization details of the full-rank learned map. Although validation chooses Procrustes in this particular split, the candidate family also includes ridge and low-rank solutions and is selected without test data. Both factorspecific and shared maps improve over native retrieval, but neither reaches matched grounding on any of the three factors in Table 15. This provides a second, independently fitted whole-vocabulary comparison rather than treating a single global learner as definitive.

## E.3 POST-GLOBAL RESIDUAL ACCESS GAINS

The direct comparison above still leaves open whether global alignment and matched grounding address the same part of the mismatch. We therefore apply them sequentially. For post-global evaluation, the selected global map is first frozen. Image directions are then recomputed from calibration data, and a new factor-level α is selected on validation. The global map is never updated using the residual result, and test data are used only for the final evaluation. A positive gain therefore means that value-matched image-side structure remains useful for improving access after the broad transformation has already been applied.

Table 16: Residual access gains after global alignment. Macro-mAP averages context and three unseen-combination protocols.
<table><tr><td colspan="5">Shared global</td><td colspan="2">Factor global</td></tr><tr><td>Backbone</td><td>Global</td><td>+Matched</td><td>Improved</td><td>Global</td><td>+Matched</td><td>Improved</td></tr><tr><td>CLIP B/16</td><td>.680</td><td>.743</td><td>11/12</td><td>.735</td><td>.765</td><td>10/12</td></tr><tr><td>EVA02-B/16</td><td>.725</td><td>.794</td><td>10/12</td><td>.738</td><td>.791</td><td>10/12</td></tr><tr><td>FG-CLIP Base</td><td>.708</td><td>.785</td><td>10/12</td><td>.753</td><td>.797</td><td>8/12</td></tr><tr><td>SigLIP SO400M</td><td>.764</td><td>.823</td><td>9/12</td><td>.775</td><td>.827</td><td>9/12</td></tr><tr><td>SigLIP2 Base</td><td>.740</td><td>.813</td><td>11/12</td><td>.753</td><td>.816</td><td>11/12</td></tr><tr><td>SigLIP2 Large</td><td>.758</td><td>.794</td><td>7/12</td><td>.762</td><td>.795</td><td>7/12</td></tr><tr><td>SigLIP2 SO400M</td><td>.699</td><td>.747</td><td>9/12</td><td>.717</td><td>.752</td><td>8/12</td></tr><tr><td>Macro</td><td>.725</td><td>.785</td><td>67/84</td><td>.748</td><td>.792</td><td>63/84</td></tr></table>

The table averages four held-out conditions, with all three target factors evaluated under each condition, giving 12 endpoints per backbone. Adding matched grounding raises the shared-global macro from .725 to .785 and the factor-global macro from .748 to .792. The gain is positive for 67/84 and 63/84 individual endpoints, respectively, so it is strong but not universal. In the primary context holdout alone, adding matched grounding yields positive gains for all 21 factor–backbone endpoints: shared rises from .682 to .802 and factor global from .716 to .806. These residual gains show that the tested broad maps leave residual value-specific access gaps that matched visual grounding can further reduce.

## E.4 FULL-SUPPORT CACHE ADAPTATION

We additionally test whether comparable held-out gains can be obtained by directly retrieving through the labeled calibration support. We implement a training-free Tip-Adapter-style cache (Zhang et al., 2021) using all calibration images, while keeping the image and text encoders frozen. This is a fullsupport cache control rather than a canonical few-shot reproduction of Tip-Adapter or its fine-tuned variant.

Let K contain the normalized calibration image embeddings and let L contain class-count-normalized factor-wise one-hot labels. For a normalized test-image embedding z, the cache score is

$$
\begin{array} { r } { s _ { \mathrm { c a c h e } } ( z ) = \exp \bigl ( \beta ( z K ^ { \intercal } - 1 ) \bigr ) L . } \end{array}
$$

The final score combines the native text similarity with the cache score:

$$
s _ { \mathrm { T i p } } ( z ) = s _ { \mathrm { n a t i v e } } ( z ) + \eta s _ { \mathrm { c a c h e } } ( z ) .
$$

The cache contains all 7,680 calibration images, corresponding to 768 examples per pattern value, 640 per hue value, and 960 per shape value. The affinity sharpness $\beta$ and mixture strength η are selected using validation mAP, with factor-value top-1 used as a tie-breaker. We also evaluate the cache-only and nearest-support limits to ensure that the selected result is not caused by a truncated hyperparameter range. All choices are fixed before evaluation on the primary-context test gallery.

The full-support cache yields substantial gains over native retrieval for every backbone and factor. Target-versus-rest grounding nevertheless achieves higher full-gallery mAP in all 21 backbone–factor cases. Averaged across all factors and backbones, mAP increases from .616 under native retrieval to .755 with the Tip-Adapter-style cache and .817 with target-versus-rest grounding. Thus, labeled support can itself yield substantial gains through cache-based adaptation, but the matched targetversus-rest intervention yields larger full-gallery gains without retaining the calibration set as an inference-time cache.

Table 17: Full-support cache adaptation across backbones and factors. Each factor cell reports native → Tip-Adapter-style cache → target-versus-rest grounding full-gallery mAP under the primary context holdout. All hyperparameters are selected on validation and fixed before test evaluation.
<table><tr><td>Backbone</td><td>Pattern</td><td>Hue</td><td>Shape</td></tr><tr><td>CLIP ViT-B/16</td><td>.361 → .800 → .830</td><td>.649 → .703 → .809</td><td>.519 → .677 → .723</td></tr><tr><td>EVA02-B/16</td><td>.515 → .797 → .843</td><td>.586 → .644 → .796</td><td>.563 → .740 → .804</td></tr><tr><td>FG-CLIP Base</td><td>.555 → .827 → .848</td><td>.678 → .767 → .892</td><td>.599 → .688 → .740</td></tr><tr><td>SigLIP SO400M</td><td>.663 → .847 → .891</td><td>.667 → .753 → .825</td><td>.730 → .827 → .850</td></tr><tr><td>SigLIP2 Base</td><td>.635 → .841 → .878</td><td>.678 → .723 → .854</td><td>.702 → .760 → .813</td></tr><tr><td>SigLIP2 Large</td><td>.620 → .822 → .875</td><td>.590 → .657 → .736</td><td>.744 → .786 → .837</td></tr><tr><td>SigLIP2 SO400M</td><td>.621 → .821 → .867</td><td>.488 → .560 → .592</td><td>.764 → .815 → .851</td></tr><tr><td>Seven-backbone macro</td><td> $. 5 6 7 \to . 8 2 2 \to . 8 6 2$ </td><td> $. 6 1 9 \to . 6 8 7 \to . 7 8 6$ </td><td> $. 6 6 0 \to . 7 5 6 \to . 8 0 3$ </td></tr></table>

## F EXACT COMPOSITION DETAILS

## F.1 COMMON FACTOR-POSTERIOR READOUT

Factor-level mAP evaluates one semantic axis at a time and does not show that the exact shape–hue– pattern target outranks near-miss images that differ in one or more factors. We therefore evaluate exact compositions using the same readout for native and matched queries, isolating the effect of improved factor-level access.

For each context–renderer-seed pair in the evaluation split, we form a separate retrieval gallery. Under the context holdout, each gallery contains all 960 shape–hue–pattern triples exactly once. For unseen-pair protocols, evaluation is restricted to triples in the corresponding held-out factor-pair cells. We average over all valid query–gallery pairs, yielding $6 \times 2 \times 9 6 0 = 1 \bar { 1 } { , } 5 2 0$ instances for the context holdout and $1 2 \times 2 \times 1 2 0 = 2 { , } 8 8 0$ for each unseen-pair protocol.

We use

$$
\log p _ { f } ( v \mid x ) = { \frac { \langle q _ { f , v } , x \rangle } { \tau } } - \log \sum _ { u \in \mathcal { V } _ { f } } \exp \left( { \frac { \langle q _ { f , u } , x \rangle } { \tau } } \right) , \quad S ( y , x ) = \sum _ { f } w _ { f } \log p _ { f } ( y _ { f } \mid x ) .\tag{14}
$$

Factor-level $\alpha _ { f }$ values are fixed first. We select $\tau \in \{ . 0 0 5 , . 0 1 , . 0 2 , . 0 5 , . 1 , . 2 , . 5 \}$ and positive simplex weights on a 0.1 grid using validation exact MRR followed by R@1. Native and matched use the same gallery and composition operator.

Table 18: Exact composition. Seven-backbone macro results.
<table><tr><td></td><td colspan="3">Native</td><td colspan="3">Matched</td></tr><tr><td>Condition</td><td>R@1</td><td>R@5</td><td>mAP</td><td>R@1</td><td>R@5</td><td>mAP</td></tr><tr><td>Context</td><td>.357</td><td>.714</td><td>.517</td><td>.711</td><td>.962</td><td>.819</td></tr><tr><td>Unseen H×S</td><td>.631</td><td>.929</td><td>.764</td><td>.928</td><td>.998</td><td>.960</td></tr><tr><td>Unseen P×S</td><td>.622</td><td>.952</td><td>.766</td><td>.876</td><td>.995</td><td>.931</td></tr><tr><td>Unseen P×H</td><td>.661</td><td>.937</td><td>.779</td><td>.828</td><td>.990</td><td>.900</td></tr></table>

Because each evaluated query has a single exact positive, AP equals the reciprocal rank of that target, so the reported mAP is numerically identical to mean reciprocal rank. Under the context holdout, matched grounding raises exact R@1 from .357 to .711 and R@5 from .714 to .962. Gains persist

for unseen factor-pair combinations: in unseen H×S, for example, pattern queries are evaluated on hue–shape cells withheld from calibration, with P×S and P×H defined analogously. Thus, stronger factor-level access carries over to exact retrieval of new multi-factor combinations.

## G SELECTIVITY AND SAMPLE EFFICIENCY

## G.1 VALIDATION-GUIDED SELECTIVE GROUNDING

Matched grounding need not be applied to every value. Some native queries are already effective, while others show a clear validation gain after grounding. This experiment asks whether held-out audit evidence can select between the native and grounded query for each value without consulting test performance.

We first select one factor-wide grounding strength α on validation. For each value, we then estimate the uncertainty of its validation AP gain using 2,000 bootstrap resamples over context–rendererreplica groups. Grounding is retained only when the lower bound of the resulting 95% confidence interval is positive; otherwise, the native query is kept. All decisions are fixed before test evaluation. Here, the unseen H×S protocol uses the same held-out hue–shape cells when evaluating all three target factors; it is distinct from the target-specific unseen-combination macro reported in the main breadth analysis.

Table 19: Selective grounding on SigLIP2 Base.
<table><tr><td>Protocol</td><td>Factor</td><td>α</td><td>Ground/keep</td><td>Native</td><td>All</td><td>Selective</td><td>Improved/declined</td></tr><tr><td>Context</td><td>Pattern</td><td>.70</td><td>10/0</td><td>.635</td><td>.878</td><td>.878</td><td>10/0</td></tr><tr><td rowspan="5">Unseen H×S</td><td>Hue</td><td>.65</td><td>11/1</td><td>.678</td><td>.854</td><td>.855</td><td>11/0</td></tr><tr><td>Shape</td><td>2.90</td><td>7/1</td><td>.702</td><td>.813</td><td>.813</td><td>7/0</td></tr><tr><td>Pattern</td><td>.70</td><td>9/1</td><td>.673</td><td>.895</td><td>.895</td><td>9/0</td></tr><tr><td>Hue</td><td>.10</td><td>7/5</td><td>.671</td><td>.698</td><td>.697</td><td>6/1</td></tr><tr><td>Shape</td><td>7.35</td><td>7/1</td><td>.721</td><td>.793</td><td>.809</td><td>7/0</td></tr><tr><td>Macro/total</td><td></td><td></td><td>51/9</td><td>.6797</td><td>.8220</td><td>.8245</td><td>50/1</td></tr></table>

Across the two protocols and three factors, the gate grounds 51 of 60 values and retains the native query for nine. Relative to native mAP of .6797, grounding all values reaches .8220 and the validationguided policy reaches .8245. Of the 51 grounded decisions, 50 improve on test and one declines. The small aggregate difference from grounding everything is not the primary point; the result demonstrates that held-out audit evidence can localize where grounding is useful without inspecting test outcomes. We treat this as a consequence of the audit rather than an additional central method claim.

## G.2 LABELED SUPPORT-SIZE SCALING

The main experiments use all available calibration images, but the visual contrast may be estimated from substantially less labeled support. We vary the number of calibration images per value while keeping the validation and test sets unchanged. This experiment measures both matched access gain and the agreement between each estimated direction and its full-support counterpart.

For each value, we sample n ∈ {1, 2, 4, 8, 16, 32, 64, 128, 256} calibration images without replacement and re-estimate its centroid and target-versus-average-rest direction. Sampling is balanced across values, and each finite support budget is evaluated using 12 independent calibration resamples. For every resample, the grounding strength α is selected again on the unchanged validation split. The test set is never sampled or used for selection. One-shot support therefore uses one image per value, corresponding to 10, 12, or 8 total calibration images for pattern, hue, or shape, respectively.

Table 20 reports results for SigLIP2 Base, averaged across the three factors and two held-out protocols, yielding six factor–protocol endpoints. The first protocol is the primary context holdout. The second applies the same held-out hue–shape cells when evaluating all three target factors. Direction cosine measures the cosine similarity between each support-limited direction and the corresponding direction estimated from all calibration images.

Table 20: Matched access gains with increasing labeled support. Finite-budget results average 12 calibration resamples.
<table><tr><td>Images/value</td><td>1</td><td>2</td><td>4</td><td>8</td><td>16</td><td>32</td><td>64</td><td>128</td><td>256</td><td>Full</td></tr><tr><td>Test mAP</td><td>.695</td><td>.711</td><td>.730</td><td>.755</td><td>.774</td><td>.795</td><td>.806</td><td>.812</td><td>.819</td><td>.822</td></tr><tr><td>Gain</td><td>+.015</td><td>+.032</td><td>+.051</td><td>+.076</td><td>+.095</td><td>+.115</td><td>+.127</td><td>+.133</td><td>+.139</td><td>+.142</td></tr><tr><td>Direction cosine</td><td>.466</td><td>.594</td><td>.719</td><td>.825</td><td>.897</td><td>.946</td><td>.972</td><td>.986</td><td>.994</td><td>1.000</td></tr></table>

Table 21: Factor-resolved support-size subset. Entries are test mAP.
<table><tr><td>Factor</td><td>1</td><td>8</td><td>32</td><td>256</td><td>Full</td></tr><tr><td>Pattern</td><td>.688</td><td>.802</td><td>.860</td><td>.884</td><td>.887</td></tr><tr><td>Hue</td><td>.670</td><td>.698</td><td>.731</td><td>.769</td><td>.776</td></tr><tr><td>Shape</td><td>.728</td><td>.767</td><td>.793</td><td>.805</td><td>.803</td></tr></table>

The first table aggregates the six factor–protocol endpoints, whereas Table 21 separates the three factors after averaging the two protocols. Access gains and direction quality increase smoothly with support size. Thirty-two images per value achieve about 81% of the full-support gain, and 64 achieve about 89%. Shape is already close to its full-support mAP at moderate budgets, whereas pattern benefits more strongly from additional support. The positive one-shot mean does not imply a stable one-shot direction: its mean cosine to the full-support direction is only .466. Thus the sweep shows that substantial access gains can be obtained with limited labeled support at the aggregate level, while also showing why the full calibration set is preferable for a stable geometric estimate.

## H NATURAL-IMAGE PROTOCOLS AND FULL RESULTS

## H.1 SHARED EVALUATION PROTOCOL

All natural-image experiments use frozen, normalized image and text embeddings. For each target value, we estimate normalized calibration centroids and construct a normalized target-versus-averagerest visual direction. A single factor-level grounding strength is selected on validation by full-gallery macro-mAP and then fixed before test evaluation. Test images are never used to select prompts, directions, grounding strengths, thresholds, or checkpoints.

Because the natural-image datasets differ in their annotations and grouping structure, we construct dataset-specific calibration, validation, and test partitions while preserving this shared evaluation contract. The primary metric is full-gallery macro-mAP.

## H.2 BACKBONE-RESOLVED RESULTS

Table 22 reports every backbone-level result behind the seven-backbone macro results in the main paper. The columns intentionally retain separate datasets and targets rather than pooling incompatible natural protocols: DTD tests texture classes, Fashionpedia tests garment pattern and length, COCO-Facet tests localized material, and UT-Zappos tests footwear material under product-disjoint splits. Each cell reports native retrieval followed by matched visual grounding; the final row averages backbones within each setting, not examples across datasets.

Matched visual grounding yields positive gains for every evaluated backbone on DTD, Fashionpedia pattern, Fashionpedia length, and COCO-Facet material. It yields positive gains for six of seven backbones on UT-Zappos. The remaining EVA02-B/16 endpoint changes from 0.6653 to 0.6649 and is therefore unchanged at the three-decimal precision used in the paper. Because the 96-product UT-Zappos cohort also exhibits substantial variation across its five product-disjoint rotations, we treat it as a boundary result rather than part of the stable-transfer count.

Table 22: Backbone-resolved natural-image transfer. Each cell reports native → matched fullgallery macro-mAP. Fashionpedia pattern uses the garment-category holdout, and COCO-Facet material uses the natural localized crop. Matched grounding improves all 28 backbone–setting endpoints in the four stable transfer settings; UT-Zappos is reported separately as a higher-variance boundary.
<table><tr><td rowspan="2">Backbone</td><td rowspan="2">DTD Texture</td><td colspan="2">Fashionpedia</td><td rowspan="2">COCO-Facet Material</td><td rowspan="2">UT-Zappos Material</td></tr><tr><td>Pattern</td><td>Length</td></tr><tr><td>CLIP ViT-B/16</td><td>.359 → .668</td><td>.452 → .618</td><td>.178 →.317</td><td>.604 → .850</td><td>.522 → .617</td></tr><tr><td>EVA02-B/16</td><td>.444 → .731</td><td>.567 → .768</td><td>.220 → .361</td><td>.668 → .819</td><td>.665 → .665</td></tr><tr><td>FG-CLIP Base</td><td>.500 → .751</td><td>.566 → .789</td><td>.241 → .352</td><td>.601 → .899</td><td>.631 → .676</td></tr><tr><td>SigLIP SO400M</td><td>.621 → .803</td><td>.631 → .799</td><td>.297 → .396</td><td>.615 → .869</td><td>.677 → .751</td></tr><tr><td>SigLIP2 Base</td><td>.564 → .771</td><td>.620 → .789</td><td>.285 → .370</td><td>.627 → .874</td><td>.678 → .717</td></tr><tr><td>SigLIP2 Large</td><td>.588 → .797</td><td>.620 → .778</td><td>.308 → .403</td><td>.628 → .860</td><td>.662 → .726</td></tr><tr><td>SigLIP2 SO400M</td><td>.607 → .798</td><td>.614 → .793</td><td>.302 → .387</td><td>.623 → .851</td><td>.642 → .724</td></tr><tr><td>Seven-backbone macro</td><td>.526 → .760</td><td>.581 → .762</td><td>.262 → .369</td><td>.624 → .860</td><td>.639 → .697</td></tr></table>

## H.3 DATASET-SPECIFIC PROTOCOLS

DTD texture. We use all 5,640 DTD images and all 47 official texture classes. Each of the ten official partitions provides 40 calibration, 40 validation, and 40 test images per class. The official class token is used as the fixed native query. Directions and grounding strengths are re-estimated independently within each partition, and the table reports performance averaged over the ten partitions.

Fashionpedia garment pattern. We use localized garment crops and fine-grained attributes from Fashionpedia. Each annotated garment box is placed on an aspect-ratio-preserving square canvas and resized to the model input resolution. The pattern vocabulary contains six values: plain, floral, stripe, check, abstract, and dot.

The resulting cohort contains 2,400 crops, balanced over the Cartesian product of six patterns and four garment categories. We perform four leave-one-category-out evaluations. For each run, one garment category supplies the entire test gallery and is excluded from both direction estimation and grounding-strength selection. This tests whether pattern directions estimated from other garment categories transfer to a previously unseen garment category.

Fashionpedia garment length. We separately evaluate an eight-value garment-length vocabulary: mini, micro, above-the-knee, knee, midi, below-the-knee, maxi, andfloor. The cohort contains 2,847 localized dress crops.

Five source-image-disjoint calibration–validation rotations are constructed from the Fashionpedia training set. The untouched Fashionpedia validation gallery is used as test in every rotation. Consequently, variation across rotations measures robustness to the calibration and validation samples rather than uncertainty from different test galleries.

During cohort construction, local English descriptions from LOTS/Sketchy (Girella et al., 2025) were linked to the Fashionpedia examples. However, the evaluated images, garment boxes, category labels, and visual attributes come from Fashionpedia, and retrieval uses fixed factor prompts rather than the LOTS sketch-generation task. We therefore refer to the evaluated dataset as Fashionpedia while recording both sources in the data provenance.

COCO-Facet material. We join official COCO-Facet material labels with COCO-Stuff localization (Caesar et al., 2018). We retain images containing exactly one requested material label and exclude images containing a segment from another requested material class.

For each image, the natural crop is the tight bounding box of the largest individual target-material segment. It is not the union of spatially disconnected segments. We apply fixed inclusion criteria requiring the target mask to occupy at least 1% of the image, fill at least 25% of its bounding box, and have a box covering at most 50% of the full image. We additionally require at least 24 images per material class.

The resulting cohort contains 306 images: 172 metal, 29 stone, and 105 wood. Five deterministic rotations are image-disjoint across calibration, validation, and test. The main-paper result uses the unmodified natural crop.

UT-Zappos material. We evaluate four footwear materials: Leather, Suede, Patent Leather, and Synthetic. The cohort is restricted to a fixed footwear stratum and balances material counts within retained toe-style–heel-height cells.

We retain one deterministic image per product, yielding 96 products with 24 products per material. Products never cross calibration, validation, and test roles. Results are averaged over five productdisjoint rotations.

## H.4 CROSS-DATASET REUSE OF SHARED VISUAL DISTINCTIONS

The matched natural-image experiments above estimate visual contrasts and select grounding strength separately within each target dataset. We additionally ask a stricter question: when two domains share the same visual distinction, can image-side directions estimated in one domain remain useful in the other without any target-side fitting or validation? We test this using the four pattern values shared by FactorAtlas and Fashionpedia.

For each backbone, we construct FactorAtlas pattern directions from the complete ten-value vocabulary using the four calibration contexts and reuse the corresponding grounding strength selected in FactorAtlas. No Fashionpedia image is used to estimate a direction or select the grounding strength. We map FactorAtlas plain, striped, checkered, and dotted to Fashionpedia plain, stripe, check, and dot, respectively.

We evaluate all four Fashionpedia category-held-out partitions. Each held-out gallery contains 600 images spanning all six Fashionpedia pattern labels, so the two unmapped labels remain natural negatives rather than being removed from the gallery. Across seven backbones, directly reusing the FactorAtlas directions improves macro-mAP from .583 to .719 (Table 23). The aggregate gain is positive for all seven backbones and for all 28 backbone–partition evaluations. At the individual backbone–value level, 20 of 28 cases improve, indicating that cross-dataset reuse is consistent in aggregate but remains value-dependent. These results suggest that when the same visual distinction is shared across domains, its image-side direction can remain useful beyond the data from which it was estimated.

Table 23: Cross-dataset reuse of shared pattern distinctions. For each backbone, FactorAtlasderived pattern directions and its fixed FactorAtlas grounding strength are reused without Fashionpedia fitting or validation. Values are four-value macro-mAP averaged over the four Fashionpedia category-held-out partitions.
<table><tr><td>Backbone</td><td>Native</td><td>Reused direction</td><td>∆</td></tr><tr><td>CLIP ViT-B/16</td><td>.414</td><td>.490</td><td>+.075</td></tr><tr><td>EVA02-B/16</td><td>.575</td><td>.720</td><td>+.145</td></tr><tr><td>FG-CLIP Base</td><td>.568</td><td>.800</td><td>+.232</td></tr><tr><td>SigLIP SO400M</td><td>.640</td><td>.767</td><td>+.127</td></tr><tr><td>SigLIP2 Base</td><td>.631</td><td>.795</td><td>+.164</td></tr><tr><td>SigLIP2 Large</td><td>.628</td><td>.750</td><td>+.121</td></tr><tr><td>SigLIP2 SO400M</td><td>.623</td><td>.711</td><td>+.088</td></tr><tr><td>Seven-backbone macro</td><td>.583</td><td>.719</td><td>+.136</td></tr></table>
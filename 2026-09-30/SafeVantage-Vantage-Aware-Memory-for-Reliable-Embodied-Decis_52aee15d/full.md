# SafeVantage: Vantage-Aware Memory for Reliable Embodied Decisions

Sean Hardesty Lewis<sup>1</sup>, Zuyi Guo<sup>2</sup>, Benwang Chen<sup>2</sup>, Zirui Liu<sup>3</sup>, Hongyi Lin<sup>4</sup>, and Heye Huang<sup>2,†</sup>

<sup>1</sup>Massachusetts Institute of Technology <sup>2</sup>Korea Advanced Institute of Science and Technology <sup>3</sup>Nanyang Technological University <sup>4</sup>Tsinghua University

![](images/26afc464be0e9fd2370bfda1f2b060107d197d0f685686d0202d7be25357edab.jpg)  
Fig. 1. Overview of SafeVantage, a vantage-aware memory for reliable embodied decisions. It retains the viewpoints that support each claim and uses the evidence to select the next view and make the final YES/NO/ABSTAIN decisions.

Abstract— Reliable embodied decisions under partial observability require informative observations and sufficient supporting evidence. However, semantic scores alone do not reveal which viewpoints justify a claim or where additional evidence should be acquired. We introduce SafeVantage, a vantage-aware semantic memory and active acquisition framework that retains each claim’s supporting views, camera poses, and estimated target location, keeping positive support distinct from search coverage. A learned candidate-observability model uses claimgrounded geometry to predict target visibility at reachable viewpoints. These predictions guide view selection through expected reduction in terminal decision loss, accounting for travel cost and geometrically distinct corroboration. A calibrated head then combines support, spatial consistency, and coverage to produce YES, NO, or ABSTAIN decisions. We evaluate SafeVantage on a category-presence benchmark spanning 232 unseen ProcTHOR houses and 7,424 paired episodes per method and action budget. Compared with validation-selected equalbudget baselines, SafeVantage achieves macro-F1 gains of 24.7% and 12.0% at eight and twelve actions, respectively, with lower

risk and higher answer rates at both budgets and 31.7% less travel at eight actions. Equal-input HM3D experiments show lower selective risk under fixed observations, while controlled ScanNet interventions show that restoring supporting views improves downstream VLM answers. Ablations further support the contribution of candidate observability to decision quality and acquisition efficiency. Results demonstrate the value of claimlevel viewpoint evidence for connecting semantic memory, active acquisition, and reliable decision-making. Code is available at https://safevantage.github.io

## I. INTRODUCTION

Embodied agents increasingly rely on spatial memory to reason about objects and answer questions in partially observed environments. OpenScene, VLMaps, and Concept-Graphs connect language with persistent 2D or 3D scene representations [1–3], while recent EQA systems combine memory, exploration, and foundation models for embodied reasoning [4–6]. However, reliable decision-making requires more than retrieving a likely semantic match. The memory must also indicate whether the available observations actually provide sufficient evidence for that claim.

Existing approaches organize multi-view observations into maps, graphs, or selected snapshots [2, 3, 7]. These representations support retrieval, but a semantic score alone does not reveal which viewpoints support a claim, whether they are spatially distinct, or how clearly the target was observed. These differences matter under partial observability: a single weak detection provides different evidence from corroborated observations, while missing detections may reflect incomplete search coverage rather than true absence. Embodied systems have used confidence to stop exploration or request help [8, 9]. These limitations motivate a claim-level evidence representation that preserves supporting observations, uses their provenance to guide subsequent acquisition, and distinguishes insufficient evidence from evidence of absence through selective abstention.

In this work, we propose SafeVantage, a vantage-aware semantic memory that associates each claim with its supporting views, estimated target location, and unresolved evidence. The same evidence state drives the acquisition-to-decision loop: the policy explores for relevant evidence, uses claimgrounded target geometry to predict the observability of candidate views, and uses supporting-view provenance to seek geometrically distinct corroboration. The accumulated evidence then yields a selective YES, NO, or ABSTAIN decision, distinguishing insufficient observation from evidence of absence. Fig. 1 illustrates the overview of SafeVantage. We evaluate SafeVantage on an active-view benchmark constructed from the official ProcTHOR-10K splits [10]. Across 232 unseen houses and 7,424 paired episodes per method and action budget, SafeVantage improves macro-F1 and reduces decision risk against the strongest equal-budget baselines at both action budgets, while also reducing travel. We further isolate the role of vantage-aware evidence through equal-input HM3D comparisons, controlled ScanNet viewremoval interventions, and component ablations.

Our main contributions are summarized as follows:

1) We introduce SafeVantage, a vantage-aware semantic memory that preserves claim-level viewpoint provenance, target localization, and unresolved evidence instead of collapsing multi-view observations into a single semantic score.

2) We develop a claim-conditioned active acquisition framework in which retained target geometry predicts candidate-view observability, while supporting-view provenance guides geometrically distinct corroboration before selective YES/NO/ABSTAIN decisions.

3) We establish a ProcTHOR active-view evaluation with 5 equal-budget acquisition baselines on which SafeVantage attains the best risk, macro-F1, and answer rate at both action budgets, complemented by HM3D equalinput evaluation, ScanNet interventions, and component ablations.

## II. RELATED WORK

## A. Open-vocabulary Spatial Memory

Open-vocabulary spatial representations connect language with scene geometry for semantic retrieval and navigation. VLMaps and OpenScene associate vision–language features with 2D maps and dense 3D geometry [1, 2], while LERF embeds language into radiance fields [11]. ConceptGraphs and OpenVoxel organize observations into object-centric graphs and captioned voxel groups [3, 12]. HOV-SG introduces floor–room–object hierarchies for navigation [13], and FARM combines relational memory with viewpoint evidence [14]. These methods support semantic localization and retrieval, with some also retaining source observations. Rather than proposing another mapping backbone, SafeVantage uses claim-level viewpoint provenance for both active acquisition and selective decisions. ConceptGraphs and VLMaps serve as map-centric alternatives in our equal-input evaluation.

## B. Embodied Question Answering and Exploration

EQA requires agents to collect and integrate observations under partial observability [4]. OpenEQA benchmarks episodic and active answering, Explore-EQA calibrates exploration stopping, and EfficientEQA studies efficient openvocabulary answering [5, 6, 9]. MemoryEQA, GraphEQA, and 3D-Mem organize observation histories, semantic scene graphs, and informative snapshots for retrieval and reasoning [7, 15, 16]. EXPRESS-Bench evaluates explorationgrounded answers, while Mind Palace studies long-term active EQA with structured memory and value-of-information stopping [17, 18]. DAAAM and UQ-DAAAM extend memory to dynamic scenes and uncertainty-aware refinement [19, 20]. These approaches improve memory construction, retrieval, and exploration for answering questions. Our work focuses on category-presence decisions and uses each claim’s supporting observations to guide further acquisition and assess whether a decision is sufficiently supported.

## C. Active Perception and View Selection

Active perception selects observations to obtain taskrelevant information. For semantic navigation, SemExp learns goal-oriented exploration from episodic semantic maps [21], while PONI predicts potential functions over semantic maps to guide object search [22]. VLFM uses vision–language value maps for frontier selection [23], and SG-Nav combines online 3D scene graphs and LLM reasoning with re-perception to address unreliable detections [24]. These methods use semantic information to direct exploration and verify targets. SafeVantage conditions acquisition on a claim-grounded evidence state: retained target geometry predicts candidate observability, and candidate views are selected according to expected reduction in terminal decision loss. Our equal-budget baselines isolate coverage-driven, room-prior, and detectorconfidence-guided acquisition rather than reproduce these complete navigation systems.

![](images/0755448fe7070e4bbc3937931728b139f119a54309ecfe5a80f021e1e0d7c9fb.jpg)  
Fig. 2: Viewpoint support and search coverage provide complementary evidence. Navigable viewpoints differ in the target evidence they reveal. Supporting observations provide positive evidence, whereas partial or context-only observation do not justify absence. Retained views and target geometry inform subsequent acquisition and selective decisions.

## D. Selective Prediction and Abstention

Selective prediction introduces a reject option to trade coverage against prediction risk [8, 25]. In robotics, KnowNo uses conformal prediction to determine when a planner should request assistance under ambiguity [26]. Explore-EQA calibrates exploration stopping [6], while AbstainEQA evaluates abstention when perceptual evidence is unavailable or underspecified [27]. UQ-DAAAM further incorporates cross-view semantic uncertainty into memory refinement [20]. SafeVantage shares the abstention objective but focuses on the evidence supplied to the decision rule: supporting viewpoints and their spatial consistency remain explicit, while search coverage is represented separately from positive support. Controlled view-removal and restoration experiments further test how access to supporting evidence affects downstream VLM answers.

## III. METHOD

We consider category-presence queries under partial observability. Given a claim $c ,$ SafeVantage acquires observations within a fixed action horizon B and a geodesic travel budget, then returns YES, NO, or ABSTAIN. The framework comprises semantic memory, active acquisition, and selective decisions.

## A. Vantage-aware Semantic Memory

At viewpoint $v _ { i } ,$ , the agent obtains an observation $o _ { i } =$ $( I _ { i } , D _ { i } , T _ { i } , \tau _ { i } )$ containing RGB, optional depth, camera pose, and timestamp. Let $H _ { t }$ denote the observation history and $\delta _ { i } ( c ) \in \{ 0 , 1 \}$ indicate whether the perception adapter reports claim $c .$ The supporting viewpoints are $V _ { t } ( c ) = \{ v _ { i } \ | \ o _ { i } \in$ $H _ { t } , \ \delta _ { i } ( c ) = 1 \}$ with support count support $_ t ( c ) = | V _ { t } ( c ) |$ Each supporting viewpoint retains its source observation and camera pose. When depth is available, back-projection provides per-view target estimates $\hat { \mathbf { x } } _ { c } ^ { ( i ) }$ and an approximate aggregate location $\hat { \mathbf { x } } _ { c , t } .$ . The memory record is $m _ { t } ( c ) ~ =$ $( c , V _ { t } ( c ) , \mathrm { s u p p o r t } _ { t } ( c ) , \hat { \mathbf { x } } _ { c , t } )$ , where the target location is optional. Because the consistency gate (Eq. 3) accepts only pairs of per-view estimates within radius r, detections of instances separated by more than r cannot corroborate one another. The estimate $\hat { \mathbf { x } } _ { c , t }$ summarizes the supporting detections by their scores and is used to score candidate views; it does not assume a unique object instance. Support is adapterspecific: ProcTHOR uses YOLO-World scores and boxes [28], while HM3D uses Qwen2-VL-7B per-view object lists [29]. Simulator semantics are not queried by the test-time policy; target visibility is additionally used as a training target for the candidate-observability model. Support counts record detections; their spatial diversity and geometric agreement are assessed separately.

Target visibility, property or relation sufficiency, search coverage, and answer correctness are distinct. For category presence, supporting detections provide positive evidence, whereas missing support may reflect incomplete observation rather than absence (Fig. 2). Coverage therefore enters the decision features separately from support. Retained source observations also enable the controlled ScanNet interventions described in the experiments.

## B. Claim-conditioned Active Acquisition

Let $\boldsymbol { A } _ { t }$ contain unvisited viewpoints reachable within the remaining geodesic budget, and let $d _ { t } ( a )$ denote the incremental geodesic travel distance to candidate a in meters.

For each candidate $^ { a , }$ SafeVantage predicts whether the target is likely to be observable from that viewpoint. We form a geometric feature vector $\psi _ { t } ( a , c )$ from the current claim state and candidate geometry (distance to the estimated target, heading alignment, and a geodesic baseline feature), standardize it with statistics frozen from the training scenes, and score it with a logistic outcome model $\hat { o } _ { t } ( a , c ) ~ =$ $\sigma ( \pmb { \theta } ^ { \top } \tilde { \psi } _ { t } ( a , c ) + b _ { o } )$ , fit only on training scenes using target observability as its label. At test time the selector sees only the acquired observations, the resulting claim state, and candidate geometry; oracle visibility, simulator instance identity, and test labels are never queried.

Candidate views are selected by their expected reduction in terminal decision loss. Let $\hat { p } _ { t }$ be the calibrated belief that c holds given the current top supporting score. A candidate yields a positive reading with probability $p _ { t } ^ { + } = \hat { p } _ { t } \hat { o } _ { t } ( a , c ) +$ $( 1 - { \hat { p } } _ { t } ) \phi _ { : }$ where $\phi$ is a fixed detector false-positive rate; each outcome induces a Bayes posterior $q _ { t } ^ { + } = \hat { p } _ { t } \hat { o } _ { t } / p _ { t } ^ { + }$ $q _ { t } ^ { - } = \hat { p } _ { t } \big ( 1 - \hat { o } _ { t } \big ) / \big ( 1 - p _ { t } ^ { + } \big )$ and a hypothetical evidence state $H _ { t } ^ { + }$ or $H _ { t } ^ { - }$ formed with the online update used at execution. Under decision $y \in \left\{ \mathrm { Y E S } , \mathrm { N O } , \mathrm { A B S T A I N } \right\}$ and belief $q ,$ the terminal loss $\ell ( y , q ) = 5 ( 1 - q ) \mathbf { 1 } [ y = \mathrm { Y E S } ] + 2 q \mathbf { 1 } [ y = \mathrm { N O } ] +$ $0 . 5 1 [ y = \mathrm { A B S T A I N } ]$ is the per-decision form of the risk in Eq. 5. Applying the selective rule $\pi _ { \mathrm { a c t i v e } }$ (Eq. 4) to each state, the next viewpoint maximizes the travel-discounted expected loss reduction:

$$
a _ { t } ^ { * } = \arg \operatorname* { m a x } _ { a \in \mathcal { A } _ { t } } \frac { L _ { t } ( c ) - \bar { L } _ { t } ( a ; c ) - \lambda _ { d } d _ { t } ( a ) } { 0 . 5 + d _ { t } ( a ) } , \quad \lambda _ { d } = 0 . 0 0 1 ,\tag{1}
$$

where $L _ { t } ( c ) \ = \ \ell ( \pi _ { \mathrm { a c t i v e } } ( H _ { t } ) , \hat { p } _ { t } )$ is the no-acquisition loss and $\begin{array} { r l r } { \bar { L } _ { t } ( a ; c ) } & { { } = } & { p _ { t } ^ { + } \ell ( \pi _ { \mathrm { a c t i v e } } ( H _ { t } ^ { + } ) , q _ { t } ^ { + } ) + ( 1 - } \end{array}$ $p _ { t } ^ { + } ) \ell ( \pi _ { \mathrm { a c t i v e } } ( H _ { t } ^ { - } ) , q _ { t } ^ { - } )$ is the expected loss after acquiring $a .$ Each acquired observation updates the claim state before the next candidate is scored; action horizons and geodesic travel limits are fixed by the evaluation protocol.

Exploration before grounding. Before a grounded supporting detection provides an estimate of the target location, the target-dependent features in $\psi _ { t } ( a , c )$ are unavailable. The policy therefore explores using spatial novelty, category– room evidence, and travel cost to score reachable candidate views. This exploration branch remains active throughout episodes in which the category is absent. Once a supporting detection yields a target estimate $\hat { \mathbf { x } } _ { c , t }$ , the policy switches to the candidate-observability selector in Eq. 1. Observations with no detection increase search coverage, while grounded positive detections update the supporting-view set and target estimate. Neither branch uses oracle visibility or simulator instance identity at test time.

## C. Evidence-based Selective Decisions

Active decision head. At the fixed horizon, the feature vector $\phi ( H _ { B } , c )$ summarizes the top three detection scores from distinct positions, support counts at five thresholds, observed-position coverage, room and observation counts, grounded-evidence fraction, minimum inter-view target distance, maximum context evidence, and category identity. An $\ell _ { 2 } \cdot$ -regularized logistic model estimates category-presence probability:

$$
\begin{array} { r } { p _ { B } ( c ) = \sigma ( \mathbf { w } ^ { \top } \phi ( H _ { B } , c ) + b ) , } \end{array}\tag{2}
$$

where $\sigma ( z ) = ( 1 + \exp ( - z ) ) ^ { - 1 }$ . The parameters w and b are fitted separately for each policy using training-only observations and five scene-disjoint folds.

Positive commitments additionally require geometrically consistent support. Let $\mathcal { P } _ { B } ( c )$ contain observation pairs meeting the support threshold, originating from distinct positions, and providing valid target estimates. The consistency gate is

$$
\begin{array} { r } { g _ { B } ( c ) = \mathbf { 1 } \left[ \exists ( i , j ) \in \mathcal { P } _ { B } ( c ) : \left\| \hat { \mathbf { x } } _ { c } ^ { ( i ) } - \hat { \mathbf { x } } _ { c } ^ { ( j ) } \right\| _ { 2 } \leq r \right] , } \end{array}\tag{3}
$$

where $r$ is the fixed agreement radius. The decision rule is

$$
\pi _ { \mathrm { a c t i v e } } ( H _ { B } , c ) = \left\{ \begin{array} { l l } { \mathrm { Y E S , } } & { p _ { B } ( c ) \geq \alpha \mathrm { ~ \land ~ } g _ { B } ( c ) = 1 , } \\ { \mathrm { N O , } } & { p _ { B } ( c ) \leq \beta , } \\ { \mathrm { A B S T A I N , } } & { \mathrm { o t h e r w i s e , } } \end{array} \right.\tag{4}
$$

with $0 \leq \beta < \alpha \leq 1$ . Thresholds for SafeVantage are selected by nested validation and frozen before testing. Coverage contributes through the feature vector rather than guaranteeing absence.

Static memory adapter. For equal-input HM3D comparisons, each method’s scalar query score is mapped to YES/NO/ABSTAIN by the same two-threshold adapter, calibrated on the calibration scenes and fixed for evaluation; ties favor higher answer coverage, then lower false-positive rate. Decision cost. We evaluate incorrect commitments, abstention, and resource use through

$$
\begin{array} { r l } & { R = 5 \mathbf { 1 } [ \mathrm { F P } ] + 2 \mathbf { 1 } [ \mathrm { F N } ] } \\ & { ~ + 0 . 5 \mathbf { 1 } [ \mathrm { r e j e c t } ] + 0 . 0 0 1 C _ { \mathrm { r e s o u r c e } } . } \end{array}\tag{5}
$$

FP and FN denote incorrect positive and negative commitments; reject denotes ABSTAIN. False positives receive the largest penalty. $C _ { \mathrm { r e s o u r c e } }$ is travel in meters for active acquisition and view count for static evaluation. This evaluation cost is distinct from the logistic training objective.

## IV. EXPERIMENTS

## A. Experimental Setup

ProcTHOR benchmark. Our primary evaluation uses official ProcTHOR-10K train, validation, and test identities [10] and derives a 24-category vocabulary from training metadata. Every scene provides 40 navigable positions with 4 camera yaws, yielding a common 160-view action graph with geodesic edge costs. For each eligible scene, the evaluator constructs a balanced query set by sampling 4 present and 4 absent categories and assigning 4 deterministic starts to each category, producing 32 paired episodes per scene. Of 256 heldout identities, 255 materialized successfully; 232 satisfied the balanced-query construction, yielding 7,424 episodes per method and action budget. All policies receive RGB-D observations, poses, the training-derived vocabulary, and YOLO-World scores [28]. Oracle visibility and simulator instance identity are not queried by the test-time selector; visibility is additionally used as a training-only target for the candidate-observability model and for evaluation.

Baselines. To comprehensively evaluate SafeVantage, we select 5 baselines:

• Random: Select a deterministic hash-random candidate.

• Nearest Frontier: Minimize incremental geodesic distance.

TABLE I. Equal-budget acquisition-to-decision performance on 232 held-out ProcTHOR houses. Lower risk and travel are better; higher macro-F1 and answer rate are better. Bold marks the best value within each budget. Travel is measured in meters. <sup>†</sup> represents the primary comparison policy.
<table><tr><td>Budget Method</td><td></td><td>Risk↓</td><td>Macro-F1↑</td><td>Answer↑</td><td>Travel↓</td></tr><tr><td rowspan="6">8</td><td>SafeVantage</td><td>0.384</td><td>0.604</td><td>0.581</td><td>4.94</td></tr><tr><td>Random</td><td>0.427</td><td>0.434</td><td>0.350</td><td>19.12</td></tr><tr><td>Nearest frontier</td><td>0.398</td><td>0.491</td><td>0.394</td><td>3.46</td></tr><tr><td>Coverage gain</td><td>0.422</td><td>0.406</td><td>0.305</td><td>6.31</td></tr><tr><td>Category-room prior</td><td>0.396</td><td>0.484</td><td>0.377</td><td>7.23</td></tr><tr><td>Detector-conf. gain</td><td>0.407</td><td>0.446</td><td>0.333</td><td>5.94</td></tr><tr><td rowspan="6">12</td><td>SafeVantage</td><td>0.365</td><td>0.656</td><td>0.639</td><td>7.44</td></tr><tr><td>Random</td><td>0.420</td><td>0.451</td><td>0.361</td><td>19.25</td></tr><tr><td>Nearest frontier</td><td>0.395</td><td>0.464</td><td>0.337</td><td>5.28</td></tr><tr><td>Coverage gain</td><td>0.385</td><td>0.534</td><td>0.471</td><td>8.97</td></tr><tr><td>Category-room prior</td><td>0.382</td><td>0.490</td><td>0.356</td><td>8.16</td></tr><tr><td>Detector-conf. gain †</td><td>0.369</td><td>0.585</td><td>0.516</td><td>8.42</td></tr></table>

• Coverage Gain: Maximize newly covered positions.

• Category-Room Prior: Combine coverage with a training-derived room model.

• Detector-Confidence Gain: Seek a distinct nearby view around the strongest detection.

Different policies induce different distributions over observation histories, so 5 scene-disjoint folds are used to fit a selective head for each policy. Nested validation then selects category–room prior at 8 actions and detector-confidence gain at 12 actions as the primary comparison policies. SafeVantage uses the claim-grounded memory state to predict candidate observability and selects viewpoints according to expected reduction in terminal decision loss under fixed sensing and travel budgets.

Metrics. All ProcTHOR comparisons are paired by scene, category, start, and budget. We report task risk, macro-F1 over positive and negative decisions, false-positive rate on absent queries, answer rate, accuracy conditional on answering, and mean geodesic travel. Macro-F1 is the mean of the YES and NO F1 scores; an ABSTAIN is not counted as a positive or negative prediction and acts as a miss for the true class, so abstaining lowers recall and cannot inflate macro-F1. We estimate confidence intervals using 10,000 paired bootstrap draws that resample whole scenes, preserving the 32 correlated episodes within each house.

## B. Held-out Acquisition-to-decision Performance

We compare SafeVantage with five acquisition baselines at both budgets using shared scenes, detectors, action graphs, and decision heads (Table I).

At 8 actions, SafeVantage achieves the lowest risk and the highest macro-F1 and answer rate among all evaluated policies. Relative to the validation-selected category-room prior, macro-F1 increases from 0.4845 to 0.6044, an absolute improvement of 0.1199, and answer rate increases from 0.3772 to 0.5807. Mean travel decreases from 7.23 to 4.94 m, corresponding to a reduction of 31.7%. Mean risk decreases from 0.3964 to 0.3842. The lower risk together with substantially higher answer rate shows that the improvement is not obtained by increasing the frequency of ABSTAIN decisions. At 12 actions, detector-confidence gain is the validation-selected comparator. SafeVantage increases macro-F1 from 0.5855 to 0.6559, an absolute improvement of 0.0704, and answer rate increases from 0.5163 to 0.6389. Mean risk decreases from 0.3685 to 0.3649, while mean travel decreases from 8.42 to 7.44 m.

The comparison with all five baselines shows the same overall pattern. SafeVantage ranks first in risk, macro-F1, and answer rate at both action budgets, and it travels less than four of the five baselines. Nearest frontier is the only method with lower travel, but the reduction is accompanied by substantially weaker decisions. These results highlight that minimizing travel alone does not necessarily yield evidence that supports reliable semantic decisions. In contrast, effective acquisition must balance movement efficiency with the informativeness of the observations collected. Fig. 3 illustrates one heldout episode in which SafeVantage converts an initial partial detection into corroborated evidence and answers correctly while travelling less than the comparison policy.

## C. Secondary Evidence in HM3D and ScanNet

Having tested the complete policy under the ProcTHOR benchmark, we conduct two secondary studies that examine supporting-view evidence under fixed observations and controlled interventions. On HM3D [30], we compare semantic memory methods using identical RGB-D/pose inputs and a shared selective-decision rule. On ScanNet [31], we remove and restore geometrically verified target-visible views while preserving the remaining input conditions, measuring their effect on downstream VLM answers.

Equal-input evaluation on HM3D. We compare SafeVantage, Qwen2 caption-RAG, ConceptGraphs, and VLMaps on 36 HM3D-Sem scenes and 1,436 category-presence queries. Every method receives the same 160 RGB-D observations and camera poses per scene, with calibration repeated over nine disjoint 4-scene subsets. At the minimum-risk operating point, SafeVantage attains risk 0.524, versus 0.715, 0.617, and 0.618 for caption-RAG, ConceptGraphs, and VLMaps, and also the lowest out-of-calibration risk, matched-coverage risk, and AURC (Table II).

Supporting-view intervention on ScanNet. To test whether supporting observations materially affect downstream answers, we conduct a controlled intervention on 73 ScanNet questions. Official instance geometry, camera poses, and depth determine whether the queried instance is visible, independently of the VLM reader. For each question, targetvisible frames are replaced at their original ranks by targetabsent frames while the question, reader, prompt configuration, input ranks, and history length remain fixed. Starting from this target-absent history, we restore the strongest supporting view, the strongest 4 views, or the complete original supportingview set, thereby varying the available target evidence while preserving the input-history structure. As shown in Table III, Qwen2-VL token F1 increases from 0.2483 without targetvisible views to 0.4410 with 4 restored views and 0.4540 with the complete set; Qwen2.5-VL improves from 0.1888 to 0.3215 under complete restoration. In a separate negativedecision diagnostic, we combine view-opportunity and sourcegroup coverage with evidence scores, calibrating the threshold on 6 development scenes under a 5% false-negative-decision constraint and evaluating on 12 held-out ScanNet scenes. The rule achieves 0.983 observed-absent recall and 0.935 negative-decision precision, versus 0.051 and 0.500 for a score-only rule, with a 0.033 erroneous-negative-decision rate over non-absent states. Together, these results show that supporting views improve downstream answers, while coverage helps distinguish observed absence from unresolved missing evidence.

TABLE II. Equal-input HM3D experiments. $R _ { 9 }$ repeats calibration across nine folds; $R _ { \ @ \mathrm { c o v } }$ matches coverage. Lower risk and AURC are better.
<table><tr><td>Method</td><td>Cov.</td><td>Acc.</td><td>FP</td><td>Recall</td><td>Risk</td><td> $R _ { 9 }$ </td><td> $R _ { \ @ \mathbf { c o v } }$ </td><td>AURC</td></tr><tr><td>SafeVantage</td><td>0.436</td><td>0.962</td><td>0.059</td><td>0.584</td><td>0.524</td><td>0.538</td><td>0.538</td><td>0.649</td></tr><tr><td>Qwen2 caption-RAG</td><td>0.565</td><td>0.881</td><td>0.240</td><td>0.692</td><td>0.715</td><td>0.638</td><td>0.685</td><td>0.715</td></tr><tr><td>ConceptGraphs</td><td>0.432</td><td>0.920</td><td>0.123</td><td>0.552</td><td>0.617</td><td>0.594</td><td>0.707</td><td>0.726</td></tr><tr><td>VLMaps</td><td>0.541</td><td>0.916</td><td>0.162</td><td>0.689</td><td>0.618</td><td>0.656</td><td>0.910</td><td>0.850</td></tr></table>

![](images/8a56e9b592331ab8bdf8ff0f8bdc95a4749dd850366154c4a00a49ac7a8d1339.jpg)  
Fig. 3: One held-out episode for “Is there a paper towel in the room?” Both policies start from the same start with an 8-action budget and a positive ground-truth label. Category–room prior does not observe the target, answers NO, and travels 13.7 m; SafeVantage acquires 4 supporting views, answers YES, and travels 5.4 m.

TABLE III. ScanNet supporting-view intervention. Intervals compare restored conditions with target-absent histories.
<table><tr><td>reader</td><td>evidence set</td><td>F1</td><td>∆ vs absent</td><td>scene CI</td></tr><tr><td>Qwen2-VL</td><td>target absent</td><td>t 0.2483</td><td>1</td><td></td></tr><tr><td>Qwen2-VL</td><td>strongest 1</td><td>0.3565</td><td>+0.1082</td><td>[-0.0181, 0.2290]</td></tr><tr><td>Qwen2-VL</td><td>strongest 4</td><td>0.4410</td><td>+0.1927</td><td>[0.0691, 0.2977]</td></tr><tr><td>Qwen2-VL</td><td>complete</td><td>0.4540</td><td>+0.2057</td><td>[0.0752, 0.3133]</td></tr><tr><td></td><td>Qwen2.5-VL target absent 0.1888</td><td></td><td>_</td><td></td></tr><tr><td>Qwen2.5-VL complete</td><td></td><td>0.3215</td><td>+0.1327</td><td>[0.0513, 0.2053]</td></tr></table>

![](images/d482f5105810d888f8f3a880f4267ca5b68a08db0ee12c8e6d09b476822469bb.jpg)  
Fig. 4: Removing vantage-aware state lowers macro-F1. Bars show full policy minus grouped removal on 96 validation houses; whiskers are paired 95% intervals.

## D. Ablations

We finally examine which acquisition components contribute to decision quality and how their effects depend on the final evidence gate. All variants are evaluated on 96 query-eligible ProcTHOR validation houses at 8- and 12- action budgets under a shared 20 m travel cap, yielding 3,072 paired episodes per variant and budget.

Grouped removal. Jointly removing supporting-view identity, target-ray alignment, and yaw diversity (retaining room evidence, coverage, base geometry, and the travel objective) lowers macro-F1 by 4.35/1.15 points at 8/12 actions (95% CIs [3.23, 5.50]/[0.16, 2.16]; Fig. 4). The effect persists with the pair gate disabled in both policies, separating acquisition from decision-time consistency.

Candidate observability. Removing the candidateobservability model while holding the remaining controller fixed lowers macro-F1 by 7.52 and 8.03 points at 8 and 12 actions, respectively. Mean travel simultaneously increases from 4.94 to 6.30 m and from 7.44 to 8.95 m. So, candidate observability improves both decision quality and acquisition efficiency across both action horizons.

Component contributions. Table IV reports 5 singlecomponent removals and a confidence-backbone-only variant. Removing supporting-view identity or target-ray alignment reduces macro-F1 by 5.54/3.23 and 5.58/3.30 points at 8/12 actions, respectively. Directed exploration has a strongly budget-dependent contribution: removing it costs 0.60 points at 8 actions but 12.26 points at 12 actions. Retaining only the confidence backbone loses 4.60/13.14 points. Other geometric terms show smaller or inconsistent marginal effects.

## E. Evaluation on a Disjoint Holdout

To test generalization beyond the original 232-house evaluation, we held the method, thresholds, and evaluator fixed and tested on a disjoint cohort of 233 additional ProcTHOR houses that were never used for training, validation, or the original evaluation. On this cohort, SafeVantage improves macro-F1 by 7.3 and 4.3 points at 8 and 12 actions, respectively (95% paired scene-bootstrap intervals [6.4, 8.1] and [3.3, 5.2]). Risk decreases by 0.024 and 0.015, respectively, with higher answer rate and lower travel at both horizons. These results confirm that the acquisition advantage transfers to unseen houses.

## V. LIMITATIONS

Our study focuses on category-presence queries, providing a controlled setting for examining the role of viewpoint evidence. Extending this formulation to attributes, relations, and temporal changes may require richer representations of evidence sufficiency. The acquisition policy uses a fixed perception stack and a candidate-observability model, with observations selected from a discrete viewpoint graph under fixed action budgets. Robustness to changes in perception and sensing conditions therefore remains an area for further evaluation. While the HM3D and ScanNet studies provide complementary evidence analyses, continuous navigation and real-robot evaluation are natural next steps toward assessing the framework under broader operating conditions.

TABLE IV. Component ablations on 96 ProcTHOR validation houses. Loss is the macro-F1 drop from the full model (pp).
<table><tr><td></td><td colspan="2">8 actions</td><td colspan="2">12 actions</td></tr><tr><td>Variant</td><td>F1↑</td><td>Loss (pp)↓</td><td>F1↑</td><td>Loss (pp)↓</td></tr><tr><td>Full SafeVantage</td><td>0.5307</td><td></td><td>0.6149</td><td></td></tr><tr><td>— claim provenance</td><td>0.4753</td><td>+5.54</td><td>0.5826</td><td>+3.23</td></tr><tr><td>— target-ray alignment</td><td>0.4749</td><td>+5.58</td><td>0.5819</td><td>+3.30</td></tr><tr><td>— yaw diversity</td><td>0.5311</td><td>-0.04</td><td>0.6162</td><td>-0.13</td></tr><tr><td>– base geometry</td><td>0.5255</td><td>+0.52</td><td>0.6167</td><td>-0.18</td></tr><tr><td>– directed exploration</td><td>0.5247</td><td>+0.60</td><td>0.4923</td><td>+12.26</td></tr><tr><td>Confidence backbone only 0.4847</td><td></td><td>+4.60</td><td>0.4835</td><td>+13.14</td></tr></table>

## VI. CONCLUSION

In this work, we addressed the problem of acquiring sufficient visual evidence for reliable embodied decisions under partial observability. We introduced SafeVantage, a vantage-aware semantic memory that retains each claim’s supporting views and target geometry while keeping positive support distinct from search coverage. This retained vantage evidence is what enables the rest of the system: it grounds the learned candidate-observability model, drives view selection by expected reduction in decision loss, and, through a geometric consistency gate, calibrated YES/NO/ABSTAIN decisions. Experiments on held-out ProcTHOR houses show higher macro-F1 and answer rates with lower risk and travel than the validation-selected equal-budget baselines at both action horizons. Complementary HM3D comparisons demonstrate lower selective risk under identical observations, while ScanNet interventions show that restoring supporting views improves downstream VLM answers. Ablations further support the contributions of candidate observability and claim provenance. Together, these results support treating memory as an evidence state that guides both observation acquisition and semantic commitments, rather than solely as a representation for retrieval.

## VII. ACKNOWLEDGMENTS

The authors used generative AI tools during manuscript preparation to assist with generating and revising selected text and figures, as well as code generation and debugging.

This work was supported by the National Research Foundation of Korea (NRF) grant funded by the Korean government (MSIT) under the project “Development of Risk-Enhanced Continual Learning for Embodied Intelligence in Long-Tail Environments” (Grant No. RS-2026-25595451) and by the Korea Institute of Science and Technology Information (KISTI) R&D program through the joint research project “Development of the Next-Generation Integrated Wired/Wireless Communication Gateway (X-Gateway).”

## REFERENCES

[1] S. Peng, K. Genova, C. M. Jiang, A. Tagliasacchi, M. Pollefeys, and T. Funkhouser, “Openscene: 3d scene understanding with open vocabularies,” in IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2023.

[2] C. Huang, O. Mees, A. Zeng, and W. Burgard, “VLMaps: Visual language maps for robot navigation,” in IEEE International Conference on Robotics and Automation (ICRA), 2023.

[3] Q. Gu, A. Kuwajerwala, S. Morin, K. M. Jatavallabhula, B. Sen, A. Agarwal, C. Rivera, W. Paul, K. Ellis, R. Chellappa, C. Gan, C. M. de Melo, J. B. Tenenbaum, A. Torralba, F. Shkurti, and L. Paull, “Conceptgraphs: Open-vocabulary 3d scene graphs for perception and planning,” in IEEE International Conference on Robotics and Automation (ICRA), 2024, pp. 5021–5028.

[4] A. Das, S. Datta, G. Gkioxari, S. Lee, D. Parikh, and D. Batra, “Embodied question answering,” in IEEE/CVF

Conference on Computer Vision and Pattern Recognition (CVPR), 2018.

[5] A. Majumdar, A. Ajay, X. Zhang, P. Putta, S. Yenamandra, M. Henaff, S. Silwal, P. McVay, O. Maksymets, S. Arnaud, K. Yadav, Q. Li, B. Newman, M. Sharma, V. Berges, S. Zhang, P. Agrawal, Y. Bisk, D. Batra, M. Kalakrishnan, F. Meier, C. Paxton, A. Sax, and A. Rajeswaran, “OpenEQA: Embodied question answering in the era of foundation models,” in IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2024, pp. 16 488–16 498.

[6] A. Z. Ren, J. Clark, A. Dixit, M. Itkina, A. Majumdar, and D. Sadigh, “Explore until confident: Efficient exploration for embodied question answering,” in Robotics: Science and Systems (RSS), 2024, project page: https://explore-eqa.github.io/.

[7] Y. Yang, H. Yang, J. Zhou, P. Chen, H. Zhang, Y. Du, and C. Gan, “3D-Mem: 3d scene memory for embodied exploration and reasoning,” arXiv preprint arXiv:2411.17735, 2024.

[8] Y. Geifman and R. El-Yaniv, “SelectiveNet: A deep neural network with an integrated reject option,” in International Conference on Machine Learning (ICML), 2019, pp. 2151–2159.

[9] K. Cheng, Z. Li, X. Sun, B.-C. Min, A. S. Bedi, and A. Bera, “Efficienteqa: An efficient approach to open-vocabulary embodied question answering,” in 2025 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS). IEEE, 2025, pp. 20 357– 20 364.

[10] M. Deitke, E. VanderBilt, A. Herrasti, L. Weihs, K. Ehsani, J. Salvador, W. Han, E. Kolve, A. Kembhavi, and R. Mottaghi, “ProcTHOR: Large-scale embodied ai using procedural generation,” in Advances in Neural Information Processing Systems (NeurIPS), vol. 35, 2022.

[11] J. Kerr, C. M. Kim, K. Goldberg, A. Kanazawa, and M. Tancik, “LERF: Language embedded radiance fields,” in IEEE/CVF International Conference on Computer Vision (ICCV), 2023.

[12] S.-Y. Huang, J. Choe, Y.-C. F. Wang, and C. Sun, “OpenVoxel: Training-free grouping and captioning voxels for open-vocabulary 3d scene understanding,” in IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2026.

[13] A. Werby, C. Huang, M. Büchner, A. Valada, and W. Burgard, “Hierarchical open-vocabulary 3d scene graphs for language-grounded robot navigation,” in First Workshop on Vision-Language Models for Navigation and Manipulation at ICRA 2024, 2024.

[14] S. He, L. Huang, A. Lilja, F. Hubel, J. Frey, M. Pavone, S. S. Sastry, J. Malik, and C. Tomlin, “FARM: Find anything using relational spatial memory,” arXiv preprint arXiv:2606.15476, 2026.

[15] M. Zhai, Z. Gao, Y. Wu, and Y. Jia, “Memorycentric embodied question answer,” arXiv preprint arXiv:2505.13948, 2025.

[16] S. Saxena, B. Buchanan, C. Paxton, P. Liu, B. Chen, N. Vaskevicius, L. Palmieri, J. Francis, and O. Kroemer, “GraphEQA: Using 3d semantic scene graphs for realtime embodied question answering,” in Conference on Robot Learning (CoRL), ser. Proceedings of Machine Learning Research, vol. 305, 2025, pp. 2714–2742.

[17] K. Jiang, Y. Liu, W. Chen, J. Luo, Z. Chen, L. Pan, G. Li, and L. Lin, “Beyond the destination: A novel benchmark for exploration-aware embodied question answering,” in IEEE/CVF International Conference on Computer Vision (ICCV), 2025, introduces EXPRESS-Bench and the Fine-EQA hybrid exploration baseline.

[18] M. F. Ginting, D.-K. Kim, X. Meng, A. Reinke, B. J. Krishna, N. Kayhani, O. Peltzer, D. D. Fan, A. Shaban, S.-K. Kim, M. J. Kochenderfer, A.-a. Agha-mohammadi, and S. Omidshafiei, “Enter the mind palace: Reasoning and planning for long-term active embodied question answering,” arXiv preprint arXiv:2507.12846, 2025.

[19] N. Gorlo, L. Schmid, and L. Carlone, “Describe anything anywhere at any moment,” arXiv preprint arXiv:2512.00565, 2025.

[20] H. Zhang, N. Gorlo, and L. Carlone, “Remember with confidence: Uncertainty quantification for spatiotemporal memory with probabilistic guarantees,” arXiv preprint arXiv:2606.08277, 2026.

[21] D. S. Chaplot, D. P. Gandhi, A. Gupta, and R. R. Salakhutdinov, “Object goal navigation using goaloriented semantic exploration,” Advances in neural information processing systems, vol. 33, pp. 4247–4258, 2020.

[22] S. K. Ramakrishnan, D. S. Chaplot, Z. Al-Halah, J. Malik, and K. Grauman, “Poni: Potential functions for objectgoal navigation with interaction-free learning,” in 2022 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR). IEEE, 2022, pp. 18 868–18 878.

[23] N. Yokoyama, S. Ha, D. Batra, J. Wang, and B. Bucher, “Vlfm: Vision-language frontier maps for zero-shot semantic navigation,” in 2024 IEEE International Conference on Robotics and Automation (ICRA). IEEE, 2024, pp. 42–48.

[24] H. Yin, X. Xu, Z. Wu, J. Zhou, and J. Lu, “Sg-nav: Online 3d scene graph prompting for llm-based zeroshot object navigation,” Advances in neural information processing systems, vol. 37, pp. 5285–5307, 2024.

[25] H. Huang, J. Li, Z. Zhou, P. Liang, M. Wu, K. Jang, and J. Wang, “A knowledge-augmented dataset of high-risk driving scenarios with llm annotations for autonomous driving,” arXiv preprint arXiv:2607.07103, 2026.

[26] A. Z. Ren, A. Dixit, A. Bodrova, S. Singh, S. Tu, N. Brown, P. Xu, L. Takayama, F. Xia, J. Varley, Z. Xu, D. Sadigh, A. Zeng, and A. Majumdar, “Robots that ask for help: Uncertainty alignment for large language model planners,” in Conference on Robot Learning (CoRL), 2023, pp. 661–682.

[27] T. Wu, C. Zhou, G. Zhao, H. Cao, Y. Pu, and J. Yang, “When robots should say “i don’t know”:

Benchmarking abstention in embodied question answering,” in IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2026, project page: https://abstaineqa.github.io/.

[28] T. Cheng, L. Song, Y. Ge, W. Liu, X. Wang, and Y. Shan, “YOLO-World: Real-time open-vocabulary object detection,” in IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2024, pp. 16 901–16 911.

[29] P. Wang, S. Bai, S. Tan, S. Wang, Z. Fan, J. Bai, K. Chen, X. Liu, J. Wang, W. Ge, Y. Fan, K. Dang, M. Du, X. Ren, R. Men, D. Liu, C. Zhou, J. Zhou, and J. Lin, “Qwen2-VL: Enhancing vision-language model’s perception of the world at any resolution,” arXiv preprint arXiv:2409.12191, 2024.

[30] K. Yadav, R. Ramrakhya, S. K. Ramakrishnan, T. Gervet, J. Turner, A. Gokaslan, N. Maestre, A. X. Chang, D. Batra, M. Savva, A. W. Clegg, and D. S. Chaplot, “Habitat-matterport 3d semantics dataset,” in IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2023, pp. 4927–4936.

[31] A. Dai, A. X. Chang, M. Savva, M. Halber, T. Funkhouser, and M. Nießner, “Scannet: Richlyannotated 3d reconstructions of indoor scenes,” in IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2017, pp. 5828–5839.
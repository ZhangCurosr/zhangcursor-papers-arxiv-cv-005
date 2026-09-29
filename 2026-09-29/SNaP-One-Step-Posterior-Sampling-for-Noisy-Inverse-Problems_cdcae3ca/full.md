# SNaP: One-Step Posterior Sampling for Noisy Inverse Problems

Shirin Shoushtari<sup>1∗</sup> Edward P. Chandler<sup>2∗</sup> Xiao Shi<sup>2</sup> Ulugbek S. Kamilov<sup>2</sup>

<sup>1</sup>WashU, <sup>2</sup>UW-Madison

s.shirin@wustl.edu, epchandler@wisc.edu, xiao.shi@wisc.edu, kamilov@wisc.edu <sup>∗</sup>Equal contribution

## Abstract

Diffusion and flow-matching models can produce high-quality posterior samples for inverse problems, but typically require tens to thousands of network evaluations per draw. MeanFlow enables one-step generation, yet applying it to inverse problems leaves no intermediate steps at which to enforce measurement consistency. We introduce SNaP, a one-step MeanFlow posterior sampler for linear inverse problems with Gaussian noise. Its central innovation is a measurement-adapted source: a Gaussian distribution whose mean and anisotropic covariance are determined by the measurement operator, observation, and noise level. The source anchors well-measured directions while preserving variation where the measurements are weak or uninformative. We show that the exact conditional flow transports this source to the true posterior. Across natural-image restoration and multi-coil MRI, SNaP produces diverse, high-quality samples with one network evaluation per draw, 30 to 2250× faster than iterative samplers.

## 1 Introduction

Many imaging tasks require recovering an unknown image $\scriptstyle { \mathbf { { \vec { x } } } } _ { 1 }$ from noisy linear measurements $\pmb { y } = \pmb { A } \pmb { x } _ { 1 } + \pmb { n } .$ , where A models the imaging system and n $\bf \Pi  \tilde { \bf \Pi } \sim \mathcal { N } ( 0 , \sigma _ { \mathrm { n } } ^ { 2 } I )$ denotes additive Gaussian noise. This formulation encompasses magnetic resonance imaging (MRI), computed tomography (CT), microscopy, astronomical imaging, and remote sensing [1, 2, 3]. Because A is often illconditioned or rank-deficient, the measurements do not uniquely determine x<sub>1</sub>. Therefore, we need prior information on images for a reliable recovery. Classical methods impose handcrafted regularizers such as total variation [4]. More recently, deep generative models, including diffusion models [5, 6] and flow-matching models [7, 8, 9], have learned rich image distributions that can serve as priors for inverse problems [10].

Most generative inverse solvers use an unconditional generative model as an image prior and impose measurement consistency repeatedly during an iterative sampling trajectory. One family modifies the generative dynamics using data-fidelity gradients or pseudoinverse corrections [11, 12, 13, 14, 15, 16]. Another alternates generative updates with explicit data-consistency steps, following the plug-and-play paradigm [17, 18, 19, 20, 21]. Although these corrections improve agreement with the measurement y, they require many sequential network evaluations and therefore make posterior sampling slow. Depending on the method, generating a single sample may require tens to thousands of network evaluations (NFEs). In our CelebA Gaussian-deblurring experiment, for example, DPS [11] requires 1000 NFEs and 27 seconds to produce one (128 × 128) reconstruction (Figure 1).

Consistency models [22, 23], Shortcut Models [24], and MeanFlow[25] enable generation with one network evaluation, avoiding the cost of integrating an iterative sampling trajectory. This efficiency is particularly attractive for posterior sampling, where generating a single reconstruction can require tens to thousands of network evaluations. However, collapsing the trajectory into one step also eliminates the intermediate states for enforcing measurement consistency. A direct extension is to condition MeanFlow on the measurement while retaining the standard isotropic source. Recent studies of one-step generation have reported mean-seeking bias in MeanFlow and averaging-induced diversity degradation in class-conditional flow models [26, 27]. We observe the same failure mode in inverse problems: despite conditioning on the measurement, a MeanFlow with an isotropic source produces nearly identical reconstructions across draws, so averaging offers no improvement (Table 5). This motivates shaping the one-step transport to reflect the geometry of the forward model.

![](images/b96e5386ce1f924f261e1a07b565d2fa12e7eb4b17d007fc43286225fdc2edeb.jpg)  
Figure 1: Left: LPIPS versus wall-clock time per sample for CelebA Gaussian deblurring. SNaP achieves lowest LPIPS at a fraction of the cost of iterative samplers. Right: 4× super-resolution on AFHQ-Cat. Columns show one SNaP draw, averages of M = 4, 16, and 100 draws, and the ground truth; zoomed crops, error maps, and PSNR are shown below. Each draw requires one network evaluation. Averaging reduces pixel error and improves PSNR while smoothing fine detail.

We propose SNaP, a one-step sampler for noisy linear inverse problems. Its key innovation is a measurement-adapted Gaussian source that builds the forward model and noise level into the starting point of the transport. The source anchors well-measured directions while retaining stochastic variation in weakly measured and null-space directions. Samples from this source can be drawn efficiently, and a single network evaluation maps each draw to a reconstruction.

Our contributions are: (a) We extend one-step MeanFlow sampling to noisy linear inverse problems by constructing a measurement-adapted transport aligned with the geometry of the likelihood. (b) We derive the source center by minimizing its expected squared displacement to posterior samples among linear estimators. The resulting construction uses a Tikhonov estimate to anchor the transport. (c) Across natural-image restoration and multi-coil MRI, SNaP produces diverse, high-quality reconstructions with one network evaluation per draw. Its samples achieve near-unit calibration ratios on the tested CelebA tasks, while sampling 30 to 2250× faster than iterative posterior samplers.

## 2 Background

Inverse Problems. We consider a linear measurement operator $\pmb { { A } } \in \mathbb { R } ^ { m \times n }$ with additive white Gaussian noise,

$$
{ \pmb y } = { \pmb A } { \pmb x } _ { 1 } + { \pmb n } , \qquad { \pmb n } \sim { \mathcal { N } } ( \mathbf { 0 } , \sigma _ { \mathrm { n } } ^ { 2 } { \pmb I } ) .\tag{1}
$$

The goal is to infer the unknown image $\pmb { x } _ { 1 } \in \mathbb { R } ^ { n }$ from the measurement $\boldsymbol { y } \in \mathbb { R } ^ { m }$ ; in posterior sampling, this means drawing samples from $p ( \pmb { x } _ { 1 } \mid \pmb { y } )$ . We write the singular value decomposition of $\pmb { A }$ as $\pmb { A } = \pmb { U } \pmb { \Sigma } \pmb { V } ^ { \top }$ and let $\rho : = \mathrm { r a n k } ( A )$ . We order the singular values so that $s _ { 1 } \geq \cdot \cdot \cdot \geq s _ { \rho } > 0$ and set $s _ { i } = 0$ for $\rho < i \leq n$ . The corresponding right singular vectors are the columns of V. We denote the pseudoinverse by $A ^ { \dagger }$ and the orthogonal projection onto null(A) by $P : = I - A ^ { \dagger } A$

Flow Matching. Flow matching [7, 8, 9] learns a velocity field that transports a source distribution $p _ { 0 }$ to a data distribution $p _ { 1 }$ . We index time so that $t = 0$ corresponds to the source and $t \ : = \ : 1$ to the data. The flow $\psi _ { t }$ is defined by the ODE $\begin{array} { r } { \frac { \mathrm { d } } { \mathrm { d } t } \psi _ { t } ( { \pmb x } ) ~ = ~ { \pmb v } _ { t } ( \psi _ { t } ( { \pmb x } ) ) } \end{array}$ for $t \in \mathsf { \Gamma } ( 0 , 1 )$ , and it maps samples from $p _ { 0 }$ to samples from $p _ { 1 }$ . For the exact marginal velocity field, the terminal map satisfies $( \psi _ { 1 } ) _ { \# } p _ { 0 } = p _ { 1 }$ . This marginal velocity is generally not available in closed form, so flow matching instead regresses a sample-conditional velocity. Given a coupling of endpoints $( { \pmb x } _ { 0 } , { \pmb x } _ { 1 } )$ with $\mathbf { \mathcal { x } } _ { 0 } \sim p _ { 0 }$ and $\mathbf { \boldsymbol { x } } _ { 1 } \sim p _ { 1 }$ , the straight-line path

$$
z _ { t } = \left( 1 - t \right) \pmb { x } _ { 0 } + t \pmb { x } _ { 1 } , \qquad \pmb { v } = \pmb { x } _ { 1 } - \pmb { x } _ { 0 } ,\tag{2}
$$

has constant conditional velocity v. A network $v ^ { \theta }$ is trained by minimizing $\mathbb { E }  \pmb { v } ^ { \theta } ( z _ { t } , t ) - ( \pmb { x } _ { 1 } -$ $\pmb { x } _ { 0 } ) \| _ { 2 } ^ { 2 }$ . The expected parameter gradient of this conditional regression objective coincides with that of the marginal flow-matching objective [7]. The source is commonly chosen as $\mathcal { N } ( \mathbf { 0 } , \pmb { I } )$ because it

is easy to sample; its random initial conditions provide the variability propagated by the deterministic ODE.

Mean Flows. Sampling a flow-matching model requires integrating the velocity v, which takes many network evaluations. MeanFlow [25, 23] avoids this integration by modeling the average velocity u over an interval $[ r , t ]$ , defined as

$$
\pmb { u } ( z _ { r } , r , t ) : = \ \frac { 1 } { t - r } \int _ { r } ^ { t } \pmb { v } ( z _ { s } , s ) \mathrm { d } s .\tag{3}
$$

Once u is learned, the entire flow reduces to a single evaluation, $z _ { 1 } = z _ { 0 } + { \pmb u } ( z _ { 0 } , 0 , 1 )$ . Learning u directly from equation 3 is intractable, since it requires the integral of v. Instead, rewriting equation 3 as $\begin{array} { r } { ( t - \bar { r } ) \mathbf { \nabla } u = \bar { \int _ { r } ^ { t } } } \end{array}$ v ds and differentiating with respect to r, with t fixed, gives the MeanFlow identity

$$
{ \pmb u } ( z _ { r } , r , t ) = { \pmb v } ( z _ { r } , r ) + ( t - r ) \frac { \mathrm { d } } { \mathrm { d } r } { \pmb u } ( z _ { r } , r , t ) .\tag{4}
$$

This differential identity provides the training relation used by MeanFlow. It also connects Mean-Flow to consistency and shortcut models, which learn finite-time flow maps through related consistency relations [22, 23, 24].

Measurement-dependent sources. Standard MeanFlow formulations use a fixed, measurementindependent source. Applying MeanFlow to inverse problems therefore requires incorporating the measurement model into the one-step transport. NullFlow [28] is a one-step posterior sampler for noiseless linear inverse problems that uses a measurement-dependent source:

$$
{ \pmb x } _ { 0 } = { \pmb A } ^ { \dagger } { \pmb y } + { \pmb P } { \pmb \epsilon } , \qquad { \pmb \epsilon } \sim { \mathcal { N } } ( \mathbf { 0 } , { \pmb I } ) .\tag{5}
$$

For a noiseless measurement, the source satisfies $\pmb { A x } _ { 0 } = \pmb { y } ,$ , and restricting the velocity to $\operatorname { n u l l } ( A )$ keeps the entire trajectory in the measurement-consistent affine set $\{ { \pmb x } : { \bar { \pmb A } } { \pmb x } = { \pmb y } \}$ ; however, with noisy measurements, the posterior is no longer supported on this set. Moreover, the anchor $A ^ { \dagger } y$ contains the row-space noise term $A ^ { \dagger } n$ , which a velocity restricted to null $( A )$ cannot alter.

## 3 Method

SNaP builds the geometry of the forward model into a one-step posterior transport. Its measurementdependent source is designed to keep the path to the posterior short while preserving the randomness needed for sampling. It concentrates draws in well-measured directions and retains variation in weakly measured and null-space directions. A conditional MeanFlow then learns to transport these draws toward $p ( \pmb { x } _ { 1 } \mid \pmb { y } )$ , producing an approximate posterior sample from one source draw and one network evaluation. We now derive the source, characterize the exact transport, and describe how the model is trained and sampled.

Along the straight-line path between a source sample x<sub>0</sub> and a posterior sample $\scriptstyle { \mathbf { { \vec { x } } } } _ { 1 }$ , the sampleconditional velocity is ${ \pmb v } = { \pmb x } _ { 1 } - { \pmb x } _ { 0 }$ , and therefore $\mathbf { \dot { \| } } \pmb { v } \| _ { 2 } ^ { 2 } = \| \pmb { x } _ { 1 } - \pmb { x } _ { 0 } \| _ { 2 } ^ { 2 } [ 8 ]$ . We keep the path and the conditional-independence structure of the endpoint coupling fixed and use the source distribution as our design variable. This differs from coupling-based methods that modify the endpoint pairing [29, 30, 31]. Unlike approaches that prescribe or learn data-dependent sources [32, 33], we derive the source from the measurement model using an explicit design criterion.

Although the posterior is unavailable in closed form, the likelihood provides a tractable geometry for images compatible with the measurement. Under additive Gaussian noise,

$$
p ( \pmb { y } | \pmb { x } ) \propto \mathrm { e x p } \left( - \frac { \| \pmb { A } \pmb { x } - \pmb { y } \| _ { 2 } ^ { 2 } } { 2 \sigma _ { \mathrm { n } } ^ { 2 } } \right) ,
$$

and is maximized on the least-squares solution set

$$
S _ { y } : = \arg \operatorname* { m i n } _ { \pmb { x } } \| \pmb { A } \pmb { x } - \pmb { y } \| _ { 2 } ^ { 2 } .
$$

Its superlevel sets are convex tubes around $\mathcal { S } _ { y }$ . Along a measured right singular vector ${ \boldsymbol { v } } _ { i }$ , their width scales as $\sigma _ { \mathrm { n } } / s _ { i } \mathrm { : }$ well-measured directions are tightly constrained, weakly measured directions are less constrained, and null-space directions are unconstrained (Fig. 2a). The noise level sets the scale

![](images/cd1d52c56ba8d565ea455121276c31b79ac874fe639dd47bf67917e18f9864c5.jpg)

Figure 2: Geometry-aware one-step transport (schematic). Left: Likelihood superlevel sets form tubes around $\begin{array} { r } { \boldsymbol { S } _ { \boldsymbol { y } } \colon } \end{array}$ narrow in well-measured directions, wider in weakly measured directions, and unbounded along null(A). Gray points represent posterior samples. Middle: An isotropic source spreads samples across directions regardless of measurement strength, leading to large source-totarget displacements. Right: The SNaP source concentrates samples in well-measured directions while retaining variation in weakly measured and null-space directions, shortening displacements.

because $A x _ { 1 } - y = - n$ . Since these tubes are convex, a straight path between two points in the same tube stays inside it. This geometry motivates a source with little variation in well-measured directions and greater variation in weakly measured and null-space directions (Fig. 2c), in contrast to an isotropic source (Fig. 2b). We construct this source from the observed $\mathbf { { \boldsymbol { \mathsf { y } } } } _ { \mathrm { { \boldsymbol { \cdot } } } }$ , known A and $\sigma _ { \mathrm { n } }$ , and auxiliary randomness. We consider sources with an explicit, linear measurement-dependent center:

$$
\pmb { x } _ { 0 } = \pmb { K } \pmb { y } + \pmb { \xi } ,\tag{6}
$$

where K is fixed for a given measurement model and ξ has mean 0 and covariance Σ, independently of $( \pmb { x } _ { 1 } , \pmb { n } )$ . This choice admits an exact analysis and avoids training a separate source model [33]. The family contains the standard isotropic source when $\pmb { K } = \mathbf { 0 }$ and $\Sigma = I$ , and sources centered at the least-squares solution when $\pmb { K } = \bar { \pmb { A } } ^ { \dagger }$

Because the one-step map is deterministic given $( { \pmb x } _ { 0 } , { \pmb y } )$ , all sampling randomness originates from the source. For the family in equation 6, its expected squared displacement decomposes as

$$
\begin{array} { r } { \mathbb { E } \big [ \| \pmb { x } _ { 1 } - \pmb { x } _ { 0 } \| _ { 2 } ^ { 2 } \big ] = \mathrm { t r } ( \pmb { R } ( \pmb { K } ) ) + \mathrm { t r } ( \pmb { \Sigma } ) , \qquad \pmb { R } ( \pmb { K } ) : = \mathbb { E } \big [ ( \pmb { x } _ { 1 } - \pmb { K } \pmb { y } ) ( \pmb { x } _ { 1 } - \pmb { K } \pmb { y } ) ^ { \top } \big ] . } \end{array}\tag{7}
$$

For fixed $\Sigma .$ , minimizing the displacement reduces to minimizing the center error tr $( R ( K ) )$ over K. Minimizing over Σ would instead give $\mathbf { \delta } \Sigma = \mathbf { 0 }$ and eliminate source variability, so we optimize the center and choose the spread separately.

Proposition 1. Let $\mathbf { x } _ { 1 }$ have mean 0 and covariance $C ,$ not necessarily Gaussian, and let ${ \textbf { 3 } } =$ $\pmb { A x } _ { 1 } + \pmb { n }$ with n $\mathbf { \sigma } \cdot \sim \mathcal { N } ( \mathbf { 0 } , \sigma _ { \mathrm { n } } ^ { 2 } I )$ and $\sigma _ { \mathrm { n } } > 0$ . For any fixed $\Sigma \succeq \mathbf { 0 } ,$ , the expected squared displacement ofthe sourcefamily in equation 6 is uniquely minimized over K by

$$
\begin{array} { r } { K _ { C } : = { C A } ^ { \top } { \left( { A C A } ^ { \top } + \sigma _ { \mathrm { n } } ^ { 2 } I \right) } ^ { - 1 } . } \end{array}\tag{8}
$$

$K _ { C } y$ is thus the linear minimum-mean-squared-error estimator of x<sub>1</sub> from y, with error covariance

$$
\Sigma _ { C } : = R ( K _ { C } ) = C - K _ { C } A C .
$$

For the isotropic working covariance $C = \tau ^ { 2 } I ,$ , with $\tau > 0 ,$ , the optimal center is the Tikhonov estimate $m _ { y } = A ^ { \top } { \left( A A ^ { \top } + \lambda I \right) } ^ { - 1 } y ,$ , with $\lambda = \sigma _ { \mathrm { n } } ^ { 2 } / \tau ^ { 2 }$ , and its error covariance is $\tau ^ { 2 } W$ , where $W = I - A ^ { \top } \big ( A A ^ { \top } + \lambda I \big ) ^ { - 1 } A .$

The proof is given in App. A.1. Minimizing the displacement over Σ would give $\Sigma \ : = \ : { \bf 0 }$ and collapse the source. For a chosen working covariance $C .$ , we instead set $\Sigma = \Sigma _ { C }$ , matching the error covariance of the optimal linear estimator. Under the corresponding Gaussian working model, this is also the posterior covariance.

Because the true image covariance is unknown, we instantiate Proposition 1 with the isotropic working covariance $\breve { C } = \tau ^ { 2 } I$ , where τ controls the source scale. Under this model, $m _ { y }$ is the displacement-minimizing linear center and $\tau ^ { 2 } \mathbf { { W } }$ is its error covariance. The resulting Tikhonov center regularizes weakly measured directions and avoids amplifying their measurement noise, unlike the least-squares estimate $\boldsymbol { A } ^ { \dagger } \boldsymbol { y }$ . The working model is used only to construct the source; it does not assume that the true image distribution is isotropic or Gaussian. Using this mean and covariance, we define

$$
\begin{array} { r } { q _ { \tau } ( \pmb { x } | \pmb { y } ) = \mathcal { N } ( \pmb { x } ; m _ { y } , \tau ^ { 2 } \pmb { W } ) . } \end{array}\tag{9}
$$

In the right singular basis of A,

$$
W = V \mathrm { d i a g } ( w _ { i } ) V ^ { \top } , \qquad w _ { i } = { \frac { \sigma _ { \mathrm { n } } ^ { 2 } } { \sigma _ { \mathrm { n } } ^ { 2 } + \tau ^ { 2 } s _ { i } ^ { 2 } } } .\tag{10}
$$

The source variance along ${ \mathbf { } } v _ { i }$ is therefore $\tau ^ { 2 } w _ { i }$ . It is small in well-measured directions, approaches $\sigma _ { \mathrm { n } } ^ { 2 } / s _ { i } ^ { 2 }$ when $\tau ^ { 2 } s _ { i } ^ { 2 } \gg \sigma _ { \mathrm { n } } ^ { 2 } .$ , and equals $\tau ^ { 2 }$ on null(A). Thus, the source suppresses variation in wellmeasured directions while retaining it where the measurement is weak or uninformative. ${ \bf A s } \sigma _ { \mathrm { n } }  0$ the isotropic working source converges to $q _ { \tau } ( \cdot \mid \pmb { y } )  \mathcal { N } ( \pmb { A } ^ { \dag } \pmb { y } , \tau ^ { 2 } \pmb { P } )$ , which recovers the NullFlow source when $\tau = 1$

Replacing the isotropic source by the working posterior changes the transport, but not its target.

Proposition 2. Let $\sigma _ { \mathrm { n } } > 0$ , and let $\pmb { x } _ { 0 } \sim q _ { \tau } ( \cdot | \pmb { y } )$ and $\pmb { x } _ { 1 } \sim p ( \cdot | \pmb { y } )$ be conditionally independent given y and joined by the straight-line path in equation 2. Assume that $p ( \cdot \mid y )$ has finite second moments, and that the flow of the marginal velocity field has a terminal limit: writing $\psi _ { 0  t } f o r$ its flow mapfrom time 0 to time t, the limit $\psi _ { 0 \to 1 } : = \operatorname* { l i m } _ { t \to 1 } \psi _ { 0 \to t }$ exists $q _ { \tau } ( \cdot \vert \boldsymbol { y } )$ )-almost everywhere. Then the one-step map

$$
T _ { \pmb { y } } ( \pmb { x } ) : = \pmb { x } + \pmb { u } ( \pmb { x } , 0 , 1 | \pmb { y } )\tag{11}
$$

satisfies

$$
( T _ { \pmb { y } } ) _ { \# } q _ { \tau } ( \cdot | \ \pmb { y } ) = p ( \cdot | \ \pmb { y } ) \qquad f o r e \nu e r y \tau > 0 .\tag{12}
$$

Here $\textbf { \em u }$ is the exact average velocity of equation $_ { 3 ; }$ the trained network $u ^ { \theta }$ approximates it, so the guarantee holds for SNaP samples only up to this approximation error. The exact-map result identifies the posterior target of SNaP; the effect of the source choice on the learned one-step approximation is evaluated through the controlled comparison with isotropic sources in Table 5.

## 3.1 Training and sampling

A direct draw from $q _ { \tau } ( \cdot | \mathbf { \boldsymbol { y } } )$ would require applying the matrix square root $W ^ { 1 / 2 }$ , which is generally impractical for large imaging operators. We instead use perturb-and-solve [34, 35]:

$$
\begin{array} { r } { x _ { 0 } = \epsilon + A ^ { \top } \big ( A A ^ { \top } + \lambda I \big ) ^ { - 1 } \big ( y - A \epsilon - n ^ { \prime } \big ) , \quad \epsilon \sim \mathcal { N } ( \mathbf { 0 } , \tau ^ { 2 } I ) , \quad n ^ { \prime } \sim \mathcal { N } ( \mathbf { 0 } , \sigma _ { \mathrm { n } } ^ { 2 } I ) . } \end{array}\tag{13}
$$

Here, ϵ and $\mathbf { { \boldsymbol { n } } ^ { \prime } }$ are independent of each other and of $( \pmb { x } _ { 1 } , \pmb { n } )$ . Lemma 1 shows that ${ \pmb x } _ { 0 } | { \pmb y } \sim q _ { \tau } ( \cdot | { \pmb y } )$ exactly and that $\scriptstyle { \mathbf { { \mathit { x } } } } _ { 0 }$ and $\scriptstyle { \mathbf { { \vec { x } } } } _ { 1 }$ are conditionally independent given $\mathbf { \pmb { y } } .$ . Each source draw requires one solve with $A A ^ { \top } + \lambda I ;$ operator-specific implementations are given in App. C.2.

Training. The network parameterizes the conditional average velocity

$$
\begin{array} { r } { \pmb { u } _ { r , t } ^ { \theta } = \pmb { u } ^ { \theta } ( z _ { r } , r , t | \pmb { y } , \sigma _ { \mathrm { n } } ) . } \end{array}
$$

Because y and $\sigma _ { \mathrm { n } }$ remain fixed along the path, the conditional MeanFlow identity gives the target

$$
\pmb { u } _ { \mathrm { t g t } } = \pmb { v } + ( t - r ) \operatorname { s g } \left[ \left( \partial _ { z } \pmb { u } _ { r , t } ^ { \pmb { \theta } } \right) \pmb { v } + \partial _ { r } \pmb { u } _ { r , t } ^ { \pmb { \theta } } \right] , \qquad \pmb { v } = \pmb { x } _ { 1 } - \pmb { x } _ { 0 } .\tag{14}
$$

The derivative term is evaluated using one Jacobian–vector product, and $\mathrm { s g } [ \cdot ]$ prevents gradients from propagating through the target. We minimize

$$
\mathcal { L } ( \pmb { \theta } ) = \mathbb { E } [ \mathrm { s g } [ \omega ( \ell ) ] \mathrm { ~ } \ell ] , \qquad \mathrm { ~ } \ell : = \frac { 1 } { n } \left\| \pmb { u } _ { r , t } ^ { \theta } - \pmb { u } _ { \mathrm { t g t } } \right\| _ { 2 } ^ { 2 } ,\tag{15}
$$

using the adaptive weight $\omega ( \ell ) = ( \ell + c ) ^ { - p }$ from [25]. Implementation details and Algorithm 1 are given in App. C.3.

Sampling. At test time, SNaP draws $\scriptstyle { \mathbf { { \mathit { x } } } } _ { 0 }$ using equation 13 and returns $\begin{array} { r c l } { \widehat { \pmb x } _ { 1 } } & { = } & { { \pmb x } _ { 0 } + } \end{array}$ $\pmb { u } ^ { \theta } ( \hat { \pmb { x } } _ { 0 } , 0 , 1 | \pmb { y } , \sigma _ { \mathrm { n } } )$ . Each sample requires one source solve and one network evaluation. Redrawing $( \epsilon , n ^ { \prime } )$ with y fixed produces conditionally independent samples from the learned model.

## 4 Experiments

Datasets. We evaluate on CelebA (128×128), AFHQ-Cat (256×256), and multi-coil fastMRI brain AXT2 (320×320), averaging all metrics over 100 test measurements. Details are in App. C.1.

Table 1: CelebA, 100 test images. We report single draw (M=1) and the average of M draws for SNaP. Note that a single SNaP draw is competitive in LPIPS at one network evaluation, while averaging draws attains the best SSIM on every task and the best PSNR on three of the four. Best and second best are shown in blue and red.
<table><tr><td rowspan="2">Method</td><td rowspan="2">NFE</td><td colspan="3">Deblurring</td><td colspan="3">Super-resolution</td><td colspan="3">Random inpainting</td><td colspan="3">Box inpainting</td></tr><tr><td>PSNR↑</td><td>SSIM↑</td><td>LPIPS↓</td><td>PSNR↑</td><td>SSIM↑</td><td>LPIPS↓</td><td>PSNR↑</td><td>SSIM↑</td><td>LPIPS↓</td><td>PSNR↑</td><td>SSIM↑</td><td>LPIPS↓</td></tr><tr><td>Degraded</td><td></td><td>27.26</td><td>0.838</td><td>0.214</td><td>11.67</td><td>0.182</td><td>0.859</td><td>11.98</td><td>0.198</td><td>1.070</td><td>22.33</td><td>0.754</td><td>0.217</td></tr><tr><td>MSE regressor</td><td>1</td><td>35.74</td><td>0.953</td><td>0.029</td><td>34.17</td><td>0.945</td><td>0.033</td><td>34.42</td><td>0.955</td><td>0.022</td><td></td><td></td><td></td></tr><tr><td>PnP-GS</td><td>23</td><td>33.97</td><td>0.924</td><td>0.041</td><td>31.23</td><td>0.890</td><td>0.065</td><td>29.17</td><td>0.874</td><td>0.066</td><td></td><td></td><td>一</td></tr><tr><td>DDRM</td><td>20</td><td>35.02</td><td>0.946</td><td>0.027</td><td>32.24</td><td>0.927</td><td>0.027</td><td>32.40</td><td>0.942</td><td>0.029</td><td></td><td></td><td></td></tr><tr><td>DiffPIR</td><td>100</td><td>34.83</td><td>0.938</td><td>0.027</td><td>31.87</td><td>0.892</td><td>0.030</td><td>32.45</td><td>0.924</td><td>0.024</td><td>30.48</td><td>0.918</td><td>0.028</td></tr><tr><td>OT-ODE</td><td>180</td><td>32.96</td><td>0.920</td><td>0.029</td><td>31.34</td><td>0.903</td><td>0.027</td><td>28.69</td><td>0.871</td><td>0.049</td><td>29.37</td><td>0.919</td><td>0.038</td></tr><tr><td>PnP-Flow5</td><td>500</td><td>34.80</td><td>0.941</td><td>0.046</td><td>31.44</td><td>0.905</td><td>0.056</td><td>33.98</td><td>0.953</td><td>0.021</td><td>31.06</td><td>0.939</td><td>0.043</td></tr><tr><td>Flower5-OT</td><td>500</td><td>35.65</td><td>0.954</td><td>0.031</td><td>33.03</td><td>0.931</td><td>0.039</td><td>33.97</td><td>0.952</td><td>0.019</td><td>31.85</td><td>0.952</td><td>0.023</td></tr><tr><td>DPS</td><td>1000</td><td>34.27</td><td>0.936</td><td>0.019</td><td>32.23</td><td>0.917</td><td>0.023</td><td>33.43</td><td>0.951</td><td>0.013</td><td>30.26</td><td>0.942</td><td>0.022</td></tr><tr><td>SNAP (M=1)</td><td>1</td><td>32.90</td><td>0.919</td><td>0.018</td><td>31.27</td><td>0.903</td><td>0.023</td><td>31.97</td><td>0.927</td><td>0.018</td><td>30.73</td><td>0.934</td><td>0.021</td></tr><tr><td>SNAP (M=4)</td><td>4</td><td>34.98</td><td>0.947</td><td>0.022</td><td>33.26</td><td>0.934</td><td>0.023</td><td>33.97</td><td>0.951</td><td>0.015</td><td>32.78</td><td>0.954</td><td>0.018</td></tr><tr><td>SNAP (M=16)</td><td>16</td><td>35.68</td><td>0.954</td><td>0.028</td><td>33.95</td><td>0.943</td><td>0.029</td><td>34.64</td><td>0.957</td><td>0.018</td><td>33.43</td><td>0.959</td><td>0.020</td></tr><tr><td>SNAP (M=100)</td><td>100</td><td>35.91</td><td>0.956</td><td>0.030</td><td>34.16</td><td>0.946</td><td>0.031</td><td>34.84</td><td>0.959</td><td>0.019</td><td>33.69</td><td>0.961</td><td>0.022</td></tr></table>

Table 2: Distributional and calibration metrics for CelebA. $\Delta _ { \mathrm { c a l } }$ is the calibration ratio of App. C.4, for which 1 is ideal. Note that SNaP attains the best or second-best FID among the posteriorsampling baselines in one network evaluation. Best and second best are shown in blue and red.
<table><tr><td rowspan="2">Method</td><td rowspan="2">NFE</td><td colspan="2">Gaussian deblurring</td><td colspan="2">Super-resolution</td><td colspan="2">Box inpainting</td></tr><tr><td>FID↓</td><td> $\Delta _ { \mathrm { c a l } } (  1 )$ </td><td>FID↓</td><td> $\Delta _ { \mathrm { c a l } } (  1 )$ </td><td>FID↓</td><td> $\Delta _ { \mathrm { c a l } } (  1 )$ </td></tr><tr><td>MSE regressor</td><td>1</td><td>22.06</td><td></td><td>24.76</td><td></td><td></td><td></td></tr><tr><td>DDRM</td><td>20</td><td>20.26</td><td>0.31</td><td>18.94</td><td>0.19</td><td></td><td></td></tr><tr><td>Flower1-OT</td><td>100</td><td>17.86</td><td>0.23</td><td>23.79</td><td>0.24</td><td>18.72</td><td>0.24</td></tr><tr><td>DPS</td><td>1000</td><td>18.25</td><td>0.64</td><td>19.94</td><td>0.67</td><td>15.42</td><td>2.15</td></tr><tr><td>SNAP (ours)</td><td>1</td><td>15.80</td><td>1.03</td><td>16.80</td><td>0.99</td><td>15.53</td><td>1.08</td></tr></table>

Forward operators. We adopt the degradation settings of [20] and [21] for natural images, with $\sigma _ { \mathrm { n } } ~ = ~ 0 . 0 5$ everywhere except random inpainting, where $\sigma _ { \mathrm { n } } ~ = ~ 0 . 0 1$ . Deblurring uses a 61×61 Gaussian kernel with $\sigma _ { b } ~ = ~ 1 . 0$ on CelebA and 3.0 on AFHQ-Cat; super-resolution uses stride subsampling at 2× on CelebA and 4× on AFHQ-Cat; random inpainting masks 70% of pixels; box inpainting uses a centered $4 0 \times 4 0$ mask on CelebA and 80×80 on AFHQ-Cat. For MRI, A = MF S with sampling mask M, Fourier transform F and per-slice ESPIRiT sensitivities S, at ×4 and ×8 Cartesian acceleration, with noise scaled to 20 and 30 dB input SNR. Closed forms of equation 13 for each operator are given in App. C.2.

Baselines. For natural images, we compare against a supervised MSE regressor trained with the SNaP backbone, the plug-and-play solvers PnP-GS [36] and PnP-Flow [20], and the iterative posterior samplers DDRM [13], DiffPIR [17], OT-ODE [15], Flower [21] and DPS [11]. Flower and PnP-Flow are run at the recommended setting, where the exact budget of every method is listed in App. C.3. For MRI, we compare against zero-filled, wavelet- ${ \boldsymbol { \cdot } } { \boldsymbol { \ell } } _ { 1 }$ and TV reconstructions, the MSE regressor, and the posterior samplers CSGM [37], DiffPIR, DPS, DAPS [19] and PnP-DM [38, 39]. Variational Flow Maps (VFM) [33] also place the measurement in the source, but learn it with an adapter rather than constructing it in closed form, and report results for latent-space flow models, whereas SNaP operates in pixel space; we compare the two source constructions in App. B.4. NullFlow [28] is not defined for $\sigma _ { \mathrm { n } } > 0 ( \mathrm { S e c . } 2 )$ , so we compare against it from $\sigma _ { \mathrm { n } } = 0$ to 0.05 in App. B.5.

Metrics. We report PSNR, SSIM [40], LPIPS [41] and FID [42]. None of them says whether the spread of the draws is of the right size, so we also report the calibration ratio $\Delta _ { \mathrm { c a l } } = V / B$ , which divides the variance V of the draws by the squared error B of their mean. For an exact posterior sampler the two are equal, so $\Delta _ { \mathrm { c a l } } = \mathrm { \dot { 1 } }$ is ideal: values below 1 indicate draws collapsed toward a point estimate, and values above 1 indicate over-dispersed draws. It is a second-moment summary and does not certify conditional coverage. We give the definition and estimators in App. C.4.

Table 3: fastMRI brain, multi-coil. PSNR and SSIM over 100 test slices, at ×4 and ×8 acceleration and two input SNRs. For SNaP, we report results for a single draw M = 1 and average of M draws. Note that among posterior samplers SNaP is very competitive even using one NFE. Best and second best are shown in blue and red.
<table><tr><td rowspan="3"></td><td rowspan="3"></td><td colspan="4">×4</td><td colspan="4">×8</td></tr><tr><td colspan="2">20 dB</td><td colspan="2">30 dB</td><td colspan="2">20 dB</td><td colspan="2">30 dB</td></tr><tr><td>NFE PSNR↑</td><td>SSIM↑</td><td>PSNR↑</td><td>SSIM↑</td><td>PSNR↑</td><td>SSIM↑</td><td>PSNR↑</td><td>SSIM↑</td></tr><tr><td>Zero-filled</td><td></td><td>25.35</td><td>0.791</td><td>25.37</td><td>0.793</td><td>21.98</td><td>0.694</td><td>21.99</td><td>0.694</td></tr><tr><td>Wavelet+l1</td><td></td><td>26.57</td><td>0.672</td><td>27.75</td><td>0.768</td><td>23.02</td><td>0.551</td><td>23.81</td><td>0.643</td></tr><tr><td>TV</td><td></td><td>26.19</td><td>0.798</td><td>26.27</td><td>0.799</td><td>22.62</td><td>0.689</td><td>22.66</td><td>0.691</td></tr><tr><td>MSE regressor</td><td>1</td><td>33.80</td><td>0.909</td><td>33.85</td><td>0.912</td><td>30.71</td><td>0.875</td><td>30.73</td><td>0.878</td></tr><tr><td>DiffPIR</td><td>100</td><td>29.84</td><td>0.709</td><td>30.01</td><td>0.719</td><td>26.14</td><td>0.642</td><td>26.31</td><td>0.655</td></tr><tr><td>CSGM</td><td>1000</td><td>27.74</td><td>0.734</td><td>32.39</td><td>0.802</td><td>27.74</td><td>0.733</td><td>28.33</td><td>0.754</td></tr><tr><td>DPS</td><td>1000</td><td>28.81</td><td>0.646</td><td>29.83</td><td>0.669</td><td>24.89</td><td>0.575</td><td>25.17</td><td>0.574</td></tr><tr><td>DAPS</td><td>1000</td><td>31.55</td><td>0.818</td><td>32.44</td><td>0.782</td><td>27.31</td><td>0.749</td><td>28.01</td><td>0.714</td></tr><tr><td>PnP-DM</td><td>1000</td><td>32.16</td><td>0.778</td><td>33.39</td><td>0.786</td><td>27.18</td><td>0.701</td><td>29.00</td><td>0.712</td></tr><tr><td>SNAP (M=1)</td><td>1</td><td>32.06</td><td>0.889</td><td>32.89</td><td>0.899</td><td>28.10</td><td>0.818</td><td>29.14</td><td>0.845</td></tr><tr><td>SNAP (M=4)</td><td>4</td><td>32.68</td><td>0.904</td><td>33.80</td><td>0.919</td><td>29.06</td><td>0.843</td><td>29.88</td><td>0.864</td></tr><tr><td>SNAP (M=16)</td><td>16</td><td>32.85</td><td>0.908</td><td>34.05</td><td>0.924</td><td>29.34</td><td>0.850</td><td>30.09</td><td>0.869</td></tr></table>

Table 4: Computational cost. NFE and 1-batch wall-clock time per posterior sample. Best in each row is in bold.
<table><tr><td colspan="6">CelebA, Gaussian deblurring</td></tr><tr><td></td><td>DPS</td><td>OT-ODE</td><td>Flower</td><td>DDRM</td><td>SNAP</td></tr><tr><td>NFE↓</td><td>1000</td><td>180</td><td>100</td><td>20</td><td>1</td></tr><tr><td>Time † sample (s) ↓</td><td>27.014</td><td>6.575</td><td>2.614</td><td>0.360</td><td>0.012</td></tr><tr><td>Slowdown vs. ours ↓</td><td>2251×</td><td>548×</td><td>218×</td><td>30×</td><td></td></tr><tr><td></td><td>fastMRI brain, multi-coil ×8</td><td></td><td></td><td></td><td></td></tr><tr><td></td><td>CSGM</td><td>DAPS</td><td>PnP-DM</td><td>DPS</td><td>SNAP</td></tr><tr><td>NFE↓</td><td>1000</td><td>1000</td><td>1000</td><td>1000</td><td>1</td></tr><tr><td>Time / sample (s) ↓</td><td>127.78</td><td>96.28</td><td>92.02</td><td>76.07</td><td>0.227</td></tr><tr><td>Slowdown vs. ours ↓</td><td>563×</td><td>424×</td><td>405×</td><td>335×</td><td></td></tr></table>

## 4.1 Main Results

On CelebA (Tab. 1), a single SNaP draw attains the best LPIPS on Gaussian deblurring, and averaging draws attains the best SSIM on all four operators and the best PSNR on three of the four. Results on AFHQ-Cat are reported in App. B.1. SNaP also attains the best FID on two of the three CelebA operators and the calibration ratio closest to 1 on all three (Tab. 2), so the spread of its draws matches the error they make. On multi-coil fastMRI (Tab. 3) it is competitive with posterior sampler at both accelerations and both noise levels with only one NFE. Figures 3 and 4 show the corresponding reconstructions on natural images and on brain MRI.

## 4.2 Computational cost

Each SNaP sample costs one network evaluation and one source draw by equation 13, whose solve with $( A A ^ { \top } + \lambda \mathbf { \dot { I } } ) ^ { - 1 }$ is diagonal (or Fourier diagonal) for the image operators and reduces to a short conjugate-gradient solve for multi-coil MRI (App. C.2). As reported in Tab. 4, this makes SNaP 30 to 2250× faster per sample than the iterative samplers on CelebA deblurring and 335 to 563× faster on ×8 MRI.

## 4.3 Ablations

Source. Table 5 isolates the effect of the source by comparing our proposed source to two isotropic sources: the first is the usual flow matching source N(0, I) and the other is ${ \mathcal { N } } ( \mathbf { 0 } , \tau ^ { 2 } I )$ to control for the influence of τ. We hold the other design choices constant and y and $\sigma _ { \mathrm { n } }$ are retained in the conditioning. Under the isotropic sources, the calibration ratio falls to 0.00: the draws are nearly identical, yielding a point estimator with higher single-draw PSNR and worse LPIPS. The measurement-adapted source keeps the draws dispersed, and its single-draw advantage is perceptual rather than distortion-based. Figures 12 and 13 in App. B.3 show this collapse directly—the draws and their per-pixel spread—and trace both metrics against the averaging budget M.

![](images/4476fbc623fdb3c428003c5fc1e28ee5ba6c6e772397a8cede661e6727b922b0.jpg)

Figure 3: Visual comparison of samples for three imaging tasks, CelebA super-resolution and AFHQ-Cat box inpainting and deblurring, with PSNR and LPIPS for the displayed image. Note that SNaP requires one NFE, while DiffPIR requires 100 and DPS and Flower require 1000. SNaP attains the best LPIPS on all three.  
![](images/22fb22679eac1e23077bdb24c4b78148f7e610d40cdc556abde2cb48f0bd78aa.jpg)  
Figure 4: fastMRI brain, multi-coil, ×8 acceleration at 20 dB input SNR. Reconstructions with PSNR and SSIM, and error magnitude and a zoomed in region is displayed. Note that SNaP achieves competitive reconstruction using 1 NFE, while baselines use 1000.

Additional results. App. B.1 reports the full AFHQ-Cat tables and App. B.2 shows further reconstructions on natural images and brain MRI. App. B.5 compares SNaP with NullFlow across a range of measurement noise levels, and App. B.4 compares the two source constructions on the two-dimensional benchmark of [33]. App. B.7 verifies the method against a closed-form posterior, and App. B reports the remaining ablations, on the working-prior scale τ , the (r, t) sampling ratio and few-step composition.

## 5 Related work

Iterative posterior samplers. Most generative solvers enforce consistency with the measurement repeatedly along the sampling trajectory, either by correcting the generative dynamics with a data fidelity gradient or a pseudoinverse update [11, 12, 13, 14, 15, 16, 43], or by alternating a generative step with an explicit data-consistency step [17, 18, 19, 20, 21]. These methods reach high sample quality, but a single posterior sample costs tens to thousands of network evaluations.

Table 5: Source distribution. CelebA, Gaussian deblurring at $\sigma _ { \mathrm { n } } = 0 . 0 5 ;$ only the source differs across rows. M is the number of averaged draws, and $\Delta _ { \mathrm { c a l } }$ is the calibration ratio (App. C.4), for which 1 is ideal. An isotropic source collapses onto the posterior mean, where draws are nearidentical and averaging brings no gain.
<table><tr><td></td><td colspan="2">M = 1</td><td colspan="2"> $M = 1 0 0$ </td><td></td></tr><tr><td>Source</td><td>PSNR↑</td><td>LPIPS↓</td><td>PSNR↑</td><td>LPIPS↓</td><td> $\Delta _ { \mathrm { c a l } } (  1 )$ </td></tr><tr><td> $\mathcal { N } ( \mathbf { 0 } , \pmb { I } )$ </td><td>35.85</td><td>0.030</td><td>35.86</td><td>0.030</td><td>0.00</td></tr><tr><td> $\mathcal { N } ( \mathbf { 0 } , \tau ^ { 2 } I )$ </td><td>35.83</td><td>0.030</td><td>35.84</td><td>0.030</td><td>0.00</td></tr><tr><td> $\mathcal { N } ( m _ { y } , \tau ^ { 2 } W )$  (ours)</td><td>32.90</td><td>0.018</td><td>35.91</td><td>0.030</td><td>1.03</td></tr></table>

One-step generative models. Consistency models [22, 23], shortcut models [24] and MeanFlow [25] eliminate the trajectory and map a source sample to the data in a single network evaluation. Using them for inverse problems requires the measurement to enter through the network conditioning or through the source, since no intermediate state is available at which to enforce data consistency. For real-world super-resolution, MFSR [44] distills a text-to-image flow model into a MeanFlow in which the low-resolution image enters only through conditioning, without an explicit forward model or posterior sampling.

Measurement-dependent sources. Bridge models transport from a measurement-dependent source by starting from the degraded observation itself [45, 46, 47], but that source carries no per-direction covariance. NullFlow [28] restricts a MeanFlow to null(A) and VFM [33] learns a measurementdependent adapter jointly with the flow map, which covers a broader class of problems but yields a source that is trained. MeanFlow has also been applied to PDE-based Bayesian inverse problems with a prior-aligned base measure that does not depend on the measurement [48].

## 6 Conclusion

We introduced SNaP, a one-step posterior sampler for linear inverse problems with Gaussian noise. Instead of correcting toward the measurement at every step, SNaP builds it into the transport. The transport starts from a distribution centered at the linear estimate that minimizes the expected transport distance, with a spread that matches the remaining uncertainty. We proved that the exact onestep map carries this distribution to the target posterior. On natural images and multi-coil MRI, SNaP draws a posterior sample in a single network evaluation, 30 to 2250× faster than iterative samplers. On CelebA, averaging its draws gives the best SSIM on all four operators and the best PSNR on three. Its draws also reach the best FID on two of the three operators tested and the calibration ratio closest to 1 on all three.

Limitations. The construction is tied to linear operators, since the working posterior is available in closed form only in that setting, and nonlinear forward models require further work. Sampling the source requires one solve with $A A ^ { \top } + \lambda I$ , which is inexpensive for the operators considered here but may be demanding for operators without exploitable structure. Finally, SNaP operates in pixel space, and extending the source construction to latent-space flow models remains open.

## Acknowledgments

This work was supported in part by the National Science Foundation under CAREER award CCF-2625643 and award CCF-2622128.

## References

[1] Mario Bertero, Patrizia Boccacci, and Christine De Mol, Introduction to inverse problems in imaging, CRC Press, 2021.

[2] Michael T McCann, Kyong Hwan Jin, and Michael Unser, “Convolutional Neural Networks for Inverse Problems in Imaging: A Review,” IEEE Signal Process. Mag., vol. 34, no. 6, pp. 85–95, 2017.

[3] Gregory Ongie, Ajil Jalal, Christopher A Metzler, Richard G Baraniuk, Alexandros G Dimakis, and Rebecca Willett, “Deep Learning Techniques for Inverse Problems in Imaging,” IEEE J. Sel. Areas Inf. Theory, vol. 1, no. 1, pp. 39–56, 2020.

[4] Leonid I Rudin, Stanley Osher, and Emad Fatemi, “Nonlinear t=Total Variation Based Noise Removal Algorithms,” Physica D, vol. 60, no. 1-4, pp. 259–268, 1992.

[5] Jonathan Ho, Ajay Jain, and Pieter Abbeel, “Denoising Diffusion Probabilistic Models,” in Proc. NeurIPS, 2020.

[6] Yang Song, Jascha Sohl-Dickstein, Diederik P Kingma, Abhishek Kumar, Stefano Ermon, and Ben Poole, “Score-Based Generative Modeling through Stochastic Differential Equations,” in Proc. ICLR, 2021.

[7] Yaron Lipman, Ricky T. Q. Chen, Heli Ben-Hamu, Maximilian Nickel, and Matt Le, “Flow Matching for Generative Modeling,” in Proc. ICLR, 2023.

[8] Xingchao Liu, Chengyue Gong, and Qiang Liu, “Flow Straight and Fast: Learning to Generate and Transfer Data with Rectified Flow,” in Proc. ICLR, 2023.

[9] Michael Albergo, Nicholas M Boffi, and Eric Vanden-Eijnden, “Stochastic Interpolants: A Unifying Framework for Flows and Diffusions,” J. Mach. Learn. Res., vol. 26, no. 209, pp. 1–80, 2025.

[10] Giannis Daras, Hyungjin Chung, Chieh-Hsin Lai, Yuki Mitsufuji, Jong Chul Ye, Peyman Milanfar, Alexandros G Dimakis, and Mauricio Delbracio, “A Survey on Diffusion Models for Inverse Problems,” arXiv:2410.00083, 2024.

[11] Hyungjin Chung, Jeongsol Kim, Michael T. Mccann, Marc L. Klasky, and Jong Chul Ye, “Diffusion Posterior Sampling for General Noisy Inverse Problems,” in Proc. ICLR, 2023.

[12] Jiaming Song, Arash Vahdat, Morteza Mardani, and Jan Kautz, “Pseudoinverse-Guided Diffusion Models for Inverse Problems,” in Proc. ICLR, 2023.

[13] Bahjat Kawar, Michael Elad, Stefano Ermon, and Jiaming Song, “Denoising Diffusion Restoration Models,” in Proc. NeurIPS, 2022.

[14] Yinhuai Wang, Jiwen Yu, and Jian Zhang, “Zero-Shot Image Restoration Using Denoising Diffusion Null-Space Model,” in Proc. ICLR, 2023.

[15] Ashwini Pokle, Matthew J. Muckley, Ricky T. Q. Chen, and Brian Karrer, “Training-free Linear Image Inverses via Flows,” TMLR, 2024.

[16] Heli Ben-Hamu, Omri Puny, Itai Gat, Brian Karrer, Uriel Singer, and Yaron Lipman, “D-Flow: Differentiating through Flows for Controlled Generation,” in Proc. ICML, 2024.

[17] Yuanzhi Zhu, Kai Zhang, Jingyun Liang, Jiezhang Cao, Bihan Wen, Radu Timofte, and Luc Van Gool, “Denoising Diffusion Models for Plug-and-Play Image Restoration,” in CVPR Workshops, 2023, pp. 1219–1229.

[18] Hyungjin Chung, Suhyeon Lee, and Jong Chul Ye, “Decomposed Diffusion Sampler for Accelerating Large-Scale Inverse Problems,” in Proc. ICLR, 2024.

[19] Bingliang Zhang, Wenda Chu, Julius Berner, Chenlin Meng, Anima Anandkumar, and Yang Song, “Improving Diffusion Inverse Problem Solving with Decoupled Noise Annealing,” in Proc. CVPR, 2025.

[20] Ségolène Tiffany Martin, Anne Gagneux, Paul Hagemann, and Gabriele Steidl, “PnP-Flow: Plug-and-Play Image Restoration with Flow Matching,” in Proc. ICLR, 2025.

[21] Mehrsa Pourya, Bassam El Rawas, and Michael Unser, “Flower: A Flow-Matching Solver for Inverse Problems,” in Proc. ICLR, 2026.

[22] Yang Song, Prafulla Dhariwal, Mark Chen, and Ilya Sutskever, “Consistency Models,” in Proc. ICML, 2023.

[23] Nicholas Boffi, Michael Albergo, and Eric Vanden-Eijnden, “How to Build a Consistency Model: Learning Flow Maps via Self-Distillation,” in Proc. NeurIPS, 2025.

[24] Kevin Frans, Danijar Hafner, Sergey Levine, and Pieter Abbeel, “One Step Diffusion via Shortcut Models,” in Proc. ICLR, 2025.

[25] Zhengyang Geng, Mingyang Deng, Xingjian Bai, Zico Kolter, and Kaiming He, “Mean Flows for One-step Generative Modeling,” in Proc. NeurIPS, 2025.

[26] Xiao He, Yang Li, Peizhen Zhang, Songtao Liu, Zhao Zhong, and Nannan Wang, “Stabilizing, Scaling & Enhancing MeanFlow for Large-scale Diffusion Distillation,” arXiv preprint arXiv:2605.17834, 2026.

[27] Yexiong Lin, Jia Shi, Shanshan Ye, Wanyu Wang, Yu Yao, and Tongliang Liu, “SubFlow: Sub-mode Conditioned Flow Matching for Diverse One-Step Generation,” arXiv 2604.12273, 2026.

[28] Xiao Shi, Edward P Chandler, Chicago Y Park, Shirin Shoushtari, and Ulugbek S Kamilov, “NullFlow: One-Step Generative Reconstruction,” arXiv:2606.22696, 2026.

[29] Aram-Alexandre Pooladian, Heli Ben-Hamu, Carles Domingo-Enrich, Brandon Amos, Yaron Lipman, and Ricky T. Q. Chen, “Multisample Flow Matching: Straightening Flows with Minibatch Couplings,” in Proc. ICML, 2023.

[30] Alexander Tong, Kilian Fatras, Nikolay Malkin, Guillaume Huguet, Yanlei Zhang, Jarrid Rector-Brooks, Guy Wolf, and Yoshua Bengio, “Improving and generalizing flow-based generative models with minibatch optimal transport,” TMLR, pp. 1–34, 2024.

[31] Michael Samuel Albergo, Mark Goldstein, Nicholas Matthew Boffi, Rajesh Ranganath, and Eric Vanden-Eijnden, “Stochastic Interpolants with Data-Dependent Couplings,” in Proc. ICML, 2024.

[32] Sang gil Lee, Heeseung Kim, Chaehun Shin, Xu Tan, Chang Liu, Qi Meng, Tao Qin, Wei Chen, Sungroh Yoon, and Tie-Yan Liu, “PriorGrad: Improving Conditional Denoising Diffusion Models with Data-Dependent Adaptive Prior,” in Proc. ICLR, 2022.

[33] Abbas Mammadov, So Takao, Bohan Chen, Ricardo Baptista, Morteza Mardani, Yee Whye Teh, and Julius Berner, “Variational Flow Maps: Make Some Noise for One-Step Conditional Generation,” in Proc. ICML, 2026.

[34] George Papandreou and Alan L Yuille, “Gaussian Sampling by Local Perturbations,” in Proc. NeurIPS, 2010.

[35] Johnathan M Bardsley, Antti Solonen, Heikki Haario, and Marko Laine, “Randomize-thenoptimize: A method for sampling from posterior distributions in nonlinear inverse problems,” SIAM J. Sci. Comput., vol. 36, no. 4, pp. A1895–A1910, 2014.

[36] Samuel Hurault, Arthur Leclaire, and Nicolas Papadakis, “Gradient Step Denoiser for Convergent Plug-and-Play,” in Proc. ICLR, 2022.

[37] Ajil Jalal, Marius Arvinte, Giannis Daras, Eric Price, Alexandros G Dimakis, and Jon Tamir, “Robust Compressed Sensing MRI with Deep Generative Priors,” in Proc. NeurIPS, 2021.

[38] Zihui Wu, Yu Sun, Yifan Chen, Bingliang Zhang, Yisong Yue, and Katherine Bouman, “Principled Probabilistic Imaging using Diffusion Models as Plug-and-Play Priors,” in Proc. NeurIPS, 2024.

[39] Hongkai Zheng, Wenda Chu, Bingliang Zhang, Zihui Wu, Austin Wang, Berthy Feng, Caifeng Zou, Yu Sun, Nikola Borislavov Kovachki, Zachary E. Ross, Katherine Bouman, and Yisong Yue, “InverseBench: Benchmarking Plug-and-Play Diffusion Priors for Inverse Problems in Physical Sciences,” in Proc. ICLR, 2025.

[40] Zhou Wang, Alan C Bovik, Hamid R Sheikh, and Eero P Simoncelli, “Image Quality Assessment: From Error Visibility to Structural Similarity,” IEEE Trans. Image Proc., vol. 13, no. 4, pp. 600–612, 2004.

[41] Richard Zhang, Phillip Isola, Alexei A. Efros, Eli Shechtman, and Oliver Wang, “The Unreasonable Effectiveness of Deep Features as a Perceptual Metric,” in Proc. CVPR, 2018.

[42] Martin Heusel, Hubert Ramsauer, Thomas Unterthiner, Bernhard Nessler, and Sepp Hochreiter, “Gans Trained by a Two Time-scale Update Rule Converge to a Local Nash Equilibrium,” in Proc. NeurIPS, 2017.

[43] Jeongsol Kim, Bryan Sangwoo Kim, and Jong Chul Ye, “Flowdps: Flow-driven Posterior Sampling for Inverse Problems,” in Proc. ICCV, 2025.

[44] Ruiqing Wang, Kai Zhang, Yuanzhi Zhu, Hanshu Yan, Shilin Lu, and Jian Yang, “MFSR: MeanFlow Distillation for One Step Real-World Image Super Resolution,” arXiv:2603.20690, 2026.

[45] Guan-Horng Liu, Arash Vahdat, De-An Huang, Evangelos A Theodorou, Weili Nie, and Anima Anandkumar, “I2SB: Image-to-image Schrödinger Bridge,” in Proc. ICML, 2023.

[46] Linqi Zhou, Aaron Lou, Samar Khanna, and Stefano Ermon, “Denoising Diffusion Bridge Models,” in Proc. ICLR, 2024.

[47] Mauricio Delbracio and Peyman Milanfar, “Inversion by Direct Iteration: An Alternative to Denoising Diffusion for Image Restoration,” TMLR, 2023.

[48] Zhiqi Li, Yuchen Sun, Greg Turk, and Bo Zhu, “Functional Mean Flow in Hilbert Space,” in Proc. CVPR, 2026.

## A Proofs

## A.1 Proof of Proposition 1

Proof. Let ${ \pmb x } _ { 0 } = { \pmb K } { \pmb y } + { \pmb \xi }$ as in equation 6.

Since ${ \pmb x } _ { 1 } - { \pmb x } _ { 0 } = ( { \pmb x } _ { 1 } - { \pmb K } { \pmb y } ) - { \pmb \xi }$ and ξ has mean 0 and is independent of $\pmb { x } _ { 1 } - \pmb { K } \pmb { y }$ , for every unit vector e,

$$
\begin{array} { r } { \mathbb { E } \big [ ( e ^ { \top } ( { \pmb x } _ { 1 } - { \pmb x } _ { 0 } ) ) ^ { 2 } \big ] = e ^ { \top } R ( { \pmb K } ) e + e ^ { \top } { \pmb \Sigma } e , \qquad \mathbb { E } [ \| { \pmb x } _ { 1 } - { \pmb x } _ { 0 } \| _ { 2 } ^ { 2 } ] = \operatorname { t r } R ( { \pmb K } ) + \operatorname { t r } { \pmb \Sigma } . } \end{array}
$$

For fixed Σ, only $R ( K )$ depends on K.

Let $C _ { x y } : = \mathbb { E } [ \pmb { x } _ { 1 } \pmb { y } ^ { \top } ] = C \pmb { A } ^ { \top }$ and $C _ { y } : = \mathbb { E } [ { \pmb { y } } { \pmb { y } } ^ { \top } ] = { \pmb { A } } C { \pmb { A } } ^ { \top } + \sigma _ { \mathrm { n } } ^ { 2 } { \pmb { I } } ,$ , which is positive definite since $\sigma _ { \mathrm { n } } > 0 .$ , so that $K _ { C } = C _ { x y } C _ { y } ^ { - 1 }$ . Expanding $\begin{array} { r } { \dot { R } ( K ) = C - K \bar { C } _ { x y } ^ { \top } - C _ { x y } K ^ { \top } + K C _ { y } K ^ { \top } } \end{array}$ and using $K _ { C } C _ { y } = C _ { x y } ,$

$$
R ( K ) = R ( K _ { C } ) + ( K - K _ { C } ) C _ { y } ( K - K _ { C } ) ^ { \top } , \qquad R ( K _ { C } ) = C - C _ { x y } C _ { y } ^ { - 1 } C _ { x y } ^ { \top } = C - K _ { C } A C .\tag{16}
$$

By equation 16, for every unit vector $e ,$

$$
\begin{array} { r } { e ^ { \top } R ( K ) e = e ^ { \top } R ( K _ { C } ) e + \left\| C _ { y } ^ { 1 / 2 } ( K - K _ { C } ) ^ { \top } e \right\| _ { 2 } ^ { 2 } \geq e ^ { \top } R ( K _ { C } ) e , } \end{array}
$$

so $K _ { C }$ minimizes the displacement along every direction. Taking the trace,

$$
\mathrm { t r } R ( { \boldsymbol { K } } ) = \mathrm { t r } R ( K _ { C } ) + \left. C _ { y } ^ { 1 / 2 } ( K - K _ { C } ) ^ { \top } \right. _ { F } ^ { 2 } ,
$$

and since $C _ { y } ^ { 1 / 2 }$ is invertible, the second term vanishes if and only if $K = K _ { C }$ . The total displacement is therefore minimized if and only if $K = K _ { C }$ , for every $\dot { \Sigma }$

Isotropic case. For $C = \tau ^ { 2 } I$ and $\lambda = \sigma _ { \mathrm { n } } ^ { 2 } / \tau ^ { 2 } ;$ , dividing by τ<sup>2</sup> gives $K _ { C } = A ^ { \top } ( A A ^ { \top } + \lambda I ) ^ { - 1 }$ , so $K _ { C } y = m _ { y }$ , and $\pmb { R } ( \pmb { K _ { C } } ) = \tau ^ { 2 } \big ( \pmb { I } - \pmb { A } ^ { \top } ( \pmb { A } \pmb { A } ^ { \top } + \lambda \pmb { I } ) ^ { - 1 } \pmb { A } \big ) = \tau ^ { 2 } \pmb { W } .$ □

Remarks. (i) The proof uses only the first two moments of $_ { x _ { 1 } , n }$ and $\xi ;$ neither the images, the noise, nor the perturbation need to be Gaussian. (ii) If $\pmb { x } _ { 1 } \sim \mathcal { N } ( \mathbf { 0 } , C )$ , Gaussian conditioning gives $\pmb { x } _ { 1 } | \pmb { y } \sim \mathcal { N } ( K _ { C } \pmb { y } , \pmb { \Sigma } _ { C } )$ , so the source with center $K _ { C } y$ and Gaussian perturbation of covariance $\Sigma _ { C }$ is the exact posterior; for $C = \tau ^ { 2 } I$ it is the working posterior equation 9. (iii) For $C = \tau ^ { 2 } I ,$ as $\sigma _ { \mathrm { n } }  0$ with τ fixed, $\lambda  0$ , and the standard limit of Tikhonov regularization gives $K _ { C }  A ^ { \dagger }$ and $W  I - A ^ { \dagger } A = P$ , so the source tends to $\mathcal { N } ( A ^ { \dagger } y , \tau ^ { 2 } P )$

## A.2 Proof of Proposition 2

Proof. Let $p _ { t } ( \cdot | \boldsymbol { y } )$ denote the distribution of $z _ { t } = ( 1 - t ) \pmb { x } _ { 0 } + t \pmb { x } _ { 1 }$ , so that $p _ { 0 } ( \cdot | \pmb { y } ) = q _ { \tau } ( \cdot | \pmb { y } )$ and $p _ { 1 } ( \cdot \mid \pmb { y } ) = p ( \cdot \mid \pmb { y } )$ . The marginal velocity

$$
{ \pmb v } _ { t } ( z , { \pmb y } ) = \mathbb { E } [ { \pmb x } _ { 1 } - { \pmb x } _ { 0 } \mid { \pmb z } _ { t } = z , { \pmb y } ]\tag{17}
$$

generates $\{ p _ { t } ( \cdot | \pmb { y } ) \} _ { t \in [ 0 , 1 ] }$ through the continuity equation [7, 9].

Since $\sigma _ { \mathrm { n } } ~ > ~ 0$ , we have $\lambda > 0$ and $w _ { i } \in ( 0 , 1 ]$ for all i, so W is positive definite. For $t < 1$ the interpolant ${ \boldsymbol { z } } _ { t }$ is therefore the sum of tx<sub>1</sub> and an independent Gaussian with nondegenerate covariance $( 1 - t ) ^ { 2 } \tau ^ { 2 } W$ . Its density $p _ { t } ( \cdot \mid \boldsymbol { y } )$ is thus smooth and strictly positive. Since $\scriptstyle { \mathbf { \mathscr { x } } } _ { 1 }$ has finite second moments, standard arguments show that $\pmb { v } _ { t } ( \cdot , \pmb { y } )$ is locally Lipschitz, uniformly for $t \in [ 0 , T ]$ and every $T < 1 \left[ 9 \right]$ . The $\begin{array} { r } { \mathrm { O D E } \frac { \mathrm { d } } { \mathrm { d } t } z _ { t } = { v } _ { t } ( z _ { t } , { y } ) } \end{array}$ then has a unique solution on $[ 0 , 1 )$ , and its flow map satisfies $( \psi _ { 0 \to t } ) _ { \# } q _ { \tau } ( \cdot | \pmb { y } ) = \breve { p } _ { t } ( \cdot | \pmb { y } )$ for every $t < 1$

Since $\begin{array} { r } { \mathbb { E } [ \| z _ { t } - x _ { 1 } \| _ { 2 } ^ { 2 } ] = ( 1 - t ) ^ { 2 } \mathbb { E } [ \| x _ { 1 } - x _ { 0 } \| _ { 2 } ^ { 2 } ] \to 0 } \end{array}$ , the distribution $p _ { t } ( \cdot \mid \boldsymbol { y } )$ converges to $p ( \cdot \mid y )$ as $t  1$ . By the terminal-limit assumption of Proposition $\begin{array} { r } { 2 , \psi _ { 0 \to 1 } ( \pmb { x } ) : = \operatorname* { l i m } _ { t \to 1 } \psi _ { 0 \to t } ( \pmb { x } ) } \end{array}$ exists for q -almost every x. Almost-sure convergence implies convergence in distribution, so

$$
( \psi _ { 0 \to 1 } ) _ { \# } q _ { \tau } ( \cdot | \ : y ) = p ( \cdot | \ : y ) .\tag{18}
$$

Integrating the ODE from 0 to t and letting $t \to 1$ gives

$$
\psi _ { 0 \to 1 } ( \pmb { x } _ { 0 } ) - \pmb { x } _ { 0 } = \int _ { 0 } ^ { 1 } \pmb { v } _ { s } \big ( \psi _ { 0 \to s } ( \pmb { x } _ { 0 } ) , \pmb { y } \big ) \mathrm { d } s = \pmb { u } ( \pmb { x } _ { 0 } , 0 , 1 | \pmb { y } ) ,\tag{19}
$$

where the last equality is the average velocity in equation 3. Hence $\psi _ { 0 \to 1 } = T _ { y }$ , and equation 18 proves equation 12. The target $p ( \cdot \mid \boldsymbol { y } )$ does not depend on $\tau ,$ so the conclusion holds for every $\tau > 0$ □

## A.3 Perturb-and-solve sampling

Direct sampling would require a square root of W. The perturb-and-solve sampler avoids this square root while preserving the same Gaussian law.

Lemma 1. Let $\epsilon \sim \mathcal { N } ( \mathbf { 0 } , \tau ^ { 2 } I )$ and $\pmb { n } ^ { \prime } \sim \mathcal { N } ( \mathbf { 0 } , \sigma _ { \mathrm { n } } ^ { 2 } \pmb { I } )$ be independent of each other and $o f \left( \pmb { x } _ { 1 } , \pmb { n } \right)$ and let $\scriptstyle { \mathbf { { \mathit { x } } } } _ { 0 }$ be given by equation 13. Then $\mathbf { \boldsymbol { x } } _ { 0 } \left| \mathbf { \boldsymbol { y } } \sim \mathcal { N } ( m _ { y } , \tau ^ { 2 } \mathbf { \boldsymbol { W } } ) \right.$ for every $\pmb { A } \in \mathbb { R } ^ { m \times n }$ and every $\sigma _ { \mathrm { n } } > 0 .$

Proof. Write $K : = A ^ { \top } ( A A ^ { \top } + \lambda I ) ^ { - 1 }$ , so that $m _ { y } = K y$ and $W = I - K A$ by Proposition 1. Substituting these into equation 13 gives

$$
\begin{array} { r } { x _ { 0 } = m _ { y } + W \epsilon - K n ^ { \prime } . } \end{array}\tag{20}
$$

Given $^ { y , }$ this is an affine function of two independent Gaussian vectors, so $\scriptstyle { \mathbf { { \mathit { x } } } } _ { 0 }$ is Gaussian with mean $m _ { y }$ and covariance $\tau ^ { 2 } W ^ { 2 } + \sigma _ { \mathrm { n } } ^ { 2 } K K ^ { \intercal }$ . Both W and $\pmb { K K } ^ { \top }$ are diagonal in the right singular basi of A, with entries $w _ { i }$ and $s _ { i } ^ { 2 } / ( \stackrel {  } { s _ { i } ^ { 2 } } + \lambda ) ^ { 2 }$ . Using $\sigma _ { \mathrm { n } } ^ { 2 } = \tau ^ { 2 } \lambda ,$ , the i-th diagonal entry of the covariance is

$$
\tau ^ { 2 } w _ { i } ^ { 2 } + \tau ^ { 2 } \lambda \frac { s _ { i } ^ { 2 } } { ( s _ { i } ^ { 2 } + \lambda ) ^ { 2 } } = \tau ^ { 2 } \frac { \lambda ^ { 2 } + \lambda s _ { i } ^ { 2 } } { ( s _ { i } ^ { 2 } + \lambda ) ^ { 2 } } = \tau ^ { 2 } \frac { \lambda } { s _ { i } ^ { 2 } + \lambda } = \tau ^ { 2 } w _ { i } ,
$$

so the covariance equals $\tau ^ { 2 } \mathbf { { W } }$

Each source draw requires applications of A and $A ^ { \top }$ and one solve with $A A ^ { \top } + \lambda I$ . No square root of W is required. Operator-specific implementations are given in App. C.2.

## B Additional Results

## B.1 Full results on AFHQ-Cat

Table 6: AFHQ-Cat, 100 test images. We report single draw (M=1) and the average of M draws for SNaP. Note that a single SNaP draw is competitive in LPIPS at one network evaluation, and averaging a few draws recovers distortion at a cost still far below the iterative samplers. On AFHQ-Cat the number of time steps used by PnP-Flow and Flower depends on the operator, so their NFE does too; the per-operator budgets are listed in App. C.3. The MSE regressor is omitted on box inpainting, where its single output is the blurred conditional mean of the masked region and is not comparable with the other methods. Best and second best are shown in blue and red.

<table><tr><td rowspan="2">Method</td><td rowspan="2">NFE</td><td colspan="3">Deblurring</td><td colspan="3">Super-resolution</td><td colspan="3">Random inpainting</td><td colspan="3">Box inpainting</td></tr><tr><td>PSNR↑</td><td>SSIM↑</td><td>LPIPS↓</td><td>PSNR↑</td><td>SSIM↑</td><td>LPIPS↓</td><td>PSNR↑</td><td>SSIM↑</td><td>LPIPS↓</td><td>PSNR↑</td><td>SSIM↑</td><td>LPIPS↓</td></tr><tr><td>Degraded</td><td></td><td>24.39</td><td>0.532</td><td>0.543</td><td>11.98</td><td>0.212</td><td>0.900</td><td>13.25</td><td>0.214</td><td>1.091</td><td>21.57</td><td>0.736</td><td>0.214</td></tr><tr><td>MSE regressor</td><td>1</td><td>29.82</td><td>0.806</td><td>0.289</td><td>29.16</td><td>0.817</td><td>0.201</td><td>33.23</td><td>0.915</td><td>0.066</td><td></td><td></td><td></td></tr><tr><td>PnP-GS</td><td>23</td><td>28.39</td><td>0.787</td><td>0.387</td><td>24.44</td><td>0.639</td><td>0.411</td><td>29.89</td><td>0.841</td><td>0.123</td><td></td><td></td><td></td></tr><tr><td>PnP-Flow5</td><td>500-2500</td><td>29.01</td><td>0.785</td><td>0.312</td><td>28.01</td><td>0.791</td><td>0.167</td><td>34.47</td><td>0.933</td><td>0.044</td><td>27.14</td><td>0.900</td><td>0.127</td></tr><tr><td>DDRM</td><td>20</td><td>29.01</td><td>0.784</td><td>0.194</td><td>27.09</td><td>0.781</td><td>0.186</td><td>32.20</td><td>0.907</td><td>0.064</td><td></td><td></td><td></td></tr><tr><td>DiffPIR</td><td>100</td><td>28.89</td><td>0.773</td><td>0.187</td><td>23.16</td><td>0.635</td><td>0.291</td><td>31.76</td><td>0.881</td><td>0.053</td><td>27.71</td><td>0.880</td><td>0.059</td></tr><tr><td>OT-ODE</td><td>180</td><td>27.82</td><td>0.735</td><td>0.126</td><td>26.71</td><td>0.737</td><td>0.108</td><td>29.98</td><td>0.849</td><td>0.086</td><td>24.47</td><td>0.873</td><td>0.095</td></tr><tr><td>Flower5-OT DPS</td><td>500-2500</td><td>29.73</td><td>0.801</td><td>0.264</td><td>27.09</td><td>0.767</td><td>0.272</td><td>34.41</td><td>0.932</td><td>0.047</td><td>27.32</td><td>0.922</td><td>0.067</td></tr><tr><td></td><td>1000</td><td>25.90</td><td>0.661</td><td>0.148</td><td>25.98</td><td>0.694</td><td>0.144</td><td>32.06</td><td>0.896</td><td>0.036</td><td>26.78</td><td>0.900</td><td>0.054</td></tr><tr><td>SNAP (M=1)</td><td>1</td><td>26.25</td><td>0.674</td><td>0.159</td><td>26.07</td><td>0.703</td><td>0.130</td><td>30.48</td><td>0.853</td><td>0.066</td><td>26.46</td><td>0.891</td><td>0.054</td></tr><tr><td>SNAP (M=4)</td><td>4</td><td>28.33</td><td>0.750</td><td>0.157</td><td>28.07</td><td>0.777</td><td>0.120</td><td>32.38</td><td>0.896</td><td>0.052</td><td>28.49</td><td>0.914</td><td>0.047</td></tr><tr><td>SNAP (M=16)</td><td>16</td><td>29.09</td><td>0.779</td><td>0.231</td><td>28.77</td><td>0.802</td><td>0.157</td><td>33.04</td><td>0.909</td><td>0.056</td><td>29.32</td><td>0.922</td><td>0.056</td></tr><tr><td>SNAP (M=100)</td><td>100</td><td>29.32</td><td>0.788</td><td>0.320</td><td>28.99</td><td>0.810</td><td>0.187</td><td>33.25</td><td>0.913</td><td>0.058</td><td>29.53</td><td>0.925</td><td>0.069</td></tr></table>

Table 6 reports the same protocol as Tab. 1 on AFHQ-Cat.

![](images/1b305c59323f25aa452f168904b39cda4f8e5c4d56684232a1c4b70f51f96a84.jpg)  
Figure 5: Box inpainting on CelebA.

## B.2 Additional visual results

Figure 5 shows box inpainting on CelebA.

Figure 6 shows box inpainting on AFHQ-Cat.

Figure 7 shows multi-coil fastMRI at ×8 acceleration and 30 dB input SNR.

Figure 8 shows Gaussian deblurring on CelebA.

Figure 9 shows several SNaP draws for box inpainting.

Figure 10 shows further SNaP draws for box inpainting.

Figure 11 shows further SNaP draws for box inpainting.

## B.3 Ablations

Noise misspecification. Since $\sigma _ { \mathrm { n } }$ enters the source in closed form, a wrong value at test time changes how the source allocates uncertainty rather than breaking the method: by equation 10, overestimating $\sigma _ { \mathrm { n } }$ loosens the source toward the working prior, while under-estimating it shrinks the source toward the least-squares anchor. Table 7 tests the first case: the model and its source keep $\sigma _ { \mathrm { n } } = 0 . 0 5$ , while the test measurements carry noise between 0.01 and 0.05, an over-estimate of up to a factor of 5. Reconstruction degrades gracefully. PSNR still rises as the measurements get cleaner, from 32.90 to 34.29 dB, and LPIPS stays between 0.017 and 0.020, although the gain over the input shrinks from 5.09 to 4.33 dB as the mismatch grows. Over this range the method therefore does not require $\sigma _ { \mathrm { n } }$ to be known exactly; under-estimation is not tested here.

Measurement  
DPS  
DiffPIR  
Flower1-OT  
SNAP  
Ground truth  
![](images/55a0e028a809f8859da4fb8b572759c58ff2558e6571fa34d6f9717fef384061.jpg)

Figure 6: Box inpainting on AFHQ-Cat.  
Table 7: CelebA Gaussian deblurring trained at $\sigma _ { \mathrm { n } } = 0 . 0 5$ and evaluated on y with different measurement noise level. † marks the matched case, where the test noise equals the training noise.
<table><tr><td>Measurement noise  $\sigma _ { \mathrm { n } }$ </td><td>PSNR↑</td><td>SSIM↑</td><td>LPIPS↓</td><td>input PSNR</td><td>Gain</td></tr><tr><td>0.01</td><td>34.29</td><td>0.940</td><td>0.020</td><td>29.96</td><td>+4.33</td></tr><tr><td>0.02</td><td>34.15</td><td>0.939</td><td>0.019</td><td>29.62</td><td>+4.53</td></tr><tr><td>0.03</td><td>33.89</td><td>0.936</td><td>0.018</td><td>29.12</td><td>+4.77</td></tr><tr><td>0.04</td><td>33.50</td><td>0.930</td><td>0.017</td><td>28.52</td><td>+4.98</td></tr><tr><td> $0 . 0 5 ^ { \dagger }$ </td><td>32.90</td><td>0.919</td><td>0.018</td><td>27.81</td><td>+5.09</td></tr></table>

Working-prior scale τ. Proposition 2 states that τ shapes the source and the transport but not the target, so it is a design parameter rather than a modelling choice. Table 8 reports CelebA Gaussian deblurring for $\tau \in \{ 0 . 1 5 , 0 . 5 , 1 . 0 \}$ . Distortion is indistinguishable between $\tau = 0 . 1 5$ and $\tau = 0 . 5$ at every M and falls by about 0.4 dB at $\tau = 1 . 0$ , while LPIPS is best at $\tau = 0 . 5$ . The relative spread of the draws is 5.49%, 5.48% and 5.72% for the three settings, essentially unchanged, as the invariance predicts. The method therefore does not require τ to be tuned precisely, and we use the per-operator values listed in App. C.3.

![](images/19417d321b4f4b0920baffe15666782e07a05091e1bb911292dc2871fee2b234.jpg)  
Figure 7: fastMRI brain, multi-coil, ×8 acceleration at 30 dB input SNR.

Sampling ratio $( r , t )$ . The fraction $p _ { \mathrm { r a t i o } } ~ : = ~ \mathrm { P r } [ r = t ]$ of training samples drawn with $r \ = \ t$ controls how often the objective reduces to plain conditional flow matching. Table 9 reports CelebA Gaussian deblurring at $\tau = 0 . 1 5$ for $p _ { \mathrm { r a t i o } } \in \{ 0 . 2 5 , 0 . 5 , 0 . 7 5 \}$ . Setting $p _ { \mathrm { r a t i o } } = 0 . 2 5$ is worse at every $M ,$ by about 0.3 dB PSNR at a single draw, while 0.5 and 0.75 are indistinguishable within the precision reported. We therefore use $p _ { \mathrm { r a t i o } } = 0 . 5$ throughout. The trend in M is the same for all three settings, with PSNR and SSIM improving as draws are averaged and LPIPS degrading, as the average moves toward the posterior mean.

![](images/f0c31d0561937e7bbd33bd5f879f9dbdcadd91598a0ff01a3dfef13881c12d53.jpg)  
Figure 8: Gaussian deblurring on CelebA.

![](images/f4e1cbac24138560e835dea9a9a66a2af01812176d791293c5ae691f9c346ab1.jpg)  
Figure 9: Independent SNaP draws for box inpainting.

Few-step composition. Composing $\mathbf { \Delta } \mathbf { u } _ { r , t }$ over a uniform partition of [0, 1] gives k-step sampling with no retraining, and Table 10 reports $k \in \{ 1 , 2 , 4 , 8 \}$ on CelebA Gaussian deblurring. More steps are not monotonically better. All metrics peak at $k = 2$ and decline afterwards, and $k = 8$ is worse than a single step for $M \geq 1 6$ . The model is trained to jump from the source to the data, so discretizing the trajectory evaluates $u ^ { \theta }$ at intermediate (r, t) pairs at which its error compounds across steps rather than cancelling. The gain from k = 2 is small, so we report k = 1 throughout.

![](images/e53175320b55e5b692d87a2b0ef1b550f4c0372f0a4114404b59bfe8fe46cb88.jpg)

Figure 10: Further independent SNaP draws for box inpainting.  
![](images/99f4716ef0a6381c3c3cf10b9de3770ded245d8b5ce1af425379ba77505671b2.jpg)  
Figure 11: Further independent SNaP draws for box inpainting.

Source distribution. Table 5 isolates the source quantitatively; Figures 12 and 13 show the same comparison sample by sample and as a function of the averaging budget M. All three settings share the network, the conditioning, and the training schedule, differing only in the distribution the one-step map starts from: our measurement-adapted posterior source at $\tau = 0 . 1 5$ versus isotropic Gaussian sources at $\tau = 0 . 1 5$ and $\tau = 1 . 0$

Figure 12 shows six draws per source for one measurement. Under the proposed posterior source the draws differ visibly in the hair, the skin texture and the region occluded by the sunglasses, and the standard deviation over draws concentrates on exactly the edges and high-frequency structures that the blur has suppressed, leaving the smooth background, which the measurement already determines, near zero. Under either isotropic Gaussian source the six draws are visually indistinguishable and the standard-deviation map is essentially black at the same ×20 gain: the map has become a point estimator that happens to take a random input. Raising the Gaussian scale from $\tau = 0 . 1 5$ to $\tau = 1 . 0$ does not restore the spread, so the shape of the source is what dictates the spread rather than its size. Figure 13 makes the consequence quantitative.

## B.4 Comparison with VFM

VFM [33] also places the measurement in the source, but learns it: an adapter $q _ { \phi } ( z | \boldsymbol { y } )$ is trained jointly with an unconditional flow map. Its results are reported in the latent space of a pretrained autoencoder, whereas $\mathrm { { \bf S N a P } }$ operates in pixel space, so the published numbers are not directly comparable. We therefore compare the two constructions on the 2D benchmark of [33]. The prior is a 4×4 checkerboard $\mathsf { o n } [ - 2 , 2 ] ^ { 2 }$ with 20k training samples, observed through $\pmb { y } = \pmb { a } ^ { \top } \pmb { x } + \varepsilon$ with $\varepsilon \sim \mathcal { N } ( 0 , \sigma ^ { 2 } )$ and $\sigma \ = \ 0 . 1 .$ We use two forward operators that differ only in the direction of a, an aligned operator ${ \pmb a } = ( 1 , 0 )$ , which is the one used in the original paper, and a rotated operator $\begin{array} { r } { \pmb { a } = ( \cos \frac { \pi } { 5 } , \sin \frac { \pi } { 5 } ) } \end{array}$ All methods share one backbone, a 6-layer width-512 SiLU MLP in x -prediction form with random-Fourier embeddings of r, t and σ, and one budget, AdamW with learning rate $2 \times 1 0 ^ { - 4 }$ , batch 2048, 50k steps and EMA 0.999. The only thing that varies is where the measurement enters the generative process. The learned-adapter baseline follows VFM. The flow map never sees y, and all measurement information is carried by a diagonal Gaussian adapter $q _ { \phi } ( z | \pmb { y } ) = \mathcal { N } ( \pmb { \mu } _ { \phi } ( \pmb { y } )$ , diag $\sigma _ { \phi } ^ { 2 } ( \pmb { y } ) )$ trained jointly with the flow map under the variational objective, with an observation term, a KL term and an EMA copy of the weights for stability. We also report a frozen-θ ablation in which the pretrained flow map is held fixed and only $q _ { \phi }$ is trained. SNaP instead keeps the flow map conditioned on y and draws its source in closed form by equation 13, with no adapter and no auxiliary losses.

Table 8: Working-prior scale τ . CelebA Gaussian deblurring, 100 test measurements, input PSNR 27.24 dB. M is the number of averaged draws
<table><tr><td rowspan="2">M</td><td colspan="3"> $\tau = 0 . 1 5$ </td><td colspan="3"> $\tau = 0 . 5$ </td><td colspan="3"> $\tau = 1 . 0$ </td></tr><tr><td>PSNR↑</td><td>SSIM↑</td><td>LPIPS↓</td><td>PSNR↑</td><td>SSIM↑</td><td>LPIPS↓</td><td>PSNR↑</td><td>SSIM↑</td><td>LPIPS↓</td></tr><tr><td>1</td><td>32.941</td><td>0.9198</td><td>0.0190</td><td>32.922</td><td>0.9187</td><td>0.0186</td><td>32.512</td><td>0.9060</td><td>0.0244</td></tr><tr><td>4</td><td>34.966</td><td>0.9463</td><td>0.0209</td><td>34.948</td><td>0.9457</td><td>0.0181</td><td>34.588</td><td>0.9385</td><td>0.0197</td></tr><tr><td>16</td><td>35.688</td><td>0.9537</td><td>0.0275</td><td>35.663</td><td>0.9532</td><td>0.0234</td><td>35.323</td><td>0.9475</td><td>0.0249</td></tr><tr><td>100</td><td>35.913</td><td>0.9558</td><td>0.0300</td><td>35.885</td><td>0.9553</td><td>0.0256</td><td>35.552</td><td>0.9500</td><td>0.0273</td></tr></table>

Table 9: (r, t) sampling ratio. CelebA Gaussian deblurring, $\tau = 0 . 1 5$ , 100 test measurements, input PSNR 27.24 dB. M is the number of averaged draws.
<table><tr><td rowspan="2">M</td><td colspan="3"> $p _ { \mathrm { r a t i o } } = 0 . 2 5$ </td><td colspan="3"> $p _ { \mathrm { r a t i o } } = 0 . 5$ </td><td colspan="3"> $p _ { \mathrm { r a t i o } } = 0 . 7 5$ </td></tr><tr><td>PSNR↑</td><td>SSIM↑</td><td>LPIPS↓</td><td>PSNR↑</td><td>SSIM↑</td><td>LPIPS↓</td><td>PSNR↑</td><td>SSIM↑</td><td>LPIPS↓</td></tr><tr><td>1</td><td>32.567</td><td>0.9130</td><td>0.0216</td><td>32.941</td><td>0.9198</td><td>0.0190</td><td>32.976</td><td>0.9210</td><td>0.0189</td></tr><tr><td>4</td><td>34.679</td><td>0.9429</td><td>0.0197</td><td>34.966</td><td>0.9463</td><td>0.0209</td><td>34.976</td><td>0.9465</td><td>0.0206</td></tr><tr><td>16</td><td>35.438</td><td>0.9513</td><td>0.0266</td><td>35.688</td><td>0.9537</td><td>0.0275</td><td>35.682</td><td>0.9536</td><td>0.0268</td></tr><tr><td>100</td><td>35.676</td><td>0.9537</td><td>0.0296</td><td>35.913</td><td>0.9558</td><td>0.0300</td><td>35.902</td><td>0.9557</td><td>0.0292</td></tr></table>

The source covariance $\pmb { W } = \pmb { I } - \pmb { a } \pmb { a } ^ { \top } / ( \| \pmb { a } \| _ { 2 } ^ { 2 } + \lambda )$ is diagonal in the basis $\{ a , a ^ { \perp } \}$ , and in the coordinate basis only when the two coincide. For ${ \pmb a } = ( 1 , 0 )$ they coincide, so a diagonal adapter is exactly expressive and the aligned operator is a parity check. Rotating a breaks the coincidence while holding the prior, the noise level, the architecture and the training budget fixed. The resulting gap can be quantified before any training: the KL divergence from $\mathcal { N } ( \boldsymbol { m _ { y } } , \tau ^ { 2 } \mathbf { \bar { W } } )$ to its best diagonal approximation is 0 nats for the aligned operator at every τ, and 1.576 nats for the rotated at $\tau = 1$

On the aligned operator, where the diagonal adapter can represent the exact source, all three methods are close, with SNaP already the most accurate (Table 11). On the rotated operator the two constructions separate. The jointly trained adapter degrades by roughly a factor of seven and the frozen variant by more than an order of magnitude, while SNaP changes far less, since its source rotates with the operator by construction. The gaps are significant under a Welch test against the jointly trained adapter, with $p = 0 . 0 0 0 5$ on the aligned operator and $p = 0 . 0 3 8$ on the rotated one. Figure 14 shows the mechanism, and the effect is also visible in the support of the samples, with a support accuracy of $0 . 9 7 \pm 0 . 0 0 4$ for SNaP against $0 . 8 1 \pm 0 . 0 3$ for the jointly trained adapter on the rotated operator.

## B.5 Comparison with NullFlow

NullFlow is defined only for noiseless measurements, so it cannot enter the benchmarks of Sec. 4. We instead sweep the test noise on CelebA box inpainting from $\sigma _ { n } = 0$ , where NullFlow is exact, to the $\sigma _ { n } ~ = ~ 0 . 0 5$ of our main experiments (Table 12). NullFlow is trained noiselessly with an LPIPS and feature loss, and SNaP at $\sigma _ { n } = 0 . 0 5$ with the standard MeanFlow loss, so LPIPS favors NullFlow.

Table 10: Trajectory steps k. CelebA Gaussian deblurring, $\tau = 0 . 1 5$ , 100 test measurements, input PSNR 27.24 dB. M is the number of averaged draws. Best in each row and metric is in bold.
<table><tr><td rowspan="2">M</td><td colspan="3"> $k = 1$ </td><td colspan="3"> $k = 2$ </td><td colspan="3"> $k = 4$ </td><td colspan="3"> $k = 8$ </td></tr><tr><td>PSNR↑</td><td>SSIM↑</td><td>LPIPS↓</td><td>PSNR↑</td><td>SSIM↑</td><td>LPIPS↓</td><td>PSNR↑</td><td>SSIM↑</td><td>LPIPS↓</td><td>PSNR↑</td><td>SSIM↑</td><td>LPIPS↓</td></tr><tr><td>1</td><td>32.941</td><td>0.9198</td><td>0.0190</td><td>33.084</td><td>0.9211</td><td>0.0172</td><td>33.037</td><td>0.9205</td><td>0.0172</td><td>32.992</td><td>0.9200</td><td>0.0173</td></tr><tr><td>4</td><td>34.966</td><td>0.9463</td><td>0.0209</td><td>35.044</td><td>0.9469</td><td>0.0206</td><td>35.001</td><td>0.9465</td><td>0.0206</td><td>34.942</td><td>0.9458</td><td>0.0210</td></tr><tr><td>16</td><td>35.688</td><td>0.9537</td><td>0.0275</td><td>35.729</td><td>0.9540</td><td>0.0275</td><td>35.688</td><td>0.9536</td><td>0.0276</td><td>35.621</td><td>0.9529</td><td>0.0281</td></tr><tr><td>100</td><td>35.913</td><td>0.9558</td><td>0.0300</td><td>35.943</td><td>0.9561</td><td>0.0302</td><td>35.902</td><td>0.9557</td><td>0.0303</td><td>35.833</td><td>0.9550</td><td>0.0308</td></tr></table>

![](images/2044cbdec1871b5159f2e81f32cdaf18cacb1b55b54ccffb62d717b7cb6cde43.jpg)  
Figure 12: Source distribution, qualitative. Six independent draws from a fixed CelebA Gaussiandeblurring measurement under the three sources, with the per-pixel standard deviation over draws on the same ×20 scale in every row. The posterior source varies where the measurement is uninformative; both Gaussian sources collapse onto a single reconstruction.

Table 12 reports both methods on CelebA box inpainting, using a single draw and λ frozen at each model’s training value. $\mathrm { { A t } } \ \sigma _ { \mathrm { { n } } } = 0$ , NullFlow is the stronger method, which is expected, since its source is exact in the noiseless limit and every state satisfies $\mathbf { \nabla } A \mathbf { \mathit { x } } = \mathbf { \mathit { y } }$ . Its accuracy falls quickly as noise is added, and by $\sigma _ { \mathrm { n } } = 0 . 0 5$ its SSIM has dropped from 0.977 to 0.855 and its LPIPS has more than quadrupled. This is the failure mode identified in Sec. 2, where the least-squares anchor carries $A ^ { \dagger } n$ that a null-space flow cannot remove. SNaP changes little over the same range and is the better method at $\sigma _ { \mathrm { n } } = 0 . 0 5$ on all three metrics. The two methods therefore occupy different regimes, with NullFlow suited to noiseless measurements and SNaP to noisy ones.

## B.6 Irreducible regression error: SNaP vs. isotropic sources

Proposition 2 fixes the target of the exact one-step map, but not how hard that map is to learn. We therefore compare sources through the regression problem that training solves. $\mathbf { A } \mathbf { t } \ r = t$ the Mean-Flow target equation 14 reduces to the conditional velocity ${ \pmb x } _ { 1 } - { \pmb x } _ { 0 }$ , which the network regresses on $( z _ { t } , y )$ with $z _ { t } = ( 1 - t ) { \pmb x } _ { 0 } + t { \pmb x } _ { 1 }$ . The error left by the best predictor,

$$
\mathcal { E } _ { t } ( \pmb { y } ) : = \mathbb { E } [ \mathrm { t r } \mathrm { C o v } ( \pmb { x } _ { 1 } - \pmb { x } _ { 0 } | \pmb { z } _ { t } , \pmb { y } ) | \pmb { y } ] ,\tag{21}
$$

cannot be removed by any network, so a source with a smaller $\mathcal { E } _ { t }$ poses an easier regression problem at time t.

Lemma 2. Let $\sigma _ { \mathrm { n } } > 0 ,$ , fix y and $t \in ( 0 , 1 )$ , and let $\pmb { x } _ { 1 } \sim p ( \cdot | \pmb { y } )$ have finite second moments. Let $\pmb { x } _ { 0 } \sim q _ { \tau } ( \cdot | \pmb { y } )$ , and let $\widetilde { \pmb { x } } _ { 0 } = \widetilde { \pmb { K } } \pmb { y } + \widetilde { \pmb { \xi } }$ be a source oftheform equation 6 with $\widetilde { \pmb { \xi } } \sim \mathcal { N } ( \mathbf { 0 } , \widetilde { \pmb { \Sigma } } )$ and $\widetilde { \pmb { \Sigma } } \succeq$ $\tau ^ { 2 } \mathbf { { W } }$ , each conditionally independent of $\mathbf { \dot { x } } _ { 1 }$ given y. Let $\mathcal { E } _ { t } ( \pmb { y } )$ and $\widetilde { \mathcal { E } } _ { t } ( y )$ be the errors equation 21 for the interpolants $z _ { t } = ( 1 - t ) \pmb { x } _ { 0 } + t \pmb { x } _ { 1 }$ and $\widetilde { z } _ { t } = ( 1 - t ) \widetilde { \pmb { x } } _ { 0 } + t { \pmb { x } } _ { 1 }$ . Then $\mathcal { E } _ { t } ( \pmb { y } ) \leq \widetilde { \mathcal { E } } _ { t } ( \pmb { y } )$ . In particular, this holdsfor the isotropic source ${ \mathcal { N } } ( \mathbf { 0 } , \sigma ^ { 2 } I )$ for every $\sigma \geq \tau ,$ , since $\tau ^ { 2 } W \preceq \tau ^ { 2 } I \preceq \sigma ^ { 2 } I$

The lemma requires nothing of the posterior beyond finite second moments, and the center of the source does not enter it: given $^ { \mathbf { \delta } _ { \mathbf { \delta } _ { \mathbf { \delta } _ { \mathbf { \delta } _ { \mathbf { \delta } _ { \mathbf { \delta } _ { \delta } } } } } } }$ the center only shifts ${ \boldsymbol { z } } _ { t }$ by a constant, which conditioning removes. The center is what Proposition 1 chooses to shorten the displacement; the lemma shows that the covariance $\tau ^ { 2 } \mathbf { { W } }$ is what lowers the regression error. Both isotropic sources of Table 5 satisfy its assumption, since $\tau = 0 . 1 5 \leq 1$ there.

![](images/88debcca54e8b67c44353efe4d331d2cd0babfd89866659e89fed7ab7794bac2.jpg)

![](images/8db2b95fd396e623b3052a92f148e0a490f1539a982703828728bff8f65e01b0.jpg)  
Figure 13: Source distribution, averaging budget. PSNR (left, higher is better) and LPIPS (right, lower is better) against the number of averaged draws M on CelebA Gaussian deblurring. Only the posterior source responds to $M ;$ ; the Gaussian sources are flat because their draws carry almost no variance to average away.

![](images/9272d8a796cbd2fd3b78ad21ef3eafd2f3ca6a2640eb703c9345d95b1c1cfd1c.jpg)  
Figure 14: One-step $( K { = } 1 )$ posterior samples on the 4×4 checkerboard. Rows are the two forward operators and columns are the learned adapter of VFM, the SNaP source, and the exact rejectionsampled posterior. The green line is the measurement constraint ${ \pmb a } ^ { \top } { \pmb x } = { \pmb y } .$ , and the dashed ellipse on each method panel is that method’s 2σ source covariance, the learned diagonal $q _ { \phi } ( z | \boldsymbol { y } )$ for VFM and the closed-form $\tau ^ { 2 } W$ for SNaP. In the top row the measurement direction is axis-aligned, both sources align with the constraint, and both recover the posterior. In the bottom row the constraint tilts. The $\bar { \bf S N a P }$ source tilts with it, whereas the diagonal adapter cannot and degenerates to a nearisotropic blob, so its samples concentrate on one mode and spill off the checkerboard support.

Proof. Given y, write ${ \pmb x } _ { 0 } = m _ { y } + \pmb \eta$ and $\widetilde { \pmb { x } } _ { 0 } = \pmb { c } + \widetilde { \pmb { \xi } }$ with $c : = \widetilde { K } y$ , where $\pmb { \eta } \sim \mathcal { N } ( \mathbf { 0 } , \tau ^ { 2 } W )$ and $\boldsymbol { \xi }$ are independent of $\scriptstyle { \mathbf { { \vec { x } } } } _ { 1 }$ . Since $m _ { y }$ and c are constants given y, the maps $z \mapsto ( z - ( 1 - t ) m _ { y } ) / t$ and $z \mapsto ( z - ( 1 - t ) \pmb { c } ) / t$ are invertible, so conditioning on $z _ { t }$ and on $\widetilde { z } _ { t }$ is the same as conditioning on

$$
\begin{array} { r } { \bar { z } _ { t } : = \pmb { x } _ { 1 } + \frac { 1 - t } { t } \pmb { \eta } \qquad \mathrm { a n d } \qquad \bar { \tilde { z } } _ { t } : = \pmb { x } _ { 1 } + \frac { 1 - t } { t } \widetilde { \xi } , } \end{array}
$$

respectively. Each is $\scriptstyle { \mathbf { \mathscr { x } } } _ { 1 }$ observed through additive Gaussian noise, and the two differ only in that noise.

Table 11: Posterior $\mathrm { M M D ^ { 2 } \left( \downarrow \right) }$ on the two-dimensional checkerboard benchmark at $K { = } 1$ , mean ± standard deviation over three seeds, each at its validation-selected setting. The aligned operator is a parity check, where a diagonal adapter is exactly expressive, and the rotated operator differs only in the direction of a.
<table><tr><td>Method</td><td>Aligned (parity)</td><td>Rotated (discriminating)</td></tr><tr><td>SNaP</td><td> $0 . 0 0 2 7 \pm 0 . 0 0 0 3$ </td><td> $0 . 0 1 7 1 \pm 0 . 0 0 1 2$ </td></tr><tr><td>VFM (joint)</td><td> $0 . 0 1 0 9 \pm 0 . 0 0 0 7$ </td><td> $0 . 0 7 3 7 \pm 0 . 0 1 9 9$ </td></tr><tr><td>VFM (frozen θ)</td><td> $0 . 0 1 0 2 \pm 0 . 0 0 0 3$ </td><td> $0 . 1 9 0 0 \pm 0 . 0 0 1 5$ </td></tr></table>

Table 12: NullFlow vs. SNaP on CelebA box inpainting (M=1) as test noise grows.
<table><tr><td rowspan="2">Test  $\sigma _ { \mathrm { n } }$ </td><td colspan="3">NullFlow</td><td colspan="3">SNaP</td></tr><tr><td>PSNR↑</td><td>SSIM↑</td><td>LPIPS↓</td><td>PSNR↑</td><td>SSIM↑</td><td>LPIPS↓</td></tr><tr><td>0.00</td><td>33.50</td><td>0.977</td><td>0.012</td><td>31.37</td><td>0.947</td><td>0.025</td></tr><tr><td>0.01</td><td>33.27</td><td>0.970</td><td>0.012</td><td>31.35</td><td>0.947</td><td>0.024</td></tr><tr><td>0.02</td><td>32.65</td><td>0.952</td><td>0.016</td><td>31.30</td><td>0.946</td><td>0.023</td></tr><tr><td>0.05</td><td>30.11</td><td>0.855</td><td>0.055</td><td>30.74</td><td>0.934</td><td>0.021</td></tr></table>

Since $\widetilde { \pmb { \Sigma } } - \tau ^ { 2 } \pmb { W } \succeq \mathbf { 0 }$ , let $\zeta \sim \mathcal { N } ( \mathbf { 0 } , \widetilde { \Sigma } - \tau ^ { 2 } W )$ be independent of $( \pmb { x } _ { 1 } , \pmb { \eta } )$ given y. Then $\eta + \zeta \sim$ $\mathcal { N } ( \mathbf { 0 } , \widetilde { \pmb { \Sigma } } )$ , and because $\widetilde { \mathcal { E } } _ { t } ( y )$ depends only on the joint law of $\scriptstyle { \mathbf { \mathscr { x } } } _ { 1 }$ and $\widetilde { z } _ { t }$ given y, we may take $\widetilde { \pmb { \xi } } = \pmb { \eta } + \pmb { \zeta }$ . This gives $\begin{array} { r } { \bar { \tilde { z } } _ { t } = \bar { z } _ { t } + \frac { 1 - t } { t } \zeta , } \end{array}$ so ${ \pmb x } _ { 1 }  \bar { \pmb z } _ { t }  \bar { \hat { \pmb z } } _ { t }$ is a Markov chain given y.

Let $\pmb { \mu } _ { t } : = \mathbb { E } [ \pmb { x } _ { 1 } | \bar { \pmb { z } } _ { t } , \pmb { y } ]$ and $\widetilde { \pmb \mu } _ { t } : = \mathbb { E } [ \pmb { x } _ { 1 } | \bar { \widetilde { \pmb z } } _ { t } , \pmb y ]$ . The Markov property gives $\mathbb { E } [ { \pmb x } _ { 1 } \mid \bar { { \boldsymbol z } } _ { t } , \bar { \tilde { { \boldsymbol z } } } _ { t } , { \pmb y } ] = { \pmb \mu } _ { t }$ and the tower rule gives $\widetilde { \pmb { \mu } } _ { t } = \mathbb { E } [ \pmb { \mu } _ { t } | \bar { \widetilde { \pmb { z } } } _ { t } , \pmb { y } ]$ . Hence ${ \pmb x } _ { 1 } - { \pmb \mu } _ { t }$ is orthogonal to $\pmb { \mu } _ { t } - \widetilde { \pmb { \mu } } _ { t }$ , and

$$
\begin{array} { r } { \mathbb { E } \Big [ \| \boldsymbol { x } _ { 1 } - \widetilde { \boldsymbol { \mu } } _ { t } \| _ { 2 } ^ { 2 } \vert \boldsymbol { y } \Big ] = \mathbb { E } \Big [ \| \boldsymbol { x } _ { 1 } - \boldsymbol { \mu } _ { t } \| _ { 2 } ^ { 2 } \vert \boldsymbol { y } \Big ] + \mathbb { E } \Big [ \| \boldsymbol { \mu } _ { t } - \widetilde { \boldsymbol { \mu } } _ { t } \| _ { 2 } ^ { 2 } \vert \boldsymbol { y } \Big ] . } \end{array}\tag{22}
$$

Finally, ${ \pmb x } _ { 1 } - { \pmb x } _ { 0 } = ( { \pmb x } _ { 1 } - { \pmb z } _ { t } ) / ( 1 - t )$ and ${ \boldsymbol { z } } _ { t }$ is fixed once it is conditioned on, so $\mathrm { C o v } ( { \pmb x } _ { 1 } \textrm { -- }$ $\pmb { x } _ { 0 } \mid z _ { t } , y ) = ( 1 - t ) ^ { - 2 } \operatorname { C o v } ( \pmb { x } _ { 1 } \mid z _ { t } , y )$ and $\mathscr { E } _ { t } ( \pmb { y } ) = ( 1 - t ) ^ { - 2 } \mathbb { E } [ \left\| \pmb { x } _ { 1 } - \pmb { \mu } _ { t } \right\| _ { 2 } ^ { 2 } | \pmb { y } ]$ ; likewise $\widetilde { \mathcal { E } } _ { t } ( y ) =$ $( 1 - t ) ^ { - 2 } \mathbb { E } [ \left\| \pmb { x } _ { 1 } - \widetilde { \pmb { \mu } } _ { t } \right\| _ { 2 } ^ { 2 } | \pmb { y } ]$ . Dividing equation 22 by $( 1 - t ) ^ { 2 }$ gives

$$
\widetilde { \mathcal { E } } _ { t } ( { \pmb y } ) - \mathcal { E } _ { t } ( { \pmb y } ) = \frac { 1 } { ( 1 - t ) ^ { 2 } } \mathbb { E } \Big [ \| { \pmb \mu } _ { t } - { \widetilde { \pmb \mu } } _ { t } \| _ { 2 } ^ { 2 } | { \pmb y } \Big ] \ \ge \ 0 .
$$

Displacement. The same source also shortens the path relative to the isotropic source. Let $\widetilde { \pmb { x } } _ { 0 } \sim$ ${ \mathcal { N } } ( { \bar { \mathbf { 0 } } } , \tau ^ { 2 } I )$ be independent of $( \pmb { x } _ { 1 } , \pmb { n } )$ , and let $E _ { i } : = \mathbb { E } [ ( { \pmb v } _ { i } ^ { \top } { \pmb x } _ { 1 } ) ^ { 2 } ]$ for a right singular vector ${ \mathbf { } } v _ { i }$ of A. For any $\scriptstyle { \mathbf { \mathscr { x } } } _ { 1 }$ with finite second moments,

$$
\begin{array} { r } { \mathbb { E } \big [ ( { \pmb v } _ { i } ^ { \top } ( { \pmb x } _ { 1 } - \widetilde { { \pmb x } } _ { 0 } ) ) ^ { 2 } \big ] - \mathbb { E } \big [ ( { \pmb v } _ { i } ^ { \top } ( { \pmb x } _ { 1 } - { \pmb x } _ { 0 } ) ) ^ { 2 } \big ] = ( 1 - { \pmb w } _ { i } ^ { 2 } ) E _ { i } + \tau ^ { 2 } ( 1 - { \pmb w } _ { i } ) ^ { 2 } \ \ge \ 0 , } \end{array}\tag{23}
$$

with equality if and only if $s _ { i } = 0 .$ . Indeed, writing $a _ { i } : = { \pmb v } _ { i } ^ { \top } { \pmb x } _ { 1 }$ and $n _ { i } : = { \pmb u } _ { i } ^ { \top } { \pmb n }$ (with $n _ { i } : = 0$ if $s _ { i } = 0 )$ , the Tikhonov center gives $\begin{array} { r } { \pmb { v } _ { i } ^ { \top } ( \pmb { x } _ { 1 } - \pmb { x } _ { 0 } ) = w _ { i } \pmb { a } _ { i } - \frac { s _ { i } } { s _ { \bot } ^ { 2 } + \lambda } \pmb { n } _ { i } - \pmb { v } _ { i } ^ { \top } \pmb { \eta } , } \end{array}$ , a sum of uncorrelated terms with second moments $w _ { i } ^ { 2 } E _ { i } , \tau ^ { 2 } w _ { i } ( 1 - w _ { i } )$ and $\tau ^ { 2 } w _ { i }$ , whereas the isotropic source gives $E _ { i } + \tau ^ { 2 }$

Evaluation. Figure 15 evaluates $\mathcal { E } _ { t }$ for $q _ { \tau } ( \cdot | \mathbf { \boldsymbol { y } } )$ and ${ \mathcal { N } } ( \mathbf { 0 } , \tau ^ { 2 } I )$ in two settings where it is available exactly. On the two-dimensional mixture of App. B.7 at $y = 2 . 5$ , the conditional covariance is computed by quadrature against the closed-form posterior. The two sources coincide in the null direction there, so the gap comes entirely from the observed direction. On CelebA Gaussian deblurring, gap is evaluated under a stationary Gaussian prior with the empirical power spectrum, for which A, W and the posterior covariance are simultaneously diagonal in the Fourier basis. There the gap is present at every t and peaks at 56%, since the graded Fourier weights $w _ { i }$ leave the source something to exploit in every direction.

![](images/f569da667bf9d98f0b2fec38354d2b3b5f796ba42813c737ae59cdf2c9087eef.jpg)

![](images/e41041c10f49e322dacceda4eb52fc57e11a5a032c83c187dde3992f288e1473.jpg)  
Figure 15: Irreducible velocity-regression error $\mathcal { E } _ { t }$ of equation 21 under the SNaP source $q _ { \tau } ( \cdot | \mathbf { \boldsymbol { y } } )$ and the isotropic source $\mathcal { N } ( \mathbf { 0 } , \dot { \tau } ^ { 2 } I )$ . Left: two-dimensional mixture of $\mathbf { A p p }$ ${ \bf B } . 7 \left( \tau = 3 , \sigma _ { \mathrm { n } } = 0 . 3 \right.$ $y = 2 . 5 )$ , exact. Right: CelebA Gaussian deblurring $( \tau = 0 . 1 5 , \sigma _ { \mathrm { n } } = 0 . 0 5 )$ under a stationary Gaussian prior. Lemma 2 guarantees the ordering; the shaded region is the gap.

Scope. Three limits apply. First, the lemma concerns the $r = t$ part of the objective. For $r < t$ the target equation 14 also contains the network’s own derivatives, so the lemma does not transfer verbatim to the interior of the $( r , t )$ range; a fraction $p _ { \mathrm { r a t i o } }$ of training pairs have $r = t \left( { \mathrm { T a b l e } } 9 \right)$ Second, like the displacement in equation $7 , \mathcal { E } _ { t }$ decreases as the source covariance shrinks, so it cannot be used to choose Σ, and the lemma only compares $q _ { \tau }$ with sources that are at least as spread. A narrower source can have a smaller $\mathcal { E } _ { t } \mathrm { : }$ on the mixture of App. B.7, where $\tau = 3 ,$ , the source $\mathcal { N } ( \mathbf { 0 } , \pmb { I } )$ has a smaller $\mathcal { E } _ { t }$ than $q _ { \tau }$ at every t, because it is narrower in the null direction. Third, the lemma compares regression problems, not trained samplers; Table 5 and App. B.7 compare the samplers directly, with everything but the source held fixed.

App. B.6 shows that, for any posterior with finite second moments, the velocity-regression error that no network can remove is smaller under $q _ { \tau } ( \cdot | \mathbf { \boldsymbol { y } } )$ than under any isotropic source ${ \mathcal { N } } ( \mathbf { 0 } , \sigma ^ { 2 } I )$ with $\sigma \geq \tau$ (Lemma 2); Table 5 and $\mathrm { A p p }$ . B.7 measure the effect on the learned map.

## B.7 Verification on a Gaussian-mixture inverse problem

We verify the construction on a problem where the posterior is available in closed form. With a fourcomponent Gaussian-mixture prior on $\mathbb { R } ^ { 2 }$ and observation of the first coordinate $( A = e _ { \mathrm { 0 } } ^ { \mathrm { T } } , \sigma _ { \mathrm { n } } =$ 0.3), the posterior is itself a Gaussian mixture, and for $y \approx 2 . 5$ it is bimodal over x $[ 1 ] \in \{ \pm 2 . 5 \}$ with the MMSE estimate falling between the modes. The setting therefore separates posterior sampling from posterior-mean estimation: a point estimator lands in the empty valley between the modes, and a mode-collapsed sampler reports the component variance $\varsigma ^ { 2 } = 0 . 1 \dot { 2 }$ in the null direction rather than the posterior variance 6.36.

Setup. The clean data $\pmb { x } _ { 1 } \in \mathbb { R } ^ { 2 }$ is drawn from a four-component equal-weight Gaussian mixture with isotropic component covariance $\varsigma ^ { 2 } I \left( \varsigma = 0 . 3 5 \right)$ and means at $( \pm a , \pm a ) , a = 2 . 5$ . The forward operator observes only the first coordinate, so $m = 1 , n = 2$ and nul $1 ( A ) = \operatorname { s p a n } ( e _ { 1 } )$ . The working-prior scale is $\tau = 3$

Closed-form posterior. For each mixture component $k ,$

$$
\widetilde { \boldsymbol { \Sigma } } _ { k } = \left( \boldsymbol { \Sigma } _ { k } ^ { - 1 } + \boldsymbol { A } ^ { \top } \boldsymbol { A } / \sigma _ { \mathrm { n } } ^ { 2 } \right) ^ { - 1 } , \quad \widetilde { \boldsymbol { \mu } } _ { k } = \widetilde { \boldsymbol { \Sigma } } _ { k } \left( \boldsymbol { \Sigma } _ { k } ^ { - 1 } \boldsymbol { \mu } _ { k } + \boldsymbol { A } ^ { \top } \boldsymbol { y } / \sigma _ { \mathrm { n } } ^ { 2 } \right) ,\tag{24}
$$

and

$$
\widetilde { \pi } _ { k } \propto \pi _ { k } { \mathcal { N } } \big ( y ; A \mu _ { k } , A \Sigma _ { k } A ^ { \top } + \sigma _ { \mathrm { n } } ^ { 2 } \big ) , \qquad p ( x _ { 1 } | y ) = \sum _ { k } \widetilde { \pi } _ { k } { \mathcal { N } } ( \widetilde { \mu } _ { k } , \widetilde { \Sigma } _ { k } ) .\tag{25}
$$

All quantities labeled “posterior” below are computed from this expression.

Network and optimization. An x-prediction MLP (width 256, depth 4, SiLU) with random-Fourier embeddings (dimension 64) for r, t and $\sigma _ { \mathrm { n } }$ , conditioned on y, with $\begin{array} { r l } { \boldsymbol { u } _ { r , t } ^ { \theta } } & { { } = } \end{array}$ $\big ( \mathrm { n e t } ( z , r , t , \pmb { y } , \sigma _ { \mathrm { n } } ) - z \big ) / ( 1 - r )$ . We use the objective equation 15 with $p = 1$ and $c = 1 0 ^ { - 3 } ;$ t is logit-normal $( \mu = 0 , \sigma = 1 )$ and $r \ = \ t$ with probability 0.75, else $r \sim \mathcal { U } ( 0 , t )$ . AdamW, learning rate $2 \times 1 0 ^ { - 4 } .$ , batch size 4096, 20,000 steps, EMA decay 0.999. One-step sampling is $\widehat { \pmb { x } } _ { 1 } = \mathrm { n e t } ( \pmb { x } _ { 0 } , r = 0 , t = 1 , \pmb { y } , \sigma _ { \mathrm { n } } ) ;$ ; few-step sampling composes ${ \bf \delta u } _ { r , t }$ over a uniform partition of [0, 1].

The source is exact. Drawing $2 \times 1 0 ^ { 5 }$ samples from the perturb-and-solve construction in equation 13 at $y = 2 . 5$ and comparing against the closed form reproduces $\mathcal { N } ( \boldsymbol { m } _ { \boldsymbol { y } } , \tau ^ { 2 } \boldsymbol { W } )$ to Monte-Carlo accuracy: $\| \Delta m _ { y } \| = 3 . 8 \times 1 0 ^ { - \bar { 3 } }$ and $\| \Delta \mathrm { c o v } \| _ { F } = 4 . 8 \times 1 0 ^ { - 3 }$ (Table 15). The anisotropy predicted by equation 10 is visible directly: the observed direction is pinned at variance 0.089 while the null direction carries the full working-prior variance $\tau ^ { 2 } = 9$

Posterior recovery. Figure 16 shows the source, the closed-form posterior, and $2 \times 1 0 ^ { 4 }$ one-step samples for three in-distribution measurements. The flow transports the vertically-spread source onto both posterior modes at ${ \pmb x } [ 1 ] = \pm 2 . 5$ while pinning $\begin{array} { r } { { \pmb x } [ 0 ] \approx { \ b y } , } \end{array}$ and the MMSE estimate falls in the empty valley between them. In the null direction the one-step sampler produces variance 6.66 against the closed-form 6.36; a mode-collapsed sampler would report $\zeta ^ { 2 } = 0 . 1 2$ . Table 13 reports mean, per-direction variance and $\mathbf { M M D ^ { 2 } }$ at three measurements.

Per-direction calibration. Projecting onto the right singular vectors of A (here $e _ { 0 }$ with s = 1 and e<sub>1</sub> with $s = 0 )$ , the flow assigns almost all of the null direction’s freedom (6.66 against posterior 6.36, source 9.0) and contracts the well-measured direction from the source value 0.089 toward the posterior 0.053, reaching 0.079 (Table 16).

Gaussian ground truth. With a single Gaussian prior matched to the working prior $( \Sigma = \tau ^ { 2 } I ,$ $\tau = \varsigma = 0 . 3 5 )$ , the source is the posterior and the optimal map is the identity. Retraining reproduces the source to $\| \Delta m _ { y } \| = 5 . 6 \times \mathrm { \hat { 1 0 ^ { - 4 } } }$ and matches the Gaussian posterior with mean errors ≤ 0.05 and $\mathrm { M M D ^ { 2 } \leq 3 \times i 0 ^ { - 3 } }$ (Table 14).

Table 13: Posterior match at three in-distribution measurements.
<table><tr><td>y</td><td>mean (post)</td><td>mean (SNaP)</td><td>∥|∆mean||</td><td>Var (observed, null) post → SNaP</td><td> $\mathrm { M M D ^ { 2 } }$ </td></tr><tr><td>2.5</td><td>(2.5,0)</td><td> $\left( 2 . 4 7 4 , 0 . 0 6 5 \right)$ </td><td>0.069</td><td> $( 0 . 0 5 2 , 6 . 3 7 )  ( 0 . 0 7 9 , 6 . 6 6 )$ </td><td> $4 . 0 \times 1 0 ^ { - 3 }$ </td></tr><tr><td>2.0</td><td>(2.212,0)</td><td> $\left( 2 . 0 4 5 , 0 . 0 5 1 \right)$ </td><td>0.174</td><td> $( 0 . 0 5 2 , 6 . 3 7 )  ( 0 . 0 8 4 , 6 . 6 4 )$ </td><td> $1 . 1 \times 1 0 ^ { - 2 }$ </td></tr><tr><td>-2.5</td><td>(−2.5,0)</td><td> $\left( - 2 . 4 4 4 , 0 . 1 0 8 \right)$ </td><td>0.121</td><td> $( 0 . 0 5 2 , 6 . 3 7 )  ( 0 . 0 8 1 , 6 . 5 9 )$ </td><td> $2 . 6 \times 1 0 ^ { - 3 }$ </td></tr></table>

Table 14: Single-Gaussian ground-truth check. The map reduces to the Wiener estimate, with the posterior mean recovered to within 0.05 and the posterior variances over-estimated by 37–39%.
<table><tr><td>y</td><td>mean (post)</td><td>mean (SNaP)</td><td> $\| \Delta \mathrm { m e a n } \|$ </td><td>Var (observed, null) post → SNaP</td><td> $\mathrm { M M D ^ { 2 } }$ </td></tr><tr><td>0</td><td>(0,0)</td><td>(-0.040, 0.026)</td><td>0.048</td><td> $( 0 . 0 5 2 , 0 . 1 2 3 )  ( 0 . 0 7 1 , 0 . 1 7 1 )$ </td><td> $2 . 7 \times 1 0 ^ { - 3 }$ </td></tr><tr><td>1</td><td>(0.576, 0)</td><td>(0.553, 0.012)</td><td>0.026</td><td> $( 0 . 0 5 2 , 0 . 1 2 3 )  ( 0 . 0 6 9 , 0 . 1 6 0 )$ </td><td> $1 . 7 \times 1 0 ^ { - 3 }$ </td></tr></table>

## C Implementation Details

## C.1 Datasets and evaluation protocol

CelebA. Official aligned images with the canonical partition of 162,770/19,867/19,962 for the train/validation/test images, center-cropped to 178×178 and resized to 128×128. We report results on the first 100 images of the test set.

AFHQ-Cat. 5,153/500/100 train/validation/test images, resized to 256×256. AFHQ-Cat has no official validation split. We followed [20] for train/val/test split.

fastMRI brain. The AXT2 subset, complex multi-coil k-space at 320×320. We split at the volume (patient) level into $6 1 5 / 7 7 / 7 7$ train/val/test volumes, giving 4,305/539/539 slices after discarding the first four and last five slices of every volume, which contain little or no anatomy. Coil handling is full multi-coil SENSE, not RSS or an emulated single-coil reduction. The ground truth is the ESPIRiT coil-combined complex image $\begin{array} { r } { \pmb { x } _ { 1 } = \sum _ { i } \bar { \pmb { S } } _ { i } \mathrm { c r o p } ( \pmb { F } ^ { H } \pmb { k } _ { i } ) } \end{array}$ on a 320×320 grid, retained as two real channels (real/imaginary) and normalized per slice by max $| { \pmb x } _ { 1 } | ;$ phase is modeled throughout and magnitude is taken only when computing PSNR/SSIM. The forward operator is $A = M \bar { F } S$ with the same per-slice ESPIRiT maps at train and test time, zero-padded to 20 coils.

Table 15: The square-root-free sampler equation 13 reproduces $\mathcal { N } ( \boldsymbol { m } _ { \boldsymbol { y } } , \tau ^ { 2 } \boldsymbol { W } )$ to Monte-Carlo accuracy at $y = 2 . { \bar { 5 } }$
<table><tr><td></td><td> $\mathbf { \nabla } _ { m _ { y } }$ </td><td> $\tau ^ { 2 } W \mathrm { ( d i a g ) }$ </td><td>error</td></tr><tr><td>closed form</td><td> $[ 2 . 4 7 5 2 , 0 ]$ </td><td>[0.0891, 9.000]</td><td></td></tr><tr><td>RTO equation 13</td><td>[2.4752, 0.0038]</td><td>[0.0889, 8.995]</td><td> $\left\| \Delta \boldsymbol { m } _ { y } \right\| = 3 . 8 \times 1 0 ^ { - 3 } , \left\| \Delta \mathrm { c o v } \right\| _ { F } = 4 . 8 \times 1 0 ^ { - 3 }$ </td></tr></table>

Table 16: Per-direction variance at $y = 2 . 5$ (equation 10). The source over-disperses both directions relative to the posterior; the trained flow moves both toward $\mathbf { i t } ,$ almost exactly in the null direction and partially in the observed one.
<table><tr><td>direction</td><td>Var posterior</td><td>Var one-step SNaP</td><td>Var source</td></tr><tr><td> $e _ { 0 } ( \mathrm { o b s e r v e d } , s = 1 )$ </td><td>0.0518</td><td>0.0791</td><td>0.0891</td></tr><tr><td> $e _ { 1 } ( \mathrm { n u l l } , s = 0 )$ </td><td>6.372</td><td>6.656</td><td>9.000</td></tr></table>

Because the coils couple, the measurement-space Gram $A A ^ { H }$ is not diagonal and the posteriorsource solve admits no closed form, unlike the inpainting, super-resolution, and deblurring operators, where $A ^ { H } A$ is diagonal in the pixel or Fourier basis and the solve costs $O ( n )$ or a single FFT pair. We therefore solve it with conjugate gradient in the algebraically equivalent image-space form $\mathbf { \check { A } } ^ { H } ( \mathbf { A } \mathbf { A } ^ { H } + \lambda \mathbf { I } ) ^ { - 1 } = ( \mathbf { A } ^ { H } \mathbf { A } + \lambda \mathbf { \check { I } } ) ^ { - 1 } \mathbf { \check { A } } ^ { H }$ , whose spectrum lies in $[ \lambda , \widetilde { \mathbf { \Lambda } } \approx 1 ]$ and whose right-hand side already lies in range $( A ^ { H } )$ ; the measurement-space form is rank-deficient by the coil count and, at the same iteration budget, returns an anchor worse than the plain adjoint.

## C.2 Solutions of the source construction

The source equation 13 requires one application of $K = A ^ { \top } ( A A ^ { \top } + \lambda I ) ^ { - 1 }$ per measurement, with $\lambda = \sigma _ { \mathrm { n } } ^ { 2 } \bar { / } \tau ^ { 2 }$ . Writing $\mathbf { \boldsymbol { A } } = \pmb { U } \pmb { \Sigma V } ^ { \top }$ , the covariance is $\tau ^ { 2 } \mathbf { { W } }$ with $\pmb { W } = \pmb { V } \mathrm { d i a g } ( \pmb { w } _ { i } ) \pmb { V } ^ { \top }$ and $w _ { i } = \lambda / ( s _ { i } ^ { 2 } + \lambda )$ . We summarise the cases used in this work.

Image deblurring. For deblurring $\mathbf { A } = \mathbf { H }$ is a spatially invariant blur. Under circular boundary conditions $\mathbf { H } = { { F } ^ { H } } \Delta { { F } }$ with Λ the Fourier coefficients $\hat { h }$ of the kernel, so $\mathbf { H } \mathbf { H } ^ { H } = { \pmb { F } } ^ { H } \left| \mathbf { \Lambda } \right| ^ { 2 } { \pmb { F } }$ and

$$
m _ { y } = F ^ { H } \left[ \frac { \overline { { { \Lambda } } } F y } { \left| \Lambda \right| ^ { 2 } + \lambda } \right] , \qquad w _ { i } = \frac { \lambda } { \left| \hat { h } _ { i } \right| ^ { 2 } + \lambda } ,\tag{26}
$$

elementwise in the Fourier domain, at the cost of two FFTs. Because $\left| \hat { h } _ { i } \right| > 0$ for every $i ,$ no $w _ { i }$ reaches 1 and no direction is unmeasured; the spectrum instead decays smoothly.

Image inpainting. For inpainting $A = M$ is a row-selection matrix that keeps the observed coordinates, so $M M ^ { \top } = I _ { m }$ and the solve is elementwise:

$$
( \pmb { m } _ { \pmb { y } } ) _ { i } = \left\{ \frac { y _ { i } } { 1 + \lambda } , \begin{array} { l l } { i \in \Omega , } \\ { 0 , } \end{array} \right. \qquad \pmb { w } _ { i } = \left\{ \frac { \lambda } { 1 + \lambda } , \begin{array} { l l } { i \in \Omega , } \\ { 1 , } \end{array} \right.\tag{27}
$$

where Ω is the set of observed pixels. Unobserved coordinates carry the full working-prior variance $\tau ^ { 2 } .$ , while observed coordinates are shrunk toward the prior mean by the factor $1 / ( 1 + \lambda )$ and retain variance $\tau ^ { 2 } \lambda / ( 1 + \lambda )$ . Box and random inpainting share this structure and differ only in the geometry of Ω.

Image super-resolution. We use box-decimated super-resolution by implementing strided subsampling with no anti-alias prefilter, $\pmb { A x } = \pmb { x } [ . ~ . ~ , \pmb { : } ~ s , \pmb { : } ~ s ] ,$ , so $\pmb { A } \pmb { A } ^ { \top } = \mathbf { \dot { I } } , \pmb { A } ^ { \top }$ is zero-filled upsampling, and equation 27 applies with Ω the sampled grid locations.

Multi-coil MRI. With $A = M F S$ the sensitivity maps S are not unitary, so $A A ^ { \top }$ is not diagonal and the solve with $A A ^ { \top } + \lambda I$ has no closed form. We solve it with conjugate gradients, warm-started at the zero-filled adjoint. Measured iteration counts and cost are reported in Table 17.

General linear operators. For a general A with no exploitable structure, equation 13 is solved by conjugate gradients. All operators used in this paper fall under the cases above. The solve is required once per draw and does not depend on $t , r ,$ or the network. For a fixed operator, the solve can be reduced to a diagonal inverse for diagonal (and Fourier diagonal) forward models or use a short conjugate gradient run for most others.

![](images/3d6bee2f89c3ae6a769f9c3d595cab097be4205dcd1fb8929710bca343b4e61f.jpg)  
Figure 16: Posterior recovery for observations $y = 2 . 5$ (left) and $y = - 2 . 5$ (right). The top row compares samples from the source distribution (green), the closed-form posterior (blue), and the one-step model (red). Dashed vertical lines indicate the observed values. The bottom row shows the corresponding marginals along the unobserved coordinate $x _ { 1 } ;$ the dashed line marks the posterior mean. The one-step model captures both posterior modes, although it exhibits some within-mode distortion relative to the exact posterior.

Table 17: Cost of the source construction equation 13. Larger λ (noisier data) improves conditioning and accelerates CG.
<table><tr><td>Problem</td><td> $A A ^ { \top }$  structure</td><td>Cost</td><td>Notes</td></tr><tr><td>Inpainting (random)</td><td> ${ \pmb I } _ { m }$ </td><td>free</td><td>W diagonal</td></tr><tr><td>Inpainting (box)</td><td> ${ \pmb I } _ { m }$ </td><td>free</td><td>identical structure</td></tr><tr><td>MRI, single-coil</td><td> ${ \pmb I } _ { m }$ </td><td>free</td><td>diagonal in k-space</td></tr><tr><td>SR, average-pool</td><td> $d ^ { - 2 } \pmb { I }$ </td><td>free</td><td>disjoint rows of norm  $1 / d$ </td></tr><tr><td>Deblurring, Gaussian</td><td>diag. in Fourier</td><td>2 FFTs</td><td> $\begin{array} { r } { \pmb { A } \pmb { A } ^ { \top } = \left| \hat { h } \right| ^ { 2 } } \\ { . } \end{array}$ </td></tr><tr><td>Deblurring, uniform</td><td>diag. in Fourier</td><td>2 FFTs</td><td>Î has exact zeros</td></tr><tr><td>SR, bicubic</td><td>block-Fourier</td><td>O(n log n)</td><td>polyphase folding identity</td></tr><tr><td>MRI, multi-coil</td><td>not diagonal</td><td>CG, 10–20 it. typical</td><td>standard SENSE</td></tr></table>

## C.3 Training and implementation details

Architecture. The backbone is a DDPM-style residual U-Net (GroupNorm , Swish, variancescaled initialization, zero-initialized output and residual-branch convolutions) with three total scalar conditioning inputs. The time t is embedded by a sinusoidal positional encoding followed by a two-layer MLP into a 4 ch-dimensional vector; the second MeanFlow time r and the noise level $\sigma _ { \mathrm { n } }$ each get their own identical embedding, and the three are summed before entering the body, where each residual block adds a per-channel affine projection of the sum to its first activation. The $\sigma _ { \mathrm { n } }$ embedding takes 10 $\sigma _ { \mathrm { n } }$ as input so that the small noise levels used here occupy a useful part of the sinusoidal range. Conditioning on y is by concatenation rather than FiLM, with input $[ z _ { t } ,$ anchor] of 2C channels, where the anchor is the adjoint $\pmb { A } ^ { \top } \pmb { y }$

<table><tr><td>Algorithm 1 SNaP training step.</td><td></td></tr><tr><td>1:  $\pmb { x } _ { 1 } \sim p , \quad \pmb { y }  \pmb { A } \pmb { x } _ { 1 } + \pmb { n }$ </td><td> training pair, equation 1</td></tr><tr><td>2:  $t , r \gets \mathbf { S A M P L E T I M E S } ( )$ </td><td>▶ t logit-normal;  $r = t$  with probability  $p _ { \mathrm { r a t i o } }$ </td></tr><tr><td>3:  $\pmb { x } _ { 0 } \gets \mathrm { S A M P L E S O U R C E } ( \pmb { y } , \sigma _ { \mathrm { n } } )$ </td><td> perturb-and-solve, equation 13</td></tr><tr><td>4:  $z _ { r } \gets ( 1 - r ) \pmb { x } _ { 0 } + r \pmb { x } _ { 1 } ; \quad \pmb { v } \gets \pmb { x } _ { 1 } - \pmb { x } _ { 0 }$ </td><td></td></tr><tr><td>5:  $\pmb { u } , \ \mathrm { d } _ { r } \pmb { u } \gets \mathrm { J V P } \big ( \pmb { u } ^ { \theta } ; ( z _ { r } , r , t ) , ( \pmb { v } , 1 , 0 ) \big )$ </td><td> one forward-mode pass</td></tr><tr><td>6:  ${ \pmb u } _ { \mathrm { t g t } }  { \pmb v } + ( t - r ) \mathrm { s g } [ \mathrm { d } _ { r } { \pmb u } ]$ </td><td> MeanFlow target, equation 14</td></tr><tr><td>7:  $\ell \gets \| \boldsymbol { u } - \boldsymbol { u } _ { \mathrm { t g t } } \| _ { 2 } ^ { 2 } / n$ </td><td></td></tr><tr><td>8:  ${ \mathcal { L } } \gets \operatorname { s g } [ \omega ( \ell ) ] \bar { \ell }$ </td><td> weighted objective, equation 15</td></tr><tr><td>9:  $\pmb \theta \gets \mathrm { A D A M W } \big ( \pmb \theta , \nabla _ { \pmb \theta } \mathcal { L } \big )$ </td><td> optimizer step</td></tr></table>

Configurations: CelebA $1 2 8 \times 1 2 8$ uses $\mathrm { c h } = 6 4 ,$ ch\_mult $= ( 1 , 2 , 2 , 2 )$ , two residual blocks per level and attention at $1 6 ^ { 2 } ~ ( 9 . 1 \mathrm { { M } }$ parameters); AFHQ-Cat 256×256 uses ch = 64, ch\_mult = (1, 2, 4, 4, 8, 8) down to an 8×8 bottleneck, two residual blocks, attention at $3 2 ^ { 2 }$ and $\mathrm { 1 6 ^ { 2 } }$ and dropout 0.1 (110.5M); fastMRI brain 320×320 uses $\mathrm { c h } = 3 2 .$ , ch ${ \underline { { \mathrm { m u l t } } } } = ( 1 , 2 , 4 , 8 , 8 )$ , four residual blocks and attention at $4 0 ^ { 2 }$ (40.7M). Attention is a hand-written block of 1×1 convolutional Q, K, V projections, a batched matrix product scaled by $C ^ { - 1 / 2 }$ , and a float32 softmax, since torch.func.jvp has no forward-mode rule for the fused scaled-dot-product kernels. The same restriction rules out torch.utils.checkpoint, so activation memory is a hard ceiling and width is added only at the two deepest levels.

Training. Given a clean image $\scriptstyle { \mathbf { { \vec { x } } } } _ { 1 }$ we form $\pmb { y } = \pmb { A } \pmb { x } _ { 1 } + \pmb { n } .$ , draw the source $\mathbf { \boldsymbol { x } } _ { 0 } \sim \mathcal { N } ( m _ { y } , \tau ^ { 2 } W )$ by equation 13, and set $\begin{array} { r } { z _ { t } = ( 1 - t ) \pmb { x } _ { 0 } + t \pmb { x } _ { 1 } } \end{array}$ and ${ \pmb v } = { \pmb x } _ { 1 } - { \pmb x } _ { 0 }$ . Times are drawn as $t = \sigma ( \mu + s \epsilon )$ with $\epsilon \sim \mathcal { N } ( 0 , 1 ) , \mu = 0$ and $s = 1$ (logit-normal on $( 0 , 1 ) )$ ; with probability $p _ { \mathrm { r a t i o } } = 0 . 5$ we set $r = t ,$ , and otherwise $r \sim \mathcal { U } ( 0 , t )$ . A further 10% of the $r \neq t$ samples, or 5% of the batch, is redrawn at the corner $r \sim \mathcal { U } ( 0 , 0 . 1 ) , t \sim \mathcal { U } ( 0 . 9 , 1 )$ , because the base schedule reaches the slice at which one-step sampling is scored with probability only $7 . 3 \times 1 0 ^ { - 4 }$

The network is an $\scriptstyle { \mathbf { { \vec { x } } } } _ { 1 }$ -predictor, so ${ \pmb u } ( z _ { r } , r , 1 ) = \big ( f _ { \pmb \theta } ( z _ { r } , r , 1 ) - z _ { r } \big ) / ( 1 - r )$ . The total derivative is one forward-mode pass, with the differentiated arguments $( z , r , t )$ carrying tangent v on z, 1 on r and 0 on t, while the anchor, the coverage map and $\sigma _ { \mathrm { n } }$ are closed over and carry zero tangent. The regression target is $\pmb { u } _ { \mathrm { t g t } } = \pmb { v } + ( t - r ) \mathrm { s g } [ \mathrm { d } \pmb { u } / \mathrm { d } r ]$ and the loss is equation 15 with $\begin{array} { r } { \Delta = \pmb { u } ^ { \tilde { \pmb { \theta } } } - \pmb { u } _ { \mathrm { t g t } } , } \end{array}$ $p = 1$ and $c = 1 0 ^ { - 3 }$ . The stop-gradient is on the weight only, and a second stop-gradient is applied to the derivative term. On CelebA and AFHQ, ℓ is first divided by its all-reduced running mean (decay 0.99) so that c is dimensionless and the loss shape is stable across τ and across training; the MRI runs use the unnormalized form. No endpoint $\| \hat { \pmb x } _ { 1 } - { \pmb x } _ { 1 } \| _ { 2 } ^ { 2 }$ term is used, since its population minimizer is the posterior mean and would invalidate every calibration metric.

Optimization is AdamW with weight decay $1 0 ^ { - 4 }$ , cosine annealing to $1 0 ^ { - 6 }$ over the full epoch budget, no warm-up and gradient-norm clipping at 1.0. The learning rate is $2 \times 1 0 ^ { - 4 }$ on CelebA and AFHQ and $1 \times 1 0 ^ { - 4 }$ on MRI. Everything is trained in float32, since forward-mode AD is not autocast-safe. The effective batch is 32 throughout (CelebA 8 × 4 GPUs; AFHQ $2 \times 4$ accumulation steps ×4 GPUs; MRI 4 × 4 GPUs). Budgets are 200 epochs (50,000 steps) on CelebA, 120 epochs (15,000 steps) for AFHQ super-resolution and 80 epochs (10,000 steps) for the remaining AFHQ operators, and 80 epochs (43,040 steps) for brain MRI, where an epoch is a fixed-size draw with replacement rather than a pass over the data. EMA is implemented but disabled in all reported runs, having never beaten the raw weights.

Choice of τ. We set $\tau ^ { 2 }$ to the mean posterior variance left along the directions the measurement leaves open, estimated offline on 256 training images as $\tau ^ { 2 } = { \langle d , \tilde { W } d \rangle } / \operatorname { t r } W$ , where $\pmb { d } = \pmb { x } - \hat { \pmb { x } } ( \pmb { y } )$ is the residual of the linear-MMSE (Wiener) estimate under the empirical image power spectrum. Since W depends on τ, we solve this by fixed-point iteration. The linear estimate is suboptimal, so its residual upper-bounds the posterior variance and τ errs toward over-dispersion. This gives $\tau = 0 . 1$ for $\times 2$ super-resolution and random inpainting on CelebA, 0.15 for deblurring and 0.3 for box inpainting; $\tau = 0 . 1$ for $\times 4$ super-resolution, random inpainting and deblurring on AFHQ and

0.5 for box inpainting; and $\tau = 0 . 2 5$ for brain MRI. The ablation in App. B shows that reconstruction quality changes little across this range. Dataset splits and preprocessing are given in App. C.1.

Source solves. The draw equation 13 needs only $A , A ^ { \top }$ , and one solve with $( A A ^ { \top } + \lambda I ) ^ { - 1 }$ where $\lambda = \sigma _ { \mathrm { n } } ^ { 2 } / \tau ^ { 2 }$ is floored at $1 0 ^ { - 8 }$ , and for the operators used here that solve is a diagonal inverse, except for multi-coil MRI.

Circular Gaussian deblurring is Fourier-diagonal, with $A A ^ { \top } = | H ( \omega ) | ^ { 2 }$ and one division per mode. Box-decimation super-resolution has $\pmb { A } \pmb { A } ^ { \top } = \pmb { I }$ on the low-resolution grid, so the solve is the scalar division by $1 + \lambda$ . Inpainting has $A A ^ { \top } = M$ for a $0 / 1$ mask, giving a pixel-diagonal division by $M + \lambda$ . Super-resolution and inpainting are ${ \mathcal { O } } ( n )$ and Gaussian deblurring is $\mathcal { O } ( \bar { n } \log { n } )$ from the FFT pair; none add any measurable time to a training step.

Multi-coil MRI is the only iterative case, and we solve it in image space using $A ^ { \top } ( A A ^ { \top } + \lambda I ) ^ { - 1 } =$ $( A ^ { \top } A + \lambda I ) ^ { - 1 } A ^ { \top }$ , running conjugate gradient on $A ^ { \top } A + \lambda I$ , whose spectrum lies in $[ \lambda , \approx 1 ]$ and whose right-hand side already lies in range $( A ^ { \top } )$ ), with a cap of 200 iterations and relative-residual tolerance $1 0 ^ { - 6 }$ , which is typically reached in 10–20 iterations and per-sample reductions so that batched slices do not share a step size. The measurement-space route is implemented but not used, since $A A ^ { \top }$ acts on k-space while factoring through the image and is therefore rank-deficient by the coil count on top of the mask; a short CG there leaves the anchor over-scaled at 6.5 dB, worse than the plain adjoint at 24.4 dB, against 26.0 dB for the image-space form at equal cost. Requesting the anchor $m _ { y }$ inside the sampler splits the single solve into two by linearity, which is free for the diagonal operators and doubles the CG only for MRI.

Cost. All models were trained on four A6000 GPUs with manual data-parallel gradient all-reduce. Measured end-to-end wall-clock: AFHQ-Cat super-resolution 6 h 07 m for 120 epochs; AFHQ random inpainting, box inpainting and deblurring 4 h 04 m, 4 h 00 m and 4 h 02 m for 80 epochs each; CelebA box inpainting 3 h 29 m for 200 epochs; and brain MRI at $\times 8$ acceleration 19 h 05 m for 80 epochs. Throughput on the 256-pixel AFHQ configuration is 6.6 images/s/GPU at 26.1 GB peak memory with batch size 2; the 110.5M model costs 18% more memory and 21% less throughput than a 28.0M baseline, because memory is activation- rather than parameter-dominated and the added width sits at the $1 6 ^ { 2 }$ and $8 ^ { 2 }$ levels. At inference an M-sample posterior costs M network evaluations and M source draws. Each draw is $O ( n )$ for super-resolution and inpainting, O(n log n) for Gaussian deblurring, and a CG solve for multi-coil MRI.

Baseline budgets. The NFE column of each table counts network evaluations per reported output. SNaP and the MSE regressor use one evaluation per draw, so an M-draw row costs M evaluations. On natural images PnP-GS uses 23, DDRM 20, DiffPIR 100, OT-ODE 180 and DPS 1000; on fastMRI, DiffPIR uses 100 and CSGM, DPS, DAPS and PnP-DM use 1000. The subscript on PnP-Flow1, PnP-Flow5, Flower1-OT and Flower5-OT is $N _ { \mathrm { A v g } } ,$ , the number of network evaluations averaged at each time step, so a run of N steps costs $N \cdot \bar { N } _ { \mathrm { A v g } }$ evaluations; we use the values tabulated by [21]. On CelebA both methods use $N = 1 0 0$ for every operator, so Tab. 1 reports the five-evaluation variants at 500 evaluations, and the single-evaluation variant Flower1-OT of Tab. 2 at 100. On AFHQ-Cat the step counts depend on the operator: PnP-Flow uses $N = 5 0 0$ for deblurring and super-resolution, 200 for random inpainting and 100 for box inpainting, while Flower uses N = 100, 500, 200 and 100 on those same four operators. The budgets behind the NFE range quoted in Tab. 6 are therefore 2500, 2500, 1000 and 500 evaluations for PnP-Flow5, and 500, 2500, 1000 and 500 for Flower5-OT. Every SNaP entry in Tab. 2 is a single draw $( M { = } 1 )$ ; averaged draws are reported only in the distortion tables and in the ablations of App. B.

## C.4 Metric details

For natural images, PSNR is computed on [0, 1] with data range 1, SSIM on the clamped [0, 1] image, and LPIPS with an AlexNet backbone on the clamped [−1, 1] image. Images are normalized to [−1, 1] for training and mapped back to [0, 1] for scoring. For MRI, PSNR and SSIM are computed on magnitude images, per slice, with the data range set to the maximum intensity of the groundtruth slice. We do not report LPIPS on MRI: its backbone is trained on RGB natural images and its perceptual judgments do not transfer to grayscale magnitude reconstructions.

Calibration ratio. We check whether the spread of a sampler’s draws has the right size for a sampler of $p ( \pmb { x } \mid \pmb { y } )$ . Let $p _ { \mathrm { s } } ( { \pmb x } \mid { \pmb y } )$ be the distribution of the draws and $\pmb { \mu } _ { \mathrm { s } } ( \pmb { y } )$ its mean, and let $( \pmb { x } _ { 1 } , \pmb { y } )$ be a ground-truth image and its measurement. Define

$$
B : = \mathbb { E } _ { \boldsymbol { x } _ { 1 } , \boldsymbol { y } } \big [ \| \pmb { \mu _ { \mathrm { s } } } ( \pmb { y } ) - \pmb { x } _ { 1 } \| _ { 2 } ^ { 2 } \big ] , \qquad V : = \mathbb { E } _ { \boldsymbol { y } } \big [ \operatorname { t r } \operatorname { C o v } _ { \mathrm { s } } ( \pmb { x } \mid \pmb { y } ) \big ] .
$$

If $p _ { \mathrm { s } } = p .$ , then $\pmb { \mu } _ { \mathrm { s } } ( \pmb { y } ) = \mathbb { E } [ \pmb { x } \mid \pmb { y } ]$ and

$$
B = \mathbb { E } _ { \pmb { y } } \Big [ \mathbb { E } _ { { \pmb x } _ { 1 } \mid { \pmb y } } \Big [ \left\| \mathbb { E } [ { \pmb x } \mid { \pmb y } ] - { \pmb x } _ { 1 } \right\| _ { 2 } ^ { 2 } ] \Big ] = \mathbb { E } _ { \pmb { y } } \big [ \mathrm { t r } \mathrm { C o v } ( { \pmb x } \mid { \pmb y } ) \big ] = V .
$$

Hence $V / B = 1$ is a necessary condition for $\begin{array} { r } { p _ { \mathrm { s } } = p , } \end{array}$ and we report the calibration ratio

$$
\Delta _ { \mathrm { c a l } } : = \frac { V } { B } .\tag{28}
$$

Values below 1 indicate draws with too little spread, collapsed toward a point estimate, and values above 1 indicate over-dispersed draws.

Estimation. For each test pair $( \pmb { x } _ { 1 } ^ { ( i ) } , \pmb { y } ^ { ( i ) } ) , i = 1 , \ldots , N$ , we draw $\pmb { s } _ { ( 1 ) } ( \pmb { y } ^ { ( i ) } ) , \ldots , \pmb { s } _ { ( M ) } ( \pmb { y } ^ { ( i ) } )$ with mean $\bar { \pmb { s } } ( \pmb { y } ^ { ( i ) } )$ and use

$$
\hat { V } : =  \frac { 1 } { N ( M - 1 ) } \sum _ { i = 1 } ^ { N } \sum _ { j = 1 } ^ { M } \| \pmb { s } _ { ( j ) } ( \pmb { y } ^ { ( i ) } ) - \bar { \pmb { s } } ( \pmb { y } ^ { ( i ) } ) \| _ { 2 } ^ { 2 } ,
$$

$$
\hat { B } : =  \frac { 1 } { N } \sum _ { i = 1 } ^ { N } \| \bar { \pmb { s } } ( \pmb { y } ^ { ( i ) } ) - \pmb { x } _ { 1 } ^ { ( i ) } \| _ { 2 } ^ { 2 } - \frac { \hat { V } } { M } ,
$$

which are unbiased for V and $B ;$ the correction $\hat { V } / M$ removes the variance that remains in the mean of M draws. We use $M = 1 0 0 . \Delta _ { \mathrm { c a l } }$ is related to the spread-skill ratio used in ensemble forecasting, which compares the standard deviation of the ensemble with the root-mean-squared error of its mean and therefore corresponds to $\sqrt { V / B }$

FID. FID is computed with InceptionV3 pool3 features on 2048 single-draw reconstructions, quantised to uint8, against reference statistics from 2048 ground-truth images.
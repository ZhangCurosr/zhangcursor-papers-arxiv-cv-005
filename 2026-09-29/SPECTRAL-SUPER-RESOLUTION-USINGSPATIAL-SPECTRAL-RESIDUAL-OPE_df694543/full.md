# SPECTRAL SUPER-RESOLUTION USINGSPATIAL-SPECTRAL RESIDUAL OPERATORNETWORKS

Seokhyun Chin

California Institute of Technology

1200 E California Boulevard, Pasadena, CA, USA

schin@caltech.edu

Abstract—Spectral super-resolution of multispectral satellite images can enable high temporal- and spatial-resolution hyperspectral satellite imagery at a modest cost, significantly increasing the applicability of hyperspectral remote sensing. This task is inherently ill-posed, making it well-suited for deep learningbased methods. In this study, the spectral super-resolution task is framed as an operator learning problem, and SSRON is proposed as a Deep Operator Network that effectively learns function-to-function mappings from downsampled spectra to continuous spectra. The model is trained to super-resolve Sentinel-2A-like multispectral imagery to EMIT images. Compared to baseline models, SSRON achieves superior performance across all metrics. The model also demonstrates zero-shot spectral superresolution capability by predicting bands unseen during training. Furthermore, its continuous-output formulation suggests the potential to estimate spectra at finer wavelength intervals than the native sensor. These results suggest the potential of SSRON and establishes operator learning as a promising direction for spectral super-resolution.

Index Terms—Spectral Super Resolution, Deep Operator Networks, Neural Operator, Hyperspectral Satellite Imagery

## I. INTRODUCTION

Hyperspectral satellite images (HSIs) have become important tools in earth observation. By capturing hundreds of spectral bands, they provide detailed spectral information that is unavailable in conventional satellite imagery. As a result, HSIs have been widely used in applications such as weather forecasting [1], precision agriculture [2], and water quality monitoring [3]. However, their broader use is limited by low spatio-temporal resolution and high data acquisition costs [4].

To address these limitations, three main approaches have been developed. Spatial Super Resolution seeks to directly enhance the spatial resolution of an HSI; HSI sharpening fuses HSIs with high-resolution multispectral satellite images (MSIs) to improve spatial detail; Spectral Super Resolution (SSR) seeks to enhance the spectral resolution of an MSI to obtain an HSI. HSI sharpening suffers from the lack of high-spatial-resolution HSI reference data, and Spatial Super Resolution cannot overcome the low temporal resolution of HSI or the high cost of HSI acquisition [5]. Consequently, SSR provides a more practical and flexible alternative.

However, SSR is an ill-posed inverse problem in which hundreds of HSI bands are recovered from only a limited number of MSI bands. Traditional SSR is based on shallow encoding methods such as dictionary-based methods [6–8]. With the emergence of deep learning methods, research has increasingly shifted toward their use. Specifically, variations of convolutional neural networks, generative adversarial networks, and transformer-based architectures have shown strong performance [9–14].

Neural operators have emerged as a deep learning framework that learns mappings between infinite-dimensional function spaces [15, 16]. This enables continuous function evaluation, which makes it particularly well-suited for superresolution. Indeed, various neural operator architectures have shown promise in spatial [17, 18] and temporal [19] superresolution tasks, but their application to SSR remains limited. While prior work, such as Zhang et al. [20], has explored the integration of neural operator architectures for SSR, the direct application of neural operators for SSR remains largely unexplored, with no prior work explicitly formulating the problem as an operator learning problem.

In this paper, SSR is cast as an operator learning problem and addressed using a neural operator framework. The key contributions are as follows.

• The SSR problem is formulated as an operator learning problem, laying the mathematical groundwork for applying neural operators to SSR.

• Spatial-Spectral Residual deep Operator Network (SS-RON) is proposed, which applies the Deep Operator Network (DeepONet) architecture to SSR while using a spatial–spectral residual CNN as the branch network– adapting architectural principles proven effective in SSR to the operator learning framework. While CNNs have been used in the branch networks of DeepONets [21, 22], this paper, to the best of the author’s knowledge, is the first to adapt a spatial–spectral residual CNN architecture to the operator-learning framework for spectral superresolution.

• The strength of SSRON in SSR is demonstrated, achieving improved performance compared to state-of-the-art methods. The application of this architecture to zero-shot SSR for unseen bands during training and its potential to leverage infinite-dimensional mappings is also demonstrated.

## II. PROBLEM FORMULATION

The sensor calibrated radiance value $L ( \lambda )$ can be written as [12]

$$
L ( \lambda , \vec { x } ) = I ( \lambda , \vec { x } ) R ( \lambda , \vec { x } ) ,
$$

where I denotes solar radiation and R is the reflectance of the region that the sensor takes its value from. Both the MSI and HSI are downsampled from this continuous function, with

$$
\begin{array} { l } { { L _ { m } ( i , \vec { x } ) = \displaystyle \int L ( \lambda , \vec { x } ) g _ { m } ( i , \lambda ) d \lambda ~ } } \\ { { L _ { h } ( i , \vec { x } ) = \displaystyle \int L ( \lambda , \vec { x } ) g _ { h } ( i , \lambda ) d \lambda , } } \end{array}
$$

where $g _ { m } ( i , \lambda ) , g _ { h } ( i , \lambda )$ are the Spectral Response Function (SRF) of ith band of the multispectral and hyperspectral instruments, respectively. Because the hyperspectral instruments SRF are centered about a central wavelength λ<sub>i</sub>, we can approximate,

$$
L _ { h } ( \lambda _ { i } , \vec { x } ) \approx L ( \lambda _ { i } , \vec { x } ) \int g _ { h } ( i , \lambda ) d \lambda = L ( \lambda _ { i } , \vec { x } ) .
$$

Now consider the functional

$$
L ^ { \prime } ( \vec { g } , \vec { x } ) : = \int L ( \lambda , \vec { x } ) \vec { g } ( \lambda ) d \lambda = \vec { L _ { g } } ( \vec { x } ) .
$$

Effectively, the MSI can be written as $L ^ { \prime } ( \vec { g _ { m } } , \vec { x } )$ . The SSR problem, therefore, can be framed as an operator learning problem that learns $G : L ^ { \prime } ( \vec { g } , \vec { x } ) \mapsto L ( \lambda , \vec { x } )$ . In this study, we solve a specific case of the problem where g is fixed to some $g _ { m }$ though extension of this problem to multiple $g \mathrm { s }$ are possible.

## III. METHODOLOGIES

## A. Architecture

The proposed architecture is based on the DeepONet [15], which creates a branch network that encodes the input function at discrete sensor points and a trunk network that encodes locations where the output is evaluated. The overall architecture is outlined in Figure 1.

The standard multi-layer perceptron branch network is replaced with a spatial–spectral residual CNN, motivated by the strong performance of such architectures in remote sensing SSR [11, 23, 24].

The embedding layer in Figure 1a maps each coordinate value to a vector according to the following:

$$
\mathrm { E m b e d d i n g : } c \mapsto \left[ \begin{array} { c } { c } \\ { \sin ( f _ { 0 } \cdot c ) } \\ { \cos ( f _ { 0 } \cdot c ) } \\ { \vdots } \\ { \sin ( f _ { j } \cdot c ) } \\ { \cos ( f _ { j } \cdot c ) } \end{array} \right]
$$

where $f _ { i } = 2 ^ { i }$ . The coordinates $( x , y , \lambda )$ are each embedded independently, and the resulting vectors are concatenated. The trunk network is a simple two-layer fully-connected neural network with 256 hidden dimensions.

## B. Dataset

The EMIT Imaging Spectrometer’s L1B at sensor calibrated radiance [25] was used as the HSI source. EMIT is an advanced imaging spectrometer instrument on the International Space Station that outputs 285 discrete bands from the visible to the short-infrared. Bands 74-79, 100-107, 131-143, and 191-217 were dropped due to absorption via water vapor and ozone, leaving a total of 229 bands. 10 scenes were randomly selected from December 2024 with zero cloud coverage, downloaded from NASA’s Earthdata Search. Details of the selected scenes are presented in Table I.

Spatio-temporally aligned MSIs are generated by downsampling the HSI using the spectral response function of the Sentinel 2A satellite [26]. The initial SRF was downsampled to the 229 bands and normalized, following the approach shown in [9]. Band 9 was dropped because it samples corrupted bands in the HSI. The downsampled SRF is shown in Figure 2.

TABLE I: Dataset specifications
<table><tr><td>Scene no.</td><td>Granule ID (excluding common prefix)</td></tr><tr><td>1</td><td>20241214T150837_2434910_009</td></tr><tr><td>2</td><td>20241214T151321_2434910_033</td></tr><tr><td>3</td><td>20241215T094506_2435006_010</td></tr><tr><td>4</td><td>20241215T203042_2435013_020</td></tr><tr><td>5</td><td>20241216T071956_2435105_007</td></tr><tr><td>6</td><td>20241216T193914_2435113_006</td></tr><tr><td>7</td><td>20241217T093419_2435206_006</td></tr><tr><td>8</td><td>20241223T050524_2435803_051</td></tr><tr><td>9</td><td>20241230T150153_2436510_038</td></tr><tr><td>10</td><td>20241231T141323_2436609_033</td></tr></table>

![](images/8499b2fdebaf0eec64c420e45e22304b2e869570e98cb001d78f082fcd4c7db9.jpg)  
Fig. 2: Normalized and downsampled SRF of the Sentinel 2A satellite

## C. Training

Non-overlapping 16x16 patches were selected from each scene, resulting in a total of 61,600 patches. An 8:1:1 trainvalidation-test split was used. The models were trained with an Adam optimizer at a learning rate of 5e-5, with cosine annealing and early stopping with patience of 5 epochs for 300 epochs, using Mean Absolute Error (MAE) as the loss function. All training was conducted on a single RTX 5060 Ti GPU.

![](images/3ca6502e10feeb937ff4fe4644caa692d09bf160d74c40dda398b7092660f790.jpg)  
Fig. 1: Proposed architecture of SSRON. The framework consists of a branch network and trunk network, as shown in (a). The branch network encodes the MSI into a latent representation, and the trunk network encodes the spatial-spectral query positions. (b) shows the specific architecture of the branch network.

1) Training SSRON: Unlike conventional image-to-image models, the SSRON is trained to predict the value of the learned function at a specified spatial and spectral coordinates. To enable this, 2048 possible positions (spatial and spectral) were selected in the HSI label to be predicted by the SSRON for each epoch.

2) Zero-shot super resolution: As neural operators learn maps between infinite-dimensional function spaces, they can be evaluated at previously unseen coordinates. To test this capacity, a fixed percentage of bands was intentionally excluded from training, and the model’s performance was evaluated across all bands. The excluded bands were selected to be relatively equidistant and to span the same wavelength range as the data.

## D. Baselines

AWAN, Restormer, and SSRAN architectures were selected as state-of-the-art baselines because they have demonstrated strong performance for SSR tasks [11, 27, 28]. The Ushaped Neural Operator (UNO) [29] and the Fourier Neural Operator (FNO) [16] were included as comparable baselines for operator learning. The RSNO proposed by Zhang et al. [20] was not included as a baseline because it relies on additional physics-based information as input to the model, which was neither available nor assumed in this experimental setting.

## E. Metrics

In addition to MAE, Root Mean Squared Error (RMSE), Peak Signal to Noise Ratio (PSNR), and Structural Similarity Score (SSIM) were used as evaluation metrics. SSIM is calculated like so [30]:

$$
\mathrm { S S I M } ( x , y ) = \frac { ( 2 \mu _ { x } \mu _ { y } + \epsilon _ { x } ) ( 2 \sigma _ { x y } + \epsilon _ { y } ) } { ( \mu _ { x } ^ { 2 } + \mu _ { y } ^ { 2 } + \epsilon _ { x } ) ( \sigma _ { x } ^ { 2 } + \sigma _ { y } ^ { 2 } + \epsilon _ { y } ) } ,
$$

where $\mu _ { x }$ and $\mu _ { y }$ are the mean values of images x and $y , \sigma _ { x } ^ { 2 }$ and $\sigma _ { y } ^ { 2 }$ are the variances, $\epsilon _ { x } = ( 0 . 0 1 \cdot L ) ^ { 2 } , \epsilon _ { y } = ( 0 . 0 3 \cdot L ) ^ { 2 }$ are small constants to stabilize division with L being the dynamic range of data, and $\sigma _ { x y }$ is the covariance of x and y.

## IV. RESULTS

Table II reports the results evaluated over the testing dataset. SSRON outperforms all baseline models across all tested metrics. UNO also performs strongly, outperforming all nonoperator baseline models. In contrast, the FNO shows substantially higher errors, which may be attributed to the sharpness in spectra that needs to be reconstructed; standard FNOs struggle with such high-dimensional features [31].

Figure 3 shows the per-band errors of each model. SSRON achieves a lower error across all bands, with the exception of band 115, where the UNO shows a lower error. A sharp increase in error is observed in the bands below 5 and above 200. This behavior is likely due to the limited spectral information of the Sentinel-2A satellite in those regions, as shown in Figure 2.

The relationship between SRF coverage and the errors is further supported by Pearson’s correlation test. The correlation between the MAE (↓), RMSE(↓), and PSNR(↑) with the SRF value at each wavelength is -0.222, -0.175, and 0.408 $( \mathfrak { p } \ <$ 0.01), respectively, demonstrating a clear inverse relationship between error and SRF values.

Table III reports the zero-shot spectral resolution performance of the SSRON. As the percentage of bands used decreases, MAE increases and SSIM decreases, indicating that zero-shot inference is difficult. RMSE and PSNR do not strictly follow the monotonic trend, with the 75% model

TABLE II: Error metrics over testing dataset
<table><tr><td>Model type</td><td>MAE↓</td><td>RMSE↓</td><td>PSNR ↑</td><td>SSIM ↑</td></tr><tr><td>AWAN</td><td>0.01529</td><td>0.03476</td><td>50.92</td><td>0.9861</td></tr><tr><td>FNO</td><td>0.1690</td><td>0.2745</td><td>31.23</td><td>0.2516</td></tr><tr><td>Restormer</td><td>0.01606</td><td>0.03670</td><td>50.24</td><td>0.9845</td></tr><tr><td>SSRAN</td><td>0.01871</td><td>0.03999</td><td>49.29</td><td>0.9820</td></tr><tr><td>UNO</td><td>0.01498</td><td>0.03255</td><td>51.01</td><td>0.9865</td></tr><tr><td>SSRON</td><td>0.01183</td><td>0.02599</td><td>52.67</td><td>0.9889</td></tr></table>

![](images/efe8bfaf4030a9c191cfbb129849636ff489053daa00d8007b169ba61e861494.jpg)

![](images/d1a38c06345519bda7efcf251bda5b59b4a8ed4b46f5a9cfd94e355fca8867e7.jpg)

![](images/c7bd6ee823df9f44387a9e72f7018d5eebad475beaeec26f9ff4b613d9e68650.jpg)

![](images/b38a2b30c096133c15a5e71a280b2ae78421457bcdd63cbc58b1507403ab89a9.jpg)  
Fig. 3: Per-band (a) MAE, (b) PSNR, (c) RMSE, and (d) 1-SSIM value comparison. FNO is excluded from the analysis due to extremely high errors. SSIM is reported as 1-SSIM and on a log scale for enhanced visibility.

TABLE III: Error metrics over testing dataset with reduced number of bands seen during training
<table><tr><td>% bands used</td><td>MAE↓</td><td>RMSE↓</td><td>PSNR ↑</td><td>SSIM ↑</td></tr><tr><td>100</td><td>0.01183</td><td>0.02599</td><td>52.67</td><td>0.9889</td></tr><tr><td>95</td><td>0.01422</td><td>0.04756</td><td>48.02</td><td>0.9764</td></tr><tr><td>90</td><td>0.01509</td><td>0.04836</td><td>47.50</td><td>0.9748</td></tr><tr><td>85</td><td>0.01701</td><td>0.05290</td><td>46.64</td><td>0.9730</td></tr><tr><td>80</td><td>0.01844</td><td>0.05638</td><td>46.10</td><td>0.9716</td></tr><tr><td>75</td><td>0.01946</td><td>0.05599</td><td>45.64</td><td>0.9707</td></tr><tr><td>70</td><td>0.02077</td><td>0.07779</td><td>43.27</td><td>0.9690</td></tr><tr><td>65</td><td>0.02615</td><td>0.08810</td><td>41.92</td><td>0.9644</td></tr><tr><td>60</td><td>0.02837</td><td>0.08463</td><td>42.32</td><td>0.9620</td></tr><tr><td>55</td><td>0.03239</td><td>0.1046</td><td>40.62</td><td>0.9557</td></tr><tr><td>50</td><td>0.03929</td><td>0.1075</td><td>39.84</td><td>0.9502</td></tr></table>

demonstrating a lower RMSE than the 80% model and the 60% model demonstrating a lower RMSE and higher PSNR than the 65% model. Such behavior likely relates to which bands were removed for training. If bands with higher errors in figure 3 were removed, the model would not be penalized for poor performance for those bands, increasing the error of the final trained model.

Nevertheless, SSRON remains competitive with the state-ofthe-art performance at 90% of bands used and attains a lower MAE than other methods. Because SSRON takes wavelengths as a continuous input, the model can in-principle be evaluated between the native sensor bands. The zero-shot results suggest the potential for such spectral interpolation, although validation against denser spectral measurements would be needed to confirm the accuracy of HSI beyond the native discretization.

## V. CONCLUSION AND FUTURE DIRECTIONS

This work presents SSRON, a deep operator network that uses spatial-spectral convolutions with residual connections as its branch network. Empirically, the SSRON achieves state-ofthe-art accuracy on simple SSR and demonstrates its potential for zero-shot prediction of additional bands.

The paper offers a novel perspective on the SSR task and opens a new direction for applications of operator learning. Future work should focus on optimizing neural operator architectures to further improve model performance. In addition, the continuous spectral representation enabled by this framework motivates further study of the densely sampled spectral interpolation for downstream applications, such as materials classification of imaged regions. Finally, varying the input SRF to support the general application of SSRON to various satellites is an important direction for future work.

## VI. ACKNOWLEDGEMENTS

The author is grateful to receive funding from the George W. Housner Student Discovery Fund for support in the research and attendance at the conference. The author would like to thank Enes Banushi for productive discussion on the problem formulation.

[1] W. L. Smith, Q. Zhang, M. Shao, and E. Weisz, “Improved severe weather forecasts using leo and geo satellite soundings,” Journal ofAtmospheric and Oceanic Technology, vol. 37, no. 7, p. 1203–1218, Jul. 2020.

[2] P. Singh, P. C. Pandey, G. P. Petropoulos, A. Pavlides, P. K. Srivastava, N. Koutsias, K. A. K. Deng, and Y. Bao, Hyperspectral remote sensing in precision agriculture: present status, challenges, and future trends. Elsevier, 2020, p. 121–146.

[3] S. Chander, A. Gujrati, A. V. Krishna, A. Sahay, and R. Singh, Remote sensing of inland water quality: a hyperspectral perspective. Elsevier, 2020, p. 197–219.

[4] S. Imran, S. Mandal, A. Goswami, M. Thakur, and A. Raju, “A novel deep learning-based landsat 7 etm+ multi-spectral to hyperspectral reconstruction model: Application for water bodies in an indian region,” in IGARSS 2024 - 2024 IEEE International Geoscience and Remote Sensing Symposium, 2024, pp. 3121– 3124.

[5] J. Xie, L. Fang, C. Wu, F. Xie, and J. Chanussot, “Blind spectral super-resolution by estimating spectral degradation between unpaired images,” IEEE Transactions on Geoscience and Remote Sensing, vol. 62, pp. 1–14, 2024.

[6] B. Arad and O. Ben-Shahar, Sparse Recovery of Hyperspectral Signal from Natural RGB Images. Springer International Publishing, 2016, p. 19–34.

[7] X. Han, J. Yu, J. Luo, and W. Sun, “Reconstruction from multispectral to hyperspectral image using spectral library-based dictionary learning,” IEEE Transactions on Geoscience and Remote Sensing, vol. 57, no. 3, pp. 1325–1335, 2019.

[8] K. Fotiadou, G. Tsagkatakis, and P. Tsakalides, “Spectral super resolution of hyperspectral images via coupled dictionary learning,” IEEE Transactions on Geoscience and Remote Sensing, vol. 57, no. 5, pp. 2777–2797, 2019.

[9] T. Li and Y. Gu, “Progressive spatial–spectral joint network for hyperspectral image reconstruction,” IEEE Transactions on Geoscience and Remote Sensing, vol. 60, pp. 1–14, 2022.

[10] K. Mu, Z. Zhang, Y. Qian, S. Liu, M. Sun, and R. Qi, “Srt: A spectral reconstruction network for gf-1 pms data based on transformer and resnet,” Remote Sensing, vol. 14, no. 13, p. 3163, Jul. 2022.

[11] X. Zheng, W. Chen, and X. Lu, “Spectral super-resolution of multispectral images using spatial–spectral residual attention network,” IEEE Transactions on Geoscience and Remote Sensing, vol. 60, pp. 1–14, 2022.

[12] D. Du, Y. Gu, T. Liu, and X. Li, “Spectral reconstruction from satellite multispectral imagery using convolution and transformer joint network,” IEEE Transactions on Geoscience and Remote Sensing, vol. 61, p. 1–15, 2023.

[13] R. Gonzalez, C. M. Albrecht, N. A. Ali Braham, D. Lambhate, J. L. De Sousa Almeida, P. Fraccaro, B. Blumenstiel, T. Brunschwiler, and R. Bangalore, “Multispectral to hyperspectral using pretrained foundational model,” in IGARSS 2025 - 2025 IEEE International Geoscience and Remote Sensing Symposium, 2025, pp. 785–789.

[14] T. J. Vandal, D. McDuff, W. Wang, K. Duffy, A. Michaelis, and R. R. Nemani, “Spectral synthesis for geostationary satelliteto-satellite translation,” IEEE Transactions on Geoscience and Remote Sensing, vol. 60, p. 1–11, 2022.

[15] L. Lu, P. Jin, G. Pang, Z. Zhang, and G. E. Karniadakis, “Learning nonlinear operators via DeepONet based on the universal approximation theorem of operators,” Nat. Mach. Intell., vol. 3, no. 3, pp. 218–229, Mar. 2021.

[16] Z. Li, N. B. Kovachki, K. Azizzadenesheli, B. liu, K. Bhattacharya, A. Stuart, and A. Anandkumar, “Fourier neural operator for parametric partial differential equations,” in International Conference on Learning Representations,

2021. [Online]. Available: https://openreview.net/forum?id= c8P9NQVtmnO

[17] M. Wei and X. Zhang, “Super-resolution neural operator,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), June 2023, pp. 18 247–18 256.

[18] L. Han and X. Zhang, “Scalable super-resolution neural operator,” in Proceedings of the 32nd ACM International Conference on Multimedia, ser. MM ’24. New York, NY, USA: Association for Computing Machinery, 2024, p. 10036–10045. [Online]. Available: https://doi.org/10.1145/3664647.3681374

[19] Y. Zhang, H. Zheng, D. Yang, Z. Chen, H. Ma, and W. Ding, “Space-time video super-resolution with neural operator,” IEEE Transactions on Image Processing, vol. 34, p. 6742–6754, 2025.

[20] Z. Zhang, B. Pan, and Z. Shi, “Radiative-structured neural operator for continuous spectral super-resolution,” 2025.

[21] Y. Mei, Y. Zhang, X. Zhu, R. Gou, and J. Gao, “Fully convolutional network-enhanced deeponet-based surrogate of predicting the travel-time fields,” IEEE Transactions on Geoscience and Remote Sensing, vol. 62, pp. 1–12, 2024.

[22] Z. Guo, L. Chai, S. Huang, and Y. Li, “Inversion-deeponet: A novel deeponet-based network with encoder-decoder for full waveform inversion,” arXiv preprint arXiv:2408.08005, 2024.

[23] Z. Shu, Z. Liu, J. Zhou, S. Tang, Z. Yu, and X.-J. Wu, “Spatial–spectral split attention residual network for hyperspectral image classification,” IEEE Journal of Selected Topics in Applied Earth Observations and Remote Sensing, vol. 16, pp. 419–430, 2023.

[24] S. O. Atik, “Dual-stream spectral-spatial convolutional neural network for hyperspectral image classification and optimal band selection,” Advances in Space Research, vol. 74, no. 5, p. 2025–2041, Sep. 2024.

[25] R. Green, “Emit l1b at-sensor calibrated radiance and geolocation data 60 m v001,” 2022. [Online]. Available: https: //www.earthdata.nasa.gov/data/catalog/lpcloud-emitl1brad-001

[26] E. S. A. (ESA), “Sentinel-2 spectral response functions,” June 2024. [Online]. Available: https://sentiwiki.copernicus.eu/ attachments/1692737/ COPE-GSEG-EOPG-TN-15-0007%20-%20Sentinel-2% 20Spectral%20Response%20Functions%202024%20-%204.0. xlsx?inst-v=59ac1eb3-61c9-4626-b824-7509bfd7c905

[27] S. W. Zamir, A. Arora, S. Khan, M. Hayat, F. S. Khan, and M. Yang, “Restormer: Efficient transformer for high-resolution image restoration,” in 2022 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2022, pp. 5718–5729.

[28] J. Li, C. Wu, R. Song, Y. Li, and F. Liu, “Adaptive weighted attention network with camera spectral sensitivity prior for spectral reconstruction from rgb images,” in 2020 IEEE/CVF Conference on Computer Vision and Pattern Recognition Workshops (CVPRW), 2020, pp. 1894–1903.

[29] M. A. Rahman, Z. E. Ross, and K. Azizzadenesheli, “U-NO: U-shaped neural operators,” Transactions on Machine Learning Research, 2023. [Online]. Available: https://openreview.net/ forum?id=j3oQF9coJd

[30] Z. Wang, A. Bovik, H. Sheikh, and E. Simoncelli, “Image quality assessment: from error visibility to structural similarity,” IEEE Transactions on Image Processing, vol. 13, no. 4, p. 600–612, Apr. 2004.

[31] M. Liu-Schiaffini, J. Berner, B. Bonev, T. Kurth, K. Azizzadenesheli, and A. Anandkumar, “Neural operators with localized integral and differential kernels,” arXiv preprint arXiv:2402.16845, 2024.
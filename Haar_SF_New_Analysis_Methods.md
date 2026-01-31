

---

# Technical Report: Multi-Scale Diagnostic Attribution (MSDA) for River Water Quality

**Revised and Statistically Clarified Specification**

> **Note:** This document describes the *target specification* for the MSDA framework. For details on the methods currently implemented in the package, see [Haar Implementation Details](docs/HAAR_IMPLEMENTATION_DETAILS.md).

## 1. Executive Summary

The Multi-Scale Diagnostic Attribution (MSDA) framework characterizes variability and intermittency in river water-quality time series across temporal scales using Haar-based structure functions. The framework is designed for irregular, gappy, and non-Gaussian environmental data where Fourier, ARIMA, or covariance-based approaches are inappropriate.

MSDA diagnoses dominant forcing mechanisms by jointly analyzing:

* pulse-sensitive variability driven by rare, high-magnitude events, and
* robust measures of typical background variability.

Applications include attribution of climatic versus anthropogenic forcing, evaluation of regulatory interventions, and longitudinal comparison of river reaches.

---

## 2. Mathematical Foundation

### 2.1 Time-Weighted Haar Fluctuations for Irregular Time Series

Let ( x(t) ) be an irregularly sampled time series. For a window of duration ( \tau ) starting at time ( t ), define two adjacent half-windows:
[
[t, t+\tfrac{\tau}{2}) \quad \text{and} \quad [t+\tfrac{\tau}{2}, t+\tau)
]

For irregular timestamps, means are computed as time-weighted averages:
[
\bar{x}_L(t,\tau) = \frac{\sum_{j \in L} x_j \Delta t_j}{\sum_{j \in L} \Delta t_j}, \quad
\bar{x}_R(t,\tau) = \frac{\sum_{j \in R} x_j \Delta t_j}{\sum_{j \in R} \Delta t_j}
]

The Haar fluctuation is:
[
D(t,\tau) = \bar{x}_R(t,\tau) - \bar{x}_L(t,\tau)
]

Time weights ( \Delta t_j ) are assigned as half the interval to adjacent timestamps:

* interior points: ( \Delta t_j = (t_{j+1} - t_{j-1})/2 )
* first/last point in a half-window: truncated to the window boundary

This ensures unbiased integration over irregular sampling.

**Window validity criteria:**
A half-window is considered valid if:

* cumulative time coverage ( \sum \Delta t_j \ge 0.7(\tau/2) ), and
* at least a minimum number of observations is present (default: 5).

Temporal coverage is defined by summed time weights, not by sample count.

---

### 2.2 Structure Functions

For order ( q > 0 ), the structure function is:
[
S(q,\tau) = \langle |D(t,\tau)|^q \rangle_t
]

Two complementary estimators are used.

#### 2.2.1 Second-Order Structure Function (Pulse-Sensitive)

[
S_2(\tau) = \langle D(t,\tau)^2 \rangle_t
]

This quantity is sensitive to high-magnitude, low-frequency events such as storm-driven pulses or spills.

#### 2.2.2 Robust Central Tendency of Squared Fluctuations

[
\widetilde{S}_2(\tau) = \operatorname{median}_t \left[ D(t,\tau)^2 \right]
]

This statistic is **not** a second-order moment in the strict statistical sense. It represents a robust central tendency of squared fluctuations that suppresses the influence of extreme values while remaining in squared units. It is used for comparative diagnostics rather than as a variance-equivalent quantity.

---

## 3. Scaling Analysis

### 3.1 Scaling Law and Assumptions

Structure functions are evaluated over a range of scales ( \tau ):
[
S(q,\tau) \sim \tau^{\xi(q)}
]

Scaling analysis assumes approximate statistical self-similarity over the analyzed scale range. Results are only interpreted within scale intervals where log–log linearity is empirically supported.

Scaling exponents are estimated via **robust linear regression** in log–log space using a specified estimator (default: Theil–Sen slope). Alternative estimators (e.g., Huber M-estimator) may be used but must be reported explicitly.

Uncertainty in ( \xi(q) ) is quantified using block bootstrap resampling.

---

### 3.2 Interpretation of Second-Order Scaling (( \beta = \xi(2) ))

* ( \beta \approx 0 ): white or measurement noise
* ( 0 < \beta < 1 ): persistent stochastic variability
* ( \beta \approx 1 ): scale-invariant behavior consistent with 1/f (pink) noise

A value ( \beta \approx 1 ) alone does not distinguish deterministic organization (e.g., diurnal forcing) from stochastic self-organized processes. Additional diagnostics (e.g., spectral peaks or phase coherence) are required for that distinction.

---

## 4. Intermittency Diagnostics

### 4.1 Scale-Dependent Intermittency Index

Intermittency at scale ( \tau ) is defined as:
[
I(\tau) = \log_{10} \left( \frac{S_2(\tau)}{\widetilde{S}_2(\tau)} \right)
]

Interpretation:

* ( I(\tau) > 0 ): dominance of rare, high-magnitude events
* ( I(\tau) \approx 0 ): regime-consistent variability

Negative values of ( I(\tau) ) may occur due to finite-sample effects or resampling variability and are interpreted as statistical noise rather than physically meaningful “negative intermittency.” Such values should be reported but not over-interpreted.

---

### 4.2 Multifractal Scaling (Recommended)

Scaling exponents ( \xi(q) ) are estimated for multiple orders ( q \in [1,3] ). Curvature of ( \xi(q) ) as a function of ( q ) indicates intermittency:

* linear ( \xi(q) ): monofractal (non-intermittent) scaling
* nonlinear ( \xi(q) ): multifractal, intermittent behavior

This analysis is recommended when intermittency attribution is a primary objective.

---

## 5. Attribution Framework

### 5.1 Climate-Driven Forcing

A climate signature is identified when:

* ( I(\tau) ) is elevated at short to intermediate scales,
* scaling exponents align with those of rainfall or discharge time series, and
* statistically significant scale-dependent coherence exists between water quality and hydrologic drivers.

Cross-scale coherence is assessed using external tools (e.g., wavelet coherence). This analysis is complementary to, but not contained within, the MSDA structure-function framework.

---

### 5.2 Anthropogenic Forcing

An anthropogenic signature is identified when:

* ( I(\tau) ) remains low across scales,
* ( \beta ) is stable across hydrologic regimes, and
* coherence with rainfall or discharge is weak or absent.

---

## 6. Regulatory Intervention Analysis

Pre- and post-intervention periods are compared using scaling and intermittency diagnostics.

* **Erratic discharge control:**
  Reduction in ( S_2(\tau) ) and ( I(\tau) ) at short scales.

* **Baseline load reduction:**
  Downward shift in ( \widetilde{S}_2(\tau) ) across scales with minimal change in ( \beta ).

Changes are considered significant when they exceed 95 % bootstrap confidence intervals. Statistical power depends on the duration of pre- and post-intervention records; short post-intervention periods may limit detectability.

---

## 7. Longitudinal and Reach-Scale Comparison

Upstream and downstream sites are compared via scaling behavior rather than direct concentration subtraction.

* Downstream reduction in ( I(\tau) ) indicates stabilizing point-source inputs.
* Differences in ( \beta ) indicate physical modification (e.g., dam regulation, wetland attenuation).

When sites are spatially correlated, bootstrap procedures should respect both temporal and spatial dependence (e.g., spatially blocked bootstrap).

---

## 8. Implementation Roadmap

### Phase 1: Pre-Processing

* Input: long-format data (Timestamp, Value)
* No interpolation
* Time-weighted coverage assessment
* Windowing: overlapping or non-overlapping Haar windows
* Scales ( \tau ): powers of two and/or domain-specific intervals

---

### Phase 2: Statistical Computation

* Compute time-weighted Haar fluctuations ( D(t,\tau) )
* Calculate ( S_2(\tau) ) and ( \widetilde{S}_2(\tau) )
* Estimate scaling exponents via specified robust regression
* Compute ( I(\tau) )
* Quantify uncertainty via block bootstrap

---

### Phase 3: Analytical Outputs

* Log–log structure-function plots with confidence intervals
* Scale-specific estimates of ( \beta ) and ( I(\tau) )
* Pre/post-intervention and upstream/downstream comparisons
* Multifractal spectra ( \xi(q) ) (recommended)
* External coherence analysis with hydrologic drivers where applicable

---

Here are **specific, relevant references** where Haar methods *(or closely analogous wavelet/fluctuation methods)* and **robust multiscale analysis** have been applied to distinguish **external forcing (e.g., climate drivers)** from **intrinsic or “internal” variability or human‑driven signals** — particularly in environmental and time‑series contexts. Where applicable, I include brief notes on how each reference relates to the separation problem you care about:

---

## **1) Haar / Wavelet Applications to Climatic vs. Internal Variability**

**Moore, J., Zhang, Z., & Grinsted, A. (2014). *Haar wavelet analysis of climatic time series***
*International Journal of Wavelets, Multiresolution and Information Processing, 12(2).*
This paper applies Haar wavelet analysis to climatic time series to extract intrinsic variability from background red noise. The approach helps separate scale‑dependent features potentially linked to climate forcing from stochastic (internal) variability. ([Lapin yliopiston tutkimusportaali][1])

---

## **2) Haar Structure Functions in Paleoclimatic Variability Analysis**

**Acton et al. (2023). *Haar structure functions and cross‑Haar correlation analysis of paleoclimatic time series***
*EGUsphere preprint*
This study uses Haar structure functions and cross‑Haar correlations on irregular climate proxies to interpret latitudinal dependencies in climate transitions, demonstrating how Haar‑based scaling can highlight distinct climate regime forcing mechanisms across scales. ([EGUsphere][2])

---

## **3) Comparative Estimation of Scaling for Climate Regimes**

**Hébert, R., Rehfeld, K., & Laepple, T. (2021). *Comparing estimation techniques for temporal scaling in palaeoclimate time series***
*Nonlinear Processes in Geophysics*
This paper contrasts Haar structure function methods with other irregular‑data techniques to reliably estimate scaling exponents. The focus is on correctly identifying scaling behavior tied to distinct climate processes, which helps distinguish forced vs. intrinsic variability in palaeoclimate records. ([NPG][3])

---

## **4) Lovejoy & Schertzer — Intermittency and Geophysical Scaling (Context for Forcing vs. Internal)**

**Lovejoy & Schertzer (2012). *Haar wavelets, fluctuations and structure functions: convenient choices for geophysics***
*Nonlinear Processes in Geophysics*
While not directly attributing forcings, this foundational work establishes Haar fluctuations and structure functions as tools for multiscale geophysical processes and is widely cited in studies separating deterministic external patterns (e.g., seasonal/climate) from stochastic internal variability. ([NPG][4])

---

## **5) Conceptual and Methodological Foundations Relevant to Separation of Drivers**

While not always using Haar directly, the following works place the problem of separating external forcing from internal/human influences in a multiscale decomposition context — useful for framing MSDA’s attribution:

**(Wavelet regression for hydro‑climate relationships)**
Xu, J. (2018). *Wavelet regression: multi‑time‑scale hydro‑climate relationships.* arXiv.
Wavelet decomposition combined with regression to quantify climate vs. other drivers across scales. ([arXiv][5])

**HESS Review on Nonstationary Weather & Water Extremes**
Slater et al. (2021). *Nonstationary weather and water extremes: detection, attribution, and management.* Hydrology and Earth System Sciences.
A comprehensive review emphasizing methods that separate internal variability vs. external forcings (climate change) in hydrological extremes — situating wavelet approaches in broader context. ([HESS][6])

---

## **6) Tools for Robust Wavelet‑Based Inference (Supporting Application of MSDA)**

These references aren’t directly about climate vs. human drivers, but they extend wavelet/Haar frameworks with **robust statistical inference**, which is critical when attributing drivers in noisy environmental data:

**Guerrier et al. (2020). *Robust two‑step wavelet‑based inference for time series models***
*arXiv*
Provides wavelet variance estimation methods robust to outliers and useful for multiscale decomposition, enhancing the reliability of separation between competing signal sources. ([arXiv][7])

**Haar–Fisz Transform for Locally Stationary Variances**
*Fryzlewicz & Nason (2006).*
Combines Haar wavelets with variance stabilization for addressing time‑varying second‑order behavior — useful for attributing scale‑dependent changes (e.g., forcing signals vs. noise). ([OUP Academic][8])

---

## **7) Additional Examples (Wavelet Methods for Attribution in Environmental Systems)**

These references show *wavelet-based decompositions* used to separate influences across scales (e.g., climate drivers vs. other processes), though not always using Haar specifically:

**Rodríguez‑Murillo & Filella (2020). *Wavelet coherence and causality in hydrological time series***
Application of wavelet coherence to detect scale‑dependent relationships (useful in attributing external forcing influences). ([MDPI][9])

**Wavelet-based baseflow separation**
Applied to hydrological discharge to separate baseflow (persistent, possibly anthropogenic) from climate‑driven variability, illustrating how multiscale wavelet decomposition supports attribution. ([Springer][10])

---

## **How These References Support MSDA Attribution Goals**

* *Haar wavelet analysis* can reveal **scale‑dependent variability patterns** that are associated with climate forcing vs. internal dynamics, especially when compared across multiple time series or coupled with coherence measures (Moore et al. 2014; Acton et al. 2023). ([Lapin yliopiston tutkimusportaali][1])
* *Comparative scaling studies* using Haar structure functions validate the method’s utility on irregular data and establish confidence in separating regimes (Hébert et al. 2021). ([NPG][3])
* *Robust wavelet inference* tools strengthen attribution by reducing influence of outliers and ensuring more stable interpretation of scaling changes that could reflect external vs. intrinsic drivers. ([arXiv][7])
* Broader **wavelet coherence and regression** work illustrates how multiscale decompositions can statistically link environmental variables to climate or other drivers. ([MDPI][9])

---

## **Suggested Prioritized Reading for MSDA Application**

1. **Moore, Zhang & Grinsted (2014)** — concrete application of Haar to climatic variability. ([Lapin yliopiston tutkimusportaali][1])
2. **Lovejoy & Schertzer (2012)** — theoretical basis for Haar structure functions and scaling. ([NPG][4])
3. **Hébert, Rehfeld & Laepple (2021)** — irregular data scaling via Haar analysis. ([NPG][3])
4. **Guerrier et al. (2020)** — robust wavelet inference supporting separation reliability. ([arXiv][7])
5. **Rodríguez‑Murillo & Filella (2020)** — wavelet coherence context for attribution. ([MDPI][9])

---

[1]: https://research.ulapland.fi/en/publications/haar-wavelet-analysis-of-climatic-time-series/?utm_source=chatgpt.com "Haar wavelet analysis of climatic time series"
[2]: https://egusphere.copernicus.org/preprints/2023/egusphere-2023-1858/egusphere-2023-1858.pdf?utm_source=chatgpt.com "https://doi.org/10.5194/egusphere-2023-1858"
[3]: https://npg.copernicus.org/articles/28/311/2021/?utm_source=chatgpt.com "NPG - Comparing estimation techniques for temporal scaling in palaeoclimate time series"
[4]: https://npg.copernicus.org/articles/19/513/2012/npg-19-513-2012-metrics.html?utm_source=chatgpt.com "NPG - Metrics - Haar wavelets, fluctuations and structure functions: convenient choices for geophysics"
[5]: https://arxiv.org/abs/1806.06194?utm_source=chatgpt.com "Wavelet regression: An approach for undertaking multi-time scale analyses of hydro-climate relationships"
[6]: https://hess.copernicus.org/articles/25/3897/2021/?utm_source=chatgpt.com "HESS - Nonstationary weather and water extremes: a review of methods for their detection, attribution, and management"
[7]: https://arxiv.org/abs/2001.04214?utm_source=chatgpt.com "Robust Two-Step Wavelet-Based Inference for Time Series Models"
[8]: https://academic.oup.com/jrsssb/article-abstract/68/4/611/7110627?utm_source=chatgpt.com "Haar–Fisz Estimation of Evolutionary Wavelet Spectra | Journal of the Royal Statistical Society Series B: Statistical Methodology | Oxford Academic"
[9]: https://www.mdpi.com/2306-5338/7/4/82?utm_source=chatgpt.com "Significance and Causality in Continuous Wavelet and Wavelet Coherence Spectra Applied to Hydrological Time Series | MDPI"
[10]: https://link.springer.com/article/10.1007/s13201-022-01782-5?utm_source=chatgpt.com "A new approach to use of wavelet transform for baseflow separation of Karst springs (case study: Gamasiyab spring) | Applied Water Science | Springer Nature Link"

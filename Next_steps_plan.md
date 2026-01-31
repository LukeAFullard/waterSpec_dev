
# REPORT ID: WQ-HAAR-2026-02

## Multi-Scalar Analysis of Non-Stationary Water Quality Time Series Using Haar Fluctuation Metrics

**Date:** 31 January 2026
**Prepared for:** Environmental Data Science Division

---

## 1. Executive Summary

Traditional hydrological statistics—such as Pearson correlation, linear regression, and Seasonal Kendall trend tests—implicitly assume stationarity, linearity, and regular sampling. River water-quality datasets routinely violate these assumptions: they are irregularly sampled, contain gaps, and are influenced by interacting climatic, hydrological, and anthropogenic drivers across multiple time scales.

This report presents a robust analytical framework based on **Haar Fluctuation Metrics**, a time-domain, scale-explicit approach for analyzing non-stationary environmental time series. By decomposing time series into localized, scale-dependent step changes using windowed averages, the framework accommodates uneven sampling (without interpolation) under window-density constraints and explicitly separates variability by time scale.

The method enables climate-driven variability, catchment memory, and management signals to be evaluated independently.

The toolkit comprises four stages:

1. **Temporal Instability Analysis** – detection of regime shifts, episodic events, and volatility clustering
2. **Structure Function Characterization** – quantification of system memory and dominant noise behavior
3. **Scale-Specific Attribution** – separation of flow-driven (climatic) variability from flow-independent processes using cross-Haar correlation and surrogates
4. **Lagged Response Analysis** – estimation of dominant hydrological and biogeochemical response times

**Management outcome:**
This framework allows regulators and scientists to distinguish anthropogenic change from natural climatic variability, identify characteristic recovery horizons imposed by catchment memory, compare sites and interventions on a physically meaningful basis, and detect changes invisible to monotonic trend tests.

---

## 2. Theoretical Framework

### 2.1 Generalized Haar Fluctuation

Let ( X(t) ) denote a scalar water-quality time series (e.g., nitrate concentration). For a central time ( t ) and full timescale ( \tau ), the **Haar fluctuation** is defined as:

[
\Delta X(t,\tau)
================

## \overline{X}_{[t,,t+\tau/2]}

\overline{X}_{[t-\tau/2,,t]}
]

where ( \overline{X}_{[a,b]} ) denotes the arithmetic mean of all observations within the continuous-time window ([a,b]).

This symmetric two-half-window definition produces a localized, scale-dependent increment that does not assume global stationarity or periodicity. Each Haar fluctuation represents the mean step change of the system at scale ( \tau ).

---

### 2.2 Handling Uneven Sampling and Uncertainty

Irregular sampling and missing observations are handled by averaging all available observations within each window, subject to a temporal tolerance around the nominal window boundaries.

Key considerations:

* **Bias control:** Windowed averaging avoids bias introduced by unequal sample spacing.
* **Uncertainty:** Windows with few observations have larger standard errors; effective sample counts should be tracked.
* **Intra-window dependence:** Strong autocorrelation within windows may inflate effective variance at small scales.
* **Interpretation:** Small-scale Haar statistics should be interpreted as indicators of catchment-scale variability rather than measurement precision.
* **Regulatory guidance:** For compliance or reporting, exclude windows with ( n < 2 ) or explicitly quantify sensitivity to this choice.

---

### 2.3 Robust Variability Metric: First-Order Structure Function

To quantify variability robustly, we use the **first-order structure function**:

[
S_1(\tau) = \left\langle |\Delta X(t,\tau)| \right\rangle
]

where the average is taken over all valid times ( t ).

Because ( S_1 ) relies on first-order (absolute) increments, it is robust to heavy-tailed distributions and does not require finite second moments. This property is critical for river water-quality time series dominated by episodic transport events.

---

## 3. Structure Function Scaling and System Memory

### 3.1 Scaling Behavior

If variability exhibits self-similar behavior across a range of scales, the structure function follows a power law:

[
S_1(\tau) \propto \tau^{m}
]

where ( m ) is the **Haar scaling exponent**. The value of ( m ) quantifies how variability grows with increasing time scale and provides a direct measure of system memory.

---

### 3.2 Relationship to Spectral Exponents

Under conditions of asymptotic scaling and approximately stationary increments over the analyzed scale range, the first-order structure function corresponds to a power spectral density of the form:

[
P(f) \propto f^{-\beta}
]

with the diagnostic relationship:

[
\beta = 2m + 1
]

**Interpretive note:**
This conversion is conditional and should be treated as diagnostic rather than inferential. It is valid when scaling holds over at least one decade in ( \tau ) and increments are weakly stationary over that range.

---

### 3.3 Diagnostic Interpretation

| Haar slope (m) | Spectral exponent (β) | Diagnosis          | Physical interpretation                          |
| -------------- | --------------------: | ------------------ | ------------------------------------------------ |
| −0.50          |                   0.0 | White noise        | Flashy system; minimal storage; rapid flushing   |
| −0.25          |                   0.5 | Weak persistence   | Limited buffering; shallow groundwater influence |
| 0.00           |                   1.0 | Pink noise         | Strong persistence; aquifer or lake dominance    |
| +0.50          |                   2.0 | Integrated process | Cumulative storage effects; legacy influence     |

Positive slopes indicate increasing long-range dependence and prolonged system memory.

---

## 4. The Four-Stage Haar Toolkit

### Method 1: Temporal Instability Analysis (Raw Haar Fluctuations)

**Objective:** Detect regime shifts, volatility clustering, and potential measurement artifacts.

For a fixed, policy-relevant scale ( \tau ), raw Haar fluctuations ( \Delta X(t,\tau) ) are plotted against time.

**Interpretation:**

* Isolated high-magnitude excursions indicate abrupt shifts in mean state (e.g., infrastructure failure or intervention).
* Sustained periods of elevated variability indicate transitional or unstable regimes.
* Variable-specific anomalies inconsistent across parameters suggest sampling or laboratory errors.

---

### Method 2: System Characterization (Structure Function Scaling)

**Objective:** Quantify intrinsic system memory and dominant noise behavior.

The structure function ( S_1(\tau) ) is computed across a range of scales and plotted in log–log space. Linear scaling regions are identified and used to estimate the scaling exponent ( m ).

Bootstrap resampling (using block lengths consistent with observed autocorrelation) is recommended to estimate confidence intervals for ( m ).

**Assumptions for scaling inference:**

* Scaling holds over a sufficiently wide range (ideally ≥ one decade).
* Increments are approximately weakly stationary over that range.
* Record length is sufficient relative to the largest analyzed scale.

---

### Method 3: Cross-Haar Correlation and Scale-Specific Attribution

**Objective:** Separate flow-driven (climatic) variability from flow-independent processes.

At each scale ( \tau ), Pearson correlation is computed between Haar fluctuations of concentration ( C ) and discharge ( Q ):

[
\rho_{CQ}(\tau) = \text{corr}\big(\Delta C(t,\tau),, \Delta Q(t,\tau)\big)
]

**Choice of correlation metric:**

* **Pearson correlation** is used as the primary metric because Haar increments are scale-filtered and tend toward symmetric, approximately linear relationships. Pearson preserves amplitude information, which is physically meaningful for transport processes.
* **Spearman rank correlation** is used as a sensitivity analysis for highly skewed variables (e.g., turbidity, E. coli) or when nonlinear threshold behavior persists after Haar filtering.

**Phenomenological decomposition:**

[
\Delta C = \alpha(\tau),\Delta Q + \varepsilon(\tau)
]

where ( \alpha(\tau) ) represents scale-specific sensitivity to flow and ( \varepsilon(\tau) ) represents variance unexplained by discharge.

This decomposition is diagnostic, not causal.

---


**PART 2: Significance, Attribution, and Final Recommendations**

---

## 5. Significance Testing Using Surrogates

### 5.1 Why Classical Significance Tests Fail Here

Classical parametric tests assume:

* independent samples,
* Gaussian residuals,
* stationarity.

Haar fluctuations violate **independence** by construction (overlapping windows), and river systems violate **stationarity** by physics. Therefore, naive p-values on correlations or slopes are not defensible.

**Surrogate analysis** provides scale-consistent significance testing without violating physical assumptions.

---

### 5.2 Surrogate Strategy

Surrogates are generated to preserve selected properties of the original time series while destroying the hypothesized coupling.

**Recommended surrogate types:**

| Purpose                 | Surrogate type              | Preserves                             | Destroys               |
| ----------------------- | --------------------------- | ------------------------------------- | ---------------------- |
| Cross-scale correlation | Phase-randomized surrogates | Power spectrum, marginal distribution | Temporal alignment     |
| Lag detection           | Block-shuffled surrogates   | Short-term autocorrelation            | Long-term dependence   |
| Nonlinearity check      | IAAFT surrogates            | Spectrum + distribution               | Higher-order structure |

For discharge-driven attribution, **phase-randomized surrogates of ( Q )** are preferred, leaving ( C ) unchanged.

---

### 5.3 Surrogate-Based Significance of Cross-Haar Correlation

At each scale ( \tau ):

1. Compute observed cross-Haar correlation:
   [
   \rho_{\text{obs}}(\tau)
   ]

2. Generate ( N ) surrogate discharge series ( Q^{(s)} ).

3. Compute:
   [
   \rho^{(s)}(\tau) = \text{corr}\big(\Delta C(t,\tau),, \Delta Q^{(s)}(t,\tau)\big)
   ]

4. Estimate empirical significance:
   [
   p(\tau) = \frac{1}{N} \sum_s \mathbb{I}\big(|\rho^{(s)}| \ge |\rho_{\text{obs}}|\big)
   ]

**Interpretation:**

* Scales with significant correlation identify **flow-controlled regimes**.
* Non-significant scales identify **flow-independent dynamics** (e.g. legacy nutrients, in-stream processing).

---

### 5.4 Multiple Testing Across Scales

Because scales are not independent, strict Bonferroni correction is inappropriate.

Recommended approaches:

* Control the **false discovery rate (FDR)** across scales.
* Alternatively, interpret **contiguous scale bands** of significance rather than isolated points.

---

## 6. Lagged Cross-Haar Correlation

### 6.1 Purpose

Lagged cross-Haar correlation detects **response delays** between discharge forcing and water-quality response.

This is critical for:

* groundwater residence time inference,
* biogeochemical processing delays,
* hysteresis detection.

---

### 6.2 Definition

For lag ( \ell ):

[
\rho_{CQ}(\tau,\ell)
====================

\text{corr}\big(\Delta C(t,\tau),, \Delta Q(t-\ell,\tau)\big)
]

Computed over a grid of ( (\tau,\ell) ).

---

### 6.3 Interpretation

| Pattern                 | Interpretation                                   |
| ----------------------- | ------------------------------------------------ |
| Peak at ( \ell = 0 )    | Immediate flushing / transport                   |
| Peak at ( \ell > 0 )    | Delayed mobilization (groundwater, soil storage) |
| Sign change with lag    | Event-scale hysteresis                           |
| Broad ridge in ( \ell ) | Distributed residence times                      |

Surrogate testing applies identically, but surrogates must preserve the autocorrelation structure of ( Q ).

---

## 7. Addressing the Core Water-Science Questions

### 7.1 Can We Separate Climate/Flow Effects from Trend?

Yes.

* Flow influence is isolated via scale-specific cross-Haar correlation.
* Long-scale structure function behavior reflects intrinsic catchment memory.
* Residual Haar increments (( \varepsilon(\tau) )) quantify non-climatic variability.

This avoids explicit detrending, which risks removing physically meaningful low-frequency dynamics.

---

### 7.2 Can We Compare Upstream and Downstream Sites?

Yes, directly and physically.

Compare:

* structure function slopes ( m(\tau) ),
* breakpoints in scaling,
* scale-specific flow sensitivity ( \alpha(\tau) ).

Differences indicate:

* additional storage,
* attenuation or amplification,
* management effects between sites.

---

### 7.3 Can We Detect Regulatory Change (Before vs After)?

Yes — but not with a single trend slope.

Recommended approach:

1. Compute Haar metrics separately for pre- and post-intervention periods.
2. Compare:

   * overall scaling exponent,
   * loss or emergence of long-memory regimes,
   * reduction in flow sensitivity at regulatory scales.

A successful intervention typically:

* reduces long-scale variance,
* weakens cross-Haar correlation with discharge,
* shortens effective memory.

---

### 7.4 Can We Identify Characteristic Memory Scales?

Yes — this is one of the strongest advantages of the Haar approach.

Memory scales manifest as:

* breaks in structure-function scaling,
* plateaus or slope changes,
* collapse of cross-correlation beyond a threshold scale.

These correspond to:

* aquifer residence times,
* reservoir flushing times,
* soil nutrient turnover horizons.

---

## 8. Pearson vs Spearman: Final Guidance

**Primary recommendation:**
Use **Pearson correlation** on Haar fluctuations.

**Rationale:**

* Haar filtering suppresses skewness and nonstationarity.
* Transport processes are amplitude-driven.
* Pearson preserves physical scaling.

**Mandatory sensitivity check:**
Recompute using **Spearman** when:

* distributions remain heavy-tailed after Haar filtering,
* thresholds dominate (e.g. pathogen mobilization),
* regulatory interpretation depends on monotonicity rather than magnitude.

Discrepancies between Pearson and Spearman are **diagnostic**, not problematic.

---

## 9. Confidence Intervals for Structure Functions and Slopes

### 9.1 Bootstrap Strategy

Use **block bootstrap** on Haar increments:

* Block length selected via integral time scale or first zero-crossing of autocorrelation.
* Resample blocks with replacement.
* Recompute ( S_1(\tau) ) and scaling slope for each bootstrap replicate.

---

### 9.2 Outputs

* Pointwise confidence intervals for ( S_1(\tau) ).
* Confidence intervals for scaling exponent ( m ).
* Confidence intervals for derived ( \beta ).

---

## 10. Additional Considerations Identified in Final Review

### 10.1 Window Overlap Dependence

Overlapping Haar windows induce dependence. This affects:

* variance estimation,
* naive degrees of freedom.

Surrogate and bootstrap methods implicitly account for this and should always be preferred.

---

### 10.2 Minimum Record Length

Practical guidance:

* Maximum reliable scale ( \tau_{\max} \approx T/5 ).
* Scaling inference requires ≥ 50 effective Haar increments per scale.

---

### 10.3 Mean vs Median Haar

Median-based Haar fluctuations are possible and recommended when:

* extreme events dominate means,
* interest is in regime shifts rather than flux.

Mean-based Haar is preferred for transport and mass-balance interpretation.

---

### 10.4 Units and Communication

Haar fluctuations retain original units. This is a major advantage for stakeholder communication:

> “Variability at 1-year scale decreased by 30% post-intervention.”

---

## 11. Final Recommendations

This framework should be adopted when:

* climate confounding is unavoidable,
* data are irregular or sparse,
* management operates at specific time scales.

**Key strengths:**

* physically interpretable,
* robust to nonstationarity,
* scale-explicit,
* regulator-friendly.

**Key caution:**

* Haar methods diagnose dynamics; they do not replace mechanistic models.

Used together, Haar fluctuation metrics and surrogate-based inference provide one of the most defensible pathways currently available for multi-scale water-quality attribution.

---



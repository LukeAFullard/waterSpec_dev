
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

**Implementation Platform:** Python (specifically extending the `waterSpec` package).
**Complexity Note:** While naive implementation of sliding windows is $O(n^2)$, this framework recommends using **cumulative sum (integral image)** techniques to achieve $O(n)$ performance, essential for large network analysis.

**Management outcome:**
This framework allows regulators and scientists to distinguish anthropogenic change from natural climatic variability, identify characteristic recovery horizons imposed by catchment memory, compare sites and interventions on a physically meaningful basis, and detect changes invisible to monotonic trend tests.

---

## 2. Theoretical Framework

### 2.1 Generalized Haar Fluctuation

Let $X(t)$ denote a scalar water-quality time series (e.g., nitrate concentration). For a central time $t$ and full timescale $\tau$, the **Haar fluctuation** is defined as:

$$
\Delta X(t,\tau) = \overline{X}_{[t, t+\tau/2]} - \overline{X}_{[t-\tau/2, t]}
$$

where $\overline{X}_{[a,b]}$ denotes the arithmetic mean of all observations within the continuous-time window $[a,b]$.

This symmetric two-half-window definition produces a localized, scale-dependent increment that does not assume global stationarity or periodicity. Each Haar fluctuation represents the mean step change of the system at scale $\tau$.

---

### 2.2 Handling Uneven Sampling and Uncertainty

Irregular sampling and missing observations are handled by averaging all available observations within each window, subject to a temporal tolerance around the nominal window boundaries.

Key considerations:

* **Bias control:** Windowed averaging avoids bias introduced by unequal sample spacing.
* **Window Overlap (CRITICAL):**
    *   **Specification:** To maximize statistical power, windows should overlap significantly. We recommend a "sliding step" of $\Delta t_{step} = \text{min}(\text{median}(dt), \tau/10)$. This typically results in >90% overlap at large scales.
    *   **Effective Sample Size:** Overlap induces correlation. The effective sample size for variance estimation is approximately $n_{eff} \approx N \times (1 - \text{overlap\_fraction})$ or $N_{windows} \times (\Delta t_{step} / \tau)$.
    *   **Requirement:** Bootstrap or surrogate methods (Section 5) are **mandatory** when using overlapping windows to correct for this dependence.
* **Uncertainty:** Windows with few observations have larger standard errors.
* **Minimum Sample Size:**
    *   **Exploratory:** Exclude windows with $n < 5$.
    *   **Reporting/Compliance:** Exclude windows with $n < 10$ to ensure statistical stability of the mean.
    *   **Jurisdiction Check:** Always cross-reference with specific regulatory requirements (e.g., USEPA, EU WFD) which may mandate specific minimums.
* **Intra-window dependence:** Strong autocorrelation within windows may inflate effective variance at small scales.

---

### 2.3 Robust Variability Metric: First-Order Structure Function

To quantify variability robustly, we use the **first-order structure function**:

$$
S_1(\tau) = \left\langle |\Delta X(t,\tau)| \right\rangle
$$

where the average is taken over all valid times $t$.

Because $S_1$ relies on first-order (absolute) increments, it is robust to heavy-tailed distributions and does not require finite second moments. This property is critical for river water-quality time series dominated by episodic transport events.

---

## 3. Structure Function Scaling and System Memory

### 3.1 Scaling Behavior

If variability exhibits self-similar behavior across a range of scales, the structure function follows a power law:

$$
S_1(\tau) \propto \tau^{m}
$$

where $m$ is the **Haar scaling exponent**. The value of $m$ quantifies how variability grows with increasing time scale and provides a direct measure of system memory.

**Segmented Scaling:**
Real-world systems often exhibit regime shifts. The framework supports **Segmented Haar Fits**:
*   **Algorithm:** Use PELT (Pruned Exact Linear Time) or Binary Segmentation on the log-log residuals.
*   **Selection:** Use BIC (Bayesian Information Criterion) to select the optimal number of breakpoints (typically 0, 1, or 2).
*   **Constraint:** Enforce a minimum segment length of 0.5 decades in scale ($\log_{10} \tau$).

---

### 3.2 Relationship to Spectral Exponents

Under conditions of asymptotic scaling and approximately stationary increments over the analyzed scale range, the first-order structure function corresponds to a power spectral density of the form:

$$
P(f) \propto f^{-\beta}
$$

with the diagnostic relationship:

$$
\beta = 2m + 1
$$

**Interpretive note:**
This conversion is conditional and should be treated as diagnostic rather than inferential. It is valid when scaling holds over at least one decade in $\tau$ and increments are weakly stationary over that range.

---

### 3.3 Diagnostic Interpretation

| Haar slope $m$ | Spectral exponent $\beta$ | Diagnosis          | Physical interpretation                          |
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

For a fixed, policy-relevant scale $\tau$, raw Haar fluctuations $\Delta X(t,\tau)$ are plotted against time.

**Interpretation:**

* Isolated high-magnitude excursions indicate abrupt shifts in mean state (e.g., infrastructure failure or intervention).
* Sustained periods of elevated variability indicate transitional or unstable regimes.
* Variable-specific anomalies inconsistent across parameters suggest sampling or laboratory errors.

---

### Method 2: System Characterization (Structure Function Scaling)

**Objective:** Quantify intrinsic system memory and dominant noise behavior.

The structure function $S_1(\tau)$ is computed across a range of scales and plotted in log–log space. Linear scaling regions are identified and used to estimate the scaling exponent $m$.

Bootstrap resampling (using block lengths consistent with observed autocorrelation) is recommended to estimate confidence intervals for $m$.

**Assumptions for scaling inference:**

* Scaling holds over a sufficiently wide range (ideally ≥ one decade).
* Increments are approximately weakly stationary over that range.
* Record length is sufficient relative to the largest analyzed scale.

---

### Method 3: Cross-Haar Correlation and Scale-Specific Attribution

**Objective:** Separate flow-driven (climatic) variability from flow-independent processes.

**Implementation Requirement: Bivariate Alignment**
*   **Matching Tolerance:** Match $C(t)$ and $Q(t)$ within a tolerance of $\pm \text{min}(\tau/20, \text{2 hours})$.
*   **Interpolation:** Strictly **no interpolation** of water quality data. Discharge ($Q$) may be linearly interpolated to match $C$ timestamps if $Q$ resolution is significantly higher (e.g., 15-min $Q$ vs. monthly $C$).
*   **Transformation:** Discharge $Q$ should generally be log-transformed ($\ln Q$) prior to analysis to linearize the C-Q relationship, unless specific conditions dictate otherwise.

At each scale $\tau$, Pearson correlation is computed between Haar fluctuations of concentration $C$ and discharge $Q$:

$$
\rho_{CQ}(\tau) = \text{corr}\big(\Delta C(t,\tau), \Delta Q(t,\tau)\big)
$$

**Choice of correlation metric:**

* **Pearson correlation** is used as the primary metric because Haar increments are scale-filtered and tend toward symmetric, approximately linear relationships. Pearson preserves amplitude information, which is physically meaningful for transport processes.
* **Spearman rank correlation** is used as a sensitivity analysis for highly skewed variables (e.g., turbidity, E. coli) or when nonlinear threshold behavior persists after Haar filtering.

**Phenomenological decomposition:**

$$
\Delta C = \alpha(\tau)\Delta Q + \varepsilon(\tau)
$$

where $\alpha(\tau)$ represents scale-specific sensitivity to flow and $\varepsilon(\tau)$ represents variance unexplained by discharge.

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

For discharge-driven attribution, **phase-randomized surrogates of $Q$** are preferred, leaving $C$ unchanged.

---

### 5.3 Surrogate-Based Significance of Cross-Haar Correlation

At each scale $\tau$:

1. Compute observed cross-Haar correlation:
   $$ \rho_{\text{obs}}(\tau) $$

2. Generate $N$ surrogate discharge series $Q^{(s)}$.

3. Compute:
   $$ \rho^{(s)}(\tau) = \text{corr}\big(\Delta C(t,\tau), \Delta Q^{(s)}(t,\tau)\big) $$

4. Estimate empirical significance:
   $$ p(\tau) = \frac{1}{N} \sum_s \mathbb{I}\big(|\rho^{(s)}| \ge |\rho_{\text{obs}}|\big) $$

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

For lag $\ell$:

$$
\rho_{CQ}(\tau,\ell) = \text{corr}\big(\Delta C(t,\tau), \Delta Q(t-\ell,\tau)\big)
$$

Computed over a grid of $(\tau,\ell)$.

---

### 6.3 Interpretation

| Pattern                 | Interpretation                                   |
| ----------------------- | ------------------------------------------------ |
| Peak at $\ell = 0$      | Immediate flushing / transport                   |
| Peak at $\ell > 0$      | Delayed mobilization (groundwater, soil storage) |
| Sign change with lag    | Event-scale hysteresis                           |
| Broad ridge in $\ell$   | Distributed residence times                      |

Surrogate testing applies identically, but surrogates must preserve the autocorrelation structure of $Q$.

---

## 7. Addressing the Core Water-Science Questions

### 7.1 Can We Separate Climate/Flow Effects from Trend?

Yes.

* Flow influence is isolated via scale-specific cross-Haar correlation.
* Long-scale structure function behavior reflects intrinsic catchment memory.
* Residual Haar increments $\varepsilon(\tau)$ quantify non-climatic variability.

This avoids explicit detrending, which risks removing physically meaningful low-frequency dynamics.

---

### 7.2 Can We Compare Upstream and Downstream Sites?

Yes, directly and physically.

Compare:

* structure function slopes $m(\tau)$,
* breakpoints in scaling,
* scale-specific flow sensitivity $\alpha(\tau)$.

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

### 7.5 Can We Map Catchment Memory & Legacies?

Yes. Use the scaling exponent $m$ from Method 2 to characterize sites across a network.

* **Application:** Identify "legacy-dominated" catchments where $m$ is high ($>0$). These are sites where current water quality is strongly influenced by historical inputs.
* **Benefit:** Sets realistic expectations for recovery horizons (e.g., "This site has 10-year memory; policy impact may be delayed until the next decade").

---

### 7.6 Can We Attribute Variability to Large-Scale Climate Cycles?

Yes. Expand Method 3 (Cross-Haar Correlation) to use climate indices.

* **Application:** Correlate Haar fluctuations of water quality with indices like the North Atlantic Oscillation (NAO) or ENSO at multi-year scales ($\tau = 2\text{--}7$ years).
* **Benefit:** Explicitly separates regional climatic forcing from local anthropogenic impacts.

---

### 7.7 Can We Analyze Network-Scale Sensitivity?

Yes. Compare scale-specific flow sensitivity $\alpha(\tau)$ across sites.

* **Application:** Plot $\alpha(\tau)$ as a function of catchment area or land use across dozens of sites.
* **Benefit:** Detects whether signals are averaged out in larger catchments or if specific land uses amplify climate sensitivity at certain scales.

---

### 7.8 Can We Detect Regulatory Thresholds in the Time Domain?

Yes. Use Method 1 (Raw Haar Fluctuations) at policy-relevant scales.

* **Application:** Plot the 5-year or 10-year step change in concentration over time as a moving window.
* **Benefit:** Provides a physically intuitive "recovery trajectory" that stakeholders can easily interpret compared to abstract trend statistics.

---

## 8. Extended Applications in Water Science

### 8.1 Real-Time Anomaly Detection (Early Warning)

**Concept:** Leverage Method 1 (Temporal Instability) in a sliding window context.

**Utility:**
Standard threshold violations often flag natural diurnal cycles or seasonal peaks as anomalies. By monitoring the *Haar fluctuation* at specific short scales (e.g., $\tau=6$ hours or $\tau=1$ day), operators can distinguish abrupt mechanistic failures or spill events from gradual baseline shifts.

**Implementation:**
*   Implement a "Sliding Haar" class that updates $\Delta X(t, \tau)$ incrementally as new data arrives.
*   Define dynamic thresholds based on historical volatility (e.g., $3\sigma$ of the past month's Haar fluctuations).
*   Flag events where the *change* in concentration exceeds the expected volatility for that specific time scale.

### 8.2 Spatial Scaling (River Network Analysis)

**Concept:** Apply Haar analysis to spatial series $C(x)$ where $x$ is distance downstream or catchment area, instead of time $t$.

**Utility:**
*   Identifies "Hot Spots" and critical source areas for pollution.
*   Quantifies how heterogeneity scales with catchment size via the spatial scaling exponent $m_{space}$.
*   Different $m_{space}$ values indicate different dominant spatial processes (e.g., point-source dominance vs. diffuse non-point source loading).

**Implementation:**
*   Replace time $t$ with spatial coordinate $L$ (river kilometer) or $A$ (cumulative drainage area).
*   Compute Haar fluctuations over spatial windows.
*   Requires synoptic sampling data (longitudinal surveys).

### 8.3 Hysteresis Classification

**Concept:** Analyze the Phase Space of Haar Fluctuations ($\Delta C$ vs $\Delta Q$) at specific scales.

**Utility:**
Disentangles complex transport mechanisms. Classical hysteresis loops often conflate event-scale flushing (clockwise) with seasonal groundwater depletion (counter-clockwise). By analyzing $\Delta C(\tau)$ vs $\Delta Q(\tau)$, one can separate these loops by frequency.

**Implementation:**
*   Extend Method 4.
*   For a given scale $\tau$, plot $\Delta C(t, \tau)$ vs $\Delta Q(t, \tau)$ as a connected path.
*   **Loop Area Metric:** Calculate the signed area $A = \frac{1}{2} \sum (\Delta C_i \Delta Q_{i+1} - \Delta C_{i+1} \Delta Q_i)$.
*   **Sign Convention:** Positive Area ($A>0$) $\rightarrow$ Counter-Clockwise (Groundwater/delayed). Negative Area ($A<0$) $\rightarrow$ Clockwise (Flushing/rapid).

---

## 9. Pearson vs Spearman: Final Guidance

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

## 10. Confidence Intervals for Structure Functions and Slopes

### 10.1 Bootstrap Strategy

Use **block bootstrap** on Haar increments:

* **Block Length Selection:** Use the **integral time scale** $\tau_{int} = \sum_{k=0}^{\infty} \rho(k)$.
    *   **Fallback:** If $\rho(k)$ does not decay to zero, use the first zero-crossing or a conservative lower bound of $3\tau_{max}$.
*   Resample blocks with replacement.
*   Recompute $S_1(\tau)$ and scaling slope for each bootstrap replicate.

### 10.2 Outputs

* Pointwise confidence intervals for $S_1(\tau)$.
* Confidence intervals for scaling exponent $m$.
* Confidence intervals for derived $\beta$.

---

## 11. Additional Considerations Identified in Final Review

### 11.1 Window Overlap Dependence

Overlapping Haar windows induce dependence. This affects variance estimation and naive degrees of freedom.

* **Guidance:** To maximize the utility of long-term records (e.g., 20–30 years of monthly data), the use of **overlapping windows** is strongly recommended to increase the count of valid increments. Surrogate and bootstrap methods must be used to correctly account for the resulting dependence.

---

### 11.2 Minimum Record Length

Practical guidance:

* Maximum reliable scale $\tau_{\max} \approx T/5$.
* Scaling inference requires ≥ 50 effective Haar increments per scale (where $n_{eff}$ accounts for overlap as defined in Section 2.2).

---

### 11.3 Mean vs Median Haar

Median-based Haar fluctuations are possible and recommended when:

* extreme events dominate means,
* interest is in regime shifts rather than flux.

Mean-based Haar is preferred for transport and mass-balance interpretation.

---

### 11.4 Scale Coarseness and Process Resolution

Monthly datasets (common for long-term records) cannot resolve fast hydrological processes like storm-event response times or diurnal cycles.

* **Guidance:** Use Method 4 (Lagged Response) primarily for seasonal to annual response times when working with monthly data. High-frequency sensor data is required for event-scale dynamics.

---

### 11.5 Computational Cost of Surrogates

Surrogate analysis (Method 5) is computationally intensive when applied to large site networks.

* **Guidance:** For networks with $>100$ sites, consider parallelizing surrogate generation or using regional discharge surrogate libraries to reduce redundant calculations.

---

### 11.6 Boundary Effects at Large Scales

At the largest scales ($\tau \approx T/2$), the number of valid Haar fluctuations decreases significantly.

* **Guidance:** Treat Method 1 (Temporal Instability) results near the beginning and end of the record with caution at multi-year scales.

---

### 11.7 Units and Communication

Haar fluctuations retain original units. This is a major advantage for stakeholder communication:

> “Variability at 1-year scale decreased by 30% post-intervention.”

---

## 12. Final Recommendations

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

## 13. Validation Framework

To ensure the reliability of the implemented toolkit, a rigorous validation suite must be established:

1.  **Synthetic Data Tests (fBm):** Generate fractional Brownian motion (fBm) series with known Hurst exponents ($H$). Verify that the Haar scaling exponent $m$ recovers the theoretical relationship $m = H - 1$ (for noise) or appropriate equivalent within 95% confidence intervals.
2.  **Known-Solution Benchmarks:** Use "sawtooth" and sine-wave synthetic signals to verify that the Haar filter correctly identifies the characteristic scale (period) as a break in the structure function.
3.  **Cross-Validation:** Split long-term real-world datasets (e.g., USGS high-frequency nitrate) into training/testing halves to verify the stability of estimated exponents.

---

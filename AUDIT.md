# Audit Report: Multi-Scalar Analysis of Non-Stationary Water Quality Time Series Using Haar Fluctuation Metrics

**Date:** 31 January 2026
**Auditor:** Jules (AI Software Engineer)
**Reference Document:** `Next_steps_plan.md`

---

## 1. Executive Summary

The document `Next_steps_plan.md` outlines a comprehensive and scientifically robust framework for analyzing water quality time series using Haar Fluctuation Metrics. The plan proposes a four-stage toolkit designed to handle irregular sampling and non-stationarity—two chronic issues in hydrological data.

**Verdict:** The plan is **highly worthwhile** to pursue. It moves the codebase from a basic spectral analysis tool (`waterSpec` currently) to a diagnostic engine capable of attributing causes (climate vs. management) and characterizing system memory.

However, a critical audit of the current codebase reveals that **most of the advanced functionality described in the plan is currently missing**, and there is a **significant implementation discrepancy** regarding window overlaps that needs immediate correction.

---

## 2. Gap Analysis: Plan vs. Current Codebase

The following table summarizes the status of the planned methods in the current `src/waterSpec/` implementation:

| Planned Method | Component | Status | Notes |
| :--- | :--- | :--- | :--- |
| **Core Haar Logic** | Calculation | ⚠️ **Partial** | Implemented in `haar_analysis.py`, but uses **non-overlapping windows**, contradicting the plan. |
| **Method 1** | Temporal Instability | ❌ **Missing** | No function exists to return/plot raw fluctuation time series $\Delta X(t,\tau)$. |
| **Method 2** | Structure Function | ✅ **Basic** | `fit_haar_slope` exists and uses robust Mann-Kendall regression. |
| | Segmented Scaling | ❌ **Missing** | Current Haar fitter only supports a single slope. `spectral_analyzer.py` supports segmented fits for Periodograms only. |
| **Method 3** | Cross-Haar Correlation | ❌ **Missing** | No implementation for bivariate analysis (Concentration vs. Discharge). |
| **Method 4** | Lagged Response | ❌ **Missing** | No implementation for lagged cross-correlations. |
| **Method 5** | Surrogate Testing | ❌ **Missing** | Bootstrap exists for slopes, but phase-randomized surrogates for correlation significance are missing. |

---

## 3. Critical Flaws and Issues

### 3.1. The Window Overlap Discrepancy (Major Issue)
Section 10.1 of the plan states:
> *"To maximize the utility of long-term records... the use of **overlapping windows** is strongly recommended to increase the count of valid increments."*

**Current Implementation:**
The function `calculate_haar_fluctuations` in `src/waterSpec/haar_analysis.py` implements **non-overlapping windows**.
```python
# Current code snippet behavior
current_idx = idx_end  # Jumps to the end of the current window
```
**Impact:** This drastically reduces the effective sample size, especially at large scales ($\tau$). For a 20-year record, analyzing a 5-year scale ($\tau=5$) yields only ~4 points with non-overlapping windows, whereas overlapping windows would yield many more, allowing for more robust statistical inference. **This must be fixed to align with the plan.**

### 3.2. Lack of Segmented Haar Fits
The plan (Section 7.4) emphasizes identifying "characteristic memory scales" via "breaks in structure-function scaling".
**Current Implementation:**
`fit_haar_slope` only fits a single power law ($S_1 \propto \tau^m$).
**Impact:** The software cannot currently detect regime shifts in memory (e.g., transition from surface runoff to groundwater dominance) using Haar metrics, which is a key selling point of the plan.

### 3.3. Missing Bivariate Framework
The plan relies heavily on correlating Water Quality ($C$) with Discharge ($Q$) to separate climate effects from management (Method 3).
**Current Implementation:**
The `Analysis` class is strictly univariate. There is no infrastructure to ingest two time series, align them, and compute their joint Haar fluctuations.

---

## 4. Strengths of the Plan

1.  **Physical Interpretability:** Unlike black-box machine learning or abstract spectral densities, Haar fluctuations ($\Delta X$) retain the units of the data (e.g., mg/L), making them intelligible to stakeholders (Section 10.7).
2.  **Robustness to Irregularity:** The plan correctly identifies that standard FFT or even Lomb-Scargle (for slopes) can be biased by gappiness. The Haar approach is natively adapted to this.
3.  **Differentiation of Drivers:** The proposal to use Cross-Haar correlation (Method 3) is a powerful, physically grounded way to separate flow-driven variance from other sources without assuming a static rating curve.

---

## 5. Suggestions for Extended Applications in Water Science

Beyond the scope of the current plan, these methods could be applied to:

### 5.1. Model Evaluation "Turing Test"
**Concept:** Use Haar metrics to validate hydrological models (SWAT, HSPF, LSTM).
**Application:** A model might achieve a high Nash-Sutcliffe Efficiency (NSE) but fail to reproduce the correct *spectral slope* (Haar exponent $m$). This indicates the model is "getting the right answer for the wrong reason" (e.g., lacking long-term groundwater memory).
**Action:** Add a module to compare `Haar(Observed)` vs. `Haar(Simulated)`.

### 5.2. Real-Time Anomaly Detection (Early Warning)
**Concept:** Leverage Method 1 (Temporal Instability).
**Application:** for drinking water intakes, run a "sliding Haar" calculation. A sudden spike in Haar fluctuation at small scales ($\tau=1$ hour) could flag sensor errors or rapid contamination events (e.g., spills) more robustly than simple thresholds, which trigger false alarms on natural diurnal cycles.

### 5.3. Spatial Scaling (River Network Analysis)
**Concept:** The plan focuses on *temporal* scaling ($\tau$). The same math applies to *spatial* scaling ($L$).
**Application:** Calculate Haar fluctuations of synoptic sampling data along a river network. The scaling exponent $m_{space}$ would quantify how heterogeneity grows with catchment area, identifying "Hot Spots" or critical source areas for pollution.

### 5.4. Hysteresis Classification
**Concept:** Extend Method 4 (Lagged Response).
**Application:** By plotting $\Delta C$ vs. $\Delta Q$ at a specific scale $\tau$, one can classify hysteresis loops (clockwise vs. counter-clockwise) specifically for *storm events* (small $\tau$) vs. *seasonal patterns* (large $\tau$). This disentangles complex transport mechanisms that are often lumped together in standard hysteresis analysis.

---

## 6. Recommendation

**Proceed with the plan.**

Prioritize the following implementation order:
1.  **Refactor `haar_analysis.py`** to support overlapping windows.
2.  **Implement Segmented Regression** for Haar plots (porting logic from `spectral_analyzer.py`).
3.  **Build the `BivariateAnalysis` class** to support Method 3 (Cross-Haar).

# Project Audit Report: waterSpec

**Date:** October 26, 2023
**Auditor:** Jules (AI Software Engineer)
**Scope:** Methodological soundness, statistical defensibility, and legal robustness of spectral analysis methods for river/lake water quality.

## Executive Summary

The `waterSpec` project rests on a scientifically sound foundation, correctly identifying the limitations of standard Fourier/Lomb-Scargle methods when applied to irregularly sampled environmental data. The implementation of **Haar Wavelet Analysis** for spectral slope estimation is a significant strength, providing robust quantification of "system memory" (persistence) that is resilient to sampling gaps.

To ensure results withstand rigorous cross-examination (e.g., in a court of law), the package requires:
1.  **Time-Frequency Localization:** Assessing *when* spectral characteristics change (via Weighted Wavelet Z-Transform).
2.  ** rigorous Artifact Rejection:** Moving **PSRESP** (Power Spectral Response) from a standalone tool to a core validation step to prove peaks are not aliasing artifacts.
3.  **Stricter Null Hypotheses:** explicitly modeling **Red Noise (AR1)** to prevent false positives in trend/cycle detection.

---

## 1. Methodological Audit

### A. Lomb-Scargle Periodogram
*   **Status:** Implemented (`astropy.timeseries.LombScargle`).
*   **Verdict:** **Sound but Vulnerable.**
*   **Analysis:** Lomb-Scargle is the standard for irregular data but suffers from **spectral leakage**. Gaps in water quality sampling (e.g., missing weekends, storm-chasing) act as a "window function" that can redistribute power, creating artificial peaks.
*   **Legal Risk:** An opposing expert could claim a detected cycle is an artifact of the sampling schedule.
*   **Mitigation:** Must be paired with **PSRESP** (see below) to prove non-artifactuality.

### B. Haar Wavelet Analysis
*   **Status:** Implemented (`waterSpec.haar_analysis`).
*   **Verdict:** **Highly Defensible.**
*   **Analysis:** This method estimates the first-order structure function ($S_1$). It is far more robust to irregular sampling than Fourier-based methods for estimating the **spectral slope ($\beta$)**. The use of `MannKS` (Mann-Kendall on the spectrum) for slope fitting adds non-parametric robustness.
*   **Application:** Excellent for distinguishing between "event-driven" (runoff, $\beta \approx 0$) vs. "storage-driven" (groundwater, $\beta > 1$) contamination transport.

### C. PSRESP (Power Spectral Response)
*   **Status:** Implemented (`waterSpec.psresp`) but not integrated into the main workflow.
*   **Verdict:** **The "Gold Standard" for Defensibility.**
*   **Analysis:** PSRESP uses forward modeling (Monte Carlo simulation) to degrade synthetic data with the *exact* observation timestamps. If a peak survives this test, it is robust against sampling artifacts.
*   **Recommendation:** This must be elevated from a "power user" tool to a standard validation step in the `Analysis` class.

---

## 2. Statistical Defensibility

*   **Confidence Intervals:** The use of **Bootstrap** resampling (block/wild) is excellent and defensible. It makes fewer assumptions than parametric tests.
*   **Peak Significance:** The project uses False Alarm Probability (FAP) and False Discovery Rate (FDR). This is standard practice.
*   **Null Hypothesis (The "Red Noise" Gap):**
    *   Currently, the system leans towards Power Law fits.
    *   **Risk:** Many environmental variables are simple AR1 (Red Noise) processes. If we test against White Noise, everything looks significant.
    *   **Recommendation:** Explicitly implement an **AR1 (Autoregressive)** null hypothesis. If a cycle beats AR1, it is strong evidence of an external forcing (e.g., specific discharge schedule).

---

## 3. New Method Recommendations

### A. Weighted Wavelet Z-Transform (WWZ)
*   **Why:** Lomb-Scargle gives an *average* spectrum. In legal/regulatory contexts, it is crucial to know **when** a signal appeared. Did the contaminant cycle start *after* the factory opened?
*   **What it is:** A wavelet-like transform specifically designed for irregular data. It provides a map of Power vs. Time vs. Frequency.
*   **Action:** Implement `waterSpec.wwz`.

### B. Red Noise (AR1) Validation
*   **Why:** To refute the claim "this is just natural persistence."
*   **Action:** Add AR1 generation to the simulation utilities and allow it as a null model in `Analysis`.

---

## 4. "Court of Law" Checklist

| Requirement | Current Status | Proposed Upgrade |
| :--- | :--- | :--- |
| **Reproducibility** | High (Python code, seeds) | No change needed. |
| **Robustness to Gaps** | High (Haar) | **WWZ** for time-localized gap handling. |
| **Artifact Rejection** | Medium (Lomb-Scargle) | **Integrate PSRESP** automatically. |
| **False Positive Control**| Medium (Power Law) | **Add AR1 Red Noise** baseline. |
| **Uncertainty Quantification** | High (Bootstrap) | No change needed. |

## 5. Proposed Roadmap

1.  **Implement `waterSpec.wwz`:** Add Weighted Wavelet Z-Transform for time-localized analysis.
2.  **Enhance `waterSpec.utils_sim`:** Add explicit AR1 (Red Noise) PSD models.
3.  **Upgrade `Analysis.run_full_analysis`:**
    *   Add an option `validate_model=True` that triggers PSRESP.
    *   Include "Success Fraction" (p-value) in the final report.

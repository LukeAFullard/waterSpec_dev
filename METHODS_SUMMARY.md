# waterSpec: Methods Summary

This document summarizes the methods available in the `waterSpec` package, outlining their strengths and weaknesses.

## Primary Spectral Analysis Methods

### 1. Lomb-Scargle Periodogram
**Implementation:** `waterSpec.spectral_analyzer.calculate_periodogram` (uses `astropy.timeseries.LombScargle`)
**Usage:** Used by default in `Analysis` and `SiteComparison` classes.

*   **Description:** Calculates the power spectrum of a time series. It is specifically designed to handle unevenly sampled data by fitting sinusoids of various frequencies.
*   **Strengths:**
    *   **Irregular Sampling:** Can handle unevenly sampled data without interpolation.
    *   **Peak Detection:** Excellent for identifying specific periodicities (peaks) in the data (e.g., diurnal, seasonal cycles).
    *   **Error Handling:** Can incorporate measurement errors (`dy`) into the analysis.
    *   **Standardization:** Well-established method in astronomy and time-series analysis.
*   **Weaknesses:**
    *   **Spectral Slope Bias:** Can produce biased estimates of the power-law spectral slope ($\beta$) if the sampling is highly irregular or has large gaps (spectral leakage).
    *   **Computation:** Can be slower than FFT for very large datasets, though Astropy's implementation is optimized.

### 2. Haar Wavelet Analysis
**Implementation:** `waterSpec.haar_analysis.HaarAnalysis`
**Usage:** Integrated into `Analysis` (via `run_full_analysis(..., run_haar=True)`) or available as a standalone class `HaarAnalysis`.

*   **Description:** Calculates the first-order structure function ($S_1$) of the time series using Haar wavelets. The spectral slope $\beta$ is estimated from the scaling exponent $H$ of the structure function ($\beta = 1 + 2H$).
*   **Strengths:**
    *   **Slope Estimation:** More robust than Lomb-Scargle for estimating the spectral slope ($\beta$) on unevenly sampled data. Less affected by gaps and irregularity.
    *   **Simplicity:** Conceptually simple and direct calculation of fluctuations at different timescales.
*   **Weaknesses:**
    *   **Peak Detection:** Not suitable for detecting specific narrowband periodicities (peaks).

### 3. PSRESP (Power Spectral Response)
**Implementation:** `waterSpec.psresp.psresp_fit`
**Usage:** Available as a standalone function.

*   **Description:** A forward-modeling approach (based on Uttley et al. 2002) that fits a spectral model (e.g., power law) by simulating many time series with known PSDs, degrading them with the observed sampling pattern (window function) and noise, and comparing the simulated periodograms to the observed one.
*   **Strengths:**
    *   **Rigorous Bias Handling:** Explicitly accounts for spectral leakage and redistribution caused by the sampling pattern (window function) and aliasing. This is often the "gold standard" for testing spectral models on complex window functions.
    *   **Goodness of Fit:** Provides a "success fraction" (similar to a p-value) to assess whether the model is consistent with the data.
*   **Weaknesses:**
    *   **Computational Cost:** Extremely expensive compared to other methods, as it requires generating and analyzing thousands of simulated time series.
    *   **Model Dependence:** Requires assuming a spectral model form (e.g., power law) to fit, rather than just estimating a spectrum non-parametrically.

---

## Analysis Workflows

### 4. Automated Spectral Analysis
**Implementation:** `waterSpec.Analysis` class
**Usage:** `Analysis(...).run_full_analysis()`

*   **Description:** A high-level wrapper that automates the entire pipeline: loading data, preprocessing (detrending, censoring), calculating the periodogram (Lomb-Scargle), fitting models, detecting peaks, and interpreting results.
*   **Strengths:**
    *   **Automation:** Drastically reduces code required for a complete analysis.
    *   **Model Selection:** Automatically selects between a standard power-law model and a segmented model (with breakpoints) using the Bayesian Information Criterion (BIC).
    *   **Robustness:** Includes robust uncertainty estimation via bootstrap resampling (block, wild, etc.).
    *   **Diagnostics:** Provides diagnostics for preprocessing and fitting quality.
*   **Weaknesses:**
    *   **Complexity:** Has many configuration parameters that may overwhelm new users.
    *   **Bias Risk:** Relies on Lomb-Scargle, so the spectral slope warning for highly irregular data applies.

### 5. Changepoint Analysis
**Implementation:** `waterSpec.changepoint_detector` and `waterSpec.Analysis`
**Usage:** `Analysis(..., changepoint_mode='auto'|'manual')`

*   **Description:** Detects shifts in the time series mean or variance (using the PELT algorithm) and performs separate spectral analyses on the segments before and after the changepoint.
*   **Strengths:**
    *   **Non-stationarity:** Allows analyzing data that undergoes regime shifts (e.g., before/after an intervention).
    *   **Comparison:** Automatically compares the spectral slopes of the different regimes.
*   **Weaknesses:**
    *   **Data Requirements:** Requires enough data in each segment to perform a valid spectral analysis.
    *   **Dependencies:** Relies on the external `ruptures` library.

### 6. Site Comparison
**Implementation:** `waterSpec.comparison.SiteComparison`
**Usage:** `SiteComparison(...).run_comparison()`

*   **Description:** Runs spectral analysis on two different time series side-by-side and statistically compares their spectral characteristics.
*   **Strengths:**
    *   **Direct Comparison:** Facilitates comparing upstream vs. downstream, or different parameters (e.g., concentration vs. discharge).
    *   **Statistical Rigor:** Provides confidence intervals and interpretation for the differences.
*   **Weaknesses:**
    *   **Alignment:** Assumes the two time series are comparable (e.g., same time units).

---

## Supporting Methods

### 7. Segmented Regression
**Implementation:** `waterSpec.fitter.fit_segmented_spectrum`
**Usage:** Used internally by `Analysis` and `SiteComparison`.

*   **Description:** Fits a piecewise linear model (with 1 or 2 breakpoints) to the log-log power spectrum.
*   **Strengths:**
    *   **Multi-scaling:** Can identify if the system behaves differently at different timescales (e.g., short-term vs. long-term persistence).
*   **Weaknesses:**
    *   **Overfitting:** Can overfit if the data is noisy; mitigated by BIC selection.

### 8. Peak Detection
**Implementation:** `waterSpec.spectral_analyzer`
**Usage:** Used internally by `Analysis`.

*   **Methods:**
    *   **False Alarm Probability (FAP):** Uses `astropy`'s FAP to find peaks significantly above the noise level.
    *   **Residual Outliers:** Detects peaks that are outliers from the fitted power-law model.
*   **Strengths:**
    *   **FAP:** rigorous and independent of the spectral shape.
    *   **Residual:** Good for finding peaks superimposed on a "colored noise" background (power law).
*   **Weaknesses:**
    *   **Residual:** Depends heavily on the quality of the power-law fit.

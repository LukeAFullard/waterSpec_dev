# waterSpec: Validity and Methodology Analysis

**Abstract**
This document provides a rigorous, critical review of the spectral and statistical methodologies implemented in the `waterSpec` package. Written from an academic and senior statistical perspective, it evaluates the mathematical foundations, assumptions, edge cases, and validity boundaries of each method. It is intended for researchers and practitioners who need to defend the use of `waterSpec` in peer-reviewed publications.

---

## 1. Introduction

Environmental time series analysis is fraught with challenges: missing data, irregular sampling, non-stationarity, intermittency, and power-law scaling. The `waterSpec` package attempts to navigate these issues using a combination of traditional spectral methods (Lomb-Scargle) and multiscale approaches (Haar Wavelet Analysis).

However, the application of these methods requires strict adherence to their underlying assumptions. Misapplication can lead to spurious scaling exponents, artificial correlations, or misidentified periodicities. This document questions the validity of each component, detailing when they are robust and when they fail.

---

## 2. Methodology Critique

### 2.1 Lomb-Scargle Periodogram (LS)

**Mathematical Foundation:**
The LS periodogram is effectively a least-squares fit of sinusoids to data. It was derived to handle unequally spaced data in astrophysics.

**Validity & Strengths:**
*   **Peak Detection:** It is statistically robust for detecting deterministic, narrowband periodicities (e.g., diurnal, annual cycles) superimposed on white noise. The False Alarm Probability (FAP) provides a rigorous frequentist significance threshold.
*   **No Interpolation:** By avoiding interpolation, it prevents the artificial introduction of high-frequency noise or smoothing artifacts.

**Weaknesses & Failure Modes (When NOT to use):**
*   **Spectral Slope Bias:** The most critical weakness of LS is its vulnerability to *spectral leakage* when estimating the continuum spectral slope ($\beta$) of red noise processes in highly irregular or gappy data. Energy from low frequencies "leaks" into high frequencies due to the window function (the sampling pattern), flattening the apparent spectrum.
*   **Conclusion:** If the Coefficient of Variation (CV) of the sampling interval is high (> 0.5), or if there are massive gaps (e.g., > 10% of total duration), **do not use LS to estimate $\beta$**. Use Haar Wavelets instead.

### 2.2 Haar Wavelet Analysis (First-Order Structure Function)

**Mathematical Foundation:**
Haar analysis calculates the variance of the difference in means between adjacent non-overlapping (or overlapping) windows of size $\tau$. The scaling exponent $m$ relates to the spectral exponent $\beta$ via $\beta = 2m + 1$.

**Validity & Strengths:**
*   **Robust to Gaps:** Because it operates in the time domain, a gap simply means a specific window is skipped. It does not corrupt the estimates at other scales.
*   **Non-Stationarity:** It inherently handles non-stationary processes (e.g., random walks, $\beta > 1$) better than Fourier methods because the differencing operation acts as a local detrending mechanism.

**Critical Analysis of Overlapping Windows:**
*   `waterSpec` defaults to overlapping windows to increase statistical power (reducing variance of the estimate). However, overlapping windows introduce *autocorrelation* between the fluctuation estimates at a given scale.
*   **Statistical Consequence:** While the mean estimate of $S_1(\tau)$ remains unbiased, the standard error is artificially reduced if standard OLS regression is used to fit the slope.
*   **Mitigation:** `waterSpec` uses block bootstrapping or the Wild bootstrap to estimate confidence intervals on the fit. This is mathematically necessary because standard OLS assumptions are violated.

**Small-Sample Bias Correction:**
*   Standard Haar variance underestimates true variance when the number of data points per window is small. `waterSpec` implements `aggregation="std_corrected"`. This is crucial for high-frequency (small $\tau$) validity.

### 2.3 Multifractal Intermittency Correction ($K(2)$)

**Mathematical Foundation:**
The relationship $\beta = 2m + 1$ assumes the process is monofractal (Gaussian). For intermittent environmental processes (e.g., rainfall, solute flushing), this fails. The Universal Multifractal framework provides a correction: $\beta = 1 + 2H - K(2)$, where $K(2)$ characterizes the intermittency.

**Validity:**
*   This is a highly advanced feature. Estimating $K(2)$ (often via the slope of the 2nd vs 1st order structure functions) requires vast amounts of high-quality data.
*   **Warning:** Applying the $K(2)$ correction to short, noisy time series will likely inject more variance than it removes bias. It should only be used when physical evidence suggests strong intermittency (e.g., storm-event transport mechanisms) and datasets are extensive ($N > 10^4$).

### 2.4 Segmented Spectral Fits

**Mathematical Foundation:**
Using `mannks` or `piecewise-regression`, the package fits broken stick models to the log-log spectrum to identify scales where process dominance shifts.

**Validity & The BIC Criterion:**
*   **Overfitting Risk:** It is trivial to fit a multi-segmented line to a noisy spectrum and lower the Residual Sum of Squares (RSS).
*   **Defense:** The package's reliance on the Bayesian Information Criterion (BIC) to select between a standard line and a segmented line is statistically sound. BIC heavily penalizes additional parameters. If BIC selects a segmented fit, the regime shift is robustly supported by the data.

### 2.5 Bivariate (Cross-Haar) Analysis

**Mathematical Foundation:**
Calculates the Pearson correlation between the Haar fluctuations of two variables at scale $\tau$.

**Validity:**
*   **Scale-Dependent Correlation:** This is a powerful and valid method for decoupling short-term hysteresis from long-term trends.
*   **Assumptions:** It assumes the relationship between the variables at a given scale is linear (Pearson). If the relationship is highly non-linear, Cross-Haar correlation will underestimate the dependency.

### 2.6 Partial Cross-Haar Analysis (Experimental)

**Mathematical Foundation:**
Calculates the partial correlation $\rho_{XY|Z}$ using the linear partial correlation formula applied to the Haar fluctuations of X, Y, and Z.

**Critical Questioning of Validity:**
*   **The Assumption of Multivariate Normality:** The standard partial correlation formula explicitly assumes that the variables (in this case, the *fluctuations* of X, Y, and Z at scale $\tau$) follow a multivariate Gaussian distribution.
*   **The Reality of Environmental Data:** Environmental fluctuations, especially at smaller scales, are notoriously non-Gaussian (heavy-tailed, skewed).
*   **Verdict:** The warning attached to this function in the codebase (`multivariate.py:151`) is entirely justified. While mathematically computable, the *statistical interpretation* of $\rho_{XY|Z}$ as the "true" conditional dependency is weak if the fluctuations are highly non-Gaussian.
*   **Recommendation:** Use only as a qualitative exploratory tool. Do not base definitive causal conclusions solely on this metric without verifying the distributional assumptions of the fluctuations at each scale.

### 2.7 Lomb-Scargle Cross-Spectrum (Phase Lag)

**Mathematical Foundation:**
Extends LS to two variables to find the phase difference (lead/lag) at specific frequencies.

**Validity:**
*   **Noise Sensitivity:** Phase estimation is highly sensitive to noise. If the Cross-Spectral Power (Coherence) is low at a given frequency, the estimated phase lag is meaningless (essentially a random variable uniform on $[-\pi, \pi]$).
*   **Interpretation:** Only interpret phase lags at frequencies where both variables exhibit significant power and high coherence.

---

## 3. Discussion: Methodological Synergies and Limitations

**The Pre-processing Dilemma:**
`waterSpec` includes tools for linear and LOESS detrending.
*   *The Trap:* Detrending removes low-frequency power. If you are analyzing a non-stationary process (e.g., groundwater levels), detrending before spectral analysis will artificially flatten the spectrum at large scales, destroying the very information you are trying to measure ($\beta > 1$).
*   *Rule of Thumb:* Only detrend if you are strictly interested in the stationary fluctuations around a known, deterministic trend (e.g., climate change warming curve), and you are prepared to ignore the largest scales.

**Surrogate Data Testing:**
The package uses phase-randomized surrogates.
*   *Validity:* This perfectly preserves the linear autocorrelation structure (the power spectrum) while destroying non-linearities and phase relationships. It is the gold standard null model for testing the significance of peaks or Cross-Haar correlations against a red-noise background.
*   *Limitation:* Phase randomization of highly irregular data with large gaps (via interpolation/FFT/inverse-interpolation) can introduce artifacts. `waterSpec` correctly issues warnings when generating surrogates for data with large gaps.

---

## 4. Conclusion

The `waterSpec` package implements statistically rigorous methods, but provides enough rope for a careless user to hang themselves.

1.  **Lomb-Scargle** is for finding periodic peaks in uneven data, not for slope estimation in highly gappy data.
2.  **Haar Analysis** is the superior tool for scaling exponents ($\beta$) in irregular data.
3.  **BIC model selection** protects against overfitting breakpoints.
4.  **Partial Cross-Haar** must be treated with extreme skepticism due to its reliance on Gaussian assumptions applied to potentially non-Gaussian fluctuations.

Researchers must justify their method choices (LS vs Haar) based on the sampling irregularity and the specific scientific question (peaks vs slopes), and respect the boundaries established by the implemented warnings.

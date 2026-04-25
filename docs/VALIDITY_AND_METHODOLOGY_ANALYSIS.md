# waterSpec: Validity and Methodology Analysis

**Abstract**
This document provides a rigorous, critical review of the spectral and statistical methodologies implemented in the `waterSpec` package. Written from an academic and senior statistical perspective, it evaluates the mathematical foundations, assumptions, edge cases, and validity boundaries of each method. It is intended for researchers and practitioners who need to defend the use of `waterSpec` in peer-reviewed publications.

---

## 1. Introduction

Environmental time series analysis is fraught with challenges: missing data, irregular sampling, non-stationarity, intermittency, and power-law scaling. The `waterSpec` package attempts to navigate these issues using a combination of traditional spectral methods (Lomb-Scargle) and multiscale approaches (Haar Wavelet Analysis).

However, the application of these methods requires strict adherence to their underlying assumptions. Misapplication can lead to spurious scaling exponents, artificial correlations, or misidentified periodicities. This document questions the validity of each component, detailing when they are robust and when they fail, and provides key literature references justifying these claims.

---

## 2. Methodology Critique

### 2.1 Lomb-Scargle Periodogram (LS)

**Mathematical Foundation:**
The LS periodogram is effectively a least-squares fit of sinusoids to data. It was derived to handle unequally spaced data in astrophysics (Lomb, 1976; Scargle, 1982).

**Validity & Strengths:**
*   **Peak Detection:** It is statistically robust for detecting deterministic, narrowband periodicities (e.g., diurnal, annual cycles) superimposed on white noise. The False Alarm Probability (FAP) provides a rigorous frequentist significance threshold, utilizing extreme value statistics to account for multiple independent frequencies. VanderPlas (2018) provides an extensive review of its validity for periodic detection, and Baluev (2008) details the exact extreme value distributions necessary for precise FAP computation.
*   **No Interpolation:** By avoiding interpolation, it prevents the artificial introduction of high-frequency noise or smoothing artifacts.

**Weaknesses & Failure Modes (When NOT to use):**
*   **Spectral Slope Bias:** The most critical weakness of LS is its vulnerability to *spectral leakage* when estimating the continuum spectral slope ($\beta$) of red noise processes in highly irregular or gappy data. Energy from low frequencies "leaks" into high frequencies due to the window function (the sampling pattern), flattening the apparent spectrum.
*   **Conclusion:** If the Coefficient of Variation (CV) of the sampling interval is high (> 0.5), or if there are massive gaps (e.g., > 10% of total duration), **do not use LS to estimate $\beta$**. Use Haar Wavelets instead.

**References:**
*   Baluev, R. V. (2008). Assessing the statistical significance of periodogram peaks. *Monthly Notices of the Royal Astronomical Society*, 385(3), 1279-1285.
*   Lomb, N. R. (1976). Least-squares frequency analysis of unequally spaced data. *Astrophysics and Space Science*, 39, 447-462.
*   Scargle, J. D. (1982). Studies in astronomical time series analysis. II - Statistical aspects of spectral analysis of unevenly spaced data. *The Astrophysical Journal*, 263, 835-853.
*   VanderPlas, J. T. (2018). Understanding the Lomb-Scargle Periodogram. *The Astrophysical Journal Supplement Series*, 236(1), 16.

### 2.2 Haar Wavelet Analysis (First-Order Structure Function)

**Mathematical Foundation:**
Haar analysis calculates the variance of the difference in means between adjacent non-overlapping (or overlapping) windows of size $\tau$. The scaling exponent $m$ relates to the spectral exponent $\beta$ via $\beta = 2m + 1$ (Lovejoy & Schertzer, 2012).

**Validity & Strengths:**
*   **Robust to Gaps:** Because it operates in the time domain, a gap simply means a specific window is skipped. It does not corrupt the estimates at other scales.
*   **Non-Stationarity:** It inherently handles non-stationary processes (e.g., random walks, $\beta > 1$) better than Fourier methods because the differencing operation acts as a local detrending mechanism.

**Critical Analysis of Overlapping Windows:**
*   `waterSpec` defaults to overlapping windows to increase statistical power (reducing variance of the estimate), analogous to the Maximum Overlap Discrete Wavelet Transform (MODWT). However, overlapping windows introduce *autocorrelation* between the fluctuation estimates at a given scale.
*   **Statistical Consequence:** While the mean estimate of $S_1(\tau)$ remains unbiased, the standard error is artificially reduced if standard OLS regression is used to fit the slope, as the degrees of freedom are fewer than the number of overlapping windows. Percival (1995) details the variance properties of the overlapping Haar wavelet variance.
*   **Mitigation:** `waterSpec` uses block bootstrapping or the Wild bootstrap to estimate confidence intervals on the fit. This is mathematically necessary because standard OLS assumptions are violated.

**Small-Sample Bias Correction:**
*   Standard Haar variance underestimates true variance when the number of data points per window is small. `waterSpec` implements `aggregation="std_corrected"`. This is crucial for high-frequency (small $\tau$) validity.

**References:**
*   Lovejoy, S., & Schertzer, D. (2012). *The Weather and Climate: Emergent Laws and Multifractal Cascades*. Cambridge University Press.
*   Percival, D. P. (1995). On estimation of the wavelet variance. *Biometrika*, 82(3), 619-631.

### 2.3 Multifractal Intermittency Correction ($K(2)$)

**Mathematical Foundation:**
The relationship $\beta = 2m + 1$ assumes the process is monofractal (Gaussian). For intermittent environmental processes (e.g., rainfall, solute flushing), this fails. The Universal Multifractal framework provides a correction: $\beta = 1 + 2H - K(2)$, where $K(2)$ characterizes the intermittency (Schertzer & Lovejoy, 1987).

**Validity:**
*   This is a highly advanced feature. Estimating $K(2)$ (often via the slope of the 2nd vs 1st order structure functions) requires vast amounts of high-quality data.
*   **Warning:** Applying the $K(2)$ correction to short, noisy time series will likely inject more variance than it removes bias. It should only be used when physical evidence suggests strong intermittency (e.g., storm-event transport mechanisms) and datasets are extensive ($N > 10^4$).

**References:**
*   Schertzer, D., & Lovejoy, S. (1987). Physical modeling and analysis of rain and clouds by anisotropic scaling multiplicative processes. *Journal of Geophysical Research: Atmospheres*, 92(D8), 9693-9714.

### 2.4 Segmented Spectral Fits

**Mathematical Foundation:**
Using `mannks` or `piecewise-regression`, the package fits broken stick models to the log-log spectrum to identify scales where process dominance shifts (Toms & Lesperance, 2003).

**Validity & The BIC Criterion:**
*   **Overfitting Risk:** It is trivial to fit a multi-segmented line to a noisy spectrum and lower the Residual Sum of Squares (RSS).
*   **Defense:** The package's reliance on the Bayesian Information Criterion (BIC) to select between a standard line and a segmented line is statistically sound. BIC heavily penalizes additional parameters (Schwarz, 1978). If BIC selects a segmented fit, the regime shift is robustly supported by the data.

**References:**
*   Schwarz, G. (1978). Estimating the dimension of a model. *The Annals of Statistics*, 6(2), 461-464.
*   Toms, P. S., & Lesperance, M. L. (2003). Piecewise regression: a tool for identifying ecological thresholds. *Ecology*, 84(8), 2034-2041.

### 2.5 Bivariate (Cross-Haar) Analysis

**Mathematical Foundation:**
Calculates the Pearson correlation between the Haar fluctuations of two variables at scale $\tau$.

**Validity:**
*   **Scale-Dependent Correlation:** This is a powerful and valid method for decoupling short-term hysteresis from long-term trends. It serves as a time-domain analog to Cross-Wavelet Transform (XWT) and Wavelet Coherence approaches (Grinsted et al., 2004), without requiring continuous data interpolation.
*   **Assumptions:** It assumes the relationship between the variables at a given scale is linear (Pearson). If the relationship is highly non-linear, Cross-Haar correlation will underestimate the dependency.

**References:**
*   Grinsted, A., Moore, J. C., & Jevrejeva, S. (2004). Application of the cross wavelet transform and wavelet coherence to geophysical time series. *Nonlinear processes in geophysics*, 11(5/6), 561-566.
*   Torrence, C., & Compo, G. P. (1998). A practical guide to wavelet analysis. *Bulletin of the American Meteorological society*, 79(1), 61-78.

### 2.6 Partial Cross-Haar Analysis (Experimental)

**Mathematical Foundation:**
Calculates the partial correlation $\rho_{XY|Z}$ using the linear partial correlation formula applied to the Haar fluctuations of X, Y, and Z.

**Critical Questioning of Validity:**
*   **The Assumption of Multivariate Normality:** The standard partial correlation formula explicitly assumes that the variables (in this case, the *fluctuations* of X, Y, and Z at scale $\tau$) follow a multivariate Gaussian distribution.
*   **The Reality of Environmental Data:** Environmental fluctuations, especially at smaller scales, are notoriously non-Gaussian (heavy-tailed, skewed).
**Verdict:** The warning attached to this function in the codebase is entirely justified. While mathematically computable, the *statistical interpretation* of $\rho_{XY|Z}$ as the "true" conditional dependency is weak if the fluctuations are highly non-Gaussian. Methodologically, this concept draws from Partial Wavelet Coherence (Ng & Chan, 2012), which is generally formulated for continuous frequency domains rather than discrete temporal structures.
*   **Recommendation:** Use only as a qualitative exploratory tool. Do not base definitive causal conclusions solely on this metric without verifying the distributional assumptions of the fluctuations at each scale.

**References:**
*   Ng, E. K., & Chan, J. C. (2012). Geophysical applications of partial wavelet coherence and multiple wavelet coherence. *Journal of Atmospheric and Oceanic Technology*, 29(12), 1845-1853.

### 2.7 Lomb-Scargle Cross-Spectrum (Phase Lag)

**Mathematical Foundation:**
Extends LS to two variables to find the phase difference (lead/lag) at specific frequencies.

**Validity:**
*   **Noise Sensitivity:** Phase estimation is highly sensitive to noise. If the Cross-Spectral Power (Coherence) is low at a given frequency, the estimated phase lag is meaningless (essentially a random variable uniform on $[-\pi, \pi]$).
*   **Interpretation:** Only interpret phase lags at frequencies where both variables exhibit significant power and high coherence.

**References:**
*   Hocke, K. (1998). Phase estimation with the Lomb-Scargle periodogram method. *Annales Geophysicae*, 16(3), 356-358.

---

## 3. Discussion: Methodological Synergies and Limitations

**The Pre-processing Dilemma:**
`waterSpec` includes tools for linear and LOESS detrending.
*   *The Trap:* Detrending removes low-frequency power. If you are analyzing a non-stationary process (e.g., groundwater levels), detrending before spectral analysis will artificially flatten the spectrum at large scales, destroying the very information you are trying to measure ($\beta > 1$). This effect is akin to applying a high-pass filter, altering the scaling behavior at low frequencies.
*   *Rule of Thumb:* Only detrend if you are strictly interested in the stationary fluctuations around a known, deterministic trend (e.g., climate change warming curve), and you are prepared to ignore the largest scales where the trend dominates.

**Surrogate Data Testing:**
The package uses phase-randomized surrogates.
*   *Validity:* This perfectly preserves the linear autocorrelation structure (the power spectrum) while destroying non-linearities and phase relationships. It is the gold standard null model for testing the significance of peaks or Cross-Haar correlations against a red-noise background (Theiler et al., 1992). For multivariate applications (e.g., Cross-Haar), preserving the individual auto-spectra while randomizing cross-dependencies is theoretically robust under the Prichard & Theiler (1994) framework for multivariate surrogates.
*   *Limitation:* Phase randomization of highly irregular data with large gaps (via interpolation/FFT/inverse-interpolation) can introduce artifacts. `waterSpec` correctly issues warnings when generating surrogates for data with large gaps.

**References:**
*   Prichard, D., & Theiler, J. (1994). Generating surrogate data for time series with several simultaneously measured variables. *Physical review letters*, 73(7), 951.
*   Theiler, J., Eubank, S., Longtin, A., Galdrikian, B., & Farmer, J. D. (1992). Testing for nonlinearity in time series: the method of surrogate data. *Physica D: Nonlinear Phenomena*, 58(1-4), 77-94.

---

## 4. Conclusion

The `waterSpec` package implements statistically rigorous methods, but provides enough rope for a careless user to hang themselves.

1.  **Lomb-Scargle** is for finding periodic peaks in uneven data, not for slope estimation in highly gappy data.
2.  **Haar Analysis** is the superior tool for scaling exponents ($\beta$) in irregular data.
3.  **BIC model selection** protects against overfitting breakpoints.
4.  **Partial Cross-Haar** must be treated with extreme skepticism due to its reliance on Gaussian assumptions applied to potentially non-Gaussian fluctuations.

Researchers must justify their method choices (LS vs Haar) based on the sampling irregularity and the specific scientific question (peaks vs slopes), and respect the boundaries established by the implemented warnings.

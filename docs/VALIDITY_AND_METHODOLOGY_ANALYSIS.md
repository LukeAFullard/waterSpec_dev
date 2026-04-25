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
The LS periodogram is effectively a least-squares fit of sinusoids to data. It was derived to handle unequally spaced data in astrophysics (Lomb, 1976; Scargle, 1982). For a time series $x(t_i)$ evaluated at arbitrary times $t_i$, the normalized periodogram at angular frequency $\omega = 2\pi f$ is defined as:

$$ P_N(\omega) = \frac{1}{2\sigma^2} \left\{ \frac{\left[\sum_i (x_i - \bar{x}) \cos \omega(t_i - \tau)\right]^2}{\sum_i \cos^2 \omega(t_i - \tau)} + \frac{\left[\sum_i (x_i - \bar{x}) \sin \omega(t_i - \tau)\right]^2}{\sum_i \sin^2 \omega(t_i - \tau)} \right\} $$

where $\tau$ is a frequency-dependent time offset specified by:
$$ \tan(2\omega\tau) = \frac{\sum_i \sin(2\omega t_i)}{\sum_i \cos(2\omega t_i)} $$
This specific choice of $\tau$ makes the periodogram identical to a least-squares fit of the model $x(t) = A \cos(\omega t) + B \sin(\omega t)$ to the data.

**Validity & Strengths:**
*   **Peak Detection:** It is statistically robust for detecting deterministic, narrowband periodicities (e.g., diurnal, annual cycles) superimposed on white noise. The False Alarm Probability (FAP) provides a rigorous frequentist significance threshold, utilizing extreme value statistics to account for multiple independent frequencies. VanderPlas (2018) provides an extensive review of its validity for periodic detection, and Baluev (2008) details the exact extreme value distributions necessary for precise FAP computation.
*   **No Interpolation:** By avoiding interpolation, it prevents the artificial introduction of high-frequency noise or smoothing artifacts.

**Weaknesses & Failure Modes (When NOT to use):**
*   **Bootstrap Performance:** `waterSpec` allows using `fap_method="bootstrap"` for empirical FAP estimation via the `find_significant_peaks` function. However, this is computationally expensive. The `find_significant_peaks` implementation explicitly emits a performance `UserWarning` when this method is selected, and a secondary warning if it detects more than 5 peaks, as the bootstrap algorithm's slow execution time scales poorly with multiple peak extraction.
*   **Spectral Slope Bias:** The most critical weakness of LS is its vulnerability to *spectral leakage* when estimating the continuum spectral slope ($\beta$) of red noise processes in highly irregular or gappy data. Energy from low frequencies "leaks" into high frequencies due to the window function (the sampling pattern), flattening the apparent spectrum.
*   **Aliasing and the Spectral Window:** Uneven sampling does not completely eliminate aliasing; it merely redistributes aliased power into a complex, continuous background. The "spectral window function" (the Fourier transform of the sampling times) dictates how true peaks are convolved and where "ghost" peaks appear. Highly periodic gaps (e.g., missing weekend data, diurnal gaps) create strong aliases that mimic true physical signals (VanderPlas, 2018).
*   **Conclusion:** If the Coefficient of Variation (CV) of the sampling interval is high (> 0.5), or if there are massive gaps (e.g., > 10% of total duration), **do not use LS to estimate $\beta$**. Use Haar Wavelets instead.
**References:**
*   Baluev, R. V. (2008). Assessing the statistical significance of periodogram peaks. *Monthly Notices of the Royal Astronomical Society*, 385(3), 1279-1285.
*   Lomb, N. R. (1976). Least-squares frequency analysis of unequally spaced data. *Astrophysics and Space Science*, 39, 447-462.
*   Scargle, J. D. (1982). Studies in astronomical time series analysis. II - Statistical aspects of spectral analysis of unevenly spaced data. *The Astrophysical Journal*, 263, 835-853.
*   VanderPlas, J. T. (2018). Understanding the Lomb-Scargle Periodogram. *The Astrophysical Journal Supplement Series*, 236(1), 16.

### 2.2 Haar Wavelet Analysis (First-Order Structure Function)

**Mathematical Foundation:**
Haar analysis calculates the variance of the difference in means between adjacent non-overlapping (or overlapping) windows of size $\tau$. The first-order structure function $S_1(\tau)$ is computed as the variance of the fluctuation $\Delta x(\tau, t)$:

$$ \Delta x(\tau, t) = \langle x(u) \rangle_{t < u < t+\tau} - \langle x(u) \rangle_{t-\tau < u < t} $$

where $\langle \cdot \rangle$ denotes the mean over the specified time interval. The scaling exponent $m$ is derived from the relation:
$$ S_1(\tau) \propto \tau^m $$
This exponent $m$ relates to the spectral exponent $\beta$ via the equation $\beta = 2m + 1$ (Lovejoy & Schertzer, 2012).

**Validity & Strengths:**
*   **Robust to Gaps:** Because it operates in the time domain, a gap simply means a specific window is skipped. It does not corrupt the estimates at other scales.
*   **Non-Stationarity:** It inherently handles non-stationary processes (e.g., random walks, $\beta > 1$) better than Fourier methods because the differencing operation acts as a local detrending mechanism.

**Critical Analysis of Overlapping Windows:**
*   `waterSpec` defaults to overlapping windows to increase statistical power (reducing variance of the estimate), analogous to the Maximum Overlap Discrete Wavelet Transform (MODWT). However, overlapping windows introduce *autocorrelation* between the fluctuation estimates at a given scale.
*   **Statistical Consequence:** While the mean estimate of $S_1(\tau)$ remains unbiased, the standard error is artificially reduced if standard OLS regression is used to fit the slope, as the true degrees of freedom are much fewer than the number of overlapping windows. Percival (1995) details the variance properties of the overlapping Haar wavelet variance, demonstrating how naive OLS drastically underestimates CI bounds.
*   **Mitigation:** `waterSpec` uses moving block bootstrapping (via indices) or parametric Monte Carlo surrogates to estimate confidence intervals on the fit. This is mathematically necessary because standard OLS assumptions are violated. Furthermore, robust Theil-Sen regression via `MannKS.trend_test` ensures that outlier fluctuation scales do not completely bias the spectral exponent fit.

**Small-Sample Bias Correction:**
*   Standard Haar variance underestimates true variance when the number of data points per window is small. `waterSpec` implements an explicitly unbiased standard deviation estimator (`std_corrected`) utilizing correction factors derived from Gamma functions under the assumption of local normality: $\sigma_{unbiased} = \sigma_{sample} \cdot \sqrt{\frac{2}{N-1}} \exp(\Gamma(\frac{N}{2}) - \Gamma(\frac{N-1}{2}))$. This is crucial for high-frequency (small $\tau$) validity where point density per window drops, guaranteeing slopes remain undistorted.

**Custom Statistics (Percentiles & Medians):**
*   `waterSpec` allows evaluating fluctuations using custom statistics like percentiles (e.g., 95th) instead of means. While useful for examining the scaling of extremes, standard scaling relations ($\beta = 2m + 1$) are explicitly derived for variances (or mean-squared fluctuations). The theoretical translation of percentile-based slopes to traditional spectral $\beta$ is not firmly established in linear spectral theory and should be treated as an empirical scaling index.

**Aggregation Methods for the Structure Function ($S_1$):**
*   **Mean & Median:** Aggregating window fluctuations via the "mean" of the absolute differences provides a robust and distribution-agnostic standard $S_1$. "Median" acts as an even more robust metric against single extreme fluctuation outliers.
*   **Root Mean Square (RMS):** Approximates $\sqrt{S_2}$, connecting the first-order analysis directly to the second moment necessary for exploring multifractal scaling features (e.g., estimating intermittency $K(2)$).
*   **Small-Sample Unbiased Estimator (`std_corrected`):** When sample sizes per window drop below large-N asymptotic thresholds (often $< 100$ points per window), sample variance metrics systematically underestimate the true population variance. `waterSpec` provides `std_corrected`, an aggregation strategy explicitly designed to combat this. Assuming the underlying sequence of fluctuations approaches Gaussianity via the Central Limit Theorem, it computes standard deviations with small sample bias correction factors derived from exact Gamma functions (`gammaln`). This perfectly matches the bias-correction approach of external robust toolkits like GapWaveSpectra, ensuring that high-frequency scaling slopes remain undistorted by point-density drop-offs at small temporal scales $\tau$.

**Edge Effects (Cone of Influence):**
*   Similar to the Continuous Wavelet Transform (CWT), Haar analysis suffers from edge effects near the beginning and end of the time series where windows are truncated or data is sparse. This creates a "Cone of Influence" (COI). Interpretations of long-scale fluctuations near the series boundaries must be treated with caution, as they are calculated from artificially shortened effective window lengths.

**References:**
*   Lovejoy, S., & Schertzer, D. (2012). *The Weather and Climate: Emergent Laws and Multifractal Cascades*. Cambridge University Press.
*   Percival, D. P. (1995). On estimation of the wavelet variance. *Biometrika*, 82(3), 619-631.
### 2.3 Multifractal Intermittency Correction ($K(2)$)

**Mathematical Foundation:**
The relationship $\beta = 2m + 1$ assumes the process is monofractal (Gaussian). For intermittent environmental processes (e.g., rainfall, solute flushing), this fails. The Universal Multifractal framework provides a correction:

$$ \beta = 1 + 2H - K(2) $$

where $H$ is the Hurst exponent (related to the first-order mean fluctuation) and $K(2)$ characterizes the intermittency. $K(q)$ is the scaling moment function such that the moments of the fluctuations scale as $\langle (\Delta x)^q \rangle \propto \tau^{qH - K(q)}$. For the second moment (variance or power spectrum), $K(2)$ quantifies the deviation from simple monofractal scaling due to intermittent bursts of variance.

**Validity:**
*   This is a highly advanced feature. Estimating $K(2)$ (often via the slope of the 2nd vs 1st order structure functions) requires vast amounts of high-quality data.
*   **Warning:** Applying the $K(2)$ correction to short, noisy time series will likely inject more variance than it removes bias. It should only be used when physical evidence suggests strong intermittency (e.g., storm-event transport mechanisms) and datasets are extensive ($N > 10^4$).

**References:**
*   Schertzer, D., & Lovejoy, S. (1987). Physical modeling and analysis of rain and clouds by anisotropic scaling multiplicative processes. *Journal of Geophysical Research: Atmospheres*, 92(D8), 9693-9714.

### 2.4 Segmented Spectral Fits

**Mathematical Foundation:**
Using `mannks` or `piecewise-regression`, the package fits broken stick models to the log-log spectrum to identify scales where process dominance shifts (Toms & Lesperance, 2003). The codebase leverages the robust `MannKS` package (`MannKS.segmented_trend_test`) to calculate breakpoints and segments, inherently utilizing block bootstrapping to preserve the spectral autocorrelation structure when determining confidence bounds. To ensure reproducibility, `fit_standard_model` and `fit_segmented_spectrum` extract an integer `mannks_seed` from various `seed` types (including `np.random.Generator` using `seed.integers()`) to explicitly pass to `MannKS` via the `random_state` argument. Furthermore, when using the `theil-sen` method, `fit_standard_model` gracefully falls back to `scipy.stats.theilslopes` if `MannKS.trend_test` fails.

**Validity & The BIC Criterion:**
*   **Overfitting Risk:** It is trivial to fit a multi-segmented line to a noisy spectrum and lower the Residual Sum of Squares (RSS).
*   **Defense:** The package's reliance on the Bayesian Information Criterion (BIC) to select between a standard line and a segmented line is statistically sound. BIC heavily penalizes additional parameters via the formulation: $BIC = n \ln(RSS/n) + k \ln(n)$, where $n$ is the number of data points and $k$ is the number of parameters. If BIC selects a segmented fit, the regime shift is robustly supported by the data. The internal `_calculate_bic` routine correctly traps perfectly overfitted "zero RSS" models ($RSS < 10^{-12}$), returning $BIC = \infty$ and emitting a `UserWarning`, effectively banning artificial segments driven by numerical artifacts. This prevents the algorithm from selecting mathematically degenerate piecewise models.

**References:**
*   Schwarz, G. (1978). Estimating the dimension of a model. *The Annals of Statistics*, 6(2), 461-464.
*   Toms, P. S., & Lesperance, M. L. (2003). Piecewise regression: a tool for identifying ecological thresholds. *Ecology*, 84(8), 2034-2041.
### 2.5 Bivariate (Cross-Haar) Analysis

**Mathematical Foundation:**
Calculates the Pearson correlation between the Haar fluctuations of two variables, $X$ and $Y$, at scale $\tau$. Let $\Delta x(\tau, t)$ and $\Delta y(\tau, t)$ be the fluctuations computed at overlapping or non-overlapping time windows of length $\tau$. The Cross-Haar correlation is:

$$ \rho_{XY}(\tau) = \frac{\sum_t (\Delta x(\tau, t) - \overline{\Delta x})(\Delta y(\tau, t) - \overline{\Delta y})}{\sqrt{\sum_t (\Delta x(\tau, t) - \overline{\Delta x})^2 \sum_t (\Delta y(\tau, t) - \overline{\Delta y})^2}} $$

**Validity & Interpretation:**
*   **Scale-Dependent Correlation:** This is a powerful and valid method for decoupling short-term hysteresis from long-term trends. It serves as a time-domain analog to Cross-Wavelet Transform (XWT) and Wavelet Coherence approaches (Grinsted et al., 2004), without requiring continuous data interpolation.
*   **Lead/Lag and Phase Dynamics:** Unlike complex wavelets, Cross-Haar only computes real Pearson correlations (effectively $0$ or $\pi$ phase shifts, representing positive or negative correlations). If two signals have a persistent orthogonal phase shift (e.g., $\pi/2$, a quarter-cycle lag), the Cross-Haar correlation will tend toward zero, failing to capture the causal dependency. Bivariate Haar is strictly for *in-phase* or *anti-phase* scale-dependent relationships.
*   **Assumptions:** It assumes the relationship between the variables at a given scale is linear (Pearson). If the relationship is highly non-linear, Cross-Haar correlation will underestimate the dependency.

**References:**
*   Grinsted, A., Moore, J. C., & Jevrejeva, S. (2004). Application of the cross wavelet transform and wavelet coherence to geophysical time series. *Nonlinear processes in geophysics*, 11(5/6), 561-566.
*   Torrence, C., & Compo, G. P. (1998). A practical guide to wavelet analysis. *Bulletin of the American Meteorological society*, 79(1), 61-78.

### 2.6 Hysteresis Classification within Bivariate Analysis

**Mathematical Foundation:**
Extends bivariate analysis by quantifying the loop area and direction (clockwise vs. counter-clockwise) in the phase space of Haar fluctuations for two variables at a specific scale $\tau$. It utilizes the shoelace formula to compute the signed polygon area formed by the sequential fluctuation pairs $(X_i, Y_i) = (\Delta x(\tau, t_i), \Delta y(\tau, t_i))$ at the chosen scale:

$$ \text{Area}(\tau) = \frac{1}{2} \sum_{i=1}^{n-1} (X_i Y_{i+1} - X_{i+1} Y_i) $$

**Validity & Interpretation:**
*   **Scale-Specific Hysteresis:** Traditional hysteresis analysis (e.g., C-Q loops during storms) is often confounded by long-term baseline shifts. By isolating fluctuations at scale $\tau$, this method provides a mathematically rigorous way to evaluate hysteresis generated strictly by processes operating at that specific timescale (e.g., event-based flushing), decoupling it from seasonal or inter-annual trends.
*   **Loop Area Significance:** The area of the loop quantifies the magnitude of the hysteresis (the degree to which the relationship depends on the trajectory, or "memory", of the system). The shoelace formula (signed polygon area) calculation elegantly captures both the overall magnitude of the deviation from linearity and the prevailing temporal ordering of the events at that scale.
*   **Directionality:** The sign of the area indicates the direction. A clockwise loop (often implying the source is rapidly depleted or proximal) is distinct from a counter-clockwise loop (often implying a delayed or distal source).
*   **Limitations:** The metric is sensitive to noise at small scales and requires overlapping windows to adequately resolve the shape of the loop in phase space. If the chosen scale $\tau$ does not match the actual physical timescale of the hysteresis-generating event, the computed area will be near zero or uninterpretable.

**References:**
*   Lloyd, C. E., Freer, J. E., Johnes, P. J., & Collins, A. L. (2016). Technical Note: Testing an improved index for analysing storm discharge-concentration hysteresis. *Hydrology and Earth System Sciences*, 20(2), 625-632.
*   Zuecco, G., Penna, D., Borga, M., & van Meerveld, H. J. (2016). A versatile index to characterize hysteresis between hydrological variables at the runoff event timescale. *Hydrological Processes*, 30(9), 1449-1466.
### 2.7 Partial Cross-Haar Analysis (Experimental)

**Mathematical Foundation:**
Calculates the partial correlation $\rho_{XY|Z}$ using the linear partial correlation formula applied to the Haar fluctuations of X, Y, and Z.

**Critical Questioning of Validity:**
*   **The Assumption of Multivariate Normality:** The standard partial correlation formula explicitly assumes that the variables (in this case, the *fluctuations* of X, Y, and Z at scale $\tau$) follow a multivariate Gaussian distribution.
*   **The Reality of Environmental Data:** Environmental fluctuations, especially at smaller scales, are notoriously non-Gaussian (heavy-tailed, skewed).
**Verdict:** The warning attached to this function in the codebase is entirely justified. While mathematically computable, the *statistical interpretation* of $\rho_{XY|Z}$ as the "true" conditional dependency is weak if the fluctuations are highly non-Gaussian. Methodologically, this concept draws from Partial Wavelet Coherence (Ng & Chan, 2012), which is generally formulated for continuous frequency domains rather than discrete temporal structures.
*   **Recommendation:** Use only as a qualitative exploratory tool. Do not base definitive causal conclusions solely on this metric without verifying the distributional assumptions of the fluctuations at each scale.

**References:**
*   Ng, E. K., & Chan, J. C. (2012). Geophysical applications of partial wavelet coherence and multiple wavelet coherence. *Journal of Atmospheric and Oceanic Technology*, 29(12), 1845-1853.

### 2.8 Lomb-Scargle Cross-Spectrum (Phase Lag)

**Mathematical Foundation:**
Extends LS to two variables to find the phase difference (lead/lag) at specific frequencies.

**Validity & Limitations:**
*   **Computational Rigor and Cache Optimization:** Under the hood, `calculate_ls_cross_spectrum` avoids iterating frequency-by-frequency in pure Python, which destroys cache locality. Instead, it dynamically batches frequencies—targeting ~2MB payload chunks explicitly aligned to maximize CPU L3 cache hits—to perform vectorized block `np.linalg.solve` routines over multidimensional frequency arrays.
*   **Mathematical Identity Shortcuts:** The batch processing loop further optimizes scaling performance by computing the $\omega$ array per batch (`2 * np.pi * f_batch[:, np.newaxis]`) outside the loop and slicing it. Inside the batch, trigonometric sum evaluations are dramatically reduced using exact mathematical identities (e.g., `Swss = sum_w - Swcc`). This circumvents redundant floating-point processing overhead, preserving full mathematical exactness while rendering Lomb-Scargle Cross-Spectrum computationally tractable for massive, highly irregular temporal sequences without encountering Out-Of-Memory (OOM) halts.
*   **Pre-Calculated Boundary Searches:** Beyond Lomb-Scargle, scaling and windowing functions across the `waterSpec` package (including `calculate_sliding_haar`, `bivariate.py`, and `multivariate.py`) utilize strict boolean masking bounds (e.g., `t_starts[t_starts + window_size <= time[-1]]`) alongside vectorized `np.searchsorted`. Pre-calculating window boundaries ensures zero iterative Python function call overhead when fetching index blocks for complex mathematical evaluations and prevents exceeding array bounds due to floating point inaccuracies.
*   **Noise Sensitivity and Coherence Thresholding:** Phase estimation is highly sensitive to noise. If the Cross-Spectral Power (Coherence) is low at a given frequency, the estimated phase lag is meaningless (essentially a random variable uniform on $[-\pi, \pi]$). A rigorous statistical threshold for coherence must be established (e.g., via Monte Carlo surrogates) before interpreting phase lags.
*   **Interpretation of Phase Wraparound:** Phase is circular (defined modulo $2\pi$). Interpreting a phase difference as a definitive time lag (e.g., $\Delta t = \Delta\phi / (2\pi f)$) is ambiguous without prior physical constraints on causality, as a lag of $\Delta\phi$ is indistinguishable from a lead of $2\pi - \Delta\phi$.
*   **Conclusion:** Only interpret phase lags at frequencies where both variables exhibit significant, localized power above a red-noise background, and the cross-coherence exceeds a strict surrogate-derived threshold.

**References:**
*   Hocke, K. (1998). Phase estimation with the Lomb-Scargle periodogram method. *Annales Geophysicae*, 16(3), 356-358.
### 2.9 Changepoint Detection (PELT Algorithm)

**Mathematical Foundation:**
The package utilizes the Pruned Exact Linear Time (PELT) algorithm via the `ruptures` library to detect shifts in the mean or variance of a time series. The algorithm seeks to minimize a penalized cost function:

$$ \min_{m, \tau_1, \dots, \tau_m} \left[ \sum_{i=1}^{m+1} C(y_{\tau_{i-1}:\tau_i}) + \beta m \right] $$

where $C$ is a cost function (e.g., Gaussian log-likelihood or $L_2$ norm) evaluating the fit of the segment $y_{\tau_{i-1}:\tau_i}$, $m$ is the number of changepoints, $\tau_i$ are the changepoint indices, and $\beta$ is the penalty parameter to guard against overfitting.

**Validity & Limitations:**
*   **Algorithmic Efficiency:** PELT is mathematically exact for finding the global minimum of the penalized cost function and operates efficiently even on large datasets (Killick et al., 2012). It achieves $O(N)$ computational time under the assumption that the number of true changepoints increases linearly with $N$.
*   **Penalty Selection (AIC vs. BIC):** The number of detected changepoints is extremely sensitive to the chosen penalty factor ($\beta$). `waterSpec` typically utilizes a penalty mathematically akin to BIC ($p \log(n)$), which heavily penalizes complexity and favors fewer, more statistically profound regime shifts. Using an AIC-like penalty ($2p$) often results in massive overfitting, tracking high-frequency noise rather than structural shifts.
*   **The Autocorrelation Problem:** PELT and similar changepoint algorithms assume that the residuals (data minus the fitted piecewise model) are independent, identically distributed (i.i.d.) random variables. Environmental time series are almost universally autocorrelated (red noise).
*   **False Positives:** Applying standard changepoint detection to highly autocorrelated data will drastically inflate the false positive rate, identifying "regime shifts" that are merely normal low-frequency stochastic excursions of a red noise process.
*   **Recommendation:** Ensure data is appropriately pre-whitened or explicitly model the autocorrelation structure (e.g., using AR cost functions) before interpreting changepoints in continuous variables.

**References:**
*   Killick, R., Fearnhead, P., & Eckley, I. A. (2012). Optimal detection of changepoints with a linear computational cost. *Journal of the American Statistical Association*, 107(500), 1590-1598.

### 2.10 Spatial and Sliding Haar Analysis

**Spatial Haar (Distance instead of Time):**
*   **Validity:** The mathematics of the Haar structure function are agnostic to the dimension (time vs. space). Analyzing spatial longitudinal profiles (e.g., river chemistry downstream) is perfectly valid, provided the spatial series follows the same assumptions of self-affinity or stationary increments as time series.
*   **Warning:** Rivers are networks, not 1D lines. Applying 1D Spatial Haar across confluences where major tributary inputs occur violates the assumption of a continuous generating process, introducing massive artifactual steps. It should be applied to single, uninterrupted reaches.

**Sliding Haar (Real-time Volatility):**
*   **Validity:** Calculating continuous fluctuations via a sliding window is effectively applying a band-pass filter (specifically, a Haar wavelet filter) to the data. This is robust for detecting localized anomalies or periods of heightened variance at a specific scale $\tau$.
*   **Edge Effects:** Like any moving window operation, estimates at the beginning and end of the dataset suffer from truncation.

### 2.11 Event-Based Segmentation via Sliding Haar Volatility

**Mathematical Foundation:**
Implemented via `SegmentedRegimeAnalysis.segment_by_fluctuation`, this method combines Sliding Haar volatility computation with thresholding based on the Median Absolute Deviation (MAD) to dynamically segment a time series into distinct operational regimes (e.g., "event" vs. "background") at a designated scale.

**Validity & Interpretation:**
*   **Dynamic vs. Static Thresholds:** Unlike traditional hydrograph separation that relies on absolute magnitude thresholds or arbitrary baseflow separation filters, this method triggers based on *volatility* (variance over scale $\tau$). This provides a mathematically objective definition of an "event" as a period where the system's rate of change significantly deviates from its background scaling behavior.
*   **Robust Baseline Estimation:** Using the median absolute fluctuation as the baseline estimator is statistically robust against outliers and heavily skewed extreme events, ensuring that massive storms do not artificially inflate the threshold and mask smaller events.
*   **Limitations:** The binary classification (event vs. non-event) depends heavily on the chosen scale $\tau$ and the `threshold_factor`. It assumes that the "background" regime is characterized by a relatively constant, low-volatility state, which may be violated in strongly non-stationary or highly intermittent systems where the "background" itself exhibits complex multi-scale variation.

**References:**
*   Meylan, P., Favre, A.-C., & Musy, A. (2012). *Predictive Hydrology: A Frequency Analysis Approach*. CRC Press.

---

## 3. Discussion: Methodological Synergies and Limitations

**Data Preprocessing, Censored Data, and the Spectral Trap:**
`waterSpec` includes robust tools (`handle_censored_data`, `detrend_loess`) to prepare messy data, but these steps fundamentally alter the spectrum.
*   *Censored Data (Non-detects):* Data like `<2.0` represents a truncated distribution. The chosen strategy (dropping, substitution, or multiplication) injects specific frequencies or destroys them. Substituting all non-detects with `0` or half the detection limit creates artificial flatlines (variance = 0), manifesting as heavily biased low-frequency artifacts. `waterSpec` uses a robust regex parser to safely translate these, but relies on the user to understand the spectral impact of the substitution strategy.
*   *The Detrending Trap:* Detrending via linear regression or LOESS removes low-frequency power. If analyzing a non-stationary process (e.g., groundwater levels where $\beta > 1$), detrending before spectral analysis artificially flattens the spectrum at large scales, destroying the very information you are trying to measure. This acts as an arbitrary high-pass filter.
*   *Rule of Thumb:* Only detrend if you are strictly interested in the stationary fluctuations around a known, deterministic trend (e.g., climate change warming curve), and are prepared to ignore the largest scales where the trend dominates.

**Surrogate Data Testing (Phase Randomization vs. Parametric Power Law):**
The package provides two primary surrogate null models to test significance, but they have strictly non-overlapping valid use cases based on sampling regularity.

1.  **Phase Randomization (FFT-based):**
    *   *Validity:* This perfectly preserves the linear autocorrelation structure (the power spectrum) while destroying non-linearities and phase relationships. It is the gold standard null model for testing the significance of peaks or Cross-Haar correlations against a red-noise background (Theiler et al., 1992). `waterSpec` utilizes NumPy broadcasting to vectorize surrogate generation, performing `irfft` with `axis=-1` to handle all surrogates simultaneously for maximum performance.
    *   *Fatal Flaw for Irregular Data:* The FFT algorithm intrinsically assumes regular, evenly spaced sampling. `waterSpec` correctly warns that applying `generate_phase_randomized_surrogates` directly to highly irregular data yields fundamentally invalid distributions.

2.  **Parametric Power Law Surrogates (Timmer & Koenig 1995):**
    *   *Validity:* For irregular data, the robust approach is to simulate a continuous high-resolution process with a target theoretical spectrum ($\beta$), and then *resample* it to the exact irregular timestamps of the observations (Timmer & Koenig, 1995). `waterSpec` implements this via `generate_power_law_surrogates`. This correctly propagates the spectral leakage and aliasing caused by the irregular sampling window into the null distribution.
    *   *Limitation:* This is a parametric test. It tests against a *theoretical* $\beta$ model, not the exact empirical spectrum of the data like phase randomization does.

3.  **Bootstrapping Strategies in Spectral Fitting:**
    The `fitter.py` module deploys multiple rigorous bootstrapping strategies to formulate defensible confidence intervals depending on the data's residual structure:
    *   *Pairs Bootstrapping:* The simplest OLS formulation. Resamples ($X$, $Y$) pairs with replacement. Defensible when errors are heteroscedastic but completely independent.
    *   *Residual Bootstrapping:* Resamples the centered residuals derived from the OLS fit. Defensible strictly under homoscedasticity and zero autocorrelation. It fundamentally breaks if the Durbin-Watson statistic falls outside $[1.5, 2.5]$.
    *   *Wild Bootstrapping:* `waterSpec` implements wild bootstrapping utilizing Rademacher distributions (multiplying centered residuals randomly by $-1$ or $1$). This is highly defensible for datasets exhibiting severe heteroscedasticity, as it perfectly preserves the exact variance structure at each individual temporal or frequency index.
    *   *Moving Block Bootstrapping:* When non-linearities or explicit autocorrelations are present alongside irregular sampling, the aforementioned strategies fail. Block bootstrapping resamples contiguous chunks (blocks) of data, preserving the short-range autocorrelation structure and non-linear properties inherently contained within the block size, while scrambling long-range dependence.
    *   *Vectorized Bootstrap Execution:* For standard methods (like OLS pairs), `waterSpec` executes bootsrapping by executing vectorized multidimensional OLS equations over all bootstrap iterations simultaneously. This mathematically avoids iterative loops, providing massive computational acceleration. Conversely, the `theil-sen` robust fit relies on sequential loop iterations due to algorithm complexity.
    *   *Random Number Isolation:* `waterSpec` strictly uses `np.random.SeedSequence.spawn(...)` across the package (especially in `model_selector.py`) to enforce statistically independent bit streams for child processes or parallelized fits. This guarantees zero overlap in bootstrap ensemble sampling, securing the mathematical integrity of significance tests.

4.  **Block-Shuffled Surrogates:**
    *   *Validity:* An alternative to purely spectral surrogates is `generate_block_shuffled_surrogates`. This method shuffles localized chunks of indices. It successfully destroys long-term memory (scales larger than `block_size`) while preserving the exact probability distribution and short-term intra-block structure.
    *   *Limitation:* Similar to phase randomization, this operates strictly on *indices*. If the data contains large, irregular sampling gaps, a block of $N$ indices does not correspond to a uniform duration of time. Applying this to heavily gappy data will corrupt the temporal axis, rendering the surrogate analysis invalid.

5.  **Empirical P-Value Calculation:**
    *   *Validity:* When evaluating the significance of a metric against a distribution of surrogate metrics (e.g. `calculate_significance_p_value`), `waterSpec` calculates empirical p-values using the conservative $(k+1)/(n+1)$ formula (where $k$ is the number of surrogate values $\geq$ the observed value, and $n$ is the total number of surrogates). This formulation prevents zero-p-value bounds and correctly accounts for the observation itself as part of the null distribution, providing a rigorously sound test even for small surrogate ensembles. The implementation robustly handles empty surrogate arrays by safely returning `np.nan`. Furthermore, by default it performs a rigorous *two-sided* test checking absolute magnitudes (`two_sided=True`), ensuring that deviations in either direction (positive or negative correlations) are appropriately penalized against the null distribution.

**References:**
*   Prichard, D., & Theiler, J. (1994). Generating surrogate data for time series with several simultaneously measured variables. *Physical review letters*, 73(7), 951.
*   Theiler, J., Eubank, S., Longtin, A., Galdrikian, B., & Farmer, J. D. (1992). Testing for nonlinearity in time series: the method of surrogate data. *Physica D: Nonlinear Phenomena*, 58(1-4), 77-94.
*   Timmer, J., & Koenig, M. (1995). On generating power law noise. *Astronomy and Astrophysics*, 300, 707.

---

## 4. Conclusion

The `waterSpec` package implements statistically rigorous methods, but provides enough rope for a careless user to hang themselves.

1.  **Lomb-Scargle** is for finding periodic peaks in uneven data, not for slope estimation in highly gappy data.
2.  **Haar Analysis** is the superior tool for scaling exponents ($\beta$) in irregular data.
3.  **BIC model selection** protects against overfitting breakpoints.
4.  **Partial Cross-Haar** must be treated with extreme skepticism due to its reliance on Gaussian assumptions applied to potentially non-Gaussian fluctuations.
5.  **Surrogate Generation** must be carefully matched to the sampling regime: Phase Randomization for even data, Timmer & Koenig simulation for uneven data.

Researchers must justify their method choices (LS vs Haar) based on the sampling irregularity and the specific scientific question (peaks vs slopes), and respect the boundaries established by the implemented warnings.

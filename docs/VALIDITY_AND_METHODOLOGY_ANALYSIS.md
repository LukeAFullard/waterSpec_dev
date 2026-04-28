# waterSpec: Validity and Methodology Analysis

**Abstract**
This document provides a rigorous, critical review of the spectral and statistical methodologies implemented in the `waterSpec` package. Written from an academic and senior statistical perspective, it evaluates the mathematical foundations, assumptions, edge cases, and validity boundaries of each method. It is intended for researchers and practitioners who need to defend the use of `waterSpec` in peer-reviewed publications.

---

## 1. Introduction

Environmental and geophysical time series analysis is fundamentally challenged by "messy" real-world data: missing observations, irregular sampling intervals, non-stationarity, extreme intermittency, and pervasive power-law scaling (long-range dependence or "red noise"). The `waterSpec` package attempts to navigate these issues using a combination of traditional spectral methods (Lomb-Scargle) and multiscale temporal approaches (Haar Wavelet Analysis).

However, the application of these methods requires strict adherence to their underlying mathematical assumptions. Misapplication can lead to spurious scaling exponents, artificial correlations, false detection of regime shifts, or misidentified periodicities. This document critically questions the validity of each analytical component, detailing when they are statistically robust and when they fail, providing key literature references justifying these claims.

---

## 2. Methodology Critique

### 2.1 Lomb-Scargle Periodogram (LS)

**Mathematical Foundation:**
The LS periodogram is effectively a least-squares fit of sinusoids to data. It was explicitly derived to handle unequally spaced data in astrophysics without requiring artificial interpolation (Lomb, 1976; Scargle, 1982). For a time series $x(t_i)$ evaluated at arbitrary times $t_i$, the normalized periodogram at angular frequency $\omega = 2\pi f$ is defined as:

$$ P_N(\omega) = \frac{1}{2\sigma^2} \left\{ \frac{\left[\sum_i (x_i - \bar{x}) \cos \omega(t_i - \tau)\right]^2}{\sum_i \cos^2 \omega(t_i - \tau)} + \frac{\left[\sum_i (x_i - \bar{x}) \sin \omega(t_i - \tau)\right]^2}{\sum_i \sin^2 \omega(t_i - \tau)} \right\} $$

where $\tau$ is a frequency-dependent time offset specified by:
$$ \tan(2\omega\tau) = \frac{\sum_i \sin(2\omega t_i)}{\sum_i \cos(2\omega t_i)} $$
This specific choice of $\tau$ makes the periodogram identical to a least-squares fit of the model $x(t) = A \cos(\omega t) + B \sin(\omega t)$ to the data.

**Validity & Strengths:**
*   **Peak Detection:** It is statistically robust for detecting deterministic, narrowband periodicities (e.g., diurnal, annual cycles) superimposed on white noise. The False Alarm Probability (FAP) provides a rigorous frequentist significance threshold, utilizing extreme value statistics to account for multiple independent frequencies. VanderPlas (2018) provides an extensive review of its validity for periodic detection, and Baluev (2008) details the exact extreme value distributions necessary for precise analytical FAP computation, circumventing the need for computationally expensive Monte Carlo simulations in many cases.
*   **No Interpolation:** By avoiding interpolation, it prevents the artificial introduction of high-frequency noise or smoothing artifacts that plague standard FFT applications on gappy data.

*   **Generalized Lomb-Scargle (GLS):** The standard LS periodogram inherently assumes that the data is centered around a known, true zero mean. In heavily irregular data, the sample mean can deviate significantly from the true underlying population mean. The Generalized Lomb-Scargle (GLS) periodogram, formalized by Zechmeister and Kürster (2009), resolves this by explicitly incorporating a floating mean parameter ($c$) into the sinusoidal fit: $y(t) = A \cos(\omega t) + B \sin(\omega t) + c$. This strictly prevents large-scale low-frequency artifacts caused by mean offsets from spuriously elevating high-frequency spectral power. It represents a mathematically superior approach for unevenly sampled series where calculating a simple global sample mean is biased by the clustering of sampling points. `waterSpec` utilizes `astropy.timeseries.LombScargle`, which defaults to this generalized formulation (`fit_mean=True`), ensuring rigorous spectral estimates independent of baseline shifts. As Zechmeister and Kürster point out, failure to fit a floating mean when sample clustering induces artificial offsets will systematically alias low-frequency power into high-frequency domains, destroying the integrity of empirical spectral slopes.


**Weaknesses & Failure Modes (When NOT to use):**
*   **Pseudo-Nyquist Constraints:** In evenly sampled data, the highest resolvable frequency is the Nyquist limit ($f_N = 1/(2\Delta t)$). In unevenly sampled data, a "Pseudo-Nyquist" frequency exists, often defined by the shortest sampling interval ($f_{pN} \approx 1/(2\Delta t_{min})$) (Eyer & Bartholdi, 1999). Searching for periodicities up to very high frequencies drastically increases computational load and exacerbates aliasing.
*   **Bootstrap Performance:** `waterSpec` allows using `fap_method="bootstrap"` for empirical FAP estimation via the `find_significant_peaks` function. However, this is computationally expensive. The `find_significant_peaks` implementation explicitly emits a performance `UserWarning` when this method is selected, and a secondary warning if it detects more than 5 peaks, as the bootstrap algorithm's slow execution time scales poorly with multiple peak extraction.
*   **Spectral Slope Bias:** The most critical weakness of LS is its vulnerability to *spectral leakage* when estimating the continuum spectral slope ($\beta$) of red noise processes in highly irregular or gappy data. Energy from low frequencies "leaks" into high frequencies due to the convolution with the spectral window function, flattening the apparent spectrum and heavily biasing $\beta$ downwards.
*   **Aliasing and the Spectral Window:** Uneven sampling does not completely eliminate aliasing; it merely redistributes aliased power into a complex, continuous background. The "spectral window function" (the discrete Fourier transform of the exact sampling times) dictates how true peaks are convolved and where "ghost" peaks appear. Highly periodic gaps (e.g., missing weekend data, diurnal missing gaps) create strong aliases that precisely mimic true physical signals (VanderPlas, 2018).
*   **Conclusion:** If the Coefficient of Variation (CV) of the sampling intervals is high (> 0.5), or if there are massive gaps (e.g., > 10% of total duration), **do not use LS to estimate $\beta$**. Use Haar Wavelets instead.

**References:**
*   Baluev, R. V. (2008). Assessing the statistical significance of periodogram peaks. *Monthly Notices of the Royal Astronomical Society*, 385(3), 1279-1285.
*   Eyer, L., & Bartholdi, P. (1999). Variable stars: Which Nyquist frequency?. *Astronomy and Astrophysics Supplement Series*, 135(1), 1-3.
*   Lomb, N. R. (1976). Least-squares frequency analysis of unequally spaced data. *Astrophysics and Space Science*, 39, 447-462.
*   Scargle, J. D. (1982). Studies in astronomical time series analysis. II - Statistical aspects of spectral analysis of unevenly spaced data. *The Astrophysical Journal*, 263, 835-853.
*   VanderPlas, J. T. (2018). Understanding the Lomb-Scargle Periodogram. *The Astrophysical Journal Supplement Series*, 236(1), 16.
*   Zechmeister, M., & Kürster, M. (2009). The generalised Lomb-Scargle periodogram. A new formalism for the floating-mean and Keplerian periodograms. *Astronomy & Astrophysics*, 496(2), 577-584.


### 2.2 Haar Wavelet Analysis (First-Order Structure Function)

**Mathematical Foundation:**
Haar analysis calculates the variance of the difference in means between adjacent non-overlapping (or overlapping) windows of size $\tau$. The first-order structure function $S_1(\tau)$ is computed as the variance of the fluctuation $\Delta x(\tau, t)$:

$$ \Delta x(\tau, t) = \langle x(u) \rangle_{t < u < t+\tau} - \langle x(u) \rangle_{t-\tau < u < t} $$

where $\langle \cdot \rangle$ denotes the mean over the specified time interval. The scaling exponent $m$ is derived from the relation:
$$ S_1(\tau) \propto \tau^m $$
This exponent $m$ relates directly to the spectral exponent $\beta$ via the equation $\beta = 2m + 1$ (Lovejoy & Schertzer, 2012). Mathematically, the Haar variance is directly proportional to the Allan Variance ($\sigma_A^2(\tau) = \frac{1}{2} \langle (\langle x \rangle_{t+\tau} - \langle x \rangle_t)^2 \rangle$), a metric widely used in time and frequency metrology to quantify the stability of oscillators. For processes characterized by fractional Gaussian noise (fGn), the Allan variance strictly follows a power law, scaling proportionally to $\tau^{2H-2}$ where H is the Hurst exponent. This exact mathematical equivalence deeply validates Haar variance as a robust, non-parametric estimator of multi-scale variance capable of accurately characterizing fractional non-stationary processes.

**Validity & Strengths:**
*   **Robust to Gaps:** Because it operates strictly in the time domain, a gap simply means a specific paired window is skipped. It does not globally corrupt the estimates at other scales via leakage, unlike Fourier transforms.
*   **Non-Stationarity:** It inherently handles non-stationary processes (e.g., random walks, fractional Brownian motion where $\beta > 1$) superior to Fourier methods because the adjacent differencing operation acts as a localized detrending mechanism.

**Critical Analysis of Overlapping Windows:**
*   `waterSpec` defaults to overlapping windows (sliding the window point-by-point) to increase statistical power, analogous to the Maximum Overlap Discrete Wavelet Transform (MODWT). However, overlapping fundamentally introduces strong *autocorrelation* between the fluctuation estimates at a given scale.
*   **Statistical Consequence:** While the mean estimate of $S_1(\tau)$ remains unbiased, the standard error is artificially reduced if standard OLS regression is naively applied to fit the slope. The true Effective Degrees of Freedom (EDOF) for overlapping windows of scale $\tau$ over a total time $T$ remains $EDOF \approx T/\tau$, not the inflated sample size $N_{overlap} = N - \tau$. Percival (1995) rigorously proved how naive variance estimates drastically underestimate confidence intervals for overlapping wavelet variance.
*   **Mitigation:** `waterSpec` utilizes moving block bootstrapping (via indices) or parametric Monte Carlo surrogates to precisely estimate confidence intervals on the fit. This is mathematically essential because standard OLS homoscedasticity and independence assumptions are violated. Furthermore, robust Theil-Sen regression via `MannKS.trend_test` ensures that outlier fluctuation scales do not completely bias the spectral exponent fit.

**Small-Sample Bias Correction:**
*   Standard Haar variance severely underestimates the true population variance when the number of data points per window is small. `waterSpec` implements an explicitly unbiased standard deviation estimator (`std_corrected`) utilizing correction factors derived from Gamma functions under the assumption of local normality: $\sigma_{unbiased} = \sigma_{sample} \cdot \sqrt{\frac{2}{N-1}} \exp(\Gamma(\frac{N-1}{2}) - \Gamma(\frac{N}{2}))$. This is crucial for high-frequency (small $\tau$) validity where point density per window drops. If the underlying data is heavily skewed or leptokurtic (non-Gaussian), this correction may slightly over- or under-correct, but it remains vastly superior to the uncorrected biased sample variance.

**Custom Statistics (Percentiles & Medians):**
*   `waterSpec` allows evaluating fluctuations using custom statistics like percentiles (e.g., 95th) instead of means. While useful for examining the scaling of extreme events, standard scaling relations ($\beta = 2m + 1$) are explicitly derived from the properties of variances (or mean-squared fluctuations). The theoretical translation of percentile-based slopes to traditional spectral $\beta$ is not firmly established in linear spectral theory and must be treated purely as an empirical scaling index.

**Aggregation Methods for the Structure Function ($S_1$):**
*   **Mean & Median:** Aggregating window fluctuations via the "mean" of the absolute differences provides a robust and distribution-agnostic standard $S_1$. "Median" acts as an even more robust metric against single extreme fluctuation outliers.
*   **Root Mean Square (RMS):** Approximates $\sqrt{S_2}$, connecting the first-order analysis directly to the second moment necessary for exploring multifractal scaling features (e.g., estimating intermittency $K(2)$).

**Edge Effects (Cone of Influence):**
*   Similar to the Continuous Wavelet Transform (CWT), Haar analysis suffers from edge effects near the beginning and end of the time series where windows are truncated or data is sparse. This creates a "Cone of Influence" (COI). Interpretations of long-scale fluctuations near the series boundaries must be treated with extreme caution, as they are calculated from artificially shortened effective window lengths.

**References:**
*   Lovejoy, S., & Schertzer, D. (2012). *The Weather and Climate: Emergent Laws and Multifractal Cascades*. Cambridge University Press.
*   Percival, D. P. (1995). On estimation of the wavelet variance. *Biometrika*, 82(3), 619-631.

### 2.3 Multifractal Intermittency Correction ($K(2)$)

**Mathematical Foundation:**
The relationship $\beta = 2m + 1$ is rigorously bounded and strictly valid only for $-1 < m < 1$, which corresponds to $-1 < \beta < 3$. Outside this range, the structure function no longer maintains a simple power-law relationship to the spectral exponent. Furthermore, it fundamentally assumes the process is monofractal (a single Hurst exponent $H$ characterizes the scaling of all statistical moments). For intermittent environmental processes (e.g., heavy-tailed rainfall distributions, episodic solute flushing), this breaks down. The Universal Multifractal framework provides a necessary correction:

$$ \beta = 1 + 2H - K(2) $$

where $H$ is the Hurst exponent (related to the first-order mean fluctuation scaling, $q=1$) and $K(2)$ characterizes the intermittency. In multifractal theory, $K(q)$ is the scaling moment function such that the moments of the fluctuations scale as $\langle (\Delta x)^q \rangle \propto \tau^{qH - K(q)}$. For the second moment (variance or power spectrum, $q=2$), $K(2)$ directly quantifies the deviation from simple monofractal scaling due to intermittent bursts of variance.

**Validity:**
*   This is a highly advanced feature. Accurately estimating $K(2)$ (often empirically derived via the difference in slope between the 2nd and 1st order structure functions) requires vast amounts of high-quality, high-frequency data to correctly sample the rare, extreme bursts defining the intermittency.
*   **Warning:** Applying the $K(2)$ correction to short, noisy time series will likely inject more variance into the $\beta$ estimate than the bias it attempts to remove. It should only be deployed when physical evidence strongly suggests extreme intermittency (e.g., turbulent cascades, storm-event transport mechanisms) and datasets are extensive ($N > 10^4$) and span at least two decades of reliable temporal scale.

**References:**
*   Schertzer, D., & Lovejoy, S. (1987). Physical modeling and analysis of rain and clouds by anisotropic scaling multiplicative processes. *Journal of Geophysical Research: Atmospheres*, 92(D8), 9693-9714.

### 2.4 Segmented Spectral Fits

**Mathematical Foundation:**
Using `mannks` or `piecewise-regression`, the package fits continuous broken-stick models to the log-log spectrum to mathematically identify scales where process dominance shifts (e.g., a shift from white noise to red noise) (Toms & Lesperance, 2003). A single-breakpoint piecewise model follows: $y = \beta_0 + \beta_1 x + \beta_2 (x - c) H(x - c)$, where $H$ is the Heaviside step function and $c$ is the breakpoint scale. The codebase leverages the robust `MannKS` package (`MannKS.segmented_trend_test`) to calculate breakpoints, inherently utilizing block bootstrapping to preserve the spectral autocorrelation structure when determining confidence bounds. To ensure reproducibility across standard, segmented, and Haar spectral analyses, `fit_standard_model`, `fit_segmented_spectrum`, `fit_haar_slope`, and `fit_segmented_haar` rigorously extract and explicitly map seed values (`random_state`) through to the robust `MannKS` fitting methodologies. Furthermore, when using the `theil-sen` method, `fit_standard_model` gracefully falls back to `scipy.stats.theilslopes` if `MannKS.trend_test` fails.

**Validity & The BIC Criterion:**
*   **Overfitting Risk:** It is mathematically trivial to fit a multi-segmented line to a highly noisy empirical spectrum and lower the Residual Sum of Squares (RSS).
*   **Defense:** The package's reliance on the Bayesian Information Criterion (BIC) to select between a standard line and a segmented line provides a statistically rigorous penalty against complexity. BIC is formulated as: $BIC = n \ln(RSS/n) + k \ln(n)$, where $n$ is the number of valid data points and $k$ is the number of model parameters. If BIC selects a segmented fit, the regime shift is robustly supported by the data variance reduction overcoming the penalty.
*   **The Assumption of Normality:** The standard $RSS$-based formulation of BIC intrinsically assumes that the model residuals are i.i.d. Gaussian. However, log-log spectral estimates often exhibit highly skewed, non-Gaussian error distributions (e.g., Chi-squared for Fourier periodograms). While Haar variances are more robust, the Gaussian assumption is an approximation. Therefore, the internal `_calculate_bic` routine specifically traps perfectly overfitted "zero RSS" models ($RSS < 10^{-12}$), returning $BIC = \infty$ and emitting a `UserWarning`. This safely prevents the algorithm from selecting mathematically degenerate piecewise models driven by numerical artifacts.

**References:**
*   Schwarz, G. (1978). Estimating the dimension of a model. *The Annals of Statistics*, 6(2), 461-464.
*   Toms, P. S., & Lesperance, M. L. (2003). Piecewise regression: a tool for identifying ecological thresholds. *Ecology*, 84(8), 2034-2041.

### 2.5 Bivariate (Cross-Haar) Analysis

**Mathematical Foundation:**
Calculates the Pearson correlation between the localized Haar fluctuations of two variables, $X$ and $Y$, specifically at scale $\tau$. Let $\Delta x(\tau, t)$ and $\Delta y(\tau, t)$ be the fluctuations computed at overlapping or non-overlapping time windows of length $\tau$. The Cross-Haar correlation is:

$$ \rho_{XY}(\tau) = \frac{\sum_t (\Delta x(\tau, t) - \overline{\Delta x})(\Delta y(\tau, t) - \overline{\Delta y})}{\sqrt{\sum_t (\Delta x(\tau, t) - \overline{\Delta x})^2 \sum_t (\Delta y(\tau, t) - \overline{\Delta y})^2}} $$

**Validity & Interpretation:**
*   **Scale-Dependent Correlation:** This is a powerful, theoretically sound method for decoupling short-term hysteresis from long-term trends. It serves as a strict time-domain analog to Cross-Wavelet Transform (XWT) and Wavelet Coherence approaches (Grinsted et al., 2004), without requiring continuous data interpolation or complex wavelets.
*   **Lead/Lag and Phase Dynamics:** Unlike complex continuous wavelets which can resolve continuous arbitrary phase angles $[-\pi, \pi]$, Cross-Haar strictly computes real Pearson correlations. This effectively implies it only measures $0$ (in-phase) or $\pi$ (anti-phase) relationships. If two signals have a persistent, orthogonal phase shift (e.g., $\pi/2$, a perfect quarter-cycle lag), $\Delta x$ and $\Delta y$ become mathematically orthogonal, and the Cross-Haar correlation will tend exactly toward zero, utterly failing to capture the strong causal dependency. Bivariate Haar is strictly limited to identifying *in-phase* or *anti-phase* scale-dependent relationships.
*   **Assumptions:** It inherently assumes the relationship between the variables at a given scale is strictly linear (Pearson). If the relationship is highly non-linear or defined by complex limit cycles, Cross-Haar correlation will vastly underestimate the true dependency.

**References:**
*   Grinsted, A., Moore, J. C., & Jevrejeva, S. (2004). Application of the cross wavelet transform and wavelet coherence to geophysical time series. *Nonlinear processes in geophysics*, 11(5/6), 561-566.
*   Torrence, C., & Compo, G. P. (1998). A practical guide to wavelet analysis. *Bulletin of the American Meteorological society*, 79(1), 61-78.

### 2.6 Hysteresis Classification within Bivariate Analysis

**Mathematical Foundation:**
Extends bivariate analysis by structurally quantifying the loop area and direction (clockwise vs. counter-clockwise) in the phase space of Haar fluctuations for two variables at a specific scale $\tau$. It rigorously utilizes the discrete shoelace formula (Surveyor's formula) to compute the signed polygon area formed by the sequential continuous fluctuation pairs $(X_i, Y_i) = (\Delta x(\tau, t_i), \Delta y(\tau, t_i))$ mapping the trajectory at the chosen scale:

$$ \text{Area}(\tau) = \frac{1}{2} \sum_{i=1}^{n-1} (X_i Y_{i+1} - X_{i+1} Y_i) $$

**Validity & Interpretation:**
*   **Scale-Specific Hysteresis:** Traditional concentration-discharge (C-Q) hysteresis analysis is frequently confounded by underlying baseline shifts or nested storms. By completely isolating fluctuations at scale $\tau$, this method provides a mathematically pure way to evaluate hysteresis generated strictly by processes operating at that specific timescale (e.g., singular event-based flushing), effectively decoupling it from seasonal cycles or multi-year trends.
*   **Loop Area Significance:** In phase space, the area of the loop physically quantifies the magnitude of the hysteresis (the degree to which the system state depends on its trajectory, representing "memory" or non-equilibrium transport). Analogous to calculating the work done in a thermodynamic cycle, this area accurately measures the total deviation from linearity. The shoelace formulation elegantly captures both the overall magnitude and the prevailing temporal ordering. Zuecco et al. (2016) extensively validate the use of normalized area indices to classify complex multi-peak storm hysteresis that defies simple linear metrics.
*   **Directionality:** The sign of the computed area explicitly indicates the direction. A clockwise loop (positive area, often implying a rapidly depleted, supply-limited, or proximal source in concentration-discharge relationships) is strictly distinguished from a counter-clockwise loop (negative area, often implying a delayed, distant, transport-limited, or groundwater-driven source). Complex figure-eight loops will yield net areas reflecting the dominant loop geometry. Lloyd et al. (2016) discuss the complexities of such indices in correctly identifying source dynamics during complex nested storm events.
*   **Limitations:** The metric is intensely sensitive to high-frequency stochastic noise at small scales and demands overlapping windows to adequately trace a smooth loop shape in phase space. If the arbitrarily chosen scale $\tau$ does not precisely match the characteristic timescale of the underlying hysteresis-generating physical event, the computed area will collapse to near zero, rendering the analysis uninterpretable.

**References:**
*   Lloyd, C. E., Freer, J. E., Johnes, P. J., & Collins, A. L. (2016). Technical Note: Testing an improved index for analysing storm discharge-concentration hysteresis. *Hydrology and Earth System Sciences*, 20(2), 625-632.
*   Zuecco, G., Penna, D., Borga, M., & van Meerveld, H. J. (2016). A versatile index to characterize hysteresis between hydrological variables at the runoff event timescale. *Hydrological Processes*, 30(9), 1449-1466.

### 2.7 Partial Cross-Haar Analysis (Experimental)

**Mathematical Foundation:**
Calculates the partial correlation $\rho_{XY|Z}$ utilizing the classical linear partial correlation algebraic formula applied to the localized Haar fluctuations of X, Y, and Z:

$$ \rho_{XY|Z}(\tau) = \frac{\rho_{XY}(\tau) - \rho_{XZ}(\tau)\rho_{YZ}(\tau)}{\sqrt{(1 - \rho_{XZ}(\tau)^2)(1 - \rho_{YZ}(\tau)^2)}} $$

**Critical Questioning of Validity:**
*   **The Assumption of Multivariate Normality:** The standard partial correlation formula explicitly and fundamentally assumes that the variables (in this case, the isolated *fluctuations* of X, Y, and Z at scale $\tau$) strictly follow a multivariate Gaussian distribution.
*   **The Reality of Environmental Data:** Environmental fluctuations, especially localized at smaller temporal scales, are notoriously non-Gaussian (heavy-tailed, highly skewed, structurally intermittent).
*   **Linear Residualization:** The formula essentially calculates the correlation of the residuals of X and Y after linearly regressing them against Z. If Z influences X or Y non-linearly, linearly partialling out Z will leave substantial residual confounding, rendering $\rho_{XY|Z}$ meaningless and potentially leading to aggressively false causal inferences.
**Verdict:** The severe warning prominently attached to this function in the codebase is entirely justified. While mathematically computable without failure, the *statistical interpretation* of $\rho_{XY|Z}$ as the "true" conditional dependency is exceptionally weak if the fluctuations are highly non-Gaussian or non-linear. Methodologically, this concept draws vaguely from Partial Wavelet Coherence (Ng & Chan, 2012), which is generally formulated for continuous frequency domains with phase properties rather than discrete real-valued temporal structures.
*   **Recommendation:** Use purely as a qualitative exploratory tool. Do not base definitive causal conclusions or publishable claims solely on this metric without thoroughly verifying the stringent distributional assumptions of the fluctuations at each specific scale.

**References:**
*   Ng, E. K., & Chan, J. C. (2012). Geophysical applications of partial wavelet coherence and multiple wavelet coherence. *Journal of Atmospheric and Oceanic Technology*, 29(12), 1845-1853.

### 2.8 Lomb-Scargle Cross-Spectrum (Phase Lag)

**Mathematical Foundation:**
Extends the generalized Lomb-Scargle formalism to two simultaneously observed variables to find the phase difference (lead/lag dynamics) at specific continuous frequencies.

**Validity & Limitations:**
*   **Computational Rigor and Cache Optimization:** Under the hood, `calculate_ls_cross_spectrum` avoids iterating frequency-by-frequency in pure Python, which destroys CPU cache locality. Instead, it dynamically batches frequencies—targeting optimal ~2MB payload chunks explicitly aligned to maximize CPU L3 cache hits—to perform highly vectorized block `np.linalg.solve` routines over multidimensional frequency arrays.
*   **Mathematical Identity Shortcuts:** The batch processing loop further optimizes scaling performance by computing the $\omega$ array per batch (`2 * np.pi * f_batch[:, np.newaxis]`) outside the primary loop and slicing it. Inside the batch, massive trigonometric sum evaluations are dramatically reduced using exact mathematical identities (e.g., `Swss = sum_w - Swcc`). This structurally circumvents redundant floating-point processing overhead, preserving full mathematical exactness while rendering the Lomb-Scargle Cross-Spectrum computationally tractable for massive, highly irregular temporal sequences without encountering catastrophic Out-Of-Memory (OOM) halts.
*   **Pre-Calculated Boundary Searches:** Beyond Lomb-Scargle, scaling and windowing functions across the entire `waterSpec` package (including `calculate_sliding_haar`, `bivariate.py`, and `multivariate.py`) utilize strict boolean masking bounds (e.g., `t_starts[t_starts + window_size <= time[-1]]`) alongside vectorized `np.searchsorted`. Pre-calculating window boundaries physically ensures zero iterative Python function call overhead when fetching index blocks for complex mathematical evaluations and fundamentally prevents exceeding array bounds due to insidious floating-point inaccuracies.
*   **Noise Sensitivity and Coherence Thresholding:** Phase estimation is profoundly sensitive to noise. If the true Cross-Spectral Power (Coherence, $C_{XY}^2 = |P_{XY}|^2 / (P_{XX} P_{YY})$) is low at a given frequency, the mathematically estimated phase lag is meaningless (essentially manifesting as a random variable distributed uniformly on $[-\pi, \pi]$). The raw pointwise coherence values outputted by `calculate_ls_cross_spectrum` are essentially exactly 1.0 at every frequency point due to the lack of inherent spectral smoothing. Meaningful coherence estimation requires applying spectral smoothing over a frequency band containing multiple independent estimates. A rigorous statistical threshold for coherence must be rigidly established (e.g., via Phase Randomized Monte Carlo surrogates) *before* attempting to interpret phase lags.
*   **Interpretation of Phase Wraparound:** Phase is inherently circular (strictly defined modulo $2\pi$). Translating a phase difference into a definitive temporal time lag ($\Delta t = \Delta\phi / (2\pi f)$) is deeply ambiguous without strong prior physical constraints on causality, as a mathematically calculated lag of $\Delta\phi$ is entirely indistinguishable from a physical lead of $2\pi - \Delta\phi$, or a lag of $\Delta\phi + 2\pi N$.
*   **Conclusion:** Researchers must strictly interpret phase lags *only* at frequencies where both variables exhibit highly significant, localized power exceeding a red-noise background, and where the cross-coherence strongly exceeds a strict surrogate-derived $95\%$ significance threshold.

**References:**
*   Hocke, K. (1998). Phase estimation with the Lomb-Scargle periodogram method. *Annales Geophysicae*, 16(3), 356-358. Hocke demonstrated that exact complex phase extraction is rigorously possible via standard least-squares inversion of the non-orthogonal sine and cosine basis functions on uneven grids.

### 2.9 Changepoint Detection (PELT Algorithm)

**Mathematical Foundation:**
The package utilizes the mathematically exact Pruned Exact Linear Time (PELT) algorithm via the robust `ruptures` library to detect distinct structural shifts in the mean or variance of a time series. The algorithm fundamentally seeks to locate the global minimum of a penalized cost function:

$$ \min_{m, \tau_1, \dots, \tau_m} \left[ \sum_{i=1}^{m+1} C(y_{\tau_{i-1}:\tau_i}) + \beta_{penalty} m \right] $$

where $C$ is a statistical cost function (e.g., Gaussian negative log-likelihood to detect shifts in both mean and variance, or simple $L_2$ norm for mean-only), $m$ is the total number of changepoints, $\tau_i$ are the integer changepoint indices, and $\beta_{penalty}$ is the crucial penalty parameter designed to guard against catastrophic overfitting.

**Validity & Limitations:**
*   **Algorithmic Efficiency:** PELT is mathematically guaranteed to find the exact global minimum of the penalized cost function and operates highly efficiently even on massive datasets (Killick et al., 2012). It achieves theoretical $O(N)$ linear computational time strictly under the assumption that the true number of changepoints increases linearly with dataset size $N$. The pruning mechanism relies on a fundamental mathematical insight: if the unpenalized cost to segment a prefix of data up to a candidate point plus the cost to bridge the remainder exceeds the cost of a previously evaluated changepoint configuration by more than the penalty constant, that candidate point can be safely discarded forever.
*   **Penalty Selection (AIC vs. BIC):** The total number of detected changepoints is intensely sensitive to the chosen penalty factor ($\beta_{penalty}$). `waterSpec` typically advocates for a penalty mathematically akin to BIC ($p \log(n)$), which heavily and appropriately penalizes structural complexity, favoring fewer, vastly more statistically profound regime shifts. Deploying a weaker AIC-like penalty ($2p$) on environmental data almost universally results in massive overfitting, tracking high-frequency stochastic noise rather than genuine structural system shifts.
*   **The Autocorrelation Problem:** PELT and virtually all standard changepoint algorithms fundamentally assume that the model residuals (the data minus the fitted piecewise constant/linear model) are independent, identically distributed (i.i.d.) random variables. However, environmental time series are almost universally strongly autocorrelated, frequently characterized by fractional Gaussian noise, red noise, or long-range dependence.
*   **False Positives in Red Noise:** Applying standard changepoint detection to highly autocorrelated data will drastically and systematically inflate the false positive rate. Because red noise processes are dominated by low-frequency power, they naturally generate prolonged excursions or "local trends" that mathematically masquerade exactly like structural, deterministic mean shifts to the cost function. Consequently, using PELT without rigorous pre-whitening or explicitly modeling the underlying autocorrelation structure (e.g., using AR-based cost functions) will identify dense clusters of "regime shifts" that are merely normal, stochastically generated low-frequency fluctuations inherent to the red noise process.
*   **Recommendation:** Strictly ensure data is appropriately pre-whitened or statistically robust against long-range dependence before interpreting structural changepoints in continuous environmental variables. The concrete recommendation is to use AR(p) pre-whitening with the AR order $p$ explicitly selected by AIC, applying PELT exclusively to the fitted model residuals. Lund & Reeves (2002) explicitly demonstrate that a standard two-phase regression model (functionally similar to a changepoint mean-shift test) will drastically misidentify changepoints if underlying red noise processes are conflated with step functions.

**References:**
*   Killick, R., Fearnhead, P., & Eckley, I. A. (2012). Optimal detection of changepoints with a linear computational cost. *Journal of the American Statistical Association*, 107(500), 1590-1598.
*   Lund, R., & Reeves, J. (2002). Detection of undocumented changepoints: A revision of the two-phase regression model. *Journal of Climate*, 15(18), 2547-2554.

### 2.10 Spatial and Sliding Haar Analysis

**Spatial Haar (Distance instead of Time):**
*   **Validity:** The mathematics governing the Haar structure function are fundamentally agnostic to the underlying dimension (time vs. spatial distance). Analyzing spatial longitudinal profiles (e.g., continuous river chemistry measurements downstream) is perfectly mathematically valid, provided the spatial series strictly follows the exact same underlying assumptions of self-affinity or stationary spatial increments as temporal series.
*   **Warning:** Utilizing Spatial Haar on river systems demands caution. Rivers are highly branched hierarchical networks, not singular 1D lines. Applying 1D Spatial Haar broadly across network confluences where massive, abrupt tributary inputs occur aggressively violates the mathematical assumption of a continuous stochastic generating process, introducing massive artifactual steps into the variance structure. It should be strictly applied to singular, uninterrupted, structurally homogeneous reaches.
*   **Taylor's Hypothesis:** If spatial scaling is being interpreted as temporal scaling via a flow velocity (Taylor's hypothesis of frozen turbulence), the velocity field must be confirmed to be highly uniform; otherwise, the spatial-to-temporal scale mapping becomes non-linear and invalid.

**Sliding Haar (Real-time Volatility):**
*   **Validity:** Calculating continuous, localized fluctuations via a sliding window is mathematically equivalent to applying a continuous band-pass filter (specifically, a discrete Haar wavelet filter response) to the data series. This is highly robust for dynamically detecting localized anomalies or periods of structural heightened variance at a specific, targeted scale $\tau$.
*   **Edge Effects:** Like any mathematical moving window operation, estimates rigorously calculated at the extreme beginning and end of the dataset suffer from structural truncation and must be handled carefully.

### 2.11 Event-Based Segmentation via Sliding Haar Volatility

**Mathematical Foundation:**
Implemented via `SegmentedRegimeAnalysis.segment_by_fluctuation`, this advanced method strictly combines Sliding Haar localized volatility computation with robust statistical thresholding based on the Median Absolute Deviation (MAD) to dynamically segment a time series into distinct, physically meaningful operational regimes (e.g., "event" transport vs. "background" steady-state) precisely at a designated scale $\tau$.

**Validity & Interpretation:**
*   **Dynamic vs. Static Thresholds:** Unlike traditional simplistic hydrograph separation that relies heavily on absolute magnitude thresholds or arbitrary empirical baseflow separation filters, this method triggers rigorously based on localized *volatility* (the variance over scale $\tau$). This provides a powerful, mathematically objective definition of an "event" as a temporal period where the system's rate of change significantly and structurally deviates from its established background scaling behavior.
*   **Robust Baseline Estimation:** The MAD is mathematically defined as $MAD = \text{median}(|X_i - \text{median}(X)|)$. Utilizing the median absolute fluctuation as the primary baseline estimator is statistically robust (possessing a 50% breakdown point) against extreme outliers and heavily skewed, massive intermittent events. This ensures that a single massive storm does not artificially inflate the baseline threshold, which would erroneously mask and suppress the detection of smaller, yet structurally significant, subsequent events.
*   **Limitations:** The fundamental binary classification (event vs. non-event) depends intensely on the explicitly chosen physical scale $\tau$ and the user-defined `threshold_factor` ($k \times MAD$). It structurally assumes that the "background" regime is characterized by a relatively constant, low-volatility, steady-state, which may be violently violated in strongly non-stationary or highly intermittent systems where the "background" itself exhibits complex, multi-scale stochastic variation.

**References:**
*   Meylan, P., Favre, A.-C., & Musy, A. (2012). *Predictive Hydrology: A Frequency Analysis Approach*. CRC Press.

---

## 3. Discussion: Methodological Synergies and Limitations

**Data Preprocessing, Censored Data, and the Spectral Trap:**
`waterSpec` includes robust practical tools (`handle_censored_data`, `detrend_loess`) to prepare messy operational data, but these necessary steps fundamentally and irreversibly alter the frequency spectrum.
*   *Censored Data (Non-detects):* Data legally reported as `<2.0` physically represents a statistically truncated distribution. The chosen strategy (dropping, substitution, or scaling) synthetically injects specific frequencies or utterly destroys them. Substituting all non-detects with `0` or exactly half the detection limit creates artificial, prolonged flatlines (where local variance exactly equals 0), manifesting as massive, heavily biased low-frequency spectral artifacts. `waterSpec` rigorously uses a robust regex parser to safely translate these, but wholly relies on the analyst to deeply understand the devastating spectral impact of the chosen substitution strategy.
*   *The Detrending Trap:* Detrending via linear OLS regression or LOESS computationally removes low-frequency power. LOESS (Locally Estimated Scatterplot Smoothing) functions strictly as an arbitrary high-pass filter, where the span parameter dictates the cutoff frequency. If analyzing a fundamentally non-stationary process (e.g., groundwater levels where the theoretical $\beta > 1$), detrending prior to spectral analysis artificially flattens the spectrum at large scales, systematically destroying the very long-range information you are actively attempting to measure.
*   *Rule of Thumb:* Only perform mathematical detrending if you are strictly interested in isolating the stationary fluctuations around a physically known, deterministic trend (e.g., isolating weather from a climate change warming curve), and are prepared to permanently ignore the largest scales where the trend physically dominates.

**Surrogate Data Testing (Phase Randomization vs. Parametric Power Law):**
The package provides two primary, rigorous surrogate null models to test statistical significance, but they possess strictly non-overlapping valid mathematical use cases based on the underlying sampling regularity.

1.  **Phase Randomization (FFT-based) & IAAFT:**
    *   *Validity:* Standard Phase Randomization perfectly preserves the exact linear autocorrelation structure (the power spectrum) while completely destroying any non-linearities and deterministic phase relationships. It acts as the ultimate gold standard null model for testing the significance of localized peaks or Cross-Haar correlations against an empirical red-noise background (Theiler et al., 1992). `waterSpec` heavily utilizes highly optimized NumPy broadcasting to vectorize surrogate generation, executing `irfft` with `axis=-1` to computationally handle massive surrogate ensembles simultaneously for maximum processing performance.
    *   *IAAFT (Iterative Amplitude Adjusted Fourier Transform):* A critical limitation of simple Phase Randomization is that while it perfectly preserves the power spectrum, it alters the fundamental probability distribution (amplitude distribution) of the underlying data, often forcing it toward Gaussianity. The Iterated Amplitude Adjusted Fourier Transform (IAAFT) algorithm mathematically resolves this by iteratively projecting the sequence between the strict frequency domain (preserving amplitudes of the Fourier transform to maintain the power spectrum) and the strict temporal domain (rank-ordering to exactly match the original amplitude distribution). As demonstrated by Schreiber & Schmitz (1996, 2000), iterating this process until convergence guarantees a surrogate sequence that perfectly preserves the original amplitude distribution while maintaining a highly accurate approximation of the original linear autocorrelation structure. This is strictly required when the original data is highly non-Gaussian, preventing the surrogate null distribution from being synthetically narrowed or skewed by artificial Gaussianization.
    *   *Implementation:* `waterSpec` actively deploys `generate_iaaft_surrogates` for evenly-sampled data within its Bivariate Analysis routines, ensuring that rigorous phase randomization does not systematically corrupt the marginal heavy-tailed distributions characteristic of extreme hydrological events.
    *   *Fatal Flaw for Irregular Data:* The underlying FFT algorithms intrinsically and fundamentally assume regular, perfectly evenly spaced sampling. `waterSpec` correctly and strictly warns that applying FFT-based surrogates (like PR or IAAFT) directly to highly irregular, gappy data mathematically yields fundamentally invalid distributions, as the FFT perceives the sampling gaps as physical zero-values, warping the entire spectrum.

2.  **Parametric Power Law Surrogates (Timmer & Koenig 1995):**
    *   *Validity:* For highly irregular data, the only mathematically robust approach is to simulate a continuous, massively high-resolution theoretical process possessing a targeted theoretical spectrum ($\beta$), and subsequently *resample* this continuous process exactly to the precise irregular timestamps of the true observations (Timmer & Koenig, 1995). `waterSpec` implements this precisely via `generate_power_law_surrogates`. This correctly and elegantly propagates the severe spectral leakage and aliasing explicitly caused by the irregular sampling window directly into the null distribution, providing an honest baseline. As Timmer & Koenig mathematically formalized, simulating inverse power law noise accurately requires generating strictly independent complex Gaussian random variables in the frequency domain scaled by the desired power law, before inverse transforming, preventing artificial circular convolutions inherent to naive temporal filtering.
    *   *Limitation:* This remains a strictly parametric test. It tests the data against a *theoretical* $\beta$ model, not the exact empirical, localized spectrum of the data as phase randomization does.

3.  **Bootstrapping Strategies in Spectral Fitting:**
    The `fitter.py` module deploys multiple mathematically rigorous bootstrapping strategies to formulate defensible confidence intervals depending strictly on the data's residual structure:
    *   *Pairs Bootstrapping:* The simplest OLS formulation. Resamples paired ($X$, $Y$) observations with replacement. Mathematically defensible strictly when errors are heteroscedastic but completely temporally independent.
    *   *Residual Bootstrapping:* Resamples the strictly centered residuals derived from the initial OLS fit. Defensible strictly and solely under the assumption of perfect homoscedasticity and zero autocorrelation. It mathematically breaks down and generates severely overly narrow confidence intervals if the Durbin-Watson statistic falls outside the strict $[1.5, 2.5]$ bounds.
    *   *Wild Bootstrapping:* `waterSpec` implements highly robust wild bootstrapping utilizing the Mammen (1993) two-point distribution. This methodology is exceptionally defensible for datasets exhibiting severe structural heteroscedasticity. By multiplying the residual at a specific index $i$ by a random variable with mean zero, variance one, and specifically $E[W^3]=1$, wild bootstrapping strictly maintains the zero-mean assumption of the errors while perfectly preserving the exact heteroscedastic variance structure (the squared residual) at each individual specific temporal or frequency index (Liu, 1988; Mammen, 1993). The Mammen distribution is particularly favored over the symmetric Rademacher distribution for log-transformed environmental data as it explicitly preserves the skewness inherent in the localized residual error formulation.
    *   *Moving Block Bootstrapping:* When intense non-linearities or explicit, long-range autocorrelations are present alongside irregular sampling, all aforementioned strategies mathematically fail. Moving block bootstrapping (Künsch, 1989) resolves this by resampling contiguous sequential chunks (blocks) of data rather than individual points. This critically and rigidly preserves the short-range empirical autocorrelation structure and non-linear dependencies inherently contained within the designated block size, while structurally scrambling long-range dependence, producing vastly more defensible standard errors in heavily autocorrelated environmental data. Künsch (1989) proved that proper block size selection is critical: the block length must increase asymptotically with sample size, but at a strictly slower rate ($l \propto n^{1/3}$), to ensure asymptotic consistency of the variance estimator.
    *   *Vectorized Bootstrap Execution:* For standard methods (like OLS pairs), `waterSpec` executes bootstrapping by running highly vectorized multidimensional OLS tensor equations over all bootstrap iterations simultaneously. This mathematically avoids iterative Python loops, providing massive computational acceleration. Conversely, the highly robust `theil-sen` fit strictly relies on sequential loop iterations due to profound algorithmic complexity.
    *   *Random Number Isolation:* `waterSpec` strictly uses modern `np.random.SeedSequence.spawn(...)` across the entire package (especially critical in `model_selector.py`) to enforce perfectly statistically independent bit streams for child processes or parallelized fits. This guarantees absolute zero overlap in bootstrap ensemble sampling, securing the rigorous mathematical integrity of all significance tests.

4.  **Block-Shuffled Surrogates:**
    *   *Validity:* A robust alternative to purely spectral, frequency-domain surrogates is `generate_block_shuffled_surrogates`. This method temporally shuffles localized contiguous chunks of indices. It successfully destroys structural long-term memory (scales strictly larger than `block_size`) while perfectly preserving the exact probability distribution (all moments) and the short-term intra-block dynamic structure.
    *   *Limitation:* Similar to phase randomization, this operates strictly on *indices*, not physical time. The block size selection is critical: too small mathematically destroys the dynamics of interest, while too large leaves too few blocks to shuffle, creating a surrogate distribution with catastrophically low variance. Furthermore, if the data contains massive, irregular sampling gaps, a contiguous block of $N$ indices does not correspond to a uniform duration of physical time. Applying this blindly to heavily gappy data will severely corrupt the physical temporal axis, rendering the surrogate analysis entirely mathematically invalid.

5.  **Empirical P-Value Calculation:**
    *   *Validity:* When mathematically evaluating the significance of an observed metric against an ensemble distribution of surrogate metrics (e.g. via `calculate_significance_p_value`), `waterSpec` calculates exact empirical p-values utilizing the highly conservative $(k+1)/(n+1)$ formula (where $k$ is the absolute number of surrogate values $\geq$ the observed absolute value, and $n$ is the total ensemble size). As rigorously demonstrated by Phipson and Smyth (2010), calculating empirical permutation p-values via the traditional $k/n$ ratio systematically underestimates the true p-value, leading to inflated Type I error rates. This arises from the mistaken idea of using permutation merely to estimate the tail probability; rather, permutation exactly generates a discrete null distribution. Under this paradigm, where the observation itself is considered a random draw from the null, $k=0$ erroneously implying absolute certainty ($p=0$) is mathematically impossible. The $(k+1)/(n+1)$ formulation corrects this bias by structurally incorporating the observation into the null distribution space. This provides a profoundly sound test even for radically small surrogate ensembles. The implementation strictly and robustly handles empty surrogate arrays by safely returning `np.nan`. Furthermore, by default it performs a rigorous *two-sided* test checking absolute magnitudes (`two_sided=True`), ensuring that structural deviations in either direction (both massive positive or negative correlations) are appropriately symmetrically penalized against the true null distribution.

**References:**
*   Künsch, H. R. (1989). The jackknife and the bootstrap for general stationary observations. *The Annals of Statistics*, 17(3), 1217-1241.
*   Liu, R. Y. (1988). Bootstrap procedures under some non-i.i.d. models. *The Annals of Statistics*, 16(4), 1696-1708.
*   Mammen, E. (1993). Bootstrap and wild bootstrap for high dimensional linear models. *The Annals of Statistics*, 21(1), 255-285.
*   Phipson, B., & Smyth, G. K. (2010). Permutation P-values should never be zero: calculating exact P-values when permutations are randomly drawn. *Statistical Applications in Genetics and Molecular Biology*, 9(1), Article 39.
*   Prichard, D., & Theiler, J. (1994). Generating surrogate data for time series with several simultaneously measured variables. *Physical review letters*, 73(7), 951.
*   Schreiber, T., & Schmitz, A. (1996). Improved surrogate data for nonlinearity tests. *Physical Review Letters*, 77(4), 635-638.
*   Theiler, J., Eubank, S., Longtin, A., Galdrikian, B., & Farmer, J. D. (1992). Testing for nonlinearity in time series: the method of surrogate data. *Physica D: Nonlinear Phenomena*, 58(1-4), 77-94.
*   Timmer, J., & Koenig, M. (1995). On generating power law noise. *Astronomy and Astrophysics*, 300, 707.

---

## 4. Conclusion

The `waterSpec` package implements highly statistically rigorous methods, mathematically optimized for massive datasets, but provides enough analytical rope for a careless user to completely compromise their research.

1.  **Lomb-Scargle** is mathematically constructed for finding narrow periodic peaks in uneven data, not for estimating broad continuum spectral slopes in highly gappy, irregular data where it fundamentally fails.
2.  **Haar Analysis** is the mathematically superior, mathematically proven tool for extracting scaling exponents ($\beta$) in heavily irregular environmental data.
3.  **BIC model selection** strictly and robustly mathematically protects against the trivial overfitting of structural breakpoints.
4.  **Partial Cross-Haar** must be treated with extreme, intense skepticism due to its profound reliance on strict Gaussian linear assumptions applied to inherently and predictably non-Gaussian environmental fluctuations.
5.  **Surrogate Generation** must be rigorously and carefully matched strictly to the sampling regime: FFT Phase Randomization for perfectly even data, and Timmer & Koenig parametric simulation for realistically uneven data.

Researchers must rigidly and mathematically justify their method choices (LS vs Haar) based strictly on the sampling irregularity and the specific scientific hypothesis (peaks vs slopes), and absolutely respect the statistical boundaries and assumptions established by the implemented warnings.

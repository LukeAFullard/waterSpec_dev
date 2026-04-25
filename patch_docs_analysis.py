import re

with open("docs/VALIDITY_AND_METHODOLOGY_ANALYSIS.md", "r") as f:
    content = f.read()

# 1. Expand BIC under 2.4 Segmented Spectral Fits
bic_original = r"""**Validity & The BIC Criterion:**
*   **Overfitting Risk:** It is trivial to fit a multi-segmented line to a noisy spectrum and lower the Residual Sum of Squares (RSS).
*   **Defense:** The package's reliance on the Bayesian Information Criterion (BIC) to select between a standard line and a segmented line is statistically sound. BIC heavily penalizes additional parameters (Schwarz, 1978). If BIC selects a segmented fit, the regime shift is robustly supported by the data. The internal `_calculate_bic` routine correctly traps perfectly overfitted "zero RSS" models ($RSS < 10^{-12}$), returning $BIC = \infty$ and emitting a `UserWarning`, effectively banning artificial segments driven by numerical artifacts."""

bic_replacement = r"""**Validity & The BIC Criterion:**
*   **Overfitting Risk:** It is trivial to fit a multi-segmented line to a noisy spectrum and lower the Residual Sum of Squares (RSS).
*   **Defense:** The package's reliance on the Bayesian Information Criterion (BIC) to select between a standard line and a segmented line is statistically sound. BIC heavily penalizes additional parameters via the formulation: $BIC = n \ln(RSS/n) + k \ln(n)$, where $n$ is the number of data points and $k$ is the number of parameters. If BIC selects a segmented fit, the regime shift is robustly supported by the data. The internal `_calculate_bic` routine correctly traps perfectly overfitted "zero RSS" models ($RSS < 10^{-12}$), returning $BIC = \infty$ and emitting a `UserWarning`, effectively banning artificial segments driven by numerical artifacts. This prevents the algorithm from selecting mathematically degenerate piecewise models."""

content = content.replace(bic_original, bic_replacement)

# 2. Expand 2.2 Haar Wavelet Analysis overlapping windows
haar_overlap_replacement = r"""**Critical Analysis of Overlapping Windows:**
*   `waterSpec` defaults to overlapping windows to increase statistical power (reducing variance of the estimate), analogous to the Maximum Overlap Discrete Wavelet Transform (MODWT). However, overlapping windows introduce *autocorrelation* between the fluctuation estimates at a given scale.
*   **Statistical Consequence:** While the mean estimate of $S_1(\tau)$ remains unbiased, the standard error is artificially reduced if standard OLS regression is used to fit the slope, as the true degrees of freedom are much fewer than the number of overlapping windows. Percival (1995) details the variance properties of the overlapping Haar wavelet variance, demonstrating how naive OLS drastically underestimates CI bounds.
*   **Mitigation:** `waterSpec` uses moving block bootstrapping (via indices) or parametric Monte Carlo surrogates to estimate confidence intervals on the fit. This is mathematically necessary because standard OLS assumptions are violated. Furthermore, robust Theil-Sen regression via `MannKS.trend_test` ensures that outlier fluctuation scales do not completely bias the spectral exponent fit.

**Small-Sample Bias Correction:**
*   Standard Haar variance underestimates true variance when the number of data points per window is small. `waterSpec` implements an explicitly unbiased standard deviation estimator (`std_corrected`) utilizing correction factors derived from Gamma functions under the assumption of local normality: $\sigma_{unbiased} = \sigma_{sample} \cdot \sqrt{\frac{2}{N-1}} \exp(\Gamma(\frac{N}{2}) - \Gamma(\frac{N-1}{2}))$. This is crucial for high-frequency (small $\tau$) validity where point density per window drops, guaranteeing slopes remain undistorted."""

content = re.sub(
    r"\*\*Critical Analysis of Overlapping Windows:\*\*.*?\*\*Custom Statistics \(Percentiles & Medians\):\*\*",
    # Need to properly escape backslashes for re.sub replacement parameter
    haar_overlap_replacement.replace('\\', '\\\\') + "\n\n**Custom Statistics (Percentiles & Medians):**",
    content,
    flags=re.DOTALL
)


# 3. Add Preprocessing to 3. Discussion
preprocessing_original = r"""**The Pre-processing Dilemma:**
`waterSpec` includes tools for linear and LOESS detrending.
*   *The Trap:* Detrending removes low-frequency power. If you are analyzing a non-stationary process (e.g., groundwater levels), detrending before spectral analysis will artificially flatten the spectrum at large scales, destroying the very information you are trying to measure ($\beta > 1$). This effect is akin to applying a high-pass filter, altering the scaling behavior at low frequencies.
*   *Rule of Thumb:* Only detrend if you are strictly interested in the stationary fluctuations around a known, deterministic trend (e.g., climate change warming curve), and you are prepared to ignore the largest scales where the trend dominates."""

preprocessing_addition = r"""**Data Preprocessing, Censored Data, and the Spectral Trap:**
`waterSpec` includes robust tools (`handle_censored_data`, `detrend_loess`) to prepare messy data, but these steps fundamentally alter the spectrum.
*   *Censored Data (Non-detects):* Data like `<2.0` represents a truncated distribution. The chosen strategy (dropping, substitution, or multiplication) injects specific frequencies or destroys them. Substituting all non-detects with `0` or half the detection limit creates artificial flatlines (variance = 0), manifesting as heavily biased low-frequency artifacts. `waterSpec` uses a robust regex parser to safely translate these, but relies on the user to understand the spectral impact of the substitution strategy.
*   *The Detrending Trap:* Detrending via linear regression or LOESS removes low-frequency power. If analyzing a non-stationary process (e.g., groundwater levels where $\beta > 1$), detrending before spectral analysis artificially flattens the spectrum at large scales, destroying the very information you are trying to measure. This acts as an arbitrary high-pass filter.
*   *Rule of Thumb:* Only detrend if you are strictly interested in the stationary fluctuations around a known, deterministic trend (e.g., climate change warming curve), and are prepared to ignore the largest scales where the trend dominates."""

content = content.replace(preprocessing_original, preprocessing_addition)

with open("docs/VALIDITY_AND_METHODOLOGY_ANALYSIS.md", "w") as f:
    f.write(content)

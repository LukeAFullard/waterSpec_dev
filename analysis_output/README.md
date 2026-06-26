# Water Temperature Pukeokahu Analysis

This README contains the results of a spectral analysis on the Water Temperature data at Pukeokahu (Rangitikei river).

## Overview

The dataset was preprocessed by resolving mixed date formats and removing missing rows. Following that, it was processed with `waterSpec` to estimate spectral slopes ($eta$) and identify significant periodicities. Because the original data contains irregular gaps, a robust combination of **Haar Wavelet Fluctuation Analysis** (for spectral slope estimation and breakpoint detection) and **Lomb-Scargle** (for peak detection) was used.

## Methodological Caveat: Lomb-Scargle vs. Haar

During the analysis, the standard Lomb-Scargle (LS) periodogram detected a spectral breakpoint at roughly **13 days**. However, the Haar Fluctuation Analysis detected the primary regime shift at roughly **267 days**.

**Why is there such a massive discrepancy, and which do we trust?**

The discrepancy arises from how the two methods handle the irregular sampling and missing gaps in the Pukeokahu dataset:
1. **Lomb-Scargle Bias:** The LS periodogram fits sine waves globally across the data. When large gaps exist, high-frequency sine waves alias and distort, which often artificially flattens the high-frequency spectrum. The 13-day breakpoint from LS is almost certainly a mathematical artifact caused by these data gaps, not a real physical shift.
2. **Haar Robustness:** Haar Fluctuation Analysis operates locally in the time domain, simply averaging differences between adjacent blocks. It natively skips gaps without corrupting other scales.

Therefore, **we discard the Lomb-Scargle slope and breakpoint estimates** and rely entirely on the **Haar analysis** for determining the true scaling behavior and regime shifts. We retain Lomb-Scargle *only* for its highly accurate detection of specific periodic peaks (e.g., the annual cycle).

---

## Confidence and Methodological Limits

`waterSpec` provides built-in metrics to evaluate the reliability of the Haar method for this specific dataset:

1. **Effective Degrees of Freedom ($N_{eff}$):**
   Because Haar uses sliding windows, the number of independent samples drops at larger timescales. For this dataset, the shortest scales have $N_{eff} pprox 1355$ (excellent confidence). However, the largest scales (approaching the multi-year range) drop to $N_{eff} pprox 1.1$. This triggers an internal warning in `waterSpec` that maximum-scale variance estimates are highly uncertain due to the finite length of the dataset. To safeguard against this, the analysis uses **Weighted Least Squares (WLS)**, which correctly down-weights these unreliable large scales during slope fitting.
2. **Goodness of Fit ($R^2$):**
   The standard (single straight line) Haar fit returned an $R^2$ of **-0.98**, meaning a single slope is mathematically worse than a flat horizontal line across the spectrum. This strictly invalidates the single-slope model for this dataset. However, the Segmented regime model successfully captures the distinct "peak and drop" variance structure (rising to a peak at ~196 days before collapsing), yielding a valid weighted $R^2$ of **0.56**. This confirms that the segmented model is both statistically necessary and appropriate.

## Analysis Results and Interpretation

### Spectral Scaling and Multifractal Intermittency (Haar)

When calculating the spectral slope natively on the irregular data using the Haar First-Order Structure Function (SF), the following global parameters were found:

- **Hurst Exponent ($H$):** 0.22
- **Standard Beta ($eta$):** 1.44
- **Intermittency Correction ($K(2)$):** 0.77
- **Multifractal Corrected Beta ($eta_{multi}$):** 0.67

**Interpretation:** The standard slope ($eta pprox 1.44$) initially suggests a process leaning toward Fractional Brownian Motion (strong memory). However, the high $K(2)$ value (0.77) indicates that the time series is highly intermittent—characterized by extreme, non-Gaussian fluctuations (e.g., sudden storm events or rapid temperature drops). When correcting for this intermittency ($eta_{corrected} = 1 + 2H - K(2)$), the effective scaling drops to **$eta = 0.67$**, revealing that the underlying, steady-state process is actually closer to event-driven Fractional Gaussian Noise ($0 < eta < 1$) rather than a persistent random walk.

### Segmented Regime Fit (Haar)

The segmented regression applied to the Haar fluctuations indicates a distinct shift in process dynamics taking place around roughly **267 days** (approx. 9 months):

- **Low-Frequency (Long-term) Fit (Scales > ~267 days):**
  - $eta_1$ = -1.23
  - **Interpretation:** $eta < 0$ (Blue/Violet Noise). At very long (inter-annual) timescales, the temperature is strongly anti-persistent and bounded. The signal rapidly reverses back toward a stationary multi-year mean, meaning there is no long-term memory or drift from year to year.

- **Breakpoint:** ~266.8 days
  - **Interpretation:** This crossover makes strong physical sense. It corresponds to the transition point where short-term, persistent day-to-day weather and flow variations hit the "ceiling" of the annual solar cycle (12 months), forcing the system to become bounded and mean-reverting at scales larger than a year.

- **High-Frequency (Short-term) Fit (Scales < ~267 days):**
  - $eta_2$ = 1.91
  - **Interpretation:** $eta pprox 2$ (Brownian Noise). This indicates a strong random walk process or highly persistent behavior (storage-dominated) dominating short-term and seasonal temperature variability. Water temperatures within a given year strongly depend on the previous days/weeks' temperature, driven by thermal mass and progressive seasonal heating/cooling cycles.

### Significant Periodicities Found (Lomb-Scargle)

While LS slopes are biased by gaps, its peak detection remains robust. Significant periodic cycles were identified at a 1.0% False Alarm Probability (FAP) level:

  - **Period: 12.1 months** - Represents the primary annual (seasonal) solar and atmospheric temperature cycle.
  - **Period: 5.9 months** - Represents the semi-annual harmonic, capturing seasonal asymmetries or bi-annual shifts in river flow and temperature dynamics.

## Generated Plots

Below are the visual outputs from the analysis:

### Haar Wavelet Fluctuation Spectrum
![Haar Spectrum](Water_Temperature_haar_plot.png)

### Lomb-Scargle Periodogram
![Lomb-Scargle Spectrum](Water_Temperature_spectrum_plot.png)

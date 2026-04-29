# Haar Wavelet Analysis Guide\n\n## From HAAR_ANALYSIS_GUIDE.md

# Comprehensive Guide to Haar Analysis in waterSpec

This guide explains the advanced features of the Haar Analysis module in `waterSpec`, focusing on statistical aggregation methods and multifractal intermittency corrections.

## 1. Aggregation Methods

The `calculate_haar_fluctuations` function (and the `HaarAnalysis` class) supports different methods for aggregating fluctuations within a time window. This choice affects the robustness and statistical properties of the estimated spectral slope.

### Available Methods

You can select the method using the `aggregation` parameter:

```python
ha = HaarAnalysis(time, data)
ha.run(aggregation="mean", ci_level=95.0) # Default
```

| Method | Description | Use Case |
| :--- | :--- | :--- |
| **"mean"** | Mean Absolute Fluctuation. Calculates $\langle | \Delta f | \rangle$. | **Default.** Robust, distribution-agnostic. Best for general use. |
| **"std_corrected"** | Small-Sample Corrected Standard Deviation (converted to MAD). Matches `GapWaveSpectra`. | **Gaussian Data.** Best for short time series where small-sample bias is a concern, provided the data is roughly Gaussian. |
| **"rms"** | Root Mean Square Fluctuation. Calculates $\sqrt{\langle \Delta f^2 \rangle}$. | **Higher Moments.** Used internally for intermittency calculations ($S_2$). |
| **"median"** | Median Absolute Fluctuation. | **Outliers.** Highly robust to spikes/outliers. |

### The "std_corrected" Method

This method is implemented to match the statistical approach of the `GapWaveSpectra` reference project.

1.  **Zero-Mean Enforcement:** It concatenates the fluctuations $\Delta f$ with their negatives $-\Delta f$ to ensure a zero-mean distribution.
2.  **Unbiased Estimator:** It calculates the sample standard deviation $s$ and applies a correction factor $c_4(N)$ derived from the Gamma function to obtain an unbiased estimate of the population standard deviation $\sigma$.
    $$ \hat{\sigma} = \frac{s}{c_4(N)} $$
3.  **Conversion to MAD:** It assumes the fluctuations follow a Gaussian distribution and converts the estimated $\sigma$ to the equivalent Mean Absolute Deviation (MAD):
    $$ S_1 \approx \hat{\sigma} \sqrt{\frac{2}{\pi}} $$

**When to use:**
*   When comparing results directly with `GapWaveSpectra`.
*   When analyzing short time series where the bias of the sample mean/std is significant.
*   **Caution:** Requires the fluctuations (differences of window means) to be approximately Gaussian. Thanks to the Central Limit Theorem, this holds for most finite-variance processes at window sizes $N \ge 5$.

---

## 2. Multifractal Intermittency Correction ($K(2)$)

Standard spectral analysis assumes the process is **monofractal**, meaning a single scaling exponent $H$ describes the entire process. The spectral slope $\beta$ is then related to $H$ by:

$$ \beta_{mono} = 1 + 2H $$

However, many environmental processes (e.g., rainfall, turbulence, sediment transport) are **multifractal**, exhibiting **intermittency** (burstiness). In these cases, the monofractal assumption overestimates the spectral slope.

The **Universal Multifractal** relation accounts for this using the intermittency correction $K(2)$:

$$ \beta_{multi} = 1 + 2H - K(2) $$

Where $K(2)$ is the codimension of the second moment (variance), defined as:

$$ K(2) = 2\zeta(1) - \zeta(2) $$

*   $\zeta(1) = H$: Scaling exponent of the first moment (Mean Absolute Fluctuation).
*   $\zeta(2)$: Scaling exponent of the second moment (Variance/RMS).

### How to Use

Enable the calculation by setting `calc_intermittency=True` in the `run` method:

```python
ha = HaarAnalysis(time, data)
results = ha.run(calc_intermittency=True, ci_level=95.0)

print(f"Monofractal Beta: {results['beta']}")
print(f"Multifractal Beta: {results['beta_multifractal']}")
print(f"Intermittency K(2): {results['K2']}")
```

### Interpretation

*   **$K(2) \approx 0$:** The process is effectively monofractal. The standard $\beta$ is accurate.
*   **$K(2) > 0$:** The process is intermittent (multifractal). The standard $\beta$ is likely an overestimate. Use `beta_multifractal`.

### Validation

Validation on simulated multifractal random walks (log-normal cascades) shows that this correction significantly reduces error. For example, for a process with true $\beta \approx 1.8$, the standard method might estimate $\beta \approx 2.3$, while the corrected method yields $\beta \approx 2.0$.

---

## 3. Segmented vs. Multifractal Analysis

It is important to distinguish between two types of "scaling complexity" handled by `waterSpec`:

### Segmented Analysis (Scale Breaks)
*   **What it detects:** A change in the scaling exponent $\beta$ at a specific timescale (breakpoint).
*   **Example:** A watershed acts as a fractal filter ($\beta \approx 0.5$) at short scales (< 1 day) but exhibits long-term memory ($\beta \approx 1.5$) at seasonal scales.
*   **Tool:** `HaarAnalysis.run(max_breakpoints=1)` or `fit_segmented_haar`.
*   **Visual:** The log-log plot of $S_1$ vs $\Delta t$ is a bent line (two linear segments).

### Multifractal Analysis (Intermittency)
*   **What it detects:** A non-linear relationship between the scaling exponents of different moments ($q$).
*   **Example:** Turbulence or rainfall where intense events scale differently than mean events.
*   **Tool:** `HaarAnalysis.run(calc_intermittency=True)`.
*   **Visual:** The log-log plot of $S_1$ vs $\Delta t$ is straight, but the slope $H$ does not predict the spectral slope $\beta$ using the simple linear formula. The "curvature" is in the $\zeta(q)$ vs $q$ plot, not the $S_1$ plot.

**Summary:**
*   Use **Segmented** analysis if the physics changes across *time scales*.
*   Use **Multifractal** correction if the physics involves *intermittent bursts* across all scales.
\n\n## From HAAR_IMPLEMENTATION_DETAILS.md

# Detailed Implementation of Haar Methods in waterSpec

This document provides a technical deep-dive into the Haar Wavelet methods implemented in the `waterSpec` package.

## 1. Overview

The Haar Structure Function (HSF) method is the primary Haar-based technique in `waterSpec`. It is used to estimate the spectral slope ($\beta$) of a time series, especially when the data is irregularly sampled or contains large gaps.

The package currently implements the **First-Order Haar Structure Function ($S_1$)**.

## 2. Mathematical Foundation

### 2.1 The Haar Fluctuation ($\Delta F$)

The basic building block is the Haar fluctuation, which measures the difference in central tendency between two adjacent windows of time.

For a time scale (lag) $\tau$, we consider an interval $[t, t + \tau]$. This interval is split into two equal halves:
- Left half: $L = [t, t + \tau/2)$
- Right half: $R = [t + \tau/2, t + \tau]$

The Haar fluctuation $D(t, \tau)$ is defined as:
$$D(t, \tau) = \bar{x}_R - \bar{x}_L$$
where $\bar{x}_R$ and $\bar{x}_L$ are the means of the data points falling within the right and left half-windows, respectively.

### 2.2 The First-Order Structure Function ($S_1$)

The first-order structure function is the average of the absolute fluctuations across the entire time series:
$$S_1(\tau) = \langle |D(t, \tau)| \rangle_t$$

### 2.3 Scaling and Spectral Slope

In a process with power-law scaling (colored noise), the structure function follows:
$$S_1(\tau) \propto \tau^m$$
where $m$ is the fluctuation scaling slope.

The spectral slope $\beta$ (from $P(f) \propto f^{-\beta}$) is related to $m$ by:
$$\beta = 2m + 1$$

*Note: In `waterSpec`, we use $S_1$ directly. For a white noise process ($\beta=0$), $m \approx -0.5$. for a pink noise process ($\beta=1$), $m \approx 0$. For Brownian motion ($\beta=2$), $m \approx 0.5$.*

## 3. Implementation Details

The implementation is located in `src/waterSpec/haar_analysis.py`.

### 3.1 Handling Irregular Sampling

The function `calculate_haar_fluctuations` robustly handles irregular sampling using the following algorithm:

1.  **Lag Generation**: A set of lag times $\tau$ is generated (usually logarithmically spaced).
2.  **Window Search**: For each $\tau$, the algorithm slides through the time series.
3.  **Non-overlapping Selection**: It finds a starting point $t$, then uses `numpy.searchsorted` to quickly find data points in $[t, t+\tau/2)$ and $[t+\tau/2, t+\tau)$.
4.  **Window Validity**: A fluctuation is only calculated if **both** half-windows contain at least one data point.
5.  **Iteration**: After a successful calculation for $[t, t+\tau]$, the algorithm jumps to the first data point $\ge t+\tau$ to ensure that the fluctuations are calculated from non-overlapping segments. If a window is invalid (missing data), it moves to the next available data point and tries again.

### 3.2 Fitting

The function `fit_haar_slope` performs a robust linear regression on $\log_{10}(S_1)$ vs $\log_{10}(\tau)$ using the Mann-Kendall/Theil-Sen estimator (via the `MannKS` library). This provides a more reliable estimate of the scaling slope $m$ that is less sensitive to outliers. It returns $m$, $\beta$, $R^2$ (calculated via OLS for reference), and the intercept.

## 4. Usage in waterSpec

### 4.1 Standalone Usage

```python
from waterSpec.haar_analysis import HaarAnalysis

haar = HaarAnalysis(time, data, time_unit="days")
results = haar.run(num_lags=30, ci_level=95.0)
print(f"Beta: {results['beta']}")
haar.plot(output_path="haar_plot.png")
```

### 4.2 Integrated Usage

In the `Analysis` class, Haar analysis can be enabled by setting `run_haar=True`:

```python
analyzer = Analysis(...)
results = analyzer.run_full_analysis(output_dir="output", run_haar=True)
# Haar results are stored in results['haar_results']
```

### 4.3 Segmented Haar Analysis

While the `HaarAnalysis` class currently only performs linear fitting, you can manually perform segmented fitting on the Haar structure function using the package's fitter:

```python
from waterSpec.haar_analysis import HaarAnalysis
from waterSpec.fitter import fit_segmented_spectrum

haar = HaarAnalysis(time, data)
res = haar.run()

# Fit a segmented model to the structure function
# fit_segmented_spectrum also uses robust regression internally
segmented_fit = fit_segmented_spectrum(res['lags'], res['s1'], n_breakpoints=1, ci=95.0)
```
\n\n## From haar_method_description.md

# Statistical Description of the Haar Structure Function Method for Beta Estimation

## Overview

The Haar Structure Function (HSF) method is a robust technique for estimating the spectral slope ($\beta$) of a time series, particularly suited for data that is short, irregularly sampled, or non-stationary. This method, advocated by Lovejoy and Schertzer and applied in recent paleoclimate studies (e.g., Hébert et al., 2021), operates in the time domain using Haar wavelets, avoiding many of the artifacts associated with Fourier-based spectral analysis on sparse data (such as fractal aliasing).

## Mathematical Formulation

The core of the method is the calculation of the first-order structure function, $S_1(\Delta t)$, which measures the average magnitude of fluctuations at different time scales (lags), $\Delta t$.

### 1. Haar Fluctuation ($\Delta F$)

For a given time interval $\Delta t$, the Haar fluctuation is defined as the difference between the mean of the signal in the second half of the interval and the mean of the signal in the first half.

Let $C(t)$ be the time series concentration (or value) at time $t$. The fluctuation over an interval $[t, t + \Delta t]$ is:

$$
\Delta F(\Delta t) = \overline{C}_{(t+\Delta t/2, t+\Delta t)} - \overline{C}_{(t, t+\Delta t/2)}
$$

Where $\overline{C}_{(a, b)}$ denotes the mean value of the data points falling within the time interval $(a, b)$.

**Note on Scaling:**
In some formulations, the fluctuation is defined as a derivative approximation ($\frac{\Delta \overline{C}}{\Delta t}$). However, for the purpose of estimating the Haar scaling slope $m$ consistent with $1/f^\beta$ noise (where white noise corresponds to $m=-0.5$), we utilize the difference of means directly. Dividing by $\Delta t$ would shift the exponent by -1, leading to inconsistent interpretation of standard noise colors.

### 2. The Structure Function ($S_1$)

The first-order structure function is the average of the absolute values of these fluctuations over the entire time series:

$$
S_1(\Delta t) = \langle |\Delta F(\Delta t)| \rangle
$$

For irregular data, this average is computed by identifying all available non-overlapping pairs of intervals of duration $\Delta t/2$ and computing the difference of their means. Our implementation uses a sliding window approach that maximizes data usage while ensuring that each calculated fluctuation represents a distinct, non-overlapping segment locally (though the search for segments scans the whole series).

### 3. Fractal Scaling and Haar Slope $m$

In fractal processes, the structure function follows a power law scaling relationship with the time lag:

$$
S_1(\Delta t) \propto \Delta t^{m}
$$

By plotting $\log(S_1)$ against $\log(\Delta t)$, the fluctuation slope $m$ can be estimated as the slope of the linear fit.

## Relation to Spectral Slope ($\beta$)

The slope $m$ derived from the Haar analysis is directly related to the power spectral density slope $\beta$ (where $P(f) \propto f^{-\beta}$) by the following relation:

$$
\beta = 2m + 1
$$

### Note on Haar Slope $m$ vs. Hurst Exponent $H$
It is common to see the relationship $\beta = 2H - 1$ (for stationary noise) or $\beta = 2H + 1$ (for non-stationary motion). Our measured slope $m$ unifies these:
*   For non-stationary processes (fBm), $m = H$.
*   For stationary processes (fGn), $m = H - 1$.
*   See [Spectral Slope vs. Hurst](spectral_slope_vs_hurst.md) for a detailed explanation.

### Interpretation of Regimes

The value of $\beta$ (and consequently $m$) provides insight into the "color" or memory of the noise process:

| Noise Type | Beta ($\beta$) | Haar Slope ($m$) | Description |
| :--- | :--- | :--- | :--- |
| **White Noise** | $\approx 0$ | $\approx -0.5$ | No correlation; memoryless process. |
| **Pink Noise** | $\approx 1$ | $\approx 0$ | $1/f$ noise; long-range dependence. |
| **Brownian Noise** | $\approx 2$ | $\approx 0.5$ | Random walk; integrated white noise. |
| **Black Noise** | $> 2$ | $> 0.5$ | Strong persistence/trends. |

## Advantages for Environmental Data

1.  **Robustness to Gaps:** Unlike the Fast Fourier Transform (FFT), which requires evenly spaced data, the HSF method naturally handles gaps. It simply skips intervals where data is missing, calculating statistics only on valid segments.
2.  **Short Time Series:** Spectral methods often become unstable or yield high variance for short records ($N < 100$). The HSF method provides a more stable estimate of the scaling behavior by averaging fluctuations in the time domain.
3.  **Stationarity:** The method can effectively distinguish between stationary ($m < 0$) and non-stationary ($m > 0$) regimes, a distinction that is often blurred in periodogram analysis.

## Implementation Details

Our implementation (`src/waterSpec/haar_analysis.py`) performs the following steps:
1.  **Lag Generation:** Generates a sequence of logarithmically spaced lag times ($\Delta t$) from the minimum resolution up to half the series duration.
2.  **Fluctuation Calculation:** For each $\Delta t$, iterates through the time series to find valid windows $[t, t+\Delta t/2)$ and $[t+\Delta t/2, t+\Delta t)$ containing data.
3.  **Averaging:** Computes the mean difference for each valid window and averages their absolute values.
4.  **Fitting:** Performs a linear regression on the log-log data to determine $m$ and $\beta$.
\n\n
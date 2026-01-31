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
$$S_1(\tau) \propto \tau^H$$
where $H$ is the fluctuation scaling exponent.

The spectral slope $\beta$ (from $P(f) \propto f^{-\beta}$) is related to $H$ by:
$$\beta = 1 + 2H$$

*Note: In `waterSpec`, we use $S_1$ directly. For a white noise process ($\beta=0$), $H \approx -0.5$. for a pink noise process ($\beta=1$), $H \approx 0$. For Brownian motion ($\beta=2$), $H \approx 0.5$.*

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

The function `fit_haar_slope` performs a linear regression on $\log(S_1)$ vs $\log(\tau)$ using Ordinary Least Squares (`numpy.polyfit`). It returns $H$, $\beta$, and the $R^2$ of the fit.

## 4. Comparison with MSDA Framework

The technical report `Haar_SF_New_Analysis_Methods.md` describes a "Multi-Scale Diagnostic Attribution" (MSDA) framework. It is important to note the differences between the *current implementation* and the *proposed MSDA specification*:

| Feature | Current Implementation | MSDA Specification |
| :--- | :--- | :--- |
| **Structure Function Order** | $S_1$ only | $S_q$ (specifically $S_2$) |
| **Mean Calculation** | Standard Arithmetic Mean | Time-Weighted Mean |
| **Window Validity** | $\ge 1$ point in each half | $\ge 5$ points AND $\ge 70\%$ time coverage |
| **Robustness** | Median NOT used for scaling | $\widetilde{S}_2$ (Median of squared fluctuations) |
| **Intermittency** | Not implemented | $I(\tau) = \log_{10}(S_2 / \widetilde{S}_2)$ |
| **Regression** | OLS (`polyfit`) | Robust (Theil-Sen) |
| **Uncertainty** | Not implemented in `HaarAnalysis.run` | Block Bootstrap |

## 5. Usage in waterSpec

### 5.1 Standalone Usage

```python
from waterSpec.haar_analysis import HaarAnalysis

haar = HaarAnalysis(time, data, time_unit="days")
results = haar.run(num_lags=30)
print(f"Beta: {results['beta']}")
haar.plot(output_path="haar_plot.png")
```

### 5.2 Integrated Usage

In the `Analysis` class, Haar analysis can be enabled by setting `run_haar=True`:

```python
analyzer = Analysis(...)
results = analyzer.run_full_analysis(output_dir="output", run_haar=True)
# Haar results are stored in results['haar_results']
```

### 5.3 Segmented Haar Analysis

While the `HaarAnalysis` class currently only performs linear fitting, you can manually perform segmented fitting on the Haar structure function using the package's fitter:

```python
from waterSpec.haar_analysis import HaarAnalysis
from waterSpec.fitter import fit_segmented_spectrum

haar = HaarAnalysis(time, data)
res = haar.run()

# Fit a segmented model to the structure function
segmented_fit = fit_segmented_spectrum(res['lags'], res['s1'], n_breakpoints=1)
```

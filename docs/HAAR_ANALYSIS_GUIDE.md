# Comprehensive Guide to Haar Analysis in waterSpec

This guide explains the advanced features of the Haar Analysis module in `waterSpec`, focusing on statistical aggregation methods and multifractal intermittency corrections.

## 1. Aggregation Methods

The `calculate_haar_fluctuations` function (and the `HaarAnalysis` class) supports different methods for aggregating fluctuations within a time window. This choice affects the robustness and statistical properties of the estimated spectral slope.

### Available Methods

You can select the method using the `aggregation` parameter:

```python
ha = HaarAnalysis(time, data)
ha.run(aggregation="mean") # Default
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
results = ha.run(calc_intermittency=True)

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

# Analyzing Extremes with Haar Percentiles

While traditional spectral analysis focuses on the mean behavior of a system (variance/power), environmental systems often exhibit different scaling behaviors for their extremes. For example, peak flows (high percentiles) might scale differently than base flows (median or low percentiles) due to different underlying physical drivers (e.g., surface runoff vs. groundwater).

`waterSpec` allows you to probe these behaviors by calculating Haar fluctuations on specific percentiles, rather than just the mean.

## Theory: Why Percentiles?

The standard Haar fluctuation for a time lag $\Delta t$ is defined as the difference in **means** between two adjacent windows of size $\Delta t/2$:
$$ \Delta F(\Delta t) = | \bar{X}_{right} - \bar{X}_{left} | $$

By replacing the mean ($\bar{X}$) with a percentile ($P_{q}$), we can define a fluctuation of extremes:
$$ \Delta F_q(\Delta t) = | P_{q, right} - P_{q, left} | $$

If the system is **Gaussian**, all percentiles scale with the same exponent $\beta$ as the mean. However, for **Multi-fractal** or non-Gaussian systems, different percentiles may scale with different exponents. This divergence indicates complex, intermittent behavior often found in turbulent flows or flashy hydrologic catchments.

## Usage Guide

You can perform this analysis using the standard `Analysis` class by specifying the `haar_statistic`, `haar_percentile`, and `haar_percentile_method` arguments.

### Step 1: Load Data

```python
import numpy as np
from waterSpec import Analysis

# Example: Generate synthetic data (replace with your data loading)
time = np.arange(1000)
data = np.random.randn(1000) # Gaussian white noise

analyzer = Analysis(
    time_array=time,
    data_array=data,
    data_col="Value",
    time_col="Time"
)
```

### Step 2: Run Analysis on the 95th Percentile

To study the scaling of high extremes (e.g., the 95th percentile), set `haar_statistic="percentile"` and `haar_percentile=95`.

We recommend using the **Hazen** plotting position method (`haar_percentile_method="hazen"`) for environmental data, as it provides a neutral estimate of percentiles regardless of distribution shape, though `linear`, `weibull`, and others supported by `numpy.percentile` are available.

```python
results_95 = analyzer.run_full_analysis(
    output_dir="output_95th",
    run_haar=True,
    haar_statistic="percentile",
    haar_percentile=95,
    haar_percentile_method="hazen"
)

print(f"95th Percentile Beta: {results_95['haar_results']['beta']:.2f}")
```

### Step 3: Compare with the Mean and Median

To detect multi-fractality or non-Gaussian scaling, compare the slopes ($\beta$) of the mean, median, and extremes.

```python
# Mean (Standard)
results_mean = analyzer.run_full_analysis(
    output_dir="output_mean",
    run_haar=True,
    haar_statistic="mean"
)

# Median (Robust to outliers)
results_median = analyzer.run_full_analysis(
    output_dir="output_median",
    run_haar=True,
    haar_statistic="median"
)

print(f"Beta (Mean):   {results_mean['haar_results']['beta']:.2f}")
print(f"Beta (Median): {results_median['haar_results']['beta']:.2f}")
print(f"Beta (95th):   {results_95['haar_results']['beta']:.2f}")
```

## Interpretation

| Result | Interpretation |
| :--- | :--- |
| $\beta_{95} \approx \beta_{mean}$ | **Mono-fractal / Gaussian:** The extremes scale identically to the average. The process is likely dominated by a single scaling mechanism (e.g., simple diffusion). |
| $\beta_{95} > \beta_{mean}$ | **Enhanced Persistence in Extremes:** Extreme events (peaks) are more persistent (have longer memory) than the average behavior. |
| $\beta_{95} < \beta_{mean}$ | **Whitening of Extremes:** Extreme events are more random/uncorrelated than the average. This is common in systems with flashy, short-duration spikes (e.g., surface runoff) superimposed on a slower base signal. |

## Notes

*   **Data Requirements:** Percentile estimation requires sufficient data points within each window. The default `min_samples_per_window` is 5, but for high percentiles (e.g., 95th, 99th), you may need larger windows to get stable estimates. The code will automatically skip windows with insufficient data.
*   **Computational Cost:** Calculating percentiles is computationally more expensive than calculating means. For very large datasets, this analysis may take longer.

## Case Study: Stochastic Volatility

A classic example of divergent scaling is a **Stochastic Volatility** process, often used to model financial markets or turbulent flows. In this model, the signal is "uncorrelated" white noise, but its variance (volatility) changes over time with a persistent memory.

$$ y_t = \epsilon_t \cdot \sigma_t $$

where $\epsilon_t$ is white noise ($\beta \approx 0$) and $\log \sigma_t$ is a persistent process (e.g., $\beta \approx 1.5$).

### Running the Demo

`waterSpec` includes a demonstration script for this scenario: `examples/demo_stochastic_volatility.py`.

```bash
python examples/demo_stochastic_volatility.py
```

### Expected Output

When running this analysis, you will see that the **mean** fluctuation scales like white noise, while the **90th percentile** tracks the persistent volatility structure.

```text
Running Haar Analysis (Mean)...
-> Beta (Mean): 0.14 (Expected ~0.0 for white noise structure)

Running Haar Analysis (90th Percentile)...
-> Beta (90th): 1.24 (Expected >0.0, tracking volatility)
```

This confirms that examining percentiles can reveal hidden structure in data that appears random to standard spectral analysis.

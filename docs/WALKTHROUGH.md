# waterSpec: Step-by-Step Walkthrough

This document provides a comprehensive guide to using the advanced features of `waterSpec`, including Haar fluctuation analysis, segmented regression, bivariate analysis, and rigorous statistical validation methods.

Each section includes a self-contained Python code example.

---

## 1. Haar Analysis with Overlapping Windows

**Scientific Context:**
Haar analysis quantifies how the variability of a time series changes with time scale ($\tau$). For short or irregular records, overlapping windows maximize the statistical power.

**Example Code:**

```python
import numpy as np
from waterSpec.haar_analysis import HaarAnalysis

# 1. Generate synthetic Fractional Brownian Motion (fBm)
np.random.seed(42)
n_points = 500
time = np.arange(n_points)
data = np.cumsum(np.random.randn(n_points))

# 2. Run analysis
haar = HaarAnalysis(time, data, time_unit="days")
results = haar.run(min_lag=2, max_lag=100, overlap=True, overlap_step_fraction=0.1)

print(f"Estimated Spectral Slope (beta): {results['beta']:.2f}")
# Expected: Beta approx 2.0
```

---

## 2. Detecting Regime Shifts (Segmented Haar Fits)

**Scientific Context:**
Systems often exhibit different memory behaviors at different scales (e.g., runoff dominated at short scales vs. groundwater at long scales).

**Example Code:**

```python
import numpy as np
from waterSpec.haar_analysis import HaarAnalysis

# 1. Generate a "Regime Shift" signal
time = np.arange(1000)
noise = np.random.randn(1000) # Short scales
trend = 0.05 * np.cumsum(np.random.randn(1000)) # Long scales
data = noise + trend

haar = HaarAnalysis(time, data, time_unit="hours")

# 2. Run Segmented Analysis
results = haar.run(min_lag=2, max_lag=200, overlap=True, max_breakpoints=1)
seg = results['segmented_results']

print(f"Breakpoint: {seg['breakpoints'][0]:.1f} hours")
print(f"Slopes (H): {seg['Hs']}")
```

---

## 3. Bivariate Analysis (Concentration-Discharge)

**Scientific Context:**
Analyzes the correlation of fluctuations ($\Delta C$ vs $\Delta Q$) across scales.

**Example Code:**

```python
import numpy as np
from waterSpec.bivariate import BivariateAnalysis

# 1. Generate Synthetic Data
time_q = np.arange(0, 365, 1.0)
Q = np.exp(np.sin(time_q/10) + np.random.normal(0, 0.2, len(time_q)))
time_c = np.arange(0, 365, 14.0)
C = 0.5 * np.interp(time_c, time_q, Q) + np.random.normal(0, 0.1, len(time_c))

# 2. Run Analysis
biv = BivariateAnalysis(time_c, C, "Nitrate", time_q, Q, "Discharge", time_unit="days")
biv.align_data(tolerance=1.0)
cross_results = biv.run_cross_haar_analysis(lags=np.array([14, 30]), overlap=True)

print(f"Correlations: {cross_results['correlation']}")
```

---

## 4. Hysteresis Analysis

**Scientific Context:**
Quantifies the loop direction (clockwise vs counter-clockwise) in C-Q relationships at specific scales.

**Example Code:**

```python
# (Continuing from Bivariate setup)
hyst_stats = biv.calculate_hysteresis_metrics(tau=10.0)
print(f"Loop Area: {hyst_stats['area']:.4f}")
```

---

## 5. Real-Time Anomaly Detection (Sliding Haar)

**Scientific Context:**
Detects sudden changes in system volatility (variance) rather than just mean shifts.

**Example Code:**

```python
from waterSpec.haar_analysis import calculate_sliding_haar

time = np.arange(200)
data = np.random.normal(0, 0.5, 200)
data[100:120] = np.random.normal(0, 2.5, 20) # Volatility burst

t_centers, fluctuations = calculate_sliding_haar(time, data, window_size=10.0)
```

---

## 6. Lagged Response Analysis

**Scientific Context:**
Identifies the characteristic delay time between a driver (e.g., Q) and response (e.g., C).

**Example Code:**

```python
# (Using BivariateAnalysis object)
res = biv.run_lagged_cross_haar(tau=20.0, offsets=np.arange(0, 15, 1.0), overlap=True)
print(f"Peak Lag: {res['lag_offsets'][np.argmax(res['correlation'])]} days")
```

---

## 7. Partial Cross-Haar Analysis

**Scientific Context:**
Distinguishes direct correlation from spurious correlation driven by a third variable (e.g., Rain).

**Example Code:**

```python
from waterSpec import calculate_partial_cross_haar

# ... (Generate time, C, Q, Rain arrays) ...
# results = calculate_partial_cross_haar(time, C, Q, Rain, lags=..., overlap=True)
```

---

## 8. Event-Based Segmentation

**Scientific Context:**
Separates "storm event" behavior from "baseflow" behavior using volatility thresholds.

**Example Code:**

```python
from waterSpec import SegmentedRegimeAnalysis
# results = SegmentedRegimeAnalysis.segment_by_fluctuation(time, data, scale=6.0, threshold_factor=3.0)
```

---

## 9. Weighted Wavelet Z-Transform (WWZ)

**Scientific Context:**
Standard spectral methods (Lomb-Scargle) provide a *global* average of periodicities. However, environmental signals are often transient (e.g., a diurnal cycle that only appears during summer low flows). **WWZ** is a time-frequency method specifically designed for **irregularly sampled data**. It maps *when* specific frequencies are active.

**Example Code:**

```python
import numpy as np
from waterSpec.wwz import calculate_wwz

# 1. Generate a transient signal (sine wave only in the middle)
time = np.sort(np.random.uniform(0, 100, 200)) # Irregular sampling
signal = np.zeros_like(time)
mask = (time > 30) & (time < 70)
signal[mask] = np.sin(2 * np.pi * 0.5 * time[mask]) # 0.5 Hz signal
data = signal + 0.1 * np.random.randn(200)

# 2. Run WWZ
freqs = np.linspace(0.1, 1.0, 50)
taus = np.linspace(0, 100, 100) # Time points to evaluate

wwz_matrix, fs, ts = calculate_wwz(
    time, data, freqs, taus=taus, decay_constant=0.005
)

# 3. Interpret
# wwz_matrix[i, j] is the "Z-statistic" power at freq[i] and time[j]
# High Z (> 20-50) indicates significant periodicity.
max_idx = np.unravel_index(np.nanargmax(wwz_matrix), wwz_matrix.shape)
print(f"Peak Frequency: {fs[max_idx[0]]:.2f} Hz")
print(f"Peak Time: {ts[max_idx[1]]:.1f}")
```

**Interpretation:**
*   **Localization:** The output matrix allows you to plot a "heatmap" (Spectrogram) of Power vs. Time.
*   **Defensibility:** In a legal context, this proves that a signal coincides exactly with a specific time window (e.g., "The 24-hour cycle only appeared while the treatment plant was offline").

---

## 10. Statistical Validation (PSRESP)

**Scientific Context:**
A major criticism of spectral analysis on irregular data is "aliasing" or "spectral leakage"—fake peaks caused by the sampling gaps (e.g., weekly sampling creating a fake monthly cycle). **PSRESP (Power Spectral Response)** uses forward modeling (Monte Carlo simulation) to prove that your detected peaks are NOT artifacts of the sampling schedule.

**Example Code:**

```python
from waterSpec import Analysis

# 1. Initialize Analysis
# Assume we have irregular data loaded
analyzer = Analysis(file_path="data.csv", time_col="time", data_col="value")

# 2. Run with Validation
# Setting validate_model=True triggers the PSRESP check.
# It fits a model (e.g. Power Law), simulates 1000 synthetic datasets
# with the SAME irregular timestamps, and compares them to your data.
results = analyzer.run_full_analysis(
    output_dir="output",
    validate_model=True,
    n_bootstraps=1000
)

# 3. Check Success Fraction
# This is essentially a p-value for the entire spectral fit.
sf = results.get("psresp_success_fraction", 0.0)
print(f"Model Success Fraction: {sf:.3f}")
```

**Interpretation:**
*   **Success Fraction > 0.1:** The model (e.g., simple Red Noise) explains the data well. Any "peaks" you see are likely consistent with random noise given your sampling gaps.
*   **Success Fraction < 0.01:** The model is rejected. This is GOOD if you are claiming there is a specific periodic signal (like a diurnal cycle) that the background noise model cannot explain. It strengthens the claim that the signal is "real" and not an artifact.

---

## 11. Red Noise (AR1) Null Hypothesis

**Scientific Context:**
Standard tests often check if a signal is different from "White Noise" (random scatter). But nature has memory (persistence)—today is correlated with yesterday. This "Red Noise" can create fake long-term trends. Testing against a **Red Noise (AR1)** null hypothesis provides a much stricter and more defensible bar for statistical significance.

**Example Code:**

```python
import numpy as np
from waterSpec.utils_sim import simulate_tk95, red_noise_psd

# 1. Define AR1 Parameters
# tau = Decorrelation time (memory length)
# variance = Total variance
tau = 5.0 # e.g., 5 days
variance = 1.0

# 2. Generate a Red Noise realization
time, red_noise = simulate_tk95(
    red_noise_psd,
    params=(tau, variance),
    N=1000,
    dt=1.0,
    seed=42
)

# 3. Use this for validation (e.g. in PSRESP)
# (Advanced usage: Pass this function/params to psresp_fit)
```

**Interpretation:**
*   **Why it matters:** If you claim a "significant trend" or "cycle", an opposing expert might say "that's just natural persistence." By simulating Red Noise that has the *same* persistence ($\tau$) as your data and showing your signal is *still* an outlier, you refute that counter-argument.

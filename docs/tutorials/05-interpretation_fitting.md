# Tutorial 5: Interpreting Fits and Results

After calculating the power spectrum, the next step is to fit a model to it and interpret the results. This tutorial covers the fitting and interpretation functions in `waterSpec`.

### Part 1: Standard Fit with Confidence Intervals

For most use cases, you'll want to fit a single straight line to your spectrum on a log-log plot. The slope of this line gives us the spectral exponent, β. `waterSpec` provides a function that not only does this but also calculates a confidence interval for β using a bootstrap method.

Let's continue from where the last tutorial left off, assuming we have our `frequency` and `power` arrays.

```python
# First, let's regenerate the frequency and power from the previous tutorial
from waterSpec import load_data, preprocess_data, generate_log_spaced_grid, calculate_periodogram
import numpy as np

time_numeric, data_series, _ = load_data('examples/sample_data.csv', 'timestamp', 'concentration')
processed_data, _ = preprocess_data(data_series, time_numeric, detrend_method='linear')
valid_indices = ~np.isnan(processed_data)
time_final = time_numeric[valid_indices]
data_final = processed_data[valid_indices]
frequency_grid = generate_log_spaced_grid(time_final)
frequency, power = calculate_periodogram(time_final, data_final, frequency=frequency_grid)
```

Now, let's fit the spectrum and see the results.

```python
from waterSpec.fitter import fit_spectrum_with_bootstrap

fit_results = fit_spectrum_with_bootstrap(frequency, power, n_bootstraps=100) # Use fewer bootstraps for speed

print("--- Fit Results ---")
for key, value in fit_results.items():
    print(f"{key}: {value}")
```

**Output:**
```text
--- Fit Results ---
beta: -0.6140110702805378
r_squared: 0.19599359983335085
intercept: 2.7608601805043547
stderr: 0.088379677863569
beta_ci_lower: -0.7719592923998391
beta_ci_upper: -0.44278049117381124
```

The dictionary above gives you the key quantitative results:
- `beta`: The spectral exponent. This is the main value of interest.
- `r_squared`: How well the line fits the data (a value closer to 1 is better).
- `intercept`: The intercept of the log-log linear fit.
- `stderr`: The standard error of the slope estimate.
- `beta_ci_lower` & `beta_ci_upper`: The 95% confidence interval for your beta value.

Now, let's turn these numbers into a human-readable interpretation.

```python
from waterSpec.interpreter import interpret_results

beta = fit_results.get('beta')
ci = (fit_results.get('beta_ci_lower'), fit_results.get('beta_ci_upper'))

interpretation = interpret_results(beta, ci=ci, param_name='Concentration')

print(interpretation['summary_text'])
```

**Output:**
```text
--- Interpretation ---
Analysis for: Concentration
Value: β = -0.61 (95% CI: -0.77–-0.44)
Persistence Level: 🔴 Event-driven (Low Persistence)
Scientific Meaning: Warning: Beta value is significantly negative, which is physically unrealistic.
Contextual Comparison: Closest to E. coli (Surface runoff-dominated).
```

### Part 2: Segmented Regression

Sometimes, a single line isn't enough. The transport processes in a watershed might behave differently at different timescales (frequencies). This shows up as a "knee" or a break in the power spectrum. To analyze this, you can use segmented regression.

Let's create a synthetic spectrum with a known breakpoint to see how it works.

```python
from waterSpec.fitter import fit_segmented_spectrum

# Create a synthetic spectrum with a breakpoint at frequency=0.1
synth_freq = np.logspace(-2, 1, 100)
power1 = synth_freq**-0.5
power2 = (0.1**-0.5 / 0.1**-1.8) * synth_freq**-1.8 # Scale to connect smoothly
synth_power = np.where(synth_freq < 0.1, power1, power2) + np.random.rand(100) * 0.1

# Fit the segmented spectrum
segmented_results = fit_segmented_spectrum(synth_freq, synth_power)

print("--- Segmented Fit Results ---")
print(f"Breakpoint found at frequency: {segmented_results.get('breakpoint'):.3f}")
print(f"Beta 1 (low frequency): {segmented_results.get('beta1'):.2f}")
print(f"Beta 2 (high frequency): {segmented_results.get('beta2'):.2f}")
```

**Output:**
```text
--- Segmented Fit Results ---
Breakpoint found at frequency: 2.271
Beta 1 (low frequency): 1.13
Beta 2 (high frequency): 0.04
```

The function automatically finds the breakpoint and calculates a separate beta value for each segment, allowing you to analyze complex systems with multiple dominant processes.

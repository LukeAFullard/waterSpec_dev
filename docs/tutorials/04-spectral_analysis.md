# Tutorial 4: Under the Hood (Core Spectral Analysis)

The `run_analysis` function is a powerful, all-in-one tool. However, advanced users may want more control over the analysis by using the core functions directly. This tutorial will show you how to build a custom workflow.

### The Manual Analysis Pipeline

When you use the core functions, you are essentially performing the steps of `run_analysis` one by one:

1.  **Load Data**: Use `load_data`.
2.  **Preprocess Data**: Use `preprocess_data`.
3.  **Generate a Frequency Grid**: Use `generate_log_spaced_grid`.
4.  **Calculate the Periodogram**: Use `calculate_periodogram`.
5.  **Fit the Spectrum**: Use functions like `fit_spectrum`.
6.  **Interpret the Results**: Use `interpret_results`.

In this tutorial, we will focus on **steps 3 and 4**.

### Step-by-Step Example

First, let's get some data loaded and preprocessed so we have something to work with.

```python
from waterSpec import load_data, preprocess_data
import numpy as np

# 1. Load Data
time_numeric, data_series, _ = load_data('examples/sample_data.csv', 'timestamp', 'concentration')

# 2. Preprocess Data (let's just detrend for this example)
processed_data, _ = preprocess_data(data_series, time_numeric, detrend_method='linear')

# 3. Remove NaNs (important step before analysis!)
valid_indices = ~np.isnan(processed_data)
time_final = time_numeric[valid_indices]
data_final = processed_data[valid_indices]
```

### Step 3: Generate a Frequency Grid

The `calculate_periodogram` function requires a grid of frequencies at which to calculate the power. A logarithmically spaced grid is best for finding the spectral exponent. We provide a handy helper function for this.

```python
from waterSpec import generate_log_spaced_grid

frequency_grid = generate_log_spaced_grid(time_final)

print(f"Generated a grid of {len(frequency_grid)} frequencies.")
print(f"First 5 frequencies (Hz): {frequency_grid[:5]}")
```

**Output:**
```text
--- Step 3: Generate Frequency Grid ---
Generated a grid of 200 frequencies.
First 5 frequencies (Hz): [2.36205593e-07 2.40032977e-07 2.43922378e-07 2.47874802e-07
 2.51891269e-07]
```

### Step 4: Calculate the Periodogram

Now we have all the ingredients to calculate the Lomb-Scargle periodogram. We pass our final time and data arrays, along with our new frequency grid, to the `calculate_periodogram` function.

```python
from waterSpec import calculate_periodogram

frequency, power = calculate_periodogram(
    time=time_final,
    data=data_final,
    frequency=frequency_grid
)

print("Calculated power for each frequency.")
print("First 5 power values:")
print(power[:5])
```

**Output:**
```text
--- Step 4: Calculate Periodogram ---
Calculated power for each frequency.
First 5 power values:
[0.00371392 0.0037753  0.00383515 0.00389199 0.0039441 ]
```

### Full Control

You now have the `frequency` and `power` arrays, ready to be passed to the fitting and interpretation functions, which we will cover in the next tutorial. This manual, step-by-step approach gives you full control over your analysis.

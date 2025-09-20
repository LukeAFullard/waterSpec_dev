# Tutorial 3: Advanced Preprocessing

Before we can analyze our data, we need to make sure it's clean and ready. This is the preprocessing step. `waterSpec` provides a powerful set of tools to handle common issues in environmental data, such as censored values and trends.

This tutorial will guide you through the preprocessing options available in the main `run_analysis` function (and the underlying `preprocess_data` function).

### Part 1: Handling Censored Data

Water quality data often contains 'censored' values, where a measurement is below a detection limit (e.g., `<5`). `waterSpec` can handle these automatically using the `censor_strategy` parameter.

We'll use a special dataset, `censored_data.csv`, for this example.

```python
from waterSpec import load_data
import numpy as np

# Load the censored data
time, data, _ = load_data(
    file_path='examples/censored_data.csv',
    time_col='timestamp',
    data_col='concentration'
)

print(f"Original censored data: {data.to_list()}")
```

Now, let's see how the different `censor_strategy` options work. We will also pass `min_length=1` for this small example dataset to prevent it from raising an error about data length.

```python
from waterSpec.preprocessor import preprocess_data

# Strategy 1: 'drop' (default) - treats censored values as missing (NaN)
processed_drop, _ = preprocess_data(data, time, censor_strategy='drop', min_length=1)
print(f"Strategy 'drop': {processed_drop}")

# Strategy 2: 'use_detection_limit' - uses the numeric limit
processed_limit, _ = preprocess_data(data, time, censor_strategy='use_detection_limit', min_length=1)
print(f"Strategy 'use_detection_limit': {processed_limit}")

# Strategy 3: 'multiplier' - multiplies the limit by a factor
processed_multiplier, _ = preprocess_data(
    data, time, censor_strategy='multiplier', censor_options={'lower_multiplier': 0.5}, min_length=1
)
print(f"Strategy 'multiplier': {processed_multiplier}")
```

**Output:**
```text
--- Censored Data Handling ---
Original censored data: ['10.1', '<5.0', '10.3', '>100', '11.0']
Strategy 'drop': [10.1  nan 10.3  nan 11. ]
Strategy 'use_detection_limit': [ 10.1   5.   10.3 100.   11. ]
Strategy 'multiplier': [ 10.1   2.5  10.3 110.   11. ]
```

### Part 2: Transformations (Log and Normalize)

Sometimes, it's useful to transform your data before analysis.

- **Log-transformation** (`log_transform_data=True`) is useful when your data has a skewed distribution or when you suspect multiplicative trends.
- **Normalization** (`normalize_data=True`) scales your data to have a mean of 0 and a standard deviation of 1. This is useful when comparing the spectral properties of different parameters with different units or scales.

```python
# Let's use the original sample data for this
_, sample_data, _ = load_data('examples/sample_data.csv', 'timestamp', 'concentration')
time_numeric = np.arange(len(sample_data))

# Apply log-transformation
log_data, _ = preprocess_data(sample_data.copy(), time_numeric, log_transform_data=True)
print(f"Original data (first 5): {sample_data.head().values}")
print(f"Log-transformed (first 5): {log_data[:5]}")

# Apply normalization
norm_data, _ = preprocess_data(sample_data.copy(), time_numeric, normalize_data=True)
print(f"\nNormalized data has mean: {np.nanmean(norm_data):.2f} and std: {np.nanstd(norm_data):.2f}")
```

**Output:**
```text
--- Transformations ---
Original data (first 5): [10.  10.2 10.1 10.5 10.3]
Log-transformed (first 5): [2.30258509 2.32238772 2.31253542 2.35137526 2.3321439 ]
Normalized data has mean: 0.00 and std: 1.00
```

### Part 3: Detrending

Spectral analysis assumes that the time series is 'stationary' (its statistical properties don't change over time). A trend violates this assumption. `waterSpec` can remove linear or non-linear trends.

Let's create some synthetic data with a clear linear trend to see this in action.

```python
import pandas as pd

# Create synthetic data with a linear trend
time_synth = np.arange(100)
trend = 0.5 * time_synth
series_trended = trend + np.random.randn(100)

# Remove the trend
series_detrended, _ = preprocess_data(pd.Series(series_trended), time_synth, detrend_method='linear')

print(f"Mean of original trended data: {np.mean(series_trended):.2f}")
print(f"Mean of detrended data: {np.mean(series_detrended):.2f}")
```

**Output:**
```text
--- Detrending ---
Mean of original trended data: 24.51
Mean of detrended data: -0.00
```

For more complex, non-linear trends, you can use `detrend_method='loess'`. That's it for preprocessing!

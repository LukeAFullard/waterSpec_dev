# Tutorial 3: Advanced Preprocessing

Before analysis, data must be cleaned and prepared. This is the preprocessing step, and in the new `waterSpec` workflow, it's handled automatically and configured when you create your `Analysis` object.

This tutorial will guide you through the preprocessing options you can pass to the `Analysis` constructor.

### Part 1: Handling Censored Data

Water quality data often contains 'censored' values (e.g., `<5.0`). `waterSpec` can handle these automatically using the `censor_strategy` parameter during initialization.

We'll use a special dataset, `censored_data.csv`, for this example.

```python
from waterSpec import Analysis

# Strategy 1: 'drop' (default) - treats censored values as missing (NaN)
analyzer_drop = Analysis(
    'examples/censored_data.csv', 'timestamp', 'concentration', censor_strategy='drop'
)
# The .data attribute holds the final, processed data. NaNs have been removed.
print(f"Strategy 'drop': {analyzer_drop.data}")

# Strategy 2: 'use_detection_limit' - uses the numeric limit
analyzer_limit = Analysis(
    'examples/censored_data.csv', 'timestamp', 'concentration', censor_strategy='use_detection_limit'
)
print(f"Strategy 'use_detection_limit': {analyzer_limit.data}")

# Strategy 3: 'multiplier' - multiplies the limit by a factor
analyzer_multiplier = Analysis(
    'examples/censored_data.csv', 'timestamp', 'concentration',
    censor_strategy='multiplier', censor_options={'lower_multiplier': 0.5, 'upper_multiplier': 1.1}
)
print(f"Strategy 'multiplier': {analyzer_multiplier.data}")
```

### Part 2: Transformations (Log and Normalize)

You can transform your data by passing boolean flags to the `Analysis` constructor.

- **Log-transformation** (`log_transform_data=True`) is useful for skewed data.
- **Normalization** (`normalize_data=True`) scales data to a mean of 0 and a standard deviation of 1.

```python
# Apply log-transformation during initialization
analyzer_log = Analysis(
    'examples/sample_data.csv', 'timestamp', 'concentration', log_transform_data=True
)
print(f"Log-transformed (first 5): {analyzer_log.data[:5]}")

# Apply normalization during initialization
analyzer_norm = Analysis(
    'examples/sample_data.csv', 'timestamp', 'concentration', normalize_data=True
)
print(f"\nNormalized data has mean: {analyzer_norm.data.mean():.2f} and std: {analyzer_norm.data.std():.2f}")
```

### Part 3: Detrending

Spectral analysis assumes stationarity. You can remove linear or non-linear trends using the `detrend_method` argument.

Let's see this in action by initializing an `Analysis` object with detrending enabled.

```python
import numpy as np
import pandas as pd

# Create synthetic data with a linear trend
time_synth = np.arange(100)
trend = 0.5 * time_synth
series_trended = trend + np.random.randn(100)

# Create a temporary file for the example
temp_file_path = "temp_trended_data.csv"
pd.DataFrame({'time': time_synth, 'value': series_trended}).to_csv(temp_file_path, index=False)

# Initialize the Analysis object with linear detrending
analyzer_detrended = Analysis(
    temp_file_path, 'time', 'value', detrend_method='linear'
)

print(f"Mean of original trended data: {series_trended.mean():.2f}")
print(f"Mean of detrended data: {analyzer_detrended.data.mean():.2f}")

# Clean up the temporary file
import os
os.remove(temp_file_path)
```

For more complex, non-linear trends, you can use `detrend_method='loess'`.

That's it for preprocessing! By setting these options during initialization, you ensure that the `Analysis` object is always ready for the final analysis step.

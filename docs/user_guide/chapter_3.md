# Chapter 3: Data Ingestion & Preprocessing

Environmental time-series data is notoriously messy. Sensor drift, power failures, equipment calibration limits, and extreme weather events often result in datasets riddled with gaps, missing values, and unparseable characters. **waterSpec** was built from the ground up to anticipate and manage these exact challenges, ensuring robust mathematical evaluation without catastrophic pipeline failures.

This chapter explains how **waterSpec** safely ingests, cleans, and transforms your raw data before running its core spectral algorithms.

## 3.1 Loading Data

**waterSpec** is designed to be highly user-friendly, ingesting raw data directly through its primary `Analysis` class. You do not need to pre-format your data into rigid mathematical matrices; the package handles the translation from raw files to functional arrays.

Here is a quick example of how to load a standard dataset:

```python
from waterSpec import Analysis

analyzer = Analysis(
    file_path='data/my_river_data.csv',
    time_col='timestamp',
    data_col='nitrate_concentration',
    param_name='Nitrate-N'
)
```

By default, the `file_path` argument natively supports standard formats including CSV, Excel (`.xlsx`), and JSON.

### Time Parsing and Numerical Conversion
Handling timestamps over decades-long monitoring projects can be computationally tricky. When **waterSpec** ingests your file, it automatically parses standard Datetime strings and converts them into relative numerical times (e.g., elapsed seconds from the start of the record). It also natively supports raw numerical epoch arrays.

Under the hood, **waterSpec** explicitly casts these temporal arrays to 64-bit floating-point numbers (`np.float64`) *before* applying relative subtractions. This technical precision prevents `int64` numerical overflow and wrap-around errors, ensuring complete stability even for datasets spanning centuries.

## 3.2 Handling Messy Data

When dealing with real-world hydrology and biogeochemistry, raw files almost never represent a perfect mathematical continuum. **waterSpec** incorporates defensive preprocessing steps to secure the pipeline.

### NaNs and Infs (Missing & Infinite Values)
A single infinite value can trigger a catastrophic cancellation or Denial of Service (DoS) failure in complex mathematical solvers. **waterSpec** automatically scans for, identifies, and neutralizes Not-a-Number (`NaN`) values, as well as positive and negative Infinity (`np.inf` and `-np.inf`).

Infinite values are safely replaced with `NaN`, after which all `NaN` entries are securely stripped from the processing array. The pipeline gracefully retains the original timestamps of the valid data points, treating the stripped values as irregular gaps rather than failing.

### Censored Data Resolution
Environmental laboratory results often contain censored values indicating instrument detection limits, such as `< 0.05 mg/L` or `> 100 NTU`.

Because these unparseable string entries cannot be directly evaluated in the frequency domain, **waterSpec** defaults to a conservative approach: it automatically converts these entries to `NaN` and drops them from the continuous array.

However, many data scientists prefer to impute these values (e.g., replacing `< 0.05` with half the detection limit, `0.025`). If you wish to apply custom censored-data resolution logic, you can easily bypass the internal file loader by preprocessing your dataset in `pandas` and passing the raw arrays directly into the `Analysis` class:

```python
import pandas as pd
from waterSpec import Analysis

# Custom preprocessing
df = pd.read_csv('data/my_river_data.csv')
df['nitrate'] = df['nitrate'].replace('< 0.05', 0.025).astype(float)

# Direct array ingestion
analyzer = Analysis(
    time_array=df['timestamp'].values,
    data_array=df['nitrate'].values,
    param_name='Nitrate-N'
)
```

## 3.3 Data Transformation & Detrending

To meet the strict assumptions of spectral density estimation, time-series data often requires transformation and detrending to ensure the resulting power spectrum reflects true systemic memory rather than mathematical artifacts.

### Log-Transformations
Environmental parameters often exhibit highly skewed, log-normal distributions where variance scales with the mean. Spectral analysis is frequently performed on log-transformed data to stabilize this variance and reduce the disproportionate leverage of extreme peak events (e.g., flood spikes).

**waterSpec** handles these log-transformations automatically within its internal pipelines when statistical models require them. It does so defensively, safely ignoring or replacing zero or negative values before logging to prevent undefined mathematical states.

### Detrending Methodologies
Global trends—such as a decade-long increase in baseline temperatures or a gradual reduction in agricultural runoff—introduce a massive amount of low-frequency "red" noise into the power spectrum. If left uncorrected, these trends can artificially steepen spectral slopes, leading to false conclusions about the system's fractal memory.

**waterSpec** provides built-in methodologies to flatten these artifacts:

*   **Linear Detrending (Default):** The package computes a simple straight-line Ordinary Least Squares (OLS) fit across the entire dataset and subtracts it from the raw values. This perfectly removes simple global trends without distorting high-frequency signals.
*   **LOESS Smoothing:** For more complex background shifts (Locally Estimated Scatterplot Smoothing), **waterSpec** can fit and subtract non-linear, low-frequency background trends, isolating only the fluctuations you care about.

> **Note: When to turn Detrending Off**
> Detrending is automatically managed by the analysis pipeline to ensure robust baseline statistical fitting. However, if your specific research goal *is* to analyze the very long-term memory of the system, including these decadal climatic or environmental shifts, you must instruct the analyzer to leave the background trend intact by explicitly passing `detrend_method=None` during initialization.
# waterSpec: Spectral Analysis of Environmental Time Series

`waterSpec` is a Python package for performing spectral analysis on environmental time series, particularly those that are irregularly sampled. It provides tools to load data, calculate the Lomb-Scargle periodogram, fit the power spectrum to determine the spectral exponent (β), and interpret the results in a hydrological context.

This package is inspired by the methods described in Liang et al. (2021).

## Features

- **Flexible Data Loading**: Load time series data from `.csv`, `.xlsx`, and `.json` files.
- **Censored Data Handling**: Strategies to handle left-censored (`<`) and right-censored (`>`) data.
- **Advanced Preprocessing**: Includes linear and non-linear (LOESS) detrending options.
- **Core Spectral Analysis**:
    - Lomb-Scargle Periodogram for unevenly spaced data.
    - **Standard and Segmented Regression**: Fit a single slope (standard) or detect changes in scaling with a two-slope segmented regression for multifractal analysis.
    - **Uncertainty Analysis**: Bootstrap resampling for confidence intervals on β.
- **Interpretation and Plotting**:
    - Scientific interpretation of the β value.
    - Publication-quality plotting of the power spectrum.
- **Simple Workflow**: A high-level `run_analysis` function to perform a full analysis in one step.

## Installation

You can install `waterSpec` directly from this repository using pip:

```bash
pip install .
```

## Quick Start

Here is a quick example of how to use `waterSpec` to analyze a time series from a CSV file using the high-level `run_analysis` function.

```python
import waterSpec as ws
import os
import pprint

# Define the path to your data file
file_path = 'examples/sample_data.csv'

# Create a directory for the plots if it doesn't exist
if not os.path.exists('plots'):
    os.makedirs('plots')
output_plot_path = 'plots/spectrum_plot.png'

# Run the full analysis with a single function call
# The run_analysis function can also handle censored data, segmented regression, and different detrending methods, e.g.:
# results = ws.run_analysis(..., censor_strategy='multiplier', analysis_type='segmented', detrend_method='loess')
results = ws.run_analysis(
    file_path,
    time_col='timestamp',
    data_col='concentration',
    do_plot=True,
    output_path=output_plot_path
)

# Print the results
print("Analysis Results:")
pprint.pprint(results)

print(f"\nSpectrum plot saved to: {output_plot_path}")

```

This will print a dictionary of results and produce a plot of the power spectrum and its fit, saved to `plots/spectrum_plot.png`.

---
*Liang X, Schilling KE, Jones CS, Zhang Y-K. 2021. Temporal scaling of long-term co-occurring agricultural contaminants and the implications for conservation planning. Environmental Research Letters 16:094015.*

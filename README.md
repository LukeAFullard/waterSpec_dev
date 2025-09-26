# waterSpec: Spectral Analysis of Environmental Time Series

<p align="center">
  <img src="assets/logo.png" alt="waterSpec Logo" width="90%"/>
</p>

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
- **Simple Workflow**: A class-based workflow that performs a full analysis, including generating plots and summaries, with a single command.

## Installation

You can install `waterSpec` directly from this repository using pip:

```bash
pip install .
```

## Quick Start

The new class-based workflow makes running a complete analysis simple and intuitive.

```python
from waterSpec import Analysis
import pprint

# 1. Initialize the Analysis object with your data.
#    This step loads and preprocesses the data internally.
analyzer = Analysis(
    file_path='examples/sample_data.csv',
    time_col='timestamp',
    data_col='concentration'
)

# 2. Run the full analysis with a single command.
#    This performs the analysis and saves all outputs to the specified directory.
results = analyzer.run_full_analysis(output_dir='readme_output')

# The method returns a dictionary with all the numerical results.
print("Analysis Results:")
pprint.pprint(results)
```

This single command will create an `readme_output` directory with two files:
- `concentration_spectrum_plot.png`: A publication-quality plot of the power spectrum.
- `concentration_summary.txt`: A text file with a detailed interpretation of the results.

## Limitations

*   **Error Propagation in Detrending**: While `waterSpec` supports measurement uncertainties (`dy`) and propagates them through normalization and log-transformation, error propagation is **not** currently implemented for the `linear` or `loess` detrending functions. The uncertainty of the detrended signal is assumed to be the same as the original signal's uncertainty. This is a common simplification but should be considered when analyzing data with large trends.

---
*Liang X, Schilling KE, Jones CS, Zhang Y-K. 2021. Temporal scaling of long-term co-occurring agricultural contaminants and the implications for conservation planning. Environmental Research Letters 16:094015.*

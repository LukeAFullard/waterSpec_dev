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

This package is not yet on PyPI. To install it, clone this repository and install it in editable mode using pip:

```bash
git clone https://github.com/example/waterSpec.git
cd waterSpec
pip install -e .
```

## Quick Start: A Class-Based Workflow

The recommended workflow is centered around the `waterSpec.Analysis` object. You initialize this object with your dataset, and it handles all the data loading and preprocessing internally. You can then run a complete analysis with a single command.

### Step 1: Initialize the Analysis

First, import the `Analysis` class and create an instance. You only need to provide your file path and column names once.

```python
from waterSpec import Analysis

# Define the path to your data file
file_path = 'examples/sample_data.csv'

# Create the analyzer object
# This loads and preprocesses the data immediately.
analyzer = Analysis(
    file_path=file_path,
    time_col='timestamp',
    data_col='concentration',
    param_name='Sample Concentration' # Optional: for better plot titles and summaries
)

print("Analysis object created and data preprocessed.")
```

### Step 2: Run the Full Analysis

Now, simply call the `run_full_analysis()` method. You just need to tell it where to save the outputs.

This single command will:
1.  Run the automated analysis to find the best-fitting model (standard or segmented).
2.  Calculate significant periodicities using a robust, data-driven method.
3.  Generate and save a publication-quality plot of the power spectrum.
4.  Generate and save a detailed text summary of the results.

```python
# Define an output directory
output_dir = 'example_output'

# Run the entire workflow
results_dict = analyzer.run_full_analysis(output_dir=output_dir)

# You can inspect the full results dictionary
# import pprint
# pprint.pprint(results_dict)
```

### Step 3: Review the Outputs

After the command finishes, your specified output directory (`example_output/` in this case) will contain a plot and a text summary. Below are the assets generated for our example data.

**1. The Spectrum Plot**

A plot showing the log-log power spectrum, the best-fit line(s), the estimated spectral exponent (β), and any significant peaks.

<p align="center">
  <img src="assets/readme_spec_plot.png" alt="Example waterSpec Plot" width="80%"/>
</p>

**2. The Summary File**

A text file containing a comprehensive, human-readable interpretation of the analysis.

```
Automatic Analysis for: readme_example
-----------------------------------
Model Comparison (Lower BIC is better):
  - Standard Fit:   BIC = 90.05 (β = -0.37)
  - Segmented Fit:  BIC = 47.73 (β1 = 0.36, β2 = -1.52)
==> Chosen Model: Segmented
-----------------------------------

Details for Chosen (Segmented) Model:
Segmented Analysis for: readme_example
Breakpoint Period ≈ 10.3 days
-----------------------------------
Low-Frequency (Long-term) Fit:
  β1 = 0.36
  Interpretation: -0.5 < β < 1 (fGn-like): Weak persistence or anti-persistence, suggesting event-driven transport.
  Persistence: 🔴 Event-driven
-----------------------------------
High-Frequency (Short-term) Fit:
  β2 = -1.52
  Interpretation: Warning: Beta value is significantly negative, which is physically unrealistic.
  Persistence: 🔴 Event-driven

-----------------------------------
Significant Periodicities Found:
  - Period: 3.0 days (Fit Residual: 4.29)
  - Period: 5.9 days (Fit Residual: 2.51)
```

## Limitations

*   **Error Propagation in Detrending**: While `waterSpec` supports measurement uncertainties (`dy`) and propagates them through normalization and log-transformation, error propagation is **not** currently implemented for the `linear` or `loess` detrending functions. The uncertainty of the detrended signal is assumed to be the same as the original signal's uncertainty. This is a common simplification but should be considered when analyzing data with large trends.

---
*Liang X, Schilling KE, Jones CS, Zhang Y-K. 2021. Temporal scaling of long-term co-occurring agricultural contaminants and the implications for conservation planning. Environmental Research Letters 16:094015.*
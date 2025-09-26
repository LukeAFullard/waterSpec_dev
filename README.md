# waterSpec: Spectral Analysis of Environmental Time Series

<p align="center">
  <img src="assets/logo.png" alt="waterSpec Logo" width="90%"/>
</p>

`waterSpec` is a Python package for performing spectral analysis on environmental time series, particularly those that are irregularly sampled. It provides a simple, powerful workflow to characterize the temporal scaling and periodic behavior of environmental data.

The methods used in this package are inspired by the work of *Liang et al. (2021)*.

## Core Features

`waterSpec` is designed to do two things well:

1.  **Determine the Spectral Slope (β)**: It calculates the Lomb-Scargle periodogram of a time series and fits a regression to the power spectrum. It can automatically compare models with 0, 1, or 2 breakpoints to find the best fit, providing one or more β values that describe the temporal scaling (e.g., persistence, memory) of the system.

2.  **Find Significant Peaks**: It identifies statistically significant periodicities (peaks) in the periodogram, which correspond to recurring cycles in the time series data (e.g., seasonal or weekly patterns).

## Installation

This package is not yet on PyPI. To install it, clone this repository and install it in editable mode using pip:

```bash
git clone https://github.com/example/waterSpec.git
cd waterSpec
pip install -e .
```

## Quick Start

The recommended workflow is centered around the `waterSpec.Analysis` object. You can run a complete analysis, generating a plot and a detailed text summary, with just a few lines of code.

```python
from waterSpec import Analysis

# 1. Define the path to your data file
file_path = 'examples/sample_data.csv'

# 2. Create the analyzer object
# This loads and preprocesses the data immediately.
analyzer = Analysis(
    file_path=file_path,
    time_col='timestamp',
    data_col='concentration',
    param_name='Sample Concentration' # Used for plot titles and summaries
)

# 3. Run the full analysis
# This command runs the analysis, saves the outputs, and returns the results.
results = analyzer.run_full_analysis(output_dir='example_output')

# The summary text is available in the returned dictionary
print(results['summary_text'])
```

## Example Output

Running the code above will produce a plot (`example_output/Sample_Concentration_spectrum_plot.png`) and the following text summary in `example_output/Sample_Concentration_summary.txt`:

```
Automatic Analysis for: Sample Concentration
-----------------------------------
Model Comparison (Lower BIC is better):
  - Standard         BIC = 90.05    (β = -0.37)
  - Segmented (1 BP)   BIC = 47.73    (β1=0.36, β2=-1.52)
==> Chosen Model: Segmented 1bp
-----------------------------------

Details for Chosen (Segmented 1bp) Model:
Segmented Analysis for: Sample Concentration
--- Breakpoint @ ~10.3 days ---
Low-Frequency (Long-term) Fit:
  β1 = 0.36
  Interpretation: -0.5 < β < 1 (fGn-like): Weak persistence or anti-persistence, suggesting event-driven transport.
  Persistence: 🔴 Event-driven
High-Frequency (Short-term) Fit:
  β2 = -1.52
  Interpretation: Warning: Beta value is significantly negative, which is physically unrealistic.
  Persistence: 🔴 Event-driven

-----------------------------------
Significant Periodicities Found:
  - Period: 3.0 days (Fit Residual: 4.29)
  - Period: 5.9 days (Fit Residual: 2.51)
```

This output shows the two core features in action:
*   **Spectral Slope (β) Analysis**: The `Model Comparison` section shows that a segmented model with one breakpoint (1 BP) was the best fit for this data (lower BIC is better). The `Details` section provides the two resulting β values for the low-frequency and high-frequency parts of the spectrum.
*   **Peak Detection**: The `Significant Periodicities Found` section lists the two recurring cycles that were identified in the data.

## Advanced Usage: Two-Breakpoint Analysis

By default, `waterSpec` compares a standard model (0 breakpoints) with a 1-breakpoint model. To have the analysis also consider a 2-breakpoint model, set `max_breakpoints=2`:

```python
# The analysis will now compare 0, 1, and 2 breakpoint models
results = analyzer.run_full_analysis(
    output_dir='example_output_2bp',
    max_breakpoints=2
)
print(results['summary_text'])
```

## Citation

If you use `waterSpec` in your research, please consider citing the original work that inspired its methods:

*Liang X, Schilling KE, Jones CS, Zhang Y-K. 2021. Temporal scaling of long-term co-occurring agricultural contaminants and the implications for conservation planning. Environmental Research Letters 16:094015.*
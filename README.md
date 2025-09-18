# waterSpec: Spectral Analysis of Environmental Time Series

`waterSpec` is a Python package for performing spectral analysis on environmental time series, particularly those that are irregularly sampled. It provides tools to load data, calculate the Lomb-Scargle periodogram, fit the power spectrum to determine the spectral exponent (β), and interpret the results in a hydrological context.

This package is inspired by the methods described in Liang et al. (2021).

## Installation

You can install `waterSpec` directly from this repository using pip:

```bash
pip install .
```

## Quick Start

Here is a quick example of how to use `waterSpec` to analyze a time series from a CSV file.

```python
import waterSpec as ws
import os

# 1. Load the data
# This example uses the sample data provided with the package.
file_path = 'examples/sample_data.csv'
time, concentration = ws.load_data(file_path, time_col='timestamp', data_col='concentration')

# 2. Preprocess the data (optional)
# For this example, we'll just detrend the data.
detrended_concentration = ws.detrend(concentration)

# 3. Calculate the power spectrum
frequency, power = ws.calculate_periodogram(time, detrended_concentration)

# 4. Fit the spectrum to find the spectral exponent (beta)
fit_results = ws.fit_spectrum(frequency, power)
beta = fit_results['beta']

# 5. Interpret the result
interpretation = ws.interpret_beta(beta)
print(f"Spectral Exponent (β): {beta:.2f}")
print(f"Interpretation: {interpretation}")

# 6. Plot the results
# Create a directory for the plots if it doesn't exist
if not os.path.exists('plots'):
    os.makedirs('plots')
output_plot_path = 'plots/spectrum_plot.png'
ws.plot_spectrum(frequency, power, fit_results=fit_results, output_path=output_plot_path, show=False)

print(f"\nSpectrum plot saved to: {output_plot_path}")
```

This will produce a plot of the power spectrum and its fit, saved to `plots/spectrum_plot.png`.

---
*Liang X, Schilling KE, Jones CS, Zhang Y-K. 2021. Temporal scaling of long-term co-occurring agricultural contaminants and the implications for conservation planning. Environmental Research Letters 16:094015.*

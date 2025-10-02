# waterSpec: Spectral Analysis of Environmental Time Series

![waterSpec Logo](assets/logo.png)

`waterSpec` is a Python package for performing spectral analysis on environmental time series, particularly those that are irregularly sampled. It provides two core capabilities:

## Core Features

### 1. Spectral Power Coefficient (Beta) Estimation

Estimate the **spectral exponent (β)** that characterizes how variance is distributed across timescales in your data. `waterSpec` automatically fits and compares:

- **Linear models**: A single power-law relationship across all frequencies
- **Segmented models**: Multiple power-law regimes separated by breakpoints, revealing shifts in system behavior

The package uses the **Bayesian Information Criterion (BIC)** to objectively select the best model, balancing goodness-of-fit against model complexity.

### 2. Peak Detection

Identify **statistically significant periodicities** in your time series using:

- **False Alarm Probability (FAP)**: Traditional, robust method independent of background model
- **Residual-based detection**: FDR-corrected outlier analysis relative to the fitted spectrum

Both methods provide quantitative significance levels and handle irregularly sampled data.

[![Python Version](https://img.shields.io/badge/python-3.8%2B-blue)]()
[![License](https://img.shields.io/badge/license-MIT-green)]()

---

## Why Use waterSpec?

Environmental time series contain information at many timescales. `waterSpec` helps you answer questions like:

- **Is this contaminant transported by surface runoff or subsurface flow?** Different transport mechanisms leave distinct "fingerprints" in the frequency spectrum.
- **Are there hidden cycles in my data?** Detect statistically significant periodicities (seasonal, weekly, diurnal) that may not be obvious in raw plots.
- **How does temporal variability change across timescales?** Identify breakpoints where system behavior shifts from one mode to another.
- **How uncertain are my results?** Robust bootstrap confidence intervals quantify the reliability of all estimates.

The package implements methods inspired by **Liang et al. (2021)**, who used spectral analysis to characterize agricultural contaminant transport in watersheds.

---

## Key Features

### 🎯 Automatic Model Selection
Fits both **linear** (single power-law) and **segmented** (multi-regime) spectral models, then uses the Bayesian Information Criterion (BIC) to objectively choose the best model for your data—preventing both underfitting and overfitting.

### 📊 Robust Statistical Analysis
- **Bootstrap confidence intervals** for all parameters (spectral exponents, breakpoints)
- **Multiple peak detection methods**: False Alarm Probability (FAP) or residual-based with FDR correction
- **Handles irregular sampling** via Lomb-Scargle periodogram
- **Autocorrelation-aware bootstrapping** with block and wild bootstrap options

### 🔬 Designed for Environmental Data
- **Flexible preprocessing**: censored data handling, log-transforms, linear/LOESS detrending, normalization
- **Physical interpretation**: automatic comparison to benchmark transport regimes (e.g., nitrate, TSS, E. coli)
- **Traffic-light persistence indicators**: instant visual assessment of system behavior

### 📈 Publication-Ready Outputs
- High-resolution plots with confidence bands, breakpoint annotations, and detected periodicities
- Detailed text summaries with scientific interpretation and uncertainty quantification
- All results returned as structured dictionaries for further analysis

---

## Installation

**Standard installation:**
```bash
git clone https://github.com/yourusername/waterSpec.git
cd waterSpec
pip install -e .
```

**For development (includes testing dependencies):**
```bash
pip install -e '.[test]'
```

---

## Quick Start

Analyze a time series in just a few lines:

```python
from waterSpec import Analysis

# Create analyzer and load data
analyzer = Analysis(
    file_path='data/nitrate_timeseries.csv',
    time_col='timestamp',
    data_col='concentration',
    param_name='Nitrate-N at Site A'
)

# Run complete analysis with automatic model selection
results = analyzer.run_full_analysis(
    output_dir='output',
    ci_method='bootstrap'  # Use robust bootstrap CIs
)

# View summary
print(results['summary_text'])
```

**That's it!** The package automatically:
1. Loads and validates your data
2. Calculates the Lomb-Scargle periodogram
3. Fits and compares multiple spectral models
4. Detects significant periodicities
5. Generates publication-quality plots and interpretations

---

## Example Output

### Segmented Spectrum Analysis

When your data exhibits different scaling behavior at different frequencies (common in environmental systems with multiple transport pathways):

![Segmented Spectrum Plot](example_output/Nitrate_Concentration_at_Site_A_spectrum_plot.png)

```
Automatic Analysis for: Nitrate Concentration at Site A
-----------------------------------
Model Comparison (Lower BIC is better):
  - Standard        BIC = 90.05    (β = -0.37)
  - Segmented (1 BP) BIC = 47.73    (β₁=0.36, β₂=-1.52)

==> Chosen Model: Segmented 1bp

Low-Frequency (Long-term) Fit:
  β₁ = 0.36 (95% CI: -0.11–0.83)
  Interpretation: Weak persistence, suggesting event-driven transport
--- Breakpoint @ ~10.3 days ---
High-Frequency (Short-term) Fit:
  β₂ = -1.52 (95% CI: -1.97–-1.08)
  
Significant Periodicities Found:
  - Period: 3.0 days (Fit Residual: 4.29)
  - Period: 5.9 days (Fit Residual: 2.51)
```

### Peak Detection

Identify statistically significant cycles in your data:

![Peak Detection Example](example_output/Peak_Detection_Example_spectrum_plot.png)

The package correctly identifies a synthetic 30-day cycle using False Alarm Probability (FAP) testing.

---

## Understanding Spectral Exponents (β)

The spectral exponent β characterizes how power (variance) is distributed across frequencies:

| β Range | Scientific Meaning | Environmental Interpretation |
|---------|-------------------|------------------------------|
| **β ≈ 0** | White noise | Completely random, uncorrelated process |
| **0 < β < 1** | Fractional Gaussian noise (fGn) | **Event-driven**: storms, surface runoff, episodic inputs |
| **β ≈ 1** | Pink noise (1/f) | Balanced persistence, common in natural systems |
| **1 < β < 3** | Fractional Brownian motion (fBm) | **Storage-dominated**: groundwater, reservoir effects, strong persistence |
| **β > 3** | Black noise | Very smooth, may indicate non-stationarity |

**Typical values for water quality parameters** (from Liang et al. 2021):
- E. coli: β ≈ 0.1–0.5 (surface runoff)
- TSS: β ≈ 0.4–0.8 (surface runoff)
- Ortho-P: β ≈ 0.6–1.2 (mixed pathways)
- Nitrate-N: β ≈ 1.5–2.0 (subsurface flow)
- Chloride: β ≈ 1.3–1.7 (subsurface flow)

---

## Advanced Usage

### Preprocessing Options

Handle real-world data complexities:

```python
analyzer = Analysis(
    file_path='data/field_measurements.csv',
    time_col='date',
    data_col='value',
    # Handle non-detect values
    censor_strategy='use_detection_limit',  # Options: 'drop', 'use_detection_limit', 'multiplier'
    # Transform data
    log_transform_data=True,  # Recommended for concentrations
    # Remove trends
    detrend_method='loess',  # Options: None, 'linear', 'loess'
    # Normalize
    normalize_data=False,
    verbose=True  # Enable detailed logging
)
```

### Fine-Tune Analysis Parameters

```python
results = analyzer.run_full_analysis(
    output_dir='output',
    # Model complexity
    max_breakpoints=2,  # Try 0, 1, and 2 breakpoint models
    # Frequency grid
    samples_per_peak=10,  # Higher = denser grid
    nyquist_factor=1.0,
    # Confidence intervals
    ci_method='bootstrap',  # 'bootstrap' (robust) or 'parametric' (fast)
    n_bootstraps=2000,  # More samples = better CIs
    bootstrap_type='residuals',  # 'pairs', 'residuals', 'block', 'wild'
    # Peak detection
    peak_detection_method='fap',  # 'fap' or 'residual'
    fap_threshold=0.01,  # 1% false alarm probability
    fap_method='baluev',  # 'baluev' (fast) or 'bootstrap' (slow)
    # Reproducibility
    seed=42
)
```

### Confidence Interval Methods

**Bootstrap (recommended for final analysis):**
- Non-parametric, makes minimal assumptions
- Robust to violations of normality
- Computationally intensive

**Parametric (recommended for exploration):**
- Fast, based on statistical theory
- Assumes normally distributed errors
- Good for initial analyses

### Choosing the Right Bootstrap Method

Bootstrapping is a powerful technique for estimating confidence intervals, but the best method depends on the characteristics of your data. `waterSpec` offers four types, controlled by the `bootstrap_type` parameter. Here’s a guide to help you choose:

| Method         | When to Use                                                                                                | How it Works                                                                       |
|----------------|------------------------------------------------------------------------------------------------------------|------------------------------------------------------------------------------------|
| **`'pairs'`**  | **Default choice.** Good when you are uncertain about the model's fit or assumptions.                        | Resamples (frequency, power) data points together. Makes the fewest assumptions.   |
| **`'residuals'`**| When your spectral model is a very good fit and residuals appear random (no autocorrelation).              | Resamples the model's residuals. More powerful if the model is correct.            |
| **`'block'`**  | When you suspect **autocorrelation** in the data (i.e., values are correlated with their neighbors).           | Resamples blocks of data to preserve the original correlation structure.           |
| **`'wild'`**   | When you suspect **heteroscedasticity** (i.e., the variance of the residuals is not constant).                 | Creates synthetic residuals with a zero mean but preserves their original variance. |

**Decision Flowchart:**

1.  **Start with `'pairs'`** if you are unsure. It's the most robust and makes the fewest assumptions.
2.  **Inspect your model fit.** If the fitted line follows the data well and the residuals look like random noise, **`'residuals'`** can provide more statistical power.
3.  **Check for autocorrelation.** Use the `durbin_watson_stat` in the results or plot the residuals. If you see patterns (e.g., runs of positive or negative residuals), your data is likely autocorrelated. Switch to **`'block'`**.
4.  **Check for heteroscedasticity.** If the spread of your residuals changes across the frequency range, the variance is not constant. Switch to **`'wild'`**.

---

## File Format Requirements

`waterSpec` accepts **CSV**, **Excel** (.xlsx, .xls), or **JSON** files with:

- A **time column** (any datetime format)
- A **data column** (numeric values)
- An optional **error column** (measurement uncertainties)

**Example CSV:**
```csv
timestamp,concentration,error
2020-01-01 00:00:00,2.5,0.1
2020-01-02 06:30:00,3.1,0.15
2020-01-03 12:00:00,1.8,0.08
...
```

Column names are **case-insensitive**. Missing values (NaN) are automatically handled.

---

## Interpreting Results

### The `results` Dictionary

```python
results = {
    'chosen_model': 'segmented_1bp',  # or 'standard'
    'beta': 1.23,  # (for standard) or
    'betas': [0.5, 1.8],  # (for segmented)
    'breakpoints': [0.01],  # in frequency units
    'beta_ci_lower': 1.1,  # Confidence intervals
    'beta_ci_upper': 1.4,
    'bic': 45.2,  # Model selection criterion
    'significant_peaks': [  # Detected periodicities
        {'frequency': 0.033, 'power': 15.2, 'fap': 0.001}
    ],
    'summary_text': '...',  # Full interpretation
    'preprocessing_diagnostics': {...}
}
```

### Uncertainty Warnings

The package automatically flags potential issues:
- Wide confidence intervals (high uncertainty)
- Breakpoints near data boundaries (unstable fits)
- Low bootstrap success rates
- Autocorrelation in residuals
- Excessive variance removed by detrending

---

## Dependencies

### Required
- `numpy` ≥ 1.20
- `pandas` ≥ 1.3
- `scipy` ≥ 1.7
- `matplotlib` ≥ 3.4
- `astropy` ≥ 5.0 (for Lomb-Scargle periodogram)
- `statsmodels` ≥ 0.13

### Optional but Recommended
- `piecewise-regression` (for segmented models)

### Citation for Segmented Regression

If you use segmented model results, please cite:

> Pilgrim, C. (2021). piecewise-regression (aka segmented regression) in Python. *Journal of Open Source Software*, 6(68), 3859. https://doi.org/10.21105/joss.03859

---

## Citing waterSpec

If you use `waterSpec` in your research, please cite the underlying methodology:

> Liang, X., Schilling, K. E., Jones, C. S., & Zhang, Y. K. (2021). Temporal scaling of long-term co-occurring agricultural contaminants and the implications for conservation planning. *Environmental Research Letters*, 16(9), 094015. https://doi.org/10.1088/1748-9326/ac19dd

---

## Best Practices

### 1. Start Simple
Begin with default parameters and `ci_method='parametric'` for fast exploration. Switch to `ci_method='bootstrap'` for final analysis.

### 2. Preprocess Thoughtfully
- **Log-transform** concentration data to stabilize variance
- **Detrending** can significantly affect results—only remove trends you're certain are not part of the signal
- Use `verbose=True` to monitor preprocessing steps

### 3. Check Diagnostics
Review the `preprocessing_diagnostics` and any warnings about:
- Censored data handling
- Detrending R²
- Bootstrap success rates
- Autocorrelation (Durbin-Watson statistic)

### 4. Validate Peak Detection
If using `peak_detection_method='residual'`, ensure the background spectral model fits well. Use `'fap'` for more robust detection independent of the model.

### 5. Report Uncertainty
Always report confidence intervals alongside point estimates. Wide CIs indicate high uncertainty and suggest either collecting more data or reconsidering the model.

---

## Troubleshooting

**"Not enough valid data points"**
- Check for excessive NaN values after preprocessing
- Reduce `min_valid_data_points` if your dataset is small
- Review censored data handling strategy

**"Bootstrap iterations failed"**
- Try `bootstrap_type='pairs'` instead of `'residuals'`
- Reduce `n_bootstraps` or switch to `ci_method='parametric'`
- Check for outliers or extreme values

**"No significant breakpoint found"**
- Your data may truly be better described by a single power law
- The BIC comparison still identifies the most appropriate model
- Consider increasing `p_threshold` (less conservative)

**Negative β values**
- May indicate aliasing (undersample high frequencies)
- Could suggest inappropriate preprocessing (over-detrending)
- Check for white noise dominance in your signal

---

## Future Work

While `waterSpec` focuses on frequency-domain analysis, future versions may incorporate time-domain methods to provide a more holistic view of system dynamics. A planned enhancement is the integration of **changepoint detection** algorithms (e.g., PELT).

This would allow users to:
- Identify discrete shifts in the mean or variance of a time series.
- Automatically segment the data around these changepoints.
- Perform separate spectral analyses on each segment to understand how system behavior has changed over time.

---

## Contributing

Contributions are welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Submit a pull request

For bug reports or feature requests, please open an issue on GitHub.

---

## License

This project is licensed under the MIT License - see the LICENSE file for details.

---

## Acknowledgments

- Methodology based on Liang et al. (2021)
- Segmented regression powered by the `piecewise-regression` library
- Lomb-Scargle implementation from Astropy

---

## Support

For questions, issues, or discussions:
- 📧 Email: your.email@example.com
- 🐛 Bug reports: GitHub Issues
- 📖 Documentation: [ReadTheDocs](#) (coming soon)

---

**Start exploring the temporal structure of your environmental data today!**

# Chapter 5: Core Univariate Analysis

Once your data is successfully loaded and preprocessed within the `Analysis` object, you are ready to delve into the core spectral evaluation. At this stage, the user can trigger an automated pipeline that tests multiple hypotheses simultaneously using the comprehensive `run_full_analysis()` method. This single execution acts as the analytical engine of **waterSpec**, extracting structural properties, spectral slopes, and statistically significant periodicities all at once.

## 5.1 Performing a Standard Spectral Fit

Executing the core univariate analysis is designed to be straightforward while remaining highly configurable. Below is a basic code snippet demonstrating how to run the analysis pipeline:

```python
results = analyzer.run_full_analysis(
    output_dir='output_folder',
    fit_method='theil-sen', # Robust linear regression
    ci_method='bootstrap'   # Robust error bars
)
```

By default, **waterSpec** performs a standard spectral fit on the log-log power spectrum. It utilizes a robust statistical estimator, such as `theil-sen` (or standard `ols`), to calculate a single global spectral slope ($\beta$). This spectral slope is critical for understanding the overarching fractal properties and memory of the system. Using a robust estimator like Theil-Sen ensures that anomalous spectral power or high-frequency artifacts do not disproportionately leverage and skew the baseline slope fit.

## 5.2 Detecting Regime Shifts (Segmented Models)

Natural systems rarely follow a single power law across all possible timescales. For instance, a river's discharge might exhibit event-driven, highly persistent behavior—resembling fractional Gaussian noise (fGn)—at daily measurement scales. However, when observed over seasonal or inter-annual scales, the same system might transition into a storage-dominated regime resembling fractional Brownian motion (fBm).

To capture these complex dynamics, **waterSpec** allows you to test for structural breaks in the scaling behavior. If you set `max_breakpoints=1` (or up to 2), the package employs the robust `MannKS` algorithm to automatically detect "kinks" or regime shifts within the log-log spectrum.

When evaluating these structural breaks, **waterSpec** inherently protects you from overfitting. The pipeline automatically fits both the standard single-slope model and the segmented model, automatically selecting the "best" representation by calculating the Bayesian Information Criterion (BIC). To prevent overfitting (finding false breakpoints), the BIC heavily penalizes the segmented model for extra degrees of freedom, specifically, evaluating it as $2k+2$ parameters (where $k$ is the number of breakpoints). This strict penalty safeguards against identifying false breakpoints driven merely by random variance in the spectral estimates.

> **Warning: PELT False Positives in Red Noise**
> When detecting structural mean or variance shifts in the time domain, applying standard changepoint algorithms (like PELT) directly to highly autocorrelated "red noise" data ($\beta > 0$) will drastically inflate the false positive rate. Red noise naturally generates low-frequency, prolonged excursions that algorithms mathematically misidentify as deterministic structural shifts. To mitigate this, you must strictly pre-whiten your data (e.g., using an AR(p) model with order selected by AIC) and run changepoint detection exclusively on the fitted model residuals.

## 5.3 Understanding the Valid Frequency Range

When interpreting Lomb-Scargle periodograms, it is crucial to understand the physically meaningful frequency bounds of your analysis to avoid interpreting mathematical artifacts as real signals:

*   **Minimum Frequency ($f_{min}$):** The lowest resolvable frequency is approximately $1/T$, where $T$ is the total record length of your dataset. Frequencies below this bound represent signals longer than your entire dataset and cannot be reliably measured.
*   **Maximum Frequency ($f_{max}$):** The highest resolvable frequency is bounded by the pseudo-Nyquist frequency, approximately $1 / (2 \cdot \Delta t_{min})$, where $\Delta t_{min}$ is the minimum sampling interval. Spectral peaks or behavior beyond this frequency are highly susceptible to aliasing and should generally be ignored.

## 5.4 Peak Detection & Significance

Beyond the overall fractal slope ($\beta$), users often want to find specific, repeating cycles—such as diurnal fluctuations from evapotranspiration or annual snowmelt cycles.

However, **waterSpec** doesn't just guess if a peak in the periodogram is real; it rigidly calculates a False Alarm Probability (FAP) for the observed frequencies. You can control this calculation using the `fap_method` parameter, which offers two distinct approaches:

*   `'baluev'`: The default setting. This utilizes an analytical, extremely fast approximation of statistical significance.
*   `'bootstrap'`: An exact, non-parametric FAP calculation achieved via rigorous Monte Carlo resampling.

> **Pro Tip: The Computational Cost of Bootstrap FAP**
> While the `'bootstrap'` method provides an exact, distribution-free significance threshold, it is highly computationally expensive, particularly for large datasets. **waterSpec** will issue a warning if this is used on lengthy time series. For most practical use cases, the `'baluev'` approximation provides an excellent and rapid alternative.

Because a standard periodogram checks hundreds or thousands of independent frequencies simultaneously, finding a "significant" peak purely by chance is highly likely. To guarantee mathematically rigorous scientific results, **waterSpec** automatically applies the **Benjamini-Yekutieli False Discovery Rate (FDR)** correction *before* selecting any peaks. By applying this rigorous correction across all hypotheses beforehand, the package prevents severe selection bias and ensures that only true periodic signals are classified as statistically significant cycles.

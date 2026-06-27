# Standard Analysis Pipeline

## What is a "Standard" Analysis?

In `waterSpec`, environmental and hydrological datasets present unique challenges: they are often unevenly sampled, contain missing data, and feature intermittent extreme events (bursts or pulses). Standard analytical techniques built for purely even data often fall short in these domains.

To address these challenges comprehensively without overwhelming users, `waterSpec` provides a **Standard Analysis Workflow** via `Analysis.run_standard_analysis()`. This workflow bundles our most scientifically robust methods tailored specifically for natural systems into a single pipeline.

## Steps Performed in `run_standard_analysis()`

The standard pipeline performs the following sequence of analyses:

### 1. Data Preprocessing & Validation
*   Validates input structure (time series arrays or DataFrames).
*   Converts date/time formats into purely numeric elapsed time arrays.
*   Ensures chronological sorting and proper handling of non-numeric data.
*   (Optional) Accepts explicit commands to linearly detrend or log-transform data if user-specified, otherwise uses the raw data values.

### 2. Global Spectral Slope Estimation (Haar Wavelet Analysis)
*   **Why Haar?** Lomb-Scargle periodograms are excellent for detecting peaks, but they can produce biased estimates of the spectral slope ($\beta$) when data is highly irregular. Haar Wavelet Fluctuation analysis computes the first-order structure function directly in the time domain, strictly handling varying gaps without interpolation artifacts.
*   **Methodology:**
    *   **Overlapping Windows:** The standard analysis sets `haar_overlap=True` to maximize statistical power. This is critical for short or non-continuous records.
    *   **Calculation:** Calculates scaling exponent $m$ and maps it to the basic spectral exponent $\beta$.

### 3. Segmented Haar Fits (Timescale Regime Shifts)
*   Environmental systems rarely exhibit single-scale behavior (e.g., surface runoff memory vs. groundwater memory).
*   **Methodology:** The standard pipeline enables `haar_max_breakpoints=1`, fitting both a continuous linear model and a segmented (broken) linear model to the Haar fluctuation spectrum. It uses the Bayesian Information Criterion (BIC) to automatically determine if a statistically significant regime shift (scale-break) exists.

### 4. Intermittency Correction (Multifractal Analysis)
*   Natural phenomena like rainfall or contaminant transport often feature intermittent extremes that inject artificial variance and distort the standard estimate of $\beta$.
*   **Methodology:** The standard analysis pipeline explicitly enables `calc_intermittency=True`. This computes the multi-fractal intermittency correction, $K(2)$, comparing average scaling ($\zeta(1)$) with variance scaling ($\zeta(2)$).
*   **Output:** The report returns:
    1.  The **Raw $\beta$**: Describing the steady-state baseflow process.
    2.  The **$K(2)$ parameter**: Quantifying the degree of extreme intermittency.
    3.  The **Multifractal $\beta$**: The adjusted, true power-spectral density describing the total variance of the combined system ($\beta_{multi} = 1 + 2H - K(2)$).
    4.  **R-squared ($R^2$)**: Quantifying the goodness of fit for the chosen scaling laws.


### 5. Deterministic Peak Detection (Lomb-Scargle)
*   While Haar estimates the broadband slope, the pipeline runs a Lomb-Scargle Periodogram to detect strong, distinct deterministic cycles (e.g., diurnal 24-hour cycles, annual seasonal cycles).
*   **Methodology:** Evaluates potential peaks using the False Alarm Probability (FAP - Baluev analytical approximation) framework. Only mathematically significant cycles exceeding the rigorous threshold are reported.

### 6. Uncertainty Quantification (Bootstrapping)
*   **Methodology:** Enforces `ci_method="bootstrap"`. Point estimates without error bounds are insufficient for publication. The pipeline defaults to robust Block Bootstrapping to generate 95% confidence intervals for all calculated parameters (spectral slopes, $K(2)$, peak powers).

### 7. Automated Interpretation & Reporting
*   The final step translates mathematical variables into physical hydrologic/environmental meaning.
*   **Methodology:** Maps the resulting $\beta$ metrics to scientific domains (e.g., classifying a result as 'Event-driven Fractional Gaussian Noise' or 'Storage-dominated Fractional Brownian Motion'). Generates human-readable summary text and visual plots comparing the standard vs. segmented fits.

## Usage

```python
from waterSpec import Analysis

analyzer = Analysis(
    file_path='data/my_timeseries.csv',
    time_col='datetime',
    data_col='value',
    param_name='Parameter X'
)

# Run the standard pipeline tailored for environmental data
results = analyzer.run_standard_analysis(output_dir='my_results')
```

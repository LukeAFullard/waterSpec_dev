# waterSpec Reporting System Plan

Designing a comprehensive reporting system for `waterSpec` requires balancing scientific rigor (for peer-reviewed publications) with practical usability (for quick exploratory data analysis). Because `waterSpec` handles complex, irregularly sampled time-series data with advanced statistical tests (e.g., wild bootstrapping, surrogate testing, Lomb-Scargle periodograms, and Haar wavelet analysis), the reporting needs to be transparent and reproducible.

Here is a conceptual breakdown of what a complete reporting suite for `waterSpec` looks like:

## 1. Report Formats

To accommodate different workflows, the reporting module outputs in three tiers:

*   **HTML (Interactive / Exploratory):** A standalone HTML file generated via simple templates, embedding base64 encoded Matplotlib plots (or interactive Plotly equivalents). Useful for quick sharing and browsing.
*   **JSON + CSV (Machine Readable):**
    *   `results.json`: Full nested dictionary of all parameters, breakpoints, confidence intervals, p-values, and wild bootstrap seeds for absolute reproducibility.
    *   `metrics.csv`: Tabular summaries of spectral slopes, intercept, and hysteresis areas for bulk multi-site comparison.
*   **PDF (Publication / Archival) (Future Work):** A static report generated via LaTeX or a library like ReportLab/WeasyPrint. This ensures high-quality vector graphics (PDF/EPS) and formatted mathematical equations suitable for appendices in research papers.

## 2. Core Sections & Included Plots

A standard `waterSpec` report is modular, triggering different sections based on the analysis performed (Univariate, Bivariate, or Regime/Changepoint).

### A. Metadata & Data Overview
*   **Results Included:** Site ID/Name, variable names (e.g., Discharge, Turbidity), start/end dates, total N, number of missing values/gaps, and sampling regularity.

### B. Frequency Domain Analysis (Lomb-Scargle Periodogram)
*   **Results Included:**
    *   Spectral Slope (β) and standard error.
    *   Segmented fit results: Breakpoint frequency, low-frequency β1, high-frequency β2.
    *   Model selection metrics: BIC (degrees of freedom penalized as 3k+2).
    *   False Discovery Rate (FDR) corrected thresholds via Benjamini-Yekutieli.

### C. Time-Scale Analysis (Haar Wavelet Fluctuations)
*   **Results Included:**
    *   Haar scaling exponent (α).
    *   Scale breakpoint if non-stationarity across scales is detected.

### D. Bivariate & System Dynamics (Cross-Spectra & Hysteresis)
*   **Results Included:**
    *   Hysteresis metrics: Loop Area (Zuecco normalized) and Direction (Clockwise vs. Counter-Clockwise based on the explicitly closed shoelace formula polygon).

## 3. Interpretation Guide (The "So What?")

The report provides an automated "Interpretation Summary" based on the results.

*   **Spectral Slopes (β and α):**
    *   If β ≈ 0: The report classifies this as *White Noise* (uncorrelated, memoryless system).
    *   If β ≈ 1: *Pink Noise* / *Flicker Noise* (fractal scaling, long-term memory, typical of large, well-mixed hydrological catchments).
    *   If β ≈ 2: *Red Noise* / *Brownian Motion* (highly persistent, heavily damped system).

## 4. Code Architecture Implementation

Implemented as a `Reporter` module:

```python
from waterSpec.reporting import ReportGenerator

# 1. Run standard analysis
analyzer = Analysis(time, discharge)
results = analyzer.run_full_analysis()

# 2. Pass to the reporter
report = ReportGenerator(
    results=results,
    metadata={"site": "USGS-012345", "variable": "Discharge"}
)

# 3. Generate outputs
report.to_html("usgs_012345_report.html")
report.to_json("usgs_012345_data.json")
report.to_csv("usgs_012345_metrics.csv")
```

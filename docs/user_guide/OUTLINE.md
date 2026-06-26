# `waterSpec`: Comprehensive User Guide Outline

## Chapter 1: Introduction to Spectral Analysis in Hydrology
*   **1.1 What is waterSpec?**
    *   Package scope, goals, and intended audience.
*   **1.2 Theoretical Background**
    *   Time domain vs. Frequency domain.
    *   Understanding the Spectral Slope (β) and its physical meaning (White vs. Pink vs. Brown noise).
    *   Memory, persistence, and fractional Brownian motion (fBm) vs. fractional Gaussian noise (fGn).
*   **1.3 The Challenge of Environmental Data**
    *   Dealing with irregular sampling, gaps, and sensor jitter.
    *   Why traditional FFT methods fail on uneven data.

## Chapter 2: Installation and Setup
*   **2.1 Requirements & Dependencies** (Python versions, `mannks`, `ruptures`, etc.)
*   **2.2 Installation Methods** (Standard pip, development/test environment).
*   **2.3 Validating the Installation** (Running the test suite, `benchmark` scripts).

## Chapter 3: Data Ingestion & Preprocessing
*   **3.1 Loading Data**
    *   Supported formats (CSV, Excel, JSON).
    *   Parsing temporal formats (epochs, datetimes, relative elapsed times).
*   **3.2 Handling Messy Data**
    *   Identifying and stripping `NaN` and `Inf` values.
    *   Censored data resolution (e.g., handling "< 0.05" limits of detection).
*   **3.3 Data Transformation**
    *   Log-transformations (and handling zero/negative values).
    *   Detrending methodologies (Linear vs. LOESS smoothing).

## Chapter 4: Choosing Your Method: Lomb-Scargle vs. Haar
*   **4.1 The Lomb-Scargle Periodogram**
    *   Best use cases: Finding specific periodicities and harmonic signals.
    *   Assumptions and vulnerabilities (The uneven-sampling variance inflation problem).
*   **4.2 Haar Wavelet Fluctuation Analysis**
    *   Best use cases: Determining robust spectral slopes (β) on gappy data.
    *   How Haar maps to power-law scaling ($S(\tau) \propto \tau^m \rightarrow \beta = 2m + 1$).
*   **4.3 Decision Matrix:** A flowchart for selecting the right algorithm based on data properties (Sampling regularity, length, goals).

## Chapter 5: Core Univariate Analysis
*   **5.1 Performing a Standard Spectral Fit**
    *   Using the `Analysis` class.
    *   Fitting the power law (OLS vs. Theil-Sen/MannKS).
*   **5.2 Detecting Regime Shifts (Segmented Models)**
    *   When system memory changes across scales (e.g., event-scale vs. seasonal storage).
    *   Configuring `max_breakpoints` and interpreting the Bayesian Information Criterion (BIC) penalty.
*   **5.3 Peak Detection & Significance**
    *   Identifying narrow-band signals (e.g., Diurnal cycles).
    *   False Alarm Probabilities (FAP): Analytical (Baluev) vs. Bootstrapped limits.
    *   Correcting for multiple testing (Benjamini-Yekutieli FDR).

## Chapter 6: Advanced Haar Techniques
*   **6.1 Optimizing Window Generation**
    *   Overlapping vs. Non-overlapping windows (Trade-offs in variance vs. independence).
    *   Understanding the "Cone of Influence" and safe Maximum Lags ($T/2$).
*   **6.2 Small-Sample Corrections**
    *   Why EDOF (Effective Degrees of Freedom) matters.
    *   Applying the $c_4$ correction factor for unbiased standard deviation (`aggregation="std_corrected"`).
*   **6.3 Multifractal Intermittency**
    *   What to do when processes are "bursty" (Intermittent).
    *   Applying the $K(2)$ correction to prevent artificially flattened slopes.
*   **6.4 Analysis of Extremes**
    *   Moving beyond the mean: Using Percentiles (e.g., 90th percentile) to study extreme value scaling.

## Chapter 7: Bivariate and Multivariate Dynamics
*   **7.1 Cross-Spectra and Coherence**
    *   Analyzing phase lags (Lead/Lag times) between two time series (e.g., Rainfall vs. Discharge).
*   **7.2 Cross-Haar Correlation**
    *   Scale-dependent correlations (Do variables correlate more at daily or seasonal scales?).
*   **7.3 Hysteresis Classification**
    *   Quantifying loop area and direction (Clockwise vs. Counter-Clockwise) using the Shoelace formula.
    *   Zuecco et al. (2016) continuous normalization techniques.
*   **7.4 Partial Cross-Haar (Experimental)**
    *   Distinguishing direct correlations from spurious ones driven by a third variable.

## Chapter 8: Quantifying Uncertainty (Bootstrapping & Surrogates)
*   **8.1 The Need for Robust Error Bars**
*   **8.2 Non-Parametric Wild Bootstrapping**
    *   Moving block bootstrap and the Mammen (1993) two-point distribution for skewed residuals.
*   **8.3 Surrogate Data Testing**
    *   What are surrogates and what null hypotheses do they test?
    *   Phase-Randomized Surrogates (Tk95 / IAAFT).
    *   Block-shuffled / Permutation Surrogates.

## Chapter 9: Interpreting and Exporting Results
*   **9.1 The Interpretation Module**
    *   Translating mathematical outputs into environmental/hydrological meaning.
*   **9.2 Generating Reports**
    *   Using `ReportGenerator`.
    *   Navigating the HTML, JSON, and CSV output formats.
*   **9.3 Integrating with External Workflows**
    *   Parsing the output structures for downstream Pandas/R analysis.

## Chapter 10: Troubleshooting and FAQs
*   **10.1 Common Warnings Explained** (e.g., "Minimum effective sample size is X", "RSS extremely small").
*   **10.2 Debugging Failed Fits** (What to do when models don't converge).
*   **10.3 Understanding Matrix Singularities and `LinAlgError`**

## Appendix
*   **A. Library of Common Beta (β) Values** (Reference tables for common parameters like Nitrate, TSS, E.Coli).
*   **B. Mathematical Formulas and Derivations** (Strict definitions of Parseval's theorem in Tk95, Shoelace logic, etc.).
*   **C. Glossary of Terms**
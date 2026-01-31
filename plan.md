# Project Plan and Features

This document outlines the major features and improvements implemented in the `waterSpec` package.

## 1. Production Readiness Audit and Enhancements

A thorough audit of the `waterSpec` codebase was conducted to improve its robustness, correctness, and user experience for production-level use.

### Key Improvements:

*   **Corrected Error Propagation for Detrending**:
    *   The `preprocessor.detrend` function was rewritten to use `statsmodels.OLS`. This correctly handles unevenly spaced time series, which was a limitation of the previous `scipy.signal.detrend` implementation.
    *   Crucially, the new function now correctly propagates errors. It combines the original measurement uncertainty with the uncertainty from the linear fit in quadrature, leading to more accurate error estimates in the final results.

*   **Configurable Statistical Thresholds**:
    *   The `p-value` threshold for the Davies test for a significant breakpoint in the 1-breakpoint segmented regression is now configurable via the `p_threshold_davies` parameter in `run_full_analysis`. This provides users with more flexibility in the statistical analysis.

*   **Improved User Feedback**:
    *   Error messages have been enhanced to be more informative. For example, if the data loading fails due to a non-monotonic time series, the error now includes the specific timestamp that caused the violation, making it easier for users to debug their input data.

*   **Validation Script**:
    *   A new end-to-end validation script has been added at `validation/validate_analysis.py`.
    *   This script runs a full analysis on a sample dataset and compares the output summary against a known-good reference file. This serves as a powerful regression test to ensure that changes to the codebase do not unintentionally alter the final results.

## 2. Multi-Breakpoint Segmented Regression

This feature extends the spectral analysis capabilities to automatically fit and compare models with more than one breakpoint.

### Feature Details:

*   **Automatic Model Selection**:
    *   The `run_full_analysis` method now accepts a `max_breakpoints` parameter (defaulting to `1` for backward compatibility).
    *   When `max_breakpoints` is set to `2`, the analysis pipeline will automatically fit three separate models:
        1.  A standard linear fit (0 breakpoints)
        2.  A segmented fit with 1 breakpoint
        3.  A segmented fit with 2 breakpoints
    *   The best model is automatically selected by comparing the **Bayesian Information Criterion (BIC)** of all successfully fitted models. The model with the lowest BIC is chosen.

*   **Flexible Output and Visualization**:
    *   The text summary (`summary.txt`) has been updated to dynamically display the comparison between all tested models, clearly indicating their BIC scores and which model was chosen.
    *   The spectral plot (`spectrum_plot.png`) has been enhanced to correctly visualize a fit with any number of breakpoints, plotting each segment and breakpoint with a clear label.

*   **Generalized Implementation**:
    *   The core fitting, interpretation, and plotting functions have been refactored to be more general, paving the way for potentially supporting more than two breakpoints in the future.

## 3. Multi-Scalar Haar Analysis & Bivariate Tools

This feature set implements the advanced framework for analyzing non-stationary water quality time series, focusing on physical interpretability and robustness to irregular sampling.

### Completed Implementations:

*   **Robust Haar Analysis with Overlapping Windows**:
    *   Refactored `haar_analysis.py` to support **overlapping windows** (sliding time steps). This maximizes the effective sample size for long-term records, which is critical for statistically significant results at large scales (e.g., decadal).
    *   Implemented effective sample size ($n_{eff}$) calculation to account for the dependence induced by overlap.

*   **Segmented Haar Scaling**:
    *   Added `fit_segmented_haar` using robust regression (MannKS) to detect **regime shifts** in system memory.
    *   The system can now automatically identify characteristic scales (breakpoints in the $S_1(\tau)$ vs $\tau$ plot) where the dominant transport mechanism changes (e.g., from surface runoff to groundwater baseflow).

*   **Bivariate Analysis Framework (Cross-Haar)**:
    *   Created `waterSpec.bivariate` module with a `BivariateAnalysis` class.
    *   **Time Alignment:** Supports aligning two distinct time series (e.g., Concentration vs. Discharge) with configurable tolerance and methods (`nearest`, `interpolate`).
    *   **Cross-Haar Correlation:** Computes the correlation between fluctuations of two variables at specific time scales ($\rho_{CQ}(\tau)$).
    *   **Lagged Response:** Computes cross-correlation at various time lags to identify response delays (e.g., how long after a discharge peak does the concentration respond?).

*   **Surrogate Significance Testing**:
    *   Created `waterSpec.surrogates` module.
    *   Implemented **Phase Randomization** (preserves spectrum/autocorrelation, destroys nonlinearity/phase) and **Block Shuffling** (preserves short-term distribution, destroys long-term memory).
    *   These tools allow for rigorous significance testing of Cross-Haar correlations, avoiding the pitfalls of standard p-values in autocorrelated environmental data.

*   **Integrated Workflow**:
    *   The main `Analysis.run_full_analysis` method now accepts parameters to control Haar overlap (`haar_overlap`) and segmentation (`haar_max_breakpoints`), making these advanced features accessible via the standard API.

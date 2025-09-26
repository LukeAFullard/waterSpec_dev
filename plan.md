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
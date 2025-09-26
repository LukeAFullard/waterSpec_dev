# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] - 2025-09-25

### Added

-   Initial release of the `waterSpec` package.
-   Core `Analysis` class for a complete, end-to-end spectral analysis workflow.
-   Data loading from CSV, Excel, and JSON files with robust validation.
-   Comprehensive preprocessing module (`preprocessor.py`) supporting censored data handling, linear and LOESS detrending, log-transformation, and normalization.
-   Spectral analysis using `astropy.LombScargle` for irregularly sampled data.
-   Advanced spectral fitting module (`fitter.py`) featuring:
    -   Robust Theil-Sen regression for spectral slope (β) calculation.
    -   Uncertainty estimation for β via residual bootstrapping.
    -   Segmented regression to detect multifractal behavior.
    -   Automatic model selection (standard vs. segmented) using BIC.
-   Data-driven peak significance testing based on residuals from the fitted spectral model.
-   Automated, publication-ready interpretation module (`interpreter.py`) with scientific context, benchmark comparisons, and a persistence traffic-light system.
-   High-quality plotting module (`plotting.py`) that generates a single, annotated figure containing the spectrum, fitted models, confidence intervals, significant peaks, and a full text summary.
-   A validation suite that benchmarks the package's performance against the established `dplR` R package.
-   Project documentation including an improved `README.md`, `CONTRIBUTING.md`, and `CHANGELOG.md`.
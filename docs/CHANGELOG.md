# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
-   Added `src/waterSpec/benchmarks.json` to store scientific benchmark data for spectral slopes.
-   Added `scripts/show_benchmarks.py` to display these benchmarks to the user.

### Changed
-   Refactored `src/waterSpec/interpreter.py` to load benchmark data from the external JSON file.

## [0.1.0] - 2025-09-25

### Added

-   Initial release of the `waterSpec` package.
-   Core `Analysis` class for running a full spectral analysis workflow.
-   Support for Lomb-Scargle periodogram for unevenly spaced time series.
-   Automated model selection between standard and segmented (multifractal) fits using BIC.
-   Robust, data-driven peak significance testing using residuals from the noise model fit.
-   Functions for data loading (CSV, Excel, JSON), preprocessing (detrending, censoring), and interpretation.
-   Comprehensive plotting utility to visualize the spectrum, fit, and results.
-   Example notebook (`demo.ipynb`) and sample datasets.
-   Unit tests and validation scripts.
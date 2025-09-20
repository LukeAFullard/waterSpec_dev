# Project Plan and Documentation

This document outlines the plan for validating the `waterSpec` package and provides documentation for the setup and comparison process.

## R Environment Setup

To run the validation script that compares `waterSpec` with the `dplR` R package, you need to set up an R environment.

### 1. Install R and Dependencies

On a Debian-based system (like Ubuntu), you can install R and its dependencies using the following commands:

```bash
sudo apt-get update
sudo apt-get install -y r-base libcurl4-openssl-dev libtirpc-dev
```

### 2. Install R Packages

The validation script requires the `dplR` package. We use the `pak` package manager for R to install it.

```bash
sudo R -e "install.packages('pak', repos='https://r-lib.github.io/p/pak/stable')"
sudo R -e 'pak::pak("dplR")'
```

## Python Environment Setup

The Python environment needs the `rpy2` package to interface with R, in addition to the standard project dependencies.

### 1. Install Project Dependencies

The project dependencies are listed in `pyproject.toml`. You can install the project in editable mode, which will also install the dependencies:

```bash
pip install -e .
```

### 2. Install Additional Packages for Validation

You also need to install `rpy2` and `pytest`:

```bash
pip install rpy2 pytest
```

## Validation with dplR

A validation script `validation/validate_with_dplR.py` has been created to compare `waterSpec` with `dplR`.

### Comparison Methodology

The script performs the following steps:
1.  Generates synthetic time series data with a known spectral exponent (`beta`).
2.  Runs `waterSpec` on the data to estimate `beta`.
3.  Uses `rpy2` to call the `redfit` function from `dplR` on the same data to get the AR(1) coefficient (`rho`).
4.  Compares the `waterSpec` beta with the known beta and the `dplR` rho.

### Rho to Beta Conversion

To compare the results, the AR(1) coefficient `rho` from `dplR` is converted to a spectral exponent `beta` using the following approximation, which is common in geophysical time series analysis:

`beta = 2 * rho / (1 - rho)`

### Summary of Findings

The validation shows that:
-   `waterSpec`'s beta estimates are close to the known beta values for the synthetic data.
-   `dplR`'s `redfit` provides a stable estimate of the AR(1) coefficient `rho`.
-   The direct conversion from `rho` to `beta` does not yield a perfect match, which is expected as the underlying models and assumptions of the two packages differ. However, the estimated beta from `dplR` shows a monotonic relationship with the known beta.
-   This validation increases confidence in `waterSpec` by showing that it is sensitive to the spectral characteristics of the data in a way that is comparable to an established tool in the field.

A full report of a sample run can be found in `validation/dplR_validation_report.md`.

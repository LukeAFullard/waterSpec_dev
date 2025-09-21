# Plan for Python Package: Spectral Scaling of Environmental Time Series

## Motivation and References

### Motivation
Building a Python package for spectral analysis of water quality data, specifically to calculate and interpret the scaling exponent (β), is motivated by the need to analyze irregularly sampled time series data common in environmental monitoring. In hydrology and environmental science, temporal scaling reveals fractal behaviors in variables like pollutant concentrations, helping identify dominant transport pathways (e.g., surface runoff vs. subsurface drainage), persistence (correlation over time), and periodicities (e.g., seasonal cycles). This can inform conservation planning, such as matching practices to pathways in agricultural watersheds to reduce nutrient/sediment export.

The core inspiration comes from Liang et al. (2021), which uses spectral analysis on Raccoon River data to show increasing β from bacteria (0.27, surface-dominated) to nitrate (1.73, subsurface-dominated), indicating chemostatic behavior for nitrate due to large soil reservoirs. Extending this to a reusable package allows users to apply similar analyses to their data, handling gaps without interpolation, and interpreting results for practical insights (e.g., low β suggests event-driven runoff controls like buffers; high β suggests subsurface interventions like bioreactors).

Broader applications include anomaly detection in hydrological signals, uncertainty quantification in models, and scaling parameter estimation for simulations. This package promotes reproducible science by encapsulating methods in an installable tool, following best practices for data science workflows.

### References and Relevant Information
- **Primary Reference**: Liang X, Schilling KE, Jones CS, Zhang Y-K. 2021. Temporal scaling of long-term co-occurring agricultural contaminants and the implications for conservation planning. *Environmental Research Letters* 16:094015. DOI: 10.1088/1748-9326/ac19dd.
- **Key Hydrology References**:
  - Kirchner JW, Feng X, Neal C. 2000. Fractal scaling of catchment solute transport. *PNAS* 97:14265–14270.
  - Gelhar LW. 1974. Stochastic analysis of phreatic aquifers. *WRR* 10:539–545.
  - Aubert AH et al. 2014. (Multifractal nitrate patterns.)
  - Feder J. 1988. *Fractals*. Plenum Press.
  - Turcotte DL. 1992. *Fractals and Chaos in Geology and Geophysics*. Cambridge University Press.
  - Mandelbrot BB, Van Ness JW. 1968. *Fractional Brownian motions, fractional noises and applications*. SIAM Review 10:422–437.
- **Additional References**:
  - Montanari A et al. 2019. Improving spectral analysis of hydrological signals.
  - Hartmann A et al. 2017. Temporal scale-dependent sensitivity in hydrological models.
  - Kang S, Lin H. 2007. Wavelet analysis in hydrology.
- **Python Implementation References**:
  - SciPy Documentation: `lombscargle`.
  - Astropy Timeseries: Lomb-Scargle tutorial (preferred for robustness).
  - VanderPlas J. 2015/2017. Fast Lomb-Scargle in Python.
- **Package Building Best Practices**:
  - Python Packaging Authority guides.
  - Rosenthal E. 2022. Everything gets a package.
  - Cook J. 2024. Organizing data analyses with packages.
  - Real Python. Using Python for data analysis.
- **Technical Notes**:
  - Use Python 3.8+.
  - Handle uneven data with Lomb-Scargle (Astropy preferred).
  - Interpret β:
    - 0 < β < 1: fGn, weak persistence, event-driven.
    - 1 < β < 3: fBm, strong persistence, damped.
    - β ~ 0: white noise.
    - β ~ 2: Brownian.
  - Assumptions: Stationarity; data >100 points for reliability.
  - Limitations: Aliasing at low frequencies; fitting ranges critical.

---

## Step-by-Step Plan to Build the Python Package

We will call the package **`waterSpec`** (preferred name). Alternatives considered: `hydroscale`, `aquaSpectra`, `temposcale`.

### Step 1: Set Up Development Environment
1.  Create project directory and Git repo.
2.  Create virtual environment (`python -m venv venv`).
3.  Install dependencies: `numpy`, `scipy`, `pandas`, `matplotlib`, `astropy`, `setuptools`, `wheel`, `twine`, `pytest`.
4.  Save dependencies in `requirements.txt`.
5.  Organize folder structure:
    ```
    waterSpec/
    |-- __init__.py
    |-- data_loader.py
    |-- preprocessor.py
    |-- spectral_analyzer.py
    |-- fitter.py
    |-- interpreter.py
    |-- utils.py
    |-- tests/
    |   |-- test_analysis.py
    |-- examples/
    |   |-- example_data.csv
    |   |-- demo.ipynb
    |-- docs/
    |-- README.md
    |-- setup.py
    |-- LICENSE
    |-- MANIFEST.in
    ```

### Step 2: Data Loading (`data_loader.py`)
- **Flexible Data Formats**: Functions to load data from various formats including CSV, TSV, Excel (`.xlsx`), and JSON files. The function will auto-detect the format from the file extension.
- **Time Column Parsing**: Parse time columns and convert them to a numeric representation (e.g., seconds since epoch).
- **Validation**: Ensure time is monotonic and there are no `NaN` values in the final numeric arrays.
- **Output**: Return numpy arrays (t, y) for direct use in analysis functions.

### Step 3: Preprocessing (`preprocessor.py`)
- **Detrending**: Provide options for both linear and non-linear trend removal. Non-linear detrending will be handled using a LOESS smoother from the `statsmodels` library.
- **Standard Options**: Provide functions for other common preprocessing tasks like normalizing and log-transformation.
- **Censored Data Handling**: Implement strategies for handling censored data common in environmental monitoring (e.g., values reported as `<DL` or `>UL`).
  - **Strategy 1 (Ignore)**: Remove censor marks and use the raw numeric value.
  - **Strategy 2 (Multiplier)**: Replace censored values with the detection/quantification limit multiplied by a user-defined factor. For example, `<5` could be replaced with `5 * 0.5`.
  - The chosen strategy will be a parameter in the preprocessing function.
- **Gap Handling**: Handle irregular gaps by passing them directly to the Lomb-Scargle algorithm (no interpolation).
- **Edge Case Checks**: Validate data for sufficient length (e.g., >10 points) and handle cases with all `NaN` values.

### Step 4: Spectral Analysis (`spectral_analyzer.py`)
- Default to **Astropy’s LombScargle** (robust handling of irregular sampling, normalization, significance).
- Fallback: SciPy `lombscargle`.
- Compute periodogram on log-spaced frequency grid.
- Add peak detection for periodicities.
- Option: cross-spectral analysis for discharge vs. concentration (for chemostatic behavior).
- Add FFT-based option for evenly spaced data.

### Step 5: Fitting (`fitter.py`)
- **Log-log Linear Regression**: Fit a single line to the power spectrum on a log-log plot to determine the primary spectral exponent (β).
- **Segmented Regression**: For detecting multifractal behavior (i.e., changes in scaling), implement auto-crossover detection using segmented regression. This will be achieved using the `piecewise-regression` Python package to find breakpoints ("knees") in the log-log spectrum.
- **Outputs**: The fitting functions will return key metrics such as β, R-squared, standard error, and breakpoint locations for segmented fits.

**Uncertainty Handling**:
- **Bootstrap resampling**: resample residuals or blocks of time series; re-estimate β to get confidence intervals.
- **Monte Carlo surrogates**: generate surrogate time series (e.g., phase randomization) to test significance of scaling.
- **Jackknife/leave-one-out**: test sensitivity of β to missing data.
- **Frequency band sensitivity**: compute β across multiple overlapping frequency windows and report variation.
- **Bayesian estimation (optional)**: use Bayesian regression to estimate posterior distribution of β.

Outputs: point estimate, 95% CI (bootstrap), optional posterior summary.

### Step 6: Interpretation (`interpreter.py`)

#### Goals
- Provide **automated, easy-to-understand interpretations** of β values.
- Include **uncertainty-aware summaries**, benchmark comparisons, and conservation suggestions.
- Offer **visual and textual outputs** for accessibility.

#### Features
1.  **Automated Summary Function**
    ```python
    from waterSpec import interpret_results

    results = interpret_results(beta=1.7, ci=(1.5, 1.9), param_name="Nitrate")
    print(results["text"])
    ```
    **Example output:**
    ```
    Nitrate (β = 1.7, 95% CI [1.5–1.9])
    Interpretation: Strong persistence, subsurface-dominated transport (fBm-like).
    Similar to nitrate behavior in Liang et al. (2021).
    Suggested focus: subsurface interventions (e.g., bioreactors, drainage management).
    ```

2.  **Benchmark Comparison Table**
    Automatically compares β against known pollutants. It highlights the “closest match” for the user’s variable.

| Parameter | Typical β Range | Interpretation | Dominant Pathway |
| :--- | :--- | :--- | :--- |
| E. coli | 0.1 – 0.5 | Weak persistence | Surface runoff |
| TSS | 0.4 – 0.8 | Weak persistence | Surface runoff |
| Ortho-P | 0.6 – 1.2 | Mixed | Surface/Shallow subsurface |
| Chloride | 1.3 – 1.7 | Strong persistence | Subsurface |
| Nitrate-N | 1.5 – 2.0 | Strong persistence | Subsurface |
| Discharge (Q) | 1.0 – 1.8 | Persistent | Integrated signal |

3.  **Uncertainty-Aware Messages**
    > If the confidence interval is wide:
    > “Interpretation uncertain: the β range spans both event-driven and subsurface processes. More data may be needed to draw a firm conclusion.”

4.  **Traffic-Light System for Persistence**
    - **β < 0.5 → 🔴 Event-driven**
    - **0.5 – 1.0 → 🟡 Mixed / weak persistence**
    - **β > 1.0 → 🟢 Persistent / subsurface dominated**

5.  **Optional Plot Overlay**
    - Automatically plot spectrum with β fit line.
    - Annotate with a text box summarizing the interpretation.

6.  **Custom Domain Profiles**
    - Hydrology (runoff vs subsurface)
    - Ecology (short vs long ecological memory)
    - Climate (noise vs oscillations)

### Step 7: Integration and Plotting
- Expose functions in `__init__.py`.
- Add plotting utilities (log-log plots, fits, annotated peaks, confidence bands).
- Templates styled after Liang et al. (2021).

### Step 8: Documentation and Examples
- **README**: Install, quick-start example.
- **Example notebook**: load sample data → preprocess → compute spectrum → fit β with uncertainty → interpret.
- Sphinx-compatible docstrings.
- Provide tutorials for hydrology users.

### Step 9: Testing (`tests/`)
- Synthetic test cases:
  - White noise (β ≈ 0).
  - Random walk (β ≈ 2).
  - Known fGn/fBm simulations.
  - Sine wave with gaps (check peak detection).
- Validate uncertainty estimates (bootstrap CI covers true β in simulations).
- Validate automated interpretation outputs.
- Use `pytest`.

### Step 10: Packaging and Distribution
- `setup.py` and `pyproject.toml`.
- Build: `python -m build`.
- Install locally: `pip install .`.
- Upload to PyPI (optional).
- Semantic versioning.

### Step 11: Validation and Iteration
- Test with article-like data.
- Confirm β matches expectations.
- Validate bootstrap and surrogate-based uncertainty.
- Confirm automated interpretations are intuitive for users.
- Add error handling and logging.

### Extensions (Future Work)

Based on a detailed review of the Lomb-Scargle method and the `astropy` implementation, the following features are recommended to significantly improve the scientific rigor and utility of the `waterSpec` package.

#### 1. Statistical Significance Testing (False Alarm Probability)

-   **Description:** Implement the calculation of the False Alarm Probability (FAP) to determine the statistical significance of detected peaks in the periodogram. A low FAP (e.g., < 0.01) indicates a high confidence that a peak is a true periodic signal and not a result of random noise.
-   **Value:** This is a critical feature for scientific validity, allowing users to differentiate between meaningful periodicities and statistical artifacts.
-   **Implementation:**
    -   Use the `astropy.LombScargle.false_alarm_level` method, preferably with the robust `'bootstrap'` option.
    -   Add a `fap_threshold` parameter to `run_analysis` to filter for significant peaks.
    -   Report the FAP of significant peaks in the results dictionary and plot annotations.

#### 2. Support for Measurement Uncertainties (`dy`)

-   **Description:** Allow users to provide per-point measurement errors for their time series data. The Lomb-Scargle algorithm can use these errors (`dy`) to perform a weighted analysis, giving more influence to more certain data points.
-   **Value:** Most environmental data has associated measurement uncertainty. Incorporating it will lead to more statistically robust and accurate spectral estimates. This is also a prerequisite for PSD normalization.
-   **Implementation:**
    -   Add an optional `error_col` argument to the `load_data` function.
    -   Pass the `dy` array through the `preprocess_data` and `run_analysis` functions to the `LombScargle` constructor.

#### 3. Generalized Lomb-Scargle (Multi-Term Models)

-   **Description:** Expose the `nterms` parameter of the `LombScargle` constructor. This allows the algorithm to fit a more complex model composed of multiple Fourier terms, which can better capture the shape of non-sinusoidal periodic signals.
-   **Value:** Real-world signals (like seasonal environmental cycles) are often not perfect sine waves. This feature would enable more accurate modeling of such signals.
-   **Implementation:**
    -   Add an `nterms` parameter to the `calculate_periodogram` and `run_analysis` functions.
    -   Document the limitation that Astropy's FAP calculation is not implemented for `nterms > 1`.

#### 4. PSD Normalization and Model Visualization

-   **Description:**
    -   **PSD Normalization:** Allow the periodogram power to be normalized as a true Power Spectral Density (PSD), giving the y-axis physical units (e.g., `concentration² / frequency`).
    -   **Model Visualization:** Add a new plotting utility to show the best-fit Lomb-Scargle model overlaid on the original data, phased to a specific peak's frequency.
-   **Value:** PSD normalization aids in the physical interpretation and comparison of results. Model visualization provides a powerful diagnostic tool to visually confirm the quality of a periodic fit.
-   **Implementation:**
    -   Expose the `normalization` parameter in `run_analysis` and `calculate_periodogram`.
    -   Create a new `plot_phased_model` function that uses the `astropy.LombScargle.model()` method.

---
- **Advanced Censored Data Methods**: Implement more statistically robust methods for handling censored data, such as distribution fitting or survival analysis techniques.
- **Wavelet Analysis**: Add wavelet analysis for time-frequency scaling.
- **Cross-Spectral Analysis**: Implement cross-spectral analysis for comparing pollutant vs. discharge coherence.
- **Multi-Pollutant Workflows**: Create workflows to easily compare β across multiple variables in a dataset.
- **Visualization Templates**: Develop more advanced visualization templates that reproduce plots from key literature like Liang et al. (2021).
- **Educational Modules**: Build educational modules and tutorials for hydrology students and professionals.

---

## Additional Suggestions for the Plan

To further enhance the project's quality and maintainability, consider incorporating the following steps into your development workflow:

1.  **Continuous Integration/Continuous Deployment (CI/CD)**
    - **What:** Set up a CI pipeline using a service like GitHub Actions.
    - **Why:** To automatically run tests, check code formatting, and build the package on every push and pull request. This ensures that new changes don't break existing functionality and maintain a high standard of code quality. It can also automate publishing the package to PyPI upon creating a new release.

2.  **Code Quality and Style Enforcement**
    - **What:** Integrate automated code formatting (e.g., `black`) and linting (e.g., `ruff` or `flake8`) into the development process.
    - **Why:** This enforces a consistent code style across the entire project, making the code more readable and easier to maintain. Using pre-commit hooks can automate this process, ensuring that all committed code adheres to the style guide.

3.  **Community and Contribution Guidelines**
    - **What:** Create `CONTRIBUTING.md` and `CODE_OF_CONDUCT.md` files in the root of the repository.
    - **Why:** A `CONTRIBUTING.md` file provides clear guidelines for others on how to contribute to your project, report issues, and submit pull requests. A `CODE_OF_CONDUCT.md` helps foster a positive and inclusive community around the project.

4.  **Changelog Management**
    - **What:** Maintain a `CHANGELOG.md` file.
    - **Why:** This file provides a clear and chronological record of all notable changes made to the project, such as new features, bug fixes, and performance improvements. It is invaluable for users and contributors to understand the evolution of the package between different versions.

5.  **Modern Python Packaging**
    - **What:** Consolidate project configuration into `pyproject.toml`.
    - **Why:** The `pyproject.toml` file is the new standard for configuring Python packages. It can replace `setup.py`, `setup.cfg`, `requirements.txt`, and configuration files for tools like `pytest`, `black`, and `ruff`. This simplifies project management by having a single source of truth for metadata, dependencies, and tool settings.

## Summary

This plan provides an end-to-end workflow to build a Python package (`waterSpec`) that enables robust spectral analysis of irregularly sampled environmental time series. It integrates best practices (testing, packaging, documentation), scientific rigor (fractal scaling, β interpretation, uncertainty), and practical value (domain-specific interpretations for conservation planning).

By including an automated interpretation module, the package ensures results are easy to understand, actionable, and useful for both researchers and practitioners. The additional suggestions aim to improve the development process, making it more robust, collaborative, and maintainable in the long run.

---

## Recently Completed Tasks

### Task: Refactor Frequency Grid Generation

**Goal:** Improve the code structure by moving the frequency grid generation logic to a separate module.

**Plan:**

1.  **Create New Module:** Create a new file `src/waterSpec/frequency_generator.py`.
2.  **Move Logic:** Move the frequency grid generation logic from `src/waterSpec/workflow.py` to a new function `generate_log_spaced_grid` in the new module.
3.  **Update Workflow:** Update `src/waterSpec/workflow.py` to use the new function.
4.  **Verify:** Run the test suite to ensure no regressions were introduced.

---

### Task: Add Flexible LOESS Detrending Options

**Goal:** Allow users to have more control over the non-linear LOESS detrending by exposing more parameters from the `statsmodels.lowess` function.

**Plan:**

1.  **Update `detrend_loess`:** Modify the function to accept `**kwargs` and pass them to `statsmodels.lowess`.
2.  **Update `preprocess_data` and `run_analysis`:** Add a `detrend_options` dictionary parameter to these functions to pass the options down.
3.  **Update Tests:** Add a test to verify that custom options are correctly passed and have an effect.

---

### Task: Complete Post-Audit Refinements

**Note:** The tasks in this section were originally listed as "Ongoing" but were found to be complete during a subsequent audit.

**Goal:** Address key weaknesses and missing features identified in a prior audit to improve scientific validity and usability.

**Plan:**

1.  **Refine `spectral_analyzer.py`:**
    *   **Action:** Modify `calculate_periodogram` to remove the unsafe `autopower` default, requiring a frequency grid to be passed explicitly. This prevents misuse and incorrect scientific results.
    *   **Action:** Add support for measurement uncertainties (`dy`). This involves updating `data_loader.py` to read an error column and passing this information through the `workflow.py` to the `LombScargle` constructor.
    *   **Action:** Implement False Alarm Probability (FAP) calculations to assess the statistical significance of detected peaks.

2.  **Enhance `preprocessor.py`:**
    *   **Action:** Integrate `normalize()` and `log_transform()` into the main `preprocess_data` function. This will be done by adding boolean flags (`normalize=False`, `log_transform=False`) to the function signature for easier access.

3.  **Enhance `plotting.py`:**
    *   **Action:** Add the automated text summary from `interpreter.py` as an annotation on the plot. This will make the visualization a complete, self-contained result.

4.  **Update Tests:**
    *   **Action:** Update all tests affected by the above changes and add new tests to cover the new functionality (e.g., `dy` support, FAP, new preprocessor options, plot annotations).

---

## Issues Discovered During Development

This section documents issues and design improvements that were discovered and fixed during the development and tutorial-writing process. This is a normal part of building robust software.

1.  **Preprocessor Flexibility (`censor_options`)**
    *   **Issue:** The main `preprocess_data` wrapper function did not provide a way to pass options (like `lower_multiplier`) to the underlying `handle_censored_data` function.
    *   **Fix:** Added a `censor_options` dictionary parameter to `preprocess_data` and the main `run_analysis` workflow, allowing users to pass detailed options for the censoring strategy.

2.  **Plotting Label Robustness**
    *   **Issue:** The plotting function would raise a `ValueError` if it tried to format the FAP threshold for the plot legend when the threshold value was not available.
    *   **Fix:** Added a check in `plotting.py` to ensure the FAP threshold value is a number before attempting to format it, making the function more robust.

3.  **Log-Spaced Grid for Peak Detection**
    *   **Issue:** It was discovered that using a log-spaced frequency grid, while optimal for fitting the spectral slope (β), can be problematic for peak detection. The high density of points at low frequencies can cause the DC component or minor trends to create a large, spurious peak at or near frequency zero, masking real periodic signals.
    *   **Fix/Workaround:** The FAP tutorial (`06-advanced_peak_finding.ipynb`) was updated to use a **linear** frequency grid for its example, which is better suited for resolving specific peaks. A note was added explaining this distinction to the user. A proper fix to the `generate_log_spaced_grid` function may be considered in the future to make it more robust for general use.

---

## Validation with dplR (Completed Task)

This section documents the setup and comparison process for validating the `waterSpec` package against the `dplR` R package.

### R Environment Setup

To run the validation script, you need to set up an R environment.

#### 1. Install R and Dependencies
On a Debian-based system (like Ubuntu), you can install R and its dependencies using the following commands:

```bash
sudo apt-get update
sudo apt-get install -y r-base libcurl4-openssl-dev libtirpc-dev
```

#### 2. Install R Packages
The validation script requires the `dplR` package. We use the `pak` package manager for R to install it.

```bash
sudo R -e "install.packages('pak', repos='https://r-lib.github.io/p/pak/stable')"
sudo R -e 'pak::pak("dplR")'
```

### Python Environment Setup

The Python environment needs the `rpy2` package to interface with R, in addition to the standard project dependencies.

#### 1. Install Project Dependencies
The project dependencies are listed in `pyproject.toml`. You can install the project in editable mode, which will also install the dependencies:

```bash
pip install -e .
```

#### 2. Install Additional Packages for Validation
You also need to install `rpy2` and `pytest`:

```bash
pip install rpy2 pytest
```

### Comparison Methodology

A validation script `validation/validate_with_dplR.py` has been created to compare `waterSpec` with `dplR`. The script performs the following steps:
1.  Generates synthetic time series data with a known spectral exponent (`beta`).
2.  Runs `waterSpec` on the data to estimate `beta`.
3.  Uses `rpy2` to call the `redfit` function from `dplR` on the same data to get the AR(1) coefficient (`rho`).
4.  Compares the `waterSpec` beta with the known beta and the `dplR` rho.

#### Rho to Beta Conversion
To compare the results, the AR(1) coefficient `rho` from `dplR` is converted to a spectral exponent `beta` using the following approximation, which is common in geophysical time series analysis:

`beta = 2 * rho / (1 - rho)`

### Summary of Findings

The validation shows that:
-   `waterSpec`'s beta estimates are close to the known beta values for the synthetic data.
-   `dplR`'s `redfit` provides a stable estimate of the AR(1) coefficient `rho`.
-   The direct conversion from `rho` to `beta` does not yield a perfect match, which is expected as the underlying models and assumptions of the two packages differ. However, the estimated beta from `dplR` shows a monotonic relationship with the known beta.
-   This validation increases confidence in `waterSpec` by showing that it is sensitive to the spectral characteristics of the data in a way that is comparable to an established tool in the field.

A full report of a sample run can be found in `validation/dplR_validation_report.md`.

---

### Future Validation: Significant Peak Detection with dplR (To Do)

**Goal:** To validate that the significant peak detection in `waterSpec` (using False Alarm Probability) is comparable to the significance testing in `dplR`'s `redfit` function.

**Status:** The Python script for this validation (`validation/validate_peak_detection.py`) has been written. However, it cannot be executed in an environment that does not have the R programming language and the `dplR` R package installed.

**Required Action:** To complete this validation, run the script in a suitable environment by executing `python validation/validate_peak_detection.py`.

**Methodology:**

1.  **Synthetic Data Generation:** A synthetic time series will be created with two components:
    *   A "red noise" background with a known spectral exponent (e.g., `beta = 1.5`).
    *   A pure sine wave of a known frequency and amplitude injected into the noise.

2.  **Analysis with `waterSpec`:** The synthetic data will be analyzed using `waterSpec.Analysis`. The test will verify that the known frequency of the injected sine wave is identified as a significant peak according to the specified `fap_threshold`.

3.  **Analysis with `dplR`'s `redfit`:** The same synthetic data will be passed to the `redfit` function via the `rpy2` bridge. The test will verify that the power of the known frequency lies above the 95% or 99% confidence curve generated by `redfit`.

4.  **Comparison:** The validation is considered successful if both packages are able to consistently identify the known, injected signal as statistically significant. This comparison will provide high confidence that `waterSpec`'s FAP-based method is performing correctly and is benchmarked against established methods in the field.

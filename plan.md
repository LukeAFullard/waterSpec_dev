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
1. Create project directory and Git repo.
2. Create virtual environment (`python -m venv venv`).
3. Install dependencies: `numpy`, `scipy`, `pandas`, `matplotlib`, `astropy`, `setuptools`, `wheel`, `twine`, `pytest`.
4. Save dependencies in `requirements.txt`.
5. Organize folder structure:

   waterSpec/
init.py
data_loader.py
preprocessor.py
spectral_analyzer.py
fitter.py
interpreter.py
utils.py
tests/
test_analysis.py
examples/
example_data.csv
demo.ipynb
docs/
README.md
setup.py
LICENSE
MANIFEST.in


### Step 2: Data Loading (`data_loader.py`)
- Functions to load CSV/TSV, parse time columns, convert to numeric days/seconds.
- Validation: monotonic time, no NaNs.
- Return numpy arrays (t, y).

### Step 3: Preprocessing (`preprocessor.py`)
- Options: detrend, normalize, log-transform.
- Handle irregular gaps (pass to Lomb-Scargle, no interpolation).
- Edge checks (len < 10, all NaNs).

### Step 4: Spectral Analysis (`spectral_analyzer.py`)
- Default to **Astropy’s LombScargle** (robust handling of irregular sampling, normalization, significance).
- Fallback: SciPy `lombscargle`.
- Compute periodogram on log-spaced frequency grid.
- Add peak detection for periodicities.
- Option: cross-spectral analysis for discharge vs. concentration (for chemostatic behavior).
- Add FFT-based option for evenly spaced data.

### Step 5: Fitting (`fitter.py`)
- Log-log linear regression on spectrum.
- Auto-crossover detection (segmented regression / knee detection) for multifractals.
- Return β, R², stderr, fit line.

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
1. **Automated Summary Function**  
```python
from waterSpec import interpret_results

results = interpret_results(beta, ci=(1.5,1.9), param_name="Nitrate")
print(results["text"])

Example output:
Nitrate (β = 1.7, 95% CI [1.5–1.9])
Interpretation: Strong persistence, subsurface-dominated transport (fBm-like).
Similar to nitrate behavior in Liang et al. (2021).
Suggested focus: subsurface interventions (e.g., bioreactors, drainage management).

Benchmark Comparison Table

Automatically compares β against known pollutants (E. coli, TSS, OP, Cl, NO₃-N, Q).

Highlights “closest match” for user’s variable.

Uncertainty-Aware Messages

If CI is wide:

“Interpretation uncertain: β range spans event-driven and subsurface processes. More data recommended.”

Traffic-Light System for Persistence

β < 0.5 → 🔴 Event-driven

0.5–1.0 → 🟡 Mixed / weak persistence

1.0 → 🟢 Persistent / subsurface dominated

Optional Plot Overlay

Automatically plot spectrum with β fit line.

Annotate with text box summarizing interpretation.

Custom Domain Profiles

Hydrology (runoff vs subsurface)

Ecology (short vs long ecological memory)

Climate (noise vs oscillations)

Step 7: Integration and Plotting

Expose functions in __init__.py.

Add plotting utilities (log-log plots, fits, annotated peaks, confidence bands).

Templates styled after Liang et al. (2021).

Step 8: Documentation and Examples

README: install, quick-start example.

Example notebook: load sample data → preprocess → compute spectrum → fit β with uncertainty → interpret.

Sphinx-compatible docstrings.

Provide tutorials for hydrology users.

Step 9: Testing (tests/)

Synthetic test cases:

White noise (β ≈ 0).

Random walk (β ≈ 2).

Known fGn/fBm simulations.

Sine wave with gaps (check peak detection).

Validate uncertainty estimates (bootstrap CI covers true β in simulations).

Validate automated interpretation outputs.

Use pytest.

Step 10: Packaging and Distribution

setup.py and pyproject.toml.

Build: python -m build.

Install locally: pip install ..

Upload to PyPI (optional).

Semantic versioning.

Step 11: Validation and Iteration

Test with article-like data.

Confirm β matches expectations.

Validate bootstrap and surrogate-based uncertainty.

Confirm automated interpretations are intuitive for users.

Add error handling and logging.

Extend with wavelet analysis, cross-spectra, multi-pollutant comparisons.

Extensions (Future Work)

Wavelet analysis for time-frequency scaling.

Cross-spectral analysis for pollutant vs. discharge coherence.

Multi-pollutant workflows to compare β across variables.

Visualization templates reproducing Liang et al. plots.

Educational modules for hydrology training.

Summary

This plan provides an end-to-end workflow to build a Python package (waterSpec) that enables robust spectral analysis of irregularly sampled environmental time series. It integrates best practices (testing, packaging, documentation), scientific rigor (fractal scaling, β interpretation, uncertainty), and practical value (domain-specific interpretations for conservation planning).

By including an automated interpretation module, the package ensures results are easy to understand, actionable, and useful for both researchers and practitioners.

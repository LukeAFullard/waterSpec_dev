# Tutorial 5: Understanding the Results

After running an analysis, `waterSpec` provides a rich dictionary of results and a detailed text summary. This tutorial explains how to access and interpret these outputs.

### The `results` Dictionary

The `run_full_analysis()` method returns a Python dictionary that contains all the quantitative results of the analysis. This dictionary is also stored in the `analyzer.results` attribute.

Let's run an analysis and inspect the results.

```python
from waterSpec import Analysis
import pprint

# 1. Create and run the analysis
analyzer = Analysis(
    file_path='examples/sample_data.csv',
    time_col='timestamp',
    data_col='concentration'
)
results = analyzer.run_full_analysis(output_dir='docs/tutorials/interpretation_outputs')

# 2. Print the full results dictionary
print("--- Full Results Dictionary ---")
pprint.pprint(results)
```

### Key Results Explained

While the dictionary is comprehensive, here are the most important keys to understand:

- **`summary_text`**: A full, human-readable interpretation of the analysis. This is the same text that gets saved to the summary file.
- **`analysis_mode`**: Will always be `'auto'` when using `run_full_analysis`.
- **`chosen_model`**: The model selected by the automatic analysis (`'standard'` or `'segmented'`).
- **`bic_comparison`**: A dictionary showing the Bayesian Information Criterion (BIC) for both the standard and segmented fits. The model with the *lower* BIC is chosen.

#### If the chosen model is `standard`:
- **`beta`**: The main result. The spectral exponent calculated from the best-fit line.
- **`beta_ci_lower` / `beta_ci_upper`**: The lower and upper bounds of the 95% confidence interval for beta.
- **`r_squared`**: The R-squared value of the fit.

#### If the chosen model is `segmented`:
- **`beta1` / `beta2`**: The spectral exponents for the first (low-frequency) and second (high-frequency) segments of the spectrum.
- **`breakpoint`**: The frequency (in Hz) at which the spectrum "breaks".

- **`significant_peaks`**: A list of any statistically significant periodicities found in the data, along with their False Alarm Probability (FAP).

### Accessing Specific Results

You can easily access any specific value from the results dictionary.

```python
# Check which model was chosen
chosen_model = results['chosen_model']
print(f"The best-fitting model was: {chosen_model}")

# Get the primary beta value(s)
if chosen_model == 'standard':
    beta = results['beta']
    print(f"The spectral exponent is: {beta:.2f}")
else:
    beta1 = results['beta1']
    beta2 = results['beta2']
    print(f"The spectral exponents are: β1={beta1:.2f}, β2={beta2:.2f}")
```

By exploring this dictionary, you have full programmatic access to every detail of the analysis for your own custom reports and workflows.

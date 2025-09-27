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
- **`chosen_model`**: The model selected by the automatic analysis (`'standard'`, `'segmented_1bp'`, etc.).
- **`all_models`**: A list of dictionaries, with each dictionary containing the full results for each model type that was fit.
- **`uncertainty_warnings`**: A list of strings, with each string being a specific warning about the uncertainty of a model parameter. This is useful for programmatic checks.

#### If the chosen model is `standard`:
- **`beta`**: The main result. The spectral exponent calculated from the best-fit line.
- **`confidence_interval`**: A tuple containing the lower and upper bounds of the 95% confidence interval for beta.

#### If the chosen model is `segmented`:
- **`betas`**: A list of the spectral exponents for each segment of the spectrum.
- **`breakpoints`**: A list of the breakpoint frequencies (in Hz).
- **`betas_ci`**: A list of tuples, where each tuple is the 95% CI for the corresponding beta.
- **`breakpoints_ci`**: A list of tuples, where each tuple is the 95% CI for the corresponding breakpoint.

- **`significant_peaks`**: A list of any statistically significant periodicities found in the data.

### Understanding the Uncertainty Report

A key feature of `waterSpec` is its focus on quantifying and reporting uncertainty. If the confidence interval for any parameter is found to be excessively wide, a warning will be added to the `uncertainty_warnings` list and included in a special **Uncertainty Report** section in the `summary_text`.

This helps you avoid over-interpreting models that are not well-constrained by the data.

- **Beta (β) Uncertainty**: A warning is triggered if the 95% CI for any β value is wider than a set threshold (default is 0.5). This suggests that the slope of that segment is not well-defined.
- **Breakpoint Uncertainty**: A warning is triggered if the 95% CI for a breakpoint's period spans more than an order of magnitude (e.g., "10 days to 110 days"). This indicates that the location of the break is highly uncertain.

### Accessing Specific Results

You can easily access any specific value from the results dictionary for your own custom reports and workflows.

```python
# Check which model was chosen
chosen_model = results['chosen_model']
print(f"\nThe best-fitting model was: {chosen_model}")

# Get the primary beta value(s)
if 'segmented' not in chosen_model:
    beta = results['beta_value']
    ci = results['confidence_interval']
    print(f"The spectral exponent is: {beta:.2f} (95% CI: {ci[0]:.2f}–{ci[1]:.2f})")
else:
    betas = results['betas']
    print(f"The spectral exponents are: " + ", ".join([f"β{i+1}={b:.2f}" for i, b in enumerate(betas)]))

# Programmatically check for uncertainty issues
if results['uncertainty_warnings']:
    print("\nWarning: High uncertainty was detected in the following parameters:")
    for warning in results['uncertainty_warnings']:
        print(f"- {warning}")
```

By exploring this dictionary, you have full programmatic access to every detail of the analysis.

# Tutorial 4: Accessing the Core Spectral Analysis

The `run_full_analysis` method is a powerful, all-in-one tool. However, advanced users may want to access the results of the core spectral analysis for custom plotting or further calculations.

The `Analysis` object conveniently stores these results for you after you run an analysis.

### The Analysis Pipeline

The `run_full_analysis` method performs these steps internally:

1.  **Generates a Frequency Grid**: Creates an appropriate grid of frequencies for the analysis.
2.  **Calculates the Periodogram**: Computes the Lomb-Scargle periodogram to get the power at each frequency.
3.  **Fits and Interprets**: Fits the spectrum and interprets the results.

This tutorial focuses on how to access the outputs of **steps 1 and 2**.

### Step-by-Step Example

First, let's create an `Analysis` object and run the analysis, just like in the quickstart.

```python
from waterSpec import Analysis

# 1. Create the analyzer object
analyzer = Analysis(
    file_path='examples/sample_data.csv',
    time_col='timestamp',
    data_col='concentration'
)

# 2. Run the analysis
analyzer.run_full_analysis(output_dir='docs/tutorials/spectral_analysis_outputs')
```

### Accessing the Results

After the analysis has run, the `analyzer` object now contains the core spectral data as attributes.

- `analyzer.frequency`: A NumPy array of the frequencies (in Hz) used for the periodogram.
- `analyzer.power`: A NumPy array of the corresponding power values.

You can now use these arrays for any custom work you need to do.

```python
# Access the frequency and power arrays
frequency_array = analyzer.frequency
power_array = analyzer.power

print(f"The frequency array has {len(frequency_array)} points.")
print(f"First 5 frequencies (Hz): {frequency_array[:5]}")

print(f"\nThe power array has {len(power_array)} points.")
print(f"First 5 power values: {power_array[:5]}")
```

### Full Control

By accessing these attributes, you have the raw materials of the spectral analysis at your fingertips. You could, for example, pass them to a different plotting library or perform your own custom fitting routines.

In the next tutorial, we'll look at the rich dictionary of results returned by the analysis, which contains the outputs of the fitting and interpretation steps.

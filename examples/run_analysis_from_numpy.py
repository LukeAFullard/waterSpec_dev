import os
import numpy as np
from waterSpec.analysis import Analysis

# This example demonstrates how to run a spectral analysis by passing
# NumPy arrays directly to the Analysis class.

# --- 1. Create NumPy arrays for time and data ---
# In a real-world scenario, you might have these arrays from another
# scientific computing library or from your own calculations.
# Here, we'll generate some synthetic data for demonstration.
n_points = 200
# Create a time array representing days
time_array = np.arange(n_points, dtype=np.float64)
# Create a data array with a sine wave and some noise
data_array = 5 * np.sin(2 * np.pi * time_array / 30) + np.random.randn(n_points)
# Optionally, create an error array
error_array = np.full(n_points, 0.5)  # Constant error of 0.5

# --- 2. Initialize the Analysis class with the NumPy arrays ---
# Instead of `file_path` or `dataframe`, we pass the NumPy arrays directly.
# We must also provide `time_col` and `data_col` names, which will be used
# to create a DataFrame internally.
output_dir = "example_output/from_numpy_example"
analyzer = Analysis(
    time_col="time",
    data_col="concentration",
    error_col="error",
    time_array=time_array,
    data_array=data_array,
    error_array=error_array,
    param_name="Synthetic Data from NumPy",
    input_time_unit="days",  # Specify the unit of the input time array
    time_unit="days",      # Specify the desired output time unit
    verbose=True,
)

# --- 3. Run the full analysis ---
# This performs the periodogram calculation, model fitting, and peak detection.
print(f"Running analysis for '{analyzer.param_name}'...")
results = analyzer.run_full_analysis(
    output_dir=output_dir,
    peak_detection_method="fap",
    fap_threshold=0.01,
    n_bootstraps=10,  # Reduce bootstraps for a faster example run
)

# --- 4. Print a summary of the results ---
print("\n" + "=" * 50)
print("      ANALYSIS COMPLETE")
print("=" * 50)
print(f"Results saved to: {output_dir}")
print(results["summary_text"])
import os
import pandas as pd
from waterSpec.analysis import Analysis

# This example demonstrates how to run a spectral analysis by passing a
# pandas DataFrame directly to the Analysis class.

# --- 1. Load data into a pandas DataFrame ---
# In a real-world scenario, you might already have a DataFrame from a
# database query, an API call, or another data processing step.
# Here, we'll load it from a sample CSV file for demonstration purposes.
file_path = os.path.join(os.path.dirname(__file__), "sample_data.csv")
try:
    df = pd.read_csv(file_path)
except FileNotFoundError:
    print(f"Error: The data file was not found at {file_path}")
    exit()

# --- 2. Initialize the Analysis class with the DataFrame ---
# Instead of providing a `file_path`, we pass the DataFrame directly
# using the `dataframe` argument. All other parameters remain the same.
output_dir = "example_output/from_dataframe_example"
analyzer = Analysis(
    dataframe=df,
    time_col="timestamp",
    data_col="concentration",
    param_name="Nitrate-N from DataFrame",
    time_unit="days",  # Specify the desired time unit for analysis
    verbose=True,
)

# --- 3. Run the full analysis ---
# This performs the periodogram calculation, model fitting, and peak detection.
# The results, including a plot and a summary text file, are saved to the
# specified output directory.
print(f"Running analysis for '{analyzer.param_name}'...")
results = analyzer.run_full_analysis(
    output_dir=output_dir,
    # --- Optional analysis parameters ---
    # fit_method="theil-sen",  # Use Theil-Sen for robust slope fitting
    # peak_detection_method="fap",  # Use False Alarm Probability for peak detection
    # fap_threshold=0.01,  # Set a significance level for peak detection
)

# --- 4. Print a summary of the results ---
print("\n" + "=" * 50)
print("      ANALYSIS COMPLETE")
print("=" * 50)
print(f"Results saved to: {output_dir}")
print(results["summary_text"])
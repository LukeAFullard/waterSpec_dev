"""
This script provides a best-practice example of how to run a robust analysis
using the waterSpec package. It includes:
- A try...except block to gracefully handle potential errors during analysis.
- The use of the `verbose=True` flag to enable detailed logging.
- A demonstration of how to access and interpret the `uncertainty_warnings`
  that are new in this version of the package.
"""

import os
from waterSpec import Analysis

# --- 1. Define Configuration ---
# Use a sample data file that is known to have some issues
FILE_PATH = "examples/sample_data.csv"
TIME_COL = "timestamp"
DATA_COL = "concentration"
PARAM_NAME = "Sample Concentration (Robust Analysis)"
OUTPUT_DIR = "example_robust_output"

# Create the output directory if it doesn't exist
os.makedirs(OUTPUT_DIR, exist_ok=True)


def run_robust_analysis():
    """
    Runs the waterSpec analysis within a robust error-handling framework.
    """
    print(f"--- Starting Robust Analysis for: {PARAM_NAME} ---")

    try:
        # --- 2. Initialize the Analysis Object ---
        # Using verbose=True will print detailed logs of each step.
        print("\nStep 1: Initializing analysis and preprocessing data...")
        analyzer = Analysis(
            file_path=FILE_PATH,
            time_col=TIME_COL,
            data_col=DATA_COL,
            param_name=PARAM_NAME,
            verbose=True,  # Enable detailed logging
        )
        print("Initialization and preprocessing complete.")

        # --- 3. Run the Full Analysis ---
        # This includes periodogram calculation, model fitting, and peak detection.
        print("\nStep 2: Running the full spectral analysis...")
        results = analyzer.run_full_analysis(
            output_dir=OUTPUT_DIR,
            seed=42,  # Use a seed for reproducible results
        )
        print("Full analysis complete.")

        # --- 4. Review the Results ---
        print("\nStep 3: Reviewing results and uncertainty warnings...")
        # The full, formatted summary is in the 'summary_text' key
        print("\n--- Full Analysis Summary ---")
        print(results["summary_text"])

        # The new `uncertainty_warnings` key contains a list of any warnings
        # generated during the analysis. This is useful for programmatic checks.
        if results["uncertainty_warnings"]:
            print("\n--- Uncertainty Warnings Detected ---")
            for warning in results["uncertainty_warnings"]:
                print(f"- {warning}")
        else:
            print("\n--- No Uncertainty Warnings Detected ---")

        print(f"\nAnalysis successful. Outputs saved to '{OUTPUT_DIR}'.")

    except FileNotFoundError as e:
        print(f"\nERROR: The data file was not found.")
        print(f"  Details: {e}")
    except ValueError as e:
        print(f"\nERROR: A data validation error occurred.")
        print(f"  Details: {e}")
    except Exception as e:
        print(f"\nAn unexpected error occurred during the analysis.")
        print(f"  Details: {type(e).__name__}: {e}")


if __name__ == "__main__":
    run_robust_analysis()
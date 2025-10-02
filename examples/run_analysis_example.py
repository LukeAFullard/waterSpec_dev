"""
This script demonstrates how to use the waterSpec Analysis class to perform
a changepoint analysis on a time series.
"""

import os
import sys

# Add the project root to the Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from src.waterSpec import Analysis


def main():
    """
    Runs a changepoint analysis on sample data.
    """
    # Use the sample data included in the repository that is known to have
    # segmented characteristics.
    file_path = 'examples/segmented_data.csv'
    param_name = "Synthetic Segmented Signal"
    output_dir = "example_output/changepoint_analysis"
    os.makedirs(output_dir, exist_ok=True)


    print(f"--- Initializing Changepoint Analysis for: {param_name} ---")
    try:
        # Initialize the analyzer with changepoint_mode='auto' to trigger
        # an analysis that splits the data into two segments.
        analyzer = Analysis(
            file_path=file_path,
            time_col='timestamp',
            data_col='value',
            param_name=param_name,
            verbose=True,
            changepoint_mode='auto',
        )
    except (FileNotFoundError, ValueError) as e:
        print(f"\nERROR: Failed to initialize analyzer. Details: {e}")
        return

    print("\n--- Running Changepoint Analysis ---")
    try:
        results = analyzer.run_full_analysis(
            output_dir=output_dir,
            seed=42,
            n_bootstraps=10,  # Reduced for a quick example run
            changepoint_plot_style='separate'
        )
        print("\n--- Changepoint Analysis Summary ---")
        print(results['summary_text'])

    except Exception as e:
        import traceback
        print(f"Changepoint analysis failed with an unexpected error: {e}")
        traceback.print_exc()


    print(
        f"\nAnalysis complete. Check the '{output_dir}' directory for plots and summaries."
    )


if __name__ == "__main__":
    main()
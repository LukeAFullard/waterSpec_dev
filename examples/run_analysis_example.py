"""
This script demonstrates how to use the waterSpec Analysis class to perform
spectral analysis on a time series. It shows both a default analysis and an
advanced analysis that considers up to 2 breakpoints, while also demonstrating
best practices like verbose logging and checking for uncertainty warnings.
"""

import os
from waterSpec import Analysis


def main():
    """
    Runs two types of analysis on sample data to demonstrate key features.
    """
    # Use the sample data included in the repository that is known to have
    # segmented characteristics.
    file_path = 'examples/segmented_data.csv'
    param_name = "Synthetic Segmented Signal"
    output_dir = "example_outputs"

    print(f"--- Initializing Analysis for: {param_name} ---")
    try:
        # Initialize the analyzer with verbose=True to get detailed logs
        analyzer = Analysis(
            file_path=file_path,
            time_col='timestamp',
            data_col='value',
            param_name=param_name,
            verbose=True,
        )
    except (FileNotFoundError, ValueError) as e:
        print(f"\nERROR: Failed to initialize analyzer. Details: {e}")
        return

    # --- 1. Default Analysis (max_breakpoints=1) ---
    print("\n--- Running Default Analysis (comparing 0 and 1 breakpoint models) ---")
    default_output_path = os.path.join(output_dir, "default_analysis")
    try:
        results_default = analyzer.run_full_analysis(
            output_dir=default_output_path, seed=42
        )
        print("\n--- Default Analysis Summary ---")
        print(results_default['summary_text'])
        # Check for uncertainty warnings
        if results_default['uncertainty_warnings']:
            print("\n--- Uncertainty Warnings Detected ---")
            for warning in results_default['uncertainty_warnings']:
                print(f"- {warning}")
    except Exception as e:
        print(f"Default analysis failed with an unexpected error: {e}")

    # --- 2. Two-Breakpoint Analysis (max_breakpoints=2) ---
    print(
        "\n--- Running Two-Breakpoint Analysis (comparing 0, 1, and 2 "
        "breakpoint models) ---"
    )
    twobp_output_path = os.path.join(output_dir, "2bp_analysis")
    try:
        # Reuse the same analyzer object for a different analysis
        results_2bp = analyzer.run_full_analysis(
            output_dir=twobp_output_path,
            max_breakpoints=2,
            seed=42,
        )
        print("\n--- Two-Breakpoint Analysis Summary ---")
        print(results_2bp['summary_text'])
        # Check for uncertainty warnings
        if results_2bp['uncertainty_warnings']:
            print("\n--- Uncertainty Warnings Detected ---")
            for warning in results_2bp['uncertainty_warnings']:
                print(f"- {warning}")
    except Exception as e:
        print(f"Two-breakpoint analysis failed with an unexpected error: {e}")

    print(
        f"\nAnalysis complete. Check the '{default_output_path}' and "
        f"'{twobp_output_path}' directories for plots and summaries."
    )


if __name__ == "__main__":
    main()
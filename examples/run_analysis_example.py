import os
from waterSpec import Analysis

def main():
    """
    This script demonstrates how to use the waterSpec Analysis class to perform
    spectral analysis on a time series. It shows both the default behavior
    (comparing 0 and 1 breakpoint models) and the advanced usage for
    considering up to 2 breakpoints.
    """
    # Use the sample data included in the repository that is known to have
    # segmented characteristics.
    file_path = 'examples/segmented_data.csv'
    param_name = "Synthetic Segmented Signal"
    output_dir = "example_outputs"

    # Create a single analyzer object for the dataset.
    # The Analysis class handles all data loading and preprocessing.
    print(f"Initializing analysis for '{file_path}'...")
    try:
        analyzer = Analysis(
            file_path=file_path,
            time_col='timestamp',
            data_col='value',
            param_name=param_name
        )
    except FileNotFoundError:
        print(f"\nError: Data file not found at '{file_path}'.")
        print("Please ensure you are running this script from the root of the waterSpec repository.")
        return

    # --- 1. Default Analysis (max_breakpoints=1) ---
    print("\n--- Running Default Analysis (comparing 0 and 1 breakpoint models) ---")

    # The `run_full_analysis` method automatically compares models,
    # generates outputs, and returns the results.
    default_output_path = os.path.join(output_dir, "default_analysis")
    results_default = analyzer.run_full_analysis(output_dir=default_output_path)

    print("\n--- Default Analysis Summary ---")
    print(results_default['summary_text'])
    print("-" * 40)


    # --- 2. Two-Breakpoint Analysis (max_breakpoints=2) ---
    print("\n--- Running Two-Breakpoint Analysis (comparing 0, 1, and 2 breakpoint models) ---")

    # We can reuse the same analyzer object.
    # By setting max_breakpoints=2, the analysis will automatically compare
    # models with 0, 1, and 2 breakpoints and select the best one based on BIC.
    twobp_output_path = os.path.join(output_dir, "2bp_analysis")
    results_2bp = analyzer.run_full_analysis(
        output_dir=twobp_output_path,
        max_breakpoints=2
    )

    print("\n--- Two-Breakpoint Analysis Summary ---")
    print(results_2bp['summary_text'])
    print("-" * 40)

    print(f"\nAnalysis complete. Check the '{default_output_path}' and '{twobp_output_path}' directories for plots and summaries.")


if __name__ == "__main__":
    main()
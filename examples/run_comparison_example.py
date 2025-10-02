import os
import sys

# Add the project root to the Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from src.waterSpec.comparison import SiteComparison


def run_example_comparison():
    """
    An example function that demonstrates how to use the SiteComparison class
    to compare spectral analyses from two different datasets.
    """
    print("Running Site Comparison Example...")

    # Define configurations for the two sites to be compared
    # This example uses two different sample datasets provided in the examples directory
    site1_config = {
        "name": "ForestedCatchment",
        "file_path": "examples/sample_data.csv",
        "time_col": "timestamp",
        "data_col": "concentration",
        "param_name": "Concentration",
        "time_unit": "days",
    }

    site2_config = {
        "name": "UrbanCatchment",
        "file_path": "examples/periodic_data.csv",
        "time_col": "timestamp",
        "data_col": "value",
        "param_name": "Value",
        "time_unit": "days",
    }

    # Create the output directory if it doesn't exist
    output_dir = "example_output/site_comparison"
    os.makedirs(output_dir, exist_ok=True)
    print(f"Output will be saved to: {output_dir}")

    try:
        # Initialize the SiteComparison class with the two site configurations
        comparison_analyzer = SiteComparison(
            site1_config=site1_config,
            site2_config=site2_config,
            verbose=True  # Set to True for detailed logging
        )

        # Run the full comparison analysis
        # This will analyze each site and then generate comparison outputs
        results = comparison_analyzer.run_comparison(
            output_dir=output_dir,
            max_breakpoints=1,
            seed=42,  # for reproducibility
            n_bootstraps=10,  # Reduced for a quick example run
            comparison_plot_style="overlaid"  # 'separate' or 'overlaid'
        )

        print("\nSite comparison analysis complete.")
        print(f"Summary file and plots are saved in '{output_dir}'.")
        # Print the generated summary to the console
        print("\n--- Comparison Summary ---")
        print(results["summary_text"])

    except Exception as e:
        print(f"\nAn error occurred during the comparison analysis: {e}")
        # In a real application, you might want to log the full traceback
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    run_example_comparison()
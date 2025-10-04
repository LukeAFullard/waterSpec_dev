import os
import pprint

import numpy as np
import pandas as pd
from fbm import FBM

from waterSpec import Analysis

# --- Helper Functions from Tutorials ---


def generate_periodic_data():
    """Generates synthetic periodic data for tutorials."""
    print("Generating synthetic periodic data...")
    n_points = 365
    time = pd.to_datetime(np.arange(n_points), unit="D", origin="2022-01-01")
    period = 30  # days
    frequency_cpd = 1.0 / period  # cycles per day
    rng = np.random.default_rng(42)
    series = 5 * np.sin(2 * np.pi * frequency_cpd * np.arange(n_points)) + rng.normal(
        0, 0.1, n_points
    )

    file_path = "examples/periodic_data.csv"
    df = pd.DataFrame({"timestamp": time, "value": series})
    df.to_csv(file_path, index=False)
    print(f"  - Saved to '{file_path}'")
    return file_path


def generate_segmented_data():
    """Generates synthetic data with a spectral break for tutorials."""
    print("Generating synthetic data with spectral break...")
    rng = np.random.default_rng(42)
    n_points = 500
    time = np.sort(rng.uniform(0, n_points, n_points))
    hurst = 0.9
    f = FBM(n=n_points - 1, hurst=hurst, length=n_points, method="daviesharte")
    fbm_signal = f.fbm()
    noise = rng.normal(0, 15, n_points)
    combined_signal = fbm_signal + noise
    df = pd.DataFrame(
        {"timestamp": pd.to_datetime(time, unit="D"), "value": combined_signal}
    )

    file_path = "examples/segmented_data.csv"
    df.to_csv(file_path, index=False)
    print(f"  - Saved to '{file_path}'")
    return file_path


# --- Documentation Generation Functions ---


def generate_readme_outputs():
    """Runs the example from the README.md file."""
    print("\n--- Generating README Outputs ---")
    output_dir = "readme_output"
    analyzer = Analysis(
        file_path="examples/sample_data.csv",
        time_col="timestamp",
        data_col="concentration",
    )
    results = analyzer.run_full_analysis(output_dir=output_dir)
    print("README Example Analysis Results:")
    pprint.pprint(results)
    print(f"  - Outputs saved to '{output_dir}/'")


def generate_tutorial_01_outputs():
    """Generates outputs for Tutorial 01: Quickstart."""
    print("\n--- Generating Tutorial 01 Outputs (Quickstart) ---")
    output_dir = "docs/tutorials/quickstart_outputs"
    analyzer = Analysis(
        file_path="examples/sample_data.csv",
        time_col="timestamp",
        data_col="concentration",
    )
    analyzer.run_full_analysis(output_dir=output_dir)
    print(f"  - Outputs saved to '{output_dir}/'")


def generate_tutorial_06_outputs(periodic_data_path):
    """Generates outputs for Tutorial 06: Advanced Peak Finding."""
    print("\n--- Generating Tutorial 06 Outputs (Advanced Peak Finding) ---")
    output_dir = "docs/tutorials/fap_outputs"
    analyzer = Analysis(
        file_path=periodic_data_path,
        time_col="timestamp",
        data_col="value",
        detrend_method=None,
    )
    analyzer.run_full_analysis(
        output_dir=output_dir,
        peak_detection_method="fap",  # This tutorial specifically demonstrates FAP
        fap_threshold=0.01,
        grid_type="linear",
    )
    print(f"  - Outputs saved to '{output_dir}/'")


def generate_tutorial_07_outputs(periodic_data_path):
    """Generates outputs for Tutorial 07: Plotting."""
    print("\n--- Generating Tutorial 07 Outputs (Plotting) ---")
    output_dir = "docs/tutorials/plotting_outputs"
    analyzer = Analysis(
        file_path=periodic_data_path,
        time_col="timestamp",
        data_col="value",
        detrend_method=None,
    )
    analyzer.run_full_analysis(
        output_dir=output_dir,
        peak_detection_method="fap",
        fap_threshold=0.01,
        grid_type="linear",
    )
    print(f"  - Outputs saved to '{output_dir}/'")


def generate_tutorial_08_outputs(segmented_data_path):
    """Generates outputs for Tutorial 08: Segmented Fitting."""
    print("\n--- Generating Tutorial 08 Outputs (Segmented Fitting) ---")
    output_dir = "docs/tutorials/segmented_outputs"
    analyzer = Analysis(
        file_path=segmented_data_path,
        time_col="timestamp",
        data_col="value",
        param_name="Synthetic Signal",
    )
    analyzer.run_full_analysis(output_dir=output_dir)
    print(f"  - Outputs saved to '{output_dir}/'")


def main():
    """
    Main function to generate all documentation outputs.
    """
    print("Starting documentation output generation...")

    # Ensure necessary directories exist
    os.makedirs("examples", exist_ok=True)
    os.makedirs("docs/tutorials", exist_ok=True)

    # Generate synthetic data needed for tutorials
    periodic_data_path = generate_periodic_data()
    segmented_data_path = generate_segmented_data()

    # Run all generation functions
    generate_readme_outputs()
    generate_tutorial_01_outputs()
    generate_tutorial_06_outputs(periodic_data_path)
    generate_tutorial_07_outputs(periodic_data_path)
    generate_tutorial_08_outputs(segmented_data_path)

    print("\nAll documentation outputs have been generated successfully.")


if __name__ == "__main__":
    main()

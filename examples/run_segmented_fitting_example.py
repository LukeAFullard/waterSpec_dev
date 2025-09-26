import os

import numpy as np
import pandas as pd
from fbm import FBM
from waterSpec.workflow import run_analysis


def generate_data():
    """Generates synthetic data with a spectral break."""
    # Generate a time vector (irregularly sampled)
    np.random.seed(42)
    n_points = 500
    time = np.sort(np.random.uniform(0, n_points, n_points))

    # Generate a persistent signal (fBm)
    hurst = 0.9
    f = FBM(n=n_points - 1, hurst=hurst, length=n_points, method="daviesharte")
    fbm_signal = f.fbm()

    # Generate white noise
    noise = np.random.normal(0, 15, n_points)

    # Combine the signals
    combined_signal = fbm_signal + noise

    # Create a DataFrame
    df = pd.DataFrame(
        {"timestamp": pd.to_datetime(time, unit="D"), "value": combined_signal}
    )
    return df


def main():
    """Main function to run the example."""
    df = generate_data()

    # Save to a temporary file
    file_path = "segmented_data.csv"
    df.to_csv(file_path, index=False)

    # --- 1. Run a forced 'segmented' analysis to test the new plotting ---
    output_dir = "validation/plots"
    os.makedirs(output_dir, exist_ok=True)
    plot_path_segmented = os.path.join(output_dir, "test_segmented_fit_plot.png")

    print("--- Running FORCED SEGMENTED analysis ---")
    results_seg = run_analysis(
        file_path=file_path,
        time_col="timestamp",
        data_col="value",
        param_name="Synthetic Signal (Forced Segmented)",
        analysis_type="segmented",
        do_plot=True,
        output_path=plot_path_segmented,
    )

    print("\nForced Segmented Fit Results:")
    if results_seg.get("breakpoint"):
        print(f"  Breakpoint Frequency: {results_seg.get('breakpoint'):.4f}")
        print(f"  Beta 1 (Low Freq): {results_seg.get('beta1'):.2f}")
        print(f"  Beta 2 (High Freq): {results_seg.get('beta2'):.2f}")
    else:
        print("  No significant breakpoint found.")
    print(f"\nPlot saved to: {plot_path_segmented}")
    print("-" * 30)

    # --- 2. Run an 'auto' analysis to test the model selection ---
    plot_path_auto = os.path.join(output_dir, "test_auto_fit_plot.png")
    print("\n--- Running AUTO analysis ---")
    results_auto = run_analysis(
        file_path=file_path,
        time_col="timestamp",
        data_col="value",
        param_name="Synthetic Signal (Auto Selected)",
        analysis_type="auto",
        do_plot=True,
        output_path=plot_path_auto,
    )

    print("\nAuto Analysis Results:")
    bic_comp = results_auto.get("bic_comparison", {})
    print(f"  BIC (Standard): {bic_comp.get('standard'):.2f}")
    print(f"  BIC (Segmented): {bic_comp.get('segmented'):.2f}")
    print(f"  Chosen Model: {results_auto.get('chosen_model')}")
    print("\n" + results_auto.get("summary_text", ""))
    print(f"\nPlot saved to: {plot_path_auto}")
    print("-" * 30)

    # --- 3. Run with default analysis type to ensure it's 'auto' ---
    plot_path_default = os.path.join(output_dir, "test_default_fit_plot.png")
    print("\n--- Running DEFAULT analysis ---")
    results_default = run_analysis(
        file_path=file_path,
        time_col="timestamp",
        data_col="value",
        param_name="Synthetic Signal (Default)",
        # analysis_type is omitted to test the default
        do_plot=True,
        output_path=plot_path_default,
    )

    print("\nDefault Analysis Results:")
    bic_comp_def = results_default.get("bic_comparison", {})
    print(f"  BIC (Standard): {bic_comp_def.get('standard'):.2f}")
    print(f"  BIC (Segmented): {bic_comp_def.get('segmented'):.2f}")
    print(f"  Chosen Model: {results_default.get('chosen_model')}")
    print("\n" + results_default.get("summary_text", ""))
    print(f"\nPlot saved to: {plot_path_default}")

    # Clean up the temporary data file
    os.remove(file_path)


if __name__ == "__main__":
    main()

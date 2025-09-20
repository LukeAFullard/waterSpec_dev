import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from waterSpec.workflow import run_analysis
from fbm import FBM

def generate_data():
    """Generates synthetic data with a spectral break."""
    # Generate a time vector (irregularly sampled)
    np.random.seed(42)
    n_points = 500
    time = np.sort(np.random.uniform(0, n_points, n_points))

    # Generate a persistent signal (fBm)
    hurst = 0.9
    f = FBM(n=n_points-1, hurst=hurst, length=n_points, method='daviesharte')
    fbm_signal = f.fbm()

    # Generate white noise
    noise = np.random.normal(0, 15, n_points)

    # Combine the signals
    combined_signal = fbm_signal + noise

    # Create a DataFrame
    df = pd.DataFrame({
        'timestamp': pd.to_datetime(time, unit='D'),
        'value': combined_signal
    })
    return df

def main():
    """Main function to run the example."""
    df = generate_data()

    # Save to a temporary file
    file_path = 'segmented_data.csv'
    df.to_csv(file_path, index=False)

    # Define the output path for the plot
    output_dir = "docs/tutorials"
    os.makedirs(output_dir, exist_ok=True)
    plot_path = os.path.join(output_dir, "08_segmented_fit_plot.png")

    print("Running segmented analysis...")
    results = run_analysis(
        file_path=file_path,
        time_col='timestamp',
        data_col='value',
        param_name='Synthetic Signal',
        analysis_type='segmented',
        do_plot=True,
        output_path=plot_path
    )

    print("\\nSegmented Fit Results:")
    print(f"  Breakpoint Frequency: {results.get('breakpoint'):.4f}")
    print(f"  Beta 1 (Low Frequency): {results.get('beta1'):.2f}")
    print(f"  Beta 2 (High Frequency): {results.get('beta2'):.2f}")
    print(f"\\nPlot saved to: {plot_path}")

    # Clean up the temporary data file
    os.remove(file_path)

if __name__ == "__main__":
    main()

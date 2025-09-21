import numpy as np
import pandas as pd
from fbm import FBM
from waterSpec import Analysis

def generate_data():
    """Generates synthetic data with a spectral break."""
    np.random.seed(42)
    n_points = 500
    time = np.sort(np.random.uniform(0, n_points, n_points))
    hurst = 0.9
    f = FBM(n=n_points-1, hurst=hurst, length=n_points, method='daviesharte')
    fbm_signal = f.fbm()
    noise = np.random.normal(0, 15, n_points)
    combined_signal = fbm_signal + noise
    df = pd.DataFrame({
        'timestamp': pd.to_datetime(time, unit='D'),
        'value': combined_signal
    })
    return df

def run_tutorial():
    """Runs the segmented fitting tutorial."""
    print("--- Running Tutorial 08: Segmented Fitting ---")

    # 1. Generate and save the data
    df = generate_data()
    file_path = 'examples/segmented_data.csv'
    df.to_csv(file_path, index=False)
    print(f"Generated synthetic data and saved to '{file_path}'")

    # 2. Initialize the analyzer
    analyzer = Analysis(
        file_path=file_path,
        time_col='timestamp',
        data_col='value',
        param_name='Synthetic Signal'
    )

    # 3. Run the full analysis
    results = analyzer.run_full_analysis(output_dir='docs/tutorials/segmented_outputs')

    print("--- Tutorial 08 Complete ---")
    print("Outputs saved to 'docs/tutorials/segmented_outputs/'")

if __name__ == "__main__":
    run_tutorial()

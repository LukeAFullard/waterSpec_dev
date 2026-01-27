import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import sys
from waterSpec import Analysis
from waterSpec.haar_analysis import HaarAnalysis

def run_ls_analysis(df, name):
    print(f"Running Lomb-Scargle for {name}...")
    # Use a temporary file for the Analysis class
    temp_csv = f"examples/temp_{name}.csv"
    df.to_csv(temp_csv, index=False)

    try:
        analyzer = Analysis(
            file_path=temp_csv,
            time_col='timestamp',
            data_col='value',
            param_name=name,
            censor_strategy='drop',
            detrend_method='linear',
            normalize_data=True
        )

        # Run standard analysis (no segmentation for fair comparison with single-slope Haar)
        results = analyzer.run_full_analysis(
            output_dir=f"examples/output_{name}",
            fit_method='ols',
            ci_method='parametric',
            max_breakpoints=0, # Force single slope for comparison
            normalization='standard',
            peak_detection_method=None # Skip peak detection for speed
        )
        return results['beta']
    finally:
        if os.path.exists(temp_csv):
            os.remove(temp_csv)

def run_haar_analysis(df, name):
    print(f"Running Haar for {name}...")
    # Convert timestamps to seconds for HaarAnalysis
    time_sec = (df['timestamp'] - df['timestamp'].min()).dt.total_seconds().values
    data = df['value'].values

    haar = HaarAnalysis(time_sec, data)
    res = haar.run()
    return res

def main():
    csv_path = "examples/usgs_discharge_05451500.csv"
    if not os.path.exists(csv_path):
        print(f"Error: Data file {csv_path} not found.")
        sys.exit(1)

    print("Loading USGS data...")
    df = pd.read_csv(csv_path)
    df['timestamp'] = pd.to_datetime(df['timestamp'])

    # 1. Full Data Analysis
    print("\n--- Full Data Analysis (n={}) ---".format(len(df)))
    beta_ls_full = run_ls_analysis(df, "full_ls")
    res_haar_full = run_haar_analysis(df, "full_haar")
    beta_haar_full = res_haar_full['beta']

    # 2. Subsampled Analysis (50%)
    print("\n--- Subsampled Data Analysis (50%) ---")
    df_sub = df.sample(frac=0.5, random_state=42).sort_values('timestamp')
    print("Subsampled n={}".format(len(df_sub)))

    beta_ls_sub = run_ls_analysis(df_sub, "sub_ls")
    res_haar_sub = run_haar_analysis(df_sub, "sub_haar")
    beta_haar_sub = res_haar_sub['beta']

    # 3. Report
    print("\n" + "="*50)
    print("METHOD COMPARISON SUMMARY")
    print("="*50)
    print(f"{'Method':<15} | {'Full Data Beta':<15} | {'Subsampled Beta':<15} | {'Diff':<10}")
    print("-" * 62)
    print(f"{'Lomb-Scargle':<15} | {beta_ls_full:<15.4f} | {beta_ls_sub:<15.4f} | {abs(beta_ls_full - beta_ls_sub):.4f}")
    print(f"{'Haar':<15} | {beta_haar_full:<15.4f} | {beta_haar_sub:<15.4f} | {abs(beta_haar_full - beta_haar_sub):.4f}")
    print("-" * 62)

    # Compare methods on full data
    diff_methods = abs(beta_ls_full - beta_haar_full)
    print(f"\nDifference between LS and Haar (Full Data): {diff_methods:.4f}")

    # 4. Generate Plot
    print("\nGenerating comparison plot...")
    plt.figure(figsize=(10, 6))

    # Plot Full Haar
    lags_full = res_haar_full['lags'] / 86400.0 # Convert to days
    s1_full = res_haar_full['s1']
    plt.loglog(lags_full, s1_full, 'o-', label=f'Full Data ($\\beta$={beta_haar_full:.2f})', alpha=0.7)

    # Plot Subsampled Haar
    lags_sub = res_haar_sub['lags'] / 86400.0 # Convert to days
    s1_sub = res_haar_sub['s1']
    plt.loglog(lags_sub, s1_sub, 's--', label=f'50% Subsampled ($\\beta$={beta_haar_sub:.2f})', alpha=0.7)

    plt.xlabel('Lag Time (Days)')
    plt.ylabel('Haar Structure Function $S_1$')
    plt.title('Haar Analysis Robustness: USGS Discharge Data')
    plt.legend()
    plt.grid(True, which="both", ls="-", alpha=0.3)

    plot_path = "examples/usgs_haar_comparison.png"
    plt.savefig(plot_path)
    print(f"Plot saved to {plot_path}")

if __name__ == "__main__":
    main()

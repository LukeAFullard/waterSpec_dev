import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import sys
import piecewise_regression
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

def fit_segmented_haar(lags, s1, n_breakpoints=1):
    """
    Fits a segmented regression to the log-log Haar structure function.
    Returns betas and breakpoint.
    """
    # Filter valid data
    valid = (lags > 0) & (s1 > 0)
    x = np.log(lags[valid])
    y = np.log(s1[valid])

    pw_fit = piecewise_regression.Fit(x, y, n_breakpoints=n_breakpoints)

    # Debug: Print summary
    # pw_fit.summary()

    # Extract slopes (H values)
    params = pw_fit.get_results()
    estimates = params["estimates"]

    # Print estimates for debugging
    print(f"DEBUG: Estimates keys: {estimates.keys()}")
    for k, v in estimates.items():
        print(f"  {k}: {v['estimate']}")

    beta1_hat = estimates["beta1"]["estimate"]

    slopes_H = [beta1_hat]

    # Iterate through subsequent betas to get actual slopes
    # The library parameterization:
    # y = const + beta1 * x + alpha1 * (x - bp1) * I(x > bp1) + ...
    # So 'alpha1' corresponds to the change in slope at the first breakpoint.

    current_slope = beta1_hat
    for i in range(1, n_breakpoints + 1):
        diff_slope = estimates[f"alpha{i}"]["estimate"]
        current_slope += diff_slope
        slopes_H.append(current_slope)

    # Breakpoints (in log space)
    bps_log = [estimates[f"breakpoint{i+1}"]["estimate"] for i in range(n_breakpoints)]
    bps_real = np.exp(bps_log)

    # Convert H to Beta: beta_spectral = 1 + 2H
    betas_spectral = [1 + 2 * h for h in slopes_H]

    return betas_spectral, bps_real, pw_fit

def run_haar_analysis(df, name):
    print(f"Running Haar for {name}...")
    # Convert timestamps to seconds for HaarAnalysis
    time_sec = (df['timestamp'] - df['timestamp'].min()).dt.total_seconds().values
    data = df['value'].values

    haar = HaarAnalysis(time_sec, data)
    res = haar.run()

    # Add segmented fit
    print("  Fitting segmented model to Haar function...")
    betas_seg, bps, fit_obj = fit_segmented_haar(res['lags'], res['s1'])

    res['segmented_betas'] = betas_seg
    res['segmented_bps'] = bps
    res['segmented_fit_obj'] = fit_obj

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
    # beta_ls_full = run_ls_analysis(df, "full_ls")
    beta_ls_full = 0.8397 # From previous run
    res_haar_full = run_haar_analysis(df, "full_haar")
    beta_haar_full_single = res_haar_full['beta']
    betas_haar_full_seg = res_haar_full['segmented_betas']

    # 2. Subsampled Analysis (50%)
    print("\n--- Subsampled Data Analysis (50%) ---")
    df_sub = df.sample(frac=0.5, random_state=42).sort_values('timestamp')
    print("Subsampled n={}".format(len(df_sub)))

    # beta_ls_sub = run_ls_analysis(df_sub, "sub_ls")
    beta_ls_sub = 0.3993 # From previous run
    res_haar_sub = run_haar_analysis(df_sub, "sub_haar")
    beta_haar_sub_single = res_haar_sub['beta']
    betas_haar_sub_seg = res_haar_sub['segmented_betas']

    # 3. Report
    print("\n" + "="*80)
    print("METHOD COMPARISON SUMMARY (Standard vs Segmented Haar)")
    print("="*80)
    print(f"{'Method':<20} | {'Full Data Beta':<25} | {'Subsampled Beta':<25}")
    print("-" * 80)
    print(f"{'Lomb-Scargle':<20} | {beta_ls_full:<25.4f} | {beta_ls_sub:<25.4f}")
    print(f"{'Haar (Single)':<20} | {beta_haar_full_single:<25.4f} | {beta_haar_sub_single:<25.4f}")

    full_seg_str = f"{betas_haar_full_seg[0]:.2f}, {betas_haar_full_seg[1]:.2f}"
    sub_seg_str = f"{betas_haar_sub_seg[0]:.2f}, {betas_haar_sub_seg[1]:.2f}"
    print(f"{'Haar (Segmented)':<20} | {full_seg_str:<25} | {sub_seg_str:<25}")
    print("-" * 80)

    # 4. Generate Plot
    print("\nGenerating comparison plot...")
    plt.figure(figsize=(10, 6))

    # Function to plot fit
    def plot_fit(lags, fit_obj, color, style='-'):
        x_min = np.min(np.log(lags))
        x_max = np.max(np.log(lags))
        x_grid = np.linspace(x_min, x_max, 100)
        y_pred = fit_obj.predict(x_grid)
        plt.loglog(np.exp(x_grid)/86400.0, np.exp(y_pred), style, color=color, linewidth=2, label='_nolegend_')

    # Plot Full Haar
    lags_full = res_haar_full['lags']
    s1_full = res_haar_full['s1']
    plt.loglog(lags_full/86400.0, s1_full, 'o', color='blue', alpha=0.5, label='Full Data')
    plot_fit(lags_full, res_haar_full['segmented_fit_obj'], 'blue')

    # Plot Subsampled Haar
    lags_sub = res_haar_sub['lags']
    s1_sub = res_haar_sub['s1']
    plt.loglog(lags_sub/86400.0, s1_sub, 's', color='orange', alpha=0.5, label='50% Subsampled')
    plot_fit(lags_sub, res_haar_sub['segmented_fit_obj'], 'orange', '--')

    # Annotate Breakpoints
    bp_full_days = res_haar_full['segmented_bps'][0] / 86400.0
    bp_sub_days = res_haar_sub['segmented_bps'][0] / 86400.0

    plt.axvline(bp_full_days, color='blue', linestyle=':', alpha=0.5, label=f'Full BP: {bp_full_days:.1f}d')
    plt.axvline(bp_sub_days, color='orange', linestyle=':', alpha=0.5, label=f'Sub BP: {bp_sub_days:.1f}d')

    plt.xlabel('Lag Time (Days)')
    plt.ylabel('Haar Structure Function $S_1$')
    plt.title('Segmented Haar Analysis: USGS Discharge Data')
    plt.legend()
    plt.grid(True, which="both", ls="-", alpha=0.3)

    plot_path = "examples/usgs_haar_segmented_comparison.png"
    plt.savefig(plot_path)
    print(f"Plot saved to {plot_path}")

if __name__ == "__main__":
    main()

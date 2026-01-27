import matplotlib
matplotlib.use('Agg') # Force non-interactive backend
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import glob
from waterSpec import Analysis
from waterSpec.haar_analysis import HaarAnalysis
import piecewise_regression
import warnings

# Suppress warnings for clean output
warnings.filterwarnings("ignore")

def fit_segmented_haar(lags, s1, n_breakpoints=1):
    """
    Fits a segmented regression to the log-log Haar structure function.
    Returns betas and breakpoint.
    """
    valid = (lags > 0) & (s1 > 0)
    x = np.log(lags[valid])
    y = np.log(s1[valid])

    # Try segmented fit
    try:
        pw_fit = piecewise_regression.Fit(x, y, n_breakpoints=n_breakpoints)
        params = pw_fit.get_results()
        estimates = params["estimates"]

        beta1_hat = estimates["beta1"]["estimate"]
        slopes_H = [beta1_hat]
        current_slope = beta1_hat
        for i in range(1, n_breakpoints + 1):
            diff_slope = estimates[f"alpha{i}"]["estimate"]
            current_slope += diff_slope
            slopes_H.append(current_slope)

        bps_log = [estimates[f"breakpoint{i+1}"]["estimate"] for i in range(n_breakpoints)]
        bps_real = np.exp(bps_log)

        betas_spectral = [1 + 2 * h for h in slopes_H]
        return betas_spectral, bps_real, pw_fit, True
    except:
        # Fallback to linear
        coeffs = np.polyfit(x, y, 1)
        beta_spectral = 1 + 2 * coeffs[0]
        return [beta_spectral], [], np.poly1d(coeffs), False

def run_comparison(file_path, param_name, output_dir):
    print(f"Processing {param_name} ({file_path})...")

    # Load Data
    try:
        df = pd.read_csv(file_path)
    except Exception as e:
        print(f"Failed to read {file_path}: {e}")
        return

    # Handle different time columns
    time_col = 'timestamp' if 'timestamp' in df.columns else 'time'
    if time_col not in df.columns:
        if 'datetime' in df.columns:
            time_col = 'datetime'
        else:
            print(f"Skipping {file_path}: No valid time column")
            return

    df[time_col] = pd.to_datetime(df[time_col])
    time_sec = (df[time_col] - df[time_col].min()).dt.total_seconds().values
    data = pd.to_numeric(df['value'], errors='coerce').values

    # Remove NaNs
    mask = ~np.isnan(data)
    time_sec = time_sec[mask]
    data = data[mask]

    if len(data) < 10:
        print(f"Skipping {file_path}: Not enough data")
        return

    # --- Lomb-Scargle ---
    print(f"  Running Lomb-Scargle analysis...")
    analyzer = Analysis(
        file_path=file_path,
        time_col=time_col,
        data_col='value',
        param_name=param_name,
        detrend_method='linear',
        normalize_data=True
    )

    # Reduce samples_per_peak for speed if dataset is large
    samples_per_peak = 2 if len(data) > 10000 else 5

    ls_results = analyzer.run_full_analysis(
        output_dir=output_dir,
        fit_method='ols',
        ci_method='parametric',
        n_bootstraps=0,
        max_breakpoints=1,
        samples_per_peak=samples_per_peak,
        peak_detection_method=None # Skip peak detection for speed
    )

    # --- Haar Analysis ---
    print(f"  Running Haar analysis...")
    haar = HaarAnalysis(time_sec, data)
    haar_res = haar.run()

    betas_haar, bps_haar, fit_obj_haar, is_segmented_haar = fit_segmented_haar(haar_res['lags'], haar_res['s1'])

    # --- Plotting ---
    print(f"  Generating comparison plot...")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # LS Plot
    freq = analyzer.frequency
    power = analyzer.power
    ax1.loglog(freq, power, 'k.', alpha=0.3, label='Periodogram')

    # Plot LS Fit
    if ls_results['chosen_model_type'] == 'segmented':
        ax1.loglog(10**ls_results['log_freq'], 10**ls_results['fitted_log_power'], 'r-', linewidth=2, label=f"Fit (Seg)")
        betas_str = ", ".join([f"{b:.2f}" for b in ls_results['betas']])
        title_ls = f"Lomb-Scargle (Beta: {betas_str})"
    else:
        ax1.loglog(10**ls_results['log_freq'], 10**ls_results['fitted_log_power'], 'r-', linewidth=2, label=f"Fit (Std)")
        title_ls = f"Lomb-Scargle (Beta: {ls_results['beta']:.2f})"

    ax1.set_xlabel('Frequency (1/day)') # Assuming units roughly
    ax1.set_ylabel('Power Density')
    ax1.set_title(title_ls)
    ax1.legend()
    ax1.grid(True, which="both", alpha=0.3)

    # Haar Plot
    lags_days = haar_res['lags'] / 86400.0
    ax2.loglog(lags_days, haar_res['s1'], 'b.', alpha=0.5, label='Haar Structure')

    # Plot Haar Fit
    if is_segmented_haar:
        x_grid = np.linspace(np.min(np.log(haar_res['lags'])), np.max(np.log(haar_res['lags'])), 100)
        y_pred = fit_obj_haar.predict(x_grid)
        ax2.loglog(np.exp(x_grid)/86400.0, np.exp(y_pred), 'orange', linewidth=2, label='Segmented Fit')
        betas_str = ", ".join([f"{b:.2f}" for b in betas_haar])
        title_haar = f"Haar Analysis (Beta: {betas_str})"
    else:
        # Linear fit
        x_vals = np.log(haar_res['lags'])
        y_pred = fit_obj_haar(x_vals)
        ax2.loglog(np.exp(x_vals)/86400.0, np.exp(y_pred), 'orange', linewidth=2, label='Linear Fit')
        title_haar = f"Haar Analysis (Beta: {betas_haar[0]:.2f})"

    ax2.set_xlabel('Lag (Days)')
    ax2.set_ylabel('Structure Function S1')
    ax2.set_title(title_haar)
    ax2.legend()
    ax2.grid(True, which="both", alpha=0.3)

    plt.suptitle(f"Spectral Analysis Comparison: {param_name}", fontsize=16)
    plt.tight_layout()

    plot_filename = f"{param_name.replace(' ', '_')}_comparison.png"
    plt.savefig(os.path.join(output_dir, plot_filename))
    print(f"  Saved {plot_filename}")
    plt.close(fig)

if __name__ == "__main__":
    output_dir = "examples/real_data_output"
    os.makedirs(output_dir, exist_ok=True)

    datasets = [
        ("examples/usgs_discharge_05451500.csv", "Iowa River Discharge"),
        ("examples/usgs_conductance_05420500.csv", "Mississippi River Conductance"),
        ("examples/usgs_turbidity_05420500.csv", "Mississippi River Turbidity"),
        ("examples/usgs_discharge_potomac_01646500.csv", "Potomac River Discharge"),
        ("examples/usgs_conductance_potomac_01646500.csv", "Potomac River Conductance"),
        ("examples/usgs_nitrate_wapello_05465500.csv", "Iowa River Nitrate"),
        ("examples/simulated_ecoli.csv", "Simulated E. coli")
    ]

    for file_path, name in datasets:
        if os.path.exists(file_path):
            run_comparison(file_path, name, output_dir)
        else:
            print(f"Warning: {file_path} not found.")

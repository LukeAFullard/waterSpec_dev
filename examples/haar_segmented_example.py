"""
Example: Haar Analysis with Segmented Fit

This example demonstrates how to perform Haar Fluctuation Analysis on a time series
and then fit a segmented model to the resulting structure function.

This workflow is useful for:
1. Handling irregularly sampled data (Haar is robust to this).
2. Identifying different scaling regimes (multifractality) using segmented regression.
3. Handling censored data (using Analysis class for preprocessing).

Note on Interpretation:
The Haar Structure Function S1(dt) scales as dt^H.
The spectral slope beta is related to H by: beta = 1 + 2H.
fit_segmented_spectrum fits log(y) = -beta * log(x) + c.
When applying it to Haar (x=lags, y=S1), the returned 'beta' is -H.
So we recover H = -beta_fitted, and then calculate spectral beta.
"""

import matplotlib.pyplot as plt
import numpy as np
import os
import sys

# Add the src directory to the path so we can import waterSpec if running from examples/
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

from waterSpec import Analysis
from waterSpec.haar_analysis import HaarAnalysis
from waterSpec.fitter import fit_segmented_spectrum

def main():
    # Path to example data (adjust if running from different directory)
    # Using one of the provided test data files which contains censored values
    data_file = os.path.join(os.path.dirname(__file__), '../assets/data/Site_1_NO3.xlsx')

    if not os.path.exists(data_file):
        print(f"Data file not found: {data_file}")
        return

    print(f"Loading data from {data_file}...")

    # 1. Load and Preprocess Data
    # We use the Analysis class to handle loading, timestamp parsing, and
    # censored data handling (e.g. converting '<0.001' to 0.001).
    try:
        analyzer = Analysis(
            file_path=data_file,
            time_col='SampleDateTime',
            data_col='Value',
            param_name='Nitrate',
            censor_strategy='use_detection_limit',
            time_unit='days' # Convert time to days for easier interpretation
        )
    except Exception as e:
        print(f"Error loading data: {e}")
        return

    time = analyzer.time
    data = analyzer.data
    print(f"Data loaded successfully. {len(data)} valid points.")
    print(f"Time range: {time.min():.2f} to {time.max():.2f} days")

    # 2. Run Haar Analysis
    print("\nRunning Haar Analysis...")
    # Calculate Structure Function S1 vs Lags
    ha = HaarAnalysis(time, data, time_unit=analyzer.time_unit)
    res = ha.run(num_lags=50)

    lags = res['lags']
    s1 = res['s1']

    # 3. Fit Segmented Spectrum
    print("Fitting Segmented Spectrum to Haar Structure Function...")

    # We use fit_segmented_spectrum to find breakpoints in the scaling behavior.
    # Note: fit_segmented_spectrum expects (frequency, power) and fits log-log linear segments.
    # We pass (lags, s1).
    fit_res = fit_segmented_spectrum(
        lags,
        s1,
        n_breakpoints=1,     # Look for 1 breakpoint
        n_bootstraps=1000,   # Bootstrap for confidence intervals
        ci_method='bootstrap',
        bootstrap_type='block',
        seed=42
    )

    # 4. Interpret Results
    print("\n" + "="*50)
    print("RESULTS")
    print("="*50)

    n_breakpoints = fit_res.get('n_breakpoints', 0)
    print(f"Number of breakpoints found: {n_breakpoints}")

    betas_fitted = fit_res.get('betas', [fit_res.get('beta')])

    if n_breakpoints > 0:
        breakpoints = fit_res.get('breakpoints', [])
        print(f"Breakpoint at lag: {breakpoints[0]:.2f} {analyzer.time_unit}")

    print("\nScaling Regimes:")
    # Lags are sorted small to large.
    # Segment 0: Small lags (High frequency / Short term)
    # Segment 1: Large lags (Low frequency / Long term)

    descriptions = ["Short-term (High Freq)", "Long-term (Low Freq)"]
    if n_breakpoints == 0:
        descriptions = ["Global"]

    for i, b_fit in enumerate(betas_fitted):
        # Interpret H from fitted slope
        # The fitter returns 'beta' as the negative slope of log-log plot.
        # log(S1) ~ H * log(dt)  => slope = H
        # fitted 'beta' = -slope = -H
        # So H = -fitted_beta
        H = -b_fit

        # Calculate Spectral Slope beta (for P(f) ~ f^-beta)
        # beta_spec = 1 + 2H
        beta_spec = 1 + 2*H

        desc = descriptions[i] if i < len(descriptions) else f"Segment {i}"
        print(f"  {desc}:")
        print(f"    Fitted Slope (H)     = {H:.3f}")
        print(f"    Spectral Slope (beta)= {beta_spec:.3f}")

    # 5. Plotting
    output_plot = 'haar_segmented_example_plot.png'
    print(f"\nGenerating plot: {output_plot}")

    plt.figure(figsize=(10, 6))

    # Plot Haar Structure Function points
    plt.loglog(lags, s1, 'o-', label='Haar Structure Function $S_1(\\Delta t)$', alpha=0.7, markersize=5)

    # Plot Fitted Segmented Model
    if 'fitted_log_power' in fit_res:
        # fitted_log_power corresponds to log10(S1) in our case
        fitted_s1 = 10**fit_res['fitted_log_power']
        plt.loglog(lags, fitted_s1, 'r--', label='Segmented Fit', linewidth=2)

    # Mark breakpoints
    if n_breakpoints > 0:
        for bp in fit_res.get('breakpoints', []):
            plt.axvline(bp, color='k', linestyle=':', label=f'Breakpoint: {bp:.2f} d')

    plt.xlabel(f'Lag $\\Delta t$ ({analyzer.time_unit})')
    plt.ylabel('Structure Function $S_1(\\Delta t)$')
    plt.title(f'Haar Analysis with Segmented Fit: {analyzer.param_name}')
    plt.legend()
    plt.grid(True, which="both", ls="-", alpha=0.5)

    plt.savefig(output_plot)
    print("Done.")

if __name__ == "__main__":
    main()

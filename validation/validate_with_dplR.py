"""
Validation script to compare waterSpec with dplR's redfit function.
"""
import os
import tempfile
import numpy as np
import pandas as pd
import rpy2.robjects as robjects
from rpy2.robjects.packages import importr
from rpy2.robjects import pandas2ri
from rpy2.robjects.conversion import localconverter
from scipy.stats import linregress

from waterSpec import run_analysis

def generate_synthetic_series(n_points=2048, beta=0, seed=42):
    """
    Generates a synthetic time series with a known spectral exponent (beta).
    """
    rng = np.random.default_rng(seed)
    freq = np.fft.rfftfreq(n_points)
    freq[0] = 1e-9 # Avoid division by zero

    power_spectrum = freq ** (-beta)

    if beta < 0:
        taper = 1 - (freq / np.max(freq))**2
        power_spectrum *= taper

    amplitude_spectrum = np.sqrt(power_spectrum)
    random_phases = rng.uniform(0, 2 * np.pi, len(freq))
    fourier_spectrum = amplitude_spectrum * np.exp(1j * random_phases)
    series = np.fft.irfft(fourier_spectrum, n=n_points)

    time = pd.to_datetime(np.arange(n_points), unit='D', origin='2000-01-01')
    return time, series

def create_temp_csv(time, series):
    """Creates a temporary CSV file and returns its path."""
    temp_file = tempfile.NamedTemporaryFile(mode='w+', delete=False, suffix='.csv', newline='')
    df = pd.DataFrame({'time': time, 'value': series})
    df.to_csv(temp_file.name, index=False)
    return temp_file.name

def estimate_beta_from_spectrum(freq, power, band=(0.01, 0.2)):
    """
    Estimates the spectral exponent beta from a power spectrum by fitting a
    line to the log-log plot over a specified frequency band.
    """
    # Select the frequency band
    sel = (freq > band[0]) & (freq < band[1])
    if not np.any(sel):
        return np.nan

    # Filter out zero or negative power values to avoid log errors
    valid_power = power[sel] > 0
    if not np.any(valid_power):
        return np.nan

    log_f = np.log10(freq[sel][valid_power])
    log_S = np.log10(power[sel][valid_power])

    # Perform linear regression
    slope, _, _, _, _ = linregress(log_f, log_S)

    # Beta is the negative of the slope
    return -slope

def main():
    """
    Main function to run the validation analysis.
    """
    dplr = importr('dplR')
    betas_to_test = np.linspace(0.0, 2.5, 6)
    results = []
    temp_files = []

    try:
        with localconverter(robjects.default_converter + pandas2ri.converter) as cv:
            for i, known_beta in enumerate(betas_to_test):
                time, series = generate_synthetic_series(beta=known_beta, seed=i)
                file_path = create_temp_csv(time, series)
                temp_files.append(file_path)

                # waterSpec analysis
                ws_results = run_analysis(
                    file_path,
                    time_col='time',
                    data_col='value',
                    param_name=f"Beta={known_beta:.2f}",
                    detrend_method='linear',
                    fit_method='theil-sen',
                    n_bootstraps=50,
                    do_plot=False
                )
                waterspec_beta = ws_results.get('beta')

                # dplR analysis
                if known_beta > 0:
                    redfit_results = dplr.redfit(
                        robjects.FloatVector(series),
                        t=robjects.IntVector(np.arange(len(series))),
                        nsim=100
                    )
                    freq_idx = list(redfit_results.names()).index('freq')
                    gxxc_idx = list(redfit_results.names()).index('gxxc')

                    dplr_freq = np.array(redfit_results[freq_idx])
                    dplr_power = np.array(redfit_results[gxxc_idx])

                    dplr_beta = estimate_beta_from_spectrum(dplr_freq, dplr_power)
                else:
                    dplr_beta = np.nan

                results.append({
                    'known_beta': known_beta,
                    'waterspec_beta': waterspec_beta,
                    'dplr_beta': dplr_beta
                })
    finally:
        for f in temp_files:
            try:
                os.remove(f)
            except OSError:
                pass

    print("--- Validation Results: waterSpec vs dplR ---")
    print(f"| {'Known Beta':<12} | {'waterSpec Beta':<16} | {'dplR Beta':<12} |")
    print(f"|{'-'*14}|{'-'*18}|{'-'*14}|")
    for res in results:
        ws_beta_str = f"{res['waterspec_beta']:.4f}" if res['waterspec_beta'] is not None else "N/A"
        dplr_beta_str = f"{res['dplr_beta']:.4f}" if not np.isnan(res['dplr_beta']) else "N/A"
        print(f"| {res['known_beta']:.2f}{' ':<7} | {ws_beta_str:<16} | {dplr_beta_str:<12} |")

    print("\n--- Comparison Summary ---")
    print("This validation compares the spectral exponent (beta) estimated by `waterSpec`")
    print("with a beta estimated from the bias-corrected spectrum returned by `dplR::redfit`.")
    print("The `dplR` beta is calculated by fitting a linear regression to the log-log spectrum")
    print("over a fixed frequency band (0.01 to 0.2).")


if __name__ == "__main__":
    main()

"""
Comprehensive validation script to compare two peak detection methods within waterSpec:
1. The 'residual' method.
2. The newly implemented 'redfit' method.

This script is designed to be called for a single pair of beta and amplitude
values and will print a single row of a markdown table as its output.
"""

import argparse
import contextlib
import os
import shutil
import sys
import tempfile

import numpy as np
import pandas as pd

# This is a bit of a hack to make sure the local src folder is in the path
# for testing, so we can import waterSpec
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))
from waterSpec.analysis import Analysis

# --- Test Parameters ---
N_POINTS = 512
SIGNAL_PERIOD_DAYS = 50.0
REDFIT_NSIM = 250  # A reasonable number for a single run


def generate_synthetic_series_with_peak(beta, signal_amp, seed=42):
    """
    Generates a synthetic time series with colored noise and an injected sine wave.
    """
    rng = np.random.default_rng(seed)
    freq = np.fft.rfftfreq(N_POINTS, d=1)
    freq[0] = 1e-9
    power_spectrum = freq ** (-beta)
    noise_fft = np.sqrt(power_spectrum) * np.exp(
        1j * rng.uniform(0, 2 * np.pi, len(freq))
    )
    noise = np.fft.irfft(noise_fft, n=N_POINTS)
    noise = (noise - np.mean(noise)) / np.std(noise)
    time_steps = np.arange(N_POINTS)
    signal_freq = 1.0 / SIGNAL_PERIOD_DAYS
    signal = signal_amp * np.sin(2 * np.pi * signal_freq * time_steps)
    series = noise + signal
    time_index = pd.to_datetime(time_steps, unit="D", origin="2000-01-01")
    return time_index, series


def create_temp_csv(time, series, temp_dir):
    """Creates a temporary CSV file within a given directory."""
    file_path = os.path.join(temp_dir, "peak_data.csv")
    pd.DataFrame({"time": time, "value": series}).to_csv(file_path, index=False)
    return file_path


def run_single_validation(beta, signal_amp, temp_dir):
    """
    Runs a single validation case and returns a tuple of booleans indicating
    if the peak was found by each of the two methods.
    """
    signal_freq_hz = (1.0 / SIGNAL_PERIOD_DAYS) / 86400.0
    time, series = generate_synthetic_series_with_peak(
        beta, signal_amp, seed=int(beta * 10 + signal_amp * 10)
    )
    file_path = create_temp_csv(time, series, temp_dir)

    # Suppress stdout from the analysis runs
    with open(os.devnull, "w") as f, contextlib.redirect_stdout(f):
        # --- 1. waterSpec Analysis (Residual Method) ---
        try:
            ws_analyzer_resid = Analysis(
                file_path, time_col="time", data_col="value", detrend_method=None
            )
            ws_results_resid = ws_analyzer_resid.run_full_analysis(
                output_dir=os.path.join(temp_dir, "resid"),
                grid_type="linear",
                peak_detection_method="residual",
            )
            ws_resid_found = False
            if (
                "significant_peaks" in ws_results_resid
                and ws_results_resid["significant_peaks"]
            ):
                for peak in ws_results_resid["significant_peaks"]:
                    if abs(peak["frequency"] - signal_freq_hz) < (
                        signal_freq_hz * 0.15
                    ):
                        ws_resid_found = True
                        break
        except Exception:
            ws_resid_found = "ERROR"

        # --- 2. waterSpec Analysis (Redfit Method) ---
        try:
            ws_analyzer_redfit = Analysis(
                file_path, time_col="time", data_col="value", detrend_method=None
            )
            ws_results_redfit = ws_analyzer_redfit.run_full_analysis(
                output_dir=os.path.join(temp_dir, "redfit"),
                grid_type="linear",
                peak_detection_method="redfit",
                peak_detection_redfit_nsim=REDFIT_NSIM,
            )
            ws_redfit_found = False
            peaks_95ci = ws_results_redfit.get("significant_peaks_by_ci", {}).get(
                95, []
            )
            if peaks_95ci:
                for peak in peaks_95ci:
                    if abs(peak["frequency"] - signal_freq_hz) < (
                        signal_freq_hz * 0.15
                    ):
                        ws_redfit_found = True
                        break
        except Exception:
            ws_redfit_found = "ERROR"

    return ws_resid_found, ws_redfit_found


def main():
    """
    Main function to parse args and run a single validation case.
    Prints a markdown table row as output.
    """
    parser = argparse.ArgumentParser(
        description="Run a single waterSpec validation case."
    )
    parser.add_argument(
        "--beta", type=float, required=True, help="Beta value for noise generation."
    )
    parser.add_argument(
        "--amplitude",
        type=float,
        required=True,
        help="Amplitude of the injected signal.",
    )
    args = parser.parse_args()

    beta = args.beta
    amp = args.amplitude

    temp_dir = tempfile.mkdtemp()

    try:
        ws_resid, ws_redfit = run_single_validation(beta, amp, temp_dir)

        ws_resid_str = "✅ Found" if ws_resid else "❌ Not Found"
        ws_redfit_str = "✅ Found" if ws_redfit else "❌ Not Found"

        # Print the markdown row
        print(
            f"| {beta:<6.1f} | {amp:<10.2f} | {ws_resid_str:<22} | {ws_redfit_str:<20} |"
        )

    finally:
        shutil.rmtree(temp_dir)


if __name__ == "__main__":
    main()

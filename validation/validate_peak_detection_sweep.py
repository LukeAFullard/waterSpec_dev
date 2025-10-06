"""
Comprehensive validation script to compare the peak detection of waterSpec
(using the residual method) with dplR's redfit function across a range of
noise colors (beta) and signal amplitudes.
"""
import logging
import os
import shutil
import sys
import tempfile

import numpy as np
import pandas as pd
import rpy2.robjects as robjects
from rpy2.robjects import pandas2ri
from rpy2.robjects.conversion import localconverter
from rpy2.robjects.packages import importr


# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)
from src.waterSpec import Analysis

# Set up basic logging
logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


# --- Test Parameters ---
BETA_VALUES = [0.0, 0.5, 1.0, 1.5, 2.0]  # From white to red/brown noise
AMPLITUDE_VALUES = [2.0, 1.5, 1.0, 0.8, 0.5, 0.3]  # From strong to weak signals
N_POINTS = 1024
SIGNAL_FREQ_CPD = 1 / 50  # Signal with a ~50-day period


def generate_synthetic_series_with_peak(beta, signal_amp, seed=42):
    """
    Generates a synthetic time series with colored noise and an injected sine wave.
    """
    rng = np.random.default_rng(seed)
    freq = np.fft.rfftfreq(N_POINTS, d=1)
    freq[0] = 1e-9
    power_spectrum = freq ** (-beta)
    noise = np.fft.irfft(
        np.sqrt(power_spectrum) * np.exp(1j * rng.uniform(0, 2 * np.pi, len(freq))),
        n=N_POINTS,
    )
    time_steps = np.arange(N_POINTS)
    signal = signal_amp * np.sin(2 * np.pi * SIGNAL_FREQ_CPD * time_steps)
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
    Runs a single validation case for a given beta and signal amplitude.
    Returns a tuple: (waterSpec_found_peak, dplR_found_peak)
    """
    signal_freq_hz = SIGNAL_FREQ_CPD / 86400.0
    time, series = generate_synthetic_series_with_peak(beta, signal_amp)
    file_path = create_temp_csv(time, series, temp_dir)

    # --- waterSpec Analysis ---
    try:
        ws_analyzer = Analysis(
            file_path, time_col="time", data_col="value", detrend_method=None
        )
        ws_results = ws_analyzer.run_full_analysis(
            output_dir=temp_dir, grid_type="linear", peak_detection_method="residual"
        )
        ws_peak_found = False
        if "significant_peaks" in ws_results and ws_results["significant_peaks"]:
            for peak in ws_results["significant_peaks"]:
                if abs(peak["frequency"] - signal_freq_hz) < (
                    signal_freq_hz * 0.15
                ):  # 15% tolerance
                    ws_peak_found = True
                    break
    except Exception as e:
        logger.error(
            "waterSpec analysis failed for beta=%.1f, amp=%.2f: %s",
            beta,
            signal_amp,
            e,
            exc_info=True,
        )
        ws_peak_found = "ERROR"

    # --- dplR redfit Analysis ---
    try:
        dplr = importr("dplR")
        with localconverter(robjects.default_converter + pandas2ri.converter):
            redfit_results = dplr.redfit(
                robjects.FloatVector(series), nsim=500, mctest=True
            )
        names = list(redfit_results.names())
        freq = np.array(redfit_results[names.index("freq")])
        power = np.array(redfit_results[names.index("gxxc")])
        ci95 = np.array(redfit_results[names.index("ci95")])
        peak_idx = np.argmin(np.abs(freq - SIGNAL_FREQ_CPD))
        dplr_peak_found = power[peak_idx] > ci95[peak_idx]
    except Exception as e:
        logger.error(
            "dplR analysis failed for beta=%.1f, amp=%.2f: %s",
            beta,
            signal_amp,
            e,
            exc_info=True,
        )
        dplr_peak_found = "ERROR"

    return ws_peak_found, dplr_peak_found


def main():
    """Main function to run the validation sweep."""
    results_data = []
    temp_dir = tempfile.mkdtemp()
    print("Running Validation Sweep...")
    print(f"{'Beta':<6} | {'Amplitude':<10} | {'waterSpec':<12} | {'dplR':<12}")
    print("-" * 49)

    try:
        for beta in BETA_VALUES:
            for amp in AMPLITUDE_VALUES:
                ws_found, dplr_found = run_single_validation(beta, amp, temp_dir)

                if ws_found == "ERROR":
                    ws_str = "🔥 ERROR"
                else:
                    ws_str = "✅ Found" if ws_found else "❌ Not Found"

                if dplr_found == "ERROR":
                    dplr_str = "🔥 ERROR"
                else:
                    dplr_str = "✅ Found" if dplr_found else "❌ Not Found"

                results_data.append(
                    {
                        "beta": beta,
                        "amplitude": amp,
                        "waterSpec": ws_str,
                        "dplR": dplr_str,
                    }
                )
                print(f"{beta:<6.1f} | {amp:<10.2f} | {ws_str:<12} | {dplr_str:<12}")
    finally:
        shutil.rmtree(temp_dir)  # Clean up temp directory

    print("\n--- Validation Sweep Summary ---")
    df = pd.DataFrame(results_data)
    print(df.to_markdown(index=False))


if __name__ == "__main__":
    main()
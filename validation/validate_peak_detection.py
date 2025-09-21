"""
Validation script to compare the significant peak detection of waterSpec
with dplR's redfit function.
"""
import os
import tempfile
import numpy as np
import pandas as pd
import rpy2.robjects as robjects
from rpy2.robjects.packages import importr
from rpy2.robjects import pandas2ri
from rpy2.robjects.conversion import localconverter

from waterSpec import Analysis

def generate_synthetic_series_with_peak(
    n_points=1024,
    beta=1.5,
    signal_freq=0.1,
    signal_amp=0.5,
    seed=42
):
    """
    Generates a synthetic time series with red noise and an injected sine wave.
    """
    rng = np.random.default_rng(seed)

    # 1. Generate red noise
    freq = np.fft.rfftfreq(n_points, d=1) # d=1 for daily frequency
    freq[0] = 1e-9 # Avoid division by zero
    power_spectrum = freq ** (-beta)
    amplitude_spectrum = np.sqrt(power_spectrum)
    random_phases = rng.uniform(0, 2 * np.pi, len(freq))
    fourier_spectrum = amplitude_spectrum * np.exp(1j * random_phases)
    noise = np.fft.irfft(fourier_spectrum, n=n_points)

    # 2. Generate sine wave signal
    time_steps = np.arange(n_points)
    signal = signal_amp * np.sin(2 * np.pi * signal_freq * time_steps)

    # 3. Combine them
    series = noise + signal
    time_index = pd.to_datetime(time_steps, unit='D', origin='2000-01-01')

    return time_index, series

def create_temp_csv(time, series):
    """Creates a temporary CSV file and returns its path."""
    temp_dir = tempfile.mkdtemp()
    file_path = os.path.join(temp_dir, 'peak_data.csv')
    df = pd.DataFrame({'time': time, 'value': series})
    df.to_csv(file_path, index=False)
    return file_path, temp_dir

def main():
    """Main function to run the validation."""
    # --- Setup ---
    dplr = importr('dplR')
    signal_freq = 0.1  # The known frequency of our injected signal
    signal_amp = 0.8   # The amplitude of the signal
    fap_threshold = 0.01 # The significance threshold for waterSpec

    temp_dir = None

    print("--- Peak Detection Validation: waterSpec vs dplR ---")
    print(f"Injecting sine wave with freq={signal_freq}, amp={signal_amp}")
    print(f"waterSpec significance threshold (FAP): {fap_threshold}")
    print(f"dplR significance threshold: 95% confidence interval")
    print("-" * 50)

    try:
        # --- Data Generation ---
        time, series = generate_synthetic_series_with_peak(
            signal_freq=signal_freq, signal_amp=signal_amp
        )
        file_path, temp_dir = create_temp_csv(time, series)

        # --- waterSpec Analysis ---
        print("Running waterSpec Analysis...")
        ws_analyzer = Analysis(
            file_path,
            time_col='time',
            data_col='value',
            detrend_method=None
        )
        # Use a linear grid for better peak resolution
        ws_results = ws_analyzer.run_full_analysis(
            output_dir=temp_dir,
            fap_threshold=fap_threshold,
            grid_type='linear',
            n_bootstraps=10 # Use fewer bootstraps to speed up validation
        )

        ws_peak_found = False
        if 'significant_peaks' in ws_results and ws_results['significant_peaks']:
            for peak in ws_results['significant_peaks']:
                # Check if a detected peak is close to our known signal frequency
                if abs(peak['frequency'] - signal_freq) < 0.01:
                    ws_peak_found = True
                    print(f"  [SUCCESS] waterSpec found a significant peak at frequency {peak['frequency']:.4f}")
                    break
        if not ws_peak_found:
            print("  [FAILURE] waterSpec did not find the significant peak.")

        # --- dplR redfit Analysis ---
        print("\nRunning dplR redfit Analysis...")
        dplr_peak_found = False
        with localconverter(robjects.default_converter + pandas2ri.converter) as cv:
            redfit_results = dplr.redfit(
                robjects.FloatVector(series),
                nsim=500, # Use a reasonable number of simulations
                mctest=True
            )

            # Extract results from the R object
            # Find the index for each named element
            names = list(redfit_results.names())
            freq_idx = names.index('freq')
            power_idx = names.index('gxxc')
            ci95_idx = names.index('ci95')

            freq = np.array(redfit_results[freq_idx])
            power = np.array(redfit_results[power_idx])
            ci95 = np.array(redfit_results[ci95_idx])

            # Find the index of the frequency closest to our signal
            peak_idx = np.argmin(np.abs(freq - signal_freq))
            peak_power = power[peak_idx]
            peak_ci95 = ci95[peak_idx]

            if peak_power > peak_ci95:
                dplr_peak_found = True
                print(f"  [SUCCESS] dplR found a significant peak at frequency {freq[peak_idx]:.4f}")
                print(f"            (Power={peak_power:.2f} > 95% CI={peak_ci95:.2f})")
            else:
                print("  [FAILURE] dplR did not find the significant peak.")
                print(f"            (Power={peak_power:.2f} <= 95% CI={peak_ci95:.2f})")

        # --- Final Summary ---
        print("-" * 50)
        print("\n--- Validation Summary ---")
        if ws_peak_found and dplr_peak_found:
            print("✅ SUCCESS: Both waterSpec and dplR identified the injected signal as significant.")
        else:
            print("❌ FAILURE: One or both methods failed to identify the signal.")
        print(f"  - waterSpec Peak Found: {ws_peak_found}")
        print(f"  - dplR Peak Found:    {dplr_peak_found}")

    finally:
        # Clean up the temporary directory
        if temp_dir:
            import shutil
            shutil.rmtree(temp_dir)

if __name__ == "__main__":
    main()

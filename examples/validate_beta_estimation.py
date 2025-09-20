import os
import tempfile
import numpy as np
import pandas as pd
from waterSpec import run_analysis

def generate_synthetic_series(n_points=2048, beta=0, seed=42):
    """
    Generates a synthetic time series with a known spectral exponent (beta).
    """
    rng = np.random.default_rng(seed)
    freq = np.fft.rfftfreq(n_points)
    # Avoid division by zero at the DC component
    freq[0] = 1e-9

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

def main():
    """
    Main function to run the validation analysis and print results for different
    preprocessing methods on the same underlying dataset.
    """
    betas_to_test = np.linspace(0.0, 3.0, 7)
    # Use a list of tuples to define the analysis cases
    analysis_cases = [
        {'label': 'None', 'detrend_method': None, 'normalize_data': False},
        {'label': 'Linear Detrend', 'detrend_method': 'linear', 'normalize_data': False},
        {'label': 'LOESS Detrend', 'detrend_method': 'loess', 'normalize_data': False},
        {'label': 'Normalize Only', 'detrend_method': None, 'normalize_data': True},
    ]

    print("## Beta Estimation Accuracy with Different Preprocessing\n")
    print("This table shows how different preprocessing methods affect the estimation of Beta for the same synthetic time series (generated without an artificial trend).")
    print("\n| Known Beta | Preprocessing Method | Estimated Beta | Difference |")
    print("|------------|----------------------|----------------|------------|")

    temp_files = []
    try:
        for i, known_beta in enumerate(betas_to_test):
            # For each known beta, generate one series
            time, series = generate_synthetic_series(beta=known_beta, seed=i)
            file_path = create_temp_csv(time, series)
            temp_files.append(file_path)

            # Now, test this same series with all analysis cases
            for case in analysis_cases:
                try:
                    results = run_analysis(
                        file_path,
                        time_col='time',
                        data_col='value',
                        param_name=f"Beta={known_beta:.2f}",
                        detrend_method=case['detrend_method'],
                        normalize_data=case['normalize_data'],
                        n_bootstraps=50,
                        do_plot=False
                    )
                    estimated_beta = results.get('beta')
                    difference = estimated_beta - known_beta if estimated_beta is not None else 'N/A'

                    if estimated_beta is not None:
                         print(f"| {known_beta:10.2f} | {case['label']:20} | {estimated_beta:14.2f} | {difference:10.2f} |")
                    else:
                         print(f"| {known_beta:10.2f} | {case['label']:20} | {'Failed':>14} | {'N/A':>10} |")

                except Exception as e:
                    print(f"| {known_beta:10.2f} | {case['label']:20} | {'Error':>14} | {'N/A':>10} |")
    finally:
        # Clean up all temporary files at the end
        for f in temp_files:
            try:
                os.remove(f)
            except OSError:
                pass

if __name__ == "__main__":
    main()

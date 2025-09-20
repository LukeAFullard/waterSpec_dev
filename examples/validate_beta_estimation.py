import os
import tempfile
import numpy as np
import pandas as pd
from waterSpec import run_analysis

def generate_synthetic_series(n_points=2048, beta=0, seed=42):
    """
    Generates a synthetic time series with a known spectral exponent (beta).
    This function is adapted from the test suite.
    """
    rng = np.random.default_rng(seed)
    freq = np.fft.rfftfreq(n_points)
    # Avoid division by zero at the DC component
    freq[0] = 1e-9

    power_spectrum = freq ** (-beta)

    # For negative beta, power can become very large at high frequencies.
    # We can apply a taper to avoid numerical instability, though it may affect the slope.
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
    Main function to run the validation analysis and print results.
    """
    # Generate a wider range of beta values
    betas_to_test = np.linspace(-0.25, 3.0, 14)

    # --- Create plots directory ---
    plots_dir = "validation/plots"
    os.makedirs(plots_dir, exist_ok=True)

    # --- Print Results Table ---
    print("## Results Table\n")
    print("| Known Beta | Estimated Beta | 95% CI          | Difference |")
    print("|------------|----------------|-----------------|------------|")

    temp_files = []
    plot_paths = []

    try:
        for i, known_beta in enumerate(betas_to_test):
            # Use the loop index for a unique, non-negative seed
            time, series = generate_synthetic_series(beta=known_beta, seed=i)
            file_path = create_temp_csv(time, series)
            temp_files.append(file_path)

            # Define a unique path for each plot
            plot_path = os.path.join(plots_dir, f"beta_{known_beta:.2f}.png")
            plot_paths.append(plot_path)

            # Run the analysis
            results = run_analysis(
                file_path,
                time_col='time',
                data_col='value',
                param_name=f"Synthetic Series (β={known_beta:.2f})",
                detrend_method=None,
                n_bootstraps=50,  # Increase bootstraps for more stable CI
                do_plot=True,
                output_path=plot_path
            )

            estimated_beta = results.get('beta')
            ci_lower = results.get('beta_ci_lower')
            ci_upper = results.get('beta_ci_upper')

            if estimated_beta is not None and ci_lower is not None:
                difference = estimated_beta - known_beta
                ci_str = f"[{ci_lower:.2f}–{ci_upper:.2f}]"
                print(f"| {known_beta:10.2f} | {estimated_beta:14.2f} | {ci_str:15} | {difference:10.2f} |")
            else:
                print(f"| {known_beta:10.2f} | {'Analysis Failed':>14} | {'N/A':15} | {'N/A':>10} |")

        # --- Print Plots Section ---
        print("\n## Analysis Plots\n")
        for i, known_beta in enumerate(betas_to_test):
            relative_plot_path = os.path.join("plots", f"beta_{known_beta:.2f}.png")
            print(f"### Known Beta = {known_beta:.2f}")
            print(f"![Plot for Beta = {known_beta:.2f}]({relative_plot_path})\n")

    finally:
        # Clean up temporary files
        for f in temp_files:
            os.remove(f)

if __name__ == "__main__":
    main()

import os
import tempfile
import warnings

import numpy as np
import pandas as pd

from waterSpec import Analysis


def generate_synthetic_series(n_points=2048, beta=0, seed=42):
    """
    Generates a synthetic time series with a known spectral exponent (beta).
    Uses a Fourier-based method to create data with the desired properties.
    """
    rng = np.random.default_rng(seed)
    freq = np.fft.rfftfreq(n_points)
    freq[0] = 1e-9  # Avoid division by zero, which would cause -inf power

    # Create a power spectrum with the desired slope
    power_spectrum = freq ** (-beta)

    # Apply a taper to prevent aliasing for negative beta values
    if beta < 0:
        taper = 1 - (freq / np.max(freq)) ** 2
        power_spectrum *= taper

    # Combine with random phases to create a realistic signal
    amplitude_spectrum = np.sqrt(power_spectrum)
    random_phases = rng.uniform(0, 2 * np.pi, len(freq))
    fourier_spectrum = amplitude_spectrum * np.exp(1j * random_phases)

    # Inverse transform to get the time series
    series = np.fft.irfft(fourier_spectrum, n=n_points)
    time = pd.to_datetime(np.arange(n_points), unit="D", origin="2000-01-01")

    return time, series


def create_temp_csv(time, series):
    """Creates a temporary CSV file and returns its path."""
    temp_file = tempfile.NamedTemporaryFile(
        mode="w+", delete=False, suffix=".csv", newline=""
    )
    df = pd.DataFrame({"time": time, "value": series})
    df.to_csv(temp_file.name, index=False)
    return temp_file.name


def main():
    """
    Main function to run the validation analysis, comparing OLS and Theil-Sen
    fitting methods across different detrending scenarios.
    """
    betas_to_test = np.linspace(0.0, 2.5, 6)
    detrend_methods = [None, "linear", "loess"]
    fit_methods = ["ols", "theil-sen"]

    print("## Beta Estimation Accuracy: OLS vs. Theil-Sen\n")
    print(
        "This table compares the performance of the Ordinary Least Squares (OLS) "
        "and the robust Theil-Sen fitting methods across different detrending "
        "scenarios."
    )
    print(
        "\n| Known Beta | Detrend Method | Fit Method  | Estimated Beta | Difference |"
    )
    print("|------------|----------------|-------------|----------------|------------|")

    temp_dir = tempfile.mkdtemp()
    temp_files = []
    try:
        for i, known_beta in enumerate(betas_to_test):
            time, series = generate_synthetic_series(beta=known_beta, seed=i)
            file_path = create_temp_csv(time, series)
            temp_files.append(file_path)

            for detrend_method in detrend_methods:
                # Use the modern Analysis class
                analyzer = Analysis(
                    file_path,
                    time_col="time",
                    data_col="value",
                    detrend_method=detrend_method,
                )

                for fit_method in fit_methods:
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")  # Suppress warnings for this run
                        # Run the analysis with a simplified workflow
                        # We only need the standard model fit, so we set max_breakpoints=0
                        results = analyzer.run_full_analysis(
                            output_dir=temp_dir,
                            fit_method=fit_method,
                            max_breakpoints=0,
                            n_bootstraps=0,  # No need for CIs in this validation
                        )

                    estimated_beta = results.get("beta")
                    difference = (
                        estimated_beta - known_beta
                        if estimated_beta is not None and np.isfinite(estimated_beta)
                        else "N/A"
                    )

                    detrend_str = str(detrend_method) if detrend_method else "None"
                    beta_str = f"{estimated_beta:.2f}" if np.isfinite(estimated_beta) else "Failed"
                    diff_str = f"{difference:.2f}" if isinstance(difference, float) else "N/A"

                    row = (
                        f"| {known_beta:10.2f} | {detrend_str:14} | "
                        f"{fit_method:11} | {beta_str:>14} | {diff_str:>10} |"
                    )
                    print(row)
    finally:
        # Clean up temporary files and directory
        for f in temp_files:
            try:
                os.remove(f)
            except OSError:
                pass
        try:
            os.rmdir(temp_dir)
        except OSError:
            pass


if __name__ == "__main__":
    main()
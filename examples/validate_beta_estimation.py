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
    freq[0] = 1e-9  # Avoid division by zero

    power_spectrum = freq ** (-beta)

    if beta < 0:
        taper = 1 - (freq / np.max(freq)) ** 2
        power_spectrum *= taper

    amplitude_spectrum = np.sqrt(power_spectrum)
    random_phases = rng.uniform(0, 2 * np.pi, len(freq))
    fourier_spectrum = amplitude_spectrum * np.exp(1j * random_phases)
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
    betas_to_test = np.linspace(0.0, 2.5, 6)  # Reduced for speed
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

    temp_files = []
    try:
        for i, known_beta in enumerate(betas_to_test):
            time, series = generate_synthetic_series(beta=known_beta, seed=i)
            file_path = create_temp_csv(time, series)
            temp_files.append(file_path)

            for detrend_method in detrend_methods:
                for fit_method in fit_methods:
                    try:
                        results = run_analysis(
                            file_path,
                            time_col="time",
                            data_col="value",
                            param_name=f"Beta={known_beta:.2f}",
                            detrend_method=detrend_method,
                            fit_method=fit_method,
                            n_bootstraps=50,
                            do_plot=False,
                        )
                        estimated_beta = results.get("beta")
                        difference = (
                            estimated_beta - known_beta
                            if estimated_beta is not None
                            else "N/A"
                        )

                        detrend_str = (
                            str(detrend_method)
                            if detrend_method is not None
                            else "None"
                        )

                        if estimated_beta is not None:
                            row = (
                                f"| {known_beta:10.2f} | {detrend_str:14} | "
                                f"{fit_method:11} | {estimated_beta:14.2f} | "
                                f"{difference:10.2f} |"
                            )
                            print(row)
                        else:
                            row = (
                                f"| {known_beta:10.2f} | {detrend_str:14} | "
                                f"{fit_method:11} | {'Failed':>14} | {'N/A':>10} |"
                            )
                            print(row)

                    except Exception:
                        detrend_str = (
                            str(detrend_method)
                            if detrend_method is not None
                            else "None"
                        )
                        row = (
                            f"| {known_beta:10.2f} | {detrend_str:14} | "
                            f"{fit_method:11} | {'Error':>14} | {'N/A':>10} |"
                        )
                        print(row)
    finally:
        for f in temp_files:
            try:
                os.remove(f)
            except OSError:
                pass


if __name__ == "__main__":
    main()

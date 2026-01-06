import numpy as np
import pandas as pd
from waterSpec import Analysis
import matplotlib.pyplot as plt
import os

def generate_colored_noise(beta, n_points, seed=None):
    """
    Generates colored noise with a power-law spectrum P(f) ~ f^(-beta).
    """
    if seed is not None:
        np.random.seed(seed)

    # Generate white noise
    white_noise = np.random.normal(0, 1, n_points)

    # FFT
    fft_vals = np.fft.rfft(white_noise)
    frequencies = np.fft.rfftfreq(n_points)

    # Scale amplitudes to get 1/f^(beta/2) amplitude spectrum => 1/f^beta power spectrum
    # Avoid division by zero at f=0
    scaling = np.ones_like(frequencies)
    with np.errstate(divide='ignore'):
        scaling[1:] = frequencies[1:] ** (-beta / 2.0)

    # Zero out the DC component (mean)
    scaling[0] = 0

    fft_shaped = fft_vals * scaling

    # Inverse FFT
    colored_noise = np.fft.irfft(fft_shaped, n=n_points)

    # Normalize
    colored_noise = (colored_noise - np.mean(colored_noise)) / np.std(colored_noise)

    return colored_noise

def create_irregular_sampling(time, data, fraction=0.7, seed=None):
    """
    Randomly subsamples the time series to create irregular sampling.
    """
    if seed is not None:
        np.random.seed(seed)

    n = len(time)
    n_keep = int(n * fraction)
    indices = np.sort(np.random.choice(n, n_keep, replace=False))

    return time[indices], data[indices]

def verify_synthetic_data():
    output_dir = "validation_output"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    scenarios = [
        {"name": "White Noise", "beta": 0.0},
        {"name": "Pink Noise", "beta": 1.0},
        {"name": "Red Noise", "beta": 2.0},
    ]

    # Reduced parameters for speed
    n_points = 500
    seed = 42
    n_bootstraps = 100

    print(f"{'Scenario':<20} | {'Sampling':<15} | {'True Beta':<10} | {'Est. Beta':<10} | {'95% CI':<20} | {'Status':<10}", flush=True)
    print("-" * 100, flush=True)

    for scenario in scenarios:
        true_beta = scenario["beta"]
        name = scenario["name"]

        # 1. Evenly Spaced
        time_even = np.linspace(0, 100, n_points)
        data_even = generate_colored_noise(true_beta, n_points, seed=seed)

        analyzer_even = Analysis(
            time_array=time_even,
            data_array=data_even,
            time_col="time",
            data_col="value",
            input_time_unit="seconds",
            param_name=f"{name} (Even)",
            min_valid_data_points=50,
            detrend_method="linear"
        )

        try:
            res_even = analyzer_even.run_full_analysis(
                output_dir=os.path.join(output_dir, f"{name}_even"),
                fit_method="theil-sen",
                ci_method="bootstrap",
                n_bootstraps=n_bootstraps,
                normalization="standard",
            )
            est_beta = res_even['beta']
            ci_lower = res_even['beta_ci_lower']
            ci_upper = res_even['beta_ci_upper']

            success = ci_lower <= true_beta <= ci_upper
            if not success and abs(est_beta - true_beta) < 0.25:
                 status = "CLOSE"
            else:
                 status = "PASS" if success else "FAIL"

            print(f"{name:<20} | {'Even':<15} | {true_beta:<10.2f} | {est_beta:<10.2f} | [{ci_lower:.2f}, {ci_upper:.2f}]  | {status:<10}", flush=True)

        except Exception as e:
            print(f"{name:<20} | {'Even':<15} | {true_beta:<10.2f} | {'ERROR':<10} | {str(e)}", flush=True)

        # 2. Unevenly Spaced
        time_uneven, data_uneven = create_irregular_sampling(time_even, data_even, fraction=0.6, seed=seed)

        analyzer_uneven = Analysis(
            time_array=time_uneven,
            data_array=data_uneven,
            time_col="time",
            data_col="value",
            input_time_unit="seconds",
            param_name=f"{name} (Uneven)",
            min_valid_data_points=50,
            detrend_method="linear"
        )

        try:
            res_uneven = analyzer_uneven.run_full_analysis(
                output_dir=os.path.join(output_dir, f"{name}_uneven"),
                fit_method="theil-sen",
                ci_method="bootstrap",
                n_bootstraps=n_bootstraps,
                normalization="standard",
            )
            est_beta = res_uneven['beta']
            ci_lower = res_uneven['beta_ci_lower']
            ci_upper = res_uneven['beta_ci_upper']

            success = ci_lower <= true_beta <= ci_upper
            if not success and abs(est_beta - true_beta) < 0.25:
                 status = "CLOSE"
            else:
                 status = "PASS" if success else "FAIL"

            print(f"{name:<20} | {'Uneven':<15} | {true_beta:<10.2f} | {est_beta:<10.2f} | [{ci_lower:.2f}, {ci_upper:.2f}]  | {status:<10}", flush=True)

        except Exception as e:
            print(f"{name:<20} | {'Uneven':<15} | {true_beta:<10.2f} | {'ERROR':<10} | {str(e)}", flush=True)

if __name__ == "__main__":
    verify_synthetic_data()

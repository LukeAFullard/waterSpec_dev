import numpy as np
import pandas as pd
import sys
import os

# Ensure we can import waterSpec
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

from waterSpec.haar_analysis import HaarAnalysis

def generate_colored_noise(beta, n_points, seed=None):
    """Generates colored noise with a power-law spectrum P(f) ~ f^(-beta)."""
    if seed is not None:
        np.random.seed(seed)
    white_noise = np.random.normal(0, 1, n_points)
    fft_vals = np.fft.rfft(white_noise)
    frequencies = np.fft.rfftfreq(n_points)
    scaling = np.ones_like(frequencies)
    with np.errstate(divide='ignore'):
        scaling[1:] = frequencies[1:] ** (-beta / 2.0)
    scaling[0] = 0
    fft_shaped = fft_vals * scaling
    colored_noise = np.fft.irfft(fft_shaped, n=n_points)
    return (colored_noise - np.mean(colored_noise)) / np.std(colored_noise)

def create_irregular_sampling(time, data, fraction=0.7, seed=None):
    """Randomly subsamples the time series to create irregular sampling."""
    if seed is not None:
        np.random.seed(seed)
    n = len(time)
    n_keep = int(n * fraction)
    indices = np.sort(np.random.choice(n, n_keep, replace=False))
    return time[indices], data[indices]

def validate_haar():
    print("# Haar Robust Beta Estimation Validation")
    print("\nComparing True Beta with estimated Haar Beta using robust fitting (Theil-Sen).\n")

    true_betas = [0.0, 0.5, 1.0, 1.5, 2.0, 2.5]
    n_points = 2000
    fraction = 0.6
    seed = 42

    results = []

    print(f"{'True Beta':<10} | {'Sampling':<10} | {'Est. Beta':<10} | {'Diff':<10} | {'Status':<10}")
    print("-" * 60)

    for tb in true_betas:
        # Generate full even series
        time_even = np.arange(n_points)
        data_even = generate_colored_noise(tb, n_points, seed=seed)

        # 1. Even sampling
        ha_even = HaarAnalysis(time_even, data_even)
        res_even = ha_even.run(num_lags=30)
        eb_even = res_even['beta']
        diff_even = eb_even - tb
        status_even = "PASS" if abs(diff_even) < 0.25 else "FAIL"

        print(f"{tb:<10.1f} | {'Even':<10} | {eb_even:<10.2f} | {diff_even:<+10.2f} | {status_even:<10}")
        results.append({'true_beta': tb, 'sampling': 'Even', 'est_beta': eb_even, 'diff': diff_even, 'status': status_even})

        # 2. Irregular sampling
        time_uneven, data_uneven = create_irregular_sampling(time_even, data_even, fraction=fraction, seed=seed)
        ha_uneven = HaarAnalysis(time_uneven, data_uneven)
        res_uneven = ha_uneven.run(num_lags=30)
        eb_uneven = res_uneven['beta']
        diff_uneven = eb_uneven - tb
        status_uneven = "PASS" if abs(diff_uneven) < 0.25 else "FAIL"

        print(f"{tb:<10.1f} | {'Uneven':<10} | {eb_uneven:<10.2f} | {diff_uneven:<+10.2f} | {status_uneven:<10}")
        results.append({'true_beta': tb, 'sampling': 'Uneven', 'est_beta': eb_uneven, 'diff': diff_uneven, 'status': status_uneven})

    # Summary
    df = pd.DataFrame(results)
    pass_count = (df['status'] == 'PASS').sum()
    total_count = len(df)
    print(f"\nOverall Success Rate: {pass_count}/{total_count} ({pass_count/total_count:.0%})")

if __name__ == "__main__":
    validate_haar()

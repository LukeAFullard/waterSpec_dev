import numpy as np
import matplotlib.pyplot as plt
import os
from waterSpec.haar_analysis import calculate_haar_fluctuations, fit_haar_slope

def validate_micro_correctness():
    """
    Validates the exact calculation of the percentile statistic on a small, known dataset.
    """
    print("\n--- Micro Validation (Small Dataset) ---")

    # known dataset: 20 points (0 to 19)
    time = np.arange(20)
    data = np.arange(1, 21) # 1 to 20

    # We will test Lag = 10.
    # The code calculates ONE fluctuation for this lag if overlap=False (or True, since only one fits).
    # Window: t=0 to t=10.
    # Split point: t=5.

    # Left Half: t in [0, 5). Indices: 0, 1, 2, 3, 4.
    # Data: 1, 2, 3, 4, 5.
    data_left = data[0:5]

    # Right Half: t in [5, 10). Indices: 5, 6, 7, 8, 9.
    # Data: 6, 7, 8, 9, 10.
    data_right = data[5:10]

    # 90th Percentile Calculation (Hazen method)
    val_left = np.percentile(data_left, 90, method='hazen')
    val_right = np.percentile(data_right, 90, method='hazen')

    expected_diff = np.abs(val_right - val_left)

    print(f"Left Data: {data_left}")
    print(f"Right Data: {data_right}")
    print(f"Manual 90th Left: {val_left}")
    print(f"Manual 90th Right: {val_right}")
    print(f"Expected Fluctuation: {expected_diff}")

    # Run Code
    lags, s1, counts, _ = calculate_haar_fluctuations(
        time, data,
        lag_times=np.array([10.0]),
        min_samples_per_window=5, # Need 5 points per half
        overlap=False,
        statistic="percentile",
        percentile=90,
        percentile_method="hazen"
    )

    if len(s1) == 0:
        print("❌ Micro validation FAILED: No fluctuations calculated.")
        return

    print(f"Code Calculated Fluctuation: {s1[0]}")

    if np.isclose(s1[0], expected_diff):
        print("✅ Micro validation PASSED.")
    else:
        print(f"❌ Micro validation FAILED. Expected {expected_diff}, got {s1[0]}")

def validate_macro_scaling():
    """
    Validates that for Gaussian White Noise, the 90th percentile scales similarly to the mean.
    """
    print("\n--- Macro Validation (Gaussian White Noise) ---")

    np.random.seed(42)
    n = 10000
    time = np.arange(n)
    data = np.random.randn(n)

    # Run Mean Analysis
    lags_m, s1_m, _, _ = calculate_haar_fluctuations(time, data, num_lags=20)
    res_m = fit_haar_slope(lags_m, s1_m)
    beta_mean = res_m['beta']

    # Run 90th Percentile Analysis
    lags_p, s1_p, _, _ = calculate_haar_fluctuations(
        time, data, num_lags=20,
        statistic="percentile", percentile=90
    )
    res_p = fit_haar_slope(lags_p, s1_p)
    beta_90 = res_p['beta']

    print(f"Beta (Mean): {beta_mean:.3f} (Expected ~0 for White Noise)")
    print(f"Beta (90th): {beta_90:.3f}")

    # For Gaussian processes, self-similarity implies all moments scale with same H.
    # So Beta should be roughly equal.
    diff = abs(beta_mean - beta_90)
    print(f"Difference: {diff:.3f}")

    if diff < 0.2: # Allow some noise
        print("✅ Macro validation PASSED (Scaling is consistent).")
    else:
        print("❌ Macro validation FAILED (Scaling diverges significantly).")

if __name__ == "__main__":
    validate_micro_correctness()
    validate_macro_scaling()

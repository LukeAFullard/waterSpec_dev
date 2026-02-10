
import numpy as np
import sys
from waterSpec.haar_analysis import calculate_haar_fluctuations, fit_haar_slope

def run_simulation(distribution='gaussian', N=500, n_simulations=50):
    """
    Simulates white noise (Beta=0) with different distributions and checks if
    the estimated Beta from Haar analysis (with std_corrected aggregation) is biased.

    The 'std_corrected' method assumes Gaussian fluctuations. However, due to the
    Central Limit Theorem (CLT), the fluctuations (which are differences of window means)
    converge to Gaussian as window size increases, even if the raw data is non-Gaussian.

    This validation checks if this assumption holds sufficiently well for power-law estimation.
    """
    betas_mean = []
    betas_std = []

    true_beta = 0.0 # White noise

    print(f"\n--- Simulation: {distribution}, N={N}, {n_simulations} runs ---")

    for _ in range(n_simulations):
        if distribution == 'gaussian':
            # Mean 0, Std 1
            data = np.random.standard_normal(N)
        elif distribution == 'laplace':
            # Mean 0, Scale 1/sqrt(2) for Std 1. Heavier tails.
            data = np.random.laplace(loc=0.0, scale=1.0/np.sqrt(2), size=N)
        elif distribution == 'uniform':
            # Mean 0, Width sqrt(12) for Std 1. Lighter tails.
            width = np.sqrt(12)
            data = np.random.uniform(-width/2, width/2, size=N)

        time = np.arange(N)

        # Standard (Mean) Aggregation: Distribution-agnostic but biased for small samples
        lags_m, s1_m, _, _ = calculate_haar_fluctuations(
            time, data, statistic="mean", aggregation="mean", overlap=True
        )
        res_m = fit_haar_slope(lags_m, s1_m)
        if not np.isnan(res_m['beta']):
            betas_mean.append(res_m['beta'])

        # Corrected (Std) Aggregation: Assumes Gaussian fluctuations
        lags_s, s1_s, _, _ = calculate_haar_fluctuations(
            time, data, statistic="mean", aggregation="std_corrected", overlap=True
        )
        res_s = fit_haar_slope(lags_s, s1_s)
        if not np.isnan(res_s['beta']):
            betas_std.append(res_s['beta'])

    mean_beta_m = np.nanmean(betas_mean)
    std_beta_m = np.nanstd(betas_mean)
    mean_beta_s = np.nanmean(betas_std)
    std_beta_s = np.nanstd(betas_std)

    print(f"  True Beta: {true_beta:.2f}")
    print(f"  Aggregation='mean':          Beta = {mean_beta_m:.3f} +/- {std_beta_m:.3f}")
    print(f"  Aggregation='std_corrected': Beta = {mean_beta_s:.3f} +/- {std_beta_s:.3f}")

    # Calculate bias relative to Mean method (which is theoretically unbiased for slope? Or similarly biased?)
    # Actually, both should be close to 0.

    bias_s = abs(mean_beta_s - true_beta)

    if bias_s > 0.1:
        print(f"  [FAIL] Significant bias observed for {distribution} (Bias: {bias_s:.3f} > 0.1)")
        return False
    else:
        print(f"  [PASS] Bias within acceptable limits (Bias: {bias_s:.3f} <= 0.1)")
        return True

if __name__ == "__main__":
    np.random.seed(42)

    results = []
    # Gaussian (Reference)
    results.append(run_simulation('gaussian', N=500, n_simulations=50))
    # Laplace (Heavier tails - tests if CLT saves us)
    results.append(run_simulation('laplace', N=500, n_simulations=50))
    # Uniform (Lighter tails)
    results.append(run_simulation('uniform', N=500, n_simulations=50))

    if all(results):
        print("\nAll robustness checks passed.")
        sys.exit(0)
    else:
        print("\nSome robustness checks failed.")
        sys.exit(1)

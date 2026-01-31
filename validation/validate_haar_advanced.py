import numpy as np
import pandas as pd
import sys
import os

# Ensure we can import waterSpec
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

from waterSpec.haar_analysis import HaarAnalysis

def generate_regime_shift_data(n_points=2000, crossover_scale=50, noise_amp=1.0, trend_amp=0.05, seed=42):
    """
    Generates data with a regime shift from White Noise (beta=0) to Random Walk (beta=2).

    Small scales: Dominated by White Noise (rapid fluctuations).
    Large scales: Dominated by Random Walk trend.

    Crossover happens roughly when noise_amp * sqrt(tau) approx trend_amp * tau?
    Haar fluctuation for noise ~ sigma_noise / sqrt(tau).
    Haar fluctuation for walk ~ sigma_walk * sqrt(tau).
    Crossover: sigma_n / sqrt(tau) = sigma_w * sqrt(tau) -> tau = sigma_n / sigma_w.

    Here we construct explicitly:
    Signal = WhiteNoise + IntegratedWhiteNoise
    """
    np.random.seed(seed)

    # Component 1: White Noise (beta=0, m=-0.5)
    noise = np.random.normal(0, noise_amp, n_points)

    # Component 2: Random Walk (beta=2, m=0.5)
    # Scaled so that at crossover_scale, the fluctuations are comparable?
    # Actually simpler: just add them. The scaling laws will dominate at respective scales.
    walk = np.cumsum(np.random.normal(0, trend_amp, n_points))

    data = noise + walk
    time = np.arange(n_points)

    return time, data

def validate_overlap_benefit():
    print("\n### 1. Validation of Overlapping Windows Benefit ###")

    # Short record where overlap matters most
    n_points = 200
    true_beta = 2.0 # (Brown Noise)
    # Generate approx pink noise by integrating white noise 0.5? Or just use library if avail.
    # Let's use simple Brownian (beta=2) for clarity on short record.
    np.random.seed(101)
    time = np.arange(n_points)
    data = np.cumsum(np.random.randn(n_points)) # Beta=2

    ha = HaarAnalysis(time, data)

    # Run Non-Overlapping
    res_no = ha.run(overlap=False, num_lags=15, n_bootstraps=200)
    beta_no = res_no['beta']
    ci_width_no = res_no['beta_ci_upper'] - res_no['beta_ci_lower']

    # Run Overlapping
    res_ov = ha.run(overlap=True, overlap_step_fraction=0.1, num_lags=15, n_bootstraps=200)
    beta_ov = res_ov['beta']
    ci_width_ov = res_ov['beta_ci_upper'] - res_ov['beta_ci_lower']

    print(f"{'Method':<15} | {'Beta Est':<10} | {'CI Width':<10} | {'Effective N (Max Scale)':<25}")
    print("-" * 70)
    print(f"{'Non-Overlap':<15} | {beta_no:<10.2f} | {ci_width_no:<10.2f} | {res_no['counts'][-1]:<25}")
    print(f"{'Overlap':<15} | {beta_ov:<10.2f} | {ci_width_ov:<10.2f} | {res_ov['n_effective'][-1]:<25.1f}")

    # Success Criteria: Overlap should have tighter CI (smaller width) or at least more effective samples
    if ci_width_ov < ci_width_no * 1.1: # Allow some margin, but usually it's smaller
        print("\nPASS: Overlapping windows provided comparable or better precision.")
    else:
        print(f"\nWARN: Overlap CI ({ci_width_ov:.2f}) wider than Non-Overlap ({ci_width_no:.2f}).")

def validate_regime_shift():
    print("\n### 2. Validation of Segmented Haar Fit (Regime Shift) ###")

    # Generate data
    # Noise amp 1.0. Walk amp 0.1.
    # Crossover scale approx (1.0/0.1)^2 ? No.
    # Fluc noise: 1/sqrt(tau). Fluc walk: 0.1 * sqrt(tau).
    # 1/sqrt(tau) = 0.1 * sqrt(tau) => 1/0.1 = tau => tau = 10.
    # So breakpoint around 10.

    # Increase noise amplitude to 5.0 to dominate small scales
    # Theory: tau_crossover s.t. 5*sqrt(2/tau) = 0.2*sqrt(tau)
    # => 25*2/tau = 0.04*tau => 1250 = tau^2 => tau = 35.3
    time, data = generate_regime_shift_data(n_points=5000, noise_amp=5.0, trend_amp=0.2)

    ha = HaarAnalysis(time, data, time_unit="hours")

    # Analyze with 1 breakpoint allowed
    res = ha.run(min_lag=2, max_lag=500, num_lags=30, overlap=True, max_breakpoints=1, n_bootstraps=100)

    seg = res['segmented_results']

    if seg is None:
        print("FAIL: No segmented model selected (BIC chose standard).")
        return

    bp = seg['breakpoints'][0]
    beta1 = seg['betas'][0]
    beta2 = seg['betas'][1]

    expected_bp = 35.3
    print(f"detected Breakpoint: {bp:.1f} (Expected ~{expected_bp:.1f})")
    print(f"Segment 1 Beta (Short): {beta1:.2f} (Expected ~0.0 White Noise)")
    print(f"Segment 2 Beta (Long):  {beta2:.2f} (Expected ~2.0 Random Walk)")

    # Tolerances
    # Allow factor of 2 for breakpoint
    bp_ok = (expected_bp / 2) <= bp <= (expected_bp * 2)
    beta1_ok = -0.5 <= beta1 <= 0.5
    beta2_ok = 1.5 <= beta2 <= 2.5

    if bp_ok and beta1_ok and beta2_ok:
        print("\nPASS: Successfully detected regime shift and scaling exponents.")
    else:
        print("\nFAIL: Detected parameters outside expected ranges.")

if __name__ == "__main__":
    validate_overlap_benefit()
    validate_regime_shift()

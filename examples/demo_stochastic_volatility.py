import numpy as np
import matplotlib.pyplot as plt
from waterSpec import Analysis
from waterSpec.utils_sim import simulate_tk95
from waterSpec.utils_sim.models import power_law

def run_stochastic_volatility_demo():
    print("Generating Stochastic Volatility Process...", flush=True)
    np.random.seed(42)
    n = 10000  # Reduced from 50000 for speed

    # 1. Generate latent volatility process (h_t) with persistence (Beta ~ 1.5)
    # simulate_tk95 requires a PSD function. We use power_law: P(f) = A * f^(-beta)
    volatility_beta = 1.5
    # simulate_tk95 returns (time, flux)
    _, h_t = simulate_tk95(
        psd_func=power_law,
        params=(volatility_beta, 1.0), # beta=1.5, amplitude=1.0
        N=n,
        dt=1.0,
        seed=42
    )

    # Normalize h_t to reasonable range for exponentiation (e.g., sigma varies by factor of 10)
    h_t = (h_t - np.mean(h_t)) / np.std(h_t) * 1.0
    sigma_t = np.exp(h_t)

    # 2. Generate White Noise Driver (epsilon_t)
    epsilon_t = np.random.randn(n)

    # 3. Combine: y_t = epsilon_t * sigma_t
    data = epsilon_t * sigma_t
    time = np.arange(n)

    print(f"Data generated. N={n}.", flush=True)

    # Initialize Analyzer
    analyzer = Analysis(
        time_array=time,
        data_array=data,
        data_col="Value",
        time_col="Time",
        input_time_unit="seconds"
    )

    # --- Analysis 1: Mean Behavior ---
    print("\nRunning Haar Analysis (Mean)...", flush=True)
    # Using run_full_analysis but setting max_freq to something small to speed up LS or just rely on Haar
    # Actually, let's just use the Haar part directly via internal method or just bear the cost for 10k points.
    # 10k points LS is fast enough.

    res_mean = analyzer.run_full_analysis(
        output_dir="output_demo_mean",
        run_haar=True,
        haar_statistic="mean",
        n_bootstraps=100,
        samples_per_peak=1 # Speed up LS
    )
    beta_mean = res_mean['haar_results']['beta']
    print(f"-> Beta (Mean): {beta_mean:.2f} (Expected ~0.0 for white noise structure)", flush=True)

    # --- Analysis 2: 90th Percentile Behavior ---
    print("\nRunning Haar Analysis (90th Percentile)...", flush=True)
    res_90 = analyzer.run_full_analysis(
        output_dir="output_demo_90",
        run_haar=True,
        haar_statistic="percentile",
        haar_percentile=90,
        haar_percentile_method="hazen",
        n_bootstraps=100,
        samples_per_peak=1
    )
    beta_90 = res_90['haar_results']['beta']
    print(f"-> Beta (90th): {beta_90:.2f} (Expected >0.0, tracking volatility)", flush=True)

    # Check results
    if beta_mean < 0.3 and beta_90 > 0.8:
        print("\n✅ SUCCESS: Successfully demonstrated divergent scaling.", flush=True)
        print("   The mean behaves like noise, but the extremes behave like a persistent process.", flush=True)
    else:
        print("\n❌ FAILURE: Did not observe expected divergence.", flush=True)
        print(f"   Mean: {beta_mean}, 90th: {beta_90}", flush=True)

if __name__ == "__main__":
    run_stochastic_volatility_demo()

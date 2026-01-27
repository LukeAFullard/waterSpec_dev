import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from waterSpec.haar_analysis import HaarAnalysis

def generate_random_walk(n_points, seed=None):
    if seed is not None:
        np.random.seed(seed)

    # Random walk: cumulative sum of white noise
    white_noise = np.random.normal(0, 1, n_points)
    data = np.cumsum(white_noise)
    time = np.arange(n_points)
    return time, data

def run_test():
    # 1. Generate Random Walk (beta=2, H=0.5)
    n_points = 2000
    beta_true = 2.0
    seed = 42

    print(f"Generating Random Walk (True beta={beta_true})...")
    time_even, data_even = generate_random_walk(n_points, seed=seed)

    # 2. Create Unevenly Sampled Data
    # Randomly keep only 40% of the data points
    rng = np.random.default_rng(seed)
    mask = rng.random(n_points) < 0.4
    time_uneven = time_even[mask]
    data_uneven = data_even[mask]

    print(f"Original points: {n_points}")
    print(f"Uneven points: {len(time_uneven)} (Subsampled)")

    # 3. Analyze Even Data
    print("\n--- Analyzing Evenly Sampled Data ---")
    haar_even = HaarAnalysis(time_even, data_even)
    res_even = haar_even.run()
    print(f"Estimated H: {res_even['H']:.4f}")
    print(f"Estimated beta: {res_even['beta']:.4f}")
    print(f"R2: {res_even['r2']:.4f}")

    # 4. Analyze Uneven Data
    print("\n--- Analyzing Unevenly Sampled Data ---")
    haar_uneven = HaarAnalysis(time_uneven, data_uneven)
    res_uneven = haar_uneven.run()
    print(f"Estimated H: {res_uneven['H']:.4f}")
    print(f"Estimated beta: {res_uneven['beta']:.4f}")
    print(f"R2: {res_uneven['r2']:.4f}")

    # 5. Plot Comparison
    plt.figure(figsize=(12, 6))

    # Plot Even
    plt.loglog(res_even['lags'], res_even['s1'], 'o-', label=f'Even ($\\beta$={res_even["beta"]:.2f})', alpha=0.7)

    # Plot Uneven
    plt.loglog(res_uneven['lags'], res_uneven['s1'], 's--', label=f'Uneven ($\\beta$={res_uneven["beta"]:.2f})', alpha=0.7)

    plt.xlabel('Lag Time')
    plt.ylabel('Haar Structure Function $S_1$')
    plt.title(f'Haar Analysis Robustness: Even vs Uneven Sampling (True $\\beta$={beta_true})')
    plt.legend()
    plt.grid(True, which="both", ls="-", alpha=0.3)

    output_path = "examples/haar_uneven_comparison.png"
    plt.savefig(output_path)
    print(f"\nComparison plot saved to {output_path}")

if __name__ == "__main__":
    run_test()

import numpy as np
import matplotlib.pyplot as plt
import os
import sys

# Add src to path
sys.path.append(os.path.join(os.getcwd(), 'src'))

from waterSpec.haar_analysis import HaarAnalysis

def generate_red_noise(n_points, seed=42):
    """
    Generates Red Noise (Brownian Motion, beta=2).
    """
    np.random.seed(seed)
    # Brownian motion is the cumulative sum of white noise
    white_noise = np.random.randn(n_points)
    brownian_noise = np.cumsum(white_noise)
    return brownian_noise

def run_example():
    output_dir = "example_output"
    os.makedirs(output_dir, exist_ok=True)

    n_points = 10000
    print(f"Generating Red Noise (Beta=2, N={n_points})...")
    time = np.arange(n_points)
    data = generate_red_noise(n_points)

    # Run Haar Analysis with 100 lags
    print("Running Haar Analysis with 100 lags...")
    ha = HaarAnalysis(time, data)
    res = ha.run(num_lags=100)

    print("-" * 40)
    print(f"Estimated H:    {res['H']:.4f}")
    print(f"Estimated Beta: {res['beta']:.4f} (Expected ~2.0)")
    print(f"R-squared:      {res['r2']:.4f}")
    print("-" * 40)

    # Plot
    plot_path = os.path.join(output_dir, "haar_red_noise_100lags.png")
    ha.plot(output_path=plot_path)
    print(f"Plot saved to {plot_path}")

if __name__ == "__main__":
    run_example()

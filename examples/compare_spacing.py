import numpy as np
import matplotlib.pyplot as plt
import os
import sys

# Add src to path
sys.path.append(os.path.join(os.getcwd(), 'src'))

from waterSpec.haar_analysis import HaarAnalysis

def generate_red_noise(n_points, seed=42):
    np.random.seed(seed)
    white_noise = np.random.randn(n_points)
    brownian_noise = np.cumsum(white_noise)
    return brownian_noise

def run_comparison():
    output_dir = "example_output"
    os.makedirs(output_dir, exist_ok=True)

    n_points = 10000
    print(f"Generating Red Noise (N={n_points})...")
    time = np.arange(n_points)
    data = generate_red_noise(n_points)

    # 1. Log Spacing
    print("Running with Log Spacing...")
    ha_log = HaarAnalysis(time, data)
    res_log = ha_log.run(num_lags=50, log_spacing=True)

    # 2. Linear Spacing
    print("Running with Linear Spacing...")
    ha_lin = HaarAnalysis(time, data)
    res_lin = ha_lin.run(num_lags=50, log_spacing=False)

    print("-" * 60)
    print(f"{'Method':<15} | {'Beta':<10} | {'H':<10} | {'R2':<10}")
    print("-" * 60)
    print(f"{'Log Spacing':<15} | {res_log['beta']:<10.4f} | {res_log['H']:<10.4f} | {res_log['r2']:<10.4f}")
    print(f"{'Linear Spacing':<15} | {res_lin['beta']:<10.4f} | {res_lin['H']:<10.4f} | {res_lin['r2']:<10.4f}")
    print("-" * 60)

    # Plot Comparison
    plt.figure(figsize=(10, 6))

    # Plot Linear results
    plt.loglog(res_lin['lags'], res_lin['s1'], 'bo', alpha=0.5, label='Linear Spacing')

    # Plot Log results
    plt.loglog(res_log['lags'], res_log['s1'], 'rs', alpha=0.5, label='Log Spacing')

    # Plot Fit from Log (usually better for power laws)
    H_log = res_log['H']
    lags = res_log['lags']
    s1 = res_log['s1']
    fit_vals = np.exp(np.polyval(np.polyfit(np.log(lags), np.log(s1), 1), np.log(lags)))
    plt.loglog(lags, fit_vals, 'k--', label=f'Fit (Log): H={H_log:.2f}')

    plt.xlabel('Lag Time $\Delta t$')
    plt.ylabel('Structure Function $S_1(\\Delta t)$')
    plt.title('Haar Analysis: Log vs Linear Spacing (N=50 lags)')
    plt.legend()
    plt.grid(True, which="both", ls="-", alpha=0.2)

    plot_path = os.path.join(output_dir, "haar_spacing_comparison.png")
    plt.savefig(plot_path)
    print(f"Comparison plot saved to {plot_path}")

if __name__ == "__main__":
    run_comparison()

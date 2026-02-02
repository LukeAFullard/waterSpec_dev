
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from waterSpec.utils_sim.tk95 import simulate_tk95
from waterSpec.haar_analysis import HaarAnalysis

def power_law_psd(f, beta):
    return 1.0 / (f ** beta)

def run_sweep():
    betas = np.arange(0, 3.25, 0.25)
    results = []

    N = 4096
    dt = 1.0

    # We will do a few iterations per beta to get a stable average?
    # The user didn't ask for multiple realizations, but it might be safer.
    # I'll do 1 realization for now as per "generate data...". If it's noisy I might increase.
    # Actually, let's do 1 realization but print it out.

    for beta in betas:
        print(f"Processing beta = {beta}")

        # Generate even data
        # Note: input beta for PSD.
        # Haar beta should match this.
        time, flux = simulate_tk95(power_law_psd, (beta,), N, dt, seed=42)

        # Generate uneven data (50% subsample)
        rng = np.random.default_rng(42)
        idx = np.sort(rng.choice(N, size=N//2, replace=False))
        time_uneven = time[idx]
        flux_uneven = flux[idx]

        scenarios = [
            ("Even", False, time, flux),
            ("Even", True, time, flux),
            ("Uneven", False, time_uneven, flux_uneven),
            ("Uneven", True, time_uneven, flux_uneven),
        ]

        for condition, overlap, t, f in scenarios:
            try:
                haar = HaarAnalysis(t, f)
                # We use standard settings
                res = haar.run(overlap=overlap, num_lags=20, n_bootstraps=0) # No bootstrap for speed, we just want beta

                recovered_beta = res['beta']
                recovered_H = res['H']
                r2 = res['r2']

                results.append({
                    "input_beta": beta,
                    "condition": condition,
                    "overlap": overlap,
                    "recovered_beta": recovered_beta,
                    "recovered_H": recovered_H,
                    "r2": r2
                })
            except Exception as e:
                print(f"Failed for beta={beta}, {condition}, overlap={overlap}: {e}")

    df = pd.DataFrame(results)
    print("\nResults Summary:")
    print(df.to_string())

    df.to_csv("beta_sweep_results.csv", index=False)

    # Plotting
    plt.figure(figsize=(10, 6))

    conditions = [
        ("Even", False, "o-", "Even, No Overlap"),
        ("Even", True, "x--", "Even, Overlap"),
        ("Uneven", False, "s-", "Uneven, No Overlap"),
        ("Uneven", True, "^--", "Uneven, Overlap"),
    ]

    for cond_name, overlap_val, fmt, label in conditions:
        subset = df[(df["condition"] == cond_name) & (df["overlap"] == overlap_val)]
        plt.plot(subset["input_beta"], subset["recovered_beta"], fmt, label=label)

    plt.plot([0, 3], [0, 3], 'k-', label="1:1 Ideal")
    plt.xlabel("Input Beta")
    plt.ylabel("Recovered Beta")
    plt.title("Haar Analysis: Input vs Recovered Beta")
    plt.legend()
    plt.grid(True)
    plt.savefig("beta_sweep_plot.png")
    print("Plot saved to beta_sweep_plot.png")

if __name__ == "__main__":
    run_sweep()

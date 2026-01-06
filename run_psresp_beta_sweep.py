import numpy as np
import pandas as pd
import os
import sys
from waterSpec.psresp import psresp_fit, power_law, simulate_tk95, resample_to_times

# Reusing generation logic to match previous tests
def generate_colored_noise(beta, n_points, seed=None):
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
    if seed is not None:
        np.random.seed(seed)
    n = len(time)
    n_keep = int(n * fraction)
    indices = np.sort(np.random.choice(n, n_keep, replace=False))
    return time[indices], data[indices]

def run_sweep():
    output_dir = "validation_psresp"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    true_betas = np.arange(0.0, 3.25, 0.25)

    # Search grid: wider and finer than the sweep steps to avoid edge effects and quantization
    search_betas = np.arange(0.0, 4.1, 0.1)
    params_list = [(b, 1.0) for b in search_betas]

    n_points = 500
    fraction = 0.6
    M = 100 # Simulations per fit
    oversample = 5
    length_factor = 10.0
    n_bins = 15
    seed_base = 42

    results = []

    print(f"{'True Beta':<10} | {'Est. Even':<10} | {'Est. Uneven':<10}")
    print("-" * 36)

    for i, true_beta in enumerate(true_betas):
        # Generate Data
        seed = seed_base + i
        time_even = np.linspace(0, 100, n_points)
        data_even = generate_colored_noise(true_beta, n_points, seed=seed)

        time_uneven, data_uneven = create_irregular_sampling(time_even, data_even, fraction=fraction, seed=seed)

        # Fit Even
        res_even = psresp_fit(
            time_even, data_even, err_obs=None,
            psd_func=power_law,
            params_list=params_list,
            M=M,
            oversample=oversample,
            length_factor=length_factor,
            binning=True,
            n_bins=n_bins,
            normalization='standard',
            n_jobs=-1
        )
        est_even = res_even['best_params'][0]

        # Fit Uneven
        # Use higher oversample for uneven if needed, but 5 is usually OK for this check
        res_uneven = psresp_fit(
            time_uneven, data_uneven, err_obs=None,
            psd_func=power_law,
            params_list=params_list,
            M=M,
            oversample=10,
            length_factor=length_factor,
            binning=True,
            n_bins=n_bins,
            normalization='standard',
            n_jobs=-1
        )
        est_uneven = res_uneven['best_params'][0]

        print(f"{true_beta:<10.2f} | {est_even:<10.2f} | {est_uneven:<10.2f}", flush=True)

        results.append({
            "True_Beta": true_beta,
            "Est_Beta_Even": est_even,
            "Est_Beta_Uneven": est_uneven,
            "Chi2_Even": res_even['best_chi2'],
            "Chi2_Uneven": res_uneven['best_chi2']
        })

    # Save to CSV
    df = pd.DataFrame(results)
    csv_path = os.path.join(output_dir, "beta_sweep_results.csv")
    df.to_csv(csv_path, index=False)
    print(f"\nResults saved to {csv_path}")

    # Generate Markdown Table
    md_path = os.path.join(output_dir, "beta_sweep_results.md")
    with open(md_path, "w") as f:
        f.write("# PSRESP Beta Sweep Results\n\n")
        f.write("| True Beta | Est. Beta (Even) | Est. Beta (Uneven) | Chi2 Even | Chi2 Uneven |\n")
        f.write("| --- | --- | --- | --- | --- |\n")
        for r in results:
            f.write(f"| {r['True_Beta']:.2f} | {r['Est_Beta_Even']:.2f} | {r['Est_Beta_Uneven']:.2f} | {r['Chi2_Even']:.2f} | {r['Chi2_Uneven']:.2f} |\n")
    print(f"Markdown report saved to {md_path}")

if __name__ == "__main__":
    run_sweep()

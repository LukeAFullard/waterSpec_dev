import numpy as np
import matplotlib.pyplot as plt
import os
from waterSpec.psresp import psresp_fit, power_law, simulate_tk95, resample_to_times

def generate_colored_noise(beta, n_points, seed=None):
    """
    Generates colored noise with a power-law spectrum P(f) ~ f^(-beta).
    Reused logic from validation/verify_synthetic.py for consistency.
    """
    if seed is not None:
        np.random.seed(seed)

    # Generate white noise
    white_noise = np.random.normal(0, 1, n_points)

    # FFT
    fft_vals = np.fft.rfft(white_noise)
    frequencies = np.fft.rfftfreq(n_points)

    # Scale amplitudes to get 1/f^(beta/2) amplitude spectrum => 1/f^beta power spectrum
    scaling = np.ones_like(frequencies)
    with np.errstate(divide='ignore'):
        scaling[1:] = frequencies[1:] ** (-beta / 2.0)
    scaling[0] = 0
    fft_shaped = fft_vals * scaling

    # Inverse FFT
    colored_noise = np.fft.irfft(fft_shaped, n=n_points)
    colored_noise = (colored_noise - np.mean(colored_noise)) / np.std(colored_noise)
    return colored_noise

def create_irregular_sampling(time, data, fraction=0.7, seed=None):
    if seed is not None:
        np.random.seed(seed)
    n = len(time)
    n_keep = int(n * fraction)
    indices = np.sort(np.random.choice(n, n_keep, replace=False))
    return time[indices], data[indices]

def run_psresp_analysis():
    output_dir = "validation_psresp"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Parameters matching validation/verify_synthetic.py
    true_beta = 2.0
    n_points = 500
    seed = 42
    fraction = 0.6

    print(f"Generating Red Noise (beta={true_beta}, N={n_points})...")

    # 1. Evenly Spaced
    time_even = np.linspace(0, 100, n_points)
    dt = time_even[1] - time_even[0]
    data_even = generate_colored_noise(true_beta, n_points, seed=seed)

    # 2. Unevenly Spaced
    time_uneven, data_uneven = create_irregular_sampling(time_even, data_even, fraction=fraction, seed=seed)

    # Define search grid for beta
    betas = np.arange(0.0, 3.1, 0.1)
    params_list = [(b, 1.0) for b in betas] # Assume amplitude is handled or fitted?
    # Note: PSRESP usually fits amplitude too, or normalizes variance.
    # Our `simulate_tk95` uses the provided amplitude.
    # However, LombScargle normalization 'standard' divides by variance.
    # If we use 'standard' normalization, amplitude cancels out relative to variance.
    # So fitting beta alone is sufficient if we use 'standard' or normalized 'psd'.
    # Let's use normalization='standard' and fixed amplitude=1.0 for simulation,
    # as the simulated series will be normalized by LS 'standard' mode anyway?
    # Actually, `simulate_tk95` output has variance determined by PSD integral.
    # If we fix amp=1, variance depends on beta.
    # But `normalization='standard'` in LS makes the peak power ~1 (or N/2).
    # So relative shape (slope) is what matters.

    print("\nRunning PSRESP on Even Sampling...")
    res_even = psresp_fit(
        time_even, data_even, err_obs=None,
        psd_func=power_law,
        params_list=params_list,
        M=100, # More sims for robust result
        oversample=5,
        length_factor=10.0,
        binning=True,
        n_bins=15,
        normalization='standard',
        n_jobs=-1
    )

    print("\nRunning PSRESP on Uneven Sampling...")
    res_uneven = psresp_fit(
        time_uneven, data_uneven, err_obs=None,
        psd_func=power_law,
        params_list=params_list,
        M=100,
        oversample=10, # Higher oversample for uneven
        length_factor=10.0,
        binning=True,
        n_bins=15,
        normalization='standard',
        n_jobs=-1
    )

    # Results
    print("\n" + "="*60)
    print(f"{'Scenario':<20} | {'True Beta':<10} | {'Est. Beta':<10} | {'Chi2':<10}")
    print("-" * 60)
    print(f"{'Red Noise Even':<20} | {true_beta:<10.2f} | {res_even['best_params'][0]:<10.2f} | {res_even['best_chi2']:.2f}")
    print(f"{'Red Noise Uneven':<20} | {true_beta:<10.2f} | {res_uneven['best_params'][0]:<10.2f} | {res_uneven['best_chi2']:.2f}")
    print("="*60)

    # Plotting
    plot_results(res_even, "Even Sampling", os.path.join(output_dir, "psresp_red_even.png"))
    plot_results(res_uneven, "Uneven Sampling", os.path.join(output_dir, "psresp_red_uneven.png"))

    # Plot Chi2 Surface
    plot_chi2(res_even, res_uneven, betas, os.path.join(output_dir, "psresp_chi2_surface.png"))

def plot_results(result, title, filename):
    plt.figure(figsize=(10, 6))

    # Plot Observed Binned Power
    # Note: result keys: 'target_freqs', 'target_power' (binned)
    plt.loglog(result['target_freqs'], result['target_power'], 'ko-', label='Observed (Binned)', lw=2)

    # Find best model mean power
    best_params = result['best_params']
    # We need to find the result entry corresponding to best_params
    best_entry = next(r for r in result['results'] if r['params'] == best_params)

    # Mean simulated power (binned)
    # The stored 'mean_sim_log_power' is log10(power).
    mean_power = 10**best_entry['mean_sim_log_power']

    # Std dev of log power
    std_log_power = best_entry['std_sim_log_power']
    upper_power = 10**(best_entry['mean_sim_log_power'] + std_log_power)
    lower_power = 10**(best_entry['mean_sim_log_power'] - std_log_power)

    plt.loglog(result['target_freqs'], mean_power, 'r--', label=f'Best Model (beta={best_params[0]:.1f})', lw=2)
    plt.fill_between(result['target_freqs'], lower_power, upper_power, color='r', alpha=0.2, label='1-sigma Sim. Spread')

    plt.title(f"{title} - PSRESP Fit")
    plt.xlabel("Frequency")
    plt.ylabel("Power")
    plt.legend()
    plt.grid(True, which="both", alpha=0.3)
    plt.savefig(filename)
    plt.close()

def plot_chi2(res_even, res_uneven, betas, filename):
    plt.figure(figsize=(10, 6))

    chi2_even = [r['chi2'] for r in res_even['results']]
    chi2_uneven = [r['chi2'] for r in res_uneven['results']]

    plt.plot(betas, chi2_even, 'b-o', label='Even Sampling')
    plt.plot(betas, chi2_uneven, 'r-s', label='Uneven Sampling')

    # Mark minima
    min_even = min(chi2_even)
    best_beta_even = betas[chi2_even.index(min_even)]
    plt.axvline(best_beta_even, color='b', linestyle=':', alpha=0.5)

    min_uneven = min(chi2_uneven)
    best_beta_uneven = betas[chi2_uneven.index(min_uneven)]
    plt.axvline(best_beta_uneven, color='r', linestyle=':', alpha=0.5)

    plt.xlabel("Beta (Spectral Slope)")
    plt.ylabel("Chi-Squared")
    plt.title("Goodness of Fit Surface")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(filename)
    plt.close()

if __name__ == "__main__":
    run_psresp_analysis()

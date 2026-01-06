import numpy as np
import pytest
from waterSpec.psresp import psresp_fit, simulate_tk95, power_law, resample_to_times

def test_psresp_even_sampling_red_noise():
    """
    Test PSRESP recovery of Red Noise (beta=2.0) with Even Sampling.
    """
    # 1. Setup Evenly Sampled Data
    np.random.seed(123)
    N_obs = 100
    T_obs = 100.0
    dt = T_obs / N_obs
    t_obs = np.arange(N_obs) * dt

    # 2. Generate Synthetic Data (Red Noise, beta=2.0)
    true_beta = 2.0
    true_amp = 1.0

    # Generate high-res underlying signal to avoid generation artifacts
    N_fine = 1000
    dt_fine = T_obs / N_fine
    t_fine, x_fine = simulate_tk95(power_law, (true_beta, true_amp), N_fine, dt_fine)

    # Resample to observed times (even)
    x_obs = resample_to_times(t_fine, x_fine, t_obs)

    # Add small measurement noise
    noise_level = 0.1
    err_obs = np.ones_like(x_obs) * noise_level
    x_obs += np.random.normal(0, noise_level, size=len(x_obs))

    # 3. PSRESP Search
    # We search for beta in a grid.
    betas_to_test = [1.0, 1.5, 2.0, 2.5]
    params_list = [(b, true_amp) for b in betas_to_test]

    # Run PSRESP
    result = psresp_fit(
        t_obs, x_obs, err_obs,
        psd_func=power_law,
        params_list=params_list,
        M=50, # Sufficient for clear distinction
        oversample=5,
        length_factor=5.0, # Handle red noise leakage
        binning=True,
        n_bins=10,
        n_jobs=1 # Simpler for test stability
    )

    best_beta = result["best_params"][0]

    # With beta=2.0, it is distinctively red.
    assert best_beta == 2.0, f"Expected beta=2.0, got {best_beta}. Chi2 values: {[ (r['params'][0], r['chi2']) for r in result['results'] ]}"

def test_psresp_uneven_sampling_red_noise():
    """
    Test PSRESP recovery of Red Noise (beta=2.0) with Uneven Sampling.
    Uneven sampling introduces spectral leakage which standard LS might misinterpret,
    but PSRESP should handle by forward modelling the window.
    """
    # 1. Setup Unevenly Sampled Data
    np.random.seed(456)
    N_obs = 100
    T_obs = 100.0

    # Random timestamps (Poisson sampling approximation)
    t_obs = np.sort(np.random.uniform(0, T_obs, N_obs))

    # 2. Generate Synthetic Data (Red Noise, beta=2.0)
    true_beta = 2.0
    true_amp = 1.0

    N_fine = 2000
    dt_fine = T_obs / N_fine
    t_fine, x_fine = simulate_tk95(power_law, (true_beta, true_amp), N_fine, dt_fine)

    # Resample to uneven times
    x_obs = resample_to_times(t_fine, x_fine, t_obs)

    # Add measurement noise
    noise_level = 0.1 * np.std(x_obs) # 10% noise
    err_obs = np.ones_like(x_obs) * noise_level
    x_obs += np.random.normal(0, noise_level, size=len(x_obs))

    # 3. PSRESP Search
    betas_to_test = [1.0, 1.5, 2.0, 2.5, 3.0]
    params_list = [(b, true_amp) for b in betas_to_test]

    # Run PSRESP
    # We use longer length_factor because red noise leakage is significant for uneven sampling of steep spectra
    result = psresp_fit(
        t_obs, x_obs, err_obs,
        psd_func=power_law,
        params_list=params_list,
        M=50,
        oversample=5,
        length_factor=10.0,
        binning=True,
        n_bins=10,
        n_jobs=1
    )

    best_beta = result["best_params"][0]

    assert best_beta == 2.0, f"Expected beta=2.0, got {best_beta}. Chi2 values: {[ (r['params'][0], r['chi2']) for r in result['results'] ]}"

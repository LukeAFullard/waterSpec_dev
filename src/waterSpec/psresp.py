from __future__ import annotations

import warnings
import os
from typing import Callable, List, Tuple, Optional, Union, Dict, Any
import concurrent.futures

import numpy as np
from astropy.timeseries import LombScargle

# Import shared simulation utilities
from .utils_sim import simulate_tk95, resample_to_times

def _run_single_simulation(
    i: int,
    psd_func: Callable,
    params: Tuple,
    t_obs_relative: np.ndarray,
    err_obs: np.ndarray,
    freqs: np.ndarray,
    N_sim: int,
    dt_sim: float,
    normalization: str = "psd",
    seed: Optional[int] = None
) -> np.ndarray:
    """
    Helper function to run a single simulation iteration.
    To be used with multiprocessing.
    """
    # Seed is handled inside simulate_tk95 now if passed,
    # but simulate_tk95 expects an integer seed, not setting global seed.
    # The original code set global seed.
    # The new simulate_tk95 takes a 'seed' argument.

    t_sim, x_sim = simulate_tk95(psd_func, params, N_sim, dt_sim, seed=seed)

    # 2. Resample
    # t_obs_relative is already shifted to start at 0 (or close to 0)
    x_resampled = resample_to_times(t_sim, x_sim, t_obs_relative)

    # 3. Add noise
    if err_obs is not None:
        rng = np.random.default_rng(seed) # Use same seed for noise
        noise = rng.normal(0, err_obs)
        x_resampled += noise

    # 4. Compute Periodogram
    ls = LombScargle(t_obs_relative, x_resampled, dy=err_obs)
    power = ls.power(freqs, normalization=normalization)

    return power

def bin_power_spectrum(
    freqs: np.ndarray,
    power: np.ndarray,
    bins: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Bin the power spectrum into specified frequency bins.
    Returns bin centers and mean power in each bin.
    """
    digitized = np.digitize(freqs, bins)
    binned_power = []
    binned_freqs = []

    for i in range(1, len(bins)):
        mask = digitized == i
        if np.any(mask):
            binned_power.append(np.mean(power[mask]))
            binned_freqs.append(np.mean(freqs[mask])) # Or geometric mean? Arithmetic is fine for now.
        else:
            binned_power.append(np.nan)
            binned_freqs.append((bins[i-1] + bins[i])/2)

    return np.array(binned_freqs), np.array(binned_power)

def psresp_fit(
    t_obs: np.ndarray,
    x_obs: np.ndarray,
    err_obs: np.ndarray,
    psd_func: Callable,
    params_list: List[Tuple],
    freqs: np.ndarray = None,
    M: int = 500,
    oversample: int = 5,
    length_factor: float = 10.0,
    normalization: str = "psd",
    n_jobs: int = -1,
    binning: bool = True,
    n_bins: int = 20,
    seed: Optional[int] = None
) -> Dict[str, Any]:
    """
    Perform PSRESP (Power Spectral Response) analysis.

    Args:
        ...
        binning: If True, bin the power spectrum in log-space before comparison.
        n_bins: Number of bins if binning is enabled.
        n_jobs: Number of parallel jobs. If -1, uses all available CPUs (limited to 32 to prevent issues).
    """

    # Handle time offset
    t_start = t_obs.min()
    t_obs_relative = t_obs - t_start
    T_obs = t_obs.max() - t_start
    N_obs = len(t_obs)
    dt_avg = T_obs / (N_obs - 1)

    # Simulation properties
    dt_sim = dt_avg / oversample
    T_sim = T_obs * length_factor
    N_sim = int(np.ceil(T_sim / dt_sim))

    # Memory safety check
    estimated_points = N_sim * M
    if estimated_points > 1e8: # > 100M points total processing
         warnings.warn(f"High memory usage expected: M={M} simulations of N={N_sim} points. Consider reducing M or oversample.", UserWarning)

    # Frequency grid
    if freqs is None:
        f_min = 1.0 / T_obs
        f_max = 0.5 / dt_avg
        # High resolution grid for LS calculation
        n_freqs = int((f_max - f_min) * T_obs * oversample)
        n_freqs = max(n_freqs, 100)
        freqs = np.linspace(f_min, f_max, n_freqs)

    # Setup bins if requested
    if binning:
        f_min_bin = freqs.min()
        f_max_bin = freqs.max()
        bins = np.logspace(np.log10(f_min_bin), np.log10(f_max_bin), n_bins + 1)

    # Calculate observed periodogram
    ls_obs = LombScargle(t_obs_relative, x_obs, dy=err_obs)
    obs_power = ls_obs.power(freqs, normalization=normalization)

    if binning:
        obs_bin_freqs, obs_bin_power = bin_power_spectrum(freqs, obs_power, bins)
        # Use binned values for comparison
        target_power = obs_bin_power
        # Filter NaNs (empty bins)
        valid_mask = ~np.isnan(target_power)
        target_power = target_power[valid_mask]
        target_freqs = obs_bin_freqs[valid_mask]
    else:
        target_power = obs_power
        target_freqs = freqs

    results = []
    target_log_power = np.log10(target_power)

    # Generate seeds for simulations
    # We generate M seeds once and reuse them for each parameter set.
    # This uses Common Random Numbers (CRN) which reduces variance when comparing models.
    sim_seeds = [None] * M
    if seed is not None:
        ss = np.random.SeedSequence(seed)
        sim_seeds = ss.generate_state(M)

    # Determine max workers safely
    if n_jobs < 0:
        max_workers = os.cpu_count() or 1
        # Limit to 32 to prevent excessive overhead/OS limits
        max_workers = min(max_workers, 32)
    else:
        max_workers = max(1, n_jobs) # Ensure at least 1 worker

    for params in params_list:
        sim_binned_powers = []

        # Process in chunks to manage memory and future overhead
        chunk_size = max(1, M // max_workers) # Ensure reasonable chunk size, or just submit all if M is small
        # Actually, Python's ProcessPoolExecutor handles queuing well, but if we create all M futures
        # and M is huge, that consumes memory. But M=500 is fine.
        # The main memory issue is if N_sim is huge.

        # We will submit all for simplicity but rely on max_workers limiting active processes.
        # But we must be careful not to hold all result arrays in memory if they are huge and binning=False.

        with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = [
                executor.submit(
                    _run_single_simulation,
                    i, psd_func, params, t_obs_relative, err_obs, freqs, N_sim, dt_sim, normalization,
                    seed=sim_seeds[i]
                )
                for i in range(M)
            ]

            for future in concurrent.futures.as_completed(futures):
                p = future.result()
                if binning:
                    _, bp = bin_power_spectrum(freqs, p, bins)
                    sim_binned_powers.append(bp[valid_mask])
                else:
                    sim_binned_powers.append(p)

        sim_binned_powers = np.array(sim_binned_powers) # Shape (M, n_bins_valid)

        # Statistics on log power
        sim_log_powers = np.log10(sim_binned_powers)
        mean_sim_log_power = np.mean(sim_log_powers, axis=0)
        var_sim_log_power = np.var(sim_log_powers, axis=0, ddof=1)

        var_sim_log_power[var_sim_log_power == 0] = 1e-10

        chi2 = np.sum( (target_log_power - mean_sim_log_power)**2 / var_sim_log_power )

        chi2_sims = np.sum( (sim_log_powers - mean_sim_log_power)**2 / var_sim_log_power, axis=1 )
        count_worse = np.sum(chi2_sims >= chi2)
        success_fraction = count_worse / M

        results.append({
            "params": params,
            "chi2": chi2,
            "success_fraction": success_fraction,
            "mean_sim_log_power": mean_sim_log_power,
            "std_sim_log_power": np.sqrt(var_sim_log_power),
        })

    best_result = min(results, key=lambda x: x["chi2"])

    return {
        "best_params": best_result["params"],
        "best_chi2": best_result["chi2"],
        "results": results,
        "freqs": freqs, # Original fine grid
        "obs_power": obs_power, # Original fine power
        "target_freqs": target_freqs, # Used for fitting (binned)
        "target_power": target_power # Used for fitting (binned)
    }

from __future__ import annotations

import warnings
from typing import Callable, List, Tuple, Optional, Union, Dict, Any
import concurrent.futures

import numpy as np
from astropy.timeseries import LombScargle

# Lazy import or try-except for wavelet module
try:
    from .wavelet import compute_wwz
except ImportError:
    compute_wwz = None


def simulate_tk95(
    psd_func: Callable,
    params: Tuple,
    N: int,
    dt: float
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Simulate a time series with a given PSD using the Timmer & Koenig (1995) method.

    Args:
        psd_func (Callable): Function that computes the PSD. Signature: func(f, *params).
        params (Tuple): Parameters for the PSD function.
        N (int): Number of time points to simulate.
        dt (float): Time step.

    Returns:
        Tuple[np.ndarray, np.ndarray]: (time, flux) arrays.
    """
    freqs = np.fft.rfftfreq(N, d=dt)

    psd = np.zeros_like(freqs)
    mask = freqs > 0
    psd[mask] = psd_func(freqs[mask], *params)

    scale = np.sqrt(psd * N / (2 * dt))

    real_part = np.random.normal(0, 1, size=len(freqs)) * scale
    imag_part = np.random.normal(0, 1, size=len(freqs)) * scale

    real_part[0] = 0
    imag_part[0] = 0

    if N % 2 == 0:
        imag_part[-1] = 0
        real_part[-1] *= np.sqrt(2)

    fourier_components = real_part + 1j * imag_part

    flux = np.fft.irfft(fourier_components, n=N)
    time = np.arange(N) * dt

    return time, flux

def resample_to_times(
    source_time: np.ndarray,
    source_flux: np.ndarray,
    target_time: np.ndarray
) -> np.ndarray:
    """
    Resample the simulated time series to the observed time stamps using linear interpolation.
    NOTE: target_time must be relative to the start of the simulation (normally 0).
    If target_time has large offsets (like MJD), subtract the start time before calling this.
    """
    return np.interp(target_time, source_time, source_flux)

def power_law(f: Union[float, np.ndarray], beta: float, amp: float) -> Union[float, np.ndarray]:
    """
    Power law PSD: P(f) = amp * f^(-beta)
    """
    return amp * (f**(-beta))

def _compute_spectra(
    t: np.ndarray,
    x: np.ndarray,
    dy: Optional[np.ndarray],
    freqs: np.ndarray,
    method: str = "ls",
    normalization: str = "psd",
    **kwargs
) -> np.ndarray:
    """
    Compute power spectrum using specified method.
    """
    if method == "ls":
        ls = LombScargle(t, x, dy=dy)
        return ls.power(freqs, normalization=normalization)

    elif method == "wwz":
        if compute_wwz is None:
            raise ImportError("pyleoclim is required for WWZ in PSRESP.")

        # WWZ typically determines its own frequency grid, but we can try to influence it
        # or interpolate to the target grid.
        # compute_wwz wrapper takes freq_min, freq_max, n_scales.

        # Ideally we pass 'freqs' directly if supported, or interpolate.
        # pyleoclim's wwz supports passing 'freq' directly if freq_method is None or ignored?
        # My current wrapper compute_wwz assumes 'log' method if not specified.
        # Let's try to pass freq_min/max based on freqs range and interpolate result.

        # Optimization: PSRESP often requires specific frequencies.
        # Using interpolation is robust.

        # Determine n_scales from freqs length
        n_scales = len(freqs)
        f_min = freqs.min()
        f_max = freqs.max()

        # Call WWZ
        # We might want to pass 'decay_constant' via kwargs
        decay_constant = kwargs.get('decay_constant', 1.0 / (8 * np.pi**2))

        res = compute_wwz(
            t, x,
            freq_method='log',
            n_scales=n_scales,
            freq_min=f_min,
            freq_max=f_max,
            decay_constant=decay_constant
        )

        # Interpolate global_power to requested freqs
        # Note: res.frequencies might not exactly match requested freqs
        power_interp = np.interp(freqs, res.frequencies, res.global_power)
        return power_interp

    else:
        raise ValueError(f"Unknown spectral method: {method}")


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
    method: str = "ls",
    **kwargs
) -> np.ndarray:
    """
    Helper function to run a single simulation iteration.
    To be used with multiprocessing.
    """
    np.random.seed() # Reseed with entropy

    t_sim, x_sim = simulate_tk95(psd_func, params, N_sim, dt_sim)

    # 2. Resample
    # t_obs_relative is already shifted to start at 0 (or close to 0)
    x_resampled = resample_to_times(t_sim, x_sim, t_obs_relative)

    # 3. Add noise
    if err_obs is not None:
        noise = np.random.normal(0, err_obs)
        x_resampled += noise

    # 4. Compute Periodogram
    power = _compute_spectra(
        t_obs_relative, x_resampled, err_obs, freqs,
        method=method, normalization=normalization, **kwargs
    )

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
    method: str = "ls",
    **kwargs
) -> Dict[str, Any]:
    """
    Perform PSRESP (Power Spectral Response) analysis.

    Args:
        method: Spectral estimation method ('ls' or 'wwz').
        **kwargs: Arguments passed to the spectral estimator (e.g., decay_constant for WWZ).
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
    obs_power = _compute_spectra(
        t_obs_relative, x_obs, err_obs, freqs,
        method=method, normalization=normalization, **kwargs
    )

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

    for params in params_list:
        sim_binned_powers = []

        with concurrent.futures.ProcessPoolExecutor(max_workers=n_jobs if n_jobs > 0 else None) as executor:
            futures = [
                executor.submit(
                    _run_single_simulation,
                    i, psd_func, params, t_obs_relative, err_obs, freqs, N_sim, dt_sim, normalization, method, **kwargs
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


import numpy as np
from typing import Callable, Tuple, Union

def simulate_tk95(
    psd_func: Callable = None,
    params: Tuple = None,
    N: int = None,
    dt: float = None,
    seed: int = None,
    n_simulations: int = None,
    precomputed_scale: np.ndarray = None
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Simulate time series with a given PSD using the Timmer & Koenig (1995) method.

    Args:
        psd_func (Callable, optional): Function that computes the PSD. Signature: func(f, *params).
        params (Tuple, optional): Parameters for the PSD function.
        N (int): Number of time points to simulate.
        dt (float): Time step.
        seed (int, optional): Random seed.
        n_simulations (int, optional): Number of time series to simulate.
                                       If None, returns a 1D array of shape (N,).
                                       If provided, returns a 2D array of shape (n_simulations, N).
        precomputed_scale (np.ndarray, optional): Precomputed scale array (sqrt(PSD * N / (2 * dt))).
                                                If provided, psd_func and params are ignored.

    Returns:
        Tuple[np.ndarray, np.ndarray]: (time, flux) arrays.
    """
    if N is None or dt is None:
        raise ValueError("N and dt must be provided.")

    rng = np.random.default_rng(seed)

    if precomputed_scale is not None:
        scale = precomputed_scale
    else:
        if psd_func is None or params is None:
            raise ValueError("psd_func and params must be provided if precomputed_scale is None.")
        freqs = np.fft.rfftfreq(N, d=dt)
        psd = np.zeros_like(freqs)
        mask = freqs > 0
        psd[mask] = psd_func(freqs[mask], *params)
        scale = np.sqrt(psd * N / (2 * dt))

    if n_simulations is None:
        real_part = rng.standard_normal(size=len(scale)) * scale
        imag_part = rng.standard_normal(size=len(scale)) * scale
        real_part[0] = 0
        imag_part[0] = 0
        if N % 2 == 0:
            imag_part[-1] = 0
            real_part[-1] *= np.sqrt(2)
    else:
        real_part = rng.standard_normal(size=(n_simulations, len(scale))) * scale
        imag_part = rng.standard_normal(size=(n_simulations, len(scale))) * scale
        real_part[:, 0] = 0
        imag_part[:, 0] = 0
        if N % 2 == 0:
            imag_part[:, -1] = 0
            real_part[:, -1] *= np.sqrt(2)

    fourier_components = real_part + 1j * imag_part

    # irfft works on the last axis by default, which is what we want for n_simulations provided
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

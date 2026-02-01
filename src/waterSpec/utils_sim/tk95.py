
import numpy as np
from typing import Callable, Tuple, Union

def simulate_tk95(
    psd_func: Callable,
    params: Tuple,
    N: int,
    dt: float,
    seed: int = None
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
    rng = np.random.default_rng(seed)

    freqs = np.fft.rfftfreq(N, d=dt)

    psd = np.zeros_like(freqs)
    mask = freqs > 0
    psd[mask] = psd_func(freqs[mask], *params)

    scale = np.sqrt(psd * N / (2 * dt))

    real_part = rng.standard_normal(size=len(freqs)) * scale
    imag_part = rng.standard_normal(size=len(freqs)) * scale

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


import numpy as np
import warnings
from typing import Optional
from waterSpec.utils_sim import simulate_tk95, resample_to_times, power_law

def generate_phase_randomized_surrogates(
    data: np.ndarray,
    n_surrogates: int = 100,
    seed: Optional[int] = None
) -> np.ndarray:
    """
    Generates surrogates by randomizing the Fourier phases while preserving the
    amplitude spectrum (and thus autocorrelation). This is Method 5.2 from the Audit Plan.

    .. warning::
        **Assumes Even Sampling**

        This function uses the standard Fast Fourier Transform (FFT), which implicitly
        assumes the data is evenly sampled (constant time step).

        If you use this on irregularly sampled data, the resulting surrogates and
        any statistical tests derived from them will be **invalid**.

        For irregular data, you must either:
        1. Interpolate your data to a regular grid before calling this function (as done in `BivariateAnalysis`).
        2. Use `generate_power_law_surrogates` which is designed for irregular data.

    Args:
        data (np.ndarray): Input time series (must be evenly spaced or interpolated).
                           NaNs will be treated as zeros or propagate errors in FFT.
        n_surrogates (int): Number of surrogates to generate.
        seed (int): Random seed.

    Returns:
        np.ndarray: Array of shape (n_surrogates, n_points) containing surrogate series.
    """
    rng = np.random.default_rng(seed)
    n = len(data)

    if np.any(np.isnan(data)):
        warnings.warn(
            "Input data contains NaNs. FFT will propagate these, resulting in garbage surrogates. "
            "Please fill or interpolate NaNs before generating phase-randomized surrogates.",
            UserWarning
        )

    # FFT
    fft_data = np.fft.rfft(data)
    n_freqs = len(fft_data)

    # Amplitudes (preserve these)
    amplitudes = np.abs(fft_data)

    # Generate random phases for all surrogates at once
    # Shape: (n_surrogates, n_freqs)
    phases = rng.uniform(-np.pi, np.pi, size=(n_surrogates, n_freqs))

    # Keep DC component (idx 0) phase as 0 (real)
    phases[:, 0] = 0

    # If n is even, Nyquist component (last) must also be real
    if n % 2 == 0:
        phases[:, -1] = 0

    # Construct new complex spectrum using broadcasting
    # amplitudes shape: (n_freqs,) -> broadcasts to (n_surrogates, n_freqs)
    new_fft = amplitudes * np.exp(1j * phases)

    # Inverse FFT along the last axis
    # returns shape: (n_surrogates, n)
    surrogates = np.fft.irfft(new_fft, n=n, axis=-1)

    # Original IAAFT methods iterate to match amplitude distribution too.
    # This is basic Phase Randomization (preserves linear correlation, but not distribution).
    # For simple robustness checks, this is usually sufficient for testing linear correlation significance.
    # But if data is highly non-Gaussian, IAAFT is better.
    # For now, we stick to simple Phase Randomization as per plan (Method 5.2 table col 2).

    return surrogates

def generate_block_shuffled_surrogates(
    data: np.ndarray,
    block_size: int,
    n_surrogates: int = 100,
    seed: Optional[int] = None
) -> np.ndarray:
    """
    Generates surrogates by shuffling blocks of data.
    Destroys long-term memory > block_size, preserves short-term structure.

    .. warning::
        **Assumes Indices map to Time**

        This method shuffles blocks of *indices*. If your data is irregularly sampled,
        a block of N indices does not correspond to a constant duration of time.
        Use with caution on irregular data.
    """
    rng = np.random.default_rng(seed)
    n = len(data)
    n_blocks = n // block_size

    # Truncate to multiple of block size for simplicity
    n_trunc = n_blocks * block_size
    reshaped = data[:n_trunc].reshape(n_blocks, block_size)

    surrogates = []

    for _ in range(n_surrogates):
        # Shuffle rows (blocks)
        perm = rng.permutation(n_blocks)
        shuffled = reshaped[perm].flatten()

        # If we had leftovers, append them (or handle circular)
        if n > n_trunc:
            # Just append original tail? Or wrap?
            # Appending original tail might preserve local correlation at end.
            shuffled = np.concatenate([shuffled, data[n_trunc:]])

        surrogates.append(shuffled)

    return np.array(surrogates)

def calculate_significance_p_value(
    observed_metric: float,
    surrogate_metrics: np.ndarray,
    two_sided: bool = True
) -> float:
    """
    Calculates empirical p-value.

    p = (1 + count(surr >= obs)) / (N + 1)
    """
    n_surr = len(surrogate_metrics)
    if n_surr == 0:
        return np.nan

    if two_sided:
        # Check absolute magnitude
        count = np.sum(np.abs(surrogate_metrics) >= np.abs(observed_metric))
    else:
        # One-sided (obs > surr)
        count = np.sum(surrogate_metrics >= observed_metric)

    return (count + 1) / (n_surr + 1)

def generate_power_law_surrogates(
    time: np.ndarray,
    beta: float,
    n_surrogates: int = 100,
    seed: Optional[int] = None,
    oversample: int = 5
) -> np.ndarray:
    """
    Generates surrogates with a specific power-law spectral slope (1/f^beta)
    sampled at the specific (potentially irregular) timestamps provided.

    This uses the Timmer & Koenig (1995) method to simulate a high-resolution
    process and then resamples it to the observed times. This is the "Lomb-Scargle"
    compatible surrogate method (model-based).

    Args:
        time (np.ndarray): Observed timestamps.
        beta (float): Spectral slope (e.g. 0=white, 1=pink, 2=brown).
        n_surrogates (int): Number of surrogates.
        seed (int): Random seed.
        oversample (int): Factor to oversample the simulation grid relative to average sampling.

    Returns:
        np.ndarray: Array of shape (n_surrogates, len(time)).
    """
    # Determine simulation parameters
    t_start = time.min()
    t_relative = time - t_start
    duration = time.max() - t_start
    n_points = len(time)
    dt_avg = duration / (n_points - 1) if n_points > 1 else 1.0

    dt_sim = dt_avg / oversample
    # Ensure simulation covers the full duration plus a bit
    N_sim = int(np.ceil(duration / dt_sim)) + 100

    # 1. Simulate high-res regular processes in one batch.
    # Amplitude 1.0 is arbitrary; usually we match variance later.
    # Generating in batches avoids redundant PSD calculations and uses vectorized operations.
    t_sim, x_sims = simulate_tk95(
        psd_func=power_law,
        params=(beta, 1.0),
        N=N_sim,
        dt=dt_sim,
        seed=seed,
        n_simulations=n_surrogates
    )

    surrogates = []

    for i in range(n_surrogates):
        x_sim = x_sims[i]

        # 2. Resample to observed times
        x_resampled = resample_to_times(t_sim, x_sim, t_relative)

        # 3. Normalize to zero mean, unit variance (standard practice for shape comparison)
        # Or match original variance? Let's standardize.
        if np.std(x_resampled) > 0:
            x_resampled = (x_resampled - np.mean(x_resampled)) / np.std(x_resampled)

        surrogates.append(x_resampled)

    return np.array(surrogates)

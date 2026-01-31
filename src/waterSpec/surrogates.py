
import numpy as np
from typing import Optional

def generate_phase_randomized_surrogates(
    data: np.ndarray,
    n_surrogates: int = 100,
    seed: Optional[int] = None
) -> np.ndarray:
    """
    Generates surrogates by randomizing the Fourier phases while preserving the
    amplitude spectrum (and thus autocorrelation).

    Args:
        data (np.ndarray): Input time series (must be evenly spaced or interpolated).
        n_surrogates (int): Number of surrogates to generate.
        seed (int): Random seed.

    Returns:
        np.ndarray: Array of shape (n_surrogates, n_points) containing surrogate series.
    """
    rng = np.random.default_rng(seed)
    n = len(data)

    # FFT
    fft_data = np.fft.rfft(data)
    n_freqs = len(fft_data)

    # Amplitudes (preserve these)
    amplitudes = np.abs(fft_data)

    surrogates = []

    for _ in range(n_surrogates):
        # Generate random phases in [-pi, pi]
        # Keep DC component (idx 0) phase as 0 (real)
        # If n is even, Nyquist component (last) must also be real

        phases = rng.uniform(-np.pi, np.pi, size=n_freqs)
        phases[0] = 0
        if n % 2 == 0:
            phases[-1] = 0

        # Construct new complex spectrum
        new_fft = amplitudes * np.exp(1j * phases)

        # Inverse FFT
        surr = np.fft.irfft(new_fft, n=n)

        # Original IAAFT methods iterate to match amplitude distribution too.
        # This is basic Phase Randomization (preserves linear correlation, but not distribution).
        # For simple robustness checks, this is usually sufficient for testing linear correlation significance.
        # But if data is highly non-Gaussian, IAAFT is better.
        # For now, we stick to simple Phase Randomization as per plan (Method 5.2 table col 2).

        surrogates.append(surr)

    return np.array(surrogates)

def generate_block_shuffled_surrogates(
    data: np.ndarray,
    block_size: int,
    n_surrogates: int = 100,
    seed: Optional[int] = None
) -> np.ndarray:
    """
    Generates surrogates by shuffling blocks of data.
    Destroys long-term memory > block_size, preserves short-term structure.
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

import numpy as np

def interpret_beta(beta):
    """
    Provides a scientific interpretation of the spectral exponent (beta).

    Args:
        beta (float): The spectral exponent value.

    Returns:
        str: A string containing the interpretation.

    Raises:
        ValueError: If the beta value is significantly negative.
    """
    # Using np.isclose for floating point comparisons to handle noisy data
    # where beta might be slightly negative but effectively zero.
    if np.isclose(beta, 0, atol=0.1):
        return "β ≈ 0: White noise. The time series is random and uncorrelated."

    if beta < -0.1:
        raise ValueError("Beta value cannot be significantly negative.")

    if 0.1 <= beta < 0.5:
        return "0 < β < 0.5 (fGn): Weak persistence. Signal is highly event-driven (e.g., surface runoff)."

    if 0.5 <= beta < 1.0:
        return "0.5 ≤ β < 1 (fGn): Weak persistence. Suggests mixed event-driven and subsurface processes."

    if np.isclose(beta, 1, atol=0.1):
        return "β ≈ 1: Pink noise (1/f noise). A common signal in nature, with a mix of persistence and randomness."

    if 1 < beta < 3:
        if np.isclose(beta, 2, atol=0.1):
             return "β ≈ 2: Brownian motion (random walk). Strong persistence and accumulation processes."
        return "1 < β < 3 (fBm): Strong persistence. Signal is damped and influenced by storage (e.g., subsurface flow)."

    if beta >= 3:
        return "β ≥ 3: Black noise. Very strong persistence, smooth signal. May indicate non-stationarity."

    # Fallback for any unexpected cases, though the logic above should be comprehensive.
    return "No interpretation available for this beta value."

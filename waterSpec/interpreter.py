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
    if beta < -0.1:
        raise ValueError("Beta value cannot be significantly negative.")

    # Prioritize the more specific "isclose" checks first.
    if np.isclose(beta, 0, atol=0.1):
        return "β ≈ 0: White noise. The time series is random and uncorrelated."

    if np.isclose(beta, 1, atol=0.1):
        return "β ≈ 1: Pink noise (1/f noise). A common signal in nature, with a mix of persistence and randomness."

    if np.isclose(beta, 2, atol=0.1):
        return "β ≈ 2: Brownian motion (random walk). Strong persistence and accumulation processes."

    # Now handle the broader, mutually exclusive ranges.
    if 0.1 <= beta < 0.9: # Changed from < 0.5 and < 1.0 to be more distinct
        return "0.1 < β < 0.9 (fGn): Weak persistence. Signal may be event-driven or have mixed processes."

    if 1.1 <= beta < 1.9:
        return "1.1 < β < 1.9 (fBm): Strong persistence. Signal is damped and influenced by storage (e.g., subsurface flow)."

    if 2.1 <= beta < 3:
        return "2.1 < β < 3 (fBm): Very strong persistence and accumulation processes."

    if beta >= 3:
        return "β ≥ 3: Black noise. Very strong persistence, smooth signal. May indicate non-stationarity."

    # Fallback for any unexpected cases, though the logic above should be comprehensive.
    return "No interpretation available for this beta value."

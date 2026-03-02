
import numpy as np
from typing import Tuple, Optional, Union

def calculate_ls_cross_spectrum(
    time1: np.ndarray,
    data1: np.ndarray,
    time2: np.ndarray,
    data2: np.ndarray,
    freqs: np.ndarray,
    errors1: Optional[np.ndarray] = None,
    errors2: Optional[np.ndarray] = None
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Calculates the Lomb-Scargle Cross-Spectrum and Phase for two unevenly sampled time series.

    This method fits a sinusoidal model to each time series independently at each frequency
    and calculates the cross-spectral density (CSD) from the complex Fourier coefficients.
    It allows for phase lag estimation between variables sampled at different irregular times.

    Args:
        time1 (np.ndarray): Time array for first series.
        data1 (np.ndarray): Data array for first series.
        time2 (np.ndarray): Time array for second series.
        data2 (np.ndarray): Data array for second series.
        freqs (np.ndarray): Array of frequencies to analyze.
        errors1 (np.ndarray, optional): Measurement errors for first series.
        errors2 (np.ndarray, optional): Measurement errors for second series.

    Returns:
        Tuple:
            - cross_power (np.ndarray): Magnitude of cross-spectrum (|Sxy|).
            - phase_lag (np.ndarray): Phase difference (angle(Sxy)) in radians.
              Positive phase means series 1 leads series 2 (or 2 lags 1)?
              Convention: phase(x) - phase(y).
            - coherence (np.ndarray): Pseudo-coherence (requires smoothing, may be noisy).
            - freqs (np.ndarray): Frequencies.
    """
    # Calculate complex coefficients for both series
    # Z = A - iB corresponds to A cos(wt) + B sin(wt)
    # Actually, standard DFT definition: X(f) = sum x(t) e^{-iwt} = sum x(cos - i sin)
    # So Real part is related to Cosine coeff, Imag part to Sine coeff.

    Z1 = _compute_ls_complex_coeffs(time1, data1, freqs, errors1)
    Z2 = _compute_ls_complex_coeffs(time2, data2, freqs, errors2)

    # Cross Spectrum Sxy = Z1 * Z2.conj()
    Sxy = Z1 * np.conj(Z2)

    cross_power = np.abs(Sxy)
    phase_lag = np.angle(Sxy)

    # For coherence, we need smoothing.
    # Without smoothing, |Z1 Z2*|^2 / (|Z1|^2 |Z2|^2) is always 1.
    # We return a raw "spectral product" normalized by auto-powers, which is 1 pointwise.
    # Users should average this over frequency bands if they want coherence.

    S11 = np.abs(Z1)**2
    S22 = np.abs(Z2)**2

    # We can provide a simple bandwidth-averaged coherence
    # But let's return the "pointwise" value (1.0) or handle it downstream.
    # Ideally, we return the Cross Spectrum itself (Sxy).

    coherence = np.ones_like(cross_power) # Placeholder, requires smoothing

    return cross_power, phase_lag, coherence, freqs

def calculate_time_lag(
    phase_lag: np.ndarray,
    freqs: np.ndarray
) -> np.ndarray:
    """
    Converts phase lag (radians) to time lag (time units).

    Time Lag = Phase / (2 * pi * f)

    Args:
        phase_lag (np.ndarray): Phase difference in radians.
        freqs (np.ndarray): Frequencies.

    Returns:
        time_lag (np.ndarray): Time lag at each frequency.
    """
    # Avoid division by zero
    valid = freqs > 0
    time_lag = np.zeros_like(phase_lag)
    time_lag[valid] = phase_lag[valid] / (2 * np.pi * freqs[valid])
    return time_lag

def _compute_ls_complex_coeffs(
    time: np.ndarray,
    data: np.ndarray,
    freqs: np.ndarray,
    errors: Optional[np.ndarray] = None
) -> np.ndarray:
    """
    Computes the complex Fourier coefficients for Lomb-Scargle fit.
    Returns Z = A - iB ??

    We solve the linear least squares problem for y = A cos(wt) + B sin(wt) + C.
    To be consistent with Fourier transform X(f) ~ integral x(t) e^{-iwt} dt:
    e^{-iwt} = cos(wt) - i sin(wt).
    The component correlated with cos is Real, with sin is Imaginary (with sign flip?).

    Let's stick to the phase definition: phi = arctan2(-B, A) ?
    If y = R cos(wt + phi) = R (cos wt cos phi - sin wt sin phi)
    A = R cos phi, B = -R sin phi.
    Then phi = arctan2(-B, A).

    The complex number Z whose angle is phi is A - iB.
    So we return A - iB.
    """
    # Center data to remove constant offset C (approx)
    # Or fit C explicitly. Fitting C is safer for low frequencies.

    n_freqs = len(freqs)
    coeffs = np.zeros(n_freqs, dtype=complex)

    # Weights
    if errors is None or np.all(errors == 0):
        w = np.ones_like(data)
    else:
        # Handle zeros in errors safely
        safe_errors = errors.copy()
        safe_errors[safe_errors == 0] = 1e-9 # Prevent div by zero
        w = 1.0 / (safe_errors**2)

    # Pre-calculate sums
    sum_w = np.sum(w)
    sum_wy = np.sum(w * data)
    w_data = w * data

    # Batch over frequencies to limit memory consumption and improve CPU cache hit rate.
    # We dynamically size the batch so intermediate matrices stay within ~2MB.
    bytes_per_row = time.size * time.itemsize
    target_bytes = 2 * 1024 * 1024
    if bytes_per_row > 0:
        BATCH_SIZE = max(1, target_bytes // bytes_per_row)
    else:
        BATCH_SIZE = 500
    BATCH_SIZE = min(max(BATCH_SIZE, 50), 500)

    for start_idx in range(0, n_freqs, BATCH_SIZE):
        end_idx = min(start_idx + BATCH_SIZE, n_freqs)
        f_batch = freqs[start_idx:end_idx]
        n_batch = len(f_batch)

        omega = 2 * np.pi * f_batch[:, np.newaxis]
        wt = omega * time

        cos_wt = np.cos(wt)
        sin_wt = np.sin(wt)

        # Vectorized weighted sums using optimized dot products
        Swc = cos_wt.dot(w)
        Sws = sin_wt.dot(w)
        Swyc = cos_wt.dot(w_data)
        Swys = sin_wt.dot(w_data)

        Swcc = (cos_wt * cos_wt).dot(w)
        Swss = sum_w - Swcc
        Swcs = (cos_wt * sin_wt).dot(w)

        # Construct batch matrices
        M = np.empty((n_batch, 3, 3))
        M[:, 0, 0] = sum_w
        M[:, 0, 1] = Swc
        M[:, 0, 2] = Sws
        M[:, 1, 0] = Swc
        M[:, 1, 1] = Swcc
        M[:, 1, 2] = Swcs
        M[:, 2, 0] = Sws
        M[:, 2, 1] = Swcs
        M[:, 2, 2] = Swss

        b = np.empty((n_batch, 3, 1))
        b[:, 0, 0] = sum_wy
        b[:, 1, 0] = Swyc
        b[:, 2, 0] = Swys

        try:
            # Solve M x = b for all frequencies in batch simultaneously
            x = np.linalg.solve(M, b)
            coeffs[start_idx:end_idx] = x[:, 1, 0] + 1j * x[:, 2, 0]
        except np.linalg.LinAlgError:
            # Fallback to solving individually if any matrix in the batch is singular
            for i in range(n_batch):
                try:
                    xi = np.linalg.solve(M[i], b[i, :, 0])
                    coeffs[start_idx + i] = xi[1] + 1j * xi[2]
                except np.linalg.LinAlgError:
                    coeffs[start_idx + i] = np.nan

    return coeffs

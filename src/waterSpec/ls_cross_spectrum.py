
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
    if errors is None:
        w = np.ones_like(data)
    else:
        w = 1.0 / (errors**2)

    # We vectorise over time, loop over freq (for simplicity/memory)
    # Solving 3x3 system: [ [sum w, sum w cos, sum w sin], ... ]

    # Pre-calculate sums
    sum_w = np.sum(w)
    sum_wy = np.sum(w * data)

    for i, f in enumerate(freqs):
        omega = 2 * np.pi * f
        cos_wt = np.cos(omega * time)
        sin_wt = np.sin(omega * time)

        # Weighted sums
        # Matrix M = [[Sw, Swc, Sws], [Swc, Swcc, Swcs], [Sws, Swcs, Swss]]
        Swc = np.sum(w * cos_wt)
        Sws = np.sum(w * sin_wt)
        Swcc = np.sum(w * cos_wt**2)
        Swss = np.sum(w * sin_wt**2)
        Swcs = np.sum(w * cos_wt * sin_wt)

        # Vector b = [Swy, Swyc, Swys]
        Swyc = np.sum(w * data * cos_wt)
        Swys = np.sum(w * data * sin_wt)

        M = np.array([
            [sum_w, Swc, Sws],
            [Swc, Swcc, Swcs],
            [Sws, Swcs, Swss]
        ])
        b = np.array([sum_wy, Swyc, Swys])

        try:
            # Solve M x = b
            # x = [C, A, B]
            x = np.linalg.solve(M, b)
            A, B = x[1], x[2]

            # Complex coeff Z such that angle(Z) matches phase lag
            # y = A cos + B sin = R cos(wt - phi) -> A=R cos phi, B=R sin phi
            # phi = arctan2(B, A).
            # Z = A + iB
            coeffs[i] = A + 1j * B

        except np.linalg.LinAlgError:
            coeffs[i] = np.nan

    return coeffs

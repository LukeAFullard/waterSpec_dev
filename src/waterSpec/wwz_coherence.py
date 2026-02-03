
import numpy as np
from typing import Tuple, Optional
from scipy.ndimage import gaussian_filter

def calculate_wwz_coherence(
    time1: np.ndarray,
    data1: np.ndarray,
    time2: np.ndarray,
    data2: np.ndarray,
    freqs: np.ndarray,
    taus: Optional[np.ndarray] = None,
    decay_constant: float = 0.00125,
    smoothing_window: float = 2.0
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Calculates the Weighted Wavelet Z-transform (WWZ) Coherence between two time series.

    This method estimates the time-localized correlation between two irregularly sampled
    time series. It computes the complex wavelet coefficients for each series and then
    the magnitude squared coherence.

    Coherence requires smoothing in time/frequency to provide meaningful values (otherwise it is 1).
    Here we apply Gaussian smoothing along the time axis.

    Args:
        time1 (np.ndarray): Time array for first series.
        data1 (np.ndarray): Data array for first series.
        time2 (np.ndarray): Time array for second series.
        data2 (np.ndarray): Data array for second series.
        freqs (np.ndarray): Array of frequencies to analyze.
        taus (np.ndarray, optional): Array of time shifts (epochs). Defaults to linspace over shared range.
        decay_constant (float): Decay constant 'c' for the Morlet wavelet.
        smoothing_window (float): Standard deviation (in indices) for the Gaussian smoothing kernel applied to the power/cross-spectra.

    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray]:
            - coherence: 2D array of Magnitude Squared Coherence (freqs x taus). Values in [0, 1].
            - freqs: Frequencies used.
            - taus: Time shifts used.
    """
    # 1. Define shared time grid (taus) if not provided
    if taus is None:
        t_min = max(time1.min(), time2.min())
        t_max = min(time1.max(), time2.max())
        # Use average density to determine steps? Or just 100 points?
        # Let's use 200 points as a safe default for a map
        taus = np.linspace(t_min, t_max, 200)

    n_freqs = len(freqs)
    n_taus = len(taus)

    # 2. Compute Complex Wavelet Coefficients for both series
    # We need a helper to get the complex coefficients W = A - iB (or similar representation)
    # The standard WWZ returns Power (Z). We need the actual projection coefficients.

    Wx = _compute_complex_wwz(time1, data1, freqs, taus, decay_constant)
    Wy = _compute_complex_wwz(time2, data2, freqs, taus, decay_constant)

    # 3. Compute Spectra
    # Auto-spectra (Power)
    Sxx = np.abs(Wx)**2
    Syy = np.abs(Wy)**2

    # Cross-spectrum
    Sxy = Wx * np.conj(Wy)

    # 4. Smoothing
    # Coherence = |<Sxy>|^2 / (<Sxx> * <Syy>)
    # We apply smoothing convolution along the time axis (axis 1)

    # Check for NaNs
    Sxx = np.nan_to_num(Sxx)
    Syy = np.nan_to_num(Syy)
    Sxy = np.nan_to_num(Sxy)

    smooth_Sxx = gaussian_filter(Sxx, sigma=[0, smoothing_window])
    smooth_Syy = gaussian_filter(Syy, sigma=[0, smoothing_window])
    smooth_Sxy_real = gaussian_filter(Sxy.real, sigma=[0, smoothing_window])
    smooth_Sxy_imag = gaussian_filter(Sxy.imag, sigma=[0, smoothing_window])
    smooth_Sxy = smooth_Sxy_real + 1j * smooth_Sxy_imag

    # 5. Coherence
    numerator = np.abs(smooth_Sxy)**2
    denominator = smooth_Sxx * smooth_Syy

    # Avoid division by zero
    valid = denominator > 1e-12
    coherence = np.zeros_like(numerator)
    coherence[valid] = numerator[valid] / denominator[valid]

    # Clip to [0, 1] just in case of float errors
    coherence = np.clip(coherence, 0.0, 1.0)

    return coherence, freqs, taus

def _compute_complex_wwz(time, data, freqs, taus, decay_constant):
    """
    Computes complex wavelet coefficients for WWZ.
    Returns array of shape (n_freqs, n_taus).
    """
    # This largely duplicates logic from wwz.py but returns coefficients
    # W ~ (c1 - i*c2) ??
    # Let's define the complex coefficient as the amplitude of the fit:
    # f(t) ~ A cos + B sin.
    # Complex representation: Z = A - iB (so that Z * e^{iwt} = (A-iB)(cos+isin) = Acos + Bsin + i(...))

    delta_t = time[None, :] - taus[:, None] # (n_taus, n_points)
    delta_t_sq = delta_t ** 2
    data_b = data[None, :]

    coeffs_matrix = np.zeros((len(freqs), len(taus)), dtype=complex)

    for i, freq in enumerate(freqs):
        omega = 2 * np.pi * freq
        z = decay_constant * (omega ** 2)
        weights = np.exp(-1.0 * z * delta_t_sq)

        sum_w = np.sum(weights, axis=1)
        valid_mask = sum_w > 1e-9
        if not np.any(valid_mask): continue

        phi2 = np.cos(omega * delta_t)
        phi3 = np.sin(omega * delta_t)

        # Weighted sums
        S_w = sum_w
        S_c = np.sum(weights * phi2, axis=1)
        S_s = np.sum(weights * phi3, axis=1)
        S_cc = np.sum(weights * phi2 * phi2, axis=1)
        S_ss = np.sum(weights * phi3 * phi3, axis=1)
        S_cs = np.sum(weights * phi2 * phi3, axis=1)
        S_y = np.sum(weights * data_b, axis=1)
        S_yc = np.sum(weights * data_b * phi2, axis=1)
        S_ys = np.sum(weights * data_b * phi3, axis=1)

        # Build 3x3 systems (Constant, Cos, Sin)
        # M = [[Sw, Sc, Ss], [Sc, Scc, Scs], [Ss, Scs, Sss]]
        # B = [Sy, Syc, Sys]

        M = np.zeros((len(taus), 3, 3))
        M[:, 0, 0] = S_w
        M[:, 0, 1] = S_c; M[:, 1, 0] = S_c
        M[:, 0, 2] = S_s; M[:, 2, 0] = S_s
        M[:, 1, 1] = S_cc
        M[:, 1, 2] = S_cs; M[:, 2, 1] = S_cs
        M[:, 2, 2] = S_ss

        B = np.zeros((len(taus), 3))
        B[:, 0] = S_y
        B[:, 1] = S_yc
        B[:, 2] = S_ys

        # Solve
        det = np.linalg.det(M)
        valid_matrix = np.abs(det) > 1e-12
        mask = valid_mask & valid_matrix

        if np.any(mask):
            try:
                sol = np.linalg.solve(M[mask], B[mask][..., None])
                # sol shape: (n_valid, 3, 1)
                # Coeffs: [offset, A (cos), B (sin)]
                A = sol[:, 1, 0]
                B_sin = sol[:, 2, 0]

                # Complex coeff Z = A - iB
                coeffs_matrix[i, mask] = A - 1j * B_sin
            except np.linalg.LinAlgError:
                pass

    return coeffs_matrix

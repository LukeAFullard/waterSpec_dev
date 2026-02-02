import numpy as np
from scipy.stats import f as f_dist
from typing import Tuple, Optional, Union

def calculate_wwz(
    time: np.ndarray,
    data: np.ndarray,
    freqs: np.ndarray,
    taus: Optional[np.ndarray] = None,
    decay_constant: float = 0.00125,
    parallel: bool = False # Placeholder for future optimization
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Calculates the Weighted Wavelet Z-transform (WWZ) for unevenly sampled data.

    Based on Foster (1996), "Wavelets for period analysis of unevenly sampled time series".

    The WWZ statistic follows an F-distribution with (N_eff - 3, 2) degrees of freedom,
    providing a robust measure of local spectral power.

    Args:
        time (np.ndarray): Array of time points (irregularly sampled).
        data (np.ndarray): Array of data values.
        freqs (np.ndarray): Array of frequencies to analyze.
        taus (np.ndarray, optional): Array of time shifts (epochs) to center the wavelet.
            If None, uses the input `time` array (or a regular grid spanning it).
            Defaults to None.
        decay_constant (float, optional): The decay constant 'c' for the Morlet wavelet window.
            Controls the trade-off between time and frequency resolution.
            Standard value is often c = 1/(2*w^2) where w is width.
            Foster suggests c = 1/(8*pi^2) ~= 0.0125 for broad search,
            or smaller values for better frequency resolution.
            Defaults to 0.00125.

    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray]:
            - wwz_matrix: 2D array of WWZ power (shape: len(freqs) x len(taus)).
            - freqs: The frequency array used.
            - taus: The time shift array used.
    """
    n = len(time)
    if taus is None:
        taus = np.linspace(time.min(), time.max(), len(time))

    n_freqs = len(freqs)
    n_taus = len(taus)

    wwz_matrix = np.zeros((n_freqs, n_taus))

    # Vectorize over tau for each frequency
    # We iterate over frequency to keep memory usage manageable.

    # Pre-calculate T and Data grids?
    # Shape of Delta_T will be (n_taus, n_points)
    # Delta_T[j, k] = time[k] - taus[j]
    # This might be large (e.g. 1000x1000 = 1MB). Totally fine.

    # Taus shape: (n_taus, 1)
    # Time shape: (1, n_points)
    delta_t = time[None, :] - taus[:, None] # (n_taus, n_points)
    delta_t_sq = delta_t ** 2

    # Data shape: (1, n_points)
    # Broadcast data for weighted sums
    data_b = data[None, :]
    data_sq_b = (data**2)[None, :]

    for i, freq in enumerate(freqs):
        omega = 2 * np.pi * freq
        z = decay_constant * (omega ** 2)

        # Calculate weights for all taus at once
        # weights shape: (n_taus, n_points)
        weights = np.exp(-1.0 * z * delta_t_sq)

        # Check for essentially zero weights sum
        sum_w = np.sum(weights, axis=1) # (n_taus,)
        valid_mask = sum_w > 1e-9

        if not np.any(valid_mask):
            continue

        # Filter valid calculations to avoid division by zero
        # We perform calculations only on valid indices or mask later
        # Let's perform on all and mask result

        # Basis functions
        # phi2 = cos, phi3 = sin
        # shape: (n_taus, n_points)
        phi2 = np.cos(omega * delta_t)
        phi3 = np.sin(omega * delta_t)

        # Weighted sums (vectorized over n_points axis=1)
        # resulting shape: (n_taus,)

        S_w = sum_w
        S_c   = np.sum(weights * phi2, axis=1)
        S_s   = np.sum(weights * phi3, axis=1)
        S_cc  = np.sum(weights * phi2 * phi2, axis=1)
        S_ss  = np.sum(weights * phi3 * phi3, axis=1)
        S_cs  = np.sum(weights * phi2 * phi3, axis=1)
        S_y   = np.sum(weights * data_b, axis=1)
        S_yc  = np.sum(weights * data_b * phi2, axis=1)
        S_ys  = np.sum(weights * data_b * phi3, axis=1)
        S_yy  = np.sum(weights * data_sq_b, axis=1)

        # Calculate Weighted Variation (TSS)
        # TSS = S_yy - (S_y^2 / S_w)
        TSS = S_yy - (S_y**2 / S_w)

        # 3x3 Matrix Inversion (Analytic or solve small systems)
        # Since we have N_taus independent 3x3 systems, we can use
        # numpy.linalg.solve with broadcasting if shape is (N_taus, 3, 3)

        # Construct M stack: (n_taus, 3, 3)
        M = np.zeros((n_taus, 3, 3))
        M[:, 0, 0] = S_w
        M[:, 0, 1] = S_c; M[:, 1, 0] = S_c
        M[:, 0, 2] = S_s; M[:, 2, 0] = S_s
        M[:, 1, 1] = S_cc
        M[:, 1, 2] = S_cs; M[:, 2, 1] = S_cs
        M[:, 2, 2] = S_ss

        # Construct B stack: (n_taus, 3)
        B = np.zeros((n_taus, 3))
        B[:, 0] = S_y
        B[:, 1] = S_yc
        B[:, 2] = S_ys

        # Solve M * x = B for x (coeffs)
        # Only for valid masks
        coeffs = np.zeros((n_taus, 3))

        # Using linalg.solve on the whole stack
        # Handle singular matrices by catching? Or check det first.
        # Check determinanats
        det = np.linalg.det(M)
        valid_matrix_mask = np.abs(det) > 1e-12

        final_mask = valid_mask & valid_matrix_mask

        if np.any(final_mask):
            try:
                # Reshape B to (k, 3, 1) to ensure it's treated as column vectors
                # This prevents ambiguity in broadcasting rules for stacked solve
                B_masked = B[final_mask][..., None]
                M_masked = M[final_mask]

                sol = np.linalg.solve(M_masked, B_masked)

                # Squeeze back to (k, 3)
                coeffs[final_mask] = sol[..., 0]
            except np.linalg.LinAlgError:
                # Fallback or just ignore (zeros)
                pass

        # Model SS
        # Power = (coeffs dot B) - (S_y^2 / S_w)
        # Dot product for each tau: sum(coeffs * B, axis=1)
        term1 = np.sum(coeffs * B, axis=1)
        model_SS = term1 - (S_y**2 / S_w)

        # N_eff
        sum_w2 = np.sum(weights**2, axis=1)
        n_eff = (S_w**2) / sum_w2

        # Residual SS
        residual_SS = TSS - model_SS

        # Calculate Z
        # Z = (N_eff - 3)/2 * (Model_SS / Residual_SS)

        z_scores = np.zeros(n_taus)

        # Case 1: Normal Residuals
        normal_mask = (residual_SS > 1e-12) & final_mask
        if np.any(normal_mask):
             z_scores[normal_mask] = ((n_eff[normal_mask] - 3.0) / 2.0) * (model_SS[normal_mask] / residual_SS[normal_mask])

        # Case 2: Near-Perfect Fit (Residuals ~ 0, Model >> 0)
        # Assign a high value (e.g. based on machine precision limit)
        perfect_mask = (residual_SS <= 1e-12) & (model_SS > 1e-12) & final_mask
        if np.any(perfect_mask):
             # Represent as infinity as statistical significance is effectively absolute
             z_scores[perfect_mask] = np.inf

        wwz_matrix[i, :] = z_scores

    return wwz_matrix, freqs, taus

def calculate_wwz_statistics(
    wwz_matrix: np.ndarray,
    n_eff: Union[float, np.ndarray],
    n_params: int = 3
) -> np.ndarray:
    """
    Calculates p-values for WWZ statistics based on F-distribution.

    The WWZ statistic roughly follows F(n_params-1, n_eff-n_params).
    Actually, Foster (1996) defines Z as (N_eff - 3)/2 * (Model_SS / Residual_SS).
    This Z corresponds to F * (n_params-1) / 2? No.

    Foster (1996) Eq 4-1: Z = (R_w(omega, tau) / (1 - R_w(omega, tau))) * (N_eff - 3)/2
    Where R_w is weighted variation explained.
    This is equivalent to F-statistic * (k/nu2) * nu2/2 ?

    More simply, standard F-test: F = (Model_SS / (p-1)) / (Residual_SS / (n-p))
    Here p=3 (constant, sin, cos). So df1 = 2, df2 = N_eff - 3.
    F = (Model_SS / 2) / (Residual_SS / (N_eff - 3))
      = (N_eff - 3)/2 * (Model_SS / Residual_SS)

    Wait, that IS exactly the Z definition used in the code!
    So Z = F.

    Therefore, Z ~ F(2, N_eff - 3).

    Args:
        wwz_matrix: The Z-score matrix.
        n_eff: Effective number of points (can be scalar or array same shape as wwz).
               Note: The calculate_wwz function does not currently return n_eff.
               We might need to update it to return n_eff or recalculate it.
               For now, users often approximate N_eff or we need to extract it.

    Returns:
        p_values: Array of p-values (1 - CDF).
    """
    # Degrees of freedom
    df1 = 2
    df2 = n_eff - 3

    # Handle broadcasting if n_eff is array
    # p-value = survival function (1 - cdf)
    p_values = f_dist.sf(wwz_matrix, df1, df2)
    return p_values

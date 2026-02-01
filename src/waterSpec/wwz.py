import numpy as np
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
    # 1. Setup
    n = len(time)
    if taus is None:
        taus = np.linspace(time.min(), time.max(), len(time))

    n_freqs = len(freqs)
    n_taus = len(taus)

    wwz_matrix = np.zeros((n_freqs, n_taus))

    # Pre-calculate data variance for normalization?
    # WWZ is naturally normalized as a Z-statistic (like F-test).

    # 2. Main Loop (Vectorized over time, iterated over freq/tau or freq)
    # To be efficient, we iterate over tau and freq.
    # Fully vectorizing over 3 dimensions (t, tau, freq) might explode memory.
    # Let's loop over Frequency (outer) and Tau (inner), vectorizing the weighted sums over N data points.

    for i, freq in enumerate(freqs):
        omega = 2 * np.pi * freq

        for j, tau in enumerate(taus):
            # Calculate Weights: w_i = exp(-c * omega^2 * (t_i - tau)^2)
            # Note: Foster (1996) uses w = exp(-c * omega^2 * (t - tau)^2)
            # This ensures the window size scales with frequency (wavelet property).

            # Distance from center
            delta_t = time - tau
            z = c_omega2 = decay_constant * (omega ** 2)
            weights = np.exp(-1.0 * z * (delta_t ** 2))

            # Avoid numerical instability if weights are too small
            # (though with exp, they just go to 0, which is fine)

            sum_w = np.sum(weights)
            if sum_w < 1e-9:
                wwz_matrix[i, j] = 0.0
                continue

            # Weighted averages (moment 0)
            V_x = np.sum(weights * data) / sum_w
            V_1 = 1.0 # By definition of weighted average of 1s

            # Define basis functions centered at tau
            # phi1 = 1
            # phi2 = cos(omega(t - tau))
            # phi3 = sin(omega(t - tau))

            # We need weighted projections onto these basis functions.
            # Foster (1996) simplifies this by computing weighted variations.

            # Let's follow the standard projection formulation for Weighted Least Squares.
            # We want to fit y = a + b*cos(...) + c*sin(...)
            # But the basis functions are not orthogonal under the weighted inner product.
            # So we build the Normal Equations matrix.

            phi2 = np.cos(omega * delta_t)
            phi3 = np.sin(omega * delta_t)

            # Weighted sums
            # S_k = sum(w * phi_k)
            # But we can simplify by centering the data?
            # Foster calculates N_eff and weighted variations.

            # Let's use the explicit matrix inversion (3x3) for clarity and correctness.
            # Design matrix A (N x 3): [1, cos, sin]
            # Weight matrix W (diagonal)
            # Solve (A^T W A) beta = A^T W y

            # Construct A^T W A  (3x3 symmetric matrix)
            # Elements: sum(w * phi_k * phi_m)

            S_w   = np.sum(weights)
            S_c   = np.sum(weights * phi2)
            S_s   = np.sum(weights * phi3)
            S_cc  = np.sum(weights * phi2 * phi2)
            S_ss  = np.sum(weights * phi3 * phi3)
            S_cs  = np.sum(weights * phi2 * phi3)
            S_y   = np.sum(weights * data)
            S_yc  = np.sum(weights * data * phi2)
            S_ys  = np.sum(weights * data * phi3)
            S_yy  = np.sum(weights * data * data) # Total weighted power

            # Matrix M = [[S_w, S_c, S_s], [S_c, S_cc, S_cs], [S_s, S_cs, S_ss]]
            # Vector B = [S_y, S_yc, S_ys]

            # However, Foster (1996) provides a cleaner form for the Z-statistic
            # by orthogonalizing.
            # "The Weighted Wavelet Z-transform is defined as the reduction in the weighted
            # variation of the data due to the fitted model function."

            # Weighted Variation of data: V_y = sum(w * y^2) / S_w - (sum(w*y)/S_w)^2
            # Actually simpler: Total Sum of Squares (Weighted)
            # TSS = sum(w * (y - weighted_mean_y)^2)

            weighted_mean = S_y / S_w
            TSS = S_yy - (S_y**2 / S_w) # Weighted Total Sum of Squares

            if TSS <= 0:
                wwz_matrix[i, j] = 0.0
                continue

            # Model Sum of Squares (RSS_model)
            # We solve the 2x2 system for the oscillating terms after projecting out the mean?
            # Or just solve the full 3x3.
            # Let's solve the 3x3 to get the fitted coefficients [a, b, c].
            # fitted_y = a + b*cos + c*sin
            # RSS = sum(w * (y - fitted_y)^2)

            # Z = (N_eff - 3)/2 * (TSS - RSS) / RSS  (This is F-stat form)
            # Foster defines WWZ = (N_eff - 3)/2 * (V_y - V_res) / V_res?
            # Actually Foster defines Z = (N_eff - 1)/2 * V_mod / V_res ?

            # Let's compute the "Power" as the Model Sum of Squares relative to Total.
            # Or simply the Model Sum of Squares (Weighted).

            # Inversion of 3x3 symmetric matrix
            det_M = (S_w * (S_cc * S_ss - S_cs**2) -
                     S_c * (S_c * S_ss - S_cs * S_s) +
                     S_s * (S_c * S_cs - S_cc * S_s))

            if np.abs(det_M) < 1e-12:
                wwz_matrix[i, j] = 0.0
                continue

            # We only need the reduction in variance.
            # Let's use numpy.linalg.solve for stability
            M = np.array([[S_w, S_c, S_s],
                          [S_c, S_cc, S_cs],
                          [S_s, S_cs, S_ss]])
            B = np.array([S_y, S_yc, S_ys])

            try:
                coeffs = np.linalg.solve(M, B) # [a, b, c]
            except np.linalg.LinAlgError:
                 wwz_matrix[i, j] = 0.0
                 continue

            # RSS (Weighted Residual Sum of Squares)
            # RSS = sum(w * (y - (a + b*c + c*s))^2)
            # Expand: sum(w*y^2) - 2*coeffs*B + coeffs*M*coeffs
            # = S_yy - coeffs.T @ M @ coeffs?
            # Let's verify:
            # (y - Xb)^T W (y - Xb) = yWy - 2bXWy + bXWXb
            # XTy = B. XWX = M.
            # RSS = S_yy - 2*coeffs.dot(B) + coeffs.dot(M.dot(coeffs))
            # Since M.coeffs = B, then coeffs.dot(M.dot(coeffs)) = coeffs.dot(B).
            # So RSS = S_yy - coeffs.dot(B).

            model_SS = np.dot(coeffs, B) - (S_y**2 / S_w)
            # Note: We subtract the mean component to get the power of the *oscillation*
            # The "Total Sum of Squares" usually refers to variation around the mean.
            # The 3-param model includes the mean (a).
            # The 1-param model (just mean) has SS = S_y^2 / S_w.
            # So the contribution of the oscillation (params b, c) is:
            # Power = (coeffs.dot(B)) - (S_y^2 / S_w)

            # N_eff calculation (Effective number of points in the window)
            # N_eff = (sum w)^2 / sum(w^2)
            sum_w2 = np.sum(weights**2)
            n_eff = (S_w**2) / sum_w2

            # WWZ Statistic (Foster 1996, eq 5.1 - 5.4ish)
            # Z = (N_eff - 3)/2 * Power / (TSS - Power)
            # where TSS - Power = Residual SS

            residual_SS = TSS - model_SS

            if residual_SS <= 1e-12:
                 wwz_matrix[i, j] = np.nan # Perfect fit?
            else:
                 wwz = ((n_eff - 3.0) / 2.0) * (model_SS / residual_SS)
                 wwz_matrix[i, j] = wwz

    return wwz_matrix, freqs, taus

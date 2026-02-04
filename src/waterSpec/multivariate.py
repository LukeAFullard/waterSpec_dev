
import numpy as np
from typing import Dict, List, Optional
from .haar_analysis import _compute_statistic

def calculate_multivariate_fluctuations(
    time: np.ndarray,
    datasets: List[np.ndarray],
    lags: np.ndarray,
    overlap: bool = True,
    overlap_step_fraction: float = 0.1,
    min_samples_per_window: int = 5,
    statistic: str = "mean",
    percentile: Optional[float] = None,
    percentile_method: str = "hazen"
) -> Dict[float, List[np.ndarray]]:
    """
    Calculates Haar fluctuations for multiple aligned time series on the exact same windows.

    Args:
        time (np.ndarray): Shared time array.
        datasets (List[np.ndarray]): List of data arrays (must be same length as time).
        lags (np.ndarray): Array of lag times.
        overlap (bool): Whether to use overlapping windows.
        overlap_step_fraction (float): Fraction of lag to step forward.
        min_samples_per_window (int): Minimum samples required in each half-window.
        statistic (str): Statistic to use ("mean", "median", "percentile").
        percentile (float): Percentile value.
        percentile_method (str): Percentile method.

    Returns:
        Dict mapping lag (float) to a list of fluctuation arrays [fluc_1, fluc_2, ...].
        Each fluc_i is an array of fluctuation values for that variable at that lag.
    """
    results = {}

    # Validate input lengths
    for i, d in enumerate(datasets):
        if len(d) != len(time):
            raise ValueError(f"Dataset {i} length ({len(d)}) does not match time array length ({len(time)}).")

    # Sort by time just in case (though we assume alignment)
    sort_idx = np.argsort(time)
    time = time[sort_idx]
    datasets = [d[sort_idx] for d in datasets]

    t_min, t_max = time[0], time[-1]
    n_vars = len(datasets)

    for tau in lags:
        fluctuations = [[] for _ in range(n_vars)]

        step_size = tau * overlap_step_fraction if overlap else tau
        t_start = t_min

        while t_start + tau <= time[-1]: # Use time[-1] instead of t_max for precision
            t_mid = t_start + tau / 2
            t_end = t_start + tau

            # Indices in the ALIGNED arrays
            idx_start = np.searchsorted(time, t_start, side='left')
            idx_mid = np.searchsorted(time, t_mid, side='left')
            idx_end = np.searchsorted(time, t_end, side='left')

            # Check if all variables have sufficient data in this window
            valid_window = True
            diffs = []

            for data in datasets:
                v_left = data[idx_start:idx_mid]
                v_right = data[idx_mid:idx_end]

                if len(v_left) < min_samples_per_window or len(v_right) < min_samples_per_window:
                    valid_window = False
                    break

                # We want absolute fluctuation?
                # No, for correlation we want signed fluctuation.

                val_right = _compute_statistic(v_right, statistic, percentile, percentile_method)
                val_left = _compute_statistic(v_left, statistic, percentile, percentile_method)
                d = val_right - val_left

                diffs.append(d)

            if valid_window:
                for i in range(n_vars):
                    fluctuations[i].append(diffs[i])

            if overlap:
                t_start += step_size
            else:
                t_start = t_end
                if t_start >= time[-1]: break

        # Convert to numpy arrays
        results[tau] = [np.array(f) for f in fluctuations]

    return results

def calculate_partial_cross_haar(
    time: np.ndarray,
    data_x: np.ndarray,
    data_y: np.ndarray,
    data_z: np.ndarray,
    lags: np.ndarray,
    overlap: bool = True,
    overlap_step_fraction: float = 0.1,
    min_samples_per_window: int = 5,
    statistic: str = "mean",
    percentile: Optional[float] = None,
    percentile_method: str = "hazen"
) -> Dict:
    """
    Calculates Partial Cross-Haar Correlation between X and Y controlling for Z.

    WARNING: This method is experimental. Applying partial correlation to Haar fluctuations
    assumes that the fluctuations at a given scale follow a multivariate Gaussian distribution
    and that the linear partial correlation formula is valid for these increments.
    This assumption has not been rigorously validated in the literature for all environmental
    processes. Use with caution.

    Formula: rho_{XY.Z} = (rho_{XY} - rho_{XZ} * rho_{YZ}) / sqrt((1 - rho_{XZ}^2) * (1 - rho_{YZ}^2))

    Args:
        time (np.ndarray): Shared time array.
        data_x (np.ndarray): Variable X.
        data_y (np.ndarray): Variable Y.
        data_z (np.ndarray): Control variable Z.
        lags (np.ndarray): Array of lag times.
        ...

    Returns:
        Dict containing arrays for lags, rho_xy, rho_xz, rho_yz, partial_corr, n_pairs.
    """
    import warnings
    warnings.warn(
        "calculate_partial_cross_haar is experimental and its statistical validity "
        "for Haar fluctuations has not been fully established. Interpret results with caution.",
        UserWarning
    )

    # Calculate fluctuations
    fluc_dict = calculate_multivariate_fluctuations(
        time, [data_x, data_y, data_z], lags, overlap, overlap_step_fraction, min_samples_per_window,
        statistic, percentile, percentile_method
    )

    results = {
        'lags': [],
        'rho_xy': [],
        'rho_xz': [],
        'rho_yz': [],
        'partial_corr': [],
        'n_pairs': []
    }

    for tau in lags:
        flucs = fluc_dict[tau]
        fx, fy, fz = flucs[0], flucs[1], flucs[2]

        n = len(fx)
        if n < 3:
            results['lags'].append(tau)
            results['rho_xy'].append(np.nan)
            results['rho_xz'].append(np.nan)
            results['rho_yz'].append(np.nan)
            results['partial_corr'].append(np.nan)
            results['n_pairs'].append(n)
            continue

        # Pearson correlations
        r_xy = np.corrcoef(fx, fy)[0, 1]
        r_xz = np.corrcoef(fx, fz)[0, 1]
        r_yz = np.corrcoef(fy, fz)[0, 1]

        # Partial correlation
        denom_sq = (1 - r_xz**2) * (1 - r_yz**2)

        if denom_sq <= 0:
            # Should not happen theoretically unless r=1, but float errors
            p_corr = np.nan
        else:
            denom = np.sqrt(denom_sq)
            p_corr = (r_xy - r_xz * r_yz) / denom

        results['lags'].append(tau)
        results['rho_xy'].append(r_xy)
        results['rho_xz'].append(r_xz)
        results['rho_yz'].append(r_yz)
        results['partial_corr'].append(p_corr)
        results['n_pairs'].append(n)

    # Convert lists to arrays
    for key in results:
        results[key] = np.array(results[key])

    return results

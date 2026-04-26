"""
Automatic and manual changepoint detection for time series segmentation.
"""

import warnings
import numpy as np
from typing import Optional, Tuple, Dict

_RUPTURES_MISSING_MSG = (
    "The 'ruptures' package is required for automatic changepoint detection. "
    "Install it with 'pip install ruptures'."
)
try:
    import ruptures as rpt
except ImportError:
    rpt = None

def detect_changepoint_pelt(
    time: np.ndarray,
    data: np.ndarray,
    model: str = "rbf",  # "rbf", "l2", "l1", "normal", "ar"
    penalty: float = None,
    min_size: int = None,
    jump: int = 1,
) -> Optional[int]:
    """
    Detects a single changepoint using PELT algorithm.

    Args:
        time: Numeric time array
        data: Data values (should already be preprocessed)
        model: Cost function for detection
            - "rbf": Detects mean/variance changes (recommended default)
            - "l2": Mean shifts only
            - "normal": Assumes Gaussian data
            - "ar": For autoregressive structure
        penalty: Penalty value (higher = fewer changepoints)
            If None, uses 2*log(n) (BIC-like criterion)
        min_size: Minimum samples between changepoints
            If None, uses n/10 or 20, whichever is larger
        jump: Subsample factor for speed (1 = no subsampling)

    Returns:
        Index of changepoint in original data, or None if no significant changepoint
    """
    if rpt is None:
        raise ImportError(_RUPTURES_MISSING_MSG)

    n = len(data)

    # Set intelligent defaults
    if min_size is None:
        min_size = max(20, n // 10)  # At least 20 points or 10% of series

    if penalty is None:
        penalty = 2 * np.log(n)  # BIC-like penalty

    # Validate we have enough data
    if n < 2 * min_size:
        raise ValueError(
            f"Time series too short (n={n}) for changepoint detection with "
            f"min_size={min_size}. Minimum required: {2*min_size}."
        )

    # Reshape data for ruptures (requires 2D: samples × features)
    signal = data.reshape(-1, 1)

    # Create and fit model
    algo = rpt.Pelt(model=model, min_size=min_size, jump=jump)
    algo.fit(signal)

    # Predict changepoints using the penalty value.
    # This finds the optimal number of changepoints for the given penalty.
    # The result includes the end of the series, so a single changepoint
    # will result in a list of two indices: [changepoint_index, n].
    result = algo.predict(pen=penalty)

    # Extract changepoint (exclude the terminal index)
    changepoints = [cp for cp in result if cp < n]

    if len(changepoints) == 0:
        return None

    # Return the single changepoint index
    return changepoints[0]


def get_changepoint_time(
    changepoint_idx: int,
    time: np.ndarray,
    time_unit: str = "seconds",
) -> str:
    """
    Converts a changepoint index to a human-readable time description.
    """
    cp_time = time[changepoint_idx]

    # Convert to days for readability
    if time_unit == "seconds":
        cp_days = cp_time / 86400
    elif time_unit == "hours":
        cp_days = cp_time / 24
    else:  # days
        cp_days = cp_time

    # Format based on magnitude
    if cp_days < 2:
        return f"{cp_days * 24:.1f} hours"
    elif cp_days < 60:
        return f"{cp_days:.1f} days"
    elif cp_days < 730:
        return f"{cp_days / 30.44:.1f} months"
    else:
        return f"{cp_days / 365.25:.1f} years"
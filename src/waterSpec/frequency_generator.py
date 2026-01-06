"""
This module provides functions for generating frequency grids for spectral analysis.
"""

import numpy as np
import warnings
from typing import Optional

# Define a constant for the coefficient of variation (CV) threshold.
# If the CV of sampling intervals exceeds this value, the sampling is
# considered highly irregular, and a warning is issued. A high CV
# means the standard deviation of intervals is large relative to the median
# interval, making the concept of a single Nyquist frequency less reliable.
CV_THRESHOLD_FOR_IRREGULAR_SAMPLING = 0.5


def generate_frequency_grid(
    time_numeric: np.ndarray,
    num_points: int = 200,
    grid_type: str = "log",
    max_freq: Optional[float] = None,
    nyquist_factor: float = 1.0,
    time_unit: str = "seconds",
):
    """
    .. deprecated:: 0.2.0
        This function is deprecated and will be removed in a future version.
        Use the `autopower` method of an `astropy.timeseries.LombScargle`
        object for robust, automatic frequency grid generation.

    Generates a frequency grid, with options for linear or log spacing, and
    flexible upper frequency limits.

    Args:
        time_numeric (np.ndarray): The numeric time array, in units specified
            by the `time_unit` parameter.
        num_points (int, optional): The number of frequency points to generate.
            Defaults to 200.
        grid_type (str, optional): The type of grid to generate ('log' or 'linear').
            Defaults to 'log'.
        max_freq (Optional[float], optional): A specific maximum frequency to use
            for the grid. If provided, this overrides the automatic Nyquist
            frequency calculation. Units must be consistent with `1 / time_unit`.
            Defaults to None.
        nyquist_factor (float, optional): A scaling factor to apply to the
            heuristic Nyquist frequency (0.5 / median interval). For example,
            a factor of 0.8 would set the maximum frequency to 80% of the
            heuristic Nyquist. This is ignored if `max_freq` is set.
            Defaults to 1.0.
        time_unit (str, optional): The unit of the input `time_numeric` array.
            This is used for documentation and to ensure output units are
            understood. It does not perform any conversion. Common values are
            'seconds', 'hours', 'days'. Defaults to 'seconds'.

    Returns:
        np.ndarray: The frequency grid, in units of `1 / time_unit`.
    """
    # This function is retained for backward compatibility but is no longer
    # recommended for rigorous analysis. Users should prefer `autopower` from
    # astropy.timeseries.LombScargle.

    # --- Input Validation ---
    if not isinstance(num_points, int) or num_points <= 1:
        raise ValueError("num_points must be an integer greater than 1.")
    if not isinstance(nyquist_factor, (int, float)) or nyquist_factor <= 0:
        raise ValueError("nyquist_factor must be a positive number.")
    if max_freq is not None and (not isinstance(max_freq, (int, float)) or max_freq <= 0):
        raise ValueError("max_freq, if provided, must be a positive number.")
    if time_numeric.size < 2:
        raise ValueError(
            "At least two data points are required to generate a frequency grid."
        )

    duration = np.max(time_numeric) - np.min(time_numeric)
    if duration <= 0:
        raise ValueError(
            "Time series duration must be positive to generate a frequency grid."
        )

    # The minimum frequency is determined by the total duration of the series.
    min_freq = 1 / duration

    if max_freq is not None:
        # User has provided a hard override for the maximum frequency.
        max_f = max_freq
    else:
        # For irregularly sampled data, the Nyquist frequency is not strictly defined.
        # A common and robust heuristic is to use the median sampling interval,
        # as it is less sensitive to outliers (i.e., large gaps) than the mean.
        sampling_intervals = np.diff(time_numeric)
        median_sampling_interval = np.median(sampling_intervals)
        if median_sampling_interval <= 0:
            raise ValueError("Median sampling interval must be positive.")

        # Check for highly irregular sampling, which can make the Nyquist frequency
        # misleading. A high coefficient of variation suggests this.
        interval_cv = np.std(sampling_intervals) / median_sampling_interval
        if interval_cv > CV_THRESHOLD_FOR_IRREGULAR_SAMPLING:
            warnings.warn(
                f"The time series has highly irregular sampling (CV of intervals = {interval_cv:.2f}). "
                "The concept of a single Nyquist frequency may be misleading. "
                "Interpret high-frequency results with caution, as aliasing effects may be complex. "
                f"The output frequency units are 1/{time_unit}.",
                UserWarning,
            )

        # Calculate the maximum frequency using the heuristic and the user-provided factor.
        heuristic_nyquist = 0.5 / median_sampling_interval
        max_f = nyquist_factor * heuristic_nyquist

    # The minimum frequency must be less than the maximum frequency to define
    # a valid range for the grid. If it's not, the time series is too short
    # or has too few points for a meaningful analysis.
    if min_freq >= max_f:
        raise ValueError(
            f"The calculated minimum frequency ({min_freq:.2g}) is not less than the "
            f"maximum frequency ({max_f:.2g}). This can happen if the time series "
            "duration is too short or the sampling rate is too low. "
            "Consider adjusting `nyquist_factor` or providing `max_freq`."
        )

    if grid_type == "log":
        frequency_grid = np.geomspace(min_freq, max_f, num=num_points)
    elif grid_type == "linear":
        frequency_grid = np.linspace(min_freq, max_f, num=num_points)
    else:
        raise ValueError("grid_type must be either 'log' or 'linear'")

    return frequency_grid

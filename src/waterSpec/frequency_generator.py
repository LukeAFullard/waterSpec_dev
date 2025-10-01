"""
This module provides functions for generating frequency grids for spectral analysis.
"""

import numpy as np
import warnings


def generate_frequency_grid(time_numeric, num_points=200, grid_type="log"):
    """
    Generates a frequency grid, either logarithmically or linearly spaced.

    A log-spaced grid is often a good choice for spectral slope analysis, while
    a linear grid is better for resolving specific peaks in a periodogram.

    Important: The units of the frequency grid are determined by the units of
    the input `time_numeric` array. If the time is in days, the resulting
    frequency will be in cycles per day (1/day).

    Args:
        time_numeric (np.ndarray): The numeric time array. Its units dictate
            the units of the output frequency grid.
        num_points (int, optional): The number of frequency points to generate.
                                    Defaults to 200.
        grid_type (str, optional): The type of grid to generate ('log' or 'linear').
                                   Defaults to 'log'.

    Returns:
        np.ndarray: The frequency grid, in units of 1/[time_units].
    """
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

    # For irregularly sampled data, the Nyquist frequency is not strictly defined.
    # A common and robust heuristic is to use the median sampling interval,
    # as it is less sensitive to outliers (i.e., large gaps) than the mean.
    sampling_intervals = np.diff(time_numeric)
    median_sampling_interval = np.median(sampling_intervals)
    if median_sampling_interval <= 0:
        raise ValueError("Median sampling interval must be positive.")

    # Check for highly irregular sampling, which can make the Nyquist frequency
    # misleading. A high coefficient of variation suggests this.
    # A threshold of 0.5 means the std dev is 50% of the median interval.
    interval_cv = np.std(sampling_intervals) / median_sampling_interval
    if interval_cv > 0.5:
        warnings.warn(
            f"The time series has highly irregular sampling (CV of intervals = {interval_cv:.2f}). "
            "The concept of a single Nyquist frequency may be misleading. "
            "Interpret high-frequency results with caution, as aliasing effects may be complex.",
            UserWarning,
        )

    nyquist_freq = 0.5 / median_sampling_interval

    # The minimum frequency must be less than the Nyquist frequency to define
    # a valid range for the grid. If it's not, the time series is too short
    # or has too few points for a meaningful analysis.
    if min_freq >= nyquist_freq:
        raise ValueError(
            "The time series duration is too short relative to the median "
            "sampling interval. A meaningful frequency grid cannot be generated."
        )

    if grid_type == "log":
        frequency_grid = np.logspace(
            np.log10(min_freq), np.log10(nyquist_freq), num=num_points
        )
    elif grid_type == "linear":
        frequency_grid = np.linspace(min_freq, nyquist_freq, num=num_points)
    else:
        raise ValueError("grid_type must be either 'log' or 'linear'")

    return frequency_grid

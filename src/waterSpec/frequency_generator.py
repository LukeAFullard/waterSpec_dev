"""
This module provides functions for generating frequency grids for spectral analysis.
"""

import numpy as np


def generate_frequency_grid(time_numeric, num_points=200, grid_type="log"):
    """
    Generates a frequency grid, either logarithmically or linearly spaced.

    A log-spaced grid is often a good choice for spectral slope analysis, while
    a linear grid is better for resolving specific peaks in a periodogram.

    Args:
        time_numeric (np.ndarray): The numeric time array.
        num_points (int, optional): The number of frequency points to generate.
                                    Defaults to 200.
        grid_type (str, optional): The type of grid to generate ('log' or 'linear').
                                   Defaults to 'log'.

    Returns:
        np.ndarray: The frequency grid.
    """
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
    median_sampling_interval = np.median(np.diff(time_numeric))
    if median_sampling_interval <= 0:
        raise ValueError("Median sampling interval must be positive.")
    nyquist_freq = 0.5 / median_sampling_interval

    # In rare cases (e.g., very short series), min_freq can be >= nyquist_freq.
    # We adjust min_freq downwards to ensure a valid frequency range. This is an
    # arbitrary adjustment but prevents the function from failing.
    if min_freq >= nyquist_freq:
        min_freq = nyquist_freq / 100

    if grid_type == "log":
        frequency_grid = np.logspace(
            np.log10(min_freq), np.log10(nyquist_freq), num=num_points
        )
    elif grid_type == "linear":
        frequency_grid = np.linspace(min_freq, nyquist_freq, num=num_points)
    else:
        raise ValueError("grid_type must be either 'log' or 'linear'")

    return frequency_grid

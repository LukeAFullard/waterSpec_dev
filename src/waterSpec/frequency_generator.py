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
    min_freq = 1 / duration
    nyquist_freq = 0.5 / np.median(np.diff(time_numeric))

    # Ensure min_freq is positive and less than nyquist_freq
    if min_freq <= 0:
        min_freq = 1e-9  # A small positive number
    if min_freq >= nyquist_freq:
        min_freq = nyquist_freq / 100  # Adjust if duration is too short

    if grid_type == "log":
        frequency_grid = np.logspace(
            np.log10(min_freq), np.log10(nyquist_freq), num=num_points
        )
    elif grid_type == "linear":
        frequency_grid = np.linspace(min_freq, nyquist_freq, num=num_points)
    else:
        raise ValueError("grid_type must be either 'log' or 'linear'")

    return frequency_grid

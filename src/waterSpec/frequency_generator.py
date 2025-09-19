"""
This module provides functions for generating frequency grids for spectral analysis.
"""
import numpy as np

def generate_log_spaced_grid(time_numeric, num_points=200):
    """
    Generates a logarithmically spaced frequency grid.

    This is often a good choice for analyzing geophysical time series, which may
    have long-term memory.

    Args:
        time_numeric (np.ndarray): The numeric time array.
        num_points (int, optional): The number of frequency points to generate.
                                    Defaults to 200.

    Returns:
        np.ndarray: The logarithmically spaced frequency grid.
    """
    duration = np.max(time_numeric) - np.min(time_numeric)
    min_freq = 1 / duration
    nyquist_freq = 0.5 / np.median(np.diff(time_numeric))

    # Ensure min_freq is positive and less than nyquist_freq
    if min_freq <= 0:
        min_freq = 1e-6  # A small positive number
    if min_freq >= nyquist_freq:
        min_freq = nyquist_freq / 100  # Adjust if duration is too short

    frequency_grid = np.logspace(np.log10(min_freq), np.log10(nyquist_freq), num=num_points)

    return frequency_grid

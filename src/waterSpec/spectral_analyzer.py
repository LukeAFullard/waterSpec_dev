import numpy as np
from astropy.timeseries import LombScargle

def calculate_periodogram(time, data, normalization='standard', nyquist_factor=5):
    """
    Calculates the Lomb-Scargle periodogram for a time series.

    Args:
        time (np.ndarray): The time array.
        data (np.ndarray): The data values.
        normalization (str, optional): The normalization to use for the periodogram.
                                       Defaults to 'standard'.
        nyquist_factor (int, optional): The factor to use for determining the
                                        maximum frequency. Defaults to 5.

    Returns:
        tuple[np.ndarray, np.ndarray]: A tuple containing two numpy arrays:
                                       - frequency
                                       - power
    """
    ls = LombScargle(time, data)

    # astropy's LombScargle can automatically determine a good frequency grid.
    frequency, power = ls.autopower(normalization=normalization,
                                    nyquist_factor=nyquist_factor)

    return frequency, power

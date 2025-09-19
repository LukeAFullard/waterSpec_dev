import numpy as np
from astropy.timeseries import LombScargle

def calculate_periodogram(time, data, frequency=None, normalization='standard',
                          nyquist_factor=5, samples_per_peak=10):
    """
    Calculates the Lomb-Scargle periodogram for a time series.

    Args:
        time (np.ndarray): The time array.
        data (np.ndarray): The data values.
        frequency (np.ndarray, optional): The frequency grid to use. If not
                                          provided, astropy will determine a
                                          good grid automatically. Defaults to
                                          None.
        normalization (str, optional): The normalization to use for the periodogram.
                                       Defaults to 'standard'.
        nyquist_factor (int, optional): The factor to use for determining the
                                        maximum frequency when frequency is not
                                        provided. Defaults to 5.
        samples_per_peak (int, optional): The number of samples to use per peak
                                          when frequency is not provided.
                                          Defaults to 10.

    Returns:
        tuple[np.ndarray, np.ndarray]: A tuple containing two numpy arrays:
                                       - frequency
                                       - power
    """
    ls = LombScargle(time, data)

    if frequency is not None:
        power = ls.power(frequency, normalization=normalization)
    else:
        # astropy's LombScargle can automatically determine a good frequency grid.
        frequency, power = ls.autopower(normalization=normalization,
                                        nyquist_factor=nyquist_factor,
                                        samples_per_peak=samples_per_peak)

    return frequency, power

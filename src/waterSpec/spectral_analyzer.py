import numpy as np
from astropy.timeseries import LombScargle

def calculate_periodogram(time, data, normalization='standard', n_log_freq=100):
    """
    Calculates the Lomb-Scargle periodogram on a log-spaced frequency grid.

    Args:
        time (np.ndarray): The time array.
        data (np.ndarray): The data values.
        normalization (str, optional): The normalization for the periodogram.
                                       Defaults to 'standard'.
        n_log_freq (int, optional): The number of frequency points to generate
                                    on the log-spaced grid. Defaults to 100.

    Returns:
        tuple[np.ndarray, np.ndarray]: A tuple containing two numpy arrays:
                                       - frequency
                                       - power
    """
    ls = LombScargle(time, data)

    # Determine the frequency range for the log-spaced grid
    t_span = time.max() - time.min()
    dt = np.median(np.diff(time))

    # Minimum frequency is 1/T
    min_freq = 1 / t_span
    # Maximum frequency is the Nyquist frequency
    max_freq = 1 / (2 * dt)

    # Generate a logarithmically spaced frequency grid
    frequency = np.logspace(np.log10(min_freq), np.log10(max_freq), n_log_freq)

    # Calculate the power at the specified frequencies
    power = ls.power(frequency, normalization=normalization)

    return frequency, power

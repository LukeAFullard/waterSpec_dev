import numpy as np
from scipy import stats

def fit_spectrum(frequency, power):
    """
    Fits a line to the power spectrum on a log-log plot to find the spectral exponent (beta).

    Args:
        frequency (np.ndarray): The frequency array from the periodogram.
        power (np.ndarray): The power array from the periodogram.

    Returns:
        dict: A dictionary containing the fit results:
              - 'beta': The spectral exponent (the negative of the slope).
              - 'r_squared': The R-squared value of the fit.
              - 'intercept': The intercept of the log-log regression.
              - 'stderr': The standard error of the estimated slope.
    """
    # Ensure there are no zero or negative values before log-transforming
    # This is important as frequency or power can sometimes be zero.
    valid_indices = (frequency > 0) & (power > 0)
    if np.sum(valid_indices) < 2:
        # Not enough data points to perform a linear regression
        return {
            'beta': np.nan,
            'r_squared': np.nan,
            'intercept': np.nan,
            'stderr': np.nan,
        }

    log_freq = np.log(frequency[valid_indices])
    log_power = np.log(power[valid_indices])

    # Perform linear regression on the log-log data
    lin_reg_result = stats.linregress(log_freq, log_power)

    # The spectral exponent (beta) is the negative of the slope
    beta = -lin_reg_result.slope

    # Store the results in a dictionary
    fit_results = {
        'beta': beta,
        'r_squared': lin_reg_result.rvalue**2,
        'intercept': lin_reg_result.intercept,
        'stderr': lin_reg_result.stderr,
    }

    return fit_results

import numpy as np
import pandas as pd
from scipy import signal
import statsmodels.api as sm

def detrend(data):
    """
    Removes the linear trend from a time series.

    Args:
        data (np.ndarray): The input time series data.

    Returns:
        np.ndarray: The detrended time series.
    """
    return signal.detrend(data)

def normalize(data):
    """
    Normalizes a time series to have a mean of 0 and a standard deviation of 1.

    Args:
        data (np.ndarray): The input time series data.

    Returns:
        np.ndarray: The normalized time series.
    """
    mean = np.mean(data)
    std_dev = np.std(data)

    if std_dev == 0:
        # If std is zero, the data is constant. Normalizing would be meaningless
        # and cause division by zero. Return a zero array of the same shape.
        return np.zeros_like(data)

    return (data - mean) / std_dev

def log_transform(data):
    """
    Applies a natural logarithm transformation to the data.

    Args:
        data (np.ndarray): The input time series data.

    Returns:
        np.ndarray: The log-transformed time series.

    Raises:
        ValueError: If the data contains non-positive values.
    """
    if np.any(data <= 0):
        raise ValueError("log-transform requires all data to be positive")

    return np.log(data)

def handle_censored_data(data_series, strategy='drop', lower_multiplier=0.5, upper_multiplier=1.1):
    """
    Handles censored data in a pandas Series.

    Args:
        data_series (pd.Series): The data series, which may contain strings with '<' or '>'.
        strategy (str, optional): The strategy to use. One of ['drop', 'use_detection_limit', 'multiplier'].
                                  Defaults to 'drop'.
                                  - 'drop': Censored values are replaced with NaN and will be ignored in subsequent analysis.
                                  - 'use_detection_limit': (Formerly 'ignore') Uses the numeric value of the detection
                                    limit (e.g., '<5' becomes 5).
                                    **Warning**: This is a statistically biased approach and should be used with caution.
                                  - 'multiplier': Multiplies the detection limit by a factor.
        lower_multiplier (float, optional): The multiplier for left-censored data ('<').
                                            Defaults to 0.5.
        upper_multiplier (float, optional): The multiplier for right-censored data ('>').
                                            Defaults to 1.1.

    Returns:
        np.ndarray: A numeric numpy array with censored values handled.
    """
    # Convert series to numeric, coercing all non-numeric values (including censored) to NaN
    numeric_series = pd.to_numeric(data_series, errors='coerce')

    # Convert to string series to safely use string methods
    str_series = data_series.astype(str)

    # Find left-censored values
    left_censored_mask = str_series.str.startswith('<', na=False)
    left_values = pd.to_numeric(str_series[left_censored_mask].str.lstrip('<'), errors='coerce')

    # Find right-censored values
    right_censored_mask = str_series.str.startswith('>', na=False)
    right_values = pd.to_numeric(str_series[right_censored_mask].str.lstrip('>'), errors='coerce')

    if strategy == 'drop':
        # Set censored values to NaN. This is already done by the initial `pd.to_numeric`
        pass
    elif strategy == 'use_detection_limit':
        numeric_series[left_censored_mask] = left_values
        numeric_series[right_censored_mask] = right_values
    elif strategy == 'multiplier':
        numeric_series[left_censored_mask] = left_values * lower_multiplier
        numeric_series[right_censored_mask] = right_values * upper_multiplier
    else:
        raise ValueError("Invalid strategy. Choose from ['drop', 'use_detection_limit', 'multiplier']")

    # Return as a numpy array
    return numeric_series.to_numpy()

def detrend_loess(x, y, frac=0.5):
    """
    Removes a non-linear trend from a time series using LOESS.

    Args:
        x (np.ndarray): The independent variable (e.g., time).
        y (np.ndarray): The dependent variable (the data).
        frac (float, optional): The fraction of the data used for smoothing.
                                Defaults to 0.5.

    Returns:
        np.ndarray: The detrended time series (residuals).
    """
    # lowess returns a 2D array, where the second column is the smoothed data
    smoothed = sm.nonparametric.lowess(y, x, frac=frac)
    residuals = y - smoothed[:, 1]
    return residuals

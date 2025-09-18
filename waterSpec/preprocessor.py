import numpy as np
import pandas as pd
from scipy import signal

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

def handle_censored_data(data_series, strategy='ignore', lower_multiplier=0.5, upper_multiplier=1.1):
    """
    Handles censored data in a pandas Series.

    Args:
        data_series (pd.Series): The data series, which may contain strings with '<' or '>'.
        strategy (str, optional): The strategy to use. One of ['ignore', 'multiplier'].
                                  Defaults to 'ignore'.
        lower_multiplier (float, optional): The multiplier for left-censored data ('<').
                                            Defaults to 0.5.
        upper_multiplier (float, optional): The multiplier for right-censored data ('>').
                                            Defaults to 1.1.

    Returns:
        np.ndarray: A numeric numpy array with censored values handled.
    """
    # Ensure data is in string format to search for symbols
    processed_series = data_series.astype(str)

    # Create a numeric series for the results, coercing errors to NaN
    numeric_series = pd.to_numeric(processed_series, errors='coerce').to_numpy()

    # Find left-censored values
    left_censored_mask = processed_series.str.startswith('<', na=False)
    left_values = pd.to_numeric(processed_series[left_censored_mask].str.lstrip('<'), errors='coerce')

    # Find right-censored values
    right_censored_mask = processed_series.str.startswith('>', na=False)
    right_values = pd.to_numeric(processed_series[right_censored_mask].str.lstrip('>'), errors='coerce')

    if strategy == 'ignore':
        numeric_series[left_censored_mask] = left_values
        numeric_series[right_censored_mask] = right_values
    elif strategy == 'multiplier':
        numeric_series[left_censored_mask] = left_values * lower_multiplier
        numeric_series[right_censored_mask] = right_values * upper_multiplier
    else:
        raise ValueError("Invalid strategy. Choose from ['ignore', 'multiplier']")

    return numeric_series

import numpy as np
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

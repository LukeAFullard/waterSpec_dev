import numpy as np
import pandas as pd
from scipy import signal
import statsmodels.api as sm
import warnings

def _validate_data_length(data, min_length=10):
    """
    Validates that the data has a sufficient number of non-NaN values.
    """
    valid_points = np.sum(~np.isnan(data))
    if valid_points < min_length:
        raise ValueError(
            f"The time series has only {valid_points} valid data points, "
            f"which is less than the required minimum of {min_length} for analysis."
        )

def detrend(data):
    """
    Removes the linear trend from a time series.
    """
    valid_indices = ~np.isnan(data)
    if np.sum(valid_indices) < 2:
        return data
    data[valid_indices] = signal.detrend(data[valid_indices])
    return data

def normalize(data):
    """
    Normalizes a time series to have a mean of 0 and a standard deviation of 1.
    """
    valid_data = data[~np.isnan(data)]
    if len(valid_data) == 0:
        return data

    mean = np.mean(valid_data)
    std_dev = np.std(valid_data)

    if std_dev == 0:
        data[~np.isnan(data)] = 0
        return data

    data[~np.isnan(data)] = (data[~np.isnan(data)] - mean) / std_dev
    return data

def log_transform(data):
    """
    Applies a natural logarithm transformation to the data.
    """
    valid_data = data[~np.isnan(data)]
    if np.any(valid_data <= 0):
        raise ValueError("log-transform requires all data to be positive")

    data[~np.isnan(data)] = np.log(data[~np.isnan(data)])
    return data

def handle_censored_data(data_series, strategy='drop', lower_multiplier=0.5, upper_multiplier=1.1):
    """
    Handles censored data in a pandas Series.
    """
    numeric_series = pd.to_numeric(data_series, errors='coerce')
    str_series = data_series.astype(str)

    left_censored_mask = str_series.str.startswith('<', na=False)
    left_values = pd.to_numeric(str_series[left_censored_mask].str.lstrip('<'), errors='coerce')

    right_censored_mask = str_series.str.startswith('>', na=False)
    right_values = pd.to_numeric(str_series[right_censored_mask].str.lstrip('>'), errors='coerce')

    if strategy == 'drop':
        pass
    elif strategy == 'use_detection_limit':
        numeric_series[left_censored_mask] = left_values
        numeric_series[right_censored_mask] = right_values
    elif strategy == 'multiplier':
        numeric_series[left_censored_mask] = left_values * lower_multiplier
        numeric_series[right_censored_mask] = right_values * upper_multiplier
    else:
        raise ValueError("Invalid strategy. Choose from ['drop', 'use_detection_limit', 'multiplier']")

    return numeric_series.to_numpy()

def detrend_loess(x, y, **kwargs):
    """
    Removes a non-linear trend from a time series using LOESS.

    This function is a wrapper around `statsmodels.nonparametric.lowess.lowess`.
    Any additional keyword arguments are passed directly to the statsmodels function.

    Args:
        x (np.ndarray): The independent variable (time).
        y (np.ndarray): The dependent variable (data).
        **kwargs: Additional keyword arguments for `statsmodels.lowess`.
                  Common arguments include `frac` (default 0.67) and `it` (default 3).
    """
    # Set a default for `frac` if not provided, consistent with the old signature
    if 'frac' not in kwargs:
        kwargs['frac'] = 0.5

    valid_indices = ~np.isnan(y)
    if np.sum(valid_indices) < 2:
        return y

    x_valid = x[valid_indices]
    y_valid = y[valid_indices]

    smoothed = sm.nonparametric.lowess(y_valid, x_valid, **kwargs)

    detrended_y = np.full_like(y, np.nan)
    detrended_y[valid_indices] = y_valid - smoothed[:, 1]

    return detrended_y

def preprocess_data(data_series, time_numeric, censor_strategy='drop', detrend_method=None, detrend_options=None, min_length=10):
    """
    A wrapper function that applies a series of preprocessing steps.
    """
    if detrend_options is None:
        detrend_options = {}

    processed_data = handle_censored_data(data_series, strategy=censor_strategy)
    _validate_data_length(processed_data, min_length=min_length)

    if detrend_method == 'linear':
        processed_data = detrend(processed_data)
    elif detrend_method == 'loess':
        processed_data = detrend_loess(time_numeric, processed_data, **detrend_options)
    elif detrend_method is not None:
        warnings.warn(f"Unknown detrending method '{detrend_method}'. No detrending will be applied.", UserWarning)

    return processed_data

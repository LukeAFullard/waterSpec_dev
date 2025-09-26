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

def detrend(data, errors=None):
    """
    Removes the linear trend from a time series.

    Note on error propagation: This function does not modify the input errors.
    The errors on the detrended signal are assumed to be the same as the
    original errors, which is a reasonable approximation for linear detrending.
    """
    valid_indices = ~np.isnan(data)
    if np.sum(valid_indices) < 2:
        return data, errors
    data[valid_indices] = signal.detrend(data[valid_indices])
    return data, errors

def normalize(data, errors=None):
    """
    Normalizes a time series to have a mean of 0 and a standard deviation of 1.
    Also propagates errors if they are provided.
    """
    valid_indices = ~np.isnan(data)
    valid_data = data[valid_indices]
    if len(valid_data) == 0:
        return data, errors

    std_dev = np.std(valid_data)

    if std_dev > 0:
        data[valid_indices] = (valid_data - np.mean(valid_data)) / std_dev
        if errors is not None:
            valid_errors = errors[valid_indices]
            errors[valid_indices] = valid_errors / std_dev
    else:
        # If std_dev is 0, the data is constant. Center it at 0.
        data[valid_indices] = 0
        # Errors remain unchanged in scale, but if data is constant, maybe they should be 0?
        # For now, we leave them as they are.

    return data, errors

def log_transform(data, errors=None):
    """
    Applies a natural logarithm transformation to the data.
    Also propagates errors if they are provided, using dy/y.
    """
    valid_indices = ~np.isnan(data)
    valid_data = data[valid_indices]

    if np.any(valid_data <= 0):
        raise ValueError("log-transform requires all data to be positive")

    # Propagate errors first, before transforming the data
    if errors is not None:
        valid_errors = errors[valid_indices]
        # Avoid division by zero, though we already checked for valid_data > 0
        errors[valid_indices] = valid_errors / valid_data

    data[valid_indices] = np.log(valid_data)

    return data, errors

def handle_censored_data(data_series, strategy='drop', lower_multiplier=0.5, upper_multiplier=1.1):
    """
    Handles censored data in a pandas Series by replacing censor marks before
    coercing the series to a numeric type.
    """
    if strategy not in ['drop', 'use_detection_limit', 'multiplier']:
        raise ValueError("Invalid strategy. Choose from ['drop', 'use_detection_limit', 'multiplier']")

    # Work on a copy to avoid modifying the original DataFrame's slice
    series = data_series.copy()

    # Convert to string to safely find censor marks
    str_series = series.astype(str)

    # --- Handle left-censored data (e.g., "<5") ---
    left_censored_mask = str_series.str.startswith('<', na=False)
    if left_censored_mask.any():
        # Get the numeric value of the detection limit
        left_values = pd.to_numeric(str_series[left_censored_mask].str.lstrip('<'), errors='coerce')

        # Replace the string value (e.g., "<5") with the appropriate numeric value
        if strategy == 'drop':
            series.loc[left_censored_mask] = np.nan
        elif strategy == 'use_detection_limit':
            series.loc[left_censored_mask] = left_values
        elif strategy == 'multiplier':
            series.loc[left_censored_mask] = left_values * lower_multiplier

    # --- Handle right-censored data (e.g., ">50") ---
    right_censored_mask = str_series.str.startswith('>', na=False)
    if right_censored_mask.any():
        # Get the numeric value of the detection limit
        right_values = pd.to_numeric(str_series[right_censored_mask].str.lstrip('>'), errors='coerce')

        # Replace the string value (e.g., ">50") with the appropriate numeric value
        if strategy == 'drop':
            series.loc[right_censored_mask] = np.nan
        elif strategy == 'use_detection_limit':
            series.loc[right_censored_mask] = right_values
        elif strategy == 'multiplier':
            series.loc[right_censored_mask] = right_values * upper_multiplier

    # --- Final conversion to numeric ---
    # Now that censor marks are handled, convert the entire series to numeric.
    # Any remaining non-numeric strings (e.g., "apple") or values that could
    # not be coerced from the censor marks will become NaN.
    numeric_series = pd.to_numeric(series, errors='coerce')

    return numeric_series.to_numpy()

def detrend_loess(x, y, errors=None, **kwargs):
    """
    Removes a non-linear trend from a time series using LOESS.

    This function is a wrapper around `statsmodels.nonparametric.lowess.lowess`.
    Any additional keyword arguments are passed directly to the statsmodels function.

    Note on error propagation: This function does not propagate errors. The input
    error series is passed through unchanged. Proper error propagation for LOESS
    is a complex topic, and this simplification should be considered when
    interpreting the results.

    Args:
        x (np.ndarray): The independent variable (time).
        y (np.ndarray): The dependent variable (data).
        errors (np.ndarray, optional): The measurement errors. Defaults to None.
        **kwargs: Additional keyword arguments for `statsmodels.lowess`.
                  Common arguments include `frac` (default 0.5) and `it` (default 3).
    """
    # Set a default for `frac` if not provided, consistent with the old signature
    if 'frac' not in kwargs:
        kwargs['frac'] = 0.5

    valid_indices = ~np.isnan(y)
    if np.sum(valid_indices) < 2:
        return y, errors

    x_valid = x[valid_indices]
    y_valid = y[valid_indices]

    smoothed = sm.nonparametric.lowess(y_valid, x_valid, **kwargs)

    detrended_y = np.full_like(y, np.nan)
    detrended_y[valid_indices] = y_valid - smoothed[:, 1]

    return detrended_y, errors

def preprocess_data(
    data_series,
    time_numeric,
    error_series=None,
    censor_strategy='drop',
    censor_options=None,
    log_transform_data=False,
    detrend_method=None,
    normalize_data=False,
    detrend_options=None,
    min_length=10
):
    """
    A wrapper function that applies a series of preprocessing steps.
    The order of operations is:
    1. Handle censored data
    2. Log-transform (if specified)
    3. Detrend (if specified)
    4. Normalize (if specified)
    """
    if detrend_options is None:
        detrend_options = {}
    if censor_options is None:
        censor_options = {}

    processed_data = handle_censored_data(data_series, strategy=censor_strategy, **censor_options)
    _validate_data_length(processed_data, min_length=min_length)

    processed_errors = None
    if error_series is not None:
        processed_errors = error_series.to_numpy(copy=True)
        nan_mask = np.isnan(processed_data)
        processed_errors[nan_mask] = np.nan

    if log_transform_data:
        processed_data, processed_errors = log_transform(processed_data, processed_errors)

    if detrend_method == 'linear':
        processed_data, processed_errors = detrend(processed_data, processed_errors)
    elif detrend_method == 'loess':
        processed_data, processed_errors = detrend_loess(time_numeric, processed_data, errors=processed_errors, **detrend_options)
    elif detrend_method is not None:
        warnings.warn(f"Unknown detrending method '{detrend_method}'. No detrending will be applied.", UserWarning)

    if normalize_data:
        processed_data, processed_errors = normalize(processed_data, processed_errors)

    return processed_data, processed_errors

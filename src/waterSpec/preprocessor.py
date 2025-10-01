import warnings

import numpy as np
import pandas as pd
import statsmodels.api as sm


def detrend(x, data, errors=None):
    """
    Removes the linear trend from a time series using Ordinary Least Squares.
    This function returns new arrays and does not modify inputs.
    """
    # Create copies to avoid modifying the original arrays
    detrended_data = np.copy(data)
    propagated_errors = np.copy(errors) if errors is not None else None

    valid_indices = ~np.isnan(detrended_data)
    if np.sum(valid_indices) < 2:
        return detrended_data, propagated_errors  # Not enough points to detrend

    x_valid = x[valid_indices]
    y_valid = detrended_data[valid_indices]

    X_with_const = sm.add_constant(x_valid)
    model = sm.OLS(y_valid, X_with_const)
    results = model.fit()

    trend = results.predict(X_with_const)
    detrended_data[valid_indices] = y_valid - trend

    if propagated_errors is not None:
        errors_valid = propagated_errors[valid_indices]
        # To propagate errors correctly, we need the standard error for a
        # single observation's prediction, not the mean prediction. `se_obs`
        # correctly includes the residual standard deviation.
        prediction_se = results.get_prediction(X_with_const).se_obs
        # Propagate errors: new_err^2 = old_err^2 + trend_err^2
        new_err_sq = np.square(errors_valid) + np.square(prediction_se)
        propagated_errors[valid_indices] = np.sqrt(new_err_sq)

    return detrended_data, propagated_errors


def normalize(data, errors=None):
    """
    Normalizes a time series to have a mean of 0 and a standard deviation of 1.
    This function returns new arrays and does not modify inputs.
    """
    normalized_data = np.copy(data)
    normalized_errors = np.copy(errors) if errors is not None else None

    valid_indices = ~np.isnan(normalized_data)
    valid_data = normalized_data[valid_indices]
    if len(valid_data) == 0:
        return normalized_data, normalized_errors

    std_dev = np.std(valid_data)

    if std_dev > 1e-9:  # Use a small threshold to avoid division by zero
        normalized_data[valid_indices] = (valid_data - np.mean(valid_data)) / std_dev
        if normalized_errors is not None:
            errors_valid = normalized_errors[valid_indices]
            normalized_errors[valid_indices] = errors_valid / std_dev
    else:
        # If std_dev is negligible, data is constant. Center it at 0.
        normalized_data[valid_indices] = 0
        if normalized_errors is not None:
            # Errors cannot be meaningfully scaled for constant data.
            warnings.warn(
                "Data has zero variance; cannot normalize errors. Setting to NaN.",
                UserWarning,
            )
            normalized_errors[valid_indices] = np.nan

    return normalized_data, normalized_errors


def log_transform(data, errors=None):
    """
    Applies a natural logarithm transformation to the data and propagates errors.
    This function returns new arrays and does not modify inputs.
    """
    transformed_data = np.copy(data)
    transformed_errors = np.copy(errors) if errors is not None else None

    valid_indices = ~np.isnan(transformed_data)
    valid_data = transformed_data[valid_indices]

    non_positive_mask = valid_data <= 0
    if np.any(non_positive_mask):
        num_non_positive = np.sum(non_positive_mask)
        raise ValueError(
            f"Log-transform requires all data to be positive, but {num_non_positive} "
            "non-positive value(s) were found."
        )

    if transformed_errors is not None:
        valid_errors = transformed_errors[valid_indices]
        # Propagate errors before transforming data: new_err = old_err / value
        transformed_errors[valid_indices] = valid_errors / valid_data

    transformed_data[valid_indices] = np.log(valid_data)

    return transformed_data, transformed_errors


def handle_censored_data(
    data_series,
    strategy="drop",
    lower_multiplier=0.5,
    upper_multiplier=1.1,
    left_censor_symbol="<",
    right_censor_symbol=">",
):
    """
    Handles censored data in a pandas Series by replacing censor marks before
    coercing the series to a numeric type.
    """
    if not isinstance(data_series, pd.Series):
        series = pd.Series(data_series).copy()
    else:
        series = data_series.copy()

    if pd.api.types.is_numeric_dtype(series):
        return series.to_numpy()  # No processing needed if already numeric

    if strategy not in ["drop", "use_detection_limit", "multiplier"]:
        raise ValueError(
            "Invalid censor strategy. Choose from "
            "['drop', 'use_detection_limit', 'multiplier']"
        )

    str_series = series.astype(str)
    original_series = series.copy()  # Keep a copy for finding offenders

    # --- Handle left-censored data (e.g., "<5") ---
    left_mask = str_series.str.startswith(left_censor_symbol, na=False)
    if left_mask.any():
        values = pd.to_numeric(
            str_series[left_mask].str.lstrip(left_censor_symbol), errors="coerce"
        )
        if strategy == "drop":
            series.loc[left_mask] = np.nan
        elif strategy == "use_detection_limit":
            series.loc[left_mask] = values
        elif strategy == "multiplier":
            series.loc[left_mask] = values * lower_multiplier

    # --- Handle right-censored data (e.g., ">50") ---
    right_mask = str_series.str.startswith(right_censor_symbol, na=False)
    if right_mask.any():
        values = pd.to_numeric(
            str_series[right_mask].str.lstrip(right_censor_symbol), errors="coerce"
        )
        if strategy == "drop":
            series.loc[right_mask] = np.nan
        elif strategy == "use_detection_limit":
            series.loc[right_mask] = values
        elif strategy == "multiplier":
            series.loc[right_mask] = values * upper_multiplier

    # --- Final conversion and warning for remaining non-numeric values ---
    numeric_series = pd.to_numeric(series, errors="coerce")
    final_nan_mask = numeric_series.isnull()
    original_nan_mask = original_series.isnull()

    # Identify values that became NaN during processing
    newly_nan_mask = final_nan_mask & ~original_nan_mask
    if newly_nan_mask.any():
        offenders = original_series[newly_nan_mask].unique()
        warnings.warn(
            "Non-numeric or unhandled censored values were found in the data column "
            "and have been converted to NaN. Examples: "
            f"{list(offenders[:5])}",
            UserWarning,
        )

    return numeric_series.to_numpy()


def detrend_loess(x, y, errors=None, frac=0.5, **kwargs):
    """
    Removes a non-linear trend from a time series using LOESS.

    Args:
        frac (float): The fraction of the data used when estimating each
            y-value. Should be between 0 and 1.
    """
    # 1. Validate parameters
    if not isinstance(frac, (int, float)) or not (0 < frac <= 1):
        raise ValueError("`frac` must be a number between 0 and 1.")

    # 2. Prepare data
    valid_indices = ~np.isnan(y)
    num_valid = np.sum(valid_indices)

    # LOESS requires a minimum number of points to function properly.
    # This check prevents statsmodels from throwing an error.
    min_points_for_loess = 5
    if num_valid < min_points_for_loess:
        warnings.warn(
            f"Not enough data points ({num_valid}) for LOESS detrending. "
            f"A minimum of {min_points_for_loess} is required. Skipping.",
            UserWarning,
        )
        return y, errors

    x_valid = x[valid_indices]
    y_valid = y[valid_indices]

    smoothed_y = sm.nonparametric.lowess(y_valid, x_valid, frac=frac, **kwargs)[:, 1]
    residuals = y_valid - smoothed_y
    detrended_y = np.full_like(y, np.nan)
    detrended_y[valid_indices] = residuals

    if errors is not None:
        warnings.warn(
            "Error propagation for LOESS detrending is not supported. "
            "Errors will not be modified.",
            UserWarning,
        )

    return detrended_y, errors


def preprocess_data(
    data_series,
    time_numeric,
    error_series=None,
    censor_strategy="drop",
    censor_options=None,
    log_transform_data=False,
    detrend_method=None,
    normalize_data=False,
    detrend_options=None,
):
    """
    A wrapper function that applies a series of preprocessing steps in a
    defined order:
    1. Handle censored data
    2. Log-transform (if specified)
    3. Detrend (if specified)
    4. Normalize (if specified)
    """
    if detrend_options is None:
        detrend_options = {}
    if censor_options is None:
        censor_options = {}

    # 1. Handle censored data
    processed_data = handle_censored_data(
        data_series, strategy=censor_strategy, **censor_options
    )

    # Align errors with data (set error to NaN where data is NaN)
    processed_errors = None
    if error_series is not None:
        processed_errors = error_series.to_numpy(copy=True)
        nan_mask = np.isnan(processed_data)
        processed_errors[nan_mask] = np.nan

    # 2. Log-transform
    if log_transform_data:
        processed_data, processed_errors = log_transform(
            processed_data, processed_errors
        )

    # 3. Detrend
    if detrend_method == "linear":
        processed_data, processed_errors = detrend(
            time_numeric, processed_data, errors=processed_errors
        )
    elif detrend_method == "loess":
        processed_data, processed_errors = detrend_loess(
            time_numeric, processed_data, errors=processed_errors, **detrend_options
        )
    elif detrend_method is not None:
        warnings.warn(
            f"Unknown detrending method '{detrend_method}'. No detrending applied.",
            UserWarning,
        )

    # 4. Normalize
    if normalize_data:
        processed_data, processed_errors = normalize(processed_data, processed_errors)

    return processed_data, processed_errors
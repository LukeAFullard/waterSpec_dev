import re
import warnings
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
import statsmodels.api as sm

# Define a constant for the R-squared threshold above which a warning is
# issued to the user about the amount of variance removed by detrending.
# A high R-squared might indicate that the linear trend is a dominant
# feature of the data, and its removal should be consciously validated.
R_SQUARED_THRESHOLD_FOR_DETRENDING_WARNING = 0.75


def detrend(
    x: np.ndarray, data: np.ndarray, errors: Optional[np.ndarray] = None
) -> Tuple[np.ndarray, Optional[np.ndarray], Dict]:
    """
    Removes the linear trend from a time series using Ordinary Least Squares
    and provides diagnostics about the removed trend.
    This function returns new arrays and does not modify inputs.
    """
    # Create copies to avoid modifying the original arrays
    detrended_data = np.copy(data)
    propagated_errors = np.copy(errors) if errors is not None else None
    diagnostics = {"r_squared_of_trend": np.nan, "trend_removed": False}

    valid_indices = ~np.isnan(detrended_data)
    if np.sum(valid_indices) < 2:
        return detrended_data, propagated_errors, diagnostics

    x_valid = x[valid_indices]
    y_valid = detrended_data[valid_indices]

    # Calculate variance for diagnostics later
    original_variance = np.var(y_valid)

    X_with_const = sm.add_constant(x_valid)
    model = sm.OLS(y_valid, X_with_const)
    results = model.fit()

    trend = results.predict(X_with_const)
    detrended_data[valid_indices] = y_valid - trend
    diagnostics["trend_removed"] = True
    diagnostics["r_squared_of_trend"] = results.rsquared

    # Warn if the trend accounts for a very large portion of the signal's variance
    if results.rsquared > R_SQUARED_THRESHOLD_FOR_DETRENDING_WARNING:
        warnings.warn(
            f"Linear detrending removed a significant portion "
            f"({results.rsquared:.2%}) of the data's variance. "
            "Ensure this is the expected behavior.",
            UserWarning,
        )

    if propagated_errors is not None:
        errors_valid = propagated_errors[valid_indices]
        # To propagate errors correctly, we need the standard error for a
        # single observation's prediction, not the mean prediction. `se_obs`
        # correctly includes the residual standard deviation.
        prediction_se = results.get_prediction(X_with_const).se_obs
        # Propagate errors: new_err^2 = old_err^2 + trend_err^2
        new_err_sq = np.square(errors_valid) + np.square(prediction_se)
        propagated_errors[valid_indices] = np.sqrt(new_err_sq)

    return detrended_data, propagated_errors, diagnostics


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

    # Use sample standard deviation (ddof=1) as it is generally a more
    # appropriate estimator when working with a sample of data.
    std_dev = np.std(valid_data, ddof=1)

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

    # First, find non-positive values and handle them.
    # We consider all non-NaN values for this check.
    initial_valid_mask = ~np.isnan(transformed_data)
    non_positive_mask = (transformed_data <= 0) & initial_valid_mask

    if np.any(non_positive_mask):
        num_non_positive = np.sum(non_positive_mask)
        warnings.warn(
            f"Log-transform requires all data to be positive. Found {num_non_positive} "
            f"non-positive value(s), which will be converted to NaN.",
            UserWarning,
        )
        # Set non-positive values to NaN
        transformed_data[non_positive_mask] = np.nan
        if transformed_errors is not None:
            transformed_errors[non_positive_mask] = np.nan

    # Now, proceed with the original logic. All remaining values are positive.
    valid_indices = ~np.isnan(transformed_data)
    valid_data = transformed_data[valid_indices]

    if transformed_errors is not None:
        valid_errors = transformed_errors[valid_indices]
        # Propagate errors before transforming data: new_err = old_err / value
        # This is now safe because valid_data only contains positive numbers.
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
    non_detect_symbols=None,
):
    """
    Handles censored data in a pandas Series using a robust, regex-based
    approach that supports custom symbols and various formats.
    """
    if not isinstance(data_series, pd.Series):
        series = pd.Series(data_series).copy()
    else:
        series = data_series.copy()

    if pd.api.types.is_numeric_dtype(series):
        return series.to_numpy()

    if strategy not in ["drop", "use_detection_limit", "multiplier"]:
        raise ValueError(
            "Invalid censor strategy. Choose from "
            "['drop', 'use_detection_limit', 'multiplier']"
        )

    if non_detect_symbols is None:
        non_detect_symbols = ["ND", "non-detect", "BDL"]

    # --- Prepare for processing ---
    str_series = series.astype(str).str.strip()
    # This series will be modified with the results of censoring
    processed_series = series.copy()
    original_series = series.copy()  # For comparison later

    # --- Handle non-detect symbols first (e.g., 'ND', 'BDL') ---
    if non_detect_symbols:
        nd_pattern = "|".join(map(re.escape, non_detect_symbols))
        nd_mask = str_series.str.fullmatch(nd_pattern, case=False, na=False)
        processed_series[nd_mask] = np.nan

    # --- Regex for censored values with numbers (e.g., '<5', '>10.2') ---
    l_sym = re.escape(left_censor_symbol)
    r_sym = re.escape(right_censor_symbol)
    pattern = re.compile(f"^({l_sym}|{r_sym})\\s*([0-9.eE+-]+)$", re.IGNORECASE)

    # Iterate over the series to find and replace censored values
    for idx, value in str_series.items():
        match = pattern.match(str(value))
        if match:
            symbol = match.group(1)
            num_val = float(match.group(2))

            if symbol.lower() == left_censor_symbol.lower():
                if strategy == "drop":
                    processed_series.at[idx] = np.nan
                elif strategy == "use_detection_limit":
                    processed_series.at[idx] = num_val
                elif strategy == "multiplier":
                    processed_series.at[idx] = num_val * lower_multiplier
            elif symbol.lower() == right_censor_symbol.lower():
                if strategy == "drop":
                    processed_series.at[idx] = np.nan
                elif strategy == "use_detection_limit":
                    processed_series.at[idx] = num_val
                elif strategy == "multiplier":
                    processed_series.at[idx] = num_val * upper_multiplier

    # --- Final conversion and warning for remaining non-numeric values ---
    numeric_series = pd.to_numeric(processed_series, errors="coerce")

    # Identify values that became NaN during processing but weren't NaN before
    newly_nan_mask = numeric_series.isnull() & original_series.notnull()
    if newly_nan_mask.any():
        offenders = original_series[newly_nan_mask].unique()
        warnings.warn(
            "Non-numeric or unhandled censored values were found in the data column "
            "and have been converted to NaN. Examples: "
            f"{list(offenders[:5])}",
            UserWarning,
        )

    return numeric_series.to_numpy()


def detrend_loess(
    x: np.ndarray,
    y: np.ndarray,
    errors: Optional[np.ndarray] = None,
    frac: float = 0.5,
    n_bootstrap: int = 0,
    **kwargs,
) -> Tuple[np.ndarray, Optional[np.ndarray], Dict]:
    """
    Removes a non-linear trend from a time series using LOESS and optionally
    estimates trend uncertainty using a residual-resampling bootstrap.

    Args:
        frac (float): The fraction of data used for each y-value estimation.
        n_bootstrap (int): The number of bootstrap iterations to perform for
            uncertainty estimation. If 0, no bootstrapping is performed.
    """
    # 1. Validate parameters
    if not isinstance(frac, (int, float)) or not (0 < frac <= 1):
        raise ValueError("`frac` must be a number between 0 and 1.")
    if not isinstance(n_bootstrap, int) or n_bootstrap < 0:
        raise ValueError("`n_bootstrap` must be a non-negative integer.")

    diagnostics = {"variance_explained_by_trend": np.nan, "trend_removed": False}
    # 2. Prepare data
    valid_indices = ~np.isnan(y)
    num_valid = np.sum(valid_indices)

    min_points_for_loess = 5
    if num_valid < min_points_for_loess:
        warnings.warn(
            f"Not enough data points ({num_valid}) for LOESS detrending. "
            f"A minimum of {min_points_for_loess} is required. Skipping.",
            UserWarning,
        )
        return y, errors, diagnostics

    x_valid = x[valid_indices]
    y_valid = y[valid_indices]
    original_variance = np.var(y_valid)

    # Perform the initial LOESS fit
    trend = sm.nonparametric.lowess(y_valid, x_valid, frac=frac, **kwargs)[:, 1]
    residuals = y_valid - trend
    detrended_y = np.full_like(y, np.nan)
    detrended_y[valid_indices] = residuals

    # Calculate diagnostics
    residual_variance = np.var(residuals)
    if original_variance > 0:
        variance_explained = 1 - (residual_variance / original_variance)
        diagnostics["variance_explained_by_trend"] = variance_explained
        if variance_explained > 0.75:
            warnings.warn(
                f"LOESS detrending removed a significant portion "
                f"({variance_explained:.2%}) of the data's variance. "
                "Ensure this is the expected behavior.",
                UserWarning,
            )
    diagnostics["trend_removed"] = True

    # Propagate errors
    propagated_errors = errors.copy() if errors is not None else None
    if propagated_errors is not None:
        errors_valid = propagated_errors[valid_indices]
        if n_bootstrap > 0:
            # --- Bootstrap to estimate trend uncertainty ---
            bootstrap_trends = np.zeros((n_bootstrap, num_valid))
            for i in range(n_bootstrap):
                # Resample residuals and create a synthetic dataset
                resampled_residuals = np.random.choice(
                    residuals, size=num_valid, replace=True
                )
                synthetic_y = trend + resampled_residuals
                # Fit LOESS to the synthetic data
                bootstrap_trends[i, :] = sm.nonparametric.lowess(
                    synthetic_y, x_valid, frac=frac, **kwargs
                )[:, 1]

            # Calculate the standard deviation of the trends as the uncertainty
            trend_uncertainty = np.std(bootstrap_trends, axis=0)

            # Add in quadrature with original errors
            new_err_sq = np.square(errors_valid) + np.square(trend_uncertainty)
            propagated_errors[valid_indices] = np.sqrt(new_err_sq)
        else:
            warnings.warn(
                "Error propagation for LOESS detrending is not currently supported "
                "without bootstrapping (set `n_bootstrap` > 0). The uncertainties "
                "on the detrended data will be the same as the original, which may "
                "be an underestimate.",
                UserWarning,
            )

    return detrended_y, propagated_errors, diagnostics


def preprocess_data(
    data_series: pd.Series,
    time_numeric: np.ndarray,
    error_series: Optional[pd.Series] = None,
    censor_strategy: str = "drop",
    censor_options: Optional[Dict] = None,
    log_transform_data: bool = False,
    detrend_method: Optional[str] = None,
    normalize_data: bool = False,
    detrend_options: Optional[Dict] = None,
) -> Tuple[np.ndarray, Optional[np.ndarray], Dict]:
    """
    A wrapper function that applies a series of preprocessing steps in a
    defined order:
    1. Handle censored data
    2. Log-transform (if specified)
    3. Detrend (if specified)
    4. Normalize (if specified)

    .. note::
        The handling of censored data (`censor_strategy`) can significantly
        impact the results of subsequent spectral analysis. Dropping data,
        substituting with the detection limit, or using a multiplier can all
        introduce biases in power-law slope estimates. Users should carefully
        consider the nature of their data and the potential effects of their
        chosen strategy. For datasets with a large fraction of non-detects,
        more advanced methods like Tobit models or multiple imputation may be
        more appropriate.
    """
    if detrend_options is None:
        detrend_options = {}
    if censor_options is None:
        censor_options = {}

    diagnostics = {}

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
    detrend_diagnostics = {}
    if detrend_method == "linear":
        processed_data, processed_errors, detrend_diagnostics = detrend(
            time_numeric, processed_data, errors=processed_errors
        )
    elif detrend_method == "loess":
        processed_data, processed_errors, detrend_diagnostics = detrend_loess(
            time_numeric, processed_data, errors=processed_errors, **detrend_options
        )
    elif detrend_method is not None:
        warnings.warn(
            f"Unknown detrending method '{detrend_method}'. No detrending applied.",
            UserWarning,
        )
    diagnostics["detrending"] = detrend_diagnostics

    # 4. Normalize
    if normalize_data:
        processed_data, processed_errors = normalize(processed_data, processed_errors)

    return processed_data, processed_errors, diagnostics
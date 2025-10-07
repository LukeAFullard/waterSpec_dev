import re
import warnings
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd


def _moving_block_bootstrap_indices(
    n_points: int, block_size: int, rng: np.random.Generator
) -> np.ndarray:
    """
    Generates bootstrap indices for a single moving-block bootstrap sample.

    This function is used for block bootstrapping procedures where the
    correlation structure of the data (e.g., time series residuals) needs to
    be preserved. This implementation uses a circular block bootstrap to
    ensure that all data points have an equal probability of being included,
    avoiding the bias from non-circular methods where `n_points` is not
    divisible by `block_size`.

    Args:
        n_points: The total number of data points.
        block_size: The size of each block.
        rng: The random number generator.

    Returns:
        An array of indices for one bootstrap sample.
    """
    # Ensure block size is valid
    block_size = min(block_size, n_points)
    if block_size <= 0:
        return np.arange(n_points)

    indices = []
    while len(indices) < n_points:
        # Choose a random start and create a block of indices, wrapping around
        start = rng.integers(0, n_points)
        block = (start + np.arange(block_size)) % n_points
        indices.extend(block)

    # Truncate to the original number of points
    return np.array(indices[:n_points])


# Define a constant for the R-squared threshold above which a warning is
# issued to the user about the amount of variance removed by detrending.
# A high R-squared might indicate that the linear trend is a dominant
# feature of the data, and its removal should be consciously validated.
R_SQUARED_THRESHOLD_FOR_DETRENDING_WARNING = 0.75


def detrend(
    x: np.ndarray, data: np.ndarray, errors: Optional[np.ndarray] = None
) -> Tuple[np.ndarray, Optional[np.ndarray], Dict]:
    """
    Removes the linear trend from a time series and provides diagnostics.

    If measurement errors are provided and contain no NaNs, this function uses
    Weighted Least Squares (WLS) for a more accurate trend estimate, with
    weights set to 1/error^2. Otherwise, it falls back to Ordinary Least
    Squares (OLS).

    Error propagation for the detrended data is calculated as:
    new_err = sqrt(old_err^2 + trend_err^2), where trend_err is the standard
    error of the trend prediction. This assumes the measurement and trend
    errors are independent.

    This function returns new arrays and does not modify inputs.
    """
    try:
        import statsmodels.api as sm
    except ImportError:
        raise ImportError(
            "statsmodels is required for detrending. Install with `pip install statsmodels`"
        )
    # Create copies to avoid modifying the original arrays
    detrended_data = np.copy(data)
    propagated_errors = np.copy(errors) if errors is not None else None
    diagnostics = {"r_squared_of_trend": np.nan, "trend_removed": False}

    valid_indices = ~np.isnan(detrended_data)
    if np.sum(valid_indices) < 2:
        return detrended_data, propagated_errors, diagnostics

    x_valid = x[valid_indices]
    y_valid = detrended_data[valid_indices]

    # --- Model fitting: Choose between WLS and OLS ---
    X_with_const = sm.add_constant(x_valid)
    use_wls = False
    if propagated_errors is not None:
        errors_valid = propagated_errors[valid_indices]
        # Check if there are any NaNs within the valid error region
        if not np.any(np.isnan(errors_valid)):
            use_wls = True

    if use_wls:
        # All conditions met for WLS
        errors_valid = propagated_errors[valid_indices]
        weights = 1.0 / np.square(errors_valid)
        model = sm.WLS(y_valid, X_with_const, weights=weights)
    else:
        # Fallback to OLS
        if propagated_errors is not None:
            # If we are falling back because of NaNs in errors, warn the user.
            warnings.warn(
                "NaNs found in the 'errors' array. Falling back to OLS for "
                "detrending. The trend will not be weighted by measurement error.",
                UserWarning,
            )
        model = sm.OLS(y_valid, X_with_const)

    results = model.fit()
    # --- End of model fitting ---

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


def normalize(data, errors=None, name: Optional[str] = "series"):
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
        # If std_dev is negligible, data is constant. Raise an error.
        series_name = name if name is not None else "series"
        raise ValueError(
            f"Series '{series_name}' has zero variance and cannot be normalized."
        )

    return normalized_data, normalized_errors


def log_transform(data, errors=None):
    """
    Applies a base-10 logarithm transformation to the data and propagates errors.
    This function assumes that the input data has already been sanitized to
    ensure all values are positive. It returns new arrays and does not
    modify inputs.
    """
    transformed_data = np.copy(data)
    transformed_errors = np.copy(errors) if errors is not None else None

    # Identify non-positive data points, which will result in NaNs after log transform.
    # This must be done before the transformation itself.
    non_positive_mask = data <= 0

    if transformed_errors is not None:
        # Where data is non-positive, error propagation is undefined and should also be NaN.
        transformed_errors[non_positive_mask] = np.nan
        # Propagate errors for valid data points. For a base-10 log, the formula is:
        # σ_log10(x) = σ_x / (x * ln(10))
        valid_indices = ~non_positive_mask & ~np.isnan(data)
        valid_data = data[valid_indices]
        valid_errors = transformed_errors[valid_indices]
        # Using np.log(10) which is the natural logarithm of 10.
        transformed_errors[valid_indices] = valid_errors / (valid_data * np.log(10))

    # Now, apply the log10 transform. This will correctly produce NaNs for non-positive values.
    # Using np.errstate to suppress warnings about invalid values (log of non-positive).
    with np.errstate(invalid="ignore"):
        transformed_data = np.log10(data)

    return transformed_data, transformed_errors


def handle_censored_data(
    data_series,
    strategy="drop",
    lower_multiplier=0.5,
    upper_multiplier=1.1,
    left_censor_symbol="<",
    right_censor_symbol=">",
    non_detect_symbols=None,
    decimal_separator=".",
):
    """
    Handles censored data in a pandas Series using a robust, regex-based
    approach that supports custom symbols and various formats.

    .. note::
        This function processes non-detect symbols (e.g., "ND", "BDL") before
        handling censored numerical values (e.g., "<5"). Non-detect symbols
        are always converted to ``np.nan``, regardless of the chosen ``strategy``.
        This means strategies like "multiplier" or "use_detection_limit" do not
        apply to them.

    Args:
        data_series (pd.Series or array-like): The input data.
        strategy (str): The strategy for handling censored data.
            - 'drop': Replace censored values with NaN.
            - 'use_detection_limit': Use the numeric value of the detection limit.
            - 'multiplier': Multiply the detection limit by a factor.
        lower_multiplier (float): Multiplier for left-censored values (e.g., <5).
        upper_multiplier (float): Multiplier for right-censored values (e.g., >100).
        left_censor_symbol (str): The symbol for left-censored data.
        right_censor_symbol (str): The symbol for right-censored data.
        non_detect_symbols (list of str, optional): A list of strings to be
            treated as non-detects (e.g., "ND"). These are always converted
            to NaN.
        decimal_separator (str): The character used as the decimal separator,
            either '.' (default) or ','. This is crucial for correctly
            interpreting numbers in different locales. For example, '1,234.5'
            (US/UK) vs '1.234,5' (EU).
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

    if decimal_separator not in [".", ","]:
        raise ValueError("`decimal_separator` must be either '.' or ','.")

    if non_detect_symbols is None:
        non_detect_symbols = ["ND", "non-detect", "BDL"]

    # Determine the thousands separator based on the decimal separator.
    if decimal_separator == ".":
        thousands_separator = ","
    else:
        thousands_separator = "."

    # --- Prepare for processing ---
    str_series = series.astype(str).str.strip()
    # This series will be modified with the results of censoring
    processed_series = series.copy()
    original_series = series.copy()  # For comparison later
    col_name = series.name if series.name is not None else "data_series"
    num_affected = 0

    # --- Handle non-detect symbols first (e.g., 'ND', 'BDL') ---
    if non_detect_symbols:
        nd_pattern = "|".join(map(re.escape, non_detect_symbols))
        nd_mask = str_series.str.fullmatch(nd_pattern, case=False, na=False)
        num_affected += nd_mask.sum()
        processed_series[nd_mask] = np.nan

    # --- Regex for censored values with numbers (e.g., '<5', '>10.2') ---
    l_sym = re.escape(left_censor_symbol)
    r_sym = re.escape(right_censor_symbol)
    pattern = re.compile(
        f"^({l_sym}|{r_sym})\\s*([0-9.,]+(?:[eE][+-]?[0-9]+)?)\\s*.*$", re.IGNORECASE
    )

    # Iterate over the series to find and replace censored values
    for idx, value in str_series.items():
        match = pattern.match(str(value))
        if match:
            num_affected += 1
            symbol = match.group(1)
            num_str = match.group(2)

            # Clean the numeric string based on the specified separators
            cleaned_num_str = num_str.replace(thousands_separator, "")
            standardized_num_str = cleaned_num_str.replace(decimal_separator, ".")
            num_val = float(standardized_num_str)

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
    def _final_converter(value):
        if isinstance(value, (int, float)):
            return float(value)
        try:
            # It's a string that needs cleaning based on locale.
            cleaned_num_str = str(value).replace(thousands_separator, "")
            standardized_num_str = cleaned_num_str.replace(decimal_separator, ".")
            return float(standardized_num_str)
        except (ValueError, TypeError):
            return np.nan

    numeric_series = processed_series.apply(_final_converter)

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

    if num_affected > 0:
        warnings.warn(
            f"{num_affected} censored or non-finite values replaced in '{col_name}'.",
            RuntimeWarning,
        )

    return numeric_series.to_numpy()


def detrend_loess(
    x: np.ndarray,
    y: np.ndarray,
    errors: Optional[np.ndarray] = None,
    frac: float = 0.5,
    n_bootstrap: int = 0,
    bootstrap_block_size: Optional[int] = None,
    seed: Optional[np.random.SeedSequence] = None,
    **kwargs,
) -> Tuple[np.ndarray, Optional[np.ndarray], Dict]:
    """
    Removes a non-linear trend from a time series using LOESS and optionally
    estimates trend uncertainty using a block bootstrap on the residuals.

    Args:
        x (np.ndarray): The independent variable (e.g., time).
        y (np.ndarray): The dependent variable (data series).
        errors (np.ndarray, optional): Measurement errors for `y`.
        frac (float): The fraction of data used for each y-value estimation.
            This is a key parameter for LOESS.
        n_bootstrap (int): The number of bootstrap iterations to perform for
            uncertainty estimation. If 0, no bootstrapping is performed and
            error propagation is not fully supported.
        bootstrap_block_size (int, optional): The block size for the moving
            block bootstrap of residuals. If None, a rule-of-thumb
            `n_points**(1/3)` is used.
        seed (int, optional): A seed for the random number generator to ensure
            reproducibility of the bootstrap. Defaults to None.
        **kwargs: Additional keyword arguments passed to
            `statsmodels.nonparametric.lowess`.
    """
    try:
        import statsmodels.api as sm
    except ImportError:
        raise ImportError(
            "statsmodels is required for detrending. Install with `pip install statsmodels`"
        )
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
            # --- Block bootstrap to estimate trend uncertainty ---
            # If the seed is an integer, create a SeedSequence from it for robust
            # RNG creation. If it's already a SeedSequence, it will be used directly.
            rng = np.random.default_rng(
                np.random.SeedSequence(seed) if isinstance(seed, int) else seed
            )
            if bootstrap_block_size is None:
                # Rule-of-thumb for block size, with a minimum of 3.
                block_size = max(3, int(np.ceil(num_valid ** (1 / 3))))
            else:
                block_size = bootstrap_block_size

            bootstrap_trends = np.zeros((n_bootstrap, num_valid))
            for i in range(n_bootstrap):
                # Resample residuals using moving blocks to preserve autocorrelation
                indices = _moving_block_bootstrap_indices(num_valid, block_size, rng)
                resampled_residuals = residuals[indices]

                # Create a synthetic dataset
                synthetic_y = trend + resampled_residuals
                # Fit LOESS to the synthetic data
                bootstrap_trends[i, :] = sm.nonparametric.lowess(
                    synthetic_y, x_valid, frac=frac, **kwargs
                )[:, 1]

            # Calculate the standard deviation of the trends as the uncertainty
            trend_uncertainty = np.std(bootstrap_trends, axis=0)

            # Propagate errors: new_err^2 = original_err^2 + trend_err^2
            # The uncertainty of the detrended signal is the quadrature sum of the
            # original measurement error and the uncertainty in the trend model.
            propagated_errors[valid_indices] = np.sqrt(
                np.square(errors_valid) + np.square(trend_uncertainty)
            )
        else:
            # If errors are provided, bootstrapping is required for propagation.
            # Raise an error to prevent silent underestimation of uncertainty.
            raise ValueError(
                "Error propagation for LOESS detrending requires bootstrapping. "
                "Please set `n_bootstrap` to a value > 0 (e.g., 100) to "
                "estimate trend uncertainty, or handle errors manually."
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
        data_series,
        strategy=censor_strategy,
        **censor_options,
    )

    # Align errors with data (set error to NaN where data is NaN)
    processed_errors = None
    if error_series is not None:
        processed_errors = error_series.to_numpy(copy=True)
        nan_mask = np.isnan(processed_data)
        processed_errors[nan_mask] = np.nan

    # 2. Log-transform
    if log_transform_data:
        # Validate that all data is positive before log-transform. This check
        # is crucial because censoring strategies (e.g., 'use_detection_limit')
        # can introduce zeros, which are invalid for log-transformation.
        non_positive_mask = (~np.isnan(processed_data)) & (processed_data <= 0)
        if np.any(non_positive_mask):
            num_non_positive = np.sum(non_positive_mask)
            raise ValueError(
                f"Log-transform requires all data to be positive, but {num_non_positive} "
                "non-positive value(s) were found. This can occur if the original "
                "data contains zeros or if a censoring strategy (e.g., "
                "'use_detection_limit' with a limit of 0) introduced them. "
                "Please clean the data or adjust the censoring strategy."
            )

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
        col_name = data_series.name if data_series.name is not None else "data_series"
        processed_data, processed_errors = normalize(
            processed_data, processed_errors, name=col_name
        )

    return processed_data, processed_errors, diagnostics
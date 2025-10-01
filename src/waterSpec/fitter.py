import logging
import warnings
from typing import Dict, Optional

import numpy as np
from scipy import stats

try:
    import piecewise_regression
except ImportError:
    piecewise_regression = None
    _PIECEWISE_MISSING_MSG = (
        "The 'piecewise-regression' package is required for segmented "
        "spectrum fitting. Install it with 'pip install piecewise-regression' "
        "or include it in your environment."
    )
try:
    from statsmodels.stats.stattools import durbin_watson
except ImportError:
    durbin_watson = None
    _STATSMODELS_MISSING_MSG = (
        "The 'statsmodels' package is required for residual bootstrapping and "
        "autocorrelation checks. Install it with 'pip install statsmodels' "
        "or include it in your environment."
    )


def _calculate_bic(y: np.ndarray, y_pred: np.ndarray, n_params: int) -> float:
    """Calculates the Bayesian Information Criterion (BIC)."""
    n = len(y)
    if n == 0:
        return np.nan
    rss = np.sum((y - y_pred) ** 2)
    if rss < 1e-12:
        warnings.warn(
            "Near-zero RSS found, indicating a perfect fit. "
            "This may be due to overfitting or numerical instability. "
            "Returning BIC as infinity.",
            UserWarning,
        )
        return np.inf
    bic = n * np.log(rss / n) + n_params * np.log(n)
    return bic


def _calculate_aic(y: np.ndarray, y_pred: np.ndarray, n_params: int) -> float:
    """Calculates the Akaike Information Criterion (AIC)."""
    n = len(y)
    if n == 0:
        return np.nan
    rss = np.sum((y - y_pred) ** 2)
    if rss < 1e-12:
        return -np.inf  # A perfect fit corresponds to a minimal AIC
    aic = n * np.log(rss / n) + 2 * n_params
    return aic


def fit_standard_model(
    frequency: np.ndarray,
    power: np.ndarray,
    method: str = "theil-sen",
    ci_method: str = "bootstrap",
    bootstrap_type: str = "residuals",
    n_bootstraps: int = 1000,
    ci: int = 95,
    seed: Optional[int] = None,
    logger: Optional[logging.Logger] = None,
) -> Dict:
    """
    Fits a standard (non-segmented) model to the power spectrum and estimates
    confidence intervals for the spectral exponent (beta).

    This function consolidates the fitting, BIC calculation, and confidence
    interval estimation into a single, robust workflow.

    Args:
        frequency (np.ndarray): The frequency array.
        power (np.ndarray): The power array.
        method (str, optional): The fitting method ('theil-sen' or 'ols').
        ci_method (str, optional): Method for CI calculation.
        bootstrap_type (str, optional): The bootstrap method to use. Can be
            'pairs' (default), which resamples (x, y) pairs, or 'residuals',
            which resamples the model residuals.
        n_bootstraps (int, optional): Number of bootstrap samples.
        ci (int, optional): The desired confidence interval in percent.
        seed (int, optional): A seed for the random number generator.
        logger (logging.Logger, optional): A logger for warnings.
    """
    logger = logger or logging.getLogger(__name__)
    # 1. Validate inputs and filter data
    if not isinstance(frequency, np.ndarray) or not isinstance(power, np.ndarray):
        raise TypeError("Input 'frequency' and 'power' must be numpy arrays.")
    if frequency.shape != power.shape:
        raise ValueError("'frequency' and 'power' must have the same shape.")
    if not np.all(np.isfinite(frequency)) or not np.all(np.isfinite(power)):
        raise ValueError("Input arrays must contain finite values.")
    if n_bootstraps < 0:
        raise ValueError("'n_bootstraps' must be non-negative.")
    if not 0 < ci < 100:
        raise ValueError("'ci' must be between 0 and 100.")
    if method not in ["ols", "theil-sen"]:
        raise ValueError(f"Unknown fitting method: '{method}'. Choose 'ols' or 'theil-sen'.")
    if bootstrap_type not in ["pairs", "residuals"]:
        raise ValueError(
            f"Unknown bootstrap_type: '{bootstrap_type}'. Choose 'pairs' or 'residuals'."
        )
    valid_indices = (frequency > 0) & (power > 0)
    if np.sum(valid_indices) < 2:
        return {"beta": np.nan, "bic": np.nan, "beta_ci_lower": np.nan, "beta_ci_upper": np.nan}

    log_freq = np.log(frequency[valid_indices])
    log_power = np.log(power[valid_indices])
    n_points = len(log_power)

    if n_points < 30 and "bootstrap" in ci_method:
        logger.warning(
            f"Dataset has only {n_points} points. Bootstrap CIs may be "
            "unreliable. Consider using parametric CIs or collecting more data."
        )

    # 2. Perform the initial fit (OLS or Theil-Sen)
    fit_results = {}
    try:
        if method == "ols":
            res = stats.linregress(log_freq, log_power)
            slope, intercept, r_value, _, stderr = res
            r_squared = r_value**2
            # Calculate adjusted R-squared
            adj_r_squared = 1 - (1 - r_squared) * (n_points - 1) / (n_points - 2 - 1)
            fit_results.update({
                "r_squared": r_squared,
                "adj_r_squared": adj_r_squared,
                "stderr": stderr,
            })
        elif method == "theil-sen":
            res = stats.theilslopes(log_power, log_freq, alpha=1 - (ci / 100))
            slope, intercept, low_slope, high_slope = res
            fit_results.update({"slope_ci_lower": low_slope, "slope_ci_upper": high_slope})

        fit_results.update({"beta": -slope, "intercept": intercept})

    except Exception as e:
        failure_reason = (
            f"Initial standard model fit failed with method '{method}'. Error: {e}"
        )
        logger.warning(failure_reason)
        return {
            "beta": np.nan,
            "bic": np.inf,
            "aic": np.inf,
            "beta_ci_lower": np.nan,
            "beta_ci_upper": np.nan,
            "failure_reason": failure_reason,
        }

    # 3. Calculate BIC and AIC
    log_power_fit = slope * log_freq + intercept
    n_params = 2
    bic = _calculate_bic(log_power, log_power_fit, n_params)
    aic = _calculate_aic(log_power, log_power_fit, n_params)
    fit_results["bic"] = bic
    fit_results["aic"] = aic

    # 4. Calculate Confidence Intervals
    beta_ci_lower, beta_ci_upper = np.nan, np.nan
    if ci_method == "bootstrap":
        # Check for autocorrelation in residuals if using residual bootstrap
        if bootstrap_type == "residuals" and durbin_watson is None:
            logger.error(_STATSMODELS_MISSING_MSG)
            raise ImportError(_STATSMODELS_MISSING_MSG)

        residuals = log_power - log_power_fit
        if durbin_watson:
            dw_stat = durbin_watson(residuals)
            fit_results["durbin_watson_stat"] = dw_stat
            # A DW statistic between 1.5 and 2.5 is generally considered normal.
            # Values outside this range suggest autocorrelation.
            if not 1.5 < dw_stat < 2.5:
                logger.warning(
                    f"Durbin-Watson statistic is {dw_stat:.2f}, indicating "
                    "potential autocorrelation in the model residuals. "
                    "The 'residuals' bootstrap method is recommended in this case."
                )

        rng = np.random.default_rng(seed)
        beta_estimates = []
        n_points = len(log_freq)
        error_counts = {}

        for _ in range(n_bootstraps):
            try:
                if bootstrap_type == "pairs":
                    indices = rng.choice(np.arange(n_points), size=n_points, replace=True)
                    resampled_log_freq = log_freq[indices]
                    resampled_log_power = log_power[indices]
                elif bootstrap_type == "residuals":
                    # Resample residuals
                    resampled_residuals = rng.choice(
                        residuals - np.mean(residuals), size=n_points, replace=True
                    )
                    # Create a new synthetic dataset
                    resampled_log_power = log_power_fit + resampled_residuals
                    resampled_log_freq = log_freq  # Keep original frequencies
                else:
                    # This case is handled by the initial validation, but included for safety
                    continue

                # Refit the model on the resampled data
                if method == "ols":
                    resampled_slope = stats.linregress(
                        resampled_log_freq, resampled_log_power
                    ).slope
                else:  # theil-sen
                    resampled_slope = stats.theilslopes(
                        resampled_log_power, resampled_log_freq
                    )[0]
                beta_estimates.append(-resampled_slope)
            except Exception as e:
                error_type = type(e).__name__
                error_counts[error_type] = error_counts.get(error_type, 0) + 1
                # Log the specific error for a failed iteration
                msg = f"Bootstrap iteration failed with error: {e}"
                if logger:
                    logger.debug(msg)
                continue

        MIN_BOOTSTRAP_SAMPLES = 50  # Min samples for a reliable CI
        if len(beta_estimates) < MIN_BOOTSTRAP_SAMPLES:
            if error_counts:
                error_summary = ", ".join(
                    [f"{err}: {count}" for err, count in error_counts.items()]
                )
                logger.warning(f"Bootstrap errors occurred: {error_summary}")
            logger.warning(
                f"Only {len(beta_estimates)}/{n_bootstraps} bootstrap iterations "
                f"succeeded for the standard model (minimum required: {MIN_BOOTSTRAP_SAMPLES}). "
                "Confidence intervals are unreliable and will be set to NaN."
            )
            beta_ci_lower, beta_ci_upper = np.nan, np.nan
        else:
            MIN_BOOTSTRAP_SUCCESS_RATIO = 0.9
            if len(beta_estimates) < n_bootstraps * MIN_BOOTSTRAP_SUCCESS_RATIO:
                logger.warning(
                    f"Only {len(beta_estimates)}/{n_bootstraps} bootstrap iterations "
                    f"succeeded (success rate < {MIN_BOOTSTRAP_SUCCESS_RATIO:.0%}). "
                    "The resulting confidence interval may be less reliable."
                )
            # Filter out non-finite values (NaN, inf) before calculating percentiles
            finite_betas = [b for b in beta_estimates if np.isfinite(b)]
            if len(finite_betas) < MIN_BOOTSTRAP_SAMPLES:
                logger.warning(
                    f"After filtering, only {len(finite_betas)} finite bootstrap "
                    f"estimates remain (minimum required: {MIN_BOOTSTRAP_SAMPLES}). "
                    "Confidence intervals are unreliable and will be set to NaN."
                )
                beta_ci_lower, beta_ci_upper = np.nan, np.nan
            else:
                p_lower = (100 - ci) / 2
                p_upper = 100 - p_lower
                beta_ci_lower = np.percentile(finite_betas, p_lower)
                beta_ci_upper = np.percentile(finite_betas, p_upper)

    elif ci_method == "parametric":
        if method == "ols":
            stderr = fit_results.get("stderr", np.nan)
            if np.isfinite(stderr):
                t_val = stats.t.ppf((1 + ci / 100) / 2, len(log_freq) - 2)
                half_width = t_val * stderr
                slope_ci_lower, slope_ci_upper = slope - half_width, slope + half_width
                # Note: The CI for beta is inverted because beta = -slope.
                # A lower bound on the slope corresponds to an upper bound on beta.
                beta_ci_lower, beta_ci_upper = -slope_ci_upper, -slope_ci_lower
        elif method == "theil-sen":
            slope_ci_lower = fit_results.get("slope_ci_lower", np.nan)
            slope_ci_upper = fit_results.get("slope_ci_upper", np.nan)
            # Note: The CI for beta is inverted because beta = -slope.
            beta_ci_lower, beta_ci_upper = -slope_ci_upper, -slope_ci_lower
    else:
        raise ValueError(f"Unknown ci_method: '{ci_method}'")

    fit_results.update({"beta_ci_lower": beta_ci_lower, "beta_ci_upper": beta_ci_upper})

    # 5. Store supplementary data for plotting and diagnostics
    fit_results.update({
        "log_freq": log_freq,
        "log_power": log_power,
        "residuals": log_power - log_power_fit,
        "fitted_log_power": log_power_fit,
    })

    return fit_results


# Define a constant for the minimum number of data points required per
# segment in a regression to ensure stable fits.
MIN_POINTS_PER_SEGMENT = 10


def _bootstrap_segmented_fit(
    pw_fit,
    log_freq,
    log_power,
    n_bootstraps,
    ci,
    seed,
    bootstrap_type="residuals",
    logger=None,
):
    """
    Performs robust bootstrap resampling for a fitted piecewise model.
    This also calculates the confidence interval of the fitted line itself.
    """
    logger = logger or logging.getLogger(__name__)
    n_breakpoints = pw_fit.n_breakpoints
    rng = np.random.default_rng(seed)
    n_points = len(log_freq)

    bootstrap_betas = [[] for _ in range(n_breakpoints + 1)]
    bootstrap_breakpoints = [[] for _ in range(n_breakpoints)]
    bootstrap_fits = []  # Store each bootstrap fit line
    successful_fits = 0
    error_counts = {}

    # For residual bootstrap, we need the initial fitted values and residuals
    if bootstrap_type == "residuals":
        initial_fitted_power = pw_fit.predict(log_freq)
        initial_residuals = log_power - initial_fitted_power

    # Extract the starting breakpoints from the initial fit's results.
    # This was the source of a bug where bootstrap CIs would always fail.
    # The previous code tried to access a non-existent `estimates` attribute
    # on the `pw_fit` object.
    initial_results = pw_fit.get_results()
    initial_estimates = initial_results.get("estimates")

    if not initial_estimates:
        logger.warning("Could not perform bootstrap: initial fit failed or produced no estimates.")
        return {
            "betas_ci": [(np.nan, np.nan)] * (n_breakpoints + 1),
            "breakpoints_ci": [(np.nan, np.nan)] * n_breakpoints,
            "fit_ci_lower": None,
            "fit_ci_upper": None,
        }

    for _ in range(n_bootstraps):
        try:
            if bootstrap_type == "pairs":
                indices = rng.choice(np.arange(n_points), size=n_points, replace=True)
                resampled_log_freq = log_freq[indices]
                resampled_log_power = log_power[indices]

                # Sort the resampled data by frequency
                sort_order = np.argsort(resampled_log_freq)
                resampled_log_freq_sorted = resampled_log_freq[sort_order]
                resampled_log_power_sorted = resampled_log_power[sort_order]
            elif bootstrap_type == "residuals":
                resampled_residuals = rng.choice(
                    initial_residuals - np.mean(initial_residuals),
                    size=n_points,
                    replace=True,
                )
                resampled_log_power_sorted = initial_fitted_power + resampled_residuals
                resampled_log_freq_sorted = log_freq  # Keep original frequencies
            else:
                continue

            bootstrap_pw_fit = piecewise_regression.Fit(
                resampled_log_freq_sorted,
                resampled_log_power_sorted,
                n_breakpoints=n_breakpoints,
            )

            # Bug fix: Correctly get results from the bootstrap fit object.
            # The previous code incorrectly tried to access a non-existent `estimates` attribute.
            bootstrap_results = bootstrap_pw_fit.get_results()
            if not bootstrap_results["converged"]:
                continue

            estimates = bootstrap_results["estimates"]
            if not estimates:
                continue

            for i in range(n_breakpoints):
                bp_val = np.exp(estimates[f"breakpoint{i+1}"]["estimate"])
                bootstrap_breakpoints[i].append(bp_val)

            slopes = [estimates["alpha1"]["estimate"]]
            for i in range(1, n_breakpoints + 1):
                slopes.append(slopes[-1] + estimates[f"beta{i}"]["estimate"])

            for i, slope in enumerate(slopes):
                bootstrap_betas[i].append(-slope)

            # Store the predicted line from this bootstrap sample
            bootstrap_fits.append(bootstrap_pw_fit.predict(log_freq))
            successful_fits += 1
        except Exception as e:
            error_type = type(e).__name__
            error_counts[error_type] = error_counts.get(error_type, 0) + 1
            # Log the specific error for a failed iteration
            msg = f"Segmented bootstrap iteration failed with error: {e}"
            if logger:
                logger.debug(msg)
            continue

    MIN_BOOTSTRAP_SAMPLES = 50  # Min samples for a reliable CI
    if successful_fits < MIN_BOOTSTRAP_SAMPLES:
        if error_counts:
            error_summary = ", ".join(
                [f"{err}: {count}" for err, count in error_counts.items()]
            )
            logger.warning(f"Bootstrap errors occurred: {error_summary}")
        logger.warning(
            f"Only {successful_fits}/{n_bootstraps} bootstrap iterations "
            f"succeeded for the segmented model (minimum required: {MIN_BOOTSTRAP_SAMPLES}). "
            "Confidence intervals are unreliable and will be set to NaN."
        )
        return {
            "betas_ci": [(np.nan, np.nan)] * (n_breakpoints + 1),
            "breakpoints_ci": [(np.nan, np.nan)] * n_breakpoints,
            "fit_ci_lower": None,
            "fit_ci_upper": None,
        }

    # Warn if the success rate is low, but still above the minimum sample count
    MIN_BOOTSTRAP_SUCCESS_RATIO = 0.9
    if successful_fits < n_bootstraps * MIN_BOOTSTRAP_SUCCESS_RATIO:
        logger.warning(
            f"Only {successful_fits}/{n_bootstraps} bootstrap iterations for the "
            f"segmented model succeeded (success rate < {MIN_BOOTSTRAP_SUCCESS_RATIO:.0%}). "
            "The confidence intervals may be less reliable."
        )

    lower_p, upper_p = (100 - ci) / 2, 100 - (100 - ci) / 2
    ci_results = {
        "betas_ci": [],
        "breakpoints_ci": [],
        "fit_ci_lower": None,
        "fit_ci_upper": None,
    }

    # Calculate CIs for the fitted line itself
    bootstrap_fits_arr = np.array(bootstrap_fits)
    # Filter out any columns (frequencies) that contain non-finite values
    # before calculating percentiles for the fit CI.
    finite_fit_cols = np.all(np.isfinite(bootstrap_fits_arr), axis=0)
    if np.any(finite_fit_cols):
        ci_results["fit_ci_lower"] = np.percentile(
            bootstrap_fits_arr[:, finite_fit_cols], lower_p, axis=0
        )
        ci_results["fit_ci_upper"] = np.percentile(
            bootstrap_fits_arr[:, finite_fit_cols], upper_p, axis=0
        )

    for i in range(n_breakpoints + 1):
        # Filter out non-finite beta estimates before calculating CIs
        finite_betas = [b for b in bootstrap_betas[i] if np.isfinite(b)]
        if len(finite_betas) >= MIN_BOOTSTRAP_SAMPLES:
            lower = np.percentile(finite_betas, lower_p)
            upper = np.percentile(finite_betas, upper_p)
            ci_results["betas_ci"].append((lower, upper))
        else:
            ci_results["betas_ci"].append((np.nan, np.nan))

    for i in range(n_breakpoints):
        # Filter out non-finite breakpoint estimates
        finite_bps = [bp for bp in bootstrap_breakpoints[i] if np.isfinite(bp)]
        if len(finite_bps) >= MIN_BOOTSTRAP_SAMPLES:
            lower = np.percentile(finite_bps, lower_p)
            upper = np.percentile(finite_bps, upper_p)
            ci_results["breakpoints_ci"].append((lower, upper))
        else:
            ci_results["breakpoints_ci"].append((np.nan, np.nan))

    return ci_results


def _extract_parametric_segmented_cis(pw_fit, n_breakpoints, ci=95, logger=None):
    """
    Extracts parametric CIs from a fitted piecewise_regression model.

    Note: The library provides CIs for all slopes (alphas) and breakpoints.
    This function extracts them.
    """
    logger = logger or logging.getLogger(__name__)
    msg = (
        "Parametric confidence intervals for segmented models assume normality "
        "of errors and may be less reliable than bootstrap intervals. "
        "Consider using ci_method='bootstrap' for more robust results."
    )
    logger.warning(msg)

    results = pw_fit.get_results()
    if not results or "estimates" not in results:
        return {"betas_ci": [], "breakpoints_ci": []}

    estimates = results["estimates"]
    betas_ci = []
    breakpoints_ci = []

    # Bug fix: The piecewise-regression library provides CIs for all alphas.
    # The previous implementation only extracted the first one. This loop
    # now correctly extracts CIs for all segment slopes.
    for i in range(1, n_breakpoints + 2):
        alpha_key = f"alpha{i}"
        alpha_info = estimates.get(alpha_key, {})
        alpha_ci = alpha_info.get("confidence_interval")

        if alpha_ci and all(c is not None for c in alpha_ci):
            # The CI for beta is inverted because beta = -slope. A lower bound
            # on the slope corresponds to an upper bound on beta.
            betas_ci.append((-alpha_ci[1], -alpha_ci[0]))
        else:
            betas_ci.append((np.nan, np.nan))

    # CIs for the breakpoints
    for i in range(1, n_breakpoints + 1):
        bp_info = estimates.get(f"breakpoint{i}", {})
        bp_ci_log = bp_info.get("confidence_interval")
        if bp_ci_log and all(c is not None for c in bp_ci_log):
            # Convert from log space back to frequency space
            breakpoints_ci.append((np.exp(bp_ci_log[0]), np.exp(bp_ci_log[1])))
        else:
            breakpoints_ci.append((np.nan, np.nan))

    return {"betas_ci": betas_ci, "breakpoints_ci": breakpoints_ci}


def fit_segmented_spectrum(
    frequency: np.ndarray,
    power: np.ndarray,
    n_breakpoints: int = 1,
    p_threshold: float = 0.05,
    ci_method: str = "bootstrap",
    bootstrap_type: str = "residuals",
    n_bootstraps: int = 1000,
    ci: int = 95,
    seed: Optional[int] = None,
    logger: Optional[logging.Logger] = None,
) -> Dict:
    """
    Fits a segmented regression and estimates confidence intervals.

    Args:
        frequency (np.ndarray): The frequency array.
        power (np.ndarray): The power array.
        n_breakpoints (int, optional): The number of breakpoints to fit.
            Defaults to 1.
        p_threshold (float, optional): The p-value threshold for the Davies
            test for a significant breakpoint (only for 1-breakpoint models).
            Defaults to 0.05.
        ci_method (str, optional): The method for calculating confidence
            intervals ('bootstrap' or 'parametric'). Defaults to 'bootstrap'.
        bootstrap_type (str, optional): The bootstrap method to use. Can be
            'pairs' (default), which resamples (x, y) pairs, or 'residuals',
            which resamples the model residuals.
        n_bootstraps (int, optional): Number of bootstrap samples for CI.
            Only used if `ci_method` is `'bootstrap'`. Defaults to 1000.
        ci (int, optional): The desired confidence interval in percent.
            Defaults to 95.
        seed (int, optional): A seed for the random number generator.
        logger (logging.Logger, optional): A logger for warnings.

    Returns:
        dict: A dictionary containing the fit results, including CIs.
    """
    logger = logger or logging.getLogger(__name__)
    if piecewise_regression is None:
        logger.warning(
            "Segmented fitting failed because the 'piecewise-regression' "
            "package is not installed. Falling back to a standard linear fit. "
            "To enable segmented fitting, please install the package, e.g., "
            "with 'pip install piecewise-regression'."
        )
        return fit_standard_model(
            frequency,
            power,
            method="theil-sen",  # Default to a robust method
            ci_method=ci_method,
            n_bootstraps=n_bootstraps,
            ci=ci,
            seed=seed,
            logger=logger,
        )

    # Input validation
    if not isinstance(frequency, np.ndarray) or not isinstance(power, np.ndarray):
        raise TypeError("Input 'frequency' and 'power' must be numpy arrays.")
    if frequency.shape != power.shape:
        raise ValueError("'frequency' and 'power' must have the same shape.")
    if not np.all(np.isfinite(frequency)) or not np.all(np.isfinite(power)):
        raise ValueError("Input arrays must contain finite values.")
    if n_breakpoints <= 0:
        raise ValueError("'n_breakpoints' must be a positive integer.")
    if not 0 < p_threshold < 1:
        raise ValueError("'p_threshold' must be between 0 and 1.")
    if n_bootstraps < 0:
        raise ValueError("'n_bootstraps' must be non-negative.")
    if not 0 < ci < 100:
        raise ValueError("'ci' must be between 0 and 100.")

    # Log-transform the data
    if n_breakpoints > 1:
        msg = (
            f"Fitting a model with {n_breakpoints} breakpoints. "
            "WARNING: Statistical significance is only tested for 1-breakpoint models "
            "(via Davies test). Models with more than one breakpoint are chosen "
            "based on BIC alone, which can lead to overfitting, especially with "
            "noisy data. Interpret these results with caution."
        )
        logger.warning(msg)

    valid_indices = (frequency > 0) & (power > 0)
    min_points = MIN_POINTS_PER_SEGMENT * (n_breakpoints + 1)
    if np.sum(valid_indices) < min_points:
        summary = f"Not enough data points for {n_breakpoints}-breakpoint regression."
        return {"model_summary": summary, "n_breakpoints": n_breakpoints, "bic": np.inf, "aic": np.inf}

    log_freq = np.log(frequency[valid_indices])
    log_power = np.log(power[valid_indices])
    n_points = len(log_power)

    if n_points < 30 and "bootstrap" in ci_method:
        logger.warning(
            f"Dataset has only {n_points} points. Bootstrap CIs may be "
            "unreliable. Consider using parametric CIs or collecting more data."
        )

    # Fit the piecewise regression model
    try:
        pw_fit = piecewise_regression.Fit(
            log_freq, log_power, n_breakpoints=n_breakpoints
        )
        fit_summary = pw_fit.get_results()
        converged = fit_summary["converged"]
    except Exception as e:
        logger.warning(f"Segmented regression failed with an unexpected error: {e}")
        return {
            "model_summary": "Segmented regression failed with an unexpected error.",
            "n_breakpoints": n_breakpoints,
            "bic": np.inf,
            "aic": np.inf,
        }

    # Check for convergence and statistical significance
    davies_p_value = pw_fit.davies if n_breakpoints == 1 else None
    if not converged or (davies_p_value is not None and davies_p_value > p_threshold):
        summary = "Model did not converge"
        if davies_p_value is not None and davies_p_value > p_threshold:
            summary = f"No significant breakpoint found (Davies test p > {p_threshold})"
        return {
            "model_summary": summary,
            "n_breakpoints": n_breakpoints,
            "davies_p_value": davies_p_value,
            "bic": np.inf,
            "aic": np.inf,
        }

    # --- Extract results ---
    fit_summary = pw_fit.get_results()
    estimates = fit_summary["estimates"]
    fitted_log_power = pw_fit.predict(log_freq)

    # Calculate AIC and Adjusted R-squared
    n_params = 2 * (n_breakpoints + 1)  # 2 params (slope, intercept) per segment
    r_squared = fit_summary.get("r_squared", np.nan)
    adj_r_squared = 1 - (1 - r_squared) * (n_points - 1) / (n_points - n_params - 1)
    aic = _calculate_aic(log_power, fitted_log_power, n_params)

    # Store base results
    results = {
        "bic": fit_summary.get("bic"),
        "aic": aic,
        "r_squared": r_squared,
        "adj_r_squared": adj_r_squared,
        "model_summary": str(pw_fit.summary()),
        "model_object": pw_fit,
        "log_freq": log_freq,
        "log_power": log_power,
        "residuals": log_power - fitted_log_power,
        "fitted_log_power": fitted_log_power,
        "n_breakpoints": n_breakpoints,
        "davies_p_value": davies_p_value,
    }

    # --- Extract breakpoints and betas (slopes) ---
    breakpoints = []
    for i in range(1, n_breakpoints + 1):
        bp_log_freq = estimates[f"breakpoint{i}"]["estimate"]
        breakpoints.append(np.exp(bp_log_freq))

    # Check if breakpoints are too close to the boundaries
    log_freq_range = log_freq.max() - log_freq.min()
    boundary_threshold = 0.05  # 5% of the log-frequency range
    for bp_log in [estimates[f"breakpoint{i}"]["estimate"] for i in range(1, n_breakpoints + 1)]:
        if (
            bp_log < log_freq.min() + boundary_threshold * log_freq_range
            or bp_log > log_freq.max() - boundary_threshold * log_freq_range
        ):
            logger.warning(
                "A breakpoint is very close to the data boundary, "
                "which may indicate an unstable fit."
            )
            break  # Only need to warn once

    slopes = []
    current_slope = estimates["alpha1"]["estimate"]
    slopes.append(current_slope)
    for i in range(1, n_breakpoints + 1):
        current_slope += estimates[f"beta{i}"]["estimate"]
        slopes.append(current_slope)

    betas = [-s for s in slopes]

    results["breakpoints"] = breakpoints
    results["betas"] = betas

    # --- Calculate intercepts for each segment ---
    intercepts = []
    if "const" in estimates:
        current_intercept = estimates["const"]["estimate"]
        intercepts.append(current_intercept)
        for i in range(1, n_breakpoints + 1):
            beta_i = estimates[f"beta{i}"]["estimate"]
            breakpoint_i = estimates[f"breakpoint{i}"]["estimate"]
            current_intercept -= beta_i * breakpoint_i
            intercepts.append(current_intercept)
    else:
        logger.warning(
            "Could not find 'const' in piecewise regression estimates. "
            "Intercepts will not be available in the results."
        )
        intercepts = [np.nan] * (n_breakpoints + 1)
    results["intercepts"] = intercepts

    # --- Calculate Confidence Intervals based on the chosen method ---
    if ci_method == "bootstrap":
        if bootstrap_type == "residuals" and durbin_watson is None:
            logger.error(_STATSMODELS_MISSING_MSG)
            raise ImportError(_STATSMODELS_MISSING_MSG)

        if durbin_watson:
            dw_stat = durbin_watson(results["residuals"])
            results["durbin_watson_stat"] = dw_stat
            if not 1.5 < dw_stat < 2.5:
                logger.warning(
                    f"Durbin-Watson statistic is {dw_stat:.2f}, indicating "
                    "potential autocorrelation in the model residuals. "
                    "The 'residuals' bootstrap method is recommended."
                )

        if n_bootstraps > 0:
            ci_results = _bootstrap_segmented_fit(
                pw_fit,
                log_freq,
                log_power,
                n_bootstraps,
                ci,
                seed,
                bootstrap_type=bootstrap_type,
                logger=logger,
            )
            results.update(ci_results)
        else:
            # If bootstrap is chosen but n_bootstraps is 0, return NaNs.
            results["betas_ci"] = [(np.nan, np.nan)] * (n_breakpoints + 1)
            results["breakpoints_ci"] = [(np.nan, np.nan)] * n_breakpoints
    elif ci_method == "parametric":
        ci_results = _extract_parametric_segmented_cis(
            pw_fit, n_breakpoints, ci=ci, logger=logger
        )
        results.update(ci_results)

    return results

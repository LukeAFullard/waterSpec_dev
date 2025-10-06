import logging
import warnings
from typing import Dict, Optional

import numpy as np
from scipy import stats

from .preprocessor import _moving_block_bootstrap_indices

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
            "Near-zero RSS found, indicating a perfect fit. This may be due to "
            "overfitting or numerical instability. Returning a very large "
            "negative BIC to prevent downstream issues with -inf.",
            UserWarning,
        )
        return -1e300
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
    bootstrap_type: str = "block",
    bootstrap_block_size: Optional[int] = None,
    n_bootstraps: int = 200,
    ci: int = 95,
    seed: Optional[int] = None,
    logger: Optional[logging.Logger] = None,
) -> Dict:
    """
    Fits a standard (non-segmented) model to the power spectrum and estimates
    confidence intervals for the spectral exponent (beta).

    This function consolidates the fitting, BIC calculation, and confidence
    interval estimation into a single, robust workflow.

    .. note::
        The spectral exponent, beta (β), is defined as the negative of the
        slope of the log-log power spectrum (P(f) ∝ f^−β). A positive beta
        indicates persistence (long-term memory), where low frequencies have
        more power, while a negative beta indicates anti-persistence.

    Args:
        frequency (np.ndarray): The frequency array.
        power (np.ndarray): The power array.
        method (str, optional): The fitting method ('theil-sen' or 'ols').
        ci_method (str, optional): Method for CI calculation.
        bootstrap_type (str, optional): The bootstrap method to use. Can be
            'pairs', 'residuals', 'block', or 'wild'. 'block' is recommended
            for data with suspected autocorrelation. 'wild' is recommended
            for heteroscedastic residuals.
        bootstrap_block_size (int, optional): The block size for the moving
            block bootstrap. If None, a rule-of-thumb `n_points**(1/3)` is
            used. This default may be too small for data with strong
            autocorrelation. For best results, users should choose a block
            size that reflects the data's correlation length (e.g., ~10x the
            period of the longest significant cycle). Only applicable when
            `bootstrap_type` is 'block'.
        n_bootstraps (int, optional): Number of bootstrap samples.
        ci (int, optional): The desired confidence interval in percent.
        seed (int, optional): A seed for the random number generator.
        logger (logging.Logger, optional): A logger for warnings.
    """
    logger = logger or logging.getLogger(__name__)

    # Ensure frequency and power are sorted by frequency.
    # This prevents issues with downstream operations that assume sorted data.
    order = np.argsort(frequency)
    frequency = frequency[order]
    power = power[order]

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
    if bootstrap_type not in ["pairs", "residuals", "block", "wild"]:
        raise ValueError(
            f"Unknown bootstrap_type: '{bootstrap_type}'. Choose 'pairs', 'residuals', 'block', or 'wild'."
        )
    # Add a small floor to power to prevent log(0) issues with very weak signals.
    power = np.maximum(power, 1e-100)
    valid_indices = (frequency > 0) & (power > 0)
    if np.sum(valid_indices) < 2:
        failure_reason = "Not enough valid (positive) data points to fit the model."
        logger.warning(failure_reason)
        return {
            "beta": np.nan,
            "bic": np.inf,
            "aic": np.inf,
            "beta_ci_lower": np.nan,
            "beta_ci_upper": np.nan,
            "failure_reason": failure_reason,
        }

    log_freq = np.log10(frequency[valid_indices])
    log_power = np.log10(power[valid_indices])
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
            dof = n_points - 2
            if dof > 0:
                adj_r_squared = 1 - (1 - r_squared) * (n_points - 1) / dof
            else:
                adj_r_squared = np.nan
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

    except (ValueError, np.linalg.LinAlgError) as e:
        failure_reason = (
            f"Initial standard model fit failed with method '{method}' due to a numerical or data issue: {e}"
        )
        logger.warning(failure_reason, exc_info=True)
        return {
            "beta": np.nan,
            "bic": np.inf,
            "aic": np.inf,
            "beta_ci_lower": np.nan,
            "beta_ci_upper": np.nan,
            "failure_reason": failure_reason,
        }
    except Exception as e:
        failure_reason = (
            f"An unexpected error occurred during the initial standard model fit with method '{method}': {e!r}"
        )
        logger.error(
            "An unexpected error occurred during the initial standard model fit with method '%s'.",
            method,
            exc_info=True,
        )
        raise RuntimeError(failure_reason) from e

    # 3. Calculate BIC and AIC
    log_power_fit = slope * log_freq + intercept
    n_params = 2
    bic = _calculate_bic(log_power, log_power_fit, n_params)
    aic = _calculate_aic(log_power, log_power_fit, n_params)
    fit_results["bic"] = bic
    fit_results["aic"] = aic

    # 4. Calculate Confidence Intervals
    residuals = log_power - log_power_fit
    beta_ci_lower, beta_ci_upper = np.nan, np.nan
    if ci_method == "bootstrap":
        # Check for autocorrelation in residuals if using residual bootstrap
        if bootstrap_type == "residuals" and durbin_watson is None:
            logger.error(_STATSMODELS_MISSING_MSG)
            raise ImportError(_STATSMODELS_MISSING_MSG)
        if durbin_watson:
            dw_stat = durbin_watson(residuals)
            fit_results["durbin_watson_stat"] = dw_stat
            # A DW statistic between 1.5 and 2.5 is generally considered normal.
            # Values outside this range suggest autocorrelation.
            if not 1.5 < dw_stat < 2.5:
                logger.warning(
                    f"Durbin-Watson statistic is {dw_stat:.2f}, indicating "
                    "potential first-order autocorrelation in the model "
                    "residuals. This test does not detect higher-order "
                    "correlation structures. The 'residuals' bootstrap method "
                    "assumes independent residuals and may produce unreliable "
                    "confidence intervals. Consider using 'block' or 'wild' "
                    "bootstrap if autocorrelation or heteroscedasticity is suspected."
                )

        rng = np.random.default_rng(seed)
        beta_estimates = []
        n_points = len(log_freq)
        error_counts = {}

        if bootstrap_type == "block":
            block_size = bootstrap_block_size
            if block_size is None:
                # Rule-of-thumb for block size, with a minimum of 3 for effectiveness.
                block_size = max(3, int(np.ceil(n_points ** (1 / 3))))
                logger.info(
                    f"No 'bootstrap_block_size' provided for block bootstrap. "
                    f"Using rule-of-thumb size: {block_size}"
                )

            # Ensure block size is valid and not larger than the dataset
            block_size = min(block_size, n_points)
            if n_points < 3 * block_size:
                logger.warning(
                    f"The number of data points ({n_points}) is less than 3 times "
                    f"the block size ({block_size}). The block bootstrap may be "
                    "ineffective."
                )
            if block_size >= n_points:
                logger.warning(
                    f"Block size ({block_size}) is >= number of points "
                    f"({n_points}). This is equivalent to a 'pairs' bootstrap."
                )

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
                elif bootstrap_type == "block":
                    indices = _moving_block_bootstrap_indices(n_points, block_size, rng)
                    resampled_log_freq = log_freq[indices]
                    resampled_log_power = log_power[indices]
                elif bootstrap_type == "wild":
                    # Wild bootstrap using Rademacher distribution, which does
                    # not assume constant variance of residuals.
                    u = rng.choice([-1, 1], size=n_points, replace=True)
                    resampled_log_power = log_power_fit + residuals * u
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
            except (ValueError, np.linalg.LinAlgError) as e:
                error_type = type(e).__name__
                error_counts[error_type] = error_counts.get(error_type, 0) + 1
                # Log the specific error for a failed iteration
                msg = f"Bootstrap iteration failed with a numerical error: {e}"
                if logger:
                    logger.debug(msg)
                continue
            except Exception as e:
                error_type = type(e).__name__
                error_counts[error_type] = error_counts.get(error_type, 0) + 1
                # Log the specific error for a failed iteration
                if logger:
                    logger.debug(
                        "Bootstrap iteration failed with an unexpected error: %s",
                        e,
                        exc_info=True,
                    )
                continue

        MIN_BOOTSTRAP_SAMPLES = 50  # Min samples for a reliable CI
        success_rate = len(beta_estimates) / n_bootstraps if n_bootstraps > 0 else 0
        MIN_SUCCESS_RATE = 0.5

        error_summary = ""
        if error_counts:
            error_summary = ", ".join(
                [f"{err}: {count}" for err, count in error_counts.items()]
            )
            logger.warning(f"Bootstrap errors occurred: {error_summary}")

        fit_results["bootstrap_success_rate"] = success_rate
        fit_results["bootstrap_n_success"] = len(beta_estimates)
        fit_results["bootstrap_error_summary"] = error_summary

        if n_bootstraps > 0 and success_rate < MIN_SUCCESS_RATE:
            raise ValueError(
                f"Bootstrap success rate ({success_rate:.0%}) was below the required threshold ({MIN_SUCCESS_RATE:.0%}). "
                f"Only {len(beta_estimates)}/{n_bootstraps} iterations succeeded. Errors: {error_summary}"
            )

        if len(beta_estimates) < MIN_BOOTSTRAP_SAMPLES:
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
            # Shapiro-Wilk test for normality of residuals
            if len(residuals) > 3:
                shapiro_stat, shapiro_p = stats.shapiro(residuals)
                if shapiro_p < 0.05:
                    logger.warning(
                        f"Residuals may not be normally distributed (Shapiro-Wilk "
                        f"p-value: {shapiro_p:.3f}). Parametric confidence "
                        "intervals may be unreliable."
                    )

            stderr = fit_results.get("stderr", np.nan)
            dof = len(log_freq) - 2
            if np.isfinite(stderr) and dof > 0:
                t_val = stats.t.ppf((1 + ci / 100) / 2, dof)
                half_width = t_val * stderr
                slope_ci_lower, slope_ci_upper = slope - half_width, slope + half_width
                # Note: The CI for beta is inverted because beta = -slope.
                # A lower bound on the slope corresponds to an upper bound on beta.
                beta_ci_lower, beta_ci_upper = -slope_ci_upper, -slope_ci_lower
            elif dof <= 0:
                logger.warning(
                    f"Not enough data points ({len(log_freq)}) to calculate "
                    "parametric confidence intervals (requires > 2). CIs will be NaN."
                )
        elif method == "theil-sen":
            slope_ci_lower = fit_results.get("slope_ci_lower", np.nan)
            slope_ci_upper = fit_results.get("slope_ci_upper", np.nan)
            # Note: The CI for beta is inverted because beta = -slope.
            beta_ci_lower, beta_ci_upper = -slope_ci_upper, -slope_ci_lower
    else:
        raise ValueError(f"Unknown ci_method: '{ci_method}'")

    fit_results.update({"beta_ci_lower": beta_ci_lower, "beta_ci_upper": beta_ci_upper})

    # 5. Check for heteroscedasticity and store supplementary data
    spearman_corr_log_freq, spearman_corr_freq = np.nan, np.nan
    if len(residuals) > 1:
        # Check for correlation between frequency and the magnitude of residuals
        spearman_corr_log_freq, _ = stats.spearmanr(log_freq, np.abs(residuals))
        spearman_corr_freq, _ = stats.spearmanr(10**log_freq, np.abs(residuals))
        if np.abs(spearman_corr_log_freq) > 0.3:
            logger.warning(
                "Residuals show potential heteroscedasticity (Spearman "
                f"correlation with log_freq: {spearman_corr_log_freq:.2f}). The BIC value may be less "
                "reliable for model selection."
            )

    fit_results.update({
        "log_freq": log_freq,
        "log_power": log_power,
        "residuals": residuals,
        "fitted_log_power": log_power_fit,
        "spearman_corr_log_freq": spearman_corr_log_freq,
        "spearman_corr_freq": spearman_corr_freq,
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
    bootstrap_type="block",
    bootstrap_block_size: Optional[int] = None,
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

    # For residual and wild bootstrap, we need the initial fitted values and residuals
    if bootstrap_type in ["residuals", "wild"]:
        initial_fitted_power = pw_fit.predict(log_freq)
        initial_residuals = log_power - initial_fitted_power
    elif bootstrap_type == "block":
        block_size = bootstrap_block_size
        if block_size is None:
            # Rule-of-thumb for block size, with a minimum of 3 for effectiveness.
            block_size = max(3, int(np.ceil(n_points ** (1 / 3))))
            logger.info(
                f"No 'bootstrap_block_size' provided for segmented block "
                f"bootstrap. Using rule-of-thumb size: {block_size}"
            )
        # Ensure block size is valid and not larger than the dataset
        block_size = min(block_size, n_points)
        if n_points < 3 * block_size:
            logger.warning(
                f"The number of data points ({n_points}) is less than 3 times "
                f"the block size ({block_size}). The block bootstrap may be "
                "ineffective."
            )
        if block_size >= n_points:
            logger.warning(
                f"Block size ({block_size}) is >= number of points "
                f"({n_points}). This is equivalent to a 'pairs' bootstrap."
            )

    # Extract the starting breakpoints from the initial fit's results.
    # This was the source of a bug where bootstrap CIs would always fail.
    # The previous code tried to access a non-existent `estimates` attribute
    # on the `pw_fit` object.
    initial_results = pw_fit.get_results()
    initial_estimates = initial_results.get("estimates")

    if not initial_estimates:
        raise RuntimeError(
            "Bootstrap confidence intervals cannot be calculated because the "
            "initial segmented fit did not produce valid estimates. This may "
            "indicate a version incompatibility with piecewise-regression."
        )

    # Extract the initial breakpoint estimates to use as start_values.
    # This helps stabilize the bootstrap fits by starting them from a good
    # initial guess, which is critical for preventing convergence failures.
    start_values = []
    for i in range(1, n_breakpoints + 1):
        bp_info = initial_estimates.get(f"breakpoint{i}", {})
        bp_val = bp_info.get("estimate")
        if bp_val is not None:
            start_values.append(bp_val)
        else:
            # If any breakpoint estimate is missing, it's safer not to provide
            # start_values, as the library expects a full list.
            logger.warning(
                f"Could not extract initial estimate for breakpoint {i}. "
                "Bootstrap iterations will not use start_values, which may "
                "reduce stability."
            )
            start_values = None
            break

    # If start_values list is incomplete, set to None so the library uses its own guess.
    if start_values is not None and len(start_values) != n_breakpoints:
        logger.warning(
            "Could not extract all initial breakpoint estimates. Bootstrap "
            "iterations will not use start_values."
        )
        start_values = None

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
            elif bootstrap_type == "block":
                indices = _moving_block_bootstrap_indices(n_points, block_size, rng)
                resampled_log_freq = log_freq[indices]
                resampled_log_power = log_power[indices]
                # Sort the resampled data by frequency, which is required
                # for the piecewise regression library.
                sort_order = np.argsort(resampled_log_freq)
                resampled_log_freq_sorted = resampled_log_freq[sort_order]
                resampled_log_power_sorted = resampled_log_power[sort_order]
            elif bootstrap_type == "wild":
                # Wild bootstrap using Rademacher distribution
                u = rng.choice([-1, 1], size=n_points, replace=True)
                resampled_log_power_sorted = initial_fitted_power + initial_residuals * u
                resampled_log_freq_sorted = log_freq  # Keep original frequencies
            else:
                continue

            bootstrap_pw_fit = piecewise_regression.Fit(
                resampled_log_freq_sorted,
                resampled_log_power_sorted,
                n_breakpoints=n_breakpoints,
                start_values=start_values,
            )

            # Bug fix: Correctly get results from the bootstrap fit object.
            bootstrap_results = bootstrap_pw_fit.get_results()
            if not bootstrap_results.get("converged"):
                if logger:
                    logger.debug("Segmented bootstrap iteration failed to converge.")
                continue

            estimates = bootstrap_results.get("estimates")
            if not estimates:
                if logger:
                    logger.debug(
                        "Segmented bootstrap iteration converged but returned no estimates."
                    )
                continue

            # Safely extract breakpoints
            for i in range(n_breakpoints):
                bp_info = estimates.get(f"breakpoint{i+1}", {})
                bp_val = bp_info.get("estimate")
                if bp_val is not None:
                    bootstrap_breakpoints[i].append(10**bp_val)
                else:
                    bootstrap_breakpoints[i].append(np.nan)

            # Safely extract slopes
            current_slope = estimates.get("alpha1", {}).get("estimate")
            if current_slope is None:
                continue  # Skip this bootstrap if the first slope is missing

            slopes = [current_slope]
            for i in range(1, n_breakpoints + 1):
                beta_val = estimates.get(f"beta{i}", {}).get("estimate")
                if beta_val is None:
                    slopes = []  # Invalidate the list
                    break
                slopes.append(slopes[-1] + beta_val)

            if not slopes:
                continue  # Skip if any subsequent beta was missing

            for i, slope in enumerate(slopes):
                bootstrap_betas[i].append(-slope)

            # Store the predicted line from this bootstrap sample
            bootstrap_fits.append(bootstrap_pw_fit.predict(log_freq))
            successful_fits += 1
        except (ValueError, np.linalg.LinAlgError, RuntimeError) as e:
            error_type = type(e).__name__
            error_counts[error_type] = error_counts.get(error_type, 0) + 1
            # Log the specific error for a failed iteration
            msg = f"Segmented bootstrap iteration failed with a numerical/runtime error: {e}"
            if logger:
                logger.debug(msg)
            continue
        except Exception as e:
            error_type = type(e).__name__
            error_counts[error_type] = error_counts.get(error_type, 0) + 1
            if logger:
                logger.debug(
                    "An unexpected error occurred in a segmented bootstrap iteration: %s",
                    e,
                    exc_info=True,
                )
            continue

    MIN_BOOTSTRAP_SAMPLES = 50  # Min samples for a reliable CI
    success_rate = successful_fits / n_bootstraps if n_bootstraps > 0 else 0
    # Raise an error if the success rate is below 80%, as high failure rates
    # can lead to unreliable CIs.
    MIN_SUCCESS_RATE = 0.8

    error_summary = ""
    if error_counts:
        error_summary = ", ".join(
            [f"{err}: {count}" for err, count in error_counts.items()]
        )
        logger.warning(f"Bootstrap errors occurred: {error_summary}")

    if n_bootstraps > 0 and success_rate < MIN_SUCCESS_RATE:
        # This error is caught by the calling function to trigger a fallback.
        raise ValueError(
            f"Bootstrap success rate ({success_rate:.0%}) was below the required threshold ({MIN_SUCCESS_RATE:.0%}). "
            f"Only {successful_fits}/{n_bootstraps} iterations succeeded. Errors: {error_summary}"
        )

    if successful_fits < MIN_BOOTSTRAP_SAMPLES:
        logger.warning(
            f"Only {successful_fits}/{n_bootstraps} bootstrap iterations "
            f"succeeded for the segmented model (minimum required: {MIN_BOOTSTRAP_SAMPLES}). "
            "Confidence intervals are unreliable and will be set to NaN."
        )
        # This case is now less likely to be hit first, but is kept as a
        # safeguard for when n_bootstraps is very low.
        return {
            "betas_ci": [(np.nan, np.nan)] * (n_breakpoints + 1),
            "breakpoints_ci": [(np.nan, np.nan)] * n_breakpoints,
            "fit_ci_lower": None,
            "fit_ci_upper": None,
        }

    lower_p, upper_p = (100 - ci) / 2, 100 - (100 - ci) / 2
    ci_results = {
        "betas_ci": [],
        "breakpoints_ci": [],
        "fit_ci_lower": None,
        "fit_ci_upper": None,
    }

    # Calculate CIs for the fitted line itself
    if not bootstrap_fits:
        logger.warning(
            "All bootstrap iterations failed for the segmented model; "
            "the confidence interval of the fit line cannot be calculated."
        )
        bootstrap_fits_arr = np.array([])
    else:
        bootstrap_fits_arr = np.array(bootstrap_fits)

    # This check handles cases where bootstrap runs failed and produced no fits.
    if bootstrap_fits_arr.ndim == 2 and bootstrap_fits_arr.shape[1] > 0:
        # Initialize full-size arrays with NaNs. This ensures that if the
        # bootstrap fails for some frequencies, those points are excluded from
        # plotting without causing a dimension mismatch.
        n_freq_points = bootstrap_fits_arr.shape[1]
        fit_ci_lower = np.full(n_freq_points, np.nan)
        fit_ci_upper = np.full(n_freq_points, np.nan)

        # Identify columns (frequencies) where all bootstrap fits were finite.
        finite_fit_cols = np.all(np.isfinite(bootstrap_fits_arr), axis=0)

        # Calculate percentiles only for the valid columns.
        if np.any(finite_fit_cols):
            # Calculate CIs on the subset of data that is valid.
            lower_bounds = np.percentile(
                bootstrap_fits_arr[:, finite_fit_cols], lower_p, axis=0
            )
            upper_bounds = np.percentile(
                bootstrap_fits_arr[:, finite_fit_cols], upper_p, axis=0
            )
            # Place the calculated CIs back into the full-size arrays.
            fit_ci_lower[finite_fit_cols] = lower_bounds
            fit_ci_upper[finite_fit_cols] = upper_bounds

        ci_results["fit_ci_lower"] = fit_ci_lower
        ci_results["fit_ci_upper"] = fit_ci_upper
    # If bootstrap_fits_arr is empty or not 2D, CIs will remain as None,
    # which is the default initialized value.

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
            try:
                lower = np.percentile(finite_bps, lower_p)
                upper = np.percentile(finite_bps, upper_p)
                ci_results["breakpoints_ci"].append((lower, upper))
            except IndexError:
                logger.warning(f"Could not calculate bootstrap CI for breakpoint {i+1} due to an index error.")
                ci_results["breakpoints_ci"].append((np.nan, np.nan))
        else:
            logger.warning(
                f"Not enough successful bootstrap samples to calculate confidence interval for breakpoint {i+1} "
                f"(requires {MIN_BOOTSTRAP_SAMPLES}, got {len(finite_bps)}). CI will be NaN."
            )
            ci_results["breakpoints_ci"].append((np.nan, np.nan))

    ci_results["bootstrap_success_rate"] = success_rate
    ci_results["bootstrap_n_success"] = successful_fits
    ci_results["bootstrap_error_summary"] = error_summary
    return ci_results


def _extract_parametric_segmented_cis(
    pw_fit, residuals, n_breakpoints, ci=95, logger=None
):
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

    # Shapiro-Wilk test for normality of residuals
    if len(residuals) > 3:
        shapiro_stat, shapiro_p = stats.shapiro(residuals)
        if shapiro_p < 0.05:
            logger.warning(
                f"Residuals may not be normally distributed (Shapiro-Wilk "
                f"p-value: {shapiro_p:.3f}). Parametric confidence "
                "intervals may be unreliable."
            )

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
        try:
            bp_info = estimates.get(f"breakpoint{i}", {})
            bp_ci_log = bp_info.get("confidence_interval")

            if bp_ci_log and all(c is not None for c in bp_ci_log) and len(bp_ci_log) == 2:
                # Convert from log space back to frequency space
                breakpoints_ci.append((10 ** bp_ci_log[0], 10 ** bp_ci_log[1]))
            else:
                logger.warning(
                    f"Could not extract valid parametric confidence interval for breakpoint {i}. "
                    f"CI data was: {bp_ci_log}"
                )
                breakpoints_ci.append((np.nan, np.nan))
        except (TypeError, IndexError) as e:
            logger.warning(
                f"An error occurred while extracting parametric confidence interval for breakpoint {i}: {e}"
            )
            breakpoints_ci.append((np.nan, np.nan))

    return {"betas_ci": betas_ci, "breakpoints_ci": breakpoints_ci}


def fit_segmented_spectrum(
    frequency: np.ndarray,
    power: np.ndarray,
    n_breakpoints: int = 1,
    p_threshold: float = 0.05,
    ci_method: str = "bootstrap",
    bootstrap_type: str = "block",
    bootstrap_block_size: Optional[int] = None,
    n_bootstraps: int = 200,
    ci: int = 95,
    seed: Optional[int] = None,
    logger: Optional[logging.Logger] = None,
) -> Dict:
    """
    Fits a segmented regression and estimates confidence intervals.

    .. note::
        The spectral exponent, beta (β), is defined as the negative of the
        slope of the log-log power spectrum (P(f) ∝ f^−β). A positive beta
        indicates persistence (long-term memory), where low frequencies have
        more power, while a negative beta indicates anti-persistence.

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
            'pairs', 'residuals', 'block', or 'wild'. 'block' is recommended
            for data with suspected autocorrelation. 'wild' is recommended for
            heteroscedastic residuals.
        bootstrap_block_size (int, optional): The block size for the moving
            block bootstrap. If None, a rule-of-thumb `n_points**(1/3)` is
            used. This default may be too small for data with strong
            autocorrelation. For best results, users should choose a block
            size that reflects the data's correlation length (e.g., ~10x the
            period of the longest significant cycle). Only applicable when
            `bootstrap_type` is 'block'.
        n_bootstraps (int, optional): Number of bootstrap samples for CI.
            Only used if `ci_method` is `'bootstrap'`. Defaults to 1000.
        ci (int, optional): The desired confidence interval in percent.
            Defaults to 95.
        seed (int, optional): A seed for the random number generator.
        logger (logging.Logger, optional): A logger for warnings.

    Returns:
        dict: A dictionary containing the fit results, including CIs.

    Notes:
        While the function enforces a minimum of 10 data points per segment
        (`MIN_POINTS_PER_SEGMENT`), this is a technical minimum for the fit to
        run and does not guarantee statistical power. For reliable results, a
        larger sample size is strongly recommended.

        - **1-breakpoint models:** A minimum of 50 data points is recommended to
          reliably detect a change in slope and calculate stable confidence
          intervals.
        - **2-breakpoint models:** A minimum of 100 data points is recommended.

        Insufficient data can lead to models that fail to detect true
        breakpoints or produce wide, unreliable confidence intervals.

    Warning:
        For models with `n_breakpoints > 1`, this function does not perform a
        statistical significance test for the breakpoints (such as the Davies
        test). Model selection is based on the Bayesian Information Criterion
        (BIC) alone. This approach may lead to overfitting, particularly with
        noisy data, as BIC might favor more complex models that do not
        represent a statistically significant improvement. Users should interpret
        multi-breakpoint models with caution and consider the physical context.
    """
    logger = logger or logging.getLogger(__name__)

    # Ensure frequency and power are sorted by frequency.
    # This prevents issues with downstream operations that assume sorted data.
    order = np.argsort(frequency)
    frequency = frequency[order]
    power = power[order]

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
    if bootstrap_type not in ["pairs", "residuals", "block", "wild"]:
        raise ValueError(
            f"Unknown bootstrap_type: '{bootstrap_type}'. Choose 'pairs', 'residuals', 'block', or 'wild'."
        )
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

    # Add a small floor to power to prevent log(0) issues with very weak signals.
    power = np.maximum(power, 1e-100)
    valid_indices = (frequency > 0) & (power > 0)
    min_points = MIN_POINTS_PER_SEGMENT * (n_breakpoints + 1)
    if np.sum(valid_indices) < min_points:
        failure_reason = (
            f"Not enough data points ({np.sum(valid_indices)}) for a "
            f"{n_breakpoints}-breakpoint regression (requires at least "
            f"{min_points})."
        )
        logger.warning(failure_reason)
        return {
            "failure_reason": failure_reason,
            "n_breakpoints": n_breakpoints,
            "bic": np.inf,
            "aic": np.inf,
        }

    log_freq = np.log10(frequency[valid_indices])
    log_power = np.log10(power[valid_indices])
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
        converged = fit_summary.get("converged", False)
    except (ValueError, RuntimeError, np.linalg.LinAlgError) as e:
        failure_reason = f"Segmented regression failed with a numerical or data issue: {e}"
        logger.warning(failure_reason)
        return {
            "failure_reason": failure_reason,
            "n_breakpoints": n_breakpoints,
            "bic": np.inf,
            "aic": np.inf,
        }
    except Exception as e:
        failure_reason = f"Segmented regression failed with an unexpected error: {e!r}"
        logger.error(
            "Segmented regression failed with an unexpected error.", exc_info=True
        )
        raise RuntimeError(failure_reason) from e

    # Check for convergence and statistical significance.
    # The Davies test p-value may not always be available.
    try:
        davies_p_value = pw_fit.davies
        if davies_p_value is not None and n_breakpoints > 1:
            logger.info(
                f"Davies test p-value ({davies_p_value:.3f}) was found for a "
                f"{n_breakpoints}-breakpoint model. This is unusual but will be stored."
            )
    except AttributeError:
        davies_p_value = None
        if n_breakpoints == 1:
            logger.warning(
                "Could not access Davies test p-value for the 1-breakpoint model. "
                "The 'piecewise-regression' library version may have changed."
            )

    if not converged or (davies_p_value is not None and davies_p_value > p_threshold):
        failure_reason = "Model did not converge."
        if davies_p_value is not None and davies_p_value > p_threshold:
            failure_reason = (
                f"No significant breakpoint found (Davies test p-value "
                f"{davies_p_value:.3f} > {p_threshold})."
            )
        logger.warning(failure_reason)
        return {
            "failure_reason": failure_reason,
            "n_breakpoints": n_breakpoints,
            "davies_p_value": davies_p_value,
            "bic": np.inf,
            "aic": np.inf,
        }

    # --- Extract results ---
    estimates = fit_summary.get("estimates")
    if not estimates:
        failure_reason = "Fit converged but no estimates were returned."
        logger.warning(failure_reason)
        return {
            "failure_reason": failure_reason,
            "n_breakpoints": n_breakpoints,
            "bic": np.inf,
            "aic": np.inf,
        }

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
    log_breakpoints = []
    for i in range(1, n_breakpoints + 1):
        bp_info = estimates.get(f"breakpoint{i}", {})
        bp_log_freq = bp_info.get("estimate")
        if bp_log_freq is not None:
            breakpoints.append(10**bp_log_freq)
            log_breakpoints.append(bp_log_freq)
        else:
            breakpoints.append(np.nan)
            log_breakpoints.append(np.nan)

    # Check if any breakpoints are too close to the boundaries
    log_freq_range = log_freq.max() - log_freq.min()
    boundary_threshold = 0.05  # 5% of the log-frequency range
    boundary_violations = []
    for i, bp_log in enumerate(log_breakpoints):
        if np.isfinite(bp_log) and (
            bp_log < log_freq.min() + boundary_threshold * log_freq_range
            or bp_log > log_freq.max() - boundary_threshold * log_freq_range
        ):
            boundary_violations.append(i + 1)
    if boundary_violations:
        logger.warning(
            f"Breakpoint(s) {boundary_violations} are very close to the data "
            "boundaries (within 5% of the log-frequency range), which may "
            "indicate an unstable fit."
        )

    slopes = []
    current_slope = estimates.get("alpha1", {}).get("estimate")
    if current_slope is not None:
        slopes.append(current_slope)
        for i in range(1, n_breakpoints + 1):
            beta_val = estimates.get(f"beta{i}", {}).get("estimate")
            if beta_val is None:
                slopes = []  # Invalidate slopes
                break
            current_slope += beta_val
            slopes.append(current_slope)

    betas = [-s for s in slopes] if slopes else [np.nan] * (n_breakpoints + 1)

    results["breakpoints"] = breakpoints
    results["betas"] = betas

    # --- Calculate intercepts for each segment ---
    intercepts = []
    const_info = estimates.get("const", {})
    current_intercept = const_info.get("estimate")

    if current_intercept is not None:
        intercepts.append(current_intercept)
        for i in range(1, n_breakpoints + 1):
            beta_info = estimates.get(f"beta{i}", {})
            beta_i = beta_info.get("estimate")
            bp_info = estimates.get(f"breakpoint{i}", {})
            breakpoint_i = bp_info.get("estimate")

            if beta_i is None or breakpoint_i is None:
                current_intercept = np.nan
            else:
                current_intercept -= beta_i * breakpoint_i
            intercepts.append(current_intercept)
    else:
        logger.warning(
            "Could not find 'const' in piecewise regression estimates. "
            "Intercepts will not be available in the results."
        )
        intercepts = [np.nan] * (n_breakpoints + 1)
    results["intercepts"] = intercepts

    # --- Check for heteroscedasticity ---
    residuals = results["residuals"]
    spearman_corr_log_freq, spearman_corr_freq = np.nan, np.nan
    if len(residuals) > 1:
        spearman_corr_log_freq, _ = stats.spearmanr(log_freq, np.abs(residuals))
        spearman_corr_freq, _ = stats.spearmanr(10**log_freq, np.abs(residuals))
        if np.abs(spearman_corr_log_freq) > 0.3:
            logger.warning(
                "Residuals show potential heteroscedasticity (Spearman "
                f"correlation with log_freq: {spearman_corr_log_freq:.2f}). The BIC value may be less "
                "reliable for model selection."
            )
    results["spearman_corr_log_freq"] = spearman_corr_log_freq
    results["spearman_corr_freq"] = spearman_corr_freq

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
                    "potential first-order autocorrelation in the model "
                    "residuals. This test does not detect higher-order "
                    "correlation structures. The 'residuals' bootstrap method "
                    "assumes independent residuals and may produce unreliable "
                    "confidence intervals. Consider using 'block' or 'wild' "
                    "bootstrap if autocorrelation or heteroscedasticity is suspected."
                )

        if n_bootstraps > 0:
            try:
                ci_results = _bootstrap_segmented_fit(
                    pw_fit,
                    log_freq,
                    log_power,
                    n_bootstraps,
                    ci,
                    seed,
                    bootstrap_type=bootstrap_type,
                    bootstrap_block_size=bootstrap_block_size,
                    logger=logger,
                )
                results.update(ci_results)
            except ValueError as e:
                logger.warning(
                    f"Segmented bootstrap failed due to a high error rate: {e}. "
                    "Falling back to parametric confidence intervals, which may be less reliable."
                )
                ci_results = _extract_parametric_segmented_cis(
                    pw_fit, results["residuals"], n_breakpoints, ci=ci, logger=logger
                )
                results.update(ci_results)
                # Add a field to indicate that a fallback occurred.
                results["ci_method_fallback"] = "parametric"
        else:
            # If bootstrap is chosen but n_bootstraps is 0, return NaNs.
            results["betas_ci"] = [(np.nan, np.nan)] * (n_breakpoints + 1)
            results["breakpoints_ci"] = [(np.nan, np.nan)] * n_breakpoints
    elif ci_method == "parametric":
        ci_results = _extract_parametric_segmented_cis(
            pw_fit, results["residuals"], n_breakpoints, ci=ci, logger=logger
        )
        results.update(ci_results)

    return results

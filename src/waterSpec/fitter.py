import logging
import traceback
import warnings
from typing import Dict, Optional

import numpy as np
from numpy.random import SeedSequence
from scipy import stats

# Import MannKS for robust linear and segmented fitting
import MannKS

from .preprocessor import _moving_block_bootstrap_indices
from .utils import make_rng, spawn_generators

MIN_BOOTSTRAP_SAMPLES = 50

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
            "RSS extremely small; excluding from BIC comparison.", UserWarning
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
    bootstrap_type: str = "block",
    bootstrap_block_size: Optional[int] = None,
    n_bootstraps: int = 2000,
    ci: int = 95,
    seed: Optional[np.random.SeedSequence] = None,
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

    # Pre-fit check for non-finite values in inputs
    finite_mask = np.isfinite(frequency) & np.isfinite(power)
    if not np.all(finite_mask):
        n_total = len(frequency)
        n_non_finite = n_total - np.sum(finite_mask)
        logger.warning(
            f"Input arrays contain {n_non_finite}/{n_total} non-finite "
            f"values, which will be ignored."
        )
        frequency = frequency[finite_mask]
        power = power[finite_mask]

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
    if ci_method not in ["parametric", "bootstrap"]:
        raise ValueError(f"Unknown ci_method: '{ci_method}'. Choose 'parametric' or 'bootstrap'.")

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

    # 2. Perform the fitting
    fit_results = {}

    # Check if we should use MannKS (for robust method)
    if method == "theil-sen":
        try:
            mannks_block_size = 'auto'
            if bootstrap_type == 'block' and bootstrap_block_size is not None:
                mannks_block_size = bootstrap_block_size

            mannks_seed = None
            if isinstance(seed, (int, np.integer)):
                mannks_seed = int(seed)
            elif isinstance(seed, np.random.SeedSequence):
                mannks_seed = int(seed.generate_state(1)[0])

            # MannKS.trend_test provides robust slope and CIs
            res = MannKS.trend_test(
                log_power,
                log_freq,
                alpha=1-(ci/100),
                block_size=mannks_block_size,
                n_bootstrap=n_bootstraps,
                random_state=mannks_seed
            )

            slope = res.slope
            intercept = res.intercept
            slope_ci_lower = res.lower_ci
            slope_ci_upper = res.upper_ci

            fit_results.update({
                "beta": -slope,
                "intercept": intercept,
                "beta_ci_lower": -slope_ci_upper,
                "beta_ci_upper": -slope_ci_lower,
                "slope_ci_lower": slope_ci_lower,
                "slope_ci_upper": slope_ci_upper,
                "ci_computed": True,
                # Add placeholders for consistency
                "bootstrap_success_rate": np.nan,
                "bootstrap_n_success": n_bootstraps,
                "bootstrap_error_summary": "",
            })

            # Calculate fitted values for residuals
            log_power_fit = slope * log_freq + intercept
            residuals = log_power - log_power_fit

            # Use MannKS results to bypass the manual bootstrap/CI block
            # But we still need BIC/AIC and heteroscedasticity checks

            n_params = 2
            bic = _calculate_bic(log_power, log_power_fit, n_params)
            aic = _calculate_aic(log_power, log_power_fit, n_params)
            fit_results["bic"] = bic
            fit_results["aic"] = aic

            # Add supplemental data
            spearman_corr_log_freq, spearman_corr_freq = np.nan, np.nan
            if len(residuals) > 1:
                spearman_corr_log_freq, _ = stats.spearmanr(log_freq, np.abs(residuals))
                spearman_corr_freq, _ = stats.spearmanr(10**log_freq, np.abs(residuals))

            fit_results.update({
                "log_freq": log_freq,
                "log_power": log_power,
                "residuals": residuals,
                "fitted_log_power": log_power_fit,
                "spearman_corr_log_freq": spearman_corr_log_freq,
                "spearman_corr_freq": spearman_corr_freq,
            })

            if durbin_watson:
                dw_stat = durbin_watson(residuals)
                fit_results["durbin_watson_stat"] = dw_stat

            return fit_results

        except Exception as e:
            logger.warning(
                f"MannKS fit failed: {e}. Falling back to standard implementation."
            )
            # Fall through to existing implementation if MannKS fails
            pass

    # --- Standard implementation (fallback or OLS) ---

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
            # Fallback Theil-Sen if MannKS failed
            res = stats.theilslopes(log_power, log_freq, alpha=1 - (ci / 100))
            slope, intercept, low_slope, high_slope = res
            fit_results.update({"slope_ci_lower": low_slope, "slope_ci_upper": high_slope})

        fit_results.update({"beta": -slope, "intercept": intercept})

    except (ValueError, np.linalg.LinAlgError) as e:
        failure_reason = (
            f"Initial standard model fit failed with method '{method}' due to a numerical or data issue: {e}"
        )
        logger.warning(failure_reason, exc_info=True)
        result = {
            "beta": np.nan,
            "bic": np.inf,
            "aic": np.inf,
            "beta_ci_lower": np.nan,
            "beta_ci_upper": np.nan,
            "failure_reason": failure_reason,
        }
        if logger.isEnabledFor(logging.DEBUG):
            result["traceback"] = traceback.format_exc()
        return result
    except Exception as e:
        failure_reason = (
            f"An unexpected error occurred during the initial standard model fit with method '{method}': {e!r}"
        )
        logger.error(
            "Standard model fit crashed: %s",
            e,
            exc_info=True,
        )
        result = {
            "beta": np.nan,
            "bic": np.inf,
            "aic": np.inf,
            "beta_ci_lower": np.nan,
            "beta_ci_upper": np.nan,
            "failure_reason": failure_reason,
        }
        if logger.isEnabledFor(logging.DEBUG):
            result["traceback"] = traceback.format_exc()
        return result

    # 3. Calculate BIC and AIC
    log_power_fit = slope * log_freq + intercept
    n_params = 2
    bic = _calculate_bic(log_power, log_power_fit, n_params)
    aic = _calculate_aic(log_power, log_power_fit, n_params)
    fit_results["bic"] = bic
    fit_results["aic"] = aic

    # 4. Calculate Confidence Intervals
    residuals = log_power - log_power_fit

    # BUG #1 FIX: Validate residuals before bootstrap
    if not np.all(np.isfinite(residuals)):
        msg = "Residuals contain non-finite values; cannot perform bootstrap."
        logger.error(msg)
        # ensure consistent result dict keys (match successful return)
        fit_results.setdefault("beta", np.nan)
        fit_results["beta_ci_lower"] = np.nan
        fit_results["beta_ci_upper"] = np.nan
        fit_results["failure_reason"] = "non_finite_residuals"
        fit_results["ci_computed"] = False
        return fit_results

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

        rng = make_rng(seed)
        beta_estimates = []
        n_points = len(log_freq)
        error_counts = {}

        if bootstrap_type == "block":
            if bootstrap_block_size is None:
                # Rule-of-thumb for block size, with a minimum of 3 for effectiveness.
                block_size = max(3, int(np.ceil(n_points ** (1 / 3))))
                logger.info(
                    f"No 'bootstrap_block_size' provided for block bootstrap. "
                    f"Using rule-of-thumb size: {block_size}"
                )
            else:
                if not isinstance(bootstrap_block_size, int) or bootstrap_block_size <= 0:
                    raise ValueError("bootstrap_block_size must be a positive integer.")
                if bootstrap_block_size >= n_points:
                    raise ValueError(
                        f"Block size ({bootstrap_block_size}) must be smaller than the "
                        f"number of data points ({n_points})."
                    )
                block_size = bootstrap_block_size
            if n_points < 3 * block_size:
                raise ValueError(
                    f"The number of data points ({n_points}) is less than 3 times "
                    f"the block size ({block_size}). The block bootstrap is "
                    "ineffective for such short series."
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
                    centered_residuals = residuals - np.mean(residuals)
                    resampled_log_power = log_power_fit + centered_residuals * u
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
            logger.error(
                "Bootstrap success rate (%s) was below the required threshold (%s). "
                "Only %d/%d iterations succeeded. Errors: %s. CIs will be unreliable.",
                f"{success_rate:.0%}",
                f"{MIN_SUCCESS_RATE:.0%}",
                len(beta_estimates),
                n_bootstraps,
                error_summary,
            )
            fit_results["failure_reason"] = (
                f"Bootstrap success rate ({success_rate:.0%}) was below the required threshold ({MIN_SUCCESS_RATE:.0%}). "
                f"Errors: {error_summary}"
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
            if 3 < len(residuals) <= 5000:
                shapiro_stat, shapiro_p = stats.shapiro(residuals)
                if shapiro_p < 0.05:
                    logger.warning(
                        f"Residuals may not be normally distributed (Shapiro-Wilk "
                        f"p-value: {shapiro_p:.3f}). Parametric confidence "
                        "intervals may be unreliable."
                    )
            elif len(residuals) > 5000:
                logger.info("Dataset too large for Shapiro-Wilk test; skipping normality check.")

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


def fit_segmented_spectrum(
    frequency: np.ndarray,
    power: np.ndarray,
    n_breakpoints: int = 1,
    p_threshold: float = 0.05,
    ci_method: str = "bootstrap",
    bootstrap_type: str = "block",
    bootstrap_block_size: Optional[int] = None,
    n_bootstraps: int = 2000,
    ci: int = 95,
    seed: Optional[np.random.SeedSequence] = None,
    logger: Optional[logging.Logger] = None,
) -> Dict:
    """
    Fits a segmented regression and estimates confidence intervals using MannKS.
    """
    if logger is None:
        import logging
        logger = logging.getLogger(__name__)

    # Ensure frequency and power are sorted by frequency.
    # This prevents issues with downstream operations that assume sorted data.
    order = np.argsort(frequency)
    frequency = frequency[order]
    power = power[order]

    # Input validation
    if not isinstance(frequency, np.ndarray) or not isinstance(power, np.ndarray):
        raise TypeError("Input 'frequency' and 'power' must be numpy arrays.")
    if frequency.shape != power.shape:
        raise ValueError("'frequency' and 'power' must have the same shape.")

    # Pre-fit check for non-finite values in inputs
    finite_mask = np.isfinite(frequency) & np.isfinite(power)
    if not np.all(finite_mask):
        n_total = len(frequency)
        n_non_finite = n_total - np.sum(finite_mask)
        logger.warning(
            f"Input arrays contain {n_non_finite}/{n_total} non-finite "
            f"values, which will be ignored."
        )
        frequency = frequency[finite_mask]
        power = power[finite_mask]

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

    # Use MannKS.segmented_trend_test
    try:
        mannks_seed = None
        if isinstance(seed, (int, np.integer)):
            mannks_seed = int(seed)
        elif isinstance(seed, np.random.SeedSequence):
            mannks_seed = int(seed.generate_state(1)[0])

        # Pass n_bootstrap if ci_method is bootstrap?
        # MannKS uses bagging which involves bootstrapping.
        # It also has n_bootstrap argument.

        mannks_block_size = 'auto'
        if bootstrap_type == 'block' and bootstrap_block_size is not None:
            mannks_block_size = bootstrap_block_size

        res = MannKS.segmented_trend_test(
            log_power,
            log_freq,
            n_breakpoints=n_breakpoints,
            alpha=1-(ci/100),
            n_bootstrap=n_bootstraps,
            random_state=mannks_seed,
            block_size=mannks_block_size
        )

        # Extract results
        breakpoints = res.breakpoints

        # MannKS returns breakpoints in time units (log_freq).
        # We need to convert them back to linear frequency for the result dictionary.
        linear_breakpoints = 10**breakpoints if breakpoints is not None else []

        # Segments DataFrame
        segments_df = res.segments

        slopes = segments_df['slope'].values
        intercepts = segments_df['intercept'].values

        # Betas are negative slopes
        betas = -slopes

        # CIs
        # Slope CIs are in segments_df
        lower_cis = segments_df['lower_ci'].values
        upper_cis = segments_df['upper_ci'].values

        # Beta CIs (inverted slope CIs)
        betas_ci = list(zip(-upper_cis, -lower_cis))

        # Breakpoint CIs
        # res.breakpoint_cis contains tuples of (lower, upper) in log_freq domain
        # Convert to linear frequency
        bp_cis = []
        if res.breakpoint_cis:
            for lower, upper in res.breakpoint_cis:
                bp_cis.append((10**lower, 10**upper))
        else:
            bp_cis = [(np.nan, np.nan)] * n_breakpoints

        # Calculate fitted values for residuals
        fitted_log_power = np.zeros_like(log_power)

        # Reconstruct fitted line
        # This is a bit tricky since we have multiple segments.
        # We can use the breakpoints to determine which segment applies.
        # However, MannKS results should ideally provide fitted values.
        # It doesn't seem to.
        # But we have slopes and intercepts for each segment.
        # Wait, the intercepts in MannKS segments DF are "intercept".
        # Are they for the full line equation y = mx + c valid for that segment?
        # Let's assume yes.

        sorted_bp = np.sort(breakpoints)
        # Add bounds
        bounds = np.concatenate([[-np.inf], sorted_bp, [np.inf]])

        for i in range(len(slopes)):
            mask = (log_freq > bounds[i]) & (log_freq <= bounds[i+1])
            # Handle first point inclusively if needed, or strictly
            if i == 0:
                mask = (log_freq >= bounds[i]) & (log_freq <= bounds[i+1])

            fitted_log_power[mask] = slopes[i] * log_freq[mask] + intercepts[i]

        residuals = log_power - fitted_log_power

        results = {
            "bic": res.bic,
            "aic": res.aic,
            "n_breakpoints": res.n_breakpoints,
            "breakpoints": linear_breakpoints,
            "betas": betas,
            "intercepts": intercepts,
            "betas_ci": betas_ci,
            "breakpoints_ci": bp_cis,
            "log_freq": log_freq,
            "log_power": log_power,
            "residuals": residuals,
            "fitted_log_power": fitted_log_power,
            "ci_computed": True,
            "model_object": res,
        }

        # Heteroscedasticity checks
        spearman_corr_log_freq, spearman_corr_freq = np.nan, np.nan
        if len(residuals) > 1:
            spearman_corr_log_freq, _ = stats.spearmanr(log_freq, np.abs(residuals))
            spearman_corr_freq, _ = stats.spearmanr(10**log_freq, np.abs(residuals))
        results["spearman_corr_log_freq"] = spearman_corr_log_freq
        results["spearman_corr_freq"] = spearman_corr_freq

        if durbin_watson:
             results["durbin_watson_stat"] = durbin_watson(residuals)

        return results

    except Exception as e:
        failure_reason = f"MannKS segmented fit failed: {e}"
        logger.error(failure_reason, exc_info=True)
        return {
            "failure_reason": failure_reason,
            "n_breakpoints": n_breakpoints,
            "bic": np.inf,
            "aic": np.inf,
        }

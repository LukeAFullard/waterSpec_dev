import logging
import warnings

import numpy as np
import piecewise_regression
from scipy import stats


def _calculate_bic(y, y_pred, n_params):
    """Calculates the Bayesian Information Criterion (BIC)."""
    n = len(y)
    if n == 0:
        return np.nan
    rss = np.sum((y - y_pred) ** 2)
    # If RSS is zero or very close to it, the log will be -inf.
    if rss < 1e-12:
        return -np.inf
    bic = n * np.log(rss / n) + n_params * np.log(n)
    return bic


def fit_standard_model(
    frequency,
    power,
    method="theil-sen",
    ci_method="bootstrap",
    n_bootstraps=1000,
    ci=95,
    seed=None,
    logger=None,
):
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
        n_bootstraps (int, optional): Number of bootstrap samples.
        ci (int, optional): The desired confidence interval in percent.
        seed (int, optional): A seed for the random number generator.
        logger (logging.Logger, optional): A logger for warnings.
    """
    # 1. Validate inputs and filter data
    if method not in ["ols", "theil-sen"]:
        raise ValueError(f"Unknown fitting method: '{method}'. Choose 'ols' or 'theil-sen'.")
    valid_indices = (frequency > 0) & (power > 0)
    if np.sum(valid_indices) < 2:
        return {"beta": np.nan, "bic": np.nan, "beta_ci_lower": np.nan, "beta_ci_upper": np.nan}

    log_freq = np.log(frequency[valid_indices])
    log_power = np.log(power[valid_indices])

    # 2. Perform the initial fit (OLS or Theil-Sen)
    fit_results = {}
    try:
        if method == "ols":
            res = stats.linregress(log_freq, log_power)
            slope, intercept, r_value, _, stderr = res
            fit_results.update({"r_squared": r_value**2, "stderr": stderr})
        elif method == "theil-sen":
            res = stats.theilslopes(log_power, log_freq, alpha=1 - (ci / 100))
            slope, intercept, low_slope, high_slope = res
            fit_results.update({"slope_ci_lower": low_slope, "slope_ci_upper": high_slope})

        fit_results.update({"beta": -slope, "intercept": intercept})

    except Exception as e:
        failure_reason = (
            f"Initial standard model fit failed with method '{method}'. Error: {e}"
        )
        if logger:
            logger.warning(failure_reason)
        else:
            warnings.warn(failure_reason, UserWarning)
        return {
            "beta": np.nan,
            "bic": np.inf,  # Use infinite BIC to ensure this model is not chosen
            "beta_ci_lower": np.nan,
            "beta_ci_upper": np.nan,
            "failure_reason": failure_reason,
        }

    # 3. Calculate BIC
    log_power_fit = slope * log_freq + intercept
    bic = _calculate_bic(log_power, log_power_fit, n_params=2)
    fit_results["bic"] = bic

    # 4. Calculate Confidence Intervals
    beta_ci_lower, beta_ci_upper = np.nan, np.nan
    if ci_method == "bootstrap":
        residuals = log_power - log_power_fit
        rng = np.random.default_rng(seed)
        beta_estimates = []

        for _ in range(n_bootstraps):
            try:
                resampled_residuals = rng.choice(residuals, size=len(residuals), replace=True)
                synthetic_log_power = log_power_fit + resampled_residuals
                if method == "ols":
                    resampled_slope = stats.linregress(log_freq, synthetic_log_power).slope
                else:  # theil-sen
                    resampled_slope = stats.theilslopes(synthetic_log_power, log_freq)[0]
                beta_estimates.append(-resampled_slope)
            except Exception as e:
                # Log the specific error for a failed iteration
                msg = f"Bootstrap iteration failed with error: {e}"
                if logger:
                    logger.debug(msg)
                continue

        MIN_BOOTSTRAP_SUCCESS_RATIO = 0.8
        if len(beta_estimates) > 0:
            if len(beta_estimates) < n_bootstraps * MIN_BOOTSTRAP_SUCCESS_RATIO:
                msg = (
                    f"Only {len(beta_estimates)}/{n_bootstraps} bootstrap iterations "
                    f"succeeded (less than {MIN_BOOTSTRAP_SUCCESS_RATIO * 100}%). "
                    "The resulting confidence interval may be unreliable."
                )
                if logger:
                    logger.warning(msg)
                else:
                    warnings.warn(msg, UserWarning)
            p_lower = (100 - ci) / 2
            p_upper = 100 - p_lower
            beta_ci_lower = np.percentile(beta_estimates, p_lower)
            beta_ci_upper = np.percentile(beta_estimates, p_upper)

    elif ci_method == "parametric":
        if method == "ols":
            stderr = fit_results.get("stderr", np.nan)
            if np.isfinite(stderr):
                t_val = stats.t.ppf((1 + ci / 100) / 2, len(log_freq) - 2)
                half_width = t_val * stderr
                slope_ci_lower, slope_ci_upper = slope - half_width, slope + half_width
                beta_ci_lower, beta_ci_upper = -slope_ci_upper, -slope_ci_lower
        elif method == "theil-sen":
            slope_ci_lower = fit_results.get("slope_ci_lower", np.nan)
            slope_ci_upper = fit_results.get("slope_ci_upper", np.nan)
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
MIN_POINTS_PER_SEGMENT = 5


def _bootstrap_segmented_fit(pw_fit, log_freq, log_power, n_bootstraps, ci, seed, logger=None):
    """
    Performs robust bootstrap resampling for a fitted piecewise model.
    This also calculates the confidence interval of the fitted line itself.
    """
    n_breakpoints = pw_fit.n_breakpoints
    log_power_fit = pw_fit.predict(log_freq)
    residuals = log_power - log_power_fit
    rng = np.random.default_rng(seed)

    bootstrap_betas = [[] for _ in range(n_breakpoints + 1)]
    bootstrap_breakpoints = [[] for _ in range(n_breakpoints)]
    bootstrap_fits = []  # Store each bootstrap fit line
    successful_fits = 0

    # Extract the starting breakpoints from the initial fit's results.
    # This was the source of a bug where bootstrap CIs would always fail.
    # The previous code tried to access a non-existent `estimates` attribute
    # on the `pw_fit` object.
    initial_results = pw_fit.get_results()
    initial_estimates = initial_results.get("estimates")

    if not initial_estimates:
        if logger:
            logger.warning("Could not perform bootstrap: initial fit failed or produced no estimates.")
        else:
            warnings.warn("Could not perform bootstrap: initial fit failed or produced no estimates.", UserWarning)
        return {
            "betas_ci": [(np.nan, np.nan)] * (n_breakpoints + 1),
            "breakpoints_ci": [(np.nan, np.nan)] * n_breakpoints,
            "fit_ci_lower": None,
            "fit_ci_upper": None,
        }

    start_values = [
        initial_estimates[f"breakpoint{i+1}"]["estimate"]
        for i in range(n_breakpoints)
    ]

    for _ in range(n_bootstraps):
        resampled_residuals = rng.choice(residuals, size=len(residuals), replace=True)
        synthetic_log_power = log_power_fit + resampled_residuals

        try:
            bootstrap_pw_fit = piecewise_regression.Fit(
                log_freq,
                synthetic_log_power,
                n_breakpoints=n_breakpoints,
                start_values=start_values,
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
            # Log the specific error for a failed iteration
            msg = f"Segmented bootstrap iteration failed with error: {e}"
            if logger:
                logger.debug(msg)
            continue

    if successful_fits < n_bootstraps * 0.8:
        msg = (
            f"Only {successful_fits}/{n_bootstraps} bootstrap iterations for the "
            "segmented model succeeded. The confidence intervals may be unreliable."
        )
        if logger:
            logger.warning(msg)
        else:
            warnings.warn(msg, UserWarning)

    lower_p, upper_p = (100 - ci) / 2, 100 - (100 - ci) / 2
    ci_results = {
        "betas_ci": [],
        "breakpoints_ci": [],
        "fit_ci_lower": None,
        "fit_ci_upper": None,
    }

    # Calculate CIs for the fitted line itself
    if bootstrap_fits:
        bootstrap_fits_arr = np.array(bootstrap_fits)
        ci_results["fit_ci_lower"] = np.percentile(
            bootstrap_fits_arr, lower_p, axis=0
        )
        ci_results["fit_ci_upper"] = np.percentile(
            bootstrap_fits_arr, upper_p, axis=0
        )

    for i in range(n_breakpoints + 1):
        if bootstrap_betas[i]:
            lower = np.percentile(bootstrap_betas[i], lower_p)
            upper = np.percentile(bootstrap_betas[i], upper_p)
            ci_results["betas_ci"].append((lower, upper))
        else:
            ci_results["betas_ci"].append((np.nan, np.nan))

    for i in range(n_breakpoints):
        if bootstrap_breakpoints[i]:
            lower = np.percentile(bootstrap_breakpoints[i], lower_p)
            upper = np.percentile(bootstrap_breakpoints[i], upper_p)
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
    msg = (
        "Parametric confidence intervals for segmented models assume normality "
        "of errors and may be less reliable than bootstrap intervals. "
        "Consider using ci_method='bootstrap' for more robust results."
    )
    if logger:
        logger.warning(msg)
    else:
        warnings.warn(msg, UserWarning)

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
            # Beta is the negative of the slope, so the CI is inverted.
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
    frequency,
    power,
    n_breakpoints=1,
    p_threshold=0.05,
    ci_method="bootstrap",
    n_bootstraps=1000,
    ci=95,
    seed=None,
    logger=None,
):
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
        n_bootstraps (int, optional): Number of bootstrap samples for CI.
            Only used if `ci_method` is `'bootstrap'`. Defaults to 1000.
        ci (int, optional): The desired confidence interval in percent.
            Defaults to 95.
        seed (int, optional): A seed for the random number generator.
        logger (logging.Logger, optional): A logger for warnings.

    Returns:
        dict: A dictionary containing the fit results, including CIs.
    """
    # Log-transform the data
    if n_breakpoints > 1:
        msg = (
            f"Fitting a model with {n_breakpoints} breakpoints. Note that statistical "
            "significance is only tested for the 1-breakpoint model (Davies test). "
            "The selection of models with >1 breakpoint is based solely on BIC, "
            "which may risk overfitting."
        )
        if logger:
            logger.warning(msg)
        else:
            warnings.warn(msg, UserWarning)

    valid_indices = (frequency > 0) & (power > 0)
    min_points = MIN_POINTS_PER_SEGMENT * (n_breakpoints + 1)
    if np.sum(valid_indices) < min_points:
        summary = f"Not enough data points for {n_breakpoints}-breakpoint regression."
        return {"model_summary": summary, "n_breakpoints": n_breakpoints, "bic": np.inf}

    log_freq = np.log(frequency[valid_indices])
    log_power = np.log(power[valid_indices])

    # Fit the piecewise regression model
    try:
        pw_fit = piecewise_regression.Fit(
            log_freq, log_power, n_breakpoints=n_breakpoints
        )
        fit_summary = pw_fit.get_results()
        converged = fit_summary["converged"]
    except Exception as e:
        if logger:
            logger.warning(f"Segmented regression failed with an unexpected error: {e}")
        else:
            warnings.warn(
                f"Segmented regression failed with an unexpected error: {e}", UserWarning
            )
        return {
            "model_summary": "Segmented regression failed with an unexpected error.",
            "n_breakpoints": n_breakpoints,
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
            "bic": np.inf,  # Return infinite BIC for failed fits
        }

    # --- Extract results ---
    fit_summary = pw_fit.get_results()
    estimates = fit_summary["estimates"]
    fitted_log_power = pw_fit.predict(log_freq)

    # Store base results
    results = {
        "bic": fit_summary.get("bic"),
        "r_squared": fit_summary.get("r_squared"),
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

    slopes = []
    current_slope = estimates["alpha1"]["estimate"]
    slopes.append(current_slope)
    for i in range(1, n_breakpoints + 1):
        current_slope += estimates[f"beta{i}"]["estimate"]
        slopes.append(current_slope)

    betas = [-s for s in slopes]

    results["breakpoints"] = breakpoints
    results["betas"] = betas

    # --- Calculate Confidence Intervals based on the chosen method ---
    if ci_method == "bootstrap":
        if n_bootstraps > 0:
            ci_results = _bootstrap_segmented_fit(
                pw_fit, log_freq, log_power, n_bootstraps, ci, seed, logger=logger
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

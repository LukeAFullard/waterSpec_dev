import numpy as np
import piecewise_regression
from scipy import stats


def fit_spectrum(frequency, power, method="theil-sen"):
    """
    Fits a line to the power spectrum on a log-log plot to find the spectral
    exponent (beta).

    Args:
        frequency (np.ndarray): The frequency array from the periodogram.
        power (np.ndarray): The power array from the periodogram.
        method (str, optional): The fitting method to use.
            'theil-sen' for the robust Theil-Sen estimator (default).
            'ols' for Ordinary Least Squares.

    Returns:
        dict: A dictionary containing the fit results:
              - 'beta': The spectral exponent (the negative of the slope).
              - 'r_squared': The R-squared value of the fit (OLS only).
              - 'intercept': The intercept of the log-log regression.
              - 'stderr': The standard error of the slope (OLS only).
    """
    # Ensure there are no zero or negative values before log-transforming
    # This is important as frequency or power can sometimes be zero.
    valid_indices = (frequency > 0) & (power > 0)
    if np.sum(valid_indices) < 2:
        # Not enough data points to perform a linear regression
        return {
            "beta": np.nan,
            "r_squared": np.nan,
            "intercept": np.nan,
            "stderr": np.nan,
        }

    log_freq = np.log(frequency[valid_indices])
    log_power = np.log(power[valid_indices])

    if method == "ols":
        # Perform linear regression on the log-log data
        lin_reg_result = stats.linregress(log_freq, log_power)
        slope = lin_reg_result.slope
        intercept = lin_reg_result.intercept
        r_squared = lin_reg_result.rvalue**2
        stderr = lin_reg_result.stderr
    elif method == "theil-sen":
        # Use the robust Theil-Sen estimator from SciPy
        res = stats.theilslopes(log_power, log_freq, 0.95)
        slope = res[0]
        intercept = res[1]
        # Theil-Sen does not provide R-squared or standard error directly
        r_squared = np.nan
        stderr = np.nan
    else:
        raise ValueError(
            f"Unknown fitting method: '{method}'. Choose 'ols' or 'theil-sen'."
        )

    # The spectral exponent (beta) is the negative of the slope
    beta = -slope

    # Store the results in a dictionary
    fit_results = {
        "beta": beta,
        "r_squared": r_squared,
        "intercept": intercept,
        "stderr": stderr,
    }

    return fit_results


def _calculate_bic(y, y_pred, n_params):
    """Calculates the Bayesian Information Criterion (BIC)."""
    n = len(y)
    if n == 0:
        return np.nan
    rss = np.sum((y - y_pred) ** 2)
    if rss == 0:  # Perfect fit, BIC is -inf
        return -np.inf
    bic = n * np.log(rss / n) + n_params * np.log(n)
    return bic


import warnings


def fit_spectrum_with_bootstrap(
    frequency, power, method="theil-sen", n_bootstraps=1000, ci=95, seed=None
):
    """
    Fits the power spectrum and estimates confidence intervals for beta using
    bootstrap resampling.

    Args:
        frequency (np.ndarray): The frequency array.
        power (np.ndarray): The power array.
        method (str, optional): The fitting method ('theil-sen' or 'ols').
            Defaults to 'theil-sen'.
        n_bootstraps (int, optional): The number of bootstrap samples to
            generate. Defaults to 1000.
        ci (int, optional): The desired confidence interval in percent.
            Defaults to 95.
        seed (int, optional): A seed for the random number generator to ensure
            reproducibility. Defaults to None.

    Returns:
        dict: A dictionary containing the fit results, including the bootstrap
              confidence interval:
              - 'beta': The spectral exponent from the original data.
              - 'r_squared': The R-squared value of the original fit.
              - 'intercept': The intercept of the original fit.
              - 'stderr': The standard error of the original fit.
              - 'bic': Bayesian Information Criterion for the fit.
              - 'beta_ci_lower': The lower bound of the confidence interval for beta.
              - 'beta_ci_upper': The upper bound of the confidence interval for beta.
    """
    # Get the initial fit
    initial_fit = fit_spectrum(frequency, power, method=method)
    if np.isnan(initial_fit["beta"]):
        # If the initial fit failed, we can't do bootstrap
        initial_fit.update(
            {"beta_ci_lower": np.nan, "beta_ci_upper": np.nan, "bic": np.nan}
        )
        return initial_fit

    # Log-transform the data
    valid_indices = (frequency > 0) & (power > 0)
    log_freq = np.log(frequency[valid_indices])
    log_power = np.log(power[valid_indices])

    # Calculate the fitted line and residuals from the initial fit
    slope = -initial_fit["beta"]
    intercept = initial_fit["intercept"]
    log_power_fit = slope * log_freq + intercept
    residuals = log_power - log_power_fit

    # Calculate BIC for the initial fit (2 parameters: slope and intercept)
    bic = _calculate_bic(log_power, log_power_fit, 2)
    initial_fit["bic"] = bic

    # Perform bootstrap resampling
    beta_estimates = np.zeros(n_bootstraps)
    rng = np.random.default_rng(seed)

    for i in range(n_bootstraps):
        # Resample residuals
        resampled_residuals = rng.choice(residuals, size=len(residuals), replace=True)

        # Create a new synthetic log-power series
        synthetic_log_power = log_power_fit + resampled_residuals

        # Fit the synthetic data using the specified method
        if method == "ols":
            resampled_fit = stats.linregress(log_freq, synthetic_log_power)
            resampled_slope = resampled_fit.slope
        elif method == "theil-sen":
            resampled_fit = stats.theilslopes(synthetic_log_power, log_freq)
            resampled_slope = resampled_fit[0]

        # Store the new beta estimate
        beta_estimates[i] = -resampled_slope

    # Calculate the confidence interval
    lower_percentile = (100 - ci) / 2
    upper_percentile = 100 - lower_percentile
    beta_ci_lower = np.percentile(beta_estimates, lower_percentile)
    beta_ci_upper = np.percentile(beta_estimates, upper_percentile)

    # Add the confidence interval to the results
    initial_fit["beta_ci_lower"] = beta_ci_lower
    initial_fit["beta_ci_upper"] = beta_ci_upper

    # Store log-transformed data and residuals for potential use in other functions
    initial_fit["log_freq"] = log_freq
    initial_fit["log_power"] = log_power
    initial_fit["residuals"] = residuals
    initial_fit["fitted_log_power"] = log_power_fit

    return initial_fit


# Define a constant for the minimum number of data points required per
# segment in a regression to ensure stable fits.
MIN_POINTS_PER_SEGMENT = 5


def _bootstrap_segmented_fit(pw_fit, log_freq, log_power, n_bootstraps, ci, seed):
    """
    Performs bootstrap resampling for a fitted piecewise regression model.

    Args:
        pw_fit: A fitted `piecewise_regression.Fit` object.
        log_freq (np.ndarray): The log-transformed frequency data.
        log_power (np.ndarray): The log-transformed power data.
        n_bootstraps (int): The number of bootstrap samples to generate.
        ci (int): The desired confidence interval in percent.
        seed (int): A seed for the random number generator.

    Returns:
        dict: A dictionary containing the confidence intervals for betas and
              breakpoints. Returns empty dict if bootstrapping fails.
    """
    n_breakpoints = pw_fit.n_breakpoints
    log_power_fit = pw_fit.predict(log_freq)
    residuals = log_power - log_power_fit
    rng = np.random.default_rng(seed)

    # Store estimates from each bootstrap iteration
    bootstrap_betas = [[] for _ in range(n_breakpoints + 1)]
    bootstrap_breakpoints = [[] for _ in range(n_breakpoints)]

    for _ in range(n_bootstraps):
        resampled_residuals = rng.choice(residuals, size=len(residuals), replace=True)
        synthetic_log_power = log_power_fit + resampled_residuals

        try:
            # Fit the synthetic data, using the original fit's breakpoints
            # as a starting point to speed up convergence.
            bootstrap_pw_fit = piecewise_regression.Fit(
                log_freq,
                synthetic_log_power,
                n_breakpoints=n_breakpoints,
                start_values=pw_fit.get_results()["estimates"]["breakpoint1"][
                    "estimate"
                ],
            )
            if not bootstrap_pw_fit.get_results()["converged"]:
                continue

            # --- Extract estimates from the bootstrap fit ---
            estimates = bootstrap_pw_fit.get_results()["estimates"]
            # Breakpoints
            for i in range(n_breakpoints):
                bp_val = np.exp(estimates[f"breakpoint{i+1}"]["estimate"])
                bootstrap_breakpoints[i].append(bp_val)
            # Betas
            slopes = []
            current_slope = estimates["alpha1"]["estimate"]
            slopes.append(current_slope)
            for i in range(1, n_breakpoints + 1):
                current_slope += estimates[f"beta{i}"]["estimate"]
                slopes.append(current_slope)
            betas = [-s for s in slopes]
            for i in range(n_breakpoints + 1):
                bootstrap_betas[i].append(betas[i])
        except Exception:
            # If a bootstrap fit fails, just skip it.
            continue

    # --- Calculate confidence intervals ---
    lower_p = (100 - ci) / 2
    upper_p = 100 - lower_p
    ci_results = {"betas_ci": [], "breakpoints_ci": []}

    # Beta CIs
    for i in range(n_breakpoints + 1):
        if bootstrap_betas[i]:
            lower = np.percentile(bootstrap_betas[i], lower_p)
            upper = np.percentile(bootstrap_betas[i], upper_p)
            ci_results["betas_ci"].append((lower, upper))
        else:
            ci_results["betas_ci"].append((np.nan, np.nan))

    # Breakpoint CIs
    for i in range(n_breakpoints):
        if bootstrap_breakpoints[i]:
            lower = np.percentile(bootstrap_breakpoints[i], lower_p)
            upper = np.percentile(bootstrap_breakpoints[i], upper_p)
            ci_results["breakpoints_ci"].append((lower, upper))
        else:
            ci_results["breakpoints_ci"].append((np.nan, np.nan))

    return ci_results


def fit_segmented_spectrum(
    frequency,
    power,
    n_breakpoints=1,
    p_threshold=0.05,
    n_bootstraps=1000,
    ci=95,
    seed=None,
):
    """
    Fits a segmented regression and estimates confidence intervals via bootstrap.

    Args:
        frequency (np.ndarray): The frequency array.
        power (np.ndarray): The power array.
        n_breakpoints (int, optional): The number of breakpoints to fit.
            Defaults to 1.
        p_threshold (float, optional): The p-value threshold for the Davies
            test for a significant breakpoint (only for 1-breakpoint models).
            Defaults to 0.05.
        n_bootstraps (int, optional): Number of bootstrap samples for CI.
            Set to 0 to disable. Defaults to 1000.
        ci (int, optional): The desired confidence interval in percent.
            Defaults to 95.
        seed (int, optional): A seed for the random number generator.
            Defaults to None.

    Returns:
        dict: A dictionary containing the fit results, including CIs.
    """
    # Log-transform the data
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

    # For backward compatibility, add single values if only one breakpoint
    if n_breakpoints == 1:
        results["breakpoint"] = breakpoints[0]
        results["beta1"] = betas[0]
        results["beta2"] = betas[1]

    results["breakpoints"] = breakpoints
    results["betas"] = betas

    # --- Perform bootstrap if requested ---
    if n_bootstraps > 0:
        ci_results = _bootstrap_segmented_fit(
            pw_fit, log_freq, log_power, n_bootstraps, ci, seed
        )
        results.update(ci_results)
    else:
        # Ensure CI keys exist even when bootstrap is skipped
        results["betas_ci"] = [(np.nan, np.nan)] * (n_breakpoints + 1)
        results["breakpoints_ci"] = [(np.nan, np.nan)] * n_breakpoints

    return results

import numpy as np
from scipy import stats
import piecewise_regression

def fit_spectrum(frequency, power, method='theil-sen'):
    """
    Fits a line to the power spectrum on a log-log plot to find the spectral exponent (beta).

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
            'beta': np.nan,
            'r_squared': np.nan,
            'intercept': np.nan,
            'stderr': np.nan,
        }

    log_freq = np.log(frequency[valid_indices])
    log_power = np.log(power[valid_indices])

    if method == 'ols':
        # Perform linear regression on the log-log data
        lin_reg_result = stats.linregress(log_freq, log_power)
        slope = lin_reg_result.slope
        intercept = lin_reg_result.intercept
        r_squared = lin_reg_result.rvalue**2
        stderr = lin_reg_result.stderr
    elif method == 'theil-sen':
        # Use the robust Theil-Sen estimator from SciPy
        res = stats.theilslopes(log_power, log_freq, 0.95)
        slope = res[0]
        intercept = res[1]
        # Theil-Sen does not provide R-squared or standard error directly
        r_squared = np.nan
        stderr = np.nan
    else:
        raise ValueError(f"Unknown fitting method: '{method}'. Choose 'ols' or 'theil-sen'.")

    # The spectral exponent (beta) is the negative of the slope
    beta = -slope

    # Store the results in a dictionary
    fit_results = {
        'beta': beta,
        'r_squared': r_squared,
        'intercept': intercept,
        'stderr': stderr,
    }

    return fit_results

def _calculate_bic(y, y_pred, n_params):
    """Calculates the Bayesian Information Criterion (BIC)."""
    n = len(y)
    if n == 0:
        return np.nan
    rss = np.sum((y - y_pred)**2)
    if rss == 0: # Perfect fit, BIC is -inf
        return -np.inf
    bic = n * np.log(rss / n) + n_params * np.log(n)
    return bic

def fit_spectrum_with_bootstrap(frequency, power, method='theil-sen', n_bootstraps=1000, ci=95):
    """
    Fits the power spectrum and estimates confidence intervals for beta using bootstrap resampling.

    Args:
        frequency (np.ndarray): The frequency array.
        power (np.ndarray): The power array.
        method (str, optional): The fitting method ('theil-sen' or 'ols'). Defaults to 'theil-sen'.
        n_bootstraps (int, optional): The number of bootstrap samples to generate. Defaults to 1000.
        ci (int, optional): The desired confidence interval in percent. Defaults to 95.

    Returns:
        dict: A dictionary containing the fit results, including the bootstrap confidence interval:
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
    if np.isnan(initial_fit['beta']):
        # If the initial fit failed, we can't do bootstrap
        initial_fit.update({'beta_ci_lower': np.nan, 'beta_ci_upper': np.nan, 'bic': np.nan})
        return initial_fit

    # Log-transform the data
    valid_indices = (frequency > 0) & (power > 0)
    log_freq = np.log(frequency[valid_indices])
    log_power = np.log(power[valid_indices])

    # Calculate the fitted line and residuals from the initial fit
    slope = -initial_fit['beta']
    intercept = initial_fit['intercept']
    log_power_fit = slope * log_freq + intercept
    residuals = log_power - log_power_fit

    # Calculate BIC for the initial fit (2 parameters: slope and intercept)
    bic = _calculate_bic(log_power, log_power_fit, 2)
    initial_fit['bic'] = bic

    # Perform bootstrap resampling
    beta_estimates = np.zeros(n_bootstraps)
    rng = np.random.default_rng()

    for i in range(n_bootstraps):
        # Resample residuals
        resampled_residuals = rng.choice(residuals, size=len(residuals), replace=True)

        # Create a new synthetic log-power series
        synthetic_log_power = log_power_fit + resampled_residuals

        # Fit the synthetic data using the specified method
        if method == 'ols':
            resampled_fit = stats.linregress(log_freq, synthetic_log_power)
            resampled_slope = resampled_fit.slope
        elif method == 'theil-sen':
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
    initial_fit['beta_ci_lower'] = beta_ci_lower
    initial_fit['beta_ci_upper'] = beta_ci_upper

    # Store log-transformed data and residuals for potential use in other functions
    initial_fit['log_freq'] = log_freq
    initial_fit['log_power'] = log_power
    initial_fit['residuals'] = residuals
    initial_fit['fitted_log_power'] = log_power_fit

    return initial_fit

def fit_segmented_spectrum(frequency, power):
    """
    Fits a segmented regression to the power spectrum to find breakpoints.

    Args:
        frequency (np.ndarray): The frequency array.
        power (np.ndarray): The power array.

    Returns:
        dict: A dictionary containing the fit results.
    """
    # Log-transform the data
    valid_indices = (frequency > 0) & (power > 0)
    if np.sum(valid_indices) < 5: # Need enough points for segmented regression
        return {
            'breakpoint': np.nan, 'beta1': np.nan, 'beta2': np.nan,
            'model_summary': "Not enough data points for segmented regression."
        }

    log_freq = np.log(frequency[valid_indices])
    log_power = np.log(power[valid_indices])

    # Fit the piecewise regression model, specifying 1 breakpoint
    pw_fit = piecewise_regression.Fit(log_freq, log_power, n_breakpoints=1)

    # Check for convergence and statistical significance of the breakpoint
    davies_p_value = pw_fit.davies
    if not pw_fit.get_results()["converged"] or davies_p_value > 0.05:
        return {
            'breakpoint': np.nan, 'beta1': np.nan, 'beta2': np.nan,
            'model_summary': "No significant breakpoint found (Davies test p > 0.05) or model did not converge."
        }

    # Extract the results using the correct API
    fit_summary = pw_fit.get_results()
    estimates = fit_summary["estimates"]

    breakpoint_log_freq = estimates["breakpoint1"]["estimate"]
    breakpoint_freq = np.exp(breakpoint_log_freq)

    alpha1 = estimates["alpha1"]["estimate"]
    alpha2 = estimates["alpha2"]["estimate"]

    beta1 = -alpha1
    beta2 = -alpha2

    # Get additional fit statistics
    bic = fit_summary.get("bic")
    r_squared = fit_summary.get("r_squared")

    # Calculate the fitted line and residuals
    fitted_log_power = pw_fit.predict(log_freq)
    residuals = log_power - fitted_log_power

    results = {
        'breakpoint': breakpoint_freq,
        'beta1': beta1,
        'beta2': beta2,
        'bic': bic,
        'r_squared': r_squared,
        'model_summary': str(pw_fit.summary()),
        'model_object': pw_fit,
        # Store log-transformed data and residuals for consistent plotting/calculations
        'log_freq': log_freq,
        'log_power': log_power,
        'residuals': residuals,
        'fitted_log_power': fitted_log_power
    }

    return results

import numpy as np
from scipy import stats
import piecewise_regression

def fit_spectrum(frequency, power):
    """
    Fits a line to the power spectrum on a log-log plot to find the spectral exponent (beta).

    Args:
        frequency (np.ndarray): The frequency array from the periodogram.
        power (np.ndarray): The power array from the periodogram.

    Returns:
        dict: A dictionary containing the fit results:
              - 'beta': The spectral exponent (the negative of the slope).
              - 'r_squared': The R-squared value of the fit.
              - 'intercept': The intercept of the log-log regression.
              - 'stderr': The standard error of the estimated slope.
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

    # Perform linear regression on the log-log data
    lin_reg_result = stats.linregress(log_freq, log_power)

    # The spectral exponent (beta) is the negative of the slope
    beta = -lin_reg_result.slope

    # Store the results in a dictionary
    fit_results = {
        'beta': beta,
        'r_squared': lin_reg_result.rvalue**2,
        'intercept': lin_reg_result.intercept,
        'stderr': lin_reg_result.stderr,
    }

    return fit_results

def fit_spectrum_with_bootstrap(frequency, power, n_bootstraps=1000, ci=95):
    """
    Fits the power spectrum and estimates confidence intervals for beta using bootstrap resampling.

    Args:
        frequency (np.ndarray): The frequency array.
        power (np.ndarray): The power array.
        n_bootstraps (int, optional): The number of bootstrap samples to generate. Defaults to 1000.
        ci (int, optional): The desired confidence interval in percent. Defaults to 95.

    Returns:
        dict: A dictionary containing the fit results, including the bootstrap confidence interval:
              - 'beta': The spectral exponent from the original data.
              - 'r_squared': The R-squared value of the original fit.
              - 'intercept': The intercept of the original fit.
              - 'stderr': The standard error of the original fit.
              - 'beta_ci_lower': The lower bound of the confidence interval for beta.
              - 'beta_ci_upper': The upper bound of the confidence interval for beta.
    """
    # Get the initial fit
    initial_fit = fit_spectrum(frequency, power)
    if np.isnan(initial_fit['beta']):
        # If the initial fit failed, we can't do bootstrap
        initial_fit.update({'beta_ci_lower': np.nan, 'beta_ci_upper': np.nan})
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

    # Perform bootstrap resampling
    beta_estimates = np.zeros(n_bootstraps)
    rng = np.random.default_rng()
    for i in range(n_bootstraps):
        # Resample residuals
        resampled_residuals = rng.choice(residuals, size=len(residuals), replace=True)

        # Create a new synthetic log-power series
        synthetic_log_power = log_power_fit + resampled_residuals

        # Fit the synthetic data
        resampled_fit = stats.linregress(log_freq, synthetic_log_power)

        # Store the new beta estimate
        beta_estimates[i] = -resampled_fit.slope

    # Calculate the confidence interval
    lower_percentile = (100 - ci) / 2
    upper_percentile = 100 - lower_percentile
    beta_ci_lower = np.percentile(beta_estimates, lower_percentile)
    beta_ci_upper = np.percentile(beta_estimates, upper_percentile)

    # Add the confidence interval to the results
    initial_fit['beta_ci_lower'] = beta_ci_lower
    initial_fit['beta_ci_upper'] = beta_ci_upper

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
    estimates = pw_fit.get_results()["estimates"]

    breakpoint_log_freq = estimates["breakpoint1"]["estimate"]
    breakpoint_freq = np.exp(breakpoint_log_freq)

    alpha1 = estimates["alpha1"]["estimate"]
    alpha2 = estimates["alpha2"]["estimate"]

    beta1 = -alpha1
    beta2 = -alpha2

    results = {
        'breakpoint': breakpoint_freq,
        'beta1': beta1,
        'beta2': beta2,
        'model_summary': str(pw_fit.summary())
    }

    return results

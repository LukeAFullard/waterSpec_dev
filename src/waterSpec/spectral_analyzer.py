import numpy as np
from astropy.timeseries import LombScargle
from scipy.signal import find_peaks


def calculate_periodogram(
    time, data, frequency=None, dy=None, normalization="standard"
):
    """
    Calculates the Lomb-Scargle periodogram for a time series.

    This function requires a pre-computed frequency grid. Using a log-spaced
    grid is recommended for spectral analysis. The `waterSpec.generate_log_spaced_grid`
    function can be used to create a suitable grid.

    Args:
        time (np.ndarray): The time array.
        data (np.ndarray): The data values.
        frequency (np.ndarray): The frequency grid to use for the periodogram.
        dy (np.ndarray, optional): Measurement uncertainties for each data point.
                                   Defaults to None.
        normalization (str, optional): The normalization to use for the periodogram.
                                       Defaults to 'standard'.

    Returns:
        tuple[np.ndarray, np.ndarray, LombScargle]: A tuple containing:
                                       - frequency
                                       - power
                                       - The LombScargle object instance
    """
    if frequency is None:
        raise ValueError(
            "A frequency grid must be provided. The use of autopower has been "
            "removed to prevent incorrect grid generation."
        )

    ls = LombScargle(time, data, dy=dy)

    power = ls.power(frequency, normalization=normalization)

    return frequency, power, ls


import warnings


def find_significant_peaks(
    ls, frequency, power, fap_threshold=0.01, fap_method="baluev", **fap_kwargs
):
    """
    Finds statistically significant peaks in a periodogram using a False Alarm
    Probability threshold.

    Args:
        ls (LombScargle): The LombScargle object instance.
        frequency (np.ndarray): The frequency grid.
        power (np.ndarray): The power values corresponding to the frequency grid.
        fap_threshold (float, optional): The FAP level for significance.
            Defaults to 0.01.
        fap_method (str, optional): The method for FAP calculation
            ('bootstrap', 'baluev', etc.). Defaults to 'baluev'.
        **fap_kwargs: Additional keyword arguments for
            `astropy.LombScargle.false_alarm_level`.

    Returns:
        tuple: A tuple containing:
               - list: A list of dictionaries, each describing a significant peak.
               - float: The power level corresponding to the FAP threshold.
    """
    # Warn the user if they are using the slow bootstrap method
    if fap_method == "bootstrap":
        warnings.warn(
            "Using the 'bootstrap' method for FAP calculation. This can be "
            "very slow for large datasets. Consider using an analytical method "
            "like 'baluev' for faster results.",
            UserWarning,
        )

    # Calculate the power level corresponding to the FAP threshold.
    # We use np.max() because, depending on the method, false_alarm_level
    # can return an array. We need a single scalar for the height threshold.
    fap_level = np.max(
        ls.false_alarm_level(fap_threshold, method=fap_method, **fap_kwargs)
    )

    # Find all peaks in the spectrum that are above the significance level.
    # Using the 'height' parameter is more efficient than finding all peaks
    # and then filtering.
    significant_peaks_indices, _ = find_peaks(power, height=fap_level)

    # Add a warning if per-peak FAP calculation is likely to be slow
    if fap_method == "bootstrap" and len(significant_peaks_indices) > 5:
        warnings.warn(
            f"Calculating the individual FAP for {len(significant_peaks_indices)} "
            "peaks using the 'bootstrap' method may be very slow.",
            UserWarning,
        )

    significant_peaks = []
    for peak_idx in significant_peaks_indices:
        peak_freq = frequency[peak_idx]
        peak_power = power[peak_idx]
        # Calculate the FAP of this specific peak
        peak_fap = ls.false_alarm_probability(
            peak_power, method=fap_method, **fap_kwargs
        )
        significant_peaks.append(
            {"frequency": peak_freq, "power": peak_power, "fap": peak_fap}
        )

    # Sort peaks by power in descending order
    significant_peaks.sort(key=lambda p: p["power"], reverse=True)

    return significant_peaks, fap_level


from scipy import stats


def find_peaks_via_residuals(fit_results, ci=95):
    """
    Finds significant peaks by identifying outliers in the residuals of the
    spectral fit using a statistically robust threshold.

    Args:
        fit_results (dict): The dictionary returned by the fitting functions.
                            Must contain 'residuals', 'fitted_log_power', 'log_freq',
                            and 'log_power'.
        ci (int, optional): The confidence level for the significance
                            threshold, in percent. Defaults to 95.

    Returns:
        tuple: A tuple containing:
               - list: A list of dictionaries, each describing a significant peak.
               - float: The residual threshold used for significance.
    """
    required_keys = ["residuals", "fitted_log_power", "log_freq", "log_power"]
    if not all(key in fit_results for key in required_keys):
        missing_keys = [key for key in required_keys if key not in fit_results]
        raise ValueError(f"fit_results is missing required keys: {missing_keys}")

    if not (0 < ci < 100):
        raise ValueError("Confidence interval 'ci' must be between 0 and 100.")

    residuals = fit_results["residuals"]
    log_freq = fit_results["log_freq"]
    log_power = fit_results["log_power"]

    # Calculate a statistically-based threshold assuming residuals are ~normally distributed.
    # This is more robust than a simple percentile.
    residual_std = np.std(residuals)

    # We are interested in positive peaks, so this is a one-tailed test.
    # Convert CI to a significance level (alpha)
    alpha = 1 - (ci / 100)

    # The critical Z-value for a one-tailed test.
    z_critical = stats.norm.ppf(1 - alpha)

    # The threshold is the critical value times the standard deviation of the residuals.
    # We add the mean, which should be close to zero for good residuals, but for safety.
    residual_threshold = np.mean(residuals) + z_critical * residual_std

    # Use a dedicated peak-finding algorithm on the residuals. This is more
    # robust than grouping contiguous regions, as it prevents a single broad
    # peak from being split into multiple reported peaks.
    peak_indices, _ = find_peaks(residuals, height=residual_threshold)

    if len(peak_indices) == 0:
        return [], residual_threshold

    significant_peaks = []
    for peak_idx in peak_indices:
        significant_peaks.append(
            {
                "frequency": np.exp(log_freq[peak_idx]),
                "power": np.exp(log_power[peak_idx]),
                "residual": residuals[peak_idx],
            }
        )

    # Sort peaks by residual in descending order
    significant_peaks.sort(key=lambda p: p["residual"], reverse=True)

    return significant_peaks, residual_threshold

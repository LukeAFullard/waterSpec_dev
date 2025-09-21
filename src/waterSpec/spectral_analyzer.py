import numpy as np
from astropy.timeseries import LombScargle

from scipy.signal import find_peaks

def calculate_periodogram(time, data, frequency=None, dy=None, normalization='standard'):
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
        raise ValueError("A frequency grid must be provided. The use of autopower has been removed to prevent incorrect grid generation.")

    ls = LombScargle(time, data, dy=dy)

    power = ls.power(frequency, normalization=normalization)

    return frequency, power, ls


def find_significant_peaks(ls, frequency, power, fap_threshold=0.01, fap_method='bootstrap', **fap_kwargs):
    """
    Finds statistically significant peaks in a periodogram using a False Alarm Probability threshold.

    Args:
        ls (LombScargle): The LombScargle object instance.
        frequency (np.ndarray): The frequency grid.
        power (np.ndarray): The power values corresponding to the frequency grid.
        fap_threshold (float, optional): The FAP level for significance. Defaults to 0.01.
        fap_method (str, optional): The method for FAP calculation ('bootstrap', 'baluev', etc.).
                                    Defaults to 'bootstrap'.
        **fap_kwargs: Additional keyword arguments for `astropy.LombScargle.false_alarm_level`.

    Returns:
        tuple: A tuple containing:
               - list: A list of dictionaries, each describing a significant peak.
               - float: The power level corresponding to the FAP threshold.
    """
    # Calculate the power level corresponding to the FAP threshold
    fap_level = ls.false_alarm_level(fap_threshold, method=fap_method, **fap_kwargs)

    # Find all peaks in the spectrum
    all_peaks, _ = find_peaks(power)

    # Filter for peaks that are above the significance level
    significant_peaks_indices = [p for p in all_peaks if power[p] > fap_level]

    significant_peaks = []
    for peak_idx in significant_peaks_indices:
        peak_freq = frequency[peak_idx]
        peak_power = power[peak_idx]
        # Calculate the FAP of this specific peak
        peak_fap = ls.false_alarm_probability(peak_power, method=fap_method, **fap_kwargs)
        significant_peaks.append({
            'frequency': peak_freq,
            'power': peak_power,
            'fap': peak_fap
        })

    # Sort peaks by power in descending order
    significant_peaks.sort(key=lambda p: p['power'], reverse=True)

    return significant_peaks, fap_level


def find_peaks_via_residuals(fit_results, ci=95):
    """
    Finds significant peaks by identifying outliers in the residuals of the spectral fit.

    Args:
        fit_results (dict): The dictionary returned by the fitting functions.
                            Must contain 'residuals', 'fitted_log_power', 'log_freq',
                            and 'log_power'.
        ci (int, optional): The confidence interval to use for the significance
                            threshold, in percent. Defaults to 95.

    Returns:
        tuple: A tuple containing:
               - list: A list of dictionaries, each describing a significant peak.
               - float: The residual threshold used for significance.
    """
    if 'residuals' not in fit_results or 'fitted_log_power' not in fit_results:
        raise ValueError("fit_results dictionary must contain 'residuals' and 'fitted_log_power'.")

    residuals = fit_results['residuals']
    log_freq = fit_results['log_freq']
    log_power = fit_results['log_power']

    # Calculate the significance threshold from the residuals
    residual_threshold = np.percentile(residuals, ci)

    # Find indices where the residual exceeds the threshold
    significant_indices = np.where(residuals > residual_threshold)[0]
    if len(significant_indices) == 0:
        return [], residual_threshold

    # Group contiguous significant indices into regions
    significant_groups = np.split(significant_indices, np.where(np.diff(significant_indices) != 1)[0] + 1)

    significant_peaks = []
    for group in significant_groups:
        if len(group) == 0:
            continue
        # For each significant region, find the point with the highest power
        max_power_index_in_group = group[np.argmax(log_power[group])]

        significant_peaks.append({
            'frequency': np.exp(log_freq[max_power_index_in_group]),
            'power': np.exp(log_power[max_power_index_in_group]),
            'residual': residuals[max_power_index_in_group]
        })

    # Sort peaks by residual in descending order
    significant_peaks.sort(key=lambda p: p['residual'], reverse=True)

    return significant_peaks, residual_threshold

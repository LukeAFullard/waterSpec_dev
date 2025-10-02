import numpy as np
from astropy.timeseries import LombScargle
from scipy.signal import find_peaks


def calculate_periodogram(
    time: np.ndarray,
    data: np.ndarray,
    dy: np.ndarray = None,
    normalization: str = None,
    nyquist_factor: float = 1.0,
    samples_per_peak: int = 5,
    minimum_frequency: float = None,
    maximum_frequency: float = None,
) -> tuple[np.ndarray, np.ndarray, LombScargle]:
    """
    Calculates the Lomb-Scargle periodogram using Astropy's `autopower` method.

    This function automatically generates an appropriate frequency grid based on
    the data properties and the provided parameters.

    The normalization is chosen automatically based on the presence of measurement
    errors (`dy`), but can be overridden. If errors are provided, 'psd' is the
    default, which is essential for quantitative spectral slope analysis.

    Args:
        time (np.ndarray): The time array.
        data (np.ndarray): The data values.
        dy (np.ndarray, optional): Measurement uncertainties for each data point.
            If provided, `normalization` defaults to 'psd'. Defaults to None.
        normalization (str, optional): The normalization method. Defaults to 'psd'
            if `dy` is given, otherwise 'standard'.
        nyquist_factor (float, optional): The factor by which to scale the Nyquist
            frequency. Defaults to 1.0.
        samples_per_peak (int, optional): The number of samples to generate per
            periodogram peak. Higher values lead to a denser frequency grid.
            Defaults to 5.
        minimum_frequency (float, optional): The lowest frequency to include in
            the grid. Defaults to a value determined by `autopower`.
        maximum_frequency (float, optional): The highest frequency to include in
            the grid. Overrides `nyquist_factor` if set. Defaults to None.

    Returns:
        tuple[np.ndarray, np.ndarray, LombScargle]: A tuple containing:
            - frequency: The automatically generated frequency grid.
            - power: The power values for the corresponding frequencies.
            - ls_instance: The LombScargle object instance.
    """
    # Automatically select normalization if not specified.
    # 'psd' is crucial for physical units when errors are known.
    if normalization is None:
        normalization = "psd" if dy is not None else "standard"

    ls_instance = LombScargle(time, data, dy=dy)

    frequency, power = ls_instance.autopower(
        nyquist_factor=nyquist_factor,
        samples_per_peak=samples_per_peak,
        minimum_frequency=minimum_frequency,
        maximum_frequency=maximum_frequency,
        normalization=normalization,
    )

    return frequency, power, ls_instance


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
from statsmodels.stats.multitest import fdrcorrection
from typing import Dict, List, Tuple


def find_peaks_via_residuals(fit_results: Dict, fdr_level: float = 0.05) -> Tuple[List[Dict], float]:
    """
    Finds significant peaks by identifying outliers in the residuals of the
    spectral fit using the Benjamini-Yekutieli False Discovery Rate (FDR)
    procedure.

    This method is more robust than the standard Benjamini-Hochberg procedure
    as it accounts for potential positive correlations between tests (i.e.,
    between adjacent frequencies in the periodogram), providing better control
    over the false discovery rate.

    Args:
        fit_results (dict): The dictionary returned by the fitting functions.
                            Must contain 'residuals', 'fitted_log_power', 'log_freq',
                            and 'log_power'.
        fdr_level (float, optional): The false discovery rate level for significance.
                                     Defaults to 0.05.

    Returns:
        tuple: A tuple containing:
               - list: A list of dictionaries, each describing a significant peak.
               - float: The minimum residual value of a significant peak, which
                        can be used as an effective threshold for plotting.
    """
    required_keys = ["residuals", "fitted_log_power", "log_freq", "log_power"]
    if not all(key in fit_results for key in required_keys):
        missing_keys = [key for key in required_keys if key not in fit_results]
        raise ValueError(f"fit_results is missing required keys: {missing_keys}")

    if not (0 < fdr_level < 1):
        raise ValueError("fdr_level must be between 0 and 1.")

    residuals = fit_results["residuals"]
    log_freq = fit_results["log_freq"]
    log_power = fit_results["log_power"]
    n_comparisons = len(residuals)

    if n_comparisons == 0:
        return [], np.nan

    # Use robust statistics (median and MAD) for standardization
    residual_median = np.median(residuals)
    residual_mad_std = stats.median_abs_deviation(residuals, scale="normal")

    if residual_mad_std < 1e-12:
        return [], np.inf  # No variance in residuals

    # Calculate one-tailed p-values for positive residuals (potential peaks)
    z_scores = (residuals - residual_median) / residual_mad_std
    p_values = 1 - stats.norm.cdf(z_scores)

    # Apply Benjamini-Yekutieli FDR correction for correlated tests
    is_significant, _ = fdrcorrection(p_values, alpha=fdr_level, method="negcorr")

    if not np.any(is_significant):
        return [], np.inf

    # Find all peaks in the residual series first
    all_peak_indices, _ = find_peaks(residuals)

    # Filter to keep only those peaks that are statistically significant
    significant_peak_indices = [
        idx for idx in all_peak_indices if is_significant[idx]
    ]

    if not significant_peak_indices:
        return [], np.inf

    significant_peaks = []
    for peak_idx in significant_peak_indices:
        significant_peaks.append(
            {
                "frequency": 10 ** log_freq[peak_idx],
                "power": 10 ** log_power[peak_idx],
                "residual": residuals[peak_idx],
            }
        )

    # Sort peaks by residual in descending order
    significant_peaks.sort(key=lambda p: p["residual"], reverse=True)

    # The effective threshold is the smallest residual that was deemed significant
    min_significant_residual = min(p["residual"] for p in significant_peaks)

    return significant_peaks, min_significant_residual

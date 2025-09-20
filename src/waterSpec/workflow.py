from .data_loader import load_data
from .preprocessor import preprocess_data
from .spectral_analyzer import calculate_periodogram, find_significant_peaks
from .fitter import fit_spectrum_with_bootstrap, fit_segmented_spectrum
from .interpreter import interpret_results
from .plotting import plot_spectrum
from .frequency_generator import generate_log_spaced_grid
import warnings
import numpy as np

def run_analysis(
    file_path,
    time_col,
    data_col,
    error_col=None,
    param_name=None,
    censor_strategy='drop',
    log_transform_data=False,
    detrend_method='linear',
    normalize_data=False,
    detrend_options=None,
    analysis_type='standard',
    n_bootstraps=1000,
    fap_threshold=None,
    do_plot=False,
    output_path=None
):
    """
    High-level function to run a complete spectral analysis workflow.
    """
    time_numeric, data_series, error_series = load_data(
        file_path, time_col, data_col, error_col=error_col
    )
    processed_data, processed_errors = preprocess_data(
        data_series,
        time_numeric,
        error_series=error_series,
        censor_strategy=censor_strategy,
        log_transform_data=log_transform_data,
        detrend_method=detrend_method,
        normalize_data=normalize_data,
        detrend_options=detrend_options
    )

    valid_indices = ~np.isnan(processed_data)
    if np.sum(valid_indices) < 10:
        raise ValueError("Not enough valid data points remaining after preprocessing.")
    time_numeric = time_numeric[valid_indices]
    processed_data = processed_data[valid_indices]
    dy = processed_errors[valid_indices] if processed_errors is not None else None

    custom_frequency = generate_log_spaced_grid(time_numeric)
    frequency, power = calculate_periodogram(time_numeric, processed_data, frequency=custom_frequency, dy=dy)

    if analysis_type == 'standard':
        fit_results = fit_spectrum_with_bootstrap(frequency, power, n_bootstraps=n_bootstraps)
        beta = fit_results.get('beta')
        ci = (fit_results.get('beta_ci_lower'), fit_results.get('beta_ci_upper'))
    elif analysis_type == 'segmented':
        fit_results = fit_segmented_spectrum(frequency, power)
        beta = fit_results.get('beta1')
        ci = None
    else:
        raise ValueError("analysis_type must be 'standard' or 'segmented'")

    if param_name is None:
        param_name = data_col

    if beta is None or not np.isfinite(beta):
        interpretation = {"summary_text": "Analysis failed: Could not determine a valid beta value."}
    else:
        interpretation = interpret_results(beta, ci=ci, param_name=param_name)

    interpretation.update(fit_results)

    if fap_threshold is not None:
        # Use the original (non-detrended) data for peak finding for better physical interpretation
        # Note: This is a design choice. Using processed_data is also valid.
        # For now, we use processed_data to be consistent with the beta fit.
        significant_peaks, fap_level = find_significant_peaks(
            time_numeric, processed_data, frequency, dy=dy, fap_threshold=fap_threshold
        )
        interpretation['significant_peaks'] = significant_peaks
        interpretation['fap_level'] = fap_level

    if do_plot:
        if output_path is None:
            warnings.warn("`do_plot` is True but `output_path` is not specified. Plot will not be saved.", UserWarning)

        plot_spectrum(
            frequency,
            power,
            fit_results=interpretation, # Pass the whole dict now
            analysis_type=analysis_type,
            output_path=output_path,
            param_name=param_name
        )

    if 'interpretation' not in interpretation:
        interpretation['interpretation'] = interpretation.get('summary_text', '')

    # The test expects the interpretation text in a specific key.
    interpretation['interpretation'] = interpretation.get('summary_text')

    return interpretation

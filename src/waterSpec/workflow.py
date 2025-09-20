from .data_loader import load_data
from .preprocessor import preprocess_data
from .spectral_analyzer import calculate_periodogram, find_significant_peaks
from .fitter import fit_spectrum_with_bootstrap, fit_segmented_spectrum
from .interpreter import interpret_results
from .plotting import plot_spectrum
from .frequency_generator import generate_frequency_grid
import warnings
import numpy as np

def run_analysis(
    file_path,
    time_col,
    data_col,
    error_col=None,
    param_name=None,
    censor_strategy='drop',
    censor_options=None,
    log_transform_data=False,
    detrend_method='linear',
    normalize_data=False,
    detrend_options=None,
    analysis_type='standard',
    n_bootstraps=1000,
    fap_threshold=None,
    grid_type='log',
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
        censor_options=censor_options,
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

    custom_frequency = generate_frequency_grid(time_numeric, grid_type=grid_type)
    frequency, power, ls_obj = calculate_periodogram(
        time_numeric, processed_data, frequency=custom_frequency, dy=dy
    )

    # --- Analysis and Fitting ---
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

    # --- Interpretation ---
    if param_name is None:
        param_name = data_col

    if beta is None or not np.isfinite(beta):
        interp_results = {"summary_text": "Analysis failed: Could not determine a valid beta value."}
    else:
        interp_results = interpret_results(beta, ci=ci, param_name=param_name)

    # --- Result Aggregation ---
    results = {}
    results.update(fit_results)
    results.update(interp_results)

    # For clarity and backward compatibility, create a specific 'interpretation' key
    # that holds the main summary text.
    results['interpretation'] = results.get('summary_text')


    # --- Peak Significance ---
    if fap_threshold is not None:
        # Note: The LombScargle object `ls_obj` was created with the processed (e.g., detrended)
        # data. This is consistent with the beta fit and ensures that the FAP calculations
        # are based on the same data used for the spectral slope analysis.
        significant_peaks, fap_level = find_significant_peaks(
            ls_obj, frequency, power, fap_threshold=fap_threshold
        )
        results['significant_peaks'] = significant_peaks
        results['fap_level'] = fap_level
        results['fap_threshold'] = fap_threshold

    # --- Plotting ---
    if do_plot:
        if output_path is None:
            warnings.warn("`do_plot` is True but `output_path` is not specified. Plot will not be saved.", UserWarning)

        plot_spectrum(
            frequency,
            power,
            fit_results=results,
            analysis_type=analysis_type,
            output_path=output_path,
            param_name=param_name
        )

    return results

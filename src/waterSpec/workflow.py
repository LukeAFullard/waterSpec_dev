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
    analysis_type='auto',
    fit_method='theil-sen',
    n_bootstraps=1000,
    fap_threshold=None,
    grid_type='log',
    do_plot=False,
    output_path=None
):
    """
    High-level function to run a complete spectral analysis workflow.

    Args:
        file_path (str): Path to the data file.
        time_col (str): Name of the time column.
        data_col (str): Name of the data column.
        error_col (str, optional): Name of the measurement error column. Defaults to None.
        param_name (str, optional): Name of the parameter being analyzed (for plots/text).
                                    Defaults to `data_col`.
        censor_strategy (str, optional): Strategy for handling censored data.
                                         Defaults to 'drop'.
        censor_options (dict, optional): Options for the censoring strategy.
        log_transform_data (bool, optional): If True, log-transform the data. Defaults to False.
        detrend_method (str, optional): Method for detrending ('linear' or 'loess').
                                        Defaults to 'linear'.
        normalize_data (bool, optional): If True, normalize the data. Defaults to False.
        detrend_options (dict, optional): Options for the detrending method.
        analysis_type (str, optional): Type of analysis ('standard' or 'segmented').
                                       Defaults to 'standard'.
        fit_method (str, optional): Method for spectral slope fitting ('theil-sen' or 'ols').
                                    Defaults to 'theil-sen'.
        n_bootstraps (int, optional): Number of bootstrap samples for CI. Defaults to 1000.
        fap_threshold (float, optional): False Alarm Probability threshold for peak detection.
                                         Defaults to None.
        grid_type (str, optional): Type of frequency grid ('log' or 'linear'). Defaults to 'log'.
        do_plot (bool, optional): If True, generate and save a plot. Defaults to False.
        output_path (str, optional): Path to save the plot. Required if `do_plot` is True.

    Returns:
        dict: A dictionary containing all analysis results.
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
    original_analysis_type = analysis_type

    if analysis_type == 'auto':
        # Perform both analyses
        standard_results = fit_spectrum_with_bootstrap(
            frequency, power, method=fit_method, n_bootstraps=n_bootstraps
        )
        segmented_results = fit_segmented_spectrum(frequency, power)

        # Compare BIC values
        # A lower BIC is preferred. We handle NaN cases by treating them as infinitely bad.
        bic_standard = standard_results.get('bic', np.inf)
        bic_segmented = segmented_results.get('bic', np.inf)

        if np.isnan(bic_standard): bic_standard = np.inf
        if np.isnan(bic_segmented): bic_segmented = np.inf

        # Choose the best model and create a comprehensive results dictionary
        if bic_segmented < bic_standard:
            chosen_model_results = segmented_results
            analysis_type = 'segmented'
        else:
            chosen_model_results = standard_results
            analysis_type = 'standard'

        # Start with the chosen model's results at the top level for compatibility
        fit_results = chosen_model_results.copy()

        # Add detailed results from both fits under specific keys
        fit_results['standard_fit'] = standard_results
        fit_results['segmented_fit'] = segmented_results

        # Store comparison info
        fit_results['bic_comparison'] = {'standard': bic_standard, 'segmented': bic_segmented}
        fit_results['chosen_model'] = analysis_type
        # Also store the original analysis type requested
        fit_results['analysis_mode'] = original_analysis_type


    elif analysis_type == 'standard':
        fit_results = fit_spectrum_with_bootstrap(
            frequency, power, method=fit_method, n_bootstraps=n_bootstraps
        )
    elif analysis_type == 'segmented':
        fit_results = fit_segmented_spectrum(frequency, power)
    else:
        raise ValueError("analysis_type must be 'standard', 'segmented', or 'auto'")

    # --- Peak Significance ---
    if fap_threshold is not None:
        # Note: The LombScargle object `ls_obj` was created with the processed (e.g., detrended)
        # data. This is consistent with the beta fit and ensures that the FAP calculations
        # are based on the same data used for the spectral slope analysis.
        significant_peaks, fap_level = find_significant_peaks(
            ls_obj, frequency, power, fap_threshold=fap_threshold
        )
        # Add peak info to the fit_results so the interpreter can see it
        fit_results['significant_peaks'] = significant_peaks
        fit_results['fap_level'] = fap_level
        fit_results['fap_threshold'] = fap_threshold

    # --- Interpretation ---
    if param_name is None:
        param_name = data_col

    # Check if the fit was successful before interpreting
    fit_successful = ('beta' in fit_results and np.isfinite(fit_results['beta'])) or \
                     ('beta1' in fit_results and np.isfinite(fit_results['beta1']))

    if not fit_successful:
        interp_results = {"summary_text": "Analysis failed: Could not determine a valid spectral slope."}
        # Ensure essential keys exist even in failure
        fit_results.setdefault('summary_text', interp_results["summary_text"])
        results = fit_results
    else:
        interp_results = interpret_results(fit_results, param_name=param_name)
        # --- Result Aggregation ---
        results = {}
        results.update(fit_results)
        results.update(interp_results)

    # For clarity and backward compatibility, create a specific 'interpretation' key
    # that holds the main summary text.
    results['interpretation'] = results.get('summary_text')

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

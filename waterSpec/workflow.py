from .data_loader import load_data
from .preprocessor import detrend, handle_censored_data, detrend_loess
from .spectral_analyzer import calculate_periodogram
from .fitter import fit_spectrum_with_bootstrap, fit_segmented_spectrum
from .interpreter import interpret_beta
from .plotting import plot_spectrum

def run_analysis(file_path, time_col, data_col, n_bootstraps=1000,
                 censor_strategy='drop', lower_multiplier=0.5, upper_multiplier=1.1,
                 analysis_type='standard', detrend_method='linear',
                 do_plot=False, output_path=None):
    """
    Runs the full spectral analysis workflow on a given time series file.

    This high-level function handles loading, preprocessing, spectral analysis,
    fitting, interpretation, and plotting.

    Args:
        file_path (str): The path to the data file.
        time_col (str): The name of the column containing the timestamps.
        data_col (str): The name of the column containing the data values.
        n_bootstraps (int, optional): The number of bootstrap samples for uncertainty analysis.
                                      Defaults to 1000.
        censor_strategy (str, optional): The strategy for handling censored data.
                                         One of ['drop', 'use_detection_limit', 'multiplier']. Defaults to 'drop'.
        lower_multiplier (float, optional): Multiplier for left-censored data ('<'). Defaults to 0.5.
        upper_multiplier (float, optional): Multiplier for right-censored data ('>'). Defaults to 1.1.
        analysis_type (str, optional): The type of spectral fit to perform.
                                       One of ['standard', 'segmented']. Defaults to 'standard'.
        detrend_method (str, optional): The detrending method to use.
                                        One of ['linear', 'loess']. Defaults to 'linear'.
        do_plot (bool, optional): If True, a plot of the spectrum will be generated.
                                  Defaults to False.
        output_path (str, optional): The path to save the plot image. If None and do_plot is True,
                                     the plot is displayed interactively. Defaults to None.

    Returns:
        dict: A dictionary containing the complete analysis results.
    """
    # 1. Load data
    time, data = load_data(file_path, time_col, data_col)

    # 2. Preprocess data
    # Handle censored data first
    numeric_data = handle_censored_data(
        data,
        strategy=censor_strategy,
        lower_multiplier=lower_multiplier,
        upper_multiplier=upper_multiplier
    )
    # Then detrend
    if detrend_method == 'linear':
        preprocessed_data = detrend(numeric_data)
    elif detrend_method == 'loess':
        preprocessed_data = detrend_loess(time, numeric_data)
    else:
        raise ValueError("Invalid detrend_method. Choose from ['linear', 'loess']")

    # 3. Calculate periodogram
    frequency, power = calculate_periodogram(time, preprocessed_data)

    # 4. Fit spectrum
    if analysis_type == 'standard':
        fit_results = fit_spectrum_with_bootstrap(frequency, power, n_bootstraps=n_bootstraps)
        # 5. Interpret beta
        interpretation = interpret_beta(fit_results['beta'])
        final_results = fit_results.copy()
        final_results['interpretation'] = interpretation
    elif analysis_type == 'segmented':
        fit_results = fit_segmented_spectrum(frequency, power)
        # Interpretation for segmented fit can be more complex, returning raw results for now.
        final_results = fit_results
    else:
        raise ValueError("Invalid analysis_type. Choose from ['standard', 'segmented']")

    # 6. Plot if requested
    if do_plot:
        # Note: plotting for segmented fit is not yet customized.
        plot_spectrum(frequency, power, fit_results=fit_results, output_path=output_path, show=(output_path is None))

    return final_results

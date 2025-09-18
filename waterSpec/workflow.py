from .data_loader import load_data
from .preprocessor import detrend
from .spectral_analyzer import calculate_periodogram
from .fitter import fit_spectrum_with_bootstrap
from .interpreter import interpret_beta
from .plotting import plot_spectrum

def run_analysis(file_path, time_col, data_col, n_bootstraps=1000, do_plot=False, output_path=None):
    """
    Runs the full spectral analysis workflow on a given time series file.

    This high-level function handles loading, preprocessing, spectral analysis,
    fitting, interpretation, and plotting.

    Args:
        file_path (str): The path to the CSV file.
        time_col (str): The name of the column containing the timestamps.
        data_col (str): The name of the column containing the data values.
        n_bootstraps (int, optional): The number of bootstrap samples for uncertainty analysis.
                                      Defaults to 1000.
        do_plot (bool, optional): If True, a plot of the spectrum will be generated.
                                  Defaults to False.
        output_path (str, optional): The path to save the plot image. If None and do_plot is True,
                                     the plot is displayed interactively. Defaults to None.

    Returns:
        dict: A dictionary containing the complete analysis results, including:
              - 'beta', 'r_squared', 'intercept', 'stderr'
              - 'beta_ci_lower', 'beta_ci_upper'
              - 'interpretation'
    """
    # 1. Load data
    time, data = load_data(file_path, time_col, data_col)

    # 2. Preprocess data (currently just detrending)
    preprocessed_data = detrend(data)

    # 3. Calculate periodogram
    frequency, power = calculate_periodogram(time, preprocessed_data)

    # 4. Fit spectrum with uncertainty
    fit_results = fit_spectrum_with_bootstrap(frequency, power, n_bootstraps=n_bootstraps)

    # 5. Interpret beta
    interpretation = interpret_beta(fit_results['beta'])

    # 6. Combine results
    final_results = fit_results.copy()
    final_results['interpretation'] = interpretation

    # 7. Plot if requested
    if do_plot:
        plot_spectrum(frequency, power, fit_results=fit_results, output_path=output_path, show=(output_path is None))

    return final_results

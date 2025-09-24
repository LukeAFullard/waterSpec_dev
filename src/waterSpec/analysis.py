import numpy as np
import warnings
import os

from .data_loader import load_data
from .preprocessor import preprocess_data
from .frequency_generator import generate_frequency_grid
from .spectral_analyzer import calculate_periodogram, find_significant_peaks, find_peaks_via_residuals
from .fitter import fit_spectrum_with_bootstrap, fit_segmented_spectrum
from .interpreter import interpret_results
from .plotting import plot_spectrum

class Analysis:
    """
    A class to perform a complete spectral analysis of a time series.

    This class encapsulates the data loading, preprocessing, analysis,
    and output generation, providing a streamlined workflow.
    """
    def __init__(
        self,
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
        detrend_options=None
    ):
        """
        Initializes the Analysis object by loading and preprocessing the data.

        Args:
            file_path (str): Path to the data file.
            time_col (str): Name of the time column.
            data_col (str): Name of the data column.
            error_col (str, optional): Name of the measurement error column. Defaults to None.
            param_name (str, optional): Name of the parameter for plots/text. Defaults to data_col.
            censor_strategy (str, optional): Strategy for handling censored data. Defaults to 'drop'.
            censor_options (dict, optional): Options for the censoring strategy.
            log_transform_data (bool, optional): If True, log-transform the data. Defaults to False.
            detrend_method (str, optional): Method for detrending ('linear' or 'loess'). Defaults to 'linear'.
            normalize_data (bool, optional): If True, normalize the data. Defaults to False.
            detrend_options (dict, optional): Options for the detrending method.
        """
        self.param_name = param_name if param_name is not None else data_col

        # Load and preprocess data
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

        # Store the processed data as instance attributes
        self.time = time_numeric[valid_indices]
        self.data = processed_data[valid_indices]
        self.errors = processed_errors[valid_indices] if processed_errors is not None else None

        # Attributes to be populated by analysis methods
        self.results = None
        self.frequency = None
        self.power = None
        self.ls_obj = None

    def run_full_analysis(
        self,
        output_dir,
        fit_method='theil-sen',
        n_bootstraps=1000,
        fap_threshold=0.01,
        grid_type='log',
        fap_method='baluev',
        normalization='standard',
        peak_detection_method='fap',
        peak_detection_ci=95
    ):
        """
        Runs the complete analysis workflow and saves all outputs to a directory.

        This method performs an automatic analysis to choose the best model
        (standard or segmented), calculates significant peaks, and saves a
        plot and a text summary of the results.

        Args:
            output_dir (str): The path to the directory where outputs will be saved.
            fit_method (str, optional): Method for spectral slope fitting. Defaults to 'theil-sen'.
            n_bootstraps (int, optional): Number of bootstrap samples for CI. Defaults to 1000.
            grid_type (str, optional): Type of frequency grid ('log' or 'linear'). Defaults to 'log'.
            peak_detection_method (str, optional): Method for peak detection.
                                                   'residual' (default) uses the new robust method.
                                                   'fap' uses the old False Alarm Probability method.
            peak_detection_ci (int, optional): Confidence interval for residual method. Defaults to 95.
            fap_threshold (float, optional): FAP threshold for 'fap' method. Defaults to 0.01.
            fap_method (str, optional): Method for FAP calculation ('bootstrap', 'baluev', etc.).
                                       Defaults to 'bootstrap'.
            normalization (str, optional): Normalization for the periodogram. Defaults to 'standard'.

        Returns:
            dict: A dictionary containing all analysis results.
        """
        # Ensure the output directory exists
        os.makedirs(output_dir, exist_ok=True)

        # --- Periodogram Calculation ---
        self.frequency = generate_frequency_grid(self.time, grid_type=grid_type)
        self.frequency, self.power, self.ls_obj = calculate_periodogram(
            self.time, self.data, frequency=self.frequency, dy=self.errors,
            normalization=normalization
        )

        # --- Auto-Analysis Mode ---
        standard_results = fit_spectrum_with_bootstrap(
            self.frequency, self.power, method=fit_method, n_bootstraps=n_bootstraps
        )
        segmented_results = fit_segmented_spectrum(self.frequency, self.power)

        is_segmented_valid = 'beta1' in segmented_results and np.isfinite(segmented_results['beta1'])
        chosen_model_results = standard_results
        analysis_type = 'standard'

        if is_segmented_valid:
            bic_standard = standard_results.get('bic', np.inf)
            bic_segmented = segmented_results.get('bic', np.inf)
            if np.isnan(bic_standard): bic_standard = np.inf
            if np.isnan(bic_segmented): bic_segmented = np.inf
            if bic_segmented < bic_standard:
                chosen_model_results = segmented_results
                analysis_type = 'segmented'

        fit_results = chosen_model_results.copy()
        fit_results['standard_fit'] = standard_results
        fit_results['segmented_fit'] = segmented_results
        fit_results['bic_comparison'] = {
            'standard': standard_results.get('bic'),
            'segmented': segmented_results.get('bic')
        }
        fit_results['chosen_model'] = analysis_type
        fit_results['analysis_mode'] = 'auto' # This method always runs in auto mode

        # --- Peak Significance (only if fit was successful) ---
        fit_successful = ('beta' in fit_results and np.isfinite(fit_results['beta'])) or \
                         ('beta1' in fit_results and np.isfinite(fit_results['beta1']))

        if fit_successful:
            if peak_detection_method == 'residual':
                significant_peaks, residual_threshold = find_peaks_via_residuals(
                    fit_results, ci=peak_detection_ci
                )
                fit_results['significant_peaks'] = significant_peaks
                fit_results['residual_threshold'] = residual_threshold
                fit_results['peak_detection_ci'] = peak_detection_ci

            elif peak_detection_method == 'fap':
                if fap_threshold is not None:
                    significant_peaks, fap_level = find_significant_peaks(
                        self.ls_obj, self.frequency, self.power,
                        fap_threshold=fap_threshold, fap_method=fap_method
                    )
                    fit_results['significant_peaks'] = significant_peaks
                    fit_results['fap_level'] = fap_level
                    fit_results['fap_threshold'] = fap_threshold
            elif peak_detection_method is not None:
                warnings.warn(f"Unknown peak_detection_method '{peak_detection_method}'. No peak detection will be performed.", UserWarning)

        # --- Interpretation ---
        if not fit_successful:
            interp_results = {"summary_text": "Analysis failed: Could not determine a valid spectral slope."}
            fit_results.setdefault('summary_text', interp_results["summary_text"])
            self.results = fit_results
        else:
            interp_results = interpret_results(fit_results, param_name=self.param_name)
            self.results = {**fit_results, **interp_results}
            self.results['interpretation'] = self.results.get('summary_text')

        # --- Generate and Save Outputs ---
        # Plot
        plot_path = os.path.join(output_dir, f"{self.param_name}_spectrum_plot.png")
        plot_spectrum(
            self.frequency,
            self.power,
            fit_results=self.results,
            analysis_type=analysis_type,
            output_path=plot_path,
            param_name=self.param_name
        )

        # Summary Text
        summary_path = os.path.join(output_dir, f"{self.param_name}_summary.txt")
        with open(summary_path, 'w') as f:
            f.write(self.results['summary_text'])

        print(f"Analysis complete. Outputs saved to '{output_dir}'.")
        return self.results

import logging
import os
import re
import warnings

import numpy as np

from .data_loader import load_data
from .fitter import fit_segmented_spectrum, fit_standard_model
from .frequency_generator import generate_frequency_grid
from .interpreter import interpret_results
from .plotting import plot_spectrum
from .preprocessor import preprocess_data
from .spectral_analyzer import (
    calculate_periodogram,
    find_peaks_via_residuals,
    find_significant_peaks,
)

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
        time_format=None,
        sheet_name=0,
        param_name=None,
        censor_strategy="drop",
        censor_options=None,
        log_transform_data=False,
        detrend_method="linear",
        normalize_data=False,
        detrend_options=None,
        min_valid_data_points=10,
        verbose=False,
    ):
        """
        Initializes the Analysis object by loading and preprocessing the data.

        Args:
            file_path (str): The full path to the data file (CSV or Excel).
            time_col (str): The name of the column containing timestamps.
            data_col (str): The name of the column containing data values.
            error_col (str, optional): The name of the column for measurement
                errors. If provided, these will be used for weighted periodogram
                calculation. Defaults to None.
            time_format (str, optional): The specific `strptime` format of the
                time column to speed up parsing. If None, it is inferred.
            sheet_name (str or int, optional): If loading an Excel file, specify
                the sheet name or index. Defaults to 0.
            param_name (str, optional): A descriptive name for the data parameter
                being analyzed (e.g., "Nitrate Concentration"). Used for plot
                titles and summaries. If None, defaults to `data_col`.
            censor_strategy (str, optional): The strategy for handling censored
                (non-detect) data. See `preprocess.py` for options.
            censor_options (dict, optional): Options for the censoring strategy.
            log_transform_data (bool, optional): If True, log-transform the data.
            detrend_method (str, optional): Method to detrend the time series
                ('linear', 'loess', or None). Defaults to 'linear'.
            normalize_data (bool, optional): If True, normalize the data.
            detrend_options (dict, optional): Options for the detrending method.
            min_valid_data_points (int, optional): The minimum number of valid
                data points required to proceed with an analysis. Defaults to 10.
            verbose (bool, optional): If True, sets logging level to INFO.
                Defaults to False (logging level WARNING).
        """
        self.param_name = param_name if param_name is not None else data_col
        self._setup_logger(level=logging.INFO if verbose else logging.WARNING)

        if not isinstance(min_valid_data_points, int) or min_valid_data_points <= 0:
            raise ValueError("`min_valid_data_points` must be a positive integer.")
        self.min_valid_data_points = min_valid_data_points

        # Load and preprocess data
        self.logger.info("Loading and preprocessing data...")
        time_numeric, data_series, error_series = load_data(
            file_path,
            time_col,
            data_col,
            error_col=error_col,
            time_format=time_format,
            sheet_name=sheet_name,
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
            detrend_options=detrend_options,
        )

        valid_indices = ~np.isnan(processed_data)
        if np.sum(valid_indices) < self.min_valid_data_points:
            raise ValueError(
                f"Not enough valid data points ({np.sum(valid_indices)}) "
                f"remaining after preprocessing. Minimum required: "
                f"{self.min_valid_data_points}."
            )

        self.time = time_numeric[valid_indices]
        self.data = processed_data[valid_indices]
        self.errors = (
            processed_errors[valid_indices] if processed_errors is not None else None
        )
        self.logger.info("Data loading and preprocessing complete.")

        # Attributes to be populated by analysis methods
        self.results = None
        self.frequency = None
        self.power = None
        self.ls_obj = None

    def _setup_logger(self, level):
        """Configures a logger for the Analysis instance."""
        self.logger = logging.getLogger(f"waterSpec.{self.param_name}")
        # Prevent duplicate handlers if logger is already configured
        if not self.logger.handlers:
            self.logger.setLevel(level)
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
            )
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)

    def _sanitize_filename(self, filename):
        """Sanitizes a string to be a valid filename."""
        s = str(filename).strip().replace(" ", "_")
        s = re.sub(r"(?u)[^-\w.]", "", s)
        return s

    def _calculate_periodogram(self, grid_type, normalization, num_grid_points):
        """Generates frequency grid and calculates the Lomb-Scargle periodogram."""
        self.logger.info("Calculating Lomb-Scargle periodogram...")
        self.frequency = generate_frequency_grid(
            self.time, num_points=num_grid_points, grid_type=grid_type
        )
        self.frequency, self.power, self.ls_obj = calculate_periodogram(
            self.time,
            self.data,
            frequency=self.frequency,
            dy=self.errors,
            normalization=normalization,
        )
        self.logger.info("Periodogram calculation complete.")

    def _perform_model_selection(
        self, fit_method, ci_method, n_bootstraps, p_threshold, max_breakpoints, seed
    ):
        """
        Performs fits for models with different numbers of breakpoints and
        selects the best one using BIC.
        """
        self.logger.info("Performing model selection based on BIC...")
        all_models = []

        # Fit the standard model (0 breakpoints)
        self.logger.info("Fitting standard model (0 breakpoints)...")
        standard_results = fit_standard_model(
            self.frequency,
            self.power,
            method=fit_method,
            ci_method=ci_method,
            n_bootstraps=n_bootstraps,
            seed=seed,
            logger=self.logger,
        )
        if "bic" in standard_results and np.isfinite(standard_results["bic"]):
            standard_results["model_type"] = "standard"
            standard_results["n_breakpoints"] = 0
            all_models.append(standard_results)
            self.logger.info(f"Standard model fit complete. BIC: {standard_results['bic']:.2f}")

        # Fit segmented models (1 and possibly 2 breakpoints)
        for n_bp in range(1, max_breakpoints + 1):
            self.logger.info(f"Fitting segmented model ({n_bp} breakpoint(s))...")
            seg_results = fit_segmented_spectrum(
                self.frequency,
                self.power,
                n_breakpoints=n_bp,
                p_threshold=p_threshold,
                ci_method=ci_method,
                n_bootstraps=n_bootstraps,
                seed=seed,
                logger=self.logger,
            )
            if "bic" in seg_results and np.isfinite(seg_results["bic"]):
                seg_results["model_type"] = f"segmented_{n_bp}bp"
                all_models.append(seg_results)
                self.logger.info(f"Segmented model ({n_bp} bp) fit complete. BIC: {seg_results['bic']:.2f}")

        if not all_models:
            self.logger.error("Model fitting failed for all model types.")
            return {
                "betas": [np.nan],
                "n_breakpoints": 0,
                "chosen_model_type": "standard",
                "summary_text": "Model fitting failed for all model types.",
            }

        best_model = min(all_models, key=lambda x: x["bic"])
        self.logger.info(
            f"Best model selected: {best_model['model_type']} "
            f"(BIC: {best_model['bic']:.2f})"
        )

        fit_results = best_model.copy()
        fit_results["all_models"] = all_models
        fit_results["chosen_model"] = best_model["model_type"]
        fit_results["analysis_mode"] = "auto"
        fit_results["ci_method"] = ci_method

        if best_model["n_breakpoints"] == 0:
            fit_results["chosen_model_type"] = "standard"
        else:
            fit_results["chosen_model_type"] = "segmented"

        return fit_results

    def _detect_significant_peaks(
        self,
        fit_results,
        peak_detection_method,
        peak_detection_ci,
        fap_threshold,
        fap_method,
    ):
        """Detects significant peaks based on the chosen method."""
        is_standard_success = "beta" in fit_results and np.isfinite(fit_results.get("beta"))
        is_segmented_success = (
            "betas" in fit_results
            and len(fit_results.get("betas", [])) > 0
            and np.isfinite(fit_results["betas"][0])
        )
        fit_successful = is_standard_success or is_segmented_success

        if not fit_successful:
            return fit_results

        self.logger.info(f"Detecting significant peaks using '{peak_detection_method}' method...")
        if peak_detection_method == "residual":
            if fap_method != "baluev" or fap_threshold != 0.01:
                self.logger.warning(
                    "'peak_detection_method' is 'residual', so 'fap_method' and "
                    "'fap_threshold' parameters are ignored."
                )
            peaks, threshold = find_peaks_via_residuals(
                fit_results, ci=peak_detection_ci
            )
            fit_results["significant_peaks"] = peaks
            fit_results["residual_threshold"] = threshold
            fit_results["peak_detection_ci"] = peak_detection_ci
        elif peak_detection_method == "fap" and fap_threshold is not None:
            peaks, level = find_significant_peaks(
                self.ls_obj,
                self.frequency,
                self.power,
                fap_threshold=fap_threshold,
                fap_method=fap_method,
            )
            fit_results["significant_peaks"] = peaks
            fit_results["fap_level"] = level
            fit_results["fap_threshold"] = fap_threshold
        elif peak_detection_method is not None and peak_detection_method != "fap":
            self.logger.warning(
                f"Unknown peak_detection_method '{peak_detection_method}'. "
                "No peak detection performed."
            )

        if "significant_peaks" in fit_results:
            self.logger.info(
                f"Found {len(fit_results['significant_peaks'])} significant peaks."
            )
        return fit_results

    def _generate_outputs(self, results, output_dir):
        """Generates and saves the plot and summary text file."""
        self.logger.info(f"Generating outputs in directory: {output_dir}")
        os.makedirs(output_dir, exist_ok=True)
        sanitized_name = self._sanitize_filename(self.param_name)

        # Plot
        plot_path = os.path.join(output_dir, f"{sanitized_name}_spectrum_plot.png")
        plot_spectrum(
            self.frequency,
            self.power,
            fit_results=results,
            analysis_type=results["chosen_model_type"],
            output_path=plot_path,
            param_name=self.param_name,
        )

        # Summary Text
        summary_path = os.path.join(output_dir, f"{sanitized_name}_summary.txt")
        with open(summary_path, "w") as f:
            f.write(results["summary_text"])
        self.logger.info(f"Plot saved to {plot_path}")
        self.logger.info(f"Summary saved to {summary_path}")

    def run_full_analysis(
        self,
        output_dir,
        fit_method="theil-sen",
        ci_method="bootstrap",
        n_bootstraps=1000,
        fap_threshold=0.01,
        grid_type="log",
        num_grid_points=200,
        fap_method="baluev",
        normalization="standard",
        peak_detection_method="residual",
        peak_detection_ci=95,
        p_threshold=0.05,
        max_breakpoints=1,
        seed=None,
    ):
        """
        Runs the complete analysis workflow and saves all outputs to a directory.

        This method performs an automatic analysis to choose the best spectral
        model (standard or segmented), calculates significant peaks, and saves a
        plot and a text summary of the results.

        Args:
            output_dir (str): Path to the directory where outputs will be saved.
            fit_method (str, optional): Method for spectral slope fitting.
                Can be 'theil-sen' (default) or 'ols'.
            ci_method (str, optional): The method for calculating confidence
                intervals. Can be 'bootstrap' (default) for a robust,
                non-parametric method, or 'parametric' for a faster method
                based on statistical theory.
            n_bootstraps (int, optional): Number of bootstrap samples for CI.
                Only used if `ci_method` is 'bootstrap'. Defaults to 1000.
            grid_type (str, optional): Type of frequency grid ('log' or 'linear').
                Defaults to 'log'.
            num_grid_points (int, optional): The number of points to generate
                for the frequency grid. Defaults to 200.
            peak_detection_method (str, optional): Method for peak detection.
                - 'residual' (default): Identifies peaks that are significant
                  outliers from the fitted spectral model. Recommended.
                - 'fap': Uses the traditional False Alarm Probability (FAP)
                  method.
                When using 'residual', `fap_method` and `fap_threshold` are ignored.
            peak_detection_ci (int, optional): The confidence interval (in %)
                to use for the residual-based peak detection method.
                Defaults to 95.
            fap_threshold (float, optional): The significance level for the FAP
                peak detection method. Only used when `peak_detection_method`
                is 'fap'. Defaults to 0.01.
            fap_method (str, optional): The method for calculating the FAP.
                Can be 'baluev' (default) or 'bootstrap'. Only used when
                `peak_detection_method` is 'fap'.
            normalization (str, optional): Normalization for the periodogram.
                Defaults to 'standard'.
            p_threshold (float, optional): The p-value threshold for the
                Davies test for a significant breakpoint in segmented
                regression (only for 1-breakpoint models). Defaults to 0.05.
            max_breakpoints (int, optional): The maximum number of breakpoints
                to consider in segmented regression (0, 1, or 2). Defaults to 1.
            seed (int, optional): A seed for the random number generator to
                ensure reproducibility of bootstrap analysis. Defaults to None.

        Returns:
            dict: A dictionary containing all analysis results.
        """
        # Validate ci_method
        if ci_method not in ["bootstrap", "parametric"]:
            raise ValueError(
                "Invalid `ci_method`. Must be 'bootstrap' or 'parametric'."
            )

        # 1. Calculate Periodogram
        self._calculate_periodogram(grid_type, normalization, num_grid_points)

        # 2. Fit Spectrum and Select Best Model
        fit_results = self._perform_model_selection(
            fit_method, ci_method, n_bootstraps, p_threshold, max_breakpoints, seed
        )

        # 3. Detect Significant Peaks
        fit_results = self._detect_significant_peaks(
            fit_results,
            peak_detection_method,
            peak_detection_ci,
            fap_threshold,
            fap_method,
        )

        # 4. Interpret Results
        self.logger.info("Interpreting final results and generating summary...")
        is_standard_success = "beta" in fit_results and np.isfinite(
            fit_results.get("beta")
        )
        is_segmented_success = (
            "betas" in fit_results
            and len(fit_results.get("betas", [])) > 0
            and np.isfinite(fit_results["betas"][0])
        )
        fit_successful = is_standard_success or is_segmented_success
        if not fit_successful:
            interp_results = {
                "summary_text": (
                    "Analysis failed: Could not determine a valid spectral slope."
                )
            }
            self.results = {**fit_results, **interp_results}
        else:
            interp_results = interpret_results(fit_results, param_name=self.param_name)
            self.results = {**fit_results, **interp_results}

        # 5. Generate and Save Outputs
        self._generate_outputs(self.results, output_dir)

        self.logger.info(f"Analysis complete. Outputs saved to '{output_dir}'.")
        return self.results

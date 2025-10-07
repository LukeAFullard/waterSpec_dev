import logging
import os
import re
import warnings
from typing import Dict, Optional

import numpy as np
import pandas as pd


from .changepoint_detector import (
    detect_changepoint_pelt,
    get_changepoint_time,
)
from .data_loader import load_data, process_dataframe
from .fitter import fit_segmented_spectrum, fit_standard_model
from .frequency_generator import generate_frequency_grid
from .interpreter import interpret_results
from .plotting import plot_changepoint_analysis, plot_spectrum
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
        time_col: str,
        data_col: str,
        file_path: Optional[str] = None,
        dataframe: Optional[pd.DataFrame] = None,
        time_array: Optional[np.ndarray] = None,
        data_array: Optional[np.ndarray] = None,
        error_array: Optional[np.ndarray] = None,
        error_col: Optional[str] = None,
        time_format: Optional[str] = None,
        input_time_unit: Optional[str] = None,
        sheet_name: int = 0,
        time_unit: str = "seconds",
        param_name: Optional[str] = None,
        censor_strategy: str = "drop",
        censor_options: Optional[Dict] = None,
        log_transform_data: bool = False,
        detrend_method: Optional[str] = None,
        normalize_data: bool = False,
        detrend_options: Optional[Dict] = None,
        min_valid_data_points: int = 10,
        changepoint_mode: str = "none",
        changepoint_index: Optional[int] = None,
        changepoint_options: Optional[Dict] = None,
    ):
        """
        Initializes the Analysis object by loading and preprocessing the data.

        The constructor accepts one of three data sources:
        1. A file path (`file_path`).
        2. A pandas DataFrame (`dataframe`).
        3. NumPy arrays (`time_array`, `data_array`, and optionally `error_array`).

        Args:
            time_col (str): The name of the column for timestamps.
            data_col (str): The name of the column for data values.
            file_path (str, optional): Path to the data file (CSV, JSON, Excel).
            dataframe (pd.DataFrame, optional): DataFrame with time series data.
            time_array (np.ndarray, optional): NumPy array of time values.
            data_array (np.ndarray, optional): NumPy array of data values.
            error_array (np.ndarray, optional): NumPy array of error values.
            error_col (str, optional): Name for the error column.
            time_format (str, optional): The `strptime` format of the time
                column.
            input_time_unit (str, optional): The unit of a numeric time column.
            sheet_name (int, optional): Sheet index for Excel files.
            time_unit (str, optional): The desired output unit for time.
            param_name (str, optional): A descriptive name for the data parameter.
            censor_strategy (str, optional): The strategy for handling censored data.
            censor_options (dict, optional): Options for the censoring strategy.
                This can include `decimal_separator` ('.' or ',') to handle
                different number formats.
            log_transform_data (bool, optional): If True, log-transform the data.
            detrend_method (str, optional): Method to detrend the time series.
            normalize_data (bool, optional): If True, normalize the data.
            detrend_options (dict, optional): Options for the detrending method.
            min_valid_data_points (int, optional): Minimum valid data points.
            verbose (bool, optional): If True, sets logging level to INFO.
            changepoint_mode (str, optional): "none", "auto", or "manual".
            changepoint_index (int, optional): Index for manual changepoint.
            changepoint_options (dict, optional): Options for auto changepoint.
        """
        self.param_name = param_name if param_name is not None else data_col
        self.logger = logging.getLogger(__name__)

        if not isinstance(min_valid_data_points, int) or min_valid_data_points <= 0:
            raise ValueError("`min_valid_data_points` must be a positive integer.")
        self.min_valid_data_points = min_valid_data_points
        self.time_unit = time_unit

        # --- Data Input Validation ---
        input_methods = [
            file_path is not None,
            dataframe is not None,
            (time_array is not None and data_array is not None),
        ]
        if sum(input_methods) > 1:
            raise ValueError(
                "Please provide only one data source: `file_path`, `dataframe`, or arrays."
            )

        # --- Data Loading and Preprocessing ---
        self.logger.info("Loading and preprocessing data...")

        if file_path is not None:
            time_numeric, data_series, error_series = load_data(
                file_path=file_path,
                time_col=time_col,
                data_col=data_col,
                error_col=error_col,
                time_format=time_format,
                input_time_unit=input_time_unit,
                sheet_name=sheet_name,
                output_time_unit=self.time_unit,
            )
        elif dataframe is not None:
            time_numeric, data_series, error_series = process_dataframe(
                df=dataframe,
                time_col=time_col,
                data_col=data_col,
                error_col=error_col,
                time_format=time_format,
                input_time_unit=input_time_unit,
                output_time_unit=self.time_unit,
            )
        elif time_array is not None and data_array is not None:
            # Create a temporary DataFrame from the NumPy arrays
            df_dict = {time_col: time_array, data_col: data_array}
            if error_array is not None:
                if error_col is None:
                    error_col = "errors"  # Default name if not provided
                df_dict[error_col] = error_array

            temp_df = pd.DataFrame(df_dict)

            time_numeric, data_series, error_series = process_dataframe(
                df=temp_df,
                time_col=time_col,
                data_col=data_col,
                error_col=error_col,
                time_format=time_format,
                input_time_unit=input_time_unit,
                output_time_unit=self.time_unit,
            )
        else:
            raise ValueError(
                "A valid data source must be provided: either `file_path`, `dataframe`, "
                "or both `time_array` and `data_array`."
            )

        # --- Pre-computation validation ---
        # Validate LOESS configuration early to provide a clear error message.
        # This prevents a late failure inside the `preprocess_data` function.
        detrend_options = detrend_options or {}
        if (
            detrend_method == "loess"
            and error_series is not None
            and detrend_options.get("n_bootstrap", 0) <= 0
        ):
            raise ValueError(
                "When using 'loess' detrending with error data, `n_bootstrap` must be > 0 "
                "for error propagation. Please set `n_bootstrap` in `detrend_options`. "
                "For example: `detrend_options={'n_bootstrap': 100}`."
            )

        processed_data, processed_errors, diagnostics = preprocess_data(
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

        self.preprocessing_diagnostics = diagnostics
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

        # Changepoint configuration
        self.changepoint_mode = changepoint_mode
        self.changepoint_index = changepoint_index
        self.changepoint_options = changepoint_options or {}

        if changepoint_mode not in ["none", "auto", "manual"]:
            raise ValueError("`changepoint_mode` must be 'none', 'auto', or 'manual'.")
        if changepoint_mode == "manual":
            if changepoint_index is None:
                raise ValueError(
                    "`changepoint_index` is required when `changepoint_mode` is 'manual'."
                )
            if not (0 < changepoint_index < len(self.time)):
                raise ValueError(
                    f"Manual `changepoint_index` ({changepoint_index}) is out of "
                    f"the valid data range [1, {len(self.time) - 1}]."
                )

        # Attributes to be populated by analysis methods
        self.results = None
        self.frequency = None
        self.power = None
        self.ls_obj = None

    def _detect_changepoint(self) -> Optional[int]:
        """Detects changepoint if in auto mode."""
        if self.changepoint_mode != "auto":
            return None

        self.logger.info("Running automatic changepoint detection...")
        cp_idx = detect_changepoint_pelt(
            self.time, self.data, **self.changepoint_options
        )

        if cp_idx is None:
            self.logger.info("No significant changepoint detected.")
        else:
            cp_time_str = get_changepoint_time(cp_idx, self.time, self.time_unit)
            self.logger.info(
                f"Changepoint detected at index {cp_idx} (~{cp_time_str})."
            )
        return cp_idx

    def _sanitize_filename(self, filename):
        """Sanitizes a string to be a valid filename."""
        s = str(filename).strip().replace(" ", "_")
        s = re.sub(r"(?u)[^-\w.]", "", s)
        return s

    def _calculate_periodogram(
        self,
        normalization,
        nyquist_factor,
        max_freq,
        samples_per_peak,
    ):
        """Calculates the Lomb-Scargle periodogram using autopower."""
        self.logger.info("Calculating Lomb-Scargle periodogram...")
        # Note: The 'max_freq' from the run_full_analysis method is passed
        # as 'maximum_frequency' to the underlying function.
        self.frequency, self.power, self.ls_obj = calculate_periodogram(
            self.time,
            self.data,
            dy=self.errors,
            normalization=normalization,
            nyquist_factor=nyquist_factor,
            samples_per_peak=samples_per_peak,
            maximum_frequency=max_freq,
        )
        self.logger.info("Periodogram calculation complete.")

    def _perform_model_selection(
        self,
        fit_method,
        ci_method,
        bootstrap_type,
        n_bootstraps,
        p_threshold,
        max_breakpoints,
        seed,
    ):
        """
        Performs fits for models with different numbers of breakpoints and
        selects the best one using BIC.
        """
        self.logger.info("Performing model selection based on BIC...")
        all_models = []
        failed_model_reasons = []

        # If a seed is provided, create a SeedSequence to generate independent
        # seeds for each model. This is crucial for reproducible and
        # independent bootstrap resampling across different models.
        if seed is not None:
            ss = np.random.SeedSequence(seed)
            # We are fitting up to `max_breakpoints` + 1 models (standard, 1bp, 2bp, etc.)
            child_seeds = ss.spawn(max_breakpoints + 1)
        else:
            child_seeds = [None] * (max_breakpoints + 1)

        # Fit the standard model (0 breakpoints)
        self.logger.info("Fitting standard model (0 breakpoints)...")
        try:
            standard_results = fit_standard_model(
                self.frequency,
                self.power,
                method=fit_method,
                ci_method=ci_method,
                bootstrap_type=bootstrap_type,
                n_bootstraps=n_bootstraps,
                seed=child_seeds[0],
                logger=self.logger,
            )
            if "bic" in standard_results and np.isfinite(standard_results["bic"]):
                standard_results["model_type"] = "standard"
                standard_results["n_breakpoints"] = 0
                all_models.append(standard_results)
                self.logger.info(
                    "Standard model fit complete. "
                    f"BIC: {standard_results['bic']:.2f}"
                )
            else:
                reason = standard_results.get("failure_reason", "Unknown error")
                failed_model_reasons.append(f"Standard model (0 breakpoints): {reason}")
                self.logger.warning(f"Standard model fit failed: {reason}")
        except (ValueError, ImportError) as e:
            reason = f"A critical error occurred during standard model setup: {e!r}"
            failed_model_reasons.append(f"Standard model (0 breakpoints): {reason}")
            self.logger.error(
                "Standard model fit crashed due to a critical error: %s", e, exc_info=True
            )
        except Exception as e:
            reason = f"An unexpected error occurred: {e!r}"
            failed_model_reasons.append(f"Standard model (0 breakpoints): {reason}")
            self.logger.error("Standard model fit crashed: %s", e, exc_info=True)

        # Fit segmented models (1 and possibly 2 breakpoints)
        for n_breakpoints in range(1, max_breakpoints + 1):
            self.logger.info(f"Fitting segmented model with {n_breakpoints} breakpoint(s)...")
            try:
                # Spawn a new seed for each model to ensure independent bootstrap samples.
                model_seed = child_seeds[n_breakpoints]
                seg_results = fit_segmented_spectrum(
                    self.frequency,
                    self.power,
                    n_breakpoints=n_breakpoints,
                    p_threshold=p_threshold,
                    ci_method=ci_method,
                    bootstrap_type=bootstrap_type,
                    n_bootstraps=n_bootstraps,
                    seed=model_seed,
                    logger=self.logger,
                )
                if "bic" in seg_results and np.isfinite(seg_results["bic"]):
                    seg_results["model_type"] = f"segmented_{n_breakpoints}bp"
                    all_models.append(seg_results)
                    self.logger.info(
                        f"Segmented model ({n_breakpoints} breakpoint(s)) fit complete. "
                        f"BIC: {seg_results['bic']:.2f}"
                    )
                else:
                    reason = seg_results.get("failure_reason", "Model did not converge or was not significant")
                    failed_model_reasons.append(
                        f"Segmented model ({n_breakpoints} breakpoint(s)): {reason}"
                    )
                    self.logger.warning(
                        f"Segmented model ({n_breakpoints} breakpoint(s)) fit failed: {reason}"
                    )
            except (ValueError, ImportError) as e:
                reason = f"A critical error occurred during segmented model setup: {e!r}"
                failed_model_reasons.append(
                    f"Segmented model ({n_breakpoints} breakpoint(s)): {reason}"
                )
                self.logger.error(
                    "Segmented model (%d breakpoint(s)) fit crashed due to a critical error: %s",
                    n_breakpoints,
                    e,
                    exc_info=True,
                )
            except Exception as e:
                reason = f"An unexpected error occurred: {e!r}"
                failed_model_reasons.append(
                    f"Segmented model ({n_breakpoints} breakpoint(s)): {reason}"
                )
                self.logger.error(
                    "Segmented model (%d breakpoint(s)) fit crashed: %s",
                    n_breakpoints,
                    e,
                    exc_info=True,
                )

        if not all_models:
            failure_summary = (
                "Model fitting failed for all attempted models. Reasons:\n"
                + "\n".join(f"- {reason}" for reason in failed_model_reasons)
            )
            self.logger.error(failure_summary)
            raise RuntimeError(failure_summary)

        best_model = min(all_models, key=lambda x: x["bic"])
        self.logger.info(
            f"Best model selected: {best_model['model_type']} "
            f"(BIC: {best_model['bic']:.2f})"
        )

        fit_results = best_model.copy()
        fit_results["all_models"] = all_models
        fit_results["failed_model_reasons"] = failed_model_reasons
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
        peak_fdr_level,
        fap_threshold,
        fap_method,
    ):
        """Detects significant peaks based on the chosen method."""
        self.logger.info(f"Detecting significant peaks using '{peak_detection_method}' method...")
        if peak_detection_method == "residual":
            if fap_method != "baluev" or fap_threshold != 0.01:
                self.logger.warning(
                    "'peak_detection_method' is 'residual', so 'fap_method' and "
                    "'fap_threshold' parameters are ignored."
                )
            peaks, threshold = find_peaks_via_residuals(
                fit_results, fdr_level=peak_fdr_level
            )
            fit_results["significant_peaks"] = peaks
            fit_results["residual_threshold"] = threshold
            fit_results["peak_fdr_level"] = peak_fdr_level
        elif peak_detection_method == "fap" and fap_threshold is not None:
            if peak_fdr_level != 0.05:
                self.logger.warning(
                    "'peak_detection_method' is 'fap', so the 'peak_fdr_level' "
                    "parameter is ignored."
                )
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
            output_path=plot_path,
            param_name=self.param_name,
        )

        # Summary Text
        summary_path = os.path.join(output_dir, f"{sanitized_name}_summary.txt")
        with open(summary_path, "w") as f:
            f.write(results["summary_text"])
        self.logger.info(f"Plot saved to {plot_path}")
        self.logger.info(f"Summary saved to {summary_path}")

    def _generate_changepoint_outputs(self, results, output_dir, plot_style="separate"):
        """
        Generates and saves the plot and summary text file for a changepoint analysis.
        """
        self.logger.info(f"Generating changepoint outputs in directory: {output_dir}")
        os.makedirs(output_dir, exist_ok=True)
        sanitized_name = self._sanitize_filename(self.param_name)

        # Generate the comparison plot with the specified style
        plot_changepoint_analysis(
            results,
            output_dir,
            self.param_name,
            plot_style=plot_style,
        )

        plot_filename = (
            f"{sanitized_name}_changepoint_{plot_style}.png"
        )
        plot_path = os.path.join(
            output_dir, plot_filename
        )

        # Save the combined summary text
        summary_path = os.path.join(
            output_dir, f"{sanitized_name}_changepoint_summary.txt"
        )
        with open(summary_path, "w") as f:
            f.write(results["summary_text"])

        self.logger.info(f"Changepoint comparison plot saved to {plot_path}")
        self.logger.info(f"Changepoint summary saved to {summary_path}")

    def _analyze_with_changepoint(
        self, changepoint_idx: int, output_dir: str, **analysis_kwargs
    ) -> Dict:
        """
        Performs separate spectral analysis on segments before/after changepoint.
        """
        # Split data into segments
        time_before = self.time[:changepoint_idx]
        data_before = self.data[:changepoint_idx]
        errors_before = (
            self.errors[:changepoint_idx] if self.errors is not None else None
        )

        time_after = self.time[changepoint_idx:]
        data_after = self.data[changepoint_idx:]
        errors_after = (
            self.errors[changepoint_idx:] if self.errors is not None else None
        )

        # --- Segment Validation ---
        # This is the critical validation step. It happens *after* splitting the
        # preprocessed data, ensuring the segments used for fitting are valid.
        n_before, n_after = len(data_before), len(data_after)

        if n_before < self.min_valid_data_points:
            raise ValueError(
                f"Segment before changepoint has insufficient valid data "
                f"({n_before} points) after preprocessing. Minimum "
                f"required: {self.min_valid_data_points}."
            )
        if n_after < self.min_valid_data_points:
            raise ValueError(
                f"Segment after changepoint has insufficient valid data "
                f"({n_after} points) after preprocessing. Minimum "
                f"required: {self.min_valid_data_points}."
            )

        # Warn if segments are highly imbalanced
        if min(n_before, n_after) > 0:  # Avoid division by zero
            ratio = max(n_before, n_after) / min(n_before, n_after)
            if ratio > 5:
                self.logger.warning(
                    f"Changepoint creates imbalanced segments (n_before={n_before}, "
                    f"n_after={n_after}, ratio {ratio:.1f}:1). "
                    "Results may be less reliable for the smaller segment."
                )

        cp_time_str = get_changepoint_time(
            changepoint_idx, self.time, self.time_unit
        )

        # Create separate kwargs for each segment to handle seeding correctly.
        analysis_kwargs_before = analysis_kwargs.copy()
        analysis_kwargs_after = analysis_kwargs.copy()

        # If a seed is provided, spawn two independent child seeds for the two
        # segments to ensure that their bootstrap analyses are independent.
        if analysis_kwargs.get("seed") is not None:
            # If the seed is an integer, create a SeedSequence.
            seed_val = analysis_kwargs["seed"]
            ss = np.random.SeedSequence(seed_val) if isinstance(seed_val, int) else seed_val
            child_seeds = ss.spawn(2)
            analysis_kwargs_before["seed"] = child_seeds[0]
            analysis_kwargs_after["seed"] = child_seeds[1]

        # Analyze each segment
        self.logger.info(
            f"Analyzing segment BEFORE changepoint (n={len(time_before)})..."
        )
        results_before = self._run_segment_analysis(
            time_before,
            data_before,
            errors_before,
            segment_name=f"Before {cp_time_str}",
            **analysis_kwargs_before,
        )

        self.logger.info(
            f"Analyzing segment AFTER changepoint (n={len(time_after)})..."
        )
        results_after = self._run_segment_analysis(
            time_after,
            data_after,
            errors_after,
            segment_name=f"After {cp_time_str}",
            **analysis_kwargs_after,
        )

        # Combine results
        combined_results = {
            "changepoint_analysis": True,
            "changepoint_index": changepoint_idx,
            "changepoint_time": cp_time_str,
            "changepoint_mode": self.changepoint_mode,
            "segment_before": results_before,
            "segment_after": results_after,
            "preprocessing_diagnostics": self.preprocessing_diagnostics,
        }

        # Generate combined summary
        combined_results["summary_text"] = self._generate_changepoint_summary(
            combined_results
        )

        # Generate and save outputs
        self._generate_changepoint_outputs(
            combined_results,
            output_dir,
            plot_style=analysis_kwargs.get("changepoint_plot_style", "separate"),
        )

        return combined_results

    def _run_segment_analysis(
        self, time, data, errors, segment_name, **analysis_kwargs
    ) -> Dict:
        """Runs full spectral analysis on a single segment."""
        # Store current data temporarily
        orig_time, orig_data, orig_errors = self.time, self.data, self.errors
        orig_freq, orig_power, orig_ls = (
            self.frequency,
            self.power,
            self.ls_obj,
        )

        # Swap in segment data
        self.time, self.data, self.errors = time, data, errors

        # Calculate periodogram
        self._calculate_periodogram(
            analysis_kwargs.get("normalization", "standard"),
            analysis_kwargs.get("nyquist_factor", 1.0),
            analysis_kwargs.get("max_freq"),
            analysis_kwargs.get("samples_per_peak", 5),
        )

        # Fit models
        fit_results = self._perform_model_selection(
            fit_method=analysis_kwargs.get("fit_method", "theil-sen"),
            ci_method=analysis_kwargs.get("ci_method", "bootstrap"),
            bootstrap_type=analysis_kwargs.get("bootstrap_type", "block"),
            n_bootstraps=analysis_kwargs.get("n_bootstraps", 1000),
            p_threshold=analysis_kwargs.get("p_threshold", 0.05),
            max_breakpoints=analysis_kwargs.get("max_breakpoints", 1),
            seed=analysis_kwargs.get("seed"),
        )

        # Detect peaks
        fit_results = self._detect_significant_peaks(
            fit_results,
            analysis_kwargs.get("peak_detection_method", "fap"),
            analysis_kwargs.get("peak_fdr_level", 0.05),
            analysis_kwargs.get("fap_threshold", 0.01),
            analysis_kwargs.get("fap_method", "baluev"),
        )

        # Interpret
        interp_results = interpret_results(
            fit_results,
            param_name=f"{self.param_name} ({segment_name})",
            time_unit=self.time_unit,
        )

        segment_results = {
            **fit_results,
            **interp_results,
            "segment_name": segment_name,
            "n_points": len(time),
            "time_range": (time[0], time[-1]),
            "frequency": self.frequency,
            "power": self.power,
        }

        # Restore original data
        self.time, self.data, self.errors = orig_time, orig_data, orig_errors
        self.frequency, self.power, self.ls_obj = (
            orig_freq,
            orig_power,
            orig_ls,
        )

        return segment_results

    def _generate_changepoint_summary(self, results: Dict) -> str:
        """Generates a comparison summary for a changepoint analysis."""
        from .interpreter import get_persistence_traffic_light

        cp_time = results["changepoint_time"]
        mode = results["changepoint_mode"]
        before = results["segment_before"]
        after = results["segment_after"]

        header = f"Changepoint Analysis ({mode.capitalize()} Detection)\n"
        header += "=" * 60 + "\n\n"
        header += f"Changepoint Located: {cp_time}\n"
        header += f"  Segment BEFORE: n={before['n_points']}\n"
        header += f"  Segment AFTER:  n={after['n_points']}\n"
        header += "\n" + "-" * 60 + "\n\n"

        # Extract primary beta from each segment
        def get_beta(seg):
            if "betas" in seg:  # Segmented model
                return seg["betas"][0]
            return seg.get("beta", np.nan)  # Standard model

        beta_before = get_beta(before)
        beta_after = get_beta(after)

        # Build comparison section
        comparison = "REGIME COMPARISON:\n"
        comparison += (
            f"  Before: β ≈ {beta_before:.2f} ({get_persistence_traffic_light(beta_before)})\n"
        )
        comparison += (
            f"  After:  β ≈ {beta_after:.2f} ({get_persistence_traffic_light(beta_after)})\n\n"
        )

        delta_beta = beta_after - beta_before
        if abs(delta_beta) > 0.3:  # Threshold for significant change
            if delta_beta > 0:
                comparison += f"  * Significant INCREASE in persistence (+{delta_beta:.2f})\n"
                comparison += (
                    "    System may have shifted toward more storage-dominated pathways.\n"
                )
            else:
                comparison += f"  * Significant DECREASE in persistence ({delta_beta:.2f})\n"
                comparison += (
                    "    System may have shifted toward more event-driven pathways.\n"
                )
        else:
            comparison += "  * No substantial change in persistence regime detected.\n"

        comparison += "\n" + "=" * 60 + "\n\n"

        # Combine with individual segment summaries
        summary_before = before["summary_text"]
        summary_after = after["summary_text"]

        full_summary = header + comparison + summary_before + "\n\n" + "=" * 60 + "\n\n" + summary_after
        return full_summary

    def _validate_run_parameters(
        self,
        fit_method,
        ci_method,
        bootstrap_type,
        n_bootstraps,
        fap_threshold,
        samples_per_peak,
        fap_method,
        normalization,
        peak_detection_method,
        peak_fdr_level,
        p_threshold,
        max_breakpoints,
        nyquist_factor,
        max_freq,
    ):
        """Validates parameters for the `run_full_analysis` method."""
        if fit_method not in ["theil-sen", "ols"]:
            raise ValueError("`fit_method` must be 'theil-sen' or 'ols'.")
        if ci_method not in ["bootstrap", "parametric"]:
            raise ValueError("`ci_method` must be 'bootstrap' or 'parametric'.")
        if bootstrap_type not in ["pairs", "residuals", "block", "wild"]:
            raise ValueError(
                "`bootstrap_type` must be 'pairs', 'residuals', 'block', or 'wild'."
            )
        if not isinstance(n_bootstraps, int) or n_bootstraps < 0:
            raise ValueError("`n_bootstraps` must be a non-negative integer.")
        if not (isinstance(fap_threshold, float) and 0 < fap_threshold < 1):
            raise ValueError("`fap_threshold` must be a float between 0 and 1.")
        if not isinstance(samples_per_peak, int) or samples_per_peak <= 0:
            raise ValueError("`samples_per_peak` must be a positive integer.")
        if fap_method not in ["baluev", "bootstrap"]:
            raise ValueError("`fap_method` must be 'baluev' or 'bootstrap'.")
        if normalization not in ["standard", "model", "log", "psd"]:
            raise ValueError(
                "`normalization` must be one of 'standard', 'model', 'log', or 'psd'."
            )
        if peak_detection_method not in ["residual", "fap", None]:
            raise ValueError(
                "`peak_detection_method` must be 'residual', 'fap', or None."
            )
        if not (isinstance(peak_fdr_level, float) and 0 < peak_fdr_level < 1):
            raise ValueError("`peak_fdr_level` must be a float between 0 and 1.")
        if not (isinstance(p_threshold, float) and 0 < p_threshold < 1):
            raise ValueError("`p_threshold` must be a float between 0 and 1.")
        if max_breakpoints not in [0, 1, 2]:
            raise ValueError("`max_breakpoints` must be an integer (0, 1, or 2).")
        if not isinstance(nyquist_factor, (int, float)) or nyquist_factor <= 0:
            raise ValueError("`nyquist_factor` must be a positive number.")
        if max_freq is not None and (not isinstance(max_freq, (int, float)) or max_freq <= 0):
            raise ValueError("`max_freq`, if provided, must be a positive number.")

    def run_full_analysis(
        self,
        output_dir,
        fit_method="theil-sen",
        ci_method="bootstrap",
        bootstrap_type="block",
        n_bootstraps=500,
        samples_per_peak=5,
        nyquist_factor=1.0,
        max_freq=None,
        peak_detection_method="fap",
        fap_threshold=0.01,
        fap_method="baluev",
        peak_fdr_level=0.05,
        normalization="standard",
        p_threshold=0.05,
        max_breakpoints=1,
        seed=None,
        changepoint_plot_style="separate",
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
            bootstrap_type (str, optional): The bootstrap method to use if
                `ci_method` is 'bootstrap'. Can be 'block' (default), 'wild',
                'residuals', or 'pairs'. The 'block' method is recommended for
                spectral analysis as it preserves autocorrelation. 'wild' is
                useful for heteroscedastic data. 'residuals' and 'pairs' are
                generally not recommended for spectral data.
            n_bootstraps (int, optional): Number of bootstrap samples for CI.
                Only used if `ci_method` is 'bootstrap'. Defaults to 500.
            samples_per_peak (int, optional): The number of samples to generate
                per periodogram peak, controlling the density of the frequency
                grid generated by `autopower`. Defaults to 5.
            nyquist_factor (float, optional): Scaling factor for the heuristic
                Nyquist frequency (0.5 / median interval). Ignored if `max_freq`
                is set. Defaults to 1.0.
            max_freq (float, optional): User-defined maximum frequency for the
                grid, overriding automatic calculation. The unit should be the
                inverse of `time_unit`. Defaults to None.
            peak_detection_method (str, optional): Method for peak detection.
                The available options are:

                - `'fap'` (default): Uses the traditional False Alarm Probability
                  (FAP) method, which is robust and does not depend on the
                  goodness-of-fit of the background spectral model.
                - `'residual'`: Identifies peaks that are significant outliers
                  from the fitted spectral model using a False Discovery Rate
                  (FDR) approach. This method's accuracy depends on a correct
                  background model.

                When using `'fap'`, `peak_fdr_level` is ignored. When using
                `'residual'`, `fap_method` and `fap_threshold` are ignored.
            peak_fdr_level (float, optional): The false discovery rate level
                to use for the residual-based peak detection method.
                Defaults to 0.05.
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
        # 1. Validate all run parameters
        self._validate_run_parameters(
            fit_method,
            ci_method,
            bootstrap_type,
            n_bootstraps,
            fap_threshold,
            samples_per_peak,
            fap_method,
            normalization,
            peak_detection_method,
            peak_fdr_level,
            p_threshold,
            max_breakpoints,
            nyquist_factor,
            max_freq,
        )
        analysis_kwargs = {
            "fit_method": fit_method,
            "ci_method": ci_method,
            "bootstrap_type": bootstrap_type,
            "n_bootstraps": n_bootstraps,
            "samples_per_peak": samples_per_peak,
            "nyquist_factor": nyquist_factor,
            "max_freq": max_freq,
            "peak_detection_method": peak_detection_method,
            "fap_threshold": fap_threshold,
            "fap_method": fap_method,
            "peak_fdr_level": peak_fdr_level,
            "normalization": normalization,
            "p_threshold": p_threshold,
            "max_breakpoints": max_breakpoints,
            "seed": seed,
            "changepoint_plot_style": changepoint_plot_style,
        }

        # Determine changepoint
        changepoint_idx = None
        if self.changepoint_mode == "auto":
            changepoint_idx = self._detect_changepoint()
        elif self.changepoint_mode == "manual":
            changepoint_idx = self.changepoint_index

        # Branch based on whether we have a changepoint
        if changepoint_idx is not None:
            self.logger.info("Performing changepoint-segmented analysis...")
            self.results = self._analyze_with_changepoint(
                changepoint_idx, output_dir, **analysis_kwargs
            )
            return self.results

        # Standard analysis (existing code)
        self.logger.info("Performing standard (non-segmented) analysis...")
        # 2. Calculate Periodogram
        self._calculate_periodogram(
            normalization, nyquist_factor, max_freq, samples_per_peak
        )

        # 3. Fit Spectrum and Select Best Model
        try:
            fit_results = self._perform_model_selection(
                fit_method,
                ci_method,
                bootstrap_type,
                n_bootstraps,
                p_threshold,
                max_breakpoints,
                seed,
            )
        except RuntimeError as e:
            self.logger.error(f"Analysis failed during model fitting: {e}")
            self.results = {
                "betas": [np.nan],
                "n_breakpoints": 0,
                "chosen_model_type": "failure",
                "summary_text": f"Analysis failed: {e}",
                "failure_reason": str(e),
                "preprocessing_diagnostics": self.preprocessing_diagnostics,
            }
            self._generate_outputs(self.results, output_dir)
            return self.results

        # 4. Detect Significant Peaks
        fit_results = self._detect_significant_peaks(
            fit_results,
            peak_detection_method,
            peak_fdr_level,
            fap_threshold,
            fap_method,
        )

        # 5. Interpret Results
        self.logger.info("Interpreting final results and generating summary...")
        interp_results = interpret_results(
            fit_results, param_name=self.param_name, time_unit=self.time_unit
        )
        self.results = {**fit_results, **interp_results}

        # Add preprocessing diagnostics to the final results
        self.results["preprocessing_diagnostics"] = self.preprocessing_diagnostics

        # 6. Generate and Save Outputs
        self._generate_outputs(self.results, output_dir)

        self.logger.info(f"Analysis complete. Outputs saved to '{output_dir}'.")
        return self.results

import logging
import os
import re
from typing import Dict

import numpy as np

from .data_loader import load_data
from .fitter import fit_segmented_spectrum, fit_standard_model
from .interpreter import get_persistence_traffic_light, interpret_results
from .plotting import plot_site_comparison
from .preprocessor import preprocess_data
from .spectral_analyzer import (
    calculate_periodogram,
    find_peaks_via_residuals,
    find_significant_peaks,
)


class SiteComparison:
    """
    A class to compare the spectral analysis of two different time series (sites).

    This class encapsulates the data loading, preprocessing, analysis,
    and output generation for comparing two distinct datasets.
    """

    def __init__(
        self,
        site1_config: Dict,
        site2_config: Dict,
        min_valid_data_points: int = 10,
    ):
        """
        Initializes the SiteComparison object by loading and preprocessing data for two sites.

        Args:
            site1_config (Dict): A dictionary with data loading and preprocessing
                parameters for the first site. It should include a unique 'name'
                and keys like 'file_path', 'time_col', 'data_col'.
            site2_config (Dict): A dictionary of parameters for the second site.
            min_valid_data_points (int, optional): The minimum number of valid
                data points required to proceed with an analysis. Defaults to 10.
            verbose (bool, optional): If True, sets logging level to INFO.
                Defaults to False (logging level WARNING).
        """
        self.site1_name = site1_config.get("name", "Site 1")
        self.site2_name = site2_config.get("name", "Site 2")
        self.logger = logging.getLogger(__name__)

        if not isinstance(min_valid_data_points, int) or min_valid_data_points <= 0:
            raise ValueError("`min_valid_data_points` must be a positive integer.")
        self.min_valid_data_points = min_valid_data_points

        # Load and preprocess data for Site 1
        self.logger.info(f"Loading and preprocessing data for {self.site1_name}...")
        self.site1_data = self._load_and_process_site(site1_config, self.site1_name)

        # Load and preprocess data for Site 2
        self.logger.info(f"Loading and preprocessing data for {self.site2_name}...")
        self.site2_data = self._load_and_process_site(site2_config, self.site2_name)

        self.results = None

    def _load_and_process_site(self, config: Dict, site_name: str) -> Dict:
        """Loads and preprocesses data for a single site."""
        time_numeric, data_series, error_series = load_data(
            file_path=config["file_path"],
            time_col=config["time_col"],
            data_col=config["data_col"],
            error_col=config.get("error_col"),
            time_format=config.get("time_format"),
            sheet_name=config.get("sheet_name", 0),
            output_time_unit=config.get("time_unit", "seconds"),
        )

        processed_data, processed_errors, diagnostics = preprocess_data(
            data_series,
            time_numeric,
            error_series=error_series,
            censor_strategy=config.get("censor_strategy", "drop"),
            censor_options=config.get("censor_options"),
            log_transform_data=config.get("log_transform_data", False),
            detrend_method=config.get("detrend_method"),
            normalize_data=config.get("normalize_data", False),
            detrend_options=config.get("detrend_options"),
        )

        valid_indices = ~np.isnan(processed_data)
        if np.sum(valid_indices) < self.min_valid_data_points:
            raise ValueError(
                f"Not enough valid data points ({np.sum(valid_indices)}) for site "
                f"'{site_name}' after preprocessing. Minimum required: "
                f"{self.min_valid_data_points}."
            )

        return {
            "time": time_numeric[valid_indices],
            "data": processed_data[valid_indices],
            "errors":
                processed_errors[valid_indices] if processed_errors is not None else None,
            "diagnostics": diagnostics,
            "param_name": config.get("param_name", config["data_col"]),
            "time_unit": config.get("time_unit", "seconds"),
        }

    def _sanitize_filename(self, filename):
        """Sanitizes a string to be a valid filename."""
        s = str(filename).strip().replace(" ", "_")
        s = re.sub(r"(?u)[^-\w.]", "", s)
        return s

    def run_comparison(
        self,
        output_dir,
        fit_method="theil-sen",
        ci_method="bootstrap",
        bootstrap_type="block",
        n_bootstraps=200,
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
        comparison_plot_style="separate",
    ):
        """
        Runs the complete comparison analysis and saves all outputs to a directory.
        """
        # Validate run parameters
        self._validate_run_parameters(
            fit_method, ci_method, bootstrap_type, n_bootstraps, fap_threshold,
            samples_per_peak, fap_method, normalization, peak_detection_method,
            peak_fdr_level, p_threshold, max_breakpoints, nyquist_factor, max_freq
        )

        # Explicitly create a dictionary of analysis parameters
        analysis_kwargs = {
            "fit_method": fit_method, "ci_method": ci_method,
            "bootstrap_type": bootstrap_type, "n_bootstraps": n_bootstraps,
            "samples_per_peak": samples_per_peak, "nyquist_factor": nyquist_factor,
            "max_freq": max_freq, "peak_detection_method": peak_detection_method,
            "fap_threshold": fap_threshold, "fap_method": fap_method,
            "peak_fdr_level": peak_fdr_level, "normalization": normalization,
            "p_threshold": p_threshold, "max_breakpoints": max_breakpoints,
            "seed": seed,
        }

        # Analyze each site
        self.logger.info(f"Analyzing site: {self.site1_name}...")
        results_site1 = self._run_site_analysis(
            self.site1_data, self.site1_name, analysis_kwargs
        )

        self.logger.info(f"Analyzing site: {self.site2_name}...")
        # Use a different seed for the second site if a seed is provided
        site2_seed = (seed + 1) if seed is not None else None
        results_site2 = self._run_site_analysis(
            self.site2_data,
            self.site2_name,
            {**analysis_kwargs, "seed": site2_seed},
        )

        # Combine results
        self.results = {
            "site1": results_site1,
            "site2": results_site2,
            "comparison_name": f"{self.site1_name}_vs_{self.site2_name}",
        }

        # Generate summary and outputs
        self.results["summary_text"] = self._generate_comparison_summary(self.results)
        self._generate_comparison_outputs(
            self.results, output_dir, plot_style=comparison_plot_style
        )

        self.logger.info(f"Comparison analysis complete. Outputs will be saved to '{output_dir}'.")
        return self.results

    def _run_site_analysis(
        self, site_data: Dict, site_name: str, analysis_kwargs: Dict
    ) -> Dict:
        """Runs full spectral analysis on a single site's data."""
        time = site_data["time"]
        data = site_data["data"]
        errors = site_data["errors"]
        param_name = site_data["param_name"]
        time_unit = site_data["time_unit"]

        # 1. Calculate Periodogram
        frequency, power, ls_obj = calculate_periodogram(
            time, data, dy=errors,
            normalization=analysis_kwargs["normalization"],
            nyquist_factor=analysis_kwargs["nyquist_factor"],
            samples_per_peak=analysis_kwargs["samples_per_peak"],
            maximum_frequency=analysis_kwargs["max_freq"],
        )

        # 2. Fit Spectrum and Select Best Model
        fit_results = self._perform_model_selection(
            frequency, power,
            fit_method=analysis_kwargs["fit_method"],
            ci_method=analysis_kwargs["ci_method"],
            bootstrap_type=analysis_kwargs["bootstrap_type"],
            n_bootstraps=analysis_kwargs["n_bootstraps"],
            p_threshold=analysis_kwargs["p_threshold"],
            max_breakpoints=analysis_kwargs["max_breakpoints"],
            seed=analysis_kwargs["seed"],
        )

        # 3. Detect Significant Peaks
        fit_results = self._detect_significant_peaks(
            fit_results, ls_obj, frequency, power,
            peak_detection_method=analysis_kwargs["peak_detection_method"],
            peak_fdr_level=analysis_kwargs["peak_fdr_level"],
            fap_threshold=analysis_kwargs["fap_threshold"],
            fap_method=analysis_kwargs["fap_method"],
        )

        # 4. Interpret Results
        interp_results = interpret_results(
            fit_results, param_name=param_name, time_unit=time_unit
        )

        return {
            **fit_results, **interp_results,
            "site_name": site_name,
            "n_points": len(time),
            "time_range": (time[0], time[-1]),
            "frequency": frequency,
            "power": power,
            "preprocessing_diagnostics": site_data["diagnostics"],
        }

    def _generate_comparison_summary(self, results: Dict) -> str:
        """Generates a comparison summary for a two-site analysis."""
        site1 = results["site1"]
        site2 = results["site2"]
        site1_name = self.site1_name
        site2_name = self.site2_name

        header = f"Site Comparison: {site1_name} vs. {site2_name}\n"
        header += "=" * 60 + "\n\n"
        header += f"  Site 1 ({site1_name}): n={site1['n_points']}\n"
        header += f"  Site 2 ({site2_name}): n={site2['n_points']}\n"
        header += "\n" + "-" * 60 + "\n\n"

        # Extract primary beta from each site
        def get_beta(res):
            if "betas" in res:  # Segmented model
                return res["betas"][0]
            return res.get("beta", np.nan)  # Standard model

        beta1 = get_beta(site1)
        beta2 = get_beta(site2)

        # Build comparison section
        comparison = "SITE COMPARISON:\n"
        comparison += (
            f"  {site1_name}: β ≈ {beta1:.2f} ({get_persistence_traffic_light(beta1)})\n"
        )
        comparison += (
            f"  {site2_name}: β ≈ {beta2:.2f} ({get_persistence_traffic_light(beta2)})\n\n"
        )

        delta_beta = beta2 - beta1
        if abs(delta_beta) > 0.3:  # Threshold for significant change
            if delta_beta > 0:
                comparison += f"  * {site2_name} shows significantly HIGHER persistence than {site1_name} (+ {delta_beta:.2f})\n"
            else:
                comparison += f"  * {site2_name} shows significantly LOWER persistence than {site1_name} ({delta_beta:.2f})\n"
        else:
            comparison += "  * No substantial difference in persistence regime detected between sites.\n"

        comparison += "\n" + "=" * 60 + "\n\n"

        # Combine with individual site summaries
        summary1 = site1["summary_text"]
        summary2 = site2["summary_text"]

        full_summary = header + comparison + summary1 + "\n\n" + "=" * 60 + "\n\n" + summary2
        return full_summary

    def _generate_comparison_outputs(self, results, output_dir, plot_style="separate"):
        """
        Generates and saves the plot and summary text file for a comparison analysis.
        """
        self.logger.info(f"Generating comparison outputs in directory: {output_dir}")
        os.makedirs(output_dir, exist_ok=True)
        sanitized_name = self._sanitize_filename(results["comparison_name"])

        # Generate the comparison plot
        plot_site_comparison(
            results,
            output_dir,
            plot_style=plot_style,
        )

        # Save the combined summary text
        summary_path = os.path.join(
            output_dir, f"{sanitized_name}_comparison_summary.txt"
        )
        with open(summary_path, "w") as f:
            f.write(results["summary_text"])

        self.logger.info(f"Comparison summary saved to {summary_path}")

    def _perform_model_selection(
        self, frequency, power, fit_method, ci_method, bootstrap_type,
        n_bootstraps, p_threshold, max_breakpoints, seed
    ):
        """Performs fits and selects the best model using BIC."""
        all_models = []

        # Standard model
        try:
            standard_results = fit_standard_model(
                frequency, power,
                method=fit_method,
                ci_method=ci_method,
                bootstrap_type=bootstrap_type,
                n_bootstraps=n_bootstraps,
                seed=seed,
                logger=self.logger,
            )
            if "bic" in standard_results and np.isfinite(standard_results["bic"]):
                standard_results["n_breakpoints"] = 0
                all_models.append(standard_results)
        except Exception as e:
            self.logger.error("Standard model fit crashed: %s", e, exc_info=True)

        # Segmented models
        for n_bp in range(1, max_breakpoints + 1):
            try:
                model_seed = seed + n_bp if seed is not None else None
                seg_results = fit_segmented_spectrum(
                    frequency, power, n_breakpoints=n_bp,
                    p_threshold=p_threshold,
                    ci_method=ci_method,
                    bootstrap_type=bootstrap_type,
                    n_bootstraps=n_bootstraps,
                    seed=model_seed,
                    logger=self.logger,
                )
                if "bic" in seg_results and np.isfinite(seg_results["bic"]):
                    all_models.append(seg_results)
            except Exception as e:
                self.logger.error(
                    "Segmented model (%d bp) fit crashed: %s", n_bp, e, exc_info=True
                )

        if not all_models:
            raise RuntimeError("Model fitting failed for all attempted models.")

        best_model = min(all_models, key=lambda x: x["bic"])

        # Add metadata for interpretation and plotting
        if best_model.get("n_breakpoints", 0) == 0:
            best_model["chosen_model_type"] = "standard"
        else:
            best_model["chosen_model_type"] = "segmented"

        return best_model

    def _detect_significant_peaks(
        self, fit_results, ls_obj, frequency, power, peak_detection_method,
        peak_fdr_level, fap_threshold, fap_method
    ):
        """Detects significant peaks."""
        if peak_detection_method == "residual":
            peaks, threshold = find_peaks_via_residuals(
                fit_results, fdr_level=peak_fdr_level
            )
            fit_results["significant_peaks"] = peaks
            fit_results["residual_threshold"] = threshold
            fit_results["peak_fdr_level"] = peak_fdr_level
        elif peak_detection_method == "fap":
            peaks, level = find_significant_peaks(
                ls_obj, frequency, power,
                fap_threshold=fap_threshold,
                fap_method=fap_method,
            )
            fit_results["significant_peaks"] = peaks
            fit_results["fap_level"] = level
            fit_results["fap_threshold"] = fap_threshold
        return fit_results

    def _validate_run_parameters(
        self, fit_method, ci_method, bootstrap_type, n_bootstraps,
        fap_threshold, samples_per_peak, fap_method, normalization,
        peak_detection_method, peak_fdr_level, p_threshold,
        max_breakpoints, nyquist_factor, max_freq
    ):
        """Validates parameters for the `run_comparison` method."""
        # This is a direct copy from the Analysis class for simplicity
        if fit_method not in ["theil-sen", "ols"]:
            raise ValueError("`fit_method` must be 'theil-sen' or 'ols'.")
        if ci_method not in ["bootstrap", "parametric"]:
            raise ValueError("`ci_method` must be 'bootstrap' or 'parametric'.")
        if bootstrap_type not in ["pairs", "residuals", "block", "wild"]:
            raise ValueError("`bootstrap_type` must be 'pairs', 'residuals', 'block', or 'wild'.")
        if not isinstance(n_bootstraps, int) or n_bootstraps < 0:
            raise ValueError("`n_bootstraps` must be a non-negative integer.")
        if not (isinstance(fap_threshold, float) and 0 < fap_threshold < 1):
            raise ValueError("`fap_threshold` must be a float between 0 and 1.")
        if not isinstance(samples_per_peak, int) or samples_per_peak <= 0:
            raise ValueError("`samples_per_peak` must be a positive integer.")
        if fap_method not in ["baluev", "bootstrap"]:
            raise ValueError("`fap_method` must be 'baluev' or 'bootstrap'.")
        if normalization not in ["standard", "model", "log", "psd"]:
            raise ValueError("`normalization` must be one of 'standard', 'model', 'log', or 'psd'.")
        if peak_detection_method not in ["residual", "fap", None]:
            raise ValueError("`peak_detection_method` must be 'residual', 'fap', or None.")
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
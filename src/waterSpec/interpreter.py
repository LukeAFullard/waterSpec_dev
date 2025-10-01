import numpy as np
import pandas as pd


def _get_uncertainty_warning(ci, threshold, name="value"):
    """Generates a warning if a confidence interval is wider than a threshold."""
    if ci is None or not np.all(np.isfinite(ci)):
        return None
    width = ci[1] - ci[0]
    if width > threshold:
        return (
            f"Warning: The 95% CI for {name} is wide ({width:.2f} > {threshold}), "
            "suggesting high uncertainty."
        )
    return None


# Benchmark table based on Liang et al. (2021) and general hydrological knowledge.
BENCHMARK_TABLE = pd.DataFrame(
    [
        {
            "Parameter": "E. coli",
            "Typical β Range": (0.1, 0.5),
            "Interpretation": "Weak persistence",
            "Dominant Pathway": "Surface runoff",
        },
        {
            "Parameter": "Total Suspended Solids (TSS)",
            "Typical β Range": (0.4, 0.8),
            "Interpretation": "Weak persistence",
            "Dominant Pathway": "Surface runoff",
        },
        {
            "Parameter": "Ortho-P",
            "Typical β Range": (0.6, 1.2),
            "Interpretation": "Mixed",
            "Dominant Pathway": "Surface/Shallow subsurface",
        },
        {
            "Parameter": "Chloride",
            "Typical β Range": (1.3, 1.7),
            "Interpretation": "Strong persistence",
            "Dominant Pathway": "Subsurface",
        },
        {
            "Parameter": "Nitrate-N",
            "Typical β Range": (1.5, 2.0),
            "Interpretation": "Strong persistence",
            "Dominant Pathway": "Subsurface",
        },
        {
            "Parameter": "Discharge (Q)",
            "Typical β Range": (1.0, 1.8),
            "Interpretation": "Persistent",
            "Dominant Pathway": "Integrated signal",
        },
    ]
).set_index("Parameter")


# Define constants for scientific interpretation for clarity and maintainability.
BETA_TOLERANCE = 0.2  # Tolerance for comparing beta to common noise models
BETA_UPPER_BOUND = 3.0  # Physical upper bound for beta

# Define thresholds for persistence categories
PERSISTENCE_EVENT_DRIVEN_THRESHOLD = 0.5
PERSISTENCE_MIXED_THRESHOLD = 1.0


# A constant to define the threshold for confidence interval width. If the
# width of a 95% CI for a parameter (like beta) exceeds this value, a
# warning is generated, as it suggests high uncertainty in the estimate.
CI_WIDTH_THRESHOLD_FOR_WARNING = 0.5


def get_scientific_interpretation(beta):
    """Provides a scientific interpretation of the spectral exponent (beta)."""
    if np.isclose(beta, 0, atol=BETA_TOLERANCE):
        return "β ≈ 0 (White Noise): Uncorrelated, random process."
    elif np.isclose(beta, 1, atol=BETA_TOLERANCE):
        return "β ≈ 1 (Pink Noise): Stronger persistence, common in natural systems."
    elif np.isclose(beta, 2, atol=BETA_TOLERANCE):
        return "β ≈ 2 (Brownian Noise): Random walk process."
    elif beta < 0:
        return (
            "β < 0 (Warning: Physically Unrealistic): This may indicate aliasing, "
            "inappropriate detrending, or white noise dominance. Review your "
            "preprocessing choices."
        )
    elif 0 < beta < 1:
        return (
            f"0 < β < 1 (fGn-like): Weakly persistent, suggesting event-driven transport."
        )
    elif 1 < beta < BETA_UPPER_BOUND:
        return (
            f"1 < β < {BETA_UPPER_BOUND} (fBm-like): Strong persistence, suggesting transport is "
            "damped by storage."
        )
    elif beta >= BETA_UPPER_BOUND:
        return f"β ≥ {BETA_UPPER_BOUND}: Black noise, very smooth signal, may indicate non-stationarity."
    return "No interpretation available."


def get_persistence_traffic_light(beta):
    """Returns a traffic-light style summary of persistence."""
    if beta < PERSISTENCE_EVENT_DRIVEN_THRESHOLD:
        return "Low (Event-driven)"
    if PERSISTENCE_EVENT_DRIVEN_THRESHOLD <= beta <= PERSISTENCE_MIXED_THRESHOLD:
        return "Medium (Mixed)"
    if beta > PERSISTENCE_MIXED_THRESHOLD:
        return "High (Storage-dominated)"
    return "Unknown"


# Define time constants for period formatting for clarity and maintainability.
SECONDS_PER_DAY = 86400
DAYS_PER_MONTH = 30.44  # Average days in a month
DAYS_PER_YEAR = 365.25  # Accounts for leap years


def _format_period(frequency_hz):
    """
    Converts a frequency in Hz to a human-readable period string.

    Note: This function assumes the input frequency is in units of 1/seconds (Hz),
    which is consistent with the `time_numeric_sec` output from `load_data`.
    """
    if frequency_hz <= 0:
        return "N/A"

    period_seconds = 1 / frequency_hz
    period_days = period_seconds / SECONDS_PER_DAY

    # Switch to years if > 1.5 years, and months if > 1.5 months for intuitive formatting.
    if period_days >= DAYS_PER_YEAR * 1.5:
        return f"{period_days / DAYS_PER_YEAR:.1f} years"
    elif period_days >= DAYS_PER_MONTH * 1.5:
        return f"{period_days / DAYS_PER_MONTH:.1f} months"
    else:
        return f"{period_days:.1f} days"


def compare_to_benchmarks(beta):
    """Compares a beta value to the benchmark table."""
    closest_match = None
    min_distance = float("inf")

    for param, row in BENCHMARK_TABLE.iterrows():
        lower, upper = row["Typical β Range"]
        if lower <= beta <= upper:
            return f"Similar to {param} ({row['Dominant Pathway']}-dominated)."

        mean_beta = (lower + upper) / 2
        distance = abs(beta - mean_beta)
        if distance < min_distance:
            min_distance = distance
            closest_match = f"Closest to {param} ({row['Dominant Pathway']}-dominated)."

    return closest_match


def _generate_segment_interpretation(
    title, beta, beta_ci, beta_name, ci_method_str, uncertainty_threshold
):
    """Generates the interpretation text for a single spectral segment."""
    beta_str = f"{beta_name} = {beta:.2f}"
    if np.all(np.isfinite(beta_ci)):
        beta_str += f" (95% CI: {beta_ci[0]:.2f}–{beta_ci[1]:.2f}{ci_method_str})"

    warning = _get_uncertainty_warning(beta_ci, uncertainty_threshold, name=beta_name)

    summary_lines = [
        f"{title}:",
        f"  {beta_str}",
        f"  Interpretation: {get_scientific_interpretation(beta)}",
        f"  Persistence: {get_persistence_traffic_light(beta)}",
    ]
    return summary_lines, warning


def interpret_results(
    fit_results,
    param_name="Parameter",
    uncertainty_threshold=CI_WIDTH_THRESHOLD_FOR_WARNING,
    breakpoint_uncertainty_threshold=10,
):
    """
    Generates a comprehensive, human-readable interpretation of the analysis results.

    This function is designed to be called from the main Analysis class and
    returns only the interpretation dictionary, not the full results set.
    """
    auto_summary_header = ""
    uncertainty_warnings = []
    summary_text = ""

    # If this was an auto-analysis, generate a special header
    if fit_results.get("analysis_mode") == "auto":
        model_summaries = []
        for model in fit_results["all_models"]:
            n_breakpoints = model["n_breakpoints"]
            bic_str = f"{model['bic']:.2f}"
            if n_breakpoints == 0:
                name = "Standard"
                beta_str = f"β = {model['beta']:.2f}"
            else:
                name = f"Segmented ({n_breakpoints} BP)"
                betas = ", ".join(
                    [f"β{i+1}={b:.2f}" for i, b in enumerate(model["betas"])]
                )
                beta_str = f"{betas}"
            model_summaries.append(f"  - {name:<15} BIC = {bic_str:<8} ({beta_str})")

        # Add reasons for failed models, if any, for diagnostic purposes
        failed_reasons = fit_results.get("failed_model_reasons", [])
        if failed_reasons:
            model_summaries.append("\n  Models that failed to fit:")
            for reason in failed_reasons:
                model_summaries.append(f"    - {reason}")

        chosen_model_name = fit_results["chosen_model"].replace("_", " ").capitalize()
        auto_summary_header = (
            f"Automatic Analysis for: {param_name}\n"
            "-----------------------------------\n"
            "Model Comparison (Lower BIC is better):\n"
            + "\n".join(model_summaries)
            + f"\n\n==> Chosen Model: {chosen_model_name}\n"
            "-----------------------------------\n\n"
            f"Details for Chosen ({chosen_model_name}) Model:\n"
        )

    # --- Generate the main summary for the chosen model ---
    n_breakpoints = fit_results.get("n_breakpoints", 0)
    ci_method = fit_results.get("ci_method", "bootstrap")
    ci_method_str = f" ({ci_method})"

    if n_breakpoints > 0:
        # --- Segmented Model Summary ---
        summary_parts = [f"Segmented Analysis for: {param_name}"]

        # --- Handle the first segment (Low-Frequency) ---
        beta1 = fit_results["betas"][0]
        beta1_ci = fit_results.get("betas_ci", [(np.nan, np.nan)])[0]
        segment_summary, warning = _generate_segment_interpretation(
            "Low-Frequency (Long-term) Fit",
            beta1,
            beta1_ci,
            "β1",
            ci_method_str,
            uncertainty_threshold,
        )
        summary_parts.extend(segment_summary)
        if warning:
            uncertainty_warnings.append(warning)

        # --- Loop through each breakpoint and the segment that follows it ---
        for i in range(n_breakpoints):
            bp_freq = fit_results["breakpoints"][i]
            breakpoints_ci_list = fit_results.get("breakpoints_ci", [])
            bp_ci = (
                breakpoints_ci_list[i]
                if i < len(breakpoints_ci_list)
                else (np.nan, np.nan)
            )

            bp_name = f"Breakpoint {i+1} Period"
            bp_str = f"~{_format_period(bp_freq)}"
            if np.all(np.isfinite(bp_ci)):
                # Note: A lower frequency corresponds to a higher period, so the
                # order of the CI bounds must be swapped when formatting.
                period_ci_str = f"{_format_period(bp_ci[1])}–{_format_period(bp_ci[0])}"
                bp_str += f" (95% CI: {period_ci_str}{ci_method_str})"
                if bp_ci[1] / bp_ci[0] > breakpoint_uncertainty_threshold:
                    uncertainty_warnings.append(
                        f"Warning: The 95% CI for {bp_name} spans more than an "
                        f"order of magnitude ({period_ci_str}), indicating high uncertainty "
                        "in its location."
                    )
            summary_parts.append(f"--- Breakpoint {i+1} @ {bp_str} ---")

            # Determine segment name and get its data
            name = (
                "High-Frequency (Short-term) Fit"
                if (i + 1) == n_breakpoints
                else f"Mid-Frequency Segment {i+1}"
            )
            beta_next = fit_results["betas"][i + 1]
            betas_ci_list = fit_results.get("betas_ci", [])
            beta_next_ci = (
                betas_ci_list[i + 1]
                if (i + 1) < len(betas_ci_list)
                else (np.nan, np.nan)
            )
            beta_name = f"β{i+2}"

            segment_summary, warning = _generate_segment_interpretation(
                name,
                beta_next,
                beta_next_ci,
                beta_name,
                ci_method_str,
                uncertainty_threshold,
            )
            summary_parts.extend(segment_summary)
            if warning:
                uncertainty_warnings.append(warning)
        summary_text = "\n".join(summary_parts)
    else:
        # --- Standard Model Summary ---
        beta = fit_results["beta"]
        ci = (fit_results.get("beta_ci_lower"), fit_results.get("beta_ci_upper"))
        beta_str = f"β = {beta:.2f}"
        if ci[0] is not None and ci[1] is not None and np.all(np.isfinite(ci)):
            beta_str += f" (95% CI: {ci[0]:.2f}–{ci[1]:.2f}{ci_method_str})"
            warning = _get_uncertainty_warning(ci, uncertainty_threshold, name="β")
            if warning:
                uncertainty_warnings.append(warning)

        summary_text = "\n".join(
            [
                f"Standard Analysis for: {param_name}",
                f"Value: {beta_str}",
                f"Persistence Level: {get_persistence_traffic_light(beta)}",
                f"Scientific Meaning: {get_scientific_interpretation(beta)}",
                f"Contextual Comparison: {compare_to_benchmarks(beta)}",
            ]
        )

    # --- Append shared sections (peaks and warnings) ---
    if "significant_peaks" in fit_results and fit_results["significant_peaks"]:
        peaks_summary = "\n\n-----------------------------------\n"
        if "fap_level" in fit_results:
            peaks_summary += (
                "Significant Periodicities Found "
                f"(at {fit_results['fap_threshold']*100:.1f}% FAP Level):\n"
            )
        else:
            peaks_summary += "Significant Periodicities Found:\n"

        for peak in fit_results["significant_peaks"]:
            period_str = f"  - Period: {_format_period(peak['frequency'])}"
            if "residual" in peak:
                peaks_summary += (
                    f"{period_str} (Fit Residual: {peak['residual']:.2f})\n"
                )
            else:
                peaks_summary += f"{period_str}\n"
        summary_text += peaks_summary.rstrip()
    else:
        no_peaks_msg = "\n\nNo significant periodicities were found"
        if "fap_level" in fit_results:
            no_peaks_msg += (
                f" at the {fit_results['fap_threshold']*100:.1f}% FAP level."
            )
        elif "residual_threshold" in fit_results:
            no_peaks_msg += (
                f" at the {fit_results.get('peak_fdr_level', 0.05)*100:.0f}% FDR level."
            )
        else:
            no_peaks_msg += "."
        summary_text += no_peaks_msg

    # Append any collected uncertainty warnings
    if uncertainty_warnings:
        summary_text += "\n\n-----------------------------------\n"
        summary_text += "Uncertainty Report:\n"
        summary_text += "\n".join([f"  - {w}" for w in uncertainty_warnings])

    return {
        "summary_text": auto_summary_header + summary_text,
        "uncertainty_warnings": uncertainty_warnings,
    }
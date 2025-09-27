import numpy as np
import pandas as pd

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
            "Parameter": "TSS",
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


def get_scientific_interpretation(beta):
    """Provides a scientific interpretation of the spectral exponent (beta)."""
    # Relax the check to allow for slightly negative beta values, which can
    # occur with noisy data.
    if beta < -0.5:
        return (
            "Warning: Beta value is significantly negative, which is physically "
            "unrealistic."
        )
    if np.isclose(beta, 0, atol=0.2):
        return "β ≈ 0: White noise (uncorrelated)."
    if np.isclose(beta, 1, atol=0.2):
        return "β ≈ 1: Pink noise (1/f), common in nature."
    if np.isclose(beta, 2, atol=0.2):
        return "β ≈ 2: Brownian noise (random walk)."
    if -0.5 <= beta < 1:
        return (
            "-0.5 < β < 1 (fGn-like): Weak persistence or anti-persistence, "
            "suggesting event-driven transport."
        )
    if 1 < beta < 3:
        return (
            "1 < β < 3 (fBm-like): Strong persistence, suggesting transport is "
            "damped by storage."
        )
    if beta >= 3:
        return "β ≥ 3: Black noise, very smooth signal, may indicate non-stationarity."
    return "No interpretation available."


def get_persistence_traffic_light(beta):
    """Returns a traffic-light style summary of persistence."""
    if beta < 0.5:
        return "🔴 Event-driven"
    if 0.5 <= beta <= 1.0:
        return "🟡 Mixed / weak persistence"
    if beta > 1.0:
        return "🟢 Persistent / subsurface dominated"
    return "⚪ Unknown"


def _format_period(frequency_hz):
    """Converts a frequency in Hz to a human-readable period string."""
    if frequency_hz <= 0:
        return "N/A"

    period_seconds = 1 / frequency_hz
    period_days = period_seconds / 86400

    if period_days > 365.25 * 2:  # Over 2 years
        return f"{period_days / 365.25:.1f} years"
    elif period_days > 30.44 * 2:  # Over 2 months
        return f"{period_days / 30.44:.1f} months"
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


def interpret_results(fit_results, param_name="Parameter", uncertainty_threshold=0.5):
    """
    Generates a comprehensive, human-readable interpretation of the analysis results.
    Handles standard, segmented, and auto analysis types.
    """
    auto_summary_header = ""
    # If this was an auto-analysis, generate a special header
    if fit_results.get("analysis_mode") == "auto":
        model_summaries = []
        for model in fit_results["all_models"]:
            n_bp = model["n_breakpoints"]
            bic_str = f"{model['bic']:.2f}"
            if n_bp == 0:
                name = "Standard"
                beta_str = f"β = {model['beta']:.2f}"
            else:
                name = f"Segmented ({n_bp} BP)"
                betas = ", ".join(
                    [f"β{i+1}={b:.2f}" for i, b in enumerate(model["betas"])]
                )
                beta_str = f"{betas}"
            model_summaries.append(f"  - {name:<15} BIC = {bic_str:<8} ({beta_str})")

        chosen_model_name = fit_results["chosen_model"].replace("_", " ").capitalize()
        auto_summary_header = (
            f"Automatic Analysis for: {param_name}\n"
            "-----------------------------------\n"
            "Model Comparison (Lower BIC is better):\n"
            + "\n".join(model_summaries)
            + f"\n==> Chosen Model: {chosen_model_name}\n"
            "-----------------------------------\n\n"
            f"Details for Chosen ({chosen_model_name}) Model:\n"
        )

    # --- Generate the main summary for the chosen model ---
    n_breakpoints = fit_results.get("n_breakpoints", 0)

    if n_breakpoints > 0:
        # --- Segmented Model Summary ---
        summary_parts = [f"Segmented Analysis for: {param_name}"]

        # --- Handle the first segment (always "Low-Frequency") ---
        beta1 = fit_results["betas"][0]
        beta1_ci = fit_results.get("betas_ci", [(np.nan, np.nan)])[0]
        beta1_str = (
            f"β1 = {beta1:.2f} (95% CI: {beta1_ci[0]:.2f}–{beta1_ci[1]:.2f})"
            if np.all(np.isfinite(beta1_ci))
            else f"β1 = {beta1:.2f}"
        )
        summary_parts.extend(
            [
                "Low-Frequency (Long-term) Fit:",
                f"  {beta1_str}",
                f"  Interpretation: {get_scientific_interpretation(beta1)}",
                f"  Persistence: {get_persistence_traffic_light(beta1)}",
            ]
        )

        # --- Loop through each breakpoint and the segment that follows it ---
        for i in range(n_breakpoints):
            beta_next = fit_results["betas"][i + 1]
            beta_next_ci = fit_results.get("betas_ci", [(np.nan, np.nan)] * (i + 2))[i + 1]
            bp_freq = fit_results["breakpoints"][i]
            bp_ci = fit_results.get("breakpoints_ci", [(np.nan, np.nan)] * (i + 1))[i]

            beta_next_str = (
                f"β{i+2} = {beta_next:.2f} (95% CI: {beta_next_ci[0]:.2f}–{beta_next_ci[1]:.2f})"
                if np.all(np.isfinite(beta_next_ci))
                else f"β{i+2} = {beta_next:.2f}"
            )
            bp_str = (
                f"~{_format_period(bp_freq)} (95% CI: {_format_period(bp_ci[0])}–{_format_period(bp_ci[1])})"
                if np.all(np.isfinite(bp_ci))
                else f"~{_format_period(bp_freq)}"
            )

            # Determine segment name
            name = (
                "High-Frequency (Short-term)"
                if (i + 1) == n_breakpoints
                else f"Mid-Frequency Segment {i+1}"
            )

            summary_parts.append(f"--- Breakpoint {i+1} @ {bp_str} ---")
            summary_parts.extend(
                [
                    f"{name} Fit:",
                    f"  {beta_next_str}",
                    f"  Interpretation: {get_scientific_interpretation(beta_next)}",
                    f"  Persistence: {get_persistence_traffic_light(beta_next)}",
                ]
            )

        summary_text = "\n".join(summary_parts)
        results_dict = {
            "analysis_type": "segmented",
            "n_breakpoints": n_breakpoints,
            "betas": fit_results["betas"],
            "breakpoints": fit_results["breakpoints"],
            "betas_ci": fit_results.get("betas_ci"),
            "breakpoints_ci": fit_results.get("breakpoints_ci"),
        }
    else:
        # --- Standard Model Summary ---
        beta = fit_results["beta"]
        ci = (fit_results.get("beta_ci_lower"), fit_results.get("beta_ci_upper"))
        beta_str = (
            f"β = {beta:.2f} (95% CI: {ci[0]:.2f}–{ci[1]:.2f})"
            if (
                ci[0] is not None
                and np.isfinite(ci[0])
                and ci[1] is not None
                and np.isfinite(ci[1])
            )
            else f"β = {beta:.2f}"
        )
        summary_text = "\n".join([
            f"Standard Analysis for: {param_name}",
            f"Value: {beta_str}",
            f"Persistence Level: {get_persistence_traffic_light(beta)}",
            f"Scientific Meaning: {get_scientific_interpretation(beta)}",
            f"Contextual Comparison: {compare_to_benchmarks(beta)}",
        ])

        uncertainty_warning = None
        if (
            ci[0] is not None and ci[1] is not None and
            (ci[1] - ci[0]) > uncertainty_threshold
        ):
            uncertainty_warning = (
                f"Warning: The confidence interval width ({ci[1] - ci[0]:.2f}) is "
                "large, suggesting high uncertainty."
            )
            summary_text += f"\n\n{uncertainty_warning}"

        results_dict = {
            "analysis_type": "standard",
            "beta_value": beta,
            "confidence_interval": ci,
            "uncertainty_warning": uncertainty_warning,
        }

    # --- Append shared sections (peaks) and prepend auto-summary ---
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
        summary_text += "\n\nNo significant periodicities were found."

    results_dict["summary_text"] = auto_summary_header + summary_text
    return results_dict
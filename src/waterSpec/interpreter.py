import numpy as np
import pandas as pd

# Benchmark table based on Liang et al. (2021) and general hydrological knowledge.
BENCHMARK_TABLE = pd.DataFrame([
    {"Parameter": "E. coli", "Typical β Range": (0.1, 0.5), "Interpretation": "Weak persistence", "Dominant Pathway": "Surface runoff"},
    {"Parameter": "TSS", "Typical β Range": (0.4, 0.8), "Interpretation": "Weak persistence", "Dominant Pathway": "Surface runoff"},
    {"Parameter": "Ortho-P", "Typical β Range": (0.6, 1.2), "Interpretation": "Mixed", "Dominant Pathway": "Surface/Shallow subsurface"},
    {"Parameter": "Chloride", "Typical β Range": (1.3, 1.7), "Interpretation": "Strong persistence", "Dominant Pathway": "Subsurface"},
    {"Parameter": "Nitrate-N", "Typical β Range": (1.5, 2.0), "Interpretation": "Strong persistence", "Dominant Pathway": "Subsurface"},
    {"Parameter": "Discharge (Q)", "Typical β Range": (1.0, 1.8), "Interpretation": "Persistent", "Dominant Pathway": "Integrated signal"},
]).set_index("Parameter")

def get_scientific_interpretation(beta):
    """Provides a scientific interpretation of the spectral exponent (beta)."""
    # Relax the check to allow for slightly negative beta values, which can occur with noisy data.
    if beta < -0.5:
        return "Warning: Beta value is significantly negative, which is physically unrealistic."
    if np.isclose(beta, 0, atol=0.2): return "β ≈ 0: White noise (uncorrelated)."
    if np.isclose(beta, 1, atol=0.2): return "β ≈ 1: Pink noise (1/f), common in nature."
    if np.isclose(beta, 2, atol=0.2): return "β ≈ 2: Brownian noise (random walk)."
    if -0.5 <= beta < 1: return f"-0.5 < β < 1 (fGn-like): Weak persistence or anti-persistence, suggesting event-driven transport."
    if 1 < beta < 3: return f"1 < β < 3 (fBm-like): Strong persistence, suggesting transport is damped by storage."
    if beta >= 3: return "β ≥ 3: Black noise, very smooth signal, may indicate non-stationarity."
    return "No interpretation available."

def get_persistence_traffic_light(beta):
    """Returns a traffic-light style summary of persistence."""
    if beta < 0.5: return "🔴 Event-driven (Low Persistence)"
    if 0.5 <= beta <= 1.0: return "🟡 Mixed / Weak Persistence"
    if beta > 1.0: return "🟢 Persistent / Subsurface Dominated (High Persistence)"
    return "⚪ Unknown"

def _format_period(frequency_hz):
    """Converts a frequency in Hz to a human-readable period string."""
    if frequency_hz <= 0:
        return "N/A"

    period_seconds = 1 / frequency_hz
    period_days = period_seconds / 86400

    if period_days > 365.25 * 2: # Over 2 years
        return f"{period_days / 365.25:.1f} years"
    elif period_days > 30.44 * 2: # Over 2 months
        return f"{period_days / 30.44:.1f} months"
    else:
        return f"{period_days:.1f} days"

def compare_to_benchmarks(beta):
    """Compares a beta value to the benchmark table."""
    closest_match = None
    min_distance = float('inf')

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
    if fit_results.get('analysis_mode') == 'auto':
        chosen_model = fit_results['chosen_model']
        bic_comp = fit_results['bic_comparison']

        standard_fit = fit_results.get('standard_fit', {})
        segmented_fit = fit_results.get('segmented_fit', {})

        standard_beta = standard_fit.get('beta', np.nan)
        segmented_beta1 = segmented_fit.get('beta1', np.nan)
        segmented_beta2 = segmented_fit.get('beta2', np.nan)

        def format_bic(val):
            return f"{val:.2f}" if val is not None and np.isfinite(val) else "N/A"

        standard_bic_str = format_bic(bic_comp.get('standard'))
        segmented_bic_str = format_bic(bic_comp.get('segmented'))

        standard_beta_str = f"β = {standard_beta:.2f}" if np.isfinite(standard_beta) else ""
        segmented_beta_str = f"β1 = {segmented_beta1:.2f}, β2 = {segmented_beta2:.2f}" if np.isfinite(segmented_beta1) else ""

        auto_summary_header = (
            f"Automatic Analysis for: {param_name}\n"
            f"-----------------------------------\n"
            f"Model Comparison (Lower BIC is better):\n"
            f"  - Standard Fit:   BIC = {standard_bic_str} ({standard_beta_str})\n"
            f"  - Segmented Fit:  BIC = {segmented_bic_str} ({segmented_beta_str})\n"
            f"==> Chosen Model: {chosen_model.capitalize()}\n"
            f"-----------------------------------\n\n"
            f"Details for Chosen ({chosen_model.capitalize()}) Model:\n"
        )

    is_segmented = 'beta2' in fit_results and np.isfinite(fit_results['beta2'])

    if is_segmented:
        beta1, beta2, breakpoint_freq = fit_results['beta1'], fit_results['beta2'], fit_results['breakpoint']
        interp1, interp2 = get_scientific_interpretation(beta1), get_scientific_interpretation(beta2)
        summary_text = (
            f"Segmented Analysis for: {param_name}\n"
            f"Breakpoint Period ≈ {_format_period(breakpoint_freq)}\n"
            f"-----------------------------------\n"
            f"Low-Frequency (Long-term) Fit:\n"
            f"  β1 = {beta1:.2f}\n"
            f"  Interpretation: {interp1}\n"
            f"  Persistence: {get_persistence_traffic_light(beta1)}\n"
            f"-----------------------------------\n"
            f"High-Frequency (Short-term) Fit:\n"
            f"  β2 = {beta2:.2f}\n"
            f"  Interpretation: {interp2}\n"
            f"  Persistence: {get_persistence_traffic_light(beta2)}"
        )
        results_dict = {"analysis_type": "segmented", "beta1": beta1, "beta2": beta2, "breakpoint": breakpoint_freq}
    else:
        beta = fit_results['beta']
        ci = (fit_results.get('beta_ci_lower'), fit_results.get('beta_ci_upper'))
        sci_interp, traffic_light, benchmark_comp = get_scientific_interpretation(beta), get_persistence_traffic_light(beta), compare_to_benchmarks(beta)
        beta_str = f"β = {beta:.2f} (95% CI: {ci[0]:.2f}–{ci[1]:.2f})" if ci[0] is not None and ci[1] is not None else f"β = {beta:.2f}"
        summary_text = (
            f"Standard Analysis for: {param_name}\n"
            f"Value: {beta_str}\n"
            f"Persistence Level: {traffic_light}\n"
            f"Scientific Meaning: {sci_interp}\n"
            f"Contextual Comparison: {benchmark_comp}"
        )
        uncertainty_warning = None
        if ci[0] is not None and ci[1] is not None and (ci[1] - ci[0]) > uncertainty_threshold:
            uncertainty_warning = f"Warning: The confidence interval width ({ci[1] - ci[0]:.2f}) is large, suggesting high uncertainty."
            summary_text += f"\n\n{uncertainty_warning}"
        results_dict = {"analysis_type": "standard", "beta_value": beta, "confidence_interval": ci, "persistence_level": traffic_light, "scientific_interpretation": sci_interp, "benchmark_comparison": benchmark_comp, "uncertainty_warning": uncertainty_warning, "benchmark_table": BENCHMARK_TABLE.to_dict(orient='index')}

    # --- Append shared sections (peaks) and prepend auto-summary ---
    if 'significant_peaks' in fit_results and fit_results['significant_peaks']:
        peaks_summary = "\n\n-----------------------------------\nSignificant Periodicities Found:"
        for peak in fit_results['significant_peaks']:
            period_str = f"\n  - Period: {_format_period(peak['frequency'])}"
            if 'fap' in peak:
                peaks_summary += f"{period_str} (FAP: {peak['fap']:.2E})"
            elif 'residual' in peak:
                peaks_summary += f"{period_str} (Fit Residual: {peak['residual']:.2f})"
            else:
                peaks_summary += period_str
        summary_text += peaks_summary

    results_dict["summary_text"] = auto_summary_header + summary_text if auto_summary_header else summary_text
    return results_dict

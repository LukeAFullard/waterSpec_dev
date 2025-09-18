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
        raise ValueError("Beta value is significantly negative, which is physically unrealistic.")
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

def interpret_results(beta, ci=None, param_name="Parameter", uncertainty_threshold=0.5):
    """
    Generates a comprehensive, human-readable interpretation of the analysis results.
    """
    sci_interp = get_scientific_interpretation(beta)
    traffic_light = get_persistence_traffic_light(beta)
    benchmark_comp = compare_to_benchmarks(beta)

    if ci:
        beta_str = f"β = {beta:.2f} (95% CI: {ci[0]:.2f}–{ci[1]:.2f})"
    else:
        beta_str = f"β = {beta:.2f}"

    summary_text = (
        f"Analysis for: {param_name}\n"
        f"Value: {beta_str}\n"
        f"Persistence Level: {traffic_light}\n"
        f"Scientific Meaning: {sci_interp}\n"
        f"Contextual Comparison: {benchmark_comp}"
    )

    uncertainty_warning = None
    if ci:
        ci_width = ci[1] - ci[0]
        if ci_width > uncertainty_threshold:
            uncertainty_warning = (
                f"Warning: The confidence interval width ({ci_width:.2f}) is large, "
                "suggesting high uncertainty."
            )
            summary_text += f"\n\n{uncertainty_warning}"

    return {
        "summary_text": summary_text,
        "beta_value": beta,
        "confidence_interval": ci,
        "persistence_level": traffic_light,
        "scientific_interpretation": sci_interp,
        "benchmark_comparison": benchmark_comp,
        "uncertainty_warning": uncertainty_warning,
        "benchmark_table": BENCHMARK_TABLE.to_dict(orient='index')
    }

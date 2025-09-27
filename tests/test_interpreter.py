import numpy as np
import pytest

from waterSpec.interpreter import (
    _format_period,
    compare_to_benchmarks,
    get_persistence_traffic_light,
    get_scientific_interpretation,
    interpret_results,
)


def test_get_scientific_interpretation():
    assert "White noise" in get_scientific_interpretation(0)
    assert "fGn-like" in get_scientific_interpretation(0.5)
    assert "Pink noise" in get_scientific_interpretation(1.0)
    assert "fBm-like" in get_scientific_interpretation(1.5)
    assert "Brownian noise" in get_scientific_interpretation(2.0)
    assert "Warning" in get_scientific_interpretation(-1.0)


def test_get_persistence_traffic_light():
    assert "Event-driven" in get_persistence_traffic_light(0.2)
    assert "Mixed" in get_persistence_traffic_light(0.7)
    assert "Persistent" in get_persistence_traffic_light(1.5)


def test_compare_to_benchmarks():
    assert "E. coli" in compare_to_benchmarks(0.3)
    assert "Nitrate-N" in compare_to_benchmarks(1.8)
    # A beta of 0.9 falls directly in the Ortho-P range (0.6-1.2)
    assert "Ortho-P" in compare_to_benchmarks(0.9)


def test_interpret_results_basic():
    fit_results = {"beta": 1.7, "betas": [1.7], "n_breakpoints": 0}
    results = interpret_results(fit_results, param_name="Nitrate")
    summary = results["summary_text"]
    assert "Analysis for: Nitrate" in summary
    assert "β = 1.70" in summary
    assert "Persistent" in summary
    assert "fBm-like" in summary
    assert "Chloride" in summary
    assert not results["uncertainty_warnings"]


def test_interpret_results_with_ci():
    fit_results = {
        "beta": 1.7,
        "betas": [1.7],
        "n_breakpoints": 0,
        "beta_ci_lower": 1.5,
        "beta_ci_upper": 1.9,
    }
    results = interpret_results(fit_results, param_name="Nitrate")
    assert "95% CI: 1.50–1.90" in results["summary_text"]
    assert not results["uncertainty_warnings"]


def test_interpret_results_with_wide_ci():
    fit_results = {
        "beta": 1.7,
        "betas": [1.7],
        "n_breakpoints": 0,
        "beta_ci_lower": 1.0,
        "beta_ci_upper": 2.4,
    }
    results = interpret_results(fit_results, param_name="Nitrate")
    assert "Uncertainty Report:" in results["summary_text"]
    assert len(results["uncertainty_warnings"]) == 1
    assert "Warning: The 95% CI for β is wide" in results["uncertainty_warnings"][0]


def test_interpret_results_no_param_name():
    fit_results = {"beta": 0.5, "betas": [0.5], "n_breakpoints": 0}
    results = interpret_results(fit_results)
    assert "Analysis for: Parameter" in results["summary_text"]


def test_interpret_results_auto_mode():
    """
    Test the interpreter's output when provided with results from an 'auto' analysis.
    """
    standard_model = {"beta": 1.2, "betas": [1.2], "bic": 120.5, "n_breakpoints": 0}
    segmented_model = {
        "betas": [0.8, 1.5],
        "breakpoints": [0.1],
        "bic": 150.2,
        "n_breakpoints": 1,
    }

    fit_results = {
        "analysis_mode": "auto",
        "chosen_model": "standard",
        "all_models": [standard_model, segmented_model],
        **standard_model,  # The chosen model's results are at the top level
    }

    results = interpret_results(fit_results, param_name="Test Param")
    summary = results["summary_text"]

    # Check for the auto-analysis header and content
    assert "Automatic Analysis for: Test Param" in summary
    assert "Model Comparison (Lower BIC is better):" in summary

    # Check for the correctly formatted model comparison lines
    expected_line_1 = f"  - {'Standard':<15} BIC = {120.5:<8.2f} (β = 1.20)"
    expected_line_2 = (
        f"  - {'Segmented (1 BP)':<15} BIC = {150.2:<8.2f} (β1=0.80, β2=1.50)"
    )

    assert expected_line_1 in summary
    assert expected_line_2 in summary
    assert "Chosen Model: Standard" in summary

    # Check that the detailed interpretation for the chosen model is present
    assert "Details for Chosen (Standard) Model:" in summary
    assert "Persistence Level: 🟢 Persistent" in summary


def test_interpret_results_with_peaks():
    """
    Test that the summary text includes information about significant peaks.
    """
    yearly_freq_hz = 1 / (365.25 * 86400)
    fit_results = {
        "beta": 0.8,
        "betas": [0.8],
        "n_breakpoints": 0,
        "significant_peaks": [
            {"frequency": yearly_freq_hz, "residual": 5.5},
            {"frequency": 1 / (30 * 86400), "residual": 4.2},
        ],
    }

    results = interpret_results(fit_results)
    summary = results["summary_text"]

    assert "Significant Periodicities Found:" in summary
    assert "Period: 12.0 months" in summary
    assert "Period: 30.0 days" in summary
    assert "(Fit Residual: 5.50)" in summary
    assert "(Fit Residual: 4.20)" in summary


# --- New tests for period formatting ---


@pytest.mark.parametrize(
    "frequency_hz, expected_string",
    [
        (1 / (10 * 86400), "10.0 days"),
        (1 / (90 * 86400), "3.0 months"),
        (1 / (3 * 365.25 * 86400), "3.0 years"),
        (1 / (800 * 365.25 * 86400), "800.0 years"),
        (0, "N/A"),
    ],
)
def test_format_period(frequency_hz, expected_string):
    assert _format_period(frequency_hz) == expected_string


def test_interpret_results_segmented():
    """
    Test the interpreter output for a segmented fit, checking for the new
    breakpoint period format.
    """
    breakpoint_freq = 1 / (90 * 86400)
    fit_results = {
        "betas": [0.5, 1.5],
        "breakpoints": [breakpoint_freq],
        "n_breakpoints": 1,
        "significant_peaks": [],
    }

    results = interpret_results(fit_results, param_name="Ortho-P")
    summary = results["summary_text"]

    assert "Segmented Analysis for: Ortho-P" in summary
    assert "--- Breakpoint 1 @ ~3.0 months ---" in summary
    assert "Low-Frequency (Long-term) Fit" in summary
    assert "β1 = 0.50" in summary
    assert "High-Frequency (Short-term) Fit" in summary
    assert "β2 = 1.50" in summary


# --- New tests for increased coverage ---


def test_get_uncertainty_warning_handles_nan():
    """Test that _get_uncertainty_warning returns None for NaN CI."""
    from waterSpec.interpreter import _get_uncertainty_warning

    assert _get_uncertainty_warning((np.nan, np.nan), 0.5) is None


def test_get_scientific_interpretation_handles_nan():
    """Test get_scientific_interpretation with a NaN beta value."""
    assert "No interpretation available" in get_scientific_interpretation(np.nan)


def test_compare_to_benchmarks_logic():
    """Test compare_to_benchmarks for values inside and between ranges."""
    # A beta of 1.25 falls *within* the "Discharge (Q)" range (1.0-1.8)
    assert "Similar to Discharge (Q)" in compare_to_benchmarks(1.25)
    # A beta of 2.5 is outside all ranges and is closest to the mean of Nitrate-N (1.75).
    assert "Closest to Nitrate-N" in compare_to_benchmarks(2.5)


def test_interpret_results_auto_mode_segmented_chosen():
    """Test auto mode summary when a segmented model is chosen."""
    fit_results = {
        "analysis_mode": "auto",
        "chosen_model": "segmented_1bp",
        "all_models": [
            {"beta": 1.2, "betas": [1.2], "bic": 120, "n_breakpoints": 0},
            {
                "betas": [0.8, 1.5],
                "breakpoints": [0.1],
                "bic": 100,
                "n_breakpoints": 1,
            },
        ],
        "betas": [0.8, 1.5],
        "breakpoints": [0.1],
        "n_breakpoints": 1,
    }
    results = interpret_results(fit_results)
    summary = results["summary_text"]
    assert "Chosen Model: Segmented 1bp" in summary
    assert "Details for Chosen (Segmented 1bp) Model:" in summary


def test_interpret_results_segmented_wide_ci():
    """Test uncertainty warnings for segmented models with wide CIs."""
    fit_results = {
        "betas": [0.5, 1.5],
        "betas_ci": [(0.1, 0.9), (1.0, 2.8)],  # Wide CI for beta2
        "breakpoints": [0.1],
        "breakpoints_ci": [(0.01, 0.5)],  # Wide CI for breakpoint
        "n_breakpoints": 1,
    }
    results = interpret_results(
        fit_results, breakpoint_uncertainty_threshold=10  # Lower threshold for testing
    )
    summary = results["summary_text"]
    warnings = results["uncertainty_warnings"]

    assert "Uncertainty Report:" in summary
    assert len(warnings) == 3
    assert "Warning: The 95% CI for β1 is wide" in warnings[0]
    assert "Warning: The 95% CI for β2 is wide" in warnings[1]
    assert (
        "Warning: The 95% CI for Breakpoint 1 Period spans more than an order of magnitude"
        in warnings[2]
    )


def test_interpret_results_segmented_nan_ci():
    """Test segmented summary when CIs are NaN."""
    fit_results = {
        "betas": [0.5, 1.5],
        "betas_ci": [(0.1, 0.9), (np.nan, np.nan)],
        "breakpoints": [0.1],
        "n_breakpoints": 1,
    }
    results = interpret_results(fit_results)
    summary = results["summary_text"]
    # Check that the CI string is not present for the second beta
    assert "β1 = 0.50 (95% CI: 0.10–0.90" in summary
    # Find the line containing "β2 = 1.50" and check that it doesn't contain "CI"
    beta2_line = [
        line for line in summary.split("\n") if line.strip().startswith("β2 = 1.50")
    ][0]
    assert "CI" not in beta2_line


def test_interpret_results_no_peaks_generic_message():
    """Test the generic 'no peaks found' message."""
    fit_results = {"beta": 1.0, "betas": [1.0], "n_breakpoints": 0, "significant_peaks": []}
    # No 'fap_level' or 'residual_threshold' keys are provided
    results = interpret_results(fit_results)
    assert "No significant periodicities were found." in results["summary_text"]

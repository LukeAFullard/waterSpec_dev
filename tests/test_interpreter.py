import pytest

from waterSpec.interpreter import (
    _format_period,
    compare_to_benchmarks,
    get_persistence_traffic_light,
    get_scientific_interpretation,
    interpret_results,
)


def test_get_scientific_interpretation():
    assert "(White Noise)" in get_scientific_interpretation(0)
    assert "fGn-like" in get_scientific_interpretation(0.5)
    assert "(Pink Noise)" in get_scientific_interpretation(1.0)
    assert "fBm-like" in get_scientific_interpretation(1.5)
    assert "(Brownian Noise)" in get_scientific_interpretation(2.0)
    assert "Anti-persistent" in get_scientific_interpretation(-1.0)


def test_get_persistence_traffic_light():
    assert "Low (Event-driven)" in get_persistence_traffic_light(0.2)
    assert "Medium (Mixed)" in get_persistence_traffic_light(0.7)
    assert "High (Storage-dominated)" in get_persistence_traffic_light(1.5)


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
    assert "High (Storage-dominated)" in summary
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
    assert "High (Storage-dominated)" in summary


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


def test_interpret_results_auto_mode_with_failures():
    """
    Test that the auto-analysis summary includes reasons for failed models.
    """
    # This is the only successful model, so it will be the chosen one.
    standard_model = {"beta": 1.2, "betas": [1.2], "bic": 120.5, "n_breakpoints": 0}

    fit_results = {
        "analysis_mode": "auto",
        "chosen_model": "standard",
        "all_models": [standard_model],  # Only successful models are in here
        "failed_model_reasons": [
            "Segmented model (1 bp): No significant breakpoint found (Davies test p > 0.05)"
        ],
        **standard_model,  # Unpack the chosen model's results
    }

    results = interpret_results(fit_results, param_name="Test Param")
    summary = results["summary_text"]

    # Check that the header and successful model are present
    assert "Automatic Analysis for: Test Param" in summary
    assert "Standard" in summary
    assert "BIC = 120.50" in summary
    assert "Chosen Model: Standard" in summary

    # Check that the failed model reason is also included in the summary
    assert "Models that failed to fit:" in summary
    assert "Segmented model (1 bp): No significant breakpoint found" in summary


def test_interpret_results_segmented_with_ci():
    """
    Test that the breakpoint period CI is formatted in the correct
    (low-high) order, which is the reverse of the frequency CI.
    """
    from waterSpec.interpreter import DAYS_PER_MONTH, SECONDS_PER_DAY

    # Calculate the exact frequencies that will format to "2.0 months" and "4.0 months"
    # Note: A lower frequency corresponds to a higher period.
    period_4_months = 4 * DAYS_PER_MONTH * SECONDS_PER_DAY
    period_2_months = 2 * DAYS_PER_MONTH * SECONDS_PER_DAY

    low_freq = 1 / period_4_months
    high_freq = 1 / period_2_months

    fit_results = {
        "betas": [0.5, 1.5],
        "breakpoints": [(low_freq + high_freq) / 2],  # Midpoint
        "breakpoints_ci": [(low_freq, high_freq)],
        "n_breakpoints": 1,
        "significant_peaks": [],
        "ci_method": "bootstrap",  # This key is required to show the CI string
    }

    results = interpret_results(fit_results, param_name="Test")
    summary = results["summary_text"]

    # The period CI should be (2.0 months–4.0 months)
    assert "(95% CI: 2.0 months–4.0 months" in summary

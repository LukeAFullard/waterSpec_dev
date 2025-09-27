import numpy as np
import pytest

from waterSpec.fitter import fit_segmented_spectrum, fit_standard_model


@pytest.fixture
def synthetic_spectrum():
    """
    Generates a synthetic power spectrum with a known spectral exponent (beta).
    """
    # Define spectral parameters
    known_beta = 1.5
    n_points = 100

    # Generate a frequency array (log-spaced is common for spectra)
    frequency = np.logspace(-3, 0, n_points)

    # Generate the power spectrum with some noise
    # power = C * frequency ** -beta
    # On a log-log plot, this is: log(power) = log(C) - beta * log(frequency)
    rng = np.random.default_rng(42)
    noise = rng.normal(0, 0.1, n_points)
    log_power = -known_beta * np.log(frequency) + noise
    power = np.exp(log_power)

    return frequency, power, known_beta


def test_fit_standard_model_returns_correct_beta(synthetic_spectrum):
    """
    Test that fit_standard_model correctly estimates the spectral exponent (beta).
    """
    frequency, power, known_beta = synthetic_spectrum

    # Fit the spectrum
    fit_results = fit_standard_model(frequency, power)

    # Check that the returned beta is close to the known beta
    assert "beta" in fit_results
    assert fit_results["beta"] == pytest.approx(
        known_beta, abs=0.1
    )  # Allow some tolerance due to noise


def test_fit_standard_model_returns_good_fit_metrics(synthetic_spectrum):
    """
    Test that fit_standard_model returns a good R-squared value for a clean signal
    when using the OLS method.
    """
    frequency, power, _ = synthetic_spectrum

    # Fit the spectrum using OLS specifically for this test
    fit_results = fit_standard_model(frequency, power, method="ols")

    # Check that the R-squared value indicates a good fit
    assert "r_squared" in fit_results
    assert fit_results["r_squared"] > 0.95


def test_fit_standard_model_theil_sen(synthetic_spectrum):
    """
    Test that fit_standard_model with method='theil-sen' correctly estimates beta.
    """
    frequency, power, known_beta = synthetic_spectrum

    # Fit the spectrum using the Theil-Sen estimator
    fit_results = fit_standard_model(frequency, power, method="theil-sen")

    # Check that the returned beta is close to the known beta
    assert "beta" in fit_results
    assert fit_results["beta"] == pytest.approx(known_beta, abs=0.1)

    # Check that the other metrics are not present, as expected for Theil-Sen
    assert "r_squared" not in fit_results
    assert "stderr" not in fit_results


def test_fit_standard_model_with_bootstrap_ci(synthetic_spectrum):
    """
    Test that fit_standard_model returns a confidence interval for beta.
    """
    frequency, power, known_beta = synthetic_spectrum

    # Fit the spectrum with bootstrap
    fit_results = fit_standard_model(
        frequency, power, n_bootstraps=100, seed=42
    )  # Use a small number for testing

    # Check that the results dictionary contains the required keys
    assert "beta" in fit_results
    assert "beta_ci_lower" in fit_results
    assert "beta_ci_upper" in fit_results

    # Check that the main beta estimate is within the confidence interval
    assert (
        fit_results["beta_ci_lower"]
        <= fit_results["beta"]
        <= fit_results["beta_ci_upper"]
    )

    # Check that the known beta is within the confidence interval
    # (it should be, most of the time)
    assert (
        (fit_results["beta_ci_lower"] - 0.01)
        <= known_beta
        <= (fit_results["beta_ci_upper"] + 0.01)
    )

    # Check that the confidence interval is not excessively wide
    assert (fit_results["beta_ci_upper"] - fit_results["beta_ci_lower"]) < 1.0


@pytest.fixture
def multifractal_spectrum():
    """
    Generates a synthetic power spectrum with two slopes and a known breakpoint.
    """
    n_points = 200
    breakpoint_freq = 0.1
    beta1 = 0.5
    beta2 = 1.8

    # Generate a frequency array
    frequency = np.logspace(-3, 1, n_points)

    # Create the two-slope power spectrum
    power = np.zeros(n_points)

    # First segment
    mask1 = frequency < breakpoint_freq
    power[mask1] = frequency[mask1] ** -beta1

    # Second segment
    mask2 = frequency >= breakpoint_freq
    # We need to scale the second segment to connect smoothly at the breakpoint
    scale_factor = (breakpoint_freq**-beta1) / (breakpoint_freq**-beta2)
    power[mask2] = scale_factor * (frequency[mask2] ** -beta2)

    # Add some noise
    rng = np.random.default_rng(42)
    noise = rng.normal(0, 0.1, n_points)
    log_power = np.log(power) + noise
    power = np.exp(log_power)

    return frequency, power, breakpoint_freq, beta1, beta2


def test_fit_segmented_spectrum(multifractal_spectrum):
    """
    Test that fit_segmented_spectrum correctly identifies the breakpoint and slopes.
    """
    frequency, power, known_breakpoint, known_beta1, known_beta2 = multifractal_spectrum

    # Fit the segmented spectrum
    results = fit_segmented_spectrum(frequency, power)

    # Check that the results contain the expected keys
    assert "breakpoint" in results
    assert "beta1" in results
    assert "beta2" in results

    # Check that the identified breakpoint is close to the known breakpoint
    # The breakpoint is on a log scale, so we check the log values
    assert np.log10(results["breakpoint"]) == pytest.approx(
        np.log10(known_breakpoint), abs=0.5
    )

    # Check that the estimated betas are close to the known betas
    assert results["beta1"] == pytest.approx(known_beta1, abs=0.3)
    assert results["beta2"] == pytest.approx(known_beta2, abs=0.3)


# --- Edge Case Tests ---


def test_fit_standard_model_white_noise():
    """Test fitting a flat spectrum (white noise), where beta should be ~0."""
    frequency = np.linspace(0.01, 1, 100)
    # Power is constant for white noise, with some random variation
    rng = np.random.default_rng(42)
    power = np.ones_like(frequency) + rng.normal(0, 0.1, len(frequency))

    fit_results = fit_standard_model(frequency, power)
    assert fit_results["beta"] == pytest.approx(0.0, abs=0.1)


def test_fit_standard_model_insufficient_data():
    """Test standard model function when the initial fit fails."""
    # Not enough data points to perform a fit
    frequency = np.array([1.0])
    power = np.array([1.0])

    results = fit_standard_model(frequency, power)

    # Check that all results are NaN
    assert np.isnan(results["beta"])
    assert np.isnan(results["bic"])
    assert np.isnan(results["beta_ci_lower"])
    assert np.isnan(results["beta_ci_upper"])


def test_fit_standard_model_is_reproducible(synthetic_spectrum):
    """
    Test that the bootstrap function produces the same results when the same
    seed is provided.
    """
    frequency, power, _ = synthetic_spectrum

    # Fit twice with the same seed
    results1 = fit_standard_model(
        frequency, power, n_bootstraps=50, seed=123
    )
    results2 = fit_standard_model(
        frequency, power, n_bootstraps=50, seed=123
    )

    # Fit once with a different seed
    results3 = fit_standard_model(
        frequency, power, n_bootstraps=50, seed=456
    )

    # The first two results should be identical
    assert results1["beta_ci_lower"] == results2["beta_ci_lower"]
    assert results1["beta_ci_upper"] == results2["beta_ci_upper"]

    # The third result should be different
    assert results1["beta_ci_lower"] != results3["beta_ci_lower"]


def test_fit_segmented_spectrum_handles_exceptions(
    multifractal_spectrum, mocker
):
    """
    Test that fit_segmented_spectrum catches exceptions from the underlying
    library and returns a meaningful summary.
    """
    frequency, power, _, _, _ = multifractal_spectrum

    # Mock the Fit class to raise an exception upon initialization
    mocker.patch(
        "waterSpec.fitter.piecewise_regression.Fit",
        side_effect=RuntimeError("Test Exception"),
    )

    with pytest.warns(UserWarning, match="Segmented regression failed"):
        results = fit_segmented_spectrum(frequency, power)

    assert "model_summary" in results
    assert "failed with an unexpected error" in results["model_summary"]


def test_fit_segmented_spectrum_p_threshold(multifractal_spectrum, mocker):
    """
    Test that the p_threshold for the Davies test is correctly used to
    determine if a segmented fit is statistically significant.
    """
    frequency, power, _, _, _ = multifractal_spectrum

    # Mock the result of the piecewise_regression fit to control the Davies p-value
    mock_fit_result = mocker.MagicMock()
    mock_fit_result.davies = 0.1  # Let's say the p-value is 0.1
    mock_fit_result.get_results.return_value = {
        "converged": True,
        "bic": 100,
        "r_squared": 0.9,
        "estimates": {
            "breakpoint1": {"estimate": np.log(0.1)},
            "alpha1": {"estimate": -0.5},
            "beta1": {"estimate": -1.3},
        },
    }
    mock_fit_result.summary.return_value = "Mock Summary"
    # The predict method needs to return something with the correct shape
    valid_indices = (frequency > 0) & (power > 0)
    log_freq = np.log(frequency[valid_indices])
    mock_fit_result.predict.return_value = np.zeros_like(log_freq)

    mocker.patch(
        "waterSpec.fitter.piecewise_regression.Fit", return_value=mock_fit_result
    )

    # --- Case 1: p_threshold is lower than the p-value (e.g., 0.05 < 0.1) ---
    # The fit should be rejected.
    results_rejected = fit_segmented_spectrum(
        frequency, power, n_breakpoints=1, p_threshold=0.05
    )

    assert "model_summary" in results_rejected
    assert "No significant breakpoint found" in results_rejected["model_summary"]
    assert "breakpoint" not in results_rejected  # The fit details should not be present

    # --- Case 2: p_threshold is higher than the p-value (e.g., 0.15 > 0.1) ---
    # The fit should be accepted.
    results_accepted = fit_segmented_spectrum(
        frequency, power, n_breakpoints=1, p_threshold=0.15, n_bootstraps=0
    )

    assert "model_summary" in results_accepted
    assert "No significant breakpoint found" not in results_accepted["model_summary"]
    assert "breakpoint" in results_accepted  # The fit details should be present
    assert results_accepted["davies_p_value"] == 0.1


def test_fit_standard_model_with_parametric_ci_ols(synthetic_spectrum):
    """Test parametric CI calculation for the OLS method."""
    frequency, power, known_beta = synthetic_spectrum
    results = fit_standard_model(
        frequency, power, method="ols", ci_method="parametric"
    )

    assert "beta_ci_lower" in results and "beta_ci_upper" in results
    assert np.isfinite(results["beta_ci_lower"])
    assert np.isfinite(results["beta_ci_upper"])
    assert results["beta_ci_lower"] < results["beta_ci_upper"]
    # Check if the known beta is within the CI
    assert results["beta_ci_lower"] <= known_beta <= results["beta_ci_upper"]


def test_fit_standard_model_with_parametric_ci_theil_sen(synthetic_spectrum):
    """Test parametric CI calculation for the Theil-Sen method."""
    frequency, power, known_beta = synthetic_spectrum
    results = fit_standard_model(
        frequency, power, method="theil-sen", ci_method="parametric"
    )

    assert "beta_ci_lower" in results and "beta_ci_upper" in results
    assert np.isfinite(results["beta_ci_lower"])
    assert np.isfinite(results["beta_ci_upper"])
    assert results["beta_ci_lower"] < results["beta_ci_upper"]
    # Theil-sen is robust, so the CI should contain the known beta
    assert results["beta_ci_lower"] <= known_beta <= results["beta_ci_upper"]


def test_fit_segmented_spectrum_with_parametric_ci(multifractal_spectrum):
    """Test parametric CI calculation for segmented models."""
    frequency, power, known_breakpoint, known_beta1, known_beta2 = multifractal_spectrum

    with pytest.warns(UserWarning, match="Parametric confidence intervals"):
        results = fit_segmented_spectrum(frequency, power, ci_method="parametric")

    assert "betas_ci" in results and "breakpoints_ci" in results
    assert len(results["betas_ci"]) == 2
    assert len(results["breakpoints_ci"]) == 1

    # Check CI for the first slope (should be valid)
    beta1_ci = results["betas_ci"][0]
    assert np.all(np.isfinite(beta1_ci))
    assert beta1_ci[0] <= known_beta1 <= beta1_ci[1]

    # Check CI for the second slope (should be NaN, as it's not directly available)
    beta2_ci = results["betas_ci"][1]
    assert np.all(np.isnan(beta2_ci))

    # Check CI for the breakpoint (should be valid)
    breakpoint_ci = results["breakpoints_ci"][0]
    assert np.all(np.isfinite(breakpoint_ci))
    assert breakpoint_ci[0] <= known_breakpoint <= breakpoint_ci[1]

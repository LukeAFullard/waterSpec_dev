import numpy as np
import pytest

from waterSpec.fitter import fit_segmented_spectrum


def test_fit_segmented_spectrum_against_known_data():
    """
    Validates the segmented fitter against a known ground truth, based on the
    example data from the piecewise-regression library's documentation.

    This ensures that our implementation correctly interprets the results from
    the underlying library.
    """
    # 1. Generate synthetic data with known parameters
    # This code is adapted from the official piecewise-regression documentation
    np.random.seed(0)
    alpha_1 = -4
    alpha_2 = -2
    constant = 100
    breakpoint_1 = 7
    n_points = 200
    xx = np.linspace(0, 20, n_points)
    yy = (
        constant
        + alpha_1 * xx
        + (alpha_2 - alpha_1) * np.maximum(xx - breakpoint_1, 0)
        + np.random.normal(size=n_points)
    )

    # In our library, frequency is x and power is y. The fitting is done on
    # the log-log scale. We can simulate this by treating the generated
    # xx and yy as our log_freq and log_power.
    # To use our fitter, we need to convert back to linear scale first.
    frequency = np.exp(xx)
    power = np.exp(yy)

    # 2. Fit the data using our segmented spectrum fitter
    # We set n_bootstraps=0 because we are only interested in the core estimates
    results = fit_segmented_spectrum(
        frequency, power, n_breakpoints=1, n_bootstraps=0
    )

    # 3. Assert that the results match the known parameters
    # The known slopes (alpha) need to be converted to our beta convention (-slope)
    known_beta1 = -alpha_1
    known_beta2 = -alpha_2
    known_breakpoint = np.exp(breakpoint_1)  # Convert back to linear frequency

    assert "breakpoints" in results
    assert "betas" in results
    assert len(results["betas"]) == 2
    assert len(results["breakpoints"]) == 1

    # Use a relatively loose tolerance to account for noise in the data
    assert results["betas"][0] == pytest.approx(known_beta1, abs=0.5)
    assert results["betas"][1] == pytest.approx(known_beta2, abs=0.5)

    # The fit happens in the log-frequency domain, so we should compare the
    # breakpoint in that domain to avoid large relative errors caused by np.exp()
    estimated_log_breakpoint = np.log(results["breakpoints"][0])
    assert estimated_log_breakpoint == pytest.approx(breakpoint_1, rel=0.1)


def test_fit_segmented_spectrum_against_muggeo_data():
    """
    Validates the fitter against the test data from the original
    piecewise-regression repository to ensure consistent results.
    """
    # 1. Load the data using the same method as the reference library's tests
    from importlib.machinery import SourceFileLoader
    data_loader = SourceFileLoader('data', 'tests/data/data.txt')
    data = data_loader.load_module()

    log_freq = np.array(data.MUGGEO_2_XX)
    log_power = np.array(data.MUGGEO_2_YY)

    # Our fitter expects data in the linear domain, so we convert it
    frequency = np.exp(log_freq)
    power = np.exp(log_power)

    # 2. Fit the data with our function
    results = fit_segmented_spectrum(
        frequency, power, n_breakpoints=1, n_bootstraps=0
    )

    # 3. Define the expected results from the reference test `test_fit.py`
    # The reference test asserts the following values (rounded):
    # - alpha1 (slope 1) = 0.94
    # - beta1 (change in slope) = 1.96
    # - breakpoint = 1.61

    # Convert to our convention (beta = -slope)
    expected_beta1 = -0.94472
    # The second slope is alpha1 + beta1
    expected_beta2 = -(0.94472 + 1.95719)
    expected_log_breakpoint = 1.608

    # 4. Assert that our results match the expected values
    assert "breakpoints" in results
    assert "betas" in results
    assert len(results["betas"]) == 2
    assert len(results["breakpoints"]) == 1

    estimated_log_breakpoint = np.log(results["breakpoints"][0])

    # Use a tolerance that aligns with the reference test's assertions
    assert results["betas"][0] == pytest.approx(expected_beta1, abs=0.1)
    assert results["betas"][1] == pytest.approx(expected_beta2, abs=0.1)
    assert estimated_log_breakpoint == pytest.approx(expected_log_breakpoint, abs=0.1)
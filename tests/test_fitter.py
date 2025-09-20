import numpy as np
import pytest
from waterSpec.fitter import fit_spectrum, fit_spectrum_with_bootstrap, fit_segmented_spectrum

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

def test_fit_spectrum_returns_correct_beta(synthetic_spectrum):
    """
    Test that fit_spectrum correctly estimates the spectral exponent (beta).
    """
    frequency, power, known_beta = synthetic_spectrum

    # Fit the spectrum
    fit_results = fit_spectrum(frequency, power)

    # Check that the returned beta is close to the known beta
    assert 'beta' in fit_results
    assert fit_results['beta'] == pytest.approx(known_beta, abs=0.1) # Allow some tolerance due to noise

def test_fit_spectrum_returns_good_fit_metrics(synthetic_spectrum):
    """
    Test that fit_spectrum returns a good R-squared value for a clean signal
    when using the OLS method.
    """
    frequency, power, _ = synthetic_spectrum

    # Fit the spectrum using OLS specifically for this test
    fit_results = fit_spectrum(frequency, power, method='ols')

    # Check that the R-squared value indicates a good fit
    assert 'r_squared' in fit_results
    assert fit_results['r_squared'] > 0.95

def test_fit_spectrum_theil_sen(synthetic_spectrum):
    """
    Test that fit_spectrum with method='theil-sen' correctly estimates beta.
    """
    frequency, power, known_beta = synthetic_spectrum

    # Fit the spectrum using the Theil-Sen estimator
    fit_results = fit_spectrum(frequency, power, method='theil-sen')

    # Check that the returned beta is close to the known beta
    assert 'beta' in fit_results
    assert fit_results['beta'] == pytest.approx(known_beta, abs=0.1)

    # Check that the other metrics are NaN as expected
    assert np.isnan(fit_results['r_squared'])
    assert np.isnan(fit_results['stderr'])

def test_fit_spectrum_with_bootstrap(synthetic_spectrum):
    """
    Test that fit_spectrum_with_bootstrap returns a confidence interval for beta.
    """
    # Set a seed for reproducibility of the bootstrap
    np.random.seed(42)

    frequency, power, known_beta = synthetic_spectrum

    # Fit the spectrum with bootstrap
    fit_results = fit_spectrum_with_bootstrap(frequency, power, n_bootstraps=100) # Use a small number for testing

    # Check that the results dictionary contains the required keys
    assert 'beta' in fit_results
    assert 'beta_ci_lower' in fit_results
    assert 'beta_ci_upper' in fit_results

    # Check that the main beta estimate is within the confidence interval
    assert fit_results['beta_ci_lower'] <= fit_results['beta'] <= fit_results['beta_ci_upper']

    # Check that the known beta is within the confidence interval (it should be, most of the time)
    assert (fit_results['beta_ci_lower'] - 0.01) <= known_beta <= (fit_results['beta_ci_upper'] + 0.01)

    # Check that the confidence interval is not excessively wide
    assert (fit_results['beta_ci_upper'] - fit_results['beta_ci_lower']) < 1.0

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
    power[mask1] = (frequency[mask1] ** -beta1)

    # Second segment
    mask2 = frequency >= breakpoint_freq
    # We need to scale the second segment to connect smoothly at the breakpoint
    scale_factor = (breakpoint_freq ** -beta1) / (breakpoint_freq ** -beta2)
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
    assert 'breakpoint' in results
    assert 'beta1' in results
    assert 'beta2' in results

    # Check that the identified breakpoint is close to the known breakpoint
    # The breakpoint is on a log scale, so we check the log values
    assert np.log10(results['breakpoint']) == pytest.approx(np.log10(known_breakpoint), abs=0.5)

    # Check that the estimated betas are close to the known betas
    assert results['beta1'] == pytest.approx(known_beta1, abs=0.3)
    assert results['beta2'] == pytest.approx(known_beta2, abs=0.3)

# --- Edge Case Tests ---

def test_fit_spectrum_white_noise():
    """Test fitting a flat spectrum (white noise), where beta should be ~0."""
    frequency = np.linspace(0.01, 1, 100)
    # Power is constant for white noise, with some random variation
    rng = np.random.default_rng(42)
    power = np.ones_like(frequency) + rng.normal(0, 0.1, len(frequency))

    fit_results = fit_spectrum(frequency, power)
    assert fit_results['beta'] == pytest.approx(0.0, abs=0.1)

def test_fit_spectrum_bootstrap_insufficient_data():
    """Test bootstrap function when the initial fit fails."""
    # Not enough data points to perform a fit
    frequency = np.array([1.0])
    power = np.array([1.0])

    results = fit_spectrum_with_bootstrap(frequency, power)

    # Check that all results are NaN
    assert np.isnan(results['beta'])
    assert np.isnan(results['r_squared'])
    assert np.isnan(results['beta_ci_lower'])
    assert np.isnan(results['beta_ci_upper'])

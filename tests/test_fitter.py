import numpy as np
import pytest
from waterSpec.fitter import fit_spectrum

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
    Test that fit_spectrum returns a good R-squared value for a clean signal.
    """
    frequency, power, _ = synthetic_spectrum

    # Fit the spectrum
    fit_results = fit_spectrum(frequency, power)

    # Check that the R-squared value indicates a good fit
    assert 'r_squared' in fit_results
    assert fit_results['r_squared'] > 0.95

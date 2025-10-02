import numpy as np
import pytest

from waterSpec.spectral_analyzer import calculate_periodogram, find_significant_peaks


@pytest.fixture
def synthetic_signal():
    """
    Generates a synthetic time series with a known frequency on an irregular time grid.
    """
    # Define signal parameters
    known_frequency = 0.1  # cycles/day
    n_points = 200
    duration = 100  # days

    # Create an irregular time array
    rng = np.random.default_rng(42)
    time = rng.uniform(0, duration, n_points)
    time = np.sort(time)

    # Create the sine wave signal with some noise
    y = np.sin(2 * np.pi * known_frequency * time) + rng.normal(0, 0.1, n_points)

    return time, y, known_frequency


def test_calculate_periodogram_finds_peak_frequency(synthetic_signal):
    """
    Test that calculate_periodogram (using autopower) correctly identifies
    the peak frequency of a synthetic signal.
    """
    time, y, known_frequency = synthetic_signal

    # Calculate the periodogram using autopower.
    # Use a dense grid to ensure the peak is found accurately.
    frequency, power, _ = calculate_periodogram(
        time, y, samples_per_peak=10, nyquist_factor=2
    )

    # Find the frequency with the maximum power
    peak_frequency = frequency[np.argmax(power)]

    # Assert that the found frequency is close to the known frequency
    assert peak_frequency == pytest.approx(known_frequency, abs=0.01)


def test_calculate_periodogram_with_dy(synthetic_signal):
    """
    Test that calculate_periodogram runs without error when `dy` is provided
    and that the output shapes are consistent.
    """
    time, y, _ = synthetic_signal
    # Create a dummy error array
    dy = np.ones_like(y) * 0.1

    # Calculate the periodogram. The main assertion is that this runs without crashing.
    frequency, power, ls_obj = calculate_periodogram(time, y, dy=dy)

    # Check that the output shapes are correct and consistent
    assert frequency.ndim == 1
    assert power.ndim == 1
    assert frequency.shape == power.shape
    assert not np.isnan(power).any()


def test_find_significant_peaks(synthetic_signal):
    """
    Test the find_significant_peaks function with a strong signal.
    """
    time, y, known_frequency = synthetic_signal

    # First, calculate the periodogram to get the ls object and power.
    # Use a dense grid to ensure the peak is found accurately.
    frequency, power, ls_obj = calculate_periodogram(
        time, y, samples_per_peak=10, nyquist_factor=2
    )

    # Find peaks with a reasonable FAP threshold
    peaks, fap_level = find_significant_peaks(
        ls_obj, frequency, power, fap_threshold=0.05, fap_method="baluev"
    )

    assert isinstance(peaks, list)
    # Check that fap_level is a scalar number
    assert isinstance(float(fap_level), float)
    # With the strong synthetic signal, we should find at least one peak
    assert len(peaks) > 0
    # The most significant peak should be our known frequency
    assert peaks[0]["frequency"] == pytest.approx(known_frequency, abs=0.01)
    # The FAP of the peak should be very low
    assert peaks[0]["fap"] < 1e-5


# --- Tests for the find_peaks_via_residuals function ---


@pytest.fixture
def sample_fit_results():
    """
    Creates a sample fit_results dictionary with one clear peak in the residuals.
    This uses log10, consistent with the fitter module.
    """
    log_freq = np.linspace(np.log10(1e-4), np.log10(0.5), 100)
    fitted_log_power = -1.5 * log_freq + 2
    rng = np.random.default_rng(0)

    # Create residuals with some background noise and one clear peak
    residuals = rng.normal(loc=0, scale=0.1, size=100)
    residuals[50] = 3.0  # This is the only point that should be significant

    log_power = fitted_log_power + residuals
    return {
        "log_freq": log_freq,
        "log_power": log_power,
        "residuals": residuals,
        "fitted_log_power": fitted_log_power,
    }


def test_find_peaks_via_residuals_finds_peak(sample_fit_results):
    """Test that the residual method finds the known injected peak."""
    from waterSpec.spectral_analyzer import find_peaks_via_residuals

    # With a reasonable FDR level, the strong peak with residual 3.0 should be found.
    peaks, threshold = find_peaks_via_residuals(sample_fit_results, fdr_level=0.05)

    assert len(peaks) == 1
    assert peaks[0]["residual"] == pytest.approx(3.0)
    # Check that the returned frequency matches the one at index 50
    expected_freq = 10 ** sample_fit_results["log_freq"][50]
    assert peaks[0]["frequency"] == pytest.approx(expected_freq)
    # The threshold should be the value of the smallest significant peak's residual
    assert threshold == pytest.approx(3.0)


def test_find_peaks_via_residuals_no_significant_peak(sample_fit_results):
    """
    Test that the method returns an empty list if no residual crosses the
    threshold, using data with no significant outliers.
    """
    from waterSpec.spectral_analyzer import find_peaks_via_residuals

    # Overwrite the residuals with a sample that has no extreme outliers.
    rng = np.random.default_rng(0)
    sample_fit_results["residuals"] = rng.normal(loc=0, scale=0.1, size=100)

    # With a very stringent FDR level, no peaks should be found.
    peaks, threshold = find_peaks_via_residuals(sample_fit_results, fdr_level=1e-6)

    assert len(peaks) == 0
    # The threshold should be inf when no peaks are found
    assert threshold == np.inf


def test_find_peaks_via_residuals_raises_error_on_missing_keys():
    """Test that the function raises a ValueError if required keys are missing."""
    from waterSpec.spectral_analyzer import find_peaks_via_residuals

    with pytest.raises(ValueError, match="fit_results is missing required keys"):
        find_peaks_via_residuals({"log_freq": [1, 2, 3]})
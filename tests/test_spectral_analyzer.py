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
    Test that calculate_periodogram correctly identifies the peak frequency
    of a synthetic signal when a grid is provided.
    """
    time, y, known_frequency = synthetic_signal

    # Generate a frequency grid for the test. The function now requires it.
    # The grid must span the known frequency.
    duration = np.max(time) - np.min(time)
    test_frequency = np.linspace(1/duration, 0.5 / np.median(np.diff(time)), 500)

    # Calculate the periodogram
    frequency, power, _ = calculate_periodogram(time, y, frequency=test_frequency)

    # Find the frequency with the maximum power
    peak_frequency = frequency[np.argmax(power)]

    # Assert that the found frequency is close to the known frequency
    assert peak_frequency == pytest.approx(known_frequency, abs=0.01)


def test_calculate_periodogram_with_dy(synthetic_signal):
    """
    Test that calculate_periodogram runs without error when `dy` is provided.
    """
    time, y, known_frequency = synthetic_signal
    # Create a dummy error array
    dy = np.ones_like(y) * 0.1

    # Generate a frequency grid
    duration = np.max(time) - np.min(time)
    test_frequency = np.linspace(1/duration, 0.5 / np.median(np.diff(time)), 500)

    # Calculate the periodogram
    # The main assertion is that this runs without crashing.
    frequency, power, ls_obj = calculate_periodogram(time, y, frequency=test_frequency, dy=dy)

    # Check that the output shapes are correct
    assert frequency.shape == test_frequency.shape
    assert power.shape == test_frequency.shape
    assert not np.isnan(power).any()


def test_calculate_periodogram_raises_error_without_frequency():
    """
    Test that calculate_periodogram raises a ValueError if no frequency is provided.
    """
    time = np.arange(10)
    y = np.sin(time)
    with pytest.raises(ValueError, match="A frequency grid must be provided"):
        calculate_periodogram(time, y, frequency=None)


def test_find_significant_peaks(synthetic_signal):
    """
    Test the find_significant_peaks function with a strong signal.
    """
    time, y, known_frequency = synthetic_signal

    # Generate a frequency grid
    duration = np.max(time) - np.min(time)
    frequency = np.linspace(1/duration, 0.5 / np.median(np.diff(time)), 1000)

    # First, calculate the periodogram to get the ls object and power
    frequency, power, ls_obj = calculate_periodogram(time, y, frequency=frequency)

    # Find peaks with a reasonable FAP threshold
    # Note: bootstrap can be slow, so we may want to use a faster method for CI/CD tests
    peaks, fap_level = find_significant_peaks(
        ls_obj, frequency, power, fap_threshold=0.05, fap_method='baluev'
    )

    assert isinstance(peaks, list)
    # Check that fap_level is a scalar number
    assert isinstance(float(fap_level), float)
    # With the strong synthetic signal, we should find at least one peak
    assert len(peaks) > 0
    # The most significant peak should be our known frequency
    assert peaks[0]['frequency'] == pytest.approx(known_frequency, abs=0.01)
    # The FAP of the peak should be very low
    assert peaks[0]['fap'] < 1e-5


# --- Tests for the new find_peaks_via_residuals function ---

@pytest.fixture
def sample_fit_results():
    """
    Creates a deterministic sample fit_results dictionary for testing.
    """
    log_freq = np.linspace(np.log(1e-4), np.log(0.5), 100)
    fitted_log_power = -1.5 * log_freq + 2

    # Create a deterministic set of residuals with one clear peak
    residuals = np.full(100, 0.1)
    residuals[50] = 3.0 # This is the only point that should be significant

    log_power = fitted_log_power + residuals
    return {
        'log_freq': log_freq,
        'log_power': log_power,
        'residuals': residuals,
        'fitted_log_power': fitted_log_power
    }

def test_find_peaks_via_residuals_finds_peak(sample_fit_results):
    """Test that the residual method finds the known injected peak."""
    from waterSpec.spectral_analyzer import find_peaks_via_residuals
    # With ci=95, the 95th percentile of our deterministic residuals will be 0.1,
    # so the peak with residual 3.0 should be found.
    peaks, threshold = find_peaks_via_residuals(sample_fit_results, ci=95)

    assert len(peaks) == 1
    assert peaks[0]['residual'] == pytest.approx(3.0)
    # Check that the returned frequency matches the one at index 50
    expected_freq = np.exp(sample_fit_results['log_freq'][50])
    assert peaks[0]['frequency'] == pytest.approx(expected_freq)

def test_find_peaks_via_residuals_no_significant_peak(sample_fit_results):
    """Test that the method returns an empty list if no residual crosses the threshold."""
    from waterSpec.spectral_analyzer import find_peaks_via_residuals
    # With our deterministic data, ci=99.9 will set a threshold > 0.1 but < 3.0.
    # The test is to see if we can set a threshold so high that nothing is found.
    # With ci=100, the threshold will be 3.0, and `>` comparison will fail.
    peaks, threshold = find_peaks_via_residuals(sample_fit_results, ci=100)
    assert len(peaks) == 0

def test_find_peaks_via_residuals_raises_error_on_missing_keys():
    """Test that the function raises a ValueError if required keys are missing."""
    from waterSpec.spectral_analyzer import find_peaks_via_residuals
    with pytest.raises(ValueError, match="must contain 'residuals'"):
        find_peaks_via_residuals({'log_freq': [1,2,3]})

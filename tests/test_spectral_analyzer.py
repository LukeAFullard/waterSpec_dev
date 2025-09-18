import numpy as np
import pytest
from waterSpec.spectral_analyzer import calculate_periodogram

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
    of a synthetic signal.
    """
    time, y, known_frequency = synthetic_signal

    # Calculate the periodogram
    frequency, power = calculate_periodogram(time, y)

    # Find the frequency with the maximum power
    peak_frequency = frequency[np.argmax(power)]

    # Assert that the found frequency is close to the known frequency
    assert peak_frequency == pytest.approx(known_frequency, abs=0.01)

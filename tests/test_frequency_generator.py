import numpy as np
import pytest

from waterSpec.frequency_generator import generate_frequency_grid


@pytest.fixture
def sample_time_array():
    """Provides a sample time array for testing."""
    return np.linspace(0, 100, 200)


def test_generate_frequency_grid_linear(sample_time_array):
    """
    Test that the grid generated with grid_type='linear' is linearly spaced.
    """
    linear_grid = generate_frequency_grid(sample_time_array, grid_type="linear")

    # The differences between consecutive elements should be constant
    diffs = np.diff(linear_grid)
    assert np.allclose(diffs, diffs[0])


def test_generate_frequency_grid_log(sample_time_array):
    """
    Test that the grid generated with grid_type='log' is logarithmically spaced.
    """
    log_grid = generate_frequency_grid(sample_time_array, grid_type="log")

    # The differences between the logs of consecutive elements should be constant
    log_diffs = np.diff(np.log10(log_grid))
    assert np.allclose(log_diffs, log_diffs[0])


def test_generate_frequency_grid_default_is_log(sample_time_array):
    """
    Test that the default grid type is 'log'.
    """
    default_grid = generate_frequency_grid(sample_time_array)
    log_grid = generate_frequency_grid(sample_time_array, grid_type="log")

    # The default grid should be identical to a log-generated grid
    assert np.array_equal(default_grid, log_grid)


def test_generate_frequency_grid_invalid_type(sample_time_array):
    """
    Test that an invalid grid_type raises a ValueError.
    """
    with pytest.raises(ValueError, match="grid_type must be either 'log' or 'linear'"):
        generate_frequency_grid(sample_time_array, grid_type="invalid_type")


def test_generate_frequency_grid_zero_duration():
    """
    Test that a time series with zero duration raises a ValueError.
    """
    time = np.ones(10)  # All timestamps are the same
    with pytest.raises(
        ValueError, match="Time series duration must be positive"
    ):
        generate_frequency_grid(time)


def test_generate_frequency_grid_min_freq_adjustment():
    """
    Test the edge case where min_freq >= nyquist_freq, which should trigger
    an internal adjustment and still produce a valid grid.
    """
    # A short time series with a large gap can cause this condition.
    # duration = 1001, min_freq = 1/1001
    # avg_sampling = (1000 + 1)/2 = 500.5, nyquist = 0.5/500.5 = 1/1001
    # This makes min_freq == nyquist_freq, triggering the adjustment.
    time = np.array([0, 1000, 1001])
    # The test is that this runs without error.
    freq_grid = generate_frequency_grid(time)
    assert len(freq_grid) > 0
    assert np.all(np.isfinite(freq_grid))

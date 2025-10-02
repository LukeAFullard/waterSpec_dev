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


def test_nyquist_frequency_is_robust_to_outliers():
    """
    Tests that the Nyquist frequency calculation uses the median sampling
    interval, making it robust to large data gaps (outliers).
    """
    # Create a time series with mostly regular sampling (1-day intervals)
    # but with one large gap of 100 days.
    regular_time = np.arange(10)  # 10 points with 1-day interval
    gap = np.array([110])  # A single point after a 100-day gap
    irregular_time = np.concatenate([regular_time, gap])

    # The median interval should be 1.0.
    median_interval = np.median(np.diff(irregular_time))
    assert median_interval == 1.0
    expected_nyquist = 0.5 / median_interval  # Should be 0.5

    # The mean interval would be skewed by the large gap.
    mean_interval = np.mean(np.diff(irregular_time))
    incorrect_nyquist = 0.5 / mean_interval  # Would be much lower

    assert expected_nyquist != incorrect_nyquist

    # Generate the frequency grid
    grid = generate_frequency_grid(irregular_time)

    # The maximum frequency in the grid should be based on the median, not the mean.
    assert np.max(grid) == pytest.approx(expected_nyquist)


def test_max_freq_override(sample_time_array):
    """Test that `max_freq` overrides the automatic Nyquist calculation."""
    custom_max_freq = 0.75
    grid = generate_frequency_grid(sample_time_array, max_freq=custom_max_freq)
    assert np.max(grid) == pytest.approx(custom_max_freq)


def test_nyquist_factor(sample_time_array):
    """Test that `nyquist_factor` correctly scales the Nyquist frequency."""
    # Default Nyquist for the sample array
    median_interval = np.median(np.diff(sample_time_array))
    expected_nyquist = 0.5 / median_interval

    # With a factor of 0.5, the max frequency should be half
    grid = generate_frequency_grid(sample_time_array, nyquist_factor=0.5)
    assert np.max(grid) == pytest.approx(0.5 * expected_nyquist)


def test_time_unit_in_warning_message():
    """
    Test that the `time_unit` is correctly reflected in the warning for
    highly irregular sampling.
    """
    # Create a time series with highly irregular sampling
    irregular_time = np.array([0, 1, 10, 11, 100])
    with pytest.warns(UserWarning, match="The output frequency units are 1/days") as record:
        generate_frequency_grid(irregular_time, time_unit="days")

    # Check that a warning was indeed issued
    assert len(record) == 1


def test_invalid_inputs(sample_time_array):
    """Test that invalid inputs for new parameters raise ValueError."""
    with pytest.raises(ValueError, match="nyquist_factor must be a positive number"):
        generate_frequency_grid(sample_time_array, nyquist_factor=0)

    with pytest.raises(ValueError, match="nyquist_factor must be a positive number"):
        generate_frequency_grid(sample_time_array, nyquist_factor=-1.0)

    with pytest.raises(ValueError, match="max_freq, if provided, must be a positive number"):
        generate_frequency_grid(sample_time_array, max_freq=0)

    with pytest.raises(ValueError, match="max_freq, if provided, must be a positive number"):
        generate_frequency_grid(sample_time_array, max_freq=-10.0)

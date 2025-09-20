import pytest
import numpy as np
from waterSpec.frequency_generator import generate_frequency_grid

@pytest.fixture
def sample_time_array():
    """Provides a sample time array for testing."""
    return np.linspace(0, 100, 200)

def test_generate_frequency_grid_linear(sample_time_array):
    """
    Test that the grid generated with grid_type='linear' is linearly spaced.
    """
    linear_grid = generate_frequency_grid(sample_time_array, grid_type='linear')

    # The differences between consecutive elements should be constant
    diffs = np.diff(linear_grid)
    assert np.allclose(diffs, diffs[0])

def test_generate_frequency_grid_log(sample_time_array):
    """
    Test that the grid generated with grid_type='log' is logarithmically spaced.
    """
    log_grid = generate_frequency_grid(sample_time_array, grid_type='log')

    # The differences between the logs of consecutive elements should be constant
    log_diffs = np.diff(np.log10(log_grid))
    assert np.allclose(log_diffs, log_diffs[0])

def test_generate_frequency_grid_default_is_log(sample_time_array):
    """
    Test that the default grid type is 'log'.
    """
    default_grid = generate_frequency_grid(sample_time_array)
    log_grid = generate_frequency_grid(sample_time_array, grid_type='log')

    # The default grid should be identical to a log-generated grid
    assert np.array_equal(default_grid, log_grid)

def test_generate_frequency_grid_invalid_type(sample_time_array):
    """
    Test that an invalid grid_type raises a ValueError.
    """
    with pytest.raises(ValueError, match="grid_type must be either 'log' or 'linear'"):
        generate_frequency_grid(sample_time_array, grid_type='invalid_type')

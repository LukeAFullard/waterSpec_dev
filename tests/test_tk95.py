import numpy as np
import pytest
from waterSpec.utils_sim.tk95 import resample_to_times

def test_resample_to_times_basic():
    """Test basic interpolation behavior with exact matches and midpoints."""
    source_time = np.array([0.0, 1.0, 2.0, 3.0])
    source_flux = np.array([0.0, 10.0, 20.0, 30.0])

    # Target times matching source times
    target_time_exact = np.array([0.0, 1.0, 2.0, 3.0])
    result_exact = resample_to_times(source_time, source_flux, target_time_exact)
    np.testing.assert_array_almost_equal(result_exact, source_flux)

    # Target times halfway between source times
    target_time_interp = np.array([0.5, 1.5, 2.5])
    expected_interp = np.array([5.0, 15.0, 25.0])
    result_interp = resample_to_times(source_time, source_flux, target_time_interp)
    np.testing.assert_array_almost_equal(result_interp, expected_interp)

def test_resample_to_times_extrapolation():
    """Test extrapolation behavior.
    np.interp normally repeats edge values for values outside the source range.
    """
    source_time = np.array([0.0, 1.0, 2.0])
    source_flux = np.array([10.0, 20.0, 30.0])

    # Target times outside the source time range
    target_time_extrap = np.array([-1.0, 3.0])
    expected_extrap = np.array([10.0, 30.0])  # Edge values repeated
    result_extrap = resample_to_times(source_time, source_flux, target_time_extrap)
    np.testing.assert_array_almost_equal(result_extrap, expected_extrap)

def test_resample_to_times_single_value():
    """Test interpolation to a single time point."""
    source_time = np.array([0.0, 1.0, 2.0])
    source_flux = np.array([10.0, 20.0, 30.0])

    target_time_single = np.array([1.25])
    expected_single = np.array([22.5])
    result_single = resample_to_times(source_time, source_flux, target_time_single)
    np.testing.assert_array_almost_equal(result_single, expected_single)

def test_resample_to_times_unordered_target():
    """Test interpolation with unordered target times."""
    source_time = np.array([0.0, 1.0, 2.0])
    source_flux = np.array([10.0, 20.0, 30.0])

    target_time_unordered = np.array([1.5, 0.5])
    expected_unordered = np.array([25.0, 15.0])
    result_unordered = resample_to_times(source_time, source_flux, target_time_unordered)
    np.testing.assert_array_almost_equal(result_unordered, expected_unordered)

def test_resample_to_times_empty_target():
    """Test interpolation with empty target array returns empty array."""
    source_time = np.array([0.0, 1.0, 2.0])
    source_flux = np.array([10.0, 20.0, 30.0])

    target_time_empty = np.array([])
    result_empty = resample_to_times(source_time, source_flux, target_time_empty)
    assert len(result_empty) == 0
    assert result_empty.shape == (0,)

def test_resample_to_times_2d_raises():
    """Test that passing a 2D source_flux raises ValueError from np.interp."""
    source_time = np.array([0.0, 1.0, 2.0])
    # simulate_tk95 can return 2D array if n_simulations is set,
    # but resample_to_times expects 1D per numpy.interp behavior.
    source_flux_2d = np.array([[10.0, 20.0, 30.0], [40.0, 50.0, 60.0]])
    target_time = np.array([0.5, 1.5])

    with pytest.raises(ValueError):
        resample_to_times(source_time, source_flux_2d, target_time)

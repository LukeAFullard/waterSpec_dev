import numpy as np
import pytest
import pandas as pd
from waterSpec.preprocessor import detrend, normalize, log_transform, handle_censored_data, detrend_loess

# Sample data for testing
@pytest.fixture
def sample_data():
    """Provides a sample numpy array for testing."""
    return np.array([1.0, 2.0, 3.0, 4.0, 5.0])

@pytest.fixture
def trending_data():
    """Provides a sample numpy array with a linear trend."""
    # A simple linear trend with some noise
    x = np.arange(10)
    y = 2 * x + 1 + np.random.randn(10) * 0.1
    return y

def test_detrend(trending_data):
    """Test the detrend function."""
    detrended_data = detrend(trending_data)
    # A simple check: the mean of the detrended data should be close to zero.
    assert np.mean(detrended_data) == pytest.approx(0.0, abs=1e-9)
    # Check that it returns an array of the same shape
    assert detrended_data.shape == trending_data.shape

def test_normalize(sample_data):
    """Test the normalize function."""
    normalized_data = normalize(sample_data)
    # The mean of normalized data should be close to 0
    assert np.mean(normalized_data) == pytest.approx(0.0, abs=1e-9)
    # The standard deviation of normalized data should be close to 1
    assert np.std(normalized_data) == pytest.approx(1.0, abs=1e-9)

def test_log_transform(sample_data):
    """Test the log_transform function."""
    transformed_data = log_transform(sample_data)
    expected_data = np.log(sample_data)
    np.testing.assert_array_almost_equal(transformed_data, expected_data)

def test_log_transform_with_zero():
    """Test log_transform with data containing zero, which should raise an error."""
    data_with_zero = np.array([0.0, 1.0, 2.0])
    with pytest.raises(ValueError, match="log-transform requires all data to be positive"):
        log_transform(data_with_zero)

def test_log_transform_with_negative():
    """Test log_transform with negative data, which should raise an error."""
    data_with_negative = np.array([-1.0, 1.0, 2.0])
    with pytest.raises(ValueError, match="log-transform requires all data to be positive"):
        log_transform(data_with_negative)

@pytest.fixture
def censored_data_series():
    """Provides a pandas Series with censored data."""
    return pd.Series(['10.1', '<5.0', '10.3', '>100', '11.0'])

def test_handle_censored_data_use_detection_limit_strategy(censored_data_series):
    """Test the 'use_detection_limit' strategy for censored data."""
    result = handle_censored_data(censored_data_series, strategy='use_detection_limit')
    expected = np.array([10.1, 5.0, 10.3, 100.0, 11.0])
    np.testing.assert_array_almost_equal(result, expected)

def test_handle_censored_data_drop_strategy(censored_data_series):
    """Test the default 'drop' strategy for censored data."""
    # Default strategy is 'drop'
    result = handle_censored_data(censored_data_series)
    expected = np.array([10.1, np.nan, 10.3, np.nan, 11.0])
    # Use assert_equal for arrays with NaNs
    np.testing.assert_equal(result, expected)

def test_handle_censored_data_multiplier_strategy(censored_data_series):
    """Test the 'multiplier' strategy for censored data."""
    result = handle_censored_data(
        censored_data_series,
        strategy='multiplier',
        lower_multiplier=0.5,
        upper_multiplier=1.1
    )
    expected = np.array([10.1, 2.5, 10.3, 110.0, 11.0])
    np.testing.assert_array_almost_equal(result, expected)

def test_handle_censored_data_invalid_strategy(censored_data_series):
    """Test that an invalid strategy raises an error."""
    with pytest.raises(ValueError, match="Invalid strategy. Choose from \\['drop', 'use_detection_limit', 'multiplier'\\]"):
        handle_censored_data(censored_data_series, strategy='invalid_strategy')

@pytest.fixture
def nonlinear_data():
    """Provides a sample numpy array with a non-linear trend."""
    x = np.linspace(0, 10, 100)
    # Quadratic trend + some noise
    trend = 0.1 * x**2 - 0.5 * x
    y = trend + np.random.randn(100) * 0.1
    return x, y

def test_detrend_loess(nonlinear_data):
    """Test the LOESS detrending function."""
    x, y = nonlinear_data
    detrended_y = detrend_loess(x, y)

    # The mean of the residuals should be close to zero
    assert np.mean(detrended_y) == pytest.approx(0.0, abs=1e-1)
    # Check that it returns an array of the same shape
    assert detrended_y.shape == y.shape

# --- Edge Case Tests ---

def test_handle_censored_data_no_censoring():
    """Test handle_censored_data with a series containing no censored values."""
    data = pd.Series(['1.0', '2.0', '3.0'])
    result = handle_censored_data(data, strategy='use_detection_limit')
    expected = np.array([1.0, 2.0, 3.0])
    np.testing.assert_array_almost_equal(result, expected)

def test_handle_censored_data_all_censored():
    """Test handle_censored_data with a series of only censored values."""
    data = pd.Series(['<1.0', '>2.0', '<3.0'])
    result = handle_censored_data(data, strategy='drop')
    assert np.all(np.isnan(result))

def test_detrend_short_series():
    """Test detrending a very short series (should not fail)."""
    data = np.array([10.0, 10.5])
    result = detrend(data)
    assert result.shape == data.shape
    assert np.mean(result) == pytest.approx(0.0)

def test_normalize_short_series():
    """Test normalizing a very short series."""
    data = np.array([10.0, 20.0])
    result = normalize(data)
    assert np.mean(result) == pytest.approx(0.0)
    assert np.std(result) == pytest.approx(1.0)

def test_normalize_constant_series():
    """Test normalizing a constant series (should return zeros)."""
    data = np.array([5.0, 5.0, 5.0])
    result = normalize(data)
    expected = np.array([0.0, 0.0, 0.0])
    np.testing.assert_array_almost_equal(result, expected)

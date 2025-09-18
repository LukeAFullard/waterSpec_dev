import numpy as np
import pytest
from waterSpec.preprocessor import detrend, normalize, log_transform

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

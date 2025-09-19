import numpy as np
import pandas as pd
import pytest
from waterSpec.preprocessor import (
    detrend,
    normalize,
    log_transform,
    handle_censored_data,
    detrend_loess,
    preprocess_data,
    _validate_data_length
)

# A dataset with more than 10 points for validation checks
@pytest.fixture
def sample_data():
    return np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0])

def test_detrend(sample_data):
    """Test the detrend function."""
    trended_data = sample_data
    detrended = detrend(trended_data.copy())
    assert np.mean(detrended) == pytest.approx(0)
    assert not np.array_equal(detrended, trended_data)

def test_normalize(sample_data):
    """Test the normalize function."""
    normalized_data = normalize(sample_data.copy())
    assert np.mean(normalized_data) == pytest.approx(0)
    assert np.std(normalized_data) == pytest.approx(1)

def test_log_transform(sample_data):
    """Test the log_transform function with valid data."""
    transformed_data = log_transform(sample_data.copy())
    expected_data = np.log(sample_data)
    np.testing.assert_array_almost_equal(transformed_data, expected_data)

def test_log_transform_non_positive():
    """Test that log_transform raises ValueError for non-positive data."""
    with pytest.raises(ValueError, match="log-transform requires all data to be positive"):
        log_transform(np.array([1.0, 2.0, 0.0, 4.0, 5.0]))
    with pytest.raises(ValueError, match="log-transform requires all data to be positive"):
        log_transform(np.array([1.0, -2.0, 3.0, 4.0, 5.0]))

def test_handle_censored_data():
    """Test the handle_censored_data function."""
    data_series = pd.Series(["1", "<2", "3", ">4", "5"])
    result = handle_censored_data(data_series, strategy='drop')
    assert np.isnan(result[1])
    assert np.isnan(result[3])
    assert result[0] == 1

    result = handle_censored_data(data_series, strategy='use_detection_limit')
    assert result[1] == 2.0
    assert result[3] == 4.0

    result = handle_censored_data(data_series, strategy='multiplier', lower_multiplier=0.5, upper_multiplier=1.1)
    assert result[1] == 1.0
    assert result[3] == 4.4

def test_detrend_loess(sample_data):
    """Test the LOESS detrending function."""
    time = np.arange(len(sample_data))
    trended_data = sample_data + np.sin(time)
    detrended = detrend_loess(time, trended_data)
    assert np.var(detrended) < np.var(trended_data)

def test_detrend_loess_with_options(sample_data):
    """Test that LOESS detrending accepts and uses custom options."""
    time = np.arange(len(sample_data))
    # Add an outlier to make the robustness iterations have an effect
    trended_data = sample_data + np.sin(time)
    trended_data[5] = 20

    # Detrend with default options (which include robustness iterations)
    detrended_default = detrend_loess(time, trended_data.copy())

    # Detrend with robustness iterations turned off
    detrended_no_iter = detrend_loess(time, trended_data.copy(), it=0)

    # The results should be different because the outlier is handled differently
    assert not np.array_equal(detrended_default, detrended_no_iter)

def test_validate_data_length():
    """Test the data length validation function."""
    _validate_data_length(np.random.rand(20), min_length=10)
    with pytest.raises(ValueError, match="has only 5 valid data points"):
        _validate_data_length(np.random.rand(5), min_length=10)

def test_preprocess_data_wrapper(sample_data):
    """Test the main preprocess_data wrapper function."""
    time = np.arange(len(sample_data))
    data_series = pd.Series(sample_data)

    processed = preprocess_data(data_series, time, detrend_method='linear')
    assert np.mean(processed) == pytest.approx(0)

    with pytest.warns(UserWarning, match="Unknown detrending method"):
        preprocess_data(data_series, time, detrend_method='bad_method')

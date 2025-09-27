import numpy as np
import pandas as pd
import pytest
import statsmodels.api as sm

from waterSpec.preprocessor import (
    _validate_data_length,
    detrend,
    detrend_loess,
    handle_censored_data,
    log_transform,
    normalize,
    preprocess_data,
)


# A dataset with more than 10 points for validation checks
@pytest.fixture
def sample_data():
    return np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0])


def test_detrend(sample_data):
    """Test the detrend function."""
    trended_data = sample_data
    time = np.arange(len(trended_data))
    detrended, _ = detrend(time, trended_data.copy())
    assert np.mean(detrended) == pytest.approx(0)
    # For a linear trend, the detrended data should not be the same
    assert not np.allclose(detrended, trended_data)


def test_normalize(sample_data):
    """Test the normalize function."""
    normalized_data, errors = normalize(sample_data.copy())
    assert errors is None
    assert np.mean(normalized_data) == pytest.approx(0)
    assert np.std(normalized_data) == pytest.approx(1)


def test_log_transform(sample_data):
    """Test the log_transform function with valid data."""
    transformed_data, errors = log_transform(sample_data.copy())
    assert errors is None
    expected_data = np.log(sample_data)
    np.testing.assert_array_almost_equal(transformed_data, expected_data)


def test_log_transform_non_positive():
    """Test that log_transform raises ValueError for non-positive data."""
    with pytest.raises(
        ValueError, match="Log-transform requires all data to be positive, but 1 non-positive value"
    ):
        log_transform(np.array([1.0, 2.0, 0.0, 4.0, 5.0]))
    with pytest.raises(
        ValueError, match="Log-transform requires all data to be positive, but 1 non-positive value"
    ):
        log_transform(np.array([1.0, -2.0, 3.0, 4.0, 5.0]))


def test_handle_censored_data():
    """Test the handle_censored_data function."""
    data_series = pd.Series(["1", "<2", "3", ">4", "5"])
    result = handle_censored_data(data_series, strategy="drop")
    assert np.isnan(result[1])
    assert np.isnan(result[3])
    assert result[0] == 1

    result = handle_censored_data(data_series, strategy="use_detection_limit")
    assert result[1] == 2.0
    assert result[3] == 4.0

    result = handle_censored_data(
        data_series, strategy="multiplier", lower_multiplier=0.5, upper_multiplier=1.1
    )
    assert result[1] == 1.0
    assert result[3] == 4.4


def test_detrend_loess(sample_data):
    """Test the LOESS detrending function."""
    time = np.arange(len(sample_data))
    trended_data = sample_data + np.sin(time)
    detrended, _ = detrend_loess(time, trended_data)
    assert np.var(detrended) < np.var(trended_data)


def test_detrend_loess_with_options(sample_data):
    """Test that LOESS detrending accepts and uses custom options."""
    time = np.arange(len(sample_data))
    # Add an outlier to make the robustness iterations have an effect
    trended_data = sample_data + np.sin(time)
    trended_data[5] = 20

    # Detrend with default options (which include robustness iterations)
    detrended_default, _ = detrend_loess(time, trended_data.copy())

    # Detrend with robustness iterations turned off
    detrended_no_iter, _ = detrend_loess(time, trended_data.copy(), it=0)

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

    processed, errors = preprocess_data(data_series, time, detrend_method="linear")
    assert errors is None
    assert np.mean(processed) == pytest.approx(0)

    with pytest.warns(UserWarning, match="Unknown detrending method"):
        processed, errors = preprocess_data(
            data_series, time, detrend_method="bad_method"
        )


# --- New tests for error propagation and wrapper functionality ---


def test_log_transform_with_errors(sample_data):
    """Test that log_transform correctly propagates errors."""
    data = sample_data.copy()
    errors = np.full_like(data, 0.1)

    transformed_data, transformed_errors = log_transform(data, errors)

    expected_errors = 0.1 / sample_data
    np.testing.assert_array_almost_equal(transformed_errors, expected_errors)


def test_normalize_with_errors(sample_data):
    """Test that normalize correctly propagates errors."""
    data = sample_data.copy()
    errors = np.full_like(data, 0.5)
    data_std = np.std(data)

    transformed_data, transformed_errors = normalize(data, errors)

    expected_errors = 0.5 / data_std
    np.testing.assert_array_almost_equal(transformed_errors, expected_errors)


def test_preprocess_data_with_transforms(sample_data):
    """Test the preprocess_data wrapper with all transformations enabled."""
    time = np.arange(len(sample_data))
    data_series = pd.Series(sample_data)
    error_series = pd.Series(np.full_like(sample_data, 0.1))

    # Apply all transformations
    processed_data, processed_errors = preprocess_data(
        data_series,
        time,
        error_series=error_series,
        log_transform_data=True,
        detrend_method="linear",
        normalize_data=True,
    )

    # Check the final data properties
    assert np.mean(processed_data) == pytest.approx(0)
    assert np.std(processed_data) == pytest.approx(1)

    # Check the final error propagation
    # 1. After log-transform: error' = error / data
    # 2. After normalization: error'' = error' / std(log_detrended_data)
    # This is tricky to test exactly without re-implementing the logic,
    # so we'll just check that the errors were transformed and are not None.
    assert processed_errors is not None
    assert not np.array_equal(processed_errors, error_series.to_numpy())


def test_preprocess_data_error_propagation_correctness(sample_data):
    """
    Tests the correctness of the error propagation by comparing the output of
    the wrapper function with the output of the individual functions applied
    sequentially.
    """
    time = np.arange(len(sample_data))
    data_series = pd.Series(sample_data)
    initial_errors = np.full_like(sample_data, 0.1)
    error_series = pd.Series(initial_errors)

    # --- Run the full pipeline using the wrapper ---
    _, processed_errors_from_wrapper = preprocess_data(
        data_series,
        time,
        error_series=error_series.copy(),
        log_transform_data=True,
        detrend_method="linear",
        normalize_data=True,
    )

    # --- Manually apply the steps sequentially to calculate expected errors ---
    data_manual = sample_data.copy()
    errors_manual = initial_errors.copy()

    # 1. Log transform
    data_manual, errors_manual = log_transform(data_manual, errors_manual)
    # 2. Detrend
    data_manual, errors_manual = detrend(time, data_manual, errors_manual)
    # 3. Normalize
    _, errors_manual = normalize(data_manual, errors_manual)

    # --- Compare the results ---
    np.testing.assert_array_almost_equal(
        processed_errors_from_wrapper, errors_manual
    )


def test_normalize_zero_std_dev():
    """Test normalize function when standard deviation is zero."""
    data = np.array([5.0, 5.0, 5.0, 5.0])
    errors = np.array([0.1, 0.1, 0.1, 0.1])
    normalized_data, normalized_errors = normalize(data.copy(), errors.copy())
    assert np.all(normalized_data == 0)
    # Errors should be NaN if variance is zero, as they cannot be scaled
    assert np.all(np.isnan(normalized_errors))


def test_detrend_loess_with_errors(sample_data):
    """Test that LOESS detrending correctly propagates errors."""
    time = np.arange(len(sample_data))
    # Create some non-linear data with noise
    trended_data = np.sin(time * 0.5) + time * 0.1
    errors = np.full_like(trended_data, 0.1)

    # Detrend with LOESS
    detrended_data, detrended_errors = detrend_loess(
        time, trended_data.copy(), errors=errors.copy(), frac=0.5
    )

    # The new implementation does not propagate errors for LOESS and returns
    # them unchanged, so the expected errors are the original errors.
    assert detrended_errors is not None
    np.testing.assert_array_almost_equal(detrended_errors, errors)


def test_detrend_with_nan_in_errors(sample_data):
    """Test detrend function when the error array contains NaNs."""
    trended_data = sample_data.copy()
    time = np.arange(len(trended_data))
    errors = np.full_like(trended_data, 0.1, dtype=float)
    errors[3] = np.nan

    # The warning is no longer issued; NaNs are propagated directly.
    _, detrended_errors = detrend(time, trended_data.copy(), errors=errors.copy())

    # The NaN should propagate through the calculation.
    assert np.isnan(detrended_errors[3])
    # And other values should be finite.
    assert np.all(np.isfinite(np.delete(detrended_errors, 3)))


def test_preprocess_data_nan_propagation():
    """
    Test that NaNs introduced during censoring are correctly propagated to the
    error array in the preprocess_data wrapper.
    """
    time = np.arange(12)
    # Create a series that will have 2 NaNs after processing, leaving 10
    # valid points, which is the minimum required.
    data_list = [str(i) for i in range(10)] + ["<2", "foo"]
    data_series_censored = pd.Series(data_list)
    error_series = pd.Series(np.linspace(0.1, 0.5, 12))

    with pytest.warns(UserWarning, match="Non-numeric or unhandled censored values"):
        processed_data, processed_errors = preprocess_data(
            data_series_censored,
            time,
            error_series=error_series,
            censor_strategy="drop",
        )

    # '<2' (at index 10) and 'foo' (at index 11) should become NaN
    assert np.isnan(processed_data[10])
    assert np.isnan(processed_data[11])
    # The corresponding errors should also be NaN
    assert np.isnan(processed_errors[10])
    assert np.isnan(processed_errors[11])
    # The valid points should still have their errors
    assert not np.isnan(processed_errors[0])
    assert not np.isnan(processed_errors[9])

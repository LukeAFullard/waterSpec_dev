import warnings

import numpy as np
import pandas as pd
import pytest
import statsmodels.api as sm

from waterSpec.preprocessor import (
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
    detrended, _, diagnostics = detrend(time, trended_data.copy())
    assert np.mean(detrended) == pytest.approx(0)
    # For a linear trend, the detrended data should not be the same
    assert not np.allclose(detrended, trended_data)
    assert diagnostics["r_squared_of_trend"] > 0.99


def test_normalize(sample_data):
    """Test the normalize function."""
    normalized_data, errors = normalize(sample_data.copy())
    assert errors is None
    assert np.mean(normalized_data) == pytest.approx(0)
    assert np.std(normalized_data, ddof=1) == pytest.approx(1)


def test_log_transform(sample_data):
    """Test the log_transform function with valid data."""
    transformed_data, errors = log_transform(sample_data.copy())
    assert errors is None
    expected_data = np.log(sample_data)
    np.testing.assert_array_almost_equal(transformed_data, expected_data)


def test_log_transform_non_positive():
    """Test that log_transform handles non-positive values by warning and returning NaN."""
    # Test with a zero value
    data_with_zero = np.array([1.0, 2.0, 0.0, 4.0, 5.0])
    with pytest.warns(UserWarning, match="Found 1 non-positive value"):
        transformed_data, _ = log_transform(data_with_zero)

    # Check that the non-positive value became NaN
    assert np.isnan(transformed_data[2])
    # Check that the other values were transformed correctly
    expected_finite = np.log(np.array([1.0, 2.0, 4.0, 5.0]))
    actual_finite = np.delete(transformed_data, 2)
    np.testing.assert_array_almost_equal(actual_finite, expected_finite)

    # Test with a negative value and errors
    data_with_negative = np.array([1.0, -2.0, 3.0, 4.0, 5.0])
    errors = np.full_like(data_with_negative, 0.1)
    with pytest.warns(UserWarning, match="Found 1 non-positive value"):
        transformed_data_neg, transformed_errors_neg = log_transform(
            data_with_negative, errors
        )

    # Check that the non-positive value and its error became NaN
    assert np.isnan(transformed_data_neg[1])
    assert np.isnan(transformed_errors_neg[1])

    # Check that other values and errors were transformed correctly
    expected_data_finite = np.log(np.array([1.0, 3.0, 4.0, 5.0]))
    actual_data_finite = np.delete(transformed_data_neg, 1)
    np.testing.assert_array_almost_equal(actual_data_finite, expected_data_finite)

    expected_errors_finite = np.array([0.1, 0.1, 0.1, 0.1]) / np.array(
        [1.0, 3.0, 4.0, 5.0]
    )
    actual_errors_finite = np.delete(transformed_errors_neg, 1)
    np.testing.assert_array_almost_equal(actual_errors_finite, expected_errors_finite)


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
    detrended, _, diagnostics = detrend_loess(time, trended_data)
    assert np.var(detrended) < np.var(trended_data)
    assert diagnostics["trend_removed"] is True


def test_detrend_loess_with_options(sample_data):
    """Test that LOESS detrending accepts and uses custom options."""
    time = np.arange(len(sample_data))
    # Add an outlier to make the robustness iterations have an effect
    trended_data = sample_data + np.sin(time)
    trended_data[5] = 20

    # Detrend with default options (which include robustness iterations)
    detrended_default, _, _ = detrend_loess(time, trended_data.copy())

    # Detrend with robustness iterations turned off
    detrended_no_iter, _, _ = detrend_loess(time, trended_data.copy(), it=0)

    # The results should be different because the outlier is handled differently
    assert not np.array_equal(detrended_default, detrended_no_iter)


def test_preprocess_data_wrapper(sample_data):
    """Test the main preprocess_data wrapper function."""
    time = np.arange(len(sample_data))
    data_series = pd.Series(sample_data)

    processed, errors, diagnostics = preprocess_data(
        data_series, time, detrend_method="linear"
    )
    assert errors is None
    assert np.mean(processed) == pytest.approx(0)
    assert diagnostics["detrending"]["trend_removed"] is True

    with pytest.warns(UserWarning, match="Unknown detrending method"):
        _, _, diagnostics = preprocess_data(
            data_series, time, detrend_method="bad_method"
        )
        assert diagnostics["detrending"] == {}


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
    data_std = np.std(data, ddof=1)

    transformed_data, transformed_errors = normalize(data, errors)

    expected_errors = 0.5 / data_std
    np.testing.assert_array_almost_equal(transformed_errors, expected_errors)


def test_preprocess_data_with_transforms(sample_data):
    """Test the preprocess_data wrapper with all transformations enabled."""
    time = np.arange(len(sample_data))
    data_series = pd.Series(sample_data)
    error_series = pd.Series(np.full_like(sample_data, 0.1))

    # Apply all transformations
    processed_data, processed_errors, _ = preprocess_data(
        data_series,
        time,
        error_series=error_series,
        log_transform_data=True,
        detrend_method="linear",
        normalize_data=True,
    )

    # Check the final data properties
    assert np.mean(processed_data) == pytest.approx(0)
    assert np.std(processed_data, ddof=1) == pytest.approx(1)

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
    _, processed_errors_from_wrapper, _ = preprocess_data(
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
    data_manual, errors_manual, _ = detrend(time, data_manual, errors_manual)
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
    """Test that LOESS detrending correctly warns when not propagating errors."""
    time = np.arange(len(sample_data))
    # Create some non-linear data with noise
    trended_data = np.sin(time * 0.5) + time * 0.1
    errors = np.full_like(trended_data, 0.1)

    # Detrend with LOESS without bootstrapping
    with pytest.warns(
        UserWarning,
        match="Error propagation for LOESS detrending is not currently supported "
        "without bootstrapping",
    ):
        _detrended_data, detrended_errors, _ = detrend_loess(
            time, trended_data.copy(), errors=errors.copy(), frac=0.5, n_bootstrap=0
        )

    # Without bootstrapping, errors should be returned unchanged.
    assert detrended_errors is not None
    np.testing.assert_array_almost_equal(detrended_errors, errors)


def test_detrend_loess_bootstrap_error_propagation(sample_data):
    """Test that LOESS detrending with bootstrapping correctly propagates errors."""
    time = np.arange(len(sample_data))
    trended_data = np.sin(time * 0.5) + time * 0.1
    errors = np.full_like(trended_data, 0.1)

    # Detrend with LOESS with bootstrapping enabled.
    # We expect no warnings about error propagation.
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")  # Capture all warnings

        _detrended_data, detrended_errors, _ = detrend_loess(
            time,
            trended_data.copy(),
            errors=errors.copy(),
            frac=0.5,
            n_bootstrap=10,  # Use a small number for speed
        )

        # Check that no unsupported error propagation warning was issued
        for warning_message in w:
            assert "Error propagation for LOESS" not in str(warning_message.message)

    # With bootstrapping, errors should be larger than the original errors
    assert detrended_errors is not None
    assert np.all(detrended_errors > errors)

    # Check that the shape is still correct
    assert detrended_errors.shape == errors.shape


def test_detrend_with_nan_in_errors(sample_data):
    """
    Test that detrend falls back to OLS and warns when errors contain NaNs.
    """
    trended_data = sample_data.copy()
    time = np.arange(len(trended_data))
    errors = np.full_like(trended_data, 0.1, dtype=float)
    errors[3] = np.nan  # Introduce a NaN

    # Expect a warning about falling back to OLS because of the NaN
    with pytest.warns(UserWarning, match="Falling back to OLS"):
        _detrended_data, detrended_errors, _ = detrend(
            time, trended_data.copy(), errors=errors.copy()
        )

    # The NaN in the input error should still result in a NaN in the output error
    # because the original errors are the base for propagation.
    assert np.isnan(detrended_errors[3])
    # And other values should be finite and correctly propagated.
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
        processed_data, processed_errors, _ = preprocess_data(
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


def test_transformations_are_pure(sample_data):
    """
    Test that the core transformation functions do not modify the original
    input arrays (i.e., they are pure functions).
    """
    # Create original arrays
    original_data = sample_data.copy()
    original_errors = np.full_like(original_data, 0.1)
    time = np.arange(len(original_data))

    # Create copies for comparison after the functions are called
    data_before = original_data.copy()
    errors_before = original_errors.copy()

    # Call the functions without passing copies
    detrend(time, original_data, original_errors)
    normalize(original_data, original_errors)
    log_transform(original_data, original_errors)

    # Assert that the original arrays have not been modified
    np.testing.assert_array_equal(
        original_data,
        data_before,
        err_msg="The 'detrend', 'normalize', or 'log_transform' function modified the input data array in-place.",
    )
    np.testing.assert_array_equal(
        original_errors,
        errors_before,
        err_msg="The 'detrend', 'normalize', or 'log_transform' function modified the input errors array in-place.",
    )


def test_handle_censored_data_invalid_option():
    """
    Test that handle_censored_data raises a TypeError for an unknown kwarg.
    This is a regression test to ensure the API is strict and fails fast.
    """
    data_series = pd.Series(["1", "<2", "3"])
    # The 'bad_option' is not a valid argument and should cause a TypeError.
    with pytest.raises(TypeError):
        handle_censored_data(data_series, strategy="multiplier", bad_option=True)


def test_detrend_wls_correctness():
    """
    Tests that the detrend function correctly uses WLS when provided with valid
    errors, producing a different (and more accurate) result than OLS.
    """
    # 1. Create synthetic data where OLS and WLS give different results
    np.random.seed(0)
    time = np.arange(20)
    true_trend = 0.5 * time
    noise = np.random.normal(0, 0.5, size=time.shape)
    data = true_trend + noise

    # 2. Create non-uniform errors. Make the first half of the points
    # high-confidence (small error) and the second half low-confidence (large error).
    errors = np.full_like(data, 10.0)
    errors[:10] = 0.1  # High-confidence points

    # The WLS fit should be much closer to the first 10 points than an OLS fit.

    # 3. Manually perform WLS to get the expected detrended result
    X_with_const = sm.add_constant(time)
    weights = 1.0 / (errors**2)
    wls_model = sm.WLS(data, X_with_const, weights=weights)
    wls_results = wls_model.fit()
    expected_trend_wls = wls_results.predict(X_with_const)
    expected_detrended_data_wls = data - expected_trend_wls

    # For comparison, ensure that an OLS fit would be different
    ols_model = sm.OLS(data, X_with_const)
    ols_results = ols_model.fit()
    assert not np.allclose(wls_results.params, ols_results.params)

    # 4. Call the `detrend` function from the preprocessor
    # We expect no warnings because the errors are valid for WLS.
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        detrended_data_from_func, _, _ = detrend(
            time, data.copy(), errors=errors.copy()
        )
        # Check that no fallback warning was issued
        for warning_message in w:
            assert "Falling back to OLS" not in str(warning_message.message)

    # 5. Assert that the function's output matches the manual WLS result
    np.testing.assert_array_almost_equal(
        detrended_data_from_func, expected_detrended_data_wls
    )

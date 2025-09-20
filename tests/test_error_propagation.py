import numpy as np
import pandas as pd
import pytest
from scipy import signal
from src.waterSpec.preprocessor import preprocess_data

def test_error_propagation_with_detrend_and_normalize():
    """
    Tests that errors are correctly propagated when both detrending and
    normalization are applied.

    This test verifies that the `errors` array is correctly passed through
    the `detrend` step and then scaled correctly by the `normalize` step.
    The normalization should be based on the standard deviation of the
    *detrended* data, not the original data.
    """
    # 1. Create Sample Data
    time = np.arange(100)
    trend = 2 * time + 50
    np.random.seed(0)
    noise = np.random.randn(100) * 5
    data_with_noise = trend + noise
    errors = np.full_like(data_with_noise, 10.0, dtype=float)

    data_series = pd.Series(data_with_noise)
    error_series = pd.Series(errors)

    # 2. Calculate the correct expected value
    # To avoid floating point discrepancies, we must replicate the exact
    # operations that occur inside the function being tested.

    # First, the data is detrended using scipy.signal.detrend.
    actual_detrended_data = signal.detrend(data_with_noise)

    # Second, the `normalize` function calculates the std dev of this detrended data.
    std_dev_of_actual_detrended_data = np.std(actual_detrended_data)

    # Finally, the errors are divided by this standard deviation.
    expected_errors = 10.0 / std_dev_of_actual_detrended_data

    # 3. Run the function to get the actual processed errors
    _, processed_errors = preprocess_data(
        data_series,
        time,
        error_series=error_series,
        detrend_method='linear',
        normalize_data=True
    )

    # 4. Verify the results
    assert processed_errors is not None
    assert np.allclose(processed_errors, expected_errors)

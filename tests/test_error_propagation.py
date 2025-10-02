import numpy as np
import pandas as pd

from waterSpec.preprocessor import detrend, normalize, preprocess_data


def test_error_propagation_with_detrend_and_normalize():
    """
    Tests that errors are correctly propagated when both detrending and
    normalization are applied, using the new error propagation logic.
    """
    # 1. Create Sample Data
    time = np.arange(100, dtype=float)
    trend = 2 * time + 50
    rng = np.random.default_rng(0)
    noise = rng.standard_normal(100) * 5
    data_with_noise = trend + noise
    errors = np.full_like(data_with_noise, 10.0, dtype=float)

    data_series = pd.Series(data_with_noise)
    error_series = pd.Series(errors)

    # 2. Run the full pipeline to get the actual processed errors
    _, processed_errors, _ = preprocess_data(
        data_series,
        time,
        error_series=error_series.copy(),
        detrend_method="linear",
        normalize_data=True,
    )

    # 3. Manually calculate the expected error propagation
    manual_data = data_with_noise.copy()
    manual_errors = errors.copy()

    # Step 1: Detrending
    # This now returns modified errors that include uncertainty from the fit
    manual_data, manual_errors, _ = detrend(time, manual_data, manual_errors)

    # Step 2: Normalization
    # This takes the already-modified errors and scales them
    _, manual_errors = normalize(manual_data, manual_errors)

    # 4. Verify the results
    assert processed_errors is not None
    assert np.allclose(processed_errors, manual_errors)

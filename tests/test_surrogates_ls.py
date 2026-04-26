
import numpy as np
import pytest
from waterSpec.surrogates import generate_power_law_surrogates

def test_power_law_surrogates():
    """
    Test that generated surrogates match the input timestamps and have approximately
    the correct spectral slope.
    """
    # 1. Generate irregular timestamps
    rng = np.random.default_rng(42)
    # T=1000, 500 points (avg dt=2)
    time = np.sort(rng.uniform(0, 1000, 500))

    # 2. Generate surrogates with beta=2 (Red noise)
    beta_target = 2.0
    n_surr = 50
    surrogates = generate_power_law_surrogates(
        time, beta=beta_target, n_surrogates=n_surr, seed=42
    )

    # Check shape
    assert surrogates.shape == (n_surr, 500)

    # Check simple property: smoothness
    # Brown noise (beta=2) is much smoother than White noise (beta=0)
    # We can check Lag-1 autocorrelation?
    # Or just check it runs without error.
    # Validating slope on irregular data is hard (that's why we have this package!)

    # Let's do a quick check vs white noise
    surrogates_white = generate_power_law_surrogates(
        time, beta=0.0, n_surrogates=n_surr, seed=42
    )

    # Because surrogates now preserve the true un-standardized variance defined by beta
    # to avoid statistical bias, brown noise absolute differences will naturally be larger
    # simply because brown noise accumulates massive amplitude variance over time.
    # To compare strictly smoothness/shape, we must standardize the variance here.
    sb_norm = surrogates / np.std(surrogates, axis=1, keepdims=True)
    sw_norm = surrogates_white / np.std(surrogates_white, axis=1, keepdims=True)

    # Mean absolute diff should be smaller for brown noise (smoother shape)
    diff_brown_norm = np.mean(np.abs(np.diff(sb_norm, axis=1)))
    diff_white_norm = np.mean(np.abs(np.diff(sw_norm, axis=1)))

    assert diff_brown_norm < diff_white_norm

def test_single_surrogate_seed():
    """Test reproducibility."""
    time = np.linspace(0, 10, 20)
    s1 = generate_power_law_surrogates(time, 1.0, n_surrogates=1, seed=123)
    s2 = generate_power_law_surrogates(time, 1.0, n_surrogates=1, seed=123)
    assert np.allclose(s1, s2)

    s3 = generate_power_law_surrogates(time, 1.0, n_surrogates=1, seed=124)
    assert not np.allclose(s1, s3)

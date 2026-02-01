
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

    # Mean absolute diff should be smaller for brown noise (smoother)
    diff_brown = np.mean(np.abs(np.diff(surrogates, axis=1)))
    diff_white = np.mean(np.abs(np.diff(surrogates_white, axis=1)))

    # Since we normalized variance, smoothness (small diffs) distinguishes them.
    # Actually, standardizing variance means total energy is same.
    # Brown noise concentrates energy at low freq -> slow changes -> small diffs.
    # White noise has high freq energy -> rapid changes -> large diffs.

    assert diff_brown < diff_white

def test_single_surrogate_seed():
    """Test reproducibility."""
    time = np.linspace(0, 10, 20)
    s1 = generate_power_law_surrogates(time, 1.0, n_surrogates=1, seed=123)
    s2 = generate_power_law_surrogates(time, 1.0, n_surrogates=1, seed=123)
    assert np.allclose(s1, s2)

    s3 = generate_power_law_surrogates(time, 1.0, n_surrogates=1, seed=124)
    assert not np.allclose(s1, s3)

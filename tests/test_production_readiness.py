
import numpy as np
import pytest
from waterSpec.causality.ccm import convergent_cross_mapping
from waterSpec.psresp import psresp_fit
from waterSpec.wwz import calculate_wwz_statistics
from waterSpec.haar_analysis import HaarAnalysis
from waterSpec.utils_sim import simulate_tk95, power_law

def test_ccm_uneven_rejection():
    """Test that CCM raises ValueError for unevenly sampled data by default."""
    # Create longer time series to avoid "too short" error
    t = np.sort(np.random.uniform(0, 100, 50)) # 50 points, uneven
    # Ensure at least one gap > median
    t[25] += 5.0

    x = np.sin(t)
    y = np.cos(t)

    # Should raise ValueError because sampling is irregular
    with pytest.raises(ValueError, match="Time series is unevenly sampled"):
        convergent_cross_mapping(t, x, y, E=2, allow_interpolation=False)

    # Should pass (with warning) if allowed
    with pytest.warns(UserWarning, match="Time series appears to be unevenly sampled"):
        res = convergent_cross_mapping(t, x, y, E=2, allow_interpolation=True)
        assert 'rho' in res

def test_psresp_parallel_safety():
    """Test that PSRESP runs with parallel settings without error."""
    # Small dataset
    t = np.linspace(0, 10, 50)
    x = np.random.randn(50)
    err = np.ones(50)*0.1

    # 2 params to test
    params = [(1.5, 1.0), (1.0, 1.0)]

    # Run with n_jobs=-1 (should default to safe max)
    res = psresp_fit(
        t, x, err, power_law, params,
        M=4, # Small M
        oversample=2,
        n_jobs=-1
    )

    assert "best_params" in res
    assert len(res["results"]) == 2

def test_wwz_statistics():
    """Test WWZ p-value calculation."""
    # High Z should have low p
    # Use np.inf to match what code produces for perfect fits
    z_scores = np.array([np.inf, 100.0, 0.0])
    n_eff = 50.0

    p = calculate_wwz_statistics(z_scores, n_eff)

    assert p[0] == 0.0 or p[0] < 1e-100 # Infinity -> 0 p-value
    assert p[1] < 1e-5 # High Z -> small p-value
    assert p[2] == 1.0 # Zero Z -> 1.0 p-value

# Test beta from 0 to 3 in steps of 0.25
@pytest.mark.parametrize("input_beta", np.arange(0, 3.25, 0.25))
def test_haar_beta_recovery(input_beta):
    """Test that Haar analysis recovers beta for even and uneven data."""
    # Generate data
    N = 2048
    dt = 1.0
    # Pass amp=1.0 as second param to power_law
    t, x = simulate_tk95(power_law, (input_beta, 1.0), N, dt, seed=42)

    # Even
    haar = HaarAnalysis(t, x)
    res = haar.run(overlap=True, num_lags=15, n_bootstraps=0)
    recovered_beta = res['beta']

    # Allow some tolerance, especially for beta=3 it gets harder
    tol = 0.4
    assert abs(recovered_beta - input_beta) < tol, f"Even: Input {input_beta}, Got {recovered_beta}"

    # Uneven (50% subsample)
    rng = np.random.default_rng(42)
    idx = np.sort(rng.choice(N, size=N//2, replace=False))
    t_uneven = t[idx]
    x_uneven = x[idx]

    haar_uneven = HaarAnalysis(t_uneven, x_uneven)
    res_u = haar_uneven.run(overlap=True, num_lags=15, n_bootstraps=0)
    recovered_beta_u = res_u['beta']

    assert abs(recovered_beta_u - input_beta) < tol, f"Uneven: Input {input_beta}, Got {recovered_beta_u}"

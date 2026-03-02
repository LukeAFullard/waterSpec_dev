import numpy as np
import pytest
from waterSpec.carma_model import fit_carma_drw

def generate_synthetic_drw(n_points: int, tau: float, sigma: float, mu: float, dt_mean: float, seed: int = 42):
    """
    Generate an irregularly sampled synthetic Damped Random Walk (CARMA(1,0)) process.
    """
    rng = np.random.default_rng(seed)

    # Generate irregular time steps around dt_mean
    dt = rng.exponential(scale=dt_mean, size=n_points - 1)
    # Ensure no zero or negative time steps
    dt = np.clip(dt, a_min=dt_mean * 0.1, a_max=None)
    time = np.concatenate(([0.0], np.cumsum(dt)))

    y = np.zeros(n_points)
    # Asymptotic variance
    var_asymp = (sigma**2 * tau) / 2.0

    # Initialize from stationary distribution
    y[0] = rng.normal(loc=mu, scale=np.sqrt(var_asymp))

    # Generate points iteratively
    for i in range(1, n_points):
        d_t = dt[i-1]
        exp_factor = np.exp(-d_t / tau)

        # Conditional mean
        mean_cond = mu + (y[i-1] - mu) * exp_factor

        # Conditional variance
        var_cond = var_asymp * (1.0 - np.exp(-2.0 * d_t / tau))

        y[i] = rng.normal(loc=mean_cond, scale=np.sqrt(var_cond))

    return time, y

def test_fit_carma_drw_basic():
    """Test fitting a basic DRW process without errors."""
    tau_true = 50.0
    sigma_true = 0.5
    mu_true = 10.0

    time, y = generate_synthetic_drw(
        n_points=2000,
        tau=tau_true,
        sigma=sigma_true,
        mu=mu_true,
        dt_mean=1.0,
        seed=123
    )

    result = fit_carma_drw(time, y)

    assert result["success"] is True
    assert "tau" in result
    assert "sigma" in result
    assert "mu" in result
    assert "psd_func" in result
    assert "beta_func" in result

    # Check values are within reasonable tolerances
    # Due to sample variance, we allow 30% relative error for tau
    assert np.isclose(result["tau"], tau_true, rtol=0.3)
    assert np.isclose(result["sigma"], sigma_true, rtol=0.2)
    assert np.isclose(result["mu"], mu_true, rtol=0.1)

def test_fit_carma_drw_with_errors():
    """Test fitting a DRW process with observation errors."""
    tau_true = 100.0
    sigma_true = 0.2
    mu_true = -5.0

    time, y_true = generate_synthetic_drw(
        n_points=4000,
        tau=tau_true,
        sigma=sigma_true,
        mu=mu_true,
        dt_mean=2.0,
        seed=456
    )

    # Add measurement noise
    rng = np.random.default_rng(456)
    noise_sigma = 0.1
    errors = np.full_like(y_true, noise_sigma)
    y_obs = y_true + rng.normal(0, noise_sigma, size=len(y_true))

    result = fit_carma_drw(time, y_obs, errors=errors)

    assert result["success"] is True
    assert np.isclose(result["tau"], tau_true, rtol=0.3)
    assert np.isclose(result["sigma"], sigma_true, rtol=0.3)
    assert np.isclose(result["mu"], mu_true, rtol=0.2)

def test_fit_carma_drw_callable_functions():
    """Verify the returned PSD and beta functions are callable and behave correctly."""
    tau_true = 20.0
    sigma_true = 1.0
    mu_true = 0.0

    time, y = generate_synthetic_drw(
        n_points=1000,
        tau=tau_true,
        sigma=sigma_true,
        mu=mu_true,
        dt_mean=1.0,
        seed=789
    )

    result = fit_carma_drw(time, y)
    psd_func = result["psd_func"]
    beta_func = result["beta_func"]

    # Test at a few frequencies
    freqs = np.array([0.001, 0.01, 0.1, 1.0])
    psd_vals = psd_func(freqs)
    beta_vals = beta_func(freqs)

    assert len(psd_vals) == 4
    assert np.all(psd_vals > 0)

    # Beta should be 0 at low frequencies and 2 at high frequencies
    assert len(beta_vals) == 4
    assert np.all(beta_vals >= 0) and np.all(beta_vals <= 2.0)

    # Low frequency beta should be smaller than high frequency beta
    assert beta_vals[0] < beta_vals[-1]

def test_fit_carma_drw_duplicate_times_error():
    """Test that a ValueError is raised if there are duplicate or non-increasing times."""
    time = np.array([1.0, 2.0, 2.0, 3.0])
    y = np.array([10.0, 11.0, 12.0, 13.0])

    with pytest.raises(ValueError, match="strictly increasing"):
        fit_carma_drw(time, y)

def test_fit_carma_drw_unordered_times():
    """Test that unordered times are automatically sorted by the function."""
    time = np.array([3.0, 1.0, 4.0, 2.0])
    y = np.array([13.0, 11.0, 14.0, 12.0])

    # The function should sort and NOT raise an error
    # It might not converge perfectly due to so few points, but it shouldn't crash
    result = fit_carma_drw(time, y)

    assert "success" in result

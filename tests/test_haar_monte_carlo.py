import numpy as np
import pytest
from waterSpec.haar_analysis import HaarAnalysis
from waterSpec.surrogates import generate_power_law_surrogates

def generate_irregular_red_noise(n=500, beta=2.0, seed=42):
    """Generates irregularly sampled red noise."""
    rng = np.random.default_rng(seed)
    t = np.sort(rng.uniform(0, 1000, n))
    y = generate_power_law_surrogates(t, beta=beta, n_surrogates=1, seed=seed)[0]
    return t, y

def test_monte_carlo_bootstrap():
    """
    Test that Monte Carlo bootstrap runs and produces valid CIs.
    """
    beta_true = 2.0
    t, y = generate_irregular_red_noise(n=200, beta=beta_true, seed=123)

    haar = HaarAnalysis(t, y)
    # Use fewer bootstraps for speed in testing
    res = haar.run(num_lags=15, n_bootstraps=20, bootstrap_method="monte_carlo", seed=999)

    assert res['bootstrap_method'] == 'monte_carlo'
    assert 'boot_betas' in res
    assert len(res['boot_betas']) == 20

    beta_est = res['beta']
    ci_lower = res['beta_ci_lower']
    ci_upper = res['beta_ci_upper']

    print(f"True Beta: {beta_true}")
    print(f"Est Beta: {beta_est}")
    print(f"CI: [{ci_lower}, {ci_upper}]")

    # The CI should be somewhat centered around the estimate
    assert ci_lower < beta_est < ci_upper

    # Check that std dev is positive
    std_beta = np.std(res['boot_betas'])
    assert std_beta > 0

def test_monte_carlo_bootstrap_white_noise():
    """Test with white noise."""
    beta_true = 0.0
    t, y = generate_irregular_red_noise(n=200, beta=beta_true, seed=456)
    haar = HaarAnalysis(t, y)
    res = haar.run(num_lags=15, n_bootstraps=20, bootstrap_method="monte_carlo", seed=888)

    beta_est = res['beta']
    print(f"White Noise Est: {beta_est}")
    # White noise beta is 0.
    assert -0.5 < beta_est < 0.5

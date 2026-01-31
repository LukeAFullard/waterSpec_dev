import numpy as np
import pytest
from waterSpec.haar_analysis import calculate_haar_fluctuations, fit_haar_slope, HaarAnalysis

def generate_noise(n, color='white'):
    """Generates synthetic noise."""
    np.random.seed(42)
    if color == 'white':
        return np.random.randn(n)
    elif color == 'brownian':
        return np.cumsum(np.random.randn(n))
    elif color == 'pink':
        # Simple approximation or use a library if available.
        # For testing, white and brownian are easier to verify bounds.
        # Let's stick to white (beta=0, H=-0.5) and brownian (beta=2, H=0.5).
        # Note: The formula given is beta = 1 + 2H.
        # White noise: beta ~ 0 => H ~ -0.5
        # Brownian noise: beta ~ 2 => H ~ 0.5
        # Pink noise: beta ~ 1 => H ~ 0
        pass
    return np.random.randn(n)

def test_haar_white_noise():
    # White noise should have beta ~ 0, so H ~ -0.5
    n = 10000
    time = np.arange(n)
    data = generate_noise(n, 'white')

    lags, s1, counts = calculate_haar_fluctuations(time, data, num_lags=20)
    H, beta, r2, intercept = fit_haar_slope(lags, s1)

    print(f"White Noise: H={H}, beta={beta}, R2={r2}, Intercept={intercept}")

    # Allow some margin of error
    assert -0.6 < H < -0.4
    assert -0.2 < beta < 0.2

def test_haar_brownian_noise():
    # Brownian noise should have beta ~ 2, so H ~ 0.5
    n = 10000
    time = np.arange(n)
    data = generate_noise(n, 'brownian')

    lags, s1, counts = calculate_haar_fluctuations(time, data, num_lags=20)
    H, beta, r2, intercept = fit_haar_slope(lags, s1)

    print(f"Brownian Noise: H={H}, beta={beta}, R2={r2}, Intercept={intercept}")

    assert 0.4 < H < 0.6
    assert 1.8 < beta < 2.2

def test_haar_class():
    n = 100
    time = np.sort(np.random.rand(n) * 100)
    data = np.random.randn(n)

    ha = HaarAnalysis(time, data)
    res = ha.run(num_lags=10)

    assert "H" in res
    assert "beta" in res
    assert len(res["lags"]) <= 10

def test_short_time_series():
    # Test with N < 100 as recommended in the prompt
    n = 50
    time = np.arange(n)
    data = np.random.randn(n)

    lags, s1, counts = calculate_haar_fluctuations(time, data, num_lags=10)
    # Ensure it doesn't crash and returns something
    assert len(lags) > 0
    assert len(s1) == len(lags)

def test_irregular_sampling():
    # Create irregular time steps
    n = 1000
    time = np.sort(np.random.rand(n) * 1000)
    data = np.random.randn(n) # White noise

    lags, s1, counts = calculate_haar_fluctuations(time, data, num_lags=20)
    H, beta, r2, intercept = fit_haar_slope(lags, s1)

    print(f"Irregular White Noise: H={H}, beta={beta}, Intercept={intercept}")
    # Should still be roughly white noise
    assert -0.7 < H < -0.3 # Wider tolerance for irregular

import numpy as np
import pytest
from waterSpec.utils_sim.models import red_noise_psd, power_law

def test_red_noise_psd_float():
    """Test red_noise_psd with a single float."""
    f = 1.0
    tau = 2.0
    variance = 3.0

    # expected = (2 * variance * tau) / (1 + (2 * pi * f * tau)^2)
    expected = (2 * 3.0 * 2.0) / (1 + (2 * np.pi * 1.0 * 2.0)**2)
    result = red_noise_psd(f, tau, variance)

    assert isinstance(result, float)
    np.testing.assert_allclose(result, expected)

def test_red_noise_psd_array():
    """Test red_noise_psd with a numpy array."""
    f = np.array([0.1, 1.0, 10.0])
    tau = 2.0
    variance = 3.0

    expected = (2 * 3.0 * 2.0) / (1 + (2 * np.pi * f * 2.0)**2)
    result = red_noise_psd(f, tau, variance)

    assert isinstance(result, np.ndarray)
    assert result.shape == f.shape
    np.testing.assert_allclose(result, expected)

def test_red_noise_psd_zero_frequency():
    """Test red_noise_psd at zero frequency."""
    f = 0.0
    tau = 2.0
    variance = 3.0

    expected = (2 * 3.0 * 2.0) / (1 + 0)
    result = red_noise_psd(f, tau, variance)

    np.testing.assert_allclose(result, expected)

def test_power_law_float():
    """Test power_law with a single float."""
    f = 2.0
    beta = 1.5
    amp = 3.0

    expected = 3.0 * (2.0**(-1.5))
    result = power_law(f, beta, amp)

    assert isinstance(result, float)
    np.testing.assert_allclose(result, expected)

def test_power_law_array():
    """Test power_law with a numpy array."""
    f = np.array([0.1, 1.0, 10.0])
    beta = 1.5
    amp = 3.0

    expected = 3.0 * (f**(-1.5))
    result = power_law(f, beta, amp)

    assert isinstance(result, np.ndarray)
    assert result.shape == f.shape
    np.testing.assert_allclose(result, expected)

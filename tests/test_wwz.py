
import numpy as np
import pytest
from waterSpec.wwz import calculate_wwz

def test_wwz_synthetic_signal():
    """
    Test WWZ on a synthetic signal with a known transient oscillation.
    Signal: 0.5 Hz sine wave present only in the middle of the time series.
    """
    # 1. Setup
    np.random.seed(42)
    n = 200
    time = np.sort(np.random.uniform(0, 100, n)) # Irregular sampling

    # Transient signal: 0.5 Hz between t=30 and t=70
    freq_true = 0.5
    signal = np.sin(2 * np.pi * freq_true * time)
    signal[(time < 30) | (time > 70)] = 0

    noise = np.random.normal(0, 0.1, n)
    data = signal + noise

    # 2. Run WWZ
    freqs = np.linspace(0.1, 1.0, 50)
    taus = np.linspace(0, 100, 100)

    wwz, fs, ts = calculate_wwz(time, data, freqs, taus=taus, decay_constant=0.005)

    # 3. Validation
    # Shape check
    assert wwz.shape == (50, 100)

    # Peak check
    max_idx = np.unravel_index(np.nanargmax(wwz), wwz.shape)
    best_freq = freqs[max_idx[0]]
    best_time = ts[max_idx[1]]

    print(f"Detected: Freq={best_freq:.3f}, Time={best_time:.1f}")

    # Allow some tolerance due to noise and irregularity
    assert 0.45 < best_freq < 0.55
    assert 30 < best_time < 70

    # Check that power is low outside the signal region
    # Region before t=20
    early_mask_t = ts < 20
    early_power = np.max(wwz[:, early_mask_t])
    peak_power = np.max(wwz)

    # Early power should be significantly lower than peak
    assert early_power < peak_power * 0.2

def test_wwz_constant_signal():
    """Test WWZ on a constant signal (should have low power at high freqs)."""
    time = np.linspace(0, 10, 50)
    data = np.ones(50) * 5.0
    freqs = np.linspace(1, 5, 10)

    wwz, _, _ = calculate_wwz(time, data, freqs)

    # Should be close to zero (numerical noise)
    assert np.nanmax(wwz) < 1e-5

def test_wwz_empty_or_small():
    """Test robustness to small inputs."""
    time = np.array([0, 1])
    data = np.array([1, 1])
    freqs = np.array([1.0])

    wwz, _, _ = calculate_wwz(time, data, freqs)
    # Should not crash, output shape (1, 2)
    assert wwz.shape == (1, 2)

def test_wwz_vectorization_consistency():
    """
    Check if vectorized calculation matches singular calculation logic (conceptually).
    We construct a case where we manually check one point.
    """
    time = np.linspace(0, 10, 20)
    data = np.sin(2 * np.pi * 1.0 * time)
    freqs = np.array([1.0])
    taus = np.array([5.0])

    wwz, _, _ = calculate_wwz(time, data, freqs, taus=taus, decay_constant=0.01)

    # Should be high
    assert wwz[0, 0] > 10.0

import numpy as np
import pytest
from waterSpec.wwz_coherence import calculate_wwz_coherence

def test_wwz_coherence_coupled_signal():
    """
    Test WWZ Coherence on two synthetic signals with a shared transient oscillation.
    Signal 1 and Signal 2 both have a 0.5 Hz sine wave present in the middle of the time series.
    They should show high coherence in that region.
    """
    # 1. Setup
    np.random.seed(42)
    n = 200
    time = np.sort(np.random.uniform(0, 100, n)) # Irregular sampling

    # Transient signal: 0.5 Hz between t=30 and t=70
    freq_true = 0.5
    shared_signal = np.sin(2 * np.pi * freq_true * time)
    shared_signal[(time < 30) | (time > 70)] = 0

    noise1 = np.random.normal(0, 0.1, n)
    noise2 = np.random.normal(0, 0.1, n)

    data1 = shared_signal + noise1
    data2 = shared_signal + noise2

    # 2. Run WWZ Coherence
    freqs = np.linspace(0.1, 1.0, 50)
    taus = np.linspace(0, 100, 100)

    coherence, fs, ts = calculate_wwz_coherence(
        time, data1, time, data2, freqs, taus=taus, decay_constant=0.005, smoothing_window=2.0
    )

    # 3. Validation
    # Shape check
    assert coherence.shape == (50, 100)

    # Find region of high coherence
    # It should be high around freq=0.5 and time between 30 and 70

    freq_idx = np.argmin(np.abs(fs - 0.5))
    time_mask = (ts > 40) & (ts < 60) # Central part of the oscillation

    # Coherence should be high here
    mean_coh_in_region = np.mean(coherence[freq_idx, time_mask])
    assert mean_coh_in_region > 0.8, f"Expected high coherence, got {mean_coh_in_region:.3f}"

    # Check outside the region (e.g., at time < 20)
    early_time_mask = ts < 20
    mean_coh_early = np.mean(coherence[freq_idx, early_time_mask])

    # Early coherence should be lower
    assert mean_coh_early < 0.7, f"Expected low coherence early, got {mean_coh_early:.3f}"

def test_wwz_coherence_uncoupled_signal():
    """
    Test WWZ Coherence on two independent noise signals.
    The overall coherence should be low.
    """
    np.random.seed(43)
    n = 200
    time1 = np.linspace(0, 100, n)
    time2 = np.linspace(0, 100, n)

    data1 = np.random.normal(0, 1.0, n)
    data2 = np.random.normal(0, 1.0, n)

    freqs = np.linspace(0.1, 1.0, 20)
    taus = np.linspace(0, 100, 50)

    coherence, fs, ts = calculate_wwz_coherence(
        time1, data1, time2, data2, freqs, taus=taus, decay_constant=0.005, smoothing_window=2.0
    )

    # Average coherence should be low
    mean_coherence = np.mean(coherence)
    assert mean_coherence < 0.5, f"Expected low overall coherence, got {mean_coherence:.3f}"

def test_wwz_coherence_identical_signals():
    """
    Test WWZ Coherence on the exact same signal.
    The coherence should be very close to 1 everywhere valid.
    """
    np.random.seed(44)
    n = 100
    time = np.linspace(0, 50, n)
    data = np.sin(2 * np.pi * 0.5 * time) + np.random.normal(0, 0.1, n)

    freqs = np.linspace(0.1, 1.0, 10)
    taus = np.linspace(0, 50, 20)

    coherence, _, _ = calculate_wwz_coherence(
        time, data, time, data, freqs, taus=taus, decay_constant=0.005, smoothing_window=2.0
    )

    # Where power is non-zero, coherence should be ~1
    # We can check the maximum coherence
    assert np.max(coherence) > 0.99

def test_wwz_coherence_empty_or_small():
    """Test robustness to small inputs."""
    time1 = np.array([0, 1, 2])
    data1 = np.array([1, -1, 1])
    time2 = np.array([0, 1, 2])
    data2 = np.array([-1, 1, -1])
    freqs = np.array([1.0])

    coherence, _, _ = calculate_wwz_coherence(time1, data1, time2, data2, freqs)

    # Should not crash, we don't care about the value for such small inputs
    assert coherence.shape[0] == 1
    # Check default taus length
    assert coherence.shape[1] == 200

def test_wwz_coherence_taus_none():
    """Test when taus is None, it defaults to 200 points across the shared time range."""
    time1 = np.linspace(10, 90, 100)
    data1 = np.random.randn(100)
    time2 = np.linspace(20, 100, 100)
    data2 = np.random.randn(100)

    freqs = np.array([0.5, 1.0])

    coherence, fs, ts = calculate_wwz_coherence(time1, data1, time2, data2, freqs)

    assert len(ts) == 200
    assert ts.min() == 20.0 # max of mins (10, 20) -> actually min of mins or what?
    # the code says: t_min = max(time1.min(), time2.min()) -> max(10, 20) = 20
    # t_max = min(time1.max(), time2.max()) -> min(90, 100) = 90
    assert ts.max() == 90.0

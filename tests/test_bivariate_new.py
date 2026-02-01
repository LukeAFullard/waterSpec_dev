
import numpy as np
import pytest
from waterSpec.bivariate import BivariateAnalysis

def test_irregular_surrogates():
    """
    Test that calculate_significance handles irregular data without crashing
    and returns valid p-values.
    """
    # Create irregular sine wave
    rng = np.random.default_rng(42)
    # 50 irregular points
    time = np.sort(rng.uniform(0, 100, 50))
    data1 = np.sin(time / 10.0)
    data2 = np.cos(time / 10.0) # Correlated (phase shifted)

    biv = BivariateAnalysis(time, data1, "Sine", time, data2, "Cosine")
    biv.align_data(tolerance=0.1) # Self-alignment

    lags = np.logspace(0, 1.5, 5)

    # This should trigger the resampling logic
    res = biv.calculate_significance(lags, n_surrogates=10, seed=42)

    assert "p_values" in res
    assert len(res["p_values"]) == len(lags)
    assert not np.all(np.isnan(res["p_values"]))

def test_spectral_coherence():
    """
    Test magnitude squared coherence calculation.
    """
    time = np.linspace(0, 100, 100)
    data1 = np.sin(2 * np.pi * 0.1 * time) # 0.1 Hz
    data2 = np.sin(2 * np.pi * 0.1 * time + np.pi/4) + 0.1 * np.random.randn(100)

    biv = BivariateAnalysis(time, data1, "Sig1", time, data2, "Sig2")
    biv.align_data(tolerance=0.01)

    coh_res = biv.calculate_spectral_coherence(min_freq=0.01, max_freq=0.5)

    freqs = coh_res['frequency']
    coherence = coh_res['coherence']

    # Check peak coherence at 0.1 Hz
    target_idx = np.argmin(np.abs(freqs - 0.1))
    assert coherence[target_idx] > 0.8 # Should be high

def test_large_gap_warning():
    """
    Test that large gaps trigger a warning in surrogates.
    """
    # Create data with a huge gap
    # Use floats for time to avoid pandas int tolerance issues
    time = np.concatenate([np.arange(0.0, 10.0), np.arange(100.0, 110.0)])
    data = np.random.randn(20)

    biv = BivariateAnalysis(time, data, "V1", time, data, "V2")
    biv.align_data(tolerance=0.1)

    lags = np.array([1, 2])

    with pytest.warns(UserWarning, match="Large data gap"):
        biv.calculate_significance(lags, n_surrogates=5)

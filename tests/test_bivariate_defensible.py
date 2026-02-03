
import numpy as np
import pytest
from waterSpec.bivariate import BivariateAnalysis

def test_bivariate_ls_cross_analysis():
    t1 = np.sort(np.random.uniform(0, 100, 50))
    t2 = np.sort(np.random.uniform(0, 100, 50))

    # Simple sine wave correlation
    freq = 0.1
    y1 = np.sin(2 * np.pi * freq * t1)
    y2 = np.sin(2 * np.pi * freq * t2)

    biv = BivariateAnalysis(t1, y1, "Site1", t2, y2, "Site2")

    freqs = np.linspace(0.05, 0.2, 20)

    results = biv.run_ls_cross_analysis(freqs)

    assert 'cross_power' in results
    assert 'phase_lag' in results

    # Check peak power is near freq 0.1
    peak_idx = np.argmax(results['cross_power'])
    assert np.isclose(results['freqs'][peak_idx], freq, atol=0.02)

def test_bivariate_wwz_coherence():
    t1 = np.sort(np.random.uniform(0, 100, 50))
    t2 = np.sort(np.random.uniform(0, 100, 50))

    freq = 0.1
    y1 = np.sin(2 * np.pi * freq * t1)
    y2 = np.sin(2 * np.pi * freq * t2)

    biv = BivariateAnalysis(t1, y1, "Site1", t2, y2, "Site2")

    freqs = np.linspace(0.05, 0.2, 20)

    results = biv.run_wwz_coherence_analysis(freqs)

    assert 'coherence' in results
    assert results['coherence'].shape == (len(freqs), 200) # Default 200 taus

    # Check coherence is high at 0.1 Hz
    freq_idx = np.argmin(np.abs(freqs - 0.1))
    mean_coh = np.mean(results['coherence'][freq_idx, :])
    assert mean_coh > 0.5

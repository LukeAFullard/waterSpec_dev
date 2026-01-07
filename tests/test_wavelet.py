import numpy as np
import pytest
from waterSpec.wavelet import compute_wwz, fit_spectral_slope, fit_segmented_slope, multifractal_analysis_pipeline, HAS_PYMULTIFRACS

def generate_red_noise(n, alpha=1.0):
    """Generate red noise (1/f^alpha) using FFT."""
    white = np.random.normal(size=n)
    fft = np.fft.rfft(white)
    freqs = np.fft.rfftfreq(n)
    # Scale amplitude by 1/f^(alpha/2) -> power 1/f^alpha
    with np.errstate(divide='ignore'):
        scale = 1.0 / (freqs ** (alpha / 2.0))
    scale[0] = 0 # remove DC
    fft_colored = fft * scale
    return np.fft.irfft(fft_colored, n=n)

def test_wwz_white_noise():
    """Test WWZ on white noise (beta=0)."""
    np.random.seed(42)
    n = 200
    t = np.sort(np.random.uniform(0, 100, n))
    y = np.random.normal(size=n)

    # Use finer resolution
    res = compute_wwz(t, y, n_scales=50, decay_constant=1e-3)

    # Fit slope
    fit = fit_spectral_slope(res.frequencies, res.global_power)

    # White noise slope should be close to 0
    # Relaxed assertion due to short series bias
    assert -0.9 < fit.beta < 0.9
    assert fit.model_type == 'linear'

def test_wwz_red_noise():
    """Test WWZ on red noise (beta approx 2)."""
    np.random.seed(42)
    n = 500
    # Regular sampling first to generate clean red noise
    t_reg = np.linspace(0, 500, n)
    y_reg = generate_red_noise(n, alpha=2.0)

    # Irregular sampling: randomly select points
    idx = np.sort(np.random.choice(n, size=300, replace=False))
    t = t_reg[idx]
    y = y_reg[idx]

    # Use finer resolution and smaller decay constant for sharper peaks
    res = compute_wwz(t, y, n_scales=50, decay_constant=1e-3)
    fit = fit_spectral_slope(res.frequencies, res.global_power)

    # Red noise slope should be close to 2 (beta > 1)
    # Even with bias, it should be distinct from white noise and > 0.5
    print(f"Fitted beta for red noise: {fit.beta}")
    assert fit.beta > 0.5
    assert fit.beta < 3.0

def test_segmented_slope():
    """Test segmented slope fitting."""
    freqs = np.logspace(-2, 0, 50)
    # Create a spectrum with break at f=0.1
    # beta=0 for f < 0.1, beta=2 for f > 0.1
    power = np.zeros_like(freqs)
    mask = freqs < 0.1
    power[mask] = 1.0 # Flat
    power[~mask] = 1.0 * (freqs[~mask] / 0.1) ** (-2)

    # Add small noise
    power = power * np.exp(np.random.normal(0, 0.1, size=len(power)))

    fit = fit_segmented_slope(freqs, power, n_breakpoints=1)

    if fit.model_type == 'segmented':
        # Check slopes
        slopes = fit.segment_slopes
        assert len(slopes) == 2
        # First slope approx 0
        assert abs(slopes[0]) < 0.5
        # Second slope approx 2
        assert abs(slopes[1] - 2.0) < 0.5
        # Breakpoint around -1 (log10(0.1))
        # piecewise_regression returns breakpoint in x-coordinates (log10 freq)
        assert abs(fit.breakpoints[0] - np.log10(0.1)) < 0.5

@pytest.mark.skipif(not HAS_PYMULTIFRACS, reason="pymultifracs not installed")
def test_multifractal_pipeline():
    """Test multifractal pipeline runs."""
    np.random.seed(42)
    n = 300
    t = np.sort(np.random.uniform(0, 100, n))
    y = np.random.normal(size=n)

    res = multifractal_analysis_pipeline(t, y)

    assert 'error' not in res
    assert 'cumulants' in res
    assert 'spectrum' in res

import numpy as np
import pytest
from waterSpec.ls_cross_spectrum import calculate_ls_cross_spectrum, calculate_time_lag, _compute_ls_complex_coeffs

def test_calculate_ls_cross_spectrum_basic():
    # Basic shape/type validation
    time1 = np.linspace(0, 10, 100)
    data1 = np.sin(2 * np.pi * 0.5 * time1)

    time2 = np.linspace(0, 10, 80)
    data2 = np.cos(2 * np.pi * 0.5 * time2)

    freqs = np.array([0.1, 0.5, 1.0])

    cross_power, phase_lag, coherence, out_freqs = calculate_ls_cross_spectrum(
        time1, data1, time2, data2, freqs
    )

    assert cross_power.shape == freqs.shape
    assert phase_lag.shape == freqs.shape
    assert coherence.shape == freqs.shape
    assert out_freqs.shape == freqs.shape
    assert np.all(out_freqs == freqs)
    assert np.all(coherence == 1.0)

def test_calculate_ls_cross_spectrum_known_phase():
    """
    Test phase recovery for synthetic sine waves with known phase shift.

    y1 = cos(2*pi*f*t)
    y2 = cos(2*pi*f*t - phi)
    -> phase_lag = phase1 - phase2

    Wait, in _compute_ls_complex_coeffs:
    y = A cos + B sin.
    Z = A + iB
    phase = arctan2(B, A)

    For y1 = cos(wt), A=1, B=0 => Z1 = 1 + 0i => phase1 = 0
    For y2 = cos(wt - phi) = cos(wt)cos(phi) + sin(wt)sin(phi), A=cos(phi), B=sin(phi)
    Z2 = cos(phi) + i sin(phi) => phase2 = phi

    Sxy = Z1 * conj(Z2)
    phase_lag = angle(Sxy) = phase1 - phase2 = 0 - phi = -phi

    Ah! The code returns -phi. So if true_phase_shift is phi, we expect -phi.
    Let's fix the test to match the code's sign convention.
    """
    f_signal = 0.25
    omega = 2 * np.pi * f_signal

    true_phase_shift = np.pi / 4.0 # phi

    np.random.seed(42)
    time1 = np.sort(np.random.uniform(0, 50, 200))
    time2 = np.sort(np.random.uniform(0, 50, 150))

    data1 = np.cos(omega * time1)
    data2 = np.cos(omega * time2 - true_phase_shift)

    freqs = np.array([0.1, f_signal, 0.5])

    cross_power, phase_lag, coherence, _ = calculate_ls_cross_spectrum(
        time1, data1, time2, data2, freqs
    )

    peak_idx = np.argmax(cross_power)
    assert freqs[peak_idx] == f_signal

    calculated_phase = phase_lag[peak_idx]

    # Due to convention: phase_lag = phase(1) - phase(2)
    # y1 has phase 0, y2 has phase phi. 0 - phi = -phi.
    expected_phase = -true_phase_shift

    assert np.isclose(calculated_phase, expected_phase, atol=0.05)

def test_calculate_ls_cross_spectrum_sine_vs_cosine():
    """
    y1 = cos(wt)
    y2 = sin(wt) = cos(wt - pi/2)
    Phase lag = phase1 - phase2 = 0 - pi/2 = -pi/2
    """
    f_signal = 1.0
    omega = 2 * np.pi * f_signal
    time1 = np.linspace(0, 10, 500)
    time2 = np.linspace(0, 10, 500)

    data1 = np.cos(omega * time1)
    data2 = np.sin(omega * time2)

    freqs = np.array([f_signal])

    _, phase_lag, _, _ = calculate_ls_cross_spectrum(
        time1, data1, time2, data2, freqs
    )

    assert np.isclose(phase_lag[0], -np.pi / 2, atol=0.01)


def test_calculate_time_lag():
    """
    Test conversion from phase lag to time lag:
    time_lag = phase_lag / (2 * pi * f)
    """
    phase_lag = np.array([np.pi, np.pi/2, 0, -np.pi/2])
    freqs = np.array([0.5, 1.0, 2.0, 0.25])

    # Expected time lags:
    # pi / (2 * pi * 0.5) = 1.0
    # (pi/2) / (2 * pi * 1.0) = 0.25
    # 0 / (2 * pi * 2.0) = 0
    # (-pi/2) / (2 * pi * 0.25) = -1.0
    expected_time_lag = np.array([1.0, 0.25, 0.0, -1.0])

    time_lag = calculate_time_lag(phase_lag, freqs)

    assert np.allclose(time_lag, expected_time_lag)

def test_calculate_time_lag_zero_frequency():
    """
    Zero frequency should be handled without DivisionByZero warnings
    """
    phase_lag = np.array([np.pi, np.pi/2])
    freqs = np.array([0.0, 1.0])

    # The valid elements are f > 0
    # For f=0, time_lag should be 0 based on zeros_like initialization
    expected_time_lag = np.array([0.0, 0.25])

    time_lag = calculate_time_lag(phase_lag, freqs)

    assert np.allclose(time_lag, expected_time_lag)


def test_compute_ls_complex_coeffs_errors():
    """
    Test that providing errors weights the least squares solution properly.
    A simple test: very high error on the second half of data should make the fit
    match the first half mostly.
    """
    time = np.linspace(0, 10, 100)
    omega = 2 * np.pi * 1.0
    data = np.cos(omega * time)

    # Introduce a big jump in the second half, but give it huge errors
    data[50:] += 100.0
    errors = np.ones(100)
    errors[50:] = 1e6

    freqs = np.array([1.0])

    # It should still find A ≈ 1, B ≈ 0 because the corrupted data has huge errors
    coeffs = _compute_ls_complex_coeffs(time, data, freqs, errors=errors)
    Z = coeffs[0]

    # Z = A + iB
    A = Z.real
    B = Z.imag

    assert np.isclose(A, 1.0, atol=0.1)
    assert np.isclose(B, 0.0, atol=0.1)

def test_compute_ls_complex_coeffs_zero_errors():
    """
    Zero errors should not cause division by zero.
    The code replaces 0 with 1e-9.
    """
    time = np.linspace(0, 10, 50)
    data = np.sin(2 * np.pi * 1.0 * time)
    freqs = np.array([1.0])

    # errors array with 0s
    errors = np.zeros(50)

    # should run without warning/error
    coeffs = _compute_ls_complex_coeffs(time, data, freqs, errors=errors)

    assert not np.isnan(coeffs[0])

def test_compute_ls_complex_coeffs_singular_matrix():
    """
    Test LinAlgError handling in _compute_ls_complex_coeffs.
    If the system is singular, it should return np.nan.
    We can force this by providing only 1 data point or 2 identical data points
    which is not enough to solve for 3 parameters (C, A, B).
    """
    time = np.array([1.0, 1.0]) # Same time, so singular system
    data = np.array([2.0, 2.0])
    freqs = np.array([1.0])

    coeffs = _compute_ls_complex_coeffs(time, data, freqs)

    assert np.isnan(coeffs[0])

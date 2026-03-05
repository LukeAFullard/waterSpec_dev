import numpy as np
import pytest
from waterSpec.psresp import psresp_fit, bin_power_spectrum
from waterSpec.utils_sim import simulate_tk95, power_law, resample_to_times

def test_bin_power_spectrum():
    """Test the functionality of bin_power_spectrum."""
    freqs = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
    power = np.array([10,  20,  30,  40,  50,  60,  70,  80,  90, 100])

    # Test 1: Standard case
    bins = np.array([0.0, 0.35, 0.65, 1.1])
    # Bins:
    # Bin 1 (0.0 to 0.35): freqs [0.1, 0.2, 0.3], powers [10, 20, 30]
    # Bin 2 (0.35 to 0.65): freqs [0.4, 0.5, 0.6], powers [40, 50, 60]
    # Bin 3 (0.65 to 1.1): freqs [0.7, 0.8, 0.9, 1.0], powers [70, 80, 90, 100]
    binned_freqs, binned_power = bin_power_spectrum(freqs, power, bins)

    expected_freqs = np.array([0.2, 0.5, 0.85])
    expected_power = np.array([20.0, 50.0, 85.0])

    np.testing.assert_allclose(binned_freqs, expected_freqs)
    np.testing.assert_allclose(binned_power, expected_power)

    # Test 2: Edge case with empty bins
    bins_empty = np.array([0.0, 0.35, 0.38, 0.65, 1.1])
    # Bin 2 (0.35 to 0.38) is empty
    binned_freqs_empty, binned_power_empty = bin_power_spectrum(freqs, power, bins_empty)

    expected_freqs_empty = np.array([0.2, 0.365, 0.5, 0.85])
    expected_power_empty = np.array([20.0, np.nan, 50.0, 85.0])

    np.testing.assert_allclose(binned_freqs_empty, expected_freqs_empty)
    np.testing.assert_allclose(binned_power_empty, expected_power_empty, equal_nan=True)

    # Test 3: Edge case where frequencies are out of bounds
    bins_out_of_bounds = np.array([0.25, 0.85])
    # freqs [0.1, 0.2] are below the first bin edge
    # freqs [0.9, 1.0] are above the last bin edge
    # The only valid bin is 1: (0.25 to 0.85) -> freqs [0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
    binned_freqs_oob, binned_power_oob = bin_power_spectrum(freqs, power, bins_out_of_bounds)

    expected_freqs_oob = np.array([0.55])
    expected_power_oob = np.array([55.0])

    np.testing.assert_allclose(binned_freqs_oob, expected_freqs_oob)
    np.testing.assert_allclose(binned_power_oob, expected_power_oob)

def test_simulate_tk95_power_law():
    """Test that TK95 produces a time series with roughly the correct PSD slope."""
    N = 10000
    dt = 1.0
    beta = 1.5
    amp = 10.0

    # Simulate
    t, x = simulate_tk95(power_law, (beta, amp), N, dt)

    assert len(t) == N
    assert len(x) == N

    # Check PSD of result
    freqs = np.fft.rfftfreq(N, d=dt)
    fft_x = np.fft.rfft(x)

    power = np.abs(fft_x)**2

    # Ignore DC and very low/high freq
    mask = (freqs > 0.01) & (freqs < 0.4)
    log_f = np.log10(freqs[mask])
    log_p = np.log10(power[mask])

    slope, intercept = np.polyfit(log_f, log_p, 1)

    assert np.isclose(slope, -beta, atol=0.2)

def test_psresp_fit_recovery():
    """Test that PSRESP can recover the input beta parameter."""
    # 1. Generate synthetic "observed" data
    np.random.seed(42)
    N_obs = 100
    T_obs = 100.0
    t_obs = np.sort(np.random.uniform(0, T_obs, N_obs)) # Irregular sampling

    # Generate underlying signal using TK95 on fine grid then resample
    true_beta = 1.5
    true_amp = 1.0
    N_fine = 2000
    dt_fine = T_obs / N_fine
    t_fine, x_fine = simulate_tk95(power_law, (true_beta, true_amp), N_fine, dt_fine)

    x_obs = resample_to_times(t_fine, x_fine, t_obs)
    err_obs = np.ones_like(x_obs) * 0.1
    x_obs += np.random.normal(0, 0.1, size=len(x_obs))

    # 2. Run PSRESP
    # Search grid
    betas = [1.0, 1.5, 2.0]
    params_list = [(b, true_amp) for b in betas]

    # Use coarse settings for speed in test
    freqs = np.logspace(np.log10(2/T_obs), np.log10(0.5 * N_obs/T_obs), 20)

    # Use serial execution for test (n_jobs=1) to avoid overhead/issues in test env
    result = psresp_fit(
        t_obs, x_obs, err_obs,
        power_law,
        params_list,
        freqs=freqs,
        M=50, # Small number of sims
        oversample=5,
        length_factor=2.0,
        n_jobs=1,
        binning=False, # Check without binning first
        seed=42
    )

    best_beta = result["best_params"][0]

    assert best_beta == 1.5

    # Check structure
    assert "chi2" in result["results"][0]
    assert "success_fraction" in result["results"][0]

def test_psresp_parallel():
    """Test PSRESP works with parallel execution."""
    np.random.seed(42)
    t_obs = np.linspace(0, 10, 20)
    x_obs = np.random.normal(0, 1, 20)
    err_obs = np.ones(20) * 0.1

    params_list = [(1.5, 1.0)]
    freqs = np.linspace(0.1, 0.5, 5)

    result = psresp_fit(
        t_obs, x_obs, err_obs,
        power_law,
        params_list,
        freqs=freqs,
        M=10,
        n_jobs=2 # Use 2 workers
    )

    assert len(result["results"]) == 1
    assert result["results"][0]["params"] == (1.5, 1.0)

def test_psresp_large_offset():
    """Test that PSRESP handles large time offsets (e.g. MJD)."""
    np.random.seed(42)
    N_obs = 100
    T_obs = 100.0
    offset = 50000.0 # Large offset
    t_obs = np.sort(np.random.uniform(0, T_obs, N_obs)) + offset

    # Generate underlying signal using TK95 on fine grid then resample
    true_beta = 1.5
    true_amp = 1.0
    N_fine = 2000
    dt_fine = T_obs / N_fine
    t_fine, x_fine = simulate_tk95(power_law, (true_beta, true_amp), N_fine, dt_fine)

    # Resample needs to handle offset manually if we were doing it outside psresp_fit,
    # but inside psresp_fit it should handle it.
    # Here we are generating the "observed" data, so we must be careful.
    # If we use `resample_to_times` with large offset on `t_fine` (starts at 0), we get flat line.
    # So we must shift `t_obs` for generating the synthetic data too.
    x_obs = resample_to_times(t_fine, x_fine, t_obs - offset)
    err_obs = np.ones_like(x_obs) * 0.1
    x_obs += np.random.normal(0, 0.1, size=len(x_obs))

    params_list = [(1.5, 1.0)]

    # If the fix works, this should run without error and give reasonable result
    # (previously would fail or give garbage because of interp)
    result = psresp_fit(
        t_obs, x_obs, err_obs,
        power_law,
        params_list,
        M=20,
        oversample=2,
        length_factor=2.0,
        n_jobs=1
    )

    assert result["best_params"][0] == 1.5

def test_psresp_binning():
    """Test that binning logic works."""
    t_obs = np.linspace(0, 100, 100)
    x_obs = np.random.normal(0, 1, 100)
    err_obs = np.ones(100) * 0.1

    params_list = [(1.5, 1.0)]

    # Use default freqs generation which produces a fine grid, then bin
    result = psresp_fit(
        t_obs, x_obs, err_obs,
        power_law,
        params_list,
        M=10,
        oversample=2,
        n_jobs=1,
        binning=True,
        n_bins=5
    )

    assert len(result["target_power"]) <= 5
    assert len(result["target_freqs"]) <= 5

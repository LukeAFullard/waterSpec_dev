import numpy as np
import pytest
from waterSpec.psresp import simulate_tk95, psresp_fit, power_law, resample_to_times

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

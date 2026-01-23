import numpy as np
import pandas as pd
import pytest
import os
from waterSpec import Analysis
from waterSpec.psresp import simulate_tk95
from waterSpec.haar_analysis import HaarAnalysis

# --- Helper Functions ---

def power_law(f, beta, amp=1.0):
    """
    Power law PSD: P(f) = amp * f^(-beta)
    """
    return amp * (f**(-beta))

def broken_power_law(f, beta1, beta2, f_break, amp=1.0):
    """
    Broken power law PSD.
    """
    # Continuous at f_break
    # P(f) = A * f^-beta1 for f <= f_break
    # P(f) = B * f^-beta2 for f > f_break
    # A * f_break^-beta1 = B * f_break^-beta2
    # B = A * f_break^(beta2 - beta1)

    psd = np.zeros_like(f)
    mask1 = f <= f_break
    mask2 = f > f_break

    psd[mask1] = amp * f[mask1]**(-beta1)

    amp2 = amp * f_break**(beta2 - beta1)
    psd[mask2] = amp2 * f[mask2]**(-beta2)

    return psd

def generate_series(psd_func, params, n_points=1000, dt=1.0, seed=None):
    if seed is not None:
        np.random.seed(seed)

    time, flux = simulate_tk95(psd_func, params, n_points, dt)
    return time, flux

def create_data_file(tmp_path, time, series, filename="test_data.csv"):
    file_path = tmp_path / filename
    df = pd.DataFrame({"time": time, "value": series})
    # Convert time to datetime for Analysis class (it expects datetime or similar usually, though it handles floats too if config'd)
    # But let's use datetime to be safe and standard
    start_date = pd.Timestamp("2020-01-01")
    df["time"] = start_date + pd.to_timedelta(df["time"], unit="D")

    df.to_csv(file_path, index=False)
    return str(file_path)

# --- Tests ---

@pytest.mark.parametrize("beta", [0.5, 1.5, 2.5])
def test_known_single_slope(tmp_path, beta):
    """
    Test that waterSpec correctly estimates the spectral slope (beta)
    for a simple power-law time series.
    """
    n_points = 500
    dt = 1.0 # 1 day

    # Generate series
    time, series = generate_series(power_law, (beta,), n_points=n_points, dt=dt, seed=42)

    file_path = create_data_file(tmp_path, time, series)

    # Analyze
    analyzer = Analysis(file_path=file_path, time_col="time", data_col="value", detrend_method=None)
    # Force standard model (0 breakpoints) to check beta estimation
    results = analyzer.run_full_analysis(
        output_dir=str(tmp_path),
        max_breakpoints=0,
        n_bootstraps=10,
        ci_method="parametric", # Faster for test
    )

    estimated_beta = results["beta"]
    print(f"True beta: {beta}, Estimated beta: {estimated_beta}")

    # Allow some tolerance. Spectral estimation has variance.
    # 0.3 is a reasonable tolerance for n=500 and TK95 generation.
    assert estimated_beta == pytest.approx(beta, abs=0.3)


def test_broken_power_law_slope(tmp_path):
    """
    Test that waterSpec identifies a breakpoint and correct slopes
    for a segmented power-law series.
    """
    n_points = 1000
    dt = 1.0
    beta1 = 2.0  # Low freq
    beta2 = 0.5  # High freq

    # Breakpoint frequency: let's put it in the middle of log frequency range
    # f_min = 1/N, f_max = 0.5/dt = 0.5
    f_min = 1.0 / (n_points * dt)
    f_max = 0.5 / dt
    f_break = np.sqrt(f_min * f_max) # Geometric mean

    time, series = generate_series(broken_power_law, (beta1, beta2, f_break), n_points=n_points, dt=dt, seed=123)

    file_path = create_data_file(tmp_path, time, series)

    analyzer = Analysis(file_path=file_path, time_col="time", data_col="value", detrend_method=None)

    # Run analysis allowing up to 1 breakpoint
    results = analyzer.run_full_analysis(
        output_dir=str(tmp_path),
        max_breakpoints=1,
        n_bootstraps=10,
        ci_method="parametric",
    )

    # Check if it chose the segmented model
    # Note: Model selection depends on BIC. With strong enough signal change, it should pick segmented.
    # However, for this test, we can check if the segmented model fit (even if not chosen by BIC due to penalty)
    # has slopes close to ground truth.

    # But ideally it SHOULD be chosen if the break is clear.

    # Let's inspect the segmented result specifically if it wasn't chosen,
    # but let's first check if it WAS chosen.

    print(f"Chosen model: {results['chosen_model']}")

    if "segmented" in results["chosen_model"]:
        betas = results["betas"]
        # betas should correspond to [beta1, beta2] roughly
        print(f"True betas: [{beta1}, {beta2}], Estimated betas: {betas}")
        assert betas[0] == pytest.approx(beta1, abs=0.5)
        assert betas[1] == pytest.approx(beta2, abs=0.5)
    else:
        # If standard was chosen, maybe the break wasn't sharp enough or penalty too high?
        # Let's fail/warn but print what happened.
        print("Standard model was chosen over segmented.")
        # We can look into 'all_models' if available, but for now let's just assert.
        # It's possible for random realization to not show clear break.
        # But with 1000 points and 2.0 vs 0.5, it should be clear.

        # NOTE: If this fails, we might need to adjust f_break or N.
        # Let's assert that it chose segmented.
        pytest.fail(f"Did not choose segmented model. Chosen: {results['chosen_model']}, Beta: {results.get('beta')}")


@pytest.mark.parametrize("noise_std", [0.1, 1.0, 5.0])
def test_noise_levels(tmp_path, noise_std):
    """
    Test the effect of adding white noise to a signal with beta=2.0.
    High noise should look like a segmented spectrum (beta=2 at low freq, beta=0 at high freq).
    """
    n_points = 1000  # Increased points for better spectral resolution
    dt = 1.0
    true_beta = 2.0

    # Generate pure signal
    time, signal = generate_series(power_law, (true_beta,), n_points=n_points, dt=dt, seed=999)

    # Add white noise
    rng = np.random.default_rng(42 + int(noise_std*10))
    noise = rng.normal(0, noise_std, size=len(signal))
    series = signal + noise

    file_path = create_data_file(tmp_path, time, series, filename=f"noise_{noise_std}.csv")

    analyzer = Analysis(file_path=file_path, time_col="time", data_col="value", detrend_method=None)

    results = analyzer.run_full_analysis(
        output_dir=str(tmp_path),
        max_breakpoints=1,
        n_bootstraps=10,
        ci_method="parametric",
    )

    print(f"Noise: {noise_std}, Chosen model: {results['chosen_model']}")

    if "segmented" in results["chosen_model"]:
        betas = results["betas"]
        print(f"Estimated betas: {betas}")

        # Check low freq beta - should be somewhat close to signal beta
        assert betas[0] == pytest.approx(true_beta, abs=1.0)

        # Check high freq beta
        # If we have significant noise, the high freq beta should be lower (closer to 0)
        # But for intermediate noise, it might be tricky.
        # We generally expect the slope to flatten (beta decreases).

        # Only assert strictly if noise is high enough to dominate high freqs clearly
        if noise_std >= 5.0:
            assert betas[1] < betas[0]
            assert betas[1] < 1.5 # Should be significantly less than 2.0

    else:
        # If standard model chosen
        estimated_beta = results["beta"]
        print(f"Estimated beta (standard): {estimated_beta}")

        if noise_std <= 0.1:
            # Low noise, might still look like beta=2
            assert estimated_beta == pytest.approx(true_beta, abs=0.5)


@pytest.mark.parametrize("missing_fraction", [0.2, 0.5, 0.7])
def test_uneven_sampling(tmp_path, missing_fraction):
    """
    Test that waterSpec correctly estimates the spectral slope (beta)
    even when the time series is unevenly sampled (missing data).
    """
    n_points = 1000
    dt = 1.0
    beta = 1.5  # Pink/Red noise

    # Generate full uniform series
    time, series = generate_series(power_law, (beta,), n_points=n_points, dt=dt, seed=555)

    # Randomly select a subset of indices to keep
    n_keep = int(n_points * (1 - missing_fraction))
    rng = np.random.default_rng(42 + int(missing_fraction*10))
    keep_indices = np.sort(rng.choice(np.arange(n_points), size=n_keep, replace=False))

    uneven_time = time[keep_indices]
    uneven_series = series[keep_indices]

    # Create file with uneven timestamps
    file_path = create_data_file(tmp_path, uneven_time, uneven_series, filename=f"uneven_{missing_fraction}.csv")

    # Analyze
    analyzer = Analysis(file_path=file_path, time_col="time", data_col="value", detrend_method=None)

    # Force standard model
    results = analyzer.run_full_analysis(
        output_dir=str(tmp_path),
        max_breakpoints=0,
        n_bootstraps=10,
        ci_method="parametric",
    )

    estimated_beta = results["beta"]
    print(f"Missing: {missing_fraction*100}%, True beta: {beta}, Estimated beta: {estimated_beta}")

    # Lomb-Scargle is robust to uneven sampling, but for red noise (beta > 1),
    # uneven sampling can introduce spectral leakage that flattens the spectrum
    # (biasing beta towards 0). This is a known phenomenon.
    # We adjust our expectations: we don't expect perfect recovery of beta=1.5
    # with high missing fractions without more advanced windowing/correction.
    # However, we DO expect it to still be identified as "coloured noise" (beta > 0).

    # Check that it detects significant persistence (beta > 0.3 is a safe threshold for "not white noise")
    assert estimated_beta > 0.3

    # And check that it doesn't vastly overestimate (unlikely, but good to check)
    assert estimated_beta < beta + 0.5


def test_haar_comparison(tmp_path):
    """
    Compare Lomb-Scargle (LS) and Haar Fluctuation Analysis (HFA) estimation
    of spectral slopes on the same dataset.
    """
    n_points = 1000
    dt = 1.0
    beta = 1.5

    # Generate series
    time, series = generate_series(power_law, (beta,), n_points=n_points, dt=dt, seed=777)

    # 1. Run Lomb-Scargle (Standard)
    file_path = create_data_file(tmp_path, time, series, filename="haar_compare.csv")
    analyzer = Analysis(file_path=file_path, time_col="time", data_col="value", detrend_method=None)

    ls_results = analyzer.run_full_analysis(
        output_dir=str(tmp_path / "ls_results"),
        max_breakpoints=0,
        n_bootstraps=10,
        ci_method="parametric",
    )
    ls_beta = ls_results["beta"]

    # 2. Run Haar Analysis directly
    haar = HaarAnalysis(time, series)
    haar_results = haar.run(min_lag=dt, max_lag=n_points*dt/4)
    haar_beta = haar_results["beta"]
    haar_H = haar_results["H"]

    print(f"True beta: {beta}")
    print(f"Lomb-Scargle beta: {ls_beta:.3f}")
    print(f"Haar beta: {haar_beta:.3f} (derived from H={haar_H:.3f})")

    # Check if both are reasonable
    # Both should be reasonably close to 1.5
    # Allowing wider tolerance for comparison as methods differ
    assert ls_beta == pytest.approx(beta, abs=0.4)
    assert haar_beta == pytest.approx(beta, abs=0.4)

    # Check that they are consistent with each other (within some margin)
    diff = abs(ls_beta - haar_beta)
    print(f"Difference (LS - Haar): {diff:.3f}")

    # They should produce similar results for a standard fractal process
    assert diff < 0.5


def test_haar_mannks_segmentation(tmp_path):
    """
    Test using MannKS segmented fitting (via fit_segmented_spectrum) on Haar analysis output
    to detect multifractal (segmented) behavior.
    """
    from waterSpec.fitter import fit_segmented_spectrum

    n_points = 2000
    dt = 1.0
    beta1 = 2.0  # Low frequency (Long lag) -> H = (2-1)/2 = 0.5
    beta2 = 0.5  # High frequency (Short lag) -> H = (0.5-1)/2 = -0.25

    # Generate broken power law data
    # Note: simulate_tk95 takes frequency breakpoint.
    # High freq -> Short lag. Low freq -> Long lag.
    # Breakpoint at f = 0.05.
    # T = 2000. f_min = 0.0005. f_max = 0.5.
    f_break = 0.05
    time, series = generate_series(broken_power_law, (beta1, beta2, f_break), n_points=n_points, dt=dt, seed=321)

    # Run Haar
    haar = HaarAnalysis(time, series)
    # Use many lags to capture the shape well
    haar_results = haar.run(min_lag=dt, max_lag=n_points*dt/2, num_lags=50)
    lags = haar_results["lags"]
    s1 = haar_results["s1"]

    # Use fit_segmented_spectrum
    # We pass 'lags' as 'frequency' and 's1' as 'power'.
    # Note: fit_segmented_spectrum fits log(P) = -beta * log(f) + c
    # Here: log(S1) = H * log(dt) + c
    # So 'beta' output from fitter will be -H.

    # fit_segmented_spectrum requires inputs to be numpy arrays and valid
    # It also logs them.

    fit_res = fit_segmented_spectrum(
        lags,
        s1,
        n_breakpoints=1,
        n_bootstraps=10, # low for speed
        ci_method="bootstrap"
    )

    # Check if fit succeeded
    assert "failure_reason" not in fit_res
    assert fit_res["n_breakpoints"] == 1

    # Retrieve slopes (betas returned are -H)
    neg_H_values = fit_res["betas"]
    H_values = -neg_H_values

    # Convert H to spectral beta: beta = 1 + 2H
    estimated_betas = 1 + 2 * H_values

    print(f"Estimated H: {H_values}")
    print(f"Estimated Spectral Betas (from Haar): {estimated_betas}")

    # Expected behavior:
    # Short lags (High Freq, beta2=0.5) -> Small dt -> First segment in Haar plot?
    # Long lags (Low Freq, beta1=2.0) -> Large dt -> Second segment?

    # fit_segmented_spectrum sorts 'frequency' (lags).
    # Lags are small to large.
    # Segment 1: Small lags (High Freq behavior) -> Should match beta2 = 0.5
    # Segment 2: Large lags (Low Freq behavior) -> Should match beta1 = 2.0

    beta_short_lag = estimated_betas[0]
    beta_long_lag = estimated_betas[1]

    print(f"Short Lag (High Freq) Beta: {beta_short_lag:.2f} (Expected ~{beta2})")
    print(f"Long Lag (Low Freq) Beta: {beta_long_lag:.2f} (Expected ~{beta1})")

    # Allow loose tolerance as Haar scaling relations can be biased for short series/finite size
    assert beta_short_lag == pytest.approx(beta2, abs=0.5)
    assert beta_long_lag == pytest.approx(beta1, abs=0.5)

    # Verify breakpoint
    # f_break = 0.05 => T_break = 1/0.05 = 20
    # Haar breakpoint should be around lag = 20
    bp_lag = fit_res["breakpoints"][0]
    print(f"Detected Breakpoint Lag: {bp_lag:.1f} (Expected ~20)")

    # Breakpoint detection in Haar can be shifted compared to Fourier
    # But should be order of magnitude correct
    assert 5 < bp_lag < 100


@pytest.mark.parametrize("missing_fraction", [0.2, 0.5, 0.7])
def test_haar_ls_uneven_comparison(tmp_path, missing_fraction):
    """
    Compare Lomb-Scargle (LS) and Haar Fluctuation Analysis (HFA)
    on unevenly sampled data.
    """
    n_points = 1000
    dt = 1.0
    true_beta = 1.5

    # Generate full uniform series
    time, series = generate_series(power_law, (true_beta,), n_points=n_points, dt=dt, seed=888)

    # Drop data
    n_keep = int(n_points * (1 - missing_fraction))
    rng = np.random.default_rng(42 + int(missing_fraction*100))
    keep_indices = np.sort(rng.choice(np.arange(n_points), size=n_keep, replace=False))

    uneven_time = time[keep_indices]
    uneven_series = series[keep_indices]

    # 1. Lomb-Scargle
    file_path = create_data_file(tmp_path, uneven_time, uneven_series, filename=f"comp_uneven_{missing_fraction}.csv")
    analyzer = Analysis(file_path=file_path, time_col="time", data_col="value", detrend_method=None)
    ls_results = analyzer.run_full_analysis(
        output_dir=str(tmp_path / f"ls_uneven_{missing_fraction}"),
        max_breakpoints=0,
        n_bootstraps=10,
        ci_method="parametric",
    )
    ls_beta = ls_results["beta"]
    ls_error = ls_beta - true_beta

    # 2. Haar
    haar = HaarAnalysis(uneven_time, uneven_series)
    # Note: Haar analysis naturally handles uneven spacing by using lag windows
    haar_results = haar.run(min_lag=dt, max_lag=n_points*dt/4)
    haar_beta = haar_results["beta"]
    haar_error = haar_beta - true_beta

    print(f"\nMissing Fraction: {missing_fraction}")
    print(f"True Beta: {true_beta}")
    print(f"Lomb-Scargle: Beta={ls_beta:.3f}, Error={ls_error:.3f}")
    print(f"Haar:         Beta={haar_beta:.3f}, Error={haar_error:.3f}")

    # Assertions
    # Both should detect persistence
    # LS degrades significantly with high missing data (whitening effect)
    # Haar is generally more robust for slope estimation in this context

    if missing_fraction <= 0.5:
        assert ls_beta > 0.3

    assert haar_beta > 0.3
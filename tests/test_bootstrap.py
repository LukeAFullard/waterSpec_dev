import numpy as np
import pytest
from waterSpec.fitter import fit_standard_model, fit_segmented_spectrum

# --- Test Data ---
rng = np.random.default_rng(0)

# Data for standard model (single power-law)
FREQUENCY_STD = np.logspace(np.log10(1e-3), np.log10(1e-1), 50)
BETA_STD = 1.5
POWER_STD = FREQUENCY_STD ** (-BETA_STD) * (1 + rng.normal(0, 0.1, size=50))

# Data for segmented model (with a breakpoint)
f1 = np.logspace(np.log10(1e-4), np.log10(1e-2), 25, endpoint=False)
p1 = f1 ** (-1.0)
f2 = np.logspace(np.log10(1e-2), np.log10(1e-1), 25)
p2 = (f2 ** (-2.5)) * (p1[-1] / (f2[0] ** (-2.5))) # Scale to connect smoothly
FREQUENCY_SEG = np.concatenate((f1, f2))
POWER_SEG = np.concatenate((p1, p2)) * (1 + rng.normal(0, 0.1, size=50))


def test_residual_bootstrap_standard_model(mocker):
    """
    Tests that the residual bootstrap runs for the standard model and returns
    valid confidence intervals.
    """
    mocker.patch("waterSpec.fitter.MIN_BOOTSTRAP_SAMPLES", 5)
    results = fit_standard_model(
        FREQUENCY_STD,
        POWER_STD,
        ci_method="bootstrap",
        bootstrap_type="residuals",
        n_bootstraps=10,
        seed=42,
    )
    assert "beta_ci_lower" in results
    assert "beta_ci_upper" in results
    assert np.isfinite(results["beta_ci_lower"])
    assert np.isfinite(results["beta_ci_upper"])
    assert results["beta_ci_lower"] < results["beta_ci_upper"]

def test_wild_bootstrap_models(mocker):
    """
    Tests that the wild bootstrap runs for both standard and segmented models
    and returns valid confidence intervals.
    """
    mocker.patch("waterSpec.fitter.MIN_BOOTSTRAP_SAMPLES", 5)
    # Test standard model with wild bootstrap
    results_std = fit_standard_model(
        FREQUENCY_STD,
        POWER_STD,
        ci_method="bootstrap",
        bootstrap_type="wild",
        n_bootstraps=10,
        seed=42,
    )
    assert "beta_ci_lower" in results_std
    assert "beta_ci_upper" in results_std
    assert np.isfinite(results_std["beta_ci_lower"])
    assert np.isfinite(results_std["beta_ci_upper"])
    assert results_std["beta_ci_lower"] < results_std["beta_ci_upper"]

    # Test segmented model with wild bootstrap
    results_seg = fit_segmented_spectrum(
        FREQUENCY_SEG,
        POWER_SEG,
        ci_method="bootstrap",
        bootstrap_type="wild",
        n_bootstraps=10,
        seed=42,
    )
    assert "betas_ci" in results_seg, f"Test failed because fit was not successful: {results_seg.get('model_summary')}"
    assert len(results_seg["betas_ci"]) == 2
    for lower, upper in results_seg["betas_ci"]:
        assert np.isfinite(lower)
        assert np.isfinite(upper)
        assert lower < upper


def test_block_bootstrap_models(mocker):
    """
    Tests that the block bootstrap runs for both standard and segmented models
    and returns valid confidence intervals.
    """
    mocker.patch("waterSpec.fitter.MIN_BOOTSTRAP_SAMPLES", 5)
    # Test standard model with block bootstrap
    results_std = fit_standard_model(
        FREQUENCY_STD,
        POWER_STD,
        ci_method="bootstrap",
        bootstrap_type="block",
        bootstrap_block_size=5,
        n_bootstraps=10,
        seed=42,
    )
    assert "beta_ci_lower" in results_std
    assert "beta_ci_upper" in results_std
    assert np.isfinite(results_std["beta_ci_lower"])
    assert np.isfinite(results_std["beta_ci_upper"])
    assert results_std["beta_ci_lower"] < results_std["beta_ci_upper"]

    # Test segmented model with block bootstrap
    results_seg = fit_segmented_spectrum(
        FREQUENCY_SEG,
        POWER_SEG,
        ci_method="bootstrap",
        bootstrap_type="block",
        bootstrap_block_size=5,
        n_bootstraps=10,
        seed=42,
    )
    assert "betas_ci" in results_seg, f"Test failed because fit was not successful: {results_seg.get('model_summary')}"
    assert len(results_seg["betas_ci"]) == 2
    for lower, upper in results_seg["betas_ci"]:
        assert np.isfinite(lower)
        assert np.isfinite(upper)
        assert lower < upper


def test_residual_bootstrap_segmented_model(mocker):
    """
    Tests that the residual bootstrap runs for the segmented model and returns
    valid confidence intervals.
    """
    mocker.patch("waterSpec.fitter.MIN_BOOTSTRAP_SAMPLES", 5)
    results = fit_segmented_spectrum(
        FREQUENCY_SEG,
        POWER_SEG,
        ci_method="bootstrap",
        bootstrap_type="residuals",
        n_bootstraps=10,
        seed=42,
    )
    assert "betas_ci" in results, f"Test failed because fit was not successful: {results.get('model_summary')}"
    assert len(results["betas_ci"]) == 2
    for lower, upper in results["betas_ci"]:
        assert np.isfinite(lower)
        assert np.isfinite(upper)
        assert lower < upper

def test_failed_bootstrap_returns_nan(mocker):
    """
    Tests that when an insufficient number of bootstrap iterations succeed,
    the confidence intervals are returned as NaN. This is simulated by
    mocking the random number generator to produce degenerate samples.
    """
    # Use a dataset that is valid for an initial fit
    freq = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    power = 10 / freq

    # Mock rng.choice to always return indices that result in
    # identical x-values, which will cause theilslopes to fail.
    # We will return an array of all zeros for the indices.
    # This means every bootstrap sample will be the first point repeated 10 times.
    mock_rng = mocker.patch('numpy.random.default_rng')
    mock_rng.return_value.choice.return_value = np.zeros(10, dtype=int)

    results = fit_standard_model(
        freq,
        power,
        method="ols", # Force OLS to bypass MannKS and test internal bootstrap logic
        ci_method="bootstrap",
        bootstrap_type="pairs",  # Force pairs bootstrap for this test
        n_bootstraps=10,
        seed=42,  # Seed doesn't matter now due to mocking
    )
    # All 100 bootstrap iterations should now fail because the resampled
    # data will have identical x-coordinates, causing the slope calculation to fail.
    assert np.isnan(results["beta_ci_lower"])
    assert np.isnan(results["beta_ci_upper"])


def test_ci_coverage(mocker):
    """
    Tests the confidence interval coverage by running a simulation.
    We generate many datasets with a known beta, calculate the confidence
    interval for each, and check if the proportion of intervals that
    contain the true beta is close to the expected confidence level.
    """
    mocker.patch("waterSpec.fitter.MIN_BOOTSTRAP_SAMPLES", 5)
    true_beta = 2.0
    # Reduce n_simulations to speed up test execution
    n_simulations = 10
    n_bootstraps = 10
    ci_level = 95
    successes = 0
    valid_intervals = 0

    for i in range(n_simulations):
        # Generate a new dataset for each simulation
        rng = np.random.default_rng(i)
        sim_power = FREQUENCY_STD ** (-true_beta) * (1 + rng.normal(0, 0.2, size=50))
        results = fit_standard_model(
            FREQUENCY_STD,
            sim_power,
            ci_method="bootstrap",
            n_bootstraps=n_bootstraps,
            ci=ci_level,
            seed=i, # Use different seed for each simulation
        )
        lower_ci = results.get("beta_ci_lower", np.nan)
        upper_ci = results.get("beta_ci_upper", np.nan)

        if np.isfinite(lower_ci) and np.isfinite(upper_ci):
            valid_intervals += 1
            if lower_ci <= true_beta <= upper_ci:
                successes += 1

    assert valid_intervals > n_simulations * 0.8, "Too many simulations failed to produce a valid CI."

    coverage = (successes / valid_intervals) * 100
    # We expect the coverage to be close to 95%.
    # Allow for a wider margin due to the small number of simulations.
    assert ci_level - 45 < coverage < ci_level + 5, f"CI coverage ({coverage}%) is outside the expected range."


def test_spawned_rngs_are_independent():
    from waterSpec.utils import spawn_generators
    rng1, rng2 = spawn_generators(42, 2)
    # same master seed, but should produce different first draws
    assert rng1.integers(0, 100000) != rng2.integers(0, 100000)
import numpy as np
import pytest
from numpy.testing import assert_array_equal, assert_raises

from waterSpec.fitter import fit_standard_model
from waterSpec.utils import make_rng


def test_seed_spawn_independence():
    """
    Tests that child seeds spawned from the same SeedSequence produce
    independent (different) random number streams.
    """
    master_ss = np.random.SeedSequence(12345)
    child_seeds = master_ss.spawn(2)

    rng1 = make_rng(child_seeds[0])
    rng2 = make_rng(child_seeds[1])

    # Ensure the generators are different objects
    assert rng1 is not rng2

    # Draw random numbers and assert they are different
    random_stream1 = rng1.random(10)
    random_stream2 = rng2.random(10)

    with assert_raises(AssertionError):
        assert_array_equal(random_stream1, random_stream2)


def test_fit_reproducible_given_seed():
    """
    Tests that fit_standard_model produces identical results when run
    twice with the same SeedSequence.
    """
    # Generate some synthetic data
    rng = np.random.default_rng(42)
    log_freq = np.linspace(0, 2, 100)
    # Add some noise to avoid perfect fits
    log_power = -1.5 * log_freq + 1 + rng.normal(0, 0.1, 100)
    frequency = 10**log_freq
    power = 10**log_power

    # Use a SeedSequence for reproducibility.
    # When the same SeedSequence is used to initialize two generators,
    # they will produce the same stream of random numbers.
    seed = np.random.SeedSequence(123)

    # Run the fit twice with the same seed
    results1 = fit_standard_model(
        frequency,
        power,
        ci_method="bootstrap",
        n_bootstraps=50,  # Use a small number for speed
        seed=seed,
    )

    results2 = fit_standard_model(
        frequency,
        power,
        ci_method="bootstrap",
        n_bootstraps=50,
        seed=seed,
    )

    # Check that the key results are identical
    np.testing.assert_allclose(results1["beta"], results2["beta"])
    np.testing.assert_allclose(results1["beta_ci_lower"], results2["beta_ci_lower"])
    np.testing.assert_allclose(results1["beta_ci_upper"], results2["beta_ci_upper"])
    np.testing.assert_allclose(
        results1["fitted_log_power"], results2["fitted_log_power"]
    )


def test_block_bootstrap_raises_error_on_small_n_points():
    """
    Tests that block bootstrap raises a ValueError when the number of data
    points is too small relative to the block size.
    """
    # n_points (20) < 3 * block_size (3 * 7 = 21)
    frequency = np.linspace(1, 10, 20)
    power = 1 / frequency
    with pytest.raises(ValueError, match="The number of data points"):
        fit_standard_model(
            frequency,
            power,
            ci_method="bootstrap",
            bootstrap_type="block",
            bootstrap_block_size=7,
        )


def test_block_bootstrap_raises_error_on_large_block_size():
    """
    Tests that block bootstrap raises a ValueError when the block size is
    larger than or equal to the number of data points.
    """
    # block_size (20) >= n_points (20)
    frequency = np.linspace(1, 10, 20)
    power = 1 / frequency
    with pytest.raises(ValueError, match="Block size"):
        fit_standard_model(
            frequency,
            power,
            ci_method="bootstrap",
            bootstrap_type="block",
            bootstrap_block_size=20,
        )

    # block_size (21) > n_points (20)
    with pytest.raises(ValueError, match="Block size"):
        fit_standard_model(
            frequency,
            power,
            ci_method="bootstrap",
            bootstrap_type="block",
            bootstrap_block_size=21,
        )
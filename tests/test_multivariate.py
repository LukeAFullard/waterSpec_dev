
import numpy as np
import pytest
from waterSpec.multivariate import calculate_partial_cross_haar, calculate_multivariate_fluctuations

def generate_spurious_correlation(n=1000):
    np.random.seed(42)
    time = np.arange(n)

    # Common cause Z
    Z = np.sin(2 * np.pi * time / 50) + np.random.normal(0, 0.1, n)

    # X and Y derived from Z
    X = Z + np.random.normal(0, 0.5, n)
    Y = Z + np.random.normal(0, 0.5, n)

    return time, X, Y, Z

def generate_direct_correlation(n=1000):
    np.random.seed(42)
    time = np.arange(n)

    # Independent Z
    Z = np.random.normal(0, 1, n)

    # X drives Y
    X = np.sin(2 * np.pi * time / 50) + np.random.normal(0, 0.1, n)
    Y = X + np.random.normal(0, 0.5, n)

    return time, X, Y, Z

def test_partial_cross_haar_spurious():
    time, X, Y, Z = generate_spurious_correlation()

    lags = np.array([10, 25, 50])

    results = calculate_partial_cross_haar(
        time, X, Y, Z, lags, overlap=True
    )

    # Check correlations
    rho_xy = results['rho_xy']
    partial = results['partial_corr']

    print(f"Spurious - Rho_XY: {rho_xy}, Partial: {partial}")

    # Rho_XY should be high
    assert np.all(rho_xy > 0.5)

    # Partial correlation should be low (ideally 0, but noise makes it non-zero)
    # Removing Z should remove most of the correlation
    assert np.all(np.abs(partial) < 0.3)
    assert np.all(np.abs(partial) < np.abs(rho_xy))

def test_partial_cross_haar_direct():
    time, X, Y, Z = generate_direct_correlation()

    lags = np.array([10, 25, 50])

    results = calculate_partial_cross_haar(
        time, X, Y, Z, lags, overlap=True
    )

    rho_xy = results['rho_xy']
    partial = results['partial_corr']

    print(f"Direct - Rho_XY: {rho_xy}, Partial: {partial}")

    # Rho_XY should be high
    assert np.all(rho_xy > 0.5)

    # Partial correlation should remain high because Z explains nothing
    assert np.all(partial > 0.5)
    # Should be close to original correlation
    assert np.all(np.abs(partial - rho_xy) < 0.2)

def test_multivariate_fluctuations_structure():
    # Smoke test for structure
    time = np.arange(100)
    d1 = np.random.randn(100)
    d2 = np.random.randn(100)
    d3 = np.random.randn(100)

    lags = np.array([10])
    results = calculate_partial_cross_haar(time, d1, d2, d3, lags)

    assert len(results['lags']) == 1
    assert 'rho_xy' in results

def test_multivariate_fluctuations_basic():
    time = np.arange(20)
    # Series 1 is constant, fluctuation should be 0
    d1 = np.ones(20)
    # Series 2 is linear, fluc over lag tau should be tau/2
    d2 = np.arange(20, dtype=float)

    lags = np.array([4])
    results = calculate_multivariate_fluctuations(
        time, [d1, d2], lags, overlap=True, min_samples_per_window=2
    )

    assert 4 in results
    flucs = results[4]
    assert len(flucs) == 2
    # For lag 4, half-window size is 2.
    # d1 fluc should be all 0
    assert np.allclose(flucs[0], 0.0)
    # d2 is linear y=x. mean of [2, 3] - mean of [0, 1] = 2.5 - 0.5 = 2.0 = tau/2
    assert np.allclose(flucs[1], 2.0)

def test_multivariate_fluctuations_mismatched_lengths():
    time = np.arange(10)
    d1 = np.arange(10)
    d2 = np.arange(11) # Mismatched length
    lags = np.array([4])

    with pytest.raises(ValueError, match="Dataset 1 length \\(11\\) does not match time array length \\(10\\)."):
        calculate_multivariate_fluctuations(time, [d1, d2], lags)

def test_multivariate_fluctuations_no_valid_windows():
    time = np.arange(10)
    d1 = np.arange(10)
    # Lag is larger than the entire time series
    lags = np.array([20])

    results = calculate_multivariate_fluctuations(
        time, [d1], lags, min_samples_per_window=2
    )

    assert 20 in results
    # Should return an empty array for the dataset
    assert len(results[20][0]) == 0

def test_multivariate_fluctuations_non_overlapping():
    time = np.arange(10)
    d1 = np.arange(10)

    lags = np.array([4])
    # non-overlapping steps forward by tau=4
    # Window 1: [0, 4]
    # Window 2: [4, 8]
    # Window 3: [8, 12] - exceeds time[-1]=9

    results = calculate_multivariate_fluctuations(
        time, [d1], lags, overlap=False, min_samples_per_window=2
    )

    assert 4 in results
    # Expect 2 windows
    assert len(results[4][0]) == 2
    # Fluctuation for linear data should be tau/2 = 2.0
    assert np.allclose(results[4][0], 2.0)

def test_multivariate_fluctuations_statistics():
    time = np.arange(10)
    d1 = np.array([1, 10, 2, 20, 3, 30, 4, 40, 5, 50])

    lags = np.array([4])

    # Using 'median' statistic
    results_median = calculate_multivariate_fluctuations(
        time, [d1], lags, overlap=False, statistic="median", min_samples_per_window=2
    )

    # Window 1: left=[1, 10], right=[2, 20] -> diff = median([2, 20]) - median([1, 10]) = 11.0 - 5.5 = 5.5
    # Window 2: left=[3, 30], right=[4, 40] -> diff = median([4, 40]) - median([3, 30]) = 22.0 - 16.5 = 5.5
    assert np.allclose(results_median[4][0], 5.5)

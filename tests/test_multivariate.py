
import numpy as np
import pytest
from waterSpec.multivariate import calculate_partial_cross_haar

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

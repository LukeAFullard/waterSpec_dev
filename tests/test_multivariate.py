
import numpy as np
import pytest
from waterSpec.multivariate import MultivariateAnalysis

def generate_chain_structure(n=1000):
    """
    Generates X -> Y -> Z chain.
    X causes Y. Y causes Z.
    X and Z are correlated only via Y.
    Partial correlation Corr(X, Z | Y) should be ~ 0.
    """
    np.random.seed(42)
    time = np.arange(n)

    # X: Random walk
    X = np.cumsum(np.random.randn(n))

    # Y = 0.8*X + noise
    Y = 0.8 * X + 10 * np.random.randn(n)

    # Z = 0.8*Y + noise
    Z = 0.8 * Y + 10 * np.random.randn(n)

    return time, X, Y, Z

def test_alignment_three_vars():
    t1 = np.array([0, 10, 20])
    d1 = np.array([1, 2, 3])

    t2 = np.array([0, 11, 21])
    d2 = np.array([10, 20, 30])

    t3 = np.array([1, 10, 19])
    d3 = np.array([100, 200, 300])

    inputs = [
        {'time': t1, 'data': d1, 'name': 'V1'},
        {'time': t2, 'data': d2, 'name': 'V2'},
        {'time': t3, 'data': d3, 'name': 'V3'}
    ]

    multi = MultivariateAnalysis(inputs)
    aligned = multi.align_data(tolerance=2)

    # Expected alignment (tol=2):
    # t=0: V1=1, V2=10, V3=100 (t3=1 matches t1=0 diff 1)
    # t=10: V1=2, V2=20 (t2=11), V3=200 (t3=10)
    # t=20: V1=3, V2=30 (t2=21), V3=300 (t3=19)

    assert len(aligned) == 3
    assert aligned.iloc[0]['V3'] == 100
    assert aligned.iloc[1]['V2'] == 20

def test_partial_correlation_chain():
    t, X, Y, Z = generate_chain_structure(n=5000) # Need many points for stable Haar corr

    inputs = [
        {'time': t, 'data': X, 'name': 'X'},
        {'time': t, 'data': Y, 'name': 'Y'},
        {'time': t, 'data': Z, 'name': 'Z'}
    ]

    multi = MultivariateAnalysis(inputs)
    # Use a safe tolerance for integer time steps
    multi.align_data(tolerance=0.1)

    # Check simple correlation X-Z first (should be high)
    # At lag=1, fluctuations approximate differencing.
    # dX ~ white noise.
    # dY ~ 0.8 dX + white noise.
    # dZ ~ 0.8 dY + white noise.
    # So fluctuations are correlated.

    lags = np.array([10, 20, 50])

    # We want partial correlation of X and Z given Y.
    res = multi.run_partial_cross_haar_analysis(
        target_var1='X',
        target_var2='Z',
        conditioning_vars=['Y'],
        lags=lags
    )

    p_corrs = np.array(res['partial_correlation'])

    print(f"Partial Correlations (X, Z | Y): {p_corrs}")

    # Should be close to 0
    assert np.all(np.abs(p_corrs) < 0.2)

    # For comparison, check X-Z without conditioning?
    # We can run partial without conditioning (empty list) effectively
    # But logic requires at least 1? No, logic builds matrix of all vars.
    # If we pass conditioning=[], all_vars=[X, Z]. Matrix 2x2.
    # P = inv(C).
    # rho_12 = -P12 / sqrt(P11 P22).
    # For 2x2, this is exactly the standard correlation (with sign flip maybe? No).
    # Let's verify standard correlation is high.

    res_std = multi.run_partial_cross_haar_analysis(
        target_var1='X',
        target_var2='Z',
        conditioning_vars=[], # No conditioning
        lags=lags
    )

    std_corrs = np.array(res_std['partial_correlation'])
    print(f"Standard Correlations (X, Z): {std_corrs}")

    # Should be significantly higher than partial
    assert np.mean(std_corrs) > np.mean(p_corrs) + 0.1

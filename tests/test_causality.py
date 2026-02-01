
import numpy as np
import pytest
from waterSpec.causality import convergent_cross_mapping, find_optimal_embedding

def test_ccm_causality():
    """
    Test CCM on a simple coupled system where X -> Y.
    Y[t] = X[t-1] + ...
    So Y is the Effect, X is the Cause.
    We predict Cause (X) using Effect (Y) manifold.
    So Y manifold -> Predict X should be good.
    """
    # Generate simple coupled data
    T = 100
    X = np.sin(np.linspace(0, 20, T))
    # Y is driven by X with lag 1
    Y = np.roll(X, 1) + 0.1 * np.random.randn(T)
    Y[0] = 0

    time = np.arange(T)

    # Check X -> Y (X causes Y)
    # Using Y (effect) to predict X (cause)
    res = convergent_cross_mapping(time, X=X, Y=Y, E=2, tau=1)

    rho = res['rho']
    # Correlation should be positive and high at end
    assert rho[-1] > 0.5
    # RMSE should decrease (optional check)

def test_optimal_embedding():
    # Use a chaotic map (logistic map) or something with structure
    # For a simple sine wave, E=2 is sufficient (circle).
    data = np.sin(np.linspace(0, 20, 100))

    # Now prediction horizon is 1 step ahead
    E_opt = find_optimal_embedding(data, max_E=5, tp=1)

    assert E_opt >= 1
    assert E_opt <= 5

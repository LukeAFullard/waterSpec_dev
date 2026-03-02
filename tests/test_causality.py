
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

def test_optimal_embedding_short_data():
    """Test that find_optimal_embedding handles very short data correctly."""
    # Data is too short for any embedding to be valid
    short_data = np.arange(5)
    with pytest.raises(ValueError, match="Time series too short"):
        find_optimal_embedding(short_data, max_E=5, tp=1)

def test_optimal_embedding_invalid_tp():
    """Test that finding optimal embedding with invalid tp fails."""
    data = np.sin(np.linspace(0, 20, 100))
    with pytest.raises(ValueError, match="Prediction horizon tp must be >= 1"):
        find_optimal_embedding(data, max_E=5, tp=0)

def test_irregular_sampling_ccm():
    """
    Test that CCM handles irregular sampling by interpolating.
    We create a regular system, then subsample it irregularly.
    """
    # 1. Create high-res regular system
    t_fine = np.linspace(0, 50, 500)
    X_fine = np.sin(t_fine)
    Y_fine = np.cos(t_fine) # Coupled

    # 2. Subsample irregularly
    rng = np.random.default_rng(42)
    indices = np.sort(rng.choice(len(t_fine), 100, replace=False))

    t_irr = t_fine[indices]
    X_irr = X_fine[indices]
    Y_irr = Y_fine[indices]

    # Ensure it is actually irregular
    dt = np.diff(t_irr)
    assert not np.allclose(dt, np.median(dt))

    # 3. Run CCM
    # Should warn about interpolation if allowed
    with pytest.warns(UserWarning, match="unevenly sampled"):
        res = convergent_cross_mapping(t_irr, X_irr, Y_irr, E=2, tau=1, allow_interpolation=True)

    # Should still give good result
    assert res['rho'][-1] > 0.8

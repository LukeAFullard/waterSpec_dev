
import numpy as np
import pytest
from waterSpec.haar_analysis import HaarAnalysis, calculate_haar_fluctuations

def test_intermittency_calculation():
    # 1. Monofractal Case (White Noise)
    # K(2) should be close to 0. Beta should be close to 0 (since 2H ~ -1, beta ~ 0. Wait.)
    # White noise: Beta = 0. H = -0.5.
    # S1 ~ dt^(-0.5). S2 ~ dt^(-1.0).
    # zeta1 = -0.5. zeta2 = -1.0.
    # K(2) = 2*(-0.5) - (-1.0) = -1 + 1 = 0.
    # Beta_multi = 1 + 2(-0.5) - 0 = 0.

    np.random.seed(42)
    time = np.arange(1000)
    data = np.random.standard_normal(1000)

    ha = HaarAnalysis(time, data)
    res = ha.run(calc_intermittency=True)

    assert "K2" in res
    assert "beta_multifractal" in res
    assert np.isclose(res["K2"], 0.0, atol=0.2) # Allow some statistical fluctuation
    assert np.isclose(res["beta_multifractal"], 0.0, atol=0.2)

def test_intermittency_manual_call():
    # Test calling calculate_intermittency manually
    time = np.arange(100)
    data = np.random.standard_normal(100)
    ha = HaarAnalysis(time, data)

    # Should fail if run() not called
    with pytest.raises(ValueError, match="Run standard analysis first"):
        ha.calculate_intermittency()

    ha.run()
    K2 = ha.calculate_intermittency()
    assert K2 is not None
    assert ha.K2 is not None

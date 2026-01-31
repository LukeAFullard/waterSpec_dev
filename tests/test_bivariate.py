import numpy as np
import pytest
import pandas as pd
from waterSpec.bivariate import BivariateAnalysis

def generate_correlated_series(n=1000):
    np.random.seed(42)
    time = np.arange(n)

    # Common signal
    signal = np.sin(2 * np.pi * time / 50)

    # Var 1
    data1 = signal + 0.5 * np.random.randn(n)

    # Var 2 (correlated with Var 1)
    data2 = 0.8 * signal + 0.5 * np.random.randn(n)

    return time, data1, time, data2

def test_bivariate_alignment():
    time1 = np.array([0, 10, 20, 30], dtype=float)
    data1 = np.array([1, 2, 3, 4], dtype=float)

    time2 = np.array([0, 11, 19, 31], dtype=float)
    data2 = np.array([10, 20, 30, 40], dtype=float)

    biv = BivariateAnalysis(time1, data1, "V1", time2, data2, "V2")
    aligned = biv.align_data(tolerance=2, method='nearest')

    assert len(aligned) == 4
    # Check values
    # 0 -> 0 (diff 0) -> match
    # 10 -> 11 (diff 1) -> match
    # 20 -> 19 (diff 1) -> match
    # 30 -> 31 (diff 1) -> match

    # Check aligned values specifically
    # V2 corresponding to time1=10 is 20 (from time2=11)
    val_at_10 = aligned.loc[aligned['time'] == 10, 'V2'].values[0]
    assert val_at_10 == 20

def test_cross_haar_correlation():
    t1, d1, t2, d2 = generate_correlated_series()
    biv = BivariateAnalysis(t1, d1, "V1", t2, d2, "V2")
    biv.align_data(tolerance=1)

    lags = np.array([10, 20, 50])
    res = biv.run_cross_haar_analysis(lags)

    corrs = np.array(res['correlation'])
    # Should be positive correlation
    assert np.all(corrs > 0.5)

def test_calculate_significance():
    # Use shorter series for speed
    t1, d1, t2, d2 = generate_correlated_series(n=200)
    biv = BivariateAnalysis(t1, d1, "V1", t2, d2, "V2")
    biv.align_data(tolerance=1)

    lags = np.array([10, 20])
    # With 20 surrogates, min p-value is 1/21 ~ 0.047
    res = biv.calculate_significance(lags, n_surrogates=20, seed=42)

    assert 'p_values' in res
    p_values = res['p_values']

    # Since they are correlated, p-values should be small (significant)
    # We assert < 0.1 to be safe given random noise
    assert np.all(p_values < 0.1)

def test_hysteresis_metrics():
    # Construct a loop
    t = np.linspace(0, 2*np.pi, 100)
    d1 = np.sin(t)
    d2 = np.cos(t) # Circular loop

    biv = BivariateAnalysis(t, d1, "V1", t, d2, "V2")
    biv.align_data(tolerance=0.1)

    hyst = biv.calculate_hysteresis_metrics(tau=0.1)

    assert hyst['area'] != 0
    assert hyst['direction'] in ["Clockwise", "Counter-Clockwise"]

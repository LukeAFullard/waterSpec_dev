
import numpy as np
import pytest
from waterSpec.haar_analysis import HaarAnalysis
from waterSpec.spectral_analyzer import calculate_periodogram
from waterSpec.causality import convergent_cross_mapping
from waterSpec.bivariate import BivariateAnalysis
from waterSpec.surrogates import generate_power_law_surrogates

def generate_irregular_red_noise(n=500, beta=2.0, seed=42):
    """Generates irregularly sampled red noise (Brownian motion)."""
    rng = np.random.default_rng(seed)
    # Irregular time
    t = np.sort(rng.uniform(0, 1000, n))
    # Generate on regular grid then interpolate?
    # Better: Use the exact method we added (Lomb-Scargle surrogates)
    # We can use generate_power_law_surrogates for this!
    y = generate_power_law_surrogates(t, beta=beta, n_surrogates=1, seed=seed)[0]
    return t, y

def test_audit_spectral_accuracy():
    """
    AUDIT CHECK 1: Spectral Slope Accuracy on Irregular Data.
    Legal Defensibility: Haar should work, LS might be biased.
    """
    t, y = generate_irregular_red_noise(beta=2.0, seed=101)

    # 1. Haar Analysis (Should be robust)
    haar = HaarAnalysis(t, y)
    res = haar.run(num_lags=20, n_bootstraps=50)
    beta_haar = res['beta']

    # Allow some tolerance (statistical estimation)
    print(f"Haar Beta: {beta_haar}")
    assert 1.5 < beta_haar < 2.5, "Haar Analysis failed to recover Red Noise slope"

def test_audit_causality_detection():
    """
    AUDIT CHECK 2: Causality in Irregular Coupled System.
    Legal Defensibility: Must distinguish cause from noise.

    We use a continuous smooth system (Sum of Sines driving Non-linear Y)
    because interpolation (required for irregular CCM) fails on discrete maps like Logistic.
    """
    # Continuous Quasi-periodic Driver X
    t_reg = np.linspace(0, 100, 1000) # High res
    x = np.sin(t_reg) + np.sin(np.sqrt(2)*t_reg) + np.sin(np.sqrt(5)*t_reg)

    # Non-linear Response Y driven by X with lag
    # Y(t) = X(t - lag)^3
    # Need to look back.
    lag_idx = 10
    y = np.roll(x, lag_idx)**3
    # Add noise
    y += 0.1 * np.random.normal(size=len(y))

    # Trim edges
    t_reg = t_reg[lag_idx:-lag_idx]
    x = x[lag_idx:-lag_idx]
    y = y[lag_idx:-lag_idx]

    # Subsample irregularly
    rng = np.random.default_rng(202)
    idx = np.sort(rng.choice(len(t_reg), 400, replace=False))
    t_irr = t_reg[idx]
    x_irr = x[idx]
    y_irr = y[idx]

    # Test X causes Y
    # Y is the "Effect", so Y's manifold should predict X
    # Filter warnings about uneven sampling (which we expect)
    with pytest.warns(UserWarning, match="unevenly sampled"):
        res_x_cause_y = convergent_cross_mapping(t_irr, X=x_irr, Y=y_irr, E=3, tau=2)

    rho_final = res_x_cause_y['rho'][-1]
    print(f"CCM Rho (Effect Y predicting Cause X): {rho_final}")

    # Should be high (detection of coupling)
    assert rho_final > 0.6, "CCM failed to detect causal link X->Y in irregular continuous data"

def test_audit_null_hypothesis_validity():
    """
    AUDIT CHECK 3: False Positive Rate on Uncorrelated Red Noise.
    Legal Defensibility: Should NOT find significance between unrelated red noise signals.
    Naive correlation often fails this (spurious correlation).
    """
    # Two independent red noise signals
    t, y1 = generate_irregular_red_noise(beta=2.0, seed=301)
    _, y2 = generate_irregular_red_noise(beta=2.0, seed=302) # Diff seed

    biv = BivariateAnalysis(t, y1, "SiteA", t, y2, "SiteB")
    biv.align_data(tolerance=0.1)

    # Calculate significance of correlation at lag 10
    # Use the new robust method
    lags = np.array([10.0])
    res = biv.calculate_significance(lags, n_surrogates=50, seed=42)

    p_val = res['p_values'][0]
    corr_obs = res['observed_correlation'][0]

    print(f"Observed Corr: {corr_obs}, p-value: {p_val}")

    # We expect p > 0.05 (Not significant)
    # Note: 1/20 chance of failure at 0.05 level.
    # With random seeds, this is probabilistic.
    # But for Red Noise, naive corr is often HIGH, so p-value check protects us.

    # If surrogates work, the distribution of surrogate correlations should cover the observed one.
    assert p_val > 0.01, f"False Positive! Found significant correlation (p={p_val}) between independent red noise."

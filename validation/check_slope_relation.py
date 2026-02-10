
import numpy as np
import matplotlib.pyplot as plt
from waterSpec.haar_analysis import calculate_haar_fluctuations, fit_haar_slope

def generate_lognormal_cascade(N, lambda_scale=2, sigma=0.4):
    """
    Generates a log-normal multifractal cascade (Conservative Measure).
    This mimics a multifractal noise (fGn-like).
    """
    steps = int(np.log(N) / np.log(lambda_scale))
    measure = np.ones(1)

    for _ in range(steps):
        mu = -0.5 * sigma**2
        multipliers = np.exp(mu + sigma * np.random.standard_normal(len(measure) * lambda_scale))
        measure = np.repeat(measure, lambda_scale) * multipliers

    # Add random signs to make it zero-mean noise
    signs = np.random.choice([-1, 1], size=len(measure))
    return measure * signs

def run_check(process_type='fgn', N=8192):
    print(f"\n--- Checking Slope Relation for {process_type.upper()} ---")

    # Generate Data
    noise = generate_lognormal_cascade(N, sigma=0.4)
    if process_type == 'fgn':
        data = noise
    else: # fbm
        data = np.cumsum(noise)

    time = np.arange(N)

    # 1. Calculate Haar Slope m (from S1)
    lags, s1, _, _ = calculate_haar_fluctuations(time, data, statistic="mean", aggregation="mean", overlap=True)
    res_s1 = fit_haar_slope(lags, s1)
    m = res_s1['H'] # Slope of log(S1) vs log(dt)

    # 2. Calculate Haar S2 Slope (zeta2/2)
    lags_rms, s_rms, _, _ = calculate_haar_fluctuations(time, data, statistic="mean", aggregation="rms", overlap=True)
    res_rms = fit_haar_slope(lags_rms, s_rms, n_bootstraps=0)
    m2 = res_rms['H'] # Slope of log(S_rms)
    zeta2 = 2 * m2

    # 3. Calculate K(2) = 2*m - zeta2
    # Wait, K(2) is defined relative to H (scaling exponent of first moment).
    # If m is scaling of first moment, then K(2) = 2*m - zeta2.
    # But usually H refers to the *underlying* scaling (monofractal part).
    # For multifractal: zeta(q) = qH - K(q).
    # zeta(1) = H - K(1).
    # Usually K(1) != 0 for conservative fields?
    # For conservative cascade, K(1) = 0. So zeta(1) = H.
    # So m = H.
    # K(2) = 2H - zeta(2) = 2m - zeta2.
    K2 = 2 * m - zeta2

    # 4. Calculate True Beta (FFT)
    freqs = np.fft.rfftfreq(N)
    psd = np.abs(np.fft.rfft(data))**2
    valid = (freqs > 0) & (freqs < 0.1) # Focus on scaling range
    slope_fft, _ = np.polyfit(np.log10(freqs[valid]), np.log10(psd[valid]), 1)
    beta_true = -slope_fft

    # 5. Check Formulas
    # Standard: Beta = 2m + 1
    beta_std = 2 * m + 1

    # Corrected: Beta = 2m + 1 - K(2)?
    # Or Beta = 1 + zeta(2)?
    # Beta = 1 + zeta(2) = 1 + 2m - K(2).
    # This matches 2m + 1 - K(2).
    beta_corr = 1 + zeta2

    print(f"  m (S1 slope): {m:.4f}")
    print(f"  zeta(2) (S2 slope): {zeta2:.4f}")
    print(f"  K(2) (2m - zeta2): {K2:.4f}")
    print(f"  True Beta (FFT): {beta_true:.4f}")
    print(f"  Beta (Standard 2m+1): {beta_std:.4f}")
    print(f"  Beta (Corrected 1+zeta2): {beta_corr:.4f}")

    err_std = abs(beta_std - beta_true)
    err_corr = abs(beta_corr - beta_true)

    print(f"  Error Standard: {err_std:.4f}")
    print(f"  Error Corrected: {err_corr:.4f}")

    if err_corr < err_std:
        print("  -> Correction Improves Estimation!")
    else:
        print("  -> Correction Does Not Improve Estimation.")

if __name__ == "__main__":
    np.random.seed(42)
    # Check fGn (Intermittent Noise)
    run_check('fgn')
    # Check fBm (Intermittent Walk)
    run_check('fbm')

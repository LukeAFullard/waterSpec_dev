
import numpy as np
import matplotlib.pyplot as plt
from waterSpec.haar_analysis import calculate_haar_fluctuations, fit_haar_slope

def generate_lognormal_cascade(N, lambda_scale=2, sigma=0.5):
    """
    Generates a log-normal multifractal cascade (Measures).
    This produces a highly intermittent field (conservative measure).

    The spectral slope Beta for a conservative cascade is typically 1 - K(2).
    K(2) for log-normal is C1 * (2^2 - 2) / (alpha-1) ... wait.
    For log-normal (alpha=2): K(q) = C1/2 * q^2 (approx?).
    Actually simpler: K(q) = (C1 / (alpha-1)) * (q^alpha - q).
    For alpha=2: K(q) = C1 * (q^2 - q).
    So K(2) = C1 * (4-2) = 2*C1.

    If we construct it with specific mu/sigma:
    M = exp(mu + sigma * Z). E[M] = 1 => mu = -sigma^2/2.
    E[M^q] = exp(q*mu + q^2*sigma^2/2) = exp(-q*sigma^2/2 + q^2*sigma^2/2) = exp(sigma^2/2 * (q^2-q)).
    So K(q) = sigma^2/2 * (q^2 - q) / log(lambda)? No, scaling exponent involves log(lambda).

    Let's just use it to generate intermittent data.
    """
    steps = int(np.log(N) / np.log(lambda_scale))
    measure = np.ones(1)

    for _ in range(steps):
        mu = -0.5 * sigma**2
        multipliers = np.exp(mu + sigma * np.random.standard_normal(len(measure) * lambda_scale))
        measure = np.repeat(measure, lambda_scale) * multipliers

    return measure

def fractional_integration(data, H):
    """
    Fractionally integrates a series (assuming it's noise) to give it spectral slope +2H.
    """
    N = len(data)
    fft = np.fft.rfft(data)
    freqs = np.fft.rfftfreq(N)
    # Filter: f^-H
    # If noise is white (beta=0), integration gives beta=2H.
    # If noise is multifractal (beta = 1 - K(2)), integration gives beta = 1 - K(2) + 2H?

    with np.errstate(divide='ignore'):
        filter_ = np.where(freqs > 0, freqs**(-H), 0)

    fft_filtered = fft * filter_
    return np.fft.irfft(fft_filtered, n=N)

def calculate_intermittency_correction(time, data):
    # 1. Calculate H from S1 (First Order)
    # S1 ~ dt^H (or dt^zeta(1))
    lags, s1, _, _ = calculate_haar_fluctuations(
        time, data, statistic="mean", aggregation="mean", overlap=True
    )
    res_s1 = fit_haar_slope(lags, s1)
    H_est = res_s1['H'] # This is zeta(1)

    # 2. Calculate S2 from RMS
    # S_rms = sqrt(S2) ~ dt^(zeta(2)/2)
    lags, s_rms, _, _ = calculate_haar_fluctuations(
        time, data, statistic="mean", aggregation="rms", overlap=True
    )
    res_rms = fit_haar_slope(lags, s_rms)
    zeta2_half = res_rms['H']
    zeta2 = 2 * zeta2_half

    # 3. Calculate K(2)
    # K(2) = 2*zeta(1) - zeta(2)
    K2 = 2 * H_est - zeta2

    # 4. Calculate Beta from Power Spectrum directly (standard method)
    # Using simple Periodogram for check
    freqs = np.fft.rfftfreq(len(data), d=time[1]-time[0])
    psd = np.abs(np.fft.rfft(data))**2
    # Fit log-log excluding low/high freq
    valid = (freqs > 0) & (freqs < 0.5)
    slope, _ = np.polyfit(np.log10(freqs[valid]), np.log10(psd[valid]), 1)
    beta_fft = -slope

    # 5. Calculate Beta from H (Monofractal assumption)
    beta_mono = 1 + 2 * H_est

    # 6. Calculate Beta from H and K(2) (Multifractal correction)
    # Formula: Beta = 1 + 2H - K(2)
    # Wait, H here usually refers to zeta(1).
    # Beta = 1 + 2*zeta(1) - (2*zeta(1) - zeta(2)) = 1 + zeta(2).
    # This matches the standard definition Beta = 1 + zeta(2).
    beta_multi = 1 + 2 * H_est - K2

    return {
        "H (zeta1)": H_est,
        "zeta2": zeta2,
        "K(2)": K2,
        "Beta (FFT)": beta_fft,
        "Beta (Monofractal 1+2H)": beta_mono,
        "Beta (Multifractal 1+2H-K2)": beta_multi
    }

if __name__ == "__main__":
    np.random.seed(42)
    # Generate Multifractal Noise (Log-Normal Cascade)
    # The noise field itself has beta = 1 - K(2).
    # If we want a process with significant H, we integrate.
    N = 8192
    noise = generate_lognormal_cascade(N, sigma=0.4)
    # Integrate to H=0.3
    # Resulting Beta should be 1 - K(2) + 2H.
    # Wait, for a conservative cascade, K(1) = 0.
    # Structure function scaling zeta(q) = qH - K(q).
    # S1 ~ dt^(H - K(1)) = dt^H.
    # So H_est from S1 should be exactly H.
    # S2 ~ dt^(2H - K(2)).
    # Beta = 1 + zeta(2) = 1 + 2H - K(2).

    # Let's create a walk (process) with H=0.5 (Brownian-like but multifractal)
    # Integrating noise (H_noise = -0.5) by factor 1 (H_diff=1) gives H=0.5.
    # The 'measure' from cascade is the noise variance? No, it's the noise magnitude |dw|.
    # We need signed noise. Multiply by random sign?
    signs = np.random.choice([-1, 1], size=N)
    signed_noise = noise * signs

    # Integrate to get the process
    process = np.cumsum(signed_noise)
    time = np.arange(N)

    print("Analyzing Multifractal Random Walk...")
    results = calculate_intermittency_correction(time, process)

    for k, v in results.items():
        print(f"  {k}: {v:.4f}")

    print("\nVerification:")
    diff_mono = abs(results["Beta (FFT)"] - results["Beta (Monofractal 1+2H)"])
    diff_multi = abs(results["Beta (FFT)"] - results["Beta (Multifractal 1+2H-K2)"])

    print(f"  Error Monofractal Assumption: {diff_mono:.4f}")
    print(f"  Error Multifractal Correction: {diff_multi:.4f}")

    if diff_multi < diff_mono:
        print("  -> Multifractal correction improves Beta estimation.")
    else:
        print("  -> Multifractal correction did not improve estimation (likely monofractal or noise).")

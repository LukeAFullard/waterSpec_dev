import numpy as np
import matplotlib.pyplot as plt

def generate_lognormal_cascade(N, lambda_scale=2, sigma=0.5, seed=None):
    """
    Generates a log-normal multifractal cascade (Measures).
    This produces a highly intermittent field (conservative measure).

    The scaling exponent K(q) for log-normal cascade is:
    K(q) = (sigma^2 / (2 * log(lambda_scale))) * (q^2 - q)
    So K(2) = (sigma^2 / (2 * log(lambda_scale))) * (4 - 2) = sigma^2 / log(lambda_scale)
    """
    if seed is not None:
        np.random.seed(seed)

    steps = int(np.log(N) / np.log(lambda_scale))
    measure = np.ones(1)

    for _ in range(steps):
        mu = -0.5 * sigma**2
        multipliers = np.exp(mu + sigma * np.random.standard_normal(len(measure) * lambda_scale))
        measure = np.repeat(measure, lambda_scale) * multipliers

    return measure

def fractional_integration(data, H):
    """
    Fractionally integrates a series to give it spectral slope +2H.
    """
    N = len(data)
    fft = np.fft.rfft(data)
    freqs = np.fft.rfftfreq(N)

    with np.errstate(divide='ignore'):
        filter_ = np.where(freqs > 0, freqs**(-H), 0)

    fft_filtered = fft * filter_
    return np.fft.irfft(fft_filtered, n=N)

def generate_multifractal_series(N, H_target, sigma_cascade, lambda_scale=2, seed=None):
    """
    Generates a multifractal series with known H and K(2).
    """
    if seed is not None:
        np.random.seed(seed)

    noise = generate_lognormal_cascade(N, lambda_scale=lambda_scale, sigma=sigma_cascade, seed=seed)

    signs = np.random.choice([-1, 1], size=N)
    signed_noise = noise * signs

    if H_target == 0.5:
        process = np.cumsum(signed_noise)
    else:
        # Need to subtract 0.5 because cumsum adds 0.5 to H. So we use fractional integration with H_target - 0.5
        integrated_noise = fractional_integration(signed_noise, H_target - 0.5)
        process = np.cumsum(integrated_noise)

    time = np.arange(N)

    true_K2 = (sigma_cascade**2) / np.log(lambda_scale)
    true_H = H_target
    true_beta_multi = 1 + 2 * true_H - true_K2

    return time, process, true_H, true_K2, true_beta_multi

def generate_lognormal_cascade(N, lambda_scale=2, sigma=0.5, seed=None):
    """
    Generates a log-normal multifractal cascade (Measures).
    This produces a highly intermittent field (conservative measure).

    The scaling exponent K(q) for log-normal cascade is:
    K(q) = (sigma^2 / (2 * log(lambda_scale))) * (q^2 - q)
    So K(2) = (sigma^2 / (2 * log(lambda_scale))) * (4 - 2) = sigma^2 / log(lambda_scale)
    """
    if seed is not None:
        np.random.seed(seed)

    steps = int(np.log(N) / np.log(lambda_scale))
    measure = np.ones(1)

    for _ in range(steps):
        mu = -0.5 * sigma**2
        multipliers = np.exp(mu + sigma * np.random.standard_normal(len(measure) * lambda_scale))
        measure = np.repeat(measure, lambda_scale) * multipliers

    return measure

def fractional_integration(data, H):
    """
    Fractionally integrates a series to give it spectral slope +2H.
    """
    N = len(data)
    fft = np.fft.rfft(data)
    freqs = np.fft.rfftfreq(N)

    with np.errstate(divide='ignore'):
        filter_ = np.where(freqs > 0, freqs**(-H), 0)

    fft_filtered = fft * filter_
    return np.fft.irfft(fft_filtered, n=N)

def generate_multifractal_series(N, H_target, sigma_cascade, lambda_scale=2, seed=None):
    """
    Generates a multifractal series with known H and K(2).
    """
    if seed is not None:
        np.random.seed(seed)

    noise = generate_lognormal_cascade(N, lambda_scale=lambda_scale, sigma=sigma_cascade, seed=seed)

    signs = np.random.choice([-1, 1], size=N)
    signed_noise = noise * signs

    if H_target == 0.5:
        process = np.cumsum(signed_noise)
    else:
        # Need to subtract 0.5 because cumsum adds 0.5 to H. So we use fractional integration with H_target - 0.5
        integrated_noise = fractional_integration(signed_noise, H_target - 0.5)
        process = np.cumsum(integrated_noise)

    time = np.arange(N)

    true_K2 = (sigma_cascade**2) / np.log(lambda_scale)
    true_H = H_target
    true_beta_multi = 1 + 2 * true_H - true_K2

    return time, process, true_H, true_K2, true_beta_multi

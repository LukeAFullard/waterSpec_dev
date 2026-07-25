import numpy as np
import matplotlib.pyplot as plt
import os
import csv
from waterSpec.utils_sim.tk95 import simulate_tk95
from waterSpec.utils_sim.models import power_law

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

def record_result(section_id, test_id, N, method, truth_str, estimate_str, pass_str, results_file='validation/results/section6_results.csv'):
    os.makedirs(os.path.dirname(results_file), exist_ok=True)
    file_exists = os.path.isfile(results_file)
    with open(results_file, mode='a', newline='') as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(['section_id', 'test_id', 'N', 'method', 'truth', 'estimate', 'pass'])
        writer.writerow([section_id, test_id, N, method, truth_str, estimate_str, pass_str])

def record_result_v2(results_file, test_id, seed, params, estimate, truth, ci_low, ci_high, passed):
    os.makedirs(os.path.dirname(results_file), exist_ok=True)
    file_exists = os.path.isfile(results_file)
    with open(results_file, mode='a', newline='') as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(['test_id', 'seed', 'params', 'estimate', 'truth', 'ci_low', 'ci_high', 'pass'])
        writer.writerow([test_id, seed, params, estimate, truth, ci_low, ci_high, passed])

def generate_colored_noise(N, beta, amp=1.0, dt=1.0, seed=None):
    if seed is not None:
        np.random.seed(seed)
    return simulate_tk95(psd_func=power_law, params=(beta, amp), N=N, dt=dt, seed=seed)

def apply_uniform_missingness(time, data, missing_fraction, seed=None):
    if seed is not None:
        np.random.seed(seed)
    N = len(time)
    keep_indices = np.random.choice(N, size=int(N * (1 - missing_fraction)), replace=False)
    keep_indices.sort()
    return time[keep_indices], data[keep_indices]

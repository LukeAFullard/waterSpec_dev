import numpy as np
import pandas as pd
import os
import csv
from pathlib import Path

from waterSpec.utils_sim.tk95 import simulate_tk95
from waterSpec.utils_sim.models import power_law

def generate_colored_noise(beta, amp=1.0, N=4096, dt=1.0, seed=None):
    """
    Generates colored noise matching a target beta using simulate_tk95.
    (Note: waterSpec uses the convention P(f) ~ f^-beta).
    """
    time, data = simulate_tk95(psd_func=power_law, params=(beta, amp), N=N, dt=dt, seed=seed)
    return time, data

def generate_broken_power_law(beta1, beta2, break_freq, amp=1.0, N=4096, dt=1.0, seed=None):
    """
    Generates a broken power-law spectrum time series using simulate_tk95.
    """
    def broken_power_law(f, b1, b2, f_break, a):
        p = np.zeros_like(f)
        mask = f <= f_break
        p[mask] = a * (f[mask] ** -b1)
        # Match power at the break frequency
        p_break = a * (f_break ** -b1)
        a2 = p_break / (f_break ** -b2)
        p[~mask] = a2 * (f[~mask] ** -b2)
        return p

    time, data = simulate_tk95(psd_func=broken_power_law, params=(beta1, beta2, break_freq, amp), N=N, dt=dt, seed=seed)
    return time, data

def apply_uneven_sampling(time, data, missing_fraction, method='uniform', seed=None):
    """
    Takes an evenly-sampled series and returns a random/patterned subset.
    methods: 'uniform' (randomly drop points)
    """
    if seed is not None:
        np.random.seed(seed)

    n_points = len(time)
    if method == 'uniform':
        keep_indices = np.sort(np.random.choice(n_points, size=int(n_points * (1 - missing_fraction)), replace=False))
        return time[keep_indices], data[keep_indices]
    else:
        raise NotImplementedError(f"Sampling method {method} not implemented")

def inject_seasonality(time, data, period, amplitude, phase=0.0):
    """
    Adds a sinusoid of specified period, amplitude, and phase to a base series.
    """
    sinusoid = amplitude * np.sin(2 * np.pi * (time / period) + phase)
    return time, data + sinusoid

def record_result(test_id, seed, params_dict, estimate, truth, ci_low, ci_high, passed, results_dir="validation/results"):
    """
    Appends a row to a CSV file in the results directory.
    """
    os.makedirs(results_dir, exist_ok=True)
    file_path = os.path.join(results_dir, "validation_results.csv")

    file_exists = os.path.isfile(file_path)

    with open(file_path, mode='a', newline='') as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(['test_id', 'seed', 'params', 'estimate', 'truth', 'ci_low', 'ci_high', 'passed'])

        # Format params as a string representation of the dict
        params_str = str(params_dict).replace(',', ';') # Avoid CSV comma confusion

        writer.writerow([test_id, seed, params_str, estimate, truth, ci_low, ci_high, passed])

def get_seed(section, trial_index):
    """Global RNG seeding strategy: seed = 1000 * section + trial_index"""
    return int(1000 * section + trial_index)

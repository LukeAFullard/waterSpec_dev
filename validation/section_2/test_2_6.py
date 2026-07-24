import numpy as np
import matplotlib.pyplot as plt
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from common import generate_broken_power_law, record_result, get_seed
from waterSpec.spectral_analyzer import calculate_periodogram
from waterSpec.fitter import fit_segmented_spectrum
from waterSpec.haar_analysis import HaarAnalysis
from waterSpec.utils_sim.tk95 import simulate_tk95

os.makedirs('validation/section_2/results', exist_ok=True)

N = 4096
dt = 1.0
beta1_true = 0.3
beta2_true = 1.0
beta3_true = 1.8
min_f = 1/N
max_f = 1/2
log_range = np.log10(max_f) - np.log10(min_f)

f_break1 = 10**(np.log10(min_f) + 0.33 * log_range)
f_break2 = 10**(np.log10(min_f) + 0.66 * log_range)

lag_break1 = 1.0 / f_break2 # Note: high frequency -> small lag
lag_break2 = 1.0 / f_break1

def two_break_power_law(f, b1, b2, b3, f_b1, f_b2, a):
    p = np.zeros_like(f)
    mask1 = f <= f_b1
    mask2 = (f > f_b1) & (f <= f_b2)
    mask3 = f > f_b2

    p[mask1] = a * (f[mask1] ** -b1)

    p_break1 = a * (f_b1 ** -b1)
    a2 = p_break1 / (f_b1 ** -b2)
    p[mask2] = a2 * (f[mask2] ** -b2)

    p_break2 = a2 * (f_b2 ** -b2)
    a3 = p_break2 / (f_b2 ** -b3)
    p[mask3] = a3 * (f[mask3] ** -b3)
    return p

n_trials = 10
passed_ls = 0
passed_haar = 0

print(f"--- Testing 2.6: Two breakpoints (f_b1={f_break1:.5f}, f_b2={f_break2:.5f}) ---")

for i in range(n_trials):
    seed = get_seed(2, 6000 + i)
    time, data = simulate_tk95(psd_func=two_break_power_law, params=(beta1_true, beta2_true, beta3_true, f_break1, f_break2, 1.0), N=N, dt=dt, seed=seed)

    # --- LS ---
    frequency, power, _ = calculate_periodogram(time, data, samples_per_peak=1)
    try:
        res_ls = fit_segmented_spectrum(frequency, power, n_breakpoints=2, ci_method="parametric")
        if res_ls.get('n_breakpoints') == 2:
            b1, b2, b3 = res_ls['betas']
            fb1, fb2 = res_ls['breakpoints']

            b1_err = abs(b1 - beta1_true) <= 0.3
            b2_err = abs(b2 - beta2_true) <= 0.3
            b3_err = abs(b3 - beta3_true) <= 0.3
            fb1_err = abs(np.log10(fb1) - np.log10(f_break1)) <= 0.3 * abs(np.log10(f_break1))
            fb2_err = abs(np.log10(fb2) - np.log10(f_break2)) <= 0.3 * abs(np.log10(f_break2))

            passed_trial_ls = b1_err and b2_err and b3_err and fb1_err and fb2_err
            if passed_trial_ls:
                passed_ls += 1
            record_result("2.6_LS", seed, {}, {}, {}, None, None, passed_trial_ls, results_dir="validation/section_2/results")
    except Exception as e:
        pass

    # --- Haar ---
    try:
        haar = HaarAnalysis(time, data)
        res_haar = haar.run(max_breakpoints=2)
        if res_haar.get('n_breakpoints') == 2:
            # Haar measures scale from small to large, which corresponds to freq from large to small.
            hb3, hb2, hb1 = res_haar['betas']
            hlb1, hlb2 = res_haar['breakpoints']

            hb3_err = abs(hb3 - beta3_true) <= 0.4
            hb2_err = abs(hb2 - beta2_true) <= 0.4
            hb1_err = abs(hb1 - beta1_true) <= 0.4
            hlb1_err = abs(np.log10(hlb1) - np.log10(lag_break1)) <= 0.3 * abs(np.log10(lag_break1))
            hlb2_err = abs(np.log10(hlb2) - np.log10(lag_break2)) <= 0.3 * abs(np.log10(lag_break2))

            passed_trial_haar = hb1_err and hb2_err and hb3_err and hlb1_err and hlb2_err
            if passed_trial_haar:
                passed_haar += 1
            record_result("2.6_Haar", seed, {}, {}, {}, None, None, passed_trial_haar, results_dir="validation/section_2/results")
    except Exception as e:
        pass

print(f"Passed LS: {passed_ls}/{n_trials}")
print(f"Passed Haar: {passed_haar}/{n_trials}")

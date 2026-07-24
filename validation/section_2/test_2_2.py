import numpy as np
import matplotlib.pyplot as plt
import os
import sys
import warnings

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from common import generate_broken_power_law, record_result, get_seed

from waterSpec.spectral_analyzer import calculate_periodogram
from waterSpec.fitter import fit_segmented_spectrum
from waterSpec.haar_analysis import HaarAnalysis

warnings.filterwarnings('ignore')

N = 2048
dt = 1.0
beta1_true = 0.3
beta2_true = 1.8
min_f = 1/N
max_f = 1/2
log_range = np.log10(max_f) - np.log10(min_f)

fractions = [0.10, 0.25, 0.50, 0.75, 0.90]

for frac in fractions:
    break_freq_true = 10**(np.log10(min_f) + frac * log_range)
    break_lag_true = 1.0 / break_freq_true
    print(f"\n--- Testing break at {frac*100}% (f_break={break_freq_true:.5f}, lag_break={break_lag_true:.2f}) ---")

    passed_ls = 0
    passed_haar = 0
    for i in range(5):
        seed = get_seed(2, int(100 * frac) * 100 + i)
        time, data = generate_broken_power_law(beta1_true, beta2_true, break_freq_true, amp=1.0, N=N, dt=dt, seed=seed)

        # --- LS ---
        frequency, power, _ = calculate_periodogram(time, data, samples_per_peak=1)
        try:
            res_ls = fit_segmented_spectrum(frequency, power, n_breakpoints=1, ci_method="parametric")
            if res_ls.get('n_breakpoints') == 1:
                b1_ls = res_ls['betas'][0]
                b2_ls = res_ls['betas'][1]
                f_break_ls = res_ls['breakpoints'][0]

                b1_err_ls = abs(b1_ls - beta1_true)
                b2_err_ls = abs(b2_ls - beta2_true)
                f_err_ls = abs(np.log10(f_break_ls) - np.log10(break_freq_true))

                passed_trial_ls = b1_err_ls <= 0.2 and b2_err_ls <= 0.2 and f_err_ls <= 0.3 * abs(np.log10(break_freq_true))
                if passed_trial_ls:
                    passed_ls += 1
                record_result("2.2_LS", seed, {"frac": frac, "beta1": beta1_true, "beta2": beta2_true, "break_freq": break_freq_true},
                              {"b1": b1_ls, "b2": b2_ls, "f_break": f_break_ls},
                              {"b1": beta1_true, "b2": beta2_true, "f_break": break_freq_true},
                              None, None, passed_trial_ls, results_dir="validation/section_2/results")
        except Exception:
            pass

        # --- Haar ---
        try:
            haar = HaarAnalysis(time, data)
            res_haar = haar.run(max_breakpoints=1)

            if res_haar.get('n_breakpoints') == 1:
                b_small_lag = res_haar['betas'][0] # High freq
                b_large_lag = res_haar['betas'][1] # Low freq
                lag_break = res_haar['breakpoints'][0]

                b_small_lag_pass = abs(b_small_lag - beta2_true) <= 0.3
                b_large_lag_pass = abs(b_large_lag - beta1_true) <= 0.3
                lag_break_log_diff = abs(np.log10(lag_break) - np.log10(break_lag_true))
                lag_break_pass = lag_break_log_diff <= 0.3 * abs(np.log10(break_lag_true))

                passed_trial_haar = b_small_lag_pass and b_large_lag_pass and lag_break_pass
                if passed_trial_haar:
                    passed_haar += 1
                record_result("2.2_Haar", seed, {"frac": frac, "beta_low_f": beta1_true, "beta_high_f": beta2_true, "break_lag": break_lag_true},
                              {"b_low_f": b_large_lag, "b_high_f": b_small_lag, "lag_break": lag_break},
                              {"b_low_f": beta1_true, "b_high_f": beta2_true, "lag_break": break_lag_true},
                              None, None, passed_trial_haar, results_dir="validation/section_2/results")
        except Exception:
            pass

    print(f"Passed LS: {passed_ls}/5 trials")
    print(f"Passed Haar: {passed_haar}/5 trials")

import numpy as np
import matplotlib.pyplot as plt
import os
import sys
import warnings

# Add parent directory to path to import common
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from common import generate_broken_power_law, record_result, get_seed

from waterSpec.spectral_analyzer import calculate_periodogram
from waterSpec.fitter import fit_segmented_spectrum
from waterSpec.haar_analysis import HaarAnalysis

os.makedirs('validation/section_2/plots', exist_ok=True)
os.makedirs('validation/section_2/results', exist_ok=True)

N = 2048
dt = 1.0
beta1_true = 0.3
beta2_true = 1.8
min_f = 1/N
max_f = 1/2
break_freq_true = np.sqrt(min_f * max_f)

n_trials = 20
passed_ls = 0
passed_haar = 0

print(f"Target Break Frequency: {break_freq_true:.5f}")
break_lag_true = 1.0 / break_freq_true
print(f"Target Break Lag: {break_lag_true:.2f}")

warnings.filterwarnings('ignore')

for i in range(n_trials):
    seed = get_seed(2, i)
    time, data = generate_broken_power_law(beta1_true, beta2_true, break_freq_true, amp=1.0, N=N, dt=dt, seed=seed)

    # --- LS Analysis ---
    frequency, power, _ = calculate_periodogram(time, data, samples_per_peak=1)

    try:
        res_ls = fit_segmented_spectrum(frequency, power, n_breakpoints=1, ci_method="parametric")

        b1_ls = res_ls['betas'][0]
        b2_ls = res_ls['betas'][1]
        f_break_ls = res_ls['breakpoints'][0]

        b1_pass_ls = abs(b1_ls - beta1_true) <= 0.2
        b2_pass_ls = abs(b2_ls - beta2_true) <= 0.2
        f_break_log_diff_ls = abs(np.log10(f_break_ls) - np.log10(break_freq_true))
        f_break_pass_ls = f_break_log_diff_ls <= 0.3 * abs(np.log10(break_freq_true))

        passed_ls_trial = b1_pass_ls and b2_pass_ls and f_break_pass_ls
        if passed_ls_trial:
            passed_ls += 1

        record_result("2.1_LS", seed, {"beta1": beta1_true, "beta2": beta2_true, "break_freq": break_freq_true},
                      {"b1": b1_ls, "b2": b2_ls, "f_break": f_break_ls},
                      {"b1": beta1_true, "b2": beta2_true, "f_break": break_freq_true},
                      None, None, passed_ls_trial, results_dir="validation/section_2/results")

        print(f"Trial {i} (LS): b1={b1_ls:.3f}, b2={b2_ls:.3f}, f_break={f_break_ls:.5f} -> Pass: {passed_ls_trial}")
    except Exception as e:
        record_result("2.1_LS", seed, {"beta1": beta1_true, "beta2": beta2_true, "break_freq": break_freq_true},
                      "Error", "Error", None, None, False, results_dir="validation/section_2/results")

    # --- Haar Analysis ---
    try:
        haar = HaarAnalysis(time, data)
        # Note: 'analysis_mode' is not a parameter of run(), it's returned by the class
        res_haar = haar.run(max_breakpoints=1)

        if res_haar.get('n_breakpoints') == 1:
            b_small_lag = res_haar['betas'][0] # High freq
            b_large_lag = res_haar['betas'][1] # Low freq
            lag_break = res_haar['breakpoints'][0]

            b_small_lag_pass = abs(b_small_lag - beta2_true) <= 0.3  # Haar has a bit more bias
            b_large_lag_pass = abs(b_large_lag - beta1_true) <= 0.3

            lag_break_log_diff = abs(np.log10(lag_break) - np.log10(break_lag_true))
            lag_break_pass = lag_break_log_diff <= 0.3 * abs(np.log10(break_lag_true))

            passed_haar_trial = b_small_lag_pass and b_large_lag_pass and lag_break_pass
            if passed_haar_trial:
                passed_haar += 1

            record_result("2.1_Haar", seed, {"beta_low_f": beta1_true, "beta_high_f": beta2_true, "break_lag": break_lag_true},
                          {"b_low_f": b_large_lag, "b_high_f": b_small_lag, "lag_break": lag_break},
                          {"b_low_f": beta1_true, "b_high_f": beta2_true, "lag_break": break_lag_true},
                          None, None, passed_haar_trial, results_dir="validation/section_2/results")

            print(f"Trial {i} (Haar): b_high_f={b_small_lag:.3f}, b_low_f={b_large_lag:.3f}, lag_break={lag_break:.2f} -> Pass: {passed_haar_trial}")
        else:
            print(f"Trial {i} (Haar): preferred standard model, not segmented")
            record_result("2.1_Haar", seed, {"beta_low_f": beta1_true, "beta_high_f": beta2_true, "break_lag": break_lag_true},
                          "Preferred Standard", "Preferred Standard", None, None, False, results_dir="validation/section_2/results")

    except Exception as e:
        print(f"Trial {i} (Haar) failed: {e}")
        record_result("2.1_Haar", seed, {"beta_low_f": beta1_true, "beta_high_f": beta2_true, "break_lag": break_lag_true},
                      "Error", "Error", None, None, False, results_dir="validation/section_2/results")

print(f"\nPassed LS: {passed_ls}/{n_trials} trials.")
print(f"Passed Haar: {passed_haar}/{n_trials} trials.")

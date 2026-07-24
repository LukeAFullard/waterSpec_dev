import numpy as np
import os
import sys
import warnings

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from common import generate_broken_power_law, record_result, get_seed
from waterSpec.spectral_analyzer import calculate_periodogram
from waterSpec.model_selector import ModelSelector
from waterSpec.haar_analysis import HaarAnalysis

warnings.filterwarnings('ignore')

N = 2048
dt = 1.0
min_f = 1/N
max_f = 1/2
break_freq_true = np.sqrt(min_f * max_f)
beta1_true = 0.3

diffs = [0.2, 0.5, 1.0, 2.0]
selector = ModelSelector()

for diff in diffs:
    beta2_true = beta1_true + diff
    print(f"\n--- Testing slope diff {diff} (b1={beta1_true}, b2={beta2_true}) ---")

    passed_ls = 0
    passed_haar = 0
    for i in range(10):
        seed = get_seed(2, int(diff*100)*100 + i)
        time, data = generate_broken_power_law(beta1_true, beta2_true, break_freq_true, amp=1.0, N=N, dt=dt, seed=seed)

        # --- LS ---
        frequency, power, _ = calculate_periodogram(time, data, samples_per_peak=1)
        try:
            best_res_ls = selector.select_best_model(
                frequency, power,
                fit_method="theil-sen",
                ci_method="parametric",
                bootstrap_type="block",
                n_bootstraps=100,
                max_breakpoints=1,
                seed=seed
            )
            preferred_segmented_ls = (best_res_ls.get('n_breakpoints', 0) == 1)
            if preferred_segmented_ls:
                passed_ls += 1
            record_result("2.3_LS", seed, {"diff": diff, "beta1": beta1_true, "beta2": beta2_true, "break_freq": break_freq_true},
                          {"preferred_segmented": preferred_segmented_ls},
                          {"preferred_segmented": True},
                          None, None, preferred_segmented_ls, results_dir="validation/section_2/results")
        except Exception as e:
            pass

        # --- Haar ---
        try:
            haar = HaarAnalysis(time, data)
            # using 'analysis_mode' to get best model automatically if supported, or just run with max_breakpoints=1
            # Actually, HaarAnalysis.run() selects based on BIC. So it might return 0 or 1 breakpoints.
            res_haar = haar.run(max_breakpoints=1)
            preferred_segmented_haar = (res_haar.get('n_breakpoints', 0) == 1)
            if preferred_segmented_haar:
                passed_haar += 1
            record_result("2.3_Haar", seed, {"diff": diff, "beta1": beta1_true, "beta2": beta2_true, "break_freq": break_freq_true},
                          {"preferred_segmented": preferred_segmented_haar},
                          {"preferred_segmented": True},
                          None, None, preferred_segmented_haar, results_dir="validation/section_2/results")
        except Exception as e:
            pass

    print(f"Passed LS: {passed_ls}/10 trials")
    print(f"Passed Haar: {passed_haar}/10 trials")

import numpy as np
import os
import sys
import warnings

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from common import generate_colored_noise, record_result, get_seed
from waterSpec.spectral_analyzer import calculate_periodogram
from waterSpec.model_selector import ModelSelector
from waterSpec.haar_analysis import HaarAnalysis

warnings.filterwarnings('ignore')

N = 2048
dt = 1.0
beta_true = 1.0

n_trials = 20
passed_ls = 0
passed_haar = 0
selector = ModelSelector()

print(f"--- Testing no break (pink noise) ---")

for i in range(n_trials):
    seed = get_seed(2, 4000 + i)
    time, data = generate_colored_noise(beta_true, amp=1.0, N=N, dt=dt, seed=seed)

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
        preferred_single_ls = (best_res_ls.get('n_breakpoints', 0) == 0)
        if preferred_single_ls:
            passed_ls += 1
        record_result("2.4_LS", seed, {"beta": beta_true},
                      {"preferred_single": preferred_single_ls},
                      {"preferred_single": True},
                      None, None, preferred_single_ls, results_dir="validation/section_2/results")
    except Exception as e:
        pass

    # --- Haar ---
    try:
        haar = HaarAnalysis(time, data)
        res_haar = haar.run(max_breakpoints=1)
        preferred_single_haar = (res_haar.get('n_breakpoints', 0) == 0)
        if preferred_single_haar:
            passed_haar += 1
        record_result("2.4_Haar", seed, {"beta": beta_true},
                      {"preferred_single": preferred_single_haar},
                      {"preferred_single": True},
                      None, None, preferred_single_haar, results_dir="validation/section_2/results")
    except Exception as e:
        pass

print(f"Passed LS: {passed_ls}/{n_trials} trials")
print(f"Passed Haar: {passed_haar}/{n_trials} trials")

import numpy as np
import os
import sys
import warnings
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from common import generate_broken_power_law, generate_colored_noise, record_result, get_seed
from waterSpec.analysis import Analysis

warnings.filterwarnings('ignore')

N = 1024
dt = 1.0
min_f = 1/N
max_f = 1/2
break_freq_true = np.sqrt(min_f * max_f)
beta1_true = 0.3
beta2_true = 1.8

n_trials = 10
passed_break_ls = 0
passed_no_break_ls = 0

print("--- Testing 2.5: Automatic model selection end-to-end ---")

for i in range(n_trials):
    seed = get_seed(2, 5000 + i)

    # 1. True-break dataset
    time_b, data_b = generate_broken_power_law(beta1_true, beta2_true, break_freq_true, amp=1.0, N=N, dt=dt, seed=seed)
    df_b = pd.DataFrame({"time": time_b, "data": data_b})
    analysis_b = Analysis(time_col="time", data_col="data", dataframe=df_b, input_time_unit="seconds")
    res_b = analysis_b.run_full_analysis(
        output_dir=f"validation/section_2/results/out_b_{i}",
        ci_method="parametric",
        max_breakpoints=1,
        peak_detection_method=None,
        run_haar=True,
        haar_max_breakpoints=1,
        calc_intermittency=False
    )
    chosen_ls_b = res_b.get("chosen_model_type", "")
    passed_b_ls = chosen_ls_b == "segmented"
    if passed_b_ls: passed_break_ls += 1

    # 2. No-break dataset
    time_nb, data_nb = generate_colored_noise(1.0, amp=1.0, N=N, dt=dt, seed=seed)
    df_nb = pd.DataFrame({"time": time_nb, "data": data_nb})
    analysis_nb = Analysis(time_col="time", data_col="data", dataframe=df_nb, input_time_unit="seconds")
    res_nb = analysis_nb.run_full_analysis(
        output_dir=f"validation/section_2/results/out_nb_{i}",
        ci_method="parametric",
        max_breakpoints=1,
        peak_detection_method=None,
        run_haar=True,
        haar_max_breakpoints=1,
        calc_intermittency=False
    )
    chosen_ls_nb = res_nb.get("chosen_model_type", "")
    passed_nb_ls = chosen_ls_nb == "standard"
    if passed_nb_ls: passed_no_break_ls += 1

    print(f"Trial {i} (LS): True-break chosen={chosen_ls_b} (pass={passed_b_ls}); No-break chosen={chosen_ls_nb} (pass={passed_nb_ls})")

print(f"\nPassed break cases (LS): {passed_break_ls}/{n_trials}")
print(f"Passed no-break cases (LS): {passed_no_break_ls}/{n_trials}")

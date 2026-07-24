import numpy as np
import os
import sys
import pandas as pd
import warnings

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from common import generate_colored_noise, get_seed
from waterSpec.analysis import Analysis

warnings.filterwarnings('ignore')

N = 1000
dt = 1.0

print("--- Testing 2.9: Before/after split significance testing ---")

passed_diff = 0
passed_no_diff = 0

for i in range(10):
    seed = get_seed(2, 9000 + i)

    # 1. Real difference (persistence shift)
    t1, d1 = generate_colored_noise(0.0, amp=1.0, N=N//2, dt=dt, seed=seed)
    t2, d2 = generate_colored_noise(2.0, amp=1.0, N=N//2, dt=dt, seed=seed+1)
    t = np.concatenate([t1, t2 + t1[-1] + dt])
    d = np.concatenate([d1, d2])

    df_diff = pd.DataFrame({"time": t, "data": d})
    analysis_diff = Analysis(time_col="time", data_col="data", dataframe=df_diff, input_time_unit="seconds")
    try:
        res_diff = analysis_diff._prepare_changepoint_segments(
            N//2,
            run_ls=True,
            run_haar=False,
            ci_method="parametric",
            max_breakpoints=1,
            calc_intermittency=False
        )
        comp = analysis_diff._compare_segments(res_diff[0], res_diff[1], "ls")
        if comp.get('significant_difference', False):
            passed_diff += 1
    except Exception as e:
        pass

    # 2. No difference (negative control)
    t_nodiff, d_nodiff = generate_colored_noise(1.0, amp=1.0, N=N, dt=dt, seed=seed)
    df_nodiff = pd.DataFrame({"time": t_nodiff, "data": d_nodiff})
    analysis_nodiff = Analysis(time_col="time", data_col="data", dataframe=df_nodiff, input_time_unit="seconds")
    try:
        res_nodiff = analysis_nodiff._prepare_changepoint_segments(
            N//2,
            run_ls=True,
            run_haar=False,
            ci_method="parametric",
            max_breakpoints=1,
            calc_intermittency=False
        )
        comp_nodiff = analysis_nodiff._compare_segments(res_nodiff[0], res_nodiff[1], "ls")
        if not comp_nodiff.get('significant_difference', True):
            passed_no_diff += 1
    except Exception as e:
        pass

print(f"Passed real difference: {passed_diff}/10")
print(f"Passed no difference: {passed_no_diff}/10")

import numpy as np
import os
import sys
import matplotlib.pyplot as plt
from pathlib import Path
import warnings

sys.path.insert(0, str(Path(os.getcwd())))
from validation.common import generate_colored_noise, apply_uneven_sampling, record_result, get_seed
from waterSpec.spectral_analyzer import calculate_periodogram
from waterSpec.fitter import fit_standard_model
from waterSpec.haar_analysis import calculate_haar_fluctuations, fit_haar_slope

RESULTS_DIR = "validation/section_3/results"
PLOTS_DIR = "validation/section_3/plots"
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(PLOTS_DIR, exist_ok=True)

N_POINTS = 4096

def test_3_6_stress():
    print("Running 3.6 Stress Test...")
    frac = 0.98 # 98% missing
    beta = 1.0
    seed = get_seed(3, 6000 + int(beta*10) * 100)
    time, data = generate_colored_noise(beta=beta, N=N_POINTS, seed=seed)
    time_sub, data_sub = apply_uneven_sampling(time, data, frac, method='uniform', seed=seed)

    # We want to capture warnings or see if it fails gracefully
    import io
    from contextlib import redirect_stderr
    import traceback

    output = []

    output.append(f"Original length: {len(time)}, Subsampled length: {len(time_sub)}")

    try:
        f, p, _ = calculate_periodogram(time_sub, data_sub)
        ls_res = fit_standard_model(f, p, method='theil-sen', ci_method='parametric')
        output.append(f"Stress test LS returned beta: {ls_res['beta']}")
    except Exception as e:
        output.append(f"Stress test LS failed with exception: {type(e).__name__}: {e}")

    try:
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            lags, H2, counts, n_eff = calculate_haar_fluctuations(time_sub, data_sub, max_lag=N_POINTS/5.0)
            haar_res = fit_haar_slope(lags, H2, n_effective=n_eff)

            for warning in w:
                output.append(f"Haar Warning: {warning.message}")

            output.append(f"Stress test Haar returned beta: {haar_res['beta']}")
    except Exception as e:
        output.append(f"Stress test Haar failed with exception: {type(e).__name__}: {e}")

    for line in output:
        print(line)

    # Record to a simple text file for the report
    with open(os.path.join(RESULTS_DIR, "stress_test_output.txt"), "w") as f:
        f.write("\n".join(output))

if __name__ == '__main__':
    test_3_6_stress()

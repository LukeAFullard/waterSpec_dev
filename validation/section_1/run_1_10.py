import sys, os, warnings
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from validation.common import generate_colored_noise, get_seed
from waterSpec.spectral_analyzer import calculate_periodogram
from waterSpec.fitter import fit_standard_model

warnings.filterwarnings("ignore")

def run_test():
    print(f"\n--- Running Test 1.10 (Target Beta = 1.0) ---")
    n_trials = 100
    beta_target = 1.0

    methods = [
        ("parametric", None),
        ("bootstrap", "block"),
        ("bootstrap", "wild"),
        ("bootstrap", "pairs"),
        ("bootstrap", "residuals")
    ]

    results = {f"{m[0]}_{m[1]}": {'hits': 0, 'widths': []} for m in methods}

    for trial in range(n_trials):
        seed = get_seed(1, trial) + 1010
        time, data = generate_colored_noise(beta_target, amp=1.0, N=1024, dt=1.0, seed=seed)

        freq, power, _ = calculate_periodogram(time, data, samples_per_peak=1)

        for ci_method, boot_type in methods:
            key = f"{ci_method}_{boot_type}"
            res = fit_standard_model(freq, power, method="ols", ci_method=ci_method, bootstrap_type=boot_type if boot_type else "pairs", seed=seed)
            ci_low = res['beta_ci_lower']
            ci_high = res['beta_ci_upper']
            results[key]['widths'].append(ci_high - ci_low)
            if ci_low <= beta_target <= ci_high:
                results[key]['hits'] += 1

    for key, res in results.items():
        cov = res['hits'] / n_trials
        mean_width = np.mean(res['widths'])
        flag = "!!! FLAG: UNDER-COVERAGE" if cov < 0.85 else "PASS"
        print(f"{key:25s}: CI Coverage = {cov*100:5.1f}%, Mean Width = {mean_width:6.3f} -> {flag}")

run_test()

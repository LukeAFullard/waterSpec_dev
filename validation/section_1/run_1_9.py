import sys, os, warnings
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from validation.common import generate_colored_noise, get_seed
from waterSpec.spectral_analyzer import calculate_periodogram
from waterSpec.fitter import fit_standard_model

warnings.filterwarnings("ignore")

def run_test(test_id, beta_target):
    print(f"\n--- Running Test {test_id} (Target Beta = {beta_target}) ---")
    n_trials = 30

    results = {
        'TS_clean': {'betas': [], 'ci_widths': []},
        'OLS_clean': {'betas': [], 'ci_widths': []},
        'TS_outlier': {'betas': [], 'ci_widths': []},
        'OLS_outlier': {'betas': [], 'ci_widths': []}
    }

    for trial in range(n_trials):
        seed = get_seed(1, trial) + int(abs(beta_target) * 1000)
        time, data = generate_colored_noise(beta_target, amp=1.0, N=4096, dt=1.0, seed=seed)

        freq, power, _ = calculate_periodogram(time, data, samples_per_peak=1)

        # OLS clean
        res_ols = fit_standard_model(freq, power, method="ols", ci_method="parametric", seed=seed)
        results['OLS_clean']['betas'].append(res_ols['beta'])
        results['OLS_clean']['ci_widths'].append(res_ols['beta_ci_upper'] - res_ols['beta_ci_lower'])

        # TS clean
        res_ts = fit_standard_model(freq, power, method="theil-sen", ci_method="parametric", seed=seed)
        results['TS_clean']['betas'].append(res_ts['beta'])
        results['TS_clean']['ci_widths'].append(res_ts['beta_ci_upper'] - res_ts['beta_ci_lower'])

        # Inject outlier near the middle of log frequency
        mid_idx = len(power) // 2
        power_outlier = power.copy()
        power_outlier[mid_idx] *= 100.0

        # OLS outlier
        res_ols_out = fit_standard_model(freq, power_outlier, method="ols", ci_method="parametric", seed=seed)
        results['OLS_outlier']['betas'].append(res_ols_out['beta'])
        results['OLS_outlier']['ci_widths'].append(res_ols_out['beta_ci_upper'] - res_ols_out['beta_ci_lower'])

        # TS outlier
        res_ts_out = fit_standard_model(freq, power_outlier, method="theil-sen", ci_method="parametric", seed=seed)
        results['TS_outlier']['betas'].append(res_ts_out['beta'])
        results['TS_outlier']['ci_widths'].append(res_ts_out['beta_ci_upper'] - res_ts_out['beta_ci_lower'])


    for method, res in results.items():
        betas = res['betas']
        if not betas:
            continue

        mean_beta = np.mean(betas)
        bias = mean_beta - beta_target
        mean_ci_width = np.mean(res['ci_widths'])

        print(f"{method:12s}: Mean beta = {mean_beta:6.3f} (bias {bias:+6.3f}), Mean CI Width = {mean_ci_width:6.3f}")

betas = [0.0, 1.0, 2.0, -1.0, -2.0]

for beta in betas:
    run_test(f"1.9_beta{beta}", beta)

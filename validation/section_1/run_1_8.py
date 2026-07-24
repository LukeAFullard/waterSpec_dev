import sys, os, warnings
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from validation.common import generate_colored_noise, record_result, get_seed
from waterSpec.spectral_analyzer import calculate_periodogram
from waterSpec.fitter import fit_standard_model
from waterSpec.haar_analysis import HaarAnalysis

warnings.filterwarnings("ignore")

def run_test(test_id, beta_target, N):
    print(f"\n--- Running Test {test_id} (Target Beta = {beta_target}, N = {N}) ---", flush=True)
    n_trials = 30

    results = {
        'TS': {'betas': [], 'ci_widths': []},
        'OLS': {'betas': [], 'ci_widths': []},
        'HAAR': {'betas': [], 'ci_widths': []}
    }

    for trial in range(n_trials):
        seed = get_seed(1, trial) + N
        time, data = generate_colored_noise(beta_target, amp=1.0, N=N, dt=1.0, seed=seed)

        freq, power, _ = calculate_periodogram(time, data, samples_per_peak=1)

        # OLS
        res_ols = fit_standard_model(freq, power, method="ols", ci_method="parametric", seed=seed)
        results['OLS']['betas'].append(res_ols['beta'])
        results['OLS']['ci_widths'].append(res_ols['beta_ci_upper'] - res_ols['beta_ci_lower'])

        # Theil-Sen
        res_ts = fit_standard_model(freq, power, method="theil-sen", ci_method="parametric", seed=seed)
        results['TS']['betas'].append(res_ts['beta'])
        results['TS']['ci_widths'].append(res_ts['beta_ci_upper'] - res_ts['beta_ci_lower'])

        # Haar
        try:
            haar_analyzer = HaarAnalysis(time, data)
            haar_results = haar_analyzer.run(calc_intermittency=False, max_breakpoints=0, correct_periodicity=False)
            results['HAAR']['betas'].append(haar_results['beta'])
            ci_low_h = haar_results.get('beta_ci_lower', haar_results.get('ci_lower', 0))
            ci_high_h = haar_results.get('beta_ci_upper', haar_results.get('ci_upper', 0))
            results['HAAR']['ci_widths'].append(ci_high_h - ci_low_h)
        except Exception as e:
            pass

    for method, res in results.items():
        betas = res['betas']
        if not betas:
            print(f"{method.upper():5s}: FAILED (No results)", flush=True)
            continue

        mean_beta = np.mean(betas)
        bias = mean_beta - beta_target
        mean_ci_width = np.mean(res['ci_widths'])

        print(f"{method.upper():5s}: Mean beta = {mean_beta:6.3f} (bias {bias:+6.3f}), Mean CI Width = {mean_ci_width:6.3f}", flush=True)

N_values = [128, 512, 2048, 8192, 32768]

for N in N_values:
    run_test(f"1.8_N{N}", 1.0, N)

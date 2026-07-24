import sys, os, warnings
import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from validation.common import generate_colored_noise, record_result, get_seed
from waterSpec.spectral_analyzer import calculate_periodogram
from waterSpec.fitter import fit_standard_model
from waterSpec.haar_analysis import HaarAnalysis

warnings.filterwarnings("ignore")

def plot_and_save(freq, power, beta_ts, beta_ols, beta_haar, test_id, target, time, data, exclude_n=0):
    os.makedirs('validation/section_1/plots', exist_ok=True)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    ax1.plot(time, data, color='blue', alpha=0.7)
    ax1.set_title(f'Time Series (target $\\beta={target}$)')
    ax1.set_xlabel('Time')
    ax1.set_ylabel('Data')

    ax2.loglog(freq, power, label='Power', alpha=0.5)
    freq_norm = freq / freq[0]
    y_fit_target = power[0] * freq_norm**(-target)
    ax2.loglog(freq, y_fit_target, 'k-', linewidth=2, label=f'Target $\\beta={target}$')

    if beta_ts is not None:
        y_fit_ts = power[0] * freq_norm**(-beta_ts)
        ax2.loglog(freq, y_fit_ts, 'r--', label=f'Theil-Sen $\\beta={beta_ts:.2f}$')

    if beta_ols is not None:
        y_fit_ols = power[0] * freq_norm**(-beta_ols)
        ax2.loglog(freq, y_fit_ols, 'g:', label=f'OLS $\\beta={beta_ols:.2f}$')

    if beta_haar is not None:
        y_fit_haar = power[0] * freq_norm**(-beta_haar)
        ax2.loglog(freq, y_fit_haar, 'b-.', label=f'Haar $\\beta={beta_haar:.2f}$')

    ax2.set_title(f'Test {test_id} Spectra')
    ax2.legend()
    plt.tight_layout()
    plt.savefig(f'validation/section_1/plots/test_{test_id}.png')
    plt.close()

def run_test(test_id, beta_target):
    print(f"\n--- Running Test {test_id} (Target Beta = {beta_target}) ---")
    n_trials = 30

    results = {
        'TS': {'betas': [], 'ci_hits': 0},
        'OLS': {'betas': [], 'ci_hits': 0},
        'HAAR': {'betas': [], 'ci_hits': 0}
    }

    first_plot_done = False

    for trial in range(n_trials):
        seed = get_seed(1, trial) + int(abs(beta_target) * 1000)
        time, data = generate_colored_noise(beta_target, amp=1.0, N=4096, dt=1.0, seed=seed)

        freq, power, _ = calculate_periodogram(time, data, samples_per_peak=1)

        freq_fit = freq
        power_fit = power

        # OLS
        res_ols = fit_standard_model(freq_fit, power_fit, method="ols", ci_method="parametric", seed=seed)
        beta_ols = res_ols['beta']
        ci_low_ols, ci_high_ols = res_ols['beta_ci_lower'], res_ols['beta_ci_upper']
        results['OLS']['betas'].append(beta_ols)
        if ci_low_ols <= beta_target <= ci_high_ols:
            results['OLS']['ci_hits'] += 1
        record_result(f"{test_id}_OLS", seed, {'beta': beta_target}, beta_ols, beta_target, ci_low_ols, ci_high_ols, ci_low_ols <= beta_target <= ci_high_ols, results_dir="validation/section_1/results")

        # Theil-Sen
        res_ts = fit_standard_model(freq_fit, power_fit, method="theil-sen", ci_method="parametric", seed=seed)
        beta_ts = res_ts['beta']
        ci_low_ts, ci_high_ts = res_ts['beta_ci_lower'], res_ts['beta_ci_upper']
        results['TS']['betas'].append(beta_ts)
        if ci_low_ts <= beta_target <= ci_high_ts:
            results['TS']['ci_hits'] += 1
        record_result(f"{test_id}_TS", seed, {'beta': beta_target}, beta_ts, beta_target, ci_low_ts, ci_high_ts, ci_low_ts <= beta_target <= ci_high_ts, results_dir="validation/section_1/results")

        # Haar
        beta_haar = None
        try:
            haar_analyzer = HaarAnalysis(time, data)
            # Ensure bootstrap_ci is False to speed up script and since we just need the beta and some CI
            haar_results = haar_analyzer.run(calc_intermittency=False, max_breakpoints=0, correct_periodicity=False)
            beta_haar = haar_results['beta']
            ci_low_h = haar_results.get('beta_ci_lower', haar_results.get('ci_lower', 0))
            ci_high_h = haar_results.get('beta_ci_upper', haar_results.get('ci_upper', 0))
            results['HAAR']['betas'].append(beta_haar)
            if ci_low_h <= beta_target <= ci_high_h:
                results['HAAR']['ci_hits'] += 1
            record_result(f"{test_id}_HAAR", seed, {'beta': beta_target}, beta_haar, beta_target, ci_low_h, ci_high_h, ci_low_h <= beta_target <= ci_high_h, results_dir="validation/section_1/results")
        except Exception as e:
            # Haar can fail or complain about boundaries
            pass

        if not first_plot_done:
            plot_and_save(freq, power, beta_ts, beta_ols, beta_haar, test_id, beta_target, time, data)
            first_plot_done = True

    final_passes = {}
    for method, res in results.items():
        betas = res['betas']
        if not betas:
            print(f"{method.upper():5s}: FAILED (No results)")
            final_passes[method] = False
            continue

        mean_beta = np.mean(betas)
        bias = mean_beta - beta_target
        ci_cov = res['ci_hits'] / n_trials

        pass_bias = abs(bias) <= 0.15
        pass_cov = ci_cov >= 0.90

        passed = pass_bias and pass_cov
        final_passes[method] = passed

        print(f"{method.upper():5s}: Mean beta = {mean_beta:6.3f} (bias {bias:+6.3f}), CI cov = {ci_cov*100:5.1f}% -> {'PASS' if passed else 'FAIL'}")

run_test("1.6_0.3", 0.3)
run_test("1.6_0.7", 0.7)
run_test("1.6_1.3", 1.3)
run_test("1.6_1.7", 1.7)

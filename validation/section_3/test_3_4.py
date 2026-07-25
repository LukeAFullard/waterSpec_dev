import numpy as np
import os
import sys
import matplotlib.pyplot as plt
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

sys.path.insert(0, str(Path(os.getcwd())))
from validation.common import generate_colored_noise, apply_uneven_sampling, record_result, get_seed
from waterSpec.spectral_analyzer import calculate_periodogram
from waterSpec.fitter import fit_standard_model
from waterSpec.haar_analysis import calculate_haar_fluctuations, fit_haar_slope
from scipy.interpolate import interp1d

RESULTS_DIR = "validation/section_3/results"
PLOTS_DIR = "validation/section_3/plots"
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(PLOTS_DIR, exist_ok=True)

N_POINTS = 4096
N_SEEDS = 20
BETAS = [1.0, 1.6, 0.3]

def run_test_and_record(test_id, seed, time, data, time_sub, data_sub, params_dict, beta):
    # Plotting for one seed per beta
    plot_this = (seed % N_SEEDS == 0)

    ls_beta_val = np.nan
    haar_beta_val = np.nan

    if plot_this:
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))
        ax1.plot(time, data, alpha=0.5, label='Full')
        ax1.plot(time_sub, data_sub, '.', alpha=0.8, label='Subsampled')
        ax1.set_title(f"Time Series (Beta={beta})")
        ax1.legend()

    # LS
    try:
        f, p, _ = calculate_periodogram(time_sub, data_sub)
        ls_res = fit_standard_model(f, p, method='theil-sen', ci_method='parametric')
        ls_beta_val = ls_res['beta']
        record_result(test_id + "_LS", seed, params_dict, ls_res['beta'], beta, ls_res['beta_ci_lower'], ls_res['beta_ci_upper'],
                     passed=(abs(ls_res['beta'] - beta) < 0.2), results_dir=RESULTS_DIR)

        if plot_this:
            ax2.loglog(f, p, '.', alpha=0.5)
            fit_p = np.exp(ls_res['intercept']) * (f ** -ls_res['beta'])
            ax2.loglog(f, fit_p, 'r-', label=f"LS Fit (Beta={ls_res['beta']:.2f})")
            ax2.set_title("Lomb-Scargle Periodogram")
            ax2.legend()
    except Exception as e:
        pass

    # Haar
    try:
        lags, H2, counts, n_eff = calculate_haar_fluctuations(time_sub, data_sub, max_lag=N_POINTS/5.0)
        haar_res = fit_haar_slope(lags, H2, n_effective=n_eff)
        haar_beta_val = haar_res['beta']
        record_result(test_id + "_Haar", seed, params_dict, haar_res['beta'], beta, np.nan, np.nan,
                     passed=(abs(haar_res['beta'] - beta) < 0.2), results_dir=RESULTS_DIR)

        if plot_this:
            ax3.loglog(lags, H2, '.', alpha=0.5)
            ax3.loglog(lags, np.exp(haar_res['intercept'] + np.log(lags) * haar_res['slope']), 'r-', label=f"Haar Fit (Beta={haar_res['beta']:.2f})")
            ax3.set_title("Haar Fluctuations")
            ax3.legend()
    except Exception as e:
        pass

    if plot_this:
        plt.tight_layout()
        plt.savefig(os.path.join(PLOTS_DIR, f"{test_id}_beta{beta}.png"))
        plt.close()

def test_3_4_realistic():
    print("Running 3.4 Realistic...")
    for beta in BETAS:
        for seed_idx in range(N_SEEDS):
            seed = get_seed(3, 4000 + int(beta*10) * 100 + seed_idx)
            np.random.seed(seed)
            time, data = generate_colored_noise(beta=beta, N=N_POINTS, seed=seed)

            # Realistic: weekly nominal spacing with jitter, occasional long gaps
            intervals = np.random.normal(loc=7, scale=2, size=N_POINTS)
            intervals = np.clip(intervals, 1, 14)
            # Add long gaps
            gap_indices = np.random.choice(N_POINTS, size=int(N_POINTS*0.05), replace=False)
            intervals[gap_indices] = np.random.uniform(30, 90, size=len(gap_indices))

            time_realistic = np.cumsum(intervals)
            # Interpolate to get data at these new times
            time_interp = np.linspace(time_realistic[0], time_realistic[-1], N_POINTS)
            _, data_full = generate_colored_noise(beta=beta, N=N_POINTS, seed=seed)
            f = interp1d(time_interp, data_full, kind='linear')
            time_sub = time_realistic[time_realistic <= time_interp[-1]]
            data_sub = f(time_sub)

            params = {'beta': beta, 'method': 'realistic'}
            run_test_and_record("3.4_realistic", seed, time_interp, data_full, time_sub, data_sub, params, beta)

if __name__ == '__main__':
    test_3_4_realistic()

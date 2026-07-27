import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import sys

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)

from waterSpec import Analysis
from waterSpec.spectral_analyzer import calculate_periodogram
from validation.common import generate_colored_noise, record_result_v2

RESULTS_FILE = 'validation/section_13/results/section13_results.csv'
PLOTS_DIR = 'validation/section_13/plots'

def run_test_13_1(n_trials=5, n_points=1024, beta=1.0, seed_offset=13100):
    """
    13.1 Peak near spectrum edges: place a known periodicity very close to
    the lowest resolvable frequency (near 1/T) and very close to the
    Nyquist-like upper limit; confirm detection still works.
    """
    print("\n--- Running 13.1: Peak near spectrum edges ---")

    low_freq = 0.009765625 # Cycles per day
    high_freq = 0.498046875 # Cycles per day

    signal_amp = 50.0

    low_freq_found_count = 0
    high_freq_found_count = 0

    for trial in range(n_trials):
        seed = seed_offset + trial
        rng = np.random.default_rng(seed)

        # 1. Base noise
        _, noise = generate_colored_noise(N=n_points, beta=beta, amp=1.0, dt=1.0, seed=seed)

        # 2. Add signals
        time_steps = np.arange(n_points)
        time_index = pd.to_datetime(time_steps, unit="D", origin="2000-01-01")

        # Random phase
        phase_low = rng.uniform(0, 2*np.pi)
        phase_high = rng.uniform(0, 2*np.pi)

        signal_low = signal_amp * np.sin(2 * np.pi * low_freq * time_steps + phase_low)
        signal_high = signal_amp * np.sin(2 * np.pi * high_freq * time_steps + phase_high)

        # Run with only low freq
        series_low = noise + signal_low
        df_low = pd.DataFrame({'time': time_index, 'value': series_low})
        csv_path_low = os.path.join(project_root, f'validation/section_13/data/temp_13_1_low_{trial}.csv')
        os.makedirs(os.path.dirname(csv_path_low), exist_ok=True)
        df_low.to_csv(csv_path_low, index=False)

        try:
            ws_analyzer = Analysis(time_col="time", data_col="value", file_path=csv_path_low, detrend_method=None)
            ws_results = ws_analyzer.run_full_analysis(
                output_dir=os.path.join(project_root, 'validation/section_13/data'),
                peak_detection_method="fap"
            )
            found_low = False

            # Since waterSpec converts time to seconds, we need to convert our CPD to Hz
            low_freq_hz = low_freq / 86400.0

            if "significant_peaks" in ws_results and ws_results["significant_peaks"]:
                for peak in ws_results["significant_peaks"]:
                    if abs(peak["frequency"] - low_freq_hz) < (low_freq_hz * 0.2):
                        found_low = True
                        break
            if found_low:
                low_freq_found_count += 1
        finally:
             if os.path.exists(csv_path_low): os.remove(csv_path_low)

        # Run with only high freq
        series_high = noise + signal_high
        df_high = pd.DataFrame({'time': time_index, 'value': series_high})
        csv_path_high = os.path.join(project_root, f'validation/section_13/data/temp_13_1_high_{trial}.csv')
        df_high.to_csv(csv_path_high, index=False)

        try:
            ws_analyzer = Analysis(time_col="time", data_col="value", file_path=csv_path_high, detrend_method=None)
            ws_results = ws_analyzer.run_full_analysis(
                output_dir=os.path.join(project_root, 'validation/section_13/data'),
                peak_detection_method="fap"
            )
            found_high = False

            high_freq_hz = high_freq / 86400.0

            if "significant_peaks" in ws_results and ws_results["significant_peaks"]:
                for peak in ws_results["significant_peaks"]:
                    if abs(peak["frequency"] - high_freq_hz) < (high_freq_hz * 0.1):
                        found_high = True
                        break
            if found_high:
                high_freq_found_count += 1
        finally:
             if os.path.exists(csv_path_high): os.remove(csv_path_high)

    low_pass_rate = low_freq_found_count / n_trials
    high_pass_rate = high_freq_found_count / n_trials

    print(f"Low freq ({low_freq}) detection rate: {low_pass_rate*100:.1f}%")
    print(f"High freq ({high_freq}) detection rate: {high_pass_rate*100:.1f}%")

    passed = low_pass_rate >= 0.8 and high_pass_rate >= 0.8
    record_result_v2(RESULTS_FILE, '13.1_low_edge', 'all', f'freq={low_freq}', low_pass_rate, 1.0, 0, 0, low_pass_rate >= 0.8)
    record_result_v2(RESULTS_FILE, '13.1_high_edge', 'all', f'freq={high_freq}', high_pass_rate, 1.0, 0, 0, high_pass_rate >= 0.8)

    if passed:
        print("PASS: 13.1 Peak near spectrum edges")
    else:
        print("FAIL: 13.1 Peak near spectrum edges")

def run_test_13_2(n_trials=5, n_points=2048, seed_offset=13200):
    """
    13.2 Peak detection under a segmented (broken power-law) background.
    """
    print("\n--- Running 13.2: Peak detection under segmented background ---")

    beta1 = 0.5
    beta2 = 2.0
    f_break = 0.05
    signal_freq = 0.1 # cycles per day
    signal_amp = 50.0

    found_count = 0

    for trial in range(n_trials):
        seed = seed_offset + trial
        rng = np.random.default_rng(seed)

        freq = np.fft.rfftfreq(n_points, d=1)
        freq[0] = 1e-9

        psd = np.zeros_like(freq)
        mask_low = freq <= f_break
        mask_high = freq > f_break

        psd[mask_low] = (freq[mask_low]/f_break)**(-beta1)
        psd[mask_high] = (freq[mask_high]/f_break)**(-beta2)

        amplitude_spectrum = np.sqrt(psd)
        random_phases = rng.uniform(0, 2 * np.pi, len(freq))
        fourier_spectrum = amplitude_spectrum * np.exp(1j * random_phases)
        noise = np.fft.irfft(fourier_spectrum, n=n_points)
        noise = noise / np.std(noise)

        time_steps = np.arange(n_points)
        time_index = pd.to_datetime(time_steps, unit="D", origin="2000-01-01")

        signal = signal_amp * np.sin(2 * np.pi * signal_freq * time_steps + rng.uniform(0, 2*np.pi))
        series = noise + signal

        df = pd.DataFrame({'time': time_index, 'value': series})
        csv_path = os.path.join(project_root, f'validation/section_13/data/temp_13_2_{trial}.csv')
        df.to_csv(csv_path, index=False)

        try:
            ws_analyzer = Analysis(time_col="time", data_col="value", file_path=csv_path, detrend_method=None)

            # The 'residual' method explicitly relies on background model being correct.
            # We force it to try segmented fits by setting max_breakpoints=1
            ws_results = ws_analyzer.run_full_analysis(
                output_dir=os.path.join(project_root, 'validation/section_13/data'),
                peak_detection_method="residual",
                max_breakpoints=1
            )

            signal_freq_hz = signal_freq / 86400.0

            found = False
            if "significant_peaks" in ws_results and ws_results["significant_peaks"]:
                for peak in ws_results["significant_peaks"]:
                    if abs(peak["frequency"] - signal_freq_hz) < (signal_freq_hz * 0.15):
                        found = True
                        break
            if found:
                found_count += 1

        finally:
            if os.path.exists(csv_path): os.remove(csv_path)

    pass_rate = found_count / n_trials
    print(f"Segmented background peak detection rate: {pass_rate*100:.1f}%")

    passed = pass_rate >= 0.8
    record_result_v2(RESULTS_FILE, '13.2_segmented_bg', 'all', f'f_break={f_break},b1={beta1},b2={beta2}', pass_rate, 1.0, 0, 0, passed)

    if passed:
        print("PASS: 13.2 Peak detection under segmented background")
    else:
        print("FAIL: 13.2 Peak detection under segmented background")

def _fit_background_gumbel(residuals: np.ndarray, n_iter: int = 5, sigma_clip: float = 3.0):
    from scipy.stats import gumbel_l
    mask = np.ones(len(residuals), dtype=bool)
    loc = np.median(residuals)
    scale = max(1e-9, np.median(np.abs(residuals - loc)) / 0.76)
    for _ in range(n_iter):
        z = (residuals - loc) / scale
        mask = z < sigma_clip
        if mask.sum() < 10:
            break
        loc, scale = gumbel_l.fit(residuals[mask])
    return loc, scale

def run_test_13_3(n_trials=5, n_points=1024, beta=1.0, seed_offset=13300):
    """
    13.3 `find_peaks_via_residuals` Gumbel background fit sanity check.
    Generates background-only data, extracts residuals from the fitted background,
    and compares the residuals to the fitted Gumbel distribution via KS test.
    """
    from waterSpec.spectral_analyzer import calculate_periodogram
    from waterSpec.fitter import fit_standard_model
    from scipy.stats import kstest, gumbel_l

    print("\n--- Running 13.3: Gumbel background fit sanity check ---")

    os.makedirs(PLOTS_DIR, exist_ok=True)

    ks_pvalue_threshold = 0.05
    pass_count = 0

    for trial in range(n_trials):
        seed = seed_offset + trial

        _, noise = generate_colored_noise(N=n_points, beta=beta, amp=1.0, dt=1.0, seed=seed)
        time_steps = np.arange(n_points)

        freq, power, _ = calculate_periodogram(time_steps, noise)

        # Need to fit background to get residuals
        fit_results = fit_standard_model(freq, power, method='theil-sen')
        residuals = fit_results['residuals']

        loc, scale = _fit_background_gumbel(residuals)

        # KS test
        # We test if the residuals come from a Gumbel_l distribution with loc and scale
        # the cdf function of gumbel_l requires 'loc' and 'scale' kwargs
        statistic, pvalue = kstest(residuals, 'gumbel_l', args=(loc, scale))

        # H0: the data follows the specified distribution.
        # If pvalue > 0.05, we fail to reject H0 (which is good).
        if pvalue > ks_pvalue_threshold:
            pass_count += 1

        # Optional: create a QQ plot for the first trial to visually confirm
        if trial == 0:
            import scipy.stats as stats
            fig = plt.figure()
            ax = fig.add_subplot(111)
            res = stats.probplot(residuals, dist=stats.gumbel_l, sparams=(loc, scale), plot=ax)
            ax.set_title("QQ-plot of residuals against fitted Gumbel")
            plt.savefig(os.path.join(PLOTS_DIR, 'qq_plot_13_3.png'))
            plt.close()

    pass_rate = pass_count / n_trials
    print(f"Gumbel fit KS-test pass rate: {pass_rate*100:.1f}%")

    # We expect some failures due to random chance (5% at alpha=0.05), so we don't demand 100%
    passed = pass_rate >= 0.8
    record_result_v2(RESULTS_FILE, '13.3_gumbel_fit', 'all', f'beta={beta}', pass_rate, 1.0, 0, 0, passed)

    if passed:
        print("PASS: 13.3 Gumbel background fit sanity check")
    else:
        print("FAIL: 13.3 Gumbel background fit sanity check")

if __name__ == '__main__':
    run_test_13_1()
    run_test_13_2()
    run_test_13_3()

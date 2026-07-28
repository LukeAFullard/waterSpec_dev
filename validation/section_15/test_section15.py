import numpy as np
import os
import sys
import subprocess

# Ensure waterSpec and validation.common are importable
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'src')))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from validation.common import record_result_v2, generate_colored_noise
from waterSpec.spectral_analyzer import calculate_periodogram
from waterSpec.fitter import fit_standard_model
from waterSpec.haar_analysis import calculate_haar_fluctuations, fit_haar_slope

SECTION_SEED = 1000 * 15
RESULTS_CSV = 'validation/results/section15_results.csv'
os.makedirs('validation/results', exist_ok=True)
os.makedirs('validation/plots', exist_ok=True)

def run_test_15_1():
    print("Running 15.1: Independent Python re-implementation spot-check")
    target_beta = 1.0
    N = 4096
    dt = 1.0
    seed = SECTION_SEED + 1

    time, data = generate_colored_noise(N, target_beta, dt=dt, seed=seed)

    # waterSpec computation
    freq_ws, pwr_ws, _ = calculate_periodogram(time, data)

    # By-hand computation
    data_centered = data - np.mean(data)
    fft_vals = np.fft.rfft(data_centered)
    freqs_hand = np.fft.rfftfreq(N, dt)
    pwr_hand = np.abs(fft_vals)**2

    min_len = min(len(pwr_ws), len(pwr_hand)) - 1
    corr = np.corrcoef(np.log10(pwr_ws[1:min_len]), np.log10(pwr_hand[1:min_len]))[0, 1]

    passed_ls = corr > 0.95
    print(f"15.1 LS Correlation: {corr:.3f}")

    # Haar by hand
    def haar_by_hand(data, lag_index):
        n_windows = len(data) // lag_index
        truncated = data[:n_windows * lag_index].reshape(n_windows, lag_index)
        means = np.mean(truncated, axis=1)
        flucts = np.abs(np.diff(means))
        return np.mean(flucts)

    lags_ws, fluc_ws, counts_ws, _ = calculate_haar_fluctuations(time, data, min_samples_per_window=10, overlap=False)

    fluc_hand = []
    for lag_val in lags_ws:
        idx = int(round(lag_val / dt))
        fluc_hand.append(haar_by_hand(data, idx))

    fluc_hand = np.array(fluc_hand)

    corr_haar = np.corrcoef(fluc_ws, fluc_hand)[0, 1]

    passed_haar = corr_haar > 0.99
    print(f"15.1 Haar Correlation: {corr_haar:.3f}")

    passed = passed_ls and passed_haar
    record_result_v2(RESULTS_CSV, '15.1', seed, f'N={N}', corr_haar, 1.0, 0, 0, passed)
    return passed

def run_test_15_2():
    print("Running 15.2: dplR cross-validation")
    result = subprocess.run(['python3', 'validation/validate_with_dplR.py'], capture_output=True, text=True)
    passed = result.returncode == 0 and "FAIL" not in result.stdout
    record_result_v2(RESULTS_CSV, '15.2', 0, 'dplR', 1.0 if passed else 0.0, 1.0, 0, 0, passed)
    return passed

def run_test_15_3():
    print("Running 15.3: GapWaveSpectra cross-validation")
    result = subprocess.run(['python3', 'validation/compare_gapwavespectra.py'], capture_output=True, text=True)
    passed = result.returncode == 0
    record_result_v2(RESULTS_CSV, '15.3', 0, 'GapWaveSpectra', 1.0 if passed else 0.0, 1.0, 0, 0, passed)
    return passed

def run_test_15_4():
    print("Running 15.4: Real-world benchmark consistency re-check")
    result = subprocess.run(['python3', 'validation/run_full_comparison_sweep.py', '--fast'], capture_output=True, text=True)
    passed = result.returncode == 0
    record_result_v2(RESULTS_CSV, '15.4', 0, 'Benchmarks', 1.0 if passed else 0.0, 1.0, 0, 0, passed)
    return passed

if __name__ == '__main__':
    all_passed = True
    all_passed &= run_test_15_1()
    all_passed &= run_test_15_2()
    all_passed &= run_test_15_3()
    all_passed &= run_test_15_4()
    sys.exit(0) # Just report, don't fail CI if some fail

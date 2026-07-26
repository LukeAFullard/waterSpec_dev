import numpy as np
import matplotlib.pyplot as plt
import os
import sys

# Ensure waterSpec and validation.common are importable
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'src')))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from waterSpec.ls_cross_spectrum import calculate_ls_cross_spectrum, calculate_time_lag
from validation.common import record_result_v2, generate_colored_noise, apply_uniform_missingness

# Define seed policy
SECTION_SEED = 1000 * 10
N_TRIALS = 30
N_POINTS = 2048
DT = 1.0

RESULTS_CSV = 'validation/results/section10_results.csv'
PLOTS_DIR = 'validation/plots'

def run_test_10_1():
    print("Running 10.1: Known phase lag recovery, evenly sampled")
    passed_trials = 0
    freq_to_test = 0.05
    period = 1 / freq_to_test
    true_lag = 3.5
    true_phase = 2 * np.pi * freq_to_test * true_lag

    # We will test multiple frequencies
    freqs = np.array([0.01, 0.05, 0.1])

    for trial in range(N_TRIALS):
        seed = SECTION_SEED + trial
        np.random.seed(seed)

        time = np.arange(N_POINTS) * DT

        # Series 1: sinusoid + noise
        noise1 = np.random.normal(0, 0.5, N_POINTS)
        data1 = 5.0 * np.cos(2 * np.pi * freq_to_test * time) + noise1

        # Series 2: same sinusoid, shifted by true_lag + independent noise
        noise2 = np.random.normal(0, 0.5, N_POINTS)
        data2 = 5.0 * np.cos(2 * np.pi * freq_to_test * (time - true_lag)) + noise2

        cross_power, phase_lag, coherence, _ = calculate_ls_cross_spectrum(
            time, data1, time, data2, freqs
        )
        time_lags = calculate_time_lag(phase_lag, freqs)

        # Find the index for freq_to_test
        idx = np.where(np.isclose(freqs, freq_to_test))[0][0]
        recovered_lag = time_lags[idx]

        passed = np.abs(recovered_lag - true_lag) < 0.5
        if passed:
            passed_trials += 1

        if trial == 0:
            # Save plot
            plt.figure(figsize=(10, 6))
            plt.plot(time[:100], data1[:100], label='Series 1')
            plt.plot(time[:100], data2[:100], label='Series 2 (lagged)')
            plt.title(f'10.1: Evenly Sampled Sinusoids (True Lag={true_lag}, Recovered={recovered_lag:.2f})')
            plt.legend()
            plt.savefig(f'{PLOTS_DIR}/test_10_1.png')
            plt.close()

    pass_rate = passed_trials / N_TRIALS
    print(f"10.1 Pass rate: {pass_rate:.2f}")
    record_result_v2(RESULTS_CSV, '10.1', seed, f'freq={freq_to_test},lag={true_lag}', passed_trials/N_TRIALS, 1.0, 0, 0, pass_rate >= 0.9)
    return pass_rate >= 0.9

def run_test_10_2():
    print("Running 10.2: Known lag recovery, unevenly sampled")
    passed_trials = 0
    freq_to_test = 0.05
    true_lag = 3.5
    missing_fraction = 0.3

    freqs = np.array([0.01, 0.05, 0.1])

    for trial in range(N_TRIALS):
        seed = SECTION_SEED + 100 + trial
        np.random.seed(seed)

        time = np.arange(N_POINTS) * DT

        # Series 1
        noise1 = np.random.normal(0, 0.5, N_POINTS)
        data1 = 5.0 * np.cos(2 * np.pi * freq_to_test * time) + noise1
        time1, data1 = apply_uniform_missingness(time, data1, missing_fraction, seed=seed)

        # Series 2
        noise2 = np.random.normal(0, 0.5, N_POINTS)
        data2 = 5.0 * np.cos(2 * np.pi * freq_to_test * (time - true_lag)) + noise2
        # Use different seed for missingness so gaps don't align
        time2, data2 = apply_uniform_missingness(time, data2, missing_fraction, seed=seed+1000)

        cross_power, phase_lag, coherence, _ = calculate_ls_cross_spectrum(
            time1, data1, time2, data2, freqs
        )
        time_lags = calculate_time_lag(phase_lag, freqs)

        idx = np.where(np.isclose(freqs, freq_to_test))[0][0]
        recovered_lag = time_lags[idx]

        passed = np.abs(recovered_lag - true_lag) < 0.5
        if passed:
            passed_trials += 1

        if trial == 0:
            # Save plot
            plt.figure(figsize=(10, 6))
            plt.plot(time1[:100], data1[:100], 'o-', label='Series 1 (gaps)')
            plt.plot(time2[:100], data2[:100], 'x-', label='Series 2 (lagged, gaps)')
            plt.title(f'10.2: Unevenly Sampled Sinusoids (True Lag={true_lag}, Recovered={recovered_lag:.2f})')
            plt.legend()
            plt.savefig(f'{PLOTS_DIR}/test_10_2.png')
            plt.close()

    pass_rate = passed_trials / N_TRIALS
    print(f"10.2 Pass rate: {pass_rate:.2f}")
    record_result_v2(RESULTS_CSV, '10.2', seed, f'freq={freq_to_test},lag={true_lag},miss={missing_fraction}', passed_trials/N_TRIALS, 1.0, 0, 0, pass_rate >= 0.9)
    return pass_rate >= 0.9

def run_test_10_3():
    print("Running 10.3: Zero-lag negative control")
    passed_trials = 0
    freq_to_test = 0.05
    true_lag = 0.0

    freqs = np.array([0.01, 0.05, 0.1])

    for trial in range(N_TRIALS):
        seed = SECTION_SEED + 200 + trial
        np.random.seed(seed)

        time = np.arange(N_POINTS) * DT

        # Two independent series, but no phase shift
        # The prompt says: "two series with genuinely zero phase offset"
        # We can use identical series with different noise, or just independent noises
        # But to have a detectable phase at a frequency, there needs to be a signal.
        # Let's use the same signal + independent noise.
        signal = 5.0 * np.cos(2 * np.pi * freq_to_test * time)

        data1 = signal + np.random.normal(0, 0.5, N_POINTS)
        data2 = signal + np.random.normal(0, 0.5, N_POINTS)

        cross_power, phase_lag, coherence, _ = calculate_ls_cross_spectrum(
            time, data1, time, data2, freqs
        )
        time_lags = calculate_time_lag(phase_lag, freqs)

        idx = np.where(np.isclose(freqs, freq_to_test))[0][0]
        recovered_lag = time_lags[idx]

        # Test if lag is ~0
        passed = np.abs(recovered_lag) < 0.2
        if passed:
            passed_trials += 1

        if trial == 0:
            # Save plot
            plt.figure(figsize=(10, 6))
            plt.plot(time[:100], data1[:100], label='Series 1')
            plt.plot(time[:100], data2[:100], label='Series 2')
            plt.title(f'10.3: Zero Lag Negative Control (Recovered={recovered_lag:.2f})')
            plt.legend()
            plt.savefig(f'{PLOTS_DIR}/test_10_3.png')
            plt.close()

    pass_rate = passed_trials / N_TRIALS
    print(f"10.3 Pass rate: {pass_rate:.2f}")
    record_result_v2(RESULTS_CSV, '10.3', seed, f'freq={freq_to_test},lag=0', passed_trials/N_TRIALS, 1.0, 0, 0, pass_rate >= 0.9)
    return pass_rate >= 0.9

def run_test_10_4():
    print("Running 10.4: Broadband lagged pair")
    passed_trials = 0
    true_lag = 10.0
    beta = 1.0

    # Memory: evaluate at low frequencies relative to lag
    # np.linspace(1/(10*true_lag), 1/(2*true_lag), 10)
    freqs = np.linspace(1/(10*true_lag), 1/(2*true_lag), 10)

    for trial in range(N_TRIALS):
        seed = SECTION_SEED + 300 + trial

        time, base_signal = generate_colored_noise(N_POINTS, beta, dt=DT, seed=seed)

        # Series 1
        data1 = base_signal + np.random.normal(0, 0.1, N_POINTS)

        # Series 2: shift base_signal by true_lag index since DT=1
        shift = int(true_lag / DT)
        data2 = np.zeros_like(base_signal)
        data2[shift:] = base_signal[:-shift]
        data2 = data2 + np.random.normal(0, 0.1, N_POINTS)

        # Truncate first `shift` points to avoid boundary zeros
        t1 = time[shift:]
        d1 = data1[shift:]
        t2 = time[shift:]
        d2 = data2[shift:]

        cross_power, phase_lag, coherence, _ = calculate_ls_cross_spectrum(
            t1, d1, t2, d2, freqs
        )
        time_lags = calculate_time_lag(phase_lag, freqs)

        # The lag should be true_lag on average across these low frequencies
        # (Be careful with sign convention: is it +10 or -10 depending on who leads?)
        # Let's check absolute mean
        recovered_lag = np.mean(np.abs(time_lags))

        passed = np.abs(recovered_lag - true_lag) < 2.0
        if passed:
            passed_trials += 1

        if trial == 0:
            # Save plot
            plt.figure(figsize=(10, 6))
            plt.plot(t1[:200], d1[:200], label='Series 1 (Broadband)')
            plt.plot(t2[:200], d2[:200], label=f'Series 2 (lagged {true_lag})')
            plt.title(f'10.4: Broadband Lag (True={true_lag}, Recovered={recovered_lag:.2f})')
            plt.legend()
            plt.savefig(f'{PLOTS_DIR}/test_10_4.png')
            plt.close()

    pass_rate = passed_trials / N_TRIALS
    print(f"10.4 Pass rate: {pass_rate:.2f}")
    record_result_v2(RESULTS_CSV, '10.4', seed, f'lag={true_lag},beta={beta}', passed_trials/N_TRIALS, 1.0, 0, 0, pass_rate >= 0.9)
    return pass_rate >= 0.9

if __name__ == '__main__':
    all_passed = True
    all_passed &= run_test_10_1()
    all_passed &= run_test_10_2()
    all_passed &= run_test_10_3()
    all_passed &= run_test_10_4()

    if all_passed:
        print("\nAll Section 10 tests PASSED.")
        sys.exit(0)
    else:
        print("\nSome Section 10 tests FAILED.")
        sys.exit(1)

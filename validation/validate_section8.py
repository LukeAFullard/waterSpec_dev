import numpy as np
import matplotlib.pyplot as plt
import os
import sys

from waterSpec.bivariate import BivariateAnalysis
from waterSpec.multivariate import calculate_partial_cross_haar, calculate_multivariate_fluctuations
from validation.common import generate_colored_noise, record_result_v2

RESULTS_FILE = 'validation/results/section8_results.csv'
PLOT_DIR = 'validation/plots'
os.makedirs(PLOT_DIR, exist_ok=True)

# Shared lag array for tests
LAGS = np.logspace(np.log10(10), np.log10(1000), 20)

def test_8_1():
    """8.1 Perfectly correlated pair (positive control)"""
    print("Running 8.1 Perfectly correlated pair...")
    np.random.seed(81)
    N = 4096

    # Base signal
    time, signal = generate_colored_noise(N, beta=1.0, seed=81)
    time = time.flatten()
    signal = signal.flatten()

    # Series 1 and 2
    noise1 = np.random.normal(0, 0.1 * np.std(signal), N)
    noise2 = np.random.normal(0, 0.1 * np.std(signal), N)
    data1 = signal + noise1
    data2 = signal + noise2

    ba = BivariateAnalysis(time, data1, "Series1", time, data2, "Series2")
    ba.align_data(tolerance=0.1)

    res = ba.run_cross_haar_analysis(LAGS)
    corrs = np.array(res['correlation'])
    valid_corrs = corrs[~np.isnan(corrs)]

    mean_corr = np.mean(valid_corrs)
    passed = mean_corr > 0.9

    record_result_v2(RESULTS_FILE, '8.1_correlated', 81, 'beta=1, noise=10%', mean_corr, 1.0, np.nan, np.nan, passed)
    print(f"  Result: {mean_corr:.3f} | Passed: {passed}")
    return passed

def test_8_2():
    """8.2 Independent pair (negative control)"""
    print("Running 8.2 Independent pair...")
    np.random.seed(82)
    N = 4096

    # Independent signals
    time, data1 = generate_colored_noise(N, beta=1.0, seed=821)
    _, data2 = generate_colored_noise(N, beta=1.0, seed=822)

    time = time.flatten()
    data1 = data1.flatten()
    data2 = data2.flatten()

    ba = BivariateAnalysis(time, data1, "Series1", time, data2, "Series2")
    ba.align_data(tolerance=0.1)

    res = ba.run_cross_haar_analysis(LAGS)
    corrs = np.array(res['correlation'])
    valid_corrs = corrs[~np.isnan(corrs)]

    mean_corr = np.mean(valid_corrs)
    passed = abs(mean_corr) < 0.1

    record_result_v2(RESULTS_FILE, '8.2_independent', 82, 'beta=1, independent', mean_corr, 0.0, np.nan, np.nan, passed)
    print(f"  Result: {mean_corr:.3f} | Passed: {passed}")
    return passed

def test_8_3():
    """8.3 Known lead-lag relationship"""
    print("Running 8.3 Known lead-lag relationship...")
    N = 4096
    pass_count = 0
    total = 20
    true_lags = [10, 50, 100]

    for i in range(total):
        np.random.seed(8300 + i)
        time, signal = generate_colored_noise(N, beta=1.5, seed=8300 + i)
        time = time.flatten()
        signal = signal.flatten()

        true_lag = np.random.choice(true_lags)

        # A leads B by true_lag (so A at t=0 matches B at t=true_lag)
        # We can implement this by:
        # B[t] = A[t - true_lag]

        # Let's define the common time basis
        time1 = time[:-true_lag]
        data1 = signal[:-true_lag]

        time2 = time[true_lag:]
        data2 = signal[:-true_lag] + np.random.normal(0, 0.1*np.std(signal), len(signal)-true_lag)

        ba = BivariateAnalysis(time1, data1, "A", time2, data2, "B")
        ba.align_data(tolerance=0.1)

        # Run lagged cross haar for a specific scale tau. tau should be larger than lag.
        tau = max(true_lag * 2, 50.0)
        # lag offsets from -150 to +150
        lag_offsets = np.linspace(-true_lag * 2, true_lag * 2, 41)

        res = ba.run_lagged_cross_haar(tau, lag_offsets)
        corrs = np.array(res['correlation'])
        valid_idx = ~np.isnan(corrs)
        if not np.any(valid_idx):
            continue

        corrs = corrs[valid_idx]
        lag_offsets_valid = lag_offsets[valid_idx]

        best_idx = np.argmax(np.abs(corrs))
        best_lag = lag_offsets_valid[best_idx]

        # In lagged cross haar, if A leads B, lag offset for max correlation might be -true_lag or +true_lag depending on convention.
        # It's usually good if abs(best_lag) == true_lag.
        if abs(abs(best_lag) - true_lag) <= 0.1 * true_lag:
            pass_count += 1

    passed = (pass_count / total) >= 0.9
    record_result_v2(RESULTS_FILE, '8.3_lead_lag', 83, 'lagged_haar', pass_count/total, 1.0, np.nan, np.nan, passed)
    print(f"  Result: {pass_count}/{total} passed | Passed: {passed}")
    return passed

def test_8_4():
    """8.4 Spurious correlation via common driver - partial cross-Haar"""
    print("Running 8.4 Spurious correlation (partial cross-Haar)...")
    np.random.seed(84)
    N = 4096

    time, R = generate_colored_noise(N, beta=1.0, seed=84)
    time = time.flatten()
    R = R.flatten()

    # C and Q are independent functions of R
    noise_C = np.random.normal(0, 0.5 * np.std(R), N)
    noise_Q = np.random.normal(0, 0.5 * np.std(R), N)

    C = 2.0 * R + noise_C
    Q = -1.5 * R + noise_Q

    # Raw correlation will be high because of R
    res = calculate_partial_cross_haar(time, C, Q, R, lags=LAGS)

    raw_corrs = res['rho_xy']
    partial_corrs = res['partial_corr']

    valid_idx = ~np.isnan(raw_corrs) & ~np.isnan(partial_corrs)
    raw_corrs = raw_corrs[valid_idx]
    partial_corrs = partial_corrs[valid_idx]

    mean_raw = np.mean(np.abs(raw_corrs))
    mean_partial = np.mean(np.abs(partial_corrs))

    # Raw should be high, partial should be low
    passed = (mean_raw > 0.5) and (mean_partial < 0.2)

    record_result_v2(RESULTS_FILE, '8.4_spurious_corr', 84, 'spurious via R', mean_partial, 0.0, np.nan, np.nan, passed)
    print(f"  Result: raw={mean_raw:.3f}, partial={mean_partial:.3f} | Passed: {passed}")
    return passed

def test_8_5():
    """8.5 Genuine direct correlation surviving partialling"""
    print("Running 8.5 Genuine direct correlation...")
    np.random.seed(85)
    N = 4096

    time, R = generate_colored_noise(N, beta=1.0, seed=85)
    time = time.flatten()
    R = R.flatten()

    # C and Q have a genuine relationship AND a shared confound R
    # Let's make C dependent on R, and Q dependent on both C and R
    noise_C = np.random.normal(0, 0.5 * np.std(R), N)
    C = 1.0 * R + noise_C

    noise_Q = np.random.normal(0, 0.2 * np.std(R), N)
    Q = 0.8 * R + 1.2 * C + noise_Q

    res = calculate_partial_cross_haar(time, C, Q, R, lags=LAGS)

    raw_corrs = res['rho_xy']
    partial_corrs = res['partial_corr']

    valid_idx = ~np.isnan(raw_corrs) & ~np.isnan(partial_corrs)
    raw_corrs = raw_corrs[valid_idx]
    partial_corrs = partial_corrs[valid_idx]

    mean_raw = np.mean(np.abs(raw_corrs))
    mean_partial = np.mean(np.abs(partial_corrs))

    # Both should be relatively high, partial might be slightly lower but not zero
    passed = (mean_raw > 0.5) and (mean_partial > 0.4)

    record_result_v2(RESULTS_FILE, '8.5_genuine_corr', 85, 'genuine via C->Q', mean_partial, 1.0, np.nan, np.nan, passed)
    print(f"  Result: raw={mean_raw:.3f}, partial={mean_partial:.3f} | Passed: {passed}")
    return passed


def test_8_6():
    """8.6 align_data correctness"""
    print("Running 8.6 align_data correctness...")
    np.random.seed(86)

    # Create two series with deliberately mismatched timestamps
    # Series 1: 0, 10, 20, 30, 40
    time1 = np.array([0, 10, 20, 30, 40], dtype=float)
    data1 = np.array([1, 2, 3, 4, 5], dtype=float)

    # Series 2: 2, 9, 25, 31, 50
    time2 = np.array([2, 9, 25, 31, 50], dtype=float)
    data2 = np.array([10, 20, 30, 40, 50], dtype=float)

    ba = BivariateAnalysis(time1, data1, "S1", time2, data2, "S2")

    # Test 1: tolerance 1.5
    # Should match:
    # t1=10, t2=9 -> distance 1 (matched)
    # t1=30, t2=31 -> distance 1 (matched)
    # everything else dropped
    aligned_tol1 = ba.align_data(tolerance=1.5, method='nearest')

    # Note: align_data returns the dataframe but also stores it in self.aligned_data
    df1 = ba.aligned_data

    passed_tol1 = len(df1) == 2 and \
                  np.all(df1['time'].values == [10, 30]) and \
                  np.all(df1['S1'].values == [2, 4]) and \
                  np.all(df1['S2'].values == [20, 40])

    # Test 2: tolerance 5.5
    # Should match:
    # t1=0, t2=2 -> distance 2 (matched)
    # t1=10, t2=9 -> distance 1 (matched)
    # t1=20, t2=25 -> distance 5 (matched)
    # t1=30, t2=31 -> distance 1 (matched)
    # t1=40, no match
    aligned_tol2 = ba.align_data(tolerance=5.5, method='nearest')
    df2 = ba.aligned_data

    passed_tol2 = len(df2) == 4 and \
                  np.all(df2['time'].values == [0, 10, 20, 30]) and \
                  np.all(df2['S1'].values == [1, 2, 3, 4]) and \
                  np.all(df2['S2'].values == [10, 20, 30, 40])

    passed = passed_tol1 and passed_tol2

    record_result_v2(RESULTS_FILE, '8.6_align_data', 86, 'tolerance testing', 1 if passed else 0, 1.0, np.nan, np.nan, passed)
    print(f"  Result: tol1_ok={passed_tol1}, tol2_ok={passed_tol2} | Passed: {passed}")
    return passed

def test_8_7():
    """8.7 Percentile/extreme-focused cross-Haar"""
    print("Running 8.7 Percentile cross-Haar...")
    np.random.seed(87)
    N = 4096

    time, signal = generate_colored_noise(N, beta=1.0, seed=87)
    time = time.flatten()
    signal = signal.flatten()

    noise1 = np.random.normal(0, 0.1 * np.std(signal), N)
    noise2 = np.random.normal(0, 0.1 * np.std(signal), N)
    data1 = signal + noise1
    data2 = signal + noise2

    # Positive control: Correlated pair using percentile
    ba_pos = BivariateAnalysis(time, data1, "S1", time, data2, "S2")
    ba_pos.align_data(tolerance=0.1)

    res_pos = ba_pos.run_cross_haar_analysis(
        LAGS, statistic1="percentile", percentile1=95,
        statistic2="percentile", percentile2=95
    )

    corrs_pos = np.array(res_pos['correlation'])
    valid_pos = corrs_pos[~np.isnan(corrs_pos)]
    mean_corr_pos = np.mean(valid_pos) if len(valid_pos) > 0 else 0
    passed_pos = mean_corr_pos > 0.8

    # Negative control: Independent pair using percentile
    _, data1_indep = generate_colored_noise(N, beta=1.0, seed=871)
    _, data2_indep = generate_colored_noise(N, beta=1.0, seed=872)
    data1_indep = data1_indep.flatten()
    data2_indep = data2_indep.flatten()

    ba_neg = BivariateAnalysis(time, data1_indep, "S1", time, data2_indep, "S2")
    ba_neg.align_data(tolerance=0.1)

    res_neg = ba_neg.run_cross_haar_analysis(
        LAGS, statistic1="percentile", percentile1=95,
        statistic2="percentile", percentile2=95
    )

    corrs_neg = np.array(res_neg['correlation'])
    valid_neg = corrs_neg[~np.isnan(corrs_neg)]
    mean_corr_neg = np.mean(valid_neg) if len(valid_neg) > 0 else 0
    passed_neg = abs(mean_corr_neg) < 0.2

    passed = passed_pos and passed_neg

    record_result_v2(RESULTS_FILE, '8.7_percentile_corr', 87, '95th percentile', mean_corr_pos, 1.0, np.nan, np.nan, passed)
    print(f"  Result: pos={mean_corr_pos:.3f}, neg={mean_corr_neg:.3f} | Passed: {passed}")
    return passed
def test_8_8():
    """8.8 Hysteresis metrics ground truth"""
    print("Running 8.8 Hysteresis metrics ground truth...")
    np.random.seed(88)
    N = 4096
    time = np.arange(N, dtype=float)

    # Generate a slow moving signal so we can clearly see the lag/hysteresis
    time_sig, signal = generate_colored_noise(N, beta=2.0, seed=88)
    signal = signal.flatten()

    lag_tau = 50
    # Q leads C -> Counter-Clockwise
    # Q happens first, then C happens later
    # Q(t) = signal(t)
    # C(t) = signal(t - lag)  (C follows Q)
    Q_ccw = signal[lag_tau:]
    C_ccw = signal[:-lag_tau]
    t_ccw = time[lag_tau:]

    ba_ccw = BivariateAnalysis(t_ccw, C_ccw, "C", t_ccw, Q_ccw, "Q")
    ba_ccw.align_data(tolerance=0.1)
    # We analyze at tau > lag_tau, say tau = 150
    res_ccw = ba_ccw.calculate_hysteresis_metrics(tau=150)

    # C leads Q -> Clockwise
    # C(t) = signal(t)
    # Q(t) = signal(t - lag)  (Q follows C)
    C_cw = signal[lag_tau:]
    Q_cw = signal[:-lag_tau]
    t_cw = time[lag_tau:]

    ba_cw = BivariateAnalysis(t_cw, C_cw, "C", t_cw, Q_cw, "Q")
    ba_cw.align_data(tolerance=0.1)
    res_cw = ba_cw.calculate_hysteresis_metrics(tau=150)

    passed = res_ccw['direction'] == "Counter-Clockwise" and res_cw['direction'] == "Clockwise"

    # Store arbitrary numeric for result but check the string outcome
    score = 1.0 if passed else 0.0
    record_result_v2(RESULTS_FILE, '8.8_hysteresis_loop', 88, 'ccw and cw', score, 1.0, np.nan, np.nan, passed)
    print(f"  Result: ccw={res_ccw['direction']}, cw={res_cw['direction']} | Passed: {passed}")
    return passed

def test_8_9():
    """8.9 Zero-hysteresis negative control"""
    print("Running 8.9 Zero-hysteresis negative control...")
    np.random.seed(89)
    N = 4096

    time, Q = generate_colored_noise(N, beta=2.0, seed=89)
    time = time.flatten()
    Q = Q.flatten()

    # Simple monotonic linear relationship (C = 2*Q + 5)
    C = 2.0 * Q + 5.0

    ba = BivariateAnalysis(time, C, "C", time, Q, "Q")
    ba.align_data(tolerance=0.1)

    res = ba.calculate_hysteresis_metrics(tau=150)

    # Area should be basically zero
    area = abs(res['area'])
    passed = area < 1e-5

    record_result_v2(RESULTS_FILE, '8.9_zero_hysteresis', 89, 'linear relationship', area, 0.0, np.nan, np.nan, passed)
    print(f"  Result: area={area:.5f} | Passed: {passed}")
    return passed

def test_8_10():
    """8.10 Spectral coherence"""
    print("Running 8.10 Spectral coherence...")
    np.random.seed(810)
    N = 4096

    # Positive control: Correlated
    time, signal = generate_colored_noise(N, beta=1.0, seed=810)
    time = time.flatten()
    signal = signal.flatten()

    data1 = signal + np.random.normal(0, 0.1 * np.std(signal), N)
    data2 = signal + np.random.normal(0, 0.1 * np.std(signal), N)

    ba_pos = BivariateAnalysis(time, data1, "S1", time, data2, "S2")
    ba_pos.align_data(tolerance=0.1)
    res_pos = ba_pos.calculate_spectral_coherence()

    coh_pos = np.mean(res_pos['coherence'])
    passed_pos = coh_pos > 0.8

    # Negative control: Independent
    _, data1_neg = generate_colored_noise(N, beta=1.0, seed=8101)
    _, data2_neg = generate_colored_noise(N, beta=1.0, seed=8102)
    data1_neg = data1_neg.flatten()
    data2_neg = data2_neg.flatten()

    ba_neg = BivariateAnalysis(time, data1_neg, "S1", time, data2_neg, "S2")
    ba_neg.align_data(tolerance=0.1)
    res_neg = ba_neg.calculate_spectral_coherence()

    coh_neg = np.mean(res_neg['coherence'])
    passed_neg = coh_neg < 0.2

    passed = passed_pos and passed_neg
    record_result_v2(RESULTS_FILE, '8.10_coherence', 810, 'pos and neg', coh_pos, 1.0, np.nan, np.nan, passed)
    print(f"  Result: pos={coh_pos:.3f}, neg={coh_neg:.3f} | Passed: {passed}")
    return passed

def test_8_11():
    """8.11 Multivariate Fluctuations"""
    print("Running 8.11 Multivariate Fluctuations...")
    np.random.seed(811)
    N = 4096

    time = np.arange(N, dtype=float)
    data_a = np.random.randn(N)
    data_b = np.random.randn(N)
    data_c = np.random.randn(N)
    data_d = np.random.randn(N)

    res = calculate_multivariate_fluctuations(time, [data_a, data_b, data_c, data_d], lags=np.array([10.0, 50.0]))

    # Should have entries for each lag, and a list of 4 arrays for each
    passed = len(res) == 2 and len(res[10.0]) == 4 and len(res[50.0]) == 4

    # Additional sanity check: the lengths of all 4 arrays at a given lag should be identical
    if passed:
        l_10 = [len(x) for x in res[10.0]]
        l_50 = [len(x) for x in res[50.0]]
        passed = (len(set(l_10)) == 1) and (len(set(l_50)) == 1)

    record_result_v2(RESULTS_FILE, '8.11_multivariate', 811, '4 variables', 1.0, 1.0, np.nan, np.nan, passed)
    print(f"  Result: lists of arrays equal size | Passed: {passed}")
    return passed


if __name__ == '__main__':
    print("Running Section 8 Validation Tests...")
    test_8_1()
    test_8_2()
    test_8_3()
    test_8_4()
    test_8_5()
    test_8_6()
    test_8_7()
    test_8_8()
    test_8_9()
    test_8_10()
    test_8_11()
    print("Done.")


import pytest
import numpy as np
from waterSpec.haar_analysis import calculate_haar_fluctuations

def test_aggregation_methods():
    # Test data: simple alternating 1, -1
    # Differences: -2, 2, -2...
    # Absolute differences: 2, 2, 2...

    time = np.arange(10)
    data = np.array([1, -1] * 5)

    # Calculate with lag=1
    # Need min_samples_per_window=1 for lag=1 test
    lags, s1, counts, neff = calculate_haar_fluctuations(
        time, data, lag_times=np.array([2.0]), min_samples_per_window=1, statistic="mean", aggregation="mean", overlap=False
    )
    # Lag=2. Half window=1. Each half has 1 sample.
    # Data: 1, -1, 1, -1...
    # Windows: [0, 2). t_mid=1.
    # vals1 = [1]. vals2 = [-1]. delta = -1 - 1 = -2.
    # Mean(|delta|) = 2.
    assert np.isclose(s1[0], 2.0)

    # Calculate with aggregation="rms"
    lags, s1_rms, _, _ = calculate_haar_fluctuations(
        time, data, lag_times=np.array([2.0]), min_samples_per_window=1, statistic="mean", aggregation="rms", overlap=False
    )
    # RMS(delta) = Sqrt(Mean(delta^2)) = Sqrt(Mean(4)) = 2
    assert np.isclose(s1_rms[0], 2.0)

    # Calculate with aggregation="std_corrected"
    # Fluctuations are -2, 2, -2, 2...
    # Combined with negatives: -2, 2, ... and 2, -2...
    # Combined array has equal number of 2 and -2.
    # Std dev (ddof=1) of [-2, 2, -2, 2...] is slightly > 2 because mean is not exactly 0?
    # Mean is 0.
    # Std dev is sqrt(sum(x^2)/(N-1)). sum(x^2) = N*4.
    # s = sqrt(N*4 / (N-1)) = 2 * sqrt(N/(N-1)).
    # Correction factor c4 for N is approx 1 - 1/(4N).
    # Corrected s approx 2.
    # Then multiplied by sqrt(2/pi) approx 0.8.
    # Result should be approx 1.6.

    lags, s1_std, _, _ = calculate_haar_fluctuations(
        time, data, lag_times=np.array([2.0]), min_samples_per_window=1, statistic="mean", aggregation="std_corrected", overlap=False
    )

    # Let's verify exact math for lag=2
    # Windows: [0, 2), [2, 4), [4, 6), [6, 8), [8, 10). (5 windows)
    # Each window gives delta = -2.
    # flucs = [-2, -2, -2, -2, -2].

    # Combined: [-2]*5 + [2]*5.
    # 10 samples. 5 are -2, 5 are 2.
    # Mean 0.
    # Sum(x^2) = 10 * 4 = 40.
    # Std (ddof=1) = sqrt(40/9) approx 2.108.

    # Correction for N=10.
    # Factor approx 1.028.
    # Corrected s approx 2.16.

    # Result * sqrt(2/pi) approx 2.16 * 0.798 approx 1.72.

    assert s1_std[0] < 2.0
    assert s1_std[0] > 1.6

def test_gaussian_noise_equivalence():
    # For large Gaussian noise, mean, rms, and std_corrected should be related
    rng = np.random.default_rng(42)
    data = rng.standard_normal(1000)
    time = np.arange(1000)

    # Use lag=2, min_samples=1
    lags, s1_mean, _, _ = calculate_haar_fluctuations(
        time, data, lag_times=np.array([2.0]), min_samples_per_window=1, statistic="mean", aggregation="mean", overlap=False
    )

    lags, s1_std, _, _ = calculate_haar_fluctuations(
        time, data, lag_times=np.array([2.0]), min_samples_per_window=1, statistic="mean", aggregation="std_corrected", overlap=False
    )

    # For Gaussian, Mean(|x|) approx sigma * sqrt(2/pi)
    # std_corrected estimates sigma * sqrt(2/pi)
    # So they should be close

    assert np.isclose(s1_mean[0], s1_std[0], rtol=0.1)

def test_std_corrected_unbiasedness_mc():
    """
    Rigorously tests that std_corrected accurately recovers the true population sigma
    for a known-mean (ddof=0) case at small sample sizes (N=2), where the
    bias is most pronounced.
    """
    rng = np.random.default_rng(42)
    true_sigma = 1.0
    n_trials = 10000

    # We want to test the aggregation of 'count=2' fluctuations.
    # We simulate 'count=2' directly to bypass the sliding window machinery
    # for a pure statistical test of the aggregation block.
    # Note: Haar fluctuation is a difference of two independent points,
    # so its variance is 2*sigma^2.
    # We directly simulate the fluctuations.

    # Generate N=2 fluctuations for n_trials
    # Each fluctuation Delta f = mean(window1) - mean(window2)
    # Let's say window size is 1 point. Delta f ~ N(0, 2*sigma^2)
    # The true sigma of the original process is true_sigma.
    # S1 relates to sigma via S1 = sqrt(2/pi) * std(Delta f) / sqrt(2)
    # Actually std_corrected returns sigma_est * sqrt(2/pi) where sigma_est estimates std(Delta f).

    recovered_sigmas = []

    # Since we can't easily isolate just the block, let's use the public function
    # calculate_haar_fluctuations on a dataset tailored to give exactly count=2 per trial.
    # A timeseries of 6 points gives 2 complete fluctuations of size 1 (lag=2).
    # t = [0, 1, 2, 3, 4, 5]
    # step_size defaults to window_size (overlap=False), so we get:
    # win1=[0], win2=[1] -> fluc1
    # win3=[2], win4=[3] -> fluc2
    # So count=2 fluctuations.

    s1_results = []
    for _ in range(n_trials):
        time = np.arange(6)
        data = rng.normal(0, true_sigma, size=6)
        lags, s1_std, counts, _ = calculate_haar_fluctuations(
            time, data, lag_times=np.array([2.0]), min_samples_per_window=1, statistic="mean", aggregation="std_corrected", overlap=False
        )
        assert counts[0] == 2
        s1_results.append(s1_std[0])

    mean_s1 = np.mean(s1_results)

    # theoretical expected value for S1 with std_corrected is sigma_est * sqrt(2/pi).
    # Since the underlying fluctuation variance is 2*sigma^2 for lag=2 (window_size=1),
    # the true population std of the fluctuations is sqrt(2)*sigma.
    # The std_corrected function computes an unbiased estimate of this std, and multiplies by sqrt(2/pi).
    # So expected S1 = sqrt(2/pi) * sqrt(2) * true_sigma = (2/sqrt(pi)) * true_sigma.

    expected_s1 = (2.0 / np.sqrt(np.pi)) * true_sigma

    # Tolerance of 3%
    assert np.isclose(mean_s1, expected_s1, rtol=0.03), f"Expected {expected_s1}, got {mean_s1}"

def test_std_corrected_multiple_counts():
    """
    Rigorously tests that std_corrected accurately recovers the true population sigma
    for a known-mean (ddof=0) case at multiple small sample sizes (N=2, 5, 10, 30).
    """
    rng = np.random.default_rng(42)
    true_sigma = 1.0
    n_trials = 5000

    counts_to_test = [2, 5, 10, 30]

    for test_count in counts_to_test:
        s1_results = []
        L = test_count * 2 + 2
        for _ in range(n_trials):
            time = np.arange(L)
            data = rng.normal(0, true_sigma, size=L)
            lags, s1_std, counts, _ = calculate_haar_fluctuations(
                time, data, lag_times=np.array([2.0]), min_samples_per_window=1, statistic="mean", aggregation="std_corrected", overlap=False
            )
            assert counts[0] == test_count
            s1_results.append(s1_std[0])

        mean_s1 = np.mean(s1_results)
        expected_s1 = (2.0 / np.sqrt(np.pi)) * true_sigma

        assert np.isclose(mean_s1, expected_s1, rtol=0.03), f"Failed for count={test_count}: Expected {expected_s1}, got {mean_s1}"

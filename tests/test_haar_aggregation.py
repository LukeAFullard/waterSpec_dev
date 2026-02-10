
import pytest
import numpy as np
from waterSpec.haar_analysis import calculate_haar_fluctuations, _small_sample_std

def test_small_sample_std_correction():
    # Test correction for N=2
    # Standard deviation of [1, -1] is sqrt(2). Sample std (ddof=1) is sqrt(2).
    # Correction factor for N=2 is sqrt(pi/2) approx 1.253
    # Result should be sqrt(2) * sqrt(pi/2) = sqrt(pi) approx 1.772

    data = np.array([1.0, -1.0])
    s = np.std(data, ddof=1)
    corrected_s = _small_sample_std(data)

    expected = s * np.sqrt(np.pi / 2) # approx 1.77245
    assert np.isclose(corrected_s, expected)

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

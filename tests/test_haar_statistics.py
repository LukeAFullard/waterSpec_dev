import numpy as np
import pytest
from waterSpec.haar_analysis import calculate_haar_fluctuations, HaarAnalysis

def test_haar_mean_default():
    """Test that default statistic is mean and matches manual calculation."""
    time = np.arange(11) # 0 to 10
    data = np.array([1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2]) # 11 points
    # Window size 10. t_start=0, t_end=10.
    # Midpoint t=5.
    # Half 1: 0 <= t < 5. Indices 0,1,2,3,4. Values: [1,1,1,1,1]. Mean=1.
    # Half 2: 5 <= t < 10. Indices 5,6,7,8,9. Values: [2,2,2,2,2]. Mean=2.
    # Diff = 1.0

    lags, s1, counts, neff = calculate_haar_fluctuations(
        time, data, lag_times=np.array([10.0]), min_samples_per_window=5, overlap=False
    )

    assert len(s1) == 1
    assert s1[0] == 1.0

def test_haar_median():
    """Test using median statistic."""
    time = np.arange(11)
    # Half 1 (0-4): [1, 1, 100, 1, 1] -> median=1
    # Half 2 (5-9): [2, 2, 200, 2, 2] -> median=2
    data = np.array([1, 1, 100, 1, 1, 2, 2, 200, 2, 2, 999])

    lags, s1, counts, neff = calculate_haar_fluctuations(
        time, data, lag_times=np.array([10.0]), min_samples_per_window=5, statistic="median", overlap=False
    )

    assert len(s1) == 1
    assert s1[0] == 1.0

    # Check mean for comparison (should be different)
    lags_mean, s1_mean, _, _ = calculate_haar_fluctuations(
        time, data, lag_times=np.array([10.0]), min_samples_per_window=5, statistic="mean", overlap=False
    )
    # Mean of half 1: (104)/5 = 20.8
    # Mean of half 2: (208)/5 = 41.6
    # Diff = 20.8
    assert abs(s1_mean[0] - 20.8) < 0.001

def test_haar_percentile():
    """Test using percentile statistic."""
    time = np.arange(11)
    data = np.arange(11)
    # Half 1: 0,1,2,3,4.
    # Half 2: 5,6,7,8,9.

    # 50th percentile (median) of 0..4 is 2.
    # 50th percentile of 5..9 is 7.
    # Diff = 5.

    lags, s1, counts, neff = calculate_haar_fluctuations(
        time, data, lag_times=np.array([10.0]), min_samples_per_window=5,
        statistic="percentile", percentile=50, overlap=False
    )
    assert len(s1) == 1
    assert np.isclose(s1[0], 5.0)

def test_haar_percentile_hazen():
    """Test specific percentile method (Hazen)."""
    # Use lag 6. Time 0..6 (7 points).
    time = np.arange(7)
    data = np.array([1, 2, 3, 11, 12, 13, 99])
    # Lag 6. t_start=0, t_end=6. Mid=3.
    # Half 1: 0,1,2 -> [1, 2, 3]
    # Half 2: 3,4,5 -> [11, 12, 13]

    # 50th percentile (median) is 2 and 12. Diff 10.
    lags, s1, _, _ = calculate_haar_fluctuations(
        time, data, lag_times=np.array([6.0]), min_samples_per_window=3,
        statistic="percentile", percentile=50, percentile_method="hazen", overlap=False
    )
    assert len(s1) == 1
    assert np.isclose(s1[0], 10.0)

def test_haar_percentile_missing_arg():
    """Test error raised if percentile not provided."""
    time = np.arange(10)
    data = np.arange(10)

    with pytest.raises(ValueError, match="percentile must be provided"):
        calculate_haar_fluctuations(time, data, statistic="percentile")

def test_haar_analysis_integration():
    """Test integration into HaarAnalysis class."""
    time = np.arange(21)
    data = np.random.randn(21)

    haar = HaarAnalysis(time, data)
    res = haar.run(
        statistic="percentile", percentile=95, percentile_method="linear",
        num_lags=5
    )

    assert "beta" in res
    assert res["statistic"] == "percentile"
    # Ensure it didn't crash and computed something
    assert len(res["s1"]) > 0

def test_invalid_statistic():
    time = np.arange(10)
    data = np.arange(10)
    with pytest.raises(ValueError, match="Unknown statistic"):
        calculate_haar_fluctuations(time, data, statistic="invalid")

import numpy as np
import pytest
from waterSpec.haar_analysis import calculate_haar_fluctuations, calculate_sliding_haar, HaarAnalysis
from waterSpec.segmentation import SegmentedRegimeAnalysis

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

def test_sliding_haar_percentiles():
    """Test sliding Haar calculation with percentiles."""
    time = np.arange(11)
    # Window size 10.
    # t=0..10. Mid=5. Left=0..4, Right=5..9.
    # Sliding step 1. Next window t=1..11? (Data only goes to 10).
    # So basically one window.

    # Left: 0,1,2,3,4. 90th percentile (Hazen) approx 3.6
    # Right: 5,6,7,8,9. 90th percentile (Hazen) approx 8.6
    # Diff approx 5.

    data = np.arange(11)

    t_centers, fluctuations = calculate_sliding_haar(
        time, data, window_size=10.0, step_size=10.0, min_samples_per_window=5,
        statistic="percentile", percentile=50
    )

    assert len(fluctuations) == 1
    assert np.isclose(fluctuations[0], 5.0) # Median diff is exactly 5

def test_segmentation_integration():
    """Test that segmentation accepts percentile arguments."""
    # Simple case: burst of variance
    time = np.arange(20)
    data = np.zeros(20)
    # Introduce an "extreme" event in middle
    # t=10. Window size 10 -> [5, 15]
    # Left [5, 10], Right [10, 15]
    data[10] = 100 # Huge outlier

    # If using mean, this window will have large fluctuation.
    # If using median, it won't (median of 5 zeros is 0).

    seg = SegmentedRegimeAnalysis()

    # Using Median: fluctuation should be 0 (mostly, unless noise makes it jump)
    # The point is to test that the ARGUMENT is passed through.

    # With a massive spike, median fluctuation is still small (just background noise).
    res_median = seg.segment_by_fluctuation(
        time, data, scale=10.0, statistic="median"
    )
    # Should find few or no events (only noise), or at least much fewer/smaller.

    # Using Max (100th percentile): fluctuation should be huge.
    # However, segmentation relies on (fluctuation > factor * median_fluctuation).
    # If using 'max', the *median* fluctuation will also be driven by noise max range.
    # But the local fluctuation around the spike will be 100.

    # Let's verify that the 'statistic' argument is actually used by checking
    # the returned 'fluctuations' array in the result dict.

    # For the window around index 10:
    # Scale 10. Sliding step 2 (scale/5).
    # One window will be approx [5, 15]. Left [5,10], Right [10,15].
    # Left max ~ 0 (noise). Right max = 100. Diff ~ 100.

    res_max = seg.segment_by_fluctuation(
        time, data, scale=10.0, statistic="percentile", percentile=100
    )

    max_fluc = np.max(np.abs(res_max['fluctuations']))
    assert max_fluc > 90 # Should see the spike

    # Now check median statistic
    res_median = seg.segment_by_fluctuation(
        time, data, scale=10.0, statistic="median"
    )
    max_fluc_median = np.max(np.abs(res_median['fluctuations']))
    assert max_fluc_median < 10 # Should ignore the spike (median of [noise... 100 ... noise] is noise)


import numpy as np
import pytest
from waterSpec.segmentation import SegmentedRegimeAnalysis

def test_segmentation_bursty_data():
    """
    Test that segmentation correctly identifies a high-volatility burst.
    """
    np.random.seed(42)
    time = np.arange(0, 200, 1.0)

    # Background: low noise
    data = np.random.normal(0, 0.1, 200)

    # Event: high noise (burst) in the middle
    # t=80 to t=120
    data[80:120] += np.random.normal(0, 2.0, 40) # Add large fluctuations

    # Scale to detect: window size 10
    scale = 10.0

    results = SegmentedRegimeAnalysis.segment_by_fluctuation(
        time, data, scale, threshold_factor=3.0
    )

    events = results['events']
    background = results['background']

    # We expect at least one event covering roughly 80-120
    # Note: sliding window smearing might make it start earlier/end later by scale/2
    assert len(events) >= 1

    # Check bounds of the main event
    # Should contain the core burst
    covered = False
    for start, end in events:
        if start <= 90 and end >= 110:
            covered = True
            break
    assert covered, f"Did not detect the main event 80-120. Found: {events}"

    # Background should handle the rest
    assert len(background) >= 1

def test_extract_segments():
    time = np.arange(100)
    data = np.arange(100) # data = time

    segments = [(10, 20), (50, 60)]

    extracted = SegmentedRegimeAnalysis.extract_segments(time, data, segments)

    assert len(extracted) == 2

    # Check first segment
    t1, d1 = extracted[0]
    assert t1[0] == 10
    assert t1[-1] == 20
    assert np.array_equal(t1, d1)

    # Check second segment
    t2, d2 = extracted[1]
    assert t2[0] == 50
    assert t2[-1] == 60

def test_no_events():
    """Test with uniform low noise."""
    np.random.seed(42)
    time = np.arange(100)
    data = np.random.normal(0, 1.0, 100)

    # Threshold factor very high
    results = SegmentedRegimeAnalysis.segment_by_fluctuation(
        time, data, scale=5.0, threshold_factor=100.0
    )

    assert len(results['events']) == 0
    assert len(results['background']) == 1
    assert results['background'][0] == (0, 99)

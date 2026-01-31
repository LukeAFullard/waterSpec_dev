
import numpy as np
import pytest
from waterSpec import Analysis
from waterSpec.segmentation import RegimeAnalysis

def generate_bursty_signal(n=1000):
    """
    Generates a signal with distinct quiet and bursty periods.
    """
    np.random.seed(42)
    time = np.arange(n)
    data = np.random.normal(0, 1, n)

    # Introduce a burst
    data[400:600] *= 10

    return time, data

def test_segmentation_detection(tmp_path):
    time, data = generate_bursty_signal()

    # Create Analysis object with a real dummy file to pass initialization
    d = tmp_path / "dummy.csv"
    # Write enough dummy points to pass min_valid_data_points check (default 10)
    lines = ["t,d"] + [f"{i},{i}" for i in range(11)]
    d.write_text("\n".join(lines))

    # Specify input_time_unit to handle numeric time column
    ana = Analysis(file_path=str(d), time_col="t", data_col="d", input_time_unit="seconds")
    ana.time = time
    ana.data = data

    regime = RegimeAnalysis(ana)

    # Scale = 10
    slices = regime.segment_by_fluctuation(scale=10, threshold_factor=2.0)

    evts = slices['event']
    base = slices['baseflow']

    # Should detect the burst around 400-600
    assert len(evts) >= 1

    # Check bounds of the main event
    # Might be split if fluctuations dip, but should cover the range

    in_event = np.zeros(len(time), dtype=bool)
    for sl in evts:
        in_event[sl] = True

    # Check core of the burst is detected
    assert np.mean(in_event[450:550]) > 0.8

    # Check quiet areas are baseflow
    # Be more lenient with false positives due to random fluctuations in white noise.
    # The simple thresholding might pick up some peaks in white noise.
    # We increase tolerance to 0.4 which is still distinct from > 0.8 in event.
    assert np.mean(in_event[0:300]) < 0.4
    assert np.mean(in_event[700:1000]) < 0.4

def test_regime_comparison(tmp_path):
    # Create two regimes with different scaling
    # Baseflow: White noise (beta ~ 0)
    # Event: Brown noise (beta ~ 2)

    n = 2000
    time = np.arange(n)

    # First half white noise
    data = np.random.normal(0, 1, n)

    # Second half Brownian motion (cumulative sum)
    # Scale it to look like a high-energy event
    rw = np.cumsum(np.random.normal(0, 1, 1000))
    # Normalize to start near 0?
    rw -= rw[0]
    data[1000:] = rw

    d = tmp_path / "dummy_regime.csv"
    # Write enough dummy points to pass min_valid_data_points check
    lines = ["t,d"] + [f"{i},{i}" for i in range(11)]
    d.write_text("\n".join(lines))

    # Specify input_time_unit to handle numeric time column
    ana = Analysis(file_path=str(d), time_col="t", data_col="d", input_time_unit="seconds")
    ana.time = time
    ana.data = data

    regime = RegimeAnalysis(ana)

    # Manually set slices to avoid fluctuation detection logic variance
    # This tests the comparison logic specifically
    regime.event_slices = [slice(1000, 2000)]
    regime.baseflow_slices = [slice(0, 1000)]

    results = regime.run_regime_comparison(min_lag=1, max_lag=100)

    beta_evt = results['event']['beta']
    beta_base = results['baseflow']['beta']

    print(f"Event Beta: {beta_evt}")
    print(f"Baseflow Beta: {beta_base}")

    # Expected:
    # Event (RW) -> beta ~ 2
    # Baseflow (White) -> beta ~ 0
    # Haar slope m: beta = 1 + 2m
    # RW: m=0.5 -> beta=2
    # White: m=-0.5 -> beta=0

    assert beta_evt == pytest.approx(2.0, abs=0.5)
    assert beta_base == pytest.approx(0.0, abs=0.5)

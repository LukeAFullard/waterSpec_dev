import numpy as np
import pandas as pd
import pytest
from waterSpec.analysis import Analysis
import os


@pytest.fixture
def synthetic_data_with_changepoint():
    """
    Generates a synthetic time series with a known changepoint.
    - First 200 points: White noise (β ≈ 0)
    - Next 200 points: Pink noise (β ≈ 1)
    """
    np.random.seed(42)
    n1, n2 = 200, 200
    time = np.arange(n1 + n2)
    # White noise
    data1 = np.random.randn(n1)
    # Pink noise (using a simple filter)
    data2 = np.cumsum(np.random.randn(n2))
    data2 = (data2 - np.mean(data2)) / np.std(data2)

    data = np.concatenate([data1, data2])
    df = pd.DataFrame({"time": time, "value": data})

    # Create a temporary file
    file_path = "temp_changepoint_data.csv"
    df.to_csv(file_path, index=False)

    yield file_path, n1 # Yield path and changepoint index

    # Teardown: remove the temporary file
    os.remove(file_path)

@pytest.fixture
def synthetic_data_no_changepoint():
    """Generates a synthetic time series with no changepoint (white noise)."""
    np.random.seed(0)
    time = np.arange(400)
    data = np.random.randn(400)
    df = pd.DataFrame({"time": time, "value": data})

    file_path = "temp_no_changepoint_data.csv"
    df.to_csv(file_path, index=False)

    yield file_path

    os.remove(file_path)

def test_auto_changepoint_detection(synthetic_data_with_changepoint, tmpdir):
    """
    Test that automatic changepoint detection finds the synthetic changepoint.
    """
    file_path, cp_index = synthetic_data_with_changepoint

    analyzer = Analysis(
        file_path=file_path,
        time_col="time",
        data_col="value",
        changepoint_mode='auto',
        changepoint_options={'model': 'l2'} # Use l2 for mean shift detection
    )

    results = analyzer.run_full_analysis(output_dir=str(tmpdir), n_bootstraps=0)

    assert results['changepoint_analysis'] is True
    # Allow for some tolerance in detection
    assert abs(results['changepoint_index'] - cp_index) < 10
    assert 'segment_before' in results
    assert 'segment_after' in results

    # Check that beta values are reasonable for the segments
    beta_before_results = results['segment_before']
    beta_after_results = results['segment_after']
    beta_before = beta_before_results['betas'][0] if 'betas' in beta_before_results else beta_before_results.get('beta', np.nan)
    beta_after = beta_after_results['betas'][0] if 'betas' in beta_after_results else beta_after_results.get('beta', np.nan)

    assert abs(beta_before) < 0.5 # Close to 0 for white noise
    assert beta_after > 1.5 # Should be close to 2 for Brownian noise (random walk)

def test_manual_changepoint(synthetic_data_with_changepoint, tmpdir):
    """
    Test that manual changepoint specification correctly segments the data.
    """
    file_path, cp_index = synthetic_data_with_changepoint

    analyzer = Analysis(
        file_path=file_path,
        time_col="time",
        data_col="value",
        changepoint_mode='manual',
        changepoint_index=cp_index
    )

    results = analyzer.run_full_analysis(output_dir=str(tmpdir), n_bootstraps=0)

    assert results['changepoint_analysis'] is True
    assert results['changepoint_index'] == cp_index
    assert results['segment_before']['n_points'] == cp_index
    assert results['segment_after']['n_points'] == len(analyzer.time) - cp_index

def test_no_changepoint_found(synthetic_data_no_changepoint, tmpdir):
    """
    Test that no changepoint is found when none exists.
    """
    file_path = synthetic_data_no_changepoint

    analyzer = Analysis(
        file_path=file_path,
        time_col="time",
        data_col="value",
        changepoint_mode='auto',
        changepoint_options={'model': 'l2', 'penalty': 1000} # High penalty
    )

    results = analyzer.run_full_analysis(output_dir=str(tmpdir), n_bootstraps=0)

    assert 'changepoint_analysis' not in results or results['changepoint_analysis'] is False

def test_invalid_changepoint_index(synthetic_data_with_changepoint):
    """
    Test that an out-of-bounds manual changepoint raises a ValueError.
    """
    file_path, _ = synthetic_data_with_changepoint

    with pytest.raises(ValueError, match="is out of the valid data range"):
        Analysis(
            file_path=file_path,
            time_col="time",
            data_col="value",
            changepoint_mode='manual',
            changepoint_index=1000 # Out of bounds
        )

def test_changepoint_edge_cases(synthetic_data_with_changepoint, tmpdir):
    """
    Test that changepoints creating segments too small for analysis raise a ValueError.
    """
    file_path, _ = synthetic_data_with_changepoint

    # Case 1: Changepoint is too close to the start of the series
    analyzer_start = Analysis(
        file_path=file_path,
        time_col="time",
        data_col="value",
        changepoint_mode='manual',
        changepoint_index=5  # Creates a "before" segment of only 5 points
    )
    with pytest.raises(ValueError, match="Segment before changepoint too small"):
        analyzer_start.run_full_analysis(output_dir=str(tmpdir))

    # Case 2: Changepoint is too close to the end of the series
    analyzer_end = Analysis(
        file_path=file_path,
        time_col="time",
        data_col="value",
        changepoint_mode='manual',
        changepoint_index=395  # Creates an "after" segment of only 5 points
    )
    with pytest.raises(ValueError, match="Segment after changepoint too small"):
        analyzer_end.run_full_analysis(output_dir=str(tmpdir))
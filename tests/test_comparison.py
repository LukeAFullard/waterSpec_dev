import os
from unittest.mock import ANY, patch

import numpy as np
import pandas as pd
import pytest

from src.waterSpec.comparison import SiteComparison

# Helper function to create test data files (adapted from test_analysis.py)
def create_test_data_file(
    tmp_path,
    filename,
    time,
    series,
    errors=None,
    time_col="time",
    data_col="value",
    error_col="error",
):
    file_path = tmp_path / filename
    if isinstance(time, pd.DatetimeIndex):
        time_to_write = time.strftime("%Y-%m-%d %H:%M:%S")
    else:
        time_to_write = time
    data_dict = {time_col: time_to_write, data_col: series}
    if errors is not None:
        data_dict[error_col] = errors
    df = pd.DataFrame(data_dict)
    df.to_csv(file_path, index=False)
    return str(file_path)

# --- Tests for the SiteComparison Class ---

def test_site_comparison_initialization(tmp_path):
    """Test that the SiteComparison class initializes and loads data for two sites."""
    # Create two different data files
    time1 = pd.date_range("2023-01-01", periods=100, freq="D")
    series1 = np.random.rand(100)
    file1_path = create_test_data_file(tmp_path, "data1.csv", time1, series1, time_col="timestamp", data_col="value1")

    time2 = pd.date_range("2023-01-01", periods=120, freq="D")
    series2 = np.random.rand(120)
    file2_path = create_test_data_file(tmp_path, "data2.csv", time2, series2, time_col="timestamp", data_col="value2")

    site1_config = {"name": "SiteA", "file_path": file1_path, "time_col": "timestamp", "data_col": "value1"}
    site2_config = {"name": "SiteB", "file_path": file2_path, "time_col": "timestamp", "data_col": "value2"}

    comparison = SiteComparison(site1_config, site2_config)

    assert comparison is not None
    assert comparison.site1_name == "SiteA"
    assert comparison.site2_name == "SiteB"
    assert len(comparison.site1_data["time"]) == 100
    assert len(comparison.site2_data["time"]) == 120

def test_site_comparison_run_comparison_creates_outputs(tmp_path):
    """Test that run_comparison creates the plot and summary files."""
    # Use existing sample data for a more realistic test
    site1_config = {
        "name": "Forested",
        "file_path": "examples/sample_data.csv",
        "time_col": "timestamp",
        "data_col": "concentration",
    }
    site2_config = {
        "name": "Periodic",
        "file_path": "examples/periodic_data.csv",
        "time_col": "timestamp",
        "data_col": "value",
    }
    output_dir = tmp_path / "comparison_results"

    comparison = SiteComparison(site1_config, site2_config)
    results = comparison.run_comparison(output_dir=str(output_dir), n_bootstraps=10)

    # Check that output files were created
    expected_plot = output_dir / "Forested_vs_Periodic_comparison_separate.png"
    expected_summary = output_dir / "Forested_vs_Periodic_comparison_summary.txt"
    assert expected_plot.exists()
    assert expected_summary.exists()

    # Check that the summary contains expected text
    summary_text = expected_summary.read_text()
    assert "Site Comparison: Forested vs. Periodic" in summary_text
    assert "SITE COMPARISON:" in summary_text
    assert "Standard Analysis for: concentration" in summary_text
    assert "Segmented Analysis for: value" in summary_text

    # Check that the results dictionary is populated correctly
    assert "summary_text" in results
    assert "site1" in results and "site2" in results
    assert results["site1"]["site_name"] == "Forested"
    assert results["site2"]["site_name"] == "Periodic"

def test_site_comparison_insufficient_data_raises_error(tmp_path):
    """Test that SiteComparison raises a ValueError if one site has insufficient data."""
    # Site 1 has enough data
    time1 = pd.date_range("2023-01-01", periods=100, freq="D")
    series1 = np.random.rand(100)
    file1_path = create_test_data_file(tmp_path, "data1.csv", time1, series1)

    # Site 2 has only 5 data points
    time2 = pd.date_range("2023-01-01", periods=5, freq="D")
    series2 = np.random.rand(5)
    file2_path = create_test_data_file(tmp_path, "data2.csv", time2, series2)

    site1_config = {"name": "GoodSite", "file_path": file1_path, "time_col": "time", "data_col": "value"}
    site2_config = {"name": "BadSite", "file_path": file2_path, "time_col": "time", "data_col": "value"}

    with pytest.raises(ValueError, match="Not enough valid data points \\(5\\) for site 'BadSite'"):
        SiteComparison(site1_config, site2_config)

@patch("src.waterSpec.comparison.SiteComparison._run_site_analysis")
def test_site_comparison_summary_logic(mock_run_analysis, tmp_path):
    """Test the summary logic based on mocked analysis results for two sites."""
    # Define a minimal set of keys needed for plotting to avoid KeyErrors
    plotting_keys = {
        "frequency": np.array([0.1, 0.2, 0.3]),
        "power": np.array([10, 5, 2]),
        "log_freq": np.log10(np.array([0.1, 0.2, 0.3])),
        "intercept": 1.0,
        "significant_peaks": [],
    }

    # Mock the results for two sites to control the comparison
    mock_run_analysis.side_effect = [
        {  # Site 1: High persistence
            "beta": 1.8,
            "site_name": "Site1",
            "n_points": 100,
            "summary_text": "Standard Analysis for: Site1\nValue: β = 1.80",
            "chosen_model_type": "standard",
            **plotting_keys,
        },
        {  # Site 2: Low persistence
            "beta": 0.2,
            "site_name": "Site2",
            "n_points": 100,
            "summary_text": "Standard Analysis for: Site2\nValue: β = 0.20",
            "chosen_model_type": "standard",
            **plotting_keys,
        },
    ]

    # Create dummy files, as they are needed for initialization
    time = pd.date_range("2023-01-01", periods=20, freq="D")
    series = np.random.rand(20)
    file1_path = create_test_data_file(tmp_path, "dummy1.csv", time, series)
    file2_path = create_test_data_file(tmp_path, "dummy2.csv", time, series)

    site1_config = {"name": "Site1", "file_path": file1_path, "time_col": "time", "data_col": "value"}
    site2_config = {"name": "Site2", "file_path": file2_path, "time_col": "time", "data_col": "value"}

    comparison = SiteComparison(site1_config, site2_config)
    results = comparison.run_comparison(output_dir=tmp_path)

    summary = results["summary_text"]
    assert "Site2 shows significantly LOWER persistence than Site1 (-1.60)" in summary
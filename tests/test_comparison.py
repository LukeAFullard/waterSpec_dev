import os
from unittest.mock import ANY, patch

import numpy as np
import pandas as pd
import pytest

from waterSpec.comparison import SiteComparison

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

@patch("waterSpec.comparison.SiteComparison._run_site_analysis")
def test_site_comparison_run_comparison_creates_outputs(mock_run_analysis, tmp_path):
    """Test that run_comparison creates the plot and summary files."""
    # Define mock return values for each site analysis
    # These need to contain all keys required for summary generation and plotting
    site1_results = {
        "site_name": "SiteA",
        "param_name": "value1",
        "chosen_model_type": "standard",
        "beta": 1.5,
        "intercept": 1.0,
        "summary_text": "Standard Analysis for: value1",
        "log_freq": np.log10(np.linspace(0.1, 1, 10)),
        "log_power": np.random.rand(10),
        "fitted_log_power": np.random.rand(10),
        "n_breakpoints": 0,
        "frequency": np.linspace(0.1, 1, 10),
        "power": np.random.rand(10),
        "chosen_model": "standard",
        "ci_method": "bootstrap",
        "n_points": 50,
    }
    site2_results = {
        "site_name": "SiteB",
        "param_name": "value2",
        "chosen_model_type": "segmented",
        "betas": [0.5, 1.8],
        "breakpoints": [0.5],
        "summary_text": "Segmented Analysis for: value2",
        "log_freq": np.log10(np.linspace(0.1, 1, 10)),
        "log_power": np.random.rand(10),
        "fitted_log_power": np.random.rand(10),
        "n_breakpoints": 1,
        "frequency": np.linspace(0.1, 1, 10),
        "power": np.random.rand(10),
        "chosen_model": "segmented_1bp",
        "ci_method": "bootstrap",
        "n_points": 60,
    }
    mock_run_analysis.side_effect = [site1_results, site2_results]

    # Create dummy data files (still needed for initialization)
    time1 = pd.to_datetime(np.arange(50) * 86400, unit="s")
    series1 = 10 + np.random.randn(50)
    file1_path = create_test_data_file(tmp_path, "data1.csv", time1, series1, time_col="timestamp", data_col="value1")

    time2 = pd.to_datetime(np.arange(60) * 86400, unit="s")
    series2 = 20 + np.random.randn(60)
    file2_path = create_test_data_file(tmp_path, "data2.csv", time2, series2, time_col="timestamp", data_col="value2")

    site1_config = {"name": "SiteA", "file_path": file1_path, "time_col": "timestamp", "data_col": "value1"}
    site2_config = {"name": "SiteB", "file_path": file2_path, "time_col": "timestamp", "data_col": "value2"}

    output_dir = tmp_path / "comparison_results"

    comparison = SiteComparison(site1_config, site2_config)
    results = comparison.run_comparison(output_dir=str(output_dir), n_bootstraps=10)

    # Check that output files were created
    expected_plot = output_dir / "SiteA_vs_SiteB_comparison_separate.png"
    expected_summary = output_dir / "SiteA_vs_SiteB_comparison_summary.txt"
    assert expected_plot.exists()
    assert expected_summary.exists()

    # Check that the summary contains expected text
    summary_text = expected_summary.read_text()
    assert "Site Comparison: SiteA vs. SiteB" in summary_text
    assert "SITE COMPARISON:" in summary_text
    assert "Standard Analysis for: value1" in summary_text
    assert "Segmented Analysis for: value2" in summary_text

    # Check that the results dictionary is populated correctly
    assert "summary_text" in results
    assert "site1" in results and "site2" in results
    assert results["site1"]["site_name"] == "SiteA"
    assert results["site2"]["site_name"] == "SiteB"

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

@patch("waterSpec.comparison.fit_segmented_spectrum")
@patch("waterSpec.comparison.fit_standard_model")
def test_site_comparison_generates_correct_summary_and_plot_data(
    mock_fit_standard, mock_fit_segmented, tmp_path
):
    """
    Test that the full comparison workflow correctly processes data and generates
    the necessary outputs by mocking the underlying model fitters.
    """
    # Mock the underlying fitters to return predictable results
    # Site 1 will get a standard model fit
    mock_fit_standard.return_value = {
        "beta": 1.8,
        "intercept": 1,
        "log_freq": np.log10(np.array([0.1, 0.2, 0.3])),
        "bic": 100,
        "n_breakpoints": 0,
    }
    # Site 2 will get a segmented model fit
    mock_fit_segmented.return_value = {
        "betas": [-0.2, 2.0],
        "breakpoints": [0.1],
        "fitted_log_power": np.array([1, 2, 3]),
        "log_freq": np.log10(np.array([0.1, 0.2, 0.3])),
        "bic": 90,
        "n_breakpoints": 1,
    }

    # Create dummy data files
    time = pd.date_range("2023-01-01", periods=100, freq="D")
    series = np.random.rand(100)
    file1_path = create_test_data_file(tmp_path, "data1.csv", time, series)
    file2_path = create_test_data_file(tmp_path, "data2.csv", time, series)

    site1_config = {"name": "HighPersistenceSite", "file_path": file1_path, "time_col": "time", "data_col": "value"}
    site2_config = {"name": "LowPersistenceSite", "file_path": file2_path, "time_col": "time", "data_col": "value"}

    comparison = SiteComparison(site1_config, site2_config)
    # For Site 1, force only standard model. For Site 2, allow segmented.
    # This ensures our mocks are called as expected.
    with patch.object(comparison, '_run_site_analysis', wraps=comparison._run_site_analysis) as spy:
        results = comparison.run_comparison(
            output_dir=tmp_path, n_bootstraps=0, max_breakpoints=0
        )
        # Manually trigger second analysis for second site with different settings
        analysis_kwargs = {"max_breakpoints": 1, "n_bootstraps": 0, "seed": 43, "fit_method": "theil-sen", "ci_method": "bootstrap", "bootstrap_type": "block", "samples_per_peak": 5, "nyquist_factor": 1.0, "max_freq": None, "peak_detection_method": "fap", "fap_threshold": 0.01, "fap_method": "baluev", "peak_fdr_level": 0.05, "normalization": "standard", "p_threshold": 0.05}
        comparison.results["site2"] = comparison._run_site_analysis(comparison.site2_data, "LowPersistenceSite", analysis_kwargs)
        comparison.results["summary_text"] = comparison._generate_comparison_summary(comparison.results)


    # Check that the summary logic is correct
    summary = comparison.results["summary_text"]
    assert "LowPersistenceSite shows significantly LOWER persistence than HighPersistenceSite (-2.00)" in summary

    # Check that the results dictionary for site 1 has plotting keys
    site1_res = comparison.results["site1"]
    assert "log_freq" in site1_res
    assert "intercept" in site1_res
    assert site1_res["chosen_model_type"] == "standard"

    # Check that the results dictionary for site 2 has plotting keys
    site2_res = comparison.results["site2"]
    assert "fitted_log_power" in site2_res
    assert site2_res["chosen_model_type"] == "segmented"
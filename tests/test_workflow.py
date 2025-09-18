import pytest
from waterSpec.workflow import run_analysis
import os
import numpy as np

def test_run_analysis_workflow():
    """
    Test the full analysis workflow with the `run_analysis` function.
    """
    file_path = 'examples/sample_data.csv'

    results = run_analysis(file_path, time_col='timestamp', data_col='concentration', n_bootstraps=10)

    assert results
    assert isinstance(results, dict)

    expected_keys = ['beta', 'r_squared', 'intercept', 'stderr', 'beta_ci_lower', 'beta_ci_upper', 'summary_text']
    for key in expected_keys:
        assert key in results, f"Expected key '{key}' not found in results."

    # Assert that beta is a float, but don't be too strict about its value for this synthetic data.
    assert isinstance(results['beta'], float)
    assert np.isfinite(results['beta']) # Ensure it's not NaN or inf

    assert isinstance(results['r_squared'], float)
    assert isinstance(results['summary_text'], str)
    assert len(results['summary_text']) > 0

def test_workflow_with_plotting(tmp_path):
    """Test that the workflow can generate a plot."""
    file_path = 'examples/sample_data.csv'
    output_plot_path = tmp_path / "spectrum_plot.png"

    results = run_analysis(
        file_path,
        time_col='timestamp',
        data_col='concentration',
        do_plot=True,
        output_path=str(output_plot_path),
        n_bootstraps=10
    )

    assert os.path.exists(output_plot_path)
    assert 'beta' in results

def test_workflow_segmented(tmp_path):
    """Test the segmented analysis workflow."""
    file_path = 'examples/sample_data.csv'

    results = run_analysis(
        file_path,
        time_col='timestamp',
        data_col='concentration',
        analysis_type='segmented'
    )

    assert 'beta1' in results
    assert 'beta2' in results
    assert 'breakpoint' in results

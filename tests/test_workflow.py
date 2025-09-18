import pytest
from waterSpec.workflow import run_analysis

def test_run_analysis_workflow():
    """
    Test the full analysis workflow with the `run_analysis` function.
    """
    file_path = 'examples/sample_data.csv'

    # Run the full analysis
    results = run_analysis(file_path, time_col='timestamp', data_col='concentration', n_bootstraps=100)

    # Check that the results dictionary is not empty and is a dictionary
    assert results
    assert isinstance(results, dict)

    # Check for the presence of all expected keys
    expected_keys = ['beta', 'r_squared', 'intercept', 'stderr', 'beta_ci_lower', 'beta_ci_upper', 'interpretation']
    for key in expected_keys:
        assert key in results

    # Check the types of the returned values
    assert isinstance(results['beta'], float)
    assert isinstance(results['r_squared'], float)
    assert isinstance(results['interpretation'], str)

    # Check that the interpretation string is not empty
    assert len(results['interpretation']) > 0

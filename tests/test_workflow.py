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

def test_run_analysis_with_censored_data():
    """
    Test the full analysis workflow with censored data.
    """
    file_path = 'examples/censored_data.csv'

    # Run the analysis with the 'multiplier' strategy
    results = run_analysis(
        file_path,
        time_col='timestamp',
        data_col='concentration',
        n_bootstraps=100,
        censor_strategy='multiplier'
    )

    # Check that the results dictionary is not empty and is a dictionary
    assert results
    assert isinstance(results, dict)

    # Check that the beta value is calculated and is a float
    assert 'beta' in results
    assert isinstance(results['beta'], float)

import numpy as np

def test_run_analysis_segmented_no_breakpoint():
    """
    Test that segmented analysis on linear data fails gracefully.
    """
    # Using standard data that doesn't have a clear breakpoint
    file_path = 'examples/sample_data.csv'

    results = run_analysis(
        file_path,
        time_col='timestamp',
        data_col='concentration',
        analysis_type='segmented'
    )

    # The model should not converge on this data, and return NaNs
    assert results
    assert isinstance(results, dict)
    assert np.isnan(results['breakpoint'])
    assert np.isnan(results['beta1'])
    assert np.isnan(results['beta2'])

def test_run_analysis_loess_detrend():
    """
    Test the workflow with LOESS detrending.
    """
    file_path = 'examples/sample_data.csv'

    # Run the analysis with LOESS detrending
    results = run_analysis(
        file_path,
        time_col='timestamp',
        data_col='concentration',
        n_bootstraps=100,
        detrend_method='loess'
    )

    # Check that the results dictionary is not empty and is a dictionary
    assert results
    assert isinstance(results, dict)

    # Check that the beta value is calculated and is a float
    assert 'beta' in results
    assert isinstance(results['beta'], float)

import numpy as np
import pytest
import pandas as pd
from waterSpec.data_loader import load_data

def test_load_data_from_csv():
    """
    Test that load_data function loads data correctly from a CSV file.
    """
    # Define the path to the sample data
    file_path = 'examples/sample_data.csv'

    # Call the function to load data
    time, concentration = load_data(file_path, time_col='timestamp', data_col='concentration')

    # Check the output types
    assert isinstance(time, np.ndarray), "Time data should be a numpy array"
    assert isinstance(concentration, pd.Series), "Concentration data should be a pandas Series"

    # Check if the arrays have the correct length
    assert len(time) == 4, "Should be 4 time points"
    assert len(concentration) == 4, "Should be 4 concentration values"

    # Check if the time array is numeric (e.g., float for seconds/days since epoch)
    assert np.issubdtype(time.dtype, np.number), "Time array should be numeric"

    # Check a value to make sure data is loaded correctly
    assert concentration[0] == 10.1, "First concentration value is incorrect"

def test_load_data_from_json():
    """
    Test that load_data function loads data correctly from a JSON file.
    """
    # Define the path to the sample data
    file_path = 'examples/sample_data.json'

    # Call the function to load data
    time, concentration = load_data(file_path, time_col='timestamp', data_col='concentration')

    # Check the output types
    assert isinstance(time, np.ndarray), "Time data should be a numpy array"
    assert isinstance(concentration, pd.Series), "Concentration data should be a pandas Series"

    # Check if the arrays have the correct length
    assert len(time) == 4, "Should be 4 time points"
    assert len(concentration) == 4, "Should be 4 concentration values"

    # Check if the time array is numeric
    assert np.issubdtype(time.dtype, np.number), "Time array should be numeric"

    # Check a value to make sure data is loaded correctly
    assert concentration[0] == 10.1, "First concentration value is incorrect"

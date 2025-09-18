import numpy as np
import pytest
from waterSpec.data_loader import load_data

def test_load_data():
    """
    Test that load_data function loads data correctly.
    """
    # Define the path to the sample data
    file_path = 'examples/sample_data.csv'

    # Call the function to load data
    time, concentration = load_data(file_path, time_col='timestamp', data_col='concentration')

    # Check if the outputs are numpy arrays
    assert isinstance(time, np.ndarray), "Time data should be a numpy array"
    assert isinstance(concentration, np.ndarray), "Concentration data should be a numpy array"

    # Check if the arrays have the correct length
    assert len(time) == 4, "Should be 4 time points"
    assert len(concentration) == 4, "Should be 4 concentration values"

    # Check if the time array is numeric (e.g., float for seconds/days since epoch)
    assert np.issubdtype(time.dtype, np.number), "Time array should be numeric"

    # Check a value to make sure data is loaded correctly
    assert concentration[0] == 10.1, "First concentration value is incorrect"

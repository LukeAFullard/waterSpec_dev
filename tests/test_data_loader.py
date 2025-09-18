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

def test_load_data_empty_file(tmp_path):
    """Test that loading an empty file raises a ValueError."""
    file_path = tmp_path / "empty.csv"
    file_path.write_text("timestamp,concentration\n") # Just the header

    with pytest.raises(ValueError):
        # This should fail because there are no data rows, leading to errors
        # when trying to access columns that don't really exist.
        # Pandas behavior can vary, but it should result in an error.
        load_data(file_path, time_col='timestamp', data_col='concentration')

def test_load_data_missing_column(tmp_path):
    """Test that a missing data column raises a ValueError."""
    file_path = tmp_path / "missing_col.csv"
    file_path.write_text("timestamp,value\n2021-01-01,10\n")

    with pytest.raises(ValueError, match="Data column 'concentration' not found in the file."):
        load_data(file_path, time_col='timestamp', data_col='concentration')

def test_load_data_with_nans(tmp_path):
    """Test that data with NaN values is loaded correctly."""
    file_path = tmp_path / "data_with_nans.csv"
    file_path.write_text("timestamp,concentration\n2021-01-01,10\n2021-01-02,\n2021-01-03,12\n")

    time, data = load_data(file_path, time_col='timestamp', data_col='concentration')

    assert len(time) == 3
    assert len(data) == 3
    assert pd.isna(data[1])
    assert data[0] == 10

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

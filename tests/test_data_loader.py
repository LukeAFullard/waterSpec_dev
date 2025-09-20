import numpy as np
import pandas as pd
import pytest
from waterSpec.data_loader import load_data
import os

@pytest.fixture
def create_test_csv(tmp_path):
    d = tmp_path / "sub"
    d.mkdir()
    p = d / "test.csv"
    p.write_text("timestamp,concentration\n2023-01-01,10.1\n2023-01-02,10.5\n2023-01-03,10.3\n2023-01-04,11.0")
    return str(p)

def test_load_data_from_csv(create_test_csv):
    """
    Test that load_data function loads data correctly from a CSV file.
    """
    time, concentration, errors = load_data(create_test_csv, time_col='timestamp', data_col='concentration')

    assert isinstance(time, np.ndarray)
    assert isinstance(concentration, pd.Series)
    assert errors is None
    assert len(time) == 4
    assert len(concentration) == 4

def test_load_data_with_nans(tmp_path):
    """Test loading data with NaN values and ensure a warning is raised."""
    file_path = tmp_path / "nan_data.csv"
    file_path.write_text("timestamp,concentration\n2023-01-01,10.1\n2023-01-02,\n2023-01-03,10.3")
    with pytest.warns(UserWarning, match="contains NaN or null values"):
        time, concentration, _ = load_data(file_path, time_col='timestamp', data_col='concentration')
    assert len(time) == 3
    assert concentration.isnull().sum() == 1

def test_load_data_empty_file(tmp_path):
    """Test that loading an empty file raises a ValueError."""
    file_path = tmp_path / "empty.csv"
    file_path.write_text("timestamp,concentration\n")
    with pytest.raises(ValueError, match="The provided file is empty"):
        load_data(file_path, time_col='timestamp', data_col='concentration')

def test_load_data_missing_column(create_test_csv):
    """Test that a missing column raises a ValueError."""
    with pytest.raises(ValueError, match="Data column 'bad_col' not found"):
        load_data(create_test_csv, time_col='timestamp', data_col='bad_col')

def test_load_non_monotonic_time(tmp_path):
    """Test that non-monotonic time raises a ValueError."""
    file_path = tmp_path / "non_monotonic.csv"
    file_path.write_text("timestamp,concentration\n2023-01-03,10.3\n2023-01-01,10.1\n2023-01-02,10.5")
    # The loader now sorts the data, so this should pass without error.
    # We will add a duplicate timestamp to test the strict monotonicity check.
    file_path.write_text("timestamp,concentration\n2023-01-01,10.3\n2023-01-01,10.1\n2023-01-02,10.5")
    with pytest.raises(ValueError, match="not strictly monotonic increasing"):
        load_data(file_path, time_col='timestamp', data_col='concentration')

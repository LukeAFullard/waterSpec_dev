
import numpy as np
import pandas as pd
import pytest

from waterSpec.data_loader import load_data


@pytest.fixture
def create_test_csv(tmp_path):
    d = tmp_path / "sub"
    d.mkdir()
    p = d / "test.csv"
    p.write_text(
        "timestamp,concentration\n2023-01-01,10.1\n2023-01-02,10.5\n2023-01-03,10.3\n2023-01-04,11.0"
    )
    return str(p)


@pytest.fixture
def create_test_excel(tmp_path):
    """Creates a test Excel file."""
    file_path = tmp_path / "test.xlsx"
    df = pd.DataFrame(
        {"time": pd.to_datetime(["2023-01-01", "2023-01-02"]), "value": [1.0, 2.0]}
    )
    df.to_excel(file_path, index=False, engine="openpyxl")
    return str(file_path)


@pytest.fixture
def create_test_json(tmp_path):
    """Creates a test JSON file (record-oriented)."""
    file_path = tmp_path / "test.json"
    df = pd.DataFrame(
        {"time": pd.to_datetime(["2023-01-01", "2023-01-02"]), "value": [1.0, 2.0]}
    )
    df.to_json(file_path, orient="records", date_format="iso")
    return str(file_path)


def test_load_data_from_json(create_test_json):
    """Test loading data from a .json file."""
    time, value, _ = load_data(create_test_json, time_col="time", data_col="value")
    assert len(time) == 2
    assert value.iloc[1] == 2.0


def test_load_data_from_excel(create_test_excel):
    """Test loading data from an .xlsx file."""
    time, value, _ = load_data(create_test_excel, time_col="time", data_col="value")
    assert len(time) == 2
    assert value.iloc[1] == 2.0


def test_load_data_from_csv(create_test_csv):
    """
    Test that load_data function loads data correctly from a CSV file.
    """
    time, concentration, errors = load_data(
        create_test_csv, time_col="timestamp", data_col="concentration"
    )

    assert isinstance(time, np.ndarray)
    assert isinstance(concentration, pd.Series)
    assert errors is None
    assert len(time) == 4
    assert len(concentration) == 4


def test_load_data_with_nans(tmp_path):
    """
    Test loading data with NaN values. It should warn and then drop the NaN row.
    """
    file_path = tmp_path / "nan_data.csv"
    file_path.write_text(
        "timestamp,concentration\n2023-01-01,10.1\n2023-01-02,\n2023-01-03,10.3"
    )
    with pytest.warns(UserWarning, match="1 rows were dropped"):
        time, concentration, _ = load_data(
            file_path, time_col="timestamp", data_col="concentration"
        )
    # The loader should drop the row with the NaN value
    assert len(time) == 2
    assert len(concentration) == 2
    assert concentration.isnull().sum() == 0


def test_load_data_empty_file(tmp_path):
    """Test that loading an empty file raises a ValueError."""
    file_path = tmp_path / "empty.csv"
    file_path.write_text("timestamp,concentration\n")
    with pytest.raises(ValueError, match="The provided DataFrame is empty"):
        load_data(file_path, time_col="timestamp", data_col="concentration")


def test_load_data_missing_column(create_test_csv):
    """Test that a missing column raises a ValueError."""
    with pytest.raises(ValueError, match="Data column 'bad_col' not found"):
        load_data(create_test_csv, time_col="timestamp", data_col="bad_col")


def test_load_data_unparseable_time(tmp_path):
    """Test that an unparseable time column raises a ValueError."""
    file_path = tmp_path / "bad_time.csv"
    file_path.write_text(
        "time,value\nJan 1, 2023,10.1\nNOT A DATE,10.5\n2023-01-03,10.3"
    )
    # Note: "Jan 1, 2023" might be split by comma into "Jan 1" and " 2023".
    # "Jan 1" is parseable by pandas (current year). "NOT A DATE" is not.
    # Depending on pandas version, we might get 1 or 2 failures.
    # We match "value(s)" to be flexible, or just update to 1 if we are sure.
    # In this environment, "Jan 1" is parsed, so we get 1 failure.
    with pytest.raises(
        ValueError, match=r"\d+ value\(s\) in the time column 'time' could not be parsed"
    ):
        load_data(file_path, time_col="time", data_col="value")


def test_load_data_raises_on_duplicate_timestamps(tmp_path):
    """
    Test that duplicate timestamps raise a specific ValueError.
    """
    file_path = tmp_path / "duplicate_time.csv"
    # Create a file with a duplicate timestamp. The loader should sort this and
    # then find the duplicate.
    file_path.write_text(
        "timestamp,concentration\n2023-01-01,10.3\n2023-01-02,10.5\n2023-01-01,10.1"
    )

    # The error message should specifically mention duplicate timestamps.
    expected_error_msg = (
        "Duplicate timestamp found in time column 'timestamp'. "
        "The data must have unique and strictly increasing time points. "
        "First duplicate found at index 1 with value: 2023-01-01 00:00:00"
    )
    with pytest.raises(ValueError, match=expected_error_msg):
        load_data(file_path, time_col="timestamp", data_col="concentration")


def test_load_data_raises_on_duplicate_numeric_timestamps(tmp_path):
    """
    Test that duplicate numeric timestamps raise a specific ValueError.
    """
    file_path = tmp_path / "duplicate_numeric_time.csv"
    # Create a file with a duplicate numeric timestamp.
    file_path.write_text("time_hours,value\n10,10.3\n12,10.5\n10,10.1")

    # The error message should specifically mention duplicate timestamps.
    expected_error_msg = (
        "Duplicate timestamp found in time column 'time_hours'. "
        "The data must have unique and strictly increasing time points. "
        "First duplicate found at index 1 with value: 10"
    )
    with pytest.raises(ValueError, match=expected_error_msg):
        load_data(
            file_path,
            time_col="time_hours",
            data_col="value",
            input_time_unit="hours",
        )


def test_load_data_from_excel_sheet_by_name():
    """Test loading data from a specific sheet of an .xlsx file by name."""
    file_path = "tests/data/multi_sheet_data.xlsx"
    # Load from the second sheet, which has different values
    time, value, _ = load_data(
        file_path, time_col="timestamp", data_col="value", sheet_name="Data_Sheet_2"
    )
    assert len(time) == 3
    # Check a value from the second sheet
    assert value.iloc[0] == 100


def test_load_data_from_excel_sheet_by_index():
    """Test loading data from a specific sheet of an .xlsx file by index."""
    file_path = "tests/data/multi_sheet_data.xlsx"
    # Load from the second sheet (index 1)
    time, value, _ = load_data(
        file_path, time_col="timestamp", data_col="value", sheet_name=1
    )
    assert len(time) == 3
    # Check a value from the second sheet
    assert value.iloc[0] == 100


def test_load_data_with_time_format(tmp_path):
    """Test loading data with a specific time format string."""
    file_path = tmp_path / "formatted_time.csv"
    file_path.write_text("day,value\n01/01/2023,1\n02/01/2023,2")
    time, _, _ = load_data(
        file_path, time_col="day", data_col="value", time_format="%d/%m/%Y"
    )
    assert len(time) == 2
    # The first timestamp should be 0.0 seconds (relative time)
    assert time[0] == 0.0
    # The second timestamp should be one day later (86400 seconds)
    assert np.isclose(time[1], 86400.0)


def test_load_data_with_incorrect_time_format(tmp_path):
    """Test that an incorrect time format string raises a ValueError."""
    file_path = tmp_path / "formatted_time.csv"
    file_path.write_text("day,value\n2023-01-01,1\n2023-01-02,2")
    with pytest.raises(ValueError, match="Please check that the format string"):
        load_data(
            file_path, time_col="day", data_col="value", time_format="%d-%m-%Y"
        )


def test_load_data_unsupported_format(tmp_path):
    """Test that an unsupported file format raises an IOError."""
    file_path = tmp_path / "test.txt"
    file_path.write_text("data")
    with pytest.raises(IOError, match="Failed to read the file"):
        load_data(file_path, time_col="time", data_col="value")


def test_load_data_missing_error_column(create_test_csv):
    """Test that a missing error column, when specified, issues a UserWarning."""
    with pytest.warns(UserWarning, match="Error column 'bad_error_col' not found"):
        load_data(
            create_test_csv,
            time_col="timestamp",
            data_col="concentration",
            error_col="bad_error_col",
        )


def test_load_data_bad_error_column(tmp_path):
    """Test that a non-numeric error column issues a UserWarning."""
    file_path = tmp_path / "bad_error.csv"
    file_path.write_text("time,value,error\n2023-01-01,10,1\n2023-01-02,11,foo")
    with pytest.warns(UserWarning, match="could not be converted to a numeric type"):
        load_data(file_path, time_col="time", data_col="value", error_col="error")


def test_load_data_negative_error_column(tmp_path):
    """Test that a negative value in the error column raises a warning."""
    file_path = tmp_path / "neg_error.csv"
    file_path.write_text("time,value,error\n2023-01-01,10,1\n2023-01-02,11,-0.5")
    with pytest.warns(UserWarning, match="error column contains negative values"):
        load_data(file_path, time_col="time", data_col="value", error_col="error")


def test_load_high_frequency_data(tmp_path):
    """
    Test that timestamps with sub-second differences are handled correctly
    and do not raise a monotonicity error.
    """
    file_path = tmp_path / "high_freq.csv"
    # Timestamps are 100ms apart
    file_path.write_text(
        "timestamp,value\n"
        "2023-01-01T00:00:00.100Z,1\n"
        "2023-01-01T00:00:00.200Z,2\n"
        "2023-01-01T00:00:00.300Z,3\n"
    )

    # Before the fix, this would have raised a ValueError because the integer
    # conversion of time would treat all timestamps as the same second.
    # The function should now run without raising an error.
    time_numeric, data, _ = load_data(
        file_path, time_col="timestamp", data_col="value"
    )

    assert len(time_numeric) == 3
    assert len(data) == 3

    # Check that the time difference is approximately 0.1 seconds
    time_diffs = np.diff(time_numeric)
    assert np.all(np.isclose(time_diffs, 0.1))


def test_load_data_ambiguous_columns(tmp_path):
    """
    Test that a file with case-insensitive duplicate column names raises a
    ValueError. This is a regression test for a previously fixed bug.
    """
    file_path = tmp_path / "ambiguous_cols.csv"
    # The column names "Value" and "value" are ambiguous
    file_path.write_text("timestamp,Value,value\n2023-01-01,10,100")
    with pytest.raises(
        ValueError, match="Duplicate column names found \\(case-insensitive\\)"
    ):
        load_data(file_path, time_col="timestamp", data_col="Value")


def test_load_data_time_unit_conversion(tmp_path):
    """Test time unit conversion for the output time array."""
    file_path = tmp_path / "time_conversion.csv"
    file_path.write_text("timestamp,value\n2023-01-01T00:00:00Z,1\n2023-01-02T00:00:00Z,2")

    # Default is seconds
    time_sec, _, _ = load_data(file_path, time_col="timestamp", data_col="value")
    assert np.isclose(time_sec[1], 86400.0)

    # Test conversion to days
    time_days, _, _ = load_data(
        file_path, time_col="timestamp", data_col="value", output_time_unit="days"
    )
    assert np.isclose(time_days[1], 1.0)

    # Test conversion to hours
    time_hours, _, _ = load_data(
        file_path, time_col="timestamp", data_col="value", output_time_unit="hours"
    )
    assert np.isclose(time_hours[1], 24.0)

    # Test invalid unit
    with pytest.raises(ValueError, match="Invalid output_time_unit"):
        load_data(
            file_path, time_col="timestamp", data_col="value", output_time_unit="weeks"
        )


def test_load_data_with_numeric_time_input(tmp_path):
    """Test loading data where the time column is already numeric."""
    file_path = tmp_path / "numeric_time.csv"
    file_path.write_text("time_days,value\n0,10\n1,20\n2,30")

    # 1. Test input in 'days' and default output in 'seconds'
    time_sec, data, _ = load_data(
        file_path,
        time_col="time_days",
        data_col="value",
        input_time_unit="days",
    )
    assert data.iloc[0] == 10
    assert np.allclose(time_sec, [0, 86400, 172800])

    # 2. Test input in 'days' and output in 'hours'
    time_hours, _, _ = load_data(
        file_path,
        time_col="time_days",
        data_col="value",
        input_time_unit="days",
        output_time_unit="hours",
    )
    assert np.allclose(time_hours, [0, 24, 48])

    # 3. Test invalid input unit
    with pytest.raises(ValueError, match="Invalid input_time_unit"):
        load_data(
            file_path,
            time_col="time_days",
            data_col="value",
            input_time_unit="years",
        )


def test_load_data_numeric_time_without_unit_raises_error(tmp_path):
    """
    Test that load_data raises a ValueError if the time column is numeric
    but input_time_unit is not provided.
    """
    file_path = tmp_path / "numeric_time_no_unit.csv"
    file_path.write_text("time,value\n1,10\n2,20\n3,30")

    with pytest.raises(
        ValueError,
        match="The time column 'time' is numeric, but `input_time_unit` was not provided.",
    ):
        load_data(file_path, time_col="time", data_col="value")

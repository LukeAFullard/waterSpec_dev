import pandas as pd
import numpy as np
import os

def load_data(file_path, time_col, data_col):
    """
    Loads time series data from a CSV, JSON, or Excel file.

    The file type is inferred from the file extension.

    Args:
        file_path (str): The path to the data file.
        time_col (str): The name of the column containing the timestamps.
        data_col (str): The name of the column containing the data values.

    Returns:
        tuple[np.ndarray, pd.Series]: A tuple containing:
                                      - time (np.ndarray, as numeric, seconds since epoch)
                                      - data values (pd.Series)

    Raises:
        ValueError: If the file format is not supported or if columns are not found.
    """
    # Get the file extension to determine the file type
    _, file_extension = os.path.splitext(file_path)

    if file_extension.lower() == '.csv':
        df = pd.read_csv(file_path)
    elif file_extension.lower() == '.xlsx':
        df = pd.read_excel(file_path)
    elif file_extension.lower() == '.json':
        df = pd.read_json(file_path)
    else:
        raise ValueError(f"Unsupported file format: {file_extension}")

    # Validate that the required columns exist
    if time_col not in df.columns:
        raise ValueError(f"Time column '{time_col}' not found in the file.")
    if data_col not in df.columns:
        raise ValueError(f"Data column '{data_col}' not found in the file.")

    # Convert the time column to datetime objects
    # The `to_datetime` function is powerful and can infer many formats.
    df[time_col] = pd.to_datetime(df[time_col])

    # Convert datetime objects to a numeric representation (seconds since epoch)
    # The .astype(np.int64) gives nanoseconds, so divide by 10**9 for seconds.
    time_numeric = df[time_col].astype(np.int64) // 10**9

    # Extract the data column
    data_values = df[data_col]

    # Return time as a numpy array and data as a pandas Series
    # to support object types (like strings for censored data).
    return time_numeric.to_numpy(), data_values

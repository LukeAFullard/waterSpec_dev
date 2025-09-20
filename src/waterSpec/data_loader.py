import pandas as pd
import numpy as np
import os
import warnings

def load_data(file_path, time_col, data_col, error_col=None):
    """
    Loads time series data from a CSV, JSON, or Excel file.

    Args:
        file_path (str): The path to the data file.
        time_col (str): The name of the column containing timestamps.
        data_col (str): The name of the column containing data values.
        error_col (str, optional): The name of the column containing measurement
                                   uncertainties (dy). Defaults to None.

    Returns:
        tuple: A tuple containing:
               - np.ndarray: The numeric time array.
               - pd.Series: The data values.
               - pd.Series or None: The measurement uncertainties, or None if not provided.
    """
    _, file_extension = os.path.splitext(file_path)

    if file_extension.lower() == '.csv':
        df = pd.read_csv(file_path)
    elif file_extension.lower() == '.xlsx':
        df = pd.read_excel(file_path)
    elif file_extension.lower() == '.json':
        df = pd.read_json(file_path)
    else:
        raise ValueError(f"Unsupported file format: {file_extension}")

    if df.empty:
        raise ValueError("The provided file is empty or contains only a header.")

    if time_col not in df.columns:
        raise ValueError(f"Time column '{time_col}' not found in the file.")
    if data_col not in df.columns:
        raise ValueError(f"Data column '{data_col}' not found in the file.")

    df[time_col] = pd.to_datetime(df[time_col])
    df = df.sort_values(by=time_col).reset_index(drop=True)
    time_numeric = df[time_col].astype(np.int64) // 10**9

    # Check for strict monotonicity. np.diff should always be > 0.
    time_diffs = np.diff(time_numeric)
    if (time_diffs <= 0).any():
        first_error_index = np.where(time_diffs <= 0)[0][0]
        raise ValueError(
            f"Time column is not strictly monotonic increasing. "
            f"First violation (duplicate or out-of-order timestamp) found at index {first_error_index+1}."
        )

    data_values = df[data_col]
    error_values = None

    if error_col:
        if error_col not in df.columns:
            raise ValueError(f"Error column '{error_col}' not found in the file.")
        error_values = df[error_col]
        if error_values.isnull().any():
            warnings.warn("The error column contains NaN or null values.", UserWarning)
        if (error_values < 0).any():
            warnings.warn("The error column contains negative values.", UserWarning)

    if data_values.isnull().any():
        warnings.warn("The data column contains NaN or null values.", UserWarning)

    return time_numeric.to_numpy(), data_values, error_values

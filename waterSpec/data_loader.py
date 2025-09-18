import pandas as pd
import numpy as np

def load_data(file_path, time_col, data_col):
    """
    Loads time series data from a CSV file.

    Args:
        file_path (str): The path to the CSV file.
        time_col (str): The name of the column containing the timestamps.
        data_col (str): The name of the column containing the data values.

    Returns:
        tuple[np.ndarray, np.ndarray]: A tuple containing two numpy arrays:
                                       - time (as numeric, seconds since epoch)
                                       - data values
    """
    # Read the CSV file into a pandas DataFrame
    df = pd.read_csv(file_path)

    # Validate that the required columns exist
    if time_col not in df.columns:
        raise ValueError(f"Time column '{time_col}' not found in the CSV file.")
    if data_col not in df.columns:
        raise ValueError(f"Data column '{data_col}' not found in the CSV file.")

    # Convert the time column to datetime objects
    # The `to_datetime` function is powerful and can infer many formats.
    df[time_col] = pd.to_datetime(df[time_col])

    # Convert datetime objects to a numeric representation (seconds since epoch)
    # The .astype(np.int64) gives nanoseconds, so divide by 10**9 for seconds.
    time_numeric = df[time_col].astype(np.int64) // 10**9

    # Extract the data column
    data_values = df[data_col]

    # Return as numpy arrays
    return time_numeric.to_numpy(), data_values.to_numpy()

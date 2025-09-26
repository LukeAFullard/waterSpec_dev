import os
import warnings

import numpy as np
import pandas as pd


def load_data(
    file_path, time_col, data_col, error_col=None, time_format=None, sheet_name=0
):
    """
    Loads time series data from a CSV, JSON, or Excel file.
    """
    # 1. Load data from file
    _, file_extension = os.path.splitext(file_path)
    if file_extension.lower() == ".csv":
        df = pd.read_csv(file_path, low_memory=False, index_col=False)
    elif file_extension.lower() in [".xlsx", ".xls"]:
        df = pd.read_excel(file_path, sheet_name=sheet_name)
    elif file_extension.lower() == ".json":
        df = pd.read_json(file_path)
    else:
        raise ValueError(f"Unsupported file format: {file_extension}")

    if df.empty:
        raise ValueError("The provided file is empty or contains only a header.")

    # 2. Check for column existence
    if time_col not in df.columns:
        raise ValueError(f"Time column '{time_col}' not found in the file.")
    if data_col not in df.columns:
        raise ValueError(f"Data column '{data_col}' not found in the file.")

    # 3. Perform validation and type coercion on copies of the series
    # Time column
    original_time_na = df[time_col].isna().sum()
    try:
        time_series = pd.to_datetime(df[time_col], format=time_format, errors="coerce")
    except Exception as e:
        raise ValueError(f"Time format error: {e}")

    if time_series.isna().sum() > original_time_na:
        msg = f"Time column '{time_col}' could not be parsed as datetime objects."
        if time_format:
            msg += f" Please check that the format string '{time_format}' is correct."
        raise ValueError(msg)

    # Data column
    # We do not coerce to numeric here. The preprocessor will handle this,
    # as it needs to parse strings for censored data marks (e.g., "<5").
    data_series = df[data_col]

    # Error column
    error_series = None
    if error_col:
        if error_col not in df.columns:
            warnings.warn(
                f"Error column '{error_col}' not found in the file. "
                "No errors will be used.",
                UserWarning,
            )
            error_series = None
        else:
            original_error_na = df[error_col].isna().sum()
            error_series = pd.to_numeric(df[error_col], errors="coerce")
            if error_series.isna().sum() > original_error_na:
                raise ValueError(
                    f"Error column '{error_col}' could not be converted to a numeric type."
                )
        if error_series is not None and (error_series.dropna() < 0).any():
            warnings.warn("The error column contains negative values.", UserWarning)

    # 4. Issue warnings for any NaNs present in the coerced data
    if data_series.isnull().any():
        warnings.warn("The data column contains NaN or null values.", UserWarning)
    if error_series is not None and error_series.isnull().any():
        warnings.warn("The error column contains NaN or null values.", UserWarning)

    # 5. Create a new, clean DataFrame from the validated series
    clean_df = pd.DataFrame(
        {
            "time": time_series,
            "data": data_series,
        }
    )
    if error_series is not None:
        clean_df["error"] = error_series

    # 6. Drop rows with NaNs in essential columns
    clean_df = clean_df.dropna(subset=["time", "data"]).reset_index(drop=True)

    # 7. Sort by time and check for monotonicity
    clean_df = clean_df.sort_values(by="time").reset_index(drop=True)
    time_numeric = (clean_df["time"].astype(np.int64) // 10**9).to_numpy()

    if len(time_numeric) > 1:
        time_diffs = np.diff(time_numeric)
        if (time_diffs <= 0).any():
            # Find the first index where the time difference is not positive
            first_error_idx = np.where(time_diffs <= 0)[0][0]
            # The violation is at the next timestamp in the original series
            violating_timestamp = clean_df["time"].iloc[first_error_idx + 1]
            raise ValueError(
                "Time column is not strictly monotonic increasing. "
                "First violation (duplicate or out-of-order timestamp) "
                f"found at index {first_error_idx + 1} "
                f"with value: {violating_timestamp}"
            )

    return time_numeric, clean_df["data"], clean_df.get("error")
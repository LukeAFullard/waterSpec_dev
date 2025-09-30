import os
import warnings

import numpy as np
import pandas as pd


def load_data(
    file_path, time_col, data_col, error_col=None, time_format=None, sheet_name=0
):
    """
    Loads time series data from a CSV, JSON, or Excel file, performing robust
    validation.
    """
    # 1. Validate file existence
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"The specified file was not found: {file_path}")

    # 2. Load data from file
    _, file_extension = os.path.splitext(file_path)
    try:
        if file_extension.lower() == ".csv":
            # low_memory=False prevents pandas from inferring column types from
            # chunks of the file, which can lead to mixed-type columns.
            df = pd.read_csv(file_path, low_memory=False, index_col=False)
        elif file_extension.lower() in [".xlsx", ".xls"]:
            df = pd.read_excel(file_path, sheet_name=sheet_name)
        elif file_extension.lower() == ".json":
            df = pd.read_json(file_path)
        else:
            raise ValueError(f"Unsupported file format: {file_extension}")
    except Exception as e:
        raise IOError(f"Failed to read the file at {file_path}. Reason: {e}")

    if df.empty:
        raise ValueError("The provided data file is empty.")

    # 3. Create a mapping of original column names to lowercase
    col_map = {col.lower(): col for col in df.columns}
    time_col_lower = time_col.lower()
    data_col_lower = data_col.lower()

    # Check for column existence (case-insensitive)
    if time_col_lower not in col_map:
        raise ValueError(f"Time column '{time_col}' not found in the file.")
    if data_col_lower not in col_map:
        raise ValueError(f"Data column '{data_col}' not found in the file.")

    # Get original column names from the map
    time_col_orig = col_map[time_col_lower]
    data_col_orig = col_map[data_col_lower]

    # 4. Perform validation and type coercion on copies of the series
    # Time column
    if df[time_col_orig].isnull().all():
        raise ValueError(f"The time column '{time_col}' contains no valid data.")

    original_time_na = df[time_col_orig].isna().sum()
    try:
        time_series = pd.to_datetime(
            df[time_col_orig], format=time_format, errors="coerce"
        )
    except Exception as e:
        raise ValueError(f"Could not parse time column '{time_col}'. Reason: {e}")

    coerced_time_na = time_series.isna().sum()
    if coerced_time_na > original_time_na:
        num_failed = coerced_time_na - original_time_na
        msg = (
            f"{num_failed} value(s) in the time column '{time_col}' could not be "
            f"parsed as datetime objects and were converted to NaT."
        )
        if time_format:
            msg += f" Please check that the format string '{time_format}' is correct."
        raise ValueError(msg)

    # Data column
    if df[data_col_orig].isnull().all():
        raise ValueError(f"The data column '{data_col}' contains no valid data.")
    original_data_na = df[data_col_orig].isna().sum()
    data_series = pd.to_numeric(df[data_col_orig], errors="coerce")
    coerced_data_na = data_series.isna().sum()
    if coerced_data_na > original_data_na:
        num_failed_data = coerced_data_na - original_data_na
        warnings.warn(
            f"{num_failed_data} value(s) in the data column '{data_col}' could not be "
            "converted to a numeric type and were set to NaN.",
            UserWarning,
        )

    # Error column
    error_series = None
    if error_col:
        error_col_lower = error_col.lower()
        if error_col_lower not in col_map:
            warnings.warn(
                f"Error column '{error_col}' not found. No errors will be used.",
                UserWarning,
            )
        else:
            error_col_orig = col_map[error_col_lower]
            if df[error_col_orig].isnull().all():
                warnings.warn(
                    f"The error column '{error_col}' contains no valid data.",
                    UserWarning,
                )

            original_error_na = df[error_col_orig].isna().sum()
            error_series = pd.to_numeric(df[error_col_orig], errors="coerce")
            coerced_error_na = error_series.isna().sum()

            if coerced_error_na > original_error_na:
                num_failed_errors = coerced_error_na - original_error_na
                warnings.warn(
                    f"{num_failed_errors} value(s) in the error column '{error_col}' "
                    "could not be converted to a numeric type and were set to NaN.",
                    UserWarning,
                )
            if error_series is not None and (error_series.dropna() < 0).any():
                warnings.warn(
                    "The error column contains negative values.", UserWarning
                )

    # 5. Create a new, clean DataFrame from the validated series
    clean_df = pd.DataFrame(
        {
            "time": time_series,
            "data": data_series,
        }
    )
    if error_series is not None:
        clean_df["error"] = error_series

    # 6. Drop rows with NaNs in essential columns (time or data)
    initial_rows = len(clean_df)
    clean_df = clean_df.dropna(subset=["time", "data"]).reset_index(drop=True)
    rows_after_drop = len(clean_df)

    if rows_after_drop < initial_rows:
        warnings.warn(
            f"{initial_rows - rows_after_drop} rows were dropped due to missing "
            "time or data values.",
            UserWarning,
        )

    if rows_after_drop == 0:
        raise ValueError(
            "No valid data remains after removing rows with missing time or data values."
        )

    # 7. Sort by time and check for monotonicity
    clean_df = clean_df.sort_values(by="time").reset_index(drop=True)
    # Convert to nanoseconds since epoch to check for monotonicity with full precision.
    time_numeric_ns = clean_df["time"].astype(np.int64).to_numpy()

    # Check for strict monotonicity. After sorting, any non-positive difference
    # indicates a duplicate or out-of-order timestamp.
    if len(time_numeric_ns) > 1:
        time_diffs = np.diff(time_numeric_ns)
        if not np.all(time_diffs > 0):
            # Find the index of the first violation for a precise error message.
            first_violation_idx = np.flatnonzero(time_diffs <= 0)[0] + 1
            violating_timestamp = clean_df["time"].iloc[first_violation_idx]
            raise ValueError(
                "Time column is not strictly monotonic increasing. This can be "
                "caused by duplicate or out-of-order timestamps. First "
                f"violation found at index {first_violation_idx} with value: "
                f"{violating_timestamp}"
            )

    # Return time in seconds (as a float) for compatibility with downstream
    # analysis functions (e.g., Lomb-Scargle).
    time_numeric_sec = time_numeric_ns / 1e9
    return time_numeric_sec, clean_df["data"], clean_df.get("error")
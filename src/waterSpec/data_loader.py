import os
import warnings
from typing import Optional, Tuple, Union, Dict

import numpy as np
import pandas as pd


def _validate_input_dataframe(df: pd.DataFrame) -> None:
    """Validates that the input is a non-empty DataFrame."""
    if not isinstance(df, pd.DataFrame):
        raise TypeError("Input `df` must be a pandas DataFrame.")
    if df.empty:
        raise ValueError("The provided DataFrame is empty.")


def _resolve_column_names(
    df: pd.DataFrame, time_col: str, data_col: str, error_col: Optional[str] = None
) -> Tuple[Dict[str, str], str, str, Optional[str]]:
    """
    Creates a case-insensitive map of columns and resolves provided column names.

    Returns:
        Tuple: (col_map, time_col_orig, data_col_orig, error_col_orig)
    """
    # Check for ambiguities (duplicate lowercased column names)
    lower_cols = pd.Series([c.lower() for c in df.columns])
    if lower_cols.duplicated().any():
        counts = lower_cols.value_counts()
        duplicates = counts[counts > 1].index.tolist()
        raise ValueError(
            "Duplicate column names found (case-insensitive): "
            f"{duplicates}. Please rename columns to be unique."
        )

    col_map = {col.lower(): col for col in df.columns}
    time_col_lower = time_col.lower()
    data_col_lower = data_col.lower()

    if time_col_lower not in col_map:
        raise ValueError(
            f"Time column '{time_col}' not found. Available columns: {list(df.columns)}"
        )
    if data_col_lower not in col_map:
        raise ValueError(
            f"Data column '{data_col}' not found. Available columns: {list(df.columns)}"
        )

    time_col_orig = col_map[time_col_lower]
    data_col_orig = col_map[data_col_lower]

    error_col_orig = None
    if error_col:
        error_col_lower = error_col.lower()
        if error_col_lower not in col_map:
            warnings.warn(
                f"Error column '{error_col}' not found. No errors will be used.",
                UserWarning,
            )
        else:
            error_col_orig = col_map[error_col_lower]

    return col_map, time_col_orig, data_col_orig, error_col_orig


def _process_time_column(
    df: pd.DataFrame,
    time_col_orig: str,
    time_col_name: str,
    time_format: Optional[str],
    input_time_unit: Optional[str],
) -> pd.Series:
    """Validates and processes the time column into a numeric or datetime Series."""
    series = df[time_col_orig]

    if series.isnull().all():
        raise ValueError(f"The time column '{time_col_name}' contains no valid data.")

    if input_time_unit:
        # Time is already numeric
        if input_time_unit not in ["seconds", "days", "hours"]:
            raise ValueError(
                "Invalid input_time_unit. Choose from 'seconds', 'days', or 'hours'."
            )
        time_series = pd.to_numeric(series, errors="coerce")
        if time_series.isnull().all():
            raise ValueError(
                f"Time column '{time_col_name}' has no valid numeric data."
            )
        return time_series

    # Expect datetime-like
    if pd.api.types.is_numeric_dtype(series.dtype):
        raise ValueError(
            f"The time column '{time_col_name}' is numeric, but `input_time_unit` was not provided. "
            "To process numeric time values, please specify `input_time_unit` "
            "(e.g., 'seconds', 'days', or 'hours')."
        )

    original_na = series.isna().sum()
    try:
        time_series = pd.to_datetime(series, format=time_format, errors="coerce")
    except Exception as e:
        raise ValueError(
            f"Could not parse time column '{time_col_name}'. Reason: {e}"
        ) from e

    coerced_na = time_series.isna().sum()
    if coerced_na > original_na:
        num_failed = coerced_na - original_na
        msg = (
            f"{num_failed} value(s) in the time column '{time_col_name}' could not be "
            f"parsed as datetime objects and were converted to NaT."
        )
        if time_format:
            msg += f" Please check that the format string '{time_format}' is correct."
        raise ValueError(msg)

    return time_series


def _process_data_column(
    df: pd.DataFrame,
    data_col_orig: str,
    data_col_name: str,
    coerce_to_numeric: bool,
) -> pd.Series:
    """Validates and processes the data column."""
    series = df[data_col_orig]

    if series.isnull().all():
        raise ValueError(f"The data column '{data_col_name}' contains no valid data.")

    if coerce_to_numeric:
        original_na = series.isna().sum()
        data_series = pd.to_numeric(series, errors="coerce")
        coerced_na = data_series.isna().sum()
        if coerced_na > original_na:
            num_failed = coerced_na - original_na
            warnings.warn(
                f"{num_failed} value(s) in the data column '{data_col_name}' could not be "
                "converted to a numeric type and were set to NaN.",
                UserWarning,
            )
        return data_series
    else:
        return series.copy()


def _process_error_column(
    df: pd.DataFrame,
    error_col_orig: Optional[str],
    error_col_name: Optional[str],
) -> Optional[pd.Series]:
    """Validates and processes the error column if it exists."""
    if not error_col_orig:
        return None

    series = df[error_col_orig]
    if series.isnull().all():
        warnings.warn(
            f"The error column '{error_col_name}' contains no valid data.",
            UserWarning,
        )

    original_na = series.isna().sum()
    error_series = pd.to_numeric(series, errors="coerce")
    coerced_na = error_series.isna().sum()

    if coerced_na > original_na:
        num_failed = coerced_na - original_na
        warnings.warn(
            f"{num_failed} value(s) in the error column '{error_col_name}' "
            "could not be converted to a numeric type and were set to NaN.",
            UserWarning,
        )

    if (error_series.dropna() < 0).any():
        warnings.warn("The error column contains negative values.", UserWarning)

    return error_series


def _check_monotonicity_and_duplicates(
    time_array: np.ndarray, time_col_name: str, clean_df: pd.DataFrame
) -> None:
    """Checks that the numeric time array is strictly increasing."""
    if len(time_array) > 1:
        time_diffs = np.diff(time_array)
        if not np.all(time_diffs > 0):
            # Find the location of the first violation
            first_violation_idx = np.flatnonzero(time_diffs <= 0)[0]
            violating_timestamp = clean_df["time"].iloc[first_violation_idx + 1]

            if time_diffs[first_violation_idx] == 0:
                raise ValueError(
                    f"Duplicate timestamp found in time column '{time_col_name}'. "
                    "The data must have unique and strictly increasing time points. "
                    f"First duplicate found at index {first_violation_idx + 1} "
                    f"with value: {violating_timestamp}"
                )
            else:
                raise ValueError(
                    "Time column is not strictly monotonic increasing after sorting, "
                    "which may indicate a data corruption or sorting issue. First "
                    f"out-of-order timestamp found at index {first_violation_idx + 1} "
                    f"with value: {violating_timestamp}"
                )


def _convert_and_sort_time(
    clean_df: pd.DataFrame,
    time_col_name: str,
    input_time_unit: Optional[str],
    output_time_unit: str,
) -> Tuple[np.ndarray, pd.DataFrame]:
    """
    Sorts the DataFrame by time, checks monotonicity, and converts time to
    the requested output unit relative to the first measurement.
    """
    clean_df = clean_df.sort_values(by="time").reset_index(drop=True)

    if not input_time_unit:
        # Convert datetime to int64 nanoseconds since epoch.
        if pd.api.types.is_datetime64_any_dtype(clean_df["time"]):
            if clean_df["time"].dt.tz is not None:
                clean_df["time"] = (
                    clean_df["time"].dt.tz_convert("UTC").dt.tz_localize(None)
                )

        # Force datetime64[ns] to ensure we have nanoseconds before viewing as int64
        time_numeric_ns = (
            clean_df["time"].astype("datetime64[ns]").to_numpy().view(np.int64)
        )

        _check_monotonicity_and_duplicates(
            time_numeric_ns, time_col_name, clean_df
        )

        # Make time relative *before* converting to float to preserve precision
        time_numeric_ns_relative = time_numeric_ns - time_numeric_ns[0]
        time_numeric_sec = time_numeric_ns_relative.astype(np.float64) / 1e9
    else:
        # Time is already numeric
        time_numeric = clean_df["time"].to_numpy().astype(np.float64)
        _check_monotonicity_and_duplicates(time_numeric, time_col_name, clean_df)

        # Convert input time to seconds for internal consistency
        if input_time_unit == "seconds":
            time_numeric_sec = time_numeric
        elif input_time_unit == "days":
            time_numeric_sec = time_numeric * 86400.0
        elif input_time_unit == "hours":
            time_numeric_sec = time_numeric * 3600.0
        else:
            # Should be caught earlier, but safe fallback
            time_numeric_sec = time_numeric

        # Make relative to the first measurement
        time_numeric_sec = time_numeric_sec - time_numeric_sec[0]

    # Warn if the time span is very large
    if len(time_numeric_sec) > 1:
        time_span_days = time_numeric_sec[-1] / (3600 * 24)
        if time_span_days > 100 * 365:
            warnings.warn(
                f"The time series spans a very large range ({time_span_days / 365.25:.1f} "
                "years). While we use float64 to prevent overflow, be mindful of "
                "potential floating-point precision issues in downstream analysis.",
                UserWarning,
            )

    # Convert to output unit
    if output_time_unit == "seconds":
        time_out = time_numeric_sec
    elif output_time_unit == "days":
        time_out = time_numeric_sec / 86400.0
    elif output_time_unit == "hours":
        time_out = time_numeric_sec / 3600.0
    else:
        raise ValueError(
            "Invalid output_time_unit. Choose from 'seconds', 'days', or 'hours'."
        )

    return time_out, clean_df


def process_dataframe(
    df: pd.DataFrame,
    time_col: str,
    data_col: str,
    error_col: Optional[str] = None,
    time_format: Optional[str] = None,
    input_time_unit: Optional[str] = None,
    output_time_unit: str = "seconds",
    coerce_to_numeric: bool = True,
) -> Tuple[np.ndarray, pd.Series, Optional[pd.Series]]:
    """
    Processes a DataFrame containing time series data, performing robust
    validation and returning time as a numeric array in the specified units.

    Args:
        df (pd.DataFrame): The input DataFrame.
        time_col (str): The name of the column containing timestamps.
        data_col (str): The name of the column containing data values.
        error_col (Optional[str], optional): The name of the column containing
            error values. Defaults to None.
        time_format (Optional[str], optional): The `strftime` format for parsing
            datetime-like columns. If None, `pd.to_datetime` will infer the
            format. Not used if `input_time_unit` is set. Defaults to None.
        input_time_unit (Optional[str], optional): The unit of the time column
            if it's already numeric. Can be 'seconds', 'days', or 'hours'.
            If None, the time column is assumed to be a datetime-like object
            that needs parsing. Defaults to None.
        output_time_unit (str, optional): The desired unit for the output time
            array. Can be 'seconds', 'days', or 'hours'. Defaults to 'seconds'.
        coerce_to_numeric (bool, optional): If True, forces the data column to
            be numeric, converting non-numeric values to NaN. If False, retains
            original values (useful for censored data handling). Defaults to True.

    Returns:
        Tuple[np.ndarray, pd.Series, Optional[pd.Series]]: A tuple containing:
            - A NumPy array of numeric time values, in the unit specified by
              `output_time_unit`, relative to the first measurement.
            - A Pandas Series of data values.
            - A Pandas Series of error values, or None if not provided.
    """
    _validate_input_dataframe(df)
    df = df.copy()

    _, time_col_orig, data_col_orig, error_col_orig = _resolve_column_names(
        df, time_col, data_col, error_col
    )

    time_series = _process_time_column(
        df, time_col_orig, time_col, time_format, input_time_unit
    )
    data_series = _process_data_column(
        df, data_col_orig, data_col, coerce_to_numeric
    )
    error_series = _process_error_column(df, error_col_orig, error_col)

    # Create a new, clean DataFrame from the validated series
    clean_df = pd.DataFrame({"time": time_series, "data": data_series})
    if error_series is not None:
        clean_df["error"] = error_series

    # Drop rows with NaNs in essential columns (time or data)
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

    # Sort, check monotonicity, and convert time
    time_out, clean_df_sorted = _convert_and_sort_time(
        clean_df, time_col, input_time_unit, output_time_unit
    )

    return time_out, clean_df_sorted["data"], clean_df_sorted.get("error")


def load_data(
    file_path: str,
    time_col: str,
    data_col: str,
    error_col: Optional[str] = None,
    time_format: Optional[str] = None,
    input_time_unit: Optional[str] = None,
    sheet_name: Union[int, str] = 0,
    output_time_unit: str = "seconds",
    coerce_to_numeric: bool = True,
) -> Tuple[np.ndarray, pd.Series, Optional[pd.Series]]:
    """
    Loads time series data from a CSV, JSON, or Excel file and processes it.

    This function serves as a wrapper that first loads data from a file into a
    pandas DataFrame and then passes it to the `process_dataframe` function
    for validation, cleaning, and transformation.

    Args:
        file_path (str): The path to the data file.
        time_col (str): The name of the column containing timestamps.
        data_col (str): The name of the column containing data values.
        error_col (Optional[str], optional): The name of the column containing
            error values. Defaults to None.
        time_format (Optional[str], optional): The `strftime` format for parsing
            datetime-like columns. Passed to `process_dataframe`.
        input_time_unit (Optional[str], optional): The unit of the time column
            if it's already numeric. Passed to `process_dataframe`.
        sheet_name (Union[int, str], optional): The name or index of the sheet
            to read from for Excel files. Defaults to 0.
        output_time_unit (str, optional): The desired unit for the output time
            array. Passed to `process_dataframe`.
        coerce_to_numeric (bool, optional): If True, forces the data column to
            be numeric. Defaults to True.

    Returns:
        Tuple[np.ndarray, pd.Series, Optional[pd.Series]]: A tuple containing:
            - A NumPy array of numeric time values.
            - A Pandas Series of data values.
            - A Pandas Series of error values, or None.

    Notes:
        - For Excel files (`.xlsx`, `.xls`), `pd.read_excel` is used. This may
          require installing additional dependencies such as `openpyxl` or
          `xlrd` depending on the file format.
        - For CSV files, `pd.read_csv` is used with its default settings.
    """
    # 1. Validate file existence
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"The specified file was not found: {file_path}")

    # 2. Load data from file
    _, file_extension = os.path.splitext(file_path)
    try:
        if file_extension.lower() == ".csv":
            df = pd.read_csv(file_path, low_memory=False, index_col=False)
        elif file_extension.lower() in [".xlsx", ".xls"]:
            df = pd.read_excel(file_path, sheet_name=sheet_name)
        elif file_extension.lower() == ".json":
            df = pd.read_json(file_path)
        else:
            raise ValueError(f"Unsupported file format: {file_extension}")
    except Exception as e:
        raise IOError(
            f"Failed to read the file at {file_path}. Reason: {e}"
        ) from e

    # 3. Process the loaded DataFrame
    return process_dataframe(
        df,
        time_col,
        data_col,
        error_col=error_col,
        time_format=time_format,
        input_time_unit=input_time_unit,
        output_time_unit=output_time_unit,
        coerce_to_numeric=coerce_to_numeric,
    )
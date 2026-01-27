import pandas as pd
import urllib.request
import urllib.parse
import urllib.error
import io
import sys
import os

def fetch_usgs_data(site, parameter, start_date, end_date, output_file):
    """
    Fetches instantaneous value (IV) data from USGS NWIS.
    """
    base_url = "https://nwis.waterservices.usgs.gov/nwis/iv/"
    params = {
        "format": "rdb",
        "sites": site,
        "startDT": start_date,
        "endDT": end_date,
        "parameterCd": parameter,
        "siteStatus": "all"
    }

    query_string = urllib.parse.urlencode(params)
    url = f"{base_url}?{query_string}"

    print(f"Fetching data from USGS for Site {site}, Parameter {parameter}...")
    try:
        with urllib.request.urlopen(url) as response:
            content = response.read().decode('utf-8')
    except urllib.error.URLError as e:
        print(f"Error fetching data: {e}")
        return False

    # USGS RDB format has a header with comments (#) and then a data table.
    # The line after the column names is a type definition line (e.g., 5s, 15s) which needs to be skipped.

    # Filter out comment lines
    lines = [line for line in content.splitlines() if not line.startswith("#")]

    if not lines:
        print("No data returned.")
        return False

    # The first line is the header, the second is the type definition
    try:
        # Load into pandas, skipping the type definition line (row 1, 0-indexed)
        # We need to handle the fact that read_csv expects a file-like object
        data_io = io.StringIO("\n".join(lines))
        df = pd.read_csv(data_io, sep="\t", skiprows=[1])
    except Exception as e:
        print(f"Error parsing data: {e}")
        print("First few lines of content:")
        print("\n".join(lines[:5]))
        return False

    # Rename columns for clarity
    # USGS returns columns like 'datetime', '02_00060', '02_00060_cd'
    # We want 'timestamp', 'value', 'qualifiers'

    # Find the value column (usually ends with the parameter code, e.g., ..._00060)
    # But often it's the second or third column.
    # Typical columns: agency_cd, site_no, datetime, tz_cd, <value_col>, <code_col>

    value_col = None
    # 1. Look for explicit match: ends with parameter code and NOT _cd
    for col in df.columns:
        if col.endswith(f"_{parameter}") and not col.endswith("_cd"):
             value_col = col
             break

    # 2. Look for any match containing parameter code
    if value_col is None:
        for col in df.columns:
            if parameter in col and not col.endswith("_cd"):
                value_col = col
                break

    # 3. Fallback: look for 5th column if available (standard RDB layout)
    if value_col is None:
         if len(df.columns) > 4:
             # Check if it's numeric-ish or not a metadata column
             candidate = df.columns[4]
             if not candidate.endswith("_cd") and candidate not in ["agency_cd", "site_no", "datetime", "tz_cd"]:
                value_col = candidate

    if value_col is None:
        print(f"Could not identify value column for parameter {parameter}.")
        print("Columns:", df.columns)
        return False

    print(f"Identified value column: {value_col}")

    # clean dataframe
    clean_df = df[['datetime', value_col]].copy()
    clean_df.columns = ['timestamp', 'value']

    # Convert timestamp
    clean_df['timestamp'] = pd.to_datetime(clean_df['timestamp'])

    # Drop duplicates (keep first)
    # This handles DST overlap if timezone info is missing or ambiguous
    # Ideally, we would use the UTC column if available, but cleaning by index is robust enough for this example.
    initial_count = len(clean_df)
    clean_df = clean_df.drop_duplicates(subset='timestamp', keep='first')
    dropped_count = initial_count - len(clean_df)
    if dropped_count > 0:
        print(f"Dropped {dropped_count} duplicate timestamp(s).")

    # Sort just in case
    clean_df = clean_df.sort_values('timestamp')

    # Save to CSV
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    clean_df.to_csv(output_file, index=False)
    print(f"Data saved to {output_file}")
    print(f"Rows: {len(clean_df)}")
    return True

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Fetch USGS NWIS IV data.")
    parser.add_argument("--site", default="05451500", help="USGS Site ID (default: 05451500 - Iowa River at Wapello)")
    parser.add_argument("--param", default="00060", help="Parameter Code (default: 00060 - Discharge)")
    parser.add_argument("--start", default="2020-01-01", help="Start Date (YYYY-MM-DD)")
    parser.add_argument("--end", default="2020-12-31", help="End Date (YYYY-MM-DD)")
    parser.add_argument("--output", default="examples/usgs_discharge_05451500.csv", help="Output CSV path")

    args = parser.parse_args()

    fetch_usgs_data(args.site, args.param, args.start, args.end, args.output)

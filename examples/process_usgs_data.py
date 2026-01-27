import pandas as pd
import sys
import os

def process_rdb(input_file, output_file, param_code):
    print(f"Processing {input_file}...")

    # Read the file, identifying the header line
    # RDB files usually have a header line, then a data type line
    # We need to find the header line. Usually starts with agency_cd or similar.
    # Comments start with #.

    if not os.path.exists(input_file):
        print(f"Error: {input_file} does not exist")
        return

    with open(input_file, 'r') as f:
        lines = f.readlines()

    data_lines = [line for line in lines if not line.startswith('#')]

    # The first line of data_lines is header
    # The second line is data types
    # The rest are data

    if len(data_lines) < 3:
        print(f"Error: Not enough data in {input_file}")
        return

    header = data_lines[0].strip().split('\t')
    # Find the value column. It usually contains the param code.
    # E.g. 68478_00060_00003

    val_col = None
    for col in header:
        if param_code in col and not col.endswith('_cd'):
            val_col = col
            break

    if not val_col:
        print(f"Error: Could not find column for parameter {param_code} in {header}")
        return

    print(f"Found value column: {val_col}")

    # Load into pandas
    try:
        df = pd.read_csv(input_file, sep='\t', comment='#', header=0, parse_dates=['datetime'])
    except Exception as e:
        print(f"Pandas read error: {e}")
        return

    # Drop the second row (type definition)
    if 'agency_cd' in df.columns and str(df.iloc[0]['agency_cd']).endswith('s'):
         df = df.iloc[1:].copy()

    # Select columns
    df_clean = df[['datetime', val_col]].copy()
    df_clean.columns = ['timestamp', 'value']

    # Convert value to numeric, coercing errors
    df_clean['value'] = pd.to_numeric(df_clean['value'], errors='coerce')

    # Drop NaNs
    df_clean.dropna(subset=['value'], inplace=True)

    # Save
    df_clean.to_csv(output_file, index=False)
    print(f"Saved {len(df_clean)} records to {output_file}")

if __name__ == "__main__":
    # Nitrate (99133)
    process_rdb("usgs_nitrate_wapello_99133_real.txt", "examples/usgs_nitrate_wapello_05465500.csv", "99133")

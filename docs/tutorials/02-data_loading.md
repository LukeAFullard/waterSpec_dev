# Tutorial 2: A Guide to Data Loading

The `load_data` function is your trusty crane for bringing data aboard from all sorts of places. This tutorial will show you how to load data from the most common file formats and how to include measurement errors for a more precise analysis.

### Step 1: Loading a Classic CSV File

The most common format is the humble CSV file. Let's start by loading our `sample_data.csv`. All you need to do is provide the file path and tell the function which columns contain your timestamps and data.

```python
from waterSpec import load_data

time, data, errors = load_data(
    file_path='examples/sample_data.csv',
    time_col='timestamp',
    data_col='concentration'
)

print(f"Loaded {len(time)} data points from CSV.")
print(f"Errors column is: {errors}")
```

**Output:**
```text
--- CSV Loading ---
Loaded 50 data points from CSV.
Errors column is: None
```

### Step 2: Handling Other Formats (JSON and Excel)

The `load_data` function is smart enough to figure out the file type from its extension. You don't need to change anything in your code to load a `.json` or `.xlsx` file. Let's try it!

```python
# Load from a JSON file
time_json, data_json, _ = load_data(
    file_path='examples/sample_data.json',
    time_col='timestamp',
    data_col='concentration'
)
print(f"Loaded {len(time_json)} data points from JSON.")

# Load from an Excel file
time_excel, data_excel, _ = load_data(
    file_path='examples/sample_data.xlsx',
    time_col='timestamp',
    data_col='concentration'
)
print(f"Loaded {len(time_excel)} data points from Excel.")
```

**Output:**
```text
--- Other Format Loading ---
Loaded 4 data points from JSON.
Loaded 50 data points from Excel.
```

### Step 3: Including Measurement Errors

Real-world data often comes with measurement uncertainties. Providing these errors (`dy`) to the analysis functions can lead to more robust and statistically sound results. To do this, simply add the `error_col` argument to the `load_data` function, pointing it to the column containing your error values.

```python
time_err, data_err, errors_err = load_data(
    file_path='examples/sample_data_with_errors.csv',
    time_col='timestamp',
    data_col='concentration',
    error_col='concentration_error'  # Here's the new argument!
)

print(f"Loaded {len(time_err)} data points with errors.")
print("First 5 error values:")
print(errors_err.head())
```

**Output:**
```text
--- Loading with Errors ---
Loaded 50 data points with errors.
First 5 error values:
0    0.41
1    0.28
2    0.44
3    0.38
4    0.14
Name: concentration_error, dtype: float64
```

### Onward!

You're now a master of loading cargo. In the next tutorial, we'll look at what to do with the data once it's aboard: preprocessing.

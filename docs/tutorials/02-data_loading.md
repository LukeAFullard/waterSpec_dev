# Tutorial 2: A Guide to Data Loading

In the new `waterSpec` workflow, data loading is the first and most fundamental step, handled automatically when you create an `Analysis` object. This tutorial will show you how to load data from the most common file formats and how to include measurement errors for a more precise analysis.

### The `Analysis` Class Constructor

Everything starts with the `Analysis` class. Its constructor (`__init__`) is designed to be your single point of entry for data. It's smart enough to figure out the file type from its extension.

### Step 1: Loading a Classic CSV File

Let's start by creating an `Analysis` object from our `sample_data.csv`. All you need to do is provide the file path and tell the constructor which columns contain your timestamps and data.

```python
from waterSpec import Analysis

analyzer_csv = Analysis(
    file_path='examples/sample_data.csv',
    time_col='timestamp',
    data_col='concentration'
)

print(f"Successfully created an Analysis object from a CSV file.")
print(f"Number of data points loaded: {len(analyzer_csv.data)}")
```

### Step 2: Handling Other Formats (JSON and Excel)

You don't need to change anything in your code to load a `.json` or `.xlsx` file. Just point the `file_path` to the right file.

```python
# Load from a JSON file
analyzer_json = Analysis(
    file_path='examples/sample_data.json',
    time_col='timestamp',
    data_col='concentration'
)
print(f"Successfully loaded from JSON. Points: {len(analyzer_json.data)}")

# Load from an Excel file
analyzer_excel = Analysis(
    file_path='examples/sample_data.xlsx',
    time_col='timestamp',
    data_col='concentration'
)
print(f"Successfully loaded from Excel. Points: {len(analyzer_excel.data)}")
```

### Step 3: Including Measurement Errors

Real-world data often comes with measurement uncertainties. Providing these errors (`dy`) can lead to more robust and statistically sound results. To do this, simply add the `error_col` argument to the constructor, pointing it to the column containing your error values.

```python
analyzer_with_errors = Analysis(
    file_path='examples/sample_data_with_errors.csv',
    time_col='timestamp',
    data_col='concentration',
    error_col='concentration_error'  # Here's the new argument!
)

print(f"Successfully loaded data with an error column.")
print(f"Number of error values loaded: {len(analyzer_with_errors.errors)}")
```

### Onward!

You're now a master of loading cargo. The `Analysis` object has already preprocessed this data for you. In the next tutorial, we'll look at the different preprocessing options you can control during this initialization step.

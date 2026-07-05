from waterSpec import Analysis

# 1. Define the path to your data file
file_path = 'examples/sample_data.csv'

# 2. Create the analyzer object
# This loads and preprocesses the data immediately.
analyzer = Analysis(
    file_path=file_path,
    time_col='timestamp',
    data_col='concentration',
    param_name='Nitrate Concentration at Site A' # A descriptive name for plots
)

# 3. Run the full analysis
# This command runs the analysis, saves the outputs, and returns the results.
results = analyzer.run_full_analysis(
    output_dir='example_output',
    ci_method='parametric' # Use faster CI calculation
)

# The summary text is available in the returned dictionary
print(results['summary_text'])
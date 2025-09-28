from waterSpec import Analysis

# 1. Define the path to your data file
file_path = 'examples/segmented_data.csv'

# 2. Create the analyzer object
analyzer = Analysis(
    file_path=file_path,
    time_col='timestamp',
    data_col='value',
    param_name='Segmented Spectrum Example'
)

# 3. Run the full analysis
# We'll run a faster analysis by reducing the grid points and using parametric CIs
results = analyzer.run_full_analysis(
    output_dir='example_output',
    num_grid_points=100,      # Lower resolution for speed
    ci_method='parametric'    # Use faster CI calculation
)

# The summary text is available in the returned dictionary
print(results['summary_text'])
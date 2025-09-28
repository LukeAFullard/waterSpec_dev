from waterSpec import Analysis

# 1. Define the path to your data file
file_path = 'examples/periodic_data.csv'

# 2. Create the analyzer object
analyzer = Analysis(
    file_path=file_path,
    time_col='timestamp',
    data_col='value',
    param_name='Peak Detection Example'
)

# 3. Run the full analysis
# We'll use settings that are friendly for a quick example run
results = analyzer.run_full_analysis(
    output_dir='example_output',
    ci_method='parametric',
    peak_detection_method='fap', # Use FAP for this example
    fap_threshold=0.05
)

# The summary text is available in the returned dictionary
print(results['summary_text'])
from waterSpec import Analysis

print("--- Running Tutorial 01: Quickstart ---")

# Create the analyzer object
analyzer = Analysis(
    file_path='examples/sample_data.csv',
    time_col='timestamp',
    data_col='concentration'
)

# Run the full analysis
results = analyzer.run_full_analysis(output_dir='docs/tutorials/quickstart_outputs')

print("--- Tutorial 01 Complete ---")
print("Outputs saved to 'docs/tutorials/quickstart_outputs/'")

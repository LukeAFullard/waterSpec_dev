from waterSpec import Analysis
import os

# Define the path to your data file
file_path = 'examples/sample_data.csv'

# Create the analyzer object
# This loads and preprocesses the data immediately.
analyzer = Analysis(
    file_path=file_path,
    time_col='timestamp',
    data_col='concentration',
    param_name='Nitrate Concentration' # Optional: for better plot titles and summaries
)

print("Analysis object created and data preprocessed.")

# Define an output directory
output_dir = 'readme_output'

# Run the entire workflow
results_dict = analyzer.run_full_analysis(output_dir=output_dir)

# The results are also stored in the object and can be inspected
# print(analyzer.results['summary_text'])
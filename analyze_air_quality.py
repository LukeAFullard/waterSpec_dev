from waterSpec import Analysis
import os
import pandas as pd

# Load the data and handle duplicates
df = pd.read_excel('assets/data/Daily_air_quality.xlsx')
df_cleaned = df.groupby('Sample Date')['Concentration (ug/m3)'].mean().reset_index()

# Save the cleaned data to a new CSV file
cleaned_data_path = 'assets/data/cleaned_air_quality.csv'
df_cleaned.to_csv(cleaned_data_path, index=False)


# Create output directory if it doesn't exist
output_dir = 'air_quality_report'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Create analyzer and load data
analyzer = Analysis(
    file_path=cleaned_data_path,
    time_col='Sample Date',
    data_col='Concentration (ug/m3)',
    param_name='Daily Air Quality'
)

# Run complete analysis with automatic model selection
results = analyzer.run_full_analysis(
    output_dir=output_dir,
    ci_method='parametric',  # Use parametric CIs for speed
    max_breakpoints=2
)

# View summary
print(results['summary_text'])

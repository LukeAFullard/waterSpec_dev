import numpy as np
import pandas as pd
from waterSpec import Analysis

print("--- Running Tutorial 06: Advanced Peak Finding ---")

# Create a synthetic time series
n_points = 365
time = pd.to_datetime(np.arange(n_points), unit='D', origin='2022-01-01')
period = 30 # days
frequency_cpd = 1.0 / period # cycles per day
rng = np.random.default_rng(42)
series = 5 * np.sin(2 * np.pi * frequency_cpd * np.arange(n_points)) + rng.normal(0, 0.1, n_points)

# Save it to a temporary file
file_path = 'examples/periodic_data.csv'
df = pd.DataFrame({'timestamp': time, 'value': series})
df.to_csv(file_path, index=False)


# Initialize the analyzer
analyzer = Analysis(
    file_path='examples/periodic_data.csv',
    time_col='timestamp',
    data_col='value',
    detrend_method=None # Important: Don't detrend away the signal!
)

# Run the analysis with FAP detection
results = analyzer.run_full_analysis(
    output_dir='docs/tutorials/fap_outputs',
    fap_threshold=0.01,
    grid_type='linear' # Use linear grid for better peak resolution
)

print("--- Tutorial 06 Complete ---")
print("Outputs saved to 'docs/tutorials/fap_outputs/'")

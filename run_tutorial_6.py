import numpy as np
import pandas as pd
from waterSpec import run_analysis
from waterSpec import load_data
from waterSpec.spectral_analyzer import find_significant_peaks, calculate_periodogram
import os

# --- Block 1: Create data ---
# Create a synthetic time series
n_points = 365
time = pd.to_datetime(np.arange(n_points), unit='D', origin='2022-01-01')
period = 30 # days
frequency_cpd = 1.0 / period # cycles per day
rng = np.random.default_rng(42)
series = 5 * np.sin(2 * np.pi * frequency_cpd * np.arange(n_points)) + rng.normal(0, 0.1, n_points)

# Save it to a temporary file
# Ensure the examples directory exists
os.makedirs('examples', exist_ok=True)
file_path = 'examples/periodic_data.csv'
df = pd.DataFrame({'timestamp': time, 'value': series})
df.to_csv(file_path, index=False)
print(f"Created data file at: {file_path}")

# --- Block 2: Manually Run Peak Detection ---
print("\n--- Manual Peak Detection ---")
# Load the data
time_numeric, data_series, _ = load_data(file_path, 'timestamp', 'value')

# Generate a LINEAR frequency grid
duration = np.max(time_numeric) - np.min(time_numeric)
min_freq_hz = 1 / duration
nyquist_freq_hz = 0.5 / np.median(np.diff(time_numeric))
linear_frequency_grid = np.linspace(min_freq_hz, nyquist_freq_hz, 5000)

# Calculate periodogram to get the LombScargle object
_, power, ls_obj = calculate_periodogram(
    time_numeric,
    data_series.to_numpy(),
    frequency=linear_frequency_grid
)

# Find significant peaks
peaks, fap_level = find_significant_peaks(
    ls_obj,
    linear_frequency_grid,
    power,
    fap_threshold=0.01,
    fap_method='baluev' # Use a fast method for the tutorial
)

print("Found significant peaks:")
for peak in peaks:
    freq_cpd = peak['frequency'] * 86400 # Convert from Hz to cycles/day
    print(f"  - Frequency: {freq_cpd:.4f} cycles/day, Power: {peak['power']:.2f}, FAP: {peak['fap']:.2E}")

# --- Block 3: Visualize with run_analysis ---
print("\n--- Visualizing with run_analysis ---")
plot_path = 'docs/tutorials/06_fap_plot.png'
# Ensure the tutorials directory exists
os.makedirs('docs/tutorials', exist_ok=True)

results = run_analysis(
    file_path=file_path,
    time_col='timestamp',
    data_col='value',
    detrend_method=None, # Important: Don't detrend away the signal we want to find!
    fap_threshold=0.01,
    do_plot=True,
    output_path=plot_path,
    grid_type='linear' # Use linear grid for better peak resolution
)

print(f"Plot with FAP annotations saved to: {plot_path}")

print("\n--- Summary Text from Report ---")
print(results['summary_text'])

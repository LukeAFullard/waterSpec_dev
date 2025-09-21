from waterSpec import Analysis
from waterSpec.plotting import plot_spectrum
import os

print("--- Running Tutorial 07: Plotting ---")

# We use the 'periodic_data.csv' file created in the previous tutorials.
file_path = 'examples/periodic_data.csv'
output_dir = 'docs/tutorials/plotting_outputs'

# Initialize and run the analysis
analyzer = Analysis(
    file_path=file_path,
    time_col='timestamp',
    data_col='value',
    detrend_method=None
)
analyzer.run_full_analysis(
    output_dir=output_dir,
    fap_threshold=0.01,
    grid_type='linear'
)

# Manually call the plotting function for a custom plot
custom_plot_path = os.path.join(output_dir, 'custom_plot.svg')
plot_spectrum(
    frequency=analyzer.frequency,
    power=analyzer.power,
    fit_results=analyzer.results,
    analysis_type=analyzer.results['chosen_model'],
    output_path=custom_plot_path,
    param_name=analyzer.param_name
)

print("--- Tutorial 07 Complete ---")
print(f"Outputs saved to '{output_dir}'")

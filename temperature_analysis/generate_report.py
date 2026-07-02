import os
import copy
from waterSpec import Analysis
from waterSpec.reporting import ReportGenerator
from waterSpec.interpreter import interpret_results
import warnings

warnings.filterwarnings("ignore")

file_path = 'daily_temperature_station_id_32504.csv'
out_dir = 'temperature_analysis_output'

analyzer = Analysis(
    file_path=file_path,
    time_col='time',
    data_col='temperature',
    param_name='Daily Air Temperature'
)

# Run full analysis
results = analyzer.run_full_analysis(
    output_dir=out_dir,
    ci_method='parametric',
    run_haar=True
)

original_interp_text = results.get('summary_text', '')

if 'haar_results' in results:
    haar_res = results['haar_results'].copy()
    if 'n_eff' in haar_res:
        import numpy as np
        if isinstance(haar_res['n_eff'], (np.ndarray, list)):
            haar_res['n_eff'] = np.mean(haar_res['n_eff'])
    if 'n_effective' in haar_res:
        import numpy as np
        if isinstance(haar_res['n_effective'], (np.ndarray, list)):
            haar_res['n_effective'] = np.mean(haar_res['n_effective'])

    # For sub-models if applicable
    for k, v in haar_res.items():
        if isinstance(v, dict):
            if 'n_eff' in v:
                if isinstance(v['n_eff'], (np.ndarray, list)):
                    v['n_eff'] = np.mean(v['n_eff'])
            if 'n_effective' in v:
                if isinstance(v['n_effective'], (np.ndarray, list)):
                    v['n_effective'] = np.mean(v['n_effective'])

    if 'analysis_mode' in haar_res:
         del haar_res['analysis_mode']

    haar_out = interpret_results(haar_res, param_name="Daily Air Temperature (Haar Analysis)")
    haar_interp = haar_out['summary_text']

    # Update summary text and replace hydrology terminology
    combined_text = original_interp_text + "\n\n-----------------------------------\n\n" + haar_interp
    combined_text = combined_text.replace("suggesting event-driven transport", "suggesting short-term atmospheric variability")
    combined_text = combined_text.replace("suggesting transport is damped by storage", "suggesting long-term climatic persistence")
    combined_text = combined_text.replace("Low (Event-driven)", "Low (High-frequency variability)")
    combined_text = combined_text.replace("High (Storage-dominated)", "High (Climatic memory)")
    combined_text = combined_text.replace("Contextual Comparison: Similar to Ortho-P (Surface/Shallow subsurface-dominated).", "")

    results['summary_text'] = combined_text

# Explicitly add plot paths to results so ReportGenerator can find them
spectrum_plot = os.path.join(out_dir, 'Daily_Air_Temperature_spectrum_plot.png')
haar_plot = os.path.join(out_dir, 'Daily_Air_Temperature_haar_plot.png')

if os.path.exists(spectrum_plot):
    results['spectrum_plot_path'] = spectrum_plot
if os.path.exists(haar_plot):
    if 'haar_results' not in results:
        results['haar_results'] = {}
    results['haar_results']['haar_plot_path'] = haar_plot

# Generate markdown report
report_gen = ReportGenerator(results)
report_gen.to_markdown("final_report.md")

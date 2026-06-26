import pandas as pd
import numpy as np
import os
from waterSpec.analysis import Analysis
from waterSpec.reporting import ReportGenerator

analyzer = Analysis(
    file_path='Water_Temperature_Pukeokahu_fixed.csv',
    time_col='Time',
    data_col='Rangitikei at Pukeokahu',
    param_name='Water Temperature'
)

results = analyzer.run_full_analysis(
    output_dir='analysis_output',
    ci_method='parametric',
    run_haar=True,
    haar_overlap=True,
    samples_per_peak=1
)

report_gen = ReportGenerator(results)
os.makedirs('analysis_output', exist_ok=True)
report_gen.to_json(os.path.join('analysis_output', 'report.json'))
report_gen.to_csv(os.path.join('analysis_output', 'report.csv'))
report_gen.to_markdown(os.path.join('analysis_output', 'report.md'))

try:
    report_gen.to_html(os.path.join('analysis_output', 'report.html'))
except Exception as e:
    print("HTML exception:", e)

with open(os.path.join('analysis_output', 'README.md'), 'w') as f:
    f.write("# Water Temperature Pukeokahu Analysis\n\n")
    f.write("## Interpretation and Discussion\n\n")
    f.write(results['summary_text'])

print("Analysis Complete.")

import numpy as np
import os
import sys

# Ensure waterSpec and validation.common are importable
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'src')))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from validation.common import record_result_v2, generate_colored_noise
from waterSpec.analysis import Analysis
from waterSpec.interpreter import interpret_results, get_scientific_interpretation

SECTION_SEED = 1000 * 16
RESULTS_CSV = 'validation/results/section16_results.csv'
os.makedirs('validation/results', exist_ok=True)
os.makedirs('validation/plots', exist_ok=True)
os.makedirs('validation/section_16_out', exist_ok=True)

def run_test_16_1():
    print("Running 16.1: run_full_analysis output completeness")
    target_beta = 1.0
    N = 1024
    dt = 1.0
    seed = SECTION_SEED + 1

    time, data = generate_colored_noise(N, target_beta, dt=dt, seed=seed)

    analysis = Analysis(time_array=time, data_array=data, time_col='time', data_col='data', input_time_unit='seconds')
    out_dir = 'validation/section_16_out/completeness'
    res = analysis.run_full_analysis(output_dir=out_dir)

    files_expected = ['data_summary.txt', 'data_spectrum_plot.png']
    all_exist = True
    for f in files_expected:
        if not os.path.exists(os.path.join(out_dir, f)):
            print(f"Missing expected output file: {f}")
            all_exist = False

    passed = all_exist
    record_result_v2(RESULTS_CSV, '16.1', seed, f'N={N}', 1.0 if passed else 0.0, 1.0, 0, 0, passed)
    return passed

def run_test_16_2():
    print("Running 16.2: interpret_results correctness")
    res_white = get_scientific_interpretation(0.0)
    res_pink = get_scientific_interpretation(1.0)
    res_fgn = get_scientific_interpretation(0.5)
    res_fbm = get_scientific_interpretation(1.5)
    res_black = get_scientific_interpretation(2.5)

    passed = True
    if 'White Noise' not in res_white: passed = False
    if 'Pink Noise' not in res_pink: passed = False
    if 'fGn-like' not in res_fgn: passed = False
    if 'fBm-like' not in res_fbm: passed = False

    record_result_v2(RESULTS_CSV, '16.2', 0, '', 1.0 if passed else 0.0, 1.0, 0, 0, passed)
    return passed

def run_test_16_3():
    print("Running 16.3: ReportGenerator templates")
    from waterSpec.reporting import ReportGenerator

    time, data = generate_colored_noise(1024, 1.0, dt=1.0, seed=123)
    analysis = Analysis(time_array=time, data_array=data, time_col='time', data_col='data', input_time_unit='seconds')
    out_dir = 'validation/section_16_out/completeness'
    res = analysis.run_full_analysis(output_dir=out_dir)

    reporter = ReportGenerator(res, metadata={"variable": "TestVar"})
    reporter.to_html(os.path.join(out_dir, "analysis_report.html"))
    reporter.to_markdown(os.path.join(out_dir, "analysis_report.md"))

    html_exists = os.path.exists(os.path.join(out_dir, 'analysis_report.html'))
    md_exists = os.path.exists(os.path.join(out_dir, 'analysis_report.md'))
    passed = html_exists and md_exists
    record_result_v2(RESULTS_CSV, '16.3', 0, '', 1.0 if passed else 0.0, 1.0, 0, 0, passed)
    return passed

def run_test_16_4():
    print("Running 16.4: Plotting sanity")
    out_dir = 'validation/section_16_out/completeness'
    plot_path = os.path.join(out_dir, 'data_spectrum_plot.png')
    passed = os.path.exists(plot_path) and os.path.getsize(plot_path) > 1000
    record_result_v2(RESULTS_CSV, '16.4', 0, '', 1.0 if passed else 0.0, 1.0, 0, 0, passed)
    return passed

if __name__ == '__main__':
    all_passed = True
    all_passed &= run_test_16_1()
    all_passed &= run_test_16_2()
    all_passed &= run_test_16_3()
    all_passed &= run_test_16_4()
    sys.exit(0)

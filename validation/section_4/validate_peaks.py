import sys
import os
import numpy as np
import pandas as pd
import warnings

# Add root so we can import waterSpec and common validation scripts
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from validation.common import generate_colored_noise, inject_seasonality, record_result, get_seed
from src.waterSpec.analysis import Analysis

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")
os.makedirs(RESULTS_DIR, exist_ok=True)
DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
os.makedirs(DATA_DIR, exist_ok=True)

N_TRIALS = 10
N_POINTS = 2048

def run_peak_test(test_id, period, true_periods, amplitude, beta, method, threshold=0.01, fdr_level=0.05, trials=N_TRIALS):
    successes = 0
    for i in range(trials):
        seed = get_seed(4, int(test_id.replace('.','').replace('_amp_','').replace('_fap','').replace('_residual','').split('_')[0])*1000 + i)
        t, data = generate_colored_noise(beta=beta, N=N_POINTS, seed=seed)

        if period is not None:
            if isinstance(period, list):
                for p in period:
                    t, data = inject_seasonality(t, data, period=p, amplitude=amplitude)
            else:
                t, data = inject_seasonality(t, data, period=period, amplitude=amplitude)

        df = pd.DataFrame({"time": t, "data": data})

        tmp_file = os.path.join(DATA_DIR, f"peak_data_{test_id}_{i}.csv")
        df.to_csv(tmp_file, index=False)

        analyzer = Analysis(time_col="time", data_col="data", file_path=tmp_file, time_format="numeric", input_time_unit="days")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            results = analyzer.run_full_analysis(
                output_dir=os.path.join(RESULTS_DIR, f"output_section4_{test_id}"),
                peak_detection_method=method,
                peak_fdr_level=fdr_level,
                fap_threshold=threshold,
                run_haar=False,
                ci_method='parametric', # parametric is much faster for Lomb-Scargle
                samples_per_peak=1 # to avoid slow grid generation
            )

        peaks = results.get('significant_peaks', [])
        found_frequencies = [p['frequency'] for p in peaks]

        passed = False
        if true_periods is not None:
            # Positive control
            all_found = True
            for tp in true_periods:
                true_f = 1.0 / (tp * 86400.0)
                # resolution is approx 1/(N * dt_seconds)
                res = 1.0 / (N_POINTS * 86400.0)

                # allow some tolerance
                found = any(abs(f - true_f) <= 10*res for f in found_frequencies)
                if not found:
                    all_found = False
                    break
            if all_found:
                passed = True
        else:
            # Negative control: no peaks should be found
            if len(peaks) == 0:
                passed = True

        if passed:
            successes += 1

        record_result(
            test_id=test_id,
            seed=seed,
            params_dict={"beta": beta, "method": method, "amplitude": amplitude},
            estimate=len(peaks),
            truth=len(true_periods) if true_periods else 0,
            ci_low=np.nan,
            ci_high=np.nan,
            passed=passed,
            results_dir=RESULTS_DIR
        )
    print(f"Test {test_id} (Method: {method}, Amp: {amplitude}): {successes}/{trials} passed.")
    return successes

def main():
    print("Running Section 4: Seasonality / Periodicity (Peak Detection)")

    # 4.1 Pure periodic + white noise, peak detection ("fap")
    run_peak_test("4.1", period=365.25, true_periods=[365.25], amplitude=3.0, beta=0.0, method="fap")

    # 4.2 Peak detection with residual/FDR method
    run_peak_test("4.2", period=365.25, true_periods=[365.25], amplitude=3.0, beta=0.0, method="residual")

    # 4.3 Weak periodicity / detection threshold sweep
    print("Running sweep for 4.3 (fap)")
    for amp in [5.0, 2.0, 1.0, 0.5, 0.2, 0.1]:
        run_peak_test(f"4.3_amp_{amp}_fap", period=365.25, true_periods=[365.25], amplitude=amp, beta=1.0, method="fap", trials=5)

    print("Running sweep for 4.3 (residual)")
    for amp in [5.0, 2.0, 1.0, 0.5, 0.2, 0.1]:
        run_peak_test(f"4.3_amp_{amp}_residual", period=365.25, true_periods=[365.25], amplitude=amp, beta=1.0, method="residual", trials=5)

    # 4.4 Multiple simultaneous periodicities
    run_peak_test("4.4", period=[365.25, 7.0], true_periods=[365.25, 7.0], amplitude=2.0, beta=1.0, method="residual")

    # 4.5 False-positive rate (negative control)
    print("Running 4.5 (fap negative control) - Beta 1.0")
    run_peak_test("4.5_fap", period=None, true_periods=None, amplitude=0.0, beta=1.0, method="fap", trials=20)
    print("Running 4.5 (residual negative control) - Beta 1.0")
    run_peak_test("4.5_residual", period=None, true_periods=None, amplitude=0.0, beta=1.0, method="residual", trials=20)

if __name__ == "__main__":
    main()

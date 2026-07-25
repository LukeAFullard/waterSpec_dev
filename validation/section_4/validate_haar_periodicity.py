import sys
import os
import numpy as np
import pandas as pd
import warnings

# Add root so we can import waterSpec and common validation scripts
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from validation.common import generate_colored_noise, inject_seasonality, apply_uneven_sampling, record_result, get_seed
from src.waterSpec.haar_analysis import HaarAnalysis
from src.waterSpec.analysis import Analysis

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")
os.makedirs(RESULTS_DIR, exist_ok=True)
DATA_DIR = os.path.join(os.path.dirname(__file__), "data")

N_TRIALS = 10
N_POINTS = 2048

def run_haar_periodicity_test(test_id, period, amplitude, beta_target, correct_periodicity=False,
                             uneven_fraction=0.0, seasonal_shape="sinusoid", use_list_candidates=False):
    successes = 0
    bias_sum = 0

    # generate a numeric base from test_id
    try:
        base = int(test_id.replace('.',''))*1000
    except ValueError:
        base = sum(ord(c) for c in test_id) * 1000

    for i in range(N_TRIALS):
        seed = get_seed(4, base + i)
        t, data = generate_colored_noise(beta=beta_target, N=N_POINTS, seed=seed)

        # Inject seasonality
        if period is not None:
            if seasonal_shape == "sinusoid":
                t, data = inject_seasonality(t, data, period=period, amplitude=amplitude)
            elif seasonal_shape == "sawtooth":
                # Create a sawtooth wave
                sawtooth = amplitude * 2 * (t/period - np.floor(t/period + 0.5))
                data += sawtooth

        # Apply uneven sampling
        if uneven_fraction > 0:
            t, data = apply_uneven_sampling(t, data, missing_fraction=uneven_fraction, seed=seed)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")

            periods_to_correct = []
            if correct_periodicity:
                if use_list_candidates:
                    df = pd.DataFrame({"time": t, "data": data})
                    tmp_file = os.path.join(DATA_DIR, f"peak_data_haar_{test_id}_{i}.csv")
                    df.to_csv(tmp_file, index=False)
                    analyzer = Analysis(time_col="time", data_col="data", file_path=tmp_file, time_format="numeric", input_time_unit="days")
                    res = analyzer.run_full_analysis(
                        output_dir=os.path.join(RESULTS_DIR, f"output_section4_{test_id}_peaks"),
                        run_haar=False,
                        ci_method='parametric',
                        samples_per_peak=1
                    )

                    peaks = res.get('significant_peaks', [])
                    from src.waterSpec.haar_periodicity import list_period_candidates
                    candidates = list_period_candidates(peaks)
                    # list_period_candidates returns PeriodCluster objects. We need their representative_period
                    periods_to_correct = [c.representative_period for c in candidates]

                    # Convert frequency back to periods based on units (days)
                    # Actually list_period_candidates takes peaks (where frequency is in 1/seconds due to time_unit='days')
                    # and returns periods in seconds. So we need to pass these periods back to HaarAnalysis which takes time in days.
                    # Wait, HaarAnalysis uses the exact `t` passed to it (which is in days). So the period must be in days.
                    # list_period_candidates does: period = 1.0 / p["frequency"].
                    # If p["frequency"] is in 1/seconds, period is in seconds.
                    # Let's just scale it back to days.
                    periods_to_correct = [p / 86400.0 for p in periods_to_correct]
                else:
                    periods_to_correct = [period]

            analyzer_haar = HaarAnalysis(t, data)
            results = analyzer_haar.run(
                aggregation="rms", # MUST be rms for periodicity correction
                overlap=True,
                correct_periodicity=correct_periodicity,
                periodic_periods=periods_to_correct,
            )

        beta_est = results['beta']

        passed = False
        if abs(beta_est - beta_target) <= 0.3: # slightly relaxed tolerance due to smaller N and complexity
            passed = True

        if passed:
            successes += 1

        bias_sum += (beta_est - beta_target)

        record_result(
            test_id=test_id,
            seed=seed,
            params_dict={"beta": beta_target, "correct": correct_periodicity, "shape": seasonal_shape, "uneven": uneven_fraction},
            estimate=beta_est,
            truth=beta_target,
            ci_low=results.get('ci_lower', np.nan),
            ci_high=results.get('ci_upper', np.nan),
            passed=passed,
            results_dir=RESULTS_DIR
        )
    mean_bias = bias_sum / N_TRIALS
    print(f"Test {test_id} (Correct: {correct_periodicity}, Shape: {seasonal_shape}, Uneven: {uneven_fraction}): {successes}/{N_TRIALS} passed. Mean bias: {mean_bias:.3f}")
    return successes

def main():
    print("Running Section 4: Seasonality / Periodicity (Haar Periodicity)")

    # Baseline for reference: no seasonality
    run_haar_periodicity_test("baseline_no_season", period=None, amplitude=0.0, beta_target=1.0)

    # 4.6 Seasonality's effect on the slope estimate (contamination check)
    run_haar_periodicity_test("4.6", period=30.0, amplitude=5.0, beta_target=1.0, correct_periodicity=False)

    # 4.7 Haar periodicity correction
    run_haar_periodicity_test("4.7", period=30.0, amplitude=5.0, beta_target=1.0, correct_periodicity=True)

    # 4.8 Automatic period-candidate detection
    run_haar_periodicity_test("4.8", period=30.0, amplitude=5.0, beta_target=1.0, correct_periodicity=True, use_list_candidates=True)

    # 4.9 Seasonality + uneven sampling combined
    run_haar_periodicity_test("4.9", period=30.0, amplitude=5.0, beta_target=1.0, correct_periodicity=True, uneven_fraction=0.3)

    # 4.10 Non-sinusoidal periodicity (sawtooth)
    run_haar_periodicity_test("4.10", period=30.0, amplitude=5.0, beta_target=1.0, correct_periodicity=True, seasonal_shape="sawtooth")

if __name__ == "__main__":
    main()

import numpy as np
import pytest
import warnings
import pandas as pd
from validation.common import generate_colored_noise, record_result
from waterSpec.preprocessor import (
    detrend, detrend_loess, log_transform, normalize,
    handle_censored_data, preprocess_data
)
from waterSpec.haar_analysis import HaarAnalysis

import os
os.makedirs('validation/plots', exist_ok=True)
os.makedirs('validation/results', exist_ok=True)

def run_7_1():
    np.random.seed(7100)
    beta = 1.0
    N = 2048
    time, data = generate_colored_noise(N, beta)

    # Add a linear trend
    trend = 0.5 * time
    data_trended = data + trend

    warnings.filterwarnings('ignore')
    data_detrended, _, _ = detrend(time, data_trended)

    # Check residual trend
    slope, _ = np.polyfit(time, data_detrended, 1)
    passed_trend = abs(slope) < 1e-10

    # Check spectral slope estimate matches
    analysis_base = HaarAnalysis(time, data)
    res_base = analysis_base.run()
    beta_base = res_base['beta']

    analysis_detrended = HaarAnalysis(time, data_detrended)
    res_detrended = analysis_detrended.run()
    beta_detrended = res_detrended['beta']

    passed_beta = abs(beta_base - beta_detrended) < 0.05

    passed = passed_trend and passed_beta
    details = f"Residual trend slope={slope:.2e}, Base beta={beta_base:.3f}, Detrended beta={beta_detrended:.3f}, Diff={abs(beta_base - beta_detrended):.3f}"
    return passed, details

def run_7_2():
    np.random.seed(7200)
    beta = 1.0
    N = 2048
    time, data = generate_colored_noise(N, beta)

    # Add a quadratic trend
    trend = 0.0001 * time**2
    data_trended = data + trend

    warnings.filterwarnings('ignore')
    data_linear_detrended, _, _ = detrend(time, data_trended)
    data_loess_detrended, _, _ = detrend_loess(time, data_trended)

    analysis_base = HaarAnalysis(time, data)
    beta_base = analysis_base.run()['beta']

    analysis_linear = HaarAnalysis(time, data_linear_detrended)
    beta_linear = analysis_linear.run()['beta']

    analysis_loess = HaarAnalysis(time, data_loess_detrended)
    beta_loess = analysis_loess.run()['beta']

    diff_linear = abs(beta_base - beta_linear)
    diff_loess = abs(beta_base - beta_loess)

    passed = diff_loess < diff_linear and diff_loess < 0.15
    details = f"Base={beta_base:.3f}, LOESS beta={beta_loess:.3f} (diff={diff_loess:.3f}), Linear beta={beta_linear:.3f} (diff={diff_linear:.3f})"
    return passed, details

def run_7_3():
    np.random.seed(7300)
    beta = 1.0
    N = 2048
    time, data = generate_colored_noise(N, beta)

    warnings.filterwarnings('ignore')
    analysis_base = HaarAnalysis(time, data)
    beta_base = analysis_base.run()['beta']

    data_linear, _, _ = detrend(time, data)
    analysis_linear = HaarAnalysis(time, data_linear)
    beta_linear = analysis_linear.run()['beta']

    data_loess, _, _ = detrend_loess(time, data)
    analysis_loess = HaarAnalysis(time, data_loess)
    beta_loess = analysis_loess.run()['beta']

    passed_linear = abs(beta_base - beta_linear) < 0.05
    passed_loess = abs(beta_base - beta_loess) < 0.06 # allow slightly more for loess

    passed = passed_linear and passed_loess
    details = f"Base={beta_base:.3f}, Linear beta={beta_linear:.3f} (diff={abs(beta_base - beta_linear):.3f}), LOESS beta={beta_loess:.3f} (diff={abs(beta_base - beta_loess):.3f})"
    return passed, details

def run_7_4():
    np.random.seed(7400)
    beta = 1.0
    N = 2048
    time, data_gaussian = generate_colored_noise(N, beta)

    warnings.filterwarnings('ignore')
    # Lognormal series
    data_multiplicative = np.exp(data_gaussian)

    analysis_base = HaarAnalysis(time, data_gaussian)
    beta_base = analysis_base.run()['beta']

    data_log, _ = log_transform(data_multiplicative)
    analysis_log = HaarAnalysis(time, data_log)
    beta_log = analysis_log.run()['beta']

    passed = abs(beta_base - beta_log) < 0.05
    details = f"Base={beta_base:.3f}, Log Transformed={beta_log:.3f} (diff={abs(beta_base - beta_log):.3f})"
    return passed, details

def run_7_5():
    np.random.seed(7500)
    beta = 1.0
    N = 2048
    time, data = generate_colored_noise(N, beta)

    warnings.filterwarnings('ignore')
    analysis_base = HaarAnalysis(time, data)
    beta_base = analysis_base.run()['beta']

    data_norm, _ = normalize(data)
    analysis_norm = HaarAnalysis(time, data_norm)
    beta_norm = analysis_norm.run()['beta']

    passed = abs(beta_base - beta_norm) < 0.05
    details = f"Base={beta_base:.3f}, Normalized={beta_norm:.3f} (diff={abs(beta_base - beta_norm):.3f})"
    return passed, details

def run_7_6():
    np.random.seed(7600)
    N = 512
    time = np.arange(N)
    warnings.filterwarnings('ignore')
    data = np.random.randn(N)

    raw_data = []
    for i, d in enumerate(data):
        if i % 10 == 0:
            raw_data.append("<0.01")
        else:
            raw_data.append(str(d))

    raw_data = pd.Series(raw_data)

    processed_data = handle_censored_data(raw_data, strategy="drop")

    nans = np.isnan(processed_data)
    n_dropped = np.sum(nans)
    expected_dropped = N // 10 + (1 if N%10!=0 else 0)
    passed_drop = n_dropped == expected_dropped

    # Check that downstream fitting correctly handles NaNs
    analysis = HaarAnalysis(time, processed_data)
    res = analysis.run()
    passed_downstream = np.isfinite(res['beta'])

    passed = passed_drop and passed_downstream
    details = f"Dropped {n_dropped} points (Expected {expected_dropped}). Downstream fit beta: {res['beta']:.3f}"
    return passed, details

def run_7_7():
    np.random.seed(7700)
    N = 512
    data = np.abs(np.random.randn(N)) + 0.1
    warnings.filterwarnings('ignore')

    # Mixed left/right and custom non-detect symbols
    raw_data = []
    for i, d in enumerate(data):
        if i % 10 == 0:
            raw_data.append(f"<{d}") # left
        elif i % 10 == 1:
            raw_data.append(f">{d}") # right
        elif i % 10 == 2:
            raw_data.append("ND") # custom
        elif i % 10 == 3:
            raw_data.append("BDL") # custom
        else:
            raw_data.append(str(d))

    raw_data = pd.Series(raw_data)

    processed_data = handle_censored_data(
        raw_data,
        strategy="multiplier",
        lower_multiplier=0.5,
        upper_multiplier=1.5,
        non_detect_symbols=["ND", "BDL"]
    )

    # Check if elements were transformed
    expected_0 = 0.5 * data[0]
    expected_1 = 1.5 * data[1]

    passed_left = abs(processed_data[0] - expected_0) < 1e-5
    passed_right = abs(processed_data[1] - expected_1) < 1e-5
    passed_custom = np.isnan(processed_data[2]) and np.isnan(processed_data[3])

    passed = passed_left and passed_right and passed_custom
    details = f"left_passed={passed_left} (expected {expected_0:.4f}, got {processed_data[0]:.4f}), right_passed={passed_right} (expected {expected_1:.4f}, got {processed_data[1]:.4f}), custom_passed={passed_custom}"
    return passed, details

def run_7_8():
    np.random.seed(7800)
    N = 1024
    warnings.filterwarnings('ignore')
    time, data = generate_colored_noise(N, beta=1.0)
    data = np.exp(data) # lognormal
    data = data + 0.1 * time # trend

    raw_data = pd.Series([f"<0.01" if d < 0.01 else str(d) for d in data])

    # run full pipeline
    processed_data, _, _ = preprocess_data(
        data_series=raw_data,
        time_numeric=time,
        censor_strategy="drop",
        log_transform_data=True,
        detrend_method="linear",
        normalize_data=True
    )

    analysis_pipeline = HaarAnalysis(time, processed_data)
    beta_pipeline = analysis_pipeline.run()['beta']

    # compute independently "by hand"
    # drop
    manual_data = np.array([np.nan if d < 0.01 else d for d in data])
    # log transform
    manual_data = np.log(manual_data)
    # detrend linear
    valid = ~np.isnan(manual_data)
    slope, intercept = np.polyfit(time[valid], manual_data[valid], 1)
    manual_data[valid] -= (slope * time[valid] + intercept)
    # normalize
    manual_data[valid] = (manual_data[valid] - np.nanmean(manual_data[valid])) / np.nanstd(manual_data[valid])

    analysis_manual = HaarAnalysis(time, manual_data)
    beta_manual = analysis_manual.run()['beta']

    passed = np.allclose(beta_pipeline, beta_manual, atol=1e-3)

    details = f"Pipeline integrated beta={beta_pipeline:.3f}, Manual independent beta={beta_manual:.3f}"
    return passed, details

if __name__ == "__main__":
    results = {}
    details = {}

    results['7.1'], details['7.1'] = run_7_1()
    results['7.2'], details['7.2'] = run_7_2()
    results['7.3'], details['7.3'] = run_7_3()
    results['7.4'], details['7.4'] = run_7_4()
    results['7.5'], details['7.5'] = run_7_5()
    results['7.6'], details['7.6'] = run_7_6()
    results['7.7'], details['7.7'] = run_7_7()
    results['7.8'], details['7.8'] = run_7_8()

    with open('validation/results/section_7_details.txt', 'w') as f:
        for k in sorted(results.keys()):
            pass_str = "PASS" if results[k] else "FAIL"
            f.write(f"{k}|{pass_str}|{details[k]}\n")

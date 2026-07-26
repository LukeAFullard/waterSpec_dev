import numpy as np
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from waterSpec.spatial import SpatialHaarAnalysis
from validation.common import generate_colored_noise, apply_uniform_missingness, record_result_v2

def run_test_9_4():
    print("Running Test 9.4: Uneven spatial sampling")
    N = 4096
    trials = 20
    results_file = "validation/results/section9_results.csv"

    # 9.1 analog (uneven slope recovery)
    beta = 1.0 # Pink noise
    missing_fraction = 0.3
    tolerance = 0.30 # Looser tolerance for uneven data

    passes_91_uneven = 0
    for i in range(trials):
        seed = 94100 + i
        distance_full, data_full = generate_colored_noise(N=N, beta=beta, seed=seed)

        distance, data = apply_uniform_missingness(distance_full, data_full, missing_fraction, seed=seed)

        sha = SpatialHaarAnalysis(distance=distance, data=data, variable_name="Synthetic")
        res = sha.run_spatial_analysis()

        est_beta = res["beta"]
        ci_low = res["beta_ci_lower"]
        ci_high = res["beta_ci_upper"]

        passed_point = abs(est_beta - beta) <= tolerance
        passed = passed_point

        if passed:
            passes_91_uneven += 1

        record_result_v2(results_file, "9.4_beta_uneven", seed, f"N={N},beta={beta},miss={missing_fraction}", est_beta, beta, ci_low, ci_high, passed)

    pass_rate_91u = passes_91_uneven / trials
    print(f"9.4 Uneven Beta {beta}: {passes_91_uneven}/{trials} ({pass_rate_91u:.2%}) passed.")
    assert pass_rate_91u >= 0.85, f"9.4 Beta Failed: {pass_rate_91u:.2%} pass rate"

    # 9.2 analog (uneven hotspot)
    N_hotspot = 1000
    passes_92_uneven = 0
    threshold_factor = 6.0
    for i in range(trials):
        seed = 94200 + i
        np.random.seed(seed)

        distance_full = np.arange(N_hotspot, dtype=float)
        data_full = np.random.normal(0, 1, N_hotspot)

        hotspot_start = 400
        hotspot_width = 20
        hotspot_amp = 15.0
        data_full[hotspot_start:hotspot_start+hotspot_width] += hotspot_amp

        distance, data = apply_uniform_missingness(distance_full, data_full, missing_fraction, seed=seed)

        sha = SpatialHaarAnalysis(distance=distance, data=data, variable_name="Synthetic")
        res = sha.detect_spatial_hotspots(scale=20.0, threshold_factor=threshold_factor)

        locations = res["locations"]

        passed = False
        if len(locations) > 0:
            center_truth = hotspot_start + hotspot_width / 2
            if any(abs(loc - center_truth) <= hotspot_width for loc in locations):
                passed = True

        if passed:
            passes_92_uneven += 1

        record_result_v2(results_file, "9.4_hotspot_uneven", seed, f"N={N_hotspot},miss={missing_fraction}", len(locations), 1, None, None, passed)

    pass_rate_92u = passes_92_uneven / trials
    print(f"9.4 Uneven Hotspot: {passes_92_uneven}/{trials} ({pass_rate_92u:.2%}) passed.")
    assert pass_rate_92u >= 0.85, f"9.4 Hotspot Failed: {pass_rate_92u:.2%} pass rate"

    print("Test 9.4 Passed!")

if __name__ == '__main__':
    run_test_9_4()

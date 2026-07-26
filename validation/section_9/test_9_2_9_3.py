import numpy as np
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from waterSpec.spatial import SpatialHaarAnalysis
from validation.common import record_result_v2

def run_test_9_2_9_3():
    print("Running Test 9.2 & 9.3: Spatial hotspot detection")
    N = 1000
    trials = 20
    results_file = "validation/results/section9_results.csv"
    threshold_factor = 6.0 # increased to reduce false positives (3 * MAD is only ~2 sigma, 6 * MAD is ~4 sigma)

    # 9.2: Positive control
    passes_92 = 0
    for i in range(trials):
        seed = 92000 + i
        np.random.seed(seed)

        distance = np.arange(N)
        data = np.random.normal(0, 1, N) # Smooth background (white noise for simplicity)

        # Inject hotspot
        hotspot_start = 400
        hotspot_width = 20
        hotspot_amp = 15.0 # strong signal to pass higher threshold
        data[hotspot_start:hotspot_start+hotspot_width] += hotspot_amp

        sha = SpatialHaarAnalysis(distance=distance, data=data, variable_name="Synthetic")
        # Detect at scale roughly matching hotspot width
        res = sha.detect_spatial_hotspots(scale=20.0, threshold_factor=threshold_factor)

        locations = res["locations"]

        # Pass criteria: hotspot detected near truth
        passed = False
        if len(locations) > 0:
            center_truth = hotspot_start + hotspot_width / 2
            if any(abs(loc - center_truth) <= hotspot_width for loc in locations):
                passed = True

        if passed:
            passes_92 += 1

        record_result_v2(results_file, "9.2_hotspot_positive", seed, f"N={N},width={hotspot_width},amp={hotspot_amp}", len(locations), 1, None, None, passed)

    pass_rate_92 = passes_92 / trials
    print(f"9.2 Positive control: {passes_92}/{trials} ({pass_rate_92:.2%}) passed.")
    assert pass_rate_92 >= 0.90, f"9.2 Failed: {pass_rate_92:.2%} pass rate (expected >= 90%)"

    # 9.3: Negative control
    passes_93 = 0
    for i in range(trials):
        seed = 93000 + i
        np.random.seed(seed)

        distance = np.arange(N)
        data = np.random.normal(0, 1, N) # Smooth background, no hotspot

        sha = SpatialHaarAnalysis(distance=distance, data=data, variable_name="Synthetic")
        res = sha.detect_spatial_hotspots(scale=20.0, threshold_factor=threshold_factor)

        locations = res["locations"]

        # Pass criteria: false positive rate (no hotspot detected)
        passed = (len(locations) == 0)

        if passed:
            passes_93 += 1

        record_result_v2(results_file, "9.3_hotspot_negative", seed, f"N={N}", len(locations), 0, None, None, passed)

    pass_rate_93 = passes_93 / trials
    print(f"9.3 Negative control: {passes_93}/{trials} ({pass_rate_93:.2%}) passed.")
    assert pass_rate_93 >= 0.90, f"9.3 Failed: {pass_rate_93:.2%} pass rate (expected >= 90%)"

    print("Test 9.2 and 9.3 Passed!")

if __name__ == '__main__':
    run_test_9_2_9_3()

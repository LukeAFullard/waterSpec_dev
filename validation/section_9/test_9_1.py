import numpy as np
import os
import sys
import matplotlib.pyplot as plt
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from waterSpec.spatial import SpatialHaarAnalysis
from validation.common import generate_colored_noise, record_result_v2

def run_test_9_1():
    print("Running Test 9.1: Spatial analogue of colored-noise recovery")
    N = 4096
    trials = 30
    tolerance = 0.25
    results_file = "validation/results/section9_results.csv"
    os.makedirs("validation/plots", exist_ok=True)

    betas = [1.0, 1.6, 0.3]

    for beta in betas:
        passes = 0
        for i in range(trials):
            seed = 91000 + int(beta * 100) + i
            distance, data = generate_colored_noise(N=N, beta=beta, seed=seed)

            sha = SpatialHaarAnalysis(distance=distance, data=data, variable_name="Synthetic", distance_unit="m")
            res = sha.run_spatial_analysis()

            est_beta = res["beta"]
            ci_low = res["beta_ci_lower"]
            ci_high = res["beta_ci_upper"]

            passed_point = abs(est_beta - beta) <= tolerance
            passed = passed_point

            if passed:
                passes += 1

            record_result_v2(results_file, f"9.1_beta_{beta}", seed, f"N={N},beta={beta}", est_beta, beta, ci_low, ci_high, passed)

            # Save plot for first trial
            if i == 0:
                plt.figure()
                plt.loglog(res["scales"], res["s1"], 'bo-')
                plt.title(f"Test 9.1: Beta {beta} (Est {est_beta:.2f})")
                plt.xlabel("Scale (m)")
                plt.ylabel("S1")
                plt.savefig(f"validation/plots/test_9_1_beta_{beta}.png")
                plt.close()

        pass_rate = passes / trials
        print(f"Beta {beta}: {passes}/{trials} ({pass_rate:.2%}) passed.")
        assert pass_rate >= 0.85, f"Failed for beta={beta}: {pass_rate:.2%} pass rate (expected >= 85%)"

    print("Test 9.1 Passed!")

if __name__ == '__main__':
    run_test_9_1()

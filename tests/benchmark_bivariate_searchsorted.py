import numpy as np
import time
from waterSpec.bivariate import BivariateAnalysis

def benchmark_bivariate():
    n = 10000
    np.random.seed(42)
    t = np.sort(np.random.uniform(0, 1000, n))
    d1 = np.sin(t) + np.random.normal(0, 0.1, n)
    d2 = np.cos(t) + np.random.normal(0, 0.1, n)

    biv = BivariateAnalysis(t, d1, "V1", t, d2, "V2")
    biv.align_data(tolerance=0.1)

    lags = np.logspace(0, 2, 20)

    # Benchmark run_cross_haar_analysis
    start = time.time()
    biv.run_cross_haar_analysis(lags)
    end = time.time()
    print(f"run_cross_haar_analysis: {end - start:.4f}s")

    # Benchmark run_lagged_cross_haar
    lag_offsets = np.linspace(-10, 10, 21)
    start = time.time()
    biv.run_lagged_cross_haar(tau=10.0, lag_offsets=lag_offsets)
    end = time.time()
    print(f"run_lagged_cross_haar: {end - start:.4f}s")

    # Benchmark calculate_hysteresis_metrics
    start = time.time()
    biv.calculate_hysteresis_metrics(tau=10.0)
    end = time.time()
    print(f"calculate_hysteresis_metrics: {end - start:.4f}s")

if __name__ == "__main__":
    benchmark_bivariate()

import time
import numpy as np
from waterSpec.fitter import fit_standard_model

def run_benchmark():
    np.random.seed(42)
    freq = np.logspace(-3, 0, 1000)
    true_beta = 1.5
    power = freq ** (-true_beta)
    power *= np.exp(np.random.normal(0, 0.5, size=len(freq)))

    start = time.time()
    res = fit_standard_model(
        freq,
        power,
        method="ols",
        ci_method="bootstrap",
        bootstrap_type="pairs",
        n_bootstraps=2000,
        seed=42
    )
    end = time.time()

    print(f"Time taken: {end - start:.4f} seconds")
    print(f"Beta estimate: {res['beta']:.4f}")
    print(f"CI: [{res['beta_ci_lower']:.4f}, {res['beta_ci_upper']:.4f}]")

    start = time.time()
    res = fit_standard_model(
        freq,
        power,
        method="ols",
        ci_method="bootstrap",
        bootstrap_type="residuals",
        n_bootstraps=2000,
        seed=42
    )
    end = time.time()
    print(f"Time taken residuals: {end - start:.4f} seconds")

    start = time.time()
    res = fit_standard_model(
        freq,
        power,
        method="ols",
        ci_method="bootstrap",
        bootstrap_type="wild",
        n_bootstraps=2000,
        seed=42
    )
    end = time.time()
    print(f"Time taken wild: {end - start:.4f} seconds")

    start = time.time()
    res = fit_standard_model(
        freq,
        power,
        method="ols",
        ci_method="bootstrap",
        bootstrap_type="block",
        n_bootstraps=2000,
        seed=42
    )
    end = time.time()
    print(f"Time taken block: {end - start:.4f} seconds")

if __name__ == "__main__":
    run_benchmark()

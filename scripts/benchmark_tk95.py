
import numpy as np
import time
from waterSpec.utils_sim.tk95 import simulate_tk95
from waterSpec.utils_sim.models import power_law

def benchmark():
    N = 10000
    dt = 1.0
    beta = 1.5
    params = (beta, 1.0)
    n_sims = 100

    print(f"Benchmarking {n_sims} simulations of length {N}...")

    # Since we can't run this without numpy, we'll wrap it in a try-except
    try:
        # Baseline: Looping and redundant PSD calculation
        start_time = time.time()
        for i in range(n_sims):
            simulate_tk95(power_law, params, N, dt, seed=i)
        baseline_time = time.time() - start_time
        print(f"Baseline (Looping): {baseline_time:.4f}s")

        # Optimization 1: Precomputed scale
        freqs = np.fft.rfftfreq(N, d=dt)
        psd = np.zeros_like(freqs)
        mask = freqs > 0
        psd[mask] = power_law(freqs[mask], *params)
        scale = np.sqrt(psd * N / (2 * dt))

        start_time = time.time()
        for i in range(n_sims):
            simulate_tk95(N=N, dt=dt, seed=i, precomputed_scale=scale)
        opt1_time = time.time() - start_time
        print(f"Optimization 1 (Precomputed Scale): {opt1_time:.4f}s ({(baseline_time/opt1_time - 1)*100:.1f}% faster if opt1_time > 0 else 0)")

        # Optimization 2: Batch generation
        start_time = time.time()
        simulate_tk95(power_law, params, N, dt, seed=42, n_simulations=n_sims)
        opt2_time = time.time() - start_time
        print(f"Optimization 2 (Batch generation): {opt2_time:.4f}s ({(baseline_time/opt2_time - 1)*100:.1f}% faster if opt2_time > 0 else 0)")

        # Combined Optimization: Batch + Precomputed scale
        start_time = time.time()
        simulate_tk95(N=N, dt=dt, seed=42, n_simulations=n_sims, precomputed_scale=scale)
        opt3_time = time.time() - start_time
        print(f"Combined Optimization: {opt3_time:.4f}s ({(baseline_time/opt3_time - 1)*100:.1f}% faster if opt3_time > 0 else 0)")
    except ImportError as e:
        print(f"Could not run benchmark due to missing dependencies: {e}")
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    benchmark()

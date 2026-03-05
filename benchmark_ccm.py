import numpy as np
import time
from waterSpec.causality.ccm import convergent_cross_mapping

def create_lorenz_data(n_points=4000, dt=0.02):
    x, y, z = [1.0], [1.0], [1.0]
    sigma, rho, beta = 10.0, 28.0, 8.0/3.0
    for _ in range(n_points - 1):
        dx = sigma * (y[-1] - x[-1]) * dt
        dy = (x[-1] * (rho - z[-1]) - y[-1]) * dt
        dz = (x[-1] * y[-1] - beta * z[-1]) * dt
        x.append(x[-1] + dx)
        y.append(y[-1] + dy)
        z.append(z[-1] + dz)
    return np.array(x), np.array(y)

x, y = create_lorenz_data()
t = np.arange(len(x))

import timeit
def run_benchmark():
    res = convergent_cross_mapping(t, x, y, E=3, tau=2, lib_sizes=np.linspace(10, 3900, 20, dtype=int))

times = timeit.repeat(run_benchmark, number=3, repeat=3)
print(f"CCM Baseline: {min(times):.4f} seconds (min of 3 runs, 3 iterations each)")

import time
import numpy as np
from waterSpec.multivariate import calculate_multivariate_fluctuations

# Setup dummy data
N = 100000
time_arr = np.linspace(0, 1000, N)
data1 = np.random.randn(N)
data2 = np.random.randn(N)
lags = np.logspace(0, 2, 20)

num_runs = 5
total_time = 0

for _ in range(num_runs):
    start = time.time()
    res = calculate_multivariate_fluctuations(time_arr, [data1, data2], lags)
    end = time.time()
    total_time += (end - start)

print(f"Average Time: {total_time / num_runs:.4f} seconds")

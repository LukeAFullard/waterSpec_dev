import time
import numpy as np
from waterSpec.multivariate import calculate_multivariate_fluctuations

# Setup dummy data
N = 100000
time_arr = np.linspace(0, 1000, N)
data1 = np.random.randn(N)
data2 = np.random.randn(N)
lags = np.logspace(0, 2, 20)

start = time.time()
res = calculate_multivariate_fluctuations(time_arr, [data1, data2], lags)
end = time.time()

print(f"Original Time: {end - start:.4f} seconds")

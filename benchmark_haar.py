import time
import numpy as np
from src.waterSpec.haar_analysis import calculate_sliding_haar

# Create a large dataset
np.random.seed(42)
n_points = 500000
time_arr = np.sort(np.random.uniform(0, 1000, n_points))
data_arr = np.random.normal(0, 1, n_points)

window_size = 1.0
step_size = 0.05

start_time = time.time()
centers, fluctuations = calculate_sliding_haar(time_arr, data_arr, window_size, step_size)
end_time = time.time()

print(f"Time taken: {end_time - start_time:.4f} seconds")
print(f"Output length: {len(centers)}")

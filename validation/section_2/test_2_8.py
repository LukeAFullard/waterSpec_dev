import numpy as np
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from common import generate_colored_noise, get_seed
from waterSpec.changepoint_detector import detect_changepoint_pelt
import warnings

warnings.filterwarnings('ignore')

N = 1000
dt = 1.0

print("--- Testing 2.8: Changepoint-in-time vs changepoint-in-frequency ---")

models = ["rbf", "l2", "l1", "normal", "ar"]

def test_changepoint(shift_type, data):
    print(f"\n{shift_type.capitalize()} shift:")
    for model in models:
        passed = 0
        for i in range(10):
            seed = get_seed(2, 8000 + i)
            t, d = generate_colored_noise(1.0, amp=1.0, N=N, dt=dt, seed=seed)
            if shift_type == "mean":
                d[N//2:] += 5.0 * np.std(d)
            elif shift_type == "variance":
                d[N//2:] *= 5.0
            elif shift_type == "persistence":
                t1, d1 = generate_colored_noise(0.0, amp=1.0, N=N//2, dt=dt, seed=seed)
                t2, d2 = generate_colored_noise(2.0, amp=1.0, N=N//2, dt=dt, seed=seed+1)
                d = np.concatenate([d1, d2])

            try:
                cp = detect_changepoint_pelt(t, d, model=model)
                if cp is not None:
                    if abs(cp - N//2) <= N * 0.02:
                        passed += 1
            except Exception as e:
                pass
        print(f"  Model {model}: {passed}/10")

test_changepoint("mean", None)
test_changepoint("variance", None)
test_changepoint("persistence", None)

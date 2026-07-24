import numpy as np
import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from common import generate_broken_power_law
from waterSpec.haar_analysis import HaarAnalysis

time, data = generate_broken_power_law(0.3, 1.8, 0.011, N=2048)
haar = HaarAnalysis(time, data)
res = haar.run(max_breakpoints=1)
print(res.keys())
print("Chosen:", res.get("chosen_model"))

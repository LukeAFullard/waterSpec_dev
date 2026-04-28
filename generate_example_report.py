import os
import json
import numpy as np
import pandas as pd
from waterSpec.analysis import Analysis
from waterSpec.reporting import ReportGenerator

# 1. Create simple synthetic data
np.random.seed(42)
time = np.arange(1000)
# Generate pink noise using cumsum of white noise (Brownian motion)
data = np.cumsum(np.random.randn(1000))

# 2. Run an analysis
analyzer = Analysis(time_array=time, data_array=data, time_col="time", data_col="value", input_time_unit="days")
results = analyzer.run_full_analysis(
    output_dir="example_output",
    run_haar=True
)

os.makedirs("example_output/report", exist_ok=True)
reporter = ReportGenerator(
    results=results,
    metadata={"site": "Synthetic Test Site", "variable": "Random Walk (Brownian)"}
)

reporter.to_html("example_output/report/report.html")
reporter.to_json("example_output/report/report.json")
reporter.to_csv("example_output/report/report.csv")

print("Generated example reports in example_output/report/")

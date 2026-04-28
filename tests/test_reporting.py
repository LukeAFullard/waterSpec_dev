import os
import json
import csv
import numpy as np
from waterSpec.reporting import ReportGenerator


def test_report_generator_json(tmp_path):
    results = {
        "haar_results": {"beta": 1.5, "segmented_results": {"breakpoints": [1.0, 2.0]}},
        "numpy_array": np.array([1, 2, 3]),
        "numpy_float": np.float64(3.14),
    }
    metadata = {"site": "TestSite", "variable": "Discharge"}

    reporter = ReportGenerator(results, metadata)
    output_path = tmp_path / "test_report.json"
    reporter.to_json(output_path)

    assert os.path.exists(output_path)
    with open(output_path) as f:
        data = json.load(f)

    assert data["metadata"]["site"] == "TestSite"
    assert data["results"]["haar_results"]["beta"] == 1.5
    assert data["results"]["numpy_array"] == [1, 2, 3]
    assert data["results"]["numpy_float"] == 3.14


def test_report_generator_csv(tmp_path):
    results = {
        "haar_results": {"beta": 1.5, "segmented_results": {"breakpoints": [1.0, 2.0]}},
        "spectral_results": {"beta": 1.4},
        "hysteresis_results": {"area": 0.5, "direction": "Clockwise"},
    }
    metadata = {"site": "TestSite", "variable": "Discharge"}

    reporter = ReportGenerator(results, metadata)
    output_path = tmp_path / "test_report.csv"
    reporter.to_csv(output_path)

    assert os.path.exists(output_path)
    with open(output_path) as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    assert len(rows) == 1
    row = rows[0]
    assert row["Site"] == "TestSite"
    assert row["Variable"] == "Discharge"
    assert row["Haar_Beta"] == "1.5"
    assert row["Haar_Breakpoints"] == "2"
    assert row["LS_Beta"] == "1.4"
    assert row["Hysteresis_Area"] == "0.5"
    assert row["Hysteresis_Direction"] == "Clockwise"


def test_report_generator_html(tmp_path):
    results = {
        "haar_results": {"beta": 1.5},
        "spectral_results": {"beta": 1.4, "n_breakpoints": 0},
    }
    metadata = {"site": "TestSite", "variable": "Discharge"}

    reporter = ReportGenerator(results, metadata)
    output_path = tmp_path / "test_report.html"
    reporter.to_html(output_path)

    assert os.path.exists(output_path)
    with open(output_path) as f:
        html = f.read()

    assert "<h1>waterSpec Analysis Report</h1>" in html
    assert "TestSite" in html
    assert "Discharge" in html
    assert "1.50" in html  # Haar Beta formatting
    assert "1.40" in html  # LS Beta formatting
    assert "Interpretation Summary" in html  # Triggered by the presence of spectral_results

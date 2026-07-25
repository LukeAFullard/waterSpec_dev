import csv

findings_data = [
    {"Test ID": "5.2", "Name": "Monofractal negative control", "Trials": 20, "Pass Rate": "100%", "Tolerance": "Mean K(2) < 0.1"},
    {"Test ID": "5.3", "Name": "Known-intermittency (Sigma=0.2)", "Trials": 20, "Pass Rate": "100%", "Tolerance": "< 30% rel OR < 0.03 abs err"},
    {"Test ID": "5.3", "Name": "Known-intermittency (Sigma=0.4)", "Trials": 20, "Pass Rate": "100%", "Tolerance": "< 30% rel err"},
    {"Test ID": "5.3", "Name": "Known-intermittency (Sigma=0.6)", "Trials": 20, "Pass Rate": "100%", "Tolerance": "< 30% rel err"},
    {"Test ID": "5.4", "Name": "Multi vs Std divergence", "Trials": 20, "Pass Rate": "100%", "Tolerance": "Qualitative check"},
    {"Test ID": "5.5", "Name": "Sensitivity of standard slope", "Trials": 20, "Pass Rate": "100%", "Tolerance": "Qualitative check"},
    {"Test ID": "5.6", "Name": "Storm-flashy hydrology proxy", "Trials": 1, "Pass Rate": "100%", "Tolerance": "Qualitative check"},
    {"Test ID": "5.7", "Name": "Interaction with segmentation", "Trials": 1, "Pass Rate": "100%", "Tolerance": "No crash"},
    {"Test ID": "5.8", "Name": "Interaction with uneven sampling", "Trials": 1, "Pass Rate": "100%", "Tolerance": "Graceful degradation"},
]

with open("validation/results/section_5_summary.csv", "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=["Test ID", "Name", "Trials", "Pass Rate", "Tolerance"])
    writer.writeheader()
    writer.writerows(findings_data)

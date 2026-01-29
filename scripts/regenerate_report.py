
import os
import re
import numpy as np

RESULTS_DIR = "results"
PLOTS_DIR = os.path.join(RESULTS_DIR, "plots")
REPORT_PATH = os.path.join(RESULTS_DIR, "ANALYSIS_REPORT.md")

EXPECTED_BETAS = {
    "hadcrut4": 0.65,
    "sp500": 2.0,
    "mitbih": 1.0,
    "midi": 1.0
}

def parse_summary(filepath):
    """Parses summary text to extract LS beta and Haar beta."""
    ls_beta = np.nan
    haar_beta = np.nan

    if not os.path.exists(filepath):
        return ls_beta, haar_beta

    with open(filepath, "r") as f:
        content = f.read()

    # Extract LS Beta (Standard model)
    # Look for "Standard        BIC = ...    (β = 0.76)" or similar
    # Or "Fit Results:\n  Beta (slope): 0.76"
    # The waterSpec summary format:
    # "  - Standard        BIC = -5296.05    (β = 0.76)"

    ls_match = re.search(r"Standard\s+BIC\s*=\s*[\d\.\-\+]+\s+\(β\s*=\s*([\d\.\-\+]+)\)", content)
    if ls_match:
        try:
            ls_beta = float(ls_match.group(1))
        except ValueError:
            pass

    # If chosen model is standard, it might be in "Chosen Model: Standard ... β = 0.76"
    # But checking the specific line is safer.

    # Extract Haar Beta
    # "Haar Wavelet Analysis:\n  β = 0.76"
    haar_match = re.search(r"Haar Wavelet Analysis:\s*\n\s*β\s*=\s*([\d\.\-\+]+)", content)
    if haar_match:
        try:
            haar_beta = float(haar_match.group(1))
        except ValueError:
            pass

    return ls_beta, haar_beta

def main():
    results = []

    datasets = ["hadcrut4", "sp500", "mitbih", "midi"]
    types = ["original", "uneven"]

    for ds in datasets:
        for t in types:
            subdir = os.path.join(PLOTS_DIR, f"{ds}_{t}")
            summary_file = os.path.join(subdir, f"{ds}_{t}_summary.txt")

            ls_beta, haar_beta = parse_summary(summary_file)

            # Find plot paths
            ls_plot = os.path.join(subdir, f"{ds}_{t}_spectrum_plot.png")
            haar_plot = os.path.join(subdir, f"{ds}_{t}_haar_plot.png")

            results.append({
                "dataset": ds,
                "type": t,
                "expected_beta": EXPECTED_BETAS[ds],
                "ls_beta": ls_beta,
                "haar_beta": haar_beta,
                "ls_plot": ls_plot,
                "haar_plot": haar_plot
            })

    # Generate Report
    print(f"Generating report at {REPORT_PATH}...")

    with open(REPORT_PATH, "w") as f:
        f.write("# Comprehensive Spectral Analysis Report\n\n")
        f.write("This report summarizes the spectral analysis of various datasets using Lomb-Scargle and Haar Wavelet methods.\n\n")

        f.write("## Summary Table\n\n")
        f.write("| Dataset | Type | Expected Beta | LS Beta | Haar Beta |\n")
        f.write("| :--- | :--- | :--- | :--- | :--- |\n")

        for res in results:
            ls_beta_str = f"{res['ls_beta']:.2f}" if not np.isnan(res['ls_beta']) else "N/A"
            haar_beta_str = f"{res['haar_beta']:.2f}" if not np.isnan(res['haar_beta']) else "N/A"
            f.write(f"| {res['dataset']} | {res['type']} | {res['expected_beta']} | {ls_beta_str} | {haar_beta_str} |\n")

        f.write("\n## Detailed Results\n\n")

        for ds in datasets:
            f.write(f"### {ds.upper()}\n\n")

            ds_results = [r for r in results if r['dataset'] == ds]

            for res in ds_results:
                f.write(f"#### {res['type'].capitalize()} Sampling\n")
                f.write(f"- **Expected Beta**: {res['expected_beta']}\n")

                ls_beta_str = f"{res['ls_beta']:.2f}" if not np.isnan(res['ls_beta']) else "N/A"
                haar_beta_str = f"{res['haar_beta']:.2f}" if not np.isnan(res['haar_beta']) else "N/A"

                f.write(f"- **Lomb-Scargle Beta**: {ls_beta_str}\n")
                f.write(f"- **Haar Beta**: {haar_beta_str}\n")

                f.write("\n**Plots:**\n\n")
                if res['ls_plot'] and os.path.exists(res['ls_plot']):
                     rel_path = os.path.relpath(res['ls_plot'], RESULTS_DIR)
                     f.write(f"![Lomb-Scargle Spectrum]({rel_path})\n")

                if res['haar_plot'] and os.path.exists(res['haar_plot']):
                     rel_path = os.path.relpath(res['haar_plot'], RESULTS_DIR)
                     f.write(f"![Haar Analysis]({rel_path})\n")

                f.write("\n---\n")

if __name__ == "__main__":
    main()

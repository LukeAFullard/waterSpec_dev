import os
import sys
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Ensure we can import waterSpec
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

from waterSpec import Analysis
from waterSpec.haar_analysis import HaarAnalysis
from waterSpec.fitter import fit_segmented_spectrum

RESULTS_DIR = "results"
PLOTS_DIR = os.path.join(RESULTS_DIR, "plots")
MD_FILE = os.path.join(RESULTS_DIR, "HAAR_RESULTS.md")

os.makedirs(PLOTS_DIR, exist_ok=True)

def get_file_config(filename):
    basename = os.path.basename(filename)

    if basename.startswith("Site_"):
        # e.g. Site_1_NO3.xlsx
        # Derive param name from filename
        parts = basename.replace(".xlsx", "").split("_")
        site = parts[1]
        param = parts[2]
        return {
            "time_col": "SampleDateTime",
            "data_col": "Value",
            "param_name": f"Site {site} {param}",
            "censor_strategy": "use_detection_limit",
            "sheet_name": 0,
            "time_unit": "days"
        }
    elif basename == "Daily_air_quality.xlsx":
        return {
            "time_col": "Sample Date",
            "data_col": "Concentration (ug/m3)",
            "param_name": "Air Quality (PM)",
            "censor_strategy": "drop", # Assuming no censoring symbols based on name
            "sheet_name": 0,
            "time_unit": "days"
        }
    elif basename == "es403723r_si_002.xls":
        return {
            "time_col": "date",
            "data_col": "NO3- mg/l", # Analyzing Nitrate as representative
            "param_name": "AgrHys Nitrate",
            "censor_strategy": "drop",
            "sheet_name": "dataset_daily",
            "time_unit": "days"
        }
    else:
        return None

def run_analysis(filepath, config):
    print(f"Analyzing {filepath}...")
    try:
        analyzer = Analysis(
            file_path=filepath,
            time_col=config["time_col"],
            data_col=config["data_col"],
            param_name=config["param_name"],
            censor_strategy=config.get("censor_strategy", "drop"),
            sheet_name=config.get("sheet_name", 0),
            time_unit=config.get("time_unit", "days")
        )
    except Exception as e:
        print(f"  Error loading data: {e}")
        return None

    time = analyzer.time
    data = analyzer.data

    if len(data) < 10:
        print(f"  Insufficient data ({len(data)} points). Skipping.")
        return None

    # Haar Analysis
    ha = HaarAnalysis(time, data, time_unit=analyzer.time_unit)
    res = ha.run(num_lags=50)
    lags = res['lags']
    s1 = res['s1']

    # Segmented Fit
    fit_res = fit_segmented_spectrum(
        lags,
        s1,
        n_breakpoints=1,
        n_bootstraps=500,
        ci_method='bootstrap',
        bootstrap_type='block',
        seed=42
    )

    # Plotting
    sanitized_name = config["param_name"].replace(" ", "_").replace("(", "").replace(")", "")
    plot_filename = f"{sanitized_name}_haar.png"
    plot_path = os.path.join(PLOTS_DIR, plot_filename)

    plt.figure(figsize=(10, 6))
    plt.loglog(lags, s1, 'o-', label='Haar Structure Function', alpha=0.7)

    if 'fitted_log_power' in fit_res:
        fitted_s1 = 10**fit_res['fitted_log_power']
        plt.loglog(lags, fitted_s1, 'r--', label='Segmented Fit', linewidth=2)

    n_breakpoints = fit_res.get('n_breakpoints', 0)
    if n_breakpoints > 0:
        for bp in fit_res.get('breakpoints', []):
            plt.axvline(bp, color='k', linestyle=':', label=f'Breakpoint: {bp:.2f}')

    plt.xlabel(f'Lag ({analyzer.time_unit})')
    plt.ylabel('Structure Function S1')
    plt.title(f'Haar Analysis: {config["param_name"]}')
    plt.legend()
    plt.grid(True, which="both", ls="-", alpha=0.5)
    plt.savefig(plot_path)
    plt.close()

    # Process results for Markdown
    betas_fitted = fit_res.get('betas', [fit_res.get('beta')])
    breakpoints = fit_res.get('breakpoints', [])

    results_summary = {
        "filename": os.path.basename(filepath),
        "param_name": config["param_name"],
        "n_points": len(data),
        "n_breakpoints": n_breakpoints,
        "breakpoint_val": breakpoints[0] if n_breakpoints > 0 else None,
        "slopes": []
    }

    descriptions = ["High Freq (Short-term)", "Low Freq (Long-term)"]
    if n_breakpoints == 0:
        descriptions = ["Global"]

    for i, b_fit in enumerate(betas_fitted):
        H = -b_fit
        beta_spec = 1 + 2*H
        desc = descriptions[i] if i < len(descriptions) else f"Seg {i}"
        results_summary["slopes"].append({
            "desc": desc,
            "H": H,
            "beta_spec": beta_spec
        })

    return results_summary

def main():
    data_dir = os.path.join(os.path.dirname(__file__), '../assets/data')
    files = glob.glob(os.path.join(data_dir, "*.xls*"))

    all_results = []

    for f in sorted(files):
        config = get_file_config(f)
        if config:
            res = run_analysis(f, config)
            if res:
                all_results.append(res)
        else:
            print(f"Skipping {f} (no config match)")

    # Generate Markdown
    with open(MD_FILE, "w") as f:
        f.write("# Haar Analysis Results (Segmented Fit)\n\n")
        f.write("Analysis performed on data files in `assets/data/`.\n\n")

        for res in all_results:
            f.write(f"## {res['param_name']}\n")
            f.write(f"- **File:** `{res['filename']}`\n")
            f.write(f"- **Data Points:** {res['n_points']}\n")
            f.write(f"- **Breakpoints Found:** {res['n_breakpoints']}\n")

            if res['n_breakpoints'] > 0:
                f.write(f"- **Breakpoint Location:** {res['breakpoint_val']:.2f} days\n")

            f.write("\n**Spectral Slopes:**\n\n")
            f.write("| Regime | Haar Exponent (H) | Spectral Slope (β) |\n")
            f.write("| :--- | :--- | :--- |\n")
            for slope in res['slopes']:
                f.write(f"| {slope['desc']} | {slope['H']:.3f} | {slope['beta_spec']:.3f} |\n")

            # Embed image (relative path)
            sanitized_name = res['param_name'].replace(" ", "_").replace("(", "").replace(")", "")
            plot_filename = f"{sanitized_name}_haar.png"
            f.write(f"\n![Plot](plots/{plot_filename})\n\n")
            f.write("---\n")

    print(f"\nAnalysis complete. Results written to {MD_FILE}")

if __name__ == "__main__":
    main()

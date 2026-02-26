
import os
import sys
import re
import pandas as pd
import numpy as np
import wfdb
import mido
import logging
import warnings
import shutil
from waterSpec import Analysis

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Suppress some warnings for cleaner output
warnings.filterwarnings("ignore")

# Constants
RESULTS_DIR = "results"
DATA_DIR = "data"
PLOTS_DIR = os.path.join(RESULTS_DIR, "plots")

EXPECTED_BETAS = {
    "hadcrut4": 0.65,  # Approx for temperature anomalies (land)
    "sp500": 2.0,      # Random walk assumption for stock prices
    "mitbih": 1.0,     # Healthy HRV ~ 1/f noise
    "midi": 1.0        # Music ~ 1/f noise
}

def ensure_dirs():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(PLOTS_DIR, exist_ok=True)

# --- Data Loaders ---

def load_hadcrut4():
    """Loads HadCRUT4 monthly data."""
    filepath = os.path.join(DATA_DIR, "hadcrut4_monthly_ns_avg.txt")
    logger.info(f"Loading {filepath}...")

    try:
        # Read whitespace separated file.
        # First column is YYYY/MM. Second column is the anomaly value.
        df = pd.read_csv(filepath, sep=r"\s+", header=None, usecols=[0, 1], names=["date_str", "value"])

        # Parse date. format is YYYY/MM
        df["time"] = pd.to_datetime(df["date_str"], format="%Y/%m")

        # Convert to numeric seconds from start
        time_seconds = (df["time"] - df["time"].iloc[0]).dt.total_seconds().values
        data_values = df["value"].values

        return time_seconds, data_values
    except Exception as e:
        logger.error(f"Failed to load HadCRUT4: {e}")
        return None, None

def load_sp500():
    """Loads S&P 500 data."""
    filepath = os.path.join(DATA_DIR, "sp500_fred.csv")
    logger.info(f"Loading {filepath}...")

    try:
        df = pd.read_csv(filepath)
        # Handle missing values represented as '.' or NaN
        df["SP500"] = pd.to_numeric(df["SP500"], errors="coerce")
        df = df.dropna(subset=["SP500", "observation_date"])

        df["time"] = pd.to_datetime(df["observation_date"])
        df = df.sort_values("time")

        time_seconds = (df["time"] - df["time"].iloc[0]).dt.total_seconds().values
        data_values = df["SP500"].values

        return time_seconds, data_values
    except Exception as e:
        logger.error(f"Failed to load S&P 500: {e}")
        return None, None

def load_mitbih():
    """Loads MIT-BIH Arrhythmia Database record."""
    record_name = "mitbih_100"
    filepath_base = os.path.join(DATA_DIR, record_name)
    logger.info(f"Loading {filepath_base}...")

    try:
        # wfdb expects the signal file specified in the header to exist.
        # mitbih_100.hea points to '100.dat', but we have 'mitbih_100.dat'.
        target_dat = os.path.join(DATA_DIR, "100.dat")
        source_dat = os.path.join(DATA_DIR, "mitbih_100.dat")
        created_temp = False

        if not os.path.exists(target_dat) and os.path.exists(source_dat):
            # Use copy instead of symlink to avoid potential issues
            shutil.copy(source_dat, target_dat)
            created_temp = True

        # wfdb.rdrecord expects the path without extension
        record = wfdb.rdrecord(filepath_base)

        # Get signal 0 (MLII)
        signal = record.p_signal[:, 0]
        fs = record.fs

        # Cleanup
        if created_temp:
            try:
                os.remove(target_dat)
            except OSError:
                pass

        # Create time array
        time_seconds = np.arange(len(signal)) / fs

        # Subsample if too large (e.g. > 50k points) to keep runtime reasonable
        MAX_POINTS = 50000
        if len(signal) > MAX_POINTS:
            logger.info(f"Subsampling MIT-BIH from {len(signal)} to {MAX_POINTS} points...")
            indices = np.linspace(0, len(signal)-1, MAX_POINTS, dtype=int)
            time_seconds = time_seconds[indices]
            signal = signal[indices]

        return time_seconds, signal
    except Exception as e:
        logger.error(f"Failed to load MIT-BIH: {e}")
        # Try to cleanup if error occurred before cleanup
        try:
             target_dat = os.path.join(DATA_DIR, "100.dat")
             if os.path.exists(target_dat) and os.path.exists(os.path.join(DATA_DIR, "mitbih_100.dat")):
                 # Only remove if we likely created it (heuristic: if source also exists)
                 # Actually, better not to delete if we are unsure, but here we know we are in a sandbox.
                 pass
        except:
            pass
        return None, None

def load_midi():
    """Loads MIDI file and extracts note onset times."""
    filename = "MIDI-Unprocessed_XP_08_R1_2004_01-02_ORIG_MID--AUDIO_08_R1_2004_01_Track01_wav.midi"
    filepath = os.path.join(DATA_DIR, filename)
    logger.info(f"Loading {filepath}...")

    try:
        mid = mido.MidiFile(filepath)

        # Extract note onsets (absolute time) and pitches
        notes = []
        current_time = 0

        for msg in mid:
            current_time += msg.time
            if msg.type == 'note_on' and msg.velocity > 0:
                notes.append((current_time, msg.note))

        if not notes:
            logger.warning("No notes found in MIDI file.")
            return None, None

        # Sort by time
        notes.sort(key=lambda x: x[0])

        time_seconds = np.array([x[0] for x in notes])
        data_values = np.array([x[1] for x in notes])

        # Remove duplicate times (Analysis requires strictly increasing for some checks,
        # though strictly speaking multiple notes can play at once.
        # For simple spectral analysis, we might want to just take the first note at a time or average.)
        # Here we will just take the unique times.

        unique_times, indices = np.unique(time_seconds, return_index=True)
        data_values = data_values[indices]
        time_seconds = unique_times

        return time_seconds, data_values
    except Exception as e:
        logger.error(f"Failed to load MIDI: {e}")
        return None, None

# --- Analysis Runner ---

def run_analysis_case(name, time, data, is_uneven, expected_beta):
    """Runs LS and Haar analysis on the given data."""

    suffix = "uneven" if is_uneven else "original"
    output_subdir = os.path.join(PLOTS_DIR, f"{name}_{suffix}")

    # Check if analysis already done
    summary_file = os.path.join(output_subdir, f"{name}_{suffix}_summary.txt")
    if os.path.exists(summary_file):
        logger.info(f"Analysis for {name} ({suffix}) already exists. Skipping.")

        ls_beta = np.nan
        haar_beta = np.nan

        try:
            with open(summary_file, 'r') as f:
                content = f.read()

            # Parse Lomb-Scargle Beta
            ls_match = re.search(r"Value:\s+β\s*=\s*([-\d\.]+)", content)
            if ls_match:
                try:
                    ls_beta = float(ls_match.group(1))
                except ValueError:
                    pass

            # Parse Haar Beta
            haar_match = re.search(r"Haar Wavelet Analysis:.*?β\s*=\s*([-\d\.]+)", content, re.DOTALL)
            if haar_match:
                try:
                    haar_beta = float(haar_match.group(1))
                except ValueError:
                    pass
        except Exception as e:
            logger.warning(f"Failed to parse summary file for {name} ({suffix}): {e}")

        analyzer = Analysis(time_array=time[:10], data_array=data[:10], time_col="time", data_col="data", input_time_unit="seconds", param_name=f"{name}_{suffix}")
        sanitized_name = analyzer._sanitize_filename(analyzer.param_name)

        return {
            "dataset": name,
            "type": suffix,
            "expected_beta": expected_beta,
            "ls_beta": ls_beta,
            "haar_beta": haar_beta,
            "ls_plot": os.path.join(output_subdir, f"{sanitized_name}_spectrum_plot.png"),
            "haar_plot": os.path.join(output_subdir, f"{sanitized_name}_haar_plot.png")
        }

    os.makedirs(output_subdir, exist_ok=True)

    logger.info(f"Running analysis for {name} ({suffix})...")

    results = {
        "dataset": name,
        "type": suffix,
        "expected_beta": expected_beta,
        "ls_beta": np.nan,
        "haar_beta": np.nan,
        "ls_plot": None,
        "haar_plot": None
    }

    try:
        analyzer = Analysis(
            time_array=time,
            data_array=data,
            time_col="time",
            data_col="value",
            input_time_unit="seconds",
            param_name=f"{name}_{suffix}",
            # Use 'parametric' CI for speed in this bulk analysis, or 'bootstrap' if robust needed.
            # Plan said "standard", let's use default (bootstrap) but limit n_bootstraps if it's too slow?
            # Default is 2000. Let's use 500 to be faster.
            min_valid_data_points=10
        )

        # Check data size and switch to OLS if too large to avoid MemoryError with Theil-Sen
        # Theil-Sen is O(N^2) memory in SciPy. N=650k is way too big (TiBs).
        # Threshold: 5000 points is roughly where it starts getting slow/heavy.
        fit_method = "theil-sen"
        # However, the fitting happens on the periodogram, not the time series.
        # The number of frequencies depends on nyquist and duration.
        # But roughly, N_freq ~ N_time / 2 (or more with oversampling).
        # So we check len(data) as a proxy.
        if len(data) > 5000:
             fit_method = "ols"
             logger.info(f"Data size {len(data)} is large. Switching to fit_method='ols' to avoid MemoryError.")

        res = analyzer.run_full_analysis(
            output_dir=output_subdir,
            max_breakpoints=0, # Non-segmented
            run_haar=True,
            n_bootstraps=500,
            fit_method=fit_method
        )

        # Extract betas
        results["ls_beta"] = res.get("beta", np.nan)
        if "haar_results" in res:
            results["haar_beta"] = res["haar_results"].get("beta", np.nan)

        # Store plot paths (relative to repo root)
        # waterSpec names plots as "{sanitized_name}_spectrum_plot.png"
        sanitized_name = analyzer._sanitize_filename(analyzer.param_name)
        results["ls_plot"] = os.path.join(output_subdir, f"{sanitized_name}_spectrum_plot.png")
        results["haar_plot"] = os.path.join(output_subdir, f"{sanitized_name}_haar_plot.png")

    except Exception as e:
        logger.error(f"Analysis failed for {name} ({suffix}): {e}")

    return results

def main():
    ensure_dirs()

    datasets = [
        ("hadcrut4", load_hadcrut4, EXPECTED_BETAS["hadcrut4"]),
        ("sp500", load_sp500, EXPECTED_BETAS["sp500"]),
        ("mitbih", load_mitbih, EXPECTED_BETAS["mitbih"]),
        ("midi", load_midi, EXPECTED_BETAS["midi"])
    ]

    all_results = []

    for name, loader_func, expected_beta in datasets:
        logger.info(f"Processing {name}...")
        time, data = loader_func()

        if time is None or data is None:
            logger.warning(f"Skipping {name} due to loading failure.")
            continue

        # Original Analysis
        res_orig = run_analysis_case(name, time, data, is_uneven=False, expected_beta=expected_beta)
        all_results.append(res_orig)

        # Uneven Analysis
        # Randomly sample 60% of data
        mask = np.random.rand(len(time)) < 0.6
        if np.sum(mask) < 10:
             logger.warning(f"Skipping uneven analysis for {name}: too few points after sampling.")
        else:
            time_uneven = time[mask]
            data_uneven = data[mask]
            res_uneven = run_analysis_case(name, time_uneven, data_uneven, is_uneven=True, expected_beta=expected_beta)
            all_results.append(res_uneven)

    # Generate Report
    generate_report(all_results)

def generate_report(results):
    report_path = os.path.join(RESULTS_DIR, "ANALYSIS_REPORT.md")
    logger.info(f"Generating report at {report_path}...")

    with open(report_path, "w") as f:
        f.write("# Comprehensive Spectral Analysis Report\n\n")
        f.write("This report summarizes the spectral analysis of various datasets using Lomb-Scargle and Haar Wavelet methods.\n\n")

        f.write("## Summary Table\n\n")
        f.write("| Dataset | Type | Expected Beta | LS Beta | Haar Beta |\n")
        f.write("| :--- | :--- | :--- | :--- | :--- |\n")

        for res in results:
            ls_beta = f"{res['ls_beta']:.2f}" if not np.isnan(res['ls_beta']) else "N/A"
            haar_beta = f"{res['haar_beta']:.2f}" if not np.isnan(res['haar_beta']) else "N/A"
            f.write(f"| {res['dataset']} | {res['type']} | {res['expected_beta']} | {ls_beta} | {haar_beta} |\n")

        f.write("\n## Detailed Results\n\n")

        # Group by dataset
        datasets = sorted(list(set(r['dataset'] for r in results)))

        for ds in datasets:
            f.write(f"### {ds.upper()}\n\n")

            ds_results = [r for r in results if r['dataset'] == ds]

            for res in ds_results:
                f.write(f"#### {res['type'].capitalize()} Sampling\n")
                f.write(f"- **Expected Beta**: {res['expected_beta']}\n")
                f.write(f"- **Lomb-Scargle Beta**: {res['ls_beta']:.2f}\n")
                f.write(f"- **Haar Beta**: {res['haar_beta']:.2f}\n")

                f.write("\n**Plots:**\n\n")
                if res['ls_plot'] and os.path.exists(res['ls_plot']):
                     # Make path relative to report file
                     rel_path = os.path.relpath(res['ls_plot'], RESULTS_DIR)
                     f.write(f"![Lomb-Scargle Spectrum]({rel_path})\n")

                if res['haar_plot'] and os.path.exists(res['haar_plot']):
                     rel_path = os.path.relpath(res['haar_plot'], RESULTS_DIR)
                     f.write(f"![Haar Analysis]({rel_path})\n")

                f.write("\n---\n")

if __name__ == "__main__":
    main()

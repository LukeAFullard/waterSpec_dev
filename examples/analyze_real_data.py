from waterSpec import Analysis
import os
import sys

def analyze_real_data(file_path, param_name="Discharge"):
    print(f"\nAnalyzing: {param_name}")
    print(f"Data Source: {file_path}")
    print("-" * 50)

    try:
        # Create analyzer
        # Discharge data usually doesn't need log transformation for spectral slope,
        # but often it is log-transformed. The benchmarks for discharge (1.0-1.8)
        # usually refer to the log-log slope of the PSD of the raw or log-transformed data.
        # Let's try raw first, as discharge is extensive.
        # Actually, for power laws, log-transforming the *data* first is common
        # if the data spans orders of magnitude (like discharge).
        # Let's stick to standard practice: simple detrending.

        analyzer = Analysis(
            file_path=file_path,
            time_col='timestamp',
            data_col='value',
            param_name=param_name,
            censor_strategy='drop',
            detrend_method='linear', # Remove long-term trend
            normalize_data=True      # Standardize to variance=1
        )

        # Run full analysis
        output_dir = os.path.join("examples", "real_data_output")
        results = analyzer.run_full_analysis(
            output_dir=output_dir,
            fit_method='ols',       # OLS is much faster and memory-efficient for large datasets
            ci_method='parametric', # Faster for this quick check
            max_breakpoints=1,
            normalization='standard',
            samples_per_peak=2      # Reduce grid density for speed
        )

        # Print summary to console
        print("\nAnalysis Summary:")
        print(results['summary_text'])

        return results

    except Exception as e:
        print(f"Analysis failed: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Analyze spectral properties of a time series.")
    parser.add_argument("file_path", help="Path to the CSV file")
    parser.add_argument("--param", default="Parameter", help="Name of the parameter being analyzed")

    args = parser.parse_args()

    if not os.path.exists(args.file_path):
        print(f"Error: Data file {args.file_path} not found.")
        sys.exit(1)

    results = analyze_real_data(args.file_path, args.param)

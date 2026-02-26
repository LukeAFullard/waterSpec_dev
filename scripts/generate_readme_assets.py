import os
import shutil
import tempfile

from waterSpec import Analysis


def main():
    """
    This script generates the assets (plot and summary text) needed for the
    main README.md file. It runs the standard analysis on the sample data
    and saves the outputs to the assets/ directory.
    """
    # Define key paths
    file_path = 'examples/sample_data.csv'
    asset_dir = 'assets'
    param_name = 'readme_example'

    # Ensure the assets directory exists
    os.makedirs(asset_dir, exist_ok=True)

    print("Running analysis to generate README assets...")

    with tempfile.TemporaryDirectory() as temp_dir:
        # 1. Initialize and run the analysis
        # We use a temporary directory because the output filenames are auto-generated.
        analyzer = Analysis(
            file_path=file_path,
            time_col='timestamp',
            data_col='concentration',
            param_name=param_name
        )
        analyzer.run_full_analysis(output_dir=temp_dir)

        # 2. Define source and destination paths
        sanitized_name = analyzer._sanitize_filename(param_name)

        source_plot_path = os.path.join(temp_dir, f"{sanitized_name}_spectrum_plot.png")
        dest_plot_path = os.path.join(asset_dir, 'readme_spec_plot.png')

        source_summary_path = os.path.join(temp_dir, f"{sanitized_name}_summary.txt")
        dest_summary_path = os.path.join(asset_dir, 'readme_summary.txt')

        # 3. Move and rename the files to their final destination
        print(f"Moving plot to {dest_plot_path}")
        shutil.move(source_plot_path, dest_plot_path)

        print(f"Moving summary to {dest_summary_path}")
        shutil.move(source_summary_path, dest_summary_path)

    print("README assets generated successfully.")

if __name__ == '__main__':
    main()

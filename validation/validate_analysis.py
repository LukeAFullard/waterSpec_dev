import filecmp
import logging
import os
import shutil
import sys
import difflib

# Add the project root to the Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from src.waterSpec import Analysis

# Set up basic logging for the script
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    stream=sys.stdout,
)
logger = logging.getLogger(__name__)


def main():
    """
    Runs a validation test to compare the current analysis output with a
    known-good reference output.
    """
    # --- Setup ---
    file_path = "examples/sample_data.csv"
    output_dir = "validation/temp_output_for_validation"
    reference_dir = "validation/reference_outputs"
    param_name = "Validation Reference"

    # Create a clean output directory
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir)

    exit_code = 1  # Default to failure
    try:
        # --- Run Analysis ---
        logger.info("Running analysis on sample data...")
        analyzer = Analysis(
            file_path=file_path,
            time_col="timestamp",
            data_col="concentration",
            param_name=param_name,
            detrend_method="linear",
        )
        analyzer.run_full_analysis(output_dir=output_dir)
        logger.info("Analysis complete.")

        # --- Compare a single file ---
        generated_filename = f"{analyzer._sanitize_filename(param_name)}_summary.txt"
        generated_file = os.path.join(output_dir, generated_filename)
        reference_file = os.path.join(reference_dir, "expected_summary.txt")

        logger.info("Comparing '%s' with '%s'...", generated_file, reference_file)

        if not os.path.exists(generated_file):
            logger.error(
                "Validation failed: Output file '%s' was not generated.",
                generated_file,
            )

        elif not os.path.exists(reference_file):
            logger.error(
                "Validation failed: Reference file '%s' does not exist.",
                reference_file,
            )

        elif filecmp.cmp(generated_file, reference_file, shallow=False):
            logger.info("✅ Validation Successful: Output matches the reference output.")
            exit_code = 0
        else:
            logger.error(
                "❌ Validation Failed: Output does not match the reference output."
            )
            # Print diff
            with open(generated_file) as f1, open(reference_file) as f2:
                diff = difflib.unified_diff(
                    f2.readlines(),
                    f1.readlines(),
                    fromfile="expected",
                    tofile="generated",
                )
                print("--- DIFF ---")
                for line in diff:
                    sys.stdout.write(line)
                print("------------")

    except Exception as e:
        logger.error(
            "Validation failed: An unexpected error occurred during execution: %s",
            e,
            exc_info=True,
        )
        # exit_code remains 1

    finally:
        # --- Teardown ---
        if os.path.exists(output_dir):
            shutil.rmtree(output_dir)
        logger.info("Cleaned up temporary output directory.")

    sys.exit(exit_code)


if __name__ == "__main__":
    main()
"""
This script provides a best-practice example of how to run a robust analysis
using the waterSpec package. It includes:
- A try...except block to gracefully handle potential errors during analysis.
- The use of the `verbose=True` flag to enable detailed logging.
- A demonstration of how to access and interpret the `uncertainty_warnings`
  that are new in this version of the package.
"""

import logging
import os
import sys
from src.waterSpec import Analysis

# Set up basic logging for the script
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    stream=sys.stdout,
)
logger = logging.getLogger(__name__)

# --- 1. Define Configuration ---
# Use a sample data file that is known to have some issues
FILE_PATH = "examples/sample_data.csv"
TIME_COL = "timestamp"
DATA_COL = "concentration"
PARAM_NAME = "Sample Concentration (Robust Analysis)"
OUTPUT_DIR = "example_robust_output"

# Create the output directory if it doesn't exist
os.makedirs(OUTPUT_DIR, exist_ok=True)


def run_robust_analysis():
    """
    Runs the waterSpec analysis within a robust error-handling framework.
    """
    logger.info("--- Starting Robust Analysis for: %s ---", PARAM_NAME)

    try:
        # --- 2. Initialize the Analysis Object ---
        # Using verbose=True will print detailed logs of each step.
        logger.info("Step 1: Initializing analysis and preprocessing data...")
        analyzer = Analysis(
            file_path=FILE_PATH,
            time_col=TIME_COL,
            data_col=DATA_COL,
            param_name=PARAM_NAME,
            verbose=True,  # Enable detailed logging
        )
        logger.info("Initialization and preprocessing complete.")

        # --- 3. Run the Full Analysis ---
        # This includes periodogram calculation, model fitting, and peak detection.
        logger.info("Step 2: Running the full spectral analysis...")
        results = analyzer.run_full_analysis(
            output_dir=OUTPUT_DIR,
            seed=42,  # Use a seed for reproducible results
        )
        logger.info("Full analysis complete.")

        # --- 4. Review the Results ---
        logger.info("Step 3: Reviewing results and uncertainty warnings...")
        # The full, formatted summary is in the 'summary_text' key
        print("\n--- Full Analysis Summary ---")
        print(results["summary_text"])

        # The new `uncertainty_warnings` key contains a list of any warnings
        # generated during the analysis. This is useful for programmatic checks.
        if results["uncertainty_warnings"]:
            logger.info("--- Uncertainty Warnings Detected ---")
            for warning in results["uncertainty_warnings"]:
                logger.warning("- %s", warning)
        else:
            logger.info("--- No Uncertainty Warnings Detected ---")

        logger.info("Analysis successful. Outputs saved to '%s'.", OUTPUT_DIR)

    except FileNotFoundError as e:
        logger.error("The data file was not found. Details: %s", e, exc_info=True)
    except ValueError as e:
        logger.error(
            "A data validation error occurred. Details: %s", e, exc_info=True
        )
    except Exception as e:
        logger.error(
            "An unexpected error occurred during the analysis: %s", e, exc_info=True
        )


if __name__ == "__main__":
    run_robust_analysis()
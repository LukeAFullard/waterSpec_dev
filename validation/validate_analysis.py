import filecmp
import os
import shutil
import sys

from waterSpec import Analysis


def main():
    """
    Runs a validation test to compare the current analysis output with a
    known-good reference output.
    """
    # --- Setup ---
    file_path = 'examples/sample_data.csv'
    output_dir = 'validation/temp_output_for_validation'
    reference_dir = 'validation/reference_outputs'
    param_name = 'Validation Reference'

    # Create a clean output directory
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir)

    exit_code = 1  # Default to failure
    try:
        # --- Run Analysis ---
        print("Running analysis on sample data...")
        analyzer = Analysis(
            file_path=file_path,
            time_col='timestamp',
            data_col='concentration',
            param_name=param_name
        )
        analyzer.run_full_analysis(output_dir=output_dir)
        print("Analysis complete.")

        # --- Compare a single file ---
        generated_filename = f"{analyzer._sanitize_filename(param_name)}_summary.txt"
        generated_file = os.path.join(output_dir, generated_filename)
        reference_file = os.path.join(reference_dir, 'expected_summary.txt')

        print(f"Comparing '{generated_file}' with '{reference_file}'...")

        if not os.path.exists(generated_file):
            print(
                f"Validation failed: Output file '{generated_file}' was not "
                "generated."
            )

        elif not os.path.exists(reference_file):
            print(
                f"Validation failed: Reference file '{reference_file}' does not "
                "exist."
            )

        elif filecmp.cmp(generated_file, reference_file, shallow=False):
            print("\n✅ Validation Successful: Output matches the reference output.")
            exit_code = 0
        else:
            print("\n❌ Validation Failed: Output does not match the reference output.")
            # Optional: print diff
            import difflib
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
        print(f"Validation failed: An error occurred during execution: {e}")
        # exit_code remains 1

    finally:
        # --- Teardown ---
        if os.path.exists(output_dir):
            shutil.rmtree(output_dir)
        print("Cleaned up temporary output directory.")

    sys.exit(exit_code)

if __name__ == "__main__":
    main()

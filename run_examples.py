from waterSpec import Analysis
import pprint

def run_readme_example():
    """
    Runs the example code from the README.md file.
    """
    print("--- Running README.md Example ---")
    # 1. Initialize the Analysis object with your data.
    analyzer = Analysis(
        file_path='examples/sample_data.csv',
        time_col='timestamp',
        data_col='concentration'
    )

    # 2. Run the full analysis with a single command.
    results = analyzer.run_full_analysis(output_dir='readme_output')

    print("README Example Analysis Results:")
    pprint.pprint(results)
    print("--- README.md Example Complete ---\n")

def run_new_example_md_example():
    """
    Runs the example code from the NEW_EXAMPLE.md file.
    """
    print("--- Running NEW_EXAMPLE.md Example ---")
    # This is the same as the README example, but we run it again
    # to ensure the outputs are created for the documentation update.
    analyzer = Analysis(
        file_path='examples/sample_data.csv',
        time_col='timestamp',
        data_col='concentration',
        param_name="Nitrate Concentration" # Use a more descriptive name
    )

    results = analyzer.run_full_analysis(output_dir='new_example_output')
    print("NEW_EXAMPLE.md Example Complete")
    print("Outputs saved to 'new_example_output/' directory.")


if __name__ == "__main__":
    run_readme_example()
    run_new_example_md_example()

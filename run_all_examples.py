import subprocess
import sys

# A list of all the example scripts to be run.
scripts_to_run = [
    "run_readme_example.py",
    "examples/run_beta_example.py",
    "examples/run_peak_example.py",
]

# Loop through the scripts and execute them one by one.
for script in scripts_to_run:
    print(f"Running {script}...")
    try:
        # We use sys.executable to ensure the script is run with the same
        # Python interpreter that is running this script.
        subprocess.run(
            [sys.executable, script],
            check=True,
            capture_output=True,
            text=True,
            timeout=600  # Set a 10-minute timeout to prevent hangs
        )
        print(f"Successfully ran {script}.")
    except subprocess.CalledProcessError as e:
        print(f"ERROR: {script} failed with exit code {e.returncode}.")
        print("--- STDOUT ---")
        print(e.stdout)
        print("--- STDERR ---")
        print(e.stderr)
    except subprocess.TimeoutExpired as e:
        print(f"ERROR: {script} timed out.")
        if e.stdout:
            print("--- STDOUT ---")
            print(e.stdout)
        if e.stderr:
            print("--- STDERR ---")
            print(e.stderr)

print("\nAll examples have been executed.")
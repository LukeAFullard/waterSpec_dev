import os
import sys

def run_test_17():
    print("Running 17: Aggregating all and compiling FINDINGS.md")
    os.makedirs('validation/results', exist_ok=True)
    import subprocess
    subprocess.run(['python3', 'validation/generate_findings_csv.py'])
    print("Check validation/FINDINGS.md and validation/results/master_summary.csv")
    return True

if __name__ == '__main__':
    run_test_17()
    sys.exit(0)

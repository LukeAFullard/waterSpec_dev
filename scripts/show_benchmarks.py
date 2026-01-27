#!/usr/bin/env python
"""
Script to display the scientific benchmarks used in waterSpec for beta interpretation.
"""
import sys
import os

# Add src to path if running from root without installation
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))

try:
    from waterSpec.interpreter import BENCHMARK_TABLE
except ImportError:
    print("Error: Could not import waterSpec. Please ensure the package is installed or src is in your path.")
    sys.exit(1)

def main():
    print("\n=== waterSpec Scientific Benchmarks for Beta (Spectral Slope) ===\n")
    print("These benchmarks are based on Liang et al. (2021) and other sources.\n")

    print(BENCHMARK_TABLE.to_string())
    print("\n")
    print("References:")
    print("  Liang, X., Schilling, K. E., Jones, C. S., & Zhang, Y. K. (2021).")
    print("  Temporal scaling of long-term co-occurring agricultural contaminants")
    print("  and the implications for conservation planning. Environmental Research Letters.")

if __name__ == "__main__":
    main()

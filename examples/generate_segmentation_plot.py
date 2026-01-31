
import numpy as np
import matplotlib.pyplot as plt
from waterSpec import SegmentedRegimeAnalysis

def generate_plot():
    # 1. Generate Synthetic "Bursty" Data
    np.random.seed(42)
    time = np.arange(0, 1000, 1.0) # Hourly
    # Baseflow: low noise
    data = np.random.normal(0, 0.1, 1000)
    # Add 3 storm events (high volatility bursts)
    data[100:150] += np.random.normal(0, 2.0, 50)
    data[400:430] += np.random.normal(0, 3.0, 30)
    data[800:850] += np.random.normal(0, 1.5, 50)

    # 2. Perform Segmentation
    # We look for volatility at a 6-hour scale
    # Events are defined as > 3x the median background volatility
    results = SegmentedRegimeAnalysis.segment_by_fluctuation(
        time, data, scale=6.0, threshold_factor=3.0
    )

    events = results['events']

    # 3. Plot
    plt.figure(figsize=(12, 6))
    plt.plot(time, data, 'k-', alpha=0.6, label='Original Data')

    # Highlight events
    for start, end in events:
        plt.axvspan(start, end, color='red', alpha=0.3, label='Detected Event' if start == events[0][0] else "")

    plt.title("Event-Based Segmentation using Sliding Haar Volatility")
    plt.xlabel("Time (hours)")
    plt.ylabel("Value")
    plt.legend()
    plt.grid(True, alpha=0.3)

    output_path = "examples/output/demo8_segmentation.png"
    plt.savefig(output_path)
    print(f"Plot saved to {output_path}")

if __name__ == "__main__":
    generate_plot()

import numpy as np
import matplotlib.pyplot as plt
import os
import sys

# Ensure waterSpec is in path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

from waterSpec.haar_analysis import HaarAnalysis, calculate_sliding_haar
from waterSpec.bivariate import BivariateAnalysis

OUTPUT_DIR = "examples/output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def demo_overlapping_haar():
    print("Running Demo 1: Overlapping Haar Analysis...")
    np.random.seed(42)
    n_points = 500
    time = np.arange(n_points)
    # Random walk (Brown noise, beta=2)
    data = np.cumsum(np.random.randn(n_points))

    haar = HaarAnalysis(time, data, time_unit="days")

    # Run with overlap
    res_ov = haar.run(overlap=True, overlap_step_fraction=0.1, n_bootstraps=200)

    # Run without overlap for comparison
    res_no = haar.run(overlap=False, n_bootstraps=200)

    plt.figure(figsize=(10, 6))
    plt.loglog(res_no['lags'], res_no['s1'], 'ko--', label='Non-Overlapping', alpha=0.5)
    plt.loglog(res_ov['lags'], res_ov['s1'], 'b.-', label='Overlapping (Smoother)')

    # Plot fits
    fit_ov = 10**res_ov['intercept'] * res_ov['lags']**res_ov['H']
    plt.loglog(res_ov['lags'], fit_ov, 'r-', label=f"Fit (Overlap): H={res_ov['H']:.2f}")

    plt.xlabel("Time Scale (days)")
    plt.ylabel("Haar Fluctuation S1")
    plt.title("Benefit of Overlapping Windows for Haar Analysis")
    plt.legend()
    plt.grid(True, which="both", alpha=0.3)
    plt.savefig(f"{OUTPUT_DIR}/demo1_overlapping_haar.png")
    plt.close()
    print(f"  -> Plot saved to {OUTPUT_DIR}/demo1_overlapping_haar.png")

def demo_segmented_fit():
    print("Running Demo 2: Regime Shift Detection (Segmented Fit)...")
    time = np.arange(1000)
    # Noise (short scale) + Trend (long scale)
    noise = np.random.normal(0, 1.0, 1000)
    trend = 0.05 * np.cumsum(np.random.normal(0, 1.0, 1000))
    data = noise + trend

    haar = HaarAnalysis(time, data, time_unit="hours")
    res = haar.run(min_lag=2, max_lag=200, overlap=True, max_breakpoints=1)

    # Plot is handled by the class method, but we want to save it
    # We can manually call plot using the results
    haar.plot(output_path=f"{OUTPUT_DIR}/demo2_regime_shift.png")
    print(f"  -> Plot saved to {OUTPUT_DIR}/demo2_regime_shift.png")

def demo_bivariate_cross_haar():
    print("Running Demo 3: Bivariate Cross-Haar Correlation...")
    time = np.arange(0, 365, 1.0)
    # Q: Driver (Discharge)
    Q = np.exp(np.sin(time/10) + np.random.normal(0, 0.2, len(time)))
    # C: Response (Conc) - strongly correlated at long scales (flushing)
    # Add noise to decouple short scales
    C = 0.5 * Q + np.random.normal(0, 1.0, len(time))

    biv = BivariateAnalysis(time, C, "Nitrate", time, Q, "Discharge")
    biv.align_data(tolerance=0.1)

    lags = np.logspace(0.5, 2.5, 15) # 3 days to ~300 days
    res = biv.run_cross_haar_analysis(lags, overlap=True)

    plt.figure(figsize=(10, 6))
    plt.semilogx(res['lags'], res['correlation'], 'o-')
    plt.axhline(0, color='k', linestyle='--')
    plt.xlabel("Time Scale (days)")
    plt.ylabel("Correlation (Delta C vs Delta Q)")
    plt.title("Scale-Dependent Concentration-Discharge Relationship")
    plt.grid(True, which="both")
    plt.savefig(f"{OUTPUT_DIR}/demo3_cross_haar.png")
    plt.close()
    print(f"  -> Plot saved to {OUTPUT_DIR}/demo3_cross_haar.png")

def demo_hysteresis():
    print("Running Demo 4: Hysteresis Loop Analysis...")
    t = np.linspace(0, 4*np.pi, 200)
    # Counter-clockwise loop (Q leads C? or Phase shift)
    # Q = cos(t), C = sin(t). Q peaks at 0, C at pi/2. C lags Q.
    Q = np.cos(t)
    C = np.sin(t)

    biv = BivariateAnalysis(t, C, "Conc", t, Q, "Q")
    biv.align_data(tolerance=0.1)

    # We need to manually extract the fluctuations to plot the loop
    # BivariateAnalysis doesn't expose raw fluctuations in the public API easily
    # But we can replicate the logic for the plot
    tau = 1.0 # Small scale ~ derivative
    # Using sliding Haar logic from calculate_sliding_haar
    tc, fluc_c = calculate_sliding_haar(t, C, window_size=tau)
    tq, fluc_q = calculate_sliding_haar(t, Q, window_size=tau)

    # Trim to match
    min_len = min(len(fluc_c), len(fluc_q))
    fluc_c = fluc_c[:min_len]
    fluc_q = fluc_q[:min_len]

    stats = biv.calculate_hysteresis_metrics(tau=tau)

    plt.figure(figsize=(8, 8))
    plt.plot(fluc_q, fluc_c, 'b.-', alpha=0.5)
    # Mark start
    plt.plot(fluc_q[0], fluc_c[0], 'go', label='Start')
    plt.plot(fluc_q[-1], fluc_c[-1], 'rs', label='End')

    plt.xlabel("Discharge Fluctuation (Delta Q)")
    plt.ylabel("Concentration Fluctuation (Delta C)")
    plt.title(f"Hysteresis Loop (Scale={tau})\nArea={stats['area']:.2f} ({stats['direction']})")
    plt.axhline(0, color='k', lw=0.5)
    plt.axvline(0, color='k', lw=0.5)
    plt.legend()
    plt.grid(True)
    plt.savefig(f"{OUTPUT_DIR}/demo4_hysteresis.png")
    plt.close()
    print(f"  -> Plot saved to {OUTPUT_DIR}/demo4_hysteresis.png")

def demo_anomaly_detection():
    print("Running Demo 5: Real-Time Anomaly Detection...")
    # Base noise
    data = np.random.normal(0, 0.5, 200)
    # Anomaly: Burst of volatility
    data[100:120] = np.random.normal(0, 3.0, 20)
    time = np.arange(200)

    # Calculate Sliding Haar
    window = 10.0
    tc, fluc = calculate_sliding_haar(time, data, window_size=window)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

    ax1.plot(time, data, 'k-', alpha=0.7)
    ax1.axvspan(100, 120, color='r', alpha=0.1, label='Anomaly Window')
    ax1.set_ylabel("Raw Data")
    ax1.set_title("Time Series with Hidden Volatility Burst")
    ax1.legend()

    ax2.plot(tc, np.abs(fluc), 'r-', label=f'Haar Fluctuation (Window={window})')
    ax2.axhline(3*0.5, color='k', linestyle='--', label='Baseline Threshold')
    ax2.set_ylabel("|Delta X|")
    ax2.set_xlabel("Time")
    ax2.legend()

    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/demo5_anomaly.png")
    plt.close()
    print(f"  -> Plot saved to {OUTPUT_DIR}/demo5_anomaly.png")

if __name__ == "__main__":
    print(f"Generating walkthrough examples in {OUTPUT_DIR}...")
    demo_overlapping_haar()
    demo_segmented_fit()
    demo_bivariate_cross_haar()
    demo_hysteresis()
    demo_anomaly_detection()
    print("Done! Check the output directory for images.")

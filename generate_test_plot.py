import numpy as np
import matplotlib.pyplot as plt
import sys
import os

from waterSpec.haar_analysis import HaarAnalysis

def power_law_psd(f, beta):
    return f ** -beta

def make_seasonal_series(
    beta_true=1.4, n_points=4000, dt=1.0, seasonal_amp_ratio=4.0,
    period=365.25, phase=0.7, seed=42,
):
    from waterSpec.utils_sim.tk95 import simulate_tk95
    time, noise = simulate_tk95(
        power_law_psd, (beta_true,), n_points, dt, seed=seed
    )
    noise = noise - np.mean(noise)
    std_noise = np.std(noise)
    amp = seasonal_amp_ratio * std_noise
    w = 2.0 * np.pi / period
    seasonal_signal = amp * np.cos(w * time + phase)
    data = noise + seasonal_signal
    return time, data, noise, seasonal_signal

beta_true = 1.4
time, data, noise, _ = make_seasonal_series(beta_true=beta_true, seed=42)

# Raw (contaminated) data
haar_raw = HaarAnalysis(time, data, time_unit="days")
res_raw = haar_raw.run(aggregation="rms", n_bootstraps=0)

# Corrected data
haar_corr = HaarAnalysis(time, data, time_unit="days")
res_corr = haar_corr.run(
    aggregation="rms",
    n_bootstraps=0,
    correct_periodicity=True,
    periodic_periods=[365.25]
)

# Oracle (pure noise)
haar_oracle = HaarAnalysis(time, noise, time_unit="days")
res_oracle = haar_oracle.run(aggregation="rms", n_bootstraps=0)

plt.figure(figsize=(10, 6))

# Plot the raw data structure function
lags = res_raw['lags']
plt.loglog(lags, res_raw['s1'], 'ro-', alpha=0.5, label=f'Raw Data (Contaminated) ($\\beta$={res_raw["beta"]:.2f})')

# Plot the pure noise (Oracle) structure function
plt.loglog(lags, res_oracle['s1'], 'k--', alpha=0.8, label=f'Oracle (Pure Noise) ($\\beta$={res_oracle["beta"]:.2f})')

# Plot the corrected structure function
plt.loglog(lags, res_corr['s1'], 'bo-', alpha=0.8, label=f'Corrected Data ($\\beta$={res_corr["beta"]:.2f})')

plt.axvline(365.25, color='gray', linestyle=':', label='Periodic Signal (365.25 days)')

plt.xlabel('Lag (days)')
plt.ylabel('Haar Fluctuation ($S_1$)')
plt.title('Haar Structure Function: Periodicity Correction')
plt.legend()
plt.grid(True, which='both', ls='--', alpha=0.5)

plt.tight_layout()
plt.savefig('periodicity_correction_test_plot.png')
print("Saved plot to periodicity_correction_test_plot.png")

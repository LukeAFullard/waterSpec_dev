# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
from scipy.special import gammaln
import matplotlib.pyplot as plt
import warnings

# --- WaterSpec Imports ---
try:
    from waterSpec.haar_analysis import calculate_haar_fluctuations, fit_haar_slope
    from waterSpec.utils_sim.tk95 import simulate_tk95
    from waterSpec.utils_sim.models import power_law
except ImportError:
    import sys
    import os
    # Add src to path if running from repo root
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))
    from waterSpec.haar_analysis import calculate_haar_fluctuations, fit_haar_slope
    from waterSpec.utils_sim.tk95 import simulate_tk95
    from waterSpec.utils_sim.models import power_law

# --- GapWaveSpectra Implementation (Extracted) ---
# Extracted from Reference_project/GapWaveSpectra/haar_structure_function.py
# and sim_GW.py to avoid dependencies on astroML, PyAstronomy, lmfit

def seq(start, end, by=None, length_out=None):
    # Simplified version of R's seq
    if by is not None:
        return np.arange(start, end + by/1000, by) # + small epsilon
    if length_out is not None:
        return np.linspace(start, end, int(length_out))
    return np.arange(start, end + 1)

def mean_in_range(t0, j, sequence, scali2, dat):
    indices = np.where((t0 >= sequence[j]) & (t0 < (sequence[j] + scali2)))
    if len(indices[0]) == 0:
        return np.nan
    return np.mean(dat[indices])

def sdsmpl(ser):
    ser = ser[~np.isnan(ser)]
    s = np.std(ser, ddof=1) # Sample std deviation (N-1)
    n = len(ser)
    if n < 2:
        return np.nan
    if n < 101:
        # Correction for small sample standard deviation of Gaussian
        # c4 factor = sqrt(2/(n-1)) * gamma(n/2) / gamma((n-1)/2)
        # Expected value of sample std S is c4 * sigma. So sigma_hat = S / c4.
        # However, GapWaveSpectra formula is: s * sqrt((n-1)/2) * exp(...)
        # Let's check the formula: exp(gammaln((n-1)/2) - gammaln(n/2)) is 1/gamma_ratio
        # This matches the inverse of the c4 factor correction.
        return s * np.sqrt((n - 1) / 2) * np.exp(gammaln((n - 1) / 2) - gammaln(n / 2))
    else:
        return s

def do_haar_gapwavespectra(dat, t0, scales, overlap=0):
    hfluc = np.zeros(len(scales))
    largestscale = t0[-1]

    for i in range(len(scales)):
        scali = scales[i]
        scali2 = scali / 2

        # Determine number of fluctuations
        maxfluc = int(np.floor((largestscale - t0[0] + 1e-9) / scali)) # Added epsilon
        if maxfluc < 1:
             hfluc[i] = np.nan
             continue

        # Sequences
        step = (1 - overlap) * scali
        if step <= 0: step = scali # Avoid infinite loop if overlap=1 (not supported here properly)

        seqa = np.arange(t0[0], largestscale - scali + step/1000, step)
        # Note: GapWaveSpectra uses a specific seq implementation.
        # For simplicity, we assume standard tiling here if overlap=0.

        seqb = seqa + scali2

        mfluca = [mean_in_range(t0, j, seqa, scali2, dat) for j in range(len(seqa))]
        mflucb = [mean_in_range(t0, j, seqb, scali2, dat) for j in range(len(seqb))]

        # Truncate
        L = min(len(mfluca), len(mflucb))
        mfluca = np.array(mfluca[:L])
        mflucb = np.array(mflucb[:L])

        flucs = mflucb - mfluca

        # Scaling
        flucs = 2 * flucs

        if np.all(np.isnan(flucs)):
            hfluc[i] = np.nan
        else:
            flucs = flucs[~np.isnan(flucs)]
            # Mean fluctuation magnitude estimate assuming Gaussian
            # Combine flucs and -flucs to center at 0
            combined = np.concatenate((flucs, -flucs))
            # sdsmpl calculates corrected sigma
            sigma_est = sdsmpl(combined)
            # Convert to Mean Absolute Deviation (MAD) equivalent for Gaussian
            # MAD = sigma * sqrt(2/pi)
            hfluc[i] = np.sqrt(2 / np.pi) * sigma_est

    return hfluc

# --- Comparison Logic ---

def run_comparison(beta_true, N, n_simulations=20):
    print(f"\n--- Comparing for Beta={beta_true}, N={N} ({n_simulations} simulations) ---")

    ws_betas = []
    ws_betas_corrected = []
    gw_betas = []

    ws_s1_errs = []
    ws_corrected_s1_errs = []
    gw_s1_errs = []

    dt = 1.0
    time = np.arange(N) * dt

    # Scales for GapWaveSpectra (dyadic or similar)
    # We use waterSpec's default lags for fairness, or simple dyadic
    scales = np.logspace(np.log10(2*dt), np.log10(N*dt/2), 10)

    for _ in range(n_simulations):
        # Generate data
        try:
            _, y = simulate_tk95(power_law, params=(beta_true, 1.0), N=N, dt=dt)
            # Standardize
            y = (y - np.mean(y)) / np.std(y)
        except Exception as e:
            print(f"Simulation failed: {e}")
            continue

        # --- waterSpec (Standard) ---
        # Note: Using overlap=False to strictly compare with do_haar_gapwavespectra which uses simple tiling.
        # In practice, overlap=True is recommended for waterSpec to improve statistics.
        lags_ws, s1_ws, _, _ = calculate_haar_fluctuations(
            time, y, lag_times=scales, statistic="mean", aggregation="mean", overlap=False
        )
        res_ws = fit_haar_slope(lags_ws, s1_ws)
        if not np.isnan(res_ws['beta']):
            ws_betas.append(res_ws['beta'])

        # --- waterSpec (Corrected) ---
        lags_ws_corr, s1_ws_corr, _, _ = calculate_haar_fluctuations(
            time, y, lag_times=scales, statistic="mean", aggregation="std_corrected", overlap=False
        )
        res_ws_corr = fit_haar_slope(lags_ws_corr, s1_ws_corr)
        if not np.isnan(res_ws_corr['beta']):
            ws_betas_corrected.append(res_ws_corr['beta'])

        # --- GapWaveSpectra ---
        # Note: GapWaveSpectra expects "scales" (1/freq) which are equivalent to lag times here.
        s1_gw = do_haar_gapwavespectra(y, time, scales)

        # Fit slope (simple log-log OLS)
        valid = (scales > 0) & (s1_gw > 0) & (~np.isnan(s1_gw))
        if np.sum(valid) >= 3:
            log_scales = np.log10(scales[valid])
            log_val = np.log10(s1_gw[valid])

            # GapWaveSpectra fits PSD ~ freq^-beta
            # PSD ~ S1^2 * scale
            # S1^2 * scale ~ scale^beta_psd * scale = scale^(beta_psd + 1)? No.
            # Let's check sim_GW.py:
            # Hfluc_scaled = Hfluc**2 / freq = Hfluc**2 * scale
            # PSD ~ scale^beta
            # So Hfluc^2 * scale ~ scale^beta
            # Hfluc^2 ~ scale^(beta-1)
            # Hfluc ~ scale^((beta-1)/2)
            # So slope of log(Hfluc) vs log(scale) is H = (beta-1)/2
            # => beta = 2*H + 1.
            # This matches waterSpec's definition.

            slope, intercept = np.polyfit(log_scales, log_val, 1)
            beta_gw = 2 * slope + 1
            gw_betas.append(beta_gw)

        # Compare S1 values directly (normalized difference)
        # Match arrays
        # s1_gw corresponds to scales
        # s1_ws corresponds to lags_ws (which is a subset of scales)

        s1_gw_matched = []
        s1_ws_matched = []
        s1_ws_corr_matched = []

        for idx, lag in enumerate(lags_ws):
            # Find closest scale (should be exact if floating point allows)
            match_idx = np.where(np.isclose(scales, lag))[0]
            if len(match_idx) > 0:
                val_gw = s1_gw[match_idx[0]]
                val_ws = s1_ws[idx]
                val_ws_corr = s1_ws_corr[idx]

                if not np.isnan(val_gw) and not np.isnan(val_ws):
                    s1_gw_matched.append(val_gw)
                    s1_ws_matched.append(val_ws)
                    s1_ws_corr_matched.append(val_ws_corr)

        s1_gw_matched = np.array(s1_gw_matched)
        s1_ws_matched = np.array(s1_ws_matched)
        s1_ws_corr_matched = np.array(s1_ws_corr_matched)

        if len(s1_gw_matched) > 0:
             # GapWaveSpectra scales fluctuations by 2 inside do_haar
             # waterSpec does not.
             # GapWaveSpectra result is expected to be 2x waterSpec result (ignoring small sample bias)

             # Adjust for comparison
             s1_gw_adj = s1_gw_matched / 2.0

             err = np.mean(np.abs(s1_gw_adj - s1_ws_matched) / s1_ws_matched)
             gw_s1_errs.append(err)

             err_corr = np.mean(np.abs(s1_gw_adj - s1_ws_corr_matched) / s1_ws_corr_matched)
             ws_corrected_s1_errs.append(err_corr)

    print(f"  True Beta: {beta_true}")
    print(f"  WaterSpec Mean Beta (Standard): {np.nanmean(ws_betas):.3f} (Std: {np.nanstd(ws_betas):.3f})")
    print(f"  WaterSpec Mean Beta (Corrected): {np.nanmean(ws_betas_corrected):.3f} (Std: {np.nanstd(ws_betas_corrected):.3f})")
    print(f"  GapWaveSp Mean Beta: {np.nanmean(gw_betas):.3f} (Std: {np.nanstd(gw_betas):.3f})")
    print(f"  Mean Rel. Diff in S1 (Std vs GapWave): {np.nanmean(gw_s1_errs)*100:.2f}%")
    print(f"  Mean Rel. Diff in S1 (Corr vs GapWave): {np.nanmean(ws_corrected_s1_errs)*100:.2f}%")

if __name__ == "__main__":
    print("Starting Comparison...")
    run_comparison(beta_true=2.0, N=1000, n_simulations=10)
    run_comparison(beta_true=2.0, N=50, n_simulations=50)
    run_comparison(beta_true=1.0, N=50, n_simulations=50)

from __future__ import annotations

import logging
import warnings
from typing import Dict, Optional, Tuple, List

import numpy as np
import matplotlib.pyplot as plt

def calculate_haar_fluctuations(
    time: np.ndarray,
    data: np.ndarray,
    lag_times: Optional[np.ndarray] = None,
    min_lag: Optional[float] = None,
    max_lag: Optional[float] = None,
    num_lags: int = 20,
    log_spacing: bool = True
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Calculates the first-order Haar structure function S_1(Delta t).

    Args:
        time (np.ndarray): Array of time points (must be sorted).
        data (np.ndarray): Array of data values.
        lag_times (np.ndarray, optional): Specific lag times (Delta t) to evaluate.
            If None, a range is generated based on min_lag, max_lag, and num_lags.
        min_lag (float, optional): Minimum lag time. Defaults to minimum time difference in data.
        max_lag (float, optional): Maximum lag time. Defaults to half the total duration.
        num_lags (int, optional): Number of lag times to generate if lag_times is None.
        log_spacing (bool, optional): If True, generate logarithmically spaced lags.

    Returns:
        tuple[np.ndarray, np.ndarray, np.ndarray]:
            - lags: The actual lag times used.
            - s1: The first-order structure function values S_1(Delta t).
            - counts: The number of pairs found for each lag.
    """
    n = len(time)
    if n < 2:
        raise ValueError("Time series must have at least 2 points.")

    # Sort data by time just in case
    sort_idx = np.argsort(time)
    time = time[sort_idx]
    data = data[sort_idx]

    # Determine lag times if not provided
    total_duration = time[-1] - time[0]

    if lag_times is None:
        if min_lag is None:
            # Estimate minimum lag as the median of non-zero time differences to avoid zeros
            dt = np.diff(time)
            valid_dt = dt[dt > 0]
            min_lag = np.median(valid_dt) if len(valid_dt) > 0 else total_duration/n
        if max_lag is None:
            max_lag = total_duration / 2.0

        if max_lag > total_duration / 2:
            warnings.warn(
                f"max_lag ({max_lag}) is greater than half the total duration ({total_duration/2}). "
                "It is recommended to use a max_lag between T/4 and T/2, and Nyquist frequency for min_lag.",
                UserWarning
            )

        if log_spacing:
            lag_times = np.logspace(np.log10(min_lag), np.log10(max_lag), num_lags)
        else:
            lag_times = np.linspace(min_lag, max_lag, num_lags)

    s1_values = []
    counts = []
    valid_lags = []

    for delta_t in lag_times:
        fluctuations = []

        # Loop over the time series using indices to find intervals
        current_idx = 0
        while current_idx < n:
            t_start = time[current_idx]
            t_mid = t_start + delta_t / 2
            t_end = t_start + delta_t

            if t_end > time[-1]:
                break

            # Identify indices for the two halves
            # optimizing search using searchsorted
            idx_mid = np.searchsorted(time, t_mid, side='left')
            idx_end = np.searchsorted(time, t_end, side='left')

            # Interval 1: [current_idx, idx_mid)
            # Interval 2: [idx_mid, idx_end)

            vals1 = data[current_idx:idx_mid]
            vals2 = data[idx_mid:idx_end]

            if len(vals1) > 0 and len(vals2) > 0:
                mean1 = np.mean(vals1)
                mean2 = np.mean(vals2)
                # Note: The prompt formula included division by delta_t, but the interpretation
                # (H=-0.5 for White Noise) implies we should analyze the scaling of the difference of means directly.
                # If we divide by delta_t, H would be shifted by -1.
                # To match the interpretation: delta_f = mean2 - mean1
                delta_f = (mean2 - mean1)
                fluctuations.append(np.abs(delta_f))

                # Advance to the first point >= t_end to ensure non-overlapping
                current_idx = idx_end
            else:
                # If not enough data in this window starting at t_start,
                # move to the next data point to try starting a window there.
                current_idx += 1

        if len(fluctuations) > 0:
            s1 = np.mean(fluctuations)
            s1_values.append(s1)
            counts.append(len(fluctuations))
            valid_lags.append(delta_t)

    return np.array(valid_lags), np.array(s1_values), np.array(counts)

def fit_haar_slope(lags: np.ndarray, s1: np.ndarray) -> Tuple[float, float, float]:
    """
    Fits a power law to the structure function: S_1(dt) ~ dt^H.
    Returns H, beta, and R^2.

    beta = 1 + 2H
    """
    # Log-log fit
    # Filter out zeros or negatives if any (shouldn't be for S1 unless empty)
    valid = (lags > 0) & (s1 > 0)
    if np.sum(valid) < 3:
        return np.nan, np.nan, np.nan

    log_lags = np.log(lags[valid])
    log_s1 = np.log(s1[valid])

    slope, intercept = np.polyfit(log_lags, log_s1, 1)

    H = slope
    beta = 1 + 2 * H

    # Calculate R2
    predicted = slope * log_lags + intercept
    ss_res = np.sum((log_s1 - predicted) ** 2)
    ss_tot = np.sum((log_s1 - np.mean(log_s1)) ** 2)
    r2 = 1 - (ss_res / ss_tot)

    return H, beta, r2

def plot_haar_analysis(
    lags: np.ndarray,
    s1: np.ndarray,
    H: float,
    beta: float,
    output_path: Optional[str] = None,
    time_unit: str = "seconds"
):
    """
    Plots the Haar Structure Function analysis results.
    """
    plt.figure(figsize=(10, 6))
    plt.loglog(lags, s1, 'o-', label='Haar Structure Function $S_1(\\Delta t)$')

    # Plot fit line
    if not np.isnan(H):
        # Create fit line
        fit_vals = s1[0] * (lags / lags[0])**H
        # Adjust intercept visually to pass through mean
        log_lags = np.log(lags)
        log_s1 = np.log(s1)
        slope, intercept = np.polyfit(log_lags, log_s1, 1)
        fit_vals = np.exp(intercept) * lags**slope

        plt.loglog(lags, fit_vals, 'r--', label=f'Fit: H={H:.2f}, $\\beta$={beta:.2f}')

    plt.xlabel(f'Lag Time $\\Delta t$ ({time_unit})')
    plt.ylabel('Structure Function $S_1(\\Delta t)$')
    plt.title('Haar Structure Function Analysis')
    plt.legend()
    plt.grid(True, which="both", ls="-", alpha=0.5)

    if output_path:
        plt.savefig(output_path)
        plt.close()
    else:
        plt.show()

class HaarAnalysis:
    def __init__(self, time: np.ndarray, data: np.ndarray, time_unit: str = "seconds"):
        self.time = time
        self.data = data
        self.time_unit = time_unit
        self.lags = None
        self.s1 = None
        self.counts = None
        self.H = None
        self.beta = None
        self.r2 = None

    def run(self, min_lag=None, max_lag=None, num_lags=20, log_spacing=True):
        self.lags, self.s1, self.counts = calculate_haar_fluctuations(
            self.time, self.data, min_lag=min_lag, max_lag=max_lag, num_lags=num_lags, log_spacing=log_spacing
        )
        self.H, self.beta, self.r2 = fit_haar_slope(self.lags, self.s1)

        return {
            "H": self.H,
            "beta": self.beta,
            "r2": self.r2,
            "lags": self.lags,
            "s1": self.s1,
            "counts": self.counts
        }

    def plot(self, output_path=None):
        if self.lags is None:
            raise ValueError("Run analysis first.")
        plot_haar_analysis(self.lags, self.s1, self.H, self.beta, output_path, time_unit=self.time_unit)

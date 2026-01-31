from __future__ import annotations

import logging
import warnings
from typing import Dict, Optional, Tuple, List

import numpy as np
import matplotlib.pyplot as plt
import MannKS

def calculate_haar_fluctuations(
    time: np.ndarray,
    data: np.ndarray,
    lag_times: Optional[np.ndarray] = None,
    min_lag: Optional[float] = None,
    max_lag: Optional[float] = None,
    num_lags: int = 20,
    log_spacing: bool = True,
    overlap: bool = True,
    overlap_step_fraction: float = 0.1
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Calculates the first-order Haar structure function S_1(Delta t).

    Haar Wavelet Analysis is a robust method for estimating spectral slopes (Beta),
    particularly for unevenly sampled time series where methods like Lomb-Scargle
    might be biased. It measures the average magnitude of fluctuations at different
    timescales (lag times).

    Args:
        time (np.ndarray): Array of time points (must be sorted).
        data (np.ndarray): Array of data values.
        lag_times (np.ndarray, optional): Specific lag times (Delta t) to evaluate.
            If None, a range is generated based on min_lag, max_lag, and num_lags.
        min_lag (float, optional): Minimum lag time. Defaults to minimum time difference in data.
        max_lag (float, optional): Maximum lag time. Defaults to half the total duration.
        num_lags (int, optional): Number of lag times to generate if lag_times is None.
        log_spacing (bool, optional): If True, generate logarithmically spaced lags.
        overlap (bool, optional): If True, use overlapping windows (sliding).
            This increases sample size but introduces dependence. Defaults to True.
        overlap_step_fraction (float, optional): Fraction of lag_time to step forward
            when overlap is True. e.g. 0.1 means step size is 0.1 * delta_t.
            Defaults to 0.1.

    Returns:
        tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
            - lags: The actual lag times used.
            - s1: The first-order structure function values S_1(Delta t).
            - counts: The number of raw pairs found for each lag.
            - n_effective: The effective sample size accounting for overlap.
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
    n_effective_values = []

    for delta_t in lag_times:
        fluctuations = []

        # Determine step size
        if overlap:
            # If overlap is enabled, slide by a fraction of delta_t
            step_size = delta_t * overlap_step_fraction
            # Ensure step size is at least the smallest data interval to make progress
            # But we are in continuous time, so we just increment current_time.
            # However, we need to iterate based on available data points to be efficient?
            # No, for irregular data, we slide a continuous window.
        else:
            step_size = delta_t

        # We will iterate by sliding a window start time
        t_start = time[0]

        while t_start + delta_t <= time[-1]:
            t_mid = t_start + delta_t / 2
            t_end = t_start + delta_t

            # Identify indices for the two halves
            # Interval 1: [t_start, t_mid)
            # Interval 2: [t_mid, t_end)

            # Using searchsorted to find indices efficiently
            idx_start = np.searchsorted(time, t_start, side='left')
            idx_mid = np.searchsorted(time, t_mid, side='left')
            idx_end = np.searchsorted(time, t_end, side='left')

            vals1 = data[idx_start:idx_mid]
            vals2 = data[idx_mid:idx_end]

            # Only calculate if we have data in both halves
            # Strict requirement: need at least 1 point in each half
            # Ideally for large windows we'd want more, but that's a filtering step later
            if len(vals1) > 0 and len(vals2) > 0:
                mean1 = np.mean(vals1)
                mean2 = np.mean(vals2)

                # delta_f = mean2 - mean1 (See discussion on H vs m)
                delta_f = (mean2 - mean1)
                fluctuations.append(np.abs(delta_f))

            # Move window
            if overlap:
                # Slide by fixed time step
                t_start += step_size
            else:
                # Move to the end of current window
                t_start = t_end

                # If the next window would start after the last data point, break early
                if t_start >= time[-1]:
                    break

                # Optimization for non-overlapping with sparse data:
                # If the next window is empty (starts in a large gap), we might want to skip ahead.
                # But strict Haar definition requires contiguous windows?
                # For irregular sampling, we usually just take the next available window defined by time.
                # Let's stick to the time grid to be consistent.
                # However, if t_start is in a huge gap, we waste iterations.
                # Given 'step_size' is usually related to 'delta_t', it's fine.

        count = len(fluctuations)
        if count > 0:
            s1 = np.mean(fluctuations)
            s1_values.append(s1)
            counts.append(count)
            valid_lags.append(delta_t)

            # Calculate Effective Sample Size
            if overlap:
                # n_eff = N * (1 - overlap_fraction)
                # overlap_fraction = (delta_t - step_size) / delta_t = 1 - step_fraction
                # So n_eff approx N * step_fraction
                # But it depends on how many we actually found vs theoretical max

                # Another approximation: n_eff = count * (step_size / delta_t)
                # Because if step_size = delta_t (no overlap), n_eff = count * 1.
                # If step_size is small, each point is redundant.

                n_eff = count * (step_size / delta_t)
                n_effective_values.append(n_eff)
            else:
                n_effective_values.append(count)

    return np.array(valid_lags), np.array(s1_values), np.array(counts), np.array(n_effective_values)

def fit_haar_slope(
    lags: np.ndarray,
    s1: np.ndarray,
    ci: float = 95,
    n_bootstraps: int = 100
) -> Tuple[float, float, float, float]:
    """
    Fits a power law to the structure function: S_1(dt) ~ dt^H using
    robust regression (Mann-Kendall/Theil-Sen).

    Returns H, beta, r2, and intercept.

    beta = 1 + 2H

    Args:
        lags (np.ndarray): Lag times.
        s1 (np.ndarray): Structure function values.
        ci (float, optional): Confidence interval percentage. Defaults to 95.
        n_bootstraps (int, optional): Number of bootstraps for CI. Defaults to 100.
    """
    # Log-log fit
    # Filter out zeros or negatives if any (shouldn't be for S1 unless empty)
    valid = (lags > 0) & (s1 > 0)
    if np.sum(valid) < 3:
        return np.nan, np.nan, np.nan, np.nan

    log_lags = np.log10(lags[valid])
    log_s1 = np.log10(s1[valid])

    # Robust fit using MannKS
    res = MannKS.trend_test(
        log_s1,
        log_lags,
        alpha=1 - (ci / 100),
        n_bootstrap=n_bootstraps
    )

    H = res.slope
    intercept = res.intercept
    beta = 1 + 2 * H

    # Calculate R2 (using OLS for a traditional goodness-of-fit measure)
    # Even though we use Theil-Sen for the slope, R2 is still useful.
    slope_ols, intercept_ols = np.polyfit(log_lags, log_s1, 1)
    predicted = slope_ols * log_lags + intercept_ols
    ss_res = np.sum((log_s1 - predicted) ** 2)
    ss_tot = np.sum((log_s1 - np.mean(log_s1)) ** 2)
    r2 = 1 - (ss_res / ss_tot)

    return H, beta, r2, intercept

def fit_segmented_haar(
    lags: np.ndarray,
    s1: np.ndarray,
    n_breakpoints: int = 1,
    ci: float = 95,
    n_bootstraps: int = 100,
    min_segment_length: int = 4
) -> Dict:
    """
    Fits a segmented power law to the structure function: S_1(dt) ~ dt^H.

    Uses MannKS for segmented robust regression.

    Args:
        lags (np.ndarray): Lag times.
        s1 (np.ndarray): Structure function values.
        n_breakpoints (int): Number of breakpoints to fit.
        ci (float): Confidence interval percentage.
        n_bootstraps (int): Number of bootstraps.
        min_segment_length (int): Minimum points per segment.

    Returns:
        Dict: Results dictionary containing slopes (H), betas, breakpoints, etc.
    """
    valid = (lags > 0) & (s1 > 0)
    if np.sum(valid) < (n_breakpoints + 1) * min_segment_length:
        return {
            "failure_reason": f"Not enough valid points ({np.sum(valid)}) for segmented fit.",
            "n_breakpoints": n_breakpoints,
            "bic": np.inf
        }

    log_lags = np.log10(lags[valid])
    log_s1 = np.log10(s1[valid])

    # Use MannKS.segmented_trend_test
    try:
        res = MannKS.segmented_trend_test(
            log_s1, # Y is log_s1 (power/variance equivalent)
            log_lags, # X is log_lags
            n_breakpoints=n_breakpoints,
            alpha=1-(ci/100),
            n_bootstrap=n_bootstraps
        )

        # Parse results similar to fit_segmented_spectrum but adapted for Haar
        # Slope is H
        segments_df = res.segments

        Hs = segments_df['slope'].values
        intercepts = segments_df['intercept'].values

        # Calculate Betas: beta = 1 + 2H
        betas = 1 + 2 * Hs

        # CIs
        h_lower = segments_df['lower_ci'].values
        h_upper = segments_df['upper_ci'].values

        # Beta CIs: beta_lower = 1 + 2 * h_lower
        betas_ci = list(zip(1 + 2 * h_lower, 1 + 2 * h_upper))

        # Breakpoints (convert back to linear scale)
        breakpoints = res.breakpoints
        linear_breakpoints = 10**breakpoints if breakpoints is not None else []

        # Breakpoint CIs
        bp_cis = []
        if res.breakpoint_cis:
            for lower, upper in res.breakpoint_cis:
                bp_cis.append((10**lower, 10**upper))
        else:
            bp_cis = [(np.nan, np.nan)] * n_breakpoints

        results = {
            "bic": res.bic,
            "aic": res.aic,
            "n_breakpoints": res.n_breakpoints,
            "breakpoints": linear_breakpoints,
            "Hs": Hs,
            "betas": betas,
            "intercepts": intercepts,
            "Hs_ci": list(zip(h_lower, h_upper)),
            "betas_ci": betas_ci,
            "breakpoints_ci": bp_cis,
            "log_lags": log_lags,
            "log_s1": log_s1,
            "ci_computed": True
        }

        return results

    except Exception as e:
        return {
            "failure_reason": f"Segmented fit failed: {e}",
            "n_breakpoints": n_breakpoints,
            "bic": np.inf
        }

def plot_haar_analysis(
    lags: np.ndarray,
    s1: np.ndarray,
    H: float,
    beta: float,
    intercept: Optional[float] = None,
    output_path: Optional[str] = None,
    time_unit: str = "seconds",
    segmented_results: Optional[Dict] = None
):
    """
    Plots the Haar Structure Function analysis results.
    """
    plt.figure(figsize=(10, 6))
    plt.loglog(lags, s1, 'o-', label='Haar Structure Function $S_1(\\Delta t)$', alpha=0.6)

    # Plot standard fit line if available
    if not np.isnan(H) and segmented_results is None:
        if intercept is None:
            log_lags = np.log10(lags)
            log_s1 = np.log10(s1)
            intercept = np.median(log_s1 - H * log_lags)
            fit_vals = 10**intercept * lags**H
        else:
            fit_vals = 10**intercept * lags**H

        plt.loglog(lags, fit_vals, 'r--', label=f'Standard Fit: H={H:.2f}, $\\beta$={beta:.2f}')

    # Plot segmented fit if available
    if segmented_results and "Hs" in segmented_results:
        Hs = segmented_results["Hs"]
        intercepts = segmented_results["intercepts"]
        breakpoints = segmented_results["breakpoints"]

        # Plot each segment
        # We need to determine the range for each segment
        sorted_bp = np.sort(breakpoints)
        bounds = np.concatenate([[lags.min()], sorted_bp, [lags.max()]])

        colors = ['r', 'g', 'm']

        for i in range(len(Hs)):
            start_lag = bounds[i]
            end_lag = bounds[i+1]

            # Generate points for line
            seg_lags = np.linspace(start_lag, end_lag, 100)
            seg_vals = 10**intercepts[i] * seg_lags**Hs[i]

            label = f'Seg {i+1}: H={Hs[i]:.2f}, $\\beta$={1+2*Hs[i]:.2f}'
            plt.loglog(seg_lags, seg_vals, '--', color=colors[i % len(colors)], label=label, linewidth=2)

        # Mark breakpoints
        for bp in breakpoints:
            plt.axvline(bp, color='k', linestyle=':', alpha=0.5, label=f'Breakpoint: {bp:.1f}')

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
        self.n_effective = None
        self.H = None
        self.beta = None
        self.r2 = None
        self.intercept = None
        self.segmented_results = None

    def run(self, min_lag=None, max_lag=None, num_lags=20, log_spacing=True, n_bootstraps=100, overlap=True, overlap_step_fraction=0.1, max_breakpoints=0):
        self.lags, self.s1, self.counts, self.n_effective = calculate_haar_fluctuations(
            self.time, self.data, min_lag=min_lag, max_lag=max_lag, num_lags=num_lags, log_spacing=log_spacing,
            overlap=overlap, overlap_step_fraction=overlap_step_fraction
        )

        # Always run standard fit
        self.H, self.beta, self.r2, self.intercept = fit_haar_slope(
            self.lags, self.s1, n_bootstraps=n_bootstraps
        )

        # Run segmented fit if requested
        if max_breakpoints > 0:
            # Try fitting 1 to max_breakpoints
            # For simplicity, let's just implement finding the best up to max_breakpoints using BIC
            best_bic = np.inf
            best_results = None

            # Also consider 0 breakpoints (standard fit) for BIC comparison
            # Calculate BIC for standard fit
            log_lags = np.log10(self.lags[self.lags > 0])
            log_s1 = np.log10(self.s1[self.lags > 0])
            predicted = self.H * log_lags + self.intercept
            rss = np.sum((log_s1 - predicted) ** 2)
            n = len(log_s1)
            bic_0 = n * np.log(rss / n) + 2 * np.log(n)

            best_bic = bic_0

            for nb in range(1, max_breakpoints + 1):
                res = fit_segmented_haar(self.lags, self.s1, n_breakpoints=nb, n_bootstraps=n_bootstraps)
                if res.get("bic", np.inf) < best_bic:
                    best_bic = res["bic"]
                    best_results = res

            self.segmented_results = best_results

        return {
            "H": self.H,
            "beta": self.beta,
            "r2": self.r2,
            "intercept": self.intercept,
            "lags": self.lags,
            "s1": self.s1,
            "counts": self.counts,
            "n_effective": self.n_effective,
            "segmented_results": self.segmented_results
        }

    def plot(self, output_path=None):
        if self.lags is None:
            raise ValueError("Run analysis first.")
        plot_haar_analysis(
            self.lags, self.s1, self.H, self.beta, self.intercept,
            output_path, time_unit=self.time_unit,
            segmented_results=self.segmented_results
        )

from __future__ import annotations

import logging
import warnings
from typing import Dict, Optional, Tuple, List, Union

import numpy as np
import matplotlib.pyplot as plt
from scipy.special import gammaln
import MannKS

from .surrogates import generate_power_law_surrogates

def _small_sample_std(data: np.ndarray) -> float:
    """
    Computes the standard deviation with a correction for small sample bias,
    assuming the data is drawn from a Gaussian distribution.
    This effectively provides an unbiased estimator for the population sigma.
    """
    n = len(data)
    if n < 2:
        return np.nan

    # Sample standard deviation (ddof=1)
    s = np.std(data, ddof=1)

    # Correction for small N
    if n < 101:
        # Correction factor derived from Gamma functions
        # This factor converts the biased estimator s to an unbiased estimator of sigma
        factor = np.exp(gammaln((n - 1) / 2) - gammaln(n / 2)) * np.sqrt((n - 1) / 2)
        return s * factor
    else:
        return s

def _compute_statistic(
    data: np.ndarray,
    statistic: str = "mean",
    percentile: Optional[float] = None,
    percentile_method: str = "hazen"
) -> float:
    """
    Computes the specified statistic for an array of data.

    Args:
        data (np.ndarray): Input data array.
        statistic (str): "mean", "median", or "percentile".
        percentile (float): Percentile to compute (0-100), required if statistic="percentile".
        percentile_method (str): Method for percentile calculation (default "hazen").
                                 See numpy.percentile documentation for options.
    """
    if statistic == "mean":
        return np.mean(data)
    elif statistic == "median":
        return np.median(data)
    elif statistic == "percentile":
        if percentile is None:
            raise ValueError("percentile must be provided when statistic is 'percentile'")
        return np.percentile(data, percentile, method=percentile_method)
    else:
        raise ValueError(f"Unknown statistic: {statistic}")

def calculate_haar_fluctuations(
    time: np.ndarray,
    data: np.ndarray,
    lag_times: Optional[np.ndarray] = None,
    min_lag: Optional[float] = None,
    max_lag: Optional[float] = None,
    num_lags: int = 20,
    log_spacing: bool = True,
    overlap: bool = True,
    overlap_step_fraction: float = 0.1,
    min_samples_per_window: int = 5,
    statistic: str = "mean",
    percentile: Optional[float] = None,
    percentile_method: str = "hazen",
    aggregation: str = "mean"
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Calculates the first-order Haar structure function S_1(Delta t) and effective sample size.

    Args:
        aggregation (str): Method to aggregate fluctuations for S1 calculation.
                           "mean": Mean Absolute Fluctuation (Standard S1).
                                   Robust and distribution-agnostic, but may be biased for small samples.
                           "median": Median Absolute Fluctuation (Robust S1).
                           "rms": Root Mean Square Fluctuation (Approximates S2^0.5).
                           "std_corrected": Unbiased estimation of MAD assuming Gaussianity (matches GapWaveSpectra).
                                            Uses small-sample correction for standard deviation.

                                            Note: This method assumes the fluctuations (differences of window means)
                                            follow a Gaussian distribution. Due to the Central Limit Theorem, this
                                            assumption holds for most finite-variance processes at sufficient window
                                            sizes (N >= 5). For very small scales (N < 5) with non-Gaussian data,
                                            this may introduce a bias in the fluctuation magnitude (intercept), but
                                            the spectral slope (beta) typically remains robust.

    Returns:
        valid_lags (np.ndarray): The lag times used.
        s1_values (np.ndarray): The Haar fluctuation values S_1(lag).
        counts (np.ndarray): Number of fluctuation pairs per lag.
        n_effective_values (np.ndarray): Effective number of independent samples (adjusts for overlap).
    """
    n = len(time)
    if n < 2:
        raise ValueError("Time series must have at least 2 points.")

    if statistic not in ["mean", "median", "percentile"]:
        raise ValueError(f"Unknown statistic: {statistic}")
    if statistic == "percentile" and percentile is None:
        raise ValueError("percentile must be provided when statistic is 'percentile'")

    if aggregation not in ["mean", "median", "rms", "std_corrected"]:
        raise ValueError(f"Unknown aggregation method: {aggregation}")

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
                "It is recommended to use a max_lag between T/4 and T/2, and Nyquist frequency for min_lag. "
                "Results for lags > T/2 are statistically unreliable.",
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
            step_size = delta_t * overlap_step_fraction
        else:
            step_size = delta_t

        # We will iterate by sliding a window start time
        t_start = time[0]

        while t_start + delta_t <= time[-1]:
            t_mid = t_start + delta_t / 2
            t_end = t_start + delta_t

            idx_start = np.searchsorted(time, t_start, side='left')
            idx_mid = np.searchsorted(time, t_mid, side='left')
            idx_end = np.searchsorted(time, t_end, side='left')

            vals1 = data[idx_start:idx_mid]
            vals2 = data[idx_mid:idx_end]

            # Only calculate if we have sufficient data in both halves
            if len(vals1) >= min_samples_per_window and len(vals2) >= min_samples_per_window:
                val1 = _compute_statistic(vals1, statistic, percentile, percentile_method)
                val2 = _compute_statistic(vals2, statistic, percentile, percentile_method)
                delta_f = (val2 - val1)
                fluctuations.append(delta_f)

            # Move window
            if overlap:
                t_start += step_size
            else:
                t_start = t_end
                if t_start >= time[-1]:
                    break

        count = len(fluctuations)
        if count > 0:
            flucs_arr = np.array(fluctuations)

            if aggregation == "mean":
                s1 = np.mean(np.abs(flucs_arr))
            elif aggregation == "median":
                s1 = np.median(np.abs(flucs_arr))
            elif aggregation == "rms":
                s1 = np.sqrt(np.mean(flucs_arr**2))
            elif aggregation == "std_corrected":
                # Matches GapWaveSpectra approach:
                # 1. Enforce zero mean by concatenating flucs and -flucs
                # 2. Use small-sample corrected standard deviation of the combined set
                # 3. Convert sigma to MAD (Mean Absolute Deviation) assuming Gaussianity
                #    MAD = sigma * sqrt(2/pi)

                # Note: GapWaveSpectra concatenates (flucs, -flucs).
                # This doubles the effective sample size and enforces mean=0.
                combined = np.concatenate((flucs_arr, -flucs_arr))
                sigma_est = _small_sample_std(combined)
                s1 = sigma_est * np.sqrt(2 / np.pi)

            s1_values.append(s1)
            counts.append(count)
            valid_lags.append(delta_t)

            # Calculate Effective Sample Size
            if overlap:
                # Approximate n_eff based on redundancy
                n_eff = count * (step_size / delta_t)
                n_effective_values.append(n_eff)
            else:
                n_effective_values.append(count)

    return np.array(valid_lags), np.array(s1_values), np.array(counts), np.array(n_effective_values)

def calculate_sliding_haar(
    time: np.ndarray,
    data: np.ndarray,
    window_size: float,
    step_size: Optional[float] = None,
    min_samples_per_window: int = 5,
    statistic: str = "mean",
    percentile: Optional[float] = None,
    percentile_method: str = "hazen"
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Calculates a continuous time series of Haar fluctuations.
    """
    if statistic not in ["mean", "median", "percentile"]:
        raise ValueError(f"Unknown statistic: {statistic}")
    if statistic == "percentile" and percentile is None:
        raise ValueError("percentile must be provided when statistic is 'percentile'")

    if step_size is None:
        step_size = window_size / 10.0

    sort_idx = np.argsort(time)
    time = time[sort_idx]
    data = data[sort_idx]

    fluctuations = []
    t_centers = []

    t_start = time[0]
    while t_start + window_size <= time[-1]:
        t_mid = t_start + window_size / 2
        idx_start = np.searchsorted(time, t_start, side='left')
        idx_mid = np.searchsorted(time, t_mid, side='left')
        idx_end = np.searchsorted(time, t_start + window_size, side='left')

        vals1 = data[idx_start:idx_mid]
        vals2 = data[idx_mid:idx_end]

        if len(vals1) >= min_samples_per_window and len(vals2) >= min_samples_per_window:
            val1 = _compute_statistic(vals1, statistic, percentile, percentile_method)
            val2 = _compute_statistic(vals2, statistic, percentile, percentile_method)
            d = val2 - val1
            fluctuations.append(d)
            t_centers.append(t_mid)

        t_start += step_size

    return np.array(t_centers), np.array(fluctuations)

def fit_haar_slope(
    lags: np.ndarray,
    s1: np.ndarray,
    ci: float = 95,
    n_bootstraps: int = 100
) -> Dict:
    """
    Fits a power law to the structure function: S_1(dt) ~ dt^H.

    Returns a dictionary with results.
    """
    valid = (lags > 0) & (s1 > 0)
    if np.sum(valid) < 3:
        return {"beta": np.nan, "H": np.nan, "r2": np.nan, "intercept": np.nan}

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

    # Calculate R2
    slope_ols, intercept_ols = np.polyfit(log_lags, log_s1, 1)
    predicted = slope_ols * log_lags + intercept_ols
    ss_res = np.sum((log_s1 - predicted) ** 2)
    ss_tot = np.sum((log_s1 - np.mean(log_s1)) ** 2)
    r2 = 1 - (ss_res / ss_tot)

    return {
        "H": H,
        "beta": beta,
        "r2": r2,
        "intercept": intercept,
        "slope_ci_lower": res.lower_ci, # This is H_lower
        "slope_ci_upper": res.upper_ci, # This is H_upper
        "beta_ci_lower": 1 + 2 * res.lower_ci,
        "beta_ci_upper": 1 + 2 * res.upper_ci
    }

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

    try:
        res = MannKS.segmented_trend_test(
            log_s1,
            log_lags,
            n_breakpoints=n_breakpoints,
            alpha=1-(ci/100),
            n_bootstrap=n_bootstraps
        )

        segments_df = res.segments

        Hs = segments_df['slope'].values
        intercepts = segments_df['intercept'].values
        betas = 1 + 2 * Hs

        h_lower = segments_df['lower_ci'].values
        h_upper = segments_df['upper_ci'].values
        betas_ci = list(zip(1 + 2 * h_lower, 1 + 2 * h_upper))

        breakpoints = res.breakpoints
        linear_breakpoints = 10**breakpoints if breakpoints is not None else []

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

        sorted_bp = np.sort(breakpoints)
        bounds = np.concatenate([[lags.min()], sorted_bp, [lags.max()]])

        colors = ['r', 'g', 'm']

        for i in range(len(Hs)):
            start_lag = bounds[i]
            end_lag = bounds[i+1]
            seg_lags = np.linspace(start_lag, end_lag, 100)
            seg_vals = 10**intercepts[i] * seg_lags**Hs[i]

            label = f'Seg {i+1}: H={Hs[i]:.2f}, $\\beta$={1+2*Hs[i]:.2f}'
            plt.loglog(seg_lags, seg_vals, '--', color=colors[i % len(colors)], label=label, linewidth=2)

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
        self.full_results = {} # Store full dictionary

    def run(self, min_lag=None, max_lag=None, num_lags=20, log_spacing=True, n_bootstraps=100, overlap=True, overlap_step_fraction=0.1, max_breakpoints=0, min_samples_per_window=5, bootstrap_method="standard", seed=None, statistic="mean", percentile=None, percentile_method="hazen", aggregation="mean"):
        """
        Runs the Haar analysis.

        Args:
            bootstrap_method (str): "standard" (MannKS fit bootstrap) or "monte_carlo" (Parametric bootstrap on time series).
                                    "monte_carlo" is recommended for irregular data to rigorously estimate spectral uncertainty.
            seed (int): Random seed for bootstrap.
            statistic (str): Statistic to use for window aggregation ("mean", "median", "percentile").
            percentile (float): Percentile to compute if statistic is "percentile".
            percentile_method (str): Method for percentile calculation (default "hazen").
            aggregation (str): Method to aggregate fluctuations ("mean", "median", "rms", "std_corrected").
                               "std_corrected" is recommended for Gaussian data or sufficiently large windows
                               to avoid small-sample bias.
        """
        self.lags, self.s1, self.counts, self.n_effective = calculate_haar_fluctuations(
            self.time, self.data, min_lag=min_lag, max_lag=max_lag, num_lags=num_lags, log_spacing=log_spacing,
            overlap=overlap, overlap_step_fraction=overlap_step_fraction, min_samples_per_window=min_samples_per_window,
            statistic=statistic, percentile=percentile, percentile_method=percentile_method, aggregation=aggregation
        )

        # Run standard fit. If monte_carlo, use 0 bootstraps for the initial fit to save time (we will bootstrap later).
        # Actually, let's keep n_bootstraps for standard even if monte_carlo, so we have a comparison?
        # No, if monte_carlo is chosen, we want the "defensible" CI to override the standard one.

        initial_bootstraps = n_bootstraps if bootstrap_method == "standard" else 0

        fit_results = fit_haar_slope(
            self.lags, self.s1, n_bootstraps=initial_bootstraps
        )

        self.H = fit_results.get("H", np.nan)
        self.beta = fit_results.get("beta", np.nan)
        self.r2 = fit_results.get("r2", np.nan)
        self.intercept = fit_results.get("intercept", np.nan)

        if bootstrap_method == "monte_carlo" and not np.isnan(self.beta):
            # Parametric Bootstrap (Monte Carlo)
            # 1. Generate N surrogates with the estimated beta
            surrogates = generate_power_law_surrogates(
                self.time, self.beta, n_surrogates=n_bootstraps, seed=seed
            )

            betas_boot = []
            Hs_boot = []

            for surr in surrogates:
                # Calculate S1 for surrogate
                lags_b, s1_b, _, _ = calculate_haar_fluctuations(
                    self.time, surr,
                    lag_times=self.lags, # Use same lags
                    overlap=overlap,
                    overlap_step_fraction=overlap_step_fraction,
                    min_samples_per_window=min_samples_per_window,
                    statistic=statistic, percentile=percentile, percentile_method=percentile_method,
                    aggregation=aggregation
                )

                # Fit slope (no bootstrap needed here, just the slope)
                res_b = fit_haar_slope(lags_b, s1_b, n_bootstraps=0)

                if not np.isnan(res_b['beta']):
                    betas_boot.append(res_b['beta'])
                    Hs_boot.append(res_b['H'])

            betas_boot = np.array(betas_boot)
            Hs_boot = np.array(Hs_boot)

            if len(betas_boot) > 10:
                std_beta = np.std(betas_boot)
                std_H = np.std(Hs_boot)

                # Update CIs using 1.96 * std (approx 95%)
                # Center around the OBSERVED estimate
                fit_results['beta_ci_lower'] = self.beta - 1.96 * std_beta
                fit_results['beta_ci_upper'] = self.beta + 1.96 * std_beta

                fit_results['slope_ci_lower'] = self.H - 1.96 * std_H
                fit_results['slope_ci_upper'] = self.H + 1.96 * std_H

                # Store full distribution for advanced users
                fit_results['boot_betas'] = betas_boot
                fit_results['boot_Hs'] = Hs_boot
                fit_results['bootstrap_method'] = 'monte_carlo'

        # Run segmented fit if requested
        if max_breakpoints > 0:
            best_bic = np.inf
            best_results = None

            # Calculate BIC for standard fit (0 breakpoints)
            valid = (self.lags > 0) & (self.s1 > 0)
            if np.sum(valid) > 2:
                log_lags = np.log10(self.lags[valid])
                log_s1 = np.log10(self.s1[valid])
                predicted = self.H * log_lags + self.intercept
                rss = np.sum((log_s1 - predicted) ** 2)
                n = len(log_s1)
                bic_0 = n * np.log(rss / n) + 2 * np.log(n)
                best_bic = bic_0

            for nb in range(1, max_breakpoints + 1):
                # Note: Segmented fit still uses standard MannKS bootstrap for now.
                # Monte Carlo for segmented fit is very expensive and complex (finding breakpoints in surrogates).
                res = fit_segmented_haar(self.lags, self.s1, n_breakpoints=nb, n_bootstraps=n_bootstraps)
                if res.get("bic", np.inf) < best_bic:
                    best_bic = res["bic"]
                    best_results = res

            self.segmented_results = best_results

        # Construct full result dictionary merging everything
        self.full_results = {
            **fit_results, # Merge H, beta, r2, intercept, CIs
            "lags": self.lags,
            "s1": self.s1,
            "counts": self.counts,
            "n_effective": self.n_effective,
            "segmented_results": self.segmented_results,
            "statistic": statistic
        }

        return self.full_results

    def plot(self, output_path=None):
        if self.lags is None:
            raise ValueError("Run analysis first.")
        plot_haar_analysis(
            self.lags, self.s1, self.H, self.beta, self.intercept,
            output_path, time_unit=self.time_unit,
            segmented_results=self.segmented_results
        )


import numpy as np
import pandas as pd
from typing import Tuple, Optional, Dict
import logging
import warnings
from scipy import stats, interpolate

from .haar_analysis import calculate_haar_fluctuations, _compute_statistic
from .surrogates import generate_phase_randomized_surrogates, calculate_significance_p_value, generate_power_law_surrogates, generate_iaaft_surrogates
from .ls_cross_spectrum import calculate_ls_cross_spectrum, calculate_time_lag

class BivariateAnalysis:
    """
    Performs bivariate analysis between two time series (e.g., Concentration and Discharge).
    Supports Cross-Haar Correlation, Lagged Response Analysis, and Cross-Spectral Analysis.
    """

    def __init__(self,
                 time1: np.ndarray, data1: np.ndarray, name1: str,
                 time2: np.ndarray, data2: np.ndarray, name2: str,
                 time_unit: str = "seconds"):
        self.time1 = time1
        self.data1 = data1
        self.name1 = name1

        self.time2 = time2
        self.data2 = data2
        self.name2 = name2

        self.time_unit = time_unit
        self.aligned_data = None
        self.logger = logging.getLogger(__name__)

    def align_data(self, tolerance: float, method: str = 'nearest') -> pd.DataFrame:
        """
        Aligns the two time series to a common timeline.

        Args:
            tolerance (float): Maximum time difference to consider a match.
            method (str): Alignment method.
                'nearest': Finds nearest neighbor within tolerance.
                'interpolate_2_to_1': Interpolates series 2 to match series 1 times.
        """
        df1 = pd.DataFrame({'time': self.time1, self.name1: self.data1})
        df2 = pd.DataFrame({'time': self.time2, self.name2: self.data2})

        if method == 'interpolate_2_to_1':
            # Interpolate data2 onto time1
            # Assuming strictly increasing time
            interp_vals = np.interp(self.time1, self.time2, self.data2, left=np.nan, right=np.nan)

            # Create aligned DF
            aligned = df1.copy()
            aligned[self.name2] = interp_vals

            # Mask out points where nearest neighbor in time2 is too far
            idx = np.searchsorted(self.time2, self.time1)
            idx = np.clip(idx, 0, len(self.time2)-1)
            dist_right = np.abs(self.time2[idx] - self.time1)
            dist_left = np.abs(self.time2[np.clip(idx-1, 0, len(self.time2)-1)] - self.time1)
            min_dist = np.minimum(dist_left, dist_right)

            aligned.loc[min_dist > tolerance, self.name2] = np.nan

            self.aligned_data = aligned.dropna()

        elif method == 'nearest':
            # Use pandas merge_asof
            df1 = df1.sort_values('time')
            df2 = df2.sort_values('time')

            # Handle tolerance based on column types
            # If time is numeric (float/int), tolerance is numeric.
            # If time is datetime, tolerance should be Timedelta.

            tol = tolerance
            if pd.api.types.is_datetime64_any_dtype(df1['time']) and self.time_unit == 'seconds':
                 tol = pd.Timedelta(seconds=tolerance)

            aligned = pd.merge_asof(
                df1, df2, on='time',
                tolerance=tol,
                direction='nearest'
            )
            self.aligned_data = aligned.dropna()

        else:
            raise ValueError(f"Unknown alignment method: {method}")

        return self.aligned_data

    @staticmethod
    def _calculate_cross_haar(
        time: np.ndarray,
        val1: np.ndarray,
        val2: np.ndarray,
        lags: np.ndarray,
        overlap: bool = True,
        overlap_step_fraction: float = 0.1,
        min_samples_per_window: int = 5,
        statistic1: str = "mean",
        percentile1: Optional[float] = None,
        percentile_method1: str = "hazen",
        statistic2: str = "mean",
        percentile2: Optional[float] = None,
        percentile_method2: str = "hazen"
    ) -> Dict:
        """Helper to calculate Cross-Haar Correlation."""
        results = {
            'lags': [],
            'correlation': [],
            'n_pairs': [],
            'slope_alpha': [] # sensitivity
        }

        # Pre-calculate time range
        t_min, t_max = time[0], time[-1]

        for tau in lags:
            fluc1 = []
            fluc2 = []

            step_size = tau * overlap_step_fraction if overlap else tau

            # Generate window boundaries
            n_windows_max = int(np.floor((t_max - t_min - tau) / step_size)) + 1
            if n_windows_max > 0:
                t_starts = t_min + np.arange(n_windows_max) * step_size
                tol = tau * 1e-9
                t_starts = t_starts[t_starts + tau <= t_max + tol]
            else:
                t_starts = np.array([])

            if len(t_starts) > 0:
                t_mids = t_starts + tau / 2
                t_ends = t_starts + tau

                # Vectorized searchsorted for all windows at this lag
                idx_starts = np.searchsorted(time, t_starts, side='left')
                idx_mids = np.searchsorted(time, t_mids, side='left')
                idx_ends = np.searchsorted(time, t_ends, side='left')

                for i in range(len(t_starts)):
                    idx_start = idx_starts[i]
                    idx_mid = idx_mids[i]
                    idx_end = idx_ends[i]

                    # Extract window data for both variables
                    v1_left = val1[idx_start:idx_mid]
                    v1_right = val1[idx_mid:idx_end]

                    v2_left = val2[idx_start:idx_mid]
                    v2_right = val2[idx_mid:idx_end]

                    # Require sufficient data in both halves for BOTH variables
                    if (len(v1_left) >= min_samples_per_window and len(v1_right) >= min_samples_per_window and
                        len(v2_left) >= min_samples_per_window and len(v2_right) >= min_samples_per_window):
                        # Use helper for flexible stats
                        stat1_r = _compute_statistic(v1_right, statistic1, percentile1, percentile_method1)
                        stat1_l = _compute_statistic(v1_left, statistic1, percentile1, percentile_method1)
                        d1 = stat1_r - stat1_l

                        stat2_r = _compute_statistic(v2_right, statistic2, percentile2, percentile_method2)
                        stat2_l = _compute_statistic(v2_left, statistic2, percentile2, percentile_method2)
                        d2 = stat2_r - stat2_l

                        fluc1.append(d1)
                        fluc2.append(d2)

            # Need at least 2 points for correlation
            if len(fluc1) >= 2:
                if np.std(fluc1, ddof=1) < 1e-12 or np.std(fluc2, ddof=1) < 1e-12:
                    # At least one variable is constant at this scale — correlation undefined
                    results['lags'].append(tau)
                    results['correlation'].append(np.nan)
                    results['n_pairs'].append(len(fluc1))
                    results['slope_alpha'].append(np.nan)
                    continue

                corr = np.corrcoef(fluc1, fluc2)[0, 1]
                # Alpha (sensitivity): slope of regression dC ~ dQ
                # dC = alpha * dQ + eps
                slope, _, _, _, _ = stats.linregress(fluc2, fluc1)

                results['lags'].append(tau)
                results['correlation'].append(corr)
                results['n_pairs'].append(len(fluc1))
                results['slope_alpha'].append(slope)
            else:
                # Still append result for this lag, but NaNs
                results['lags'].append(tau)
                results['correlation'].append(np.nan)
                results['n_pairs'].append(len(fluc1))
                results['slope_alpha'].append(np.nan)

        return results

    def run_cross_haar_analysis(
        self,
        lags: np.ndarray,
        overlap: bool = True,
        overlap_step_fraction: float = 0.1,
        min_samples_per_window: int = 5,
        statistic1: str = "mean",
        percentile1: Optional[float] = None,
        percentile_method1: str = "hazen",
        statistic2: str = "mean",
        percentile2: Optional[float] = None,
        percentile_method2: str = "hazen"
    ) -> Dict:
        """
        Calculates Cross-Haar Correlation at specified lags.
        """
        if self.aligned_data is None:
            raise ValueError("Data must be aligned first using `align_data`.")

        time = self.aligned_data['time'].values
        val1 = self.aligned_data[self.name1].values
        val2 = self.aligned_data[self.name2].values

        return self._calculate_cross_haar(
            time, val1, val2, lags, overlap, overlap_step_fraction, min_samples_per_window,
            statistic1, percentile1, percentile_method1,
            statistic2, percentile2, percentile_method2
        )

    def calculate_significance(
        self,
        lags: np.ndarray,
        n_surrogates: int = 100,
        overlap: bool = True,
        overlap_step_fraction: float = 0.1,
        min_samples_per_window: int = 5,
        seed: Optional[int] = None,
        max_gap: Optional[float] = None,
        statistic1: str = "mean",
        percentile1: Optional[float] = None,
        percentile_method1: str = "hazen",
        statistic2: str = "mean",
        percentile2: Optional[float] = None,
        percentile_method2: str = "hazen"
    ) -> Dict:
        """
        Calculates significance of Cross-Haar Correlation using phase-randomized surrogates.
        """
        if self.aligned_data is None:
            raise ValueError("Data must be aligned first using `align_data`.")

        # Warning about surrogates and non-mean statistics
        if statistic1 != "mean" or statistic2 != "mean":
            warnings.warn(
                "Using phase-randomized surrogates with non-mean statistics (e.g. percentiles) "
                "may be statistically invalid if the process is non-Gaussian, as phase randomization "
                "imposes a Gaussian distribution on the surrogates. Use with caution.",
                UserWarning
            )

        time = self.aligned_data['time'].values
        val1 = self.aligned_data[self.name1].values # Keep var1 fixed
        val2 = self.aligned_data[self.name2].values

        if len(val2) < 10:
             return {'error': 'Insufficient data for surrogates'}

        # Run observed analysis
        obs_results = self._calculate_cross_haar(
            time, val1, val2, lags, overlap, overlap_step_fraction, min_samples_per_window,
            statistic1, percentile1, percentile_method1,
            statistic2, percentile2, percentile_method2
        )
        obs_corrs = np.array(obs_results['correlation'])

        # --- Handle Sampling for Surrogates ---
        dt = np.diff(time)
        valid_dt = dt[dt > 0]
        if len(valid_dt) == 0:
             return {'error': 'Insufficient time variation for surrogates'}
        median_dt = np.median(valid_dt)

        if max_gap is None:
            max_gap = 5.0 * median_dt

        is_irregular = not np.allclose(dt, median_dt, rtol=0.05)

        warning_flags = []
        if is_irregular and np.max(dt) > max_gap:
             msg = f"Large data gap ({np.max(dt):.2f}) detected."
             warnings.warn(msg, UserWarning)
             warning_flags.append(msg)

        surr_corrs = np.zeros((n_surrogates, len(lags)))

        if is_irregular:
            from .haar_analysis import HaarAnalysis

            # Estimate spectral slope of val2
            ha = HaarAnalysis(time, val2)
            res_ha = ha.run(num_lags=20, n_bootstraps=0)
            beta_val2 = res_ha.get("beta", 1.0)
            if np.isnan(beta_val2):
                beta_val2 = 1.0

            surrogates_val2 = generate_power_law_surrogates(
                time, beta=beta_val2, n_surrogates=n_surrogates, seed=seed
            )

            # Restore original variance and mean
            stds = surrogates_val2.std(axis=1, keepdims=True)
            zero_mask = stds.ravel() < 1e-12
            if np.any(zero_mask):
                warnings.warn(f"{zero_mask.sum()} degenerate surrogates (zero variance) dropped.", UserWarning)
                surrogates_val2 = surrogates_val2[~zero_mask]

            target_std = np.std(val2, ddof=1)
            # Rescale all surrogates ONCE using a single scale factor.
            # Using the root mean of surrogate variances (not per-surrogate)
            # preserves the relative amplitude structure within the null distribution
            # while mathematically ensuring the expected variance matches the target.
            grand_var = np.mean([s.var(ddof=1) for s in surrogates_val2])
            if grand_var > 1e-24:
                scale = target_std / np.sqrt(grand_var)
                surrogates_val2 = surrogates_val2 * scale + np.mean(val2)

            # Update n_surrogates in case some were dropped
            n_surrogates_valid = surrogates_val2.shape[0]
            surr_corrs = np.zeros((n_surrogates_valid, len(lags)))

            for i in range(n_surrogates_valid):
                res = self._calculate_cross_haar(
                    time, val1, surrogates_val2[i], lags, overlap, overlap_step_fraction, min_samples_per_window,
                    statistic1, percentile1, percentile_method1,
                    statistic2, percentile2, percentile_method2
                )
                surr_corrs[i, :] = res['correlation']
        else:
            # Evenly sampled, use IAAFT directly
            surrogates_val2 = generate_iaaft_surrogates(
                val2, n_surrogates=n_surrogates, seed=seed
            )

            stds = surrogates_val2.std(axis=1, keepdims=True)
            zero_mask = stds.ravel() < 1e-12
            if np.any(zero_mask):
                warnings.warn(f"{zero_mask.sum()} degenerate surrogates (zero variance) dropped.", UserWarning)
                surrogates_val2 = surrogates_val2[~zero_mask]

            n_surrogates_valid = surrogates_val2.shape[0]
            surr_corrs = np.zeros((n_surrogates_valid, len(lags)))

            for i in range(n_surrogates_valid):
                res = self._calculate_cross_haar(
                    time, val1, surrogates_val2[i], lags, overlap, overlap_step_fraction, min_samples_per_window,
                    statistic1, percentile1, percentile_method1,
                    statistic2, percentile2, percentile_method2
                )
                surr_corrs[i, :] = res['correlation']

        # Calculate p-values per lag
        p_values = []
        for j in range(len(lags)):
            obs = obs_corrs[j]
            dist = surr_corrs[:, j]
            if np.isnan(obs) or np.all(np.isnan(dist)):
                p_values.append(np.nan)
            else:
                p_val = calculate_significance_p_value(obs, dist, two_sided=True)
                p_values.append(p_val)

        return {
            'lags': lags,
            'observed_correlation': obs_corrs,
            'p_values': np.array(p_values),
            'surrogate_correlations': surr_corrs,
            'warning_flags': warning_flags
        }

    def run_lagged_cross_haar(
        self,
        tau: float,
        lag_offsets: np.ndarray,
        overlap: bool = True,
        overlap_step_fraction: float = 0.1,
        min_samples_per_window: int = 5,
        statistic1: str = "mean",
        percentile1: Optional[float] = None,
        percentile_method1: str = "hazen",
        statistic2: str = "mean",
        percentile2: Optional[float] = None,
        percentile_method2: str = "hazen"
    ) -> Dict:
        """
        Calculates Lagged Cross-Haar Correlation for a FIXED scale tau,
        varying the lag 'ell' between the series.

        rho(tau, ell) = corr( Delta C(t, tau), Delta Q(t - ell, tau) )
        """
        if self.aligned_data is None:
             raise ValueError("Data must be aligned first.")

        time = self.aligned_data['time'].values
        val1 = self.aligned_data[self.name1].values # C
        val2 = self.aligned_data[self.name2].values # Q

        correlations = []

        # We generate the "base" fluctuation series for C (at lag 0)
        # Store (t_center, delta_C)

        t_centers = []
        fluc1_vals = []

        step_size = tau * overlap_step_fraction if overlap else tau

        n_windows_max = int(np.floor((time[-1] - time[0] - tau) / step_size)) + 1
        if n_windows_max > 0:
            t_starts = time[0] + np.arange(n_windows_max) * step_size
            tol = tau * 1e-9
            t_starts = t_starts[t_starts + tau <= time[-1] + tol]
        else:
            t_starts = np.array([])

        if len(t_starts) > 0:
            t_mids = t_starts + tau / 2
            t_ends = t_starts + tau

            idx_starts = np.searchsorted(time, t_starts, side='left')
            idx_mids = np.searchsorted(time, t_mids, side='left')
            idx_ends = np.searchsorted(time, t_ends, side='left')

            for i in range(len(t_starts)):
                idx_start = idx_starts[i]
                idx_mid = idx_mids[i]
                idx_end = idx_ends[i]

                v1_left = val1[idx_start:idx_mid]
                v1_right = val1[idx_mid:idx_end]

                if len(v1_left) >= min_samples_per_window and len(v1_right) >= min_samples_per_window:
                    s1_r = _compute_statistic(v1_right, statistic1, percentile1, percentile_method1)
                    s1_l = _compute_statistic(v1_left, statistic1, percentile1, percentile_method1)
                    d1 = s1_r - s1_l

                    t_centers.append(t_mids[i]) # Use mid point as reference
                    fluc1_vals.append(d1)

        fluc1_vals = np.array(fluc1_vals)
        t_centers = np.array(t_centers)

        if len(fluc1_vals) < 5:
            return {'lags': lag_offsets, 'correlation': [np.nan]*len(lag_offsets)}

        # Now for each lag offset, compute Q fluctuations
        for ell in lag_offsets:
            # We want Q window centered at t_center - ell
            # Window is [t_center - ell - tau/2, t_center - ell + tau/2]

            fluc2_vals = []
            valid_indices = [] # Indices in fluc1_vals that have a matching Q pair

            t_q_mids = t_centers - ell
            t_q_starts = t_q_mids - tau/2
            t_q_ends = t_q_mids + tau/2

            # Vectorized searchsorted for this lag offset
            idx_q_starts = np.searchsorted(time, t_q_starts, side='left')
            idx_q_mids = np.searchsorted(time, t_q_mids, side='left')
            idx_q_ends = np.searchsorted(time, t_q_ends, side='left')

            for i in range(len(t_centers)):
                # Check bounds
                eps = np.finfo(float).eps * (time[-1] - time[0]) * 10
                if t_q_starts[i] < time[0] - eps or t_q_ends[i] > time[-1] + eps:
                    continue

                idx_q_start = idx_q_starts[i]
                idx_q_mid = idx_q_mids[i]
                idx_q_end = idx_q_ends[i]

                v2_left = val2[idx_q_start:idx_q_mid]
                v2_right = val2[idx_q_mid:idx_q_end]

                if len(v2_left) >= min_samples_per_window and len(v2_right) >= min_samples_per_window:
                    s2_r = _compute_statistic(v2_right, statistic2, percentile2, percentile_method2)
                    s2_l = _compute_statistic(v2_left, statistic2, percentile2, percentile_method2)
                    d2 = s2_r - s2_l

                    fluc2_vals.append(d2)
                    valid_indices.append(i)

            if len(fluc2_vals) > 2:
                # Correlate matched pairs
                c1 = fluc1_vals[valid_indices]
                c2 = np.array(fluc2_vals)
                if np.std(c1) < 1e-12 or np.std(c2) < 1e-12:
                    correlations.append(np.nan)
                else:
                    corr = np.corrcoef(c1, c2)[0, 1]
                    correlations.append(corr)
            else:
                correlations.append(np.nan)

        return {
            'tau': tau,
            'lag_offsets': lag_offsets,
            'correlation': correlations
        }

    def calculate_hysteresis_metrics(
        self,
        tau: float,
        overlap: bool = True,
        overlap_step_fraction: float = 0.1,
        min_samples_per_window: int = 5,
        statistic1: str = "mean",
        percentile1: Optional[float] = None,
        percentile_method1: str = "hazen",
        statistic2: str = "mean",
        percentile2: Optional[float] = None,
        percentile_method2: str = "hazen"
    ) -> Dict:
        """
        Calculates the Hysteresis Loop Area between fluctuations of the two variables at scale tau.
        Uses the shoelace formula (signed polygon area).

        Args:
            tau (float): Time scale.

        Returns:
            Dict: {'area': float, 'normalized_area': float, 'direction': str}
        """
        if self.aligned_data is None:
            raise ValueError("Data must be aligned first.")

        time = self.aligned_data['time'].values
        val1 = self.aligned_data[self.name1].values
        val2 = self.aligned_data[self.name2].values

        fluc1 = [] # x coordinate (usually C)
        fluc2 = [] # y coordinate (usually Q)

        step_size = tau * overlap_step_fraction if overlap else tau

        n_windows_max = int(np.floor((time[-1] - time[0] - tau) / step_size)) + 1
        if n_windows_max > 0:
            t_starts = time[0] + np.arange(n_windows_max) * step_size
            tol = tau * 1e-9
            t_starts = t_starts[t_starts + tau <= time[-1] + tol]
        else:
            t_starts = np.array([])

        if len(t_starts) > 0:
            t_mids = t_starts + tau / 2
            t_ends = t_starts + tau

            # Vectorized searchsorted for all windows at this scale
            idx_starts = np.searchsorted(time, t_starts, side='left')
            idx_mids = np.searchsorted(time, t_mids, side='left')
            idx_ends = np.searchsorted(time, t_ends, side='left')

            for i in range(len(t_starts)):
                idx_start = idx_starts[i]
                idx_mid = idx_mids[i]
                idx_end = idx_ends[i]

                v1_left = val1[idx_start:idx_mid]
                v1_right = val1[idx_mid:idx_end]
                v2_left = val2[idx_start:idx_mid]
                v2_right = val2[idx_mid:idx_end]

                if (len(v1_left) >= min_samples_per_window and len(v1_right) >= min_samples_per_window and
                    len(v2_left) >= min_samples_per_window and len(v2_right) >= min_samples_per_window):
                    s1_r = _compute_statistic(v1_right, statistic1, percentile1, percentile_method1)
                    s1_l = _compute_statistic(v1_left, statistic1, percentile1, percentile_method1)
                    d1 = s1_r - s1_l

                    s2_r = _compute_statistic(v2_right, statistic2, percentile2, percentile_method2)
                    s2_l = _compute_statistic(v2_left, statistic2, percentile2, percentile_method2)
                    d2 = s2_r - s2_l

                    fluc1.append(d1)
                    fluc2.append(d2)

        if len(fluc1) < 3:
            return {'area': np.nan, 'normalized_area': np.nan, 'direction': 'insufficient_data'}

        # Shoelace formula for signed area
        # A = 0.5 * sum(x_i * y_{i+1} - x_{i+1} * y_i)
        # Here x = fluc2 (Q, independent), y = fluc1 (C, dependent).

        x = np.array(fluc2)
        y = np.array(fluc1)

        # Close the loop
        x_closed = np.append(x, x[0])
        y_closed = np.append(y, y[0])

        area = 0.5 * np.sum(x_closed[:-1] * y_closed[1:] - x_closed[1:] * y_closed[:-1])

        # Zuecco et al. (2016) normalized hysteresis index
        std_x = np.std(x)
        std_y = np.std(y)
        if std_x > 0 and std_y > 0:
            # The total time span evaluated for this tau
            t_span = time[-1] - time[0]
            # Estimate the number of full cycles observed across the entire time series.
            # Zuecco normalization is defined per-event loop. To normalize a continuously
            # accumulating multi-event area, we must divide by the number of phase cycles.
            estimated_cycles = max(1.0, t_span / tau)
            normalized_area = area / (std_x * std_y * estimated_cycles)
        else:
            normalized_area = np.nan

        direction = "Counter-Clockwise" if area > 0 else "Clockwise"
        if np.isclose(area, 0): direction = "None"

        return {'area': area, 'normalized_area': normalized_area, 'direction': direction}

    def run_ls_cross_analysis(
        self,
        freqs: np.ndarray,
        errors1: Optional[np.ndarray] = None,
        errors2: Optional[np.ndarray] = None
    ) -> Dict:
        """
        Calculates Lomb-Scargle Cross-Spectrum and Phase directly on irregular data.
        This is the statistically defensible method for phase estimation on uneven series.

        Args:
            freqs (np.ndarray): Frequencies to analyze.
            errors1 (np.ndarray, optional): Errors for first series.
            errors2 (np.ndarray, optional): Errors for second series.

        Returns:
            Dict: Contains 'cross_power', 'phase_lag', 'time_lag', 'freqs'.
        """
        # No alignment needed! Using original timestamps.

        cross_power, phase_lag, _, _ = calculate_ls_cross_spectrum(
            self.time1, self.data1,
            self.time2, self.data2,
            freqs, errors1, errors2
        )

        time_lag = calculate_time_lag(phase_lag, freqs)

        return {
            'freqs': freqs,
            'cross_power': cross_power,
            'phase_lag': phase_lag,
            'time_lag': time_lag
        }

    def calculate_spectral_coherence(
        self,
        min_freq: Optional[float] = None,
        max_freq: Optional[float] = None,
        samples_per_peak: int = 5
    ) -> Dict:
        """
        Calculates Magnitude-Squared Coherence (MSC) using interpolation and Welch's method.

        .. warning::
            **Interpolation Artifacts**

            This method interpolates data to a regular grid before calculating coherence.
            For strictly irregular data, this can introduce spectral artifacts.

            **Recommended Alternatives:**
            - For **Phase/Time Lag** analysis, use `run_ls_cross_analysis` (Lomb-Scargle Cross Spectrum).
            - For **Coherence** (time-localized), use `run_wwz_coherence_analysis` (WWZ Coherence).

            These alternatives operate directly on the irregular data without interpolation.

        Args:
            min_freq (float, optional): Min frequency.
            max_freq (float, optional): Max frequency.
        """
        if self.aligned_data is None:
            raise ValueError("Data must be aligned first.")

        time = self.aligned_data['time'].values
        val1 = self.aligned_data[self.name1].values
        val2 = self.aligned_data[self.name2].values

        # Interpolate to regular grid
        dt = np.diff(time)
        valid_dt = dt[dt > 0]
        if len(valid_dt) == 0:
            return {
                'frequency': np.array([]),
                'coherence': np.array([]),
                'warning_flags': ["Time array is constant or has length < 2; cannot calculate coherence."]
            }
        median_dt = np.median(valid_dt)

        # Check for large gaps
        max_gap = 5.0 * median_dt
        warning_flags = []
        if np.max(dt) > max_gap:
             msg = f"Large data gap ({np.max(dt):.2f}) detected."
             warnings.warn(msg + " Interpolation may introduce artifacts in coherence. Consider using `run_wwz_coherence_analysis`.", UserWarning)
             warning_flags.append(msg)

        reg_time = np.arange(time[0], time[-1], median_dt)

        if len(reg_time) < 2:
            return {
                'frequency': np.array([]),
                'coherence': np.array([]),
                'warning_flags': warning_flags + ["Interpolated regular grid has fewer than 2 points; cannot calculate coherence."]
            }

        reg_val1 = np.interp(reg_time, time, val1)
        reg_val2 = np.interp(reg_time, time, val2)

        from scipy.signal import coherence

        # Calculate coherence using Welch's method
        fs = 1.0 / median_dt if median_dt > 0 else 1.0
        nperseg = min(len(reg_val1) // 2, 256)
        if nperseg < 1:
            nperseg = 1  # Minimum window length is 1

        f, Cxy = coherence(reg_val1, reg_val2, fs=fs, nperseg=nperseg)

        # Filter range
        if min_freq is not None:
            mask = f >= min_freq
            f = f[mask]
            Cxy = Cxy[mask]
        if max_freq is not None:
            mask = f <= max_freq
            f = f[mask]
            Cxy = Cxy[mask]

        return {
            'frequency': f,
            'coherence': Cxy,
            'warning_flags': warning_flags
        }

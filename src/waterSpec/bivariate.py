
import numpy as np
import pandas as pd
from typing import Tuple, Optional, Dict
import logging
import warnings
from scipy import stats, interpolate

from .haar_analysis import calculate_haar_fluctuations
from .surrogates import generate_phase_randomized_surrogates, calculate_significance_p_value
from .ls_cross_spectrum import calculate_ls_cross_spectrum, calculate_time_lag
from .wwz_coherence import calculate_wwz_coherence

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
        overlap_step_fraction: float = 0.1
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
            t_start = t_min

            while t_start + tau <= t_max:
                t_mid = t_start + tau / 2
                t_end = t_start + tau

                # Indices in the ALIGNED arrays
                idx_start = np.searchsorted(time, t_start, side='left')
                idx_mid = np.searchsorted(time, t_mid, side='left')
                idx_end = np.searchsorted(time, t_end, side='left')

                # Extract window data for both variables
                v1_left = val1[idx_start:idx_mid]
                v1_right = val1[idx_mid:idx_end]

                v2_left = val2[idx_start:idx_mid]
                v2_right = val2[idx_mid:idx_end]

                # Require data in both halves for BOTH variables
                if len(v1_left) > 0 and len(v1_right) > 0 and len(v2_left) > 0 and len(v2_right) > 0:
                    d1 = np.mean(v1_right) - np.mean(v1_left)
                    d2 = np.mean(v2_right) - np.mean(v2_left)

                    fluc1.append(d1)
                    fluc2.append(d2)

                if overlap:
                    t_start += step_size
                else:
                    t_start = t_end
                    if t_start >= t_max: break

            # Need at least 2 points for correlation
            if len(fluc1) >= 2:
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
        overlap_step_fraction: float = 0.1
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
            time, val1, val2, lags, overlap, overlap_step_fraction
        )

    def calculate_significance(
        self,
        lags: np.ndarray,
        n_surrogates: int = 100,
        overlap: bool = True,
        overlap_step_fraction: float = 0.1,
        seed: Optional[int] = None,
        max_gap: Optional[float] = None
    ) -> Dict:
        """
        Calculates significance of Cross-Haar Correlation using phase-randomized surrogates.
        Handles irregularly sampled data by resampling to a regular grid for surrogate generation
        and then interpolating back.
        """
        if self.aligned_data is None:
            raise ValueError("Data must be aligned first using `align_data`.")

        time = self.aligned_data['time'].values
        val1 = self.aligned_data[self.name1].values # Keep var1 fixed
        val2 = self.aligned_data[self.name2].values

        if len(val2) < 10:
             return {'error': 'Insufficient data for surrogates'}

        # Run observed analysis
        obs_results = self._calculate_cross_haar(
            time, val1, val2, lags, overlap, overlap_step_fraction
        )
        obs_corrs = np.array(obs_results['correlation'])

        # --- Handle Irregular Sampling for Surrogates ---
        # 1. Create a regular time grid covering the range of the data
        # Use the median sampling interval as the step
        dt = np.diff(time)
        median_dt = np.median(dt[dt > 0])

        # Safe gap handling
        if max_gap is None:
            max_gap = 5.0 * median_dt

        warning_flags = []
        if np.max(dt) > max_gap:
             msg = f"Large data gap ({np.max(dt):.2f}) detected in surrogate generation."
             warnings.warn(msg + " Interpolation may introduce artifacts.", UserWarning)
             warning_flags.append(msg)

        reg_time = np.arange(time[0], time[-1] + median_dt, median_dt)

        # 2. Interpolate data2 onto this regular grid
        # Use linear interpolation (or could use others)
        reg_val2 = np.interp(reg_time, time, val2)

        # 3. Generate surrogates on the regular grid (FFT safe)
        reg_surrs = generate_phase_randomized_surrogates(
            reg_val2, n_surrogates=n_surrogates, seed=seed
        )

        # 4. Interpolate surrogates back to original timestamps
        # We need to do this for each surrogate
        surr_corrs = np.zeros((n_surrogates, len(lags)))

        for i in range(n_surrogates):
            # Interpolate back to original 'time'
            surr_on_orig_time = np.interp(time, reg_time, reg_surrs[i])

            res = self._calculate_cross_haar(
                time, val1, surr_on_orig_time, lags, overlap, overlap_step_fraction
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
        overlap_step_fraction: float = 0.1
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
        t_start = time[0]

        while t_start + tau <= time[-1]:
            t_mid = t_start + tau / 2
            t_end = t_start + tau

            idx_start = np.searchsorted(time, t_start, side='left')
            idx_mid = np.searchsorted(time, t_mid, side='left')
            idx_end = np.searchsorted(time, t_end, side='left')

            v1_left = val1[idx_start:idx_mid]
            v1_right = val1[idx_mid:idx_end]

            if len(v1_left) > 0 and len(v1_right) > 0:
                d1 = np.mean(v1_right) - np.mean(v1_left)
                t_centers.append(t_mid) # Use mid point as reference
                fluc1_vals.append(d1)

            if overlap:
                t_start += step_size
            else:
                t_start = t_end
                if t_start >= time[-1]: break

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

            for i, t_c in enumerate(t_centers):
                t_target_mid = t_c - ell
                t_q_start = t_target_mid - tau/2
                t_q_mid = t_target_mid
                t_q_end = t_target_mid + tau/2

                # Check bounds
                if t_q_start < time[0] or t_q_end > time[-1]:
                    continue

                idx_q_start = np.searchsorted(time, t_q_start, side='left')
                idx_q_mid = np.searchsorted(time, t_q_mid, side='left')
                idx_q_end = np.searchsorted(time, t_q_end, side='left')

                v2_left = val2[idx_q_start:idx_q_mid]
                v2_right = val2[idx_q_mid:idx_q_end]

                if len(v2_left) > 0 and len(v2_right) > 0:
                    d2 = np.mean(v2_right) - np.mean(v2_left)
                    fluc2_vals.append(d2)
                    valid_indices.append(i)

            if len(fluc2_vals) > 2:
                # Correlate matched pairs
                c1 = fluc1_vals[valid_indices]
                c2 = np.array(fluc2_vals)
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
        overlap_step_fraction: float = 0.1
    ) -> Dict:
        """
        Calculates the Hysteresis Loop Area between fluctuations of the two variables at scale tau.
        Uses the shoelace formula (signed polygon area).

        Args:
            tau (float): Time scale.

        Returns:
            Dict: {'area': float, 'direction': str}
        """
        if self.aligned_data is None:
            raise ValueError("Data must be aligned first.")

        time = self.aligned_data['time'].values
        val1 = self.aligned_data[self.name1].values
        val2 = self.aligned_data[self.name2].values

        fluc1 = [] # x coordinate (usually C)
        fluc2 = [] # y coordinate (usually Q)

        step_size = tau * overlap_step_fraction if overlap else tau
        t_start = time[0]

        while t_start + tau <= time[-1]:
            t_mid = t_start + tau / 2
            t_end = t_start + tau

            idx_start = np.searchsorted(time, t_start, side='left')
            idx_mid = np.searchsorted(time, t_mid, side='left')
            idx_end = np.searchsorted(time, t_end, side='left')

            v1_left = val1[idx_start:idx_mid]
            v1_right = val1[idx_mid:idx_end]
            v2_left = val2[idx_start:idx_mid]
            v2_right = val2[idx_mid:idx_end]

            if len(v1_left) > 0 and len(v1_right) > 0 and len(v2_left) > 0 and len(v2_right) > 0:
                d1 = np.mean(v1_right) - np.mean(v1_left)
                d2 = np.mean(v2_right) - np.mean(v2_left)
                fluc1.append(d1)
                fluc2.append(d2)

            if overlap:
                t_start += step_size
            else:
                t_start = t_end
                if t_start >= time[-1]: break

        if len(fluc1) < 3:
            return {'area': np.nan, 'direction': 'insufficient_data'}

        # Shoelace formula for signed area
        # A = 0.5 * sum(x_i * y_{i+1} - x_{i+1} * y_i)
        # Here x = fluc2 (Q, independent), y = fluc1 (C, dependent).

        x = np.array(fluc2)
        y = np.array(fluc1)

        area = 0.5 * np.sum(x[:-1] * y[1:] - x[1:] * y[:-1])

        direction = "Counter-Clockwise" if area > 0 else "Clockwise"
        if np.isclose(area, 0): direction = "None"

        return {'area': area, 'direction': direction}

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

    def run_wwz_coherence_analysis(
        self,
        freqs: np.ndarray,
        taus: Optional[np.ndarray] = None,
        decay_constant: float = 0.00125,
        smoothing_window: float = 2.0
    ) -> Dict:
        """
        Calculates WWZ Coherence directly on irregular data.
        This is the statistically defensible method for coherence estimation on uneven series.

        Args:
            freqs (np.ndarray): Frequencies.
            taus (np.ndarray, optional): Time shift grid.
            decay_constant (float): Morlet wavelet decay constant.
            smoothing_window (float): Smoothing window sigma (pixels).

        Returns:
            Dict: 'coherence', 'freqs', 'taus'.
        """
        # No alignment needed.

        coherence, _, taus_out = calculate_wwz_coherence(
            self.time1, self.data1,
            self.time2, self.data2,
            freqs, taus=taus,
            decay_constant=decay_constant,
            smoothing_window=smoothing_window
        )

        return {
            'freqs': freqs,
            'taus': taus_out,
            'coherence': coherence
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
        median_dt = np.median(dt[dt > 0])

        # Check for large gaps
        max_gap = 5.0 * median_dt
        warning_flags = []
        if np.max(dt) > max_gap:
             msg = f"Large data gap ({np.max(dt):.2f}) detected."
             warnings.warn(msg + " Interpolation may introduce artifacts in coherence. Consider using `run_wwz_coherence_analysis`.", UserWarning)
             warning_flags.append(msg)

        reg_time = np.arange(time[0], time[-1], median_dt)

        reg_val1 = np.interp(reg_time, time, val1)
        reg_val2 = np.interp(reg_time, time, val2)

        from scipy.signal import coherence

        # Calculate coherence using Welch's method
        # fs = 1 / median_dt
        fs = 1.0 / median_dt if median_dt > 0 else 1.0

        f, Cxy = coherence(reg_val1, reg_val2, fs=fs, nperseg=min(len(reg_val1)//2, 256))

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

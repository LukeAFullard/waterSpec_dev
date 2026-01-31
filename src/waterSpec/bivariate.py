
import numpy as np
import pandas as pd
from typing import Tuple, Optional, Dict
import logging
from scipy import stats

from .haar_analysis import calculate_haar_fluctuations

class BivariateAnalysis:
    """
    Performs bivariate analysis between two time series (e.g., Concentration and Discharge).
    Supports Cross-Haar Correlation and Lagged Response Analysis.
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

        results = {
            'lags': [],
            'correlation': [],
            'n_pairs': [],
            'slope_alpha': [] # sensitivity
        }

        for tau in lags:
            fluc1 = []
            fluc2 = []

            step_size = tau * overlap_step_fraction if overlap else tau
            t_start = time[0]

            # Correct loop condition:
            # We need window [t_start, t_start + tau] to be valid.
            # AND we need enough data.
            # If overlap=False, step_size=tau.

            while t_start + tau <= time[-1]:
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
                    if t_start >= time[-1]: break

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

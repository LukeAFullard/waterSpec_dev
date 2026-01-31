
import numpy as np
import pandas as pd
from typing import List, Dict, Union, Optional, Tuple
import logging
from scipy import stats
from .surrogates import generate_phase_randomized_surrogates, calculate_significance_p_value

class MultivariateAnalysis:
    """
    Performs multivariate analysis on multiple environmental time series.
    Supports Partial Cross-Haar Correlation to disentangle direct vs indirect relationships.
    """
    def __init__(self, inputs: List[Dict[str, Union[np.ndarray, str]]]):
        """
        Args:
            inputs: List of dictionaries, each containing:
                - 'time': np.ndarray
                - 'data': np.ndarray
                - 'name': str
        """
        self.inputs = inputs
        self.names = [inp['name'] for inp in inputs]
        self.aligned_data = None
        self.logger = logging.getLogger(__name__)

    def align_data(self, tolerance: float, method: str = 'nearest') -> pd.DataFrame:
        """
        Aligns all input time series to a common timeline (intersection).
        Uses the first series as the reference for time points if 'nearest',
        or merges iteratively.

        For robustness with N > 2, we use pandas merge_asof iteratively or
        reindex if they are close enough.
        """
        if not self.inputs:
            raise ValueError("No inputs to align.")

        # Convert to DataFrames
        dfs = []
        for inp in self.inputs:
            df = pd.DataFrame({'time': inp['time'], inp['name']: inp['data']})
            df = df.sort_values('time').dropna()
            dfs.append(df)

        # Iterative merge
        # We start with the first dataframe and merge others onto it
        base_df = dfs[0]

        # Determine tolerance type
        tol = tolerance

        # Ensure time is float if tolerance is float, or compatible
        if pd.api.types.is_numeric_dtype(base_df['time']):
             if not pd.api.types.is_float_dtype(base_df['time']):
                  base_df['time'] = base_df['time'].astype(float)

        for i in range(1, len(dfs)):
            next_df = dfs[i]
            if pd.api.types.is_numeric_dtype(next_df['time']):
                 if not pd.api.types.is_float_dtype(next_df['time']):
                      next_df['time'] = next_df['time'].astype(float)

            # Using merge_asof
            # direction='nearest', tolerance=tol
            merged = pd.merge_asof(
                base_df,
                next_df,
                on='time',
                direction='nearest',
                tolerance=tol
            )
            # Drop rows where we didn't find a match
            base_df = merged.dropna()

        self.aligned_data = base_df
        return self.aligned_data

    def run_partial_cross_haar_analysis(
        self,
        target_var1: str,
        target_var2: str,
        conditioning_vars: List[str],
        lags: np.ndarray,
        overlap: bool = True
    ) -> Dict:
        """
        Calculates Partial Cross-Haar Correlation between var1 and var2,
        controlling for conditioning_vars.

        rho_XY.Z = (rho_XY - rho_XZ * rho_YZ) / sqrt((1 - rho_XZ * rho_XZ)*(1 - rho_YZ * rho_YZ))
        For multivariate Z, uses inverse covariance matrix (precision matrix).
        """
        if self.aligned_data is None:
            raise ValueError("Data must be aligned first.")

        # Verify variables exist
        all_vars = [target_var1, target_var2] + conditioning_vars
        for v in all_vars:
            if v not in self.aligned_data.columns:
                raise ValueError(f"Variable '{v}' not found in aligned data.")

        results = {
            'lags': [],
            'partial_correlation': [],
            'n_pairs': []
        }

        data_arrays = {v: self.aligned_data[v].values for v in all_vars}
        time = self.aligned_data['time'].values

        for tau in lags:
            # Collect fluctuations for this lag
            fluctuations = {v: [] for v in all_vars}

            step_size = tau * 0.1 if overlap else tau
            t_start = time[0]
            t_max = time[-1]

            while t_start + tau <= t_max:
                t_mid = t_start + tau / 2
                t_end = t_start + tau

                idx_start = np.searchsorted(time, t_start, side='left')
                idx_mid = np.searchsorted(time, t_mid, side='left')
                idx_end = np.searchsorted(time, t_end, side='left')

                # Check counts
                n_left = idx_mid - idx_start
                n_right = idx_end - idx_mid

                if n_left > 0 and n_right > 0:
                    # Calculate fluctuations for all vars
                    temp_flucs = {}

                    for v in all_vars:
                        vals = data_arrays[v]
                        v_left = vals[idx_start:idx_mid]
                        v_right = vals[idx_mid:idx_end]

                        d = np.mean(v_right) - np.mean(v_left)
                        temp_flucs[v] = d

                    for v in all_vars:
                        fluctuations[v].append(temp_flucs[v])

                if overlap:
                    t_start += step_size
                else:
                    t_start = t_end
                    if t_start >= t_max: break

            n_samples = len(fluctuations[target_var1])
            results['n_pairs'].append(n_samples)

            if n_samples < len(all_vars) + 2:
                results['lags'].append(tau)
                results['partial_correlation'].append(np.nan)
                continue

            # Construct data matrix for correlation
            X_list = [fluctuations[v] for v in all_vars]
            X = np.column_stack(X_list)

            # Covariance matrix
            C = np.cov(X, rowvar=False)

            # Precision matrix (Inverse Covariance)
            try:
                P = np.linalg.inv(C)

                # Partial correlation formula using Precision Matrix P
                # rho_ij.rest = - P_ij / sqrt(P_ii * P_jj)

                # Target vars are index 0 and 1
                p_12 = -P[0, 1] / np.sqrt(P[0, 0] * P[1, 1])

                results['lags'].append(tau)
                results['partial_correlation'].append(p_12)

            except np.linalg.LinAlgError:
                results['lags'].append(tau)
                results['partial_correlation'].append(np.nan)

        return results

    def calculate_significance(
        self,
        target_var1: str,
        target_var2: str,
        conditioning_vars: List[str],
        lags: np.ndarray,
        n_surrogates: int = 100,
        seed: Optional[int] = None,
        overlap: bool = True
    ) -> Dict:
        """
        Calculates significance of Partial Cross-Haar Correlation using surrogates.
        Generates surrogates for target_var1 while keeping others fixed.
        """
        if self.aligned_data is None:
            raise ValueError("Data must be aligned first.")

        # Run observed analysis
        obs_res = self.run_partial_cross_haar_analysis(
            target_var1, target_var2, conditioning_vars, lags, overlap
        )
        obs_corrs = np.array(obs_res['partial_correlation'])

        # Generate surrogates for var1
        data1 = self.aligned_data[target_var1].values
        surrogates = generate_phase_randomized_surrogates(
            data1, n_surrogates=n_surrogates, seed=seed
        )

        surr_corrs = np.zeros((n_surrogates, len(lags)))

        # Temporary backup of original data
        original_data = data1.copy()

        for i in range(n_surrogates):
            # Inject surrogate into aligned_data
            self.aligned_data[target_var1] = surrogates[i]

            res = self.run_partial_cross_haar_analysis(
                target_var1, target_var2, conditioning_vars, lags, overlap
            )
            surr_corrs[i, :] = res['partial_correlation']

        # Restore original data
        self.aligned_data[target_var1] = original_data

        # Calculate p-values
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
            'observed_partial_correlation': obs_corrs,
            'p_values': np.array(p_values),
            'surrogate_correlations': surr_corrs
        }


import numpy as np
import pandas as pd
from typing import List, Tuple, Dict, Optional, Union
import logging
from .haar_analysis import calculate_sliding_haar
from .analysis import Analysis

class RegimeAnalysis:
    """
    Facilitates analysis of time series segmented by hydrologic regimes
    (e.g., Event vs. Baseflow).
    """
    def __init__(self, analysis_obj: Analysis):
        self.parent = analysis_obj
        self.event_slices = []
        self.baseflow_slices = []
        self.logger = logging.getLogger(__name__)

    def segment_by_fluctuation(
        self,
        scale: float,
        threshold_factor: float = 3.0,
        min_event_duration: float = 0
    ) -> Dict[str, List[slice]]:
        """
        Segments the time series based on sliding Haar fluctuations.

        Args:
            scale (float): The time scale (window size) to calculate fluctuations.
            threshold_factor (float): Multiplier for MAD to define threshold.
            min_event_duration (float): Minimum duration to count as an event.

        Returns:
            Dict with 'event' and 'baseflow' lists of slice objects (indices).
        """
        # Ensure data is loaded
        if self.parent.time is None or self.parent.data is None:
            raise ValueError("Analysis object has no data loaded.")

        time = self.parent.time
        data = self.parent.data

        # Calculate sliding Haar
        # step_size default is window/10 in calculate_sliding_haar
        t_centers, fluctuations = calculate_sliding_haar(
            time, data, window_size=scale
        )

        abs_fluc = np.abs(fluctuations)

        # Calculate robust threshold (Median Absolute Deviation)
        median_val = np.median(abs_fluc)
        mad = np.median(np.abs(abs_fluc - median_val))
        threshold = median_val + threshold_factor * mad

        # Identify event periods (in terms of t_centers)
        event_mask = abs_fluc > threshold

        # Map back to original time indices
        # This is approximate since t_centers are window centers
        # We define an event at time t if the window centered near t is high fluctuation

        # A safer way: if a window [t_start, t_end] has high fluctuation,
        # mark all points in that window as "event".

        is_event_point = np.zeros(len(time), dtype=bool)

        # We need to reconstruct the windows corresponding to t_centers
        # calculate_sliding_haar returns centers.
        # Window is [center - scale/2, center + scale/2]

        for center, is_evt in zip(t_centers, event_mask):
            if is_evt:
                t_start = center - scale/2
                t_end = center + scale/2

                # Find indices in original time
                idx_start = np.searchsorted(time, t_start, side='left')
                idx_end = np.searchsorted(time, t_end, side='left')

                is_event_point[idx_start:idx_end] = True

        # Now convert boolean mask to slices
        # Use diff to find state changes
        # Prepend/append to handle start/end

        # Helper to find runs
        def get_slices(mask):
            slices = []
            if len(mask) == 0:
                return slices

            # Find indices where mask changes
            # 0->1 (start) or 1->0 (end)
            padded = np.concatenate(([0], mask.astype(int), [0]))
            diffs = np.diff(padded)
            starts = np.where(diffs == 1)[0]
            ends = np.where(diffs == -1)[0]

            for s, e in zip(starts, ends):
                # Filter by min duration if needed (in time units)
                duration = time[e-1] - time[s]
                if duration >= min_event_duration:
                    slices.append(slice(s, e))

            return slices

        self.event_slices = get_slices(is_event_point)
        self.baseflow_slices = get_slices(~is_event_point)

        return {
            'event': self.event_slices,
            'baseflow': self.baseflow_slices
        }

    def run_regime_comparison(self, min_lag=None, max_lag=None) -> Dict:
        """
        Runs Haar analysis on concatenated event segments vs baseflow segments.
        """
        if not self.event_slices and not self.baseflow_slices:
            raise ValueError("Run segment_by_fluctuation first.")

        from .haar_analysis import HaarAnalysis, calculate_haar_fluctuations

        results = {}

        for regime, slices in [('event', self.event_slices), ('baseflow', self.baseflow_slices)]:
            if not slices:
                results[regime] = None
                continue

            # Concatenate data from slices
            # Note: Concatenating time series creates jumps in time.
            # Haar analysis handles time gaps if we treat them carefully or if we process per segment?
            # Standard HaarAnalysis takes (time, data). If time has large gaps,
            # the loop `while t_start + delta_t <= time[-1]` will try to bridge gaps.
            # We should probably run Haar on each segment and average the Structure Functions?
            # Or use a concatenated artificial time? No, that destroys physics.

            # Better approach: HaarAnalysis should support list of segments?
            # Or we just modify the input to be a list of arrays?
            # Current HaarAnalysis implementation assumes monotonic time array.

            # Workaround: "Stitch" structure functions.
            # We can calculate S1 for each segment and weight-average them.

            all_s1 = []
            all_counts = []
            common_lags = None

            # We need a common set of lags for averaging
            # We can pick lags based on the full dataset or the first segment?
            # Let's assume we use the user provided min/max lag or defaults.

            # Create a temporary Haar object to get default lags
            if common_lags is None:
                # Use full dataset to define lags
                 temp = HaarAnalysis(self.parent.time, self.parent.data)
                 # Just get lags
                 # This is a bit inefficient but safe
                 res = temp.run(min_lag=min_lag, max_lag=max_lag, num_lags=20)
                 common_lags = res['lags']

            # Accumulate fluctuations for common lags
            accum_s1 = np.zeros(len(common_lags))
            accum_counts = np.zeros(len(common_lags))

            for sl in slices:
                seg_time = self.parent.time[sl]
                seg_data = self.parent.data[sl]

                if len(seg_time) < 2: continue

                # We need calculate_haar_fluctuations to accept specific lags
                _, s1, counts, _ = calculate_haar_fluctuations(
                    seg_time, seg_data, lag_times=common_lags
                )

                # Note: s1 is returned only for valid lags.
                # But if we pass lag_times, does it return aligned?
                # calculate_haar_fluctuations returns (valid_lags, s1, counts...)
                # It might skip lags if no data.

                # We need to map back to common_lags index
                # This is tricky without refactoring calculate_haar_fluctuations slightly
                # or doing a search.

                # Let's just trust that if we pass lag_times, we can match by value?
                # Or assume it processes them in order.

                # Refactoring risk. Let's do a lookup.
                # The returned 'valid_lags' is subset of 'lag_times'.

                # Actually, calculate_haar_fluctuations logic:
                # for delta_t in lag_times: ...
                # If count > 0: append.
                # So the output arrays are compressed.

                # We need to map them.

                # Create a dict for this segment
                seg_res = dict(zip(s1, counts)) # Wait, need lag keys.
                # returned valid_lags

                # Rerun for this segment
                v_lags, v_s1, v_counts, _ = calculate_haar_fluctuations(
                    seg_time, seg_data, lag_times=common_lags, overlap=True
                )

                for i, lag in enumerate(v_lags):
                    # Find index in common_lags
                    # Use isclose for float comparison
                    idx = np.where(np.isclose(common_lags, lag))[0]
                    if len(idx) > 0:
                        k = idx[0]
                        # s1 is average fluctuation.
                        # Total fluctuation sum = s1 * count
                        accum_s1[k] += v_s1[i] * v_counts[i]
                        accum_counts[k] += v_counts[i]

            # Final weighted average
            final_s1 = np.divide(accum_s1, accum_counts, out=np.zeros_like(accum_s1), where=accum_counts>0)

            # Filter out empty lags
            valid_mask = accum_counts > 0
            final_lags = common_lags[valid_mask]
            final_s1 = final_s1[valid_mask]

            # Fit slope
            from .haar_analysis import fit_haar_slope
            fit = fit_haar_slope(final_lags, final_s1)

            results[regime] = {
                'beta': fit['beta'],
                'lags': final_lags,
                's1': final_s1
            }

        return results

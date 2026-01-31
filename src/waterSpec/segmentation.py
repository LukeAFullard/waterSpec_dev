import numpy as np
from typing import List, Tuple, Dict, Optional
from waterSpec.haar_analysis import calculate_sliding_haar

class SegmentedRegimeAnalysis:
    """
    Tools for segmenting time series based on Haar fluctuation volatility
    and analyzing the segments separately.
    """

    @staticmethod
    def segment_by_fluctuation(
        time: np.ndarray,
        data: np.ndarray,
        scale: float,
        threshold_factor: float = 3.0,
        min_event_duration: float = 0
    ) -> Dict[str, List[Tuple[float, float]]]:
        """
        Segments the time series into 'event' and 'background' regimes based on
        volatility at a specific Haar scale.

        Args:
            time (np.ndarray): Time array.
            data (np.ndarray): Data array.
            scale (float): The Haar scale (window size) to compute fluctuations.
            threshold_factor (float): Multiplier for the background noise level (MAD).
            min_event_duration (float): Minimum duration to classify as an event.

        Returns:
            Dict containing lists of (start, end) time tuples for 'events' and 'background'.
        """
        # 1. Compute Sliding Haar Fluctuations
        # We use a step size smaller than scale for resolution
        t_centers, fluctuations = calculate_sliding_haar(
            time, data, window_size=scale, step_size=scale/5.0
        )

        if len(fluctuations) == 0:
            return {'events': [], 'background': [(time[0], time[-1])]}

        # 2. Determine Threshold
        # Use Median Absolute Deviation (MAD) as robust estimator of background noise
        median_fluc = np.median(np.abs(fluctuations))
        # Consistent estimator for normal distribution sigma is 1.4826 * MAD
        # But here fluctuations are absolute differences, already somewhat rectified.
        # Let's just use the raw median level as the baseline.

        threshold = threshold_factor * median_fluc

        # 3. Identify Event Periods
        is_event = np.abs(fluctuations) > threshold

        # 4. Convert to Time Ranges
        # This is tricky because t_centers are points. We need to map back to time ranges.
        # A simple approach: if t_center is flagged, the window [t_c - scale/2, t_c + scale/2] is volatile.

        # We will construct a boolean mask for the original time array
        # This allows us to handle the "union" of overlapping windows easily.
        event_mask = np.zeros_like(time, dtype=bool)

        for t_c, flag in zip(t_centers, is_event):
            if flag:
                t_start_win = t_c - scale/2
                t_end_win = t_c + scale/2

                # Find indices in original time array
                # Use searchsorted
                idx_start = np.searchsorted(time, t_start_win, side='left')
                idx_end = np.searchsorted(time, t_end_win, side='right')

                event_mask[idx_start:idx_end] = True

        # 5. Extract Segments from Mask
        events = []
        background = []

        # Identify contiguous regions in the mask
        # Pad with False to handle edge cases
        padded_mask = np.concatenate(([False], event_mask, [False]))
        diff = np.diff(padded_mask.astype(int))

        starts = np.where(diff == 1)[0]
        ends = np.where(diff == -1)[0]

        for s, e in zip(starts, ends):
            # s is inclusive index, e is exclusive index in original time
            # Map to time values
            # Handle e=len(time) case
            t_s = time[s]
            t_e = time[e-1] if e <= len(time) else time[-1]

            duration = t_e - t_s
            if duration >= min_event_duration:
                events.append((t_s, t_e))

        # Now do background (inverse)
        # We can just iterate the gaps between events
        current_t = time[0]
        for ev_start, ev_end in events:
            if ev_start > current_t:
                background.append((current_t, ev_start))
            current_t = max(current_t, ev_end)

        if current_t < time[-1]:
            background.append((current_t, time[-1]))

        return {
            'events': events,
            'background': background,
            'threshold': threshold,
            'fluctuations': fluctuations,
            't_centers': t_centers
        }

    @staticmethod
    def extract_segments(
        time: np.ndarray,
        data: np.ndarray,
        segments: List[Tuple[float, float]]
    ) -> List[Tuple[np.ndarray, np.ndarray]]:
        """
        Extracts data corresponding to the time segments.
        Returns a list of (time_chunk, data_chunk) tuples.
        """
        extracted = []
        for start, end in segments:
            mask = (time >= start) & (time <= end)
            if np.any(mask):
                extracted.append((time[mask], data[mask]))
        return extracted

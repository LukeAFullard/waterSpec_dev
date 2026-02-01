
import numpy as np
import logging
from typing import Dict, Tuple, Optional
from .haar_analysis import calculate_haar_fluctuations, fit_haar_slope, calculate_sliding_haar

class SpatialHaarAnalysis:
    """
    Adapts Haar Wavelet Analysis for spatial datasets (e.g., river longitudinal profiles).
    Instead of 'Time' and 'Lag Time', we analyze 'Distance' and 'Spatial Scale' (Wavenumber).
    """

    def __init__(self, distance: np.ndarray, data: np.ndarray, variable_name: str, distance_unit: str = "km"):
        self.distance = distance
        self.data = data
        self.variable_name = variable_name
        self.distance_unit = distance_unit
        self.results = {}
        self.logger = logging.getLogger(__name__)

    def run_spatial_analysis(
        self,
        min_scale: Optional[float] = None,
        max_scale: Optional[float] = None,
        num_scales: int = 20,
        log_spacing: bool = True,
        overlap: bool = True,
        n_bootstraps: int = 100
    ) -> Dict:
        """
        Runs the Haar analysis on spatial data.

        Args:
            min_scale: Minimum spatial scale (lag distance).
            max_scale: Maximum spatial scale.

        Returns:
            Dict containing scales, structure function S1, and spectral slope beta.
        """
        # Re-use the temporal calculation engine
        # 'time' argument becomes 'distance'

        scales, s1, counts, n_eff = calculate_haar_fluctuations(
            time=self.distance,
            data=self.data,
            min_lag=min_scale,
            max_lag=max_scale,
            num_lags=num_scales,
            log_spacing=log_spacing,
            overlap=overlap
        )

        # Fit slope
        fit_res = fit_haar_slope(scales, s1, n_bootstraps=n_bootstraps)

        self.results = {
            "scales": scales,
            "s1": s1,
            "counts": counts,
            "n_effective": n_eff,
            "H": fit_res.get("H", np.nan),
            "beta": fit_res.get("beta", np.nan),
            "r2": fit_res.get("r2", np.nan),
            "intercept": fit_res.get("intercept", np.nan),
            "beta_ci_lower": fit_res.get("beta_ci_lower"),
            "beta_ci_upper": fit_res.get("beta_ci_upper")
        }

        return self.results

    def detect_spatial_hotspots(
        self,
        scale: float,
        threshold_factor: float = 3.0
    ) -> Dict:
        """
        Identifies spatial 'hotspots' or anomalies at a specific spatial scale.
        """

        centers, fluctuations = calculate_sliding_haar(
            self.distance, self.data, window_size=scale
        )

        # If fluctuation array is empty, return empty result
        if len(fluctuations) == 0:
             return {
                "scale": scale,
                "threshold": np.nan,
                "locations": np.array([]),
                "magnitudes": np.array([]),
                "all_centers": np.array([]),
                "all_fluctuations": np.array([])
             }

        median_fluc = np.median(np.abs(fluctuations))
        threshold = threshold_factor * median_fluc

        is_hotspot = np.abs(fluctuations) > threshold

        hotspot_locs = centers[is_hotspot]
        hotspot_mags = fluctuations[is_hotspot]

        return {
            "scale": scale,
            "threshold": threshold,
            "locations": hotspot_locs,
            "magnitudes": hotspot_mags,
            "all_centers": centers,
            "all_fluctuations": fluctuations
        }

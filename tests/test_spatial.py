
import numpy as np
import pytest
from waterSpec.spatial import SpatialHaarAnalysis

def test_spatial_haar_slope():
    """
    Test that spatial Haar correctly identifies a random walk slope.
    Random walk in space -> Brownian motion -> beta approx 2 (H approx 0.5).
    """
    rng = np.random.default_rng(42)
    dist = np.linspace(0, 1000, 500)
    # Brownian path
    steps = rng.standard_normal(500)
    data = np.cumsum(steps)

    spatial = SpatialHaarAnalysis(dist, data, "Elevation")
    res = spatial.run_spatial_analysis(num_scales=10, n_bootstraps=10)

    beta = res['beta']
    # Allow some variance
    assert 1.5 < beta < 2.5

def test_hotspot_detection():
    # Increase density of points so window size 5 catches enough points
    # range 0-100, 1000 points -> step 0.1
    # window 5 -> 50 points per window. Plenty.

    dist = np.linspace(0, 100, 1000)
    data = np.zeros(1000)
    # Add a spike at 50
    # Make it wider than one point to be safe? Or just one point?
    # One point spike at 50.0
    # index 500
    data[500] = 10.0

    spatial = SpatialHaarAnalysis(dist, data, "Conc")
    # Scale 5
    res = spatial.detect_spatial_hotspots(scale=5.0, threshold_factor=2.0)

    # Check if we got any results
    assert len(res['locations']) > 0
    # Should detect around 50
    assert np.any(np.abs(res['locations'] - 50) < 5)

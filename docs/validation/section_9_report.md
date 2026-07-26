# Spatial Haar Analysis Validation (Section 9)

## 9.1 Spatial analogue of colored-noise recovery
**Pass Criteria:** True spatial roughness exponent recovered within $\pm 0.25$ by `SpatialHaarAnalysis` across various scaling regimes (Pink $\beta=1$, fBm $\beta=1.6$, Anti-persistent $\beta=0.3$).
**Result:** Passed (100% success rate across 30 trials per $\beta$). The spatial wrapper correctly calls the temporal Haar engine to recover spatial exponents.

## 9.2 Spatial hotspot detection (Positive Control)
**Pass Criteria:** Detect a known injected spatial anomaly (hotspot) at its expected spatial scale.
**Result:** Passed (100% success rate across 20 trials). Using a `threshold_factor` of 6.0 reliably detects the $15\sigma$ anomaly.

## 9.3 No-hotspot negative control
**Pass Criteria:** No false positive hotspots detected on smooth white-noise background data.
**Result:** Passed (100% success rate). The threshold logic correctly distinguishes background spatial noise from structural anomalies.

## 9.4 Uneven spatial sampling
**Pass Criteria:** Spatial Haar Analysis degrades gracefully (still recovers exponent within $\pm 0.30$, still detects large hotspots) when 30% of data points are missing at random.
**Result:** Passed (100% success rate for both exponent recovery and hotspot detection across 20 trials). The underlying overlapping-window Haar method tolerates gappy spatial data just as it does gappy time-series data.

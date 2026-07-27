# Section 13: Peak Detection Robustness

This section contains tests to validate the robustness of the peak detection algorithm in `waterSpec`.

## Tests
- **13.1 Peak near spectrum edges**: Successfully detects peaks injected very close to the lowest resolvable frequency and near the Nyquist frequency using the FAP method.
- **13.2 Peak detection under a segmented background**: Successfully detects a peak even when the background spectrum requires a segmented (broken power-law) fit, utilizing the `residual` peak detection method.
- **13.3 Gumbel background fit sanity check**: Confirms that the residuals of the spectral fit follow the theoretical Gumbel distribution (via KS-test) when no peaks are present.
- **13.4 Uneven sampling cross-reference**: The existing `validate_peak_detection_sweep.py` was extended to run over `missing_frac` values (e.g., 0.0 and 0.3) to test peak detection robustness under uneven sampling.

All tests passed.

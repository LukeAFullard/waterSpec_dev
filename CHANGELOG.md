# Changelog

## [0.1.1] - 2026-07-06
### Changed
- **Fixed:** The `std_corrected` aggregation method in `haar_analysis.py` previously applied a ddof=1 sample standard deviation correction to a zero-mean (ddof=0) RMS calculation, causing a systematic overestimate at small window counts. The formula has been fixed to use the mathematically correct degrees-of-freedom factor. Structure function values generated with this aggregation mode will now be slightly lower at very large lags compared to previous versions.
- **Changed:** Added a soft warning to notify users when `max_lag` exceeds `T/5` (design doc threshold).
- **Docs:** Updated documentation regarding multifractal equivalence and small-sample metrics.

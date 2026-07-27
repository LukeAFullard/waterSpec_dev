# Section 14: Edge Cases & Robustness (Negative Testing)

This section ensures the package fails safely and informatively on edge cases.

## Tests
- **14.1 Very short series**: Fails with informative exception regarding minimum data points.
- **14.2 Constant / zero-variance series**: Fails with informative exception regarding zero variance.
- **14.3 Series with NaNs/Infs**: Successfully drops missing values and issues appropriate warnings.
- **14.4 Single/few unique timestamps**: Fails gracefully when data lacks proper monotonic unique timestamps.
- **14.5 Extreme outliers**: Confirms the fitting routines handle extreme outliers without crashing.
- **14.6 Heavily-censored columns**: Fails informatively with the 'drop' strategy and processes correctly (albeit with warnings) with the 'multiplier' strategy.
- **14.7 Mismatched/malformed inputs**: Low-level functions raise correct `ValueError` or `IndexError` on mismatched shapes.
- **14.8 Extremely large N**: Successfully scales and processes N=100,000 points in reasonable time (approx 6s).
- **14.9 Timezone-aware parsing**: Mixed timezones are gracefully rejected by `data_loader.py` with an informative error directing the user to standardize the timezone.

All tests passed as expected.

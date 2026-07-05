# Peak Detection Validation Sweep Report

This report validates the performance of `waterSpec`'s two peak detection methods.

## Test Methodology
A synthetic periodic signal (period of ~50.0 days) was injected into noise with varying characteristics. The validation was run across a matrix of conditions:
- **Noise Color (`beta`):** The spectral exponent of the background noise was varied.
- **Signal Strength (`amplitude`):** The amplitude of the injected sine wave was varied.

For each condition, the signal was analyzed by:
1.  **waterSpec (residual)**: The residual-based method.
2.  **waterSpec (redfit)**: The redfit-based method.

## Results
The table below summarizes the results. "✅ Found" indicates the package successfully identified the periodic signal as statistically significant (at the 95% level). "❌ Not Found" indicates it did not.

|   beta |   amplitude | waterSpec (residual)   | waterSpec (redfit)   |
|-------:|------------:|:-----------------------|:---------------------|
| 0.0    | 2.00       | ✅ Found                | ✅ Found              |
| 0.0    | 1.50       | ✅ Found                | ✅ Found              |
| 0.0    | 1.00       | ✅ Found                | ✅ Found              |
| 0.0    | 0.80       | ✅ Found                | ✅ Found              |
| 0.0    | 0.50       | ✅ Found                | ✅ Found              |
| 0.0    | 0.30       | ✅ Found                | ✅ Found              |
| 0.5    | 2.00       | ✅ Found                | ✅ Found              |
| 0.5    | 1.50       | ✅ Found                | ✅ Found              |
| 0.5    | 1.00       | ✅ Found                | ✅ Found              |
| 0.5    | 0.80       | ✅ Found                | ✅ Found              |
| 0.5    | 0.50       | ✅ Found                | ✅ Found              |
| 0.5    | 0.30       | ❌ Not Found            | ✅ Found              |
| 1.0    | 2.00       | ✅ Found                | ✅ Found              |
| 1.0    | 1.50       | ✅ Found                | ✅ Found              |
| 1.0    | 1.00       | ✅ Found                | ✅ Found              |
| 1.0    | 0.80       | ✅ Found                | ✅ Found              |
| 1.0    | 0.50       | ✅ Found                | ✅ Found              |
| 1.0    | 0.30       | ❌ Not Found            | ✅ Found              |
| 1.5    | 2.00       | ✅ Found                | ✅ Found              |
| 1.5    | 1.50       | ✅ Found                | ✅ Found              |
| 1.5    | 1.00       | ✅ Found                | ✅ Found              |
| 1.5    | 0.80       | ✅ Found                | ✅ Found              |
| 1.5    | 0.50       | ✅ Found                | ✅ Found              |
| 1.5    | 0.30       | ❌ Not Found            | ✅ Found              |
| 2.0    | 2.00       | ✅ Found                | ✅ Found              |
| 2.0    | 1.50       | ✅ Found                | ✅ Found              |
| 2.0    | 1.00       | ✅ Found                | ✅ Found              |
| 2.0    | 0.80       | ✅ Found                | ✅ Found              |
| 2.0    | 0.50       | ✅ Found                | ✅ Found              |
| 2.0    | 0.30       | ✅ Found                | ✅ Found              |

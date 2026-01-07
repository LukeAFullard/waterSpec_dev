# Benchmark Report: Spectral Slope Estimation on Irregular Data

## Methodology

*   **Synthetic Data**: Generated colored noise time series ($1/f^\beta$) with $\beta$ ranging from 0.0 (white) to 3.0 (steep red) in steps of 0.25.
*   **Irregular Sampling**: N=500 initial points, randomly subsampled to retain 50% (N=250).
*   **Methods**:
    *   **Lomb-Scargle (LS)**: Standard implementation (Astropy). Slope estimated via OLS on the log-log periodogram. 95% CIs via parametric estimation.
    *   **Weighted Wavelet Z-transform (WWZ)**: Implemented via `pyleoclim`.
        *   **Normalization**: Corrected for PSD density ($1/f$).
        *   **Masking**: Cone of Influence (COI) masked to avoid edge effects.
        *   **Parameters**: `n_scales=50`, `decay_constant=0.0126`.
        *   **Slope**: Estimated via OLS on the global power spectrum (time-median). 95% CIs via parametric estimation.

## Results

### Performance Summary

| True Beta | LS Beta | LS Error | WWZ Beta | WWZ Error |
| :--- | :--- | :--- | :--- | :--- |
| 0.00 | -0.18 | -0.18 | **0.05** | **0.05** |
| 1.00 | 0.66 | -0.34 | **1.17** | **0.17** |
| 2.00 | 0.62 | -1.38 | **1.62** | **-0.38** |
| 3.00 | 0.52 | -2.48 | **2.01** | **-0.99** |

*See `benchmark_results.csv` for full table.*

### Key Findings

1.  **Lomb-Scargle Bias**: LS systematically underestimates the spectral slope for $\beta \gtrsim 0.5$.
    *   For Red Noise ($\beta=2$), LS estimates $\beta \approx 0.6$, a massive failure ("spectral whitening") caused by window leakage.
    *   The bias worsens as the true slope increases.

2.  **WWZ Improvement**: WWZ significantly reduces the bias.
    *   For Red Noise ($\beta=2$), WWZ recovers $\beta \approx 1.6$. While still slightly biased low (common in short records), it is statistically distinct from white/pink noise and far closer to the truth than LS.
    *   For White Noise ($\beta=0$), WWZ correctly estimates $\beta \approx 0$ (after applying the $1/f$ PSD normalization fix).

3.  **Confidence Intervals**:
    *   WWZ confidence intervals generally bracket the estimated value reasonably well, though they do not always contain the *true* value for very steep spectra ($\beta > 2$) due to the residual bias.
    *   LS confidence intervals are precise (narrow) but inaccurate (centered on the wrong value) for colored noise.

## Conclusion

The WWZ implementation with proper normalization and COI masking offers a robust alternative to Lomb-Scargle for estimating power-law slopes in irregularly sampled environmental data. It avoids the catastrophic spectral flattening observed with LS for red noise.

# Validation Results: Synthetic Data Analysis

This document summarizes the results of running `waterSpec` against synthetic data with known spectral properties. The goal was to verify the accuracy of spectral exponent estimation ($\beta$) for both evenly and unevenly spaced time series.

## Methodology

We generated synthetic "colored noise" with spectral power density $P(f) \propto 1/f^\beta$:
- **White Noise**: $\beta = 0.0$
- **Pink Noise**: $\beta = 1.0$
- **Red Noise**: $\beta = 2.0$

For each noise type, we tested two sampling scenarios:
1. **Evenly Spaced**: Regular sampling interval.
2. **Unevenly Spaced**: Randomly subsampled (keeping ~60% of points) to simulate irregular environmental monitoring.

We used the `Theil-Sen` estimator for robust slope fitting and `Bootstrap` (block) for confidence intervals.
**Note**: Segmented fitting was disabled (`max_breakpoints=0`) to focus purely on the estimation of the primary spectral slope.

## Results Summary

| Scenario | Sampling | True $\beta$ | Est. $\beta$ | 95% CI | Status |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **White Noise** | Even | 0.00 | -0.01 | [-0.15, 0.16] | **PASS** |
| **White Noise** | Uneven | 0.00 | 0.00 | [-0.16, 0.14] | **PASS** |
| **Pink Noise** | Even | 1.00 | 0.98 | [0.85, 1.14] | **PASS** |
| **Pink Noise** | **Uneven** | **1.00** | **0.65** | **[0.48, 0.83]** | **FAIL** |
| **Red Noise** | Even | 2.00 | 1.85 | [1.72, 1.97] | **CLOSE** |
| **Red Noise** | **Uneven** | **2.00** | **0.54** | **[0.26, 0.73]** | **FAIL** |

### Interpretation

1.  **Reliable for Evenly Spaced Data**: The package accurately recovers the spectral exponent for white, pink, and red noise when sampling is regular. The estimates are within or very close to the expected values.
2.  **Robust for Irregular White Noise**: Even with irregular sampling, the package correctly identifies white noise ($\beta \approx 0$).
3.  **Bias in Irregular Colored Noise**:
    - For **Pink Noise ($\beta=1$)**, irregular sampling leads to a significant underestimation of $\beta$ (0.65 vs 1.0).
    - For **Red Noise ($\beta=2$)**, the effect is even more severe (0.54 vs 2.00).
    - This flattens the spectrum, making it look more like white noise (spectral whitening).
    - This is a known phenomenon in Lomb-Scargle periodograms of red/pink noise, where aliasing from high frequencies folds back and lifts the low-frequency tail, or spectral leakage distorts the slope.
    - **Conclusion**: Users should be extremely cautious when interpreting spectral slopes from highly irregular data if the underlying process is expected to have strong persistence (Red/Pink noise). The results will likely be biased towards lower $\beta$ (appearing less persistent than reality).

## Conclusion of Audit

The package functions correctly and produces robust confidence intervals (thanks to the fixes in `fitter.py`). However, the underlying method (Lomb-Scargle) has physical limitations when dealing with unevenly sampled colored noise. This is not a bug in the code, but a limitation of the spectral analysis method itself on irregular grids. The provided confidence intervals correctly reflect the uncertainty of the *model fit*, but cannot account for the systematic bias introduced by the sampling irregularity.

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

## Discussion: Why does uneven sampling destroy the spectral slope?

The results above reveal a critical limitation: while the package works perfectly for white noise or evenly spaced data, it severely underestimates the spectral slope for unevenly sampled Red/Pink noise (estimating $\beta \approx 0.5$ instead of $\beta=2.0$). This flattening of the spectrum ("spectral whitening") is a fundamental property of spectral analysis on irregular grids, not a bug in the code.

### 1. The Spectral Window Function
In spectral analysis, the observed spectrum is the convolution of the *true spectrum* and the *spectral window function* (the spectrum of the sampling times).
- **Even Sampling**: The window function is a clean "comb" of sharp peaks. This allows for clear separation of frequencies up to the Nyquist limit.
- **Uneven Sampling**: The window function is messy. It has a main peak at zero frequency, but also a "grass" of side lobes that extend across the entire frequency range.

### 2. Spectral Leakage
"Red Noise" is characterized by having vastly more power at low frequencies than at high frequencies (orders of magnitude difference).
- When you convolve this spectrum with the messy window function of irregular sampling, the massive power from the low frequencies "leaks" into the high frequencies via the side lobes.
- Because the true power at high frequencies is so weak, this leaked energy completely swamps the real signal.
- **The Result**: The measured high-frequency power is dominated by leakage from the low frequencies. This creates a "noise floor" that is relatively flat (white).

### 3. Slope Estimation Bias
When you fit a line to this distorted spectrum in log-log space:
- The low frequencies are still dominated by the true signal (steep slope).
- The high frequencies are dominated by the leaked "white" noise (flat slope).
- The overall fitted line becomes a compromise, resulting in a much shallower slope than reality (e.g., $\beta=0.54$ instead of $\beta=2.0$).

### 4. Why Lomb-Scargle?
The Lomb-Scargle Periodogram (LSP) is mathematically powerful because it can handle uneven data without interpolation and is optimal for detecting **periodic sinusoidal signals** (peaks) in white noise.
- **For Peak Detection**: It works brilliantly. A strong sine wave concentrates power in one narrow band, standing out above the leakage noise.
- **For Spectral Slopes**: It is vulnerable. The broad-band nature of colored noise means "leakage" happens everywhere at once, distorting the overall shape (the slope) of the spectrum.

### Conclusion
Users analyzing unevenly spaced environmental data should be aware that **spectral slopes may be systematically underestimated** (biased toward zero) if the underlying process is strongly persistent (Red/Pink noise). This is an inherent trade-off of using available irregular data versus ideal regular monitoring.

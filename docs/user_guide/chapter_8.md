# Chapter 8: Quantifying Uncertainty (Bootstrapping & Surrogates)

## 8.1 The Need for Robust Error Bars

When analyzing environmental time-series data, traditional parametric error bounds (such as standard Ordinary Least Squares confidence intervals) are often wildly inaccurate. Environmental signals inherently possess "memory" (they are highly autocorrelated) and frequently exhibit skewed variance distributions. Consequently, standard statistical confidence intervals fail on spectral data because the residuals are both skewed and autocorrelated, fundamentally violating the assumptions of independent, identically distributed (i.i.d.) normal errors.

To overcome these challenges and estimate true uncertainty, **waterSpec** relies heavily on robust computational resampling methods, specifically utilizing advanced non-parametric bootstrapping and rigorous surrogate data testing.

## 8.2 Non-Parametric Bootstrapping

### Moving Block Bootstrap

For Haar fluctuation analysis, standard random resampling destroys the inherent time-dependency of the data. To address this, **waterSpec** employs a **Moving Block Bootstrap** approach. This technique resamples contiguous blocks of data rather than individual data points, effectively preserving the local autocorrelation structure of the time-series while generating robust confidence intervals for the fluctuation metrics.

### Wild Bootstrap (for OLS/Spectral fits)

When performing Ordinary Least Squares (OLS) regression on spectral power fits, the residuals often exhibit heteroscedasticity (changing variance) and heavy skewness. The **Wild Bootstrap** is a resampling technique specifically designed to handle heteroscedasticity by multiplying the residuals by a random variable before reconstructing the dependent variable.

**Crucial Mathematical Implementation:**
Unlike many standard implementations that utilize a simple Rademacher coin-flip distribution (randomly assigning $1$ or $-1$), **waterSpec** deliberately applies the **Mammen (1993)** two-point distribution. The Mammen distribution is defined such that it explicitly preserves the third moment of the distribution, ensuring that $E[W^3]=1$. This mathematical property is strictly necessary to accurately capture and preserve the skewed nature of log-log periodogram residuals.

> **Note:** Preserving the skewness of residuals (via the Mammen distribution) is essential for spectral fits because log-transformed spectral power inherently follows a highly skewed, non-Gaussian distribution. Failing to account for this skewness would result in symmetric, under-penalized confidence bounds that do not represent the true error landscape of the spectrum.

## 8.3 Surrogate Data Testing

### What are Surrogates?

To test whether an observed complex signal (such as a Cross-Haar correlation) is statistically significant rather than a byproduct of random chance, we must compare it against a null hypothesis. We do this by generating **surrogate data**: "fake" datasets artificially constructed to preserve certain fundamental statistical properties of the original data (like the mean, variance, and often the power spectrum) while completely destroying the specific property we are testing for (such as phase coherence or temporal order). By analyzing hundreds of surrogates, we can observe the natural distribution of our metric under the null hypothesis.

### Phase-Randomized Surrogates (Tk95 / Power Law)

For time-series that are unevenly sampled, **waterSpec** natively utilizes the `generate_power_law_surrogates` function.

This function implements the well-established Timmer & Koenig (1995) method to generate synthetic power-law noise. **Crucially, waterSpec directly resamples the generated fractal noise to the observed irregular grid *without* applying a prior Butterworth anti-aliasing filter.** This deliberate architectural choice intentionally avoids the artificial flattening of high-frequency spectra that conventional filters cause on irregular grids. Furthermore, the algorithm applies a global variance scaling factor to strictly preserve the statistical validity of the generated surrogates, ensuring they accurately mirror the amplitude bounds of the original observation.

### Iterative Amplitude Adjusted Fourier Transform (IAAFT)

When dealing with strictly evenly sampled data, **waterSpec** provides the `generate_iaaft_surrogates` function.

The IAAFT is a powerful iterative algorithm that guarantees the preservation of two critical properties simultaneously: it perfectly maintains the exact amplitude distribution (the histogram) of the original data, *and* it iteratively converges to match the power spectrum of the original data. This dual-preservation makes IAAFT surrogates extremely robust for testing non-linear dynamics and phase synchronization in regular time-series.

### Block-Shuffled / Permutation Surrogates

When testing for independence between two variables (e.g., cross-correlation significance), it is necessary to destroy the time-alignment between them. **waterSpec** achieves this using block-shuffled permutation surrogates.

To prevent the creation of a zero-variance static "anchor" from the unshuffled leftover tail of the array, **waterSpec** applies a random cyclic shift to the final array after block shuffling. This guarantees complete temporal decoupling without breaking the global variance.

## 8.4 Calculating Empirical P-Values

Once a distribution of surrogate metrics has been computed, it must be translated into a formal statistical probability, or $p$-value.

**waterSpec** calculates empirical $p$-value using the strictly conservative formula:

$$ p = \frac{k + 1}{n + 1} $$

Where:
- $n$ is the total number of surrogate iterations.
- $k$ is the number of surrogate results that are equal to or more extreme than the actual observed data metric.

**Technical Detail:** **waterSpec** rigidly defines the denominator $n$ based strictly on the *requested* number of surrogate iterations. This safety mechanism ensures that if any surrogate iterations fail or return `NaN` (due to computational edge cases), the failure inherently penalizes the test conservatively rather than artificially shrinking the denominator and thereby falsely inflating statistical significance.
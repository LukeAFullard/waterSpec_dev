# Chapter 4: Choosing Your Method: Lomb-Scargle vs. Haar

Welcome to Chapter 4. As an environmental data scientist or hydrologist, you will frequently encounter datasets that are far from perfect. In the real world, continuous high-frequency sensor readings often mix with sporadic, weekly grab-samples, creating irregular time series that are difficult to analyze.

The **waterSpec** package provides two primary, mathematically distinct tools for analyzing the frequency domain of such challenging data: the Lomb-Scargle Periodogram and Haar Wavelet Fluctuation Analysis.

It is crucial to emphasize that these two methods ask fundamentally different mathematical questions of your data. Using the wrong method for your specific analytical goal or data structure can lead to significantly biased conclusions. This chapter is designed to guide you toward making the right scientific choice.

## 4.1 The Lomb-Scargle Periodogram

### What it does
The Lomb-Scargle Periodogram is a powerful technique that estimates the frequency spectrum of a time series by fitting sine and cosine waves to the data via least-squares regression. Because it mathematically evaluates each frequency independently, it is widely considered the gold standard for finding *specific periodicities* in a dataset, such as an exactly daily (24-hour) cycle driven by evapotranspiration, or a precise seasonal (annual) cycle.

### Best use cases
You should use the Lomb-Scargle approach when you are explicitly searching for narrowband signals or dominant repeating frequencies. It performs exceptionally well when your data is relatively evenly sampled, even if it contains minor gaps or occasional missing data points.

### The Trap (Variance Inflation)
Despite its strengths, the Lomb-Scargle Periodogram has a major vulnerability when used for broader spectral slope analysis. When your data is highly irregularly sampled or contains massive, extended gaps, the Lomb-Scargle spectrum inherently introduces a high-frequency bias known as variance inflation.

This bias artificially flattens the power-law spectral slope ($\beta$), making a strongly persistent or correlated system incorrectly appear as uncorrelated "white noise". Because of this mathematical danger, **waterSpec** will actively emit a `UserWarning` if it detects uneven sampling when you attempt a Lomb-Scargle slope fit.

## 4.2 Haar Wavelet Fluctuation Analysis

### What it does
Instead of attempting to fit continuous, globally oscillating waves, Haar Wavelet Fluctuation Analysis takes a localized approach. It calculates the difference (the fluctuation) between the averages of adjacent time windows of varying sizes (lags, denoted as $\tau$). This step-like comparison is highly robust against data irregularity because it assesses the variance across different temporal scales rather than looking for a perfect sine wave.

### Best use cases
Haar Wavelet Fluctuation Analysis is **waterSpec**'s strongly recommended tool for determining robust power-law spectral slopes ($\beta$) on irregular, gappy, or unevenly sampled data. A major advantage of this method is that it natively handles uneven data and absolutely *does not require interpolation*—a process that notoriously destroys the true spectral properties of environmental time series.

### The Math Translation
Understanding how the Haar fluctuation relates to the power spectrum is straightforward. As the window size or lag $\tau$ increases, the average magnitude of the fluctuation $S(\tau)$ scales according to a power law: $S(\tau) \propto \tau^m$, where $m$ is the Haar fluctuation slope.

The direct mathematical relationship translating this fluctuation slope to the power spectrum exponent ($\beta$) is:
$$\beta = 2m + 1$$

By analyzing the scaling of variances at different window sizes $\tau$, you can accurately estimate $\beta$ without the variance inflation traps of periodograms.

## 4.3 Decision Matrix

To ensure you choose the mathematically appropriate technique for your analysis, refer to the following decision matrix. Match your scientific goal and data characteristics to find the recommended approach in **waterSpec**:

*   *Goal:* Finding a repeating 24-hour cycle? -> **Choose Lomb-Scargle.**
*   *Goal:* Estimating the long-term memory/spectral slope ($\beta$)? -> **Choose Haar Wavelets.**
*   *Data:* Highly irregular grab samples over 10 years? -> **Choose Haar Wavelets.**
*   *Data:* 15-minute sensor data with a few missing hours? -> **Lomb-Scargle is safe.**

By matching your analytical goals and data structure to the appropriate mathematical tool, you ensure that your hydrologic and environmental interpretations remain scientifically sound and mathematically rigorous.

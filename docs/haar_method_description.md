# Statistical Description of the Haar Structure Function Method for Beta Estimation

## Overview

The Haar Structure Function (HSF) method is a robust technique for estimating the spectral slope ($\beta$) of a time series, particularly suited for data that is short, irregularly sampled, or non-stationary. This method, advocated by Lovejoy and Schertzer and applied in recent paleoclimate studies (e.g., Hébert et al., 2021), operates in the time domain using Haar wavelets, avoiding many of the artifacts associated with Fourier-based spectral analysis on sparse data (such as fractal aliasing).

## Mathematical Formulation

The core of the method is the calculation of the first-order structure function, $S_1(\Delta t)$, which measures the average magnitude of fluctuations at different time scales (lags), $\Delta t$.

### 1. Haar Fluctuation ($\Delta F$)

For a given time interval $\Delta t$, the Haar fluctuation is defined as the difference between the mean of the signal in the second half of the interval and the mean of the signal in the first half.

Let $C(t)$ be the time series concentration (or value) at time $t$. The fluctuation over an interval $[t, t + \Delta t]$ is:

$$
\Delta F(\Delta t) = \overline{C}_{(t+\Delta t/2, t+\Delta t)} - \overline{C}_{(t, t+\Delta t/2)}
$$

Where $\overline{C}_{(a, b)}$ denotes the mean value of the data points falling within the time interval $(a, b)$.

**Note on Scaling:**
In some formulations, the fluctuation is defined as a derivative approximation ($\frac{\Delta \overline{C}}{\Delta t}$). However, for the purpose of estimating the standard scaling exponent $H$ consistent with $1/f^\beta$ noise (where white noise corresponds to $H=-0.5$), we utilize the difference of means directly. Dividing by $\Delta t$ would shift the exponent by -1, leading to inconsistent interpretation of standard noise colors.

### 2. The Structure Function ($S_1$)

The first-order structure function is the average of the absolute values of these fluctuations over the entire time series:

$$
S_1(\Delta t) = \langle |\Delta F(\Delta t)| \rangle
$$

For irregular data, this average is computed by identifying all available non-overlapping pairs of intervals of duration $\Delta t/2$ and computing the difference of their means. Our implementation uses a sliding window approach that maximizes data usage while ensuring that each calculated fluctuation represents a distinct, non-overlapping segment locally (though the search for segments scans the whole series).

### 3. Fractal Scaling and Exponent $H$

In fractal processes, the structure function follows a power law scaling relationship with the time lag:

$$
S_1(\Delta t) \propto \Delta t^{H}
$$

By plotting $\log(S_1)$ against $\log(\Delta t)$, the fluctuation scaling exponent $H$ can be estimated as the slope of the linear fit.

## Relation to Spectral Slope ($\beta$)

The exponent $H$ derived from the Haar analysis is directly related to the power spectral density slope $\beta$ (where $P(f) \propto f^{-\beta}$) by the following relation:

$$
\beta = 1 + 2H
$$

### Interpretation of Regimes

The value of $\beta$ (and consequently $H$) provides insight into the "color" or memory of the noise process:

| Noise Type | Beta ($\beta$) | Haar Exponent ($H$) | Description |
| :--- | :--- | :--- | :--- |
| **White Noise** | $\approx 0$ | $\approx -0.5$ | No correlation; memoryless process. |
| **Pink Noise** | $\approx 1$ | $\approx 0$ | $1/f$ noise; long-range dependence. |
| **Brownian Noise** | $\approx 2$ | $\approx 0.5$ | Random walk; integrated white noise. |
| **Black Noise** | $> 2$ | $> 0.5$ | Strong persistence/trends. |

## Advantages for Environmental Data

1.  **Robustness to Gaps:** Unlike the Fast Fourier Transform (FFT), which requires evenly spaced data, the HSF method naturally handles gaps. It simply skips intervals where data is missing, calculating statistics only on valid segments.
2.  **Short Time Series:** Spectral methods often become unstable or yield high variance for short records ($N < 100$). The HSF method provides a more stable estimate of the scaling behavior by averaging fluctuations in the time domain.
3.  **Stationarity:** The method can effectively distinguish between stationary ($H < 0$) and non-stationary ($H > 0$) regimes, a distinction that is often blurred in periodogram analysis.

## Implementation Details

Our implementation (`src/waterSpec/haar_analysis.py`) performs the following steps:
1.  **Lag Generation:** Generates a sequence of logarithmically spaced lag times ($\Delta t$) from the minimum resolution up to half the series duration.
2.  **Fluctuation Calculation:** For each $\Delta t$, iterates through the time series to find valid windows $[t, t+\Delta t/2)$ and $[t+\Delta t/2, t+\Delta t)$ containing data.
3.  **Averaging:** Computes the mean difference for each valid window and averages their absolute values.
4.  **Fitting:** Performs a linear regression on the log-log data to determine $H$ and $\beta$.

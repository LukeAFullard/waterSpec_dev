# Understanding the Relationship Between Spectral Slope ($\beta$) and Haar Slope ($m$)

A common point of confusion in spectral analysis is the relationship between the power spectral density slope, $\beta$, and the scaling exponent derived from fluctuation analysis (like Haar wavelets or structure functions).

In `waterSpec`, we use the relationship:
$$ \beta = 2m + 1 $$
where $m$ is the slope of the Haar fluctuation plot ($\log S_1$ vs $\log \Delta t$).

However, literature on fractional Gaussian noise (fGn) often cites:
$$ \beta = 2H - 1 $$
where $H$ is the Hurst exponent.

This document clarifies why these formulas differ and how they are consistent.

## 1. The Definitions

### The Hurst Exponent ($H$)
The Hurst exponent $H$ is a parameter defined for self-similar processes, specifically **Fractional Brownian Motion (fBm)** and its derivative, **Fractional Gaussian Noise (fGn)**. By definition, $0 < H < 1$.

*   **fBm (Non-stationary, "Random Walk"-like):**
    *   Power Spectrum: $S(f) \propto f^{-(2H+1)}$
    *   Spectral Slope: $\beta = 2H + 1$
    *   Range of $\beta$: $1 < \beta < 3$

*   **fGn (Stationary, "Noise"-like):**
    *   Power Spectrum: $S(f) \propto f^{-(2H-1)}$
    *   Spectral Slope: $\beta = 2H - 1$
    *   Range of $\beta$: $-1 < \beta < 1$ (typically $0 < \beta < 1$ in nature)

### The Haar Fluctuation Slope ($m$)
`waterSpec` measures the scaling of the first-order Haar structure function $S_1(\Delta t)$, which is essentially the average absolute difference of means in a window of size $\Delta t$. We calculate $m$ as the slope of this scaling.

## 2. Unifying fGn and fBm

The Haar method applied directly to a time series produces a slope $m$ that varies continuously as the process transitions from stationary noise (fGn) to non-stationary motion (fBm).

### Case A: Analyzing fBm (Non-stationary, $\beta > 1$)
When the time series is a geometric random walk (fBm) with Hurst exponent $H$:
*   The difference of means scales with the increments of the walk.
*   Increments of fBm scale as $\Delta t^H$.
*   Therefore, the Haar slope $m \approx H$.
*   Substituting into the fBm spectral formula:
    $$ \beta = 2H + 1 \Rightarrow \beta = 2m + 1 $$

### Case B: Analyzing fGn (Stationary, $\beta < 1$)
When the time series is stationary noise (fGn) with Hurst exponent $H$:
*   The time series is the derivative (increments) of an fBm.
*   The mean of the noise over a window $\Delta t$ scales as $\Delta t^{H-1}$.
*   Therefore, the Haar slope $m \approx H - 1$.
*   Substituting into the fGn spectral formula:
    $$ \beta = 2H - 1 $$
    $$ \beta = 2(m + 1) - 1 $$
    $$ \beta = 2m + 2 - 1 $$
    $$ \beta = 2m + 1 $$

## 3. Conclusion

The formula **$\beta = 2m + 1$** is universal for the Haar analysis implemented in `waterSpec`. It correctly recovers the spectral slope for both stationary ($\beta < 1$) and non-stationary ($\beta > 1$) processes without needing to know a priori which regime the data belongs to.

*   If you measure $m = -0.5$, then $\beta = 0$ (White Noise).
*   If you measure $m = 0$, then $\beta = 1$ (Pink Noise).
*   If you measure $m = 0.5$, then $\beta = 2$ (Brown Noise).

The confusion arises only if one strictly interprets the measured slope $m$ as the theoretical Hurst exponent $H$. In reality:
*   For fBm, $m = H$.
*   For fGn, $m = H - 1$.

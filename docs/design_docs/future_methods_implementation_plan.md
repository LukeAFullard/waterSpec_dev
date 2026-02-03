# Implementation Plan for Future Irregular Spectral Methods

This document outlines the design and implementation strategy for three high-priority additions to `waterSpec`. These methods are selected because they address critical gaps in analyzing environmental time series (time-localized correlation, lead-lag relationships, and process modeling) while strictly adhering to the **"No Interpolation"** constraint for irregularly sampled data.

---

## 1. Weighted Wavelet Z-transform (WWZ) Coherence

### Overview
WWZ Coherence extends the existing WWZ implementation to estimating the time-localized correlation between two time series. Unlike global cross-correlation (which gives an average over time), WWZ Coherence produces a time-frequency map showing *when* and at *what scale* two variables are coupled.

### Implementation Strategy
1.  **Extend `wwz.py`:**
    *   Leverage the existing `calculate_wwz` projection logic.
    *   Compute the **Cross-Wavelet Spectrum** $W_{xy}(\tau, \omega)$. This requires calculating the inner product of series $X$ with the wavelet basis, series $Y$ with the wavelet basis, and then computing $W_{xy} = W_x \cdot W_y^*$.
    *   Note: The standard WWZ returns a "Z-statistic" (power), not the complex wavelet coefficient. We need to expose the complex coefficients $W_x$ and $W_y$ derived from the weighted least-squares projection onto the $[\cos(\omega t), \sin(\omega t)]$ basis.
2.  **Calculate Coherence:**
    *   Compute the Magnitude Squared Coherence:
        $$ R^2(\tau, \omega) = \frac{|S(W_{xy}(\tau, \omega))|^2}{S(|W_x(\tau, \omega)|^2) \cdot S(|W_y(\tau, \omega)|^2)} $$
    *   **Smoothing ($S$):** Coherence requires smoothing in time and/or frequency to yield values between 0 and 1 (otherwise, pointwise coherence is always 1). For irregular data, smoothing must be done carefully, likely using a Gaussian kernel on the $(\tau, \omega)$ grid.

### Key Questions Answered
*   **Transient Coupling:** *"Did the correlation between rainfall and turbidity disappear during the 2018 drought?"*
*   **Regime Shifts:** *"Is the system driven by seasonal cycles in the 1990s but by storm events in the 2020s?"*
*   **Mechanism Changes:** *"Does the coherence shift from low-frequency (baseflow) to high-frequency (runoff) during extreme wet years?"*

### Pros & Cons
*   **Pros:**
    *   **Irregular Data Native:** No interpolation required; bias-free for gappy data.
    *   **Time-Localized:** Reveals non-stationary relationships invisible to global spectra.
*   **Cons:**
    *   **Computationally Expensive:** $O(N^2)$ or worse without optimization.
    *   **Smoothing Complexity:** Defining the correct smoothing operator $S$ on an irregular grid is non-trivial.

### Caveats & Limitations
*   **Edge Effects:** Like all wavelet methods, results near the start/end of the series are unreliable (Cone of Influence).
*   **Sparse Overlap:** Requires simultaneous data points (or close enough to fall in the same wavelet window). If $X$ and $Y$ are sampled at completely different times with no overlap in the wavelet decay window, coherence is undefined.

---

## 2. Lomb-Scargle Cross-Spectrum & Phase Analysis

### Overview
The Generalized Lomb-Scargle (GLS) periodogram can be extended to bivariate analysis to estimate the **Cross-Spectrum**. The primary value here is the **Phase Spectrum** $\phi(f)$, which quantifies the time delay (lead/lag) between two variables as a function of frequency.

### Implementation Strategy
1.  **Extend `spectral_analyzer.py`:**
    *   Implement a **Complex Lomb-Scargle** or **Bi-variate Least Squares** fit.
    *   For each frequency $f$:
        *   Fit a joint model to series $X(t_i)$ and $Y(t_j)$:
            $$ \hat{X}(t) = A_x \cos(2\pi f t) + B_x \sin(2\pi f t) $$
            $$ \hat{Y}(t) = A_y \cos(2\pi f t) + B_y \sin(2\pi f t) $$
        *   Wait: Standard LS fits separately. For Cross-Spectrum, we compute the Fourier Transforms $\mathcal{F}_x(f)$ and $\mathcal{F}_y(f)$ using the LS coefficients.
    *   **Cross-Spectral Density (CSD):** $P_{xy}(f) = \mathcal{F}_x(f) \cdot \mathcal{F}_y^*(f)$.
    *   **Phase:** $\phi_{xy}(f) = \text{angle}(P_{xy}(f))$.
    *   **Time Lag:** $\text{Lag}(f) = \frac{\phi_{xy}(f)}{2\pi f}$.

### Key Questions Answered
*   **Transport Delays:** *"Does Discharge lead Nitrate concentration (flushing) or lag it?"*
*   **Frequency-Dependent Hysteresis:** *"Is the lag 2 days at the storm scale (quickflow) but 3 months at the seasonal scale (groundwater)?"*
*   **Causality Direction:** Phase slope can indicate the direction of propagation in a river network.

### Pros & Cons
*   **Pros:**
    *   **Precise Lags:** Provides continuous lag estimates, often more precise than discrete time-domain cross-correlation lag steps.
    *   **Frequency Specific:** Separates delays by timescale.
*   **Cons:**
    *   **Phase Wrapping:** Phase is cyclic ($-\pi$ to $\pi$). Large delays can "wrap around," making interpretation ambiguous without careful unwrapping.
    *   **Noise Sensitivity:** Phase estimates are very noisy when Coherence is low.

### Caveats & Limitations
*   **Stationarity Assumption:** Assumes the lag is constant over the entire time series. If the lag changes (e.g., wet vs dry years), this method averages it, potentially yielding a meaningless mean.
*   **Aliasing:** Highly irregular sampling can introduce spectral aliasing, creating "ghost" peaks and phase distortions.

---

## 3. CARMA (Continuous AutoRegressive Moving Average) Models

### Overview
CARMA models fit a stochastic differential equation (SDE) directly to the time series. Instead of asking "what is the slope?", CARMA asks "what is the process?". It is the rigorous statistical baseline for irregular memory processes.

### Implementation Strategy
1.  **New Module `stochastic_process.py`:**
    *   Use a **Kalman Filter** approach to calculate the likelihood of a CARMA(p,q) model given irregular data.
    *   **Optimization:** Use `scipy.optimize` or `emcee` (MCMC) to find the parameters (autoregressive roots, moving average coefficients) that maximize the likelihood.
    *   *Reference:* The `celerite` or `carma_pack` libraries (astronomy) are the standard reference implementations. We can implement a simplified CARMA(1,0) (Ornstein-Uhlenbeck/Damped Random Walk) first.
2.  **Model Selection:**
    *   Implement AIC/BIC calculation to choose the order (p, q).

### Key Questions Answered
*   **Process Identification:** *"Is this river a Random Walk (Brown Noise) or a Damped Random Walk (Mean Reverting)?"*
*   **Characteristic Timescales:** *"What is the e-folding recovery time (damping timescale) of the catchment after a pollution event?"* (Derived directly from CARMA parameters).
*   **Forecasting:** Provides a mathematically optimal way to forecast irregular data or fill gaps with uncertainty intervals.

### Pros & Cons
*   **Pros:**
    *   **Physically Interpretable:** Parameters map directly to SDE coefficients (damping, forcing).
    *   **Gold Standard:** The most statistically rigorous way to handle irregular correlation.
*   **Cons:**
    *   **Complexity:** Harder to explain to stakeholders than a "slope".
    *   **Computation:** Fitting can be unstable; MCMC is slow.
    *   **Stationarity:** Standard CARMA assumes stationarity (constant parameters).

### Caveats & Limitations
*   **Model Misspecification:** If the data is not well-described by a Gaussian process (e.g., highly flashy, non-Gaussian spikes), CARMA residuals will be invalid.
*   **Data Requirements:** Requires enough data points to constrain the parameters (typically N > 100 for simple models, more for high-order).

---

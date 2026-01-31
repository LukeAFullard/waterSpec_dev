# Proposal for New Features in waterSpec

**Date:** 31 January 2026
**Author:** Jules (AI Software Engineer)
**Status:** Proposal / Design Phase

## 1. Executive Summary

This document outlines the design and implementation plan for two advanced features requested to enhance the capabilities of `waterSpec` for river and lake water quality analysis:

1.  **Multivariate Haar Analysis:** Specifically, **Partial Cross-Haar Correlation**, to disentangle direct relationships between water quality variables (e.g., Concentration vs. Discharge) from confounding variables (e.g., Precipitation).
2.  **Event-Based Segmentation:** An automated method to segment time series into distinct hydrologic regimes (e.g., "Storm" vs. "Baseflow") using **Sliding Haar Fluctuations**.

These features aim to provide more rigorous causal attribution and regime-dependent analysis, which are critical for both scientific understanding and legal defensibility.

---

## 2. Multivariate Haar Analysis (Partial Cross-Haar)

### 2.1. Problem Statement
In hydrological systems, a correlation between two variables (e.g., Nitrate Concentration $C$ and Discharge $Q$) may be spurious or mediated by a third variable (e.g., Precipitation $P$). Standard Cross-Haar correlation ($\rho_{CQ}(\tau)$) does not account for this. We need a way to quantify the correlation between $C$ and $Q$ at scale $\tau$ *after removing the linear effect of $P$*.

### 2.2. Mathematical Methodology
We propose to implement **Partial Cross-Correlation** on the Haar fluctuations.

Let $\Delta C(t, \tau)$, $\Delta Q(t, \tau)$, and $\Delta P(t, \tau)$ be the Haar fluctuations of Concentration, Discharge, and Precipitation at time $t$ and scale $\tau$.

The Partial Cross-Haar Correlation $\rho_{CQ \cdot P}(\tau)$ is given by:

$$
\rho_{CQ \cdot P}(\tau) = \frac{\rho_{CQ}(\tau) - \rho_{CP}(\tau) \rho_{QP}(\tau)}{\sqrt{1 - \rho_{CP}^2(\tau)} \sqrt{1 - \rho_{QP}^2(\tau)}}
$$

where $\rho_{XY}(\tau)$ is the standard Pearson correlation between the fluctuations $\Delta X$ and $\Delta Y$ at scale $\tau$.

**Significance Testing:**
We will assess significance using a semi-parametric bootstrap or surrogate approach:
1.  Generate surrogates for $C$ (keeping $Q$ and $P$ fixed) using Phase Randomization (Method 5.2).
2.  Calculate $\rho_{CQ \cdot P}^*$ for each surrogate.
3.  Empirical p-value estimates the probability that the partial correlation is non-zero.

### 2.3. Implementation Design

**New Class:** `MultivariateAnalysis` in `src/waterSpec/multivariate.py`.

```python
class MultivariateAnalysis:
    def __init__(self, inputs: List[TimeSeriesInput]):
        # Handles alignment of 3+ variables
        pass

    def run_partial_cross_haar(self,
                               target_var1: str,
                               target_var2: str,
                               conditioning_vars: List[str],
                               lags: np.ndarray) -> Dict:
        """
        Calculates partial correlation for each lag.
        """
        pass
```

**Key Steps:**
1.  **Alignment:** Align all $N$ variables to a common timeline (intersection).
2.  **Fluctuation Calculation:** Compute Haar fluctuations for all variables at each lag $\tau$.
3.  **Correlation Matrix:** Compute the correlation matrix of fluctuations at each lag.
4.  **Partial Correlation:** Apply the recursion formula or matrix inversion (inverse covariance matrix) to find partial correlations.

---

## 3. Event-Based Segmentation

### 3.1. Problem Statement
Water quality dynamics often shift drastically during events (storms). Analyzing the entire time series as a single process can obscure these shifts. "Sliding Haar" analysis provides a real-time metric of volatility that can be used to automatically detect and segment these events.

### 3.2. Methodology
1.  **Compute Sliding Haar:** Calculate fluctuation $F(t, \tau_{event})$ at a characteristic event scale (e.g., $\tau_{event} = 6$ hours).
2.  **Thresholding:** Define an event when $F(t, \tau_{event}) > k \cdot \sigma_{background}$.
    *   $\sigma_{background}$ can be the median absolute deviation (MAD) of the fluctuation series.
    *   $k$ is a sensitivity parameter (e.g., 3).
3.  **Segmentation:**
    *   **Event Regime:** Periods where threshold is exceeded (plus buffer).
    *   **Baseflow Regime:** All other periods.
4.  **Regime Analysis:** Perform separate Spectral/Haar analyses on the concatenated "Event" segments and "Baseflow" segments.

### 3.3. Implementation Design

**New Module:** `src/waterSpec/segmentation.py`.

```python
def segment_by_fluctuation(
    time: np.ndarray,
    data: np.ndarray,
    scale: float,
    threshold_factor: float = 3.0,
    min_event_duration: float = 0
) -> Tuple[List[Slice], List[Slice]]:
    """
    Returns lists of time slices for 'events' and 'background'.
    """
    pass

class RegimeAnalysis:
    def __init__(self, original_analysis: Analysis):
        pass

    def run_regime_comparison(self, event_slices, background_slices):
        # Runs Haar analysis on concatenated segments
        pass
```

---

## 4. Work Plan for Implementation

1.  **Phase 1: Multivariate Core**
    *   Implement `MultivariateAnalysis` with robust alignment for $N$ variables.
    *   Implement Partial Correlation formula.
    *   Add unit tests with synthetic chain structures (X -> Y -> Z).

2.  **Phase 2: Segmentation Logic**
    *   Implement `segment_by_fluctuation`.
    *   Add visualization tools (plot time series colored by regime).
    *   Add tests using synthetic bursty data.

3.  **Phase 3: Integration**
    *   Expose these features via the main `Analysis` API or a new `AdvancedAnalysis` wrapper.
    *   Update documentation and examples.

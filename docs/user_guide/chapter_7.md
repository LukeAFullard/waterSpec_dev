# Chapter 7: Bivariate and Multivariate Dynamics

## Introduction

In the natural world, environmental variables rarely act alone; their dynamics are often coupled, driving complex system responses. To deeply understand these interactions, it is necessary to go beyond simple scatter plots and investigate how variables relate to one another across different timescales. The `BivariateAnalysis` class is **waterSpec**'s primary tool for comparing two time series directly in the time-scale domain.

Because environmental data is frequently sampled at varying or irregular intervals, **waterSpec** provides robust methods for initializing and aligning your data prior to analysis. The snippet below demonstrates how to initialize a `BivariateAnalysis` object and align two variables—such as Concentration and Discharge—that might have been recorded at slightly different irregular intervals:

```python
from waterSpec.bivariate import BivariateAnalysis

# Initialize with time and data arrays, along with descriptive labels
biv = BivariateAnalysis(time_c, data_c, "Conc", time_q, data_q, "Q")

# Align the data to account for irregular sampling intervals
biv.align_data(tolerance=3600, method='nearest') # 1 hour tolerance
```

## 7.1 Cross-Spectra and Coherence (Lomb-Scargle)

Understanding the temporal relationship between two signals often involves answering questions like "Does Variable A lead Variable B?" and "At what frequencies do they resonate together?" The `calculate_ls_cross_spectrum` function in **waterSpec** is designed exactly for this purpose.

**What it does:**
This function calculates the complex Lomb-Scargle cross-spectrum directly from unevenly or irregularly sampled data. By estimating the cross-spectrum, you can extract the phase lag between the two variables, revealing the exact timing differences and dominant frequencies of their interaction without the need for error-prone interpolation.

**Technical Detail:**
When calculating cross-spectra, numerical instability can arise if the time values are very large (e.g., standard Unix timestamps), leading to catastrophic precision loss. To counteract this, **waterSpec** explicitly centers the time vectors against a common reference boundary before proceeding with the matrix-solving steps. This critical mathematical stabilization preserves the exact mathematical phase differences, ensuring high-fidelity coherence and phase lag estimates regardless of your data's temporal origin.

## 7.2 Cross-Haar Correlation

While standard Pearson correlation provides a useful summary of the overall linear relationship between two variables, it only yields a single number, effectively blurring out the scale-dependent nature of environmental dynamics. The `run_cross_haar_analysis` function overcomes this limitation.

**What it does:**
Cross-Haar correlation calculates how two variables correlate *at specific scales*. For instance, two variables might exhibit a strong positive correlation during short-term, daily events (such as immediate storm runoff) but show a negative correlation over longer, seasonal scales (such as prolonged dry spells). By partitioning the correlation across different time windows, Cross-Haar analysis uncovers these hidden, scale-dependent relationships, providing a much richer understanding of system behavior.

## 7.3 Hysteresis Classification (C-Q Dynamics)

In hydrology and biogeochemistry, the relationship between two variables—like Concentration and Discharge (C-Q)—often exhibits hysteresis loops. A hysteresis loop occurs when the value of one variable differs depending on whether another variable is increasing or decreasing. For example, a pollutant's concentration might be significantly higher on the rising limb of a flood event (as accumulated sediment is flushed out) compared to the falling limb.

**How waterSpec calculates it:**
The `calculate_hysteresis_metrics(tau)` function in **waterSpec** quantifies both the Loop Area and the Direction of the hysteresis (i.e., whether the loop is Clockwise or Counter-Clockwise), giving you a precise numerical classification of the C-Q dynamics.

- *Crucial Mathematical Detail:* To accurately calculate the signed polygon area of the hysteresis loop, **waterSpec** uses the explicitly closed "Shoelace formula." This ensures that the geometric area is computed with exact mathematical closure, accurately capturing the net direction of the loop.

> **Note:**
> *Crucial Normalization Detail:* To safely evaluate continuous, multi-event time series, **waterSpec** applies the continuous normalization metric introduced by Zuecco et al. (2016). It divides the accumulated loop area by `std_x * std_y * estimated_cycles` (where `estimated_cycles = max(1.0, time_span / tau)`). This critical step prevents the calculated area from meaninglessly inflating over time, ensuring that the hysteresis metric remains a reliable indicator of system behavior regardless of the time series length.

## 7.4 Partial Cross-Haar (Experimental)

In complex environmental systems, interpreting bivariate correlation can be tricky due to the confounding influence of external factors. For instance, Concentration and Discharge might appear strongly correlated simply because they are both simultaneously driven by a third, hidden variable, such as Rainfall.

**What it does:**
The `calculate_partial_cross_haar` function attempts to isolate the true relationship between two variables by calculating the Cross-Haar correlation between Variable A and Variable B *while controlling for* Variable C. This helps you determine whether the observed scale-dependent correlation is direct or merely a spurious artifact of shared external forcing.

> **Warning:**
> The Partial Cross-Haar feature is currently marked as **Experimental** in the **waterSpec** codebase. Its statistical validity for non-stationary Haar fluctuations is still under active investigation. Researchers should exercise caution and critically interpret the results when applying this methodology to complex, non-stationary environmental data.

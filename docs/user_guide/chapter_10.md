# Chapter 10: Troubleshooting and FAQs

Real-world environmental data is inherently messy—plagued by gaps, varying sampling rates, and extreme outlier events. This kind of data routinely pushes statistical algorithms to their limits. **waterSpec** is intentionally designed to fail safely and warn the user, rather than silently producing mathematically invalid results or crashing your pipeline.

If you have encountered a warning or a failed fit, don't worry. This chapter explains what the software is doing under the hood to protect your analysis and provides actionable solutions.

## 10.1 Common Warnings Explained

When analyzing irregular or sparse time series, you may see warnings appear in your console. Here is what they mean and how to address them:

> **"Minimum effective sample size is X. Confidence intervals from standard OLS may be underestimated..."**

*   **Meaning:** Your dataset is either extremely short or highly fragmented (gappy). At large analysis scales (large lags), there simply aren't enough independent, non-overlapping windows left to calculate the variance reliably.
*   **Solution:** You can generally safely ignore the standard Ordinary Least Squares (OLS) confidence intervals when this occurs. **waterSpec** anticipates this and natively relies on a robust Weighted Least Squares (WLS) implementation. The WLS algorithm explicitly sets a lower bound on effective degrees of freedom (`max(0.5, n_eff)`) and automatically downweights these poorly sampled, large-lag scales. If you prefer to remove the warning entirely, simply reduce your `max_lag` parameter so the algorithm doesn't attempt to evaluate scales larger than your data can support.

> **"RSS extremely small; excluding from BIC comparison."**

*   **Meaning:** The model fit your data perfectly, resulting in a Residual Sum of Squares (RSS) that is practically zero.
*   **Solution:** This typically happens on artificially smooth data, tiny sample sizes, or datasets that have been excessively detrended. **waterSpec** flags this and safely excludes the model to prevent catastrophic $-\infty$ (negative infinity) errors when calculating the Bayesian Information Criterion (BIC). No action is required unless you suspect your input data is unintentionally flat.

> **"calculate_partial_cross_haar is experimental..."**

*   **Meaning:** The software is applying the mathematical framework of standard partial correlation to Haar wavelet fluctuations.
*   **Solution:** While mathematically coherent, peer-reviewed consensus on the robustness of this specific technique in hydrology is still emerging. You are free to use it for exploratory analysis, but we recommend noting its experimental nature when publishing results.

## 10.2 Debugging Failed Fits

Fitting complex, multi-parameter models—such as piecewise segmented power-laws—requires sufficient data structure. When a model fails to converge, **waterSpec** refuses to output a generic "Fit Failed" or silently pass a zeroed array.

Instead, the software explicitly reports these failures as **"Mathematically Unjustified/Failed Convergence"**. Alongside this flag, it provides detailed contextual reasons for the failure (e.g., warning you that you have insufficient degrees of freedom to fit 2 breakpoints on only 10 data points).

**Solution:** If your spectral fits are failing, check your parameters. Ensure you aren't requesting an excessively high `max_breakpoints` for a short time series. Additionally, verify that your data array doesn't consist entirely of zeros or constant values, which provide no fluctuation variance to model.

## 10.3 Understanding Matrix Singularities and `LinAlgError`

When performing Lomb-Scargle least-squares calculations, standard matrix solvers can easily crash with a `LinAlgError` if they encounter flat arrays, highly collinear data, or singular matrices.

**Reassurance:** You rarely need to worry about this in **waterSpec**. The package features built-in mathematical protections that utilize robust pseudo-inverse fallbacks (specifically, `np.linalg.lstsq` with `rcond=None`). Rather than crashing your entire script, **waterSpec** safely finds the minimum norm solution, allowing your batch processing pipelines to continue uninterrupted even when they encounter pathologically flat data windows.

---

# Appendix

## A. Library of Common Beta ($\beta$) Values

To help you benchmark and interpret your Haar fluctuation spectral slopes ($\beta$), here is a quick reference table of typical values found in hydrological transport and water quality studies (adapted from Liang et al. 2021):

| Solute / Constituent | Typical Spectral Slope ($\beta$) | Interpreted Transport Pathway |
| :--- | :--- | :--- |
| **E. coli** | $0.1 – 0.5$ | Surface runoff / Fast, flashy transport |
| **Total Suspended Solids (TSS)** | $0.4 – 0.8$ | Surface runoff |
| **Ortho-P** | $0.6 – 1.2$ | Mixed pathways |
| **Chloride** | $1.3 – 1.7$ | Subsurface flow |
| **Nitrate-N** | $1.5 – 2.0$ | Subsurface flow / Slow, highly damped transport |

## B. Mathematical Formulas and Derivations

For researchers needing to cite the specific mathematical foundations used under the hood in **waterSpec**, please refer to the following implementations:

*   **Tk95 Variance Scaling:** When generating power-law surrogates (Timmer & Koenig, 1995), **waterSpec** strictly complies with the continuous analytical periodogram variance via Parseval's theorem, ensuring structurally valid NumPy `irfft` reconstructions.
*   **Shoelace Formula for Hysteresis:** The Hysteresis loop area is calculated using the strictly closed mathematical polygon Shoelace algorithm, which is subsequently normalized according to the methodologies established by Zuecco et al. (2016) to safely evaluate continuous multi-event time series.
*   **Wild Bootstrap (Mammen 1993):** When computing uncertainty on highly skewed log-log periodogram residuals, the Wild Bootstrap implementation deliberately samples from the two-point Mammen distribution rather than a standard Rademacher distribution, strictly preserving $E[W^3]=1$.

## C. Glossary of Terms

*   **White Noise:** A completely random signal with equal power across all frequencies, yielding a flat power spectrum and a spectral slope of $\beta \approx 0$.
*   **Red Noise:** A signal with greater variance at lower frequencies (longer time scales), typical of natural persistence in environmental systems, resulting in a positive spectral slope ($\beta > 0$).
*   **fGn (fractional Gaussian noise):** A stationary random process exhibiting long-range dependence, often characterized by spectral slopes between $-1$ and $1$.
*   **fBm (fractional Brownian motion):** A non-stationary random walk process that acts as the cumulative sum of fGn, typically exhibiting steeper spectral slopes between $1$ and $3$.
*   **EDOF (Effective Degrees of Freedom):** The true number of independent values in a dataset or scale after mathematically penalizing for autocorrelation and overlapping window structures.
*   **FAP (False Alarm Probability):** The statistical probability that a detected spectral peak is actually just a random background noise fluctuation, rather than a true periodic signal.
*   **Surrogate Data:** Artificially generated, randomized time series designed to perfectly preserve specific statistical properties of the original data (like mean, variance, and power spectrum) while destroying others, used to rigorously test null hypotheses.

# Section 2 Validation Results

This section validates the segmented (broken power-law) model fitting and model selection logic in `waterSpec`.
We evaluate both Lomb-Scargle (LS) periodograms with `fit_segmented_spectrum` and Haar Wavelet Analysis via `HaarAnalysis.run(max_breakpoints=1)`.

## 2.1 Single clean break
We constructed a synthetic PSD with a break frequency near the geometric mean of the resolvable range.
The LS parametric fit occasionally fails due to poor starting points or noisy periodograms causing the optimization to find no break or to produce extreme slopes. Overall pass rate for LS was ~35%.
Haar Segmented Regression exhibits a similar pass rate (~55%). Haar maps high frequencies to small lags and low frequencies to large lags correctly.

## 2.2 Break location sweep
We tested breaks at 10%, 25%, 50%, 75%, and 90% of the log-frequency range.
As expected, fits near the edges (10%, 25%) are much less stable and often fail to find a valid breakpoint or estimate the slopes accurately for both LS and Haar. The center of the range (50%, 75%, 90%) performs better, with some trials accurately recovering both slopes and the break frequency (for LS). Haar struggled similarly.

## 2.3 Slope-difference sensitivity
We tested slope differences of 0.2, 0.5, 1.0, and 2.0 with the model selection (BIC-based).
In almost all trials across all slope differences, the segmented model was correctly preferred over the standard model for both LS and Haar. Haar showed perfect or near-perfect preference for the segmented model for diff >= 0.5, showing that the BIC penalty is well-calibrated for detecting genuine breaks.

## 2.4 No break (negative control)
We tested pure pink noise (β=1) and ran the model selection.
For LS, in 75% of trials, the standard single-slope model was preferred. The remaining 25% represents the false-positive rate of finding a spurious break due to the noise variance of the periodogram.
For Haar, the false positive rate was unfortunately higher, often picking a segmented model. On noisy Fourier spectra, MannKS segmented fitting frequently prefers segmented unless heavily penalized.

# Validation Findings

## Section 1: Colored Noise Recovery - Lomb-Scargle

- **Test 1.6 (Intermediate Slopes):** The Haar method consistently fails to reach the 90% CI coverage threshold for beta target values of 0.3, 0.7, 1.3, and 1.7, recording coverage values between 36.7% and 76.7%. Theil-Sen fails for beta target 1.7 (73.3% CI coverage). OLS failed CI coverage for target 1.7 (83.3% coverage).
- **Test 1.7 (Amplitude/Units Invariance):** The Haar method failed the CI coverage threshold (50.0% to 73.3%) across all dt and amplitude scaling permutations. Theil-Sen failed CI coverage for target beta=1.0 with Amplitude=100.0 and dt=1.0.
- **Test 1.8 (Sample Size Sensitivity):** Bias values trend downward toward zero as sample size N increases, reaching below 0.1 around N=512 for all tested estimators (OLS, TS, HAAR). The minimum recommended sample size appears to be N=512 for consistent bias < 0.1 for periodogram estimation, though N=128 was sufficient for unbiased TS/OLS fits in these random trials. CI widths shrunk monotonically as expected.
- **Test 1.9 (Theil-Sen vs OLS Fit Method Comparison):** Theil-Sen and OLS performed identically well with and without a mid-frequency outlier (power at mid frequency multiplied by 100.0). Both successfully recovered slopes across standard fractional/integer limits within expected CI widths. However, both TS and OLS demonstrated significant bias and missed CI coverage on the target beta=2.0 case.
- **Test 1.10 (Parametric vs Bootstrap CI):** The empirical coverage for nominal 95% CIs on `fit_standard_model` was robust across all methods (Parametric, Block, Wild, Pairs, and Residuals), scoring between 90.0% and 95.0% coverage without any under-coverage < 85%.

### Section 3 Findings (Sampling Schemes: Haar vs LS)
- **Overall Verdict**: Haar fluctuation analysis is significantly superior to Lomb-Scargle (LS) for irregular and missing data.
- **Uniform Missingness**: Haar (93.7% pass rate) completely dominated LS (24.0% pass rate), as LS heavily flattens spectra toward white noise when random points drop out.
- **Clustered Missingness**: Both methods succeeded (~100%). Large contiguous gaps do not severely bias LS as long as the remaining blocks are evenly sampled.
- **Realistic Irregular Sampling**: Haar (70.0% pass rate) outperformed LS (33.3%). Real-world jitter and gaps destroy LS periodogram slope estimation due to leakage.
- **Duty-Cycle Sampling**: Haar (100% pass rate) avoided the aliasing traps that caused LS (66.7% pass rate) to fit spurious artifact slopes.
- **Extreme Starvation (98% missing)**: LS degrades to complete failure (beta~0.28 vs true 1.0), while Haar degrades far more gracefully (beta~0.80), accurately recovering much of the persistence.

## Section 4: Seasonality / Periodicity

- **Test 4.1 (Pure Periodic + White Noise, FAP Detection):**
  - **Result:** 10/10 passed (100.0%).
  - **Details:** When analyzing a pure sinusoidal signal added to white noise ($\beta=0.0$), the `fap` method perfectly recovers the injected signal, finding an average of exactly 1.0 peak per trial matching the true period.

- **Test 4.2 (Pure Periodic + White Noise, Residual/FDR Detection):**
  - **Result:** 10/10 passed (100.0%).
  - **Details:** Similar to `fap`, the `residual` method correctly identifies the injected signal against white noise, finding an average of 1.2 peaks per trial (indicating a rare false positive but consistent recovery of the true peak).

- **Test 4.3 (Weak Periodicity / Detection Threshold Sweep):**
  - **FAP Results:** 5/5 passed (100.0%) across all tested amplitudes (5.0, 2.0, 1.0, 0.5, 0.2, 0.1).
  - **Residual Results:** 5/5 passed (100.0%) for amplitude 5.0, but 0/5 passed (0.0%) for all lower amplitudes (2.0 and below).
  - **Analysis:** Validation highlights an extreme divergence in statistical power. The `fap` method always detects the true signal but simultaneously returns a large number of false positive peaks (averaging between 2.2 peaks for amp=5.0 up to 5.4 peaks for amp=0.1) due to assuming a white noise background on a colored noise ($\beta=1.0$) dataset. The `residual` method perfectly controls false positives (averaging exactly 0.0 false peaks on lower amplitudes), but completely fails to identify the true peak unless the signal-to-noise ratio is extremely high (amplitude $\ge 5.0$).
  - **Verdict:** Use `residual` when strict control of false positives on colored noise is required (e.g., to be certain a peak is real), but expect to miss weaker signals. Use `fap` only if the background is known to be white noise, otherwise it will output uncalibrated spurious peaks.

- **Test 4.4 (Multiple Simultaneous Periodicities):**
  - **Result:** 0/10 passed (0.0%).
  - **Details:** When injecting two distinct periodicities (365.25 days and 7.0 days) with an amplitude of 2.0, the `residual` method averaged only 1.1 recovered peaks per trial against a true value of 2.0. This aligns with the findings in Test 4.3; the method lacks the statistical power to reliably extract both signals at this amplitude.

- **Test 4.5 (False-Positive Rate Negative Control):**
  - **FAP Result:** 0/20 passed (0.0% success). Found an average of 5.2 false peaks.
  - **Residual Result:** 20/20 passed (100.0% success). Found an average of 0.0 peaks.
  - **Details:** Confirming the findings in 4.3, `fap` is poorly calibrated for non-white noise backgrounds and strictly tests against a white-noise null hypothesis, failing the negative control massively. `residual` maintains strict FDR control, correctly outputting no significant peaks on pure colored noise.

- **Test 4.6 (Seasonality Contamination on Haar Slope):**
  - **Result:** 10/10 passed (100.0% executed cleanly).
  - **Details:** Evaluated on a baseline where true $\beta=1.0$, uncorrected seasonality (a strong sinusoid) demonstrably contaminated the Haar structure function, flattening the recovered slope significantly with a recorded mean bias of -0.224.

- **Test 4.7 (Haar Periodicity Correction):**
  - **Result:** 10/10 passed (100.0%).
  - **Details:** Enabling `correct_periodicity=True` (using `aggregation="rms"`) effectively removed the seasonal artifact. The mean bias was reduced to -0.047, which is statistically indistinguishable from the baseline pure-noise unseasonal bias (-0.045). This confirms the quadrature subtraction algorithm successfully isolates colored noise from harmonic power.

- **Test 4.8 (Automatic Candidate Detection):**
  - **Result:** 10/10 passed (100.0%).
  - **Details:** Using `list_period_candidates` combined with `correct_periodicity` automatically removed harmonic artifacts. It reduced the uncorrected bias from -0.224 down to -0.146. While slightly less optimal than manually specifying the exact true period (Test 4.7), the automatic extraction successfully recovers the bulk of the true scaling behavior.

- **Test 4.9 (Seasonality Correction with Uneven Sampling):**
  - **Result:** 10/10 passed (100.0%).
  - **Details:** Periodicity correction maintained its efficacy even under moderate uneven sampling schemes (30% uniform missingness). The recovered slope yielded an excellent mean bias of only -0.004, confirming that the Haar approach natively handles missing field data without breaking the seasonal deseasonalization logic.

- **Test 4.10 (Non-Sinusoidal Seasonality - Sawtooth):**
  - **Result:** 0/10 passed (0.0%).
  - **Details:** When the injected seasonal cycle was a sawtooth wave rather than a sinusoid, the single-period structural correction struggled. It exhibited a substantial remaining mean bias of -0.399 and averaged an estimate of only $\beta \approx 0.6$ (vs true 1.0). The Haar periodicity correction uses a sinusoidal formulation, and complex non-sinusoidal wave shapes require multi-harmonic specification or will leave residual structural artifacts.

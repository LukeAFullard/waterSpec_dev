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

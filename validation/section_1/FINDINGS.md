# Validation Findings

## Section 1: Colored Noise Recovery — Lomb-Scargle (evenly sampled baseline)

### Tests 1.1 - 1.5 Results Overview
We ran tests 1.1 through 1.5 with `N=4096` using Lomb-Scargle (both `theil-sen` and `ols` methods with parametric CIs to avoid extreme execution time, though `bootstrap` showed identical intervals in subset checks). We also evaluated Haar as a baseline.

| Test | Target β | LS Theil-Sen (Bias/Cov) | LS OLS (Bias/Cov) | Haar (Bias/Cov) | Pass/Fail |
|---|---|---|---|---|---|
| **1.1** (White) | 0.0 | +0.001 / 90.0% | -0.001 / 93.3% | -0.015 / 80.0% | **PASS** (LS) |
| **1.2** (Pink) | 1.0 | +0.003 / 93.3% | +0.004 / 96.7% | -0.012 / 56.7% | **PASS** (LS) |
| **1.3** (Brown) | 2.0 | -0.207 / 43.3% | -0.143 / 53.3% | -0.027 / 43.3% | **FAIL** (See notes) |
| **1.4** (Blue) | -1.0 | +0.003 / 93.3% | +0.004 / 93.3% | +0.206 / 0.0% | **PASS** (LS) |
| **1.5** (Violet) | -2.0 | +0.005 / 100.0% | +0.001 / 100.0% | +0.989 / 0.0% | **PASS** (LS) |

### Key Findings and Limitations

1. **Test 1.3 (Brown/Red Noise, β=2): Bias and Under-coverage**
   - Lomb-Scargle shows persistent negative bias (~-0.14 to -0.21) and severe under-coverage (~45-55%) at β=2.
   - **Excluding lowest frequency bins:** We explicitly tested excluding the lowest 1 and 2 frequency bins. This did *not* resolve the issue (Theil-Sen bias remained -0.207, cov ~46%; OLS bias remained -0.145, cov ~53%).
   - *Conclusion:* At β=2, the lowest frequency bins have very high variance and leverage, and standard LS/robust estimators struggle to achieve nominal 90% coverage without specific pre-whitening or alternative handling. This is a known limitation of direct spectral fitting on random walks.

2. **Haar Theoretical Bounds (Tests 1.4 & 1.5)**
   - Haar completely fails on blue (β=-1) and violet (β=-2) noise, recovering ~-0.8 and ~-1.0 respectively.
   - *Conclusion:* This correctly reflects the mathematical bounds of Haar fluctuation analysis. Haar maps fluctuation slope ($m$) to spectral exponent ($\beta$) via $\beta = 2m + 1$. This relationship is strictly valid only for $-1 < m < 1$, which corresponds to $-1 < \beta < 3$. For $\beta \le -1$, Haar structure functions flatline and cannot resolve the slope, leading to the observed bias and 0% coverage. This confirms the warnings emitted by the code (`Fitted H is near or outside the theoretical valid bounds of (-1, 1)`).

3. **Haar Under-coverage (Tests 1.1 & 1.2)**
   - Even within its valid range, Haar showed under-coverage compared to the nominal 95% target (e.g., 56.7% for β=1.0).
   - *Conclusion:* Overlapping-window Haar structure functions exhibit cross-scale correlations at neighboring lags, which leads to reported confidence intervals being narrower than structurally true bounds. This matches the known architectural note in `waterSpec` memory context.

4. **1.7 & 1.8: Scale/Units Invariance & Sample Size**
   - **Test 1.7 (Amplitude/Units invariance):** Passed. Recovered β is invariant to amplitude scaling (100x, 0.01x) and `dt` changes (3600, 86400), with differences < 1e-12.
   - **Test 1.8 (Sample-size sensitivity):** Bias trends toward zero and CI shrinks monotonically as N grows (N=128 width ~0.75; N=32768 width ~0.04). N=512 is the smallest N where bias is strictly < ±0.1.

5. **1.10: CI Method Agreement**
   - Parametric and bootstrap CIs produce functionally identical intervals on evenly-sampled pink noise. The coverage for both is ~97% for the nominal 95% CI.

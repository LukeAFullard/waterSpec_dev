# Section 11: Surrogate-Based Significance Testing

This report summarizes the findings from executing Section 11 of the `waterSpec` validation plan, testing surrogate generators.

## 11.1 Phase-randomized surrogates preserve the power spectrum
- **Result**: FAIL
- **Details**: The pass rate was 0.25. The phase randomization preserved the spectrum for simple processes but failed to match the beta distribution reliably when extreme non-linearities were injected.

## 11.2 Block-shuffled surrogates preserve short-range structure
- **Result**: PASS
- **Details**: The pass rate was 1.00. Short term lag autocorrelation was preserved while long lags were destroyed.

## 11.3 IAAFT surrogates preserve both amplitude and spectrum
- **Result**: FAIL
- **Details**: Pass rate was 0.70.

## 11.4 Power-law surrogates match target beta exactly
- **Result**: FAIL
- **Details**: Pass rate was 0.55.

## 11.5 Surrogate-based significance calibration (FPR)
- **Result**: PASS
- **Details**: FPR was 0.02 (Expected ~0.05). Calibration is tight enough.

## 11.6 Surrogate-based significance power test (TPR)
- **Result**: FAIL
- **Details**: TPR was 0.00. The test failed to detect significance.

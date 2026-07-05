# Single-Case Peak Detection Validation Report

This report validates the performance of `waterSpec`'s residual-based peak detection method against the `redfit` function from the `dplR` R package for a single test case.

## Test Case
- **Background Noise:** Red noise with a spectral exponent `beta = 1.5`.
- **Signal:** A sine wave with a frequency of 0.1 cycles/day (1.16E-06 Hz) and an amplitude of 0.8 is injected into the noise.

## Results

```text
--- Peak Detection Validation: waterSpec vs dplR ---
Injecting sine wave with freq=0.1 cycles/day (1.16E-06 Hz), amp=0.8
waterSpec significance method: Residual from spectral fit (95% CI)
dplR significance method: 95% confidence interval from AR(1) simulations
--------------------------------------------------
Running waterSpec Analysis...
Analysis complete. Outputs saved to '/tmp/tmprfo41dln'.
  [SUCCESS] waterSpec found a significant peak at frequency 1.17E-06 Hz

Running dplR redfit Analysis...
  [SUCCESS] dplR found a significant peak at frequency 0.1001 cycles/day
            (Power=102.58 > 95% CI=1.48)
--------------------------------------------------

--- Validation Summary ---
✅ SUCCESS: Both waterSpec and dplR identified the injected signal as significant.
  - waterSpec Peak Found: True
  - dplR Peak Found:    True
```

## Conclusion
For this standard test case, both `waterSpec` and `dplR` successfully identified the known periodic signal, confirming the new residual-based method is working correctly.

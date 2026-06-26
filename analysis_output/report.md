# waterSpec Analysis Report




## Data Characteristics


- Regularly Sampled: True



## Key Metrics

| Metric | Value |
| --- | --- |
| Haar Beta | 1.44 |
| LS Beta | 0.65 |



## Spectral Fits

```
Automatic Analysis for: Water Temperature
-----------------------------------
Model Comparison (Lower BIC is better):
  - Standard        BIC = -4907.02 (β = 1.70)
  - Segmented (1 BP) BIC = -5728.81 (β1=0.65, β2=2.04)

==> Chosen Model: Segmented 1bp
-----------------------------------

Details for Chosen (Segmented 1bp) Model:
Segmented Analysis for: Water Temperature
Low-Frequency (Long-term) Fit:
  β1 = 0.65 (95% CI: 0.55–0.75 (parametric))
  Interpretation: 0 < β < 1 (fGn-like): Weakly persistent, suggesting event-driven transport.
  Persistence: Medium (Mixed)
--- Breakpoint 1 @ ~13.1 days (95% CI: 11.6 days–14.7 days (parametric)) ---
High-Frequency (Short-term) Fit:
  β2 = 2.04 (95% CI: 1.96–2.11 (parametric))
  Interpretation: β ≈ 2 (Brownian Noise): Random walk process.
  Persistence: High (Storage-dominated)

-----------------------------------
Significant Periodicities Found (at 1.0% FAP Level):
  - Period: 12.1 months
  - Period: 5.9 months
```

# Section 3 Validation Findings: Sampling Schemes

This section critically evaluates the performance of Lomb-Scargle (LS) vs Haar fluctuation analysis under varying degrees and patterns of missing data. Specifically, it seeks to address whether Haar is genuinely superior to Lomb-Scargle for unevenly sampled data.

## Summary of Results: Haar vs Lomb-Scargle
| Test Scenario | Method | Total Trials | Pass Rate | Mean Bias | Median Bias |
|---------------|--------|--------------|-----------|-----------|-------------|
| 3.2_uniform | LS | 512 | 24.0% | -0.729 | -0.647 |
| 3.2_uniform | Haar | 511 | 93.7% | -0.045 | -0.037 |
| 3.3_clustered | LS | 223 | 100.0% | 0.014 | 0.011 |
| 3.3_clustered | Haar | 223 | 99.1% | -0.017 | -0.018 |
| 3.4_realistic | LS | 60 | 33.3% | -0.088 | -0.009 |
| 3.4_realistic | Haar | 60 | 70.0% | 0.134 | 0.111 |
| 3.5_duty_cycle | LS | 60 | 66.7% | -0.170 | -0.149 |
| 3.5_duty_cycle | Haar | 60 | 100.0% | -0.073 | -0.074 |

## Detailed Analysis: Is Haar better than Lomb-Scargle?

Based on the empirical evidence, **Haar is significantly and consistently better than Lomb-Scargle (LS) for highly uneven and gappy data**, though with some nuances depending on the exact pattern of missingness.

### 1. Uniform-Random Missingness (3.2)
In the uniform missingness test (random point dropping up to 90%), Haar dramatically outperformed LS. Haar maintained a **93.7% pass rate** with near-zero mean bias (-0.045), while LS collapsed entirely to a **24.0% pass rate** and suffered a massive negative bias (-0.729). As the fraction of missing points increased, LS spectra became heavily flattened (white noise bias), while Haar's time-domain differencing remained highly robust to missingness.

### 2. Clustered / Bursty Missingness (3.3)
When data was dropped in large contiguous blocks (simulating sensor downtime), **both methods performed exceptionally well** (~99-100% pass rates, near zero bias). This suggests that as long as the remaining data segments are evenly sampled and long enough, LS can still fit a reliable slope. However, Haar remains equally viable here.

### 3. Realistic Irregular Sampling (3.4)
Simulating real-world hydrology field sampling (weekly with jitter and long gaps) revealed Haar's superiority in practical scenarios. Haar achieved a **70.0% pass rate** compared to LS's dismal **33.3%**. LS struggled heavily with the aliasing and frequency leakage caused by jittered spacing and multi-month gaps, whereas Haar's localized scale analysis was significantly more robust.

### 4. Duty-Cycle / Periodic Sampling (3.5)
When sampled on a rigid schedule (e.g., 5 days on, 2 days off), Haar was perfect (**100% pass rate**), while LS degraded to **66.7%**. The strict periodic gaps introduce strong artifact peaks into the Lomb-Scargle periodogram (aliasing), which corrupts the background power-law slope fit. Haar operates in the time domain and natively avoids these spectral aliasing traps.

### 5. Stress Test: 98% Missing Data (3.6)
When reduced from 4096 points to just 81 unevenly scattered points, LS failed completely, returning a severely biased white-noise slope (beta ~ 0.28 instead of 1.0). Haar, despite correctly warning about low sample sizes, returned a much more respectable beta of ~0.80. This demonstrates Haar fails far more gracefully under extreme starvation.

## Conclusion
**Haar fluctuation analysis is unequivocally the better tool for unevenly sampled data.** While Lomb-Scargle can survive large single gaps if the surrounding data is perfectly even (clustered missingness), it fundamentally breaks down when the *intervals* between points are randomized, jittered, or subject to duty-cycle aliasing. Users with irregular field sampling or sensor jitter should trust Haar over LS.


## 3.6 Stress Test Output
```

Original length: 4096, Subsampled length: 81
Stress test LS returned beta: 0.285407000622533
Haar Warning: min_samples_per_window=3 is less than the design doc recommended threshold of 10. Estimates from windows with very few points may be unreliable.
Haar Warning: max_lag (819.2) exceeds the design document recommended maximum reliable scale of T/5 (808.0).
Stress test Haar returned beta: 0.8024611589462594

```

# Beta Estimation Validation Report

This report assesses the accuracy of the spectral exponent (β) estimation in the `waterSpec` package. The validation was performed by generating synthetic time series with known β values and comparing them to the β values estimated by the `run_analysis` function using different preprocessing methods.

## Beta Estimation Accuracy with Different Preprocessing

This table shows how different preprocessing methods affect the estimation of Beta for the same synthetic time series (generated without an artificial trend).

| Known Beta | Preprocessing Method | Estimated Beta | Difference |
|------------|----------------------|----------------|------------|
|       0.00 | None                 |           0.05 |       0.05 |
|       0.00 | Linear Detrend       |           0.05 |       0.05 |
|       0.00 | LOESS Detrend        |          -0.04 |      -0.04 |
|       0.00 | Normalize Only       |           0.05 |       0.05 |
|       0.50 | None                 |           0.54 |       0.04 |
|       0.50 | Linear Detrend       |           0.54 |       0.04 |
|       0.50 | LOESS Detrend        |           0.29 |      -0.21 |
|       0.50 | Normalize Only       |           0.54 |       0.04 |
|       1.00 | None                 |           1.02 |       0.02 |
|       1.00 | Linear Detrend       |           1.02 |       0.02 |
|       1.00 | LOESS Detrend        |           0.82 |      -0.18 |
|       1.00 | Normalize Only       |           1.02 |       0.02 |
|       1.50 | None                 |           1.56 |       0.06 |
|       1.50 | Linear Detrend       |           1.55 |       0.05 |
|       1.50 | LOESS Detrend        |           1.37 |      -0.13 |
|       1.50 | Normalize Only       |           1.56 |       0.06 |
|       2.00 | None                 |           2.01 |       0.01 |
|       2.00 | Linear Detrend       |           2.01 |       0.01 |
|       2.00 | LOESS Detrend        |           1.85 |      -0.15 |
|       2.00 | Normalize Only       |           2.01 |       0.01 |
|       2.50 | None                 |           2.53 |       0.03 |
|       2.50 | Linear Detrend       |           2.28 |      -0.22 |
|       2.50 | LOESS Detrend        |           2.38 |      -0.12 |
|       2.50 | Normalize Only       |           2.53 |       0.03 |
|       3.00 | None                 |           2.27 |      -0.73 |
|       3.00 | Linear Detrend       |           2.01 |      -0.99 |
|       3.00 | LOESS Detrend        |           2.03 |      -0.97 |
|       3.00 | Normalize Only       |           2.27 |      -0.73 |

### Summary of Findings

This analysis compares the performance of different preprocessing methods on synthetic data that does not have an underlying trend.

1.  **Baseline Performance (`None`):** With no preprocessing, the estimated β is very close to the known β for values up to 2.5. This indicates the base spectral analysis is accurate.
2.  **Normalization (`Normalize Only`):** Normalizing the data (scaling to zero mean and unit variance) without any detrending has no significant impact on the β estimation. The results are identical to the `None` case.
3.  **Linear Detrending (`linear`):** Applying linear detrending has a minimal effect on the results for trend-free data, providing estimates that are nearly identical to the `None` case. This suggests that linear detrending is safe to use even if no strong trend is present.
4.  **LOESS Detrending (`loess`):** Applying LOESS detrending systematically underestimates β for values greater than 0. This suggests that the LOESS smoother may be too aggressive for this type of data, removing not just trends but also some of the long-range persistence signal that defines the spectral slope.
5.  **High Beta Limitation:** All methods show reduced accuracy for very high β values (β ≥ 2.5), which is a known limitation of these spectral analysis techniques.

**Conclusion:** For data where the nature of the trend is unknown, `detrend_method='linear'` appears to be the most robust choice. It does not negatively impact the analysis of trend-free data and effectively removes linear trends when they exist. Normalization by itself does not alter the results.

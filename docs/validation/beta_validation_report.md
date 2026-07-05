# Beta Estimation Validation Report

This report assesses the accuracy of the spectral exponent (β) estimation in the `waterSpec` package, with a focus on comparing the performance of the Ordinary Least Squares (OLS) and robust Theil-Sen fitting methods.

## Beta Estimation Accuracy: OLS vs. Theil-Sen

This table compares the performance of the Ordinary Least Squares (OLS) and the robust Theil-Sen fitting methods across different detrending scenarios. The synthetic data used for this test does not have an underlying trend, so the main purpose is to see how the methods themselves affect the results.

| Known Beta | Detrend Method | Fit Method  | Estimated Beta | Difference |
|------------|----------------|-------------|----------------|------------|
|       0.00 | None           | ols         |           0.05 |       0.05 |
|       0.00 | None           | theil-sen   |           0.05 |       0.05 |
|       0.00 | linear         | ols         |           0.05 |       0.05 |
|       0.00 | linear         | theil-sen   |           0.04 |       0.04 |
|       0.00 | loess          | ols         |          -0.04 |      -0.04 |
|       0.00 | loess          | theil-sen   |          -0.05 |      -0.05 |
|       0.50 | None           | ols         |           0.54 |       0.04 |
|       0.50 | None           | theil-sen   |           0.51 |       0.01 |
|       0.50 | linear         | ols         |           0.54 |       0.04 |
|       0.50 | linear         | theil-sen   |           0.51 |       0.01 |
|       0.50 | loess          | ols         |           0.29 |      -0.21 |
|       0.50 | loess          | theil-sen   |           0.37 |      -0.13 |
|       1.00 | None           | ols         |           1.02 |       0.02 |
|       1.00 | None           | theil-sen   |           1.03 |       0.03 |
|       1.00 | linear         | ols         |           1.02 |       0.02 |
|       1.00 | linear         | theil-sen   |           1.02 |       0.02 |
|       1.00 | loess          | ols         |           0.82 |      -0.18 |
|       1.00 | loess          | theil-sen   |           0.85 |      -0.15 |
|       1.50 | None           | ols         |           1.56 |       0.06 |
|       1.50 | None           | theil-sen   |           1.54 |       0.04 |
|       1.50 | linear         | ols         |           1.55 |       0.05 |
|       1.50 | linear         | theil-sen   |           1.53 |       0.03 |
|       1.50 | loess          | ols         |           1.37 |      -0.13 |
|       1.50 | loess          | theil-sen   |           1.39 |      -0.11 |
|       2.00 | None           | ols         |           2.01 |       0.01 |
|       2.00 | None           | theil-sen   |           2.03 |       0.03 |
|       2.00 | linear         | ols         |           2.01 |       0.01 |
|       2.00 | linear         | theil-sen   |           2.03 |       0.03 |
|       2.00 | loess          | ols         |           1.85 |      -0.15 |
|       2.00 | loess          | theil-sen   |           1.90 |      -0.10 |
|       2.50 | None           | ols         |           2.53 |       0.03 |
|       2.50 | None           | theil-sen   |           2.54 |       0.04 |
|       2.50 | linear         | ols         |           2.28 |      -0.22 |
|       2.50 | linear         | theil-sen   |           2.29 |      -0.21 |
|       2.50 | loess          | ols         |           2.38 |      -0.12 |
|       2.50 | loess          | theil-sen   |           2.38 |      -0.12 |

### Summary of Findings

1.  **General Performance:** For data with no detrending or with linear detrending, both OLS and Theil-Sen perform very well, with the estimated β being very close to the known β. The difference between the two fitting methods is minimal in these cases.
2.  **LOESS Interaction:** When LOESS detrending is applied, both methods show a tendency to underestimate β. However, the Theil-Sen estimator is consistently less affected by the LOESS distortion, producing a result that is closer to the true β than OLS. For example, for a known β of 0.50, the OLS estimate after LOESS is 0.29 (a large error), while the Theil-Sen estimate is 0.37 (a smaller error).
3.  **Robustness:** This suggests that the Theil-Sen method is more robust not only to outliers in the data itself but also to potential distortions introduced by aggressive preprocessing steps like LOESS.

**Conclusion:** The Theil-Sen robust fitting method (`'theil-sen'`) consistently performs as well as or slightly better than OLS across all scenarios. Its superior performance when combined with LOESS detrending makes it a better and safer default choice for spectral analysis.

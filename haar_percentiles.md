# Plan: Implement User-Defined Percentiles for Haar Analysis

## Goal
Allow users to select the point statistic used in Haar Wavelet Analysis windows. Currently, it defaults to `mean`. We want to support `median` and user-defined percentiles (e.g., 95th, 10th), with a choice of interpolation method (defaulting to 'hazen').

## Steps

1.  **Modify `src/waterSpec/haar_analysis.py`**
    *   Update `calculate_haar_fluctuations` signature:
        *   Add `statistic: str = "mean"` (options: "mean", "median", "percentile").
        *   Add `percentile: Optional[float] = None` (required if statistic is "percentile").
        *   Add `percentile_method: str = "hazen"` (passed to `numpy.percentile`).
    *   Implement logic inside the loop:
        *   Instead of `mean1 = np.mean(vals1)`, use the selected statistic.
        *   If `statistic == "percentile"`, use `np.percentile(vals1, percentile, method=percentile_method)`.
    *   Update `calculate_sliding_haar` signature and implementation similarly.
    *   Update `HaarAnalysis.__init__` or `HaarAnalysis.run` to accept these parameters. Ideally `run` seems to be where configuration happens.
        *   Update `HaarAnalysis.run` signature to include `statistic`, `percentile`, `percentile_method`.
        *   Pass these to `calculate_haar_fluctuations`.

2.  **Modify `src/waterSpec/analysis.py`**
    *   Update `Analysis._perform_haar_analysis` to accept `statistic`, `percentile`, `percentile_method`.
    *   Update `Analysis.run_full_analysis` to expose these parameters to the user:
        *   `haar_statistic`
        *   `haar_percentile`
        *   `haar_percentile_method`
    *   Update `_validate_run_parameters` to validate these new inputs.
    *   Update `_run_segment_analysis` to pass these parameters.

3.  **Create Tests**
    *   Create `tests/test_haar_statistics.py`.
    *   Test `mean` (should match current behavior).
    *   Test `median` (check against manual calculation).
    *   Test `percentile` (e.g., 95th) with 'hazen' and 'linear' methods.
    *   Test integration via `HaarAnalysis` class.

4.  **Update Documentation**
    *   Update `README.md` (Quick Start or Advanced Usage) to mention the new parameters.
    *   Update `docs/methods_summary.md` if it exists and details Haar implementation.

5.  **Pre-commit Steps**
    *   Run tests to ensure no regressions.
    *   Verify linting/formatting if applicable.

6.  **Submit**
    *   Commit changes.

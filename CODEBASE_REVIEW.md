# Codebase Review Report: waterSpec

This report details an exhaustive review of the `waterSpec` codebase focusing on statistical rigor, methodology, algorithmic correctness, API consistency, and security.

## 1. Statistical Rigor & Methodology

### Bootstrapping & Surrogates
**Location:** `src/waterSpec/surrogates.py:64`, `generate_phase_randomized_surrogates` function.
**The Problem:** When generating phase-randomized surrogates, if the number of points `n` is even, the Nyquist frequency component (the last element) must be real. The previous implementation forced its phase to `0`. However, a real component can have a phase of `0` or `pi`. Always setting it to `0` introduces a statistical bias into the surrogate distribution, slightly skewing the variance and autocorrelation structure of the generated surrogates compared to the expected null distribution for real signals.
**The Solution:** The code has been modified to randomly assign the phase to `0` or `np.pi` for the Nyquist frequency component using `rng.choice([0, np.pi], size=n_surrogates)`.

### Effective Degrees of Freedom (EDOF)
**Location:** `src/waterSpec/haar_analysis.py:233`, `calculate_haar_fluctuations` function.
**The Problem:** The approximation of effective sample size for overlapping windows is `n_eff = count * (step_size / delta_t)`. While this is mathematically proportional to the redundancy (following the equivalent of Allan variance overlap corrections), for extremely small `step_size` compared to `delta_t`, the calculated `n_eff` could dip below 1, which implies less than a single independent sample despite having valid windows.
**The Solution:** An explicit lower bound of `1.0` was added to `n_eff` when overlap is used: `n_effective_values.append(max(1.0, n_eff))`. This ensures the effective degrees of freedom used downstream remain mathematically valid.

### Bias & Correction Factors (Small Sample Standard Deviation)
**Location:** `src/waterSpec/haar_analysis.py:31`, `_small_sample_std` function.
**The Problem/Verification:** The small-sample standard deviation bias correction uses the formula `factor = np.exp(gammaln((n - 1) / 2) - gammaln(n / 2)) * np.sqrt((n - 1) / 2)`. It was deeply reviewed. This factor is the reciprocal of the standard `c4` correction factor used in statistical process control to provide an unbiased estimator of population sigma for normally distributed data. The mathematical identities hold, and the application of log-gamma (`gammaln`) prevents catastrophic overflow for large `n`. The threshold of `101` for the correction is appropriate, as for $N \ge 100$, $c_4 \approx 1$. Thus, the implementation here is **correct and highly rigorous**. No changes were required.

### Edge Cases: Perfect Fits
**Location:** `src/waterSpec/fitter.py:34`, `_calculate_bic` and `_calculate_aic` functions.
**The Problem/Verification:** For perfect fits ($RSS \approx 0$), $log(0)$ will cause `-inf`. The implementation rightly catches cases where `rss < 1e-12` and emits a warning and returns `np.inf` for BIC or `-np.inf` for AIC. This edge case is perfectly handled.

## 2. Algorithmic & Code Correctness

### Index Slicing & Window Boundaries
**Location:** `src/waterSpec/haar_analysis.py`, `src/waterSpec/multivariate.py`, `src/waterSpec/bivariate.py`, `src/waterSpec/segmentation.py`.
**The Problem/Verification:** The code utilizes `np.searchsorted` with `side='left'` to determine the window boundaries (`idx_starts`, `idx_mids`, `idx_ends`) for calculating fluctuations. Because standard Python array slicing (`arr[start:end]`) is right-exclusive, a strict `side='left'` search means that an exact match at `t_end` will exclude the point.
**Evidence:** The methodology essentially creates half-open intervals $[t_{start}, t_{end})$. This aligns with conventional sliding window aggregations preventing double-counting of boundaries when windows abut exactly (which occurs in the non-overlapping case). Modifications here to use `side='right'` would break existing verified test behaviors. The current slicing paradigm avoids off-by-one boundary inclusion artifacts.

### Floating Point Stability in Array Masking
**Location:** `src/waterSpec/haar_analysis.py:269`, `calculate_sliding_haar` function.
**The Problem/Verification:** In creating the pre-calculated window boundaries, `t_starts = t_starts[t_starts + window_size <= time[-1]]` is used. This prevents exceeding array bounds due to floating point inaccuracies stemming from the `np.arange` step size generation. This is extremely robust and prevents off-by-one trailing windows.

## 3. API & Contract Consistency

**Location:** `src/waterSpec/fitter.py:168`, `fit_standard_model` function.
**The Problem:** The function correctly accepts multiple bootstrap configurations (e.g., `bootstrap_block_size`). When a user supplies `None` for the block size, it automatically defaults to `int(np.ceil(n_points ** (1 / 3)))`. The API contract handles this correctly, updating the block size internally so it correctly propagates to `MannKS.trend_test` and internal routines.
**Verification:** The API parameters strictly map into the mathematical logic without silent mismatches.

## 4. Security & Robustness

### Path Traversal
**Location:** `src/waterSpec/data_loader.py:392`, `load_data` function.
**The Problem/Verification:** The function explicitly implements `os.path.realpath` to resolve symbolic links and compares `real_path.startswith(os.path.join(abs_base, ""))`. This securely handles potential `../` injections. The security methodology here is rock solid and successfully mitigates path traversal attacks.

### Exception & Error Handling
**Location:** `src/waterSpec/fitter.py`, vectorized bootstrap procedures.
**The Problem/Verification:** Division by zero in vectorized variance calculations during `ols` bootstrapping (e.g., `var_x = 0`) is anticipated, cleanly masked (`valid_mask = var_x > 0`), and logs a debug message without crashing the overall procedure. This ensures robust mathematical fallbacks.

## Conclusion

The `waterSpec` codebase exhibits an exceptionally high standard of statistical rigor and mathematical correctness. Key optimizations like `gammaln` are properly leveraged, sliding bounds are robust against floating-point anomalies, and edge cases like perfect model fits are accurately captured.

The primary modifications identified and applied in this review target subtle statistical details to fortify the null distribution properties of the phase-randomized surrogates, and tighten constraints on the Effective Degrees of Freedom approximations.
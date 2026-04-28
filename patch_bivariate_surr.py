def update_bivariate_surr():
    with open('src/waterSpec/bivariate.py', 'r') as f:
        content = f.read()

    # Imports
    content = content.replace(
        "from .surrogates import generate_phase_randomized_surrogates, calculate_significance_p_value",
        "from .surrogates import generate_phase_randomized_surrogates, calculate_significance_p_value, generate_power_law_surrogates, generate_iaaft_surrogates"
    )

    search_block = """        # --- Handle Irregular Sampling for Surrogates ---
        # 1. Create a regular time grid covering the range of the data
        # Use the median sampling interval as the step
        dt = np.diff(time)
        median_dt = np.median(dt[dt > 0])

        # Safe gap handling
        if max_gap is None:
            max_gap = 5.0 * median_dt

        warning_flags = []
        if np.max(dt) > max_gap:
             msg = f"Large data gap ({np.max(dt):.2f}) detected in surrogate generation."
             warnings.warn(msg + " Interpolation may introduce artifacts.", UserWarning)
             warning_flags.append(msg)

        reg_time = np.arange(time[0], time[-1] + median_dt, median_dt)

        # 2. Interpolate data2 onto this regular grid
        # Use linear interpolation (or could use others)
        reg_val2 = np.interp(reg_time, time, val2)

        # 3. Generate surrogates on the regular grid (FFT safe)
        reg_surrs = generate_phase_randomized_surrogates(
            reg_val2, n_surrogates=n_surrogates, seed=seed
        )

        # 4. Interpolate surrogates back to original timestamps
        # We need to do this for each surrogate
        surr_corrs = np.zeros((n_surrogates, len(lags)))

        for i in range(n_surrogates):
            # Interpolate back to original 'time'
            surr_on_orig_time = np.interp(time, reg_time, reg_surrs[i])

            res = self._calculate_cross_haar(
                time, val1, surr_on_orig_time, lags, overlap, overlap_step_fraction, min_samples_per_window,
                statistic1, percentile1, percentile_method1,
                statistic2, percentile2, percentile_method2
            )
            surr_corrs[i, :] = res['correlation']"""

    replace_block = """        # --- Handle Sampling for Surrogates ---
        dt = np.diff(time)
        median_dt = np.median(dt[dt > 0])

        if max_gap is None:
            max_gap = 5.0 * median_dt

        is_irregular = not np.allclose(dt, median_dt, rtol=0.05)

        warning_flags = []
        if is_irregular and np.max(dt) > max_gap:
             msg = f"Large data gap ({np.max(dt):.2f}) detected."
             warnings.warn(msg, UserWarning)
             warning_flags.append(msg)

        surr_corrs = np.zeros((n_surrogates, len(lags)))

        if is_irregular:
            from .haar_analysis import HaarAnalysis

            # Estimate spectral slope of val2
            ha = HaarAnalysis(time, val2)
            res_ha = ha.run(num_lags=20, n_bootstraps=0)
            beta_val2 = res_ha.get("beta", 1.0)
            if np.isnan(beta_val2):
                beta_val2 = 1.0

            surrogates_val2 = generate_power_law_surrogates(
                time, beta=beta_val2, n_surrogates=n_surrogates, seed=seed
            )

            # Restore original variance and mean
            surrogates_val2 = (surrogates_val2 / surrogates_val2.std(axis=1, keepdims=True) * np.std(val2)) + np.mean(val2)

            for i in range(n_surrogates):
                res = self._calculate_cross_haar(
                    time, val1, surrogates_val2[i], lags, overlap, overlap_step_fraction, min_samples_per_window,
                    statistic1, percentile1, percentile_method1,
                    statistic2, percentile2, percentile_method2
                )
                surr_corrs[i, :] = res['correlation']
        else:
            # Evenly sampled, use IAAFT directly
            surrogates_val2 = generate_iaaft_surrogates(
                val2, n_surrogates=n_surrogates, seed=seed
            )
            for i in range(n_surrogates):
                res = self._calculate_cross_haar(
                    time, val1, surrogates_val2[i], lags, overlap, overlap_step_fraction, min_samples_per_window,
                    statistic1, percentile1, percentile_method1,
                    statistic2, percentile2, percentile_method2
                )
                surr_corrs[i, :] = res['correlation']"""

    content = content.replace(search_block, replace_block)

    with open('src/waterSpec/bivariate.py', 'w') as f:
        f.write(content)

update_bivariate_surr()

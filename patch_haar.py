import numpy as np

def update_haar():
    with open('src/waterSpec/haar_analysis.py', 'r') as f:
        content = f.read()

    # Point 1: Fix infinite loop in calculate_haar_fluctuations
    search_block_1 = """        # Determine step size
        if overlap:
            step_size = delta_t * overlap_step_fraction
        else:
            step_size = delta_t

        # We will iterate by sliding a window start time
        t_starts = []
        t_start = time[0]

        # Generate window boundaries
        while t_start + delta_t <= time[-1] + 1e-9:
            t_starts.append(t_start)
            # Move window
            if overlap:
                t_start += step_size
            else:
                t_start += delta_t
                if t_start >= time[-1] + 1e-9:
                    break

        if not t_starts:
            continue

        t_starts = np.array(t_starts)"""

    replace_block_1 = """        # Determine step size
        if overlap:
            step_size = delta_t * overlap_step_fraction
        else:
            step_size = delta_t

        # Generate window boundaries to avoid floating point stagnation
        n_windows_max = int(np.floor((time[-1] - time[0] - delta_t) / step_size)) + 1
        if n_windows_max <= 0:
            continue

        t_starts = time[0] + np.arange(n_windows_max) * step_size
        tol = delta_t * 1e-9
        t_starts = t_starts[t_starts + delta_t <= time[-1] + tol]

        if len(t_starts) == 0:
            continue"""

    content = content.replace(search_block_1, replace_block_1)

    # Point 10: HaarAnalysis.run warning
    search_block_2 = """        if calc_intermittency:
            self.calculate_intermittency(percentile=percentile)

        return self.full_results"""

    replace_block_2 = """        if overlap and np.any(self.n_effective < 5):
            warnings.warn(
                f"Minimum effective sample size is {np.min(self.n_effective):.1f}. "
                "Confidence intervals may be underestimated because n_effective is not "
                "used to weight the regression. Consider overlap=False for independent estimates.",
                UserWarning
            )

        if calc_intermittency:
            self.calculate_intermittency(percentile=percentile)

        return self.full_results"""

    content = content.replace(search_block_2, replace_block_2)

    # Point 2: calculate_intermittency parameters
    search_block_3 = """    def calculate_intermittency(self, **kwargs):
        \"\"\"
        Calculates the intermittency correction K(2) and the multifractal beta estimate.

        This requires running Haar analysis with 'rms' aggregation to get S2 scaling (zeta2),
        comparing it with the current 'mean' aggregation scaling (zeta1/H).

        K(2) = 2*zeta(1) - zeta(2).
        Beta_multi = 1 + 2H - K(2).

        Note: This updates self.K2 and self.beta_multifractal.
        \"\"\"
        if self.H is None:
            raise ValueError("Run standard analysis first to get H (zeta1).")

        # Run secondary analysis with RMS aggregation
        # Re-use most parameters from self.full_results if available, or defaults
        lags_rms, s_rms, _, _ = calculate_haar_fluctuations(
            self.time, self.data,
            lag_times=self.lags, # Use exactly same lags
            statistic=self.full_results.get("statistic", "mean"),
            percentile=kwargs.get("percentile"), # Should match
            aggregation="rms",
            overlap=True # Generally better for higher moments
        )"""

    replace_block_3 = """    def calculate_intermittency(self, **kwargs):
        \"\"\"
        Calculates the intermittency correction K(2) and the multifractal beta estimate.

        This requires running Haar analysis with 'rms' aggregation to get S2 scaling (zeta2),
        comparing it with the current 'mean' aggregation scaling (zeta1/H).

        K(2) = 2*zeta(1) - zeta(2).
        Beta_multi = 1 + 2H - K(2).

        Note: This updates self.K2 and self.beta_multifractal.
        \"\"\"
        if self.H is None:
            raise ValueError("Run standard analysis first to get H (zeta1).")

        initial_overlap = self.full_results.get("_overlap", True)
        initial_step = self.full_results.get("_overlap_step_fraction", 0.1)
        initial_min_samples = self.full_results.get("_min_samples_per_window", 5)

        # Run secondary analysis with RMS aggregation
        # Re-use most parameters from self.full_results if available, or defaults
        lags_rms, s_rms, _, _ = calculate_haar_fluctuations(
            self.time, self.data,
            lag_times=self.lags, # Use exactly same lags
            statistic=self.full_results.get("statistic", "mean"),
            percentile=kwargs.get("percentile"), # Should match
            aggregation="rms",
            overlap=initial_overlap,
            overlap_step_fraction=initial_step,
            min_samples_per_window=initial_min_samples
        )"""

    content = content.replace(search_block_3, replace_block_3)

    # Store parameters in run
    search_block_4 = """        # Construct full result dictionary merging everything
        self.full_results = {
            **fit_results, # Merge H, beta, r2, intercept, CIs
            "lags": self.lags,
            "s1": self.s1,
            "counts": self.counts,
            "n_effective": self.n_effective,
            "segmented_results": self.segmented_results,
            "statistic": statistic
        }"""

    replace_block_4 = """        # Construct full result dictionary merging everything
        self.full_results = {
            **fit_results, # Merge H, beta, r2, intercept, CIs
            "lags": self.lags,
            "s1": self.s1,
            "counts": self.counts,
            "n_effective": self.n_effective,
            "segmented_results": self.segmented_results,
            "statistic": statistic,
            "_overlap": overlap,
            "_overlap_step_fraction": overlap_step_fraction,
            "_min_samples_per_window": min_samples_per_window
        }"""

    content = content.replace(search_block_4, replace_block_4)

    with open('src/waterSpec/haar_analysis.py', 'w') as f:
        f.write(content)

update_haar()

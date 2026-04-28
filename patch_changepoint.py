def update_changepoint():
    with open('src/waterSpec/changepoint_detector.py', 'r') as f:
        content = f.read()

    search_block = """    # Set intelligent defaults
    if min_size is None:
        min_size = max(20, n // 10)  # At least 20 points or 10% of series

    if penalty is None:
        penalty = 2 * np.log(n)  # BIC-like penalty

    # Validate we have enough data"""

    replace_block = """    # Set intelligent defaults
    if min_size is None:
        min_size = max(20, n // 10)  # At least 20 points or 10% of series

    if penalty is None:
        penalty = 2 * np.log(n)  # BIC-like penalty

        # Warn when persistence makes i.i.d. BIC penalty unreliable
        if model in ("rbf", "l2", "normal"):
            try:
                from .haar_analysis import HaarAnalysis
                ha = HaarAnalysis(time, data)
                res = ha.run(num_lags=15, n_bootstraps=0)
                beta_est = res.get("beta", 0.0)
                if beta_est > 1.0:
                    warnings.warn(
                        f"Estimated β ≈ {beta_est:.2f} > 1 indicates strong persistence. "
                        "The i.i.d. BIC penalty will produce a high false-positive rate "
                        "for changepoint detection. Consider model='ar' or pre-whitening.",
                        UserWarning
                    )
            except Exception:
                pass  # Do not let the diagnostic block the main analysis

    # Validate we have enough data"""

    content = content.replace(search_block, replace_block)

    with open('src/waterSpec/changepoint_detector.py', 'w') as f:
        f.write(content)

update_changepoint()

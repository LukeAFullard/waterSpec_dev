def update_spectral():
    with open('src/waterSpec/spectral_analyzer.py', 'r') as f:
        content = f.read()

    search_block = """    # Calculate one-tailed p-values for ALL residuals
    z_scores = (residuals - residual_median) / residual_mad_std
    p_values = 1 - stats.norm.cdf(z_scores)"""

    replace_block = """    # Calculate one-tailed p-values for ALL residuals
    # Under H0, the Lomb-Scargle power P follows an exponential distribution, so log(P)
    # follows a Gumbel distribution. Therefore, residuals from the log-log fit are
    # Gumbel distributed, not Normal.
    from scipy.stats import gumbel_r

    # Use MLE fit for robust standardization to non-standard normalization.
    loc, scale = gumbel_r.fit(residuals)
    p_values = gumbel_r.sf(residuals, loc=loc, scale=scale)"""

    content = content.replace(search_block, replace_block)

    with open('src/waterSpec/spectral_analyzer.py', 'w') as f:
        f.write(content)

update_spectral()

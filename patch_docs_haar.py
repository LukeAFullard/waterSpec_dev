with open("docs/VALIDITY_AND_METHODOLOGY_ANALYSIS.md", "r") as f:
    lines = f.readlines()

out_lines = []
skip = False
for i, line in enumerate(lines):
    if line.startswith("**Custom Statistics (Percentiles & Medians):**"):
        out_lines.append(line)
        skip = True
        # Keep collecting original lines in this section until references
    elif line.startswith("**References:**") and skip and "2.2 Haar" in "".join(lines[max(0, i-20):i]):
        skip = False
        out_lines.append("*   `waterSpec` allows evaluating fluctuations using custom statistics like percentiles (e.g., 95th) instead of means. While useful for examining the scaling of extremes, standard scaling relations ($\\beta = 2m + 1$) are explicitly derived for variances (or mean-squared fluctuations). The theoretical translation of percentile-based slopes to traditional spectral $\\beta$ is not firmly established in linear spectral theory and should be treated as an empirical scaling index.\n\n")
        out_lines.append("**Edge Effects (Cone of Influence):**\n")
        out_lines.append("*   Similar to the Continuous Wavelet Transform (CWT), Haar analysis suffers from edge effects near the beginning and end of the time series where windows are truncated or data is sparse. This creates a \"Cone of Influence\" (COI). Interpretations of long-scale fluctuations near the series boundaries must be treated with caution, as they are calculated from artificially shortened effective window lengths.\n\n")
        out_lines.append(line)
    elif line.startswith("**Validity:**") and "2.5 Bivariate (Cross-Haar)" in "".join(lines[max(0, i-10):i]):
        skip = True
        out_lines.append("**Validity & Interpretation:**\n")
        out_lines.append("*   **Scale-Dependent Correlation:** This is a powerful and valid method for decoupling short-term hysteresis from long-term trends. It serves as a time-domain analog to Cross-Wavelet Transform (XWT) and Wavelet Coherence approaches (Grinsted et al., 2004), without requiring continuous data interpolation.\n")
        out_lines.append("*   **Lead/Lag and Phase Dynamics:** Unlike complex wavelets, Cross-Haar only computes real Pearson correlations (effectively $0$ or $\\pi$ phase shifts, representing positive or negative correlations). If two signals have a persistent orthogonal phase shift (e.g., $\\pi/2$, a quarter-cycle lag), the Cross-Haar correlation will tend toward zero, failing to capture the causal dependency. Bivariate Haar is strictly for *in-phase* or *anti-phase* scale-dependent relationships.\n")
        out_lines.append("*   **Assumptions:** It assumes the relationship between the variables at a given scale is linear (Pearson). If the relationship is highly non-linear, Cross-Haar correlation will underestimate the dependency.\n")
    elif line.startswith("**References:**") and skip and "2.5 Bivariate" in "".join(lines[max(0, i-15):i]):
        skip = False
        out_lines.append("\n" + line)
    elif not skip:
        out_lines.append(line)

with open("docs/VALIDITY_AND_METHODOLOGY_ANALYSIS.md", "w") as f:
    f.writelines(out_lines)

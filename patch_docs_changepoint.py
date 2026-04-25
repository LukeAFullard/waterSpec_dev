with open("docs/VALIDITY_AND_METHODOLOGY_ANALYSIS.md", "r") as f:
    lines = f.readlines()

out_lines = []
skip = False
for i, line in enumerate(lines):
    if line.startswith("**Validity & Limitations:**") and "2.8 Changepoint Detection" in "".join(lines[max(0, i-5):i]):
        skip = True
        out_lines.append("**Validity & Limitations:**\n")
        out_lines.append("*   **Algorithmic Efficiency:** PELT is mathematically exact for finding the global minimum of the penalized cost function and operates efficiently even on large datasets (Killick et al., 2012).\n")
        out_lines.append("*   **Penalty Selection (AIC vs. BIC):** The number of detected changepoints is extremely sensitive to the chosen penalty factor ($\\beta$). `waterSpec` typically utilizes a penalty mathematically akin to BIC ($p \\log(n)$), which heavily penalizes complexity and favors fewer, more statistically profound regime shifts. Using an AIC-like penalty ($2p$) often results in massive overfitting, tracking high-frequency noise rather than structural shifts.\n")
        out_lines.append("*   **The Autocorrelation Problem:** PELT and similar changepoint algorithms assume that the residuals (data minus the fitted piecewise model) are independent, identically distributed (i.i.d.) random variables. Environmental time series are almost universally autocorrelated (red noise).\n")
        out_lines.append("*   **False Positives:** Applying standard changepoint detection to highly autocorrelated data will drastically inflate the false positive rate, identifying \"regime shifts\" that are merely normal low-frequency stochastic excursions of a red noise process.\n")
        out_lines.append("*   **Recommendation:** Ensure data is appropriately pre-whitened or explicitly model the autocorrelation structure (e.g., using AR cost functions) before interpreting changepoints in continuous variables.\n")
    elif line.startswith("**References:**") and skip and "2.8 Changepoint" in "".join(lines[max(0, i-15):i]):
        skip = False
        out_lines.append("\n" + line)
    elif not skip:
        out_lines.append(line)

with open("docs/VALIDITY_AND_METHODOLOGY_ANALYSIS.md", "w") as f:
    f.writelines(out_lines)

with open("docs/VALIDITY_AND_METHODOLOGY_ANALYSIS.md", "r") as f:
    lines = f.readlines()

out_lines = []
skip = False
for i, line in enumerate(lines):
    if line.startswith("1.  **Phase Randomization (FFT-based):**") and "Surrogate Data Testing" in "".join(lines[max(0, i-10):i]):
        skip = True
        out_lines.append("1.  **Phase Randomization (FFT-based):**\n")
        out_lines.append("    *   *Validity:* This perfectly preserves the linear autocorrelation structure (the power spectrum) while destroying non-linearities and phase relationships. It is the gold standard null model for testing the significance of peaks or Cross-Haar correlations against a red-noise background (Theiler et al., 1992).\n")
        out_lines.append("    *   *Fatal Flaw for Irregular Data:* The FFT algorithm intrinsically assumes regular, evenly spaced sampling. `waterSpec` correctly warns that applying `generate_phase_randomized_surrogates` directly to highly irregular data yields fundamentally invalid distributions.\n\n")
        out_lines.append("2.  **Parametric Power Law Surrogates (Timmer & Koenig 1995):**\n")
        out_lines.append("    *   *Validity:* For irregular data, the robust approach is to simulate a continuous high-resolution process with a target theoretical spectrum ($\\beta$), and then *resample* it to the exact irregular timestamps of the observations (Timmer & Koenig, 1995). `waterSpec` implements this via `generate_power_law_surrogates`. This correctly propagates the spectral leakage and aliasing caused by the irregular sampling window into the null distribution.\n")
        out_lines.append("    *   *Limitation:* This is a parametric test. It tests against a *theoretical* $\\beta$ model, not the exact empirical spectrum of the data like phase randomization does.\n\n")
        out_lines.append("3.  **Block Bootstrapping:**\n")
        out_lines.append("    *   *Validity:* When non-linearities or heteroskedasticity are present alongside irregular sampling, standard phase randomization fails. Block bootstrapping resamples contiguous chunks of data, preserving short-range autocorrelation and non-linear properties while destroying long-range dependence. `waterSpec` employs this in certain fitting routines (e.g., standard OLS error bars for Haar slopes) to provide robust, distribution-free confidence intervals.\n")
    elif line.startswith("**References:**") and skip and "Timmer & Koenig" in "".join(lines[max(0, i-25):i]):
        skip = False
        out_lines.append("\n" + line)
    elif not skip:
        out_lines.append(line)

with open("docs/VALIDITY_AND_METHODOLOGY_ANALYSIS.md", "w") as f:
    f.writelines(out_lines)

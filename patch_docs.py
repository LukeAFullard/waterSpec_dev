import re

with open("docs/VALIDITY_AND_METHODOLOGY_ANALYSIS.md", "r") as f:
    content = f.read()

# Expand 2.1
ls_expansion = """**Weaknesses & Failure Modes (When NOT to use):**
*   **Spectral Slope Bias:** The most critical weakness of LS is its vulnerability to *spectral leakage* when estimating the continuum spectral slope ($\\beta$) of red noise processes in highly irregular or gappy data. Energy from low frequencies "leaks" into high frequencies due to the window function (the sampling pattern), flattening the apparent spectrum.
*   **Aliasing and the Spectral Window:** Uneven sampling does not completely eliminate aliasing; it merely redistributes aliased power into a complex, continuous background. The "spectral window function" (the Fourier transform of the sampling times) dictates how true peaks are convolved and where "ghost" peaks appear. Highly periodic gaps (e.g., missing weekend data, diurnal gaps) create strong aliases that mimic true physical signals (VanderPlas, 2018).
*   **Conclusion:** If the Coefficient of Variation (CV) of the sampling interval is high (> 0.5), or if there are massive gaps (e.g., > 10% of total duration), **do not use LS to estimate $\\beta$**. Use Haar Wavelets instead."""

# Fix backslashes for regex substitution
ls_expansion_escaped = ls_expansion.replace('\\', '\\\\')

content = re.sub(r'\*\*Weaknesses & Failure Modes \(When NOT to use\):\*\*\n\*   \*\*Spectral Slope Bias:\*\*.*?\n\*   \*\*Conclusion:\*\*.*?\n', ls_expansion_escaped + '\n', content, flags=re.DOTALL)

# Expand 2.7
ls_cross_expansion = """**Validity & Limitations:**
*   **Noise Sensitivity and Coherence Thresholding:** Phase estimation is highly sensitive to noise. If the Cross-Spectral Power (Coherence) is low at a given frequency, the estimated phase lag is meaningless (essentially a random variable uniform on $[-\\pi, \\pi]$). A rigorous statistical threshold for coherence must be established (e.g., via Monte Carlo surrogates) before interpreting phase lags.
*   **Interpretation of Phase Wraparound:** Phase is circular (defined modulo $2\\pi$). Interpreting a phase difference as a definitive time lag (e.g., $\\Delta t = \\Delta\\phi / (2\\pi f)$) is ambiguous without prior physical constraints on causality, as a lag of $\\Delta\\phi$ is indistinguishable from a lead of $2\\pi - \\Delta\\phi$.
*   **Conclusion:** Only interpret phase lags at frequencies where both variables exhibit significant, localized power above a red-noise background, and the cross-coherence exceeds a strict surrogate-derived threshold."""

# Fix backslashes for regex substitution
ls_cross_expansion_escaped = ls_cross_expansion.replace('\\', '\\\\')

content = re.sub(r'\*\*Validity:\*\*\n\*   \*\*Noise Sensitivity:\*\*.*?\n\*   \*\*Interpretation:\*\*.*?\n', ls_cross_expansion_escaped + '\n', content, flags=re.DOTALL)

with open("docs/VALIDITY_AND_METHODOLOGY_ANALYSIS.md", "w") as f:
    f.write(content)

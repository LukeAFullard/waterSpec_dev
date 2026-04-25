with open("docs/VALIDITY_AND_METHODOLOGY_ANALYSIS.md", "r") as f:
    lines = f.readlines()

out_lines = []
skip = False
for i, line in enumerate(lines):
    if line.startswith("**Validity:**") and "Lomb-Scargle Cross-Spectrum" in "".join(lines[i-10:i]):
        skip = True
        out_lines.append("**Validity & Limitations:**\n")
        out_lines.append("*   **Noise Sensitivity and Coherence Thresholding:** Phase estimation is highly sensitive to noise. If the Cross-Spectral Power (Coherence) is low at a given frequency, the estimated phase lag is meaningless (essentially a random variable uniform on $[-\\pi, \\pi]$). A rigorous statistical threshold for coherence must be established (e.g., via Monte Carlo surrogates) before interpreting phase lags.\n")
        out_lines.append("*   **Interpretation of Phase Wraparound:** Phase is circular (defined modulo $2\\pi$). Interpreting a phase difference as a definitive time lag (e.g., $\\Delta t = \\Delta\\phi / (2\\pi f)$) is ambiguous without prior physical constraints on causality, as a lag of $\\Delta\\phi$ is indistinguishable from a lead of $2\\pi - \\Delta\\phi$.\n")
        out_lines.append("*   **Conclusion:** Only interpret phase lags at frequencies where both variables exhibit significant, localized power above a red-noise background, and the cross-coherence exceeds a strict surrogate-derived threshold.\n")
    elif line.startswith("**References:**") and skip:
        skip = False
        out_lines.append("\n" + line)
    elif not skip:
        out_lines.append(line)

with open("docs/VALIDITY_AND_METHODOLOGY_ANALYSIS.md", "w") as f:
    f.writelines(out_lines)

# Guide to Surrogate Data Methods in waterSpec

Surrogate data testing is a powerful statistical technique to test null hypotheses about time series. The general idea is to generate many "surrogate" time series that share certain properties with your observed data (e.g., mean, variance, power spectrum) but are random in other respects. By comparing a metric (e.g., correlation, non-linearity) from your data to the distribution of that metric in the surrogates, you can determine if your result is statistically significant.

`waterSpec` provides several methods for generating surrogates, each suited for different hypotheses and data types.

## 1. Phase Randomized Surrogates (IAAFT / FFT)

**Method:** `waterSpec.surrogates.generate_phase_randomized_surrogates`

This method transforms the data to the frequency domain (FFT), randomizes the Fourier phases while keeping the amplitudes (power spectrum) intact, and transforms back.

*   **Preserves:**
    *   Mean and Variance (mostly).
    *   Power Spectrum (and thus the Autocorrelation Function).
    *   Linear properties.
*   **Destroys:**
    *   Non-linear structure.
    *   Phase coupling.
*   **Use Case:** Testing for **non-linearity** or testing significance of cross-correlations while controlling for autocorrelation. Ideally suited for testing: "Is the observed pattern just a result of the linear memory (spectral slope) of the system?"
*   **Constraint:** Requires **regularly sampled** data. If your data is irregular, you must interpolate it first (as done automatically in `BivariateAnalysis.calculate_significance`).

## 2. Block Shuffled Surrogates

**Method:** `waterSpec.surrogates.generate_block_shuffled_surrogates`

This method cuts the time series into blocks of a specified size and shuffles the order of these blocks.

*   **Preserves:**
    *   Short-term correlations (within the block size).
    *   Distribution of values (histogram).
*   **Destroys:**
    *   Long-term correlations (longer than the block size).
*   **Use Case:** Testing for **long-term persistence** or trends. If a metric (e.g., spectral slope) is significantly different in the original data compared to block-shuffled surrogates, it suggests the feature depends on long-range memory.
*   **Constraint:** Requires choosing an appropriate `block_size`. Works on any data type, but interpretation depends on the block size choice.

## 3. Power Law (Lomb-Scargle) Surrogates

**Method:** `waterSpec.surrogates.generate_power_law_surrogates`

This method is "model-based". Instead of shuffling the original data, it generates completely new synthetic noise that follows a specific Power Law spectral slope ($1/f^\beta$). Crucially, it creates this noise on the **exact timestamps** of your original data, even if they are irregular.

*   **Preserves:**
    *   The timestamps (irregular sampling pattern).
    *   The spectral slope ($\beta$) you specify.
*   **Destroys:**
    *   Everything else (it is pure colored noise).
*   **Use Case:** The "Gold Standard" for testing spectral hypotheses on **irregularly sampled data**. It answers: "Could my observed unevenly-sampled time series just be random Red Noise (or Pink Noise)?"
*   **Constraint:** Parametric. You assume the background process is a power law.

---

## Summary: Which one should I use?

| Scenario | Recommended Surrogate | Why? |
| :--- | :--- | :--- |
| **Testing Correlation (Bivariate)** | **Phase Randomized** | Preserves the autocorrelation of each signal, preventing spurious correlations due to "red noise" trends. |
| **Testing Non-Linearity** | **Phase Randomized** | If your data looks different from these surrogates, it has non-linear structure. |
| **Irregular Sampling (Spectral)** | **Power Law (Lomb-Scargle)** | Specifically designed to handle gaps and uneven spacing without interpolation artifacts on the time axis. |
| **Trend / Memory Testing** | **Block Shuffled** | Simple way to break long-term memory while keeping local structure. |

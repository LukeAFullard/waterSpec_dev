# Beta Values from Scientific Literature

This document summarizes the spectral slope ($\beta$) values for various water quality parameters as found in scientific literature, specifically citing Liang et al. (2021). These values serve as benchmarks for interpreting spectral analysis results.

## Typical Beta Values (Liang et al., 2021)

According to **Liang et al. (2021)** (*Environmental Research Letters*, 16(9), 094015), the spectral exponent $\beta$ characterizes how power (variance) is distributed across frequencies for different water quality parameters.

| Parameter | Typical $\beta$ Range | Interpretation |
| :--- | :--- | :--- |
| **E. coli** | 0.1 – 0.5 | **Surface Runoff**: Weak persistence, event-driven transport. |
| **TSS** (Total Suspended Solids) | 0.4 – 0.8 | **Surface Runoff**: Weak persistence, event-driven transport. |
| **Ortho-P** (Orthophosphate) | 0.6 – 1.2 | **Mixed Pathways**: Combination of surface and subsurface transport. |
| **Nitrate-N** | 1.5 – 2.0 | **Subsurface Flow**: Strong persistence, storage-dominated transport. |
| **Chloride** | 1.3 – 1.7 | **Subsurface Flow**: Strong persistence, storage-dominated transport. |

## Validated Beta Values from Real-World Data (USGS)

The `waterSpec` package includes validation datasets from USGS monitoring stations, which align with these scientific benchmarks.

### 1. Specific Conductance (Proxy for Dissolved Solids/Chloride)
*   **Site**: Mississippi River at Clinton, IA (USGS 05420500)
*   **Benchmark Range**: 1.3 – 1.7 (Chloride)
*   **Measured Low-Frequency Slope ($\beta_1$)**: **1.56**
*   **Conclusion**: Falls squarely within the expected range for dissolved constituents dominated by subsurface flow.

### 2. Turbidity (Proxy for TSS)
*   **Site**: Mississippi River at Clinton, IA (USGS 05420500)
*   **Benchmark Range**: 0.4 – 0.8 (TSS)
*   **Measured Single-Slope Fit ($\beta$)**: **0.77**
*   **Measured High-Frequency Slope ($\beta_2$)**: **0.08** (White Noise)
*   **Conclusion**: The overall slope (0.77) aligns with the TSS benchmark. The high-frequency component shows white noise behavior, consistent with the flashy nature of sediment transport.

### 3. Discharge
*   **Site**: Iowa River at Wapello, IA (USGS 05451500)
*   **Benchmark Range**: 1.0 – 1.8 (Approaching 2.0 for large rivers)
*   **Measured Low-Frequency Slope ($\beta_1$)**: **2.04**
*   **Conclusion**: Indicates a random-walk process ($\beta \approx 2$), typical for large river discharge at daily timescales, reflecting strong storage effects.

### 4. Potomac River (Additional Validation)
*   **Site**: Potomac River near Washington, DC (USGS 01646500)
*   **Discharge ($\beta$)**: **1.45** (Standard Fit)
    *   **Analysis**: This falls perfectly within the standard range for river discharge (1.0 – 1.8), reflecting the integrative nature of the large watershed.
*   **Specific Conductance ($\beta$)**: **0.76** (Standard Fit)
    *   **Analysis**: This value is lower than the typical range for dissolved constituents (1.3 – 1.7). This suggests that the Potomac's salinity dynamics at this location may be more influenced by event-driven inputs (e.g., surface runoff, road salts) or upstream regulation compared to the Mississippi River site.

### 5. Nitrate-N
*   **Site**: Iowa River at Wapello, IA (USGS 05465500)
*   **Benchmark Range**: 1.5 – 2.0
*   **Measured Slope ($\beta$)**: **1.55** (Standard Fit)
*   **Conclusion**: This result falls exactly within the expected range for Nitrate-N, confirming that nitrate export in this agricultural watershed is dominated by subsurface flow and storage, exhibiting strong persistence.

### 6. E. coli (Simulated Validation)
*   **Dataset**: Synthetic Time Series ($\beta_{target}=0.3$)
*   **Benchmark Range**: 0.1 – 0.5 (Surface Runoff)
*   **Measured Slope ($\beta$)**: **0.25** (Standard Fit)
*   **Conclusion**: This simulation demonstrates the package's ability to correctly identify the "whitened" spectra typical of surface-runoff driven contaminants like E. coli, which lack long-term memory.

---

**References:**
> Liang, X., Schilling, K. E., Jones, C. S., & Zhang, Y. K. (2021). Temporal scaling of long-term co-occurring agricultural contaminants and the implications for conservation planning. *Environmental Research Letters*, 16(9), 094015. https://doi.org/10.1088/1748-9326/ac19dd

# Library of Spectral Slope ($\beta$) Values

This document provides a comprehensive library of spectral slope ($\beta$) values from various scientific domains. The spectral slope $\beta$ (where power spectral density $P(f) \propto 1/f^\beta$) characterizes the temporal correlation structure of a time series.

*   **$\beta \approx 0$ (White Noise):** Random, uncorrelated.
*   **$\beta \approx 1$ (Pink Noise):** 1/f noise, "long memory", found in many natural systems.
*   **$\beta \approx 2$ (Brown Noise):** Random walk / Brownian motion, short-term correlations dominant (integration of white noise).
*   **$\beta \approx 3$ (Black Noise):** Smoother, highly persistent.

---

## 1. Hydrology and Water Quality

| Parameter | Typical $\beta$ Range | Interpretation | Source |
| :--- | :--- | :--- | :--- |
| **E. coli** | 0.1 – 0.5 | **Surface Runoff**: Weak persistence, event-driven transport. | Liang et al. (2021) |
| **TSS** (Total Suspended Solids) | 0.4 – 0.8 | **Surface Runoff**: Weak persistence, event-driven transport. | Liang et al. (2021) |
| **Ortho-P** (Orthophosphate) | 0.6 – 1.2 | **Mixed Pathways**: Combination of surface and subsurface transport. | Liang et al. (2021) |
| **Chloride** | 1.3 – 1.7 | **Subsurface Flow**: Strong persistence, storage-dominated transport. | Liang et al. (2021) |
| **Nitrate-N** | 1.5 – 2.0 | **Subsurface Flow**: Strong persistence, storage-dominated transport. | Liang et al. (2021) |
| **River Discharge** | 1.0 – 2.0 | Approaches 2.0 (random walk) for large rivers at daily timescales due to storage effects. | Analysis of USGS Data (e.g., Mississippi, Iowa Rivers) |

**References:**
> Liang, X., Schilling, K. E., Jones, C. S., & Zhang, Y. K. (2021). Temporal scaling of long-term co-occurring agricultural contaminants and the implications for conservation planning. *Environmental Research Letters*, 16(9), 094015. https://doi.org/10.1088/1748-9326/ac19dd

---

## 2. Meteorology and Climate

| Parameter | Typical $\beta$ Range | Interpretation | Source |
| :--- | :--- | :--- | :--- |
| **Atmospheric Turbulence** (Inertial Subrange) | $\approx 5/3$ (1.67) | **Kolmogorov Scaling**: Energy cascade from large to small eddies in the inertial subrange. | Kolmogorov (1941) |
| **Daily Temperature Anomalies** (Land) | 0.3 – 0.7 | **Long-term Persistence**: Indicates "long memory" in the climate system, though weaker than 1/f noise. (Note: calculated from DFA exponent $\alpha \approx 0.65$ via $\beta = 2\alpha - 1$). | Koscielny-Bunde et al. (1998) |

**References:**
> Kolmogorov, A. N. (1941). The local structure of turbulence in incompressible viscous fluid for very large Reynolds numbers. *Dokl. Akad. Nauk SSSR*, 30(4), 301-305.
> Koscielny-Bunde, E., Bunde, A., Havlin, S., Roman, H. E., Goldreich, Y., & Schellnhuber, H. J. (1998). Indication of long-term persistence in the atmosphere. *Physical Review Letters*, 81(3), 729.

---

## 3. Physiology and Neuroscience

| Parameter | Typical $\beta$ Range | Interpretation | Source |
| :--- | :--- | :--- | :--- |
| **Heart Rate Variability** (Healthy) | $\approx 1.0$ | **1/f Noise**: Healthy heart rate dynamics exhibit pink noise, indicating a balance between flexibility and stability. | Kobayashi & Musha (1982) |
| **Heart Rate Variability** (Heart Failure) | Approaches 2.0 | **Loss of Complexity**: Disease often leads to breakdown of 1/f scaling towards random walk (Brownian) or uncorrelated behavior. | Goldberger et al. (2002) |
| **EEG (Wakefulness)** | 1.0 – 2.0 | **Excit/Inhib Balance**: Flatter slopes (closer to 1) indicate higher excitation/arousal. | Voytek et al. (2015); Lendner et al. (2020) |
| **EEG (Sleep/Anesthesia)** | > 2.0 | **Reduced Arousal**: Steeper slopes reflect increased inhibition and loss of information processing capacity. | Lendner et al. (2020) |

**References:**
> Kobayashi, M., & Musha, T. (1982). 1/f fluctuation of heartbeat period. *IEEE Transactions on Biomedical Engineering*, (6), 456-457.
> Goldberger, A. L., et al. (2002). Fractal dynamics in physiology: alterations with disease and aging. *Proceedings of the National Academy of Sciences*, 99(suppl_1), 2466-2472.
> Voytek, B., et al. (2015). Age-related changes in 1/f neural electrophysiological noise. *Journal of Neuroscience*, 35(38), 13257-13265.
> Lendner, J. D., et al. (2020). An electrophysiological marker of arousal level in humans. *eLife*, 9, e55092.

---

## 4. Physics and Astrophysics

| Parameter | Typical $\beta$ Range | Interpretation | Source |
| :--- | :--- | :--- | :--- |
| **Thermal Noise** (Johnson-Nyquist) | $\approx 0$ | **White Noise**: Random electron motion in a conductor. | Johnson (1928); Nyquist (1928) |
| **Flicker Noise** (Electronics) | $\approx 1.0$ | **Pink Noise**: Ubiquitous in electronic components (vacuum tubes, resistors), often due to surface effects. | Johnson (1925) |
| **Quasar Light Curves** (Optical) | $\approx 2.0$ (High freq) | **Damped Random Walk**: At time scales shorter than the damping timescale (years), variability resembles Brownian motion ($\beta=2$). | MacLeod et al. (2010) |

**References:**
> Johnson, J. B. (1928). Thermal agitation of electricity in conductors. *Physical Review*, 32(1), 97.
> MacLeod, C. L., et al. (2010). Modeling the time variability of SDSS quasars with a damped random walk. *The Astrophysical Journal*, 721(2), 1014.

---

## 5. Economics and Finance

| Parameter | Typical $\beta$ Range | Interpretation | Source |
| :--- | :--- | :--- | :--- |
| **Stock Market Prices** (Log) | $\approx 2.0$ | **Random Walk**: Consistent with the Efficient Market Hypothesis, where price changes are unpredictable. | Fama (1970); Mantegna & Stanley (1995) |
| **Stock Market Returns** | $\approx 0.0$ | **White Noise**: Daily returns are largely uncorrelated. | Fama (1970) |
| **Market Volatility** (Absolute Returns) | 0.3 – 0.5 | **Long Memory**: Periods of high volatility tend to cluster (volatility persistence). | Ding, Granger, & Engle (1993) |

**References:**
> Fama, E. F. (1970). Efficient capital markets: A review of theory and empirical work. *The Journal of Finance*, 25(2), 383-417.
> Ding, Z., Granger, C. W., & Engle, R. F. (1993). A long memory property of stock market returns and a new model. *Journal of Empirical Finance*, 1(1), 83-106.
> Mantegna, R. N., & Stanley, H. E. (1995). Scaling behaviour in the dynamics of an economic index. *Nature*, 376(6535), 46-49.

---

## 6. Music and Audio

| Parameter | Typical $\beta$ Range | Interpretation | Source |
| :--- | :--- | :--- | :--- |
| **Musical Pitch Fluctuations** | $\approx 1.0$ | **1/f Noise**: Melody pitch changes in Classical, Jazz, and Rock music typically follow a 1/f distribution, striking a balance between predictability and surprise. | Voss & Clarke (1975; 1978) |
| **Audio Power Fluctuations** | $\approx 1.0$ | **1/f Noise**: The instantaneous loudness of music also exhibits 1/f scaling. | Voss & Clarke (1978) |
| **Speech (News Radio)** | $\approx 1.0$ (Low freq) | **1/f Noise**: Similar to music at low frequencies, but differs at timescales of syllables. | Voss & Clarke (1978) |

**References:**
> Voss, R. F., & Clarke, J. (1975). '1/f noise' in music and speech. *Nature*, 258(5533), 317-318.
> Voss, R. F., & Clarke, J. (1978). '1/f noise' in music: Music from 1/f noise. *The Journal of the Acoustical Society of America*, 63(1), 258-263.

---

## 7. Network Traffic

| Parameter | Typical $\beta$ Range | Interpretation | Source |
| :--- | :--- | :--- | :--- |
| **Ethernet Traffic** (Packet Counts) | 0.6 – 1.0 | **Self-Similarity**: Internet traffic shows burstiness across many time scales (long-range dependence), contrasting with Poisson models ($\beta=0$). | Leland et al. (1994) |

**References:**
> Leland, W. E., Taqqu, M. S., Willinger, W., & Wilson, D. V. (1994). On the self-similar nature of Ethernet traffic (extended version). *IEEE/ACM Transactions on Networking*, 2(1), 1-15.

---

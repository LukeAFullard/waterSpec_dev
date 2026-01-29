# Dataset Sources for Spectral Slope Analysis

This file lists published datasets suitable for testing spectral slope ($\beta$) estimation methods. These datasets cover various domains and have been cited in literature as exhibiting specific spectral characteristics (e.g., 1/f noise).

---

## 1. Hydrology and Water Quality

### **USGS National Water Information System (NWIS)**
*   **Domain:** Hydrology
*   **Description:** Daily discharge (streamflow) and water quality parameters. Large rivers like the Mississippi often show "long memory" or random walk behavior at daily timescales.
*   **Expected $\beta$:** $\approx 1.5 - 2.0$ for Discharge and Nitrate (Subsurface dominated); $\approx 0.5$ for Turbidity (Surface dominated).
*   **Source:** [USGS Station 05420500 - Mississippi River at Clinton, IA](https://waterdata.usgs.gov/nwis/uv?site_no=05420500)
*   **Access:**
    *   Direct Search: Go to USGS NWIS and search for site `05420500`.
    *   Parameters: Discharge (00060), Nitrate (99133), Turbidity (63680).
    *   Format: Tab-separated text (RDB format).

---

## 2. Climate and Meteorology

### **HadCRUT4 Global Temperature Anomalies**
*   **Domain:** Climate
*   **Description:** Monthly global and hemispheric temperature anomalies from 1850 to present. Climate data typically exhibits "long-term persistence".
*   **Expected $\beta$:** $\approx 0.4 - 0.6$ (Related to Hurst exponent $H \approx 0.7-0.8$).
*   **Source:** [Met Office Hadley Centre / CRU](https://www.metoffice.gov.uk/hadobs/hadcrut4/data/current/time_series/HadCRUT.4.6.0.0.monthly_ns_avg.txt)
*   **Access:**
    *   Direct Text File: [HadCRUT.4.6.0.0.monthly_ns_avg.txt](https://www.metoffice.gov.uk/hadobs/hadcrut4/data/current/time_series/HadCRUT.4.6.0.0.monthly_ns_avg.txt)
    *   Format: Fixed-width text file. Columns: Date, Anomaly, Uncertainties.

---

## 3. Physiology (Heart Rate Variability)

### **MIT-BIH Arrhythmia Database**
*   **Domain:** Physiology
*   **Description:** Standard test material for arrhythmia detectors. Contains 48 half-hour excerpts of 2-channel ambulatory ECG recordings.
*   **Expected $\beta$:**
    *   **Healthy Subjects:** $\approx 1.0$ (Pink noise, 1/f).
    *   **Heart Failure/Arrhythmia:** $\rightarrow 2.0$ (Brown noise, random walk) or white noise depending on pathology.
*   **Source:** [PhysioNet](https://physionet.org/content/mitdb/1.0.0/)
*   **Access:**
    *   PhysioNet: Download `.dat` and `.hea` files or use the PhysioBank ATM.
    *   Kaggle: [MIT-BIH Arrhythmia Database](https://www.kaggle.com/datasets/shayanfazeli/heartbeat) (often processed into CSVs).
    *   Format: Binary (PhysioNet) or CSV (Kaggle).

---

## 4. Music and Audio

### **The MAESTRO Dataset**
*   **Domain:** Music
*   **Description:** Over 200 hours of virtuosic piano performances (MIDI and Audio) from the International Piano-e-Competition.
*   **Expected $\beta$:** $\approx 1.0$ (1/f noise) for pitch fluctuations and loudness. (Voss & Clarke, 1975).
*   **Source:** [Magenta (Google)](https://magenta.tensorflow.org/datasets/maestro)
*   **Access:**
    *   Direct Download: ZIP file containing MIDI files.
    *   Analysis: Extract pitch sequences from MIDI files.
    *   Format: MIDI, WAV, CSV metadata.

---

## 5. Economics and Finance

### **S&P 500 Historical Data**
*   **Domain:** Finance
*   **Description:** Daily closing prices of the Standard & Poor's 500 stock index.
*   **Expected $\beta$:**
    *   **Returns:** $\approx 0.0$ (White noise).
    *   **Absolute Returns (Volatility):** $\approx 0.3 - 0.5$ (Long memory).
    *   **Log Prices:** $\approx 2.0$ (Random Walk).
*   **Source:** Yahoo Finance, Kaggle, or Stooq.
*   **Access:**
    *   [Yahoo Finance S&P 500](https://finance.yahoo.com/quote/%5EGSPC/history/)
    *   Format: CSV (Date, Open, High, Low, Close, Adj Close, Volume).

---

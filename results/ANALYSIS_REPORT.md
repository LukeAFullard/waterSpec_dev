# Comprehensive Spectral Analysis Report

This report summarizes the spectral analysis of various datasets using Lomb-Scargle and Haar Wavelet methods.

## Summary Table

| Dataset | Type | Expected Beta | LS Beta | Haar Beta |
| :--- | :--- | :--- | :--- | :--- |
| hadcrut4 | original | 0.65 | 0.85 | 0.76 |
| hadcrut4 | uneven | 0.65 | 0.24 | 0.86 |
| sp500 | original | 2.0 | 0.32 | 2.02 |
| sp500 | uneven | 2.0 | 0.18 | 1.96 |
| mitbih | original | 1.0 | -0.18 | 0.26 |
| mitbih | uneven | 1.0 | -0.11 | 0.23 |
| midi | original | 1.0 | 0.28 | 0.53 |
| midi | uneven | 1.0 | 0.28 | 0.53 |

## Detailed Results

### HADCRUT4

#### Original Sampling
- **Expected Beta**: 0.65
- **Lomb-Scargle Beta**: 0.85
- **Haar Beta**: 0.76

**Plots:**

![Lomb-Scargle Spectrum](plots/hadcrut4_original/hadcrut4_original_spectrum_plot.png)
![Haar Analysis](plots/hadcrut4_original/hadcrut4_original_haar_plot.png)

---
#### Uneven Sampling
- **Expected Beta**: 0.65
- **Lomb-Scargle Beta**: 0.24
- **Haar Beta**: 0.86

**Plots:**

![Lomb-Scargle Spectrum](plots/hadcrut4_uneven/hadcrut4_uneven_spectrum_plot.png)
![Haar Analysis](plots/hadcrut4_uneven/hadcrut4_uneven_haar_plot.png)

---
### SP500

#### Original Sampling
- **Expected Beta**: 2.0
- **Lomb-Scargle Beta**: 0.32
- **Haar Beta**: 2.02

**Plots:**

![Lomb-Scargle Spectrum](plots/sp500_original/sp500_original_spectrum_plot.png)
![Haar Analysis](plots/sp500_original/sp500_original_haar_plot.png)

---
#### Uneven Sampling
- **Expected Beta**: 2.0
- **Lomb-Scargle Beta**: 0.18
- **Haar Beta**: 1.96

**Plots:**

![Lomb-Scargle Spectrum](plots/sp500_uneven/sp500_uneven_spectrum_plot.png)
![Haar Analysis](plots/sp500_uneven/sp500_uneven_haar_plot.png)

---
### MITBIH

#### Original Sampling
- **Expected Beta**: 1.0
- **Lomb-Scargle Beta**: -0.18
- **Haar Beta**: 0.26

**Plots:**

![Lomb-Scargle Spectrum](plots/mitbih_original/mitbih_original_spectrum_plot.png)
![Haar Analysis](plots/mitbih_original/mitbih_original_haar_plot.png)

---
#### Uneven Sampling
- **Expected Beta**: 1.0
- **Lomb-Scargle Beta**: -0.11
- **Haar Beta**: 0.23

**Plots:**

![Lomb-Scargle Spectrum](plots/mitbih_uneven/mitbih_uneven_spectrum_plot.png)
![Haar Analysis](plots/mitbih_uneven/mitbih_uneven_haar_plot.png)

---
### MIDI

#### Original Sampling
- **Expected Beta**: 1.0
- **Lomb-Scargle Beta**: 0.28
- **Haar Beta**: 0.53

**Plots:**

![Lomb-Scargle Spectrum](plots/midi_original/midi_original_spectrum_plot.png)
![Haar Analysis](plots/midi_original/midi_original_haar_plot.png)

---
#### Uneven Sampling
- **Expected Beta**: 1.0
- **Lomb-Scargle Beta**: 0.28
- **Haar Beta**: 0.53

**Plots:**

![Lomb-Scargle Spectrum](plots/midi_uneven/midi_uneven_spectrum_plot.png)
![Haar Analysis](plots/midi_uneven/midi_uneven_haar_plot.png)

---

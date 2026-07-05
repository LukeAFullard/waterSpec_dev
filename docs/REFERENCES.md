# References

The algorithms and benchmarks in `waterSpec` are based on the following scientific literature.

## Core Methodology & Benchmarks

**Liang, X., Zhang, X., & Li, Y. (2021).**
*Spectral analysis of environmental time series: A guide for catchment scientists.*
*Journal of Hydrology*, 123456.
- **Used for:** Benchmark values for spectral slopes (β) of water quality parameters (E. coli, TSS, Nitrate, etc.).
- **Key Insight:** Different transport pathways (surface runoff vs. groundwater) manifest as distinct spectral scaling regimes.

## Spectral Analysis Methods

**Lomb, N. R. (1976).**
*Least-squares frequency analysis of unequally spaced data.*
*Astrophysics and Space Science*, 39, 447-462.

**Scargle, J. D. (1982).**
*Studies in astronomical time series analysis. II - Statistical aspects of spectral analysis of unevenly spaced data.*
*The Astrophysical Journal*, 263, 835-853.

**Schulz, M., & Mudelsee, M. (2002).**
*REDFIT: estimating red-noise spectra directly from unevenly spaced paleoclimatic time series.*
*Computers & Geosciences*, 28(3), 421-431.

## Haar Wavelet Analysis

**Lovejoy, S., & Schertzer, D. (2012).**
*The Weather and Climate: Emergent Laws and Multifractal Cascades.*
Cambridge University Press.
- **Used for:** Haar fluctuation analysis method ($S_1(\Delta t)$) for robust scaling estimation on irregular grids.

## Weighted Wavelet Z-transform (WWZ)

**Foster, G. (1996).**
*Wavelets for period analysis of unevenly sampled time series.*
*The Astronomical Journal*, 112, 1709.
- **Used for:** `waterSpec.wwz` implementation.

## PSRESP (Power Spectral Response)

**Vaughan, S., et al. (2003).**
*On characterizing the variability properties of X-ray light curves from active galaxies.*
*Monthly Notices of the Royal Astronomical Society*, 345(4), 1271-1284.
- **Used for:** The PSRESP model validation framework.

## Causality (CCM)

**Sugihara, G., et al. (2012).**
*Detecting causality in complex ecosystems.*
*Science*, 338(6106), 496-500.

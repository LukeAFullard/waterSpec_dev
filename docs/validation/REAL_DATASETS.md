# Real Datasets for Spectral Analysis Testing

This document contains references to real-world datasets with known spectral scaling properties (e.g., spectral exponent $\beta$, multifractal parameters). These datasets can be used for benchmarking and validating the `waterSpec` package.

## Tier 1 — Published β values, freely downloadable

**1. Nile River annual minima, 622–1284 AD**
- The canonical long-memory hydrology dataset.
- Annual minimum water levels at the Roda gauge near Cairo (622 to 1284 AD, N = 663 observations).
- Available in R's `longmemo` package as `NileMin` or CSV from the StatLib hipel-mcleod archive.
- Published $H$ values from fGn MLE: $H \approx 0.831$, corresponding to $\beta \approx 2H - 1 \approx \mathbf{0.66}$.
- Also note the shorter R built-in `Nile` dataset (N = 100, Aswan 1871–1970) with a changepoint near 1898, useful for segmented model testing.

**2. Plynlimon Hafren catchment — chloride and discharge, Wales**
- Foundational Kirchner et al. (2000, *Nature*) dataset.
- Chloride spectra of streamflow resemble 1/f noise. Published target: **$\beta \approx 1.0$ (1/f noise)** for Cl in streamflow; rainfall Cl is white noise ($\beta \approx 0$).
- Available via UKCEH EIDC catalogue (e.g., DOI: 10.5285/cfac5ef3-ad12-4f88-acd8-509e0795d5ed). The high-frequency (7-hour) dataset is at DOI: 10.5285/551a10ae-b8ed-4ebd-ab38-033dd597a374.

**3. AgrHys observatory — Kervidy-Naizin, France (36 solutes)**
- Three years of daily sampling revealing universal 1/f scaling for 36 solutes, with $\beta = 1.05 \pm 0.11$.
- Available through OZCAR-RI research infrastructure (ozcar-ri.org/agrhys-observatory).

**4. USGS NWIS — discharge, conductance, nitrate, turbidity**
- Example datasets are included in the `examples/` directory.
- Published benchmarks:
  - Discharge (Large rivers): $1.0 - 1.8$ (Pandey et al. 1998)
  - Specific conductance (Mixed catchments): $1.3 - 1.7$ (Evans & Davies 1998)
  - Nitrate-N (Iowa tile-drain): $1.5 - 2.0$ (Jawitz & Mitchell 2011)
  - Turbidity/TSS (Event-driven): $0.4 - 0.8$ (Walling & Webb 1982)

**5. Rappbode headwater catchment, Germany — high-frequency NO₃, DOC, Q, EC**
- 15-minute water quality data (Jan 2018 - Aug 2023) available on CUAHSI HydroShare (DOI: 10.4211/hs.9be43573ba754ec1b3650ce233fc99de).
- Expected NO₃ $\beta \approx 1.0$.

**6. GRDC global river discharge archive**
- Mean daily/monthly discharge data globally.
- Recommendations: Rhine at Cologne (published $H \approx 0.72$, $\beta \approx 0.44$), Mississippi at St. Louis, Danube.

**7. CAMELS-Chem (US, 516 catchments, Cl, NO₃, DOC, 1980–2018)**
- Grab-sample chemistry, useful for bulk decadal Lomb-Scargle testing.

## Tier 2 — Published multifractal K(2) values

- **Pandey, Lovejoy & Schertzer (1998)**: Analyzed 19 US rivers (Mississippi, Ohio, Missouri). Found universal multifractal parameters $C_1 \approx 0.1-0.15$ and $\alpha_m \approx 1.5-1.8$.
- Calculate $K(2) = C_1 \cdot \frac{2^{\alpha_m} - 2}{\alpha_m - 1}$. For these rivers, $K(2) \approx 0.15-0.25$.
- Available via USGS NWIS.

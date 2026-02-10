# Spectral Analysis of Gravity Waves in The Presence of Observational Gaps 

## Description
"GapWaveSpectra" is a robust and versatile Python library developed for in-depth analysis and estimation of gravity wave (GW) spectral properties, particularly in the presence of observational gaps - a common issue in empirical atmospheric measurements. This library is an essential tool for researchers and meteorologists interested in understanding the impact of GWs on atmospheric dynamics, weather, and climate patterns.

One of the key spectral properties it helps estimate is the spectral power-law exponent (β), which indicates how GW energy varies with frequency over specific GW scales. To achieve this, the library incorporates three well-known estimation methods: Fast Fourier Transform (FFT), Generalised Lomb-Scargle periodogram (GLS), and Haar Structure Function (HSF).

A distinguishing feature of GapWaveSpectra is its capability to critically evaluate the effectiveness of these three methods using time-series of synthetic observational data. By delivering a thorough analysis of these methods, GapWaveSpectra aims to serve as a comprehensive guide for researchers to choose the best suited spectral estimation method for precise calculations of β from gapped observational datasets.

## Usage
An example of how to use this library to simulate time series with similar characterestics to observed gravity waves, add gaps to the simulation, produce their spectra and estimate their exponent (slope) of spectral power law.
```
import numpy as np
import matplotlib.pylab as plt
from sim_GW import PowerLaw
complete_time,complete_data,input_freq = PowerLaw().create_TS(time_max=6*60,sampling_rate=1/5,
                    slope=-3,TS_rand_seed=30,freq='random',nwaves=35,period_max=6*60,period_min=10,longer=False,same_amp=False)
gapped_time,gapped_data = PowerLaw().add_gaps(complete_time,complete_data,
                            rand_seed=30,percentage=50)
df = PowerLaw().calculate_PSD(gapped_time,gapped_data,input_freq,process=False,
                   time_max=6*60,slope=-3,percentage=50,plot_psd=True)
```

## Support
The corresponding author is reachable for comments and questions at mossad@iap-kborn.de .

## Authors and acknowledgment
This library was written and edited by Mohamed Mossad, a PhD candidate at the Leibniz Insitute for Atmospheric Physics (IAP) and Rostock University, under the supervision of Irina Strelnikova, Robin Wing and Gerd Baumgarten.
## License
See the LICENSE file for license rights and limitations (MIT).

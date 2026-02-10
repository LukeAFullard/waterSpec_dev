# -*- coding: utf-8 -*-
"""
Created on Thu Jul  6 10:26:54 2023

@author: Mohamed Mossad (IAP)
"""

import numpy as np
import matplotlib.pylab as plt
from astroML.time_series import generate_power_law
from PyAstronomy.pyTiming import pyPeriod
import pandas as pd
import random
from math import floor
from scipy import signal, interpolate, fft
from haar_structure_function import do_haar
from lmfit import Parameters
from scipy.optimize import minimize
from lmfit.models import PowerLawModel
from PyAstronomy import pyasl


class PowerLaw:
    """A class for time series generation whose spectra follow power laws, it does signal processing,
    and related utilities."""
    
    def __init__(self):
        """Initialize the class with parameters for controlling plotting of outputs."""

      
    def create_TS(self, time_max, sampling_rate, slope, TS_rand_seed, freq='random', nwaves=35,
                  period_max = 6*60, period_min = 10, longer=False, same_amp=True):
        """
        Function to create a time series data with the option to use random, Timmer or Harmonics simulation.
        Parameters:
           time_max: int
               Total time span of the time series
           sampling_rate: float
               How many data points are sampled per step 
           slope: float, default: None
               The slope (-beta) of the loglog power law spectrum. Useful only in simulation
           TS_rand_seed: int
               Seed for random generation of frequencies and phases
           freq: string, default='random'
               Simulation type 'random' or 'Timmer' or 'harmonics'
           nwaves: int, default= 35
               Number of sinusoids to sum for the power law
           period_max: float, default = 6 h
               Maximum of periods simulated
           period_min: float, default = 10 min
               Minimum of periods simulated
           longer: bool, default=False
               Whether to add longer periods to the simulation
           same_amp: bool, default=True
               Whether the amplitudes of waves with longer periods have 
               the same amplitude as the waves with the lowest frequency

         Output:
           time_even: array
               Evenly sampled time array                    
           complete_data: array
               Evenly sampled amplitude array
           frequencies: array
               Simulated frequencies of each of the nwaves sinusoids.
               
        """

        # Definitions and initializations
        beta = -1*slope
        dt = 1/sampling_rate
        N = int(time_max * sampling_rate)
        time_even = np.linspace(0,  time_max, N, endpoint=False)
        if nwaves is None:
            nwaves = int(N / 2)

        if freq=='random':
            np.random.seed(TS_rand_seed)
            while True:
                 frequencies =  np.random.uniform(1/period_max, 1/period_min,nwaves) # best
                 if (max(1/frequencies)>(period_max-0.01) and min(1/frequencies)<(period_min+0.01)):
                    break          
        
            if longer==True:
                extra_waves=3
                phase = np.random.uniform(-np.pi,np.pi,nwaves+extra_waves)
                extra_periods=np.array([period_max+ 6*60,period_max+ 4*60,period_max+ 2*60])
                extra_freq=1/extra_periods
                if same_amp==True:
                    extra_Amp = np.ones((len(extra_freq)))*min(frequencies)**(slope/2)
                else:
                    extra_Amp =extra_freq**(slope/2)
                Amp = frequencies**(slope/2)
                Amp = np.concatenate((extra_Amp, Amp))
                frequencies = np.concatenate((extra_freq, frequencies))
                Amp = Amp/np.mean(Amp)    
                complete_data = np.zeros(N)
                for i in range(nwaves+extra_waves):
                    larger_signal = Amp[i]*np.sin(2*np.pi*frequencies[i]*time_even+phase[i])
                    complete_data += larger_signal
            else:
                phase = np.random.uniform(-np.pi,np.pi,nwaves)
                Amp = frequencies**(slope/2)
                Amp = Amp/np.mean(Amp)
                complete_data = np.zeros(N)
                for i in range(nwaves):
                   sinusoid = Amp[i]*np.sin(2*np.pi*frequencies[i]*time_even + phase[i])
                   complete_data += sinusoid
           
        elif freq=='Timmer': #according to Timmer and Koenig's 1995 simulation paper
            factor = 0.1
            complete_data = factor * generate_power_law(N, dt, beta, random_state= TS_rand_seed)
            f_0= 1/time_max   
            nwaves = int(N/2)    
            frequencies=np.array([f_0*(i+1) for i in range(nwaves)])
        elif freq=="harmonics":          #harmonics
            f_0= 1/time_max   
            nwaves = int(N/2)    
            phase = np.random.uniform(0,2*np.pi,nwaves)
            complete_data = np.zeros(N)
            for i in range(nwaves):
              complete_data += ((f_0*(i+1))**(slope/2))*np.sin(2*np.pi*(f_0*(i+1))*time_even +phase[i])     

        return time_even, complete_data, frequencies

    def add_gaps(self, time_even, complete_data, rand_seed, percentage):
        """
        Function to add gaps to the data. Returns the subset of data after removing gaps.
        Parameters:
        time_even : ndarray
            Evenly sampled time array
        complete_data : ndarray
            Evenly sampled amplitude array
        rand_seed: int
            Seed for random generation of gaps
        input_freq : ndarray, default: None
            The simulated frequency array of sinusoids
        dt: float, default: 5 min
            The time step of measurement
        time_max: int, default: None
            The last point in the time array
        percentage : int, default: None
            The percentage of gaps in data. 
        
        Output:    
            timesteps_subset: array
                Gapped time array
            signal_subsetL array
                Gapped amplitude array
   
        """

        gap_elements = ((len(complete_data)*percentage)/100)   
        # add gaps without including edges
        random.seed(rand_seed)
        gaps = random.sample(list(np.arange(1,len(complete_data[:-1]))),floor(gap_elements))    
        timesteps_subset = np.delete(time_even,gaps)
        signal_subset = np.delete(complete_data,gaps)
        return timesteps_subset, signal_subset


    def interpolate_TS(self, time_even, gapped_time, gapped_data):
        """
        Method to apply interpolation on the time series data. Needed for FFT!
        """

        data_interp = interpolate.interp1d(gapped_time, gapped_data, kind='linear',fill_value="extrapolate")
        data_even= data_interp(time_even)  # use interpolation function returned by `interp1d`
        return data_even

    def preprocess(self, time_array, sig, process=False):
        """
        Method to preprocess a signal. Removes NaNs, detrends and prewhitens the time series.
        """

        sig = sig[~np.isnan(sig)]
        time_array = time_array[~np.isnan(time_array)]
        if process==True:
            data_p=np.zeros_like(sig)
            for i in range(len(sig)-1):  #prewhitening by first differencing
                 data_p[i] =sig[i+1]-sig[i]
            sig = np.copy(data_p)     
        sig = sig.tolist()
        sig = signal.detrend(sig,type='constant') #remove mean

        return time_array, sig

    def do_postprocess(self, f, psd_, dx, k, process=False):
        """
        Method to postprocess a signal. Applies Hanning smoothing and postcoloring of the spectra.
        """
        if process == False:
            psd_pd = np.copy(psd_)
        else:
            # Do hann smoothing
            psd_h = np.copy(psd_)
            if k < 2:
                psd_h = pyasl.smooth(psd_, 11, 'hanning')

            # Do postcoloring
            psd_pd = np.copy(psd_h)
            for i in range(1, len(psd_)):
                psd_pd[i] = psd_h[i] / (2 * (1 - np.cos(2 * np.pi * f[i] * dx)))

        return psd_pd

    def calculate_PSD(self,gapped_time,gapped_data,input_freq=None,
                dt=5,time_max=None,slope=None,percentage=None,process=False,plot_psd=False):
         """
        Compute the power spectral density of time series.

        Parameters:
        gapped_time : ndarray
            The time array of data
        gapped_data : ndarray
            The amplitude array of data
        input_freq : ndarray, default: None
            The simulated frequency array of sinusoids
        dt: float, default: 5 min
            The time step of measurement
        time_max: int, default: None
            The last point in the time array
        slope: float, default: None
            The slope (-beta) of the loglog power law spectrum. Useful only in simulation.
        percentage : int, default: None
            The percentage of gaps in data. 
        process : bool, default: False
            Whether to apply prewhitening of time series and postdarkening of spectrum. 
        plot_psd : bool, default: False
            Whether to plot both time series and spectrum. 
          
        Output:    
            df: dataframe
                The estimated slopes, frequencies and corresponding PSD by all three methods.    
        """
         if time_max==None:
             time_max=gapped_time[-1]
        
         sampling_rate=1/dt
         N=int(sampling_rate*time_max) 
         delta_t = dt
         time_even = np.arange(0, time_max, delta_t)
         data_even = self.interpolate_TS(time_even,gapped_time,gapped_data)
         gapped_time, gapped_data = self.preprocess(gapped_time, gapped_data,process)
         time_even, data_even = self.preprocess(time_even, data_even,process)
         sampling_freq = np.arange(1/(dt*N),1/(2*dt),1/(dt*N))
         fNy = 1/(2*dt)   #= 10min 
         sample_freq = fft.rfftfreq(1*len(time_even), d=delta_t)[1:-1]
         sig_fft = fft.rfft(data_even,1*len(time_even))[1:-1]
         sig_fft *= np.conj(sig_fft)
         clp = pyPeriod.Gls((gapped_time, gapped_data),freq=sampling_freq)
         Hfluc = do_haar(np.array(gapped_data),gapped_time,scales=1/sampling_freq)
         Hfluc_scaled = Hfluc**(2) /sampling_freq
         psd_scaled_ =[2*dt* sig_fft.real / N ,dt*N*np.abs(clp._a**2 + clp._b**2)/2,2*dt*Hfluc_scaled / N]
         psd_freq =[sample_freq,sampling_freq,sampling_freq]
         psd_scaled=[]
         for i in range(3):

             psd_scaled.append(self.do_postprocess(psd_freq[i],psd_scaled_[i],dt,i,process))

         methods = ['FFT','GLS','HSF']
         slopes = np.zeros(3)
         intercepts = np.zeros(3)
         colors = ['green','blue','red']
         colors_fit = ['darkgreen','royalblue','lightcoral']
         markers= ['s','.','P'] 
         for i in range(3):
             slopes[i],intercepts[i] = self.get_slope(psd_freq[i], psd_scaled[i],i,method='mle')

         if plot_psd:     
             fig, axs = plt.subplots(2, 1, layout='constrained',gridspec_kw={'height_ratios': [1, 2]}, sharex=False, sharey=False)
             for i in range(3):

               axs[1].loglog(psd_freq[i], psd_scaled[i], '-',c=colors[i],marker=markers[i],label =r'$\beta$[%s] = %.2g'%(methods[i],-1*slopes[i]) )
               y_fit = np.exp(slopes[i]*np.log(psd_freq[i]) + np.log(intercepts[i])) # calculate the fitted values of y                 
               axs[1].loglog(psd_freq[i],y_fit,':',c=colors_fit[i])
             axs[1].loglog(input_freq,input_freq**(slope),'*',c='black',label=r'Sim. $\beta$ = %.1g'%(-1*slope))
             axs[1].vlines(1/time_max, 0, 1, 'k',transform=axs[1].get_xaxis_transform(),linestyles= 'dotted',label ='1/T')
             axs[1].vlines(fNy, 0, 1, 'darkviolet', transform=axs[1].get_xaxis_transform(),linestyles= 'dotted',label ='Nyquist limit')    
             if percentage==0:
                 axs[0].plot(gapped_time/60, gapped_data, 'b-o',linewidth=2, markersize=5,label= 'non-gapped data')
             else:
                 axs[0].plot(time_even/60, data_even, ':or',linewidth=2, markersize=5,label='interpolated')
                 axs[0].plot(gapped_time/60, gapped_data, '-sb',linewidth=2, markersize=5,label='gapped')
             axs[0].legend(loc='lower right',fontsize="10")
             axs[1].legend(loc='upper right', fontsize="8")
             axs[0].set_xlabel("Time [hour]", size=12)
             axs[0].set_ylabel("Amplitude [K]",size=12)    
             axs[1].set_xlabel("Frequency [cycle/min]", size=12)
             axs[1].set_ylabel("PSD [$K^2$ /(cycle/min)]", size=12)  
             axs[1].set_ylim(10**-1,10**8)
             axs[0].set_ylim(-25,25)
             axs[0].set_xlim(-0.5,8)
             fig.savefig('Figures/power law/time_spectra/TS_%i_%i.pdf' %(slope,percentage), format='pdf',transparent=True)
             plt.show()
                    
         df = pd.DataFrame({"slope_FFT": [slopes[0]],"slope_GLS": [slopes[1]],"slope_HSF": [slopes[2]],'FFT freq':[psd_freq[0]],'GLS freq':[psd_freq[1]],'HSF freq':[psd_freq[2]],
                            'FFT psd':[psd_scaled[0]],'GLS psd':[psd_scaled[1]],'HSF psd':[psd_scaled[2]]},index=[[percentage],[slope]])

         return df
    def func_powerlaw(self,x, params): 
     """
       power law model to fit
     """
     slope_ = params[0]
     coefficient = params[1]
     return  (x**slope_)*coefficient
    def neglnlike(self,params, x, y): 
     """
         negative log likelihood function to minimize
     """
     model = self.func_powerlaw(x, params)
     output = np.sum(np.log(model) + y/model)
     if not np.isfinite(output):
         return 1.0e30
     return output
    def get_slope(self,freq, ps,k=None,method='mle'):
          """
            Compute the slope of the spectral density curve.
    
            Parameters:
            freq : ndarray
                The frequencies at which the spectral density is calculated
            ps : ndarray
                The spectral density values
            k : int, optional
                FFT: k=0, GLS: k=1, HSF: k=2
            method : str, optional
                The method to be used for computing the slope ('mle', 'lsq', or 'non-lin')
            Output:    
                slope: float
                    Estimated slope (-beta) from fitted spectra.
                intercept: float
                    Estimated intercept from fitted spectra.
          """
          freq = freq[~np.isnan(ps)]
          ps = ps[~np.isnan(ps)]
          if method=='mle': #maximum likelihood estimation of slope
              params = Parameters()
              params.add('slope', value=1)
              params.add('coefficient', value=1)
              res = minimize(self.neglnlike, params, args=(freq, ps),
                           method='Nelder-Mead')
              slope= res.x[0]
              intercept= res.x[1]
              return slope , intercept 
          elif method == 'lsq':    # linear least squares fitting
              logA = np.log(freq) 
              logB = np.log(ps)
             
              if k==2:
                      slope, intercept = np.polyfit(logA, logB, 1)
              else:
                      slope, intercept = np.polyfit(logA, logB, 1,w=np.sqrt(ps)) #weighted
            
              return slope , intercept
          elif method == 'non-lin': # (non-linear) power law least squares fitting
              mod = PowerLawModel()
              pars = mod.guess(ps, x=freq) #Estimate initial model parameter values from data.
              out = mod.fit( ps, pars, x=freq) #Fit the model to the data using the supplied Parameters
              slope= out.best_values['exponent']
              intercept= out.best_values['amplitude']
              return slope , intercept 
       

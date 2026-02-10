# -*- coding: utf-8 -*-
"""
Created on Sun Jul  9 22:03:31 2023

@author: Mohamed Mossad

The current Python implementation of the HSF was adapted from the original R code 
authored by Hébert et al. (2021) 

"""

import numpy as np
import pandas as pd
from scipy.special import gammaln
from sequence_gen import seq

import warnings
warnings.filterwarnings("ignore")

def mean_in_range(t0,j, sequence, scali2, dat):
    """    
        Determines the mean of fluctuations within a specific interval.
    """

    # Obtain indices of t0 array values within the defined fluctuation interval
    indices = np.where((t0 >= sequence[j]) & (t0 < (sequence[j] + scali2)))
   
    return np.mean(dat[indices])

def sdsmpl(ser):
    """
    Returns the standard deviation of the original gaussian distribution of Haar fluctuations
    """
    ser = ser[~np.isnan(ser)]
    s = np.std(ser)
    n = len(ser)
    if n < 101:
        return s * np.sqrt((n - 1) / 2) * np.exp(gammaln((n - 1) / 2) - gammaln(n / 2))
    else:
        return s
  
def do_haar(dat,t0, scales, overlap=0, return_flucs=False):
    """

    Parameters
    ----------
    dat : array
        Gapped or non-gapped amplitude array 
    t0 : array
        Gapped or non-gapped amplitude array 
    scales : array
        The sampled period (1/frequency) array to calculate the structure function for
    overlap : int, optional
        Overlap of intervals. The default is 0.
    return_flucs : bool, optional
        Whether to return all fluctuations. The default is False.

    Returns
    -------
    hfluc : array
        Fluctuations from Haar Structure Function.

    """
    # Setting up the Haar fluctuations vector Hfluc
    hfluc = np.zeros(len(scales))
    
    if return_flucs:
        all_flucs = []
      
    # This loop calculates the mean qth order fluctuation for each scale within the scales vector
    largestscale = t0[-1]
    
    for i in range(len(scales)):
        # Defining some constant values
        scali = scales[i]
        scali2 = scali / 2
        maxfluc = int(np.floor((largestscale - t0[0] + 1) / scali))
    
        # seqa refers to the start point of each fluctuation interval.
        #If overlap is 0, intervals do not overlap.
        seqa = seq(t0[0], largestscale + (1-overlap)*scali , (1-overlap)*scali)
    
        # seqb denotes the midpoint of each fluctuation interval.
        seqb = seq(t0[0]+scali2, largestscale + (1-overlap)*scali, (1-overlap)*scali)
    
        # The mean value for each halves of the intervals is calculated below.
        # Compute the means for the first sequence, seqa, and store them in mfluca
        mfluca = [mean_in_range(t0,j, seqa, scali2, dat) for j in range(len(seqa))]        
        # Compute the means for the second sequence, seqb, and store them in mflucb
        mflucb = [mean_in_range(t0,j, seqb, scali2, dat) for j in range(len(seqb))]    
        mflucb.append(np.mean(dat[np.where((t0 >= seqb[len(seqb) - 1]) & (t0 < (seqb[len(seqb) - 1] + scali2)))]))
        # Truncate mfluca and mflucb to the maximum fluctuation, maxfluc
        mfluca = mfluca[:maxfluc]
        mflucb = mflucb[:maxfluc]
    
        
        # Calculating the Haar fluctuations 
        flucs= pd.DataFrame(mflucb)- pd.DataFrame(mfluca)
    
        # The fluctuations are scaled by the canonical calibration factor so that when the slopes
        # are positive, the fluctuations are roughly equivalent to the difference fluctuations, and
        # match the anomaly fluctuation when the slopes are negative
        flucs=2*flucs
        
        if return_flucs:
            all_flucs.append(flucs)
        if np.all(flucs.isna()):
            hfluc[i] = np.nan
        else:
              flucs = flucs[~flucs.isna()]
             
              #mean fluctuations
              hfluc[i] = (np.sqrt(2 / np.pi) * sdsmpl(np.concatenate((flucs, -flucs))))
    
        hf = hfluc[i]
        if np.isnan(hf):
              pass
        elif hf < 1e-7:
              hfluc[i] = np.nan
    
       
    return hfluc

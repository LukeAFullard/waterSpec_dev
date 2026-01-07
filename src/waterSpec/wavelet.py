from __future__ import annotations

import warnings
from typing import Dict, Optional, Tuple, List, Union
from dataclasses import dataclass

import numpy as np
import pandas as pd
import pyleoclim as pyleo
from scipy.interpolate import interp1d
from scipy.optimize import curve_fit
import piecewise_regression

# Try importing pymultifracs, warn if not available
try:
    import pymultifracs.wavelet_analysis as wa
    import pymultifracs.mfa as mfa
    from pymultifracs.utils import MFractalVar
    HAS_PYMULTIFRACS = True
except ImportError:
    HAS_PYMULTIFRACS = False


@dataclass
class WaveletResult:
    """
    Data class to hold the results of wavelet analysis.
    """
    time: np.ndarray
    scales: np.ndarray
    frequencies: np.ndarray
    power: np.ndarray  # Power spectrum (time x scale)
    coi: np.ndarray # Cone of influence
    global_power: np.ndarray # Time-averaged power
    method: str


@dataclass
class SlopeFitResult:
    """
    Data class to hold the results of slope fitting.
    """
    beta: float
    beta_err: float
    intercept: float
    residuals: np.ndarray
    r_squared: float
    model_type: str  # 'linear' or 'segmented'
    breakpoints: Optional[List[float]] = None
    segment_slopes: Optional[List[float]] = None


def compute_wwz(
    time: np.ndarray,
    data: np.ndarray,
    freq_method: str = 'log',
    n_scales: int = 50,
    freq_min: Optional[float] = None,
    freq_max: Optional[float] = None,
    decay_constant: float = 1.0 / (8 * np.pi**2),
) -> WaveletResult:
    """
    Computes the Weighted Wavelet Z-transform (WWZ).

    Args:
        time (np.ndarray): Time array.
        data (np.ndarray): Data array.
        freq_method (str): Method to generate frequency vector ('log', 'linear').
        n_scales (int): Number of scales/frequencies.
        freq_min (float, optional): Minimum frequency.
        freq_max (float, optional): Maximum frequency.
        decay_constant (float): Decay constant for the Morlet wavelet (c parameter).
            Default corresponds to f0=omega0/(2pi) approx 0.8.
            Foster (1996) uses c = 1 / (8 * pi^2) approx 0.0126.

    Returns:
        WaveletResult: The result of the WWZ analysis.
    """
    # Create a pyleoclim Series object (wrapper for convenience)
    # pyleoclim expects time to be increasing

    # Define frequency vector
    if freq_max is None:
        dt = np.median(np.diff(time))
        freq_max = 1 / (2 * dt)

    if freq_min is None:
        # Record length / 4
        T = time[-1] - time[0]
        freq_min = 4 / T

    # pyleoclim wwz implementation
    # It returns a namedtuple or similar object
    # pyleoclim freq_vector_log args are fmin, fmax, nf
    freq_kwargs = {}
    if freq_method == 'log':
        freq_kwargs = {'fmin': freq_min, 'fmax': freq_max, 'nf': n_scales}

    res = pyleo.utils.wavelet.wwz(
        data,
        time,
        freq_method=freq_method,
        freq_kwargs=freq_kwargs,
        c=decay_constant
    )

    # res.amplitude is Weighted Wavelet Amplitude. Power is usually amplitude^2.
    # In recent versions of pyleoclim, the field is named 'amplitude' (not 'wwa')
    # and it returns a namedtuple with fields:
    # ('amplitude', 'phase', 'coi', 'freq', 'time', 'Neffs', 'coeff', 'scale')

    # Check if 'wwa' or 'amplitude' exists
    if hasattr(res, 'wwa'):
        amplitude = res.wwa
    else:
        amplitude = res.amplitude

    # Normalize power by frequency to get PSD-like density.
    # Wavelet power roughly scales with frequency for white noise if unnormalized.
    # Dividing by frequency corrects the "blue" bias (slope +1) observed for white noise.
    # We use res.freq (or equivalent 1/scale) for normalization.

    # Ensure correct broadcasting for normalization
    freq_grid = res.freq
    if amplitude.shape[1] == len(freq_grid):
        # (n_time, n_freq)
        power = (amplitude ** 2) / freq_grid[np.newaxis, :]
    elif amplitude.shape[0] == len(freq_grid):
        # (n_freq, n_time)
        power = (amplitude ** 2) / freq_grid[:, np.newaxis]
    else:
        # Fallback
        power = amplitude ** 2

    # Mask out values inside the Cone of Influence (COI).
    # res.coi is likely in Period units (based on pyleoclim plots).
    # Valid region: Scale (Period) < COI.
    # Note: Some definitions say COI is the valid region, others say it's the invalid.
    # Pyleoclim plots COI as a line. Usually, the region "below" the cone (longer periods near edges) is invalid.
    # So if Scale > COI, it is invalid.
    # However, res.coi at edges is usually small?
    # Inspecting output:
    # COI sample: [2e-5, 2.8, 5.7, ...] -> Starts small at edge (t=0), increases.
    # So valid periods are those SMALLER than COI?
    # Wait, usually COI is the boundary.
    # Standard wavelet plot: x=time, y=period. Cone shape.
    # Inside the cone (center of plot): Valid.
    # Outside the cone (corners): Invalid.
    # If COI is the limit of valid period:
    # At edges (t=0), valid period range is small (0 to ~0).
    # In center, valid period range is large (0 to T/sqrt(2)).
    # So Valid if Period < COI[t].

    # Verify shape of power.
    # Power is likely (n_time, n_freq).
    # Frequencies correspond to periods = 1/freq.

    periods = 1.0 / res.freq # Shape (n_freq,)

    # We need to mask power where period > coi[t]
    # power[t, f] is invalid if periods[f] > coi[t]

    # Ensure correct orientation
    if power.shape[1] == len(res.freq):
         # (n_time, n_freq)
         n_time, n_freq = power.shape
         # Create mask
         # COI is (n_time,)
         # Mask should be True where valid.
         # periods (n_freq,)

         # Broadcast comparison
         # valid if periods[f] < coi[t]
         # (1, n_freq) < (n_time, 1)

         valid_mask = periods[np.newaxis, :] < res.coi[:, np.newaxis]

         # Apply mask (set invalid to NaN)
         power_masked = power.copy()
         power_masked[~valid_mask] = np.nan

         # Aggregate
         global_power = np.nanmedian(power_masked, axis=0)

    elif power.shape[0] == len(res.freq):
         # (n_freq, n_time)
         # valid if periods[f] < coi[t]
         # (n_freq, 1) < (1, n_time)

         valid_mask = periods[:, np.newaxis] < res.coi[np.newaxis, :]

         power_masked = power.copy()
         power_masked[~valid_mask] = np.nan

         global_power = np.nanmedian(power_masked, axis=1)
    else:
         # Fallback without masking
         global_power = np.nanmedian(power, axis=0)

    return WaveletResult(
        time=res.time, # Evenly spaced time grid returned by WWZ
        scales=1.0/res.freq,
        frequencies=res.freq,
        power=power, # (n_time, n_freq)
        coi=res.coi,
        global_power=global_power,
        method="WWZ"
    )


def fit_spectral_slope(
    frequencies: np.ndarray,
    power: np.ndarray,
    freq_range: Optional[Tuple[float, float]] = None
) -> SlopeFitResult:
    """
    Fits a power-law slope (1/f^beta) to the power spectrum.
    Fits linear model to log(power) vs log(frequency).
    Slope = -beta.
    """
    if freq_range:
        mask = (frequencies >= freq_range[0]) & (frequencies <= freq_range[1])
        f_fit = frequencies[mask]
        p_fit = power[mask]
    else:
        f_fit = frequencies
        p_fit = power

    # Remove invalid values
    valid = (f_fit > 0) & (p_fit > 0) & np.isfinite(f_fit) & np.isfinite(p_fit)
    f_fit = f_fit[valid]
    p_fit = p_fit[valid]

    if len(f_fit) < 3:
        return SlopeFitResult(np.nan, np.nan, np.nan, np.array([]), np.nan, 'insufficient_data')

    log_f = np.log10(f_fit)
    log_p = np.log10(p_fit)

    # Linear fit
    # model: log_p = -beta * log_f + c
    def linear_model(x, beta, c):
        return -beta * x + c

    popt, pcov = curve_fit(linear_model, log_f, log_p)
    beta = popt[0]
    intercept = popt[1]
    beta_err = np.sqrt(np.diag(pcov))[0]

    residuals = log_p - linear_model(log_f, *popt)
    ss_res = np.sum(residuals**2)
    ss_tot = np.sum((log_p - np.mean(log_p))**2)
    r_squared = 1 - (ss_res / ss_tot)

    return SlopeFitResult(
        beta=beta,
        beta_err=beta_err,
        intercept=intercept,
        residuals=residuals,
        r_squared=r_squared,
        model_type='linear'
    )


def fit_segmented_slope(
    frequencies: np.ndarray,
    power: np.ndarray,
    n_breakpoints: int = 1
) -> SlopeFitResult:
    """
    Fits a segmented linear model to log-log spectrum.
    """
    # Use piecewise_regression
    valid = (frequencies > 0) & (power > 0) & np.isfinite(frequencies) & np.isfinite(power)
    f_fit = frequencies[valid]
    p_fit = power[valid]

    log_f = np.log10(f_fit)
    log_p = np.log10(p_fit)

    if len(log_f) < 5:
         return SlopeFitResult(np.nan, np.nan, np.nan, np.array([]), np.nan, 'insufficient_data')

    try:
        pw_fit = piecewise_regression.Fit(log_f, log_p, n_breakpoints=n_breakpoints)
        # pw_fit.summary() # prints summary

        # Extract results
        results = pw_fit.get_results()
        estimates = results['estimates'] # dict with keys like 'const', 'alpha1', 'beta1', 'breakpoint1'

        # piecewise_regression typically returns 'alpha1', 'alpha2' etc as the slopes of segments
        # and 'beta1' as differences (or vice versa depending on implementation, but 'alpha' seems to be slope in output).
        # Based on debug_piecewise.py output:
        # alpha1: 2.0 (Slope 1)
        # alpha2: 0.5 (Slope 2)
        # beta1: -1.5 (Difference? 0.5 - 2.0 = -1.5)

        # So 'alphaX' are the slopes of segment X.

        slopes = []
        breakpoints = []

        # Collect breakpoints
        for i in range(1, n_breakpoints + 1):
             bp_key = f'breakpoint{i}'
             if bp_key in estimates:
                 breakpoints.append(estimates[bp_key]['estimate'])

        # Collect slopes (alpha1, alpha2, ...)
        # We assume n_breakpoints implies n_breakpoints + 1 segments.
        for i in range(1, n_breakpoints + 2):
            slope_key = f'alpha{i}'
            if slope_key in estimates:
                 # slope is d(logP)/d(logf).
                 # beta (spectral exponent) = -slope.
                 slope_val = estimates[slope_key]['estimate']
                 slopes.append(-slope_val)
            else:
                 # Fallback if key missing
                 slopes.append(np.nan)

        return SlopeFitResult(
            beta=slopes[0], # Report first slope as primary? Or average?
            beta_err=np.nan, # Complex to propagate
            intercept=estimates['const']['estimate'],
            residuals=np.array([]), # TODO: extract residuals
            r_squared=np.nan, # TODO
            model_type='segmented',
            breakpoints=breakpoints,
            segment_slopes=slopes
        )

    except Exception as e:
        warnings.warn(f"Segmented fit failed: {e}")
        return fit_spectral_slope(frequencies, power)


def multifractal_analysis_pipeline(
    time: np.ndarray,
    data: np.ndarray,
    interpolate: bool = True,
    dt: Optional[float] = None
) -> Dict:
    """
    Performs multifractal analysis using Wavelet Leaders.

    Args:
        time: Timestamps.
        data: Values.
        interpolate: If True, interpolates irregular data to a regular grid.
        dt: Time step for interpolation. If None, uses median diff.

    Returns:
        Dict containing multifractal spectrum and diagnostics.
    """
    if not HAS_PYMULTIFRACS:
        raise ImportError("pymultifracs not installed.")

    # 1. Interpolate if necessary
    if interpolate:
        if dt is None:
            dt = np.median(np.diff(time))

        t_min, t_max = time[0], time[-1]
        t_reg = np.arange(t_min, t_max, dt)

        # Linear interpolation
        f_interp = interp1d(time, data, kind='linear', fill_value="extrapolate")
        data_reg = f_interp(t_reg)
    else:
        data_reg = data

    # 2. Wavelet Analysis (DWT)
    # pymultifracs.wavelet_analysis.wavelet_analysis
    # We need to choose j2 (max scale)

    # "Warn if record length < ~200"
    if len(data_reg) < 200:
        warnings.warn("Time series length < 200, multifractal analysis may be unreliable.")

    # 3. Wavelet Leaders
    # The wavelet_analysis function returns a WaveletDec object which has wavelet leaders?
    # Actually wa.wavelet_analysis returns a WaveletDec.
    # We need to compute leaders.

    # Placeholder for multifractal analysis
    results = {
        "interpolated": interpolate,
        "n_points": len(data_reg)
    }

    try:
        # Compute coefficients
        wt = wa.wavelet_analysis(data_reg, wt_name='db3')

        # Compute leaders
        # WaveletDec object 'wt' should have get_leaders method
        leaders = wt.get_leaders(p_exp=np.inf)

        # Run MFA on leaders
        # scaling_ranges: list of (j1, j2)
        # We need to pick a scaling range.
        j2 = leaders.j2_eff()
        j1 = 1 # Start from 1
        scaling_ranges = [(j1, j2 - 1)] # Avoid boundaries

        mf_result = mfa.mfa(leaders, scaling_ranges=scaling_ranges, n_cumul=2)

        # mf_result is MFractalVar(structure, cumulants, spectrum)

        results['cumulants'] = mf_result.cumulants
        results['spectrum'] = mf_result.spectrum
        results['structure'] = mf_result.structure
        results['scaling_range'] = scaling_ranges[0]

    except Exception as e:
        results['error'] = str(e)
        warnings.warn(f"Multifractal analysis failed: {e}")

    return results

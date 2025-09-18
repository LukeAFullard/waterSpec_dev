"""
waterSpec: A Python package for spectral analysis of environmental time series.
"""

__version__ = "0.0.1"

from .data_loader import load_data
from .preprocessor import detrend, normalize, log_transform
from .spectral_analyzer import calculate_periodogram
from .fitter import fit_spectrum
from .interpreter import interpret_beta

__all__ = [
    'load_data',
    'detrend',
    'normalize',
    'log_transform',
    'calculate_periodogram',
    'fit_spectrum',
    'interpret_beta',
]

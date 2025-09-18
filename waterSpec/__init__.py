"""
waterSpec: A Python package for spectral analysis of environmental time series.
"""

__version__ = "0.0.1"

from .data_loader import load_data
from .preprocessor import detrend, normalize, log_transform
from .spectral_analyzer import calculate_periodogram
from .fitter import fit_spectrum, fit_spectrum_with_bootstrap
from .interpreter import interpret_beta
from .plotting import plot_spectrum
from .workflow import run_analysis

__all__ = [
    'load_data',
    'detrend',
    'normalize',
    'log_transform',
    'calculate_periodogram',
    'fit_spectrum',
    'fit_spectrum_with_bootstrap',
    'interpret_beta',
    'plot_spectrum',
    'run_analysis',
]

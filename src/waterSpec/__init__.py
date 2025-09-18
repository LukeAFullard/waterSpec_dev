"""
waterSpec: A Python package for spectral analysis of environmental time series.
"""

__version__ = "0.0.1"

from .data_loader import load_data
from .preprocessor import preprocess_data
from .spectral_analyzer import calculate_periodogram
from .fitter import fit_spectrum, fit_spectrum_with_bootstrap, fit_segmented_spectrum
from .interpreter import interpret_results
from .plotting import plot_spectrum
from .workflow import run_analysis

__all__ = [
    "load_data",
    "preprocess_data",
    "calculate_periodogram",
    "fit_spectrum",
    "fit_spectrum_with_bootstrap",
    "fit_segmented_spectrum",
    "interpret_results",
    "plot_spectrum",
    "run_analysis",
]

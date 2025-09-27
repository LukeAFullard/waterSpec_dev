"""
waterSpec: A Python package for spectral analysis of environmental time series.
"""

__version__ = "0.0.1"

from .analysis import Analysis
from .data_loader import load_data
from .fitter import fit_segmented_spectrum, fit_standard_model
from .frequency_generator import generate_frequency_grid
from .interpreter import interpret_results
from .plotting import plot_spectrum
from .preprocessor import preprocess_data
from .spectral_analyzer import calculate_periodogram

__all__ = [
    "Analysis",
    "load_data",
    "preprocess_data",
    "calculate_periodogram",
    "generate_frequency_grid",
    "fit_standard_model",
    "fit_segmented_spectrum",
    "interpret_results",
    "plot_spectrum",
]

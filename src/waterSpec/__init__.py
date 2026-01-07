# src/waterSpec/__init__.py
import logging

# Configure a NullHandler for the library's root logger to avoid
# "No handler found" warnings and let downstream applications configure logging.
logging.getLogger("waterSpec").addHandler(logging.NullHandler())


"""
WaterSpec: Spectral analysis toolkit for hydrological and environmental time series.

This package estimates spectral slopes (β) using Lomb–Scargle, Weighted Wavelet Z-transform (WWZ),
and segmented regressions. It supports confidence intervals via bootstrap resampling,
and provides interpretive diagnostics including PSRESP.
"""

from importlib import import_module
from importlib.metadata import version, PackageNotFoundError

__all__ = [
    "load_data",
    "preprocess_data",
    "calculate_periodogram",
    "generate_frequency_grid",
    "fit_standard_model",
    "fit_segmented_spectrum",
    "interpret_results",
    "plot_spectrum",
    "Analysis",
    "compute_wwz",
    "fit_spectral_slope",
    "fit_segmented_slope",
    "multifractal_analysis_pipeline",
    "psresp_fit",
]

# ---- Metadata ----
try:
    __version__ = version("waterSpec")
except PackageNotFoundError:
    # Fallback version if the package is not installed.
    __version__ = "0.0.0"

# ---- Lazy import helper ----
def _lazy_import(func_name, module_name, dep_message=None):
    """Return a wrapper that imports the target function on first call."""
    def _wrapper(*args, **kwargs):
        try:
            module = import_module(f"waterSpec.{module_name}")
        except ImportError as e:
            msg = dep_message or str(e)
            raise ImportError(msg) from e
        func = getattr(module, func_name)
        return func(*args, **kwargs)
    _wrapper.__name__ = func_name
    return _wrapper

# ---- Public API (lazy wrappers) ----

load_data = _lazy_import(
    "load_data", "data_loader",
)

preprocess_data = _lazy_import(
    "preprocess_data", "preprocessor",
    dep_message="statsmodels is required for data preprocessing. Install with `pip install statsmodels`."
)

calculate_periodogram = _lazy_import(
    "calculate_periodogram", "spectral_analyzer",
    dep_message="astropy is required for periodogram calculation. Install with `pip install astropy`."
)

generate_frequency_grid = _lazy_import(
    "generate_frequency_grid", "frequency_generator"
)

fit_standard_model = _lazy_import(
    "fit_standard_model", "fitter",
)

fit_segmented_spectrum = _lazy_import(
    "fit_segmented_spectrum", "fitter",
    dep_message="piecewise_regression is required for segmented spectrum fitting. Install with `pip install piecewise-regression`."
)

interpret_results = _lazy_import(
    "interpret_results", "interpreter"
)

plot_spectrum = _lazy_import(
    "plot_spectrum", "plotting",
    dep_message="matplotlib is required for plotting. Install with `pip install matplotlib`."
)

psresp_fit = _lazy_import(
    "psresp_fit", "psresp",
    dep_message="astropy is required for PSRESP. Install with `pip install astropy`."
)

def Analysis(*args, **kwargs):
    """Lazy load Analysis class."""
    try:
        module = import_module("waterSpec.analysis")
    except ImportError as e:
        raise ImportError(
            "Failed to import Analysis class. Check that all required dependencies are installed."
        ) from e
    cls = getattr(module, "Analysis")
    return cls(*args, **kwargs)

# ---- Wavelet Wrappers ----
compute_wwz = _lazy_import(
    "compute_wwz", "wavelet",
    dep_message="pyleoclim is required for WWZ. Install with `pip install pyleoclim`."
)

fit_spectral_slope = _lazy_import(
    "fit_spectral_slope", "wavelet"
)

fit_segmented_slope = _lazy_import(
    "fit_segmented_slope", "wavelet",
    dep_message="piecewise_regression is required for segmented fitting. Install with `pip install piecewise-regression`."
)

multifractal_analysis_pipeline = _lazy_import(
    "multifractal_analysis_pipeline", "wavelet",
    dep_message="pymultifracs is required for multifractal analysis. Install with `pip install pymultifracs`."
)

# src/waterSpec/__init__.py
import logging

# Configure a NullHandler for the library's root logger to avoid
# "No handler found" warnings and let downstream applications configure logging.
logging.getLogger("waterSpec").addHandler(logging.NullHandler())


"""
WaterSpec: Spectral analysis toolkit for hydrological and environmental time series.

This package estimates spectral slopes (β) using Lomb–Scargle and segmented regressions,
supports confidence intervals via bootstrap resampling, and provides interpretive diagnostics.
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
    "calculate_partial_cross_haar",
    "SegmentedRegimeAnalysis",
    "calculate_wwz_coherence",
]

# ---- Metadata ----
try:
    __version__ = version("waterSpec")
except PackageNotFoundError:
    # Fallback version if the package is not installed.
    __version__ = "0.0.0"

# ---- Lazy import helper ----
def _lazy_import(func_name, module_name, dep_message=None, is_class=False):
    """Return a wrapper that imports the target function or class on first call."""
    if is_class:
        class LazyClassProxy:
            def __init__(self, *args, **kwargs):
                try:
                    module = import_module(f"waterSpec.{module_name}")
                except ImportError as e:
                    msg = dep_message or str(e)
                    raise ImportError(msg) from e
                cls = getattr(module, func_name)
                self._instance = cls(*args, **kwargs)

            def __getattr__(self, name):
                # If we haven't instantiated (e.g. static method call), we need the class
                try:
                    module = import_module(f"waterSpec.{module_name}")
                    cls = getattr(module, func_name)
                    return getattr(cls, name)
                except ImportError as e:
                    msg = dep_message or str(e)
                    raise ImportError(msg) from e
        return LazyClassProxy()
    else:
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

calculate_partial_cross_haar = _lazy_import(
    "calculate_partial_cross_haar", "multivariate"
)

calculate_wwz_coherence = _lazy_import(
    "calculate_wwz_coherence", "wwz_coherence",
    dep_message="scipy is required for WWZ coherence smoothing."
)

# SegmentedRegimeAnalysis is a class with static methods, so we need special handling
# or just import it directly for simplicity given the lazy loader complexity
# Let's revert the complex lazy loader and just implement a simple proxy for the class
class _SegmentedRegimeAnalysisProxy:
    def __getattr__(self, name):
        from waterSpec.segmentation import SegmentedRegimeAnalysis
        return getattr(SegmentedRegimeAnalysis, name)

SegmentedRegimeAnalysis = _SegmentedRegimeAnalysisProxy()

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
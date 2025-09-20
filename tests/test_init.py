import pytest
import waterSpec
from waterSpec import (
    generate_frequency_grid,
    calculate_periodogram,
    run_analysis
)
import types

def test_top_level_imports():
    """
    Test that key functions can be imported from the top-level package.
    """
    assert isinstance(generate_frequency_grid, types.FunctionType)
    assert isinstance(calculate_periodogram, types.FunctionType)
    assert isinstance(run_analysis, types.FunctionType)

def test_version_is_present():
    """
    Test that the package has a __version__ attribute.
    """
    assert hasattr(waterSpec, '__version__')
    assert isinstance(waterSpec.__version__, str)

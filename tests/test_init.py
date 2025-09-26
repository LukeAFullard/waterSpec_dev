import types

import waterSpec
from waterSpec import Analysis, calculate_periodogram, generate_frequency_grid


def test_top_level_imports():
    """
    Test that key functions can be imported from the top-level package.
    """
    assert isinstance(generate_frequency_grid, types.FunctionType)
    assert isinstance(calculate_periodogram, types.FunctionType)
    assert issubclass(Analysis, object)


def test_version_is_present():
    """
    Test that the package has a __version__ attribute.
    """
    assert hasattr(waterSpec, "__version__")
    assert isinstance(waterSpec.__version__, str)

import os
import pytest
import numpy as np
from waterSpec.plotting import plot_spectrum

@pytest.fixture
def spectrum_data():
    """Provides a sample power spectrum for testing."""
    frequency = np.logspace(-3, 0, 100)
    power = frequency ** -1.5
    fit_results = {'beta': 1.5, 'intercept': np.log(1.0)} # Dummy fit results
    return frequency, power, fit_results

def test_plot_spectrum_creates_file(spectrum_data, tmp_path):
    """
    Test that plot_spectrum creates an output file when a path is provided.
    """
    frequency, power, fit_results = spectrum_data
    output_file = tmp_path / "test_plot.png"

    # Call the plotting function
    plot_spectrum(frequency, power, fit_results=fit_results, output_path=str(output_file))

    # Check that the file was created
    assert os.path.exists(output_file)

    # Optional: Check that the file is not empty
    assert os.path.getsize(output_file) > 0

def test_plot_spectrum_runs_without_path(spectrum_data):
    """
    Test that plot_spectrum runs without error when no output path is given
    (e.g., it should display the plot interactively, which we can't test directly,
    but we can check that it doesn't crash).
    """
    frequency, power, fit_results = spectrum_data
    try:
        # We pass show=False to prevent a GUI window from opening during tests
        plot_spectrum(frequency, power, fit_results=fit_results, output_path=None, show=False)
    except Exception as e:
        pytest.fail(f"plot_spectrum raised an exception when no output path was provided: {e}")

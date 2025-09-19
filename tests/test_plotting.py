import pytest
import os
import numpy as np
import matplotlib.pyplot as plt
from waterSpec.plotting import plot_spectrum

@pytest.fixture
def spectrum_data():
    """Provides a synthetic power spectrum for testing."""
    frequency = np.logspace(-3, 0, 50)
    # White noise spectrum (power is constant) + some noise
    power = np.ones_like(frequency) + np.random.normal(0, 0.1, size=frequency.shape)
    fit_results = {'beta': 0.1, 'intercept': 0.0}
    return frequency, power, fit_results

def test_plot_spectrum_saves_file(spectrum_data, tmp_path):
    """Test that plot_spectrum saves a file to the specified path."""
    frequency, power, fit_results = spectrum_data
    output_file = tmp_path / "test_plot.png"

    plot_spectrum(frequency, power, fit_results, output_path=str(output_file))

    assert os.path.exists(output_file)

def test_plot_spectrum_with_ci_and_peaks(tmp_path):
    """
    Test that plot_spectrum can handle confidence intervals and find peaks.
    """
    # Create a synthetic spectrum with a clear peak
    frequency = np.linspace(0.01, 1, 500)
    power = np.exp(-0.5 * (frequency - 0.5)**2 / 0.01**2) + np.random.rand(500) * 0.1

    # Add fit results with confidence intervals
    fit_results = {
        'beta': 0.1,
        'intercept': 0.0,
        'beta_ci_lower': 0.05,
        'beta_ci_upper': 0.15
    }

    output_file = tmp_path / "test_plot_with_ci_and_peaks.png"

    try:
        plot_spectrum(
            frequency,
            power,
            fit_results=fit_results,
            analysis_type='standard',
            output_path=str(output_file)
        )
    except Exception as e:
        pytest.fail(f"plot_spectrum raised an exception with CI and peak data: {e}")

    assert os.path.exists(output_file)

def test_plot_spectrum_runs_without_path(spectrum_data):
    """
    Test that plot_spectrum runs without error when no output path is given.
    This test will 'succeed' if no exceptions are raised. We will also prevent
    the plot from actually showing and blocking the test run.
    """
    frequency, power, fit_results = spectrum_data

    # Use a non-interactive backend to prevent GUI windows during tests
    original_backend = plt.get_backend()
    plt.switch_backend('Agg')

    try:
        plot_spectrum(frequency, power, fit_results=fit_results, output_path=None)
    except Exception as e:
        pytest.fail(f"plot_spectrum raised an exception when no output path was provided: {e}")
    finally:
        # Restore the original backend
        plt.switch_backend(original_backend)

def test_plot_spectrum_segmented(spectrum_data, tmp_path):
    """
    Test that plot_spectrum can handle segmented fit results without crashing.
    """
    frequency, power, _ = spectrum_data
    output_file = tmp_path / "test_plot_segmented.png"

    # A mock model object is needed for the segmented plot
    class MockModel:
        def plot_fit(self, fig, ax, plot_data, plot_breakpoints, linewidth):
            pass

    segmented_fit_results = {
        'beta1': 0.5,
        'beta2': 1.8,
        'breakpoint': np.median(frequency),
        'model_object': MockModel()
    }

    try:
        plot_spectrum(
            frequency,
            power,
            fit_results=segmented_fit_results,
            analysis_type='segmented',
            output_path=str(output_file)
        )
    except Exception as e:
        pytest.fail(f"plot_spectrum raised an exception with segmented data: {e}")

    assert os.path.exists(output_file)

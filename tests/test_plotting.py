import os
from unittest.mock import patch

import matplotlib.pyplot as plt
import numpy as np
import pytest

from waterSpec.plotting import plot_spectrum, plot_changepoint_analysis


@pytest.fixture
def changepoint_results_data():
    """Provides a mock results dictionary for changepoint analysis."""
    # Create some dummy data for two segments
    freq = np.logspace(-3, 0, 50)
    power = np.ones_like(freq)
    fit_res = {
        "chosen_model_type": "standard",
        "beta": 1.0,
        "log_freq": np.log(freq),
        "intercept": 0,
    }

    results = {
        "changepoint_time": "10 days",
        "segment_before": {
            "frequency": freq,
            "power": power * 1.5,
            **fit_res,
            "beta": 0.5,
        },
        "segment_after": {
            "frequency": freq,
            "power": power * 0.5,
            **fit_res,
            "beta": 1.8,
        },
    }
    return results


@pytest.fixture
def spectrum_data():
    """Provides a synthetic power spectrum for testing."""
    frequency = np.logspace(-3, 0, 50)
    # White noise spectrum (power is constant) + some noise
    power = np.ones_like(frequency) + np.random.normal(0, 0.1, size=frequency.shape)
    fit_results = {"beta": 0.1, "intercept": 0.0}
    return frequency, power, fit_results


def test_plot_spectrum_saves_file(spectrum_data, tmp_path):
    """Test that plot_spectrum saves a file to the specified path."""
    frequency, power, fit_results = spectrum_data
    output_file = tmp_path / "test_plot.png"

    plot_spectrum(frequency, power, fit_results, output_path=str(output_file))

    assert os.path.exists(output_file)


def test_plot_changepoint_analysis_combined_style(
    changepoint_results_data, tmp_path
):
    """
    Test that `plot_changepoint_analysis` with `plot_style='combined'`
    creates a single combined plot.
    """
    output_dir = tmp_path / "changepoint_combined"
    os.makedirs(output_dir)

    plot_changepoint_analysis(
        changepoint_results_data,
        str(output_dir),
        param_name="TestParam",
        plot_style="combined",
    )

    expected_file = output_dir / "TestParam_changepoint_combined.png"
    assert expected_file.exists()


@patch("matplotlib.pyplot.subplots")
def test_plot_spectrum_handles_failed_fit(mock_subplots, spectrum_data, tmp_path):
    """
    Test that plot_spectrum adds a 'fitting failed' annotation if the fit
    results are invalid.
    """
    # Arrange: Create mock figure and axes, and have subplots return them.
    from unittest.mock import MagicMock
    mock_fig = MagicMock()
    mock_ax = MagicMock()
    mock_subplots.return_value = (mock_fig, mock_ax)

    frequency, power, _ = spectrum_data
    # Simulate a failed fit by providing empty results
    failed_fit_results = {}
    output_file = tmp_path / "test_plot_failed.png"

    # Act
    plot_spectrum(
        frequency,
        power,
        fit_results=failed_fit_results,
        output_path=str(output_file),
    )

    # Assert that the `text` method on the mocked axes object was called.
    mock_ax.text.assert_called_once()
    # Check that the text contains the failure message
    assert "Fit Failed" in mock_ax.text.call_args[0][2]
    # Check that the savefig function was called correctly on the figure object
    mock_fig.savefig.assert_called_once_with(str(output_file), dpi=300)


def test_plot_spectrum_with_ci_and_peaks(tmp_path):
    """
    Test that plot_spectrum can handle confidence intervals and find peaks.
    """
    # Create a synthetic spectrum with a clear peak
    frequency = np.linspace(0.01, 1, 500)
    power = np.exp(-0.5 * (frequency - 0.5) ** 2 / 0.01**2) + np.random.rand(500) * 0.1

    # Add fit results with confidence intervals
    fit_results = {
        "beta": 0.1,
        "intercept": 0.0,
        "beta_ci_lower": 0.05,
        "beta_ci_upper": 0.15,
    }

    output_file = tmp_path / "test_plot_with_ci_and_peaks.png"

    try:
        plot_spectrum(
            frequency,
            power,
            fit_results=fit_results,
            output_path=str(output_file),
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
    plt.switch_backend("Agg")

    try:
        plot_spectrum(frequency, power, fit_results=fit_results, output_path=None)
    except Exception as e:
        pytest.fail(
            f"plot_spectrum raised an exception when no output path was provided: {e}"
        )
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
        def plot_fit(self, **kwargs):
            pass

        def predict(self, x):
            return np.zeros_like(x)

    segmented_fit_results = {
        "beta1": 0.5,
        "beta2": 1.8,
        "breakpoint": np.median(frequency),
        "model_object": MockModel(),
        "log_freq": np.log(frequency),
    }

    try:
        plot_spectrum(
            frequency,
            power,
            fit_results=segmented_fit_results,
            output_path=str(output_file),
        )
    except Exception as e:
        pytest.fail(f"plot_spectrum raised an exception with segmented data: {e}")

    assert os.path.exists(output_file)


def test_plot_spectrum_multi_breakpoint(spectrum_data, tmp_path):
    """
    Test that plot_spectrum can handle multi-breakpoint (n>1) segmented fit
    results without crashing.
    """
    frequency, power, _ = spectrum_data
    output_file = tmp_path / "test_plot_multi_segmented.png"

    # A mock model object is needed for the segmented plot
    class MockModel:
        def predict(self, x):
            # Return a simple sloped line for prediction
            return -1.0 * x

    # Create a fit result dictionary that simulates a 2-breakpoint fit
    log_freq = np.log(frequency)
    multi_segmented_fit_results = {
        "n_breakpoints": 2,
        "betas": [0.2, 1.5, 0.8],
        "breakpoints": [frequency[10], frequency[30]],
        "model_object": MockModel(),
        "log_freq": log_freq,
        "fitted_log_power": -1.0 * log_freq,  # Mock the fitted line
    }

    try:
        plot_spectrum(
            frequency,
            power,
            fit_results=multi_segmented_fit_results,
            output_path=str(output_file),
        )
    except Exception as e:
        pytest.fail(
            f"plot_spectrum raised an exception with multi-breakpoint data: {e}"
        )

    assert os.path.exists(output_file)

import pytest
from waterSpec.workflow import run_analysis
import os
import numpy as np
import pandas as pd

def test_run_analysis_workflow():
    """
    Test the full analysis workflow with the `run_analysis` function.
    The sample data is noisy and should result in a warning in the interpretation.
    """
    file_path = 'examples/sample_data.csv'

    results = run_analysis(file_path, time_col='timestamp', data_col='concentration', n_bootstraps=10)

    assert 'scientific_interpretation' in results
    assert "Warning: Beta value is significantly negative" in results['scientific_interpretation']

def test_workflow_with_plotting(tmp_path):
    """
    Test that the workflow can generate a plot, and that it returns a
    warning for the noisy sample data.
    """
    file_path = 'examples/sample_data.csv'
    output_plot_path = tmp_path / "spectrum_plot.png"

    results = run_analysis(
        file_path,
        time_col='timestamp',
        data_col='concentration',
        do_plot=True,
        output_path=str(output_plot_path),
        n_bootstraps=10
    )

    # Check that the plot was created and a warning was reported.
    assert os.path.exists(output_plot_path)
    assert 'scientific_interpretation' in results
    assert "Warning: Beta value is significantly negative" in results['scientific_interpretation']

def test_workflow_segmented(tmp_path):
    """Test the segmented analysis workflow."""
    file_path = 'examples/sample_data.csv'

    results = run_analysis(
        file_path,
        time_col='timestamp',
        data_col='concentration',
        analysis_type='segmented'
    )

    assert 'beta1' in results
    assert 'beta2' in results
    assert 'breakpoint' in results


# --- New tests for known beta values ---

@pytest.mark.parametrize("known_beta, tolerance", [
    (0.0, 0.2),  # White noise, beta should be close to 0
    (1.0, 0.25), # Pink noise, beta should be close to 1
    (2.0, 0.25), # Brownian noise, beta should be close to 2
])
def test_workflow_with_known_beta(tmp_path, known_beta, tolerance):
    """
    Test the full workflow with synthetic data of a known spectral exponent.
    """
    # 1. Generate synthetic data
    time, series = generate_synthetic_series(n_points=2048, beta=known_beta)

    # 2. Create a temporary data file
    file_path = create_test_data_file(tmp_path, time, series, time_col='time', data_col='value')

    # 3. Run the analysis
    results = run_analysis(
        file_path,
        time_col='time',
        data_col='value',
        detrend_method=None,  # Detrending removes the signal we're testing for
        n_bootstraps=10 # Keep tests fast
    )

    # 4. Assert that the calculated beta is close to the known beta
    assert 'beta' in results
    assert results['beta'] == pytest.approx(known_beta, abs=tolerance)


# --- New Edge Case Tests for the Workflow ---

def test_workflow_insufficient_data_after_preprocess(tmp_path):
    """
    Test that the workflow raises a ValueError if data is too short after preprocessing.
    """
    # Create data where most values are NaN, leaving fewer than the minimum required
    time = pd.date_range(start='2000-01-01', periods=20, freq='D')
    series = np.full(20, np.nan)
    rng = np.random.default_rng(42)
    series[:5] = rng.random(5) # Only 5 valid points

    file_path = create_test_data_file(tmp_path, time, series)

    with pytest.raises(ValueError, match="The time series has only 5 valid data points"):
        run_analysis(file_path, time_col='time', data_col='value')

def test_workflow_zero_variance_data(tmp_path):
    """
    Test the workflow with data that has zero variance.
    The periodogram should have zero power, and beta should be NaN.
    """
    time = pd.date_range(start='2000-01-01', periods=100, freq='D')
    series = np.ones(100) # A flat line
    file_path = create_test_data_file(tmp_path, time, series)

    results = run_analysis(file_path, time_col='time', data_col='value', detrend_method=None)

    # For a zero-variance signal, the analysis should fail gracefully.
    # The exact beta value is less important than the interpretation.
    assert 'beta' in results
    assert "Could not determine a valid beta value" in results['summary_text']

def test_workflow_all_nan_data(tmp_path):
    """
    Test that a file containing only NaNs raises the appropriate error.
    """
    time = pd.date_range(start='2000-01-01', periods=10, freq='D')
    series = np.full(10, np.nan)
    file_path = create_test_data_file(tmp_path, time, series)

    # This should be caught by the preprocessor's length check
    with pytest.raises(ValueError, match="The time series has only 0 valid data points"):
        run_analysis(file_path, time_col='time', data_col='value')

# --- Helper functions ---

def generate_synthetic_series(n_points=1024, beta=0, seed=42):
    """
    Generates a synthetic time series with a known spectral exponent (beta).
    """
    rng = np.random.default_rng(seed)
    freq = np.fft.rfftfreq(n_points)
    freq[0] = 1e-6
    power_spectrum = freq ** (-beta)
    amplitude_spectrum = np.sqrt(power_spectrum)
    random_phases = rng.uniform(0, 2 * np.pi, len(freq))
    fourier_spectrum = amplitude_spectrum * np.exp(1j * random_phases)
    series = np.fft.irfft(fourier_spectrum, n=n_points)
    # Do not normalize non-stationary series, as it alters the spectral properties.
    # The Lomb-Scargle periodogram's own normalization will handle the variance.
    # Generate a proper DatetimeIndex
    time = pd.date_range(start='2000-01-01', periods=n_points, freq='D')
    return time, series

def create_test_data_file(tmp_path, time, series, time_col='time', data_col='value'):
    """
    Creates a temporary CSV file for testing the workflow.
    """
    file_path = tmp_path / "synthetic_data.csv"
    # Ensure time is in a format that pandas can write to CSV
    if isinstance(time, pd.DatetimeIndex):
        time_to_write = time.strftime('%Y-%m-%d')
    else:
        time_to_write = time

    df = pd.DataFrame({time_col: time_to_write, data_col: series})
    df.to_csv(file_path, index=False)
    return str(file_path)

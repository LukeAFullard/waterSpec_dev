import pytest
from waterSpec import Analysis
import os
import numpy as np
import pandas as pd
from unittest.mock import patch

# Helper function to create test data files
def create_test_data_file(tmp_path, time, series, errors=None, time_col='time', data_col='value', error_col='error'):
    file_path = tmp_path / "test_data.csv"
    if isinstance(time, pd.DatetimeIndex):
        time_to_write = time.strftime('%Y-%m-%d %H:%M:%S')
    else:
        time_to_write = time
    data_dict = {time_col: time_to_write, data_col: series}
    if errors is not None:
        data_dict[error_col] = errors
    df = pd.DataFrame(data_dict)
    df.to_csv(file_path, index=False)
    return str(file_path)

# Helper function to generate synthetic data
def generate_synthetic_series(n_points=1024, beta=0.0, seed=42):
    rng = np.random.default_rng(seed)
    freq = np.fft.rfftfreq(n_points)
    freq[0] = 1e-9  # Avoid division by zero
    power_spectrum = freq ** (-beta)
    amplitude_spectrum = np.sqrt(power_spectrum)
    random_phases = rng.uniform(0, 2 * np.pi, len(freq))
    fourier_spectrum = amplitude_spectrum * np.exp(1j * random_phases)
    series = np.fft.irfft(fourier_spectrum, n=n_points)
    time = pd.date_range(start='2000-01-01', periods=n_points, freq='D')
    return time, series

# --- Tests for the new Analysis Class ---

def test_analysis_class_initialization(tmp_path):
    """Test that the Analysis class initializes correctly and loads data."""
    time, series = generate_synthetic_series()
    file_path = create_test_data_file(tmp_path, time, series)

    analyzer = Analysis(file_path, time_col='time', data_col='value')

    assert analyzer is not None
    assert len(analyzer.time) == 1024
    assert len(analyzer.data) == 1024
    assert analyzer.param_name == 'value'

def test_analysis_run_full_analysis_creates_outputs(tmp_path):
    """Test that run_full_analysis creates the plot and summary files."""
    file_path = 'examples/sample_data.csv'
    output_dir = tmp_path / "results"

    analyzer = Analysis(file_path, time_col='timestamp', data_col='concentration')
    results = analyzer.run_full_analysis(output_dir=str(output_dir), n_bootstraps=10)

    # Check that output files were created
    expected_plot = output_dir / "concentration_spectrum_plot.png"
    expected_summary = output_dir / "concentration_summary.txt"
    assert expected_plot.exists()
    assert expected_summary.exists()

    # Check that the summary contains expected text
    summary_text = expected_summary.read_text()
    assert "Analysis for: concentration" in summary_text
    assert "β =" in summary_text

    # Check that the results dictionary is populated and valid
    assert 'summary_text' in results
    assert "Analysis for: concentration" in results['summary_text']
    assert ('beta' in results or 'beta1' in results)

@pytest.mark.parametrize("known_beta, tolerance", [
    (0.0, 0.2),
    (1.0, 0.25),
    (2.0, 0.25),
])
def test_analysis_with_known_beta(tmp_path, known_beta, tolerance):
    """Test the full workflow with synthetic data of a known spectral exponent."""
    time, series = generate_synthetic_series(n_points=2048, beta=known_beta)
    file_path = create_test_data_file(tmp_path, time, series)

    analyzer = Analysis(file_path, time_col='time', data_col='value', detrend_method=None)
    results = analyzer.run_full_analysis(output_dir=tmp_path, n_bootstraps=10)

    assert 'beta' in results
    assert results['beta'] == pytest.approx(known_beta, abs=tolerance)
    assert results['analysis_mode'] == 'auto'
    assert results['chosen_model'] == 'standard'

@patch('waterSpec.analysis.fit_segmented_spectrum')
@patch('waterSpec.analysis.fit_spectrum_with_bootstrap')
def test_analysis_auto_chooses_segmented_with_mock(mock_fit_standard, mock_fit_segmented, tmp_path):
    """Test that auto mode chooses the segmented model with a better BIC score."""
    # Add required keys for the residual peak finding to the mock return values
    dummy_array = np.array([1, 2, 3])
    mock_fit_segmented.return_value = {
        'beta1': 0.5, 'beta2': 1.8, 'breakpoint': 0.01, 'bic': 100.0,
        'residuals': dummy_array, 'fitted_log_power': dummy_array,
        'log_freq': dummy_array, 'log_power': dummy_array
    }
    mock_fit_standard.return_value = {
        'beta': 1.2, 'bic': 120.0,
        'residuals': dummy_array, 'fitted_log_power': dummy_array,
        'log_freq': dummy_array, 'log_power': dummy_array
    }

    file_path = create_test_data_file(tmp_path, pd.date_range('2023', periods=100), np.random.rand(100))
    output_dir = tmp_path / "results"

    analyzer = Analysis(file_path, time_col='time', data_col='value')
    results = analyzer.run_full_analysis(output_dir=str(output_dir))

    assert results['chosen_model'] == 'segmented'
    assert results['bic_comparison']['segmented'] == 100.0
    assert 'beta1' in results

def test_analysis_insufficient_data(tmp_path):
    """Test that the Analysis class raises an error for insufficient data."""
    time = pd.date_range(start='2000-01-01', periods=20, freq='D')
    series = np.full(20, np.nan)
    rng = np.random.default_rng(42)
    series[:5] = rng.random(5) # Only 5 valid points
    file_path = create_test_data_file(tmp_path, time, series)

    with pytest.raises(ValueError, match="The time series has only 5 valid data points"):
        Analysis(file_path, time_col='time', data_col='value')

def test_analysis_zero_variance_data(tmp_path):
    """Test the workflow with data that has zero variance."""
    time = pd.date_range(start='2000-01-01', periods=100, freq='D')
    series = np.ones(100)
    file_path = create_test_data_file(tmp_path, time, series)
    output_dir = tmp_path / "results"

    analyzer = Analysis(file_path, time_col='time', data_col='value', detrend_method=None)
    results = analyzer.run_full_analysis(output_dir=str(output_dir))

    assert "Analysis failed" in results['summary_text']

def test_deprecation_warning_for_old_workflow():
    """Test that the old run_analysis function issues a DeprecationWarning."""
    file_path = 'examples/sample_data.csv'
    with pytest.warns(DeprecationWarning, match="The 'run_analysis' function is deprecated"):
        # We need to import it here to isolate the test
        from waterSpec.workflow import run_analysis
        run_analysis(file_path, time_col='timestamp', data_col='concentration', n_bootstraps=10, analysis_type='standard')

def test_analysis_fap_threshold_is_configurable(tmp_path):
    """Test that the fap_threshold can be configured in run_full_analysis."""
    file_path = 'examples/sample_data.csv'
    output_dir = tmp_path / "results"
    custom_fap = 0.05

    analyzer = Analysis(file_path, time_col='timestamp', data_col='concentration')
    results = analyzer.run_full_analysis(
        output_dir=str(output_dir),
        n_bootstraps=10,
            peak_detection_method='fap',
        fap_threshold=custom_fap
    )

    assert 'fap_threshold' in results
    assert results['fap_threshold'] == custom_fap

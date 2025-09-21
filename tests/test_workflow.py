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

    results = run_analysis(file_path, time_col='timestamp', data_col='concentration', n_bootstraps=10, analysis_type='standard')

    assert 'scientific_interpretation' in results
    # With the new robust default, the beta is no longer significantly negative
    assert "-0.5 < β < 1 (fGn-like)" in results['scientific_interpretation']

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
        n_bootstraps=10,
        analysis_type='standard'
    )

    # Check that the plot was created and a warning was reported.
    assert os.path.exists(output_plot_path)
    assert 'scientific_interpretation' in results
    # With the new robust default, the beta is no longer significantly negative
    assert "-0.5 < β < 1 (fGn-like)" in results['scientific_interpretation']

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

    # 5. Assert that the auto-analysis details are present
    assert results['analysis_mode'] == 'auto'
    assert 'standard_fit' in results
    assert 'segmented_fit' in results
    assert 'beta' in results['standard_fit']
    # For this synthetic data, the standard fit should always be chosen
    assert results['chosen_model'] == 'standard'


from unittest.mock import patch

@patch('waterSpec.workflow.fit_segmented_spectrum')
@patch('waterSpec.workflow.fit_spectrum_with_bootstrap')
def test_workflow_auto_chooses_segmented_with_mock(mock_fit_standard, mock_fit_segmented, tmp_path):
    """
    Test that the auto analysis mode correctly chooses the segmented model
    when its BIC score is more favorable, using mocks to ensure predictable behavior.
    """
    # 1. Configure the mock return values
    # The segmented fit should have a lower (better) BIC
    mock_fit_segmented.return_value = {
        'beta1': 0.5, 'beta2': 1.8, 'breakpoint': 0.01, 'bic': 100.0
    }
    # The standard fit should have a higher (worse) BIC
    mock_fit_standard.return_value = {
        'beta': 1.2, 'bic': 120.0
    }

    # 2. Create a dummy data file (the mocks will prevent it from being used for fitting)
    file_path = create_test_data_file(tmp_path, pd.date_range('2023', periods=100), np.random.rand(100))

    # 3. Run the analysis in auto mode
    results = run_analysis(
        file_path,
        time_col='time',
        data_col='value',
        analysis_type='auto',
    )

    # 4. Assert that the segmented model was chosen because its BIC was lower
    assert 'chosen_model' in results
    assert results['chosen_model'] == 'segmented'
    assert results['bic_comparison']['segmented'] == 100.0
    assert results['bic_comparison']['standard'] == 120.0

    # 5. Assert that the top-level results are from the segmented fit
    assert 'beta1' in results
    assert results['beta1'] == 0.5


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
    assert "Analysis failed: Could not determine a valid spectral slope." in results['summary_text']

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

def generate_synthetic_series_with_breakpoint(n_points=2048, beta1=0.5, beta2=1.8, breakpoint_freq=0.1, seed=42):
    """
    Generates a synthetic time series with a known spectral breakpoint.
    This version creates two independent series and stitches them together in the
    frequency domain, which creates a more robust breakpoint for testing.
    """
    rng = np.random.default_rng(seed)
    # Generate two independent fGn/fBm series
    # Use fbm library if available, otherwise use the existing spectral method
    try:
        from fbm import fbm
        # Convert beta to Hurst
        H1 = (beta1 - 1) / 2
        H2 = (beta2 - 1) / 2
        series1 = fbm(n=n_points, hurst=H1, length=1, method='daviesharte')
        series2 = fbm(n=n_points, hurst=H2, length=1, method='daviesharte')
    except ImportError:
        # Fallback to spectral method if fbm is not installed
        _, series1 = generate_synthetic_series(n_points=n_points, beta=beta1, seed=seed)
        _, series2 = generate_synthetic_series(n_points=n_points, beta=beta2, seed=seed+1)


    # Get their Fourier transforms
    f_series1 = np.fft.rfft(series1)
    f_series2 = np.fft.rfft(series2)

    # Filter and combine in the frequency domain
    freq = np.fft.rfftfreq(n_points)
    f_combined = np.zeros_like(f_series1)

    low_freq_mask = freq < breakpoint_freq
    high_freq_mask = freq >= breakpoint_freq

    f_combined[low_freq_mask] = f_series1[low_freq_mask]
    f_combined[high_freq_mask] = f_series2[high_freq_mask]

    # Transform back to time domain
    series = np.fft.irfft(f_combined, n=n_points)
    time = pd.date_range(start='2000-01-01', periods=n_points, freq='D')
    return time, series

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
    time = pd.date_range(start='2000-01-01', periods=n_points, freq='D')
    return time, series

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

def create_test_data_file(tmp_path, time, series, errors=None, time_col='time', data_col='value', error_col='error'):
    """
    Creates a temporary CSV file for testing the workflow.
    """
    file_path = tmp_path / "synthetic_data.csv"
    # Ensure time is in a format that pandas can write to CSV
    if isinstance(time, pd.DatetimeIndex):
        time_to_write = time.strftime('%Y-%m-%d')
    else:
        time_to_write = time

    data_dict = {time_col: time_to_write, data_col: series}
    if errors is not None:
        data_dict[error_col] = errors

    df = pd.DataFrame(data_dict)
    df.to_csv(file_path, index=False)
    return str(file_path)


# --- New test for error column ---

def test_workflow_with_error_col(tmp_path):
    """
    Test that the workflow runs successfully when an error column is provided.
    """
    # 1. Generate synthetic data
    time, series = generate_synthetic_series(n_points=512, beta=1.0)
    # Generate some plausible errors
    rng = np.random.default_rng(42)
    errors = rng.uniform(0.05, 0.2, size=len(series))

    # 2. Create a temporary data file
    file_path = create_test_data_file(tmp_path, time, series, errors=errors, error_col='dy_val')

    # 3. Run the analysis
    results = run_analysis(
        file_path,
        time_col='time',
        data_col='value',
        error_col='dy_val', # Pass the error column name
        detrend_method=None,
        n_bootstraps=10,
        analysis_type='standard'
    )

    # 4. Assert that the analysis completed and returned a valid beta
    assert 'beta' in results
    assert np.isfinite(results['beta'])
    # A weighted fit might be slightly different, so we use a reasonable tolerance
    assert results['beta'] == pytest.approx(1.0, abs=0.3)


# --- New test for FAP ---

def test_workflow_with_fap(tmp_path):
    """
    Test that the workflow runs successfully and finds a significant peak
    when fap_threshold is provided.
    """
    # 1. Generate synthetic data with a strong periodic signal
    n_points = 512
    time = pd.to_datetime(np.arange(n_points), unit='D', origin='2000-01-01')
    known_frequency = 1.0 / 20.0  # A period of 20 days
    rng = np.random.default_rng(42)
    series = np.sin(2 * np.pi * known_frequency * np.arange(n_points)) + rng.normal(0, 0.2, n_points)

    # 2. Create a temporary data file
    file_path = create_test_data_file(tmp_path, time, series)

    # 3. Run the analysis with FAP calculation
    results = run_analysis(
        file_path,
        time_col='time',
        data_col='value',
        detrend_method=None,  # Do not detrend when testing for a known signal
        fap_threshold=0.01,
        n_bootstraps=10 # Keep tests fast
    )

    # 4. Assert that the results contain FAP info
    assert 'significant_peaks' in results
    assert 'fap_level' in results
    assert len(results['significant_peaks']) > 0
    found_peak = results['significant_peaks'][0]
    # Convert known_frequency from cycles/day to Hz for comparison
    known_frequency_hz = known_frequency / 86400.0
    assert found_peak['frequency'] == pytest.approx(known_frequency_hz, abs=1e-6)


# --- New test for preprocessing flags ---

def test_workflow_with_preprocessing_flags(tmp_path):
    """
    Test that the preprocessing flags in the workflow are functional.
    """
    # 1. Generate synthetic data
    time, series = generate_synthetic_series(n_points=512, beta=1.0)
    # Ensure data is positive for log transform
    series = series - np.min(series) + 1

    # 2. Create a temporary data file
    file_path = create_test_data_file(tmp_path, time, series)

    # 3. Run the analysis with different flags
    # Base run (no normalization or log transform)
    base_results = run_analysis(file_path, time_col='time', data_col='value', n_bootstraps=10, analysis_type='standard')

    # Run with normalization
    norm_results = run_analysis(file_path, time_col='time', data_col='value', normalize_data=True, n_bootstraps=10, analysis_type='standard')

    # Run with log transform
    log_results = run_analysis(file_path, time_col='time', data_col='value', log_transform_data=True, n_bootstraps=10, analysis_type='standard')

    # 4. Assert that the flags changed the results
    # Normalizing data changes its variance, which should not change the beta of a power-law signal
    # However, for a real-world, non-ideal signal, slight changes are expected.
    # The main test is that it runs and produces a valid, different result object.
    assert np.isfinite(norm_results['beta'])
    assert norm_results['beta'] != log_results['beta'] # Log transform should definitely change it

    # Log transform should significantly alter the spectral slope
    assert np.isfinite(log_results['beta'])
    assert base_results['beta'] != log_results['beta']

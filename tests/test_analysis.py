from unittest.mock import ANY, patch

import numpy as np
import pandas as pd
import pytest

from waterSpec import Analysis


# Helper function to create test data files
def create_test_data_file(
    tmp_path,
    time,
    series,
    errors=None,
    time_col="time",
    data_col="value",
    error_col="error",
):
    file_path = tmp_path / "test_data.csv"
    if isinstance(time, pd.DatetimeIndex):
        time_to_write = time.strftime("%Y-%m-%d %H:%M:%S")
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
    time = pd.date_range(start="2000-01-01", periods=n_points, freq="D")
    return time, series


# --- Tests for the new Analysis Class ---


def test_analysis_class_initialization(tmp_path):
    """Test that the Analysis class initializes correctly and loads data."""
    time, series = generate_synthetic_series()
    file_path = create_test_data_file(tmp_path, time, series)

    analyzer = Analysis(file_path, time_col="time", data_col="value")

    assert analyzer is not None
    assert len(analyzer.time) == 1024
    assert len(analyzer.data) == 1024
    assert analyzer.param_name == "value"


def test_analysis_run_full_analysis_creates_outputs(tmp_path):
    """Test that run_full_analysis creates the plot and summary files."""
    file_path = "examples/sample_data.csv"
    output_dir = tmp_path / "results"

    analyzer = Analysis(file_path, time_col="timestamp", data_col="concentration")
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
    assert "summary_text" in results
    assert "Analysis for: concentration" in results["summary_text"]
    assert "beta" in results or "beta1" in results


@pytest.mark.parametrize(
    "known_beta, tolerance",
    [
        (0.0, 0.2),
        (1.0, 0.25),
        (2.0, 0.25),
    ],
)
def test_analysis_with_known_beta(tmp_path, known_beta, tolerance):
    """Test the full workflow with synthetic data of a known spectral exponent."""
    time, series = generate_synthetic_series(n_points=2048, beta=known_beta)
    file_path = create_test_data_file(tmp_path, time, series)

    analyzer = Analysis(
        file_path, time_col="time", data_col="value", detrend_method=None
    )
    results = analyzer.run_full_analysis(output_dir=tmp_path, n_bootstraps=10)

    assert "beta" in results
    assert results["beta"] == pytest.approx(known_beta, abs=tolerance)
    assert results["analysis_mode"] == "auto"
    assert results["chosen_model"] == "standard"


@patch("waterSpec.analysis.fit_segmented_spectrum")
@patch("waterSpec.analysis.fit_spectrum_with_bootstrap")
def test_analysis_auto_chooses_segmented_with_mock(
    mock_fit_standard, mock_fit_segmented, tmp_path
):
    """Test that auto mode chooses the segmented model with a better BIC score."""
    # Configure mocks to return data with the new structure
    mock_fit_segmented.return_value = {
        "betas": [0.5, 1.8],
        "breakpoints": [0.01],
        "bic": 100.0,
        "n_breakpoints": 1,
        # Add required keys for the residual peak finding
        "residuals": np.array([1, 2, 3]),
        "fitted_log_power": np.array([1, 2, 3]),
        "log_freq": np.array([1, 2, 3]),
        "log_power": np.array([1, 2, 3]),
    }
    mock_fit_standard.return_value = {
        "beta": 1.2,
        "betas": [1.2],
        "bic": 120.0,
        "n_breakpoints": 0,
        # Add required keys for the residual peak finding
        "residuals": np.array([1, 2, 3]),
        "fitted_log_power": np.array([1, 2, 3]),
        "log_freq": np.array([1, 2, 3]),
        "log_power": np.array([1, 2, 3]),
    }

    file_path = create_test_data_file(
        tmp_path, pd.date_range("2023", periods=100), np.random.rand(100)
    )
    output_dir = tmp_path / "results"

    analyzer = Analysis(file_path, time_col="time", data_col="value")
    results = analyzer.run_full_analysis(output_dir=str(output_dir))

    assert results["chosen_model"] == "segmented_1bp"
    assert results["bic"] == 100.0
    assert results["n_breakpoints"] == 1


def test_analysis_insufficient_data(tmp_path):
    """Test that the Analysis class raises an error for insufficient data."""
    time = pd.date_range(start="2000-01-01", periods=20, freq="D")
    series = np.full(20, np.nan)
    rng = np.random.default_rng(42)
    series[:5] = rng.random(5)  # Only 5 valid points
    file_path = create_test_data_file(tmp_path, time, series)

    with pytest.raises(
        ValueError, match="The time series has only 5 valid data points"
    ):
        Analysis(file_path, time_col="time", data_col="value")


def test_analysis_zero_variance_data(tmp_path):
    """Test the workflow with data that has zero variance."""
    time = pd.date_range(start="2000-01-01", periods=100, freq="D")
    series = np.ones(100)
    file_path = create_test_data_file(tmp_path, time, series)
    output_dir = tmp_path / "results"

    analyzer = Analysis(
        file_path, time_col="time", data_col="value", detrend_method=None
    )
    results = analyzer.run_full_analysis(output_dir=str(output_dir))

    assert "Analysis failed" in results["summary_text"]


def test_analysis_fap_threshold_is_configurable(tmp_path):
    """Test that the fap_threshold can be configured in run_full_analysis."""
    file_path = "examples/sample_data.csv"
    output_dir = tmp_path / "results"
    custom_fap = 0.05

    analyzer = Analysis(file_path, time_col="timestamp", data_col="concentration")
    results = analyzer.run_full_analysis(
        output_dir=str(output_dir),
        n_bootstraps=10,
        peak_detection_method="fap",
        fap_threshold=custom_fap,
    )

    assert "fap_threshold" in results
    assert results["fap_threshold"] == custom_fap


def test_analysis_residual_method_finds_peak(tmp_path):
    """Test the full workflow with the residual method on data with a known peak."""
    # Generate a signal with a strong periodic component
    n_points = 512
    time = pd.date_range(start="2000-01-01", periods=n_points, freq="D")
    rng = np.random.default_rng(42)
    noise = rng.normal(0, 0.5, n_points)
    # Add a sine wave with a period of ~50 days
    known_freq_cpd = 1 / 50  # cycles/day
    signal = 2 * np.sin(2 * np.pi * known_freq_cpd * np.arange(n_points))
    series = noise + signal

    file_path = create_test_data_file(tmp_path, time, series)
    output_dir = tmp_path / "results_residual"

    analyzer = Analysis(
        file_path, time_col="time", data_col="value", detrend_method="linear"
    )
    results = analyzer.run_full_analysis(
        output_dir=str(output_dir),
        grid_type="linear",
        peak_detection_method="residual",
        peak_detection_ci=95,
    )

    assert "significant_peaks" in results
    assert len(results["significant_peaks"]) > 0

    # The frequency is in Hz, so convert our known frequency
    known_freq_hz = known_freq_cpd / 86400
    found_peak_freq = results["significant_peaks"][0]["frequency"]
    assert found_peak_freq == pytest.approx(known_freq_hz, rel=0.05)


def test_analysis_with_censored_data(tmp_path):
    """Test the full workflow with a data file containing censored values."""
    # The data file is in the repo, not a temporary path
    file_path = "tests/data/censored_data.csv"
    output_dir = tmp_path / "results_censored"

    # Run the analysis with a strategy to handle censored data
    analyzer = Analysis(
        file_path,
        time_col="date",
        data_col="concentration",
        censor_strategy="use_detection_limit",
    )
    results = analyzer.run_full_analysis(output_dir=str(output_dir), n_bootstraps=10)

    # Check that the analysis ran successfully and produced valid results
    assert "summary_text" in results
    assert "Analysis failed" not in results["summary_text"]
    assert "beta" in results or "beta1" in results
    assert np.isfinite(results.get("beta", np.nan)) or (
        "betas" in results and np.isfinite(results["betas"][0])
    )

    # Check that the output files were created
    expected_plot = output_dir / "concentration_spectrum_plot.png"
    expected_summary = output_dir / "concentration_summary.txt"
    assert expected_plot.exists()
    assert expected_summary.exists()


@patch("waterSpec.analysis.fit_segmented_spectrum")
@patch("waterSpec.analysis.fit_spectrum_with_bootstrap")
def test_analysis_max_breakpoints_selects_best_model(
    mock_fit_standard, mock_fit_segmented, tmp_path
):
    """
    Test that the model selection logic correctly chooses the model with the
    lowest BIC when max_breakpoints > 1.
    """
    # --- Mock Configuration ---
    # Mock the standard fit (0 breakpoints)
    dummy_data = {
        "residuals": np.array([1, 2, 3]),
        "fitted_log_power": np.array([1, 2, 3]),
        "log_freq": np.array([1, 2, 3]),
        "log_power": np.array([1, 2, 3]),
    }
    mock_fit_standard.return_value = {
        "beta": 1.0,
        "bic": 200.0,
        "n_breakpoints": 0,
        **dummy_data,
    }

    # Mock the segmented fits (1 and 2 breakpoints) to return different BICs
    def segmented_side_effect(
        frequency, power, n_breakpoints, p_threshold, **kwargs
    ):
        """
        A side effect function for the mock that mimics the behavior of
        `fit_segmented_spectrum`, accepting the new bootstrap arguments.
        """
        if n_breakpoints == 1:
            return {
                "betas": [0.5, 1.5],
                "breakpoints": [0.1],
                "bic": 150.0,
                "n_breakpoints": 1,
                **dummy_data,
            }
        elif n_breakpoints == 2:
            return {
                "betas": [0.2, 1.8, 0.5],
                "breakpoints": [0.01, 0.1],
                "bic": 100.0,
                "n_breakpoints": 2,
                **dummy_data,
            }
        return {}

    mock_fit_segmented.side_effect = segmented_side_effect

    # --- Test Execution ---
    file_path = create_test_data_file(
        tmp_path, pd.date_range("2023", periods=100), np.random.rand(100)
    )
    output_dir = tmp_path / "results"
    analyzer = Analysis(file_path, time_col="time", data_col="value")
    results = analyzer.run_full_analysis(
        output_dir=str(output_dir), max_breakpoints=2, n_bootstraps=0
    )

    # --- Assertions ---
    # Check that the best model (lowest BIC) was chosen
    assert results["chosen_model"] == "segmented_2bp"
    assert results["n_breakpoints"] == 2
    assert len(results["betas"]) == 3
    assert results["bic"] == 100.0

    # Check that all models were considered
    assert len(results["all_models"]) == 3
    bics = [m["bic"] for m in results["all_models"]]
    assert sorted(bics) == [100.0, 150.0, 200.0]

    # Check that the mocks were called with the correct default ci_method
    mock_fit_standard.assert_called_with(
        ANY,  # frequency
        ANY,  # power
        method="theil-sen",
        ci_method="bootstrap",
        n_bootstraps=0,
        seed=None,
    )
    mock_fit_segmented.assert_any_call(
        ANY,
        ANY,
        n_breakpoints=1,
        p_threshold=0.05,
        ci_method="bootstrap",
        n_bootstraps=0,
        seed=None,
    )


def test_analysis_parametric_ci_is_propagated(tmp_path):
    """
    Test that the `ci_method='parametric'` option is correctly propagated
    through the analysis and reflected in the output.
    """
    time, series = generate_synthetic_series(n_points=100, beta=1.0)
    file_path = create_test_data_file(tmp_path, time, series)

    analyzer = Analysis(file_path, time_col="time", data_col="value")
    results = analyzer.run_full_analysis(
        output_dir=tmp_path, ci_method="parametric"
    )

    assert "summary_text" in results
    assert "(parametric)" in results["summary_text"]
    assert "ci_method" in results
    assert results["ci_method"] == "parametric"

import re
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
def generate_synthetic_series(n_points=100, beta=0.0, seed=42):
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

    analyzer = Analysis(file_path=file_path, time_col="time", data_col="value")

    assert analyzer is not None


@pytest.mark.parametrize(
    "changepoint_mode, min_valid_data, cp_opts, expected_min_size, warn_msg_part",
    [
        # Scenario 1: min_size is too small, should be increased and warn
        ("auto", 30, {"min_size": 20}, 30, "Increasing changepoint min_size from 20 to 30"),
        # Scenario 2: min_size is not provided, should be set and not warn
        ("auto", 30, {}, 30, None),
        # Scenario 3: min_size is sufficient, should not be changed
        ("auto", 30, {"min_size": 40}, 40, None),
        # Scenario 4: Mode is not 'auto', so options should not be changed
        ("manual", 30, {"min_size": 10}, 10, None),
        # Scenario 5: min_size is None, should be set to the required minimum
        ("auto", 25, {"min_size": None}, 25, None),
        # Scenario 6: min_size is an invalid string, should be set and warn
        ("auto", 35, {"min_size": "invalid"}, 35, "Invalid 'min_size' value 'invalid' provided"),
        # Scenario 7: min_size is a float, gets truncated and then increased, with a warning
        ("auto", 35, {"min_size": 12.5}, 35, "Increasing changepoint min_size from 12 to 35"),
    ],
)
def test_analysis_adjusts_changepoint_min_size(
    tmp_path,
    mocker,
    changepoint_mode,
    min_valid_data,
    cp_opts,
    expected_min_size,
    warn_msg_part,
):
    """
    Test that the Analysis class correctly adjusts the changepoint min_size
    to ensure it meets the minimum requirements for spectral analysis, but
    only when in 'auto' mode.
    """
    time, series = generate_synthetic_series(n_points=100)
    file_path = create_test_data_file(tmp_path, time, series)

    # Patch the logger to capture any warnings issued during initialization
    mock_logger = mocker.patch("waterSpec.analysis.logging.getLogger").return_value

    # Initialize the Analysis class with the test parameters
    init_kwargs = {
        "file_path": file_path,
        "time_col": "time",
        "data_col": "value",
        "min_valid_data_points": min_valid_data,
        "changepoint_mode": changepoint_mode,
        "changepoint_options": cp_opts.copy(),  # Pass a copy to avoid mutation issues
    }
    # Add changepoint_index if mode is 'manual' to satisfy constructor validation
    if changepoint_mode == "manual":
        init_kwargs["changepoint_index"] = 50  # A valid index for 100 data points

    analyzer = Analysis(**init_kwargs)

    # Verify the outcome
    if changepoint_mode == "auto":
        assert analyzer.changepoint_options.get("min_size") == expected_min_size
    else:
        # If not in 'auto' mode, the original options should remain untouched
        assert analyzer.changepoint_options.get("min_size") == cp_opts.get("min_size")

    # Check if a warning was logged, as expected
    if warn_msg_part:
        mock_logger.warning.assert_called_once()
        call_args, _ = mock_logger.warning.call_args
        assert warn_msg_part in call_args[0]
    else:
        mock_logger.warning.assert_not_called()
    assert len(analyzer.time) == 100
    assert len(analyzer.data) == 100
    assert analyzer.param_name == "value"


def test_analysis_run_full_analysis_creates_outputs(tmp_path):
    """Test that run_full_analysis creates the plot and summary files."""
    time, series = generate_synthetic_series(n_points=50)
    file_path = create_test_data_file(tmp_path, time, series)
    output_dir = tmp_path / "results"

    analyzer = Analysis(
        file_path=file_path, time_col="time", data_col="value"
    )
    results = analyzer.run_full_analysis(output_dir=str(output_dir), n_bootstraps=10)

    # Check that output files were created
    expected_plot = output_dir / "value_spectrum_plot.png"
    expected_summary = output_dir / "value_summary.txt"
    assert expected_plot.exists()
    assert expected_summary.exists()

    # Check that the summary contains expected text
    summary_text = expected_summary.read_text()
    assert "Analysis for: value" in summary_text
    assert "β =" in summary_text

    # Check that the results dictionary is populated and valid
    assert "summary_text" in results
    assert "Analysis for: value" in results["summary_text"]
    assert "beta" in results or "betas" in results


@pytest.mark.parametrize(
    "known_beta, tolerance",
    [
        (0.0, 0.2),
        (1.0, 0.25),
        (2.0, 0.25),
    ],
)
def test_analysis_with_known_beta(tmp_path, known_beta, tolerance, mocker):
    """Test the full workflow with synthetic data of a known spectral exponent."""
    mocker.patch("waterSpec.fitter.MIN_BOOTSTRAP_SAMPLES", 5)
    time, series = generate_synthetic_series(n_points=100, beta=known_beta)
    file_path = create_test_data_file(tmp_path, time, series)

    analyzer = Analysis(
        file_path=file_path, time_col="time", data_col="value", detrend_method=None
    )

    # Force a standard model fit to reliably test beta estimation
    results = analyzer.run_full_analysis(
        output_dir=tmp_path, n_bootstraps=10, max_breakpoints=0
    )

    assert "beta" in results
    assert results["beta"] == pytest.approx(known_beta, abs=tolerance)
    assert results["analysis_mode"] == "auto"
    assert results["chosen_model"] == "standard"


@patch("waterSpec.analysis.fit_segmented_spectrum")
@patch("waterSpec.analysis.fit_standard_model")
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

    analyzer = Analysis(file_path=file_path, time_col="time", data_col="value")
    results = analyzer.run_full_analysis(output_dir=str(output_dir), n_bootstraps=10)

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
        ValueError,
        match="Not enough valid data points \\(5\\) remaining after "
        "preprocessing. Minimum required: 10.",
    ):
        Analysis(file_path=file_path, time_col="time", data_col="value")


def test_analysis_min_valid_data_points_configurable(tmp_path):
    """Test that the minimum data points threshold can be configured."""
    time = pd.date_range(start="2000-01-01", periods=8, freq="D")
    series = np.random.rand(8)
    file_path = create_test_data_file(tmp_path, time, series)

    # This should fail with the default of 10
    with pytest.raises(ValueError, match="Minimum required: 10"):
        Analysis(file_path=file_path, time_col="time", data_col="value")

    # This should pass with a custom threshold of 8
    analyzer = Analysis(
        file_path=file_path,
        time_col="time",
        data_col="value",
        min_valid_data_points=8,
    )
    assert len(analyzer.data) == 8


@pytest.mark.parametrize("invalid_value", [-10, 0, 9.5, "abc"])
def test_analysis_min_valid_data_points_invalid(tmp_path, invalid_value):
    """Test that invalid values for min_valid_data_points raise an error."""
    time, series = generate_synthetic_series(n_points=20)
    file_path = create_test_data_file(tmp_path, time, series)

    with pytest.raises(
        ValueError, match="`min_valid_data_points` must be a positive integer."
    ):
        Analysis(
            file_path=file_path,
            time_col="time",
            data_col="value",
            min_valid_data_points=invalid_value,
        )


def test_analysis_zero_variance_data(tmp_path):
    """Test the workflow with data that has zero variance."""
    time = pd.date_range(start="2000-01-01", periods=100, freq="D")
    series = np.ones(100) * 5
    file_path = create_test_data_file(tmp_path, time, series)
    output_dir = tmp_path / "results"

    # With normalize=True, this should fail early.
    with pytest.raises(ValueError, match="has zero variance and cannot be normalized"):
        Analysis(
            file_path=file_path,
            time_col="time",
            data_col="value",
            normalize_data=True,
        )

    # Without normalization, it should also fail during initialization.
    with pytest.raises(
        ValueError, match="processed data for 'value' has zero variance"
    ):
        Analysis(
            file_path=file_path,
            time_col="time",
            data_col="value",
            normalize_data=False,
        )


def test_analysis_fap_threshold_is_configurable(tmp_path):
    """Test that the fap_threshold can be configured in run_full_analysis."""
    time, series = generate_synthetic_series()
    file_path = create_test_data_file(tmp_path, time, series)
    output_dir = tmp_path / "results"
    custom_fap = 0.05

    analyzer = Analysis(
        file_path=file_path, time_col="time", data_col="value"
    )
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
    n_points = 100
    time = pd.date_range(start="2000-01-01", periods=n_points, freq="D")
    rng = np.random.default_rng(42)
    noise = rng.normal(0, 0.5, n_points)
    # Add a sine wave with a period of ~50 days
    known_freq_cpd = 1 / 25  # cycles/day
    signal = 2 * np.sin(2 * np.pi * known_freq_cpd * np.arange(n_points))
    series = noise + signal

    file_path = create_test_data_file(tmp_path, time, series)
    output_dir = tmp_path / "results_residual"

    analyzer = Analysis(
        file_path=file_path, time_col="time", data_col="value", detrend_method=None
    )
    results = analyzer.run_full_analysis(
        output_dir=str(output_dir),
        samples_per_peak=10,
        peak_detection_method="fap",
        fap_threshold=0.01,  # Use a low threshold to ensure detection
        n_bootstraps=10,
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
        file_path=file_path,
        time_col="date",
        data_col="concentration",
        censor_strategy="use_detection_limit",
    )
    results = analyzer.run_full_analysis(output_dir=str(output_dir), n_bootstraps=10)

    # Check that the analysis ran successfully and produced valid results
    assert "summary_text" in results
    assert "Analysis failed" not in results["summary_text"]
    assert "beta" in results or "betas" in results
    assert np.isfinite(results.get("beta", np.nan)) or (
        "betas" in results and np.isfinite(results["betas"][0])
    )

    # Check that the output files were created
    expected_plot = output_dir / "concentration_spectrum_plot.png"
    expected_summary = output_dir / "concentration_summary.txt"
    assert expected_plot.exists()
    assert expected_summary.exists()


@patch("waterSpec.analysis.fit_segmented_spectrum")
@patch("waterSpec.analysis.fit_standard_model")
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
    analyzer = Analysis(file_path=file_path, time_col="time", data_col="value")
    results = analyzer.run_full_analysis(
        output_dir=str(output_dir), max_breakpoints=2, n_bootstraps=10
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
        bootstrap_type="block",
        n_bootstraps=10,
        seed=ANY,
        logger=ANY,
    )
    mock_fit_segmented.assert_any_call(
        ANY,
        ANY,
        n_breakpoints=1,
        p_threshold=0.05,
        ci_method="bootstrap",
        bootstrap_type="block",
        n_bootstraps=10,
        seed=ANY,
        logger=ANY,
    )


def test_analysis_parametric_ci_is_propagated(tmp_path):
    """
    Test that the `ci_method='parametric'` option is correctly propagated
    through the analysis and reflected in the output.
    """
    time, series = generate_synthetic_series(n_points=100, beta=1.0)
    file_path = create_test_data_file(tmp_path, time, series)

    analyzer = Analysis(file_path=file_path, time_col="time", data_col="value")
    results = analyzer.run_full_analysis(
        output_dir=tmp_path, ci_method="parametric", n_bootstraps=10
    )

    assert "summary_text" in results
    assert "(parametric)" in results["summary_text"]
    assert "ci_method" in results
    assert results["ci_method"] == "parametric"


@pytest.mark.parametrize(
    "param, value, message",
    [
        ("fit_method", "invalid", "`fit_method` must be 'theil-sen' or 'ols'."),
        ("ci_method", "invalid", "`ci_method` must be 'bootstrap' or 'parametric'."),
        ("n_bootstraps", -1, "`n_bootstraps` must be a non-negative integer."),
        ("fap_threshold", 1.1, "`fap_threshold` must be a float between 0 and 1."),
        ("samples_per_peak", 0, "`samples_per_peak` must be a positive integer."),
        ("fap_method", "invalid", "`fap_method` must be 'baluev' or 'bootstrap'."),
        (
            "normalization",
            "invalid",
            "must be one of 'standard', 'model', 'log', or 'psd'.",
        ),
        ("peak_detection_method", "invalid", "must be 'residual', 'fap', or None."),
        ("peak_fdr_level", 1.1, "`peak_fdr_level` must be a float between 0 and 1."),
        ("p_threshold", -0.1, "`p_threshold` must be a float between 0 and 1."),
        ("max_breakpoints", 3, "`max_breakpoints` must be an integer (0, 1, or 2)."),
    ],
)
def test_run_full_analysis_invalid_parameters(tmp_path, param, value, message):
    """Test that run_full_analysis raises errors for invalid parameters."""
    time, series = generate_synthetic_series(n_points=100)
    file_path = create_test_data_file(tmp_path, time, series)
    analyzer = Analysis(file_path=file_path, time_col="time", data_col="value")

    kwargs = {"output_dir": str(tmp_path)}
    kwargs[param] = value

    with pytest.raises(ValueError, match=re.escape(message)):
        analyzer.run_full_analysis(**kwargs)


@patch("waterSpec.analysis.fit_segmented_spectrum")
@patch("waterSpec.analysis.fit_standard_model")
def test_analysis_handles_total_model_failure(
    mock_fit_standard, mock_fit_segmented, tmp_path
):
    """
    Test that if all model fits fail, run_full_analysis returns a failure
    result dictionary and generates a corresponding output file.
    """
    # --- Mock Configuration ---
    mock_fit_standard.return_value = {
        "bic": np.nan,
        "failure_reason": "Standard failed.",
    }
    mock_fit_segmented.return_value = {
        "bic": np.inf,
        "failure_reason": "Segmented failed.",
    }

    # --- Test Execution ---
    file_path = create_test_data_file(
        tmp_path, pd.date_range("2023", periods=100), np.random.rand(100)
    )
    output_dir = tmp_path / "results"
    analyzer = Analysis(file_path=file_path, time_col="time", data_col="value")
    results = analyzer.run_full_analysis(
        output_dir=str(output_dir), max_breakpoints=1, n_bootstraps=10
    )

    # --- Assertions ---
    assert results["chosen_model_type"] == "failure"
    assert "Analysis failed: All models failed" in results["summary_text"]
    assert "Standard model (0 breakpoints): Standard failed." in results["failure_reason"]
    assert "Segmented model (1 breakpoint(s)): Segmented failed." in results["failure_reason"]

    # Check that the output file reflects the failure
    summary_path = output_dir / "value_summary.txt"
    assert summary_path.exists()
    summary_content = summary_path.read_text()
    assert "Analysis failed: All models failed" in summary_content


@patch("waterSpec.analysis.fit_segmented_spectrum")
@patch("waterSpec.analysis.fit_standard_model")
def test_analysis_retains_failure_reasons_on_partial_success(
    mock_fit_standard, mock_fit_segmented, tmp_path
):
    """
    Test that failure reasons are still reported even if one model succeeds.
    """
    # Mock the standard model to fail
    mock_fit_standard.return_value = {
        "bic": np.inf,
        "failure_reason": "Standard model failed.",
    }
    # Mock the segmented model to succeed
    mock_fit_segmented.return_value = {
        "betas": [0.5, 1.8],
        "breakpoints": [0.01],
        "bic": 100.0,
        "n_breakpoints": 1,
        "residuals": np.array([1, 2, 3]),
        "fitted_log_power": np.array([1, 2, 3]),
        "log_freq": np.array([1, 2, 3]),
        "log_power": np.array([1, 2, 3]),
    }

    file_path = create_test_data_file(
        tmp_path, pd.date_range("2023", periods=100), np.random.rand(100)
    )
    analyzer = Analysis(file_path=file_path, time_col="time", data_col="value")
    results = analyzer.run_full_analysis(output_dir=tmp_path, n_bootstraps=10)

    # The successful model should be chosen
    assert results["chosen_model"] == "segmented_1bp"
    # The failure reason for the other model should be in the results
    assert "failed_model_reasons" in results
    assert len(results["failed_model_reasons"]) == 1
    assert "Standard model (0 breakpoints): Standard model failed." in results["failed_model_reasons"]


def test_analysis_warns_on_ignored_peak_fdr_level(tmp_path, mocker):
    """
    Test that a warning is logged if peak_fdr_level is passed when
    peak_detection_method is 'fap'.
    """
    time, series = generate_synthetic_series()
    file_path = create_test_data_file(tmp_path, time, series)
    analyzer = Analysis(
        file_path=file_path,
        time_col="time",
        data_col="value",
    )

    # Spy on the logger to check for the warning
    spy_logger = mocker.spy(analyzer.logger, "warning")

    analyzer.run_full_analysis(
        output_dir=tmp_path,
        peak_detection_method="fap",
        peak_fdr_level=0.1,  # A non-default value that should be ignored
        n_bootstraps=10,
    )

    spy_logger.assert_any_call(
        "'peak_detection_method' is 'fap', so the 'peak_fdr_level' "
        "parameter is ignored."
    )


def test_peak_detection_ignored_parameter_warning(tmp_path, mocker):
    """
    Test that a warning is issued if fap parameters are passed when using
    the 'residual' peak detection method, as they will be ignored.
    """
    time, series = generate_synthetic_series()
    file_path = create_test_data_file(tmp_path, time, series)
    analyzer = Analysis(
        file_path=file_path,
        time_col="time",
        data_col="value",
    )
    spy_logger = mocker.spy(analyzer.logger, "warning")

    # Test for ignored fap parameters when using 'residual' method
    analyzer.run_full_analysis(
        output_dir=tmp_path,
        peak_detection_method="residual",
        fap_method="bootstrap",  # non-default, should be ignored
        fap_threshold=0.05,  # non-default, should be ignored
        n_bootstraps=10,
    )
    spy_logger.assert_any_call(
        "'peak_detection_method' is 'residual', so 'fap_method' and "
        "'fap_threshold' parameters are ignored."
    )


# --- Tests for NumPy Array Input ---


def test_analysis_initialization_with_numpy_arrays():
    """Test that the Analysis class initializes correctly with NumPy arrays."""
    time_array = np.arange(100, dtype=np.float64)
    data_array = np.random.randn(100)
    analyzer = Analysis(
        time_col="time",
        data_col="value",
        time_array=time_array,
        data_array=data_array,
        input_time_unit="days",
    )
    assert analyzer is not None
    assert len(analyzer.time) == 100
    assert len(analyzer.data) == 100
    assert np.array_equal(analyzer.data, data_array)


def test_analysis_initialization_with_numpy_arrays_and_errors():
    """Test initialization with NumPy arrays including an error array."""
    time_array = np.arange(100, dtype=np.float64)
    data_array = np.random.randn(100)
    error_array = np.full(100, 0.1)
    analyzer = Analysis(
        time_col="time",
        data_col="value",
        error_col="error",
        time_array=time_array,
        data_array=data_array,
        error_array=error_array,
        input_time_unit="days",
    )
    assert analyzer.errors is not None
    assert len(analyzer.errors) == 100
    assert np.array_equal(analyzer.errors, error_array)


def test_analysis_run_full_analysis_with_numpy_arrays(tmp_path):
    """Test a full analysis run using NumPy arrays."""
    time_array = np.arange(200, dtype=np.float64)
    data_array = 5 * np.sin(2 * np.pi * time_array / 25) + np.random.randn(200)
    output_dir = tmp_path / "results_numpy"

    analyzer = Analysis(
        time_col="time",
        data_col="value",
        time_array=time_array,
        data_array=data_array,
        input_time_unit="days",
    )
    results = analyzer.run_full_analysis(output_dir=str(output_dir), n_bootstraps=10)

    expected_plot = output_dir / "value_spectrum_plot.png"
    expected_summary = output_dir / "value_summary.txt"
    assert expected_plot.exists()
    assert expected_summary.exists()
    assert "Analysis for: value" in results["summary_text"]


def test_analysis_raises_error_on_mixed_inputs(tmp_path):
    """Test that a ValueError is raised if mixed data sources are provided."""
    file_path = create_test_data_file(tmp_path, np.arange(10), np.random.randn(10))
    df = pd.DataFrame({"time": np.arange(10), "value": np.random.randn(10)})
    time_array = np.arange(10)
    data_array = np.random.randn(10)

    with pytest.raises(ValueError, match="Please provide only one data source"):
        Analysis(file_path=file_path, dataframe=df, time_col="t", data_col="d")

    with pytest.raises(ValueError, match="Please provide only one data source"):
        Analysis(
            file_path=file_path,
            time_array=time_array,
            data_array=data_array,
            time_col="t",
            data_col="d",
        )

    with pytest.raises(ValueError, match="Please provide only one data source"):
        Analysis(
            dataframe=df,
            time_array=time_array,
            data_array=data_array,
            time_col="t",
            data_col="d",
        )


def test_analysis_raises_error_on_incomplete_array_input():
    """Test that a ValueError is raised for incomplete NumPy array inputs."""
    time_array = np.arange(10)
    data_array = np.random.randn(10)

    with pytest.raises(ValueError, match="A valid data source must be provided"):
        Analysis(time_col="t", data_col="d", time_array=time_array)

    with pytest.raises(ValueError, match="A valid data source must be provided"):
        Analysis(time_col="t", data_col="d", data_array=data_array)


def test_analysis_raises_error_for_loess_with_errors_and_no_bootstrap():
    """
    Test that Analysis.__init__ raises a ValueError if LOESS detrending is
    requested with error propagation but without specifying n_bootstrap.
    """
    time, series, errors = (
        np.arange(100, dtype=float),
        np.random.randn(100),
        np.full(100, 0.1),
    )

    # This configuration is invalid and should fail at initialization
    with pytest.raises(
        ValueError,
        match="When using 'loess' detrending with error data, `n_bootstrap` must be > 0",
    ):
        Analysis(
            time_col="time",
            data_col="value",
            error_col="error",
            time_array=time,
            data_array=series,
            error_array=errors,
            detrend_method="loess",
            input_time_unit="days",
            # detrend_options is omitted, so n_bootstrap defaults to 0
        )

    # For completeness, test with n_bootstrap=0 explicitly
    with pytest.raises(
        ValueError,
        match="When using 'loess' detrending with error data, `n_bootstrap` must be > 0",
    ):
        Analysis(
            time_col="time",
            data_col="value",
            error_col="error",
            time_array=time,
            data_array=series,
            error_array=errors,
            detrend_method="loess",
            detrend_options={"n_bootstrap": 0},
            input_time_unit="days",
        )

    # This configuration should succeed without error
    analyzer = Analysis(
        time_col="time",
        data_col="value",
        error_col="error",
        time_array=time,
        data_array=series,
        error_array=errors,
        detrend_method="loess",
        detrend_options={"n_bootstrap": 10},
        input_time_unit="days",
    )
    assert analyzer is not None
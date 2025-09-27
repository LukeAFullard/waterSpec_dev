from unittest.mock import patch
import warnings
import pytest
import pandas as pd
import numpy as np

from waterSpec import Analysis
from waterSpec.preprocessor import preprocess_data
from waterSpec.data_loader import load_data

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

# --- Tests for Gaps in analysis.py ---

def test_analysis_warns_for_ignored_fap_params(tmp_path, caplog):
    """
    Test that a warning is logged when peak_detection_method='residual'
    but FAP parameters are provided, since they will be ignored.
    """
    file_path = "examples/sample_data.csv"
    output_dir = tmp_path / "results"

    analyzer = Analysis(file_path, time_col="timestamp", data_col="concentration", verbose=True)

    analyzer.run_full_analysis(
        output_dir=str(output_dir),
        peak_detection_method="residual",
        fap_method="bootstrap",  # Non-default value to trigger warning
        n_bootstraps=10,
    )
    assert "'fap_method' and 'fap_threshold' parameters are ignored" in caplog.text

def test_analysis_raises_error_for_unknown_peak_method(tmp_path):
    """
    Test that a ValueError is raised for an unknown peak_detection_method.
    """
    file_path = "examples/sample_data.csv"
    output_dir = tmp_path / "results"
    analyzer = Analysis(
        file_path, time_col="timestamp", data_col="concentration", verbose=True
    )

    with pytest.raises(
        ValueError, match="`peak_detection_method` must be 'residual', 'fap', or None."
    ):
        analyzer.run_full_analysis(
            output_dir=str(output_dir),
            peak_detection_method="invalid_method",
            n_bootstraps=10,
        )

def test_analysis_raises_error_for_insufficient_data_post_processing(tmp_path):
    """
    Test that a ValueError is raised if data has < 10 valid points
    AFTER preprocessing.
    """
    time = pd.date_range(start="2000-01-01", periods=20, freq="D")
    series = [1, 2, 3, 4, 5] + ["<1"] * 15
    file_path = create_test_data_file(tmp_path, time, series)

    with pytest.raises(
        ValueError,
        match=r"Not enough valid data points \(5\) remaining after preprocessing. Minimum required: 10.",
    ):
        Analysis(
            file_path,
            time_col="time",
            data_col="value",
            censor_strategy="drop",
            censor_options={"censor_symbol": "<"},
        )

# --- Tests for Gaps in preprocessor.py ---

def test_preprocess_with_loess_detrend(tmp_path):
    """
    Test that the loess detrending method runs without error.
    """
    time = np.arange(100)
    series = pd.Series(np.sin(time / 10) + np.random.rand(100))

    processed_data, _ = preprocess_data(
        series, time, detrend_method="loess"
    )
    assert not np.array_equal(series.values, processed_data)
    assert len(series) == len(processed_data)

def test_preprocess_raises_for_unknown_censor_strategy():
    """
    Test that a ValueError is raised for an unknown censor strategy.
    """
    time = np.arange(20)
    # The series must be non-numeric for the strategy check to be reached.
    series = pd.Series([str(x) for x in np.random.rand(20)])
    with pytest.raises(ValueError, match="Invalid censor strategy"):
        preprocess_data(series, time, censor_strategy="invalid")

def test_preprocess_warns_for_unknown_detrend_method():
    """
    Test that a UserWarning is raised for an unknown detrend method.
    """
    time = np.arange(20)
    series = pd.Series(np.random.rand(20))
    with pytest.warns(UserWarning, match="Unknown detrending method 'invalid'"):
        preprocess_data(series, time, detrend_method="invalid")

# --- Tests for Gaps in data_loader.py ---

def test_load_data_warns_on_missing_error_column(tmp_path):
    """
    Test that loading data with a specified but missing error column
    issues a UserWarning.
    """
    time = pd.date_range(start="2000-01-01", periods=20, freq="D")
    series = np.random.rand(20)
    file_path = create_test_data_file(tmp_path, time, series)

    with pytest.warns(UserWarning, match="Error column 'non_existent_col' not found"):
        load_data(
            file_path, time_col="time", data_col="value", error_col="non_existent_col"
        )

def test_load_data_raises_on_unparseable_time(tmp_path):
    """
    Test that a ValueError is raised when time values cannot be parsed.
    """
    time = ["not_a_date"] * 20
    series = np.random.rand(20)
    file_path = create_test_data_file(tmp_path, time, series)

    with pytest.raises(ValueError, match="could not be parsed as datetime objects"):
        load_data(file_path, time_col="time", data_col="value")

# --- Tests for Gaps in fitter.py ---

def test_fit_segmented_spectrum_with_zero_variance_residuals(tmp_path):
    """
    Test that segmented fitting handles zero-variance residuals gracefully.
    """
    from waterSpec.fitter import fit_segmented_spectrum
    freq = np.logspace(-3, 0, 100)
    power = 1e-5 * freq**-1
    with patch('statsmodels.regression.linear_model.OLS.fit'):
        # This test is primarily to ensure no unhandled exceptions occur.
        # The result might not be meaningful, but it shouldn't crash.
        results = fit_segmented_spectrum(freq, power, n_breakpoints=1, logger=None)
        assert isinstance(results, dict)


# --- Tests for Gaps in interpreter.py ---

def test_interpret_results_no_significant_peaks(tmp_path):
    """
    Test that the interpreter correctly formats the summary when no
    significant peaks are found.
    """
    from waterSpec.interpreter import interpret_results
    fit_results = {
        "beta": 1.0, "betas": [1.0], "beta_ses": [0.1], "n_breakpoints": 0,
        "chosen_model": "standard", "significant_peaks": []
    }
    interp = interpret_results(fit_results, param_name="Test")
    assert "No significant periodicities were found" in interp["summary_text"]

def test_interpret_results_fap_peak_detection(tmp_path):
    """
    Test that the interpreter includes FAP details when that peak detection
    method is used.
    """
    from waterSpec.interpreter import interpret_results
    fit_results = {
        "beta": 1.0, "betas": [1.0], "beta_ses": [0.1], "n_breakpoints": 0,
        "chosen_model": "standard",
        "significant_peaks": [{"frequency": 0.1, "period": 10, "power": 100}],
        "fap_level": 50, "fap_threshold": 0.01
    }
    interp = interpret_results(fit_results, param_name="Test")
    assert "Significant Periodicities Found (at 1.0% FAP Level)" in interp["summary_text"]

# --- Tests for Gaps in spectral_analyzer.py ---

def test_find_peaks_via_residuals_no_fit(tmp_path):
    """
    Test that peak detection via residuals raises ValueError if fit failed.
    """
    from waterSpec.spectral_analyzer import find_peaks_via_residuals
    fit_results = {"beta": np.nan}
    with pytest.raises(ValueError, match="fit_results is missing required keys"):
        find_peaks_via_residuals(fit_results, ci=95)

# --- Tests for Gaps in frequency_generator.py ---

def test_generate_frequency_grid_linear_method(tmp_path):
    """
    Test the 'linear' grid generation method.
    """
    from waterSpec.frequency_generator import generate_frequency_grid
    time = np.arange(100)
    freq = generate_frequency_grid(time, num_points=100, grid_type='linear')
    assert len(freq) == 100
    assert np.all(np.isclose(np.diff(freq), np.diff(freq)[0]))

# --- Final batch of tests for remaining gaps ---

def test_preprocess_handles_numpy_array_input():
    """
    Test that preprocess_data can accept a numpy array instead of a pandas Series.
    """
    time = np.arange(20)
    series = np.random.rand(20) # Pass as numpy array
    processed_data, _ = preprocess_data(series, time)
    assert isinstance(processed_data, np.ndarray)

def test_preprocess_handles_non_numeric_censored_values(tmp_path):
    """
    Test that non-numeric censored values are handled and that the analysis
    fails if the number of valid points drops below the minimum.
    """
    time = pd.date_range("2023-01-01", periods=5)
    # The value "ND" will be converted to NaN, leaving 4 valid points.
    series = pd.Series([1, 2, 1, "ND", 5])
    file_path = create_test_data_file(tmp_path, time, series.astype(str))

    with pytest.raises(
        ValueError,
        match=r"Not enough valid data points \(4\) remaining after preprocessing. Minimum required: 10.",
    ):
        Analysis(
            file_path,
            time_col="time",
            data_col="value",
        )

def test_load_data_all_nan_data(tmp_path):
    """
    Test loading a file where all data values are NaN, which should result
    in a ValueError after all rows are dropped.
    """
    time = pd.date_range("2023", periods=10)
    series = [np.nan] * 10
    file_path = create_test_data_file(tmp_path, time, series)

    with pytest.raises(ValueError, match="No valid data remains after removing rows"):
        load_data(file_path, time_col="time", data_col="value")


def test_interpreter_formats_two_breakpoints(tmp_path):
    """
    Test that the interpreter correctly formats the summary for a two-breakpoint model.
    """
    from waterSpec.interpreter import interpret_results
    fit_results = {
        "betas": [0.5, 1.5, 0.8], "beta_ses": [0.1, 0.1, 0.1],
        "n_breakpoints": 2, "chosen_model": "segmented_2bp",
        "breakpoints": [0.1, 0.5], "significant_peaks": []
    }
    interp = interpret_results(fit_results, param_name="Test")
    assert "--- Breakpoint 1 @ " in interp["summary_text"]
    assert "--- Breakpoint 2 @ " in interp["summary_text"]
    assert "β1 = 0.50" in interp["summary_text"]
    assert "β2 = 1.50" in interp["summary_text"]
    assert "β3 = 0.80" in interp["summary_text"]
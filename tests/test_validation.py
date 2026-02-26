import pytest
from waterSpec.utils import validate_run_parameters

def test_validate_valid_parameters():
    # Calling with valid parameters should not raise
    validate_run_parameters(
        fit_method="theil-sen",
        ci_method="bootstrap",
        bootstrap_type="block",
        n_bootstraps=100,
        fap_threshold=0.01,
        samples_per_peak=5,
        fap_method="baluev",
        normalization="standard",
        peak_detection_method="fap",
        peak_fdr_level=0.05,
        p_threshold=0.05,
        max_breakpoints=1,
        nyquist_factor=1.0,
        max_freq=None,
        haar_statistic="mean",
        haar_percentile=None,
    )

def test_validate_optional_parameters_ignored():
    # Calling with no parameters should not raise (all default to _NOT_PROVIDED)
    validate_run_parameters()

def test_validate_fit_method():
    with pytest.raises(ValueError, match="`fit_method` must be 'theil-sen' or 'ols'"):
        validate_run_parameters(fit_method="invalid")

def test_validate_ci_method():
    with pytest.raises(ValueError, match="`ci_method` must be 'bootstrap' or 'parametric'"):
        validate_run_parameters(ci_method="invalid")

def test_validate_bootstrap_type():
    with pytest.raises(ValueError, match="`bootstrap_type` must be"):
        validate_run_parameters(bootstrap_type="invalid")

def test_validate_n_bootstraps():
    with pytest.raises(ValueError, match="`n_bootstraps` must be a non-negative integer"):
        validate_run_parameters(n_bootstraps=-1)
    with pytest.raises(ValueError, match="`n_bootstraps` must be a non-negative integer"):
        validate_run_parameters(n_bootstraps="100")

def test_validate_fap_threshold():
    with pytest.raises(ValueError, match="`fap_threshold` must be a float between 0 and 1"):
        validate_run_parameters(fap_threshold=1.5)
    with pytest.raises(ValueError, match="`fap_threshold` must be a float between 0 and 1"):
        validate_run_parameters(fap_threshold=0)

def test_validate_samples_per_peak():
    with pytest.raises(ValueError, match="`samples_per_peak` must be a positive integer"):
        validate_run_parameters(samples_per_peak=0)

def test_validate_fap_method():
    with pytest.raises(ValueError, match="`fap_method` must be 'baluev' or 'bootstrap'"):
        validate_run_parameters(fap_method="invalid")

def test_validate_normalization():
    with pytest.raises(ValueError, match="`normalization` must be one of"):
        validate_run_parameters(normalization="invalid")

def test_validate_peak_detection_method():
    with pytest.raises(ValueError, match="`peak_detection_method` must be"):
        validate_run_parameters(peak_detection_method="invalid")
    # None is valid
    validate_run_parameters(peak_detection_method=None)

def test_validate_peak_fdr_level():
    with pytest.raises(ValueError, match="`peak_fdr_level` must be a float between 0 and 1"):
        validate_run_parameters(peak_fdr_level=1.5)

def test_validate_p_threshold():
    with pytest.raises(ValueError, match="`p_threshold` must be a float between 0 and 1"):
        validate_run_parameters(p_threshold=1.5)

def test_validate_max_breakpoints():
    with pytest.raises(ValueError, match="`max_breakpoints` must be an integer"):
        validate_run_parameters(max_breakpoints=3)

def test_validate_nyquist_factor():
    with pytest.raises(ValueError, match="`nyquist_factor` must be a positive number"):
        validate_run_parameters(nyquist_factor=-1)

def test_validate_max_freq():
    with pytest.raises(ValueError, match="`max_freq`, if provided, must be a positive number"):
        validate_run_parameters(max_freq=-1)
    # None is valid (if provided explicitly or defaulted)
    validate_run_parameters(max_freq=None)

def test_validate_haar_statistic():
    with pytest.raises(ValueError, match="`haar_statistic` must be"):
        validate_run_parameters(haar_statistic="invalid")

def test_validate_haar_percentile_dependency():
    # If haar_statistic is percentile, haar_percentile must be provided
    with pytest.raises(ValueError, match="`haar_percentile` must be provided"):
        validate_run_parameters(haar_statistic="percentile")

    # If haar_statistic is percentile, haar_percentile must be provided and not None
    # Note: If we pass None, it is treated as provided as None.
    with pytest.raises(ValueError, match="`haar_percentile` must be provided"):
        validate_run_parameters(haar_statistic="percentile", haar_percentile=None)

    # Valid case
    validate_run_parameters(haar_statistic="percentile", haar_percentile=90)

def test_validate_haar_percentile_range():
    with pytest.raises(ValueError, match="`haar_percentile` must be between 0 and 100"):
        validate_run_parameters(haar_percentile=-1)
    with pytest.raises(ValueError, match="`haar_percentile` must be between 0 and 100"):
        validate_run_parameters(haar_percentile=101)

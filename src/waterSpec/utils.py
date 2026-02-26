import numpy as np
import re


def make_rng(seed=None):
    """
    Creates a random number generator instance with consistent seed handling.

    This utility ensures that various seed types are handled correctly,
    preventing issues with correlated random sequences and improving
    reproducibility.

    Args:
        seed (None, int, np.random.SeedSequence, np.random.Generator):
            The seed to initialize the RNG.
            - If None, a new, unseeded generator is created.
            - If an int, it is used as the seed.
            - If a SeedSequence, it is used to spawn the generator.
            - If a Generator instance, it is returned directly.

    Returns:
        np.random.Generator: A NumPy random number generator instance.

    Raises:
        TypeError: If the seed is of an invalid type.
    """
    if isinstance(seed, np.random.Generator):
        return seed
    elif isinstance(seed, np.random.SeedSequence):
        return np.random.default_rng(seed)
    elif isinstance(seed, int):
        return np.random.default_rng(seed)
    elif seed is None:
        return np.random.default_rng()
    else:
        raise TypeError(
            f"Invalid seed type: {type(seed)}. Must be an int, "
            "np.random.SeedSequence, or np.random.Generator."
        )


def spawn_generators(master_seed, n_children):
    import numpy as np
    from numpy.random import SeedSequence
    if isinstance(master_seed, SeedSequence):
        ss = master_seed
    else:
        ss = SeedSequence(master_seed)
    return [np.random.default_rng(s) for s in ss.spawn(n_children)]


def sanitize_filename(filename):
    """Sanitizes a string to be a valid filename."""
    s = str(filename).strip().replace(" ", "_")
    s = re.sub(r"(?u)[^-\w.]", "", s)
    return s


_NOT_PROVIDED = object()


def validate_run_parameters(
    fit_method=_NOT_PROVIDED,
    ci_method=_NOT_PROVIDED,
    bootstrap_type=_NOT_PROVIDED,
    n_bootstraps=_NOT_PROVIDED,
    fap_threshold=_NOT_PROVIDED,
    samples_per_peak=_NOT_PROVIDED,
    fap_method=_NOT_PROVIDED,
    normalization=_NOT_PROVIDED,
    peak_detection_method=_NOT_PROVIDED,
    peak_fdr_level=_NOT_PROVIDED,
    p_threshold=_NOT_PROVIDED,
    max_breakpoints=_NOT_PROVIDED,
    nyquist_factor=_NOT_PROVIDED,
    max_freq=_NOT_PROVIDED,
    haar_statistic=_NOT_PROVIDED,
    haar_percentile=_NOT_PROVIDED,
    haar_percentile_method=_NOT_PROVIDED,
):
    """
    Validates common parameters for spectral analysis runs.
    Parameters passed as _NOT_PROVIDED are skipped, unless context requires them.
    """
    if fit_method is not _NOT_PROVIDED:
        if fit_method not in ["theil-sen", "ols"]:
            raise ValueError("`fit_method` must be 'theil-sen' or 'ols'.")

    if ci_method is not _NOT_PROVIDED:
        if ci_method not in ["bootstrap", "parametric"]:
            raise ValueError("`ci_method` must be 'bootstrap' or 'parametric'.")

    if bootstrap_type is not _NOT_PROVIDED:
        if bootstrap_type not in ["pairs", "residuals", "block", "wild"]:
            raise ValueError(
                "`bootstrap_type` must be 'pairs', 'residuals', 'block', or 'wild'."
            )

    if n_bootstraps is not _NOT_PROVIDED:
        if not isinstance(n_bootstraps, int) or n_bootstraps < 0:
            raise ValueError("`n_bootstraps` must be a non-negative integer.")

    if fap_threshold is not _NOT_PROVIDED:
        if not (isinstance(fap_threshold, float) and 0 < fap_threshold < 1):
            raise ValueError("`fap_threshold` must be a float between 0 and 1.")

    if samples_per_peak is not _NOT_PROVIDED:
        if not isinstance(samples_per_peak, int) or samples_per_peak <= 0:
            raise ValueError("`samples_per_peak` must be a positive integer.")

    if fap_method is not _NOT_PROVIDED:
        if fap_method not in ["baluev", "bootstrap"]:
            raise ValueError("`fap_method` must be 'baluev' or 'bootstrap'.")

    if normalization is not _NOT_PROVIDED:
        if normalization not in ["standard", "model", "log", "psd"]:
            raise ValueError(
                "`normalization` must be one of 'standard', 'model', 'log', or 'psd'."
            )

    if peak_detection_method is not _NOT_PROVIDED:
        if peak_detection_method not in ["residual", "fap", None]:
            raise ValueError(
                "`peak_detection_method` must be 'residual', 'fap', or None."
            )

    if peak_fdr_level is not _NOT_PROVIDED:
        if not (isinstance(peak_fdr_level, float) and 0 < peak_fdr_level < 1):
            raise ValueError("`peak_fdr_level` must be a float between 0 and 1.")

    if p_threshold is not _NOT_PROVIDED:
        if not (isinstance(p_threshold, float) and 0 < p_threshold < 1):
            raise ValueError("`p_threshold` must be a float between 0 and 1.")

    if max_breakpoints is not _NOT_PROVIDED:
        if max_breakpoints not in [0, 1, 2]:
            raise ValueError("`max_breakpoints` must be an integer (0, 1, or 2).")

    if nyquist_factor is not _NOT_PROVIDED:
        if not isinstance(nyquist_factor, (int, float)) or nyquist_factor <= 0:
            raise ValueError("`nyquist_factor` must be a positive number.")

    if max_freq is not _NOT_PROVIDED:
        if max_freq is not None and (not isinstance(max_freq, (int, float)) or max_freq <= 0):
            raise ValueError("`max_freq`, if provided, must be a positive number.")

    # Haar specific checks
    if haar_statistic is not _NOT_PROVIDED:
        if haar_statistic not in ["mean", "median", "percentile"]:
            raise ValueError("`haar_statistic` must be 'mean', 'median', or 'percentile'.")

        if haar_statistic == "percentile":
            if haar_percentile is _NOT_PROVIDED or haar_percentile is None:
                raise ValueError("`haar_percentile` must be provided if `haar_statistic` is 'percentile'.")

    if haar_percentile is not _NOT_PROVIDED and haar_percentile is not None:
         if not (0 <= haar_percentile <= 100):
             raise ValueError("`haar_percentile` must be between 0 and 100.")

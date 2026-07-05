"""
haar_periodicity.py
====================

Optional, standalone utilities for correcting the Haar (RMS / S2) structure
function for the known contribution of deterministic periodic signals
(e.g. an annual temperature cycle), WITHOUT modifying the raw input time
series.

Why this module exists
-----------------------
A strong deterministic periodicity (e.g. an annual cycle) smears its power
across a broad *range* of lags in a Haar/wavelet structure function -
roughly from a fraction of the period up to several multiples of it - unlike
a Lomb-Scargle periodogram, where the same signal concentrates into a small
number of narrow frequency bins that can simply be excluded from a continuum
fit. Deseasonalizing the raw data before running Haar analysis is one fix
(see docs/user_guide/HAAR_GUIDE.md); this module implements a second, complementary
approach that operates entirely on the *structure function itself*:

  1. Reconstruct a synthetic signal containing ONLY the known periodic
     component(s), evaluated at the exact same (possibly irregular)
     timestamps as the real data.
  2. Run that synthetic signal through the identical Haar structure-function
     pipeline (same lag grid, same window settings) to obtain the periodic
     component's own contribution to the RMS structure function, S2_periodic.
  3. Because variances of uncorrelated processes add, subtract
     S2_periodic^2 from the measured S2_measured^2 in quadrature to recover
     an estimate of the stochastic-process-only structure function.

Design notes
------------
* This module has NO import dependency on the rest of the `waterSpec`
  package (only numpy, and optionally scipy for weighted least squares).
  `haar_analysis.py` wires it together by passing in the fluctuation
  function (`calculate_haar_fluctuations`) as an argument. This keeps the
  module independently testable and avoids any circular-import risk.
* Nothing in this module is invoked automatically. Every entry point here is
  opt-in and must be explicitly requested via `HaarAnalysis.run(...)`.
* The correction is only mathematically defensible for the RMS-based
  structure function (S2), because only variances of uncorrelated random
  variables add linearly. It is NOT valid for the mean-absolute-fluctuation
  statistic (S1). Callers are responsible for enforcing `aggregation="rms"`
  (this is enforced in `haar_analysis.HaarAnalysis.run`).
* Selecting *which* periods to remove is deliberately left under the
  caller's explicit control - see `list_period_candidates` below - to avoid
  silently blending together near-duplicate periods that a Lomb-Scargle
  periodogram commonly reports around a single true cycle (e.g. several
  "significant" peaks between 11 and 13 months that are really just
  spectral leakage/sidebands of one annual cycle).
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# 1. Turning a (possibly messy) list of Lomb-Scargle peaks into a clean,
#    user-inspectable, user-editable list of candidate periods.
# ---------------------------------------------------------------------------

@dataclass
class PeriodCluster:
    """One group of near-duplicate periods, collapsed to one representative value."""
    representative_period: float
    member_periods: List[float] = field(default_factory=list)
    member_frequencies: List[float] = field(default_factory=list)
    member_strengths: List[float] = field(default_factory=list)

    def __repr__(self) -> str:  # pragma: no cover - cosmetic only
        members = ", ".join(f"{p:.4g}" for p in self.member_periods)
        return (
            f"PeriodCluster(representative={self.representative_period:.4g}, "
            f"n_members={len(self.member_periods)}, members=[{members}])"
        )


def consolidate_periods(
    periods: Sequence[float],
    strengths: Optional[Sequence[float]] = None,
    tolerance: float = 0.15,
    representative: str = "strongest",
) -> List[PeriodCluster]:
    """
    Groups periods that are within a relative `tolerance` of one another into
    clusters, so that near-duplicate periods (e.g. spectral-leakage sidebands
    of a single true annual cycle) are not treated as independent signals.

    Args:
        periods: Candidate periods (any positive time unit), NOT necessarily sorted.
        strengths: Optional per-period "strength" (e.g. periodogram power, or
            1/FAP). Used to pick the representative value per cluster when
            representative="strongest". If None, falls back to "median".
        tolerance: Relative difference threshold for merging two periods into
            the same cluster (default 0.15, i.e. 15%).
        representative: "strongest" (pick the highest-strength member; requires
            `strengths`), "median" (pick the median period in the cluster), or
            "largest" (pick the largest period in the cluster - conservative,
            biases towards the longer/fundamental period of a harmonic family).

    Returns:
        List[PeriodCluster], sorted by representative_period descending.

    Note:
        This is a heuristic convenience tool, NOT a substitute for domain
        knowledge. Always inspect the output (e.g. via `list_period_candidates`)
        before using it to drive a correction - see module docstring.
    """
    if representative not in ("strongest", "median", "largest"):
        raise ValueError("representative must be 'strongest', 'median', or 'largest'")
    if representative == "strongest" and strengths is None:
        warnings.warn(
            "representative='strongest' requested but no strengths were "
            "provided; falling back to representative='median'.",
            UserWarning,
        )
        representative = "median"

    periods = np.asarray(periods, dtype=float)
    if len(periods) == 0:
        return []

    freqs = 1.0 / periods
    if strengths is None:
        strengths = np.full_like(periods, np.nan)
    else:
        strengths = np.asarray(strengths, dtype=float)

    order = np.argsort(periods)[::-1]  # largest period first
    periods, freqs, strengths = periods[order], freqs[order], strengths[order]

    clusters: List[PeriodCluster] = []
    for p, f, s in zip(periods, freqs, strengths):
        if clusters and abs(p - clusters[-1].member_periods[-1]) / clusters[-1].member_periods[-1] <= tolerance:
            clusters[-1].member_periods.append(float(p))
            clusters[-1].member_frequencies.append(float(f))
            clusters[-1].member_strengths.append(float(s))
        else:
            clusters.append(
                PeriodCluster(
                    representative_period=float(p),
                    member_periods=[float(p)],
                    member_frequencies=[float(f)],
                    member_strengths=[float(s)],
                )
            )

    # Now compute the representative value per the requested rule.
    for c in clusters:
        if representative == "median":
            c.representative_period = float(np.median(c.member_periods))
        elif representative == "largest":
            c.representative_period = float(np.max(c.member_periods))
        elif representative == "strongest":
            best_idx = int(np.nanargmax(c.member_strengths))
            c.representative_period = float(c.member_periods[best_idx])

    return clusters


def list_period_candidates(
    significant_peaks: Sequence[Dict],
    tolerance: float = 0.15,
    min_period: Optional[float] = None,
    max_period: Optional[float] = None,
    representative: str = "strongest",
) -> List[PeriodCluster]:
    """
    User-facing PREVIEW helper: turns a list of significant-peak dicts (as
    returned by waterSpec's Lomb-Scargle peak detection, e.g.
    `results['significant_peaks']`, each containing at least a 'frequency'
    key and optionally 'power' or 'residual') into a clustered, inspectable
    list of period candidates for periodicity correction.

    This function does NOT get called automatically anywhere in the
    pipeline. Call it yourself, inspect/print the result, and only then
    decide which representative periods to pass to
    `HaarAnalysis.run(correct_periodicity=True, periodic_periods=[...])`.
    This two-step "preview then commit" workflow is deliberate: it stops
    near-duplicate Lomb-Scargle sidebands (e.g. several "significant" peaks
    between 11 and 13 months that all describe the same annual cycle) from
    being silently blended or double-counted, and puts the final choice of
    which lags to correct for entirely in your hands.

    Example:
        >>> clusters = list_period_candidates(results['significant_peaks'])
        >>> for c in clusters:
        ...     print(c)
        PeriodCluster(representative=31556926, n_members=9, members=[...])
        PeriodCluster(representative=15778463, n_members=1, members=[...])
        >>> chosen_periods = [c.representative_period for c in clusters[:2]]
        >>> haar.run(correct_periodicity=True, periodic_periods=chosen_periods, aggregation="rms")

    Args:
        significant_peaks: List of dicts with a 'frequency' key (1/time_unit),
            and optionally 'power' or 'residual' used as a strength metric
            for representative='strongest'.
        tolerance: Relative tolerance for merging near-duplicate periods
            (default 0.15 = 15%).
        min_period / max_period: Optional bounds (in the series' time units)
            to discard candidate periods outside a physically meaningful
            range before clustering.
        representative: See `consolidate_periods`.

    Returns:
        List[PeriodCluster], sorted by representative_period descending.
    """
    periods, strengths = [], []
    for p in significant_peaks:
        freq = p.get("frequency")
        if freq is None or freq <= 0:
            continue
        period = 1.0 / freq
        if min_period is not None and period < min_period:
            continue
        if max_period is not None and period > max_period:
            continue
        periods.append(period)
        # Prefer 'power' (higher = stronger); fall back to 1/fap; then residual.
        if "power" in p:
            strengths.append(p["power"])
        elif "fap" in p and p["fap"] > 0:
            strengths.append(1.0 / p["fap"])
        elif "residual" in p:
            strengths.append(p["residual"])
        else:
            strengths.append(np.nan)

    return consolidate_periods(
        periods, strengths=strengths, tolerance=tolerance, representative=representative
    )


# ---------------------------------------------------------------------------
# 2. Reconstructing the deterministic periodic signal at the real timestamps.
# ---------------------------------------------------------------------------

def reconstruct_periodic_signal(
    time: np.ndarray,
    data: np.ndarray,
    periods: Sequence[float],
    weights: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Fits a joint multi-harmonic sinusoidal model to `data` at the given
    (possibly irregular) `time` points via (weighted) least squares, and
    returns the deterministic reconstruction evaluated at those same points.

    Using a single *joint* least-squares fit across all requested periods
    (rather than independently fitting and summing one Lomb-Scargle
    single-frequency model per period) avoids amplitude leakage between
    periods that are not perfectly orthogonal on an irregular time grid
    (e.g. an annual fundamental and its semi-annual harmonic).

    Args:
        time: Observation times (irregular sampling is fine).
        data: Observed values, same length as `time`. NaNs are ignored in
            the fit (but the returned reconstruction is still evaluated at
            every input time).
        periods: Positive periods (same time units as `time`) to include as
            fundamental + harmonic terms. Each period contributes one
            cosine and one sine term (2 free parameters) plus a single
            shared constant offset.
        weights: Optional per-point weights (e.g. 1/sigma^2) for weighted
            least squares. If None, ordinary least squares is used.

    Returns:
        (reconstruction, coefficients):
            reconstruction: np.ndarray, same length as `time`, the fitted
                deterministic (periodic) signal, mean-subtracted.
            coefficients: np.ndarray of shape (1 + 2*len(periods),), the raw
                fitted [offset, cos_1, sin_1, cos_2, sin_2, ...] coefficients.
    """
    periods = np.atleast_1d(np.asarray(periods, dtype=float))
    if len(periods) == 0:
        raise ValueError("`periods` must contain at least one positive period.")
    if np.any(periods <= 0):
        raise ValueError("All periods must be strictly positive.")

    # Warn (but do not fail) if two requested periods are near-duplicates -
    # this usually means the caller should have consolidated them first.
    if len(periods) > 1:
        sorted_p = np.sort(periods)[::-1]
        rel_diffs = np.abs(np.diff(sorted_p)) / sorted_p[:-1]
        if np.any(rel_diffs < 0.05):
            warnings.warn(
                "Two or more requested `periods` are within 5% of each "
                "other. This can make the joint harmonic fit ill-conditioned "
                "and usually indicates unconsolidated Lomb-Scargle sidebands "
                "of the same underlying cycle. Consider using "
                "`list_period_candidates` to consolidate them first.",
                UserWarning,
            )

    time = np.asarray(time, dtype=float)
    data = np.asarray(data, dtype=float)
    valid = np.isfinite(data)
    if valid.sum() < 2 * len(periods) + 1:
        raise ValueError(
            f"Not enough valid data points ({valid.sum()}) to fit "
            f"{len(periods)} harmonic period(s) ({2 * len(periods) + 1} "
            "parameters required)."
        )

    cols = [np.ones_like(time)]
    for T in periods:
        w = 2.0 * np.pi / T
        cols.append(np.cos(w * time))
        cols.append(np.sin(w * time))
    X = np.column_stack(cols)

    if weights is not None:
        weights = np.asarray(weights, dtype=float)
        sqrt_w = np.sqrt(np.clip(weights[valid], 0, None))
        Xw = X[valid] * sqrt_w[:, None]
        yw = data[valid] * sqrt_w
        coeffs, *_ = np.linalg.lstsq(Xw, yw, rcond=None)
    else:
        coeffs, *_ = np.linalg.lstsq(X[valid], data[valid], rcond=None)

    reconstruction = X @ coeffs
    # Subtract the mean: the offset term cancels in any Haar fluctuation
    # (difference of two window means) regardless, but demeaning makes the
    # returned series directly interpretable/plottable on its own.
    reconstruction = reconstruction - np.mean(reconstruction)

    return reconstruction, coeffs


# ---------------------------------------------------------------------------
# 3. Computing the periodic component's structure function, and correcting
#    the measured one for it.
# ---------------------------------------------------------------------------

def compute_periodic_structure_function(
    time: np.ndarray,
    data: np.ndarray,
    periods: Sequence[float],
    lag_times: np.ndarray,
    fluctuation_func: Callable,
    fluctuation_kwargs: Dict,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Builds the synthetic periodic-only reconstruction and runs it through
    the SAME Haar fluctuation pipeline used for the real data, on the exact
    same lag grid, to get the periodic component's own RMS structure
    function S2_periodic(lag).

    Args:
        time, data: The real observation times/values (data is only used to
            fit the harmonic amplitudes/phases; see `reconstruct_periodic_signal`).
        periods: Explicit periods to reconstruct (already consolidated/chosen
            by the caller - see `list_period_candidates`).
        lag_times: The EXACT lag grid used for the real data's structure
            function, so the two are directly comparable point-for-point.
        fluctuation_func: The structure-function calculator to reuse -
            in practice, `waterSpec.haar_analysis.calculate_haar_fluctuations`.
            Injected as a parameter (rather than imported) so this module
            has no dependency on the rest of the package.
        fluctuation_kwargs: Keyword arguments forwarded to `fluctuation_func`
            (overlap, overlap_step_fraction, min_samples_per_window,
            statistic, percentile, percentile_method). `aggregation` is
            forced to "rms" regardless of what is passed here, since only
            the RMS-based structure function supports quadrature subtraction.

    Returns:
        (s_periodic, synthetic_signal, coefficients)
    """
    synthetic_signal, coeffs = reconstruct_periodic_signal(time, data, periods)

    kwargs = dict(fluctuation_kwargs)
    kwargs["aggregation"] = "rms"
    kwargs["lag_times"] = np.asarray(lag_times)

    _, s_periodic, _, _ = fluctuation_func(time, synthetic_signal, **kwargs)

    return s_periodic, synthetic_signal, coeffs


def correct_structure_function_for_periodicity(
    s_measured: np.ndarray,
    s_periodic: np.ndarray,
) -> Tuple[np.ndarray, Dict]:
    """
    Removes the known periodic component's contribution from the measured
    RMS structure function via quadrature (variance) subtraction:

        S2_corrected(lag)^2 = max(0, S2_measured(lag)^2 - S2_periodic(lag)^2)

    This is only valid because variances of uncorrelated random
    contributions add linearly - see the module docstring for the
    assumptions this relies on (approximate phase-independence of the
    stochastic process relative to the periodic cycle).

    Args:
        s_measured: The real data's RMS structure function, S2(lag).
        s_periodic: The periodic-only model's RMS structure function,
            S2_periodic(lag), computed on the SAME lag grid via
            `compute_periodic_structure_function`.

    Returns:
        (s_corrected, diagnostics) where diagnostics contains:
            "fraction_variance_removed": per-lag fraction of measured
                variance attributed to the periodic component.
            "overshoot_fraction": fraction of lags where the periodic
                model's variance exceeded the measured variance (clipped to
                zero). A high value (e.g. > 0.3) suggests the supplied
                periods/amplitudes are not a good match for this data and
                the correction should not be trusted.
    """
    s_measured = np.asarray(s_measured, dtype=float)
    s_periodic = np.asarray(s_periodic, dtype=float)
    if s_measured.shape != s_periodic.shape:
        raise ValueError(
            f"s_measured (shape {s_measured.shape}) and s_periodic (shape "
            f"{s_periodic.shape}) must have identical shapes - make sure "
            "both were computed on the same lag grid."
        )

    var_measured = s_measured ** 2
    var_periodic = s_periodic ** 2
    var_corrected = np.clip(var_measured - var_periodic, 0.0, None)
    s_corrected = np.sqrt(var_corrected)

    with np.errstate(divide="ignore", invalid="ignore"):
        fraction_removed = np.where(
            var_measured > 0,
            (var_measured - var_corrected) / var_measured,
            0.0,
        )

    overshoot_mask = var_periodic > var_measured
    overshoot_fraction = float(np.mean(overshoot_mask)) if len(s_measured) else 0.0

    if overshoot_fraction > 0.3:
        warnings.warn(
            f"Periodicity correction: the periodic model's variance exceeded "
            f"the measured variance at {overshoot_fraction * 100:.0f}% of "
            "lags (clipped to zero there). This suggests the supplied "
            "periods and/or their fitted amplitudes are not well matched to "
            "this data (or aggregation != 'rms' was used upstream). Treat "
            "the corrected structure function with caution.",
            UserWarning,
        )

    diagnostics = {
        "fraction_variance_removed": fraction_removed,
        "overshoot_fraction": overshoot_fraction,
    }
    return s_corrected, diagnostics

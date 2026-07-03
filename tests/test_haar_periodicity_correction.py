"""
Tests for:
  1. Optional Haar periodicity correction (waterSpec.haar_periodicity +
     HaarAnalysis.run(correct_periodicity=..., periodic_periods=...)).
  2. The fix to Haar's standard-vs-segmented BIC model selection reporting
     (HaarAnalysis.run(max_breakpoints>=1) now surfaces `chosen_model`,
     `all_models`, `analysis_mode="auto"` etc., mirroring the Lomb-Scargle
     branch, and `format_haar_summary` renders it correctly).

Run with:  pytest tests/test_haar_periodicity_correction.py -v
"""

from unittest.mock import patch

import numpy as np
import pytest

from waterSpec.utils_sim import simulate_tk95
from waterSpec.haar_analysis import HaarAnalysis, format_haar_summary
from waterSpec.haar_periodicity import (
    PeriodCluster,
    consolidate_periods,
    list_period_candidates,
    reconstruct_periodic_signal,
    correct_structure_function_for_periodicity,
)


# ---------------------------------------------------------------------------
# Helpers / synthetic data generators
# ---------------------------------------------------------------------------

def power_law_psd(f, beta, amp=1.0):
    """P(f) = amp * f^(-beta)"""
    return amp * (f ** (-beta))


def broken_power_law_psd(f, beta1, beta2, f_break, amp=1.0):
    """Continuous broken power-law PSD: beta1 below f_break, beta2 above it."""
    psd = np.zeros_like(f)
    mask1 = f <= f_break
    mask2 = ~mask1
    psd[mask1] = amp * f[mask1] ** (-beta1)
    amp2 = amp * f_break ** (beta2 - beta1)
    psd[mask2] = amp2 * f[mask2] ** (-beta2)
    return psd


def make_seasonal_series(
    beta_true=1.4, n_points=4000, dt=1.0, seasonal_amp_ratio=4.0,
    period=365.25, phase=0.7, seed=42,
):
    """
    Synthetic daily series = power-law colored noise (known beta_true) +
    a strong deterministic annual sinusoid, mimicking a real temperature
    record. `seasonal_amp_ratio` controls how many multiples of the noise's
    own std the seasonal amplitude is (4.0 is comparable to real air
    temperature: ~2x-5x the day-to-day weather noise).
    """
    time, noise = simulate_tk95(power_law_psd, (beta_true,), n_points, dt, seed=seed)
    noise_std = np.std(noise)
    seasonal_amp = seasonal_amp_ratio * noise_std
    seasonal = seasonal_amp * np.sin(2 * np.pi * time / period + phase)
    data = noise + seasonal
    return time, data, noise, seasonal


# ---------------------------------------------------------------------------
# Feature 1: Periodicity correction
# ---------------------------------------------------------------------------

class TestPeriodicityCorrectionEndToEnd:

    def test_uncorrected_fit_is_badly_biased_by_seasonality(self):
        """Sanity check that the test scenario reproduces the reported problem."""
        time, data, _, _ = make_seasonal_series()
        haar_raw = HaarAnalysis(time, data, time_unit="days")
        res_raw = haar_raw.run(aggregation="rms", n_bootstraps=0)
        # A single power law should NOT describe this contaminated curve well.
        assert res_raw["r2"] < 0.5

    def test_correction_recovers_true_beta_and_improves_fit(self):
        beta_true = 1.4
        time, data, noise, _ = make_seasonal_series(beta_true=beta_true)

        res_raw = HaarAnalysis(time, data, time_unit="days").run(
            aggregation="rms", n_bootstraps=0
        )
        res_corrected = HaarAnalysis(time, data, time_unit="days").run(
            aggregation="rms", n_bootstraps=0,
            correct_periodicity=True, periodic_periods=[365.25],
        )
        # Oracle: what you'd get if the seasonal signal had never been added.
        res_oracle = HaarAnalysis(time, noise, time_unit="days").run(
            aggregation="rms", n_bootstraps=0
        )

        beta_raw, r2_raw = res_raw["beta"], res_raw["r2"]
        beta_corr, r2_corr = res_corrected["beta"], res_corrected["r2"]

        # The corrected estimate must be closer to the truth than the raw one...
        assert abs(beta_corr - beta_true) < abs(beta_raw - beta_true)
        # ...and close to the "no seasonality at all" oracle result.
        assert beta_corr == pytest.approx(res_oracle["beta"], abs=0.25)
        # ...and a substantially better power-law fit than the raw curve.
        assert r2_corr > r2_raw
        assert r2_corr > 0.6
        # ...and within a reasonable absolute tolerance of ground truth.
        assert beta_corr == pytest.approx(beta_true, abs=0.35)

        # Diagnostics should be present, sane, and informative.
        pc = res_corrected["periodicity_correction"]
        assert pc["periods_used"] == [365.25]
        assert 0.0 <= pc["overshoot_fraction"] <= 1.0
        assert np.nanmean(pc["fraction_variance_removed"]) > 0.05

    def test_correct_periodicity_requires_rms_aggregation(self):
        time, data, _, _ = make_seasonal_series()
        with pytest.raises(ValueError, match="aggregation='rms'"):
            HaarAnalysis(time, data).run(
                aggregation="mean", correct_periodicity=True, periodic_periods=[365.25]
            )

    def test_correct_periodicity_requires_explicit_periods(self):
        time, data, _, _ = make_seasonal_series()
        with pytest.raises(ValueError, match="periodic_periods"):
            HaarAnalysis(time, data).run(aggregation="rms", correct_periodicity=True)

    def test_default_behavior_is_completely_unaffected(self):
        """
        Backward-compatibility guard: calling .run() the old way must not
        gain any new dict keys or change any existing numeric result.
        """
        time, data, _, _ = make_seasonal_series()

        haar_a = HaarAnalysis(time, data)
        res_a = haar_a.run(n_bootstraps=0, seed=1)

        haar_b = HaarAnalysis(time, data)
        res_b = haar_b.run(
            n_bootstraps=0, seed=1,
            correct_periodicity=False,  # explicit opt-out, same as default
        )

        for key in ("analysis_mode", "chosen_model", "all_models", "periodicity_correction"):
            assert key not in res_a
            assert key not in res_b

        assert res_a["beta"] == res_b["beta"]
        assert res_a["r2"] == res_b["r2"]


class TestReconstructionAndCorrectionUnits:

    def test_reconstruct_periodic_signal_matches_injected_sinusoid(self):
        """The joint harmonic-regression reconstruction should closely recover
        a known, noise-free sinusoid at irregular timestamps."""
        rng = np.random.default_rng(0)
        time = np.sort(rng.uniform(0, 3650, 2000))  # irregular sampling
        true_amp, true_period, true_phase = 5.0, 365.25, 1.2
        signal = true_amp * np.sin(2 * np.pi * time / true_period + true_phase)

        reconstruction, _ = reconstruct_periodic_signal(time, signal, periods=[true_period])
        assert np.allclose(reconstruction, signal - np.mean(signal), atol=1e-6)

    def test_reconstruct_periodic_signal_rejects_bad_input(self):
        time = np.arange(100.0)
        data = np.sin(time)
        with pytest.raises(ValueError):
            reconstruct_periodic_signal(time, data, periods=[])
        with pytest.raises(ValueError):
            reconstruct_periodic_signal(time, data, periods=[-10])

    def test_quadrature_correction_shape_mismatch_raises(self):
        with pytest.raises(ValueError):
            correct_structure_function_for_periodicity(
                np.array([1.0, 2.0]), np.array([1.0])
            )

    def test_quadrature_correction_clips_negative_variance(self):
        s_measured = np.array([1.0, 2.0, 3.0])
        s_periodic = np.array([5.0, 1.0, 1.0])  # overshoots at index 0
        s_corrected, diag = correct_structure_function_for_periodicity(s_measured, s_periodic)
        assert s_corrected[0] == 0.0  # clipped, not negative/NaN
        assert diag["overshoot_fraction"] == pytest.approx(1.0 / 3.0)


# ---------------------------------------------------------------------------
# Feature 1b: user-controlled period selection (avoiding LS sideband duplicates)
# ---------------------------------------------------------------------------

class TestUserControlledPeriodSelection:

    def test_consolidate_periods_merges_near_duplicate_sidebands(self):
        """
        Reproduces the real-world problem: Lomb-Scargle often reports many
        "significant" periods clustered around one true annual cycle
        (spectral leakage sidebands). These must collapse to ONE candidate,
        not be treated as independent periods.
        """
        sideband_periods = [335, 345, 350, 355, 360, 365, 370, 375, 380, 390, 400]
        long_period = [4000.0]
        periods = sideband_periods + long_period
        strengths = list(range(len(sideband_periods))) + [1000.0]

        clusters = consolidate_periods(periods, strengths=strengths, tolerance=0.15)

        near_annual = [c for c in clusters if 300 < c.representative_period < 450]
        assert len(near_annual) == 1
        assert len(near_annual[0].member_periods) == len(sideband_periods)

        long_cluster = [c for c in clusters if c.representative_period > 1000]
        assert len(long_cluster) == 1
        assert long_cluster[0].representative_period == pytest.approx(4000.0)

    def test_consolidate_periods_representative_strongest(self):
        periods = [360.0, 365.0, 370.0]
        strengths = [1.0, 1.0, 9.0]  # 370 is by far the strongest
        clusters = consolidate_periods(periods, strengths=strengths, tolerance=0.2, representative="strongest")
        assert len(clusters) == 1
        assert clusters[0].representative_period == pytest.approx(370.0)

    def test_list_period_candidates_end_to_end_from_peak_dicts(self):
        """
        Simulates the exact workflow a user follows: take a (messy)
        significant_peaks list as produced by waterSpec's Lomb-Scargle peak
        detection, preview the clustered candidates, and CHOOSE which ones
        to keep before ever calling HaarAnalysis.run(correct_periodicity=True).
        """
        peaks = [
            {"frequency": 1 / 365.0, "power": 10.0},
            {"frequency": 1 / 370.0, "power": 8.0},   # near-duplicate of the above
            {"frequency": 1 / 360.0, "power": 7.0},   # near-duplicate of the above
            {"frequency": 1 / 180.0, "power": 3.0},   # genuine semiannual harmonic
            {"frequency": 1 / 40.0, "power": 1.0},    # user wants to EXCLUDE this one
        ]

        clusters = list_period_candidates(peaks, tolerance=0.1)
        assert isinstance(clusters[0], PeriodCluster)

        representative_periods = sorted(round(c.representative_period) for c in clusters)
        assert representative_periods == [40, 180, 365]  # 3 clusters, sidebands merged

        # User inspects the clusters and explicitly excludes the 40-day one,
        # keeping only physically meaningful (>100 day) cycles - full manual control.
        chosen_periods = [c.representative_period for c in clusters if c.representative_period > 100]
        assert len(chosen_periods) == 2
        assert all(p > 100 for p in chosen_periods)

        # These user-chosen periods (not the raw, unconsolidated peak list) are
        # what gets passed to HaarAnalysis.run(periodic_periods=chosen_periods).
        time, data, _, _ = make_seasonal_series(period=365.25)
        res = HaarAnalysis(time, data).run(
            aggregation="rms", n_bootstraps=0,
            correct_periodicity=True, periodic_periods=chosen_periods,
        )
        assert res["periodicity_correction"]["periods_used"] == chosen_periods

    def test_min_max_period_bounds_filter_candidates(self):
        peaks = [
            {"frequency": 1 / 365.0, "power": 10.0},
            {"frequency": 1 / 20.0, "power": 5.0},
            {"frequency": 1 / 9000.0, "power": 2.0},
        ]
        clusters = list_period_candidates(peaks, min_period=100, max_period=1000)
        periods_found = [round(c.representative_period) for c in clusters]
        assert periods_found == [365]

    def test_explicit_periodic_periods_bypasses_clustering_entirely(self):
        """The primary, fully-controlled path: hand-pick periods directly,
        with no clustering/consolidation involved at all."""
        time, data, _, _ = make_seasonal_series(period=365.25)
        res = HaarAnalysis(time, data).run(
            aggregation="rms", n_bootstraps=0,
            correct_periodicity=True, periodic_periods=[365.25, 182.625],
        )
        assert res["periodicity_correction"]["periods_used"] == [365.25, 182.625]


# ---------------------------------------------------------------------------
# Feature 2: Haar standard-vs-segmented model selection reporting fix
# ---------------------------------------------------------------------------

class TestModelSelectionLogic:
    """
    Unit tests of the SELECTION LOGIC using controlled/mocked
    `fit_segmented_haar` outputs (mirroring the style of
    tests/test_model_selector.py for the Lomb-Scargle branch), so these are
    deterministic and independent of any particular synthetic dataset.
    """

    def _make_flat_data(self, n=30):
        # Simple monotonic-ish structure function; exact values don't matter
        # for these tests since fit_segmented_haar is mocked.
        lags = np.logspace(0, 3, n)
        s1 = lags ** 0.5
        return lags, s1

    def test_standard_chosen_when_segmented_bic_is_worse(self):
        lags, s1 = self._make_flat_data()
        with patch(
            "waterSpec.haar_analysis.fit_segmented_haar",
            return_value={"bic": 1e6, "betas": [0.5, 0.6], "n_breakpoints": 1,
                          "breakpoints": [10.0], "betas_ci": [(0.4, 0.6), (0.5, 0.7)],
                          "Hs": [-0.25, -0.2], "Hs_ci": [(-0.3, -0.2), (-0.25, -0.15)],
                          "breakpoints_ci": [(8.0, 12.0)]},
        ):
            haar = HaarAnalysis(np.arange(len(lags), dtype=float), s1)
            haar.lags, haar.s1 = lags, s1  # bypass fluctuation calc, test selection only
            res = haar.run(max_breakpoints=1, n_bootstraps=0)

        assert res["analysis_mode"] == "auto"
        assert res["chosen_model"] == "standard"
        assert res["n_breakpoints"] == 0
        assert len(res["all_models"]) == 2
        assert "betas" not in res or res.get("segmented_results") is None

    def test_segmented_chosen_when_its_bic_is_better(self):
        lags, s1 = self._make_flat_data()
        fake_segmented = {
            "bic": -1e6, "betas": [0.2, 1.8], "n_breakpoints": 1,
            "breakpoints": [10.0], "betas_ci": [(0.1, 0.3), (1.6, 2.0)],
            "Hs": [-0.4, 0.4], "Hs_ci": [(-0.45, -0.35), (0.35, 0.45)],
            "breakpoints_ci": [(8.0, 12.0)],
        }
        with patch("waterSpec.haar_analysis.fit_segmented_haar", return_value=fake_segmented):
            haar = HaarAnalysis(np.arange(len(lags), dtype=float), s1)
            res = haar.run(max_breakpoints=1, n_bootstraps=0)

        assert res["chosen_model"] == "segmented_1bp"
        assert res["n_breakpoints"] == 1
        assert res["betas"] == fake_segmented["betas"]
        assert res["breakpoints"] == fake_segmented["breakpoints"]

    def test_chosen_model_always_matches_bic_argmin(self):
        """Self-consistency guard, independent of the underlying statistics:
        whatever wins BIC among all_models must be what's reported."""
        lags, s1 = self._make_flat_data()
        fake_segmented = {
            "bic": -5.0, "betas": [0.3, 0.9], "n_breakpoints": 1,
            "breakpoints": [10.0], "betas_ci": [(0.2, 0.4), (0.8, 1.0)],
            "Hs": [-0.35, -0.05], "Hs_ci": [(-0.4, -0.3), (-0.1, 0.0)],
            "breakpoints_ci": [(8.0, 12.0)],
        }
        with patch("waterSpec.haar_analysis.fit_segmented_haar", return_value=fake_segmented):
            haar = HaarAnalysis(np.arange(len(lags), dtype=float), s1)
            res = haar.run(max_breakpoints=1, n_bootstraps=0)

        best = min(res["all_models"], key=lambda m: m["bic"])
        expected = "standard" if best["n_breakpoints"] == 0 else f"segmented_{best['n_breakpoints']}bp"
        assert res["chosen_model"] == expected

    def test_failed_segmented_fit_recorded_not_selected(self):
        lags, s1 = self._make_flat_data()
        with patch(
            "waterSpec.haar_analysis.fit_segmented_haar",
            return_value={"failure_reason": "did not converge", "n_breakpoints": 1, "bic": np.inf},
        ):
            haar = HaarAnalysis(np.arange(len(lags), dtype=float), s1)
            res = haar.run(max_breakpoints=1, n_bootstraps=0)

        assert res["chosen_model"] == "standard"
        assert len(res["failed_model_reasons"]) == 1
        assert "did not converge" in res["failed_model_reasons"][0]


class TestModelSelectionEndToEndAndReporting:

    def test_genuine_breakpoint_is_detected_with_sane_parameters(self):
        """
        End-to-end (no mocking) on a strongly broken power-law series: the
        BIC gap here is large enough that segmented selection is robust
        regardless of the general over-selection tendency noted below.
        """
        n_points, dt = 3000, 1.0
        beta_low_freq, beta_high_freq = 2.2, 0.4  # low-freq = long lag; high-freq = short lag
        f_min, f_max = 1.0 / (n_points * dt), 0.5 / dt
        f_break = np.sqrt(f_min * f_max)

        time, series = simulate_tk95(
            broken_power_law_psd, (beta_low_freq, beta_high_freq, f_break), n_points, dt, seed=99
        )

        res = HaarAnalysis(time, series).run(max_breakpoints=1, n_bootstraps=20, seed=99)

        assert res["analysis_mode"] == "auto"
        assert res["chosen_model"] == "segmented_1bp"
        assert res["n_breakpoints"] == 1
        # betas[0] = shortest lag = high-frequency regime = beta_high_freq
        # betas[1] = longest lag  = low-frequency regime  = beta_low_freq
        assert res["betas"][0] == pytest.approx(beta_high_freq, abs=0.5)
        assert res["betas"][1] == pytest.approx(beta_low_freq, abs=0.8)

        summary = format_haar_summary(res, param_name="Synthetic Break", time_unit="days")
        assert "Chosen Model: Segmented 1bp" in summary
        assert "Short-Term (High-Frequency) Fit" in summary
        assert "Long-Term (Low-Frequency) Fit" in summary
        assert "--- Breakpoint 1 @" in summary

    def test_standard_model_summary_renders_correctly(self):
        lags, s1 = np.logspace(0, 3, 30), None
        s1 = lags ** (0.3)  # H=0.3 -> beta=1.6, single clean power law, no noise
        haar = HaarAnalysis(np.arange(len(lags), dtype=float), s1)
        with patch(
            "waterSpec.haar_analysis.fit_segmented_haar",
            return_value={"bic": np.inf, "n_breakpoints": 1, "failure_reason": "not enough points"},
        ):
            res = haar.run(max_breakpoints=1, n_bootstraps=0)

        assert res["chosen_model"] == "standard"
        summary = format_haar_summary(res, param_name="Clean PowerLaw", time_unit="days")
        assert "Chosen Model: Standard" in summary
        assert "Standard Haar Analysis for: Clean PowerLaw" in summary

    def test_legacy_default_path_has_no_auto_fields(self):
        """max_breakpoints=0 (the default) must remain completely unaffected."""
        time, series = simulate_tk95(power_law_psd, (1.3,), 1000, 1.0, seed=5)
        res = HaarAnalysis(time, series).run(n_bootstraps=0)
        for key in ("analysis_mode", "chosen_model", "all_models", "n_breakpoints"):
            assert key not in res

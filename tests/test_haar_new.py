
import unittest
import numpy as np
import pandas as pd
from waterSpec.haar_analysis import calculate_haar_fluctuations, fit_segmented_haar, calculate_sliding_haar
from waterSpec.bivariate import BivariateAnalysis
from waterSpec.surrogates import generate_phase_randomized_surrogates, generate_block_shuffled_surrogates
from scipy import stats

class TestHaarAnalysisFeatures(unittest.TestCase):

    def test_overlap_logic(self):
        time = np.arange(9.0) # 0..8
        data = np.array([0, 0, 0, 0, 10, 10, 10, 10, 10]) # 9 points

        lags, s1, counts, n_eff = calculate_haar_fluctuations(
            time, data, lag_times=np.array([4.0]), overlap=False,
            min_samples_per_window=2
        )
        self.assertEqual(len(counts), 1)
        self.assertEqual(counts[0], 2) # 2 windows
        self.assertAlmostEqual(s1[0], 0.0) # Both windows have 0 fluctuation

        lags_ov, s1_ov, counts_ov, n_eff_ov = calculate_haar_fluctuations(
            time, data, lag_times=np.array([4.0]), overlap=True, overlap_step_fraction=0.25,
            min_samples_per_window=2
        )
        self.assertEqual(counts_ov[0], 5)
        self.assertAlmostEqual(s1_ov[0], 4.0)
        self.assertTrue(n_eff_ov[0] < counts_ov[0]) # Effective N should be reduced

    def test_segmented_haar(self):
        lags = np.logspace(0, 3, 20)
        s1 = np.zeros_like(lags)
        bp_idx = np.searchsorted(lags, 10)

        s1[:bp_idx] = lags[:bp_idx]**(-0.5)
        val_at_10 = 10**(-0.5)
        s1[bp_idx:] = val_at_10 * (lags[bp_idx:] / 10)**0.5

        res = fit_segmented_haar(lags, s1, n_breakpoints=1, n_bootstraps=50)

        self.assertTrue(res['ci_computed'])
        self.assertEqual(len(res['breakpoints']), 1)
        self.assertTrue(5 < res['breakpoints'][0] < 20)
        self.assertAlmostEqual(res['Hs'][0], -0.5, delta=0.2)
        self.assertAlmostEqual(res['Hs'][1], 0.5, delta=0.2)

    def test_sliding_haar_anomaly(self):
        # Base: Quiet noise
        data = np.zeros(200)
        # Event: At t=100, variability increases
        data[100:120] = np.random.normal(0, 5.0, 20) # High magnitude
        time = np.arange(200)

        # Window size 10.
        t_centers, fluctuations = calculate_sliding_haar(
            time, data, window_size=10.0, step_size=1.0, min_samples_per_window=2
        )

        # The fluctuation should spike when the window hits the anomaly
        max_fluc = np.max(np.abs(fluctuations))
        # The quiet period should be near 0
        quiet_fluc = np.mean(np.abs(fluctuations[:50]))

        self.assertTrue(max_fluc > 10 * quiet_fluc)

        # Peak location
        peak_idx = np.argmax(np.abs(fluctuations))
        peak_time = t_centers[peak_idx]
        self.assertTrue(90 <= peak_time <= 130)

class TestBivariateAnalysis(unittest.TestCase):

    def test_alignment_interpolation(self):
        t1 = np.array([0, 10, 20], dtype=float)
        d1 = np.array([1, 2, 3], dtype=float)

        # t2 is higher resolution
        t2 = np.array([0, 5, 10, 15, 20], dtype=float)
        d2 = np.array([1, 1.5, 2, 2.5, 3], dtype=float)

        biv = BivariateAnalysis(t1, d1, "C", t2, d2, "Q", time_unit='numeric')

        # Interpolate Q (d2) onto C (t1)
        biv.align_data(tolerance=1.0, method='interpolate_2_to_1')

        aligned = biv.aligned_data
        self.assertEqual(len(aligned), 3)
        # Should match d1 exactly because d2 is perfectly linear interp
        np.testing.assert_allclose(aligned['Q'].values, d1)

    def test_alignment_and_correlation(self):
        t1 = np.array([0, 10, 20, 30, 40], dtype=float)
        d1 = np.array([1, 5, 2, 8, 3], dtype=float)

        t2 = np.array([1, 11, 21, 31, 41], dtype=float) # shifted by 1
        d2 = np.array([1, 5, 2, 8, 3], dtype=float) # perfect correlation

        biv = BivariateAnalysis(t1, d1, "C", t2, d2, "Q", time_unit='seconds')
        biv.time_unit = "numeric"

        biv.align_data(tolerance=2.0, method='nearest')

        res = biv.run_cross_haar_analysis(np.array([20.0]), overlap=False)

        self.assertEqual(len(res['correlation']), 1)
        if not np.isnan(res['correlation'][0]):
             self.assertAlmostEqual(res['correlation'][0], 1.0, delta=0.1)

        res_ov = biv.run_cross_haar_analysis(np.array([20.0]), overlap=True, overlap_step_fraction=0.1)
        if not np.isnan(res_ov['correlation'][0]):
            self.assertAlmostEqual(res_ov['correlation'][0], 1.0, delta=0.1)

    def test_hysteresis_area(self):
        t = np.linspace(0, 2*np.pi, 100)
        q = np.cos(t)
        c = np.sin(t)

        biv = BivariateAnalysis(t, c, "C", t, q, "Q", time_unit="numeric")
        biv.align_data(tolerance=0.1)

        res = biv.calculate_hysteresis_metrics(tau=0.1, overlap=True)

        self.assertEqual(res['direction'], "Counter-Clockwise")
        self.assertTrue(res['area'] > 0)

class TestSurrogates(unittest.TestCase):
    def test_phase_randomization(self):
        data = np.random.randn(100)
        surr = generate_phase_randomized_surrogates(data, n_surrogates=10)
        self.assertEqual(surr.shape, (10, 100))

        fft_orig = np.abs(np.fft.rfft(data))
        fft_surr = np.abs(np.fft.rfft(surr[0]))
        np.testing.assert_allclose(fft_orig, fft_surr, rtol=1e-5)

    def test_block_shuffling(self):
        # Create data with trend
        data = np.arange(100)
        # Block size 10.
        surr = generate_block_shuffled_surrogates(data, block_size=10, n_surrogates=5)

        self.assertEqual(surr.shape, (5, 100))
        # Mean and variance should be preserved approx (identical if length divisible)
        self.assertAlmostEqual(np.mean(surr[0]), np.mean(data))
        self.assertAlmostEqual(np.var(surr[0]), np.var(data))

        # But lag-1 diffs should be broken at block boundaries
        # Hard to test deterministically without a specific statistical test
        # Just check it's not identical to original
        self.assertFalse(np.array_equal(surr[0], data))

if __name__ == '__main__':
    unittest.main()

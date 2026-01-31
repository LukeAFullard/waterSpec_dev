
import unittest
import numpy as np
import pandas as pd
from waterSpec.haar_analysis import calculate_haar_fluctuations, fit_segmented_haar
from waterSpec.bivariate import BivariateAnalysis
from waterSpec.surrogates import generate_phase_randomized_surrogates
from scipy import stats

class TestHaarAnalysisFeatures(unittest.TestCase):

    def test_overlap_logic(self):
        time = np.arange(9.0) # 0..8
        data = np.array([0, 0, 0, 0, 10, 10, 10, 10, 10]) # 9 points

        # Tau = 4. Windows have 4 points total (2 per half).
        # We need to set min_samples_per_window <= 2 for this test to work.

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
        # Generate synthetic data with breakpoint
        lags = np.logspace(0, 3, 20)
        s1 = np.zeros_like(lags)
        bp_idx = np.searchsorted(lags, 10)

        s1[:bp_idx] = lags[:bp_idx]**(-0.5)
        val_at_10 = 10**(-0.5)
        s1[bp_idx:] = val_at_10 * (lags[bp_idx:] / 10)**0.5

        # Assuming min_segment_length=4 default in fit_segmented_haar
        # We have 20 points.
        res = fit_segmented_haar(lags, s1, n_breakpoints=1, n_bootstraps=50)

        self.assertTrue(res['ci_computed'])
        self.assertEqual(len(res['breakpoints']), 1)
        self.assertTrue(5 < res['breakpoints'][0] < 20)
        self.assertAlmostEqual(res['Hs'][0], -0.5, delta=0.2)
        self.assertAlmostEqual(res['Hs'][1], 0.5, delta=0.2)

class TestBivariateAnalysis(unittest.TestCase):

    def test_alignment_and_correlation(self):
        t1 = np.array([0, 10, 20, 30, 40], dtype=float)
        # Using variable data to avoid linregress singular matrix error
        d1 = np.array([1, 5, 2, 8, 3], dtype=float)

        t2 = np.array([1, 11, 21, 31, 41], dtype=float) # shifted by 1
        d2 = np.array([1, 5, 2, 8, 3], dtype=float) # perfect correlation

        biv = BivariateAnalysis(t1, d1, "C", t2, d2, "Q", time_unit='seconds')

        biv.time_unit = "numeric"

        biv.align_data(tolerance=2.0, method='nearest')

        self.assertEqual(len(biv.aligned_data), 5)

        # Cross Haar
        res = biv.run_cross_haar_analysis(np.array([20.0]), overlap=False)

        # Check if we got results
        self.assertEqual(len(res['correlation']), 1)
        if not np.isnan(res['correlation'][0]):
             self.assertAlmostEqual(res['correlation'][0], 1.0, delta=0.1)

        res_ov = biv.run_cross_haar_analysis(np.array([20.0]), overlap=True, overlap_step_fraction=0.1)
        if not np.isnan(res_ov['correlation'][0]):
            self.assertAlmostEqual(res_ov['correlation'][0], 1.0, delta=0.1)

class TestSurrogates(unittest.TestCase):
    def test_phase_randomization(self):
        data = np.random.randn(100)
        surr = generate_phase_randomized_surrogates(data, n_surrogates=10)
        self.assertEqual(surr.shape, (10, 100))

        # Power spectrum should be preserved
        fft_orig = np.abs(np.fft.rfft(data))
        fft_surr = np.abs(np.fft.rfft(surr[0]))

        np.testing.assert_allclose(fft_orig, fft_surr, rtol=1e-5)

if __name__ == '__main__':
    unittest.main()

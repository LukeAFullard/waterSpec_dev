import numpy as np
import unittest
from waterSpec.bivariate import BivariateAnalysis


class TestBivariate(unittest.TestCase):
    def generate_correlated_series(self, n=1000):
        np.random.seed(42)
        time = np.arange(n)
        # Common signal
        signal = np.sin(2 * np.pi * time / 50)
        # Var 1
        data1 = signal + 0.5 * np.random.randn(n)
        # Var 2 (correlated with Var 1)
        data2 = 0.8 * signal + 0.5 * np.random.randn(n)
        return time, data1, time, data2

    def test_cross_haar_correlation(self):
        t1, d1, t2, d2 = self.generate_correlated_series()
        biv = BivariateAnalysis(t1, d1, "V1", t2, d2, "V2")
        biv.align_data(tolerance=1)
        lags = np.array([10, 20, 50])
        res = biv.run_cross_haar_analysis(lags)
        corrs = np.array(res["correlation"])
        self.assertTrue(np.all(corrs > 0.5))

    def test_lagged_cross_haar(self):
        t1, d1, t2, d2 = self.generate_correlated_series()
        biv = BivariateAnalysis(t1, d1, "V1", t2, d2, "V2")
        biv.align_data(tolerance=1)
        lag_offsets = np.array([-5, 0, 5])
        res = biv.run_lagged_cross_haar(tau=20, lag_offsets=lag_offsets)
        self.assertEqual(len(res["correlation"]), 3)
        self.assertTrue(not np.any(np.isnan(res["correlation"])))

    def test_hysteresis_metrics(self):
        t = np.linspace(0, 2 * np.pi, 100)
        d1 = np.sin(t)
        d2 = np.cos(t)
        biv = BivariateAnalysis(t, d1, "V1", t, d2, "V2")
        biv.align_data(tolerance=0.1)
        hyst = biv.calculate_hysteresis_metrics(tau=0.1)
        self.assertFalse(np.isnan(hyst["area"]))
        self.assertIn(hyst["direction"], ["Clockwise", "Counter-Clockwise"])


if __name__ == "__main__":
    unittest.main()

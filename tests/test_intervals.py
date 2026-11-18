import unittest
import numpy as np
from forecasting.data import generate_series
from forecasting.intervals import calibrate, evaluate_intervals
from forecasting.model import fit_model, predict


class IntervalTests(unittest.TestCase):
    def test_finite_sample_quantile_and_inclusive_endpoints(self):
        calibration = calibrate([1, 2, 3, 4], [0, 0, 0, 0], coverage=.6)
        self.assertEqual(calibration['order_statistic'], 3)
        self.assertEqual(calibration['radius'], 3)
        result = evaluate_intervals([3, -3, 4], [0, 0, 0], calibration)
        self.assertAlmostEqual(result['empirical_coverage'], 2 / 3)
        self.assertEqual(result['mean_width'], 6)
        self.assertAlmostEqual(result['coverage_gap'], 2 / 3 - .6)

    def test_bad_and_insufficient_inputs(self):
        for level in (0, 1, float('nan'), True, .9):
            with self.subTest(level=level), self.assertRaises(ValueError):
                calibrate([1, 2], [1, 2], level)
        for actual, predictions in (([], []), ([1], [1, 2]), ([float('nan')], [1])):
            with self.subTest(actual=actual), self.assertRaises(ValueError):
                calibrate(actual, predictions, .5)
        with self.assertRaises(ValueError):
            evaluate_intervals([1], [1], {'radius': -1, 'target_coverage': .9})
        with self.assertRaises(ValueError):
            evaluate_intervals([1], [1], {'radius': 1e308, 'target_coverage': .9})
        with self.assertRaises(ValueError):
            evaluate_intervals([1e308], [-1e308], {'radius': 1, 'target_coverage': .9})

    def test_calibration_does_not_depend_on_test_targets(self):
        series = generate_series(600, seed=42)
        model = fit_model(series, 400)
        def interval(values):
            predictions = [predict(model, values[:i], i) for i in range(400, 500)]
            return calibrate(values[400:500], predictions)
        baseline = interval(series)
        changed = series.copy()
        changed[500:] += 1000000
        self.assertEqual(baseline, interval(changed))
        np.testing.assert_array_equal(series[:500], changed[:500])


if __name__ == '__main__':
    unittest.main()

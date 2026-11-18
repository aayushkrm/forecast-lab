import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
from fastapi.testclient import TestClient

from forecasting.data import generate_series
from forecasting.model import MIN_HISTORY, build_features, fit_model, predict
from forecasting.train import evaluate


class CausalForecastTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.series = generate_series(400, seed=42)
        cls.model = fit_model(cls.series, 300)

    def test_constant_training_series_fails_before_artifact_creation(self):
        with self.assertRaisesRegex(ValueError, "nonzero standard deviation"):
            fit_model(np.full(400, 10.0), 300)

    def test_training_ignores_values_at_and_after_train_boundary(self):
        changed = self.series.copy()
        changed[300:] += 1_000_000
        self.assertEqual(self.model, fit_model(changed, 300))

    def test_forecast_ignores_target_and_future_values(self):
        index = 320
        baseline = predict(self.model, self.series[:index], index)
        changed = self.series.copy()
        changed[index:] += 1_000_000
        self.assertEqual(baseline, predict(self.model, changed[:index], index))
        self.assertEqual(evaluate(self.series, 300, 320, self.model), evaluate(changed, 300, 320, self.model))

    def test_lags_and_rolling_window_use_only_history(self):
        features = build_features(np.arange(168), 168)
        np.testing.assert_allclose(features[:6], [167, 166, 144, 120, 0, 155.5])

    def test_naive_evaluation_uses_previous_observation(self):
        result = evaluate(self.series, 300, 320, self.model)
        expected = np.abs(self.series[300:320] - self.series[299:319]).mean()
        self.assertAlmostEqual(result['naive_last_value']['mae'], expected)


class ApiTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from forecasting.api import app
        cls.app = app
        cls.series = generate_series(400, seed=42)
        cls.model = fit_model(cls.series, 300)
        cls.tmp = tempfile.TemporaryDirectory()
        cls.path = Path(cls.tmp.name) / 'model.json'
        cls.path.write_text(json.dumps(cls.model))

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def setUp(self):
        self.env = patch.dict('os.environ', {'FORECAST_MODEL_PATH': str(self.path)})
        self.env.start()
        self.client = TestClient(self.app)
        self.client.__enter__()

    def tearDown(self):
        self.client.__exit__(None, None, None)
        self.env.stop()

    def test_health_and_forecast_match_model(self):
        self.assertEqual(self.client.get('/health').json(), {'status': 'ok', 'model': 'ridge', 'model_version': 1})
        history = self.series[:168].tolist()
        response = self.client.post('/forecast', json={'history': history, 'next_index': 168})
        self.assertEqual(response.status_code, 200)
        result = response.json()
        self.assertEqual(result['next_index'], 168)
        self.assertEqual(result['model_version'], 1)
        self.assertAlmostEqual(result['prediction'], predict(self.model, history, 168))

    def test_rejects_short_and_inconsistent_history(self):
        self.assertEqual(self.client.post('/forecast', json={'history': [1] * 167, 'next_index': 168}).status_code, 422)
        self.assertEqual(self.client.post('/forecast', json={'history': [1] * 169, 'next_index': 168}).status_code, 422)

    def test_monitor_alerts_for_large_mean_shift(self):
        baseline = self.model['training_mean']
        std = self.model['training_std']
        stable = self.client.post('/monitor', json={'recent_values': [baseline] * 24})
        shifted = self.client.post('/monitor', json={'recent_values': [baseline + 3 * std] * 24})
        self.assertEqual(stable.status_code, 200)
        self.assertFalse(stable.json()['alert'])
        self.assertTrue(shifted.json()['alert'])
        self.assertAlmostEqual(shifted.json()['mean_shift_in_training_std'], 3)
        self.assertEqual(self.client.post('/monitor', json={'recent_values': [baseline] * 23}).status_code, 422)

    def test_rejects_nonfinite_history(self):
        # Raw JSON is used because HTTPX rejects non-standard NaN on serialization.
        body = '{"history": [' + ','.join(['1'] * 167 + ['NaN']) + '], "next_index": 168}'
        response = self.client.post('/forecast', content=body, headers={'Content-Type': 'application/json'})
        self.assertEqual(response.status_code, 422)


if __name__ == '__main__':
    unittest.main()

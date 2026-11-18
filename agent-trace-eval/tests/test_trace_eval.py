import json
import tempfile
import unittest
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import trace_eval as e


class TraceTests(unittest.TestCase):
    def test_order_and_arguments(self):
        c = {'id': 'x', 'task': 'task', 'calls': [{'tool': 'a', 'args': {'ok': True}}, {'tool': 'b', 'args': {}}], 'rules': {'required': ['a'], 'forbidden': ['c'], 'ordered': [['a', 'b']], 'arguments': [{'tool': 'a', 'key': 'ok', 'equals': True}]}}
        e.validate(c)
        self.assertTrue(e.grade(c)['passed'])
        c['calls'].reverse()
        self.assertFalse(e.grade(c)['passed'])

    def test_every_call_and_types(self):
        c = {'id': 'x', 'task': 'task', 'calls': [{'tool': 'a', 'args': {'ok': True}}, {'tool': 'a', 'args': {'ok': 1}}], 'rules': {'arguments': [{'tool': 'a', 'key': 'ok', 'equals': True}]}}
        self.assertFalse(e.grade(c)['passed'])

    def test_regression_and_changed_rules(self):
        old = [{'id': 'x', 'task': 'task', 'calls': [{'tool': 'a', 'args': {}}], 'rules': {'required': ['a']}}]
        new = [{'id': 'x', 'task': 'task', 'calls': [], 'rules': {'required': ['a']}}]
        self.assertEqual(e.compare(old, new)['regressions'], ['x'])
        new[0]['rules'] = {}
        with self.assertRaises(ValueError): e.compare(old, new)

    def test_task_binding(self):
        a = [{'id': 'x', 'task': 'first', 'calls': [], 'rules': {}}]
        b = [{'id': 'x', 'task': 'different', 'calls': [], 'rules': {}}]
        with self.assertRaises(ValueError): e.compare(a, b)

    def test_nonfinite_and_duplicate_keys_at_all_depths(self):
        samples = [
            '{"id":"x","task":"t","calls":[{"tool":"a","args":{"x":1e999}}],"rules":{}}',
            '{"id":"x","task":"t","calls":[],"rules":{"arguments":[{"tool":"a","key":"x","equals":-1e999}]}}',
            '{"id":"x","task":"t","calls":[{"tool":"a","args":{"x":1,"x":2}}],"rules":{}}',
            '{"id":"x","task":"t","calls":[],"rules":{"required":[],"required":[]}}',
        ]
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / 'run.jsonl'
            for sample in samples:
                p.write_text(sample + '\n')
                with self.subTest(sample=sample), self.assertRaises(ValueError): e.load_cases(p)

    def test_invalid_inputs(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / 'run.jsonl'
            p.write_text((json.dumps({'id': 'x', 'task': 'task', 'calls': [], 'rules': {}}) + '\n') * 2)
            with self.assertRaises(ValueError): e.load_cases(p)
            p.write_text('{"id":"x","task":"task","calls":[],"rules":{"unknown":[]}}\n')
            with self.assertRaises(ValueError): e.load_cases(p)


if __name__ == '__main__': unittest.main()

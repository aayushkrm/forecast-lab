"""Deterministic checks for recorded agent tool-call trajectories."""
import argparse
import json
import math
import sys
from pathlib import Path


def validate(case):
    if not isinstance(case, dict) or not isinstance(case.get('id'), str) or not case['id']:
        raise ValueError('id must be a nonempty string')
    if not isinstance(case.get('task'), str) or not case['task'].strip():
        raise ValueError('task must be a nonempty description')
    calls, rules = case.get('calls'), case.get('rules')
    if not isinstance(calls, list) or not isinstance(rules, dict):
        raise ValueError('calls must be a list and rules must be an object')
    for call in calls:
        if not isinstance(call, dict) or not isinstance(call.get('tool'), str) or not call['tool'] or not isinstance(call.get('args'), dict):
            raise ValueError('each call needs a nonempty tool and args object')
    if set(rules) - {'required', 'forbidden', 'ordered', 'arguments'}:
        raise ValueError('unknown rule field')
    for name in ('required', 'forbidden'):
        field = rules.get(name, [])
        if not isinstance(field, list) or any(not isinstance(v, str) or not v for v in field):
            raise ValueError(f'{name} must be a list of nonempty tool names')
    ordered = rules.get('ordered', [])
    if not isinstance(ordered, list) or any(not isinstance(pair, list) or len(pair) != 2 or any(not isinstance(v, str) or not v for v in pair) for pair in ordered):
        raise ValueError('ordered must be a list of [before, after] pairs')
    arguments = rules.get('arguments', [])
    if not isinstance(arguments, list):
        raise ValueError('arguments must be a list')
    for rule in arguments:
        if not isinstance(rule, dict) or set(rule) != {'tool', 'key', 'equals'} or not all(isinstance(rule[k], str) and rule[k] for k in ('tool', 'key')):
            raise ValueError('argument rules need tool, key and equals')


def unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f'duplicate JSON key: {key}')
        result[key] = value
    return result


def finite_float(value):
    number = float(value)
    if not math.isfinite(number):
        raise ValueError('nonfinite JSON number')
    return number


def load_cases(path):
    cases, seen = [], set()
    with Path(path).open(encoding='utf-8') as source:
        for line_number, line in enumerate(source, 1):
            if not line.strip():
                continue
            try:
                case = json.loads(line, object_pairs_hook=unique_object, parse_float=finite_float, parse_constant=lambda x: (_ for _ in ()).throw(ValueError('invalid JSON constant')))
                validate(case)
                if case['id'] in seen:
                    raise ValueError('duplicate case id')
                seen.add(case['id'])
                cases.append(case)
            except (ValueError, TypeError, KeyError) as exc:
                raise ValueError(f'{path}:{line_number}: {exc}') from exc
    if not cases:
        raise ValueError(f'{path}: no cases')
    return cases


def same_json(a, b):
    return json.dumps(a, sort_keys=True, separators=(',', ':')) == json.dumps(b, sort_keys=True, separators=(',', ':'))


def grade(case):
    calls, rules = case['calls'], case['rules']
    tools = [c['tool'] for c in calls]
    failures = []
    for tool in rules.get('required', []):
        if tool not in tools: failures.append(f'required tool missing: {tool}')
    for tool in rules.get('forbidden', []):
        if tool in tools: failures.append(f'forbidden tool called: {tool}')
    for before, after in rules.get('ordered', []):
        if not any(a['tool'] == before and b['tool'] == after for i, a in enumerate(calls) for b in calls[i + 1:]):
            failures.append(f'order not satisfied: {before} before {after}')
    for rule in rules.get('arguments', []):
        matched = [c for c in calls if c['tool'] == rule['tool']]
        if not matched or any(rule['key'] not in c['args'] or not same_json(c['args'][rule['key']], rule['equals']) for c in matched):
            failures.append(f"argument constraint failed: {rule['tool']}.{rule['key']}")
    return {'id': case['id'], 'passed': not failures, 'failures': failures}


def evaluate(cases):
    results = [grade(case) for case in cases]
    passed = sum(r['passed'] for r in results)
    return {'total': len(results), 'passed': passed, 'pass_rate': passed / len(results), 'cases': results}


def compare(old, new):
    old_map, new_map = {c['id']: c for c in old}, {c['id']: c for c in new}
    if old_map.keys() != new_map.keys(): raise ValueError('runs must contain the same case ids')
    for key in old_map:
        if old_map[key]['task'] != new_map[key]['task']: raise ValueError(f'task differs for case {key}')
        if not same_json(old_map[key]['rules'], new_map[key]['rules']): raise ValueError(f'rules differ for case {key}')
    old_eval, new_eval = evaluate(old), evaluate(new)
    old_results = {r['id']: r for r in old_eval['cases']}
    new_results = {r['id']: r for r in new_eval['cases']}
    return {'old': old_eval, 'new': new_eval,
            'regressions': [k for k in old_map if old_results[k]['passed'] and not new_results[k]['passed']],
            'improvements': [k for k in old_map if not old_results[k]['passed'] and new_results[k]['passed']]}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    subs = parser.add_subparsers(dest='command', required=True)
    check = subs.add_parser('eval'); check.add_argument('run')
    diff = subs.add_parser('compare'); diff.add_argument('old'); diff.add_argument('new')
    args = parser.parse_args(argv)
    try:
        output = evaluate(load_cases(args.run)) if args.command == 'eval' else compare(load_cases(args.old), load_cases(args.new))
        print(json.dumps(output, indent=2, allow_nan=False))
        return 1 if args.command == 'compare' and output['regressions'] else 0
    except (OSError, ValueError, RecursionError) as exc:
        print(f'error: {exc}', file=sys.stderr)
        return 2


if __name__ == '__main__': sys.exit(main())

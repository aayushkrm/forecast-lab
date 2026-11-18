# Agent trace evaluator

Grade recorded tool-call sequences against explicit rules. This tool reads JSONL; it does not execute an agent, evaluate chemistry, or validate tool results. The chemistry-flavored examples are invented for this project; they are not real traces or evidence of scientific validity.

Requires Python 3.10 or newer. No third-party packages are needed.

```sh
python trace_eval.py eval examples/before.jsonl
python trace_eval.py compare examples/before.jsonl examples/after.jsonl
python -m unittest discover -s tests -v
```

Each nonempty line is an object with a unique `id`, a nonempty immutable `task` description, a `calls` array, and a `rules` object. Each call contains a tool name and an `args` object. Rules are optional arrays:

- `required`: each named tool must appear at least once.
- `forbidden`: no named tool may appear.
- `ordered`: each `[before, after]` pair requires at least one occurrence of the first tool before an occurrence of the second.
- `arguments`: each `{ "tool": ..., "key": ..., "equals": ... }` rule requires at least one call to that tool and requires every such call to have the exact, type-matching top-level argument value.

A case passes when every rule passes. `eval` prints each case's failures and aggregate pass rate; its exit status is 0 even if cases fail. `compare` requires identical case IDs, task descriptions, and rules in both files, and reports pass-to-fail regressions and fail-to-pass improvements. It exits 1 on any regression and 0 otherwise. Either command exits 2 for invalid input, including duplicate JSON keys and nonfinite numbers at any depth. Comparisons track case-level pass/fail, so a changed failure in an already-failing case is visible in the output but is not counted as a regression.

This is a structural check, not proof of task success. The checks do not interpret argument meaning, validate scientific claims, assess results, or detect problems outside the supplied rules. Define rules independently of the outputs and inspect failures in context.

# octave-rl

`octave-rl` is a seeded GNU Octave coding curriculum for native
`verifiers.v1`. It generates hidden NumPy-derived cases, executes candidate
functions in an isolated Prime Sandbox, and awards fractional correctness.

## Reproducibility

- Interpreter image: `ghcr.io/gnu-octave/octave:10.2.0`
- Observed interpreter: GNU Octave 10.2.0
- Verifiers: `>=0.2.1,<0.3`
- NumPy: `>=2.0,<3`
- Each task is determined by `(level, seed, task index)`.

The package exports both `OctaveTaskset` and the brief's
`load_environment(level=1, num_tasks=500, max_turns=2,
require_vectorized=False, seed=0, **kwargs)` entry point. The factory returns
the native `verifiers.v1` taskset rather than the alternative legacy v0
`vf.Environment` form; Prime supports both contracts, but this package uses
only v1. `max_turns` maps to the persistent user's attempt budget.
The evaluation or training harness should set its turn cap to the same or a
larger value.

## Families and curriculum

Ten families cover reductions, logical indexing, reshape/permutation,
broadcast arithmetic, sliding windows, linear solves, recurrences, struct/cell
wrangling, string parsing, and signal identities.

- Level 1: diagnostic scalar/vector and simple matrix functions.
- Level 2: the working set, with more constraints and edge cases.
- Level 3: stretch tasks and tighter tolerances; optionally require no loops.

`level`, `num_tasks`, `seed`, and `require_vectorized` are typed Taskset config
fields. Every task contains six precomputed hidden cases and an internal
reference function.

## Reward and multi-turn policy

`case_fraction` is the fraction of hidden cases passed. A valid structured
Octave run adds a small `0.1` reward. Formatting and vectorization are observed
as metrics rather than hard gates.

The optional user simulator runs the submitted function after each attempt:

| Attempt | Feedback | Correctness multiplier |
| --- | --- | ---: |
| 1 | none | 1.00 |
| 2 | hidden pass count and Octave diagnostic | 0.85 |
| 3 | the same diagnostic plus one concise guide hint | 0.60 |

The guide defaults to `Qwen/Qwen3.5-35B-A3B` through Prime Inference. It sees
the public task prompt, candidate source, and first diagnostic only. It is
asked for one hint and is not given hidden inputs, expected values, or the
reference implementation. Credentials are read at runtime from
`PRIME_API_KEY` or the authenticated local Prime config and are never stored
in traces or package configuration.

## Commands

```bash
uv run validate octave-rl --taskset.level 1 --taskset.num-tasks 10 \
  --taskset.seed 42 --runtime.type prime --only-gold true

uv run eval @ configs/eval/octave-qwen-4b-two-turn.toml
uv run eval @ configs/eval/octave-qwen-4b-guided-three-turn.toml
uv run python scripts/validate_reference_pool.py
```

The configs put `max_turns` at the environment root. It is not an inference
sampling argument.

## Sandbox behavior verified

On the pinned image:

- a passing harness exits 0;
- a deliberately failing harness exits nonzero when the generated script
  explicitly calls `exit(passed < total)`;
- `assert` failures are catchable;
- diagnostics require merging stderr with stdout (`2>&1`);
- uploading `.m` files avoids nested shell-quoting problems;
- warmed Octave script startup was roughly 0.3 seconds.

An uncaught assertion did exit 1 in the tested image, but the environment does
not rely on that version-sensitive behavior. It parses the machine-readable
`RESULT passed=N total=M` line and uses explicit harness exit status.
